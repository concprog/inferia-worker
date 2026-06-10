// Command worker is the inferia-worker entry point. It:
//  1. Loads config from env vars.
//  2. Opens/creates the persisted token store; bootstraps if no token yet.
//  3. Starts the Fiber HTTP server (auth → proxy → healthz).
//  4. Starts the control-channel WS client (heartbeat + command dispatch).
//  5. Blocks on SIGINT/SIGTERM, then shuts down.
package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/gofiber/fiber/v2"

	"github.com/inferia/inferia-worker/internal/admin"
	"github.com/inferia/inferia-worker/internal/auth"
	"github.com/inferia/inferia-worker/internal/cloudenv"
	"github.com/inferia/inferia-worker/internal/config"
	"github.com/inferia/inferia-worker/internal/control"
	"github.com/inferia/inferia-worker/internal/dispatcher"
	"github.com/inferia/inferia-worker/internal/healthz"
	"github.com/inferia/inferia-worker/internal/inference"
	"github.com/inferia/inferia-worker/internal/metrics"
	"github.com/inferia/inferia-worker/internal/runtime"
	"github.com/inferia/inferia-worker/internal/runtime/dockerclient"
	"github.com/inferia/inferia-worker/internal/runtime/recipes"
	"github.com/inferia/inferia-worker/internal/shellbridge"
	"github.com/inferia/inferia-worker/internal/telemetry"
)

func main() {
	// Pre-set TOKEN_FILE_PRESENT so config validation accepts a missing
	// BOOTSTRAP_TOKEN when the persisted token already exists.
	tokenPath := os.Getenv("TOKEN_FILE")
	if tokenPath == "" {
		tokenPath = "/var/lib/inferia-worker/token"
	}
	if auth.FilePresent(tokenPath) {
		_ = os.Setenv("TOKEN_FILE_PRESENT", "true")
	}

	cfg, err := config.Load()
	if err != nil {
		log.Fatalf("config: %v", err)
	}

	tokens, err := auth.NewTokenStore(cfg.TokenFile)
	if err != nil {
		log.Fatalf("token store: %v", err)
	}

	docker, err := dockerclient.NewEngine(cfg.DockerHost)
	if err != nil {
		log.Fatalf("docker engine: %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	if err := docker.Ping(ctx); err != nil {
		log.Fatalf("docker ping: %v (is /var/run/docker.sock mounted?)", err)
	}

	rt := runtime.New(runtime.Config{
		Docker:                docker,
		Network:               cfg.ModelsNetwork,
		PullTimeout:           time.Duration(cfg.PullTimeoutSeconds) * time.Second,
		ReadinessTimeout:      time.Duration(cfg.ReadinessTimeoutSeconds) * time.Second,
		ReadinessPollInterval: 500 * time.Millisecond,
	})
	if err := rt.EnsureNetwork(ctx); err != nil {
		log.Fatalf("ensure docker network: %v", err)
	}

	runtimeInfo := cloudenv.Detect()
	log.Printf("cloudenv: kind=%s instance=%s region=%s az=%s",
		runtimeInfo.Kind, runtimeInfo.InstanceID, runtimeInfo.Region, runtimeInfo.AvailabilityZone)

	gpus, _ := telemetry.ReadGPU()
	mem, _ := telemetry.ReadMemory()

	// Bootstrap if no token persisted.
	if tokens.Get() == "" {
		log.Printf("registering with control plane at %s", cfg.ControlPlaneURL)
		b := &control.Bootstrapper{
			ControlPlaneURL: cfg.ControlPlaneURL,
			BootstrapToken:  cfg.BootstrapToken,
		}
		cpu, _ := telemetry.ReadCPU()
		gpuModels := []string{}
		for _, g := range gpus {
			gpuModels = append(gpuModels, g.Name)
		}
		// The control plane's inventory upsert reads `cpu`, `memory_gb`, and
		// `gpu` keys (see inventory_repo._upsert_worker_impl). MemInfo.Total
		// is in bytes; convert to whole GiB (round down).
		memGB := mem.Total / (1024 * 1024 * 1024)
		alloc := map[string]string{
			"cpu":        fmt.Sprintf("%d", cpu.Cores),
			"memory_gb":  fmt.Sprintf("%d", memGB),
			"gpu":        fmt.Sprintf("%d", len(gpus)),
			"gpu_models": strings.Join(gpuModels, "|"),
		}
		// Distroless images ship without nvidia-smi, so telemetry.ReadGPU()
		// always sees zero GPUs even when the nvidia container runtime is
		// passing devices through. ALLOCATABLE_GPU_OVERRIDE lets operators
		// declare the GPU count manually; ALLOCATABLE_GPU_MODELS_OVERRIDE
		// supplies the pipe-joined model-name list for the same case.
		if v := strings.TrimSpace(os.Getenv("ALLOCATABLE_GPU_OVERRIDE")); v != "" {
			alloc["gpu"] = v
		}
		if v := strings.TrimSpace(os.Getenv("ALLOCATABLE_GPU_MODELS_OVERRIDE")); v != "" {
			alloc["gpu_models"] = v
		}
		resp, err := b.Register(ctx, control.BuildRegisterRequest(control.BuildRegisterInput{
			NodeName:       cfg.NodeName,
			PoolID:         cfg.PoolID,
			AdvertiseURL:   cfg.WorkerAdvertiseURL,
			Allocatable:    alloc,
			Runtime:        runtimeInfo,
			BootstrapToken: cfg.BootstrapToken,
		}))
		if err != nil {
			log.Fatalf("bootstrap: %v", err)
		}
		if err := tokens.Set(resp.WorkerJWT); err != nil {
			log.Fatalf("persist token: %v", err)
		}
		log.Printf("registered as node_id=%s", resp.NodeID)
	}

	ready := healthz.New()

	mc := metrics.NewCollector()
	reg := inference.NewDeploymentRegistry()

	// Fiber app.
	app := fiber.New(fiber.Config{
		DisableStartupMessage: true,
		ReadTimeout:           30 * time.Second,
		WriteTimeout:          0, // streaming responses can take minutes
		IdleTimeout:           120 * time.Second,
		BodyLimit:             64 * 1024 * 1024, // 64 MiB cap
	})
	app.Use(auth.NewInferenceTokenMiddleware(cfg.InferenceToken))

	// Admin WS endpoints (live container logs + interactive shell). Mount
	// these before the generic inference proxy so they win the route match.
	// Also install shellbridge's default backends so the channel tunnel
	// path can spawn sessions on demand (the CP-driven dashboard route
	// uses these; the admin handlers do their own resolution chain).
	if raw, ok := docker.(dockerclient.RawAccessor); ok {
		admin.Register(app, raw.Raw(), rt)
		shellbridge.SetDockerClient(raw.Raw(), rt)
		shellbridge.SetDockerLogsBackend(raw.Raw(), rt)
		// Host-shell mode: untargeted ShellOpen frames (empty deployment_id
		// and empty container_id) land here. Implemented as a privileged
		// sidecar + nsenter so the operator gets a real shell on the EC2
		// host rather than docker-exec'ing into the distroless worker
		// container.
		shellbridge.SetHostShellBackend(raw.Raw())
	} else {
		log.Printf("admin endpoints: docker client does not expose Raw(); logs/shell tabs disabled")
	}

	app.Use(inference.NewProxy(inference.Config{
		Runtime: rt,
		Resolver: inference.PathResolver{
			Header: "X-Inferia-Deployment-Id",
		},
		Metrics:  mc,
		Registry: reg,
	}))
	healthz.Register(app, ready)

	// Control channel dispatcher.
	gpuName := ""
	var gpuMemMiB uint64
	if len(gpus) > 0 {
		gpuName = gpus[0].Name
		gpuMemMiB = gpus[0].MemoryTotalMiB
	}
	disp := dispatcher.NewDispatcher(
		&runtimeAdapter{r: rt},
		hostTelemetry{},
		mc,
		gpuName,
		gpuMemMiB,
	)
	disp.Registry = reg

	ch := &control.Channel{
		ChannelURL: toWS(cfg.ControlPlaneURL) + "/v1/workers/channel",
		Token: func() string {
			return tokens.Get()
		},
		HeartbeatInterval: time.Duration(cfg.HeartbeatIntervalSeconds) * time.Second,
		Dispatcher:        disp,
		DedupTTL:          5 * time.Minute,
	}
	ch.Runtime = runtimeInfo

	// Start background vLLM metrics scraper.
	disp.StartScraper(ctx, 15*time.Second)

	// Run Fiber + control channel until signal.
	var wg sync.WaitGroup
	sigs := make(chan os.Signal, 1)
	signal.Notify(sigs, syscall.SIGINT, syscall.SIGTERM)

	wg.Add(1)
	go func() {
		defer wg.Done()
		log.Printf("listening on %s", cfg.WorkerListenAddr)
		if err := app.Listen(cfg.WorkerListenAddr); err != nil && err != http.ErrServerClosed {
			log.Printf("fiber: %v", err)
			cancel()
		}
	}()

	wg.Add(1)
	go func() {
		defer wg.Done()
		log.Printf("connecting to control channel at %s", ch.ChannelURL)
		// MarkReady on first successful Hello — we approximate by marking after
		// a short delay so /readyz becomes 200 only once the channel has had a
		// chance to connect.
		go func() {
			time.Sleep(2 * time.Second)
			ready.MarkReady()
		}()
		_ = ch.Run(ctx)
	}()

	<-sigs
	log.Println("shutdown requested")
	cancel()
	_ = app.ShutdownWithTimeout(10 * time.Second)
	wg.Wait()
	log.Println("bye")
}

// runtimeAdapter narrows *runtime.Runtime to the dispatcher.Runtime interface.
type runtimeAdapter struct{ r *runtime.Runtime }

func (a *runtimeAdapter) LoadModel(ctx context.Context, id string, plan recipes.Plan) (*runtime.LoadResult, error) {
	res, err := a.r.LoadModel(ctx, id, plan)
	if err != nil {
		return nil, err
	}
	return &runtime.LoadResult{EndpointURL: res.EndpointURL}, nil
}
func (a *runtimeAdapter) UnloadModel(ctx context.Context, id string) error {
	return a.r.UnloadModel(ctx, id)
}
func (a *runtimeAdapter) LoadedDeployments() []string { return a.r.LoadedDeployments() }
func (a *runtimeAdapter) DeploymentInfo(deploymentID string) (recipe, model, phase string, pullDur, startDur time.Duration, ok bool) {
	return a.r.DeploymentInfo(deploymentID)
}
func (a *runtimeAdapter) EndpointURL(deploymentID string) string { return a.r.EndpointURL(deploymentID) }
func (a *runtimeAdapter) LoadDeploymentGroup(ctx context.Context, plan recipes.DeploymentPlan) (*runtime.DeploymentGroup, error) {
	return a.r.LoadDeploymentGroup(ctx, plan)
}
func (a *runtimeAdapter) UnloadDeploymentGroup(ctx context.Context, id string) error {
	return a.r.UnloadDeploymentGroup(ctx, id)
}

// hostTelemetry reads CPU/memory/GPU from the host.
type hostTelemetry struct{}

func (hostTelemetry) Read() map[string]string {
	cpu, _ := telemetry.ReadCPU()
	mem, _ := telemetry.ReadMemory()
	gpus, _ := telemetry.ReadGPU()
	return map[string]string{
		"cpu_pct":  fmt.Sprintf("%.2f", cpu.UsedPercent),
		"mem_used": fmt.Sprintf("%d", mem.Used),
		"gpu":      fmt.Sprintf("%d", len(gpus)),
	}
}

// toWS rewrites http(s):// to ws(s):// preserving the rest of the URL.
func toWS(u string) string {
	switch {
	case strings.HasPrefix(u, "https://"):
		return "wss://" + u[len("https://"):]
	case strings.HasPrefix(u, "http://"):
		return "ws://" + u[len("http://"):]
	default:
		return u
	}
}
