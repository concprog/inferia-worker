// Package recipes maps a recipe name (e.g. "vllm", "ollama") to the inputs
// needed to launch the corresponding model server in a Docker container.
//
// Recipes are ported from
// inferiaLLM/package/src/inferia/services/orchestration/services/adapter_engine/adapters/nosana/job_builder.py
// with the simplifications that:
//   - the worker provides auth in front of the container, so no Caddy proxy / API key
//     injection is performed here;
//   - the container is run directly with its native entrypoint, not as a shell script.
package recipes

import (
	"fmt"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

// Plan is the concrete Docker invocation produced from a (Recipe, BuildInput) pair.
// dockerclient consumes this verbatim.
type Plan struct {
	Image         string
	ContainerName string
	Cmd           []string          // CMD/args; runs with the image's default entrypoint
	Env           map[string]string // env passed to the container
	ContainerPort int               // port the model server listens on inside the container
	HostPort      int               // port to bind on the host (chosen by the worker)
	GPUIndices    []int             // GPU device indices to pass to --gpus
	ReadyPath     string            // HTTP path used for readiness probe (200 means ready)
}

// BuildInput is what the worker passes when constructing a plan.
type BuildInput struct {
	DeploymentID string
	ArtifactURI  string
	Config       map[string]any
	GPUIndices   []int
	HostPort     int
	Env          map[string]string
	GPUName      string // populated by dispatcher from telemetry.ReadGPU()
	GPUMemoryMiB uint64 // populated by dispatcher from telemetry.ReadGPU()
}

// Recipe builds a Plan for a particular engine/runtime.
type Recipe interface {
	BuildPlan(in BuildInput) (Plan, error)
}

const (
	maxConfigKeys = 64
	maxConfigKey  = 128
)

// Allowed runtime-config keys (mirrors InferiaLLM/llmd/spec_builder.py). Values
// must be safe scalar types; nested structures are dropped.
var allowedConfigKeys = map[string]struct{}{
	"tensor_parallel_size":   {},
	"pipeline_parallel_size": {},
	"dtype":                  {},
	"max_model_len":          {},
	"max_num_seqs":           {},
	"gpu_memory_utilization": {},
	"quantization":           {},
	"enforce_eager":          {},
	"trust_remote_code":      {},
	"max_batch_size":         {},
	"max_input_length":       {},
	"max_total_tokens":       {},
}

// Allowed URI schemes (mirrors spec_builder.py).
var allowedURISchemes = map[string]struct{}{
	"s3":    {},
	"gs":    {},
	"hf":    {},
	"http":  {},
	"https": {},
	"oci":   {},
}

// URI must start with scheme:// then no control chars or shell metachars.
// (Mirrors the Python pattern with the same intent.)
var uriPattern = regexp.MustCompile(`^[a-z][a-z0-9+\-.]*://[^\x00-\x1f` + "`" + `$;|&><(\)\n\r]+$`)

var registry = map[string]Recipe{
	"vllm":              vllmRecipe{image: "docker.io/vllm/vllm-openai:v0.22.1", port: 8000, readyPath: "/health"},
	"vllm-omni":         vllmRecipe{image: "docker.io/vllm/vllm-omni:v0.11.0rc1", port: 8000, readyPath: "/health"},
	"ollama":            ollamaRecipe{image: "docker.io/ollama/ollama:latest", port: 11434, readyPath: "/"},
	"infinity":          infinityRecipe{image: "michaelf34/infinity:latest", port: 7997, readyPath: "/health"},
	"triton":            tritonRecipe{image: "nvcr.io/nvidia/tritonserver:latest", port: 8000, readyPath: "/v2/health/ready"},
	"inferia-diffusion": diffusionRecipe{image: "docker.io/inferiaai/inferia-diffusion:latest", port: 8000, readyPath: "/health"},
}

// Get returns the recipe registered under name.
func Get(name string) (Recipe, error) {
	r, ok := registry[name]
	if !ok {
		return nil, fmt.Errorf("unknown recipe: %q", name)
	}
	return r, nil
}

// Names returns the sorted list of registered recipe names.
func Names() []string {
	out := make([]string, 0, len(registry))
	for k := range registry {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// validate runs shared input checks. Recipes call this first.
//
// Note: GPU index presence is NOT checked here. CPU-friendly engines
// (ollama, infinity) can run inference without a GPU, so they pass
// GPUIndices == nil. GPU-only engines (vllm, triton, diffusion) must
// additionally call requireGPU(in) to enforce ≥1 index.
func validate(in BuildInput) error {
	if in.DeploymentID == "" {
		return fmt.Errorf("DeploymentID is required")
	}
	if err := validateArtifactURI(in.ArtifactURI); err != nil {
		return err
	}
	for _, i := range in.GPUIndices {
		if i < 0 {
			return fmt.Errorf("GPU index must be ≥ 0, got %d", i)
		}
	}
	if in.HostPort < 1 || in.HostPort > 65535 {
		return fmt.Errorf("HostPort %d out of range [1, 65535]", in.HostPort)
	}
	if len(in.Config) > maxConfigKeys {
		return fmt.Errorf("config has %d keys, max %d", len(in.Config), maxConfigKeys)
	}
	for k := range in.Config {
		if len(k) > maxConfigKey {
			return fmt.Errorf("config key length %d exceeds max %d", len(k), maxConfigKey)
		}
	}
	return nil
}

// requireGPU enforces that at least one GPU index is present. GPU-only
// recipes (vllm, triton, diffusion) call this after validate(); CPU-friendly
// recipes (ollama, infinity) do not, so a deploy onto a CPU-only worker
// host succeeds for those engines.
func requireGPU(in BuildInput) error {
	if len(in.GPUIndices) == 0 {
		return fmt.Errorf("at least one GPU index is required")
	}
	return nil
}

func validateArtifactURI(uri string) error {
	if uri == "" || strings.TrimSpace(uri) == "" {
		return fmt.Errorf("artifact URI required")
	}
	if !uriPattern.MatchString(uri) {
		return fmt.Errorf("artifact URI contains invalid characters: %q", uri)
	}
	scheme := strings.SplitN(uri, "://", 2)[0]
	if _, ok := allowedURISchemes[strings.ToLower(scheme)]; !ok {
		return fmt.Errorf("artifact URI scheme %q not allowed", scheme)
	}
	return nil
}

// stripScheme returns the part after "scheme://"; if no scheme is present
// returns the input unchanged.
func stripScheme(uri string) string {
	i := strings.Index(uri, "://")
	if i < 0 {
		return uri
	}
	return uri[i+3:]
}

// sanitiseConfig keeps only allowlisted keys whose values are safe scalars
// (string, int, int64, float64, bool). Everything else is dropped.
func sanitiseConfig(in map[string]any) map[string]any {
	if len(in) == 0 {
		return map[string]any{}
	}
	out := map[string]any{}
	for k, v := range in {
		if _, ok := allowedConfigKeys[k]; !ok {
			continue
		}
		switch v.(type) {
		case string, int, int32, int64, float32, float64, bool:
			out[k] = v
		}
	}
	return out
}

func containerName(prefix, deploymentID string) string {
	// Docker disallows '/' in container names; sanitise just in case.
	safe := strings.NewReplacer("/", "-", ":", "-", "..", "-").Replace(deploymentID)
	return prefix + "-" + safe
}

func cliArg(v any) string {
	switch x := v.(type) {
	case string:
		return x
	case bool:
		if x {
			return "true"
		}
		return "false"
	case int:
		return strconv.Itoa(x)
	case int32:
		return strconv.FormatInt(int64(x), 10)
	case int64:
		return strconv.FormatInt(x, 10)
	case float32:
		return strconv.FormatFloat(float64(x), 'f', -1, 32)
	case float64:
		return strconv.FormatFloat(x, 'f', -1, 64)
	default:
		return fmt.Sprintf("%v", v)
	}
}

func dashed(k string) string {
	return "--" + strings.ReplaceAll(k, "_", "-")
}
