package recipes

import (
	"strings"
	"testing"
	"time"
)

func TestRegistry_KnownRecipes(t *testing.T) {
	want := []string{"vllm", "ollama", "vllm-omni", "sglang", "infinity", "triton", "inferia-diffusion"}
	for _, name := range want {
		if _, err := Get(name); err != nil {
			t.Errorf("Get(%q): %v", name, err)
		}
	}
}

func TestRegistry_UnknownRecipe(t *testing.T) {
	if _, err := Get("nope"); err == nil {
		t.Errorf("expected error for unknown recipe")
	}
	if _, err := Get(""); err == nil {
		t.Errorf("expected error for empty recipe")
	}
}

func TestRegistry_Names_Sorted(t *testing.T) {
	names := Names()
	if len(names) != 7 {
		t.Fatalf("got %d names", len(names))
	}
	prev := ""
	for _, n := range names {
		if n < prev {
			t.Errorf("names not sorted: %v", names)
			break
		}
		prev = n
	}
}

// Plan covers the user-controllable inputs that flow into BuildPlan: model URI,
// config map, GPU indices, host bind port, deployment id.

func TestBuildPlan_VLLM_Defaults(t *testing.T) {
	r, _ := Get("vllm")
	plan, err := r.BuildPlan(BuildInput{
		DeploymentID: "dep-1",
		ArtifactURI:  "hf://meta-llama/Llama-3.1-8B-Instruct",
		Config:       nil,
		GPUIndices:   []int{0},
		HostPort:     19000,
	})
	if err != nil {
		t.Fatalf("%v", err)
	}
	if !strings.Contains(plan.Image, "vllm-openai") {
		t.Errorf("image: %q", plan.Image)
	}
	if plan.ContainerPort != 8000 && plan.ContainerPort != 9000 {
		t.Errorf("port: %d", plan.ContainerPort)
	}
	if plan.HostPort != 19000 {
		t.Errorf("HostPort: %d", plan.HostPort)
	}
	if plan.ReadyPath == "" {
		t.Errorf("ReadyPath empty")
	}
	if plan.ContainerName == "" || !strings.Contains(plan.ContainerName, "dep-1") {
		t.Errorf("ContainerName: %q", plan.ContainerName)
	}
	// Model id should appear in the command somewhere (as the model arg).
	joined := strings.Join(plan.Cmd, " ")
	if !strings.Contains(joined, "meta-llama/Llama-3.1-8B-Instruct") {
		t.Errorf("cmd missing model: %v", plan.Cmd)
	}
}

// Regression: the vllm-openai image bakes a CUDA forward-compat libcuda ahead
// of the standard lib dirs, which mismatches a newer host kernel driver and
// makes CUDA init fail with Error 803 before any weights are fetched (deploy
// hangs in DEPLOYING). The recipe must default LD_LIBRARY_PATH so the multiarch
// dir (host driver injected by nvidia-container-toolkit) is searched FIRST.
func TestBuildPlan_VLLM_SetsLDLibraryPathForCudaCompat(t *testing.T) {
	r, _ := Get("vllm")
	plan, err := r.BuildPlan(BuildInput{
		DeploymentID: "dep-ld",
		ArtifactURI:  "hf://Qwen/Qwen3-0.6B",
		GPUIndices:   []int{0},
		HostPort:     19000,
	})
	if err != nil {
		t.Fatalf("%v", err)
	}
	ldp, ok := plan.Env["LD_LIBRARY_PATH"]
	if !ok {
		t.Fatalf("LD_LIBRARY_PATH not set; env=%v", plan.Env)
	}
	// The host-driver multiarch dir MUST come first so it wins over the image's
	// baked compat libcuda.
	if !strings.HasPrefix(ldp, "/usr/lib/x86_64-linux-gnu") {
		t.Errorf("LD_LIBRARY_PATH must start with the multiarch dir, got %q", ldp)
	}
	// The toolkit and cuda lib dirs must remain on the path for the rest of the
	// CUDA userspace.
	for _, want := range []string{"/usr/local/nvidia/lib64", "/usr/local/cuda/lib64"} {
		if !strings.Contains(ldp, want) {
			t.Errorf("LD_LIBRARY_PATH missing %q, got %q", want, ldp)
		}
	}
	// The existing CUDA_MODULE_LOADING default must still be present.
	if plan.Env["CUDA_MODULE_LOADING"] != "LAZY" {
		t.Errorf("CUDA_MODULE_LOADING=%q, want LAZY", plan.Env["CUDA_MODULE_LOADING"])
	}
}

// A deploy that explicitly supplies LD_LIBRARY_PATH (rare, advanced) must be
// able to override the default — mergeEnv applies user env over defaults.
func TestBuildPlan_VLLM_UserLDLibraryPathOverridesDefault(t *testing.T) {
	r, _ := Get("vllm")
	plan, err := r.BuildPlan(BuildInput{
		DeploymentID: "dep-ld2",
		ArtifactURI:  "hf://Qwen/Qwen3-0.6B",
		GPUIndices:   []int{0},
		HostPort:     19000,
		Env:          map[string]string{"LD_LIBRARY_PATH": "/custom/path"},
	})
	if err != nil {
		t.Fatalf("%v", err)
	}
	if plan.Env["LD_LIBRARY_PATH"] != "/custom/path" {
		t.Errorf("user LD_LIBRARY_PATH not honored, got %q", plan.Env["LD_LIBRARY_PATH"])
	}
}

func TestBuildPlan_Ollama_StripsHFScheme(t *testing.T) {
	r, _ := Get("ollama")
	plan, err := r.BuildPlan(BuildInput{
		DeploymentID: "dep-2",
		ArtifactURI:  "hf://llama3",
		GPUIndices:   []int{0},
		HostPort:     11434,
	})
	if err != nil {
		t.Fatalf("%v", err)
	}
	// The model id must appear somewhere (cmd or env) so the runtime layer can
	// pull it after the container is ready, and it must not carry the URI scheme.
	allText := strings.Join(plan.Cmd, " ")
	for k, v := range plan.Env {
		allText += " " + k + "=" + v
	}
	if strings.Contains(allText, "hf://") {
		t.Errorf("scheme leaked into plan: %v / %v", plan.Cmd, plan.Env)
	}
	if !strings.Contains(allText, "llama3") {
		t.Errorf("model id missing from plan: %v / %v", plan.Cmd, plan.Env)
	}
}

// Security: artifact URIs go into docker invocations. We accept only safe
// schemes and reject any control/metacharacters.

func TestBuildPlan_RejectsBadURISchemes(t *testing.T) {
	r, _ := Get("vllm")
	bad := []string{
		"file:///etc/passwd",
		"javascript:alert(1)",
		"data:text/plain,abc",
		"",
		"   ",
		"no-scheme-here",
		"ftp://example.com/model",
	}
	for _, uri := range bad {
		_, err := r.BuildPlan(BuildInput{
			DeploymentID: "d",
			ArtifactURI:  uri,
			GPUIndices:   []int{0},
			HostPort:     1234,
		})
		if err == nil {
			t.Errorf("expected reject for %q", uri)
		}
	}
}

func TestBuildPlan_AcceptsAllowedURISchemes(t *testing.T) {
	r, _ := Get("vllm")
	good := []string{
		"s3://bucket/path",
		"gs://bucket/path",
		"hf://org/model",
		"http://example.com/m",
		"https://example.com/m",
		"oci://registry/image:tag",
	}
	for _, uri := range good {
		if _, err := r.BuildPlan(BuildInput{
			DeploymentID: "d",
			ArtifactURI:  uri,
			GPUIndices:   []int{0},
			HostPort:     1234,
		}); err != nil {
			t.Errorf("expected accept for %q: %v", uri, err)
		}
	}
}

func TestBuildPlan_RejectsURIWithMetachars(t *testing.T) {
	r, _ := Get("vllm")
	for _, uri := range []string{
		"hf://model;rm -rf /",
		"hf://model`whoami`",
		"hf://model$(id)",
		"hf://model\nrm",
		"hf://model|cat",
	} {
		if _, err := r.BuildPlan(BuildInput{
			DeploymentID: "d",
			ArtifactURI:  uri,
			GPUIndices:   []int{0},
			HostPort:     1234,
		}); err == nil {
			t.Errorf("expected reject for %q", uri)
		}
	}
}

func TestBuildPlan_ConfigSanitisation(t *testing.T) {
	r, _ := Get("vllm")
	plan, err := r.BuildPlan(BuildInput{
		DeploymentID: "d",
		ArtifactURI:  "hf://org/m",
		GPUIndices:   []int{0},
		HostPort:     1234,
		Config: map[string]any{
			"tensor_parallel_size":   2,
			"gpu_memory_utilization": 0.9,
			"dtype":                  "bfloat16",
			"max_model_len":          4096,

			// Disallowed: should be silently dropped.
			"arbitrary_key":  "DROP",
			"trust_anything": true,

			// Wrong type for allowed key: should be dropped.
			"max_num_seqs": []int{1, 2, 3},

			// Nested dict for allowed key: dropped.
			"quantization": map[string]string{"k": "v"},
		},
	})
	if err != nil {
		t.Fatalf("%v", err)
	}
	joined := strings.Join(plan.Cmd, " ")
	if strings.Contains(joined, "arbitrary_key") || strings.Contains(joined, "DROP") {
		t.Errorf("disallowed config leaked: %v", plan.Cmd)
	}
	if !strings.Contains(joined, "tensor-parallel-size") && !strings.Contains(joined, "tensor_parallel_size") {
		t.Errorf("allowed config missing: %v", plan.Cmd)
	}
}

func TestBuildPlan_OversizedConfigRejected(t *testing.T) {
	r, _ := Get("vllm")
	cfg := map[string]any{}
	for i := 0; i < 65; i++ {
		cfg["k_"+strings.Repeat("x", i)] = i
	}
	_, err := r.BuildPlan(BuildInput{
		DeploymentID: "d",
		ArtifactURI:  "hf://org/m",
		GPUIndices:   []int{0},
		HostPort:     1234,
		Config:       cfg,
	})
	if err == nil {
		t.Errorf("expected error for >64-key config")
	}
}

func TestBuildPlan_LongKeyRejected(t *testing.T) {
	r, _ := Get("vllm")
	_, err := r.BuildPlan(BuildInput{
		DeploymentID: "d",
		ArtifactURI:  "hf://org/m",
		GPUIndices:   []int{0},
		HostPort:     1234,
		Config:       map[string]any{strings.Repeat("k", 129): 1},
	})
	if err == nil {
		t.Errorf("expected error for >128-byte key")
	}
}

func TestBuildPlan_NoGPUIndices(t *testing.T) {
	r, _ := Get("vllm")
	_, err := r.BuildPlan(BuildInput{
		DeploymentID: "d",
		ArtifactURI:  "hf://org/m",
		GPUIndices:   nil,
		HostPort:     1234,
	})
	if err == nil {
		t.Errorf("vllm expects ≥1 GPU; nil should error")
	}
}

func TestBuildPlan_NegativeGPUIndexRejected(t *testing.T) {
	r, _ := Get("vllm")
	_, err := r.BuildPlan(BuildInput{
		DeploymentID: "d",
		ArtifactURI:  "hf://org/m",
		GPUIndices:   []int{-1},
		HostPort:     1234,
	})
	if err == nil {
		t.Errorf("negative GPU index must reject")
	}
}

func TestBuildPlan_DeploymentIDRequired(t *testing.T) {
	r, _ := Get("vllm")
	if _, err := r.BuildPlan(BuildInput{
		ArtifactURI: "hf://org/m",
		GPUIndices:  []int{0},
		HostPort:    1234,
	}); err == nil {
		t.Errorf("expected error for missing DeploymentID")
	}
}

func TestBuildPlan_AllRecipesProduceValidPlan(t *testing.T) {
	// Each shipped recipe must roundtrip a minimal input without error.
	for _, name := range Names() {
		t.Run(name, func(t *testing.T) {
			r, _ := Get(name)
			plan, err := r.BuildPlan(BuildInput{
				DeploymentID: "d-" + name,
				ArtifactURI:  "hf://org/m",
				GPUIndices:   []int{0},
				HostPort:     20000,
			})
			if err != nil {
				t.Fatalf("%v", err)
			}
			if plan.Image == "" {
				t.Errorf("Image empty")
			}
			if plan.ContainerPort == 0 {
				t.Errorf("ContainerPort == 0")
			}
			if plan.ReadyPath == "" {
				t.Errorf("ReadyPath empty")
			}
			if plan.ContainerName == "" {
				t.Errorf("ContainerName empty")
			}
		})
	}
}

func TestBuildPlan_EnvVarsHonoured(t *testing.T) {
	// HF_TOKEN provided via Env propagates into the container env.
	r, _ := Get("vllm")
	plan, err := r.BuildPlan(BuildInput{
		DeploymentID: "d",
		ArtifactURI:  "hf://org/m",
		GPUIndices:   []int{0},
		HostPort:     1234,
		Env:          map[string]string{"HF_TOKEN": "secret123"},
	})
	if err != nil {
		t.Fatalf("%v", err)
	}
	if plan.Env["HF_TOKEN"] != "secret123" {
		t.Errorf("HF_TOKEN missing from plan env")
	}
}

// CPU-tier instances ship without GPUs. Engines that can run inference on
// CPU (ollama, infinity) must build a valid Plan with GPUIndices == nil;
// GPU-only engines (vllm, triton, diffusion) must still reject.
//
// The plan task originally specified these against a hypothetical
// `Prepare(plan)` API; the actual repo uses per-recipe `BuildPlan(in)`,
// so the test bodies are adapted to the existing surface but the names
// and intent map 1:1 to plan task 28.

func TestPrepareAllowsZeroGPUsForCpuFriendlyEngines(t *testing.T) {
	// ollama is in the CPU-friendly set: must NOT reject zero GPUIndices.
	r, err := Get("ollama")
	if err != nil {
		t.Fatalf("Get(ollama): %v", err)
	}
	plan, err := r.BuildPlan(BuildInput{
		DeploymentID: "d-1",
		ArtifactURI:  "hf://smollm2:135m",
		GPUIndices:   []int{},
		HostPort:     19000,
	})
	if err != nil {
		t.Fatalf("ollama with zero GPU should be allowed; got error: %v", err)
	}
	if plan.Image == "" {
		t.Errorf("expected non-empty image for ollama CPU plan")
	}
	if len(plan.GPUIndices) != 0 {
		t.Errorf("expected empty GPUIndices in plan, got %v", plan.GPUIndices)
	}
}

func TestPrepareRejectsZeroGPUsForGpuOnlyEngines(t *testing.T) {
	// vllm is GPU-only: must still reject zero GPUIndices.
	r, err := Get("vllm")
	if err != nil {
		t.Fatalf("Get(vllm): %v", err)
	}
	if _, err := r.BuildPlan(BuildInput{
		DeploymentID: "d-1",
		ArtifactURI:  "hf://Qwen/Qwen3-0.6B",
		GPUIndices:   []int{},
		HostPort:     19000,
	}); err == nil {
		t.Fatal("vllm with zero GPU should be rejected")
	}
	// triton and inferia-diffusion are also GPU-only; check them too so a
	// future refactor that drops requireGPU from any of them is caught here.
	for _, name := range []string{"triton", "inferia-diffusion", "vllm-omni", "sglang"} {
		r, err := Get(name)
		if err != nil {
			t.Fatalf("Get(%q): %v", name, err)
		}
		if _, err := r.BuildPlan(BuildInput{
			DeploymentID: "d-1",
			ArtifactURI:  "hf://org/model",
			GPUIndices:   []int{},
			HostPort:     19000,
		}); err == nil {
			t.Errorf("%s with zero GPU should be rejected", name)
		}
	}
}

func TestPrepareAllowsZeroGPUsForInfinity(t *testing.T) {
	r, err := Get("infinity")
	if err != nil {
		t.Fatalf("Get(infinity): %v", err)
	}
	plan, err := r.BuildPlan(BuildInput{
		DeploymentID: "d-1",
		ArtifactURI:  "hf://BAAI/bge-small-en-v1.5",
		GPUIndices:   []int{},
		HostPort:     19000,
	})
	if err != nil {
		t.Fatalf("infinity with zero GPU should be allowed; got error: %v", err)
	}
	// Even on CPU, the model id must propagate to the command.
	if !strings.Contains(strings.Join(plan.Cmd, " "), "BAAI/bge-small-en-v1.5") {
		t.Errorf("model id missing from infinity CPU plan cmd: %v", plan.Cmd)
	}
}

// Nil-slice (as distinct from empty-slice) must also be accepted for
// CPU-friendly engines and rejected for GPU-only engines, because Go
// treats `nil` and `[]int{}` differently in JSON/proto deserialisation
// and we don't want a regression there.
func TestPrepareCpuFriendly_NilGPUIndicesAlsoAllowed(t *testing.T) {
	for _, name := range []string{"ollama", "infinity"} {
		r, _ := Get(name)
		if _, err := r.BuildPlan(BuildInput{
			DeploymentID: "d-nil",
			ArtifactURI:  "hf://org/m",
			GPUIndices:   nil,
			HostPort:     19000,
		}); err != nil {
			t.Errorf("%s with nil GPUIndices should be allowed: %v", name, err)
		}
	}
}

// Negative GPU indices must still reject even on CPU-friendly engines —
// a negative entry is always an input bug, never an intentional CPU run.
func TestPrepareCpuFriendly_StillRejectsNegativeIndex(t *testing.T) {
	r, _ := Get("ollama")
	if _, err := r.BuildPlan(BuildInput{
		DeploymentID: "d",
		ArtifactURI:  "hf://m",
		GPUIndices:   []int{-1},
		HostPort:     19000,
	}); err == nil {
		t.Errorf("ollama with negative GPU index must still reject")
	}
}

func TestBuildPlan_HostPortRange(t *testing.T) {
	r, _ := Get("vllm")
	for _, p := range []int{0, -1, 65536, 99999} {
		if _, err := r.BuildPlan(BuildInput{
			DeploymentID: "d", ArtifactURI: "hf://o/m", GPUIndices: []int{0}, HostPort: p,
		}); err == nil {
			t.Errorf("expected error for HostPort=%d", p)
		}
	}
	if _, err := r.BuildPlan(BuildInput{
		DeploymentID: "d", ArtifactURI: "hf://o/m", GPUIndices: []int{0}, HostPort: 1,
	}); err != nil {
		t.Errorf("expected ok for HostPort=1: %v", err)
	}
	if _, err := r.BuildPlan(BuildInput{
		DeploymentID: "d", ArtifactURI: "hf://o/m", GPUIndices: []int{0}, HostPort: 65535,
	}); err != nil {
		t.Errorf("expected ok for HostPort=65535: %v", err)
	}
}

// --- Multi-container (disagg) recipe tests -----------------------------------

func TestMultiRegistry_KnownRecipes(t *testing.T) {
	if _, err := MultiGet("vllm-prefill-decode"); err != nil {
		t.Errorf("MultiGet(vllm-prefill-decode): %v", err)
	}
}

func TestMultiRegistry_UnknownRecipe(t *testing.T) {
	if _, err := MultiGet("nope"); err == nil {
		t.Errorf("expected error for unknown multi recipe")
	}
}

func TestBuildPlan_VLLMPrefillDecode_Defaults(t *testing.T) {
	mc, err := MultiGet("vllm-prefill-decode")
	if err != nil {
		t.Fatalf("%v", err)
	}
	plan, err := mc.BuildDeploymentPlan(BuildInput{
		DeploymentID:     "dep-disagg-1",
		ArtifactURI:      "hf://meta-llama/Llama-3.1-8B-Instruct",
		Config:           nil,
		GPUIndices:       []int{0, 1},
		HostPort:         1,
		PrefillReplicas:  2,
		DecodeReplicas:   2,
	})
	if err != nil {
		t.Fatalf("%v", err)
	}
	if plan.DeploymentID != "dep-disagg-1" {
		t.Errorf("DeploymentID=%q", plan.DeploymentID)
	}
	if plan.Model != "meta-llama/Llama-3.1-8B-Instruct" {
		t.Errorf("Model=%q", plan.Model)
	}
	if len(plan.Prefill) != 2 {
		t.Errorf("Prefill replicas: got %d, want 2", len(plan.Prefill))
	}
	if len(plan.Decode) != 2 {
		t.Errorf("Decode replicas: got %d, want 2", len(plan.Decode))
	}
	for i, cp := range plan.Prefill {
		if cp.Role != KvRoleProducer {
			t.Errorf("prefill[%d] Role=%q, want kv_producer", i, cp.Role)
		}
		if cp.ReplicaIdx != i {
			t.Errorf("prefill[%d] ReplicaIdx=%d", i, cp.ReplicaIdx)
		}
	}
	for i, cp := range plan.Decode {
		if cp.Role != KvRoleConsumer {
			t.Errorf("decode[%d] Role=%q, want kv_consumer", i, cp.Role)
		}
		if cp.ReplicaIdx != i {
			t.Errorf("decode[%d] ReplicaIdx=%d", i, cp.ReplicaIdx)
		}
	}
	if plan.Prefill[0].Image == "" {
		t.Errorf("Image empty")
	}
	if plan.Prefill[0].ContainerPort == 0 {
		t.Errorf("ContainerPort == 0")
	}
	if plan.Prefill[0].ReadyPath == "" {
		t.Errorf("ReadyPath empty")
	}
	for i := range plan.Decode {
		if plan.Decode[i].Image != plan.Prefill[0].Image {
			t.Errorf("decode[%d] image mismatch: %q vs %q", i, plan.Decode[i].Image, plan.Prefill[0].Image)
		}
	}
	joined := strings.Join(plan.Prefill[0].Cmd, " ")
	if !strings.Contains(joined, "meta-llama/Llama-3.1-8B-Instruct") {
		t.Errorf("cmd missing model: %v", plan.Prefill[0].Cmd)
	}
}

func TestBuildPlan_VLLMPrefillDecode_DefaultsReplicas(t *testing.T) {
	mc, _ := MultiGet("vllm-prefill-decode")
	plan, err := mc.BuildDeploymentPlan(BuildInput{
		DeploymentID: "dep-d",
		ArtifactURI:  "hf://org/m",
		GPUIndices:   []int{0},
		HostPort:     1,
	})
	if err != nil {
		t.Fatalf("%v", err)
	}
	if len(plan.Prefill) != 1 {
		t.Errorf("expected 1 prefill replica (default), got %d", len(plan.Prefill))
	}
	if len(plan.Decode) != 1 {
		t.Errorf("expected 1 decode replica (default), got %d", len(plan.Decode))
	}
}

func TestBuildPlan_VLLMPrefillDecode_RejectsBadInput(t *testing.T) {
	mc, _ := MultiGet("vllm-prefill-decode")
	if _, err := mc.BuildDeploymentPlan(BuildInput{
		ArtifactURI: "hf://org/m",
		GPUIndices:  []int{0},
		HostPort:    1,
	}); err == nil {
		t.Errorf("expected error for missing DeploymentID")
	}
	if _, err := mc.BuildDeploymentPlan(BuildInput{
		DeploymentID: "d",
		ArtifactURI:  "hf://org/m",
		GPUIndices:   []int{},
		HostPort:     1,
	}); err == nil {
		t.Errorf("expected error for zero GPU indices")
	}
}

func TestContainerPlan_ToPlan(t *testing.T) {
	cp := ContainerPlan{
		Image:         "img:tag",
		Cmd:           []string{"--arg", "val"},
		Entrypoint:    []string{"/entry.sh"},
		Env:           map[string]string{"K": "V"},
		Mounts:        []Mount{{Type: "volume", Source: "src", Target: "/tgt"}},
		ContainerPort: 8000,
		GPUIndices:    []int{0},
		ReadyPath:     "/health",
		Role:          KvRoleProducer,
		ReplicaIdx:    0,
	}
	p := cp.ToPlan("my-container", 19001)
	if p.Image != "img:tag" {
		t.Errorf("Image=%q", p.Image)
	}
	if p.ContainerName != "my-container" {
		t.Errorf("ContainerName=%q", p.ContainerName)
	}
	if p.HostPort != 19001 {
		t.Errorf("HostPort=%d", p.HostPort)
	}
	if p.ContainerPort != 8000 {
		t.Errorf("ContainerPort=%d", p.ContainerPort)
	}
	if len(p.Mounts) != 1 {
		t.Errorf("Mounts count=%d", len(p.Mounts))
	}
	if p.ReadyPath != "/health" {
		t.Errorf("ReadyPath=%q", p.ReadyPath)
	}
}

// --- Diffusion tests ---------------------------------------------------------

func TestDiffusionBuildPlan_ImageAndCmd(t *testing.T) {
	r, err := Get("inferia-diffusion")
	if err != nil {
		t.Fatalf("Get(inferia-diffusion): %v", err)
	}
	plan, err := r.BuildPlan(BuildInput{
		DeploymentID: "d-diff",
		ArtifactURI:  "hf://stabilityai/sdxl-turbo",
		GPUIndices:   []int{0},
		HostPort:     18000,
		Env:          map[string]string{"HF_TOKEN": "secret123"},
		Config: map[string]any{
			"model_type":        "image_generation",
			"trust_remote_code": true,
			"model_offload":     true,
			"group_offload":     false,
		},
	})
	if err != nil {
		t.Fatalf("BuildPlan: %v", err)
	}
	if plan.Image != "docker.io/inferiaai/inferiadiffusion:latest" {
		t.Errorf("image = %q", plan.Image)
	}
	got := strings.Join(plan.Cmd, " ")
	for _, want := range []string{
		"inferiadiffusion serve",
		"--model stabilityai/sdxl-turbo",
		"--host 0.0.0.0",
		"--port 8000",
		"--model-type image",
		"--trust-remote-code",
		"--model-offload",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("cmd %q missing %q", got, want)
		}
	}
	if strings.Contains(got, "--group-offload") {
		t.Errorf("cmd %q should not contain --group-offload", got)
	}
	if plan.Env["HF_TOKEN"] != "secret123" {
		t.Errorf("HF_TOKEN missing from plan env")
	}
	if plan.ReadyPath != "/health" {
		t.Errorf("ReadyPath = %q", plan.ReadyPath)
	}
}

func TestDiffusionBuildPlan_VideoModelType(t *testing.T) {
	r, _ := Get("inferia-diffusion")
	plan, err := r.BuildPlan(BuildInput{
		DeploymentID: "d-vid",
		ArtifactURI:  "hf://Wan-AI/Wan2.1-T2V-1.3B",
		GPUIndices:   []int{0},
		HostPort:     18001,
		Config:       map[string]any{"model_type": "video_generation"},
	})
	if err != nil {
		t.Fatalf("BuildPlan: %v", err)
	}
	if !strings.Contains(strings.Join(plan.Cmd, " "), "--model-type video") {
		t.Errorf("expected --model-type video, got %v", plan.Cmd)
	}
}

func TestDiffusionBuildPlan_NoModelTypeOmitsFlag(t *testing.T) {
	r, _ := Get("inferia-diffusion")
	plan, err := r.BuildPlan(BuildInput{
		DeploymentID: "d-auto",
		ArtifactURI:  "hf://segmind/tiny-sd",
		GPUIndices:   []int{0},
		HostPort:     18002,
	})
	if err != nil {
		t.Fatalf("BuildPlan: %v", err)
	}
	if strings.Contains(strings.Join(plan.Cmd, " "), "--model-type") {
		t.Errorf("expected no --model-type flag when unset, got %v", plan.Cmd)
	}
}

func TestDiffusionBuildPlan_UnknownModelTypeOmitsFlag(t *testing.T) {
	r, _ := Get("inferia-diffusion")
	plan, err := r.BuildPlan(BuildInput{
		DeploymentID: "d-unknown",
		ArtifactURI:  "hf://some/model",
		GPUIndices:   []int{0},
		HostPort:     18003,
		Config:       map[string]any{"model_type": "audio"},
	})
	if err != nil {
		t.Fatalf("BuildPlan: %v", err)
	}
	if strings.Contains(strings.Join(plan.Cmd, " "), "--model-type") {
		t.Errorf("expected no --model-type flag for unknown type, got %v", plan.Cmd)
	}
}

func TestDiffusionBuildPlan_BareImageAndVideoAliases(t *testing.T) {
	r, _ := Get("inferia-diffusion")
	for mt, want := range map[string]string{"image": "--model-type image", "video": "--model-type video"} {
		plan, err := r.BuildPlan(BuildInput{
			DeploymentID: "d-" + mt,
			ArtifactURI:  "hf://some/model",
			GPUIndices:   []int{0},
			HostPort:     18004,
			Config:       map[string]any{"model_type": mt},
		})
		if err != nil {
			t.Fatalf("BuildPlan(%q): %v", mt, err)
		}
		if !strings.Contains(strings.Join(plan.Cmd, " "), want) {
			t.Errorf("model_type=%q: expected %q, got %v", mt, want, plan.Cmd)
		}
	}
}

func TestDiffusionBuildPlan_ReadinessTimeoutGenerous(t *testing.T) {
	r, _ := Get("inferia-diffusion")
	plan, err := r.BuildPlan(BuildInput{
		DeploymentID: "d-rt", ArtifactURI: "hf://stabilityai/sdxl-turbo",
		GPUIndices: []int{0}, HostPort: 18005,
	})
	if err != nil {
		t.Fatalf("BuildPlan: %v", err)
	}
	if plan.ReadinessTimeout < 1800*time.Second {
		t.Errorf("diffusion ReadinessTimeout = %v, want >= 1800s", plan.ReadinessTimeout)
	}
	rv, _ := Get("vllm")
	pv, _ := rv.BuildPlan(BuildInput{DeploymentID: "d", ArtifactURI: "hf://m", GPUIndices: []int{0}, HostPort: 18006})
	if pv.ReadinessTimeout != 0 {
		t.Errorf("vllm ReadinessTimeout = %v, want 0 (global default)", pv.ReadinessTimeout)
	}
}
