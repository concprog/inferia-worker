package recipes

import (
	"fmt"

	omnicfg "github.com/inferia/inferia-worker/internal/config/vllm/omni"
)

// vllmOmniRecipe builds an invocation of vllm/vllm-omni for diffusion models
// (text-to-image, text-to-video). LLM-only flags are intentionally absent:
// max_model_len, max_num_seqs, max_num_batched_tokens, kv_cache_dtype,
// enable_prefix_caching, quantization, and Mooncake disaggregation.
type vllmOmniRecipe struct {
	image     string
	port      int
	readyPath string
}

func (r vllmOmniRecipe) BuildPlan(in BuildInput) (Plan, error) {
	if err := validate(in); err != nil {
		return Plan{}, err
	}
	if err := requireGPU(in); err != nil {
		return Plan{}, err
	}
	model := stripScheme(in.ArtifactURI)
	cfg := sanitiseConfig(in.Config)

	// Diffusion-aware GPU defaults: dtype + gpu_memory_utilization +
	// tensor_parallel_size when >1 GPU. No LLM sizing params.
	defaults, envDefaults := omnicfg.GPUDefaults(in.GPUName, len(in.GPUIndices))
	for k, v := range defaults {
		if _, ok := cfg[k]; !ok {
			cfg[k] = v
		}
	}
	envDefaults["CUDA_MODULE_LOADING"] = "LAZY"

	cmd := []string{
		"vllm", "serve", model,
		"--omni",
		"--served-model-name", model,
		"--host", "0.0.0.0",
		"--port", fmt.Sprintf("%d", r.port),
	}

	// Scalar flags valid for diffusion serving.
	for _, k := range []string{
		"tensor_parallel_size",
		"pipeline_parallel_size",
		"dtype",
		"gpu_memory_utilization",
		"quantization",
		"ignored_layers",
		"usp",
		"ring",
		"vae_patch_parallel_size",
		"hsdp_shard_size",
		"flow_shift",
		"boundary_ratio",
		"cache_backend",
	} {
		if v, ok := cfg[k]; ok {
			cmd = append(cmd, dashed(k), cliArg(v))
		}
	}

	// Boolean flags.
	if v, ok := cfg["enforce_eager"].(bool); ok && v {
		cmd = append(cmd, "--enforce-eager")
	}
	if v, ok := cfg["vae_use_slicing"].(bool); ok && v {
		cmd = append(cmd, "--vae-use-slicing")
	}
	if v, ok := cfg["vae_use_tiling"].(bool); ok && v {
		cmd = append(cmd, "--vae-use-tiling")
	}
	if v, ok := cfg["use_hsdp"].(bool); ok && v {
		cmd = append(cmd, "--use-hsdp")
	}
	if v, ok := cfg["trust_remote_code"].(bool); ok && v {
		cmd = append(cmd, "--trust-remote-code")
	}

	return Plan{
		Image:         r.image,
		ContainerName: containerName("inferia-vllm-omni", in.DeploymentID),
		Cmd:           cmd,
		Env:           mergeEnv(in.Env, envDefaults),
		ContainerPort: r.port,
		HostPort:      in.HostPort,
		GPUIndices:    in.GPUIndices,
		ReadyPath:     r.readyPath,
	}, nil
}
