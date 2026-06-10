package recipes

import (
	"fmt"

	"github.com/inferia/inferia-worker/internal/config/vllm"
)

func init() {
	multiRegistry["vllm-prefill-decode"] = vllmPrefillDecodeRecipe{
		image:     "docker.io/vllm/vllm-openai:v0.22.1",
		port:      8000,
		readyPath: "/health",
	}
}

// vllmPrefillDecodeRecipe implements MultiContainerBuilder for disagg
// prefill/decode deployments with Mooncake shared KV cache.
type vllmPrefillDecodeRecipe struct {
	image     string
	port      int
	readyPath string
}

func (r vllmPrefillDecodeRecipe) BuildDeploymentPlan(in BuildInput) (DeploymentPlan, error) {
	if err := validate(in); err != nil {
		return DeploymentPlan{}, err
	}
	if err := requireGPU(in); err != nil {
		return DeploymentPlan{}, err
	}

	prefillReplicas := in.PrefillReplicas
	if prefillReplicas < 1 {
		prefillReplicas = 1
	}
	decodeReplicas := in.DecodeReplicas
	if decodeReplicas < 1 {
		decodeReplicas = 1
	}

	model := stripScheme(in.ArtifactURI)
	cfg := sanitiseConfig(in.Config)

	// ----- GPU-aware default flags -----
	envDefaults := map[string]string{
		"CUDA_MODULE_LOADING": "LAZY",
	}
	gpuCfg, gpuEnv := vllm.GPUOptimalConfig(in.GPUName, in.GPUMemoryMiB, len(in.GPUIndices))
	for k, v := range gpuCfg {
		if _, ok := cfg[k]; !ok {
			cfg[k] = v
		}
	}
	for k, v := range gpuEnv {
		envDefaults[k] = v
	}
	envDefaults["LD_LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib64:/usr/local/cuda/lib64"

	baseCmd := []string{
		model,
		"--served-model-name", model,
		"--host", "0.0.0.0",
		"--port", fmt.Sprintf("%d", r.port),
	}
	for _, k := range []string{
		"tensor_parallel_size", "pipeline_parallel_size",
		"dtype", "max_model_len", "max_num_seqs",
		"gpu_memory_utilization", "quantization",
		"max_batch_size", "max_input_length", "max_total_tokens",
		"kv_cache_dtype", "max_num_batched_tokens",
	} {
		if v, ok := cfg[k]; ok {
			baseCmd = append(baseCmd, dashed(k), cliArg(v))
		}
	}
	if v, ok := cfg["enforce_eager"].(bool); ok && v {
		baseCmd = append(baseCmd, "--enforce-eager")
	}
	if v, ok := cfg["enable_prefix_caching"].(bool); ok && v {
		baseCmd = append(baseCmd, "--enable-prefix-caching")
	}
	if v, ok := cfg["trust_remote_code"].(bool); ok && v {
		baseCmd = append(baseCmd, "--trust-remote-code")
	}

	var planMounts []Mount
	var planEntrypoint []string
	if vllm.MooncakeEnabled() {
		planMounts = []Mount{{
			Type:     "volume",
			Source:   vllm.MooncakeConfigVolume(),
			Target:   vllm.MooncakeConfigMountPath(),
			ReadOnly: true,
		}}
		planEntrypoint = vllm.MooncakeEntrypoint()
	}

	env := mergeEnv(in.Env, envDefaults)

	// --- prefill replicas ---
	prefills := make([]ContainerPlan, prefillReplicas)
	for i := range prefills {
		pCmd := make([]string, len(baseCmd))
		copy(pCmd, baseCmd)
		pEnv := make(map[string]string, len(env))
		for k, v := range env {
			pEnv[k] = v
		}
		if vllm.MooncakeEnabled() {
			vllm.ApplyMooncakePrefillFlags(cfg, pEnv, &pCmd)
		}
		prefills[i] = ContainerPlan{
			Image:         r.image,
			Cmd:           pCmd,
			Entrypoint:    planEntrypoint,
			Env:           pEnv,
			Mounts:        planMounts,
			ContainerPort: r.port,
			GPUIndices:    in.GPUIndices,
			ReadyPath:     r.readyPath,
			Role:          KvRoleProducer,
			ReplicaIdx:    i,
		}
	}

	// --- decode replicas ---
	decodes := make([]ContainerPlan, decodeReplicas)
	for i := range decodes {
		dCmd := make([]string, len(baseCmd))
		copy(dCmd, baseCmd)
		dEnv := make(map[string]string, len(env))
		for k, v := range env {
			dEnv[k] = v
		}
		if vllm.MooncakeEnabled() {
			vllm.ApplyMooncakeDecodeFlags(cfg, dEnv, &dCmd)
		}
		decodes[i] = ContainerPlan{
			Image:         r.image,
			Cmd:           dCmd,
			Entrypoint:    planEntrypoint,
			Env:           dEnv,
			Mounts:        planMounts,
			ContainerPort: r.port,
			GPUIndices:    in.GPUIndices,
			ReadyPath:     r.readyPath,
			Role:          KvRoleConsumer,
			ReplicaIdx:    i,
		}
	}

	return DeploymentPlan{
		DeploymentID: in.DeploymentID,
		Model:        model,
		Prefill:      prefills,
		Decode:       decodes,
	}, nil
}
