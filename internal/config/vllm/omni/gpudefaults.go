// Package omni provides GPU- and model-family-aware defaults for vllm-omni
// diffusion serving. No KV-cache, sequence-length, or batching params are set
// here — those LLM concepts don't exist in diffusion pipelines.
package omni

import "strings"

// modelFamily groups known HF model IDs by inference characteristics.
type modelFamily int

const (
	familyUnknown     modelFamily = iota
	familyQwenImage               // Qwen-Image, Qwen-Image-2512 (T2I)
	familyQwenImageEdit            // Qwen-Image-Edit / -2509 / -2511 / -Layered
	familyWan                     // Wan2.1 / Wan2.2 T2V, I2V, TI2V, VACE, S2V
	familyLTX                     // Lightricks LTX-2, LTX-2.3
	familyFLUX                    // FLUX.1-dev/schnell, FLUX.2-klein, FLUX.1-Kontext
	familyHunyuan                 // HunyuanImage-3.0
	familyZImage                  // Tongyi-MAI Z-Image-Turbo
	familySDXL                    // stable-diffusion-xl (small, no quant needed)
)

// Defaults returns dtype, gpu_memory_utilization, tensor_parallel_size, and
// model-family-specific flags (quantization, video acceleration, etc.).
//
// FP8 quantization is enabled by default on Ampere+ GPUs for validated model
// families (Qwen-Image, FLUX, HunyuanImage, Z-Image). Pass quantization:"none"
// in the deployment config to explicitly disable it.
//
// User config is applied by the caller before Defaults — keys already present
// in cfg are never overwritten.
func Defaults(gpuName string, gpuMemMiB uint64, numGPUs int, modelID string) (cfg map[string]any, env map[string]string) {
	cfg = map[string]any{}
	env = map[string]string{}

	if isOlderGPU(gpuName) {
		cfg["dtype"] = "float16"
	} else {
		cfg["dtype"] = "bfloat16"
	}
	cfg["gpu_memory_utilization"] = gpuMemUtil(gpuName)
	if numGPUs > 1 {
		cfg["tensor_parallel_size"] = numGPUs
	}

	family := detectFamily(modelID)
	totalVRAMMiB := gpuMemMiB * uint64(numGPUs)
	ampere := isAmpereOrNewer(gpuName)
	applyFamilyDefaults(cfg, family, ampere, totalVRAMMiB)

	return cfg, env
}

// applyFamilyDefaults sets quantization and acceleration flags for a model family.
func applyFamilyDefaults(cfg map[string]any, family modelFamily, ampere bool, totalVRAMMiB uint64) {
	switch family {
	case familyQwenImage:
		// FP8 validated; ~48 GB needed for BF16 so enable whenever Ampere+ or tight on VRAM.
		// img_mlp layers are quality-sensitive and must stay in BF16.
		if ampere || totalVRAMMiB < 49152 {
			cfg["quantization"] = "fp8"
			cfg["ignored_layers"] = "img_mlp"
		}

	case familyQwenImageEdit:
		// Edit variants support FP8 (validated); img_mlp skip not documented for edit models.
		if ampere {
			cfg["quantization"] = "fp8"
		}

	case familyFLUX:
		// FLUX FP8 validated on Ampere+; no sensitive-layer exclusions needed.
		if ampere {
			cfg["quantization"] = "fp8"
		}

	case familyHunyuan:
		// HunyuanImage-3.0 FP8 validated on Ampere+.
		if ampere {
			cfg["quantization"] = "fp8"
		}

	case familyZImage:
		// Z-Image FP8 validated, all layers.
		if ampere {
			cfg["quantization"] = "fp8"
		}

	case familyWan:
		// Wan2.2 FP8 is "not validated" per docs — no auto-quant.
		// enforce_eager avoids torch.compile first-request latency on video models.
		cfg["enforce_eager"] = true

	case familyLTX:
		// LTX cache_dit gives ~2× speedup; enforce_eager skips torch.compile warmup.
		cfg["enforce_eager"] = true
		cfg["cache_backend"] = "cache_dit"
	}
}

// detectFamily maps a HF model ID (after scheme strip) to a known model family.
func detectFamily(modelID string) modelFamily {
	id := strings.ToLower(modelID)
	switch {
	// Qwen-Image-Edit must be checked before Qwen-Image (longer prefix wins).
	case strings.Contains(id, "qwen-image-edit"),
		strings.Contains(id, "qwen-image-layered"):
		return familyQwenImageEdit
	case strings.Contains(id, "qwen-image"):
		return familyQwenImage
	case strings.Contains(id, "wan-ai/wan"),
		strings.Contains(id, "wan2."):
		return familyWan
	case strings.Contains(id, "ltx"):
		return familyLTX
	case strings.Contains(id, "flux"):
		return familyFLUX
	case strings.Contains(id, "hunyuanimage"),
		strings.Contains(id, "hunyuan-image"),
		strings.Contains(id, "tencent/hunyuan"):
		return familyHunyuan
	case strings.Contains(id, "z-image"),
		strings.Contains(id, "tongyi-mai"):
		return familyZImage
	case strings.Contains(id, "stable-diffusion-xl"),
		strings.Contains(id, "sdxl"):
		return familySDXL
	}
	return familyUnknown
}

// isAmpereOrNewer returns true for NVIDIA GPUs with SM 80+ (Ampere, Ada, Hopper).
// These support efficient FP8 GEMM; older GPUs use a weight-only fallback.
func isAmpereOrNewer(name string) bool {
	for _, prefix := range []string{
		"NVIDIA A10", "NVIDIA A30", "NVIDIA A40", "NVIDIA A100",
		"NVIDIA L4", "NVIDIA L40",
		"NVIDIA H100", "NVIDIA H200", "NVIDIA GH200",
	} {
		if strings.HasPrefix(name, prefix) {
			return true
		}
	}
	return false
}

func isOlderGPU(name string) bool {
	for _, prefix := range []string{"Tesla T4", "Tesla V100", "NVIDIA V100"} {
		if strings.HasPrefix(name, prefix) {
			return true
		}
	}
	return false
}

func gpuMemUtil(name string) float64 {
	switch {
	case strings.HasPrefix(name, "NVIDIA H100"),
		strings.HasPrefix(name, "NVIDIA H200"),
		strings.HasPrefix(name, "NVIDIA GH200"):
		return 0.95
	case strings.HasPrefix(name, "NVIDIA A100"),
		strings.HasPrefix(name, "NVIDIA L4"),
		strings.HasPrefix(name, "NVIDIA L40S"):
		return 0.92
	default:
		return 0.90
	}
}
