package vllm_test

import (
	"testing"

	"github.com/inferia/inferia-worker/internal/config/vllm"
)

// ---------------------------------------------------------------------------
// SM lookup
// ---------------------------------------------------------------------------

func TestSMForGPU_Known(t *testing.T) {
	tests := []struct {
		name string
		want int
	}{
		{"Tesla T4", 75},
		{"Tesla V100-SXM2-32GB", 70},
		{"NVIDIA A10G", 86},
		{"NVIDIA L4", 89},
		{"NVIDIA L40S", 89},
		{"NVIDIA A100-SXM4-80GB", 80},
		{"NVIDIA A100-PCIE-40GB", 80},
		{"NVIDIA H100 80GB HBM3", 90},
		{"NVIDIA H200", 90},
		{"NVIDIA GH200", 90},
	}
	for _, tc := range tests {
		if got := vllm.SMForGPU(tc.name); got != tc.want {
			t.Errorf("SMForGPU(%q) = %d, want %d", tc.name, got, tc.want)
		}
	}
}

func TestSMForGPU_Unknown(t *testing.T) {
	if got := vllm.SMForGPU("AMD Instinct MI250"); got != 0 {
		t.Errorf("expected 0 for unknown GPU, got %d", got)
	}
	if got := vllm.SMForGPU(""); got != 0 {
		t.Errorf("expected 0 for empty name, got %d", got)
	}
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

type configCheck struct {
	key   string
	value any
}

func checkConfig(t *testing.T, cfg map[string]any, checks []configCheck) {
	t.Helper()
	for _, c := range checks {
		got, ok := cfg[c.key]
		if !ok {
			t.Errorf("cfg[%q] missing, want %v", c.key, c.value)
			continue
		}
		if got != c.value {
			t.Errorf("cfg[%q] = %v (type %T), want %v (type %T)",
				c.key, got, got, c.value, c.value)
		}
	}
}

func checkNotSet(t *testing.T, cfg map[string]any, keys ...string) {
	t.Helper()
	for _, k := range keys {
		if _, ok := cfg[k]; ok {
			t.Errorf("cfg[%q] should not be set, got %v", k, cfg[k])
		}
	}
}

// ---------------------------------------------------------------------------
// single GPU config
// ---------------------------------------------------------------------------

func TestGPUOptimalConfig_T4(t *testing.T) {
	cfg, env := vllm.GPUOptimalConfig("Tesla T4", 16384, 1)

	checkConfig(t, cfg, []configCheck{
		{"dtype", "float16"},
		{"kv_cache_dtype", "int8"},
		{"enforce_eager", true},
		{"gpu_memory_utilization", 0.92},
		{"max_model_len", 4096},
		{"max_num_seqs", 16},
		{"max_num_batched_tokens", 4096},
		{"enable_prefix_caching", true},
	})
	if env["VLLM_ATTENTION_BACKEND"] != "XFORMERS" {
		t.Errorf("VLLM_ATTENTION_BACKEND = %q, want XFORMERS", env["VLLM_ATTENTION_BACKEND"])
	}
	// VLLM_USE_FASTOKENS must NOT be set: the shipped vllm-openai image lacks
	// the fastokens package, so setting it crashes vLLM at engine init.
	if _, ok := env["VLLM_USE_FASTOKENS"]; ok {
		t.Errorf("VLLM_USE_FASTOKENS must not be set (image has no fastokens)")
	}
	checkNotSet(t, cfg, "tensor_parallel_size")
}

func TestGPUOptimalConfig_V100(t *testing.T) {
	cfg, env := vllm.GPUOptimalConfig("Tesla V100-SXM2-32GB", 32768, 1)

	checkConfig(t, cfg, []configCheck{
		{"dtype", "float16"},
		{"kv_cache_dtype", "int8"},
		{"enforce_eager", true},
		{"gpu_memory_utilization", 0.90},
		{"max_model_len", 3072},
		{"max_num_seqs", 12},
		{"max_num_batched_tokens", 3072},
	})
	if env["VLLM_ATTENTION_BACKEND"] != "XFORMERS" {
		t.Errorf("VLLM_ATTENTION_BACKEND = %q", env["VLLM_ATTENTION_BACKEND"])
	}
}

func TestGPUOptimalConfig_A10G(t *testing.T) {
	cfg, _ := vllm.GPUOptimalConfig("NVIDIA A10G", 24576, 1)

	checkConfig(t, cfg, []configCheck{
		{"dtype", "bfloat16"},
		{"kv_cache_dtype", "auto"},
		{"gpu_memory_utilization", 0.90},
		{"max_model_len", 8192},
		{"max_num_seqs", 32},
		{"max_num_batched_tokens", 8192},
		{"enable_prefix_caching", true},
	})
	checkNotSet(t, cfg, "enforce_eager", "quantization")
}

func TestGPUOptimalConfig_L4(t *testing.T) {
	cfg, _ := vllm.GPUOptimalConfig("NVIDIA L4", 24576, 1)

	checkConfig(t, cfg, []configCheck{
		{"dtype", "bfloat16"},
		{"kv_cache_dtype", "auto"},
		{"quantization", "fp8"},
		{"gpu_memory_utilization", 0.92},
		{"max_model_len", 12288},
		{"max_num_seqs", 48},
		{"max_num_batched_tokens", 12288},
	})
}

func TestGPUOptimalConfig_L40S(t *testing.T) {
	cfg, _ := vllm.GPUOptimalConfig("NVIDIA L40S", 49152, 1)

	checkConfig(t, cfg, []configCheck{
		{"dtype", "bfloat16"},
		{"kv_cache_dtype", "auto"},
		{"quantization", "fp8"},
		{"gpu_memory_utilization", 0.90},
		{"max_model_len", 16384},
		{"max_num_seqs", 64},
		{"max_num_batched_tokens", 16384},
	})
}

func TestGPUOptimalConfig_A100_40GB(t *testing.T) {
	cfg, _ := vllm.GPUOptimalConfig("NVIDIA A100-SXM4-40GB", 40960, 1)

	checkConfig(t, cfg, []configCheck{
		{"dtype", "bfloat16"},
		{"kv_cache_dtype", "auto"},
		{"gpu_memory_utilization", 0.92},
		{"max_model_len", 16384},
		{"max_num_seqs", 64},
		{"max_num_batched_tokens", 16384},
	})
}

func TestGPUOptimalConfig_A100_80GB(t *testing.T) {
	cfg, _ := vllm.GPUOptimalConfig("NVIDIA A100-SXM4-80GB", 81920, 1)

	checkConfig(t, cfg, []configCheck{
		{"dtype", "bfloat16"},
		{"kv_cache_dtype", "auto"},
		{"gpu_memory_utilization", 0.92},
		{"max_model_len", 32768},
		{"max_num_seqs", 128},
		{"max_num_batched_tokens", 16384},
	})
}

func TestGPUOptimalConfig_H100(t *testing.T) {
	cfg, _ := vllm.GPUOptimalConfig("NVIDIA H100 80GB HBM3", 81920, 1)

	checkConfig(t, cfg, []configCheck{
		{"dtype", "bfloat16"},
		{"kv_cache_dtype", "fp8"},
		{"quantization", "fp8"},
		{"gpu_memory_utilization", 0.95},
		{"max_model_len", 65536},
		{"max_num_seqs", 512},
		{"max_num_batched_tokens", 16384},
	})
}

func TestGPUOptimalConfig_H200(t *testing.T) {
	cfg, _ := vllm.GPUOptimalConfig("NVIDIA H200", 144384, 1)

	checkConfig(t, cfg, []configCheck{
		{"dtype", "bfloat16"},
		{"kv_cache_dtype", "fp8"},
		{"quantization", "fp8"},
		{"gpu_memory_utilization", 0.93},
		{"max_model_len", 65536},
		{"max_num_seqs", 512},
		{"max_num_batched_tokens", 16384},
	})
}

func TestGPUOptimalConfig_UnknownGPU(t *testing.T) {
	cfg, env := vllm.GPUOptimalConfig("AMD Instinct MI250", 131072, 1)

	checkNotSet(t, cfg, "dtype", "kv_cache_dtype")
	checkConfig(t, cfg, []configCheck{
		{"gpu_memory_utilization", 0.90},
		{"max_model_len", 4096},
		{"max_num_seqs", 32},
		{"max_num_batched_tokens", 8192},
		{"enable_prefix_caching", true},
	})
	if _, ok := env["VLLM_USE_FASTOKENS"]; ok {
		t.Errorf("VLLM_USE_FASTOKENS must not be set (image has no fastokens)")
	}
}

func TestGPUOptimalConfig_EmptyName(t *testing.T) {
	cfg, env := vllm.GPUOptimalConfig("", 0, 1)

	checkConfig(t, cfg, []configCheck{
		{"gpu_memory_utilization", 0.90},
		{"max_model_len", 4096},
		{"max_num_seqs", 32},
		{"max_num_batched_tokens", 8192},
		{"enable_prefix_caching", true},
	})
	checkNotSet(t, cfg, "dtype", "kv_cache_dtype", "quantization", "tensor_parallel_size")
	if _, ok := env["VLLM_USE_FASTOKENS"]; ok {
		t.Errorf("VLLM_USE_FASTOKENS must not be set (image has no fastokens)")
	}
}

// ---------------------------------------------------------------------------
// multi-GPU
// ---------------------------------------------------------------------------

func TestGPUOptimalConfig_A10G_Multi(t *testing.T) {
	t.Run("x4", func(t *testing.T) {
		cfg, _ := vllm.GPUOptimalConfig("NVIDIA A10G", 24576, 4)
		checkConfig(t, cfg, []configCheck{
			{"tensor_parallel_size", 4},
			{"gpu_memory_utilization", 0.90},
			{"max_model_len", 8192},
			{"max_num_seqs", 64},
			{"max_num_batched_tokens", 16384},
		})
	})
	t.Run("x8", func(t *testing.T) {
		cfg, _ := vllm.GPUOptimalConfig("NVIDIA A10G", 24576, 8)
		checkConfig(t, cfg, []configCheck{
			{"tensor_parallel_size", 8},
			{"gpu_memory_utilization", 0.88},
			{"max_model_len", 16384},
			{"max_num_seqs", 128},
			{"max_num_batched_tokens", 16384},
		})
	})
}

func TestGPUOptimalConfig_L4_Multi(t *testing.T) {
	cfg, _ := vllm.GPUOptimalConfig("NVIDIA L4", 24576, 4)
	checkConfig(t, cfg, []configCheck{
		{"tensor_parallel_size", 4},
		{"gpu_memory_utilization", 0.92},
		{"max_model_len", 12288},
		{"max_num_seqs", 64},
		{"max_num_batched_tokens", 16384},
	})
}

func TestGPUOptimalConfig_A100_Multi(t *testing.T) {
	t.Run("40GB x8", func(t *testing.T) {
		cfg, _ := vllm.GPUOptimalConfig("NVIDIA A100-SXM4-40GB", 40960, 8)
		checkConfig(t, cfg, []configCheck{
			{"tensor_parallel_size", 8},
			{"gpu_memory_utilization", 0.92},
			{"max_model_len", 16384},
			{"max_num_seqs", 128},
			{"max_num_batched_tokens", 16384},
		})
	})
	t.Run("80GB x8", func(t *testing.T) {
		cfg, _ := vllm.GPUOptimalConfig("NVIDIA A100-SXM4-80GB", 81920, 8)
		checkConfig(t, cfg, []configCheck{
			{"tensor_parallel_size", 8},
			{"gpu_memory_utilization", 0.92},
			{"max_model_len", 32768},
			{"max_num_seqs", 256},
			{"max_num_batched_tokens", 16384},
		})
	})
}

func TestGPUOptimalConfig_H100_Multi(t *testing.T) {
	cfg, _ := vllm.GPUOptimalConfig("NVIDIA H100 80GB HBM3", 81920, 8)
	checkConfig(t, cfg, []configCheck{
		{"tensor_parallel_size", 8},
		{"dtype", "bfloat16"},
		{"kv_cache_dtype", "fp8"},
		{"quantization", "fp8"},
		{"gpu_memory_utilization", 0.95},
		{"max_model_len", 65536},
		{"max_num_seqs", 512},
		{"max_num_batched_tokens", 16384},
	})
}

func TestGPUOptimalConfig_H200_Multi(t *testing.T) {
	cfg, _ := vllm.GPUOptimalConfig("NVIDIA H200", 144384, 8)
	checkConfig(t, cfg, []configCheck{
		{"tensor_parallel_size", 8},
		{"dtype", "bfloat16"},
		{"kv_cache_dtype", "fp8"},
		{"quantization", "fp8"},
		{"gpu_memory_utilization", 0.93},
		{"max_model_len", 65536},
		{"max_num_seqs", 512},
		{"max_num_batched_tokens", 16384},
	})
}
