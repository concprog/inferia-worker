package omni

import "testing"

func TestDefaults_QwenImage_H100(t *testing.T) {
	cfg, _ := Defaults("NVIDIA H100", 81920, 1, "Qwen/Qwen-Image-2512")
	if cfg["quantization"] != "fp8" {
		t.Errorf("expected fp8 quantization on H100, got %v", cfg["quantization"])
	}
	if cfg["ignored_layers"] != "img_mlp" {
		t.Errorf("expected ignored_layers=img_mlp, got %v", cfg["ignored_layers"])
	}
	if cfg["dtype"] != "bfloat16" {
		t.Errorf("expected bfloat16 on H100, got %v", cfg["dtype"])
	}
}

func TestDefaults_QwenImage_LowVRAM(t *testing.T) {
	// T4 is not Ampere, but total VRAM < 48 GB should still trigger fp8.
	cfg, _ := Defaults("Tesla T4", 16384, 1, "Qwen/Qwen-Image-2512")
	if cfg["quantization"] != "fp8" {
		t.Errorf("expected fp8 on low-VRAM GPU, got %v", cfg["quantization"])
	}
	if cfg["dtype"] != "float16" {
		t.Errorf("expected float16 on T4, got %v", cfg["dtype"])
	}
}

func TestDefaults_QwenImage_HighVRAM_OlderGPU(t *testing.T) {
	// V100 with 4×32GB = 128 GB, not Ampere, VRAM > 48 GB → no auto-fp8.
	cfg, _ := Defaults("Tesla V100", 32768, 4, "Qwen/Qwen-Image")
	if _, ok := cfg["quantization"]; ok {
		t.Errorf("unexpected quantization set on V100 with enough VRAM: %v", cfg["quantization"])
	}
}

func TestDefaults_QwenImageEdit_Ampere(t *testing.T) {
	cfg, _ := Defaults("NVIDIA A100", 40960, 1, "Qwen/Qwen-Image-Edit-2511")
	if cfg["quantization"] != "fp8" {
		t.Errorf("expected fp8 for edit variant on A100, got %v", cfg["quantization"])
	}
	if _, ok := cfg["ignored_layers"]; ok {
		t.Errorf("edit variant should not set ignored_layers, got %v", cfg["ignored_layers"])
	}
}

func TestDefaults_Wan_NoQuant(t *testing.T) {
	cfg, _ := Defaults("NVIDIA H100", 81920, 1, "Wan-AI/Wan2.2-T2V-A14B-Diffusers")
	if _, ok := cfg["quantization"]; ok {
		t.Errorf("Wan2.2 fp8 not validated — should not auto-set quantization, got %v", cfg["quantization"])
	}
	if cfg["enforce_eager"] != true {
		t.Errorf("expected enforce_eager=true for Wan model, got %v", cfg["enforce_eager"])
	}
}

func TestDefaults_LTX(t *testing.T) {
	cfg, _ := Defaults("NVIDIA H100", 81920, 1, "Lightricks/LTX-2")
	if cfg["enforce_eager"] != true {
		t.Errorf("expected enforce_eager=true for LTX, got %v", cfg["enforce_eager"])
	}
	if cfg["cache_backend"] != "cache_dit" {
		t.Errorf("expected cache_backend=cache_dit for LTX, got %v", cfg["cache_backend"])
	}
	if _, ok := cfg["quantization"]; ok {
		t.Errorf("LTX should not auto-set quantization, got %v", cfg["quantization"])
	}
}

func TestDefaults_FLUX_Ampere(t *testing.T) {
	cfg, _ := Defaults("NVIDIA L40S", 49152, 1, "black-forest-labs/FLUX.1-dev")
	if cfg["quantization"] != "fp8" {
		t.Errorf("expected fp8 for FLUX on Ada GPU, got %v", cfg["quantization"])
	}
	if _, ok := cfg["ignored_layers"]; ok {
		t.Errorf("FLUX should not set ignored_layers, got %v", cfg["ignored_layers"])
	}
}

func TestDefaults_MultiGPU_TensorParallel(t *testing.T) {
	cfg, _ := Defaults("NVIDIA H100", 81920, 4, "Qwen/Qwen-Image-2512")
	if cfg["tensor_parallel_size"] != 4 {
		t.Errorf("expected tensor_parallel_size=4, got %v", cfg["tensor_parallel_size"])
	}
}

func TestDefaults_SingleGPU_NoTensorParallel(t *testing.T) {
	cfg, _ := Defaults("NVIDIA H100", 81920, 1, "Qwen/Qwen-Image-2512")
	if _, ok := cfg["tensor_parallel_size"]; ok {
		t.Errorf("single-GPU should not set tensor_parallel_size, got %v", cfg["tensor_parallel_size"])
	}
}

func TestDefaults_UnknownModel_NoQuant(t *testing.T) {
	cfg, _ := Defaults("NVIDIA H100", 81920, 1, "some-org/some-unknown-model")
	if _, ok := cfg["quantization"]; ok {
		t.Errorf("unknown model should not auto-set quantization, got %v", cfg["quantization"])
	}
}

func TestDetectFamily(t *testing.T) {
	cases := []struct {
		id   string
		want modelFamily
	}{
		{"Qwen/Qwen-Image-2512", familyQwenImage},
		{"Qwen/Qwen-Image", familyQwenImage},
		{"Qwen/Qwen-Image-Edit", familyQwenImageEdit},
		{"Qwen/Qwen-Image-Edit-2511", familyQwenImageEdit},
		{"Qwen/Qwen-Image-Layered", familyQwenImageEdit},
		{"Wan-AI/Wan2.2-T2V-A14B-Diffusers", familyWan},
		{"Wan-AI/Wan2.1-T2V-1.3B-Diffusers", familyWan},
		{"Lightricks/LTX-2", familyLTX},
		{"black-forest-labs/FLUX.1-dev", familyFLUX},
		{"black-forest-labs/FLUX.2-klein-4B", familyFLUX},
		{"tencent/HunyuanImage-3.0", familyHunyuan},
		{"Tongyi-MAI/Z-Image-Turbo", familyZImage},
		{"stabilityai/stable-diffusion-xl-base-1.0", familySDXL},
		{"some-org/other-model", familyUnknown},
	}
	for _, c := range cases {
		got := detectFamily(c.id)
		if got != c.want {
			t.Errorf("detectFamily(%q) = %v, want %v", c.id, got, c.want)
		}
	}
}
