// Package vllm provides GPU-aware default configuration for vLLM inference
// engine. It maps GPU model names to optimal vLLM parameters (dtype,
// kv_cache_dtype, quantization, memory utilization, context length, batch
// sizing).
//
// GPU identification data (name, VRAM) is expected from the caller — typically
// telemetry.ReadGPU() wired through main.go → dispatcher → BuildInput — so
// this package has no I/O dependency on nvidia-smi.
package vllm

import (
	"sort"
	"strconv"
	"strings"
)

// GPUProfile describes a detected GPU on the host.
type GPUProfile struct {
	Name      string
	MemoryMiB uint64
	SMVersion int
}

// sizing holds the per-GPU scaling parameters for a given GPU count.
// max_num_batched_tokens is capped at 16384 across all profiles — larger
// values statically reserve activation memory and cannibalise KV cache space.
type sizing struct {
	gmu float64
	mml int
	mns int
	mbt int
}

// ---------------------------------------------------------------------------
// SM lookup table
// ---------------------------------------------------------------------------

var smTable = []struct {
	prefix string
	sm     int
}{
	{"Tesla T4", 75},
	{"Tesla V100", 70},
	{"NVIDIA A10G", 86},
	{"NVIDIA L4", 89},
	{"NVIDIA L40S", 89},
	{"NVIDIA A100", 80},
	{"NVIDIA H100", 90},
	{"NVIDIA H200", 90},
	{"NVIDIA GH200", 90},
}

// SMForGPU returns the compute capability (SM version) for a known GPU name.
// Returns 0 for unknown GPUs.
func SMForGPU(name string) int {
	for _, e := range smTable {
		if strings.HasPrefix(name, e.prefix) {
			return e.sm
		}
	}
	return 0
}

// ---------------------------------------------------------------------------
// GPU profile table — per-GPU family default config + sizing
// ---------------------------------------------------------------------------

// profile holds the SM-dependent config defaults.
type profile struct {
	dtype   string
	kvDtype string
	quant   string // "" means no default
	eager   bool
	env     map[string]string
}

// gpuProfiles maps GPU name prefix → SM-dependent config.
var gpuProfiles = map[string]profile{
	"Tesla T4":   {dtype: "float16", kvDtype: "int8", eager: true, env: map[string]string{"VLLM_ATTENTION_BACKEND": "XFORMERS"}},
	"Tesla V100": {dtype: "float16", kvDtype: "int8", eager: true, env: map[string]string{"VLLM_ATTENTION_BACKEND": "XFORMERS"}},
	"NVIDIA A10G": {dtype: "bfloat16", kvDtype: "auto"},
	"NVIDIA L4":   {dtype: "bfloat16", kvDtype: "auto", quant: "fp8"},
	"NVIDIA L40S": {dtype: "bfloat16", kvDtype: "auto", quant: "fp8"},
	"NVIDIA A100": {dtype: "bfloat16", kvDtype: "auto"},
	"NVIDIA H100": {dtype: "bfloat16", kvDtype: "fp8", quant: "fp8"},
	"NVIDIA H200": {dtype: "bfloat16", kvDtype: "fp8", quant: "fp8"},
	"NVIDIA GH200": {dtype: "bfloat16", kvDtype: "fp8", quant: "fp8"},
}

// sizingTable maps (GPU prefix, numGPUs) → sizing. Key "prefix" alone is the
// single-GPU default. Key "prefix:N" overrides for exactly N GPUs.
//
// max_num_batched_tokens (mbt) is capped at 16384 to avoid wasting GPU memory
// on activation pre-allocation. See vLLM issue #2492 and
// docs.vllm.ai/configuration/optimization for details.
var sizingTable = map[string]sizing{
	// --- single GPU ---
	"Tesla T4":              {gmu: 0.92, mml: 4096, mns: 16, mbt: 4096},
	"Tesla V100":            {gmu: 0.90, mml: 3072, mns: 12, mbt: 3072},
	"NVIDIA A10G":           {gmu: 0.90, mml: 8192, mns: 32, mbt: 8192},
	"NVIDIA L4":             {gmu: 0.92, mml: 12288, mns: 48, mbt: 12288},
	"NVIDIA L40S":           {gmu: 0.90, mml: 16384, mns: 64, mbt: 16384},
	"NVIDIA A100":           {gmu: 0.92, mml: 8192, mns: 32, mbt: 8192},
	"NVIDIA H100":           {gmu: 0.95, mml: 65536, mns: 512, mbt: 16384},
	"NVIDIA H200":           {gmu: 0.93, mml: 65536, mns: 512, mbt: 16384},
	"NVIDIA GH200":          {gmu: 0.93, mml: 65536, mns: 512, mbt: 16384},

	// --- multi-GPU overrides (key = "prefix:N") ---
	"NVIDIA A10G:4":         {gmu: 0.90, mml: 8192, mns: 64, mbt: 16384},
	"NVIDIA A10G:8":         {gmu: 0.88, mml: 16384, mns: 128, mbt: 16384},
	"NVIDIA L4:4":           {gmu: 0.92, mml: 12288, mns: 64, mbt: 16384},
	"NVIDIA A100:8":         {gmu: 0.92, mml: 16384, mns: 128, mbt: 16384},
	"NVIDIA H100:8":         {gmu: 0.95, mml: 65536, mns: 512, mbt: 16384},
	"NVIDIA H200:8":         {gmu: 0.93, mml: 65536, mns: 512, mbt: 16384},
}

// A100 VRAM-aware sizing overrides (keyed on "prefix:numGPUs:vramMiB").
// Prefix must match the full gpuProfiles key, e.g. "NVIDIA A100".
// A100-PCIe-40GB vs A100-SXM4-80GB have very different capacities.
var sizingByVRAM = map[string]sizing{
	"NVIDIA A100:8:40960": {gmu: 0.92, mml: 16384, mns: 128, mbt: 16384},
	"NVIDIA A100:8:81920": {gmu: 0.92, mml: 32768, mns: 256, mbt: 16384},
	"NVIDIA A100:1:40960": {gmu: 0.92, mml: 16384, mns: 64, mbt: 16384},
	"NVIDIA A100:1:81920": {gmu: 0.92, mml: 32768, mns: 128, mbt: 16384},
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

// GPUOptimalConfig returns the recommended vLLM config and env vars for the
// given GPU name, VRAM, and GPU count. The caller is expected to layer user
// config overrides on top (user values win).
//
// When name is empty or the GPU is unknown a safe fallback is returned.
func GPUOptimalConfig(name string, memoryMiB uint64, numGPUs int) (cfg map[string]any, env map[string]string) {
	cfg = map[string]any{}
	// NOTE: do NOT default VLLM_USE_FASTOKENS=1 here. The pinned
	// vllm/vllm-openai image does not bundle the optional `fastokens`
	// package, so setting it makes vLLM abort at engine init with
	// "ModuleNotFoundError: No module named 'fastokens' / The 'fastokens'
	// package (>= 0.2.0) is required when VLLM_USE_FASTOKENS=1" — the
	// container starts then immediately dies, so the deploy never becomes
	// ready. Only set env the shipped image supports; a user who builds a
	// fastokens-enabled image can opt in via the deploy's own env.
	env = map[string]string{}

	if name == "" {
		cfg["gpu_memory_utilization"] = 0.90
		cfg["max_model_len"] = 4096
		cfg["max_num_seqs"] = 32
		cfg["max_num_batched_tokens"] = 8192
		cfg["enable_prefix_caching"] = true
		return cfg, env
	}

	sm := SMForGPU(name)

	// 1. SM-dependent config from profile table
	if p, ok := lookupProfile(name); ok {
		if p.dtype != "" {
			cfg["dtype"] = p.dtype
		}
		if p.kvDtype != "" {
			cfg["kv_cache_dtype"] = p.kvDtype
		}
		if p.quant != "" {
			cfg["quantization"] = p.quant
		}
		if p.eager {
			cfg["enforce_eager"] = true
		}
		for k, v := range p.env {
			env[k] = v
		}
	}

	// 2. Sizing
	gpu := GPUProfile{Name: name, MemoryMiB: memoryMiB, SMVersion: sm}
	s := lookupSizing(gpu, numGPUs)
	cfg["gpu_memory_utilization"] = s.gmu
	cfg["max_model_len"] = s.mml
	cfg["max_num_seqs"] = s.mns
	cfg["max_num_batched_tokens"] = s.mbt

	// 3. Always-on
	cfg["enable_prefix_caching"] = true

	// 4. Multi-GPU
	if numGPUs > 1 {
		cfg["tensor_parallel_size"] = numGPUs
	}

	return cfg, env
}

// ---------------------------------------------------------------------------
// internal helpers
// ---------------------------------------------------------------------------

func lookupProfile(name string) (profile, bool) {
	for _, entry := range sortedProfileKeys {
		if strings.HasPrefix(name, entry) {
			p, ok := gpuProfiles[entry]
			return p, ok
		}
	}
	return profile{}, false
}

func lookupSizing(gpu GPUProfile, numGPUs int) sizing {
	name := gpu.Name
	vram := gpu.MemoryMiB

	// 1. Find the matching profile prefix.
	prefix := ""
	for _, p := range sortedProfileKeys {
		if strings.HasPrefix(name, p) {
			prefix = p
			break
		}
	}
	if prefix == "" {
		return sizing{gmu: 0.90, mml: 4096, mns: 32, mbt: 8192}
	}

	// 2. VRAM-specific match by prefix (e.g. A100:1:40960).
	if vram > 0 {
		key := prefix + ":" + strconv.Itoa(numGPUs) + ":" + strconv.FormatUint(vram, 10)
		if s, ok := sizingByVRAM[key]; ok {
			return s
		}
	}

	// 3. Multi-GPU override (e.g. A10G:4).
	if numGPUs > 1 {
		key := prefix + ":" + strconv.Itoa(numGPUs)
		if s, ok := sizingTable[key]; ok {
			return s
		}
	}

	// 4. Single-GPU default.
	if s, ok := sizingTable[prefix]; ok {
		return s
	}

	return sizing{gmu: 0.90, mml: 4096, mns: 32, mbt: 8192}
}

// sortedProfileKeys is the profile prefix list ordered longest-first so
// "NVIDIA A100" doesn't match before "NVIDIA A10G".
var sortedProfileKeys []string

func init() {
	sortedProfileKeys = make([]string, 0, len(gpuProfiles))
	for k := range gpuProfiles {
		sortedProfileKeys = append(sortedProfileKeys, k)
	}
	sort.Slice(sortedProfileKeys, func(i, j int) bool {
		return len(sortedProfileKeys[j]) < len(sortedProfileKeys[i])
	})
}
