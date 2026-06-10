// Package vllm provides GPU-aware optimal configuration for vLLM model
// servers. The mooncake sub-package adds shared-KV-cache support via the
// Mooncake distributed store.
package vllm

import "os"

// MooncakeEnabled reports whether the worker should inject mooncake KV-store
// flags into every vLLM deployment it launches. Set MOONCAKE_KV_ENABLED=true
// on the worker container.
func MooncakeEnabled() bool {
	return os.Getenv("MOONCAKE_KV_ENABLED") == "true"
}

// ApplyMooncakeFlags mutates cfg, env, and cmd in-place so the resulting
// vLLM invocation uses MooncakeStoreConnector in kv_both (single-node KV
// cache offloading) mode. This is safe to call only when MooncakeEnabled()
// returns true.
func ApplyMooncakeFlags(cfg map[string]any, env map[string]string, cmd *[]string) {
	cfg["enable_prefix_caching"] = true
	env["MOONCAKE_CONFIG_PATH"] = MooncakeConfigMountPath() + "/mooncake_config.json"
	env["PYTHONHASHSEED"] = "0"
	*cmd = append(*cmd,
		"--kv-transfer-config",
		`{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both"}`,
	)
}

// MooncakeEntrypoint returns an entrypoint override that installs the
// mooncake-transfer-engine pip package at container startup, then execs
// the original command. This avoids building a derived vLLM image.
func MooncakeEntrypoint() []string {
	return []string{
		"/bin/sh", "-c",
		"pip install --no-cache-dir mooncake-transfer-engine && exec \"$@\"",
		"entrypoint",
	}
}

// MooncakeConfigVolume returns the Docker named volume that carries the
// mooncake_config.json written by the mooncake-master container. Override
// via MOONCAKE_CONFIG_VOLUME (default "mooncake-config").
func MooncakeConfigVolume() string {
	if v := os.Getenv("MOONCAKE_CONFIG_VOLUME"); v != "" {
		return v
	}
	return "mooncake-config"
}

// MooncakeConfigMountPath returns the path inside the vLLM container where
// the mooncake config volume is mounted. Override via MOONCAKE_CONFIG_MOUNT_PATH
// (default "/etc/mooncake").
func MooncakeConfigMountPath() string {
	if v := os.Getenv("MOONCAKE_CONFIG_MOUNT_PATH"); v != "" {
		return v
	}
	return "/etc/mooncake"
}
