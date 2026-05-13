package config

import (
	"strings"
	"testing"
)

// envMap is a deterministic substitute for os.Environ used by Load.
// Tests construct one and call LoadFrom; Load (using os.Getenv) is exercised
// by a single end-to-end test.
type envMap map[string]string

func (e envMap) Get(k string) string { return e[k] }

func baseEnv() envMap {
	return envMap{
		"CONTROL_PLANE_URL":    "https://control.example.com",
		"BOOTSTRAP_TOKEN":      "valid-bootstrap-token",
		"NODE_NAME":            "gpu-node-1",
		"POOL_ID":              "11111111-1111-1111-1111-111111111111",
		"WORKER_ADVERTISE_URL": "https://worker.example.com:8080",
		"INFERENCE_TOKEN":      "valid-inference-token-value",
	}
}

func TestLoadFrom_HappyPath(t *testing.T) {
	cfg, err := LoadFrom(baseEnv().Get)
	if err != nil {
		t.Fatalf("expected ok, got %v", err)
	}
	if cfg.ControlPlaneURL != "https://control.example.com" {
		t.Errorf("ControlPlaneURL: %q", cfg.ControlPlaneURL)
	}
	if cfg.NodeName != "gpu-node-1" {
		t.Errorf("NodeName: %q", cfg.NodeName)
	}
	if cfg.WorkerListenAddr != "0.0.0.0:8080" {
		t.Errorf("default WorkerListenAddr: %q", cfg.WorkerListenAddr)
	}
	if cfg.TokenFile != "/var/lib/inferia-worker/token" {
		t.Errorf("default TokenFile: %q", cfg.TokenFile)
	}
	if cfg.DockerHost != "unix:///var/run/docker.sock" {
		t.Errorf("default DockerHost: %q", cfg.DockerHost)
	}
	if cfg.ModelsNetwork != "inferia-models" {
		t.Errorf("default ModelsNetwork: %q", cfg.ModelsNetwork)
	}
	if cfg.LogLevel != "info" {
		t.Errorf("default LogLevel: %q", cfg.LogLevel)
	}
	if cfg.PullTimeoutSeconds != 600 {
		t.Errorf("default PullTimeoutSeconds: %d", cfg.PullTimeoutSeconds)
	}
	if cfg.ReadinessTimeoutSeconds != 180 {
		t.Errorf("default ReadinessTimeoutSeconds: %d", cfg.ReadinessTimeoutSeconds)
	}
	if cfg.HeartbeatIntervalSeconds != 5 {
		t.Errorf("default HeartbeatIntervalSeconds: %d", cfg.HeartbeatIntervalSeconds)
	}
}

func TestLoadFrom_MissingRequired(t *testing.T) {
	required := []string{
		"CONTROL_PLANE_URL", "BOOTSTRAP_TOKEN", "NODE_NAME",
		"POOL_ID", "WORKER_ADVERTISE_URL", "INFERENCE_TOKEN",
	}
	for _, key := range required {
		t.Run(key, func(t *testing.T) {
			env := baseEnv()
			delete(env, key)
			if _, err := LoadFrom(env.Get); err == nil {
				t.Fatalf("expected error when %s missing", key)
			} else if !strings.Contains(err.Error(), key) {
				t.Errorf("error should mention %s, got: %v", key, err)
			}
		})
	}
}

func TestLoadFrom_BootstrapTokenOptionalIfTokenFileExists(t *testing.T) {
	// When TOKEN_EXISTS=true is set (a worker-internal signal injected by main
	// after stat'ing the token file), BOOTSTRAP_TOKEN is allowed to be missing.
	env := baseEnv()
	delete(env, "BOOTSTRAP_TOKEN")
	env["TOKEN_FILE_PRESENT"] = "true"
	if _, err := LoadFrom(env.Get); err != nil {
		t.Errorf("expected ok when TOKEN_FILE_PRESENT=true and bootstrap missing, got %v", err)
	}
}

func TestLoadFrom_URLSchemeRejection(t *testing.T) {
	cases := []struct {
		key string
		val string
	}{
		{"CONTROL_PLANE_URL", "ftp://example.com"},
		{"CONTROL_PLANE_URL", "control.example.com"}, // missing scheme
		{"CONTROL_PLANE_URL", "javascript:alert(1)"},
		{"WORKER_ADVERTISE_URL", "ftp://worker.example.com"},
		{"WORKER_ADVERTISE_URL", "not-a-url"},
	}
	for _, tc := range cases {
		t.Run(tc.key+"="+tc.val, func(t *testing.T) {
			env := baseEnv()
			env[tc.key] = tc.val
			if _, err := LoadFrom(env.Get); err == nil {
				t.Fatalf("expected error for %s=%q", tc.key, tc.val)
			}
		})
	}
}

func TestLoadFrom_NodeNameLengthBoundary(t *testing.T) {
	cases := []struct {
		name    string
		val     string
		wantErr bool
	}{
		{"empty", "", true},
		{"one_char", "a", false},
		{"255_chars", strings.Repeat("a", 255), false},
		{"256_chars", strings.Repeat("a", 256), true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			env := baseEnv()
			env["NODE_NAME"] = tc.val
			_, err := LoadFrom(env.Get)
			if tc.wantErr && err == nil {
				t.Fatalf("expected error for %q", tc.val)
			}
			if !tc.wantErr && err != nil {
				t.Fatalf("expected ok for %q, got %v", tc.val, err)
			}
		})
	}
}

func TestLoadFrom_NodeNameForbiddenChars(t *testing.T) {
	for _, val := range []string{"a/b", "a..b", "a b", "a\x00b", "a\nb"} {
		env := baseEnv()
		env["NODE_NAME"] = val
		if _, err := LoadFrom(env.Get); err == nil {
			t.Errorf("expected error for NODE_NAME=%q", val)
		}
	}
}

func TestLoadFrom_BootstrapTokenLength(t *testing.T) {
	cases := []struct {
		name    string
		val     string
		wantErr bool
	}{
		{"empty", "", true},
		{"one_byte", "x", false},
		{"4096_bytes", strings.Repeat("x", 4096), false},
		{"4097_bytes", strings.Repeat("x", 4097), true},
		{"65536_bytes_overflow", strings.Repeat("x", 65536), true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			env := baseEnv()
			env["BOOTSTRAP_TOKEN"] = tc.val
			_, err := LoadFrom(env.Get)
			if tc.wantErr && err == nil {
				t.Fatalf("expected error for %d-byte token", len(tc.val))
			}
			if !tc.wantErr && err != nil {
				t.Fatalf("expected ok for %d-byte token, got %v", len(tc.val), err)
			}
		})
	}
}

func TestLoadFrom_InferenceTokenLength(t *testing.T) {
	cases := []struct {
		name    string
		val     string
		wantErr bool
	}{
		{"empty", "", true},
		{"too_short_8_bytes", "12345678", true}, // require min 16 to avoid trivial guess
		{"min_16_bytes", "1234567890123456", false},
		{"4096_bytes", strings.Repeat("x", 4096), false},
		{"4097_bytes", strings.Repeat("x", 4097), true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			env := baseEnv()
			env["INFERENCE_TOKEN"] = tc.val
			_, err := LoadFrom(env.Get)
			if tc.wantErr && err == nil {
				t.Fatalf("expected error for %d-byte inference token", len(tc.val))
			}
			if !tc.wantErr && err != nil {
				t.Fatalf("expected ok for %d-byte inference token, got %v", len(tc.val), err)
			}
		})
	}
}

func TestLoadFrom_PoolIDIsUUID(t *testing.T) {
	cases := []struct {
		val     string
		wantErr bool
	}{
		{"11111111-1111-1111-1111-111111111111", false},
		{"not-a-uuid", true},
		{"", true},
		{"11111111111111111111111111111111", true}, // missing hyphens
	}
	for _, tc := range cases {
		env := baseEnv()
		env["POOL_ID"] = tc.val
		_, err := LoadFrom(env.Get)
		if tc.wantErr && err == nil {
			t.Errorf("expected error for POOL_ID=%q", tc.val)
		}
		if !tc.wantErr && err != nil {
			t.Errorf("expected ok for POOL_ID=%q, got %v", tc.val, err)
		}
	}
}

func TestLoadFrom_LogLevelEnum(t *testing.T) {
	for _, val := range []string{"debug", "info", "warn", "error"} {
		env := baseEnv()
		env["LOG_LEVEL"] = val
		cfg, err := LoadFrom(env.Get)
		if err != nil {
			t.Fatalf("level=%q: %v", val, err)
		}
		if cfg.LogLevel != val {
			t.Errorf("level=%q: got %q", val, cfg.LogLevel)
		}
	}
	for _, val := range []string{"trace", "verbose", "INFO", "fatal", ""} {
		env := baseEnv()
		env["LOG_LEVEL"] = val
		if _, err := LoadFrom(env.Get); err == nil && val != "" {
			t.Errorf("expected error for LOG_LEVEL=%q", val)
		}
	}
}

func TestLoadFrom_NumericOverrides(t *testing.T) {
	env := baseEnv()
	env["PULL_TIMEOUT_SECONDS"] = "1200"
	env["READINESS_TIMEOUT_SECONDS"] = "60"
	env["HEARTBEAT_INTERVAL_SECONDS"] = "10"
	cfg, err := LoadFrom(env.Get)
	if err != nil {
		t.Fatalf("%v", err)
	}
	if cfg.PullTimeoutSeconds != 1200 || cfg.ReadinessTimeoutSeconds != 60 || cfg.HeartbeatIntervalSeconds != 10 {
		t.Errorf("got %+v", cfg)
	}
}

func TestLoadFrom_NumericInvalid(t *testing.T) {
	cases := []struct {
		key string
		val string
	}{
		{"PULL_TIMEOUT_SECONDS", "abc"},
		{"PULL_TIMEOUT_SECONDS", "-5"},
		{"PULL_TIMEOUT_SECONDS", "0"},
		{"READINESS_TIMEOUT_SECONDS", "-1"},
		{"HEARTBEAT_INTERVAL_SECONDS", "0"},
		{"HEARTBEAT_INTERVAL_SECONDS", "99999"}, // upper bound check
	}
	for _, tc := range cases {
		t.Run(tc.key+"="+tc.val, func(t *testing.T) {
			env := baseEnv()
			env[tc.key] = tc.val
			if _, err := LoadFrom(env.Get); err == nil {
				t.Errorf("expected error for %s=%q", tc.key, tc.val)
			}
		})
	}
}

func TestLoad_ReadsFromOSEnv(t *testing.T) {
	// End-to-end smoke: Load() reads from os.Getenv. We set the env, call Load, restore.
	keys := []string{
		"CONTROL_PLANE_URL", "BOOTSTRAP_TOKEN", "NODE_NAME",
		"POOL_ID", "WORKER_ADVERTISE_URL", "INFERENCE_TOKEN",
	}
	saved := map[string]string{}
	for _, k := range keys {
		saved[k] = ""
		t.Setenv(k, baseEnv()[k])
	}
	cfg, err := Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.NodeName != "gpu-node-1" {
		t.Errorf("Load NodeName: %q", cfg.NodeName)
	}
}
