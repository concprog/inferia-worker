// Package config parses and validates worker configuration from environment
// variables. All values are read at startup; the resulting Config is immutable.
package config

import (
	"errors"
	"fmt"
	"net/url"
	"os"
	"regexp"
	"strconv"
	"strings"
)

// Config holds the worker's full runtime configuration.
type Config struct {
	ControlPlaneURL    string
	BootstrapToken     string // may be empty if a persisted token already exists
	NodeName           string
	PoolID             string
	WorkerAdvertiseURL string
	WorkerListenAddr   string
	InferenceToken     string
	TokenFile          string
	DockerHost         string
	ModelsNetwork      string
	LogLevel           string

	PullTimeoutSeconds       int
	ReadinessTimeoutSeconds  int
	HeartbeatIntervalSeconds int
}

const (
	maxNodeNameLen      = 255
	maxTokenLen         = 4096
	minInferenceToken   = 16
	maxTimeoutSeconds   = 86400 // 1 day cap to avoid runaway configs
	defaultListenAddr   = "0.0.0.0:8080"
	defaultTokenFile    = "/var/lib/inferia-worker/token"
	defaultDockerHost   = "unix:///var/run/docker.sock"
	defaultModelsNet    = "inferia-models"
	defaultLogLevel     = "info"
	defaultPullTimeout  = 600
	defaultReadyTimeout = 180
	defaultHeartbeat    = 5
)

var (
	allowedURLSchemes = map[string]struct{}{"http": {}, "https": {}}
	allowedLogLevels  = map[string]struct{}{"debug": {}, "info": {}, "warn": {}, "error": {}}

	// uuidPattern matches the canonical 8-4-4-4-12 lowercase or uppercase form.
	uuidPattern = regexp.MustCompile(`^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$`)

	// nodeNamePattern enforces a conservative subset: alnum, dash, underscore, dot.
	// This is reused for `docker run --name` and DNS-ish identifiers.
	nodeNamePattern = regexp.MustCompile(`^[A-Za-z0-9._-]+$`)
)

// Getter reads a single env-style key; missing keys return "".
type Getter func(string) string

// Load reads configuration from os.Getenv.
func Load() (*Config, error) {
	return LoadFrom(os.Getenv)
}

// LoadFrom reads configuration using the supplied Getter. Used by tests and by
// Load. Validates required fields, length bounds, URL schemes, log-level enum
// and numeric ranges.
func LoadFrom(getenv Getter) (*Config, error) {
	cfg := &Config{
		ControlPlaneURL:    getenv("CONTROL_PLANE_URL"),
		BootstrapToken:     getenv("BOOTSTRAP_TOKEN"),
		NodeName:           getenv("NODE_NAME"),
		PoolID:             getenv("POOL_ID"),
		WorkerAdvertiseURL: getenv("WORKER_ADVERTISE_URL"),
		WorkerListenAddr:   firstNonEmpty(getenv("WORKER_LISTEN_ADDR"), defaultListenAddr),
		InferenceToken:     getenv("INFERENCE_TOKEN"),
		TokenFile:          firstNonEmpty(getenv("TOKEN_FILE"), defaultTokenFile),
		DockerHost:         firstNonEmpty(getenv("DOCKER_HOST"), defaultDockerHost),
		ModelsNetwork:      firstNonEmpty(getenv("MODELS_NETWORK"), defaultModelsNet),
		LogLevel:           firstNonEmpty(getenv("LOG_LEVEL"), defaultLogLevel),
	}

	var errs []string

	// Required.
	if cfg.ControlPlaneURL == "" {
		errs = append(errs, "CONTROL_PLANE_URL is required")
	}
	if cfg.NodeName == "" {
		errs = append(errs, "NODE_NAME is required")
	}
	if cfg.PoolID == "" {
		errs = append(errs, "POOL_ID is required")
	}
	if cfg.WorkerAdvertiseURL == "" {
		errs = append(errs, "WORKER_ADVERTISE_URL is required")
	}
	if cfg.InferenceToken == "" {
		errs = append(errs, "INFERENCE_TOKEN is required")
	}

	// BootstrapToken is required unless a persisted token is already on disk.
	tokenFilePresent := strings.EqualFold(getenv("TOKEN_FILE_PRESENT"), "true")
	if cfg.BootstrapToken == "" && !tokenFilePresent {
		errs = append(errs, "BOOTSTRAP_TOKEN is required (no persisted token found)")
	}

	// URL schemes.
	if cfg.ControlPlaneURL != "" {
		if err := validateHTTPURL("CONTROL_PLANE_URL", cfg.ControlPlaneURL); err != nil {
			errs = append(errs, err.Error())
		}
	}
	if cfg.WorkerAdvertiseURL != "" {
		if err := validateHTTPURL("WORKER_ADVERTISE_URL", cfg.WorkerAdvertiseURL); err != nil {
			errs = append(errs, err.Error())
		}
	}

	// NodeName: length + character set.
	if cfg.NodeName != "" {
		if len(cfg.NodeName) > maxNodeNameLen {
			errs = append(errs, fmt.Sprintf("NODE_NAME exceeds %d chars (got %d)", maxNodeNameLen, len(cfg.NodeName)))
		}
		if !nodeNamePattern.MatchString(cfg.NodeName) || strings.Contains(cfg.NodeName, "..") {
			errs = append(errs, fmt.Sprintf("NODE_NAME contains forbidden characters: %q", cfg.NodeName))
		}
	}

	// PoolID: UUID format.
	if cfg.PoolID != "" && !uuidPattern.MatchString(cfg.PoolID) {
		errs = append(errs, fmt.Sprintf("POOL_ID is not a UUID: %q", cfg.PoolID))
	}

	// BootstrapToken length (only checked when supplied).
	if cfg.BootstrapToken != "" && len(cfg.BootstrapToken) > maxTokenLen {
		errs = append(errs, fmt.Sprintf("BOOTSTRAP_TOKEN exceeds %d bytes (got %d)", maxTokenLen, len(cfg.BootstrapToken)))
	}

	// InferenceToken length bounds.
	if cfg.InferenceToken != "" {
		if len(cfg.InferenceToken) < minInferenceToken {
			errs = append(errs, fmt.Sprintf("INFERENCE_TOKEN must be at least %d bytes", minInferenceToken))
		}
		if len(cfg.InferenceToken) > maxTokenLen {
			errs = append(errs, fmt.Sprintf("INFERENCE_TOKEN exceeds %d bytes (got %d)", maxTokenLen, len(cfg.InferenceToken)))
		}
	}

	// LogLevel enum.
	if _, ok := allowedLogLevels[cfg.LogLevel]; !ok {
		errs = append(errs, fmt.Sprintf("LOG_LEVEL %q not in {debug,info,warn,error}", cfg.LogLevel))
	}

	// Numeric overrides with bounds.
	var err error
	if cfg.PullTimeoutSeconds, err = parsePositiveInt(getenv, "PULL_TIMEOUT_SECONDS", defaultPullTimeout); err != nil {
		errs = append(errs, err.Error())
	}
	if cfg.ReadinessTimeoutSeconds, err = parsePositiveInt(getenv, "READINESS_TIMEOUT_SECONDS", defaultReadyTimeout); err != nil {
		errs = append(errs, err.Error())
	}
	if cfg.HeartbeatIntervalSeconds, err = parsePositiveInt(getenv, "HEARTBEAT_INTERVAL_SECONDS", defaultHeartbeat); err != nil {
		errs = append(errs, err.Error())
	}

	if len(errs) > 0 {
		return nil, errors.New(strings.Join(errs, "; "))
	}
	return cfg, nil
}

func firstNonEmpty(a, b string) string {
	if a != "" {
		return a
	}
	return b
}

func validateHTTPURL(name, raw string) error {
	u, err := url.Parse(raw)
	if err != nil {
		return fmt.Errorf("%s is not a valid URL: %v", name, err)
	}
	if u.Scheme == "" || u.Host == "" {
		return fmt.Errorf("%s must be an absolute URL with scheme and host: %q", name, raw)
	}
	if _, ok := allowedURLSchemes[strings.ToLower(u.Scheme)]; !ok {
		return fmt.Errorf("%s scheme %q not in {http,https}", name, u.Scheme)
	}
	return nil
}

func parsePositiveInt(getenv Getter, key string, dflt int) (int, error) {
	raw := getenv(key)
	if raw == "" {
		return dflt, nil
	}
	n, err := strconv.Atoi(raw)
	if err != nil {
		return 0, fmt.Errorf("%s is not an integer: %q", key, raw)
	}
	if n <= 0 {
		return 0, fmt.Errorf("%s must be > 0 (got %d)", key, n)
	}
	if n > maxTimeoutSeconds {
		return 0, fmt.Errorf("%s exceeds %d (got %d)", key, maxTimeoutSeconds, n)
	}
	return n, nil
}
