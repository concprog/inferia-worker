package runtime

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/inferia/inferia-worker/internal/runtime/recipes"
)

// recipesPlan is a local alias so helper signatures stay small.
type recipesPlan = recipes.Plan

// ollamaPull POSTs /api/pull to the local Ollama container at endpoint and
// waits for completion. endpoint is host:port form (no trailing slash, no path).
// Bounded by timeout. Returns an error wrapped with the pull stage context.
func ollamaPull(ctx context.Context, endpoint, model string, timeout time.Duration) error {
	if err := validateOllamaModelName(model); err != nil {
		return fmt.Errorf("ollama pull: %w", err)
	}

	cctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	body, _ := json.Marshal(map[string]any{"name": model, "stream": false})

	const maxAttempts = 2
	var lastErr error
	for attempt := 1; attempt <= maxAttempts; attempt++ {
		req, err := http.NewRequestWithContext(cctx, http.MethodPost,
			strings.TrimRight(endpoint, "/")+"/api/pull", bytes.NewReader(body))
		if err != nil {
			return fmt.Errorf("ollama pull: build request: %w", err)
		}
		req.Header.Set("Content-Type", "application/json")
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			lastErr = fmt.Errorf("ollama pull: post: %w", err)
			if cctx.Err() != nil {
				return lastErr
			}
			continue // network errors are retryable
		}
		raw, rerr := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		if rerr != nil {
			lastErr = fmt.Errorf("ollama pull: read body: %w", rerr)
			continue
		}
		if resp.StatusCode >= 500 {
			lastErr = fmt.Errorf("ollama pull: status=%d body=%s", resp.StatusCode, truncate(raw, 256))
			continue
		}
		if resp.StatusCode >= 400 {
			return fmt.Errorf("ollama pull: status=%d body=%s", resp.StatusCode, truncate(raw, 256))
		}
		return checkOllamaPullResponse(raw)
	}
	return lastErr
}

// validateOllamaModelName rejects empty, oversized, or shell-meta-bearing names.
func validateOllamaModelName(name string) error {
	if name == "" {
		return fmt.Errorf("model name empty")
	}
	if len(name) > 256 {
		return fmt.Errorf("model name > 256 chars")
	}
	for _, c := range name {
		if c == 0 || strings.ContainsRune(";|$`<>\n\r", c) {
			return fmt.Errorf("model name has forbidden char %q", c)
		}
	}
	return nil
}

// checkOllamaPullResponse handles both JSON-object and NDJSON-stream forms.
func checkOllamaPullResponse(body []byte) error {
	trimmed := bytes.TrimSpace(body)
	if len(trimmed) == 0 {
		return fmt.Errorf("ollama pull: empty response body")
	}
	// Last non-empty line is the terminal status (works for both stream and single-object).
	lines := bytes.Split(trimmed, []byte("\n"))
	var last []byte
	for i := len(lines) - 1; i >= 0; i-- {
		if t := bytes.TrimSpace(lines[i]); len(t) > 0 {
			last = t
			break
		}
	}
	var msg struct {
		Status string `json:"status"`
		Error  string `json:"error"`
	}
	if err := json.Unmarshal(last, &msg); err != nil {
		return fmt.Errorf("ollama pull: decode terminal line: %w", err)
	}
	if msg.Error != "" {
		return fmt.Errorf("ollama pull: server error: %s", msg.Error)
	}
	if msg.Status != "success" {
		return fmt.Errorf("ollama pull: terminal status=%q, want success", msg.Status)
	}
	return nil
}

func truncate(b []byte, n int) string {
	if len(b) <= n {
		return string(b)
	}
	return string(b[:n]) + "…"
}

// ollamaPullIfNeeded runs ollamaPull for ollama-recipe deployments. It is a
// no-op for non-ollama containers (detected by the presence of the
// INFERIA_OLLAMA_MODEL env var). ``endpoint`` MUST be the same host-bound
// address the readiness probe used (AdvertiseHost:hostPort) — NOT the bridge
// container name. The worker runs with `--network host`, so a bridge container
// name is not DNS-resolvable from the worker process; using it here produced
// `dial tcp: lookup inferia-ollama-...: server misbehaving` and failed every
// ollama load right after the readiness probe passed.
func ollamaPullIfNeeded(ctx context.Context, plan recipesPlan, endpoint string, timeout time.Duration) error {
	model := plan.Env["INFERIA_OLLAMA_MODEL"]
	if model == "" {
		return nil // not an ollama recipe
	}
	if err := ollamaPull(ctx, endpoint, model, timeout); err != nil {
		return err
	}
	// When the model was pulled from the CP cache mirror, INFERIA_OLLAMA_MODEL
	// is a host-prefixed ref (e.g. "inferiallm.wlan0.in/library/gemma3:4b") so
	// ollama hits the mirror registry. But ollama then serves it under THAT
	// name, while inference requests carry the bare deployment model id
	// ("gemma3:4b") -> model-not-found. Re-tag the pulled model to the bare
	// served name so inference resolves. No-op when not mirroring (served name
	// absent or already equal to the pulled ref).
	served := plan.Env["INFERIA_OLLAMA_SERVED_NAME"]
	if served != "" && served != model {
		if err := ollamaCopy(ctx, endpoint, model, served, timeout); err != nil {
			return fmt.Errorf("ollama re-tag mirror ref to served name: %w", err)
		}
	}
	return nil
}

// ollamaCopy POSTs /api/copy to alias an already-pulled model under a second
// name (ollama's `ollama cp`). Used to re-tag a CP-mirror ref to the bare
// served name. Bounded by timeout.
func ollamaCopy(ctx context.Context, endpoint, source, destination string, timeout time.Duration) error {
	if err := validateOllamaModelName(source); err != nil {
		return fmt.Errorf("ollama copy source: %w", err)
	}
	if err := validateOllamaModelName(destination); err != nil {
		return fmt.Errorf("ollama copy destination: %w", err)
	}
	cctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	body, _ := json.Marshal(map[string]any{"source": source, "destination": destination})
	req, err := http.NewRequestWithContext(cctx, http.MethodPost,
		strings.TrimRight(endpoint, "/")+"/api/copy", bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("ollama copy: build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return fmt.Errorf("ollama copy: post: %w", err)
	}
	raw, _ := io.ReadAll(resp.Body)
	_ = resp.Body.Close()
	if resp.StatusCode >= 400 {
		return fmt.Errorf("ollama copy: status=%d body=%s", resp.StatusCode, truncate(raw, 256))
	}
	return nil
}
