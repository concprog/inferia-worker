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
// INFERIA_OLLAMA_MODEL env var). The endpoint is the in-network address of
// the model container (same shape the readiness probe uses).
func ollamaPullIfNeeded(ctx context.Context, plan recipesPlan, timeout time.Duration) error {
	model := plan.Env["INFERIA_OLLAMA_MODEL"]
	if model == "" {
		return nil // not an ollama recipe
	}
	endpoint := fmt.Sprintf("http://%s:%d", plan.ContainerName, plan.ContainerPort)
	return ollamaPull(ctx, endpoint, model, timeout)
}
