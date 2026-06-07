package runtime

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"
)

func newOllamaServer(t *testing.T, handler http.HandlerFunc) (string, func()) {
	t.Helper()
	srv := httptest.NewServer(handler)
	u, err := url.Parse(srv.URL)
	if err != nil {
		t.Fatalf("parse server url: %v", err)
	}
	return u.Host, srv.Close
}

func TestOllamaPull_HappyPath(t *testing.T) {
	called := 0
	host, stop := newOllamaServer(t, func(w http.ResponseWriter, r *http.Request) {
		called++
		if r.URL.Path != "/api/pull" {
			t.Errorf("path = %s, want /api/pull", r.URL.Path)
		}
		if r.Method != http.MethodPost {
			t.Errorf("method = %s, want POST", r.Method)
		}
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		if body["name"] != "qwen3:0.6b" {
			t.Errorf("name = %v, want qwen3:0.6b", body["name"])
		}
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(map[string]string{"status": "success"})
	})
	defer stop()

	err := ollamaPull(context.Background(), "http://"+host, "qwen3:0.6b", 5*time.Second)
	if err != nil {
		t.Fatalf("ollamaPull returned %v, want nil", err)
	}
	if called != 1 {
		t.Errorf("called = %d, want 1", called)
	}
}

func TestOllamaPull_StreamingNDJSON(t *testing.T) {
	host, stop := newOllamaServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/x-ndjson")
		w.WriteHeader(http.StatusOK)
		flusher, _ := w.(http.Flusher)
		for _, line := range []string{
			`{"status":"pulling manifest"}`,
			`{"status":"downloading","completed":100,"total":1000}`,
			`{"status":"downloading","completed":800,"total":1000}`,
			`{"status":"verifying sha256 digest"}`,
			`{"status":"success"}`,
			``,
		} {
			_, _ = io.WriteString(w, line+"\n")
			if flusher != nil {
				flusher.Flush()
			}
		}
	})
	defer stop()

	err := ollamaPull(context.Background(), "http://"+host, "qwen3:0.6b", 5*time.Second)
	if err != nil {
		t.Fatalf("ollamaPull returned %v, want nil", err)
	}
}

func TestOllamaPull_Transient5xxRetried(t *testing.T) {
	var calls int
	host, stop := newOllamaServer(t, func(w http.ResponseWriter, r *http.Request) {
		calls++
		if calls == 1 {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, `{"status":"success"}`)
	})
	defer stop()

	err := ollamaPull(context.Background(), "http://"+host, "qwen3:0.6b", 5*time.Second)
	if err != nil {
		t.Fatalf("err = %v, want nil after retry", err)
	}
	if calls != 2 {
		t.Errorf("calls = %d, want 2", calls)
	}
}

func TestOllamaPull_Persistent5xxFails(t *testing.T) {
	var calls int
	host, stop := newOllamaServer(t, func(w http.ResponseWriter, r *http.Request) {
		calls++
		w.WriteHeader(http.StatusInternalServerError)
	})
	defer stop()

	err := ollamaPull(context.Background(), "http://"+host, "qwen3:0.6b", 5*time.Second)
	if err == nil {
		t.Fatalf("err = nil, want non-nil")
	}
	if calls != 2 {
		t.Errorf("calls = %d, want 2 (initial + 1 retry)", calls)
	}
}

func TestOllamaPull_4xxNotRetried(t *testing.T) {
	var calls int
	host, stop := newOllamaServer(t, func(w http.ResponseWriter, r *http.Request) {
		calls++
		w.WriteHeader(http.StatusNotFound)
		_, _ = io.WriteString(w, `{"error":"pull model manifest: file does not exist"}`)
	})
	defer stop()

	err := ollamaPull(context.Background(), "http://"+host, "qwen3:0.6b", 5*time.Second)
	if err == nil {
		t.Fatalf("err = nil, want non-nil")
	}
	if calls != 1 {
		t.Errorf("calls = %d, want 1 (no retry on 4xx)", calls)
	}
}

func TestOllamaPull_Timeout(t *testing.T) {
	host, stop := newOllamaServer(t, func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(500 * time.Millisecond)
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, `{"status":"success"}`)
	})
	defer stop()

	start := time.Now()
	err := ollamaPull(context.Background(), "http://"+host, "qwen3:0.6b", 100*time.Millisecond)
	elapsed := time.Since(start)
	if err == nil {
		t.Fatalf("err = nil, want timeout")
	}
	if elapsed > 400*time.Millisecond {
		t.Errorf("elapsed = %v, want quick timeout < 400ms", elapsed)
	}
}

func TestOllamaPull_NetworkError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {}))
	host := srv.Listener.Addr().String()
	srv.Close()

	err := ollamaPull(context.Background(), "http://"+host, "qwen3:0.6b", 200*time.Millisecond)
	if err == nil {
		t.Fatalf("err = nil, want network error")
	}
}

func TestOllamaPull_RejectsEmptyName(t *testing.T) {
	err := ollamaPull(context.Background(), "http://127.0.0.1:1", "", 5*time.Second)
	if err == nil {
		t.Fatalf("err = nil, want validation failure")
	}
}

func TestOllamaPull_RejectsOversizedName(t *testing.T) {
	name := strings.Repeat("a", 257)
	err := ollamaPull(context.Background(), "http://127.0.0.1:1", name, 5*time.Second)
	if err == nil {
		t.Fatalf("err = nil, want validation failure")
	}
}

func TestOllamaPull_RejectsShellMetaName(t *testing.T) {
	for _, n := range []string{
		"qwen3;rm -rf /",
		"qwen3|cat",
		"qwen3`whoami`",
		"qwen3$PATH",
		"qwen3>file",
		"qwen3<file",
		"line1\nline2",
	} {
		err := ollamaPull(context.Background(), "http://127.0.0.1:1", n, 5*time.Second)
		if err == nil {
			t.Errorf("err = nil for name=%q, want validation failure", n)
		}
	}
}

func TestOllamaPull_TerminalStatusNotSuccess(t *testing.T) {
	host, stop := newOllamaServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, `{"status":"pulling"}`+"\n"+`{"error":"manifest not found"}`)
	})
	defer stop()
	err := ollamaPull(context.Background(), "http://"+host, "qwen3:0.6b", 5*time.Second)
	if err == nil {
		t.Fatalf("err = nil, want non-nil")
	}
}

func TestOllamaPull_EmptyBody(t *testing.T) {
	host, stop := newOllamaServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	defer stop()
	err := ollamaPull(context.Background(), "http://"+host, "qwen3:0.6b", 5*time.Second)
	if err == nil {
		t.Fatalf("err = nil, want non-nil")
	}
}

func TestOllamaPull_UnparseableBody(t *testing.T) {
	host, stop := newOllamaServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, `not json`)
	})
	defer stop()
	err := ollamaPull(context.Background(), "http://"+host, "qwen3:0.6b", 5*time.Second)
	if err == nil {
		t.Fatalf("err = nil, want non-nil")
	}
}

func TestOllamaPull_TerminalStatusFailed(t *testing.T) {
	host, stop := newOllamaServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, `{"status":"failed"}`)
	})
	defer stop()
	err := ollamaPull(context.Background(), "http://"+host, "qwen3:0.6b", 5*time.Second)
	if err == nil {
		t.Fatalf("err = nil, want non-nil for non-success terminal status")
	}
}

func TestOllamaPull_LongErrorBodyTruncated(t *testing.T) {
	// Server returns 4xx with a body > 256 bytes to exercise the truncate path.
	host, stop := newOllamaServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = io.WriteString(w, strings.Repeat("x", 512))
	})
	defer stop()
	err := ollamaPull(context.Background(), "http://"+host, "qwen3:0.6b", 5*time.Second)
	if err == nil {
		t.Fatalf("err = nil, want non-nil")
	}
	// The error message should be bounded — must contain the truncation marker.
	if !strings.Contains(err.Error(), "…") {
		t.Errorf("err = %v, want truncated body marker", err)
	}
}

func TestOllamaPullIfNeeded_RetagsToServedName(t *testing.T) {
	var pulled, copied map[string]any
	host, stop := newOllamaServer(t, func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/pull":
			_ = json.NewDecoder(r.Body).Decode(&pulled)
			w.WriteHeader(http.StatusOK)
			_ = json.NewEncoder(w).Encode(map[string]string{"status": "success"})
		case "/api/copy":
			_ = json.NewDecoder(r.Body).Decode(&copied)
			w.WriteHeader(http.StatusOK)
		default:
			t.Errorf("unexpected path %s", r.URL.Path)
		}
	})
	defer stop()

	plan := recipesPlan{Env: map[string]string{
		"INFERIA_OLLAMA_MODEL":        "inferiallm.wlan0.in/library/gemma3:4b",
		"INFERIA_OLLAMA_SERVED_NAME":  "gemma3:4b",
	}}
	if err := ollamaPullIfNeeded(context.Background(), plan, "http://"+host, 5*time.Second); err != nil {
		t.Fatalf("ollamaPullIfNeeded: %v", err)
	}
	if pulled["name"] != "inferiallm.wlan0.in/library/gemma3:4b" {
		t.Errorf("pulled name = %v, want the mirror ref", pulled["name"])
	}
	if copied["source"] != "inferiallm.wlan0.in/library/gemma3:4b" || copied["destination"] != "gemma3:4b" {
		t.Errorf("copy = %v, want source=mirror-ref destination=gemma3:4b", copied)
	}
}

func TestOllamaPullIfNeeded_NoRetagWhenServedEqualsModel(t *testing.T) {
	copyCalled := false
	host, stop := newOllamaServer(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/copy" {
			copyCalled = true
		}
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(map[string]string{"status": "success"})
	})
	defer stop()
	// No served name (non-mirror deploy): pull only, no copy.
	plan := recipesPlan{Env: map[string]string{"INFERIA_OLLAMA_MODEL": "gemma3:4b"}}
	if err := ollamaPullIfNeeded(context.Background(), plan, "http://"+host, 5*time.Second); err != nil {
		t.Fatalf("ollamaPullIfNeeded: %v", err)
	}
	if copyCalled {
		t.Errorf("api/copy must NOT be called when there is no served-name re-tag")
	}
}
