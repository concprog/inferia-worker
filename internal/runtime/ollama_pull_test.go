package runtime

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
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
