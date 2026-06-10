// Package inference exposes the worker's HTTP proxy for /v1/* requests.
// Incoming auth has already been checked by the auth middleware; this layer
// resolves the target deployment and forwards (streaming-aware) to the local
// model container.
package inference

import (
	"bytes"
	"encoding/json"
	"io"
	"net"
	"net/http"
	"strings"
	"time"

	"github.com/gofiber/fiber/v2"
	"github.com/inferia/inferia-worker/internal/metrics"
)

// EndpointResolver returns the base URL of the local model container for a
// deployment, or "" if not loaded. Implemented by *runtime.Runtime.
type EndpointResolver interface {
	EndpointURL(deploymentID string) string
	DeploymentInfo(deploymentID string) (recipe, model, phase string, pullDur, startDur time.Duration, ok bool)
}

// Config wires up the proxy.
type Config struct {
	Runtime  EndpointResolver
	Resolver PathResolver
	Metrics  *metrics.Collector
	Registry *DeploymentRegistry // nil = legacy single-container mode
}

// PathResolver decides which deployment a request targets. If Resolver is set
// it wins; otherwise the named Header is consulted; otherwise "".
type PathResolver struct {
	Header   string
	Resolver func(*fiber.Ctx) string
}

// Resolve picks the deployment id for the current request.
func (p PathResolver) Resolve(c *fiber.Ctx) string {
	if p.Resolver != nil {
		return p.Resolver(c)
	}
	if p.Header != "" {
		return c.Get(p.Header)
	}
	return ""
}

// httpClient is the upstream client. A bare *http.Client is sufficient — we
// stream the response body verbatim. Long timeout because LLM generations can
// take minutes.
var httpClient = &http.Client{
	Timeout: 30 * time.Minute,
	Transport: &http.Transport{
		Proxy: http.ProxyFromEnvironment,
		DialContext: (&net.Dialer{
			Timeout:   10 * time.Second,
			KeepAlive: 30 * time.Second,
		}).DialContext,
		// Don't buffer streaming bodies; flush as bytes arrive.
		DisableCompression:    true,
		MaxIdleConns:          100,
		MaxIdleConnsPerHost:   20,
		IdleConnTimeout:       90 * time.Second,
		TLSHandshakeTimeout:   10 * time.Second,
		ResponseHeaderTimeout: 0, // unlimited; some LLMs are slow to first token
	},
}

// NewProxy returns a Fiber handler that forwards /v1/* to the resolved
// endpoint. When cfg.Registry contains a ModeDisagg entry for the
// deployment it performs two-phase P→D routing via kv_transfer_params.
// Other paths pass through untouched (so /healthz, /metrics, etc.
// remain locally served).
func NewProxy(cfg Config) fiber.Handler {
	return func(c *fiber.Ctx) error {
		if !strings.HasPrefix(c.Path(), "/v1/") {
			return c.Next()
		}
		deploymentID := cfg.Resolver.Resolve(c)

		// Disagg path: route through P→D handoff.
		if cfg.Registry != nil {
			entry, ok := cfg.Registry.LookupByID(deploymentID)
			if ok && entry.Mode == ModeDisagg {
				return handleDisagg(c, entry)
			}
		}

		// Legacy single-endpoint path.
		base := ""
		if deploymentID != "" {
			base = cfg.Runtime.EndpointURL(deploymentID)
		}
		if base == "" {
			c.Set("Retry-After", "5")
			return c.Status(fiber.StatusServiceUnavailable).SendString("deployment not loaded")
		}

		// Metrics: Mark request as active
		if cfg.Metrics != nil && deploymentID != "" {
			cfg.Metrics.IncActive(deploymentID)
			defer cfg.Metrics.DecActive(deploymentID)
		}

		// Build upstream URL.
		path := c.Path()
		query := string(c.Request().URI().QueryString())
		url := strings.TrimRight(base, "/") + path
		if query != "" {
			url += "?" + query
		}

		// Build upstream request, copying method, body, and headers.
		body := c.Body()
		upstreamReq, err := http.NewRequestWithContext(c.UserContext(), c.Method(), url, strings.NewReader(string(body)))
		if err != nil {
			return c.Status(fiber.StatusBadGateway).SendString(err.Error())
		}
		c.Request().Header.VisitAll(func(k, v []byte) {
			key := string(k)
			// Hop-by-hop headers should not be forwarded.
			if isHopByHop(key) {
				return
			}
			upstreamReq.Header.Add(key, string(v))
		})

		start := time.Now()
		resp, err := httpClient.Do(upstreamReq)
		latency := time.Since(start)

		if err != nil {
			return c.Status(fiber.StatusBadGateway).SendString("upstream: " + err.Error())
		}

		// Record metrics — must happen BEFORE SetBodyStream / return nil,
		// because the handler exits immediately after that for streaming
		// responses and the code below it is dead.
		if cfg.Metrics != nil && deploymentID != "" {
			recipe, model, _, _, _, ok := cfg.Runtime.DeploymentInfo(deploymentID)
			if ok {
				cfg.Metrics.RecordRequest(deploymentID, recipe, model, latency.Milliseconds())
			}
		}

		// Copy status + headers (excluding hop-by-hop).
		for k, vs := range resp.Header {
			if isHopByHop(k) {
				continue
			}
			for _, v := range vs {
				c.Response().Header.Add(k, v)
			}
		}
		c.Status(resp.StatusCode)

		// Streaming write: Fiber/fasthttp uses SetBodyStream to forward chunks
		// without buffering the whole response.
		c.Context().SetBodyStream(&streamReader{r: resp.Body}, -1)
		return nil
	}
}

// handleDisagg performs two-phase P→D routing for disagg deployments.
// Phase 1: send modified body to P (stream=false, max_tokens=1, do_remote_decode=true).
// Phase 2: extract kv_transfer_params from P response, inject into original
// body, send to D, and stream the response to the caller.
func handleDisagg(c *fiber.Ctx, entry *DeploymentEntry) error {
	body := c.Body()

	// --- Phase 1: Prefill ---
	prefillBody, err := buildPrefillBody(body)
	if err != nil {
		return c.Status(fiber.StatusBadGateway).SendString("prefill build: " + err.Error())
	}
	prefillURL := entry.Group.NextPrefill() + c.Path()
	if q := string(c.Request().URI().QueryString()); q != "" {
		prefillURL += "?" + q
	}
	prefillReq, err := http.NewRequestWithContext(c.UserContext(), c.Method(), prefillURL, bytes.NewReader(prefillBody))
	if err != nil {
		return c.Status(fiber.StatusBadGateway).SendString("prefill req: " + err.Error())
	}
	c.Request().Header.VisitAll(func(k, v []byte) {
		if isHopByHop(string(k)) {
			return
		}
		prefillReq.Header.Add(string(k), string(v))
	})
	prefillResp, err := httpClient.Do(prefillReq)
	if err != nil {
		return c.Status(fiber.StatusBadGateway).SendString("prefill: " + err.Error())
	}
	defer prefillResp.Body.Close()
	if prefillResp.StatusCode != http.StatusOK {
		return c.Status(fiber.StatusBadGateway).SendString("prefill: upstream error")
	}
	var prefillResult map[string]any
	if err := json.NewDecoder(prefillResp.Body).Decode(&prefillResult); err != nil {
		return c.Status(fiber.StatusBadGateway).SendString("prefill decode: "+err.Error())
	}
	kvParams, ok := prefillResult["kv_transfer_params"]
	if !ok {
		return c.Status(fiber.StatusBadGateway).SendString("prefill: missing kv_transfer_params")
	}

	// --- Phase 2: Decode ---
	decodeBody, err := buildDecodeBody(body, kvParams)
	if err != nil {
		return c.Status(fiber.StatusBadGateway).SendString("decode build: " + err.Error())
	}
	decodeURL := entry.Group.NextDecode() + c.Path()
	if q := string(c.Request().URI().QueryString()); q != "" {
		decodeURL += "?" + q
	}
	decodeReq, err := http.NewRequestWithContext(c.UserContext(), c.Method(), decodeURL, bytes.NewReader(decodeBody))
	if err != nil {
		return c.Status(fiber.StatusBadGateway).SendString("decode req: " + err.Error())
	}
	c.Request().Header.VisitAll(func(k, v []byte) {
		if isHopByHop(string(k)) {
			return
		}
		decodeReq.Header.Add(string(k), string(v))
	})
	decodeResp, err := httpClient.Do(decodeReq)
	if err != nil {
		return c.Status(fiber.StatusBadGateway).SendString("decode: " + err.Error())
	}

	// Stream decode response to client.
	for k, vs := range decodeResp.Header {
		if isHopByHop(k) {
			continue
		}
		for _, v := range vs {
			c.Response().Header.Add(k, v)
		}
	}
	c.Status(decodeResp.StatusCode)
	c.Context().SetBodyStream(&streamReader{r: decodeResp.Body}, -1)
	return nil
}

// buildPrefillBody clones the original request JSON and sets stream=false,
// max_tokens=1, do_remote_decode=true for the prefill phase.
func buildPrefillBody(original []byte) ([]byte, error) {
	if len(original) == 0 {
		return original, nil
	}
	var body map[string]any
	if err := json.Unmarshal(original, &body); err != nil {
		return nil, err
	}
	body["stream"] = false
	body["max_tokens"] = 1.0
	body["do_remote_decode"] = true
	return json.Marshal(body)
}

// buildDecodeBody clones the original request JSON and injects
// kv_transfer_params from the prefill phase for the decode phase.
func buildDecodeBody(original []byte, kvParams any) ([]byte, error) {
	if len(original) == 0 {
		return original, nil
	}
	var body map[string]any
	if err := json.Unmarshal(original, &body); err != nil {
		return nil, err
	}
	body["kv_transfer_params"] = kvParams
	return json.Marshal(body)
}

// streamReader forwards reads from the upstream body. fasthttp will call
// Close() once it finishes streaming, so we don't close on intermediate errors.
type streamReader struct{ r io.ReadCloser }

func (s *streamReader) Read(p []byte) (int, error) { return s.r.Read(p) }
func (s *streamReader) Close() error               { return s.r.Close() }

var hopByHop = map[string]struct{}{
	"connection":          {},
	"keep-alive":          {},
	"proxy-authenticate":  {},
	"proxy-authorization": {},
	"te":                  {},
	"trailers":            {},
	"transfer-encoding":   {},
	"upgrade":             {},
}

func isHopByHop(h string) bool {
	_, ok := hopByHop[strings.ToLower(h)]
	return ok
}
