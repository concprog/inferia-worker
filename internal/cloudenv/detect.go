// Package cloudenv detects the runtime environment the worker is running in
// (AWS EC2, local, etc) and exposes the result to bootstrap + control packages.
// IMDSv2 probe is the only network call; budget is 200ms total.
package cloudenv

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"sync"
	"time"
)

type Kind string

const (
	KindLocal   Kind = "local"
	KindAWSEC2  Kind = "aws-ec2"
	KindUnknown Kind = "unknown"
)

// RuntimeInfo is the small bundle of facts the worker tells the control plane
// about where it lives. All fields except Kind are best-effort.
type RuntimeInfo struct {
	Kind             Kind   `json:"runtime_env"`
	InstanceID       string `json:"instance_id,omitempty"`
	Region           string `json:"region,omitempty"`
	AvailabilityZone string `json:"availability_zone,omitempty"`
}

const (
	defaultIMDSBase = "http://169.254.169.254"
	totalBudget     = 200 * time.Millisecond
	maxFieldLen     = 128
)

// cacheOnce and cached hold the singleton result of the first Detect() call.
// Tests that exercise Detect() must reset these before running:
//
//	cacheOnce = sync.Once{}
//	cached = RuntimeInfo{}
var (
	cacheOnce sync.Once
	cached    RuntimeInfo
)

// Detect returns the runtime info, cached after the first successful call.
func Detect() RuntimeInfo {
	cacheOnce.Do(func() { cached = detectFresh() })
	return cached
}

// detectFresh re-runs the full detection. Test-only; production callers use Detect.
func detectFresh() RuntimeInfo {
	info := RuntimeInfo{Kind: KindLocal}

	if v := os.Getenv("INFERIA_RUNTIME_ENV"); v != "" {
		info.Kind = Kind(truncate(v, maxFieldLen))
	}

	if info.Kind == KindLocal {
		if probed, ok := probeIMDS(); ok {
			info = probed
		}
	}

	// Per-field env overrides (apply on top of either env-Kind or IMDS).
	if v := os.Getenv("INFERIA_INSTANCE_ID"); v != "" {
		info.InstanceID = truncate(v, maxFieldLen)
	}
	if v := os.Getenv("INFERIA_REGION"); v != "" {
		info.Region = truncate(v, maxFieldLen)
	}
	if v := os.Getenv("INFERIA_AZ"); v != "" {
		info.AvailabilityZone = truncate(v, maxFieldLen)
	}
	return info
}

// probeIMDS runs IMDSv2 with a 200ms total budget. Returns (info, true) on
// success, (zero, false) on any failure (network, non-200, parse, etc).
func probeIMDS() (RuntimeInfo, bool) {
	base := os.Getenv("INFERIA_CLOUDENV_IMDS_URL")
	if base == "" {
		base = defaultIMDSBase
	}
	ctx, cancel := context.WithTimeout(context.Background(), totalBudget)
	defer cancel()

	// Step 1: PUT to get a session token.
	tokReq, err := http.NewRequestWithContext(ctx, http.MethodPut, base+"/latest/api/token", nil)
	if err != nil {
		return RuntimeInfo{}, false
	}
	tokReq.Header.Set("X-aws-ec2-metadata-token-ttl-seconds", "60")
	client := &http.Client{}
	tokResp, err := client.Do(tokReq)
	if err != nil {
		return RuntimeInfo{}, false
	}
	defer tokResp.Body.Close()
	if tokResp.StatusCode != http.StatusOK {
		return RuntimeInfo{}, false
	}
	tokBytes, err := io.ReadAll(io.LimitReader(tokResp.Body, 1024))
	if err != nil {
		return RuntimeInfo{}, false
	}
	token := string(tokBytes)

	// Step 2: GET identity document.
	docReq, err := http.NewRequestWithContext(ctx, http.MethodGet, base+"/latest/dynamic/instance-identity/document", nil)
	if err != nil {
		return RuntimeInfo{}, false
	}
	docReq.Header.Set("X-aws-ec2-metadata-token", token)
	docResp, err := client.Do(docReq)
	if err != nil {
		return RuntimeInfo{}, false
	}
	defer docResp.Body.Close()
	if docResp.StatusCode != http.StatusOK {
		return RuntimeInfo{}, false
	}
	// Bound the body to defend against pathological responses.
	const maxBody = 64 * 1024
	docBytes, err := io.ReadAll(io.LimitReader(docResp.Body, maxBody+1))
	if err != nil {
		return RuntimeInfo{}, false
	}
	if len(docBytes) > maxBody {
		return RuntimeInfo{}, false
	}
	var doc struct {
		InstanceID       string `json:"instanceId"`
		Region           string `json:"region"`
		AvailabilityZone string `json:"availabilityZone"`
	}
	if err := json.Unmarshal(docBytes, &doc); err != nil {
		return RuntimeInfo{}, false
	}
	return RuntimeInfo{
		Kind:             KindAWSEC2,
		InstanceID:       truncate(doc.InstanceID, maxFieldLen),
		Region:           truncate(doc.Region, maxFieldLen),
		AvailabilityZone: truncate(doc.AvailabilityZone, maxFieldLen),
	}, true
}

func truncate(s string, n int) string {
	if len(s) > n {
		return s[:n]
	}
	return s
}
