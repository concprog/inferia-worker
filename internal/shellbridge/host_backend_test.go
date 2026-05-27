// Tests for the host-shell backend. The strategy mirrors
// docker_backend_test.go: stand up an httptest.Server that fakes the
// small subset of the Docker REST API the SDK calls, point a real
// docker SDK client at it, and assert on:
//
//   - request bodies (container config, host config)
//   - request counts (image pull only on cache miss)
//   - resize / wait / attach happy + error paths
//   - env-var override
package shellbridge

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/docker/docker/api/types/container"
	dclient "github.com/docker/docker/client"
)

// --- buildNsenterCmd ---------------------------------------------------------

func TestBuildNsenterCmd_DefaultRoot(t *testing.T) {
	got := buildNsenterCmd(ShellSessionConfig{})
	want := []string{"nsenter", "-t", "1", "-m", "-u", "-i", "-n", "-p", "--", "/bin/bash", "-l"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("default config:\n got=%q\nwant=%q", got, want)
	}
}

func TestBuildNsenterCmd_ExplicitRootSameAsEmpty(t *testing.T) {
	gotEmpty := buildNsenterCmd(ShellSessionConfig{})
	gotRoot := buildNsenterCmd(ShellSessionConfig{User: "root"})
	if !reflect.DeepEqual(gotEmpty, gotRoot) {
		t.Errorf("User=\"\" must match User=root: empty=%q root=%q", gotEmpty, gotRoot)
	}
}

func TestBuildNsenterCmd_NonRootUserUsesSu(t *testing.T) {
	got := buildNsenterCmd(ShellSessionConfig{User: "ubuntu"})
	want := []string{"nsenter", "-t", "1", "-m", "-u", "-i", "-n", "-p", "--", "su", "-", "ubuntu", "-s", "/bin/bash"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("non-root user:\n got=%q\nwant=%q", got, want)
	}
}

func TestBuildNsenterCmd_CustomShellHonouredOnBothBranches(t *testing.T) {
	gotRoot := buildNsenterCmd(ShellSessionConfig{Shell: "/bin/zsh"})
	wantRoot := []string{"nsenter", "-t", "1", "-m", "-u", "-i", "-n", "-p", "--", "/bin/zsh", "-l"}
	if !reflect.DeepEqual(gotRoot, wantRoot) {
		t.Errorf("root + zsh:\n got=%q\nwant=%q", gotRoot, wantRoot)
	}
	gotUser := buildNsenterCmd(ShellSessionConfig{Shell: "/bin/zsh", User: "ubuntu"})
	wantUser := []string{"nsenter", "-t", "1", "-m", "-u", "-i", "-n", "-p", "--", "su", "-", "ubuntu", "-s", "/bin/zsh"}
	if !reflect.DeepEqual(gotUser, wantUser) {
		t.Errorf("ubuntu + zsh:\n got=%q\nwant=%q", gotUser, wantUser)
	}
}

func TestBuildNsenterCmd_EmptyShellDefaultsToBash(t *testing.T) {
	got := buildNsenterCmd(ShellSessionConfig{Shell: ""})
	if got[len(got)-2] != "/bin/bash" || got[len(got)-1] != "-l" {
		t.Errorf("expected /bin/bash -l tail, got %q", got)
	}
}

// --- HostShellImage env override --------------------------------------------

func TestHostShellImage_DefaultsToUbuntu(t *testing.T) {
	t.Setenv("INFERIA_HOST_SHELL_IMAGE", "")
	if got := HostShellImage(); got != DefaultHostShellImage {
		t.Errorf("expected default %s, got %s", DefaultHostShellImage, got)
	}
}

func TestHostShellImage_EnvVarOverride(t *testing.T) {
	t.Setenv("INFERIA_HOST_SHELL_IMAGE", "foo/bar:latest")
	if got := HostShellImage(); got != "foo/bar:latest" {
		t.Errorf("expected override, got %s", got)
	}
}

func TestHostShellImage_WhitespaceTrimmed(t *testing.T) {
	// Operators sometimes drop a trailing space into env files; trim
	// so the daemon doesn't 404 on the misspelled tag.
	t.Setenv("INFERIA_HOST_SHELL_IMAGE", "  custom/img:v1  ")
	if got := HostShellImage(); got != "custom/img:v1" {
		t.Errorf("expected trimmed, got %q", got)
	}
}

// --- buildHostContainerConfigs (spec assertions) ----------------------------

func TestBuildHostContainerConfigs_PrivilegedHostNamespaces(t *testing.T) {
	cfg, host := buildHostContainerConfigs(ShellSessionConfig{}, DefaultHostShellImage)
	if cfg.Image != DefaultHostShellImage {
		t.Errorf("Image = %q, want %q", cfg.Image, DefaultHostShellImage)
	}
	if !cfg.OpenStdin || !cfg.AttachStdin || !cfg.AttachStdout || !cfg.AttachStderr {
		t.Errorf("stdin/stdout/stderr attach flags not all true: %+v", cfg)
	}
	if !cfg.Tty {
		t.Errorf("Tty must be true for interactive shell")
	}
	if !host.Privileged {
		t.Errorf("HostConfig.Privileged must be true")
	}
	if string(host.PidMode) != "host" {
		t.Errorf("PidMode = %q, want host", host.PidMode)
	}
	if string(host.NetworkMode) != "host" {
		t.Errorf("NetworkMode = %q, want host", host.NetworkMode)
	}
	if string(host.IpcMode) != "host" {
		t.Errorf("IpcMode = %q, want host", host.IpcMode)
	}
	if string(host.UTSMode) != "host" {
		t.Errorf("UTSMode = %q, want host", host.UTSMode)
	}
	if !host.AutoRemove {
		t.Errorf("AutoRemove must be true so the sidecar is cleaned up on exit")
	}
	// SecurityOpt must disable apparmor + seccomp; otherwise nsenter
	// can't traverse PID 1's namespaces on a default docker host.
	gotSec := strings.Join(host.SecurityOpt, ",")
	if !strings.Contains(gotSec, "apparmor=unconfined") {
		t.Errorf("missing apparmor=unconfined in SecurityOpt: %v", host.SecurityOpt)
	}
	if !strings.Contains(gotSec, "seccomp=unconfined") {
		t.Errorf("missing seccomp=unconfined in SecurityOpt: %v", host.SecurityOpt)
	}
}

func TestBuildHostContainerConfigs_TermEnvSet(t *testing.T) {
	// Without TERM the shell can't render colour or do line editing —
	// makes for an awful operator UX.
	cfg, _ := buildHostContainerConfigs(ShellSessionConfig{}, DefaultHostShellImage)
	found := false
	for _, e := range cfg.Env {
		if e == "TERM=xterm-256color" {
			found = true
		}
	}
	if !found {
		t.Errorf("Env missing TERM=xterm-256color, got %v", cfg.Env)
	}
}

// --- Spawn via httptest-mocked docker daemon --------------------------------

// hostDockerStub stands up a fake daemon that captures the create body
// and serves attach as a hijack. recv channels let tests assert what the
// SDK sent without flakiness from network timing.
type hostDockerStub struct {
	srv          *httptest.Server
	createBodies chan []byte
	listCalls    *atomic.Int32
	pullCalls    *atomic.Int32
	startCalls   *atomic.Int32
	resizeCalls  chan struct{ width, height uint }
	imageList    []map[string]any
	waitStatus   int64
	failPull     bool
	failCreate   bool
	failAttach   bool
	failStart    bool
}

func newHostDockerStub(t *testing.T) *hostDockerStub {
	t.Helper()
	stub := &hostDockerStub{
		createBodies: make(chan []byte, 4),
		listCalls:    &atomic.Int32{},
		pullCalls:    &atomic.Int32{},
		startCalls:   &atomic.Int32{},
		resizeCalls:  make(chan struct{ width, height uint }, 4),
		imageList:    []map[string]any{},
		waitStatus:   0,
	}
	stub.srv = httptest.NewServer(http.HandlerFunc(stub.handle))
	return stub
}

func (s *hostDockerStub) handle(w http.ResponseWriter, r *http.Request) {
	switch {
	case strings.Contains(r.URL.Path, "/_ping"):
		w.Header().Set("Api-Version", "1.43")
		w.Header().Set("Docker-Experimental", "false")
		w.WriteHeader(http.StatusOK)
	case strings.HasSuffix(r.URL.Path, "/images/json"):
		s.listCalls.Add(1)
		_ = json.NewEncoder(w).Encode(s.imageList)
	case strings.HasSuffix(r.URL.Path, "/images/create"):
		s.pullCalls.Add(1)
		if s.failPull {
			w.WriteHeader(http.StatusInternalServerError)
			_, _ = w.Write([]byte(`{"message":"pull denied"}`))
			return
		}
		// Drain progress stream — the SDK does io.Copy(io.Discard).
		_, _ = w.Write([]byte(`{"status":"Pulling"}` + "\n"))
	case strings.HasSuffix(r.URL.Path, "/containers/create"):
		if s.failCreate {
			w.WriteHeader(http.StatusInternalServerError)
			_, _ = w.Write([]byte(`{"message":"out of resources"}`))
			return
		}
		body, _ := io.ReadAll(r.Body)
		s.createBodies <- body
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"Id":"sidecar-x","Warnings":[]}`))
	case strings.Contains(r.URL.Path, "/attach"):
		if s.failAttach {
			w.WriteHeader(http.StatusInternalServerError)
			_, _ = w.Write([]byte(`{"message":"attach denied"}`))
			return
		}
		// Hijack the connection so the SDK gets a real net.Conn back.
		// The SDK's setupHijackConn requires HTTP/1.1 101 Switching
		// Protocols (not 200 OK) — otherwise it errors with
		// "unable to upgrade to tcp, received N". Mirror the daemon's
		// real upgrade response.
		hj, ok := w.(http.Hijacker)
		if !ok {
			http.Error(w, "no hijack", http.StatusInternalServerError)
			return
		}
		conn, brw, err := hj.Hijack()
		if err != nil {
			return
		}
		_, _ = brw.WriteString("HTTP/1.1 101 UPGRADED\r\nContent-Type: application/vnd.docker.raw-stream\r\nConnection: Upgrade\r\nUpgrade: tcp\r\n\r\n")
		_ = brw.Flush()
		// Optionally write a banner so reads succeed.
		_, _ = conn.Write([]byte("ready\n"))
		// Leave the connection open until tests close it.
	case strings.Contains(r.URL.Path, "/start"):
		s.startCalls.Add(1)
		if s.failStart {
			w.WriteHeader(http.StatusInternalServerError)
			_, _ = w.Write([]byte(`{"message":"start denied"}`))
			return
		}
		w.WriteHeader(http.StatusNoContent)
	case strings.Contains(r.URL.Path, "/resize"):
		// Parse w/h query params and record.
		q := r.URL.Query()
		var wv, hv uint
		fmt.Sscanf(q.Get("w"), "%d", &wv)
		fmt.Sscanf(q.Get("h"), "%d", &hv)
		select {
		case s.resizeCalls <- struct{ width, height uint }{wv, hv}:
		default:
		}
		w.WriteHeader(http.StatusOK)
	case strings.Contains(r.URL.Path, "/wait"):
		// JSON: {StatusCode: N}
		_ = json.NewEncoder(w).Encode(map[string]int64{"StatusCode": s.waitStatus})
	case strings.Contains(r.URL.Path, "/remove") || r.Method == "DELETE":
		w.WriteHeader(http.StatusNoContent)
	default:
		w.WriteHeader(http.StatusNotFound)
	}
}

func (s *hostDockerStub) client(t *testing.T) *dclient.Client {
	t.Helper()
	// SDK's hijack dialer can't dial proto="http" via net.Dial — only
	// "tcp", "unix", and "npipe" are accepted by net.Dialer.DialContext.
	// Translate the httptest URL into a tcp:// host so the SDK uses the
	// underlying TCP socket for both regular RPCs and hijacked attaches.
	hostURL := strings.Replace(s.srv.URL, "http://", "tcp://", 1)
	cli, err := dclient.NewClientWithOpts(
		dclient.WithHost(hostURL),
		dclient.WithVersion("1.43"),
	)
	if err != nil {
		t.Fatalf("NewClientWithOpts: %v", err)
	}
	return cli
}

// --- Spawn happy path: inspect create body ---------------------------------

func TestHostBackend_SpawnSendsExpectedContainerSpec(t *testing.T) {
	stub := newHostDockerStub(t)
	defer stub.srv.Close()
	cli := stub.client(t)
	b := NewHostShellBackend(cli)
	rwc, err := b.Spawn(context.Background(), ShellSessionConfig{User: "ubuntu", Shell: "/bin/bash"})
	if err != nil {
		t.Fatalf("Spawn: %v", err)
	}
	defer rwc.Close()

	// Pull once on cache miss (empty image list).
	if stub.pullCalls.Load() != 1 {
		t.Errorf("expected 1 pull, got %d", stub.pullCalls.Load())
	}

	var createBody []byte
	select {
	case createBody = <-stub.createBodies:
	case <-time.After(2 * time.Second):
		t.Fatalf("no create body captured")
	}

	// Unmarshal the SDK's combined Config+HostConfig payload.
	var spec struct {
		Image        string
		OpenStdin    bool
		AttachStdin  bool
		AttachStdout bool
		AttachStderr bool
		Tty          bool
		Env          []string
		Cmd          []string
		HostConfig   struct {
			Privileged  bool
			PidMode     string
			NetworkMode string
			IpcMode     string
			UTSMode     string
			AutoRemove  bool
			SecurityOpt []string
		}
	}
	if err := json.Unmarshal(createBody, &spec); err != nil {
		t.Fatalf("decode create body: %v\nbody=%s", err, string(createBody))
	}
	if spec.Image != DefaultHostShellImage {
		t.Errorf("Image = %q, want %q", spec.Image, DefaultHostShellImage)
	}
	if !spec.OpenStdin || !spec.AttachStdin || !spec.AttachStdout || !spec.AttachStderr {
		t.Errorf("attach flags wrong: %+v", spec)
	}
	if !spec.Tty {
		t.Errorf("Tty must be true")
	}
	if !spec.HostConfig.Privileged {
		t.Errorf("HostConfig.Privileged must be true")
	}
	if spec.HostConfig.PidMode != "host" {
		t.Errorf("PidMode = %q, want host", spec.HostConfig.PidMode)
	}
	if spec.HostConfig.NetworkMode != "host" {
		t.Errorf("NetworkMode = %q, want host", spec.HostConfig.NetworkMode)
	}
	if spec.HostConfig.IpcMode != "host" {
		t.Errorf("IpcMode = %q, want host", spec.HostConfig.IpcMode)
	}
	if spec.HostConfig.UTSMode != "host" {
		t.Errorf("UTSMode = %q, want host", spec.HostConfig.UTSMode)
	}
	if !spec.HostConfig.AutoRemove {
		t.Errorf("AutoRemove must be true")
	}
	if spec.Cmd[0] != "nsenter" {
		t.Errorf("Cmd[0] = %q, want nsenter", spec.Cmd[0])
	}
	if stub.startCalls.Load() == 0 {
		t.Errorf("expected ContainerStart, got 0 calls")
	}
}

// --- Image cache: list says present → no pull ------------------------------

func TestHostBackend_SkipsPullWhenImageCached(t *testing.T) {
	stub := newHostDockerStub(t)
	defer stub.srv.Close()
	stub.imageList = []map[string]any{
		{"Id": "sha256:1", "RepoTags": []string{DefaultHostShellImage, "ubuntu:latest"}},
	}
	cli := stub.client(t)
	b := NewHostShellBackend(cli)
	rwc, err := b.Spawn(context.Background(), ShellSessionConfig{})
	if err != nil {
		t.Fatalf("Spawn: %v", err)
	}
	defer rwc.Close()
	if stub.pullCalls.Load() != 0 {
		t.Errorf("expected 0 pulls (image cached), got %d", stub.pullCalls.Load())
	}
}

// --- Resize forwards to ContainerResize ------------------------------------

func TestHostBackend_ResizeForwardsToDaemon(t *testing.T) {
	stub := newHostDockerStub(t)
	defer stub.srv.Close()
	cli := stub.client(t)
	b := NewHostShellBackend(cli).(*hostBackend)
	rwc, err := b.Spawn(context.Background(), ShellSessionConfig{})
	if err != nil {
		t.Fatalf("Spawn: %v", err)
	}
	defer rwc.Close()
	if err := b.Resize(120, 40); err != nil {
		t.Fatalf("Resize: %v", err)
	}
	select {
	case got := <-stub.resizeCalls:
		if got.width != 120 || got.height != 40 {
			t.Errorf("resize: got w=%d h=%d, want 120/40", got.width, got.height)
		}
	case <-time.After(2 * time.Second):
		t.Fatalf("daemon did not see resize call")
	}
}

func TestHostBackend_ResizeNoOpBeforeSpawn(t *testing.T) {
	// Before Spawn, containerID is unset — Resize must return nil without
	// dialling out. Mirrors dockerShellBackend's defensive behaviour.
	b := &hostBackend{}
	if err := b.Resize(80, 24); err != nil {
		t.Errorf("expected nil err on uninitialised backend, got %v", err)
	}
}

// --- WaitExit pumps the exit code -----------------------------------------

func TestHostBackend_WaitExitReturnsStatusCode(t *testing.T) {
	stub := newHostDockerStub(t)
	stub.waitStatus = 137 // OOMkilled is the common case in prod
	defer stub.srv.Close()
	cli := stub.client(t)
	b := NewHostShellBackend(cli).(*hostBackend)
	rwc, err := b.Spawn(context.Background(), ShellSessionConfig{})
	if err != nil {
		t.Fatalf("Spawn: %v", err)
	}
	defer rwc.Close()
	code, err := b.WaitExit(context.Background())
	if err != nil {
		t.Fatalf("WaitExit: %v", err)
	}
	if code != 137 {
		t.Errorf("WaitExit code = %d, want 137", code)
	}
}

func TestHostBackend_WaitExitNilBackendIsClean(t *testing.T) {
	b := &hostBackend{}
	code, err := b.WaitExit(context.Background())
	if err != nil {
		t.Errorf("expected nil err on uninitialised backend, got %v", err)
	}
	if code != -1 {
		t.Errorf("expected -1, got %d", code)
	}
}

// TestHostBackend_WaitExitErrorPath exercises the errC branch of
// ContainerWait. We get there by pointing at a daemon that 500s on
// /wait, which the SDK surfaces as a non-nil err on the errC channel.
func TestHostBackend_WaitExitErrorPath(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.Contains(r.URL.Path, "/_ping"):
			w.Header().Set("Api-Version", "1.43")
			w.WriteHeader(http.StatusOK)
		case strings.Contains(r.URL.Path, "/wait"):
			w.WriteHeader(http.StatusInternalServerError)
			_, _ = w.Write([]byte(`{"message":"wait failed"}`))
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer srv.Close()
	hostURL := strings.Replace(srv.URL, "http://", "tcp://", 1)
	cli, err := dclient.NewClientWithOpts(dclient.WithHost(hostURL), dclient.WithVersion("1.43"))
	if err != nil {
		t.Fatalf("NewClientWithOpts: %v", err)
	}
	b := &hostBackend{cli: cli, containerID: "x"}
	code, err := b.WaitExit(context.Background())
	if err == nil {
		t.Fatalf("expected error from /wait 500")
	}
	if code != -1 {
		t.Errorf("expected -1 on error, got %d", code)
	}
}

// TestHostBackend_WaitExitContextDeadline forces the deadline branch by
// invoking WaitExit with a context that's already cancelled before the
// daemon responds. The SDK propagates the cancellation through errC.
func TestHostBackend_WaitExitContextDeadline(t *testing.T) {
	// Daemon that intentionally hangs on /wait so we hit the deadline.
	wait := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.Contains(r.URL.Path, "/_ping"):
			w.Header().Set("Api-Version", "1.43")
			w.WriteHeader(http.StatusOK)
		case strings.Contains(r.URL.Path, "/wait"):
			<-wait
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer srv.Close()
	defer close(wait)
	hostURL := strings.Replace(srv.URL, "http://", "tcp://", 1)
	cli, err := dclient.NewClientWithOpts(dclient.WithHost(hostURL), dclient.WithVersion("1.43"))
	if err != nil {
		t.Fatalf("NewClientWithOpts: %v", err)
	}
	b := &hostBackend{cli: cli, containerID: "x"}
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	code, err := b.WaitExit(ctx)
	if err == nil {
		t.Fatalf("expected ctx-deadline error, got nil (code=%d)", code)
	}
	if code != -1 {
		t.Errorf("expected -1 on deadline, got %d", code)
	}
}

// TestHostBackend_EnsureImage_ListErrorFallsThroughToPull — when
// ImageList errors (daemon refusing /images/json), ensureImage must not
// short-circuit; instead it must attempt the pull. Verifies the
// defensive fall-through path.
func TestHostBackend_EnsureImage_ListErrorFallsThroughToPull(t *testing.T) {
	stub := newHostDockerStub(t)
	defer stub.srv.Close()
	// Inject an error path for ImageList by swapping the handler.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.Contains(r.URL.Path, "/_ping"):
			w.Header().Set("Api-Version", "1.43")
			w.WriteHeader(http.StatusOK)
		case strings.HasSuffix(r.URL.Path, "/images/json"):
			w.WriteHeader(http.StatusInternalServerError)
			_, _ = w.Write([]byte(`{"message":"list denied"}`))
		case strings.HasSuffix(r.URL.Path, "/images/create"):
			// Pull succeeds — exercise the fall-through to pull.
			stub.pullCalls.Add(1)
			_, _ = w.Write([]byte(`{"status":"Pulling"}` + "\n"))
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer srv.Close()
	hostURL := strings.Replace(srv.URL, "http://", "tcp://", 1)
	cli, err := dclient.NewClientWithOpts(dclient.WithHost(hostURL), dclient.WithVersion("1.43"))
	if err != nil {
		t.Fatalf("NewClientWithOpts: %v", err)
	}
	b := &hostBackend{cli: cli, image: DefaultHostShellImage}
	if err := b.ensureImage(context.Background()); err != nil {
		t.Fatalf("ensureImage: %v", err)
	}
	if stub.pullCalls.Load() != 1 {
		t.Errorf("expected pull to be invoked once after list failure, got %d", stub.pullCalls.Load())
	}
}

// TestHostBackend_EnsureImage_NilClient is the defensive guard the
// production path hits if a misconfigured caller forgets to wire the
// docker client.
func TestHostBackend_EnsureImage_NilClient(t *testing.T) {
	b := &hostBackend{}
	if err := b.ensureImage(context.Background()); err == nil {
		t.Errorf("expected error on nil docker client")
	}
}

// --- Error paths: surface as Spawn errors so OnError fires --------------

func TestHostBackend_PullFailure(t *testing.T) {
	stub := newHostDockerStub(t)
	stub.failPull = true
	defer stub.srv.Close()
	cli := stub.client(t)
	b := NewHostShellBackend(cli)
	_, err := b.Spawn(context.Background(), ShellSessionConfig{})
	if err == nil {
		t.Fatalf("expected error from pull failure")
	}
	if !strings.Contains(err.Error(), "host shell image") {
		t.Errorf("expected 'host shell image' prefix, got %v", err)
	}
}

func TestHostBackend_CreateFailure(t *testing.T) {
	stub := newHostDockerStub(t)
	stub.failCreate = true
	stub.imageList = []map[string]any{
		{"Id": "x", "RepoTags": []string{DefaultHostShellImage}},
	}
	defer stub.srv.Close()
	cli := stub.client(t)
	b := NewHostShellBackend(cli)
	_, err := b.Spawn(context.Background(), ShellSessionConfig{})
	if err == nil {
		t.Fatalf("expected error from create failure")
	}
	if !strings.Contains(err.Error(), "host shell create") {
		t.Errorf("expected 'host shell create' prefix, got %v", err)
	}
}

func TestHostBackend_AttachFailure(t *testing.T) {
	stub := newHostDockerStub(t)
	stub.failAttach = true
	stub.imageList = []map[string]any{
		{"Id": "x", "RepoTags": []string{DefaultHostShellImage}},
	}
	defer stub.srv.Close()
	cli := stub.client(t)
	b := NewHostShellBackend(cli)
	_, err := b.Spawn(context.Background(), ShellSessionConfig{})
	if err == nil {
		t.Fatalf("expected error from attach failure")
	}
	if !strings.Contains(err.Error(), "host shell attach") {
		t.Errorf("expected 'host shell attach' prefix, got %v", err)
	}
}

func TestHostBackend_StartFailure(t *testing.T) {
	stub := newHostDockerStub(t)
	stub.failStart = true
	stub.imageList = []map[string]any{
		{"Id": "x", "RepoTags": []string{DefaultHostShellImage}},
	}
	defer stub.srv.Close()
	cli := stub.client(t)
	b := NewHostShellBackend(cli)
	_, err := b.Spawn(context.Background(), ShellSessionConfig{})
	if err == nil {
		t.Fatalf("expected error from start failure")
	}
	if !strings.Contains(err.Error(), "host shell start") {
		t.Errorf("expected 'host shell start' prefix, got %v", err)
	}
}

func TestHostBackend_NilClient(t *testing.T) {
	b := NewHostShellBackend(nil)
	_, err := b.Spawn(context.Background(), ShellSessionConfig{})
	if err == nil {
		t.Fatalf("expected error from nil docker client")
	}
	if !strings.Contains(err.Error(), "no docker client") {
		t.Errorf("expected 'no docker client' in error, got %v", err)
	}
}

// --- StartHostShell + SetHostShellBackend registration --------------------

func TestSetHostShellBackend_InstallsDefaultHostSpawn(t *testing.T) {
	old := DefaultHostSpawn
	defer func() { DefaultHostSpawn = old }()
	stub := newHostDockerStub(t)
	defer stub.srv.Close()
	cli := stub.client(t)
	SetHostShellBackend(cli)
	if DefaultHostSpawn == nil {
		t.Fatalf("DefaultHostSpawn must be installed")
	}
	backend, err := DefaultHostSpawn(ShellSessionConfig{})
	if err != nil {
		t.Fatalf("DefaultHostSpawn: %v", err)
	}
	if backend == nil {
		t.Fatalf("expected non-nil backend")
	}
}

func TestStartHostShell_UsesHostSpawn(t *testing.T) {
	old := DefaultHostSpawn
	defer func() { DefaultHostSpawn = old }()
	called := false
	DefaultHostSpawn = func(cfg ShellSessionConfig) (ShellBackend, error) {
		called = true
		return &noOpShellBackend{}, nil
	}
	sess, err := StartHostShell(context.Background(), ShellSessionConfig{})
	if err != nil {
		t.Fatalf("StartHostShell: %v", err)
	}
	if !called {
		t.Errorf("StartHostShell must invoke DefaultHostSpawn")
	}
	_ = sess.Close()
}

func TestStartHostShell_NilSpawnIsErrNoBackend(t *testing.T) {
	old := DefaultHostSpawn
	defer func() { DefaultHostSpawn = old }()
	DefaultHostSpawn = nil
	_, err := StartHostShell(context.Background(), ShellSessionConfig{})
	if err == nil {
		t.Fatalf("expected ErrNoBackend")
	}
}

// noOpShellBackend is a minimal ShellBackend that's safe to use in tests
// that don't care about the data plane — Spawn returns a closed pipe.
type noOpShellBackend struct {
	closeOnce sync.Once
}

func (n *noOpShellBackend) Spawn(ctx context.Context, cfg ShellSessionConfig) (io.ReadWriteCloser, error) {
	// Return a never-EOF pipe whose Close terminates the read pump.
	pr, pw := pipePair()
	return &pipeRWC{r: pr, w: pw}, nil
}
func (n *noOpShellBackend) Resize(cols, rows uint16) error          { return nil }
func (n *noOpShellBackend) WaitExit(ctx context.Context) (int, error) { return 0, nil }

// pipePair / pipeRWC: cheap io.ReadWriteCloser using two ends of a TCP
// pair-ish abstraction. Implemented with net.Pipe so reads block until
// the writer end is closed, matching real PTY semantics.
func pipePair() (net.Conn, net.Conn) { return net.Pipe() }

type pipeRWC struct {
	r       net.Conn
	w       net.Conn
	cmu     sync.Mutex
	cl      bool
}

func (p *pipeRWC) Read(b []byte) (int, error)  { return p.r.Read(b) }
func (p *pipeRWC) Write(b []byte) (int, error) { return p.w.Write(b) }
func (p *pipeRWC) Close() error {
	p.cmu.Lock()
	defer p.cmu.Unlock()
	if p.cl {
		return nil
	}
	p.cl = true
	_ = p.r.Close()
	_ = p.w.Close()
	return nil
}

// --- hostHijackRWC -------------------------------------------------------

func TestHostHijackRWC_NilReaderReturnsEOF(t *testing.T) {
	h := &hostHijackRWC{}
	buf := make([]byte, 1)
	if _, err := h.Read(buf); err != io.EOF {
		t.Errorf("expected EOF on nil reader, got %v", err)
	}
}

func TestHostHijackRWC_NilConnErrorsOnWrite(t *testing.T) {
	h := &hostHijackRWC{}
	if _, err := h.Write([]byte("x")); err == nil {
		t.Errorf("expected error writing to nil conn")
	}
}

func TestHostHijackRWC_CloseIdempotentWithoutConn(t *testing.T) {
	h := &hostHijackRWC{}
	if err := h.Close(); err != nil {
		t.Errorf("expected nil err closing zero-value hostHijackRWC, got %v", err)
	}
}

func TestHostHijackRWC_PassThrough(t *testing.T) {
	a, b := net.Pipe()
	defer a.Close()
	defer b.Close()
	h := &hostHijackRWC{conn: a, reader: a, tty: true}
	go func() {
		_, _ = b.Write([]byte("banner\n"))
	}()
	buf := make([]byte, 7)
	n, err := h.Read(buf)
	if err != nil || string(buf[:n]) != "banner\n" {
		t.Errorf("read: n=%d err=%v buf=%q", n, err, buf[:n])
	}
	// Best-effort write — the other end is a pipe, may buffer or accept.
	go func() {
		_, _ = io.ReadAll(b)
	}()
	if _, err := h.Write([]byte("input")); err != nil {
		t.Logf("write: %v (acceptable on pipe shutdown)", err)
	}
	_ = h.Close()
}

// --- Defence-in-depth on the container.* type system --------------------

func TestHostContainerConfig_TypesAreDockerNamespaceTypes(t *testing.T) {
	// Guard against an SDK upgrade that swaps the type out from under us;
	// the SDK's container.Config / HostConfig are what the daemon expects
	// over the wire, so we don't want to be passing a custom shape.
	cfg, host := buildHostContainerConfigs(ShellSessionConfig{}, "x")
	var _ *container.Config = cfg
	var _ *container.HostConfig = host
}
