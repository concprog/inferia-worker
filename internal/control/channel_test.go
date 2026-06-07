package control

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"nhooyr.io/websocket"
)

type fakeDispatcher struct {
	mu              sync.Mutex
	loaded          map[string]string // deploymentID → endpoint
	loadErr         error
	unloadErr       error
	loadCalls       int32
	unloadCalls     int32
	heartbeatModels []string
}

func newFakeDispatcher() *fakeDispatcher {
	return &fakeDispatcher{loaded: map[string]string{}}
}

func (f *fakeDispatcher) LoadModel(ctx context.Context, b LoadModelBody) (string, error) {
	atomic.AddInt32(&f.loadCalls, 1)
	if f.loadErr != nil {
		return "", f.loadErr
	}
	f.mu.Lock()
	f.loaded[b.DeploymentID] = "http://endpoint/" + b.DeploymentID
	f.mu.Unlock()
	return f.loaded[b.DeploymentID], nil
}

func (f *fakeDispatcher) UnloadModel(ctx context.Context, b UnloadModelBody) error {
	atomic.AddInt32(&f.unloadCalls, 1)
	if f.unloadErr != nil {
		return f.unloadErr
	}
	f.mu.Lock()
	delete(f.loaded, b.DeploymentID)
	f.mu.Unlock()
	return nil
}

func (f *fakeDispatcher) HeartbeatSnapshot() HeartbeatBody {
	f.mu.Lock()
	defer f.mu.Unlock()
	models := make([]string, 0, len(f.loaded))
	for k := range f.loaded {
		models = append(models, k)
	}
	f.heartbeatModels = models
	return HeartbeatBody{
		Used:         map[string]string{"cpu": "0.1"},
		LoadedModels: models,
	}
}

// fakeCPServer is a minimal control-plane WS endpoint for tests.
type fakeCPServer struct {
	srv          *httptest.Server
	mu           sync.Mutex
	receivedMsgs []Envelope
	authHeaders  []string
	close        bool
}

func newFakeCPServer(handler func(ctx context.Context, conn *websocket.Conn, server *fakeCPServer)) *fakeCPServer {
	cp := &fakeCPServer{}
	cp.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		cp.mu.Lock()
		cp.authHeaders = append(cp.authHeaders, r.Header.Get("Authorization"))
		cp.mu.Unlock()
		if r.URL.Path != "/v1/workers/channel" {
			http.Error(w, "wrong path", 404)
			return
		}
		c, err := websocket.Accept(w, r, &websocket.AcceptOptions{InsecureSkipVerify: true})
		if err != nil {
			return
		}
		defer c.Close(websocket.StatusInternalError, "")
		ctx := r.Context()
		// Send Hello.
		_ = writeJSON(ctx, c, Envelope{Type: MsgHello, ID: "hello-1", Body: HelloBody{ChannelID: "chan-1"}})
		handler(ctx, c, cp)
	}))
	return cp
}

func writeJSON(ctx context.Context, c *websocket.Conn, env Envelope) error {
	data, _ := json.Marshal(env)
	return c.Write(ctx, websocket.MessageText, data)
}

func TestChannel_RunUntilCtxCancel(t *testing.T) {
	disp := newFakeDispatcher()

	cp := newFakeCPServer(func(ctx context.Context, c *websocket.Conn, s *fakeCPServer) {
		// Read heartbeats until ctx ends or peer closes.
		for {
			_, data, err := c.Read(ctx)
			if err != nil {
				return
			}
			var env Envelope
			_ = json.Unmarshal(data, &env)
			s.mu.Lock()
			s.receivedMsgs = append(s.receivedMsgs, env)
			s.mu.Unlock()
		}
	})
	defer cp.srv.Close()

	ch := &Channel{
		ChannelURL:        wsURL(cp.srv.URL) + "/v1/workers/channel",
		Token:             func() string { return "jwt-x" },
		HeartbeatInterval: 30 * time.Millisecond,
		Dispatcher:        disp,
		DedupTTL:          100 * time.Millisecond,
	}
	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()
	_ = ch.Run(ctx)

	cp.mu.Lock()
	defer cp.mu.Unlock()
	if len(cp.authHeaders) == 0 || cp.authHeaders[0] != "Bearer jwt-x" {
		t.Errorf("auth headers: %v", cp.authHeaders)
	}
	// At least one heartbeat (ticker fires after first interval).
	hbCount := 0
	for _, m := range cp.receivedMsgs {
		if m.Type == MsgHeartbeat {
			hbCount++
		}
	}
	if hbCount == 0 {
		t.Errorf("expected ≥1 heartbeat, got 0; msgs=%v", cp.receivedMsgs)
	}
}

func TestChannel_LoadModelDispatched(t *testing.T) {
	disp := newFakeDispatcher()
	done := make(chan struct{})

	cp := newFakeCPServer(func(ctx context.Context, c *websocket.Conn, s *fakeCPServer) {
		// Send a LoadModel command; expect a CommandResult back.
		cmd := Envelope{
			Type: MsgLoadModel, ID: "cmd-1",
			Body: LoadModelBody{
				DeploymentID: "dep-1", Recipe: "vllm",
				Model:      ModelRef{ArtifactURI: "hf://o/m"},
				GPUIndices: []int{0}, Port: 19000,
			},
		}
		_ = writeJSON(ctx, c, cmd)

		// Read response.
		_, data, err := c.Read(ctx)
		if err != nil {
			return
		}
		var env Envelope
		_ = json.Unmarshal(data, &env)
		for env.Type != MsgCommandResult {
			_, data, err = c.Read(ctx)
			if err != nil {
				return
			}
			_ = json.Unmarshal(data, &env)
		}
		s.mu.Lock()
		s.receivedMsgs = append(s.receivedMsgs, env)
		s.mu.Unlock()
		close(done)
	})
	defer cp.srv.Close()

	ch := &Channel{
		ChannelURL:        wsURL(cp.srv.URL) + "/v1/workers/channel",
		Token:             func() string { return "jwt-x" },
		HeartbeatInterval: 1 * time.Hour, // suppress heartbeats for clarity
		Dispatcher:        disp,
		DedupTTL:          100 * time.Millisecond,
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	go ch.Run(ctx)

	select {
	case <-done:
	case <-ctx.Done():
		t.Fatalf("did not receive CommandResult in time")
	}
	if atomic.LoadInt32(&disp.loadCalls) != 1 {
		t.Errorf("LoadModel called %d times", disp.loadCalls)
	}
	cp.mu.Lock()
	defer cp.mu.Unlock()
	var result *CommandResultBody
	for _, m := range cp.receivedMsgs {
		if m.Type == MsgCommandResult {
			b, _ := json.Marshal(m.Body)
			var r CommandResultBody
			_ = json.Unmarshal(b, &r)
			result = &r
			break
		}
	}
	if result == nil || result.Status != "ok" || result.InReplyTo != "cmd-1" {
		t.Errorf("bad CommandResult: %+v", result)
	}
}

func TestChannel_DedupOnReplay(t *testing.T) {
	disp := newFakeDispatcher()

	cp := newFakeCPServer(func(ctx context.Context, c *websocket.Conn, s *fakeCPServer) {
		cmd := Envelope{
			Type: MsgUnloadModel, ID: "cmd-uniq",
			Body: UnloadModelBody{DeploymentID: "dep-x"},
		}
		_ = writeJSON(ctx, c, cmd)
		_ = writeJSON(ctx, c, cmd) // replay
		time.Sleep(120 * time.Millisecond)
	})
	defer cp.srv.Close()

	ch := &Channel{
		ChannelURL:        wsURL(cp.srv.URL) + "/v1/workers/channel",
		Token:             func() string { return "jwt-x" },
		HeartbeatInterval: 1 * time.Hour,
		Dispatcher:        disp,
		DedupTTL:          500 * time.Millisecond,
	}
	ctx, cancel := context.WithTimeout(context.Background(), 500*time.Millisecond)
	defer cancel()
	_ = ch.Run(ctx)

	if got := atomic.LoadInt32(&disp.unloadCalls); got != 1 {
		t.Errorf("replay should not re-execute, got %d", got)
	}
}

func TestChannel_DialFailure(t *testing.T) {
	ch := &Channel{
		ChannelURL:        "ws://127.0.0.1:1/v1/workers/channel",
		Token:             func() string { return "jwt" },
		HeartbeatInterval: 1 * time.Hour,
		Dispatcher:        newFakeDispatcher(),
		DedupTTL:          time.Minute,
	}
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	err := ch.connectOnce(ctx)
	if err == nil {
		t.Errorf("expected dial error")
	}
}

func TestBackoffPolicy(t *testing.T) {
	cases := []struct {
		attempt  int
		min, max time.Duration
	}{
		{1, 800 * time.Millisecond, 1200 * time.Millisecond},
		{2, 1500 * time.Millisecond, 2500 * time.Millisecond},
		{3, 3 * time.Second, 5 * time.Second},
		{10, 23 * time.Second, 37 * time.Second}, // capped near 30s, ±20% jitter
		{50, 23 * time.Second, 37 * time.Second},
	}
	for _, tc := range cases {
		d := backoffFor(tc.attempt)
		if d < tc.min || d > tc.max {
			t.Errorf("attempt=%d: got %v, want in [%v, %v]", tc.attempt, d, tc.min, tc.max)
		}
	}
}

func wsURL(httpURL string) string {
	if len(httpURL) > 7 && httpURL[:7] == "http://" {
		return "ws://" + httpURL[7:]
	}
	if len(httpURL) > 8 && httpURL[:8] == "https://" {
		return "wss://" + httpURL[8:]
	}
	return httpURL
}

func TestHello_IncludesCloudEnv(t *testing.T) {
	// Confirms HelloBody carries the four cloud-env fields when constructed
	// with runtime values. Marshal to JSON and assert all fields appear.
	body := HelloBody{
		RuntimeEnv:       "aws-ec2",
		InstanceID:       "i-hello",
		Region:           "us-east-1",
		AvailabilityZone: "us-east-1c",
	}
	data, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	s := string(data)
	for _, want := range []string{
		`"runtime_env":"aws-ec2"`,
		`"instance_id":"i-hello"`,
		`"region":"us-east-1"`,
		`"availability_zone":"us-east-1c"`,
	} {
		if !strings.Contains(s, want) {
			t.Errorf("HelloBody missing %q: %s", want, s)
		}
	}
}

func TestHello_OmitsCloudEnvWhenLocal(t *testing.T) {
	// A zero HelloBody (worker running locally with no cloud-env fields set)
	// must not emit runtime_env, instance_id, region, availability_zone, or
	// server_time. server_time uses *time.Time so nil is omitempty-friendly.
	body := HelloBody{}
	data, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	s := string(data)
	for _, unwanted := range []string{`"runtime_env"`, `"instance_id"`, `"region"`, `"availability_zone"`} {
		if strings.Contains(s, unwanted) {
			t.Errorf("expected %s omitted, got %s", unwanted, s)
		}
	}
	// server_time must also be omitted — nil *time.Time satisfies omitempty.
	if strings.Contains(s, `"server_time"`) {
		t.Errorf("server_time should be omitted on zero HelloBody, got %s", s)
	}
}
