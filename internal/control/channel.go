package control

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math/rand"
	"net/http"
	"time"

	"nhooyr.io/websocket"
)

// Dispatcher is the worker's runtime-shaped surface the channel calls into for
// LoadModel/UnloadModel and to snapshot Heartbeat state. The runtime package
// provides the concrete implementation.
type Dispatcher interface {
	LoadModel(ctx context.Context, body LoadModelBody) (endpointURL string, err error)
	UnloadModel(ctx context.Context, body UnloadModelBody) error
	HeartbeatSnapshot() HeartbeatBody
}

// Channel manages a single long-lived WebSocket connection. Run() loops
// forever, reconnecting with exponential backoff, until ctx is cancelled.
type Channel struct {
	ChannelURL        string
	Token             func() string // freshly fetched every connect attempt
	HeartbeatInterval time.Duration
	Dispatcher        Dispatcher
	DedupTTL          time.Duration

	// internal
	dedup *dedup
}

// Run blocks until ctx is cancelled, dialing/reconnecting in a loop.
func (ch *Channel) Run(ctx context.Context) error {
	if ch.dedup == nil {
		ttl := ch.DedupTTL
		if ttl == 0 {
			ttl = 5 * time.Minute
		}
		ch.dedup = newDedup(ttl)
	}
	attempt := 0
	for {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		err := ch.connectOnce(ctx)
		if ctx.Err() != nil {
			return ctx.Err()
		}
		attempt++
		wait := backoffFor(attempt)
		select {
		case <-time.After(wait):
		case <-ctx.Done():
			return ctx.Err()
		}
		_ = err
	}
}

// connectOnce dials, then runs read/write loops until either side closes.
func (ch *Channel) connectOnce(ctx context.Context) error {
	header := http.Header{}
	header.Set("Authorization", "Bearer "+ch.Token())
	conn, _, err := websocket.Dial(ctx, ch.ChannelURL, &websocket.DialOptions{
		HTTPHeader: header,
	})
	if err != nil {
		return fmt.Errorf("dial: %w", err)
	}
	defer conn.Close(websocket.StatusNormalClosure, "")

	// Run read + write concurrently. Cancel when either returns.
	innerCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	readErr := make(chan error, 1)
	writeErr := make(chan error, 1)

	go func() { readErr <- ch.readLoop(innerCtx, conn) }()
	go func() { writeErr <- ch.heartbeatLoop(innerCtx, conn) }()

	select {
	case err := <-readErr:
		return err
	case err := <-writeErr:
		return err
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (ch *Channel) readLoop(ctx context.Context, conn *websocket.Conn) error {
	for {
		_, data, err := conn.Read(ctx)
		if err != nil {
			return err
		}
		var env Envelope
		if err := json.Unmarshal(data, &env); err != nil {
			continue // skip malformed frames
		}
		// Dispatch handle() in a goroutine so a long-running LoadModel
		// (image pull, container start, readiness wait) doesn't block the
		// read loop. Blocking the loop starves WS keepalive pongs and the
		// connection times out client-side.
		go ch.handle(ctx, conn, env)
	}
}

func (ch *Channel) heartbeatLoop(ctx context.Context, conn *websocket.Conn) error {
	interval := ch.HeartbeatInterval
	if interval == 0 {
		interval = 5 * time.Second
	}
	tick := time.NewTicker(interval)
	defer tick.Stop()
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-tick.C:
			snap := ch.Dispatcher.HeartbeatSnapshot()
			env := Envelope{
				Type: MsgHeartbeat,
				ID:   newID(),
				TS:   time.Now().UTC().Format(time.RFC3339Nano),
				Body: snap,
			}
			if err := writeEnvelope(ctx, conn, env); err != nil {
				return err
			}
		}
	}
}

// handle routes one inbound envelope.
func (ch *Channel) handle(ctx context.Context, conn *websocket.Conn, env Envelope) {
	switch env.Type {
	case MsgHello, MsgPing:
		return
	case MsgLoadModel:
		var body LoadModelBody
		if err := remarshal(env.Body, &body); err != nil {
			ch.replyFailed(ctx, conn, env.ID, "bad LoadModel body")
			return
		}
		result, _ := ch.dedup.Run(env.ID, func() CommandResultBody {
			endpoint, err := ch.Dispatcher.LoadModel(ctx, body)
			if err != nil {
				return CommandResultBody{InReplyTo: env.ID, Status: "failed", Detail: err.Error()}
			}
			return CommandResultBody{InReplyTo: env.ID, Status: "ok", EndpointURL: endpoint}
		})
		ch.reply(ctx, conn, result)
	case MsgUnloadModel:
		var body UnloadModelBody
		if err := remarshal(env.Body, &body); err != nil {
			ch.replyFailed(ctx, conn, env.ID, "bad UnloadModel body")
			return
		}
		result, _ := ch.dedup.Run(env.ID, func() CommandResultBody {
			if err := ch.Dispatcher.UnloadModel(ctx, body); err != nil {
				return CommandResultBody{InReplyTo: env.ID, Status: "failed", Detail: err.Error()}
			}
			return CommandResultBody{InReplyTo: env.ID, Status: "ok"}
		})
		ch.reply(ctx, conn, result)
	}
}

func (ch *Channel) reply(ctx context.Context, conn *websocket.Conn, body CommandResultBody) {
	env := Envelope{
		Type: MsgCommandResult,
		ID:   newID(),
		TS:   time.Now().UTC().Format(time.RFC3339Nano),
		Body: body,
	}
	_ = writeEnvelope(ctx, conn, env)
}

func (ch *Channel) replyFailed(ctx context.Context, conn *websocket.Conn, inReplyTo, detail string) {
	ch.reply(ctx, conn, CommandResultBody{InReplyTo: inReplyTo, Status: "failed", Detail: detail})
}

// writeEnvelope marshals + writes one frame. Bounded write deadline.
func writeEnvelope(ctx context.Context, conn *websocket.Conn, env Envelope) error {
	data, err := json.Marshal(env)
	if err != nil {
		return err
	}
	writeCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	return conn.Write(writeCtx, websocket.MessageText, data)
}

// remarshal converts an arbitrary `any` (json.Unmarshal yields map[string]any
// for object bodies) into a concrete typed struct.
func remarshal(in any, out any) error {
	data, err := json.Marshal(in)
	if err != nil {
		return err
	}
	return json.Unmarshal(data, out)
}

// backoffFor returns the reconnect delay for the n-th attempt. Exponential
// with jitter, capped at ~30s. Attempt 1 ≈ 1s, attempt 2 ≈ 2s, attempt 3 ≈ 4s, …
func backoffFor(attempt int) time.Duration {
	if attempt < 1 {
		attempt = 1
	}
	base := time.Second * (1 << uint(minInt(attempt-1, 5))) // 1, 2, 4, 8, 16, 32
	if base > 30*time.Second {
		base = 30 * time.Second
	}
	// ±20% jitter.
	jitter := float64(base) * (0.8 + 0.4*rand.Float64()) // #nosec G404 — non-crypto jitter
	return time.Duration(jitter)
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// newID generates a short id for outbound envelopes. Not cryptographic.
func newID() string {
	return fmt.Sprintf("%d-%d", time.Now().UnixNano(), rand.Int63()) // #nosec G404
}

var _ = errors.New // keep "errors" import alive for future failure paths
