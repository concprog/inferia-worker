// Tests for the worker-side control protocol JSON wire format.
//
// The Go types here MUST stay byte-compatible with the Pydantic models in
// InferiaLLM/.../worker_controller/protocol.py. These tests pin the
// snake_case field names + omitempty semantics so a future drive-by rename
// fails CI rather than silently breaking the WS channel.
package control

import (
	"encoding/json"
	"reflect"
	"testing"
)

// TestNewBodyJSON pins the exact JSON wire format for one representative
// payload per new shell/logs envelope body. The expected strings are written
// to match what Go's encoding/json emits (alphabetic-by-struct-field-order)
// and what the Python pydantic side will deserialize.
func TestNewBodyJSON(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name   string
		body   any
		wantJS string
	}{
		{
			name: "shell_open_full",
			body: ShellOpenBody{
				StreamID:     "s1",
				Shell:        "/bin/zsh",
				User:         "root",
				DeploymentID: "dep-1",
				ContainerID:  "ctr-1",
				Cols:         120,
				Rows:         40,
			},
			wantJS: `{"stream_id":"s1","shell":"/bin/zsh","user":"root","deployment_id":"dep-1","container_id":"ctr-1","cols":120,"rows":40}`,
		},
		{
			name:   "shell_open_minimal",
			body:   ShellOpenBody{StreamID: "s1"},
			wantJS: `{"stream_id":"s1"}`,
		},
		{
			name:   "shell_input",
			body:   ShellInputBody{StreamID: "s1", Data: "ls\n"},
			wantJS: `{"stream_id":"s1","data":"ls\n"}`,
		},
		{
			name:   "shell_resize",
			body:   ShellResizeBody{StreamID: "s1", Cols: 80, Rows: 24},
			wantJS: `{"stream_id":"s1","cols":80,"rows":24}`,
		},
		{
			name:   "shell_close",
			body:   ShellCloseBody{StreamID: "s1"},
			wantJS: `{"stream_id":"s1"}`,
		},
		{
			name:   "shell_output",
			body:   ShellOutputBody{StreamID: "s1", Data: "hello\r\n"},
			wantJS: `{"stream_id":"s1","data":"hello\r\n"}`,
		},
		{
			name:   "shell_exit_full",
			body:   ShellExitBody{StreamID: "s1", ExitCode: 137, Reason: "killed"},
			wantJS: `{"stream_id":"s1","exit_code":137,"reason":"killed"}`,
		},
		{
			name:   "shell_exit_minimal",
			body:   ShellExitBody{StreamID: "s1"},
			wantJS: `{"stream_id":"s1"}`,
		},
		{
			name:   "shell_error",
			body:   ShellErrorBody{StreamID: "s1", Message: "container not found"},
			wantJS: `{"stream_id":"s1","message":"container not found"}`,
		},
		{
			name: "logs_open_full",
			body: LogsOpenBody{
				StreamID:     "s2",
				DeploymentID: "dep-1",
				ContainerID:  "ctr-1",
			},
			wantJS: `{"stream_id":"s2","deployment_id":"dep-1","container_id":"ctr-1"}`,
		},
		{
			name:   "logs_open_minimal",
			body:   LogsOpenBody{StreamID: "s2"},
			wantJS: `{"stream_id":"s2"}`,
		},
		{
			name:   "logs_line_stdout",
			body:   LogsLineBody{StreamID: "s2", Stream: "stdout", Data: "starting\n"},
			wantJS: `{"stream_id":"s2","stream":"stdout","data":"starting\n"}`,
		},
		{
			name:   "logs_line_stderr",
			body:   LogsLineBody{StreamID: "s2", Stream: "stderr", Data: "oops\n"},
			wantJS: `{"stream_id":"s2","stream":"stderr","data":"oops\n"}`,
		},
		{
			// Stream is omitempty: when zero-value it disappears from the
			// wire; the Python side's default of "stdout" fills it back in.
			name:   "logs_line_default_stream",
			body:   LogsLineBody{StreamID: "s2", Data: "line\n"},
			wantJS: `{"stream_id":"s2","data":"line\n"}`,
		},
		{
			name:   "logs_end_full",
			body:   LogsEndBody{StreamID: "s2", Reason: "container_exited"},
			wantJS: `{"stream_id":"s2","reason":"container_exited"}`,
		},
		{
			name:   "logs_end_minimal",
			body:   LogsEndBody{StreamID: "s2"},
			wantJS: `{"stream_id":"s2"}`,
		},
		{
			name:   "logs_close",
			body:   LogsCloseBody{StreamID: "s2"},
			wantJS: `{"stream_id":"s2"}`,
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got, err := json.Marshal(tc.body)
			if err != nil {
				t.Fatalf("marshal: %v", err)
			}
			if string(got) != tc.wantJS {
				t.Errorf("wire format drift\n got: %s\nwant: %s", got, tc.wantJS)
			}
		})
	}
}

// TestNewBodyRoundTrip ensures every new body type round-trips through JSON
// with DeepEqual fidelity. Catches accidental tag typos that would cause
// fields to silently drop during unmarshal.
func TestNewBodyRoundTrip(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name string
		// orig must be a non-pointer struct value; dst must be a pointer to
		// the same struct type so json.Unmarshal can populate it.
		orig any
		dst  any
	}{
		{
			name: "shell_open",
			orig: ShellOpenBody{
				StreamID:     "sid",
				Shell:        "/bin/bash",
				User:         "1000:1000",
				DeploymentID: "dep",
				ContainerID:  "ctr",
				Cols:         132,
				Rows:         50,
			},
			dst: &ShellOpenBody{},
		},
		{
			name: "shell_input",
			orig: ShellInputBody{StreamID: "sid", Data: "echo hi\n"},
			dst:  &ShellInputBody{},
		},
		{
			name: "shell_resize",
			orig: ShellResizeBody{StreamID: "sid", Cols: 200, Rows: 60},
			dst:  &ShellResizeBody{},
		},
		{
			name: "shell_close",
			orig: ShellCloseBody{StreamID: "sid"},
			dst:  &ShellCloseBody{},
		},
		{
			name: "shell_output",
			orig: ShellOutputBody{StreamID: "sid", Data: "\x1b[32mok\x1b[0m"},
			dst:  &ShellOutputBody{},
		},
		{
			name: "shell_exit",
			orig: ShellExitBody{StreamID: "sid", ExitCode: 1, Reason: "non-zero"},
			dst:  &ShellExitBody{},
		},
		{
			name: "shell_error",
			orig: ShellErrorBody{StreamID: "sid", Message: "spawn failed: exec format"},
			dst:  &ShellErrorBody{},
		},
		{
			name: "logs_open",
			orig: LogsOpenBody{StreamID: "sid", DeploymentID: "dep", ContainerID: "ctr"},
			dst:  &LogsOpenBody{},
		},
		{
			name: "logs_line",
			orig: LogsLineBody{StreamID: "sid", Stream: "stderr", Data: "warn: x"},
			dst:  &LogsLineBody{},
		},
		{
			name: "logs_end",
			orig: LogsEndBody{StreamID: "sid", Reason: "timeout"},
			dst:  &LogsEndBody{},
		},
		{
			name: "logs_close",
			orig: LogsCloseBody{StreamID: "sid"},
			dst:  &LogsCloseBody{},
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			data, err := json.Marshal(tc.orig)
			if err != nil {
				t.Fatalf("marshal: %v", err)
			}
			if err := json.Unmarshal(data, tc.dst); err != nil {
				t.Fatalf("unmarshal: %v", err)
			}
			// Dereference dst for comparison with the value-typed orig.
			got := reflect.ValueOf(tc.dst).Elem().Interface()
			if !reflect.DeepEqual(tc.orig, got) {
				t.Errorf("round-trip lost data\norig: %#v\n got: %#v", tc.orig, got)
			}
		})
	}
}

// TestMessageTypeConstants pins the exact string values of the new
// MessageType constants. A typo here on either side of the worker/CP wire
// silently breaks routing.
func TestMessageTypeConstants(t *testing.T) {
	t.Parallel()
	cases := []struct {
		got, want MessageType
	}{
		// Pre-existing constants — re-asserted so any future rename also
		// fails this targeted test (cheap insurance).
		{MsgHello, "Hello"},
		{MsgHeartbeat, "Heartbeat"},
		{MsgLoadModel, "LoadModel"},
		{MsgUnloadModel, "UnloadModel"},
		{MsgCommandResult, "CommandResult"},
		{MsgPing, "Ping"},
		// New shell + logs constants.
		{MsgShellOpen, "ShellOpen"},
		{MsgShellInput, "ShellInput"},
		{MsgShellResize, "ShellResize"},
		{MsgShellClose, "ShellClose"},
		{MsgShellOutput, "ShellOutput"},
		{MsgShellExit, "ShellExit"},
		{MsgShellError, "ShellError"},
		{MsgLogsOpen, "LogsOpen"},
		{MsgLogsClose, "LogsClose"},
		{MsgLogsLine, "LogsLine"},
		{MsgLogsEnd, "LogsEnd"},
	}
	for _, tc := range cases {
		if tc.got != tc.want {
			t.Errorf("message type drift: got %q want %q", tc.got, tc.want)
		}
	}
}

// TestEnvelopeWithNewBodies verifies that the existing Envelope wrapper
// still works when carrying each new body type, including the two-phase
// unmarshal (envelope first, then body via json.RawMessage). This mirrors
// how channel.go dispatches frames.
func TestEnvelopeWithNewBodies(t *testing.T) {
	t.Parallel()

	// Build one envelope per new shell/logs body. After marshal we
	// unmarshal into an Envelope whose Body is json.RawMessage so we can
	// then re-parse into the concrete body type — same pattern as
	// channel.handle().
	cases := []struct {
		name    string
		msgType MessageType
		body    any
		decode  func(json.RawMessage) (any, error)
	}{
		{
			name:    "shell_open",
			msgType: MsgShellOpen,
			body:    ShellOpenBody{StreamID: "s1", Shell: "/bin/sh", Cols: 80, Rows: 24},
			decode: func(raw json.RawMessage) (any, error) {
				var b ShellOpenBody
				return b, json.Unmarshal(raw, &b)
			},
		},
		{
			name:    "shell_input",
			msgType: MsgShellInput,
			body:    ShellInputBody{StreamID: "s1", Data: "q"},
			decode: func(raw json.RawMessage) (any, error) {
				var b ShellInputBody
				return b, json.Unmarshal(raw, &b)
			},
		},
		{
			name:    "shell_resize",
			msgType: MsgShellResize,
			body:    ShellResizeBody{StreamID: "s1", Cols: 100, Rows: 30},
			decode: func(raw json.RawMessage) (any, error) {
				var b ShellResizeBody
				return b, json.Unmarshal(raw, &b)
			},
		},
		{
			name:    "shell_close",
			msgType: MsgShellClose,
			body:    ShellCloseBody{StreamID: "s1"},
			decode: func(raw json.RawMessage) (any, error) {
				var b ShellCloseBody
				return b, json.Unmarshal(raw, &b)
			},
		},
		{
			name:    "shell_output",
			msgType: MsgShellOutput,
			body:    ShellOutputBody{StreamID: "s1", Data: "$ "},
			decode: func(raw json.RawMessage) (any, error) {
				var b ShellOutputBody
				return b, json.Unmarshal(raw, &b)
			},
		},
		{
			name:    "shell_exit",
			msgType: MsgShellExit,
			body:    ShellExitBody{StreamID: "s1", ExitCode: 0},
			decode: func(raw json.RawMessage) (any, error) {
				var b ShellExitBody
				return b, json.Unmarshal(raw, &b)
			},
		},
		{
			name:    "shell_error",
			msgType: MsgShellError,
			body:    ShellErrorBody{StreamID: "s1", Message: "bad shell"},
			decode: func(raw json.RawMessage) (any, error) {
				var b ShellErrorBody
				return b, json.Unmarshal(raw, &b)
			},
		},
		{
			name:    "logs_open",
			msgType: MsgLogsOpen,
			body:    LogsOpenBody{StreamID: "s2", DeploymentID: "dep"},
			decode: func(raw json.RawMessage) (any, error) {
				var b LogsOpenBody
				return b, json.Unmarshal(raw, &b)
			},
		},
		{
			name:    "logs_line",
			msgType: MsgLogsLine,
			body:    LogsLineBody{StreamID: "s2", Stream: "stdout", Data: "ready"},
			decode: func(raw json.RawMessage) (any, error) {
				var b LogsLineBody
				return b, json.Unmarshal(raw, &b)
			},
		},
		{
			name:    "logs_end",
			msgType: MsgLogsEnd,
			body:    LogsEndBody{StreamID: "s2", Reason: "exited"},
			decode: func(raw json.RawMessage) (any, error) {
				var b LogsEndBody
				return b, json.Unmarshal(raw, &b)
			},
		},
		{
			name:    "logs_close",
			msgType: MsgLogsClose,
			body:    LogsCloseBody{StreamID: "s2"},
			decode: func(raw json.RawMessage) (any, error) {
				var b LogsCloseBody
				return b, json.Unmarshal(raw, &b)
			},
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			origEnv := Envelope{
				Type: tc.msgType,
				ID:   "uuid-" + tc.name,
				TS:   "2026-05-27T00:00:00Z",
				Body: tc.body,
			}
			raw, err := json.Marshal(origEnv)
			if err != nil {
				t.Fatalf("marshal envelope: %v", err)
			}

			// Two-phase unmarshal: envelope first, then concrete body.
			type rawEnv struct {
				Type MessageType     `json:"type"`
				ID   string          `json:"id"`
				TS   string          `json:"ts"`
				Body json.RawMessage `json:"body,omitempty"`
			}
			var got rawEnv
			if err := json.Unmarshal(raw, &got); err != nil {
				t.Fatalf("unmarshal envelope: %v", err)
			}
			if got.Type != tc.msgType {
				t.Errorf("envelope type: got %q want %q", got.Type, tc.msgType)
			}
			if got.ID != origEnv.ID {
				t.Errorf("envelope id: got %q want %q", got.ID, origEnv.ID)
			}
			if got.TS != origEnv.TS {
				t.Errorf("envelope ts: got %q want %q", got.TS, origEnv.TS)
			}
			decoded, err := tc.decode(got.Body)
			if err != nil {
				t.Fatalf("decode body: %v", err)
			}
			if !reflect.DeepEqual(decoded, tc.body) {
				t.Errorf("body mismatch\norig: %#v\n got: %#v", tc.body, decoded)
			}
		})
	}
}

// TestEnvelopeBodyOmitEmpty pins that a nil-Body envelope omits the field
// entirely (Python side then sees body=None via its default). Existing
// behavior, re-asserted alongside the new types.
func TestEnvelopeBodyOmitEmpty(t *testing.T) {
	t.Parallel()
	env := Envelope{Type: MsgPing, ID: "x", TS: "t"}
	raw, err := json.Marshal(env)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	want := `{"type":"Ping","id":"x","ts":"t"}`
	if string(raw) != want {
		t.Errorf("nil-body envelope wire format\n got: %s\nwant: %s", raw, want)
	}
}
