# Claude Code ↔ OpenAI Adapter (Go)

A small Go library and HTTP service that translates between Anthropic Messages (as used by Claude Code) and OpenAI Chat Completions. It maps messages, tools, and streaming in both directions so you can:

- Run a server that looks Anthropic-compatible but uses OpenAI under the hood, or
- Import and call the mapping functions directly inside your own service.

What’s included
- Library: `pkg/adapter` with mapping and streaming converters (no env reads, no I/O side effects).
- HTTP handlers: `pkg/adapterhttp` with `NewMessagesHandler` and `NewChatCompletionsHandler` for easy embedding.
- CLI server: `cmd/adapter` reads env vars and exposes `/v1/messages` and `/v1/chat/completions`.

Key features
- Bidirectional mapping of messages and tool calls/results.
- Streaming conversions: OpenAI chunks → Anthropic SSE, Anthropic SSE → OpenAI chunks.
- Stable tool_calls indices in streaming (name and arguments share the same index).

Notes
- Focuses on the core Claude Code flows (text + tools). Vision is not covered.
- Tool-call arguments in OpenAI streaming are accumulated per call; in Anthropic streaming they arrive as `input_json_delta` pieces.

## Use as a Library

Import paths (inside this module):
- `claude-openai-adapter/pkg/adapter`
- `claude-openai-adapter/pkg/adapterhttp`

If you use this from another repo, either:
- Set a proper VCS module path for this repo (e.g., `github.com/you/claude-openai-adapter`) and update `go.mod`, or
- Use a `replace` directive while developing locally:
  - In your app’s `go.mod`:
    - `require claude-openai-adapter v0.0.0`
    - `replace claude-openai-adapter => ../path/to/this/repo`

### Mapping only

Example: map Anthropic → OpenAI request, then OpenAI → Anthropic response.

```
import (
  ad "claude-openai-adapter/pkg/adapter"
)

areq := ad.AnthropicMessageRequest{ /* fill fields */ }
oreq, _ := ad.AnthropicToOpenAI(areq)

// ... call OpenAI ... then map back
var oresp ad.OpenAIChatResponse
aresp, _ := ad.OpenAIToAnthropic(oresp, areq.Model)
```

### Streaming conversion

```
// OpenAI → Anthropic SSE
_ = ad.ConvertOpenAIStreamToAnthropic(ctx, requestedModel, openAIStreamBody, func(event string, payload any){
  // write: "event: <event>\n" + "data: <json>\n\n"
})

// Anthropic SSE → OpenAI chunks
_ = ad.ConvertAnthropicStreamToOpenAI(ctx, openaiModel, anthropicStreamBody, func(chunk map[string]any){
  // write: "data: <json>\n\n"; caller adds [DONE]
})
```

### HTTP handlers

```
import (
  ad "claude-openai-adapter/pkg/adapter"
  httpad "claude-openai-adapter/pkg/adapterhttp"
)

cfg := httpad.Config{
  AnthropicBaseURL:   "https://api.anthropic.com",
  AnthropicAPIKey:    "...",
  AnthropicVersion:   "2023-06-01",
  OpenAIBaseURL:      "https://api.openai.com",
  OpenAIAPIKey:       "...",
  ModelMap:           "claude-x=gpt-y\nclaude-z=gpt-a",
  DefaultOpenAIModel: "gpt-4o-mini",
}

mux := http.NewServeMux()
mux.Handle("/v1/messages", httpad.NewMessagesHandler(cfg, http.DefaultClient))
mux.Handle("/v1/chat/completions", httpad.NewChatCompletionsHandler(cfg, http.DefaultClient))
```

## Run the CLI Server

Build and run:

```
go build -o adapter ./cmd/adapter
OPENAI_API_KEY=sk-... ./adapter
```

Point Claude Code (or any Anthropic client) to your server, e.g. `http://localhost:8080/v1/messages`.

### Environment variables

- `OPENAI_API_KEY`: OpenAI API key.
- `OPENAI_BASE_URL`: Default `https://api.openai.com`.
- `OPENAI_MODEL`: Fallback model if no mapping; default `gpt-4o-mini`.
- `MODEL_MAP`: Newline-separated `anthropicModel=openaiModel`. Example: `claude-sonnet-4-20250514=gpt-4o`.
- `PORT`: Default `8080` (also supports `ADAPTER_LISTEN`).
- `ADAPTER_LISTEN`: Port to listen on (default `8080`).
- `ADAPTER_LOG_FILE`: File path to write logs (example `logs/adapter.log`).
  - Daily rotation (UTC). Pointer file `adapter.log` contains the current file path.
- `ADAPTER_LOG_LEVEL`: `debug` or `info` (default `info`).
- `ADAPTER_LOG_EVENTS`: `1/true` to log each SSE event with a compact payload preview.
- `OPENAI_MAX_TOKENS_CAP`: Optional int; caps `max_tokens` before calling OpenAI to avoid 400s.
- Reverse proxy to Anthropic (for OpenAI-compatible entry):
  - `ANTHROPIC_API_KEY`
  - `ANTHROPIC_BASE_URL` (default `https://api.anthropic.com`)
  - `ANTHROPIC_VERSION` (default `2023-06-01`)

Debug toggles
- `ADAPTER_NO_STREAM`: `1/true/yes` to force non-streaming.
- Request overrides: header `X-Debug-No-Stream: 1`; query `?debug_no_stream=1` or `?no_stream=1`.

### Claude Code sidecar (quick start)

Run the adapter on 8080 and point your gateway/client to it as an Anthropic-compatible server:

```
go build -o adapter ./cmd/adapter
ADAPTER_LOG_FILE=logs/adapter.log \
ADAPTER_LOG_LEVEL=debug \
ADAPTER_LOG_EVENTS=1 \
OPENAI_API_KEY=sk-... \
./adapter

# In your gateway (Anthropic passthrough):
#   TOKLIGENCE_ANTHROPIC_PASSTHROUGH_ENABLED=true
#   TOKLIGENCE_ANTHROPIC_BASE_URL=http://127.0.0.1:8080
```

Tail logs: `tail -f logs/adapter-YYYY-MM-DD.log`.

## Endpoints

- `POST /v1/messages` (Anthropic-compatible)
  - Input: `model`, `messages`, `system`, `tools`, `max_tokens`, `temperature`, `stop_sequences`, `stream`.
  - Output: Anthropic `message` or Anthropic-style SSE stream.

- `POST /v1/chat/completions` (OpenAI-compatible)
  - Input: OpenAI Chat Completions request.
  - Output: OpenAI response or OpenAI streaming chunks. Streaming preserves function call deltas.

## Tests

Run tests (no real network; HTTP calls are stubbed):

```
go test ./...
```

If you’re in a restricted sandbox, a local build cache helps:

```
GOCACHE=$(pwd)/.gocache go test ./... -v
```

Log-driven tests
- Some tests read local JSONL logs to assert parity. They auto-skip if the files are missing.
- Paths referenced: `~/.codex/sessions/.../*.jsonl`, `~/.claude/projects/.../*.jsonl`.

## Implementation Notes

- Content types supported: `text`, `tool_use`, `tool_result`.
- Streaming: In Anthropic→OpenAI, tool_calls name and arguments now share a stable index.
- Error-tolerance: invalid tool-call arguments fall back to `{ "_": "raw" }` in non-streaming; empty `{}` in streaming aggregation.

## Development

- Layout
  - `pkg/adapter`: pure mapping and streaming logic, no I/O.
  - `pkg/adapterhttp`: HTTP handlers, request shaping, and stream bridging.
  - `pkg/logging`: tiny daily rotating writer for CLI logs.
  - `cmd/adapter`: thin server wiring + env-based config.

- Common flows
  - Anthropic Messages → OpenAI Chat Completions (tools and text supported)
  - OpenAI streams → Anthropic SSE events (message_start/content_block_*/message_delta/message_stop)
  - Anthropic SSE → OpenAI streaming chunks (`data:` lines; caller adds `[DONE]`)

- Debugging
  - Enable `ADAPTER_LOG_LEVEL=debug` and `ADAPTER_LOG_EVENTS=1` to record per-event stream logs.
  - Each request logs upstream POST URL, a truncated request body preview, status, and latency.


## Migration (from single-file main)

- Use `pkg/adapter` for pure mapping and stream conversion.
- Use `pkg/adapterhttp` to embed handlers; CLI moved to `cmd/adapter`.
- HTTP behavior and endpoints remain the same as before.
