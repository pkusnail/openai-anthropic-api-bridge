package adapterhttp

import (
    "bytes"
    "context"
    "encoding/json"
    "fmt"
    "io"
    "net/http"
    "os"
    "strconv"
    "strings"
    "time"

    "claude-openai-adapter/pkg/adapter"
)

var (
    debugEnabled  = false
    logEvents     = false
)

// SetDebug enables verbose logging for the adapter
func SetDebug(v bool) { debugEnabled = v }

// SetLogEvents controls per-event SSE logging
func SetLogEvents(v bool) { logEvents = v }

type Config struct {
    AnthropicBaseURL   string
    AnthropicAPIKey    string
    AnthropicVersion   string
    OpenAIBaseURL      string
    OpenAIAPIKey       string
    ModelMap           string // line-delimited: "claude-x=gpt-y"
    DefaultOpenAIModel string // fallback when mapping missing
}

func trimRightSlash(s string) string { return strings.TrimRight(s, "/") }

func mapModelFromConfig(anthropicModel string, cfg Config) string {
    mm := cfg.ModelMap
    if mm != "" {
        scanner := strings.NewReader(mm)
        var buf strings.Builder
        line := ""
        data, _ := io.ReadAll(scanner)
        for _, ch := range strings.Split(string(data), "\n") {
            line = strings.TrimSpace(ch)
            if line == "" || strings.HasPrefix(line, "#") { continue }
            kv := strings.SplitN(line, "=", 2)
            if len(kv) == 2 && strings.TrimSpace(kv[0]) == anthropicModel {
                return strings.TrimSpace(kv[1])
            }
        }
        _ = buf
    }
    if cfg.DefaultOpenAIModel != "" { return cfg.DefaultOpenAIModel }
    return "gpt-4o-mini"
}

func writeJSON(w http.ResponseWriter, code int, v interface{}) {
    w.Header().Set("Content-Type", "application/json")
    w.WriteHeader(code)
    _ = json.NewEncoder(w).Encode(v)
}

// Messages handler (Anthropic-compatible) that proxies to OpenAI
func NewMessagesHandler(cfg Config, client *http.Client) http.Handler {
    if client == nil { client = http.DefaultClient }
    base := trimRightSlash(cfg.OpenAIBaseURL)
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        if r.Method != http.MethodPost { http.Error(w, "method not allowed", http.StatusMethodNotAllowed); return }
        var areq adapter.AnthropicMessageRequest
        if err := json.NewDecoder(r.Body).Decode(&areq); err != nil { http.Error(w, "invalid json", http.StatusBadRequest); return }
        if areq.Stream && debugNoStream(r) { areq.Stream = false }
        oreq, err := adapter.AnthropicToOpenAI(areq)
        if err != nil { http.Error(w, "invalid messages: "+err.Error(), http.StatusBadRequest); return }
        // Apply model mapping via config
        oreq.Model = mapModelFromConfig(areq.Model, cfg)
        if debugEnabled {
            info := map[string]interface{}{"model": areq.Model, "stream": areq.Stream, "messages": len(areq.Messages), "tools": len(areq.Tools)}
            b, _ := json.Marshal(info)
            fmt.Printf("[adapter/messages] incoming=%s\n", string(b))
        }
        if areq.Stream {
            proxyStream(w, r.Context(), client, base, cfg.OpenAIAPIKey, oreq, areq)
            return
        }
        proxyOnce(w, r.Context(), client, base, cfg.OpenAIAPIKey, oreq, areq)
    })
}

// preview trims a byte slice to a maximum and adds ellipsis for logging
func preview(b []byte, max int) []byte {
    if len(b) <= max { return b }
    if max < 3 { return b[:max] }
    out := make([]byte, max)
    copy(out, b[:max-3])
    copy(out[max-3:], []byte("..."))
    return out
}

type statusWriter struct { http.ResponseWriter; status int; written int }
func (s *statusWriter) WriteHeader(code int) { s.status = code; s.ResponseWriter.WriteHeader(code) }
func (s *statusWriter) Write(b []byte) (int, error) { n, err := s.ResponseWriter.Write(b); s.written += n; return n, err }

// ChatCompletions handler (OpenAI-compatible) that proxies to Anthropic
func NewChatCompletionsHandler(cfg Config, client *http.Client) http.Handler {
    if client == nil { client = http.DefaultClient }
    base := trimRightSlash(cfg.AnthropicBaseURL)
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        if r.Method != http.MethodPost { http.Error(w, "method not allowed", http.StatusMethodNotAllowed); return }
        var oreq adapter.OpenAIChatRequest
        if err := json.NewDecoder(r.Body).Decode(&oreq); err != nil { http.Error(w, "invalid json", http.StatusBadRequest); return }
        if oreq.Stream && debugNoStream(r) { oreq.Stream = false }
        areq, err := adapter.OpenAIToAnthropicRequest(oreq)
        if err != nil { http.Error(w, "invalid messages: "+err.Error(), http.StatusBadRequest); return }
        if areq.Stream {
            proxyToAnthropicStream(w, r.Context(), client, base, cfg, areq, oreq.Model)
            return
        }
        proxyToAnthropicOnce(w, r.Context(), client, base, cfg, areq, oreq.Model)
    })
}

func proxyOnce(w http.ResponseWriter, ctx context.Context, client *http.Client, base, apiKey string, oreq adapter.OpenAIChatRequest, areq adapter.AnthropicMessageRequest) {
    reqBody, _ := json.Marshal(oreq)
    req, _ := http.NewRequestWithContext(ctx, http.MethodPost, base+"/v1/chat/completions", bytes.NewReader(reqBody))
    req.Header.Set("Content-Type", "application/json")
    if apiKey != "" { req.Header.Set("Authorization", "Bearer "+apiKey) }
    resp, err := client.Do(req)
    if err != nil { http.Error(w, "openai request failed: "+err.Error(), http.StatusBadGateway); return }
    defer resp.Body.Close()
    if resp.StatusCode >= 300 {
        body, _ := io.ReadAll(io.LimitReader(resp.Body, 8192))
        http.Error(w, fmt.Sprintf("openai error %d: %s", resp.StatusCode, string(body)), http.StatusBadGateway)
        return
    }
    var oresp adapter.OpenAIChatResponse
    if err := json.NewDecoder(resp.Body).Decode(&oresp); err != nil { http.Error(w, "invalid openai response", http.StatusBadGateway); return }
    aresp, err := adapter.OpenAIToAnthropic(oresp, areq.Model)
    if err != nil { http.Error(w, "mapping error: "+err.Error(), http.StatusBadGateway); return }
    writeJSON(w, http.StatusOK, aresp)
}

func proxyStream(w http.ResponseWriter, ctx context.Context, client *http.Client, base, apiKey string, oreq adapter.OpenAIChatRequest, areq adapter.AnthropicMessageRequest) {
    oreq.Stream = true
    reqBody, _ := json.Marshal(oreq)
    req, _ := http.NewRequestWithContext(ctx, http.MethodPost, base+"/v1/chat/completions", bytes.NewReader(reqBody))
    req.Header.Set("Content-Type", "application/json")
    req.Header.Set("Accept", "text/event-stream")
    if apiKey != "" { req.Header.Set("Authorization", "Bearer "+apiKey) }
    start := time.Now()
    if debugEnabled { fmt.Printf("[adapter/openai(stream)] POST %s body=%s\n", req.URL.String(), string(preview(reqBody, 512))) }
    resp, err := client.Do(req)
    if err != nil { http.Error(w, "openai stream failed: "+err.Error(), http.StatusBadGateway); return }
    defer resp.Body.Close()
    if debugEnabled { fmt.Printf("[adapter/openai(stream)] status=%d in %s\n", resp.StatusCode, time.Since(start)) }
    if resp.StatusCode >= 300 {
        body, _ := io.ReadAll(io.LimitReader(resp.Body, 8192))
        http.Error(w, fmt.Sprintf("openai error %d: %s", resp.StatusCode, string(body)), http.StatusBadGateway)
        return
    }
    w.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
    w.Header().Set("Cache-Control", "no-cache")
    w.Header().Set("Connection", "keep-alive")
    flusher, ok := w.(http.Flusher)
    if !ok { http.Error(w, "streaming unsupported", http.StatusInternalServerError); return }
    _ = adapter.ConvertOpenAIStreamToAnthropic(ctx, areq.Model, resp.Body, func(event string, payload interface{}) {
        if logEvents && debugEnabled {
            if payload != nil { pb, _ := json.Marshal(payload); fmt.Printf("[adapter/sse->anthropic] event=%s payload=%s\n", event, string(preview(pb, 256))) } else { fmt.Printf("[adapter/sse->anthropic] event=%s\n", event) }
        }
        fmt.Fprintf(w, "event: %s\n", event)
        if payload != nil {
            b, _ := json.Marshal(payload)
            fmt.Fprintf(w, "data: %s\n\n", string(b))
        } else {
            fmt.Fprintf(w, "data: {}\n\n")
        }
        flusher.Flush()
    })
}

func proxyToAnthropicOnce(w http.ResponseWriter, ctx context.Context, client *http.Client, base string, cfg Config, areq adapter.AnthropicMessageRequest, openaiModel string) {
    body, _ := json.Marshal(areq)
    req, _ := http.NewRequestWithContext(ctx, http.MethodPost, base+"/v1/messages", bytes.NewReader(body))
    req.Header.Set("Content-Type", "application/json")
    if cfg.AnthropicAPIKey != "" { req.Header.Set("x-api-key", cfg.AnthropicAPIKey) }
    if cfg.AnthropicVersion != "" { req.Header.Set("anthropic-version", cfg.AnthropicVersion) } else { req.Header.Set("anthropic-version", "2023-06-01") }
    resp, err := client.Do(req)
    if err != nil { http.Error(w, "anthropic request failed: "+err.Error(), http.StatusBadGateway); return }
    defer resp.Body.Close()
    if resp.StatusCode >= 300 {
        b, _ := io.ReadAll(io.LimitReader(resp.Body, 8192))
        http.Error(w, fmt.Sprintf("anthropic error %d: %s", resp.StatusCode, string(b)), http.StatusBadGateway)
        return
    }
    var aresp adapter.AnthropicMessageResponse
    if err := json.NewDecoder(resp.Body).Decode(&aresp); err != nil { http.Error(w, "invalid anthropic response", http.StatusBadGateway); return }
    oresp, err := adapter.AnthropicToOpenAIResponse(aresp, openaiModel)
    if err != nil { http.Error(w, "mapping error: "+err.Error(), http.StatusBadGateway); return }
    writeJSON(w, http.StatusOK, oresp)
}

func proxyToAnthropicStream(w http.ResponseWriter, ctx context.Context, client *http.Client, base string, cfg Config, areq adapter.AnthropicMessageRequest, openaiModel string) {
    areq.Stream = true
    body, _ := json.Marshal(areq)
    req, _ := http.NewRequestWithContext(ctx, http.MethodPost, base+"/v1/messages", bytes.NewReader(body))
    req.Header.Set("Content-Type", "application/json")
    if cfg.AnthropicAPIKey != "" { req.Header.Set("x-api-key", cfg.AnthropicAPIKey) }
    if cfg.AnthropicVersion != "" { req.Header.Set("anthropic-version", cfg.AnthropicVersion) } else { req.Header.Set("anthropic-version", "2023-06-01") }
    resp, err := client.Do(req)
    if err != nil { http.Error(w, "anthropic stream failed: "+err.Error(), http.StatusBadGateway); return }
    defer resp.Body.Close()
    if resp.StatusCode >= 300 {
        b, _ := io.ReadAll(io.LimitReader(resp.Body, 8192))
        http.Error(w, fmt.Sprintf("anthropic error %d: %s", resp.StatusCode, string(b)), http.StatusBadGateway)
        return
    }
    w.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
    w.Header().Set("Cache-Control", "no-cache")
    w.Header().Set("Connection", "keep-alive")
    flusher, ok := w.(http.Flusher)
    if !ok { http.Error(w, "streaming unsupported", http.StatusInternalServerError); return }
    _ = adapter.ConvertAnthropicStreamToOpenAI(ctx, openaiModel, resp.Body, func(chunk map[string]interface{}) {
        if logEvents && debugEnabled { b, _ := json.Marshal(chunk); fmt.Printf("[adapter/sse->openai] chunk=%s\n", string(preview(b, 256))) }
        b, _ := json.Marshal(chunk)
        fmt.Fprintf(w, "data: %s\n\n", string(b))
        flusher.Flush()
    })
    fmt.Fprintf(w, "data: [DONE]\n\n")
    flusher.Flush()
}

func debugNoStream(r *http.Request) bool {
    if v := strings.ToLower(strings.TrimSpace(os.Getenv("ADAPTER_NO_STREAM"))); v == "1" || v == "true" || v == "yes" { return true }
    if v := strings.ToLower(strings.TrimSpace(r.Header.Get("X-Debug-No-Stream"))); v == "1" || v == "true" || v == "yes" { return true }
    q := r.URL.Query()
    if v := strings.ToLower(strings.TrimSpace(q.Get("debug_no_stream"))); v == "1" || v == "true" || v == "yes" { return true }
    if v := strings.ToLower(strings.TrimSpace(q.Get("no_stream"))); v == "1" || v == "true" || v == "yes" { return true }
    return false
}

// Optional small logging middleware (used by cmd/adapter)
func Logging(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        start := time.Now()
        sw := &statusWriter{ResponseWriter: w, status: 200}
        next.ServeHTTP(sw, r)
        dur := time.Since(start)
        fmt.Printf("%s %s %s %d %dB %s\n", r.RemoteAddr, r.Method, r.URL.Path, sw.status, sw.written, strconv.FormatInt(dur.Milliseconds(), 10)+"ms")
    })
}
