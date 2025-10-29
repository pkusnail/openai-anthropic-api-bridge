package adapterhttp_test

import (
    "bufio"
    "bytes"
    "encoding/json"
    "errors"
    "io"
    "net/http"
    "net/http/httptest"
    "os"
    "path/filepath"
    "strings"
    "testing"

    ad "claude-openai-adapter/pkg/adapter"
    httpad "claude-openai-adapter/pkg/adapterhttp"
)

type roundTripperFunc func(req *http.Request) (*http.Response, error)
func (f roundTripperFunc) RoundTrip(req *http.Request) (*http.Response, error) { return f(req) }

// --- Logs helpers ---
type codexItem struct { Type string `json:"type"`; Payload json.RawMessage `json:"payload"` }
type codexFunctionCall struct { Type string `json:"type"`; Name string `json:"name"`; Arguments string `json:"arguments"` }
type claudeItem struct { Type string `json:"type"`; Message json.RawMessage `json:"message"` }
type claudeMessage struct { Role string `json:"role"`; Content []map[string]any `json:"content"` }

func readJSONL(path string, limit int) ([][]byte, error) {
    f, err := os.Open(path)
    if err != nil { return nil, err }
    defer f.Close()
    var lines [][]byte
    s := bufio.NewScanner(f)
    s.Buffer(make([]byte, 0, 1024*1024), 10*1024*1024)
    for s.Scan() {
        b := append([]byte(nil), s.Bytes()...)
        lines = append(lines, b)
        if limit > 0 && len(lines) >= limit { break }
    }
    if err := s.Err(); err != nil { return nil, err }
    return lines, nil
}

func parseCodexToolCall(path string) (toolName, argPath string, _ error) {
    lines, err := readJSONL(path, 0)
    if err != nil { return "", "", err }
    for _, line := range lines {
        var it codexItem
        if err := json.Unmarshal(line, &it); err != nil { continue }
        if it.Type != "response_item" { continue }
        var maybe codexFunctionCall
        if err := json.Unmarshal(it.Payload, &maybe); err != nil { continue }
        if strings.ToLower(maybe.Type) == "function_call" && maybe.Name != "" {
            var args map[string]any
            if err := json.Unmarshal([]byte(maybe.Arguments), &args); err == nil {
                if p, ok := args["path"].(string); ok { return maybe.Name, p, nil }
            }
        }
    }
    return "", "", errors.New("no function_call found")
}

func parseClaudeToolUse(path string) (toolName, filePath string, _ error) {
    lines, err := readJSONL(path, 0)
    if err != nil { return "", "", err }
    for _, line := range lines {
        var it claudeItem
        if err := json.Unmarshal(line, &it); err != nil { continue }
        if it.Type != "assistant" { continue }
        var msg claudeMessage
        if err := json.Unmarshal(it.Message, &msg); err != nil { continue }
        for _, c := range msg.Content {
            if c["type"] == "tool_use" {
                name, _ := c["name"].(string)
                if in, ok := c["input"].(map[string]any); ok {
                    if fp, ok := in["file_path"].(string); ok { return name, fp, nil }
                }
            }
        }
    }
    return "", "", errors.New("no tool_use found")
}

// --- Tests ---

func TestMessagesHandler_Streaming(t *testing.T) {
    prev := http.DefaultTransport
    t.Cleanup(func(){ http.DefaultTransport = prev })
    http.DefaultTransport = roundTripperFunc(func(req *http.Request) (*http.Response, error) {
        resp := &http.Response{StatusCode: 200, Header: make(http.Header)}
        resp.Header.Set("Content-Type", "text/event-stream")
        s := ""+
            "data: {\"id\":\"1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"He\"}}]}\n\n"+
            "data: {\"id\":\"1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"id\":\"call_123\",\"type\":\"function\",\"index\":0,\"function\":{\"name\":\"sum\"}}]}}]}\n\n"+
            "data: {\"id\":\"1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"type\":\"function\",\"function\":{\"arguments\":\"{\\\"a\\\":1\"}}]}}]}\n\n"+
            "data: {\"id\":\"1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"type\":\"function\",\"function\":{\"arguments\":\",\\\"b\\\":2}\"}}]}}]}\n\n"+
            "data: {\"id\":\"1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"llo\"}}]}\n\n"+
            "data: [DONE]\n\n"
        resp.Body = io.NopCloser(strings.NewReader(s))
        return resp, nil
    })

    cfg := httpad.Config{ OpenAIBaseURL: "http://openai.local" }
    h := httpad.NewMessagesHandler(cfg, http.DefaultClient)
    areq := ad.AnthropicMessageRequest{ Model: "claude-foo", Stream: true, Messages: []ad.AnthropicMsg{{Role:"user", Content: json.RawMessage(`"Hi"`)}} }
    b, _ := json.Marshal(areq)
    req := httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewReader(b))
    w := httptest.NewRecorder()
    h.ServeHTTP(w, req)
    res := w.Result()
    if ct := res.Header.Get("Content-Type"); !strings.Contains(ct, "text/event-stream") { t.Fatalf("content-type: %s", ct) }
    data, _ := io.ReadAll(res.Body)
    s := string(data)
    if !strings.Contains(s, "event: message_start") { t.Fatalf("missing message_start: %s", s) }
    if !strings.Contains(s, "event: content_block_start") { t.Fatalf("missing content_block_start: %s", s) }
    if !strings.Contains(s, "event: content_block_delta") { t.Fatalf("missing content_block_delta: %s", s) }
    if !strings.Contains(s, "He") || !strings.Contains(s, "llo") { t.Fatalf("missing text deltas: %s", s) }
    if !strings.Contains(s, "event: message_stop") { t.Fatalf("missing message_stop: %s", s) }
    if !strings.Contains(s, "\"type\":\"tool_use\"") || !strings.Contains(s, "\"name\":\"sum\"") { t.Fatalf("missing tool_use block: %s", s) }
    if !strings.Contains(s, "\"input\":{\"a\":1,\"b\":2}") { t.Fatalf("missing tool_use input: %s", s) }
}

func TestMessagesHandler_ForceNoStream(t *testing.T) {
    prev := http.DefaultTransport
    t.Cleanup(func(){ http.DefaultTransport = prev })
    http.DefaultTransport = roundTripperFunc(func(req *http.Request) (*http.Response, error) {
        resp := &http.Response{StatusCode: 200, Header: make(http.Header)}
        resp.Header.Set("Content-Type", "application/json")
        resp.Body = io.NopCloser(strings.NewReader(`{
            "id":"chatcmpl_test","object":"chat.completion","model":"gpt-4o-mini",
            "choices":[{"index":0,"finish_reason":"stop","message":{
                "role":"assistant",
                "content":"Hi debug"
            }}]
        }`))
        return resp, nil
    })
    cfg := httpad.Config{ OpenAIBaseURL: "http://openai.local" }
    h := httpad.NewMessagesHandler(cfg, http.DefaultClient)
    areq := ad.AnthropicMessageRequest{ Model: "claude-x", Stream: true, Messages: []ad.AnthropicMsg{{Role:"user", Content: json.RawMessage(`"hi"`)}} }
    b, _ := json.Marshal(areq)
    req := httptest.NewRequest(http.MethodPost, "/v1/messages?debug_no_stream=1", bytes.NewReader(b))
    req.Header.Set("X-Debug-No-Stream", "1")
    w := httptest.NewRecorder()
    h.ServeHTTP(w, req)
    res := w.Result()
    if ct := res.Header.Get("Content-Type"); !strings.Contains(ct, "application/json") { t.Fatalf("content-type: %s", ct) }
    var a ad.AnthropicMessageResponse
    if err := json.NewDecoder(res.Body).Decode(&a); err != nil { t.Fatalf("decode: %v", err) }
    if len(a.Content) == 0 || a.Content[0]["type"] != "text" { t.Fatalf("bad content: %#v", a.Content) }
}

func TestChatCompletions_NonStreaming(t *testing.T) {
    prev := http.DefaultTransport
    t.Cleanup(func(){ http.DefaultTransport = prev })
    http.DefaultTransport = roundTripperFunc(func(req *http.Request) (*http.Response, error) {
        if req.URL.Path != "/v1/messages" { t.Fatalf("unexpected path: %s", req.URL.Path) }
        resp := &http.Response{StatusCode: 200, Header: make(http.Header)}
        resp.Header.Set("Content-Type", "application/json")
        resp.Body = io.NopCloser(strings.NewReader(`{
            "id":"msg_x","type":"message","role":"assistant","model":"claude-x",
            "content":[
                {"type":"text","text":"Hello A"},
                {"type":"tool_use","id":"call_5","name":"sum","input":{"a":1}}
            ]
        }`))
        return resp, nil
    })
    cfg := httpad.Config{ AnthropicBaseURL: "http://anth.local" }
    h := httpad.NewChatCompletionsHandler(cfg, http.DefaultClient)
    oreq := ad.OpenAIChatRequest{ Model: "gpt-4o-mini", Messages: []ad.OpenAIMessage{{Role:"user", Content: "hi"}} }
    b, _ := json.Marshal(oreq)
    req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(b))
    w := httptest.NewRecorder()
    h.ServeHTTP(w, req)
    res := w.Result()
    if res.StatusCode != 200 { t.Fatalf("status: %d", res.StatusCode) }
    var oresp ad.OpenAIChatResponse
    if err := json.NewDecoder(res.Body).Decode(&oresp); err != nil { t.Fatalf("decode: %v", err) }
    if len(oresp.Choices) != 1 { t.Fatalf("choices: %d", len(oresp.Choices)) }
    if oresp.Choices[0].Message.Content.(string) != "Hello A" { t.Fatalf("text: %#v", oresp.Choices[0].Message.Content) }
    if len(oresp.Choices[0].Message.ToolCalls) != 1 || oresp.Choices[0].Message.ToolCalls[0].Function.Name != "sum" { t.Fatalf("tool_calls: %#v", oresp.Choices[0].Message.ToolCalls) }
}

func TestChatCompletions_Streaming_WithToolArgs(t *testing.T) {
    prev := http.DefaultTransport
    t.Cleanup(func(){ http.DefaultTransport = prev })
    http.DefaultTransport = roundTripperFunc(func(req *http.Request) (*http.Response, error) {
        resp := &http.Response{StatusCode: 200, Header: make(http.Header)}
        resp.Header.Set("Content-Type", "text/event-stream")
        s := ""+
        "event: message_start\n"+
        "data: {\"type\":\"message_start\",\"message\":{}}\n\n"+
        "event: content_block_start\n"+
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\"}}\n\n"+
        "event: content_block_delta\n"+
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"He\"}}\n\n"+
        "event: content_block_stop\n"+
        "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n"+
        "event: content_block_start\n"+
        "data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"call_9\",\"name\":\"sum\"}}\n\n"+
        "event: content_block_delta\n"+
        "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"a\\\":1\"}}\n\n"+
        "event: content_block_delta\n"+
        "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\",\\\"b\\\":2}\"}}\n\n"+
        "event: content_block_start\n"+
        "data: {\"type\":\"content_block_start\",\"index\":2,\"content_block\":{\"type\":\"text\"}}\n\n"+
        "event: content_block_delta\n"+
        "data: {\"type\":\"content_block_delta\",\"index\":2,\"delta\":{\"type\":\"text_delta\",\"text\":\"llo\"}}\n\n"+
        "event: message_stop\n"+
        "data: {\"type\":\"message_stop\"}\n\n"
        resp.Body = io.NopCloser(strings.NewReader(s))
        return resp, nil
    })
    cfg := httpad.Config{ AnthropicBaseURL: "http://anth.local" }
    h := httpad.NewChatCompletionsHandler(cfg, http.DefaultClient)
    oreq := ad.OpenAIChatRequest{ Model: "gpt-4o-mini", Stream: true, Messages: []ad.OpenAIMessage{{Role:"user", Content:"hi"}} }
    b, _ := json.Marshal(oreq)
    req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(b))
    w := httptest.NewRecorder()
    h.ServeHTTP(w, req)
    res := w.Result()
    if ct := res.Header.Get("Content-Type"); !strings.Contains(ct, "text/event-stream") { t.Fatalf("ct: %s", ct) }
    data, _ := io.ReadAll(res.Body)
    s := string(data)
    if !strings.Contains(s, "\"role\":\"assistant\"") { t.Fatalf("missing role chunk: %s", s) }
    if !strings.Contains(s, "He") || !strings.Contains(s, "llo") { t.Fatalf("missing text: %s", s) }
    if !strings.Contains(s, "\"tool_calls\"") || !strings.Contains(s, "\"name\":\"sum\"") { t.Fatalf("missing tool_calls: %s", s) }
    if !strings.Contains(s, "\\\"a\\\":1") || !strings.Contains(s, "\\\"b\\\":2") { t.Fatalf("missing arg pieces: %s", s) }
    if !strings.Contains(s, "[DONE]") { t.Fatalf("missing done: %s", s) }
}

func TestChatCompletions_Roundtrip_ToolUseThenResult(t *testing.T) {
    prev := http.DefaultTransport
    t.Cleanup(func(){ http.DefaultTransport = prev })
    http.DefaultTransport = roundTripperFunc(func(req *http.Request) (*http.Response, error) {
        if req.URL.Path != "/v1/messages" { return &http.Response{StatusCode: 404, Body: io.NopCloser(strings.NewReader(""))}, nil }
        var areq ad.AnthropicMessageRequest
        if err := json.NewDecoder(req.Body).Decode(&areq); err != nil { return &http.Response{StatusCode: 400, Body: io.NopCloser(strings.NewReader("bad json"))}, nil }
        if len(areq.Messages) < 2 { return &http.Response{StatusCode: 400, Body: io.NopCloser(strings.NewReader("missing messages"))}, nil }
        var p0 []ad.AnthropicContent
        _ = json.Unmarshal(areq.Messages[0].Content, &p0)
        if len(p0) != 1 || p0[0].Type != "tool_use" || p0[0].ID != "call_42" || p0[0].Name != "Read" {
            return &http.Response{StatusCode: 422, Body: io.NopCloser(strings.NewReader("wrong tool_use"))}, nil
        }
        var p1 []ad.AnthropicContent
        _ = json.Unmarshal(areq.Messages[1].Content, &p1)
        if len(p1) != 1 || p1[0].Type != "tool_result" || p1[0].ToolUseID != "call_42" || p1[0].Content.(string) != "OK" {
            return &http.Response{StatusCode: 422, Body: io.NopCloser(strings.NewReader("wrong tool_result"))}, nil
        }
        resp := &http.Response{StatusCode: 200, Header: make(http.Header)}
        resp.Header.Set("Content-Type", "application/json")
        resp.Body = io.NopCloser(strings.NewReader(`{"id":"msg_rt","type":"message","role":"assistant","model":"claude-x","content":[{"type":"text","text":"Done"}]}`))
        return resp, nil
    })
    cfg := httpad.Config{ AnthropicBaseURL: "http://anth.local" }
    h := httpad.NewChatCompletionsHandler(cfg, http.DefaultClient)
    oreq := ad.OpenAIChatRequest{
        Model: "gpt-x",
        Messages: []ad.OpenAIMessage{
            {Role: "assistant", ToolCalls: []ad.OpenAIToolCall{{ ID: "call_42", Type: "function", Function: ad.OpenAIToolCallFunction{Name: "Read", Arguments: `{"path":"/home/alejandroseaah/Pictures/cc.png"}`}}}},
            {Role: "tool", ToolCallID: "call_42", Content: "OK"},
        },
    }
    b, _ := json.Marshal(oreq)
    req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(b))
    w := httptest.NewRecorder()
    h.ServeHTTP(w, req)
    res := w.Result()
    if res.StatusCode != 200 { body, _ := io.ReadAll(res.Body); t.Fatalf("status: %d body: %s", res.StatusCode, string(body)) }
    var oresp ad.OpenAIChatResponse
    if err := json.NewDecoder(res.Body).Decode(&oresp); err != nil { t.Fatalf("decode: %v", err) }
    if len(oresp.Choices) != 1 || oresp.Choices[0].Message.Content.(string) != "Done" { t.Fatalf("final content wrong: %#v", oresp) }
}

func TestMessagesHandler_Roundtrip_ToolUseThenResult(t *testing.T) {
    prev := http.DefaultTransport
    t.Cleanup(func(){ http.DefaultTransport = prev })
    http.DefaultTransport = roundTripperFunc(func(req *http.Request) (*http.Response, error) {
        if req.URL.Path != "/v1/chat/completions" { return &http.Response{StatusCode: 404, Body: io.NopCloser(strings.NewReader(""))}, nil }
        var oreq ad.OpenAIChatRequest
        if err := json.NewDecoder(req.Body).Decode(&oreq); err != nil { return &http.Response{StatusCode: 400, Body: io.NopCloser(strings.NewReader("bad json"))}, nil }
        if len(oreq.Messages) < 2 { return &http.Response{StatusCode: 400, Body: io.NopCloser(strings.NewReader("missing messages"))}, nil }
        if oreq.Messages[0].Role != "assistant" || len(oreq.Messages[0].ToolCalls) != 1 || oreq.Messages[0].ToolCalls[0].Function.Name != "Read" {
            return &http.Response{StatusCode: 422, Body: io.NopCloser(strings.NewReader("wrong tool_calls"))}, nil
        }
        if oreq.Messages[1].Role != "tool" || oreq.Messages[1].ToolCallID != "call_42" || (oreq.Messages[1].Content.(string)) != "OK" {
            return &http.Response{StatusCode: 422, Body: io.NopCloser(strings.NewReader("wrong tool result msg"))}, nil
        }
        resp := &http.Response{StatusCode: 200, Header: make(http.Header)}
        resp.Header.Set("Content-Type", "application/json")
        resp.Body = io.NopCloser(strings.NewReader(`{"id":"chatcmpl_rt","object":"chat.completion","model":"gpt-x","choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"All set"}}]}`))
        return resp, nil
    })
    cfg := httpad.Config{ OpenAIBaseURL: "http://openai.local" }
    h := httpad.NewMessagesHandler(cfg, http.DefaultClient)
    in := map[string]any{"path": "/home/alejandroseaah/Pictures/cc.png"}
    inRaw, _ := json.Marshal(in)
    parts0 := []ad.AnthropicContent{{Type: "tool_use", ID: "call_42", Name: "Read", Input: (*json.RawMessage)(&inRaw)}}
    raw0, _ := json.Marshal(parts0)
    parts1 := []ad.AnthropicContent{{Type: "tool_result", ToolUseID: "call_42", Content: "OK"}}
    raw1, _ := json.Marshal(parts1)
    areq := ad.AnthropicMessageRequest{ Model: "claude-x", Messages: []ad.AnthropicMsg{{Role: "assistant", Content: raw0},{Role: "user", Content: raw1}} }
    b, _ := json.Marshal(areq)
    req := httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewReader(b))
    w := httptest.NewRecorder()
    h.ServeHTTP(w, req)
    res := w.Result()
    if res.StatusCode != 200 { body, _ := io.ReadAll(res.Body); t.Fatalf("status: %d body: %s", res.StatusCode, string(body)) }
    var aresp ad.AnthropicMessageResponse
    if err := json.NewDecoder(res.Body).Decode(&aresp); err != nil { t.Fatalf("decode: %v", err) }
    if len(aresp.Content) == 0 || aresp.Content[0]["type"] != "text" || aresp.Content[0]["text"].(string) != "All set" { t.Fatalf("final anthropic content wrong: %#v", aresp.Content) }
}

func TestMessagesHandler_Streaming_TwoToolCalls(t *testing.T) {
    prev := http.DefaultTransport
    t.Cleanup(func(){ http.DefaultTransport = prev })
    http.DefaultTransport = roundTripperFunc(func(req *http.Request) (*http.Response, error) {
        resp := &http.Response{StatusCode: 200, Header: make(http.Header)}
        resp.Header.Set("Content-Type", "text/event-stream")
        s := ""+
            "data: {\"id\":\"1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi\"}}]}\n\n"+
            "data: {\"id\":\"1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"id\":\"call_1\",\"type\":\"function\",\"index\":0,\"function\":{\"name\":\"sum\"}}]}}]}\n\n"+
            "data: {\"id\":\"1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"type\":\"function\",\"function\":{\"arguments\":\"{\\\"a\\\":1\"}}]}}]}\n\n"+
            "data: {\"id\":\"1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"type\":\"function\",\"function\":{\"arguments\":\",\\\"b\\\":2}\"}}]}}]}\n\n"+
            "data: {\"id\":\"1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"id\":\"call_2\",\"type\":\"function\",\"index\":1,\"function\":{\"name\":\"get_info\"}}]}}]}\n\n"+
            "data: {\"id\":\"1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":1,\"type\":\"function\",\"function\":{\"arguments\":\"{\\\"id\\\":\\\"X\\\"\"}}]}}]}\n\n"+
            "data: {\"id\":\"1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":1,\"type\":\"function\",\"function\":{\"arguments\":\",\\\"q\\\":\\\"qq\\\"}\"}}]}}]}\n\n"+
            "data: [DONE]\n\n"
        resp.Body = io.NopCloser(strings.NewReader(s))
        return resp, nil
    })
    cfg := httpad.Config{ OpenAIBaseURL: "http://openai.local" }
    h := httpad.NewMessagesHandler(cfg, http.DefaultClient)
    areq := ad.AnthropicMessageRequest{ Model: "claude-foo", Stream: true, Messages: []ad.AnthropicMsg{{Role:"user", Content: json.RawMessage(`"Hi"`)}} }
    b, _ := json.Marshal(areq)
    req := httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewReader(b))
    w := httptest.NewRecorder()
    h.ServeHTTP(w, req)
    res := w.Result()
    data, _ := io.ReadAll(res.Body)
    s := string(data)
    if !strings.Contains(s, "\"type\":\"tool_use\"") || !strings.Contains(s, "\"name\":\"sum\"") || !strings.Contains(s, "\"input\":{\"a\":1,\"b\":2}") { t.Fatalf("missing sum tool_use: %s", s) }
    if !strings.Contains(s, "\"type\":\"tool_use\"") || !strings.Contains(s, "\"name\":\"get_info\"") || !strings.Contains(s, "\"input\":{\"id\":\"X\",\"q\":\"qq\"}") { t.Fatalf("missing get_info tool_use: %s", s) }
}

func TestMessagesHandler_Streaming_InvalidArgsBecomeEmptyObject(t *testing.T) {
    prev := http.DefaultTransport
    t.Cleanup(func(){ http.DefaultTransport = prev })
    http.DefaultTransport = roundTripperFunc(func(req *http.Request) (*http.Response, error) {
        resp := &http.Response{StatusCode: 200, Header: make(http.Header)}
        resp.Header.Set("Content-Type", "text/event-stream")
        s := ""+
            "data: {\"id\":\"1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"id\":\"call_bad\",\"type\":\"function\",\"index\":0,\"function\":{\"name\":\"do\"}}]}}]}\n\n"+
            "data: {\"id\":\"1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"type\":\"function\",\"function\":{\"arguments\":\"NOT_JSON\"}}]}}]}\n\n"+
            "data: [DONE]\n\n"
        resp.Body = io.NopCloser(strings.NewReader(s))
        return resp, nil
    })
    cfg := httpad.Config{ OpenAIBaseURL: "http://openai.local" }
    h := httpad.NewMessagesHandler(cfg, http.DefaultClient)
    areq := ad.AnthropicMessageRequest{ Model: "claude-foo", Stream: true, Messages: []ad.AnthropicMsg{{Role:"user", Content: json.RawMessage(`"Hi"`)}} }
    b, _ := json.Marshal(areq)
    req := httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewReader(b))
    w := httptest.NewRecorder()
    h.ServeHTTP(w, req)
    res := w.Result()
    data, _ := io.ReadAll(res.Body)
    s := string(data)
    if !strings.Contains(s, "\"type\":\"tool_use\"") || !strings.Contains(s, "\"input\":{}") { t.Fatalf("expected empty input object when args invalid: %s", s) }
}

// --- Logs-based tests ---

func Test_LogParity_CodexVsClaude_ImageTool(t *testing.T) {
    codexPath := "/home/alejandroseaah/.codex/sessions/2025/10/28/rollout-2025-10-28T00-45-58-70a339ef-0669-4cce-9a68-32d3c8f59a5e.jsonl"
    claudePath := "/home/alejandroseaah/.claude/projects/-home-alejandroseaah-api-adapter/12be9ce7-b8f1-4f60-8d52-41872010610d.jsonl"
    if _, err := os.Stat(codexPath); err != nil { t.Skipf("codex log missing: %v", err) }
    if _, err := os.Stat(claudePath); err != nil { t.Skipf("claude log missing: %v", err) }
    cName, cPath, err := parseCodexToolCall(codexPath)
    if err != nil { t.Fatalf("parse codex: %v", err) }
    aName, aPath, err := parseClaudeToolUse(claudePath)
    if err != nil { t.Fatalf("parse claude: %v", err) }
    if filepath.Base(cPath) != "cc.png" || filepath.Base(aPath) != "cc.png" { t.Fatalf("unexpected file names: codex=%q claude=%q", cPath, aPath) }
    if cName == "" || aName == "" { t.Fatalf("tool names empty: codex=%q claude=%q", cName, aName) }
}

func Test_Mapping_WithLogDerived_ToolUseBothWays(t *testing.T) {
    codexPath := "/home/alejandroseaah/.codex/sessions/2025/10/28/rollout-2025-10-28T00-45-58-70a339ef-0669-4cce-9a68-32d3c8f59a5e.jsonl"
    claudePath := "/home/alejandroseaah/.claude/projects/-home-alejandroseaah-api-adapter/12be9ce7-b8f1-4f60-8d52-41872010610d.jsonl"
    if _, err := os.Stat(codexPath); err != nil { t.Skipf("codex log missing: %v", err) }
    if _, err := os.Stat(claudePath); err != nil { t.Skipf("claude log missing: %v", err) }
    aToolName, aFile, err := parseClaudeToolUse(claudePath)
    if err != nil { t.Fatalf("parse claude: %v", err) }
    input := map[string]any{"file_path": aFile}
    inRaw, _ := json.Marshal(input)
    parts := []ad.AnthropicContent{{Type: "tool_use", ID: "call_log", Name: aToolName, Input: (*json.RawMessage)(&inRaw)}}
    raw, _ := json.Marshal(parts)
    areq := ad.AnthropicMessageRequest{ Model: "claude-x", Messages: []ad.AnthropicMsg{{ Role: "assistant", Content: raw }}, }
    oreq, err := ad.AnthropicToOpenAI(areq)
    if err != nil { t.Fatalf("AnthropicToOpenAI: %v", err) }
    if len(oreq.Messages) != 1 || len(oreq.Messages[0].ToolCalls) != 1 { t.Fatalf("tool_calls missing: %#v", oreq.Messages) }
    tc := oreq.Messages[0].ToolCalls[0]
    if tc.Function.Name != aToolName { t.Fatalf("tool name: %s", tc.Function.Name) }
    var args map[string]any
    if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err != nil { t.Fatalf("args json: %v", err) }
    if args["file_path"] != aFile { t.Fatalf("args mismatch: %#v", args) }

    cToolName, cPath, err := parseCodexToolCall(codexPath)
    if err != nil { t.Fatalf("parse codex: %v", err) }
    oreq2 := ad.OpenAIChatRequest{ Model: "gpt-x", Messages: []ad.OpenAIMessage{{ Role: "assistant", ToolCalls: []ad.OpenAIToolCall{{ ID: "call_from_codex", Type: "function", Function: ad.OpenAIToolCallFunction{Name: cToolName, Arguments: string(mustJSON(map[string]string{"path": cPath}))} }}, }}, }
    areq2, err := ad.OpenAIToAnthropicRequest(oreq2)
    if err != nil { t.Fatalf("OpenAIToAnthropicRequest: %v", err) }
    if len(areq2.Messages) != 1 { t.Fatalf("messages len: %d", len(areq2.Messages)) }
    var parts2 []ad.AnthropicContent
    if err := json.Unmarshal(areq2.Messages[0].Content, &parts2); err != nil { t.Fatalf("unmarshal parts: %v", err) }
    var found bool
    for _, p := range parts2 {
        if p.Type == "tool_use" && p.Name == cToolName && p.Input != nil {
            var in map[string]any
            _ = json.Unmarshal(*p.Input, &in)
            if in["path"] == cPath { found = true; break }
        }
    }
    if !found { t.Fatalf("tool_use not mapped from OpenAI tool_call: %#v", string(areq2.Messages[0].Content)) }
}

func mustJSON(v any) []byte { b, _ := json.Marshal(v); return b }

func TestFromLogs_OpenAIToAnthropic_Roundtrip(t *testing.T) {
    codexPath := "/home/alejandroseaah/.codex/sessions/2025/10/28/rollout-2025-10-28T00-45-58-70a339ef-0669-4cce-9a68-32d3c8f59a5e.jsonl"
    if _, err := os.Stat(codexPath); err != nil { t.Skipf("codex log missing: %v", err) }
    funcName, imgPath, err := parseCodexToolCall(codexPath)
    if err != nil { t.Fatalf("parse codex: %v", err) }
    prev := http.DefaultTransport
    t.Cleanup(func(){ http.DefaultTransport = prev })
    http.DefaultTransport = roundTripperFunc(func(req *http.Request) (*http.Response, error) {
        if req.URL.Path != "/v1/messages" { return &http.Response{StatusCode: 404, Body: io.NopCloser(strings.NewReader(""))}, nil }
        var areq ad.AnthropicMessageRequest
        if err := json.NewDecoder(req.Body).Decode(&areq); err != nil { return &http.Response{StatusCode: 400, Body: io.NopCloser(strings.NewReader("bad json"))}, nil }
        var p0 []ad.AnthropicContent
        _ = json.Unmarshal(areq.Messages[0].Content, &p0)
        var in map[string]any
        _ = json.Unmarshal(*p0[0].Input, &in)
        if p0[0].Name != funcName || in["path"] != imgPath { return &http.Response{StatusCode: 422, Body: io.NopCloser(strings.NewReader("wrong tool_use"))}, nil }
        var p1 []ad.AnthropicContent
        _ = json.Unmarshal(areq.Messages[1].Content, &p1)
        if p1[0].Type != "tool_result" || p1[0].Content.(string) != "OK" { return &http.Response{StatusCode: 422, Body: io.NopCloser(strings.NewReader("wrong tool_result"))}, nil }
        resp := &http.Response{StatusCode: 200, Header: make(http.Header)}
        resp.Header.Set("Content-Type", "application/json")
        resp.Body = io.NopCloser(strings.NewReader(`{"id":"msg_logs","type":"message","role":"assistant","model":"claude-x","content":[{"type":"text","text":"Done logs"}]}`))
        return resp, nil
    })
    cfg := httpad.Config{ AnthropicBaseURL: "http://anth.local" }
    h := httpad.NewChatCompletionsHandler(cfg, http.DefaultClient)
    oreq := ad.OpenAIChatRequest{ Model: "gpt-x", Messages: []ad.OpenAIMessage{{ Role: "assistant", ToolCalls: []ad.OpenAIToolCall{{ ID: "call_logs", Type: "function", Function: ad.OpenAIToolCallFunction{Name: funcName, Arguments: string(mustJSON(map[string]string{"path": imgPath}))} }}, }, {Role: "tool", ToolCallID: "call_logs", Content: "OK"}} }
    b, _ := json.Marshal(oreq)
    req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(b))
    w := httptest.NewRecorder()
    h.ServeHTTP(w, req)
    res := w.Result()
    if res.StatusCode != 200 { body, _ := io.ReadAll(res.Body); t.Fatalf("status: %d body: %s", res.StatusCode, string(body)) }
}

func TestFromLogs_AnthropicToOpenAI_Roundtrip(t *testing.T) {
    claudePath := "/home/alejandroseaah/.claude/projects/-home-alejandroseaah-api-adapter/12be9ce7-b8f1-4f60-8d52-41872010610d.jsonl"
    if _, err := os.Stat(claudePath); err != nil { t.Skipf("claude log missing: %v", err) }
    toolName, filePath, err := parseClaudeToolUse(claudePath)
    if err != nil { t.Fatalf("parse claude: %v", err) }
    prev := http.DefaultTransport
    t.Cleanup(func(){ http.DefaultTransport = prev })
    http.DefaultTransport = roundTripperFunc(func(req *http.Request) (*http.Response, error) {
        if req.URL.Path != "/v1/chat/completions" { return &http.Response{StatusCode: 404, Body: io.NopCloser(strings.NewReader(""))}, nil }
        var oreq ad.OpenAIChatRequest
        if err := json.NewDecoder(req.Body).Decode(&oreq); err != nil { return &http.Response{StatusCode: 400, Body: io.NopCloser(strings.NewReader("bad json"))}, nil }
        var args map[string]any
        _ = json.Unmarshal([]byte(oreq.Messages[0].ToolCalls[0].Function.Arguments), &args)
        if oreq.Messages[0].ToolCalls[0].Function.Name != toolName || args["file_path"] != filePath { return &http.Response{StatusCode: 422, Body: io.NopCloser(strings.NewReader("wrong tool_calls"))}, nil }
        if oreq.Messages[1].Role != "tool" || oreq.Messages[1].ToolCallID != "call_logs" || (oreq.Messages[1].Content.(string)) != "OK" { return &http.Response{StatusCode: 422, Body: io.NopCloser(strings.NewReader("wrong tool result msg"))}, nil }
        resp := &http.Response{StatusCode: 200, Header: make(http.Header)}
        resp.Header.Set("Content-Type", "application/json")
        resp.Body = io.NopCloser(strings.NewReader(`{"id":"chatcmpl_logs","object":"chat.completion","model":"gpt-x","choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"All set logs"}}]}`))
        return resp, nil
    })
    cfg := httpad.Config{ OpenAIBaseURL: "http://openai.local" }
    h := httpad.NewMessagesHandler(cfg, http.DefaultClient)
    in := map[string]any{"file_path": filePath}
    inRaw, _ := json.Marshal(in)
    parts0 := []ad.AnthropicContent{{Type: "tool_use", ID: "call_logs", Name: toolName, Input: (*json.RawMessage)(&inRaw)}}
    raw0, _ := json.Marshal(parts0)
    parts1 := []ad.AnthropicContent{{Type: "tool_result", ToolUseID: "call_logs", Content: "OK"}}
    raw1, _ := json.Marshal(parts1)
    areq := ad.AnthropicMessageRequest{ Model: "claude-x", Messages: []ad.AnthropicMsg{{Role: "assistant", Content: raw0}, {Role: "user", Content: raw1}} }
    b, _ := json.Marshal(areq)
    req := httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewReader(b))
    w := httptest.NewRecorder()
    h.ServeHTTP(w, req)
    res := w.Result()
    if res.StatusCode != 200 { body, _ := io.ReadAll(res.Body); t.Fatalf("status: %d body: %s", res.StatusCode, string(body)) }
}

