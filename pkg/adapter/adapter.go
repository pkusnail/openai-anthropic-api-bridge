package adapter

import (
    "bufio"
    "context"
    "encoding/json"
    "errors"
    "fmt"
    "io"
    "sort"
    "strings"
    "time"
)

// ============ Anthropic (Claude) message API shapes (subset) ============

type AnthropicMessageRequest struct {
    Model         string            `json:"model"`
    System        json.RawMessage   `json:"system,omitempty"`
    Messages      []AnthropicMsg    `json:"messages"`
    Tools         []AnthropicTool   `json:"tools,omitempty"`
    MaxTokens     int               `json:"max_tokens,omitempty"`
    Temperature   *float64          `json:"temperature,omitempty"`
    StopSequences []string          `json:"stop_sequences,omitempty"`
    Stream        bool              `json:"stream,omitempty"`
}

type AnthropicMsg struct {
    Role    string          `json:"role"`
    Content json.RawMessage `json:"content"` // string or []AnthropicContent
}

type AnthropicContent struct {
    Type       string           `json:"type"`          // text | tool_use | tool_result
    Text       string           `json:"text,omitempty"` // text
    // tool_use
    ID         string           `json:"id,omitempty"`
    Name       string           `json:"name,omitempty"`
    Input      *json.RawMessage `json:"input,omitempty"`
    // tool_result
    ToolUseID  string           `json:"tool_use_id,omitempty"`
    Content    interface{}      `json:"content,omitempty"` // usually string
}

type AnthropicTool struct {
    Name        string                 `json:"name"`
    Description string                 `json:"description,omitempty"`
    InputSchema map[string]interface{} `json:"input_schema"`
}

// Response (non-stream)
type AnthropicMessageResponse struct {
    ID           string                   `json:"id"`
    Type         string                   `json:"type"` // "message"
    Role         string                   `json:"role"` // assistant
    Model        string                   `json:"model"`
    Content      []map[string]interface{} `json:"content"`
    StopReason   *string                  `json:"stop_reason"`
    StopSequence *string                  `json:"stop_sequence"`
    Usage        *AnthropicUsage          `json:"usage,omitempty"`
}

type AnthropicUsage struct {
    InputTokens  int `json:"input_tokens"`
    OutputTokens int `json:"output_tokens"`
}

// ============ OpenAI Chat Completions shapes (subset) ============

type OpenAIChatRequest struct {
    Model       string           `json:"model"`
    Messages    []OpenAIMessage  `json:"messages"`
    Tools       []OpenAITool     `json:"tools,omitempty"`
    Temperature *float64         `json:"temperature,omitempty"`
    MaxTokens   int              `json:"max_tokens,omitempty"`
    Stop        []string         `json:"stop,omitempty"`
    Stream      bool             `json:"stream,omitempty"`
}

type OpenAIMessage struct {
    Role       string           `json:"role"`
    Content    interface{}      `json:"content,omitempty"`      // string or []parts
    Name       string           `json:"name,omitempty"`
    ToolCallID string           `json:"tool_call_id,omitempty"` // for role=tool
    ToolCalls  []OpenAIToolCall `json:"tool_calls,omitempty"`   // for assistant
}

type OpenAITool struct {
    Type     string         `json:"type"` // "function"
    Function OpenAIFunction `json:"function"`
}

type OpenAIFunction struct {
    Name        string                 `json:"name"`
    Description string                 `json:"description,omitempty"`
    Parameters  map[string]interface{} `json:"parameters,omitempty"`
}

type OpenAIToolCallFunction struct {
    Name      string `json:"name"`
    Arguments string `json:"arguments"`
}

type OpenAIToolCall struct {
    ID       string                 `json:"id"`
    Type     string                 `json:"type"` // "function"
    Function OpenAIToolCallFunction `json:"function"`
}

type OpenAIChatResponse struct {
    ID      string `json:"id"`
    Object  string `json:"object"`
    Model   string `json:"model"`
    Choices []struct {
        Index        int           `json:"index"`
        FinishReason string        `json:"finish_reason"`
        Message      OpenAIMessage `json:"message"`
    } `json:"choices"`
    Usage *struct {
        PromptTokens     int `json:"prompt_tokens"`
        CompletionTokens int `json:"completion_tokens"`
        TotalTokens      int `json:"total_tokens"`
    } `json:"usage,omitempty"`
}

// Streaming chunk
type OpenAIStreamChunk struct {
    ID      string `json:"id"`
    Object  string `json:"object"`
    Model   string `json:"model"`
    Choices []struct {
        Index int `json:"index"`
        Delta struct {
            Role      string           `json:"role,omitempty"`
            Content   string           `json:"content,omitempty"`
            ToolCalls []struct {
                ID       string `json:"id,omitempty"`
                Type     string `json:"type"`
                Index    int    `json:"index"`
                Function struct {
                    Name      string `json:"name,omitempty"`
                    Arguments string `json:"arguments,omitempty"`
                } `json:"function"`
            } `json:"tool_calls,omitempty"`
        } `json:"delta"`
        FinishReason string `json:"finish_reason,omitempty"`
    } `json:"choices"`
}

// ============ Utilities & helpers ============

func parseAnthropicContent(raw json.RawMessage) ([]AnthropicContent, bool, error) {
    if len(raw) == 0 || string(raw) == "null" { return nil, false, nil }
    var s string
    if err := json.Unmarshal(raw, &s); err == nil {
        return []AnthropicContent{{Type: "text", Text: s}}, true, nil
    }
    var arr []AnthropicContent
    if err := json.Unmarshal(raw, &arr); err == nil {
        return arr, false, nil
    }
    return nil, false, fmt.Errorf("unsupported content: %s", string(raw))
}

func mapToolsToOpenAI(tools []AnthropicTool) []OpenAITool {
    if len(tools) == 0 { return nil }
    out := make([]OpenAITool, 0, len(tools))
    for _, t := range tools {
        out = append(out, OpenAITool{
            Type: "function",
            Function: OpenAIFunction{
                Name:        t.Name,
                Description: t.Description,
                Parameters:  t.InputSchema,
            },
        })
    }
    return out
}

func systemToOpenAI(sysRaw json.RawMessage) *OpenAIMessage {
    if len(sysRaw) == 0 || string(sysRaw) == "null" { return nil }
    var s string
    if err := json.Unmarshal(sysRaw, &s); err == nil && strings.TrimSpace(s) != "" {
        msg := OpenAIMessage{Role: "system", Content: s}
        return &msg
    }
    parts, _, err := parseAnthropicContent(sysRaw)
    if err == nil && len(parts) > 0 {
        var buf []string
        for _, p := range parts {
            if p.Type == "text" && strings.TrimSpace(p.Text) != "" { buf = append(buf, p.Text) }
        }
        if len(buf) > 0 { msg := OpenAIMessage{Role: "system", Content: strings.Join(buf, "\n\n")}; return &msg }
    }
    return nil
}

// ConvertMessagesToOpenAI builds OpenAI messages from Anthropic message history.
func ConvertMessagesToOpenAI(req AnthropicMessageRequest) ([]OpenAIMessage, error) {
    var out []OpenAIMessage
    if sm := systemToOpenAI(req.System); sm != nil { out = append(out, *sm) }
    for _, m := range req.Messages {
        parts, _, err := parseAnthropicContent(m.Content)
        if err != nil { return nil, err }
        switch m.Role {
        case "user":
            var pendingUserText []string
            flushUser := func() {
                if len(pendingUserText) > 0 {
                    out = append(out, OpenAIMessage{Role: "user", Content: strings.Join(pendingUserText, "\n\n")})
                    pendingUserText = nil
                }
            }
            for _, p := range parts {
                switch p.Type {
                case "text":
                    if strings.TrimSpace(p.Text) != "" { pendingUserText = append(pendingUserText, p.Text) }
                case "tool_result":
                    flushUser()
                    contentStr := ""
                    switch v := p.Content.(type) {
                    case string:
                        contentStr = v
                    case nil:
                        contentStr = ""
                    default:
                        b, _ := json.Marshal(v)
                        contentStr = string(b)
                    }
                    out = append(out, OpenAIMessage{ Role: "tool", ToolCallID: p.ToolUseID, Content: contentStr })
                }
            }
            flushUser()
        case "assistant":
            var textBuf []string
            var toolCalls []OpenAIToolCall
            for _, p := range parts {
                switch p.Type {
                case "text":
                    if p.Text != "" { textBuf = append(textBuf, p.Text) }
                case "tool_use":
                    args := "{}"
                    if p.Input != nil && *p.Input != nil { args = string(*p.Input) }
                    toolCalls = append(toolCalls, OpenAIToolCall{ ID: p.ID, Type: "function", Function: OpenAIToolCallFunction{Name: p.Name, Arguments: args} })
                }
            }
            msg := OpenAIMessage{Role: "assistant"}
            if len(textBuf) > 0 { msg.Content = strings.Join(textBuf, "\n\n") }
            if len(toolCalls) > 0 { msg.ToolCalls = toolCalls }
            out = append(out, msg)
        default:
            // ignore
        }
    }
    return out, nil
}

// AnthropicToOpenAI builds a full OpenAIChatRequest from an AnthropicMessageRequest.
func AnthropicToOpenAI(areq AnthropicMessageRequest) (OpenAIChatRequest, error) {
    msgs, err := ConvertMessagesToOpenAI(areq)
    if err != nil { return OpenAIChatRequest{}, err }
    return OpenAIChatRequest{
        Model:       areq.Model, // model mapping handled by caller if needed
        Messages:    msgs,
        Tools:       mapToolsToOpenAI(areq.Tools),
        Temperature: areq.Temperature,
        MaxTokens:   areq.MaxTokens,
        Stop:        areq.StopSequences,
        Stream:      areq.Stream,
    }, nil
}

// ============ Reverse direction (OpenAI request -> Anthropic request) ============

func mapToolsToAnthropic(tools []OpenAITool) []AnthropicTool {
    if len(tools) == 0 { return nil }
    out := make([]AnthropicTool, 0, len(tools))
    for _, t := range tools {
        if strings.ToLower(t.Type) != "function" { continue }
        out = append(out, AnthropicTool{
            Name:        t.Function.Name,
            Description: t.Function.Description,
            InputSchema: t.Function.Parameters,
        })
    }
    return out
}

// OpenAIToAnthropicRequest converts an OpenAI Chat request to Anthropic Messages request.
func OpenAIToAnthropicRequest(oreq OpenAIChatRequest) (AnthropicMessageRequest, error) {
    var systemStr string
    var msgs []AnthropicMsg
    for _, m := range oreq.Messages {
        switch m.Role {
        case "system":
            if systemStr == "" {
                if s, ok := m.Content.(string); ok {
                    systemStr = s
                } else if arr, ok := m.Content.([]interface{}); ok {
                    var buf []string
                    for _, it := range arr {
                        if mp, ok := it.(map[string]interface{}); ok {
                            if mp["type"] == "text" {
                                if ts, ok := mp["text"].(string); ok && strings.TrimSpace(ts) != "" { buf = append(buf, ts) }
                            }
                        }
                    }
                    if len(buf) > 0 { systemStr = strings.Join(buf, "\n\n") }
                }
            }
        case "user":
            if s, ok := m.Content.(string); ok {
                arr := []AnthropicContent{{Type: "text", Text: s}}
                raw, _ := json.Marshal(arr)
                msgs = append(msgs, AnthropicMsg{Role: "user", Content: raw})
            } else if arr, ok := m.Content.([]interface{}); ok {
                var parts []AnthropicContent
                for _, it := range arr {
                    if mp, ok := it.(map[string]interface{}); ok {
                        if mp["type"] == "text" {
                            if ts, ok := mp["text"].(string); ok && strings.TrimSpace(ts) != "" { parts = append(parts, AnthropicContent{Type:"text", Text: ts}) }
                        }
                    }
                }
                if len(parts) > 0 { raw, _ := json.Marshal(parts); msgs = append(msgs, AnthropicMsg{Role:"user", Content: raw}) }
            }
        case "assistant":
            var parts []AnthropicContent
            if s, ok := m.Content.(string); ok && strings.TrimSpace(s) != "" { parts = append(parts, AnthropicContent{Type: "text", Text: s}) }
            if arr, ok := m.Content.([]interface{}); ok {
                for _, it := range arr {
                    if mp, ok := it.(map[string]interface{}); ok {
                        if mp["type"] == "text" {
                            if ts, ok := mp["text"].(string); ok && strings.TrimSpace(ts) != "" { parts = append(parts, AnthropicContent{Type:"text", Text: ts}) }
                        }
                    }
                }
            }
            for _, tc := range m.ToolCalls {
                var inRaw json.RawMessage
                if tc.Function.Arguments != "" { inRaw = json.RawMessage([]byte(tc.Function.Arguments)) }
                parts = append(parts, AnthropicContent{Type: "tool_use", ID: tc.ID, Name: tc.Function.Name, Input: &inRaw})
            }
            if len(parts) > 0 { raw, _ := json.Marshal(parts); msgs = append(msgs, AnthropicMsg{Role: "assistant", Content: raw}) }
        case "tool":
            var contentStr string
            switch v := m.Content.(type) {
            case string:
                contentStr = v
            case nil:
                contentStr = ""
            default:
                b, _ := json.Marshal(v)
                contentStr = string(b)
            }
            parts := []AnthropicContent{{Type: "tool_result", ToolUseID: m.ToolCallID, Content: contentStr}}
            raw, _ := json.Marshal(parts)
            msgs = append(msgs, AnthropicMsg{Role: "user", Content: raw})
        }
    }
    var sysRaw json.RawMessage
    if systemStr != "" { sysRaw = json.RawMessage([]byte(strconvQuote(systemStr))) }
    return AnthropicMessageRequest{
        Model:         oreq.Model,
        System:        sysRaw,
        Messages:      msgs,
        Tools:         mapToolsToAnthropic(oreq.Tools),
        MaxTokens:     oreq.MaxTokens,
        Temperature:   oreq.Temperature,
        StopSequences: oreq.Stop,
        Stream:        oreq.Stream,
    }, nil
}

func strconvQuote(s string) string { b, _ := json.Marshal(s); return string(b) }

// AnthropicToOpenAIResponse converts Anthropic non-streaming response to OpenAI format.
func AnthropicToOpenAIResponse(a AnthropicMessageResponse, openaiModel string) (OpenAIChatResponse, error) {
    var contentStr string
    var toolCalls []OpenAIToolCall
    for _, c := range a.Content {
        if t, ok := c["type"].(string); ok {
            switch t {
            case "text":
                if s, ok := c["text"].(string); ok {
                    if contentStr == "" { contentStr = s } else { contentStr += "\n\n" + s }
                }
            case "tool_use":
                name, _ := c["name"].(string)
                id, _ := c["id"].(string)
                args := "{}"
                if in, ok := c["input"]; ok && in != nil {
                    b, _ := json.Marshal(in)
                    if len(b) > 0 { args = string(b) }
                }
                toolCalls = append(toolCalls, OpenAIToolCall{ID: id, Type: "function", Function: OpenAIToolCallFunction{Name: name, Arguments: args}})
            }
        }
    }
    msg := OpenAIMessage{Role: "assistant"}
    if contentStr != "" { msg.Content = contentStr }
    if len(toolCalls) > 0 { msg.ToolCalls = toolCalls }
    finish := "stop"
    if a.StopReason != nil && *a.StopReason == "tool_use" { finish = "tool_calls" }
    return OpenAIChatResponse{
        ID:     a.ID,
        Object: "chat.completion",
        Model:  openaiModel,
        Choices: []struct {
            Index        int           `json:"index"`
            FinishReason string        `json:"finish_reason"`
            Message      OpenAIMessage `json:"message"`
        }{{Index: 0, FinishReason: finish, Message: msg}},
        Usage: nil,
    }, nil
}

// OpenAIToAnthropic maps a non-streaming OpenAI response to Anthropic message.
func OpenAIToAnthropic(oresp OpenAIChatResponse, requestedModel string) (AnthropicMessageResponse, error) {
    return mapOpenAIToAnthropic(oresp, requestedModel)
}

func mapOpenAIToAnthropic(oresp OpenAIChatResponse, requestedModel string) (AnthropicMessageResponse, error) {
    if len(oresp.Choices) == 0 { return AnthropicMessageResponse{}, fmt.Errorf("no choices") }
    choice := oresp.Choices[0]
    content := make([]map[string]interface{}, 0, 2)
    if s, ok := choice.Message.Content.(string); ok && s != "" {
        content = append(content, map[string]interface{}{"type": "text", "text": s})
    } else if arr, ok := choice.Message.Content.([]interface{}); ok {
        var buf []string
        for _, it := range arr {
            if mp, ok := it.(map[string]interface{}); ok {
                if mp["type"] == "text" {
                    if ts, ok := mp["text"].(string); ok && strings.TrimSpace(ts) != "" { buf = append(buf, ts) }
                }
            }
        }
        if len(buf) > 0 { content = append(content, map[string]interface{}{"type":"text","text": strings.Join(buf, "\n\n")}) }
    }
    for _, tc := range choice.Message.ToolCalls {
        var argsObj interface{}
        if json.Valid([]byte(tc.Function.Arguments)) {
            if err := json.Unmarshal([]byte(tc.Function.Arguments), &argsObj); err != nil { argsObj = map[string]interface{}{"_": tc.Function.Arguments} }
        } else {
            argsObj = map[string]interface{}{"_": tc.Function.Arguments}
        }
        content = append(content, map[string]interface{}{"type": "tool_use", "id": tc.ID, "name": tc.Function.Name, "input": argsObj})
    }
    var stopReason *string
    if choice.FinishReason != "" {
        sr := choice.FinishReason
        if len(choice.Message.ToolCalls) > 0 { sr = "tool_use" }
        stopReason = &sr
    }
    var usage *AnthropicUsage
    if oresp.Usage != nil { usage = &AnthropicUsage{InputTokens: oresp.Usage.PromptTokens, OutputTokens: oresp.Usage.CompletionTokens} }
    return AnthropicMessageResponse{ ID: fmt.Sprintf("msg_%d", time.Now().UnixNano()), Type: "message", Role: "assistant", Model: requestedModel, Content: content, StopReason: stopReason, StopSequence: nil, Usage: usage }, nil
}

// ============ Streaming conversions ============

// ConvertOpenAIStreamToAnthropic converts OpenAI SSE chunks to Anthropic-style events via enc callback.
func ConvertOpenAIStreamToAnthropic(ctx context.Context, requestedModel string, body io.Reader, enc func(event string, payload interface{})) error {
    enc("message_start", map[string]interface{}{"type": "message_start", "message": map[string]interface{}{"id": fmt.Sprintf("msg_%d", time.Now().UnixNano()), "type": "message", "role": "assistant", "model": requestedModel, "content": []interface{}{}}})
    sentTextStart := false
    totalText := ""
    type toolBuf struct{ id, name string; idx int; args string }
    toolByIdx := map[int]*toolBuf{}
    reader := bufio.NewReader(body)
    for {
        select { case <-ctx.Done(): return ctx.Err(); default: }
        line, err := reader.ReadString('\n')
        if err != nil { if errors.Is(err, io.EOF) { break }; break }
        line = strings.TrimSpace(line)
        if line == "" || !strings.HasPrefix(line, "data: ") { continue }
        payload := strings.TrimPrefix(line, "data: ")
        if payload == "[DONE]" { break }
        var chunk OpenAIStreamChunk
        if err := json.Unmarshal([]byte(payload), &chunk); err != nil { continue }
        if len(chunk.Choices) == 0 { continue }
        d := chunk.Choices[0].Delta
        if d.Content != "" {
            if !sentTextStart {
                enc("content_block_start", map[string]interface{}{"type": "content_block_start", "index": 0, "content_block": map[string]interface{}{"type": "text", "text": ""}})
                sentTextStart = true
            }
            totalText += d.Content
            enc("content_block_delta", map[string]interface{}{"type": "content_block_delta", "index": 0, "delta": map[string]interface{}{"type": "text_delta", "text": d.Content}})
        }
        if len(d.ToolCalls) > 0 {
            for _, tc := range d.ToolCalls {
                b, ok := toolByIdx[tc.Index]
                if !ok { b = &toolBuf{idx: tc.Index}; toolByIdx[tc.Index] = b }
                if tc.ID != "" { b.id = tc.ID }
                if tc.Function.Name != "" { b.name = tc.Function.Name }
                if tc.Function.Arguments != "" { b.args += tc.Function.Arguments }
            }
        }
    }
    if sentTextStart { enc("content_block_stop", map[string]interface{}{"type": "content_block_stop", "index": 0}) }
    if len(toolByIdx) > 0 {
        idxs := make([]int, 0, len(toolByIdx))
        for k := range toolByIdx { idxs = append(idxs, k) }
        sort.Ints(idxs)
        for i, idx := range idxs {
            b := toolByIdx[idx]
            var inputObj interface{} = map[string]interface{}{}
            if strings.TrimSpace(b.args) != "" && json.Valid([]byte(b.args)) {
                var tmp interface{}
                if err := json.Unmarshal([]byte(b.args), &tmp); err == nil { inputObj = tmp }
            }
            enc("content_block_start", map[string]interface{}{"type": "content_block_start", "index": i + 1, "content_block": map[string]interface{}{"type": "tool_use", "id": b.id, "name": b.name, "input": inputObj}})
            enc("content_block_stop", map[string]interface{}{"type": "content_block_stop", "index": i + 1})
        }
    }
    enc("message_delta", map[string]interface{}{
        "type":  "message_delta",
        "delta": map[string]interface{}{"stop_reason": "end_turn"},
        "usage": map[string]int{"input_tokens": 0, "output_tokens": len(totalText) / 4},
    })
    enc("message_stop", map[string]interface{}{"type": "message_stop"})
    return nil
}

// ConvertAnthropicStreamToOpenAI converts Anthropic SSE events to OpenAI streaming chunks.
func ConvertAnthropicStreamToOpenAI(ctx context.Context, openaiModel string, body io.Reader, emit func(chunk map[string]interface{})) error {
    roleSent := false
    nextToolIdx := 0
    contentIdxToToolIdx := map[int]int{}
    toolArgsByToolIdx := map[int]string{}
    reader := bufio.NewReader(body)
    send := func(delta map[string]interface{}, finishReason string) {
        ch := map[string]interface{}{"id": fmt.Sprintf("chatcmplchunk_%d", time.Now().UnixNano()), "object": "chat.completion.chunk", "model": openaiModel, "choices": []map[string]interface{}{{"index": 0, "delta": delta}}}
        if finishReason != "" { ch["choices"].([]map[string]interface{})[0]["finish_reason"] = finishReason }
        emit(ch)
    }
    for {
        select { case <-ctx.Done(): return ctx.Err(); default: }
        line, err := reader.ReadString('\n')
        if err != nil { if errors.Is(err, io.EOF) { break }; break }
        line = strings.TrimSpace(line)
        if line == "" { continue }
        if !strings.HasPrefix(line, "event:") { continue }
        ev := strings.TrimSpace(strings.TrimPrefix(line, "event:"))
        dataLine, err2 := reader.ReadString('\n')
        if err2 != nil { break }
        if !strings.HasPrefix(dataLine, "data:") { continue }
        payload := strings.TrimSpace(strings.TrimPrefix(dataLine, "data:"))
        switch ev {
        case "message_start":
            if !roleSent { send(map[string]interface{}{"role": "assistant"}, ""); roleSent = true }
        case "content_block_start":
            var obj struct { Type string `json:"type"`; Index int `json:"index"`; ContentBlock map[string]interface{} `json:"content_block"` }
            if err := json.Unmarshal([]byte(payload), &obj); err != nil { continue }
            if t, _ := obj.ContentBlock["type"].(string); t == "tool_use" {
                id, _ := obj.ContentBlock["id"].(string)
                name, _ := obj.ContentBlock["name"].(string)
                toolIdx := nextToolIdx
                contentIdxToToolIdx[obj.Index] = toolIdx
                delta := map[string]interface{}{"tool_calls": []map[string]interface{}{{"id": id, "type": "function", "index": toolIdx, "function": map[string]interface{}{"name": name}}}}
                send(delta, "")
                nextToolIdx++
            }
        case "content_block_delta":
            var obj struct { Type string `json:"type"`; Index int `json:"index"`; Delta map[string]interface{} `json:"delta"` }
            if err := json.Unmarshal([]byte(payload), &obj); err != nil { continue }
            if obj.Delta == nil { continue }
            if obj.Delta["type"] == "text_delta" {
                if s, _ := obj.Delta["text"].(string); s != "" { send(map[string]interface{}{"content": s}, "") }
            } else if obj.Delta["type"] == "input_json_delta" {
                piece, _ := obj.Delta["partial_json"].(string)
                if piece == "" { if v, ok := obj.Delta["delta"].(string); ok { piece = v } }
                toolIdx, ok := contentIdxToToolIdx[obj.Index]
                if !ok { continue }
                toolArgsByToolIdx[toolIdx] += piece
                delta := map[string]interface{}{"tool_calls": []map[string]interface{}{{"index": toolIdx, "type": "function", "function": map[string]interface{}{"arguments": piece}}}}
                send(delta, "")
            }
        case "message_delta":
            // ignore for now
        case "message_stop":
            send(map[string]interface{}{}, "stop")
        }
    }
    return nil
}
