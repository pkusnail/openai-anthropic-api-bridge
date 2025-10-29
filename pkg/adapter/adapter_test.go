package adapter_test

import (
    "context"
    "encoding/json"
    "strings"
    "testing"

    ad "claude-openai-adapter/pkg/adapter"
)

func mustRaw(s string) json.RawMessage { return json.RawMessage([]byte(s)) }

func TestConvertMessages_SimpleText(t *testing.T) {
    req := ad.AnthropicMessageRequest{
        Model:  "claude-sonnet-4-20250514",
        System: json.RawMessage(`"You are helpful"`),
        Messages: []ad.AnthropicMsg{
            {Role: "user", Content: json.RawMessage(`[{"type":"text","text":"Hello"}]`)},
        },
    }
    msgs, err := ad.ConvertMessagesToOpenAI(req)
    if err != nil { t.Fatalf("convert failed: %v", err) }
    if len(msgs) != 2 { t.Fatalf("expected 2 msgs, got %d", len(msgs)) }
    if msgs[0].Role != "system" || msgs[0].Content.(string) != "You are helpful" { t.Fatalf("bad system: %#v", msgs[0]) }
    if msgs[1].Role != "user" || msgs[1].Content.(string) != "Hello" { t.Fatalf("bad user: %#v", msgs[1]) }
}

func TestSystem_AsArrayOfText(t *testing.T) {
    req := ad.AnthropicMessageRequest{
        System: json.RawMessage(`[{"type":"text","text":"A"},{"type":"text","text":"B"}]`),
        Messages: []ad.AnthropicMsg{{Role:"user", Content: json.RawMessage(`"hi"`) }},
    }
    msgs, err := ad.ConvertMessagesToOpenAI(req)
    if err != nil { t.Fatalf("convert failed: %v", err) }
    if len(msgs) < 2 { t.Fatalf("expected at least 2 msgs, got %d", len(msgs)) }
    if msgs[0].Role != "system" { t.Fatalf("role: %s", msgs[0].Role) }
    if got := msgs[0].Content.(string); got != "A\n\nB" { t.Fatalf("system content: %q", got) }
}

func TestConvertMessages_UserWithToolResultOrdering(t *testing.T) {
    req := ad.AnthropicMessageRequest{
        Messages: []ad.AnthropicMsg{
            {Role: "user", Content: mustRaw(`[
                {"type":"text","text":"A"},
                {"type":"tool_result","tool_use_id":"call_1","content":"RESULT"},
                {"type":"text","text":"B"}
            ]`)},
        },
    }
    msgs, err := ad.ConvertMessagesToOpenAI(req)
    if err != nil { t.Fatalf("convert failed: %v", err) }
    if len(msgs) != 3 { t.Fatalf("expected 3 msgs, got %d (%#v)", len(msgs), msgs) }
    if msgs[0].Role != "user" || msgs[0].Content.(string) != "A" { t.Fatalf("bad[0]: %#v", msgs[0]) }
    if msgs[1].Role != "tool" || msgs[1].ToolCallID != "call_1" || msgs[1].Content.(string) != "RESULT" { t.Fatalf("bad[1]: %#v", msgs[1]) }
    if msgs[2].Role != "user" || msgs[2].Content.(string) != "B" { t.Fatalf("bad[2]: %#v", msgs[2]) }
}

func TestConvertMessages_AssistantWithToolUse(t *testing.T) {
    req := ad.AnthropicMessageRequest{
        Messages: []ad.AnthropicMsg{
            {Role: "assistant", Content: mustRaw(`[
                {"type":"text","text":"Thinking"},
                {"type":"tool_use","id":"call_2","name":"search","input":{"q":"x"}}
            ]`)},
        },
    }
    msgs, err := ad.ConvertMessagesToOpenAI(req)
    if err != nil { t.Fatalf("convert failed: %v", err) }
    if len(msgs) != 1 { t.Fatalf("expected 1 msg, got %d", len(msgs)) }
    if msgs[0].Role != "assistant" { t.Fatalf("role: %s", msgs[0].Role) }
    if msgs[0].Content.(string) != "Thinking" { t.Fatalf("text: %#v", msgs[0].Content) }
    if len(msgs[0].ToolCalls) != 1 { t.Fatalf("tool_calls: %#v", msgs[0].ToolCalls) }
    if msgs[0].ToolCalls[0].ID != "call_2" || msgs[0].ToolCalls[0].Function.Name != "search" { t.Fatalf("tool call not mapped: %#v", msgs[0].ToolCalls[0]) }
}

func TestAnthropicToOpenAI_BuildsRequestFields(t *testing.T) {
    temp := 0.3
    areq := ad.AnthropicMessageRequest{
        Model: "claude-x",
        Temperature: &temp,
        MaxTokens: 128,
        StopSequences: []string{"STOP"},
        Tools: []ad.AnthropicTool{{Name:"sum", InputSchema: map[string]interface{}{"type":"object"}}},
        Messages: []ad.AnthropicMsg{{Role:"user", Content: json.RawMessage(`"hi"`) }},
    }
    oreq, err := ad.AnthropicToOpenAI(areq)
    if err != nil { t.Fatalf("AnthropicToOpenAI: %v", err) }
    if oreq.MaxTokens != 128 || oreq.Temperature == nil || *oreq.Temperature != temp { t.Fatalf("sampling fields wrong: %#v", oreq) }
    if len(oreq.Stop) != 1 || oreq.Stop[0] != "STOP" { t.Fatalf("stop wrong: %#v", oreq.Stop) }
    if len(oreq.Tools) != 1 || oreq.Tools[0].Function.Name != "sum" { t.Fatalf("tools wrong: %#v", oreq.Tools) }
}

func TestConvertMessages_AssistantToolOnly(t *testing.T) {
    areq := ad.AnthropicMessageRequest{
        Messages: []ad.AnthropicMsg{{Role:"assistant", Content: mustRaw(`[{"type":"tool_use","id":"call_x","name":"search","input":{}}]`) }},
    }
    msgs, err := ad.ConvertMessagesToOpenAI(areq)
    if err != nil { t.Fatalf("convert: %v", err) }
    if len(msgs) != 1 { t.Fatalf("len: %d", len(msgs)) }
    if msgs[0].Role != "assistant" || len(msgs[0].ToolCalls) != 1 { t.Fatalf("assistant tool only wrong: %#v", msgs[0]) }
}

func TestOpenAIToAnthropicRequest_ToolResultMapping(t *testing.T) {
    oreq := ad.OpenAIChatRequest{
        Model: "gpt-x",
        Messages: []ad.OpenAIMessage{
            {Role: "assistant", ToolCalls: []ad.OpenAIToolCall{{ ID: "call_42", Type: "function", Function: ad.OpenAIToolCallFunction{Name: "Read", Arguments: `{"path":"/tmp/a.png"}`}}}},
            {Role: "tool", ToolCallID: "call_42", Content: "OK"},
        },
    }
    areq, err := ad.OpenAIToAnthropicRequest(oreq)
    if err != nil { t.Fatalf("OpenAIToAnthropicRequest: %v", err) }
    if len(areq.Messages) != 2 { t.Fatalf("messages len: %d", len(areq.Messages)) }
    var parts0 []ad.AnthropicContent
    if err := json.Unmarshal(areq.Messages[0].Content, &parts0); err != nil { t.Fatalf("parts0: %v", err) }
    if len(parts0) != 1 || parts0[0].Type != "tool_use" || parts0[0].ID != "call_42" || parts0[0].Name != "Read" { t.Fatalf("assistant tool_use wrong: %#v", parts0) }
    var parts1 []ad.AnthropicContent
    if err := json.Unmarshal(areq.Messages[1].Content, &parts1); err != nil { t.Fatalf("parts1: %v", err) }
    if len(parts1) != 1 || parts1[0].Type != "tool_result" || parts1[0].ToolUseID != "call_42" { t.Fatalf("user tool_result wrong: %#v", parts1) }
    if s, ok := parts1[0].Content.(string); !ok || s != "OK" { t.Fatalf("tool_result content: %#v", parts1[0].Content) }
}

func TestOpenAIToAnthropic_InvalidArgsFallback(t *testing.T) {
    oresp := ad.OpenAIChatResponse{
        Choices: []struct{ Index int `json:"index"`; FinishReason string `json:"finish_reason"`; Message ad.OpenAIMessage `json:"message"` }{
            {Index:0, FinishReason:"stop", Message: ad.OpenAIMessage{Role:"assistant", ToolCalls: []ad.OpenAIToolCall{{ ID:"t1", Type:"function", Function: ad.OpenAIToolCallFunction{Name:"weird", Arguments:"not_json"} }}}},
        },
        Usage: nil,
    }
    aresp, err := ad.OpenAIToAnthropic(oresp, "claude-x")
    if err != nil { t.Fatalf("OpenAIToAnthropic: %v", err) }
    if len(aresp.Content) != 1 { t.Fatalf("content len: %d", len(aresp.Content)) }
    c := aresp.Content[0]
    if c["type"] != "tool_use" || c["name"] != "weird" { t.Fatalf("tool_use wrong: %#v", c) }
    in, _ := c["input"].(map[string]interface{})
    if in["_"] != "not_json" { t.Fatalf("fallback input wrong: %#v", in) }
}

func TestConvertAnthropicStreamToOpenAI_ToolCallIndexPresence(t *testing.T) {
    s := ""+
        "event: message_start\n"+
        "data: {\"type\":\"message_start\",\"message\":{}}\n\n"+
        "event: content_block_start\n"+
        "data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"call_x\",\"name\":\"alpha\"}}\n\n"+
        "event: content_block_delta\n"+
        "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"a\\\":1}\"}}\n\n"+
        "event: message_stop\n"+
        "data: {\"type\":\"message_stop\"}\n\n"
    var chunks []ad.OpenAIStreamChunk
    _ = ad.ConvertAnthropicStreamToOpenAI(context.Background(), "gpt-x", strings.NewReader(s), func(m map[string]interface{}){
        b, _ := json.Marshal(m)
        var c ad.OpenAIStreamChunk
        _ = json.Unmarshal(b, &c)
        chunks = append(chunks, c)
    })
    var nameIdx, argIdx = -1, -1
    for _, c := range chunks {
        for _, tc := range c.Choices[0].Delta.ToolCalls {
            if tc.Function.Name == "alpha" { nameIdx = tc.Index }
            if tc.Function.Arguments != "" { argIdx = tc.Index }
        }
    }
    if nameIdx == -1 || argIdx == -1 { t.Fatalf("missing tool_calls name/args deltas: %#v", chunks) }
    if nameIdx != argIdx { t.Fatalf("indices should match: name=%d args=%d", nameIdx, argIdx) }
}

func TestConvertAnthropicStreamToOpenAI_InterleavedTwoTools(t *testing.T) {
    s := ""+
        "event: message_start\n"+
        "data: {\"type\":\"message_start\",\"message\":{}}\n\n"+
        "event: content_block_start\n"+
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\"}}\n\n"+
        "event: content_block_delta\n"+
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi\"}}\n\n"+
        "event: content_block_stop\n"+
        "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n"+
        "event: content_block_start\n"+
        "data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"call_a\",\"name\":\"alpha\"}}\n\n"+
        "event: content_block_delta\n"+
        "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"a\\\":1}\"}}\n\n"+
        "event: content_block_start\n"+
        "data: {\"type\":\"content_block_start\",\"index\":2,\"content_block\":{\"type\":\"tool_use\",\"id\":\"call_b\",\"name\":\"beta\"}}\n\n"+
        "event: content_block_delta\n"+
        "data: {\"type\":\"content_block_delta\",\"index\":2,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"q\\\":\\\"x\\\"}\"}}\n\n"+
        "event: message_stop\n"+
        "data: {\"type\":\"message_stop\"}\n\n"
    var chunks []ad.OpenAIStreamChunk
    _ = ad.ConvertAnthropicStreamToOpenAI(context.Background(), "gpt-x", strings.NewReader(s), func(m map[string]interface{}){
        b, _ := json.Marshal(m)
        var c ad.OpenAIStreamChunk
        _ = json.Unmarshal(b, &c)
        chunks = append(chunks, c)
    })
    var sawAlpha, sawBeta, sawAArg, sawBArg, sawText bool
    for _, c := range chunks {
        if c.Choices == nil || len(c.Choices) == 0 { continue }
        d := c.Choices[0].Delta
        if d.Content != "" && strings.Contains(d.Content, "Hi") { sawText = true }
        for _, tc := range d.ToolCalls {
            if tc.Function.Name == "alpha" { sawAlpha = true }
            if tc.Function.Name == "beta" { sawBeta = true }
            if strings.Contains(tc.Function.Arguments, "\"a\":1") { sawAArg = true }
            if strings.Contains(tc.Function.Arguments, "\"q\":\"x\"") { sawBArg = true }
        }
    }
    if !(sawText && sawAlpha && sawBeta && sawAArg && sawBArg) {
        t.Fatalf("missing pieces: text=%v alpha=%v beta=%v aArg=%v bArg=%v", sawText, sawAlpha, sawBeta, sawAArg, sawBArg)
    }
}

