package main

import (
    "errors"
    "io"
    "log"
    "net/http"
    "os"
    "path/filepath"
    "strings"

    "claude-openai-adapter/pkg/adapterhttp"
    apilog "claude-openai-adapter/pkg/logging"
)

func env(key, def string) string { v := os.Getenv(key); if v == "" { return def }; return v }

func healthHandler(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusOK); _, _ = w.Write([]byte("ok\n")) }

func setupLogger() {
    level := strings.ToLower(env("ADAPTER_LOG_LEVEL", "info"))
    logPath := strings.TrimSpace(os.Getenv("ADAPTER_LOG_FILE"))
    var out io.Writer = os.Stdout
    if logPath != "" && logPath != "-" {
        // ensure directory exists
        _ = os.MkdirAll(filepath.Dir(logPath), 0o755)
        rot, err := apilog.NewRotatingWriter(logPath, 300*1024*1024) // 300MB per file
        if err == nil {
            out = io.MultiWriter(os.Stdout, rot)
        }
    }
    log.SetOutput(out)
    log.SetFlags(log.LstdFlags | log.Lmicroseconds)
    if level == "debug" {
        adapterhttp.SetDebug(true)
    }
    if strings.ToLower(strings.TrimSpace(env("ADAPTER_LOG_EVENTS", ""))) == "true" || env("ADAPTER_LOG_EVENTS", "") == "1" {
        adapterhttp.SetLogEvents(true)
    }
}

func main() {
    setupLogger()
    cfg := adapterhttp.Config{
        AnthropicBaseURL:   env("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
        AnthropicAPIKey:    os.Getenv("ANTHROPIC_API_KEY"),
        AnthropicVersion:   env("ANTHROPIC_VERSION", "2023-06-01"),
        OpenAIBaseURL:      env("OPENAI_BASE_URL", "https://api.openai.com"),
        OpenAIAPIKey:       os.Getenv("OPENAI_API_KEY"),
        ModelMap:           os.Getenv("MODEL_MAP"),
        DefaultOpenAIModel: env("OPENAI_MODEL", "gpt-4o-mini"),
    }

    client := http.DefaultClient
    mux := http.NewServeMux()
    mux.HandleFunc("/health", healthHandler)
    mux.Handle("/v1/messages", adapterhttp.NewMessagesHandler(cfg, client))
    mux.Handle("/v1/chat/completions", adapterhttp.NewChatCompletionsHandler(cfg, client))

    port := env("ADAPTER_LISTEN", env("PORT", "8080"))
    srv := &http.Server{ Addr: ":" + port, Handler: adapterhttp.Logging(mux) }
    log.Printf("Claude<->OpenAI adapter listening on :%s", port)
    if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) { log.Fatal(err) }
}
