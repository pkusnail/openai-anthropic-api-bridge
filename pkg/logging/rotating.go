package logging

import (
    "fmt"
    "io"
    "os"
    "path/filepath"
    "strings"
    "sync"
    "time"
)

// RotatingWriter writes logs to a daily file with optional size-based rollover.
// Files are named: <base>-YYYY-MM-DD[-N].log (UTC date).
type RotatingWriter struct {
    basePath string
    maxBytes int64

    mu       sync.Mutex
    curDate  string
    curIndex int
    f        *os.File
    size     int64
}

func NewRotatingWriter(path string, maxBytes int64) (io.Writer, error) {
    rw := &RotatingWriter{basePath: path, maxBytes: maxBytes}
    if err := rw.rotateIfNeeded(0); err != nil { return nil, err }
    return rw, nil
}

func (w *RotatingWriter) Write(p []byte) (int, error) {
    w.mu.Lock()
    defer w.mu.Unlock()
    if err := w.rotateIfNeeded(len(p)); err != nil { return 0, err }
    n, err := w.f.Write(p)
    if err == nil { w.size += int64(n) }
    return n, err
}

func (w *RotatingWriter) rotateIfNeeded(incoming int) error {
    today := time.Now().UTC().Format("2006-01-02")
    if w.f == nil || w.curDate != today {
        w.curDate = today
        w.curIndex = 1
        return w.openCurrent()
    }
    if w.maxBytes > 0 && w.size+int64(incoming) > w.maxBytes {
        w.curIndex++
        return w.openCurrent()
    }
    return nil
}

func (w *RotatingWriter) openCurrent() error {
    if w.f != nil { _ = w.f.Close() }
    dir, name := filepath.Split(w.basePath)
    if dir == "" { dir = "." }
    _ = os.MkdirAll(dir, 0o755)
    ext := filepath.Ext(name)
    base := strings.TrimSuffix(name, ext)
    if ext == "" { ext = ".log" }
    filename := fmt.Sprintf("%s-%s%s", base, w.curDate, ext)
    if w.curIndex > 1 {
        filename = fmt.Sprintf("%s-%s-%d%s", base, w.curDate, w.curIndex, ext)
    }
    full := filepath.Join(dir, filename)
    f, err := os.OpenFile(full, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
    if err != nil { return err }
    st, _ := f.Stat()
    w.f = f
    if st != nil { w.size = st.Size() } else { w.size = 0 }
    // Update pointer file (best-effort): basePath -> current file path
    tmp := w.basePath + ".tmp"
    if ff, err := os.OpenFile(tmp, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644); err == nil {
        _, _ = fmt.Fprintf(ff, "current log file: %s\n", full)
        _ = ff.Close()
        _ = os.Rename(tmp, w.basePath)
    }
    return nil
}

