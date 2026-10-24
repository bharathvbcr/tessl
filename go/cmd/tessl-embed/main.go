//go:build darwin && arm64

// Command tessl-embed serves Google EmbeddingGemma 2, run by tessl on Metal,
// behind Ollama's /api/embed protocol, so an Ollama client switches to it by
// base URL and model name alone.
//
//	go run ./cmd/tessl-embed -snapshot ~/.cache/huggingface/hub/models--google--embeddinggemma-2/snapshots/<rev>
//
// It listens on 127.0.0.1 only (default port 11435; Ollama's is 11434) and
// answers to embeddinggemma-2-tessl (the snapshot's default prompt, which for
// this checkpoint is none), embeddinggemma-2-tessl:document,
// embeddinggemma-2-tessl:query, and embeddinggemma-2-tessl:<name> for every
// other prompt the snapshot names. See go/README.md.
//
// Request texts are never logged.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"syscall"
	"time"

	"github.com/bharathvbcr/tessl/go/embedgemma2"
)

const version = "0.1.0"

func main() {
	if err := run(); err != nil {
		log.Printf("tessl-embed: %v", err)
		os.Exit(1)
	}
}

func run() error {
	def := limits{
		MaxBodyBytes:     32 << 20,
		MaxBatch:         embedgemma2.MaxBatch,
		MaxRequestTokens: 1 << 18,
		Timeout:          120 * time.Second,
		MaxConcurrent:    2,
		MaxQueue:         64,
	}
	var (
		snapshot  = flag.String("snapshot", os.Getenv("EMBEDGEMMA2_SNAPSHOT"), "the google/embeddinggemma-2 snapshot directory (default $EMBEDGEMMA2_SNAPSHOT)")
		port      = flag.Int("port", 11435, "TCP port on 127.0.0.1 (never Ollama's 11434)")
		maxTokens = flag.Int("max-tokens", 8192, "per-text token limit, 1 to 32768; longer texts are truncated (or refused with truncate:false)")
		pidfile   = flag.String("pidfile", "", "write the process id here once listening, and remove it on exit")
		lim       = def
	)
	flag.Int64Var(&lim.MaxBodyBytes, "max-body-bytes", def.MaxBodyBytes, "largest request body")
	flag.IntVar(&lim.MaxBatch, "max-batch", def.MaxBatch, fmt.Sprintf("most inputs per request, at most %d", embedgemma2.MaxBatch))
	flag.IntVar(&lim.MaxRequestTokens, "max-request-tokens", def.MaxRequestTokens, "most tokens per request, after truncation")
	flag.DurationVar(&lim.Timeout, "timeout", def.Timeout, "per-request limit, waiting included")
	flag.IntVar(&lim.MaxConcurrent, "max-concurrent", def.MaxConcurrent, "requests working on the model at once")
	flag.IntVar(&lim.MaxQueue, "max-queue", def.MaxQueue, "requests waiting for a working slot; more are answered 503")
	flag.Parse()

	if *snapshot == "" {
		return errors.New("no snapshot: pass -snapshot or set EMBEDGEMMA2_SNAPSHOT")
	}
	if *port < 1 || *port > 65535 || *port == 11434 {
		return fmt.Errorf("port %d: choose 1..65535 other than Ollama's 11434", *port)
	}
	info, err := snapshotInfo(*snapshot)
	if err != nil {
		return err
	}

	// Bind before loading, so a taken port fails in milliseconds, not seconds.
	addr := net.JoinHostPort("127.0.0.1", strconv.Itoa(*port))
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		return err
	}

	t0 := time.Now()
	m, err := embedgemma2.Load(*snapshot, embedgemma2.Options{TextOnly: true})
	if err != nil {
		_ = ln.Close()
		return fmt.Errorf("loading %s: %w", *snapshot, err)
	}
	defer func() {
		if err := m.Close(); err != nil {
			log.Printf("tessl-embed: closing the model: %v", err)
		}
	}()
	ctx := context.Background()
	if *maxTokens != m.MaxTokens() {
		if err := m.SetMaxTokens(ctx, *maxTokens); err != nil {
			_ = ln.Close()
			return err
		}
	}
	prompts, err := m.Prompts(ctx)
	if err != nil {
		_ = ln.Close()
		return err
	}

	fatal := make(chan error, 1)
	s, err := newServer(ctx, m, prompts, lim, info, func(err error) { fatal <- err })
	if err != nil {
		_ = ln.Close()
		return err
	}
	// A warm-up embed, so a snapshot that loads but cannot run fails here, and
	// the first request does not pay for the pipelines.
	if _, err := m.EmbedTexts(ctx, []string{"warm-up"}, "", 0); err != nil {
		_ = ln.Close()
		return fmt.Errorf("warm-up embed: %w", err)
	}
	log.Printf("tessl-embed %s: %s on http://%s (dim %d, max %d tokens, loaded in %v); names %s, %s:document, %s:query",
		version, filepath.Base(info.Digest), addr, m.EmbeddingDim(), m.MaxTokens(), time.Since(t0).Round(time.Millisecond), baseName, baseName, baseName)

	if *pidfile != "" {
		if err := os.WriteFile(*pidfile, []byte(strconv.Itoa(os.Getpid())+"\n"), 0o644); err != nil {
			_ = ln.Close()
			return err
		}
		defer os.Remove(*pidfile)
	}

	srv := &http.Server{
		Handler:           logRequests(s.handler()),
		ReadHeaderTimeout: 10 * time.Second,
		ReadTimeout:       lim.Timeout,
		WriteTimeout:      lim.Timeout + 10*time.Second,
		IdleTimeout:       120 * time.Second,
		MaxHeaderBytes:    64 << 10,
	}
	serveErr := make(chan error, 1)
	go func() { serveErr <- srv.Serve(ln) }()

	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
	var exitErr error
	select {
	case got := <-sig:
		log.Printf("tessl-embed: %v, shutting down", got)
	case err := <-fatal:
		exitErr = fmt.Errorf("the model is unusable, exiting: %w", err)
	case err := <-serveErr:
		return err
	}
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	if err := srv.Shutdown(shutdownCtx); err != nil {
		log.Printf("tessl-embed: shutdown: %v", err)
	}
	return exitErr
}

// snapshotInfo reads what /api/tags reports: the weights' size and time, and
// the snapshot revision (the directory's name in the Hugging Face cache).
func snapshotInfo(dir string) (modelInfo, error) {
	resolved, err := filepath.EvalSymlinks(dir)
	if err != nil {
		return modelInfo{}, fmt.Errorf("snapshot: %w", err)
	}
	st, err := os.Stat(filepath.Join(resolved, "model.safetensors"))
	if err != nil {
		return modelInfo{}, fmt.Errorf("snapshot: %w", err)
	}
	return modelInfo{Size: st.Size(), Digest: filepath.Base(resolved), ModifiedAt: st.ModTime().UTC()}, nil
}

type statusWriter struct {
	http.ResponseWriter
	status int
}

func (w *statusWriter) WriteHeader(code int) {
	w.status = code
	w.ResponseWriter.WriteHeader(code)
}

// logRequests logs method, path, status and time; never a body.
func logRequests(h http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t := time.Now()
		sw := &statusWriter{ResponseWriter: w, status: http.StatusOK}
		h.ServeHTTP(sw, r)
		log.Printf("%s %s %d %v", r.Method, r.URL.Path, sw.status, time.Since(t).Round(time.Microsecond))
	})
}
