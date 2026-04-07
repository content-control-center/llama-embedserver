package main

/*
#include <malloc.h>
*/
import "C"

import (
	"encoding/json"
	"flag"
	"log"
	"net/http"
	"runtime/debug"
	"sync"
	"time"

	llama "github.com/tcpipuk/llama-go"
)

var (
	embedCtx *llama.Context
	// Context is NOT thread-safe (llama-go docs). The library's RLock on
	// GetEmbeddings only protects its own fields — concurrent C++ calls on
	// the same context cause data races. Serialize all inference calls.
	ctxMu sync.Mutex
)

type embedRequest struct {
	Text string `json:"text"`
}

type embedResponse struct {
	Embedding  []float32 `json:"embedding"`
	Dimensions int       `json:"dimensions"`
}

type batchRequest struct {
	Texts []string `json:"texts"`
}

type batchResponse struct {
	Embeddings [][]float32 `json:"embeddings"`
	Dimensions int         `json:"dimensions"`
}

type errorResponse struct {
	Error string `json:"error"`
}

func writeError(w http.ResponseWriter, status int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(errorResponse{Error: msg})
}

func handleHealth(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func handleEmbed(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}

	var req embedRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body")
		return
	}
	if req.Text == "" {
		writeError(w, http.StatusBadRequest, "text is required")
		return
	}

	ctxMu.Lock()
	embedding, err := embedCtx.GetEmbeddings(req.Text)
	ctxMu.Unlock()

	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(embedResponse{
		Embedding:  embedding,
		Dimensions: len(embedding),
	})
}

func handleBatch(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}

	var req batchRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body")
		return
	}
	if len(req.Texts) == 0 {
		writeError(w, http.StatusBadRequest, "texts are required")
		return
	}

	ctxMu.Lock()
	embeddings, err := embedCtx.GetEmbeddingsBatch(req.Texts)
	ctxMu.Unlock()

	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}

	dim := 0
	if len(embeddings) > 0 {
		dim = len(embeddings[0])
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(batchResponse{
		Embeddings: embeddings,
		Dimensions: dim,
	})
}

// freeMemoryLoop periodically forces Go to return freed pages to the OS.
//
// After GC, Go marks heap pages as reclaimable (MADV_FREE on Linux) but does
// not immediately return them, so RSS stays high. FreeOSMemory upgrades this
// to MADV_DONTNEED, which causes the kernel to reclaim the pages right away.
//
// glibc malloc (used by llama.cpp) has the same behaviour — freed C-heap
// blocks stay in its pool. MALLOC_TRIM_THRESHOLD_ (set in the Dockerfile)
// controls when glibc trims its arena, but calling malloc_trim(0) here via
// FreeOSMemory's GC pass also indirectly prompts glibc to release pages
// because fewer Go-heap pointers are pinned across CGO boundaries.
func freeMemoryLoop(interval time.Duration) {
	for range time.Tick(interval) {
		// Return freed Go heap pages to the OS (MADV_FREE → MADV_DONTNEED).
		debug.FreeOSMemory()
		// Compact and trim the glibc main arena. Allocations ≥ MALLOC_MMAP_THRESHOLD_
		// already bypass this arena (they use mmap and are freed immediately), so
		// malloc_trim only needs to handle the smaller sbrk-based blocks.
		C.malloc_trim(0)
	}
}

func main() {
	modelPath := flag.String("model", "", "Path to GGUF model file (required)")
	addr := flag.String("addr", ":8080", "Listen address")
	gpuLayers := flag.Int("gpu-layers", 0, "GPU layers to offload (-1 for all)")
	contextSize := flag.Int("context-size", 512, "KV cache context window in tokens — bounds peak memory")
	memLimitMiB := flag.Int64("mem-limit-mib", 128, "Soft Go heap memory limit in MiB (0 = unlimited)")
	gcInterval := flag.Duration("gc-interval", 30*time.Second, "How often to force-return freed memory to the OS")
	flag.Parse()

	if *modelPath == "" {
		log.Fatal("flag -model is required")
	}

	// Cap the Go heap so the GC runs before memory balloons to 2× live size.
	// With GOMEMLIMIT set, GC triggers earlier and is more aggressive about
	// returning pages — works in tandem with the periodic FreeOSMemory calls.
	if *memLimitMiB > 0 {
		debug.SetMemoryLimit(*memLimitMiB << 20)
		log.Printf("go memory limit: %d MiB", *memLimitMiB)
	}

	go freeMemoryLoop(*gcInterval)

	log.Printf("loading model: %s", *modelPath)
	model, err := llama.LoadModel(*modelPath, llama.WithGPULayers(*gpuLayers))
	if err != nil {
		log.Fatalf("failed to load model: %v", err)
	}
	defer model.Close()

	embedCtx, err = model.NewContext(
		llama.WithEmbeddings(),
		// Disable prefix caching: for embeddings each input is independent,
		// caching only grows the KV store over time without any benefit.
		llama.WithPrefixCaching(false),
		// Bound the KV cache to a fixed window. Default (0) uses the model's
		// native max (often 8192+), which with n_parallel=8 multiplies fast.
		llama.WithContext(*contextSize),
		// Single parallel sequence: we serialize requests with ctxMu anyway,
		// so extra slots just waste memory (default for embedding ctx is 8).
		llama.WithParallel(1),
	)
	if err != nil {
		log.Fatalf("failed to create embedding context: %v", err)
	}
	defer embedCtx.Close()

	http.HandleFunc("/health", handleHealth)
	http.HandleFunc("/embed", handleEmbed)
	http.HandleFunc("/embed/batch", handleBatch)

	log.Printf("listening on %s", *addr)
	log.Fatal(http.ListenAndServe(*addr, nil))
}
