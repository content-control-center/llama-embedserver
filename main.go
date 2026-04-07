package main

/*
#include <malloc.h>
*/
import "C"

import (
	"encoding/json"
	"flag"
	"log"
	"net"
	"net/http"
	"runtime/debug"
	"sync"
	"time"

	llama "github.com/tcpipuk/llama-go"
	"google.golang.org/grpc"
)

// Embedder is the inference capability the server depends on.
// *llama.Context satisfies this interface directly.
type Embedder interface {
	GetEmbeddings(text string) ([]float32, error)
	GetEmbeddingsBatch(texts []string) ([][]float32, error)
}

type server struct {
	embedder   Embedder
	dimensions int
	// mu serializes all calls into the llama.cpp context.
	// Context is NOT thread-safe (llama-go docs) — the library's RLock on
	// GetEmbeddings only protects Go fields, not the underlying C++ state.
	mu sync.Mutex
}

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

type infoResponse struct {
	Dimensions int `json:"dimensions"`
}

type errorResponse struct {
	Error string `json:"error"`
}

func writeError(w http.ResponseWriter, status int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(errorResponse{Error: msg})
}

func (s *server) handleHealth(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (s *server) handleInfo(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(infoResponse{Dimensions: s.dimensions})
}

func (s *server) handleEmbed(w http.ResponseWriter, r *http.Request) {
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

	s.mu.Lock()
	embedding, err := s.embedder.GetEmbeddings(req.Text)
	s.mu.Unlock()

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

func (s *server) handleBatch(w http.ResponseWriter, r *http.Request) {
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

	s.mu.Lock()
	embeddings, err := s.embedder.GetEmbeddingsBatch(req.Texts)
	s.mu.Unlock()

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
// malloc_trim(0) compacts the glibc sbrk arena for any remaining small
// allocations not covered by MALLOC_MMAP_THRESHOLD_. jemalloc (injected via
// LD_PRELOAD in entrypoint.sh) provides a compatible implementation and
// triggers its own background decay in response.
func freeMemoryLoop(interval time.Duration) {
	for range time.Tick(interval) {
		debug.FreeOSMemory()
		C.malloc_trim(0)
	}
}

func main() {
	modelPath := flag.String("model", "", "Path to GGUF model file (required)")
	addr := flag.String("addr", ":8080", "HTTP listen address")
	grpcAddr := flag.String("grpc-addr", ":9090", "gRPC listen address")
	gpuLayers := flag.Int("gpu-layers", 0, "GPU layers to offload (-1 for all)")
	contextSize := flag.Int("context-size", 512, "KV cache context window in tokens — bounds peak memory")
	threads := flag.Int("threads", 2, "CPU threads for llama.cpp inference — requests are serialized so more than 2-4 rarely helps")
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

	embedCtx, err := model.NewContext(
		llama.WithEmbeddings(),
		// Disable prefix caching: for embeddings each input is independent,
		// caching only grows the KV store over time without any benefit.
		llama.WithPrefixCaching(false),
		// Bound the KV cache to a fixed window. Default (0) uses the model's
		// native max (often 8192+), which with n_parallel=8 multiplies fast.
		llama.WithContext(*contextSize),
		// Single parallel sequence: we serialize requests with mu anyway,
		// so extra slots just waste memory (default for embedding ctx is 8).
		llama.WithParallel(1),
		// Cap the llama.cpp compute thread pool. Requests are serialized so
		// all threads serve a single inference at a time — NumCPU() threads
		// just inflate the thread count with no throughput benefit.
		llama.WithThreads(*threads),
	)
	if err != nil {
		log.Fatalf("failed to create embedding context: %v", err)
	}
	defer embedCtx.Close()

	// Probe embed: discovers the model's embedding dimensionality at startup.
	// A single short text is enough; the result is discarded.
	probe, err := embedCtx.GetEmbeddings("a")
	if err != nil {
		log.Fatalf("probe embed (dimension discovery): %v", err)
	}
	log.Printf("embedding dimensions: %d", len(probe))

	srv := &server{embedder: embedCtx, dimensions: len(probe)}

	// gRPC server — shares srv (and its mutex) with the HTTP server.
	go func() {
		lis, err := net.Listen("tcp", *grpcAddr)
		if err != nil {
			log.Fatalf("gRPC listen: %v", err)
		}
		grpcSrv := grpc.NewServer()
		RegisterEmbedServiceServer(grpcSrv, &grpcServer{srv: srv})
		log.Printf("gRPC listening on %s", *grpcAddr)
		if err := grpcSrv.Serve(lis); err != nil {
			log.Fatalf("gRPC serve: %v", err)
		}
	}()

	http.HandleFunc("/health", srv.handleHealth)
	http.HandleFunc("/info", srv.handleInfo)
	http.HandleFunc("/embed", srv.handleEmbed)
	http.HandleFunc("/embed/batch", srv.handleBatch)

	log.Printf("HTTP listening on %s", *addr)
	log.Fatal(http.ListenAndServe(*addr, nil))
}
