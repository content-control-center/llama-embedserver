package main

import (
	"encoding/json"
	"flag"
	"log"
	"net/http"

	llama "github.com/tcpipuk/llama-go"
)

var embedCtx *llama.Context

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

	embedding, err := embedCtx.GetEmbeddings(req.Text)
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

	embeddings, err := embedCtx.GetEmbeddingsBatch(req.Texts)
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

func main() {
	modelPath := flag.String("model", "", "Path to GGUF model file (required)")
	addr := flag.String("addr", ":8080", "Listen address")
	gpuLayers := flag.Int("gpu-layers", 0, "GPU layers to offload (-1 for all)")
	flag.Parse()

	if *modelPath == "" {
		log.Fatal("flag -model is required")
	}

	log.Printf("loading model: %s", *modelPath)
	model, err := llama.LoadModel(*modelPath, llama.WithGPULayers(*gpuLayers))
	if err != nil {
		log.Fatalf("failed to load model: %v", err)
	}
	defer model.Close()

	embedCtx, err = model.NewContext(llama.WithEmbeddings())
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
