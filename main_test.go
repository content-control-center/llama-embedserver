package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
)

// stubEmbedder is a test double for the Embedder interface.
type stubEmbedder struct {
	vec []float32
	err error
}

func (s *stubEmbedder) GetEmbeddings(_ string) ([]float32, error) {
	return s.vec, s.err
}

func (s *stubEmbedder) GetEmbeddingsBatch(texts []string) ([][]float32, error) {
	if s.err != nil {
		return nil, s.err
	}
	result := make([][]float32, len(texts))
	for i := range texts {
		result[i] = append([]float32{}, s.vec...)
	}
	return result, nil
}

// jsonBody marshals v into a buffer for use as an HTTP request body.
func jsonBody(t testing.TB, v any) *bytes.Buffer {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("marshal request body: %v", err)
	}
	return bytes.NewBuffer(b)
}

func newTestServer(e Embedder) *server {
	return &server{embedder: e}
}

// --- handleHealth ---

func TestHandleHealth(t *testing.T) {
	srv := newTestServer(nil)
	w := httptest.NewRecorder()
	srv.handleHealth(w, httptest.NewRequest(http.MethodGet, "/health", nil))

	if w.Code != http.StatusOK {
		t.Fatalf("want 200, got %d", w.Code)
	}
	if ct := w.Header().Get("Content-Type"); ct != "application/json" {
		t.Errorf("want Content-Type application/json, got %q", ct)
	}
	var resp map[string]string
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if resp["status"] != "ok" {
		t.Errorf("want status=ok, got %q", resp["status"])
	}
}

// --- handleEmbed ---

func TestHandleEmbed_WrongMethod(t *testing.T) {
	srv := newTestServer(&stubEmbedder{})
	w := httptest.NewRecorder()
	srv.handleEmbed(w, httptest.NewRequest(http.MethodGet, "/embed", nil))
	if w.Code != http.StatusMethodNotAllowed {
		t.Errorf("want 405, got %d", w.Code)
	}
}

func TestHandleEmbed_InvalidJSON(t *testing.T) {
	srv := newTestServer(&stubEmbedder{})
	w := httptest.NewRecorder()
	srv.handleEmbed(w, httptest.NewRequest(http.MethodPost, "/embed", bytes.NewBufferString("not-json")))
	if w.Code != http.StatusBadRequest {
		t.Errorf("want 400, got %d", w.Code)
	}
}

func TestHandleEmbed_EmptyText(t *testing.T) {
	srv := newTestServer(&stubEmbedder{})
	w := httptest.NewRecorder()
	srv.handleEmbed(w, httptest.NewRequest(http.MethodPost, "/embed", jsonBody(t, embedRequest{Text: ""})))
	if w.Code != http.StatusBadRequest {
		t.Errorf("want 400, got %d", w.Code)
	}
}

func TestHandleEmbed_EmbedderError(t *testing.T) {
	srv := newTestServer(&stubEmbedder{err: errors.New("model exploded")})
	w := httptest.NewRecorder()
	srv.handleEmbed(w, httptest.NewRequest(http.MethodPost, "/embed", jsonBody(t, embedRequest{Text: "hello"})))

	if w.Code != http.StatusInternalServerError {
		t.Errorf("want 500, got %d", w.Code)
	}
	var resp errorResponse
	json.NewDecoder(w.Body).Decode(&resp)
	if resp.Error != "model exploded" {
		t.Errorf("want error message propagated, got %q", resp.Error)
	}
}

func TestHandleEmbed_Success(t *testing.T) {
	vec := []float32{0.1, 0.2, 0.3, 0.4}
	srv := newTestServer(&stubEmbedder{vec: vec})
	w := httptest.NewRecorder()
	srv.handleEmbed(w, httptest.NewRequest(http.MethodPost, "/embed", jsonBody(t, embedRequest{Text: "hello world"})))

	if w.Code != http.StatusOK {
		t.Fatalf("want 200, got %d", w.Code)
	}
	var resp embedResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if resp.Dimensions != len(vec) {
		t.Errorf("want dimensions=%d, got %d", len(vec), resp.Dimensions)
	}
	if len(resp.Embedding) != len(vec) {
		t.Errorf("want %d floats in embedding, got %d", len(vec), len(resp.Embedding))
	}
	for i, v := range resp.Embedding {
		if v != vec[i] {
			t.Errorf("embedding[%d]: want %f, got %f", i, vec[i], v)
		}
	}
}

// --- handleBatch ---

func TestHandleBatch_WrongMethod(t *testing.T) {
	srv := newTestServer(&stubEmbedder{})
	w := httptest.NewRecorder()
	srv.handleBatch(w, httptest.NewRequest(http.MethodGet, "/embed/batch", nil))
	if w.Code != http.StatusMethodNotAllowed {
		t.Errorf("want 405, got %d", w.Code)
	}
}

func TestHandleBatch_InvalidJSON(t *testing.T) {
	srv := newTestServer(&stubEmbedder{})
	w := httptest.NewRecorder()
	srv.handleBatch(w, httptest.NewRequest(http.MethodPost, "/embed/batch", bytes.NewBufferString("{bad json}")))
	if w.Code != http.StatusBadRequest {
		t.Errorf("want 400, got %d", w.Code)
	}
}

func TestHandleBatch_EmptyTexts(t *testing.T) {
	srv := newTestServer(&stubEmbedder{})
	w := httptest.NewRecorder()
	srv.handleBatch(w, httptest.NewRequest(http.MethodPost, "/embed/batch", jsonBody(t, batchRequest{Texts: []string{}})))
	if w.Code != http.StatusBadRequest {
		t.Errorf("want 400, got %d", w.Code)
	}
}

func TestHandleBatch_EmbedderError(t *testing.T) {
	srv := newTestServer(&stubEmbedder{err: errors.New("batch failed")})
	w := httptest.NewRecorder()
	srv.handleBatch(w, httptest.NewRequest(http.MethodPost, "/embed/batch", jsonBody(t, batchRequest{Texts: []string{"a", "b"}})))
	if w.Code != http.StatusInternalServerError {
		t.Errorf("want 500, got %d", w.Code)
	}
}

func TestHandleBatch_Success(t *testing.T) {
	vec := []float32{0.5, 0.6}
	srv := newTestServer(&stubEmbedder{vec: vec})
	texts := []string{"first", "second", "third"}
	w := httptest.NewRecorder()
	srv.handleBatch(w, httptest.NewRequest(http.MethodPost, "/embed/batch", jsonBody(t, batchRequest{Texts: texts})))

	if w.Code != http.StatusOK {
		t.Fatalf("want 200, got %d", w.Code)
	}
	var resp batchResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if len(resp.Embeddings) != len(texts) {
		t.Errorf("want %d embeddings, got %d", len(texts), len(resp.Embeddings))
	}
	if resp.Dimensions != len(vec) {
		t.Errorf("want dimensions=%d, got %d", len(vec), resp.Dimensions)
	}
	for i, emb := range resp.Embeddings {
		if len(emb) != len(vec) {
			t.Errorf("embedding[%d]: want length %d, got %d", i, len(vec), len(emb))
		}
	}
}

// --- Concurrent safety (run with -race) ---

func TestHandleEmbed_Concurrent(t *testing.T) {
	vec := make([]float32, 64)
	srv := newTestServer(&stubEmbedder{vec: vec})

	var wg sync.WaitGroup
	for range 50 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			req := httptest.NewRequest(http.MethodPost, "/embed", jsonBody(t, embedRequest{Text: "concurrent test"}))
			srv.handleEmbed(httptest.NewRecorder(), req)
		}()
	}
	wg.Wait()
}

func TestHandleBatch_Concurrent(t *testing.T) {
	vec := make([]float32, 64)
	srv := newTestServer(&stubEmbedder{vec: vec})

	var wg sync.WaitGroup
	for range 50 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			req := httptest.NewRequest(http.MethodPost, "/embed/batch", jsonBody(t, batchRequest{Texts: []string{"a", "b", "c"}}))
			srv.handleBatch(httptest.NewRecorder(), req)
		}()
	}
	wg.Wait()
}
