//go:build integration

package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"

	llama "github.com/tcpipuk/llama-go"
)

// loadTestServer loads a real model and returns a wired server.
// Skips the test if TEST_MODEL is not set.
func loadTestServer(t *testing.T) *server {
	t.Helper()
	modelPath := os.Getenv("TEST_MODEL")
	if modelPath == "" {
		t.Skip("TEST_MODEL not set — skipping integration test")
	}

	model, err := llama.LoadModel(modelPath, llama.WithGPULayers(0))
	if err != nil {
		t.Fatalf("load model: %v", err)
	}
	t.Cleanup(func() { model.Close() })

	ctx, err := model.NewContext(
		llama.WithEmbeddings(),
		llama.WithPrefixCaching(false),
		llama.WithContext(512),
		llama.WithParallel(1),
	)
	if err != nil {
		t.Fatalf("new context: %v", err)
	}
	t.Cleanup(func() { ctx.Close() })

	return &server{embedder: ctx}
}

func TestIntegration_Health(t *testing.T) {
	// Health does not touch the model — just verify the full server wires up.
	srv := loadTestServer(t)
	w := httptest.NewRecorder()
	srv.handleHealth(w, httptest.NewRequest(http.MethodGet, "/health", nil))
	if w.Code != http.StatusOK {
		t.Fatalf("want 200, got %d", w.Code)
	}
}

func TestIntegration_Embed(t *testing.T) {
	srv := loadTestServer(t)
	w := httptest.NewRecorder()
	srv.handleEmbed(w, httptest.NewRequest(http.MethodPost, "/embed", jsonBody(t, embedRequest{Text: "Hello world"})))

	if w.Code != http.StatusOK {
		t.Fatalf("want 200, got %d: %s", w.Code, w.Body.String())
	}
	var resp embedResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if resp.Dimensions == 0 {
		t.Error("expected non-zero dimensions")
	}
	if len(resp.Embedding) != resp.Dimensions {
		t.Errorf("embedding length %d != dimensions %d", len(resp.Embedding), resp.Dimensions)
	}
}

func TestIntegration_EmbedBatch(t *testing.T) {
	srv := loadTestServer(t)
	texts := []string{"Hello world", "Goodbye world", "Testing embeddings"}
	w := httptest.NewRecorder()
	srv.handleBatch(w, httptest.NewRequest(http.MethodPost, "/embed/batch", jsonBody(t, batchRequest{Texts: texts})))

	if w.Code != http.StatusOK {
		t.Fatalf("want 200, got %d: %s", w.Code, w.Body.String())
	}
	var resp batchResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(resp.Embeddings) != len(texts) {
		t.Errorf("want %d embeddings, got %d", len(texts), len(resp.Embeddings))
	}
	if resp.Dimensions == 0 {
		t.Error("expected non-zero dimensions")
	}
	for i, emb := range resp.Embeddings {
		if len(emb) != resp.Dimensions {
			t.Errorf("embedding[%d]: length %d != dimensions %d", i, len(emb), resp.Dimensions)
		}
	}
}

// TestIntegration_EmbedDifferentTexts verifies that distinct inputs produce distinct vectors.
func TestIntegration_EmbedDifferentTexts(t *testing.T) {
	srv := loadTestServer(t)

	embed := func(text string) []float32 {
		w := httptest.NewRecorder()
		srv.handleEmbed(w, httptest.NewRequest(http.MethodPost, "/embed", jsonBody(t, embedRequest{Text: text})))
		if w.Code != http.StatusOK {
			t.Fatalf("want 200, got %d", w.Code)
		}
		var resp embedResponse
		json.NewDecoder(w.Body).Decode(&resp)
		return resp.Embedding
	}

	e1 := embed("Hello world")
	e2 := embed("Goodbye world")

	if len(e1) != len(e2) {
		t.Fatalf("embedding length mismatch: %d vs %d", len(e1), len(e2))
	}
	identical := true
	for i := range e1 {
		if e1[i] != e2[i] {
			identical = false
			break
		}
	}
	if identical {
		t.Error("different texts produced identical embeddings")
	}
}

// TestIntegration_EmbedConsistency verifies the same input always yields the same vector.
func TestIntegration_EmbedConsistency(t *testing.T) {
	srv := loadTestServer(t)

	embed := func() []float32 {
		w := httptest.NewRecorder()
		srv.handleEmbed(w, httptest.NewRequest(http.MethodPost, "/embed", jsonBody(t, embedRequest{Text: "consistency check"})))
		var resp embedResponse
		json.NewDecoder(w.Body).Decode(&resp)
		return resp.Embedding
	}

	e1 := embed()
	e2 := embed()

	if len(e1) != len(e2) {
		t.Fatalf("embedding length changed between calls: %d vs %d", len(e1), len(e2))
	}
	for i := range e1 {
		if e1[i] != e2[i] {
			t.Errorf("embedding not deterministic at index %d: %f vs %f", i, e1[i], e2[i])
			break
		}
	}
}
