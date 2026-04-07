package main

import (
	"context"
	"errors"
	"testing"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func newGRPCServer(e Embedder) *grpcServer {
	return &grpcServer{srv: &server{embedder: e}}
}

// --- Health ---

func TestGRPC_Health(t *testing.T) {
	g := newGRPCServer(nil)
	resp, err := g.Health(context.Background(), &HealthRequest{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Status != "ok" {
		t.Errorf("want status=ok, got %q", resp.Status)
	}
}

// --- Embed ---

func TestGRPC_Embed_EmptyText(t *testing.T) {
	g := newGRPCServer(&stubEmbedder{})
	_, err := g.Embed(context.Background(), &EmbedRequest{Text: ""})
	if code := status.Code(err); code != codes.InvalidArgument {
		t.Errorf("want InvalidArgument, got %v", code)
	}
}

func TestGRPC_Embed_EmbedderError(t *testing.T) {
	g := newGRPCServer(&stubEmbedder{err: errors.New("model exploded")})
	_, err := g.Embed(context.Background(), &EmbedRequest{Text: "hello"})
	if code := status.Code(err); code != codes.Internal {
		t.Errorf("want Internal, got %v", code)
	}
}

func TestGRPC_Embed_Success(t *testing.T) {
	vec := []float32{0.1, 0.2, 0.3}
	g := newGRPCServer(&stubEmbedder{vec: vec})
	resp, err := g.Embed(context.Background(), &EmbedRequest{Text: "hello world"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if int(resp.Dimensions) != len(vec) {
		t.Errorf("want dimensions=%d, got %d", len(vec), resp.Dimensions)
	}
	if len(resp.Embedding) != len(vec) {
		t.Errorf("want %d floats, got %d", len(vec), len(resp.Embedding))
	}
	for i, v := range resp.Embedding {
		if v != vec[i] {
			t.Errorf("embedding[%d]: want %f, got %f", i, vec[i], v)
		}
	}
}

// --- EmbedBatch ---

func TestGRPC_EmbedBatch_EmptyTexts(t *testing.T) {
	g := newGRPCServer(&stubEmbedder{})
	_, err := g.EmbedBatch(context.Background(), &EmbedBatchRequest{Texts: []string{}})
	if code := status.Code(err); code != codes.InvalidArgument {
		t.Errorf("want InvalidArgument, got %v", code)
	}
}

func TestGRPC_EmbedBatch_EmbedderError(t *testing.T) {
	g := newGRPCServer(&stubEmbedder{err: errors.New("batch failed")})
	_, err := g.EmbedBatch(context.Background(), &EmbedBatchRequest{Texts: []string{"a", "b"}})
	if code := status.Code(err); code != codes.Internal {
		t.Errorf("want Internal, got %v", code)
	}
}

func TestGRPC_EmbedBatch_Success(t *testing.T) {
	vec := []float32{0.4, 0.5}
	g := newGRPCServer(&stubEmbedder{vec: vec})
	texts := []string{"first", "second", "third"}
	resp, err := g.EmbedBatch(context.Background(), &EmbedBatchRequest{Texts: texts})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(resp.Embeddings) != len(texts) {
		t.Errorf("want %d embeddings, got %d", len(texts), len(resp.Embeddings))
	}
	if int(resp.Dimensions) != len(vec) {
		t.Errorf("want dimensions=%d, got %d", len(vec), resp.Dimensions)
	}
	for i, ev := range resp.Embeddings {
		if len(ev.Values) != len(vec) {
			t.Errorf("embedding[%d]: want %d values, got %d", i, len(vec), len(ev.Values))
		}
	}
}

// --- Concurrent safety (run with -race) ---

func TestGRPC_Embed_Concurrent(t *testing.T) {
	vec := make([]float32, 64)
	g := newGRPCServer(&stubEmbedder{vec: vec})

	done := make(chan struct{}, 50)
	for range 50 {
		go func() {
			g.Embed(context.Background(), &EmbedRequest{Text: "concurrent"}) //nolint:errcheck
			done <- struct{}{}
		}()
	}
	for range 50 {
		<-done
	}
}
