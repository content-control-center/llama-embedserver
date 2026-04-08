# llama-go Embedding Server

A minimal REST service for generating text embeddings using [llama-go](https://github.com/tcpipuk/llama-go) (Go bindings for llama.cpp), packaged as a Docker image.

---

## Build

```bash
docker build -t llama-embedserver .
```

The build clones the llama-go repo with submodules, compiles llama.cpp from source (static linking), and produces a minimal `debian:bookworm-slim` image with the server binary.

---

## Usage

The model (`embeddinggemma-300M Q8_0`, ~329MB) is baked into the image at build time. No volume mounts needed.

```bash
docker run --rm -p 8080:8080 llama-embedserver
```

### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `-model` | `/model.gguf` | Path to the GGUF model file |
| `-addr` | `:8080` | HTTP listen address |
| `-grpc-addr` | `:9090` | gRPC listen address |
| `-gpu-layers` | `0` | GPU layers to offload (`-1` for all) |
| `-context-size` | `512` | KV cache window in tokens — lower = less memory |
| `-threads` | `2` | llama.cpp compute threads per inference — requests are serialized so more than 2–4 rarely helps |
| `-mem-limit-mib` | `128` | Soft Go heap limit in MiB (`0` = unlimited) |
| `-gc-interval` | `30s` | How often freed memory is force-returned to the OS |

---

## API

### `GET /health`

```bash
curl http://localhost:8080/health
```
```json
{"status": "ok"}
```

### `POST /embed` — single embedding

```bash
curl -X POST http://localhost:8080/embed \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world"}'
```
```json
{
  "embedding": [0.023, -0.147, ...],
  "dimensions": 896
}
```

### `POST /embed/batch` — batch embeddings

```bash
curl -X POST http://localhost:8080/embed/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Hello world", "Goodbye world"]}'
```
```json
{
  "embeddings": [[0.023, -0.147, ...], [0.051, 0.093, ...]],
  "dimensions": 896
}
```

---

## gRPC API

The server also exposes a gRPC interface on port `9090` with the same three operations. The proto definition is in [`embedder.proto`](embedder.proto).

```bash
docker run --rm -p 8080:8080 -p 9090:9090 llama-embedserver
```

### Health

```bash
grpcurl -plaintext localhost:9090 embedder.EmbedService/Health
```
```json
{ "status": "ok" }
```

### Embed — single embedding

```bash
grpcurl -plaintext -d '{"text": "Hello world"}' \
  localhost:9090 embedder.EmbedService/Embed
```
```json
{
  "embedding": [0.023, -0.147, "..."],
  "dimensions": 896
}
```

### EmbedBatch — batch embeddings

```bash
grpcurl -plaintext -d '{"texts": ["Hello world", "Goodbye world"]}' \
  localhost:9090 embedder.EmbedService/EmbedBatch
```
```json
{
  "embeddings": [
    {"values": [0.023, -0.147, "..."]},
    {"values": [0.051,  0.093, "..."]}
  ],
  "dimensions": 896
}
```

---

## Genkit integration

[llama-genkit-embedder](https://github.com/alephbet-ai/llama-genkit-embedder) is a [Genkit](https://firebase.google.com/docs/genkit/go) embedder plugin that connects to this server over HTTP(s), making it a drop-in embedder for Genkit flows and retrievers.

### Install

```bash
go get github.com/alephbet-ai/llama-genkit-embedder
```

### Register the plugin

```go
import (
    "github.com/firebase/genkit/go/genkit"
    llamaembedder "github.com/alephbet-ai/llama-genkit-embedder"
)

func main() {
    ctx := context.Background()

    g, err := genkit.Init(ctx)
    if err != nil {
        log.Fatal(err)
    }

    embedder, err := llamaembedder.Init(g, &llamaembedder.Config{
        LlamaEmbedServerAddress: "http://localhost:8080",
    })
    if err != nil {
        log.Fatal(err)
    }

    // Use the embedder in a Genkit flow or retriever
    _ = embedder
}
```

### Use in a retriever

```go
resp, err := ai.Embed(ctx, embedder, &ai.EmbedRequest{
    Documents: []*ai.Document{
        ai.DocumentFromText("Hello world", nil),
    },
})
```

---

## Image structure

| Stage | Base | Purpose |
|-------|------|---------|
| `model` | `debian:bookworm-slim` | Downloads the GGUF model (~329MB) |
| `builder` | `golang:bookworm` | Clones repo, builds `libbinding.a` (static), compiles server binary |
| final | `debian:bookworm-slim` | Runtime — binary + baked-in model |

The model and build are separate stages so Docker caches them independently — rebuilding after code changes won't re-download the model.

---

## Memory management

The server runs in a CGO environment where memory is split across two heaps — Go's managed heap and the C heap used by llama.cpp. Each requires a different reclamation strategy.

### Go heap

- **`debug.SetMemoryLimit`** caps the Go heap at `-mem-limit-mib` (default 128 MiB). Without a limit, Go's GC targets 2× the live heap size, letting RSS balloon before triggering collection.
- **`debug.FreeOSMemory`** is called every `-gc-interval` (default 30s). Go's GC frees objects but only marks pages as `MADV_FREE` — the kernel may reclaim them eventually but they still count toward RSS. `FreeOSMemory` upgrades this to `MADV_DONTNEED`, forcing immediate OS reclaim.

### C heap (llama.cpp)

glibc's default allocator (ptmalloc2) has two fundamental problems for long-running servers:

**Fragmentation** — ptmalloc2 manages memory via a contiguous `sbrk` heap. Freed blocks can only be returned to the OS if the *top* of the heap is free. A single live allocation anywhere below traps all freed memory above it, causing RSS to grow monotonically over a long run even though allocations are balanced.

**Arena pooling** — freed blocks are kept in a free-list for reuse rather than returned to the OS, so RSS stays high even after `free()`.

Tuning glibc (`MALLOC_MMAP_THRESHOLD_`, `MALLOC_TRIM_THRESHOLD_`, `malloc_trim`) attacks symptoms but cannot fix the fragmentation problem — it is structural.

**The fix is jemalloc**, injected at runtime via `LD_PRELOAD` by the container entrypoint. jemalloc replaces ptmalloc2 for all C/C++ allocations (llama.cpp, ggml). Go manages its own heap directly via `mmap` and is unaffected.

| `MALLOC_CONF` option | What it does |
|----------------------|-------------|
| `background_thread:true` | Spawns a dedicated thread that returns dirty pages to the OS on a timer — no manual `malloc_trim` calls needed. |
| `dirty_decay_ms:1000` | Dirty pages (freed, `MADV_DONTNEED` pending) are released to the OS after 1 s. |
| `muzzy_decay_ms:1000` | Muzzy pages (freed, `MADV_FREE`, reclaimable by kernel) are fully released after 1 s. |
| `narenas:1` | Single arena prevents per-thread arena proliferation, which would multiply fragmented regions. |

`malloc_trim(0)` is still called in the Go periodic loop as a fallback — jemalloc provides a compatible implementation that triggers its own cleanup.

### KV cache

llama.cpp allocates a KV cache sized `n_parallel × context_size × bytes_per_token` upfront. The defaults keep this small:

- `WithParallel(1)` — single sequence slot (library default for embedding contexts is 8)
- `WithContext(512)` — 512-token window instead of the model's native max (often 8192+)
- `WithPrefixCaching(false)` — disables KV cache accumulation across requests (irrelevant for independent embedding inputs)
