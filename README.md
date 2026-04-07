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
| `-addr` | `:8080` | Listen address |
| `-gpu-layers` | `0` | GPU layers to offload (`-1` for all) |
| `-context-size` | `512` | KV cache window in tokens — lower = less memory |
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

llama.cpp uses glibc's ptmalloc2 allocator, which has two problems for long-running servers:

**Fragmentation** — ptmalloc2 manages memory via a contiguous `sbrk` heap. Freed blocks are returned to the OS only if the *top* of the heap is free. A single live allocation anywhere below traps all freed memory above it. Over a long run this causes RSS to grow monotonically even though allocations are balanced.

**Arena pooling** — freed blocks are kept in a free-list for reuse rather than returned to the OS, so RSS stays high even after `free()`.

Three mitigations are applied:

| Mechanism | What it does |
|-----------|-------------|
| `MALLOC_MMAP_THRESHOLD_=65536` | Allocations ≥ 64 KiB (llama.cpp batch buffers, ggml compute) use `mmap` instead of `sbrk`. `mmap` blocks are always returned to the OS on `free()` — fragmentation is impossible for these. |
| `MALLOC_TRIM_THRESHOLD_=131072` | After a `free()`, glibc trims contiguous free space at the top of the `sbrk` arena back to the OS if it exceeds 128 KiB. |
| `malloc_trim(0)` (called every `-gc-interval`) | Explicitly compacts and trims the `sbrk` arena on schedule, reclaiming whatever contiguous free space has accumulated at the top since the last trim. |

### KV cache

llama.cpp allocates a KV cache sized `n_parallel × context_size × bytes_per_token` upfront. The defaults keep this small:

- `WithParallel(1)` — single sequence slot (library default for embedding contexts is 8)
- `WithContext(512)` — 512-token window instead of the model's native max (often 8192+)
- `WithPrefixCaching(false)` — disables KV cache accumulation across requests (irrelevant for independent embedding inputs)
