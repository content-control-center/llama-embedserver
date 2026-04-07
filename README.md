# llama-go Embedding Server

A minimal REST service for generating text embeddings using [llama-go](https://github.com/tcpipuk/llama-go) (Go bindings for llama.cpp), packaged as a Docker image.

## Build

```bash
docker build -t llama-embedserver .
```

The build clones the llama-go repo with submodules, compiles llama.cpp from source (static linking), and produces a minimal `debian:bookworm-slim` image with the server binary.

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

## Image structure

| Stage | Base | Purpose |
|-------|------|---------|
| `model` | `debian:bookworm-slim` | Downloads the GGUF model (~329MB) |
| `builder` | `golang:bookworm` | Clones repo, builds `libbinding.a` (static), compiles server binary |
| final | `debian:bookworm-slim` | Runtime — binary + baked-in model |

The model and build are separate stages so Docker caches them independently — rebuilding after code changes won't re-download the model.
