# llama-go Embedding Server

A minimal REST service for generating text embeddings using [llama-go](https://github.com/tcpipuk/llama-go) (Go bindings for llama.cpp), packaged as a Docker image.

## Build

```bash
docker build -t llama-embedserver .
```

The build clones the llama-go repo with submodules, compiles llama.cpp from source (static linking), and produces a minimal `debian:bookworm-slim` image with the server binary.

## Usage

Mount a directory containing your GGUF embedding model and run:

```bash
docker run --rm -p 8080:8080 \
  -v /path/to/models:/models \
  llama-embedserver \
  -model /models/model.gguf
```

### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `-model` | *(required)* | Path to the GGUF model file |
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

## Tested models

- `nomic-embed-text-v1.5.f32.gguf`
- `Qwen3-0.6B-Q8_0.gguf`

## Image structure

| Stage | Base | Purpose |
|-------|------|---------|
| `builder` | `golang:bookworm` | Clones repo, builds `libbinding.a` (static), compiles server binary |
| final | `debian:bookworm-slim` | Runtime — binary only, ~100MB |

Models are not baked into the image. Mount them as a volume to keep the image small and reusable across models.
