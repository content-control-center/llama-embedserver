# llama-go-docker

Docker image for [llama-go](https://github.com/tcpipuk/llama-go) — Go bindings for running LLMs locally via llama.cpp.

## Build

```bash
docker build -t llama-go .
```

The build clones the repo with submodules, compiles llama.cpp from source, and produces a minimal `debian:bookworm-slim` image with just the compiled binary. This will take a few minutes.

## Usage

Mount a directory containing your GGUF model and pass the model path:

```bash
docker run --rm -v ./models:/models llama-go \
  -m /models/Qwen3-0.6B-Q8_0.gguf -p "Hello world" -n 50
```

### Download a test model

```bash
wget https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf
docker run --rm -v $PWD:/models llama-go \
  -m /models/Qwen3-0.6B-Q8_0.gguf -p "Hello world" -n 50
```

### Options

| Flag | Description |
|------|-------------|
| `-m` | Path to the GGUF model file (required) |
| `-p` | Prompt text |
| `-n` | Number of tokens to generate |

The default command runs: `simple -m /models/model.gguf -p "Hello world" -n 50`

## Image structure

| Stage | Base | Purpose |
|-------|------|---------|
| `builder` | `golang:latest` | Clones repo, builds `libbinding.a`, compiles Go binary |
| final | `debian:bookworm-slim` | Runtime — binary only, ~100MB |

Models are not baked into the image. Mount them as a volume to keep the image small and reusable across models.

--- 

## Tested models

- nomic-embed-text-v1.5.f32.gguf (embedding)
- Qwen3-0.6B-Q8_0.gguf
