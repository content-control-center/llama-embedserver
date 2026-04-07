# Stage 1: Download the model
FROM debian:bookworm-slim AS model

RUN apt-get update && apt-get install -y wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q -O /model.gguf \
    https://huggingface.co/Bhuvneesh/embeddinggemma-300m-Q8_0-GGUF/resolve/main/embeddinggemma-300m-q8_0.gguf

# Stage 2: Build the static library and Go binary
FROM golang:bookworm AS builder

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN git clone --recurse-submodules https://github.com/tcpipuk/llama-go .

RUN make libbinding.a CMAKE_ARGS="-DBUILD_SHARED_LIBS=OFF"

# Copy the embedding server source into the cloned module so it resolves imports locally
COPY main.go ./cmd/embedserver/main.go

RUN LIBRARY_PATH=/workspace \
    C_INCLUDE_PATH=/workspace \
    CGO_ENABLED=1 \
    go build -o /usr/local/bin/embedserver ./cmd/embedserver

# Stage 3: Minimal runtime image
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    libstdc++6 \
    libgomp1 \
    libcurl4 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/bin/embedserver /usr/local/bin/embedserver
COPY --from=model /model.gguf /model.gguf

# Force C allocations ≥ 64 KiB to use mmap instead of sbrk. mmap blocks are
# always returned to the OS on free(), bypassing ptmalloc2's arena entirely —
# no heap fragmentation possible for llama.cpp's larger batch/buffer allocs.
ENV MALLOC_MMAP_THRESHOLD_=65536
# Trim the sbrk arena after freeing small blocks. Works in tandem with the
# malloc_trim(0) call in the Go periodic loop and with MALLOC_MMAP_THRESHOLD_
# (which handles large blocks via mmap instead).
ENV MALLOC_TRIM_THRESHOLD_=131072

EXPOSE 8080

ENTRYPOINT ["embedserver"]
CMD ["-model", "/model.gguf"]
