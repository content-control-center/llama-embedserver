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
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN git clone --recurse-submodules https://github.com/tcpipuk/llama-go .

RUN make libbinding.a CMAKE_ARGS="-DBUILD_SHARED_LIBS=OFF"

# Install protoc Go plugins — cached until plugin versions change.
RUN go install google.golang.org/protobuf/cmd/protoc-gen-go@latest \
 && go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest

# Generate gRPC code — invalidated only when the proto file changes.
COPY embedder.proto ./cmd/embedserver/embedder.proto
RUN PATH="$PATH:$(go env GOPATH)/bin" \
    protoc \
      --proto_path=cmd/embedserver \
      --go_out=cmd/embedserver \
      --go_opt=paths=source_relative \
      --go-grpc_out=cmd/embedserver \
      --go-grpc_opt=paths=source_relative \
      embedder.proto

# Copy application source — invalidated when source changes.
COPY main.go grpc_server.go ./cmd/embedserver/
COPY main_test.go grpc_server_test.go main_integration_test.go ./cmd/embedserver/

# Fetch gRPC runtime dependencies into the module, then build.
RUN GOFLAGS="-mod=mod" go get \
    google.golang.org/grpc \
    google.golang.org/protobuf

RUN LIBRARY_PATH=/workspace \
    C_INCLUDE_PATH=/workspace \
    CGO_ENABLED=1 \
    GOFLAGS="-mod=mod" \
    go build -o /usr/local/bin/embedserver ./cmd/embedserver

# Stage 3: Minimal runtime image
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    libstdc++6 \
    libgomp1 \
    libcurl4 \
    ca-certificates \
    libjemalloc2 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/bin/embedserver /usr/local/bin/embedserver
COPY --from=model /model.gguf /model.gguf
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# jemalloc tuning (active when entrypoint.sh injects it via LD_PRELOAD):
#   background_thread - dedicated thread returns dirty pages to OS on a timer
#   dirty_decay_ms    - pages held dirty (MADV_DONTNEED pending) for 1 s then released
#   muzzy_decay_ms    - pages held muzzy (MADV_FREE, reclaimable) for 1 s then released
#   narenas:1         - single arena avoids per-thread arena fragmentation
ENV MALLOC_CONF="background_thread:true,dirty_decay_ms:1000,muzzy_decay_ms:1000,narenas:1"

EXPOSE 8080
EXPOSE 9090

ENTRYPOINT ["/entrypoint.sh"]
CMD ["-model", "/model.gguf"]
