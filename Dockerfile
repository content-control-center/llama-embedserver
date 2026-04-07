# Stage 1: Build the static library and Go binary
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

# Stage 2: Minimal runtime image
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    libstdc++6 \
    libgomp1 \
    libcurl4 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/bin/embedserver /usr/local/bin/embedserver

# Mount your model directory here: -v /path/to/models:/models
VOLUME ["/models"]

EXPOSE 8080

ENTRYPOINT ["embedserver"]
CMD ["-model", "/models/model.gguf"]
