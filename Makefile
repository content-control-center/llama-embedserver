IMAGE      ?= llama-embedserver
PORT       ?= 8080
MODELS_DIR ?= $(PWD)/models
TEST_MODEL ?= /models/model.gguf

# go test flags passed inside the builder container
GOTEST_FLAGS ?= -v -count=1

# Environment required for CGO to find libbinding.a and headers
GOTEST_ENV = LIBRARY_PATH=/workspace C_INCLUDE_PATH=/workspace CGO_ENABLED=1

.PHONY: build run test test-race test-integration push clean help

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*##"}; {printf "  %-20s %s\n", $$1, $$2}'

build: ## Build the production Docker image
	docker build -t $(IMAGE) .

run: build ## Run the embedding server on PORT (default 8080)
	docker run --rm -p $(PORT):8080 $(IMAGE)

test: _builder ## Run unit tests inside the builder container
	docker run --rm $(IMAGE)-builder \
		sh -c "$(GOTEST_ENV) go test $(GOTEST_FLAGS) ./cmd/embedserver/"

test-race: _builder ## Run unit tests with the race detector
	docker run --rm $(IMAGE)-builder \
		sh -c "$(GOTEST_ENV) go test -race $(GOTEST_FLAGS) ./cmd/embedserver/"

test-integration: _builder ## Run integration tests (set MODELS_DIR and TEST_MODEL)
	docker run --rm \
		-v $(MODELS_DIR):/models \
		-e TEST_MODEL=$(TEST_MODEL) \
		$(IMAGE)-builder \
		sh -c "$(GOTEST_ENV) go test -tags integration $(GOTEST_FLAGS) ./cmd/embedserver/"

push: build ## Push the production image to Docker Hub
	docker push $(IMAGE)

clean: ## Remove built Docker images
	docker rmi $(IMAGE) $(IMAGE)-builder 2>/dev/null || true

# Build only the builder stage (shared by all test targets, cached by Docker)
_builder:
	docker build --target builder -t $(IMAGE)-builder .

.DEFAULT_GOAL := help
