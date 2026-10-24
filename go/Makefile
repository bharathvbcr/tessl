# The Go binding for EmbeddingGemma 2: build the engine archive, then check the package.
#
#   make -C go unit    no GPU: engine tests, gofmt, go vet, the Go unit tests
#   make -C go device  the fixtures and the device tests (Metal; the tiny checkpoint)
#   make -C go test    everything: unit, then device
#   make -C go real    the real-checkpoint tests (set EMBEDGEMMA2_SNAPSHOT,
#                      EMBEDGEMMA2_REF_DIR and EMBEDGEMMA2_MM_REF_DIR)
#   make -C go bench   the overhead benchmark, rewriting bench/results/embedgemma2_go_m5pro.json
#   make -C go serve        tessl-embed, the Ollama-protocol server (set EMBEDGEMMA2_SNAPSHOT)
#   make -C go serve-real   its real-checkpoint tests against Hugging Face (EMBEDGEMMA2_SNAPSHOT)
#   make -C go serve-bench  tessl-embed against Ollama, rewriting bench/results/tessl_embed_m5pro.json
#
# The Go package links libgusset.a, which is gusset and tessl in one Rust staticlib
# (engine/). Go's build cache keys a cgo package on the text of its flags and not on
# the bytes of the archive they name, so every Go command here passes the archive's
# hash in CGO_CFLAGS: a rebuilt archive is a new key, and a stale test binary cannot
# pass for the new engine.

SHELL   := /bin/bash -o pipefail
GO      ?= go
ENGINE  := engine
ARCHIVE := $(ENGINE)/target/release/libgusset.a
REPO    := ..

# Evaluated by the shell at run time, after the archive is built.
CGO     := CGO_CFLAGS="-DTESSL_ENGINE_SHA=$$(shasum -a 256 $(ARCHIVE) | cut -c1-16)"

.PHONY: engine engine-test fmt-check vet unit fixtures device test real bench serve serve-build serve-real serve-bench need-snapshot clean

engine:
	cargo build --release --locked --manifest-path $(ENGINE)/Cargo.toml --target-dir $(ENGINE)/target

engine-test:
	cargo test --release --locked --manifest-path $(ENGINE)/Cargo.toml --target-dir $(ENGINE)/target

fmt-check:
	@bad="$$(gofmt -l .)"; if [ -n "$$bad" ]; then echo "gofmt would change:" >&2; echo "$$bad" >&2; exit 1; fi
	cargo fmt --manifest-path $(ENGINE)/Cargo.toml --check

vet: engine
	$(CGO) $(GO) vet ./...

# No device and no checkpoint. A skipped test is not hidden: -v prints every SKIP.
unit: engine engine-test fmt-check vet
	$(CGO) $(GO) test -count=1 -run 'TestUnit' ./...

# tessl's own C ABI, called from Rust, writes the tiny checkpoint, the raw inputs and
# the rows it answers; the Go device tests compare the Go path with them bit for bit.
fixtures:
	cargo test --release --locked --manifest-path $(REPO)/Cargo.toml --test go_fixtures

# TESSL_GO_REQUIRE_DEVICE turns missing fixtures from a skip into a failure.
device: engine fixtures
	TESSL_GO_REQUIRE_DEVICE=1 $(CGO) $(GO) test -count=1 ./...

test: unit device

real: engine
	@test -n "$$EMBEDGEMMA2_SNAPSHOT" && test -n "$$EMBEDGEMMA2_REF_DIR" && test -n "$$EMBEDGEMMA2_MM_REF_DIR" || { \
	  echo "set EMBEDGEMMA2_SNAPSHOT, EMBEDGEMMA2_REF_DIR and EMBEDGEMMA2_MM_REF_DIR" >&2; exit 2; }
	$(CGO) $(GO) test -count=1 -v -run 'TestReal' ./embedgemma2/

# Go call overhead against the same work called from Rust directly (min of N), written
# to the repository's bench/results; `cargo run --release --bin plot_results` draws it.
ROUNDS   ?= 9
ITERS    ?= 150
# The benchmark waits (up to WAIT) for the 1-minute load average to fall to MAX_LOAD,
# and writes nothing if it is above it at the end: a timing from a busy machine
# measures the machine. MAX_LOAD=0 turns the check off.
MAX_LOAD ?= 12
WAIT     ?= 20m

bench: engine fixtures
	cargo build --release --locked --manifest-path $(ENGINE)/Cargo.toml --target-dir $(ENGINE)/target --bin bench_direct
	$(CGO) $(GO) run ./cmd/bench_overhead -rounds $(ROUNDS) -iters $(ITERS) -max-load $(MAX_LOAD) -wait $(WAIT) \
	  -direct $(ENGINE)/target/release/bench_direct -out $(REPO)/bench/results/embedgemma2_go_m5pro.json

# tessl-embed: EmbeddingGemma 2 behind Ollama's /api/embed on 127.0.0.1 (cmd/tessl-embed).
SERVER    := $(ENGINE)/target/release/tessl-embed
PORT      ?= 11435
SERVE_PID := $(ENGINE)/target/tessl-embed.pid

need-snapshot:
	@test -n "$$EMBEDGEMMA2_SNAPSHOT" || { echo "set EMBEDGEMMA2_SNAPSHOT to a google/embeddinggemma-2 snapshot directory" >&2; exit 2; }

# Always rebuilt: the archive's hash in CGO_CFLAGS decides whether Go relinks.
serve-build: engine
	$(CGO) $(GO) build -o $(SERVER) ./cmd/tessl-embed

serve: need-snapshot serve-build
	$(SERVER) -snapshot "$$EMBEDGEMMA2_SNAPSHOT" -port $(PORT)

# TESSL_EMBED_REQUIRE_REAL turns a missing snapshot from a skip into a failure.
serve-real: need-snapshot engine
	TESSL_EMBED_REQUIRE_REAL=1 $(CGO) $(GO) test -count=1 -v -run 'TestReal' ./cmd/tessl-embed/

# Starts the server, waits for its pidfile (written once it listens and has
# embedded a warm-up text), runs the three arms against Ollama on 11434
# (nomic-embed-text and embeddinggemma-2:270m pulled), and stops the server by
# that pid. ROUNDS, MAX_LOAD and WAIT as for bench.
serve-bench: need-snapshot serve-build
	@rm -f $(SERVE_PID); $(SERVER) -snapshot "$$EMBEDGEMMA2_SNAPSHOT" -port $(PORT) -pidfile $(SERVE_PID) & \
	  trap 'test -f $(SERVE_PID) && kill $$(cat $(SERVE_PID))' EXIT; \
	  for i in $$(seq 1 240); do test -f $(SERVE_PID) && break; sleep 0.5; done; \
	  test -f $(SERVE_PID) || { echo "tessl-embed did not start" >&2; exit 1; }; \
	  $(GO) run ./cmd/embed_bench -tessl http://127.0.0.1:$(PORT) -rounds $(ROUNDS) -max-load $(MAX_LOAD) -wait $(WAIT) \
	    -out $(REPO)/bench/results/tessl_embed_m5pro.json

clean:
	cargo clean --manifest-path $(ENGINE)/Cargo.toml --target-dir $(ENGINE)/target
