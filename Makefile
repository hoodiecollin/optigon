# Optigon — root entry points. Everything runs from the repo root; no `cd`
# required of a caller. Rust is the workspace root; Bun/Turbo sit on top.

.PHONY: help install build test fmt lint core-test node-build node-demo py-build py-demo clean

help:
	@echo "Optigon targets:"
	@echo "  make install     install JS workspace deps (bun)"
	@echo "  make build       build everything (cargo + napi addon)"
	@echo "  make test        cargo test across the workspace"
	@echo "  make core-test   test just optigon-core"
	@echo "  make node-build  build the Node/Bun native addon (release)"
	@echo "  make node-demo   run the end-to-end Node demo (Mode 1)"
	@echo "  make py-build     build + install the Python extension (maturin)"
	@echo "  make py-demo     run the end-to-end Python demo (Mode 1)"
	@echo "  make fmt         format Rust + JS"
	@echo "  make lint        biome + clippy"

install:
	bun install

build:
	cargo build --workspace
	$(MAKE) node-build

test:
	cargo test --workspace

core-test:
	cargo test -p optigon-core

node-build:
	cd crates/optigon-node && bunx @napi-rs/cli@2 build --platform --release

node-build-debug:
	cd crates/optigon-node && bunx @napi-rs/cli@2 build --platform

node-demo: node-build-debug
	cd examples/node-demo && bun run demo.ts

py-build:
	cd crates/optigon-python && maturin develop --release

py-demo:
	cd examples/py-demo && python3 demo.py

fmt:
	cargo fmt
	bunx @biomejs/biome format --write .

lint:
	cargo clippy --workspace -- -D warnings
	bunx @biomejs/biome check .

clean:
	cargo clean
	rm -f crates/optigon-node/*.node crates/optigon-node/index.js crates/optigon-node/index.d.ts
