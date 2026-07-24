# Optigon — root entry points. Everything runs from the repo root; no `cd`
# required of a caller. Rust is the workspace root; Bun/Turbo sit on top.

.PHONY: help install build test fmt lint core-test node-build node-demo node-demo-dict node-demo-mode2 py-build py-demo py-demo-dict py-demo-mode2 clean

help:
	@echo "Optigon targets:"
	@echo "  make install     install JS workspace deps (bun)"
	@echo "  make build       build everything (cargo + napi addon)"
	@echo "  make test        cargo test across the workspace"
	@echo "  make core-test   test just optigon-core"
	@echo "  make node-build  build the Node/Bun native addon (release)"
	@echo "  make node-demo   run the end-to-end Node demo (sort, Mode 1)"
	@echo "  make node-demo-dict  run the Node dict demo (Mode 1)"
	@echo "  make node-demo-mode2 run the Node production A/B demo (Mode 2)"
	@echo "  make py-build     build + install the Python extension (maturin)"
	@echo "  make py-demo     run the end-to-end Python demo (sort, Mode 1)"
	@echo "  make py-demo-dict    run the Python dict demo (Mode 1)"
	@echo "  make py-demo-mode2   run the Python production A/B demo (Mode 2)"
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

node-demo-dict: node-build-debug
	cd examples/node-demo && bun run dict-demo.ts

node-demo-mode2: node-build-debug
	cd examples/node-demo && bun run mode2-demo.ts

py-build:
	cd crates/optigon-python && maturin develop --release

py-demo:
	cd examples/py-demo && python3 demo.py

py-demo-dict:
	cd examples/py-demo && python3 dict_demo.py

py-demo-mode2:
	cd examples/py-demo && python3 mode2_demo.py

fmt:
	cargo fmt
	bunx @biomejs/biome format --write .

lint:
	cargo clippy --workspace -- -D warnings
	bunx @biomejs/biome check .

clean:
	cargo clean
	rm -f crates/optigon-node/*.node crates/optigon-node/index.js crates/optigon-node/index.d.ts
