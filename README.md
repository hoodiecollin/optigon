<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="brand/optigon-horizontal-dark.svg">
    <img alt="Optigon — the many-faceted optimizer" src="brand/optigon-horizontal-light.svg" width="400">
  </picture>
</p>

<p align="center">
  <img alt="License: MIT OR Apache-2.0" src="https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue">
  <img alt="Status: experimental" src="https://img.shields.io/badge/status-experimental-orange">
</p>

**Package several interchangeable implementations of an operation (sorting,
dictionary lookup, and more to come) behind one domain-level interface, and learn
— per workload — which implementation is fastest, using regret-scored adaptive
dispatch.** One Rust core, shipped as native addons to **TypeScript (Node + Bun)**
and **Python**.

> **Status: experimental.** Optigon is an early, single-author project under
> active development. Two domains (`sort`, `dict`) and both training modes work
> end-to-end, but the API is unstable, there's no published release yet, and
> distribution (npm prebuilds, a wheel matrix) is still to come. Running from
> source is the supported path today. No stability or support guarantees.

The name reads two ways, both apt: *opti- + -gon* (an optimization **polygon** —
one core presenting the optimal face per workload) and *opti- + -agon* (an
optimal **contest** — implementations compete, the winner ships).

> Optigon is the productization of the research in `ml-prototyping`, whose writeup
> (`docs/adaptive-dispatch.md`, in that repo) established the thesis: a cheap
> learned model beats any fixed choice by ~12–20× on regret, and is economically
> viable in real time *iff* the feature cost is near-free. Optigon puts that model
> on a Rust core (real `candle`, the port target of the `@voidloop/ml-core`
> mirror) and exposes it as installable libraries.

## How it works

```
        TS (Node/Bun) ──napi──┐            ┌── Python ──pyo3──┐
                              ▼            ▼
                   ┌────────────────────────────────────┐
                   │  optigon-core (Rust, candle)         │
                   │   Domain trait → packaged impls      │  ← sort, dict, …
                   │   cheap feature extraction           │
                   │   Chooser: train + select + run      │  ← regret-scored MLP
                   │   Config: pin / forbid / bias         │  ← consumer steering
                   │   Recorder: capture (features,cost)  │  ← Mode-1 training data
                   │   OnlineAb: A/B switch + log + retrain│  ← Mode-2 (production)
                   └────────────────────────────────────┘
```

Everything above the `Domain` trait is generic, so a new domain is *one trait
impl* and it inherits training, selection, evaluation, steering, online A/B
capture, and both bindings for free. The bindings are thin 1:1 wrappers — no
logic of their own. Impls may be **inapplicable** to some workloads (e.g. the
`dict` domain's direct-address table needs a bounded key range); such impls are
masked out of the loss, the regret argmin, and selection automatically.

### Two ways to train

1. **Test-driven (Mode 1).** Point a `Recorder` at your workloads as your tests
   exercise the domain interface; it measures every implementation and records
   the cost row. Train from those rows. **The more diverse your tests, the better
   the chooser.** (This is what the demos do.)
2. **Production A/B (Mode 2).** `OnlineAb` serves each production call with one
   implementation (epsilon-greedy: exploit the model, occasionally explore),
   logging every measured outcome as a single-observation row — the *same* shape
   Mode 1 produces, just one column filled. Export the log to JSONL, retrain a
   fresh chooser offline, and redeploy it warm. Bandit-style partial feedback is
   enough to recover a chooser that beats the best fixed impl.

## What this is (and isn't)

It is a **learned dispatch layer**: given a workload, it predicts which of several
correct-but-differently-performing implementations to run, and runs it. Every
packaged impl produces the *same* result — the chooser only ever trades on speed,
never on correctness, so a mispredicted choice is slower, never wrong.

It is **not** a general autotuner or a JIT. It won't invent implementations, tune
their parameters, or rewrite your code; you supply the impls behind a `Domain`
trait and Optigon learns to pick among them. It's aimed at operations with (a)
several reasonable implementations whose winner depends on the input, and (b) a
cheap-to-extract feature that predicts that winner. Where those don't hold, a
fixed choice is fine and Optigon buys you nothing.

## Quickstart

```bash
make install           # bun install (JS workspace deps)
make core-test         # cargo test -p optigon-core  (domain + model + chooser)
make node-demo         # sort chooser, Mode 1 (build the .node addon + run)
make node-demo-dict    # dict chooser, Mode 1 (applicability masking)
make node-demo-mode2   # sort chooser, Mode 2 (production A/B capture loop)
make py-build          # build + install the Python extension (needs maturin)
make py-demo           # sort chooser, Mode 1
make py-demo-dict      # dict chooser, Mode 1
make py-demo-mode2     # sort chooser, Mode 2
```

The Mode-1 demos train a chooser by "running tests" and show it beating the best
single fixed impl on regret, then persist and reload the model. The Mode-2 demos
simulate production traffic, capture a bandit-style log, retrain offline from it,
and redeploy the recovered chooser.

### TypeScript / Bun

```ts
import { SortChooser } from "optigon-node"

const chooser = new SortChooser()
for (const keys of myWorkloads) chooser.observe(keys) // Mode-1 capture
chooser.train()
const sorted = chooser.sort(keys) // picks the fastest sort, runs it
```

### Python

```python
from optigon import SortChooser

chooser = SortChooser()
for keys in my_workloads:
    chooser.observe(keys)         # Mode-1 capture
chooser.train()
sorted_keys = chooser.sort(keys)  # picks the fastest sort, runs it
```

## Layout

```
Cargo.toml                 Rust workspace root
package.json / turbo.json  Bun + Turbo layer (scripting, caching) on top
crates/
  optigon-core/            language-agnostic core (domains, candle model, chooser,
                           Mode-1 Recorder + Mode-2 OnlineAb)
  optigon-node/            napi-rs addon → one .node for Node + Bun
  optigon-python/          PyO3 extension → abi3 wheel (one per platform)
examples/
  node-demo/  py-demo/     end-to-end demos (Mode-1 sort + dict, Mode-2 sort)
```

Structure modeled on `~/Projects/ForgeDB`'s native-binding stack. Distribution
(cargo-dist for npm prebuilds, a maturin wheel matrix) and the remaining domains
(joins, string search, cache, compression) follow this same `Domain`-trait
pattern.

## Docs

- [docs/architecture.md](./docs/architecture.md) — the `Domain` trait, the
  chooser (features → regret-scored model → select + run), and both training
  modes (Mode-1 `Recorder`, Mode-2 `OnlineAb`).

## Status

Two domains end-to-end — **sort** and **dict** (the latter exercising
applicability masking) — plus **both training modes**: Mode-1 test-driven capture
and Mode-2 production A/B (online capture → JSONL log → offline retrain →
redeploy). Rust core (candle training + inference) + napi + pyo3 + demos, all
green (11 core tests). The API is unstable and there's no published release yet;
see the experimental note at the top.

## Contributing

Contributions are welcome, though the project is early and single-author, so the
accepted-domain surface is still moving. The one gate to run before opening a PR
(the same four checks CI runs):

```bash
cargo fmt --all -- --check         # formatting
cargo clippy --workspace -- -D warnings
cargo test --workspace             # 11 core tests
bunx @biomejs/biome check .        # JS/TS lint + format
```

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](./LICENSE-APACHE))
- MIT license ([LICENSE-MIT](./LICENSE-MIT))

at your option. Unless you explicitly state otherwise, any contribution
intentionally submitted for inclusion in this project by you, as defined in the
Apache-2.0 license, shall be dual licensed as above, without any additional terms
or conditions.
