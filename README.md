# Optigon

**The many-faceted optimizer.** Optigon packages several interchangeable
implementations of an operation (sorting, and more to come) behind one
domain-level interface, and learns — per workload — which implementation is
fastest, using **regret-scored adaptive dispatch**. One Rust core, shipped as
native addons to **TypeScript (Node + Bun)** and **Python**.

The name reads two ways, both apt: *opti- + -gon* (an optimization **polygon** —
one core presenting the optimal face per workload) and *opti- + -agon* (an
optimal **contest** — implementations compete, the winner ships).

> Optigon is the productization of the research in `ml-prototyping`, whose
> writeup (`docs/adaptive-dispatch.md`) established the thesis: a cheap learned
> model beats any fixed choice by ~12–20× on regret, and is economically viable
> in real time *iff* the feature cost is near-free. Optigon puts that model on a
> Rust core (real `candle`, the port target of the `@voidloop/ml-core` mirror)
> and exposes it as installable libraries.

## How it works

```
        TS (Node/Bun) ──napi──┐            ┌── Python ──pyo3──┐
                              ▼            ▼
                   ┌────────────────────────────────────┐
                   │  optigon-core (Rust, candle)         │
                   │   Domain trait → packaged impls      │  ← the sorts, etc.
                   │   cheap feature extraction           │
                   │   Chooser: train + select + run      │  ← regret-scored MLP
                   │   Config: pin / forbid / bias         │  ← consumer steering
                   │   Recorder: capture (features,cost)  │  ← feeds training
                   └────────────────────────────────────┘
```

Everything above the `Domain` trait is generic, so a new domain is *one trait
impl* and it inherits training, selection, evaluation, steering, and both
bindings for free. The bindings are thin 1:1 wrappers — no logic of their own.

### Two ways to train

1. **Test-driven (Mode 1).** Point a `Recorder` at your workloads as your tests
   exercise the domain interface; it measures every implementation and records
   the cost row. Train from those rows. **The more diverse your tests, the better
   the chooser.** (This is what the demos do.)
2. **Production A/B (Mode 2).** Capture inputs + measured costs in production by
   switching implementations per call, then retrain offline. Same row shape; the
   online path is not built in this first slice.

## Quickstart

```bash
make install        # bun install (JS workspace deps)
make core-test      # cargo test -p optigon-core  (the domain + model + chooser)
make node-demo      # build the .node addon and run the end-to-end Node demo
make py-build       # build + install the Python extension (needs maturin)
make py-demo        # run the end-to-end Python demo
```

Both demos train a sort chooser by "running tests" and show it beating the best
single fixed sort by ~10–35× on regret, then persist and reload the model.

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
  optigon-core/            language-agnostic core (domain, sort, candle model, chooser)
  optigon-node/            napi-rs addon → one .node for Node + Bun
  optigon-python/          PyO3 extension → abi3 wheel (one per platform)
examples/
  node-demo/  py-demo/     end-to-end Mode-1 demos
```

Structure modeled on `~/Projects/ForgeDB`'s native-binding stack. Distribution
(cargo-dist for npm prebuilds, a maturin wheel matrix) and the remaining five
domains (dictionary, joins, string search, cache, compression) follow this same
`Domain`-trait pattern.

## Status

Walking skeleton: the **sort** domain end-to-end — Rust core (candle training +
inference) + napi + pyo3 + Mode-1 demos, all green. License: MIT OR Apache-2.0.
