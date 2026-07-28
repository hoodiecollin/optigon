# Architecture

Optigon is one generic core plus per-domain trait impls. Everything that isn't a
`Domain` impl — the chooser, the model, capture, steering, both bindings — is
written once, generically, and every domain inherits it. This doc walks the core
in the order data flows through it.

```
                TS (Node/Bun) ──napi──┐          ┌── Python ──pyo3──┐
                                      ▼          ▼
                     ┌──────────────────────────────────────────┐
                     │  optigon-core (Rust, candle)              │
                     │    Domain trait  → packaged impls         │  sort, dict, …
                     │    features()    → cheap workload vector   │
                     │    MlpModel      → per-impl log-cost       │  regret-scored
                     │    Chooser       → train / select / run    │
                     │    Config        → pin / forbid / bias      │  steering
                     │    Recorder      → Mode-1 capture           │
                     │    OnlineAb      → Mode-2 A/B + retrain      │
                     └──────────────────────────────────────────┘
```

## The `Domain` trait (`domain.rs`)

A `Domain` is a family of interchangeable implementations of one operation, plus
the cheap features that predict which is fastest. The contract:

- `impl_names()` — implementation ids in a **fixed order**. Cost columns, the
  model's output vector, and selection all index by this order, so it must never
  be reordered.
- `features(input)` — a cheap, fixed-length (`FEATURE_DIM`) workload descriptor.
  Must be far cheaper than running an impl; that near-free cost is the economic
  precondition the research thesis rests on.
- `run(impl_idx, input)` — execute one impl.
- `cost(impl_idx, input)` — ground-truth cost (lower = better); the training
  target and the A/B measurement.
- `applicable(impl_idx, input)` — defaults to `true`. Inapplicable impls (e.g.
  `dict`'s direct-address table on an unbounded key range) are masked out of the
  loss, the regret argmin, and inference.

Adding a domain is implementing this trait and nothing else; it inherits
training, selection, evaluation, steering, online A/B, and both bindings for
free. Packaged domains today: `sort.rs`, `dict.rs`.

## The model (`model.rs`)

A tiny 1-hidden-layer MLP regressor (hidden width 32) on real `candle`. It
predicts a **per-impl log-cost vector**; the pick is the argmin over applicable
impls, so training is *regret-aware* — it optimizes for picking the cheapest
impl, not for predicting every cost accurately. Capacity is deliberately small;
per the research writeup, model size is not the bottleneck.

It is a mechanical port of ml-prototyping's `@voidloop/ml-core` training loop.
Two spots are intentionally *not* mechanical: gradient clipping is rewritten in
terms of candle's real `GradStore`, and the ml-core relu-NaN-at-0 sharp edge
simply doesn't exist in real candle.

## Capture — where training rows come from (`capture.rs`)

Both training modes emit the *same* row shape — `RawRow { features, costs, mask }`
— so the masked training loop consumes them identically:

- **Mode 1 (test-driven):** a `Recorder` observes workloads as the consumer's
  tests drive the domain interface, running **every** applicable impl to fill the
  full cost row. More diverse tests → a better chooser.
- **Mode 2 (production A/B):** fills a **single** cost column — the one impl
  actually served — leaving the mask a one-hot. Partial, bandit-style feedback.

`capture.rs` also holds the regret evaluation used to report a trained chooser
against the oracle and the best fixed impl.

## The chooser (`chooser.rs`)

`Chooser<D>` is the generic glue: it owns a trained `MlpModel`, the feature
standardization stats (mean/std), and the steering `Config`. Because it's written
entirely against the `Domain` trait, every domain gets `train` / `select` /
`run` / `evaluate` for free. Training log-clamps raw costs (a `COST_FLOOR` guards
against `ln(0)`), standardizes features, and fits the model. The trained state
persists as safetensors weights plus a JSON `Meta` sidecar (domain, impl names,
standardization stats, layer dims) — enough to reload and run without retraining.

## Steering (`config.rs`)

A consumer can steer the default policy per call **without retraining**:

- `bias` — additive per-impl nudge on predicted log-cost (negative favors).
- `forbidden` — hard-mask an impl out of selection.
- `force` — pin one impl and bypass the model (still applicability-checked).

This is the per-call analog of ml-prototyping's `CostObjective`.

## Mode 2 online path (`online.rs`)

In production you serve each request with exactly one impl, so you only learn
that impl's cost on that workload. `OnlineAb` A/B-switches which impl serves each
call — mostly exploit the current model, occasionally explore a random applicable
impl (epsilon-greedy) — and logs every outcome as a single-column `RawRow`. The
protocol is two calls: `choose` (the A/B decision) and `record` (log the impl run
and the measured cost); `dispatch` fuses choose → run → measure → record for
simulations and demos. Logs persist as JSONL (`export_log` / `read_log_jsonl`),
and `retrain_from_log` rebuilds a fresh `Chooser` offline from a log file — the
genuinely offline arm of the loop. Once exploration has covered the impls across
the feature space, a plain retrain recovers the full cost surface from partial
feedback.

## Bindings

`optigon-node` (napi-rs → one `.node` for Node + Bun) and `optigon-python` (PyO3
→ an abi3 wheel) are thin 1:1 wrappers over a concrete `Chooser<Sort>` etc. — no
logic of their own. Both rely on `panic = "unwind"` (set at the workspace root)
so the FFI layer can catch a Rust panic and surface it as a JS/Python exception
rather than aborting the host runtime.
