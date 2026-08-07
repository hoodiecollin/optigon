# optigon-core

The language-agnostic core of [Optigon](https://github.com/hoodiecollin/optigon):
package several interchangeable implementations of an operation behind one
`Domain` trait, and learn — per workload — which implementation is fastest,
using regret-scored adaptive dispatch.

> **Status: experimental.** Early, single-author, and the API is unstable. No
> stability or support guarantees.

Everything above the `Domain` trait is generic, so a new domain is *one trait
impl* and it inherits training, selection, evaluation, steering, and online A/B
capture for free.

```rust
use optigon_core::{Chooser, Domain, Recorder, TrainConfig, sort::Sort};

let mut recorder: Recorder<Sort> = Recorder::new();
for keys in workloads {
    recorder.observe(&keys); // measures every applicable impl
}

let mut chooser: Chooser<Sort> = Chooser::new();
chooser.train(&recorder, &TrainConfig::default())?;

let sorted = chooser.run(&keys); // picks the fastest sort, runs it
```

## What it is

A **learned dispatch layer**. Every implementation packaged behind a `Domain`
produces the same result, so the chooser only ever trades on speed: **a
mispredicted choice is slower, never wrong.** That is a property of the design,
not of the model — an untrained or actively adversarial chooser still returns
correct output.

The corollary is worth stating too: Optigon can be *slower* than a fixed choice.
Feature extraction plus inference is real work, and it pays off only where that
cost is near-free relative to the operation.

It is **not** an autotuner or a JIT. It will not invent implementations, tune
their parameters, or rewrite your code — you supply the impls.

## What's here

- `Domain` — the trait: input, output, impl names, cheap features, cost, and an
  applicability mask for impls that don't apply to every workload.
- Two packaged domains — `sort` (insertion, quick, merge, radix) and `dict`
  (linear, binary, hash, direct).
- `Chooser` — regret-scored MLP over [candle](https://github.com/huggingface/candle):
  train, select, run, evaluate.
- `Recorder` (Mode 1, test-driven capture) and `OnlineAb` (Mode 2, production
  epsilon-greedy A/B) — both emit the same row shape, so a chooser can be
  retrained offline from production logs.
- `Config` — steer without retraining: pin, forbid, or bias an impl.

## Known interface gaps

Gaps in a shipped abstraction, tracked in the repo:

- **`cost()` is a scalar** — multi-objective domains don't fit yet ([#8](https://github.com/hoodiecollin/optigon/issues/8)).
- **`applicable()` sees only the input** — not the deployment environment ([#9](https://github.com/hoodiecollin/optigon/issues/9)).
- **`run()` is infallible** — an impl that fails on data has only `panic!` ([#10](https://github.com/hoodiecollin/optigon/issues/10)).

See [`WHAT_IT_IS.md`](https://github.com/hoodiecollin/optigon/blob/main/WHAT_IT_IS.md)
for the honest per-feature limits, and
[`docs/architecture.md`](https://github.com/hoodiecollin/optigon/blob/main/docs/architecture.md)
for how the pieces fit.

## Bindings

TypeScript (Node/Bun, napi-rs) and Python (PyO3) wrappers live in the same
repository. They are thin 1:1 wrappers with no logic of their own.

## License

Dual-licensed under either [MIT](https://github.com/hoodiecollin/optigon/blob/main/LICENSE-MIT)
or [Apache-2.0](https://github.com/hoodiecollin/optigon/blob/main/LICENSE-APACHE),
at your option.
