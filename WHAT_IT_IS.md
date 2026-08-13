# What Optigon is — and isn't

An honest per-feature account of what exists in the code today, what it guarantees, and
where it stops. **Verify every maturity claim here against the code before relying on
it** — this file is a claim about the tree, and claims drift. **Where the README
over-promises, this document wins.**

Status of the whole project: **experimental**. Single author, no published release, API
unstable, no distribution story yet. Running from source is the only supported path.

## The one guarantee that is load-bearing

**A mispredicted choice is slower, never wrong.** Every implementation packaged behind a
`Domain` produces the same result; the chooser only ever trades on speed. This is a
property of the design, not of the model — an untrained, badly trained, or actively
adversarial chooser still returns correct output. Everything else in this file is a
performance or ergonomics claim; this one is a correctness claim, and it holds.

The corollary is also true and worth stating plainly: **Optigon can be slower than a
fixed choice.** Feature extraction plus inference is real work. It pays off only where
that cost is near-free relative to the operation, which is the economic precondition the
whole thesis rests on.

## Per-feature state

| Feature | State | Honest limit |
|---|---|---|
| `Domain` trait | Works | `run()` is infallible and `cost()` is a scalar — see the interface gaps below |
| `sort` domain | Works end to end | 4 impls (insertion, quick, merge, radix), `Vec<i32>` only |
| `dict` domain | Works end to end | 4 impls (linear, binary, hash, direct); `direct` is masked out on unbounded key ranges |
| MLP model (`candle`) | Works | 1 hidden layer, width 32, deliberately small; a port of `@voidloop/ml-core`'s loop |
| Regret-scored training | Works | Optimizes for picking the cheapest impl, not for accurate cost prediction — the reported per-impl costs are a means, not an output |
| Applicability masking | Works | Input-derived only; cannot see the deployment environment (#9) |
| Mode 1 (`Recorder`) | Works | Chooser quality is bounded by how diverse your test workloads are |
| Mode 2 (`OnlineAb`) | Works in core | **Bindings expose it for `sort` only** — there is no `DictOnline` in either binding |
| Config steering (pin / forbid / bias) | Works | — |
| Node binding (napi) | Works | Hand-written per domain: `SortChooser`, `DictChooser`, `SortOnline`. Adding a domain means writing another ~130-line class by hand (#1) |
| Python binding (pyo3) | Works | Same shape, same limit: `SortChooser`, `DictChooser`, `SortOnline` |
| Test coverage | 11 unit tests in `optigon-core` | Bindings are covered by the `examples/` demos, not by unit tests |

## What it is

- A **learned dispatch layer**. Given a workload it predicts which of several
  correct-but-differently-performing implementations is fastest, then runs that one.
- **Generic above the `Domain` trait.** A new domain is one trait impl; it inherits
  training, selection, evaluation, steering, and online A/B for free. The *bindings* are
  the exception — those are still per-domain handwork (#1).
- **Two training modes that emit the same row shape**: offline capture from your tests
  (Mode 1) and epsilon-greedy production A/B (Mode 2), so a chooser can be retrained
  offline from production logs and redeployed warm.

## What it is not

- **Not an autotuner or a JIT.** It will not invent implementations, tune their
  parameters, or rewrite your code. You supply the impls.
- **Not a correctness mechanism.** It never chooses between implementations that differ
  in output. If your impls disagree, Optigon will happily serve you the disagreement.
- **Not multi-objective.** `cost()` is one `f64`. Domains whose implementations trade
  quality against speed (compression is the canonical case) do not fit the current
  interface — that is the whole content of #8, and #7 exists to confront it.
- **Not distributable yet.** No npm prebuilds, no wheel matrix, no published release.
  `make node-build` / `make py-build` from source is the supported path.
- **Not stable.** The API will break. There is no deprecation policy because there is
  nothing published to deprecate against.

## Known interface gaps

These are gaps in a *shipped* abstraction, tracked as `improvement`, and they bound what
domains can be added without an interface change:

- **#8 — cost is a scalar.** Blocks every multi-objective domain (compression, sketches,
  approximate search).
- **#9 — `applicable()` sees only the input.** It cannot express feasibility that depends
  on the deployment (whether an index exists, how much memory is available).
- **#10 — `run()` is infallible.** An impl that fails on data rather than on programmer
  error has only `panic!` available to it.

Epic #11 tracks domain expansion together with these gaps, because expansion is what
exposes them.
