# Version roadmap

The honest state of the current release effort. **GitHub Issues are authoritative** —
this file is a narrative summary of them and can lag; when they disagree, the issues
win. Nothing here is a task list (§11: the backlog lives in Issues).

Milestones are *versions*, and a milestone closing is not a shipment: issues close into
a milestone until it is tagged, and on this roadmap they read as **pending release**
until the `vX.Y.Z` GitHub Release exists.

## Situation

Optigon is pre-release and experimental. The core thesis is demonstrated end to end —
two domains (`sort`, `dict`), both training modes, both bindings, regret-scored
selection on real `candle` — but nothing has been tagged, nothing is published, and the
API is unstable.

**No milestone has scope locked yet.** `v0.1.0`, `v0.2.0` and `v0.3.0` exist on GitHub
as empty containers; every open issue currently sits on the maturity axis only. That is
a deliberate resting state, not an oversight: scheduling is a commitment, and the first
scope-lock should be a decision made once, not drifted into.

```
gh issue list --label plan-next   # committed, unscheduled — what v0.1.0 would draw from
gh issue list --label idea        # speculative — needs an rfc first
gh issue list --milestone v0.1.0  # scheduled (currently empty)
```

## Complete (in code, unreleased)

Verified against the tree, not asserted from memory — see
[`WHAT_IT_IS.md`](./WHAT_IT_IS.md) for the per-feature limits:

- `Domain` trait and the generic machinery above it — chooser, model, capture, config,
  online A/B — all domain-agnostic.
- `sort` (4 impls) and `dict` (4 impls) domains, including applicability masking.
- Regret-scored MLP training on real `candle`.
- Mode 1 (`Recorder`) and Mode 2 (`OnlineAb`) capture, emitting one row shape.
- Node (napi) and Python (pyo3) bindings for `SortChooser`, `DictChooser`, `SortOnline`.
- 11 passing unit tests in `optigon-core`; end-to-end demos for both bindings and both
  modes under `examples/`.

## Still deferred

Everything below is open work, sequenced on engineering merit. The ordering rationale
lives on each issue; this is the shape of it:

| Area | Issues | Why it waits |
|---|---|---|
| Interface gaps in a shipped abstraction | #8 scalar cost · #9 input-only applicability · #10 infallible `run()` | Each is exposed *by* domain expansion, so they are being confronted alongside it rather than speculatively |
| Domain expansion | #2 topk · #3 substring search · #4 dedup · #5 join · #6 shortest path · #7 compression | Deliberately ordered: four scalar-cost domains before compression, so the multi-objective design (#8) meets real breadth instead of being fitted to one case |
| Binding ergonomics | #1 generate napi + pyo3 classes from a `Domain` impl | The core is generic; the bindings are not. Every new domain currently costs two hand-written ~130-line classes |
| Distribution | not yet filed | No npm prebuilds and no wheel matrix. Running from source is the supported path today |

Epic #11 is the umbrella over domain expansion and the interface gaps together.

## Not on the spine

`experiment` issues never carry a milestone (§4) — a spike's deliverable is a decision
that feeds the spine, not an artifact that rides it. There are none open at present.
