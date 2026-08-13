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

**Scope is now locked across all three milestones** (2026-08-07). The shape of the
decision was: fix the trait, then publish it, then grow it.

| Milestone | What ships | Issues |
|---|---|---|
| **v0.1.0** | The `Domain`-trait shape — the three interface gaps in a shipped abstraction, plus the binding generator that removes per-domain handwork | #8, #9, #10, #1 |
| **v0.2.0** | First published release: crates.io + npm prebuilds + PyPI wheels, release automation, CI that builds the artifacts, branch protection. Plus the website and launch promotion, neither of which gates the tag | Epic #15 (10 children), gate #26, site #27, promo #28–#30 |
| **v0.3.0** | Domain expansion on a settled interface | #2, #3, #4, #5, #6, #7 |

```bash
gh issue list --milestone v0.1.0            # the cycle in flight
gh issue list --label release-gate --state open   # any row blocks its milestone's tag
npx @hoodiecollin/pm-playbook ladder        # the derived rung, including the unmilestoned ideas
```

## Why this order

**The trait comes first because it is the product.** #8 (cost is a scalar), #9
(`applicable()` sees only the input) and #10 (`run()` is infallible) each change the
signature of `Domain` — the one thing a consumer implements. Publishing before they land
would burn the first published API on a shape already scheduled to break three ways;
adding domains before they land would mean writing six trait impls against that same
shape and then rewriting all six. Both failure modes are avoided by the same ordering.

This is wired as **native GitHub blocking relationships**, not prose: every domain issue
(#2–#7) is *blocked by* #8, #9, #10, and #1. #1 earns its place there because each domain
issue's own acceptance criteria already require the generator ("via the generator macro —
do not hand-write").

**Publishing sits between them** rather than last. Two domains is a defensible first
release, the honest scope note already says as much, and a published artifact is worth
more than a sixth domain nobody can install.

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

Everything below is open work. The ordering rationale lives on each issue; this is the
shape of it:

| Area | Issues | Milestone |
|---|---|---|
| Interface gaps in a shipped abstraction | #8 scalar cost · #9 input-only applicability · #10 infallible `run()` | v0.1.0 |
| Binding ergonomics | #1 generate napi + pyo3 classes from a `Domain` impl | v0.1.0 |
| Distribution — three channels, none of them built | Epic #15: names, package hygiene, crates.io, npm prebuilds, wheels, CI, automation, community health, sweep, tag | v0.2.0 |
| Website + launch | #27 site · #28 repo presence · #29 the writeup · #30 the announcement | v0.2.0 |
| Domain expansion | #2 topk · #3 substring search · #4 dedup · #5 join · #6 shortest path · #7 compression | v0.3.0 |

Epic #11 is the umbrella over domain expansion and the interface gaps together; it spans
v0.1.0 and v0.3.0 and so carries no milestone of its own. Epic #15 is the umbrella over
the first published release and sits wholly on v0.2.0.

## Not on the spine

`experiment` issues never carry a milestone (§4) — a spike's deliverable is a decision
that feeds the spine, not an artifact that rides it. There are none open at present.
