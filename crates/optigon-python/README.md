# optigon

Learned per-workload dispatch, as a Python native extension. Package several
interchangeable implementations of an operation behind one interface, and learn
— per workload — which one is fastest, using regret-scored adaptive dispatch.

A thin [PyO3](https://pyo3.rs) wrapper over the Rust `optigon-core`; all the
logic lives there. Source, docs, and issues:
**<https://github.com/hoodiecollin/optigon>**

> **Status: experimental.** Early, single-author, and the API is unstable. No
> stability or support guarantees.

```python
from optigon import SortChooser

chooser = SortChooser()
for keys in my_workloads:
    chooser.observe(keys)         # Mode-1 capture: measures every impl
chooser.train()

sorted_keys = chooser.sort(keys)  # picks the fastest sort, runs it
```

Every implementation behind a domain produces the same result, so the chooser
only ever trades on speed: **a mispredicted choice is slower, never wrong.** The
corollary is also true — Optigon can be *slower* than a fixed choice, because
feature extraction plus inference is real work. It pays off only where that cost
is near-free relative to the operation.

## What's exposed

- `SortChooser` — the sort domain (insertion, quick, merge, radix)
- `DictChooser` — dictionary lookup (linear, binary, hash, direct), with
  `direct` masked out automatically when the key range is unbounded
- `SortOnline` — Mode-2 production A/B capture: serve each call with one impl,
  log the measured cost, retrain offline from the log, redeploy warm

## Building from source

```
maturin develop --release      # or `make py-build` from the repo root
```

## License

Dual-licensed under either MIT or Apache-2.0, at your option.
