"""Optigon py dict-demo — the second domain, end-to-end, Mode 1 (test-driven
training) through the native Python extension.

Mirrors examples/node-demo/dict-demo.ts: generate diverse dictionary workloads
(keys to index + queries to answer), `observe` each so the chooser measures
every applicable lookup, `train`, then `evaluate` on held-out workloads and show
the learned chooser beating the best single fixed lookup structure.

The point of this domain: `direct` (a direct-address table) is only applicable
for a bounded key range, so no fixed policy can commit to it — but the learned
chooser reaches for it exactly on the narrow-key workloads where it wins.

Run from the repo root with `make py-demo-dict`."""

from optigon import DictChooser, dict_impl_names


def mulberry32(seed):
    """Same PRNG as the Rust core, so workloads look alike across languages."""
    s = seed & 0xFFFFFFFF

    def nxt():
        nonlocal s
        s = (s + 0x6D2B79F5) & 0xFFFFFFFF
        t = (s ^ (s >> 15)) * (1 | s) & 0xFFFFFFFF
        t = (t + ((t ^ (t >> 7)) * (61 | t) & 0xFFFFFFFF)) & 0xFFFFFFFF ^ t
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296.0

    return nxt


def build_scenario(n_keys, n_queries, key_space_mult, hit_rate, seed):
    rng = mulberry32(seed ^ 0x85EBCA6B)
    n_keys = max(1, round(n_keys))
    n_queries = max(1, round(n_queries))
    key_space = max(2, round(n_keys * key_space_mult))
    hit_rate = min(1.0, max(0.0, hit_rate))
    keys = [int(rng() * key_space) for _ in range(n_keys)]
    queries = []
    for _ in range(n_queries):
        if rng() < hit_rate:
            queries.append(keys[int(rng() * n_keys) % n_keys])
        else:
            queries.append(int(rng() * (key_space * 2)))
    return keys, queries


def corpus(count, base_seed):
    out = []
    for i in range(count):
        seed = base_seed + i
        r = i % 4
        if r == 0:
            out.append(build_scenario(8 + (i % 32), 8 + (i % 32), 1.0, 0.5, seed))  # tiny
        elif r == 1:
            out.append(build_scenario(1000, 1000, 8.0, 0.5, seed))  # medium
        elif r == 2:
            out.append(build_scenario(4000, 4000, 512.0, 0.3, seed))  # wide → direct masked
        else:
            out.append(build_scenario(4000, 6000, 0.1, 0.7, seed))  # narrow → direct wins
    return out


def main():
    print("Optigon — dict chooser demo (Mode 1: trained by running tests)\n")
    print(f"impls: {', '.join(dict_impl_names())}\n")

    chooser = DictChooser()

    print("observing workloads (measuring every applicable lookup on each)…")
    for keys, queries in corpus(200, 1):
        chooser.observe(keys, queries)
    print(f"  observed {chooser.observed()} workloads")

    print("training…")
    t = chooser.train(600)
    print(f"  loss {t.initial_loss:.4f} → {t.final_loss:.4f} over {t.steps} steps\n")

    eval_set = corpus(100, 9999)
    report = chooser.evaluate(
        [w[0] for w in eval_set],
        [w[1] for w in eval_set],
    )
    print("held-out evaluation:")
    print(
        f"  best fixed lookup : {report.best_fixed_impl_name:<9} mean regret {report.best_fixed_mean:.4f}"
    )
    print(
        f"  learned chooser   : {'adaptive':<9} mean regret {report.learned_mean:.4f}  "
        f"(optimal pick {report.optimal_rate * 100:.1f}%)"
    )
    ratio = report.best_fixed_mean / max(report.learned_mean, 1e-9)
    print(f"  → learned is ~{ratio:.1f}× lower regret than any fixed choice\n")

    samples = [
        ("tiny            ", build_scenario(16, 16, 1.0, 0.5, 7)),
        ("wide-key large  ", build_scenario(5000, 5000, 512.0, 0.3, 8)),
        ("narrow-key large", build_scenario(5000, 8000, 0.1, 0.7, 9)),
    ]
    print("per-workload picks:")
    for label, (keys, queries) in samples:
        print(
            f"  {label} (keys={len(keys)}, queries={len(queries)}) → "
            f"{chooser.selected_name(keys, queries)}"
        )

    import os

    prefix = os.path.join(os.path.dirname(__file__), "optigon-dict")
    chooser.save(prefix)
    reloaded = DictChooser.load(prefix)
    print(
        f"\nsaved to {prefix}.{{safetensors,meta.json}}; "
        f"reloaded model is trained: {reloaded.is_trained()}"
    )


if __name__ == "__main__":
    main()
