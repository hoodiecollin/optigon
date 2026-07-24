"""Optigon py-demo — the walking skeleton's end-to-end proof, Mode 1
(test-driven training) through the native Python extension.

Mirrors examples/node-demo/demo.ts: generate diverse sort workloads, `observe`
each so the chooser measures every sort, `train`, then `evaluate` on held-out
workloads and show the learned chooser beating the best single fixed sort.

Run from the repo root with `make py-demo` (uses the venv the extension is
installed into)."""

from optigon import SortChooser, sort_impl_names


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


def build_scenario(n, sortedness, key_space_mult, seed):
    rng = mulberry32(seed ^ 0x9E3779B1)
    n = max(2, round(n))
    key_space = max(2, round(n * key_space_mult))
    sortedness = min(1.0, max(0.0, sortedness))
    keys = [int(rng() * key_space) for _ in range(n)]
    keys.sort()
    disturb = round((1.0 - sortedness) * n)
    for _ in range(disturb):
        i = int(rng() * n) % n
        keys[i] = int(rng() * key_space)
    return keys


def corpus(count, base_seed):
    out = []
    for i in range(count):
        seed = base_seed + i
        r = i % 4
        if r == 0:
            out.append(build_scenario(16 + (i % 48), 0.2, 4.0, seed))  # tiny
        elif r == 1:
            out.append(build_scenario(2000 + (i % 2000), 0.9, 8.0, seed))  # large nearly-sorted
        elif r == 2:
            out.append(build_scenario(4000 + (i % 4000), 0.1, 64.0, seed))  # wide random
        else:
            out.append(build_scenario(8000 + (i % 8000), 0.0, 0.2, seed))  # large narrow-key (radix)
    return out


def main():
    print("Optigon — sort chooser demo (Mode 1: trained by running tests)\n")
    print(f"impls: {', '.join(sort_impl_names())}\n")

    chooser = SortChooser()

    print("observing workloads (measuring every sort on each)…")
    for keys in corpus(240, 1):
        chooser.observe(keys)
    print(f"  observed {chooser.observed()} workloads")

    print("training…")
    t = chooser.train(600)
    print(f"  loss {t.initial_loss:.4f} → {t.final_loss:.4f} over {t.steps} steps\n")

    report = chooser.evaluate(corpus(120, 9999))
    print("held-out evaluation:")
    print(
        f"  best fixed sort : {report.best_fixed_impl_name:<9} mean regret {report.best_fixed_mean:.4f}"
    )
    print(
        f"  learned chooser : {'adaptive':<9} mean regret {report.learned_mean:.4f}  "
        f"(optimal pick {report.optimal_rate * 100:.1f}%)"
    )
    ratio = report.best_fixed_mean / max(report.learned_mean, 1e-9)
    print(f"  → learned is ~{ratio:.1f}× lower regret than any fixed choice\n")

    samples = [
        ("tiny nearly-sorted", build_scenario(24, 0.95, 4.0, 7)),
        ("wide random       ", build_scenario(6000, 0.05, 64.0, 8)),
        ("large narrow-key  ", build_scenario(12000, 0.0, 0.15, 9)),
    ]
    print("per-workload picks:")
    for label, keys in samples:
        print(f"  {label} (n={len(keys)}) → {chooser.selected_name(keys)}")

    import os

    prefix = os.path.join(os.path.dirname(__file__), "optigon-sort")
    chooser.save(prefix)
    reloaded = SortChooser.load(prefix)
    print(
        f"\nsaved to {prefix}.{{safetensors,meta.json}}; "
        f"reloaded model is trained: {reloaded.is_trained()}"
    )


if __name__ == "__main__":
    main()
