"""Optigon py mode2-demo — the production A/B path (Mode 2), end to end.

Mirrors examples/node-demo/mode2-demo.ts. Unlike Mode 1 (which runs every impl
offline to build a full cost row), Mode 2 serves each call with exactly ONE impl
and only learns that impl's cost on that workload. It covers the space by A/B
switching (epsilon-greedy), logs every outcome as a single-observation row, and
later retrains offline from the accumulated bandit-style log:

  1. cold-start capture : a fresh dispatcher explores, serving traffic and
     logging one measured cost per call,
  2. export             : write the capture log to JSONL,
  3. offline retrain    : fit a fresh chooser straight from the log file and
     save new safetensors,
  4. redeploy           : warm-start online serving from the retrained model.

Run from the repo root with `make py-demo-mode2`."""

import os

from optigon import SortChooser, SortOnline, retrain_sort_from_log, sort_impl_names


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
            out.append(build_scenario(16 + (i % 48), 0.2, 4.0, seed))
        elif r == 1:
            out.append(build_scenario(2000 + (i % 2000), 0.9, 8.0, seed))
        elif r == 2:
            out.append(build_scenario(4000 + (i % 4000), 0.1, 64.0, seed))
        else:
            out.append(build_scenario(8000 + (i % 8000), 0.0, 0.2, seed))
    return out


def main():
    print("Optigon — sort chooser demo (Mode 2: production A/B capture)\n")
    print(f"impls: {', '.join(sort_impl_names())}\n")

    here = os.path.dirname(__file__)

    # 1. Cold-start capture: explore + serve + log, one measured cost per call.
    online = SortOnline(0.15, 1)
    print("simulating production traffic (A/B switching, 1 measured cost/call)…")
    for keys in corpus(320, 1):
        online.dispatch(keys)
    print(
        f"  served {online.observations()} calls; "
        f"serving model trained yet? {online.is_trained()}\n"
    )

    # 2. Export the capture log.
    log_path = os.path.join(here, "optigon-online-log.jsonl")
    if os.path.exists(log_path):
        os.remove(log_path)
    online.export_log(log_path)
    print(f"exported {online.observations()} single-observation rows to")
    print(f"  {log_path}\n")

    # 3. Offline retrain straight from the log file → fresh safetensors.
    out_prefix = os.path.join(here, "optigon-online-model")
    print("offline retrain from the captured log…")
    t = retrain_sort_from_log(log_path, out_prefix, 800)
    print(f"  loss {t.initial_loss:.4f} → {t.final_loss:.4f} over {t.steps} steps")
    print(f"  saved {out_prefix}.{{safetensors,meta.json}}\n")

    # The offline artifact, scored on held-out workloads.
    trained = SortChooser.load(out_prefix)
    report = trained.evaluate(corpus(120, 9999))
    print("held-out evaluation (chooser recovered from partial A/B feedback):")
    print(
        f"  best fixed sort : {report.best_fixed_impl_name:<9} mean regret {report.best_fixed_mean:.4f}"
    )
    print(
        f"  learned chooser : {'adaptive':<9} mean regret {report.learned_mean:.4f}  "
        f"(optimal pick {report.optimal_rate * 100:.1f}%)"
    )
    ratio = report.best_fixed_mean / max(report.learned_mean, 1e-9)
    print(f"  → learned is ~{ratio:.1f}× lower regret than any fixed choice\n")

    # 4. Redeploy: warm-start online serving from the retrained model.
    online2 = SortOnline.with_model(out_prefix, 0.1, 2)
    print(
        f"redeployed online (warm-started, exploiting); trained? {online2.is_trained()}"
    )
    samples = [
        ("tiny nearly-sorted", build_scenario(24, 0.95, 4.0, 7)),
        ("wide random       ", build_scenario(6000, 0.05, 64.0, 8)),
        ("large narrow-key  ", build_scenario(12000, 0.0, 0.15, 9)),
    ]
    print("greedy picks now:")
    for label, keys in samples:
        print(f"  {label} (n={len(keys)}) → {online2.selected_name(keys)}")


if __name__ == "__main__":
    main()
