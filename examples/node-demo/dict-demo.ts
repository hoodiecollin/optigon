// Optigon node dict-demo — the second domain, end-to-end, Mode 1 (test-driven
// training) through the native addon.
//
//   1. generate diverse dictionary workloads (keys to index + queries to answer),
//   2. `observe` each so the chooser measures every applicable lookup on it,
//   3. `train`, then
//   4. `evaluate` on held-out workloads and show the learned chooser beating the
//      best single fixed lookup structure.
//
// The point of this domain: `direct` (a direct-address table) is only applicable
// for a bounded key range, so no fixed policy can commit to it — but the learned
// chooser reaches for it exactly on the narrow-key workloads where it wins.
//
// Run from the repo root with `bun node:demo:dict` (or `make node-demo-dict`).

import { DictChooser, dictImplNames } from "optigon"

// mulberry32 — same PRNG as the Rust core, so workloads look alike across sides.
function mulberry32(seed: number): () => number {
	let s = seed >>> 0
	return () => {
		s = (s + 0x6d2b79f5) | 0
		let t = Math.imul(s ^ (s >>> 15), 1 | s)
		t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296
	}
}

// Mirror of the core's build_dict_scenario.
function buildScenario(
	nKeys: number,
	nQueries: number,
	keySpaceMult: number,
	hitRate: number,
	seed: number,
): { keys: number[]; queries: number[] } {
	const rng = mulberry32(seed ^ 0x85ebca6b)
	const k = Math.max(1, Math.round(nKeys))
	const q = Math.max(1, Math.round(nQueries))
	const keySpace = Math.max(2, Math.round(k * keySpaceMult))
	const hit = Math.min(1, Math.max(0, hitRate))
	const keys = new Array<number>(k)
	for (let i = 0; i < k; i++) keys[i] = Math.floor(rng() * keySpace)
	const queries = new Array<number>(q)
	for (let i = 0; i < q; i++) {
		if (rng() < hit) queries[i] = keys[Math.floor(rng() * k) % k]
		else queries[i] = Math.floor(rng() * (keySpace * 2))
	}
	return { keys, queries }
}

// A workload distribution spanning the regimes each lookup wins.
function corpus(
	count: number,
	baseSeed: number,
): { keys: number[]; queries: number[] }[] {
	const out: { keys: number[]; queries: number[] }[] = []
	for (let i = 0; i < count; i++) {
		const seed = baseSeed + i
		const r = i % 4
		if (r === 0)
			out.push(buildScenario(8 + (i % 32), 8 + (i % 32), 1.0, 0.5, seed)) // tiny
		else if (r === 1)
			out.push(buildScenario(1000, 1000, 8.0, 0.5, seed)) // medium
		else if (r === 2)
			out.push(buildScenario(4000, 4000, 512.0, 0.3, seed)) // wide → direct masked
		else out.push(buildScenario(4000, 6000, 0.1, 0.7, seed)) // narrow → direct wins
	}
	return out
}

console.log("Optigon — dict chooser demo (Mode 1: trained by running tests)\n")
console.log(`impls: ${dictImplNames().join(", ")}\n`)

const chooser = new DictChooser()

console.log("observing workloads (measuring every applicable lookup on each)…")
for (const w of corpus(200, 1)) chooser.observe(w.keys, w.queries)
console.log(`  observed ${chooser.observed()} workloads`)

console.log("training…")
const t = chooser.train(600)
console.log(
	`  loss ${t.initialLoss.toFixed(4)} → ${t.finalLoss.toFixed(4)} over ${t.steps} steps\n`,
)

const evalSet = corpus(100, 9999)
const report = chooser.evaluate(
	evalSet.map((w) => w.keys),
	evalSet.map((w) => w.queries),
)
const pct = (x: number) => `${(x * 100).toFixed(1)}%`
console.log("held-out evaluation:")
console.log(
	`  best fixed lookup : ${report.bestFixedImplName.padEnd(9)} mean regret ${report.bestFixedMean.toFixed(4)}`,
)
console.log(
	`  learned chooser   : ${"adaptive".padEnd(9)} mean regret ${report.learnedMean.toFixed(4)}  (optimal pick ${pct(report.optimalRate)})`,
)
const ratio = report.bestFixedMean / Math.max(report.learnedMean, 1e-9)
console.log(
	`  → learned is ~${ratio.toFixed(1)}× lower regret than any fixed choice\n`,
)

// A few concrete picks, showing the winner changing with the workload — and the
// chooser reaching for `direct` only where the key range is bounded.
const samples: [string, { keys: number[]; queries: number[] }][] = [
	["tiny            ", buildScenario(16, 16, 1.0, 0.5, 7)],
	["wide-key large  ", buildScenario(5000, 5000, 512.0, 0.3, 8)],
	["narrow-key large", buildScenario(5000, 8000, 0.1, 0.7, 9)],
]
console.log("per-workload picks:")
for (const [label, w] of samples) {
	console.log(
		`  ${label} (keys=${w.keys.length}, queries=${w.queries.length}) → ${chooser.selectedName(w.keys, w.queries)}`,
	)
}

// Persist + reload round-trip.
const prefix = `${import.meta.dir}/optigon-dict`
chooser.save(prefix)
const reloaded = DictChooser.load(prefix)
console.log(
	`\nsaved to ${prefix}.{safetensors,meta.json}; reloaded model is trained: ${reloaded.isTrained()}`,
)
