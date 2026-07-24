// Optigon node-demo — the walking skeleton's end-to-end proof, Mode 1
// (test-driven training) through the native addon.
//
//   1. generate a diverse set of sort workloads (as a test suite would exercise
//      the domain interface),
//   2. `observe` each one so the chooser measures every sort on it,
//   3. `train`, then
//   4. `evaluate` on held-out workloads and show the learned chooser beating the
//      best single fixed sort.
//
// Run from the repo root with `bun node:demo` (or `make node-demo`).

import { SortChooser, sortImplNames } from "optigon-node"

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

// Mirror of the core's build_sort_scenario: n / sortedness / keySpaceMult.
function buildScenario(
	n: number,
	sortedness: number,
	keySpaceMult: number,
	seed: number,
): number[] {
	const rng = mulberry32(seed ^ 0x9e3779b1)
	n = Math.max(2, Math.round(n))
	const keySpace = Math.max(2, Math.round(n * keySpaceMult))
	sortedness = Math.min(1, Math.max(0, sortedness))
	const keys = new Array<number>(n)
	for (let i = 0; i < n; i++) keys[i] = Math.floor(rng() * keySpace)
	keys.sort((a, b) => a - b)
	const disturb = Math.round((1 - sortedness) * n)
	for (let t = 0; t < disturb; t++) {
		const i = Math.floor(rng() * n) % n
		keys[i] = Math.floor(rng() * keySpace)
	}
	return keys
}

// A workload distribution that spans the regimes each sort wins.
function corpus(count: number, baseSeed: number): number[][] {
	const out: number[][] = []
	for (let i = 0; i < count; i++) {
		const seed = baseSeed + i
		const r = i % 4
		if (r === 0)
			out.push(buildScenario(16 + (i % 48), 0.2, 4.0, seed)) // tiny
		else if (r === 1)
			out.push(buildScenario(2000 + (i % 2000), 0.9, 8.0, seed)) // large nearly-sorted
		else if (r === 2)
			out.push(buildScenario(4000 + (i % 4000), 0.1, 64.0, seed)) // wide random
		else out.push(buildScenario(8000 + (i % 8000), 0.0, 0.2, seed)) // large narrow-key (radix)
	}
	return out
}

console.log("Optigon — sort chooser demo (Mode 1: trained by running tests)\n")
console.log(`impls: ${sortImplNames().join(", ")}\n`)

const chooser = new SortChooser()

console.log("observing workloads (measuring every sort on each)…")
for (const keys of corpus(240, 1)) chooser.observe(keys)
console.log(`  observed ${chooser.observed()} workloads`)

console.log("training…")
const t = chooser.train(600)
console.log(
	`  loss ${t.initialLoss.toFixed(4)} → ${t.finalLoss.toFixed(4)} over ${t.steps} steps\n`,
)

const report = chooser.evaluate(corpus(120, 9999))
const pct = (x: number) => `${(x * 100).toFixed(1)}%`
console.log("held-out evaluation:")
console.log(
	`  best fixed sort : ${report.bestFixedImplName.padEnd(9)} mean regret ${report.bestFixedMean.toFixed(4)}`,
)
console.log(
	`  learned chooser : ${"adaptive".padEnd(9)} mean regret ${report.learnedMean.toFixed(4)}  (optimal pick ${pct(report.optimalRate)})`,
)
const ratio = report.bestFixedMean / Math.max(report.learnedMean, 1e-9)
console.log(
	`  → learned is ~${ratio.toFixed(1)}× lower regret than any fixed choice\n`,
)

// A few concrete picks, showing the winner changing with the workload.
const samples: [string, number[]][] = [
	["tiny nearly-sorted", buildScenario(24, 0.95, 4.0, 7)],
	["wide random       ", buildScenario(6000, 0.05, 64.0, 8)],
	["large narrow-key   ", buildScenario(12000, 0.0, 0.15, 9)],
]
console.log("per-workload picks:")
for (const [label, keys] of samples) {
	console.log(`  ${label} (n=${keys.length}) → ${chooser.selectedName(keys)}`)
}

// Persist + reload round-trip.
const prefix = `${import.meta.dir}/optigon-sort`
chooser.save(prefix)
const reloaded = SortChooser.load(prefix)
console.log(
	`\nsaved to ${prefix}.{safetensors,meta.json}; reloaded model is trained: ${reloaded.isTrained()}`,
)
