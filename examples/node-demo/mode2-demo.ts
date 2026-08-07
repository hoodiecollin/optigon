// Optigon node mode2-demo — the production A/B path (Mode 2), end to end.
//
// Unlike Mode 1 (which runs every impl offline to build a full cost row), Mode 2
// serves each call with exactly ONE impl and only learns that impl's cost on
// that workload. It covers the space by A/B switching (epsilon-greedy), logs
// every outcome as a single-observation row, and later retrains offline from the
// accumulated bandit-style log. The loop demonstrated here:
//
//   1. cold-start capture : a fresh dispatcher explores, serving traffic and
//      logging one measured cost per call,
//   2. export             : write the capture log to JSONL,
//   3. offline retrain    : fit a fresh chooser straight from the log file and
//      save new safetensors,
//   4. redeploy           : warm-start online serving from the retrained model,
//      now exploiting it (with a little residual exploration).
//
// Run from the repo root with `bun node:demo:mode2` (or `make node-demo-mode2`).

import {
	SortChooser,
	SortOnline,
	retrainSortFromLog,
	sortImplNames,
} from "optigon"

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

function buildScenario(
	n: number,
	sortedness: number,
	keySpaceMult: number,
	seed: number,
): number[] {
	const rng = mulberry32(seed ^ 0x9e3779b1)
	const size = Math.max(2, Math.round(n))
	const keySpace = Math.max(2, Math.round(size * keySpaceMult))
	const sorted = Math.min(1, Math.max(0, sortedness))
	const keys = new Array<number>(size)
	for (let i = 0; i < size; i++) keys[i] = Math.floor(rng() * keySpace)
	keys.sort((a, b) => a - b)
	const disturb = Math.round((1 - sorted) * size)
	for (let t = 0; t < disturb; t++) {
		const i = Math.floor(rng() * size) % size
		keys[i] = Math.floor(rng() * keySpace)
	}
	return keys
}

function corpus(count: number, baseSeed: number): number[][] {
	const out: number[][] = []
	for (let i = 0; i < count; i++) {
		const seed = baseSeed + i
		const r = i % 4
		if (r === 0) out.push(buildScenario(16 + (i % 48), 0.2, 4.0, seed))
		else if (r === 1) out.push(buildScenario(2000 + (i % 2000), 0.9, 8.0, seed))
		else if (r === 2)
			out.push(buildScenario(4000 + (i % 4000), 0.1, 64.0, seed))
		else out.push(buildScenario(8000 + (i % 8000), 0.0, 0.2, seed))
	}
	return out
}

const pct = (x: number) => `${(x * 100).toFixed(1)}%`

console.log("Optigon — sort chooser demo (Mode 2: production A/B capture)\n")
console.log(`impls: ${sortImplNames().join(", ")}\n`)

// 1. Cold-start capture: explore + serve + log, one measured cost per call.
const online = new SortOnline(0.15, 1)
console.log(
	"simulating production traffic (A/B switching, 1 measured cost/call)…",
)
for (const keys of corpus(320, 1)) online.dispatch(keys)
console.log(
	`  served ${online.observations()} calls; serving model trained yet? ${online.isTrained()}\n`,
)

// 2. Export the capture log.
const logPath = `${import.meta.dir}/optigon-online-log.jsonl`
online.exportLog(logPath)
console.log(`exported ${online.observations()} single-observation rows to`)
console.log(`  ${logPath}\n`)

// 3. Offline retrain straight from the log file → fresh safetensors.
const outPrefix = `${import.meta.dir}/optigon-online-model`
console.log("offline retrain from the captured log…")
const t = retrainSortFromLog(logPath, outPrefix, 800)
console.log(
	`  loss ${t.initialLoss.toFixed(4)} → ${t.finalLoss.toFixed(4)} over ${t.steps} steps`,
)
console.log(`  saved ${outPrefix}.{safetensors,meta.json}\n`)

// The offline artifact, scored on held-out workloads.
const trained = SortChooser.load(outPrefix)
const report = trained.evaluate(corpus(120, 9999))
console.log(
	"held-out evaluation (chooser recovered from partial A/B feedback):",
)
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

// 4. Redeploy: warm-start online serving from the retrained model.
const online2 = SortOnline.withModel(outPrefix, 0.1, 2)
console.log(
	`redeployed online (warm-started, exploiting); trained? ${online2.isTrained()}`,
)
const samples: [string, number[]][] = [
	["tiny nearly-sorted", buildScenario(24, 0.95, 4.0, 7)],
	["wide random       ", buildScenario(6000, 0.05, 64.0, 8)],
	["large narrow-key   ", buildScenario(12000, 0.0, 0.15, 9)],
]
console.log("greedy picks now:")
for (const [label, keys] of samples) {
	console.log(`  ${label} (n=${keys.length}) → ${online2.selectedName(keys)}`)
}
