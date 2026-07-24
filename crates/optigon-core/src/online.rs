//! Mode 2 — production A/B capture and offline retraining.
//!
//! Mode 1 ([`Recorder`]) is an offline harness: it runs *every* impl on each
//! workload to record a full cost row. Mode 2 is the online path, and it can't
//! do that — in production you serve each request with exactly one impl, so you
//! only ever learn that one impl's cost on that one workload. The trick is to
//! *A/B switch* which impl serves each call: mostly exploit the current model,
//! occasionally explore a random applicable impl, and log every outcome as a
//! single-column [`RawRow`] (one-hot mask). The masked training loop consumes
//! those partial-feedback rows exactly like Mode-1's full rows — so once enough
//! exploration has covered the impls across the feature space, a plain retrain
//! recovers the full cost surface from bandit-style logs.
//!
//! The protocol is two calls:
//!   1. [`OnlineAb::choose`] — the A/B decision: which impl to serve this call.
//!   2. [`OnlineAb::record`] — log the impl the consumer ran and the cost it
//!      measured on its *real* workload.
//!
//! [`OnlineAb::dispatch`] fuses the two (choose → run → measure → record) for
//! simulations and demos where Optigon runs the impl itself. Logs persist as
//! JSONL via [`OnlineAb::export_log`] / [`read_log_jsonl`], and
//! [`retrain_from_log`] rebuilds a fresh [`Chooser`] from a log file — the
//! genuinely-offline arm of the loop.

use std::marker::PhantomData;
use std::time::Instant;

use candle_core::{Error, Result};

use crate::capture::RawRow;
use crate::chooser::Chooser;
use crate::config::Config;
use crate::domain::Domain;
use crate::model::{TrainConfig, TrainOutcome};
use crate::prng::Mulberry32;

/// An online A/B dispatcher: serves calls by epsilon-greedy switching over a
/// wrapped [`Chooser`], logging every outcome as a single-observation [`RawRow`]
/// for later offline retraining.
pub struct OnlineAb<D: Domain> {
    chooser: Chooser<D>,
    epsilon: f64,
    rng: Mulberry32,
    log: Vec<RawRow>,
    _pd: PhantomData<D>,
}

impl<D: Domain> OnlineAb<D> {
    /// A fresh dispatcher with no serving model yet: until [`OnlineAb::retrain`]
    /// is called it explores uniformly over applicable impls (pure A/B), which
    /// is exactly the coverage you want for the cold-start capture phase.
    pub fn new(epsilon: f64, seed: u32) -> Self {
        Self {
            chooser: Chooser::new(),
            epsilon: epsilon.clamp(0.0, 1.0),
            rng: Mulberry32::new(seed),
            log: Vec::new(),
            _pd: PhantomData,
        }
    }

    /// Start serving an already-trained chooser (warm start): exploit it with
    /// probability `1 - epsilon`, explore otherwise, and keep logging.
    pub fn from_chooser(chooser: Chooser<D>, epsilon: f64, seed: u32) -> Self {
        Self {
            chooser,
            epsilon: epsilon.clamp(0.0, 1.0),
            rng: Mulberry32::new(seed),
            log: Vec::new(),
            _pd: PhantomData,
        }
    }

    /// Read-only access to the serving chooser (for `evaluate` / `save`).
    pub fn chooser(&self) -> &Chooser<D> {
        &self.chooser
    }

    /// Steer the underlying chooser (forbid/pin/bias) — honored by both the
    /// exploit path and the exploration feasibility set.
    pub fn set_config(&mut self, config: Config) {
        self.chooser.set_config(config);
    }

    pub fn is_trained(&self) -> bool {
        self.chooser.is_trained()
    }

    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }

    /// The A/B decision for one call: which impl to serve. Explores (uniform
    /// over feasible impls) when untrained or with probability `epsilon`;
    /// otherwise exploits the chooser. Does **not** run anything or log.
    pub fn choose(&mut self, input: &D::Input) -> usize {
        let cfg = self.chooser.config();
        let feasible: Vec<usize> = (0..D::impl_count())
            .filter(|&i| D::applicable(i, input) && !cfg.is_forbidden(i))
            .collect();
        if feasible.is_empty() {
            return 0;
        }
        let explore = !self.chooser.is_trained() || self.rng.next_f64() < self.epsilon;
        if explore {
            let j = (self.rng.next_f64() * feasible.len() as f64).floor() as usize;
            feasible[j.min(feasible.len() - 1)]
        } else {
            self.chooser.select(input)
        }
    }

    /// Log one production outcome: the consumer served `input` with `impl_idx`
    /// and measured `cost`. Stored as a one-hot [`RawRow`] (only this impl's
    /// column filled), the same shape Mode-1 capture produces.
    pub fn record(&mut self, input: &D::Input, impl_idx: usize, cost: f64) {
        let k = D::impl_count();
        let mut costs = vec![0.0f64; k];
        let mut mask = vec![0.0f32; k];
        if impl_idx < k {
            costs[impl_idx] = cost;
            mask[impl_idx] = 1.0;
        }
        self.log.push(RawRow {
            features: D::features(input),
            costs,
            mask,
        });
    }

    /// Convenience for simulations/demos: choose an impl, run it, measure its
    /// wall-clock cost, log the outcome, and return the output. In real
    /// production the consumer runs its own workload and calls
    /// [`OnlineAb::choose`] + [`OnlineAb::record`] instead.
    pub fn dispatch(&mut self, input: &D::Input) -> D::Output {
        let idx = self.choose(input);
        let t0 = Instant::now();
        let out = D::run(idx, input);
        let cost = t0.elapsed().as_secs_f64() * 1000.0;
        self.record(input, idx, cost);
        out
    }

    /// Number of outcomes logged so far.
    pub fn observations(&self) -> usize {
        self.log.len()
    }

    pub fn is_empty(&self) -> bool {
        self.log.is_empty()
    }

    /// The accumulated partial-feedback log.
    pub fn rows(&self) -> &[RawRow] {
        &self.log
    }

    /// Drain and return the log (e.g. to ship it elsewhere for offline training).
    pub fn take_log(&mut self) -> Vec<RawRow> {
        std::mem::take(&mut self.log)
    }

    /// Retrain the *serving* chooser in place from everything logged so far,
    /// then keep serving with the fresher model. This is the online arm; the
    /// offline arm is [`retrain_from_log`] over an exported file.
    pub fn retrain(&mut self, cfg: &TrainConfig) -> Result<TrainOutcome> {
        self.chooser.train_rows(&self.log, cfg)
    }

    /// Append the log to `path` as JSONL (one [`RawRow`] per line), creating or
    /// extending the file so a long-running service can checkpoint incrementally.
    pub fn export_log(&self, path: &str) -> Result<()> {
        append_rows_jsonl(path, &self.log)
    }
}

/// Read a JSONL log written by [`OnlineAb::export_log`] back into rows.
pub fn read_log_jsonl(path: &str) -> Result<Vec<RawRow>> {
    let text = std::fs::read_to_string(path).map_err(|e| Error::Msg(format!("log read: {e}")))?;
    let mut rows = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let row: RawRow = serde_json::from_str(line)
            .map_err(|e| Error::Msg(format!("log parse (line {}): {e}", i + 1)))?;
        rows.push(row);
    }
    Ok(rows)
}

/// The offline arm: read an exported log and fit a fresh [`Chooser<D>`] from it.
/// The caller then `save`s the result to new safetensors.
pub fn retrain_from_log<D: Domain>(path: &str, cfg: &TrainConfig) -> Result<Chooser<D>> {
    let rows = read_log_jsonl(path)?;
    if rows.is_empty() {
        return Err(Error::Msg("retrain_from_log: log is empty".into()));
    }
    let mut chooser = Chooser::<D>::new();
    chooser.train_rows(&rows, cfg)?;
    Ok(chooser)
}

fn append_rows_jsonl(path: &str, rows: &[RawRow]) -> Result<()> {
    use std::io::Write;
    let mut buf = String::new();
    for r in rows {
        let line = serde_json::to_string(r).map_err(|e| Error::Msg(format!("log encode: {e}")))?;
        buf.push_str(&line);
        buf.push('\n');
    }
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| Error::Msg(format!("log open: {e}")))?;
    file.write_all(buf.as_bytes())
        .map_err(|e| Error::Msg(format!("log write: {e}")))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sort::{Sort, build_sort_scenario};

    fn corpus(count: usize, base_seed: u32) -> Vec<Vec<i32>> {
        let mut out = Vec::new();
        for i in 0..count {
            let seed = base_seed.wrapping_add(i as u32);
            let (n, sortedness, mult) = match i % 4 {
                0 => (16 + (i % 48), 0.2, 4.0),
                1 => (2000 + (i % 2000), 0.9, 8.0),
                2 => (4000 + (i % 4000), 0.1, 64.0),
                _ => (8000 + (i % 8000), 0.0, 0.2),
            };
            out.push(build_sort_scenario(n, sortedness, mult, seed));
        }
        out
    }

    #[test]
    fn online_ab_learns_from_partial_feedback() {
        // Simulate production traffic: dispatch each workload with A/B switching,
        // capturing one measured cost per call. Then retrain from those
        // single-observation rows and confirm the recovered chooser beats the
        // best fixed sort — i.e. bandit-style partial feedback is enough.
        let mut online: OnlineAb<Sort> = OnlineAb::new(0.2, 12345);
        for input in corpus(400, 1) {
            let _ = online.dispatch(&input);
        }
        // Every logged row is one-hot: exactly one impl observed per call.
        assert_eq!(online.observations(), 400);
        for row in online.rows() {
            let observed = row.mask.iter().filter(|&&m| m > 0.0).count();
            assert_eq!(observed, 1, "Mode-2 rows must be single-observation");
        }

        let cfg = TrainConfig {
            steps: 800,
            ..Default::default()
        };
        online.retrain(&cfg).unwrap();
        assert!(online.is_trained());

        let report = online.chooser().evaluate(&corpus(120, 9999));
        assert!(
            report.learned_mean < report.best_fixed_mean,
            "Mode-2 learned {:.4} should beat best-fixed {:.4}",
            report.learned_mean,
            report.best_fixed_mean
        );
    }

    #[test]
    fn log_jsonl_round_trips_and_retrains_offline() {
        let mut online: OnlineAb<Sort> = OnlineAb::new(1.0, 7); // pure exploration
        for input in corpus(200, 3) {
            let _ = online.dispatch(&input);
        }
        let dir = std::env::temp_dir();
        let path = dir
            .join("optigon-online-test-log.jsonl")
            .to_string_lossy()
            .into_owned();
        let _ = std::fs::remove_file(&path);
        online.export_log(&path).unwrap();

        let rows = read_log_jsonl(&path).unwrap();
        assert_eq!(rows.len(), online.observations());

        // Offline: rebuild a fresh chooser straight from the log file.
        let cfg = TrainConfig {
            steps: 400,
            ..Default::default()
        };
        let chooser = retrain_from_log::<Sort>(&path, &cfg).unwrap();
        assert!(chooser.is_trained());
        let _ = std::fs::remove_file(&path);
    }
}
