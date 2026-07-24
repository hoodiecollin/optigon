//! Optigon's Node/Bun addon — a thin napi-rs wrapper over `optigon-core`.
//!
//! It invents no logic: every method forwards to a `Chooser<Sort>` /
//! `Recorder<Sort>` in the core. One `.node` serves both Node and Bun. Every
//! entry point is wrapped in `catch_unwind` (paired with `panic = "unwind"` in
//! the workspace release profile) so a core panic becomes a thrown JS `Error`
//! rather than aborting the runtime.

use std::panic::{AssertUnwindSafe, catch_unwind};

use napi::bindgen_prelude::*;
use napi_derive::napi;
use optigon_core::{
    Chooser, Config, Domain, OnlineAb, Recorder, TrainConfig,
    dict::{Dict, DictInput},
    read_log_jsonl,
    sort::Sort,
};

/// Run `f`, converting a panic into a JS error instead of unwinding across FFI.
fn guard<T>(f: impl FnOnce() -> Result<T>) -> Result<T> {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(r) => r,
        Err(_) => Err(Error::from_reason("optigon: internal panic")),
    }
}

fn to_err<E: std::fmt::Display>(e: E) -> Error {
    Error::from_reason(e.to_string())
}

/// Outcome of a training run.
#[napi(object)]
pub struct TrainResult {
    pub steps: u32,
    pub initial_loss: f64,
    pub final_loss: f64,
}

/// How a trained chooser scores vs the oracle and the best fixed impl.
#[napi(object)]
pub struct RegretReport {
    pub learned_mean: f64,
    pub optimal_rate: f64,
    pub best_fixed_impl: u32,
    pub best_fixed_impl_name: String,
    pub best_fixed_mean: f64,
    pub n: u32,
}

/// The sort domain's implementation ids, in fixed order.
#[napi]
pub fn sort_impl_names() -> Vec<String> {
    Sort::impl_names().iter().map(|s| s.to_string()).collect()
}

/// A learned sort-algorithm dispatcher: observe workloads (Mode-1 capture),
/// train, then let it pick the fastest sort per input.
#[napi(js_name = "SortChooser")]
pub struct SortChooser {
    recorder: Recorder<Sort>,
    chooser: Chooser<Sort>,
}

#[napi]
impl SortChooser {
    #[napi(constructor)]
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self {
            recorder: Recorder::new(),
            chooser: Chooser::new(),
        }
    }

    /// Load a chooser previously saved with `save`.
    #[napi(factory)]
    pub fn load(prefix: String) -> Result<Self> {
        guard(|| {
            Ok(Self {
                recorder: Recorder::new(),
                chooser: Chooser::<Sort>::load(&prefix).map_err(to_err)?,
            })
        })
    }

    /// Observe a workload (Mode-1 capture): measures every sort on `keys` and
    /// records the cost row training will learn from.
    #[napi]
    pub fn observe(&mut self, keys: Vec<i32>) -> Result<()> {
        guard(|| {
            self.recorder.observe(&keys);
            Ok(())
        })
    }

    /// Number of workloads observed so far.
    #[napi]
    pub fn observed(&self) -> u32 {
        self.recorder.len() as u32
    }

    /// Train (or retrain) the chooser on everything observed. `steps` overrides
    /// the training step count.
    #[napi]
    pub fn train(&mut self, steps: Option<u32>) -> Result<TrainResult> {
        guard(|| {
            let cfg = TrainConfig {
                steps: steps.unwrap_or(1500) as usize,
                ..Default::default()
            };
            let out = self.chooser.train(&self.recorder, &cfg).map_err(to_err)?;
            Ok(TrainResult {
                steps: out.steps_run as u32,
                initial_loss: out.initial_loss as f64,
                final_loss: out.final_loss as f64,
            })
        })
    }

    /// Whether the chooser has been trained.
    #[napi]
    pub fn is_trained(&self) -> bool {
        self.chooser.is_trained()
    }

    /// The index of the impl the chooser would pick for `keys`.
    #[napi]
    pub fn select(&self, keys: Vec<i32>) -> Result<u32> {
        guard(|| Ok(self.chooser.select(&keys) as u32))
    }

    /// The name of the impl the chooser would pick for `keys`.
    #[napi]
    pub fn selected_name(&self, keys: Vec<i32>) -> Result<String> {
        guard(|| Ok(Sort::impl_names()[self.chooser.select(&keys)].to_string()))
    }

    /// Pick the fastest sort for `keys` and run it, returning the sorted array.
    #[napi]
    pub fn sort(&self, keys: Vec<i32>) -> Result<Vec<i32>> {
        guard(|| Ok(self.chooser.run(&keys)))
    }

    /// Score the chooser on a set of workloads.
    #[napi]
    pub fn evaluate(&self, inputs: Vec<Vec<i32>>) -> Result<RegretReport> {
        guard(|| {
            let r = self.chooser.evaluate(&inputs);
            Ok(RegretReport {
                learned_mean: r.learned_mean,
                optimal_rate: r.optimal_rate,
                best_fixed_impl: r.best_fixed_impl as u32,
                best_fixed_impl_name: Sort::impl_names()[r.best_fixed_impl].to_string(),
                best_fixed_mean: r.best_fixed_mean,
                n: r.n as u32,
            })
        })
    }

    /// Pin a specific impl (by index) and bypass the model, or clear the pin
    /// with `null`/`undefined`.
    #[napi]
    pub fn pin(&mut self, impl_idx: Option<u32>) {
        let mut c = self.chooser.config().clone();
        c.force = impl_idx.map(|i| i as usize);
        self.chooser.set_config(c);
    }

    /// Forbid (or re-allow) an impl by index — it is masked out of selection.
    #[napi]
    pub fn forbid(&mut self, impl_idx: u32, forbidden: bool) {
        let mut c = self.chooser.config().clone();
        if c.forbidden.len() < Sort::impl_count() {
            c.forbidden = vec![false; Sort::impl_count()];
        }
        if let Some(x) = c.forbidden.get_mut(impl_idx as usize) {
            *x = forbidden;
        }
        self.chooser.set_config(c);
    }

    /// Clear all steering (pin, forbid, bias).
    #[napi]
    pub fn clear_steering(&mut self) {
        self.chooser.set_config(Config::default());
    }

    /// Persist to `<prefix>.safetensors` + `<prefix>.meta.json`.
    #[napi]
    pub fn save(&self, prefix: String) -> Result<()> {
        guard(|| self.chooser.save(&prefix).map_err(to_err))
    }
}

/// The dict domain's implementation ids, in fixed order.
#[napi]
pub fn dict_impl_names() -> Vec<String> {
    Dict::impl_names().iter().map(|s| s.to_string()).collect()
}

/// A learned dictionary-lookup dispatcher: observe (keys, queries) workloads
/// (Mode-1 capture), train, then let it pick the fastest lookup structure per
/// workload. `direct` (a direct-address table) is applicable only for a bounded
/// key range; the chooser masks it out otherwise.
#[napi(js_name = "DictChooser")]
pub struct DictChooser {
    recorder: Recorder<Dict>,
    chooser: Chooser<Dict>,
}

#[napi]
impl DictChooser {
    #[napi(constructor)]
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self {
            recorder: Recorder::new(),
            chooser: Chooser::new(),
        }
    }

    /// Load a chooser previously saved with `save`.
    #[napi(factory)]
    pub fn load(prefix: String) -> Result<Self> {
        guard(|| {
            Ok(Self {
                recorder: Recorder::new(),
                chooser: Chooser::<Dict>::load(&prefix).map_err(to_err)?,
            })
        })
    }

    /// Observe a workload (Mode-1 capture): measures every applicable lookup on
    /// `(keys, queries)` and records the cost row training will learn from.
    #[napi]
    pub fn observe(&mut self, keys: Vec<i32>, queries: Vec<i32>) -> Result<()> {
        guard(|| {
            self.recorder.observe(&DictInput { keys, queries });
            Ok(())
        })
    }

    /// Number of workloads observed so far.
    #[napi]
    pub fn observed(&self) -> u32 {
        self.recorder.len() as u32
    }

    /// Train (or retrain) the chooser on everything observed.
    #[napi]
    pub fn train(&mut self, steps: Option<u32>) -> Result<TrainResult> {
        guard(|| {
            let cfg = TrainConfig {
                steps: steps.unwrap_or(1500) as usize,
                ..Default::default()
            };
            let out = self.chooser.train(&self.recorder, &cfg).map_err(to_err)?;
            Ok(TrainResult {
                steps: out.steps_run as u32,
                initial_loss: out.initial_loss as f64,
                final_loss: out.final_loss as f64,
            })
        })
    }

    /// Whether the chooser has been trained.
    #[napi]
    pub fn is_trained(&self) -> bool {
        self.chooser.is_trained()
    }

    /// The index of the impl the chooser would pick for `(keys, queries)`.
    #[napi]
    pub fn select(&self, keys: Vec<i32>, queries: Vec<i32>) -> Result<u32> {
        guard(|| Ok(self.chooser.select(&DictInput { keys, queries }) as u32))
    }

    /// The name of the impl the chooser would pick for `(keys, queries)`.
    #[napi]
    pub fn selected_name(&self, keys: Vec<i32>, queries: Vec<i32>) -> Result<String> {
        guard(|| {
            let idx = self.chooser.select(&DictInput { keys, queries });
            Ok(Dict::impl_names()[idx].to_string())
        })
    }

    /// Pick the fastest lookup for `(keys, queries)` and run it, returning the
    /// per-query results (the stored value, or `-1` on a miss).
    #[napi]
    pub fn lookup(&self, keys: Vec<i32>, queries: Vec<i32>) -> Result<Vec<i32>> {
        guard(|| Ok(self.chooser.run(&DictInput { keys, queries })))
    }

    /// Score the chooser on a set of workloads (parallel `keys` / `queries` lists).
    #[napi]
    pub fn evaluate(
        &self,
        keys_list: Vec<Vec<i32>>,
        queries_list: Vec<Vec<i32>>,
    ) -> Result<RegretReport> {
        guard(|| {
            if keys_list.len() != queries_list.len() {
                return Err(Error::from_reason(
                    "evaluate: keys_list and queries_list must be the same length",
                ));
            }
            let inputs: Vec<DictInput> = keys_list
                .into_iter()
                .zip(queries_list)
                .map(|(keys, queries)| DictInput { keys, queries })
                .collect();
            let r = self.chooser.evaluate(&inputs);
            Ok(RegretReport {
                learned_mean: r.learned_mean,
                optimal_rate: r.optimal_rate,
                best_fixed_impl: r.best_fixed_impl as u32,
                best_fixed_impl_name: Dict::impl_names()[r.best_fixed_impl].to_string(),
                best_fixed_mean: r.best_fixed_mean,
                n: r.n as u32,
            })
        })
    }

    /// Pin a specific impl (by index) and bypass the model, or clear the pin.
    #[napi]
    pub fn pin(&mut self, impl_idx: Option<u32>) {
        let mut c = self.chooser.config().clone();
        c.force = impl_idx.map(|i| i as usize);
        self.chooser.set_config(c);
    }

    /// Forbid (or re-allow) an impl by index — it is masked out of selection.
    #[napi]
    pub fn forbid(&mut self, impl_idx: u32, forbidden: bool) {
        let mut c = self.chooser.config().clone();
        if c.forbidden.len() < Dict::impl_count() {
            c.forbidden = vec![false; Dict::impl_count()];
        }
        if let Some(x) = c.forbidden.get_mut(impl_idx as usize) {
            *x = forbidden;
        }
        self.chooser.set_config(c);
    }

    /// Clear all steering (pin, forbid, bias).
    #[napi]
    pub fn clear_steering(&mut self) {
        self.chooser.set_config(Config::default());
    }

    /// Persist to `<prefix>.safetensors` + `<prefix>.meta.json`.
    #[napi]
    pub fn save(&self, prefix: String) -> Result<()> {
        guard(|| self.chooser.save(&prefix).map_err(to_err))
    }
}

/// Mode-2 (production A/B) online capture for the sort domain: switch impls per
/// call, log each measured outcome as a single-observation row, and retrain from
/// that bandit-style log. `choose` + `record` is the online protocol; `dispatch`
/// fuses them for simulations.
#[napi(js_name = "SortOnline")]
pub struct SortOnline {
    inner: OnlineAb<Sort>,
}

#[napi]
impl SortOnline {
    /// A fresh dispatcher. `epsilon` is the exploration rate (default 0.1); until
    /// retrained it explores uniformly over the applicable sorts.
    #[napi(constructor)]
    #[allow(clippy::new_without_default)]
    pub fn new(epsilon: Option<f64>, seed: Option<u32>) -> Self {
        Self {
            inner: OnlineAb::new(epsilon.unwrap_or(0.1), seed.unwrap_or(0)),
        }
    }

    /// Warm-start online serving from a chooser saved with `SortChooser.save`.
    #[napi(factory)]
    pub fn with_model(prefix: String, epsilon: Option<f64>, seed: Option<u32>) -> Result<Self> {
        guard(|| {
            let chooser = Chooser::<Sort>::load(&prefix).map_err(to_err)?;
            Ok(Self {
                inner: OnlineAb::from_chooser(chooser, epsilon.unwrap_or(0.1), seed.unwrap_or(0)),
            })
        })
    }

    /// The A/B decision for one call: the index of the sort to serve `keys` with
    /// (explore or exploit). Does not run or log anything.
    #[napi]
    pub fn choose(&mut self, keys: Vec<i32>) -> Result<u32> {
        guard(|| Ok(self.inner.choose(&keys) as u32))
    }

    /// Log one production outcome: `keys` was served with `impl_idx` at a
    /// measured `cost_ms`.
    #[napi]
    pub fn record(&mut self, keys: Vec<i32>, impl_idx: u32, cost_ms: f64) -> Result<()> {
        guard(|| {
            self.inner.record(&keys, impl_idx as usize, cost_ms);
            Ok(())
        })
    }

    /// Convenience: choose a sort, run it, measure it, log the outcome, and
    /// return the sorted array — for sims where Optigon runs the impl itself.
    #[napi]
    pub fn dispatch(&mut self, keys: Vec<i32>) -> Result<Vec<i32>> {
        guard(|| Ok(self.inner.dispatch(&keys)))
    }

    /// Number of outcomes logged so far.
    #[napi]
    pub fn observations(&self) -> u32 {
        self.inner.observations() as u32
    }

    /// Whether the serving chooser has been trained yet.
    #[napi]
    pub fn is_trained(&self) -> bool {
        self.inner.is_trained()
    }

    /// Retrain the serving chooser in place from everything logged so far.
    #[napi]
    pub fn retrain(&mut self, steps: Option<u32>) -> Result<TrainResult> {
        guard(|| {
            let cfg = TrainConfig {
                steps: steps.unwrap_or(1500) as usize,
                ..Default::default()
            };
            let out = self.inner.retrain(&cfg).map_err(to_err)?;
            Ok(TrainResult {
                steps: out.steps_run as u32,
                initial_loss: out.initial_loss as f64,
                final_loss: out.final_loss as f64,
            })
        })
    }

    /// The sort the serving chooser would greedily pick for `keys` (no explore).
    #[napi]
    pub fn selected_name(&self, keys: Vec<i32>) -> Result<String> {
        guard(|| Ok(Sort::impl_names()[self.inner.chooser().select(&keys)].to_string()))
    }

    /// Score the serving chooser on a set of workloads.
    #[napi]
    pub fn evaluate(&self, inputs: Vec<Vec<i32>>) -> Result<RegretReport> {
        guard(|| {
            let r = self.inner.chooser().evaluate(&inputs);
            Ok(RegretReport {
                learned_mean: r.learned_mean,
                optimal_rate: r.optimal_rate,
                best_fixed_impl: r.best_fixed_impl as u32,
                best_fixed_impl_name: Sort::impl_names()[r.best_fixed_impl].to_string(),
                best_fixed_mean: r.best_fixed_mean,
                n: r.n as u32,
            })
        })
    }

    /// Append the capture log to `path` as JSONL (one row per line).
    #[napi]
    pub fn export_log(&self, path: String) -> Result<()> {
        guard(|| self.inner.export_log(&path).map_err(to_err))
    }

    /// Persist the serving chooser to `<prefix>.safetensors` + `.meta.json`.
    #[napi]
    pub fn save(&self, prefix: String) -> Result<()> {
        guard(|| self.inner.chooser().save(&prefix).map_err(to_err))
    }
}

/// The offline arm of Mode 2: read a JSONL capture log, fit a fresh sort chooser
/// from it, and save it to `<out_prefix>.safetensors` + `.meta.json`. Returns
/// the training outcome.
#[napi]
pub fn retrain_sort_from_log(
    log_path: String,
    out_prefix: String,
    steps: Option<u32>,
) -> Result<TrainResult> {
    guard(|| {
        let rows = read_log_jsonl(&log_path).map_err(to_err)?;
        let cfg = TrainConfig {
            steps: steps.unwrap_or(1500) as usize,
            ..Default::default()
        };
        let mut chooser = Chooser::<Sort>::new();
        let out = chooser.train_rows(&rows, &cfg).map_err(to_err)?;
        chooser.save(&out_prefix).map_err(to_err)?;
        Ok(TrainResult {
            steps: out.steps_run as u32,
            initial_loss: out.initial_loss as f64,
            final_loss: out.final_loss as f64,
        })
    })
}
