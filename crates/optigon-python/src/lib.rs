//! Optigon's Python extension — a thin PyO3 wrapper over `optigon-core`.
//!
//! Like the napi addon, it invents no logic: every method forwards to a
//! `Chooser<Sort>` / `Recorder<Sort>` in the core, and every entry point is
//! wrapped in `catch_unwind` (with `panic = "unwind"` in the release profile) so
//! a core panic becomes a Python exception rather than aborting the interpreter.

use std::panic::{AssertUnwindSafe, catch_unwind};

use optigon_core::{
    Chooser, Config, Domain, OnlineAb, Recorder, TrainConfig,
    dict::{Dict, DictInput},
    read_log_jsonl,
    sort::Sort,
};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// Run `f`, converting a panic into a Python exception instead of unwinding
/// across the FFI boundary.
fn guard<T>(f: impl FnOnce() -> PyResult<T>) -> PyResult<T> {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(r) => r,
        Err(_) => Err(PyRuntimeError::new_err("optigon: internal panic")),
    }
}

fn to_err<E: std::fmt::Display>(e: E) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

/// Outcome of a training run.
#[pyclass]
#[derive(Clone)]
pub struct TrainResult {
    #[pyo3(get)]
    pub steps: u32,
    #[pyo3(get)]
    pub initial_loss: f64,
    #[pyo3(get)]
    pub final_loss: f64,
}

#[pymethods]
impl TrainResult {
    fn __repr__(&self) -> String {
        format!(
            "TrainResult(steps={}, initial_loss={:.4}, final_loss={:.4})",
            self.steps, self.initial_loss, self.final_loss
        )
    }
}

/// How a trained chooser scores vs the oracle and the best fixed impl.
#[pyclass]
#[derive(Clone)]
pub struct RegretReport {
    #[pyo3(get)]
    pub learned_mean: f64,
    #[pyo3(get)]
    pub optimal_rate: f64,
    #[pyo3(get)]
    pub best_fixed_impl: u32,
    #[pyo3(get)]
    pub best_fixed_impl_name: String,
    #[pyo3(get)]
    pub best_fixed_mean: f64,
    #[pyo3(get)]
    pub n: u32,
}

#[pymethods]
impl RegretReport {
    fn __repr__(&self) -> String {
        format!(
            "RegretReport(learned_mean={:.4}, optimal_rate={:.3}, best_fixed_impl_name={:?}, best_fixed_mean={:.4}, n={})",
            self.learned_mean,
            self.optimal_rate,
            self.best_fixed_impl_name,
            self.best_fixed_mean,
            self.n
        )
    }
}

/// The sort domain's implementation ids, in fixed order.
#[pyfunction]
fn sort_impl_names() -> Vec<String> {
    Sort::impl_names().iter().map(|s| s.to_string()).collect()
}

/// A learned sort-algorithm dispatcher: observe workloads (Mode-1 capture),
/// train, then let it pick the fastest sort per input.
#[pyclass]
pub struct SortChooser {
    recorder: Recorder<Sort>,
    chooser: Chooser<Sort>,
}

#[pymethods]
impl SortChooser {
    #[new]
    fn new() -> Self {
        Self {
            recorder: Recorder::new(),
            chooser: Chooser::new(),
        }
    }

    /// Load a chooser previously saved with `save`.
    #[staticmethod]
    fn load(prefix: String) -> PyResult<Self> {
        guard(|| {
            Ok(Self {
                recorder: Recorder::new(),
                chooser: Chooser::<Sort>::load(&prefix).map_err(to_err)?,
            })
        })
    }

    /// Observe a workload (Mode-1 capture): measures every sort on `keys`.
    fn observe(&mut self, keys: Vec<i32>) -> PyResult<()> {
        guard(|| {
            self.recorder.observe(&keys);
            Ok(())
        })
    }

    /// Number of workloads observed so far.
    fn observed(&self) -> u32 {
        self.recorder.len() as u32
    }

    /// Train (or retrain) on everything observed. `steps` overrides the count.
    #[pyo3(signature = (steps=None))]
    fn train(&mut self, steps: Option<u32>) -> PyResult<TrainResult> {
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
    fn is_trained(&self) -> bool {
        self.chooser.is_trained()
    }

    /// The index of the impl the chooser would pick for `keys`.
    fn select(&self, keys: Vec<i32>) -> PyResult<u32> {
        guard(|| Ok(self.chooser.select(&keys) as u32))
    }

    /// The name of the impl the chooser would pick for `keys`.
    fn selected_name(&self, keys: Vec<i32>) -> PyResult<String> {
        guard(|| Ok(Sort::impl_names()[self.chooser.select(&keys)].to_string()))
    }

    /// Pick the fastest sort for `keys` and run it, returning the sorted list.
    fn sort(&self, keys: Vec<i32>) -> PyResult<Vec<i32>> {
        guard(|| Ok(self.chooser.run(&keys)))
    }

    /// Score the chooser on a set of workloads.
    fn evaluate(&self, inputs: Vec<Vec<i32>>) -> PyResult<RegretReport> {
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

    /// Pin a specific impl (by index) and bypass the model, or clear with `None`.
    #[pyo3(signature = (impl_idx=None))]
    fn pin(&mut self, impl_idx: Option<u32>) {
        let mut c = self.chooser.config().clone();
        c.force = impl_idx.map(|i| i as usize);
        self.chooser.set_config(c);
    }

    /// Forbid (or re-allow) an impl by index — it is masked out of selection.
    fn forbid(&mut self, impl_idx: u32, forbidden: bool) {
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
    fn clear_steering(&mut self) {
        self.chooser.set_config(Config::default());
    }

    /// Persist to `<prefix>.safetensors` + `<prefix>.meta.json`.
    fn save(&self, prefix: String) -> PyResult<()> {
        guard(|| self.chooser.save(&prefix).map_err(to_err))
    }
}

/// The dict domain's implementation ids, in fixed order.
#[pyfunction]
fn dict_impl_names() -> Vec<String> {
    Dict::impl_names().iter().map(|s| s.to_string()).collect()
}

/// A learned dictionary-lookup dispatcher: observe (keys, queries) workloads
/// (Mode-1 capture), train, then let it pick the fastest lookup structure per
/// workload. `direct` (a direct-address table) is applicable only for a bounded
/// key range; the chooser masks it out otherwise.
#[pyclass]
pub struct DictChooser {
    recorder: Recorder<Dict>,
    chooser: Chooser<Dict>,
}

#[pymethods]
impl DictChooser {
    #[new]
    fn new() -> Self {
        Self {
            recorder: Recorder::new(),
            chooser: Chooser::new(),
        }
    }

    /// Load a chooser previously saved with `save`.
    #[staticmethod]
    fn load(prefix: String) -> PyResult<Self> {
        guard(|| {
            Ok(Self {
                recorder: Recorder::new(),
                chooser: Chooser::<Dict>::load(&prefix).map_err(to_err)?,
            })
        })
    }

    /// Observe a workload (Mode-1 capture): measures every applicable lookup on
    /// `(keys, queries)`.
    fn observe(&mut self, keys: Vec<i32>, queries: Vec<i32>) -> PyResult<()> {
        guard(|| {
            self.recorder.observe(&DictInput { keys, queries });
            Ok(())
        })
    }

    /// Number of workloads observed so far.
    fn observed(&self) -> u32 {
        self.recorder.len() as u32
    }

    /// Train (or retrain) on everything observed. `steps` overrides the count.
    #[pyo3(signature = (steps=None))]
    fn train(&mut self, steps: Option<u32>) -> PyResult<TrainResult> {
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
    fn is_trained(&self) -> bool {
        self.chooser.is_trained()
    }

    /// The index of the impl the chooser would pick for `(keys, queries)`.
    fn select(&self, keys: Vec<i32>, queries: Vec<i32>) -> PyResult<u32> {
        guard(|| Ok(self.chooser.select(&DictInput { keys, queries }) as u32))
    }

    /// The name of the impl the chooser would pick for `(keys, queries)`.
    fn selected_name(&self, keys: Vec<i32>, queries: Vec<i32>) -> PyResult<String> {
        guard(|| {
            let idx = self.chooser.select(&DictInput { keys, queries });
            Ok(Dict::impl_names()[idx].to_string())
        })
    }

    /// Pick the fastest lookup for `(keys, queries)` and run it, returning the
    /// per-query results (the stored value, or `-1` on a miss).
    fn lookup(&self, keys: Vec<i32>, queries: Vec<i32>) -> PyResult<Vec<i32>> {
        guard(|| Ok(self.chooser.run(&DictInput { keys, queries })))
    }

    /// Score the chooser on a set of workloads (parallel `keys` / `queries` lists).
    fn evaluate(
        &self,
        keys_list: Vec<Vec<i32>>,
        queries_list: Vec<Vec<i32>>,
    ) -> PyResult<RegretReport> {
        guard(|| {
            if keys_list.len() != queries_list.len() {
                return Err(PyRuntimeError::new_err(
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

    /// Pin a specific impl (by index) and bypass the model, or clear with `None`.
    #[pyo3(signature = (impl_idx=None))]
    fn pin(&mut self, impl_idx: Option<u32>) {
        let mut c = self.chooser.config().clone();
        c.force = impl_idx.map(|i| i as usize);
        self.chooser.set_config(c);
    }

    /// Forbid (or re-allow) an impl by index — it is masked out of selection.
    fn forbid(&mut self, impl_idx: u32, forbidden: bool) {
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
    fn clear_steering(&mut self) {
        self.chooser.set_config(Config::default());
    }

    /// Persist to `<prefix>.safetensors` + `<prefix>.meta.json`.
    fn save(&self, prefix: String) -> PyResult<()> {
        guard(|| self.chooser.save(&prefix).map_err(to_err))
    }
}

/// Mode-2 (production A/B) online capture for the sort domain: switch impls per
/// call, log each measured outcome as a single-observation row, and retrain from
/// that bandit-style log. `choose` + `record` is the online protocol; `dispatch`
/// fuses them for simulations.
#[pyclass]
pub struct SortOnline {
    inner: OnlineAb<Sort>,
}

#[pymethods]
impl SortOnline {
    /// A fresh dispatcher. `epsilon` is the exploration rate (default 0.1); until
    /// retrained it explores uniformly over the applicable sorts.
    #[new]
    #[pyo3(signature = (epsilon=0.1, seed=0))]
    fn new(epsilon: f64, seed: u32) -> Self {
        Self {
            inner: OnlineAb::new(epsilon, seed),
        }
    }

    /// Warm-start online serving from a chooser saved with `SortChooser.save`.
    #[staticmethod]
    #[pyo3(signature = (prefix, epsilon=0.1, seed=0))]
    fn with_model(prefix: String, epsilon: f64, seed: u32) -> PyResult<Self> {
        guard(|| {
            let chooser = Chooser::<Sort>::load(&prefix).map_err(to_err)?;
            Ok(Self {
                inner: OnlineAb::from_chooser(chooser, epsilon, seed),
            })
        })
    }

    /// The A/B decision for one call: the index of the sort to serve `keys` with
    /// (explore or exploit). Does not run or log anything.
    fn choose(&mut self, keys: Vec<i32>) -> PyResult<u32> {
        guard(|| Ok(self.inner.choose(&keys) as u32))
    }

    /// Log one production outcome: `keys` was served with `impl_idx` at a
    /// measured `cost_ms`.
    fn record(&mut self, keys: Vec<i32>, impl_idx: u32, cost_ms: f64) -> PyResult<()> {
        guard(|| {
            self.inner.record(&keys, impl_idx as usize, cost_ms);
            Ok(())
        })
    }

    /// Convenience: choose a sort, run it, measure it, log the outcome, and
    /// return the sorted list — for sims where Optigon runs the impl itself.
    fn dispatch(&mut self, keys: Vec<i32>) -> PyResult<Vec<i32>> {
        guard(|| Ok(self.inner.dispatch(&keys)))
    }

    /// Number of outcomes logged so far.
    fn observations(&self) -> u32 {
        self.inner.observations() as u32
    }

    /// Whether the serving chooser has been trained yet.
    fn is_trained(&self) -> bool {
        self.inner.is_trained()
    }

    /// Retrain the serving chooser in place from everything logged so far.
    #[pyo3(signature = (steps=None))]
    fn retrain(&mut self, steps: Option<u32>) -> PyResult<TrainResult> {
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
    fn selected_name(&self, keys: Vec<i32>) -> PyResult<String> {
        guard(|| Ok(Sort::impl_names()[self.inner.chooser().select(&keys)].to_string()))
    }

    /// Score the serving chooser on a set of workloads.
    fn evaluate(&self, inputs: Vec<Vec<i32>>) -> PyResult<RegretReport> {
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
    fn export_log(&self, path: String) -> PyResult<()> {
        guard(|| self.inner.export_log(&path).map_err(to_err))
    }

    /// Persist the serving chooser to `<prefix>.safetensors` + `.meta.json`.
    fn save(&self, prefix: String) -> PyResult<()> {
        guard(|| self.inner.chooser().save(&prefix).map_err(to_err))
    }
}

/// The offline arm of Mode 2: read a JSONL capture log, fit a fresh sort chooser
/// from it, and save it to `<out_prefix>.safetensors` + `.meta.json`. Returns
/// the training outcome.
#[pyfunction]
#[pyo3(signature = (log_path, out_prefix, steps=None))]
fn retrain_sort_from_log(
    log_path: String,
    out_prefix: String,
    steps: Option<u32>,
) -> PyResult<TrainResult> {
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

/// The `optigon` extension module.
#[pymodule]
fn optigon(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SortChooser>()?;
    m.add_class::<DictChooser>()?;
    m.add_class::<SortOnline>()?;
    m.add_class::<TrainResult>()?;
    m.add_class::<RegretReport>()?;
    m.add_function(wrap_pyfunction!(sort_impl_names, m)?)?;
    m.add_function(wrap_pyfunction!(dict_impl_names, m)?)?;
    m.add_function(wrap_pyfunction!(retrain_sort_from_log, m)?)?;
    Ok(())
}
