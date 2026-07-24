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
use optigon_core::{Chooser, Config, Domain, Recorder, TrainConfig, sort::Sort};

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
