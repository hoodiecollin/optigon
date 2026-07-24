//! The chooser — the generic glue that turns a [`Domain`] plus captured data
//! into a learned per-workload dispatcher.
//!
//! `Chooser<D>` owns a trained [`MlpModel`], the feature standardization stats,
//! and the steering [`Config`]. It is written entirely against the `Domain`
//! trait, so every packaged domain gets `train` / `select` / `run` / `evaluate`
//! for free. The napi and pyo3 bindings wrap a concrete `Chooser<Sort>` etc.

use std::marker::PhantomData;

use candle_core::{Error, Result};
use serde::{Deserialize, Serialize};

use crate::capture::{RawRow, Recorder, RegretReport};
use crate::config::Config;
use crate::domain::Domain;
use crate::model::{MlpModel, RegressionSample, TrainConfig, TrainOutcome};

/// Costs below this floor (e.g. a sub-microsecond measured time that rounded to
/// 0) are clamped before taking a log, so a `ln(0) = -inf` can't poison a target.
const COST_FLOOR: f64 = 1e-6;
/// Hidden width — capacity is not the bottleneck (see the research writeup).
const HIDDEN: usize = 32;

/// Persisted sidecar next to the safetensors weights: everything needed to run
/// the model that isn't a weight tensor.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct Meta {
    domain: String,
    impl_names: Vec<String>,
    feature_mean: Vec<f32>,
    feature_std: Vec<f32>,
    in_dim: usize,
    hidden: usize,
    out_dim: usize,
}

pub struct Chooser<D: Domain> {
    model: Option<MlpModel>,
    mean: Vec<f32>,
    std: Vec<f32>,
    config: Config,
    _pd: PhantomData<D>,
}

impl<D: Domain> Default for Chooser<D> {
    fn default() -> Self {
        Self {
            model: None,
            mean: vec![0.0; D::FEATURE_DIM],
            std: vec![1.0; D::FEATURE_DIM],
            config: Config::default(),
            _pd: PhantomData,
        }
    }
}

impl<D: Domain> Chooser<D> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_config(&mut self, config: Config) {
        self.config = config;
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn is_trained(&self) -> bool {
        self.model.is_some()
    }

    /// Train from everything a [`Recorder`] captured (Mode 1). Convenience over
    /// [`Chooser::train_rows`].
    pub fn train(&mut self, recorder: &Recorder<D>, cfg: &TrainConfig) -> Result<TrainOutcome> {
        self.train_rows(recorder.rows(), cfg)
    }

    /// Train from raw captured rows: standardize features, log the costs into
    /// per-impl targets (masking inapplicable impls), and fit the MLP.
    pub fn train_rows(&mut self, rows: &[RawRow], cfg: &TrainConfig) -> Result<TrainOutcome> {
        if rows.is_empty() {
            return Err(Error::Msg("train: no captured rows".into()));
        }
        let in_dim = D::FEATURE_DIM;
        let out_dim = D::impl_count();
        let (mean, std) = feature_stats(rows, in_dim);

        let samples: Vec<RegressionSample> = rows
            .iter()
            .map(|r| {
                let features = standardize(&r.features, &mean, &std);
                let targets = r
                    .costs
                    .iter()
                    .zip(&r.mask)
                    .map(|(&c, &m)| {
                        if m > 0.0 {
                            c.max(COST_FLOOR).ln() as f32
                        } else {
                            0.0
                        }
                    })
                    .collect();
                RegressionSample {
                    features,
                    targets,
                    mask: r.mask.clone(),
                }
            })
            .collect();

        let model = MlpModel::new(in_dim, HIDDEN, out_dim, cfg.seed)?;
        let outcome = model.train(&samples, cfg)?;
        self.model = Some(model);
        self.mean = mean;
        self.std = std;
        Ok(outcome)
    }

    /// Predicted per-impl log-cost for a workload (bias applied, `None` if untrained).
    fn predict(&self, input: &D::Input) -> Option<Vec<f32>> {
        let model = self.model.as_ref()?;
        let std = standardize(&D::features(input), &self.mean, &self.std);
        let mut p = model.predict_row(&std).ok()?;
        for (i, v) in p.iter_mut().enumerate() {
            *v += self.config.bias_for(i);
        }
        Some(p)
    }

    /// Choose an impl for `input`: the argmin of predicted (biased) cost over
    /// applicable, non-forbidden impls. Falls back to the first feasible impl
    /// when untrained. Honors a forced impl if it is feasible.
    pub fn select(&self, input: &D::Input) -> usize {
        let feasible = |i: usize| D::applicable(i, input) && !self.config.is_forbidden(i);
        if let Some(f) = self.config.force
            && feasible(f)
        {
            return f;
        }
        let scores = self.predict(input);
        let mut best: Option<(usize, f32)> = None;
        for i in 0..D::impl_count() {
            if !feasible(i) {
                continue;
            }
            let s = scores.as_ref().map(|p| p[i]).unwrap_or(0.0);
            if best.is_none() || s < best.unwrap().1 {
                best = Some((i, s));
            }
        }
        best.map(|(i, _)| i).unwrap_or(0)
    }

    /// Select an impl and run it — the primary consumer entry point.
    pub fn run(&self, input: &D::Input) -> D::Output {
        D::run(self.select(input), input)
    }

    /// Score this chooser on a set of workloads: mean regret, optimal-pick rate,
    /// and the best-fixed-impl bar it must beat.
    pub fn evaluate(&self, inputs: &[D::Input]) -> RegretReport {
        let n = inputs.len();
        let k = D::impl_count();
        let mut learned_sum = 0.0;
        let mut optimal = 0usize;
        // Per-impl regret sums + counts for the best-fixed baseline.
        let mut fixed_sum = vec![0.0f64; k];
        let mut fixed_cnt = vec![0usize; k];

        for input in inputs {
            let costs: Vec<Option<f64>> = (0..k)
                .map(|i| D::applicable(i, input).then(|| D::cost(i, input)))
                .collect();
            let oracle = costs
                .iter()
                .filter_map(|c| *c)
                .fold(f64::INFINITY, f64::min);
            if !oracle.is_finite() {
                continue;
            }
            let pick = self.select(input);
            let chosen = costs[pick].unwrap_or(oracle);
            learned_sum += (chosen - oracle) / oracle;
            if (chosen - oracle).abs() <= 1e-12 {
                optimal += 1;
            }
            for i in 0..k {
                if let Some(c) = costs[i] {
                    fixed_sum[i] += (c - oracle) / oracle;
                    fixed_cnt[i] += 1;
                }
            }
        }

        // Best fixed impl = lowest mean regret among impls applicable everywhere.
        let mut best_fixed_impl = 0;
        let mut best_fixed_mean = f64::INFINITY;
        for i in 0..k {
            if fixed_cnt[i] == n && n > 0 {
                let m = fixed_sum[i] / n as f64;
                if m < best_fixed_mean {
                    best_fixed_mean = m;
                    best_fixed_impl = i;
                }
            }
        }

        RegretReport {
            learned_mean: if n > 0 { learned_sum / n as f64 } else { 0.0 },
            optimal_rate: if n > 0 {
                optimal as f64 / n as f64
            } else {
                0.0
            },
            best_fixed_impl,
            best_fixed_mean,
            n,
        }
    }

    /// Persist to `<prefix>.safetensors` (+ `<prefix>.meta.json`).
    pub fn save(&self, prefix: &str) -> Result<()> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| Error::Msg("save: chooser is not trained".into()))?;
        model.save(&format!("{prefix}.safetensors"))?;
        let meta = Meta {
            domain: D::NAME.to_string(),
            impl_names: D::impl_names().iter().map(|s| s.to_string()).collect(),
            feature_mean: self.mean.clone(),
            feature_std: self.std.clone(),
            in_dim: model.in_dim,
            hidden: model.hidden,
            out_dim: model.out_dim,
        };
        let json = serde_json::to_string_pretty(&meta)
            .map_err(|e| Error::Msg(format!("meta serialize: {e}")))?;
        std::fs::write(format!("{prefix}.meta.json"), json)
            .map_err(|e| Error::Msg(format!("meta write: {e}")))?;
        Ok(())
    }

    /// Load a chooser saved by [`Chooser::save`].
    pub fn load(prefix: &str) -> Result<Self> {
        let json = std::fs::read_to_string(format!("{prefix}.meta.json"))
            .map_err(|e| Error::Msg(format!("meta read: {e}")))?;
        let meta: Meta =
            serde_json::from_str(&json).map_err(|e| Error::Msg(format!("meta parse: {e}")))?;
        if meta.domain != D::NAME {
            return Err(Error::Msg(format!(
                "model domain {:?} != {:?}",
                meta.domain,
                D::NAME
            )));
        }
        let model = MlpModel::load(&format!("{prefix}.safetensors"))?;
        Ok(Self {
            model: Some(model),
            mean: meta.feature_mean,
            std: meta.feature_std,
            config: Config::default(),
            _pd: PhantomData,
        })
    }
}

/// Per-dimension mean/std over captured features (std floored to 1 where a
/// feature is constant, so standardization can't divide by zero).
fn feature_stats(rows: &[RawRow], in_dim: usize) -> (Vec<f32>, Vec<f32>) {
    let n = rows.len() as f32;
    let mut mean = vec![0.0f32; in_dim];
    for r in rows {
        for (j, &v) in r.features.iter().enumerate() {
            mean[j] += v;
        }
    }
    for m in &mut mean {
        *m /= n;
    }
    let mut var = vec![0.0f32; in_dim];
    for r in rows {
        for (j, &v) in r.features.iter().enumerate() {
            let d = v - mean[j];
            var[j] += d * d;
        }
    }
    let std = var
        .iter()
        .map(|&s| {
            let sd = (s / n).sqrt();
            if sd > 1e-8 { sd } else { 1.0 }
        })
        .collect();
    (mean, std)
}

fn standardize(features: &[f32], mean: &[f32], std: &[f32]) -> Vec<f32> {
    features
        .iter()
        .enumerate()
        .map(|(j, &v)| (v - mean[j]) / std[j])
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sort::{Sort, build_sort_scenario};

    fn corpus(count: usize, base_seed: u32) -> Vec<Vec<i32>> {
        // Diverse workloads spanning the regimes each sort wins.
        let mut out = Vec::new();
        for i in 0..count {
            let seed = base_seed.wrapping_add(i as u32);
            let (n, sortedness, mult) = match i % 4 {
                0 => (16 + (i % 48), 0.2, 4.0),      // tiny
                1 => (2000 + (i % 2000), 0.9, 8.0),  // large nearly-sorted
                2 => (4000 + (i % 4000), 0.1, 64.0), // wide random
                _ => (8000 + (i % 8000), 0.0, 0.2),  // large narrow-key (radix)
            };
            out.push(build_sort_scenario(n, sortedness, mult, seed));
        }
        out
    }

    #[test]
    fn learned_chooser_beats_best_fixed_on_sort() {
        let mut rec: Recorder<Sort> = Recorder::new();
        for input in corpus(240, 1) {
            rec.observe(&input);
        }
        let mut chooser: Chooser<Sort> = Chooser::new();
        let cfg = TrainConfig {
            steps: 800,
            ..Default::default()
        };
        chooser.train(&rec, &cfg).unwrap();

        let eval = corpus(120, 9999);
        let report = chooser.evaluate(&eval);
        // The learned chooser should beat the best single fixed sort. (Measured
        // wall-clock is noisy, so the bar is "clearly better", not a fixed number.)
        assert!(
            report.learned_mean < report.best_fixed_mean,
            "learned {:.4} should beat best-fixed impl {} at {:.4}",
            report.learned_mean,
            report.best_fixed_impl,
            report.best_fixed_mean
        );
    }

    #[test]
    fn learned_chooser_beats_best_fixed_on_dict() {
        use crate::dict::{Dict, DictInput, build_dict_scenario};

        // A corpus spanning the regimes each lookup wins, including wide-key
        // workloads where `direct` is masked out and narrow-key ones where it
        // wins — so the generic mask path is exercised end to end.
        fn corpus(count: usize, base_seed: u32) -> Vec<DictInput> {
            let mut out = Vec::new();
            for i in 0..count {
                let seed = base_seed.wrapping_add(i as u32);
                out.push(match i % 4 {
                    0 => build_dict_scenario(8 + (i % 32), 8 + (i % 32), 1.0, 0.5, seed), // tiny
                    1 => build_dict_scenario(1000, 1000, 8.0, 0.5, seed),                 // medium
                    2 => build_dict_scenario(4000, 4000, 512.0, 0.3, seed), // wide → direct masked
                    _ => build_dict_scenario(4000, 6000, 0.1, 0.7, seed),   // narrow → direct wins
                });
            }
            out
        }

        let mut rec: Recorder<Dict> = Recorder::new();
        for input in corpus(200, 1) {
            rec.observe(&input);
        }
        let mut chooser: Chooser<Dict> = Chooser::new();
        let cfg = TrainConfig {
            steps: 800,
            ..Default::default()
        };
        chooser.train(&rec, &cfg).unwrap();

        let report = chooser.evaluate(&corpus(100, 9999));
        assert!(
            report.learned_mean < report.best_fixed_mean,
            "learned {:.4} should beat best-fixed impl {} at {:.4}",
            report.learned_mean,
            report.best_fixed_impl,
            report.best_fixed_mean
        );
    }

    #[test]
    fn config_can_pin_and_forbid() {
        let chooser: Chooser<Sort> = Chooser::new();
        let input = build_sort_scenario(1000, 0.5, 4.0, 3);

        let mut pinned = Chooser::<Sort>::new();
        pinned.set_config(Config {
            force: Some(2),
            ..Default::default()
        });
        assert_eq!(pinned.select(&input), 2, "force should pin impl 2");

        let mut forbid = Chooser::<Sort>::new();
        forbid.set_config(Config {
            forbidden: vec![true, false, false, false],
            ..Default::default()
        });
        assert_ne!(
            forbid.select(&input),
            0,
            "forbidden impl 0 must not be picked"
        );

        let _ = chooser; // untrained default still selects a feasible impl
        assert!(Chooser::<Sort>::new().select(&input) < Sort::impl_count());
    }
}
