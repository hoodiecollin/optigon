//! The chooser model — a tiny 1-hidden-layer MLP regressor on real `candle`.
//!
//! This is the mechanical port of ml-prototyping's
//! `packages/ml-training/src/{mlp-model,train}.ts`, which were written against
//! `@voidloop/ml-core` (a 1:1 TS mirror of candle) precisely so this port is a
//! line-for-line translation. Two spots are NOT mechanical and are called out:
//!   1. grad clipping rewrites the ml-core `_setByVarId` GradStore hack in terms
//!      of candle's real `GradStore::get`/`insert` (see `clip_grads`);
//!   2. the ml-core relu-NaN-at-0 sharp edge does not exist in real candle, so
//!      that latent risk simply disappears.
//!
//! The model predicts a per-impl **log-cost** vector; the argmin over applicable
//! impls is the pick, so training is regret-aware rather than accuracy-optimizing.

use candle_core::{DType, Device, Error, Result, Tensor, Var};
use candle_nn::{AdamW, Optimizer, ParamsAdamW};

use crate::prng::Mulberry32;

/// One training row: cheap features → per-impl log-cost targets, with a mask
/// zeroing out inapplicable impls (excluded from the loss and the argmin).
#[derive(Clone, Debug)]
pub struct RegressionSample {
    pub features: Vec<f32>,
    pub targets: Vec<f32>,
    pub mask: Vec<f32>,
}

/// Optimizer / loop hyperparameters. Defaults match the shipped chooser
/// (`docs/plans/chooser-candle-refactor.md`).
#[derive(Clone, Debug)]
pub struct TrainConfig {
    pub steps: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub weight_decay: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub grad_clip_norm: f64,
    pub seed: u32,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            steps: 1500,
            batch_size: 64,
            learning_rate: 0.05,
            weight_decay: 0.0,
            beta1: 0.9,
            beta2: 0.999,
            grad_clip_norm: 1.0,
            seed: 0x00c0_ffee,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TrainOutcome {
    pub steps_run: usize,
    pub initial_loss: f32,
    pub final_loss: f32,
}

/// Fixed parameter order: [W1, b1, W2, b2]. Save/load keys follow it.
const PARAM_NAMES: [&str; 4] = ["l1.weight", "l1.bias", "l2.weight", "l2.bias"];

pub struct MlpModel {
    w1: Var,
    b1: Var,
    w2: Var,
    b2: Var,
    pub in_dim: usize,
    pub hidden: usize,
    pub out_dim: usize,
}

impl MlpModel {
    /// Deterministic Glorot-ish init from `seed` (via mulberry32 + Box-Muller),
    /// mirroring `mlp-model.ts` so runs reproduce.
    pub fn new(in_dim: usize, hidden: usize, out_dim: usize, seed: u32) -> Result<Self> {
        let dev = Device::Cpu;
        let mut rng = Mulberry32::new(seed);
        let w1 = Var::from_tensor(&random_tensor(
            &[hidden, in_dim],
            1.0 / (in_dim as f64).sqrt(),
            &mut rng,
            &dev,
        )?)?;
        let b1 = Var::from_tensor(&Tensor::zeros(hidden, DType::F32, &dev)?)?;
        let w2 = Var::from_tensor(&random_tensor(
            &[out_dim, hidden],
            1.0 / (hidden as f64).sqrt(),
            &mut rng,
            &dev,
        )?)?;
        let b2 = Var::from_tensor(&Tensor::zeros(out_dim, DType::F32, &dev)?)?;
        Ok(Self {
            w1,
            b1,
            w2,
            b2,
            in_dim,
            hidden,
            out_dim,
        })
    }

    fn vars(&self) -> Vec<Var> {
        vec![
            self.w1.clone(),
            self.b1.clone(),
            self.w2.clone(),
            self.b2.clone(),
        ]
    }

    /// Forward pass: `x` is `[N, in_dim]` (already standardized), out `[N, out_dim]`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = x
            .matmul(&self.w1.as_tensor().t()?)?
            .broadcast_add(self.b1.as_tensor())?
            .relu()?;
        h.matmul(&self.w2.as_tensor().t()?)?
            .broadcast_add(self.b2.as_tensor())
    }

    /// Predict the per-impl log-cost vector for a single standardized feature row.
    pub fn predict_row(&self, features: &[f32]) -> Result<Vec<f32>> {
        let x = Tensor::from_vec(features.to_vec(), (1, self.in_dim), &Device::Cpu)?;
        let out = self.forward(&x)?;
        Ok(out.to_vec2::<f32>()?.into_iter().next().unwrap())
    }

    /// The regression training loop — the twin of `train.ts::trainSupervisedRegress`.
    pub fn train(&self, samples: &[RegressionSample], cfg: &TrainConfig) -> Result<TrainOutcome> {
        if samples.is_empty() {
            return Err(Error::Msg("train: no samples".into()));
        }
        let out_dim = self.out_dim;
        let dev = Device::Cpu;
        let mut opt = AdamW::new(
            self.vars(),
            ParamsAdamW {
                lr: cfg.learning_rate,
                beta1: cfg.beta1,
                beta2: cfg.beta2,
                eps: 1e-8,
                weight_decay: cfg.weight_decay,
            },
        )?;

        let mut rng = Mulberry32::new(cfg.seed);
        let indices = shuffle(samples.len(), &mut rng);
        let mut cursor = 0usize;
        let vars = self.vars();
        let mut initial = f32::NAN;
        let mut last = f32::NAN;

        for step in 0..cfg.steps {
            let (feat, targ, mask, applicable) = sample_batch(
                samples,
                &indices,
                cursor,
                cfg.batch_size,
                self.in_dim,
                out_dim,
            );
            cursor = (cursor + cfg.batch_size) % indices.len();

            let x = Tensor::from_vec(feat, (cfg.batch_size, self.in_dim), &dev)?;
            let target = Tensor::from_vec(targ, (cfg.batch_size, out_dim), &dev)?;
            let mask_t = Tensor::from_vec(mask, (cfg.batch_size, out_dim), &dev)?;

            let pred = self.forward(&x)?;
            let loss = masked_mse(&pred, &target, &mask_t, applicable)?;
            let loss_val = loss.to_scalar::<f32>()?;
            if step == 0 {
                initial = loss_val;
            }
            last = loss_val;

            let mut grads = loss.backward()?;
            clip_grads(&vars, &mut grads, cfg.grad_clip_norm)?;
            opt.step(&grads)?;
        }

        Ok(TrainOutcome {
            steps_run: cfg.steps,
            initial_loss: initial,
            final_loss: last,
        })
    }

    /// Save weights as safetensors (round-trips with the Rust `safetensors` crate).
    pub fn save(&self, path: &str) -> Result<()> {
        let mut map = std::collections::HashMap::new();
        map.insert(PARAM_NAMES[0].to_string(), self.w1.as_tensor().clone());
        map.insert(PARAM_NAMES[1].to_string(), self.b1.as_tensor().clone());
        map.insert(PARAM_NAMES[2].to_string(), self.w2.as_tensor().clone());
        map.insert(PARAM_NAMES[3].to_string(), self.b2.as_tensor().clone());
        candle_core::safetensors::save(&map, path)
    }

    /// Load weights saved by [`MlpModel::save`], inferring shapes from the tensors.
    pub fn load(path: &str) -> Result<Self> {
        let dev = Device::Cpu;
        let t = candle_core::safetensors::load(path, &dev)?;
        let get = |k: &str| -> Result<Tensor> {
            t.get(k)
                .cloned()
                .ok_or_else(|| Error::Msg(format!("safetensors missing key {k}")))
        };
        let w1 = get(PARAM_NAMES[0])?;
        let w2 = get(PARAM_NAMES[2])?;
        let (hidden, in_dim) = w1.dims2()?;
        let (out_dim, _) = w2.dims2()?;
        Ok(Self {
            w1: Var::from_tensor(&w1)?,
            b1: Var::from_tensor(&get(PARAM_NAMES[1])?)?,
            w2: Var::from_tensor(&w2)?,
            b2: Var::from_tensor(&get(PARAM_NAMES[3])?)?,
            in_dim,
            hidden,
            out_dim,
        })
    }
}

/// MSE over only the unmasked (applicable) outputs. With an all-ones mask this
/// is a plain mean over all elements. `applicable` is the count of `mask==1`.
fn masked_mse(pred: &Tensor, target: &Tensor, mask: &Tensor, applicable: f64) -> Result<Tensor> {
    let se = pred.sub(target)?.sqr()?.mul(mask)?;
    se.sum_all()?.affine(1.0 / applicable.max(1.0), 0.0)
}

/// Global-norm gradient clipping — the candle-idiomatic rewrite of ml-core's
/// `_setByVarId` GradStore hack. Reads each grad via `GradStore::get`, and if the
/// global norm exceeds `max_norm`, reinserts the scaled grad via `insert`.
fn clip_grads(
    vars: &[Var],
    grads: &mut candle_core::backprop::GradStore,
    max_norm: f64,
) -> Result<()> {
    if max_norm <= 0.0 {
        return Ok(());
    }
    let mut sumsq = 0f64;
    for v in vars {
        if let Some(g) = grads.get(v.as_tensor()) {
            for x in g.flatten_all()?.to_vec1::<f32>()? {
                sumsq += (x as f64) * (x as f64);
            }
        }
    }
    let norm = sumsq.sqrt();
    if norm <= max_norm || !norm.is_finite() {
        return Ok(());
    }
    let scale = max_norm / norm;
    for v in vars {
        // Scope the immutable borrow so the mutable `insert` below is legal.
        let scaled = match grads.get(v.as_tensor()) {
            Some(g) => Some(g.affine(scale, 0.0)?),
            None => None,
        };
        if let Some(s) = scaled {
            grads.insert(v.as_tensor(), s);
        }
    }
    Ok(())
}

fn shuffle(n: usize, rng: &mut Mulberry32) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..n).collect();
    // Fisher-Yates with the shared PRNG.
    for i in (1..n).rev() {
        let j = (rng.next_f64() * (i as f64 + 1.0)).floor() as usize;
        idx.swap(i, j.min(i));
    }
    idx
}

/// Gather a batch (wrapping around the shuffled indices), returning flattened
/// feature/target/mask buffers plus the applicable (unmasked) count.
fn sample_batch(
    samples: &[RegressionSample],
    indices: &[usize],
    cursor: usize,
    batch: usize,
    in_dim: usize,
    out_dim: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, f64) {
    let mut feat = Vec::with_capacity(batch * in_dim);
    let mut targ = Vec::with_capacity(batch * out_dim);
    let mut mask = Vec::with_capacity(batch * out_dim);
    let mut applicable = 0f64;
    for i in 0..batch {
        let s = &samples[indices[(cursor + i) % indices.len()]];
        feat.extend_from_slice(&s.features);
        targ.extend_from_slice(&s.targets);
        mask.extend_from_slice(&s.mask);
        for &m in &s.mask {
            applicable += m as f64;
        }
    }
    (feat, targ, mask, applicable)
}

fn random_tensor(dims: &[usize], scale: f64, rng: &mut Mulberry32, dev: &Device) -> Result<Tensor> {
    let n: usize = dims.iter().product();
    let data: Vec<f32> = (0..n).map(|_| (gaussian(rng) * scale) as f32).collect();
    Tensor::from_vec(data, dims, dev)
}

/// Box-Muller standard normal from the shared uniform PRNG (matches `mlp-model.ts`).
fn gaussian(rng: &mut Mulberry32) -> f64 {
    let u = rng.next_f64().max(1e-9);
    let v = rng.next_f64();
    (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn training_reduces_loss_and_learns_argmin() {
        // Teacher: 2 impls. For feature f0 < 0, impl 0 is cheaper; for f0 > 0,
        // impl 1. Targets are log-costs. The model should drive loss down and
        // recover the argmin split.
        let mut rng = Mulberry32::new(42);
        let mut samples = Vec::new();
        for _ in 0..512 {
            let f0 = (rng.next_f64() * 2.0 - 1.0) as f32;
            let f1 = (rng.next_f64() * 2.0 - 1.0) as f32;
            let (c0, c1): (f32, f32) = if f0 < 0.0 { (0.1, 1.0) } else { (1.0, 0.1) };
            samples.push(RegressionSample {
                features: vec![f0, f1],
                targets: vec![c0.ln(), c1.ln()],
                mask: vec![1.0, 1.0],
            });
        }
        let model = MlpModel::new(2, 16, 2, 7).unwrap();
        let cfg = TrainConfig {
            steps: 400,
            ..Default::default()
        };
        let out = model.train(&samples, &cfg).unwrap();
        assert!(
            out.final_loss < out.initial_loss * 0.5,
            "loss did not fall: {} -> {}",
            out.initial_loss,
            out.final_loss
        );
        // argmin recovered on held-out points
        let neg = model.predict_row(&[-0.8, 0.3]).unwrap();
        let pos = model.predict_row(&[0.8, -0.3]).unwrap();
        assert!(neg[0] < neg[1], "expected impl 0 for f0<0: {neg:?}");
        assert!(pos[1] < pos[0], "expected impl 1 for f0>0: {pos:?}");
    }

    #[test]
    fn safetensors_round_trip() {
        let model = MlpModel::new(4, 8, 3, 1).unwrap();
        let path = std::env::temp_dir().join("optigon_model_test.safetensors");
        let p = path.to_str().unwrap();
        model.save(p).unwrap();
        let loaded = MlpModel::load(p).unwrap();
        assert_eq!((loaded.in_dim, loaded.hidden, loaded.out_dim), (4, 8, 3));
        let a = model.predict_row(&[0.1, 0.2, 0.3, 0.4]).unwrap();
        let b = loaded.predict_row(&[0.1, 0.2, 0.3, 0.4]).unwrap();
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-6, "reload changed prediction");
        }
    }
}
