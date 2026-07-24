//! Capture — where training data comes from.
//!
//! Both training modes feed the same `(features, per-impl cost, mask)` rows:
//!   - **Mode 1 (test-driven):** a [`Recorder`] observes workloads as the
//!     consumer's tests drive the domain interface, running *every* impl to
//!     record the full cost row. More diverse tests → a better chooser.
//!   - **Mode 2 (prod A/B):** capture in production by switching impls per call
//!     and logging the measured cost (the online path; same row shape, not built
//!     in this walking-skeleton slice).
//!
//! This module also holds the regret evaluation used to report how good a
//! trained chooser is versus the oracle and the best fixed impl.

use std::marker::PhantomData;

use serde::{Deserialize, Serialize};

use crate::domain::Domain;

/// One captured workload: cheap features plus the measured cost of each impl
/// (with a mask zeroing inapplicable impls). Costs are raw (not yet log'd or
/// standardized); the chooser does that when it trains.
///
/// Mode 1 fills every applicable column; Mode 2 (online A/B) fills a single
/// column — the one impl it actually ran and measured — leaving the mask a
/// one-hot. The training loop masks per row, so both shapes train identically.
/// `serde` lets Mode-2 logs persist to disk for genuinely offline retraining.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RawRow {
    pub features: Vec<f32>,
    pub costs: Vec<f64>,
    pub mask: Vec<f32>,
}

/// Accumulates [`RawRow`]s from observed workloads (Mode 1).
pub struct Recorder<D: Domain> {
    rows: Vec<RawRow>,
    _pd: PhantomData<D>,
}

impl<D: Domain> Default for Recorder<D> {
    fn default() -> Self {
        Self {
            rows: Vec::new(),
            _pd: PhantomData,
        }
    }
}

impl<D: Domain> Recorder<D> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Observe one workload: extract features and measure every applicable impl.
    pub fn observe(&mut self, input: &D::Input) {
        let features = D::features(input);
        let n = D::impl_count();
        let mut costs = vec![0.0f64; n];
        let mut mask = vec![0.0f32; n];
        for i in 0..n {
            if D::applicable(i, input) {
                costs[i] = D::cost(i, input);
                mask[i] = 1.0;
            }
        }
        self.rows.push(RawRow {
            features,
            costs,
            mask,
        });
    }

    pub fn rows(&self) -> &[RawRow] {
        &self.rows
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }
}

/// How a trained chooser scores against the oracle and the best fixed impl.
#[derive(Clone, Debug)]
pub struct RegretReport {
    /// Mean regret of the learned chooser: `(chosen - oracle) / oracle`.
    pub learned_mean: f64,
    /// Fraction of workloads where the chooser picked the true argmin.
    pub optimal_rate: f64,
    /// The single best fixed impl (lowest mean regret if always chosen).
    pub best_fixed_impl: usize,
    /// That impl's mean regret — the honest bar the chooser must beat.
    pub best_fixed_mean: f64,
    pub n: usize,
}
