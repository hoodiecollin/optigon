//! Consumer-facing steering config.
//!
//! The chooser is the default policy, but a consumer can *steer* it without
//! retraining: nudge the predicted cost of an impl, forbid an impl outright, or
//! pin one impl and bypass the model entirely. This is the lightweight, per-call
//! analog of ml-prototyping's `CostObjective` (bias ≈ composite weighting,
//! forbid ≈ a hard applicability constraint).

/// Steering knobs applied on top of the learned prediction at dispatch time.
#[derive(Clone, Debug, Default)]
pub struct Config {
    /// Additive per-impl bias on predicted **log-cost** (negative favors the
    /// impl, positive penalizes it). Empty = no bias. Length, when non-empty,
    /// must equal the domain's impl count.
    pub bias: Vec<f32>,
    /// Impls the consumer forbids — always masked out of selection. Empty = none.
    pub forbidden: Vec<bool>,
    /// Pin a specific impl and bypass the model entirely (still checked against
    /// applicability). `None` = let the chooser decide.
    pub force: Option<usize>,
}

impl Config {
    pub fn is_forbidden(&self, impl_idx: usize) -> bool {
        self.forbidden.get(impl_idx).copied().unwrap_or(false)
    }

    pub fn bias_for(&self, impl_idx: usize) -> f32 {
        self.bias.get(impl_idx).copied().unwrap_or(0.0)
    }
}
