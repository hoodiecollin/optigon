//! The domain abstraction.
//!
//! A `Domain` is a family of interchangeable implementations of one operation
//! (sorting, dictionary lookup, joining, …) plus the *cheap* workload features
//! that predict which implementation is fastest. Everything above this trait —
//! the chooser, the candle model, the capture hook, the bindings — is written
//! generically against it, so adding the next domain is implementing this trait
//! and nothing else.

/// A packaged family of interchangeable implementations.
pub trait Domain {
    /// The workload handed to every implementation (e.g. the keys to sort).
    type Input;
    /// What an implementation returns (e.g. the sorted keys).
    type Output;

    /// Stable domain name, e.g. `"sort"`.
    const NAME: &'static str;
    /// Length of the feature vector `features` returns — fixed per domain.
    const FEATURE_DIM: usize;

    /// Implementation ids in a FIXED order. Cost/target columns and the model's
    /// output vector all follow this order, so it must never be reordered.
    fn impl_names() -> &'static [&'static str];

    /// Cheap workload features — O(sample), never as expensive as running an
    /// implementation. Length must equal `FEATURE_DIM`.
    fn features(input: &Self::Input) -> Vec<f32>;

    /// Run implementation `impl_idx` on `input`.
    fn run(impl_idx: usize, input: &Self::Input) -> Self::Output;

    /// Ground-truth cost of `impl_idx` on `input` (lower is better). Measured or
    /// modeled per domain; drives training targets and A/B capture.
    fn cost(impl_idx: usize, input: &Self::Input) -> f64;

    /// Whether `impl_idx` is applicable to `input`. Inapplicable impls are
    /// masked out of the loss, the regret argmin, and inference.
    fn applicable(_impl_idx: usize, _input: &Self::Input) -> bool {
        true
    }

    /// Number of implementations in the family.
    fn impl_count() -> usize {
        Self::impl_names().len()
    }
}
