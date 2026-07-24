//! # optigon-core
//!
//! The language-agnostic core of Optigon: a [`Domain`] abstraction over a family
//! of interchangeable implementations, the packaged domains themselves (starting
//! with [`sort`]), and — layered on top — the regret-scored chooser that learns
//! which implementation to run per workload (candle training + inference).
//!
//! The napi and pyo3 bindings are thin wrappers over this crate; all real logic
//! lives here.

pub mod capture;
pub mod chooser;
pub mod config;
pub mod dict;
pub mod domain;
pub mod model;
pub mod online;
pub mod prng;
pub mod sort;

pub use capture::{RawRow, Recorder, RegretReport};
pub use chooser::Chooser;
pub use config::Config;
pub use dict::Dict;
pub use domain::Domain;
pub use model::{TrainConfig, TrainOutcome};
pub use online::{OnlineAb, read_log_jsonl, retrain_from_log};
pub use sort::Sort;
