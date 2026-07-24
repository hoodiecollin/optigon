//! The dictionary domain — the second packaged family.
//!
//! Given a set of integer keys to build a lookup structure over and a stream of
//! query keys, answer each query with the stored value (the key itself here) or
//! `-1` on a miss. Four structures with genuinely different sweet spots:
//!   - linear : scan the keys per query — wins for tiny key sets (no build cost)
//!   - binary : sort once, binary-search each query — the balanced middle ground
//!   - hash   : a `HashMap`, O(1) lookups — wins when keys *and* queries are large
//!   - direct : a direct-address table indexed by `key - min` — fastest when it
//!     applies, but **only applicable for a bounded key range**
//!
//! `direct` is the point of this domain: it exercises the `applicable` mask that
//! the sort walking-skeleton never did. On a wide key range it is masked out of
//! the loss, the regret argmin, and inference — and it is excluded from the
//! best-fixed baseline because it can't be the fixed choice everywhere. The
//! learned chooser gets to win partly by reaching for an impl no fixed policy
//! could commit to.
//!
//! Input is a struct (`DictInput`), not a flat vector, which is the other thing
//! this domain proves: the `Domain` trait is generic over arbitrary inputs.
//! Cost is measured wall-clock (median of a few reps), same as `sort`.

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use crate::domain::Domain;
use crate::prng::Mulberry32;

/// Implementation ids in fixed order — cost/target columns follow this.
pub const DICT_IMPLS: [&str; 4] = ["linear", "binary", "hash", "direct"];

/// `direct` is applicable only when the key range fits in this many slots — past
/// this a direct-address table is not a sane structure and is masked out.
const DIRECT_MAX_SLOTS: i64 = 1 << 20; // ~1M slots
/// Number of keys sampled for cheap features.
const SAMPLE: usize = 256;

/// A dictionary workload: the keys to index, and the queries to answer.
#[derive(Clone, Debug)]
pub struct DictInput {
    pub keys: Vec<i32>,
    pub queries: Vec<i32>,
}

pub struct Dict;

impl Domain for Dict {
    type Input = DictInput;
    type Output = Vec<i32>;

    const NAME: &'static str = "dict";
    const FEATURE_DIM: usize = 4;

    fn impl_names() -> &'static [&'static str] {
        &DICT_IMPLS
    }

    fn features(input: &Self::Input) -> Vec<f32> {
        extract_features(input)
    }

    fn run(impl_idx: usize, input: &Self::Input) -> Self::Output {
        match impl_idx {
            0 => linear_lookup(input),
            1 => binary_lookup(input),
            2 => hash_lookup(input),
            3 => direct_lookup(input),
            _ => panic!("dict: impl_idx {impl_idx} out of range"),
        }
    }

    fn applicable(impl_idx: usize, input: &Self::Input) -> bool {
        match impl_idx {
            // `direct` needs a non-empty, bounded key range.
            3 => match key_range(&input.keys) {
                Some((lo, hi)) => (hi as i64 - lo as i64) < DIRECT_MAX_SLOTS,
                None => false,
            },
            _ => true,
        }
    }

    fn cost(impl_idx: usize, input: &Self::Input) -> f64 {
        // Median of a few reps of a fresh build+query pass, in milliseconds. A
        // length guard catches a broken impl producing a training row.
        let reps = 3;
        let mut ts = Vec::with_capacity(reps);
        for _ in 0..reps {
            let t0 = Instant::now();
            let out = Self::run(impl_idx, input);
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            if out.len() != input.queries.len() {
                panic!(
                    "dict: impl {} produced wrong-length output",
                    DICT_IMPLS[impl_idx]
                );
            }
            ts.push(ms);
        }
        ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        ts[ts.len() / 2]
    }
}

// --- implementations --------------------------------------------------------

/// Value stored for a present key. Identity keeps the correctness check trivial:
/// a hit returns the query key, a miss returns `-1` (keys are non-negative).
#[inline]
fn value_of(key: i32) -> i32 {
    key
}

fn linear_lookup(input: &DictInput) -> Vec<i32> {
    let mut out = Vec::with_capacity(input.queries.len());
    for &q in &input.queries {
        let mut v = -1;
        for &k in &input.keys {
            if k == q {
                v = value_of(k);
                break;
            }
        }
        out.push(v);
    }
    out
}

fn binary_lookup(input: &DictInput) -> Vec<i32> {
    let mut sorted = input.keys.clone();
    sorted.sort_unstable();
    let mut out = Vec::with_capacity(input.queries.len());
    for &q in &input.queries {
        out.push(if sorted.binary_search(&q).is_ok() {
            value_of(q)
        } else {
            -1
        });
    }
    out
}

fn hash_lookup(input: &DictInput) -> Vec<i32> {
    let map: HashMap<i32, i32> = input.keys.iter().map(|&k| (k, value_of(k))).collect();
    let mut out = Vec::with_capacity(input.queries.len());
    for &q in &input.queries {
        out.push(*map.get(&q).unwrap_or(&-1));
    }
    out
}

fn direct_lookup(input: &DictInput) -> Vec<i32> {
    let mut out = Vec::with_capacity(input.queries.len());
    let (lo, hi) = match key_range(&input.keys) {
        Some(r) => r,
        None => {
            // No keys → every query misses.
            out.resize(input.queries.len(), -1);
            return out;
        }
    };
    let size = (hi as i64 - lo as i64 + 1) as usize;
    let mut table = vec![-1i32; size];
    for &k in &input.keys {
        table[(k - lo) as usize] = value_of(k);
    }
    for &q in &input.queries {
        if q < lo || q > hi {
            out.push(-1);
        } else {
            out.push(table[(q - lo) as usize]);
        }
    }
    out
}

/// Exact (min, max) over the keys, or `None` when empty.
fn key_range(keys: &[i32]) -> Option<(i32, i32)> {
    if keys.is_empty() {
        return None;
    }
    let mut lo = keys[0];
    let mut hi = keys[0];
    for &k in &keys[1..] {
        lo = lo.min(k);
        hi = hi.max(k);
    }
    Some((lo, hi))
}

// --- features ---------------------------------------------------------------

/// Cheap sampled features (all O(SAMPLE), deterministic strided sampling):
///   f0 log2(n_keys)
///   f1 log2(n_queries)
///   f2 log2(sampled key range)   — small ⇒ direct-address friendly
///   f3 dupFraction of keys       — 1 - distinct/sampled
fn extract_features(input: &DictInput) -> Vec<f32> {
    let nk = input.keys.len();
    let nq = input.queries.len();
    if nk == 0 {
        return vec![0.0, (nq as f32 + 1.0).log2(), 0.0, 0.0];
    }
    let s = nk.min(SAMPLE);
    let span = nk.max(1);
    let mut lo = i32::MAX;
    let mut hi = i32::MIN;
    let mut seen = HashSet::with_capacity(s);
    for t in 0..s {
        let i = ((t * span) / s).min(nk - 1);
        let k = input.keys[i];
        lo = lo.min(k);
        hi = hi.max(k);
        seen.insert(k);
    }
    let range = ((hi - lo) as i64 + 1).max(1) as f32;
    let dup = 1.0 - seen.len() as f32 / (s.max(1) as f32);
    vec![
        (nk as f32 + 1.0).log2(),
        (nq as f32 + 1.0).log2(),
        range.log2(),
        dup,
    ]
}

// --- scenario builder (tests + demos) ---------------------------------------

/// Build a dictionary workload:
///   n_keys          number of keys to index
///   n_queries       number of lookups to answer
///   key_space_mult  keys drawn from [0, round(n_keys * mult)); small ⇒ narrow
///                   range (direct-friendly), large ⇒ wide range (direct masked)
///   hit_rate        fraction of queries drawn from the key set (rest likely miss)
pub fn build_dict_scenario(
    n_keys: usize,
    n_queries: usize,
    key_space_mult: f64,
    hit_rate: f64,
    seed: u32,
) -> DictInput {
    let mut rng = Mulberry32::new(seed ^ 0x85eb_ca6b);
    let n_keys = n_keys.max(1);
    let n_queries = n_queries.max(1);
    let key_space = ((n_keys as f64 * key_space_mult).round() as i64).max(2);
    let hit_rate = hit_rate.clamp(0.0, 1.0);

    let keys: Vec<i32> = (0..n_keys)
        .map(|_| (rng.next_f64() * key_space as f64).floor() as i32)
        .collect();
    let queries: Vec<i32> = (0..n_queries)
        .map(|_| {
            if rng.next_f64() < hit_rate {
                keys[(rng.next_f64() * n_keys as f64).floor() as usize % n_keys]
            } else {
                // Draw from a wider space so the query is likely absent.
                (rng.next_f64() * (key_space as f64 * 2.0)).floor() as i32
            }
        })
        .collect();
    DictInput { keys, queries }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A reference lookup all impls must agree with.
    fn reference(input: &DictInput) -> Vec<i32> {
        let set: HashSet<i32> = input.keys.iter().copied().collect();
        input
            .queries
            .iter()
            .map(|&q| if set.contains(&q) { q } else { -1 })
            .collect()
    }

    #[test]
    fn every_applicable_impl_agrees_with_reference() {
        // Span the regimes, and deliberately include a wide-key scenario where
        // `direct` is inapplicable and a narrow one where it is.
        let scenarios = [
            build_dict_scenario(16, 16, 1.0, 0.5, 1),     // tiny narrow
            build_dict_scenario(1000, 1000, 8.0, 0.5, 2), // medium
            build_dict_scenario(4000, 4000, 512.0, 0.3, 3), // wide  → direct masked
            build_dict_scenario(4000, 6000, 0.1, 0.7, 4), // narrow→ direct wins
        ];
        // Sanity: the mask actually flips across the corpus.
        assert!(
            !Dict::applicable(3, &scenarios[2]),
            "direct should be masked on wide keys"
        );
        assert!(
            Dict::applicable(3, &scenarios[3]),
            "direct should apply on narrow keys"
        );

        for input in &scenarios {
            let expected = reference(input);
            for idx in 0..Dict::impl_count() {
                if !Dict::applicable(idx, input) {
                    continue;
                }
                let out = Dict::run(idx, input);
                assert_eq!(
                    out,
                    expected,
                    "impl {} disagrees with reference (n_keys={}, n_queries={})",
                    DICT_IMPLS[idx],
                    input.keys.len(),
                    input.queries.len()
                );
            }
        }
    }

    #[test]
    fn features_have_fixed_dim_and_are_deterministic() {
        let input = build_dict_scenario(2000, 1500, 8.0, 0.5, 7);
        let f1 = Dict::features(&input);
        let f2 = Dict::features(&input);
        assert_eq!(f1.len(), Dict::FEATURE_DIM);
        assert_eq!(f1, f2, "features must be deterministic for a fixed input");
    }
}
