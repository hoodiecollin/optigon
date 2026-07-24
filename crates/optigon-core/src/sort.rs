//! The sort domain — the first packaged family and the walking-skeleton domain.
//!
//! Four sorts with genuinely different sweet spots, so the fastest one really
//! does depend on the workload:
//!   - insertion : tiny n, or nearly-sorted input (adaptive)
//!   - quick     : general-purpose random data, wide key range
//!   - merge     : stable O(n log n) everywhere, wins when quick degrades
//!   - radix     : large n with a bounded key range (no comparisons)
//!
//! Ported from ml-prototyping `packages/chooser/src/sort/sortenv.ts`. Keys are
//! non-negative `i32`. Cost is measured wall-clock (median of a few reps); the
//! sizes here are large enough for a timer to resolve. Features are sampled
//! deterministically (strided), so a given input always yields the same vector.

use std::time::Instant;

use crate::domain::Domain;
use crate::prng::Mulberry32;

/// Implementation ids in fixed order — cost/target columns follow this.
pub const SORT_IMPLS: [&str; 4] = ["insertion", "quick", "merge", "radix"];

/// quick/merge fall back to insertion below this width.
const INSERTION_CUTOFF: usize = 32;
/// Number of positions sampled for cheap features.
const SAMPLE: usize = 256;

pub struct Sort;

impl Domain for Sort {
    type Input = Vec<i32>;
    type Output = Vec<i32>;

    const NAME: &'static str = "sort";
    const FEATURE_DIM: usize = 4;

    fn impl_names() -> &'static [&'static str] {
        &SORT_IMPLS
    }

    fn features(input: &Self::Input) -> Vec<f32> {
        extract_features(input)
    }

    fn run(impl_idx: usize, input: &Self::Input) -> Self::Output {
        match impl_idx {
            0 => insertion_sort(input),
            1 => quick_sort(input),
            2 => merge_sort(input),
            3 => radix_sort(input),
            _ => panic!("sort: impl_idx {impl_idx} out of range"),
        }
    }

    fn cost(impl_idx: usize, input: &Self::Input) -> f64 {
        // Median of a few reps of a fresh sort, in milliseconds. A correctness
        // guard (last >= first) catches a broken impl producing a training row.
        let reps = 3;
        let mut ts = Vec::with_capacity(reps);
        for _ in 0..reps {
            let t0 = Instant::now();
            let out = Self::run(impl_idx, input);
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            if out.len() > 1 && out[out.len() - 1] < out[0] {
                panic!(
                    "sort: impl {} produced unsorted output",
                    SORT_IMPLS[impl_idx]
                );
            }
            ts.push(ms);
        }
        ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        ts[ts.len() / 2]
    }
}

// --- implementations --------------------------------------------------------

fn insertion_range(a: &mut [i32], lo: usize, hi: usize) {
    if hi <= lo {
        return;
    }
    for i in (lo + 1)..=hi {
        let v = a[i];
        let mut j = i;
        while j > lo && a[j - 1] > v {
            a[j] = a[j - 1];
            j -= 1;
        }
        a[j] = v;
    }
}

fn insertion_sort(src: &[i32]) -> Vec<i32> {
    let mut a = src.to_vec();
    let n = a.len();
    if n > 1 {
        insertion_range(&mut a, 0, n - 1);
    }
    a
}

/// Median-of-three Hoare quicksort with an insertion cutoff and an explicit
/// stack — robust O(n log n) on sorted / reverse / duplicate-heavy input.
fn quick_sort(src: &[i32]) -> Vec<i32> {
    let mut a = src.to_vec();
    let n = a.len();
    if n < 2 {
        return a;
    }
    let mut stack: Vec<(usize, usize)> = vec![(0, n - 1)];
    while let Some((lo, hi)) = stack.pop() {
        if hi <= lo {
            continue;
        }
        if hi - lo < INSERTION_CUTOFF {
            insertion_range(&mut a, lo, hi);
            continue;
        }
        let mid = lo + (hi - lo) / 2;
        let (x, y, z) = (a[lo], a[mid], a[hi]);
        let pivot = if x < y {
            if y < z {
                y
            } else if x < z {
                z
            } else {
                x
            }
        } else if x < z {
            x
        } else if y < z {
            z
        } else {
            y
        };
        // Hoare partition around the pivot value.
        let mut i = lo as isize - 1;
        let mut j = hi as isize + 1;
        let p = loop {
            loop {
                i += 1;
                if a[i as usize] >= pivot {
                    break;
                }
            }
            loop {
                j -= 1;
                if a[j as usize] <= pivot {
                    break;
                }
            }
            if i >= j {
                break j as usize;
            }
            a.swap(i as usize, j as usize);
        };
        // Push both sides; the (hi <= lo) guard handles empty partitions.
        stack.push((lo, p));
        stack.push((p + 1, hi));
    }
    a
}

/// Bottom-up merge sort with insertion-sorted initial runs for locality.
fn merge_sort(src: &[i32]) -> Vec<i32> {
    let n = src.len();
    let mut a = src.to_vec();
    if n < 2 {
        return a;
    }
    let mut buf = vec![0i32; n];
    let mut lo = 0;
    while lo < n {
        insertion_range(&mut a, lo, (lo + INSERTION_CUTOFF - 1).min(n - 1));
        lo += INSERTION_CUTOFF;
    }
    // `a` holds the current source; `buf` the destination. Swap each pass.
    let mut width = INSERTION_CUTOFF;
    let mut in_a = true;
    while width < n {
        {
            let (src2, dst): (&[i32], &mut [i32]) =
                if in_a { (&a, &mut buf) } else { (&buf, &mut a) };
            let mut lo = 0;
            while lo < n {
                let mid = (lo + width).min(n);
                let hi = (lo + 2 * width).min(n);
                let (mut i, mut j, mut k) = (lo, mid, lo);
                while i < mid && j < hi {
                    if src2[i] <= src2[j] {
                        dst[k] = src2[i];
                        i += 1;
                    } else {
                        dst[k] = src2[j];
                        j += 1;
                    }
                    k += 1;
                }
                while i < mid {
                    dst[k] = src2[i];
                    i += 1;
                    k += 1;
                }
                while j < hi {
                    dst[k] = src2[j];
                    j += 1;
                    k += 1;
                }
                lo += 2 * width;
            }
        }
        in_a = !in_a;
        width *= 2;
    }
    if in_a { a } else { buf }
}

/// LSD radix sort, base 256, pass count adapted to the actual max key so a
/// narrow key range is cheap. Keys are non-negative.
fn radix_sort(src: &[i32]) -> Vec<i32> {
    let n = src.len();
    let mut a: Vec<u32> = src.iter().map(|&x| x as u32).collect();
    if n < 2 {
        return a.iter().map(|&x| x as i32).collect();
    }
    let max = *a.iter().max().unwrap();
    let mut out = vec![0u32; n];
    let mut shift = 0u32;
    while (max >> shift) > 0 {
        let mut count = [0usize; 256];
        for &v in &a {
            count[((v >> shift) & 0xff) as usize] += 1;
        }
        for b in 1..256 {
            count[b] += count[b - 1];
        }
        for i in (0..n).rev() {
            let d = ((a[i] >> shift) & 0xff) as usize;
            count[d] -= 1;
            out[count[d]] = a[i];
        }
        std::mem::swap(&mut a, &mut out);
        shift += 8;
    }
    a.iter().map(|&x| x as i32).collect()
}

// --- features ---------------------------------------------------------------

/// Cheap sampled features (all O(SAMPLE), deterministic strided sampling):
///   f0 log2(n)
///   f1 presortedness  (fraction of sampled adjacent pairs already in order)
///   f2 dupFraction    (1 - distinct/sampled)
///   f3 log2(keyRange) (from sampled min/max)
fn extract_features(keys: &[i32]) -> Vec<f32> {
    let n = keys.len();
    if n < 2 {
        return vec![0.0, 1.0, 0.0, 0.0];
    }
    let s = n.min(SAMPLE);
    // Evenly spaced start positions across [0, n-2] so we can read pair (i, i+1).
    let span = (n - 1).max(1);
    let mut in_order = 0usize;
    let mut pairs = 0usize;
    let mut lo = i32::MAX;
    let mut hi = i32::MIN;
    let mut seen = std::collections::HashSet::with_capacity(s);
    for t in 0..s {
        let i = ((t * span) / s).min(n - 2);
        let (a, b) = (keys[i], keys[i + 1]);
        if a <= b {
            in_order += 1;
        }
        pairs += 1;
        seen.insert(a);
        lo = lo.min(a);
        hi = hi.max(a);
    }
    let presort = if pairs > 0 {
        in_order as f32 / pairs as f32
    } else {
        1.0
    };
    let dup = 1.0 - seen.len() as f32 / (s.max(1) as f32);
    let range = ((hi - lo) as i64 + 1).max(1) as f32;
    vec![(n as f32).log2(), presort, dup, range.log2()]
}

// --- scenario builder (tests + demos) ---------------------------------------

/// Build a sort workload from a config, matching `sortenv.ts::buildSortScenario`:
///   n            array size
///   sortedness   0 (random) .. 1 (fully sorted)
///   key_space_mult  keys drawn from [0, round(n * mult)); small => duplicates +
///                   narrow range (radix-friendly), large => wide keys
pub fn build_sort_scenario(n: usize, sortedness: f64, key_space_mult: f64, seed: u32) -> Vec<i32> {
    let mut rng = Mulberry32::new(seed ^ 0x9e37_79b1);
    let n = n.max(2);
    let key_space = ((n as f64 * key_space_mult).round() as i64).max(2);
    let sortedness = sortedness.clamp(0.0, 1.0);

    let mut keys: Vec<i32> = (0..n)
        .map(|_| (rng.next_f64() * key_space as f64).floor() as i32)
        .collect();
    keys.sort_unstable();
    let disturb = ((1.0 - sortedness) * n as f64).round() as usize;
    for _ in 0..disturb {
        let i = (rng.next_f64() * n as f64).floor() as usize % n;
        keys[i] = (rng.next_f64() * key_space as f64).floor() as i32;
    }
    keys
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_sorted(a: &[i32]) -> bool {
        a.windows(2).all(|w| w[0] <= w[1])
    }

    #[test]
    fn every_impl_sorts_every_scenario() {
        // Span the regimes each sort is meant to win: tiny, nearly-sorted, wide
        // random, and large narrow-key (radix-friendly).
        let scenarios = [
            build_sort_scenario(8, 0.0, 4.0, 1),
            build_sort_scenario(500, 0.95, 8.0, 2),
            build_sort_scenario(4000, 0.1, 64.0, 3),
            build_sort_scenario(16384, 0.0, 0.25, 4),
        ];
        for keys in &scenarios {
            let mut expected = keys.clone();
            expected.sort_unstable();
            for idx in 0..Sort::impl_count() {
                let out = Sort::run(idx, keys);
                assert!(
                    is_sorted(&out),
                    "impl {} left output unsorted (n={})",
                    SORT_IMPLS[idx],
                    keys.len()
                );
                assert_eq!(
                    out, expected,
                    "impl {} disagrees with reference sort",
                    SORT_IMPLS[idx]
                );
            }
        }
    }

    #[test]
    fn features_have_fixed_dim_and_are_deterministic() {
        let keys = build_sort_scenario(2000, 0.5, 4.0, 7);
        let f1 = Sort::features(&keys);
        let f2 = Sort::features(&keys);
        assert_eq!(f1.len(), Sort::FEATURE_DIM);
        assert_eq!(f1, f2, "features must be deterministic for a fixed input");
    }
}
