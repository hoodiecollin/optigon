//! mulberry32 — the exact PRNG the ml-prototyping TS corpus generators use
//! (`sortenv.ts`), ported verbatim so Rust-side synthetic workloads reproduce
//! bit-for-bit from `(cfg, seed)`. `Math.imul` is a 32-bit wrapping multiply, so
//! `u32::wrapping_mul` reproduces it exactly; `>>> 0` is identity on `u32`.

pub struct Mulberry32 {
    s: u32,
}

impl Mulberry32 {
    pub fn new(seed: u32) -> Self {
        Self { s: seed }
    }

    /// Next float in [0, 1), matching the TS `mulberry32` output stream.
    pub fn next_f64(&mut self) -> f64 {
        self.s = self.s.wrapping_add(0x6d2b_79f5);
        let s = self.s;
        let mut t = (s ^ (s >> 15)).wrapping_mul(1 | s);
        t = t.wrapping_add((t ^ (t >> 7)).wrapping_mul(61 | t)) ^ t;
        ((t ^ (t >> 14)) as f64) / 4_294_967_296.0
    }
}
