//! The MQ arithmetic *decoder* (jbig2decplan.md §10, T.88 Annex E).
//!
//! This is the inverse of the encoder's [`crate::encode::arith::Jbig2ArithCoder`]
//! register machine. It shares only the probability table
//! ([`crate::shared::mq_table`]); the decoder registers live here.
//!
//! The software conventions follow T.88 Annex E (INITDEC / DECODE / BYTEIN /
//! RENORMD, Figures E.15–E.20). Running past the end of the coded data feeds
//! `0xFF` bytes — the spec's "marker not found" convention — which produces a
//! stream of MPS/1-bits. Every consumer (generic region, integer decoder, …)
//! decodes a *bounded* number of bits, so a truncated stream always terminates;
//! it never loops or panics.
//!
//! The [`MqContext`] handle indexes [`MQ_STATES`], whose next-state fields
//! already fold in the conditional exchange (`SWITCH`) and encode the current
//! MPS in the index (`>= 47`). `decode_bit` therefore returns an absolute `0/1`
//! bit and rewrites the context index in place — no separate MPS/switch state.

use crate::shared::mq_table::{MQ_STATES, MqContext};

/// Packed MQ state entry for branch-free decoding:
/// `(qe << 16) | (next_lps << 8) | next_mps`.
const MQ_DECODE_STATES: [u32; 94] = {
    let mut table = [0u32; 94];
    let mut i = 0;
    while i < 94 {
        let e = MQ_STATES[i];
        table[i] = ((e.qe as u32) << 16) | ((e.next_lps as u32) << 8) | (e.next_mps as u32);
        i += 1;
    }
    table
};

/// A renormalisation shifts by at most 15 bits (`A >= 1` after exchange).
const MAX_RENORM_SHIFT: u32 = 15;

/// MQ arithmetic decoder over a borrowed coded-data slice.
///
/// Uses a 64-bit code register `c` with `Chigh` in the top 16 bits and up to
/// 32 buffered input bits below it, so renormalisation (`RENORMD`) is a single
/// shift with `leading_zeros()` and `BYTEIN` refills multiple bytes at a time.
/// `a` carries the 16-bit interval register (`0x8000..=0xFFFF`); `ct` counts the
/// buffered bits; `bp` indexes the *current* byte `B = data[bp]`.
#[derive(Clone, Debug)]
pub struct ArithmeticDecoder<'a> {
    data: &'a [u8],
    /// Index of the current byte `B`. May advance one past `data.len()`, after
    /// which reads yield the `0xFF` past-end convention.
    bp: usize,
    c: u64,
    a: u32,
    ct: u32,
}

impl<'a> ArithmeticDecoder<'a> {
    /// Initialise the decoder over `data` (T.88 INITDEC, Figure E.20).
    pub fn new(data: &'a [u8]) -> Self {
        let mut dec = ArithmeticDecoder {
            data,
            bp: 0,
            c: 0,
            a: 0x8000,
            ct: 0,
        };
        // C = B << 48 (B is the first byte, or 0xFF past the end).
        let b0 = dec.byte_at(0);
        dec.c = u64::from(b0) << 48;
        dec.fill();
        dec.c <<= 7;
        dec.ct -= 7;
        if dec.ct < MAX_RENORM_SHIFT {
            dec.fill();
        }
        dec
    }

    /// The byte at absolute index `i`, or `0xFF` past the end (T.88 marker
    /// convention for running past the coded data).
    #[inline(always)]
    fn byte_at(&self, i: usize) -> u8 {
        match self.data.get(i) {
            Some(&b) => b,
            None => 0xFF,
        }
    }

    /// Buffer input bytes into `c` using T.88 Figure E.19 (`BYTEIN`), stopping
    /// once at least 25 bits are buffered (`ct > 24`).
    ///
    /// A stuffed byte `b1` in `0x80..=0x8F` after `0xFF` adds `b1 << 9`, whose
    /// bit 7 carries into the preceding byte's position; we only consume it
    /// once `ct == 0` so `Chigh` never observes that carry earlier than
    /// byte-at-a-time Annex E does (preserving bit-exact equivalence even on
    /// arbitrary malformed streams).
    #[inline(always)]
    fn fill(&mut self) {
        while self.ct <= 24 {
            let pos = 32 - self.ct;
            let b0 = self.byte_at(self.bp);
            if b0 != 0xFF {
                let b1 = self.byte_at(self.bp + 1);
                // Fast two-byte refill when the next byte is also plain and we
                // have room for 16 bits in the 32-bit buffer below Chigh.
                if b1 != 0xFF && self.ct <= 16 {
                    let b2 = self.byte_at(self.bp + 2);
                    let pair = (u64::from(b1) << 8) | u64::from(b2);
                    self.bp += 2;
                    self.c = self.c.wrapping_add(pair << pos);
                    self.ct += 16;
                } else {
                    self.bp += 1;
                    self.c = self.c.wrapping_add(u64::from(b1) << (8 + pos));
                    self.ct += 8;
                }
            } else {
                let b1 = self.byte_at(self.bp + 1);
                if b1 > 0x8F {
                    // Marker (or past-end): feed 1-bits without consuming the byte.
                    self.c = self.c.wrapping_add(0xFF00u64 << pos);
                    self.ct += 8;
                } else if b1 >= 0x80 && self.ct > 0 {
                    // Defer carry-bearing stuffed byte until `ct == 0`.
                    break;
                } else {
                    self.bp += 1;
                    self.c = self.c.wrapping_add(u64::from(b1) << (9 + pos));
                    self.ct += 7;
                }
            }
        }
    }

    /// Renormalise `c` when a deferred `0xFF 0x80..=0x8F` byte left fewer than
    /// `shift` bits buffered.
    #[cold]
    #[inline(never)]
    fn renorm_c_slow(&mut self, mut shift: u32) {
        if self.ct > 0 {
            let s = self.ct;
            self.c <<= s;
            shift -= s;
            self.ct = 0;
        }
        loop {
            self.fill();
            if shift <= self.ct {
                self.c <<= shift;
                self.ct -= shift;
                break;
            }
            let s = self.ct;
            self.c <<= s;
            shift -= s;
            self.ct = 0;
        }
        if self.ct < MAX_RENORM_SHIFT {
            self.fill();
        }
    }

    /// Decode one bit under `context`, updating the context state in place
    /// (T.88 DECODE / MPS_EXCHANGE / LPS_EXCHANGE, Figures E.15–E.17).
    ///
    /// The common case, an MPS that leaves `A` normalised, is tested first
    /// and needs no context write and no renormalisation; every other case
    /// takes [`Self::decode_exchange`], which is branch-free and renormalises
    /// in one shift.
    #[inline(always)]
    pub fn decode_bit(&mut self, context: &mut MqContext) -> bool {
        let s = context.0 as usize;
        let packed = match MQ_DECODE_STATES.get(s) {
            Some(&p) => p,
            None => return false,
        };
        let qe = packed >> 16;
        let a = self.a.wrapping_sub(qe);
        let chigh = (self.c >> 48) as u32;
        if chigh >= qe && a & 0x8000 != 0 {
            self.a = a;
            self.c = self.c.wrapping_sub(u64::from(qe) << 48);
            return s >= 47;
        }
        self.decode_exchange(context, packed, s >= 47, a, chigh)
    }

    /// The exchange half of DECODE: an LPS sub-interval, or an MPS that needs
    /// the conditional exchange and renormalisation. `a` is `A - Qe`.
    #[inline(always)]
    fn decode_exchange(
        &mut self,
        context: &mut MqContext,
        packed: u32,
        mps: bool,
        a: u32,
        chigh: u32,
    ) -> bool {
        let qe = packed >> 16;
        let lps = u32::from(chigh < qe);
        let lps_mask = lps.wrapping_neg();
        // MPS exchange subtracts Qe from Chigh; LPS exchange leaves C unchanged.
        self.c = self.c.wrapping_sub(u64::from(qe & !lps_mask) << 48);
        // Conditional exchange swaps the meaning of the two sub-intervals when
        // `A - Qe < Qe`, so the decoded symbol is the LPS iff exactly one of
        // "LPS sub-interval" and "exchanged" holds.
        let is_lps_symbol = lps ^ u32::from(a < qe);
        context.0 = ((packed >> (8 * is_lps_symbol)) & 0x7F) as u8;
        let a = (a & !lps_mask) | (qe & lps_mask);
        // RENORMD: `a` is in `1..=0x7FFF`, so `shift` is in `1..=15`.
        let shift = a.leading_zeros() - 16;
        self.a = a << shift;
        if shift <= self.ct {
            self.c <<= shift;
            self.ct -= shift;
            if self.ct < MAX_RENORM_SHIFT {
                self.fill();
            }
        } else {
            self.renorm_c_slow(shift);
        }
        (u32::from(mps) ^ is_lps_symbol) != 0
    }

    /// Decode a run of up to `count` symbols under `context` that are each the
    /// context's MPS, equal to `value`, *without* renormalisation, and return
    /// how many were decoded (all of them equal `value`).
    ///
    /// Each such step of [`Self::decode_bit`] subtracts the same `Qe` from `A`
    /// and from the high half of `C` and leaves the context state unchanged,
    /// so `n` steps fold into one subtraction of `n * Qe`. The run stops before
    /// the first symbol that would take the exchange path (an LPS, or `A`
    /// falling below 0x8000) or when the context's MPS is not `value`; the
    /// caller decodes that symbol with [`Self::decode_bit`]. The decoder state
    /// and context afterwards are observationally identical to `n` `decode_bit`
    /// calls.
    #[inline(always)]
    pub(crate) fn decode_mps_run(&mut self, context: MqContext, value: bool, count: u32) -> u32 {
        let s = context.0 as usize;
        let packed = match MQ_DECODE_STATES.get(s) {
            Some(&p) => p,
            None => return 0,
        };
        if (s >= 47) != value || self.a < 0x8000 {
            return 0;
        }
        let qe = packed >> 16;
        let chigh = (self.c >> 48) as u32;
        let avail = chigh.min(self.a - 0x8000);
        if avail < qe {
            return 0;
        }
        // count <= 32 and Qe <= 0x5601, so the product cannot overflow.
        let total = qe * count;
        let n = if avail >= total { count } else { avail / qe };
        let d = n * qe;
        self.a -= d;
        self.c = self.c.wrapping_sub(u64::from(d) << 48);
        n
    }

    /// Bytes consumed so far (index of the current byte `B`). Informational;
    /// the generic region does not rely on exact terminator alignment.
    #[inline]
    pub fn bytes_consumed(&self) -> usize {
        self.bp
    }
}

/// The pre-optimisation byte-at-a-time 32-bit Annex E MQ decoder, kept
/// verbatim as the equivalence reference for tests.
#[cfg(all(test, feature = "encode"))]
#[derive(Clone, Debug)]
pub(crate) struct ReferenceArithmeticDecoder<'a> {
    data: &'a [u8],
    bp: usize,
    c: u32,
    a: u32,
    ct: i32,
}

#[cfg(all(test, feature = "encode"))]
impl<'a> ReferenceArithmeticDecoder<'a> {
    pub(crate) fn new(data: &'a [u8]) -> Self {
        let mut dec = Self {
            data,
            bp: 0,
            c: 0,
            a: 0,
            ct: 0,
        };
        let b0 = dec.byte_at(0);
        dec.c = (b0 as u32) << 16;
        dec.byte_in();
        dec.c <<= 7;
        dec.ct -= 7;
        dec.a = 0x8000;
        dec
    }

    #[inline(always)]
    fn byte_at(&self, i: usize) -> u8 {
        match self.data.get(i) {
            Some(&b) => b,
            None => 0xFF,
        }
    }

    #[inline(always)]
    fn byte_in(&mut self) {
        if self.byte_at(self.bp) == 0xFF {
            let b1 = self.byte_at(self.bp + 1);
            if b1 > 0x8F {
                self.c = self.c.wrapping_add(0xFF00);
                self.ct = 8;
            } else {
                self.bp += 1;
                self.c = self.c.wrapping_add((b1 as u32) << 9);
                self.ct = 7;
            }
        } else {
            self.bp += 1;
            let b1 = self.byte_at(self.bp);
            self.c = self.c.wrapping_add((b1 as u32) << 8);
            self.ct = 8;
        }
    }

    #[inline(always)]
    fn renorm(&mut self) {
        loop {
            if self.ct == 0 {
                self.byte_in();
            }
            self.a <<= 1;
            self.c <<= 1;
            self.ct -= 1;
            if self.a & 0x8000 != 0 {
                break;
            }
        }
    }

    pub(crate) fn decode_bit_reference(&mut self, context: &mut MqContext) -> bool {
        let s = context.0 as usize;
        let entry = match MQ_STATES.get(s) {
            Some(e) => *e,
            None => return false,
        };
        let qe = entry.qe as u32;
        let mps = s >= 47;

        self.a = self.a.wrapping_sub(qe);

        let d;
        if (self.c >> 16) < qe {
            if self.a < qe {
                d = mps;
                context.0 = entry.next_mps;
            } else {
                d = !mps;
                context.0 = entry.next_lps;
            }
            self.a = qe;
            self.renorm();
        } else {
            self.c = self.c.wrapping_sub(qe << 16);
            if self.a & 0x8000 == 0 {
                if self.a < qe {
                    d = !mps;
                    context.0 = entry.next_lps;
                } else {
                    d = mps;
                    context.0 = entry.next_mps;
                }
                self.renorm();
            } else {
                d = mps;
            }
        }
        d
    }
}

#[cfg(all(test, feature = "encode"))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::encode::arith::Jbig2ArithCoder;

    /// Encode a sequence of (context-id, bit) decisions with the encoder, then
    /// decode them back and assert every bit matches. Contexts are a small
    /// dense array so the decoder and encoder agree on context identity.
    fn roundtrip(decisions: &[(usize, bool)], num_ctx: usize) {
        let mut enc = Jbig2ArithCoder::new();
        for &(ctx, bit) in decisions {
            enc.encode_bit(ctx, bit);
        }
        let data = enc.into_vec();

        let mut dec = ArithmeticDecoder::new(&data);
        let mut ctxs = vec![MqContext::default(); num_ctx];
        for (i, &(ctx, bit)) in decisions.iter().enumerate() {
            let got = dec.decode_bit(&mut ctxs[ctx]);
            assert_eq!(got, bit, "decision {i} (ctx {ctx}) mismatch");
        }
    }

    #[test]
    fn single_context_alternating() {
        let decisions: Vec<(usize, bool)> = (0..500).map(|i| (0usize, i % 2 == 0)).collect();
        roundtrip(&decisions, 1);
    }

    #[test]
    fn long_mps_run() {
        // A single context fed a long run of identical bits drives it deep into
        // the high-MPS states; the decoder must track the same trajectory.
        let decisions: Vec<(usize, bool)> = (0..4000).map(|_| (0usize, true)).collect();
        roundtrip(&decisions, 1);
        let decisions0: Vec<(usize, bool)> = (0..4000).map(|_| (0usize, false)).collect();
        roundtrip(&decisions0, 1);
    }

    #[test]
    fn lps_heavy_pseudorandom() {
        // Deterministic LCG producing an LPS-heavy, bursty pattern.
        let mut state: u32 = 0x1234_5678;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            state
        };
        let decisions: Vec<(usize, bool)> = (0..8000)
            .map(|_| {
                let r = next();
                let ctx = (r >> 3) as usize % 16;
                // Skew towards the less-probable symbol occasionally.
                let bit = (r & 0x7) == 0;
                (ctx, bit)
            })
            .collect();
        roundtrip(&decisions, 16);
    }

    #[test]
    fn stuffing_around_0xff() {
        // Force many carries so the encoder emits 0xFF stuff bytes, exercising
        // the decoder's BYTEIN stuffing branch.
        let mut decisions = Vec::new();
        for i in 0..3000u32 {
            // Mix of contexts and a bias that produces long carry chains.
            decisions.push(((i % 4) as usize, (i.count_ones() & 1) == 0));
        }
        roundtrip(&decisions, 4);
    }

    #[test]
    fn many_contexts() {
        let mut decisions = Vec::new();
        let mut s: u32 = 99;
        for _ in 0..5000 {
            s = s.wrapping_mul(48271) % 0x7FFF_FFFF;
            decisions.push(((s as usize) % 512, (s & 1) == 0));
        }
        roundtrip(&decisions, 512);
    }

    #[test]
    fn truncated_input_does_not_panic() {
        // Decoding past the end of a (deliberately short) buffer must terminate
        // by feeding 1-bits, never panic or loop.
        let data = [0x00u8, 0x11];
        let mut dec = ArithmeticDecoder::new(&data);
        let mut ctx = MqContext::default();
        for _ in 0..100_000 {
            let _ = dec.decode_bit(&mut ctx);
        }
    }

    #[test]
    fn empty_input_does_not_panic() {
        let data: [u8; 0] = [];
        let mut dec = ArithmeticDecoder::new(&data);
        let mut ctx = MqContext::default();
        for _ in 0..1000 {
            let _ = dec.decode_bit(&mut ctx);
        }
    }

    #[test]
    fn fast_decoder_matches_reference_on_adversarial_bytes() {
        let cases: &[&[u8]] = &[
            &[],
            &[0xFF],
            &[0xFF, 0x00],
            &[0xFF, 0x7F],
            &[0xFF, 0x80],
            &[0xFF, 0x8F],
            &[0xFF, 0x90],
            &[0xFF, 0xAC],
            &[0x00, 0xFF, 0x80, 0x00, 0xFF, 0x8F, 0x12, 0x34],
            &[0xFF, 0x80, 0xFF, 0x81, 0xFF, 0x8F, 0xFF, 0xFF, 0xAC],
            &[0x80, 0x00, 0x00, 0xFF, 0x8E, 0x55, 0xAA, 0xFF, 0x00],
        ];
        for (ci, data) in cases.iter().enumerate() {
            let mut fast = ArithmeticDecoder::new(data);
            let mut refd = ReferenceArithmeticDecoder::new(data);
            let mut cf = [MqContext::default(); 8];
            let mut cr = [MqContext::default(); 8];
            for step in 0..400usize {
                let idx = (step * 5 + ci) & 7;
                assert_eq!(
                    fast.decode_bit(&mut cf[idx]),
                    refd.decode_bit_reference(&mut cr[idx]),
                    "case {ci} step {step}"
                );
                assert_eq!(cf[idx], cr[idx], "case {ci} step {step} ctx");
            }
        }
    }
}
