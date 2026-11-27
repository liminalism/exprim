//! Fused, streaming inverse 9/7 synthesis for lattice-aligned tiles.
//!
//! The classic in-place inverse runs two sweeps per resolution level — every
//! row through the horizontal lifting, then every column band through the
//! vertical lifting — so a plane that does not fit in cache is read and
//! written twice per level. This module instead walks each level once: every
//! input row is interleaved and lifted horizontally as it enters an eight-row
//! ring, the four vertical lifting steps run pipelined behind it (a step for
//! row `y` as soon as its neighbours have reached the previous step), and each
//! finished row leaves the ring once. Output goes to a separate buffer, so the
//! subband rows still waiting to be read are never overwritten; levels
//! ping-pong between two buffers.
//!
//! [`LevelSynth`] exposes one level as a row producer. A consumer can take the
//! finest level's rows straight out of the ring — the packed-RGB path pairs
//! three components' producers with the colour transform and byte packing, so
//! the full-resolution planes are never materialised. A producer can also
//! start part-way down (`first_row`): output row `y` depends on input rows
//! `y - 4 ..= y + 4` only, so a pipeline primed four rows early yields the
//! same values as one run from the top, which makes row bands independent.
//!
//! The arithmetic is exactly the sequence of `irrev97`'s in-place path —
//! scale, then `x += c * (left + right)` per lifting step, each in IEEE
//! single precision with no fused multiply-add — so the reconstructed samples
//! are bit-identical to it. `tests` checks that for both the scalar and the
//! `wide` kernels across sizes, levels, degenerate axes and band starts.

use super::InverseDwtTiming;
use super::SplitSubbands;
use super::irrev97::{ALPHA, BETA, DELTA, GAMMA, INV_K, K};
use crate::simd::Kernel;
#[cfg(feature = "simd")]
use wide::f32x8;

/// Output row `y` is final once input row `y + LAG` has entered the ring.
const LAG: usize = 4;
/// Ring slots. Rows referenced at step `r` lie in `[r - 5, r]` (the mirrored
/// bottom neighbour reaches back furthest), so eight slots never alias.
const RING: usize = 8;

/// Sizes of the resolutions `0 ..= levels` (coarse to fine): `sizes[k]` is the
/// output of synthesis level `k`, `sizes[0]` the LL band.
fn resolution_sizes(width: usize, height: usize, levels: usize) -> Vec<(usize, usize)> {
    let mut sizes = Vec::with_capacity(levels + 1);
    let (mut w, mut h) = (width, height);
    sizes.push((w, h));
    for _ in 0..levels {
        w = w.div_ceil(2);
        h = h.div_ceil(2);
        sizes.push((w, h));
    }
    sizes.reverse();
    sizes
}

/// Inverse 9/7 synthesis of `data` (standard nested subband layout, origin on
/// the decomposition lattice) through `levels` levels. Returns the
/// reconstructed plane; `data` is consumed and may be returned as-is when
/// there is nothing to do.
#[allow(dead_code)]
pub(crate) fn inverse_97_2d_even(
    data: Vec<f32>,
    width: usize,
    height: usize,
    levels: u8,
    kernel: Kernel,
    timing: Option<&mut InverseDwtTiming>,
) -> Vec<f32> {
    inverse_97_2d_even_with_pool(
        data,
        width,
        height,
        levels,
        kernel,
        timing,
        &crate::decode::Jp2BufferPool::disabled(),
    )
}

pub(crate) fn inverse_97_2d_even_with_pool(
    data: Vec<f32>,
    width: usize,
    height: usize,
    levels: u8,
    kernel: Kernel,
    mut timing: Option<&mut InverseDwtTiming>,
    pool: &crate::decode::Jp2BufferPool,
) -> Vec<f32> {
    debug_assert_eq!(data.len(), width * height);
    if width == 0 || height == 0 || levels == 0 {
        return data;
    }
    let prefix = inverse_97_2d_even_but_last_with_pool(
        data,
        width,
        height,
        levels,
        kernel,
        timing.as_deref_mut(),
        pool,
    );
    let level_start = timing.as_ref().map(|_| std::time::Instant::now());
    let mut out = pool.take_for_overwrite_f32(width * height);
    let horizontal_ns = {
        let mut synth = prefix.final_level(kernel, 0);
        for y in 0..height {
            out[y * width..(y + 1) * width].copy_from_slice(synth.next_row());
        }
        synth.horizontal_ns
    };
    prefix.recycle(pool);
    if let (Some(timing), Some(start)) = (timing, level_start) {
        record_level(timing, start, horizontal_ns);
    }
    out
}

/// Inverse 9/7 synthesis from [`SplitSubbands<f32>`] to a full plane.
#[allow(dead_code)]
pub(crate) fn inverse_97_2d_even_split(
    split: SplitSubbands<f32>,
    levels: u8,
    kernel: Kernel,
    timing: Option<&mut InverseDwtTiming>,
) -> Vec<f32> {
    inverse_97_2d_even_split_with_pool(
        split,
        levels,
        kernel,
        timing,
        &crate::decode::Jp2BufferPool::disabled(),
    )
}

pub(crate) fn inverse_97_2d_even_split_with_pool(
    split: SplitSubbands<f32>,
    levels: u8,
    kernel: Kernel,
    mut timing: Option<&mut InverseDwtTiming>,
    pool: &crate::decode::Jp2BufferPool,
) -> Vec<f32> {
    let (width, height) = (split.width, split.height);
    if width == 0 || height == 0 || levels == 0 {
        pool.park_f32(split.hl1);
        pool.park_f32(split.lh_hh1);
        return split.coarse;
    }
    let prefix = inverse_97_2d_even_but_last_split_with_pool(
        split,
        levels,
        kernel,
        timing.as_deref_mut(),
        pool,
    );
    let level_start = timing.as_ref().map(|_| std::time::Instant::now());
    let mut out = pool.take_for_overwrite_f32(width * height);
    let horizontal_ns = {
        let mut synth = prefix.final_level(kernel, 0);
        for y in 0..height {
            out[y * width..(y + 1) * width].copy_from_slice(synth.next_row());
        }
        synth.horizontal_ns
    };
    prefix.recycle(pool);
    if let (Some(timing), Some(start)) = (timing, level_start) {
        record_level(timing, start, horizontal_ns);
    }
    out
}

fn record_level(timing: &mut InverseDwtTiming, start: std::time::Instant, horizontal_ns: u64) {
    let total = u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX);
    let horizontal = horizontal_ns.min(total);
    timing.horizontal_ns = timing.horizontal_ns.saturating_add(horizontal);
    timing.vertical_ns = timing.vertical_ns.saturating_add(total - horizontal);
    timing.level_ns.push(total);
}

enum PrefixBands {
    Monolithic {
        data: Vec<f32>,
        ll: Option<Vec<f32>>,
        ll_width: usize,
    },
    Split {
        ll1: Vec<f32>,
        hl1: Vec<f32>,
        lh_hh1: Vec<f32>,
    },
}

/// A plane with every synthesis level but the finest applied: the finest
/// level's LL input is `ll` (or the LL corner of `data` when there was only
/// one level) and its HL/LH/HH bands are still in `data` or in split subbands.
pub(crate) struct SynthesisPrefix {
    bands: PrefixBands,
    width: usize,
    height: usize,
}

/// Run all levels but the last. `levels` must be at least 1 and the plane
/// non-empty.
#[allow(dead_code)]
pub(crate) fn inverse_97_2d_even_but_last(
    data: Vec<f32>,
    width: usize,
    height: usize,
    levels: u8,
    kernel: Kernel,
    timing: Option<&mut InverseDwtTiming>,
) -> SynthesisPrefix {
    inverse_97_2d_even_but_last_with_pool(
        data,
        width,
        height,
        levels,
        kernel,
        timing,
        &crate::decode::Jp2BufferPool::disabled(),
    )
}

pub(crate) fn inverse_97_2d_even_but_last_with_pool(
    data: Vec<f32>,
    width: usize,
    height: usize,
    levels: u8,
    kernel: Kernel,
    mut timing: Option<&mut InverseDwtTiming>,
    pool: &crate::decode::Jp2BufferPool,
) -> SynthesisPrefix {
    debug_assert!(levels >= 1 && width > 0 && height > 0);
    debug_assert_eq!(data.len(), width * height);
    let levels = usize::from(levels);
    let sizes = resolution_sizes(width, height, levels);
    let mut prev: Option<Vec<f32>> = None;
    let mut spare: Vec<f32> = Vec::new();
    for level in 1..levels {
        let (rw, rh) = sizes[level];
        let need = rw * rh;
        let mut out = std::mem::take(&mut spare);
        if out.capacity() < need {
            pool.park_f32(out);
            out = pool.take_for_overwrite_f32(need);
        } else if out.len() < need {
            out.resize(need, 0.0);
        } else {
            out.truncate(need);
        }
        let source = match &prev {
            None => LlSource {
                rows: &data,
                stride: width,
            },
            Some(prev) => LlSource {
                rows: prev,
                stride: sizes[level - 1].0,
            },
        };
        let level_start = timing.as_ref().map(|_| std::time::Instant::now());
        let mut synth =
            LevelSynth::new_monolithic(source, &data, width, rw, rh, kernel, 0, timing.is_some());
        for y in 0..rh {
            out[y * rw..(y + 1) * rw].copy_from_slice(synth.next_row());
        }
        if let (Some(timing), Some(start)) = (timing.as_deref_mut(), level_start) {
            record_level(timing, start, synth.horizontal_ns);
        }
        spare = prev.replace(out).unwrap_or_default();
    }
    pool.park_f32(spare);
    let ll_width = sizes[levels - 1].0;
    SynthesisPrefix {
        bands: PrefixBands::Monolithic {
            data,
            ll: prev,
            ll_width,
        },
        width,
        height,
    }
}

/// Run all levels but the last on [`SplitSubbands<f32>`], consuming and
/// freeing `split.coarse` so retained prefix storage stays `width * height`
/// (`4 B/px`) instead of `5/4 * width * height` (`5 B/px`).
#[allow(dead_code)]
pub(crate) fn inverse_97_2d_even_but_last_split(
    split: SplitSubbands<f32>,
    levels: u8,
    kernel: Kernel,
    timing: Option<&mut InverseDwtTiming>,
) -> SynthesisPrefix {
    inverse_97_2d_even_but_last_split_with_pool(
        split,
        levels,
        kernel,
        timing,
        &crate::decode::Jp2BufferPool::disabled(),
    )
}

pub(crate) fn inverse_97_2d_even_but_last_split_with_pool(
    split: SplitSubbands<f32>,
    levels: u8,
    kernel: Kernel,
    timing: Option<&mut InverseDwtTiming>,
    pool: &crate::decode::Jp2BufferPool,
) -> SynthesisPrefix {
    debug_assert!(levels >= 1 && split.width > 0 && split.height > 0);
    let sn = split.width.div_ceil(2);
    let sv = split.height.div_ceil(2);
    let ll1 = if levels > 1 {
        inverse_97_2d_even_with_pool(split.coarse, sn, sv, levels - 1, kernel, timing, pool)
    } else {
        split.coarse
    };
    SynthesisPrefix {
        width: split.width,
        height: split.height,
        bands: PrefixBands::Split {
            ll1,
            hl1: split.hl1,
            lh_hh1: split.lh_hh1,
        },
    }
}

impl SynthesisPrefix {
    /// Return all owned subband planes to `pool` once synthesis has finished.
    pub(crate) fn recycle(self, pool: &crate::decode::Jp2BufferPool) {
        match self.bands {
            PrefixBands::Monolithic { data, ll, .. } => {
                if let Some(ll) = ll {
                    pool.park_f32(ll);
                }
                pool.park_f32(data);
            }
            PrefixBands::Split { ll1, hl1, lh_hh1 } => {
                pool.park_f32(ll1);
                pool.park_f32(hl1);
                pool.park_f32(lh_hh1);
            }
        }
    }
    /// A producer for the finest level's rows, starting at output row
    /// `first_row`.
    pub(crate) fn final_level(&self, kernel: Kernel, first_row: usize) -> LevelSynth<'_> {
        let (ll, hl, lh_hh) = self.final_slices();
        LevelSynth::new_split(
            ll,
            hl,
            lh_hh,
            self.width,
            self.height,
            kernel,
            first_row,
            false,
        )
    }

    /// Reuse an existing [`LevelSynthScratch`] across row bands so worker
    /// threads do not reallocate ring buffers per band.
    pub(crate) fn final_level_with_scratch<'a>(
        &'a self,
        kernel: Kernel,
        first_row: usize,
        scratch: &'a mut LevelSynthScratch,
    ) -> LevelSynthBorrowed<'a> {
        let (ll, hl, lh_hh) = self.final_slices();
        LevelSynthBorrowed::new(
            ll,
            hl,
            lh_hh,
            self.width,
            self.height,
            kernel,
            first_row,
            scratch,
        )
    }

    fn final_slices(&self) -> (LlSource<'_>, LlSource<'_>, LlSource<'_>) {
        let sn = self.width.div_ceil(2);
        let dn = self.width - sn;
        let sv = self.height.div_ceil(2);
        match &self.bands {
            PrefixBands::Monolithic { data, ll, ll_width } => {
                let source = match ll {
                    None => LlSource {
                        rows: data,
                        stride: self.width,
                    },
                    Some(ll) => LlSource {
                        rows: ll,
                        stride: *ll_width,
                    },
                };
                let hl = LlSource {
                    rows: if dn > 0 { &data[sn..] } else { &[] },
                    stride: self.width,
                };
                let lh_hh = LlSource {
                    rows: if sv < self.height {
                        &data[sv * self.width..]
                    } else {
                        &[]
                    },
                    stride: self.width,
                };
                (source, hl, lh_hh)
            }
            PrefixBands::Split { ll1, hl1, lh_hh1 } => (
                LlSource {
                    rows: ll1,
                    stride: sn,
                },
                LlSource {
                    rows: hl1,
                    stride: dn,
                },
                LlSource {
                    rows: lh_hh1,
                    stride: self.width,
                },
            ),
        }
    }

    /// Width of the finest level.
    pub(crate) fn width(&self) -> usize {
        self.width
    }

    /// Height of the finest level.
    pub(crate) fn height(&self) -> usize {
        self.height
    }
}

/// Where a level's subband rows come from.
#[derive(Clone, Copy)]
struct LlSource<'a> {
    rows: &'a [f32],
    stride: usize,
}

/// Reusable scratch buffers for [`LevelSynthBorrowed`].
#[derive(Default)]
pub(crate) struct LevelSynthScratch {
    lo: Vec<f32>,
    hi: Vec<f32>,
    ring: Vec<f32>,
}

impl LevelSynthScratch {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    fn prepare(&mut self, rw: usize) {
        let sn = rw.div_ceil(2);
        let dn = rw - sn;
        if self.lo.len() < sn + 2 {
            self.lo.resize(sn + 2, 0.0);
        }
        if self.hi.len() < dn + 2 {
            self.hi.resize(dn + 2, 0.0);
        }
        if self.ring.len() < rw * RING {
            self.ring.resize(rw * RING, 0.0);
        }
    }
}

/// One synthesis level as a row producer: `next_row` returns the output rows
/// in order from `first_row`.
pub(crate) struct LevelSynth<'a> {
    source: LlSource<'a>,
    hl: LlSource<'a>,
    lh_hh: LlSource<'a>,
    rw: usize,
    rh: usize,
    /// Low-pass sample count horizontally.
    sn: usize,
    kernel: Kernel,
    scratch: LevelSynthScratch,
    /// Next pipeline step (the input row it loads).
    step: usize,
    /// Next output row `next_row` returns.
    next: usize,
    /// Time spent in the horizontal synthesis, when timing was requested.
    timed: bool,
    horizontal_ns: u64,
}

impl<'a> LevelSynth<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new_monolithic(
        source: LlSource<'a>,
        bands: &'a [f32],
        band_stride: usize,
        rw: usize,
        rh: usize,
        kernel: Kernel,
        first_row: usize,
        timed: bool,
    ) -> Self {
        let sn = rw.div_ceil(2);
        let dn = rw - sn;
        let sv = rh.div_ceil(2);
        let hl = LlSource {
            rows: if dn > 0 { &bands[sn..] } else { &[] },
            stride: band_stride,
        };
        let lh_hh = LlSource {
            rows: if sv < rh {
                &bands[sv * band_stride..]
            } else {
                &[]
            },
            stride: band_stride,
        };
        Self::new_split(source, hl, lh_hh, rw, rh, kernel, first_row, timed)
    }

    #[allow(clippy::too_many_arguments)]
    fn new_split(
        source: LlSource<'a>,
        hl: LlSource<'a>,
        lh_hh: LlSource<'a>,
        rw: usize,
        rh: usize,
        kernel: Kernel,
        first_row: usize,
        timed: bool,
    ) -> Self {
        debug_assert!(rw > 0 && rh > 0 && first_row < rh);
        let sn = rw.div_ceil(2);
        let mut scratch = LevelSynthScratch::new();
        scratch.prepare(rw);
        Self {
            source,
            hl,
            lh_hh,
            rw,
            rh,
            sn,
            kernel,
            scratch,
            step: first_row.saturating_sub(LAG),
            next: first_row,
            timed,
            horizontal_ns: 0,
        }
    }

    /// The next output row, as a slice into the ring (valid until the next
    /// call).
    pub(crate) fn next_row(&mut self) -> &[f32] {
        step_next_row_97(
            self.source,
            self.hl,
            self.lh_hh,
            self.rw,
            self.rh,
            self.sn,
            self.kernel,
            &mut self.scratch,
            &mut self.step,
            &mut self.next,
            self.timed,
            &mut self.horizontal_ns,
        )
    }
}

/// Borrowed-scratch synthesis producer for multi-band workers.
pub(crate) struct LevelSynthBorrowed<'a> {
    source: LlSource<'a>,
    hl: LlSource<'a>,
    lh_hh: LlSource<'a>,
    rw: usize,
    rh: usize,
    sn: usize,
    kernel: Kernel,
    scratch: &'a mut LevelSynthScratch,
    step: usize,
    next: usize,
    horizontal_ns: u64,
}

impl<'a> LevelSynthBorrowed<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        source: LlSource<'a>,
        hl: LlSource<'a>,
        lh_hh: LlSource<'a>,
        rw: usize,
        rh: usize,
        kernel: Kernel,
        first_row: usize,
        scratch: &'a mut LevelSynthScratch,
    ) -> Self {
        debug_assert!(rw > 0 && rh > 0 && first_row < rh);
        scratch.prepare(rw);
        Self {
            source,
            hl,
            lh_hh,
            rw,
            rh,
            sn: rw.div_ceil(2),
            kernel,
            scratch,
            step: first_row.saturating_sub(LAG),
            next: first_row,
            horizontal_ns: 0,
        }
    }

    pub(crate) fn next_row(&mut self) -> &[f32] {
        step_next_row_97(
            self.source,
            self.hl,
            self.lh_hh,
            self.rw,
            self.rh,
            self.sn,
            self.kernel,
            self.scratch,
            &mut self.step,
            &mut self.next,
            false,
            &mut self.horizontal_ns,
        )
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn step_next_row_97<'s>(
    source: LlSource<'_>,
    hl: LlSource<'_>,
    lh_hh: LlSource<'_>,
    rw: usize,
    rh: usize,
    sn: usize,
    kernel: Kernel,
    scratch: &'s mut LevelSynthScratch,
    step: &mut usize,
    next: &mut usize,
    timed: bool,
    horizontal_ns: &mut u64,
) -> &'s [f32] {
    debug_assert!(*next < rh);
    let y = *next;
    *next += 1;
    if rh < 2 {
        // A one-row resolution is an identity vertically (no scaling
        // either, see `irrev97::inverse_97_vertical_in_place`).
        let (lo_row, hi_row) = input_row_slices_97(source, hl, lh_hh, rw, sn, 0);
        let row_start = timed.then(std::time::Instant::now);
        synth_row(
            lo_row,
            hi_row,
            &mut scratch.ring[..rw],
            1.0,
            &mut scratch.lo,
            &mut scratch.hi,
            kernel,
        );
        if let Some(start) = row_start {
            *horizontal_ns = horizontal_ns
                .saturating_add(u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX));
        }
        return &scratch.ring[..rw];
    }
    // Row `y` is final once step `y + LAG` has run.
    while *step <= y + LAG {
        advance_step_97(
            source,
            hl,
            lh_hh,
            rw,
            rh,
            sn,
            kernel,
            scratch,
            step,
            timed,
            horizontal_ns,
        );
    }
    let slot = (y % RING) * rw;
    &scratch.ring[slot..slot + rw]
}

#[inline]
fn input_row_slices_97<'a>(
    source: LlSource<'a>,
    hl: LlSource<'a>,
    lh_hh: LlSource<'a>,
    rw: usize,
    sn: usize,
    r: usize,
) -> (&'a [f32], &'a [f32]) {
    let dn = rw - sn;
    if r.is_multiple_of(2) {
        let i = r / 2;
        let lo = &source.rows[i * source.stride..i * source.stride + sn];
        let hi = if dn > 0 {
            &hl.rows[i * hl.stride..i * hl.stride + dn]
        } else {
            &[]
        };
        (lo, hi)
    } else {
        let j = (r - 1) / 2;
        let base = j * lh_hh.stride;
        (
            &lh_hh.rows[base..base + sn],
            &lh_hh.rows[base + sn..base + rw],
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn advance_step_97(
    source: LlSource<'_>,
    hl: LlSource<'_>,
    lh_hh: LlSource<'_>,
    rw: usize,
    rh: usize,
    sn: usize,
    kernel: Kernel,
    scratch: &mut LevelSynthScratch,
    step: &mut usize,
    timed: bool,
    horizontal_ns: &mut u64,
) {
    let r = *step;
    *step += 1;
    if r < rh {
        let (lo_row, hi_row) = input_row_slices_97(source, hl, lh_hh, rw, sn, r);
        let row_scale = if r.is_multiple_of(2) { K } else { INV_K };
        let row_start = timed.then(std::time::Instant::now);
        let slot = (r % RING) * rw;
        synth_row(
            lo_row,
            hi_row,
            &mut scratch.ring[slot..slot + rw],
            row_scale,
            &mut scratch.lo,
            &mut scratch.hi,
            kernel,
        );
        if let Some(start) = row_start {
            *horizontal_ns = horizontal_ns
                .saturating_add(u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX));
        }
    }
    let mirror = |x: isize| -> usize {
        if x < 0 {
            (-x) as usize
        } else if x as usize >= rh {
            2 * (rh - 1) - x as usize
        } else {
            x as usize
        }
    };
    let r = r as isize;
    for (back, parity, coeff) in [
        (1isize, 0usize, -DELTA),
        (2, 1, -GAMMA),
        (3, 0, -BETA),
        (4, 1, -ALPHA),
    ] {
        let x = r - back;
        if x < 0 || x as usize >= rh || (x as usize) % 2 != parity {
            continue;
        }
        let target = (x as usize % RING) * rw;
        let above = (mirror(x - 1) % RING) * rw;
        let below = (mirror(x + 1) % RING) * rw;
        lift_ring_row(&mut scratch.ring, rw, target, above, below, coeff, kernel);
    }
}

/// `ring[target] += coeff * (ring[above] + ring[below])` over `rw` samples.
/// The target row always has the opposite parity of its neighbours, so the
/// three slots are distinct unless `above == below` (a mirrored boundary).
fn lift_ring_row(
    ring: &mut [f32],
    rw: usize,
    target: usize,
    above: usize,
    below: usize,
    coeff: f32,
    kernel: Kernel,
) {
    debug_assert_ne!(target, above);
    debug_assert_ne!(target, below);
    if above == below {
        let [t, n] = ring
            .get_disjoint_mut([target..target + rw, above..above + rw])
            .expect("target and mirrored neighbour rows are disjoint ring slots");
        lift_row(t, n, n, coeff, kernel);
    } else {
        let [t, a, b] = ring
            .get_disjoint_mut([target..target + rw, above..above + rw, below..below + rw])
            .expect("target and neighbour rows are disjoint ring slots");
        lift_row(t, a, b, coeff, kernel);
    }
}

/// Horizontal 1-D synthesis of one even-origin row from its deinterleaved
/// low-pass (`lo_in`, `ceil(n/2)` samples) and high-pass (`hi_in`) halves,
/// interleaved into `out` and multiplied by `row_scale` (the vertical scaling
/// step, applied here so the row is written once).
///
/// Lifting runs on the two halves while they are still separate, in padded
/// scratch rows whose guard samples carry the whole-sample symmetric
/// extension (`X[-1] = X[1]`, `X[n] = X[n-2]`), then the halves are
/// interleaved. Per sample this is the same `*= K` / `*= 1/K`, then
/// `x += c * (left + right)` four times, as `irrev97`'s in-place row path.
fn synth_row(
    lo_in: &[f32],
    hi_in: &[f32],
    out: &mut [f32],
    row_scale: f32,
    lo: &mut [f32],
    hi: &mut [f32],
    kernel: Kernel,
) {
    let sn = lo_in.len();
    let dn = hi_in.len();
    let n = sn + dn;
    debug_assert_eq!(out.len(), n);
    if n < 2 {
        // Even-origin length-one signal: the horizontal synthesis is an identity.
        if n == 1 {
            out[0] = lo_in[0] * row_scale;
        }
        return;
    }
    debug_assert!(lo.len() >= sn + 2 && hi.len() >= dn + 2);
    let lo = &mut lo[..sn + 2];
    let hi = &mut hi[..dn + 2];
    scale_into(&mut lo[1..sn + 1], lo_in, K, kernel);
    scale_into(&mut hi[1..dn + 1], hi_in, INV_K, kernel);

    // Even samples: X[2i] += c * (X[2i-1] + X[2i+1]) = c * (H[i-1] + H[i]),
    // with H[-1] = H[0] and, for odd n, H[sn-1] = H[dn-1].
    // Odd samples: X[2i+1] += c * (L[i] + L[i+1]), with L[sn] = L[sn-1] for
    // even n.
    hi[0] = hi[1];
    hi[dn + 1] = hi[dn];
    lift_row(
        &mut lo[1..sn + 1],
        &hi[0..sn],
        &hi[1..sn + 1],
        -DELTA,
        kernel,
    );
    lo[sn + 1] = lo[sn];
    lift_row(
        &mut hi[1..dn + 1],
        &lo[1..dn + 1],
        &lo[2..dn + 2],
        -GAMMA,
        kernel,
    );
    hi[0] = hi[1];
    hi[dn + 1] = hi[dn];
    lift_row(
        &mut lo[1..sn + 1],
        &hi[0..sn],
        &hi[1..sn + 1],
        -BETA,
        kernel,
    );
    lo[sn + 1] = lo[sn];
    lift_row(
        &mut hi[1..dn + 1],
        &lo[1..dn + 1],
        &lo[2..dn + 2],
        -ALPHA,
        kernel,
    );

    interleave_scaled(&lo[1..sn + 1], &hi[1..dn + 1], out, row_scale, kernel);
}

/// `target[i] += coeff * (left[i] + right[i])`.
#[inline]
fn lift_row(target: &mut [f32], left: &[f32], right: &[f32], coeff: f32, kernel: Kernel) {
    match kernel {
        Kernel::Scalar => super::irrev97::lift_row(target, left, right, coeff, false),
        Kernel::Wide => super::irrev97::lift_row(target, left, right, coeff, true),
        Kernel::Avx2 => avx2::lift_row(target, left, right, coeff),
    }
}

/// `out[i] = input[i] * scale`.
fn scale_into(out: &mut [f32], input: &[f32], scale: f32, kernel: Kernel) {
    debug_assert_eq!(out.len(), input.len());
    if kernel == Kernel::Avx2 {
        avx2::scale_into(out, input, scale);
        return;
    }
    let mut i = 0usize;
    #[cfg(feature = "simd")]
    if kernel == Kernel::Wide {
        let scale_v = f32x8::splat(scale);
        while i + 8 <= out.len() {
            let v = f32x8::new(input[i..i + 8].try_into().expect("8 lanes")) * scale_v;
            out[i..i + 8].copy_from_slice(&v.to_array());
            i += 8;
        }
    }
    let _ = kernel;
    while i < out.len() {
        out[i] = input[i] * scale;
        i += 1;
    }
}

/// `out[2i] = lo[i] * scale`, `out[2i+1] = hi[i] * scale`, with the odd
/// trailing low-pass sample when `out` has odd length.
fn interleave_scaled(lo: &[f32], hi: &[f32], out: &mut [f32], scale: f32, kernel: Kernel) {
    let dn = hi.len();
    debug_assert_eq!(out.len(), lo.len() + dn);
    if kernel == Kernel::Avx2 {
        avx2::interleave_scaled(lo, hi, out, scale);
        return;
    }
    let mut i = 0usize;
    #[cfg(feature = "simd")]
    if kernel == Kernel::Wide {
        let scale_v = f32x8::splat(scale);
        while i + 8 <= dn {
            let l = (f32x8::new(lo[i..i + 8].try_into().expect("8 lanes")) * scale_v).to_array();
            let h = (f32x8::new(hi[i..i + 8].try_into().expect("8 lanes")) * scale_v).to_array();
            let dst = &mut out[2 * i..2 * i + 16];
            for k in 0..8 {
                dst[2 * k] = l[k];
                dst[2 * k + 1] = h[k];
            }
            i += 8;
        }
    }
    let _ = kernel;
    while i < dn {
        out[2 * i] = lo[i] * scale;
        out[2 * i + 1] = hi[i] * scale;
        i += 1;
    }
    if lo.len() > dn {
        out[2 * dn] = lo[dn] * scale;
    }
}

/// The row kernels compiled with AVX2 enabled: plain loops that LLVM
/// vectorises to 256-bit lanes, with the same per-sample operation order as
/// the scalar and `wide` paths (no fused multiply-add: only `avx2` is
/// enabled), so the results are bit-identical.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx2 {
    #[target_feature(enable = "avx2")]
    unsafe fn lift_row_impl(target: &mut [f32], left: &[f32], right: &[f32], coeff: f32) {
        let n = target.len().min(left.len()).min(right.len());
        let (target, left, right) = (&mut target[..n], &left[..n], &right[..n]);
        for x in 0..n {
            target[x] += coeff * (left[x] + right[x]);
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn scale_into_impl(out: &mut [f32], input: &[f32], scale: f32) {
        let n = out.len().min(input.len());
        let (out, input) = (&mut out[..n], &input[..n]);
        for x in 0..n {
            out[x] = input[x] * scale;
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn interleave_scaled_impl(lo: &[f32], hi: &[f32], out: &mut [f32], scale: f32) {
        let dn = hi.len().min(lo.len());
        let (pairs, tail) = out.split_at_mut((2 * dn).min(out.len()));
        for (i, pair) in pairs.chunks_exact_mut(2).enumerate() {
            pair[0] = lo[i] * scale;
            pair[1] = hi[i] * scale;
        }
        if let (Some(last), Some(&value)) = (tail.first_mut(), lo.get(dn)) {
            *last = value * scale;
        }
    }

    // SAFETY (all three): `Kernel::Avx2` is only selected after runtime
    // detection of AVX2 (`simd::x86`).
    #[inline]
    pub(super) fn lift_row(target: &mut [f32], left: &[f32], right: &[f32], coeff: f32) {
        unsafe { lift_row_impl(target, left, right, coeff) }
    }

    #[inline]
    pub(super) fn scale_into(out: &mut [f32], input: &[f32], scale: f32) {
        unsafe { scale_into_impl(out, input, scale) }
    }

    #[inline]
    pub(super) fn interleave_scaled(lo: &[f32], hi: &[f32], out: &mut [f32], scale: f32) {
        unsafe { interleave_scaled_impl(lo, hi, out, scale) }
    }
}

/// Without x86 the AVX2 flavour is never selected; route to scalar so the
/// match arms stay total.
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
mod avx2 {
    pub(super) fn lift_row(target: &mut [f32], left: &[f32], right: &[f32], coeff: f32) {
        super::super::irrev97::lift_row(target, left, right, coeff, false)
    }

    pub(super) fn scale_into(out: &mut [f32], input: &[f32], scale: f32) {
        super::scale_into(out, input, scale, crate::simd::Kernel::Scalar)
    }

    pub(super) fn interleave_scaled(lo: &[f32], hi: &[f32], out: &mut [f32], scale: f32) {
        super::interleave_scaled(lo, hi, out, scale, crate::simd::Kernel::Scalar)
    }
}

#[cfg(test)]
mod tests {
    use super::{inverse_97_2d_even, inverse_97_2d_even_but_last};
    use crate::simd::Kernel;

    fn pattern(width: usize, height: usize, seed: u32) -> Vec<f32> {
        let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(12345);
        (0..width * height)
            .map(|i| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                let noise = (state % 2001) as f32 / 8.0 - 125.0;
                let smooth = ((i % width) as f32 * 0.31).sin() * 60.0
                    + ((i / width) as f32 * 0.17).cos() * 40.0;
                smooth + noise
            })
            .collect()
    }

    fn reference(width: usize, height: usize, levels: u8, seed: u32) -> (Vec<f32>, Vec<f32>) {
        let original = pattern(width, height, seed);
        let mut reference = original.clone();
        super::super::irrev97::inverse_97_2d_in_place_reference(
            &mut reference,
            width,
            height,
            levels,
            false,
        );
        (original, reference)
    }

    fn assert_bits(streamed: &[f32], reference: &[f32], width: usize, what: &str) {
        assert_eq!(streamed.len(), reference.len());
        for (i, (a, b)) in streamed.iter().zip(reference).enumerate() {
            assert!(
                a.to_bits() == b.to_bits(),
                "{what}: sample {i} ({}, {}) {a} != {b}",
                i % width,
                i / width
            );
        }
    }

    fn check(width: usize, height: usize, levels: u8, kernel: Kernel) {
        let seed = (width * 131 + height * 7) as u32 + levels as u32;
        let (original, reference) = reference(width, height, levels, seed);
        let streamed = inverse_97_2d_even(original.clone(), width, height, levels, kernel, None);
        assert_bits(
            &streamed,
            &reference,
            width,
            &format!("{width}x{height} levels={levels} {kernel:?}"),
        );
        if levels >= 1 {
            let sn = width.div_ceil(2);
            let dn = width - sn;
            let sv = height.div_ceil(2);
            let dv = height - sv;
            let mut coarse = vec![0.0f32; sn * sv];
            let mut hl1 = vec![0.0f32; dn * sv];
            let mut lh_hh1 = vec![0.0f32; width * dv];
            for y in 0..sv {
                coarse[y * sn..(y + 1) * sn].copy_from_slice(&original[y * width..y * width + sn]);
                if dn > 0 {
                    hl1[y * dn..(y + 1) * dn]
                        .copy_from_slice(&original[y * width + sn..(y + 1) * width]);
                }
            }
            if dv > 0 {
                lh_hh1.copy_from_slice(&original[sv * width..]);
            }
            let split = crate::dwt::SplitSubbands {
                width,
                height,
                coarse,
                hl1,
                lh_hh1,
            };
            let split_out = super::inverse_97_2d_even_split(split, levels, kernel, None);
            assert_bits(
                &split_out,
                &reference,
                width,
                &format!("split {width}x{height} levels={levels} {kernel:?}"),
            );
        }
    }

    /// Row bands started part-way down must reproduce the full run exactly.
    fn check_banded(width: usize, height: usize, levels: u8, kernel: Kernel, first_row: usize) {
        let seed = (width * 17 + height * 3) as u32 + levels as u32 + first_row as u32;
        let (original, reference) = reference(width, height, levels, seed);
        let prefix = inverse_97_2d_even_but_last(original, width, height, levels, kernel, None);
        let mut synth = prefix.final_level(kernel, first_row);
        for y in first_row..height {
            let row = synth.next_row();
            assert_bits(
                row,
                &reference[y * width..(y + 1) * width],
                width,
                &format!(
                    "{width}x{height} levels={levels} {kernel:?} band from {first_row}, row {y}"
                ),
            );
        }
    }

    #[test]
    fn streaming_matches_in_place_reference_small_sizes() {
        let kernels = Kernel::available();
        for height in 1..=19usize {
            for width in 1..=19usize {
                for levels in 0..=4u8 {
                    for &kernel in &kernels {
                        check(width, height, levels, kernel);
                    }
                }
            }
        }
    }

    #[test]
    fn streaming_matches_in_place_reference_larger_sizes() {
        let kernels = Kernel::available();
        for &(width, height, levels) in &[
            (97usize, 61usize, 5u8),
            (128, 128, 5),
            (257, 33, 3),
            (33, 257, 3),
            (300, 211, 5),
            (1, 300, 5),
            (300, 1, 5),
            (2, 2, 1),
        ] {
            for &kernel in &kernels {
                check(width, height, levels, kernel);
            }
        }
    }

    #[test]
    fn banded_final_level_matches_full_run() {
        let kernels = Kernel::available();
        for &(width, height, levels) in
            &[(37usize, 29usize, 1u8), (64, 50, 3), (13, 2, 2), (5, 9, 1)]
        {
            for first_row in 0..height {
                for &kernel in &kernels {
                    check_banded(width, height, levels, kernel, first_row);
                }
            }
        }
    }

    #[test]
    fn profiled_timing_reports_every_level() {
        let data = pattern(64, 48, 9);
        let mut timing = super::InverseDwtTiming::default();
        let out = inverse_97_2d_even(data, 64, 48, 3, Kernel::Scalar, Some(&mut timing));
        assert_eq!(out.len(), 64 * 48);
        assert_eq!(timing.level_ns.len(), 3);
    }
}
