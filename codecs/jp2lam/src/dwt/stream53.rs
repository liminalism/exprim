//! Fused, streaming inverse reversible 5/3 synthesis for lattice-aligned tiles.
//!
//! Mirrors [`super::stream97`] for the reversible 5/3 wavelet (ISO/IEC 15444-1
//! Annex F.3.4 / F.3.7). The classic in-place inverse sweeps each level twice
//! (horizontal row lifting with three row-sized scratch copies, then vertical
//! 256-column strip transposes) and scans the entire plane beforehand to verify
//! `i32` headroom.
//!
//! Because reversible 5/3 synthesis has only two lifting stages —
//! `even = low - ((high_left + high_right + 2) >> 2)` followed by
//! `odd = high + ((even_left + even_right) >> 1)` — output row `y` depends only
//! on input rows `y - 2 ..= y + 2` (`LAG = 2`), and a four-row ring (`RING = 4`)
//! never aliases. Each input row is horizontally lifted and interleaved once as
//! it enters the ring, the two vertical lifting steps run pipelined behind it,
//! and finished rows stream directly into the next level or into the fused
//! inverse-RCT byte packer without materializing full-resolution `i32` planes.
//!
//! Runtime `#[target_feature(enable = "avx2")]` row kernels use `std::arch`
//! directly, so builds with `default-features = false` (such as `exprim`) still
//! run 256-bit SIMD lifting and interleaving. Every kernel reproduces
//! `rev53::inverse_53_2d_in_place` bit-for-bit.

use super::InverseDwtTiming;
use super::SplitSubbands;
use crate::simd::Kernel;
#[cfg(feature = "simd")]
use wide::i32x8;

/// Output row `y` is final once input row `y + LAG` has entered the ring.
const LAG: usize = 2;
/// Ring slots. Active rows at step `r` span `[r - 3, r]`, and target rows
/// always have opposite parity from their mirrored vertical neighbours, so four
/// slots never alias.
const RING: usize = 4;

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

/// True when `max_abs` (the largest `|coeff|` in the subband plane) is small
/// enough that every intermediate sum across `levels` 2-D reversible 5/3
/// synthesis levels — plus an optional inverse RCT (`db + dr`, `dr + g`,
/// `db + g`) — stays strictly inside signed `i32` without wrapping.
pub(crate) fn i32_lifting_is_safe(max_abs: u32, levels: u8, with_rct: bool) -> bool {
    let mut bound = i128::from(max_abs);
    let limit = i128::from(i32::MAX);
    for _ in 0..usize::from(levels).saturating_mul(2) {
        // Horizontal and vertical undo-update compute `left + right + 2` in i32.
        if bound * 2 + 2 > limit {
            return false;
        }
        let updated = bound + ((bound * 2 + 2) >> 2);
        // Horizontal and vertical undo-predict compute `even_l + even_r` in i32.
        if updated * 2 > limit {
            return false;
        }
        let predicted = bound + updated;
        bound = updated.max(predicted);
        if bound > limit {
            return false;
        }
    }
    if with_rct {
        // Inverse RCT computes `db + dr`, `g = y - ((db + dr) >> 2)`, and
        // `r = dr + g` / `b = db + g` in i32.
        if bound * 2 > limit {
            return false;
        }
        let max_g = bound + (bound >> 1) + 1;
        if bound + max_g > limit {
            return false;
        }
    }
    true
}

/// Streaming inverse 5/3 synthesis of `data` (standard nested subband layout,
/// even lattice origin) through `levels` levels. Caller must ensure
/// [`i32_lifting_is_safe`] holds for `data`.
pub(crate) fn inverse_53_2d_even(
    data: Vec<i32>,
    width: usize,
    height: usize,
    levels: u8,
    kernel: Kernel,
    mut timing: Option<&mut InverseDwtTiming>,
) -> Vec<i32> {
    debug_assert_eq!(data.len(), width * height);
    if width == 0 || height == 0 || levels == 0 {
        return data;
    }
    let prefix =
        inverse_53_2d_even_but_last(data, width, height, levels, kernel, timing.as_deref_mut());
    let level_start = timing.as_ref().map(|_| std::time::Instant::now());
    let mut synth = prefix.final_level(kernel, 0);
    let mut out = vec![0i32; width * height];
    for y in 0..height {
        out[y * width..(y + 1) * width].copy_from_slice(synth.next_row());
    }
    if let (Some(timing), Some(start)) = (timing, level_start) {
        record_level(timing, start, synth.horizontal_ns);
    }
    out
}

/// Streaming inverse 5/3 synthesis from [`SplitSubbands<i32>`] to a full plane.
pub(crate) fn inverse_53_2d_even_split(
    split: SplitSubbands<i32>,
    levels: u8,
    kernel: Kernel,
    mut timing: Option<&mut InverseDwtTiming>,
) -> Vec<i32> {
    let (width, height) = (split.width, split.height);
    if width == 0 || height == 0 || levels == 0 {
        return split.coarse;
    }
    let prefix = inverse_53_2d_even_but_last_split(split, levels, kernel, timing.as_deref_mut());
    let level_start = timing.as_ref().map(|_| std::time::Instant::now());
    let mut synth = prefix.final_level(kernel, 0);
    let mut out = vec![0i32; width * height];
    for y in 0..height {
        out[y * width..(y + 1) * width].copy_from_slice(synth.next_row());
    }
    if let (Some(timing), Some(start)) = (timing, level_start) {
        record_level(timing, start, synth.horizontal_ns);
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

enum PrefixBands53 {
    Monolithic {
        data: Vec<i32>,
        ll: Option<Vec<i32>>,
        ll_width: usize,
    },
    Split {
        ll1: Vec<i32>,
        hl1: Vec<i32>,
        lh_hh1: Vec<i32>,
    },
}

/// A reversible 5/3 plane with every synthesis level but the finest applied.
pub(crate) struct SynthesisPrefix53 {
    bands: PrefixBands53,
    width: usize,
    height: usize,
}

/// Run all levels but the last on a monolithic nested-subband buffer.
pub(crate) fn inverse_53_2d_even_but_last(
    data: Vec<i32>,
    width: usize,
    height: usize,
    levels: u8,
    kernel: Kernel,
    mut timing: Option<&mut InverseDwtTiming>,
) -> SynthesisPrefix53 {
    debug_assert!(levels >= 1 && width > 0 && height > 0);
    debug_assert_eq!(data.len(), width * height);
    let levels = usize::from(levels);
    let sizes = resolution_sizes(width, height, levels);
    let mut prev: Option<Vec<i32>> = None;
    let mut spare: Vec<i32> = Vec::new();
    for level in 1..levels {
        let (rw, rh) = sizes[level];
        let source = match &prev {
            None => RowSlice {
                rows: &data,
                stride: width,
            },
            Some(prev) => RowSlice {
                rows: prev,
                stride: sizes[level - 1].0,
            },
        };
        let level_start = timing.as_ref().map(|_| std::time::Instant::now());
        let mut out = std::mem::take(&mut spare);
        if out.len() < rw * rh {
            out.resize(rw * rh, 0);
        }
        let mut synth =
            LevelSynth53::new_monolithic(source, &data, width, rw, rh, kernel, 0, timing.is_some());
        for y in 0..rh {
            out[y * rw..(y + 1) * rw].copy_from_slice(synth.next_row());
        }
        if let (Some(timing), Some(start)) = (timing.as_deref_mut(), level_start) {
            record_level(timing, start, synth.horizontal_ns);
        }
        spare = prev.replace(out).unwrap_or_default();
    }
    let ll_width = sizes[levels - 1].0;
    SynthesisPrefix53 {
        bands: PrefixBands53::Monolithic {
            data,
            ll: prev,
            ll_width,
        },
        width,
        height,
    }
}

/// Run all levels but the last on [`SplitSubbands<i32>`], consuming and freeing
/// the coarse subband buffer so total retained storage stays `width * height`
/// samples (`4 B/px`) rather than `5/4 * width * height` (`5 B/px`).
pub(crate) fn inverse_53_2d_even_but_last_split(
    split: SplitSubbands<i32>,
    levels: u8,
    kernel: Kernel,
    timing: Option<&mut InverseDwtTiming>,
) -> SynthesisPrefix53 {
    debug_assert!(levels >= 1 && split.width > 0 && split.height > 0);
    let sn = split.width.div_ceil(2);
    let sv = split.height.div_ceil(2);
    let ll1 = if levels > 1 {
        inverse_53_2d_even(split.coarse, sn, sv, levels - 1, kernel, timing)
    } else {
        split.coarse
    };
    SynthesisPrefix53 {
        width: split.width,
        height: split.height,
        bands: PrefixBands53::Split {
            ll1,
            hl1: split.hl1,
            lh_hh1: split.lh_hh1,
        },
    }
}

impl SynthesisPrefix53 {
    /// A producer for the finest level's rows, starting at output row `first_row`.
    pub(crate) fn final_level(&self, kernel: Kernel, first_row: usize) -> LevelSynth53<'_> {
        match &self.bands {
            PrefixBands53::Monolithic { data, ll, ll_width } => {
                let source = match ll {
                    None => RowSlice {
                        rows: data,
                        stride: self.width,
                    },
                    Some(ll) => RowSlice {
                        rows: ll,
                        stride: *ll_width,
                    },
                };
                LevelSynth53::new_monolithic(
                    source,
                    data,
                    self.width,
                    self.width,
                    self.height,
                    kernel,
                    first_row,
                    false,
                )
            }
            PrefixBands53::Split { ll1, hl1, lh_hh1 } => {
                let sn = self.width.div_ceil(2);
                let dn = self.width - sn;
                LevelSynth53::new_split(
                    RowSlice {
                        rows: ll1,
                        stride: sn,
                    },
                    RowSlice {
                        rows: hl1,
                        stride: dn,
                    },
                    RowSlice {
                        rows: lh_hh1,
                        stride: self.width,
                    },
                    self.width,
                    self.height,
                    kernel,
                    first_row,
                    false,
                )
            }
        }
    }

    /// Reuse an existing [`LevelSynth53Scratch`] across bands so worker threads
    /// do not reallocate ring buffers per band.
    pub(crate) fn final_level_with_scratch<'a>(
        &'a self,
        kernel: Kernel,
        first_row: usize,
        scratch: &'a mut LevelSynth53Scratch,
    ) -> LevelSynth53Borrowed<'a> {
        let (ll, hl, lh_hh) = self.final_slices();
        LevelSynth53Borrowed::new(
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

    fn final_slices(&self) -> (RowSlice<'_>, RowSlice<'_>, RowSlice<'_>) {
        let sn = self.width.div_ceil(2);
        let dn = self.width - sn;
        let sv = self.height.div_ceil(2);
        match &self.bands {
            PrefixBands53::Monolithic { data, ll, ll_width } => {
                let source = match ll {
                    None => RowSlice {
                        rows: data,
                        stride: self.width,
                    },
                    Some(ll) => RowSlice {
                        rows: ll,
                        stride: *ll_width,
                    },
                };
                let hl = RowSlice {
                    rows: if dn > 0 { &data[sn..] } else { &[] },
                    stride: self.width,
                };
                let lh_hh = RowSlice {
                    rows: if sv < self.height {
                        &data[sv * self.width..]
                    } else {
                        &[]
                    },
                    stride: self.width,
                };
                (source, hl, lh_hh)
            }
            PrefixBands53::Split { ll1, hl1, lh_hh1 } => (
                RowSlice {
                    rows: ll1,
                    stride: sn,
                },
                RowSlice {
                    rows: hl1,
                    stride: dn,
                },
                RowSlice {
                    rows: lh_hh1,
                    stride: self.width,
                },
            ),
        }
    }

    pub(crate) fn width(&self) -> usize {
        self.width
    }

    pub(crate) fn height(&self) -> usize {
        self.height
    }
}

#[derive(Clone, Copy)]
struct RowSlice<'a> {
    rows: &'a [i32],
    stride: usize,
}

/// Reusable scratch buffers for [`LevelSynth53Borrowed`].
#[derive(Default)]
pub(crate) struct LevelSynth53Scratch {
    lo: Vec<i32>,
    hi: Vec<i32>,
    ring: Vec<i32>,
}

impl LevelSynth53Scratch {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    fn prepare(&mut self, rw: usize) {
        let sn = rw.div_ceil(2);
        let dn = rw - sn;
        if self.lo.len() < sn + 2 {
            self.lo.resize(sn + 2, 0);
        }
        if self.hi.len() < dn + 2 {
            self.hi.resize(dn + 2, 0);
        }
        if self.ring.len() < rw * RING {
            self.ring.resize(rw * RING, 0);
        }
    }
}

/// One synthesis level as a row producer owning its ring scratch.
pub(crate) struct LevelSynth53<'a> {
    ll: RowSlice<'a>,
    hl: RowSlice<'a>,
    lh_hh: RowSlice<'a>,
    rw: usize,
    rh: usize,
    sn: usize,
    kernel: Kernel,
    scratch: LevelSynth53Scratch,
    step: usize,
    next: usize,
    timed: bool,
    horizontal_ns: u64,
}

impl<'a> LevelSynth53<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new_monolithic(
        ll: RowSlice<'a>,
        bands: &'a [i32],
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
        let hl = RowSlice {
            rows: if dn > 0 { &bands[sn..] } else { &[] },
            stride: band_stride,
        };
        let lh_hh = RowSlice {
            rows: if sv < rh {
                &bands[sv * band_stride..]
            } else {
                &[]
            },
            stride: band_stride,
        };
        Self::new_split(ll, hl, lh_hh, rw, rh, kernel, first_row, timed)
    }

    #[allow(clippy::too_many_arguments)]
    fn new_split(
        ll: RowSlice<'a>,
        hl: RowSlice<'a>,
        lh_hh: RowSlice<'a>,
        rw: usize,
        rh: usize,
        kernel: Kernel,
        first_row: usize,
        timed: bool,
    ) -> Self {
        debug_assert!(rw > 0 && rh > 0 && first_row < rh);
        let sn = rw.div_ceil(2);
        let mut scratch = LevelSynth53Scratch::new();
        scratch.prepare(rw);
        Self {
            ll,
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

    pub(crate) fn next_row(&mut self) -> &[i32] {
        step_next_row(
            self.ll,
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
pub(crate) struct LevelSynth53Borrowed<'a> {
    ll: RowSlice<'a>,
    hl: RowSlice<'a>,
    lh_hh: RowSlice<'a>,
    rw: usize,
    rh: usize,
    sn: usize,
    kernel: Kernel,
    scratch: &'a mut LevelSynth53Scratch,
    step: usize,
    next: usize,
    horizontal_ns: u64,
}

impl<'a> LevelSynth53Borrowed<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        ll: RowSlice<'a>,
        hl: RowSlice<'a>,
        lh_hh: RowSlice<'a>,
        rw: usize,
        rh: usize,
        kernel: Kernel,
        first_row: usize,
        scratch: &'a mut LevelSynth53Scratch,
    ) -> Self {
        debug_assert!(rw > 0 && rh > 0 && first_row < rh);
        scratch.prepare(rw);
        Self {
            ll,
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

    pub(crate) fn next_row(&mut self) -> &[i32] {
        step_next_row(
            self.ll,
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
fn step_next_row<'s>(
    ll: RowSlice<'_>,
    hl: RowSlice<'_>,
    lh_hh: RowSlice<'_>,
    rw: usize,
    rh: usize,
    sn: usize,
    kernel: Kernel,
    scratch: &'s mut LevelSynth53Scratch,
    step: &mut usize,
    next: &mut usize,
    timed: bool,
    horizontal_ns: &mut u64,
) -> &'s [i32] {
    debug_assert!(*next < rh);
    let y = *next;
    *next += 1;
    if rh < 2 {
        let (lo_row, hi_row) = input_row_slices(ll, hl, lh_hh, rw, sn, 0);
        let row_start = timed.then(std::time::Instant::now);
        synth_row(
            lo_row,
            hi_row,
            &mut scratch.ring[..rw],
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
    while *step <= y + LAG {
        advance_step(
            ll,
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
fn input_row_slices<'a>(
    ll: RowSlice<'a>,
    hl: RowSlice<'a>,
    lh_hh: RowSlice<'a>,
    rw: usize,
    sn: usize,
    r: usize,
) -> (&'a [i32], &'a [i32]) {
    let dn = rw - sn;
    if r.is_multiple_of(2) {
        let i = r / 2;
        let lo = &ll.rows[i * ll.stride..i * ll.stride + sn];
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
fn advance_step(
    ll: RowSlice<'_>,
    hl: RowSlice<'_>,
    lh_hh: RowSlice<'_>,
    rw: usize,
    rh: usize,
    sn: usize,
    kernel: Kernel,
    scratch: &mut LevelSynth53Scratch,
    step: &mut usize,
    timed: bool,
    horizontal_ns: &mut u64,
) {
    let r = *step;
    *step += 1;
    if r < rh {
        let (lo_row, hi_row) = input_row_slices(ll, hl, lh_hh, rw, sn, r);
        let row_start = timed.then(std::time::Instant::now);
        let slot = (r % RING) * rw;
        synth_row(
            lo_row,
            hi_row,
            &mut scratch.ring[slot..slot + rw],
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
    // Stage 1 (back = 1, even row): undo update `even -= (high_above + high_below + 2) >> 2`.
    let x1 = r - 1;
    if x1 >= 0 && (x1 as usize) < rh && (x1 as usize).is_multiple_of(2) {
        let target = (x1 as usize % RING) * rw;
        let above = (mirror(x1 - 1) % RING) * rw;
        let below = (mirror(x1 + 1) % RING) * rw;
        undo_update_ring_row(&mut scratch.ring, rw, target, above, below, kernel);
    }
    // Stage 2 (back = 2, odd row): undo predict `odd += (even_above + even_below) >> 1`.
    let x2 = r - 2;
    if x2 >= 0 && (x2 as usize) < rh && !(x2 as usize).is_multiple_of(2) {
        let target = (x2 as usize % RING) * rw;
        let above = (mirror(x2 - 1) % RING) * rw;
        let below = (mirror(x2 + 1) % RING) * rw;
        undo_predict_ring_row(&mut scratch.ring, rw, target, above, below, kernel);
    }
}

fn undo_update_ring_row(
    ring: &mut [i32],
    rw: usize,
    target: usize,
    above: usize,
    below: usize,
    kernel: Kernel,
) {
    debug_assert_ne!(target, above);
    debug_assert_ne!(target, below);
    if above == below {
        let [t, n] = ring
            .get_disjoint_mut([target..target + rw, above..above + rw])
            .expect("target and mirrored neighbour rows are disjoint ring slots");
        undo_update_in_place(t, n, n, kernel);
    } else {
        let [t, a, b] = ring
            .get_disjoint_mut([target..target + rw, above..above + rw, below..below + rw])
            .expect("target and neighbour rows are disjoint ring slots");
        undo_update_in_place(t, a, b, kernel);
    }
}

fn undo_predict_ring_row(
    ring: &mut [i32],
    rw: usize,
    target: usize,
    above: usize,
    below: usize,
    kernel: Kernel,
) {
    debug_assert_ne!(target, above);
    debug_assert_ne!(target, below);
    if above == below {
        let [t, n] = ring
            .get_disjoint_mut([target..target + rw, above..above + rw])
            .expect("target and mirrored neighbour rows are disjoint ring slots");
        undo_predict_in_place(t, n, n, kernel);
    } else {
        let [t, a, b] = ring
            .get_disjoint_mut([target..target + rw, above..above + rw, below..below + rw])
            .expect("target and neighbour rows are disjoint ring slots");
        undo_predict_in_place(t, a, b, kernel);
    }
}

/// Horizontal 1-D reversible 5/3 synthesis of one even-origin row from its
/// deinterleaved low-pass (`lo_in`, `ceil(n/2)` samples) and high-pass
/// (`hi_in`, `floor(n/2)` samples) halves into `out`.
fn synth_row(
    lo_in: &[i32],
    hi_in: &[i32],
    out: &mut [i32],
    lo: &mut [i32],
    hi: &mut [i32],
    kernel: Kernel,
) {
    let sn = lo_in.len();
    let dn = hi_in.len();
    let n = sn + dn;
    debug_assert_eq!(out.len(), n);
    if n < 2 {
        if n == 1 {
            out[0] = lo_in[0];
        }
        return;
    }
    debug_assert!(lo.len() >= sn + 2 && hi.len() >= dn + 2);
    let lo = &mut lo[..sn + 2];
    let hi = &mut hi[..dn + 2];
    hi[1..dn + 1].copy_from_slice(hi_in);
    hi[0] = hi[1];
    hi[dn + 1] = hi[dn];

    // Undo update: even[i] = low[i] - ((high_left[i] + high_right[i] + 2) >> 2).
    undo_update_into(
        &mut lo[1..sn + 1],
        lo_in,
        &hi[0..sn],
        &hi[1..sn + 1],
        kernel,
    );

    // Guard for even-length rows: X(n) mirrors to X(n - 2) = even[sn - 1].
    lo[sn + 1] = lo[sn];
    // Undo predict: odd[i] = high[i] + ((even[i] + even[i + 1]) >> 1).
    undo_predict_in_place(&mut hi[1..dn + 1], &lo[1..dn + 1], &lo[2..dn + 2], kernel);

    interleave_i32(&lo[1..sn + 1], &hi[1..dn + 1], out, kernel);
}

#[inline]
fn undo_update_into(out: &mut [i32], lo: &[i32], left: &[i32], right: &[i32], kernel: Kernel) {
    if kernel == Kernel::Avx2 {
        avx2::undo_update_into(out, lo, left, right);
        return;
    }
    let n = out.len().min(lo.len()).min(left.len()).min(right.len());
    let mut i = 0usize;
    #[cfg(feature = "simd")]
    if kernel == Kernel::Wide {
        let two = i32x8::new([2; 8]);
        while i + 8 <= n {
            let l = i32x8::new(left[i..i + 8].try_into().expect("8 lanes"));
            let r = i32x8::new(right[i..i + 8].try_into().expect("8 lanes"));
            let v = i32x8::new(lo[i..i + 8].try_into().expect("8 lanes"));
            let res = v - ((l + r + two) >> 2i32);
            out[i..i + 8].copy_from_slice(&res.to_array());
            i += 8;
        }
    }
    let _ = kernel;
    while i < n {
        out[i] = lo[i] - ((left[i] + right[i] + 2) >> 2);
        i += 1;
    }
}

#[inline]
fn undo_update_in_place(target: &mut [i32], left: &[i32], right: &[i32], kernel: Kernel) {
    if kernel == Kernel::Avx2 {
        avx2::undo_update_in_place(target, left, right);
        return;
    }
    let n = target.len().min(left.len()).min(right.len());
    let mut i = 0usize;
    #[cfg(feature = "simd")]
    if kernel == Kernel::Wide {
        let two = i32x8::new([2; 8]);
        while i + 8 <= n {
            let l = i32x8::new(left[i..i + 8].try_into().expect("8 lanes"));
            let r = i32x8::new(right[i..i + 8].try_into().expect("8 lanes"));
            let t = i32x8::new(target[i..i + 8].try_into().expect("8 lanes"));
            let res = t - ((l + r + two) >> 2i32);
            target[i..i + 8].copy_from_slice(&res.to_array());
            i += 8;
        }
    }
    let _ = kernel;
    while i < n {
        target[i] -= (left[i] + right[i] + 2) >> 2;
        i += 1;
    }
}

#[inline]
fn undo_predict_in_place(target: &mut [i32], left: &[i32], right: &[i32], kernel: Kernel) {
    if kernel == Kernel::Avx2 {
        avx2::undo_predict_in_place(target, left, right);
        return;
    }
    let n = target.len().min(left.len()).min(right.len());
    let mut i = 0usize;
    #[cfg(feature = "simd")]
    if kernel == Kernel::Wide {
        while i + 8 <= n {
            let l = i32x8::new(left[i..i + 8].try_into().expect("8 lanes"));
            let r = i32x8::new(right[i..i + 8].try_into().expect("8 lanes"));
            let t = i32x8::new(target[i..i + 8].try_into().expect("8 lanes"));
            let res = t + ((l + r) >> 1i32);
            target[i..i + 8].copy_from_slice(&res.to_array());
            i += 8;
        }
    }
    let _ = kernel;
    while i < n {
        target[i] += (left[i] + right[i]) >> 1;
        i += 1;
    }
}

#[inline]
fn interleave_i32(lo: &[i32], hi: &[i32], out: &mut [i32], kernel: Kernel) {
    let dn = hi.len();
    debug_assert_eq!(out.len(), lo.len() + dn);
    if kernel == Kernel::Avx2 {
        avx2::interleave_i32(lo, hi, out);
        return;
    }
    for i in 0..dn {
        out[2 * i] = lo[i];
        out[2 * i + 1] = hi[i];
    }
    if lo.len() > dn {
        out[2 * dn] = lo[dn];
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx2 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx2")]
    unsafe fn undo_update_into_impl(out: &mut [i32], lo: &[i32], left: &[i32], right: &[i32]) {
        let n = out.len().min(lo.len()).min(left.len()).min(right.len());
        let two = _mm256_set1_epi32(2);
        let mut x = 0usize;
        while x + 8 <= n {
            // SAFETY: `x + 8 <= n` and every slice is at least `n` elements long.
            unsafe {
                let v = _mm256_loadu_si256(lo.as_ptr().add(x).cast());
                let l = _mm256_loadu_si256(left.as_ptr().add(x).cast());
                let r = _mm256_loadu_si256(right.as_ptr().add(x).cast());
                let d = _mm256_srai_epi32(_mm256_add_epi32(_mm256_add_epi32(l, r), two), 2);
                _mm256_storeu_si256(out.as_mut_ptr().add(x).cast(), _mm256_sub_epi32(v, d));
            }
            x += 8;
        }
        while x < n {
            out[x] = lo[x] - ((left[x] + right[x] + 2) >> 2);
            x += 1;
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn undo_update_in_place_impl(target: &mut [i32], left: &[i32], right: &[i32]) {
        let n = target.len().min(left.len()).min(right.len());
        let two = _mm256_set1_epi32(2);
        let mut x = 0usize;
        while x + 8 <= n {
            // SAFETY: `x + 8 <= n` and every slice is at least `n` elements long.
            unsafe {
                let t = _mm256_loadu_si256(target.as_ptr().add(x).cast());
                let l = _mm256_loadu_si256(left.as_ptr().add(x).cast());
                let r = _mm256_loadu_si256(right.as_ptr().add(x).cast());
                let d = _mm256_srai_epi32(_mm256_add_epi32(_mm256_add_epi32(l, r), two), 2);
                _mm256_storeu_si256(target.as_mut_ptr().add(x).cast(), _mm256_sub_epi32(t, d));
            }
            x += 8;
        }
        while x < n {
            target[x] -= (left[x] + right[x] + 2) >> 2;
            x += 1;
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn undo_predict_in_place_impl(target: &mut [i32], left: &[i32], right: &[i32]) {
        let n = target.len().min(left.len()).min(right.len());
        let mut x = 0usize;
        while x + 8 <= n {
            // SAFETY: `x + 8 <= n` and every slice is at least `n` elements long.
            unsafe {
                let t = _mm256_loadu_si256(target.as_ptr().add(x).cast());
                let l = _mm256_loadu_si256(left.as_ptr().add(x).cast());
                let r = _mm256_loadu_si256(right.as_ptr().add(x).cast());
                let d = _mm256_srai_epi32(_mm256_add_epi32(l, r), 1);
                _mm256_storeu_si256(target.as_mut_ptr().add(x).cast(), _mm256_add_epi32(t, d));
            }
            x += 8;
        }
        while x < n {
            target[x] += (left[x] + right[x]) >> 1;
            x += 1;
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn interleave_i32_impl(lo: &[i32], hi: &[i32], out: &mut [i32]) {
        let dn = hi.len().min(lo.len());
        let mut i = 0usize;
        while i + 8 <= dn && 2 * i + 16 <= out.len() {
            // SAFETY: `i + 8 <= dn` and `2 * i + 16 <= out.len()`.
            unsafe {
                let l = _mm256_loadu_si256(lo.as_ptr().add(i).cast());
                let h = _mm256_loadu_si256(hi.as_ptr().add(i).cast());
                let u0 = _mm256_unpacklo_epi32(l, h);
                let u1 = _mm256_unpackhi_epi32(l, h);
                let out0 = _mm256_permute2x128_si256(u0, u1, 0x20);
                let out1 = _mm256_permute2x128_si256(u0, u1, 0x31);
                let dst = out.as_mut_ptr().add(2 * i).cast::<__m256i>();
                _mm256_storeu_si256(dst, out0);
                _mm256_storeu_si256(dst.add(1), out1);
            }
            i += 8;
        }
        while i < dn {
            out[2 * i] = lo[i];
            out[2 * i + 1] = hi[i];
            i += 1;
        }
        if lo.len() > dn && 2 * dn < out.len() {
            out[2 * dn] = lo[dn];
        }
    }

    // SAFETY (all four): `Kernel::Avx2` is only selected after runtime AVX2 detection.
    #[inline]
    pub(super) fn undo_update_into(out: &mut [i32], lo: &[i32], left: &[i32], right: &[i32]) {
        unsafe { undo_update_into_impl(out, lo, left, right) }
    }

    #[inline]
    pub(super) fn undo_update_in_place(target: &mut [i32], left: &[i32], right: &[i32]) {
        unsafe { undo_update_in_place_impl(target, left, right) }
    }

    #[inline]
    pub(super) fn undo_predict_in_place(target: &mut [i32], left: &[i32], right: &[i32]) {
        unsafe { undo_predict_in_place_impl(target, left, right) }
    }

    #[inline]
    pub(super) fn interleave_i32(lo: &[i32], hi: &[i32], out: &mut [i32]) {
        unsafe { interleave_i32_impl(lo, hi, out) }
    }
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
mod avx2 {
    pub(super) fn undo_update_into(out: &mut [i32], lo: &[i32], left: &[i32], right: &[i32]) {
        super::undo_update_into(out, lo, left, right, crate::simd::Kernel::Scalar);
    }

    pub(super) fn undo_update_in_place(target: &mut [i32], left: &[i32], right: &[i32]) {
        super::undo_update_in_place(target, left, right, crate::simd::Kernel::Scalar);
    }

    pub(super) fn undo_predict_in_place(target: &mut [i32], left: &[i32], right: &[i32]) {
        super::undo_predict_in_place(target, left, right, crate::simd::Kernel::Scalar);
    }

    pub(super) fn interleave_i32(lo: &[i32], hi: &[i32], out: &mut [i32]) {
        super::interleave_i32(lo, hi, out, crate::simd::Kernel::Scalar);
    }
}

#[cfg(test)]
mod tests {
    use super::{
        LevelSynth53Scratch, inverse_53_2d_even, inverse_53_2d_even_but_last,
        inverse_53_2d_even_split,
    };
    use crate::dwt::SplitSubbands;
    use crate::simd::Kernel;

    fn pattern(width: usize, height: usize, seed: u32) -> Vec<i32> {
        let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(54321);
        (0..width * height)
            .map(|i| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                let noise = (state % 1025) as i32 - 512;
                let smooth = ((i % width) as i32 * 3 - (i / width) as i32 * 2) % 257;
                smooth + noise
            })
            .collect()
    }

    fn reference(width: usize, height: usize, levels: u8, seed: u32) -> (Vec<i32>, Vec<i32>) {
        let original = pattern(width, height, seed);
        let mut reference = original.clone();
        super::super::rev53::inverse_53_2d_in_place(&mut reference, width, height, levels)
            .expect("in-place 5/3 reference");
        (original, reference)
    }

    fn split_from_flat(flat: &[i32], width: usize, height: usize) -> SplitSubbands<i32> {
        let sn = width.div_ceil(2);
        let dn = width - sn;
        let sv = height.div_ceil(2);
        let dv = height - sv;
        let mut coarse = vec![0i32; sn * sv];
        let mut hl1 = vec![0i32; dn * sv];
        let mut lh_hh1 = vec![0i32; width * dv];
        for y in 0..sv {
            coarse[y * sn..(y + 1) * sn].copy_from_slice(&flat[y * width..y * width + sn]);
            if dn > 0 {
                hl1[y * dn..(y + 1) * dn].copy_from_slice(&flat[y * width + sn..(y + 1) * width]);
            }
        }
        if dv > 0 {
            lh_hh1.copy_from_slice(&flat[sv * width..]);
        }
        SplitSubbands {
            width,
            height,
            coarse,
            hl1,
            lh_hh1,
        }
    }

    fn check(width: usize, height: usize, levels: u8, kernel: Kernel) {
        let seed = (width * 131 + height * 7) as u32 + levels as u32;
        let (original, reference) = reference(width, height, levels, seed);
        let streamed = inverse_53_2d_even(original.clone(), width, height, levels, kernel, None);
        assert_eq!(
            streamed, reference,
            "monolithic {width}x{height} levels={levels} {kernel:?}"
        );
        if levels >= 1 {
            let split = split_from_flat(&original, width, height);
            let split_out = inverse_53_2d_even_split(split, levels, kernel, None);
            assert_eq!(
                split_out, reference,
                "split {width}x{height} levels={levels} {kernel:?}"
            );
        }
    }

    fn check_banded(width: usize, height: usize, levels: u8, kernel: Kernel, first_row: usize) {
        let seed = (width * 17 + height * 3) as u32 + levels as u32 + first_row as u32;
        let (original, reference) = reference(width, height, levels, seed);
        let prefix = inverse_53_2d_even_but_last(original, width, height, levels, kernel, None);
        let mut scratch = LevelSynth53Scratch::new();
        let mut synth = prefix.final_level_with_scratch(kernel, first_row, &mut scratch);
        for y in first_row..height {
            let row = synth.next_row();
            assert_eq!(
                row,
                &reference[y * width..(y + 1) * width],
                "{width}x{height} levels={levels} {kernel:?} band from {first_row}, row {y}"
            );
        }
    }

    #[test]
    fn streaming_53_matches_in_place_reference_small_sizes() {
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
    fn streaming_53_matches_in_place_reference_larger_sizes() {
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
    fn banded_final_level_53_matches_full_run() {
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
}
