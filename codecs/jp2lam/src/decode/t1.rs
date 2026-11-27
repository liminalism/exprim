//! Tier-1 code-block decoding for the default Part 1 MQ path.
//!
//! This mirrors the encoder pass scan in `encode/backend/native/t1.rs`:
//! top bit-plane cleanup only, then SP -> MR -> cleanup for lower bit-planes.

use crate::error::{Jp2LamError, Result};
use crate::j2k::decode_markers::{CodeBlockStyle, CodestreamHeader, WaveletTransform};
use crate::mq::{MqDecoder, RawDecoder, T1_CTXNO_AGG, T1_CTXNO_MAG, T1_CTXNO_UNI};
use crate::plan::BandOrientation;
use crate::tier1::consts::{
    T1_LUT_SGN_E, T1_LUT_SGN_N, T1_LUT_SGN_S, T1_LUT_SGN_W, T1_SIGMA_E, T1_SIGMA_N, T1_SIGMA_NE,
    T1_SIGMA_NEIGHBOURS, T1_SIGMA_NW, T1_SIGMA_S, T1_SIGMA_W,
};
use crate::tier1::helpers::orientation_index;
use crate::tier1::luts::{LUT_CTXNO_SC, LUT_CTXNO_ZC, LUT_SPB};
use crate::tier1::stripe::{
    StripeFlags, T1_CHI_0_I, T1_CHI_1_I, T1_CHI_2_I, T1_CHI_5_I, T1_MU_0, T1_PI_0, T1_PI_ALL,
    T1_SIGMA_15, T1_SIGMA_16, T1_SIGMA_17, T1_SIGMA_THIS,
};
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};

use super::stats::StatsSink;
use super::t2::{DecodedCodewordSegment, DecodedTilePackets};

#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DecodedCodeBlockCoefficients {
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) coefficients: Vec<i32>,
}

/// A fully decoded tile-component coefficient plane, already in the exact
/// numeric domain the inverse DWT consumes.
///
/// Reversible 5/3 coding carries no quantization, so Tier-1 writes the raw
/// integer magnitudes the inverse 5/3 lifting steps expect. Irreversible 9/7
/// coding is dequantized as each code block is emitted, so Tier-1 writes the
/// final `f32` subband samples directly and no separate full-image `i32`
/// coefficient plane or dequantization sweep is materialized.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum CoefficientPlane {
    Integer(Vec<i32>),
    SplitInteger(crate::dwt::SplitSubbands<i32>),
    Real(Vec<f32>),
    SplitReal(crate::dwt::SplitSubbands<f32>),
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DecodedTileCoefficients {
    pub(crate) component: usize,
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) max_abs: u32,
    pub(crate) plane: CoefficientPlane,
}

fn flatten_split_i32(
    split: crate::dwt::SplitSubbands<i32>,
    pool: &super::Jp2BufferPool,
) -> Vec<i32> {
    let (width, height) = (split.width, split.height);
    let sn = width.div_ceil(2);
    let dn = width - sn;
    let sv = height.div_ceil(2);
    let dv = height - sv;
    let mut out = pool.take_for_overwrite_i32(width * height);
    for y in 0..sv {
        out[y * width..y * width + sn].copy_from_slice(&split.coarse[y * sn..(y + 1) * sn]);
        if dn > 0 {
            out[y * width + sn..(y + 1) * width].copy_from_slice(&split.hl1[y * dn..(y + 1) * dn]);
        }
    }
    if dv > 0 {
        out[sv * width..].copy_from_slice(&split.lh_hh1);
    }
    pool.park_i32(split.coarse);
    pool.park_i32(split.hl1);
    pool.park_i32(split.lh_hh1);
    out
}

fn flatten_split_f32(
    split: crate::dwt::SplitSubbands<f32>,
    pool: &super::Jp2BufferPool,
) -> Vec<f32> {
    let (width, height) = (split.width, split.height);
    let sn = width.div_ceil(2);
    let dn = width - sn;
    let sv = height.div_ceil(2);
    let dv = height - sv;
    let mut out = pool.take_for_overwrite_f32(width * height);
    for y in 0..sv {
        out[y * width..y * width + sn].copy_from_slice(&split.coarse[y * sn..(y + 1) * sn]);
        if dn > 0 {
            out[y * width + sn..(y + 1) * width].copy_from_slice(&split.hl1[y * dn..(y + 1) * dn]);
        }
    }
    if dv > 0 {
        out[sv * width..].copy_from_slice(&split.lh_hh1);
    }
    pool.park_f32(split.coarse);
    pool.park_f32(split.hl1);
    pool.park_f32(split.lh_hh1);
    out
}

impl DecodedTileCoefficients {
    /// Take the reversible integer plane, erroring if this tile was decoded as
    /// an irreversible (dequantized `f32`) plane.
    #[cfg(test)]
    #[allow(dead_code)]
    pub(crate) fn into_integer(self) -> Result<Vec<i32>> {
        self.into_integer_with_pool(&super::Jp2BufferPool::disabled())
    }

    pub(crate) fn into_integer_with_pool(self, pool: &super::Jp2BufferPool) -> Result<Vec<i32>> {
        match self.plane {
            CoefficientPlane::Integer(values) => Ok(values),
            CoefficientPlane::SplitInteger(split) => Ok(flatten_split_i32(split, pool)),
            CoefficientPlane::Real(_) | CoefficientPlane::SplitReal(_) => Err(invalid(
                "reversible reconstruction requested for an irreversible coefficient plane",
            )),
        }
    }

    /// Take the irreversible, already-dequantized `f32` plane, erroring if this
    /// tile was decoded as a reversible integer plane.
    #[cfg(test)]
    #[allow(dead_code)]
    pub(crate) fn into_real(self) -> Result<Vec<f32>> {
        self.into_real_with_pool(&super::Jp2BufferPool::disabled())
    }

    pub(crate) fn into_real_with_pool(self, pool: &super::Jp2BufferPool) -> Result<Vec<f32>> {
        match self.plane {
            CoefficientPlane::Real(values) => Ok(values),
            CoefficientPlane::SplitReal(split) => Ok(flatten_split_f32(split, pool)),
            CoefficientPlane::Integer(_) | CoefficientPlane::SplitInteger(_) => Err(invalid(
                "irreversible reconstruction requested for a reversible coefficient plane",
            )),
        }
    }
}

pub(crate) struct Tier1Scratch {
    width: usize,
    height: usize,
    /// Decoded (signed) coefficients in coding order: the four samples of a
    /// stripe column are adjacent, sample `(x, y)` at
    /// `((y / 4) * width + x) * 4 + y % 4`. The passes walk columns, so each
    /// column's samples share a cache line and need no row arithmetic.
    coefficients: Vec<i32>,
    flags: StripeFlags,
    /// [`Self::coefficients`] output, filled on request.
    row_major: Vec<i32>,
}

impl Tier1Scratch {
    pub(crate) fn new() -> Self {
        Self {
            width: 0,
            height: 0,
            coefficients: Vec::new(),
            flags: StripeFlags::new(),
            row_major: Vec::new(),
        }
    }

    /// The decoded block, row-major `width × height`.
    pub(crate) fn coefficients(&mut self) -> &[i32] {
        let mut row_major = std::mem::take(&mut self.row_major);
        row_major.clear();
        for y in 0..self.height {
            let (columns, ci) = self.row(y);
            row_major.extend(columns.iter().map(|column| column[ci]));
        }
        self.row_major = row_major;
        &self.row_major
    }

    /// Row `y` of the decoded block: the columns of its stripe and the
    /// position of the row within each.
    #[inline(always)]
    fn row(&self, y: usize) -> (&[[i32; 4]], usize) {
        let stripe = 4 * self.width * (y / 4);
        let (columns, _) = self.coefficients[stripe..stripe + 4 * self.width].as_chunks::<4>();
        (columns, y % 4)
    }

    fn reset(&mut self, width: usize, height: usize) -> Result<()> {
        let coefficient_count = width
            .checked_mul(height.div_ceil(4))
            .and_then(|columns| columns.checked_mul(4))
            .ok_or_else(|| invalid("code-block coefficient count overflow"))?;
        self.width = width;
        self.height = height;
        self.coefficients.clear();
        self.coefficients.resize(coefficient_count, 0);
        self.flags.reset(width, height);
        Ok(())
    }
}

#[cfg(test)]
pub(crate) fn decode_tile_components(
    header: &CodestreamHeader,
    packets: &DecodedTilePackets<'_>,
) -> Result<Vec<DecodedTileCoefficients>> {
    let mut stats = StatsSink::disabled();
    decode_tile_components_profiled(header, packets, &mut stats)
}

#[cfg(test)]
pub(crate) fn decode_tile_components_profiled(
    header: &CodestreamHeader,
    packets: &DecodedTilePackets<'_>,
    stats: &mut StatsSink<'_>,
) -> Result<Vec<DecodedTileCoefficients>> {
    let mut scratch = Tier1Scratch::new();
    decode_tile_components_with_scratch(header, packets, stats, &mut scratch, 1)
}

#[cfg(test)]
pub(crate) fn decode_tile_components_with_scratch(
    header: &CodestreamHeader,
    packets: &DecodedTilePackets<'_>,
    stats: &mut StatsSink<'_>,
    scratch: &mut Tier1Scratch,
    worker_count: usize,
) -> Result<Vec<DecodedTileCoefficients>> {
    let mut worker_scratches = Vec::new();
    let disabled = super::Jp2BufferPool::disabled();
    decode_tile_component_range_with_scratch(
        header,
        packets,
        0..header.siz.components.len(),
        false,
        stats,
        scratch,
        &mut worker_scratches,
        worker_count,
        &disabled,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_tile_component_range_with_scratch(
    header: &CodestreamHeader,
    packets: &DecodedTilePackets<'_>,
    component_range: std::ops::Range<usize>,
    allow_split: bool,
    stats: &mut StatsSink<'_>,
    scratch: &mut Tier1Scratch,
    worker_scratches: &mut Vec<Tier1Scratch>,
    worker_count: usize,
    pool: &super::Jp2BufferPool,
    cancel: Option<&(dyn Fn() -> bool + Sync)>,
) -> Result<Vec<DecodedTileCoefficients>> {
    let component_count = header.siz.components.len();
    let end_comp = component_range.end.min(component_count);
    let start_comp = component_range.start.min(end_comp);
    let active_range = start_comp..end_comp;
    if cancel.is_some_and(|f| f()) {
        return Err(Jp2LamError::Cancelled);
    }
    let levels = header.cod.decomposition_levels;
    let split_ok = allow_split && levels >= 1;
    let mut components = Vec::with_capacity(active_range.len());
    for component in active_range.clone() {
        let (cw, ch) = header.tile_component_dims(component)?;
        let use_split = split_ok && cw > 0 && ch > 0;
        let plane = match (header.transform_for(component), use_split) {
            (WaveletTransform::Irreversible97, true) => CoefficientPlane::SplitReal(
                crate::dwt::SplitSubbands::<f32>::new_zeroed_with_pool(cw, ch, pool),
            ),
            (WaveletTransform::Irreversible97, false) => {
                CoefficientPlane::Real(pool.take_zeroed_f32(cw * ch))
            }
            (WaveletTransform::Reversible53, true) => CoefficientPlane::SplitInteger(
                crate::dwt::SplitSubbands::<i32>::new_zeroed_with_pool(cw, ch, pool),
            ),
            (WaveletTransform::Reversible53, false) => {
                CoefficientPlane::Integer(pool.take_zeroed_i32(cw * ch))
            }
        };
        components.push(DecodedTileCoefficients {
            component,
            width: cw,
            height: ch,
            max_abs: 0,
            plane,
        });
    }
    if components.is_empty() {
        return Ok(components);
    }

    if stats.is_enabled() || worker_count <= 1 {
        let planes = BlockPlanes::new(&mut components, start_comp);
        let mut decoded_idx = 0usize;
        for block in &packets.codeblocks {
            if !active_range.contains(&block.component) {
                continue;
            }
            if (decoded_idx & 15) == 0 && cancel.is_some_and(|f| f()) {
                return Err(Jp2LamError::Cancelled);
            }
            decoded_idx += 1;
            let params = block_params(header, block)?;
            decode_codeblock_segments_into(
                params.block_width,
                params.block_height,
                block.band,
                params.max_bitplanes,
                block.zero_bitplanes,
                block.passes,
                header.cod.code_block_style,
                &block.segments,
                scratch,
                stats,
            )?;
            let output_start = stats.start();
            // SAFETY: nothing else writes the planes while this loop runs.
            unsafe {
                planes.write_block(
                    block.component,
                    block.x0 as usize,
                    block.y0 as usize,
                    scratch,
                    params.step,
                )?;
            }
            stats.finish(output_start, |stats, elapsed| {
                stats.tier1_block_output_ns = stats.tier1_block_output_ns.saturating_add(elapsed);
            });
        }
        planes.store_max_abs(&mut components);
    } else {
        decode_codeblocks_parallel(
            header,
            packets,
            active_range,
            &mut components,
            worker_scratches,
            worker_count,
            cancel,
        )?;
    }

    Ok(components)
}

struct BlockParams {
    block_width: usize,
    block_height: usize,
    max_bitplanes: u8,
    /// Scalar dequantization step for the irreversible 9/7 path; `None` for the
    /// reversible 5/3 path, which carries no quantization.
    step: Option<f32>,
}

/// Resolve a code block's decode geometry, magnitude bit-plane count, and (for
/// the irreversible path) its fused dequantization step from the codestream
/// headers. Shared by the serial and parallel Tier-1 loops.
fn block_params(
    header: &CodestreamHeader,
    block: &super::t2::DecodedCodeBlock<'_>,
) -> Result<BlockParams> {
    // Per-component quantization: a QCC override for this component, else the
    // default QCD (ISO/IEC 15444-1 Annex A.6.5).
    let comp_quant = header.quant_for(block.component);
    let quant = *comp_quant
        .steps
        .get(block.band_index % comp_quant.steps.len())
        .ok_or_else(|| invalid("missing quantization step for decoded subband"))?;
    let precision = header
        .siz
        .components
        .get(block.component)
        .ok_or_else(|| invalid("decoded code-block references invalid component"))?
        .precision;
    if matches!(
        header.transform_for(block.component),
        WaveletTransform::Reversible53
    ) {
        let maximum_exponent =
            crate::dwt::norms::reversible_exponent(u32::from(precision), block.band);
        if quant.exponent > maximum_exponent {
            return Err(invalid(format!(
                "reversible subband exponent {} exceeds the {}-bit component bound {maximum_exponent}",
                quant.exponent, precision
            )));
        }
    }
    let max_bitplanes = comp_quant.guard_bits.saturating_sub(1) + quant.exponent;
    let step = if matches!(
        header.transform_for(block.component),
        WaveletTransform::Irreversible97
    ) {
        Some(super::reconstruct::quant_step(
            u32::from(precision),
            block.band,
            quant,
        ))
    } else {
        None
    };
    Ok(BlockParams {
        block_width: (block.x1 - block.x0) as usize,
        block_height: (block.y1 - block.y0) as usize,
        max_bitplanes,
        step,
    })
}

/// A `Send`/`Sync` view of every tile-component plane as raw base pointers, so
/// the parallel Tier-1 loop can write each code block into its own disjoint
/// rectangle without a per-block result allocation.
enum PlanePtr {
    Integer {
        ptr: *mut i32,
        stride: usize,
        len: usize,
    },
    SplitInteger {
        coarse: *mut i32,
        hl1: *mut i32,
        lh_hh1: *mut i32,
        width: usize,
        sn: usize,
        dn: usize,
        sv: usize,
        dv: usize,
    },
    Real {
        ptr: *mut f32,
        stride: usize,
        len: usize,
    },
    SplitReal {
        coarse: *mut f32,
        hl1: *mut f32,
        lh_hh1: *mut f32,
        width: usize,
        sn: usize,
        dn: usize,
        sv: usize,
        dv: usize,
    },
}

enum ResolvedSubbandPtr {
    Integer {
        ptr: *mut i32,
        stride: usize,
        len: usize,
        x0: usize,
        y0: usize,
    },
    Real {
        ptr: *mut f32,
        stride: usize,
        len: usize,
        x0: usize,
        y0: usize,
    },
}

struct BlockPlanes {
    start_component: usize,
    planes: Vec<PlanePtr>,
    max_abs: Vec<AtomicU32>,
}

// SAFETY: `BlockPlanes` only ever hands out `&mut` to pairwise-disjoint plane
// rectangles (one code block each). The raw pointers alias the `components`
// planes, which outlive every parallel task, and no task reads another task's
// region, so sharing the base pointers across threads is sound.
unsafe impl Send for BlockPlanes {}
unsafe impl Sync for BlockPlanes {}

impl BlockPlanes {
    fn new(components: &mut [DecodedTileCoefficients], start_component: usize) -> Self {
        let mut max_abs = Vec::with_capacity(components.len());
        let planes = components
            .iter_mut()
            .map(|component| {
                max_abs.push(AtomicU32::new(0));
                let w = component.width;
                let h = component.height;
                match &mut component.plane {
                    CoefficientPlane::Integer(values) => PlanePtr::Integer {
                        ptr: values.as_mut_ptr(),
                        stride: w,
                        len: values.len(),
                    },
                    CoefficientPlane::SplitInteger(split) => {
                        let sn = w.div_ceil(2);
                        let sv = h.div_ceil(2);
                        PlanePtr::SplitInteger {
                            coarse: split.coarse.as_mut_ptr(),
                            hl1: split.hl1.as_mut_ptr(),
                            lh_hh1: split.lh_hh1.as_mut_ptr(),
                            width: w,
                            sn,
                            dn: w - sn,
                            sv,
                            dv: h - sv,
                        }
                    }
                    CoefficientPlane::Real(values) => PlanePtr::Real {
                        ptr: values.as_mut_ptr(),
                        stride: w,
                        len: values.len(),
                    },
                    CoefficientPlane::SplitReal(split) => {
                        let sn = w.div_ceil(2);
                        let sv = h.div_ceil(2);
                        PlanePtr::SplitReal {
                            coarse: split.coarse.as_mut_ptr(),
                            hl1: split.hl1.as_mut_ptr(),
                            lh_hh1: split.lh_hh1.as_mut_ptr(),
                            width: w,
                            sn,
                            dn: w - sn,
                            sv,
                            dv: h - sv,
                        }
                    }
                }
            })
            .collect();
        Self {
            start_component,
            planes,
            max_abs,
        }
    }

    fn store_max_abs(&self, components: &mut [DecodedTileCoefficients]) {
        for (comp, max_slot) in components.iter_mut().zip(&self.max_abs) {
            comp.max_abs = max_slot.load(Ordering::Relaxed);
        }
    }

    fn resolve(&self, slot: usize, x0: usize, y0: usize) -> Result<ResolvedSubbandPtr> {
        let plane = self
            .planes
            .get(slot)
            .ok_or_else(|| invalid("decoded code-block references invalid component"))?;
        Ok(match *plane {
            PlanePtr::Integer { ptr, stride, len } => ResolvedSubbandPtr::Integer {
                ptr,
                stride,
                len,
                x0,
                y0,
            },
            PlanePtr::SplitInteger {
                coarse,
                hl1,
                lh_hh1,
                width,
                sn,
                dn,
                sv,
                dv,
            } => {
                if y0 < sv {
                    if x0 < sn {
                        ResolvedSubbandPtr::Integer {
                            ptr: coarse,
                            stride: sn,
                            len: sn * sv,
                            x0,
                            y0,
                        }
                    } else {
                        ResolvedSubbandPtr::Integer {
                            ptr: hl1,
                            stride: dn,
                            len: dn * sv,
                            x0: x0 - sn,
                            y0,
                        }
                    }
                } else {
                    ResolvedSubbandPtr::Integer {
                        ptr: lh_hh1,
                        stride: width,
                        len: width * dv,
                        x0,
                        y0: y0 - sv,
                    }
                }
            }
            PlanePtr::Real { ptr, stride, len } => ResolvedSubbandPtr::Real {
                ptr,
                stride,
                len,
                x0,
                y0,
            },
            PlanePtr::SplitReal {
                coarse,
                hl1,
                lh_hh1,
                width,
                sn,
                dn,
                sv,
                dv,
            } => {
                if y0 < sv {
                    if x0 < sn {
                        ResolvedSubbandPtr::Real {
                            ptr: coarse,
                            stride: sn,
                            len: sn * sv,
                            x0,
                            y0,
                        }
                    } else {
                        ResolvedSubbandPtr::Real {
                            ptr: hl1,
                            stride: dn,
                            len: dn * sv,
                            x0: x0 - sn,
                            y0,
                        }
                    }
                } else {
                    ResolvedSubbandPtr::Real {
                        ptr: lh_hh1,
                        stride: width,
                        len: width * dv,
                        x0,
                        y0: y0 - sv,
                    }
                }
            }
        })
    }

    /// # Safety
    /// The caller must guarantee that the `(component, rectangle)` written here
    /// does not overlap any other rectangle written concurrently.
    unsafe fn write_block(
        &self,
        component: usize,
        x0: usize,
        y0: usize,
        block: &Tier1Scratch,
        step: Option<f32>,
    ) -> Result<()> {
        if block.width == 0 || block.height == 0 {
            return Ok(());
        }
        let slot = component
            .checked_sub(self.start_component)
            .ok_or_else(|| invalid("decoded code-block references inactive component"))?;
        match self.resolve(slot, x0, y0)? {
            ResolvedSubbandPtr::Integer {
                ptr,
                stride,
                len,
                x0,
                y0,
            } => {
                if x0
                    .checked_add(block.width)
                    .is_none_or(|end_x| end_x > stride)
                {
                    return Err(invalid("decoded block extends past subband width"));
                }
                let last_row_end = (y0 + block.height - 1)
                    .checked_mul(stride)
                    .and_then(|row| row.checked_add(x0))
                    .and_then(|start| start.checked_add(block.width))
                    .ok_or_else(|| invalid("block copy offset overflow"))?;
                if last_row_end > len {
                    return Err(invalid("decoded block extends past subband"));
                }
                // SAFETY: bounds checked above and code blocks are pairwise disjoint.
                let block_max = unsafe {
                    write_integer_block(
                        ptr,
                        stride,
                        x0,
                        y0,
                        block,
                        crate::simd::PRIMITIVES.kernel == crate::simd::Kernel::Avx2,
                    )
                };
                if block_max > 0 {
                    self.max_abs[slot].fetch_max(block_max, Ordering::Relaxed);
                }
            }
            ResolvedSubbandPtr::Real {
                ptr,
                stride,
                len,
                x0,
                y0,
            } => {
                let step = step
                    .ok_or_else(|| invalid("irreversible plane write missing quantization step"))?;
                if x0
                    .checked_add(block.width)
                    .is_none_or(|end_x| end_x > stride)
                {
                    return Err(invalid("decoded block extends past subband width"));
                }
                let last_row_end = (y0 + block.height - 1)
                    .checked_mul(stride)
                    .and_then(|row| row.checked_add(x0))
                    .and_then(|start| start.checked_add(block.width))
                    .ok_or_else(|| invalid("block copy offset overflow"))?;
                if last_row_end > len {
                    return Err(invalid("decoded block extends past subband"));
                }
                // SAFETY: bounds checked above and code blocks are pairwise disjoint.
                unsafe {
                    write_real_block(
                        ptr,
                        stride,
                        x0,
                        y0,
                        block,
                        step,
                        crate::simd::PRIMITIVES.kernel == crate::simd::Kernel::Avx2,
                    );
                }
            }
        }
        Ok(())
    }
}

/// Write a decoded reversible code-block from stripe-column scratch to row-major
/// destination and return `max(|v|)` across all samples written.
///
/// # Safety
/// `ptr` must be valid for writes over `(y0..y0 + block.height)` at row `stride`
/// starting at `x0` for `block.width` elements, disjoint from concurrent writers.
unsafe fn write_integer_block(
    ptr: *mut i32,
    stride: usize,
    x0: usize,
    y0: usize,
    block: &Tier1Scratch,
    use_avx2: bool,
) -> u32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if use_avx2 {
        // SAFETY: `use_avx2` is true only when `PRIMITIVES.kernel == Kernel::Avx2`.
        return unsafe { block_out_avx2::write_integer_block(ptr, stride, x0, y0, block) };
    }
    let _ = use_avx2;
    let mut max_abs = 0u32;
    for y in 0..block.height {
        let dst = (y0 + y) * stride + x0;
        let (columns, ci) = block.row(y);
        // SAFETY: caller verified bounds and disjointness.
        let dst_slice = unsafe { std::slice::from_raw_parts_mut(ptr.add(dst), block.width) };
        for (out, column) in dst_slice.iter_mut().zip(columns) {
            let v = column[ci];
            *out = v;
            let u = v.unsigned_abs();
            if u > max_abs {
                max_abs = u;
            }
        }
    }
    max_abs
}

/// Dequantize and write an irreversible code-block from stripe-column scratch.
///
/// # Safety
/// Same bounds and disjointness requirements as [`write_integer_block`].
unsafe fn write_real_block(
    ptr: *mut f32,
    stride: usize,
    x0: usize,
    y0: usize,
    block: &Tier1Scratch,
    step: f32,
    use_avx2: bool,
) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if use_avx2 {
        // SAFETY: `use_avx2` is true only when `PRIMITIVES.kernel == Kernel::Avx2`.
        unsafe { block_out_avx2::write_real_block(ptr, stride, x0, y0, block, step) };
        return;
    }
    let _ = use_avx2;
    for y in 0..block.height {
        let dst = (y0 + y) * stride + x0;
        let (columns, ci) = block.row(y);
        // SAFETY: caller verified bounds and disjointness.
        let dst_slice = unsafe { std::slice::from_raw_parts_mut(ptr.add(dst), block.width) };
        for (out, column) in dst_slice.iter_mut().zip(columns) {
            *out = crate::simd::scalar::dequantize_i32_to_f32(column[ci], step);
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod block_out_avx2 {
    use super::Tier1Scratch;
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    /// Load 8 `[i32; 4]` stripe columns starting at `cols_ptr` and transpose
    /// them into four 8-wide row vectors `(r0, r1, r2, r3)`.
    ///
    /// # Safety
    /// `cols_ptr` must point to at least 8 `[i32; 4]` columns (32 `i32`s), and
    /// AVX2 must be enabled.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn transpose_8x4(cols_ptr: *const [i32; 4]) -> (__m256i, __m256i, __m256i, __m256i) {
        // SAFETY: caller guarantees 8 `[i32; 4]` columns are readable.
        unsafe {
            let load_pair = |k: usize| {
                let lo = _mm_loadu_si128(cols_ptr.add(k).cast());
                let hi = _mm_loadu_si128(cols_ptr.add(k + 4).cast());
                _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1)
            };
            let a0 = load_pair(0);
            let a1 = load_pair(1);
            let a2 = load_pair(2);
            let a3 = load_pair(3);
            let t0 = _mm256_unpacklo_epi32(a0, a1);
            let t1 = _mm256_unpacklo_epi32(a2, a3);
            let t2 = _mm256_unpackhi_epi32(a0, a1);
            let t3 = _mm256_unpackhi_epi32(a2, a3);
            (
                _mm256_unpacklo_epi64(t0, t1),
                _mm256_unpackhi_epi64(t0, t1),
                _mm256_unpacklo_epi64(t2, t3),
                _mm256_unpackhi_epi64(t2, t3),
            )
        }
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn write_integer_block(
        ptr: *mut i32,
        stride: usize,
        x0: usize,
        y0: usize,
        block: &Tier1Scratch,
    ) -> u32 {
        let bw = block.width;
        let bh = block.height;
        let full_stripes = bh / 4;
        let mut max_vec = _mm256_setzero_si256();
        let mut scalar_max = 0u32;

        for s in 0..full_stripes {
            let (columns, _) = block.row(4 * s);
            let row0 = (y0 + 4 * s) * stride + x0;
            let mut x = 0usize;
            while x + 8 <= bw {
                // SAFETY: `x + 8 <= bw` and all 4 rows in `4*s .. 4*s + 4` are in bounds.
                unsafe {
                    let (r0, r1, r2, r3) = transpose_8x4(columns.as_ptr().add(x));
                    max_vec = _mm256_max_epu32(max_vec, _mm256_abs_epi32(r0));
                    max_vec = _mm256_max_epu32(max_vec, _mm256_abs_epi32(r1));
                    max_vec = _mm256_max_epu32(max_vec, _mm256_abs_epi32(r2));
                    max_vec = _mm256_max_epu32(max_vec, _mm256_abs_epi32(r3));
                    _mm256_storeu_si256(ptr.add(row0 + x).cast(), r0);
                    _mm256_storeu_si256(ptr.add(row0 + stride + x).cast(), r1);
                    _mm256_storeu_si256(ptr.add(row0 + 2 * stride + x).cast(), r2);
                    _mm256_storeu_si256(ptr.add(row0 + 3 * stride + x).cast(), r3);
                }
                x += 8;
            }
            while x < bw {
                let col = columns[x];
                for (ci, &v) in col.iter().enumerate() {
                    // SAFETY: in bounds.
                    unsafe { *ptr.add(row0 + ci * stride + x) = v };
                    scalar_max = scalar_max.max(v.unsigned_abs());
                }
                x += 1;
            }
        }
        for y in (full_stripes * 4)..bh {
            let dst = (y0 + y) * stride + x0;
            let (columns, ci) = block.row(y);
            for (x, column) in columns.iter().enumerate() {
                let v = column[ci];
                // SAFETY: in bounds.
                unsafe { *ptr.add(dst + x) = v };
                scalar_max = scalar_max.max(v.unsigned_abs());
            }
        }
        let mut lanes = [0u32; 8];
        // SAFETY: `lanes` is 32 bytes.
        unsafe { _mm256_storeu_si256(lanes.as_mut_ptr().cast(), max_vec) };
        lanes.into_iter().fold(scalar_max, u32::max)
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn write_real_block(
        ptr: *mut f32,
        stride: usize,
        x0: usize,
        y0: usize,
        block: &Tier1Scratch,
        step: f32,
    ) {
        let bw = block.width;
        let bh = block.height;
        let full_stripes = bh / 4;
        let zero_i32 = _mm256_setzero_si256();
        let half = _mm256_set1_ps(0.5);
        let step_v = _mm256_set1_ps(step);
        let sign_bit = _mm256_castsi256_ps(_mm256_set1_epi32(i32::MIN));

        // Exact IEEE-754 match for `scalar::dequantize_i32_to_f32`:
        // for q == 0 -> +0.0f32; for 0 < |q| <= i32::MAX -> sign * (|q| as f32 + 0.5) * step.
        // Tier-1 guarantees `magnitude_bitplanes <= 31`, so `q != i32::MIN`.
        let dequant = |qv: __m256i| -> __m256 {
            let abs_q = _mm256_abs_epi32(qv);
            let mag = _mm256_mul_ps(_mm256_add_ps(_mm256_cvtepi32_ps(abs_q), half), step_v);
            let signed = _mm256_xor_ps(mag, _mm256_and_ps(_mm256_castsi256_ps(qv), sign_bit));
            let is_zero = _mm256_castsi256_ps(_mm256_cmpeq_epi32(qv, zero_i32));
            _mm256_andnot_ps(is_zero, signed)
        };

        for s in 0..full_stripes {
            let (columns, _) = block.row(4 * s);
            let row0 = (y0 + 4 * s) * stride + x0;
            let mut x = 0usize;
            while x + 8 <= bw {
                // SAFETY: `x + 8 <= bw` and all 4 rows in `4*s .. 4*s + 4` are in bounds.
                unsafe {
                    let (r0, r1, r2, r3) = transpose_8x4(columns.as_ptr().add(x));
                    _mm256_storeu_ps(ptr.add(row0 + x), dequant(r0));
                    _mm256_storeu_ps(ptr.add(row0 + stride + x), dequant(r1));
                    _mm256_storeu_ps(ptr.add(row0 + 2 * stride + x), dequant(r2));
                    _mm256_storeu_ps(ptr.add(row0 + 3 * stride + x), dequant(r3));
                }
                x += 8;
            }
            while x < bw {
                let col = columns[x];
                for (ci, &v) in col.iter().enumerate() {
                    // SAFETY: in bounds.
                    unsafe {
                        *ptr.add(row0 + ci * stride + x) =
                            crate::simd::scalar::dequantize_i32_to_f32(v, step);
                    }
                }
                x += 1;
            }
        }
        for y in (full_stripes * 4)..bh {
            let dst = (y0 + y) * stride + x0;
            let (columns, ci) = block.row(y);
            for (x, column) in columns.iter().enumerate() {
                // SAFETY: in bounds.
                unsafe {
                    *ptr.add(dst + x) =
                        crate::simd::scalar::dequantize_i32_to_f32(column[ci], step);
                }
            }
        }
    }
}

/// Decode every active code block across the active worker budget, reusing
/// caller-owned [`Tier1Scratch`] instances across tiles and decodes.
fn decode_codeblocks_parallel(
    header: &CodestreamHeader,
    packets: &DecodedTilePackets<'_>,
    active_range: std::ops::Range<usize>,
    components: &mut [DecodedTileCoefficients],
    worker_scratches: &mut Vec<Tier1Scratch>,
    worker_count: usize,
    cancel: Option<&(dyn Fn() -> bool + Sync)>,
) -> Result<()> {
    let start_comp = active_range.start;
    let planes = BlockPlanes::new(components, start_comp);
    let style = header.cod.code_block_style;
    let blocks = &packets.codeblocks;

    let mut order: Vec<usize> = (0..blocks.len())
        .filter(|&i| active_range.contains(&blocks[i].component))
        .collect();
    if order.is_empty() {
        return Ok(());
    }
    order.sort_unstable_by_key(|&i| {
        let block = &blocks[i];
        let area = u64::from(block.x1.saturating_sub(block.x0))
            * u64::from(block.y1.saturating_sub(block.y0));
        core::cmp::Reverse(area.saturating_mul(u64::from(block.passes)))
    });

    let cursor = AtomicUsize::new(0);
    let failed = AtomicBool::new(false);
    let cancelled = AtomicBool::new(false);

    let task_count = worker_count.min(order.len()).max(1);
    if worker_scratches.len() < task_count {
        worker_scratches.resize_with(task_count, Tier1Scratch::new);
    }
    let results = worker_scratches[..task_count]
        .par_iter_mut()
        .map(|scratch| -> Result<()> {
            let mut sink = StatsSink::disabled();
            loop {
                if failed.load(Ordering::Relaxed) {
                    return Ok(());
                }
                let next = cursor.fetch_add(1, Ordering::Relaxed);
                if (next & 15) == 0 && cancel.is_some_and(|f| f()) {
                    cancelled.store(true, Ordering::Relaxed);
                    failed.store(true, Ordering::Relaxed);
                    return Err(Jp2LamError::Cancelled);
                }
                let Some(&index) = order.get(next) else {
                    return Ok(());
                };
                let block = &blocks[index];
                let params = match block_params(header, block) {
                    Ok(params) => params,
                    Err(error) => {
                        failed.store(true, Ordering::Relaxed);
                        return Err(error);
                    }
                };
                if let Err(error) = decode_codeblock_segments_into(
                    params.block_width,
                    params.block_height,
                    block.band,
                    params.max_bitplanes,
                    block.zero_bitplanes,
                    block.passes,
                    style,
                    &block.segments,
                    scratch,
                    &mut sink,
                ) {
                    failed.store(true, Ordering::Relaxed);
                    return Err(error);
                }
                // SAFETY: code blocks tile pairwise-disjoint rectangles of each
                // component plane, so no two parallel writes overlap.
                let write = unsafe {
                    planes.write_block(
                        block.component,
                        block.x0 as usize,
                        block.y0 as usize,
                        scratch,
                        params.step,
                    )
                };
                if let Err(error) = write {
                    failed.store(true, Ordering::Relaxed);
                    return Err(error);
                }
            }
        })
        .collect::<Vec<_>>();

    if cancelled.load(Ordering::Relaxed) {
        return Err(Jp2LamError::Cancelled);
    }
    for result in results {
        result?;
    }
    planes.store_max_abs(components);
    Ok(())
}

#[cfg(test)]
pub(crate) fn decode_tile_coefficients(
    header: &CodestreamHeader,
    packets: &DecodedTilePackets<'_>,
) -> Result<DecodedTileCoefficients> {
    let mut components = decode_tile_components(header, packets)?;
    if components.len() != 1 {
        return Err(invalid(
            "single-component decode requested for multi-component tile",
        ));
    }
    Ok(components.remove(0))
}

/// Consume the segmentation symbol that closes a cleanup pass: four symbols
/// coded with the UNIFORM context spelling `0b1010` (ISO/IEC 15444-1 D.5).
/// A different value means the codeword has desynchronized - unless this is
/// the last pass the stream carries for the block: a rate-truncated codeword
/// ends mid-pass, its final symbols are the decoder's 0xff fill, and the
/// marker after them is the first thing to go. That tail is the truncation
/// the encoder chose, not an error, so only an earlier mismatch is one.
pub(super) fn read_segmentation_symbol(decoder: &mut MqDecoder<'_>, last_pass: bool) -> Result<()> {
    let mut value = 0u8;
    for _ in 0..4 {
        value = (value << 1) | decoder.decode_with_ctx(T1_CTXNO_UNI);
    }
    if value != 0xA && !last_pass {
        return Err(invalid(format!(
            "segmentation symbol mismatch: expected 0xA, found {value:#x}"
        )));
    }
    Ok(())
}

/// Decode a payload prefix of `pass_count` coding passes into `scratch`.
///
/// Used by the perceptual candidate reconstructor so a stored Tier-1
/// truncation can be inverted without serializing a codestream.
pub(crate) fn decode_pass_prefix(
    width: usize,
    height: usize,
    band: BandOrientation,
    max_bitplanes: u8,
    zero_bitplanes: u32,
    pass_count: u32,
    data: &[u8],
    scratch: &mut Tier1Scratch,
) -> Result<()> {
    let segment = DecodedCodewordSegment {
        passes: pass_count,
        data: std::borrow::Cow::Borrowed(data),
    };
    let mut stats = StatsSink::disabled();
    decode_codeblock_segments_into(
        width,
        height,
        band,
        max_bitplanes,
        zero_bitplanes,
        pass_count,
        CodeBlockStyle::default(),
        std::slice::from_ref(&segment),
        scratch,
        &mut stats,
    )
}

/// Decode one code-block's coding passes.
///
/// `segmentation_symbols` mirrors the COD code-block style bit (ISO/IEC
/// 15444-1 D.5): when set, each cleanup pass is followed by the four-symbol
/// `0xA` marker coded with the UNIFORM context, which must be consumed to
/// keep the MQ decoder aligned.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_codeblock(
    width: usize,
    height: usize,
    band: BandOrientation,
    max_bitplanes: u8,
    zero_bitplanes: u32,
    pass_count: u32,
    segmentation_symbols: bool,
    data: &[u8],
) -> Result<DecodedCodeBlockCoefficients> {
    let segment = DecodedCodewordSegment {
        passes: pass_count,
        data: std::borrow::Cow::Borrowed(data),
    };
    decode_codeblock_segments(
        width,
        height,
        band,
        max_bitplanes,
        zero_bitplanes,
        pass_count,
        CodeBlockStyle {
            segmentation_symbols,
            ..CodeBlockStyle::default()
        },
        std::slice::from_ref(&segment),
    )
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn decode_codeblock_segments(
    width: usize,
    height: usize,
    band: BandOrientation,
    max_bitplanes: u8,
    zero_bitplanes: u32,
    pass_count: u32,
    style: CodeBlockStyle,
    segments: &[DecodedCodewordSegment<'_>],
) -> Result<DecodedCodeBlockCoefficients> {
    let mut scratch = Tier1Scratch::new();
    let mut stats = StatsSink::disabled();
    decode_codeblock_segments_into(
        width,
        height,
        band,
        max_bitplanes,
        zero_bitplanes,
        pass_count,
        style,
        segments,
        &mut scratch,
        &mut stats,
    )?;
    Ok(DecodedCodeBlockCoefficients {
        width,
        height,
        coefficients: scratch.coefficients().to_vec(),
    })
}

#[allow(clippy::too_many_arguments)]
fn decode_codeblock_segments_into(
    width: usize,
    height: usize,
    band: BandOrientation,
    max_bitplanes: u8,
    zero_bitplanes: u32,
    pass_count: u32,
    style: CodeBlockStyle,
    segments: &[DecodedCodewordSegment<'_>],
    scratch: &mut Tier1Scratch,
    stats: &mut StatsSink<'_>,
) -> Result<()> {
    scratch.reset(width, height)?;
    if pass_count == 0 {
        return Ok(());
    }
    let segment_passes = segments.iter().try_fold(0u32, |total, segment| {
        total
            .checked_add(segment.passes)
            .ok_or_else(|| invalid("codeword-segment pass count overflow"))
    })?;
    if segment_passes != pass_count {
        return Err(invalid(format!(
            "codeword segments carry {segment_passes} passes, expected {pass_count}"
        )));
    }

    let zero_bitplanes = u8::try_from(zero_bitplanes)
        .map_err(|_| invalid("zero-bitplane count exceeds supported range"))?;
    let magnitude_bitplanes = max_bitplanes
        .checked_sub(zero_bitplanes)
        .ok_or_else(|| invalid("zero-bitplane count exceeds subband bit-plane count"))?;
    // A block may be signalled with more coding passes than its bit-planes
    // can hold (or with passes and no bit-planes at all): some encoders
    // include every block of an empty region with a zero-length pass in each
    // layer. There is nothing those passes could code, so decoding stops at
    // the last bit-plane, as OpenJPEG's does, instead of rejecting the image.
    if magnitude_bitplanes == 0 {
        return Ok(());
    }
    // Coefficients are represented as signed i32 magnitudes throughout this
    // decoder. A 32nd magnitude bit would make `1i32 << bitplane` negative,
    // and any still-higher bit would overflow the shift entirely. Reject such
    // signaling before entering the coding passes instead of panicking or
    // silently wrapping on malformed QCD/QCC metadata.
    if magnitude_bitplanes > 31 {
        return Err(invalid(format!(
            "code-block magnitude bit-plane count {magnitude_bitplanes} exceeds the i32 decoder limit of 31"
        )));
    }
    let max_passes = 1u32 + 3u32 * u32::from(magnitude_bitplanes.saturating_sub(1));
    let pass_limit = pass_count.min(max_passes);

    let top = magnitude_bitplanes - 1;
    let mut pass_index = 0u32;
    let mut contexts = None;

    for segment in segments {
        if segment.passes == 0 {
            return Err(invalid("codeword segment carries zero coding passes"));
        }
        // D.6: a raw segment holds significance and refinement passes of the
        // fifth and later significant bit-planes; MQ contexts are untouched.
        if style.pass_is_raw(pass_index) {
            let mut raw = RawDecoder::new(&segment.data);
            for _ in 0..segment.passes {
                if pass_index == pass_limit {
                    return Ok(());
                }
                let (pass_kind, bitplane) = pass_kind_and_bitplane(pass_index, top)?;
                let pass_start = stats.start();
                match pass_kind {
                    PassKind::Significance => sigpass_decode_raw(
                        &mut raw,
                        bitplane,
                        &mut scratch.flags,
                        &mut scratch.coefficients,
                        width,
                        height,
                        style.vertical_causal,
                    ),
                    PassKind::Refinement => refpass_decode_raw(
                        &mut raw,
                        bitplane,
                        &mut scratch.flags,
                        &mut scratch.coefficients,
                        width,
                        height,
                    ),
                    PassKind::Cleanup => {
                        return Err(invalid("cleanup pass inside a raw bypass codeword segment"));
                    }
                }
                stats.finish(pass_start, |stats, elapsed| match pass_kind {
                    PassKind::Significance => {
                        stats.tier1_significance_ns =
                            stats.tier1_significance_ns.saturating_add(elapsed);
                    }
                    PassKind::Refinement => {
                        stats.tier1_refinement_ns =
                            stats.tier1_refinement_ns.saturating_add(elapsed);
                    }
                    PassKind::Cleanup => {}
                });
                pass_index += 1;
            }
            continue;
        }
        let mut decoder = match contexts {
            Some(saved) if !style.reset_contexts => {
                MqDecoder::new_with_contexts(&segment.data, saved)
            }
            _ => MqDecoder::new(&segment.data),
        };

        for _ in 0..segment.passes {
            if pass_index == pass_limit {
                return Ok(());
            }
            let (pass_kind, bitplane) = pass_kind_and_bitplane(pass_index, top)?;
            if style.pass_is_raw(pass_index) {
                return Err(invalid("bypass-coded pass inside an MQ codeword segment"));
            }
            let pass_start = stats.start();
            match pass_kind {
                PassKind::Significance => sigpass_decode(
                    &mut decoder,
                    band,
                    bitplane,
                    &mut scratch.flags,
                    &mut scratch.coefficients,
                    width,
                    height,
                    style.vertical_causal,
                ),
                PassKind::Refinement => refpass_decode(
                    &mut decoder,
                    bitplane,
                    &mut scratch.flags,
                    &mut scratch.coefficients,
                    width,
                    height,
                ),
                PassKind::Cleanup => {
                    cleanup_decode(
                        &mut decoder,
                        band,
                        bitplane,
                        &mut scratch.flags,
                        &mut scratch.coefficients,
                        width,
                        height,
                        style.vertical_causal,
                    );
                    if style.segmentation_symbols {
                        read_segmentation_symbol(&mut decoder, pass_index + 1 == pass_count)?;
                    }
                    debug_assert!(
                        !scratch.flags.has_visited(),
                        "cleanup pass must clear every visited flag"
                    );
                }
            }
            stats.finish(pass_start, |stats, elapsed| match pass_kind {
                PassKind::Significance => {
                    stats.tier1_significance_ns =
                        stats.tier1_significance_ns.saturating_add(elapsed);
                }
                PassKind::Refinement => {
                    stats.tier1_refinement_ns = stats.tier1_refinement_ns.saturating_add(elapsed);
                }
                PassKind::Cleanup => {
                    stats.tier1_cleanup_ns = stats.tier1_cleanup_ns.saturating_add(elapsed);
                }
            });
            pass_index += 1;
            if style.reset_contexts {
                decoder.reset_contexts();
            }
        }
        contexts = Some(decoder.contexts());
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
pub(super) enum PassKind {
    Significance,
    Refinement,
    Cleanup,
}

pub(super) fn pass_kind_and_bitplane(pass_index: u32, top: u8) -> Result<(PassKind, u8)> {
    if pass_index == 0 {
        return Ok((PassKind::Cleanup, top));
    }
    let after_top = pass_index - 1;
    let plane_offset = after_top / 3 + 1;
    let bitplane = u32::from(top)
        .checked_sub(plane_offset)
        .and_then(|value| u8::try_from(value).ok())
        .ok_or_else(|| invalid("coding-pass index exceeds magnitude bit-planes"))?;
    let kind = match after_top % 3 {
        0 => PassKind::Significance,
        1 => PassKind::Refinement,
        _ => PassKind::Cleanup,
    };
    Ok((kind, bitplane))
}

/// The per-pass kernels below walk a code block stripe by stripe (four rows)
/// and column by column, holding the column's packed flag word in a register
/// and writing it back once.
///
/// Every sample is coded in exactly one of a bit-plane's three passes, so a
/// pass that asked each sample in turn whether it belongs to it would spend
/// most of its time on samples it then skips. Instead each kernel derives,
/// from the column's flag word, the set of samples it has to code — one bit
/// per sample at bit `3 * ci`, the same spacing the flag word uses — and
/// visits only those. A column with nothing to code costs one test.
///
/// Decoded values are stored signed: a newly significant sample at bit-plane
/// `b` becomes `±(2^b | 2^(b-1))` (the midpoint of its uncertainty interval)
/// and refinement moves it by `±2^(b-1)`, so the final magnitude is the same
/// as accumulating an unsigned magnitude and applying the sign at the end.
///
/// One bit per sample of a full stripe column, at bit `3 * ci`.
const COLUMN_SAMPLES: u32 = 0o1111;

/// One four-row stripe of a code block: the three flag rows a column walk
/// touches (the stripe above, the stripe itself, the stripe below), each
/// exactly `width + 2` words, and the stripe's samples, one `[i32; 4]` per
/// column, exactly `width` of them. With the lengths visible to the
/// compiler, indexing column `x < width` and its neighbours needs no bounds
/// checks.
struct Stripe<'a> {
    north: &'a mut [u32],
    mid: &'a mut [u32],
    south: &'a mut [u32],
    columns: &'a mut [[i32; 4]],
    /// [`COLUMN_SAMPLES`] restricted to the rows this stripe has: the last
    /// stripe of a block whose height is not a multiple of four is short.
    rows: u32,
}

impl<'a> Stripe<'a> {
    #[inline(always)]
    fn new(
        flags: &'a mut StripeFlags,
        data: &'a mut [i32],
        stripe: usize,
        width: usize,
        height: usize,
    ) -> Self {
        let stride = width + 2;
        debug_assert_eq!(stride, flags.stride());
        let words = flags.words_mut();
        let (_, rest) = words.split_at_mut(stripe * stride);
        let (north, rest) = rest.split_at_mut(stride);
        let (mid, rest) = rest.split_at_mut(stride);
        let south = &mut rest[..stride];
        let (columns, _) = data[4 * stripe * width..].as_chunks_mut::<4>();
        let rows = (height - 4 * stripe).min(4) as u32;
        Self {
            north,
            mid,
            south,
            columns: &mut columns[..width],
            rows: COLUMN_SAMPLES >> (3 * (4 - rows)),
        }
    }

    /// The sample of column `x` whose bit in a sample set is `shift` (`0, 3, 6, 9`).
    #[inline(always)]
    fn sample(&mut self, x: usize, shift: u32) -> &mut i32 {
        debug_assert!(matches!(shift, 0 | 3 | 6 | 9));
        &mut self.columns[x][((shift + 3) >> 2) as usize & 3]
    }
}

/// The samples of a column the significance propagation pass codes: not yet
/// significant, not already coded in this pass, and with at least one
/// significant neighbour (D.3.1).
#[inline(always)]
fn significance_candidates(word: u32) -> u32 {
    // Bit `3 j` of `any_in_row` says row `j` of the 3 × 6 neighbourhood has
    // a significant sample, and of `beside` that its west or east one is.
    // Sample `ci` sits in row `ci + 1`: its neighbours are all of rows `ci`
    // and `ci + 2` and the two beside it.
    let any_in_row = word | (word >> 1) | (word >> 2);
    let beside = word | (word >> 2);
    let has_neighbour = any_in_row | (any_in_row >> 6) | (beside >> 3);
    has_neighbour & !(word >> 4) & !(word >> 21) & COLUMN_SAMPLES
}

/// Sign-coding LUT index (Table D.3 contribution bits) for the sample at
/// `shift` from its column word and the west and east column words.
#[inline(always)]
fn sign_lut_index(word: u32, west: u32, east: u32, shift: u32) -> usize {
    let mut lu = (word >> shift) & (T1_SIGMA_N | T1_SIGMA_W | T1_SIGMA_E | T1_SIGMA_S);
    lu |= (west >> (T1_CHI_1_I + shift)) & T1_LUT_SGN_W;
    lu |= (east >> (T1_CHI_1_I - 2 + shift)) & T1_LUT_SGN_E;
    // The sign above sample 0 is the stripe above's; above the others it is
    // the previous sample's.
    let north = if shift == 0 {
        T1_CHI_0_I - 4
    } else {
        T1_CHI_1_I - 7 + shift
    };
    lu |= (word >> north) & T1_LUT_SGN_N;
    lu |= (word >> (T1_CHI_2_I - 6 + shift)) & T1_LUT_SGN_S;
    (lu & 0xff) as usize
}

/// Record that the sample at `shift` of column `x` became significant with
/// sign `sign` (1 = negative): its own word (kept in `word`) and the
/// neighbouring columns' words learn about it. In vertically causal mode the
/// stripe above never learns about row 0 (Annex D.7).
#[inline(always)]
fn mark_significant(
    s: &mut Stripe<'_>,
    x: usize,
    word: &mut u32,
    shift: u32,
    sign: u32,
    vertical_causal: bool,
) {
    s.mid[x] |= T1_SIGMA_E << shift;
    *word |= ((sign << T1_CHI_1_I) | T1_SIGMA_THIS) << shift;
    s.mid[x + 2] |= T1_SIGMA_W << shift;
    if shift == 0 && !vertical_causal {
        s.north[x + 1] |= (sign << T1_CHI_5_I) | T1_SIGMA_16;
        s.north[x] |= T1_SIGMA_17;
        s.north[x + 2] |= T1_SIGMA_15;
    }
    if shift == 9 {
        s.south[x + 1] |= (sign << T1_CHI_0_I) | T1_SIGMA_N;
        s.south[x] |= T1_SIGMA_NE;
        s.south[x + 2] |= T1_SIGMA_NW;
    }
}

/// Decode the sign of a sample that just became significant and store its
/// midpoint value.
#[inline(always)]
fn decode_new_significant(
    decoder: &mut MqDecoder<'_>,
    s: &mut Stripe<'_>,
    x: usize,
    word: &mut u32,
    shift: u32,
    one_plus_half: i32,
    vertical_causal: bool,
) {
    let lu = sign_lut_index(*word, s.mid[x], s.mid[x + 2], shift);
    let sign = u32::from(decoder.decode_with_ctx(LUT_CTXNO_SC[lu]) ^ LUT_SPB[lu]);
    *s.sample(x, shift) = if sign != 0 {
        -one_plus_half
    } else {
        one_plus_half
    };
    mark_significant(s, x, word, shift, sign, vertical_causal);
}

/// Significance propagation pass (D.3.1) over the whole block.
#[allow(clippy::too_many_arguments)]
fn sigpass_decode(
    decoder: &mut MqDecoder<'_>,
    band: BandOrientation,
    bitplane: u8,
    flags: &mut StripeFlags,
    data: &mut [i32],
    width: usize,
    height: usize,
    vertical_causal: bool,
) {
    let one = 1i32 << bitplane;
    let one_plus_half = one | (one >> 1);
    let lut = &LUT_CTXNO_ZC[orientation_index(band)];
    for stripe in 0..height.div_ceil(4) {
        let mut s = Stripe::new(flags, data, stripe, width, height);
        for x in 0..width {
            let mut word = s.mid[x + 1];
            let mut pending = significance_candidates(word) & s.rows;
            if pending == 0 {
                continue;
            }
            loop {
                let shift = pending.trailing_zeros();
                let ctx = lut[((word >> shift) & T1_SIGMA_NEIGHBOURS) as usize];
                word |= T1_PI_0 << shift;
                if decoder.decode_with_ctx(ctx) != 0 {
                    decode_new_significant(
                        decoder,
                        &mut s,
                        x,
                        &mut word,
                        shift,
                        one_plus_half,
                        vertical_causal,
                    );
                    // The sample below just gained a neighbour.
                    pending = significance_candidates(word) & s.rows;
                }
                // Only samples after this one in the scan remain.
                pending &= !1 << shift;
                if pending == 0 {
                    break;
                }
            }
            s.mid[x + 1] = word;
        }
    }
}

/// Raw significance pass (D.6): the same traversal and membership test as
/// [`sigpass_decode`], with the significance bit and then the sign read as
/// plain bits, no context and no sign prediction.
fn sigpass_decode_raw(
    raw: &mut RawDecoder<'_>,
    bitplane: u8,
    flags: &mut StripeFlags,
    data: &mut [i32],
    width: usize,
    height: usize,
    vertical_causal: bool,
) {
    let one = 1i32 << bitplane;
    let one_plus_half = one | (one >> 1);
    for stripe in 0..height.div_ceil(4) {
        let mut s = Stripe::new(flags, data, stripe, width, height);
        for x in 0..width {
            let mut word = s.mid[x + 1];
            let mut pending = significance_candidates(word) & s.rows;
            if pending == 0 {
                continue;
            }
            loop {
                let shift = pending.trailing_zeros();
                word |= T1_PI_0 << shift;
                if raw.decode_bit() != 0 {
                    let sign = u32::from(raw.decode_bit());
                    *s.sample(x, shift) = if sign != 0 {
                        -one_plus_half
                    } else {
                        one_plus_half
                    };
                    mark_significant(&mut s, x, &mut word, shift, sign, vertical_causal);
                    pending = significance_candidates(word) & s.rows;
                }
                pending &= !1 << shift;
                if pending == 0 {
                    break;
                }
            }
            s.mid[x + 1] = word;
        }
    }
}

/// The samples of a column the magnitude refinement pass codes: significant
/// before this bit-plane's significance pass (D.3.3).
#[inline(always)]
fn refinement_samples(word: u32) -> u32 {
    (word >> 4) & !(word >> 21) & COLUMN_SAMPLES
}

/// Midpoint update for a refined sample. Entering MR at plane b the magnitude
/// carries a 2^b offset (midpoint of the previous uncertainty interval
/// [0, 2^(b+1))). Learning bit b halves the interval: offset becomes 2^(b-1).
///   bit=1: known += 2^b            → mag += 2^(b-1)
///   bit=0: known unchanged         → mag -= 2^b - 2^(b-1)
/// At b=0 this reduces to bit=1: exact, bit=0: mag -= 1 (exact).
#[inline(always)]
fn refine_sample(value: &mut i32, bit: u8, one: i32) {
    let half = one >> 1;
    let delta = half - (one & (i32::from(bit) - 1));
    // Apply the sample's sign to `delta` without a branch.
    let sign = *value >> 31;
    *value += (delta ^ sign) - sign;
}

/// Magnitude refinement context (Table D.4) of the sample at `shift`.
#[inline(always)]
fn magnitude_context(word: u32, shift: u32) -> u8 {
    let refined_before = (word >> shift) & T1_MU_0 != 0;
    let has_neighbour = (word >> shift) & T1_SIGMA_NEIGHBOURS != 0;
    T1_CTXNO_MAG + (2 * u8::from(refined_before)).max(u8::from(has_neighbour))
}

/// Magnitude refinement pass (D.3.3) over the whole block.
fn refpass_decode(
    decoder: &mut MqDecoder<'_>,
    bitplane: u8,
    flags: &mut StripeFlags,
    data: &mut [i32],
    width: usize,
    height: usize,
) {
    let one = 1i32 << bitplane;
    for stripe in 0..height.div_ceil(4) {
        let mut s = Stripe::new(flags, data, stripe, width, height);
        for x in 0..width {
            let word = s.mid[x + 1];
            let refined = refinement_samples(word);
            if refined == 0 {
                continue;
            }
            let mut pending = refined;
            loop {
                let shift = pending.trailing_zeros();
                let bit = decoder.decode_with_ctx(magnitude_context(word, shift));
                refine_sample(s.sample(x, shift), bit, one);
                pending &= pending - 1;
                if pending == 0 {
                    break;
                }
            }
            s.mid[x + 1] = word | (refined * T1_MU_0);
        }
    }
}

/// Raw magnitude refinement pass (D.6): [`refpass_decode`] with the
/// refinement bit read as a plain bit.
fn refpass_decode_raw(
    raw: &mut RawDecoder<'_>,
    bitplane: u8,
    flags: &mut StripeFlags,
    data: &mut [i32],
    width: usize,
    height: usize,
) {
    let one = 1i32 << bitplane;
    for stripe in 0..height.div_ceil(4) {
        let mut s = Stripe::new(flags, data, stripe, width, height);
        for x in 0..width {
            let word = s.mid[x + 1];
            let refined = refinement_samples(word);
            let mut pending = refined;
            while pending != 0 {
                let shift = pending.trailing_zeros();
                refine_sample(s.sample(x, shift), raw.decode_bit(), one);
                pending &= pending - 1;
            }
            s.mid[x + 1] = word | (refined * T1_MU_0);
        }
    }
}

/// Cleanup pass (D.3.4) over the whole block, with run-length coding of
/// columns whose word is zero in a full stripe.
#[allow(clippy::too_many_arguments)]
fn cleanup_decode(
    decoder: &mut MqDecoder<'_>,
    band: BandOrientation,
    bitplane: u8,
    flags: &mut StripeFlags,
    data: &mut [i32],
    width: usize,
    height: usize,
    vertical_causal: bool,
) {
    let one = 1i32 << bitplane;
    let one_plus_half = one | (one >> 1);
    let lut = &LUT_CTXNO_ZC[orientation_index(band)];
    for stripe in 0..height.div_ceil(4) {
        let mut s = Stripe::new(flags, data, stripe, width, height);
        for x in 0..width {
            let mut word = s.mid[x + 1];
            let mut pending;
            if word == 0 && s.rows == COLUMN_SAMPLES {
                // Run-length mode: one symbol says the whole column stays
                // insignificant; otherwise two more name the first sample
                // that does not, and only its sign is left to read.
                if decoder.decode_with_ctx(T1_CTXNO_AGG) == 0 {
                    continue;
                }
                let hi = u32::from(decoder.decode_with_ctx(T1_CTXNO_UNI));
                let lo = u32::from(decoder.decode_with_ctx(T1_CTXNO_UNI));
                let shift = 3 * ((hi << 1) | lo);
                decode_new_significant(
                    decoder,
                    &mut s,
                    x,
                    &mut word,
                    shift,
                    one_plus_half,
                    vertical_causal,
                );
                pending = COLUMN_SAMPLES & (!1 << shift);
            } else {
                // Everything neither significant nor coded in this
                // bit-plane's significance pass.
                pending = !(word >> 4) & !(word >> 21) & s.rows;
            }
            while pending != 0 {
                let shift = pending.trailing_zeros();
                let ctx = lut[((word >> shift) & T1_SIGMA_NEIGHBOURS) as usize];
                if decoder.decode_with_ctx(ctx) != 0 {
                    decode_new_significant(
                        decoder,
                        &mut s,
                        x,
                        &mut word,
                        shift,
                        one_plus_half,
                        vertical_causal,
                    );
                }
                pending &= pending - 1;
            }
            s.mid[x + 1] = word & !T1_PI_ALL;
        }
    }
}

pub(super) fn invalid(message: impl Into<String>) -> Jp2LamError {
    Jp2LamError::DecodeFailed(message.into())
}

#[cfg(test)]
mod tests {
    use super::{decode_codeblock, decode_codeblock_segments};
    use crate::decode::t2::DecodedCodewordSegment;
    use crate::j2k::decode_markers::CodeBlockStyle;
    use crate::mq::{MqCoder, T1_CTXNO_MAG, T1_CTXNO_ZC};
    use crate::plan::BandOrientation;
    use crate::tier1::helpers::{sign_context, sign_prediction_bit};
    use std::borrow::Cow;

    #[test]
    fn cleanup_only_single_sample_decodes_positive_one() {
        let mut coder = MqCoder::new();
        coder.encode_with_ctx(T1_CTXNO_ZC, 1);
        let sign_ctx = sign_context(0);
        let prediction = sign_prediction_bit(0);
        coder.encode_with_ctx(sign_ctx, prediction);
        let bytes = coder.finish();

        let decoded =
            decode_codeblock(1, 1, BandOrientation::Ll, 1, 0, 1, false, &bytes).expect("decode");
        assert_eq!(decoded.coefficients, vec![1]);
    }

    #[test]
    fn cleanup_and_refinement_single_sample_decodes_negative_three() {
        let mut coder = MqCoder::new();
        coder.encode_with_ctx(T1_CTXNO_ZC, 1);
        let sign_ctx = sign_context(0);
        let prediction = sign_prediction_bit(0);
        coder.encode_with_ctx(sign_ctx, 1 ^ prediction);
        coder.encode_with_ctx(T1_CTXNO_MAG, 1);
        let bytes = coder.finish();

        let decoded =
            decode_codeblock(1, 1, BandOrientation::Ll, 2, 0, 3, false, &bytes).expect("decode");
        assert_eq!(decoded.coefficients, vec![-3]);
    }

    #[test]
    fn d4_d5_termall_reset_and_segmark_decode_across_pass_segments() {
        let sign_ctx = sign_context(0);
        let prediction = sign_prediction_bit(0);

        let mut cleanup = MqCoder::new();
        cleanup.encode_with_ctx(T1_CTXNO_ZC, 1);
        cleanup.encode_with_ctx(sign_ctx, prediction);
        cleanup.segmark_encode();
        let cleanup = cleanup.finish();

        // The significance pass has no symbols for an already significant
        // one-sample block. A zero-byte terminated segment is valid; Annex
        // D.4.1 supplies synthetic 0xFF bytes if the decoder needs symbols.
        let significance = Vec::new();

        let mut refinement = MqCoder::new();
        refinement.encode_with_ctx(T1_CTXNO_MAG, 1);
        let refinement = refinement.finish();

        let segments = [
            DecodedCodewordSegment {
                passes: 1,
                data: Cow::Borrowed(&cleanup),
            },
            DecodedCodewordSegment {
                passes: 1,
                data: Cow::Borrowed(&significance),
            },
            DecodedCodewordSegment {
                passes: 1,
                data: Cow::Borrowed(&refinement),
            },
        ];
        let decoded = decode_codeblock_segments(
            1,
            1,
            BandOrientation::Ll,
            2,
            0,
            3,
            CodeBlockStyle {
                reset_contexts: true,
                terminate_each_pass: true,
                predictable_termination: true,
                segmentation_symbols: true,
                ..CodeBlockStyle::default()
            },
            &segments,
        )
        .expect("terminated pass segments");
        assert_eq!(decoded.coefficients, vec![3]);
    }

    #[test]
    fn d41_empty_segment_uses_synthesized_end_bytes() {
        let decoded = decode_codeblock(1, 1, BandOrientation::Ll, 1, 0, 1, false, &[])
            .expect("MQ decoder synthesizes end bytes");
        assert_eq!(decoded.coefficients, vec![0]);
    }

    #[test]
    fn passes_beyond_the_last_bit_plane_are_ignored() {
        let mut coder = MqCoder::new();
        coder.encode_with_ctx(T1_CTXNO_ZC, 1);
        let sign_ctx = sign_context(0);
        coder.encode_with_ctx(sign_ctx, sign_prediction_bit(0));
        let bytes = coder.finish();

        // One bit-plane holds one (cleanup) pass; a second signalled pass and
        // passes signalled with no bit-plane at all code nothing.
        let decoded = decode_codeblock(1, 1, BandOrientation::Ll, 1, 0, 2, false, &bytes)
            .expect("extra pass is ignored");
        assert_eq!(decoded.coefficients, vec![1]);
        let decoded = decode_codeblock(1, 1, BandOrientation::Ll, 1, 1, 1, false, &bytes)
            .expect("passes without bit-planes are ignored");
        assert_eq!(decoded.coefficients, vec![0]);
    }

    #[test]
    fn magnitude_bitplanes_that_do_not_fit_i32_fail_fast() {
        let err = decode_codeblock(1, 1, BandOrientation::Ll, 32, 0, 1, false, &[0])
            .expect_err("32 magnitude bits cannot be represented by i32")
            .to_string();

        assert!(err.contains("i32 decoder limit"), "{err}");
    }
}

/// The packed stripe-column decoder must reproduce the previous per-sample
/// decoder bit for bit, for every code-block style switch and for truncated
/// pass counts. Blocks are produced by the in-tree encoder.
#[cfg(test)]
mod stripe_equivalence_tests {
    use super::decode_codeblock_segments;
    use crate::decode::t1_reference::decode_codeblock_segments_reference;
    use crate::decode::t2::DecodedCodewordSegment;
    use crate::encode::backend::native::NativeComponentCoefficients;
    use crate::encode::backend::native::layout::build_component_layout;
    use crate::encode::backend::native::t1::{
        NativeEncodedTier1CodeBlock, NativeTier1CodeBlock, analyze_component_layout,
        encode_codeblock_styled,
    };
    use crate::j2k::decode_markers::CodeBlockStyle;
    use crate::plan::CodeBlockSize;
    use std::borrow::Cow;

    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.0 >> 33) as u32
        }
    }

    /// Coefficient fields with different statistics: sparse (mostly run-length
    /// coded), dense small, dense with a few huge values (many bit-planes).
    fn field(rng: &mut Lcg, len: usize, flavour: u32) -> Vec<i32> {
        (0..len)
            .map(|_| {
                let r = rng.next();
                let sign = if r & 1 == 0 { 1 } else { -1 };
                match flavour {
                    0 => {
                        if r % 11 == 0 {
                            sign * ((r >> 8) % 40) as i32
                        } else {
                            0
                        }
                    }
                    1 => sign * ((r >> 8) % 9) as i32,
                    2 => {
                        if r % 37 == 0 {
                            sign * (1000 + (r >> 8) % 3000) as i32
                        } else {
                            sign * ((r >> 8) % 5) as i32
                        }
                    }
                    _ => {
                        if r % 3 == 0 {
                            sign * ((r >> 8) % 300) as i32
                        } else {
                            0
                        }
                    }
                }
            })
            .collect()
    }

    fn encoded_blocks(
        width: usize,
        height: usize,
        levels: u8,
        coefficients: Vec<i32>,
        style: CodeBlockStyle,
    ) -> Vec<(NativeTier1CodeBlock, NativeEncodedTier1CodeBlock)> {
        let layout = build_component_layout(
            &NativeComponentCoefficients {
                x0: 0,
                y0: 0,
                width,
                height,
                levels,
                data: coefficients,
            },
            CodeBlockSize {
                width: 64,
                height: 64,
            },
        )
        .expect("layout");
        analyze_component_layout(&layout)
            .bands
            .into_iter()
            .flat_map(|band| band.blocks)
            .map(|block| {
                let encoded = encode_codeblock_styled(&block, style);
                (block, encoded)
            })
            .collect()
    }

    /// Split a styled block's payload into its terminated codeword segments,
    /// keeping only the first `pass_count` passes (a whole number of segments
    /// or a truncated last one, as Tier-2 would deliver them).
    fn segments(
        encoded: &NativeEncodedTier1CodeBlock,
        pass_count: usize,
    ) -> Vec<DecodedCodewordSegment<'static>> {
        let mut out = Vec::new();
        let mut start = 0usize;
        let mut passes = 0u32;
        for (index, pass) in encoded.passes.iter().enumerate().take(pass_count) {
            passes += 1;
            if pass.is_terminated || index + 1 == pass_count {
                out.push(DecodedCodewordSegment {
                    passes,
                    data: Cow::Owned(encoded.payload[start..pass.cumulative_length].to_vec()),
                });
                start = pass.cumulative_length;
                passes = 0;
            }
        }
        out
    }

    fn check(
        style: CodeBlockStyle,
        width: usize,
        height: usize,
        levels: u8,
        flavour: u32,
        seed: u64,
    ) {
        let mut rng = Lcg(seed);
        let coefficients = field(&mut rng, width * height, flavour);
        for (block, encoded) in encoded_blocks(width, height, levels, coefficients, style) {
            if encoded.passes.is_empty() {
                continue;
            }
            let max_bitplanes = block.zero_bitplanes + block.magnitude_bitplanes;
            let total = encoded.passes.len();
            // Every prefix length up to a few, then the whole block.
            let mut counts: Vec<usize> = (1..=total.min(4)).collect();
            counts.push(total);
            if total > 6 {
                counts.push(total - 1);
                counts.push(total / 2);
            }
            for pass_count in counts {
                let segments = segments(&encoded, pass_count);
                let fast = decode_codeblock_segments(
                    block.width,
                    block.height,
                    block.band,
                    max_bitplanes,
                    u32::from(block.zero_bitplanes),
                    pass_count as u32,
                    style,
                    &segments,
                )
                .map(|decoded| decoded.coefficients)
                .map_err(|error| error.to_string());
                let reference = decode_codeblock_segments_reference(
                    block.width,
                    block.height,
                    block.band,
                    max_bitplanes,
                    u32::from(block.zero_bitplanes),
                    pass_count as u32,
                    style,
                    &segments,
                )
                .map_err(|error| error.to_string());
                // A truncated prefix that ends between two cleanup passes can
                // leave a segmentation symbol short of bytes; both decoders
                // must then report the same error.
                assert_eq!(
                    fast, reference,
                    "{style:?} {}x{} {:?} flavour {flavour} passes {pass_count}/{total}",
                    block.width, block.height, block.band
                );
                if pass_count == total {
                    assert_eq!(
                        fast.as_deref().ok(),
                        Some(block.coefficients.as_slice()),
                        "{style:?} {}x{} {:?} flavour {flavour}: full decode is lossless",
                        block.width,
                        block.height,
                        block.band
                    );
                }
            }
        }
    }

    fn styles() -> Vec<CodeBlockStyle> {
        let d = CodeBlockStyle::default();
        vec![
            d,
            CodeBlockStyle {
                vertical_causal: true,
                ..d
            },
            CodeBlockStyle {
                reset_contexts: true,
                ..d
            },
            CodeBlockStyle {
                terminate_each_pass: true,
                ..d
            },
            CodeBlockStyle { bypass: true, ..d },
            CodeBlockStyle {
                segmentation_symbols: true,
                ..d
            },
            CodeBlockStyle {
                bypass: true,
                vertical_causal: true,
                reset_contexts: true,
                terminate_each_pass: true,
                predictable_termination: true,
                segmentation_symbols: true,
            },
            CodeBlockStyle {
                bypass: true,
                vertical_causal: true,
                segmentation_symbols: true,
                ..d
            },
        ]
    }

    #[test]
    fn packed_decoder_matches_reference_on_square_blocks() {
        for (i, style) in styles().into_iter().enumerate() {
            for flavour in 0..4 {
                check(
                    style,
                    64,
                    64,
                    0,
                    flavour,
                    1 + i as u64 * 10 + flavour as u64,
                );
                check(
                    style,
                    128,
                    128,
                    1,
                    flavour,
                    100 + i as u64 * 10 + flavour as u64,
                );
            }
        }
    }

    /// With segmentation symbols every cleanup pass ends in a marker the
    /// decoder verifies, so every pass prefix a layer can end on must decode
    /// without a marker error: the per-pass lengths have to cover the bytes
    /// the MQ coder still holds in its registers at the pass boundary.
    #[test]
    fn segmentation_symbol_prefixes_decode_at_every_pass() {
        let d = CodeBlockStyle::default();
        let styles = [
            CodeBlockStyle {
                segmentation_symbols: true,
                ..d
            },
            CodeBlockStyle {
                segmentation_symbols: true,
                reset_contexts: true,
                ..d
            },
            CodeBlockStyle {
                segmentation_symbols: true,
                vertical_causal: true,
                ..d
            },
            CodeBlockStyle {
                segmentation_symbols: true,
                bypass: true,
                ..d
            },
            CodeBlockStyle {
                segmentation_symbols: true,
                bypass: true,
                vertical_causal: true,
                reset_contexts: true,
                ..d
            },
        ];
        let mut checked = 0usize;
        for (i, style) in styles.into_iter().enumerate() {
            for &(w, h, levels) in &[
                (64usize, 64usize, 0u8),
                (128, 128, 1),
                (61, 59, 1),
                (5, 7, 0),
            ] {
                for flavour in 0..4u32 {
                    let mut rng = Lcg(5000 + i as u64 * 100 + (w * h) as u64 + flavour as u64);
                    let coefficients = field(&mut rng, w * h, flavour);
                    for (block, encoded) in encoded_blocks(w, h, levels, coefficients, style) {
                        let total = encoded.passes.len();
                        let max_bitplanes = block.zero_bitplanes + block.magnitude_bitplanes;
                        for pass_count in 1..=total {
                            let segments = segments(&encoded, pass_count);
                            let decoded = decode_codeblock_segments(
                                block.width,
                                block.height,
                                block.band,
                                max_bitplanes,
                                u32::from(block.zero_bitplanes),
                                pass_count as u32,
                                style,
                                &segments,
                            )
                            .unwrap_or_else(|error| {
                                panic!(
                                    "{style:?} {}x{} {:?} flavour {flavour} passes {pass_count}/{total}: {error}",
                                    block.width, block.height, block.band
                                )
                            });
                            if pass_count == total {
                                assert_eq!(decoded.coefficients, block.coefficients);
                            }
                            checked += 1;
                        }
                    }
                }
            }
        }
        assert!(checked > 1000, "{checked} prefixes checked");
    }

    #[test]
    fn packed_decoder_matches_reference_on_odd_blocks() {
        for (i, style) in styles().into_iter().enumerate() {
            for &(w, h, levels) in &[
                (1usize, 1usize, 0u8),
                (5, 7, 0),
                (17, 33, 0),
                (64, 6, 0),
                (3, 64, 0),
                (9, 10, 1),
                (61, 59, 1),
                (2, 5, 0),
                (64, 63, 0),
                (63, 1, 0),
                (1, 63, 0),
            ] {
                for flavour in [1u32, 3] {
                    check(
                        style,
                        w,
                        h,
                        levels,
                        flavour,
                        1000 + i as u64 * 100 + (w * h) as u64 + flavour as u64,
                    );
                }
            }
        }
    }
}
