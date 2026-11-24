//! Generic refinement-region decoding, GRTEMPLATE 0 (jbig2decplan.md §17, T.88
//! §6.3).
//!
//! Inverts the encoder's
//! [`crate::encode::arith::Jbig2ArithCoder::encode_refinement_region`]: a 13-bit
//! context is formed from a 3×3 reference neighbourhood, three already-decoded
//! target pixels, and one target adaptive-template pixel (GRAT1, nominally
//! `(-1, -1)`). The reference is offset from the target by `(GRDX, GRDY)`.
//!
//! The context bank is shared across every refined instance of one text region
//! (T.88 §6.4.11 carries the GR statistics across instances) and is *not* reset
//! per instance — the caller resets it once per text-region segment.

use crate::decode::arith::ArithmeticDecoder;
use crate::decode::error::{DecodeError, LimitError};
use crate::shared::bitmap::{CombinationOperator, MonoBitmap};
use crate::shared::limits::DecodeLimits;
use crate::shared::mq_table::MqContext;
use crate::shared::reader::Reader;

/// The number of refinement contexts. GRTEMPLATE-0 uses a 13-bit context;
/// GRTEMPLATE-1 uses a 10-bit context that fits in the same bank.
pub const REFINEMENT_CONTEXT_COUNT: usize = 1 << 13;

// SLTP pseudo-pixel contexts for TPGRON (T.88 §6.3.5.6, Figures 14/15 set the
// centre reference pixel to 1). The SLTP context is the index whose only set
// bit is the centre-reference pixel GRREFERENCE(w, h). In this crate's context
// numbering that pixel is bit 4 for GRTEMPLATE-0 (0x010) and bit 7 for
// GRTEMPLATE-1 (0x080). These match pdfium's CJBig2_GRRDProc, which decodes
// SLTP with grContexts[0x0010] (template 0) and grContexts[0x0008] (template 1)
// — both the centre reference in pdfium's own numbering. The earlier 0x040
// (bit 6 = GRREFERENCE(w+1, h)) desynced every template-1 TPGRON region
// (pdf.js bitmap-refine-template1-tpgron).
const SLTP_CTX_GR0: usize = 0x010;
const SLTP_CTX_GR1: usize = 0x080;

/// A parsed standalone generic-refinement-region segment header (T.88 §7.4.7),
/// plus the arithmetic-coded payload that follows it.
pub struct RefinementRegionSegment<'a> {
    pub width: u32,
    pub height: u32,
    pub x: u32,
    pub y: u32,
    /// External combination operator (region-info flags low 3 bits).
    pub comb_operator: u8,
    /// GRTEMPLATE (0 or 1).
    pub grtemplate: u8,
    /// TPGRON typical-prediction flag.
    pub tpgron: bool,
    /// GRAT1 (target adaptive-template pixel), nominally `(-1, -1)`. Unused for
    /// GRTEMPLATE-1, which has no adaptive pixel.
    pub grat: (i8, i8),
    /// GRAT2 (reference adaptive-template pixel), nominally `(-1, -1)`. Forms
    /// context bit 0 for GRTEMPLATE-0 (T.88 Fig. 12); unused for GRTEMPLATE-1.
    pub grat2: (i8, i8),
    pub data: &'a [u8],
}

/// Parse a generic refinement region segment data header (T.88 §7.4.7.1–7.4.7.3).
pub fn parse_refinement_region(payload: &[u8]) -> Result<RefinementRegionSegment<'_>, DecodeError> {
    let mut r = Reader::new(payload);
    // §7.4.1 region segment information field.
    let width = r.read_u32_be()?;
    let height = r.read_u32_be()?;
    let x = r.read_u32_be()?;
    let y = r.read_u32_be()?;
    let region_flags = r.read_u8()?;
    let comb_operator = region_flags & 0x07;

    // §7.4.7.2 generic refinement region segment flags.
    let flags = r.read_u8()?;
    let grtemplate = flags & 0x01;
    let tpgron = flags & 0x02 != 0;

    // §7.4.7.3 AT flags: present only when GRTEMPLATE=0 (two AT pairs, 4 bytes).
    // Only GRAT1 is used by the template-0 context; template 1 has no AT pixel.
    let (grat, grat2) = if grtemplate == 0 {
        let g1x = r.read_i8()?;
        let g1y = r.read_i8()?;
        let g2x = r.read_i8()?;
        let g2y = r.read_i8()?;
        ((g1x, g1y), (g2x, g2y))
    } else {
        ((-1, -1), (-1, -1))
    };

    let data = &payload[r.position()..];
    Ok(RefinementRegionSegment {
        width,
        height,
        x,
        y,
        comb_operator,
        grtemplate,
        tpgron,
        grat,
        grat2,
        data,
    })
}

/// Extract the page-buffer sub-region `[x, y, width, height]` as a fresh bitmap,
/// the `GRREFERENCE` for a standalone refinement region with no referred region
/// segment (T.88 §7.4.7.4). Pixels of the box that fall outside the page are 0.
pub fn page_reference_window(
    page: &MonoBitmap,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    limits: &DecodeLimits,
) -> Result<MonoBitmap, DecodeError> {
    let pixels = (width as u64)
        .checked_mul(height as u64)
        .ok_or(DecodeError::Overflow {
            operation: "refinement window pixels",
        })?;
    if pixels > limits.max_page_pixels {
        return Err(DecodeError::limit(LimitError::Pixels {
            what: "refinement window",
            value: pixels,
            limit: limits.max_page_pixels,
        }));
    }
    let mut window = MonoBitmap::new(width, height, false, limits)?;
    if x < page.width()
        && y < page.height()
        && let (Ok(xi), Ok(yi)) = (i32::try_from(x), i32::try_from(y))
    {
        window.combine(
            page,
            xi.wrapping_neg(),
            yi.wrapping_neg(),
            CombinationOperator::Replace,
        );
    }
    Ok(window)
}

/// Read a reference/target pixel at signed coordinates, zero outside the bitmap.
#[inline]
fn pget(bm: &MonoBitmap, x: i64, y: i64) -> u32 {
    if x < 0 || y < 0 || x >= bm.width() as i64 || y >= bm.height() as i64 {
        0
    } else {
        bm.get(x as u32, y as u32) as u32
    }
}

/// Decode a GRTEMPLATE-0 generic refinement region (back-compat wrapper).
#[allow(clippy::too_many_arguments)]
pub fn decode_refinement_region(
    dec: &mut ArithmeticDecoder<'_>,
    reference: &MonoBitmap,
    width: u32,
    height: u32,
    grdx: i32,
    grdy: i32,
    grat: (i8, i8),
    contexts: &mut [MqContext],
    limits: &DecodeLimits,
) -> Result<MonoBitmap, DecodeError> {
    decode_refinement_region_templated(
        dec,
        reference,
        width,
        height,
        grdx,
        grdy,
        0,
        false,
        grat,
        (-1, -1),
        contexts,
        limits,
    )
}

/// Decode a generic refinement region into a fresh `width × height` bitmap,
/// using `reference` (offset by `(grdx, grdy)`) as the predictor (T.88 §6.3).
/// `grtemplate` selects the 13-bit (0) or 10-bit (1) context; `tpgron` enables
/// typical prediction (§6.3.5.6). `contexts` must be at least
/// [`REFINEMENT_CONTEXT_COUNT`] entries and is *not* reset here.
#[allow(clippy::too_many_arguments)]
pub fn decode_refinement_region_templated(
    dec: &mut ArithmeticDecoder<'_>,
    reference: &MonoBitmap,
    width: u32,
    height: u32,
    grdx: i32,
    grdy: i32,
    grtemplate: u8,
    tpgron: bool,
    grat: (i8, i8),
    grat2: (i8, i8),
    contexts: &mut [MqContext],
    limits: &DecodeLimits,
) -> Result<MonoBitmap, DecodeError> {
    if contexts.len() < REFINEMENT_CONTEXT_COUNT {
        return Err(DecodeError::Overflow {
            operation: "refinement context array too small",
        });
    }
    let mut target = MonoBitmap::new(width, height, false, limits)?;
    if width == 0 || height == 0 {
        return Ok(target);
    }

    if grtemplate == 0 {
        if grat == (-1, -1) && grat2 == (-1, -1) {
            decode_refinement_windowed::<0, false>(
                &mut target,
                dec,
                reference,
                grdx,
                grdy,
                tpgron,
                grat,
                grat2,
                contexts,
            );
        } else {
            decode_refinement_windowed::<0, true>(
                &mut target,
                dec,
                reference,
                grdx,
                grdy,
                tpgron,
                grat,
                grat2,
                contexts,
            );
        }
    } else {
        decode_refinement_windowed::<1, false>(
            &mut target,
            dec,
            reference,
            grdx,
            grdy,
            tpgron,
            grat,
            grat2,
            contexts,
        );
    }

    Ok(target)
}

#[inline(always)]
fn sample_row(row: &[u32], width: u32, x: i64) -> u32 {
    if x < 0 || x >= width as i64 || row.is_empty() {
        return 0;
    }
    let xu = x as usize;
    (row[xu >> 5] >> (31 - (xu & 31) as u32)) & 1
}

/// 32 pixels of `row` starting at signed pixel position `start`, MSB first,
/// with pixels outside `0..width` (or an empty out-of-range row) reading 0.
#[inline(always)]
fn row_window(row: &[u32], start: i64, tail_mask: u32) -> u32 {
    if row.is_empty() {
        return 0;
    }
    let wi = start.div_euclid(32);
    let off = start.rem_euclid(32) as u32;
    let get = |i: i64| -> u32 {
        if i < 0 {
            0
        } else {
            let idx = i as usize;
            match row.get(idx) {
                Some(&w) if idx + 1 == row.len() => w & tail_mask,
                Some(&w) => w,
                None => 0,
            }
        }
    };
    let hi = get(wi);
    if off == 0 {
        hi
    } else {
        (hi << off) | (get(wi + 1) >> (32 - off))
    }
}

/// Sample a custom GRAT1 target pixel at `(x + dx, y + dy)` while row `y` is
/// being decoded directly into `cur` (completed words `..block`) and `pixels`
/// (the low `i` bits of block `block`).
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn target_at_pixel(
    above: &[u32],
    cur: &[u32],
    width: u32,
    stride: usize,
    y: u32,
    x: i64,
    pixels: u32,
    i: u32,
    dx: i64,
    dy: i64,
) -> u32 {
    let xx = x + dx;
    if xx < 0 || xx >= width as i64 || dy > 0 {
        return 0;
    }
    if dy < 0 {
        let yy = y as i64 + dy;
        if yy < 0 {
            0
        } else {
            let row_start = (yy as usize) * stride;
            sample_row(&above[row_start..row_start + stride], width, xx)
        }
    } else if dx >= 0 {
        // Non-causal on the current row: target starts all-zero.
        0
    } else {
        let dist = (-dx) as u32;
        if dist <= i {
            (pixels >> (dist - 1)) & 1
        } else {
            sample_row(cur, width, xx)
        }
    }
}

/// Windowed refinement decoder for GRTEMPLATE `T` (0 or 1), nominal or custom
/// AT pixels, with or without TPGRON.
///
/// Three 3-bit rolling reference windows (`r_prev`, `r_cur`, `r_next`) and one
/// 3-bit target-above window (`t_prev`) are fed from 32-bit block look-ahead
/// registers loaded once per 32-pixel block. Under TPGRON the 3×3 uniformity
/// test (`typical_pixel`) reduces to checking those 9 bits directly.
#[allow(clippy::too_many_arguments)]
fn decode_refinement_windowed<const T: u8, const CUSTOM_AT: bool>(
    target: &mut MonoBitmap,
    decoder: &mut ArithmeticDecoder<'_>,
    reference: &MonoBitmap,
    grdx: i32,
    grdy: i32,
    tpgron: bool,
    grat: (i8, i8),
    grat2: (i8, i8),
    contexts: &mut [MqContext],
) {
    let width = target.width();
    let height = target.height();
    let stride = target.stride_words() as usize;
    let tgt_tail_mask = match width & 31 {
        0 => u32::MAX,
        bits => u32::MAX << (32 - bits),
    };
    let ref_w = reference.width();
    let ref_h = reference.height() as i64;
    let ref_tail_mask = match ref_w & 31 {
        0 => u32::MAX,
        bits => u32::MAX << (32 - bits),
    };
    let grat_x = grat.0 as i64;
    let grat_y = grat.1 as i64;
    let grat2_x = grat2.0 as i64;
    let grat2_y = grat2.1 as i64;
    let sltp_ctx = if T == 0 { SLTP_CTX_GR0 } else { SLTP_CTX_GR1 };

    let ctxs: &mut [MqContext] = &mut contexts[..REFINEMENT_CONTEXT_COUNT];
    let mut dec = decoder.clone();
    let mut ltp = false;
    let rx0 = -(grdx as i64);

    let ref_row_or_empty = |ry_row: i64| -> &[u32] {
        if ry_row < 0 || ry_row >= ref_h {
            &[]
        } else {
            reference.row(ry_row as u32)
        }
    };

    for y in 0..height {
        if tpgron {
            let sltp = dec.decode_bit(&mut ctxs[sltp_ctx]);
            ltp ^= sltp;
        }
        let ry = y as i64 - grdy as i64;
        let row_rp = ref_row_or_empty(ry - 1);
        let row_rc = ref_row_or_empty(ry);
        let row_rn = ref_row_or_empty(ry + 1);

        let (above, cur) = target.rows_above_and_mut(y);
        let prev_tgt: &[u32] = if y >= 1 {
            let start = (y as usize - 1) * stride;
            &above[start..start + stride]
        } else {
            &[]
        };

        // Initial 3-bit windows centred at `x = 0` (`rx = rx0`).
        let (mut r_prev, mut r_cur, mut r_next) = if T == 0 {
            (
                sample_row(row_rp, ref_w, rx0 - 1)
                    | (sample_row(row_rp, ref_w, rx0) << 1)
                    | (sample_row(row_rp, ref_w, rx0 + 1) << 2),
                sample_row(row_rc, ref_w, rx0 - 1)
                    | (sample_row(row_rc, ref_w, rx0) << 1)
                    | (sample_row(row_rc, ref_w, rx0 + 1) << 2),
                sample_row(row_rn, ref_w, rx0 - 1)
                    | (sample_row(row_rn, ref_w, rx0) << 1)
                    | (sample_row(row_rn, ref_w, rx0 + 1) << 2),
            )
        } else {
            (
                (sample_row(row_rp, ref_w, rx0 - 1) << 2)
                    | (sample_row(row_rp, ref_w, rx0) << 1)
                    | sample_row(row_rp, ref_w, rx0 + 1),
                (sample_row(row_rc, ref_w, rx0 - 1) << 2)
                    | (sample_row(row_rc, ref_w, rx0) << 1)
                    | sample_row(row_rc, ref_w, rx0 + 1),
                (sample_row(row_rn, ref_w, rx0 - 1) << 2)
                    | (sample_row(row_rn, ref_w, rx0) << 1)
                    | sample_row(row_rn, ref_w, rx0 + 1),
            )
        };
        let mut t_prev = (sample_row(prev_tgt, width, 0) << 1) | sample_row(prev_tgt, width, 1);
        let mut t_left = 0u32;

        for block in 0..stride {
            let x_block = (block as u32) * 32;
            let count = (width - x_block).min(32);
            let ref_start = x_block as i64 + rx0 + 2;
            let mut win_rp = row_window(row_rp, ref_start, ref_tail_mask);
            let mut win_rc = row_window(row_rc, ref_start, ref_tail_mask);
            let mut win_rn = row_window(row_rn, ref_start, ref_tail_mask);
            let mut win_tp = row_window(prev_tgt, x_block as i64 + 2, tgt_tail_mask);

            let mut pixels = 0u32;
            for i in 0..count {
                let bit = if ltp {
                    let ref9 = r_prev | (r_cur << 3) | (r_next << 6);
                    if ref9 == 0 {
                        0
                    } else if ref9 == 0x1FF {
                        1
                    } else {
                        let cx = if T == 0 {
                            let mut c = r_prev
                                | (r_cur << 3)
                                | (t_left << 6)
                                | (r_next << 7)
                                | (t_prev << 10);
                            if CUSTOM_AT {
                                let x = (x_block + i) as i64;
                                let rx = x + rx0;
                                c = (c & !0x1001)
                                    | pget(reference, rx + grat2_x, ry + grat2_y)
                                    | (target_at_pixel(
                                        above, cur, width, stride, y, x, pixels, i, grat_x, grat_y,
                                    ) << 12);
                            }
                            c
                        } else {
                            t_left
                                | (t_prev << 1)
                                | ((r_next & 3) << 4)
                                | (r_cur << 6)
                                | (((r_prev >> 1) & 1) << 9)
                        };
                        dec.decode_bit(&mut ctxs[cx as usize & 0x1FFF]) as u32
                    }
                } else {
                    let cx = if T == 0 {
                        let mut c =
                            r_prev | (r_cur << 3) | (t_left << 6) | (r_next << 7) | (t_prev << 10);
                        if CUSTOM_AT {
                            let x = (x_block + i) as i64;
                            let rx = x + rx0;
                            c = (c & !0x1001)
                                | pget(reference, rx + grat2_x, ry + grat2_y)
                                | (target_at_pixel(
                                    above, cur, width, stride, y, x, pixels, i, grat_x, grat_y,
                                ) << 12);
                        }
                        c
                    } else {
                        t_left
                            | (t_prev << 1)
                            | ((r_next & 3) << 4)
                            | (r_cur << 6)
                            | (((r_prev >> 1) & 1) << 9)
                    };
                    dec.decode_bit(&mut ctxs[cx as usize & 0x1FFF]) as u32
                };

                pixels = (pixels << 1) | bit;
                t_left = bit;
                if T == 0 {
                    r_prev = (r_prev >> 1) | ((win_rp >> 31) << 2);
                    r_cur = (r_cur >> 1) | ((win_rc >> 31) << 2);
                    r_next = (r_next >> 1) | ((win_rn >> 31) << 2);
                } else {
                    r_prev = ((r_prev << 1) & 7) | (win_rp >> 31);
                    r_cur = ((r_cur << 1) & 7) | (win_rc >> 31);
                    r_next = ((r_next << 1) & 7) | (win_rn >> 31);
                }
                t_prev = ((t_prev << 1) & 7) | (win_tp >> 31);
                win_rp <<= 1;
                win_rc <<= 1;
                win_rn <<= 1;
                win_tp <<= 1;
            }
            cur[block] = pixels << (32 - count);
        }
    }
    *decoder = dec;
}

/// GRTEMPLATE-0 context (T.88 Figure 12), kept for unit tests and equivalence
/// reference.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
#[inline]
fn context_gr0(
    target: &MonoBitmap,
    reference: &MonoBitmap,
    x: i64,
    y: i64,
    rx: i64,
    ry: i64,
    grat_x: i64,
    grat_y: i64,
    grat2_x: i64,
    grat2_y: i64,
) -> usize {
    let mut cx: usize = 0;
    cx |= pget(reference, rx + grat2_x, ry + grat2_y) as usize;
    cx |= (pget(reference, rx, ry - 1) as usize) << 1;
    cx |= (pget(reference, rx + 1, ry - 1) as usize) << 2;
    cx |= (pget(reference, rx - 1, ry) as usize) << 3;
    cx |= (pget(reference, rx, ry) as usize) << 4;
    cx |= (pget(reference, rx + 1, ry) as usize) << 5;
    cx |= (pget(target, x - 1, y) as usize) << 6;
    cx |= (pget(reference, rx - 1, ry + 1) as usize) << 7;
    cx |= (pget(reference, rx, ry + 1) as usize) << 8;
    cx |= (pget(reference, rx + 1, ry + 1) as usize) << 9;
    cx |= (pget(target, x + 1, y - 1) as usize) << 10;
    cx |= (pget(target, x, y - 1) as usize) << 11;
    cx |= (pget(target, x + grat_x, y + grat_y) as usize) << 12;
    cx
}

/// GRTEMPLATE-1 context (T.88 Figure 13), kept for equivalence reference.
#[cfg(test)]
#[inline]
fn context_gr1(
    target: &MonoBitmap,
    reference: &MonoBitmap,
    x: i64,
    y: i64,
    rx: i64,
    ry: i64,
) -> usize {
    let mut cx: usize = 0;
    cx |= pget(target, x - 1, y) as usize;
    cx |= (pget(target, x + 1, y - 1) as usize) << 1;
    cx |= (pget(target, x, y - 1) as usize) << 2;
    cx |= (pget(target, x - 1, y - 1) as usize) << 3;
    cx |= (pget(reference, rx + 1, ry + 1) as usize) << 4;
    cx |= (pget(reference, rx, ry + 1) as usize) << 5;
    cx |= (pget(reference, rx + 1, ry) as usize) << 6;
    cx |= (pget(reference, rx, ry) as usize) << 7;
    cx |= (pget(reference, rx - 1, ry) as usize) << 8;
    cx |= (pget(reference, rx, ry - 1) as usize) << 9;
    cx
}

#[cfg(test)]
#[inline]
fn typical_pixel(reference: &MonoBitmap, rx: i64, ry: i64) -> Option<bool> {
    let mut sum = 0u32;
    for dy in -1..=1 {
        for dx in -1..=1 {
            sum += pget(reference, rx + dx, ry + dy);
        }
    }
    match sum {
        0 => Some(false),
        9 => Some(true),
        _ => None,
    }
}

/// Pre-optimisation per-pixel refinement decoder, kept under `#[cfg(test)]` as
/// the bit-exactness oracle for the windowed decoder.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn decode_refinement_reference(
    dec: &mut ArithmeticDecoder<'_>,
    reference: &MonoBitmap,
    width: u32,
    height: u32,
    grdx: i32,
    grdy: i32,
    grtemplate: u8,
    tpgron: bool,
    grat: (i8, i8),
    grat2: (i8, i8),
    contexts: &mut [MqContext],
    limits: &DecodeLimits,
) -> Result<MonoBitmap, DecodeError> {
    let mut target = MonoBitmap::new(width, height, false, limits)?;
    if width == 0 || height == 0 {
        return Ok(target);
    }
    let grat_x = grat.0 as i64;
    let grat_y = grat.1 as i64;
    let grat2_x = grat2.0 as i64;
    let grat2_y = grat2.1 as i64;
    let sltp_ctx = if grtemplate == 0 {
        SLTP_CTX_GR0
    } else {
        SLTP_CTX_GR1
    };
    let mut ltp = false;
    for y in 0..height as i64 {
        if tpgron {
            let sltp = dec.decode_bit(&mut contexts[sltp_ctx]);
            ltp ^= sltp;
        }
        for x in 0..width as i64 {
            let rx = x - grdx as i64;
            let ry = y - grdy as i64;
            #[allow(clippy::collapsible_if)]
            if ltp {
                if let Some(v) = typical_pixel(reference, rx, ry) {
                    if v {
                        target.set(x as u32, y as u32, true);
                    }
                    continue;
                }
            }
            let cx = if grtemplate == 0 {
                context_gr0(
                    &target, reference, x, y, rx, ry, grat_x, grat_y, grat2_x, grat2_y,
                )
            } else {
                context_gr1(&target, reference, x, y, rx, ry)
            };
            if dec.decode_bit(&mut contexts[cx]) {
                target.set(x as u32, y as u32, true);
            }
        }
    }
    Ok(target)
}

#[cfg(all(test, feature = "encode"))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::encode::arith::Jbig2ArithCoder;
    use crate::encode::sym::BitImage;

    /// Round-trip: encode the difference between a target and a reference glyph
    /// with the encoder's refinement coder, then decode it back exactly.
    #[test]
    fn refinement_roundtrip() {
        let limits = DecodeLimits::default();
        // Reference glyph.
        let mut refimg = BitImage::new(12, 16).unwrap();
        for y in 0..16 {
            for x in 0..12 {
                if x == 0 || x == 11 || y == 0 || y == 15 {
                    refimg.set(x, y, true);
                }
            }
        }
        // Target: reference with a few pixels toggled (same size, GRDX=GRDY=0).
        let mut target = refimg.clone();
        target.set(5, 5, true);
        target.set(6, 8, true);
        target.set(2, 10, false);

        let mut enc = Jbig2ArithCoder::new();
        enc.encode_refinement_region(&target, &refimg, 0, 0, 0, &[(-1, -1)])
            .unwrap();
        let data = enc.into_vec();

        let ref_mono = MonoBitmap::from_bit_image(&refimg, &limits).unwrap();
        let mut dec = ArithmeticDecoder::new(&data);
        let mut ctx = vec![MqContext::default(); REFINEMENT_CONTEXT_COUNT];
        let out = decode_refinement_region(
            &mut dec,
            &ref_mono,
            12,
            16,
            0,
            0,
            (-1, -1),
            &mut ctx,
            &limits,
        )
        .unwrap();

        for y in 0..16u32 {
            for x in 0..12u32 {
                assert_eq!(out.get(x, y), target.get(x, y), "pixel ({x},{y})");
            }
        }
    }
}

// Decoder-only regression: the GRTEMPLATE-0 refinement context must read the
// *reference* adaptive pixel at the parsed GRAT2 offset (T.88 Fig. 12, bit 8 in
// pdfium's numbering / bit 0 here), not a fixed (rx-1, ry-1). A custom SDRAT2 /
// SBRAT2 that is dropped desyncs every custom-AT refinement region — the
// pdf.js bitmap-symbol-symbolrefineone-customat fixture blanked because of it.
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod grat2_tests {
    use super::*;

    #[test]
    fn context_gr0_reads_the_custom_grat2_reference_pixel() {
        let limits = DecodeLimits::default();
        let target = MonoBitmap::new(8, 8, false, &limits).unwrap();
        let mut reference = MonoBitmap::new(8, 8, false, &limits).unwrap();
        // A lone black pixel far from the fixed neighbourhood window around
        // (rx±1, ry±1) = (4..=6, 4..=6), so only the GRAT2 tap can see it.
        reference.set(2, 3, true);

        let (x, y, rx, ry) = (5i64, 5i64, 5i64, 5i64);
        // Nominal GRAT1 target pixel, so bit 12 stays 0.
        let (grat_x, grat_y) = (-1i64, -1i64);

        // Nominal GRAT2 = (-1,-1) reads reference(4,4) = white → context 0.
        let nominal = context_gr0(&target, &reference, x, y, rx, ry, grat_x, grat_y, -1, -1);
        assert_eq!(nominal, 0, "nominal GRAT2 must not see the (2,3) pixel");

        // Custom GRAT2 pointing at (2,3): rx+dx=2, ry+dy=3 → dx=-3, dy=-2.
        let custom = context_gr0(&target, &reference, x, y, rx, ry, grat_x, grat_y, -3, -2);
        assert_eq!(
            custom, 1,
            "custom GRAT2 must read the reference pixel into bit 0"
        );
    }

    #[test]
    fn windowed_refinement_matches_reference() {
        let limits = DecodeLimits::default();
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || -> u32 {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (seed >> 33) as u32
        };

        let widths = [1u32, 2, 7, 15, 31, 32, 33, 64, 65, 97];
        for case in 0..160usize {
            let grtemplate = (case % 2) as u8;
            let tpgron = (case / 2) % 2 == 1;
            let w = widths[case % widths.len()];
            let h = 1 + (next() % 24);
            let rw = widths[(case + 3) % widths.len()];
            let rh = 1 + (next() % 24);
            let grdx = (next() % 11) as i32 - 5;
            let grdy = (next() % 11) as i32 - 5;
            let (grat, grat2) = if grtemplate == 0 && (case / 4) % 2 == 1 {
                // Exercise custom GRAT1 (including current-row and earlier-row)
                // and custom GRAT2.
                let g1y = -((next() % 4) as i8);
                let g1x = if g1y == 0 {
                    -1 - (next() % 40) as i8
                } else {
                    (next() % 11) as i8 - 5
                };
                let g2x = (next() % 9) as i8 - 4;
                let g2y = (next() % 9) as i8 - 4;
                ((g1x, g1y), (g2x, g2y))
            } else {
                ((-1, -1), (-1, -1))
            };

            // Build reference bitmap with uniform regions (to trigger TPGRON
            // typical_pixel both 0 and 9) mixed with sparse noise.
            let mut reference = MonoBitmap::new(rw, rh, (case / 8) % 2 == 1, &limits).unwrap();
            for y in 0..rh {
                for x in 0..rw {
                    if next() % 5 == 0 {
                        reference.set(x, y, next() % 2 == 0);
                    }
                }
            }

            let data_len = 32 + (next() % 256) as usize;
            let data: Vec<u8> = (0..data_len)
                .map(|_| if next() % 7 == 0 { 0xFF } else { next() as u8 })
                .collect();

            let mut d_new = ArithmeticDecoder::new(&data);
            let mut d_ref = ArithmeticDecoder::new(&data);
            let mut c_new = vec![MqContext::default(); REFINEMENT_CONTEXT_COUNT];
            let mut c_ref = vec![MqContext::default(); REFINEMENT_CONTEXT_COUNT];

            let bm_new = decode_refinement_region_templated(
                &mut d_new, &reference, w, h, grdx, grdy, grtemplate, tpgron, grat, grat2,
                &mut c_new, &limits,
            )
            .unwrap();
            let bm_ref = decode_refinement_reference(
                &mut d_ref, &reference, w, h, grdx, grdy, grtemplate, tpgron, grat, grat2,
                &mut c_ref, &limits,
            )
            .unwrap();

            assert_eq!(
                bm_new, bm_ref,
                "case {case} t={grtemplate} tpgron={tpgron} {w}x{h} ref={rw}x{rh} off=({grdx},{grdy}) at={grat:?}/{grat2:?}"
            );
            assert_eq!(c_new, c_ref, "case {case} contexts differ");
        }
    }
}
