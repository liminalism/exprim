// src/iw44/masking.rs

use crate::encode::iw44::transform::Encode;
use crate::image::image_formats::Bitmap;

/// Convert Bitmap mask to i8 mask buffer.
///
/// The image data in the coefficient buffer is stored bottom-up (see
/// `Encode::from_u8_image_with_stride`, which flips rows to match the C++
/// GPixmap convention), so the mask must be flipped the same way or the
/// don't-care regions land on the wrong rows.
pub fn image_to_mask8(mask_img: &Bitmap, bw: usize, ih: usize) -> Vec<i8> {
    let mut mask8 = vec![0i8; bw * ih];
    let mh = mask_img.height() as usize;
    for y in 0..ih {
        let src_y = ih - 1 - y; // match the bottom-up flip applied to image data
        if src_y >= mh {
            continue;
        }
        for x in 0..(mask_img.width() as usize).min(bw) {
            // Non-zero mask pixels indicate masked-out regions
            let mask_val = mask_img.get_pixel(x as u32, src_y as u32).y;
            mask8[y * bw + x] = if mask_val > 0 { 1 } else { 0 };
        }
    }
    mask8
}

/// Performs the "interpolate_mask" step from IW44: fill in masked-out
/// pixels by averaging neighbors across scales, so that later wavelet
/// decompositions don't waste bits on irrelevant regions.
///
/// Port of `interpolate_mask(short*,int,int,int,const signed char*,int)`
/// from IW44EncodeCodec.cpp. Updated to work with i32 data for consistency
/// with transform.rs and eliminate unnecessary type conversions.
pub fn interpolate_mask(
    data: &mut [i16],
    w: usize,
    h: usize,
    rowsize: usize,
    mask: &[i8],
    mskrowsize: usize,
) {
    // 1) build a count buffer: non-masked => high weight, masked => zero
    let mut count = vec![0i32; w * h];
    for y in 0..h {
        for x in 0..w {
            let m = mask[y * mskrowsize + x];
            count[y * w + x] = if m != 0 { 0 } else { 0x1000 };
        }
    }
    // 2) copy original data into a scratch (convert i16 to i32 for intermediate calculations)
    let mut scratch = vec![0i32; w * h];
    for y in 0..h {
        for x in 0..w {
            scratch[y * w + x] = data[y * rowsize + x] as i32;
        }
    }
    // 3) iterate over scales
    let mut state = MaskInterpolation {
        data,
        count,
        scratch,
        w,
        h,
        rowsize,
    };
    let mut split = 1;
    let mut scale = 2;
    let mut again = true;
    while again && scale < w && scale < h {
        again = false;
        for i in (0..h).step_by(scale) {
            for j in (0..w).step_by(scale) {
                again |= state.average_cell(i, j, split, scale);
            }
        }
        split = scale;
        scale <<= 1;
    }
}

/// Working state of `interpolate_mask`: the image, per-pixel weights (zero
/// where nothing is known yet) and the averaged values of the previous scale.
struct MaskInterpolation<'a> {
    data: &'a mut [i16],
    count: Vec<i32>,
    scratch: Vec<i32>,
    w: usize,
    h: usize,
    rowsize: usize,
}

impl MaskInterpolation<'_> {
    /// Averages the known samples of the `scale`-sized cell at (`i`, `j`),
    /// fills its masked pixels with that average and stores it for the next
    /// scale. Returns `true` when the cell holds no information yet, so a
    /// coarser scale must try again.
    fn average_cell(&mut self, i: usize, j: usize, split: usize, scale: usize) -> bool {
        let (w, h) = (self.w, self.h);
        // compute weighted average over the square [i..i+scale)×[j..j+scale)
        let istart = if i + split > h {
            i.saturating_sub(scale)
        } else {
            i
        };
        let jstart = if j + split > w {
            j.saturating_sub(scale)
        } else {
            j
        };
        let mut gray_sum = 0i32;
        let mut total_w = 0i32;
        let mut saw_zero = false;
        let iend = (i + scale).min(h);
        let jend = (j + scale).min(w);
        let mut ii = istart;
        while ii < iend {
            let mut jj = jstart;
            while jj < jend {
                let wght = self.count[ii * w + jj];
                if wght > 0 {
                    total_w += wght;
                    gray_sum += wght * self.scratch[ii * w + jj];
                } else if ii >= i && jj >= j {
                    saw_zero = true;
                }
                jj += split;
            }
            ii += split;
        }
        let idx = i * w + j;
        if total_w == 0 {
            // still no information here; we'll try again at a coarser scale
            self.count[idx] = 0;
            return true;
        }
        // fill masked pixels if we saw them
        let gray = (gray_sum / total_w) as i32;
        if saw_zero {
            self.fill_unknown(i..iend, j..jend, gray);
        }
        // store for next iteration
        self.count[idx] = (total_w >> 2) as i32;
        self.scratch[idx] = gray;
        false
    }

    /// Writes `gray` into every pixel of the given rows and columns that has
    /// no information yet, and marks it known.
    fn fill_unknown(
        &mut self,
        rows: std::ops::Range<usize>,
        cols: std::ops::Range<usize>,
        gray: i32,
    ) {
        for yy in rows {
            for xx in cols.clone() {
                let cidx = yy * self.w + xx;
                if self.count[cidx] == 0 {
                    self.data[yy * self.rowsize + xx] = gray as i16;
                    self.count[cidx] = 1;
                }
            }
        }
    }
}

/// Performs the "forward_mask" multiscale masked wavelet decomposition
/// from IW44EncodeCodec.cpp. Updated to work with i32 data for consistency
/// with transform.rs and eliminate unnecessary type conversions.
/// At each scale it zeroes out wavelet coefficients under the mask,
/// then reconstructs and re-decomposes to freeze those regions.
pub fn forward_mask(
    data: &mut [i16],
    w: usize,
    h: usize,
    rowsize: usize,
    begin: usize,
    end: usize,
    mask: &[i8],
    mskrowsize: usize,
) {
    // 1) copy mask into an aligned 1-per-pixel array
    let mut smask = vec![0i8; w * h];
    for y in 0..h {
        for x in 0..w {
            smask[y * w + x] = mask[y * mskrowsize + x];
        }
    }
    // 2) scratch buffer for single-level decomposition (now i32)
    let mut scratch = vec![0i32; w * h];
    let mut scratch_i16 = vec![0i16; w * h];

    let mut scale = begin.next_power_of_two();
    while scale < end {
        // copy every scale-th sample into scratch (convert i16 to i32)
        for y in (0..h).step_by(scale) {
            for x in (0..w).step_by(scale) {
                scratch[y * w + x] = data[y * rowsize + x] as i32;
            }
        }
        // full-band forward transform - use new API
        let levels = ((scale * 2).trailing_zeros() as usize)
            .saturating_sub(1)
            .min(5);
        for i in 0..scratch.len() {
            scratch_i16[i] = scratch[i] as i16;
        }
        Encode::forward(&mut scratch_i16, w, h, w, levels);
        for i in 0..scratch.len() {
            scratch[i] = scratch_i16[i] as i32;
        }

        // zero out masked detail coefficients
        for y in (0..h).step_by(scale * 2) {
            zero_masked_details(&mut scratch, &smask, y, w, h, scale);
        }

        // reconstruct back to pixel domain (inverse IW44 transform would be called here in reference impl, but is not implemented in encoder)
        // restore visible pixels so they remain exact (convert i16 to i32)
        for y in (0..h).step_by(scale) {
            restore_visible_row(&mut scratch, &smask, data, y * w, y * rowsize, w, scale);
        }

        // re-decompose to freeze the mask out
        for i in 0..scratch.len() {
            scratch_i16[i] = scratch[i] as i16;
        }
        Encode::forward(&mut scratch_i16, w, h, w, levels);
        for i in 0..scratch.len() {
            scratch[i] = scratch_i16[i] as i32;
        }

        // copy the frozen coefficients back into data (convert i32 to i16)
        for y in (0..h).step_by(scale) {
            for x in (0..w).step_by(scale) {
                data[y * rowsize + x] = scratch[y * w + x] as i16;
            }
        }

        // update the mask for the next coarser scale
        for y in (0..h).step_by(scale * 2) {
            coarsen_mask_row(&mut smask, y, w, h, scale);
        }

        scale <<= 1;
    }
}

/// Zeroes the masked detail coefficients of the decomposition row pair
/// starting at row `y`: the horizontal band on row `y` and the vertical band
/// on row `y + scale`.
fn zero_masked_details(
    scratch: &mut [i32],
    smask: &[i8],
    y: usize,
    w: usize,
    h: usize,
    scale: usize,
) {
    // horizontal band
    for x in (scale..w).step_by(scale * 2) {
        if smask[y * w + x] != 0 {
            scratch[y * w + x] = 0;
        }
    }
    // vertical band
    if y + scale >= h {
        return;
    }
    for x in (0..w).step_by(scale) {
        if smask[(y + scale) * w + x] != 0 {
            scratch[(y + scale) * w + x] = 0;
        }
    }
}

/// Restores the exact pixel values of the visible (unmasked) samples of one
/// row; `mask_row` and `data_row` are the row's offsets in `smask`/`scratch`
/// and in `data`.
fn restore_visible_row(
    scratch: &mut [i32],
    smask: &[i8],
    data: &[i16],
    mask_row: usize,
    data_row: usize,
    w: usize,
    scale: usize,
) {
    for x in (0..w).step_by(scale) {
        if smask[mask_row + x] == 0 {
            scratch[mask_row + x] = data[data_row + x] as i32;
        }
    }
}

/// Recomputes the mask of one row for the next coarser scale: a sample stays
/// masked only when it, the sample below and both horizontal neighbours are.
fn coarsen_mask_row(smask: &mut [i8], y: usize, w: usize, h: usize, scale: usize) {
    for x in (0..w).step_by(scale * 2) {
        let m00 = smask[y * w + x] != 0;
        let m10 = y + scale < h && smask[(y + scale) * w + x] != 0;
        let left = x >= scale && smask[y * w + x - scale] != 0;
        let right = x + scale < w && smask[y * w + x + scale] != 0;
        smask[y * w + x] = if m00 && m10 && left && right { 1 } else { 0 };
    }
}
