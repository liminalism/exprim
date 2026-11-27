//! The decoder-native packed one-bit bitmap (jbig2decplan.md §7, §23).
//!
//! Rows are packed into `u32` words, one row starting on each word boundary.
//! Conventions (matching the encoder's `BitImage`):
//!
//! * The most-significant bit of a word is the leftmost pixel.
//! * A set bit means black.
//! * Padding bits past the row width are always zero.
//!
//! This layout keeps composition (`combine`) word-wise and leaves the door
//! open to SIMD later without requiring it now.

use crate::shared::error::{DecodeError, LimitError};
use crate::shared::limits::DecodeLimits;

/// How a source bitmap is combined onto a destination (T.88 external and
/// region combination operators).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CombinationOperator {
    Or,
    And,
    Xor,
    Xnor,
    Replace,
}

/// Which bit value means black in a caller's bilevel buffer.
///
/// JBIG2 (like PBM and Leptonica) stores 1 = black. TIFF `MinIsBlack`, most
/// 1bpp DIBs with a black-first palette, and every grayscale format store
/// 0 = black. Conflating the two produces a valid but fully inverted image,
/// so every API that crosses a caller's buffer names its polarity.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Polarity {
    /// A set bit (or non-zero byte) is black: JBIG2's own convention.
    #[default]
    OneIsBlack,
    /// A clear bit (or zero byte) is black.
    ZeroIsBlack,
}

/// A caller-supplied buffer cannot hold the rows asked of it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum BufferLayoutError {
    #[error("stride {stride} is shorter than a row ({row_bytes} bytes)")]
    StrideTooShort { stride: usize, row_bytes: usize },
    #[error("buffer has {len} bytes but needs {required}")]
    BufferTooSmall { len: usize, required: usize },
    #[error("buffer size overflows the address space")]
    Overflow,
}

/// A packed one-bit-per-pixel bitmap with word-aligned rows.
#[derive(Clone, Default, PartialEq, Eq)]
pub struct MonoBitmap {
    width: u32,
    height: u32,
    stride_words: u32,
    words: Vec<u32>,
}

impl core::fmt::Debug for MonoBitmap {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MonoBitmap")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("stride_words", &self.stride_words)
            .finish()
    }
}

#[inline]
fn stride_words_for(width: u32) -> u32 {
    // ceil(width / 32); width <= max_width (checked in `new`) so no overflow.
    width.div_ceil(32)
}

impl MonoBitmap {
    /// Allocate a `width` × `height` bitmap, all white (or all black if
    /// `fill_black`), after checking the dimensions against `limits`.
    ///
    /// Returns a typed error rather than panicking or attempting an oversized
    /// allocation on malformed input.
    pub fn new(
        width: u32,
        height: u32,
        fill_black: bool,
        limits: &DecodeLimits,
    ) -> Result<Self, DecodeError> {
        let mut bm = Self {
            width: 0,
            height: 0,
            stride_words: 0,
            words: Vec::new(),
        };
        bm.reset(width, height, fill_black, limits)?;
        Ok(bm)
    }

    /// Reset this bitmap to a fresh `width` × `height` all-white (or all-black
    /// if `fill_black`) bitmap, **reusing** the backing allocation when its
    /// capacity already suffices. Recycles a pooled page/region bitmap for the
    /// zero-alloc `decode_embedded_into` path. Dimensions are checked against
    /// `limits` exactly as [`MonoBitmap::new`].
    pub fn reset(
        &mut self,
        width: u32,
        height: u32,
        fill_black: bool,
        limits: &DecodeLimits,
    ) -> Result<(), DecodeError> {
        if width > limits.max_width {
            return Err(DecodeError::limit(LimitError::Dimension {
                dimension: "width",
                value: width as u64,
                limit: limits.max_width as u64,
            }));
        }
        if height > limits.max_height {
            return Err(DecodeError::limit(LimitError::Dimension {
                dimension: "height",
                value: height as u64,
                limit: limits.max_height as u64,
            }));
        }
        let pixels = (width as u64)
            .checked_mul(height as u64)
            .ok_or(DecodeError::Overflow {
                operation: "bitmap pixel count",
            })?;
        if pixels > limits.max_region_pixels {
            return Err(DecodeError::limit(LimitError::Pixels {
                what: "bitmap",
                value: pixels,
                limit: limits.max_region_pixels,
            }));
        }

        let stride_words = stride_words_for(width);
        let total_words =
            (stride_words as u64)
                .checked_mul(height as u64)
                .ok_or(DecodeError::Overflow {
                    operation: "bitmap word count",
                })?;
        let total_words = usize::try_from(total_words).map_err(|_| DecodeError::Overflow {
            operation: "bitmap word count to usize",
        })?;

        self.words.clear();
        self.words.resize(total_words, 0);
        if fill_black && width > 0 {
            fill_black_rows(&mut self.words, width, stride_words);
        }
        self.width = width;
        self.height = height;
        self.stride_words = stride_words;
        Ok(())
    }

    /// Grow the bitmap to at least `new_height` rows, preserving existing
    /// content; new rows are white. A no-op if already tall enough. Used for
    /// striped pages of unknown height (T.88 §7.4.8.5), so growth is bounded by
    /// `max_page_pixels` rather than `max_region_pixels`.
    pub fn grow_to_height(
        &mut self,
        new_height: u32,
        max_pixels: u64,
        limits: &DecodeLimits,
    ) -> Result<(), DecodeError> {
        if new_height <= self.height {
            return Ok(());
        }
        if new_height > limits.max_height {
            return Err(DecodeError::limit(LimitError::Dimension {
                dimension: "height",
                value: new_height as u64,
                limit: limits.max_height as u64,
            }));
        }
        let pixels =
            (self.width as u64)
                .checked_mul(new_height as u64)
                .ok_or(DecodeError::Overflow {
                    operation: "striped page pixel count",
                })?;
        if pixels > max_pixels {
            return Err(DecodeError::limit(LimitError::Pixels {
                what: "striped page",
                value: pixels,
                limit: max_pixels,
            }));
        }
        let total_words = (self.stride_words as u64)
            .checked_mul(new_height as u64)
            .and_then(|w| usize::try_from(w).ok())
            .ok_or(DecodeError::Overflow {
                operation: "striped page word count",
            })?;
        self.words.resize(total_words, 0);
        self.height = new_height;
        Ok(())
    }

    /// Overwrite `self` with the contents of `src`, **reusing** `self`'s backing
    /// allocation when its capacity already suffices (the zero-alloc steady state
    /// for `decode_embedded_into`). Only reallocates when `src` is larger than
    /// `self`'s current capacity.
    pub fn assign_from(&mut self, src: &MonoBitmap) {
        self.width = src.width;
        self.height = src.height;
        self.stride_words = src.stride_words;
        self.words.clear();
        self.words.extend_from_slice(&src.words);
    }

    #[inline]
    pub fn width(&self) -> u32 {
        self.width
    }

    #[inline]
    pub fn height(&self) -> u32 {
        self.height
    }

    #[inline]
    pub fn stride_words(&self) -> u32 {
        self.stride_words
    }

    /// Bytes occupied by the packed pixel backing store.
    #[inline]
    pub fn storage_bytes(&self) -> usize {
        self.words.len().saturating_mul(core::mem::size_of::<u32>())
    }

    /// The packed words of row `y`. Panics only on an out-of-range `y`, which
    /// is a caller bug (decoders validate `y` against `height` first).
    #[inline]
    pub fn row(&self, y: u32) -> &[u32] {
        let start = (y as usize) * (self.stride_words as usize);
        &self.words[start..start + self.stride_words as usize]
    }

    /// The packed words of row `y`, mutably.
    #[inline]
    pub fn row_mut(&mut self, y: u32) -> &mut [u32] {
        let start = (y as usize) * (self.stride_words as usize);
        &mut self.words[start..start + self.stride_words as usize]
    }

    /// Borrow all previously decoded rows `0..y` (flat packed words, stride
    /// `stride_words`) alongside row `y` mutably, so decoders can write
    /// directly into the bitmap with no intermediate scratch copy.
    #[cfg(feature = "decode")]
    #[inline]
    pub(crate) fn rows_above_and_mut(&mut self, y: u32) -> (&[u32], &mut [u32]) {
        let stride = self.stride_words as usize;
        let split = (y as usize) * stride;
        let (above, rest) = self.words.split_at_mut(split);
        (above, &mut rest[..stride])
    }

    /// Copy row `src_y` directly into row `dst_y` in one `copy_within` (used
    /// for TPGDON duplicate rows).
    #[cfg(feature = "decode")]
    #[inline]
    pub(crate) fn copy_row(&mut self, src_y: u32, dst_y: u32) {
        let stride = self.stride_words as usize;
        let src = (src_y as usize) * stride;
        let dst = (dst_y as usize) * stride;
        self.words.copy_within(src..src + stride, dst);
    }

    /// Zero every word of row `y`.
    #[cfg(feature = "decode")]
    #[inline]
    pub(crate) fn clear_row(&mut self, y: u32) {
        self.row_mut(y).fill(0);
    }

    /// Read pixel `(x, y)`. Out-of-range coordinates read as white (`false`)
    /// so decoded coordinates can never trigger a panic.
    #[inline]
    pub fn get(&self, x: u32, y: u32) -> bool {
        if x >= self.width || y >= self.height {
            return false;
        }
        let word = self.row(y)[(x / 32) as usize];
        (word >> (31 - (x % 32))) & 1 != 0
    }

    /// Set pixel `(x, y)`. Out-of-range coordinates are ignored (no panic).
    #[inline]
    pub fn set(&mut self, x: u32, y: u32, black: bool) {
        if x >= self.width || y >= self.height {
            return;
        }
        let bit = 1u32 << (31 - (x % 32));
        let word = &mut self.row_mut(y)[(x / 32) as usize];
        if black {
            *word |= bit;
        } else {
            *word &= !bit;
        }
    }

    /// Composite `source` onto `self` with its top-left pixel at destination
    /// coordinate `(x, y)` (signed — negative or overhanging placement is
    /// clipped). Whole interior words are combined at once; only the first and
    /// last partial words of each row are masked.
    pub fn combine(&mut self, source: &MonoBitmap, x: i32, y: i32, operator: CombinationOperator) {
        if source.width == 0 || source.height == 0 || self.width == 0 || self.height == 0 {
            return;
        }

        // Clip the source column range [sx_start, sx_end) so the destination
        // column dx = x + sx stays within [0, width).
        // Negate in i64 so `x == i32::MIN` (reachable from a mutated region
        // coordinate) cannot overflow.
        let sx_start = if x < 0 { -(x as i64) } else { 0 };
        let sx_end = (source.width as i64).min(self.width as i64 - x as i64);
        if sx_start >= sx_end {
            return;
        }
        // Clip the source row range [sy_start, sy_end) once up front.
        let sy_start = if y < 0 { -(y as i64) } else { 0 };
        let sy_end = (source.height as i64).min(self.height as i64 - y as i64);
        if sy_start >= sy_end {
            return;
        }

        let dx_start = (x as i64 + sx_start) as usize;
        let dx_end = (x as i64 + sx_end) as usize;
        let dy_start = (y as i64 + sy_start) as usize;
        let sy_start_u = sy_start as usize;
        let sy_end_u = sy_end as usize;
        let dst_stride = self.stride_words as usize;
        let src_stride = source.stride_words as usize;

        // Fast path for single-word, horizontally unclipped sources (covers
        // nearly every text-region glyph and halftone pattern). `MonoBitmap`
        // keeps padding bits past `source.width` zero, so OR and XOR need no
        // masking at all.
        if src_stride == 1 && sx_start == 0 && sx_end == source.width as i64 {
            let w_idx = dx_start >> 5;
            let shift = (dx_start & 31) as u32;
            let spans_two = shift + source.width > 32;
            let src_slice = &source.words[sy_start_u..sy_end_u];
            let mut dst_pos = dy_start * dst_stride + w_idx;
            match operator {
                CombinationOperator::Or => {
                    if !spans_two {
                        for &sw in src_slice {
                            self.words[dst_pos] |= sw >> shift;
                            dst_pos += dst_stride;
                        }
                    } else {
                        let rshift = 32 - shift;
                        for &sw in src_slice {
                            self.words[dst_pos] |= sw >> shift;
                            self.words[dst_pos + 1] |= sw << rshift;
                            dst_pos += dst_stride;
                        }
                    }
                    return;
                }
                CombinationOperator::Xor => {
                    if !spans_two {
                        for &sw in src_slice {
                            self.words[dst_pos] ^= sw >> shift;
                            dst_pos += dst_stride;
                        }
                    } else {
                        let rshift = 32 - shift;
                        for &sw in src_slice {
                            self.words[dst_pos] ^= sw >> shift;
                            self.words[dst_pos + 1] ^= sw << rshift;
                            dst_pos += dst_stride;
                        }
                    }
                    return;
                }
                CombinationOperator::Replace => {
                    if !spans_two {
                        let mask = mask_range(shift, shift + source.width);
                        let inv = !mask;
                        for &sw in src_slice {
                            self.words[dst_pos] = (self.words[dst_pos] & inv) | (sw >> shift);
                            dst_pos += dst_stride;
                        }
                    } else {
                        let rshift = 32 - shift;
                        let mask0 = mask_range(shift, 32);
                        let inv0 = !mask0;
                        let mask1 = mask_range(0, shift + source.width - 32);
                        let inv1 = !mask1;
                        for &sw in src_slice {
                            self.words[dst_pos] = (self.words[dst_pos] & inv0) | (sw >> shift);
                            self.words[dst_pos + 1] =
                                (self.words[dst_pos + 1] & inv1) | (sw << rshift);
                            dst_pos += dst_stride;
                        }
                    }
                    return;
                }
                CombinationOperator::And => {
                    if !spans_two {
                        let inv = !mask_range(shift, shift + source.width);
                        for &sw in src_slice {
                            self.words[dst_pos] &= (sw >> shift) | inv;
                            dst_pos += dst_stride;
                        }
                    } else {
                        let rshift = 32 - shift;
                        let inv0 = !mask_range(shift, 32);
                        let inv1 = !mask_range(0, shift + source.width - 32);
                        for &sw in src_slice {
                            self.words[dst_pos] &= (sw >> shift) | inv0;
                            self.words[dst_pos + 1] &= (sw << rshift) | inv1;
                            dst_pos += dst_stride;
                        }
                    }
                    return;
                }
                CombinationOperator::Xnor => {
                    if !spans_two {
                        let mask = mask_range(shift, shift + source.width);
                        let inv = !mask;
                        for &sw in src_slice {
                            let d = self.words[dst_pos];
                            self.words[dst_pos] = (d & inv) | (!(d ^ (sw >> shift)) & mask);
                            dst_pos += dst_stride;
                        }
                    } else {
                        let rshift = 32 - shift;
                        let mask0 = mask_range(shift, 32);
                        let inv0 = !mask0;
                        let mask1 = mask_range(0, shift + source.width - 32);
                        let inv1 = !mask1;
                        for &sw in src_slice {
                            let d0 = self.words[dst_pos];
                            self.words[dst_pos] = (d0 & inv0) | (!(d0 ^ (sw >> shift)) & mask0);
                            let d1 = self.words[dst_pos + 1];
                            self.words[dst_pos + 1] =
                                (d1 & inv1) | (!(d1 ^ (sw << rshift)) & mask1);
                            dst_pos += dst_stride;
                        }
                    }
                    return;
                }
            }
        }

        // General multi-word / clipped path: precompute column parameters once
        // outside the row loop.
        let w_first = dx_start >> 5;
        let w_last = (dx_end - 1) >> 5;
        let first_lo = (dx_start & 31) as u32;
        let last_hi = (((dx_end - 1) & 31) + 1) as u32;
        let first_mask = if w_first == w_last {
            mask_range(first_lo, last_hi)
        } else {
            mask_range(first_lo, 32)
        };
        let last_mask = mask_range(0, last_hi);
        let a_first = 32 * (w_first as i64) - x as i64;
        let off = a_first.rem_euclid(32) as u32;
        let wi_first = a_first.div_euclid(32);

        let mut dst_row_start = dy_start * dst_stride;
        let mut src_row_start = sy_start_u * src_stride;
        for _ in sy_start_u..sy_end_u {
            let dst_row = &mut self.words[dst_row_start..dst_row_start + dst_stride];
            let src_row = &source.words[src_row_start..src_row_start + src_stride];
            combine_row_fast(
                dst_row, src_row, w_first, w_last, first_mask, last_mask, wi_first, off, operator,
            );
            dst_row_start += dst_stride;
            src_row_start += src_stride;
        }
    }

    /// Bytes per row in the packed MSB-first form: `ceil(width / 8)`.
    #[inline]
    pub fn packed_row_bytes(&self) -> usize {
        (self.width as usize).div_ceil(8)
    }

    /// Copy the bitmap into a caller's buffer as MSB-first packed rows
    /// `dst_stride` bytes apart, in the caller's `polarity`.
    ///
    /// `dst` needs `dst_stride * (height - 1) + packed_row_bytes()` bytes: the
    /// last row need not be padded out to the stride. Bytes between the end of
    /// a row and the next stride are left untouched. Bits past `width` in a
    /// row's last byte are written as white in the requested polarity.
    pub fn copy_packed_rows(
        &self,
        dst: &mut [u8],
        dst_stride: usize,
        polarity: Polarity,
    ) -> Result<(), BufferLayoutError> {
        let row_bytes = self.packed_row_bytes();
        if dst_stride < row_bytes {
            return Err(BufferLayoutError::StrideTooShort {
                stride: dst_stride,
                row_bytes,
            });
        }
        let required = if self.height == 0 {
            0
        } else {
            dst_stride
                .checked_mul(self.height as usize - 1)
                .and_then(|bytes| bytes.checked_add(row_bytes))
                .ok_or(BufferLayoutError::Overflow)?
        };
        if dst.len() < required {
            return Err(BufferLayoutError::BufferTooSmall {
                len: dst.len(),
                required,
            });
        }
        let flip = match polarity {
            Polarity::OneIsBlack => 0u8,
            Polarity::ZeroIsBlack => 0xFF,
        };
        for y in 0..self.height {
            let start = y as usize * dst_stride;
            let out = &mut dst[start..start + row_bytes];
            for (chunk, word) in out.chunks_mut(4).zip(self.row(y)) {
                let bytes = word.to_be_bytes();
                for (dst_byte, &src_byte) in chunk.iter_mut().zip(&bytes) {
                    *dst_byte = src_byte ^ flip;
                }
            }
        }
        Ok(())
    }

    /// The bitmap as one byte per pixel, row-major, 1 = black: the layout the
    /// encoder's byte-per-pixel entry points accept, so a round trip compares
    /// equal.
    pub fn to_bytes_per_pixel(&self) -> Vec<u8> {
        let width = self.width as usize;
        let mut out = Vec::with_capacity(width.saturating_mul(self.height as usize));
        for y in 0..self.height {
            let row = self.row(y);
            for x in 0..width {
                out.push(((row[x >> 5] >> (31 - (x & 31))) & 1) as u8);
            }
        }
        out
    }

    /// A borrowed MSB-first byte view of the bitmap: each row is
    /// `ceil(width / 8)` bytes, most-significant bit = leftmost pixel.
    pub fn as_msb_bytes(&self) -> PackedRows<'_> {
        PackedRows {
            bitmap: self,
            bytes_per_row: (self.width as usize).div_ceil(8),
        }
    }

    // ── Conversions to/from the encoder's BitImage ──────────────────────────

    /// Build a `MonoBitmap` from the encoder's `BitImage`.
    ///
    /// The two types now share a layout — `ceil(width / 32)` words per row,
    /// MSB-first within each word, padding bits zero — so this is a bulk word
    /// copy rather than the per-pixel `get`/`set` loop it used to be.
    #[cfg(feature = "encode")]
    pub fn from_bit_image(
        img: &crate::encode::sym::BitImage,
        limits: &DecodeLimits,
    ) -> Result<Self, DecodeError> {
        let width = u32::try_from(img.width).map_err(|_| DecodeError::Overflow {
            operation: "BitImage width to u32",
        })?;
        let height = u32::try_from(img.height).map_err(|_| DecodeError::Overflow {
            operation: "BitImage height to u32",
        })?;
        let mut out = MonoBitmap::new(width, height, false, limits)?;
        debug_assert_eq!(out.words.len(), img.packed_words().len());
        out.words.copy_from_slice(img.packed_words());
        Ok(out)
    }

    /// Convert to the encoder's `BitImage`. Errors if the dimensions fall
    /// outside what `BitImage` accepts (e.g. zero-sized).
    #[cfg(feature = "encode")]
    pub fn to_bit_image(&self) -> Result<crate::encode::sym::BitImage, DecodeError> {
        let mut img = crate::encode::sym::BitImage::new(self.width, self.height).map_err(|_| {
            DecodeError::Overflow {
                operation: "MonoBitmap to BitImage dimensions",
            }
        })?;
        img.copy_words_from(&self.words)
            .map_err(|_| DecodeError::Overflow {
                operation: "MonoBitmap to BitImage storage",
            })?;
        Ok(img)
    }
}

/// Set every in-width bit of every row to black, leaving row padding zero.
fn fill_black_rows(words: &mut [u32], width: u32, stride_words: u32) {
    if stride_words == 0 {
        return;
    }
    let full_words = (width / 32) as usize;
    let rem = width % 32;
    let last_mask = if rem == 0 { 0 } else { !0u32 << (32 - rem) };
    for row in words.chunks_mut(stride_words as usize) {
        for w in row.iter_mut().take(full_words) {
            *w = !0u32;
        }
        // When width is not a multiple of 32 there is exactly one partial word
        // at index `full_words` (stride_words = full_words + 1), so this index
        // is always in range.
        if rem != 0 {
            row[full_words] = last_mask;
        }
    }
}

/// Combine one source row's clipped span into one destination row using
/// precomputed word range, edge masks, and bit alignment.
#[allow(clippy::too_many_arguments)]
#[inline]
fn combine_row_fast(
    dst_row: &mut [u32],
    src_row: &[u32],
    w_first: usize,
    w_last: usize,
    first_mask: u32,
    last_mask: u32,
    wi_first: i64,
    off: u32,
    operator: CombinationOperator,
) {
    let words = &mut dst_row[w_first..=w_last];
    let count = words.len();
    if off == 0 {
        for (i, dword) in words.iter_mut().enumerate() {
            let src_val = get_word(src_row, wi_first + i as i64);
            let mask = if i == 0 {
                first_mask
            } else if i + 1 == count {
                last_mask
            } else {
                !0u32
            };
            *dword = apply_operator(*dword, src_val, mask, operator);
        }
    } else {
        let rshift = 32 - off;
        let mut hi = get_word(src_row, wi_first);
        for (i, dword) in words.iter_mut().enumerate() {
            let lo = get_word(src_row, wi_first + i as i64 + 1);
            let src_val = (hi << off) | (lo >> rshift);
            hi = lo;
            let mask = if i == 0 {
                first_mask
            } else if i + 1 == count {
                last_mask
            } else {
                !0u32
            };
            *dword = apply_operator(*dword, src_val, mask, operator);
        }
    }
}

/// Build an MSB-first mask with bits set in `[from, to)` (bit positions
/// measured from the MSB, 0..=32).
#[inline]
fn mask_range(from: u32, to: u32) -> u32 {
    debug_assert!(from <= to && to <= 32);
    if from >= to {
        return 0;
    }
    // Bits [from, to) from the MSB. `to == 32` and `from == 0` handled without
    // shift overflow.
    let high = if from == 0 { !0u32 } else { !0u32 >> from };
    let low = if to == 32 { 0 } else { !0u32 >> to };
    high & !low
}

#[inline]
fn get_word(row: &[u32], idx: i64) -> u32 {
    if idx < 0 {
        return 0;
    }
    row.get(idx as usize).copied().unwrap_or(0)
}

/// Combine destination word `d` with source word `s`, affecting only the bits
/// set in `mask`.
#[inline]
fn apply_operator(d: u32, s: u32, mask: u32, op: CombinationOperator) -> u32 {
    match op {
        CombinationOperator::Or => d | (s & mask),
        CombinationOperator::And => d & (s | !mask),
        CombinationOperator::Xor => d ^ (s & mask),
        CombinationOperator::Xnor => (d & !mask) | (!(d ^ s) & mask),
        CombinationOperator::Replace => (d & !mask) | (s & mask),
    }
}

/// A borrowed MSB-first byte view produced by [`MonoBitmap::as_msb_bytes`].
pub struct PackedRows<'a> {
    bitmap: &'a MonoBitmap,
    bytes_per_row: usize,
}

impl PackedRows<'_> {
    /// Bytes per row (`ceil(width / 8)`).
    #[inline]
    pub fn bytes_per_row(&self) -> usize {
        self.bytes_per_row
    }

    #[inline]
    pub fn height(&self) -> u32 {
        self.bitmap.height
    }

    /// The MSB-first bytes of row `y`.
    pub fn row(&self, y: u32) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.bytes_per_row);
        let row = self.bitmap.row(y);
        for word in row {
            out.extend_from_slice(&word.to_be_bytes());
        }
        out.truncate(self.bytes_per_row);
        out
    }

    /// The whole bitmap as row-major MSB-first bytes.
    pub fn to_vec(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.bytes_per_row * self.bitmap.height as usize);
        for y in 0..self.bitmap.height {
            let mut remaining = self.bytes_per_row;
            for word in self.bitmap.row(y) {
                let bytes = word.to_be_bytes();
                let take = remaining.min(bytes.len());
                out.extend_from_slice(&bytes[..take]);
                remaining -= take;
                if remaining == 0 {
                    break;
                }
            }
        }
        out
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn limits() -> DecodeLimits {
        DecodeLimits::default()
    }

    const ODD_WIDTHS: [u32; 14] = [1, 2, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65];

    #[test]
    fn new_zeroes_and_dimensions() {
        for &w in &ODD_WIDTHS {
            let bm = MonoBitmap::new(w, 3, false, &limits()).unwrap();
            assert_eq!(bm.width(), w);
            assert_eq!(bm.height(), 3);
            assert_eq!(bm.stride_words(), w.div_ceil(32));
            for y in 0..3 {
                for x in 0..w {
                    assert!(!bm.get(x, y), "w={w} ({x},{y})");
                }
            }
        }
    }

    #[test]
    fn fill_black_sets_only_in_width_and_zeroes_padding() {
        for &w in &ODD_WIDTHS {
            let bm = MonoBitmap::new(w, 2, true, &limits()).unwrap();
            for y in 0..2 {
                for x in 0..w {
                    assert!(bm.get(x, y), "w={w} ({x},{y}) should be black");
                }
                // Padding bits in the last word must stay zero.
                let last = *bm.row(y).last().unwrap();
                let rem = w % 32;
                if rem != 0 {
                    let pad_mask = !0u32 >> rem; // low (32-rem) bits
                    assert_eq!(last & pad_mask, 0, "w={w} padding not zero");
                }
            }
        }
    }

    #[test]
    fn get_set_roundtrip_and_out_of_range_is_safe() {
        let mut bm = MonoBitmap::new(40, 5, false, &limits()).unwrap();
        bm.set(0, 0, true);
        bm.set(39, 4, true);
        bm.set(33, 2, true);
        assert!(bm.get(0, 0));
        assert!(bm.get(39, 4));
        assert!(bm.get(33, 2));
        assert!(!bm.get(1, 0));
        // Out of range: no panic, reads white, writes ignored.
        assert!(!bm.get(40, 0));
        assert!(!bm.get(0, 5));
        bm.set(1000, 1000, true);
        assert!(!bm.get(1000, 1000));
    }

    #[test]
    fn as_msb_bytes_matches_pixels() {
        let mut bm = MonoBitmap::new(9, 2, false, &limits()).unwrap();
        // row 0: 1010_1010 1  -> 0xAA, 0x80
        for x in (0..9).step_by(2) {
            bm.set(x, 0, true);
        }
        let packed = bm.as_msb_bytes();
        assert_eq!(packed.bytes_per_row(), 2);
        assert_eq!(packed.row(0), vec![0xAA, 0x80]);
        assert_eq!(packed.row(1), vec![0x00, 0x00]);
    }

    /// Reference (per-pixel) combine for differential testing.
    fn combine_reference(
        dst: &MonoBitmap,
        src: &MonoBitmap,
        x: i32,
        y: i32,
        op: CombinationOperator,
    ) -> MonoBitmap {
        let mut out = dst.clone();
        for sy in 0..src.height() {
            for sx in 0..src.width() {
                let dx = x + sx as i32;
                let dy = y + sy as i32;
                if dx < 0 || dy < 0 || dx >= dst.width() as i32 || dy >= dst.height() as i32 {
                    continue;
                }
                let s = src.get(sx, sy);
                let d = dst.get(dx as u32, dy as u32);
                let r = match op {
                    CombinationOperator::Or => d | s,
                    CombinationOperator::And => d & s,
                    CombinationOperator::Xor => d ^ s,
                    CombinationOperator::Xnor => !(d ^ s),
                    CombinationOperator::Replace => s,
                };
                out.set(dx as u32, dy as u32, r);
            }
        }
        out
    }

    fn checkerboard(w: u32, h: u32, phase: u32) -> MonoBitmap {
        let mut bm = MonoBitmap::new(w, h, false, &limits()).unwrap();
        for y in 0..h {
            for x in 0..w {
                if (x + y + phase).is_multiple_of(2) {
                    bm.set(x, y, true);
                }
            }
        }
        bm
    }

    fn assert_same(a: &MonoBitmap, b: &MonoBitmap) {
        assert_eq!(a.width(), b.width());
        assert_eq!(a.height(), b.height());
        for y in 0..a.height() {
            for x in 0..a.width() {
                assert_eq!(a.get(x, y), b.get(x, y), "pixel ({x},{y})");
            }
        }
        // Word representation, including padding, must match exactly.
        assert_eq!(a.words, b.words, "word buffers differ");
    }

    #[test]
    fn combine_all_operators_widths_and_offsets() {
        let ops = [
            CombinationOperator::Or,
            CombinationOperator::And,
            CombinationOperator::Xor,
            CombinationOperator::Xnor,
            CombinationOperator::Replace,
        ];
        for &dw in &ODD_WIDTHS {
            for &sw in &[1u32, 5, 8, 17, 32, 33] {
                let dh = 6u32;
                let sh = 4u32;
                let dst = checkerboard(dw, dh, 0);
                let src = checkerboard(sw, sh, 1);
                for &op in &ops {
                    // Sweep unaligned and clipped placements, including
                    // negative and overhanging positions.
                    for x in [-3i32, -1, 0, 1, 5, dw as i32 - 2, dw as i32 + 4] {
                        for y in [-2i32, 0, 1, dh as i32 - 1, dh as i32 + 2] {
                            let mut got = dst.clone();
                            got.combine(&src, x, y, op);
                            let want = combine_reference(&dst, &src, x, y, op);
                            assert_same(&got, &want);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn combine_preserves_destination_padding() {
        // A Replace that covers the whole width must not set padding bits.
        let dst = MonoBitmap::new(17, 3, false, &limits()).unwrap();
        let src = MonoBitmap::new(17, 3, true, &limits()).unwrap();
        let mut got = dst.clone();
        got.combine(&src, 0, 0, CombinationOperator::Replace);
        for y in 0..3 {
            let last = *got.row(y).last().unwrap();
            assert_eq!(last & (!0u32 >> 17), 0, "padding modified");
        }
    }

    #[cfg(feature = "encode")]
    #[test]
    fn bit_image_roundtrip() {
        let bm = checkerboard(37, 11, 0);
        let img = bm.to_bit_image().unwrap();
        let back = MonoBitmap::from_bit_image(&img, &limits()).unwrap();
        assert_same(&bm, &back);
    }

    #[test]
    fn new_rejects_oversized_dimensions() {
        let narrow = DecodeLimits {
            max_width: 10,
            ..Default::default()
        };
        assert!(matches!(
            MonoBitmap::new(11, 1, false, &narrow),
            Err(DecodeError::Limit(LimitError::Dimension { .. }))
        ));
        let few_pixels = DecodeLimits {
            max_region_pixels: 100,
            ..Default::default()
        };
        assert!(matches!(
            MonoBitmap::new(50, 50, false, &few_pixels),
            Err(DecodeError::Limit(LimitError::Pixels { .. }))
        ));
    }
}
