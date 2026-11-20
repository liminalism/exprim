//! The CPU render surface and pixel compositing.
//!
//! The internal buffer is always 8-bit RGBA with **premultiplied alpha**,
//! matching the frozen output contract (`OutputFormat::Rgba8PremultipliedSrgb`)
//! so the common path is a straight ownership handoff. Compositing is
//! source-over in the device (sRGB-encoded) space — the same space every 8-bit
//! PDF compositor uses; linear-light compositing is a later, opt-in refinement.
//!
//! `Gray8` output is produced by a final downconversion, so there is a single
//! compositing path regardless of the requested format.

use crate::page_ir::DeviceRect;
use crate::render_api::{Background, OutputFormat};
use std::{mem::MaybeUninit, sync::Arc};

/// An RGBA8 premultiplied render target. A surface may be a sub-region of the
/// device plane: `origin` is the device coordinate of its local `(0, 0)`, so
/// bounded offscreen group buffers share the executor's absolute-coordinate
/// math (advice §9).
// `Clone` snapshots the pixel buffer — used by knockout groups, whose every
// element composites against the group's *initial* backdrop (§11.6.6).
#[derive(Debug)]
pub struct Surface {
    pub width: usize,
    pub height: usize,
    pub origin_x: usize,
    pub origin_y: usize,
    /// `width * height * 4` bytes, premultiplied RGBA, row-major.
    data: Arc<[u8]>,
}

// A clone is a knockout-group backdrop snapshot, not another handle to the
// same mutable render target. Keep that value semantics even though the main
// page now lives in its final Arc allocation.
impl Clone for Surface {
    fn clone(&self) -> Self {
        Self {
            width: self.width,
            height: self.height,
            origin_x: self.origin_x,
            origin_y: self.origin_y,
            data: Arc::from(self.data.as_ref()),
        }
    }
}

impl Surface {
    /// The surface's pixel count, or `None` when `width * height * 4` does not
    /// fit a `usize`.
    ///
    /// Release builds have `overflow-checks = false`, so an unchecked
    /// `width * height * 4` would wrap silently and leave a small allocation
    /// paired with large `width`/`height` fields — the buffer and the geometry
    /// would disagree, which is the shape every `assume_init` and raw-store in
    /// this module relies on being impossible. Callers already bound the
    /// output against `max_page_bytes`, so this only ever fires on a caller
    /// bug; degrade to an empty surface rather than a torn one.
    fn checked_pixels(width: usize, height: usize) -> Option<usize> {
        let pixels = width.checked_mul(height)?;
        pixels.checked_mul(4).map(|_| pixels)
    }

    /// Allocate the main page surface (origin `(0,0)`) and paint its background.
    pub fn new(width: usize, height: usize, background: Background) -> Self {
        Self::new_recycled(width, height, background, None)
    }

    /// Allocate the main page surface, reusing `recycled` when it is an
    /// exactly-sized buffer this call can take sole ownership of.
    ///
    /// A page surface is a large, short-lived allocation: at sweep resolution
    /// a single page is well past glibc's 32 MiB dynamic `mmap` ceiling, so
    /// every render `mmap`s fresh pages and pays a minor fault plus a kernel
    /// page-zero for each of them *before* the background fill can write a
    /// byte. Reusing the previous render's buffer keeps those pages resident,
    /// which turns the whole background paint into a plain resident memset.
    /// The bytes written are exactly those [`Surface::new`] would have
    /// written, so the surface the executor sees is identical either way.
    pub fn new_recycled(
        width: usize,
        height: usize,
        background: Background,
        recycled: Option<Arc<[u8]>>,
    ) -> Self {
        let Some(pixels) = Self::checked_pixels(width, height) else {
            return Self::empty();
        };
        let len = pixels * 4;
        let data = match reuse(recycled, len) {
            Some(mut buf) => {
                paint_background(unique_arc_mut(&mut buf), background);
                buf
            }
            None => match background {
                Background::Transparent => filled_arc(len, 0),
                Background::White => filled_arc(len, 0xFF),
                Background::Solid(c) => repeated_rgba_arc(pixels, solid_rgba(c)),
            },
        };
        Self {
            width,
            height,
            origin_x: 0,
            origin_y: 0,
            data,
        }
    }

    /// A page surface whose every row the caller is about to overwrite (strips
    /// copied in): `recycled` is reused as it is, without the pass over the
    /// whole buffer that painting a background costs.
    pub(crate) fn new_unpainted(width: usize, height: usize, recycled: Option<Arc<[u8]>>) -> Self {
        let Some(pixels) = Self::checked_pixels(width, height) else {
            return Self::empty();
        };
        let len = pixels * 4;
        Self {
            width,
            height,
            origin_x: 0,
            origin_y: 0,
            data: reuse(recycled, len).unwrap_or_else(|| zeroed_arc(len)),
        }
    }

    /// The premultiplied RGBA bytes, row-major, `width * 4` per row.
    pub(crate) fn pixels(&self) -> &[u8] {
        &self.data
    }

    /// Consume the surface into tightly packed 8-bit RGB of exactly
    /// `width * height * 3` bytes, in a buffer of its own. Both planes exist
    /// while the copy runs (seven bytes per pixel); a page too large for
    /// that is rendered in strips instead, each converted with
    /// [`rgba_rows_to_rgb8`]. The surface's buffer comes back through
    /// `parked` for reuse.
    pub(crate) fn into_exact_rgb(self, parked: &mut Option<Arc<[u8]>>) -> (usize, Arc<[u8]>) {
        let mut rgb = zeroed_arc(self.width * self.height * 3);
        rgba_rows_to_rgb8(&self.data, unique_arc_mut(&mut rgb));
        let stride = self.width * 3;
        *parked = Some(self.data);
        (stride, rgb)
    }

    /// Give up the pixel buffer, to hand back through [`Self::new_recycled`].
    pub(crate) fn into_buffer(self) -> Arc<[u8]> {
        self.data
    }

    /// Allocate a transparent offscreen surface covering `bounds` (device
    /// space) — for an isolated transparency group.
    pub fn offscreen(bounds: DeviceRect) -> Self {
        let width = bounds.width as usize;
        let height = bounds.height as usize;
        // Same reasoning as `Surface::new`: group bounds are clamped to the
        // page in lowering, so this cannot overflow in practice — but the
        // buffer and the geometry must never be allowed to disagree.
        let Some(pixels) = Self::checked_pixels(width, height) else {
            return Self::empty();
        };
        Self {
            width,
            height,
            origin_x: bounds.x.max(0) as usize,
            origin_y: bounds.y.max(0) as usize,
            data: filled_arc(pixels * 4, 0),
        }
    }

    /// A zero-sized surface. Every row/pixel loop over it is a no-op, so it is
    /// the safe degradation for a geometry that cannot be allocated.
    fn empty() -> Self {
        Self {
            width: 0,
            height: 0,
            origin_x: 0,
            origin_y: 0,
            data: filled_arc(0, 0),
        }
    }

    /// Allocate an offscreen covering `bounds` (device space) **seeded with the
    /// parent's backdrop** — for a *non-isolated* transparency group, whose
    /// elements composite against the group's backdrop rather than a transparent
    /// one (ISO 32000-1 §11.4.7; mirrors PDFium's `GetDIBits` +
    /// `CreateWithBackdrop` in `CPDF_RenderStatus::ProcessTransparency`). Pixels
    /// of `bounds` outside `parent` are left transparent.
    pub fn offscreen_seeded(parent: &Surface, bounds: DeviceRect) -> Self {
        let mut off = Surface::offscreen(bounds);
        let stride = off.width * 4;
        // The overlap is a rectangle, so its column span is the same on every
        // row: resolve it once and copy each row in one `copy_from_slice`
        // rather than testing and copying pixel by pixel. This is the common
        // wrapper case for non-isolated groups, so it runs over whole-page
        // areas.
        let x_start = off.origin_x.max(parent.origin_x);
        let x_end = (off.origin_x + off.width).min(parent.origin_x + parent.width);
        if x_end <= x_start {
            return off;
        }
        let span = (x_end - x_start) * 4;
        let dst_off = (x_start - off.origin_x) * 4;
        let src_off = (x_start - parent.origin_x) * 4;

        let y_start = off.origin_y.max(parent.origin_y);
        let y_end = (off.origin_y + off.height).min(parent.origin_y + parent.height);
        for abs_y in y_start..y_end {
            let ly = abs_y - off.origin_y;
            let prow = parent.local_row(abs_y - parent.origin_y);
            let orow = &mut unique_arc_mut(&mut off.data)[ly * stride..(ly + 1) * stride];
            orow[dst_off..dst_off + span].copy_from_slice(&prow[src_off..src_off + span]);
        }
        off
    }

    pub fn bytes(&self) -> u64 {
        self.data.len() as u64
    }

    /// Mutable access to one row's RGBA bytes, addressed by **absolute** device
    /// `y` — the buffer the span kernels write into.
    #[inline]
    pub fn row_mut(&mut self, y: usize) -> &mut [u8] {
        let stride = self.width * 4;
        let local = y - self.origin_y;
        &mut unique_arc_mut(&mut self.data)[local * stride..(local + 1) * stride]
    }

    /// Contiguous mutable rows for absolute device-Y range `[y0, y1)`, clipped
    /// to this surface. Used by the axis-aligned image fast paths to paint
    /// independent rows in parallel without re-borrowing the surface each row.
    ///
    /// Returns `(buffer, first_absolute_y, row_stride_bytes)`.
    #[inline]
    pub(crate) fn rows_mut_abs(&mut self, y0: usize, y1: usize) -> (&mut [u8], usize, usize) {
        let stride = self.width * 4;
        let local0 = y0.saturating_sub(self.origin_y).min(self.height);
        let local1 = y1.saturating_sub(self.origin_y).min(self.height);
        let local0 = local0.min(local1);
        let first_abs_y = self.origin_y + local0;
        (
            &mut unique_arc_mut(&mut self.data)[local0 * stride..local1 * stride],
            first_abs_y,
            stride,
        )
    }

    /// Immutable access to one row by **local** index (for compositing a group
    /// back onto its parent).
    #[inline]
    pub fn local_row(&self, local_y: usize) -> &[u8] {
        let stride = self.width * 4;
        &self.data[local_y * stride..(local_y + 1) * stride]
    }

    /// Restore pixels from an equal-geometry snapshot without replacing this
    /// surface's allocation. Knockout groups use this between elements so the
    /// scratch surface keeps one allocation for the whole group.
    pub(crate) fn copy_pixels_from(&mut self, source: &Self) {
        debug_assert_eq!(self.width, source.width);
        debug_assert_eq!(self.height, source.height);
        debug_assert_eq!(self.origin_x, source.origin_x);
        debug_assert_eq!(self.origin_y, source.origin_y);
        unique_arc_mut(&mut self.data).copy_from_slice(&source.data);
    }

    /// Consume the surface into output bytes of the requested format.
    /// Returns `(stride, pixels)`.
    #[allow(
        unsafe_code,
        reason = "Arc::new_uninit_slice hands back MaybeUninit; see SAFETY comment"
    )]
    pub fn into_output(self, format: OutputFormat) -> (usize, Arc<[u8]>) {
        self.into_output_recycling(format, &mut None)
    }

    /// Consume the surface into tightly packed 8-bit RGB **inside its own
    /// buffer**: pixel `i` moves from bytes `4i..4i+3` to `3i..3i+3`, front to
    /// back, so a write never passes an unread pixel. Returns `(stride,
    /// pixels)` with `stride = width * 3`; the buffer keeps its RGBA length,
    /// so the last quarter is unspecified. No second plane exists at any
    /// point, where converting into a fresh RGB buffer holds seven bytes per
    /// pixel at the instant of the copy. The alpha byte is dropped exactly as
    /// the copying conversion drops it.
    pub(crate) fn into_packed_rgb(mut self, parked: &mut Option<Arc<[u8]>>) -> (usize, Arc<[u8]>) {
        let data = unique_arc_mut(&mut self.data);
        for pixel in 1..data.len() / 4 {
            let (src, dst) = (pixel * 4, pixel * 3);
            data[dst] = data[src];
            data[dst + 1] = data[src + 1];
            data[dst + 2] = data[src + 2];
        }
        *parked = Some(Arc::clone(&self.data));
        (self.width * 3, self.data)
    }

    /// [`Surface::into_output`], additionally handing this surface's buffer
    /// back through `parked` so the caller can pool it ([`SurfacePool`]) and a
    /// later render of the same geometry can repaint it in place instead of
    /// faulting in a fresh mapping. For the RGBA format the handle is a second
    /// `Arc` on the *returned* buffer: it is reused only once the consumer has
    /// dropped the page (see [`reuse`]).
    #[allow(
        unsafe_code,
        reason = "Arc::new_uninit_slice hands back MaybeUninit; see SAFETY comment"
    )]
    pub(crate) fn into_output_recycling(
        self,
        format: OutputFormat,
        parked: &mut Option<Arc<[u8]>>,
    ) -> (usize, Arc<[u8]>) {
        match format {
            OutputFormat::Rgba8PremultipliedSrgb => {
                *parked = Some(Arc::clone(&self.data));
                (self.width * 4, self.data)
            }
            OutputFormat::Gray8 => {
                let pixels = self.width * self.height;
                debug_assert_eq!(self.data.len(), pixels * 4);
                // Zero pages until written, so nothing is passed over twice
                // and a short source (which the invariant above rules out)
                // would leave zeros, never uninitialized bytes.
                let mut gray = zeroed_arc(pixels);
                rgba_rows_to_gray8(
                    &self.data,
                    unique_arc_mut(&mut gray),
                    crate::render_cpu::exec::output_fan_out(),
                );
                let width = self.width;
                // The RGBA buffer is not the returned page here, so it can be
                // parked outright — nothing else holds it.
                *parked = Some(self.data);
                (width, gray)
            }
        }
    }
}

/// Convert premultiplied RGBA pixels to Gray8, one byte each: the Gray8
/// output conversion, for a whole surface or for rows painted on a strip and
/// written straight into the page's gray buffer. Stops at the shorter slice.
///
/// Every byte equals [`gray8_luma`] of its pixel
/// (`gray8_rows_match_the_float_luma_on_every_input`). The float form made
/// converting a 600 dpi page take several times as long as rendering its
/// text (about 45 ms of a 54 ms page); the integer form here measured half
/// the page time on one thread.
///
/// With `fan_out` a large conversion is split across the rayon pool; pixels
/// are independent, so the bytes do not depend on it.
pub(crate) fn rgba_rows_to_gray8(rgba: &[u8], gray: &mut [u8], fan_out: bool) {
    /// Pixels per task, and the least that is worth splitting.
    const TASK: usize = 1 << 16;
    if fan_out && gray.len() >= 4 * TASK {
        use rayon::prelude::*;
        gray.par_chunks_mut(TASK)
            .zip(rgba.par_chunks(TASK * 4))
            .for_each(|(gray, rgba)| gray8_rows_best(rgba, gray));
    } else {
        gray8_rows_best(rgba, gray);
    }
}

/// [`gray8_rows`] in the widest build this CPU runs.
#[allow(
    unsafe_code,
    reason = "dispatches the AVX2 build of the same safe loop behind a runtime probe"
)]
fn gray8_rows_best(rgba: &[u8], gray: &mut [u8]) {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 was just detected on this CPU.
        return unsafe { gray8_rows_avx2(rgba, gray) };
    }
    gray8_rows(rgba, gray);
}

/// [`gray8_rows`] compiled for AVX2. Same code, wider vectors.
///
/// # Safety
/// The CPU must support AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
unsafe fn gray8_rows_avx2(rgba: &[u8], gray: &mut [u8]) {
    gray8_rows(rgba, gray);
}

/// The Gray8 conversion in integers.
///
/// The luma is `(2126 r + 7152 g + 722 b) / 10000` in real numbers, and
/// [`gray8_luma`] evaluates it in `f32` and rounds half away from zero. The
/// two agree whenever the real value is not exactly halfway between two
/// integers: the numerator is even, so its remainder modulo 10000 is at
/// least 2 away from 5000, 0.0002 in the quotient, while the `f32` evaluation
/// is off by well under 0.0001. On an exact half the `f32` rounding error
/// decides which way the float form goes, and no integer rule reproduces
/// that; such a pixel is rare (one colour in five thousand), so a block that
/// contains one is redone with the float form. Blocks keep the common case
/// free of a per-pixel branch.
#[inline(always)]
fn gray8_rows(rgba: &[u8], gray: &mut [u8]) {
    const BLOCK: usize = 64;
    for (dst, src) in gray.chunks_mut(BLOCK).zip(rgba.chunks(BLOCK * 4)) {
        let mut halves = 0u32;
        for (out, px) in dst.iter_mut().zip(src.chunks_exact(4)) {
            let sum = 2126 * u32::from(px[0]) + 7152 * u32::from(px[1]) + 722 * u32::from(px[2]);
            let whole = sum / 10_000;
            let rest = sum - whole * 10_000;
            halves |= u32::from(rest == 5_000);
            // At most 255: the three weights add up to 10000.
            *out = (whole + u32::from(rest > 5_000)) as u8;
        }
        if halves != 0 {
            for (out, px) in dst.iter_mut().zip(src.chunks_exact(4)) {
                *out = gray8_luma(px[0], px[1], px[2]);
            }
        }
    }
}

/// Copy premultiplied RGBA pixels to packed RGB, dropping the alpha byte.
/// Stops at the shorter slice.
pub(crate) fn rgba_rows_to_rgb8(rgba: &[u8], rgb: &mut [u8]) {
    for (dst, px) in rgb.chunks_exact_mut(3).zip(rgba.chunks_exact(4)) {
        dst.copy_from_slice(&px[..3]);
    }
}

/// The Gray8 output value of one premultiplied (composited-over-black) RGB
/// pixel: its Rec. 709 luma, rounded half away from zero.
///
/// Byte-identical to `y.round().clamp(0.0, 255.0) as u8` over all 2^24 inputs
/// (`gray8_luma_matches_round_on_every_input`), without the `roundf` libm call
/// that `round()` compiles to at the baseline x86-64 ISA, which made this
/// conversion cost more than rendering a whole text page. `y` lies in
/// `[0, 255.00002]`, so `y as i32` is `floor(y)`, `y - floor(y)` is exact,
/// and comparing it with 0.5 is exact round-half-away-from-zero. The sum is
/// kept in its original order; reassociating it changes results.
#[inline(always)]
fn gray8_luma(r: u8, g: u8, b: u8) -> u8 {
    let y = 0.2126 * r as f32 + 0.7152 * g as f32 + 0.0722 * b as f32;
    let floor = y as i32;
    (floor + i32::from(y - floor as f32 >= 0.5)).min(255) as u8
}

/// Allocate the render buffer directly in the `Arc<[u8]>` layout returned by
/// `HostPage`. Going through a `Vec<u8>` forces `Arc::from(Vec<_>)` to allocate
/// and copy the entire page because the Arc refcounts need adjacent storage.
#[allow(
    unsafe_code,
    reason = "Arc::new_uninit_slice/new_zeroed_slice hand back MaybeUninit; see SAFETY comments"
)]
pub(crate) fn filled_arc(len: usize, byte: u8) -> Arc<[u8]> {
    if byte == 0 {
        let data = Arc::<[u8]>::new_zeroed_slice(len);
        // SAFETY: `new_zeroed_slice` initialized every `u8` to a valid zero.
        return unsafe { data.assume_init() };
    }
    let mut data = Arc::<[u8]>::new_uninit_slice(len);
    unique_arc_mut(&mut data).fill(MaybeUninit::new(byte));
    // SAFETY: `fill` initialized every element of the allocation.
    unsafe { data.assume_init() }
}

/// Allocate zeroed bytes directly in their final shared layout. Callers may
/// mutate the allocation through [`unique_arc_mut`] until publishing a clone.
pub(crate) fn zeroed_arc(len: usize) -> Arc<[u8]> {
    filled_arc(len, 0)
}

/// The premultiplied RGBA bytes of a `Background::Solid` color.
#[inline]
fn solid_rgba(c: crate::page_ir::Color) -> [u8; 4] {
    let a = c.a.clamp(0.0, 1.0);
    [
        to_u8(c.r.clamp(0.0, 1.0) * a),
        to_u8(c.g.clamp(0.0, 1.0) * a),
        to_u8(c.b.clamp(0.0, 1.0) * a),
        to_u8(a),
    ]
}

/// Write `background` over an already-allocated surface buffer. Byte-for-byte
/// what the matching `filled_arc`/`repeated_rgba_arc` allocation would hold.
fn paint_background(buf: &mut [u8], background: Background) {
    match background {
        Background::Transparent => buf.fill(0),
        Background::White => buf.fill(0xFF),
        Background::Solid(c) => {
            let px = solid_rgba(c);
            for chunk in buf.chunks_exact_mut(4) {
                chunk.copy_from_slice(&px);
            }
        }
    }
}

/// Default byte cap of a backend's [`SurfacePool`].
pub(crate) const SURFACE_POOL_BYTES: usize = 256 << 20;

/// Distinct buffer sizes a [`SurfacePool`] keeps: the three most recently
/// parked. A document processor renders a page at an analysis size, the base
/// size and an OCR size in turn, and those must all stay; but the pages of a
/// scanned book each have their own size, and without this bound the pool
/// filled to its cap with buffers that were never asked for again (eight
/// pages of a 300 dpi scan kept 150 MB parked where three slots kept 57 MB).
const POOL_SIZES: usize = 3;

/// Page buffers of finished renders, kept so the next render of the same
/// geometry repaints resident memory instead of faulting in a fresh mapping.
///
/// One pool per backend, shared by every thread that renders through it, so
/// the buffers live and die with the backend (one document render session).
/// They used to be parked per thread — one slot in each worker context plus
/// two more sizes in a thread-local — which no byte limit covered and which
/// outlived the session on every pool thread that had served it: twenty
/// threads at 300 dpi kept 670 MB parked after the last page. Any thread can
/// take any free buffer here, so the pool gives at least the hits the
/// per-thread slots did while it is under its cap.
///
/// A parked buffer may still be held by the consumer of the page it carried;
/// it is handed out only once this pool is its sole owner. Parked buffers are
/// repainted in full before use, so which one a render gets never changes
/// output bytes.
#[derive(Debug)]
pub(crate) struct SurfacePool {
    cap_bytes: usize,
    /// Oldest first.
    parked: std::sync::Mutex<Vec<Arc<[u8]>>>,
}

static SLOT_HITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static SLOT_MISSES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Process-wide `(hits, misses)` of parked-surface lookups (a hit handed out a
/// parked buffer of the wanted size).
pub fn surface_slot_stats() -> (u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (SLOT_HITS.load(Relaxed), SLOT_MISSES.load(Relaxed))
}

impl SurfacePool {
    pub(crate) fn new(cap_bytes: usize) -> Self {
        Self {
            cap_bytes,
            parked: std::sync::Mutex::new(Vec::new()),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Vec<Arc<[u8]>>> {
        // A poisoned lock only means another render panicked while holding
        // it; the list of buffers is still a valid list.
        self.parked
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Take a parked buffer for a `width x height` surface, if one of exactly
    /// that size is parked and no consumer still holds it.
    pub(crate) fn take(&self, width: usize, height: usize) -> Option<Arc<[u8]>> {
        use std::sync::atomic::Ordering::Relaxed;
        let found = Surface::checked_pixels(width, height)
            .map(|pixels| pixels * 4)
            .and_then(|len| {
                let mut parked = self.lock();
                // Most recently parked first: the likeliest to be resident.
                let index = parked
                    .iter()
                    .rposition(|buf| buf.len() == len && Arc::strong_count(buf) == 1)?;
                Some(parked.remove(index))
            });
        match found {
            Some(_) => SLOT_HITS.fetch_add(1, Relaxed),
            None => SLOT_MISSES.fetch_add(1, Relaxed),
        };
        found
    }

    /// Park `buf`, dropping the oldest parked buffers beyond the byte cap. A
    /// buffer larger than the whole cap is not retained.
    pub(crate) fn park(&self, buf: Arc<[u8]>) {
        if buf.len() > self.cap_bytes {
            return;
        }
        let mut parked = self.lock();
        parked.push(buf);
        let mut sizes = [0usize; POOL_SIZES];
        let mut kept = 0;
        for buf in parked.iter().rev() {
            if kept < POOL_SIZES && !sizes[..kept].contains(&buf.len()) {
                sizes[kept] = buf.len();
                kept += 1;
            }
        }
        parked.retain(|buf| sizes[..kept].contains(&buf.len()));
        let mut total: usize = parked.iter().map(|buf| buf.len()).sum();
        while total > self.cap_bytes && parked.len() > 1 {
            total -= parked.remove(0).len();
        }
    }

    /// Drop every parked buffer.
    pub(crate) fn clear(&self) {
        self.lock().clear();
    }

    /// Bytes this pool keeps alive on its own account (buffers no consumer
    /// holds any more).
    pub(crate) fn idle_bytes(&self) -> usize {
        self.lock()
            .iter()
            .filter(|buf| Arc::strong_count(buf) == 1)
            .map(|buf| buf.len())
            .sum()
    }
}

/// Accept a recycled buffer only when it is exactly the size wanted and this
/// call can take sole ownership of it — a buffer the previous consumer still
/// holds is dropped here rather than retained, so the pool never pins memory
/// that is in use elsewhere.
fn reuse(recycled: Option<Arc<[u8]>>, len: usize) -> Option<Arc<[u8]>> {
    let mut buf = recycled?;
    if buf.len() != len {
        return None;
    }
    Arc::get_mut(&mut buf)?;
    Some(buf)
}

#[allow(
    unsafe_code,
    reason = "Arc::new_uninit_slice hands back MaybeUninit; see SAFETY comment"
)]
fn repeated_rgba_arc(pixels: usize, rgba: [u8; 4]) -> Arc<[u8]> {
    let mut data = Arc::<[u8]>::new_uninit_slice(pixels * 4);
    for chunk in unique_arc_mut(&mut data).chunks_exact_mut(4) {
        for (dst, src) in chunk.iter_mut().zip(rgba) {
            dst.write(src);
        }
    }
    // SAFETY: `chunks_exact_mut(4)` covers the complete pixels*4 allocation.
    unsafe { data.assume_init() }
}

#[inline]
fn to_u8(v: f32) -> u8 {
    (v * 255.0 + 0.5).clamp(0.0, 255.0) as u8
}

#[inline]
pub(crate) fn unique_arc_mut<T>(data: &mut Arc<[T]>) -> &mut [T] {
    let Some(data) = Arc::get_mut(data) else {
        unreachable!("a final buffer is never shared while it is mutable")
    };
    data
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // test code: panics are the assertion mechanism
    use super::*;

    #[test]
    fn rgba_output_hands_off_the_surface_pixels() {
        let mut surface = Surface::new(2, 1, Background::White);
        surface.row_mut(0)[0] = 17;

        let (stride, output) = surface.into_output(OutputFormat::Rgba8PremultipliedSrgb);

        assert_eq!(stride, 8);
        assert_eq!(&*output, &[17, 255, 255, 255, 255, 255, 255, 255]);
        assert_eq!(Arc::strong_count(&output), 1);
    }

    #[test]
    fn a_recycled_surface_is_byte_identical_to_a_fresh_one() {
        let backgrounds = [
            Background::White,
            Background::Transparent,
            Background::Solid(crate::page_ir::Color {
                r: 0.25,
                g: 0.5,
                b: 0.75,
                a: 0.5,
            }),
        ];
        for background in backgrounds {
            // A dirtied buffer of the right size, sole-owned: the shape a
            // previous render hands back.
            let recycled = filled_arc(5 * 3 * 4, 0x5A);
            let allocation = Arc::as_ptr(&recycled);
            let reused = Surface::new_recycled(5, 3, background, Some(recycled));
            let fresh = Surface::new(5, 3, background);
            assert_eq!(&*reused.data, &*fresh.data, "{background:?}");
            assert_eq!(
                Arc::as_ptr(&reused.data),
                allocation,
                "the recycled allocation should have been kept"
            );
        }
    }

    #[test]
    fn recycling_declines_a_shared_or_mismatched_buffer() {
        // Still held elsewhere: must not be written through.
        let shared = filled_arc(2 * 4, 0x11);
        let keep = Arc::clone(&shared);
        let surface = Surface::new_recycled(2, 1, Background::White, Some(shared));
        assert_ne!(Arc::as_ptr(&surface.data), Arc::as_ptr(&keep));
        assert_eq!(&*keep, &[0x11; 8]);
        assert_eq!(&*surface.data, &[255; 8]);

        // Wrong length: also declined.
        let wrong = filled_arc(4, 0x22);
        let ptr = Arc::as_ptr(&wrong);
        let surface = Surface::new_recycled(2, 1, Background::White, Some(wrong));
        assert_ne!(Arc::as_ptr(&surface.data), ptr);
        assert_eq!(&*surface.data, &[255; 8]);
    }

    /// The integer-rounded Gray8 conversion must equal the `round()` form it
    /// replaced on every (r, g, b). Over 2^24 inputs, 2,752 land on exact .5
    /// ties, which is why `(y + 0.5) as u8` and round-half-even are not
    /// substitutes.
    #[test]
    fn gray8_luma_matches_round_on_every_input() {
        for r in 0..=255u8 {
            for g in 0..=255u8 {
                for b in 0..=255u8 {
                    let y = 0.2126 * r as f32 + 0.7152 * g as f32 + 0.0722 * b as f32;
                    let want = y.round().clamp(0.0, 255.0) as u8;
                    assert_eq!(gray8_luma(r, g, b), want, "rgb ({r}, {g}, {b})");
                }
            }
        }
    }

    /// The integer Gray8 kernel must equal the float luma on every (r, g, b),
    /// in whatever position of a block the pixel falls and whichever build
    /// of the kernel this CPU selects. Rows of 256 pixels put every blue
    /// value of one (r, g) through four blocks, the exact halves among them.
    #[test]
    fn gray8_rows_match_the_float_luma_on_every_input() {
        let mut rgba = vec![0u8; 256 * 4];
        let mut got = vec![0u8; 256];
        let mut plain = vec![0u8; 256];
        let mut halves = 0u32;
        for r in 0..=255u8 {
            for g in 0..=255u8 {
                for (b, px) in rgba.chunks_exact_mut(4).enumerate() {
                    px.copy_from_slice(&[r, g, b as u8, 255]);
                    let sum = 2126 * u32::from(r) + 7152 * u32::from(g) + 722 * b as u32;
                    halves += u32::from(sum % 10_000 == 5_000);
                }
                rgba_rows_to_gray8(&rgba, &mut got, false);
                gray8_rows(&rgba, &mut plain);
                for (b, (&got, &plain)) in got.iter().zip(&plain).enumerate() {
                    let want = gray8_luma(r, g, b as u8);
                    assert_eq!(got, want, "rgb ({r}, {g}, {b})");
                    assert_eq!(plain, want, "rgb ({r}, {g}, {b}), baseline build");
                }
            }
        }
        // The fallback was exercised: these are the inputs it exists for.
        assert!(halves > 1_000, "{halves} exact halves");
        // A ragged tail and a short destination.
        let rgba: Vec<u8> = (0..4 * 77).map(|i| (i * 37 % 251) as u8).collect();
        let mut got = vec![7u8; 70];
        rgba_rows_to_gray8(&rgba, &mut got, false);
        for (i, &got) in got.iter().enumerate() {
            let px = &rgba[i * 4..];
            assert_eq!(got, gray8_luma(px[0], px[1], px[2]), "pixel {i}");
        }
        // Split across the pool, with a ragged last task: the same bytes.
        let pixels = (5 << 16) + 123;
        let rgba: Vec<u8> = (0..pixels * 4).map(|i| (i * 131 % 253) as u8).collect();
        let (mut serial, mut split) = (vec![0u8; pixels], vec![0u8; pixels]);
        rgba_rows_to_gray8(&rgba, &mut serial, false);
        rgba_rows_to_gray8(&rgba, &mut split, true);
        assert!(serial == split);
    }

    #[test]
    fn into_output_parks_the_buffer_for_the_next_render() {
        let mut pool = None;
        let surface = Surface::new(3, 2, Background::White);
        let allocation = Arc::as_ptr(&surface.data);
        let (_, pixels) =
            surface.into_output_recycling(OutputFormat::Rgba8PremultipliedSrgb, &mut pool);
        // Parked, but the consumer still owns it, so it is not reusable yet.
        assert_eq!(Arc::as_ptr(pool.as_ref().unwrap()), allocation);
        assert!(reuse(pool.clone(), 3 * 2 * 4).is_none());
        drop(pixels);
        // Once the consumer drops it, the same allocation comes back.
        let recycled = reuse(pool, 3 * 2 * 4).expect("sole owner now");
        assert_eq!(Arc::as_ptr(&recycled), allocation);

        // Gray8 does not hand back the RGBA buffer, so it is parked outright.
        let mut pool = None;
        let surface = Surface::new(3, 2, Background::White);
        let allocation = Arc::as_ptr(&surface.data);
        let (_, gray) = surface.into_output_recycling(OutputFormat::Gray8, &mut pool);
        assert_eq!(gray.len(), 6);
        assert_eq!(Arc::as_ptr(pool.as_ref().unwrap()), allocation);
        assert!(reuse(pool, 3 * 2 * 4).is_some());
    }

    #[test]
    fn alternating_sizes_reuse_their_parked_buffers() {
        // Alternate three sizes the way analysis/base/OCR renders do.
        let sizes = [(5usize, 4usize), (9, 7), (3, 2)];
        let pool = SurfacePool::new(SURFACE_POOL_BYTES);
        let mut addrs = [None; 3];
        let (h0, m0) = surface_slot_stats();
        for round in 0..4 {
            for (i, &(w, h)) in sizes.iter().enumerate() {
                let recycled = pool.take(w, h);
                if round == 0 {
                    assert!(recycled.is_none());
                } else {
                    let r = recycled.as_ref().expect("parked slot hit");
                    assert_eq!(Arc::as_ptr(r), addrs[i].unwrap(), "size {i} round {round}");
                }
                let surface = Surface::new_recycled(w, h, Background::White, recycled);
                addrs[i].get_or_insert(Arc::as_ptr(&surface.data));
                assert_eq!(Arc::as_ptr(&surface.data), addrs[i].unwrap());
                let mut parked = None;
                let (_, out) = surface.into_output_recycling(OutputFormat::Gray8, &mut parked);
                pool.park(parked.unwrap());
                assert_eq!(out.len(), w * h);
            }
        }
        let (h1, m1) = surface_slot_stats();
        assert!(h1 - h0 >= 9 && m1 - m0 >= 3);
        // Recycled output is identical to a fresh surface.
        let fresh = Surface::new(9, 7, Background::White);
        let again = Surface::new_recycled(9, 7, Background::White, pool.take(9, 7));
        assert_eq!(fresh.data, again.data);
    }

    #[test]
    fn the_pool_keeps_no_more_than_its_cap_and_never_lends_a_held_buffer() {
        // Room for two 4x4 RGBA buffers (64 bytes each).
        let pool = SurfacePool::new(128);
        let page = |w: usize, h: usize| {
            let surface = Surface::new(w, h, Background::White);
            let mut parked = None;
            let (_, pixels) =
                surface.into_output_recycling(OutputFormat::Rgba8PremultipliedSrgb, &mut parked);
            (pixels, parked.unwrap())
        };
        // Still held by its consumer: parked, but not lent out and not idle.
        let (held, handle) = page(4, 4);
        pool.park(handle);
        assert!(pool.take(4, 4).is_none());
        assert_eq!(pool.idle_bytes(), 0);
        drop(held);
        assert_eq!(pool.idle_bytes(), 64);
        // A third buffer pushes the oldest out; the cap holds.
        for _ in 0..2 {
            let (pixels, handle) = page(4, 4);
            drop(pixels);
            pool.park(handle);
        }
        assert_eq!(pool.idle_bytes(), 128);
        // Larger than the whole cap: not retained at all.
        let (pixels, handle) = page(8, 8);
        drop(pixels);
        pool.park(handle);
        assert_eq!(pool.idle_bytes(), 128);
        assert!(pool.take(8, 8).is_none());
        assert!(pool.take(4, 4).is_some());
        pool.clear();
        assert_eq!(pool.idle_bytes(), 0);

        // Only the three most recently parked sizes stay: a run of pages that
        // each have their own size does not fill the pool.
        let pool = SurfacePool::new(usize::MAX);
        for side in 1..=6usize {
            let (pixels, handle) = page(side, side);
            drop(pixels);
            pool.park(handle);
        }
        assert_eq!(pool.idle_bytes(), (16 + 25 + 36) * 4);
        assert!(pool.take(3, 3).is_none());
        assert!(pool.take(5, 5).is_some());
    }

    #[test]
    fn packing_rgb_in_place_matches_dropping_alpha_into_a_copy() {
        let (w, h) = (7usize, 5usize);
        let mut surface = Surface::new(w, h, Background::White);
        for y in 0..h {
            for (x, px) in surface.row_mut(y).chunks_exact_mut(4).enumerate() {
                px.copy_from_slice(&[(x * 31 + y) as u8, (y * 17) as u8, (x ^ y) as u8, 255]);
            }
        }
        let expected: Vec<u8> = surface
            .data
            .chunks_exact(4)
            .flat_map(|px| [px[0], px[1], px[2]])
            .collect();
        let allocation = Arc::as_ptr(&surface.data);
        let mut parked = None;
        let (stride, pixels) = surface.into_packed_rgb(&mut parked);
        assert_eq!(stride, w * 3);
        assert_eq!(Arc::as_ptr(&pixels), allocation, "no second buffer");
        assert_eq!(&pixels[..w * h * 3], &expected[..]);
        // The parked handle is the same buffer, reusable once the page is dropped.
        drop(pixels);
        assert!(reuse(parked, w * h * 4).is_some());
    }

    #[test]
    fn clone_remains_an_independent_pixel_snapshot() {
        let original = Surface::new(
            1,
            1,
            Background::Solid(crate::page_ir::Color {
                r: 0.25,
                g: 0.5,
                b: 0.75,
                a: 1.0,
            }),
        );
        let mut clone = original.clone();
        clone.row_mut(0).fill(0);

        assert_ne!(original.local_row(0), clone.local_row(0));
        assert_eq!(original.local_row(0), &[64, 128, 191, 255]);
    }

    #[test]
    fn copying_a_snapshot_reuses_the_destination_allocation() {
        let source = Surface::new(4, 3, Background::White);
        let mut scratch = Surface::new(4, 3, Background::Transparent);
        let allocation = Arc::as_ptr(&scratch.data);

        scratch.copy_pixels_from(&source);

        assert_eq!(Arc::as_ptr(&scratch.data), allocation);
        assert_eq!(scratch.data, source.data);
    }

    #[test]
    fn gray_output_is_fully_initialized() {
        let surface = Surface::new(2, 1, Background::White);
        let (stride, output) = surface.into_output(OutputFormat::Gray8);

        assert_eq!(stride, 2);
        assert_eq!(&*output, &[255, 255]);
    }
}
