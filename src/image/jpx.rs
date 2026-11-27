//! `/JPXDecode` (JPEG 2000) via the `jp2lam` crate.
//!
//! `jp2lam` is a deliberate external dependency, not a vendored module: other
//! programs call the same decoder, so it is developed as its own project and consumed
//! here through the codec seam like any other.
//!
//! A JPX stream carries its own colour space and bit depth, which take
//! precedence over the image dictionary's (ISO 32000-1 §7.4.9: `/ColorSpace`
//! may be absent, and when present it is only a hint for `/SMaskInData`
//! handling). The decoded format therefore comes from the codestream.
//!
//! Decode uses jp2lam's optimized request API via a thread-local
//! [`jp2lam::Jp2Decoder`] session (retained Tier-1 scratch; all sessions share
//! jp2lam's stable Rayon executor). For 8-bit components the codec writes the packed 8-bit raster
//! directly (`Gray8`/`Rgb8`/`Rgba8`/`Cmyk8`), removing the renderer-side
//! full-image interleave pass and the planar `i32` intermediates. Streams
//! whose components are not 8-bit (1/2/4-bit bitonal and palette scans, 12/16-bit
//! deep scans) fall back to the native planar decode and PDFium's own
//! shift-based widening (`src << (8 - prec)` below 8 bits, `src >> (prec - 8)`
//! above), which is what the legacy adapter did and is bit-exact with it.
//!
//! jp2lam also exposes opaque pad layouts (`Rgbx8`/`Bgra8`) for four-byte
//! destinations without a codestream alpha plane. The CPU paint path still
//! samples tight DeviceRGB; wire those formats when a GPU upload or BGRA
//! surface path is ready (see `codecs/jp2lam/decode-fix-plan/` in the repository).

use std::cell::RefCell;
use std::sync::Arc;

use jp2lam::{
    ColorSpace, DEFAULT_DECODE_BUFFER_POOL_BYTES, DecodeConcurrency,
    DecodeLimits as Jp2DecodeLimits, DecodeOutputFormat, DecodeRequest, DecodeResolution,
    DecodeResult, DecodeTarget, Jp2BufferPool, PreparedJp2,
};

use crate::image::codec::{DecodeLimits, DecodedFormat, DecodedImage, ImageCodec};
use crate::image::{DecodeParameters, ImageDescriptor, ImageError, StreamFilter};

/// Shared zero-capacity placeholder attached to idle thread-local decoders so
/// worker threads never hold strong references to a dropped [`JpxCodec`]'s
/// buffer pool.
fn empty_jp2_pool() -> Arc<Jp2BufferPool> {
    use std::sync::OnceLock;
    static EMPTY: OnceLock<Arc<Jp2BufferPool>> = OnceLock::new();
    Arc::clone(EMPTY.get_or_init(|| Arc::new(Jp2BufferPool::disabled())))
}

thread_local! {
    /// A reusable decoder session per render worker. `jp2lam::Jp2Decoder`
    /// retains small per-block Tier-1 scratch (~16 KiB) while large subband
    /// planes are checked out of [`JpxCodec`]'s shared [`Jp2BufferPool`] for
    /// the duration of each call and detached on return.
    static JPX_DECODER: RefCell<jp2lam::Jp2Decoder> =
        RefCell::new(jp2lam::Jp2Decoder::with_buffer_pool(empty_jp2_pool()));
}

///Temporarily attaches `pool` to `decoder` and detaches it on drop (including
/// unwinding) so no worker thread retains a session's pool after a decode.
struct ScopedDecoderPool<'a> {
    decoder: &'a mut jp2lam::Jp2Decoder,
}

impl<'a> ScopedDecoderPool<'a> {
    fn attach(decoder: &'a mut jp2lam::Jp2Decoder, pool: Arc<Jp2BufferPool>) -> Self {
        decoder.set_buffer_pool(pool);
        Self { decoder }
    }
}

impl Drop for ScopedDecoderPool<'_> {
    fn drop(&mut self) {
        self.decoder.set_buffer_pool(empty_jp2_pool());
    }
}

/// Reduced-resolution quality margin (Phase 2). When the renderer supplies a
/// device-footprint hint, the decoder drops wavelet resolutions but only while
/// the decoded resolution stays at least this multiple of the destination
/// footprint. 1.0 is the principled floor — never decode below one texel per
/// destination pixel — and, because wavelet reduction is discrete (each level
/// halves), it still leaves the decoded image between 1x and 2x the destination
/// in practice (real supersample headroom). On the two profiled hotspot pages
/// this engages one reduction level each with PDFium differential severity of
/// 0.000096 (jpx-scan) and 0.003781 (mrc) — both well under the 0.005 budget,
/// and the mrc figure is below its full-resolution baseline. A larger margin
/// (measured up through 1.35) foregoes the mrc reduction for no severity gain.
/// NOTE: the renderer-side hint (`codec_target_size`) additionally applies a
/// supersampling headroom before this margin — see
/// `exprim::render_cpu/src/prepared.rs` — so the decoded raster the resampler sees
/// stays comfortably above the destination footprint.
/// See corpus/perf/optimization-jpx-integration-20260720.md.
const JPX_QUALITY_MARGIN: f32 = 1.0;

/// The effective quality margin, overridable via `PDF_RENDERER_JPX_MARGIN` for
/// margin sweeps during measurement. Read once.
fn jpx_quality_margin() -> f32 {
    use std::sync::OnceLock;
    static MARGIN: OnceLock<f32> = OnceLock::new();
    *MARGIN.get_or_init(|| {
        std::env::var("PDF_RENDERER_JPX_MARGIN")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .filter(|m| m.is_finite() && *m >= 1.0)
            .unwrap_or(JPX_QUALITY_MARGIN)
    })
}

/// A fixed concurrency override from `PDF_RENDERER_JPX_CONCURRENCY` (`serial`,
/// `budgeted:N`, a bare `N`, or `auto`), read once. `None` selects the
/// load-aware policy. Keeps the concurrency sweep and pinning reproducible on
/// the shipping binary.
fn jpx_concurrency_override() -> Option<DecodeConcurrency> {
    use std::sync::OnceLock;
    static C: OnceLock<Option<DecodeConcurrency>> = OnceLock::new();
    *C.get_or_init(|| {
        std::env::var("PDF_RENDERER_JPX_CONCURRENCY")
            .ok()
            .and_then(|v| parse_jpx_concurrency(&v))
    })
}

/// Parses a `PDF_RENDERER_JPX_CONCURRENCY` value. A bare `1` means `serial`
/// and a bare `N > 1` means `budgeted:N`, so the single-thread benchmark
/// protocol's `=1` pins the decode rather than silently selecting the
/// load-aware policy. `auto` and anything unrecognised return `None`.
fn parse_jpx_concurrency(value: &str) -> Option<DecodeConcurrency> {
    let v = value.trim().to_ascii_lowercase();
    if v == "serial" {
        return Some(DecodeConcurrency::Serial);
    }
    let n = v.strip_prefix("budgeted:").unwrap_or(&v);
    match n.parse::<usize>().ok()? {
        0 => None,
        1 if n.len() == v.len() => Some(DecodeConcurrency::Serial),
        n => Some(DecodeConcurrency::Budgeted(n)),
    }
}

/// Optional one-line stderr telemetry per JPX decode when `JP2LAM_DECODE_TRACE`
/// is set (`1`, `true`, `yes`). Off by default.
fn jpx_decode_trace_enabled() -> bool {
    use std::sync::OnceLock;
    static TRACE: OnceLock<bool> = OnceLock::new();
    *TRACE.get_or_init(|| {
        std::env::var("JP2LAM_DECODE_TRACE")
            .ok()
            .is_some_and(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
    })
}

/// Live count of JPX decodes in flight across all render workers. Used to size
/// each decode's internal Tier-1 thread budget to the *current* machine load.
static JPX_IN_FLIGHT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Increments the in-flight decode count for its lifetime.
struct InFlightGuard;
impl InFlightGuard {
    fn enter() -> Self {
        JPX_IN_FLIGHT.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        Self
    }
    /// In-flight decodes right now, including this one (always ≥ 1).
    fn current(&self) -> usize {
        JPX_IN_FLIGHT
            .load(std::sync::atomic::Ordering::Acquire)
            .max(1)
    }
}
impl Drop for InFlightGuard {
    fn drop(&mut self) {
        JPX_IN_FLIGHT.fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
    }
}

/// Lower and upper bound on the per-decode Tier-1 thread budget.
const JPX_BUDGET_MIN: usize = 1;
const JPX_BUDGET_MAX: usize = 8;

/// Load-aware per-decode concurrency. Render workers already parallelise across
/// draws, so a decode must not grab the whole machine when many are running —
/// but when few are (the single-page / viewer case) it should use more of a big
/// JPX's internal Tier-1 parallelism. Budget = clamp(cores / in_flight, 1, 8):
/// one lone decode on a 20-core host runs at 8; a saturated scheduler settles to
/// serial-equivalent `Budgeted(1)`. Concurrent budgeted requests share jp2lam's
/// fixed global executor cap rather than each owning a private Rayon pool.
fn load_aware_budget(in_flight: usize) -> DecodeConcurrency {
    use std::sync::OnceLock;
    static CORES: OnceLock<usize> = OnceLock::new();
    let cores = *CORES.get_or_init(|| {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(JPX_BUDGET_MIN)
    });
    let budget = (cores / in_flight).clamp(JPX_BUDGET_MIN, JPX_BUDGET_MAX);
    DecodeConcurrency::Budgeted(budget)
}

/// Default byte ceiling for [`JpxCodec`]'s session-scoped coefficient buffer
/// pool (128 MiB across all render workers).
pub const DEFAULT_JPX_BUFFER_POOL_BYTES: usize = DEFAULT_DECODE_BUFFER_POOL_BYTES;

/// The `/JPXDecode` codec for a [`crate::image::CodecRegistry`].
///
/// Owns one byte-capped [`Jp2BufferPool`] shared across every worker thread
/// that decodes through this codec instance, mirroring `SurfacePool`: buffers
/// stay warm between page decodes without multiplying memory per worker, are
/// freed from any thread on [`ImageCodec::release_idle_memory`], and drop with
/// the owning backend/session.
#[derive(Debug)]
pub struct JpxCodec {
    pool: Arc<Jp2BufferPool>,
}

impl Default for JpxCodec {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for JpxCodec {
    fn drop(&mut self) {
        self.pool.clear();
    }
}

impl ImageCodec for JpxCodec {
    fn filter(&self) -> StreamFilter {
        StreamFilter::Jpx
    }

    fn decode(
        &self,
        data: &[u8],
        descriptor: &ImageDescriptor,
        params: &DecodeParameters,
        limits: &DecodeLimits,
    ) -> Result<DecodedImage, ImageError> {
        limits.check_input(data.len())?;
        // Parse the JP2 container and codestream framing once into a borrowed
        // `PreparedJp2` handle: zero ICC copy, zero duplicate header/tile-part
        // parse when `decode_prepared_into_cancellable` runs below.
        let codec_limits = decoder_limits(limits);
        let prepared = jp2lam::prepare_jp2(data, &codec_limits)
            .map_err(|e| ImageError::Decode(format!("JPX: {e}")))?;
        if limits.is_cancelled() {
            return Err(ImageError::Cancelled);
        }

        let (width, height) = (prepared.width(), prepared.height());
        if width == 0 || height == 0 {
            return Err(ImageError::Decode("JPX: zero dimension".into()));
        }
        if u64::from(width) * u64::from(height) > limits.max_pixels {
            return Err(ImageError::TooLarge { width, height });
        }

        let all_8bit = prepared
            .codestream()
            .siz
            .components
            .iter()
            .all(|component| component.precision == 8);
        let has_palette = prepared.container_palette_channels().is_some();
        let ignore_container_palette = has_palette
            && descriptor
                .color_space
                .as_ref()
                .is_some_and(|cs| cs.family == crate::color::ColorSpaceFamily::Indexed);

        let (format, channels) = if ignore_container_palette {
            (DecodedFormat::Gray8, 1usize)
        } else {
            match prepared.colorspace() {
                ColorSpace::Gray
                    if prepared.codestream().siz.components.len() == 2
                        && descriptor
                            .color_space
                            .as_ref()
                            .is_some_and(|cs| cs.components == 2) =>
                {
                    (DecodedFormat::Multi2, 2usize)
                }
                ColorSpace::Gray
                    if prepared.in_data_alpha().is_some()
                        && (!params.ignore_alpha || !all_8bit || has_palette) =>
                {
                    (DecodedFormat::GrayA8, 2usize)
                }
                ColorSpace::Gray => (DecodedFormat::Gray8, 1usize),
                ColorSpace::Srgb
                    if prepared.in_data_alpha().is_some()
                        && (!params.ignore_alpha || !all_8bit || has_palette) =>
                {
                    (DecodedFormat::Rgba8, 4usize)
                }
                ColorSpace::Srgb => (DecodedFormat::Rgb8, 3usize),
                ColorSpace::YCbCr => (DecodedFormat::Rgb8, 3usize),
                ColorSpace::Cmyk => (DecodedFormat::Cmyk8, 4usize),
                other => {
                    return Err(ImageError::Decode(format!(
                        "JPX: unsupported colorspace {other:?}"
                    )));
                }
            }
        };

        let resolution = decode_resolution(params);

        let in_flight = InFlightGuard::enter();
        let concurrency =
            jpx_concurrency_override().unwrap_or_else(|| load_aware_budget(in_flight.current()));

        let packed_direct = all_8bit && (!has_palette || ignore_container_palette);
        let decoded_dims = if packed_direct {
            Some(
                prepared
                    .decoded_dimensions(resolution)
                    .map_err(|e| ImageError::Decode(format!("JPX: {e}")))?,
            )
        } else {
            None
        };
        let trace_start = jpx_decode_trace_enabled().then(std::time::Instant::now);
        let result = if let Some((decoded_width, decoded_height)) = decoded_dims {
            self.decode_packed(
                &prepared,
                resolution,
                format,
                channels,
                decoded_width,
                decoded_height,
                limits,
                concurrency,
                ignore_container_palette,
            )
        } else {
            self.decode_native_interleaved(
                &prepared,
                resolution,
                format,
                channels,
                width,
                height,
                limits,
                concurrency,
                ignore_container_palette,
            )
        };
        let in_flight_now = in_flight.current();
        drop(in_flight);
        if let (Some(started), Ok(image)) = (trace_start, &result) {
            eprintln!(
                "[jpx-trace] src={}x{} out={}x{} fmt={:?} path={} conc={:?} in_flight={} elapsed_ms={:.3}",
                width,
                height,
                image.width,
                image.height,
                image.format,
                if packed_direct { "packed" } else { "planar" },
                concurrency,
                in_flight_now,
                started.elapsed().as_secs_f64() * 1000.0
            );
        }
        if limits.is_cancelled() {
            return Err(ImageError::Cancelled);
        }
        result
    }

    /// The wavelet levels the decode drops for this draw, read from the main
    /// header (SIZ/COD) through jp2lam's own selection rule without copying ICC
    /// profiles or scanning tile-parts.
    fn reduction_level(
        &self,
        data: &[u8],
        _descriptor: &ImageDescriptor,
        params: &DecodeParameters,
        limits: &DecodeLimits,
    ) -> Option<u8> {
        let resolution = decode_resolution(params);
        if resolution == DecodeResolution::Full {
            return Some(0);
        }
        let codec_limits = decoder_limits(limits);
        jp2lam::inspect_jp2_header_for_reduction(data, resolution, &codec_limits).ok()
    }

    fn release_idle_memory(&self) {
        self.pool.clear();
        JPX_DECODER.with(|decoder| {
            *decoder.borrow_mut() = jp2lam::Jp2Decoder::with_buffer_pool(empty_jp2_pool());
        });
    }
}

/// The jp2lam resolution request for a draw, shared by the decode and its
/// [`ImageCodec::reduction_level`].
///
/// Phase 2: a device-footprint hint lets the decoder drop wavelet resolutions
/// for a minified draw. `AtLeast` is inherently conservative — it never reduces
/// below the destination footprint scaled by the quality margin, so magnified
/// and near-1:1 draws decode at full resolution. Absent a hint (whole-document
/// paths, Phase-1 gate) the resolution is `Full` and output is byte-identical
/// to the legacy path.
fn decode_resolution(params: &DecodeParameters) -> DecodeResolution {
    match params.target_size {
        Some((tw, th)) if tw > 0 && th > 0 => DecodeResolution::AtLeast {
            width: tw,
            height: th,
            quality_margin: jpx_quality_margin(),
        },
        _ => DecodeResolution::Full,
    }
}

impl JpxCodec {
    /// Construct a `/JPXDecode` codec with the default 128 MiB shared
    /// coefficient buffer pool.
    pub fn new() -> Self {
        Self::with_buffer_pool_bytes(DEFAULT_JPX_BUFFER_POOL_BYTES)
    }

    /// Construct a `/JPXDecode` codec whose shared coefficient buffer pool
    /// retains at most `cap_bytes` when idle (`0` disables pooling).
    pub fn with_buffer_pool_bytes(cap_bytes: usize) -> Self {
        Self {
            pool: Arc::new(Jp2BufferPool::new(cap_bytes)),
        }
    }

    /// Bytes of idle coefficient plane capacity currently parked in this
    /// codec's shared pool.
    pub fn idle_buffer_bytes(&self) -> usize {
        self.pool.idle_bytes()
    }

    /// Decode straight into a packed 8-bit raster (the fast path for 8-bit
    /// components).
    #[allow(clippy::too_many_arguments)]
    fn decode_packed(
        &self,
        prepared: &PreparedJp2<'_>,
        resolution: DecodeResolution,
        format: DecodedFormat,
        channels: usize,
        width: u32,
        height: u32,
        limits: &DecodeLimits,
        concurrency: DecodeConcurrency,
        ignore_container_palette: bool,
    ) -> Result<DecodedImage, ImageError> {
        let output = match format {
            DecodedFormat::Gray8 => DecodeOutputFormat::Gray8,
            DecodedFormat::GrayA8 | DecodedFormat::Multi2 => DecodeOutputFormat::GrayA8,
            DecodedFormat::Rgb8 => DecodeOutputFormat::Rgb8,
            DecodedFormat::Rgba8 => DecodeOutputFormat::Rgba8,
            DecodedFormat::Cmyk8 => DecodeOutputFormat::Cmyk8,
            other => {
                return Err(ImageError::Decode(format!(
                    "JPX: packed output not applicable to {other:?}"
                )));
            }
        };
        let request = DecodeRequest {
            resolution,
            output,
            region: None,
            concurrency,
            ignore_container_palette,
            limits: decoder_limits(limits),
        };
        let stride = (width as usize)
            .checked_mul(channels)
            .ok_or_else(|| ImageError::Decode("JPX: stride overflow".into()))?;
        let bytes = stride
            .checked_mul(height as usize)
            .ok_or_else(|| ImageError::Decode("JPX: size overflow".into()))?;
        if bytes as u64 > limits.max_output_bytes {
            return Err(ImageError::TooLarge { width, height });
        }
        let mut output_data = zeroed_arc(bytes);
        let Some(output_slice) = Arc::get_mut(&mut output_data) else {
            return Err(ImageError::Decode(
                "JPX: output allocation unexpectedly became shared".into(),
            ));
        };
        let check_cancel = || limits.is_cancelled();
        let info = JPX_DECODER
            .with(|decoder| {
                let mut decoder = decoder.borrow_mut();
                let guard = ScopedDecoderPool::attach(&mut decoder, Arc::clone(&self.pool));
                guard.decoder.decode_prepared_into_cancellable(
                    prepared,
                    &request,
                    DecodeTarget {
                        data: output_slice,
                        width,
                        height,
                        stride,
                        format: output,
                        premultiplied: false,
                    },
                    &check_cancel,
                )
            })
            .map_err(|e| {
                if e.is_cancelled() {
                    ImageError::Cancelled
                } else {
                    ImageError::Decode(format!("JPX: {e}"))
                }
            })?;
        if limits.is_cancelled() {
            return Err(ImageError::Cancelled);
        }
        Ok(DecodedImage {
            width: info.width,
            height: info.height,
            format,
            stride: info.stride,
            data: output_data,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_native_interleaved(
        &self,
        prepared: &PreparedJp2<'_>,
        resolution: DecodeResolution,
        format: DecodedFormat,
        channels: usize,
        _width: u32,
        _height: u32,
        limits: &DecodeLimits,
        concurrency: DecodeConcurrency,
        ignore_container_palette: bool,
    ) -> Result<DecodedImage, ImageError> {
        let request = DecodeRequest {
            resolution,
            output: DecodeOutputFormat::NativePlanarI32,
            region: None,
            concurrency,
            ignore_container_palette,
            limits: decoder_limits(limits),
        };
        let check_cancel = || limits.is_cancelled();
        let result = JPX_DECODER
            .with(|decoder| {
                let mut decoder = decoder.borrow_mut();
                let guard = ScopedDecoderPool::attach(&mut decoder, Arc::clone(&self.pool));
                guard
                    .decoder
                    .decode_prepared_cancellable(prepared, &request, &check_cancel)
            })
            .map_err(|e| {
                if e.is_cancelled() {
                    ImageError::Cancelled
                } else {
                    ImageError::Decode(format!("JPX: {e}"))
                }
            })?;
        if limits.is_cancelled() {
            return Err(ImageError::Cancelled);
        }
        let image = match result {
            DecodeResult::Native(image) => image,
            DecodeResult::Raster(_) => {
                return Err(ImageError::Decode(
                    "JPX: native decode returned a packed raster".into(),
                ));
            }
        };

        // The decoded dimensions win over the container header (a reduced-
        // resolution decode returns a smaller image).
        let width = image.width;
        let height = image.height;
        if width == 0 || height == 0 {
            return Err(ImageError::Decode("JPX: zero dimension".into()));
        }
        let ncomp = image.components.len();
        if ncomp != channels {
            return Err(ImageError::Decode(format!(
                "JPX: {format:?} expects {channels} components, decoded {ncomp}"
            )));
        }
        let stride = (width as usize)
            .checked_mul(ncomp)
            .ok_or_else(|| ImageError::Decode("JPX: stride overflow".into()))?;
        let bytes = stride
            .checked_mul(height as usize)
            .ok_or_else(|| ImageError::Decode("JPX: size overflow".into()))?;
        if bytes as u64 > limits.max_output_bytes {
            return Err(ImageError::TooLarge { width, height });
        }

        // jp2lam yields planar i32 samples at each component's own precision;
        // interleave and scale to 8-bit.
        let pixels = (width as usize)
            .checked_mul(height as usize)
            .ok_or_else(|| ImageError::Decode("JPX: pixel count overflow".into()))?;
        let mut out = zeroed_arc(bytes);
        let Some(out_data) = Arc::get_mut(&mut out) else {
            return Err(ImageError::Decode(
                "JPX: native output allocation unexpectedly became shared".into(),
            ));
        };
        // When the PDF declares `/Indexed`, ignoring a JP2 container palette
        // leaves the codestream's samples as literal PDF palette indices. A
        // 4-bit index 15 must remain byte value 15 in the decoded Gray8
        // carrier; widening it to 240 makes the renderer clamp nearly every
        // sample to `/HiVal` and collapses the image to one palette color
        // (pdf.js issue12213).
        let preserve_palette_indices = ignore_container_palette && ncomp == 1;
        for (ci, comp) in image.components.iter().enumerate() {
            if comp.data.len() < pixels {
                return Err(ImageError::Decode("JPX: short component plane".into()));
            }
            let shift = comp.precision.saturating_sub(8);
            // Sub-8-bit components (bitonal scans are 1-bit) widen by a LEFT
            // shift, matching PDFium's `src << (8 - prec)` in
            // `CJpx_Decoder::Decode`. This is deliberately *not* the full-range
            // scaling OpenJPEG's writers use: at 1 bit a set sample becomes 128,
            // not 255, so a bitonal scan renders on a mid-grey field. Widening
            // to full range instead leaves pdfbox/3246 at inkΔ 0.81/0.99 and
            // pdfbox/4326 at 0.98 against PDFium; the shift puts all three at
            // exactly 0.00000. Codec correctness is unaffected either way —
            // jp2lam's samples are bit-exact with OpenJPEG (max abs diff 0) —
            // this is only how the renderer widens them into an 8-bit buffer,
            // and it is the same convention the deeper-than-8-bit branch below
            // already follows. Flip both branches together if PDFium parity is
            // ever traded for spec-literal colour mapping.
            let max_sample = (1u32 << comp.precision.clamp(1, 31)) - 1;
            for y in 0..height as usize {
                if y & 31 == 0 && limits.is_cancelled() {
                    return Err(ImageError::Cancelled);
                }
                let row = y * width as usize;
                for x in 0..width as usize {
                    let i = row + x;
                    out_data[i * ncomp + ci] = component_sample_to_u8(
                        comp.data[i],
                        comp.precision,
                        max_sample,
                        shift,
                        preserve_palette_indices,
                    );
                }
            }
        }

        Ok(DecodedImage {
            width,
            height,
            format,
            stride,
            data: out,
        })
    }
}

/// Allocate the renderer's final shared JPX output directly. The allocation is
/// uniquely owned while jp2lam fills it, then handed to `DecodedImage` unchanged.
#[allow(
    unsafe_code,
    reason = "Arc::new_zeroed_slice hands back MaybeUninit; see SAFETY comment"
)]
fn zeroed_arc(len: usize) -> Arc<[u8]> {
    let data = Arc::<[u8]>::new_zeroed_slice(len);
    // SAFETY: `new_zeroed_slice` initialized every `u8` to a valid zero value.
    unsafe { data.assume_init() }
}

fn decoder_limits(limits: &DecodeLimits) -> Jp2DecodeLimits {
    let defaults = Jp2DecodeLimits::default();
    Jp2DecodeLimits {
        max_input_bytes: defaults
            .max_input_bytes
            .min(usize::try_from(limits.max_input_bytes).unwrap_or(usize::MAX)),
        max_pixels: limits.max_pixels,
        max_working_bytes: defaults
            .max_working_bytes
            .min(usize::try_from(limits.max_working_bytes).unwrap_or(usize::MAX)),
        ..defaults
    }
}

fn component_sample_to_u8(
    value: i32,
    precision: u32,
    max_sample: u32,
    right_shift: u32,
    preserve_palette_index: bool,
) -> u8 {
    if preserve_palette_index {
        return value.clamp(0, max_sample.min(255) as i32) as u8;
    }
    if precision < 8 {
        ((value.clamp(0, max_sample as i32) as u32) << (8 - precision)).min(255) as u8
    } else {
        let value = if right_shift > 0 {
            value >> right_shift
        } else {
            value
        };
        value.clamp(0, 255) as u8
    }
}

#[cfg(test)]
mod tests {
    use super::{component_sample_to_u8, decoder_limits, load_aware_budget, parse_jpx_concurrency};
    use crate::image::codec::DecodeLimits;
    use jp2lam::DecodeConcurrency;

    #[test]
    fn jpx_concurrency_override_accepts_bare_counts() {
        assert_eq!(
            parse_jpx_concurrency("serial"),
            Some(DecodeConcurrency::Serial)
        );
        assert_eq!(
            parse_jpx_concurrency(" 1 "),
            Some(DecodeConcurrency::Serial)
        );
        assert_eq!(
            parse_jpx_concurrency("4"),
            Some(DecodeConcurrency::Budgeted(4))
        );
        assert_eq!(
            parse_jpx_concurrency("budgeted:1"),
            Some(DecodeConcurrency::Budgeted(1))
        );
        assert_eq!(
            parse_jpx_concurrency("Budgeted:3"),
            Some(DecodeConcurrency::Budgeted(3))
        );
        assert_eq!(parse_jpx_concurrency("0"), None);
        assert_eq!(parse_jpx_concurrency("budgeted:0"), None);
        assert_eq!(parse_jpx_concurrency("auto"), None);
        assert_eq!(parse_jpx_concurrency("fast"), None);
    }

    #[test]
    fn saturated_load_can_fall_back_to_one_decode_worker() {
        assert_eq!(
            load_aware_budget(usize::MAX),
            DecodeConcurrency::Budgeted(1)
        );
    }

    #[test]
    fn sub8_pdf_palette_indices_are_not_color_widened() {
        assert_eq!(component_sample_to_u8(15, 4, 15, 0, true), 15);
        assert_eq!(
            component_sample_to_u8(15, 4, 15, 0, false),
            240,
            "ordinary 4-bit color components retain PDFium's widening"
        );
    }

    #[test]
    fn renderer_limits_are_pushed_into_jp2_decode() {
        let renderer = DecodeLimits {
            max_input_bytes: 2048,
            max_pixels: 1234,
            max_output_bytes: 4096,
            max_working_bytes: 3072,
            ..DecodeLimits::default()
        };
        let mapped = decoder_limits(&renderer);
        assert_eq!(mapped.max_pixels, 1234);
        assert_eq!(mapped.max_input_bytes, 2048);
        assert_eq!(mapped.max_working_bytes, 3072);
    }
}
