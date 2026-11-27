//! C ABI for the `jbig2enc-rust` JBIG2 encoder and decoder.
//!
//! This is the binding every non-Rust consumer goes through except Python,
//! which has its own PyO3 crate next to this one. C and C++ use it directly;
//! C# (`LibraryImport`), Java 22+ (`jextract`), Swift, Go and Zig generate
//! their bindings from `include/jbig2rs.h` with no further Rust.
//!
//! It binds the crate's binding-neutral facade (`BilevelImage`,
//! `EncodeOptions`, `DocumentEncoder`, `PdfGroupEncoder`, the decoder) and
//! nothing deeper. No codec logic lives here: every function translates
//! arguments, calls one facade method, and translates the result.
//!
//! # The rules this file follows
//!
//! * **Symbols are prefixed `jbig2rs_`.** jbig2dec exports `jbig2_*`, and
//!   hosts such as Ghostscript and MuPDF link it; a shared prefix would collide.
//! * **Nothing crosses the boundary but plain data and opaque handles.** No
//!   core Rust type is `#[repr(C)]`. The two input structs are FFI-owned:
//!   [`Jbig2rsImage`] is frozen for ABI 1, and the options structs start
//!   with `struct_size` so that later versions can append fields.
//! * **Every entry point catches unwinds.** A panic crossing an `extern "C"`
//!   frame is undefined behaviour. The guard only works because every
//!   profile is `panic = "unwind"` (see the workspace manifest). A handle that
//!   panicked mid-call is poisoned and refuses further work.
//! * **Errors are a status code plus a thread-local message**
//!   ([`jbig2rs_last_error_message`]), never a Rust `String`.
//! * **Every allocation is freed by the library that made it.** Calling C
//!   `free()` on one is undefined behaviour.
//!
//! # Threading
//!
//! Encoder handles may move between threads but must not be used by two at
//! once. Distinct handles are independent. A [`Jbig2rsGlobals`] is immutable
//! once decoded and may be shared by any number of threads decoding pages.

// A C ABI is `no_mangle` plus raw pointers by definition. Each `unsafe`
// block still states the invariant it relies on.
#![allow(unsafe_code)]

use std::cell::RefCell;
use std::ffi::{CString, c_char, c_int, c_void};
use std::mem::offset_of;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::OnceLock;

use jbig2enc_rust::decode::{
    DecodeError, DecodeErrorKind, DecodeOptions, DecodedDocument, DecodedGlobals, DecoderContext,
    decode_embedded_with_globals, decode_file, decode_globals,
};
use jbig2enc_rust::{
    BilevelImage, DocumentEncoder, EncodeOptions, EncodedPdfGroup, ErrorKind, Jbig2Error,
    MonoBitmap, PdfGroupEncoder, PdfGroupOptions, PdfSplitOutput, PixelLayout, Polarity, Preset,
    ResourceLimitExceeded,
};

// ── Versions and status codes ───────────────────────────────────────────────

/// ABI major version. Bumped only for an incompatible change.
pub const JBIG2RS_ABI_MAJOR: u32 = 1;
/// ABI minor version. Bumped when functions or appended option fields arrive.
pub const JBIG2RS_ABI_MINOR: u32 = 0;

/// Call succeeded.
pub const JBIG2RS_OK: c_int = 0;
/// A required pointer argument was null.
pub const JBIG2RS_ERR_NULL_ARGUMENT: c_int = -1;
/// An argument was out of range: image dimensions, stride, buffer length, an
/// unknown format, polarity or preset, a page index.
pub const JBIG2RS_ERR_INVALID_ARGUMENT: c_int = -2;
/// The options describe a configuration the encoder refuses.
pub const JBIG2RS_ERR_INVALID_CONFIG: c_int = -3;
/// The output would exceed a configured resource budget.
pub const JBIG2RS_ERR_RESOURCE_LIMIT: c_int = -4;
/// Encoding failed.
pub const JBIG2RS_ERR_ENCODE: c_int = -5;
/// The input is not a well-formed JBIG2 stream.
pub const JBIG2RS_ERR_DECODE_MALFORMED: c_int = -6;
/// The stream is well formed but uses a feature the decoder lacks.
pub const JBIG2RS_ERR_DECODE_UNSUPPORTED: c_int = -7;
/// Decoding would exceed the decode limits.
pub const JBIG2RS_ERR_DECODE_LIMIT: c_int = -8;
/// The handle is finished, failed earlier, or was poisoned by a panic.
pub const JBIG2RS_ERR_BAD_STATE: c_int = -9;
/// A group sink returned non-zero; the group encoder stops.
pub const JBIG2RS_ERR_ABORTED: c_int = -10;
/// The library panicked. It remains usable; the handle involved does not.
pub const JBIG2RS_ERR_PANIC: c_int = -11;

/// [`Jbig2rsImage::format`]: one bit per pixel, MSB first.
pub const JBIG2RS_FORMAT_PACKED1: u32 = 0;
/// [`Jbig2rsImage::format`]: one byte per pixel.
pub const JBIG2RS_FORMAT_BYTES: u32 = 1;
/// [`Jbig2rsImage::format`]: 8-bit grayscale, black below `threshold`.
pub const JBIG2RS_FORMAT_GRAY8: u32 = 2;

/// A set bit (non-zero byte) is black: PBM, Leptonica, JBIG2 itself.
pub const JBIG2RS_ONE_IS_BLACK: u32 = 0;
/// A clear bit (zero byte) is black: TIFF `MinIsBlack`, many 1bpp DIBs.
pub const JBIG2RS_ZERO_IS_BLACK: u32 = 1;

/// Symbol coding with the encoder's defaults.
pub const JBIG2RS_PRESET_DEFAULT: u32 = 0;
/// Symbol coding tuned for text.
pub const JBIG2RS_PRESET_TEXT: u32 = 1;
/// One generic region per page: bit-exact.
pub const JBIG2RS_PRESET_GENERIC: u32 = 2;
/// Generic region without duplicate-line removal or dictionary: bit-exact,
/// and the safest output for viewers that mishandle `JBIG2Globals`.
pub const JBIG2RS_PRESET_LOSSLESS: u32 = 3;
/// Pixel-exact symbol coding.
pub const JBIG2RS_PRESET_TEXT_LOSSLESS: u32 = 4;
/// Slower pixel-exact symbol coding that measures its alternatives.
pub const JBIG2RS_PRESET_TEXT_LOSSLESS_MAX_COMPRESSION: u32 = 5;
/// Lossy glyph unification.
pub const JBIG2RS_PRESET_TEXT_SYMBOL_UNIFY: u32 = 6;
/// Lossy glyph unification with a topology guard on every substitution.
pub const JBIG2RS_PRESET_TEXT_SYMBOL_UNIFY_SAFE: u32 = 7;

// ── Error reporting ─────────────────────────────────────────────────────────

thread_local! {
    /// Last error on *this* thread, so concurrent callers cannot overwrite
    /// each other's message.
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

fn set_last_error(message: impl Into<Vec<u8>>) {
    let value =
        CString::new(message).unwrap_or_else(|_| c"error message contained a NUL byte".to_owned());
    LAST_ERROR.with(|slot| *slot.borrow_mut() = Some(value));
}

fn clear_last_error() {
    LAST_ERROR.with(|slot| *slot.borrow_mut() = None);
}

/// Message for the most recent failure **on the calling thread**, or `NULL`
/// if the last call succeeded.
///
/// The string is owned by the library and stays valid until the next call
/// into the library on this thread. Copy it to keep it; do not free it.
#[unsafe(no_mangle)]
pub extern "C" fn jbig2rs_last_error_message() -> *const c_char {
    LAST_ERROR.with(|slot| {
        slot.borrow()
            .as_ref()
            .map_or(std::ptr::null(), |message| message.as_ptr())
    })
}

/// A failed call: the status code to return and the message to record.
struct Failure(c_int, String);

impl Failure {
    fn null(what: &str) -> Self {
        Failure(
            JBIG2RS_ERR_NULL_ARGUMENT,
            format!("{what} must not be NULL"),
        )
    }

    fn invalid(message: impl Into<String>) -> Self {
        Failure(JBIG2RS_ERR_INVALID_ARGUMENT, message.into())
    }

    fn bad_state(message: impl Into<String>) -> Self {
        Failure(JBIG2RS_ERR_BAD_STATE, message.into())
    }
}

impl From<Jbig2Error> for Failure {
    fn from(error: Jbig2Error) -> Self {
        let status = match error.kind() {
            ErrorKind::InvalidInput => JBIG2RS_ERR_INVALID_ARGUMENT,
            ErrorKind::InvalidConfig => JBIG2RS_ERR_INVALID_CONFIG,
            ErrorKind::ResourceLimit => JBIG2RS_ERR_RESOURCE_LIMIT,
            _ => JBIG2RS_ERR_ENCODE,
        };
        Failure(status, error.to_string())
    }
}

impl From<DecodeError> for Failure {
    fn from(error: DecodeError) -> Self {
        let status = match error.kind() {
            DecodeErrorKind::Limit => JBIG2RS_ERR_DECODE_LIMIT,
            DecodeErrorKind::Unsupported => JBIG2RS_ERR_DECODE_UNSUPPORTED,
            _ => JBIG2RS_ERR_DECODE_MALFORMED,
        };
        Failure(status, format!("decoding JBIG2: {error}"))
    }
}

/// The group encoder works in `anyhow`; recover the class of what failed.
impl From<anyhow::Error> for Failure {
    fn from(error: anyhow::Error) -> Self {
        if error.downcast_ref::<SinkAborted>().is_some() {
            return Failure(JBIG2RS_ERR_ABORTED, error.to_string());
        }
        if error.downcast_ref::<ResourceLimitExceeded>().is_some() {
            return Failure(JBIG2RS_ERR_RESOURCE_LIMIT, error.to_string());
        }
        match error.downcast::<Jbig2Error>() {
            Ok(error) => error.into(),
            Err(error) => Failure(JBIG2RS_ERR_ENCODE, format!("{error:#}")),
        }
    }
}

/// Run `body` behind an unwind guard, recording any failure.
///
/// Every exported function that can fail goes through here. A panic that
/// escapes into C is undefined behaviour, and the decoder parses untrusted
/// input, so this is load-bearing rather than defensive.
fn guarded(body: impl FnOnce() -> Result<(), Failure>) -> c_int {
    clear_last_error();
    match catch_unwind(AssertUnwindSafe(body)) {
        Ok(Ok(())) => JBIG2RS_OK,
        Ok(Err(Failure(status, message))) => {
            set_last_error(message);
            status
        }
        Err(payload) => {
            let detail = payload
                .downcast_ref::<&'static str>()
                .map(|text| (*text).to_owned())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unknown payload".to_owned());
            set_last_error(format!("jbig2rs panicked: {detail}"));
            JBIG2RS_ERR_PANIC
        }
    }
}

/// Borrow `len` bytes at `data`, which may be NULL only when `len` is 0.
///
/// # Safety
/// A non-null `data` must point to `len` readable bytes that stay valid and
/// unmodified for the returned lifetime.
unsafe fn borrow_bytes<'a>(data: *const u8, len: usize, what: &str) -> Result<&'a [u8], Failure> {
    if len == 0 {
        return Ok(&[]);
    }
    if data.is_null() {
        return Err(Failure::null(what));
    }
    // SAFETY: non-null, and the caller guarantees `len` readable bytes.
    Ok(unsafe { std::slice::from_raw_parts(data, len) })
}

/// Dereference an input pointer argument.
///
/// # Safety
/// A non-null `pointer` must point to a valid `T` for the returned lifetime.
unsafe fn arg<'a, T>(pointer: *const T, what: &str) -> Result<&'a T, Failure> {
    // SAFETY: the caller guarantees a non-null pointer is valid.
    unsafe { pointer.as_ref() }.ok_or_else(|| Failure::null(what))
}

/// Dereference a mutable pointer argument (an output slot or a handle).
///
/// # Safety
/// A non-null `pointer` must point to a valid, unaliased `T`.
unsafe fn arg_mut<'a, T>(pointer: *mut T, what: &str) -> Result<&'a mut T, Failure> {
    // SAFETY: the caller guarantees a non-null pointer is valid and unaliased.
    unsafe { pointer.as_mut() }.ok_or_else(|| Failure::null(what))
}

// ── Library metadata ────────────────────────────────────────────────────────

/// The codec version, e.g. `"0.7.0"`. Static; do not free.
#[unsafe(no_mangle)]
pub extern "C" fn jbig2rs_version() -> *const c_char {
    static VERSION: OnceLock<CString> = OnceLock::new();
    VERSION
        .get_or_init(|| {
            CString::new(jbig2enc_rust::VERSION).unwrap_or_else(|_| c"unknown".to_owned())
        })
        .as_ptr()
}

/// The ABI version this library implements: `major << 16 | minor`. A caller
/// built against header ABI `M.m` works with any library reporting major `M`
/// and minor `>= m`.
#[unsafe(no_mangle)]
pub extern "C" fn jbig2rs_abi_version() -> u32 {
    (JBIG2RS_ABI_MAJOR << 16) | JBIG2RS_ABI_MINOR
}

// ── Buffers ─────────────────────────────────────────────────────────────────

/// An owned byte buffer produced by this library. Free it with
/// [`jbig2rs_buffer_free`] and nothing else.
#[repr(C)]
#[derive(Debug)]
pub struct Jbig2rsBuffer {
    pub data: *mut u8,
    pub len: usize,
}

impl Jbig2rsBuffer {
    const EMPTY: Self = Self {
        data: std::ptr::null_mut(),
        len: 0,
    };

    fn from_vec(bytes: Vec<u8>) -> Self {
        if bytes.is_empty() {
            return Self::EMPTY;
        }
        // A boxed slice's allocation is exactly `len` bytes, so `len` alone
        // is enough to rebuild and free it.
        let len = bytes.len();
        let data = Box::into_raw(bytes.into_boxed_slice()).cast::<u8>();
        Self { data, len }
    }
}

/// Release a buffer produced by this library and reset it to empty, so a
/// second call is harmless. NULL, or an empty buffer, is a no-op.
///
/// # Safety
/// `buffer` must be NULL or point to a `Jbig2rsBuffer` this library filled
/// in and that has not been freed since.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jbig2rs_buffer_free(buffer: *mut Jbig2rsBuffer) {
    // SAFETY: the caller guarantees `buffer` is NULL or one of ours.
    let Some(slot) = (unsafe { buffer.as_mut() }) else {
        return;
    };
    if !slot.data.is_null() {
        let slice = std::ptr::slice_from_raw_parts_mut(slot.data, slot.len);
        // SAFETY: `data`/`len` came from `Box<[u8]>::into_raw` in `from_vec`.
        drop(unsafe { Box::from_raw(slice) });
    }
    *slot = Jbig2rsBuffer::EMPTY;
}

/// Store `bytes` in an output slot, freeing nothing: the slot must be fresh.
fn emit(slot: &mut Jbig2rsBuffer, bytes: Vec<u8>) {
    *slot = Jbig2rsBuffer::from_vec(bytes);
}

// ── Images ──────────────────────────────────────────────────────────────────

/// An image in the caller's memory. Frozen for ABI 1: fields are never
/// added, removed or reordered.
///
/// Rows are top-down and `stride` bytes apart. `data` must hold
/// `stride * (height - 1) + row_bytes` bytes, where `row_bytes` is
/// `ceil(width / 8)` for `JBIG2RS_FORMAT_PACKED1` and `width` otherwise. The
/// bytes are read during the call only, never retained.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Jbig2rsImage {
    pub width: u32,
    pub height: u32,
    pub stride: usize,
    pub data: *const u8,
    /// `JBIG2RS_FORMAT_*`.
    pub format: u32,
    /// `JBIG2RS_ONE_IS_BLACK` or `JBIG2RS_ZERO_IS_BLACK`; ignored for GRAY8.
    pub polarity: u32,
    /// GRAY8 only: a pixel is black when its value is below this.
    pub threshold: u8,
}

fn polarity(value: u32) -> Result<Polarity, Failure> {
    match value {
        JBIG2RS_ONE_IS_BLACK => Ok(Polarity::OneIsBlack),
        JBIG2RS_ZERO_IS_BLACK => Ok(Polarity::ZeroIsBlack),
        other => Err(Failure::invalid(format!("unknown polarity {other}"))),
    }
}

/// Validate a caller's image descriptor and borrow its pixels.
///
/// # Safety
/// `image.data` must point to the bytes the descriptor describes.
unsafe fn bilevel(image: &Jbig2rsImage) -> Result<BilevelImage<'_>, Failure> {
    let stride = image.stride;
    let (layout, polarity) = match image.format {
        JBIG2RS_FORMAT_PACKED1 => (PixelLayout::Packed1 { stride }, polarity(image.polarity)?),
        JBIG2RS_FORMAT_BYTES => (PixelLayout::Bytes { stride }, polarity(image.polarity)?),
        JBIG2RS_FORMAT_GRAY8 => (
            PixelLayout::Gray8 {
                stride,
                threshold: image.threshold,
            },
            Polarity::ZeroIsBlack,
        ),
        other => return Err(Failure::invalid(format!("unknown image format {other}"))),
    };
    if image.width == 0 || image.height == 0 {
        return Err(Failure::invalid("width and height must both be at least 1"));
    }
    // Borrow exactly the bytes the descriptor claims, and only once it is
    // known to be self-consistent.
    let row_bytes = layout.row_bytes(image.width);
    if stride < row_bytes {
        return Err(Failure::invalid(format!(
            "stride {stride} is shorter than a {}-pixel row ({row_bytes} bytes)",
            image.width
        )));
    }
    let len = stride
        .checked_mul(image.height as usize - 1)
        .and_then(|bytes| bytes.checked_add(row_bytes))
        .ok_or_else(|| Failure::invalid("image size overflows the address space"))?;
    // SAFETY: the caller guarantees `data` holds the image it describes,
    // which is exactly `len` bytes.
    let data = unsafe { borrow_bytes(image.data, len, "image->data") }?;
    Ok(BilevelImage::new(
        image.width,
        image.height,
        data,
        layout,
        polarity,
    )?)
}

// ── Encode options ──────────────────────────────────────────────────────────

/// Encoder options. Initialise with [`jbig2rs_encode_options_init`], which
/// sets `struct_size`; later ABI minors may append fields, and the library
/// reads only the fields `struct_size` covers.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Jbig2rsEncodeOptions {
    /// `sizeof(Jbig2rsEncodeOptions)` as the caller compiled it.
    pub struct_size: u32,
    /// `JBIG2RS_PRESET_*`.
    pub preset: u32,
    /// Refinement coding: -1 keeps the preset's choice, 0 turns it off, 1
    /// turns it on (symbol-coding presets only).
    pub refine: i32,
}

/// Fill `options` with the defaults: `JBIG2RS_PRESET_DEFAULT`, refinement
/// as the preset has it. NULL is a no-op.
///
/// # Safety
/// `options` must be NULL or point to writable `Jbig2rsEncodeOptions`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jbig2rs_encode_options_init(options: *mut Jbig2rsEncodeOptions) {
    // SAFETY: the caller guarantees `options` is NULL or writable.
    if let Some(options) = unsafe { options.as_mut() } {
        *options = Jbig2rsEncodeOptions {
            struct_size: size_of::<Jbig2rsEncodeOptions>() as u32,
            preset: JBIG2RS_PRESET_DEFAULT,
            refine: -1,
        };
    }
}

/// Whether a caller's `struct_size` covers the field at `offset` of `size`.
fn covers(struct_size: u32, offset: usize, size: usize) -> bool {
    struct_size as usize >= offset + size
}

fn preset(value: u32) -> Result<Preset, Failure> {
    Ok(match value {
        JBIG2RS_PRESET_DEFAULT => Preset::Default,
        JBIG2RS_PRESET_TEXT => Preset::Text,
        JBIG2RS_PRESET_GENERIC => Preset::Generic,
        JBIG2RS_PRESET_LOSSLESS => Preset::Lossless,
        JBIG2RS_PRESET_TEXT_LOSSLESS => Preset::TextLossless,
        JBIG2RS_PRESET_TEXT_LOSSLESS_MAX_COMPRESSION => Preset::TextLosslessMaxCompression,
        JBIG2RS_PRESET_TEXT_SYMBOL_UNIFY => Preset::TextSymbolUnify,
        JBIG2RS_PRESET_TEXT_SYMBOL_UNIFY_SAFE => Preset::TextSymbolUnifySafe,
        other => return Err(Failure::invalid(format!("unknown preset {other}"))),
    })
}

/// Read caller options; NULL means the defaults.
///
/// # Safety
/// A non-null `options` must point to at least `options->struct_size` bytes.
unsafe fn encode_options(options: *const Jbig2rsEncodeOptions) -> Result<EncodeOptions, Failure> {
    let mut result = EncodeOptions::default();
    if options.is_null() {
        return Ok(result);
    }
    // SAFETY: `struct_size` is the first field and every version has it; the
    // caller guarantees the pointer is valid for at least that much.
    let struct_size = unsafe { options.cast::<u32>().read_unaligned() };
    if !covers(struct_size, 0, size_of::<u32>()) {
        return Err(Failure::invalid(
            "options->struct_size is too small; call jbig2rs_encode_options_init",
        ));
    }
    let base = options.cast::<u8>();
    // SAFETY (both reads): `covers` checks each field lies within the
    // `struct_size` bytes the caller vouched for.
    if covers(
        struct_size,
        offset_of!(Jbig2rsEncodeOptions, preset),
        size_of::<u32>(),
    ) {
        let value = unsafe {
            base.add(offset_of!(Jbig2rsEncodeOptions, preset))
                .cast::<u32>()
                .read_unaligned()
        };
        result.preset = preset(value)?;
    }
    if covers(
        struct_size,
        offset_of!(Jbig2rsEncodeOptions, refine),
        size_of::<i32>(),
    ) {
        let value = unsafe {
            base.add(offset_of!(Jbig2rsEncodeOptions, refine))
                .cast::<i32>()
                .read_unaligned()
        };
        result.refine = match value {
            -1 => None,
            0 => Some(false),
            1 => Some(true),
            other => {
                return Err(Failure::invalid(format!(
                    "refine must be -1, 0 or 1, got {other}"
                )));
            }
        };
    }
    Ok(result)
}

// ── One-shot encoding ───────────────────────────────────────────────────────

/// Encode one image as a standalone JBIG2 file (with file header) into `out`.
/// `options` may be NULL for the defaults. On failure `out` is left empty.
///
/// # Safety
/// `image` must describe readable memory; `options` must be NULL or valid;
/// `out` must point to a writable `Jbig2rsBuffer`, which is overwritten
/// without being freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jbig2rs_encode_file(
    image: *const Jbig2rsImage,
    options: *const Jbig2rsEncodeOptions,
    out: *mut Jbig2rsBuffer,
) -> c_int {
    guarded(|| {
        // SAFETY: argument validity is the caller's contract above.
        let out = unsafe { arg_mut(out, "out") }?;
        *out = Jbig2rsBuffer::EMPTY;
        let image = unsafe { bilevel(arg(image, "image")?) }?;
        let options = unsafe { encode_options(options) }?;
        let mut encoder = DocumentEncoder::from_options(&options)?;
        encoder.add_page(&image)?;
        emit(out, encoder.finish_file()?);
        Ok(())
    })
}

/// Encode one image as PDF fragments: `globals` receives the shared symbol
/// dictionary for the image's `/JBIG2Globals` stream (left empty when the
/// encoder made none), and `page` receives the image stream itself.
///
/// # Safety
/// As [`jbig2rs_encode_file`]; `globals` and `page` must both be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jbig2rs_encode_pdf(
    image: *const Jbig2rsImage,
    options: *const Jbig2rsEncodeOptions,
    globals: *mut Jbig2rsBuffer,
    page: *mut Jbig2rsBuffer,
) -> c_int {
    guarded(|| {
        // SAFETY: argument validity is the caller's contract above.
        let globals = unsafe { arg_mut(globals, "globals") }?;
        let page = unsafe { arg_mut(page, "page") }?;
        *globals = Jbig2rsBuffer::EMPTY;
        *page = Jbig2rsBuffer::EMPTY;
        let image = unsafe { bilevel(arg(image, "image")?) }?;
        let options = unsafe { encode_options(options) }?;
        let mut encoder = DocumentEncoder::from_options(&options)?;
        encoder.add_page(&image)?;
        let output = encoder.finish_pdf()?;
        let mut streams = output.page_streams.into_iter();
        let (Some(stream), None) = (streams.next(), streams.next()) else {
            return Err(Failure(
                JBIG2RS_ERR_ENCODE,
                "the encoder did not produce exactly one page stream".into(),
            ));
        };
        emit(globals, output.global_segments.unwrap_or_default());
        emit(page, stream);
        Ok(())
    })
}

// ── Multi-page encoding ─────────────────────────────────────────────────────

/// A multi-page encoder: every page shares one symbol dictionary. Opaque.
#[derive(Debug)]
pub struct Jbig2rsEncoder {
    /// `None` once finished.
    inner: Option<DocumentEncoder>,
    /// Set for the duration of every call and cleared on a normal return, so
    /// a panic mid-call leaves it set.
    poisoned: bool,
}

impl Jbig2rsEncoder {
    fn with<T>(
        &mut self,
        body: impl FnOnce(&mut Option<DocumentEncoder>) -> Result<T, Failure>,
    ) -> Result<T, Failure> {
        if self.poisoned {
            return Err(Failure::bad_state(
                "the encoder was poisoned by an earlier panic",
            ));
        }
        if self.inner.is_none() {
            return Err(Failure::bad_state("the encoder has already been finished"));
        }
        self.poisoned = true;
        let result = body(&mut self.inner);
        self.poisoned = false;
        result
    }
}

/// Create a multi-page encoder. `options` may be NULL for the defaults.
/// Free the handle with [`jbig2rs_encoder_free`], finished or not.
///
/// # Safety
/// `options` must be NULL or valid; `out` must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jbig2rs_encoder_new(
    options: *const Jbig2rsEncodeOptions,
    out: *mut *mut Jbig2rsEncoder,
) -> c_int {
    guarded(|| {
        // SAFETY: argument validity is the caller's contract above.
        let out = unsafe { arg_mut(out, "out") }?;
        *out = std::ptr::null_mut();
        let options = unsafe { encode_options(options) }?;
        let encoder = DocumentEncoder::from_options(&options)?;
        *out = Box::into_raw(Box::new(Jbig2rsEncoder {
            inner: Some(encoder),
            poisoned: false,
        }));
        Ok(())
    })
}

/// Add the next page. The pixels are read during the call only.
///
/// # Safety
/// `encoder` must be a live handle not in use on another thread; `image`
/// must describe readable memory.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jbig2rs_encoder_add_page(
    encoder: *mut Jbig2rsEncoder,
    image: *const Jbig2rsImage,
) -> c_int {
    guarded(|| {
        // SAFETY: argument validity is the caller's contract above.
        let encoder = unsafe { arg_mut(encoder, "encoder") }?;
        let image = unsafe { bilevel(arg(image, "image")?) }?;
        encoder.with(|inner| {
            let inner = inner
                .as_mut()
                .ok_or_else(|| Failure::bad_state("finished"))?;
            Ok(inner.add_page(&image)?)
        })
    })
}

/// The number of pages added so far, or 0 for a NULL or finished handle.
///
/// # Safety
/// `encoder` must be NULL or a live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jbig2rs_encoder_page_count(encoder: *const Jbig2rsEncoder) -> usize {
    // SAFETY: the caller guarantees `encoder` is NULL or live.
    unsafe { encoder.as_ref() }
        .and_then(|encoder| encoder.inner.as_ref())
        .map_or(0, DocumentEncoder::page_count)
}

/// Finish as a standalone JBIG2 file. The handle is spent afterwards (every
/// call but `free` returns `JBIG2RS_ERR_BAD_STATE`), even on failure.
///
/// # Safety
/// `encoder` must be a live handle; `out` must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jbig2rs_encoder_finish_file(
    encoder: *mut Jbig2rsEncoder,
    out: *mut Jbig2rsBuffer,
) -> c_int {
    guarded(|| {
        // SAFETY: argument validity is the caller's contract above.
        let out = unsafe { arg_mut(out, "out") }?;
        *out = Jbig2rsBuffer::EMPTY;
        let encoder = unsafe { arg_mut(encoder, "encoder") }?;
        let bytes = encoder.with(|inner| {
            let inner = inner.take().ok_or_else(|| Failure::bad_state("finished"))?;
            Ok(inner.finish_file()?)
        })?;
        emit(out, bytes);
        Ok(())
    })
}

/// PDF fragments for a document: optional globals plus one stream per page.
/// Opaque; read with the `jbig2rs_pdf_output_*` accessors.
///
/// Never instantiated: a `*Jbig2rsPdfOutput` points at a `PdfSplitOutput`.
/// The opaque marker keeps the Rust type out of the header.
#[derive(Debug)]
pub struct Jbig2rsPdfOutput {
    _opaque: [u8; 0],
}

/// # Safety
/// `output` must be NULL or point at a live `PdfSplitOutput`.
unsafe fn pdf_output<'a>(output: *const Jbig2rsPdfOutput) -> Option<&'a PdfSplitOutput> {
    // SAFETY: per the contract, the pointee is a `PdfSplitOutput`.
    unsafe { output.cast::<PdfSplitOutput>().as_ref() }
}

/// Finish as PDF fragments into a new `*out`, freed with
/// [`jbig2rs_pdf_output_free`]. The encoder handle is spent afterwards.
///
/// # Safety
/// `encoder` must be a live handle; `out` must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jbig2rs_encoder_finish_pdf(
    encoder: *mut Jbig2rsEncoder,
    out: *mut *mut Jbig2rsPdfOutput,
) -> c_int {
    guarded(|| {
        // SAFETY: argument validity is the caller's contract above.
        let out = unsafe { arg_mut(out, "out") }?;
        *out = std::ptr::null_mut();
        let encoder = unsafe { arg_mut(encoder, "encoder") }?;
        let output = encoder.with(|inner| {
            let inner = inner.take().ok_or_else(|| Failure::bad_state("finished"))?;
            Ok(inner.finish_pdf()?)
        })?;
        *out = Box::into_raw(Box::new(output)).cast::<Jbig2rsPdfOutput>();
        Ok(())
    })
}

/// Free an encoder handle. NULL is a no-op.
///
/// # Safety
/// `encoder` must be NULL or a live handle, not used again afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jbig2rs_encoder_free(encoder: *mut Jbig2rsEncoder) {
    if !encoder.is_null() {
        // SAFETY: the handle came from `Box::into_raw` in `jbig2rs_encoder_new`.
        let _ = catch_unwind(AssertUnwindSafe(|| drop(unsafe { Box::from_raw(encoder) })));
    }
}

/// The shared globals stream: `*data`/`*len` borrow from `output` (valid
/// until it is freed), and are NULL/0 when the document has no dictionary.
///
/// # Safety
/// `output` must be a live output; `data` and `len` must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jbig2rs_pdf_output_globals(
    output: *const Jbig2rsPdfOutput,
    data: *mut *const u8,
    len: *mut usize,
) -> c_int {
    guarded(|| {
        // SAFETY: argument validity is the caller's contract above.
        let (data, len) = unsafe { (arg_mut(data, "data")?, arg_mut(len, "len")?) };
        let output = unsafe { pdf_output(output) }.ok_or_else(|| Failure::null("output"))?;
        let bytes = output.global_segments.as_deref().unwrap_or_default();
        *data = if bytes.is_empty() {
            std::ptr::null()
        } else {
            bytes.as_ptr()
        };
        *len = bytes.len();
        Ok(())
    })
}

/// The number of page streams, or 0 for NULL.
///
/// # Safety
/// `output` must be NULL or a live output.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jbig2rs_pdf_output_page_count(output: *const Jbig2rsPdfOutput) -> usize {
    // SAFETY: the caller guarantees `output` is NULL or live.
    unsafe { pdf_output(output) }.map_or(0, |output| output.page_streams.len())
}

/// Page stream `index`: `*data`/`*len` borrow from `output`.
///
/// # Safety
/// As [`jbig2rs_pdf_output_globals`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jbig2rs_pdf_output_page(
    output: *const Jbig2rsPdfOutput,
    index: usize,
    data: *mut *const u8,
    len: *mut usize,
) -> c_int {
    guarded(|| {
        // SAFETY: argument validity is the caller's contract above.
        let (data, len) = unsafe { (arg_mut(data, "data")?, arg_mut(len, "len")?) };
        let output = unsafe { pdf_output(output) }.ok_or_else(|| Failure::null("output"))?;
        let stream = output.page_streams.get(index).ok_or_else(|| {
            Failure::invalid(format!(
                "page {index} out of range ({} pages)",
                output.page_streams.len()
            ))
        })?;
        *data = stream.as_ptr();
        *len = stream.len();
        Ok(())
    })
}

/// Free PDF output. NULL is a no-op. Not for the output a group sink
/// receives, which the library owns.
///
/// # Safety
/// `output` must be NULL or an output from [`jbig2rs_encoder_finish_pdf`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jbig2rs_pdf_output_free(output: *mut Jbig2rsPdfOutput) {
    if !output.is_null() {
        // SAFETY: the output came from `Box::<PdfSplitOutput>::into_raw` in `finish_pdf`.
        let output = output.cast::<PdfSplitOutput>();
        let _ = catch_unwind(AssertUnwindSafe(|| drop(unsafe { Box::from_raw(output) })));
    }
}

// ── Bounded-memory group encoding ───────────────────────────────────────────

/// One finished dictionary group, passed to a [`Jbig2rsGroupSink`]. Valid
/// only during the callback. Opaque: a `*Jbig2rsPdfGroup` points at an
/// `EncodedPdfGroup`.
#[derive(Debug)]
pub struct Jbig2rsPdfGroup {
    _opaque: [u8; 0],
}

/// # Safety
/// `group` must be NULL or the pointer passed to the current sink call.
unsafe fn pdf_group<'a>(group: *const Jbig2rsPdfGroup) -> Option<&'a EncodedPdfGroup> {
    // SAFETY: per the contract, the pointee is an `EncodedPdfGroup`.
    unsafe { group.cast::<EncodedPdfGroup>().as_ref() }
}

/// Receives each finished group. Return 0 to continue; anything else stops
/// the encoder, which then reports `JBIG2RS_ERR_ABORTED`. The callback runs
/// on the thread that called into the library. It must not unwind
/// (C++ exceptions, `longjmp`) through the library.
pub type Jbig2rsGroupSink =
    Option<unsafe extern "C" fn(user: *mut c_void, group: *const Jbig2rsPdfGroup) -> c_int>;

#[derive(Debug)]
struct SinkAborted(c_int);

impl std::fmt::Display for SinkAborted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "the group sink stopped the encoder (returned {})",
            self.0
        )
    }
}

impl std::error::Error for SinkAborted {}

type SinkFn = Box<dyn FnMut(EncodedPdfGroup) -> anyhow::Result<()>>;

/// An encoder that emits a self-contained group (globals plus page streams)
/// every `max_pages` pages or `max_bitmap_bytes` of source bitmap, so memory
/// stays bounded however long the input. Opaque.
pub struct Jbig2rsGroupEncoder {
    inner: Option<PdfGroupEncoder<'static, SinkFn>>,
    /// Set by any failure: group encoder errors are terminal, because the
    /// sink may already have written part of the output.
    failed: bool,
}

impl std::fmt::Debug for Jbig2rsGroupEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Jbig2rsGroupEncoder")
            .field("finished", &self.inner.is_none())
            .field("failed", &self.failed)
            .finish()
    }
}

impl Jbig2rsGroupEncoder {
    fn with(
        &mut self,
        body: impl FnOnce(&mut Option<PdfGroupEncoder<'static, SinkFn>>) -> anyhow::Result<()>,
    ) -> Result<(), Failure> {
        if self.failed {
            return Err(Failure::bad_state("the group encoder failed earlier"));
        }
        if self.inner.is_none() {
            return Err(Failure::bad_state(
                "the group encoder has already been finished",
            ));
        }
        self.failed = true;
        body(&mut self.inner)?;
        self.failed = false;
        Ok(())
    }
}

/// Create a group encoder. `max_pages` must be at least 1;
/// `max_bitmap_bytes` 0 means no byte bound. `sink` is called with each
/// finished group, receiving `user` unchanged. Free the handle with
/// [`jbig2rs_group_encoder_free`].
///
/// # Safety
/// `options` must be NULL or valid; `out` must be writable; `sink` must be
/// safe to call with `user` until the handle is freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jbig2rs_group_encoder_new(
    options: *const Jbig2rsEncodeOptions,
    max_pages: usize,
    max_bitmap_bytes: usize,
    sink: Jbig2rsGroupSink,
    user: *mut c_void,
    out: *mut *mut Jbig2rsGroupEncoder,
) -> c_int {
    guarded(|| {
        // SAFETY: argument validity is the caller's contract above.
        let out = unsafe { arg_mut(out, "out") }?;
        *out = std::ptr::null_mut();
        let sink = sink.ok_or_else(|| Failure::null("sink"))?;
        if max_pages == 0 {
            return Err(Failure::invalid("max_pages must be at least 1"));
        }
        let config = unsafe { encode_options(options) }?.to_config()?;
        let group_options = PdfGroupOptions {
            max_pages,
            max_source_bitmap_bytes: (max_bitmap_bytes != 0).then_some(max_bitmap_bytes),
        };
        let callback: SinkFn = Box::new(move |group: EncodedPdfGroup| {
            let pointer = std::ptr::from_ref(&group).cast::<Jbig2rsPdfGroup>();
            // SAFETY: the caller vouched for `sink` and `user`; `group` lives
            // for the duration of the call.
            let status = unsafe { sink(user, pointer) };
            if status != 0 {
                return Err(SinkAborted(status).into());
            }
            Ok(())
        });
        let encoder = PdfGroupEncoder::owned(config, group_options, callback)?;
        *out = Box::into_raw(Box::new(Jbig2rsGroupEncoder {
            inner: Some(encoder),
            failed: false,
        }));
        Ok(())
    })
}

/// Add the next page; may call the sink with a finished group first.
///
/// # Safety
/// `encoder` must be a live handle not in use elsewhere; `image` must
/// describe readable memory.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jbig2rs_group_encoder_add_page(
    encoder: *mut Jbig2rsGroupEncoder,
    image: *const Jbig2rsImage,
) -> c_int {
    guarded(|| {
        // SAFETY: argument validity is the caller's contract above.
        let encoder = unsafe { arg_mut(encoder, "encoder") }?;
        let image = unsafe { bilevel(arg(image, "image")?) }?;
        encoder.with(|inner| match inner {
            Some(inner) => inner.push_image(&image),
            None => Err(anyhow::anyhow!("finished")),
        })
    })
}

/// Emit the final partial group, if any. The handle is spent afterwards.
///
/// # Safety
/// `encoder` must be a live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jbig2rs_group_encoder_finish(encoder: *mut Jbig2rsGroupEncoder) -> c_int {
    guarded(|| {
        // SAFETY: argument validity is the caller's contract above.
        let encoder = unsafe { arg_mut(encoder, "encoder") }?;
        encoder.with(|inner| match inner.take() {
            Some(inner) => inner.finish(),
            None => Err(anyhow::anyhow!("finished")),
        })
    })
}

/// Free a group encoder. Does not flush. NULL is a no-op.
///
/// # Safety
/// `encoder` must be NULL or a live handle, not used again afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jbig2rs_group_encoder_free(encoder: *mut Jbig2rsGroupEncoder) {
    if !encoder.is_null() {
        // SAFETY: the handle came from `Box::into_raw` in `group_encoder_new`.
        let _ = catch_unwind(AssertUnwindSafe(|| drop(unsafe { Box::from_raw(encoder) })));
    }
}

/// The group's zero-based index in emission order.
///
/// # Safety
/// `group` must be the pointer passed to the current sink call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jbig2rs_pdf_group_index(group: *const Jbig2rsPdfGroup) -> usize {
    // SAFETY: the caller guarantees `group` is live.
    unsafe { pdf_group(group) }.map_or(0, |group| group.group_index)
}

/// The zero-based index, in the caller's input order, of the group's first page.
///
/// # Safety
/// `group` must be the pointer passed to the current sink call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jbig2rs_pdf_group_first_page(group: *const Jbig2rsPdfGroup) -> usize {
    // SAFETY: the caller guarantees `group` is live.
    unsafe { pdf_group(group) }.map_or(0, |group| group.first_page)
}

/// The group's streams, read with the `jbig2rs_pdf_output_*` accessors.
/// Borrowed from `group`: valid during the sink call only, and never freed
/// by the caller.
///
/// # Safety
/// `group` must be the pointer passed to the current sink call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jbig2rs_pdf_group_output(
    group: *const Jbig2rsPdfGroup,
) -> *const Jbig2rsPdfOutput {
    // SAFETY: the caller guarantees `group` is live. A `*Jbig2rsPdfOutput`
    // points at a `PdfSplitOutput` by definition.
    unsafe { pdf_group(group) }.map_or(std::ptr::null(), |group| {
        std::ptr::from_ref(&group.output).cast::<Jbig2rsPdfOutput>()
    })
}

// ── Decoding ────────────────────────────────────────────────────────────────

/// Decoder options. Initialise with [`jbig2rs_decode_options_init`].
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Jbig2rsDecodeOptions {
    /// `sizeof(Jbig2rsDecodeOptions)` as the caller compiled it.
    pub struct_size: u32,
    /// Largest page, in pixels, the decoder will allocate. 0 keeps the
    /// library default. Set it when decoding untrusted input.
    pub max_page_pixels: u64,
}

/// Fill `options` with the defaults. NULL is a no-op.
///
/// # Safety
/// `options` must be NULL or point to writable `Jbig2rsDecodeOptions`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jbig2rs_decode_options_init(options: *mut Jbig2rsDecodeOptions) {
    // SAFETY: the caller guarantees `options` is NULL or writable.
    if let Some(options) = unsafe { options.as_mut() } {
        *options = Jbig2rsDecodeOptions {
            struct_size: size_of::<Jbig2rsDecodeOptions>() as u32,
            max_page_pixels: 0,
        };
    }
}

/// Read caller decode options; NULL means the defaults.
///
/// # Safety
/// A non-null `options` must point to at least `options->struct_size` bytes.
unsafe fn decode_options(options: *const Jbig2rsDecodeOptions) -> Result<DecodeOptions, Failure> {
    let mut result = DecodeOptions::default();
    if options.is_null() {
        return Ok(result);
    }
    // SAFETY: as in `encode_options`.
    let struct_size = unsafe { options.cast::<u32>().read_unaligned() };
    if !covers(struct_size, 0, size_of::<u32>()) {
        return Err(Failure::invalid(
            "options->struct_size is too small; call jbig2rs_decode_options_init",
        ));
    }
    let offset = offset_of!(Jbig2rsDecodeOptions, max_page_pixels);
    if covers(struct_size, offset, size_of::<u64>()) {
        // SAFETY: `covers` checked the field lies within `struct_size`.
        let value = unsafe {
            options
                .cast::<u8>()
                .add(offset)
                .cast::<u64>()
                .read_unaligned()
        };
        if value != 0 {
            result.limits.max_page_pixels = value;
        }
    }
    Ok(result)
}

/// A decoded page bitmap. Opaque: a `*Jbig2rsBitmap` points at a
/// `MonoBitmap`, owned (from [`jbig2rs_decode_pdf_page`]) or borrowed from a
/// [`Jbig2rsDocument`].
#[derive(Debug)]
pub struct Jbig2rsBitmap {
    _opaque: [u8; 0],
}

/// # Safety
/// `bitmap` must be NULL or point at a live `MonoBitmap`.
unsafe fn mono_bitmap<'a>(bitmap: *const Jbig2rsBitmap) -> Option<&'a MonoBitmap> {
    // SAFETY: per the contract, the pointee is a `MonoBitmap`.
    unsafe { bitmap.cast::<MonoBitmap>().as_ref() }
}

/// Write a bitmap's size to `*width`/`*height` (either may be NULL).
///
/// # Safety
/// `bitmap` must be a live bitmap; `width` and `height` NULL or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jbig2rs_bitmap_size(
    bitmap: *const Jbig2rsBitmap,
    width: *mut u32,
    height: *mut u32,
) -> c_int {
    guarded(|| {
        // SAFETY: argument validity is the caller's contract above.
        let bitmap = unsafe { mono_bitmap(bitmap) }.ok_or_else(|| Failure::null("bitmap"))?;
        if let Some(width) = unsafe { width.as_mut() } {
            *width = bitmap.width();
        }
        if let Some(height) = unsafe { height.as_mut() } {
            *height = bitmap.height();
        }
        Ok(())
    })
}

/// Copy a bitmap into `dst` as MSB-first packed rows `dst_stride` bytes
/// apart, in `polarity` (`JBIG2RS_ONE_IS_BLACK` or `JBIG2RS_ZERO_IS_BLACK`).
/// `dst` needs `dst_stride * (height - 1) + ceil(width / 8)` bytes; bytes
/// between a row's end and the next stride are left untouched.
///
/// # Safety
/// `bitmap` must be live; `dst` must point to `dst_len` writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jbig2rs_bitmap_copy_rows(
    bitmap: *const Jbig2rsBitmap,
    dst: *mut u8,
    dst_len: usize,
    dst_stride: usize,
    polarity_value: u32,
) -> c_int {
    guarded(|| {
        // SAFETY: argument validity is the caller's contract above.
        let bitmap = unsafe { mono_bitmap(bitmap) }.ok_or_else(|| Failure::null("bitmap"))?;
        let polarity = polarity(polarity_value)?;
        if dst.is_null() {
            return Err(Failure::null("dst"));
        }
        // SAFETY: non-null, and the caller guarantees `dst_len` writable bytes.
        let dst = unsafe { std::slice::from_raw_parts_mut(dst, dst_len) };
        bitmap
            .copy_packed_rows(dst, dst_stride, polarity)
            .map_err(|error| Failure::invalid(error.to_string()))
    })
}

/// Free a bitmap from [`jbig2rs_decode_pdf_page`]. NULL is a no-op. Never
/// for a page borrowed from a document.
///
/// # Safety
/// `bitmap` must be NULL or a bitmap from `jbig2rs_decode_pdf_page`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jbig2rs_bitmap_free(bitmap: *mut Jbig2rsBitmap) {
    if !bitmap.is_null() {
        // SAFETY: the bitmap came from `Box::<MonoBitmap>::into_raw` in `decode_pdf_page`.
        let bitmap = bitmap.cast::<MonoBitmap>();
        let _ = catch_unwind(AssertUnwindSafe(|| drop(unsafe { Box::from_raw(bitmap) })));
    }
}

/// A decoded standalone JBIG2 file: one bitmap per page. Opaque.
#[derive(Debug)]
pub struct Jbig2rsDocument(DecodedDocument);

/// Decode a standalone JBIG2 file (or a bare embedded stream: the
/// organisation is detected) into a new `*out`, freed with
/// [`jbig2rs_document_free`]. `options` may be NULL. The input is read during
/// the call only.
///
/// # Safety
/// `data` must point to `len` readable bytes; `options` NULL or valid; `out`
/// writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jbig2rs_decode_file(
    data: *const u8,
    len: usize,
    options: *const Jbig2rsDecodeOptions,
    out: *mut *mut Jbig2rsDocument,
) -> c_int {
    guarded(|| {
        // SAFETY: argument validity is the caller's contract above.
        let out = unsafe { arg_mut(out, "out") }?;
        *out = std::ptr::null_mut();
        let data = unsafe { borrow_bytes(data, len, "data") }?;
        let options = unsafe { decode_options(options) }?;
        let document = decode_file(data, &options)?;
        *out = Box::into_raw(Box::new(Jbig2rsDocument(document)));
        Ok(())
    })
}

/// The number of pages, or 0 for NULL.
///
/// # Safety
/// `document` must be NULL or a live document.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jbig2rs_document_page_count(document: *const Jbig2rsDocument) -> usize {
    // SAFETY: the caller guarantees `document` is NULL or live.
    unsafe { document.as_ref() }.map_or(0, |document| document.0.pages.len())
}

/// Page `index`, borrowed from `document` (valid until it is freed; do not
/// free it), or NULL when out of range.
///
/// # Safety
/// `document` must be NULL or a live document.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jbig2rs_document_page(
    document: *const Jbig2rsDocument,
    index: usize,
) -> *const Jbig2rsBitmap {
    // SAFETY: the caller guarantees `document` is NULL or live. A
    // `*Jbig2rsBitmap` points at a `MonoBitmap` by definition.
    unsafe { document.as_ref() }
        .and_then(|document| document.0.pages.get(index))
        .map_or(std::ptr::null(), |page| {
            std::ptr::from_ref(&page.bitmap).cast::<Jbig2rsBitmap>()
        })
}

/// Free a document and every page bitmap borrowed from it. NULL is a no-op.
///
/// # Safety
/// `document` must be NULL or a live document, not used again afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jbig2rs_document_free(document: *mut Jbig2rsDocument) {
    if !document.is_null() {
        // SAFETY: the document came from `Box::into_raw` in `decode_file`.
        let _ = catch_unwind(AssertUnwindSafe(|| {
            drop(unsafe { Box::from_raw(document) })
        }));
    }
}

/// A decoded `/JBIG2Globals` stream. Immutable once decoded and safe to
/// share across threads decoding pages concurrently. Opaque.
pub struct Jbig2rsGlobals(DecodedGlobals);

impl std::fmt::Debug for Jbig2rsGlobals {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Jbig2rsGlobals").finish_non_exhaustive()
    }
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Jbig2rsGlobals>();
};

/// Decode a PDF `/JBIG2Globals` stream once, for any number of
/// [`jbig2rs_decode_pdf_page`] calls. Free with [`jbig2rs_globals_free`].
///
/// # Safety
/// `data` must point to `len` readable bytes; `options` NULL or valid; `out`
/// writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jbig2rs_globals_decode(
    data: *const u8,
    len: usize,
    options: *const Jbig2rsDecodeOptions,
    out: *mut *mut Jbig2rsGlobals,
) -> c_int {
    guarded(|| {
        // SAFETY: argument validity is the caller's contract above.
        let out = unsafe { arg_mut(out, "out") }?;
        *out = std::ptr::null_mut();
        let data = unsafe { borrow_bytes(data, len, "data") }?;
        let options = unsafe { decode_options(options) }?;
        let globals = decode_globals(data, &options)?;
        *out = Box::into_raw(Box::new(Jbig2rsGlobals(globals)));
        Ok(())
    })
}

/// Free decoded globals. NULL is a no-op.
///
/// # Safety
/// `globals` must be NULL or live, and not in use by any other thread.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jbig2rs_globals_free(globals: *mut Jbig2rsGlobals) {
    if !globals.is_null() {
        // SAFETY: the globals came from `Box::into_raw` in `globals_decode`.
        let _ = catch_unwind(AssertUnwindSafe(|| drop(unsafe { Box::from_raw(globals) })));
    }
}

/// Decode one PDF-embedded JBIG2 page stream into a new `*out`, freed with
/// [`jbig2rs_bitmap_free`]. `globals` is NULL when the image has no
/// `/JBIG2Globals`. `options` may be NULL.
///
/// # Safety
/// `globals` NULL or live; `data` must point to `len` readable bytes;
/// `options` NULL or valid; `out` writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jbig2rs_decode_pdf_page(
    globals: *const Jbig2rsGlobals,
    data: *const u8,
    len: usize,
    options: *const Jbig2rsDecodeOptions,
    out: *mut *mut Jbig2rsBitmap,
) -> c_int {
    guarded(|| {
        // SAFETY: argument validity is the caller's contract above.
        let out = unsafe { arg_mut(out, "out") }?;
        *out = std::ptr::null_mut();
        let data = unsafe { borrow_bytes(data, len, "data") }?;
        let options = unsafe { decode_options(options) }?;
        let globals = unsafe { globals.as_ref() }.map(|globals| &globals.0);
        let bitmap =
            decode_embedded_with_globals(globals, data, &options, &mut DecoderContext::new())?;
        *out = Box::into_raw(Box::new(bitmap)).cast::<Jbig2rsBitmap>();
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_panic_is_caught_at_the_boundary() {
        assert_eq!(guarded(|| panic!("simulated panic")), JBIG2RS_ERR_PANIC);
        let message = jbig2rs_last_error_message();
        assert!(!message.is_null());
        // SAFETY: just returned non-null by the library.
        let message = unsafe { std::ffi::CStr::from_ptr(message) }.to_string_lossy();
        assert!(message.contains("panicked"), "{message}");
    }

    #[test]
    fn a_panic_poisons_the_encoder_handle() {
        let mut handle = Jbig2rsEncoder {
            inner: Some(DocumentEncoder::from_options(&EncodeOptions::default()).unwrap()),
            poisoned: false,
        };
        let status = guarded(|| handle.with(|_| panic!("mid-call panic")));
        assert_eq!(status, JBIG2RS_ERR_PANIC);
        let status = guarded(|| handle.with(|_| Ok(())));
        assert_eq!(status, JBIG2RS_ERR_BAD_STATE);
    }
}
