//! Python bindings for `jbig2enc-rust`: the `jbig2enc` extension module.
//!
//! The wheel is built with maturin from this directory, which holds the
//! `pyproject.toml`, the type stubs and the Python test suite. This crate is
//! the whole extension. It is a thin layer over the codec's binding-neutral
//! facade (`BilevelImage`, `EncodeOptions`, `DocumentEncoder`), the same one
//! the C ABI in `../ffi` translates. So pixel layouts, polarity, thresholding
//! and preset handling behave identically in both.
//!
//! What this layer adds is meeting Python where it is:
//!
//! * **Any buffer works as input.** `bytes`, `bytearray`, `memoryview`, a
//!   NumPy array, a `PIL` image's `tobytes()`: anything implementing the
//!   buffer protocol. A C-contiguous buffer is read in place, with the GIL
//!   held, straight into the encoder's packed bitmap. No intermediate copy.
//! * **Polarity is explicit.** JBIG2 stores 1 = black, while every
//!   grayscale image format in Python stores 0 = black. Conflating the two
//!   produces a perfectly valid, perfectly inverted JBIG2 file. So
//!   `pixel_format="gray8"` (or [`pack_grayscale`]) thresholds and inverts
//!   in one obvious place, and packed input names its `polarity`.
//! * **The GIL is released** around every encode and decode. These calls
//!   run for hundreds of milliseconds on a real page; holding the GIL through
//!   that would serialise any caller encoding pages in a thread pool.
//! * **Errors are Python exceptions**, not status codes.
//!
//! Symbol substitution is **on by default**, matching the encoder. It is
//! what makes JBIG2 worth choosing over CCITT G4 on scanned text, but it is
//! a *lossy* transform: near-identical glyphs are replaced by a shared
//! bitmap, so a round trip is not guaranteed bit-exact. Pass
//! `symbol_mode=False` (or `preset="generic"`) for a generic region, which is
//! bit-exact and is the right choice for line art, halftones, or anything
//! where a substituted glyph would be unacceptable.
//!
//! `Jbig2Config` has more than fifty fields, nearly all of them
//! symbol-unification tuning knobs that only make sense with the T.88 spec
//! open. This module exposes only the named presets and the options that
//! were *measured* to change the encoder's output: `symbol_mode`, `refine`
//! and `lossless`. Everything else keeps the Rust defaults.
//!
//! Three plausible-looking fields are deliberately **not** exposed, because
//! they are inert in the encoder as it currently stands:
//!
//! * `dpi`: recorded in the config but never written to the stream.
//! * `duplicate_line_removal`: defaults on; toggling it changed no output
//!   in testing.
//! * `match_tolerance`: only read when `text_refine` is set, and even then
//!   produced identical bytes.
//!
//! Exposing a knob that silently does nothing is worse than omitting it, so
//! these stay out until the encoder honours them.

use jbig2enc_rust::decode::{DecodeOptions, decode_embedded, decode_file};
use jbig2enc_rust::jbig2sym::BitImage;
use jbig2enc_rust::{
    BilevelImage, DocumentEncoder, EncodeOptions, Jbig2Config, Jbig2Error, PixelLayout, Polarity,
    Preset,
};
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyBufferError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

/// Translate an encoder error into the closest Python exception.
///
/// `PackedDataDetected` gets the long-form message on purpose: passing
/// 1-bit-packed data where one byte per pixel is expected is *the* mistake
/// this API invites, and the fix is not guessable from "buffer size
/// mismatch".
fn to_py_error(error: Jbig2Error) -> PyErr {
    match error {
        Jbig2Error::PackedDataDetected => PyValueError::new_err(
            "input looks like 1-bit-packed data (8 pixels per byte), but this \
             function expects one byte per pixel. Pass pixel_format=\"packed1\", \
             or use jbig2enc.pack_grayscale() to build the buffer from an 8-bit image.",
        ),
        other => PyValueError::new_err(other.to_string()),
    }
}

/// Access a Python object through the buffer protocol.
fn buffer_bytes(object: &Bound<'_, PyAny>) -> PyResult<PyBuffer<u8>> {
    PyBuffer::<u8>::get(object).map_err(|_| {
        PyBufferError::new_err(
            "expected a bytes-like object of 8-bit values (bytes, bytearray, \
             memoryview, or a contiguous uint8 array)",
        )
    })
}

/// Validate dimensions and return the pixel count.
fn expected_len(width: u32, height: u32) -> PyResult<usize> {
    if width == 0 || height == 0 {
        return Err(PyValueError::new_err(
            "width and height must both be greater than zero",
        ));
    }
    (width as usize)
        .checked_mul(height as usize)
        .ok_or_else(|| PyValueError::new_err("width * height overflows the address space"))
}

/// How the caller's buffer encodes its pixels: the `pixel_format`,
/// `stride`, `polarity` and `threshold` keyword arguments.
#[derive(Clone, Copy, Debug)]
struct InputFormat {
    kind: FormatKind,
    stride: Option<usize>,
    polarity: Polarity,
    threshold: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FormatKind {
    Bytes,
    Packed1,
    Gray8,
}

impl Default for InputFormat {
    fn default() -> Self {
        Self {
            kind: FormatKind::Bytes,
            stride: None,
            polarity: Polarity::OneIsBlack,
            threshold: 128,
        }
    }
}

impl InputFormat {
    fn layout(&self, width: u32) -> PixelLayout {
        let width = width as usize;
        match self.kind {
            FormatKind::Bytes => PixelLayout::Bytes {
                stride: self.stride.unwrap_or(width),
            },
            FormatKind::Packed1 => PixelLayout::Packed1 {
                stride: self.stride.unwrap_or(width.div_ceil(8)),
            },
            FormatKind::Gray8 => PixelLayout::Gray8 {
                stride: self.stride.unwrap_or(width),
                threshold: self.threshold,
            },
        }
    }
}

/// Read a Python buffer into the encoder's packed bitmap.
///
/// Runs with the GIL held, which is what makes reading a C-contiguous buffer
/// in place sound: no other Python code can resize or mutate it meanwhile.
/// The result is an eighth of the size of byte-per-pixel input and is all
/// the encode needs once the GIL is released.
fn read_bitmap(
    object: &Bound<'_, PyAny>,
    width: u32,
    height: u32,
    format: &InputFormat,
) -> PyResult<BitImage> {
    let pixel_count = expected_len(width, height)?;
    let buffer = buffer_bytes(object)?;
    let layout = format.layout(width);
    if format.kind == FormatKind::Bytes && format.stride.is_none() {
        let len = buffer.len_bytes();
        if len != pixel_count {
            if len == pixel_count.div_ceil(8)
                || len == (width as usize).div_ceil(8) * height as usize
            {
                return Err(to_py_error(Jbig2Error::PackedDataDetected));
            }
            return Err(PyValueError::new_err(format!(
                "buffer has {len} bytes but {width}x{height} needs {pixel_count} \
                 (one byte per pixel)"
            )));
        }
    }
    let convert = |data: &[u8]| {
        BilevelImage::new(width, height, data, layout, format.polarity)
            .and_then(|image| image.to_bit_image())
            .map_err(to_py_error)
    };
    if buffer.is_c_contiguous() {
        // SAFETY: `buffer` holds the exporter's buffer view for its whole
        // lifetime, so `buf_ptr` points to `len_bytes` initialised bytes. The
        // GIL is held throughout, so no Python code can mutate or release them
        // while this slice lives, and the slice does not outlive `buffer`.
        let data = unsafe {
            std::slice::from_raw_parts(buffer.buf_ptr().cast::<u8>(), buffer.len_bytes())
        };
        convert(data)
    } else {
        convert(&buffer.to_vec(object.py())?)
    }
}

/// The encoder configuration and input format the keyword arguments describe.
fn options_from_kwargs(kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<(Jbig2Config, InputFormat)> {
    let mut format = InputFormat::default();
    let Some(kwargs) = kwargs else {
        return Ok((Jbig2Config::default(), format));
    };

    let mut preset: Option<Preset> = None;
    let mut symbol_mode: Option<bool> = None;
    let mut lossless = false;
    let mut refine: Option<bool> = None;
    for (key, value) in kwargs.iter() {
        let key: String = key.extract()?;
        match key.as_str() {
            "preset" => {
                let name: String = value.extract()?;
                preset = Some(Preset::from_name(&name).ok_or_else(|| {
                    let names: Vec<_> = Preset::ALL.iter().map(|preset| preset.name()).collect();
                    PyValueError::new_err(format!(
                        "unknown preset '{name}'. Supported: {}",
                        names.join(", ")
                    ))
                })?);
            }
            "symbol_mode" => symbol_mode = Some(value.extract()?),
            "lossless" => lossless = value.extract()?,
            "refine" => refine = Some(value.extract()?),
            "pixel_format" => {
                let name: String = value.extract()?;
                format.kind = match name.as_str() {
                    "bytes" => FormatKind::Bytes,
                    "packed1" => FormatKind::Packed1,
                    "gray8" => FormatKind::Gray8,
                    other => {
                        return Err(PyValueError::new_err(format!(
                            "unknown pixel_format '{other}'. Supported: bytes, packed1, gray8"
                        )));
                    }
                };
            }
            "stride" => format.stride = value.extract::<Option<usize>>()?,
            "polarity" => {
                let name: String = value.extract()?;
                format.polarity = match name.as_str() {
                    "one_is_black" => Polarity::OneIsBlack,
                    "zero_is_black" => Polarity::ZeroIsBlack,
                    other => {
                        return Err(PyValueError::new_err(format!(
                            "unknown polarity '{other}'. Supported: one_is_black, zero_is_black"
                        )));
                    }
                };
            }
            "threshold" => format.threshold = value.extract()?,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown option '{other}'. Supported: preset, symbol_mode, refine, \
                     lossless, pixel_format, stride, polarity, threshold"
                )));
            }
        }
    }

    // A preset names the whole configuration; mixing it with the older
    // switches would make one silently override the other.
    if let Some(preset) = preset {
        if symbol_mode.is_some() || lossless {
            return Err(PyValueError::new_err(
                "preset cannot be combined with symbol_mode or lossless",
            ));
        }
        let mut options = EncodeOptions::new(preset);
        options.refine = refine;
        return Ok((options.to_config().map_err(to_py_error)?, format));
    }

    // `lossless` is a preset rather than a field: it turns off symbol
    // substitution and refinement together, which is the combination that
    // makes a stream safe for viewers that mishandle shared dictionaries.
    let mut config = if lossless {
        Jbig2Config::lossless()
    } else {
        Jbig2Config::default()
    };
    match symbol_mode {
        Some(true) => config.symbol_mode = true,
        // `Jbig2Config::generic()` is the explicit off switch, not merely
        // `default()` with the flag cleared: it also clears the refinement
        // flags that only mean anything under symbol mode.
        Some(false) => config = Jbig2Config::generic(),
        None => {}
    }
    if let Some(refine) = refine {
        config.refine = refine;
    }
    // Refinement without symbol substitution is not a meaningful combination;
    // the `lossless()` preset clears both together, so mirror that rather
    // than emitting a stream whose refine flag can never fire.
    if !config.symbol_mode {
        config.refine = false;
    }
    Ok((config, format))
}

/// Encode one bilevel image as a standalone JBIG2 file.
///
/// `pixels` is one byte per pixel in row-major order by default; any
/// non-zero value is black. Returns the complete `.jb2` file bytes.
#[pyfunction]
#[pyo3(signature = (pixels, width, height, **options))]
fn encode(
    py: Python<'_>,
    pixels: &Bound<'_, PyAny>,
    width: u32,
    height: u32,
    options: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyBytes>> {
    let (config, format) = options_from_kwargs(options)?;
    let bitmap = read_bitmap(pixels, width, height, &format)?;
    let file = py
        .detach(|| {
            let mut encoder = DocumentEncoder::new(config)?;
            encoder.add_bit_image(bitmap)?;
            encoder.finish_file()
        })
        .map_err(to_py_error)?;
    Ok(PyBytes::new(py, &file).unbind())
}

/// Encode one bilevel image as a PDF-embeddable fragment.
///
/// Returns `(globals, page_data)`, where `globals` is the shared symbol
/// dictionary to store as a separate PDF object and reference from the
/// image stream's `/DecodeParms /JBIG2Globals`, or `None` when the encoder
/// produced no dictionary.
#[pyfunction]
#[pyo3(signature = (pixels, width, height, **options))]
fn encode_for_pdf(
    py: Python<'_>,
    pixels: &Bound<'_, PyAny>,
    width: u32,
    height: u32,
    options: Option<&Bound<'_, PyDict>>,
) -> PyResult<(Option<Py<PyBytes>>, Py<PyBytes>)> {
    let (config, format) = options_from_kwargs(options)?;
    let bitmap = read_bitmap(pixels, width, height, &format)?;
    let output = py
        .detach(|| {
            let mut encoder = DocumentEncoder::new(config)?;
            encoder.add_bit_image(bitmap)?;
            encoder.finish_pdf()
        })
        .map_err(to_py_error)?;
    let mut streams = output.page_streams.into_iter();
    let (Some(page), None) = (streams.next(), streams.next()) else {
        return Err(PyValueError::new_err(
            "the encoder did not produce exactly one page stream",
        ));
    };
    let globals = output
        .global_segments
        .map(|data| PyBytes::new(py, &data).unbind());
    Ok((globals, PyBytes::new(py, &page).unbind()))
}

/// `(shared globals, one stream per page)`.
type DocumentStreams = (Option<Py<PyBytes>>, Vec<Py<PyBytes>>);

/// Encode several pages against one shared symbol dictionary.
///
/// `pages` is a sequence of `(pixels, width, height)` tuples. Returns
/// `(globals, [page_data, ...])`.
///
/// This is not the same as calling [`encode_for_pdf`] per page: PDF readers
/// expect a single `/JBIG2Globals` object referenced by every page stream,
/// and the symbol indices in those streams only line up if one encoder
/// instance planned all of them. Encoding pages independently and stitching
/// the results produces undecodable pages.
#[pyfunction]
#[pyo3(signature = (pages, **options))]
fn encode_document(
    py: Python<'_>,
    pages: Vec<(Py<PyAny>, u32, u32)>,
    options: Option<&Bound<'_, PyDict>>,
) -> PyResult<DocumentStreams> {
    if pages.is_empty() {
        return Err(PyValueError::new_err("pages must not be empty"));
    }
    let (config, format) = options_from_kwargs(options)?;

    let mut bitmaps = Vec::with_capacity(pages.len());
    for (index, (pixels, width, height)) in pages.iter().enumerate() {
        let bitmap = read_bitmap(pixels.bind(py), *width, *height, &format)
            .map_err(|error| PyValueError::new_err(format!("page {index}: {error}")))?;
        bitmaps.push(bitmap);
    }

    let output = py
        .detach(|| {
            let mut encoder = DocumentEncoder::new(config)?;
            for bitmap in bitmaps {
                encoder.add_bit_image(bitmap)?;
            }
            encoder.finish_pdf()
        })
        .map_err(to_py_error)?;

    let globals = output
        .global_segments
        .map(|data| PyBytes::new(py, &data).unbind());
    let streams = output
        .page_streams
        .iter()
        .map(|stream| PyBytes::new(py, stream).unbind())
        .collect();
    Ok((globals, streams))
}

/// Threshold an 8-bit grayscale image into the buffer the encoders expect.
///
/// Grayscale images store 0 as black; JBIG2 stores 1 as black. This does the
/// comparison and the inversion together so callers do not have to
/// rediscover which way round it goes, a mistake that yields a valid but
/// fully inverted file. (`pixel_format="gray8"` on the encode functions does
/// the same without the intermediate buffer.)
///
/// A pixel is black when its value is **below** `threshold`.
#[pyfunction]
#[pyo3(signature = (pixels, width, height, threshold = 128))]
fn pack_grayscale(
    py: Python<'_>,
    pixels: &Bound<'_, PyAny>,
    width: u32,
    height: u32,
    threshold: u8,
) -> PyResult<Py<PyBytes>> {
    let expected = expected_len(width, height)?;
    let buffer = buffer_bytes(pixels)?;
    let raw = buffer.to_vec(py)?;
    if raw.len() != expected {
        return Err(PyValueError::new_err(format!(
            "buffer has {} bytes but {width}x{height} needs {expected}",
            raw.len()
        )));
    }
    let packed: Vec<u8> = raw
        .into_iter()
        .map(|value| u8::from(value < threshold))
        .collect();
    Ok(PyBytes::new(py, &packed).unbind())
}

/// One byte per pixel, 1 = black, matching what the encode functions take
/// by default so a round trip is byte-comparable.
fn unpack_bitmap(py: Python<'_>, bitmap: &jbig2enc_rust::MonoBitmap) -> (Py<PyBytes>, u32, u32) {
    let pixels = bitmap.to_bytes_per_pixel();
    (
        PyBytes::new(py, &pixels).unbind(),
        bitmap.width(),
        bitmap.height(),
    )
}

/// Decode a standalone JBIG2 file.
///
/// Returns `(pixels, width, height)` with one byte per pixel, 1 = black:
/// the same layout [`encode`] accepts, so a round trip compares equal.
///
/// Only the first page is returned; use [`decode_document`] for a
/// multi-page file.
#[pyfunction]
fn decode(py: Python<'_>, data: &[u8]) -> PyResult<(Py<PyBytes>, u32, u32)> {
    let document = py
        .detach(|| decode_file(data, &DecodeOptions::default()))
        .map_err(|error| PyValueError::new_err(format!("decoding JBIG2: {error}")))?;
    let bitmap = document
        .first_page()
        .ok_or_else(|| PyValueError::new_err("JBIG2 stream contains no pages"))?;
    Ok(unpack_bitmap(py, bitmap))
}

/// Decode every page of a standalone JBIG2 file.
///
/// Returns a list of `(pixels, width, height)` tuples in stream order.
#[pyfunction]
fn decode_document(py: Python<'_>, data: &[u8]) -> PyResult<Vec<(Py<PyBytes>, u32, u32)>> {
    let document = py
        .detach(|| decode_file(data, &DecodeOptions::default()))
        .map_err(|error| PyValueError::new_err(format!("decoding JBIG2: {error}")))?;
    Ok(document
        .pages
        .iter()
        .map(|page| unpack_bitmap(py, &page.bitmap))
        .collect())
}

/// Decode a PDF-embedded JBIG2 page stream.
///
/// `globals` is the `/JBIG2Globals` stream when the page references one, as
/// returned by [`encode_for_pdf`] or [`encode_document`]. Pass `None` for a
/// page encoded without a shared dictionary.
#[pyfunction]
#[pyo3(signature = (page_data, globals = None))]
fn decode_pdf_stream(
    py: Python<'_>,
    page_data: &[u8],
    globals: Option<&[u8]>,
) -> PyResult<(Py<PyBytes>, u32, u32)> {
    let bitmap = py
        .detach(|| decode_embedded(globals, page_data, &DecodeOptions::default()))
        .map_err(|error| PyValueError::new_err(format!("decoding JBIG2 page: {error}")))?;
    Ok(unpack_bitmap(py, &bitmap))
}

/// Encoder version string.
#[pyfunction]
fn version() -> String {
    jbig2enc_rust::get_version()
}

/// Build configuration of the linked encoder.
#[pyfunction]
fn build_info() -> String {
    jbig2enc_rust::get_build_info()
}

/// The names `preset=` accepts.
#[pyfunction]
fn presets() -> Vec<&'static str> {
    Preset::ALL.iter().map(|preset| preset.name()).collect()
}

/// The `jbig2enc` extension module.
#[pymodule]
#[pyo3(name = "jbig2enc")]
pub fn jbig2enc(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", jbig2enc_rust::VERSION)?;
    module.add_function(wrap_pyfunction!(encode, module)?)?;
    module.add_function(wrap_pyfunction!(encode_for_pdf, module)?)?;
    module.add_function(wrap_pyfunction!(encode_document, module)?)?;
    module.add_function(wrap_pyfunction!(decode, module)?)?;
    module.add_function(wrap_pyfunction!(decode_document, module)?)?;
    module.add_function(wrap_pyfunction!(decode_pdf_stream, module)?)?;
    module.add_function(wrap_pyfunction!(pack_grayscale, module)?)?;
    module.add_function(wrap_pyfunction!(presets, module)?)?;
    module.add_function(wrap_pyfunction!(version, module)?)?;
    module.add_function(wrap_pyfunction!(build_info, module)?)?;
    Ok(())
}
