//! The binding-neutral encode API.
//!
//! Everything a language binding needs, and nothing it would have to
//! reimplement: a borrowed view over the caller's pixels in whatever layout
//! they already hold ([`BilevelImage`]), a small, stable set of named
//! configurations ([`Preset`], [`EncodeOptions`]), an encoder that owns its
//! configuration and so can live behind an opaque handle ([`DocumentEncoder`]),
//! and errors with a stable class ([`Jbig2Error::kind`]).
//!
//! The C ABI (`jbig2enc-ffi`) and the Python wheel (`jbig2enc-py`) are thin
//! translations of this module. Polarity, packing, thresholding, preset
//! normalisation and error classification live here once, not in each.

use crate::Jbig2Error;
use crate::encode::document::{Jbig2Encoder, PdfSplitOutput, ResourceLimitExceeded};
use crate::encode::structs::Jbig2Config;
use crate::encode::sym::BitImage;
use crate::shared::bitmap::Polarity;

/// How a [`BilevelImage`]'s bytes encode its pixels. Every layout is
/// row-major and top-down, with rows `stride` bytes apart.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum PixelLayout {
    /// One bit per pixel, MSB first: PBM, Leptonica `PIX`, TIFF bilevel
    /// strips, 1bpp DIBs. Bits past the width in each row are ignored.
    Packed1 { stride: usize },
    /// One byte per pixel. With [`Polarity::OneIsBlack`] any non-zero byte is
    /// black; with [`Polarity::ZeroIsBlack`] only a zero byte is.
    Bytes { stride: usize },
    /// 8-bit grayscale, thresholded: a pixel is black when its value is
    /// **below** `threshold`. Polarity is ignored, since grayscale is always
    /// 0 = black.
    Gray8 { stride: usize, threshold: u8 },
}

impl PixelLayout {
    /// Bytes between the starts of consecutive rows.
    pub fn stride(&self) -> usize {
        match *self {
            PixelLayout::Packed1 { stride }
            | PixelLayout::Bytes { stride }
            | PixelLayout::Gray8 { stride, .. } => stride,
        }
    }

    /// Bytes one row of `width` pixels occupies in this layout.
    pub fn row_bytes(&self, width: u32) -> usize {
        let width = width as usize;
        match self {
            PixelLayout::Packed1 { .. } => width.div_ceil(8),
            PixelLayout::Bytes { .. } | PixelLayout::Gray8 { .. } => width,
        }
    }
}

/// A borrowed, validated view of a bilevel (or grayscale, to be thresholded)
/// image in the caller's own buffer.
///
/// Construction checks the dimensions, the stride and the buffer length, so
/// a `BilevelImage` that exists can always be converted.
#[derive(Clone, Copy, Debug)]
pub struct BilevelImage<'a> {
    width: u32,
    height: u32,
    data: &'a [u8],
    layout: PixelLayout,
    polarity: Polarity,
}

impl<'a> BilevelImage<'a> {
    /// A view over `data` in any layout. `data` needs
    /// `stride * (height - 1) + row_bytes` bytes: the last row need not be
    /// padded out to the stride.
    pub fn new(
        width: u32,
        height: u32,
        data: &'a [u8],
        layout: PixelLayout,
        polarity: Polarity,
    ) -> Result<Self, Jbig2Error> {
        let max = BitImage::MAX_DIMENSION as u64;
        for (name, value) in [("width", width), ("height", height)] {
            if value == 0 || u64::from(value) > max {
                return Err(Jbig2Error::invalid_input(format!(
                    "{name} must be between 1 and {max}, got {value}"
                )));
            }
        }
        let row_bytes = layout.row_bytes(width);
        let stride = layout.stride();
        if stride < row_bytes {
            return Err(Jbig2Error::invalid_input(format!(
                "stride {stride} is shorter than a {width}-pixel row ({row_bytes} bytes)"
            )));
        }
        let required = stride
            .checked_mul(height as usize - 1)
            .and_then(|bytes| bytes.checked_add(row_bytes))
            .ok_or_else(|| {
                Jbig2Error::invalid_input("image size overflows the address space".to_string())
            })?;
        if data.len() < required {
            // Packed data passed where bytes were declared is *the* mistake a
            // byte-per-pixel API invites; name it rather than a bare length.
            if matches!(layout, PixelLayout::Bytes { .. })
                && stride == width as usize
                && data.len() == (width as usize).div_ceil(8) * height as usize
            {
                return Err(Jbig2Error::PackedDataDetected);
            }
            return Err(Jbig2Error::invalid_input(format!(
                "buffer has {} bytes but a {width}x{height} image needs {required}",
                data.len()
            )));
        }
        Ok(Self {
            width,
            height,
            data,
            layout,
            polarity,
        })
    }

    /// Packed 1bpp rows, MSB first, `stride` bytes apart.
    pub fn packed(
        width: u32,
        height: u32,
        stride: usize,
        data: &'a [u8],
        polarity: Polarity,
    ) -> Result<Self, Jbig2Error> {
        Self::new(
            width,
            height,
            data,
            PixelLayout::Packed1 { stride },
            polarity,
        )
    }

    /// One byte per pixel, unpadded rows, any non-zero byte black: the layout
    /// [`encode_single_image`](crate::encode_single_image) has always taken.
    pub fn bytes(width: u32, height: u32, data: &'a [u8]) -> Result<Self, Jbig2Error> {
        Self::new(
            width,
            height,
            data,
            PixelLayout::Bytes {
                stride: width as usize,
            },
            Polarity::OneIsBlack,
        )
    }

    /// 8-bit grayscale, unpadded rows, black where the value is below
    /// `threshold`.
    pub fn gray8(
        width: u32,
        height: u32,
        data: &'a [u8],
        threshold: u8,
    ) -> Result<Self, Jbig2Error> {
        Self::new(
            width,
            height,
            data,
            PixelLayout::Gray8 {
                stride: width as usize,
                threshold,
            },
            Polarity::ZeroIsBlack,
        )
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn layout(&self) -> PixelLayout {
        self.layout
    }

    pub fn polarity(&self) -> Polarity {
        self.polarity
    }

    /// Convert to the encoder's packed bitmap (1 = black).
    pub fn to_bit_image(&self) -> Result<BitImage, Jbig2Error> {
        let (width, height) = (self.width as usize, self.height as usize);
        let result = match (self.layout, self.polarity) {
            (PixelLayout::Packed1 { stride }, polarity) => BitImage::from_packed_rows(
                width,
                height,
                stride,
                self.data,
                polarity == Polarity::ZeroIsBlack,
            ),
            (PixelLayout::Bytes { stride }, Polarity::OneIsBlack) => {
                BitImage::from_byte_rows(width, height, stride, self.data, |value| value != 0)
            }
            (PixelLayout::Bytes { stride }, Polarity::ZeroIsBlack) => {
                BitImage::from_byte_rows(width, height, stride, self.data, |value| value == 0)
            }
            (PixelLayout::Gray8 { stride, threshold }, _) => {
                BitImage::from_byte_rows(width, height, stride, self.data, |value| {
                    value < threshold
                })
            }
        };
        result.map_err(Jbig2Error::invalid_input)
    }
}

/// A named encoder configuration. These are the configurations that were
/// measured to matter; [`Jbig2Config`] remains the expert surface for the
/// fifty-odd tuning fields underneath them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Preset {
    /// [`Jbig2Config::default`]: symbol coding on.
    #[default]
    Default,
    /// [`Jbig2Config::text`]: symbol coding tuned for text.
    Text,
    /// [`Jbig2Config::generic`]: one generic region per page, bit-exact.
    Generic,
    /// [`Jbig2Config::lossless`]: generic region, no duplicate-line removal,
    /// no dictionary. The safest choice for viewers that mishandle globals.
    Lossless,
    /// [`Jbig2Config::text_lossless`]: pixel-exact symbol coding.
    TextLossless,
    /// [`Jbig2Config::text_lossless_max_compression`]: slower pixel-exact
    /// symbol coding that measures its alternatives.
    TextLosslessMaxCompression,
    /// [`Jbig2Config::text_symbol_unify`]: lossy glyph unification.
    TextSymbolUnify,
    /// [`Jbig2Config::text_symbol_unify_safe`]: lossy unification with a
    /// topology guard on every substitution.
    TextSymbolUnifySafe,
}

impl Preset {
    /// Every preset, in declaration order.
    pub const ALL: [Preset; 8] = [
        Preset::Default,
        Preset::Text,
        Preset::Generic,
        Preset::Lossless,
        Preset::TextLossless,
        Preset::TextLosslessMaxCompression,
        Preset::TextSymbolUnify,
        Preset::TextSymbolUnifySafe,
    ];

    /// The preset's configuration, before any [`EncodeOptions`] overrides.
    pub fn config(self) -> Jbig2Config {
        match self {
            Preset::Default => Jbig2Config::default(),
            Preset::Text => Jbig2Config::text(),
            Preset::Generic => Jbig2Config::generic(),
            Preset::Lossless => Jbig2Config::lossless(),
            Preset::TextLossless => Jbig2Config::text_lossless(),
            Preset::TextLosslessMaxCompression => Jbig2Config::text_lossless_max_compression(),
            Preset::TextSymbolUnify => Jbig2Config::text_symbol_unify(),
            Preset::TextSymbolUnifySafe => Jbig2Config::text_symbol_unify_safe(),
        }
    }

    /// Stable snake_case name, for options parsed from text.
    pub fn name(self) -> &'static str {
        match self {
            Preset::Default => "default",
            Preset::Text => "text",
            Preset::Generic => "generic",
            Preset::Lossless => "lossless",
            Preset::TextLossless => "text_lossless",
            Preset::TextLosslessMaxCompression => "text_lossless_max_compression",
            Preset::TextSymbolUnify => "text_symbol_unify",
            Preset::TextSymbolUnifySafe => "text_symbol_unify_safe",
        }
    }

    /// The preset with this [`name`](Self::name), if any.
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|preset| preset.name() == name)
    }

    /// Whether the preset uses symbol coding (and so can take `refine`).
    pub fn uses_symbols(self) -> bool {
        self.config().symbol_mode
    }
}

/// The stable option surface: a [`Preset`] plus the overrides that were
/// measured to change the output.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct EncodeOptions {
    pub preset: Preset,
    /// Override the preset's refinement-coding flag. Only meaningful with a
    /// symbol-coding preset; `Some(true)` on a generic preset is an error.
    pub refine: Option<bool>,
}

impl EncodeOptions {
    pub fn new(preset: Preset) -> Self {
        Self {
            preset,
            refine: None,
        }
    }

    pub fn with_refine(mut self, refine: bool) -> Self {
        self.refine = Some(refine);
        self
    }

    /// The validated configuration these options describe.
    pub fn to_config(&self) -> Result<Jbig2Config, Jbig2Error> {
        let mut config = self.preset.config();
        if let Some(refine) = self.refine {
            if refine && !config.symbol_mode {
                return Err(Jbig2Error::invalid_config(format!(
                    "refinement requires a symbol-coding preset; '{}' codes generic regions",
                    self.preset.name()
                )));
            }
            config.refine = refine;
        }
        validate_config(&mut config)?;
        Ok(config)
    }
}

/// Normalise and validate a configuration for [`DocumentEncoder`].
///
/// Clears `text_refine` without symbol coding (it is inert there) and asks
/// for file headers (only [`DocumentEncoder::finish_file`] reads them).
fn validate_config(config: &mut Jbig2Config) -> Result<(), Jbig2Error> {
    config.want_full_headers = true;
    if !config.symbol_mode {
        config.text_refine = false;
    }
    let invalid = |message: &str| Jbig2Error::invalid_config(message);
    if config.refine && !config.symbol_mode {
        return Err(invalid("refinement requires symbol mode to be enabled"));
    }
    config.validate_exact_symbol_mode().map_err(invalid)?;
    config.generic.validate().map_err(invalid)?;
    Ok(())
}

/// A multi-page encoder that owns its configuration.
///
/// Every page added to one `DocumentEncoder` shares one symbol dictionary,
/// so its output is a single JBIG2 file or a PDF `JBIG2Globals` stream plus
/// one page stream per page. Pages are analysed as they are added; the
/// source pixels are not retained.
///
/// `Send`, so it can be built on one thread and finished on another.
pub struct DocumentEncoder {
    inner: Jbig2Encoder<'static>,
}

impl std::fmt::Debug for DocumentEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DocumentEncoder")
            .field("pages", &self.inner.get_page_count())
            .finish_non_exhaustive()
    }
}

impl DocumentEncoder {
    /// An encoder for `config`, validated up front so that a bad
    /// configuration fails here as [`ErrorKind::InvalidConfig`](crate::ErrorKind::InvalidConfig).
    pub fn new(mut config: Jbig2Config) -> Result<Self, Jbig2Error> {
        validate_config(&mut config)?;
        Ok(Self {
            inner: Jbig2Encoder::owned(config),
        })
    }

    pub fn from_options(options: &EncodeOptions) -> Result<Self, Jbig2Error> {
        Self::new(options.to_config()?)
    }

    /// Add the next page.
    pub fn add_page(&mut self, image: &BilevelImage<'_>) -> Result<(), Jbig2Error> {
        self.add_bit_image(image.to_bit_image()?)
    }

    /// Add the next page from an already-packed bitmap.
    pub fn add_bit_image(&mut self, image: BitImage) -> Result<(), Jbig2Error> {
        self.inner
            .add_page_bitimage(image)
            .map_err(Jbig2Error::from)
    }

    pub fn page_count(&self) -> usize {
        self.inner.get_page_count()
    }

    /// Finish as a standalone JBIG2 file (with file header).
    pub fn finish_file(mut self) -> Result<Vec<u8>, Jbig2Error> {
        self.require_pages()?;
        self.inner.flush().map_err(Jbig2Error::from)
    }

    /// Finish as PDF fragments: optional globals plus one stream per page.
    pub fn finish_pdf(mut self) -> Result<PdfSplitOutput, Jbig2Error> {
        self.require_pages()?;
        self.inner.flush_pdf_split().map_err(Jbig2Error::from)
    }

    fn require_pages(&self) -> Result<(), Jbig2Error> {
        if self.inner.get_page_count() == 0 {
            return Err(Jbig2Error::invalid_input("no pages were added".to_string()));
        }
        Ok(())
    }
}

/// Classify an encoder-internal error: a configured resource budget the
/// output would exceed is classified [`ErrorKind::ResourceLimit`](crate::ErrorKind::ResourceLimit);
/// anything else is an encoding failure.
impl From<anyhow::Error> for Jbig2Error {
    fn from(error: anyhow::Error) -> Self {
        if let Some(limit) = error.downcast_ref::<ResourceLimitExceeded>() {
            return Jbig2Error::resource_limit(&limit.0);
        }
        Jbig2Error::EncodingFailed {
            message: error.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ErrorKind;

    const _: () = {
        const fn assert_send<T: Send>() {}
        assert_send::<DocumentEncoder>();
    };

    /// A deterministic test page with repeated glyph-like marks.
    fn page(width: u32, height: u32) -> Vec<u8> {
        (0..width * height)
            .map(|i| {
                let (x, y) = (i % width, i / width);
                u8::from((x / 3 + y / 5) % 7 == 0 || (x % 11 == 0 && y % 13 < 6))
            })
            .collect()
    }

    fn pack(pixels: &[u8], width: u32, height: u32, stride: usize, zero_is_black: bool) -> Vec<u8> {
        let mut out = vec![if zero_is_black { 0xFF } else { 0 }; stride * height as usize];
        for y in 0..height as usize {
            for x in 0..width as usize {
                let black = pixels[y * width as usize + x] != 0;
                if black != zero_is_black {
                    out[y * stride + x / 8] |= 0x80 >> (x % 8);
                } else {
                    out[y * stride + x / 8] &= !(0x80 >> (x % 8));
                }
            }
        }
        out
    }

    fn encode(image: &BilevelImage<'_>, preset: Preset) -> Vec<u8> {
        let mut encoder = DocumentEncoder::from_options(&EncodeOptions::new(preset)).unwrap();
        encoder.add_page(image).unwrap();
        encoder.finish_file().unwrap()
    }

    /// Checks that the page packed at `stride` in either polarity encodes
    /// to `reference`.
    fn assert_packed_layouts_match(
        pixels: &[u8],
        width: u32,
        height: u32,
        stride: usize,
        preset: Preset,
        reference: &[u8],
    ) {
        for zero_is_black in [false, true] {
            let packed = pack(pixels, width, height, stride, zero_is_black);
            let polarity = if zero_is_black {
                Polarity::ZeroIsBlack
            } else {
                Polarity::OneIsBlack
            };
            let image = BilevelImage::packed(width, height, stride, &packed, polarity).unwrap();
            assert_eq!(
                encode(&image, preset),
                reference,
                "packed stride {stride} {polarity:?}"
            );
        }
    }

    #[test]
    fn every_layout_and_polarity_encodes_identically() {
        let (width, height) = (77u32, 41u32);
        let pixels = page(width, height);
        for preset in [Preset::Default, Preset::Generic] {
            let reference = encode(
                &BilevelImage::bytes(width, height, &pixels).unwrap(),
                preset,
            );
            for stride in [10usize, 13, 16] {
                assert_packed_layouts_match(&pixels, width, height, stride, preset, &reference);
            }
            let inverted: Vec<u8> = pixels.iter().map(|&p| u8::from(p == 0)).collect();
            let image = BilevelImage::new(
                width,
                height,
                &inverted,
                PixelLayout::Bytes {
                    stride: width as usize,
                },
                Polarity::ZeroIsBlack,
            )
            .unwrap();
            assert_eq!(encode(&image, preset), reference);
            let gray: Vec<u8> = pixels
                .iter()
                .map(|&p| if p != 0 { 30 } else { 220 })
                .collect();
            let image = BilevelImage::gray8(width, height, &gray, 128).unwrap();
            assert_eq!(encode(&image, preset), reference);
        }
    }

    #[test]
    fn the_facade_matches_the_legacy_entry_points() {
        let (width, height) = (64u32, 48u32);
        let pixels = page(width, height);
        let image = BilevelImage::bytes(width, height, &pixels).unwrap();
        let legacy = crate::encode_single_image(&pixels, width, height, false).unwrap();
        assert_eq!(encode(&image, Preset::Default), legacy.page_data);

        let legacy = crate::encode_single_image(&pixels, width, height, true).unwrap();
        let mut encoder = DocumentEncoder::new(Jbig2Config::default()).unwrap();
        encoder.add_page(&image).unwrap();
        let split = encoder.finish_pdf().unwrap();
        assert_eq!(split.global_segments, legacy.global_data);
        assert_eq!(split.page_streams, vec![legacy.page_data]);
    }

    #[test]
    fn validation_errors_carry_their_kind() {
        let err = BilevelImage::bytes(0, 4, &[]).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        let err = BilevelImage::packed(9, 2, 1, &[0; 4], Polarity::OneIsBlack).unwrap_err();
        assert_eq!(
            err.kind(),
            ErrorKind::InvalidInput,
            "stride shorter than a row"
        );
        let err = BilevelImage::packed(9, 2, 2, &[0; 3], Polarity::OneIsBlack).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidInput, "buffer too short");
        // The last row need not be padded out to the stride.
        BilevelImage::packed(9, 2, 4, &[0; 6], Polarity::OneIsBlack).unwrap();
        let err = BilevelImage::bytes(16, 2, &[0; 4]).unwrap_err();
        assert!(matches!(err, Jbig2Error::PackedDataDetected));

        let err = EncodeOptions::new(Preset::Generic)
            .with_refine(true)
            .to_config()
            .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidConfig);
        let mut config = Jbig2Config::generic();
        config.refine = true;
        assert_eq!(
            DocumentEncoder::new(config).unwrap_err().kind(),
            ErrorKind::InvalidConfig
        );

        let err = DocumentEncoder::new(Jbig2Config::default())
            .unwrap()
            .finish_file()
            .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
    }

    #[test]
    fn resource_limit_refusals_are_classified() {
        let (width, height) = (64u32, 48u32);
        let pixels = page(width, height);
        let mut config = Jbig2Config::default();
        config.decoder_resource_limits.max_cold_page_work_pixels = Some(1);
        let mut encoder = DocumentEncoder::new(config).unwrap();
        encoder
            .add_page(&BilevelImage::bytes(width, height, &pixels).unwrap())
            .unwrap();
        let err = encoder.finish_pdf().unwrap_err();
        assert_eq!(err.kind(), ErrorKind::ResourceLimit, "{err}");
    }

    #[cfg(feature = "decode")]
    #[test]
    fn decoded_bitmaps_copy_out_in_any_stride_and_polarity() {
        use crate::decode::{DecodeOptions, decode_file};
        let (width, height) = (77u32, 41u32);
        let pixels = page(width, height);
        let file = encode(
            &BilevelImage::bytes(width, height, &pixels).unwrap(),
            Preset::Generic,
        );
        let document = decode_file(&file, &DecodeOptions::default()).unwrap();
        let bitmap = document.first_page().unwrap();
        assert_eq!(bitmap.to_bytes_per_pixel(), pixels);
        for stride in [10usize, 16] {
            for (polarity, zero_is_black) in
                [(Polarity::OneIsBlack, false), (Polarity::ZeroIsBlack, true)]
            {
                let mut out = vec![0u8; stride * height as usize];
                bitmap.copy_packed_rows(&mut out, stride, polarity).unwrap();
                let expected = pack(&pixels, width, height, stride, zero_is_black);
                // Bits past the width are white in the requested polarity;
                // compare only the image bits and the row bytes it occupies.
                let image = BilevelImage::packed(width, height, stride, &out, polarity).unwrap();
                let again =
                    BilevelImage::packed(width, height, stride, &expected, polarity).unwrap();
                assert_eq!(
                    image.to_bit_image().unwrap().packed_words(),
                    again.to_bit_image().unwrap().packed_words()
                );
            }
        }
        let mut short = vec![0u8; 9];
        assert!(
            bitmap
                .copy_packed_rows(&mut short, 10, Polarity::OneIsBlack)
                .is_err()
        );
    }

    #[test]
    fn every_preset_has_a_valid_round_tripping_name() {
        for preset in Preset::ALL {
            assert_eq!(Preset::from_name(preset.name()), Some(preset));
            let result = EncodeOptions::new(preset).to_config();
            if cfg!(feature = "symboldict") {
                result.unwrap();
            }
        }
        assert_eq!(Preset::from_name("nonsense"), None);
    }
}
