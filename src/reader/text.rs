use crate::page_ir::{Rect, SharedBytes};
use crate::text::{GlyphCache, TextPage, TextPageOptions};
use std::cell::RefCell;
use std::sync::Arc;

use crate::reader::session::display_extent;
use crate::reader::{PageGeometry, ReadError, RenderSession};

/// A word and its exact bounding box in source-render pixel space.
#[derive(Debug, Clone, PartialEq)]
pub struct NativeTextWord {
    /// The word's text.
    pub text: String,
    /// `[left, top, right, bottom]` in a top-left-origin pixel space.
    pub bbox: [f32; 4],
}

/// OCR-relevant page classification derived from one semantic compilation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageContentKind {
    /// The page has a usable text layer and no images.
    NativeText,
    /// The page has images and no text: a scan that needs text recognition.
    ScannedImage,
    /// The page has both a usable text layer and images.
    Hybrid,
    /// The text layer is missing, too short or unreliable, and the page is not a
    /// plain scan. Render the page to decide.
    RenderRequired,
}

/// Native text and image evidence used to decide whether OCR is necessary.
#[derive(Debug, Clone, PartialEq)]
pub struct PageTextEvidence {
    /// The classification the fields below add up to.
    pub kind: PageContentKind,
    /// The page's text in reading order.
    pub text: String,
    /// The page's words with their boxes in the requested pixel space.
    pub words: Vec<NativeTextWord>,
    /// Number of non-whitespace characters in `text`.
    pub character_count: usize,
    /// Number of U+FFFD characters in `text`: glyphs whose font gave no Unicode
    /// mapping.
    pub replacement_character_count: usize,
    /// Number of images on the page.
    pub image_count: usize,
    /// Pixel count of the largest image, in its own resolution.
    pub largest_image_pixels: u64,
    /// Whether the text layer can stand in for text recognition: at least eight
    /// characters, at most 1% of them unmapped, and at least two words whose
    /// boxes lie on the page.
    pub trustworthy: bool,
    /// Original DCT payload for a simple one-image scan page. Callers can
    /// decode it directly and avoid a full-page render.
    pub direct_scan: Option<DirectScanImage>,
}

/// The embedded JPEG of a page that is nothing but one full-page image, as
/// stored in the PDF.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectScanImage {
    /// The JPEG file bytes.
    pub jpeg: SharedBytes,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
}

/// Compile a page once and return its OCR routing evidence.
pub fn page_text_evidence(
    session: &RenderSession,
    page: u32,
    source_width: u32,
    source_height: u32,
) -> Result<PageTextEvidence, ReadError> {
    let geometry = session.page_geometry(page)?;
    let semantic = build_semantic_page(session, page)?;
    let text_page = text_page_of(session, &semantic);
    let text = text_page.all_text();
    let words = words_in_pixel_space(&text_page, geometry, source_width, source_height);
    let character_count = text
        .chars()
        .filter(|character| !character.is_whitespace())
        .count();
    let replacement_character_count = text
        .chars()
        .filter(|character| *character == '\u{fffd}')
        .count();
    let valid_word_count = words
        .iter()
        .filter(|word| {
            word.bbox[2] > word.bbox[0]
                && word.bbox[3] > word.bbox[1]
                && word.bbox[0] >= 0.0
                && word.bbox[1] >= 0.0
                && word.bbox[2] <= source_width as f32 + 1.0
                && word.bbox[3] <= source_height as f32 + 1.0
        })
        .count();
    let trustworthy = character_count >= 8
        && replacement_character_count.saturating_mul(100) <= character_count
        && valid_word_count >= 2;
    let image_count = semantic.images.len();
    let largest_image_pixels = semantic
        .images
        .iter()
        .map(|image| u64::from(image.width) * u64::from(image.height))
        .max()
        .unwrap_or(0);
    let direct_scan = direct_scan_image(&semantic);
    let kind = match (trustworthy, image_count > 0, character_count > 0) {
        (true, true, _) => PageContentKind::Hybrid,
        (true, false, _) => PageContentKind::NativeText,
        (false, true, false) => PageContentKind::ScannedImage,
        _ => PageContentKind::RenderRequired,
    };
    Ok(PageTextEvidence {
        kind,
        text,
        words,
        character_count,
        replacement_character_count,
        image_count,
        largest_image_pixels,
        trustworthy,
        direct_scan,
    })
}

fn direct_scan_image(page: &crate::content::SemanticPage) -> Option<DirectScanImage> {
    use crate::content::SemanticOp;
    if page.bounds.rotate != 0 {
        return None;
    }
    // The page must be exactly one placed image: one `Concat`, one `DrawImage`,
    // and nothing else that could paint over or beside it. A second occurrence
    // of either, or any other operation, falls through to the reject arm.
    let mut draw = None;
    let mut placement = None;
    for operation in page.ops.iter() {
        match operation {
            SemanticOp::Save | SemanticOp::Restore => {}
            SemanticOp::Concat(matrix) if placement.is_none() => placement = Some(*matrix),
            SemanticOp::DrawImage(image) if draw.is_none() => draw = Some(*image),
            _ => return None,
        }
    }
    let (Some(draw), Some(matrix)) = (draw, placement) else {
        return None;
    };
    if matrix.b.abs() > 1e-6 || matrix.c.abs() > 1e-6 {
        return None;
    }
    let crop = page.bounds.crop.normalized();
    let covered_width = matrix.a.abs();
    let covered_height = matrix.d.abs();
    if (covered_width - crop.width()).abs() > crop.width().abs() * 0.02
        || (covered_height - crop.height()).abs() > crop.height().abs() * 0.02
    {
        return None;
    }
    let image = page.images.get(draw.index())?;
    if image.is_mask
        || image.smask.is_some()
        || image.mask.is_some()
        || image.bits_per_component != 8
        || image.codec != Some(crate::page_ir::ImageCodecKind::Dct)
        || !matches!(
            image.color_space,
            Some(crate::page_ir::ImageColorSpace::Gray | crate::page_ir::ImageColorSpace::Rgb)
        )
    {
        return None;
    }
    Some(DirectScanImage {
        jpeg: image.codec_data.clone()?,
        width: image.width,
        height: image.height,
    })
}

/// Whether `page` (zero-based) has any extractable text.
///
/// A scanned page without a text layer returns `false`.
///
/// # Errors
///
/// [`ReadError::PageOutOfRange`] or [`ReadError::Compile`].
pub fn has_text_layer(session: &RenderSession, page: u32) -> Result<bool, ReadError> {
    Ok(build_text_page(session, page)?.has_text())
}

/// The text of `page` (zero-based) in reading order.
///
/// Characters the font gives no Unicode mapping for come out as U+FFFD. A
/// page without a text layer returns an empty string. The page is compiled
/// once and shared with rendering, so reading text and then rendering costs
/// one interpretation.
///
/// # Errors
///
/// [`ReadError::PageOutOfRange`] or [`ReadError::Compile`].
///
/// # Examples
///
/// ```no_run
/// # let bytes: std::sync::Arc<[u8]> = std::fs::read("document.pdf")?.into();
/// let session = exprim::RenderSession::open(bytes, None)?;
/// for page in 0..session.page_count() {
///     println!("{}", exprim::page_text(&session, page)?);
/// }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn page_text(session: &RenderSession, page: u32) -> Result<String, ReadError> {
    Ok(build_text_page(session, page)?.all_text())
}

/// The words of `page` with their bounding boxes, mapped onto a rendering of
/// the page that is `source_width` x `source_height` pixels.
///
/// Boxes are `[left, top, right, bottom]` with the origin at the top left,
/// so they can be drawn straight onto a raster of that size (the page's
/// rotation is applied). Each box is the union of the word's glyph boxes.
///
/// # Errors
///
/// [`ReadError::PageOutOfRange`] or [`ReadError::Compile`].
///
/// # Examples
///
/// ```no_run
/// # let bytes: std::sync::Arc<[u8]> = std::fs::read("document.pdf")?.into();
/// let session = exprim::RenderSession::open(bytes, None)?;
/// let (width, height) = session.page_pixel_size(0, 150.0)?;
/// for word in exprim::positioned_words(&session, 0, width, height)? {
///     let [left, top, right, bottom] = word.bbox;
///     println!("{:?} at {left},{top} to {right},{bottom}", word.text);
/// }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn positioned_words(
    session: &RenderSession,
    page: u32,
    source_width: u32,
    source_height: u32,
) -> Result<Vec<NativeTextWord>, ReadError> {
    let geometry = session.page_geometry(page)?;
    let text_page = build_text_page(session, page)?;
    Ok(words_in_pixel_space(
        &text_page,
        geometry,
        source_width,
        source_height,
    ))
}

/// Map a text page's words into source-render pixel space, dropping the
/// whitespace-only entries the word builder can emit.
fn words_in_pixel_space(
    text_page: &TextPage,
    geometry: PageGeometry,
    source_width: u32,
    source_height: u32,
) -> Vec<NativeTextWord> {
    text_page
        .words()
        .into_iter()
        .filter(|word| !word.text.trim().is_empty())
        .map(|word| NativeTextWord {
            text: word.text,
            bbox: rect_to_pixels(
                word.bbox,
                geometry.crop_box,
                geometry.rotate,
                source_width,
                source_height,
            ),
        })
        .collect()
}

fn build_text_page(session: &RenderSession, page: u32) -> Result<TextPage, ReadError> {
    let semantic = build_semantic_page(session, page)?;
    Ok(text_page_of(session, &semantic))
}

thread_local! {
    /// The glyph bounds this thread has worked out, for the session it last
    /// read text from. Without it every page parses its fonts again, which
    /// is most of the cost of reading text from a long document.
    static GLYPHS: RefCell<(Option<u64>, GlyphCache)> = RefCell::new((None, GlyphCache::new()));
}

fn text_page_of(session: &RenderSession, semantic: &crate::content::SemanticPage) -> TextPage {
    GLYPHS.with(|slot| {
        let mut slot = slot.borrow_mut();
        // Another document: let go of the fonts of the previous one.
        if slot.0 != Some(session.id()) {
            *slot = (Some(session.id()), GlyphCache::new());
        }
        TextPage::build_with_cache(semantic, &TextPageOptions::default(), &mut slot.1)
    })
}

fn build_semantic_page(
    session: &RenderSession,
    page: u32,
) -> Result<Arc<crate::content::SemanticPage>, ReadError> {
    Ok(Arc::clone(&session.compile(page)?.semantic))
}

fn rect_to_pixels(
    rect: Rect,
    crop_box: [f64; 4],
    rotate: u16,
    output_width: u32,
    output_height: u32,
) -> [f32; 4] {
    let [x0, y0, x1, y1] = crop_box;
    let crop_width = (x1 - x0).abs().max(f64::EPSILON);
    let crop_height = (y1 - y0).abs().max(f64::EPSILON);
    let rotation = rotate % 360;
    let (display_width, display_height) = display_extent(crop_width, crop_height, rotation);
    let scale_x = output_width as f64 / display_width;
    let scale_y = output_height as f64 / display_height;

    let transform = |x: f64, y: f64| -> (f64, f64) {
        let (device_x, device_y) = match rotation {
            90 => (y - y0, x - x0),
            180 => (x1 - x, y - y0),
            270 => (y1 - y, x1 - x),
            _ => (x - x0, y1 - y),
        };
        (device_x * scale_x, device_y * scale_y)
    };

    // Rotation can send any source corner to any device corner, so take the
    // axis-aligned hull of all four rather than assuming which is which.
    let corners = [
        transform(rect.x0, rect.y0),
        transform(rect.x0, rect.y1),
        transform(rect.x1, rect.y0),
        transform(rect.x1, rect.y1),
    ];
    let (mut left, mut top) = (f64::INFINITY, f64::INFINITY);
    let (mut right, mut bottom) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
    for (x, y) in corners {
        left = left.min(x);
        top = top.min(y);
        right = right.max(x);
        bottom = bottom.max(y);
    }
    [left as f32, top as f32, right as f32, bottom as f32]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_unrotated_pdf_coordinates_to_top_left_pixels() {
        let mapped = rect_to_pixels(
            Rect {
                x0: 10.0,
                y0: 20.0,
                x1: 30.0,
                y1: 40.0,
            },
            [0.0, 0.0, 100.0, 200.0],
            0,
            200,
            400,
        );
        assert_eq!(mapped, [20.0, 320.0, 60.0, 360.0]);
    }

    #[test]
    fn maps_rotated_pdf_coordinates_and_swaps_display_axes() {
        let mapped = rect_to_pixels(
            Rect {
                x0: 10.0,
                y0: 20.0,
                x1: 30.0,
                y1: 40.0,
            },
            [0.0, 0.0, 100.0, 200.0],
            90,
            400,
            200,
        );
        assert_eq!(mapped, [40.0, 20.0, 80.0, 60.0]);
    }
}
