//! Office printing over the exprim reader.
//!
//! This module sits *above* the document API ([`crate::RenderSession`]) and owns everything
//! print-specific: paper, imposition, sheet composition and spooling. The
//! renderer stays a display rasterizer.
//!
//! ```text
//!                       ┌─ pass-through: spool the original PDF bytes
//!   PrintJob ──────────►│  (CUPS, unmodified geometry, no imposition)
//!                       │
//!                       └─ raster: compose sheets ourselves, spool bitmaps
//!                          (Windows always; anywhere we impose or N-up)
//! ```
//!
//! # Printing a document
//!
//! Describe the job with [`PrintOptions`], pick a spooler, and call
//! [`print_document`]:
//!
//! ```no_run
//! use std::sync::Arc;
//! use exprim::print::{self, Duplex, NUp, PaperSize, PrintOptions, spool};
//!
//! let spooler = spool::platform_spooler();
//! let Some(printer) = spooler.default_printer()? else {
//!     return Ok(());
//! };
//! let options = PrintOptions {
//!     paper: PaperSize::A4,
//!     n_up: NUp::Two,
//!     duplex: Duplex::LongEdge,
//!     ..PrintOptions::default()
//! };
//! let bytes: Arc<[u8]> = std::fs::read("document.pdf")?.into();
//! let job = print::print_document(&*spooler, &printer, "document.pdf", bytes, &options)?;
//! println!("job {} took the {} route", job.id, job.route.kind());
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! [`print_document_with`] takes a [`PrintRequest`] for the rest: a document
//! password and explicit composition settings.
//!
//! # The two routes
//!
//! - **Pass-through** spools the original PDF bytes. It is taken when nothing
//!   in the job changes page geometry
//!   ([`PrintOptions::is_pass_through_capable`]) and the queue accepts PDF.
//! - **Composed** imposes the pages onto sheets ([`impose`]), rasterizes each
//!   sheet ([`compose_sheet`]) and spools bitmaps. N-up, booklets, margins,
//!   scaling other than shrink-to-fit or actual size, and every Windows job
//!   take this route.
//!
//! [`route_for`] answers which route a job would take without submitting it.
//!
//! # Previewing without a printer
//!
//! Impose with no hardware margins and render the sheets to PNG:
//!
//! ```no_run
//! use exprim::RenderSession;
//! use exprim::print::{self, Margins, PreviewOptions, PrintJob, PrintOptions};
//! # let bytes: std::sync::Arc<[u8]> = std::fs::read("document.pdf")?.into();
//!
//! let session = RenderSession::open(bytes, None)?;
//! let job = PrintJob::from_session(&session, PrintOptions::default())?;
//! let sheets = print::impose(&job, Margins::ZERO)?;
//! for (index, png) in print::preview_sheets(&session, &sheets, PreviewOptions::default()).enumerate() {
//!     std::fs::write(format!("sheet-{index}.png"), png?)?;
//! }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! To preview what a particular printer will do, plan with its margins
//! instead: [`Spooler::capabilities`], then [`Spooler::plan_sheets`]. Neither
//! submits a job.
//!
//! # Units and conventions
//!
//! Lengths are PostScript points (1/72 inch) with `y` up, as in PDF.
//! [`PageRange`] is one-based and inclusive, because users type it;
//! [`SourcePage::index`] and [`Placement::source_page`] are zero-based.
//!
//! # Platform notes
//!
//! - Linux and macOS print through the CUPS command-line tools.
//! - Windows spooling is experimental: it has not been verified against a
//!   physical printer queue. A Windows job uses one sheet orientation, chosen
//!   from all selected pages; a failed job is aborted, not committed in part.
//! - [`FileSpooler`] writes sheets as PNG files, for tests and for platforms
//!   with no spooler.
//! - Booklets fold along the vertical centre of the sheet. Automatic duplex
//!   needs short-edge binding on landscape sheets and long-edge on portrait
//!   ones; a mismatch is an error before anything is submitted.
//! - Prepress (CMYK separations, output ICC profiles, overprint simulation,
//!   PDF/X) is out of scope.

pub mod compose;
pub mod job;
pub mod layout;
pub mod paper;
pub mod preview;
pub mod spool;

pub use compose::{
    Band, ComposeOptions, SheetRaster, compose_sheet, compose_sheet_banded, sheet_pixel_size,
};
pub use job::{
    PrintRequest, PrintRoute, RouteKind, SubmittedJob, compose_options_for, plan_sheets,
    print_document, print_document_with, route_for, source_pages,
};
pub use layout::{Placement, Sheet, Side, expand_copies, impose, impose_uniform};
pub use paper::{Margins, Orientation, PaperSize, Rect};
pub use preview::{PreviewOptions, preview_sheets, render_preview_png};
pub use spool::{
    DeviceCapabilities, JobId, JobStatus, PrinterId, PrinterInfo, SpoolJob, Spooler,
    file::FileSpooler,
};

use std::fmt;

/// Everything that can go wrong between "the user pressed Print" and "the
/// spooler accepted the job".
#[derive(Debug, thiserror::Error)]
pub enum PrintError {
    /// The job's options do not describe a printable job.
    #[error("invalid print options: {0}")]
    InvalidOptions(String),
    /// The requested page range selects nothing, or names pages the document
    /// does not have.
    #[error("page range selects no pages")]
    EmptyRange,
    /// A sheet's imageable area is empty once hardware and user margins are
    /// applied.
    #[error(
        "imageable area is empty: paper {paper_w:.1}x{paper_h:.1}pt is smaller than its margins"
    )]
    EmptyImageableArea {
        /// Paper width in points.
        paper_w: f64,
        /// Paper height in points.
        paper_h: f64,
    },
    /// Reading or rasterizing the source document failed.
    #[error("document error: {0}")]
    Read(#[from] crate::reader::ReadError),
    /// Encoding a composed sheet or preview failed.
    #[error("failed to encode sheet image: {0}")]
    Encode(String),
    /// A sheet raster would exceed the configured pixel budget.
    #[error("sheet raster too large: {width}x{height} exceeds {max_pixels} pixels")]
    SheetTooLarge {
        /// Width the sheet raster would have, in pixels.
        width: u32,
        /// Height the sheet raster would have, in pixels.
        height: u32,
        /// The budget that was exceeded.
        max_pixels: u64,
    },
    /// The platform spooler rejected the job or is unavailable.
    #[error("spooler error: {0}")]
    Spool(String),
    /// No printer matched, or the system reports none at all.
    #[error("no such printer: {0}")]
    NoSuchPrinter(String),
    /// Filesystem failure in the `file` backend.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    /// The requested capability is not implemented on this platform.
    #[error("unsupported on this platform: {0}")]
    Unsupported(&'static str),
}

/// A 2x3 affine transform, PDF-style: `[a b c d e f]` maps `(x, y)` to
/// `(a*x + c*y + e, b*x + d*y + f)`.
///
/// The print crate carries its own matrix rather than borrowing the
/// renderer's so that [`layout`] stays pure geometry with no renderer
/// dependency.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Matrix {
    /// X scale.
    pub a: f64,
    /// Y shear: how much `x` contributes to the output `y`.
    pub b: f64,
    /// X shear: how much `y` contributes to the output `x`.
    pub c: f64,
    /// Y scale.
    pub d: f64,
    /// X translation.
    pub e: f64,
    /// Y translation.
    pub f: f64,
}

impl Matrix {
    /// The transform that leaves every point where it is.
    pub const IDENTITY: Self = Self {
        a: 1.0,
        b: 0.0,
        c: 0.0,
        d: 1.0,
        e: 0.0,
        f: 0.0,
    };

    /// A matrix from its six coefficients, in PDF order.
    #[must_use]
    pub const fn new(a: f64, b: f64, c: f64, d: f64, e: f64, f: f64) -> Self {
        Self { a, b, c, d, e, f }
    }

    /// A translation by `(tx, ty)`.
    #[must_use]
    pub const fn translate(tx: f64, ty: f64) -> Self {
        Self::new(1.0, 0.0, 0.0, 1.0, tx, ty)
    }

    /// A scale by `sx` horizontally and `sy` vertically, about the origin.
    #[must_use]
    pub const fn scale(sx: f64, sy: f64) -> Self {
        Self::new(sx, 0.0, 0.0, sy, 0.0, 0.0)
    }

    /// Rotation by `degrees` counter-clockwise about the origin. Exact for
    /// multiples of 90, which is all imposition ever needs.
    #[must_use]
    pub fn rotate_degrees(degrees: i32) -> Self {
        match degrees.rem_euclid(360) {
            0 => Self::IDENTITY,
            90 => Self::new(0.0, 1.0, -1.0, 0.0, 0.0, 0.0),
            180 => Self::new(-1.0, 0.0, 0.0, -1.0, 0.0, 0.0),
            270 => Self::new(0.0, -1.0, 1.0, 0.0, 0.0, 0.0),
            other => {
                let radians = f64::from(other) * std::f64::consts::PI / 180.0;
                let (sin, cos) = radians.sin_cos();
                Self::new(cos, sin, -sin, cos, 0.0, 0.0)
            }
        }
    }

    /// `self` then `other` — i.e. `other * self` in column-vector notation.
    #[must_use]
    pub fn then(self, other: Self) -> Self {
        Self {
            a: self.a * other.a + self.b * other.c,
            b: self.a * other.b + self.b * other.d,
            c: self.c * other.a + self.d * other.c,
            d: self.c * other.b + self.d * other.d,
            e: self.e * other.a + self.f * other.c + other.e,
            f: self.e * other.b + self.f * other.d + other.f,
        }
    }

    /// Map the point `(x, y)` through this transform.
    #[must_use]
    pub fn apply(self, x: f64, y: f64) -> (f64, f64) {
        (
            self.a * x + self.c * y + self.e,
            self.b * x + self.d * y + self.f,
        )
    }
}

/// Which PDF page box a job takes as the source extent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PageBoxKind {
    /// The visible page — the default, and what a viewer shows.
    #[default]
    Crop,
    /// The physical sheet the page was authored for.
    Media,
    /// The finished page after trimming. Prepress.
    Trim,
    /// Trim plus bleed allowance. Prepress.
    Bleed,
    /// The meaningful content extent as declared by the author.
    Art,
}

/// How a source page is fitted to the area available for it on the sheet.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Scaling {
    /// 1:1. Content that does not fit is clipped.
    ActualSize,
    /// Scale up or down to fit, preserving aspect ratio.
    FitToPage,
    /// Scale down to fit but never up. The sane default.
    #[default]
    ShrinkToFit,
    /// Cover the imageable area, cropping the overflow.
    FillPage,
    /// An explicit factor, `1.0` being 1:1.
    Percent(f64),
}

/// Sheets per side of paper, or a saddle-stitched booklet.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NUp {
    /// One page per side.
    #[default]
    One,
    /// Two pages per side.
    Two,
    /// Four pages per side, 2 x 2.
    Four,
    /// Six pages per side.
    Six,
    /// Nine pages per side, 3 x 3.
    Nine,
    /// Sixteen pages per side, 4 x 4.
    Sixteen,
    /// Two pages per side, ordered so that folding the stack in half
    /// reproduces the reading order.
    Booklet,
}

impl NUp {
    /// Source pages placed on one *side* of a sheet.
    #[must_use]
    pub const fn per_side(self) -> u32 {
        match self {
            Self::One => 1,
            Self::Two | Self::Booklet => 2,
            Self::Four => 4,
            Self::Six => 6,
            Self::Nine => 9,
            Self::Sixteen => 16,
        }
    }

    /// Cell grid `(columns, rows)` for the *portrait* orientation of the
    /// sheet. Callers swap the pair when the sheet is landscape.
    #[must_use]
    pub const fn grid(self) -> (u32, u32) {
        match self {
            Self::One => (1, 1),
            Self::Two | Self::Booklet => (2, 1),
            Self::Four => (2, 2),
            Self::Six => (3, 2),
            Self::Nine => (3, 3),
            Self::Sixteen => (4, 4),
        }
    }
}

/// The order N-up cells are filled in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NUpOrder {
    /// Left to right, then top to bottom.
    #[default]
    RightThenDown,
    /// Right to left, then top to bottom.
    LeftThenDown,
    /// Top to bottom, then left to right.
    DownThenRight,
    /// Top to bottom, then right to left.
    DownThenLeft,
}

/// Whether both sides of the paper are printed, and which edge the sheet
/// flips about.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Duplex {
    /// One side only.
    #[default]
    None,
    /// Flip about the long edge — the usual "book" binding.
    LongEdge,
    /// Flip about the short edge — "notepad" binding.
    ShortEdge,
}

impl Duplex {
    /// Whether both sides are printed.
    #[must_use]
    pub const fn is_duplex(self) -> bool {
        !matches!(self, Self::None)
    }
}

/// A one-based, inclusive page selection.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum PageRange {
    /// Every page of the document.
    #[default]
    All,
    /// Inclusive one-based spans, in the order the user wrote them.
    Spans(Vec<(u32, u32)>),
    /// Pages 1, 3, 5, and so on.
    Odd,
    /// Pages 2, 4, 6, and so on.
    Even,
}

impl PageRange {
    /// Parse `1,3-5,8-` / `all` / `odd` / `even`. `page_count` resolves an
    /// open-ended span.
    pub fn parse(text: &str, page_count: u32) -> Result<Self, PrintError> {
        let trimmed = text.trim();
        match trimmed.to_ascii_lowercase().as_str() {
            "" | "all" => return Ok(Self::All),
            "odd" => return Ok(Self::Odd),
            "even" => return Ok(Self::Even),
            _ => {}
        }
        let mut spans = Vec::new();
        for piece in trimmed.split(',') {
            let piece = piece.trim();
            if piece.is_empty() {
                continue;
            }
            let (start, end) = match piece.split_once('-') {
                None => {
                    let n = parse_page_number(piece)?;
                    (n, n)
                }
                Some((lo, hi)) => {
                    let lo = lo.trim();
                    let hi = hi.trim();
                    let start = if lo.is_empty() {
                        1
                    } else {
                        parse_page_number(lo)?
                    };
                    let end = if hi.is_empty() {
                        page_count.max(1)
                    } else {
                        parse_page_number(hi)?
                    };
                    (start, end)
                }
            };
            if start > end {
                spans.push((end, start));
            } else {
                spans.push((start, end));
            }
        }
        if spans.is_empty() {
            return Err(PrintError::EmptyRange);
        }
        Ok(Self::Spans(spans))
    }

    /// Zero-based page indices this range selects out of `page_count` pages,
    /// in the order they will print.
    #[must_use]
    pub fn resolve(&self, page_count: u32) -> Vec<u32> {
        match self {
            Self::All => (0..page_count).collect(),
            Self::Odd => (0..page_count).filter(|i| i % 2 == 0).collect(),
            Self::Even => (0..page_count).filter(|i| i % 2 == 1).collect(),
            Self::Spans(spans) => {
                let mut out = Vec::new();
                for &(start, end) in spans {
                    let start = start.max(1);
                    for page in start..=end {
                        if page <= page_count {
                            out.push(page - 1);
                        }
                    }
                }
                out
            }
        }
    }
}

fn parse_page_number(text: &str) -> Result<u32, PrintError> {
    text.parse::<u32>()
        .ok()
        .filter(|&n| n > 0)
        .ok_or_else(|| PrintError::InvalidOptions(format!("bad page number {text:?}")))
}

/// Everything the user chose in the print dialog.
#[derive(Debug, Clone, PartialEq)]
pub struct PrintOptions {
    /// The paper to print on. A4 by default.
    pub paper: PaperSize,
    /// Sheet orientation. Chosen per job to suit the pages by default.
    pub orientation: Orientation,
    /// User margins, measured from the paper edge, in points.
    pub margins: Margins,
    /// How each page is fitted to its area on the sheet.
    pub scaling: Scaling,
    /// Pages per side of paper, or booklet imposition.
    pub n_up: NUp,
    /// The order N-up cells are filled in.
    pub n_up_order: NUpOrder,
    /// Draw a hairline around each N-up cell. Off by default.
    pub n_up_border: bool,
    /// One-sided or two-sided printing.
    pub duplex: Duplex,
    /// Which pages to print.
    pub range: PageRange,
    /// Number of copies, at least 1.
    pub copies: u16,
    /// With several copies, print each copy complete before the next (1 2 3,
    /// 1 2 3) rather than each page in turn (1 1, 2 2, 3 3). On by default.
    pub collate: bool,
    /// Print the selected pages last-to-first.
    pub reverse: bool,
    /// The page box taken as each page's extent. The crop box by default.
    pub source_box: PageBoxKind,
    /// Render in grayscale. The composition path honours this; the
    /// pass-through path forwards it as a spooler option.
    pub grayscale: bool,
}

impl Default for PrintOptions {
    fn default() -> Self {
        Self {
            paper: PaperSize::A4,
            orientation: Orientation::Auto,
            margins: Margins::ZERO,
            scaling: Scaling::ShrinkToFit,
            n_up: NUp::One,
            n_up_order: NUpOrder::RightThenDown,
            n_up_border: false,
            duplex: Duplex::None,
            range: PageRange::All,
            copies: 1,
            collate: true,
            reverse: false,
            source_box: PageBoxKind::Crop,
            grayscale: false,
        }
    }
}

impl PrintOptions {
    /// Reject options that cannot describe a printable job. Cheap, and worth
    /// calling before any rendering happens.
    pub fn validate(&self) -> Result<(), PrintError> {
        if self.copies == 0 {
            return Err(PrintError::InvalidOptions(
                "copies must be at least 1".into(),
            ));
        }
        if let Scaling::Percent(p) = self.scaling
            && (!p.is_finite() || p <= 0.0 || p > 100.0)
        {
            return Err(PrintError::InvalidOptions(format!(
                "scale factor {p} is not in (0, 100]"
            )));
        }
        self.paper.validate()?;
        self.margins.validate()?;
        Ok(())
    }

    /// Whether this job can be handed to the platform spooler as the original
    /// PDF bytes, with no rasterization on our side.
    ///
    /// Copies, collation and duplex are spooler options, so they do not force
    /// composition. Anything that changes page *geometry* does.
    #[must_use]
    pub fn is_pass_through_capable(&self) -> bool {
        self.n_up == NUp::One
            && !self.n_up_border
            && matches!(self.scaling, Scaling::ShrinkToFit | Scaling::ActualSize)
            && self.source_box == PageBoxKind::Crop
            && self.margins == Margins::ZERO
    }
}

/// The source side of a job: page extents in points, straight from
/// the document API.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SourcePage {
    /// Zero-based index in the document.
    pub index: u32,
    /// Width in points, after the page's `/Rotate` is applied.
    pub width: f64,
    /// Height in points, after the page's `/Rotate` is applied.
    pub height: f64,
}

impl SourcePage {
    /// A source page from its zero-based index and displayed size in points.
    #[must_use]
    pub const fn new(index: u32, width: f64, height: f64) -> Self {
        Self {
            index,
            width,
            height,
        }
    }

    /// Whether the page is wider than it is tall.
    #[must_use]
    pub fn is_landscape(self) -> bool {
        self.width > self.height
    }
}

impl fmt::Display for SourcePage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "page {} ({:.1}x{:.1}pt)",
            self.index + 1,
            self.width,
            self.height
        )
    }
}

/// A print job as [`layout::impose`] sees it: source geometry plus options.
#[derive(Debug, Clone, PartialEq)]
pub struct PrintJob {
    /// Every page of the source document, in document order. The page range in
    /// `options` selects from these.
    pub pages: Vec<SourcePage>,
    /// What the user asked for.
    pub options: PrintOptions,
}

impl PrintJob {
    /// A job from page geometry gathered by the caller.
    /// [`from_session`](Self::from_session) gathers it from an open document.
    #[must_use]
    pub fn new(pages: Vec<SourcePage>, options: PrintOptions) -> Self {
        Self { pages, options }
    }

    /// Build a job from an open document, taking every page's geometry from
    /// the box named by `options.source_box`.
    ///
    /// Every box is populated by the document API — Bleed/Trim/Art default to
    /// the CropBox and the CropBox to the MediaBox — so a document that
    /// declares no prepress boxes prints identically whichever box is asked
    /// for.
    pub fn from_session(
        session: &crate::reader::RenderSession,
        options: PrintOptions,
    ) -> Result<Self, PrintError> {
        let count = session.page_count();
        let mut pages = Vec::with_capacity(count as usize);
        for index in 0..count {
            let geometry = session.page_geometry(index)?;
            let (width, height) =
                geometry.display_size_of_box(source_box(geometry, options.source_box));
            pages.push(SourcePage::new(index, width, height));
        }
        Ok(Self { pages, options })
    }
}

/// The page box a [`PageBoxKind`] names, out of a page's geometry.
#[must_use]
fn source_box(geometry: crate::reader::PageGeometry, kind: PageBoxKind) -> [f64; 4] {
    match kind {
        PageBoxKind::Crop => geometry.crop_box,
        PageBoxKind::Media => geometry.media_box,
        PageBoxKind::Trim => geometry.trim_box,
        PageBoxKind::Bleed => geometry.bleed_box,
        PageBoxKind::Art => geometry.art_box,
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::unwrap_used)]

    use super::*;

    #[test]
    fn matrix_composition_matches_manual_application() {
        let m = Matrix::scale(2.0, 3.0).then(Matrix::translate(10.0, -5.0));
        assert_eq!(m.apply(1.0, 1.0), (12.0, -2.0));
    }

    #[test]
    fn rotation_quarter_turns_are_exact() {
        assert_eq!(Matrix::rotate_degrees(90).apply(1.0, 0.0), (0.0, 1.0));
        assert_eq!(Matrix::rotate_degrees(180).apply(1.0, 0.0), (-1.0, 0.0));
        assert_eq!(Matrix::rotate_degrees(270).apply(1.0, 0.0), (0.0, -1.0));
        assert_eq!(Matrix::rotate_degrees(-90).apply(1.0, 0.0), (0.0, -1.0));
    }

    #[test]
    fn page_range_parses_spans_and_keywords() {
        assert_eq!(PageRange::parse("all", 10).unwrap(), PageRange::All);
        assert_eq!(
            PageRange::parse("1,3-5", 10).unwrap().resolve(10),
            vec![0, 2, 3, 4]
        );
        assert_eq!(
            PageRange::parse("8-", 10).unwrap().resolve(10),
            vec![7, 8, 9]
        );
        assert_eq!(
            PageRange::parse("5-3", 10).unwrap().resolve(10),
            vec![2, 3, 4]
        );
        assert_eq!(
            PageRange::parse("odd", 5).unwrap().resolve(5),
            vec![0, 2, 4]
        );
    }

    #[test]
    fn source_box_selects_the_named_box_and_rotates_it() {
        let geometry = crate::reader::PageGeometry {
            crop_box: [0.0, 0.0, 600.0, 800.0],
            media_box: [0.0, 0.0, 612.0, 792.0],
            bleed_box: [10.0, 10.0, 590.0, 790.0],
            trim_box: [20.0, 20.0, 580.0, 780.0],
            art_box: [30.0, 30.0, 570.0, 770.0],
            rotate: 90,
        };
        assert_eq!(source_box(geometry, PageBoxKind::Crop), geometry.crop_box);
        assert_eq!(source_box(geometry, PageBoxKind::Media), geometry.media_box);
        assert_eq!(source_box(geometry, PageBoxKind::Bleed), geometry.bleed_box);
        assert_eq!(source_box(geometry, PageBoxKind::Trim), geometry.trim_box);
        assert_eq!(source_box(geometry, PageBoxKind::Art), geometry.art_box);
        // /Rotate 90 swaps the axes: the 560x760 trim box prints 760x560.
        assert_eq!(
            geometry.display_size_of_box(source_box(geometry, PageBoxKind::Trim)),
            (760.0, 560.0)
        );
    }

    #[test]
    fn page_range_clamps_past_the_end() {
        assert_eq!(
            PageRange::parse("9-99", 10).unwrap().resolve(10),
            vec![8, 9]
        );
    }
}
