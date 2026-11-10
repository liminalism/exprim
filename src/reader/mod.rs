//! The document API: open, examine, render, and read text.
//!
//! Engine types stay behind this seam, so a caller does not couple itself to
//! the renderer's internal layers.

mod export;
mod font_metrics;
mod intake;
mod metadata;
mod outline;
mod page_cache;
mod session;
mod text;

pub use export::{
    DEFAULT_EXPORT_DPI, DEFAULT_MAX_EXPORT_PIXELS, ExportColor, ExportOptions, ImageFormat,
    encode_plane, pixel_size_for_dpi,
};
pub use font_metrics::{FaceBBox, FaceMetrics, read_face_metrics};
pub use intake::{CompileStatus, DocumentIntake, examine_document};
pub use metadata::{DocumentMetadata, extract_metadata};
pub use outline::{OwnedBookmarkNode, extract_outline};
pub use session::{
    AnalysisTarget, BaseTarget, CancellationToken, CompiledDocumentPage, DecodeReference,
    DeviceCrop, GraySuitability, GraySurface, Jbig2StencilPolarity, OcrTarget, PageGeometry,
    PageOutputPlan, PageRasterProducts, PreservedJbig2Smask, RasterFormat, RasterPlane,
    RasterProduct, ReadError, RegionTarget, RenderSession, RenderedRegion, RgbSurface,
};
pub use text::{
    DirectScanImage, NativeTextWord, PageContentKind, PageTextEvidence, has_text_layer, page_text,
    page_text_evidence, positioned_words,
};
