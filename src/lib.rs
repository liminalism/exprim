//! # exprim
//!
//! A read-only, memory-bounded, concurrent PDF engine: open a document,
//! extract its text, and rasterize its pages.
//!
//! Every page renders against an immutable document snapshot; all execution
//! state and scratch memory belong to the thread that renders the page. A
//! malformed page degrades to a typed error for that page, never a panic
//! across the API.
//!
//! ## Quick start
//!
//! Open a document from bytes, read a page's text, and export the page as a
//! PNG:
//!
//! ```
//! use std::sync::Arc;
//! use exprim::{ExportOptions, RenderSession};
//!
//! # fn one_page_pdf() -> Vec<u8> {
//! #     let content = "BT /F1 24 Tf 20 40 Td (Hello) Tj ET";
//! #     let objects = [
//! #         "<< /Type /Catalog /Pages 2 0 R >>".to_string(),
//! #         "<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_string(),
//! #         "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 100] /Contents 4 0 R \
//! #          /Resources << /Font << /F1 5 0 R >> >> >>"
//! #             .to_string(),
//! #         format!("<< /Length {} >>\nstream\n{content}\nendstream", content.len()),
//! #         "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_string(),
//! #     ];
//! #     let mut pdf = b"%PDF-1.4\n".to_vec();
//! #     let mut offsets = Vec::new();
//! #     for (index, body) in objects.iter().enumerate() {
//! #         offsets.push(pdf.len());
//! #         pdf.extend_from_slice(format!("{} 0 obj\n{body}\nendobj\n", index + 1).as_bytes());
//! #     }
//! #     let xref = pdf.len();
//! #     let size = objects.len() + 1;
//! #     pdf.extend_from_slice(format!("xref\n0 {size}\n0000000000 65535 f \n").as_bytes());
//! #     for offset in offsets {
//! #         pdf.extend_from_slice(format!("{offset:010} 00000 n \n").as_bytes());
//! #     }
//! #     pdf.extend_from_slice(
//! #         format!("trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n").as_bytes(),
//! #     );
//! #     pdf
//! # }
//! // Any PDF in memory; `std::fs::read("document.pdf")?` is the usual source.
//! let bytes: Arc<[u8]> = one_page_pdf().into();
//! let session = RenderSession::open(bytes, None)?;
//! assert_eq!(session.page_count(), 1);
//!
//! // Pages are numbered from zero.
//! let text = exprim::page_text(&session, 0)?;
//! assert!(text.contains("Hello"));
//!
//! // Render and encode in one call: PNG file bytes at 150 DPI.
//! let png = session.export_page(0, &ExportOptions::png(150.0))?;
//! assert!(png.starts_with(b"\x89PNG"));
//! # fn shareable<T: Send + Sync>(_: &T) {}
//! # shareable(&session);
//! # Ok::<(), exprim::ReadError>(())
//! ```
//!
//! ## What is where
//!
//! | To do this | Use |
//! |---|---|
//! | Open a document | [`RenderSession::open`] |
//! | Check a file before opening: encrypted, damaged, per-page health | [`examine_document`] |
//! | Page count, page boxes and rotation | [`RenderSession::page_count`], [`RenderSession::page_geometry`] |
//! | A page as a PNG or JPEG | [`RenderSession::export_page`], [`ExportOptions`] |
//! | A page as raw pixels, at any size or cropped | [`RenderSession::compile`], [`RenderSession::render`], [`RasterProduct`] |
//! | Text of a page | [`page_text`], [`has_text_layer`] |
//! | Words with their positions | [`positioned_words`] |
//! | Whether a page needs text recognition | [`page_text_evidence`] |
//! | Bookmarks, title and author | [`extract_outline`], [`extract_metadata`] |
//! | Encode pixels you already rendered | [`encode_plane`] |
//! | Metrics of a font file | [`read_face_metrics`] |
//! | Print (feature `print`) | the `print` module |
//!
//! ## Pages, units and pixels
//!
//! - Page indices are zero-based everywhere at the crate root. (Page ranges
//!   in the `print` module are one-based, because users type them.)
//! - Page geometry is in PDF points, 1/72 inch, with the origin at the bottom
//!   left. "The page" means its crop box with `/Rotate` applied, which is what
//!   a viewer shows; [`PageGeometry`] carries the other boxes.
//! - A resolution in DPI maps points to pixels: a US Letter page (612 x 792
//!   points) is 1275 x 1650 pixels at 150 DPI.
//!   [`RenderSession::page_pixel_size`] does this arithmetic.
//! - Rasters are 8-bit RGB or grayscale, rows top to bottom, composited over
//!   white, with the page's annotations drawn.
//!
//! ## Rendering more than once
//!
//! [`RenderSession::export_page`] is the whole job in one call. To render a
//! page several times (thumbnail and full size, tiles, a zoomed region),
//! compile it once and render the handle:
//!
//! ```no_run
//! use exprim::{RasterProduct, RenderSession};
//! # let bytes: std::sync::Arc<[u8]> = std::fs::read("document.pdf")?.into();
//!
//! let session = RenderSession::open(bytes, None)?;
//! let page = session.compile(0)?;
//! let thumbnail = session.render(&page, &RasterProduct::gray8(120, 170))?;
//! let full = session.render(&page, &RasterProduct::rgb8(1240, 1754))?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! A [`RasterProduct`] names the output size, the pixel format and an
//! optional [`DeviceCrop`]. The same product gives the same pixels every
//! time, on every thread.
//!
//! ## Errors and damaged files
//!
//! Every fallible call returns [`ReadError`]. Opening fails only when the
//! file as a whole cannot be read; a damaged cross-reference table is rebuilt
//! by scanning. After that, failures are per page: a page whose content
//! cannot be interpreted or rendered returns an error for that page, and the
//! others still work. Bookmark and metadata readers skip what they cannot
//! resolve and return what remains. A panic inside the renderer is caught and
//! reported as [`ReadError::Render`].
//!
//! ## Threads and memory
//!
//! A [`RenderSession`] can be shared between threads. Calls run on the thread
//! that makes them, so rendering pages in parallel is a matter of calling
//! from several threads; a `rayon` iterator over page indices works. Each
//! thread keeps its own scratch buffers and reuses them from page to page.
//!
//! Memory is bounded in three places: a session keeps at most eight compiled
//! pages (see [`RenderSession::open_with_cache_capacity`]), an export refuses
//! images over [`DEFAULT_MAX_EXPORT_PIXELS`] unless told otherwise, and the
//! parser and renderer apply fixed limits to what one document or page may
//! allocate. Long work can be stopped with a [`CancellationToken`].
//!
//! ## Encrypted documents
//!
//! Pass the password to [`RenderSession::open`]. [`examine_document`] reports
//! whether a file is encrypted and whether it needs a password, without
//! failing.
//!
//! ## Features
//!
//! - `print`: office printing (paper, margins, scaling, N-up, booklets,
//!   duplex, page ranges; CUPS and Windows spooling) in the `print` module.
//! - `profiling`: per-stage timers for benchmark tooling.
//!
//! ## Stability
//!
//! The crate root is the supported API. The engine's layers (parser, content
//! interpreter, fonts, rasterizer) are public modules so that tools built on
//! them do not need a fork, but they are hidden from this documentation and
//! carry no semver promise before 1.0: a minor release may change them.

#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

// Engine layers; `docs/ARCHITECTURE.md` lists them leaf first. Nothing below
// `page_ir` imports render types.
#[doc(hidden)]
pub mod color;
#[doc(hidden)]
pub mod content;
#[doc(hidden)]
pub mod doctor;
#[doc(hidden)]
pub mod document;
#[doc(hidden)]
pub mod font;
#[doc(hidden)]
pub mod function;
#[doc(hidden)]
pub mod geom;
#[doc(hidden)]
pub mod image;
#[doc(hidden)]
pub mod object;
#[doc(hidden)]
pub mod page_ir;
#[doc(hidden)]
pub mod postprocess;
#[doc(hidden)]
pub mod profiling;
#[doc(hidden)]
pub mod render_api;
#[doc(hidden)]
pub mod render_cpu;
#[doc(hidden)]
pub mod render_scheduler;
#[doc(hidden)]
pub mod security;
#[doc(hidden)]
pub mod source;
#[doc(hidden)]
pub mod structure;
#[doc(hidden)]
pub mod syntax;
#[doc(hidden)]
pub mod text;

#[cfg(any(test, feature = "test-support"))]
#[doc(hidden)]
pub mod test_support;

mod reader;
pub use reader::*;

#[cfg(feature = "print")]
#[cfg_attr(docsrs, doc(cfg(feature = "print")))]
pub mod print;
