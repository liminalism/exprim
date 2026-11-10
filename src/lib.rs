//! # exprim
//!
//! A read-only, memory-bounded, concurrent PDF engine: open a document,
//! extract its text, and rasterize its pages.
//!
//! Every page renders against an immutable document snapshot; all execution
//! state and scratch memory belong to the worker that renders the page. A
//! malformed page degrades to a typed error for that page, never a panic
//! across the API.
//!
//! The crate root is the document API: [`RenderSession`] opens a document and
//! renders, [`page_text`] / [`positioned_words`] read its text layer,
//! [`extract_outline`] and [`extract_metadata`] read its structure, and
//! [`encode_plane`] exports rendered pages as PNG or JPEG.
//!
//! The engine's layers are public modules so that tools built on them do not
//! need a fork, but they are hidden from the documentation and carry no
//! semver promise before 1.0: a minor release may change them.
//!
//! ## Features
//!
//! - `print`: office printing (imposition, sheet composition, CUPS and
//!   winspool spooling) in [`print`].
//! - `profiling`: per-stage timers.

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
