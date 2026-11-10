use std::sync::Arc;

use crate::doctor::{DocumentReport, OpenOutcome};
use crate::document::{DocumentError, DocumentLimits, DocumentSnapshot};
use crate::security::SecurityError;
use crate::source::{OwnedBytesSource, PdfSource};

/// exprim's own form of the renderer's per-page compile health.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompileStatus {
    /// The page compiled cleanly.
    Ok {
        /// Number of drawing operations on the page.
        operation_count: usize,
    },
    /// The page compiled, with content skipped or approximated.
    Degraded {
        /// Number of drawing operations that were kept.
        operation_count: usize,
        /// What was skipped or approximated.
        detail: String,
    },
    /// The page could not be compiled at all.
    Failed {
        /// Why.
        error: String,
    },
}

/// Read-only facts collected before a document enters the processing path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DocumentIntake {
    /// Number of pages. Zero when the document could not be opened.
    pub page_count: u32,
    /// The document is encrypted.
    pub encrypted: bool,
    /// The document is encrypted and cannot be opened without a password.
    pub needs_password: bool,
    /// The cross-reference table was damaged and was rebuilt by scanning the
    /// file.
    pub recovery_used: bool,
    /// The compile result of every page, in page order.
    pub per_page_compile: Vec<CompileStatus>,
}

/// Examine document bytes without exposing renderer report types.
///
/// `crate::doctor::examine` deliberately never fails. If it reports an open
/// failure for an encrypted file, retry the cheap typed open path so a real
/// password requirement is not confused with an unsupported or damaged file.
///
/// # Examples
///
/// Decide whether to ask for a password before opening:
///
/// ```no_run
/// # let bytes: std::sync::Arc<[u8]> = std::fs::read("document.pdf")?.into();
/// let intake = exprim::examine_document(&bytes);
/// if intake.needs_password {
///     println!("password required");
/// } else {
///     let session = exprim::RenderSession::open(bytes, None)?;
///     println!("{} pages", session.page_count());
/// }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn examine_document(bytes: &Arc<[u8]>) -> DocumentIntake {
    let limits = DocumentLimits::default();
    let report = crate::doctor::examine(source(Arc::clone(bytes)), limits.clone());
    let encrypted = report.encryption.is_some();
    let needs_password = encrypted
        && matches!(&report.open, OpenOutcome::Failed { .. })
        && matches!(
            DocumentSnapshot::open(source(Arc::clone(bytes)), limits),
            Err(DocumentError::Security(SecurityError::PasswordRequired))
        );

    from_report(report, encrypted, needs_password)
}

fn from_report(report: DocumentReport, encrypted: bool, needs_password: bool) -> DocumentIntake {
    let per_page_compile = report
        .pages
        .into_iter()
        .map(|page| match page.compile {
            crate::doctor::CompileStatus::Ok { op_count } => CompileStatus::Ok {
                operation_count: op_count,
            },
            crate::doctor::CompileStatus::Degraded { op_count, detail } => {
                CompileStatus::Degraded {
                    operation_count: op_count,
                    detail,
                }
            }
            crate::doctor::CompileStatus::Failed { error } => CompileStatus::Failed { error },
        })
        .collect::<Vec<_>>();

    DocumentIntake {
        page_count: u32::try_from(per_page_compile.len()).unwrap_or(u32::MAX),
        encrypted,
        needs_password,
        recovery_used: report.xref.recovery_used,
        per_page_compile,
    }
}

pub(crate) fn source(bytes: Arc<[u8]>) -> Arc<dyn PdfSource> {
    Arc::new(OwnedBytesSource::new(bytes))
}
