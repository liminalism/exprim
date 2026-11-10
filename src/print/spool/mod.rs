//! Spooler backends.
//!
//! The trait is the whole platform seam: everything above it is portable,
//! everything below it is one operating system's print API.

pub mod file;

#[cfg(any(target_os = "linux", target_os = "macos"))]
pub mod cups;

#[cfg(windows)]
pub mod windows;

use crate::print::paper::Margins;
use crate::print::{ComposeOptions, PrintError, PrintOptions, Sheet};

/// A printer's opaque system identifier — a CUPS queue name, a Windows
/// printer name.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PrinterId(pub String);

impl PrinterId {
    /// An id from a queue name.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    /// The queue name.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for PrinterId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// What the system says about one queue.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrinterInfo {
    /// The identifier to submit jobs with.
    pub id: PrinterId,
    /// Human-readable name, when the system offers one distinct from the id.
    pub description: Option<String>,
    /// Where the system says the printer is, when it says.
    pub location: Option<String>,
    /// Whether this is the system's default printer.
    pub is_default: bool,
    /// Whether the queue is accepting jobs.
    pub accepting_jobs: bool,
}

/// What a device can do, as far as imposition needs to know.
#[derive(Debug, Clone, PartialEq)]
pub struct DeviceCapabilities {
    /// The unprintable border. Falls back to
    /// [`DEFAULT_HARDWARE_MARGIN_PT`](crate::print::paper::DEFAULT_HARDWARE_MARGIN_PT).
    pub hardware_margins: Margins,
    /// Whether the device can print on both sides by itself.
    pub supports_duplex: bool,
    /// Whether the device prints in colour.
    pub supports_color: bool,
    /// Native device resolution in DPI, when known.
    pub resolution_dpi: Option<f64>,
    /// Whether the queue accepts `application/pdf` directly, which is what
    /// makes the pass-through path possible.
    pub accepts_pdf: bool,
}

impl Default for DeviceCapabilities {
    fn default() -> Self {
        Self {
            hardware_margins: Margins::uniform(crate::print::paper::DEFAULT_HARDWARE_MARGIN_PT),
            supports_duplex: false,
            supports_color: true,
            resolution_dpi: None,
            accepts_pdf: false,
        }
    }
}

/// The payload handed to a spooler: either the original document bytes, or
/// sheets we composed.
pub enum SpoolPayload<'a> {
    /// The original PDF, spooled unmodified.
    PassThroughPdf(&'a [u8]),
    /// Sheets to rasterize and spool. Composition is the spooler's call, so
    /// that a backend able to stream bands never holds a whole sheet.
    Sheets {
        /// The open document the sheets draw their pages from.
        session: &'a crate::reader::RenderSession,
        /// The sheets to print, in order.
        sheets: &'a [Sheet],
        /// Resolution and colour settings for composing each sheet.
        compose: ComposeOptions,
    },
}

/// One submission.
pub struct SpoolJob<'a> {
    /// The queue to print on.
    pub printer: PrinterId,
    /// Job title as it appears in the queue.
    pub title: String,
    /// The user's options. Backends read spooler-side settings such as copies
    /// and duplex from here.
    pub options: &'a PrintOptions,
    /// What to print.
    pub payload: SpoolPayload<'a>,
}

// `RenderSession` is not `Debug`, so both of these are written out: a job is
// worth printing in a log line, and the session behind it is not.
impl std::fmt::Debug for SpoolPayload<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PassThroughPdf(bytes) => f
                .debug_struct("PassThroughPdf")
                .field("bytes", &bytes.len())
                .finish(),
            Self::Sheets {
                sheets, compose, ..
            } => f
                .debug_struct("Sheets")
                .field("sheets", &sheets.len())
                .field("compose", compose)
                .finish_non_exhaustive(),
        }
    }
}

impl std::fmt::Debug for SpoolJob<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpoolJob")
            .field("printer", &self.printer)
            .field("title", &self.title)
            .field("payload", &self.payload)
            .finish_non_exhaustive()
    }
}

/// A submitted job's system identifier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JobId(pub String);

impl std::fmt::Display for JobId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Where a submitted job stands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JobStatus {
    /// Queued, not yet printing.
    Pending,
    /// Being printed.
    Processing,
    /// Finished.
    Completed,
    /// Cancelled by a user or by [`Spooler::cancel`].
    Cancelled,
    /// Stopped by the system after an error.
    Aborted,
    /// The backend cannot report on this job — a normal answer for the CLI
    /// spooler once the job has left the queue.
    Unknown,
}

/// The platform seam.
pub trait Spooler: Send + Sync {
    /// Every print queue the system knows.
    fn printers(&self) -> Result<Vec<PrinterInfo>, PrintError>;
    /// The system's default printer, if it has one.
    fn default_printer(&self) -> Result<Option<PrinterId>, PrintError> {
        Ok(self
            .printers()?
            .into_iter()
            .find(|p| p.is_default)
            .map(|p| p.id))
    }
    /// What `printer` can do: margins, duplex, colour, resolution, and whether it
    /// takes PDF directly.
    fn capabilities(&self, printer: &PrinterId) -> Result<DeviceCapabilities, PrintError>;
    /// Plan without submitting. Backends may apply job-wide geometry constraints
    /// and query margins for the requested paper and orientation here. The
    /// default uses the capability snapshot; existing spoolers need no override.
    fn plan_sheets(
        &self,
        _printer: &PrinterId,
        job: &crate::print::PrintJob,
        capabilities: &DeviceCapabilities,
    ) -> Result<Vec<Sheet>, PrintError> {
        crate::print::plan_sheets(job, capabilities)
    }
    /// Send a job to its printer and return the id the system gave it.
    fn submit(&self, job: SpoolJob<'_>) -> Result<JobId, PrintError>;
    /// Ask the system where a submitted job stands.
    fn status(&self, job: &JobId) -> Result<JobStatus, PrintError>;
    /// Ask the system to cancel a submitted job.
    fn cancel(&self, job: &JobId) -> Result<(), PrintError>;
}

/// The spooler for this platform, or the file backend when none exists.
pub fn platform_spooler() -> Box<dyn Spooler> {
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    {
        Box::new(cups::CupsSpooler::new())
    }
    #[cfg(windows)]
    {
        Box::new(windows::WindowsSpooler::new())
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
    {
        Box::new(file::FileSpooler::new(std::env::temp_dir()))
    }
}
