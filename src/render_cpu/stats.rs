//! Per-render performance instrumentation (roadmap §15, CPU renderer).
//!
//! Cheap to collect (counters + a couple of `Instant`s) and returned alongside
//! the rendered page, so iterative performance work has a stable signal to
//! measure against without a profiler in the loop.

use std::time::Duration;

#[derive(Debug, Clone, Default)]
pub struct RenderStats {
    /// The request's cancellation token fired mid-render: execution stopped
    /// at an op boundary and the surface is incomplete. The backend maps
    /// this to `RenderError::Cancelled`.
    pub cancelled: bool,
    /// Request-specific CPU lowering (transform, flatten, cull, classify).
    pub lower: Duration,
    /// Surface allocation + background fill.
    pub prep: Duration,
    /// Coverage rasterization + span compositing.
    pub raster: Duration,
    /// Conversion from the internal RGBA surface to the requested host format,
    /// including transfer into the reference-counted host buffer.
    pub output: Duration,
    /// Total display operations walked (during lowering).
    pub ops_total: u32,
    /// Prepared commands executed.
    pub commands: u32,
    /// Commands that produced paint.
    pub ops_painted: u32,
    /// Commands skipped because the feature is not implemented yet.
    pub ops_skipped: u32,
    /// Edges fed to the coverage kernel.
    pub edges: u64,
    /// Output rows touched by coverage.
    pub rows_rasterized: u64,
    /// Pixels with non-zero coverage generated.
    pub covered_pixels: u64,
    /// Transparency groups composited (offscreen layers).
    pub transparency_groups: u32,
    /// Soft masks derived and applied.
    pub soft_masks: u32,
    /// Peak surface bytes held.
    pub surface_bytes: u64,
    /// Bytes in the final host pixel buffer. During output conversion this can
    /// overlap with the internal surface allocation.
    pub output_bytes: u64,
    /// Image draws dropped because their codec (JPX/DCT/JBIG2/CCITT) could not
    /// decode. A page with `covered_pixels == 0` **and** `degraded_draws > 0`
    /// is a *silent blank* — content PDFium would paint that we dropped — not a
    /// genuinely empty page. Surfaced so differential tooling can flag it
    /// instead of scoring it clean (Workstream B3 / DEFERRED.md corpus item 1).
    pub degraded_draws: u32,
    /// A bounded, human-readable sample of the degradations (codec + reason).
    pub recovery_notes: Vec<String>,
    /// Fine-grained feature-only lowering/codec attribution.
    #[cfg(feature = "profiling")]
    pub(crate) profile: crate::profiling::ProfileReport,
    /// Executor hot-path attribution, flushed into `profile` once per page.
    #[cfg(feature = "profiling")]
    pub(crate) hot: ExecHot,
}

/// Plain-field executor attribution (feature `profiling`). Per-command code
/// adds to these integers; [`ExecHot::flush_into`] names them once per page so
/// no map update runs per draw.
#[cfg(feature = "profiling")]
#[derive(Debug, Clone, Default)]
pub(crate) struct ExecHot {
    /// Non-zero while executing a tiling-pattern cell (nested draws).
    pub(crate) tiling_depth: u32,
    /// Time every draw command, including those nested in tiling cells
    /// (`ProfileDetail::Fill`); otherwise only page-level commands are timed.
    pub(crate) time_nested: bool,
    pub(crate) ns_prebuild_masks: u64,
    pub(crate) ns_run_ops: u64,
    // Draw commands (`paint_command`), inclusive of rasterization + dispatch.
    pub(crate) ns_path: u64,
    pub(crate) ns_opaque_rects: u64,
    pub(crate) n_opaque_rects: u64,
    pub(crate) ns_solid_paths: u64,
    pub(crate) n_solid_paths: u64,
    /// Solid Normal fills painting the kernel's runs directly.
    pub(crate) ns_path_direct: u64,
    pub(crate) n_path_direct: u64,
    /// Fills modulated by a clip mask.
    pub(crate) ns_path_masked: u64,
    pub(crate) n_path_masked: u64,
    /// Fills with a soft mask, stencil, shading, or non-Normal blend.
    pub(crate) ns_path_modulated: u64,
    pub(crate) n_path_modulated: u64,
    /// The subset of `ns_path` spent on draws inside tiling cells.
    pub(crate) ns_path_in_tiling: u64,
    pub(crate) n_path_in_tiling: u64,
    /// Rows the kernel emitted that lay outside the command bounds.
    pub(crate) rows_outside_bounds: u64,
    // Clip masks.
    pub(crate) ns_clip_mask_lazy: u64,
    pub(crate) n_clip_mask_lazy: u64,
    pub(crate) n_clip_mask_prebuilt: u64,
    pub(crate) clip_mask_bytes: u64,
    // Tiling fills (`render_tiling`); `ns_tiling` is inclusive of cell draws.
    pub(crate) ns_tiling: u64,
    pub(crate) n_tiling_fills: u64,
    pub(crate) n_tiles: u64,
    pub(crate) ns_tiling_lower: u64,
    pub(crate) ns_tiling_cells: u64,
    pub(crate) ns_tiling_fmask: u64,
    pub(crate) ns_tiling_composite: u64,
    pub(crate) tiling_cell_ops: u64,
    pub(crate) tiling_cell_points: u64,
    pub(crate) tiling_surface_px: u64,
    /// Pixels of the tiling fills' shape masks with non-zero coverage: the
    /// only pixels of the tiled offscreen that are ever read.
    pub(crate) tiling_mask_px: u64,
    pub(crate) clock_reads: u64,
}

#[cfg(feature = "profiling")]
impl ExecHot {
    pub(crate) fn flush_into(&mut self, out: &mut crate::profiling::ProfileReport) {
        let durations: [(&'static str, u64); 14] = [
            ("render.exec.prebuild_masks", self.ns_prebuild_masks),
            ("render.exec.run_ops", self.ns_run_ops),
            ("render.path", self.ns_path),
            ("render.path.opaque_rects", self.ns_opaque_rects),
            ("render.path.solid_paths", self.ns_solid_paths),
            ("render.path.direct", self.ns_path_direct),
            ("render.path.masked", self.ns_path_masked),
            ("render.path.modulated", self.ns_path_modulated),
            ("render.path.in_tiling", self.ns_path_in_tiling),
            ("render.clip.mask_lazy", self.ns_clip_mask_lazy),
            ("render.tiling", self.ns_tiling),
            ("render.tiling.lower_cell", self.ns_tiling_lower),
            ("render.tiling.cells", self.ns_tiling_cells),
            ("render.tiling.fmask", self.ns_tiling_fmask),
        ];
        for (name, ns) in durations {
            if ns != 0 {
                out.add_duration(name, Duration::from_nanos(ns));
            }
        }
        if self.ns_tiling_composite != 0 {
            out.add_duration(
                "render.tiling.composite",
                Duration::from_nanos(self.ns_tiling_composite),
            );
        }
        let counters: [(&'static str, u64); 17] = [
            ("render.opaque_rects", self.n_opaque_rects),
            ("render.solid_paths", self.n_solid_paths),
            ("render.path.direct.count", self.n_path_direct),
            ("render.path.masked.count", self.n_path_masked),
            ("render.path.modulated.count", self.n_path_modulated),
            ("render.path.in_tiling.count", self.n_path_in_tiling),
            ("render.path.rows_outside_bounds", self.rows_outside_bounds),
            ("render.clip.masks_lazy", self.n_clip_mask_lazy),
            ("render.clip.masks_prebuilt", self.n_clip_mask_prebuilt),
            ("render.clip.mask_bytes", self.clip_mask_bytes),
            ("render.tiling.fills", self.n_tiling_fills),
            ("render.tiling.tiles", self.n_tiles),
            ("render.tiling.cell_ops", self.tiling_cell_ops),
            ("render.tiling.cell_points", self.tiling_cell_points),
            ("render.tiling.surface_px", self.tiling_surface_px),
            ("render.tiling.mask_px", self.tiling_mask_px),
            ("render.exec.clock_reads", self.clock_reads),
        ];
        for (name, count) in counters {
            if count != 0 {
                out.increment(name, count);
            }
        }
        *self = Self::default();
    }
}

impl RenderStats {
    /// Fold a prepared page's lowering diagnostics into these stats. Called for
    /// the top-level page and for every tiling-pattern cell lowered during
    /// execution, so no dropped codec draw is missed.
    pub(crate) fn absorb_diagnostics(
        &mut self,
        diag: &crate::render_cpu::prepared::RenderDiagnostics,
    ) {
        self.degraded_draws += diag.degraded_draws();
        let room = 32usize.saturating_sub(self.recovery_notes.len());
        if room > 0 {
            self.recovery_notes
                .extend(diag.notes().iter().take(room).cloned());
        }
    }

    /// Whether this page is a *silent blank*: an undecodable image dropped its
    /// draw and nothing else painted, so the surface is (near) empty where real
    /// content belonged. The differential/sweep layer treats this as a dropped
    /// page rather than a clean render.
    pub fn is_silent_blank(&self) -> bool {
        self.degraded_draws > 0 && self.covered_pixels == 0
    }

    /// Pages-per-second for this single page (raster stage only), for quick
    /// A/B comparisons in benchmarks.
    pub fn raster_pages_per_sec(&self) -> f64 {
        let s = self.raster.as_secs_f64();
        if s > 0.0 { 1.0 / s } else { f64::INFINITY }
    }
}
