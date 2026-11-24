//! Parallel page pipeline (roadmap §7 Phase 5):
//!
//! ```text
//! DocumentSnapshot → compile pool → bounded CompiledPage queue
//!                  → render pool  → reorder buffer → in-order output
//! ```
//!
//! Separate bounded stages so slow rendering doesn't block parsing, parsing
//! can't run unboundedly ahead, and memory stays bounded. Worker count
//! alone does NOT bound memory — jobs acquire [`MemoryBudget`] permits for
//! their estimated peak bytes before running.
//!
//! The pipeline internals are Phase 5; the *shape* (these types) is fixed
//! now, and the two genuinely load-bearing primitives — the budget and the
//! reorder buffer — are fully implemented and tested.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;

use crate::document::{DocumentSnapshot, PageIndex};
use crate::render_api::{
    CancellationToken, OutputFormat, RenderBackend, RenderError, RenderRequest, RenderedPage,
};

mod budget;
mod dispatch;
mod reorder;

pub use budget::{MemoryBudget, MemoryPermit};
pub use reorder::ReorderBuffer;

/// Scheduler configuration.
#[derive(Debug, Clone)]
pub struct SchedulerOptions {
    /// Page-compilation workers.
    pub compile_workers: usize,
    /// Concurrent render jobs.
    pub render_workers: usize,
    /// Bound on compiled pages waiting for a render slot.
    pub compiled_queue_depth: usize,
    /// Maximum number of pages admitted to the pipeline but not yet emitted.
    ///
    /// This is an ordered sliding window, not a bounded result channel: the
    /// next expected page is always admitted, so a slow early page can still
    /// complete and release later completed surfaces from the reorder buffer.
    ///
    /// The window must comfortably exceed the worker count: every page that
    /// finished out of order holds a slot while it waits in the reorder
    /// buffer, and each such slot is one fewer page the render workers can be
    /// given. With a window close to the worker count a short document never
    /// reaches full occupancy — a 22-page book at twelve workers ran ~40%
    /// slower with a window of 12 than with one of 32. The default therefore
    /// scales with the workers (see [`SchedulerOptions::default`]). Surfaces
    /// held in the reorder buffer are *not* charged to `memory_limit_bytes`
    /// (that covers active renders); they are bounded by this count and, in
    /// bytes, by [`SchedulerOptions::reorder_backlog_bytes`].
    pub reorder_window_depth: usize,
    /// Bound on the bytes of rendered pages that have finished but not yet
    /// been emitted because an earlier page is still outstanding (the reorder
    /// backlog), counted as each page's output surface
    /// (`width * height * bytes_per_pixel` of its request).
    ///
    /// A page that would finish out of order is not started while the
    /// backlog is at or over this limit, unless nothing is waiting at all; the
    /// page the collector needs next is always started. Pages already
    /// rendering when the limit is reached still finish, so the backlog can
    /// exceed it by at most one surface per render worker. A slow `emit`
    /// callback is throttled the same way, since pages it has not consumed
    /// yet stay in the backlog.
    ///
    /// The page-count window still applies; this only adds a byte bound for
    /// large surfaces. `u64::MAX` restores the count-only behaviour. The
    /// default is [`SchedulerOptions::DEFAULT_REORDER_BACKLOG_BYTES`].
    pub reorder_backlog_bytes: u64,
    /// Total bytes all in-flight jobs may hold.
    pub memory_limit_bytes: u64,
}

impl Default for SchedulerOptions {
    fn default() -> Self {
        // One partitioned physical pool (performance advice §12): compile +
        // render workers sum to ≈ the core count rather than 2×cores. Rendering
        // is the heavier stage, so it gets the larger share.
        let cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        let render_workers = (cores * 2 / 3).max(1);
        let compile_workers = cores.saturating_sub(render_workers).max(1);
        Self {
            compile_workers,
            render_workers,
            compiled_queue_depth: 12,
            reorder_window_depth: Self::reorder_window_for(compile_workers, render_workers),
            reorder_backlog_bytes: Self::DEFAULT_REORDER_BACKLOG_BYTES,
            memory_limit_bytes: 4 << 30,
        }
    }
}

impl SchedulerOptions {
    /// Default [`SchedulerOptions::reorder_backlog_bytes`]: 256 MiB.
    ///
    /// Large enough for the regime the page-count window was tuned in: it
    /// holds 32 US Letter surfaces at 2x (7.8 MB each), the window the
    /// 22-page measurement settled on for eight workers, and the whole of a
    /// short document at screen resolution, so those runs never wait on it.
    /// Small enough to matter where the window alone did not: with 7 + 13
    /// workers (window 80) a slow first page held 79 finished pages, 637 MiB
    /// at 2x and 2.6 GB at 300 dpi (33.7 MB per surface); under this limit it
    /// holds about 32 and about 7 of them respectively, plus the pages that
    /// were already rendering. It is a sixteenth of the default memory
    /// budget, which still governs the renders themselves.
    pub const DEFAULT_REORDER_BACKLOG_BYTES: u64 = 256 << 20;

    /// Reorder window that keeps `render_workers` busy on a short document.
    ///
    /// Four slots per worker: a worker's current page, the pages it finished
    /// that wait behind a slower earlier page, and a compiled page ready for
    /// it. Measured on a 22-page book at 2 + 6 workers: window 12 → 35 ms,
    /// 20 → 31 ms, 32 → 23 ms, 48 → no further gain; at 4 + 8 the knee is
    /// the same (24 → 28 ms, 48 → 31 ms). Long documents are unaffected by
    /// the window once it exceeds the worker count. Callers that set the
    /// worker counts themselves should set the window through this too, or
    /// the default 12 silently throttles them.
    pub fn reorder_window_for(compile_workers: usize, render_workers: usize) -> usize {
        (compile_workers.max(1) + render_workers.max(1))
            .saturating_mul(4)
            .max(12)
    }
}

/// A finished page emitted by the pipeline, tagged with its index so
/// downstream stages can rely on ordering.
#[derive(Debug)]
pub struct PipelineOutput {
    pub page: PageIndex,
    pub result: Result<RenderedPage, RenderError>,
    /// Time from pipeline start until this page finished rendering, captured
    /// before in-order delivery can hold it behind an earlier page.
    #[cfg(feature = "profiling")]
    pub completion_latency: std::time::Duration,
}

/// A render worker's result on its way to the collector.
struct Finished {
    page: u32,
    result: Result<RenderedPage, RenderError>,
    /// Surface bytes this result holds while it waits to be emitted.
    held_bytes: u64,
    #[cfg(feature = "profiling")]
    completion_latency: std::time::Duration,
}

/// Bytes of the output surface a request produces, which is what a finished
/// page holds while it waits in the reorder buffer.
fn output_surface_bytes(request: &RenderRequest) -> u64 {
    u64::from(request.output_size.width)
        .saturating_mul(u64::from(request.output_size.height))
        .saturating_mul(request.output_format.bytes_per_pixel() as u64)
}

/// The page-render pipeline. Owns worker pools and stage queues.
#[derive(Debug)]
pub struct RenderScheduler {
    options: SchedulerOptions,
    backend: Arc<dyn RenderBackend>,
    budget: MemoryBudget,
}

impl RenderScheduler {
    pub fn new(backend: Arc<dyn RenderBackend>, options: SchedulerOptions) -> Self {
        let budget = MemoryBudget::new(options.memory_limit_bytes);
        Self {
            options,
            backend,
            budget,
        }
    }

    pub fn options(&self) -> &SchedulerOptions {
        &self.options
    }

    pub fn budget(&self) -> &MemoryBudget {
        &self.budget
    }

    /// Render a contiguous page range, delivering results **in page order** to
    /// `emit` while compiling and rendering out of order internally.
    ///
    /// The pipeline (roadmap §7 Phase 5): a compile-worker set turns pages into
    /// `RenderRequest`s and feeds a **bounded** compiled queue (backpressure —
    /// parsing cannot run unboundedly ahead of rendering); a render-worker set
    /// drains it, reserving a [`MemoryBudget`] permit per job so worker count
    /// alone does not bound memory; a reorder buffer emits in page order.
    ///
    /// Guarantees: deterministic output for any worker counts (each page is
    /// independent — its own parse/render context, the snapshot is immutable,
    /// no document-wide lock); `cancel` stops queued and active jobs (it is
    /// checked before each stage *and* linked with each request's
    /// `limits.cancellation` so either token can stop mid-render). Cancellation
    /// drains the requested range, emitting one result per page in order;
    /// a panic or
    /// error on one page becomes that page's `Err` and never corrupts
    /// another page's job.
    pub fn render_range(
        &self,
        snapshot: &DocumentSnapshot,
        pages: std::ops::Range<u32>,
        make_request: &(
             dyn Fn(&DocumentSnapshot, PageIndex) -> Result<RenderRequest, RenderError> + Sync
         ),
        cancel: Option<&CancellationToken>,
        emit: &mut dyn FnMut(PipelineOutput),
    ) {
        use crossbeam_channel::{bounded, unbounded};

        let start_seq = pages.start as u64;
        let cancelled = || cancel.is_some_and(|t| t.is_cancelled());
        #[cfg(feature = "profiling")]
        let pipeline_start = std::time::Instant::now();

        // Stage 0: page indices to compile. Keep this bounded too: eagerly
        // enqueueing a caller-controlled `u32` range before any worker starts
        // can otherwise consume gigabytes before the actual pipeline runs.
        // The feeder admits up to a full window at once, so size this to the
        // window: a smaller bound only makes the feeder block for nothing.
        let reorder_window = self.options.reorder_window_depth.max(1) as u64;
        let (page_tx, page_rx) =
            bounded::<u32>((reorder_window as usize).max(self.options.compiled_queue_depth.max(1)));

        // Stage 1→2: compiled pages waiting for a render worker, bounded by
        // `compiled_queue_depth` (bounds compiled-page memory). Not a channel:
        // render workers take the lowest page first and hold back pages that
        // would overfill the reorder backlog, which a FIFO cannot do without
        // a worker blocking while it holds a page (see `dispatch`).
        let compile_workers = self.options.compile_workers.max(1);
        let render_workers = self.options.render_workers.max(1);
        let dispatch = dispatch::Dispatch::<Result<RenderRequest, RenderError>>::new(
            start_seq,
            self.options.compiled_queue_depth.max(1),
            self.options.reorder_backlog_bytes,
            compile_workers,
            render_workers,
        );
        let dispatch = &dispatch;
        // Stage 2→collector. Unbounded: the reorder window and the backlog
        // bound limit what can be in it, and a bounded channel here could
        // deadlock with workers that finished ahead of the head.
        let (result_tx, result_rx) = unbounded::<Finished>();
        // Advancing this channel means one in-order output was emitted. The
        // feeder uses it to extend the admission window. It only carries
        // zero-sized completion credits, never rendered surfaces.
        let (emitted_tx, emitted_rx) = unbounded::<()>();

        std::thread::scope(|scope| {
            // Admit an ordered sliding window. This bounds *all* pages that
            // can reach the reorder buffer while reserving its first slot for
            // the page the collector is waiting for. Bounding `result_tx`
            // instead can deadlock when later workers fill it before that
            // early page finishes.
            scope.spawn(move || {
                let mut pages = pages;
                let mut admitted = 0_u64;
                let mut emitted = 0_u64;
                loop {
                    let limit = emitted.saturating_add(reorder_window);
                    while admitted < limit {
                        let Some(n) = pages.next() else {
                            return;
                        };
                        if page_tx.send(n).is_err() {
                            return;
                        }
                        admitted += 1;
                    }
                    match emitted_rx.recv() {
                        Ok(()) => emitted += 1,
                        Err(_) => return,
                    }
                }
            });

            // Compile workers.
            for _ in 0..compile_workers {
                let page_rx = page_rx.clone();
                scope.spawn(move || {
                    let _live = dispatch.compiler();
                    while let Ok(n) = page_rx.recv() {
                        let item = if cancelled() {
                            Err(RenderError::Cancelled)
                        } else {
                            match catch_unwind(AssertUnwindSafe(|| {
                                make_request(snapshot, PageIndex(n))
                            })) {
                                Ok(r) => r,
                                Err(_) => Err(RenderError::Backend(format!(
                                    "compile panicked on page {n}"
                                ))),
                            }
                        };
                        let surface_bytes = match &item {
                            Ok(request) => output_surface_bytes(request),
                            Err(_) => 0,
                        };
                        if !dispatch.push(n, item, surface_bytes) {
                            break;
                        }
                    }
                });
            }
            drop(page_rx);

            // Render workers.
            for _ in 0..render_workers {
                let result_tx = result_tx.clone();
                let budget = self.budget.clone();
                let backend = self.backend.clone();
                scope.spawn(move || {
                    let _live = dispatch.renderer();
                    while let Some(ready) = dispatch.take(&cancelled, cancel.is_some()) {
                        let n = ready.page;
                        let result = match ready.item {
                            Err(e) => Err(e),
                            Ok(request) if cancelled() => {
                                drop(request);
                                Err(RenderError::Cancelled)
                            }
                            Ok(mut request) => {
                                // Preserve both cancellation sources through budget
                                // admission and active backend execution.
                                if let Some(pipeline) = cancel {
                                    request.limits.cancellation =
                                        Some(match &request.limits.cancellation {
                                            Some(local) => local.linked_with(pipeline),
                                            None => pipeline.clone(),
                                        });
                                }
                                render_job(backend.as_ref(), &budget, request)
                            }
                        };
                        // A failed page holds no surface while it waits.
                        let held_bytes = if result.is_ok() {
                            ready.surface_bytes
                        } else {
                            0
                        };
                        dispatch.finished(held_bytes);
                        let sent = result_tx.send(Finished {
                            page: n,
                            result,
                            held_bytes,
                            #[cfg(feature = "profiling")]
                            completion_latency: pipeline_start.elapsed(),
                        });
                        if sent.is_err() {
                            break;
                        }
                    }
                });
            }
            drop(result_tx);

            // Collector (this thread): reorder → in-order emit. If `emit`
            // unwinds, the guard releases every waiting worker and dropping
            // the credit sender (owned here for that reason) ends the feeder,
            // so the scope can join and re-raise the panic instead of hanging.
            let emitted_tx = emitted_tx;
            let _collector = dispatch.collector();
            let mut reorder = ReorderBuffer::new(start_seq);
            let mut deliver = |reorder: &mut ReorderBuffer<(PipelineOutput, u64)>| {
                for (seq, (out, held_bytes)) in reorder.drain_ready() {
                    emit(out);
                    // The surface is the caller's now (or dropped): it no
                    // longer counts against the backlog.
                    dispatch.emitted(seq.saturating_add(1), held_bytes);
                    let _ = emitted_tx.send(());
                }
            };
            while let Ok(done) = result_rx.recv() {
                reorder.insert(
                    u64::from(done.page),
                    (
                        PipelineOutput {
                            page: PageIndex(done.page),
                            result: done.result,
                            #[cfg(feature = "profiling")]
                            completion_latency: done.completion_latency,
                        },
                        done.held_bytes,
                    ),
                );
                deliver(&mut reorder);
            }
            deliver(&mut reorder);
        });
    }

    /// Run the existing pipeline while collecting operation-level throughput
    /// attribution. Per-page compilation/render reports remain owned by the
    /// caller's profiled request builder/backend; this method records the
    /// pipeline envelope without introducing shared counters into hot workers.
    #[cfg(feature = "profiling")]
    pub fn render_range_profiled(
        &self,
        snapshot: &DocumentSnapshot,
        pages: std::ops::Range<u32>,
        make_request: &(
             dyn Fn(&DocumentSnapshot, PageIndex) -> Result<RenderRequest, RenderError> + Sync
         ),
        cancel: Option<&CancellationToken>,
        emit: &mut dyn FnMut(PipelineOutput),
    ) -> crate::profiling::ProfileReport {
        let requested = pages.end.saturating_sub(pages.start) as u64;
        let start = std::time::Instant::now();
        let mut completed = 0u64;
        let mut succeeded = 0u64;
        let mut completion_ns = Vec::with_capacity(requested as usize);
        self.render_range(snapshot, pages, make_request, cancel, &mut |out| {
            completed += 1;
            if out.result.is_ok() {
                succeeded += 1;
            }
            completion_ns.push(out.completion_latency.as_nanos() as u64);
            emit(out);
        });
        completion_ns.sort_unstable();
        let mut profile = crate::profiling::ProfileReport::new();
        profile.add_duration("scheduler.total", start.elapsed());
        profile.increment("scheduler.requested_pages", requested);
        profile.increment("scheduler.completed_pages", completed);
        profile.increment("scheduler.succeeded_pages", succeeded);
        profile.increment(
            "scheduler.compile_workers",
            self.options.compile_workers as u64,
        );
        profile.increment(
            "scheduler.render_workers",
            self.options.render_workers as u64,
        );
        profile.increment(
            "scheduler.memory_limit_bytes",
            self.options.memory_limit_bytes,
        );
        if !completion_ns.is_empty() {
            profile.increment(
                "scheduler.page_latency_p50_ns",
                percentile(&completion_ns, 0.50),
            );
            profile.increment(
                "scheduler.page_latency_p90_ns",
                percentile(&completion_ns, 0.90),
            );
            profile.increment(
                "scheduler.page_latency_p99_ns",
                percentile(&completion_ns, 0.99),
            );
        }
        profile
    }
}

#[cfg(feature = "profiling")]
fn percentile(sorted: &[u64], quantile: f64) -> u64 {
    let index = ((sorted.len().saturating_sub(1)) as f64 * quantile).ceil() as usize;
    sorted[index.min(sorted.len() - 1)]
}

/// Reserve budget for a job's estimated peak, render it, and release on
/// return. A panic in the backend becomes an isolated `Err` for this page.
fn render_job(
    backend: &dyn RenderBackend,
    budget: &MemoryBudget,
    request: RenderRequest,
) -> Result<RenderedPage, RenderError> {
    // An estimate above the whole budget would fail the page outright with
    // `LimitExceeded`, though the estimate is a deliberately pessimistic
    // upper bound and the page may well fit. Clamp it instead: the job then
    // waits until nothing else holds the budget and runs alone.
    let working = backend.working_surface_bytes(&request);
    let estimate = estimate_job_bytes_on(&request, working).min(budget.limit());
    // Wait for budget *cancellably*: a job parked here is woken only by
    // another job's release, so an uncancellable wait would ignore the
    // pipeline token until unrelated work finished.
    let cancellation = request.limits.cancellation.clone();
    let permit = budget
        .acquire_cancellable(estimate, || {
            cancellation.as_ref().is_some_and(|t| t.is_cancelled())
        })
        .map_err(|_| RenderError::LimitExceeded("memory budget"))?;
    let Some(_permit) = permit else {
        return Err(RenderError::Cancelled);
    };
    match catch_unwind(AssertUnwindSafe(|| {
        let ticket = backend
            .submit(request)
            .map_err(|e| RenderError::Backend(e.to_string()))?;
        ticket.wait()
    })) {
        Ok(result) => result,
        // Same typed panic taxonomy as the API boundary
        // (`crate::render_api::render_blocking`): the payload message survives.
        Err(payload) => Err(RenderError::Panic {
            message: crate::render_api::panic_message(payload),
        }),
    }
}

/// Peak-byte estimate for budgeting. One outer permit owns the output surface,
/// the transparency-group intermediates and the page's image decodes. Decode
/// paths do not acquire a nested scheduler permit, so an exact-fit memory
/// limit always makes forward progress without self-deadlock.
///
/// - Surface: the CPU backend always paints into a premultiplied RGBA
///   surface; any other output format is a second plane converted from it
///   while it is still alive (`Surface::into_output_recycling`), so both are
///   charged. This function charges the whole page's surface;
///   [`estimate_job_bytes_on`] takes the plane a backend really paints on,
///   which for a large Gray8 page on the CPU backend is one strip
///   ([`RenderBackend::working_surface_bytes`]).
/// - Groups and soft masks: [`PageComplexity::offscreen_bytes`], the page's
///   user-space peaks scaled by the area factor `|a*d - b*c|` of the request
///   transform and capped by the working surface, because a group's
///   offscreen and a mask's planes are sized from painted device bounds,
///   which never exceed the surface painted on.
/// - Tiling fills: [`PageComplexity::tiling_bytes`], the largest one, capped
///   by the page rather than the working surface.
/// - Images: [`PageComplexity::estimated_image_decode_peak_bytes`], at source
///   resolution and independent of the transform.
/// - Clip masks: one output-sized coverage plane per non-rectangular clip
///   ([`PageComplexity::path_clip_count`]), capped at the backend's mask
///   budget.
///
/// The result can exceed the scheduler's whole budget; `render_job` clamps it
/// so such a page runs alone rather than failing.
///
/// [`PageComplexity::estimated_peak_bytes`]: crate::page_ir::PageComplexity::estimated_peak_bytes
/// [`PageComplexity::estimated_image_decode_peak_bytes`]: crate::page_ir::PageComplexity::estimated_image_decode_peak_bytes
pub fn estimate_job_bytes(request: &RenderRequest) -> u64 {
    let pixels =
        u64::from(request.output_size.width).saturating_mul(u64::from(request.output_size.height));
    estimate_job_bytes_on(request, pixels.saturating_mul(4))
}

/// [`estimate_job_bytes`] for a backend that paints `request` on a working
/// plane of `working_bytes` instead of on the whole page's RGBA surface: a
/// page rendered strip by strip into its output holds one strip beside it,
/// and its group offscreens are no larger than the strip. A working plane
/// larger than the page's surface is taken as the surface.
pub fn estimate_job_bytes_on(request: &RenderRequest, working_bytes: u64) -> u64 {
    let pixels =
        u64::from(request.output_size.width).saturating_mul(u64::from(request.output_size.height));
    let rgba = pixels.saturating_mul(4);
    let working = working_bytes.min(rgba);
    let surface = match request.output_format {
        // Painted whole, the surface is the output. Painted in strips, each
        // strip is copied into the output: both are held.
        OutputFormat::Rgba8PremultipliedSrgb if working < rgba => rgba.saturating_add(working),
        OutputFormat::Rgba8PremultipliedSrgb => rgba,
        other => working.saturating_add(pixels.saturating_mul(other.bytes_per_pixel() as u64)),
    };
    let complexity = &request.page.complexity;
    let groups = complexity
        .offscreen_bytes(&request.transform.matrix, working)
        .saturating_add(complexity.tiling_bytes(&request.transform.matrix, rgba));
    // Clip masks: one byte per output pixel each at most, and never more
    // resident at once than the backend's mask budget (eight page-sized
    // planes, at least 64 MiB; `render_cpu::mask`).
    let masks = pixels
        .saturating_mul(u64::from(complexity.path_clip_count))
        .min(pixels.saturating_mul(8).max(64 << 20));
    surface
        .saturating_add(groups)
        .saturating_add(masks)
        .saturating_add(
            request
                .page
                .estimated_image_decode_bytes(&request.transform.matrix),
        )
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
    use super::*;
    use crate::page_ir::{DeviceSize, Matrix, PageBounds, PageComplexity, Rect};
    use crate::render_api::{
        AnnotationMode, Background, OutputFormat, OutputResidency, PageTransform,
        RenderColorPolicy, RenderLimits, RenderQuality,
    };

    const fn assert_send_sync<T: Send + Sync>() {}
    const _: () = assert_send_sync::<RenderScheduler>();

    fn request_with_complexity(complexity: PageComplexity) -> RenderRequest {
        let mut page = crate::page_ir::CompiledPage::empty(PageBounds {
            crop: Rect {
                x0: 0.0,
                y0: 0.0,
                x1: 1.0,
                y1: 1.0,
            },
            rotate: 0,
        });
        page.complexity = complexity;
        RenderRequest {
            page: Arc::new(page),
            transform: PageTransform {
                matrix: Matrix::IDENTITY,
            },
            crop: None,
            output_size: DeviceSize {
                width: 2,
                height: 3,
            },
            output_format: OutputFormat::Rgba8PremultipliedSrgb,
            background: Background::White,
            annotations: AnnotationMode::None,
            color_policy: RenderColorPolicy::Original,
            quality: RenderQuality::Normal,
            limits: RenderLimits::default(),
            residency: OutputResidency::HostRequired,
        }
    }

    #[test]
    fn job_estimate_includes_output_groups_and_largest_cold_image() {
        // Groups are capped at two RGBA surfaces per group, so the group term
        // needs a group count to be charged at all; 7 is well under the cap.
        let request = request_with_complexity(PageComplexity {
            estimated_peak_bytes: 7,
            estimated_image_decode_peak_bytes: 11,
            transparency_group_count: 1,
            ..PageComplexity::default()
        });
        assert_eq!(estimate_job_bytes(&request), 2 * 3 * 4 + 7 + 11);
    }

    #[test]
    fn job_estimate_scales_groups_by_the_transform_area() {
        // 300 dpi: user space is scaled by 300/72 on each axis, so a group's
        // area grows by (300/72)^2 ≈ 17.4.
        let mut request = request_with_complexity(PageComplexity {
            estimated_peak_bytes: 1_000,
            transparency_group_count: 1,
            ..PageComplexity::default()
        });
        let scale = 300.0 / 72.0;
        request.transform.matrix = Matrix {
            a: scale,
            b: 0.0,
            c: 0.0,
            d: -scale,
            e: 0.0,
            f: 0.0,
        };
        request.output_size = DeviceSize {
            width: 1_000,
            height: 1_000,
        };
        let surface = 1_000 * 1_000 * 4;
        let groups = (1_000.0 * (scale * scale)).ceil() as u64;
        assert_eq!(estimate_job_bytes(&request), surface + groups);
        // A rotation does not change the area factor.
        request.transform.matrix = Matrix {
            a: 0.0,
            b: scale,
            c: scale,
            d: 0.0,
            e: 0.0,
            f: 0.0,
        };
        assert_eq!(estimate_job_bytes(&request), surface + groups);
    }

    #[test]
    fn job_estimate_caps_groups_at_two_surfaces_per_group() {
        // A form whose /BBox is far larger than the page cannot allocate more
        // than the page surface per group (twice for knockout).
        let mut request = request_with_complexity(PageComplexity {
            estimated_peak_bytes: u64::MAX / 2,
            transparency_group_count: 3,
            ..PageComplexity::default()
        });
        request.output_size = DeviceSize {
            width: 10,
            height: 10,
        };
        assert_eq!(estimate_job_bytes(&request), 400 + 3 * 2 * 400);
    }

    #[test]
    fn gray_output_charges_the_rgba_surface_it_is_converted_from() {
        let mut request = request_with_complexity(PageComplexity::default());
        request.output_format = OutputFormat::Gray8;
        // 2x3 RGBA surface plus the 2x3 gray plane converted from it.
        assert_eq!(estimate_job_bytes(&request), 2 * 3 * 4 + 2 * 3);
    }

    /// What the CPU backend reports as its working plane is what a render of
    /// the same request really paints on: one strip for a Gray8 page large
    /// enough to be rendered in strips, the whole surface otherwise.
    #[test]
    fn the_cpu_backend_reports_the_plane_its_render_paints_on() {
        let mut request = request_with_complexity(PageComplexity::default());
        request.output_size = DeviceSize {
            width: 1_000,
            height: 4_000,
        };
        let whole = 1_000 * 4_000 * 4u64;
        let backend = crate::render_cpu::CpuBackend::default().with_strip_target_bytes(4 << 20);
        let check = |backend: &crate::render_cpu::CpuBackend, request: &RenderRequest, bytes| {
            assert_eq!(backend.working_surface_bytes(request), bytes);
            let (_, stats) = backend.render_to_host(request).unwrap();
            assert_eq!(stats.surface_bytes, bytes);
        };
        // An RGBA page is its own surface.
        check(&backend, &request, whole);
        // 16 MB against a 4 MiB target: four strips of 1,000 rows.
        request.output_format = OutputFormat::Gray8;
        check(&backend, &request, 1_000 * 1_000 * 4);
        assert_eq!(
            estimate_job_bytes_on(&request, backend.working_surface_bytes(&request)),
            1_000 * 1_000 * 4 + 1_000 * 4_000
        );
        // Strips switched off, and a page under twice the target.
        let one_pass = crate::render_cpu::CpuBackend::default().with_strip_target_bytes(0);
        check(&one_pass, &request, whole);
        request.output_size.height = 2_000;
        check(&backend, &request, whole / 2);
    }

    #[test]
    fn oversized_estimate_runs_alone_instead_of_failing() {
        #[derive(Debug)]
        struct Blank;
        impl RenderBackend for Blank {
            fn id(&self) -> crate::render_api::BackendId {
                crate::render_api::BackendId::Other(9)
            }
            fn capabilities(&self) -> crate::render_api::BackendCapabilities {
                crate::render_api::BackendCapabilities {
                    formats: vec![OutputFormat::Rgba8PremultipliedSrgb],
                    max_surface: DeviceSize {
                        width: 1 << 16,
                        height: 1 << 16,
                    },
                    features: crate::page_ir::PageFeatures::all(),
                    resident_surfaces: false,
                    postprocess: crate::render_api::PostprocessCapabilities::NONE,
                }
            }
            fn supports(
                &self,
                _: &crate::page_ir::CompiledPage,
                _: &RenderRequest,
            ) -> crate::render_api::SupportLevel {
                crate::render_api::SupportLevel::Native
            }
            fn submit(
                &self,
                request: RenderRequest,
            ) -> Result<crate::render_api::RenderTicket, crate::render_api::SubmitError>
            {
                let (ticket, tx) = crate::render_api::RenderTicket::new(0);
                let w = request.output_size.width;
                let _ = tx.send(Ok(RenderedPage::Host(crate::render_api::HostPage {
                    width: w,
                    height: 1,
                    stride: w as usize * 4,
                    format: OutputFormat::Rgba8PremultipliedSrgb,
                    pixels: Arc::from(vec![0u8; w as usize * 4]),
                })));
                Ok(ticket)
            }
        }
        let request = request_with_complexity(PageComplexity {
            estimated_image_decode_peak_bytes: 1 << 40,
            ..PageComplexity::default()
        });
        let budget = MemoryBudget::new(1 << 20);
        assert!(estimate_job_bytes(&request) > budget.limit());
        let result = render_job(&Blank, &budget, request);
        assert!(
            result.is_ok(),
            "clamped to the budget, not refused: {result:?}"
        );
        assert_eq!(budget.in_use(), 0);
    }

    #[test]
    fn exact_fit_outer_permit_covers_decode_without_nested_deadlock() {
        let request = request_with_complexity(PageComplexity {
            estimated_peak_bytes: 1,
            estimated_image_decode_peak_bytes: 2,
            transparency_group_count: 1,
            ..PageComplexity::default()
        });
        let estimate = estimate_job_bytes(&request);
        let budget = MemoryBudget::new(estimate);
        let permit = budget.acquire(estimate).expect("exact-fit job is admitted");
        assert_eq!(permit.bytes(), estimate);
        assert_eq!(budget.in_use(), estimate);
        drop(permit);
        assert_eq!(budget.in_use(), 0);
    }

    #[test]
    fn job_estimate_charges_clip_masks_up_to_the_backend_budget() {
        let with_clips = |count: u32| {
            let mut request = request_with_complexity(PageComplexity {
                path_clip_count: count,
                ..PageComplexity::default()
            });
            request.output_size = DeviceSize {
                width: 1_000,
                height: 1_000,
            };
            estimate_job_bytes(&request)
        };
        let surface = 4_000_000;
        assert_eq!(with_clips(0), surface);
        // One coverage byte per pixel for each path clip...
        assert_eq!(with_clips(3), surface + 3_000_000);
        // ...but no more than the masks the backend keeps resident at once.
        assert_eq!(with_clips(10_000), surface + (64 << 20));
    }

    #[test]
    fn job_estimate_saturates_instead_of_wrapping() {
        let request = request_with_complexity(PageComplexity {
            estimated_peak_bytes: u64::MAX,
            estimated_image_decode_peak_bytes: u64::MAX,
            ..PageComplexity::default()
        });
        assert_eq!(estimate_job_bytes(&request), u64::MAX);
    }

    #[test]
    fn job_estimate_scales_dct_decode_by_request_transform() {
        use crate::page_ir::{
            Color, DisplayOp, ImageCodecKind, ImageColorSpace, ImageId, ImageIr, InterpolationMode,
            Paint, PaintId, ResourceKey, SharedBytes,
        };
        let (native_w, native_h) = (4_000u32, 3_000u32);
        let mut page = crate::page_ir::CompiledPage::empty(PageBounds {
            crop: Rect {
                x0: 0.0,
                y0: 0.0,
                x1: 100.0,
                y1: 75.0,
            },
            rotate: 0,
        });
        let img = ImageIr {
            key: ResourceKey {
                object_number: 5,
                generation: 0,
                variant: 0,
            },
            width: native_w,
            height: native_h,
            is_stencil: false,
            interpolation: InterpolationMode::Bilinear,
            soft_mask: None,
            bits_per_component: 8,
            color_space: ImageColorSpace::Rgb,
            decode: None,
            samples: None,
            codec: Some(ImageCodecKind::Dct),
            codec_data: Some(SharedBytes::from(&b"jpeg"[..])),
            codec_parms: None,
            smask: None,
            mask: None,
            smask_in_data: 0,
            lowering_degraded: false,
        };
        let (full_ret, full_work) = crate::page_ir::estimate_image_bytes_for_target(&img, None);
        page.images = Arc::from([img]);
        page.paints = Arc::from([Paint::Solid(Color::BLACK)]);
        page.operations = Arc::from([
            DisplayOp::ConcatTransform(Matrix {
                a: 100.0,
                b: 0.0,
                c: 0.0,
                d: 75.0,
                e: 0.0,
                f: 0.0,
            }),
            DisplayOp::DrawImage {
                image: ImageId(0),
                paint: PaintId(0),
                transform: Matrix::IDENTITY,
                alpha: 1.0,
                blend: crate::page_ir::BlendMode::Normal,
            },
        ]);
        page.complexity.estimated_image_decode_peak_bytes = full_ret + full_work;
        let page = Arc::new(page);

        // Thumbnail / 1x request: 100x75 footprint reduces 4000x3000 DCT by 1/8
        // to 500x375, saving ~98% of retained raster memory vs native 180 MB.
        let thumb_req = RenderRequest {
            page: Arc::clone(&page),
            transform: PageTransform {
                matrix: Matrix::IDENTITY,
            },
            crop: None,
            output_size: DeviceSize {
                width: 100,
                height: 75,
            },
            output_format: OutputFormat::Rgba8PremultipliedSrgb,
            background: Background::White,
            annotations: AnnotationMode::None,
            color_policy: RenderColorPolicy::Original,
            quality: RenderQuality::Normal,
            limits: RenderLimits::default(),
            residency: OutputResidency::HostRequired,
        };
        let full_req = RenderRequest {
            transform: PageTransform {
                matrix: Matrix {
                    a: 40.0,
                    b: 0.0,
                    c: 0.0,
                    d: 40.0,
                    e: 0.0,
                    f: 0.0,
                },
            },
            output_size: DeviceSize {
                width: 4_000,
                height: 3_000,
            },
            ..thumb_req.clone()
        };
        let thumb_img_bytes = page.estimated_image_decode_bytes(&thumb_req.transform.matrix);
        let full_img_bytes = page.estimated_image_decode_bytes(&full_req.transform.matrix);
        assert_eq!(full_img_bytes, 4_000 * 3_000 * 15);
        // Reduced 500x375 RGB8 retained + full-coeff progressive floor (2 B/sample = 72 MB),
        // strictly far below full 180 MB.
        assert_eq!(thumb_img_bytes, 500 * 375 * 3 + 4_000 * 3_000 * 3 * 2);
        assert!(estimate_job_bytes(&thumb_req) < full_img_bytes);
    }
}
