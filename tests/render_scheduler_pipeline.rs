#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // test code: panics are the assertion mechanism

//! Phase 5 exit gate (roadmap §7 Phase 5): pages compile and render
//! concurrently through the two-stage pipeline, emit in page order, are
//! deterministic across worker counts, are cancellable, and one page's failure
//! never corrupts another's.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use exprim::document::{DocumentLimits, DocumentSnapshot, PageIndex, ParseContext};
use exprim::page_ir::{DeviceSize, Matrix};
use exprim::render_api::{
    AnnotationMode, BackendCapabilities, BackendId, Background, CancellationToken, HostPage,
    OutputFormat, OutputResidency, PageTransform, PostprocessCapabilities, RenderBackend,
    RenderError, RenderLimits, RenderQuality, RenderRequest, RenderTicket, RenderedPage,
    SubmitError, SupportLevel,
};
use exprim::render_scheduler::{PipelineOutput, RenderScheduler, SchedulerOptions};
use exprim::source::{OwnedBytesSource, PdfSource};
use exprim::test_support::builder;

/// A backend that renders synchronously (like the real CPU backend does inside
/// the scheduler's render workers), sleeps to force overlap, tracks the peak
/// concurrent render count, and echoes the request's output width so per-page
/// output identity is checkable.
#[derive(Debug, Default)]
struct MockBackend {
    concurrent: AtomicUsize,
    peak: AtomicUsize,
    sleep: Duration,
    /// Width of the deliberately slow page used by the reorder-window test.
    slow_width: Option<u32>,
    /// Optional duration for the deliberately slow page. Keeping this
    /// separate from `sleep` lets the remaining pages complete while it is
    /// still in flight.
    slow_sleep: Option<Duration>,
    slow_started: AtomicBool,
    slow_finished: AtomicBool,
    /// Fast pages that reached the backend while the slow first page was
    /// outstanding. This measures completed-surface admission, not worker
    /// capacity.
    fast_started_before_slow_finished: AtomicUsize,
    /// Panic when the request output width equals this (isolation test).
    panic_width: Option<u32>,
    /// Requests that arrived carrying a cancellation token.
    saw_token: AtomicUsize,
}

struct ConcGuard<'a>(&'a AtomicUsize);
impl Drop for ConcGuard<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

impl RenderBackend for MockBackend {
    fn id(&self) -> BackendId {
        BackendId::Other(1)
    }
    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            formats: vec![OutputFormat::Rgba8PremultipliedSrgb],
            max_surface: DeviceSize {
                width: 1 << 16,
                height: 1 << 16,
            },
            features: exprim::page_ir::PageFeatures::all(),
            resident_surfaces: false,
            postprocess: PostprocessCapabilities::NONE,
        }
    }
    fn supports(
        &self,
        _page: &exprim::page_ir::CompiledPage,
        _req: &RenderRequest,
    ) -> SupportLevel {
        SupportLevel::Native
    }
    fn submit(&self, request: RenderRequest) -> Result<RenderTicket, SubmitError> {
        if request.limits.cancellation.is_some() {
            self.saw_token.fetch_add(1, Ordering::SeqCst);
        }
        let (ticket, tx) = RenderTicket::new(0);
        let c = self.concurrent.fetch_add(1, Ordering::SeqCst) + 1;
        self.peak.fetch_max(c, Ordering::SeqCst);
        let _guard = ConcGuard(&self.concurrent); // decrements even on panic
        let w = request.output_size.width;
        if self.slow_width == Some(w) {
            self.slow_started.store(true, Ordering::SeqCst);
            std::thread::sleep(self.slow_sleep.unwrap_or(self.sleep));
            self.slow_finished.store(true, Ordering::SeqCst);
        } else {
            while self.slow_width.is_some() && !self.slow_started.load(Ordering::SeqCst) {
                std::thread::yield_now();
            }
            if self.slow_width.is_some() && !self.slow_finished.load(Ordering::SeqCst) {
                self.fast_started_before_slow_finished
                    .fetch_add(1, Ordering::SeqCst);
            }
            std::thread::sleep(self.sleep);
        }
        if self.panic_width == Some(w) {
            panic!("mock backend panic on width {w}");
        }
        let host = HostPage {
            width: w,
            height: 1,
            stride: w as usize * 4,
            format: OutputFormat::Rgba8PremultipliedSrgb,
            pixels: Arc::from(vec![0u8; w as usize * 4]),
        };
        let _ = tx.send(Ok(RenderedPage::Host(host)));
        Ok(ticket)
    }
}

#[test]
fn slow_first_page_bounds_reorder_admission_without_deadlock() {
    let snap = snapshot(9);
    let backend = Arc::new(MockBackend {
        // Page zero's width is 4. The other pages finish promptly while it is
        // held, reproducing the pathological out-of-order completion case.
        sleep: Duration::ZERO,
        slow_width: Some(4),
        slow_sleep: Some(Duration::from_millis(80)),
        ..Default::default()
    });
    let make = make_requester(
        Arc::new(AtomicUsize::new(0)),
        Arc::new(AtomicUsize::new(0)),
        None,
    );
    let scheduler = RenderScheduler::new(
        backend.clone(),
        SchedulerOptions {
            // A single compiler preserves page-zero-first admission to the
            // backend. The render workers still complete later pages while
            // that first page is intentionally held.
            compile_workers: 1,
            render_workers: 3,
            compiled_queue_depth: 3,
            reorder_window_depth: 3,
            reorder_backlog_bytes: SchedulerOptions::DEFAULT_REORDER_BACKLOG_BYTES,
            memory_limit_bytes: 1 << 30,
        },
    );

    let out = collect(&scheduler, &snap, 9, &make, None);

    // The slow head does not deadlock the feeder: it eventually unlocks the
    // next window and every page is emitted in order.
    assert_eq!(out.len(), 9);
    assert_eq!(
        out.iter().map(|(page, _)| *page).collect::<Vec<_>>(),
        (0..9).collect::<Vec<_>>()
    );
    // Only the two slots after the missing head were admitted before it
    // completed. The old unbounded result path admitted all eight fast pages.
    let admitted = backend
        .fast_started_before_slow_finished
        .load(Ordering::SeqCst);
    assert!(admitted > 0, "fast pages exercised out-of-order completion");
    assert!(
        admitted <= 2,
        "window=3 retains at most two later completed surfaces, got {admitted}"
    );
}

fn snapshot(pages: u32) -> DocumentSnapshot {
    let source: Arc<dyn PdfSource> =
        Arc::new(OwnedBytesSource::new(builder::multipage_classic(pages)));
    DocumentSnapshot::open(source, DocumentLimits::default()).expect("open")
}

/// A request builder that compiles the page (real, concurrent) and encodes the
/// page index in the output width so results are page-identifiable. `fail_page`
/// makes one page's compilation return an error.
fn make_requester(
    compiles: Arc<AtomicUsize>,
    compile_peak: Arc<AtomicUsize>,
    fail_page: Option<u32>,
) -> impl Fn(&DocumentSnapshot, PageIndex) -> Result<RenderRequest, RenderError> + Sync {
    move |snapshot, page| {
        if fail_page == Some(page.0) {
            return Err(RenderError::Backend("intentional compile failure".into()));
        }
        let c = compiles.fetch_add(1, Ordering::SeqCst) + 1;
        compile_peak.fetch_max(c, Ordering::SeqCst);
        std::thread::sleep(Duration::from_millis(2));
        let mut ctx = ParseContext::new();
        let compiled = exprim::content::PageCompiler::new()
            .compile(snapshot, page, &mut ctx)
            .map_err(|e| RenderError::Backend(e.to_string()))?;
        compiles.fetch_sub(1, Ordering::SeqCst);
        Ok(RenderRequest {
            page: Arc::new(compiled),
            transform: PageTransform {
                matrix: Matrix::IDENTITY,
            },
            crop: None,
            output_size: DeviceSize {
                width: 4 + page.0,
                height: 1,
            },
            output_format: OutputFormat::Rgba8PremultipliedSrgb,
            background: Background::White,
            annotations: AnnotationMode::None,
            color_policy: exprim::render_api::RenderColorPolicy::Original,
            quality: RenderQuality::Normal,
            limits: RenderLimits::default(),
            residency: OutputResidency::HostRequired,
        })
    }
}

fn collect(
    scheduler: &RenderScheduler,
    snap: &DocumentSnapshot,
    pages: u32,
    make: &(dyn Fn(&DocumentSnapshot, PageIndex) -> Result<RenderRequest, RenderError> + Sync),
    cancel: Option<&CancellationToken>,
) -> Vec<(u32, Result<u32, String>)> {
    let mut out = Vec::new();
    scheduler.render_range(snap, 0..pages, make, cancel, &mut |o: PipelineOutput| {
        let r = match o.result {
            Ok(RenderedPage::Host(h)) => Ok(h.width),
            Ok(_) => Ok(0),
            Err(e) => Err(e.to_string()),
        };
        out.push((o.page.0, r));
    });
    out
}

#[test]
fn six_pages_compile_and_render_concurrently_in_order() {
    let snap = snapshot(6);
    let backend = Arc::new(MockBackend {
        sleep: Duration::from_millis(15),
        ..Default::default()
    });
    let compiles = Arc::new(AtomicUsize::new(0));
    let compile_peak = Arc::new(AtomicUsize::new(0));
    let make = make_requester(compiles, compile_peak.clone(), None);
    let opts = SchedulerOptions {
        compile_workers: 6,
        render_workers: 6,
        compiled_queue_depth: 6,
        reorder_window_depth: 6,
        reorder_backlog_bytes: SchedulerOptions::DEFAULT_REORDER_BACKLOG_BYTES,
        memory_limit_bytes: 1 << 30,
    };
    let scheduler = RenderScheduler::new(backend.clone(), opts);

    let out = collect(&scheduler, &snap, 6, &make, None);

    // Emitted in page order, one per page.
    let pages: Vec<u32> = out.iter().map(|(p, _)| *p).collect();
    assert_eq!(pages, vec![0, 1, 2, 3, 4, 5]);
    // Each output carries its page identity (width = 4 + page).
    for (p, r) in &out {
        assert_eq!(*r, Ok(4 + *p));
    }
    // Real concurrency was observed in both stages.
    assert!(
        backend.peak.load(Ordering::SeqCst) >= 2,
        "renders overlapped"
    );
    assert!(
        compile_peak.load(Ordering::SeqCst) >= 2,
        "compiles overlapped"
    );
}

#[test]
fn output_is_deterministic_across_worker_counts() {
    let snap = snapshot(8);
    let run = |cw: usize, rw: usize| {
        let backend = Arc::new(MockBackend {
            sleep: Duration::from_millis(1),
            ..Default::default()
        });
        let make = make_requester(
            Arc::new(AtomicUsize::new(0)),
            Arc::new(AtomicUsize::new(0)),
            None,
        );
        let opts = SchedulerOptions {
            compile_workers: cw,
            render_workers: rw,
            compiled_queue_depth: 4,
            reorder_window_depth: 4,
            reorder_backlog_bytes: SchedulerOptions::DEFAULT_REORDER_BACKLOG_BYTES,
            memory_limit_bytes: 1 << 30,
        };
        collect(&RenderScheduler::new(backend, opts), &snap, 8, &make, None)
    };
    let single = run(1, 1);
    let many = run(6, 6);
    assert_eq!(single, many, "identical output regardless of worker counts");
    assert_eq!(single.len(), 8);
}

#[test]
fn one_page_failure_does_not_affect_others() {
    let snap = snapshot(5);
    let backend = Arc::new(MockBackend {
        sleep: Duration::from_millis(1),
        ..Default::default()
    });
    // Compilation fails for page 2 only.
    let make = make_requester(
        Arc::new(AtomicUsize::new(0)),
        Arc::new(AtomicUsize::new(0)),
        Some(2),
    );
    let opts = SchedulerOptions {
        compile_workers: 4,
        render_workers: 4,
        compiled_queue_depth: 4,
        reorder_window_depth: 4,
        reorder_backlog_bytes: SchedulerOptions::DEFAULT_REORDER_BACKLOG_BYTES,
        memory_limit_bytes: 1 << 30,
    };
    let out = collect(&RenderScheduler::new(backend, opts), &snap, 5, &make, None);

    assert_eq!(out.len(), 5);
    assert!(out[2].1.is_err(), "page 2 failed");
    for p in [0u32, 1, 3, 4] {
        assert_eq!(out[p as usize].1, Ok(4 + p), "page {p} unaffected");
    }
}

#[test]
fn backend_panic_on_one_page_is_isolated() {
    let snap = snapshot(4);
    // The backend panics rendering page 1 (its output width is 4+1=5).
    let backend = Arc::new(MockBackend {
        sleep: Duration::from_millis(1),
        panic_width: Some(5),
        ..Default::default()
    });
    let make = make_requester(
        Arc::new(AtomicUsize::new(0)),
        Arc::new(AtomicUsize::new(0)),
        None,
    );
    let opts = SchedulerOptions {
        compile_workers: 4,
        render_workers: 4,
        compiled_queue_depth: 4,
        reorder_window_depth: 4,
        reorder_backlog_bytes: SchedulerOptions::DEFAULT_REORDER_BACKLOG_BYTES,
        memory_limit_bytes: 1 << 30,
    };
    let out = collect(&RenderScheduler::new(backend, opts), &snap, 4, &make, None);

    assert_eq!(out.len(), 4);
    // The caught panic surfaces as the typed RenderError::Panic — payload
    // message preserved — matching the exprim::render_api boundary taxonomy.
    let err = out[1]
        .1
        .as_ref()
        .expect_err("page 1 render panicked → isolated error");
    assert!(
        err.contains("backend panicked") && err.contains("mock backend panic on width 5"),
        "expected RenderError::Panic with payload, got: {err}"
    );
    for p in [0u32, 2, 3] {
        assert_eq!(out[p as usize].1, Ok(4 + p), "page {p} still succeeded");
    }
}

#[test]
fn pipeline_token_is_injected_into_backend_requests() {
    // The scheduler must propagate its token into `limits.cancellation` so a
    // backend can observe cancellation *mid-render* (active jobs, not just
    // queued ones). A live token must not disturb normal output.
    let snap = snapshot(4);
    let backend = Arc::new(MockBackend {
        sleep: Duration::from_millis(1),
        ..Default::default()
    });
    let make = make_requester(
        Arc::new(AtomicUsize::new(0)),
        Arc::new(AtomicUsize::new(0)),
        None,
    );
    let opts = SchedulerOptions {
        compile_workers: 2,
        render_workers: 2,
        compiled_queue_depth: 2,
        reorder_window_depth: 2,
        reorder_backlog_bytes: SchedulerOptions::DEFAULT_REORDER_BACKLOG_BYTES,
        memory_limit_bytes: 1 << 30,
    };
    let token = CancellationToken::new(); // live — never cancelled

    let out = collect(
        &RenderScheduler::new(backend.clone(), opts),
        &snap,
        4,
        &make,
        Some(&token),
    );

    assert_eq!(out.len(), 4);
    for (p, r) in &out {
        assert_eq!(*r, Ok(4 + *p));
    }
    assert_eq!(
        backend.saw_token.load(Ordering::SeqCst),
        4,
        "every request carried the pipeline token"
    );
}

#[derive(Debug)]
struct CancellingBackend {
    pipeline: CancellationToken,
}

impl RenderBackend for CancellingBackend {
    fn id(&self) -> BackendId {
        BackendId::Other(2)
    }
    fn capabilities(&self) -> BackendCapabilities {
        MockBackend::default().capabilities()
    }
    fn supports(&self, _: &exprim::page_ir::CompiledPage, _: &RenderRequest) -> SupportLevel {
        SupportLevel::Native
    }
    fn submit(&self, request: RenderRequest) -> Result<RenderTicket, SubmitError> {
        // Cancel after admission, while the backend owns a request that also
        // has a separate caller token. Neither source may mask the other.
        self.pipeline.cancel();
        let (ticket, sender) = RenderTicket::new(0);
        let error = if request
            .limits
            .cancellation
            .as_ref()
            .is_some_and(CancellationToken::is_cancelled)
        {
            RenderError::Cancelled
        } else {
            RenderError::Backend("pipeline cancellation was hidden by request token".into())
        };
        sender.send(Err(error)).unwrap();
        Ok(ticket)
    }
}

#[test]
fn pipeline_cancellation_reaches_an_active_request_with_its_own_token() {
    let snapshot = snapshot(1);
    let pipeline = CancellationToken::new();
    let local = CancellationToken::new();
    let make = make_requester(
        Arc::new(AtomicUsize::new(0)),
        Arc::new(AtomicUsize::new(0)),
        None,
    );
    let requester = |snapshot: &DocumentSnapshot, page| {
        let mut request = make(snapshot, page)?;
        request.limits.cancellation = Some(local.clone());
        Ok(request)
    };
    let scheduler = RenderScheduler::new(
        Arc::new(CancellingBackend {
            pipeline: pipeline.clone(),
        }),
        SchedulerOptions::default(),
    );
    let mut results = Vec::new();
    scheduler.render_range(&snapshot, 0..1, &requester, Some(&pipeline), &mut |out| {
        results.push(out)
    });
    assert_eq!(results.len(), 1);
    assert!(matches!(results[0].result, Err(RenderError::Cancelled)));
    assert!(
        !local.is_cancelled(),
        "cancellation must not mutate the caller's token"
    );
    assert_eq!(scheduler.budget().in_use(), 0);
}

#[test]
fn cancellation_stops_the_pipeline() {
    let snap = snapshot(6);
    let backend = Arc::new(MockBackend {
        sleep: Duration::from_millis(1),
        ..Default::default()
    });
    let make = make_requester(
        Arc::new(AtomicUsize::new(0)),
        Arc::new(AtomicUsize::new(0)),
        None,
    );
    let opts = SchedulerOptions {
        compile_workers: 2,
        render_workers: 2,
        compiled_queue_depth: 2,
        reorder_window_depth: 2,
        reorder_backlog_bytes: SchedulerOptions::DEFAULT_REORDER_BACKLOG_BYTES,
        memory_limit_bytes: 1 << 30,
    };
    let token = CancellationToken::new();
    token.cancel(); // cancelled up front → every page short-circuits

    let out = collect(
        &RenderScheduler::new(backend, opts),
        &snap,
        6,
        &make,
        Some(&token),
    );

    assert_eq!(
        out.len(),
        6,
        "every page still produces a (cancelled) result in order"
    );
    for (_, r) in &out {
        assert!(
            matches!(r, Err(e) if e.contains("cancel")),
            "cancelled: {r:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// Reorder backlog bound in bytes
// ---------------------------------------------------------------------------

/// Finished-but-unemitted surface bytes, as the outside world sees them: the
/// backend adds a page's bytes the moment it has finished rendering it, and
/// the caller's `emit` subtracts them once it has consumed the page.
#[derive(Debug, Default)]
struct Backlog {
    bytes: std::sync::atomic::AtomicU64,
    peak: std::sync::atomic::AtomicU64,
}

impl Backlog {
    fn add(&self, n: u64) {
        let now = self.bytes.fetch_add(n, Ordering::SeqCst) + n;
        self.peak.fetch_max(now, Ordering::SeqCst);
    }
    fn sub(&self, n: u64) {
        self.bytes.fetch_sub(n, Ordering::SeqCst);
    }
    fn peak(&self) -> u64 {
        self.peak.load(Ordering::SeqCst)
    }
}

/// Renders a `width x 1` RGBA surface, holding the page whose width is
/// `slow_width` for `slow` and every other page for `fast`.
#[derive(Debug)]
struct SizedBackend {
    slow_width: Option<u32>,
    slow: Duration,
    fast: Duration,
    backlog: Arc<Backlog>,
}

impl RenderBackend for SizedBackend {
    fn id(&self) -> BackendId {
        BackendId::Other(3)
    }
    fn capabilities(&self) -> BackendCapabilities {
        MockBackend::default().capabilities()
    }
    fn supports(&self, _: &exprim::page_ir::CompiledPage, _: &RenderRequest) -> SupportLevel {
        SupportLevel::Native
    }
    fn submit(&self, request: RenderRequest) -> Result<RenderTicket, SubmitError> {
        let (ticket, tx) = RenderTicket::new(0);
        let w = request.output_size.width;
        std::thread::sleep(if self.slow_width == Some(w) {
            self.slow
        } else {
            self.fast
        });
        let pixels: Arc<[u8]> = Arc::from(vec![0u8; w as usize * 4]);
        self.backlog.add(pixels.len() as u64);
        let host = HostPage {
            width: w,
            height: 1,
            stride: w as usize * 4,
            format: OutputFormat::Rgba8PremultipliedSrgb,
            pixels,
        };
        let _ = tx.send(Ok(RenderedPage::Host(host)));
        Ok(ticket)
    }
}

/// Page `n` renders at width `100 + n`: 400 + 4n surface bytes.
fn sized_requester(
    snapshot: &DocumentSnapshot,
    page: PageIndex,
) -> Result<RenderRequest, RenderError> {
    let mut ctx = ParseContext::new();
    let compiled = exprim::content::PageCompiler::new()
        .compile(snapshot, page, &mut ctx)
        .map_err(|e| RenderError::Backend(e.to_string()))?;
    Ok(RenderRequest {
        page: Arc::new(compiled),
        transform: PageTransform {
            matrix: Matrix::IDENTITY,
        },
        crop: None,
        output_size: DeviceSize {
            width: 100 + page.0,
            height: 1,
        },
        output_format: OutputFormat::Rgba8PremultipliedSrgb,
        background: Background::White,
        annotations: AnnotationMode::None,
        color_policy: exprim::render_api::RenderColorPolicy::Original,
        quality: RenderQuality::Normal,
        limits: RenderLimits::default(),
        residency: OutputResidency::HostRequired,
    })
}

/// Run `pages` pages with page 0 held for `slow_first`, returning the
/// in-order `(page, width)` pairs and the peak backlog. `emit_delay` slows
/// the caller's consumer down.
fn run_backlog(
    pages: u32,
    render_workers: usize,
    backlog_limit: u64,
    slow_first: Duration,
    emit_delay: Duration,
) -> (Vec<(u32, u32)>, u64) {
    let snap = snapshot(pages);
    let backlog = Arc::new(Backlog::default());
    let backend = Arc::new(SizedBackend {
        slow_width: Some(100),
        slow: slow_first,
        fast: Duration::from_millis(1),
        backlog: backlog.clone(),
    });
    let scheduler = RenderScheduler::new(
        backend,
        SchedulerOptions {
            compile_workers: 2,
            render_workers,
            compiled_queue_depth: 4,
            // Wide open: the page-count window alone would let every page
            // finish behind the slow head.
            reorder_window_depth: 128,
            reorder_backlog_bytes: backlog_limit,
            memory_limit_bytes: 1 << 30,
        },
    );
    let mut out = Vec::new();
    scheduler.render_range(&snap, 0..pages, &sized_requester, None, &mut |o| {
        let Ok(RenderedPage::Host(host)) = o.result else {
            panic!("page {} failed: {:?}", o.page.0, o.result.err());
        };
        std::thread::sleep(emit_delay);
        out.push((o.page.0, host.width));
        backlog.sub(host.pixels.len() as u64);
    });
    (out, backlog.peak())
}

fn assert_all_in_order(out: &[(u32, u32)], pages: u32) {
    assert_eq!(
        out.iter().map(|(page, _)| *page).collect::<Vec<_>>(),
        (0..pages).collect::<Vec<_>>()
    );
    for (page, width) in out {
        assert_eq!(*width, 100 + page, "page {page} carries its own surface");
    }
}

#[test]
fn slow_first_page_backlog_is_bounded_in_bytes() {
    const PAGES: u32 = 40;
    const WORKERS: usize = 3;
    const LIMIT: u64 = 1_000; // about two surfaces
    let largest = u64::from(100 + PAGES - 1) * 4;

    let (out, peak) = run_backlog(
        PAGES,
        WORKERS,
        LIMIT,
        Duration::from_millis(300),
        Duration::ZERO,
    );
    assert_all_in_order(&out, PAGES);
    // Out-of-order pages stop starting once the finished backlog reaches the
    // limit; pages already rendering at that moment (at most one per worker)
    // still finish into it.
    let bound = LIMIT + WORKERS as u64 * largest;
    assert!(peak <= bound, "backlog peaked at {peak} B, bound {bound} B");

    // The same run with only the page-count window lets (nearly) every page
    // finish behind the slow head, which is what the byte bound prevents.
    let (out, unbounded_peak) = run_backlog(
        PAGES,
        WORKERS,
        u64::MAX,
        Duration::from_millis(300),
        Duration::ZERO,
    );
    assert_all_in_order(&out, PAGES);
    assert!(
        unbounded_peak > bound,
        "the scenario must overflow without the byte bound: {unbounded_peak} B"
    );
}

#[test]
fn a_slow_consumer_is_throttled_by_the_backlog_bound() {
    const PAGES: u32 = 30;
    const WORKERS: usize = 3;
    const LIMIT: u64 = 1_000;
    let largest = u64::from(100 + PAGES - 1) * 4;
    // No slow page: here the caller's `emit` is the bottleneck.
    let (out, peak) = run_backlog(
        PAGES,
        WORKERS,
        LIMIT,
        Duration::from_millis(1),
        Duration::from_millis(8),
    );
    assert_all_in_order(&out, PAGES);
    // One more surface than the slow-head case: the page inside `emit` still
    // counts until `emit` returns.
    let bound = LIMIT + (WORKERS as u64 + 1) * largest;
    assert!(peak <= bound, "backlog peaked at {peak} B, bound {bound} B");
}

#[test]
fn a_zero_byte_limit_still_drains_every_page_in_order() {
    // The head-of-line page is always admitted and an empty backlog admits
    // any page, so even a limit no surface fits makes progress: it degrades
    // to rendering ahead only while nothing is waiting.
    for (pages, workers) in [(25, 1), (25, 4), (60, 8)] {
        let (out, _) = run_backlog(pages, workers, 0, Duration::from_millis(20), Duration::ZERO);
        assert_all_in_order(&out, pages);
    }
}

#[test]
fn a_tiny_byte_limit_drains_with_cancellation_and_failures() {
    // Cancelled mid-run with a limit of one byte: every page still produces
    // exactly one result, in order, and nothing deadlocks.
    let snap = snapshot(30);
    let backend = Arc::new(SizedBackend {
        slow_width: Some(100),
        slow: Duration::from_millis(50),
        fast: Duration::from_millis(1),
        backlog: Arc::new(Backlog::default()),
    });
    let scheduler = RenderScheduler::new(
        backend,
        SchedulerOptions {
            compile_workers: 2,
            render_workers: 4,
            compiled_queue_depth: 2,
            reorder_window_depth: 64,
            reorder_backlog_bytes: 1,
            memory_limit_bytes: 1 << 30,
        },
    );
    let token = CancellationToken::new();
    let fail_page = 7;
    let make = |s: &DocumentSnapshot, page: PageIndex| {
        if page.0 == fail_page {
            return Err(RenderError::Backend("intentional compile failure".into()));
        }
        sized_requester(s, page)
    };
    let mut seen = Vec::new();
    scheduler.render_range(&snap, 0..30, &make, Some(&token), &mut |o| {
        if o.page.0 == 10 {
            token.cancel();
        }
        seen.push((o.page.0, o.result.is_ok()));
    });
    assert_eq!(
        seen.iter().map(|(p, _)| *p).collect::<Vec<_>>(),
        (0..30).collect::<Vec<_>>()
    );
    assert!(!seen[fail_page as usize].1, "the failed page is an error");
    assert!(
        seen[..7].iter().all(|(_, ok)| *ok),
        "pages before it rendered"
    );
    assert_eq!(scheduler.budget().in_use(), 0);
}

/// Render 30 pages through a scheduler whose consumer panics on page 3, and
/// send whether the panic reached this caller before tearing the scheduler
/// down.
fn render_with_a_consumer_that_panics_on_page_3(done_tx: std::sync::mpsc::Sender<bool>) {
    let snap = snapshot(30);
    let backend = Arc::new(SizedBackend {
        slow_width: Some(100),
        slow: Duration::from_millis(30),
        fast: Duration::from_millis(1),
        backlog: Arc::new(Backlog::default()),
    });
    let scheduler = RenderScheduler::new(
        backend,
        SchedulerOptions {
            compile_workers: 2,
            render_workers: 3,
            compiled_queue_depth: 2,
            reorder_window_depth: 8,
            reorder_backlog_bytes: 1,
            memory_limit_bytes: 1 << 30,
        },
    );
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        scheduler.render_range(&snap, 0..30, &sized_requester, None, &mut |o| {
            if o.page.0 == 3 {
                panic!("consumer failed on page 3");
            }
        });
    }));
    let _ = done_tx.send(outcome.is_err());
}

#[test]
fn a_panicking_consumer_unwinds_out_of_the_pipeline_instead_of_hanging() {
    // Workers may be parked waiting for backlog room or for emission credits
    // when the caller's `emit` panics; all of them must be released so the
    // panic reaches the caller.
    let (done_tx, done_rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        render_with_a_consumer_that_panics_on_page_3(done_tx);
    });
    let panicked = done_rx
        .recv_timeout(Duration::from_secs(20))
        .expect("render_range returned instead of hanging");
    assert!(panicked, "the consumer's panic propagates");
}

/// Holds every job for a moment and records how many it held at once. With
/// `working` set it reports that many bytes of working surface, as a backend
/// that paints a page strip by strip does.
#[derive(Debug)]
struct StripBackend {
    working: Option<u64>,
    live: AtomicUsize,
    peak: AtomicUsize,
}

impl RenderBackend for StripBackend {
    fn id(&self) -> BackendId {
        BackendId::Other(4)
    }
    fn capabilities(&self) -> BackendCapabilities {
        MockBackend::default().capabilities()
    }
    fn supports(&self, _: &exprim::page_ir::CompiledPage, _: &RenderRequest) -> SupportLevel {
        SupportLevel::Native
    }
    fn submit(&self, request: RenderRequest) -> Result<RenderTicket, SubmitError> {
        let (ticket, tx) = RenderTicket::new(0);
        let now = self.live.fetch_add(1, Ordering::SeqCst) + 1;
        self.peak.fetch_max(now, Ordering::SeqCst);
        std::thread::sleep(Duration::from_millis(40));
        self.live.fetch_sub(1, Ordering::SeqCst);
        let (w, h) = (request.output_size.width, request.output_size.height);
        let host = HostPage {
            width: w,
            height: h,
            stride: w as usize,
            format: OutputFormat::Gray8,
            pixels: Arc::from(vec![0u8; w as usize * h as usize]),
        };
        let _ = tx.send(Ok(RenderedPage::Host(host)));
        Ok(ticket)
    }
    fn working_surface_bytes(&self, request: &RenderRequest) -> u64 {
        self.working.unwrap_or(
            u64::from(request.output_size.width) * u64::from(request.output_size.height) * 4,
        )
    }
}

/// Every page as a 100 x 100 Gray8 raster: a 40,000-byte RGBA surface for a
/// 10,000-byte output.
fn gray_requester(
    snapshot: &DocumentSnapshot,
    page: PageIndex,
) -> Result<RenderRequest, RenderError> {
    let mut request = sized_requester(snapshot, page)?;
    request.output_size = DeviceSize {
        width: 100,
        height: 100,
    };
    request.output_format = OutputFormat::Gray8;
    Ok(request)
}

/// The scheduler budgets a job for the plane its backend paints on. A Gray8
/// page painted whole needs its RGBA surface beside the gray output; painted
/// strip by strip it needs one strip, so more such pages fit a budget at once.
#[test]
fn a_backend_that_paints_in_strips_is_budgeted_for_one_strip() {
    let snap = snapshot(12);
    let request = gray_requester(&snap, PageIndex(0)).unwrap();
    let whole = exprim::render_scheduler::estimate_job_bytes(&request);
    let strip = 100 * 10 * 4;
    let striped = exprim::render_scheduler::estimate_job_bytes_on(&request, strip);
    assert_eq!(
        whole - striped,
        40_000 - strip,
        "only the surface term shrinks"
    );
    // A plane larger than the page's surface cannot raise the estimate.
    assert_eq!(
        exprim::render_scheduler::estimate_job_bytes_on(&request, u64::MAX),
        whole
    );
    // Room for one page painted whole, and for four painted in strips.
    let limit = whole + whole / 2;
    assert!(2 * whole > limit && 4 * striped <= limit);

    let peak_under = |working: Option<u64>| {
        let backend = Arc::new(StripBackend {
            working,
            live: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
        });
        let scheduler = RenderScheduler::new(
            backend.clone(),
            SchedulerOptions {
                compile_workers: 2,
                render_workers: 4,
                compiled_queue_depth: 8,
                reorder_window_depth: 16,
                reorder_backlog_bytes: u64::MAX,
                memory_limit_bytes: limit,
            },
        );
        let mut pages = 0;
        scheduler.render_range(&snap, 0..12, &gray_requester, None, &mut |o| {
            assert!(o.result.is_ok(), "page {}: {:?}", o.page.0, o.result.err());
            pages += 1;
        });
        assert_eq!(pages, 12);
        backend.peak.load(Ordering::SeqCst)
    };
    assert_eq!(peak_under(None), 1, "whole surfaces: one page at a time");
    assert!(peak_under(Some(strip)) >= 2, "strips: pages overlap");
}
