//! `probe bench`: pinned, CPU-time-based measurement with JSONL output.
//!
//! probe bench <pdf> [--api low|session] [--scenario repeat|rerender|seq|cold|zoom]
//!     [--dpi D] [--pages SPEC] [--repeat K] [--warmup W] [--format rgba|rgb8|gray8]
//!     [--zoom Z] [--focus FX,FY] [--viewport WxH] [--max-pixels N]
//!     [--pin pcore|cpu:N|mask:HEX|none] [--priority high|normal]
//!     [--detail stage|fill|row] [--hash] [--system-fonts] [--id NAME] [--jsonl FILE]
//!
//! One JSON object per line: a `meta` row, then one `page` row per render.
//! Each row carries wall time, the process's cycle count (every thread: some
//! work runs on a rayon pool thread even at one worker), the calling thread's
//! own cycle count and coarse CPU time, so a contended sample is visible
//! (`cycles` stays put while `wall_ns` grows). With `--features profiling` the row also carries every
//! named stage timer and work counter of that render.
//!
//! Scenarios (all single-threaded on the calling thread):
//!   repeat    compile + render every iteration (the session path hits its
//!             compiled-page cache after the first)
//!   rerender  compile once, render K times
//!   seq       every selected page once, in order
//!   cold      the first selected page once: open, compile, first paint
//!   zoom      rerender of a fixed viewport at `--zoom` times the page scale
//!
//! `--api low` drives `PageCompiler` + `CpuBackend` with one reused worker
//! context; `--api session` drives the public `RenderSession`. Both render the
//! same pixel dimensions, but their page matrices differ (uniform scale versus
//! fill-to-size), so hashes are comparable within an API, never across.

use std::io::Write;
use std::sync::Arc;
use std::time::Instant;

use exprim::document::{PageIndex, ParseContext};
use exprim::page_ir::{CompiledPage, DeviceSize};
use exprim::render_api::OutputFormat;
use exprim::render_cpu::{CpuBackend, CpuWorkerContext};

use crate::win;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Scenario {
    Repeat,
    Rerender,
    Seq,
    Cold,
    Zoom,
}

impl Scenario {
    fn name(self) -> &'static str {
        match self {
            Scenario::Repeat => "repeat",
            Scenario::Rerender => "rerender",
            Scenario::Seq => "seq",
            Scenario::Cold => "cold",
            Scenario::Zoom => "zoom",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Format {
    Rgba,
    Rgb8,
    Gray8,
}

/// Cost of one measured region.
#[derive(Clone, Copy, Default)]
struct Cost {
    wall_ns: u64,
    /// Process-wide cycles (the cost metric).
    cycles: u64,
    /// Cycles of the calling thread alone.
    thread_cycles: u64,
    cpu_100ns: u64,
}

struct Meter {
    start: Instant,
    cost: win::ThreadCost,
}

impl Meter {
    fn start() -> Self {
        Self {
            cost: win::thread_cost(),
            start: Instant::now(),
        }
    }

    fn stop(self) -> Cost {
        let wall_ns = self.start.elapsed().as_nanos() as u64;
        let after = win::thread_cost();
        Cost {
            wall_ns,
            cycles: after.process_cycles.saturating_sub(self.cost.process_cycles),
            thread_cycles: after.cycles.saturating_sub(self.cost.cycles),
            cpu_100ns: after.cpu_100ns.saturating_sub(self.cost.cpu_100ns),
        }
    }
}

/// A fixed viewport into a page rendered at `scale * zoom`.
#[derive(Clone, Copy)]
struct Viewport {
    zoom: f64,
    focus: (f64, f64),
    size: (u32, u32),
}

/// The pixel rectangle one render produces: full-page size at the effective
/// scale, and the window into it (the whole page unless zoomed).
#[derive(Clone, Copy)]
struct Target {
    scale: f64,
    page_w: u32,
    page_h: u32,
    x: i32,
    y: i32,
    w: u32,
    h: u32,
}

fn target_for(display_w_pt: f64, display_h_pt: f64, scale: f64, viewport: Option<Viewport>) -> Target {
    let effective = scale * viewport.map_or(1.0, |v| v.zoom);
    let page_w = (display_w_pt * effective).ceil().max(1.0) as u32;
    let page_h = (display_h_pt * effective).ceil().max(1.0) as u32;
    match viewport {
        None => Target { scale: effective, page_w, page_h, x: 0, y: 0, w: page_w, h: page_h },
        Some(v) => {
            let w = v.size.0.min(page_w).max(1);
            let h = v.size.1.min(page_h).max(1);
            let x = ((v.focus.0 * f64::from(page_w)) - f64::from(w) / 2.0)
                .round()
                .clamp(0.0, f64::from(page_w - w)) as i32;
            let y = ((v.focus.1 * f64::from(page_h)) - f64::from(h) / 2.0)
                .round()
                .clamp(0.0, f64::from(page_h - h)) as i32;
            Target { scale: effective, page_w, page_h, x, y, w, h }
        }
    }
}

fn fnv1a64(width: u32, height: u32, pixels: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    let mut eat = |bytes: &[u8]| {
        for &b in bytes {
            hash ^= u64::from(b);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    eat(&width.to_le_bytes());
    eat(&height.to_le_bytes());
    eat(pixels);
    hash
}

fn json_escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

struct Row<'a> {
    id: &'a str,
    api: &'a str,
    scenario: &'a str,
    dpi: f64,
    page: u32,
    iter: usize,
    phase: &'a str,
    target: Target,
    compile: Cost,
    render: Cost,
    hash: Option<u64>,
    output_bytes: usize,
    profile_json: String,
}

fn write_row(out: &mut dyn Write, row: &Row<'_>) {
    let mut line = format!(
        "{{\"kind\":\"page\",\"id\":\"{}\",\"api\":\"{}\",\"scenario\":\"{}\",\"dpi\":{},\"scale\":{},\
         \"page\":{},\"iter\":{},\"phase\":\"{}\",\"w\":{},\"h\":{},\"page_w\":{},\"page_h\":{},\
         \"compile_wall_ns\":{},\"compile_cycles\":{},\"compile_cpu_100ns\":{},\
         \"render_wall_ns\":{},\"render_cycles\":{},\"render_thread_cycles\":{},\"render_cpu_100ns\":{},\"output_bytes\":{}",
        json_escape(row.id),
        row.api,
        row.scenario,
        row.dpi,
        row.target.scale,
        row.page,
        row.iter,
        row.phase,
        row.target.w,
        row.target.h,
        row.target.page_w,
        row.target.page_h,
        row.compile.wall_ns,
        row.compile.cycles,
        row.compile.cpu_100ns,
        row.render.wall_ns,
        row.render.cycles,
        row.render.thread_cycles,
        row.render.cpu_100ns,
        row.output_bytes,
    );
    if let Some(hash) = row.hash {
        line.push_str(&format!(",\"hash\":\"{hash:016x}\""));
    }
    if !row.profile_json.is_empty() {
        line.push(',');
        line.push_str(&row.profile_json);
    }
    line.push_str("}\n");
    out.write_all(line.as_bytes()).expect("write row");
}

fn phase_for(iter: usize, warmup: usize) -> &'static str {
    if iter == 0 {
        "first"
    } else if iter <= warmup {
        "warmup"
    } else {
        "warm"
    }
}

pub fn bench(args: &[String]) {
    let mut a = crate::Args(args.to_vec());
    let api = a.value("--api").unwrap_or_else(|| "low".into());
    let scenario = match a.value("--scenario").as_deref().unwrap_or("repeat") {
        "repeat" => Scenario::Repeat,
        "rerender" => Scenario::Rerender,
        "seq" => Scenario::Seq,
        "cold" => Scenario::Cold,
        "zoom" => Scenario::Zoom,
        other => {
            eprintln!("unknown --scenario {other}");
            std::process::exit(2);
        }
    };
    let dpi: f64 = a.value("--dpi").map(|v| v.parse().expect("dpi")).unwrap_or(144.0);
    let pages_spec = a.value("--pages").unwrap_or_else(|| "0".into());
    let mut repeat: usize = a.value("--repeat").map(|v| v.parse().expect("repeat")).unwrap_or(10);
    let mut warmup: usize = a.value("--warmup").map(|v| v.parse().expect("warmup")).unwrap_or(2);
    let format = match a.value("--format").as_deref() {
        None => {
            if api == "session" {
                Format::Rgb8
            } else {
                Format::Rgba
            }
        }
        Some("rgba") => Format::Rgba,
        Some("rgb8") => Format::Rgb8,
        Some("gray8") => Format::Gray8,
        Some(other) => {
            eprintln!("unknown --format {other}");
            std::process::exit(2);
        }
    };
    let zoom: f64 = a.value("--zoom").map(|v| v.parse().expect("zoom")).unwrap_or(4.0);
    let focus = a
        .value("--focus")
        .map(|v| {
            let (x, y) = v.split_once(',').expect("--focus FX,FY");
            (x.parse().expect("focus x"), y.parse().expect("focus y"))
        })
        .unwrap_or((0.5, 0.5));
    let viewport_size = a
        .value("--viewport")
        .map(|v| {
            let (w, h) = v.split_once('x').expect("--viewport WxH");
            (w.parse().expect("viewport w"), h.parse().expect("viewport h"))
        })
        .unwrap_or((1280u32, 800u32));
    let max_pixels: u64 =
        a.value("--max-pixels").map(|v| v.parse().expect("max-pixels")).unwrap_or(40_000_000);
    let pin_spec = a.value("--pin").unwrap_or_else(|| "pcore".into());
    let priority_high = a.value("--priority").is_none_or(|v| v == "high");
    let detail = a.value("--detail").unwrap_or_else(|| "stage".into());
    if !matches!(detail.as_str(), "stage" | "fill" | "row") {
        eprintln!("unknown --detail {detail}");
        std::process::exit(2);
    }
    let do_hash = a.flag("--hash");
    let system_fonts = a.flag("--system-fonts");
    let id = a.value("--id");
    let jsonl = a.value("--jsonl");
    let path = a.positional().to_owned();
    let id = id.unwrap_or_else(|| {
        std::path::Path::new(&path)
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "doc".into())
    });

    // Pin before any other thread can exist, so every worker inherits it.
    let pin = win::pin(&pin_spec, priority_high);
    let cycles_per_ns = win::cycles_per_ns();
    #[cfg(feature = "profiling")]
    let clock_ns = exprim::profiling::clock_read_cost_ns();
    #[cfg(not(feature = "profiling"))]
    let clock_ns = 0u64;

    match scenario {
        Scenario::Seq | Scenario::Cold => {
            repeat = 1;
            warmup = 0;
        }
        _ => {}
    }
    let viewport = (scenario == Scenario::Zoom).then_some(Viewport {
        zoom,
        focus,
        size: viewport_size,
    });
    let scale = dpi / 72.0;

    let mut sink: Box<dyn Write> = match &jsonl {
        Some(file) => Box::new(std::io::BufWriter::new(
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(file)
                .expect("open --jsonl"),
        )),
        None => Box::new(std::io::BufWriter::new(std::io::stdout())),
    };

    let open_meter = Meter::start();
    let mut state = match api.as_str() {
        "low" => Api::low(&path, system_fonts),
        "session" => Api::session(&path),
        other => {
            eprintln!("unknown --api {other}");
            std::process::exit(2);
        }
    };
    let open = open_meter.stop();
    let count = state.page_count();
    let mut pages = crate::parse_pages(&pages_spec, count);
    pages.retain(|p| *p < count);
    if scenario == Scenario::Cold {
        pages.truncate(1);
    }
    #[cfg(feature = "profiling")]
    state.set_detail(match detail.as_str() {
        "fill" => exprim::profiling::ProfileDetail::Fill,
        "row" => exprim::profiling::ProfileDetail::Row,
        _ => exprim::profiling::ProfileDetail::Stage,
    });

    let meta = format!(
        "{{\"kind\":\"meta\",\"id\":\"{}\",\"file\":\"{}\",\"pages\":{count},\"api\":\"{api}\",\
         \"scenario\":\"{}\",\"dpi\":{dpi},\"repeat\":{repeat},\"warmup\":{warmup},\
         \"format\":\"{}\",\"profiling\":{},\"detail\":\"{detail}\",\
         \"pin\":\"{}\",\"affinity_mask\":\"{:x}\",\"efficiency_classes\":\"{}\",\
         \"priority_high\":{},\"pin_note\":\"{}\",\"cycles_per_ns\":{:.4},\"clock_read_ns\":{clock_ns},\
         \"rayon_threads\":\"{}\",\"open_wall_ns\":{},\"open_cycles\":{},\
         \"peak_working_set\":{},\"peak_commit\":{}}}\n",
        json_escape(&id),
        json_escape(&path),
        scenario.name(),
        match format {
            Format::Rgba => "rgba",
            Format::Rgb8 => "rgb8",
            Format::Gray8 => "gray8",
        },
        cfg!(feature = "profiling"),
        json_escape(&pin.requested),
        pin.mask,
        pin.efficiency_classes,
        pin.priority_high,
        json_escape(&pin.note),
        cycles_per_ns,
        std::env::var("RAYON_NUM_THREADS").unwrap_or_default(),
        open.wall_ns,
        open.cycles,
        win::memory().peak_working_set,
        win::memory().peak_commit,
    );
    sink.write_all(meta.as_bytes()).expect("write meta");

    let compile_each_iter = matches!(scenario, Scenario::Repeat | Scenario::Seq | Scenario::Cold);
    for iter in 0..(1 + warmup + repeat).max(1) {
        // `repeat` counts warm iterations; the first and warmups come before.
        if matches!(scenario, Scenario::Seq | Scenario::Cold) && iter > 0 {
            break;
        }
        for &page in &pages {
            let (display_w, display_h) = state.display_size(page);
            let target = target_for(display_w, display_h, scale, viewport);
            if u64::from(target.w) * u64::from(target.h) > max_pixels {
                let line = format!(
                    "{{\"kind\":\"skip\",\"id\":\"{}\",\"page\":{page},\"reason\":\"pixel_cap\",\"w\":{},\"h\":{}}}\n",
                    json_escape(&id),
                    target.w,
                    target.h
                );
                sink.write_all(line.as_bytes()).expect("write skip");
                continue;
            }
            let compile = if compile_each_iter || iter == 0 {
                let meter = Meter::start();
                state.compile(page);
                meter.stop()
            } else {
                Cost::default()
            };
            let meter = Meter::start();
            let rendered = state.render(page, target, format);
            let render = meter.stop();
            let row = Row {
                id: &id,
                api: &api,
                scenario: scenario.name(),
                dpi,
                page,
                iter,
                phase: phase_for(iter, warmup),
                target,
                compile,
                render,
                hash: do_hash.then(|| fnv1a64(rendered.width, rendered.height, &rendered.pixels)),
                output_bytes: rendered.pixels.len(),
                profile_json: rendered.profile_json,
            };
            write_row(sink.as_mut(), &row);
        }
    }
    let memory = win::memory();
    let tail = format!(
        "{{\"kind\":\"end\",\"id\":\"{}\",\"peak_working_set\":{},\"peak_commit\":{},\"page_faults\":{}}}\n",
        json_escape(&id),
        memory.peak_working_set.max(memory.working_set),
        memory.peak_commit,
        memory.page_faults
    );
    sink.write_all(tail.as_bytes()).expect("write end");
    sink.flush().expect("flush");
}

struct Rendered {
    width: u32,
    height: u32,
    pixels: Arc<[u8]>,
    profile_json: String,
}

/// The two measured API paths behind one interface.
enum Api {
    Low {
        snap: exprim::document::DocumentSnapshot,
        compiler: exprim::content::PageCompiler,
        backend: CpuBackend,
        worker: CpuWorkerContext,
        compiled: std::collections::HashMap<u32, Arc<CompiledPage>>,
        #[cfg(feature = "profiling")]
        compile_profile: Option<exprim::profiling::ProfileReport>,
    },
    Session {
        session: exprim::RenderSession,
        compiled: std::collections::HashMap<u32, Arc<exprim::CompiledDocumentPage>>,
    },
}

impl Api {
    fn low(path: &str, system_fonts: bool) -> Self {
        let (snap, _open) = crate::open_snapshot(path);
        let provider: Option<Arc<dyn exprim::font::SystemFontProvider>> = if system_fonts {
            Some(Arc::new(exprim::font::FolderFontProvider::system()))
        } else {
            None
        };
        Api::Low {
            snap,
            compiler: crate::compiler(provider),
            backend: CpuBackend::default(),
            worker: CpuWorkerContext::new(),
            compiled: std::collections::HashMap::new(),
            #[cfg(feature = "profiling")]
            compile_profile: None,
        }
    }

    fn session(path: &str) -> Self {
        let bytes: Arc<[u8]> = Arc::from(std::fs::read(path).expect("read input"));
        Api::Session {
            session: exprim::RenderSession::open(bytes, None).expect("open session"),
            compiled: std::collections::HashMap::new(),
        }
    }

    fn page_count(&self) -> u32 {
        match self {
            Api::Low { snap, .. } => snap.page_count(),
            Api::Session { session, .. } => session.page_count(),
        }
    }

    #[cfg(feature = "profiling")]
    fn set_detail(&mut self, detail: exprim::profiling::ProfileDetail) {
        match self {
            Api::Low { worker, .. } => worker.set_profile_detail(detail),
            Api::Session { session, .. } => session.set_profile_detail(detail),
        }
    }

    /// Displayed page size in points (rotation applied).
    fn display_size(&mut self, page: u32) -> (f64, f64) {
        match self {
            Api::Low { snap, .. } => {
                let page_ref = snap.page(PageIndex(page)).expect("page");
                let [x0, y0, x1, y1] = page_ref.crop_box;
                let (w, h) = ((x1 - x0).abs(), (y1 - y0).abs());
                match page_ref.rotate % 180 {
                    90 => (h, w),
                    _ => (w, h),
                }
            }
            Api::Session { session, .. } => {
                let geometry = session.page_geometry(page).expect("page geometry");
                (geometry.display_width(), geometry.display_height())
            }
        }
    }

    fn compile(&mut self, page: u32) {
        match self {
            Api::Low {
                snap,
                compiler,
                compiled,
                #[cfg(feature = "profiling")]
                compile_profile,
                ..
            } => {
                let mut ctx = ParseContext::new();
                #[cfg(feature = "profiling")]
                let page_ir = {
                    let (page_ir, profile) = compiler
                        .compile_profiled(snap, PageIndex(page), &mut ctx)
                        .expect("compile page");
                    *compile_profile = Some(profile);
                    page_ir
                };
                #[cfg(not(feature = "profiling"))]
                let page_ir = compiler.compile(snap, PageIndex(page), &mut ctx).expect("compile page");
                compiled.insert(page, Arc::new(page_ir));
            }
            Api::Session { session, compiled } => {
                compiled.insert(page, session.compile(page).expect("compile page"));
            }
        }
    }

    fn render(&mut self, page: u32, target: Target, format: Format) -> Rendered {
        match self {
            Api::Low {
                backend,
                worker,
                compiled,
                #[cfg(feature = "profiling")]
                compile_profile,
                ..
            } => {
                let page_ir = Arc::clone(compiled.get(&page).expect("compiled page"));
                let mut request = crate::request_for_page(page_ir, target.scale);
                request.transform.matrix.e -= f64::from(target.x);
                request.transform.matrix.f -= f64::from(target.y);
                request.output_size = DeviceSize { width: target.w, height: target.h };
                request.output_format = match format {
                    Format::Gray8 => OutputFormat::Gray8,
                    _ => OutputFormat::Rgba8PremultipliedSrgb,
                };
                #[cfg(feature = "profiling")]
                {
                    let (host, _stats, mut profile) =
                        backend.render_profiled_with(&request, worker).expect("render page");
                    if let Some(compile) = compile_profile.take() {
                        profile.merge(&compile);
                    }
                    let mut profile_json = String::new();
                    profile.write_json_fields(&mut profile_json);
                    Rendered {
                        width: host.width,
                        height: host.height,
                        pixels: host.pixels,
                        profile_json,
                    }
                }
                #[cfg(not(feature = "profiling"))]
                {
                    let (host, _stats) = backend.render_with(&request, worker).expect("render page");
                    Rendered {
                        width: host.width,
                        height: host.height,
                        pixels: host.pixels,
                        profile_json: String::new(),
                    }
                }
            }
            Api::Session { session, compiled } => {
                let handle = compiled.get(&page).expect("compiled page");
                let mut product = match format {
                    Format::Gray8 => exprim::RasterProduct::gray8(target.w, target.h),
                    _ => exprim::RasterProduct::rgb8(target.w, target.h),
                };
                if (target.w, target.h) != (target.page_w, target.page_h) {
                    product = product.with_crop(exprim::DeviceCrop {
                        x: target.x,
                        y: target.y,
                        width: target.w,
                        height: target.h,
                        page_width: target.page_w,
                        page_height: target.page_h,
                    });
                }
                #[cfg(feature = "profiling")]
                let (plane, profile_json) = {
                    let (plane, profile) =
                        session.render_profiled(handle, &product).expect("render page");
                    let mut json = String::new();
                    profile.write_json_fields(&mut json);
                    (plane, json)
                };
                #[cfg(not(feature = "profiling"))]
                let (plane, profile_json) =
                    (session.render(handle, &product).expect("render page"), String::new());
                match plane {
                    exprim::RasterPlane::Rgb8(surface) => Rendered {
                        width: surface.width,
                        height: surface.height,
                        pixels: surface.pixels,
                        profile_json,
                    },
                    exprim::RasterPlane::Gray8(surface) => Rendered {
                        width: surface.width,
                        height: surface.height,
                        pixels: surface.pixels,
                        profile_json,
                    },
                }
            }
        }
    }
}
