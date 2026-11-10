//! `probe census <pdf> [--max-pages N] [--budget-ms MS]`
//!
//! Compile-only page census: one JSON object per sampled page on stdout (or one
//! `{"file":..,"error":..}` line). No rendering. Hand-written JSON, no serde.

use std::fmt::Write as _;
use std::panic::AssertUnwindSafe;
use std::sync::Arc;
use std::time::Instant;

use exprim::content::PageCompiler;
use exprim::document::{DocumentLimits, DocumentSnapshot, PageIndex, ParseContext};
use exprim::page_ir::{
    BlendMode, CompiledPage, DisplayOp, LineCap, LineJoin, Paint, PaintOrigin, PathData, PathVerb,
};
use exprim::source::{MmapSource, PdfSource};

fn esc(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '\\' => o.push_str("\\\\"),
            '"' => o.push_str("\\\""),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(o, "\\u{:04x}", c as u32);
            }
            c => o.push(c),
        }
    }
    o
}

fn error_line(path: &str, msg: &str) {
    println!("{{\"file\":\"{}\",\"error\":\"{}\"}}", esc(path), esc(msg));
}

#[derive(Default, Clone, Copy)]
struct Geo {
    paths: u64,
    lines: u64,
    curves: u64,
}

impl Geo {
    fn add(&mut self, p: &PathData, max_verbs: &mut u64) {
        self.paths += 1;
        let mut l = 0;
        let mut c = 0;
        for v in p.verbs.iter() {
            match v {
                PathVerb::LineTo => l += 1,
                PathVerb::CurveTo => c += 1,
                _ => {}
            }
        }
        self.lines += l;
        self.curves += c;
        *max_verbs = (*max_verbs).max(p.verbs.len() as u64);
    }
    fn json(&self, name: &str, o: &mut String) {
        let _ = write!(
            o,
            ",\"{name}_paths\":{},\"{name}_lines\":{},\"{name}_curves\":{}",
            self.paths, self.lines, self.curves
        );
    }
}

/// A clip path that is a plain axis-aligned-looking rectangle: no curves and at
/// most 5 drawing verbs (move + up to 4 lines + close).
fn is_rect_like(p: &PathData) -> bool {
    let mut lines = 0;
    for v in p.verbs.iter() {
        match v {
            PathVerb::CurveTo => return false,
            PathVerb::LineTo => lines += 1,
            _ => {}
        }
    }
    lines <= 4 && p.verbs.iter().filter(|v| matches!(v, PathVerb::MoveTo)).count() <= 1
}

fn font_kind(program: &[u8]) -> &'static str {
    match program.get(..4) {
        Some([0, 1, 0, 0]) | Some(b"true") | Some(b"ttcf") => "truetype",
        Some(b"OTTO") => "cff",
        Some([1, 0, _, _]) => "cff",
        Some([0x80, 1, _, _]) | Some([b'%', b'!', _, _]) => "type1",
        _ => "other",
    }
}

fn census_page(path: &str, pages: u32, page: u32, cp: &CompiledPage, compile_ms: f64) -> String {
    let mut fill = Geo::default();
    let mut stroke = Geo::default();
    let mut clip = Geo::default();
    let (mut max_fill, mut max_stroke, mut max_clip) = (0u64, 0u64, 0u64);

    let (mut n_fill, mut n_stroke, mut n_clip, mut n_clip_nonrect, mut n_clip_text) =
        (0u64, 0u64, 0u64, 0u64, 0u64);
    let (mut n_runs, mut n_img, mut n_shade, mut n_group, mut n_smask) = (0u64, 0u64, 0u64, 0u64, 0u64);
    let (mut n_form, mut n_tiling_cell, mut n_t3glyph, mut n_annot) = (0u64, 0u64, 0u64, 0u64);
    let mut dashed = 0u64;
    let mut joins = [0u64; 3];
    let mut caps = [0u64; 3];
    let mut widths = [0u64; 3];
    let (mut alpha_lt1, mut blend_non_normal) = (0u64, 0u64);
    let mut modes = [0u64; 8];
    let mut glyphs_total = 0u64;
    let mut rotated_runs = 0u64;
    let mut max_gid = 0u32;
    let mut glyph_stroke_runs = 0u64;
    let (mut pat_draws, mut shade_paint_draws) = (0u64, 0u64);

    let alpha = |a: f32, b: BlendMode, al: &mut u64, bl: &mut u64| {
        if a < 1.0 {
            *al += 1;
        }
        if b != BlendMode::Normal {
            *bl += 1;
        }
    };

    for op in cp.operations.iter() {
        let pid = match op {
            DisplayOp::FillPath { paint, .. }
            | DisplayOp::StrokePath { paint, .. }
            | DisplayOp::DrawGlyphRun { paint, .. } => Some(*paint),
            _ => None,
        };
        if let Some(p) = pid.and_then(|p| cp.paints.get(p.0 as usize)) {
            match p {
                Paint::Pattern { .. } => pat_draws += 1,
                Paint::Shading { .. } => shade_paint_draws += 1,
                _ => {}
            }
        }
        match op {
            DisplayOp::FillPath { path, alpha: a, blend, .. } => {
                n_fill += 1;
                if let Some(p) = cp.paths.get(path.0 as usize) {
                    fill.add(p, &mut max_fill);
                }
                alpha(*a, *blend, &mut alpha_lt1, &mut blend_non_normal);
            }
            DisplayOp::StrokePath { path, style, alpha: a, blend, .. } => {
                n_stroke += 1;
                if let Some(p) = cp.paths.get(path.0 as usize) {
                    stroke.add(p, &mut max_stroke);
                }
                if let Some(s) = cp.stroke_styles.get(style.0 as usize) {
                    if !s.dash_pattern.is_empty() {
                        dashed += 1;
                    }
                    match s.join {
                        LineJoin::Miter => joins[0] += 1,
                        LineJoin::Round => joins[1] += 1,
                        LineJoin::Bevel => joins[2] += 1,
                    }
                    match s.cap {
                        LineCap::Butt => caps[0] += 1,
                        LineCap::Round => caps[1] += 1,
                        LineCap::Square => caps[2] += 1,
                    }
                    if s.width < 0.5 {
                        widths[0] += 1;
                    } else if s.width <= 2.0 {
                        widths[1] += 1;
                    } else {
                        widths[2] += 1;
                    }
                }
                alpha(*a, *blend, &mut alpha_lt1, &mut blend_non_normal);
            }
            DisplayOp::PushClip { path, .. } => {
                n_clip += 1;
                if let Some(p) = cp.paths.get(path.0 as usize) {
                    clip.add(p, &mut max_clip);
                    if !is_rect_like(p) {
                        n_clip_nonrect += 1;
                    }
                }
            }
            DisplayOp::PushClipText { .. } => n_clip_text += 1,
            DisplayOp::DrawGlyphRun { run, alpha: a, blend, stroke: gs, .. } => {
                n_runs += 1;
                if gs.is_some() {
                    glyph_stroke_runs += 1;
                }
                alpha(*a, *blend, &mut alpha_lt1, &mut blend_non_normal);
                if let Some(r) = cp.glyph_runs.get(run.0 as usize) {
                    modes[(r.render_mode & 7) as usize] += 1;
                    glyphs_total += r.glyphs.len() as u64;
                    for g in r.glyphs.iter() {
                        max_gid = max_gid.max(g.glyph);
                    }
                    let t = &r.transform;
                    let diag = t.a.abs().max(t.d.abs()).max(1e-12);
                    if t.b.abs() > 1e-4 * diag || t.c.abs() > 1e-4 * diag {
                        rotated_runs += 1;
                    }
                }
            }
            DisplayOp::DrawImage { alpha: a, blend, .. } => {
                n_img += 1;
                alpha(*a, *blend, &mut alpha_lt1, &mut blend_non_normal);
            }
            DisplayOp::DrawShading { .. } => n_shade += 1,
            DisplayOp::BeginTransparencyGroup { .. } => n_group += 1,
            DisplayOp::BeginSoftMask { .. } => n_smask += 1,
            DisplayOp::BeginPaintOrigin(o) => match o {
                PaintOrigin::FormXObject => n_form += 1,
                PaintOrigin::TilingPatternCell => n_tiling_cell += 1,
                PaintOrigin::Type3Glyph => n_t3glyph += 1,
                PaintOrigin::AnnotationAppearance => n_annot += 1,
                _ => {}
            },
            _ => {}
        }
    }

    // Tiling "draws": fills/strokes whose paint is a tiling pattern would need
    // Paint's shape; the IR exposes the pattern table, so report both.
    let n_tilings = cp.tilings.len() as u64;

    let mut kinds = [0u64; 4]; // truetype, cff, type1, other
    for f in cp.fonts.iter() {
        match font_kind(&f.program) {
            "truetype" => kinds[0] += 1,
            "cff" => kinds[1] += 1,
            "type1" => kinds[2] += 1,
            _ => kinds[3] += 1,
        }
    }
    let synthetic = cp
        .fonts
        .iter()
        .filter(|f| f.synthetic_shear != 0.0 || f.synthetic_embolden_em != 0.0)
        .count();

    let crop = cp.bounds.crop;
    let mut o = String::with_capacity(1500);
    let _ = write!(
        o,
        "{{\"file\":\"{}\",\"pages\":{},\"page\":{},\"compile_ms\":{:.2},\"page_w_pt\":{:.1},\"page_h_pt\":{:.1},\"rotate\":{}",
        esc(path),
        pages,
        page,
        compile_ms,
        crop.width(),
        crop.height(),
        cp.bounds.rotate
    );
    let _ = write!(
        o,
        ",\"ops\":{},\"fills\":{},\"strokes\":{},\"clips\":{},\"clips_nonrect\":{},\"clips_text\":{},\"glyph_runs\":{},\"images\":{},\"shadings\":{},\"tilings\":{},\"tiling_draws\":{},\"shading_paint_draws\":{},\"tiling_cells\":{},\"groups\":{},\"soft_masks\":{},\"forms\":{},\"type3_glyph_origins\":{},\"annot_origins\":{}",
        cp.operations.len(),
        n_fill,
        n_stroke,
        n_clip,
        n_clip_nonrect,
        n_clip_text,
        n_runs,
        n_img,
        n_shade,
        n_tilings,
        pat_draws,
        shade_paint_draws,
        n_tiling_cell,
        n_group,
        n_smask,
        n_form,
        n_t3glyph,
        n_annot
    );
    fill.json("fill", &mut o);
    stroke.json("stroke", &mut o);
    clip.json("clip", &mut o);
    let _ = write!(
        o,
        ",\"max_verbs_fill\":{max_fill},\"max_verbs_stroke\":{max_stroke},\"max_verbs_clip\":{max_clip}"
    );
    let _ = write!(
        o,
        ",\"dashed\":{dashed},\"join_miter\":{},\"join_round\":{},\"join_bevel\":{},\"cap_butt\":{},\"cap_round\":{},\"cap_square\":{},\"w_lt05\":{},\"w_05_2\":{},\"w_gt2\":{}",
        joins[0], joins[1], joins[2], caps[0], caps[1], caps[2], widths[0], widths[1], widths[2]
    );
    let _ = write!(
        o,
        ",\"alpha_lt1\":{alpha_lt1},\"blend_non_normal\":{blend_non_normal}"
    );
    let _ = write!(
        o,
        ",\"tr0\":{},\"tr1\":{},\"tr2\":{},\"tr3\":{},\"tr4\":{},\"tr5\":{},\"tr6\":{},\"tr7\":{},\"glyph_stroke_runs\":{glyph_stroke_runs},\"glyphs\":{glyphs_total},\"rotated_runs\":{rotated_runs},\"max_gid\":{max_gid}",
        modes[0], modes[1], modes[2], modes[3], modes[4], modes[5], modes[6], modes[7]
    );
    let _ = write!(
        o,
        ",\"fonts\":{},\"font_truetype\":{},\"font_cff\":{},\"font_type1\":{},\"font_other\":{},\"font_synthetic\":{synthetic}",
        cp.fonts.len(),
        kinds[0],
        kinds[1],
        kinds[2],
        kinds[3]
    );
    let c = &cp.complexity;
    let _ = write!(
        o,
        ",\"features\":{},\"cx_ops\":{},\"cx_segments\":{},\"cx_glyphs\":{},\"cx_image_pixels\":{},\"cx_groups\":{},\"cx_peak_bytes\":{}}}",
        cp.features.bits(),
        c.operation_count,
        c.path_segment_count,
        c.glyph_count,
        c.image_pixels,
        c.transparency_group_count,
        c.estimated_peak_bytes
    );
    o
}

pub fn census(args: &[String]) {
    let mut a = crate::Args(args.to_vec());
    let max_pages: usize = a.value("--max-pages").and_then(|v| v.parse().ok()).unwrap_or(6).max(1);
    let budget_ms: f64 = a.value("--budget-ms").and_then(|v| v.parse().ok()).unwrap_or(3000.0);
    let path = a.positional().to_owned();

    let opened = std::panic::catch_unwind(AssertUnwindSafe(|| {
        let source: Arc<dyn PdfSource> = match MmapSource::open(&path) {
            Ok(s) => Arc::new(s),
            Err(e) => return Err(format!("open source: {e}")),
        };
        DocumentSnapshot::open(source, DocumentLimits::default()).map_err(|e| format!("open document: {e}"))
    }));
    let snap = match opened {
        Ok(Ok(s)) => s,
        Ok(Err(e)) => return error_line(&path, &e),
        Err(_) => return error_line(&path, "panic while opening"),
    };
    let count = snap.page_count();
    if count == 0 {
        return error_line(&path, "zero pages");
    }
    let n = (max_pages as u32).min(count);
    let mut pages: Vec<u32> = if n == 1 {
        vec![0]
    } else {
        (0..n)
            .map(|i| ((u64::from(i) * u64::from(count - 1)) / u64::from(n - 1)) as u32)
            .collect()
    };
    pages.dedup();

    let comp: PageCompiler = crate::compiler(None);
    let t_all = Instant::now();
    for page in pages {
        if t_all.elapsed().as_secs_f64() * 1e3 > budget_ms {
            break;
        }
        let t = Instant::now();
        let r = std::panic::catch_unwind(AssertUnwindSafe(|| {
            let mut ctx = ParseContext::new();
            comp.compile(&snap, PageIndex(page), &mut ctx)
        }));
        let compile_ms = t.elapsed().as_secs_f64() * 1e3;
        match r {
            Ok(Ok(cp)) => {
                let line = std::panic::catch_unwind(AssertUnwindSafe(|| {
                    census_page(&path, count, page, &cp, compile_ms)
                }));
                match line {
                    Ok(l) => println!("{l}"),
                    Err(_) => error_line(&path, &format!("panic in census page {page}")),
                }
            }
            Ok(Err(e)) => error_line(&path, &format!("compile page {page}: {e}")),
            Err(_) => error_line(&path, &format!("panic compiling page {page}")),
        }
    }
}
