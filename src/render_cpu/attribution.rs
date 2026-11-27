//! Diagnostic per-pixel attribution for differential rendering.
//!
//! This records the topmost prepared operation whose geometric coverage
//! reaches a pixel. It deliberately ignores paint alpha, blend modes, soft
//! masks, and knockout semantics: the result is a triage aid, not proof that
//! the attributed PDF object caused a visible difference.

use crate::page_ir::{DeviceRect, PaintLeaf, PaintOrigin};

use crate::render_cpu::kernels::mul_div_255;
use crate::render_cpu::mask::{ClipGeom, ClipMask, build_clip_mask};
use crate::render_cpu::prepared::{CpuPreparedPage, GlyphPlacement, PreparedGlyphRun, PreparedOp};
use crate::render_cpu::raster::RasterKernel;

/// Default minimum geometric coverage accepted as painted, in 8-bit alpha.
pub const DEFAULT_COVERAGE_THRESHOLD: u8 = 8;

/// Two canonical, row-major diagnostic planes.
#[derive(Debug, Clone)]
pub struct AttributionMap {
    pub width: u32,
    pub height: u32,
    /// `PaintLeaf as u8`; zero is `Unpainted`.
    pub leaf: Vec<u8>,
    /// `PaintOrigin as u8`; meaningful wherever `leaf != 0`.
    pub origin: Vec<u8>,
    pub coverage_threshold: u8,
}

/// Render attribution with [`DEFAULT_COVERAGE_THRESHOLD`].
pub fn render_attribution(page: &CpuPreparedPage) -> AttributionMap {
    render_attribution_with_threshold(page, DEFAULT_COVERAGE_THRESHOLD)
}

/// Render attribution using an explicit geometric-coverage threshold.
pub fn render_attribution_with_threshold(
    page: &CpuPreparedPage,
    coverage_threshold: u8,
) -> AttributionMap {
    let len = page.size.width as usize * page.size.height as usize;
    let mut map = AttributionMap {
        width: page.size.width,
        height: page.size.height,
        leaf: vec![PaintLeaf::Unpainted as u8; len],
        origin: vec![PaintOrigin::PageContent as u8; len],
        coverage_threshold,
    };
    let mut raster = RasterKernel::default();
    let mut clips: Vec<Option<ClipMask>> = (0..page.clips.len()).map(|_| None).collect();

    let mut i = 0usize;
    while i < page.ops.len() {
        match &page.ops[i] {
            PreparedOp::Draw(cmd) => {
                let leaf = if cmd.shading.is_some() {
                    PaintLeaf::Shading
                } else {
                    PaintLeaf::Path
                };
                stamp_path(
                    page,
                    &mut map,
                    &mut raster,
                    &mut clips,
                    cmd.subpath_range,
                    cmd.rule,
                    cmd.bounds,
                    cmd.clip.filter(|_| cmd.clip_has_mask),
                    leaf,
                    cmd.origin,
                );
                i += 1;
            }
            PreparedOp::TiledFill(t) => {
                stamp_path(
                    page,
                    &mut map,
                    &mut raster,
                    &mut clips,
                    t.fill_subpaths,
                    t.fill_rule,
                    t.bounds,
                    t.clip.filter(|_| t.clip_has_mask),
                    PaintLeaf::TilingPattern,
                    // The visible paint comes from replicated pattern-cell
                    // content; this is the innermost containing construct.
                    PaintOrigin::TilingPatternCell,
                );
                i += 1;
            }
            PreparedOp::GlyphRun(run) => {
                ensure_clip(
                    page,
                    &mut raster,
                    &mut clips,
                    run.clip.filter(|_| run.clip_has_mask),
                );
                stamp_glyph_run(&mut map, run, &clips);
                i += 1;
            }
            PreparedOp::Image(image) => {
                ensure_clip(
                    page,
                    &mut raster,
                    &mut clips,
                    image.clip.filter(|_| image.clip_has_mask),
                );
                stamp_image(&mut map, image, &clips);
                i += 1;
            }
            PreparedOp::PushSoftMask { content_end, .. } => {
                // The enclosed ops paint only into the mask's offscreen.
                i = (*content_end as usize).max(i + 1);
            }
            PreparedOp::BeginGroup { .. }
            | PreparedOp::EndGroup
            | PreparedOp::PushSoftMaskNone
            | PreparedOp::PopSoftMask => i += 1,
        }
    }
    map
}

/// Where the text pass stamped: one flag per device pixel, row-major, true
/// where a glyph run's geometric coverage reached the pixel.
///
/// This mirrors the glyph-run arm of [`render_attribution_with_threshold`] —
/// same coverage threshold, same run-bounds gating, same clip-mask
/// combination, same soft-mask scope skipping — recording *that* the text
/// pass stamped instead of *what* won. Forensic occlusion analysis
/// intersects this mask with the attribution map: a glyph pixel whose winner
/// is later path or image paint is covered text, and the painter order is
/// proven rather than assumed — the glyph stamped first, so whatever won
/// over it painted later. Text lowered as outlines or placement boxes
/// (non-fill render modes, rotated or oversized runs, fonts without outline
/// programs) stamps `Path`, not `Text`, and never appears here; that bounds
/// this mask's recall, never its precision.
pub fn glyph_stamp_mask(page: &CpuPreparedPage, threshold: u8) -> Vec<bool> {
    let width = page.size.width as usize;
    let height = page.size.height as usize;
    let mut mask = vec![false; width.saturating_mul(height)];
    let mut raster = RasterKernel::default();
    let mut clips: Vec<Option<ClipMask>> = (0..page.clips.len()).map(|_| None).collect();

    let mut i = 0usize;
    while i < page.ops.len() {
        match &page.ops[i] {
            PreparedOp::GlyphRun(run) => {
                ensure_clip(
                    page,
                    &mut raster,
                    &mut clips,
                    run.clip.filter(|_| run.clip_has_mask),
                );
                let reach = GlyphReach {
                    clips: &clips,
                    width: page.size.width as i32,
                    height: page.size.height as i32,
                    threshold,
                };
                mark_glyph_run(&mut mask, width, run, reach);
                i += 1;
            }
            PreparedOp::PushSoftMask { content_end, .. } => {
                // The enclosed ops paint only into the mask's offscreen.
                i = (*content_end as usize).max(i + 1);
            }
            _ => i += 1,
        }
    }
    mask
}

/// Where a glyph run's coverage counts as painted: its clip masks, the
/// `width × height` device, and the coverage threshold.
#[derive(Clone, Copy)]
struct GlyphReach<'a> {
    clips: &'a [Option<ClipMask>],
    width: i32,
    height: i32,
    threshold: u8,
}

/// Visit, in row-major order, every device pixel of one glyph placement
/// inside the run's bounds and the device whose glyph coverage, combined
/// with the run's clip coverage there, reaches the threshold.
fn for_each_glyph_pixel(
    run: &PreparedGlyphRun,
    placement: &GlyphPlacement,
    reach: GlyphReach<'_>,
    f: &mut impl FnMut(i32, i32),
) {
    let bitmap = &placement.bitmap;
    for by in 0..bitmap.height as i32 {
        let y = placement.dy + by;
        if y < run.bounds.y
            || y >= run.bounds.y + run.bounds.height as i32
            || y < 0
            || y >= reach.height
        {
            continue;
        }
        for bx in 0..bitmap.width as i32 {
            let x = placement.dx + bx;
            if x < run.bounds.x
                || x >= run.bounds.x + run.bounds.width as i32
                || x < 0
                || x >= reach.width
            {
                continue;
            }
            let glyph = bitmap.cov[by as usize * bitmap.width as usize + bx as usize];
            let coverage = combined_coverage(
                glyph,
                clip_at(reach.clips, run.clip.filter(|_| run.clip_has_mask), x, y),
            );
            if coverage >= reach.threshold {
                f(x, y);
            }
        }
    }
}

/// Stamp `Text` on every pixel a glyph run's placements cover at the map's
/// threshold.
fn stamp_glyph_run(map: &mut AttributionMap, run: &PreparedGlyphRun, clips: &[Option<ClipMask>]) {
    let reach = GlyphReach {
        clips,
        width: map.width as i32,
        height: map.height as i32,
        threshold: map.coverage_threshold,
    };
    for placement in &run.placements {
        for_each_glyph_pixel(run, placement, reach, &mut |x, y| {
            map.stamp(x as u32, y as u32, PaintLeaf::Text, run.origin)
        });
    }
}

/// Set the flag of every pixel a glyph run's placements cover at
/// `reach.threshold` in the row-major `mask`, `width` flags per row.
fn mark_glyph_run(mask: &mut [bool], width: usize, run: &PreparedGlyphRun, reach: GlyphReach<'_>) {
    for placement in &run.placements {
        for_each_glyph_pixel(run, placement, reach, &mut |x, y| {
            mask[y as usize * width + x as usize] = true
        });
    }
}

/// Stamp every device pixel whose centre maps into an image's unit square
/// and whose clip coverage reaches the map's threshold.
fn stamp_image(
    map: &mut AttributionMap,
    image: &crate::render_cpu::image::PreparedImage,
    clips: &[Option<ClipMask>],
) {
    let threshold = map.coverage_threshold;
    let (width, height) = (map.width, map.height);
    for_each_pixel(image.bounds, width, height, |x, y| {
        let dx = x as f64 + 0.5;
        let dy = y as f64 + 0.5;
        let u = image.inv.a * dx + image.inv.c * dy + image.inv.e;
        let v = image.inv.b * dx + image.inv.d * dy + image.inv.f;
        if !((0.0..1.0).contains(&u) && (0.0..1.0).contains(&v)) {
            return;
        }
        let coverage = clip_at(
            clips,
            image.clip.filter(|_| image.clip_has_mask),
            x as i32,
            y as i32,
        );
        if coverage >= threshold {
            map.stamp(x, y, PaintLeaf::Image, image.origin);
        }
    });
}

impl AttributionMap {
    #[inline]
    fn stamp(&mut self, x: u32, y: u32, leaf: PaintLeaf, origin: PaintOrigin) {
        let index = y as usize * self.width as usize + x as usize;
        self.leaf[index] = leaf as u8;
        self.origin[index] = origin as u8;
    }
}

#[allow(clippy::too_many_arguments)]
fn stamp_path(
    page: &CpuPreparedPage,
    map: &mut AttributionMap,
    raster: &mut RasterKernel,
    clips: &mut [Option<ClipMask>],
    subpath_range: (u32, u32),
    rule: crate::render_cpu::raster::FillRule,
    bounds: DeviceRect,
    clip: Option<u32>,
    leaf: PaintLeaf,
    origin: PaintOrigin,
) {
    ensure_clip(page, raster, clips, clip);
    let subpaths = &page.subpaths[subpath_range.0 as usize..subpath_range.1 as usize];
    let width = map.width as usize;
    let height = map.height as usize;
    let threshold = map.coverage_threshold;
    raster.fill(
        &page.points,
        subpaths,
        width,
        height,
        rule,
        |y, x0, x1, coverage| {
            if y < bounds.y.max(0) as usize
                || y >= (bounds.y + bounds.height as i32).max(0) as usize
            {
                return;
            }
            let left = x0.max(bounds.x.max(0) as usize);
            let right_exclusive = (bounds.x + bounds.width as i32).max(0) as usize;
            if left >= right_exclusive {
                return;
            }
            let right = x1.min(right_exclusive - 1);
            for x in left..=right {
                let combined =
                    combined_coverage(coverage[x], clip_at(clips, clip, x as i32, y as i32));
                if combined >= threshold {
                    map.stamp(x as u32, y as u32, leaf, origin);
                }
            }
        },
    );
}

fn ensure_clip(
    page: &CpuPreparedPage,
    raster: &mut RasterKernel,
    clips: &mut [Option<ClipMask>],
    clip: Option<u32>,
) {
    if let Some(id) = clip
        && clips[id as usize].is_none()
    {
        clips[id as usize] = Some(build_clip_mask(raster, ClipGeom::of(page), id));
    }
}

#[inline]
fn combined_coverage(paint: u8, clip: u8) -> u8 {
    mul_div_255(paint as u16, clip as u16) as u8
}

#[inline]
fn clip_at(clips: &[Option<ClipMask>], clip: Option<u32>, x: i32, y: i32) -> u8 {
    let Some(mask) = clip.and_then(|id| clips[id as usize].as_ref()) else {
        return 255;
    };
    if x < mask.bounds.x
        || y < mask.bounds.y
        || x >= mask.bounds.x + mask.bounds.width as i32
        || y >= mask.bounds.y + mask.bounds.height as i32
    {
        return mask.outside;
    }
    let lx = (x - mask.bounds.x) as usize;
    let ly = (y - mask.bounds.y) as usize;
    mask.data[ly * mask.stride() + lx]
}

fn for_each_pixel(bounds: DeviceRect, width: u32, height: u32, mut f: impl FnMut(u32, u32)) {
    let x0 = bounds.x.max(0) as u32;
    let y0 = bounds.y.max(0) as u32;
    let x1 = (bounds.x + bounds.width as i32).max(0).min(width as i32) as u32;
    let y1 = (bounds.y + bounds.height as i32).max(0).min(height as i32) as u32;
    for y in y0..y1 {
        for x in x0..x1 {
            f(x, y);
        }
    }
}

#[cfg(all(test, not(feature = "profiling")))]
mod tests {
    use super::*;
    use crate::page_ir::{BlendMode, DeviceSize};
    use crate::render_cpu::prepared::{DrawClass, PreparedCommand, RenderDiagnostics};

    fn page(
        ops: Vec<PreparedOp>,
        points: Vec<[f32; 2]>,
        subpaths: Vec<(usize, usize)>,
    ) -> CpuPreparedPage {
        CpuPreparedPage {
            size: DeviceSize {
                width: 4,
                height: 4,
            },
            ops,
            clips: Vec::new(),
            points,
            subpaths,
            codecs: crate::image::CodecRegistry::default(),
            decode_limits: crate::image::DecodeLimits::default(),
            hinting: crate::font::HintingPolicy::None,
            color_policy: crate::render_api::RenderColorPolicy::Original,
            diagnostics: RenderDiagnostics::default(),
            image_cache: None,
            decode_floor: None,
            sanity_extent: 4.0,
            rows: (0, u32::MAX),
            op_extent: None,
            strip_decodes: None,
        }
    }

    fn draw(range: (u32, u32), bounds: DeviceRect, origin: PaintOrigin) -> PreparedOp {
        PreparedOp::Draw(PreparedCommand {
            origin,
            class: DrawClass::SolidPath,
            subpath_range: range,
            rule: crate::render_cpu::raster::FillRule::NonZero,
            rgb: [0, 0, 0],
            premul: [0, 0, 0, 255],
            alpha: 255,
            opaque: true,
            bounds,
            clip: None,
            clip_has_mask: false,
            blend: BlendMode::Normal,
            shading: None,
            stencil: None,
        })
    }

    #[test]
    fn blank_page_is_entirely_unpainted() {
        let map = render_attribution(&page(Vec::new(), Vec::new(), Vec::new()));
        assert!(
            map.leaf
                .iter()
                .all(|&value| value == PaintLeaf::Unpainted as u8)
        );
    }

    #[test]
    fn later_painter_replaces_origin() {
        let points = vec![
            [0.0, 0.0],
            [4.0, 0.0],
            [4.0, 4.0],
            [0.0, 4.0],
            [1.0, 1.0],
            [3.0, 1.0],
            [3.0, 3.0],
            [1.0, 3.0],
        ];
        let ops = vec![
            draw(
                (0, 1),
                DeviceRect {
                    x: 0,
                    y: 0,
                    width: 4,
                    height: 4,
                },
                PaintOrigin::FormXObject,
            ),
            draw(
                (1, 2),
                DeviceRect {
                    x: 1,
                    y: 1,
                    width: 2,
                    height: 2,
                },
                PaintOrigin::AnnotationAppearance,
            ),
        ];
        let map = render_attribution(&page(ops, points, vec![(0, 4), (4, 8)]));
        assert_eq!(map.leaf[0], PaintLeaf::Path as u8);
        assert_eq!(map.origin[0], PaintOrigin::FormXObject as u8);
        assert_eq!(
            map.origin[1 * 4 + 1],
            PaintOrigin::AnnotationAppearance as u8
        );
    }
}
