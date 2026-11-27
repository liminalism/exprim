//! Lazy region-local clip masks (performance advice §8).
//!
//! A non-rectangular clip contributes an `Alpha8` coverage mask. It is built
//! **on demand**, once per page execution, sized to the clip's device
//! envelope (never the full page), and cached by dense `ClipId`. Purely
//! rectangular clip chains never reach here — they are handled by arithmetic
//! bounds intersection during lowering.

use crate::page_ir::DeviceRect;

use crate::render_cpu::kernels::mul_div_255;
use crate::render_cpu::prepared::{ClipKind, CpuPreparedPage, PreparedClip, PreparedOp};
use crate::render_cpu::raster::{FillRule, RasterKernel};

/// The immutable, `Sync` slice of a prepared page that mask building reads. The
/// full [`CpuPreparedPage`] is `!Sync` (diagnostic cells), so masks are built
/// from this borrow instead — letting a whole page's masks build in parallel.
#[derive(Clone, Copy)]
pub struct ClipGeom<'a> {
    pub clips: &'a [PreparedClip],
    pub subpaths: &'a [(usize, usize)],
    pub points: &'a [[f32; 2]],
    pub width: usize,
    pub height: usize,
}

impl<'a> ClipGeom<'a> {
    pub fn of(page: &'a CpuPreparedPage) -> Self {
        Self {
            clips: &page.clips,
            subpaths: &page.subpaths,
            points: &page.points,
            width: page.size.width as usize,
            height: page.size.height as usize,
        }
    }
}

/// An 8-bit coverage mask over a device rectangle.
#[derive(Debug)]
pub struct ClipMask {
    pub bounds: DeviceRect,
    /// `bounds.width * bounds.height` coverage bytes, row-major.
    pub data: Vec<u8>,
    /// Coverage value for pixels *outside* `bounds`. Clip masks and ordinary
    /// soft masks use 0 (fully masked); a luminosity soft mask with a `/BC`
    /// backdrop uses the backdrop's luminosity (ISO 32000-1 §11.6.5.2).
    pub outside: u8,
    /// Whether `data` is entirely 255 (the clip is exactly its rectangular
    /// envelope, so a draw can drop the mask and use `bounds` alone). Computed
    /// once at build so consumers never rescan the whole mask — a per-glyph-run
    /// rescan was O(runs × mask area), the dominant cost on clipped-text pages.
    pub all_opaque: bool,
}

impl ClipMask {
    #[inline]
    pub fn stride(&self) -> usize {
        self.bounds.width as usize
    }

    /// Resolve this mask's geometry once, for per-pixel sampling in a draw
    /// loop. `outside` is 0 — a clip has no backdrop outside its bounds.
    #[inline]
    pub fn clip_window(&self) -> MaskWindow<'_> {
        MaskWindow::new(self, 0)
    }

    /// As [`Self::clip_window`], but for a soft mask, whose pixels outside
    /// `bounds` take the mask's own backdrop value (ISO 32000-1 §11.6.5.2).
    #[inline]
    pub fn soft_window(&self) -> MaskWindow<'_> {
        MaskWindow::new(self, self.outside)
    }
}

/// A mask's geometry, resolved once so per-pixel sampling is pure arithmetic.
///
/// Reading `bounds.x/y/width/height` and calling `stride()` is loop-invariant
/// across a whole draw, but the obvious spelling repeats all five per pixel.
/// The glyph-run and tiling paths already hoist this by hand; this is the same
/// hoist as a value, so every draw loop gets it without the copy-paste.
#[derive(Debug, Clone, Copy)]
pub struct MaskWindow<'a> {
    data: &'a [u8],
    x0: usize,
    y0: usize,
    width: usize,
    height: usize,
    outside: u16,
}

impl<'a> MaskWindow<'a> {
    #[inline]
    fn new(mask: &'a ClipMask, outside: u8) -> Self {
        Self {
            data: &mask.data,
            x0: mask.bounds.x as usize,
            y0: mask.bounds.y as usize,
            width: mask.bounds.width as usize,
            height: mask.bounds.height as usize,
            outside: outside as u16,
        }
    }

    /// Coverage at an absolute device pixel.
    #[inline]
    pub fn coverage(&self, x: usize, y: usize) -> u16 {
        let (Some(lx), Some(ly)) = (x.checked_sub(self.x0), y.checked_sub(self.y0)) else {
            return self.outside;
        };
        if lx < self.width && ly < self.height {
            self.data[ly * self.width + lx] as u16
        } else {
            self.outside
        }
    }
}

/// Smallest byte budget of a page's clip-mask store. Below it the store holds
/// dozens of masks even for a small page, which keeps the parallel batch
/// build as wide as the pool on every page that is cheap to hold.
const CLIP_MASK_BUDGET_FLOOR: usize = 64 << 20;

/// The byte budget of a page's clip masks: eight full-page coverage planes
/// (two RGBA surfaces' worth), and at least [`CLIP_MASK_BUDGET_FLOOR`].
/// `PDF_RENDERER_CLIPMASK_MB` overrides it (0 = hold one mask at a time), for
/// paired A/B isolation on one binary. Read once.
fn clip_mask_budget(page: &CpuPreparedPage) -> usize {
    use std::sync::OnceLock;
    static OVERRIDE: OnceLock<Option<usize>> = OnceLock::new();
    let forced = *OVERRIDE.get_or_init(|| {
        std::env::var("PDF_RENDERER_CLIPMASK_MB")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .map(|mb| mb.saturating_mul(1 << 20))
    });
    forced.unwrap_or_else(|| {
        let plane = (page.size.width as usize).saturating_mul(page.size.height as usize);
        plane.saturating_mul(8).max(CLIP_MASK_BUDGET_FLOOR)
    })
}

/// Below this many masks, rayon fan-out costs more than it saves.
const PARALLEL_MASK_THRESHOLD: usize = 16;

/// When and for how long each mask of a page is needed, derived from the op
/// stream once per execution.
#[derive(Debug)]
struct MaskPlan {
    /// Bytes the store may hold. One mask is always admitted, so a single
    /// mask larger than the budget still builds.
    budget: usize,
    /// `(op index of the last consumer, mask source)`, ascending. Ops run in
    /// increasing index order, each once, so a moving cursor releases every
    /// mask right after the op that used it last.
    releases: Vec<(u32, u32)>,
    next_release: usize,
    /// Per mask source: op index of its last consumer.
    last_use: Vec<u32>,
    /// Mask sources in first-use order: what a parallel batch builds next.
    order: Vec<u32>,
    next_order: usize,
    /// Build ahead across the rayon pool (many masks) or one at a time.
    parallel: bool,
    /// Recency clock and per-slot stamp, for evicting under the budget.
    clock: u32,
    stamp: Vec<u32>,
    /// Every op with a smaller index has finished (0 before any).
    finished_below: u32,
}

/// The clip masks of one page execution.
///
/// A mask is a pure function of the prepared page, so when it is built, kept
/// or dropped never changes a pixel: this store only decides residency. A
/// page-level store ([`ClipMasks::for_page`]) drops each mask after its last
/// consumer, keeps the live ones under a byte budget (evicting the least
/// recently used, which is rebuilt if it is needed again), and on clip-dense
/// pages builds ahead in budget-sized parallel batches. Before this, every
/// consumed mask was built up front and held until the page ended: 200
/// page-sized clips at 600 dpi held 3.5 GB beside a 135 MB surface.
#[derive(Debug)]
pub(crate) struct ClipMasks {
    slots: Vec<Option<ClipMask>>,
    plan: Option<MaskPlan>,
    live_bytes: usize,
    peak_bytes: usize,
    built_serial: u64,
    built_parallel: u64,
    built_bytes: u64,
}

impl ClipMasks {
    /// A store that builds on demand and keeps everything until dropped: for
    /// tiling cells and the preparation seams, whose op streams are short
    /// lived or not walked in full.
    pub(crate) fn unmanaged(clips: usize) -> Self {
        Self {
            slots: (0..clips).map(|_| None).collect(),
            plan: None,
            live_bytes: 0,
            peak_bytes: 0,
            built_serial: 0,
            built_parallel: 0,
            built_bytes: 0,
        }
    }

    /// The store for a whole page execution, planned from its op stream.
    #[cfg(any(test, feature = "profiling"))]
    pub(crate) fn for_page(page: &CpuPreparedPage) -> Self {
        Self::for_page_with_budget(page, clip_mask_budget(page))
    }

    /// Plan the next chunk of a page that is lowered and painted in pieces:
    /// `page.ops` are that chunk's ops, `page.clips` every clip so far, and
    /// `live_clips` the clip stack in force after the chunk. A mask a later
    /// chunk can still draw under (its clip is still on the stack) is kept
    /// past its last use here rather than rebuilt per chunk; any other mask
    /// left from an earlier chunk that this one does not use is dropped.
    pub(crate) fn begin_chunk(&mut self, page: &CpuPreparedPage, live_clips: &[u32]) {
        let mut pinned = vec![false; page.clips.len()];
        for &cid in live_clips {
            if let Some(source) = page
                .clips
                .get(cid as usize)
                .and_then(|clip| clip.mask_source)
                .filter(|&source| (source as usize) < pinned.len())
            {
                pinned[source as usize] = true;
            }
        }
        let mut planned = Self::for_page_with_budget(page, clip_mask_budget(page));
        let wanted: Vec<bool> = match &planned.plan {
            Some(plan) => {
                let mut wanted = vec![false; page.clips.len()];
                for &cid in &plan.order {
                    wanted[cid as usize] = true;
                }
                wanted
            }
            None => vec![false; page.clips.len()],
        };
        if let Some(plan) = &mut planned.plan {
            plan.releases.retain(|&(_, cid)| !pinned[cid as usize]);
        }
        self.slots.resize_with(page.clips.len(), || None);
        for source in 0..self.slots.len() {
            if !pinned[source] && !wanted[source] {
                self.drop_mask(source);
            }
        }
        self.plan = planned.plan;
    }

    pub(crate) fn for_page_with_budget(page: &CpuPreparedPage, budget: usize) -> Self {
        let mut store = Self::unmanaged(page.clips.len());
        const UNUSED: u32 = u32::MAX;
        let mut first_use = vec![UNUSED; page.clips.len()];
        let mut last_use = vec![0u32; page.clips.len()];
        let mut wanted = 0usize;
        for (index, op) in page.ops.iter().enumerate() {
            let clip = match op {
                PreparedOp::Draw(c) if c.clip_has_mask => c.clip,
                PreparedOp::GlyphRun(g) if g.clip_has_mask => g.clip,
                PreparedOp::Image(i) if i.clip_has_mask => i.clip,
                PreparedOp::TiledFill(t) if t.clip_has_mask => t.clip,
                _ => None,
            };
            let Some(cid) = clip else { continue };
            // Exactly the source every consumer resolves.
            let Some(source) = page
                .clips
                .get(cid as usize)
                .map(|clip| clip.mask_source.unwrap_or(cid) as usize)
                .filter(|&source| source < first_use.len())
            else {
                continue;
            };
            let index = index.min(u32::MAX as usize - 1) as u32;
            if first_use[source] == UNUSED {
                first_use[source] = index;
                wanted += 1;
            }
            last_use[source] = index;
        }
        if wanted == 0 {
            return store;
        }
        let mut order: Vec<u32> = (0..page.clips.len() as u32)
            .filter(|&cid| first_use[cid as usize] != UNUSED)
            .collect();
        order.sort_unstable_by_key(|&cid| first_use[cid as usize]);
        let mut releases: Vec<(u32, u32)> = order
            .iter()
            .map(|&cid| (last_use[cid as usize], cid))
            .collect();
        releases.sort_unstable();
        store.plan = Some(MaskPlan {
            budget,
            releases,
            next_release: 0,
            last_use,
            order,
            next_order: 0,
            parallel: wanted >= PARALLEL_MASK_THRESHOLD,
            clock: 0,
            stamp: vec![0; page.clips.len()],
            finished_below: 0,
        });
        store
    }

    /// The mask of `source`, if it is resident.
    #[inline]
    pub(crate) fn get(&self, source: usize) -> Option<&ClipMask> {
        self.slots.get(source).and_then(Option::as_ref)
    }

    #[inline]
    pub(crate) fn is_resident(&self, source: usize) -> bool {
        self.get(source).is_some()
    }

    /// Most bytes of mask coverage resident at once so far.
    pub(crate) fn peak_bytes(&self) -> usize {
        self.peak_bytes
    }

    /// `(masks built one at a time, masks built in parallel batches, bytes
    /// built)` since the last call.
    pub(crate) fn take_counters(&mut self) -> (u64, u64, u64) {
        let counters = (self.built_serial, self.built_parallel, self.built_bytes);
        self.built_serial = 0;
        self.built_parallel = 0;
        self.built_bytes = 0;
        counters
    }

    fn mask_bytes(page: &CpuPreparedPage, source: usize) -> usize {
        page.clips.get(source).map_or(0, |clip| {
            (clip.mask_bounds.width as usize).saturating_mul(clip.mask_bounds.height as usize)
        })
    }

    fn insert(&mut self, source: usize, mask: ClipMask) {
        let bytes = mask.data.len();
        if let Some(slot) = self.slots.get_mut(source) {
            if let Some(old) = slot.replace(mask) {
                self.live_bytes = self.live_bytes.saturating_sub(old.data.len());
            }
            self.live_bytes += bytes;
            self.peak_bytes = self.peak_bytes.max(self.live_bytes);
            self.built_bytes += bytes as u64;
        }
        if let Some(plan) = &mut self.plan {
            plan.clock = plan.clock.wrapping_add(1);
            if let Some(stamp) = plan.stamp.get_mut(source) {
                *stamp = plan.clock;
            }
        }
    }

    fn drop_mask(&mut self, source: usize) {
        if let Some(old) = self.slots.get_mut(source).and_then(Option::take) {
            self.live_bytes = self.live_bytes.saturating_sub(old.data.len());
        }
    }

    /// Evict least-recently-used masks other than `keep` until `incoming`
    /// more bytes fit the budget (or nothing else is resident).
    fn make_room(&mut self, keep: usize, incoming: usize) {
        let Some(budget) = self.plan.as_ref().map(|plan| plan.budget) else {
            return;
        };
        while self.live_bytes.saturating_add(incoming) > budget {
            let victim = self.plan.as_ref().and_then(|plan| {
                self.slots
                    .iter()
                    .enumerate()
                    .filter(|(source, slot)| *source != keep && slot.is_some())
                    .min_by_key(|(source, _)| plan.stamp.get(*source).copied().unwrap_or(0))
                    .map(|(source, _)| source)
            });
            match victim {
                Some(source) => self.drop_mask(source),
                None => break,
            }
        }
    }

    /// The sources a parallel batch builds: `first` (when given), then the
    /// not-yet-built sources in first-use order while they fit the budget.
    fn next_batch(&mut self, page: &CpuPreparedPage, first: Option<usize>) -> Vec<u32> {
        let mut batch: Vec<u32> = Vec::new();
        let mut bytes = self.live_bytes;
        if let Some(source) = first {
            bytes = bytes.saturating_add(Self::mask_bytes(page, source));
            batch.push(source as u32);
        }
        let Some(plan) = self.plan.as_mut() else {
            return batch;
        };
        while let Some(&cid) = plan.order.get(plan.next_order) {
            let source = cid as usize;
            let skip = Some(source) == first
                || self.slots.get(source).is_some_and(Option::is_some)
                || plan
                    .last_use
                    .get(source)
                    .is_some_and(|&last| last < plan.finished_below);
            if !skip {
                let size = Self::mask_bytes(page, source);
                if bytes.saturating_add(size) > plan.budget {
                    break;
                }
                bytes += size;
                batch.push(cid);
            }
            plan.next_order += 1;
        }
        batch
    }

    /// Build the first batch before the draw loop, as the executor always did
    /// for clip-dense pages (the build is the dominant cost there and fans out
    /// across the pool). A no-op for pages with few masks.
    pub(crate) fn prebuild(&mut self, page: &CpuPreparedPage, raster: &mut RasterKernel) -> bool {
        // Cooperative cancellation: probed once per page here, as the pre-pass
        // always did, whether or not there is anything to build. An already
        // cancelled request skips the build; the draw loop reports it.
        if page
            .decode_limits
            .should_cancel
            .as_deref()
            .is_some_and(|cancel| cancel())
        {
            return true;
        }
        if !self.plan.as_ref().is_some_and(|plan| plan.parallel) {
            return true;
        }
        let batch = self.next_batch(page, None);
        self.build_batch(page, raster, &batch)
    }

    /// Make `source`'s mask resident. `false` means the request was cancelled
    /// mid-build: nothing was stored and the caller must stop drawing.
    pub(crate) fn ensure(
        &mut self,
        page: &CpuPreparedPage,
        raster: &mut RasterKernel,
        source: usize,
    ) -> bool {
        if self.is_resident(source) {
            if let Some(plan) = &mut self.plan {
                plan.clock = plan.clock.wrapping_add(1);
                if let Some(stamp) = plan.stamp.get_mut(source) {
                    *stamp = plan.clock;
                }
            }
            return true;
        }
        self.make_room(source, Self::mask_bytes(page, source));
        let batch = if self.plan.as_ref().is_some_and(|plan| plan.parallel) {
            self.next_batch(page, Some(source))
        } else {
            vec![source as u32]
        };
        self.build_batch(page, raster, &batch)
    }

    fn build_batch(
        &mut self,
        page: &CpuPreparedPage,
        raster: &mut RasterKernel,
        batch: &[u32],
    ) -> bool {
        let geom = ClipGeom::of(page);
        let should_cancel = page.decode_limits.should_cancel.as_deref();
        if let [cid] = batch {
            #[cfg(feature = "profiling")]
            let caller_class = std::mem::replace(
                &mut raster.prof.class,
                crate::render_cpu::raster::RasterClass::Clip,
            );
            let mask = build_clip_mask_cancellable(raster, geom, *cid, should_cancel);
            #[cfg(feature = "profiling")]
            {
                raster.prof.class = caller_class;
            }
            let Some(mask) = mask else {
                return false;
            };
            self.insert(*cid as usize, mask);
            self.built_serial += 1;
            return true;
        }
        if batch.is_empty() {
            return true;
        }
        use rayon::prelude::*;
        // One scratch kernel per pool thread (grows on first fill). Each mask
        // is byte-identical whatever thread or order builds it.
        #[cfg(feature = "profiling")]
        let built: Vec<(u32, Option<ClipMask>)> = {
            // Each pool thread's kernel carries its own attribution; hand it
            // back with the mask and merge serially (summed CPU time).
            let built: Vec<(u32, Option<ClipMask>, crate::render_cpu::raster::RasterProf)> = batch
                .par_iter()
                .map_init(RasterKernel::default, |rk, &cid| {
                    rk.prof.class = crate::render_cpu::raster::RasterClass::Clip;
                    let mask = build_clip_mask_cancellable(rk, geom, cid, should_cancel);
                    (cid, mask, std::mem::take(&mut rk.prof))
                })
                .collect();
            built
                .into_iter()
                .map(|(cid, mask, worker_prof)| {
                    raster.prof.merge(&worker_prof);
                    (cid, mask)
                })
                .collect()
        };
        #[cfg(not(feature = "profiling"))]
        let built: Vec<(u32, Option<ClipMask>)> = {
            batch
                .par_iter()
                .map_init(RasterKernel::default, |rk, &cid| {
                    (
                        cid,
                        build_clip_mask_cancellable(rk, geom, cid, should_cancel),
                    )
                })
                .collect()
        };
        let mut complete = true;
        for (cid, mask) in built {
            match mask {
                Some(mask) => {
                    self.insert(cid as usize, mask);
                    self.built_parallel += 1;
                }
                // Cancelled mid-batch: keep the finished ones, report it.
                None => complete = false,
            }
        }
        complete
    }

    /// Op `index` has finished: drop every mask whose last consumer it was.
    #[inline]
    pub(crate) fn op_done(&mut self, index: usize) {
        let Some(plan) = self.plan.as_mut() else {
            return;
        };
        let index = index.min(u32::MAX as usize) as u32;
        plan.finished_below = plan.finished_below.max(index.saturating_add(1));
        while let Some(&(last, cid)) = plan.releases.get(plan.next_release) {
            if last > index {
                break;
            }
            plan.next_release += 1;
            if let Some(old) = self.slots.get_mut(cid as usize).and_then(Option::take) {
                self.live_bytes = self.live_bytes.saturating_sub(old.data.len());
            }
        }
    }
}

/// Build the combined mask for clip `cid`: the product of every path clip in
/// its ancestor chain, over the clip's rectangular envelope. Rectangular
/// ancestors contribute only to the envelope (already reflected in `bounds`).
pub fn build_clip_mask(raster: &mut RasterKernel, geom: ClipGeom, cid: u32) -> ClipMask {
    // The non-rendering attribution path has no cancellation token.
    if let Some(mask) = build_clip_mask_cancellable(raster, geom, cid, None) {
        return mask;
    }
    // `None` is unreachable without a probe, but keep the never-panic policy
    // even if that invariant changes later.
    let bounds = geom.clips[cid as usize].mask_bounds;
    ClipMask {
        bounds,
        data: vec![0; bounds.width as usize * bounds.height as usize],
        outside: 0,
        all_opaque: false,
    }
}

/// Cancellable form of [`build_clip_mask`]. A cancelled partial mask is
/// discarded rather than cached or consumed by a draw.
///
/// The mask covers the clip's `mask_bounds`: its envelope, cut to the rows of
/// the strip being painted when the page is rendered in strips. Coverage is
/// rasterized in absolute device coordinates over just those rows, so the
/// bytes kept are the bytes a whole-envelope mask has there.
pub fn build_clip_mask_cancellable(
    raster: &mut RasterKernel,
    geom: ClipGeom,
    cid: u32,
    should_cancel: Option<&(dyn Fn() -> bool + Send + Sync)>,
) -> Option<ClipMask> {
    let cb = geom.clips[cid as usize].mask_bounds;
    let w = cb.width as usize;
    let h = cb.height as usize;

    // `acc` starts empty (fully masked). The first path clip is filled directly
    // into it; later path clips intersect via `tmp` + multiply. Most chains hold
    // exactly one path clip (rect ancestors contribute only to `bounds`), so the
    // common case does one pass — no zeroed scratch, no per-pixel multiply. On
    // clip-dense maps this is the dominant per-page cost, once per basin polygon.
    let mut acc = vec![0u8; w * h];
    let mut tmp: Vec<u8> = Vec::new();
    let mut path_seen = 0u32;

    let mut cur = Some(cid);
    while let Some(id) = cur {
        let clip = &geom.clips[id as usize];
        cur = clip.parent;
        let ClipKind::Path {
            subpath_range,
            rule,
        } = clip.kind
        else {
            continue;
        };
        let subs = &geom.subpaths[subpath_range.0 as usize..subpath_range.1 as usize];
        path_seen += 1;
        if path_seen == 1 {
            // First (usually only) path: write coverage straight into `acc`,
            // which is 0 everywhere the fill does not touch (correctly masked).
            // Only the envelope's rows are kept, so only they are
            // rasterized: a clip path far taller than its envelope (a
            // page-sized path under a small rect ancestor) costs the
            // envelope, not the page.
            if !fill_envelope(raster, geom, subs, rule, cb, should_cancel, &mut acc) {
                return None;
            }
            continue;
        }
        // Additional path clip: rasterize into `tmp` and multiply in.
        if tmp.is_empty() {
            tmp = vec![0u8; w * h];
        } else {
            tmp.iter_mut().for_each(|t| *t = 0);
        }
        if !fill_envelope(raster, geom, subs, rule, cb, should_cancel, &mut tmp) {
            return None;
        }
        if !multiply_coverage(&mut acc, &tmp, should_cancel) {
            return None;
        }
    }

    let all_opaque = acc.iter().all(|&v| v == 255);
    Some(ClipMask {
        bounds: cb,
        data: acc,
        outside: 0,
        all_opaque,
    })
}

/// Rasterize one clip path's coverage over the rows of envelope `cb` into
/// `dst`, a `cb.width × cb.height` plane. `false` when cancelled.
fn fill_envelope(
    raster: &mut RasterKernel,
    geom: ClipGeom,
    subs: &[(usize, usize)],
    rule: FillRule,
    cb: DeviceRect,
    should_cancel: Option<&(dyn Fn() -> bool + Send + Sync)>,
    dst: &mut [u8],
) -> bool {
    let w = cb.width as usize;
    let h = cb.height as usize;
    let (bx, by) = (cb.x as usize, cb.y as usize);
    raster.fill_cancellable_in(
        geom.points,
        subs,
        geom.width,
        geom.height,
        rule,
        by..by + h,
        should_cancel,
        |y, x0, x1, cov| {
            if y < by || y >= by + h {
                return;
            }
            let ly = y - by;
            let lx0 = x0.max(bx);
            let lx1 = x1.min(bx + w - 1);
            if lx0 <= lx1 {
                dst[ly * w + (lx0 - bx)..=ly * w + (lx1 - bx)].copy_from_slice(&cov[lx0..=lx1]);
            }
        },
    )
}

/// Intersect `tmp`'s coverage into `acc` (`acc · tmp / 255`), polling for
/// cancellation every 64 Ki pixels. `false` when cancelled.
fn multiply_coverage(
    acc: &mut [u8],
    tmp: &[u8],
    should_cancel: Option<&(dyn Fn() -> bool + Send + Sync)>,
) -> bool {
    for (index, (a, &t)) in acc.iter_mut().zip(tmp).enumerate() {
        if index & 0xffff == 0 && should_cancel.is_some_and(|cancel| cancel()) {
            return false;
        }
        *a = mul_div_255(*a as u16, t as u16) as u8;
    }
    true
}
