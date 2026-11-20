# Design: strip rendering (banded execution)

Date: 2026-11-20. Status: phases 1 (Gray8 and RGB output in strips) and 2
(RGBA pages with large offscreens) are built and on by default; see "Phase 1
results" and "Phase 2 results". Before them, the same day: phase 0 (the
harness), the cull before lowering and the row-proportional Gray8
expansion. Phase 3 is proposed. Ledger:
`exprim.memory.strip-rendering` (this plan), `exprim.memory.remaining-levers`
(what it would close), `exprim.memory.peak-after-20261120` (the numbers below).

## What it is for

After the 2026-11-20 memory work, what a page holds at its peak is:

| Held | Size | Bounded today? |
|---|---|---|
| Page surface | 4 bytes per pixel, always RGBA | no |
| Output conversion | +1 (Gray8) or +3 (RGB8) bytes per pixel beside the surface | no |
| Group offscreens | 4 bytes per pixel of the group's bounds, per nesting level; three for a knockout group | no |
| Soft-mask rendering | 4 bytes per pixel of the mask bounds while it renders, then 1 | no |
| Tiling fill | 5 bytes per pixel of the fill bounds | no |
| Clip masks | 1 byte per pixel each | yes, eight planes |
| Lowered ops, decoded images | | yes, 16 MiB chunks |

Measured at 600 dpi, US Letter (135 MB surface): Gray8 output peaks at
174 MB for a 34 MB result; RGB8 through `RenderSession::render` at 241 MB for
a 101 MB result; a page with tiling fills, two groups and a soft mask at
309 MB. Every unbounded row is proportional to the page's pixel count.

Strip rendering bounds all of them by the strip instead of the page. The
page is rendered as horizontal strips of the device plane; each strip is a
complete lower-and-paint pass whose output window is that strip. Group,
soft-mask and tiling surfaces come out strip-sized because their bounds are
intersected with the window, and the strip is converted into the final
output buffer in the requested format, so an RGBA page surface never exists
unless RGBA is what was asked for.

Expected peaks for the same three cases, with a 32 MiB strip: about 66 MB
(Gray8), 133 MB (RGB8, exact-length buffer), and 135 MB plus strip-sized
offscreens (the transparency page). These are computed, not measured.

## The one constraint that shapes the design

Output must stay byte-identical. That rules out the obvious implementation,
rendering each strip as a region crop: a crop translates the page transform,
and floating-point transforms are not translation-invariant (a point at
5000.37 and the same point at 904.37 round differently in `f32`), so
coverage differs in the last bit. The tiling code already documents this for
cells.

So a strip is **not** a translated render. It is the same render, in the
same absolute device coordinates, with painting restricted to a row window:

- Lowering gets a window `[y0, y1)` and clamps every op's device bounds to it,
  where today it clamps to the output size (`device_bounds`, six call sites
  in `prepared.rs`). Geometry is flattened exactly as now; nothing moves.
- The strip surface is a `Surface` with `origin_y = y0`. Surfaces already
  carry an absolute origin, for group offscreens, and every paint path is
  already restricted to its op's bounds.
- Fills rasterize only the window's rows through the existing row-window
  entry (`fill_cancellable_in`), which is tested byte-identical to the full
  fill.

Strips are the outermost loop, and chunked lowering runs inside each strip
unchanged, so the lowered backlog stays at 16 MiB. The reverse nesting does
not work: ops streamed in painter order need the whole page as their
accumulation target.

## What a strip pass costs

Each op is lowered once per strip it intersects. Three things keep that small:

1. **Few strips.** Strip height is chosen from a byte target, not a row
   count: `rows = target / (4 * width)`, at least 256. With a 32 MiB target a
   600 dpi Letter page is four strips and a 300 dpi page is one or two.
2. **Culling before flattening.** A conservative device y-range per IR op
   (control-point box, inflated by stroke reach; run box for glyphs; unit
   square for images) is computed once per render and lets a strip skip ops
   that cannot touch it without flattening them.
3. **Images decode once.** Each strip finds the decode in the image cache.
   An image too large for the cache would be decoded per strip; the pass
   must pin the page's decodes for its duration, or strips must not be used
   for such a page.

Small ops (glyphs, small paths) are lowered once in total. An op that spans
the page is flattened once per strip; only its rows in the window are
rasterized.

## When to use strips

Strips are a memory tool and cost time, so they are used only where they pay:

- output is not RGBA and the surface exceeds the target (Gray8, RGB8), or
- the page's estimated offscreen bytes (groups, soft masks, tilings, scaled to
  the request) exceed the target.

Otherwise the page renders in one pass, as today. The choice is a pure
function of the request and the page, never of thread count or cache state,
and because strips are byte-identical to one pass the choice cannot change
output either way.

## Hazards to settle before any of this ships

Read from the code; each needs a test in phase 0.

1. **Tiling cells are lowered in coordinates local to the fill's bounds**
   (`shift = translate(-b.x, -b.y)` in `render_tiling_impl`). If the bounds
   shrink to the strip, the shift changes and cell geometry rounds
   differently. The fill must keep its full bounds as the coordinate origin
   and apply the window to rows only; cell lowering needs the same window
   parameter, in local coordinates.
2. **A mask that is all-opaque over a strip but not over the page** selects
   the unmasked fast path for draws in that strip. The code states the two
   paths are byte-identical where the mask is 255; strips make that claim
   load-bearing.
3. **Soft masks.** Pixels outside a mask's rendered bounds take its `outside`
   value. Bounds intersected with the window must not change which pixels
   inside the window count as outside.
4. **Image row loops** carry state from row to row (the area filter's row
   slots, parallel row bands). Their tests assert band independence; a strip
   starts mid-image, which those tests do not cover.
5. **Decisions taken from bounds.** Clip sharing keys on bounds, opaque-rect
   classification and empty-bounds culling read them. All must give the same
   pixels for the window as for the page.
6. **Per-page counters and diagnostics.** A degraded-draw note would be
   recorded once per strip, and `RenderStats` work counters count per-strip
   work. Notes must be deduplicated; the counters change meaning (as they did
   when tile culling landed).

## Phases

**Phase 0: the harness, no behaviour change.** Thread the row window through
lowering and execution behind a test-only entry. Add a property test in the
style of the existing residency tests: random clip, group, soft-mask, tiling
and image pages, random strip heights, bytes compared with one pass. Add
`PDF_RENDERER_STRIP_ROWS` so the page-hash gate can force strips on the
public fixtures and the local corpus. The outcome of this phase is the list
of hazards that actually fail, which decides whether the rest is worth doing.

**Phase 1: Gray8 and RGB8 output.** Strips write converted rows straight into
an exact-length output buffer. This removes the second plane from the
default `RenderSession::render` without changing the buffer's length, which
closes the API question the in-place call left open.

**Phase 2: offscreen-heavy RGBA pages.** Strips chosen from the estimate;
each strip is copied into the final buffer.

**Phase 3, optional.** Strips rendered in parallel for very large single
pages (what `CpuBackendOptions::tile_size` was reserved for); print composes
from strips instead of whole placed rasters.

Each phase is gated on unchanged page hashes with strips forced on.

## Phase 0 results (2026-11-20)

Built: a row window on the prepared page (`CpuPreparedPage::rows`, set
through `LowerSink::rows`), bounds of fills, strokes, images, glyph runs,
groups and soft masks clamped to it, and a strip loop in the backend behind
`PDF_RENDERER_STRIP_ROWS` that assembles the strips into the page surface.
With the knob unset nothing changes. It saves no memory yet.

**Identity holds.** A property test paints random clip, group (isolated,
knockout, wrapper) and soft-mask (alpha, luminosity, luminosity with
backdrop) pages in strips of 1 row, 7 rows, a random height and more than
the page, and compares bytes with one pass. The page-hash gate with strips
forced on: 576 configurations of the public fixtures at 97 rows per strip
and 228 configurations of 19 large local documents (JPEG 2000, JPEG, JBIG2
and CCITT scans, vector and text documents) at 211 rows, 0 differ. None of
the six hazards failed in this form, with two of them deliberately not
exercised:

- Tiling fills keep their full bounds, so their cells are lowered and painted
  exactly as before and only the final composite is limited to the strip's
  rows. Their offscreen is therefore not strip-sized yet (hazard 1 is
  avoided, not solved).
- Clip bounds are not clamped, so clip masks are still built over their full
  envelope in every strip.

**The cost is too high without culling before lowering.** Warm render
cycles, four strips against one pass:

| Page | Change |
|---|---|
| Text | +265% |
| Repeated form, 7,500 ops | +198% |
| Clip tiles | +111% |
| Round-joined contours | +63% |
| Small curves | +47% |
| Colour JPEG page scans (two documents) | +51%, +57% |
| Tiling, shading and groups | +9% |
| Three other local documents | −15% to +4% |

Every op is lowered once per strip today, so text and many-op pages
approach four times the lowering. Step 2 of "What a strip pass costs", a
conservative y-range per op checked before flattening, is therefore a
precondition for phase 1, not a refinement.

Correction, same day: this section first called the two scans JPEG 2000 and
put their cost down to the Gray8-to-RGB8 widening running once per strip.
Both are colour JPEG scans, which are never widened; their cost is the
harness's own memory traffic (see "Gray8 expansion"). The widening is real
for gray scans, which were not in this table.

## Cull before lowering (2026-11-20)

Built differently from step 2 above. A conservative y-range computed per op
kind would have to be proved to cover everything lowering can emit (stroke
reach, miter spikes, glyph overhang, image filters). Instead the range is
**recorded**: the first strip lowers every op anyway, and notes for each IR
op the device rows of everything it emitted, taken before the clamp to the
strip so that the range does not depend on which strip recorded it
(`OpExtents` in `prepared.rs`). Later strips skip a fill, stroke, glyph run,
image or shading whose recorded rows miss their window, before any
flattening, glyph lookup or image resolve. State ops (save, restore,
transforms, clips, groups, soft masks) always run. An op the cull skips is
one whose bounds would have been clamped to nothing, so identity holds by
construction. A lowering that stops early (cancellation) leaves the record
incomplete and nothing is skipped from it.

The harness also stopped paying for memory traffic that is its own and not
the design's: one strip buffer is reused for every strip of the same height,
and the page surface the strips are copied into is not painted with a
background first.

**Identity holds.** The property test runs every case with the cull off and
on. Page-hash gate with strips forced: public fixtures at 97 and at 13 rows
per strip, 576 configurations each, and the 19 large local documents at 211
rows, 228 configurations; 0 differ. Without strips, 0 differ.

**Cost, four strips against one pass** (warm render cycles, least of three
medians; the phase 0 build re-measured in the same run):

| Page | Phase 0 | With the cull |
|---|---|---|
| Text | +333% | +112% |
| Repeated form, 7,500 ops | +208% | +55% |
| Clip tiles | +128% | +37% |
| Round-joined contours | +67% | +33% |
| Small curves | +46% | +7% |
| Tiling, shading and groups | +17% | +7% |
| Page of rectangles | −4% | −26% |
| Colour JPEG page scans (two documents) | +60%, +62% | +39%, +28% |
| Three other local documents | −11% to 0% | −11% to +2% |

What is left, by cause:

- **Text** costs 8.5M cycles in one pass, nearly all of it painting the
  background of a 34 MB surface. The strips' extra 9.6M cycles match one
  more pass over those bytes, the copy of each strip into the page (removing
  the page's own background pass took the same amount off). Phase 1 replaces
  that copy with the format conversion one pass already pays for, so this
  row is read as the harness, not lowering; it was not profiled further.
- **Repeated form and clip tiles**: an op that straddles a strip boundary is
  lowered in both strips, and every clip is pushed in every strip. Clip
  bounds are still not clamped, so a mask with consumers in two strips is
  built over its full envelope twice. Clamping clip masks to the window is
  the next lever here.
- **Round-joined contours**: page-spanning paths are flattened and stroked
  once per strip; only rasterization is limited to the window's rows.
  Culling cannot help. Fewer, taller strips can.
- **Colour JPEG page scans**: one image per page, so the cull has nothing to
  skip; the gain over phase 0 is the harness change. See "Gray8 expansion"
  for what is left.

So the cull removes the cost that grew with the op count, which was the
precondition named in "Phase 0 results".

## Gray8 expansion (2026-11-20)

An opaque axis-aligned Gray8 image is expanded to RGB8 in `paint_image` so
that it qualifies for the RGB8 fast paths: three bytes per texel, for the
whole image, on every draw. In strips that is the whole image once per
strip.

It now expands only the source rows the draw's device rows can read
(`PreparedImage::source_rows_for`: the two end rows mapped through the
inverse transform, plus half the filter footprint and two rows of slack,
clamped to the image). The buffer keeps its full length, so no kernel and no
sampler changes; rows outside the band stay as allocated, zero pages that
are never touched. The alternative, a band-sized buffer with a row offset,
would have to reach every reader of `samples`, the generic sampler included.

One pass is unchanged: the band is the whole image. A test paints noise
images over every one-row window and random windows of each path a promoted
image can take (nearest, bilinear, box average, masked box average and its
generic border; upright and flipped) and compares with the whole draw; it
fails when the band is narrowed by three rows.

Measured on a local Gray8 page scan (one JPEG per page, three pages, four
strips against one pass). CPU is single-threaded cycles; wall is with all
threads:

| Scale | CPU before | CPU after | Wall before | Wall after |
|---|---|---|---|---|
| Minified, 0.5× | +45% | +5% | +161% | +52% |
| About 1:1 | +136% | +31% | +344% | +79% |
| Magnified, 2× | +68% | +37% | +115% | +71% |

Peak working set at 1:1 with strips: 158 MB to 139 MB at four strips,
152 MB to 127 MB at seventeen. Commit is unchanged, because the buffer is
still allocated at full length.

What is left on image pages is not image work. Wall time suffers more than
CPU because one pass paints in parallel row bands while the harness fills
each strip and copies it into the page serially: a second and third pass
over the page's bytes on one thread. Two colour JPEG scans, which are never
expanded, show the same shape (+150% to +170% wall at four strips). That is
the cost phase 1 is meant to remove by converting each strip straight into
the output; until that is built it is an expectation, and the copy was not
timed on its own.

Measurement note: the earlier tables in this document use process-wide cycle
counts with all threads, which include idle worker spinning and moved by up
to 40% between runs. They rank the cases correctly; the percentages for
image pages are not reliable. The table above uses the least per-page time
over repeats, summed over pages.

## Phase 1 results (2026-11-20)

Built as designed, with these choices where the design left them open:

- **When.** A page is rendered in strips when its output is converted from
  the surface (Gray8, or RGB through `render_rgb_with_options` and
  `render_packed_rgb_with_options`) and its RGBA surface would be more than
  twice the strip target. The target is 32 MiB
  (`CpuBackend::with_strip_target_bytes`, `SessionOptions::strip_bytes`,
  `PDF_RENDERER_STRIP_MB`; 0 is off). So a 300 dpi Letter or A4 page (32 to
  33 MiB) renders in one pass and a 600 dpi one in five strips. The number
  of strips is `ceil(surface / target)` and the rows are divided evenly, at
  least 256.
- **Output.** Each strip is converted into its rows of a buffer of the exact
  output length, allocated as zero pages so it grows as strips finish. RGB
  from `RenderSession::render` is produced this way too, which removes the
  session's own conversion from a finished RGBA page; a page small enough
  for one pass is converted from its surface as before.
- **One buffer per page.** Every strip is painted on a surface of the same
  height and the last uses only its top rows. A shorter last strip needed a
  second buffer; one was parked in the surface pool on every render and five
  strips' worth accumulated, which hid the whole saving in the first
  measurement.
- **Decodes across strips.** The design's third cost point. The shared image
  cache refuses an entry larger than its budget, can evict under concurrent
  pages and can be switched off, and then every strip an image crosses would
  decode it again. The render keeps a decode, or an image resolved to RGB8,
  from the strip that first draws it until the strips pass the draw's last
  row (`StripDecodes`), and only when the draw reaches below the current
  strip.

**Identity.** Page-hash gate, 0 differ: public fixtures (576 configurations
each) with strips forced at 97 and 13 rows, with the target set to 1 MiB so
that the automatic path runs at these sizes, and with defaults; the 19 local
documents (228 each) with strips forced at 211 rows, at 1 MiB, and with
defaults. A session test compares strips with one pass for Gray8, RGB and
in-place RGB on text, a gray scan and annotations, with and without an image
cache.

**Peak working set and wall time, one pass before and strips after**, 600
dpi US Letter (a 135 MB surface), public fixtures, Gray8 through the
backend:

| Page | Peak before | Peak after | Wall |
|---|---|---|---|
| Text | 175 MB | 67 MB | +2% |
| Repeated form, 7,500 ops | 229 MB | 122 MB | +7% |
| Clip tiles | 180 MB | 73 MB | +10% |
| Small curves | 186 MB | 80 MB | +1% |
| Tiling, shading, groups, soft mask | 309 MB | 94 MB | −9% |
| Page of rectangles | 174 MB | 67 MB | −7% |
| Round-joined contours | 184 MB | 88 MB | −17% |

RGB through `RenderSession::render`: text 242 MB to 134 MB (−21% wall), the
form page 248 MB to 143 MB (+6%), the transparency page 309 MB to 148 MB
(−3%). The design's computed peaks for text were 66 MB and 133 MB.

Local page scans at 150 dpi of oversized pages (146 to 154 MB surfaces),
two or three pages each, peak and wall:

| Scan | As Gray8 | As RGB |
|---|---|---|
| Gray JPEG | 390 MB to 152 MB, −9% | 528 MB to 375 MB, −37% |
| Colour JPEG | 261 MB to 139 MB, +7% | −20% (peak not comparable) |
| Colour JPEG | 327 MB to 158 MB, +8% | 522 MB to 353 MB, −11% |

The second document's RGB peak is the probe reading the whole file into
memory before the session opens it, in both builds.

Pages that stay in one pass (300 dpi Letter fixtures, and three local
documents with smaller pages) are unchanged in memory and within −1% and
+5% in time as Gray8. As RGB through the session they are 0% to 18% faster,
because the backend's conversion replaces the session's.

The time cost that phase 0 and the cull measured at four forced strips on a
300 dpi page does not appear here: the strips are five times taller in
pixels, so fewer shapes cross a boundary, and a strip fits the cache where a
page surface does not.

Not done in this phase, done after it: RGBA output (see "Phase 2
results"), and the CMYK-to-RGB8 conversion of a minified CMYK image, which
now covers only the source rows a draw reads, exactly as the Gray8
expansion does (the same row-window test covers both; no CMYK scan was at
hand to time it on).

### The scheduler's estimate

The scheduler charged every job the whole RGBA surface, which for a Gray8
page in strips is 169 MB on paper against 61 MB held. A backend now says
what it paints on (`RenderBackend::working_surface_bytes`, the whole surface
unless overridden), and `estimate_job_bytes_on` charges that plane beside
the output and caps the group term by it. The CPU backend answers from the
same function the render uses to choose its strips
(`CpuBackend::strip_rows_for`), on the size the render clamps the request
to, so the two cannot disagree; a unit test checks the answer against
`RenderStats::surface_bytes` of a render. Clip masks are not bounded by the
strip and are charged as before.

Soft masks and tiling fills were not charged at all. Content lowering now
records, per soft mask, the page-space bounds of the mask's own content (the
bounds it already computed for visible paint and threw away inside a mask),
and the largest shape filled with a tiling pattern. From those:

- `estimated_soft_mask_peak_bytes`: a walk of the op stream that mirrors
  the backend's mask stack. A mask holds four bytes per square unit while
  its content is rendered and one while it stays in force, until the save
  scope or group that set it ends; the peak of what is live together.
- `estimated_tiling_peak_bytes`: five bytes per square unit of the largest
  tiling fill (the plane its cells are painted on plus the shape's
  coverage).
- `estimated_peak_bytes` no longer counts a group the backend paints
  straight onto its parent (non-isolated, not knockout, Normal blend, fully
  opaque, no soft mask in force), which is the wrapper most producers put
  around a page or a form and which holds no surface.

`PageComplexity::offscreen_bytes` scales the group and mask figures by the
transform and caps them by the plane painted on (the strip, in strips);
`tiling_bytes` caps by the page, because a tiling fill keeps its whole
bounds in a strip. Mask bounds come from control points and generous glyph
boxes and ignore clips, so they are upper bounds; a shading inside a mask
counts as the whole page.

Measured through the scheduler, 32 pages of two local text documents at
600 dpi as Gray8, eight render workers, best of four runs, whole-surface
estimate against the strip one:

| Budget | Document 1 | Document 2 |
|---|---|---|
| 1024 MiB | 5.3 s, 5.1 s | 0.84 s, 1.25 s |
| 512 MiB | 7.3 s, 5.3 s | 1.39 s, 0.85 s |
| 256 MiB | 7.6 s, 6.8 s | 2.82 s, 1.23 s |

Document 2 finishes in either about 0.87 s or about 1.3 s from run to run
in both modes, with an unlimited budget too, so its 1024 MiB row is that
spread and not the estimate. The gain is at budgets that the whole-surface
charge filled with one or two jobs. Process peak stayed between 600 and
725 MiB in every configuration, most of it outside the job budget (finished
pages waiting for in-order emission, the caches); at 256 MiB it rose by 25
to 65 MiB, because the budget is now used by four jobs where one filled it
on paper.

## Phase 2 results (2026-11-20)

An RGBA page is its own surface, so strips cost it a strip buffer and a copy
of every strip into the page. They pay only when they shrink offscreens by
more than that. The rule: an RGBA page whose surface exceeds twice the strip
target is rendered in strips when `PageComplexity::offscreen_bytes` at page
size also exceeds twice the target (64 MiB by default). Tiling fills are
left out of that figure, since strips do not shrink them. The mechanism is
the phase 0 harness: the strip is copied into its rows of the page surface,
which is allocated unpainted.

The choice is made in `CpuBackend::strip_rows_for`, the same function the
scheduler's estimate reads, which for such a page now charges the page, one
strip, and offscreens capped by the strip.

Counting directly painted wrapper groups as offscreens would have sent most
large RGBA pages into strips for nothing (a strip buffer more, not less), so
the estimate had to stop charging them first.

Measured at 600 dpi US Letter, RGBA through the backend, peak working set
and wall time:

| Page | Before | After | Wall |
|---|---|---|---|
| Tiling, shading, groups, soft mask | 309 MB | 202 MB | +4% |
| Groups (three pages) | 258 MB | 206 MB | −6% |
| Text | 141 MB | 141 MB | 0% |
| Repeated form | 228 MB | 228 MB | −2% |

Identity: the page-hash gate with strips forced has covered RGBA strips
since phase 0; with the target at 1 MiB the automatic RGBA rule runs on the
fixtures too. A test renders a page with a half-opaque group and a soft
mask both ways and compares bytes, and checks that a page of the same size
with only a wrapper group stays in one pass.

What phase 2 does not bound: tiling fills (hazard 1 is still avoided, not
solved). Clip masks were bounded afterwards; see "Clip masks in strips".

## Clip masks in strips (2026-11-20)

Phase 0 left clip bounds alone, so every strip that drew under a clip built
that clip's mask over its whole envelope. A clip now carries two rectangles:
`bounds`, the envelope as before, and `mask_bounds`, the envelope cut to the
rows being painted, which is what its mask covers. Mask building already
rasterized in absolute device coordinates over just the rows it keeps (the
row-window fill), so the bytes are those a whole-envelope mask has there.

The envelope itself must stay whole. Draw bounds are cut by it before they
are cut to the strip, and the first strip records each op's rows from those
bounds for the other strips to cull by; a clamped envelope would have made
the first strip record its own rows for everything under a clip.

A mask that is fully opaque over a strip but not over the page now takes the
unmasked fast path in that strip (hazard 2). The strip property test, which
includes one-row strips over nested path clips, and the page-hash gates
agree with one pass.

Measured at 600 dpi US Letter as Gray8, one thread, strips before and after:

| Page | Peak | Wall |
|---|---|---|
| 500 rotated clipped forms | 122 MB to 81 MB | −11% |
| One large clip, 200 times | 78 MB to 67 MB | −6% |
| Synthetic clip mix | 85 MB to 66 MB | −10% |
| Clip tiles (80 small clips) | 72 MB, unchanged | −5% |

Pages without path clips moved by a few percent either way, which is the
noise of these runs. One-pass rendering is unaffected: there the two
rectangles are equal.

Not done: the mask budget (eight page-sized planes, at least 64 MiB) and the
scheduler's charge for clip masks are still sized by the page.

## Alternatives considered

- **Region crops per strip.** Not byte-identical; see above.
- **Banding inside each group only.** Bounds group offscreens but not the page
  surface, so Gray8 and RGB8 gain nothing; it replays a scope's ops per band
  inside the executor, which breaks the mask store's rule that each op runs
  once.
- **Surfaces in the output format** (gray or RGB kernels). No extra passes,
  but every span and image kernel exists once per format, and groups still
  need alpha.
- **Keeping the lowered page across strips.** Avoids re-lowering but brings
  back the whole-page geometry arena that chunked lowering removed.

## Decisions needed

1. Whether a time cost on the pages that qualify is acceptable, and how
   much. Measured in phase 1: within −17% and +10% on the 600 dpi fixtures.
2. The strip target and the threshold for using strips. Phase 1 took the
   proposed 32 MiB and engages at twice that; both are one constant and one
   option, and nothing else depends on them.
3. Whether `RenderStats` work counters may change meaning again. Phase 1
   let them: on a page rendered in strips they count each strip's work and
   `surface_bytes` is the strip.
4. Whether parallel strips (phase 3) are in scope.
