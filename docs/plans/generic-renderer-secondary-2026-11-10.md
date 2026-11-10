# Secondary plan: broader CPU vector and text rendering

Date: 2026-11-10. Status: G0 done and the exact part of G1 done on 2026-11-10
(branch `perf/generic-renderer-g0-g1`); G1's geometry-policy changes, G2, G3
and G4 are open. Findings:
[attribution report](../benchmarks/2026-11-10-attribution/README.md); changes
and measurements:
[performance-history entry](../../dev-notes/render-refinement/performance-history/optimization-dense-vector-20261110.md).

Make exprim useful for general PDF rendering while retaining its strength on
Lege's scanned and MRC documents. This is a secondary workstream: the scan/image
pipeline remains a primary product requirement. The first target is dense-vector
latency, followed by text throughput and general-document correctness. Keep this
work inside the standalone CPU crate; it does not require Lege or a GPU backend.

## Evidence and limits

The [four-renderer benchmark](../benchmarks/2026-11-09/README.md) establishes a
vector performance gap, but does not attribute it to parsing. Compilation,
geometry lowering, coverage rasterization, clipping, compositing, and scheduling
are distinct candidates. The born-digital text result is competitive with PDFium
and hayro in several configurations; MuPDF and parallel scaling are the clearer
targets. These distinctions determine the order of work.

Raw whole-document median throughput at 144 DPI, in pages/second; higher is better:

| Document | Workers | exprim | PDFium | MuPDF | hayro |
|---|---:|---:|---:|---:|---:|
| MRC, 632 pages | 1 | 28.79 | 13.09 | 4.64 | 16.98 |
| Vector tables, 911 pages | 1 | 156.80 | 204.95 | 129.49 | 217.90 |
| Vector tables | 8 | 718.68 | 825.43 | 556.79 | 866.46 |
| Dense vector map, 1 page | 1 | 2.92 | 10.16 | 5.65 | 20.51 |
| Born-digital Latin text, 786 pages | 1 | 463.62 | 243.58 | 542.49 | 290.96 |
| Born-digital Latin text | 8 | 1,491.31 | 1,146.42 | 2,559.36 | 1,057.74 |

Source: [medians.csv](../benchmarks/2026-11-09/medians.csv), `kind=raw`,
`scale=2`. The map takes about 343 ms in exprim, versus 98 ms in PDFium and
49 ms in hayro. Eight page workers do not improve its latency: it is one page.
At 72 DPI, text scales only 2.25 times from one to eight workers; at 144 DPI,
3.22 times. Those are useful scheduling investigations, not proof of a parser
bottleneck. The eight-worker exprim configuration also splits two compilation
workers and six render workers, so it is not eight independent page renderers.

These are speed measurements, not proof of equivalent output. Only 11 of the
786 text pages pass the strict four-engine common-page gate at 144 DPI. Reviewed
examples show font-weight differences, but that does not resolve every page.
Retain the raw timing and unresolved fidelity findings together; do not present
the accepted text subset as representative throughput or agreement as ground
truth. Exprim's partial PDFium ancestry also limits independence of agreement.

The timing adapter constructs fresh compilation state per page. The public
`RenderSession` retains its compiler and backend, so the benchmark is not a
measurement of every public-API cache benefit. Measure both paths before changing
cache policy. Historical ledger performance entries have stale pre-extraction
paths and are leads, not a current diagnosis.

## Existing work to retain

Do not reopen the closed [July text plan](../../dev-notes/render-refinement/plans/PLAN-TEXT-GAP.md)
as though its missing mechanisms were still absent. Reconcile proposed work with
the [November scoping review](../../dev-notes/render-refinement/plans/optimization-scoping-20261108.md)
and current code before each implementation milestone.

Current building blocks include:

- [Session reuse](../../src/reader/session.rs), shared font-program and rendered
  glyph-coverage caches in [prepared.rs](../../src/render_cpu/prepared.rs), and
  cache keys that account for transform and subpixel phase. The bitmap fast path
  currently restricts eligibility to modest, axis-aligned glyphs.
- A bounding-box-local coverage accumulator and an active-edge fallback in
  [raster.rs](../../src/render_cpu/raster.rs), with run-aware output. Introducing
  an active-edge table is therefore not a new optimization.
- [Stroke expansion](../../src/render_cpu/stroke.rs) into fill geometry, including
  caps, joins, dashes, and transformed pens. General path flattening still has
  fixed 16-segment cubic loops; glyph outline flattening already has a separate
  flatness policy. Investigate these paths separately.
- [Stage timers and counters](../../src/render_cpu/stats.rs) for lowering,
  preparation, rasterization, output conversion, edges, rows, and surfaces, plus
  feature-gated [profiling](../../src/profiling/mod.rs). Extend missing attribution
  rather than build another competing measurement system.

## G0 — establish attribution and a representative corpus

**Deliverable:** a reproducible profile report and ranked hotspot list, with
source/build identifiers, commands, raw measurements, and correctness baselines.
G0 is required before choosing a production optimization.

1. Keep the current map, vector tables, and Latin book as regression anchors.
   Add several independent maps/CAD drawings, charts, and diagrams, covering
   curves, dense thin strokes, dashes, clipping, repeated forms, patterns,
   shadings, and transparency. Add text cases for TrueType, CFF/Type1, Type3,
   embedded and substituted fonts, CJK/vertical text, rotated or sheared text,
   text clipping, and text rendering modes. These are coverage requirements,
   not assertions that current implementations fail them.
2. Use hash-pinned local documents and small redistributable regression fixtures.
   Record licenses before adding PDFs to the repository. Include normal page
   sizes and fixed viewport/crop requests, alongside the existing declared-size
   renders. Large JPEG page boxes must not dominate a general-renderer score.
3. Measure cold process/open/first paint, sequential pages, repeated pages,
   zoom/crop, and compiled-page rerender separately through `RenderSession`.
   Keep the existing low-level adapter as a separately labelled comparison.
   Test 72 and 144 DPI, non-integral scale, and selected 300-DPI cases under a
   documented pixel limit. Separate CPU rendering from encoding and slow sinks.
4. Attribute document open, content parsing/compilation, lowering, allocation,
   coverage, clipping/blending, and output transfer. Record per-page p50/p90/p99,
   first-page latency, cache hit/miss and retained bytes, allocation volume,
   worker utilization, queue waits, and lock contention. Correlate named timers
   with a sampled CPU/allocation profile; do not infer parser cost from page rate.
5. For vectors, count input operators, flattened edges, stroke/dash/join
   amplification, clip-mask work, accumulator versus active-edge routing, and
   covered area. For text, separate font parsing, outline generation, cache
   lookup, coverage misses, and cached blits. Use a profiling build to locate
   work and ordinary release builds to accept speed changes.

**Exit:** explain the dominant cost on the dense map, one other vector document,
and representative cold/warm text pages. Estimate the maximum end-to-end benefit
from each candidate's measured share. If parsing is small, leave lexer/parser
micro-optimizations below geometry and raster work.

## G1 — reduce dense-vector work per page

**Priority:** first implementation milestone after G0. A page scheduler alone
cannot fix the measured map deficit.

Start with changes that preserve the raster result:

- Remove repeated transforms, bounds calculations, temporary allocations, and
  redundant clip construction where profiles show them. Reuse scratch arenas
  without retaining pathological peak allocations indefinitely. Cull only with
  conservative bounds that include strokes, antialiasing, and filter extents.
- Reuse repeated form/path/clip preparation where worthwhile. Keys must include
  geometry identity, device transform, fill rule, stroke style and dash phase,
  and relevant rendering policy; clip reuse must account for its state. Charge
  retained memory and avoid caching cheap one-use geometry.
- Investigate stroke expansion and fixed-segment general-path flattening as
  potential edge amplifiers. First improve allocation and traversal while
  preserving edge order. Measure the existing local accumulator and active-edge
  fallback before changing thresholds or span processing. Avoid returning to an
  all-edges-per-row scan.

Then, only if the profile warrants output-changing geometry work, evaluate
device-space error-bounded curve subdivision and less redundant stroke outlines.
Use explicit tolerance and subdivision limits. Preserve cusps, extrema, winding,
dash phase, miter limits, caps/joins, anisotropic pens, hairlines, and clipping.
The existing stroke expansion is the reference, not an absent feature to add.
Do not silently change the separate glyph flatness/hinting policy with a path
optimization.

Spatial indexing or band/tile execution is a later option if coverage or repeated
clip work still dominates. Preserve paint order, group/soft-mask dependencies,
and seam coverage; avoid scanning every command for every tile. Consider in-page
parallelism only after reducing unnecessary serial work and establishing a shared
thread budget. Keep an explicit reference path for differential checks.

**Deliverable:** stage attribution before/after, edge/allocation counts, output
comparison, latency across the expanded vector set, and cache-memory bounds.

## G2 — improve text reuse and throughput where it remains deficient

**Dependency:** G0; may proceed independently of G1 once text attribution exists.

- First establish whether the persistent public session closes part of the gap.
  Reuse compilation/font state appropriately, and inspect cache contention,
  font identity cost, evictions, and miss paths before changing cache sizes.
  Do not add another glyph cache on top of the existing one.
- If uncached transformed glyphs matter, extend eligibility conservatively to
  rotated/sheared text. Include every state affecting coverage: full linear
  transform, subpixel translation phase, hinting and antialiasing policy, and
  font/glyph identity. Preserve Type3 and text-clip behavior and bound oversized
  glyphs. Reusing an outline may be preferable to caching many bitmap variants.
- If cached text is blend-bound, reduce per-glyph dispatch, repeated state
  setup, and surface/output traffic. Batch only compatible paint, clip, blend,
  and transparency state while preserving positions and paint order. PDF text
  is already positioned: introducing text reshaping is not a speed fix.
- Investigate font parsing or content tokenization only where measured. Optimize
  repeated decoding or allocation without weakening malformed-input handling.
  Retain complete font identity across document boundaries and close/reopen.

Treat font weight, hinting, clipping, substitution, and missing glyphs as separate
fidelity questions. Diagnose the benchmark's text disagreement with explicit
font/raster settings and source-aware fixtures. Do not darken output simply to
match a majority engine or loosen the comparison gate to obtain a better score.

**Deliverable:** cold/warm text latency and miss costs, serial and multi-worker
throughput, font coverage results, and a disposition for unresolved differences.

## G3 — balance general-document scheduling and resource lifetimes

**Dependency:** G0 utilization data; integrate with G1/G2 rather than masking their
serial costs with additional threads.

Compare 1/2/4/8 total worker budgets, the current compile/render split, and a fused
page-worker or adaptive split experiment. Count all workers and prevent nested
oversubscription. Check whether compile workers, render workers, cache locks, or
memory bandwidth limit text throughput. Evaluate persistent workers only if
repeated-call startup or state loss is material.

Preserve ordered output and bounded admission. Trace ownership of memory permits
through completed surfaces, reorder queues, callbacks, and consumer-held output;
verify byte limits with a deliberately slow consumer. Test cancellation, errors,
and session disposal. A queue bounded by page count can still retain too many
bytes when page sizes vary. Any optional unordered API must be explicit.

Two observed memory problems are guardrails for this work: the legacy-path JBIG2
run grows with document prefix length at 72 DPI with one worker (about 204 MiB
at 32 pages and 2,602 MiB at 440), and parallel large-JPEG PNG export reached
about 11.66 GiB job commit.
Neither establishes a specific leak or allocation cause. Reproduce through both
public and legacy paths; separate renderer caches, input ownership, output
retention, and encoding. Resolve these with the primary reliability workstream,
and require all new caches to have bounded, testable lifetimes.

**Deliverable:** utilization/scaling curves, first-page and tail latency, memory
ownership evidence, and a justified default worker policy.

## G4 — verify general PDF behavior and accept the improvements

Build focused semantic fixtures as the new corpus exposes gaps: fill rules and
self-intersections, stroke/clip interactions, forms and patterns, shadings,
transparency groups, soft masks, text rendering modes, and font fallback. Track
unsupported operations and degraded draws explicitly. General usability includes
rendering the content, not just finishing quickly. This is a coverage audit,
not a claim that all these features are currently deficient.

Use PDFium, MuPDF, and hayro as differential references with versions and settings
pinned, and independently constructed expected results where practical. Separate
missing/misplaced content, geometry errors, and contrast/antialias differences.
Review difficult examples at useful zoom, retain unresolved cases, and keep
correctness evidence alongside all performance claims.

Proposed acceptance gates, to ratify after G0 establishes variability:

| Area | Gate |
|---|---|
| Dense vectors | First milestone: halve the current map's approximately 343 ms at 144 DPI, with improvement across the expanded set. PDFium's approximately 98 ms is a stretch reference, not a promised result. |
| Text | Improve the measured dominant cost and end-to-end result beyond run noise. Aim for at least 4-times scaling at eight total workers at both baseline DPIs; document the measured limiting resource if that target is infeasible. Preserve serial performance. |
| Scan/MRC protection | Investigate any repeatable regression over 3% on representative MRC/JPEG and other scan classes; do not accept one over 5% without an explicit product tradeoff. Include memory, first-page latency, and tails. |
| Exact optimizations | Identical output hashes across the regression subset, integer/non-integral scales, relevant formats, and fresh/shared backends; preserve cancellation and error behavior. |
| Geometry/raster policy changes | Separate acceptance from exact changes. Require bounded device-space error, semantic fixtures and visual review; no missing content, clip leaks, or tile seams. Record intentionally changed pixels and policy. |
| Memory | New caches and queues obey documented byte budgets, survive eviction/revisit, and release on close. Long runs plateau according to those budgets rather than accumulating per page. |

Use paired, order-swapped local before/after runs (nine rounds for small targeted
cases), recording spread and hardware/build settings. At milestones, repeat the
four-engine protocol with three repetitions and the expanded corpus; retain raw
timings, memory, checksums, fidelity findings, and profiler artifacts separately.
Do not run the full corpus after every mechanical edit. Preserve portable release
defaults; blanket ISA flags, approximate math, and previously losing SIMD paths
require new workload evidence and their own correctness gates.

## Execution order and scope boundary

1. **G0:** measurement/corpus change only; publish findings and select hotspots.
2. **G1:** exact vector reductions, then independently reviewed geometry changes
   if necessary. Reassess the map and broader vector gap after each batch.
3. **G2:** text changes selected from measured miss/blend/compile costs; retain
   the already competitive serial cases as controls.
4. **G3:** select a worker policy using fixed-budget measurements, and close
   resource-lifetime issues found in the public path.
5. **G4:** consolidate feature coverage, unresolved fidelity cases, regression
   results, and public API examples into a general-renderer readiness report.

Keep each implementation batch independently reviewable and reversible. Stop
optimizing a stage once its remaining share cannot justify the complexity; move
to the next measured bottleneck. This plan does not make every general-PDF feature
a prerequisite for publishing the existing scan-oriented crate, and it does not
replace the primary codec/MRC work. Publication claims should describe the
capabilities and limitations actually verified at that release.
