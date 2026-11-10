# corpus/vector: synthetic vector stress fixtures

Synthetic, written from scratch by `probe synth` (the `lege-render-probe` harness,
`tools/render-probe/probe`), using exprim's
`test_support` PDF builder with hand-written content-stream operators. No third-party
content. Deterministic (fixed-seed xorshift generator): regenerating gives
byte-identical files. Redistributable under the repository's MIT licence.

Regenerate (from the repository root, with the probe built with `--features synth`):

    cargo build --profile debug-fast --features synth --manifest-path tools/render-probe/probe/Cargo.toml --target-dir <target>
    <target>/debug-fast/lege-render-probe synth corpus

All are single-page US Letter (612 x 792 pt). Op counts are from `pdfr dump <pdf> 0`.

| File | Exercises | Approx. ops (page IR) | Bytes |
|---|---|---|---|
| hairline-grid.pdf | 4,000 thin straight strokes (0.1-0.5 pt) on a 6 pt grid plus diagonals; butt caps, miter joins | 4,000 strokes, 4,000 line-width sets | 116,740 |
| round-contours.pdf | 600 smooth map-contour polylines of 20-60 cubics each, widths 0.5-4 pt, round caps and joins, 4 colours | 600 strokes (about 24,000 curve segments) | 885,155 |
| dashed-contours.pdf | 300 contour paths (20-50 cubics), 8 dash patterns with phases, butt/round/square caps, miter/round/bevel joins | 300 strokes, 300 dash sets | 394,612 |
| curve-fan.pdf | 6,000 small blobs (2-20 pt, 4-6 cubics, half with a same-direction inner subpath so `f` and `f*` differ) plus 20 large blobs (~100-400 pt) | 3,008 fills nonzero, 3,012 fills even-odd | 1,696,099 |
| clip-tiles.pdf | 8 x 10 cells, each `q`, circle clip (4 cubics, `W n`), 30 overlapping rect/blob fills, `Q` | 80 clips, 2,400 fills | 366,584 |
| clip-repeat.pdf | one circular clip (r = 250) re-established 200 times (`q path W n ... Q`), different fill under each | 200 clips, 200 fills | 55,654 |
| page-rects.pdf | page-sized fills (some overshooting the box), 60 nested stroked frames, table-like grid: row bands, cell shading, long thin rules as filled rects and stroked lines, checkboxes | 186 fills, 190 strokes | 10,542 |
| offpage-zoom.pdf | 3,000 strokes/fills: 300 in a 120 x 120 pt central region (246-366, 336-456), 2,700 more than 20 pt outside the page box on all sides (to +-3000 pt) | 1,509 strokes, 1,491 fills | 299,418 |
| form-repeat.pdf | one Form XObject emblem (~40 path ops, inherits colours) invoked 500 times with rotation/scale `cm` | 500 `Do`; 7,500 ops after inlining (1,000 fills, 1,500 strokes, 500 clips) | 44,709 |
| pattern-shade-alpha.pdf | 3 coloured tiling patterns (one rotated by `/Matrix`), `sh` axial and radial, axial shading pattern fill, tiling-pattern stroke, `/ca` `/CA` 0.5, two transparency-group Form XObject invocations with `/ca` 0.6, one luminosity soft mask | 63 IR ops; 3 patterns, 3 `sh`, 2 groups, 1 soft mask | 3,666 |

Sizes: five files exceed the 200 KB target. Content streams are uncompressed (the
builder has no Flate encoder) and each Bezier segment is six numbers; the sizes are the
price of the requested path counts.
