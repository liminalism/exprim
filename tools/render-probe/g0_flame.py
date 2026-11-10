"""Stage shares from a `flamegraph` (inferno) SVG.

    python g0_flame.py [--top N] [--csv OUT.csv] FILE.svg [FILE.svg ...]

Each frame in the SVG carries its sample offset and width (`fg:x`, `fg:w`) and
its row (`y`). A frame's self time is its width minus the widths of the frames
stacked directly on it.

Self samples are charged to a *stage*: the innermost frame on their stack that
names a renderer stage, so inlined `core::` helpers, unresolved system
addresses and allocator internals count toward the renderer function that
called them. Independently, each sample is tagged when anything on its stack
is the allocator, a libm rounding or trig routine, or memset/memcpy; those
tags cut across stages and answer what the stage timers cannot separate.
"""

from __future__ import annotations

import bisect
import csv
import html
import re
import sys
from collections import defaultdict

FRAME = re.compile(
    r'<g><title>(?P<title>.*?) \((?P<n>[\d,]+) samples?, [\d.]+%\)</title>'
    r'<rect[^>]*? y="(?P<y>[\d.]+)"[^>]*? fg:x="(?P<x>\d+)" fg:w="(?P<w>\d+)"')

# Innermost match on the stack wins; within one frame, the first pattern wins.
STAGES = [
    ("tiling: cull test", r"tile_reaches_mask|cell_paint_extent"),
    ("tiling: lower cell", r"prepared::lower_cell"),
    ("kernel: scanline sweep (AET)", r"fill_scanline_aet|convert_row_sparse|raster::accumulate|raster::deposit_partial"),
    ("kernel: compact deposit", r"raster::deposit_edge_row|raster::deposit_local"),
    ("kernel: edge build and compact sweep", r"RasterKernel::fill_inner|RasterKernel::fill_impl|raster::coverage_byte|raster::convert_coverage"),
    ("paint: spans and blends", r"exec::dispatch_runs|exec::dispatch_row|exec::classify_spans|render_cpu::kernels::|exec::paint_opaque_rect|exec::composite_"),
    ("glyphs: blit", r"exec::paint_glyph_run"),
    ("glyphs: cache-miss raster", r"prepared::rasterize_glyph_bitmap|prepared::flatten_outline_into"),
    ("glyphs: cache lookup and placement", r"prepared::try_cached_glyph_run|SharedGlyphCache|prepared::quantize_origin"),
    ("fonts: program resolve and parse", r"prepared::resolve_font_program|FontProgramCache|SharedFontProgramCache|FontProgramKey|exprim::font::"),
    ("glyphs: outline lowering", r"prepared::lower_glyph|prepared::append_glyph|prepared::emit_outline|prepared::append_outline"),
    ("lower: stroke expansion", r"render_cpu::stroke::|prepared::append_stroke_outline|prepared::lower_stroke|prepared::flatten_polylines"),
    ("lower: fill and clip flatten", r"prepared::flatten_into|prepared::lower_fill|prepared::push_clip|prepared::device_bounds|prepared::finish_tiled_fill|prepared::lower_tiled|prepared::enumerate_tiles"),
    ("lower: images and decode", r"prepared::lower_image|render_cpu::image::|exprim::image::|jp2lam|jbig2"),
    ("clip masks", r"render_cpu::mask::|exec::build_render_clip_mask|exec::prebuild_masks"),
    ("tiling: mask, composite, driver", r"exec::render_tiling"),
    ("images: paint", r"exec::paint_image"),
    ("shading", r"exec::shade_|prepared::prepare_shading"),
    ("executor: draw-command overhead", r"exec::paint_command|exec::run_ops|exec::execute"),
    ("lower: page loop and other", r"prepared::lower_impl|render_cpu::prepared::"),
    ("surface clear and output", r"render_cpu::surface::|Surface::|rgb_surface_from_rgba"),
    ("render: backend entry", r"CpuBackend::|render_cpu::"),
    ("compile: interpreter and parser", r"exprim::content::|exprim::syntax::|exprim::object::|exprim::document::|exprim::function::|exprim::color::|exprim::structure::|exprim::security::|exprim::source::"),
    ("session", r"exprim::reader::"),
    ("harness and process", r"lege_render_probe::|bench::|std::rt::|main$|BaseThreadInitThunk|RtlUserThreadStart|invoke_main|__scrt"),
]
STAGE_RE = [(name, re.compile(pattern)) for name, pattern in STAGES]
TAGS = [
    ("allocator", re.compile(
        r"__rust_alloc|__rust_dealloc|__rust_realloc|alloc::alloc::|RawVec|RtlAllocateHeap|RtlFreeHeap"
        r"|RtlReAllocateHeap|\bmalloc\b|\bfree\b|_free_base|_malloc_base|_realloc_base|HeapAlloc|HeapFree"
        r"|finish_grow|grow_one|do_reserve")),
    ("libm rounding (floor/ceil/round)", re.compile(r"^(floorf?|ceilf?|roundf?|truncf?)$")),
    ("libm trig and sqrt", re.compile(r"^(sinf?|cosf?|sincosf?|sqrtf?|powf?|hypotf?|atan2f?|_hypotf?)$")),
    ("memset/memcpy", re.compile(r"^(memset|memcpy|memmove)$|__memset|__memcpy|compiler_builtins::mem")),
]


def shorten(title: str) -> str:
    """`module`path::to::function<generics>(args)` -> `path::to::function`."""
    name = html.unescape(title)
    name = name.split("`", 1)[-1] if "`" in name else name
    depth = 0
    out = []
    for ch in name:
        if ch in "<(":
            depth += 1
        elif ch in ">)":
            depth = max(0, depth - 1)
        elif depth == 0:
            out.append(ch)
    return "".join(out).strip() or html.unescape(title)


def classify(stack):
    """`stack` runs leaf first; the stage of the innermost known frame."""
    for name in stack:
        for stage, pattern in STAGE_RE:
            if pattern.search(name):
                return stage
    return "unattributed"


def analyse(path):
    text = open(path, encoding="utf-8").read()
    rows = defaultdict(list)
    for m in FRAME.finditer(text):
        rows[float(m["y"])].append((int(m["x"]), int(m["w"]), shorten(m["title"])))
    if not rows:
        raise SystemExit(f"no frames in {path}")
    ys = sorted(rows)  # the root row has the largest y (not inverted)
    for y in ys:
        rows[y].sort()
    starts = {y: [x for x, _, _ in rows[y]] for y in ys}
    step = min((b - a for a, b in zip(ys, ys[1:])), default=16.0)

    stage_samples = defaultdict(int)
    tag_samples = defaultdict(int)
    leaf_samples = defaultdict(int)
    total = 0
    for index, y in enumerate(ys):
        adjacent = index > 0 and abs((y - ys[index - 1]) - step) < 1e-6
        above = rows[ys[index - 1]] if adjacent else []
        above_starts = starts[ys[index - 1]] if adjacent else []
        for x, w, name in rows[y]:
            at = bisect.bisect_left(above_starts, x)
            children = 0
            while at < len(above) and above[at][0] < x + w:
                children += above[at][1]
                at += 1
            own = w - children
            if own <= 0:
                continue
            total += own
            stack = [name]
            for below in ys[index + 1:]:
                at = bisect.bisect_right(starts[below], x) - 1
                if at < 0:
                    break
                bx, bw, bname = rows[below][at]
                if not bx <= x < bx + bw:
                    break
                stack.append(bname)
            leaf_samples[name] += own
            stage_samples[classify(stack)] += own
            for tag, pattern in TAGS:
                if any(pattern.search(frame) for frame in stack):
                    tag_samples[tag] += own
    return total, stage_samples, tag_samples, leaf_samples


def main(argv):
    top = int(argv[argv.index("--top") + 1]) if "--top" in argv else 0
    csv_out = argv[argv.index("--csv") + 1] if "--csv" in argv else None
    skip = {"--top", "--csv"}
    files = [a for i, a in enumerate(argv) if not a.startswith("--") and (i == 0 or argv[i - 1] not in skip)]
    table = []
    for path in files:
        total, stages, tags, leaves = analyse(path)
        label = path.replace("\\", "/").split("/")[-1]
        print(f"== {label}: {total} samples")
        print("   stages (innermost renderer frame on the stack):")
        for stage, n in sorted(stages.items(), key=lambda kv: -kv[1]):
            share = 100.0 * n / total
            if share >= 0.2:
                print(f"      {share:5.1f}%  {stage}")
            table.append({"file": label, "kind": "stage", "name": stage, "samples": n, "share_pct": round(share, 2)})
        print("   cross-cutting tags (anything on the stack):")
        for tag, n in sorted(tags.items(), key=lambda kv: -kv[1]):
            share = 100.0 * n / total
            print(f"      {share:5.1f}%  {tag}")
            table.append({"file": label, "kind": "tag", "name": tag, "samples": n, "share_pct": round(share, 2)})
        if top:
            print(f"   top {top} leaf frames (self time):")
            for name, n in sorted(leaves.items(), key=lambda kv: -kv[1])[:top]:
                print(f"      {100.0 * n / total:5.1f}%  {name[:110]}")
    if csv_out:
        with open(csv_out, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["file", "kind", "name", "samples", "share_pct"])
            writer.writeheader()
            writer.writerows(table)


if __name__ == "__main__":
    main(sys.argv[1:])
