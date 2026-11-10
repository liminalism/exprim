"""Collate the G0 matrix raw JSONL into CSV tables.

    python g0_tables.py [RAW_DIR] [TABLES_DIR]

Page rows are paired with the meta row of the same invocation (the most recent
meta row above them in the file).
"""
import csv
import glob
import json
import os
import statistics
import sys
from collections import defaultdict

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "docs", "benchmarks", "2026-11-10-attribution")
RAW = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, "raw")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(BASE, "tables")
os.makedirs(OUT, exist_ok=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g0_summary  # noqa: E402


def name(path):
    return os.path.basename(path)[:-len(".jsonl")]


def load_rows(path):
    """Yield (meta, page_row) pairs."""
    meta = None
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("kind") == "meta":
                meta = row
            elif row.get("kind") == "page" and meta is not None:
                yield meta, row


def cyc_ms(meta, row):
    c = meta.get("cycles_per_ns")
    return row["render_cycles"] / c / 1e6 if c else float("nan")


files = sorted(glob.glob(os.path.join(RAW, "*.jsonl")))

# a. summary.csv
summary = []
for f in files:
    for key, items in g0_summary.load(f).items():
        _, doc, api, scenario, dpi, detail = key
        warm = [(m, r) for m, r in items if r["phase"] == "warm"] or items
        cm = [cyc_ms(m, r) for m, r in warm]
        wall = [r["render_wall_ns"] / 1e6 for _, r in warm]
        first = [r for _, r in items if r["phase"] == "first"]
        med = statistics.median(cm)
        summary.append({
            "file": os.path.basename(f), "id": doc, "api": api, "scenario": scenario, "dpi": dpi,
            "detail": detail, "n": len(warm), "pages": len({r["page"] for _, r in items}),
            "cyc_min_ms": min(cm), "cyc_med_ms": med, "cyc_p90_ms": g0_summary.quantile(cm, 0.9),
            "cyc_max_ms": max(cm), "wall_med_ms": statistics.median(wall),
            "contention": statistics.median(wall) / med if med else float("nan"),
            "first_compile_ms": first[0]["compile_wall_ns"] / 1e6 if first else float("nan"),
            "first_render_ms": first[0]["render_wall_ns"] / 1e6 if first else float("nan"),
            "hashes": len({r.get("hash", "-") for _, r in items}),
        })
cols = "file,id,api,scenario,dpi,detail,n,pages,cyc_min_ms,cyc_med_ms,cyc_p90_ms,cyc_max_ms,wall_med_ms,contention,first_compile_ms,first_render_ms,hashes".split(",")


def write_csv(path, cols, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})


write_csv(os.path.join(OUT, "summary.csv"), cols, summary)

# b. pages.csv  (per file,id,page,api,scenario,dpi)
pg = defaultdict(list)
for f in files:
    for m, r in load_rows(f):
        pg[(os.path.basename(f), r["id"], r["page"], r["api"], r["scenario"], r["dpi"])].append((m, r))
pages = []
for (f, doc, page, api, sc, dpi), items in pg.items():
    warm = [(m, r) for m, r in items if r["phase"] == "warm"] or items
    cm = [cyc_ms(m, r) for m, r in warm]
    first = [r for _, r in items if r["phase"] == "first"]
    hs = sorted({r.get("hash", "-") for _, r in items})
    pages.append({
        "file": f, "id": doc, "page": page, "api": api, "scenario": sc, "dpi": dpi,
        "w": items[0][1]["w"], "h": items[0][1]["h"], "warm_n": len(warm),
        "cyc_min_ms": min(cm), "cyc_med_ms": statistics.median(cm),
        "wall_med_ms": statistics.median([r["render_wall_ns"] / 1e6 for _, r in warm]),
        "first_compile_ms": first[0]["compile_wall_ns"] / 1e6 if first else "",
        "first_render_ms": first[0]["render_wall_ns"] / 1e6 if first else "",
        "hash": hs[0] if len(hs) == 1 else "|".join(hs), "hash_differs": int(len(hs) > 1),
    })
write_csv(os.path.join(OUT, "pages.csv"),
          "file,id,page,api,scenario,dpi,w,h,warm_n,cyc_min_ms,cyc_med_ms,wall_med_ms,first_compile_ms,first_render_ms,hash,hash_differs".split(","),
          pages)

# c. stage_shares.csv
TIMES = """render.lower render.execute render.surface render.output render.path render.tiling render.tiling.lower_cell
render.tiling.cells render.path.glyph_runs render.image render.exec.prebuild_masks render.clip.mask_lazy session.rgb_pack
lower.x.fill_flatten lower.x.fill_finish lower.x.stroke_flatten lower.x.stroke_expand lower.x.stroke_finish lower.x.glyph
lower.x.image lower.x.clip lower.x.state lower.x.census""".split()
COUNTS = """render.commands render.edges render.covered_pixels render.lower.points render.tiling.fills render.tiling.tiles
render.tiling.surface_px render.tiling.mask_px render.clip.masks_lazy render.clip.masks_prebuilt render.clip.mask_bytes
render.transparency_groups render.soft_masks lower.in.fill_ops lower.in.stroke_ops lower.fill.curves lower.fill.lines
lower.fill.points lower.fill.curve_segments_adaptive lower.stroke.curves lower.stroke.out_points lower.stroke.quads
lower.stroke.disks lower.stroke.tris lower.stroke.vertices_interior lower.stroke.vertices_smooth lower.stroke.dash_pieces
lower.cull.fill_clipped lower.cull.fill_clipped_points lower.cull.stroke_clipped lower.cull.stroke_clipped_points
raster.path.fills raster.path.edges raster.path.route_accum raster.path.route_aet_sparse raster.path.route_aet_oversize
raster.path.aet_rows_dense_many raster.path.aet_rows_sparse raster.path.rows_emitted render.path.rows_outside_bounds
lower.glyph.cache_hits lower.glyph.cache_misses""".split()


def fastest_per_page(path):
    best = {}
    for m, r in load_rows(path):
        if r["phase"] != "warm":
            continue
        k = (r["id"], r["page"])
        if k not in best or r["render_cycles"] < best[k]["render_cycles"]:
            best[k] = r
    return best


stage_rows = []
for fn in ["prof-stage-low-144", "prof-stage-low-72", "prof-stage-session-144", "prof-stage-zoom-low-144"]:
    p = os.path.join(RAW, fn + ".jsonl")
    if not os.path.exists(p):
        continue
    for (doc, page), r in fastest_per_page(p).items():
        row = {"file": fn, "id": doc, "page": page, "total_ms": r.get("render.total", 0) / 1e6,
               "render_cycle_ms": ""}
        for k in TIMES:
            row[k] = r.get(k, 0) / 1e6
        for k in COUNTS:
            row[k] = int(r.get(k, 0))
        stage_rows.append(row)
write_csv(os.path.join(OUT, "stage_shares.csv"), ["file", "id", "page", "total_ms"] + TIMES + COUNTS, stage_rows)

# d. kernel_phases.csv
kern = []
p = os.path.join(RAW, "prof-fill-low-144.jsonl")
if os.path.exists(p):
    best = fastest_per_page(p)
    rkeys = sorted({k for r in best.values() for k in r if k.startswith("raster.")})
    for (doc, page), r in best.items():
        row = {"id": doc, "page": page, "total_ms": r.get("render.total", 0) / 1e6,
               "render.path": r.get("render.path", 0) / 1e6, "render.path.in_tiling": r.get("render.path.in_tiling", 0) / 1e6}
        for k in rkeys:
            v = r.get(k, 0)
            row[k] = v / 1e6 if g0_summary.is_duration(k) else v
        kern.append(row)
    write_csv(os.path.join(OUT, "kernel_phases.csv"),
              ["id", "page", "total_ms", "render.path", "render.path.in_tiling"] + rkeys, kern)

# e. hash_check.txt
PAIRS = [
    ("low-rerender-144", ["prof-stage-low-144", "prof-fill-low-144"]),
    ("low-rerender-72", ["prof-stage-low-72"]),
    ("session-rerender-144", ["prof-stage-session-144"]),
    ("zoom-low-144", ["prof-stage-zoom-low-144"]),
]


def hashmap(fn):
    p = os.path.join(RAW, fn + ".jsonl")
    d = {}
    if os.path.exists(p):
        for _, r in load_rows(p):
            d.setdefault((r["id"], r["page"], r["dpi"], r["api"], r["scenario"]), set()).add(r.get("hash"))
    return d


compared = 0
mism = []
missing = []
for base, others in PAIRS:
    hb = hashmap(base)
    for o in others:
        ho = hashmap(o)
        for k, v in ho.items():
            if k not in hb:
                missing.append((base, o, k))
                continue
            compared += 1
            if hb[k] != v or len(v) != 1:
                mism.append((base, o, k, sorted(hb[k]), sorted(v)))
with open(os.path.join(OUT, "hash_check.txt"), "w", encoding="utf-8") as fh:
    fh.write(f"compared: {compared}\nmismatched: {len(mism)}\nmissing in baseline: {len(missing)}\n\n")
    for m in mism:
        fh.write(f"MISMATCH {m[0]} vs {m[1]} {m[2]} {m[3]} vs {m[4]}\n")
    for m in missing:
        fh.write(f"MISSING {m[0]} vs {m[1]} {m[2]}\n")
print(f"summary rows {len(summary)}, pages {len(pages)}, stage {len(stage_rows)}, kernel {len(kern)}, hash compared {compared} mismatched {len(mism)}")
