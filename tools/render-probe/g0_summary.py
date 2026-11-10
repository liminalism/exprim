"""Summarise `probe bench` JSONL files.

    python g0_summary.py [--profile] [--csv OUT.csv] FILE.jsonl [FILE.jsonl ...]

For every (file, id, api, scenario, dpi, detail) group this prints the warm
render cost as *cycle time* (thread cycles divided by the run's calibrated
cycles-per-nanosecond): the minimum, median, p90 and maximum, the wall-time
median, and the contention ratio (wall / cycle time; about 1.0 when the thread
was not preempted). Cycle time is the primary metric because the machine these
runs were taken on was busy; note that it still rises when the core itself
slows down (shared SMT sibling, frequency drop), so the minimum over many
iterations is the most stable estimator and the median shows the conditions.

With `--profile` it also prints, for the fastest warm iteration of each group,
every named stage timer (ms and share of render.total) and every work counter.
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import defaultdict

META_KEYS = {
    "kind", "id", "api", "scenario", "dpi", "scale", "page", "iter", "phase", "w", "h",
    "page_w", "page_h", "compile_wall_ns", "compile_cycles", "compile_cpu_100ns",
    "render_wall_ns", "render_cycles", "render_thread_cycles", "render_cpu_100ns", "output_bytes", "hash",
    "profile_schema", "live_bytes", "peak_live_bytes",
}

# Profile keys that are durations (nanoseconds) rather than counters.
DURATION_PREFIXES = (
    "compile.", "document.open", "lower.x.", "lower.font.parse", "lower.glyph.outline_extract",
    "lower.glyph.emit", "codec.", "session.",
)
DURATION_EXACT = {
    "render.lower", "render.surface", "render.execute", "render.output", "render.total",
    "render.lower.total", "render.prepare", "render.path", "render.image", "render.tiling",
    "render.path.opaque_rects", "render.path.solid_paths", "render.path.direct",
    "render.path.masked", "render.path.modulated", "render.path.in_tiling",
    "render.path.glyph_runs", "render.exec.prebuild_masks", "render.exec.run_ops",
    "render.clip.mask_lazy", "render.tiling.lower_cell", "render.tiling.cells",
    "render.tiling.fmask", "render.tiling.composite", "render.image.indexed_promotion",
}
COUNTER_SUFFIXES = (".count", "_count", ".parses", ".hits", ".misses")


def is_duration(key: str) -> bool:
    if ".ns_" in key:
        return True
    if key in DURATION_EXACT:
        return True
    if key.endswith(COUNTER_SUFFIXES):
        return False
    return key.startswith(DURATION_PREFIXES) and not key.startswith("compile.") or (
        key.startswith("compile.") and not key.endswith(("ops", "operations", "bytes", "objects"))
    )


def quantile(values, q):
    values = sorted(values)
    if not values:
        return float("nan")
    index = min(len(values) - 1, max(0, round(q * (len(values) - 1))))
    return values[index]


def load(path):
    meta = None
    groups = defaultdict(list)
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("kind") == "meta":
                meta = row
            elif row.get("kind") == "page" and meta is not None:
                key = (path, row["id"], row["api"], row["scenario"], row["dpi"],
                       meta.get("detail", "-") if meta.get("profiling") else "off")
                groups[key].append((meta, row))
    return groups


def main(argv):
    show_profile = "--profile" in argv
    csv_out = None
    if "--csv" in argv:
        csv_out = argv[argv.index("--csv") + 1]
    files = [a for i, a in enumerate(argv) if not a.startswith("--") and (i == 0 or argv[i - 1] != "--csv")]
    table = []
    for path in files:
        for key, items in load(path).items():
            _, doc, api, scenario, dpi, detail = key
            warm = [(m, r) for m, r in items if r["phase"] == "warm"] or items
            cyc_ms = [r["render_cycles"] / m["cycles_per_ns"] / 1e6 for m, r in warm if m["cycles_per_ns"]]
            wall_ms = [r["render_wall_ns"] / 1e6 for _, r in warm]
            first = [r for _, r in items if r["phase"] == "first"]
            hashes = sorted({r.get("hash", "-") for _, r in items})
            best_meta, best = min(warm, key=lambda mr: mr[1]["render_cycles"] or mr[1]["render_wall_ns"])
            record = {
                "file": path.replace("\\", "/").split("/")[-1], "id": doc, "api": api,
                "scenario": scenario, "dpi": dpi, "detail": detail, "n": len(warm),
                "pages": len({r["page"] for _, r in items}),
                "cyc_min_ms": min(cyc_ms) if cyc_ms else float("nan"),
                "cyc_med_ms": statistics.median(cyc_ms) if cyc_ms else float("nan"),
                "cyc_p90_ms": quantile(cyc_ms, 0.9),
                "cyc_max_ms": max(cyc_ms) if cyc_ms else float("nan"),
                "wall_med_ms": statistics.median(wall_ms),
                "contention": statistics.median(wall_ms) / statistics.median(cyc_ms) if cyc_ms else float("nan"),
                "first_compile_ms": first[0]["compile_wall_ns"] / 1e6 if first else float("nan"),
                "first_render_ms": first[0]["render_wall_ns"] / 1e6 if first else float("nan"),
                "hashes": len(hashes),
            }
            table.append(record)
            print("{file}: id={id} api={api} scenario={scenario} dpi={dpi} detail={detail} n={n} pages={pages}".format(**record))
            print("   render cycle-ms min {cyc_min_ms:.2f}  med {cyc_med_ms:.2f}  p90 {cyc_p90_ms:.2f}  max {cyc_max_ms:.2f}"
                  "   wall med {wall_med_ms:.2f}  contention {contention:.2f}"
                  "   first compile {first_compile_ms:.2f} render {first_render_ms:.2f}  distinct hashes {hashes}".format(**record))
            if show_profile:
                total = best.get("render.total") or best["render_wall_ns"]
                clock = best_meta.get("clock_read_ns", 0)
                durations = sorted(((k, v) for k, v in best.items()
                                    if k not in META_KEYS and isinstance(v, (int, float)) and is_duration(k)),
                                   key=lambda kv: kv[0])
                counters = sorted(((k, v) for k, v in best.items()
                                   if k not in META_KEYS and isinstance(v, (int, float)) and not is_duration(k)),
                                  key=lambda kv: kv[0])
                reads = sum(v for k, v in counters if k.endswith("clock_reads"))
                print(f"   fastest warm iteration: render.total {total / 1e6:.2f} ms; "
                      f"clock reads {reads} x {clock} ns = {reads * clock / 1e6:.2f} ms of instrumentation")
                for k, v in durations:
                    print(f"      {k:44s} {v / 1e6:10.3f} ms  {100.0 * v / total:6.1f}%")
                for k, v in counters:
                    print(f"      {k:44s} {v:>14}")
    if csv_out and table:
        with open(csv_out, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(table[0].keys()))
            writer.writeheader()
            writer.writerows(table)


if __name__ == "__main__":
    main(sys.argv[1:])
