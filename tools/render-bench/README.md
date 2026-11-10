# Four-engine renderer benchmark

This harness ports the ledger's 2026-11-06 head-to-head to the extracted `exprim`
crate. Results are in [`docs/benchmarks/2026-11-09`](../../docs/benchmarks/2026-11-09/README.md).
It is an unpublished development tool; it changes no production rendering code.

## Setup on Windows x64

Rust 1.99.1 and Python 3.13.17 were used. From the repository root, run:

```powershell
. .\tools\render-bench\setup.ps1
$benchPython = (Resolve-Path tools\render-bench\.venv\Scripts\python.exe).Path
$env:BENCH_PDFIUM_DLL = (Resolve-Path tools\render-bench\vendor\bin\pdfium.dll).Path
& $benchPython tools\render-bench\corpus.py
& $benchPython tools\render-bench\provenance.py
& $benchPython tools\render-bench\run.py --reps 3
& $benchPython tools\render-bench\complete.py
& $benchPython tools\render-bench\memory_scaling.py
& $benchPython tools\render-bench\audit.py
& $benchPython tools\render-bench\report.py
```

`setup.ps1` checks the PDFium archive/DLL and Noto CJK font hashes. The fonts are
local controls, excluded by this directory's `.gitignore`; their license is
retained beside them. `hayro` 0.9.0, PDFium chromium/8125, and PyMuPDF 2.1 / MuPDF
1.28.6 deliberately match the prior run. Cargo lockfiles pin transitive Rust
dependencies. Python requirements pin the needed packages.

The recorded run reused the already-acquired PDFium DLL, Python packages, and
Noto fonts from the kept 2026-11-04 harness in the sibling Lege checkout. It used
the **base** Python interpreter with `PYTHONPATH` pointing at those venv packages.
This avoids a venv launcher process, but `winjob.py` accounts for its descendants
even if a venv interpreter is used. The exact command lines and binary hashes are
in the results. The acquisition-only setup script is supplied for reproduction;
it was not used to reacquire the same controls during this run.

The corpus manifest contains private absolute input paths; PDFs are not copied
or distributed. `corpus.py` selects nine files before timing and records hashes,
page counts, and an image-filter census of **every page**. Change `CHOICES` and
the output directory constants for a different corpus/run. Preserve the manifest
when reproducing the published result.

The runner resumes by skipping existing result JSON. For a fresh benchmark,
choose a new output directory; existing results are never silently overwritten.
`--smoke`, `--only jp2`, and `--engines pdfium mupdf` allow focused checks.

## Workload and correctness

The two Rust adapters use equivalent release settings: opt-level 3, fat LTO,
one codegen unit, unwind. The exprim adapter ports the prior low-level
`PageCompiler` / `CpuBackend` / bounded scheduler harness, with annotations and
system font fallback. The hayro adapter ports the prior per-thread `RenderCache`
and embedded standard-14 / Noto CJK font resolver. It preserves the native owning
pixmap instead of cloning full images. Native controls use the original ctypes
PDFium and PyMuPDF calls, with a read-only memory-mapped PDFium input instead of
two full Python copies.

The PDFium bitmap format is BGRA, but render flag `0x10`
(`FPDF_REVERSE_BYTE_ORDER`) requests RGBA bytes. The exporter preserves that
order. A generated red-page calibration checks every engine's RGB output before
the bulk raster pass; changing this exporter to BGRA decoding would be wrong.

Whole documents render at 72/144 DPI, 1/8 workers, three repetitions, engines
rotated each repetition. Every child is pinned to logical CPUs 0–11. Documents
open before the timer; page interpretation, decoding, rasterization, and native
pixels are timed. PNG, scoring, hashes, process startup, and input loading are
outside it. See `run.py` for the four environment controls on nested parallelism.

`complete.py` builds the untimed metric kernel, renders every page with every
engine at both DPIs, scores all six pairs using the old thresholds, and retains
page hashes and review sheets. It then times the **identical all-four-agree page
subset** for every engine, with three repetitions; if all pages passed, it reuses
the identical all-page jobs. A page with no common agreement is not timed in that
subset. Full PNGs are temporary and are removed only after metrics/hashes/review
sheets have been written. Dumps are not used for timing claims.
PNG encoding uses up to four threads in the native-control coordinator; PDF API
calls remain confined to worker processes. Four scoring processes compare
independent pages. Both forms of parallelism are confined to the untimed raster
pass and do not change the measured rendering budgets.

The final phase samples first/middle/last pages: one first render, five warmups,
twenty warm measurements, one worker. It verifies source hashes after all jobs.
The report generator rejects incomplete timing matrices.

## Memory

`winjob.py` launches suspended, assigns to a Job Object, sets affinity, and resumes.
It samples **all job PIDs** at 10 ms and queries Windows' continuously tracked
`PeakJobMemoryUsed`. No manual process-name matching or inference from a launcher
PID is used. The job has kill-on-close so a timeout/error cannot leave renderer
workers behind.

Retained metrics distinguish sampled aggregate RSS, sampled aggregate private
bytes, OS peak job committed memory, and the sum of individual peak working sets.
The latter bounds the sampled aggregate, since individual peaks may occur at
different times; a final per-process peak can be missed if the process exits
between polls. The continuously tracked job commit peak does not have that gap.
Shared mappings/DLLs can be counted repeatedly in aggregate RSS. Python overhead,
the native coordinator, eight rendering processes, and console hosts are included.
The headline memory table uses full-document jobs, not the agreement-filtered subset.

`audit.py` checks matrix completeness, success counts, real PDFium/MuPDF worker
capture, memory values, source stability, and page/hash coverage. It also runs an
independent child-allocation probe against the job accountant when requested with
`--probe`, after timing has finished.

The method and limitations, including compiler changes since November 6 and
exprim's partial PDFium lineage, are stated in the result report.
