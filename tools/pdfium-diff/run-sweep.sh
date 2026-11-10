#!/usr/bin/env bash
#
# Full differential sweep (day-plan "sweep 3") over ALL corpus roots.
#
#   1., 2. the local corpus directories named in CORPUS_DIRS
#   3. renderer-corpus          (pdfbox + pdf.js + hayro hand-picked regression
#                                PDFs — small but curated: ~2.8k files)
#
# ~14k PDFs total; the run takes hours. It does NOT start automatically — run it
# by hand once the known-error fixes are in. Results append to
# `<out>/results.csv` and the run is resumable (re-running skips file|page keys
# already recorded), so the existing 2026-07-18 baseline keys stay byte-identical
# and the new renderer-corpus rows are purely additive.
#
# Usage:
#   ./run-sweep.sh [libpdfium] [scale] [workdir]
# `workdir` (default: cwd) is where `pdfium-diff-out/results.csv` is written.
#
# Env:
#   CORPUS_ROOT  Drive or mount holding the corpus (required).
#   CORPUS_DIRS  `;`-separated directories under it that hold the local
#                corpus. Neither is recorded here: they are private.
set -euo pipefail

ROOT="${CORPUS_ROOT:?set CORPUS_ROOT to the drive or mount holding the local corpus}"
LIB="${1:-$ROOT/Rust-projects/pdfium-port-plan/libpdfium.so}"
SCALE="${2:-2.0}"
WORKDIR="${3:-$(pwd)}"

CORPORA=()
IFS=';' read -r -a _corpus_dirs <<< "${CORPUS_DIRS:-}"
for d in "${_corpus_dirs[@]}"; do
    [[ -n "$d" ]] && CORPORA+=("$ROOT/$d")
done
CORPORA+=("$ROOT/Rust-projects/pdfium-port-plan/renderer-corpus")

# Fail loudly if a corpus root is missing rather than silently sweeping less.
missing=0
for c in "${CORPORA[@]}"; do
    if [[ ! -d "$c" ]]; then
        echo "MISSING corpus root: $c" >&2
        missing=1
    fi
done
[[ "$missing" == 0 ]] || { echo "aborting: fix the roots above (CORPUS_ROOT=$ROOT)" >&2; exit 1; }
[[ -f "$LIB" ]] || { echo "MISSING libpdfium: $LIB" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "building release pdfium-diff…" >&2
(cd "$SCRIPT_DIR" && cargo build --release)
BIN="$SCRIPT_DIR/target/release/pdfium-diff"
[[ -f "$BIN" ]] || BIN="$SCRIPT_DIR/target/release/pdfium-diff.exe"

# The tool always writes ./pdfium-diff-out in its CWD (resumable), so just run
# from the chosen workdir.
mkdir -p "$WORKDIR"
cd "$WORKDIR"
echo "sweep → $WORKDIR/pdfium-diff-out/results.csv   lib=$LIB scale=$SCALE" >&2
echo "roots:" >&2; printf '  %s\n' "${CORPORA[@]}" >&2
exec "$BIN" "$LIB" "$SCALE" "${CORPORA[@]}"
