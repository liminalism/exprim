# Changelog

## 0.6.2 — 2026-11-24

Decoder-side optimisations, two decoder edge-case fixes, and README updates
(replacing the stale `0.6.0 (unreleased)` section). Decoded bitmaps, decoder
state and context states are bit-identical to 0.6.1 on all valid streams; the
public API and the `decode`-only feature build are unchanged.

### Decoding

- 64-bit buffered Annex E MQ arithmetic decoder (`src/decode/arith.rs`) with a
  packed state table, branch-free conditional exchange, `leading_zeros()`
  single-shift renormalisation, two-byte plain refill, and deferred handling of
  carry-bearing `0xFF 0x80..=0x8F` stuffed bytes so malformed streams remain
  bit-identical to byte-at-a-time Annex E (retained under `cfg(test)` as
  `ReferenceArithmeticDecoder`).
- Fast word-level `MonoBitmap::combine` (`src/shared/bitmap.rs`) with up-front
  X/Y clipping, a specialised single-word fast path for unclipped glyphs and
  patterns, and word-aligned multi-word row blending.
- Windowed refinement-region decoder (`src/decode/refinement.rs`) for
  `GRTEMPLATE` 0 and 1 (nominal and custom `GRAT1`/`GRAT2`, 2-op `TPGRON`
  typical-pixel test, word-wise `page_reference_window`), verified against the
  previous per-pixel decoder kept under `cfg(test)`.
- Direct-into-bitmap row writes and single-copy `TPGDON` duplicate rows in the
  generic decoder (`src/decode/generic.rs`), direct-into-page decoding for
  full-cover immediate generic regions and O(1) clean-page tracking in
  `src/decode/page.rs`.
- Word-wise halftone Gray-to-binary plane decoding without duplicate plane
  allocation (`src/decode/halftone_region.rs`), 8-bit prefix-LUT + canonical
  Huffman decoding (`src/decode/huffman.rs`), word-wise MMR black-run filling
  (`src/decode/mmr.rs`), word-wise symbol/pattern extraction
  (`src/decode/symbol_dictionary.rs`, `src/decode/pattern_dictionary.rs`), and
  tighter integer/IAID arithmetic decoding (`src/decode/integer.rs`,
  `src/decode/iaid.rs`, `src/decode/text_region.rs`).
- On `benches/decode.rs` (i7-13700H): `decode_generic_a4_300dpi` 14.05 →
  11.99 ms (~1.17× faster), `decode_symbol_page` 6.38 → 2.54 ms (~2.5×
  faster), `decode_refined_symbol_page` 6.80 → 2.52 ms (~2.7× faster),
  `decode_halftone_page` 13.59 → 5.91 ms (~2.3× faster).

### Correctness

- Pattern dictionary (`src/decode/pattern_dictionary.rs`): `HDPW == 128`
  previously panicked on `-i8::MIN` overflow in debug builds, and
  `HDPW = 129..=255` truncated `AT1 = (-HDPW, 0)` to `-128`. The generic
  decoder now supports wide current-row AT offsets down to `-255`.
- Page composition (`src/decode/page.rs`): unsigned `u32` region coordinates
  greater than `i32::MAX` previously wrapped to negative signed coordinates via
  `as i32` instead of clipping outside the page.

### Documentation and tests

- `README.md`: replaced the inaccurate `0.6.0 (unreleased)` heading with a
  `0.6` summary covering `0.6.0`–`0.6.2`, bumped the quick-start dependency
  snippet to `0.6.2`, and updated the compatibility and test notes.
- `tests/decode_refine_templates.rs`: verified `GRTEMPLATE = 1` + `TPGRON`
  against ITU-T T.88 §6.3.5.6 Figure 15 (`SLTP` context `"GR0000001000"`,
  matching PDFium and pdf.js). `grtemplate1_tpgron` now tests native decoding
  and passes under plain `cargo test`; the comparison against `jbig2dec 0.20`
  (known upstream bug) is kept as an `#[ignore]` test.

## 0.6.1 — 2026-11-10

Changes since 0.6.0. Decoded bitmaps, decoder state and context states are
bit-identical to 0.6.0; the public API and the `decode`-only feature build
are unchanged.

### Decoding

- Generic-region decoding is 3.7–11× faster on page-sized PDF masks
  (one scanned page, 2222×3191: 28.8 → 7.8 ms, 4.1 → 1.1 ns per pixel). One
  windowed decoder now serves templates 0–3, nominal and non-nominal AT
  pixels, and TPGDON. It shifts a packed-word context window fed from
  per-block row look-ahead registers, folds uniform background runs into a
  single arithmetic-register update, and tests the MQ DECODE fast path
  first. The previous decoder is kept under `cfg(test)` as the equivalence
  reference; `examples/generic_decode_timing.rs` reproduces the measurement.
  The halftone skip path still forms contexts per pixel and gains only the
  faster DECODE.

### Tests

- The test-only halftone stream writer (`tests/common/writer.rs`,
  `halftone_skip_page`) left the values of skipped grid cells in the plane
  it formed contexts from; T.88 §6.2.5.7 sets a skipped pixel to 0 before
  any later context reads it, and this decoder and hayro-jbig2 both decode
  so. The `halftone_enable_skip` test passed only because its 5×4 stream
  was too short for the contexts to diverge; streams up to 10×6 cells of
  40×40 patterns now round-trip. The decoder is unchanged.

### Packaging

- The published crate is the library alone. The `tester` and
  `halftone_tester` development binaries are no longer packaged, so
  crates.io no longer offers `cargo install jbig2enc-rust`. The integration
  tests, fixtures, benchmark, example and Python test suite stay in the
  repository. The `cli` and `halftone_bin` features remain and only matter
  in a repository checkout.
- `Cargo.lock`: `spin` 0.10.0 (yanked; reached only through the optional
  non-Windows `pprof` profiling dependency) → 0.10.1.

## 0.6.0 — 2026-11-02

Changes since the last published version, 0.5.4. Published to crates.io on
2026-11-02.

### Encoding and compression

- Faster generic encoding through packed-word context windows, native-width
  arithmetic-register storage, blank-span batching and deferred transition-table
  loads. A re-run on an unloaded machine then retained one further change:
  each bitmap context caches its probability estimate next to its state
  index, which halves the context bank and removes the table read from the
  common path. All of these optimizations preserve the existing generic
  stream bytes.
- Explicit pixel-exact symbol presets retain isolated marks; legacy text and
  unify presets remain lossy and retain their settings.
- Symbol-mode encoding is about three times faster for `text()`, unify and the
  measured exact preset, and about 1.7 times faster for `text_lossless()`, on
  text pages: run-based hole counting, word-level bitmap helpers, ink-count
  bounds that settle most clustering and unify comparisons without a search,
  a single-sweep component labelling, and a trial memo in the measured
  planner. A follow-up adds dictionary prefix checkpoints and a text-region
  payload memo to that planner (about a quarter faster together), runs
  unify's context similarity only after the bitmap screens (about 40%
  faster in document scope) and forms refinement contexts from sliding
  windows. Every stream stays byte-identical; an ignored
  `symbol_speed` timing worker reproduces the measurement. See
  [symbol-mode speed](docs/SYMBOL-SPEED.md).
- Optional residual packing, selective global/local PDF dictionaries and bounded
  measured planning compare actual encoded sizes. Safe lossy substitution uses
  topology safeguards; exact presets preserve the supplied bilevel pixels.
- Grouped PDF encoding bounds retained source pages/bitmap bytes and associates
  each group's pages with its globals. Measured resource limits remain explicit.

### Correctness and compatibility

- Correct short-reference refinement predictors for independent decoder
  interoperability, including extreme signed offsets.
- New exact/measured/safe presets use canonical arithmetic termination. Legacy
  constructors and low-level flush retain their historical bytes; callers can
  explicitly opt into canonical termination.
- Existing constructors and entry points remain. New configuration/metric fields
  require changes to exhaustive downstream struct literals; prefer constructors
  or struct-update syntax. See [API compatibility](docs/API-COMPATIBILITY.md).

### Packaging

- The Python binding (`jbig2enc` on PyPI) moves into the crate as the
  `python` Cargo feature (`src/python.rs`), with its wheel metadata, stubs
  and tests under `python/`. Both ship in the crates.io package, so the
  published source carries the Rust and Python APIs. pyo3 is an optional
  dependency; `extension-module` is enabled only by maturin for the wheel.
  The wheel's license declaration now matches the crate (`MIT AND
  Apache-2.0`).
- The README is rewritten around capabilities, usage, fidelity contracts,
  features, compatibility and licensing; timing numbers live in the linked
  reports.

### Licensing and testing

- Replace the DjVu-derived component extractor and inherited tests. Project
  contributions and original fixtures are MIT; retained arithmetic components
  keep Apache-2.0 terms/notices (package: MIT AND Apache-2.0).
- Exclude the two legacy external PBMs from Cargo distribution; retain their
  explicit unchanged-hash regression in the checkout. Default tests use three
  original generated fixtures and an independent historical byte baseline.
- Retain independent-decoder tests, context/flood-fill oracles, fixed-corpus
  benchmark hashes and measured resource reports. The existing template-one
  jbig2dec disagreement remains documented rather than reported as an oracle pass.

Performance numbers in the linked reports are corpus/compiler-specific generic
kernel measurements, not a universal whole-application speed promise. See
[hardware-profiled results](docs/GENERIC-PERF-FOLLOWUP.md), the bounded 0.6
[headroom assessment](docs/GENERIC-06-HEADROOM.md) and its
[cloud re-run](docs/GENERIC-06-CLOUD.md).
