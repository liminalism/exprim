# Changelog

## Unreleased

### Decoding performance and memory

- `Jp2BufferPool` (`DEFAULT_DECODE_BUFFER_POOL_BYTES` = 128 MiB) parks and
  reuses large Tier-1 subband (`SplitSubbands` and monolithic coefficient
  planes) and streaming 9/7 and 5/3 inverse-DWT intermediate buffers across
  `Jp2Decoder` calls (`Jp2Decoder::with_buffer_pool`, `buffer_pool`,
  `set_buffer_pool`, `release_idle_memory`). Reused Tier-1 planes are zeroed in
  user space without OS page unmap/soft-fault churn, while fully overwritten
  DWT planes skip zeroing. Measured bit-identical across five scanned-page
  codestreams: wall time drops 22–34% on 9–10.5 MP colour 9/7 scans (for
  example 109.7 → 72.4 ms single-threaded, 40.7 → 27.4 ms at four threads) and
  31–50% on a grayscale 5/3 scan (21.6 → 15.0 ms single-threaded, 11.6 → 5.8 ms
  at four threads).

## 0.4.0 — 2026-11-24

Decoded output is bit-identical to 0.3.1 on every stream 0.3.1 decoded; the
public API gains `PreparedJp2` / `prepare_jp2`,
`inspect_jp2_header_for_reduction`, `Jp2LamError::Cancelled`, and cancellable
`Jp2Decoder` entry points.

### Decoding performance

- MQ decoder in the software-conventions form (ITU-T T.88 Annex G) with a
  64-bit code register that buffers input a few bytes ahead: an MPS that
  needs no renormalisation is one subtraction and one compare, and the
  exchange and renormalisation path has no data-dependent branches. A
  context holds its whole probability state, so a decision is one load.
- The three Tier-1 passes derive, from each stripe column's flag word, the
  set of samples they code and visit only those, instead of testing every
  sample in every pass. Block scratch is kept in coding order (the four
  samples of a column adjacent).
- Gray output runs in row bands across the worker budget: the finest 9/7
  level with its conversion to bytes, and the 5/3 and planar 9/7 writers.
  The eight-bit conversion loop is separate from the rescaling one so the
  compiler vectorises it (about 3% of a 9/7 gray decode).
- Streaming four-row inverse 5/3 wavelet with runtime AVX2 lifting and fused
  finest-level RCT plus byte packing (active without the `simd` feature):
  reversible decodes drop coarse subbands as `LL1` is built, track the integer
  safety bound during Tier-1 block output instead of scanning full planes, and
  reconstruct the finest level directly into the caller's packed destination
  across budgeted row bands (`dwt_ms` 154 → 96 ms on 12.5 MP lossless RGB).
- Subband-tight Tier-1 planes (`LL1` separate from `HL1` and `LH1/HH1`), inline
  per-block merge state in Tier-2 (no per-tile hash map), pooled per-worker
  Tier-1 scratch on `Jp2Decoder`, AVX2 8×4 stripe-to-row block transpose and
  IEEE-754 midpoint dequantisation, active-component filtering when trailing
  alpha is ignored, and single-component-at-a-time packed reconstruction when
  `COD` signals no multiple-component transform (`4 B/px` working storage
  instead of `8–16 B/px`).
- Measured on 11-megapixel OpenJPEG-encoded streams, single thread, against
  0.3.1: lossless 5/3 RGB 3260 → about 2000-2300 ms, lossless gray scan
  1130 → 650 ms, 9/7 gray 276 → about 200 ms, 9/7 RGB 499 → 415 ms. With
  four workers 9/7 gray goes 107 → 62 ms.

### API

- `PreparedJp2` and `prepare_jp2` parse container and codestream framing once
  without copying embedded ICC profiles, paired with
  `Jp2Decoder::decode_prepared` and `Jp2Decoder::decode_prepared_into`.
  `inspect_jp2_header_for_reduction` computes wavelet reduction levels from the
  main header up to the first `SOT` marker without walking tile-parts.
- Cooperative cancellation via `Jp2LamError::Cancelled` (`is_cancelled`) and
  `Jp2Decoder::{decode_cancellable, decode_into_cancellable,
  decode_prepared_cancellable, decode_prepared_into_cancellable}`, polled
  across Tier-2 packet reading, serial and parallel Tier-1 code-blocks, and
  post-Tier-1 synthesis bands.
- Post-Tier-1 parallel synthesis and packing bands strictly honour
  `DecodeConcurrency::Budgeted(n)` instead of fanning out to the full Rayon
  pool when `n >= 2`.

### Decoding robustness

- Bytes after a tile's last packet are ignored instead of rejected, and a
  code block signalled with more coding passes than its bit-planes hold is
  decoded up to its last bit-plane (as OpenJPEG does). The OpenJPEG
  conformance stream `kodak_2layers_lrcp.j2c` needs both.
- `JP2LAM_TRACE_T2` is read once per process, not once per packet.

## 0.3.1 — 2026-11-09

Changes since 0.3.0 (published 2026-11-03). The decoder is bit-identical to
0.3.0 on every stream in the renderer's corpus (20 streams, two reduction
levels each) and on the crate's own test suite; the public API gains one
function and one `DecodeMetadata` method.

### Decoding performance (serial)

- Streaming inverse 9/7 wavelet: each resolution level is synthesised in one
  pass, so full-resolution subband planes are never materialised.
- Tier-1 on OpenJPEG-style packed stripe-column flags with an inlined MQ
  decoder; the three passes run on exact-length stripe slices so the compiler
  can elide bounds checks (measured neutral against unchecked indexing).
- The finest wavelet level is fused with the inverse ICT and byte packing.
- AVX2 row kernels for the lifting steps and the packer, selected at runtime
  on x86-64 regardless of the `simd` cargo feature (the renderer builds
  jp2lam with default features off). The `simd` feature's `wide` kernels
  were measured neutral on the decode path and stay off by default.
- Result: single-threaded decode is 2.0–2.3× faster at full resolution and
  1.9–2.0× at one reduction level on the two reference layers (first
  layer: 385 → 167 ms; second layer: 200 → 92 ms), under
  hayro-jpeg2000 0.5.0 on all 40 corpus measurements. Numbers and method in
  `lege-misc/dev-misc/jpxbench/`.

### API

- `decode_with_stats` returns per-stage timings beside the decoded image.
- `DecodeMetadata::reduce_levels` reports how many wavelet levels a decode
  at a given `DecodeResolution` drops, using the rule every decode path
  applies. Decodes that agree on it are identical, so a cache of reduced
  decodes (the PDF renderer's) can key on the level rather than on the
  requested footprint.

### Encoder fix

- Non-terminated coding passes in segmentation-symbol styled streams reported
  the arithmetic coder's byte count at the pass boundary, up to three bytes
  short of what a decoder needs for that pass; any layer ending one or two
  passes after a cleanup pass tripped the marker check. Lengths now carry
  OpenJPEG's +3 (MQ) / +2 (raw) correction, capped at the segment. A test
  decodes every pass prefix of segmark-styled blocks across five coding styles.
  The default single-segment path is unchanged.

### Decoder fix

- A `Serial` or `Budgeted(1)` decode now stays on the calling thread after
  Tier-1 too. The per-plane 9/7 synthesis and the fused RGB(A) packers used
  to fan out over whatever Rayon pool was ambient, so a decode on a worker
  of the PDF renderer's pool spread across that whole pool. They now follow
  the request's worker budget. Output bytes are unchanged for every budget.
  A test decodes beside a work-stealing thief worker and checks it gets no job.

## 0.3.0 — 2026-11-03

First crates.io release with multiple quality layers, the six code-block
coding switches, an alpha channel, and the exposed coding parameters.
