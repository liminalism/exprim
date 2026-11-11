# Changelog

## Unreleased

Decoded output is bit-identical to 0.3.1 on every stream 0.3.1 decoded.

### Decoding performance

- MQ decoder in the software-conventions form (ITU-T T.88 Annex G) with a
  64-bit code register that buffers input ahead, two plain bytes at a time
  away from 0xff bytes. A context holds its whole probability state, so a
  decision is one load.
- Two forms of the MQ decision. `decode_with_ctx` tests first for an MPS
  that needs no renormalisation, one subtraction and one compare; the
  cleanup pass and the run-length symbols, which mostly come out that way,
  use it. `decode_unpredictable` runs every decision through one
  branch-free sequence; refinement bits, signs and significance-pass
  decisions, which a branch predictor cannot learn, use that.
- The three Tier-1 passes derive, from each stripe column's flag word, the
  set of samples they code and visit only those, instead of testing every
  sample in every pass. Block scratch is kept in coding order (the four
  samples of a column adjacent).
- Gray output runs in row bands across the worker budget: the finest 9/7
  level with its conversion to bytes, and the 5/3 and planar 9/7 writers.
  The eight-bit conversion loop is separate from the rescaling one so the
  compiler vectorises it.
- Measured against 0.3.1 on 9 to 11 megapixel OpenJPEG-encoded streams, the
  two builds run alternately on one pinned core: lossless 5/3 gray scan
  1275 → 715 ms, lossless 5/3 RGB 3745 → 2311 ms, 9/7 RGB 636 → 471 ms,
  9/7 gray photograph 298 → 212 ms, 9/7 gray scan 144 → 108 ms. With four
  workers: 347 → 213, 1057 → 705, 177 → 133, 108 → 61 and 65 → 35 ms.
- Not adopted: decoding two code blocks at a time on one core, a stripe
  column of each in turn, so that their arithmetic decoders overlap. It
  works in isolation and is 3 to 8 % slower in the decoder. Method,
  numbers, benchmark and prototype are in
  `decode-fix-plan/mq-interleave-experiment/`.

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
