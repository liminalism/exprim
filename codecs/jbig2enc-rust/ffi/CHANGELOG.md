# Changelog

## 0.1.0 — unreleased

First release: ABI 1.0 over `jbig2enc-rust` 0.6.3.

- Encoding of packed 1bpp, byte-per-pixel and grayscale input, with any
  stride and either polarity, to standalone files or PDF fragments.
- Multi-page encoding with one shared dictionary; bounded-memory group
  encoding with a sink callback.
- Decoding of standalone files and PDF page streams, with shareable decoded
  globals, decode limits, and bitmap copy-out at any stride and polarity.
- Status codes with thread-local messages; unwind guards with handle
  poisoning; `struct_size`-versioned option structs.
