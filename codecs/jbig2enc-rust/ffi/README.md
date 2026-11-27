# jbig2enc-ffi

A stable C ABI for [`jbig2enc-rust`](https://github.com/liminalism/exprim/tree/main/codecs/jbig2enc-rust),
a pure-Rust JBIG2 encoder and decoder. It builds one library, `jbig2rs`
(`jbig2rs.dll`, `libjbig2rs.so`, `libjbig2rs.dylib`, or a static library),
plus a checked-in header, [`include/jbig2rs.h`](include/jbig2rs.h).

C and C++ use it directly. C# (`LibraryImport`), Java 22+ (`jextract`), Swift,
Go and Zig generate their bindings from the header, with no Rust beyond
building the library. Python has its own binding next to this one
(`../python`, the `jbig2enc` wheel); both are thin layers over the same codec
API, so they behave identically.

## Build

```sh
cargo build -p jbig2enc-ffi --release          # from codecs/jbig2enc-rust
```

The output is in `target/release/`:

| Platform | Shared | Import / static |
| --- | --- | --- |
| Windows (MSVC) | `jbig2rs.dll` | `jbig2rs.dll.lib` (import), `jbig2rs.lib` (static) |
| Linux | `libjbig2rs.so` | `libjbig2rs.a` |
| macOS | `libjbig2rs.dylib` | `libjbig2rs.a` |

Linking the static library also needs the platform's system libraries.
`cargo rustc -p jbig2enc-ffi --release -- --print native-static-libs` lists
them.

The `parallel` feature enables Rayon inside the codec's planner. It is off by
default, because a host that already encodes documents in parallel rarely
wants a second thread pool.

## Use

```c
#include "jbig2rs.h"

Jbig2rsImage image = {
    .width = w, .height = h, .stride = stride, .data = rows,
    .format = JBIG2RS_FORMAT_PACKED1,           /* or _BYTES, _GRAY8 */
    .polarity = JBIG2RS_ONE_IS_BLACK,           /* or _ZERO_IS_BLACK */
};
Jbig2rsEncodeOptions options;
jbig2rs_encode_options_init(&options);
options.preset = JBIG2RS_PRESET_TEXT_LOSSLESS;

Jbig2rsBuffer globals = {0}, page = {0};
if (jbig2rs_encode_pdf(&image, &options, &globals, &page) != JBIG2RS_OK) {
    fprintf(stderr, "%s\n", jbig2rs_last_error_message());
}
/* globals -> the /JBIG2Globals stream (len 0: none); page -> the image stream */
jbig2rs_buffer_free(&globals);
jbig2rs_buffer_free(&page);
```

[`examples/roundtrip.c`](examples/roundtrip.c) is a complete program: it
encodes a padded 1bpp image, decodes it, and checks every pixel.

### What it covers

- **Input** in the layouts native code already holds: MSB-first packed rows
  with any stride, as in PBM, Leptonica `PIX`, TIFF strips and 1bpp DIBs, in
  either polarity; one byte per pixel; or 8-bit grayscale with a threshold.
- **One-shot encoding** to a standalone `.jb2` file
  (`jbig2rs_encode_file`) or to PDF fragments (`jbig2rs_encode_pdf`).
- **Multi-page documents** that share one symbol dictionary
  (`jbig2rs_encoder_*`). PDF readers need one `/JBIG2Globals` referenced by
  every page stream. Pages encoded independently and stitched together do not
  decode.
- **Bounded memory** for long inputs (`jbig2rs_group_encoder_*`). The encoder
  calls your sink with a self-contained group (its own globals plus page
  streams) every N pages or N bytes of source bitmap.
- **Decoding** standalone files (`jbig2rs_decode_file`) and PDF page streams
  against globals that are decoded once and shared across threads
  (`jbig2rs_globals_decode`, `jbig2rs_decode_pdf_page`). Bitmaps are copied
  out at any stride and polarity (`jbig2rs_bitmap_copy_rows`). Decode limits
  (`max_page_pixels`) bound what untrusted input can allocate.

### Presets

| Constant | Output |
| --- | --- |
| `JBIG2RS_PRESET_DEFAULT` | Symbol coding (lossy on near-identical glyphs) |
| `JBIG2RS_PRESET_TEXT` | Symbol coding tuned for text |
| `JBIG2RS_PRESET_GENERIC` | Generic region: bit-exact |
| `JBIG2RS_PRESET_LOSSLESS` | Generic region, no dictionary: bit-exact, the safest for viewers that mishandle globals |
| `JBIG2RS_PRESET_TEXT_LOSSLESS` | Pixel-exact symbol coding |
| `JBIG2RS_PRESET_TEXT_LOSSLESS_MAX_COMPRESSION` | Pixel-exact, slower, smaller |
| `JBIG2RS_PRESET_TEXT_SYMBOL_UNIFY` | Lossy glyph unification |
| `JBIG2RS_PRESET_TEXT_SYMBOL_UNIFY_SAFE` | Lossy unification with a topology guard |

`options.refine` (-1 keep, 0 off, 1 on) applies to the symbol presets only.

## Contract

- **Ownership.** Free everything the library allocates with the matching
  `jbig2rs_*_free` function, never with `free()`. Pointers from accessors
  (`jbig2rs_pdf_output_page`, `jbig2rs_document_page`, a sink's group) are
  borrowed and stay valid until their owner is freed.
- **Errors.** Every function that can fail returns a `JBIG2RS_*` status: 0
  means success, negative values are errors. It also records a message for
  `jbig2rs_last_error_message()`, which is per thread.
- **Panics never cross the boundary.** A bug inside the codec returns
  `JBIG2RS_ERR_PANIC`. The handle involved is then poisoned
  (`JBIG2RS_ERR_BAD_STATE`), but the library stays usable.
- **Threading.** A handle may move between threads but must not be used by
  two at once; distinct handles are independent. A `Jbig2rsGlobals` is
  immutable and may be shared freely.
- **Callbacks** (group sinks) run on the calling thread and must not unwind
  through the library: no C++ exceptions, no `longjmp`.
- **Symbols** are all prefixed `jbig2rs_`, so they do not collide with
  jbig2dec's `jbig2_*` in hosts that link both.

## ABI stability

`jbig2rs_abi_version()` returns `major << 16 | minor` (currently 1.0). Within
a major version:

- Functions are never removed and their signatures never change.
- `Jbig2rsImage` is frozen.
- `Jbig2rsEncodeOptions` and `Jbig2rsDecodeOptions` begin with
  `struct_size`. New fields are appended, and the library reads only the
  fields a caller's `struct_size` covers, so callers built against older
  headers keep working. Always initialise them with the `*_options_init`
  functions.
- New status codes, presets and formats may be added. Treat an unknown
  negative status as a generic failure.

The header is generated by cbindgen and checked in. `cargo test -p
jbig2enc-ffi` fails if it drifts from the source; regenerate it with
`JBIG2RS_BLESS_HEADER=1 cargo test -p jbig2enc-ffi --test header`.

## C# example

```csharp
using System.Runtime.InteropServices;

[StructLayout(LayoutKind.Sequential)]
unsafe struct Jbig2rsImage {
    public uint Width, Height; public nuint Stride; public byte* Data;
    public uint Format, Polarity; public byte Threshold;
}
[StructLayout(LayoutKind.Sequential)]
struct Jbig2rsEncodeOptions { public uint StructSize, Preset; public int Refine; }
[StructLayout(LayoutKind.Sequential)]
unsafe struct Jbig2rsBuffer { public byte* Data; public nuint Len; }

static unsafe partial class Jbig2rs {
    [LibraryImport("jbig2rs")] public static partial void jbig2rs_encode_options_init(Jbig2rsEncodeOptions* o);
    [LibraryImport("jbig2rs")] public static partial int jbig2rs_encode_file(Jbig2rsImage* i, Jbig2rsEncodeOptions* o, Jbig2rsBuffer* b);
    [LibraryImport("jbig2rs")] public static partial void jbig2rs_buffer_free(Jbig2rsBuffer* b);
}
```

A .NET `Bitmap` in `Format1bppIndexed` maps to `JBIG2RS_FORMAT_PACKED1`, with
`BitmapData.Stride` as the stride. Its polarity depends on the palette.

## Licence

`MIT AND Apache-2.0`, the same as the codec. A binary build includes the
codec's Apache-2.0 arithmetic-coder files, so distribute `NOTICE` (included
here) with it.
