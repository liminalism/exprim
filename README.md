# exprim

A read-only, memory-bounded, concurrent PDF engine in Rust: open a document,
read its text, and rasterize its pages.

exprim is a semantic port of PDFium's behaviour with a different core. Every
page renders against an immutable document snapshot, and all execution state
and scratch memory belong to the worker rendering that page. Many threads can
render one document at once, and the output does not depend on how many.

```rust
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let bytes: Arc<[u8]> = std::fs::read("in.pdf")?.into();
    let session = exprim::RenderSession::open(bytes, None)?;

    println!("{} pages", session.page_count());
    println!("{}", exprim::page_text(&session, 0)?);

    let png = session.export_page(0, &exprim::ExportOptions::png(150.0))?;
    std::fs::write("page-1.png", png)?;
    Ok(())
}
```

## What it covers

- **Document structure:** classic and stream xrefs, incremental updates,
  object streams, damaged-file recovery, and limits on every walk.
- **Encryption:** RC4 and AES, revisions 2 to 6, user and owner passwords.
- **Filters:** Flate, LZW, ASCII85, ASCIIHex, RunLength, DCT, CCITT, JBIG2
  and JPX, with PNG and TIFF predictors.
- **Painting:** paths, clipping, all colour spaces including ICC-based and
  DeviceN, shading types 1 to 7, tiling and shading patterns, transparency
  groups, soft masks, and every blend mode.
- **Fonts:** Type 1, CFF, TrueType, OpenType and Type 3, CJK CMaps, and
  vertical text. The standard 14 faces are bundled, so output does not
  depend on the host's fonts.
- **Text:** PDFium-parity character order, bidi, normalization, word boxes,
  and text-in-rectangle.
- **Annotations and optional content:** appearance streams and OCG
  visibility.

Not rendered: 3D and multimedia annotations, JavaScript, generated form-field
appearances, and spot-colour overprint simulation.

## Guarantees

These are tested properties of the engine:

- **No panic crosses a page boundary.** A malformed page becomes a typed
  error for that page. Library code denies `unwrap`, `expect` and `panic`, and
  a mutation suite and nine fuzz targets check the invariant. Keep
  `panic = "unwind"` in your release profile: the per-page guards rely on it.
- **Memory and time are bounded.** Object counts, reference chains, decoded
  bytes, content nesting and raster size all have limits, and every codec
  input and output has a ceiling.
- **Rendering can be cancelled.** A cancellation token is checked by the
  scheduler, the rasterizer, and inside the image decoders.
- **Output is deterministic.** The same input and options produce the same
  bytes whatever the thread count, worker assignment or cache state.

## Features

| feature | effect |
|---|---|
| `print` | Office printing: imposition, N-up, sheet composition, and CUPS or winspool spooling (`exprim::print`) |
| `profiling` | Per-stage timers |

## Stability

The document API at the crate root (`RenderSession`, `page_text`,
`positioned_words`, `extract_outline`, `extract_metadata`, `ExportOptions`,
`encode_plane`) is the supported surface. The engine's layers (syntax,
document, content, render) are public modules so that tools can be built on
them, but they are hidden from the documentation and may change in any minor
release before 1.0.

MSRV: Rust 1.97.

## Licence

MIT. The bundled standard-14 font faces and the Unicode tables generated at
build time come from PDFium and are BSD-3-Clause; see
[`THIRD-PARTY-NOTICES.md`](THIRD-PARTY-NOTICES.md).
