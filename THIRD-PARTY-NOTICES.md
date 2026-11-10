# Third-party notices

exprim is MIT-licensed. Two kinds of material in the package come from the
PDFium project, copyright The PDFium Authors and Foxit Software Inc., and are
distributed under PDFium's BSD three-clause licence. Its full text is in
`upstream/pdfium/LICENSE`.

## PDFium Unicode and text-extraction data

`upstream/pdfium/` holds byte-for-byte source files from PDFium commit
`33a80ac7c309a8686d1c34732eae8559a3a9ba82`. `build.rs` checks their SHA-256
digests and generates the bidirectional, mirror and Unicode normalization
tables from them at compile time.

## Bundled standard-14 font faces

`fonts/*.otf` are PDFium's bundled Foxit faces
(`core/fxge/fontdata/chromefontdata/*.cpp`). Their CFF bytes are unchanged;
each is wrapped once, offline, in a minimal OpenType container whose tables
are derived from the CFF data. `fonts/README.md` documents the provenance.
Each source file carries:

> Copyright 2014 The PDFium Authors
> Use of this source code is governed by a BSD-style license that can be found
> in the LICENSE file.
> Original code copyright 2014 Foxit Software Inc. <http://www.foxitsoftware.com>
