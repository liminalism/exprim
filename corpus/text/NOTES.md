# corpus/text: synthetic text fixtures

Synthetic, written from scratch by `probe synth` (the `lege-render-probe` harness,
`dev-notes/render-refinement/scoping-20261108/harness/probe`), using exprim's
`test_support` PDF builder and its self-authored minimal fonts (`minimal_ttf`,
`minimal_type1`: one triangle glyph for `A`). No third-party content or font data.
Deterministic: regenerating gives byte-identical files. Redistributable under the
repository's MIT licence.

Regenerate (from the repository root, with the probe built with `--features synth`):

    cargo build --profile debug-fast --features synth --manifest-path dev-notes/render-refinement/scoping-20261108/harness/probe/Cargo.toml --target-dir <target>
    <target>/debug-fast/lege-render-probe synth corpus

All are single-page US Letter. Op counts are from `pdfr dump <pdf> 0`.

| File | Exercises | Approx. ops (page IR) | Bytes |
|---|---|---|---|
| text-modes.pdf | Helvetica-Bold 64 pt "Mode N Ag" under `Tr` 0-7 (fill, stroke, fill+stroke, invisible, and the four clip modes each followed by 16 stripe fills under the clip) | 8 text runs, 4 clip-text, 64 fills | 3,119 |
| text-rotated.pdf | one line at 10/18/28 pt under 15, 45, 90 and 133 degree rotated text matrices, plus six shear / non-uniform-scale matrices (Times-Roman, non-embedded) | 18 text runs | 1,926 |
| text-substituted.pdf | 11 non-embedded standard-14 faces (Helvetica, Times, Courier families with Bold/Oblique/Italic), a 4-line paragraph each at 9 pt | 55 text runs, 11 fonts | 6,644 |
| text-type3.pdf | Type 3 font with 6 glyph procedures (square, triangle, circle, diamond, stroked cross via `d1`, two-colour glyph via `d0`), 30 lines x 14 glyphs | 420 glyphs (420 paint-origin groups), 446 fills, 120 strokes | 3,570 |
| text-truetype.pdf | embedded minimal TrueType (`FontFile2`, WinAnsi, glyph 'A' only) at 7 sizes, one row in fill+stroke mode, three rotated rows | 10 text runs | 1,754 |
| text-type1.pdf | embedded minimal Type 1 (`FontFile`, eexec-encrypted, glyph 'A' only), same layout as above | 10 text runs | 2,434 |

The embedded fonts only define the glyph 'A', so these pages show triangle outlines
rather than readable text; they exercise glyph-cache, outline and font-program paths.
