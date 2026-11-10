# CLAUDE.md

exprim is a read-only, memory-bounded, concurrent PDF engine (reading, text,
rendering, printing), published to crates.io as one package, `exprim`. It was
extracted from the Lege repository (`../Lege-ecosystem`) on 2026-11-09; Lege
and Liminaldoc consume it by path.

## Layout

| Path | What |
|---|---|
| `src/` | The `exprim` package. Each former `pdf-*` crate is a module (`render_cpu`, `content`, `font`, …); `src/reader/` is the document API re-exported at the crate root; `src/print/` is behind the `print` feature; `src/test_support/` behind `test-support`. |
| `build.rs`, `upstream/pdfium/` | Unicode tables generated from vendored PDFium sources (SHA-256 checked) |
| `fonts/` | Bundled standard-14 faces (embedded with `include_bytes!`) |
| `tests/` | Integration suites, named `<module>_<suite>.rs`; fixtures under `tests/fixtures/<module>/` |
| `cli/` | `exprim-cli`, binary `pdfr` (dev driver), `exprim-demo` (`--features demo`) |
| `chaos-tests/` | Never-panic mutation gate |
| `agent/` | `exprim-agent`: the `exprim` CLI and MCP server. **Own workspace**: it links `lege-ocr` from `../../Lege-ecosystem` |
| `fuzz/`, `tools/pdfium-diff`, `tools/pdfium-text-diff` | Own workspaces (cargo-fuzz; PDFium differential oracle) |
| `codecs/` | `jp2lam`, `jbig2enc-rust`, `djvulibrust`: separately published codecs, excluded from the workspace |
| `corpus/`, `docs/` | Test PDFs; architecture and publication plan |
| `tools/render-probe` | Measurement probe (`bench`, `census`, `diff`, `synth`), A/B hash gate and attribution scripts; own workspace |
| `.akr/`, `docs/generated/` | Project ledger (decisions, observations, plans) and its generated views |
| `dev-notes/`, `docs/benchmarks/` | Working notes, sweep history and benchmark results. **Local only, git-ignored**: they describe private documents |

The wgpu image backend and GPU postprocess executor are **not** here: they
need `lege-gpu` and live in Lege as `lege-render-wgpu`. exprim must never
depend on a Lege crate (the agent's OCR edge is the one exception, and the
agent is unpublished).

## Build and test

```sh
cargo build                                   # library + cli + chaos tests
cargo test --workspace --features print       # full suite (~1,200 tests)
cargo clippy --workspace --all-targets --features print
cargo build --profile debug-fast -p exprim-cli --bin pdfr
(cd agent && cargo build --profile debug-fast --bin exprim)
```

Prefer `--profile debug-fast` over `--release` when iterating: release uses fat
LTO. Keep `panic = "unwind"` in every profile; the per-page `catch_unwind`
guards depend on it.

## Publishing

The package is an `include` allowlist in `Cargo.toml`: library sources, the
build script and the data it embeds, README, CHANGELOG, LICENSE-MIT and
THIRD-PARTY-NOTICES. Check with `cargo package --list --allow-dirty` before any
release. `jp2lam` 0.3.1 must be on crates.io before `exprim` can publish
(`image/jpx.rs` calls `DecodeMetadata::reduce_levels`). See
`docs/CRATES-IO-PLAN.md`.

## Rules

- Library code denies `unwrap`, `expect` and `panic` (workspace clippy lints).
- No global state: every page renders against an immutable snapshot with
  worker-owned scratch.
- Performance changes must keep page hashes identical; the history is in
  `dev-notes/render-refinement/performance-history/` (local only); findings
  that should be public go into the ledger.
- Never name a private local document in a tracked file: no path, file name,
  title or library directory. Cite an `akr secret` instead (ledger policy
  `exprim.privacy.private-document-references`).
