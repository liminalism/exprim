# Installing exprim as a Claude Code MCP server (Linux)

One-time setup to register `exprim` as a global (user-scope) MCP server
alongside `akr`, `codegraph`, and `fff`. The steps are the same on Windows,
with `exprim.exe` for the binary and no system packages to install.

## 1. System dependencies

A C toolchain for the crates that build native code:

```sh
sudo apt-get update
sudo apt-get install -y build-essential pkg-config
```

OCR is the embedded PP-OCR backend of `lege-ocr` (the `paddle-ocr` feature),
which is pure Rust: Tesseract and Leptonica are **not** needed, neither to
build nor at run time.

## 2. Build

```sh
cd <your checkouts>/exprim/agent   # Lege-ecosystem must sit beside exprim (lege-ocr)
cargo build --profile debug-fast --bin exprim
```

`debug-fast` = release optimizations, LTO off — the project's standard
day-to-day build profile (see root `CLAUDE.md`); don't use plain `--release`
for this, it drags in full LTO and takes far longer for a dev/CLI binary.

## 3. Install the binary

```sh
mkdir -p ~/.local/bin
cp target/debug-fast/exprim ~/.local/bin/exprim
```

## 4. Register with Claude Code (user/global scope)

```sh
claude mcp add --scope user exprim -- ~/.local/bin/exprim mcp
```

`--scope user` is required — plain `claude mcp add` defaults to **local**
scope (tied to whatever project you're in), which is *not* what the other
three global tools use.

## 5. Verify

```sh
claude mcp list
```

Expect all four connected:

```
codegraph: ... - ✔ Connected
akr:       ... - ✔ Connected
fff:       ... - ✔ Connected
exprim:  ~/.local/bin/exprim mcp - ✔ Connected
```

If `exprim` fails to connect, run the handshake by hand and read what comes
back:

```sh
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' | ~/.local/bin/exprim mcp
```

It should print one line of JSON listing ten tools, `pdf_inspect` first.
