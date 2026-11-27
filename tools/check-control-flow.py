#!/usr/bin/env python3
"""Flat-control-flow gate: more than three levels of control-flow nesting, or a
cognitive complexity above the clippy.toml threshold, fails.

Why a script and not just `cargo clippy`: `clippy::excessive_nesting` counts
every enclosing block from the file root, so one threshold cannot be exact. The
committed `excessive-nesting-threshold = 5` flags a fourth control level in a
method (impl + fn + 3) but only a fifth in a free function, and it flags a
third in a method of an inline module. This script runs Clippy once per
threshold and judges each function at `BASE + its own item depth`, which makes
the rule the same everywhere: three levels pass, a fourth fails.

It also audits the exceptions. An exception is a function-level
`#[allow(clippy::excessive_nesting, reason = "...")]` (or
`cognitive_complexity`); a crate- or module-wide `#![allow]`, or one with no
reason, fails the gate.

    tools/check-control-flow.py                 # every workspace
    tools/check-control-flow.py jp2lam agent    # some of them
    tools/check-control-flow.py --list          # workspace names
    tools/check-control-flow.py --exceptions    # print the exception inventory
    tools/check-control-flow.py --only src/font # report hits under a path only

Exit status: 0 clean, 1 findings, 2 a workspace failed to compile.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# name -> (directory, extra cargo arguments). Every Cargo workspace in the
# repository: the root one and the build graphs it excludes.
WORKSPACES = {
    "root": (".", ["--workspace", "--features", "print"]),
    "agent": ("agent", []),
    "fuzz": ("fuzz", []),
    "pdfium-diff": ("tools/pdfium-diff", []),
    "pdfium-text-diff": ("tools/pdfium-text-diff", []),
    "render-bench-exprim": ("tools/render-bench/exprim", []),
    "render-bench-hayro": ("tools/render-bench/hayro", []),
    "render-probe": ("tools/render-probe/probe", []),
    "jp2lam": ("codecs/jp2lam", []),
    "jp2lam-fuzz": ("codecs/jp2lam/fuzz", []),
    "jbig2enc": ("codecs/jbig2enc-rust", []),
    "jbig2enc-ffi": ("codecs/jbig2enc-rust/ffi", []),
    "jbig2enc-python": ("codecs/jbig2enc-rust/python", []),
    "jbig2enc-fuzz": ("codecs/jbig2enc-rust/fuzz", []),
    "djvulibrust": ("codecs/djvulibrust", []),
}

# A function body is one block; three control levels inside it make four.
BASE = 4
# Item depths 0..3: free fn, method, method in an inline module, and one more.
MAX_ITEM_DEPTH = 3
COGNITIVE = 25
LINTS = ("clippy::excessive_nesting", "clippy::cognitive_complexity")

FN_LINE = re.compile(
    r"^(\s*)(?:pub(?:\([^)]*\))?\s+)?"
    r'(?:(?:default|const|async|unsafe|extern(?:\s+"[^"]*")?)\s+)*fn\s+(\w+)'
)
EXCEPTION = re.compile(
    r"#(!?)\[\s*(?:allow|expect)\s*\(([^\]]*?)\)\s*\]", re.DOTALL
)


def indent_of(line):
    return len(line) - len(line.lstrip(" "))


def enclosing_fn(lines, hit_line):
    """Return (item_depth, name, line) of the fn enclosing 1-based `hit_line`.

    Relies on rustfmt layout: an item nested in `n` blocks is indented `4 * n`.
    A candidate `fn` line is rejected when a `}` at or left of its indent sits
    between it and the hit, because that fn has already closed.
    """
    hit_indent = indent_of(lines[hit_line - 1])
    closed_at = None  # smallest indent of a line starting with `}` seen so far
    for number in range(hit_line - 1, 0, -1):
        line = lines[number - 1]
        stripped = line.strip()
        if not stripped:
            continue
        indent = indent_of(line)
        found = FN_LINE.match(line)
        still_open = closed_at is None or indent < closed_at
        if found and indent <= hit_indent and still_open:
            return indent // 4, found.group(2), number
        if stripped.startswith("}") and (closed_at is None or indent < closed_at):
            closed_at = indent
    return 0, "?", hit_line


def run_clippy(name, threshold):
    """Run Clippy on one workspace at one nesting threshold; return its hits."""
    directory, extra = WORKSPACES[name]
    cwd = os.path.join(ROOT, directory)
    with tempfile.TemporaryDirectory() as conf:
        with open(os.path.join(conf, "clippy.toml"), "w") as handle:
            handle.write(f"excessive-nesting-threshold = {threshold}\n")
            handle.write(f"cognitive-complexity-threshold = {COGNITIVE}\n")
        command = ["cargo", "clippy", "--all-targets", *extra, "--message-format=json", "--"]
        command += [flag for lint in LINTS for flag in ("-W", lint)]
        env = dict(os.environ, CLIPPY_CONF_DIR=conf)
        done = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True)
    if done.returncode != 0:
        sys.stderr.write(done.stderr[-4000:])
        raise RuntimeError(f"{name}: cargo clippy failed at threshold {threshold}")
    hits = set()
    for raw in done.stdout.splitlines():
        if not raw.startswith("{"):
            continue
        message = json.loads(raw)
        if message.get("reason") != "compiler-message":
            continue
        body = message["message"]
        code = (body.get("code") or {}).get("code")
        if code not in LINTS:
            continue
        span = next(s for s in body["spans"] if s["is_primary"])
        hits.add((resolve(cwd, span["file_name"]), span["line_start"], code, body["message"]))
    return hits


def resolve(cwd, file_name):
    """Repository-relative path for a span; path dependencies report paths
    relative to a directory above the workspace that ran Clippy."""
    base = cwd
    while not os.path.exists(os.path.join(base, file_name)) and base != "/":
        base = os.path.dirname(base)
    return os.path.relpath(os.path.realpath(os.path.join(base, file_name)), ROOT)


def check_workspace(name, sources):
    """Return the findings for one workspace as (path, line, text) tuples."""
    findings = set()
    for threshold in range(BASE, BASE + MAX_ITEM_DEPTH + 1):
        for path, line, code, text in run_clippy(name, threshold):
            if code == "clippy::cognitive_complexity":
                findings.add((path, line, text))
                continue
            lines = sources.setdefault(path, read_lines(path))
            depth, function, _ = enclosing_fn(lines, line)
            if BASE + min(depth, MAX_ITEM_DEPTH) == threshold:
                findings.add((path, line, f"fourth control-flow level in `{function}`"))
    return findings


def read_lines(path):
    with open(os.path.join(ROOT, path), encoding="utf-8") as handle:
        return handle.read().split("\n")


def tracked_rust_files():
    listed = subprocess.run(
        ["git", "ls-files", "*.rs"], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return [p for p in listed.stdout.splitlines() if not p.startswith("upstream/")]


def audit_exceptions():
    """Return (inventory, violations) for the two lints' allow attributes."""
    inventory, violations = [], []
    for path in tracked_rust_files():
        text = "\n".join(read_lines(path))
        if "excessive_nesting" not in text and "cognitive_complexity" not in text:
            continue
        for found in EXCEPTION.finditer(text):
            inner, body = found.group(1), found.group(2)
            named = [lint.split("::")[1] for lint in LINTS if lint in body]
            if not named:
                continue
            line = text.count("\n", 0, found.start()) + 1
            reason = re.search(r'reason\s*=\s*"((?:[^"\\]|\\.)*)"', body, re.DOTALL)
            if inner:
                violations.append((path, line, "crate- or module-wide exception"))
            if not reason:
                violations.append((path, line, "exception without a reason"))
            summary = " ".join(reason.group(1).split()) if reason else ""
            inventory.append((path, line, "+".join(named), summary))
    return inventory, violations


def audit_configs():
    """Every clippy.toml must carry the same two thresholds as the root one."""
    listed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "clippy.toml", "*/clippy.toml"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    wanted = None
    violations = []
    for path in sorted(listed.stdout.splitlines(), key=lambda p: (p.count("/"), p)):
        keys = {}
        for line in read_lines(path):
            pair = re.match(r"\s*([\w-]+)\s*=\s*(\S+)", line)
            if pair:
                keys[pair.group(1)] = pair.group(2)
        if wanted is None:
            wanted = keys
        elif keys != wanted:
            violations.append((path, 1, "thresholds differ from the root clippy.toml"))
    return violations


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("workspaces", nargs="*", help="workspace names (default: all)")
    parser.add_argument("--list", action="store_true", help="print workspace names")
    parser.add_argument("--exceptions", action="store_true", help="print the exception inventory")
    parser.add_argument("--only", action="append", default=[], help="report hits under this path only")
    parser.add_argument("--no-clippy", action="store_true", help="audit exceptions and configs only")
    args = parser.parse_args()

    if args.list:
        print("\n".join(WORKSPACES))
        return 0
    unknown = [name for name in args.workspaces if name not in WORKSPACES]
    if unknown:
        parser.error(f"unknown workspace: {', '.join(unknown)}")

    inventory, findings = audit_exceptions()
    findings += audit_configs()
    if args.exceptions:
        for path, line, lints, reason in inventory:
            print(f"{path}:{line}: [{lints}] {reason}")

    broken = False
    sources = {}
    for name in [] if args.no_clippy else (args.workspaces or list(WORKSPACES)):
        try:
            found = check_workspace(name, sources)
        except RuntimeError as error:
            print(error, file=sys.stderr)
            broken = True
            continue
        print(f"{name}: {len(found)} finding(s)", file=sys.stderr)
        findings += sorted(found)

    findings = sorted(set(findings))
    if args.only:
        findings = [f for f in findings if any(f[0].startswith(p) for p in args.only)]
    for path, line, text in findings:
        print(f"{path}:{line}: {text}")
    print(
        f"{len(findings)} finding(s), {len(inventory)} documented exception(s)",
        file=sys.stderr,
    )
    if broken:
        return 2
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
