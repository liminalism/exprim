#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // dev example: fail-fast is fine

//! Diagnostic: census of JPX (`/JPXDecode`) decode outcomes across many PDFs.
//! Reads a newline-delimited list of PDF paths from argv[1], extracts every
//! JPX stream, decodes it with `jp2lam`, and tallies normalized error
//! signatures so remaining codec gaps in the corpus are enumerable. It also
//! scans each codestream's main and tile-part headers and counts the streams
//! that carry the markers jp2lam does not decode (POC, PPM, PPT, RGN), so a
//! census answers "does any real file need them" without a second tool.
//!
//! Usage: cargo run -p exprim-cli --example jpxcensus -- <pdf-list.txt> [max_files]
//!
//! `JPXCENSUS_PER_FILE` caps the JPX streams sampled per PDF (default 5).
//! `JPXCENSUS_DUMP_OK=<dir>` writes every stream that decodes, named
//! `<pdf stem>_obj<n>.jp2`, as benchmark fixtures.

use std::collections::BTreeMap;
use std::sync::Arc;

use exprim::document::{DocumentLimits, DocumentSnapshot, ParseContext};
use exprim::object::{ObjectId, PdfObject};
use exprim::source::OwnedBytesSource;

fn normalize(msg: &str) -> String {
    // Collapse numbers so "tile 3 TPsot 5" and "tile 0 TPsot 2" coalesce.
    let mut out = String::new();
    let mut last_digit = false;
    for ch in msg.chars() {
        if ch.is_ascii_digit() {
            if !last_digit {
                out.push('N');
            }
            last_digit = true;
        } else {
            out.push(ch);
            last_digit = false;
        }
    }
    out
}

/// Markers that change Tier-2 packet iteration or coefficient scaling and that
/// jp2lam rejects: tallied per stream so their real-world frequency is known.
const WATCHED_MARKERS: [(u16, &str); 4] = [
    (0xff5f, "POC"),
    (0xff60, "PPM"),
    (0xff61, "PPT"),
    (0xff5e, "RGN"),
];

/// The raw codestream inside a JP2 container (its `jp2c` box), or the input
/// itself when it already starts with SOC.
fn codestream(data: &[u8]) -> Option<&[u8]> {
    if data.starts_with(&[0xff, 0x4f]) {
        return Some(data);
    }
    let mut pos = 0usize;
    while pos + 8 <= data.len() {
        let lbox = u32::from_be_bytes(data[pos..pos + 4].try_into().ok()?) as u64;
        let tbox = &data[pos + 4..pos + 8];
        let (header, length) = match lbox {
            0 => (8u64, (data.len() - pos) as u64),
            1 => (
                16u64,
                u64::from_be_bytes(data.get(pos + 8..pos + 16)?.try_into().ok()?),
            ),
            n => (8u64, n),
        };
        if length < header {
            return None;
        }
        if tbox == b"jp2c" {
            let start = pos + header as usize;
            let end = (pos as u64 + length).min(data.len() as u64) as usize;
            return data.get(start..end);
        }
        pos = pos.checked_add(length as usize)?;
    }
    None
}

/// Marker codes present in the main header and in every tile-part header
/// (`(main, tile)`), walking SOT `Psot` lengths. Stops quietly at the first
/// malformed segment: the census only needs what is visibly there.
fn header_markers(cs: &[u8]) -> (Vec<u16>, Vec<u16>) {
    let mut main = Vec::new();
    let mut tile = Vec::new();
    let read16 =
        |p: usize| -> Option<u16> { cs.get(p..p + 2).map(|b| u16::from_be_bytes([b[0], b[1]])) };
    let mut pos = 2usize; // past SOC
    // Main header: every segment up to the first SOT.
    loop {
        let Some(marker) = read16(pos) else {
            return (main, tile);
        };
        if marker == 0xff90 {
            break;
        }
        let Some(len) = read16(pos + 2) else {
            return (main, tile);
        };
        main.push(marker);
        pos += 2 + len as usize;
    }
    // Tile-parts: SOT, header segments up to SOD, then skip by Psot.
    while read16(pos) == Some(0xff90) {
        let tp_start = pos;
        let Some(lsot) = read16(pos + 2) else { break };
        let Some(psot) = cs
            .get(pos + 6..pos + 10)
            .map(|b| u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
        else {
            break;
        };
        let mut hpos = pos + 2 + lsot as usize;
        while let Some(marker) = read16(hpos) {
            if marker == 0xff93 {
                break;
            }
            let Some(len) = read16(hpos + 2) else { break };
            tile.push(marker);
            hpos += 2 + len as usize;
        }
        if psot == 0 {
            break;
        }
        pos = tp_start + psot as usize;
    }
    (main, tile)
}

/// Running totals of the census.
#[derive(Default)]
struct Census {
    tally: BTreeMap<String, u32>,
    examples: BTreeMap<String, String>,
    files_done: u32,
    jpx_total: u32,
    decoded: u32,
    // Per watched marker: (streams carrying it, of those how many failed,
    // first example).
    marker_streams: BTreeMap<&'static str, (u32, u32, String)>,
    no_codestream: u32,
}

/// `<file name> obj<n>`, the label an example stream is reported under.
fn stream_label(path: &str, n: u32) -> String {
    format!(
        "{} obj{n}",
        std::path::Path::new(path)
            .file_name()
            .unwrap()
            .to_string_lossy()
    )
}

/// The decoded bytes of object `n` when it is a `/JPXDecode` stream.
fn jpx_stream_data(snap: &DocumentSnapshot, n: u32) -> Option<Vec<u8>> {
    let id = ObjectId::new(n, 0);
    let mut ctx = ParseContext::new();
    let obj = snap.objects().resolve(snap, id, &mut ctx).ok()?;
    let PdfObject::Stream(stream) = obj.as_ref() else {
        return None;
    };
    // Cheap pre-filter: only peel filters for streams that name JPX,
    // so we do not decode every Flate/CCITT stream in the document.
    let is_jpx = stream.dict.iter().any(|(_, v)| {
        match v {
        PdfObject::Name(nm) => snap.names().resolve(*nm).as_ref() == b"JPXDecode",
        PdfObject::Array(items) => items.iter().any(|it| {
            matches!(it, PdfObject::Name(nm) if snap.names().resolve(*nm).as_ref() == b"JPXDecode")
        }),
        _ => false,
    }
    });
    if !is_jpx {
        return None;
    }
    let (data, codec) = snap.decode_stream_data_to_codec(stream, &mut ctx).ok()?;
    if codec.as_deref() != Some("JPXDecode") {
        return None;
    }
    Some(data)
}

impl Census {
    /// Open one PDF and census up to `per_file` of its JPX streams. Files
    /// that cannot be read or opened are skipped without counting.
    fn scan_file(&mut self, path: &str, per_file: u32) {
        let Ok(bytes) = std::fs::read(path) else {
            return;
        };
        let source = Arc::new(OwnedBytesSource::new(bytes));
        let Ok(snap) = DocumentSnapshot::open(source, DocumentLimits::default()) else {
            return;
        };
        self.files_done += 1;
        let size = snap.structure().xref.size();
        // Sample up to this many JPX streams per file — enough to catch a
        // file's codec profile without decoding every image in a 500-page book.
        let mut jpx_in_file = 0u32;
        for n in 1..size {
            if jpx_in_file >= per_file {
                break;
            }
            let Some(data) = jpx_stream_data(&snap, n) else {
                continue;
            };
            jpx_in_file += 1;
            self.record_stream(path, n, &data);
        }
        if self.files_done % 20 == 0 {
            eprintln!(
                "... {} files, {} jpx, {} ok",
                self.files_done, self.jpx_total, self.decoded
            );
        }
    }

    /// Tally one JPX stream: its watched markers and its decode outcome.
    fn record_stream(&mut self, path: &str, n: u32, data: &[u8]) {
        self.jpx_total += 1;
        let carried: Vec<&'static str> = match codestream(data) {
            Some(cs) => {
                let (main, tile) = header_markers(cs);
                WATCHED_MARKERS
                    .iter()
                    .filter(|(code, _)| main.contains(code) || tile.contains(code))
                    .map(|&(_, name)| name)
                    .collect()
            }
            None => {
                self.no_codestream += 1;
                Vec::new()
            }
        };
        let outcome = jp2lam::decode_jp2(data);
        for name in &carried {
            let entry = self
                .marker_streams
                .entry(name)
                .or_insert_with(|| (0, 0, stream_label(path, n)));
            entry.0 += 1;
            entry.1 += u32::from(outcome.is_err());
        }
        match outcome {
            Ok(_) => {
                self.decoded += 1;
                dump_decoded(path, n, data);
            }
            Err(e) => self.record_failure(path, n, data, &e.to_string()),
        }
    }

    /// Tally a failed decode under its normalized signature.
    fn record_failure(&mut self, path: &str, n: u32, data: &[u8], message: &str) {
        let sig = normalize(message);
        // Dump one representative failing stream per signature for
        // offline inspection (opj_dump).
        if !self.examples.contains_key(&sig)
            && let Ok(dir) = std::env::var("JPXDUMPDIR")
        {
            let safe: String = sig
                .chars()
                .filter(|c| c.is_ascii_alphanumeric())
                .take(40)
                .collect();
            let _ = std::fs::write(format!("{dir}/failjpx_{safe}.jp2"), data);
        }
        *self.tally.entry(sig.clone()).or_default() += 1;
        self.examples
            .entry(sig)
            .or_insert_with(|| stream_label(path, n));
    }

    /// Print the totals, the watched-marker table and the error signatures.
    fn report(&self) {
        let Census {
            tally,
            examples,
            files_done,
            jpx_total,
            decoded,
            marker_streams,
            no_codestream,
        } = self;
        println!(
            "files={files_done} jpx_streams={jpx_total} decoded={decoded} failed={}",
            jpx_total - decoded
        );
        println!("streams_without_codestream={no_codestream}");
        println!("--- watched markers (streams carrying / of those failing) ---");
        for (_, name) in WATCHED_MARKERS {
            match marker_streams.get(name) {
                Some((count, failed, example)) => {
                    println!("{name}: {count} streams, {failed} failed, e.g. {example}")
                }
                None => println!("{name}: 0 streams"),
            }
        }
        println!("--- error signatures (count) ---");
        let mut sorted: Vec<_> = tally.iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(a.1));
        for (sig, count) in sorted {
            println!("{count:>6}  {sig}");
            if let Some(ex) = examples.get(sig) {
                println!("        e.g. {ex}");
            }
        }
    }
}

/// Write a decoded stream to `JPXCENSUS_DUMP_OK`, when set, as
/// `<pdf stem>_obj<n>.jp2`.
fn dump_decoded(path: &str, n: u32, data: &[u8]) {
    let Ok(dir) = std::env::var("JPXCENSUS_DUMP_OK") else {
        return;
    };
    let stem = std::path::Path::new(path)
        .file_stem()
        .unwrap()
        .to_string_lossy()
        .replace(|c: char| !c.is_ascii_alphanumeric(), "_");
    let _ = std::fs::write(format!("{dir}/{stem}_obj{n}.jp2"), data);
}

fn main() {
    let list_path = std::env::args()
        .nth(1)
        .expect("usage: jpxcensus <list.txt> [max]");
    let max_files: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(usize::MAX);
    let list = std::fs::read_to_string(&list_path).expect("read list");

    let per_file: u32 = std::env::var("JPXCENSUS_PER_FILE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let mut census = Census::default();
    for line in list.lines().take(max_files) {
        let path = line.trim();
        if path.is_empty() {
            continue;
        }
        census.scan_file(path, per_file);
    }
    census.report();
}
