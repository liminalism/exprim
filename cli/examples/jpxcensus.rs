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

fn main() {
    let list_path = std::env::args()
        .nth(1)
        .expect("usage: jpxcensus <list.txt> [max]");
    let max_files: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(usize::MAX);
    let list = std::fs::read_to_string(&list_path).expect("read list");

    let mut tally: BTreeMap<String, u32> = BTreeMap::new();
    let mut examples: BTreeMap<String, String> = BTreeMap::new();
    let mut files_done = 0u32;
    let mut jpx_total = 0u32;
    let mut decoded = 0u32;
    let per_file: u32 = std::env::var("JPXCENSUS_PER_FILE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    // Per watched marker: (streams carrying it, of those how many failed,
    // first example).
    let mut marker_streams: BTreeMap<&str, (u32, u32, String)> = BTreeMap::new();
    let mut no_codestream = 0u32;

    for line in list.lines().take(max_files) {
        let path = line.trim();
        if path.is_empty() {
            continue;
        }
        let Ok(bytes) = std::fs::read(path) else {
            continue;
        };
        let source = Arc::new(OwnedBytesSource::new(bytes));
        let Ok(snap) = DocumentSnapshot::open(source, DocumentLimits::default()) else {
            continue;
        };
        files_done += 1;
        let size = snap.structure().xref.size();
        // Sample up to this many JPX streams per file — enough to catch a
        // file's codec profile without decoding every image in a 500-page book.
        let mut jpx_in_file = 0u32;
        for n in 1..size {
            if jpx_in_file >= per_file {
                break;
            }
            let id = ObjectId::new(n, 0);
            let mut ctx = ParseContext::new();
            let Ok(obj) = snap.objects().resolve(&snap, id, &mut ctx) else {
                continue;
            };
            let PdfObject::Stream(stream) = obj.as_ref() else {
                continue;
            };
            // Cheap pre-filter: only peel filters for streams that name JPX,
            // so we do not decode every Flate/CCITT stream in the document.
            let is_jpx = stream.dict.iter().any(|(_, v)| match v {
                PdfObject::Name(nm) => snap.names().resolve(*nm).as_ref() == b"JPXDecode",
                PdfObject::Array(items) => items.iter().any(|it| {
                    matches!(it, PdfObject::Name(nm) if snap.names().resolve(*nm).as_ref() == b"JPXDecode")
                }),
                _ => false,
            });
            if !is_jpx {
                continue;
            }
            let Ok((data, codec)) = snap.decode_stream_data_to_codec(stream, &mut ctx) else {
                continue;
            };
            if codec.as_deref() != Some("JPXDecode") {
                continue;
            }
            jpx_total += 1;
            jpx_in_file += 1;
            let carried: Vec<&str> = match codestream(&data) {
                Some(cs) => {
                    let (main, tile) = header_markers(cs);
                    WATCHED_MARKERS
                        .iter()
                        .filter(|(code, _)| main.contains(code) || tile.contains(code))
                        .map(|&(_, name)| name)
                        .collect()
                }
                None => {
                    no_codestream += 1;
                    Vec::new()
                }
            };
            let outcome = jp2lam::decode_jp2(&data);
            for name in &carried {
                let entry = marker_streams.entry(name).or_insert_with(|| {
                    (
                        0,
                        0,
                        format!(
                            "{} obj{n}",
                            std::path::Path::new(path)
                                .file_name()
                                .unwrap()
                                .to_string_lossy()
                        ),
                    )
                });
                entry.0 += 1;
                entry.1 += u32::from(outcome.is_err());
            }
            match outcome {
                Ok(_) => {
                    decoded += 1;
                    if let Ok(dir) = std::env::var("JPXCENSUS_DUMP_OK") {
                        let stem = std::path::Path::new(path)
                            .file_stem()
                            .unwrap()
                            .to_string_lossy()
                            .replace(|c: char| !c.is_ascii_alphanumeric(), "_");
                        let _ = std::fs::write(format!("{dir}/{stem}_obj{n}.jp2"), &data);
                    }
                }
                Err(e) => {
                    let sig = normalize(&e.to_string());
                    // Dump one representative failing stream per signature for
                    // offline inspection (opj_dump).
                    if !examples.contains_key(&sig)
                        && let Ok(dir) = std::env::var("JPXDUMPDIR")
                    {
                        let safe: String = sig
                            .chars()
                            .filter(|c| c.is_ascii_alphanumeric())
                            .take(40)
                            .collect();
                        let _ = std::fs::write(format!("{dir}/failjpx_{safe}.jp2"), &data);
                    }
                    *tally.entry(sig.clone()).or_default() += 1;
                    examples.entry(sig).or_insert_with(|| {
                        format!(
                            "{} obj{n}",
                            std::path::Path::new(path)
                                .file_name()
                                .unwrap()
                                .to_string_lossy()
                        )
                    });
                }
            }
        }
        if files_done % 20 == 0 {
            eprintln!("... {files_done} files, {jpx_total} jpx, {decoded} ok");
        }
    }

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
