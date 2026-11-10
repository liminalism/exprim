//! Fuzz structural load: header scan, xref chains (classic, stream,
//! hybrid), trailers, incremental revisions, and every recovery path in
//! `exprim::structure::load_structure` — over an in-memory `PdfSource`.

#![no_main]

use libfuzzer_sys::fuzz_target;
use exprim::object::NameTable;
use exprim::source::OwnedBytesSource;
use exprim::structure::{StructureLimits, load_structure};

fuzz_target!(|data: &[u8]| {
    let source = OwnedBytesSource::new(data.to_vec());
    let names = NameTable::new();
    let limits = StructureLimits {
        max_revisions: 32,
        max_objects: 1 << 16,
        max_decoded_bytes: 1 << 22, // 4 MiB
        ..StructureLimits::default()
    };
    let _ = load_structure(&source, &names, &limits);
});
