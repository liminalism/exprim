//! The checked-in header must match what cbindgen generates from src/lib.rs.
//!
//! Set `JBIG2RS_BLESS_HEADER=1` to rewrite it after an intended ABI change.

use std::path::Path;

#[test]
fn checked_in_header_matches_the_source() {
    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let config =
        cbindgen::Config::from_file(crate_dir.join("cbindgen.toml")).expect("cbindgen.toml");
    let mut generated = Vec::new();
    cbindgen::Builder::new()
        .with_crate(crate_dir)
        .with_config(config)
        .generate()
        .expect("cbindgen generates the header")
        .write(&mut generated);
    let generated = String::from_utf8(generated).expect("header is UTF-8");

    let path = crate_dir.join("include").join("jbig2rs.h");
    if std::env::var_os("JBIG2RS_BLESS_HEADER").is_some() {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, &generated).unwrap();
        return;
    }
    let checked_in = std::fs::read_to_string(&path)
        .expect("include/jbig2rs.h exists; generate it with JBIG2RS_BLESS_HEADER=1")
        .replace("\r\n", "\n");
    assert!(
        checked_in == generated.replace("\r\n", "\n"),
        "include/jbig2rs.h is stale: rerun with JBIG2RS_BLESS_HEADER=1"
    );
}
