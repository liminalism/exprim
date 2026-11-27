//! Ignored symbol-mode timing worker over a caller-supplied PBM directory.
//!
//! PBM loading, input cloning, output hashing and optional decode
//! verification are outside the timer. Each timed unit is a fresh
//! `Jbig2Encoder`, `add_page_bitimage` for its page(s) and `flush_pdf_split`,
//! i.e. the PDF-embedding path.
//!
//! ```sh
//! SYMBOL_SPEED_DIR=/path/to/pbms cargo test --release --features parallel \
//!     --test symbol_speed -- --ignored --exact symbol_speed_worker --nocapture
//! ```
//!
//! Environment:
//! - `SYMBOL_SPEED_DIR` (required): directory of raw P4 `.pbm` pages, sorted by name.
//! - `SYMBOL_SPEED_PRESETS`: comma list of `text`, `text_lossless`,
//!   `text_lossless_max`, `text_symbol_unify`, `lossless` (default: all).
//! - `SYMBOL_SPEED_SCOPE`: `page` (one encoder per page, as the Lege PDF
//!   pipeline does; default) or `document` (one encoder for all pages).
//! - `SYMBOL_SPEED_ROUNDS`: timed rounds per preset (default 1).
//! - `SYMBOL_SPEED_LIMIT`: use only the first N pages.
//! - `SYMBOL_SPEED_VERIFY=1`: decode every output natively and count pixels
//!   that differ from the input (outside the timer).
//! - `SYMBOL_SPEED_PROFILE=/path/prefix`: pprof flamegraph per preset
//!   (requires the `profiling` feature).
//!
//! One `SYMBOL_SPEED` line per preset is printed, tab separated:
//! preset, scope, pages, rounds, seconds, seconds/page, output bytes,
//! FNV-1a 64 hash of all outputs, differing pixels (or -1), followed by stage
//! metrics of the last round in seconds.
use jbig2enc_rust::decode::{DecodeOptions, decode_embedded};
use jbig2enc_rust::jbig2enc::{Jbig2Encoder, PdfSplitOutput, SymbolModeStageMetrics};
use jbig2enc_rust::jbig2structs::Jbig2Config;
use jbig2enc_rust::jbig2sym::{BitImage, load_pbm};
use std::time::{Duration, Instant};

fn preset(name: &str) -> Jbig2Config {
    let mut cfg = match name {
        "text" => Jbig2Config::text(),
        "text_lossless" => Jbig2Config::text_lossless(),
        "text_lossless_max" => Jbig2Config::text_lossless_max_compression(),
        "text_symbol_unify" => Jbig2Config::text_symbol_unify(),
        "lossless" => Jbig2Config::lossless(),
        other => panic!("unknown preset {other}"),
    };
    // PDF embedding, as `encode_single_image_with_config` sets it.
    cfg.want_full_headers = false;
    if !cfg.symbol_mode {
        cfg.refine = false;
        cfg.text_refine = false;
    }
    cfg
}

fn fnv(hash: &mut u64, bytes: &[u8]) {
    for &byte in (bytes.len() as u64).to_le_bytes().iter().chain(bytes) {
        *hash ^= u64::from(byte);
        *hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
}

fn encode(cfg: &Jbig2Config, pages: Vec<BitImage>) -> (PdfSplitOutput, SymbolModeStageMetrics) {
    let mut encoder = Jbig2Encoder::new(cfg);
    for page in pages {
        encoder.add_page_bitimage(page).unwrap();
    }
    let output = encoder.flush_pdf_split().unwrap();
    (output, encoder.metrics_snapshot().symbol_mode)
}

fn differing_pixels(source: &BitImage, globals: Option<&[u8]>, page: &[u8]) -> u64 {
    let decoded = decode_embedded(globals, page, &DecodeOptions::default()).unwrap();
    let mut differing = 0u64;
    for y in 0..source.height {
        for x in 0..source.width {
            if source.get_usize(x, y) != decoded.get(x as u32, y as u32) {
                differing += 1;
            }
        }
    }
    differing
}

fn add_stages(total: &mut SymbolModeStageMetrics, stages: &SymbolModeStageMetrics) {
    total.cc_extraction += stages.cc_extraction;
    total.matching_dedup += stages.matching_dedup;
    total.clustering += stages.clustering;
    total.planning += stages.planning;
    total.symbol_dict_encoding += stages.symbol_dict_encoding;
    total.text_region_encoding += stages.text_region_encoding;
    total.generic_region_encoding += stages.generic_region_encoding;
}

/// Timing and output totals of one preset: the encode time over all
/// rounds, and the last round's stage metrics, output bytes, output hash
/// and differing pixels (-1 when not verified).
struct SpeedTotals {
    elapsed: Duration,
    stages: SymbolModeStageMetrics,
    bytes: usize,
    hash: u64,
    differing: i64,
}

impl SpeedTotals {
    /// Adds one unit's stage metrics and output to the last round's totals,
    /// decoding each page to count differing pixels when verifying.
    fn record_unit(
        &mut self,
        unit: &[usize],
        output: &PdfSplitOutput,
        unit_stages: &SymbolModeStageMetrics,
        images: &[BitImage],
        verify: bool,
    ) {
        add_stages(&mut self.stages, unit_stages);
        let globals = output.global_segments.as_deref();
        if let Some(globals) = globals {
            self.bytes += globals.len();
            fnv(&mut self.hash, globals);
        } else {
            fnv(&mut self.hash, &[]);
        }
        for (page, stream) in unit.iter().zip(&output.page_streams) {
            self.bytes += stream.len();
            fnv(&mut self.hash, stream);
            if verify {
                self.differing += differing_pixels(&images[*page], globals, stream) as i64;
            }
        }
        assert_eq!(output.page_streams.len(), unit.len());
    }
}

/// Encodes every unit `rounds` times with `cfg`, timing only the encodes,
/// and records the outputs of the last round.
fn time_preset(
    cfg: &Jbig2Config,
    images: &[BitImage],
    units: &[Vec<usize>],
    rounds: usize,
    verify: bool,
) -> SpeedTotals {
    let mut totals = SpeedTotals {
        elapsed: Duration::ZERO,
        stages: SymbolModeStageMetrics::default(),
        bytes: 0,
        hash: 0xcbf2_9ce4_8422_2325u64,
        differing: -1,
    };
    for round in 0..rounds {
        let last = round + 1 == rounds;
        if last {
            totals.stages = SymbolModeStageMetrics::default();
            totals.bytes = 0;
            totals.differing = if verify { 0 } else { -1 };
        }
        for unit in units {
            let input: Vec<BitImage> = unit.iter().map(|&i| images[i].clone()).collect();
            let start = Instant::now();
            let (output, unit_stages) = encode(cfg, input);
            totals.elapsed += start.elapsed();
            if !last {
                continue;
            }
            totals.record_unit(unit, &output, &unit_stages, images, verify);
        }
    }
    totals
}

#[test]
#[ignore = "symbol-mode performance worker; requires SYMBOL_SPEED_DIR"]
fn symbol_speed_worker() {
    let dir = std::env::var_os("SYMBOL_SPEED_DIR").expect("SYMBOL_SPEED_DIR");
    let mut paths: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "pbm"))
        .collect();
    paths.sort();
    if let Ok(limit) = std::env::var("SYMBOL_SPEED_LIMIT") {
        paths.truncate(limit.parse().unwrap());
    }
    assert!(!paths.is_empty(), "no .pbm files");
    let images: Vec<BitImage> = paths.iter().map(|path| load_pbm(path).unwrap()).collect();
    let presets = std::env::var("SYMBOL_SPEED_PRESETS")
        .unwrap_or("text,text_lossless,text_lossless_max,text_symbol_unify,lossless".into());
    let document = match std::env::var("SYMBOL_SPEED_SCOPE").as_deref() {
        Ok("document") => true,
        Ok("page") | Err(_) => false,
        Ok(other) => panic!("unknown scope {other}"),
    };
    let rounds: usize = std::env::var("SYMBOL_SPEED_ROUNDS")
        .unwrap_or("1".into())
        .parse()
        .unwrap();
    assert!(rounds > 0);
    let verify = std::env::var("SYMBOL_SPEED_VERIFY").is_ok_and(|v| v == "1");
    let units: Vec<Vec<usize>> = if document {
        vec![(0..images.len()).collect()]
    } else {
        (0..images.len()).map(|index| vec![index]).collect()
    };

    for name in presets.split(',').filter(|name| !name.is_empty()) {
        let cfg = preset(name);
        #[cfg(all(feature = "profiling", not(windows)))]
        let guard = std::env::var_os("SYMBOL_SPEED_PROFILE").map(|_| {
            pprof::ProfilerGuardBuilder::default()
                .frequency(1000)
                .blocklist(&["libc", "libgcc", "pthread", "vdso"])
                .build()
                .unwrap()
        });
        let SpeedTotals {
            elapsed,
            stages,
            bytes,
            hash,
            differing,
        } = time_preset(&cfg, &images, &units, rounds, verify);
        let seconds = elapsed.as_secs_f64();
        let per_page = seconds / (rounds * images.len()) as f64;
        println!(
            "SYMBOL_SPEED\t{name}\t{}\t{}\t{rounds}\t{seconds:.6}\t{per_page:.6}\t{bytes}\t{hash:016x}\t{differing}\tcc={:.4}\tmatch={:.4}\tcluster={:.4}\tplan={:.4}\tdict={:.4}\ttext={:.4}\tgeneric={:.4}",
            if document { "document" } else { "page" },
            images.len(),
            stages.cc_extraction.as_secs_f64(),
            stages.matching_dedup.as_secs_f64(),
            stages.clustering.as_secs_f64(),
            stages.planning.as_secs_f64(),
            stages.symbol_dict_encoding.as_secs_f64(),
            stages.text_region_encoding.as_secs_f64(),
            stages.generic_region_encoding.as_secs_f64(),
        );
        #[cfg(all(feature = "profiling", not(windows)))]
        if let Some(guard) = guard {
            let report = guard.report().build().unwrap();
            let prefix = std::env::var("SYMBOL_SPEED_PROFILE").unwrap();
            let path = std::path::PathBuf::from(format!("{prefix}-{name}.svg"));
            report
                .flamegraph(std::fs::File::create(&path).unwrap())
                .unwrap();
            std::fs::write(path.with_extension("txt"), format!("{report:?}")).unwrap();
        }
    }
}
