//! Comprehensive test suite for comparing JBIG2 lossless vs symbol mode encoding
//!
//! This test suite provides detailed comparisons between:
//! - Lossless mode (no symbol dictionaries)
//! - Symbol mode (with symbol dictionaries)
//!
//! Including performance metrics, compression ratios, and correctness validation.

use jbig2enc_rust as jbig2;
use std::fmt;
use std::time::Instant;

// Import the common test utilities
mod common;
use common::fixtures::{test_blank_pbm, test_symbols_pbm};
use common::load_pbm;

use jbig2::jbig2sym::BitImage;
use jbig2::{Jbig2EncodeResult, encode_single_image, encode_single_image_lossless};

/// Configuration for test mode selection
#[derive(Debug, Clone, Copy)]
enum TestMode {
    LosslessOnly,
    SymbolOnly,
    Both,
}

/// Statistics for a single encoding test
#[derive(Debug, Clone)]
struct EncodingStats {
    mode_name: String,
    input_size: usize,
    output_size: usize,
    global_dict_size: Option<usize>,
    total_output_size: usize,
    compression_ratio: f64,
    encoding_time_ms: u128,
    success: bool,
    error_message: Option<String>,
}

impl EncodingStats {
    fn new(mode_name: String, input_size: usize) -> Self {
        Self {
            mode_name,
            input_size,
            output_size: 0,
            global_dict_size: None,
            total_output_size: 0,
            compression_ratio: 0.0,
            encoding_time_ms: 0,
            success: false,
            error_message: None,
        }
    }

    fn mark_success(&mut self, result: &Jbig2EncodeResult, encoding_time: u128) {
        self.output_size = result.page_data.len();
        self.global_dict_size = result.global_data.as_ref().map(|d| d.len());
        self.total_output_size = self.output_size + self.global_dict_size.unwrap_or(0);
        self.compression_ratio = self.input_size as f64 / self.total_output_size as f64;
        self.encoding_time_ms = encoding_time;
        self.success = true;
    }

    fn mark_failure(&mut self, error: &str, encoding_time: u128) {
        self.encoding_time_ms = encoding_time;
        self.success = false;
        self.error_message = Some(error.to_string());
    }
}

impl fmt::Display for EncodingStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.success {
            writeln!(f, "=== {} Results ===", self.mode_name)?;
            writeln!(f, "  Input size:         {} bytes", self.input_size)?;
            writeln!(f, "  Page data size:     {} bytes", self.output_size)?;
            if let Some(dict_size) = self.global_dict_size {
                writeln!(f, "  Global dict size:   {} bytes", dict_size)?;
            }
            writeln!(f, "  Total output size:  {} bytes", self.total_output_size)?;
            writeln!(f, "  Compression ratio:  {:.2}x", self.compression_ratio)?;
            writeln!(f, "  Encoding time:      {} ms", self.encoding_time_ms)?;
            writeln!(f, "  Status:             ✓ SUCCESS")?;
        } else {
            writeln!(f, "=== {} Results ===", self.mode_name)?;
            writeln!(f, "  Input size:         {} bytes", self.input_size)?;
            writeln!(f, "  Encoding time:      {} ms", self.encoding_time_ms)?;
            writeln!(f, "  Status:             ✗ FAILED")?;
            if let Some(error) = &self.error_message {
                writeln!(f, "  Error:              {}", error)?;
            }
        }
        Ok(())
    }
}

/// Comparison results between different modes
#[derive(Debug)]
struct ComparisonResults {
    lossless: Option<EncodingStats>,
    symbol: Option<EncodingStats>,
}

impl ComparisonResults {
    fn new() -> Self {
        Self {
            lossless: None,
            symbol: None,
        }
    }

    fn print_comparison(&self) {
        println!("\n{}", "=".repeat(80));
        println!("                    JBIG2 ENCODING MODE COMPARISON");
        println!("{}", "=".repeat(80));

        if let Some(ref lossless) = self.lossless {
            print!("{}", lossless);
        }

        if let Some(ref symbol) = self.symbol {
            print!("{}", symbol);
        }

        // Print comparison if both modes were tested
        if let (Some(lossless), Some(symbol)) = (&self.lossless, &self.symbol) {
            if lossless.success && symbol.success {
                println!("=== Comparison ===");

                let size_diff = symbol.total_output_size as i64 - lossless.total_output_size as i64;
                let size_percent = (size_diff as f64 / lossless.total_output_size as f64) * 100.0;

                let time_diff = symbol.encoding_time_ms as i64 - lossless.encoding_time_ms as i64;
                let time_percent = (time_diff as f64 / lossless.encoding_time_ms as f64) * 100.0;

                let compression_diff = symbol.compression_ratio - lossless.compression_ratio;

                println!(
                    "  Size difference:    {} bytes ({:+.1}%)",
                    size_diff, size_percent
                );
                println!(
                    "  Time difference:    {} ms ({:+.1}%)",
                    time_diff, time_percent
                );
                println!("  Compression diff:   {:+.2}x", compression_diff);

                if symbol.total_output_size < lossless.total_output_size {
                    println!("  Winner (size):      Symbol mode");
                } else if symbol.total_output_size > lossless.total_output_size {
                    println!("  Winner (size):      Lossless mode");
                } else {
                    println!("  Winner (size):      Tie");
                }

                if symbol.encoding_time_ms < lossless.encoding_time_ms {
                    println!("  Winner (speed):     Symbol mode");
                } else if symbol.encoding_time_ms > lossless.encoding_time_ms {
                    println!("  Winner (speed):     Lossless mode");
                } else {
                    println!("  Winner (speed):     Tie");
                }
            }
        }

        println!("{}", "=".repeat(80));
    }
}

/// Convert a BitImage to the binary format expected by the encode functions
fn bitimage_to_binary(img: &BitImage) -> Vec<u8> {
    let mut binary = Vec::with_capacity(img.width as usize * img.height as usize);

    for y in 0..img.height {
        for x in 0..img.width {
            binary.push(if img.get(x as u32, y as u32) { 1 } else { 0 });
        }
    }

    binary
}

/// Writes an encoded stream to `output_file` and, when that works, decodes
/// it with jbig2dec into `decoded_file`, reporting each step for `kind`
/// (lower-case and title-case names).
fn save_and_decode(kind: (&str, &str), output_file: &str, decoded_file: &str, data: &[u8]) {
    let (lower, title) = kind;
    if let Err(e) = std::fs::write(output_file, data) {
        println!("⚠️  Warning: Could not save {} output: {}", lower, e);
        return;
    }
    println!(
        "📁 Saved {} output to: {} ({} bytes)",
        lower,
        output_file,
        data.len()
    );

    // Test decoding with jbig2dec (standalone JBIG2 files)
    let mut cmd = jbig2dec_command();
    match cmd.args(["-o", decoded_file, output_file]).output() {
        Ok(output) if output.status.success() => {
            println!("✅ {} JBIG2 successfully decoded with jbig2dec", title);
        }
        Ok(output) => {
            println!(
                "❌ {} JBIG2 decode failed: {}",
                title,
                String::from_utf8_lossy(&output.stderr)
            );
        }
        Err(e) => {
            println!("⚠️  Could not run jbig2dec: {}", e);
        }
    }
}

/// DEBUG-ONLY PROBE: PDF mode fragments are NOT a valid standalone JBIG2
/// file. Prints their sizes and, when `SAVE_PDF_PROBE` is set, saves their
/// concatenation.
fn probe_pdf_fragments(input: &[u8], width: u32, height: u32) {
    let pdf_result = match encode_single_image(input, width, height, true) {
        Ok(pdf_result) => pdf_result,
        Err(e) => {
            println!("🔧 PDF mode failed: {}", e);
            return;
        }
    };
    let page_len = pdf_result.page_data.len();
    let Some(ref dict) = pdf_result.global_data else {
        return;
    };
    println!(
        "🔧 PDF mode: global_data = {} bytes, page_data = {} bytes",
        dict.len(),
        page_len
    );

    if std::env::var("SAVE_PDF_PROBE").is_err() {
        return;
    }
    let mut combined_pdf = dict.clone();
    combined_pdf.extend_from_slice(&pdf_result.page_data);
    let test_file = "test_output_pdf_combined.jb2";
    match std::fs::write(test_file, &combined_pdf) {
        Ok(_) => println!(
            "📁 Saved debug-only PDF probe: {} ({} bytes)",
            test_file,
            combined_pdf.len()
        ),
        Err(e) => println!("⚠️  Could not save PDF probe: {}", e),
    }
}

/// Test lossless mode encoding
fn test_lossless_mode(input: &[u8], width: u32, height: u32) -> EncodingStats {
    let mut stats = EncodingStats::new("Lossless Mode".to_string(), input.len());

    let start = Instant::now();

    match encode_single_image_lossless(input, width, height, false) {
        Ok(result) => {
            let elapsed = start.elapsed().as_millis();
            stats.mark_success(&result, elapsed);
            println!("✓ Lossless encoding completed in {} ms", elapsed);

            // Save the JBIG2 output to file for decoding verification
            let output_file = "test_output_lossless.jb2";
            let page_data = result.page_data.clone();
            save_and_decode(
                ("lossless", "Lossless"),
                output_file,
                "decoded_lossless.pbm",
                &page_data,
            );
        }
        Err(e) => {
            let elapsed = start.elapsed().as_millis();
            stats.mark_failure(&e.to_string(), elapsed);
            println!("✗ Lossless encoding failed: {}", e);
        }
    }

    stats
}

/// Test symbol mode encoding
fn test_symbol_mode(input: &[u8], width: u32, height: u32) -> EncodingStats {
    let mut stats = EncodingStats::new("Symbol Mode".to_string(), input.len());

    let start = Instant::now();

    // DEBUG-ONLY PROBE: PDF mode fragments are NOT a valid standalone JBIG2 file.
    // We inspect sizes; optional concatenation guarded by SAVE_PDF_PROBE env var.
    println!("🔧 DEBUG: Inspecting PDF-mode fragment sizes (not a standalone file)");
    probe_pdf_fragments(input, width, height);

    // Now test standalone mode
    match encode_single_image(input, width, height, false) {
        Ok(result) => {
            let elapsed = start.elapsed().as_millis();
            stats.mark_success(&result, elapsed);
            println!("✓ Symbol encoding completed in {} ms", elapsed);

            // Debug: Check if page_data is empty or has content
            let page_data = result.page_data.clone();
            if page_data.is_empty() {
                println!("⚠️  WARNING: Symbol mode produced empty page_data!");
            } else {
                println!("📊 Symbol mode page_data: {} bytes", page_data.len());
            }

            // Save the JBIG2 output to file for decoding verification
            let output_file = "test_output_symbol.jb2";

            // In standalone mode (pdf_mode=false), combine global and page data if present
            let complete_data = if let Some(ref dict) = result.global_data {
                println!(
                    "📖 Global data present: {} bytes (combining with page data for standalone file)",
                    dict.len()
                );
                let mut combined = dict.clone();
                combined.extend_from_slice(&page_data);
                combined
            } else {
                println!("📖 No global data - using page data only");
                page_data
            };

            save_and_decode(
                ("symbol", "Symbol"),
                output_file,
                "decoded_symbol.pbm",
                &complete_data,
            );
        }
        Err(e) => {
            let elapsed = start.elapsed().as_millis();
            stats.mark_failure(&e.to_string(), elapsed);
            println!("✗ Symbol encoding failed: {}", e);
        }
    }

    stats
}

/// Run encoding comparison test on a given image
fn run_encoding_test(img: &BitImage, test_name: &str, mode: TestMode) -> ComparisonResults {
    println!("\n{}", "=".repeat(60));
    println!("Testing: {}", test_name);
    println!(
        "Image size: {}x{} pixels ({} bytes)",
        img.width,
        img.height,
        img.width as usize * img.height as usize
    );
    println!("{}", "=".repeat(60));

    let binary_data = bitimage_to_binary(img);
    let mut results = ComparisonResults::new();

    match mode {
        TestMode::LosslessOnly => {
            results.lossless = Some(test_lossless_mode(
                &binary_data,
                img.width as u32,
                img.height as u32,
            ));
        }
        TestMode::SymbolOnly => {
            results.symbol = Some(test_symbol_mode(
                &binary_data,
                img.width as u32,
                img.height as u32,
            ));
        }
        TestMode::Both => {
            results.lossless = Some(test_lossless_mode(
                &binary_data,
                img.width as u32,
                img.height as u32,
            ));
            results.symbol = Some(test_symbol_mode(
                &binary_data,
                img.width as u32,
                img.height as u32,
            ));
        }
    }

    results.print_comparison();
    results
}

/// Create test images of various types and complexities

/// Create a simple 8x8 checkerboard pattern
fn create_small_checkerboard() -> BitImage {
    let mut img = BitImage::new(8, 8).expect("Failed to create 8x8 image");
    for y in 0..8 {
        for x in 0..8 {
            img.set(x, y, (x + y) % 2 == 0);
        }
    }
    img
}

/// Create a larger 64x64 checkerboard pattern (good for symbol detection)
fn create_large_checkerboard() -> BitImage {
    let mut img = BitImage::new(64, 64).expect("Failed to create 64x64 image");
    for y in 0..64 {
        for x in 0..64 {
            img.set(x, y, (x + y) % 2 == 0);
        }
    }
    img
}

/// Create an image with repeated text-like patterns (good for symbol dictionaries)
fn create_text_pattern() -> BitImage {
    let mut img = BitImage::new(64, 32).expect("Failed to create text pattern");

    // Create a simple "letter" pattern (5x7 pixels)
    let letter_pattern = [
        0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b00000,
    ];

    // Repeat this pattern across the image
    for repeat_x in 0..6 {
        for repeat_y in 0..4 {
            stamp_letter(&mut img, &letter_pattern, repeat_x * 10, repeat_y * 8);
        }
    }

    img
}

/// Draws the 5-pixel-wide letter rows at `(base_x, base_y)`, clipped to the
/// 64x32 image.
fn stamp_letter(img: &mut BitImage, letter_pattern: &[u32], base_x: usize, base_y: usize) {
    for (row, &pattern) in letter_pattern.iter().enumerate() {
        for bit in 0..5 {
            let x = base_x + bit;
            let y = base_y + row;
            if x < 64 && y < 32 {
                let pixel = (pattern >> (4 - bit)) & 1 == 1;
                img.set(x as u32, y as u32, pixel);
            }
        }
    }
}

/// Create a half black, half white image (minimal compression potential)
fn create_half_split() -> BitImage {
    let mut img = BitImage::new(32, 32).expect("Failed to create half split");
    for y in 0..32 {
        for x in 0..32 {
            img.set(x, y, x < 16);
        }
    }
    img
}

/// Main test functions

#[test]
fn test_small_patterns() {
    println!("\n🧪 TESTING SMALL PATTERNS");

    // Test simple small checkerboard
    let small_checker = create_small_checkerboard();
    run_encoding_test(&small_checker, "8x8 Checkerboard", TestMode::Both);

    // Test simple split pattern
    let half_split = create_half_split();
    run_encoding_test(&half_split, "32x32 Half Split", TestMode::Both);
}

#[test]
fn test_medium_patterns() {
    println!("\n🧪 TESTING MEDIUM PATTERNS");

    // Test larger checkerboard (should benefit from symbol detection)
    let large_checker = create_large_checkerboard();
    run_encoding_test(&large_checker, "64x64 Checkerboard", TestMode::Both);

    // Test text-like pattern (should really benefit from symbol dictionaries)
    let text_pattern = create_text_pattern();
    run_encoding_test(&text_pattern, "64x32 Repeated Text Pattern", TestMode::Both);
}

#[test]
fn test_real_images() {
    println!("\n🧪 TESTING REAL IMAGES");

    // Test with actual PBM test images if available
    if std::path::Path::new(test_blank_pbm()).exists() {
        let test_img = load_pbm(test_blank_pbm());
        run_encoding_test(&test_img, "Real PBM Test Image (Small)", TestMode::Both);
    } else {
        println!(
            "⚠️  Skipping real image test - {} not found",
            test_blank_pbm()
        );
    }

    if std::path::Path::new(test_symbols_pbm()).exists() {
        let test_img1 = load_pbm(test_symbols_pbm());
        run_encoding_test(&test_img1, "Real PBM Test Image (Large)", TestMode::Both);
    } else {
        println!(
            "⚠️  Skipping real image test - {} not found",
            test_symbols_pbm()
        );
    }

    // Test with the specified test image - this is the main test as requested
    let test_image2_path = "test_image2.pbm";
    if std::path::Path::new(test_image2_path).exists() {
        println!("\n📋 RUNNING REQUESTED TEST WITH SPECIFIED IMAGE");
        println!("🎯 Testing file: {}", test_image2_path);
        let test_img2 = load_pbm(test_image2_path);
        run_encoding_test(
            &test_img2,
            "Test Image 2 (Specified by User)",
            TestMode::Both,
        );
    } else {
        println!(
            "❌ ERROR: Requested test image not found - {}",
            test_image2_path
        );
        println!("   Please ensure the test image exists in the current directory");
    }
}

/// Prints both modes' sizes, ratios and times and which is smaller and
/// faster, or which mode failed.
fn print_specified_summary(results: &ComparisonResults) {
    let (Some(lossless), Some(symbol)) = (&results.lossless, &results.symbol) else {
        return;
    };
    if !(lossless.success && symbol.success) {
        println!("❌ One or both encoding modes failed!");
        if !lossless.success {
            println!("   Lossless error: {:?}", lossless.error_message);
        }
        if !symbol.success {
            println!("   Symbol error: {:?}", symbol.error_message);
        }
        return;
    }
    println!("✅ Both encoding modes completed successfully!");
    println!();
    println!("📈 KEY METRICS:");
    println!(
        "   Lossless mode:  {} bytes, {:.2}x compression, {} ms",
        lossless.total_output_size, lossless.compression_ratio, lossless.encoding_time_ms
    );
    println!(
        "   Symbol mode:    {} bytes, {:.2}x compression, {} ms",
        symbol.total_output_size, symbol.compression_ratio, symbol.encoding_time_ms
    );

    let size_savings = lossless.total_output_size as i64 - symbol.total_output_size as i64;
    let size_percent = (size_savings as f64 / lossless.total_output_size as f64) * 100.0;

    println!();
    if size_savings > 0 {
        println!(
            "🏆 Symbol mode is BETTER: {} bytes smaller ({:.1}% reduction)",
            size_savings, size_percent
        );
    } else if size_savings < 0 {
        println!(
            "🏆 Lossless mode is BETTER: {} bytes smaller ({:.1}% reduction)",
            -size_savings, -size_percent
        );
    } else {
        println!("🤝 Both modes produced identical output sizes");
    }

    if symbol.encoding_time_ms < lossless.encoding_time_ms {
        println!(
            "⚡ Symbol mode is FASTER by {} ms",
            lossless.encoding_time_ms - symbol.encoding_time_ms
        );
    } else if symbol.encoding_time_ms > lossless.encoding_time_ms {
        println!(
            "⚡ Lossless mode is FASTER by {} ms",
            symbol.encoding_time_ms - lossless.encoding_time_ms
        );
    } else {
        println!("⚡ Both modes have identical encoding times");
    }
}

/// Lists the PBM files in the current directory.
fn list_pbm_files() {
    let Ok(entries) = std::fs::read_dir(".") else {
        return;
    };
    for entry in entries.flatten() {
        if entry.path().extension().is_some_and(|ext| ext == "pbm") {
            println!("     Found: {}", entry.path().display());
        }
    }
}

/// Prints which mode produced the smaller output and which was faster.
fn print_winners(results: &ComparisonResults) {
    let (Some(lossless), Some(symbol)) = (&results.lossless, &results.symbol) else {
        return;
    };
    if !(lossless.success && symbol.success) {
        println!("  Status: One or both modes failed");
        return;
    }
    let size_winner = if symbol.total_output_size < lossless.total_output_size {
        "Symbol"
    } else if symbol.total_output_size > lossless.total_output_size {
        "Lossless"
    } else {
        "Tie"
    };

    let speed_winner = if symbol.encoding_time_ms < lossless.encoding_time_ms {
        "Symbol"
    } else if symbol.encoding_time_ms > lossless.encoding_time_ms {
        "Lossless"
    } else {
        "Tie"
    };

    println!(
        "  Size winner:  {} ({} vs {} bytes)",
        size_winner, symbol.total_output_size, lossless.total_output_size
    );
    println!(
        "  Speed winner: {} ({} vs {} ms)",
        speed_winner, symbol.encoding_time_ms, lossless.encoding_time_ms
    );
}

#[test]
fn test_specified_image_only() {
    println!("\n🎯 FOCUSED TEST: USER-SPECIFIED IMAGE ONLY");
    println!("==================================================");

    let test_image2_path = "test_image2.pbm";
    if std::path::Path::new(test_image2_path).exists() {
        println!("📁 Loading test image: {}", test_image2_path);
        let test_img2 = load_pbm(test_image2_path);
        println!(
            "✅ Image loaded successfully: {}x{} pixels",
            test_img2.width, test_img2.height
        );

        println!("\n🔄 Running comprehensive comparison...");
        let results = run_encoding_test(&test_img2, "User Specified Test Image", TestMode::Both);

        println!("\n📊 FINAL SUMMARY FOR SPECIFIED IMAGE");
        println!("=====================================");

        print_specified_summary(&results);
    } else {
        println!(
            "❌ ERROR: Specified test image not found - {}",
            test_image2_path
        );
        println!("   Current directory contents:");
        list_pbm_files();
    }
}

#[test]
fn test_lossless_only() {
    println!("\n🧪 TESTING LOSSLESS MODE ONLY");

    let patterns = [
        ("8x8 Checkerboard", create_small_checkerboard()),
        ("64x64 Checkerboard", create_large_checkerboard()),
        ("Text Pattern", create_text_pattern()),
        ("Half Split", create_half_split()),
    ];

    for (name, img) in patterns.iter() {
        run_encoding_test(img, name, TestMode::LosslessOnly);
    }
}

#[test]
fn test_symbol_only() {
    println!("\n🧪 TESTING SYMBOL MODE ONLY");

    let patterns = [
        ("8x8 Checkerboard", create_small_checkerboard()),
        ("64x64 Checkerboard", create_large_checkerboard()),
        ("Text Pattern", create_text_pattern()),
        ("Half Split", create_half_split()),
    ];

    for (name, img) in patterns.iter() {
        run_encoding_test(img, name, TestMode::SymbolOnly);
    }
}

#[test]
fn test_comprehensive_comparison() {
    println!("\n🧪 COMPREHENSIVE MODE COMPARISON");
    println!("This test compares both encoding modes across multiple pattern types");

    let test_cases = [
        ("Small Checkerboard (8x8)", create_small_checkerboard()),
        ("Large Checkerboard (64x64)", create_large_checkerboard()),
        ("Repeated Text Pattern", create_text_pattern()),
        ("Simple Half Split", create_half_split()),
    ];

    let mut all_results = Vec::new();

    for (name, img) in test_cases.iter() {
        let results = run_encoding_test(img, name, TestMode::Both);
        all_results.push((name.to_string(), results));
    }

    // Print summary
    println!("\n{}", "=".repeat(80));
    println!("                         SUMMARY OF ALL TESTS");
    println!("{}", "=".repeat(80));

    for (name, results) in all_results.iter() {
        println!("\n{}", name);
        print_winners(results);
    }

    println!("{}", "=".repeat(80));
}

// Helper to resolve a local jbig2dec binary when running tests
fn jbig2dec_command() -> std::process::Command {
    #[cfg(windows)]
    {
        let local = std::path::Path::new(".\\jbig2dec.exe");
        if local.exists() {
            return std::process::Command::new(local);
        }
        std::process::Command::new("jbig2dec.exe")
    }
    #[cfg(not(windows))]
    {
        let local = std::path::Path::new("./jbig2dec");
        if local.exists() {
            return std::process::Command::new(local);
        }
        std::process::Command::new("jbig2dec")
    }
}
