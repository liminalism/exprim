//! This module contains the main JBIG2 encoder logic.
//!
//! (Was `encode/jbig2enc/mod.rs`; renamed to `encode/document.rs` in the
//! Phase 0 restructure. Its `symbol` subdirectory stays a child module at
//! `encode/document/symbol/` — see Gap A.)
mod grouped;
mod resources;
mod symbol;
pub use grouped::{EncodedPdfGroup, PdfGroupEncoder, PdfGroupOptions};
pub use resources::{
    DecoderResourceLimits, DocumentResourceMetrics, PageResourceMetrics, ResourceLimitExceeded,
};

#[cfg(feature = "symboldict")]
use symbol::types::*;

// Re-exports preserving the original `crate::jbig2enc::*` public paths after
// the symbol-dictionary/text-region machinery moved into `symbol::*` submodules.
pub use symbol::dictionary::{
    TextRegionSymbolInstance, build_dictionary_and_get_instances, canonicalize_dict_symbols,
    encode_symbol_dict, encode_symbol_dict_with_order,
};
pub use symbol::extraction::segment_symbols;
pub use symbol::text_region::{
    encode_page_with_symbol_dictionary, encode_text_region, encode_text_region_mapped,
};
pub use symbol::text_region_refine::encode_text_region_with_refinement;
pub use symbol::types::SymbolInstance;

use crate::jbig2arith::Jbig2ArithCoder;
use crate::jbig2classify::SymbolSignature;
#[cfg(feature = "symboldict")]
use crate::jbig2classify::{
    FamilyBucketKey, family_bucket_key_for_symbol, family_bucket_neighbors,
};
#[cfg(feature = "symboldict")]
use crate::jbig2comparator::Comparator;
// Symbol extraction using CC analysis
#[cfg(feature = "symboldict")]
use crate::jbig2cc::analyze_page;
use crate::jbig2structs::{
    FileHeader, GenericRegionParams, Jbig2Config, LossySymbolMode, PageInfo, Segment, SegmentType,
};

use crate::jbig2sym::BitImage;
#[cfg(feature = "symboldict")]
use crate::jbig2sym::Rect;
use anyhow::{Result, anyhow};

// Define debug and trace macros at the crate root
#[macro_export]
macro_rules! debug {
    ($($arg:tt)*) => {
        #[cfg(feature = "trace_encoder")]
        log::debug!($($arg)*);

        #[cfg(not(feature = "trace_encoder"))]
        let _ = format_args!($($arg)*);
    };
}

#[macro_export]
macro_rules! trace {
    ($($arg:tt)*) => {
        #[cfg(feature = "trace_encoder")]
        log::trace!($($arg)*);

        #[cfg(not(feature = "trace_encoder"))]
        let _ = format_args!($($arg)*);
    };
}

use ndarray::Array2;
use rustc_hash::FxHashMap;
#[cfg(feature = "symboldict")]
use rustc_hash::FxHashSet;
use std::hash::{Hash, Hasher};
use std::time::Duration;
#[cfg(feature = "symboldict")]
use std::time::Instant;

/// A key type for hashing bitmaps efficiently
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HashKey(u64);

impl Hash for HashKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl std::fmt::Display for HashKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "HashKey({:x})", self.0)
    }
}

#[cfg(feature = "symboldict")]
const RECENT_SYMBOL_CACHE_CAP: usize = 64;
#[cfg(feature = "symboldict")]
const SYM_UNIFY_EXACT_ANCHOR_BUDGET: usize = 32;
#[cfg(feature = "symboldict")]
const SYM_UNIFY_NEIGHBOR_ANCHOR_BUDGET: usize = 16;
const SYM_UNIFY_STRONG_ANCHOR_MIN_USAGE: usize = 8;
const SYM_UNIFY_STRONG_ANCHOR_MIN_PAGE_SPAN: usize = 4;

// Jbig2EncConfig has been removed. Use jbig2structs::Jbig2Config directly.

#[derive(Clone)]
pub struct PageData {
    pub image: BitImage,
    pub symbol_instances: Vec<SymbolInstance>,
}

#[derive(Debug, Clone, Default)]
pub struct SymbolModeStageMetrics {
    pub cc_extraction: Duration,
    pub matching_dedup: Duration,
    pub clustering: Duration,
    pub planning: Duration,
    pub symbol_dict_encoding: Duration,
    pub text_region_encoding: Duration,
    pub generic_region_encoding: Duration,
}

#[derive(Debug, Clone, Default)]
pub struct SymbolModeStats {
    /// Connected components before the legacy symbol-selection filters.
    pub input_components: usize,
    pub symbols_discovered: usize,
    pub symbols_exported: usize,
    pub avg_symbol_reuse: f64,
    pub global_symbol_count: usize,
    pub local_symbol_count: usize,
    pub comparator_calls: usize,
    pub comparator_hits: usize,
    pub exact_hits: usize,
    pub refined_hits: usize,
    pub signature_rejects: usize,
    pub topology_refinements: usize,
}

#[derive(Debug, Clone, Default)]
pub struct EncoderMetrics {
    pub symbol_mode: SymbolModeStageMetrics,
    pub symbol_stats: SymbolModeStats,
    pub optimization: SymbolOptimizationMetrics,
    pub resources: DocumentResourceMetrics,
}

#[derive(Debug, Clone)]
pub struct SymbolCostTrial {
    pub symbol: Option<usize>,
    pub representation: &'static str,
    pub policy: crate::jbig2structs::PdfDictionaryPolicy,
    pub estimated_gain_bytes: Option<i64>,
    /// Uncorrected shortlist estimate, retained for calibration diagnostics.
    pub raw_estimated_gain_bytes: Option<i64>,
    pub measured_gain_bytes: Option<i64>,
    pub encoded_bytes: usize,
    pub meets_resource_limits: bool,
}

#[derive(Debug, Clone, Default)]
pub struct SymbolOptimizationMetrics {
    pub trials: Vec<SymbolCostTrial>,
    pub candidate_comparisons: usize,
    pub baseline_bytes: usize,
    pub generic_bytes: usize,
    pub selected_bytes: usize,
    pub exact_instances: usize,
    pub refined_instances: usize,
    pub generic_instances: usize,
    pub substituted_instances: usize,
    pub calibration_samples: usize,
}

#[derive(Debug, Clone)]
pub struct PdfSplitOutput {
    pub global_segments: Option<Vec<u8>>,
    pub page_streams: Vec<Vec<u8>>,
    pub local_dict_bytes_per_page: Vec<usize>,
    pub text_region_bytes_per_page: Vec<usize>,
    pub generic_region_bytes_per_page: Vec<usize>,
}

/// Mutable state for the encoder that can change during encoding.
#[derive(Debug, Default)]
struct EncoderState {
    pdf_mode: bool,
    full_headers_remaining: bool,
    segment: bool,
    lossy_symbol_mode_applied: bool,
    ingest_debug_lines: Vec<String>,
    decision_debug_lines: Vec<String>,
}

/// Main JBIG2 encoder that handles document encoding
///
/// This struct manages the encoding state and configuration for JBIG2 documents.
/// It supports both symbol-based and generic region encoding strategies.
pub struct Jbig2Encoder<'a> {
    /// Configuration for the encoder. Borrowed by [`Jbig2Encoder::new`], owned
    /// by [`Jbig2Encoder::owned`] so an encoder can outlive its caller's config
    /// (an opaque FFI handle, a long-lived document encoder).
    config: std::borrow::Cow<'a, Jbig2Config>,

    /// Internal encoder state
    state: EncoderState,

    /// Global symbols (shared across pages)
    global_symbols: Vec<BitImage>,

    /// Usage count for each global symbol
    symbol_usage: Vec<usize>,

    /// Black pixel count cache for each global symbol (for fast pre-filtering)
    symbol_pixel_counts: Vec<usize>,

    /// Cheap structural signatures used to reject bad matches before full comparison
    symbol_signatures: Vec<SymbolSignature>,

    /// Number of distinct pages where each symbol appears
    symbol_page_count: Vec<usize>,

    /// Last page where the symbol was seen, used to deduplicate per-page membership updates
    symbol_last_page_seen: Vec<Option<usize>>,

    /// Hash map for quick symbol lookup
    hash_map: FxHashMap<HashKey, Vec<usize>>,

    /// Page data for each page in the document
    pages: Vec<PageData>,

    /// Per-page unique symbol indices, built incrementally during extraction
    page_symbol_indices: Vec<Vec<usize>>,

    /// Next available segment number
    next_segment_number: u32,

    /// Segment numbers of the global dictionary segments, in text-region symbol-ID order.
    global_dict_segment_numbers: Vec<u32>,

    /// Encoder metrics used by the benchmark harness
    metrics: EncoderMetrics,

    /// Sub-encodes shared between measured-planner trials; `None` otherwise.
    trial_memo: Option<std::sync::Mutex<symbol::memo::TrialMemo>>,
}

/// Per-page state of symbol ingest: matching scratch, debug switches and
/// log, the component count, the unification anchor map and counters.
#[cfg(feature = "symboldict")]
struct PageIngest {
    page_num: usize,
    comparator: Comparator,
    debug_matching: bool,
    no_reuse: bool,
    debug_lines: Vec<String>,
    cc_index: usize,
    sym_unify_anchor_map: Option<FxHashMap<FamilyBucketKey, Vec<usize>>>,
    initial_anchor_count: usize,
    initial_anchor_bytes: usize,
    counters: SymUnifyIngestCounters,
}

/// How a page's components were coded under symbol unification, and why
/// anchors rejected them.
#[cfg(feature = "symboldict")]
#[derive(Default)]
struct SymUnifyIngestCounters {
    recent_hits: usize,
    anchor_hits: usize,
    bucket_hits: usize,
    new_symbols: usize,
    anchor_score_rejects: usize,
    anchor_outside_rejects: usize,
    anchor_compare_rejects: usize,
    anchor_overlap_rejects: usize,
}

/// The symbols used on the current text line, most recent first, with a
/// scratch copy to iterate while matching and the line's last y.
#[cfg(feature = "symboldict")]
struct RecentSymbols {
    cache: RecentSymbolCache,
    candidates: [usize; RECENT_SYMBOL_CACHE_CAP],
    last_y: u32,
}

/// One connected component prepared for matching: its bitmap, trimmed
/// bitmap and offset, signature, ink, page position and match tolerance.
#[cfg(feature = "symboldict")]
struct PageComponent {
    symbol: BitImage,
    trimmed: BitImage,
    trim_offset: Rect,
    signature: SymbolSignature,
    pixel_count: usize,
    rect: Rect,
    max_err: u32,
}

/// Which search found a component's match.
#[cfg(feature = "symboldict")]
#[derive(Clone, Copy, PartialEq, Eq)]
enum MatchSource {
    Recent,
    Anchor,
    Bucket,
}

/// A known symbol a component is coded as: the compare error (the
/// assignment score for an anchor), the offset and the refinement need.
#[cfg(feature = "symboldict")]
struct ComponentMatch {
    symbol_index: usize,
    source: MatchSource,
    err_or_score: u32,
    dx: i32,
    dy: i32,
    needs_refinement: bool,
}

/// Writes the page 0 matching log to `jbig2_debug_page0.log`, ignoring
/// I/O failures.
#[cfg(feature = "symboldict")]
fn write_matching_log(debug_lines: &[String]) {
    let log_path = std::path::Path::new("jbig2_debug_page0.log");
    let Ok(mut f) = std::fs::File::create(log_path) else {
        return;
    };
    use std::io::Write;
    for line in debug_lines {
        let _ = writeln!(f, "{}", line);
    }
}

/// The `hash_key` dimension buckets within `dim_range` of `(h, w)`,
/// height-major; offsets that wrap below zero land far out of range and are
/// skipped.
pub(crate) fn dimension_bucket_keys(
    h: u64,
    w: u64,
    dim_range: u64,
) -> impl Iterator<Item = HashKey> {
    (0..=(dim_range * 2))
        .map(move |dh_off| h.wrapping_add(dh_off).wrapping_sub(dim_range))
        .filter(|&dh| dh < 10_000)
        .flat_map(move |dh| {
            (0..=(dim_range * 2))
                .map(move |dw_off| w.wrapping_add(dw_off).wrapping_sub(dim_range))
                .filter(|&dw| dw < 10_000)
                .map(move |dw| HashKey(dh * 10_000 + dw))
        })
}

impl<'a> Jbig2Encoder<'a> {
    /// Creates a new JBIG2 encoder with the specified configuration
    ///
    /// # Arguments
    /// * `config` - Configuration for the encoder
    pub fn new(config: &'a Jbig2Config) -> Self {
        Self::with_config(std::borrow::Cow::Borrowed(config))
    }

    /// Creates an encoder that owns its configuration, and so borrows nothing.
    pub fn owned(config: Jbig2Config) -> Jbig2Encoder<'static> {
        Jbig2Encoder::with_config(std::borrow::Cow::Owned(config))
    }

    pub(crate) fn with_config(config: std::borrow::Cow<'a, Jbig2Config>) -> Self {
        let full_headers_remaining = config.want_full_headers;
        Self {
            config,
            state: EncoderState {
                pdf_mode: false, // start in raw mode
                full_headers_remaining,
                segment: true, // Default to using segments
                lossy_symbol_mode_applied: false,
                ingest_debug_lines: Vec::new(),
                decision_debug_lines: Vec::new(),
            },
            global_symbols: Vec::new(),
            symbol_usage: Vec::new(),
            symbol_pixel_counts: Vec::new(),
            symbol_signatures: Vec::new(),
            symbol_page_count: Vec::new(),
            symbol_last_page_seen: Vec::new(),
            hash_map: FxHashMap::default(),
            pages: Vec::new(),
            page_symbol_indices: Vec::new(),
            next_segment_number: 1,
            global_dict_segment_numbers: Vec::new(),
            metrics: EncoderMetrics::default(),
            trial_memo: None,
        }
    }

    /// Fallible constructor for callers accepting external configuration.
    pub fn try_new(config: &'a Jbig2Config) -> Result<Self> {
        config
            .validate_exact_symbol_mode()
            .map_err(|error| anyhow!(error))?;
        if config.refine && !config.symbol_mode {
            return Err(anyhow!("refinement requires symbol mode to be enabled"));
        }
        Ok(Self::new(config))
    }

    pub fn dict_only(mut self) -> Self {
        self.state.full_headers_remaining = false;
        self.state.pdf_mode = true;
        self
    }

    /// Returns the number of pages currently added to the encoder
    pub fn get_page_count(&self) -> usize {
        self.pages.len()
    }

    pub fn metrics_snapshot(&self) -> EncoderMetrics {
        self.metrics.clone()
    }

    pub fn decision_debug_log(&self) -> String {
        if self.state.ingest_debug_lines.is_empty() {
            return self.state.decision_debug_lines.join("\n");
        }
        if self.state.decision_debug_lines.is_empty() {
            return self.state.ingest_debug_lines.join("\n");
        }

        let mut out = String::new();
        out.push_str(&self.state.ingest_debug_lines.join("\n"));
        out.push('\n');
        out.push_str(&self.state.decision_debug_lines.join("\n"));
        out
    }

    /// Returns debug information about symbol usage
    pub fn get_symbol_stats(&self) -> String {
        let total_symbols = self.global_symbols.len();
        let avg_usage = if total_symbols > 0 {
            self.symbol_usage.iter().sum::<usize>() as f32 / total_symbols as f32
        } else {
            0.0
        };
        let low_usage_count = self.symbol_usage.iter().filter(|&&u| u < 2).count();

        format!(
            "Total symbols: {}, Average usage: {:.1}, Low usage (<2): {}",
            total_symbols, avg_usage, low_usage_count
        )
    }

    pub fn add_page(&mut self, image: &Array2<u8>) -> Result<()> {
        let bitimage = crate::jbig2sym::array_to_bitimage(image).map_err(|error| anyhow!(error))?;
        self.add_page_bitimage(bitimage)
    }

    // `add_page_bitimage` must stay in `mod.rs` as a `pub fn`: external
    // integration test crates call it directly on `Jbig2Encoder`, and
    // `mod symbol` is a private module, so relocating it there would make
    // it externally unreachable.
    pub fn add_page_bitimage(&mut self, bitimage: BitImage) -> Result<()> {
        self.config
            .validate_exact_symbol_mode()
            .map_err(|error| anyhow!(error))?;
        if self.config.refine && !self.config.symbol_mode {
            return Err(anyhow!("refinement requires symbol mode to be enabled"));
        }
        #[cfg(feature = "symboldict")]
        if self.config.symbol_mode
            && (self.config.is_lossless || self.config.optimize_lossy_symbols)
        {
            return self.add_page_exact_symbols(bitimage);
        }
        self.page_symbol_indices.push(Vec::new());
        // Extract symbols if symbol mode is enabled
        let extract_symbols = self.config.symbol_mode && self.state.segment;
        #[cfg(feature = "symboldict")]
        let symbol_instances = self.ingest_page_symbols(&bitimage, extract_symbols);
        // Without `symboldict` there is no symbol extraction to enable.
        #[cfg(not(feature = "symboldict"))]
        let _ = extract_symbols;
        #[cfg(not(feature = "symboldict"))]
        let symbol_instances = Vec::new();

        self.pages.push(PageData {
            image: bitimage,
            symbol_instances,
        });
        Ok(())
    }

    /// Extracts the page's symbols when `extract_symbols`, matching each
    /// against the symbols seen so far, writes the page 0 matching log and the
    /// unification ingest diagnostics, and returns the page's symbol instances.
    #[cfg(feature = "symboldict")]
    fn ingest_page_symbols(
        &mut self,
        bitimage: &BitImage,
        extract_symbols: bool,
    ) -> Vec<SymbolInstance> {
        let page_num = self.pages.len();
        let mut symbol_instances = Vec::new();
        let mut ingest = self.begin_page_ingest(page_num, bitimage);

        if extract_symbols {
            self.match_page_components(bitimage, &mut ingest, &mut symbol_instances);
        }

        // Write page 0 matching debug log
        if ingest.debug_matching && !ingest.debug_lines.is_empty() {
            ingest.debug_lines.push(format!(
                "\nTotal CCs: {}, Instances: {}",
                ingest.cc_index,
                symbol_instances.len()
            ));
            write_matching_log(&ingest.debug_lines);
        }

        if self.config.lossy_symbol_mode == LossySymbolMode::SymbolUnify
            && encoder_diagnostics_enabled()
        {
            self.record_ingest_diagnostics(&ingest);
        }
        symbol_instances
    }

    /// Sets up a page's ingest: the debug switches, the matching log header
    /// and, under unification, the anchor map of the symbols seen so far.
    #[cfg(feature = "symboldict")]
    fn begin_page_ingest(&self, page_num: usize, bitimage: &BitImage) -> PageIngest {
        let debug_matching =
            page_num == 0 && std::env::var("JBIG2_DEBUG").map_or(false, |v| v == "1");
        let no_reuse = std::env::var("JBIG2_NO_REUSE").map_or(false, |v| v == "1");

        let mut debug_lines: Vec<String> = Vec::new();
        if debug_matching {
            debug_lines.push("=== PAGE 0 MATCHING LOG ===".to_string());
            debug_lines.push(format!("Image: {}x{}", bitimage.width, bitimage.height));
        }
        let sym_unify_anchor_map = (self.config.lossy_symbol_mode == LossySymbolMode::SymbolUnify
            && !self.global_symbols.is_empty())
        .then(|| self.build_sym_unify_anchor_map(page_num));
        let initial_anchor_count = sym_unify_anchor_map
            .as_ref()
            .map(|anchors| anchors.values().map(Vec::len).sum::<usize>())
            .unwrap_or(0);
        let initial_anchor_bytes = sym_unify_anchor_map
            .as_ref()
            .map(|anchors| anchor_map_dictionary_bytes(&self.global_symbols, anchors))
            .unwrap_or(0);
        PageIngest {
            page_num,
            comparator: Comparator::default(),
            debug_matching,
            no_reuse,
            debug_lines,
            cc_index: 0,
            sym_unify_anchor_map,
            initial_anchor_count,
            initial_anchor_bytes,
            counters: SymUnifyIngestCounters::default(),
        }
    }

    /// Runs connected-component analysis on the page and, unless the page
    /// is one solid component, ingests every component as a symbol.
    #[cfg(feature = "symboldict")]
    fn match_page_components(
        &mut self,
        bitimage: &BitImage,
        ingest: &mut PageIngest,
        symbol_instances: &mut Vec<SymbolInstance>,
    ) {
        let dpi = 300; // Default DPI
        let losslevel = if self.config.symbol_mode || self.config.uses_lossy_symbol_dictionary() {
            0
        } else if self.config.is_lossless {
            0
        } else {
            1
        };
        let cc_start = Instant::now();
        let cc_image = analyze_page(bitimage, dpi, losslevel);
        let extracted = cc_image.extract_shape_refs();
        self.metrics.symbol_stats.input_components += extracted.len();
        self.metrics.symbol_mode.cc_extraction += cc_start.elapsed();

        // Check if symbol extraction makes sense for this image
        // If we only get one symbol that covers the entire image,
        // it's better to use generic region encoding
        let should_use_symbols = if extracted.len() == 1 {
            let bbox = extracted[0].bbox;
            !(bbox.xmin == 0
                && bbox.ymin == 0
                && bbox.width() as usize >= bitimage.width.saturating_sub(2)
                && bbox.height() as usize >= bitimage.height.saturating_sub(2))
        } else {
            !extracted.is_empty()
        };
        if !should_use_symbols {
            return;
        }

        let matching_start = Instant::now();
        let mut recent = RecentSymbols {
            cache: RecentSymbolCache::new(RECENT_SYMBOL_CACHE_CAP),
            candidates: [0usize; RECENT_SYMBOL_CACHE_CAP],
            last_y: 0,
        };
        for shape in extracted {
            let Some(component) = self.page_component(&cc_image, &shape, &mut recent) else {
                continue;
            };
            match self.find_component_match(&component, ingest, &mut recent) {
                Some(found) => self.record_component_match(
                    found,
                    component,
                    ingest,
                    &mut recent,
                    symbol_instances,
                ),
                None => self.record_new_symbol(component, ingest, &mut recent, symbol_instances),
            }
            ingest.cc_index += 1;
        }
        self.metrics.symbol_mode.matching_dedup += matching_start.elapsed();
    }

    /// Cuts one connected component out of the page, trimmed and placed,
    /// with its signature and match tolerance; `None` for a component too
    /// small or too thin to be a symbol. A component far below the previous
    /// one starts a new line and clears the recent-symbol cache.
    #[cfg(feature = "symboldict")]
    fn page_component(
        &self,
        cc_image: &crate::jbig2cc::CCImage,
        shape: &crate::jbig2cc::ShapeRef,
        recent: &mut RecentSymbols,
    ) -> Option<PageComponent> {
        if Self::should_skip_symbol_candidate(
            shape.bbox.width().max(0) as usize,
            shape.bbox.height().max(0) as usize,
            shape.black_pixels,
        ) || shape.run_count == 0
        {
            return None;
        }
        let symbol = cc_image.get_bitmap_for_cc(shape.ccid)?;
        let (trim_offset, trimmed) = symbol.trim();
        let pixel_count = trimmed.count_ones();
        if Self::should_skip_symbol_candidate(trimmed.width, trimmed.height, pixel_count) {
            return None;
        }

        // The CC bbox is the bounding box from CC analysis.
        // trim() may remove whitespace rows/cols from the symbol.
        // Adjust position by trim offset so the dictionary bitmap
        // renders at the correct location on the page.
        let rect = Rect {
            x: shape.bbox.xmin as u32 + trim_offset.x,
            y: shape.bbox.ymin as u32 + trim_offset.y,
            width: trimmed.width as u32,
            height: trimmed.height as u32,
        };
        if rect.y > recent.last_y.saturating_add(24) {
            recent.cache.clear();
        }
        recent.last_y = rect.y;

        // Only unification's family test reads the hole count;
        // a new symbol gets its complete signature below.
        let signature = if self.config.lossy_symbol_mode == LossySymbolMode::SymbolUnify {
            Self::compute_symbol_signature(&trimmed)
        } else {
            crate::jbig2classify::compute_symbol_signature_without_holes(&trimmed)
        };

        // Error tolerance for matching.
        let area = (trimmed.width * trimmed.height) as u32;
        let max_err = if self.config.text_refine {
            (area / self.config.match_tolerance).max(3)
        } else {
            ((area as f32 * 0.05) as u32).max(2)
        };
        Some(PageComponent {
            symbol,
            trimmed,
            trim_offset,
            signature,
            pixel_count,
            rect,
            max_err,
        })
    }

    /// Looks for a known symbol to code the component as: first among the
    /// recently used symbols, then among the unification anchors, then in
    /// the dimension buckets of every symbol.
    #[cfg(feature = "symboldict")]
    fn find_component_match(
        &mut self,
        component: &PageComponent,
        ingest: &mut PageIngest,
        recent: &mut RecentSymbols,
    ) -> Option<ComponentMatch> {
        if ingest.no_reuse {
            return None;
        }
        let recent_len = recent.cache.copy_into(&mut recent.candidates);
        for &idx in &recent.candidates[..recent_len] {
            if let Some(found) =
                self.symbol_match(component, idx, MatchSource::Recent, &mut ingest.comparator)
            {
                return Some(found);
            }
        }

        if self.config.lossy_symbol_mode == LossySymbolMode::SymbolUnify
            && let Some(anchor_map) = ingest.sym_unify_anchor_map.as_ref()
            && let Some(found) = self.find_anchor_match(
                component,
                anchor_map,
                &mut ingest.comparator,
                &mut ingest.counters,
            )
        {
            return Some(found);
        }

        let h = component.trimmed.height as u64;
        let w = component.trimmed.width as u64;
        let dim_range: u64 = if self.config.text_refine { 2 } else { 0 };
        for nk in dimension_bucket_keys(h, w, dim_range) {
            let Some(bucket) = self.hash_map.get(&nk) else {
                continue;
            };
            let bucket_len = bucket.len();
            let bucket_ptr = bucket.as_ptr();
            for bucket_pos in 0..bucket_len {
                let idx = unsafe { *bucket_ptr.add(bucket_pos) };
                if let Some(found) =
                    self.symbol_match(component, idx, MatchSource::Bucket, &mut ingest.comparator)
                {
                    return Some(found);
                }
            }
        }
        None
    }

    /// Evaluates one known symbol as a match for the component.
    #[cfg(feature = "symboldict")]
    #[inline(always)]
    fn symbol_match(
        &mut self,
        component: &PageComponent,
        idx: usize,
        source: MatchSource,
        comparator: &mut Comparator,
    ) -> Option<ComponentMatch> {
        let (err, dx, dy, needs_refinement) = self.evaluate_symbol_match(
            &component.trimmed,
            component.signature,
            component.pixel_count,
            idx,
            comparator,
            component.max_err,
        )?;
        Some(ComponentMatch {
            symbol_index: idx,
            source,
            err_or_score: err,
            dx,
            dy,
            needs_refinement,
        })
    }

    /// Searches the unification anchors in the component's family bucket,
    /// then in its neighbouring buckets, each within its budget, counting
    /// the rejects by reason.
    #[cfg(feature = "symboldict")]
    fn find_anchor_match(
        &mut self,
        component: &PageComponent,
        anchor_map: &FxHashMap<FamilyBucketKey, Vec<usize>>,
        comparator: &mut Comparator,
        counters: &mut SymUnifyIngestCounters,
    ) -> Option<ComponentMatch> {
        let anchor_key = family_bucket_key_for_symbol(&component.trimmed, &component.signature);
        let mut visited = FxHashSet::default();
        let mut exact_examined = 0usize;
        if let Some(bucket) = anchor_map.get(&anchor_key) {
            for &idx in bucket {
                if exact_examined >= SYM_UNIFY_EXACT_ANCHOR_BUDGET {
                    break;
                }
                exact_examined += 1;
                if !visited.insert(idx) {
                    continue;
                }
                if let Some(found) = self.anchor_match(component, idx, comparator, counters) {
                    return Some(found);
                }
            }
        }

        let mut neighbor_examined = 0usize;
        for neighbor in family_bucket_neighbors(anchor_key) {
            if neighbor == anchor_key {
                continue;
            }
            let Some(bucket) = anchor_map.get(&neighbor) else {
                continue;
            };
            for &idx in bucket {
                if neighbor_examined >= SYM_UNIFY_NEIGHBOR_ANCHOR_BUDGET {
                    return None;
                }
                neighbor_examined += 1;
                if !visited.insert(idx) {
                    continue;
                }
                if let Some(found) = self.anchor_match(component, idx, comparator, counters) {
                    return Some(found);
                }
            }
        }
        None
    }

    /// Evaluates one anchor for the component, counting a reject by reason.
    #[cfg(feature = "symboldict")]
    fn anchor_match(
        &mut self,
        component: &PageComponent,
        idx: usize,
        comparator: &mut Comparator,
        counters: &mut SymUnifyIngestCounters,
    ) -> Option<ComponentMatch> {
        let decision = self.evaluate_symbol_unify_anchor_match(
            &component.trimmed,
            component.signature,
            component.pixel_count,
            idx,
            comparator,
        );
        let (score, dx, dy) = match decision {
            SymUnifyAnchorDecision::Accept { score, dx, dy } => (score, dx, dy),
            SymUnifyAnchorDecision::RejectScore { .. } => {
                counters.anchor_score_rejects += 1;
                return None;
            }
            SymUnifyAnchorDecision::RejectOutsideInk => {
                counters.anchor_outside_rejects += 1;
                return None;
            }
            SymUnifyAnchorDecision::RejectCompare => {
                counters.anchor_compare_rejects += 1;
                return None;
            }
            SymUnifyAnchorDecision::RejectOverlap => {
                counters.anchor_overlap_rejects += 1;
                return None;
            }
            _ => return None,
        };
        Some(ComponentMatch {
            symbol_index: idx,
            source: MatchSource::Anchor,
            err_or_score: score,
            dx,
            dy,
            needs_refinement: false,
        })
    }

    /// Codes the component as the matched symbol: counts the use, records
    /// the page, grows the anchor map, and adds the instance.
    #[cfg(feature = "symboldict")]
    fn record_component_match(
        &mut self,
        found: ComponentMatch,
        component: PageComponent,
        ingest: &mut PageIngest,
        recent: &mut RecentSymbols,
        symbol_instances: &mut Vec<SymbolInstance>,
    ) {
        let idx = found.symbol_index;
        if ingest.debug_matching {
            let line = self.matching_debug_line(&found, &component.rect, ingest.cc_index);
            ingest.debug_lines.push(line);
        }

        self.symbol_usage[idx] += 1;
        self.note_symbol_page(idx, ingest.page_num);
        if found.source != MatchSource::Recent
            && let Some(anchor_map) = ingest.sym_unify_anchor_map.as_mut()
        {
            self.maybe_add_sym_unify_anchor(anchor_map, idx, ingest.page_num);
        }
        let needs_refinement = found.needs_refinement;
        symbol_instances.push(SymbolInstance {
            symbol_index: idx,
            position: component.rect,
            instance_bitmap: component.symbol,
            needs_refinement,
            refinement_dx: if needs_refinement { found.dx } else { 0 },
            refinement_dy: if needs_refinement { found.dy } else { 0 },
        });
        recent.cache.touch(idx);
        let unify = self.config.lossy_symbol_mode == LossySymbolMode::SymbolUnify;
        match found.source {
            MatchSource::Recent if unify => ingest.counters.recent_hits += 1,
            MatchSource::Anchor => ingest.counters.anchor_hits += 1,
            MatchSource::Bucket if unify => ingest.counters.bucket_hits += 1,
            MatchSource::Recent | MatchSource::Bucket => {}
        }
    }

    /// The page 0 matching log line for a matched component.
    #[cfg(feature = "symboldict")]
    fn matching_debug_line(&self, found: &ComponentMatch, rect: &Rect, cc_index: usize) -> String {
        let idx = found.symbol_index;
        let proto = &self.global_symbols[idx];
        let (err, dx, dy) = (found.err_or_score, found.dx, found.dy);
        if found.source == MatchSource::Anchor {
            return format!(
                "CC#{:04} UNIFY  pos=({},{}) {}x{} → proto#{} {}x{} score={} dx={} dy={} [anchor]",
                cc_index,
                rect.x,
                rect.y,
                rect.width,
                rect.height,
                idx,
                proto.width,
                proto.height,
                err,
                dx,
                dy
            );
        }
        let mode = if found.needs_refinement {
            "REFINE"
        } else if err == 0 && dx == 0 && dy == 0 {
            "EXACT "
        } else {
            "LOSSY "
        };
        let suffix = if found.source == MatchSource::Recent {
            " [recent]"
        } else {
            ""
        };
        format!(
            "CC#{:04} {} pos=({},{}) {}x{} → proto#{} {}x{} err={} dx={} dy={}{}",
            cc_index,
            mode,
            rect.x,
            rect.y,
            rect.width,
            rect.height,
            idx,
            proto.width,
            proto.height,
            err,
            dx,
            dy,
            suffix
        )
    }

    /// Stores an unmatched component as a new symbol and adds its instance.
    #[cfg(feature = "symboldict")]
    fn record_new_symbol(
        &mut self,
        component: PageComponent,
        ingest: &mut PageIngest,
        recent: &mut RecentSymbols,
        symbol_instances: &mut Vec<SymbolInstance>,
    ) {
        let PageComponent {
            symbol,
            trimmed,
            trim_offset,
            signature: trimmed_sig,
            pixel_count,
            rect,
            ..
        } = component;
        let page_num = ingest.page_num;
        let signature = if self.config.lossy_symbol_mode == LossySymbolMode::SymbolUnify {
            trimmed_sig
        } else {
            crate::jbig2classify::with_hole_count(trimmed_sig, &trimmed)
        };
        let idx = self.push_symbol_with_signature(trimmed, pixel_count, page_num, signature);
        self.metrics.symbol_stats.symbols_discovered += 1;
        if ingest.debug_matching {
            ingest.debug_lines.push(format!(
                "CC#{:04} NEW    pos=({},{}) {}x{} trim_off=({},{}) → new proto#{} {}x{}",
                ingest.cc_index,
                rect.x,
                rect.y,
                rect.width,
                rect.height,
                trim_offset.x,
                trim_offset.y,
                idx,
                self.global_symbols[idx].width,
                self.global_symbols[idx].height
            ));
        }
        let key = hash_key(&self.global_symbols[idx]);
        self.hash_map.entry(key).or_default().push(idx);
        if let Some(anchor_map) = ingest.sym_unify_anchor_map.as_mut() {
            self.maybe_add_sym_unify_anchor(anchor_map, idx, page_num);
        }
        symbol_instances.push(SymbolInstance {
            symbol_index: idx,
            position: rect,
            instance_bitmap: symbol,
            needs_refinement: false,
            refinement_dx: 0,
            refinement_dy: 0,
        });
        recent.cache.touch(idx);
        if self.config.lossy_symbol_mode == LossySymbolMode::SymbolUnify {
            ingest.counters.new_symbols += 1;
        }
    }

    /// Logs the page's unification ingest counters and anchor growth.
    #[cfg(feature = "symboldict")]
    fn record_ingest_diagnostics(&mut self, ingest: &PageIngest) {
        let final_anchor_count = ingest
            .sym_unify_anchor_map
            .as_ref()
            .map(|anchors| anchors.values().map(Vec::len).sum::<usize>())
            .unwrap_or(0);
        let final_anchor_bytes = ingest
            .sym_unify_anchor_map
            .as_ref()
            .map(|anchors| anchor_map_dictionary_bytes(&self.global_symbols, anchors))
            .unwrap_or(0);
        let counters = &ingest.counters;
        self.state.ingest_debug_lines.push(format!(
            "sym_unify ingest page={}: cc={} recent_hits={} anchor_hits={} bucket_hits={} new_symbols={} initial_anchors={} final_anchors={} initial_anchor_bytes={} final_anchor_bytes={} anchor_score_rejects={} anchor_outside_rejects={} anchor_compare_rejects={} anchor_overlap_rejects={}",
            ingest.page_num + 1,
            ingest.cc_index,
            counters.recent_hits,
            counters.anchor_hits,
            counters.bucket_hits,
            counters.new_symbols,
            ingest.initial_anchor_count,
            final_anchor_count,
            ingest.initial_anchor_bytes,
            final_anchor_bytes,
            counters.anchor_score_rejects,
            counters.anchor_outside_rejects,
            counters.anchor_compare_rejects,
            counters.anchor_overlap_rejects,
        ));
    }

    pub fn collect_symbols(&mut self, roi: &Array2<u8>) -> Result<()> {
        let bitimage = crate::jbig2sym::array_to_bitimage(roi).map_err(|error| anyhow!(error))?;
        let (_, trimmed) = bitimage.trim();
        let key = hash_key(&trimmed);
        let page_num = self.pages.len();
        if self.page_symbol_indices.len() <= page_num {
            self.page_symbol_indices.resize_with(page_num + 1, Vec::new);
        }

        if !self.hash_map.contains_key(&key) {
            let pixel_count = trimmed.count_ones();
            let idx = self.push_symbol(trimmed, pixel_count, page_num);
            self.metrics.symbol_stats.symbols_discovered += 1;
            self.hash_map.insert(key, vec![idx]);
        }
        Ok(())
    }

    pub fn flush(&mut self) -> Result<Vec<u8>> {
        self.config
            .validate_exact_symbol_mode()
            .map_err(|error| anyhow!(error))?;
        let include_header = self.state.full_headers_remaining;
        self.state.decision_debug_lines.clear();
        match self.config.lossy_symbol_mode {
            LossySymbolMode::SymbolUnify => self.apply_symbol_unify()?,
            LossySymbolMode::Off => {}
        }
        let plan = self.plan_document(include_header)?;
        self.validate_plan(&plan)?;
        self.metrics.resources = DocumentResourceMetrics::from_plan(&plan)?;
        self.config
            .decoder_resource_limits
            .check(&self.metrics.resources)?;
        let output = self.serialize_full_document(&plan)?;
        self.state.full_headers_remaining = false;
        self.next_segment_number = plan.next_segment_number;
        Ok(output)
    }

    pub fn flush_pdf_split(&mut self) -> Result<PdfSplitOutput> {
        self.config
            .validate_exact_symbol_mode()
            .map_err(|error| anyhow!(error))?;
        self.state.pdf_mode = true;
        self.state.decision_debug_lines.clear();
        match self.config.lossy_symbol_mode {
            LossySymbolMode::SymbolUnify => self.apply_symbol_unify()?,
            LossySymbolMode::Off => {}
        }
        let plan = self.plan_document(false)?;
        self.validate_plan(&plan)?;
        self.metrics.resources = DocumentResourceMetrics::from_plan(&plan)?;
        self.config
            .decoder_resource_limits
            .check(&self.metrics.resources)?;
        let (
            global_segments,
            page_streams,
            local_dict_bytes_per_page,
            text_region_bytes_per_page,
            generic_region_bytes_per_page,
        ) = self.serialize_pdf_split(&plan)?;
        self.next_segment_number = plan.next_segment_number;
        Ok(PdfSplitOutput {
            global_segments,
            page_streams,
            local_dict_bytes_per_page,
            text_region_bytes_per_page,
            generic_region_bytes_per_page,
        })
    }
}

/// Encodes a generic region, optionally wrapping it in a complete JBIG2 file.
/// This function is intended to be the top-level entry point for encoding a single generic region.
pub fn encode_generic_region(img: &BitImage, cfg: &Jbig2Config) -> Result<Vec<u8>> {
    cfg.generic
        .validate()
        .map_err(|e: &'static str| anyhow!(e))?;
    // Build generic region config from high-level parameters
    let mut gr_cfg = GenericRegionParams::new(img.width as u32, img.height as u32, cfg.generic.dpi);
    gr_cfg.comb_operator = cfg.generic.comb_operator;
    gr_cfg.mmr = cfg.generic.mmr;
    gr_cfg.template = cfg.generic.template;
    gr_cfg.tpgdon = cfg.generic.tpgdon;
    for (slot, value) in gr_cfg.at.iter_mut().zip(&cfg.generic.at_pixels) {
        *slot = *value;
    }
    gr_cfg.at_pixels = cfg.generic.at_pixels.clone();
    gr_cfg.validate().map_err(|e: &'static str| anyhow!(e))?;

    let coder_data = Jbig2ArithCoder::encode_generic_payload_with_termination(
        img,
        gr_cfg.template,
        &gr_cfg.at_pixels,
        cfg.canonical_arithmetic_termination,
    )?;

    let params: GenericRegionParams = gr_cfg.clone();

    let mut generic_region_payload = params.to_bytes();
    generic_region_payload.extend_from_slice(&coder_data);

    // Create the generic region segment (segment number 1)
    let generic_region_segment = Segment {
        number: 1, // Segment number 1
        seg_type: SegmentType::ImmediateGenericRegion,
        deferred_non_retain: false,
        retain_flags: 0,
        page_association_type: 0, // Explicit page association
        referred_to: Vec::new(),
        page: Some(1),                           // Page 1
        payload: generic_region_payload.clone(), // Clone to avoid move
    };

    // If caller wants only the segment, we're done
    if !cfg.want_full_headers {
        let mut seg_bytes = Vec::new();
        generic_region_segment.write_into(&mut seg_bytes)?;
        return Ok(seg_bytes);
    }

    // Otherwise wrap it in a complete one-page JBIG2 file
    let mut out = Vec::with_capacity(generic_region_payload.len() + 64);

    // File header — sequential organisation (matches segment layout below).
    out.extend_from_slice(
        &FileHeader {
            organisation_type: false,
            unknown_n_pages: false,
            n_pages: 1,
        }
        .to_bytes(),
    );

    // Page Information segment (segment number 0)
    Segment {
        number: 0,
        seg_type: SegmentType::PageInformation,
        deferred_non_retain: false,
        retain_flags: 0,
        page_association_type: 0,
        referred_to: vec![],
        page: Some(1),
        payload: PageInfo {
            width: img.width as u32,
            height: img.height as u32,
            xres: cfg.generic.dpi,
            yres: cfg.generic.dpi,
            is_lossless: cfg.is_lossless,
            default_pixel: cfg.default_pixel,
            default_operator: cfg.generic.comb_operator,
            ..Default::default()
        }
        .to_bytes(),
    }
    .write_into(&mut out)?;

    // Generic region segment (segment number 1)
    generic_region_segment.write_into(&mut out)?;

    // EOF segment (segment number 2)
    Segment {
        number: 2,
        seg_type: SegmentType::EndOfFile,
        deferred_non_retain: false,
        retain_flags: 0,
        page_association_type: 2,
        referred_to: vec![],
        page: None,
        payload: vec![],
    }
    .write_into(&mut out)?;

    Ok(out)
}

/// Encodes a sequence of images as a JBIG2 document.
///
/// # Arguments
/// * `images` - A slice of 2D arrays containing the input images
/// * `config` - Configuration for the encoder
///
/// # Returns
/// A `Result` containing the encoded JBIG2 document as a byte vector if successful,
/// or an error if encoding fails.
pub fn encode_document(images: &[Array2<u8>], config: &Jbig2Config) -> Result<Vec<u8>> {
    let mut encoder = Jbig2Encoder::new(config);
    for image in images {
        encoder.add_page(image)?;
    }
    encoder.flush()
}

pub fn get_version() -> &'static str {
    "0.2.0"
}

#[inline]
pub fn hash_key(img: &BitImage) -> HashKey {
    // Dimension-based bucketing: symbols with similar dimensions land in the same
    // bucket, enabling fuzzy matching via the Comparator during extraction.
    // The Comparator handles size differences up to MAX_DIMENSION_DELTA (10px),
    // so we bucket by (height, width) to keep buckets tight.
    let h = img.height as u64;
    let w = img.width as u64;
    HashKey(h * 10_000 + w)
}

/// Helper function to find the first black pixel in the BitImage
/// Returns (x, y) coordinates of the first black pixel, or None if no black pixels
pub fn first_black_pixel(image: &BitImage) -> Option<(usize, usize)> {
    for y in 0..image.height {
        for (word_index, &word) in image.row_words(y).iter().enumerate() {
            if word == 0 {
                continue;
            }
            let x = word_index * 32 + word.leading_zeros() as usize;
            if x < image.width {
                return Some((x, y));
            }
        }
    }
    None
}
