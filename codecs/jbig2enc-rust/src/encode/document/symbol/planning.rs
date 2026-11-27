// `plan_document` (~1565 lines), `build_planned_page` (~260 lines), and
// `validate_plan` decide every segment, dictionary, and text-region layout
// choice for the document. They are mutually recursive and share a large
// amount of context, so splitting their bodies across files would be far
// riskier than the line-count savings are worth — this file is the
// documented exception to the ~1000-line target called out in
// REFACTOR_JBIG2ENC.md.
use super::super::{
    Jbig2Encoder, PageData, SYM_UNIFY_STRONG_ANCHOR_MIN_PAGE_SPAN,
    SYM_UNIFY_STRONG_ANCHOR_MIN_USAGE,
};
use super::dictionary::{encode_symbol_dictionary_segments, plan_symbol_dictionary_layout};
use super::residual::MAX_RESIDUAL_REGIONS;
use super::text_region::encode_text_region_mapped;
use super::text_region_refine::encode_text_region_with_refinement;
use super::types::{
    BuiltPage, CounterfactualProbeStats, DetailedCompareProbeStats, EncodedSymbolDictionary,
    PlannedDocument, PlannedPage, PlannedPageLayout, RefinementPlan, ResidualReasonCode,
    ResidualReasonStats, ResidualShapeKind, ResidualSymbolTrace, SymUnifyAnchorCandidate,
    SymUnifyAnchorDecision, SymbolDictLayout, SymbolInstance, anchor_map_dictionary_bytes,
    bitmap_proxy_bytes, classify_residual_shape, encoder_diagnostics_enabled,
    indexed_symbol_dictionary_bytes, record_counterfactual_probe, record_detailed_compare_probe,
    record_labeled_counterfactual_probe, relaxed_compare_probe_max_err, update_best_reject,
};
use crate::debug;
use crate::jbig2classify::{
    FamilyBucketKey, family_bucket_key_for_symbol, family_bucket_neighbors,
};
use crate::jbig2comparator::Comparator;
use crate::jbig2structs::{
    FileHeader, LossySymbolMode, PageInfo, PdfDictionaryPolicy, Segment, SegmentType,
};
use crate::jbig2sym::BitImage;
use anyhow::Result;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

/// Counters, samples and probe stats gathered while deciding where each
/// page-local one-off symbol is coded.
#[derive(Default)]
struct ResidualPlacementStats {
    anchor_comparator: Comparator,
    local_anchor_attach_count: usize,
    anchor_attach_count: usize,
    local_score_rescue_count: usize,
    anchor_score_rescue_count: usize,
    local_anchor_attach_sample: Vec<(usize, usize, usize)>,
    anchor_attach_sample: Vec<(usize, usize, usize)>,
    local_rescue_count: usize,
    local_rescue_sample: Vec<(usize, usize, usize, usize)>,
    residual_symbol_traces: FxHashMap<usize, ResidualSymbolTrace>,
    counterfactual_local_dim_relax2: CounterfactualProbeStats,
    counterfactual_global_overlap_skip: CounterfactualProbeStats,
}

/// What placing one page's one-off symbols needs to know about that page.
struct PageResidualContext<'p> {
    page_num: usize,
    page: &'p PageData,
    local_use_counts: HashMap<usize, usize>,
    local_anchor_candidates: Vec<usize>,
    anchor_map: Option<&'p FxHashMap<FamilyBucketKey, Vec<usize>>>,
    diagnostics_enabled: bool,
}

/// Where a page-local symbol ends up after residual planning.
enum LocalSymbolPlacement {
    /// Stays in the page's local dictionary.
    Keep,
    /// Its instances are coded as the given anchor symbol.
    Remap(usize),
    /// Its instances are coded in the page's generic residual region.
    Residual,
}

/// The best anchor found for a residual symbol, and why the others failed.
#[derive(Default)]
struct ResidualAnchorSearch {
    best_anchor: Option<SymUnifyAnchorCandidate>,
    best_reject: Option<SymUnifyAnchorDecision>,
    had_candidates: bool,
}

/// Totals of the diagnostic scan that asks which residual symbols could
/// attach to an anchor.
#[derive(Default)]
struct ResidualAnchorScan {
    attachable: usize,
    attachable_with_score_rescue: usize,
    attachable_to_any_global: usize,
    sampled: Vec<(usize, usize)>,
    sampled_score_rescue: Vec<(usize, usize)>,
    sampled_any_global: Vec<(usize, usize)>,
    reject_counts: FxHashMap<&'static str, usize>,
    area_buckets: [usize; 4],
}

impl ResidualAnchorScan {
    /// Counts a residual that a current anchor accepts.
    fn record_attached(&mut self, residual_index: usize, anchor_index: usize) {
        self.attachable += 1;
        self.attachable_with_score_rescue += 1;
        if self.sampled.len() < 16 {
            self.sampled.push((residual_index, anchor_index));
        }
    }

    /// Counts a residual that only the score rescue slack would attach.
    fn record_score_rescue(&mut self, residual_index: usize, anchor_index: usize) {
        self.attachable_with_score_rescue += 1;
        if self.sampled_score_rescue.len() < 16 {
            self.sampled_score_rescue
                .push((residual_index, anchor_index));
        }
    }

    /// Counts a residual that some global symbol, anchor-ready or not,
    /// accepts.
    fn record_any_global(&mut self, residual_index: usize, anchor_index: usize) {
        self.attachable_to_any_global += 1;
        if self.sampled_any_global.len() < 16 {
            self.sampled_any_global.push((residual_index, anchor_index));
        }
    }
}

/// A relaxed comparison kept as evidence: the result, the compare error
/// limit it was judged against, whether the dimensions matched exactly and
/// whether the anchor was strong.
type CompareProbeDetail = (crate::jbig2comparator::CompareResult, u32, bool, bool);

/// The detailed and slack-relaxed outcomes of one family of compare probes.
#[derive(Default)]
struct CompareSlackProbes {
    total_err_details: DetailedCompareProbeStats,
    slack2: CounterfactualProbeStats,
    slack4: CounterfactualProbeStats,
}

/// Diagnostic counterfactuals over the residual symbols: what a relaxed
/// prescreen, a relaxed comparison or a cross-page anchor would have rescued.
#[derive(Default)]
struct ResidualCounterfactualProbes {
    local_dim_cross_page_current: CounterfactualProbeStats,
    local_dim_cross_page_dim2: CounterfactualProbeStats,
    cross_page_comparator: Comparator,
    overlap_bypass_outcomes: FxHashMap<&'static str, CounterfactualProbeStats>,
    overlap_bypass_comparator: Comparator,
    overlap_compare_probe_outcomes: FxHashMap<&'static str, CounterfactualProbeStats>,
    overlap_compare_probe_comparator: Comparator,
    from_overlap_compare: CompareSlackProbes,
    from_global_compare: CompareSlackProbes,
}

/// What the overlap-bypass search found for one residual symbol.
#[derive(Default)]
struct OverlapBypassOutcome {
    recovered: bool,
    best_reject: Option<SymUnifyAnchorDecision>,
    best_compare: Option<CompareProbeDetail>,
}

/// The best relaxed comparison found for one residual symbol.
struct GlobalCompareProbe {
    label: &'static str,
    total_err: u32,
    detail: Option<CompareProbeDetail>,
}

/// Whether a residual symbol matches a symbol first seen on another page.
enum CrossPageMatch {
    Current,
    DimRelaxed,
    None,
}

/// The size class of a symbol's bounding box in the residual scan.
fn residual_area_bucket(area: usize) -> usize {
    match area {
        0..=16 => 0,
        17..=32 => 1,
        33..=64 => 2,
        _ => 3,
    }
}

/// Sorts labelled probe outcomes by bitmap bytes, then symbol count, then
/// label.
fn sorted_probe_outcomes(
    outcomes: FxHashMap<&'static str, CounterfactualProbeStats>,
) -> Vec<(&'static str, CounterfactualProbeStats)> {
    let mut sorted: Vec<_> = outcomes.into_iter().collect();
    sorted.sort_by(|lhs, rhs| {
        rhs.1
            .bitmap_proxy_bytes
            .cmp(&lhs.1.bitmap_proxy_bytes)
            .then_with(|| rhs.1.symbol_count.cmp(&lhs.1.symbol_count))
            .then_with(|| lhs.0.cmp(rhs.0))
    });
    sorted
}

/// Records, for each symbol of `subset`, its position in the exported
/// dictionary; symbols the dictionary does not export keep `u32::MAX`.
fn fill_dictionary_positions(
    positions: &mut [u32],
    subset: &[usize],
    encoded_dict: &EncodedSymbolDictionary,
) {
    for (refs_idx, &dict_pos) in encoded_dict.input_to_exported_pos.iter().enumerate() {
        if dict_pos != u32::MAX {
            positions[subset[refs_idx]] = dict_pos;
        }
    }
}

/// Logs any dictionary position past the end of the exported dictionary.
fn warn_out_of_range_dictionary_positions(global_sym_to_dict_pos: &[u32], exported: u32) {
    for (gs_idx, &dict_pos) in global_sym_to_dict_pos.iter().enumerate() {
        if dict_pos != u32::MAX && dict_pos >= exported {
            log::warn!(
                "BUG: global_sym_to_dict_pos[{}] = {} but num_global_dict_symbols = {}",
                gs_idx,
                dict_pos,
                exported
            );
        }
    }
}

/// Records a dictionary layout's subset-relative refinement plans in a map
/// indexed by global symbol.
fn record_refinement_plans(
    refinement_map: &mut [Option<RefinementPlan>],
    subset: &[usize],
    layout_refinements: &[Option<RefinementPlan>],
) {
    for (subset_index, refinement) in layout_refinements.iter().enumerate() {
        let Some(refinement) = refinement else {
            continue;
        };
        refinement_map[subset[subset_index]] = Some(RefinementPlan {
            prototype_input_index: subset[refinement.prototype_input_index],
            refinement_dx: refinement.refinement_dx,
            refinement_dy: refinement.refinement_dy,
        });
    }
}

/// Logs the residual anchor scan: how many residuals could attach, why the
/// rest were rejected, and samples of each.
fn push_residual_anchor_scan_lines(
    lines: &mut Vec<String>,
    scan: &ResidualAnchorScan,
    residual_unique_count: usize,
) {
    lines.push(format!(
        "sym_unify residual anchor scan: residual_unique={} attachable_to_current_anchors={} attachable_with_score_rescue={} score_rescue_extra={} unattached={}",
        residual_unique_count,
        scan.attachable,
        scan.attachable_with_score_rescue,
        scan.attachable_with_score_rescue
            .saturating_sub(scan.attachable),
        residual_unique_count.saturating_sub(scan.attachable)
    ));
    let reject_counts = &scan.reject_counts;
    lines.push(format!(
        "sym_unify residual reject breakdown: dim={} pixel_delta={} signature={} overlap={} compare={} outside_ink={} score={} area_le16={} area_le32={} area_le64={} area_gt64={}",
        reject_counts.get("dim").copied().unwrap_or(0),
        reject_counts.get("pixel_delta").copied().unwrap_or(0),
        reject_counts.get("signature").copied().unwrap_or(0),
        reject_counts.get("overlap").copied().unwrap_or(0),
        reject_counts.get("compare").copied().unwrap_or(0),
        reject_counts.get("outside_ink").copied().unwrap_or(0),
        reject_counts.get("score").copied().unwrap_or(0),
        scan.area_buckets[0],
        scan.area_buckets[1],
        scan.area_buckets[2],
        scan.area_buckets[3],
    ));
    lines.push(format!(
        "sym_unify residual any-global scan: residual_unique={} attachable_to_any_global={} extra_beyond_anchor_ready={}",
        residual_unique_count,
        scan.attachable_to_any_global,
        scan.attachable_to_any_global
            .saturating_sub(scan.attachable),
    ));
    if !scan.sampled.is_empty() {
        lines.push(format!(
            "sym_unify residual anchor sample: {:?}",
            scan.sampled
        ));
    }
    if !scan.sampled_score_rescue.is_empty() {
        lines.push(format!(
            "sym_unify residual score-rescue sample: {:?}",
            scan.sampled_score_rescue
        ));
    }
    if !scan.sampled_any_global.is_empty() {
        lines.push(format!(
            "sym_unify residual any-global sample: {:?}",
            scan.sampled_any_global
        ));
    }
}

/// Logs the outcome of every residual counterfactual probe.
fn push_residual_counterfactual_lines(
    lines: &mut Vec<String>,
    probes: ResidualCounterfactualProbes,
) {
    let current = &probes.local_dim_cross_page_current;
    let dim2 = &probes.local_dim_cross_page_dim2;
    lines.push(format!(
        "sym_unify cross-page local-dim probes: current_symbols={} current_bitmap_proxy_bytes={} current_pages={} dim2_only_symbols={} dim2_only_bitmap_proxy_bytes={} dim2_only_pages={}",
        current.symbol_count,
        current.bitmap_proxy_bytes,
        current.pages.len(),
        dim2.symbol_count,
        dim2.bitmap_proxy_bytes,
        dim2.pages.len(),
    ));
    if !current.samples.is_empty() {
        lines.push(format!(
            "sym_unify cross-page local-dim current sample: {:?}",
            current.samples
        ));
    }
    if !dim2.samples.is_empty() {
        lines.push(format!(
            "sym_unify cross-page local-dim dim2 sample: {:?}",
            dim2.samples
        ));
    }

    let sorted_overlap_bypass_outcomes = sorted_probe_outcomes(probes.overlap_bypass_outcomes);
    let overlap_bypass_total_symbols: usize = sorted_overlap_bypass_outcomes
        .iter()
        .map(|(_, stats)| stats.symbol_count)
        .sum();
    let overlap_bypass_total_bitmap_proxy_bytes: usize = sorted_overlap_bypass_outcomes
        .iter()
        .map(|(_, stats)| stats.bitmap_proxy_bytes)
        .sum();
    lines.push(format!(
        "sym_unify overlap-bypass outcomes: outcomes={} symbols={} bitmap_proxy_bytes={}",
        sorted_overlap_bypass_outcomes.len(),
        overlap_bypass_total_symbols,
        overlap_bypass_total_bitmap_proxy_bytes,
    ));
    for (label, stats) in sorted_overlap_bypass_outcomes {
        lines.push(format!(
            "  overlap-bypass {}: symbols={} pages={} black_pixels={} bitmap_proxy_bytes={} sample={:?}",
            label,
            stats.symbol_count,
            stats.pages.len(),
            stats.black_pixels,
            stats.bitmap_proxy_bytes,
            stats.samples
        ));
    }

    let sorted_overlap_compare_probe_outcomes =
        sorted_probe_outcomes(probes.overlap_compare_probe_outcomes);
    let overlap_compare_probe_total_symbols: usize = sorted_overlap_compare_probe_outcomes
        .iter()
        .map(|(_, stats)| stats.symbol_count)
        .sum();
    let overlap_compare_probe_total_bitmap_proxy_bytes: usize =
        sorted_overlap_compare_probe_outcomes
            .iter()
            .map(|(_, stats)| stats.bitmap_proxy_bytes)
            .sum();
    lines.push(format!(
        "sym_unify global-compare relaxed probe: outcomes={} symbols={} bitmap_proxy_bytes={}",
        sorted_overlap_compare_probe_outcomes.len(),
        overlap_compare_probe_total_symbols,
        overlap_compare_probe_total_bitmap_proxy_bytes,
    ));
    for (label, stats) in sorted_overlap_compare_probe_outcomes {
        lines.push(format!(
            "  global-compare relaxed {}: symbols={} pages={} black_pixels={} bitmap_proxy_bytes={} sample={:?}",
            label,
            stats.symbol_count,
            stats.pages.len(),
            stats.black_pixels,
            stats.bitmap_proxy_bytes,
            stats.samples
        ));
    }
    let overlap_details = &probes.from_overlap_compare.total_err_details;
    lines.push(format!(
        "sym_unify overlap-bypass compare total_err detail: symbols={} bitmap_proxy_bytes={} exact_dims={} strong_anchor={} shift_le1={} over_by_le2={} over_by_le4={} over_by_le8={} over_by_gt8={} sample={:?}",
        overlap_details.symbol_count,
        overlap_details.bitmap_proxy_bytes,
        overlap_details.exact_dims_count,
        overlap_details.strong_anchor_count,
        overlap_details.shift_le1_count,
        overlap_details.over_by_le2_count,
        overlap_details.over_by_le4_count,
        overlap_details.over_by_le8_count,
        overlap_details.over_by_gt8_count,
        overlap_details.samples
    ));
    let global_details = &probes.from_global_compare.total_err_details;
    lines.push(format!(
        "sym_unify global-compare total_err detail: symbols={} bitmap_proxy_bytes={} exact_dims={} strong_anchor={} shift_le1={} over_by_le2={} over_by_le4={} over_by_le8={} over_by_gt8={} sample={:?}",
        global_details.symbol_count,
        global_details.bitmap_proxy_bytes,
        global_details.exact_dims_count,
        global_details.strong_anchor_count,
        global_details.shift_le1_count,
        global_details.over_by_le2_count,
        global_details.over_by_le4_count,
        global_details.over_by_le8_count,
        global_details.over_by_gt8_count,
        global_details.samples
    ));
    lines.push(format!(
        "sym_unify compare-slack probes: global_total_err_slack2_symbols={} global_total_err_slack2_bitmap_proxy_bytes={} global_total_err_slack4_symbols={} global_total_err_slack4_bitmap_proxy_bytes={} overlap_compare_slack2_symbols={} overlap_compare_slack2_bitmap_proxy_bytes={} overlap_compare_slack4_symbols={} overlap_compare_slack4_bitmap_proxy_bytes={}",
        probes.from_global_compare.slack2.symbol_count,
        probes.from_global_compare.slack2.bitmap_proxy_bytes,
        probes.from_global_compare.slack4.symbol_count,
        probes.from_global_compare.slack4.bitmap_proxy_bytes,
        probes.from_overlap_compare.slack2.symbol_count,
        probes.from_overlap_compare.slack2.bitmap_proxy_bytes,
        probes.from_overlap_compare.slack4.symbol_count,
        probes.from_overlap_compare.slack4.bitmap_proxy_bytes
    ));
}

impl<'a> Jbig2Encoder<'a> {
    pub(crate) fn plan_document(&mut self, include_header: bool) -> Result<PlannedDocument> {
        #[cfg(feature = "symboldict")]
        if self.config.optimize_exact_symbols || self.config.optimize_lossy_symbols {
            return self.plan_optimized_exact_document(include_header);
        }
        debug!("Symbol stats before encoding: {}", self.get_symbol_stats());
        let diagnostics_enabled = encoder_diagnostics_enabled();
        let planning_start = Instant::now();

        if self.config.auto_thresh && !self.config.is_lossless {
            let clustering_start = Instant::now();
            self.cluster_symbols()?;
            self.metrics.symbol_mode.clustering += clustering_start.elapsed();
        }

        self.prune_symbols_if_needed();
        self.alias_local_symbols_to_globals()?;
        self.validate_symbol_instance_indices()?;

        // PDF permits page-associated symbol dictionaries. Keep the old policy
        // explicit for existing presets; selective policies share only symbols
        // with cross-page reuse and partition the remainder below.
        let legacy_pdf = self.state.pdf_mode
            && self.config.pdf_dictionary_policy == PdfDictionaryPolicy::AllGlobal;
        let global_symbol_indices = self.multi_page_global_candidates(legacy_pdf);
        let global_set: HashSet<usize> = global_symbol_indices.iter().copied().collect();
        if diagnostics_enabled {
            self.record_global_selection_diagnostics(&global_symbol_indices, &global_set);
        }

        let mut page_local_symbols: Vec<Vec<usize>> = if legacy_pdf {
            vec![Vec::new(); self.pages.len()]
        } else {
            self.page_symbol_indices
                .iter()
                .map(|symbols| {
                    symbols
                        .iter()
                        .copied()
                        .filter(|i| !global_set.contains(i))
                        .collect()
                })
                .collect()
        };
        let mut page_residual_symbols = vec![Vec::new(); self.pages.len()];
        if self.state.pdf_mode
            && self.config.pdf_dictionary_policy == PdfDictionaryPolicy::SharedGlobal
        {
            for (locals, residuals) in page_local_symbols
                .iter_mut()
                .zip(&mut page_residual_symbols)
            {
                residuals.append(locals);
            }
        }
        let mut page_residual_anchor_remaps: Vec<FxHashMap<usize, usize>> = (0..self.pages.len())
            .map(|_| FxHashMap::default())
            .collect();
        let sym_unify_global_anchor_map = self.planning_global_anchor_map(&global_symbol_indices);
        let sym_unify_global_anchor_bytes = sym_unify_global_anchor_map
            .as_ref()
            .map(|anchors| anchor_map_dictionary_bytes(&self.global_symbols, anchors))
            .unwrap_or(0);
        let mut placement = ResidualPlacementStats::default();
        let places_one_off_symbols = !self.config.is_lossless
            && (self.config.uses_lossy_symbol_dictionary()
                || self.config.refine
                || self.config.text_refine);
        let mut page_uses_generic_region = vec![false; self.pages.len()];
        for (page_num, page) in self.pages.iter().enumerate() {
            if places_one_off_symbols {
                let context = self.page_residual_context(
                    page_num,
                    page,
                    &page_local_symbols[page_num],
                    sym_unify_global_anchor_map.as_ref(),
                    diagnostics_enabled,
                );
                page_local_symbols[page_num] = self.place_page_local_symbols(
                    &context,
                    &page_local_symbols[page_num],
                    &mut page_residual_anchor_remaps[page_num],
                    &mut page_residual_symbols[page_num],
                    &mut placement,
                );
            }
            page_uses_generic_region[page_num] = self.page_needs_generic_region(
                page,
                &global_set,
                &mut page_local_symbols[page_num],
                &page_residual_anchor_remaps[page_num],
                &page_residual_symbols[page_num],
            );
        }

        let total_residual_symbols: usize = page_residual_symbols.iter().map(Vec::len).sum();
        let full_generic_pages = page_uses_generic_region.iter().filter(|&&v| v).count();
        if diagnostics_enabled {
            self.record_residual_planning_diagnostics(
                total_residual_symbols,
                full_generic_pages,
                &placement,
                sym_unify_global_anchor_bytes,
            );
        }
        if diagnostics_enabled
            && self.config.lossy_symbol_mode == LossySymbolMode::SymbolUnify
            && total_residual_symbols > 0
        {
            let lines = self.sym_unify_residual_probe_lines(
                &global_symbol_indices,
                &page_residual_symbols,
                &placement.residual_symbol_traces,
                sym_unify_global_anchor_map.as_ref(),
            );
            self.state.decision_debug_lines.extend(lines);
        }
        if diagnostics_enabled {
            self.record_page_residual_samples(&page_residual_symbols);
        }

        self.validate_symbol_partition(
            &global_symbol_indices,
            &page_local_symbols,
            &page_residual_symbols,
            &page_residual_anchor_remaps,
            &page_uses_generic_region,
        )?;

        let mut current_segment_number = self.next_segment_number;
        let (global_segments, encoded_global_dict, global_refinement_map) = self
            .encode_planned_global_dictionary(
                &global_symbol_indices,
                diagnostics_enabled,
                &mut current_segment_number,
            )?;

        let mut global_sym_to_dict_pos = vec![u32::MAX; self.global_symbols.len()];
        fill_dictionary_positions(
            &mut global_sym_to_dict_pos,
            &global_symbol_indices,
            &encoded_global_dict,
        );
        let num_global_dict_symbols = encoded_global_dict.exported_symbol_count;

        // Debug: check for out-of-range mappings
        if self.state.pdf_mode {
            warn_out_of_range_dictionary_positions(
                &global_sym_to_dict_pos,
                num_global_dict_symbols,
            );
        }

        let mut planned_local_export_count = 0usize;
        self.metrics.symbol_stats.global_symbol_count = num_global_dict_symbols as usize;

        let page_segment_start = current_segment_number;
        let mut page_layouts = Vec::with_capacity(self.pages.len());
        for (page_num, page) in self.pages.iter().enumerate() {
            let page_number = if self.state.pdf_mode {
                1u32
            } else {
                page_num as u32 + 1
            };
            if self.state.pdf_mode {
                current_segment_number = page_segment_start;
            }
            let page_info_segment_number = current_segment_number;
            current_segment_number += 1;
            let local_dict_layout =
                self.page_local_dictionary_layout(page, &page_local_symbols[page_num])?;
            let mut local_dict_segment_numbers = Vec::new();
            if let Some(local_dict_layout) = &local_dict_layout {
                if diagnostics_enabled {
                    self.state.decision_debug_lines.push(format!(
                        "page {} local dict layout: families={} singletons={} refined_members={} exported_members={}",
                        page_num + 1,
                        local_dict_layout.diagnostics.family_count,
                        local_dict_layout.diagnostics.singleton_family_count,
                        local_dict_layout.diagnostics.refined_member_count,
                        local_dict_layout.diagnostics.exported_member_count
                    ));
                    self.state.decision_debug_lines.extend(
                        local_dict_layout
                            .diagnostics
                            .sample_lines
                            .iter()
                            .take(16)
                            .cloned(),
                    );
                }
                for _ in 0..local_dict_layout.segment_count() {
                    local_dict_segment_numbers.push(current_segment_number);
                    current_segment_number += 1;
                }
                planned_local_export_count += local_dict_layout.export_input_indices.len();
            }
            let region_segment_number = current_segment_number;
            current_segment_number += 1;
            let has_residual_region = !page_residual_symbols[page_num].is_empty()
                && !page_uses_generic_region[page_num]
                && page.symbol_instances.iter().any(|inst| {
                    global_set.contains(&inst.symbol_index)
                        || page_local_symbols[page_num].contains(&inst.symbol_index)
                });
            let residual_region_segment_number = if has_residual_region {
                let number = current_segment_number;
                current_segment_number += if self.config.pack_residuals {
                    MAX_RESIDUAL_REGIONS as u32
                } else {
                    1
                };
                Some(number)
            } else {
                None
            };
            let end_of_page_segment_number = current_segment_number;
            current_segment_number += 1;
            let use_generic_region = page_uses_generic_region[page_num];
            if diagnostics_enabled {
                self.state.decision_debug_lines.push(format!(
                    "page {} plan: full_generic={} residual_region={} local_symbols={} residual_symbols={} anchor_remaps={} instances={}",
                    page_num + 1,
                    use_generic_region,
                    has_residual_region,
                    page_local_symbols[page_num].len(),
                    page_residual_symbols[page_num].len(),
                    page_residual_anchor_remaps[page_num].len(),
                    page.symbol_instances.len()
                ));
            }

            page_layouts.push(PlannedPageLayout {
                page_index: page_num,
                page_number,
                page_info_segment_number,
                local_dict_segment_numbers,
                local_dict_layout,
                region_segment_number,
                residual_region_segment_number,
                end_of_page_segment_number,
                local_symbols: page_local_symbols[page_num].clone(),
                residual_symbols: page_residual_symbols[page_num].clone(),
                residual_anchor_remaps: page_residual_anchor_remaps[page_num].clone(),
                use_generic_region,
            });
        }

        self.metrics.symbol_stats.local_symbol_count = planned_local_export_count;
        self.metrics.symbol_stats.symbols_exported =
            self.metrics.symbol_stats.global_symbol_count + planned_local_export_count;
        self.metrics.symbol_stats.avg_symbol_reuse =
            if self.metrics.symbol_stats.symbols_exported > 0 {
                self.symbol_usage.iter().sum::<usize>() as f64
                    / self.metrics.symbol_stats.symbols_exported as f64
            } else {
                0.0
            };

        self.metrics.symbol_mode.planning += planning_start.elapsed();

        let built_pages = self.build_planned_pages(
            &page_layouts,
            &global_sym_to_dict_pos,
            num_global_dict_symbols,
            &global_refinement_map,
        );

        let mut pages = Vec::with_capacity(built_pages.len());
        for built_page in built_pages {
            let built_page = built_page?;
            self.metrics.symbol_mode.symbol_dict_encoding += built_page.symbol_dict_time;
            self.metrics.symbol_mode.text_region_encoding += built_page.text_region_time;
            self.metrics.symbol_mode.generic_region_encoding += built_page.generic_region_time;
            self.metrics.symbol_stats.topology_refinements += built_page.topology_refinement_count;
            pages.push(built_page.page);
        }

        let eof_segment = Some(Segment {
            number: current_segment_number,
            seg_type: SegmentType::EndOfFile,
            deferred_non_retain: false,
            retain_flags: 0,
            page_association_type: 2,
            referred_to: vec![],
            page: None,
            payload: vec![],
        });
        current_segment_number += 1;

        Ok(PlannedDocument {
            file_header: if include_header {
                Some(FileHeader {
                    // Sequential organisation matches the on-disk layout this
                    // encoder produces (segment header + payload, repeated).
                    organisation_type: false,
                    unknown_n_pages: false,
                    n_pages: self.pages.len() as u32,
                })
            } else {
                None
            },
            global_segments,
            pages,
            eof_segment,
            next_segment_number: current_segment_number,
            global_bitmap_bytes: super::super::resources::dictionary_storage(
                &self.global_symbols,
                &global_sym_to_dict_pos,
            )
            .0,
            global_dictionary_pixels: super::super::resources::dictionary_storage(
                &self.global_symbols,
                &global_sym_to_dict_pos,
            )
            .1,
        })
    }

    /// The symbols the global dictionary carries: every symbol under the
    /// legacy all-global PDF policy, otherwise those used on more than one
    /// page (or all of a standalone single-page document's).
    fn multi_page_global_candidates(&self, legacy_pdf: bool) -> Vec<usize> {
        if legacy_pdf {
            return (0..self.global_symbols.len()).collect();
        }
        // Standalone mode: only multi-page symbols in global dict
        self.global_symbols
            .iter()
            .enumerate()
            .filter(|(i, _)| {
                self.symbol_page_count[*i] > 1 || (!self.state.pdf_mode && self.pages.len() == 1)
            })
            .map(|(i, _)| i)
            .collect()
    }

    /// Logs the selected global symbols and the multi-page candidates left
    /// out of the global dictionary.
    fn record_global_selection_diagnostics(
        &mut self,
        global_symbol_indices: &[usize],
        global_set: &HashSet<usize>,
    ) {
        // Every multi-page candidate is currently selected, so the
        // non-global remainder is empty; the lines keep their shape for when
        // the selection becomes selective.
        let multi_page_candidates = global_symbol_indices;
        let estimated_global_dict_bytes =
            indexed_symbol_dictionary_bytes(&self.global_symbols, global_symbol_indices);
        let low_value_global_candidates: Vec<(usize, usize, usize, i64)> = multi_page_candidates
            .iter()
            .copied()
            .filter(|symbol_index| !global_set.contains(symbol_index))
            .map(|symbol_index| {
                (
                    symbol_index,
                    self.symbol_usage[symbol_index],
                    self.symbol_page_count[symbol_index],
                    self.estimate_global_symbol_gain(symbol_index),
                )
            })
            .take(16)
            .collect();
        let multi_page_non_global = multi_page_candidates
            .len()
            .saturating_sub(global_symbol_indices.len());
        self.state.decision_debug_lines.push(format!(
            "planning globals: selected={} multi_page_non_global={} estimated_dict_bytes={} low_value_candidates={}",
            global_symbol_indices.len(),
            multi_page_non_global,
            estimated_global_dict_bytes,
            low_value_global_candidates.len()
        ));
        for (symbol_index, usage, page_span, gain) in low_value_global_candidates {
            self.state.decision_debug_lines.push(format!(
                "planning global candidate: symbol={} usage={} page_span={} estimated_global_gain={}",
                symbol_index, usage, page_span, gain
            ));
        }
    }

    /// Groups the anchor-ready global symbols by family bucket, for symbol
    /// unification only.
    fn planning_global_anchor_map(
        &self,
        global_symbol_indices: &[usize],
    ) -> Option<FxHashMap<FamilyBucketKey, Vec<usize>>> {
        if self.config.lossy_symbol_mode != LossySymbolMode::SymbolUnify {
            return None;
        }
        let mut anchors: FxHashMap<FamilyBucketKey, Vec<usize>> = FxHashMap::default();
        for &symbol_index in global_symbol_indices {
            if !self.sym_unify_anchor_ready(symbol_index, self.pages.len()) {
                continue;
            }
            let key = family_bucket_key_for_symbol(
                &self.global_symbols[symbol_index],
                &self.symbol_signatures[symbol_index],
            );
            anchors.entry(key).or_default().push(symbol_index);
        }
        Some(anchors)
    }

    /// Collects a page's local use counts and the local symbols that may
    /// serve as anchors for its one-off symbols.
    fn page_residual_context<'p>(
        &self,
        page_num: usize,
        page: &'p PageData,
        local_symbols: &[usize],
        anchor_map: Option<&'p FxHashMap<FamilyBucketKey, Vec<usize>>>,
        diagnostics_enabled: bool,
    ) -> PageResidualContext<'p> {
        let mut local_use_counts = HashMap::new();
        for instance in &page.symbol_instances {
            *local_use_counts
                .entry(instance.symbol_index)
                .or_insert(0usize) += 1;
        }
        let local_anchor_candidates: Vec<usize> = local_symbols
            .iter()
            .copied()
            .filter(|&symbol_index| {
                local_use_counts.get(&symbol_index).copied().unwrap_or(0) > 1
                    || self.should_keep_text_local_symbol(page, symbol_index)
            })
            .collect();
        PageResidualContext {
            page_num,
            page,
            local_use_counts,
            local_anchor_candidates,
            anchor_map,
            diagnostics_enabled,
        }
    }

    /// Places each of a page's local symbols, recording remaps and residuals,
    /// and returns the symbols that stay in the local dictionary.
    fn place_page_local_symbols(
        &self,
        context: &PageResidualContext<'_>,
        local_symbols: &[usize],
        anchor_remaps: &mut FxHashMap<usize, usize>,
        residual_symbols: &mut Vec<usize>,
        placement: &mut ResidualPlacementStats,
    ) -> Vec<usize> {
        let mut kept_local_symbols = Vec::with_capacity(local_symbols.len());
        for &symbol_index in local_symbols {
            match self.place_local_symbol(context, symbol_index, placement) {
                LocalSymbolPlacement::Keep => kept_local_symbols.push(symbol_index),
                LocalSymbolPlacement::Remap(anchor_index) => {
                    anchor_remaps.insert(symbol_index, anchor_index);
                }
                LocalSymbolPlacement::Residual => residual_symbols.push(symbol_index),
            }
        }
        kept_local_symbols
    }

    /// Decides where one page-local symbol is coded: symbols used more than
    /// once, and rescued text symbols, stay local; a one-off symbol moves to
    /// the best accepting local or global anchor, or else to the residual
    /// region.
    fn place_local_symbol(
        &self,
        context: &PageResidualContext<'_>,
        symbol_index: usize,
        placement: &mut ResidualPlacementStats,
    ) -> LocalSymbolPlacement {
        let page_num = context.page_num;
        let local_use_count = context
            .local_use_counts
            .get(&symbol_index)
            .copied()
            .unwrap_or(0);
        if local_use_count > 1 {
            return LocalSymbolPlacement::Keep;
        }
        if self.should_keep_text_local_symbol(context.page, symbol_index) {
            placement.local_rescue_count += 1;
            if placement.local_rescue_sample.len() < 16 {
                placement.local_rescue_sample.push((
                    page_num + 1,
                    symbol_index,
                    self.global_symbols[symbol_index].width,
                    self.global_symbols[symbol_index].height,
                ));
            }
            return LocalSymbolPlacement::Keep;
        }

        let local = self.local_residual_anchor_search(
            symbol_index,
            &context.local_anchor_candidates,
            &mut placement.anchor_comparator,
        );
        if let Some(anchor_choice) = local.best_anchor {
            placement.local_anchor_attach_count += 1;
            if anchor_choice.rescued_on_score {
                placement.local_score_rescue_count += 1;
            }
            if placement.local_anchor_attach_sample.len() < 16 {
                placement.local_anchor_attach_sample.push((
                    page_num + 1,
                    symbol_index,
                    anchor_choice.anchor_index,
                ));
            }
            return LocalSymbolPlacement::Remap(anchor_choice.anchor_index);
        }
        if context.diagnostics_enabled
            && matches!(local.best_reject, Some(SymUnifyAnchorDecision::RejectDim))
            && context
                .local_anchor_candidates
                .iter()
                .copied()
                .any(|anchor_index| {
                    anchor_index != symbol_index
                        && self.residual_symbol_accept_with_dim_limit(
                            symbol_index,
                            anchor_index,
                            &mut placement.anchor_comparator,
                            2,
                        )
                })
        {
            record_counterfactual_probe(
                &mut placement.counterfactual_local_dim_relax2,
                page_num,
                symbol_index,
                &self.global_symbols[symbol_index],
                self.symbol_pixel_counts[symbol_index],
            );
        }

        let global = match context.anchor_map {
            Some(anchor_map) => self.global_residual_anchor_search(
                symbol_index,
                anchor_map,
                &mut placement.anchor_comparator,
            ),
            None => ResidualAnchorSearch::default(),
        };
        if let Some(anchor_choice) = global.best_anchor {
            placement.anchor_attach_count += 1;
            if anchor_choice.rescued_on_score {
                placement.anchor_score_rescue_count += 1;
            }
            if placement.anchor_attach_sample.len() < 16 {
                placement.anchor_attach_sample.push((
                    page_num + 1,
                    symbol_index,
                    anchor_choice.anchor_index,
                ));
            }
            return LocalSymbolPlacement::Remap(anchor_choice.anchor_index);
        }
        if context.diagnostics_enabled
            && matches!(
                global.best_reject,
                Some(SymUnifyAnchorDecision::RejectOverlap)
            )
            && context.anchor_map.is_some_and(|anchor_map| {
                self.recovers_without_overlap_prescreen(
                    symbol_index,
                    anchor_map,
                    &mut placement.anchor_comparator,
                )
            })
        {
            record_counterfactual_probe(
                &mut placement.counterfactual_global_overlap_skip,
                page_num,
                symbol_index,
                &self.global_symbols[symbol_index],
                self.symbol_pixel_counts[symbol_index],
            );
        }
        placement.residual_symbol_traces.insert(
            symbol_index,
            ResidualSymbolTrace {
                page_num,
                local_use_count,
                had_local_candidates: local.had_candidates,
                had_global_candidates: global.had_candidates,
                local_best_reject: local.best_reject,
                global_best_reject: global.best_reject,
            },
        );
        LocalSymbolPlacement::Residual
    }

    /// Searches the page's own multiply-used symbols for an anchor.
    fn local_residual_anchor_search(
        &self,
        symbol_index: usize,
        local_anchor_candidates: &[usize],
        comparator: &mut Comparator,
    ) -> ResidualAnchorSearch {
        let mut search = ResidualAnchorSearch::default();
        for &anchor_index in local_anchor_candidates {
            if anchor_index == symbol_index {
                continue;
            }
            search.had_candidates = true;
            self.consider_residual_anchor(symbol_index, anchor_index, comparator, &mut search);
        }
        search
    }

    /// Searches the anchor-ready global symbols in the symbol's family
    /// bucket and its neighbours for an anchor.
    fn global_residual_anchor_search(
        &self,
        symbol_index: usize,
        anchor_map: &FxHashMap<FamilyBucketKey, Vec<usize>>,
        comparator: &mut Comparator,
    ) -> ResidualAnchorSearch {
        let bucket = family_bucket_key_for_symbol(
            &self.global_symbols[symbol_index],
            &self.symbol_signatures[symbol_index],
        );
        let mut visited = FxHashSet::default();
        let mut search = ResidualAnchorSearch::default();
        for neighbor in family_bucket_neighbors(bucket) {
            let Some(candidates) = anchor_map.get(&neighbor) else {
                continue;
            };
            for &anchor_index in candidates {
                if anchor_index == symbol_index || !visited.insert(anchor_index) {
                    continue;
                }
                search.had_candidates = true;
                self.consider_residual_anchor(symbol_index, anchor_index, comparator, &mut search);
            }
        }
        search
    }

    /// Scores one anchor for a residual symbol: an accepted match, or a
    /// score reject within the rescue slack, competes for best anchor; any
    /// other reject competes for best reject.
    fn consider_residual_anchor(
        &self,
        symbol_index: usize,
        anchor_index: usize,
        comparator: &mut Comparator,
        search: &mut ResidualAnchorSearch,
    ) {
        match self.residual_symbol_anchor_decision(symbol_index, anchor_index, comparator) {
            SymUnifyAnchorDecision::Accept { score, .. } => {
                self.maybe_update_best_sym_unify_anchor_candidate(
                    &mut search.best_anchor,
                    &self.global_symbols[symbol_index],
                    anchor_index,
                    score,
                    false,
                );
            }
            SymUnifyAnchorDecision::RejectScore { score, limit, .. }
                if score <= limit.saturating_add(self.config.sym_unify_score_rescue_slack) =>
            {
                self.maybe_update_best_sym_unify_anchor_candidate(
                    &mut search.best_anchor,
                    &self.global_symbols[symbol_index],
                    anchor_index,
                    score,
                    true,
                );
            }
            other => update_best_reject(&mut search.best_reject, other),
        }
    }

    /// Whether some global anchor would accept the symbol if the overlap
    /// prescreen were skipped.
    fn recovers_without_overlap_prescreen(
        &self,
        symbol_index: usize,
        anchor_map: &FxHashMap<FamilyBucketKey, Vec<usize>>,
        comparator: &mut Comparator,
    ) -> bool {
        let bucket = family_bucket_key_for_symbol(
            &self.global_symbols[symbol_index],
            &self.symbol_signatures[symbol_index],
        );
        let mut visited = FxHashSet::default();
        family_bucket_neighbors(bucket).into_iter().any(|neighbor| {
            anchor_map.get(&neighbor).is_some_and(|candidates| {
                candidates.iter().copied().any(|anchor_index| {
                    anchor_index != symbol_index
                        && visited.insert(anchor_index)
                        && self.residual_symbol_accept_without_overlap_prescreen(
                            symbol_index,
                            anchor_index,
                            comparator,
                        )
                })
            })
        })
    }

    /// Decides whether a page is coded as one generic region: a page with
    /// at most two instances whose local symbols gain nothing drops them, and
    /// a page with no symbol instance left in any dictionary has nothing for
    /// a text region to code.
    fn page_needs_generic_region(
        &self,
        page: &PageData,
        global_set: &HashSet<usize>,
        local_symbols: &mut Vec<usize>,
        anchor_remaps: &FxHashMap<usize, usize>,
        residual_symbols: &[usize],
    ) -> bool {
        // Only a page with at most two instances can fall back here, so
        // test that before the per-symbol gain scan over all instances.
        if local_symbols.len() <= 2
            && page.symbol_instances.len() <= 2
            && page.symbol_instances.iter().all(|inst| {
                !global_set.contains(&inst.symbol_index)
                    && !anchor_remaps.contains_key(&inst.symbol_index)
                    && !residual_symbols.contains(&inst.symbol_index)
            })
            && local_symbols
                .iter()
                .map(|&symbol_index| self.estimate_local_symbol_gain(page, symbol_index))
                .sum::<i64>()
                <= 0
        {
            local_symbols.clear();
            return true;
        }

        let has_kept_symbol_instances = page.symbol_instances.iter().any(|inst| {
            global_set.contains(&inst.symbol_index)
                || anchor_remaps.contains_key(&inst.symbol_index)
                || local_symbols.contains(&inst.symbol_index)
        });
        !has_kept_symbol_instances
    }

    /// Logs the residual and page-mode totals of planning and, under symbol
    /// unification, its rescue counters and counterfactual probes.
    fn record_residual_planning_diagnostics(
        &mut self,
        total_residual_symbols: usize,
        full_generic_pages: usize,
        placement: &ResidualPlacementStats,
        sym_unify_global_anchor_bytes: usize,
    ) {
        self.state.decision_debug_lines.push(format!(
            "planning residuals: {} page-local one-off symbols moved to generic residuals",
            total_residual_symbols
        ));
        self.state.decision_debug_lines.push(format!(
            "planning page modes: full_generic_pages={} text_pages={}",
            full_generic_pages,
            self.pages.len().saturating_sub(full_generic_pages)
        ));
        if self.config.lossy_symbol_mode != LossySymbolMode::SymbolUnify {
            return;
        }
        let lines = &mut self.state.decision_debug_lines;
        lines.push(format!(
            "sym_unify planning symbol rescues: local_kept={} local_anchor_remaps={} global_anchor_remaps={} local_score_rescues={} global_score_rescues={} anchor_ready_bytes={}",
            placement.local_rescue_count,
            placement.local_anchor_attach_count,
            placement.anchor_attach_count,
            placement.local_score_rescue_count,
            placement.anchor_score_rescue_count,
            sym_unify_global_anchor_bytes,
        ));
        if !placement.local_rescue_sample.is_empty() {
            lines.push(format!(
                "sym_unify planning local rescue sample: {:?}",
                placement.local_rescue_sample
            ));
        }
        if !placement.local_anchor_attach_sample.is_empty() {
            lines.push(format!(
                "sym_unify planning local-anchor sample: {:?}",
                placement.local_anchor_attach_sample
            ));
        }
        if !placement.anchor_attach_sample.is_empty() {
            lines.push(format!(
                "sym_unify planning anchor sample: {:?}",
                placement.anchor_attach_sample
            ));
        }
        let local_dim_relax2 = &placement.counterfactual_local_dim_relax2;
        let global_overlap_skip = &placement.counterfactual_global_overlap_skip;
        lines.push(format!(
            "sym_unify residual counterfactuals: local_dim_relax2_symbols={} local_dim_relax2_bitmap_proxy_bytes={} local_dim_relax2_pages={} global_overlap_skip_symbols={} global_overlap_skip_bitmap_proxy_bytes={} global_overlap_skip_pages={}",
            local_dim_relax2.symbol_count,
            local_dim_relax2.bitmap_proxy_bytes,
            local_dim_relax2.pages.len(),
            global_overlap_skip.symbol_count,
            global_overlap_skip.bitmap_proxy_bytes,
            global_overlap_skip.pages.len(),
        ));
        if !local_dim_relax2.samples.is_empty() {
            lines.push(format!(
                "sym_unify counterfactual local_dim_relax2 sample: {:?}",
                local_dim_relax2.samples
            ));
        }
        if !global_overlap_skip.samples.is_empty() {
            lines.push(format!(
                "sym_unify counterfactual global_overlap_skip sample: {:?}",
                global_overlap_skip.samples
            ));
        }
    }

    /// Logs a sample of the residual symbols of each of the first 32 pages.
    fn record_page_residual_samples(&mut self, page_residual_symbols: &[Vec<usize>]) {
        for (page_num, residuals) in page_residual_symbols.iter().enumerate().take(32) {
            if residuals.is_empty() {
                continue;
            }
            self.state.decision_debug_lines.push(format!(
                "page {} residual symbols: count={} sample={:?}",
                page_num + 1,
                residuals.len(),
                &residuals[..residuals.len().min(8)]
            ));
        }
    }

    /// The diagnostic lines that explain the residual symbols of a symbol
    /// unification plan: which could attach to an anchor, why each stayed
    /// residual, and what relaxed checks would have rescued.
    fn sym_unify_residual_probe_lines(
        &self,
        global_symbol_indices: &[usize],
        page_residual_symbols: &[Vec<usize>],
        residual_symbol_traces: &FxHashMap<usize, ResidualSymbolTrace>,
        sym_unify_global_anchor_map: Option<&FxHashMap<FamilyBucketKey, Vec<usize>>>,
    ) -> Vec<String> {
        let mut lines = Vec::new();
        let mut residual_unique = FxHashSet::default();
        for residuals in page_residual_symbols {
            residual_unique.extend(residuals.iter().copied());
        }
        let scan = self.residual_anchor_scan(global_symbol_indices, &residual_unique);
        push_residual_anchor_scan_lines(&mut lines, &scan, residual_unique.len());
        self.push_residual_reason_lines(&mut lines, residual_symbol_traces);
        let probes = self
            .residual_counterfactual_probes(residual_symbol_traces, sym_unify_global_anchor_map);
        push_residual_counterfactual_lines(&mut lines, probes);
        lines
    }

    /// Asks, for every residual symbol, whether it would attach to a current
    /// anchor, to one within the score rescue slack, or to any global symbol.
    fn residual_anchor_scan(
        &self,
        global_symbol_indices: &[usize],
        residual_unique: &FxHashSet<usize>,
    ) -> ResidualAnchorScan {
        let mut comparator = Comparator::default();
        let anchor_map = self.build_sym_unify_anchor_map(self.pages.len());
        let mut any_global_map: FxHashMap<FamilyBucketKey, Vec<usize>> = FxHashMap::default();
        for &symbol_index in global_symbol_indices {
            let key = family_bucket_key_for_symbol(
                &self.global_symbols[symbol_index],
                &self.symbol_signatures[symbol_index],
            );
            any_global_map.entry(key).or_default().push(symbol_index);
        }
        let mut scan = ResidualAnchorScan::default();
        let mut visited = FxHashSet::default();
        for residual_index in residual_unique.iter().copied() {
            let symbol = &self.global_symbols[residual_index];
            let area = symbol.width.saturating_mul(symbol.height);
            scan.area_buckets[residual_area_bucket(area)] += 1;
            let bucket =
                family_bucket_key_for_symbol(symbol, &self.symbol_signatures[residual_index]);
            match self.first_accepting_anchor(
                residual_index,
                bucket,
                &anchor_map,
                &mut visited,
                &mut comparator,
            ) {
                Ok(anchor_index) => scan.record_attached(residual_index, anchor_index),
                Err(best_reject) => {
                    *scan.reject_counts.entry(best_reject.label()).or_insert(0) += 1;
                    let rescued_anchor = self.first_score_rescue_anchor(
                        residual_index,
                        bucket,
                        &anchor_map,
                        &mut visited,
                        &mut comparator,
                    );
                    if let Some(anchor_index) = rescued_anchor {
                        scan.record_score_rescue(residual_index, anchor_index);
                    }
                }
            }

            let any_global = self.first_accepting_anchor(
                residual_index,
                bucket,
                &any_global_map,
                &mut visited,
                &mut comparator,
            );
            if let Ok(anchor_index) = any_global {
                scan.record_any_global(residual_index, anchor_index);
            }
        }
        scan
    }

    /// The first anchor in the residual's family bucket neighbourhood that
    /// accepts it, or else the highest-ranked reject seen.
    fn first_accepting_anchor(
        &self,
        residual_index: usize,
        bucket: FamilyBucketKey,
        anchor_map: &FxHashMap<FamilyBucketKey, Vec<usize>>,
        visited: &mut FxHashSet<usize>,
        comparator: &mut Comparator,
    ) -> std::result::Result<usize, SymUnifyAnchorDecision> {
        visited.clear();
        let mut best_reject = SymUnifyAnchorDecision::RejectDim;
        for neighbor in family_bucket_neighbors(bucket) {
            let Some(candidates) = anchor_map.get(&neighbor) else {
                continue;
            };
            for &anchor_index in candidates {
                if anchor_index == residual_index || !visited.insert(anchor_index) {
                    continue;
                }
                let decision =
                    self.residual_symbol_anchor_decision(residual_index, anchor_index, comparator);
                if matches!(decision, SymUnifyAnchorDecision::Accept { .. }) {
                    return Ok(anchor_index);
                }
                if decision.diagnostic_rank() > best_reject.diagnostic_rank() {
                    best_reject = decision;
                }
            }
        }
        Err(best_reject)
    }

    /// The first anchor that accepts the residual or rejects it on score
    /// within the rescue slack.
    fn first_score_rescue_anchor(
        &self,
        residual_index: usize,
        bucket: FamilyBucketKey,
        anchor_map: &FxHashMap<FamilyBucketKey, Vec<usize>>,
        visited: &mut FxHashSet<usize>,
        comparator: &mut Comparator,
    ) -> Option<usize> {
        visited.clear();
        for neighbor in family_bucket_neighbors(bucket) {
            let Some(candidates) = anchor_map.get(&neighbor) else {
                continue;
            };
            for &anchor_index in candidates {
                if anchor_index == residual_index || !visited.insert(anchor_index) {
                    continue;
                }
                match self.residual_symbol_anchor_decision(residual_index, anchor_index, comparator)
                {
                    SymUnifyAnchorDecision::Accept { .. } => return Some(anchor_index),
                    SymUnifyAnchorDecision::RejectScore { score, limit, .. }
                        if score
                            <= limit.saturating_add(self.config.sym_unify_score_rescue_slack) =>
                    {
                        return Some(anchor_index);
                    }
                    _ => {}
                }
            }
        }
        None
    }

    /// Groups the residual traces by reason and logs each reason's totals,
    /// largest bitmap cost first.
    fn push_residual_reason_lines(
        &self,
        lines: &mut Vec<String>,
        residual_symbol_traces: &FxHashMap<usize, ResidualSymbolTrace>,
    ) {
        let mut reason_stats: FxHashMap<ResidualReasonCode, ResidualReasonStats> =
            FxHashMap::default();
        for (&symbol_index, trace) in residual_symbol_traces {
            let reason = trace.reason_code();
            let stats = reason_stats.entry(reason).or_default();
            let symbol = &self.global_symbols[symbol_index];
            let instance_count = trace.local_use_count.max(1);
            stats.symbol_count += 1;
            stats.instance_count += instance_count;
            stats.black_pixels += self.symbol_pixel_counts[symbol_index] * instance_count;
            stats.bitmap_proxy_bytes += bitmap_proxy_bytes(symbol) * instance_count;
            stats.pages.insert(trace.page_num);
            match classify_residual_shape(symbol) {
                ResidualShapeKind::Tiny => stats.tiny_count += 1,
                ResidualShapeKind::PunctuationLike => stats.punctuation_like_count += 1,
                ResidualShapeKind::GlyphLike => stats.glyph_like_count += 1,
            }
            if stats.samples.len() < 8 {
                stats.samples.push((
                    trace.page_num + 1,
                    symbol_index,
                    symbol.width,
                    symbol.height,
                    trace.local_use_count,
                ));
            }
        }

        let mut sorted_reason_stats: Vec<_> = reason_stats.into_iter().collect();
        sorted_reason_stats.sort_by(|lhs, rhs| {
            rhs.1
                .bitmap_proxy_bytes
                .cmp(&lhs.1.bitmap_proxy_bytes)
                .then_with(|| rhs.1.symbol_count.cmp(&lhs.1.symbol_count))
                .then_with(|| lhs.0.label().cmp(rhs.0.label()))
        });
        let total_reason_proxy_bytes: usize = sorted_reason_stats
            .iter()
            .map(|(_, stats)| stats.bitmap_proxy_bytes)
            .sum();
        lines.push(format!(
            "sym_unify residual reason summary: reasons={} residual_symbols={} bitmap_proxy_bytes={}",
            sorted_reason_stats.len(),
            residual_symbol_traces.len(),
            total_reason_proxy_bytes,
        ));
        for (reason, stats) in sorted_reason_stats {
            lines.push(format!(
                "  residual reason {}: symbols={} instances={} pages={} black_pixels={} bitmap_proxy_bytes={} tiny={} punct_like={} glyph_like={} sample={:?}",
                reason.label(),
                stats.symbol_count,
                stats.instance_count,
                stats.pages.len(),
                stats.black_pixels,
                stats.bitmap_proxy_bytes,
                stats.tiny_count,
                stats.punctuation_like_count,
                stats.glyph_like_count,
                stats.samples
            ));
        }
    }

    /// Runs the counterfactual probe that matches each residual's reason:
    /// overlap and compare rejects against global anchors, dimension rejects
    /// against symbols of other pages.
    fn residual_counterfactual_probes(
        &self,
        residual_symbol_traces: &FxHashMap<usize, ResidualSymbolTrace>,
        sym_unify_global_anchor_map: Option<&FxHashMap<FamilyBucketKey, Vec<usize>>>,
    ) -> ResidualCounterfactualProbes {
        let mut symbol_home_page = vec![usize::MAX; self.global_symbols.len()];
        for (page_idx, symbols) in self.page_symbol_indices.iter().enumerate() {
            for &symbol_index in symbols {
                if symbol_home_page[symbol_index] == usize::MAX {
                    symbol_home_page[symbol_index] = page_idx;
                }
            }
        }
        let mut all_symbol_bucket_map: FxHashMap<FamilyBucketKey, Vec<usize>> =
            FxHashMap::default();
        for (symbol_index, symbol) in self.global_symbols.iter().enumerate() {
            let key = family_bucket_key_for_symbol(symbol, &self.symbol_signatures[symbol_index]);
            all_symbol_bucket_map
                .entry(key)
                .or_default()
                .push(symbol_index);
        }

        let mut probes = ResidualCounterfactualProbes::default();
        for (&symbol_index, trace) in residual_symbol_traces {
            match trace.reason_code() {
                ResidualReasonCode::UseCountOneGlobalRejectOverlap => self.probe_overlap_bypass(
                    symbol_index,
                    trace,
                    sym_unify_global_anchor_map,
                    &mut probes,
                ),
                ResidualReasonCode::UseCountOneGlobalRejectCompare => self.probe_global_compare(
                    symbol_index,
                    trace,
                    sym_unify_global_anchor_map,
                    &mut probes,
                ),
                ResidualReasonCode::UseCountOneLocalRejectDim => {
                    let found = self.cross_page_match(
                        symbol_index,
                        trace.page_num,
                        &symbol_home_page,
                        &all_symbol_bucket_map,
                        &mut probes.cross_page_comparator,
                    );
                    let stats = match found {
                        CrossPageMatch::Current => &mut probes.local_dim_cross_page_current,
                        CrossPageMatch::DimRelaxed => &mut probes.local_dim_cross_page_dim2,
                        CrossPageMatch::None => continue,
                    };
                    record_counterfactual_probe(
                        stats,
                        trace.page_num,
                        symbol_index,
                        &self.global_symbols[symbol_index],
                        self.symbol_pixel_counts[symbol_index],
                    );
                }
                _ => {}
            }
        }
        probes
    }

    /// Whether the anchor is used often enough, or on enough pages, to be
    /// given one extra unit of error allowance.
    fn is_strong_sym_unify_anchor(&self, anchor_index: usize) -> bool {
        self.symbol_usage[anchor_index] >= SYM_UNIFY_STRONG_ANCHOR_MIN_USAGE
            || self.symbol_page_count[anchor_index] >= SYM_UNIFY_STRONG_ANCHOR_MIN_PAGE_SPAN
    }

    /// Retries a residual rejected on overlap without the overlap prescreen,
    /// and records the outcome and, for a compare reject, how far the best
    /// relaxed comparison missed.
    fn probe_overlap_bypass(
        &self,
        symbol_index: usize,
        trace: &ResidualSymbolTrace,
        anchor_map: Option<&FxHashMap<FamilyBucketKey, Vec<usize>>>,
        probes: &mut ResidualCounterfactualProbes,
    ) {
        let outcome = match anchor_map {
            Some(anchor_map) => self.overlap_bypass_search(symbol_index, anchor_map, probes),
            None => OverlapBypassOutcome::default(),
        };
        let label = if outcome.recovered {
            "accept"
        } else {
            outcome
                .best_reject
                .map(SymUnifyAnchorDecision::label)
                .unwrap_or("no_candidates")
        };
        record_labeled_counterfactual_probe(
            &mut probes.overlap_bypass_outcomes,
            label,
            trace.page_num,
            symbol_index,
            &self.global_symbols[symbol_index],
            self.symbol_pixel_counts[symbol_index],
        );
        if label != "compare" {
            return;
        }
        let Some(detail) = outcome.best_compare else {
            return;
        };
        self.record_compare_slack_probes(
            trace.page_num,
            symbol_index,
            detail,
            &mut probes.from_overlap_compare,
        );
    }

    /// Searches the global anchors with the overlap prescreen skipped,
    /// stopping at the first accept and keeping the relaxed comparison with
    /// the least total error among compare rejects.
    fn overlap_bypass_search(
        &self,
        symbol_index: usize,
        anchor_map: &FxHashMap<FamilyBucketKey, Vec<usize>>,
        probes: &mut ResidualCounterfactualProbes,
    ) -> OverlapBypassOutcome {
        let bucket = family_bucket_key_for_symbol(
            &self.global_symbols[symbol_index],
            &self.symbol_signatures[symbol_index],
        );
        let mut visited = FxHashSet::default();
        let mut outcome = OverlapBypassOutcome::default();
        for neighbor in family_bucket_neighbors(bucket) {
            let Some(candidates) = anchor_map.get(&neighbor) else {
                continue;
            };
            for &anchor_index in candidates {
                if anchor_index == symbol_index || !visited.insert(anchor_index) {
                    continue;
                }
                let strong_anchor = self.is_strong_sym_unify_anchor(anchor_index);
                match self.residual_symbol_anchor_decision_without_overlap_prescreen(
                    symbol_index,
                    anchor_index,
                    &mut probes.overlap_bypass_comparator,
                ) {
                    SymUnifyAnchorDecision::Accept { .. } => {
                        outcome.recovered = true;
                        return outcome;
                    }
                    SymUnifyAnchorDecision::RejectCompare => {
                        update_best_reject(
                            &mut outcome.best_reject,
                            SymUnifyAnchorDecision::RejectCompare,
                        );
                        self.keep_least_error_relaxed_compare(
                            symbol_index,
                            anchor_index,
                            strong_anchor,
                            &mut probes.overlap_compare_probe_comparator,
                            &mut outcome.best_compare,
                        );
                    }
                    other => update_best_reject(&mut outcome.best_reject, other),
                }
            }
        }
        outcome
    }

    /// The compare error limit a relaxed probe judges against.
    fn probe_compare_max_err(&self, strong_anchor: bool) -> u32 {
        self.config
            .sym_unify_max_err
            .max(4)
            .saturating_add(u32::from(strong_anchor))
    }

    /// The outside-ink limit a relaxed probe judges against.
    fn probe_outside_ink_limit(&self, strong_anchor: bool) -> u32 {
        self.config
            .sym_unify_max_border_outside_ink
            .min(1)
            .saturating_add(u32::from(strong_anchor))
    }

    /// Compares the residual with the anchor under a relaxed error limit and
    /// keeps the result if it has less total error than the best so far.
    fn keep_least_error_relaxed_compare(
        &self,
        symbol_index: usize,
        anchor_index: usize,
        strong_anchor: bool,
        comparator: &mut Comparator,
        best: &mut Option<CompareProbeDetail>,
    ) {
        let candidate = &self.global_symbols[symbol_index];
        let proto = &self.global_symbols[anchor_index];
        let compare_max_err = self.probe_compare_max_err(strong_anchor);
        let Some(result) = comparator.compare_for_symbol_unify(
            candidate,
            proto,
            relaxed_compare_probe_max_err(candidate, proto),
            self.config.sym_unify_max_dx.max(0),
            self.config.sym_unify_max_dy.max(0),
        ) else {
            return;
        };
        let exact_dims = candidate.width == proto.width && candidate.height == proto.height;
        if best.is_none_or(|(current, _, _, _)| result.total_err < current.total_err) {
            *best = Some((result, compare_max_err, exact_dims, strong_anchor));
        }
    }

    /// Records a missed comparison in detail, and whether two or four more
    /// units of total error would have accepted it.
    fn record_compare_slack_probes(
        &self,
        page_num: usize,
        symbol_index: usize,
        detail: CompareProbeDetail,
        slack_probes: &mut CompareSlackProbes,
    ) {
        let (result, compare_max_err, exact_dims, strong_anchor) = detail;
        let outside_limit = self.probe_outside_ink_limit(strong_anchor);
        let score_limit = self.config.sym_unify_class_accept_limit + u32::from(strong_anchor);
        let score = Self::symbol_unify_assignment_score(&result);
        record_detailed_compare_probe(
            &mut slack_probes.total_err_details,
            page_num,
            symbol_index,
            &self.global_symbols[symbol_index],
            result,
            compare_max_err,
            exact_dims,
            strong_anchor,
        );
        if result.total_err <= compare_max_err.saturating_add(2)
            && result.outside_ink_err <= outside_limit
            && score <= score_limit
        {
            record_counterfactual_probe(
                &mut slack_probes.slack2,
                page_num,
                symbol_index,
                &self.global_symbols[symbol_index],
                self.symbol_pixel_counts[symbol_index],
            );
        }
        if result.total_err <= compare_max_err.saturating_add(4)
            && result.outside_ink_err <= outside_limit
            && score <= score_limit
        {
            record_counterfactual_probe(
                &mut slack_probes.slack4,
                page_num,
                symbol_index,
                &self.global_symbols[symbol_index],
                self.symbol_pixel_counts[symbol_index],
            );
        }
    }

    /// Compares a residual rejected on compare with every global anchor
    /// under a relaxed limit, and records why the closest one still failed.
    fn probe_global_compare(
        &self,
        symbol_index: usize,
        trace: &ResidualSymbolTrace,
        anchor_map: Option<&FxHashMap<FamilyBucketKey, Vec<usize>>>,
        probes: &mut ResidualCounterfactualProbes,
    ) {
        let mut best = GlobalCompareProbe {
            label: "no_candidates",
            total_err: u32::MAX,
            detail: None,
        };
        if let Some(anchor_map) = anchor_map {
            self.global_compare_probe_search(
                symbol_index,
                anchor_map,
                &mut probes.overlap_compare_probe_comparator,
                &mut best,
            );
        }

        record_labeled_counterfactual_probe(
            &mut probes.overlap_compare_probe_outcomes,
            best.label,
            trace.page_num,
            symbol_index,
            &self.global_symbols[symbol_index],
            self.symbol_pixel_counts[symbol_index],
        );
        if best.label != "total_err" {
            return;
        }
        let Some(detail) = best.detail else {
            return;
        };
        self.record_compare_slack_probes(
            trace.page_num,
            symbol_index,
            detail,
            &mut probes.from_global_compare,
        );
    }

    /// Visits every global anchor near the residual's family bucket and
    /// keeps the relaxed comparison with the least total error.
    fn global_compare_probe_search(
        &self,
        symbol_index: usize,
        anchor_map: &FxHashMap<FamilyBucketKey, Vec<usize>>,
        comparator: &mut Comparator,
        best: &mut GlobalCompareProbe,
    ) {
        let bucket = family_bucket_key_for_symbol(
            &self.global_symbols[symbol_index],
            &self.symbol_signatures[symbol_index],
        );
        let mut visited = FxHashSet::default();
        for neighbor in family_bucket_neighbors(bucket) {
            let Some(candidates) = anchor_map.get(&neighbor) else {
                continue;
            };
            for &anchor_index in candidates {
                if anchor_index == symbol_index || !visited.insert(anchor_index) {
                    continue;
                }
                self.update_global_compare_probe(symbol_index, anchor_index, comparator, best);
            }
        }
    }

    /// Labels one relaxed comparison of the residual with an anchor and
    /// keeps it if it has the least total error so far.
    fn update_global_compare_probe(
        &self,
        symbol_index: usize,
        anchor_index: usize,
        comparator: &mut Comparator,
        best: &mut GlobalCompareProbe,
    ) {
        let candidate = &self.global_symbols[symbol_index];
        let proto = &self.global_symbols[anchor_index];
        let strong_anchor = self.is_strong_sym_unify_anchor(anchor_index);
        let compare_max_err = self.probe_compare_max_err(strong_anchor);
        let outside_limit = self.probe_outside_ink_limit(strong_anchor);
        let relaxed = comparator.compare_for_symbol_unify(
            candidate,
            proto,
            relaxed_compare_probe_max_err(candidate, proto),
            self.config.sym_unify_max_dx.max(0),
            self.config.sym_unify_max_dy.max(0),
        );
        let (label, total_err) = match relaxed {
            Some(result) => (
                self.relaxed_compare_label(&result, compare_max_err, outside_limit, strong_anchor),
                result.total_err,
            ),
            None => ("relaxed_none", u32::MAX),
        };

        if total_err < best.total_err {
            best.total_err = total_err;
            best.label = label;
            let exact_dims = candidate.width == proto.width && candidate.height == proto.height;
            best.detail =
                relaxed.map(|result| (result, compare_max_err, exact_dims, strong_anchor));
        } else if best.total_err == u32::MAX && best.label == "no_candidates" {
            best.label = label;
        }
    }

    /// Names the first limit a relaxed comparison breaks, or "accept".
    fn relaxed_compare_label(
        &self,
        result: &crate::jbig2comparator::CompareResult,
        compare_max_err: u32,
        outside_limit: u32,
        strong_anchor: bool,
    ) -> &'static str {
        let score = Self::symbol_unify_assignment_score(result);
        let score_limit = self.config.sym_unify_class_accept_limit + u32::from(strong_anchor);
        match (
            result.total_err <= compare_max_err,
            result.outside_ink_err > outside_limit,
        ) {
            (true, true) => "outside_ink",
            (true, false) if score > score_limit => "score",
            (true, false) => "accept",
            (false, true) => "total_err+outside_ink",
            (false, false) => "total_err",
        }
    }

    /// Whether a residual rejected on dimensions matches a symbol first seen
    /// on another page, as things stand or with the dimension limit relaxed
    /// to two.
    fn cross_page_match(
        &self,
        symbol_index: usize,
        page_num: usize,
        symbol_home_page: &[usize],
        all_symbol_bucket_map: &FxHashMap<FamilyBucketKey, Vec<usize>>,
        comparator: &mut Comparator,
    ) -> CrossPageMatch {
        let bucket = family_bucket_key_for_symbol(
            &self.global_symbols[symbol_index],
            &self.symbol_signatures[symbol_index],
        );
        let mut visited = FxHashSet::default();
        let mut found_dim2 = false;
        for neighbor in family_bucket_neighbors(bucket) {
            let Some(candidates) = all_symbol_bucket_map.get(&neighbor) else {
                continue;
            };
            for &candidate_index in candidates {
                if candidate_index == symbol_index
                    || !visited.insert(candidate_index)
                    || symbol_home_page[candidate_index] == page_num
                {
                    continue;
                }
                if self.residual_symbol_matches_anchor(symbol_index, candidate_index, comparator) {
                    return CrossPageMatch::Current;
                }
                if self.residual_symbol_accept_with_dim_limit(
                    symbol_index,
                    candidate_index,
                    comparator,
                    2,
                ) {
                    found_dim2 = true;
                }
            }
        }
        if found_dim2 {
            CrossPageMatch::DimRelaxed
        } else {
            CrossPageMatch::None
        }
    }

    /// Plans and encodes the global symbol dictionary, assigning its segment
    /// number, and returns its segments, its encoding and the refinement
    /// plans of its family members.
    fn encode_planned_global_dictionary(
        &mut self,
        global_symbol_indices: &[usize],
        diagnostics_enabled: bool,
        current_segment_number: &mut u32,
    ) -> Result<(
        Vec<Segment>,
        EncodedSymbolDictionary,
        Vec<Option<RefinementPlan>>,
    )> {
        let mut global_segments = Vec::new();
        self.global_dict_segment_numbers.clear();
        let mut global_refinement_map = vec![None; self.global_symbols.len()];
        if global_symbol_indices.is_empty() {
            return Ok((
                global_segments,
                EncodedSymbolDictionary::default(),
                global_refinement_map,
            ));
        }
        let refs: Vec<&BitImage> = global_symbol_indices
            .iter()
            .map(|&i| &self.global_symbols[i])
            .collect();
        let dict_usage: Vec<usize> = global_symbol_indices
            .iter()
            .map(|&i| self.symbol_usage[i])
            .collect();
        let dict_layout = plan_symbol_dictionary_layout(&refs, &self.config, Some(&dict_usage))?;
        if diagnostics_enabled {
            self.state.decision_debug_lines.push(format!(
                "global dict layout: families={} singletons={} refined_members={} exported_members={}",
                dict_layout.diagnostics.family_count,
                dict_layout.diagnostics.singleton_family_count,
                dict_layout.diagnostics.refined_member_count,
                dict_layout.diagnostics.exported_member_count
            ));
            self.state.decision_debug_lines.extend(
                dict_layout
                    .diagnostics
                    .sample_lines
                    .iter()
                    .take(64)
                    .cloned(),
            );
        }
        let dict_start = Instant::now();
        let encoded_global_dict =
            encode_symbol_dictionary_segments(&refs, &self.config, &dict_layout)?;
        self.metrics.symbol_mode.symbol_dict_encoding += dict_start.elapsed();
        record_refinement_plans(
            &mut global_refinement_map,
            global_symbol_indices,
            &dict_layout.refinements,
        );
        let segment_number = *current_segment_number;
        *current_segment_number += 1;
        self.global_dict_segment_numbers.push(segment_number);
        global_segments.push(Segment {
            number: segment_number,
            seg_type: SegmentType::SymbolDictionary,
            deferred_non_retain: false,
            retain_flags: 0,
            page_association_type: 2,
            referred_to: Vec::new(),
            page: None,
            payload: encoded_global_dict.payload.clone(),
        });
        Ok((global_segments, encoded_global_dict, global_refinement_map))
    }

    /// Plans the page's local symbol dictionary, weighting each symbol by
    /// its uses on the page; `None` when the page codes no local symbols.
    fn page_local_dictionary_layout(
        &self,
        page: &PageData,
        local_symbols: &[usize],
    ) -> Result<Option<SymbolDictLayout>> {
        if !self.config.symbol_mode || page.symbol_instances.is_empty() || local_symbols.is_empty()
        {
            return Ok(None);
        }
        let refs: Vec<&BitImage> = local_symbols
            .iter()
            .map(|&i| &self.global_symbols[i])
            .collect();
        let mut local_usage = vec![0usize; local_symbols.len()];
        let local_index_by_symbol: HashMap<usize, usize> = local_symbols
            .iter()
            .enumerate()
            .map(|(idx, &symbol_index)| (symbol_index, idx))
            .collect();
        for instance in &page.symbol_instances {
            if let Some(&local_idx) = local_index_by_symbol.get(&instance.symbol_index) {
                local_usage[local_idx] += 1;
            }
        }
        Ok(Some(plan_symbol_dictionary_layout(
            &refs,
            &self.config,
            Some(&local_usage),
        )?))
    }

    /// Builds every planned page, in parallel across pages when the
    /// `parallel` feature is on and there is more than one independent page.
    fn build_planned_pages(
        &self,
        page_layouts: &[PlannedPageLayout],
        global_sym_to_dict_pos: &[u32],
        num_global_dict_symbols: u32,
        global_refinement_map: &[Option<RefinementPlan>],
    ) -> Vec<Result<BuiltPage>> {
        #[cfg(feature = "parallel")]
        if self.state.pdf_mode || self.pages.len() > 1 {
            return page_layouts
                .par_iter()
                .map(|layout| {
                    self.build_planned_page(
                        layout,
                        global_sym_to_dict_pos,
                        num_global_dict_symbols,
                        global_refinement_map,
                    )
                })
                .collect::<Vec<_>>();
        }

        page_layouts
            .iter()
            .map(|layout| {
                self.build_planned_page(
                    layout,
                    global_sym_to_dict_pos,
                    num_global_dict_symbols,
                    global_refinement_map,
                )
            })
            .collect::<Vec<_>>()
    }

    pub(crate) fn build_planned_page(
        &self,
        layout: &PlannedPageLayout,
        global_sym_to_dict_pos: &[u32],
        num_global_dict_symbols: u32,
        global_refinement_map: &[Option<RefinementPlan>],
    ) -> Result<BuiltPage> {
        let page = &self.pages[layout.page_index];
        let mut page_segments = Vec::new();
        let mut resources = super::super::PageResourceMetrics {
            page_pixels: super::super::resources::pixels(&page.image),
            ..Default::default()
        };
        let mut symbol_dict_time = Duration::default();
        let mut text_region_time = Duration::default();
        let mut generic_region_time = Duration::default();
        #[cfg(feature = "symboldict")]
        let mut topology_refinement_count = 0;
        #[cfg(not(feature = "symboldict"))]
        let topology_refinement_count = 0;

        let page_info_payload = PageInfo {
            width: page.image.width as u32,
            height: page.image.height as u32,
            default_pixel: false,
            // Keep the legacy generic API's page flags byte-identical.
            is_lossless: self.config.is_lossless && self.config.symbol_mode,
            xres: self.config.generic.dpi,
            yres: self.config.generic.dpi,
            ..Default::default()
        }
        .to_bytes();

        page_segments.push(Segment {
            number: layout.page_info_segment_number,
            seg_type: SegmentType::PageInformation,
            deferred_non_retain: false,
            retain_flags: 0,
            page_association_type: 0,
            referred_to: vec![],
            page: Some(layout.page_number),
            payload: page_info_payload,
        });

        if self.config.symbol_mode
            && !page.symbol_instances.is_empty()
            && !layout.use_generic_region
        {
            // Refer to the actual dictionaries present, including when a PDF
            // page has only a local dictionary and no /JBIG2Globals stream.
            let mut referred_to_for_text_region = self.global_dict_segment_numbers.clone();
            let residual_set: HashSet<usize> = layout.residual_symbols.iter().copied().collect();
            let residual_anchor_remaps = &layout.residual_anchor_remaps;

            let mut local_sym_to_dict_pos = vec![u32::MAX; self.global_symbols.len()];
            let mut local_refinement_map = vec![None; self.global_symbols.len()];
            let num_local_dict_symbols = match &layout.local_dict_layout {
                Some(local_dict_layout) => {
                    let dict_start = Instant::now();
                    let encoded_local_dict =
                        self.symbol_dictionary_for(&layout.local_symbols, local_dict_layout)?;
                    symbol_dict_time += dict_start.elapsed();

                    fill_dictionary_positions(
                        &mut local_sym_to_dict_pos,
                        &layout.local_symbols,
                        &encoded_local_dict,
                    );
                    (
                        resources.local_dictionary_bitmap_bytes,
                        resources.local_dictionary_pixels,
                    ) = super::super::resources::dictionary_storage(
                        &self.global_symbols,
                        &local_sym_to_dict_pos,
                    );
                    record_refinement_plans(
                        &mut local_refinement_map,
                        &layout.local_symbols,
                        &local_dict_layout.refinements,
                    );

                    page_segments.extend(local_dictionary_segments(
                        layout,
                        &encoded_local_dict.payload,
                    ));
                    referred_to_for_text_region
                        .extend(layout.local_dict_segment_numbers.iter().copied());
                    encoded_local_dict.exported_symbol_count
                }
                None => 0,
            };

            let mut planned_instances = Vec::with_capacity(page.symbol_instances.len());
            let mut residual_instances = Vec::new();
            for instance in &page.symbol_instances {
                if let Some(&anchor_index) = residual_anchor_remaps.get(&instance.symbol_index) {
                    let mut remapped = instance.clone();
                    remapped.symbol_index = anchor_index;
                    remapped.needs_refinement = false;
                    remapped.refinement_dx = 0;
                    remapped.refinement_dy = 0;
                    planned_instances.push(remapped);
                } else if residual_set.contains(&instance.symbol_index) {
                    residual_instances.push(instance);
                } else {
                    planned_instances.push(instance.clone());
                }
            }
            let mut needs_family_refinement = false;
            for instance in &mut planned_instances {
                if let Some(refinement) = local_refinement_map[instance.symbol_index] {
                    instance.symbol_index = refinement.prototype_input_index;
                    instance.needs_refinement = true;
                    instance.refinement_dx = refinement.refinement_dx;
                    instance.refinement_dy = refinement.refinement_dy;
                    needs_family_refinement = true;
                } else if let Some(refinement) = global_refinement_map[instance.symbol_index] {
                    instance.symbol_index = refinement.prototype_input_index;
                    instance.needs_refinement = true;
                    instance.refinement_dx = refinement.refinement_dx;
                    instance.refinement_dy = refinement.refinement_dy;
                    needs_family_refinement = true;
                }
                #[cfg(feature = "symboldict")]
                if self.config.topology_safe_substitution
                    && !instance.needs_refinement
                    && !super::topology::substitution_preserves_topology(
                        &instance.instance_bitmap,
                        &self.global_symbols[instance.symbol_index],
                    )
                {
                    // This guard runs after every remapping stage, including
                    // clustering, unification and residual-anchor rescue.
                    instance.needs_refinement = true;
                    instance.refinement_dx = 0;
                    instance.refinement_dy = 0;
                    topology_refinement_count += 1;
                }
            }

            for instance in &planned_instances {
                let source_area = super::super::resources::pixels(&instance.instance_bitmap);
                resources.text_work_pixels = resources.text_work_pixels.saturating_add(source_area);
                if instance.needs_refinement {
                    resources.refined_instances += 1;
                    resources.text_work_pixels =
                        resources
                            .text_work_pixels
                            .saturating_add(super::super::resources::pixels(
                                &self.global_symbols[instance.symbol_index],
                            ));
                } else if instance.instance_bitmap == self.global_symbols[instance.symbol_index] {
                    resources.exact_instances += 1;
                } else {
                    resources.substituted_instances += 1;
                }
            }
            resources.generic_instances = residual_instances.len();
            resources.generic_source_ink_pixels = residual_instances
                .iter()
                .map(|instance| instance.instance_bitmap.count_ones() as u64)
                .sum();

            // NOTE: PDF-embedded text regions previously stripped all per-instance
            // refinement here ("until global dictionary encoding is fixed"). That
            // turned every lossy symbol match into a hard substitution (e.g. "i"
            // rendered as "'"). Per-instance refinement (SBREFINE=1) round-trips
            // correctly through jbig2dec in embedded/PDF mode, so the refinement is
            // now retained and the lossy match is corrected at decode time.

            if !planned_instances.is_empty() {
                let text_start = Instant::now();
                let has_instance_refinement = planned_instances
                    .iter()
                    .any(|instance| instance.needs_refinement);
                let use_refinement_text_region = has_instance_refinement
                    || (!self.config.uses_lossy_symbol_dictionary()
                        && (self.config.text_refine
                            || self.config.refine
                            || needs_family_refinement));
                let dictionaries = TextRegionDictionaries {
                    global_sym_to_dict_pos,
                    num_global_dict_symbols,
                    local_sym_to_dict_pos: &local_sym_to_dict_pos,
                    num_local_dict_symbols,
                };
                let region_payload = self.page_text_region_payload(
                    layout,
                    &planned_instances,
                    use_refinement_text_region,
                    &dictionaries,
                )?;
                text_region_time += text_start.elapsed();

                page_segments.push(Segment {
                    number: layout.region_segment_number,
                    seg_type: if self.config.is_lossless && self.config.symbol_mode {
                        SegmentType::ImmediateLosslessTextRegion
                    } else {
                        SegmentType::ImmediateTextRegion
                    },
                    deferred_non_retain: false,
                    retain_flags: 0,
                    page_association_type: 0,
                    referred_to: referred_to_for_text_region,
                    page: Some(layout.page_number),
                    payload: region_payload,
                });
            }

            let generic_start = Instant::now();
            page_segments.extend(
                self.encode_residual_regions(
                    &residual_instances,
                    layout
                        .residual_region_segment_number
                        .unwrap_or(layout.region_segment_number),
                    layout.page_number,
                    if layout.residual_region_segment_number.is_some() {
                        MAX_RESIDUAL_REGIONS
                    } else {
                        1
                    },
                )?,
            );
            generic_region_time += generic_start.elapsed();
        } else {
            resources.generic_instances = page.symbol_instances.len();
            resources.generic_source_ink_pixels = page.image.count_ones() as u64;
            let generic_start = Instant::now();
            let generic_region_payload = self.page_generic_payload(layout.page_index)?;
            generic_region_time += generic_start.elapsed();

            page_segments.push(Segment {
                number: layout.region_segment_number,
                seg_type: if self.config.is_lossless && self.config.symbol_mode {
                    SegmentType::ImmediateLosslessGenericRegion
                } else {
                    SegmentType::ImmediateGenericRegion
                },
                deferred_non_retain: false,
                retain_flags: 0,
                page_association_type: 0,
                referred_to: Vec::new(),
                page: Some(layout.page_number),
                payload: generic_region_payload,
            });
        }

        resources.generic_region_pixels = page_segments
            .iter()
            .filter(|segment| {
                matches!(
                    segment.seg_type,
                    SegmentType::ImmediateGenericRegion
                        | SegmentType::ImmediateLosslessGenericRegion
                )
            })
            .map(|segment| {
                let width = u32::from_be_bytes(segment.payload[0..4].try_into().unwrap()) as u64;
                let height = u32::from_be_bytes(segment.payload[4..8].try_into().unwrap()) as u64;
                width.saturating_mul(height)
            })
            .sum();

        // For PDF split mode, skip EndOfPage segment - not needed for PDF embedding
        if !self.state.pdf_mode {
            page_segments.push(Segment {
                number: layout.end_of_page_segment_number,
                seg_type: SegmentType::EndOfPage,
                deferred_non_retain: false,
                retain_flags: 0,
                page_association_type: 0,
                referred_to: Vec::new(),
                page: Some(layout.page_number),
                payload: Vec::new(),
            });
        }

        Ok(BuiltPage {
            page: PlannedPage {
                page_number: layout.page_number,
                segments: page_segments,
                resources,
            },
            symbol_dict_time,
            text_region_time,
            generic_region_time,
            topology_refinement_count,
        })
    }

    /// Codes a page's text region, or returns the payload an earlier trial
    /// coded for the same input.
    fn page_text_region_payload(
        &self,
        layout: &PlannedPageLayout,
        planned_instances: &[SymbolInstance],
        use_refinement_text_region: bool,
        dictionaries: &TextRegionDictionaries<'_>,
    ) -> Result<Vec<u8>> {
        // Within one measured planning call the same page is coded
        // with the same positions by several trials; the memo returns
        // the earlier payload for an identical input.
        let memo_key = super::memo::TextRegionKey::new(
            layout.page_index,
            use_refinement_text_region,
            dictionaries.num_global_dict_symbols + dictionaries.num_local_dict_symbols,
            planned_instances,
            dictionaries.global_sym_to_dict_pos,
            dictionaries.local_sym_to_dict_pos,
        );
        if let Some(payload) = self.cached_text_region(&memo_key) {
            return Ok(payload);
        }
        let payload = if use_refinement_text_region {
            encode_text_region_with_refinement(
                planned_instances,
                &self.config,
                &self.global_symbols,
                dictionaries.global_sym_to_dict_pos,
                dictionaries.num_global_dict_symbols,
                dictionaries.local_sym_to_dict_pos,
                dictionaries.num_local_dict_symbols,
            )?
        } else {
            encode_text_region_mapped(
                planned_instances,
                &self.config,
                &self.global_symbols,
                dictionaries.global_sym_to_dict_pos,
                dictionaries.num_global_dict_symbols,
                dictionaries.local_sym_to_dict_pos,
                layout.page_index,
                dictionaries.num_local_dict_symbols,
            )?
        };
        self.remember_text_region(memo_key, &payload);
        Ok(payload)
    }

    pub(crate) fn validate_plan(&self, plan: &PlannedDocument) -> Result<()> {
        let mut global_numbers = HashSet::new();

        for seg in &plan.global_segments {
            if !global_numbers.insert(seg.number) {
                anyhow::bail!("Duplicate segment number in globals: {}", seg.number);
            }
        }

        for (page_idx, page) in plan.pages.iter().enumerate() {
            // In PDF mode, each page is an independent stream, so segment
            // numbers only need to be unique within globals + that page.
            let mut page_numbers = global_numbers.clone();
            for seg in &page.segments {
                if !page_numbers.insert(seg.number) {
                    anyhow::bail!(
                        "Duplicate segment number {} on page {}",
                        seg.number,
                        page_idx
                    );
                }
            }

            for seg in &page.segments {
                check_page_segment_references(
                    page.page_number,
                    seg,
                    &page_numbers,
                    &global_numbers,
                    plan.global_segments.is_empty(),
                )?;
            }
        }

        if let Some(eof) = &plan.eof_segment
            && global_numbers.contains(&eof.number)
        {
            anyhow::bail!("EOF segment number {} conflicts with globals", eof.number);
        }

        for seg in &plan.global_segments {
            for referred in &seg.referred_to {
                if !global_numbers.contains(referred) {
                    anyhow::bail!(
                        "Global segment {} refers to missing segment {}",
                        seg.number,
                        referred
                    );
                }
            }
        }

        Ok(())
    }
}

/// Checks that every segment a page segment refers to exists in the page's
/// stream or in a globals stream that is actually emitted.
fn check_page_segment_references(
    page_number: u32,
    seg: &Segment,
    page_numbers: &HashSet<u32>,
    global_numbers: &HashSet<u32>,
    no_globals_stream: bool,
) -> Result<()> {
    for referred in &seg.referred_to {
        if !page_numbers.contains(referred) {
            anyhow::bail!(
                "Page {} segment {} refers to missing segment {}",
                page_number,
                seg.number,
                referred
            );
        }
        if global_numbers.contains(referred) && no_globals_stream {
            anyhow::bail!(
                "Page {} segment {} refers to global {} but no globals stream exists",
                page_number,
                seg.number,
                referred
            );
        }
    }
    Ok(())
}

/// The dictionary positions a page's text region codes symbol IDs against.
struct TextRegionDictionaries<'d> {
    global_sym_to_dict_pos: &'d [u32],
    num_global_dict_symbols: u32,
    local_sym_to_dict_pos: &'d [u32],
    num_local_dict_symbols: u32,
}

/// The page's local symbol dictionary segments, one per planned segment
/// number, each carrying the encoded dictionary.
fn local_dictionary_segments<'l>(
    layout: &'l PlannedPageLayout,
    payload: &'l [u8],
) -> impl Iterator<Item = Segment> + 'l {
    layout
        .local_dict_segment_numbers
        .iter()
        .copied()
        .map(move |segment_number| Segment {
            number: segment_number,
            seg_type: SegmentType::SymbolDictionary,
            deferred_non_retain: false,
            retain_flags: 0,
            page_association_type: 0,
            referred_to: Vec::new(),
            page: Some(layout.page_number),
            payload: payload.to_vec(),
        })
}
