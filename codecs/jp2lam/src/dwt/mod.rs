mod irrev97;
pub(crate) mod norms;
pub(crate) mod pcrd;
mod rev53;
mod stream53;
mod stream97;

/// Subband storage that separates the coarse `LL1` quadrant (`0..sn` x `0..sv`)
/// from the finest detail bands (`HL1` and `LH1/HH1`).
///
/// Synthesizing `LL1` from `coarse` consumes and drops `coarse` before the
/// finest synthesis level begins, avoiding the extra quarter-plane (`+1 B/px`
/// per component, `+3 B/px` for RGB) that monolithic nested-subband storage
/// retains alongside `data`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SplitSubbands<T> {
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) coarse: Vec<T>,
    pub(crate) hl1: Vec<T>,
    pub(crate) lh_hh1: Vec<T>,
}

impl<T: Default + Clone> SplitSubbands<T> {
    #[allow(dead_code)]
    pub(crate) fn new_zeroed(width: usize, height: usize) -> Self {
        let sn = width.div_ceil(2);
        let dn = width - sn;
        let sv = height.div_ceil(2);
        let dv = height - sv;
        Self {
            width,
            height,
            coarse: vec![T::default(); sn * sv],
            hl1: vec![T::default(); dn * sv],
            lh_hh1: vec![T::default(); width * dv],
        }
    }
}

impl SplitSubbands<f32> {
    pub(crate) fn new_zeroed_with_pool(
        width: usize,
        height: usize,
        pool: &crate::decode::Jp2BufferPool,
    ) -> Self {
        let sn = width.div_ceil(2);
        let dn = width - sn;
        let sv = height.div_ceil(2);
        let dv = height - sv;
        Self {
            width,
            height,
            coarse: pool.take_zeroed_f32(sn * sv),
            hl1: pool.take_zeroed_f32(dn * sv),
            lh_hh1: pool.take_zeroed_f32(width * dv),
        }
    }
}

impl SplitSubbands<i32> {
    pub(crate) fn new_zeroed_with_pool(
        width: usize,
        height: usize,
        pool: &crate::decode::Jp2BufferPool,
    ) -> Self {
        let sn = width.div_ceil(2);
        let dn = width - sn;
        let sv = height.div_ceil(2);
        let dv = height - sv;
        Self {
            width,
            height,
            coarse: pool.take_zeroed_i32(sn * sv),
            hl1: pool.take_zeroed_i32(dn * sv),
            lh_hh1: pool.take_zeroed_i32(width * dv),
        }
    }
}

#[derive(Debug, Default)]
pub(crate) struct InverseDwtTiming {
    pub(crate) horizontal_ns: u64,
    pub(crate) vertical_ns: u64,
    pub(crate) level_ns: Vec<u64>,
}

impl InverseDwtTiming {
    pub(crate) fn record_level(
        &mut self,
        horizontal: std::time::Duration,
        vertical: std::time::Duration,
    ) {
        let horizontal = u64::try_from(horizontal.as_nanos()).unwrap_or(u64::MAX);
        let vertical = u64::try_from(vertical.as_nanos()).unwrap_or(u64::MAX);
        self.horizontal_ns = self.horizontal_ns.saturating_add(horizontal);
        self.vertical_ns = self.vertical_ns.saturating_add(vertical);
        self.level_ns.push(horizontal.saturating_add(vertical));
    }
}

// `forward_*_2d_in_place`/`forward_*_2d_in_place_wide` are only reached by
// `simd::scalar`/`simd::wide`'s `#[cfg(test)]` cross-check functions
// (production always goes through the tile-origin `_at` variants below) — see
// `simd::scalar::forward_97_2d`'s doc comment. `inverse_*_2d_in_place[_wide]`
// stay unconditional: they back the live `PRIMITIVES.dwt.inverse_*` dispatch
// entries used by decode.
#[cfg(all(test, feature = "simd"))]
pub(crate) use irrev97::forward_97_2d_in_place;
pub(crate) use irrev97::forward_97_2d_in_place_at;
#[cfg(all(test, feature = "simd"))]
pub(crate) use irrev97::forward_97_2d_in_place_wide;
pub(crate) use irrev97::inverse_97_2d_in_place;
pub(crate) use irrev97::inverse_97_2d_in_place_at;
#[cfg(feature = "simd")]
pub(crate) use irrev97::inverse_97_2d_in_place_wide;
#[cfg(all(test, feature = "simd"))]
pub(crate) use rev53::forward_53_2d_in_place;
pub(crate) use rev53::forward_53_2d_in_place_at;
#[cfg(all(test, feature = "simd"))]
pub(crate) use rev53::forward_53_2d_in_place_wide;
pub(crate) use rev53::inverse_53_2d_in_place;
pub(crate) use rev53::inverse_53_2d_in_place_at;
pub(crate) use rev53::inverse_53_2d_in_place_profiled;
#[cfg(feature = "simd")]
pub(crate) use rev53::inverse_53_2d_in_place_wide;
pub(crate) use stream53::{
    LevelSynth53Scratch, SynthesisPrefix53, i32_lifting_is_safe, inverse_53_2d_even,
    inverse_53_2d_even_but_last_split_with_pool, inverse_53_2d_even_but_last_with_pool,
    inverse_53_2d_even_split_with_pool, inverse_53_2d_even_with_pool,
};
pub(crate) use stream97::{
    LevelSynthScratch, SynthesisPrefix, inverse_97_2d_even,
    inverse_97_2d_even_but_last_split_with_pool, inverse_97_2d_even_but_last_with_pool,
    inverse_97_2d_even_split_with_pool, inverse_97_2d_even_with_pool,
};
