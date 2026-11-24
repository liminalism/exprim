//! x86 runtime dispatch layer for primitive kernels.
//!
//! This mirrors the bpg-rs shape: portable `wide` kernels are installed first
//! (with the `simd` feature), then x86-specific kernels may overwrite
//! individual entries when runtime CPU feature detection proves they are
//! available. The AVX2 overrides are the per-row kernels of the streaming
//! inverse 9/7 and the packed row packer (`Kernel::Avx2`), where 256-bit lanes
//! are a measured win over `wide`'s two 128-bit halves; they depend only on
//! `std::arch`, so they are installed with or without the `simd` feature.

use super::Primitives;

pub(crate) fn setup(primitives: &mut Primitives, mode: &str) {
    if avx2_enabled(mode) && avx2_available() {
        avx2::setup(primitives);
    }
}

fn avx2_enabled(mode: &str) -> bool {
    matches!(mode, "auto" | "simd" | "x86" | "avx2")
}

#[cfg(target_arch = "x86")]
fn avx2_available() -> bool {
    std::arch::is_x86_feature_detected!("avx2")
}

#[cfg(target_arch = "x86_64")]
fn avx2_available() -> bool {
    std::arch::is_x86_feature_detected!("avx2")
}

mod avx2 {
    use super::Primitives;
    use crate::simd::Kernel;
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    pub(super) fn setup(primitives: &mut Primitives) {
        primitives.dwt.inverse_97_2d_owned = inverse_97_2d_owned;
        primitives.dwt.inverse_53_2d = inverse_53_2d;
        primitives.color.inverse_rct = inverse_rct;
        primitives.kernel = Kernel::Avx2;
        primitives.backend = "avx2";
    }

    fn inverse_97_2d_owned(
        data: Vec<f32>,
        width: usize,
        height: usize,
        levels: u8,
    ) -> crate::error::Result<Vec<f32>> {
        Ok(crate::dwt::inverse_97_2d_even(
            data,
            width,
            height,
            levels,
            Kernel::Avx2,
            None,
        ))
    }

    fn inverse_53_2d(
        data: &mut [i32],
        width: usize,
        height: usize,
        levels: u8,
    ) -> crate::error::Result<()> {
        if width == 0 || height == 0 || levels == 0 {
            return Ok(());
        }
        // SAFETY: `setup` only installs `inverse_53_2d` after runtime AVX2 detection.
        let max_abs = unsafe { max_abs_i32_avx2(data) };
        if crate::dwt::i32_lifting_is_safe(max_abs, levels, false) {
            let out = crate::dwt::inverse_53_2d_even(
                data.to_vec(),
                width,
                height,
                levels,
                Kernel::Avx2,
                None,
            );
            data.copy_from_slice(&out);
            Ok(())
        } else {
            crate::dwt::inverse_53_2d_in_place(data, width, height, levels)
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn max_abs_i32_avx2(data: &[i32]) -> u32 {
        let mut acc = _mm256_setzero_si256();
        let mut has_i32_min = false;
        let mut i = 0usize;
        while i + 8 <= data.len() {
            // SAFETY: `i + 8 <= data.len()`.
            unsafe {
                let v = _mm256_loadu_si256(data.as_ptr().add(i).cast());
                let abs = _mm256_abs_epi32(v);
                acc = _mm256_max_epu32(acc, abs);
            }
            i += 8;
        }
        let mut lanes = [0u32; 8];
        // SAFETY: `lanes` is 32 bytes.
        unsafe { _mm256_storeu_si256(lanes.as_mut_ptr().cast(), acc) };
        let mut m = lanes.into_iter().max().unwrap_or(0);
        if m == i32::MIN as u32 {
            has_i32_min = true;
        }
        while i < data.len() {
            let u = data[i].unsigned_abs();
            if u > m {
                m = u;
            }
            i += 1;
        }
        if has_i32_min {
            i32::MIN.unsigned_abs()
        } else {
            m
        }
    }

    fn inverse_rct(y: &mut [i32], db: &mut [i32], dr: &mut [i32]) {
        // SAFETY: `setup` only installs `inverse_rct` after runtime AVX2 detection.
        unsafe { inverse_rct_avx2(y, db, dr) }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn inverse_rct_avx2(y: &mut [i32], db: &mut [i32], dr: &mut [i32]) {
        let n = y.len().min(db.len()).min(dr.len());
        let mut i = 0usize;
        while i + 8 <= n {
            // SAFETY: `i + 8 <= n`. Every lane checks no wrap is needed when
            // `db + dr` stays within i32 (true for all normal precision <= 24 planes).
            unsafe {
                let yy = _mm256_loadu_si256(y.as_ptr().add(i).cast());
                let dbv = _mm256_loadu_si256(db.as_ptr().add(i).cast());
                let drv = _mm256_loadu_si256(dr.as_ptr().add(i).cast());
                let g = _mm256_sub_epi32(yy, _mm256_srai_epi32(_mm256_add_epi32(dbv, drv), 2));
                _mm256_storeu_si256(y.as_mut_ptr().add(i).cast(), _mm256_add_epi32(drv, g));
                _mm256_storeu_si256(db.as_mut_ptr().add(i).cast(), g);
                _mm256_storeu_si256(dr.as_mut_ptr().add(i).cast(), _mm256_add_epi32(dbv, g));
            }
            i += 8;
        }
        while i < n {
            let yy = y[i];
            let dbv = db[i];
            let drv = dr[i];
            let g = yy - ((dbv + drv) >> 2);
            y[i] = drv + g;
            db[i] = g;
            dr[i] = dbv + g;
            i += 1;
        }
    }
}
