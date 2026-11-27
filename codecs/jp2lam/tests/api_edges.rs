//! Library-facing edges: thread limits, wide images, and lossy coding above
//! 8 bits per sample.

use jp2lam::{
    CodingOptions, ColorEncoding, ColorSpace, Component, ComponentTransform, DecodeConcurrency,
    DecodeLimits, DecodeOutputFormat, DecodeRequest, DecodeResolution, DecodeResult, DecodeTarget,
    EncodeOptions, IccComponentModel, Image, ImageView, Jp2Decoder, Jp2LamError, OutputFormat,
    RateControl, ResourceLimits, TilePolicy, decode_jp2, encode, encode_view,
    encode_view_to_writer, inspect_jp2, inspect_jp2_header_for_reduction, prepare_jp2,
};

fn textured_gray(width: u32, height: u32) -> Image {
    let data: Vec<u8> = (0..width * height)
        .map(|i| {
            let (x, y) = (i % width, i / width);
            ((x * 3 + y * 5) % 200 + (x ^ y) % 37) as u8
        })
        .collect();
    Image::from_gray_bytes(width, height, &data).expect("gray")
}

/// `max_threads` now sizes a pool for the whole encode instead of only gating
/// component parallelism. The output must not depend on it.
#[test]
fn thread_limit_does_not_change_output() {
    let image = textured_gray(300, 200);
    for rate_control in [
        RateControl::Lossless,
        RateControl::ApproxQuality(70),
        RateControl::TargetBytes(9_000),
    ] {
        let options = |max_threads| EncodeOptions {
            rate_control: Some(rate_control),
            tile_policy: TilePolicy::Fixed {
                width: 128,
                height: 128,
            },
            resource_limits: ResourceLimits {
                max_threads,
                ..ResourceLimits::default()
            },
            ..EncodeOptions::default()
        };
        let unlimited = encode(&image, &options(None)).expect("unlimited");
        assert_eq!(
            encode(&image, &options(Some(1))).expect("one thread"),
            unlimited
        );
        assert_eq!(
            encode(&image, &options(Some(3))).expect("three threads"),
            unlimited
        );

        let mut streamed = Vec::new();
        encode_view_to_writer(
            image.as_view().expect("view"),
            &options(Some(2)),
            &mut streamed,
        )
        .expect("writer path under a thread limit");
        assert_eq!(streamed, unlimited, "{rate_control:?} writer path");
    }
}

/// A 40000-sample-wide image cannot be one tile with one precinct per
/// resolution; `Auto` splits it at 32768 and the result decodes exactly.
#[test]
fn images_wider_than_a_precinct_encode_with_auto_tiles() {
    let image = textured_gray(40_000, 6);
    let single = EncodeOptions {
        rate_control: Some(RateControl::Lossless),
        ..EncodeOptions::default()
    };
    assert!(encode(&image, &single).is_err());

    let auto = EncodeOptions {
        tile_policy: TilePolicy::Auto,
        ..single
    };
    let encoded = encode(&image, &auto).expect("auto tiles");
    let meta = inspect_jp2(&encoded).expect("inspect");
    assert_eq!(meta.codestream.siz.tile_width, 32_768);
    let decoded = decode_jp2(&encoded).expect("decode");
    assert_eq!(decoded.components[0].data, image.components[0].data);
}

fn psnr_at_depth(original: &[u16], decoded: &[i32], bits: u32) -> f64 {
    let peak = f64::from((1u32 << bits) - 1);
    let mse = original
        .iter()
        .zip(decoded)
        .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
        .sum::<f64>()
        / original.len() as f64;
    10.0 * (peak * peak / mse).log10()
}

/// Lossy 9/7 coding was accepted at 10-16 bits but only lossless had
/// round-trip coverage there.
#[test]
fn lossy_high_bit_depth_round_trips_at_declared_precision() {
    let (width, height) = (96u32, 72u32);
    for bits in [10u32, 12, 14, 16] {
        let max = (1u32 << bits) - 1;
        let gray: Vec<u16> = (0..width * height)
            .map(|i| {
                let (x, y) = (i % width, i / width);
                ((x * max / width + y * max / height) / 2 + (x * 7 % 13) * (max / 400)).min(max)
                    as u16
            })
            .collect();
        let rgb: Vec<u16> = gray
            .iter()
            .flat_map(|&v| [v, (max as u16).saturating_sub(v), v / 2 + (max / 4) as u16])
            .collect();

        let views = [
            (
                "gray",
                ImageView::from_gray16(width, height, &gray, bits).expect("gray view"),
            ),
            (
                "rgb",
                ImageView::from_rgb16_interleaved(width, height, &rgb, bits).expect("rgb view"),
            ),
        ];
        for (label, view) in views {
            for rate_control in [
                RateControl::ApproxQuality(85),
                RateControl::CompressionRatio(8.0),
            ] {
                assert_lossy_round_trip(label, &view, rate_control, bits, (&gray, &rgb));
            }
        }
    }
}

/// Encodes `view` lossily at `rate_control` and checks every component
/// decodes at the declared precision within 35 dB, and that a compression
/// ratio target is met. `gray` and interleaved `rgb` are the source samples.
fn assert_lossy_round_trip(
    label: &str,
    view: &ImageView<'_>,
    rate_control: RateControl,
    bits: u32,
    (gray, rgb): (&[u16], &[u16]),
) {
    let options = EncodeOptions {
        rate_control: Some(rate_control),
        format: OutputFormat::Jp2,
        ..EncodeOptions::default()
    };
    let encoded = encode_view(view.clone(), &options)
        .unwrap_or_else(|err| panic!("{label} {bits}-bit {rate_control:?}: {err}"));
    let decoded = decode_jp2(&encoded).expect("decode");
    assert_eq!(decoded.components.len(), view.components.len());
    for (index, component) in decoded.components.iter().enumerate() {
        assert_eq!(component.precision, bits, "{label} {bits}-bit");
        let original: Vec<u16> = if label == "gray" {
            gray.to_vec()
        } else {
            rgb.iter().skip(index).step_by(3).copied().collect()
        };
        let psnr = psnr_at_depth(&original, &component.data, bits);
        assert!(
            psnr > 35.0,
            "{label} {bits}-bit {rate_control:?} component {index}: {psnr:.1} dB"
        );
    }
    if let RateControl::CompressionRatio(ratio) = rate_control {
        let source_bytes =
            f64::from(view.width * view.height) * view.components.len() as f64 * f64::from(bits)
                / 8.0;
        let achieved = source_bytes / encoded.len() as f64;
        assert!(
            achieved >= f64::from(ratio) * 0.95,
            "{label} {bits}-bit ratio {achieved:.2}"
        );
    }
}

fn minimal_rgb_restricted_icc() -> Vec<u8> {
    // 128-byte header + 4-byte tag count + 6 * 12-byte tag entries + 12-byte curv + 20-byte XYZ.
    let curv_offset: u32 = 132 + 6 * 12;
    let xyz_offset: u32 = curv_offset + 12;
    let total_len: u32 = xyz_offset + 20;
    let mut profile = vec![0u8; total_len as usize];
    profile[0..4].copy_from_slice(&total_len.to_be_bytes());
    profile[12..16].copy_from_slice(b"mntr");
    profile[16..20].copy_from_slice(b"RGB ");
    profile[20..24].copy_from_slice(b"XYZ ");
    profile[36..40].copy_from_slice(b"acsp");
    profile[128..132].copy_from_slice(&6u32.to_be_bytes());

    let tags: [(&[u8; 4], u32, u32); 6] = [
        (b"rTRC", curv_offset, 12),
        (b"gTRC", curv_offset, 12),
        (b"bTRC", curv_offset, 12),
        (b"rXYZ", xyz_offset, 20),
        (b"gXYZ", xyz_offset, 20),
        (b"bXYZ", xyz_offset, 20),
    ];
    for (idx, (sig, offset, size)) in tags.iter().enumerate() {
        let pos = 132 + idx * 12;
        profile[pos..pos + 4].copy_from_slice(*sig);
        profile[pos + 4..pos + 8].copy_from_slice(&offset.to_be_bytes());
        profile[pos + 8..pos + 12].copy_from_slice(&size.to_be_bytes());
    }
    profile[curv_offset as usize..curv_offset as usize + 4].copy_from_slice(b"curv");
    profile[xyz_offset as usize..xyz_offset as usize + 4].copy_from_slice(b"XYZ ");
    profile
}

#[test]
fn prepared_decode_and_header_reduction_match_standard_decode_for_icc_jp2_and_bare_j2k() {
    let (w, h) = (192u32, 144u32);
    let rgb: Vec<u8> = (0..w * h * 3)
        .map(|i| ((i.wrapping_mul(37) ^ (i >> 5)) & 0xFF) as u8)
        .collect();
    let image = Image::from_rgb_bytes(w, h, &rgb).expect("rgb");
    let icc = ColorEncoding::restricted_icc(minimal_rgb_restricted_icc(), IccComponentModel::Rgb)
        .expect("restricted icc");

    let jp2_icc = encode(
        &image,
        &EncodeOptions {
            rate_control: Some(RateControl::ApproxQuality(80)),
            format: OutputFormat::Jp2,
            color_encoding: Some(icc),
            ..EncodeOptions::default()
        },
    )
    .expect("encode icc jp2");
    let bare_j2k = encode(
        &image,
        &EncodeOptions {
            rate_control: Some(RateControl::Lossless),
            format: OutputFormat::J2k,
            ..EncodeOptions::default()
        },
    )
    .expect("encode bare j2k");

    let limits = DecodeLimits::default();
    let mut decoder = Jp2Decoder::new();

    for (label, bytes) in [("jp2+icc", &jp2_icc[..]), ("bare-j2k", &bare_j2k[..])] {
        let meta = inspect_jp2(bytes).expect("inspect");
        let prepared = prepare_jp2(bytes, &limits).expect("prepare");
        assert_eq!(prepared.width(), meta.width, "{label}");
        assert_eq!(prepared.height(), meta.height, "{label}");
        assert_eq!(prepared.colorspace(), meta.colorspace, "{label}");
        assert_eq!(prepared.in_data_alpha(), meta.in_data_alpha, "{label}");
        assert_eq!(
            prepared.container_palette_channels(),
            meta.container_palette_channels,
            "{label}"
        );

        for resolution in [
            DecodeResolution::Full,
            DecodeResolution::ReduceLevels(1),
            DecodeResolution::ReduceLevels(2),
            DecodeResolution::AtLeast {
                width: 40,
                height: 30,
                quality_margin: 1.0,
            },
        ] {
            let expected_level = meta.reduce_levels(resolution).expect("meta reduce_levels");
            assert_eq!(
                prepared
                    .reduce_levels(resolution)
                    .expect("prepared reduce_levels"),
                expected_level,
                "{label} {resolution:?}"
            );
            assert_eq!(
                inspect_jp2_header_for_reduction(bytes, resolution, &limits)
                    .expect("header reduce_levels"),
                expected_level,
                "{label} {resolution:?}"
            );
            let (dw, dh) = meta
                .decoded_dimensions(resolution)
                .expect("meta decoded_dimensions");
            assert_eq!(
                prepared
                    .decoded_dimensions(resolution)
                    .expect("prepared decoded_dimensions"),
                (dw, dh),
                "{label} {resolution:?}"
            );

            let request = DecodeRequest {
                resolution,
                output: DecodeOutputFormat::Rgb8,
                concurrency: DecodeConcurrency::Budgeted(2),
                ..DecodeRequest::default()
            };
            let DecodeResult::Raster(expected) =
                decoder.decode(bytes, &request).expect("standard decode")
            else {
                panic!("expected packed raster");
            };
            let DecodeResult::Raster(from_prepared) = decoder
                .decode_prepared(&prepared, &request)
                .expect("prepared decode")
            else {
                panic!("expected packed raster");
            };
            assert_eq!(from_prepared, expected, "{label} {resolution:?}");

            let stride = dw as usize * 3 + 5;
            let mut buf_std = vec![0xAAu8; stride * dh as usize];
            let mut buf_prep = vec![0xAAu8; stride * dh as usize];
            decoder
                .decode_into(
                    bytes,
                    &request,
                    DecodeTarget {
                        data: &mut buf_std,
                        width: dw,
                        height: dh,
                        stride,
                        format: DecodeOutputFormat::Rgb8,
                        premultiplied: false,
                    },
                )
                .expect("standard decode_into");
            decoder
                .decode_prepared_into(
                    &prepared,
                    &request,
                    DecodeTarget {
                        data: &mut buf_prep,
                        width: dw,
                        height: dh,
                        stride,
                        format: DecodeOutputFormat::Rgb8,
                        premultiplied: false,
                    },
                )
                .expect("prepared decode_into");
            assert_eq!(buf_prep, buf_std, "{label} {resolution:?} decode_into");
        }
    }
}

#[test]
fn cooperative_cancellation_aborts_decode_and_decode_into_cleanly() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let (w, h) = (256u32, 256u32);
    let rgb: Vec<u8> = (0..w * h * 3)
        .map(|i| ((i.wrapping_mul(53) ^ (i >> 3)) & 0xFF) as u8)
        .collect();
    let image = Image::from_rgb_bytes(w, h, &rgb).expect("rgb");
    let encoded = encode(
        &image,
        &EncodeOptions {
            rate_control: Some(RateControl::Lossless),
            coding: CodingOptions {
                code_block_size: Some((16, 16)),
                ..CodingOptions::default()
            },
            ..EncodeOptions::default()
        },
    )
    .expect("encode");

    let mut decoder = Jp2Decoder::new();
    for concurrency in [DecodeConcurrency::Serial, DecodeConcurrency::Budgeted(4)] {
        let request = DecodeRequest {
            output: DecodeOutputFormat::Rgb8,
            concurrency,
            ..DecodeRequest::default()
        };

        let polls = AtomicUsize::new(0);
        let err = decoder
            .decode_cancellable(&encoded, &request, &|| {
                polls.fetch_add(1, Ordering::Relaxed) >= 3
            })
            .expect_err("cancellable decode must abort");
        assert!(matches!(err, Jp2LamError::Cancelled));
        assert!(err.is_cancelled());

        let mut dst = vec![0u8; (w * h * 3) as usize];
        let polls = AtomicUsize::new(0);
        let err = decoder
            .decode_into_cancellable(
                &encoded,
                &request,
                DecodeTarget {
                    data: &mut dst,
                    width: w,
                    height: h,
                    stride: w as usize * 3,
                    format: DecodeOutputFormat::Rgb8,
                    premultiplied: false,
                },
                &|| polls.fetch_add(1, Ordering::Relaxed) >= 3,
            )
            .expect_err("cancellable decode_into must abort");
        assert!(err.is_cancelled());

        // Decoder scratch remains valid and reusable after an aborted call.
        let DecodeResult::Raster(ok) = decoder
            .decode(&encoded, &request)
            .expect("decoder remains usable after cancellation")
        else {
            panic!("expected raster");
        };
        assert_eq!(ok.data, rgb);
    }
}

fn cmyk_image(width: u32, height: u32) -> Image {
    let pixels = (width * height) as usize;
    let components = (0..4usize)
        .map(|ch| Component {
            data: (0..pixels)
                .map(|i| ((i * (17 + ch * 11) + ch * 43) & 0xFF) as i32)
                .collect(),
            width,
            height,
            precision: 8,
            signed: false,
            dx: 1,
            dy: 1,
        })
        .collect();
    Image {
        width,
        height,
        components,
        colorspace: ColorSpace::Cmyk,
    }
}

fn pack_planar_reference(planar: &Image, format: DecodeOutputFormat) -> Vec<u8> {
    let pixels = (planar.width * planar.height) as usize;
    let clamp_u8 = |v: i32| v.clamp(0, 255) as u8;
    let bpp = format.bytes_per_pixel();
    let mut out = Vec::with_capacity(pixels * bpp);
    for i in 0..pixels {
        match format {
            DecodeOutputFormat::Gray8 => out.push(clamp_u8(planar.components[0].data[i])),
            DecodeOutputFormat::GrayA8 => {
                out.push(clamp_u8(planar.components[0].data[i]));
                out.push(clamp_u8(planar.components[1].data[i]));
            }
            DecodeOutputFormat::Rgb8 => {
                out.push(clamp_u8(planar.components[0].data[i]));
                out.push(clamp_u8(planar.components[1].data[i]));
                out.push(clamp_u8(planar.components[2].data[i]));
            }
            DecodeOutputFormat::Rgbx8 => {
                out.push(clamp_u8(planar.components[0].data[i]));
                out.push(clamp_u8(planar.components[1].data[i]));
                out.push(clamp_u8(planar.components[2].data[i]));
                out.push(255);
            }
            DecodeOutputFormat::Rgba8 => {
                out.push(clamp_u8(planar.components[0].data[i]));
                out.push(clamp_u8(planar.components[1].data[i]));
                out.push(clamp_u8(planar.components[2].data[i]));
                out.push(clamp_u8(planar.components[3].data[i]));
            }
            DecodeOutputFormat::Bgra8 => {
                out.push(clamp_u8(planar.components[2].data[i]));
                out.push(clamp_u8(planar.components[1].data[i]));
                out.push(clamp_u8(planar.components[0].data[i]));
                out.push(
                    planar
                        .components
                        .get(3)
                        .map_or(255, |alpha| clamp_u8(alpha.data[i])),
                );
            }
            DecodeOutputFormat::Cmyk8 => {
                for ch in 0..4 {
                    out.push(clamp_u8(planar.components[ch].data[i]));
                }
            }
            DecodeOutputFormat::NativePlanarI32 => unreachable!(),
        }
    }
    out
}

#[test]
fn non_mct_sequential_component_packed_decode_matches_planar_across_layouts_and_budgets() {
    let (w, h) = (150u32, 110u32);
    let rgb_bytes: Vec<u8> = (0..w * h * 3)
        .map(|i| ((i.wrapping_mul(29) ^ (i >> 2)) & 0xFF) as u8)
        .collect();
    let rgba_bytes: Vec<u8> = (0..w * h * 4)
        .map(|i| ((i.wrapping_mul(41) ^ (i >> 3)) & 0xFF) as u8)
        .collect();
    let graya_bytes: Vec<u8> = (0..w * h * 2)
        .map(|i| ((i.wrapping_mul(61) ^ (i >> 2)) & 0xFF) as u8)
        .collect();

    let rgb = Image::from_rgb_bytes(w, h, &rgb_bytes).expect("rgb");
    let rgba = Image::from_rgba_bytes(w, h, &rgba_bytes).expect("rgba");
    let graya = Image::from_gray_alpha_bytes(w, h, &graya_bytes).expect("graya");
    let cmyk = cmyk_image(w, h);

    let cases: [(&str, &Image, Option<ColorEncoding>, &[DecodeOutputFormat]); 4] = [
        (
            "rgb-no-mct",
            &rgb,
            None,
            &[
                DecodeOutputFormat::Rgb8,
                DecodeOutputFormat::Rgbx8,
                DecodeOutputFormat::Bgra8,
            ],
        ),
        (
            "rgba-no-mct",
            &rgba,
            None,
            &[
                DecodeOutputFormat::Rgba8,
                DecodeOutputFormat::Rgb8,
                DecodeOutputFormat::Rgbx8,
                DecodeOutputFormat::Bgra8,
            ],
        ),
        (
            "graya",
            &graya,
            None,
            &[DecodeOutputFormat::GrayA8, DecodeOutputFormat::Gray8],
        ),
        (
            "cmyk",
            &cmyk,
            Some(ColorEncoding::Cmyk),
            &[DecodeOutputFormat::Cmyk8],
        ),
    ];

    let mut decoder = Jp2Decoder::new();
    for (label, image, color_encoding, formats) in cases {
        for rate_control in [RateControl::Lossless, RateControl::ApproxQuality(75)] {
            let encoded = encode(
                image,
                &EncodeOptions {
                    rate_control: Some(rate_control),
                    tile_policy: TilePolicy::Fixed {
                        width: 64,
                        height: 64,
                    },
                    color_encoding: color_encoding.clone(),
                    coding: CodingOptions {
                        component_transform: ComponentTransform::None,
                        ..CodingOptions::default()
                    },
                    ..EncodeOptions::default()
                },
            )
            .unwrap_or_else(|e| panic!("{label} {rate_control:?}: {e}"));

            let meta = inspect_jp2(&encoded).expect("inspect");
            assert!(
                !meta.codestream.cod.use_mct,
                "{label} {rate_control:?} must signal use_mct = false"
            );

            for resolution in [DecodeResolution::Full, DecodeResolution::ReduceLevels(1)] {
                assert_packed_matches_planar(
                    &mut decoder,
                    &encoded,
                    resolution,
                    formats,
                    &format!("{label} {rate_control:?}"),
                );
            }
        }
    }
}

/// Decodes `encoded` at `resolution` as native planes and, in every packed
/// format and at both concurrencies, checks the packed output equals the
/// planes packed by the reference.
fn assert_packed_matches_planar(
    decoder: &mut Jp2Decoder,
    encoded: &[u8],
    resolution: DecodeResolution,
    formats: &[DecodeOutputFormat],
    case: &str,
) {
    let DecodeResult::Native(planar) = decoder
        .decode(
            encoded,
            &DecodeRequest {
                resolution,
                output: DecodeOutputFormat::NativePlanarI32,
                ..DecodeRequest::default()
            },
        )
        .expect("planar reference")
    else {
        panic!("expected planar");
    };

    for &format in formats {
        let expected = pack_planar_reference(&planar, format);
        for concurrency in [DecodeConcurrency::Serial, DecodeConcurrency::Budgeted(4)] {
            let DecodeResult::Raster(raster) = decoder
                .decode(
                    encoded,
                    &DecodeRequest {
                        resolution,
                        output: format,
                        concurrency,
                        ..DecodeRequest::default()
                    },
                )
                .unwrap_or_else(|e| {
                    panic!("{case} {resolution:?} {format:?} {concurrency:?}: {e}")
                })
            else {
                panic!("expected raster");
            };
            assert_eq!(
                raster.data, expected,
                "{case} {resolution:?} {format:?} {concurrency:?}"
            );
        }
    }
}
