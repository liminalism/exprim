//! `probe diff <dirA> <dirB> [--tile 96] [--top 8] [--tsv FILE]`
//!
//! Pixel-delta statistics between two directories of binary P6 PPMs (as
//! written by `probe run --ppm`). Files are paired by name. Reporting tool:
//! exits 0 whenever it ran; malformed or unpaired files become `#skip` lines.

use std::collections::BTreeSet;
use std::io::Write as _;
use std::path::Path;

struct Ppm {
    w: usize,
    h: usize,
    rgb: Vec<u8>,
}

fn parse_ppm(bytes: &[u8]) -> Result<Ppm, String> {
    let mut pos = 0usize;
    let token = |pos: &mut usize| -> Option<String> {
        loop {
            while *pos < bytes.len() && bytes[*pos].is_ascii_whitespace() {
                *pos += 1;
            }
            if *pos < bytes.len() && bytes[*pos] == b'#' {
                while *pos < bytes.len() && bytes[*pos] != b'\n' {
                    *pos += 1;
                }
            } else {
                break;
            }
        }
        let start = *pos;
        while *pos < bytes.len() && !bytes[*pos].is_ascii_whitespace() {
            *pos += 1;
        }
        if start == *pos {
            None
        } else {
            Some(String::from_utf8_lossy(&bytes[start..*pos]).into_owned())
        }
    };
    let magic = token(&mut pos).ok_or("empty file")?;
    if magic != "P6" {
        return Err(format!("not a binary P6 ppm (magic {magic:?})"));
    }
    let mut num = |what: &str| -> Result<usize, String> {
        token(&mut pos)
            .ok_or_else(|| format!("truncated header ({what})"))?
            .parse::<usize>()
            .map_err(|_| format!("bad {what}"))
    };
    let w = num("width")?;
    let h = num("height")?;
    let maxval = num("maxval")?;
    if maxval != 255 {
        return Err(format!("maxval {maxval} unsupported"));
    }
    pos += 1; // single whitespace byte after maxval
    let need = w
        .checked_mul(h)
        .and_then(|p| p.checked_mul(3))
        .ok_or("dimension overflow")?;
    if pos > bytes.len() || bytes.len() - pos < need {
        return Err(format!(
            "truncated pixel data ({} of {need} bytes)",
            bytes.len().saturating_sub(pos)
        ));
    }
    Ok(Ppm { w, h, rgb: bytes[pos..pos + need].to_vec() })
}

fn load(path: &Path) -> Result<Ppm, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read error: {e}"))?;
    parse_ppm(&bytes)
}

fn ppm_names(dir: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            let name = e.file_name().to_string_lossy().into_owned();
            if name.to_ascii_lowercase().ends_with(".ppm") {
                out.insert(name);
            }
        }
    }
    out
}

pub fn diff(args: &[String]) {
    let mut a = crate::Args(args.to_vec());
    let tile: usize = a.value("--tile").and_then(|v| v.parse().ok()).unwrap_or(96).max(1);
    let top: usize = a.value("--top").and_then(|v| v.parse().ok()).unwrap_or(8);
    let tsv = a.value("--tsv");
    if a.0.len() < 2 {
        eprintln!("usage: probe diff <dirA> <dirB> [--tile 96] [--top 8] [--tsv FILE]");
        std::process::exit(2);
    }
    let (dir_a, dir_b) = (a.0[0].clone(), a.0[1].clone());
    let names_a = ppm_names(&dir_a);
    let names_b = ppm_names(&dir_b);

    let mut out = String::new();
    out.push_str(
        "#file\tw\th\tpixels\tchanged\tchanged_gt8\tchanged_gt48\tmax_abs_delta\tmean_signed_delta\tink_a\tink_b\tdark_a\tdark_b\n",
    );
    let (mut compared, mut with_change, mut gmax, mut gt48_total) = (0u64, 0u64, 0u8, 0u64);

    for name in names_a.union(&names_b) {
        if !names_a.contains(name) {
            out.push_str(&format!("#skip\t{name}\tmissing in A\n"));
            continue;
        }
        if !names_b.contains(name) {
            out.push_str(&format!("#skip\t{name}\tmissing in B\n"));
            continue;
        }
        let pa = match load(&Path::new(&dir_a).join(name)) {
            Ok(p) => p,
            Err(e) => {
                out.push_str(&format!("#skip\t{name}\tA: {e}\n"));
                continue;
            }
        };
        let pb = match load(&Path::new(&dir_b).join(name)) {
            Ok(p) => p,
            Err(e) => {
                out.push_str(&format!("#skip\t{name}\tB: {e}\n"));
                continue;
            }
        };
        if pa.w != pb.w || pa.h != pb.h {
            out.push_str(&format!(
                "#skip\t{name}\tdimension mismatch {}x{} vs {}x{}\n",
                pa.w, pa.h, pb.w, pb.h
            ));
            continue;
        }
        let (w, h) = (pa.w, pa.h);
        let tx = w.div_ceil(tile).max(1);
        let ty = h.div_ceil(tile).max(1);
        let mut tile_cnt = vec![0u32; tx * ty];
        let mut tile_max = vec![0u8; tx * ty];
        let (mut changed, mut gt8, mut gt48, mut maxd) = (0u64, 0u64, 0u64, 0u8);
        let (mut luma_diff_sum, mut ink_a, mut ink_b) = (0i64, 0u64, 0u64);
        let (mut dark_a_sum, mut dark_b_sum) = (0u64, 0u64);
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * 3;
                let (ra, ga, ba) = (pa.rgb[i] as i32, pa.rgb[i + 1] as i32, pa.rgb[i + 2] as i32);
                let (rb, gb, bb) = (pb.rgb[i] as i32, pb.rgb[i + 1] as i32, pb.rgb[i + 2] as i32);
                let d = (ra - rb).abs().max((ga - gb).abs()).max((ba - bb).abs()) as u8;
                let (sa, sb) = (ra + ga + ba, rb + gb + bb);
                luma_diff_sum += (sb - sa) as i64; // x3 scaled
                dark_a_sum += (765 - sa) as u64;
                dark_b_sum += (765 - sb) as u64;
                if sa < 600 {
                    ink_a += 1;
                }
                if sb < 600 {
                    ink_b += 1;
                }
                if d > 0 {
                    changed += 1;
                    maxd = maxd.max(d);
                    let t = (y / tile) * tx + x / tile;
                    tile_max[t] = tile_max[t].max(d);
                    if d > 8 {
                        gt8 += 1;
                    }
                    if d > 48 {
                        gt48 += 1;
                        tile_cnt[t] += 1;
                    }
                }
            }
        }
        let px = (w as u64) * (h as u64);
        let pxf = px.max(1) as f64;
        // luma = sum/3 ; dark = (255 - luma)/255 = (765 - sum)/765
        let mean_signed = luma_diff_sum as f64 / 3.0 / pxf;
        out.push_str(&format!(
            "{name}\t{w}\t{h}\t{px}\t{changed}\t{gt8}\t{gt48}\t{maxd}\t{mean_signed:.5}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\n",
            ink_a as f64 / pxf,
            ink_b as f64 / pxf,
            dark_a_sum as f64 / 765.0 / pxf,
            dark_b_sum as f64 / 765.0 / pxf,
        ));
        let mut tiles: Vec<usize> = (0..tx * ty).filter(|&t| tile_cnt[t] > 0).collect();
        tiles.sort_by(|&p, &q| tile_cnt[q].cmp(&tile_cnt[p]).then(tile_max[q].cmp(&tile_max[p])).then(p.cmp(&q)));
        for &t in tiles.iter().take(top) {
            out.push_str(&format!(
                "#tile\t{name}\t{}\t{}\t{}\t{}\n",
                (t % tx) * tile,
                (t / tx) * tile,
                tile_cnt[t],
                tile_max[t]
            ));
        }
        compared += 1;
        if changed > 0 {
            with_change += 1;
        }
        gmax = gmax.max(maxd);
        gt48_total += gt48;
    }
    out.push_str(&format!(
        "#summary files_compared={compared} files_with_change={with_change} global_max_delta={gmax} total_gt48={gt48_total}\n"
    ));
    print!("{out}");
    if let Some(path) = tsv {
        match std::fs::File::create(&path).and_then(|mut f| f.write_all(out.as_bytes())) {
            Ok(()) => {}
            Err(e) => eprintln!("probe diff: cannot write {path}: {e}"),
        }
    }
}
