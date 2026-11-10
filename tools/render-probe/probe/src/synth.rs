//! `probe synth <outdir-root>`: redistributable synthetic fixtures.
//!
//! Writes deterministic single-page PDFs to `<root>/vector/` and `<root>/text/`
//! using exprim's `test_support` PDF builder (raw objects + hand-written
//! content-stream operator text). Needs `--features synth`.

#[cfg(not(feature = "synth"))]
pub fn synth(_args: &[String]) {
    eprintln!("probe synth: rebuild with `--features synth` (needs exprim/test-support)");
    std::process::exit(2);
}

#[cfg(feature = "synth")]
pub fn synth(args: &[String]) {
    fx::run(args)
}

#[cfg(feature = "synth")]
mod fx {
    use exprim::test_support::builder::PdfBuilder;
    use exprim::test_support::fonts::{minimal_ttf, minimal_type1};
    use std::f64::consts::{PI, TAU};
    use std::fmt::Write as _;

    // ------------------------------------------------------------------
    // Deterministic RNG (xorshift64*)
    // ------------------------------------------------------------------
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
        fn f(&mut self) -> f64 {
            (self.next() >> 11) as f64 / (1u64 << 53) as f64
        }
        fn range(&mut self, lo: f64, hi: f64) -> f64 {
            lo + (hi - lo) * self.f()
        }
        fn int(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
        fn rgb(&mut self, lo: f64, hi: f64) -> (f64, f64, f64) {
            (self.range(lo, hi), self.range(lo, hi), self.range(lo, hi))
        }
    }

    // ------------------------------------------------------------------
    // Number formatting
    // ------------------------------------------------------------------
    fn trim(s: String) -> String {
        let s = if s.contains('.') {
            s.trim_end_matches('0').trim_end_matches('.').to_owned()
        } else {
            s
        };
        if s == "-0" || s.is_empty() { "0".into() } else { s }
    }
    /// One decimal (geometry).
    fn n(v: f64) -> String {
        trim(format!("{v:.1}"))
    }
    /// Two decimals (widths, colours, matrices).
    fn n2(v: f64) -> String {
        trim(format!("{v:.2}"))
    }
    fn n3(v: f64) -> String {
        trim(format!("{v:.3}"))
    }
    fn rgb_op(c: (f64, f64, f64), op: &str) -> String {
        format!("{} {} {} {op}\n", n2(c.0), n2(c.1), n2(c.2))
    }

    // ------------------------------------------------------------------
    // Document assembly
    // ------------------------------------------------------------------
    enum Ex {
        Obj(String),
        Stream(String, Vec<u8>),
    }

    /// Catalog(1) Pages(2) Page(3) Content(4); `extras` carry numbers >= 5.
    fn assemble(media: (f64, f64), content: &str, resources: &str, extras: Vec<(u32, Ex)>) -> Vec<u8> {
        let mut b = PdfBuilder::new();
        b.add_object(1, "<</Type/Catalog/Pages 2 0 R>>");
        b.add_object(
            2,
            &format!("<</Type/Pages/Kids[3 0 R]/Count 1/MediaBox[0 0 {} {}]>>", n(media.0), n(media.1)),
        );
        b.add_object(
            3,
            &format!("<</Type/Page/Parent 2 0 R/Contents 4 0 R/Resources<<{resources}>>>>"),
        );
        b.add_stream(4, "", content.as_bytes());
        for (num, ex) in extras {
            match ex {
                Ex::Obj(s) => {
                    b.add_object(num, &s);
                }
                Ex::Stream(d, data) => {
                    b.add_stream(num, &d, &data);
                }
            }
        }
        b.finish_classic_xref("/Root 1 0 R");
        b.into_bytes()
    }

    const LETTER: (f64, f64) = (612.0, 792.0);

    // ------------------------------------------------------------------
    // Geometry helpers (append operators to `out`)
    // ------------------------------------------------------------------
    const K: f64 = 0.552_284_749_8;

    fn circle(out: &mut String, cx: f64, cy: f64, r: f64) {
        let k = K * r;
        let _ = write!(
            out,
            "{} {} m {} {} {} {} {} {} c {} {} {} {} {} {} c {} {} {} {} {} {} c {} {} {} {} {} {} c h\n",
            n(cx + r), n(cy),
            n(cx + r), n(cy + k), n(cx + k), n(cy + r), n(cx), n(cy + r),
            n(cx - k), n(cy + r), n(cx - r), n(cy + k), n(cx - r), n(cy),
            n(cx - r), n(cy - k), n(cx - k), n(cy - r), n(cx), n(cy - r),
            n(cx + k), n(cy - r), n(cx + r), n(cy - k), n(cx + r), n(cy),
        );
    }

    /// Smooth closed blob of `segs` cubics, roughly `rad` radius.
    fn blob(out: &mut String, r: &mut Rng, cx: f64, cy: f64, rad: f64, segs: usize) {
        let mut pts = Vec::with_capacity(segs);
        let rot = r.range(0.0, TAU);
        for i in 0..segs {
            let a = rot + TAU * i as f64 / segs as f64 + r.range(-0.15, 0.15) * TAU / segs as f64;
            let rr = rad * r.range(0.65, 1.0);
            pts.push((cx + rr * a.cos(), cy + rr * a.sin(), a, rr));
        }
        let kk = 4.0 / 3.0 * (PI / (2.0 * segs as f64)).tan();
        let _ = writeln!(out, "{} {} m", n(pts[0].0), n(pts[0].1));
        for i in 0..segs {
            let p = pts[i];
            let q = pts[(i + 1) % segs];
            let (hp, hq) = (kk * p.3, kk * q.3);
            let (tp, tq) = (p.2 + PI / 2.0, q.2 + PI / 2.0);
            let _ = writeln!(
                out,
                "{} {} {} {} {} {} c",
                n(p.0 + hp * tp.cos()),
                n(p.1 + hp * tp.sin()),
                n(q.0 - hq * tq.cos()),
                n(q.1 - hq * tq.sin()),
                n(q.0),
                n(q.1)
            );
        }
        out.push_str("h\n");
    }

    fn wrap_angle(a: f64) -> f64 {
        (a + PI).rem_euclid(TAU) - PI
    }

    /// Smooth map-contour-like polyline of `segs` cubics (no paint operator).
    fn contour(out: &mut String, r: &mut Rng, segs: usize) {
        let (mut x, mut y) = (r.range(30.0, 582.0), r.range(30.0, 762.0));
        let mut th = r.range(0.0, TAU);
        let _ = writeln!(out, "{} {} m", n(x), n(y));
        for _ in 0..segs {
            let l = r.range(10.0, 24.0);
            let mut d = (r.f() - 0.5) * 0.6;
            if !(20.0..=592.0).contains(&x) || !(20.0..=772.0).contains(&y) {
                let to_c = (396.0 - y).atan2(306.0 - x);
                d += wrap_angle(to_c - th).clamp(-0.4, 0.4);
            }
            let th2 = th + d;
            let mid = th + d / 2.0;
            let (x2, y2) = (x + l * mid.cos(), y + l * mid.sin());
            let h = l / 3.0;
            let _ = writeln!(
                out,
                "{} {} {} {} {} {} c",
                n(x + h * th.cos()),
                n(y + h * th.sin()),
                n(x2 - h * th2.cos()),
                n(y2 - h * th2.sin()),
                n(x2),
                n(y2)
            );
            x = x2;
            y = y2;
            th = th2;
        }
    }

    // ------------------------------------------------------------------
    // Vector fixtures
    // ------------------------------------------------------------------
    fn hairline_grid() -> Vec<u8> {
        let mut r = Rng(0x1A2B_3C4D_5E6F_7081);
        let mut c = String::from("0 J 0 j 10 M\n0 G\n");
        for i in 0..4000 {
            let w = r.range(0.1, 0.5);
            let (x0, y0, x1, y1);
            if i < 1800 {
                // horizontal segments on a 6pt grid
                let row = 6.0 * (1 + r.int(130)) as f64;
                x0 = 6.0 * r.int(100) as f64;
                x1 = x0 + 6.0 * (2 + r.int(8)) as f64;
                y0 = row;
                y1 = row;
            } else if i < 3600 {
                let col = 6.0 * (1 + r.int(100)) as f64;
                y0 = 6.0 * r.int(130) as f64;
                y1 = y0 + 6.0 * (2 + r.int(8)) as f64;
                x0 = col;
                x1 = col;
            } else {
                x0 = r.range(0.0, 612.0);
                y0 = r.range(0.0, 792.0);
                let a = r.range(0.0, PI);
                let l = r.range(30.0, 120.0);
                x1 = x0 + l * a.cos();
                y1 = y0 + l * a.sin();
            }
            let _ = writeln!(c, "{} w {} {} m {} {} l S", n2(w), n(x0), n(y0), n(x1), n(y1));
        }
        assemble(LETTER, &c, "", vec![])
    }

    fn round_contours() -> Vec<u8> {
        let mut r = Rng(0x2C3D_4E5F_6071_8293);
        let palette = [(0.55, 0.30, 0.10), (0.10, 0.35, 0.65), (0.15, 0.50, 0.20), (0.45, 0.45, 0.45)];
        let mut c = String::from("1 J 1 j\n");
        for _ in 0..600 {
            let col = palette[r.int(palette.len())];
            c.push_str(&rgb_op(col, "RG"));
            let _ = writeln!(c, "{} w", n2(r.range(0.5, 4.0)));
            let segs = 20 + r.int(41);
            contour(&mut c, &mut r, segs);
            c.push_str("S\n");
        }
        assemble(LETTER, &c, "", vec![])
    }

    fn dashed_contours() -> Vec<u8> {
        let mut r = Rng(0x3D4E_5F60_7182_93A4);
        let dashes = [
            "[3 2] 0",
            "[6 2 1 2] 1.5",
            "[0.5 3] 0",
            "[10 4] 3",
            "[1 1.5] 0.25",
            "[12 3 3 3] 6",
            "[4] 2",
            "[8 3 0.5 3] 0",
        ];
        let palette = [(0.0, 0.0, 0.0), (0.7, 0.1, 0.1), (0.1, 0.3, 0.7), (0.1, 0.5, 0.2)];
        let mut c = String::new();
        for i in 0..300 {
            let _ = writeln!(c, "{} J {} j", r.int(3), r.int(3));
            c.push_str(&rgb_op(palette[i % palette.len()], "RG"));
            let _ = writeln!(c, "{} w {} d", n2(r.range(0.5, 3.0)), dashes[r.int(dashes.len())]);
            let segs = 20 + r.int(31);
            contour(&mut c, &mut r, segs);
            c.push_str("S\n");
        }
        assemble(LETTER, &c, "", vec![])
    }

    fn curve_fan() -> Vec<u8> {
        let mut r = Rng(0x4E5F_6071_8293_A4B5);
        let mut c = String::new();
        // large background shapes first
        for _ in 0..20 {
            let col = r.rgb(0.55, 0.95);
            c.push_str(&rgb_op(col, "rg"));
            let rad = r.range(100.0, 210.0);
            let (cx, cy) = (r.range(100.0, 512.0), r.range(100.0, 692.0));
            let sg = 8 + r.int(5);
            blob(&mut c, &mut r, cx, cy, rad, sg);
            if r.int(2) == 0 {
                blob(&mut c, &mut r, cx, cy, rad * 0.5, 6);
                c.push_str("f*\n");
            } else {
                c.push_str("f\n");
            }
        }
        for i in 0..6000 {
            if i % 3 == 0 {
                let col = r.rgb(0.0, 0.8);
                c.push_str(&rgb_op(col, "rg"));
            }
            let rad = r.range(1.0, 10.0);
            let (cx, cy) = (r.range(0.0, 612.0), r.range(0.0, 792.0));
            let sg = 4 + r.int(3);
            blob(&mut c, &mut r, cx, cy, rad, sg);
            if r.int(2) == 0 {
                // inner subpath, same direction: f fills it, f* leaves a hole
                blob(&mut c, &mut r, cx, cy, rad * 0.5, 4);
            }
            c.push_str(if r.int(2) == 0 { "f\n" } else { "f*\n" });
        }
        assemble(LETTER, &c, "", vec![])
    }

    fn rand_shape(c: &mut String, r: &mut Rng, cx: f64, cy: f64, size: f64) {
        if r.int(2) == 0 {
            let _ = writeln!(c, "{} {} {} {} re f", n(cx - size / 2.0), n(cy - size / 2.0), n(size), n(size * r.range(0.4, 1.2)));
        } else {
            let sg = 5 + r.int(3);
            blob(c, r, cx, cy, size / 2.0, sg);
            c.push_str(if r.int(4) == 0 { "f*\n" } else { "f\n" });
        }
    }

    fn clip_tiles() -> Vec<u8> {
        let mut r = Rng(0x5F60_7182_93A4_B5C6);
        let mut c = String::new();
        for row in 0..10 {
            for col in 0..8 {
                let (cx, cy) = (61.0 + col as f64 * 70.0, 81.0 + row as f64 * 70.0);
                c.push_str("q\n");
                circle(&mut c, cx, cy, 31.0);
                c.push_str("W n\n");
                for _ in 0..30 {
                    c.push_str(&rgb_op(r.rgb(0.1, 1.0), "rg"));
                    let (x, y) = (cx + r.range(-40.0, 40.0), cy + r.range(-40.0, 40.0));
                    let sz = r.range(8.0, 40.0);
                    rand_shape(&mut c, &mut r, x, y, sz);
                }
                c.push_str("Q\n");
            }
        }
        assemble(LETTER, &c, "", vec![])
    }

    fn clip_repeat() -> Vec<u8> {
        let mut r = Rng(0x6071_8293_A4B5_C6D7);
        let mut c = String::new();
        for _ in 0..200 {
            c.push_str("q\n");
            circle(&mut c, 306.0, 396.0, 250.0);
            c.push_str("W n\n");
            c.push_str(&rgb_op(r.rgb(0.1, 1.0), "rg"));
            let (x, y) = (r.range(60.0, 550.0), r.range(150.0, 640.0));
            let sz = r.range(40.0, 300.0);
            rand_shape(&mut c, &mut r, x, y, sz);
            c.push_str("Q\n");
        }
        assemble(LETTER, &c, "", vec![])
    }

    fn page_rects() -> Vec<u8> {
        let mut r = Rng(0x7182_93A4_B5C6_D7E8);
        let mut c = String::new();
        // page-sized fills (some overshoot the page box)
        for i in 0..6 {
            let g = 0.97 - 0.01 * i as f64;
            let _ = writeln!(c, "{} {} {} rg", n2(g), n2(g), n2(1.0 - 0.02 * i as f64));
            if i % 2 == 0 {
                c.push_str("-10 -10 632 812 re f\n");
            } else {
                c.push_str("0 0 612 792 re f\n");
            }
        }
        // nested stroked frames
        for i in 0..60 {
            let inset = 3.0 + 4.0 * i as f64;
            let g = if i % 2 == 0 { 0.2 } else { 0.6 };
            let _ = writeln!(c, "{} w {} G", n2(0.5 + 0.1 * (i % 5) as f64), n2(g));
            let _ = writeln!(c, "{} {} {} {} re S", n(inset), n(inset), n(612.0 - 2.0 * inset), n(792.0 - 2.0 * inset));
        }
        // table: row bands
        let (tx, ty, cw, rh) = (46.0, 60.0, 46.0, 24.0);
        for row in 0..28 {
            if row % 2 == 0 {
                c.push_str("0.9 0.93 1 rg\n");
            } else {
                c.push_str("1 1 1 rg\n");
            }
            let _ = writeln!(c, "{} {} {} {} re f", n(tx), n(ty + row as f64 * rh), n(cw * 12.0), n(rh));
        }
        // cell shading (random)
        for _ in 0..110 {
            c.push_str(&rgb_op(r.rgb(0.6, 1.0), "rg"));
            let (col, row) = (r.int(12), r.int(28));
            let _ = writeln!(c, "{} {} {} {} re f", n(tx + col as f64 * cw + 1.0), n(ty + row as f64 * rh + 1.0), n(cw - 2.0), n(rh - 2.0));
        }
        // rules as thin filled rects (long, axis-aligned)
        c.push_str("0 g\n");
        for row in 0..=28 {
            let _ = writeln!(c, "{} {} {} 0.5 re f", n(tx), n(ty + row as f64 * rh - 0.25), n(cw * 12.0));
        }
        for col in 0..=12 {
            let _ = writeln!(c, "{} {} 0.5 {} re f", n(tx + col as f64 * cw - 0.25), n(ty), n(rh * 28.0));
        }
        // long stroked verticals/horizontals
        c.push_str("0.3 w 0.4 G\n");
        for i in 0..30 {
            let x = tx + i as f64 * 18.0;
            let _ = writeln!(c, "{} {} m {} 760 l S", n(x), n(ty - 20.0), n(x));
        }
        // checkbox squares
        c.push_str("0.8 w 0 G\n");
        for i in 0..100 {
            let (x, y) = (50.0 + (i % 25) as f64 * 22.0, 8.0 + (i / 25) as f64 * 12.0);
            let _ = writeln!(c, "{} {} 8 8 re S", n(x), n(y));
        }
        assemble(LETTER, &c, "", vec![])
    }

    fn offpage_zoom() -> Vec<u8> {
        let mut r = Rng(0x8293_A4B5_C6D7_E8F9);
        let mut c = String::new();
        for i in 0..3000 {
            let inside = i % 10 == 0;
            let (x, y, sz) = if inside {
                (r.range(246.0, 366.0), r.range(336.0, 456.0), r.range(3.0, 25.0))
            } else {
                loop {
                    let (x, y) = (r.range(-3000.0, 3600.0), r.range(-3000.0, 3800.0));
                    if !(-20.0..=632.0).contains(&x) || !(-20.0..=812.0).contains(&y) {
                        break (x, y, r.range(5.0, 80.0));
                    }
                }
            };
            let col = r.rgb(0.0, 0.9);
            match r.int(4) {
                0 => {
                    c.push_str(&rgb_op(col, "RG"));
                    let _ = writeln!(c, "{} w {} {} m {} {} l S", n2(r.range(0.3, 3.0)), n(x), n(y), n(x + sz), n(y + sz * r.range(-1.0, 1.0)));
                }
                1 => {
                    c.push_str(&rgb_op(col, "RG"));
                    let _ = writeln!(
                        c,
                        "{} w {} {} m {} {} {} {} {} {} c S",
                        n2(r.range(0.3, 3.0)),
                        n(x), n(y),
                        n(x + sz * 0.3), n(y + sz), n(x + sz * 0.7), n(y - sz), n(x + sz), n(y)
                    );
                }
                2 => {
                    c.push_str(&rgb_op(col, "rg"));
                    let _ = writeln!(c, "{} {} {} {} re f", n(x), n(y), n(sz), n(sz * 0.7));
                }
                _ => {
                    c.push_str(&rgb_op(col, "rg"));
                    blob(&mut c, &mut r, x, y, sz / 2.0, 4);
                    c.push_str("f\n");
                }
            }
        }
        assemble(LETTER, &c, "", vec![])
    }

    fn form_repeat() -> Vec<u8> {
        let mut r = Rng(0x93A4_B5C6_D7E8_F90A);
        // Emblem in a 40x40 box centred on (20,20). Colours inherited.
        let mut f = String::new();
        circle(&mut f, 20.0, 20.0, 18.0);
        circle(&mut f, 20.0, 20.0, 14.0);
        f.push_str("f*\n");
        for i in 0..10 {
            let rr = if i % 2 == 0 { 11.0 } else { 5.0 };
            let a = PI / 2.0 + TAU * i as f64 / 10.0;
            let (px, py) = (20.0 + rr * a.cos(), 20.0 + rr * a.sin());
            let _ = write!(f, "{} {} {} ", n(px), n(py), if i == 0 { "m" } else { "l" });
        }
        f.push_str("h f\n");
        f.push_str("1.2 w 20 2 m 20 8 l S 20 38 m 20 32 l S 2 20 m 8 20 l S\n");
        f.push_str("1 g\n");
        circle(&mut f, 20.0, 20.0, 3.0);
        f.push_str("f\n");
        let mut c = String::new();
        for _ in 0..500 {
            c.push_str(&rgb_op(r.rgb(0.0, 0.9), "rg"));
            c.push_str(&rgb_op(r.rgb(0.0, 0.9), "RG"));
            let s = r.range(0.4, 1.4);
            let a = r.range(0.0, TAU);
            let (x, y) = (r.range(20.0, 590.0), r.range(20.0, 770.0));
            let _ = writeln!(
                c,
                "q {} {} {} {} {} {} cm /Fm1 Do Q",
                n3(s * a.cos()), n3(s * a.sin()), n3(-s * a.sin()), n3(s * a.cos()), n(x), n(y)
            );
        }
        assemble(
            LETTER,
            &c,
            "/XObject<</Fm1 5 0 R>>",
            vec![(5, Ex::Stream("/Type/XObject/Subtype/Form/BBox[0 0 40 40]/Resources<<>>".into(), f.into_bytes()))],
        )
    }

    fn pattern_shade_alpha() -> Vec<u8> {
        let tiling = |extra: &str, bbox: &str, step: f64, body: &str| {
            Ex::Stream(
                format!("/Type/Pattern/PatternType 1/PaintType 1/TilingType 1/BBox[{bbox}]/XStep {}/YStep {}{extra}/Resources<<>>", n(step), n(step)),
                body.as_bytes().to_vec(),
            )
        };
        let mut flower = String::from("0.9 0.4 0.1 rg\n");
        circle(&mut flower, 15.0, 15.0, 6.0);
        flower.push_str("f\n0.2 0.6 0.9 rg\n");
        for (dx, dy) in [(0.0, 9.0), (9.0, 0.0), (0.0, -9.0), (-9.0, 0.0)] {
            circle(&mut flower, 15.0 + dx, 15.0 + dy, 4.0);
            flower.push_str("f\n");
        }
        let axial = |c0: &str, c1: &str, cs: &str, coords: &str| {
            Ex::Obj(format!(
                "<</ShadingType 2/ColorSpace/{cs}/Coords[{coords}]/Extend[true true]\
                 /Function<</FunctionType 2/Domain[0 1]/C0[{c0}]/C1[{c1}]/N 1>>>>"
            ))
        };
        let extras = vec![
            (10, tiling("", "0 0 20 20", 20.0, "1 0 0 rg 0 0 10 10 re f 0 0 1 rg 10 10 10 10 re f")),
            (
                11,
                tiling(
                    "",
                    "0 0 12 12",
                    12.0,
                    "0.1 0.5 0.1 RG 1 w 0 0 m 12 12 l S 0.9 0.7 0.1 rg 8 2 3 3 re f",
                ),
            ),
            (12, tiling("/Matrix[0.7 0.7 -0.7 0.7 0 0]", "0 0 30 30", 34.0, &flower)),
            (13, Ex::Obj("<</Type/Pattern/PatternType 2/Shading 18 0 R>>".into())),
            (14, axial("1 0 0", "0 0 1", "DeviceRGB", "40 0 290 0")),
            (
                15,
                Ex::Obj(
                    "<</ShadingType 3/ColorSpace/DeviceRGB/Coords[445 565 0 445 565 55]/Extend[false false]\
                     /Function<</FunctionType 2/Domain[0 1]/C0[1 1 0]/C1[0 0.5 0]/N 1>>>>"
                        .into(),
                ),
            ),
            (
                16,
                Ex::Stream(
                    "/Type/XObject/Subtype/Form/BBox[0 0 612 792]/Group<</S/Transparency/CS/DeviceGray>>/Resources<</Shading<</ShM 19 0 R>>>>".into(),
                    b"/ShM sh".to_vec(),
                ),
            ),
            (
                17,
                Ex::Stream(
                    "/Type/XObject/Subtype/Form/BBox[0 0 200 100]/Group<</S/Transparency/CS/DeviceRGB>>/Resources<<>>".into(),
                    b"1 0 0 rg 0 0 120 80 re f 0 0.6 0 rg 60 20 130 75 re f".to_vec(),
                ),
            ),
            (18, axial("1 0.9 0", "0.5 0 0.7", "DeviceRGB", "40 390 290 490")),
            (19, axial("0", "1", "DeviceGray", "320 0 570 0")),
        ];
        let res = "/Pattern<</P1 10 0 R/P2 11 0 R/P3 12 0 R/P4 13 0 R>>\
                   /Shading<</Sh1 14 0 R/Sh2 15 0 R>>\
                   /ExtGState<</GS0<</SMask/None/ca 1/CA 1>>/GS1<</ca 0.5/CA 0.5>>/GS2<</SMask<</S/Luminosity/G 16 0 R>>>>/GS3<</ca 0.6>>>>\
                   /XObject<</Grp 17 0 R>>";
        let mut c = String::new();
        // A: tiling fills
        c.push_str("/Pattern cs\n");
        for (i, p) in ["P1", "P2", "P3"].iter().enumerate() {
            let _ = writeln!(c, "/{p} scn {} 650 160 100 re f", 40 + i * 190);
        }
        // B: sh operator (axial, radial) clipped
        c.push_str("q 40 510 250 110 re W n /Sh1 sh Q\n");
        c.push_str("q 320 510 250 110 re W n /Sh2 sh Q\n");
        // C: shading pattern fill and pattern stroke
        c.push_str("/Pattern cs /P4 scn 40 390 250 100 re f\n");
        c.push_str("/Pattern CS /P1 SCN 8 w 330 400 230 80 re S\n");
        // D: constant alpha
        c.push_str("q /GS1 gs\n1 0 0 rg 60 260 120 90 re f\n0 0.7 0 rg 120 280 120 90 re f\n0 0 1 rg 180 250 120 90 re f\n");
        c.push_str("0 G 14 w 330 260 m 560 360 l S 1 0.5 0 RG 330 360 m 560 260 l S\nQ\n");
        // E: transparency group (opaque inside, composited as a unit)
        c.push_str("0.8 0.8 0.2 rg 30 150 420 30 re f\n");
        c.push_str("q /GS3 gs 1 0 0 1 40 130 cm /Grp Do Q\n");
        c.push_str("q /GS3 gs 1 0 0 1 280 130 cm /Grp Do Q\n");
        // F: luminosity soft mask
        c.push_str("0.85 g 320 20 250 90 re f\n");
        c.push_str("q /GS2 gs 1 0 0 rg 320 20 250 90 re f /GS0 gs Q\n");
        assemble(LETTER, &c, res, extras)
    }

    // ------------------------------------------------------------------
    // Text fixtures
    // ------------------------------------------------------------------
    fn std_font(base: &str) -> String {
        format!("<</Type/Font/Subtype/Type1/BaseFont/{base}/Encoding/WinAnsiEncoding>>")
    }

    fn text_modes() -> Vec<u8> {
        let mut c = String::from("1 0 0 rg 0 0 1 RG 1.5 w\n");
        for mode in 0..8 {
            let y = 720.0 - mode as f64 * 88.0;
            c.push_str("q\n");
            let _ = writeln!(c, "BT /F1 64 Tf {mode} Tr 40 {} Td (Mode {mode} Ag) Tj ET", n(y));
            if mode >= 4 {
                // clip modes: the glyph outlines are now the clip; paint stripes
                for s in 0..16 {
                    let col = if s % 2 == 0 { "0.9 0.5 0.0 rg" } else { "0.0 0.5 0.8 rg" };
                    let _ = writeln!(c, "{col} {} {} 24 70 re f", 30 + s * 24, n(y - 14.0));
                }
            }
            c.push_str("Q\n");
        }
        assemble(
            LETTER,
            &c,
            "/Font<</F1 5 0 R>>",
            vec![(5, Ex::Obj(std_font("Helvetica-Bold")))],
        )
    }

    fn text_rotated() -> Vec<u8> {
        let mut c = String::from("0 g\n");
        let line = "The quick brown fox jumps";
        let origins = [(40.0, 60.0), (60.0, 330.0), (250.0, 140.0), (570.0, 140.0)];
        for (i, deg) in [15.0f64, 45.0, 90.0, 133.0].iter().enumerate() {
            let a = deg.to_radians();
            let (s, co) = (a.sin(), a.cos());
            for (k, size) in [10.0, 18.0, 28.0].iter().enumerate() {
                // offset along the text's "up" direction
                let off = k as f64 * 34.0;
                let (ox, oy) = (origins[i].0 - s * off, origins[i].1 + co * off);
                let _ = writeln!(
                    c,
                    "BT /F1 {size} Tf {} {} {} {} {} {} Tm ({line}) Tj ET",
                    n3(co), n3(s), n3(-s), n3(co), n(ox), n(oy)
                );
            }
        }
        // shears and non-uniform scales
        let ms = [
            ("1 0 0.5 1", 60.0, 640.0, 20.0),
            ("1 0 -0.5 1", 60.0, 690.0, 20.0),
            ("1 0.3 0 1", 330.0, 640.0, 16.0),
            ("1.6 0 0 0.6", 60.0, 740.0, 18.0),
            ("0.6 0 0 1.6", 330.0, 700.0, 14.0),
            ("1 -0.4 0.4 1", 330.0, 760.0, 12.0),
        ];
        for (m, x, y, size) in ms {
            let _ = writeln!(c, "BT /F1 {size} Tf {m} {x} {y} Tm ({line}) Tj ET");
        }
        assemble(LETTER, &c, "/Font<</F1 5 0 R>>", vec![(5, Ex::Obj(std_font("Times-Roman")))])
    }

    fn text_substituted() -> Vec<u8> {
        let mut r = Rng(0xA4B5_C6D7_E8F9_0A1B);
        let words = [
            "render", "glyph", "outline", "baseline", "advance", "kerning", "stroke", "fill", "page", "matrix",
            "pixel", "coverage", "scanline", "winding", "contour", "spline", "dash", "clip", "tile", "shade",
            "paper", "ledger", "margin", "column", "footnote", "caption", "heading", "figure", "table", "index",
        ];
        let fonts = [
            "Helvetica", "Helvetica-Bold", "Helvetica-Oblique", "Helvetica-BoldOblique",
            "Times-Roman", "Times-Bold", "Times-Italic", "Times-BoldItalic",
            "Courier", "Courier-Bold", "Courier-Oblique",
        ];
        let mut c = String::from("0 g\n");
        let mut res = String::from("/Font<<");
        let mut extras = Vec::new();
        let mut y = 770.0;
        for (i, f) in fonts.iter().enumerate() {
            let _ = write!(res, "/F{i} {} 0 R", 5 + i);
            extras.push((5 + i as u32, Ex::Obj(std_font(f))));
            let _ = writeln!(c, "BT /F{i} 9 Tf 10.5 TL 36 {} Td", n(y));
            let _ = writeln!(c, "({f}:) Tj T*");
            for _ in 0..4 {
                let mut line = String::new();
                while line.len() < 86 {
                    line.push_str(words[r.int(words.len())]);
                    line.push(' ');
                }
                let _ = writeln!(c, "({}) Tj T*", line.trim_end());
            }
            c.push_str("ET\n");
            y -= 5.0 * 10.5 + 18.0;
        }
        res.push_str(">>");
        assemble(LETTER, &c, &res, extras)
    }

    fn text_type3() -> Vec<u8> {
        let mut r = Rng(0xB5C6_D7E8_F90A_1B2C);
        let tri = String::from("800 0 0 0 700 700 d1\n0 0 m 700 0 l 350 700 l f\n");
        let sq = "800 0 0 0 700 700 d1\n50 0 600 600 re f\n".to_string();
        let mut circ = String::from("800 0 0 0 700 700 d1\n");
        circle(&mut circ, 350.0, 350.0, 330.0);
        circ.push_str("f\n");
        let dia = "800 0 0 0 700 700 d1\n350 0 m 700 350 l 350 700 l 0 350 l h f\n".to_string();
        let cross = "800 0 0 0 700 700 d1\n90 w 60 60 m 640 640 l S 60 640 m 640 60 l S\n".to_string();
        let col = "800 0 d0\n1 0 0 rg 0 0 700 700 re f 0 0 1 rg 150 150 400 400 re f\n".to_string();
        let procs = [("sq", sq), ("tri", tri), ("circ", circ), ("dia", dia), ("cross", cross), ("col", col)];
        let mut extras: Vec<(u32, Ex)> = Vec::new();
        let mut cp = String::from("<<");
        for (i, (name, body)) in procs.iter().enumerate() {
            let _ = write!(cp, "/{name} {} 0 R", 10 + i);
            extras.push((10 + i as u32, Ex::Stream(String::new(), body.clone().into_bytes())));
        }
        cp.push_str(">>");
        extras.push((6, Ex::Obj(cp)));
        extras.push((
            5,
            Ex::Obj(
                "<</Type/Font/Subtype/Type3/FontBBox[0 0 800 800]/FontMatrix[0.001 0 0 0.001 0 0]\
                 /CharProcs 6 0 R/Encoding<</Type/Encoding/Differences[97/sq/tri/circ/dia/cross/col]>>\
                 /FirstChar 97/LastChar 102/Widths[800 800 800 800 800 800]/Resources<<>>>>"
                    .into(),
            ),
        ));
        let mut c = String::new();
        for line in 0..30 {
            let y = 760.0 - line as f64 * 24.0;
            c.push_str(&rgb_op(r.rgb(0.0, 0.8), "rg"));
            let _ = write!(c, "BT /F1 {} Tf 36 {} Td (", 14 + (line % 4) * 2, n(y));
            for _ in 0..14 {
                c.push((b'a' + r.int(6) as u8) as char);
            }
            c.push_str(") Tj ET\n");
        }
        assemble(LETTER, &c, "/Font<</F1 5 0 R>>", extras)
    }

    /// Embedded-font page: rows of the single 'A' glyph at several sizes/matrices.
    fn embedded_glyph_page(subtype: &str, file_key: &str, program: Vec<u8>, width: u32, enc: &str) -> Vec<u8> {
        let mut c = String::from("0.1 0.2 0.6 rg 0.8 0.1 0.1 RG 0.8 w\n");
        let mut y = 730.0;
        for (i, size) in [8.0, 12.0, 16.0, 24.0, 36.0, 48.0, 60.0].iter().enumerate() {
            let _ = writeln!(c, "BT /F1 {size} Tf {} Tr 30 {} Td (AAAAAAAAAAAA) Tj ET", if i == 5 { 2 } else { 0 }, n(y));
            y -= size * 1.2 + 6.0;
        }
        for (k, deg) in [20.0f64, 60.0, 100.0].iter().enumerate() {
            let a = deg.to_radians();
            let _ = writeln!(
                c,
                "BT /F1 20 Tf {} {} {} {} {} {} Tm (AAAAAAAAAAAA) Tj ET",
                n3(a.cos()), n3(a.sin()), n3(-a.sin()), n3(a.cos()), 40 + k * 190, 80
            );
        }
        let len = if file_key == "FontFile" {
            format!("/Length1 {}/Length2 0/Length3 0", program.len())
        } else {
            format!("/Length1 {}", program.len())
        };
        assemble(
            LETTER,
            &c,
            "/Font<</F1 5 0 R>>",
            vec![
                (
                    5,
                    Ex::Obj(format!(
                        "<</Type/Font/Subtype/{subtype}/BaseFont/SynthTest/FirstChar 65/LastChar 65\
                         /Widths[{width}]{enc}/FontDescriptor 6 0 R>>"
                    )),
                ),
                (
                    6,
                    Ex::Obj(format!(
                        "<</Type/FontDescriptor/FontName/SynthTest/Flags 32/ItalicAngle 0\
                         /FontBBox[0 0 1000 1000]/{file_key} 7 0 R>>"
                    )),
                ),
                (7, Ex::Stream(len, program)),
            ],
        )
    }

    // ------------------------------------------------------------------
    pub fn run(args: &[String]) {
        let a = crate::Args(args.to_vec());
        let root = a.positional().to_owned();
        let list: Vec<(&str, &str, Vec<u8>)> = vec![
            ("vector", "hairline-grid.pdf", hairline_grid()),
            ("vector", "round-contours.pdf", round_contours()),
            ("vector", "dashed-contours.pdf", dashed_contours()),
            ("vector", "curve-fan.pdf", curve_fan()),
            ("vector", "clip-tiles.pdf", clip_tiles()),
            ("vector", "clip-repeat.pdf", clip_repeat()),
            ("vector", "page-rects.pdf", page_rects()),
            ("vector", "offpage-zoom.pdf", offpage_zoom()),
            ("vector", "form-repeat.pdf", form_repeat()),
            ("vector", "pattern-shade-alpha.pdf", pattern_shade_alpha()),
            ("text", "text-modes.pdf", text_modes()),
            ("text", "text-rotated.pdf", text_rotated()),
            ("text", "text-substituted.pdf", text_substituted()),
            ("text", "text-type3.pdf", text_type3()),
            (
                "text",
                "text-truetype.pdf",
                embedded_glyph_page("TrueType", "FontFile2", minimal_ttf(), 1000, "/Encoding/WinAnsiEncoding"),
            ),
            (
                "text",
                "text-type1.pdf",
                embedded_glyph_page("Type1", "FontFile", minimal_type1(), 600, ""),
            ),
        ];
        for (dir, name, bytes) in &list {
            let d = std::path::Path::new(&root).join(dir);
            if let Err(e) = std::fs::create_dir_all(&d) {
                eprintln!("probe synth: cannot create {}: {e}", d.display());
                std::process::exit(1);
            }
            if let Err(e) = std::fs::write(d.join(name), bytes) {
                eprintln!("probe synth: cannot write {name}: {e}");
                std::process::exit(1);
            }
            println!("{dir}/{name}\t{}", bytes.len());
        }
    }
}
