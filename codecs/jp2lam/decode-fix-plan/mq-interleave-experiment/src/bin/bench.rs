//! Multi-stream MQ replay benchmark.
//!
//! Replays recorded (bytes, context trace) pairs of real code blocks through
//! the production `MqDecoder`, either one block after another or with N
//! independent decoders interleaved on one core, and checks every decoded bit
//! and each decoder's final state against the recording and the sequential
//! run. See README.md for how traces are recorded and what was measured.
//!
//! ```text
//! cargo build --release
//! taskset -c 2 target/release/bench <file.trace>...
//! ```
//! Environment: `REPS` (timed repetitions, default 15), `ONLY`
//! (`fp-free`, `bl-free`, `fp-dep`, `bl-dep`, `fp-branchy`, `bl-branchy`),
//! `LANES` (one lane count, per-symbol interleave only).
#[path = "../../../../src/mq/mod.rs"]
#[allow(dead_code, unused, unexpected_cfgs, clippy::all)]
mod mq;
use mq::MqDecoder;
use std::hint::black_box;
use std::time::Instant;

struct Block {
    bytes: Vec<u8>,
    ctxs: Vec<u8>,
    bits: Vec<u8>,
}

fn load(path: &str) -> Vec<Block> {
    let d = std::fs::read(path).unwrap();
    let mut p = 0;
    let mut v = Vec::new();
    while p < d.len() {
        let nb = u32::from_le_bytes(d[p..p + 4].try_into().unwrap()) as usize;
        let ns = u32::from_le_bytes(d[p + 4..p + 8].try_into().unwrap()) as usize;
        p += 8;
        let bytes = d[p..p + nb].to_vec();
        p += nb;
        let ctxs = d[p..p + ns].to_vec();
        p += ns;
        let bits = d[p..p + ns].to_vec();
        p += ns;
        v.push(Block { bytes, ctxs, bits });
    }
    v
}

/// `DEP[ctx | bit << 5] == ctx`: an opaque table that makes the next context
/// a load that depends on the previous decoded bit, as Tier-1's context
/// formation does (flag word update, then LUT).
static DEP: [u8; 64] = {
    let mut t = [0u8; 64];
    let mut i = 0;
    while i < 64 {
        t[i] = (i & 31) as u8;
        i += 1;
    }
    t
};

#[derive(Clone, Copy, PartialEq)]
enum Chain {
    /// Context comes straight from the trace.
    Free,
    /// Context is loaded through a table indexed by the previous bit.
    Dependent,
    /// As `Dependent`, plus a conditional store taken when the bit is 1: the
    /// data-dependent branch Tier-1 makes on every significance decision.
    Branchy,
}

#[inline(always)]
fn ctx_of(dep: &[u8; 64], chain: Chain, raw: u8, prev: u8) -> u8 {
    match chain {
        Chain::Free => raw,
        Chain::Dependent | Chain::Branchy => dep[(raw as usize | ((prev as usize) << 5)) & 63],
    }
}

#[inline(always)]
fn dec<const BL: bool>(d: &mut MqDecoder<'_>, ctx: u8) -> u8 {
    if BL {
        d.decode_unpredictable(ctx)
    } else {
        d.decode_with_ctx(ctx)
    }
}

/// The decoder's full state as text, taken from a copy so that the live
/// decoder's address never escapes and its registers stay registers.
#[inline(never)]
fn snapshot_copy(d: MqDecoder<'_>) -> String {
    format!("{d:?}")
}
#[inline(always)]
fn snapshot(d: &MqDecoder<'_>) -> String {
    snapshot_copy(d.clone())
}

static mut SIDE: [u32; 16] = [0; 16];

/// What Tier-1 does when a significance decision comes out 1: something
/// behind a branch the predictor cannot learn.
#[inline(always)]
fn side_effect(chain: Chain, bit: u8, side: *mut u32) {
    if chain == Chain::Branchy && bit != 0 {
        // SAFETY: single-threaded benchmark, `side` points at SIDE.
        unsafe {
            let slot = side.add((bit as usize) & 15);
            std::ptr::write_volatile(slot, std::ptr::read_volatile(slot).wrapping_add(1));
        }
    }
}

/// One block after another. Returns final decoder states when asked.
#[inline(never)]
fn sequential<const BL: bool>(blocks: &[Block], out: &mut [Vec<u8>], chain: Chain, states: Option<&mut Vec<String>>) {
    let dep = black_box(&DEP);
    let side = black_box(std::ptr::addr_of_mut!(SIDE) as *mut u32);
    let mut states = states;
    for (block, out) in blocks.iter().zip(out.iter_mut()) {
        let mut d = MqDecoder::new(&block.bytes);
        let mut prev = 0u8;
        for (&raw, o) in block.ctxs.iter().zip(out.iter_mut()) {
            let bit = dec::<BL>(&mut d, ctx_of(dep, chain, raw, prev));
            *o = bit;
            prev = bit;
            side_effect(chain, bit, side);
        }
        if let Some(s) = states.as_deref_mut() {
            s.push(snapshot(&d));
        }
    }
}

/// N decoders interleaved, `batch` symbols per decoder per turn. A lane that
/// finishes its block takes the next undecoded block; when the blocks run out
/// the remaining lanes finish one after another. Returns the number of
/// symbols decoded inside the interleaved loop.
#[inline(never)]
fn interleaved<const N: usize, const BL: bool>(
    blocks: &[Block],
    out: &mut [Vec<u8>],
    batch: usize,
    chain: Chain,
    mut states: Option<&mut Vec<String>>,
) -> u64 {
    let dep = black_box(&DEP);
    let side = black_box(std::ptr::addr_of_mut!(SIDE) as *mut u32);
    if let Some(s) = states.as_deref_mut() {
        s.resize(blocks.len(), String::new());
    }
    let out_ptrs: Vec<*mut u8> = out.iter_mut().map(|v| v.as_mut_ptr()).collect();
    assert!(blocks.len() >= N);
    let mut next = N;
    let mut cur: [usize; N] = std::array::from_fn(|l| l);
    let mut pos = [0usize; N];
    let mut prev = [0u8; N];
    let mut d: [MqDecoder; N] = std::array::from_fn(|l| MqDecoder::new(&blocks[l].bytes));
    let mut inter = 0u64;
    'outer: loop {
        let run = (0..N).map(|l| blocks[cur[l]].ctxs.len() - pos[l]).min().unwrap();
        let ctx: [*const u8; N] = std::array::from_fn(|l| unsafe { blocks[cur[l]].ctxs.as_ptr().add(pos[l]) });
        let dst: [*mut u8; N] = std::array::from_fn(|l| unsafe { out_ptrs[cur[l]].add(pos[l]) });
        // SAFETY: `run` symbols remain in every lane's trace and output.
        unsafe {
            if batch == 1 {
                for s in 0..run {
                    for l in 0..N {
                        let bit = dec::<BL>(&mut d[l], ctx_of(dep, chain, *ctx[l].add(s), prev[l]));
                        *dst[l].add(s) = bit;
                        prev[l] = bit;
                        side_effect(chain, bit, side);
                    }
                }
            } else {
                let mut s = 0;
                while s < run {
                    let e = (s + batch).min(run);
                    for l in 0..N {
                        for k in s..e {
                            let bit = dec::<BL>(&mut d[l], ctx_of(dep, chain, *ctx[l].add(k), prev[l]));
                            *dst[l].add(k) = bit;
                            prev[l] = bit;
                        side_effect(chain, bit, side);
                        }
                    }
                    s = e;
                }
            }
        }
        inter += (run * N) as u64;
        for l in 0..N {
            pos[l] += run;
        }
        for l in 0..N {
            if pos[l] == blocks[cur[l]].ctxs.len() {
                if let Some(s) = states.as_deref_mut() {
                    s[cur[l]] = snapshot(&d[l]);
                }
                if next == blocks.len() {
                    // Out of blocks: finish the other lanes one by one.
                    for m in 0..N {
                        if m == l {
                            continue;
                        }
                        let b = &blocks[cur[m]];
                        for k in pos[m]..b.ctxs.len() {
                            let bit = dec::<BL>(&mut d[m], ctx_of(dep, chain, b.ctxs[k], prev[m]));
                            unsafe { *out_ptrs[cur[m]].add(k) = bit };
                            prev[m] = bit;
                            side_effect(chain, bit, side);
                        }
                        pos[m] = b.ctxs.len();
                        if let Some(s) = states.as_deref_mut() {
                            s[cur[m]] = snapshot(&d[m]);
                        }
                    }
                    break 'outer;
                }
                cur[l] = next;
                next += 1;
                pos[l] = 0;
                prev[l] = 0;
                d[l] = MqDecoder::new(&blocks[cur[l]].bytes);
            }
        }
    }
    inter
}

fn run_mode<const BL: bool>(blocks: &[Block], out: &mut [Vec<u8>], lanes: usize, batch: usize, chain: Chain, states: Option<&mut Vec<String>>) -> u64 {
    match lanes {
        1 => {
            sequential::<BL>(blocks, out, chain, states);
            0
        }
        2 => interleaved::<2, BL>(blocks, out, batch, chain, states),
        3 => interleaved::<3, BL>(blocks, out, batch, chain, states),
        4 => interleaved::<4, BL>(blocks, out, batch, chain, states),
        6 => interleaved::<6, BL>(blocks, out, batch, chain, states),
        8 => interleaved::<8, BL>(blocks, out, batch, chain, states),
        _ => panic!("lanes"),
    }
}

fn run(bl: bool, blocks: &[Block], out: &mut [Vec<u8>], lanes: usize, batch: usize, chain: Chain, states: Option<&mut Vec<String>>) -> u64 {
    if bl { run_mode::<true>(blocks, out, lanes, batch, chain, states) } else { run_mode::<false>(blocks, out, lanes, batch, chain, states) }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let reps: usize = std::env::var("REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(15);
    let modes: Vec<(usize, usize)> = vec![(1, 1), (2, 1), (3, 1), (4, 1), (6, 1), (8, 1), (2, 2), (2, 4), (2, 8), (4, 4)];
    // Clock calibration: a dependent add chain runs one add per cycle.
    {
        let mut x = black_box(1u64);
        let n = 400_000_000u64;
        let t = Instant::now();
        for i in 0..n {
            x = black_box(x.wrapping_add(i | 1));
        }
        let e = t.elapsed().as_secs_f64();
        black_box(x);
        println!("calibration: {:.2} ns per dependent add iteration (upper bound on cycle time)", e * 1e9 / n as f64);
    }
    for path in &args {
        let blocks = load(path);
        let total: u64 = blocks.iter().map(|b| b.ctxs.len() as u64).sum();
        let bytes: u64 = blocks.iter().map(|b| b.bytes.len() as u64).sum();
        let mut classes = [0u64; 5];
        for b in &blocks {
            for &c in &b.ctxs {
                classes[match c { 0..=8 => 0, 9..=13 => 1, 14..=16 => 2, 17 => 3, _ => 4 }] += 1;
            }
        }
        let name = std::path::Path::new(path).file_stem().unwrap().to_string_lossy().to_string();
        println!(
            "== {name}: {} blocks, {total} symbols, {bytes} bytes ({:.2} bits/symbol); ZC {:.0}% SC {:.0}% MAG {:.0}% AGG {:.0}% UNI {:.0}%",
            blocks.len(),
            bytes as f64 * 8.0 / total as f64,
            classes[0] as f64 * 100.0 / total as f64,
            classes[1] as f64 * 100.0 / total as f64,
            classes[2] as f64 * 100.0 / total as f64,
            classes[3] as f64 * 100.0 / total as f64,
            classes[4] as f64 * 100.0 / total as f64,
        );
        let only = std::env::var("ONLY").ok(); // e.g. "bl-free" / "fp-dep"
        for (chain, bl) in [(Chain::Free, false), (Chain::Free, true), (Chain::Dependent, false), (Chain::Dependent, true), (Chain::Branchy, false), (Chain::Branchy, true)] {
            let tag = format!("{}-{}", if bl { "bl" } else { "fp" }, match chain { Chain::Free => "free", Chain::Dependent => "dep", Chain::Branchy => "branchy" });
            if only.as_deref().is_some_and(|o| o != tag) { continue; }
            println!("  decoder: {}; context chain: {}", if bl { "branch-free (decode_unpredictable)" } else { "fast-path (decode_with_ctx)" }, match chain { Chain::Free => "free (trace only)", Chain::Dependent => "dependent on previous bit (table load)", Chain::Branchy => "dependent on previous bit, plus a branch on each bit" });
            // Reference states from the sequential run.
            let mut out: Vec<Vec<u8>> = blocks.iter().map(|b| vec![2u8; b.ctxs.len()]).collect();
            let mut ref_states = Vec::new();
            run(bl, &blocks, &mut out, 1, 1, chain, Some(&mut ref_states));
            for (b, o) in blocks.iter().zip(&out) {
                assert!(b.bits == *o, "sequential replay mismatch");
            }
            let mut base = 0.0;
            let lanes_only: Option<usize> = std::env::var("LANES").ok().and_then(|v| v.parse().ok());
            for &(lanes, batch) in &modes {
                if lanes_only.is_some_and(|l| (l != lanes && lanes != 1) || batch != 1) { continue; }
                // Correctness: every bit and every final decoder state.
                for o in out.iter_mut() {
                    o.fill(2);
                }
                let mut states = Vec::new();
                let inter = run(bl, &blocks, &mut out, lanes, batch, chain, Some(&mut states));
                for (i, (b, o)) in blocks.iter().zip(&out).enumerate() {
                    assert!(b.bits == *o, "bit mismatch lanes={lanes} batch={batch} block={i}");
                }
                assert!(states == ref_states, "state mismatch lanes={lanes} batch={batch}");
                // Timing.
                let mut best = f64::MAX;
                let mut all = Vec::new();
                for _ in 0..reps {
                    let t = Instant::now();
                    run(bl, black_box(&blocks), &mut out, lanes, batch, chain, None);
                    let e = t.elapsed().as_secs_f64();
                    all.push(e);
                    best = best.min(e);
                }
                all.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let med = all[all.len() / 2];
                if lanes == 1 {
                    base = best;
                }
                println!(
                    "    lanes {lanes} batch {batch:>2}: {:6.2} ns/sym (median {:6.2})  {:6.1} Msym/s  x{:.3} vs sequential  [{:.1}% of symbols interleaved]",
                    best * 1e9 / total as f64,
                    med * 1e9 / total as f64,
                    total as f64 / best / 1e6,
                    base / best,
                    inter as f64 * 100.0 / total as f64,
                );
            }
        }
    }
}
