//! Benchmark comparing Classic vs Streaming decode strategies.
//! Run with: `cargo bench --bench strategy_compare`
//!
//! Synthetic data generators fitted to real-world LZW workloads. The
//! current archetypes target scanned document imaging — specifically
//! the RVL-CDIP test2 slice (50 email + 50 form from
//! `chainyo/rvl-cdip` test-00002-of-00015.parquet), measured by
//! `tools/analyze_scanned_pages.py`. See
//! `docs/rvl-cdip-test2-analysis.md` for the analysis.
//!
//! Two-state Markov process (BG white + FG ink):
//!   1. BG state emits byte 255 most of the time, with a small chance
//!      of emitting a uniform random non-255 byte (scanner sensor noise
//!      — this is what makes every real scan have distinct_values=256
//!      even on blank pages).
//!   2. With probability `p_bg_to_fg`, transition to FG and pick an ink
//!      base byte uniformly from [fg_lo, fg_hi].
//!   3. FG state emits `fg_base + uniform(-jitter, +jitter)` so ink
//!      stretches share a narrow tone (antialiasing / gradient banding),
//!      which LZW can compress into short dictionary patterns.
//!   4. With probability `p_fg_to_bg`, return to BG.
//!
//! Parameters were fitted in `tools/fit_generator.py` by a grid search
//! that measures each trial via Pillow TIFF-LZW re-encode and matches
//! the class medians of entropy, repeat_frac, rl_mean, rl_long_frac,
//! mode_frac, and lzw_ratio. Every fitted archetype is within ±13% on
//! each axis (most within ±2%); see the fit report printed by
//! `tools/fit_generator.py` and the per-archetype comment blocks below.

use std::sync::Arc;
use weezl::{
    decode::{Configuration, TableStrategy},
    encode::Encoder,
    BitOrder, LzwStatus,
};
use zenbench::prelude::*;

fn decode_all(
    encoded: &[u8],
    out: &mut [u8],
    order: BitOrder,
    tiff: bool,
    strategy: TableStrategy,
) -> usize {
    let config = if tiff {
        Configuration::with_tiff_size_switch(order, 8)
    } else {
        Configuration::new(order, 8)
    };
    let mut dec = config.with_table_strategy(strategy).build();
    let mut inp = encoded;
    let mut cursor = out;
    let mut written = 0;
    loop {
        let r = dec.decode_bytes(inp, cursor);
        inp = &inp[r.consumed_in..];
        written += r.consumed_out;
        cursor = &mut std::mem::take(&mut cursor)[r.consumed_out..];
        match r.status {
            Ok(LzwStatus::Done | LzwStatus::NoProgress) => return written,
            Ok(LzwStatus::Ok) => {
                if inp.is_empty() && cursor.is_empty() {
                    return written;
                }
            }
            Err(_) => return written,
        }
    }
}

// ---------------------------------------------------------------------------
// Two-state Markov generator (BG white + FG ink) with scanner-noise model.
// ---------------------------------------------------------------------------
//
// See tools/fit_generator.py for the Python reference implementation —
// the two share the same xorshift32 sequence and the same branching
// structure, so Rust output matches what the fitter measured.

struct GenParams {
    /// Probability, per BG pixel, of transitioning BG → FG.
    p_bg_to_fg: f64,
    /// Probability, per FG pixel, of transitioning FG → BG.
    p_fg_to_bg: f64,
    /// Probability, per BG pixel, of emitting a uniform-random byte in
    /// 0..=254 instead of 255. Simulates scanner sensor noise; real
    /// scans have distinct_values = 256 even on nearly-blank pages.
    bg_noise_p: f64,
    /// FG base byte picked once per FG stretch, uniform in [fg_lo, fg_hi].
    fg_lo: u8,
    fg_hi: u8,
    /// FG pixels within a stretch are `fg_base + uniform(-jitter, +jitter)`,
    /// clamped to 0..=255. Controls how much LZW can compress ink runs.
    fg_jitter: u8,
}

/// xorshift32 PRNG — deterministic, no deps. Must match tools/fit_generator.py.
struct Rng(u32);
impl Rng {
    fn new(seed: u32) -> Self {
        Self(seed | 1)
    }
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        self.0
    }
}

fn generate(params: &GenParams, len: usize, seed: u32) -> Vec<u8> {
    let mut rng = Rng::new(seed);
    let u = u32::MAX as f64;
    let fg_lo = params.fg_lo as i16;
    let fg_hi = params.fg_hi as i16;
    let fg_span = (fg_hi - fg_lo + 1).max(1) as u32;
    let jitter = params.fg_jitter as i16;
    let jitter_span = (2 * jitter + 1).max(1) as u32;

    let mut out = Vec::with_capacity(len);
    let mut state = 0u8; // 0 = BG, 1 = FG
    let mut fg_base: i16 = 0;

    while out.len() < len {
        if state == 0 {
            // BG: mostly 255, rarely a uniform-random non-255 byte.
            let byte = if (rng.next() as f64 / u) < params.bg_noise_p {
                (rng.next() % 255) as u8 // 0..=254
            } else {
                255
            };
            out.push(byte);
            if (rng.next() as f64 / u) < params.p_bg_to_fg {
                state = 1;
                fg_base = fg_lo + (rng.next() % fg_span) as i16;
            }
        } else {
            // FG: ink tone with jitter around the stretch's base.
            let delta = (rng.next() % jitter_span) as i16 - jitter;
            let v = (fg_base + delta).clamp(0, 255) as u8;
            out.push(v);
            if (rng.next() as f64 / u) < params.p_fg_to_bg {
                state = 0;
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Fitted archetypes — RVL-CDIP scanned documents (test2 slice).
//
// Targets are medians over a 50-file slice per class / bucket, measured
// with `tools/analyze_scanned_pages.py` (byte entropy, repeat fraction,
// run-length mean + long_frac, mode fraction, Pillow TIFF-LZW ratio).
// Parameters were fitted with `tools/fit_generator.py --length 262144`.
//
// Fit quality (measured at len=262144, seed=0xDEADBEEF, same generator
// as the one below):
//
//   archetype         metric           target   measured  rel_err
//   SCANNED_EMAIL     entropy          0.394    0.390     1.1%
//                     repeat_frac      0.968    0.968     0.0%
//                     rl_mean         31.28    31.16      0.4%
//                     rl_long_frac     0.964    0.970     0.7%
//                     mode_frac        0.972    0.972     0.0%
//                     lzw_ratio       22.10    21.62      2.2%
//   SCANNED_BLANK     entropy          0.123    0.107    13.0%
//                     repeat_frac      0.991    0.991     0.0%
//                     rl_mean        113.37   113.24      0.1%
//                     rl_long_frac     0.991    0.993     0.2%
//                     mode_frac        0.992    0.993     0.1%
//                     lzw_ratio       56.80    56.95      0.3%
//   SCANNED_DENSE     entropy          1.115    1.063     4.7%
//                     repeat_frac      0.894    0.895     0.2%
//                     rl_mean          9.42     9.56      1.4%
//                     rl_long_frac     0.876    0.897     2.4%
//                     mode_frac        0.912    0.913     0.1%
//                     lzw_ratio        7.30     7.28      0.3%
//
// All axes within ±13%; most within ±5%. Blank's 13% entropy miss is
// because the generator models sensor noise as i.i.d. Bernoulli while
// real scanners produce spatially-correlated noise that lifts the
// byte histogram without breaking long runs as aggressively — fixing
// this would need a second-order model (e.g. block-level noise).
// ---------------------------------------------------------------------------

/// Email class median (n=50). Target: H≈0.39, rl=31, mode=0.972, ratio≈22×.
/// Models typical text document with ~3% ink coverage and short dark
/// antialiased glyphs on a clean white background.
const SCANNED_EMAIL: GenParams = GenParams {
    p_bg_to_fg: 0.0070,
    p_fg_to_bg: 0.25,
    bg_noise_p: 0.0004,
    fg_lo: 0,
    fg_hi: 150,
    fg_jitter: 3,
};

/// "Blank form" — median of the 15 form files with lzw_ratio ∈ [50, 80].
/// Target: H≈0.12, rl=113, mode=0.992, ratio≈57×. Models a mostly-blank
/// preprinted form with occasional faint rule lines or sparse text.
/// Represents the high-compression tail where chunked/streaming decode
/// tables matter most (long KwKwK runs of 255).
const SCANNED_BLANK: GenParams = GenParams {
    p_bg_to_fg: 0.0017,
    p_fg_to_bg: 0.25,
    bg_noise_p: 0.0008,
    fg_lo: 0,
    fg_hi: 120,
    fg_jitter: 12,
};

/// "Dense" document — e.g. email_003 (heavy text, low ratio).
/// Target: H≈1.12, rl=9.4, mode=0.91, ratio≈7.3×. Models a page with
/// ~9% ink coverage — dense paragraph text, filled form fields, or a
/// scanned letter with small type. Represents the low-compression
/// tail where the chunked table's extra bookkeeping is pure overhead.
const SCANNED_DENSE: GenParams = GenParams {
    p_bg_to_fg: 0.02325,
    p_fg_to_bg: 0.25,
    bg_noise_p: 0.0021,
    fg_lo: 0,
    fg_hi: 150,
    fg_jitter: 6,
};

// ---------------------------------------------------------------------------
// Bench harness
// ---------------------------------------------------------------------------

struct Workload {
    name: &'static str,
    encoded: Arc<Vec<u8>>,
    decoded_size: usize,
    order: BitOrder,
    tiff: bool,
}

fn make_workload(name: &'static str, data: &[u8], order: BitOrder, tiff: bool) -> Workload {
    let encoded = if tiff {
        Encoder::with_tiff_size_switch(order, 8)
            .encode(data)
            .unwrap()
    } else {
        Encoder::new(order, 8).encode(data).unwrap()
    };
    let mut scratch = vec![0u8; data.len() + 4096];
    let decoded_size = decode_all(&encoded, &mut scratch, order, tiff, TableStrategy::Classic);
    Workload {
        name,
        encoded: Arc::new(encoded),
        decoded_size,
        order,
        tiff,
    }
}

fn bench_workload(g: &mut BenchGroup, w: &Workload) {
    g.throughput(Throughput::Bytes(w.decoded_size as u64));
    // Bump min_sample_ns to 10ms (zenbench default is 5ms): solid-color
    // KwKwK iterations are so short that 5ms gives noisy aggregates;
    // 10ms gives a tight CV across processes. Run with
    // `--best-of-processes=3` on a busy host or `--mean-of-processes=5`
    // on a quiet one for the most stable aggregated numbers.
    g.config().min_sample_ns(10_000_000);
    g.config().max_rounds(500);
    g.config().min_rounds(100);
    let out_cap = w.decoded_size + 4096;

    for &(label, strategy) in &[
        ("classic", TableStrategy::Classic),
        ("chunked", TableStrategy::Chunked),
        ("streaming", TableStrategy::Streaming),
    ] {
        let enc = Arc::clone(&w.encoded);
        let order = w.order;
        let tiff = w.tiff;
        g.bench(label, move |b| {
            let enc = Arc::clone(&enc);
            let mut out = vec![0u8; out_cap];
            b.iter(move || {
                let n = decode_all(&enc, &mut out, order, tiff, strategy);
                black_box(&out[..n]);
                n
            });
        });
    }
}

fn bench_strategies(suite: &mut Suite) {
    let size = 256 * 1024;
    let seed = 0xDEADBEEF;

    let workloads = vec![
        // MSB + TIFF — image-tiff configuration. Three scanned-document
        // archetypes spanning the range of real RVL-CDIP content:
        // blank (rare, max compression) / email (class median) /
        // dense (low-compression text-heavy tail). Plus a solid-color
        // KwKwK baseline to stress the hottest decode path.
        make_workload(
            "scanned-email",
            &generate(&SCANNED_EMAIL, size, seed),
            BitOrder::Msb,
            true,
        ),
        make_workload(
            "scanned-blank",
            &generate(&SCANNED_BLANK, size, seed),
            BitOrder::Msb,
            true,
        ),
        make_workload(
            "scanned-dense",
            &generate(&SCANNED_DENSE, size, seed),
            BitOrder::Msb,
            true,
        ),
        make_workload("solid-kwkwk", &vec![42u8; size], BitOrder::Msb, true),
        // LSB — GIF configuration. Only the email archetype, since GIF
        // consumers hit the same byte-255-dominated distribution when
        // decoding scanned-document GIFs exported from TIFF pipelines.
        make_workload(
            "scanned-email",
            &generate(&SCANNED_EMAIL, size, seed),
            BitOrder::Lsb,
            false,
        ),
    ];

    for w in &workloads {
        let mode = if w.tiff { "tiff" } else { "gif" };
        suite.group(format!("{}/{}", mode, w.name), |g| bench_workload(g, w));
    }
}

zenbench::main!(bench_strategies);
