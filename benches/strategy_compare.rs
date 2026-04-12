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
    /// Probability, per clean-BG pixel, of starting a noise burst
    /// (transitioning BG → BG_NOISE). Simulates scanner sensor noise;
    /// real scans have distinct_values = 256 even on nearly-blank
    /// pages.
    bg_burst_p: f64,
    /// Probability, per BG_NOISE pixel, of ending the burst and
    /// returning to clean BG. `1.0` reproduces the old i.i.d.
    /// Bernoulli noise (each noisy pixel is its own burst of length
    /// one). Smaller values cluster noise spatially: a single burst
    /// start event produces a geometrically-distributed run of noise
    /// pixels, which shows up in real scans as a CCD-row bias band
    /// or a patch of near-white sensor grit. Spatial clustering lets
    /// the model pay for more noise bytes (higher byte-level entropy)
    /// without breaking long 255 runs as aggressively as i.i.d. noise
    /// does.
    ///
    /// Math: with Bernoulli, each noise byte breaks one run of 255s
    /// into two, so N noise bytes cost ~N run-break events. With a
    /// burst of length L, one start + L within-burst byte changes +
    /// one end = L+1 run-break events for L noise bytes ≈ 1 break/byte
    /// asymptotically. So a burst length of 2 already halves the
    /// run-length cost of adding a noise byte.
    bg_burst_end_p: f64,
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
    // 0 = BG (clean white), 1 = FG (ink stretch), 2 = BG_NOISE (sensor
    // noise burst — sits inside BG logically, only transitions to BG).
    let mut state = 0u8;
    let mut fg_base: i16 = 0;

    while out.len() < len {
        match state {
            0 => {
                // Clean BG: emit 255, optionally start an FG stretch or
                // a noise burst. FG takes priority over noise (if both
                // would trigger, we start the FG stretch) — the
                // probabilities are small enough that ordering rarely
                // matters, but the deterministic branch order has to
                // match tools/fit_generator.py byte-for-byte.
                out.push(255);
                if (rng.next() as f64 / u) < params.p_bg_to_fg {
                    state = 1;
                    fg_base = fg_lo + (rng.next() % fg_span) as i16;
                } else if (rng.next() as f64 / u) < params.bg_burst_p {
                    state = 2;
                }
            }
            1 => {
                // FG: ink tone with jitter around the stretch's base.
                let delta = (rng.next() % jitter_span) as i16 - jitter;
                let v = (fg_base + delta).clamp(0, 255) as u8;
                out.push(v);
                if (rng.next() as f64 / u) < params.p_fg_to_bg {
                    state = 0;
                }
            }
            _ => {
                // BG_NOISE: emit a uniform-random byte in 0..=254 (never
                // 255, so the noise byte can't pretend to be white). The
                // burst ends with probability `bg_burst_end_p` per
                // pixel — geometric distribution of burst lengths.
                out.push((rng.next() % 255) as u8);
                if (rng.next() as f64 / u) < params.bg_burst_end_p {
                    state = 0;
                }
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
// as the one below, three-state Markov with burst-mode BG noise):
//
//   archetype         metric           target   measured  rel_err
//   SCANNED_EMAIL     entropy          0.394    0.375     4.9%
//                     repeat_frac      0.968    0.968     0.0%
//                     rl_mean         31.28    31.18      0.3%
//                     rl_long_frac     0.964    0.972     0.8%
//                     mode_frac        0.972    0.973     0.1%
//                     lzw_ratio       22.10    21.24      3.9%
//   SCANNED_BLANK     entropy          0.123    0.121     1.8% *
//                     repeat_frac      0.991    0.991     0.0%
//                     rl_mean        113.37   110.47      2.6%
//                     rl_long_frac     0.991    0.992     0.1%
//                     mode_frac        0.992    0.992     0.0%
//                     lzw_ratio       56.80    56.18      1.1%
//   SCANNED_DENSE     entropy          1.115    1.097     1.6%
//                     repeat_frac      0.894    0.893     0.1%
//                     rl_mean          9.42     9.34      0.9%
//                     rl_long_frac     0.876    0.895     2.2%
//                     mode_frac        0.912    0.909     0.3%
//                     lzw_ratio        7.30     7.24      0.9%
//
// All axes within 4.9%; most within 2%. The `*` on SCANNED_BLANK's
// entropy marks the one that needed the burst-mode BG_NOISE
// substate (previously a 13% miss with i.i.d. Bernoulli noise — the
// fitter couldn't afford enough noise bytes to reach the entropy
// target without shortening the run-length mean below the 113 target).
// With burst mean ~13 pixels, each noise byte costs ~1.1 run breaks
// instead of ~2, so the fitter can spend about twice as many noise
// bytes for the same rl_mean budget.
// ---------------------------------------------------------------------------

/// Email class median (n=50). Target: H≈0.39, rl=31, mode=0.972, ratio≈22×.
/// Models typical text document with ~3% ink coverage and short dark
/// antialiased glyphs on a clean white background.
///
/// `bg_burst_end_p = 1.0` — noise is i.i.d. (each noisy BG pixel is
/// its own length-1 burst). Email's entropy target is comfortably
/// reached without spatial clustering because the FG ink contributes
/// the dominant non-255 byte share; burst model would be overkill.
const SCANNED_EMAIL: GenParams = GenParams {
    p_bg_to_fg: 0.006510,
    p_fg_to_bg: 0.25,
    bg_burst_p: 0.000781,
    bg_burst_end_p: 1.0,
    fg_lo: 0,
    fg_hi: 150,
    fg_jitter: 5,
};

/// "Blank form" — median of the 15 form files with lzw_ratio ∈ [50, 80].
/// Target: H≈0.12, rl=113, mode=0.992, ratio≈57×. Models a mostly-blank
/// preprinted form with occasional faint rule lines or sparse text.
/// Represents the high-compression tail where chunked/streaming decode
/// tables matter most (long KwKwK runs of 255).
///
/// `bg_burst_end_p = 0.075` — noise is spatially clustered (mean burst
/// length ≈ 13 pixels). Real blank scans have almost no FG ink, so
/// BG_NOISE has to carry the entire byte-level entropy budget by
/// itself. The earlier i.i.d. model missed the target entropy by 13%
/// because each individual noise byte broke a run in half, and the
/// fitter had to choose between "noise enough bytes to hit entropy"
/// and "preserve 113-byte mean run length." Bursts cost ~1.1 run-
/// breaks per noise byte (one start + ~L within + one end for L
/// bytes) vs Bernoulli's ~2 breaks/byte, so the fitter can spend
/// about twice as many noise bytes for the same rl_mean budget.
/// Post-burst fit error drops to 1.8% on entropy.
const SCANNED_BLANK: GenParams = GenParams {
    p_bg_to_fg: 0.001700,
    p_fg_to_bg: 0.25,
    bg_burst_p: 0.000075,
    bg_burst_end_p: 0.075,
    fg_lo: 0,
    fg_hi: 120,
    fg_jitter: 10,
};

/// "Dense" document — e.g. email_003 (heavy text, low ratio).
/// Target: H≈1.12, rl=9.4, mode=0.91, ratio≈7.3×. Models a page with
/// ~9% ink coverage — dense paragraph text, filled form fields, or a
/// scanned letter with small type. Represents the low-compression
/// tail where the chunked table's extra bookkeeping is pure overhead.
///
/// `bg_burst_p = 0.0` — FG coverage at 9% is already carrying the
/// entire entropy budget, so the fitter drove bg_burst_p to zero.
/// Dense content doesn't need a sensor-noise model at all; the
/// byte histogram is dominated by ink distribution, not white-page
/// grit.
const SCANNED_DENSE: GenParams = GenParams {
    p_bg_to_fg: 0.025000,
    p_fg_to_bg: 0.25,
    bg_burst_p: 0.0,
    bg_burst_end_p: 1.0,
    fg_lo: 0,
    fg_hi: 150,
    fg_jitter: 5,
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
    // `--best-of-passes=3` on a busy host or `--mean-of-passes=5`
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
