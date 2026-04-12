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
    /// Number of fixed byte patterns in the "glyph library." Each
    /// pattern is a deterministic byte sequence (generated from the
    /// PRNG seed at init) that simulates a character glyph — the same
    /// pixel block that repeats every time that "letter" appears on
    /// a page. Patterns give the LZW dictionary repeating entries to
    /// match, driving down `literal_frac` and up `short_copy_frac`
    /// to match real scanned documents. 0 = disabled (backward compat).
    n_patterns: u32,
    /// Length of each pattern in bytes (e.g. 6 ≈ one glyph's pixel
    /// footprint at low res). Only meaningful when `n_patterns > 0`.
    pattern_len: u32,
    /// When entering FG, probability of emitting a pattern instead of
    /// random jitter bytes. 0.0 = always jitter (backward compat).
    pattern_frac: f64,
    /// Row width for scanline-repeat mode. When > 0, generates
    /// `n_template_rows` template rows with the Markov model, then
    /// tiles them with per-pixel ink noise. Simulates real scanned
    /// text where each row of pixels through a text line shares the
    /// same column structure. 0 = flat stream (backward compat).
    row_width: u32,
    /// Number of distinct template rows. Adjacent rows use different
    /// templates (simulating different text lines), cycling every N
    /// rows. More templates = more inter-row diversity (higher literal)
    /// but less periodic repetition (lower width_12).
    n_template_rows: u32,
    /// Per-pixel noise on ink (non-255) pixels during row tiling.
    /// Creates row-to-row variation that fills the LZW dictionary.
    row_noise: u8,
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
    // Row-repeat mode: generate N template rows, tile with ink noise.
    if params.row_width > 0 {
        let rw = params.row_width as usize;
        let n_tpl = (params.n_template_rows as usize).max(1);
        let inner = GenParams {
            row_width: 0,
            n_template_rows: 0,
            row_noise: 0,
            ..*params
        };
        let templates: Vec<Vec<u8>> = (0..n_tpl)
            .map(|t| generate(&inner, rw, seed.wrapping_add(t as u32)))
            .collect();
        let noise = params.row_noise as i16;
        let noise_span = (2 * noise + 1).max(1) as u32;
        let mut rng = Rng::new(seed);
        let mut out = Vec::with_capacity(len);
        for i in 0..len {
            let row = i / rw;
            let v = templates[row % n_tpl][i % rw];
            let v = if noise > 0 && v != 255 {
                let delta = (rng.next() % noise_span) as i16 - noise;
                (v as i16 + delta).clamp(0, 255) as u8
            } else {
                v
            };
            out.push(v);
        }
        return out;
    }

    let mut rng = Rng::new(seed);
    let u = u32::MAX as f64;
    let fg_lo = params.fg_lo as i16;
    let fg_hi = params.fg_hi as i16;
    let fg_span = (fg_hi - fg_lo + 1).max(1) as u32;
    let jitter = params.fg_jitter as i16;
    let jitter_span = (2 * jitter + 1).max(1) as u32;

    // Build pattern library from the PRNG stream. Each pattern is a
    // fixed byte sequence that simulates a character glyph — the same
    // pixel block repeated every time that "letter" appears. Only
    // consumes PRNG values when n_patterns > 0, keeping the stream
    // identical for backward-compatible parameter sets.
    let patterns: Vec<Vec<u8>> = if params.n_patterns > 0 && params.pattern_frac > 0.0 {
        (0..params.n_patterns)
            .map(|_| {
                let base = fg_lo + (rng.next() % fg_span) as i16;
                (0..params.pattern_len)
                    .map(|_| {
                        let delta = (rng.next() % jitter_span) as i16 - jitter;
                        (base + delta).clamp(0, 255) as u8
                    })
                    .collect()
            })
            .collect()
    } else {
        Vec::new()
    };

    let mut out = Vec::with_capacity(len);
    // 0 = BG (clean white), 1 = FG (ink stretch), 2 = BG_NOISE (sensor
    // noise burst — sits inside BG logically, only transitions to BG),
    // 3 = PATTERN (emitting a glyph from the pattern library).
    let mut state = 0u8;
    let mut fg_base: i16 = 0;
    let mut pat_idx: usize = 0;
    let mut pat_pos: u32 = 0;

    while out.len() < len {
        match state {
            0 => {
                // Clean BG: emit 255, optionally start an FG stretch,
                // a pattern glyph, or a noise burst.
                out.push(255);
                if (rng.next() as f64 / u) < params.p_bg_to_fg {
                    if !patterns.is_empty() && (rng.next() as f64 / u) < params.pattern_frac {
                        // Emit a pattern "glyph."
                        state = 3;
                        pat_idx = (rng.next() % params.n_patterns) as usize;
                        pat_pos = 0;
                    } else {
                        state = 1;
                        fg_base = fg_lo + (rng.next() % fg_span) as i16;
                    }
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
            2 => {
                // BG_NOISE: emit a uniform-random byte in 0..=254 (never
                // 255, so the noise byte can't pretend to be white). The
                // burst ends with probability `bg_burst_end_p` per
                // pixel — geometric distribution of burst lengths.
                out.push((rng.next() % 255) as u8);
                if (rng.next() as f64 / u) < params.bg_burst_end_p {
                    state = 0;
                }
            }
            _ => {
                // PATTERN: emit next byte of the chosen pattern glyph.
                out.push(patterns[pat_idx][pat_pos as usize]);
                pat_pos += 1;
                if pat_pos >= params.pattern_len {
                    state = 0;
                }
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Continuous-tone photo generator — random walk model.
//
// Real film scans (Apollo Hasselblad, consumer photography) produce
// smooth gradients where neighboring pixels differ by only a few values.
// This drives LZW to build short dictionary entries (value_len ≈ 2-8)
// that recur frequently, giving high short_copy_frac (65-85%) and low
// literal_frac (14-52%). The walk model directly produces this
// correlation by emitting each pixel as prev ± small delta.
// ---------------------------------------------------------------------------

struct PhotoParams {
    /// Max step per pixel: each pixel = prev + uniform(-delta, +delta).
    walk_delta: u8,
    /// Probability of jumping to a random value (edge/region boundary).
    edge_p: f64,
    /// Number of channels (1 = grayscale, 3 = RGB interleaved).
    channels: u8,
    /// Probability per WALK pixel of entering FLAT mode.
    flat_p: f64,
    /// Probability per FLAT pixel of returning to WALK mode.
    flat_end_p: f64,
    /// FLAT mode holds a center value drawn from [flat_val_lo, flat_val_hi].
    flat_val_lo: u8,
    flat_val_hi: u8,
    /// Per-pixel noise in FLAT mode: emit center ± uniform(-noise, +noise).
    /// Simulates film grain on dark backgrounds. Breaks KwKwK runs and
    /// reduces compression-ratio overshoot from pure-constant flat regions.
    flat_noise: u8,
}

fn generate_photo(params: &PhotoParams, len: usize, seed: u32) -> Vec<u8> {
    let mut rng = Rng::new(seed);
    let u = u32::MAX as f64;
    let delta = params.walk_delta as i16;
    let delta_span = (2 * delta + 1).max(1) as u32;
    let channels = params.channels.max(1) as usize;
    let flat_val_span = (params.flat_val_hi as u32)
        .saturating_sub(params.flat_val_lo as u32)
        + 1;
    let flat_noise = params.flat_noise as i16;
    let noise_span = (2 * flat_noise + 1).max(1) as u32;

    let mut out = Vec::with_capacity(len);
    let mut val = [128i16; 3];
    let mut flat = [false; 3];
    let mut flat_hold = [0i16; 3]; // center value during flat mode

    while out.len() < len {
        for ch in 0..channels {
            if out.len() >= len {
                break;
            }
            if flat[ch] {
                // FLAT: emit center ± noise, maybe exit.
                let v = if flat_noise > 0 {
                    let n = (rng.next() % noise_span) as i16 - flat_noise;
                    (flat_hold[ch] + n).clamp(0, 255)
                } else {
                    flat_hold[ch]
                };
                out.push(v as u8);
                if (rng.next() as f64 / u) < params.flat_end_p {
                    flat[ch] = false;
                    val[ch] = flat_hold[ch]; // resume walk from center
                }
            } else {
                // WALK: edge jump, step, maybe enter flat.
                if (rng.next() as f64 / u) < params.edge_p {
                    val[ch] = (rng.next() % 256) as i16;
                }
                let step = (rng.next() % delta_span) as i16 - delta;
                val[ch] = (val[ch] + step).clamp(0, 255);
                out.push(val[ch] as u8);
                if (rng.next() as f64 / u) < params.flat_p {
                    flat[ch] = true;
                    flat_hold[ch] =
                        (params.flat_val_lo as u32 + rng.next() % flat_val_span) as i16;
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
/// Uses row-repeat mode: 4 template rows (different text lines) tiled
/// with ±12 ink noise. The scanline-aligned repetition fills the LZW
/// table efficiently — width_12 goes from 0.4% (random placement) to
/// 35% (matching real email within 1.4%). The 4 distinct templates
/// prevent inter-row long-copy dominance while maintaining enough
/// periodicity for table saturation.
///
/// Code-level fit (vs RVL-CDIP email n=50):
///   width_12:    0.354 → 0.349  (1.4%)    was 0.004 (99% off)
///   literal:     0.601 → 0.486  (19%)     was 0.522 (13% off)
///   entropy:     0.394 → 0.478  (21%)     tradeoff for width_12 fix
///   lzw_ratio:   22.1  → 17.3   (22%)     tradeoff for width_12 fix
const SCANNED_EMAIL: GenParams = GenParams {
    p_bg_to_fg: 0.007000,
    p_fg_to_bg: 0.25,
    bg_burst_p: 0.000500,
    bg_burst_end_p: 1.0,
    fg_lo: 0,
    fg_hi: 150,
    fg_jitter: 5,
    n_patterns: 40,
    pattern_len: 4,
    pattern_frac: 0.3,
    row_width: 512,
    n_template_rows: 4,
    row_noise: 12,
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
    n_patterns: 0,
    pattern_len: 0,
    pattern_frac: 0.0,
    row_width: 0,
    n_template_rows: 0,
    row_noise: 0,
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
    n_patterns: 0,
    pattern_len: 0,
    pattern_frac: 0.0,
    row_width: 0,
    n_template_rows: 0,
    row_noise: 0,
};

/// Old monochrome document scan — e.g. typewritten court opinions, typed
/// briefs, mimeographed forms. Fitted against real TIFF-LZW strips from
/// Brown v. Board of Education (SCOTUS opinion + NAACP appendix, 79
/// pages, Georgetown Law Library / IA scans, Public Domain).
///
/// These scans have higher entropy than typical office documents
/// (H≈2.1 vs ≈0.4) due to scanner noise in gray margins and ink bleed
/// on aged paper. The LZW table saturates fast (width_12 ≈ 46%) with
/// moderate copy fractions (long_copy ≈ 20%). KwKwK is negligible
/// (≈1.7%), making chunked's 8-byte suffix lookup the dominant
/// optimization — streaming's KwKwK fast path barely fires.
///
/// The pattern library (40 patterns of 6 bytes each, injected 50% of
/// the time) simulates recurring glyph pixel blocks that real text
/// produces — every copy of the letter "e" is the same 6-pixel
/// footprint. This drives `literal_frac` from 54% down to 35%
/// (matching real data within 1%) and `lzw_ratio` from 3.5:1 to
/// 4.9:1 (matching real 5.0:1 within 2%).
///
/// Byte-level fit (medians over 79 real pages):
///   entropy:    2.113 → 2.201  (4.1%)
///   repeat:     0.784 → 0.784  (0.0%)
///   rl_mean:    4.62  → 4.62   (0.1%)
///   lzw_ratio:  4.98  → 4.91   (1.3%)
///
/// Code-level fit:
///   literal:    0.348 → 0.351  (0.9%)
///   short_copy: 0.443 → 0.441  (0.4%)
///   long_copy:  0.206 → 0.208  (1.0%)
///   width_12:   0.463 → 0.440  (4.9%)
const SCANNED_MONOCHROME_OLD: GenParams = GenParams {
    p_bg_to_fg: 0.050000,
    p_fg_to_bg: 0.25,
    bg_burst_p: 0.0,
    bg_burst_end_p: 0.2,
    fg_lo: 0,
    fg_hi: 200,
    fg_jitter: 3,
    n_patterns: 40,
    pattern_len: 6,
    pattern_frac: 0.5,
    row_width: 0,
    n_template_rows: 0,
    row_noise: 0,
};

// ---------------------------------------------------------------------------
// Fitted photo archetypes — NASA Apollo Hasselblad film scans.
//
// Targets are medians over 3 Apollo mission photographs (AS11-40-5937
// Tranquility Base, as08-14-2383 Earthrise, as17-148-22742 Full Earth),
// profiled as 8-bit grayscale TIFF-LZW. All NASA public domain.
//
// The photo generator uses a random walk with flat-hold regions:
// WALK mode produces smooth gradients (neighboring pixels ± small delta),
// FLAT mode holds a near-constant value with film grain noise (center ± N).
// The alternation between textured and flat regions reproduces the
// short_copy-dominated LZW code profile of real continuous-tone photos.
// ---------------------------------------------------------------------------

/// Continuous-tone grayscale photo — e.g. Apollo "Full Earth" film scan.
/// Target: H≈3.0, ratio≈2.4, literal≈0.15, short_copy≈0.85, w12≈0.52.
/// Models a continuous-tone grayscale photograph with smooth gradients
/// and some flat dark regions (shadows, space). Represents the typical
/// photographic TIFF-LZW workload where nearly all codes are short
/// dictionary copies (value_len 2-8) and KwKwK is negligible.
///
/// Code-level fit (vs as17-148-22742 grayscale):
///   entropy:     3.01 → 3.09  (2.7%)
///   lzw_ratio:   2.44 → 2.32  (4.9%)
///   literal:     0.150 → 0.142 (5.3%)
///   short_copy:  0.850 → 0.857 (0.8%)
///   width_12:    0.523 → 0.507 (3.1%)
///   kwkwk:       0.002 → 0.002 (0%)
const PHOTO_GRAY: PhotoParams = PhotoParams {
    walk_delta: 6,
    edge_p: 0.02,
    channels: 1,
    flat_p: 0.05,
    flat_end_p: 0.005,
    flat_val_lo: 0,
    flat_val_hi: 3,
    flat_noise: 3,
};

/// Vivid color photo/painting — e.g. a museum-quality art scan or
/// color landscape photograph. Fitted against Cleveland Museum of Art
/// CC0 painting scan ("July" by Otto Bacher, 3155×5000 RGB TIFF).
///
/// This is LZW's worst case: continuous-tone color with near-maximum
/// entropy (H≈7.2) produces ratio ≈ 0.91 — LZW *expands* the data.
/// 73% of codes are literals (the dictionary almost never helps), and
/// the remaining 27% are short copies. This is what happens when
/// someone saves a photograph as TIFF-LZW instead of JPEG.
///
/// The walk uses delta=1 (minimal per-pixel step — tight local
/// correlation like real brushstrokes) with edge_p=0.03 and sparse
/// flat regions (0-50 center, noise ±6) that simulate uniform-tone
/// areas (sky, background) in paintings.
///
/// Code-level fit (vs CMA "July" RGB):
///   entropy:     7.17 → 7.21  (0.6%)
///   lzw_ratio:   0.91 → 0.91  (0%)
///   literal:     0.729 → 0.726 (0.4%)
///   short_copy:  0.270 → 0.274 (1.5%)
///   width_12:    0.522 → 0.521 (0.2%)
const PHOTO_COLOR: PhotoParams = PhotoParams {
    walk_delta: 1,
    edge_p: 0.03,
    channels: 3,
    flat_p: 0.03,
    flat_end_p: 0.10,
    flat_val_lo: 0,
    flat_val_hi: 50,
    flat_noise: 6,
};

// ---------------------------------------------------------------------------
// Flat-UI GIF generator — palette-indexed block structure.
// ---------------------------------------------------------------------------

/// Generate a flat-UI-style byte stream: palette-indexed values (0-63),
/// rectangular blocks of solid color with sharp edges. Simulates
/// screenshots, UI mockups, and flat-design graphics — the dominant
/// real-world GIF workload. Produces high compression ratios (~15-30×),
/// mostly long copies from repeated block rows, and some KwKwK from
/// solid regions.
fn generate_flat_ui(len: usize, seed: u32) -> Vec<u8> {
    let mut rng = Rng::new(seed);
    let width = 640; // typical UI screenshot width
    let palette_size = 32u32;
    let mut out = Vec::with_capacity(len);

    // Generate a "screen" of rectangular color blocks, repeated row by row.
    // Each block is a solid color spanning several columns and rows.
    let mut row = vec![0u8; width];
    let mut col = 0;
    while col < width {
        let color = (rng.next() % palette_size) as u8;
        let block_w = 20 + (rng.next() % 120) as usize; // 20-140px wide
        let end = (col + block_w).min(width);
        for c in col..end {
            row[c] = color;
        }
        col = end;
    }

    // Tile the row, occasionally changing some blocks (new "widget")
    let mut rows_until_change = 10 + (rng.next() % 40) as usize;
    while out.len() < len {
        for &b in row.iter() {
            if out.len() >= len {
                break;
            }
            out.push(b);
        }
        rows_until_change -= 1;
        if rows_until_change == 0 {
            // Repaint some blocks
            let n_changes = 1 + (rng.next() % 4) as usize;
            for _ in 0..n_changes {
                let start = (rng.next() % width as u32) as usize;
                let bw = 20 + (rng.next() % 120) as usize;
                let color = (rng.next() % palette_size) as u8;
                let end = (start + bw).min(width);
                for c in start..end {
                    row[c] = color;
                }
            }
            rows_until_change = 10 + (rng.next() % 40) as usize;
        }
    }
    out
}

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
        make_workload(
            "scanned-mono-old",
            &generate(&SCANNED_MONOCHROME_OLD, size, seed),
            BitOrder::Msb,
            true,
        ),
        make_workload("solid-kwkwk", &vec![42u8; size], BitOrder::Msb, true),
        // Photo workloads — continuous-tone imagery, high short_copy,
        // negligible KwKwK. Grayscale and RGB to cover both TIFF modes.
        make_workload(
            "photo-gray",
            &generate_photo(&PHOTO_GRAY, size, seed),
            BitOrder::Msb,
            true,
        ),
        make_workload(
            "photo-color",
            &generate_photo(&PHOTO_COLOR, size, seed),
            BitOrder::Msb,
            true,
        ),
        // LSB — GIF configuration.
        make_workload(
            "scanned-email",
            &generate(&SCANNED_EMAIL, size, seed),
            BitOrder::Lsb,
            false,
        ),
        // Flat-UI GIF: palette-indexed (0-63), large solid-color blocks
        // with sharp edges. Simulates screenshots, UI mockups, flat-design
        // graphics. High compression ratio, mostly long copies from
        // repeated UI elements. This is the workload where the KwKwK
        // optimization's `last_decoded` bookkeeping was observed to
        // regress the COPY hot path by 5-7% in earlier iterations.
        make_workload(
            "flat-ui",
            &generate_flat_ui(size, seed),
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
