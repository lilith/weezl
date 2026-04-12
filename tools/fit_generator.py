#!/usr/bin/env python3
"""Fit a two-state Markov generator against RVL-CDIP median targets.

The Rust generator in `benches/strategy_compare.rs` must reproduce the
byte statistics of real scanned documents (one-byte-dominated, long
white runs, small amount of ink). This script searches the parameter
space of a two-state Markov process plus a noise knob until the
generated data — run through an in-memory TIFF-LZW re-encode by Pillow —
lands within tolerance of the target metrics defined in
`docs/rvl-cdip-test2-characteristics.csv`.

Process:
  1. BG state emits 255 with small probability (~0.1%) of drawing a
     uniform random byte from [0, 254] (simulates scanner sensor noise,
     which in real data fills all 256 histogram bins even on blank pages).
  2. On each BG pixel we roll for a transition to FG with prob p_bg_to_fg.
  3. FG state picks a base dark-ink byte once per stretch, then emits
     fg_base ± jitter bytes that are correlated (jitter << base), so LZW
     can find short patterns in them. Transition back with p_fg_to_bg.

This Python implementation mirrors the Rust generator in
`benches/strategy_compare.rs` byte-for-byte — they share the same
xorshift32 PRNG sequence and the same branching structure, so the Rust
output passes the same metric targets as the Python fit predicts.

Usage:
    tools/fit_generator.py                 # fit all archetypes
    tools/fit_generator.py --archetype email
    tools/fit_generator.py --archetype blank
    tools/fit_generator.py --archetype dense
"""
import argparse
import io
import math
import struct
import sys
from collections import Counter

try:
    from PIL import Image
except ImportError:
    print("need Pillow: pip install Pillow", file=sys.stderr)
    sys.exit(2)


# ---------------------------------------------------------------------------
# xorshift32 + generator (must match benches/strategy_compare.rs byte-for-byte)
# ---------------------------------------------------------------------------

MASK32 = 0xFFFFFFFF


class Rng:
    def __init__(self, seed):
        self.s = (seed | 1) & MASK32

    def next(self):
        x = self.s
        x ^= (x << 13) & MASK32
        x ^= (x >> 17) & MASK32
        x ^= (x << 5) & MASK32
        self.s = x & MASK32
        return self.s


def gen_markov(params, length, seed):
    """Two-state Markov generator.

    Params:
      p_bg_to_fg   prob of entering FG each BG pixel
      p_fg_to_bg   prob of returning to BG each FG pixel
      bg_noise_p   prob that a BG pixel is a uniform-random byte in [0, 254]
      fg_lo, fg_hi FG stretch picks a base byte uniformly from [fg_lo, fg_hi]
      fg_jitter    each FG pixel emits base + uniform([-jitter, jitter])
    """
    p_bf = params["p_bg_to_fg"]
    p_fb = params["p_fg_to_bg"]
    bg_noise_p = params["bg_noise_p"]
    fg_lo = params["fg_lo"]
    fg_hi = params["fg_hi"]
    fg_jitter = params["fg_jitter"]

    rng = Rng(seed)
    out = bytearray(length)
    state = 0
    fg_base = 0
    U = float(0xFFFFFFFF)
    span_fg = max(1, fg_hi - fg_lo + 1)
    jitter_span = 2 * fg_jitter + 1
    for i in range(length):
        if state == 0:
            if (rng.next() / U) < bg_noise_p:
                out[i] = rng.next() % 255  # 0..254, uniform; never 255
            else:
                out[i] = 255
            if (rng.next() / U) < p_bf:
                state = 1
                fg_base = fg_lo + (rng.next() % span_fg)
        else:
            delta = (rng.next() % jitter_span) - fg_jitter
            v = fg_base + delta
            if v < 0:
                v = 0
            elif v > 255:
                v = 255
            out[i] = v
            if (rng.next() / U) < p_fb:
                state = 0
    return bytes(out)


# ---------------------------------------------------------------------------
# Metrics (mirror analyze_scanned_pages.py)
# ---------------------------------------------------------------------------


def entropy_bits(hist, total):
    if total == 0:
        return 0.0
    h = 0.0
    for c in hist.values():
        if c > 0:
            p = c / total
            h -= p * math.log2(p)
    return h


def run_stats(pixels):
    if not pixels:
        return {"count": 0, "mean": 0.0, "long_frac": 0.0}
    runs = []
    cur = pixels[0]
    rl = 1
    for b in pixels[1:]:
        if b == cur:
            rl += 1
        else:
            runs.append(rl)
            cur = b
            rl = 1
    runs.append(rl)
    return {
        "count": len(runs),
        "mean": sum(runs) / len(runs),
        "long_frac": sum(r for r in runs if r >= 8) / len(pixels),
    }


def repeat_fraction(pixels):
    if len(pixels) < 2:
        return 0.0
    m = sum(1 for i in range(1, len(pixels)) if pixels[i] == pixels[i - 1])
    return m / (len(pixels) - 1)


def lzw_ratio(raw_bytes, width, height):
    img = Image.frombytes("L", (width, height), raw_bytes)
    buf = io.BytesIO()
    img.save(buf, format="TIFF", compression="tiff_lzw")
    blob = buf.getvalue()
    if len(blob) < 8 or blob[:2] != b"II":
        return float("inf")
    ifd_off = struct.unpack_from("<I", blob, 4)[0]
    n_entries = struct.unpack_from("<H", blob, ifd_off)[0]
    strip_byte_counts = []
    for i in range(n_entries):
        e = ifd_off + 2 + i * 12
        tag, type_, count = struct.unpack_from("<HHI", blob, e)
        value = blob[e + 8 : e + 12]
        if tag == 0x117:
            elem = {3: 2, 4: 4}.get(type_)
            total = elem * count
            sl = value if total <= 4 else blob[
                struct.unpack_from("<I", value, 0)[0] :
                struct.unpack_from("<I", value, 0)[0] + total
            ]
            for j in range(count):
                if elem == 2:
                    v = struct.unpack_from("<H", sl, j * 2)[0]
                else:
                    v = struct.unpack_from("<I", sl, j * 4)[0]
                strip_byte_counts.append(v)
    enc = sum(strip_byte_counts)
    return len(raw_bytes) / enc if enc else float("inf")


def measure(raw, width, height):
    hist = Counter(raw)
    return {
        "entropy": entropy_bits(hist, len(raw)),
        "repeat_frac": repeat_fraction(raw),
        **{k: v for k, v in run_stats(raw).items() if k in ("mean", "long_frac")},
        "mode_frac": hist.most_common(1)[0][1] / len(raw) if raw else 0.0,
        "lzw_ratio": lzw_ratio(raw, width, height),
    }


def measure_named(raw, width, height):
    rs = run_stats(raw)
    hist = Counter(raw)
    return {
        "entropy": entropy_bits(hist, len(raw)),
        "repeat_frac": repeat_fraction(raw),
        "rl_mean": rs["mean"],
        "rl_long_frac": rs["long_frac"],
        "mode_frac": hist.most_common(1)[0][1] / len(raw) if raw else 0.0,
        "lzw_ratio": lzw_ratio(raw, width, height),
    }


# ---------------------------------------------------------------------------
# Targets — from docs/rvl-cdip-test2-characteristics.csv
# ---------------------------------------------------------------------------

TARGETS = {
    # Email class median (n=50).
    "email": {
        "entropy": 0.394,
        "repeat_frac": 0.968,
        "rl_mean": 31.28,
        "rl_long_frac": 0.964,
        "mode_frac": 0.972,
        "lzw_ratio": 22.1,
    },
    # "Blank-ish form" — median of 15 files with lzw_ratio ∈ [50, 80]
    # (the low-content, high-compression tail of the form class). More
    # realistic than the tail extreme (form_011 had ratio 71.6 at the top).
    "blank": {
        "entropy": 0.123,
        "repeat_frac": 0.991,
        "rl_mean": 113.37,
        "rl_long_frac": 0.991,
        "mode_frac": 0.992,
        "lzw_ratio": 56.8,
    },
    # Densely filled low-ratio content — e.g. email_003 (dense text,
    # H≈1.1, ratio≈7.3). Matches the lower tail of both classes.
    "dense": {
        "entropy": 1.115,
        "repeat_frac": 0.894,
        "rl_mean": 9.42,
        "rl_long_frac": 0.876,
        "mode_frac": 0.912,
        "lzw_ratio": 7.3,
    },
}


# Hand-tuned starting points from iterative grid search. The refinement loop
# below trims them further but these are already within ±15% on all metrics.
STARTING_POINTS = {
    "email": {
        "p_bg_to_fg": 0.0070,
        "p_fg_to_bg": 0.25,
        "bg_noise_p": 0.0,
        "fg_lo": 0,
        "fg_hi": 150,
        "fg_jitter": 5,
    },
    "blank": {
        "p_bg_to_fg": 0.0017,
        "p_fg_to_bg": 0.25,
        "bg_noise_p": 0.0010,
        "fg_lo": 0,
        "fg_hi": 120,
        "fg_jitter": 10,
    },
    "dense": {
        "p_bg_to_fg": 0.025,
        "p_fg_to_bg": 0.25,
        "bg_noise_p": 0.0,
        "fg_lo": 0,
        "fg_hi": 150,
        "fg_jitter": 10,
    },
}


def err(measured, target):
    parts = []
    for key, tgt in target.items():
        if tgt == 0:
            d = abs(measured[key])
        else:
            d = abs(measured[key] - tgt) / abs(tgt)
        parts.append((key, d))
    return sum(d for _, d in parts), parts


def refine(name, length, width, height, seed):
    target = TARGETS[name]
    best_params = dict(STARTING_POINTS[name])
    raw = gen_markov(best_params, length, seed)
    best_m = measure_named(raw, width, height)
    best_err, _ = err(best_m, target)

    print(
        f"[{name}] start err={best_err:.3f} "
        f"H={best_m['entropy']:.3f} rep={best_m['repeat_frac']:.3f} "
        f"rl={best_m['rl_mean']:.2f} long={best_m['rl_long_frac']:.3f} "
        f"mode={best_m['mode_frac']:.3f} ratio={best_m['lzw_ratio']:.2f}",
        file=sys.stderr,
    )

    # Small local perturbations on each axis.
    def trial(p):
        nonlocal best_err, best_m, best_params
        raw = gen_markov(p, length, seed)
        m = measure_named(raw, width, height)
        e, _ = err(m, target)
        if e < best_err:
            best_err, best_m, best_params = e, m, dict(p)
            return True
        return False

    for _ in range(3):
        improved = False
        for scale in (0.85, 0.93, 1.08, 1.17):
            p = dict(best_params)
            p["p_bg_to_fg"] = max(1e-5, best_params["p_bg_to_fg"] * scale)
            improved |= trial(p)
        for scale in (0.80, 0.90, 1.10, 1.25):
            p = dict(best_params)
            p["p_fg_to_bg"] = min(1.0, max(0.05, best_params["p_fg_to_bg"] * scale))
            improved |= trial(p)
        for d in (-0.0005, -0.0002, 0.0002, 0.0005):
            p = dict(best_params)
            p["bg_noise_p"] = max(0.0, best_params["bg_noise_p"] + d)
            improved |= trial(p)
        for d in (-2, -1, 1, 2):
            p = dict(best_params)
            p["fg_jitter"] = max(0, best_params["fg_jitter"] + d)
            improved |= trial(p)
        if not improved:
            break

    print(
        f"[{name}] final err={best_err:.3f}",
        file=sys.stderr,
    )
    return best_params, best_m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archetype", default="all", choices=["all", "email", "blank", "dense"])
    ap.add_argument("--length", type=int, default=256 * 1024)
    args = ap.parse_args()

    names = ["email", "blank", "dense"] if args.archetype == "all" else [args.archetype]

    length = args.length
    side = int(round(length**0.5))
    width, height = side, length // side
    actual_len = width * height
    print(f"# Generating {width}x{height} = {actual_len} bytes", file=sys.stderr)

    results = {}
    for n in names:
        p, m = refine(n, actual_len, width, height, 0xDEADBEEF)
        results[n] = (p, m)

    # Dump a rust-ready summary with per-metric fit quality.
    print("\n# Fit summary")
    print(f"# {'archetype':12s} {'metric':15s} {'target':>10s} {'measured':>10s} {'rel_err':>10s}")
    for n, (p, m) in results.items():
        target = TARGETS[n]
        for key in ("entropy", "repeat_frac", "rl_mean", "rl_long_frac", "mode_frac", "lzw_ratio"):
            t = target[key]
            v = m[key]
            e = abs(v - t) / abs(t) if t else 0.0
            print(f"# {n:12s} {key:15s} {t:10.3f} {v:10.3f} {e*100:9.1f}%")
        print(
            f"# {n}: p_bg_to_fg={p['p_bg_to_fg']:.6f} p_fg_to_bg={p['p_fg_to_bg']:.3f} "
            f"bg_noise_p={p['bg_noise_p']:.6f} fg_lo={p['fg_lo']} fg_hi={p['fg_hi']} "
            f"fg_jitter={p['fg_jitter']}"
        )


if __name__ == "__main__":
    main()
