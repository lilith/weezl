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
    """Four-state Markov generator (BG, FG, BG_NOISE, PATTERN).

    Mirrors the Rust `generate` in `benches/strategy_compare.rs`
    byte-for-byte: same xorshift32 stream, same branch order. Any
    change here must be made there too, and vice versa.

    Params:
      p_bg_to_fg      prob of entering FG from clean BG, per pixel
      p_fg_to_bg      prob of returning to BG from FG, per pixel
      bg_burst_p      prob of starting a noise burst from clean BG,
                      per pixel. Replaces the old `bg_noise_p`; on
                      backward-compat targets we set bg_burst_end_p=1.0
                      so the generator reproduces i.i.d. Bernoulli noise.
      bg_burst_end_p  prob of ending a noise burst, per BG_NOISE pixel.
                      With bg_burst_end_p=1.0 every burst is length one
                      (Bernoulli). With bg_burst_end_p=0.1 the burst
                      length is geometrically distributed with mean 10.
      fg_lo, fg_hi    FG stretch picks a base byte uniformly from
                      [fg_lo, fg_hi]
      fg_jitter       each FG pixel emits base + uniform([-jitter, jitter])
      n_patterns      number of fixed byte patterns in the glyph library
                      (default 0 = disabled, backward compat)
      pattern_len     length of each pattern in bytes (default 16)
      pattern_frac    when entering FG, probability of emitting a pattern
                      instead of jitter bytes (default 0.0)
      row_width       if > 0, pattern selection is position-dependent:
                      pattern_idx = f(column) instead of rng(). The same
                      column gets the same pattern on every row, simulating
                      the periodic structure of text lines where each copy
                      of "e" occupies the same column range. This makes
                      the LZW encoder see repeated prefix→pattern
                      transitions at the same stream offsets, filling
                      the dictionary much faster than random placement.
                      0 = random placement (backward compat).
    """
    p_bf = params["p_bg_to_fg"]
    p_fb = params["p_fg_to_bg"]
    # Backward compat: allow callers to pass the old `bg_noise_p` key
    # with an implicit bg_burst_end_p=1.0 (one-pixel bursts = Bernoulli).
    if "bg_burst_p" in params:
        bg_burst_p = params["bg_burst_p"]
        bg_burst_end_p = params.get("bg_burst_end_p", 1.0)
    else:
        bg_burst_p = params.get("bg_noise_p", 0.0)
        bg_burst_end_p = 1.0
    fg_lo = params["fg_lo"]
    fg_hi = params["fg_hi"]
    fg_jitter = params["fg_jitter"]
    n_patterns = params.get("n_patterns", 0)
    pattern_len = params.get("pattern_len", 16)
    pattern_frac = params.get("pattern_frac", 0.0)
    row_width = params.get("row_width", 0)

    rng = Rng(seed)

    # Generate pattern library from the PRNG stream. Each pattern is a
    # fixed byte sequence that simulates a character glyph — the same
    # block of pixels that repeats every time that "letter" appears.
    # Only consumes PRNG values when n_patterns > 0, so the stream is
    # unchanged for backward-compatible parameter sets.
    patterns = []
    if n_patterns > 0 and pattern_frac > 0.0:
        span_fg = max(1, fg_hi - fg_lo + 1)
        for _ in range(n_patterns):
            base = fg_lo + (rng.next() % span_fg)
            pat = bytearray(pattern_len)
            for j in range(pattern_len):
                jspan = 2 * fg_jitter + 1
                delta = (rng.next() % jspan) - fg_jitter
                v = base + delta
                if v < 0:
                    v = 0
                elif v > 255:
                    v = 255
                pat[j] = v
            patterns.append(bytes(pat))

    # Row-repeat mode: generate one template row with the Markov model,
    # then tile it for every row with per-pixel noise. This creates the
    # scanline-aligned periodic structure that real text has — every row
    # through a text line has ink at the same columns, which makes the
    # LZW encoder see repeated prefix→pattern transitions at the same
    # stream offsets, filling the dictionary far faster than random FG
    # placement. row_noise controls the per-pixel jitter on the
    # template (0 = exact repeat → massive KwKwK; 1-3 = realistic
    # noise → fast table fill without KwKwK inflation).
    row_noise = params.get("row_noise", 0)
    white_noise = params.get("white_noise", 0)

    n_template_rows = max(1, params.get("n_template_rows", 1))

    if row_width > 0:
        # Generate template rows using the standard Markov model.
        # Multiple templates simulate different text lines — adjacent
        # rows have different content (breaking inter-row long copies
        # and increasing literal_frac) while every Nth row repeats
        # (maintaining periodic structure for LZW table fill).
        inner_params = dict(params, row_width=0)
        templates = []
        for t in range(n_template_rows):
            # Each template uses a different seed offset for variety
            tmpl = bytearray(gen_markov(inner_params, row_width, seed + t))
            templates.append(tmpl)
        noise_span = 2 * row_noise + 1
        wnoise_span = 2 * white_noise + 1
        out = bytearray(length)
        for i in range(length):
            row = i // row_width
            template = templates[row % n_template_rows]
            v = template[i % row_width]
            if row_noise > 0 and v != 255:
                # Ink noise: ±row_noise on dark pixels.
                delta = (rng.next() % noise_span) - row_noise
                v = max(0, min(255, v + delta))
            elif white_noise > 0 and v == 255:
                # Sensor noise on white pixels.
                delta = (rng.next() % wnoise_span) - white_noise
                v = max(0, min(255, 255 + delta))
            out[i] = v
        return bytes(out)

    out = bytearray(length)
    # 0 = BG, 1 = FG, 2 = BG_NOISE, 3 = PATTERN
    state = 0
    fg_base = 0
    pat_idx = 0  # which pattern we're emitting
    pat_pos = 0  # position within the current pattern
    U = float(0xFFFFFFFF)
    span_fg = max(1, fg_hi - fg_lo + 1)
    jitter_span = 2 * fg_jitter + 1
    for i in range(length):
        if state == 0:
            # Clean BG. Emit 255, optionally start FG/PATTERN or BG_NOISE.
            out[i] = 255
            if (rng.next() / U) < p_bf:
                if patterns and (rng.next() / U) < pattern_frac:
                    # Emit a pattern "glyph".
                    state = 3
                    pat_idx = rng.next() % n_patterns
                    pat_pos = 0
                else:
                    state = 1
                    fg_base = fg_lo + (rng.next() % span_fg)
            elif (rng.next() / U) < bg_burst_p:
                state = 2
        elif state == 1:
            # FG: ink tone with jitter around stretch base.
            delta = (rng.next() % jitter_span) - fg_jitter
            v = fg_base + delta
            if v < 0:
                v = 0
            elif v > 255:
                v = 255
            out[i] = v
            if (rng.next() / U) < p_fb:
                state = 0
        elif state == 2:
            # BG_NOISE: geometric-length run of uniform 0..=254 bytes.
            out[i] = rng.next() % 255  # 0..254, uniform; never 255
            if (rng.next() / U) < bg_burst_end_p:
                state = 0
        else:
            # PATTERN: emit next byte of the chosen pattern.
            out[i] = patterns[pat_idx][pat_pos]
            pat_pos += 1
            if pat_pos >= pattern_len:
                state = 0
    return bytes(out)


def gen_photo(params, length, seed):
    """Random-walk photo generator with flat-hold regions.
    Mirrors Rust `generate_photo`.

    Two modes per pixel:
      WALK  — prev + uniform(-delta, +delta), producing smooth gradients
      FLAT  — hold a constant value (no step), producing long identical runs

    Real film scans alternate between textured regions (surface detail,
    gradients) and flat regions (sky, space, shadows, overexposed areas).
    The flat regions drive down entropy and up lzw_ratio; the walk
    regions provide the short_copy-heavy LZW code profile.

    Params:
      walk_delta        max step per pixel in WALK mode
      edge_p            prob of jumping to a random value (edge)
      channels          1 = grayscale, 3 = RGB (interleaved)
      flat_p            prob per pixel of entering FLAT mode from WALK
      flat_end_p        prob per pixel of leaving FLAT back to WALK
      flat_val_lo       flat region holds a center value in [lo, hi]
      flat_val_hi       (e.g. 0,5 for dark space; 250,255 for blown highlights)
      flat_noise        per-pixel noise in flat mode: emit center ± uniform(-noise, +noise).
                        Simulates film grain on dark backgrounds. Breaks KwKwK runs
                        (real film "black" is never truly constant) and reduces the
                        compression-ratio overshoot from pure-constant flat regions.
    """
    walk_delta = params["walk_delta"]
    edge_p = params["edge_p"]
    channels = max(1, params.get("channels", 1))
    flat_p = params.get("flat_p", 0.0)
    flat_end_p = params.get("flat_end_p", 0.0)
    flat_val_lo = params.get("flat_val_lo", 0)
    flat_val_hi = params.get("flat_val_hi", 5)
    flat_noise = params.get("flat_noise", 0)

    rng = Rng(seed)
    out = bytearray(length)
    U = float(0xFFFFFFFF)
    delta_span = 2 * walk_delta + 1
    flat_val_span = max(1, flat_val_hi - flat_val_lo + 1)
    noise_span = 2 * flat_noise + 1

    val = [128] * 3
    flat = [False] * 3
    flat_hold = [0] * 3  # center value during flat mode
    i = 0
    while i < length:
        for ch in range(channels):
            if i >= length:
                break
            if flat[ch]:
                # FLAT mode: emit center ± noise, maybe exit.
                if flat_noise > 0:
                    noise = (rng.next() % noise_span) - flat_noise
                    v = max(0, min(255, flat_hold[ch] + noise))
                else:
                    v = flat_hold[ch]
                out[i] = v
                if (rng.next() / U) < flat_end_p:
                    flat[ch] = False
                    val[ch] = flat_hold[ch]  # resume walk from center
            else:
                # WALK mode: edge jump, walk step, maybe enter flat.
                if (rng.next() / U) < edge_p:
                    val[ch] = rng.next() % 256
                step = (rng.next() % delta_span) - walk_delta
                val[ch] = max(0, min(255, val[ch] + step))
                out[i] = val[ch]
                if (rng.next() / U) < flat_p:
                    flat[ch] = True
                    flat_hold[ch] = flat_val_lo + (rng.next() % flat_val_span)
            i += 1
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
    parsed = _extract_strip_bytes_tiff(buf.getvalue())
    if parsed is None:
        return float("inf")
    strips, _decoded_len = parsed
    enc = sum(len(s) for s in strips)
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
# Code-level feature extractor — for fitting-to-performance, not bytes.
# ---------------------------------------------------------------------------
#
# Byte-level statistics (entropy, rl_mean, lzw_ratio) are properties of the
# DATA. Decoder performance is a function of the CODE STREAM — how many
# codes, how long each decoded value is, how often KwKwK happens, how
# deep the prefix chains go. Two TIFFs with identical byte entropy can
# decode at very different speeds if their LZW code-length distributions
# differ, so fitting to byte features doesn't guarantee a fitted
# archetype will reproduce the real corpus's performance curve.
#
# This extractor runs a minimal MSB TIFF-LZW decoder with TIFF early-
# change size switch, collects per-code feature histograms, and returns
# the feature vector. No timing involved; just counts. The output feeds
# into a linear cost model ("cycles ≈ α + β × codes + γ × long_copy_bytes
# + ...") that can be fit against real weezl benchmark data in a
# separate step, giving us a cheap proxy for decode throughput.


def _extract_strip_bytes_tiff(blob):
    """Parse a TIFF blob (possibly multi-strip) and return
    ``(strips, decoded_len)`` where ``strips`` is a list of raw LZW
    byte strings — one per TIFF strip — and ``decoded_len`` is the
    total number of decoded bytes in the image.

    Assumes the TIFF was emitted by Pillow with compression='tiff_lzw'.
    Walks the IFD to find StripOffsets + StripByteCounts + ImageWidth +
    ImageLength + BitsPerSample + SamplesPerPixel. Each LZW strip is
    an independent stream with its own clear code at the start.
    """
    if len(blob) < 8 or blob[:2] != b"II":
        return None
    ifd_off = struct.unpack_from("<I", blob, 4)[0]
    n_entries = struct.unpack_from("<H", blob, ifd_off)[0]
    img_w = img_h = bps = spp = 0
    strip_offsets = []
    strip_byte_counts = []

    def read_array(type_, count, value_bytes):
        """Return a list of `count` ints from an IFD value field.
        For total <= 4 bytes they're inline; otherwise the value is an
        offset to the array."""
        elem = {3: 2, 4: 4}.get(type_)
        if elem is None:
            return []
        total = elem * count
        if total <= 4:
            sl = value_bytes
        else:
            off = struct.unpack_from("<I", value_bytes, 0)[0]
            sl = blob[off : off + total]
        out = []
        for j in range(count):
            if elem == 2:
                out.append(struct.unpack_from("<H", sl, j * 2)[0])
            else:
                out.append(struct.unpack_from("<I", sl, j * 4)[0])
        return out

    for i in range(n_entries):
        e = ifd_off + 2 + i * 12
        tag, type_, count = struct.unpack_from("<HHI", blob, e)
        value = blob[e + 8 : e + 12]

        if tag == 0x100:  # ImageWidth
            img_w = read_array(type_, 1, value)[0]
        elif tag == 0x101:  # ImageLength
            img_h = read_array(type_, 1, value)[0]
        elif tag == 0x102:  # BitsPerSample
            bps = read_array(type_, 1, value)[0] if count == 1 else 8
        elif tag == 0x115:  # SamplesPerPixel
            spp = read_array(type_, 1, value)[0]
        elif tag == 0x111:  # StripOffsets
            strip_offsets = read_array(type_, count, value)
        elif tag == 0x117:  # StripByteCounts
            strip_byte_counts = read_array(type_, count, value)

    if not strip_offsets or len(strip_offsets) != len(strip_byte_counts):
        return None
    spp = spp or 1
    bps = bps or 8
    decoded_len = img_w * img_h * spp * (bps // 8)
    strips = [
        blob[off : off + n]
        for off, n in zip(strip_offsets, strip_byte_counts)
    ]
    return strips, decoded_len


class _LzwProfiler:
    """Minimal MSB TIFF-LZW decoder that collects code-level features.

    Doesn't produce decoded output beyond what's needed to advance the
    decode table — we just track string lengths and derived metadata.
    TIFF early-change semantics (width bumps one code sooner than
    standard LZW) match what weezl's tiff_size_switch mode does.
    """

    CLEAR_CODE = 256
    END_CODE = 257

    def __init__(self):
        self.codes = 0
        self.literal = 0
        self.short_copy = 0  # value_len <= 8
        self.long_copy = 0  # value_len > 8
        self.kwkwk = 0
        self.clears = 0
        self.value_len_hist = Counter()
        self.width_hist = Counter()  # codes emitted at each width
        self.max_chain_len = 0  # deepest derived-code chain encountered

    def feed(self, encoded):
        """Decode `encoded` as TIFF-LZW MSB. Collect features; discard bytes."""
        # lm1s[code] = length - 1 of the string represented by `code`
        lm1s = [0] * 4096
        for i in range(256):
            lm1s[i] = 0  # each literal has length 1

        # Bit reader: MSB-first, accumulate from left.
        bit_buf = 0
        n_bits = 0
        width = 9  # start at 9 bits for min_code_size=8
        save_code = 258  # next code to assign
        # TIFF early-change: bump width when save_code reaches (1 << width) - 1
        # (one code sooner than standard LZW's (1 << width)).
        prev_code = -1  # -1 = sentinel "no prev code"
        inp_pos = 0
        inp_len = len(encoded)

        def refill_to(target):
            nonlocal bit_buf, n_bits, inp_pos
            while n_bits < target and inp_pos < inp_len:
                bit_buf = (bit_buf << 8) | encoded[inp_pos]
                inp_pos += 1
                n_bits += 8

        while True:
            refill_to(width)
            if n_bits < width:
                break
            # Extract MSB-first: the top `width` bits of bit_buf.
            shift = n_bits - width
            code = (bit_buf >> shift) & ((1 << width) - 1)
            bit_buf &= (1 << shift) - 1
            n_bits -= width
            self.codes += 1
            self.width_hist[width] += 1

            if code == self.CLEAR_CODE:
                self.clears += 1
                save_code = 258
                width = 9
                prev_code = -1
                continue
            if code == self.END_CODE:
                break

            if code < self.CLEAR_CODE:
                # Literal.
                self.literal += 1
                self.value_len_hist[1] += 1
                value_len = 1
            elif code == save_code:
                # KwKwK: string is prev's value + prev's first byte.
                self.kwkwk += 1
                prev_len = lm1s[prev_code] + 1 if prev_code >= 0 else 0
                value_len = prev_len + 1
                self.value_len_hist[value_len] += 1
                if value_len > 8:
                    self.long_copy += 1
                else:
                    self.short_copy += 1
            elif code < save_code:
                # Regular copy — look up length.
                value_len = lm1s[code] + 1
                self.value_len_hist[value_len] += 1
                if value_len > 8:
                    self.long_copy += 1
                else:
                    self.short_copy += 1
            else:
                # Invalid (ahead of save_code + 1). Stop; count as end.
                break

            # Derive a new entry based on prev_code.
            if prev_code >= 0 and save_code < 4096:
                new_len = lm1s[prev_code] + 2  # +1 for link, +1 for base count
                # Actually: new entry's length = prev's length + 1 (one byte appended).
                new_len = (lm1s[prev_code] + 1) + 1
                lm1s[save_code] = new_len - 1
                if new_len > self.max_chain_len:
                    self.max_chain_len = new_len
                save_code += 1
                # TIFF early-change width bump: trigger when save_code
                # reaches (1 << width) - 1, one code sooner than standard.
                if save_code >= (1 << width) - 1 and width < 12:
                    width += 1

            prev_code = code

    def report(self, decoded_bytes):
        vl = self.value_len_hist
        total_vl_bytes = sum(k * v for k, v in vl.items())
        mean_vl = total_vl_bytes / max(sum(vl.values()), 1)
        long_copy_bytes = sum(k * v for k, v in vl.items() if k > 8)
        return {
            "total_codes": self.codes,
            "codes_per_decoded_byte": self.codes / max(decoded_bytes, 1),
            "literal_frac": self.literal / max(self.codes, 1),
            "short_copy_frac": self.short_copy / max(self.codes, 1),
            "long_copy_frac": self.long_copy / max(self.codes, 1),
            "kwkwk_frac": self.kwkwk / max(self.codes, 1),
            "clears": self.clears,
            "mean_value_len": mean_vl,
            "long_copy_byte_frac": long_copy_bytes / max(decoded_bytes, 1),
            "max_chain_len": self.max_chain_len,
            "width_9": self.width_hist.get(9, 0) / max(self.codes, 1),
            "width_10": self.width_hist.get(10, 0) / max(self.codes, 1),
            "width_11": self.width_hist.get(11, 0) / max(self.codes, 1),
            "width_12": self.width_hist.get(12, 0) / max(self.codes, 1),
        }


def profile_lzw(raw_bytes, width, height):
    """Encode raw bytes as TIFF-LZW via Pillow, then decode with our
    profiler to extract code-level features.

    Returns a dict of feature → value. The features are independent of
    any particular decoder strategy — they describe the LZW code stream
    itself, which is what drives decoder performance.
    """
    img = Image.frombytes("L", (width, height), raw_bytes)
    buf = io.BytesIO()
    img.save(buf, format="TIFF", compression="tiff_lzw")
    return _profile_tiff_blob(buf.getvalue())


def _profile_tiff_blob(blob):
    """Profile a TIFF-LZW blob. Each strip is a separate LZW stream,
    so we run the profiler on each and aggregate counts."""
    parsed = _extract_strip_bytes_tiff(blob)
    if parsed is None:
        return None
    strips, decoded_len = parsed
    prof = _LzwProfiler()
    for strip in strips:
        prof.feed(strip)
    return prof.report(decoded_len)


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
#
# `bg_burst_end_p = 1.0` = Bernoulli-equivalent (each noisy pixel is its own
# length-1 burst). Lower values produce spatially-clustered noise bursts
# whose length is geometrically distributed with mean 1/bg_burst_end_p.
# Used by the blank archetype to push entropy up without breaking runs.
STARTING_POINTS = {
    "email": {
        "p_bg_to_fg": 0.0070,
        "p_fg_to_bg": 0.25,
        "bg_burst_p": 0.0004,
        "bg_burst_end_p": 1.0,
        "fg_lo": 0,
        "fg_hi": 150,
        "fg_jitter": 5,
    },
    "blank": {
        # Start from a bursty baseline. Rough analytic estimate: to reach
        # entropy 0.123 from mode_frac 0.992 we need ~0.8% non-255 bytes,
        # and mean-10 bursts deliver them at ~1.1 breaks/byte instead of
        # Bernoulli's 2 breaks/byte — enough headroom to hit rl_mean 113.
        "p_bg_to_fg": 0.0017,
        "p_fg_to_bg": 0.25,
        "bg_burst_p": 0.0001,
        "bg_burst_end_p": 0.10,
        "fg_lo": 0,
        "fg_hi": 120,
        "fg_jitter": 10,
    },
    "dense": {
        "p_bg_to_fg": 0.025,
        "p_fg_to_bg": 0.25,
        "bg_burst_p": 0.0,
        "bg_burst_end_p": 1.0,
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

    for _ in range(5):
        improved = False
        for scale in (0.85, 0.93, 1.08, 1.17):
            p = dict(best_params)
            p["p_bg_to_fg"] = max(1e-5, best_params["p_bg_to_fg"] * scale)
            improved |= trial(p)
        for scale in (0.80, 0.90, 1.10, 1.25):
            p = dict(best_params)
            p["p_fg_to_bg"] = min(1.0, max(0.05, best_params["p_fg_to_bg"] * scale))
            improved |= trial(p)
        # Noise-burst start rate — how often a burst kicks off.
        for scale in (0.5, 0.75, 1.25, 2.0):
            p = dict(best_params)
            p["bg_burst_p"] = max(0.0, best_params["bg_burst_p"] * scale)
            improved |= trial(p)
        # Burst-end probability — controls mean burst length.
        # Only meaningful when bg_burst_p > 0 AND bg_burst_end_p < 1.0
        # (otherwise there are no multi-pixel bursts to tune).
        if best_params.get("bg_burst_end_p", 1.0) < 1.0 and best_params.get("bg_burst_p", 0.0) > 0.0:
            for scale in (0.5, 0.75, 1.33, 2.0):
                p = dict(best_params)
                p["bg_burst_end_p"] = min(
                    1.0, max(0.01, best_params["bg_burst_end_p"] * scale)
                )
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
            f"bg_burst_p={p['bg_burst_p']:.6f} bg_burst_end_p={p['bg_burst_end_p']:.3f} "
            f"fg_lo={p['fg_lo']} fg_hi={p['fg_hi']} fg_jitter={p['fg_jitter']}"
        )


if __name__ == "__main__":
    main()
