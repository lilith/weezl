#!/usr/bin/env python3
"""Compute LZW-relevant characteristics of scanned document TIFFs.

For each file:
  - Decode pixels (any TIFF compression) to raw 8-bit grayscale bytes.
  - Compute byte entropy (Shannon, bits/byte).
  - Compute run-length distribution: number of same-byte runs, mean/median
    run length, max run length, percentage of bytes in runs ≥ 8.
  - Compute repeat fraction: fraction of bytes equal to the previous byte.
  - Compute byte histogram → peak bin, number of distinct values, fraction
    at the mode.
  - Re-encode with TIFF-LZW (no predictor) via Pillow and report the
    compression ratio weezl would see (decoded_len / lzw_encoded_len).
  - Do the same with the horizontal-differencing predictor to see the
    effect that real TIFF writers typically use.

Run:
    tools/analyze_scanned_pages.py /path/to/tiffs > characteristics.csv
    tools/analyze_scanned_pages.py /path/to/tiffs --json > characteristics.json
"""
import argparse
import json
import math
import os
import struct
import sys
from collections import Counter
from pathlib import Path

try:
    from PIL import Image, TiffImagePlugin
except ImportError:
    print("need Pillow: pip install Pillow", file=sys.stderr)
    sys.exit(2)


def entropy_bits(histogram, total):
    """Shannon entropy in bits/byte."""
    if total == 0:
        return 0.0
    h = 0.0
    for count in histogram.values():
        if count > 0:
            p = count / total
            h -= p * math.log2(p)
    return h


def run_length_stats(pixels):
    """Run-length stats over the raw byte sequence.

    Returns mean, median, max, and the fraction of bytes in runs >= 8.
    """
    if not pixels:
        return {"count": 0, "mean": 0, "median": 0, "max": 0, "long_frac": 0.0}
    runs = []
    cur = pixels[0]
    run_len = 1
    for b in pixels[1:]:
        if b == cur:
            run_len += 1
        else:
            runs.append(run_len)
            cur = b
            run_len = 1
    runs.append(run_len)
    runs.sort()
    n = len(runs)
    mean = sum(runs) / n
    median = runs[n // 2]
    max_run = runs[-1]
    long_bytes = sum(r for r in runs if r >= 8)
    long_frac = long_bytes / len(pixels)
    return {
        "count": n,
        "mean": mean,
        "median": median,
        "max": max_run,
        "long_frac": long_frac,
    }


def repeat_fraction(pixels):
    """Fraction of positions i (i>0) where pixels[i] == pixels[i-1]."""
    if len(pixels) < 2:
        return 0.0
    matches = sum(1 for i in range(1, len(pixels)) if pixels[i] == pixels[i - 1])
    return matches / (len(pixels) - 1)


def lzw_reencode(pil_image, predictor=None):
    """Re-encode as TIFF-LZW and return the encoded strip length.

    Pillow lets us write LZW with or without the horizontal-differencing
    predictor. We encode to an in-memory buffer, parse the IFD back out to
    find the strip bytes, and return their total length.
    """
    import io

    buf = io.BytesIO()
    save_kwargs = {
        "format": "TIFF",
        "compression": "tiff_lzw",
    }
    if predictor is not None:
        # Pillow expects a dict tag override for TiffTags.PREDICTOR (317).
        tiff_info = TiffImagePlugin.ImageFileDirectory_v2()
        tiff_info[317] = predictor
        save_kwargs["tiffinfo"] = tiff_info
    # Ensure grayscale 8-bit (L mode).
    if pil_image.mode != "L":
        pil_image = pil_image.convert("L")
    pil_image.save(buf, **save_kwargs)
    bytes_ = buf.getvalue()

    # Parse IFD to find the strip bytes. We re-use the minimal walker from
    # benches/scanned_pages.rs: little-endian, StripOffsets=0x111, StripByteCounts=0x117.
    if len(bytes_) < 8 or bytes_[0:2] != b"II":
        return None, None
    ifd_off = struct.unpack_from("<I", bytes_, 4)[0]
    n_entries = struct.unpack_from("<H", bytes_, ifd_off)[0]

    def read_array(tag_entries, type_, count):
        elem = {3: 2, 4: 4}.get(type_)
        if elem is None:
            return None
        total = elem * count
        if total <= 4:
            slice_ = tag_entries
        else:
            off = struct.unpack_from("<I", tag_entries, 0)[0]
            slice_ = bytes_[off : off + total]
        out = []
        for j in range(count):
            if elem == 2:
                v = struct.unpack_from("<H", slice_, j * 2)[0]
            else:
                v = struct.unpack_from("<I", slice_, j * 4)[0]
            out.append(v)
        return out

    strip_offsets = []
    strip_byte_counts = []
    for i in range(n_entries):
        e = ifd_off + 2 + i * 12
        tag, type_, count = struct.unpack_from("<HHI", bytes_, e)
        value = bytes_[e + 8 : e + 12]
        if tag == 0x111:
            strip_offsets = read_array(value, type_, count)
        elif tag == 0x117:
            strip_byte_counts = read_array(value, type_, count)
    encoded_len = sum(strip_byte_counts)
    return encoded_len, len(bytes_)


def analyze_file(path):
    img = Image.open(path)
    if img.mode != "L":
        img = img.convert("L")
    w, h = img.size
    pixels = img.tobytes()  # raw row-major grayscale bytes
    total = len(pixels)

    hist = Counter(pixels)
    entropy = entropy_bits(hist, total)
    rl = run_length_stats(pixels)
    repeat = repeat_fraction(pixels)
    distinct = len(hist)
    mode_byte, mode_count = hist.most_common(1)[0]
    mode_frac = mode_count / total

    # LZW compression ratios — plain and with horizontal predictor.
    lzw_bytes, lzw_total = lzw_reencode(img, predictor=None)
    lzw_hpred_bytes, _ = lzw_reencode(img, predictor=2)

    def ratio(enc):
        return total / enc if enc else float("inf")

    return {
        "file": os.path.basename(path),
        "width": w,
        "height": h,
        "decoded_bytes": total,
        "entropy": round(entropy, 3),
        "repeat_frac": round(repeat, 3),
        "rl_count": rl["count"],
        "rl_mean": round(rl["mean"], 2),
        "rl_median": rl["median"],
        "rl_max": rl["max"],
        "rl_long_frac": round(rl["long_frac"], 3),
        "distinct_values": distinct,
        "mode_byte": mode_byte,
        "mode_frac": round(mode_frac, 3),
        "lzw_bytes": lzw_bytes,
        "lzw_ratio": round(ratio(lzw_bytes), 1),
        "lzw_hpred_bytes": lzw_hpred_bytes,
        "lzw_hpred_ratio": round(ratio(lzw_hpred_bytes), 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="directory or file")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    args = parser.parse_args()

    p = Path(args.path)
    if p.is_dir():
        files = sorted(str(f) for f in p.iterdir() if f.suffix.lower() in (".tif", ".tiff"))
    else:
        files = [str(p)]

    results = [analyze_file(f) for f in files]

    if args.json:
        print(json.dumps(results, indent=2))
        return

    # CSV output
    if not results:
        return
    cols = list(results[0].keys())
    print(",".join(cols))
    for r in results:
        print(",".join(str(r[c]) for c in cols))


if __name__ == "__main__":
    main()
