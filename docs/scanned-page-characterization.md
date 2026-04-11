# Scanned-page characterization for LZW benchmarking

This document records the byte-distribution characteristics of real
scanned document TIFFs, compares them to the synthetic archetypes the
strategy-compare benchmark uses, and sketches what a better synthetic
generator would need to produce.

## Corpus

`nielsr/rvl-cdip-demo` on HuggingFace — 15 images, one per document
class (minus one) from the RVL-CDIP 400k-image dataset. Each image is
an 8-bit grayscale TIFF at ~762×1000 pixels. Source compression is
PackBits; we decode to raw pixels for analysis.

The 16 RVL-CDIP classes cover real document diversity: advertisement,
budget, email, form, handwritten, invoice, letter, memo, news, note,
presentation, questionnaire, resume, scientific, specification, report.
The 15 demo images are one per class (report missing).

Run the analyzer via:

    tools/analyze_scanned_pages.py /mnt/v/input/scanned-docs/rvl-cdip-demo/tiffs

Full CSV in `docs/rvl-cdip-demo-characteristics.csv`.

## Findings

### Shape of real scanned page byte distributions

Every real scan, across every class, has **one dominant byte value = 255
(white)** that accounts for **80-98%** of all pixels. There are no
"two-palette" scans. The characteristic distribution is:

    mode byte = 255 (always)
    mode fraction = 0.80 - 0.99
    entropy = 0.26 - 2.06 bits/byte
    distinct values = 256 (all values present, but most are near-zero)

| class          | entropy | mode frac | LZW ratio (no pred) |
|----------------|---------|-----------|---------------------|
| note           | 0.26    | 0.98      | 46.0                |
| form           | 0.28    | 0.98      | 36.7                |
| letter         | 0.65    | 0.94      | 15.3                |
| invoice        | 0.83    | 0.93      | 10.7                |
| specification  | 0.85    | 0.93      |  9.8                |
| questionnaire  | 0.97    | 0.92      |  8.6                |
| email          | 1.03    | 0.92      |  7.8                |
| budget         | 1.06    | 0.91      |  9.8                |
| scientific     | 1.16    | 0.90      |  7.0                |
| handwritten    | 1.30    | 0.89      |  6.2                |
| memo           | 1.50    | 0.88      |  5.1                |
| resume         | 1.65    | 0.86      |  4.7                |
| news           | 1.96    | 0.83      |  3.8                |
| advertisement  | 2.05    | 0.80      |  4.1                |
| presentation   | 2.06    | 0.82      |  3.6                |

### Run-length structure: median 1, mean 5-70, long-tail dominated

Run-length **median is always 1**, but the **mean ranges 4.6-71**. This
is a bimodal distribution: most runs are short (single bytes inside
text strokes, at anti-aliased edges) but a few runs are very long (the
margin areas). Specifically, **75-99% of bytes live in runs ≥ 8**
(`rl_long_frac` in the CSV). The long tail is what LZW exploits.

### Horizontal prediction hurts

All the CSV's `lzw_hpred_ratio` columns are *lower* than `lzw_ratio`.
Horizontal differencing turns a run of 255s into `255, 0, 0, 0, …`,
which is worse for LZW than the raw run: the raw run collapses to a
single long string entry in the code table, but the differenced version
has a high-entropy prefix byte followed by zeros.

> **Practical implication:** real-world TIFF-LZW producers should leave
> the predictor off (or set it to 1 = "no prediction") for scanned
> grayscale documents. Most encoders set predictor=2 by default because
> it helps natural-photo content, but it's the wrong default here.
> Our synthetic corpus reflects this: we will benchmark with
> `predictor=1` to match how real document-imaging pipelines ship.

### What my ImageMagick-rendered pages got wrong

The `page-{title,sparse,dense,full}.tif` files rendered via ImageMagick
are *too clean*: entropy 0.01-0.13, mode fraction 0.99, compression
ratio 59-122×. That's ~4× better compression than the best real RVL-CDIP
scan. Real scans have anti-aliasing at letter edges, sensor noise in
the "white" margins (255 ± a few), and halo artifacts around darker
strokes. Those imperfections make real scans measurably harder to
compress — and therefore more representative of the actual LZW decode
bottleneck.

### What a realistic synthetic generator needs

A better generator model, fitted to the RVL-CDIP centroid:

    palette_mode        ≈ 255 (dominant byte)
    background_noise    ≈ N(255, 2) clamped [240, 255]
    content_frac        ≈ (1 - mode_frac)  // 2-20% of pixels
    content_byte_range  ≈ [0, 200] sampled non-uniformly
    content_run_mean    ≈ 1-3 (short strokes)
    background_run_mean ≈ derived from target repeat_frac

A simple two-state Markov process generates this well: state {BG, FG},
start in BG, switch to FG with probability `p_start`, switch back with
probability `p_end`. Inside BG, emit `255 + small_noise`. Inside FG,
emit uniform [0, 200]. Tune `p_start`, `p_end`, and FG distribution
to match the `entropy`, `mode_frac`, and `lzw_ratio` targets.

The existing `GenParams { palette_size, run_mean, nearby_prob,
nearby_range }` struct already has the right shape for this — it just
needs a fit per RVL-CDIP centroid instead of the earlier TIFF-photo
centroid fits. The fit inputs are the four columns `entropy`,
`repeat_frac`, `rl_mean`, `lzw_ratio` from the CSV, and the existing
Nelder-Mead fitter from the wuffs-parity investigation works directly.

## Limitations of the 15-image demo

Fifteen images is a reasonable stratified sample across 15 of the 16
classes — one per class — but it's too small to fit a per-class
generator reliably. For a production-quality fit we want 20-50 images
per class (300-800 total). The full RVL-CDIP dataset has 25k per class
and is reachable via HuggingFace `chainyo/rvl-cdip` parquet files; the
same analyzer works on it.

For now, the 15-image sample is enough to show that my earlier
synthetic archetypes substantially mis-characterized the real workload
(two-palette vs one-dominant-byte), and to justify re-fitting with a
better model before re-running the benchmark comparison.
