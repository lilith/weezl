# RVL-CDIP test2 slice — per-class scan characteristics

Stratified sample: 50 email + 50 form from the `chainyo/rvl-cdip`
`test-00002-of-00015.parquet` file (which happens to contain only
these two classes, n=2667 total). Selected uniformly at random with
`seed=42`. Each image is ~762×1000 8-bit grayscale, decoded via
Pillow to raw pixels before measurement.

## Per-class distribution (median + IQR over 50 images each)

| metric | email (n=50) | form (n=50) |
|--------|-------------:|------------:|
| **entropy** (bits/byte) | 0.39 [0.20–0.75] | 0.47 [0.15–1.94] |
| **repeat_frac** | 0.968 [0.93–0.99] | 0.971 [0.90–0.99] |
| **rl_mean** (bytes) | 31 [15–68] | 35 [10–102] |
| **rl_long_frac** (≥8) | 0.964 [0.92–0.98] | 0.968 [0.90–0.99] |
| **mode_frac** (byte=255) | 0.972 [0.94–0.99] | 0.962 [0.69–0.99] |
| **lzw_ratio** (plain) | 21.6 [11–40] | 24.4 [7–54] |
| **lzw_hpred_ratio** | 18.5 [9–33] | 20.0 [7–43] |

## What the numbers say

1. **Real documents are one-byte-dominated** (median 96-97% of pixels =
   255) — not two-palette, not bimodal, just "white with text
   sprinkled on top." My earlier `FLAT_UI` synthetic archetype
   (palette_size=2) mis-modeled this.

2. **Long runs of white are 96-97% of bytes.** Across 100 images, the
   median fraction of bytes sitting in runs of ≥8 is 0.96-0.97. This
   is the workload the chunked / `last_decoded` / mini-burst
   optimizations were built to exploit, and it's real — not a
   pathological edge case.

3. **LZW compression ratios are 20-25× at the median** and range from
   7× (complex form with hand-filled fields) up to 54× (blank form
   or cover sheet). The class medians are nearly identical (21.6 vs
   24.4) but the IQRs differ dramatically — **form IQR spans 7× to
   54×**, a 7.7× internal variance. A single centroid per class is
   an inadequate model; a generator needs to reproduce at least the
   quartile spread.

4. **Horizontal prediction hurts LZW on this content by 14-18%.**
   `lzw_ratio` drops from 21.6 → 18.5 (email) and 24.4 → 20.0 (form)
   when predictor=2 is enabled. This is consistent across 100 images,
   not a one-off quirk of the 15-image demo. The predictor's
   `255,255,255,...` → `255,0,0,...` transformation converts a single
   long LZW string into one byte + a run of zeros, which LZW's
   dictionary can't exploit as efficiently.

   **Recommendation for document-imaging TIFF pipelines: ship with
   `predictor=1` (no prediction).** This is the opposite of the
   default most TIFF writers use (they default to 2 because it helps
   continuous-tone photos). The defaults were written for a different
   workload than we're measuring.

5. **Entropy is extremely low.** Median 0.39-0.47 bits/byte. Pure
   white would be 0. A genuinely random 256-value distribution would
   be 8. Real document scans are closer to the "essentially all
   white" end of that spectrum than to the middle.

6. **The "form" class has a much wider IQR than "email"** on every
   metric. Forms range from "blank preprinted document" (entropy
   ~0.15, ratio 54×) to "fully hand-filled with dense content"
   (entropy 1.94, ratio 7×). Email is comparatively homogeneous
   because all email pages share the same header/footer structure.
   This suggests **per-class is too coarse** — the useful
   specialization axis isn't "email vs form" but "low-entropy vs
   high-entropy within any class."

## Practical consequences

**For the synthetic generator**: refit `FLAT_UI` with
`palette_size=1, mode_byte=255`, plus a content-byte distribution
that hits entropy 0.4-0.5 and repeat fraction 0.97. The right model
is probably a two-state Markov process (BG=white, FG=text ink) with
switch probabilities tuned to hit target `rl_mean` 30-40 and target
`lzw_ratio` 20-25.

**For the decoder specialization decision** (see
`docs/specialization-design-space.md`): the in-class variance is so
high that a compile-time `Document` profile would be suboptimal for
a big chunk of any single class. Runtime probing is the right answer.

**For the `horizontal predictor` recommendation**: add a note to
weezl's TIFF integration docs explaining that `predictor=1` is the
right choice for document scans even though encoders default to
`predictor=2`. Most users won't know.

## Limitations

- **Only two classes sampled.** `test-00002-of-00015.parquet` only
  contains email + form. Full cross-class variance needs downloading
  more parquet files (each is 138-500 MB). Budget permitting, grab
  one more to cover high-entropy classes (news, presentation,
  advertisement) and one more for intermediate (letter, memo, resume).
- **All images are ~762×1000.** Bigger scans (2550×3300 at 300 DPI)
  may show different run-length distributions because margins are a
  larger fraction of the total pixel area. The shape of the metric
  should hold; absolute compression ratios may shift upward.
- **Source is PackBits**, which is lossless but was encoded by
  whatever preprocessing RVL-CDIP's original pipeline used. The raw
  pixel bytes we measure may have been post-processed (e.g. gamma
  corrected, contrast-stretched, cropped). We can't separate
  "inherent document byte distribution" from "what RVL-CDIP's
  preprocessing did."
