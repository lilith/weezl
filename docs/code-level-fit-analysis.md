# Code-level fit analysis — byte stats don't predict decoder perf

tl;dr: our byte-stat-fitted synthetic archetypes pass their fit targets
within ±5% on entropy, run-length mean, and compression ratio, but
their **LZW code streams look nothing like real RVL-CDIP code streams**.
Four of the most decoder-relevant code-level features are off by 50-100%
on `SCANNED_BLANK` and 50-80% on `SCANNED_EMAIL`. Benching our three
strategies (Classic, Chunked, Streaming) on these archetypes will give
conclusions that don't transfer to real documents.

## What the profiler measures

`tools/fit_generator.py` now has a `_LzwProfiler` that feeds a byte
stream through Pillow's TIFF-LZW encoder, then decodes the resulting
strip with a minimal MSB-first TIFF-early-change LZW decoder,
collecting per-code feature histograms. No timing involved — just
counters. The output is a dict of:

- `total_codes`, `codes_per_decoded_byte`
- `literal_frac` (code < clear_code, value_len = 1)
- `short_copy_frac` (value_len ≤ 8, chunked/streaming fast path)
- `long_copy_frac` (value_len > 8, chunked shines, streaming pays)
- `kwkwk_frac` (`code == save_code`, pure solid-color indicator)
- `mean_value_len`, `long_copy_byte_frac`, `max_chain_len`
- `clears` (table-fill events)
- `width_{9,10,11,12}` distribution of per-code bit width

These are the decoder-relevant features: they directly map onto the
cost model for each of the three strategies in `weezl::decode`.

## Synthetic archetypes vs real RVL-CDIP: code-level divergence

### SCANNED_EMAIL (synth) vs real email class (n=50 RVL-CDIP)

```
metric                         real        synth    rel_err
total_codes              26660.0000   28424.0000      +6.6%
codes_per_decoded_byte       0.0354       0.0373      +5.4%
literal_frac                 0.6007       0.5028     -16.3%
short_copy_frac              0.1963       0.1886      -3.9%
long_copy_frac               0.2084       0.3075     +47.6% ⚠
kwkwk_frac                   0.1690       0.0751     -55.6% ⚠
mean_value_len              29.6300      26.8400      -9.4%
long_copy_byte_frac          0.9611       0.9626      +0.2%
width_9                      0.1760       0.1860      +5.7%
width_10                     0.2010       0.3350     +66.7% ⚠
width_11                     0.2600       0.4140     +59.2% ⚠
width_12                     0.3540       0.0650     -81.6% ⚠
```

Most relevant: **`width_12 = 35% vs 6.5%` (−81.6%)**. Real email
spends over a third of its decoded codes at the saturated 12-bit code
width — the LZW table has filled the full 4096 entries. Synth email
barely reaches 12 bits. This means real email's decode loop is
exercising the widest-code path far more often than synth does.

**`kwkwk_frac = 17% vs 7.5%` (−55.6%)**: real email has more than
twice the KwKwK fraction. Real email has more locally-repeating text
strokes that drive the `code == save_code` case that weezl's Streaming
optimization targets.

**`long_copy_frac = 20.8% vs 30.8%` (+47.6%)**: synth email has more
long copies. Paradoxically, synth's value_len distribution is shifted
toward longer values than real, even though the mean is lower
(26.8 vs 29.6). The shape is different — synth is bimodal around a
shifted peak, real is right-tailed.

### SCANNED_BLANK (synth) vs real form class (n=50 RVL-CDIP)

```
metric                         real        synth    rel_err
total_codes              25329.0000   11655.0000     -54.0% ⚠
codes_per_decoded_byte       0.0326       0.0153     -53.1% ⚠
literal_frac                 0.4857       0.4530      -6.7%
short_copy_frac              0.1682       0.0613     -63.6% ⚠
long_copy_frac               0.2951       0.4831     +63.7% ⚠
kwkwk_frac                   0.1552       0.3019     +94.5% ⚠
mean_value_len              32.6100      65.5400    +101.0% ⚠
long_copy_byte_frac          0.9626       0.9903      +2.9%
width_9                      0.2020       0.3620     +79.2% ⚠
width_10                     0.3360       0.5550     +65.2% ⚠
width_11                     0.2740       0.0840     -69.3% ⚠
width_12                     0.1190       0.0000    -100.0% ⚠
```

This is catastrophic. **Synth blank has half the total codes of real
form.** Mean value length is twice as long, KwKwK fraction is double,
and the code-width distribution is compressed into 9- and 10-bit
codes with zero 12-bit codes. Real form pages have real content
(printed rule lines, field labels, margin text) that breaks up the
long runs and fills the LZW table much faster. Our byte-stat-fitted
"blank" archetype is *more* pure-white than any real form in the
corpus — it passes the byte stats but represents a workload that
doesn't exist in real data.

## Why this happened

Byte-stat fitting matched six real-corpus medians (entropy,
repeat_frac, rl_mean, rl_long_frac, mode_frac, lzw_ratio) within
±5%. That bound sounds tight but it's computed over **whole-image
byte histograms and run-length distributions**. Those are smooth
summaries of the raw pixel data. The LZW encoder, however, is a
**content-sensitive pattern learner**: it builds a dictionary keyed
by byte sequences it's already seen. Two byte streams with identical
histograms and run lengths can produce very different LZW dictionaries
because the encoder is sensitive to **which** bytes appear next to
**which** other bytes — structure, not just distribution.

Concretely, a real form page's "black pixel between two white pixels"
happens at pixel positions correlated with the printed rule line.
Our synthetic generator places black pixels geometric-randomly via
`p_bg_to_fg`, with no spatial structure. Over a whole image the
marginal byte distributions look identical, but the LZW encoder
sees:

- Real form: repeatable patterns like `"ws ff ws ff"` (rule line
  crossing 8-pixel columns) → table entries for the repeat; lots of
  `width_12` codes.
- Synth blank: geometric-distributed isolated dark pixels → LZW
  never sees the same prefix twice → table fills slowly, stays at
  width 9–10, uses KwKwK for the long white runs instead.

The fix isn't parameter refit — it's a generator model upgrade.
Reproducing code-level features requires spatial structure that a
two-state Markov chain literally cannot produce, regardless of how
many parameters you tune.

## Implications for "fit for bpp perf"

The original pitch was: fit synthetic archetypes to reproduce real
decoder performance triples `(classic_bpp, chunked_bpp, streaming_bpp)`
so we don't need the real corpus for every bench. This analysis says:

1. **The current synthetic archetypes will give misleading perf
   answers** for document workloads. Even if we fit them perfectly
   to the byte-level class medians, the code-level streams diverge
   enough that a relative comparison between strategies on the synth
   won't predict the relative comparison on real data.

2. **Byte-stat fitting is still useful as a pre-screen.** If an
   archetype misses its byte-stat target, it's definitely wrong.
   Passing byte-stat fit is necessary but not sufficient.

3. **The valid path forward is one of:**
   - Bench directly on real RVL-CDIP strips. Reuse
     `benches/scanned_pages.rs`'s IFD walker with the larger corpus
     we already have at `/mnt/v/input/scanned-docs/rvl-cdip-test2/`.
     This is the simplest and most defensible option.
   - Upgrade the generator to a structured model that can produce
     spatial patterns: render-text-onto-white (PIL.ImageDraw with
     realistic fonts, margins, noise) or a block-structured 2D
     Markov process. Much more work, and still has fitting risk.
   - Hybrid: use byte-stat-fit synthetic for regression screening
     (fast, cheap, deterministic), and always validate perf
     conclusions on a small real corpus before publishing.

4. **Code-level features should become first-class fit targets.**
   The profiler is cheap enough to run during the fit inner loop.
   Add `long_copy_frac`, `kwkwk_frac`, and `width_12` to the fitting
   error function alongside the byte-stat targets. This won't fix
   the structural-model problem, but it'll at least catch
   catastrophic divergence like the blank one.

5. **"Fitting for bpp perf" is best done as a validation step, not
   a primary fit target.** Once benches are back, measure the three
   strategies' throughputs on both real and synthetic archetypes.
   If the synthetic ranking matches the real ranking for every
   archetype, our synthetic bench is adequate as a proxy. If not,
   we know which archetypes to rebuild or replace with real data.

## What this doesn't answer yet

- Whether the divergence actually matters for `throughput_rank` on
  the three strategies. The cost model says yes (long_copy_frac and
  width_12 are load-bearing for the chunked-vs-streaming crossover),
  but we haven't measured it. This is what the actual perf fit would
  confirm once benches can run.
- Whether the real email/form classes are themselves heterogeneous
  enough that a single centroid doesn't capture the per-strategy
  winner. The in-class IQR was 7× on lzw_ratio (form) — we should
  check whether it's also wide on `width_12` and `kwkwk_frac`.
- Whether a structured generator (render-text-onto-white) can
  actually reach the real code-level features. My prior says yes,
  but I haven't validated.
