# Design space for workload-specialized decoding

How should weezl's streaming decoder specialize for different byte
distributions (documents vs screenshots vs photos) without exploding
binary size or pushing caller burden through the public API?

This document surveys six options end-to-end, ranks them on four
dimensions, and picks the one to actually build first.

## Background: what needs to specialize

The streaming decoder has three orthogonal workload-sensitive knobs:

1. **Mini-burst literal + short-copy fast path.** Helps screenshots and
   rich content where short codes dominate. Hurts documents because
   the fast path is rarely entered — its code is dead weight that
   still fattens the hot loop and competes for icache.
2. **`last_decoded` reference plumbing + KwKwK `copy_within`.** Helps
   sustained KwKwK (solid-color, big white margins on documents).
   Adds ~1-2% per non-KwKwK code via per-iteration bookkeeping.
3. **Cold-path markers** for CLEAR/END/KwKwK/spill branches. Nearly
   free — the marker call is a no-op function that compiles away but
   its `#[cold]` attribute persists as a layout hint. Always a win.

Knob 3 is unconditionally correct. Knobs 1 and 2 are the decision.

## The six options

### A. Pure runtime probe, single monomorphization

**Shape.** Decoder reads ~8 KB of input (or one strip) before entering
the hot loop. Probe samples code-width growth, literal:copy ratio, and
early compression ratio. Sets a runtime enum or flag bundle, then the
main `advance()` matches on the flag at every code.

**Binary cost.** Zero extra monos. The branch body is always present.
**Caller burden.** Zero.
**Runtime cost.** One well-predicted branch per code (~1-2%).
**Flexibility.** Very high — can re-probe per strip, change strategy
mid-stream, add new workload categories without ABI churn.

**Pros.** No caller has to know anything. Works correctly for mixed
workloads. New profiles don't need new public API.

**Cons.** Every user pays the branch cost even when they would have
compiled down to the optimal specialized version. Probe cost itself
is nontrivial for tiny streams (many small GIF frames).

### B. Probe picks a `Box<dyn Stateful>` at build time

**Shape.** `TableStrategy::Auto` reads the first strip, picks among
`Classic`, `Chunked`, `Streaming`, and hands back the appropriate
trait object. The rest of the decode is virtual-dispatched exactly as
today.

**Binary cost.** Zero beyond what we already have.
**Caller burden.** Zero if they pick `Auto`; same as today otherwise.
**Runtime cost.** Zero in the inner loop — the dispatch happens once
at `build()`.
**Flexibility.** Medium — we're locked into the 3 existing strategies.
New strategies require a new enum variant and a new mono.

**Pros.** Uses machinery we already have; smallest delta from today.
Perfectly preserves `advance()` hot paths.
**Cons.** Auto's probe can be wrong on multi-strip files where strip
0 isn't representative. Requires the probe to correctly rank 3 very
different decoders based on a short sample.

### C. Runtime bool flags inside a single decoder

**Shape.** `DecodeStateStreaming` gains three bool fields —
`mini_burst_enabled`, `last_decoded_enabled`, `cold_markers_enabled` —
set at `build()` time (by user config, probe, or default). The hot
loop conditions each optimization on the flag. LLVM can't DCE because
the flags aren't const, but branch prediction nails the hot path
after warmup.

**Binary cost.** Slightly larger `advance()` (all paths present), but
same mono count as today. ~500 bytes growth.
**Caller burden.** Low — can expose a `StreamingMode::Document/Screenshot/Auto`
enum that sets the flags; `Auto` runs the probe.
**Runtime cost.** ~1-2% per code from the flag checks. Competitive
with A.
**Flexibility.** High. Adding a fourth knob is a new bool, not a new
mono and not new public API.

**Pros.** Simplest thing that works. Lets us experiment with flag
combinations during development without committing to compile-time
public API. Can become option D later if any specific flag set is
proven dominant enough to warrant dedicated monos.
**Cons.** Leaves some performance on the table vs const-generic
because LLVM can't prove the flags don't change.

### D. Const-generic profiles with shared body

**Shape.** A `StreamingProfile` trait with `const` bool fields. Four
marker types: `Compat`, `General`, `Screenshot`, `Document`. The hot
loop writes `if Prof::LAST_DECODED { ... }` and LLVM's DCE eliminates
the dead arms in each monomorphization. Crucially, the helper
functions (`reconstruct_streaming_into`, `derive`, bit packing) are
NOT profile-generic — they're shared across all profiles via
`#[inline(never)]`, so only the outer loop's ~200-400 bytes of
profile-gated code is duplicated.

**Binary cost.** 4 profile monos × 4 existing (LSB/MSB × yield) =
16 streaming monos. With shared helpers, ~200-400 bytes unique per
mono → **~3-8 KB growth total**, not the 16-32 KB I claimed earlier.
**Caller burden.** Medium — pick a profile at compile time. Library
crates that use weezl downstream can't satisfy heterogeneous users
without picking the most-general profile, which defeats the point.
**Runtime cost.** Zero. Each mono is optimal for its workload.
**Flexibility.** Lowest. Profile names and behaviors become public
API that's hard to evolve.

**Pros.** Maximum performance per mono. No runtime branches.
**Cons.** The public API commitment is real. Every profile combination
is a forever-decision until a major version bump. Library downstream
users can't take advantage.

### E. Hybrid: const-generic bit-order/yield, runtime profile

**Shape.** Keep the existing bit-order × yield const generics (they
matter for codegen — LSB's over-read trick is genuinely different
from MSB's). Profile becomes a runtime field like option C. One
enum field + match, inside each of the 4 existing monos.

**Binary cost.** Same as C (~500 bytes per mono). Same mono count as
today.
**Caller burden.** Low. `Configuration::with_streaming_profile(Auto)`.
**Runtime cost.** Same as C (~1-2%).
**Flexibility.** High. New profiles = new enum variants.

**Pros.** Matches the current const-gen story for the axes that
actually matter (bit order) while avoiding API commitment for the
axes that are workload-sensitive. Can evolve profile set without
breaking callers.
**Cons.** Same ~1-2% branch cost as C. Not obviously better than C
unless bit-order/yield specialization was carrying its weight, which
we haven't measured.

### F. Runtime self-tuning

**Shape.** Decoder starts in a neutral mode, measures its own
performance on the first few hundred codes, picks the profile that
best fits observed characteristics, then commits to it for the rest
of the stream. No caller probe needed.

**Binary cost.** Same as C (~500 bytes). Single mono per bit-order/yield.
**Caller burden.** Zero.
**Runtime cost.** ~1-2% during the tuning window, zero after (if we
switch to a specialized inner loop post-tuning).
**Flexibility.** Highest.

**Pros.** Fully automatic. Works for mixed workloads because it can
re-tune. Matches how CPU branch predictors adapt.
**Cons.** Complex. Tuning logic is an extra ~1 KB of code. Hard to
reason about — two runs on the same input can produce different
specialization histories.

## Ranking

| | A: runtime probe | B: probe→dyn | C: runtime flags | D: const profiles | E: hybrid | F: self-tuning |
|---|---|---|---|---|---|---|
| Binary size | **best** (0) | best (0) | good (+500B) | medium (+3-8 KB) | good (+500B) | **best** (+500B) |
| Caller burden | **none** | none | **none** w/ Auto | medium | low | **none** |
| Runtime cost | ~1-2% | **zero** | ~1-2% | **zero** | ~1-2% | ~1-2% tuning, zero after |
| Flexibility | **high** | medium | **high** | low | **high** | **highest** |
| API commitment | zero | same as today | zero | **high** | low | zero |
| Complexity | low | **trivial** | low | medium | medium | high |

## Recommendation

**Build option C first, with an `Auto` mode that runs a probe.** It's
the simplest thing that works. We pay 1-2% on every code for a new bool
check (acceptable given the optimization saves much more than that
when it fires), but we commit to no new public API, and the machinery
is a direct extension of what's already in the decoder.

**Upgrade to option D selectively, if and only if profiling shows
that specific profile × bit-order combinations are worth dedicated
monos.** That decision needs real-corpus measurements, which is why
it's blocked on the RVL-CDIP bulk download and a better-fitted
synthetic corpus.

**Don't build option B.** It sounds easy — "just add an Auto variant"
— but the probe logic that correctly ranks Classic vs Chunked vs
Streaming based on 8 KB is the same engineering problem as option F's
runtime tuning, and `dyn Stateful` gives up the ability to re-tune
mid-stream if strip 0 wasn't representative.

**Don't build option F.** Neat, but complex, and the self-tuning
logic requires us to solve the "identify workload from bytes" problem
twice — once in the probe and once in the adapter. Save it for
weezl 2.0 if 1.x turns out to need it.

## Implementation sketch for option C

```rust
#[derive(Clone, Copy, Debug, Default)]
pub enum StreamingMode {
    #[default]
    /// Probe the first strip / first ~8 KB of input and choose
    /// between the internal flag sets. Recommended for anyone who
    /// doesn't already know their input.
    Auto,
    /// Rich content / web screenshots: mini_burst on, last_decoded off.
    Screenshot,
    /// Document scans / high-compression palette: mini_burst off,
    /// last_decoded on.
    Document,
    /// Both optimizations enabled. Good if you have no idea and
    /// don't want the probe cost.
    General,
    /// No optimizations. Matches the pre-investigation streaming
    /// decoder. Here for A/B testing and historical regression checks.
    Compat,
}

impl Configuration {
    pub fn with_streaming_mode(mut self, mode: StreamingMode) -> Self {
        self.streaming_mode = mode;
        self
    }
}

// Internal, in DecodeStateStreaming:
struct DecodeStateStreaming<...> {
    // ...
    mini_burst_enabled: bool,
    last_decoded_enabled: bool,
    // ...
}

fn advance(&mut self, ...) -> BufferResult {
    // ...
    loop {
        // ...
        if code < self.clear_code {
            // literal path — unchanged
            if self.mini_burst_enabled {
                // mini-burst while loop
            }
        } else if code == self.save_code {
            // KwKwK path
            let first = if self.last_decoded_enabled && prev_len > STREAMING_Q {
                // fast path via last_decoded
            } else {
                // slow path via chain walk
            };
            // ...
        }
    }
}
```

The `Auto` variant dispatches in `build()` by reading ~8 KB of the
first input given to `decode_bytes`. Since `build()` returns a
`Box<dyn Stateful>` before any input has been seen, the probe actually
runs in the **first call** to `decode_bytes` — the decoder is created
in `Probing` state, the first call consumes ~8 KB, decides, and
switches to `Document` / `Screenshot` / `General` before the hot
loop runs. Subsequent calls skip the probe.

## Open questions blocked on real data

1. **Is the probe signal strong enough?** Can we reliably distinguish
   document-ish from screenshot-ish from photo-ish within the first
   8 KB? Needs real-corpus measurements across all three categories.
2. **How much does each knob actually matter on real workloads?** On
   the synthetic corpus, `last_decoded` was +290% on pure KwKwK. On
   the real RVL-CDIP data, it was +10-19%. We don't yet know the
   screenshot-category real-data number.
3. **Does the ~1-2% option-C cost matter?** Only measurable on real
   corpora after the probe is implemented.

Resolving these unblocks the D-upgrade decision.
