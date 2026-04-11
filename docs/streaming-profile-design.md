# Const-generic streaming profiles

The current `TableStrategy` enum dispatches between three distinct
decoder implementations at build time (via the `make_state!` macro
matching on bit order × yield × strategy). Each combination is a
separate monomorphization already — the enum is just a discriminated
selector, not a runtime dispatch. This is good, but the streaming
decoder currently folds several *mutually independent* optimizations
into a single hot loop:

  1. Mini-burst literal+short-copy fast path.
  2. `last_decoded` reference plumbing + KwKwK `copy_within` fast path.
  3. Cold markers for CLEAR/END/KwKwK/spill.
  4. The PreQ+SufQ table layout (unconditional).
  5. The wuffs-style single-code outer loop (unconditional).

When the KwKwK optimization gets compiled in, the extra code inflates
the function and slightly slows the highest-throughput workloads
(gif/flat-ui). When it's compiled out, the solid-color document path
regresses. There is no single setting that's ideal for every workload.

A const-generic **profile** converts these workload-sensitive knobs
into compile-time parameters. Each caller picks the right profile for
their use case; the compiler monomorphizes a dedicated hot loop per
profile with dead branches eliminated at codegen. Zero runtime cost in
any profile.

## API sketch

```rust
/// Compile-time knobs for the streaming decoder.
///
/// Implementers are zero-sized marker types whose const fields control
/// which optimizations the monomorphized `advance()` body compiles in.
/// All fields are `const fn` so LLVM can evaluate them at build time
/// and dead-code-eliminate the disabled paths.
pub trait StreamingProfile: 'static {
    /// Enable the literal/short-copy mini-burst fast path.
    /// Turning this off shrinks the hot loop by ~200 instructions, which
    /// helps benchmarks where most codes are long COPYs or KwKwK and the
    /// fast path is rarely entered.
    const MINI_BURST: bool;

    /// Maintain `last_decoded: Option<&[u8]>` and use it in the KwKwK
    /// path to memcpy the previous code's output instead of walking the
    /// table chain. ~290% win on sustained KwKwK, ~2% cost on
    /// non-KwKwK workloads due to the extra per-code bookkeeping.
    const LAST_DECODED: bool;

    /// Emit `#[cold]` marker calls in the CLEAR/END/KwKwK/spill
    /// branches so LLVM lays them out at the end of the function,
    /// keeping the hot LITERAL/COPY basic blocks compact. This is
    /// essentially free — the marker call itself is an empty function
    /// that gets fully optimized away but the layout hint persists.
    const COLD_MARKERS: bool;

    /// Human-readable name for reports.
    const NAME: &'static str;
}

/// Default profile: compatibility with the pre-optimization decoder.
/// No mini-burst, no last_decoded, no cold markers. Use this for
/// head-to-head comparison against the old behavior.
pub struct Compat;
impl StreamingProfile for Compat {
    const MINI_BURST: bool = false;
    const LAST_DECODED: bool = false;
    const COLD_MARKERS: bool = false;
    const NAME: &'static str = "compat";
}

/// Balanced profile: all three optimizations enabled. Good default for
/// general-purpose callers that don't know their input well.
pub struct General;
impl StreamingProfile for General {
    const MINI_BURST: bool = true;
    const LAST_DECODED: bool = true;
    const COLD_MARKERS: bool = true;
    const NAME: &'static str = "general";
}

/// Screenshot profile: mini-burst on, last_decoded off. Tuned for
/// rich-content web screenshots where short copies dominate and
/// KwKwK is rare. Same code size as the pre-KwKwK streaming decoder.
pub struct Screenshot;
impl StreamingProfile for Screenshot {
    const MINI_BURST: bool = true;
    const LAST_DECODED: bool = false;
    const COLD_MARKERS: bool = true;
    const NAME: &'static str = "screenshot";
}

/// Document profile: last_decoded on, mini-burst off. Tuned for
/// scanned pages with long whitespace runs — most codes are long
/// COPYs or KwKwK, and the mini-burst body is dead weight.
pub struct Document;
impl StreamingProfile for Document {
    const MINI_BURST: bool = false;
    const LAST_DECODED: bool = true;
    const COLD_MARKERS: bool = true;
    const NAME: &'static str = "document";
}
```

The decoder struct gains a profile parameter:

```rust
pub(crate) struct DecodeStateStreaming<
    P: StreamingBitPacking,
    CgC: CodegenConstants,
    Prof: StreamingProfile = General,
>
```

In `advance()`, every profile-gated block becomes a `const` condition:

```rust
// Instead of unconditional:
let mut last_decoded: Option<&[u8]> = None;
// ...
while self.n_bits >= self.width && out.len() >= STREAMING_Q {
    // mini-burst body
}

// Becomes:
let mut last_decoded: Option<&[u8]> = if Prof::LAST_DECODED { None } else { None };
// (type-shaped placeholder; LLVM eliminates if not read)
// ...
if Prof::MINI_BURST {
    while self.n_bits >= self.width && out.len() >= STREAMING_Q {
        // mini-burst body
    }
}
```

LLVM sees each `if Prof::FIELD` as a constant-false branch when the
profile disables it, and the body compiles to zero instructions in that
monomorphization. The `last_decoded` variable itself also eliminates
when unused because LLVM's DCE can prove no reads occur.

The `streaming_cold_marker` calls already exist and are cheap to gate:

```rust
if Prof::COLD_MARKERS {
    streaming_cold_marker();
}
```

## Public API integration

The existing `TableStrategy::Streaming` variant stays and maps to
`StreamingProfile = General` (the current behavior). A new
`Configuration` method exposes the profiles:

```rust
impl Configuration {
    /// Build a decoder with a specific streaming profile. This is a
    /// more specialized version of `with_table_strategy(Streaming)`.
    pub fn with_streaming_profile<Prof: StreamingProfile>(self) -> Self {
        // set an internal marker that `build()` dispatches on
    }
}
```

Because profile selection happens at build-time, `build()` has to
return a `Box<dyn Stateful>` and internally pick which monomorphization
to instantiate based on a runtime profile enum. The enum would have 4
variants (`Compat`, `General`, `Screenshot`, `Document`) and `build()`
matches on it, calling `DecodeStateStreaming::<P, CgC, Compat>::new()`
etc. Each branch of the match is its own monomorphization.

The existing 4 streaming monomorphizations (2 × yield × 2 × bitorder)
multiply by profile count: 4 → 16 if we ship 4 profiles. Binary size
grows by roughly the advance() body per profile (~2-4 KB each, so
~8-16 KB total). Acceptable.

## Tradeoffs

**Pro:**
  - Zero runtime cost per profile — dead code eliminated at compile.
  - Users can opt into expensive optimizations only when they pay off.
  - Benchmark can compare all profiles in one run, giving evidence for
    the workload→profile mapping.
  - The ugly "does the copy_within hurt flat-ui enough to skip it?"
    question disappears — you just pick the profile you need.

**Con:**
  - 16 streaming monomorphizations instead of 4. ~16 KB binary growth.
  - Users have to pick, which means we have to document the picking.
    Wrong choice = worse performance.
  - Profile set needs periodic re-fitting as workloads evolve.

## Recommended rollout

1. **Land the cold-marker pattern unconditionally** (already in tree).
   Zero cost, helps every profile.
2. **Add the `StreamingProfile` trait** with the 4 profiles above.
   Existing `TableStrategy::Streaming` → `General`.
3. **Re-fit synthetic archetypes** against the RVL-CDIP characteristics
   (see `docs/scanned-page-characterization.md`). This gives us real
   workload models to test each profile against.
4. **Expose profiles via `Configuration::with_streaming_profile<P>()`**.
5. **Document the picking** in `TableStrategy::Streaming`'s doc: "for
   document workloads, prefer `Document`; for rich web screenshots,
   prefer `Screenshot`; for unknown mixed input, `General`."

Chunked becomes the recommendation for high-compression-ratio real
documents (based on the actual benchmark result: Chunked beats
Streaming-anyprofile by 5-15% on the RVL-CDIP content class). The
`Document` profile exists for callers that want the streaming API
(mid-frame suspension, yield-on-full) but still tune for documents.
