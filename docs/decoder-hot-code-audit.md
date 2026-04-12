# Decoder hot-code audit

Snapshot-in-time `cargo asm` inspection of the current
`combined-strategies` branch decoder, post-cold-marker and
post-`last_decoded` changes. Goal: confirm the optimizations we
added are visible in codegen, and find any hot code that *should*
have similar treatment but doesn't.

All measurements were done with `cargo asm --release --lib`, reading
the assembly dumps directly (not benchmarks — system is busy).

## Streaming decoder (`DecodeStateStreaming::advance`)

Variant measured: first monomorphization, ~1107 lines of asm
(≈4–5 KB of x86_64 code). Results are consistent across the 4 streaming
monos (LSB/MSB × yield-on-full).

### What worked

1. **`reconstruct_streaming_into` is fully inlined.** Zero symbol
   references in the asm. `#[inline(always)]` is doing its job — the
   Q-chunk chain-walk body is materialized at each of its ~2 call
   sites inside `advance()`.

2. **`bump_width_slow` is correctly out-of-line.** 7 call sites in
   the asm, all jumping to the same external function. `#[cold]
   #[inline(never)]` works. Each call is reached from an inlined
   `derive()` that decrements `codes_until_bump` and branches to the
   cold stub when it hits zero.

3. **`streaming_cold_marker` calls are fully elided.** No symbol
   references — LLVM inlined the empty body and DCE'd it. But the
   `#[cold]` attribute did its job for layout: panic blocks and the
   KwKwK / CLEAR / END / spill branches all live in the tail of the
   function (asm lines 785–1107 out of 1107), not interleaved with
   the LITERAL and COPY hot blocks at the top.

### What's still there

4. **`derive()` inlines 7 times.** Each inline copy is ~20 instructions
   for the table-entry write path. 7 × 20 = ~140 instructions of
   "bloat," which sounds like a lot but is correct: `derive()` runs
   every single code and moving it out-of-line would add per-call
   function overhead that swamps the 140-instruction saving. Keep.

5. **6 panic sites remain** in `advance()`, all at the tail of the
   function (lines 785, 794, 913, 977, 1098, 1107). They come from
   safe-slice indexing where LLVM can't prove the runtime guard (e.g.
   `out[wr..wr + value_len]` where `value_len <= out.len()` was
   checked but LLVM doesn't track the relationship). Eliminating them
   would require either `unsafe { get_unchecked }` (forbidden by
   `#![forbid(unsafe_code)]`) or restructuring to array patterns
   LLVM can analyze. Two of them (lines 913 and 977) are physically
   interleaved with hot blocks (`LBB85_120`, `LBB85_123`) rather than
   pushed to the pure tail — LLVM's MachineBlockPlacement isn't
   perfect here, but most panic sites are correctly tailed.

## Classic decoder (`DecodeState<...>::advance`)

Variant measured: `<...>::advance` index 0, 1315 lines of asm. Also
consistent across the 8 classic monomorphizations (2 table types ×
2 bit orders × 2 yield modes).

### What's good

1. **`CodeBuffer` trait methods are auto-inlined.** Despite having
   **zero explicit inline annotations**, `peek_bits`, `consume_bits`,
   and `refill_bits` produce no symbol references in the asm —
   rustc's inliner correctly picked them up based on size heuristics.

2. **`Table::reconstruct`, `Table::derive`, `Table::derive_burst`
   all inline** into the classic advance body. Zero symbol refs.
   The generic `Tab: DecodeTable` monomorphization gives rustc a
   concrete type per mono, which exposes the method bodies for
   cross-function inlining.

### What's missing

3. **`LsbBuffer::next_symbol` stays out-of-line as a real call.**
   One call site in classic's `advance()`, at line 89. This is OK:
   `next_symbol` is only called once per `advance()` entry (to read
   the very first code before the burst loop takes over), not per
   code. Adding `#[inline]` to it wouldn't meaningfully help.

4. **14 panic sites in classic advance.** More than double the
   streaming count (6). The classic decoder has more bounds-checked
   indexing — specifically around the burst slice-splitting and the
   cScSc `fill_reconstruct` / `fill_cscsc` paths. Same root cause as
   streaming's 6 sites; same fix options.

5. **Classic has no `streaming_cold_marker` equivalent.** The classic
   advance has exactly the same cold-branch shape as streaming —
   CLEAR, END, invalid, spill, error paths — but no layout hints.
   Its hot blocks at the top of the function aren't as tightly
   packed as they could be, and the 14 panic sites aren't
   guaranteed to be tailed. This is the **most actionable finding
   from the audit**: apply the cold-marker pattern to classic too.

   The expected benefit is small but measurable: we saw ~3-6% on
   gif/flat-ui when we added cold markers to streaming, and the
   classic decoder's flat-ui hot loop has the same sensitivity.
   Same cost structure — an empty function call that gets optimized
   away.

## Hot code outside the DecodeTable trait

Auditing against the "is there hot code we haven't touched" question:

| Location | Inline state | Hot? | Action |
|----------|--------------|------|--------|
| `DecodeTable::{reconstruct,derive,derive_burst,first_of,depth}` | via trait mono, auto-inlined | yes, verified by asm | none |
| `CodeBuffer::peek_bits` / `consume_bits` / `refill_bits` | no attrs, auto-inlined | yes, verified by asm | none |
| `CodeBuffer::next_symbol` | no attrs, NOT inlined | no (1 call per advance, not per code) | none |
| `CodeBuffer::bump_code_size` | no attrs | no (once per size boundary) | none |
| `StreamingBitPacking::{refill_fast8,extract,peek_code,...}` | `#[inline(always)]` on every method | yes | none |
| `DecodeStateStreaming::derive` | `#[inline(always)]` | yes, inlined 7× | none |
| `DecodeStateStreaming::bump_width_slow` | `#[cold] #[inline(never)]` | no (once per width boundary) | none |
| `DecodeStateStreaming::reconstruct_streaming_into` | `#[inline(always)]` | yes, verified inlined | none |
| `streaming_cold_marker` | `#[cold] #[inline(never)]` | n/a — layout hint | none |
| `DecodeStateStreaming::first_of` | `#[inline(always)]` | only in KwKwK spill path (rare) | could demote to `#[inline]` |
| `Buffer::{fill_cscsc,fill_reconstruct,consume}` | no attrs | only in classic cScSc spill / partial-write | **could add `#[cold]` wrapper or internal cold markers** |
| `DerivationBase::derive` | no attrs, auto-inlined | yes (trivially small) | none |
| `LsbBuffer::refill_bits` / `MsbBuffer::refill_bits` | no attrs, auto-inlined in classic | yes | none |
| Classic advance dispatch: CLEAR / END / spill / invalid | no cold hint | no (but bloats fn) | **apply `streaming_cold_marker` pattern** |

## Prioritized next actions

1. **Apply cold markers to Classic advance** (high-value, low-risk).
   Same zero-cost empty-function call pattern. Would push Classic's
   14 panic sites and the CLEAR/END/spill branches to the tail,
   tightening the hot LITERAL/COPY/burst region. Expected: ~3% on
   high-throughput classic workloads, same as streaming saw.

2. **Audit `Buffer::fill_cscsc` call site** for cold-marker placement.
   `fill_cscsc` is the classic decoder's `last_decoded` equivalent
   for the cScSc case. It runs only when the decoded value doesn't
   fit in the caller's `out` slice — definitionally rare for anyone
   calling `decode()` on a large vector. A cold hint here is free.

3. **Leave derive() alone.** The 7 inline copies are the right call.
   Out-of-lining would add per-code call overhead that swamps the
   layout win.

4. **Don't chase the panic sites.** `#![forbid(unsafe_code)]` leaves
   only restructuring as an option, and the panic blocks are mostly
   already in the tail. Low effort-to-reward ratio.

5. **`DecodeStateStreaming::first_of` could be `#[inline]` instead
   of `#[inline(always)]`.** It's only called in the spill path
   (cold). Over-forcing its inlining might be adding bytes to a
   cold-ish path inside the hot function. Minor cleanup, no proven
   benefit, but worth a quick demotion.
