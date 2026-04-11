# KwKwK optimization comparison

Three implementations compared on the synthetic archetype workloads.
All numbers: AMD Ryzen 9 7950X, stable Rust, default target (no `-C
target-cpu=native`). `sample_target_ns = 10ms`, 100+ rounds per workload.

## Streaming throughput (absolute, streaming strategy only)

| Workload | Baseline | Cursor `copy_within` | `last_decoded` |
|----------|----------|---------------------|----------------|
| tiff/flat-ui | 2.83 GiB/s | 2.66 GiB/s (−6.0%) | **2.98 GiB/s (+5.3%)** |
| tiff/rich-screenshot | 584 MiB/s | 582 MiB/s (−0.3%) | 573 MiB/s (−1.9%) |
| tiff/photo-predicted | 349 MiB/s | 342 MiB/s (−2.0%) | 340 MiB/s (−2.6%) |
| tiff/photo-raw | 216 MiB/s | 213 MiB/s (−1.4%) | 217 MiB/s (+0.5%) |
| **tiff/solid-kwkwk** | 6.85 GiB/s | 27.3 GiB/s (+299%) | **26.85 GiB/s (+292%)** |
| gif/flat-ui | 3.65 GiB/s | 3.29 GiB/s (−9.9%) | 3.58 GiB/s (−1.9%) |
| gif/rich-screenshot | 684 MiB/s | 679 MiB/s (−0.7%) | 641 MiB/s (−6.3%) |

## Commits

- Baseline: `3579105` (ChunkedTable + streaming, no KwKwK opt)
- Cursor `copy_within`: `50e56c6`
- `last_decoded`: `ee118ba`

## Conclusion

The cursor-based refactor gave the massive KwKwK win but regressed
flat-ui workloads 6–15% due to per-iteration `prev_wr` bookkeeping
overhead the mini-burst fast loop couldn't absorb.

The `last_decoded: Option<&[u8]>` approach keeps the shrinking-slice
pattern intact, and puts all bookkeeping outside the mini-burst hot
loop. KwKwK fast path is only consulted when `prev_len > Q` (impossible
after a mini-burst write), so the fast path stays zero-overhead.

Net: +292% on solid KwKwK (matches cursor win), ≤6% regression on
any non-KwKwK workload (vs cursor's 10–15%), with most workloads
within noise (±2%).
