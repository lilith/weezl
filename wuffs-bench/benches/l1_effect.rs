//! Isolation bench: does weezl's slowdown on rand-4M come from cold-cache
//! output writes? If yes, running on a 4 KiB input/output (L1d-resident)
//! should dramatically close the gap.

use std::sync::Arc;

use weezl::decode::TableStrategy;
use wuffs_bench::{
    decode_weezl, decode_wuffs, encode_lsb8, make_palette16, make_random, Input,
};
use zenbench::prelude::*;

fn small_input(name: &'static str, n: usize, raw: Vec<u8>) -> Input {
    let encoded = encode_lsb8(&raw);
    let ratio = raw.len() as f64 / encoded.len() as f64;
    let _ = n;
    Input {
        name,
        raw,
        encoded,
        ratio,
    }
}

fn bench_input(g: &mut BenchGroup, input: Arc<Input>) {
    g.throughput(Throughput::Bytes(input.raw.len() as u64));
    let out_cap = input.raw.len() + 64;

    let i1 = Arc::clone(&input);
    g.bench("classic", move |b| {
        let input = Arc::clone(&i1);
        let mut out = vec![0u8; out_cap];
        b.iter(move || {
            let n = decode_weezl(&input.encoded, &mut out, TableStrategy::Classic);
            black_box(&out[..n]);
            n
        })
    });
    let i2 = Arc::clone(&input);
    g.bench("chunked", move |b| {
        let input = Arc::clone(&i2);
        let mut out = vec![0u8; out_cap];
        b.iter(move || {
            let n = decode_weezl(&input.encoded, &mut out, TableStrategy::Chunked);
            black_box(&out[..n]);
            n
        })
    });
    let i3 = Arc::clone(&input);
    g.bench("wuffs", move |b| {
        let input = Arc::clone(&i3);
        let mut out = vec![0u8; out_cap];
        b.iter(move || {
            let n = decode_wuffs(&input.encoded, &mut out, 8);
            black_box(&out[..n]);
            n
        })
    });
}

fn bench_cache_footprint(suite: &mut Suite) {
    // 4 KiB, 16 KiB, 64 KiB, 1 MiB: spans L1d → L3.
    // If weezl's gap is cache-driven, we'll see it narrow on small sizes.
    let sizes = [(4 * 1024, "4k"), (16 * 1024, "16k"), (64 * 1024, "64k"), (1024 * 1024, "1M")];
    for (sz, tag) in sizes {
        let raw = make_random(sz);
        let input = Arc::new(small_input(
            Box::leak(format!("rand-{tag}").into_boxed_str()),
            sz,
            raw,
        ));
        suite.group(format!("rand-{}", tag), |g| bench_input(g, Arc::clone(&input)));
    }
    for (sz, tag) in sizes {
        let raw = make_palette16(sz);
        let input = Arc::new(small_input(
            Box::leak(format!("pal16-{tag}").into_boxed_str()),
            sz,
            raw,
        ));
        suite.group(format!("pal16-{}", tag), |g| bench_input(g, Arc::clone(&input)));
    }
}

zenbench::main!(bench_cache_footprint);
