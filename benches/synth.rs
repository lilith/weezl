//! Synthetic-data bench covering the full compression-ratio spectrum,
//! so we can see both the speedup and regression cases for TableStrategy::Chunked.

extern crate criterion;
extern crate weezl;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use weezl::{
    decode::{Configuration, TableStrategy},
    encode::Encoder,
    BitOrder, LzwStatus,
};

fn make_solid(n: usize) -> Vec<u8> {
    vec![0u8; n]
}

fn make_run_length(n: usize) -> Vec<u8> {
    // Alternating runs of 4 bytes each — typical GIF palette screenshot pattern.
    let mut out = Vec::with_capacity(n);
    let palette = [0u8, 1, 2, 3, 4, 5, 6, 7];
    let mut i = 0;
    while out.len() < n {
        let p = palette[(i / 64) % palette.len()];
        out.push(p);
        i += 1;
    }
    out
}

fn make_palette16(n: usize) -> Vec<u8> {
    // Dithered-palette-like — 16 colors with some spatial locality.
    // 32-byte cells each picking from 16 indices.
    let mut out = Vec::with_capacity(n);
    let mut state = 0x9E3779B1u32;
    while out.len() < n {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        let base = ((state >> 24) & 0x0F) as u8;
        let run_len = 16 + ((state >> 16) & 0x1F) as usize;
        for _ in 0..run_len {
            if out.len() >= n {
                break;
            }
            out.push(base);
        }
    }
    out
}

fn make_random(n: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n);
    let mut state = 0x1234_5678_9ABC_DEFu64;
    for _ in 0..n {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        out.push((state >> 56) as u8);
    }
    out
}

// LSB matches wuffs' bit ordering (GIF style). We want apples-to-apples.
const ORDER: BitOrder = BitOrder::Lsb;

fn encode(data: &[u8]) -> Vec<u8> {
    Encoder::new(ORDER, 8).encode(data).unwrap()
}

fn run_once(encoded: &[u8], outbuf: &mut [u8], strategy: TableStrategy) -> usize {
    let mut decoder = Configuration::new(ORDER, 8)
        .with_table_strategy(strategy)
        .build();
    let mut written = 0;
    let mut data = encoded;
    loop {
        let result = decoder.decode_bytes(data, outbuf);
        let done = result.status.expect("Error");
        data = &data[result.consumed_in..];
        written += result.consumed_out;
        black_box(&outbuf[..result.consumed_out]);
        if let LzwStatus::Done = done {
            break;
        }
        if let LzwStatus::NoProgress = done {
            break;
        }
    }
    written
}

fn bench_one(
    c: &mut Criterion,
    tag: &'static str,
    decoded: Vec<u8>,
) {
    let encoded = encode(&decoded);
    let ratio = decoded.len() as f64 / encoded.len() as f64;
    let mut outbuf = vec![0u8; decoded.len() + 1024];

    let mut group = c.benchmark_group("synth");
    group.throughput(Throughput::Bytes(decoded.len() as u64));

    for &strat in &[TableStrategy::Classic, TableStrategy::Chunked] {
        let name = match strat {
            TableStrategy::Classic => "classic",
            TableStrategy::Chunked => "chunked",
        };
        let label = format!("{}/{}/r{:.1}", name, tag, ratio);
        let id = BenchmarkId::new(label, encoded.len());
        group.bench_with_input(id, &encoded, |b, encoded| {
            b.iter(|| {
                run_once(encoded, outbuf.as_mut_slice(), strat);
            })
        });
    }
    group.finish();
}

pub fn bench_all(c: &mut Criterion) {
    // Large inputs so per-run init cost is amortized.
    const N_LARGE: usize = 4 * 1024 * 1024; // 4 MiB decoded
    const N_SMALL: usize = 64 * 1024; // 64 KiB — init cost more visible

    bench_one(c, "solid-4M", make_solid(N_LARGE));
    bench_one(c, "rle-4M", make_run_length(N_LARGE));
    bench_one(c, "pal16-4M", make_palette16(N_LARGE));
    bench_one(c, "rand-4M", make_random(N_LARGE));

    bench_one(c, "solid-64k", make_solid(N_SMALL));
    bench_one(c, "pal16-64k", make_palette16(N_SMALL));
    bench_one(c, "rand-64k", make_random(N_SMALL));
}

criterion_group!(benches, bench_all);
criterion_main!(benches);
