extern crate criterion;
extern crate weezl;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::fs;
use weezl::{
    decode::{Configuration, TableStrategy},
    BitOrder, LzwStatus,
};

fn run_once(data: &[u8], outbuf: &mut [u8], strategy: TableStrategy) -> usize {
    let mut decoder = Configuration::new(BitOrder::Msb, 8)
        .with_table_strategy(strategy)
        .build();
    let mut written = 0;
    let mut data = data;
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
            panic!("Need to make progress");
        }
    }
    written
}

pub fn criterion_benchmark(c: &mut Criterion, file: &str) {
    let data = fs::read(file).expect("Benchmark input not found");
    let mut outbuf = vec![0; 1 << 26]; // 64MB, what wuff uses..

    // Warm-up call to establish throughput denominator (decoded size).
    let decoded_size = run_once(&data, &mut outbuf, TableStrategy::Classic) as u64;

    let mut group = c.benchmark_group("msb-8");
    group.throughput(Throughput::Bytes(decoded_size));

    for &strat in &[TableStrategy::Classic, TableStrategy::Chunked] {
        let tag = match strat {
            TableStrategy::Classic => "classic",
            TableStrategy::Chunked => "chunked",
        };
        let id = BenchmarkId::new(format!("{}/{}", tag, file), data.len());
        group.bench_with_input(id, &data, |b, data| {
            b.iter(|| {
                run_once(data, outbuf.as_mut_slice(), strat);
            })
        });
    }
}

pub fn bench_toml(c: &mut Criterion) {
    criterion_benchmark(c, "benches/Cargo-8-msb.lzw");
}

pub fn bench_binary(c: &mut Criterion) {
    criterion_benchmark(c, "benches/binary-8-msb.lzw");
}

pub fn bench_lib(c: &mut Criterion) {
    criterion_benchmark(c, "benches/lib-8-msb.lzw");
}

criterion_group!(benches, bench_toml, bench_binary, bench_lib);
criterion_main!(benches);
