//! Isolates the per-decoder-construction cost (Classic vs Chunked)
//! by decoding minimal inputs, so we can separate init from steady-state.

extern crate criterion;
extern crate weezl;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use weezl::{
    decode::{Configuration, TableStrategy},
    encode::Encoder,
    BitOrder, LzwStatus,
};

fn decode_trivial(encoded: &[u8], outbuf: &mut [u8], strategy: TableStrategy) {
    let mut decoder = Configuration::new(BitOrder::Msb, 8)
        .with_table_strategy(strategy)
        .build();
    loop {
        let r = decoder.decode_bytes(encoded, outbuf);
        black_box(&outbuf[..r.consumed_out]);
        match r.status.unwrap() {
            LzwStatus::Done => break,
            _ => break,
        }
    }
}

pub fn bench_init(c: &mut Criterion) {
    // One-byte input — all the cost is in constructor + init_tables + single code walk.
    let encoded = Encoder::new(BitOrder::Msb, 8).encode(&[42u8]).unwrap();
    let mut outbuf = [0u8; 16];

    c.bench_function("init/classic/1byte", |b| {
        b.iter(|| decode_trivial(&encoded, &mut outbuf, TableStrategy::Classic))
    });
    c.bench_function("init/chunked/1byte", |b| {
        b.iter(|| decode_trivial(&encoded, &mut outbuf, TableStrategy::Chunked))
    });

    // 16-byte input — still init-dominated.
    let encoded = Encoder::new(BitOrder::Msb, 8).encode(&[0u8; 16]).unwrap();
    let mut outbuf = [0u8; 64];
    c.bench_function("init/classic/16byte", |b| {
        b.iter(|| decode_trivial(&encoded, &mut outbuf, TableStrategy::Classic))
    });
    c.bench_function("init/chunked/16byte", |b| {
        b.iter(|| decode_trivial(&encoded, &mut outbuf, TableStrategy::Chunked))
    });
}

criterion_group!(benches, bench_init);
criterion_main!(benches);
