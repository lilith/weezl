//! Pure allocation cost of Decoder creation, isolated from decoding.
//!
//! Each weezl Decoder allocates several Box<[T; 4096]> arrays in its
//! constructor (2-5 depending on strategy). On Linux with glibc malloc
//! these are fast (~50-200ns per Decoder). On Windows where malloc is
//! notably slower, the per-strip Decoder creation in image-tiff's
//! image.rs:1149 can become a measurable fraction of decode time,
//! especially for small strips.
//!
//! This bench measures Decoder::new + Drop in isolation so we can see
//! the raw per-strategy alloc cost and extrapolate Windows impact.

use weezl::decode::{Configuration, TableStrategy};
use weezl::BitOrder;
use zenbench::prelude::*;

fn bench_alloc(suite: &mut Suite) {
    suite.group("alloc", |g| {
        g.throughput(Throughput::Elements(1));

        g.bench("classic/new+drop", |b| {
            b.iter(|| {
                let dec = Configuration::with_tiff_size_switch(BitOrder::Msb, 8)
                    .with_yield_on_full_buffer(true)
                    .with_table_strategy(TableStrategy::Classic)
                    .build();
                black_box(dec);
            })
        });

        g.bench("chunked/new+drop", |b| {
            b.iter(|| {
                let dec = Configuration::with_tiff_size_switch(BitOrder::Msb, 8)
                    .with_yield_on_full_buffer(true)
                    .with_table_strategy(TableStrategy::Chunked)
                    .build();
                black_box(dec);
            })
        });

        g.bench("streaming/new+drop", |b| {
            b.iter(|| {
                let dec = Configuration::with_tiff_size_switch(BitOrder::Msb, 8)
                    .with_yield_on_full_buffer(true)
                    .with_table_strategy(TableStrategy::Streaming)
                    .build();
                black_box(dec);
            })
        });

        // Sanity: reset() on an existing decoder should be much cheaper
        // than new() because it reuses the same allocations.
        g.bench("classic/reset", |b| {
            b.with_input(|| {
                Configuration::with_tiff_size_switch(BitOrder::Msb, 8)
                    .with_yield_on_full_buffer(true)
                    .with_table_strategy(TableStrategy::Classic)
                    .build()
            })
            .run(|mut dec| {
                dec.reset();
                dec
            })
        });

        g.bench("chunked/reset", |b| {
            b.with_input(|| {
                Configuration::with_tiff_size_switch(BitOrder::Msb, 8)
                    .with_yield_on_full_buffer(true)
                    .with_table_strategy(TableStrategy::Chunked)
                    .build()
            })
            .run(|mut dec| {
                dec.reset();
                dec
            })
        });

        g.bench("streaming/reset", |b| {
            b.with_input(|| {
                Configuration::with_tiff_size_switch(BitOrder::Msb, 8)
                    .with_yield_on_full_buffer(true)
                    .with_table_strategy(TableStrategy::Streaming)
                    .build()
            })
            .run(|mut dec| {
                dec.reset();
                dec
            })
        });
    });
}

zenbench::main!(bench_alloc);
