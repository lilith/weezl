//! Real MSB + TIFF early-change bench using actual TIFF files.
//!
//! Decodes the LARGEST strip from each TIFF file in the corpus through
//! Classic, Chunked, and Streaming decoders, all configured as image-tiff
//! configures them: `Configuration::with_tiff_size_switch(Msb, 8)
//! .with_table_strategy(...)`.
//!
//! wuffs_lzw is LSB-only so it's excluded from this MSB comparison.

use std::sync::Arc;

use weezl::{
    decode::{Configuration, TableStrategy},
    BitOrder, LzwStatus,
};
use wuffs_bench::{
    clic_as_lzw_tiff_nopred, clic_as_lzw_tiff_pred, qoi_as_lzw_tiff, sc_as_lzw_tiff,
    tiff_conformance_lzw, TiffStrips,
};
use zenbench::prelude::*;

fn decode_tiff(encoded: &[u8], out: &mut [u8], strategy: TableStrategy) -> usize {
    let mut dec = Configuration::with_tiff_size_switch(BitOrder::Msb, 8)
        .with_table_strategy(strategy)
        .build();
    let mut inp = encoded;
    let mut cursor = out;
    let mut written = 0usize;
    loop {
        let r = dec.decode_bytes(inp, cursor);
        inp = &inp[r.consumed_in..];
        let n = r.consumed_out;
        written += n;
        cursor = &mut std::mem::take(&mut cursor)[n..];
        match r.status {
            Ok(LzwStatus::Done) => return written,
            Ok(LzwStatus::NoProgress) => return written,
            Ok(LzwStatus::Ok) => {
                if inp.is_empty() && cursor.is_empty() {
                    return written;
                }
            }
            Err(_) => return written,
        }
    }
}

fn bench_input(g: &mut BenchGroup, input: Arc<TiffStrips>) {
    // Use the expected decompressed size for throughput; cap the out buffer
    // generously because the tag-derived size can be a conservative upper bound.
    let out_cap = input.decompressed_size.max(input.lzw_bytes.len() * 8) + 1024;

    // Determine actual decoded size by running Classic once.
    let mut scratch = vec![0u8; out_cap];
    let actual = decode_tiff(&input.lzw_bytes, &mut scratch, TableStrategy::Classic);
    if actual == 0 {
        eprintln!("warning: {} produced 0 bytes, skipping", input.name);
        return;
    }
    g.throughput(Throughput::Bytes(actual as u64));

    let i1 = Arc::clone(&input);
    g.bench(format!("classic/{}", input.name), move |b| {
        let input = Arc::clone(&i1);
        let mut out = vec![0u8; out_cap];
        b.iter(move || {
            let n = decode_tiff(&input.lzw_bytes, &mut out, TableStrategy::Classic);
            black_box(&out[..n]);
            n
        })
    });

    let i2 = Arc::clone(&input);
    g.bench(format!("chunked/{}", input.name), move |b| {
        let input = Arc::clone(&i2);
        let mut out = vec![0u8; out_cap];
        b.iter(move || {
            let n = decode_tiff(&input.lzw_bytes, &mut out, TableStrategy::Chunked);
            black_box(&out[..n]);
            n
        })
    });

    let i3 = Arc::clone(&input);
    g.bench(format!("streaming/{}", input.name), move |b| {
        let input = Arc::clone(&i3);
        let mut out = vec![0u8; out_cap];
        b.iter(move || {
            let n = decode_tiff(&input.lzw_bytes, &mut out, TableStrategy::Streaming);
            black_box(&out[..n]);
            n
        })
    });
}

fn bench_group_from_corpus(
    suite: &mut Suite,
    corpus: Vec<TiffStrips>,
    corpus_name: &str,
) {
    if corpus.is_empty() {
        eprintln!("warning: {} corpus empty", corpus_name);
        return;
    }
    // Limit to 4 files per corpus to keep bench runtime reasonable.
    for input in corpus.into_iter().take(4).map(Arc::new) {
        let group_name = format!(
            "tiff/{}-{}-s{}",
            corpus_name,
            input.name,
            input.lzw_bytes.len()
        );
        suite.group(group_name, |g| bench_input(g, Arc::clone(&input)));
    }
}

fn bench_tiff(suite: &mut Suite) {
    bench_group_from_corpus(suite, tiff_conformance_lzw(), "conform");
    bench_group_from_corpus(suite, qoi_as_lzw_tiff(), "qoi");
    bench_group_from_corpus(suite, sc_as_lzw_tiff(), "sc");
    bench_group_from_corpus(suite, clic_as_lzw_tiff_pred(), "clic-p");
    bench_group_from_corpus(suite, clic_as_lzw_tiff_nopred(), "clic-r");
}

zenbench::main!(bench_tiff);
