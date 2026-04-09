//! Does reuse-via-reset() change perf compared to new-decoder-per-strip?
//!
//! image-tiff's current usage: one new LZWReader (and therefore one new
//! weezl::decode::Decoder) per strip. For large images with many strips,
//! this pays the per-Decoder init cost on every strip.
//!
//! Alternative pattern: keep a single Decoder alive for the whole image,
//! call .reset() between strips. This is what image-gif does for
//! animation frames and it saves the alloc/init work.
//!
//! Bench both patterns on real TIFF strips and see if reuse matters.
//! If it does: image-tiff should consider changing stream.rs:143 to
//! take a persistent Decoder from a parent cache.

use std::sync::Arc;

use weezl::{
    decode::{Configuration, Decoder, TableStrategy},
    BitOrder, LzwStatus,
};
use wuffs_bench::{
    clic_as_lzw_tiff_pred, qoi_as_lzw_tiff, sc_as_lzw_tiff, tiff_conformance_lzw, TiffStrips,
};
use zenbench::prelude::*;

// Always configure MSB + TIFF early-change + yield_on_full, matching
// image-tiff's LZWReader::new at stream.rs:143.
fn new_decoder(strategy: TableStrategy) -> Decoder {
    Configuration::with_tiff_size_switch(BitOrder::Msb, 8)
        .with_yield_on_full_buffer(true)
        .with_table_strategy(strategy)
        .build()
}

fn decode_once(dec: &mut Decoder, encoded: &[u8], out: &mut [u8]) -> usize {
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
            Ok(LzwStatus::Done) | Ok(LzwStatus::NoProgress) => return written,
            Ok(LzwStatus::Ok) => {
                if inp.is_empty() && cursor.is_empty() {
                    return written;
                }
            }
            Err(_) => return written,
        }
    }
}

/// Bench the "new decoder per strip" pattern (current image-tiff behavior).
fn bench_fresh(
    g: &mut BenchGroup,
    name: String,
    inputs: Arc<Vec<TiffStrips>>,
    strategy: TableStrategy,
) {
    let total_bytes: usize = inputs.iter().map(|i| i.decompressed_size).sum();
    g.throughput(Throughput::Bytes(total_bytes as u64));
    g.bench(name, move |b| {
        let inputs = Arc::clone(&inputs);
        let out_cap = inputs
            .iter()
            .map(|i| i.decompressed_size.max(i.lzw_bytes.len() * 8))
            .max()
            .unwrap_or(4096)
            + 1024;
        let mut out = vec![0u8; out_cap];
        b.iter(move || {
            let mut total = 0usize;
            for input in inputs.iter() {
                let mut dec = new_decoder(strategy);
                let n = decode_once(&mut dec, &input.lzw_bytes, &mut out);
                total += n;
                black_box(&out[..n]);
            }
            total
        })
    });
}

/// Bench the "persistent decoder + reset per strip" pattern (what
/// image-tiff COULD do if stream.rs:143 held a long-lived Decoder).
fn bench_reuse(
    g: &mut BenchGroup,
    name: String,
    inputs: Arc<Vec<TiffStrips>>,
    strategy: TableStrategy,
) {
    let total_bytes: usize = inputs.iter().map(|i| i.decompressed_size).sum();
    g.throughput(Throughput::Bytes(total_bytes as u64));
    g.bench(name, move |b| {
        let inputs = Arc::clone(&inputs);
        let out_cap = inputs
            .iter()
            .map(|i| i.decompressed_size.max(i.lzw_bytes.len() * 8))
            .max()
            .unwrap_or(4096)
            + 1024;
        let mut out = vec![0u8; out_cap];
        let mut dec = new_decoder(strategy);
        b.iter(move || {
            let mut total = 0usize;
            for input in inputs.iter() {
                dec.reset();
                let n = decode_once(&mut dec, &input.lzw_bytes, &mut out);
                total += n;
                black_box(&out[..n]);
            }
            total
        })
    });
}

fn bench_pattern(suite: &mut Suite, corpus_name: &str, inputs: Vec<TiffStrips>) {
    if inputs.is_empty() {
        return;
    }
    // Drop tiny strips and the top few (to keep bench time bounded).
    let inputs: Vec<TiffStrips> = inputs
        .into_iter()
        .filter(|i| i.lzw_bytes.len() >= 4096)
        .take(6)
        .collect();
    if inputs.is_empty() {
        return;
    }
    let inputs = Arc::new(inputs);

    suite.group(format!("reuse-{}", corpus_name), |g| {
        for strategy in [
            TableStrategy::Classic,
            TableStrategy::Chunked,
            TableStrategy::Tight,
        ] {
            let tag = match strategy {
                TableStrategy::Classic => "cls",
                TableStrategy::Chunked => "chk",
                TableStrategy::Tight => "tgt",
                _ => "?",
            };
            bench_fresh(
                g,
                format!("{}-fresh", tag),
                Arc::clone(&inputs),
                strategy,
            );
            bench_reuse(
                g,
                format!("{}-reuse", tag),
                Arc::clone(&inputs),
                strategy,
            );
        }
    });
}

fn bench_all(suite: &mut Suite) {
    bench_pattern(suite, "conform", tiff_conformance_lzw());
    bench_pattern(suite, "qoi", qoi_as_lzw_tiff());
    bench_pattern(suite, "sc", sc_as_lzw_tiff());
    bench_pattern(suite, "clic", clic_as_lzw_tiff_pred());
}

zenbench::main!(bench_all);
