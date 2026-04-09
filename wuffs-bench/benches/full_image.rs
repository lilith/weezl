//! Full-image TIFF decode: iterate every strip, matching image-tiff's
//! image.rs:1149 loop. Each strip gets a fresh LZWReader in image-tiff;
//! we mirror that plus a reuse-via-reset variant.
//!
//! Typical image has 5-30 strips. The per-strip Decoder init cost is
//! N times per image (N = strip count), so this is where alloc cost
//! (important on Windows) would show up.

use std::sync::Arc;

use weezl::{
    decode::{Configuration, Decoder, TableStrategy},
    BitOrder, LzwStatus,
};
use wuffs_bench::{
    clic_full_strips, conform_full_strips, qoi_full_strips, sc_full_strips, TiffFullStrips,
};
use zenbench::prelude::*;

fn new_decoder(strategy: TableStrategy) -> Decoder {
    Configuration::with_tiff_size_switch(BitOrder::Msb, 8)
        .with_yield_on_full_buffer(true)
        .with_table_strategy(strategy)
        .build()
}

fn decode_strip(dec: &mut Decoder, encoded: &[u8], out: &mut [u8]) -> usize {
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

/// Decode ALL strips of ALL files, fresh Decoder per strip.
/// This is what image-tiff does today.
fn bench_fresh(
    g: &mut BenchGroup,
    name: String,
    corpus: Arc<Vec<TiffFullStrips>>,
    strategy: TableStrategy,
) {
    let total_bytes: usize = corpus.iter().map(|f| f.total_decompressed).sum();
    g.throughput(Throughput::Bytes(total_bytes as u64));
    g.bench(name, move |b| {
        let corpus = Arc::clone(&corpus);
        // Max strip decompressed size across the corpus, for scratch buf sizing
        let max_strip_rows = corpus
            .iter()
            .map(|f| f.rows_per_strip)
            .max()
            .unwrap_or(256);
        let max_row_bytes = corpus.iter().map(|f| f.row_bytes).max().unwrap_or(8192);
        let out_cap = max_strip_rows * max_row_bytes + 1024;
        let mut out = vec![0u8; out_cap];
        b.iter(move || {
            let mut total = 0usize;
            for file in corpus.iter() {
                for strip in &file.strips {
                    let mut dec = new_decoder(strategy);
                    let n = decode_strip(&mut dec, strip, &mut out);
                    total += n;
                    black_box(&out[..n]);
                }
            }
            total
        })
    });
}

/// Decode ALL strips of ALL files, one persistent Decoder per file
/// (reset between strips within the same file, like image-gif does).
fn bench_reuse_per_file(
    g: &mut BenchGroup,
    name: String,
    corpus: Arc<Vec<TiffFullStrips>>,
    strategy: TableStrategy,
) {
    let total_bytes: usize = corpus.iter().map(|f| f.total_decompressed).sum();
    g.throughput(Throughput::Bytes(total_bytes as u64));
    g.bench(name, move |b| {
        let corpus = Arc::clone(&corpus);
        let max_strip_rows = corpus
            .iter()
            .map(|f| f.rows_per_strip)
            .max()
            .unwrap_or(256);
        let max_row_bytes = corpus.iter().map(|f| f.row_bytes).max().unwrap_or(8192);
        let out_cap = max_strip_rows * max_row_bytes + 1024;
        let mut out = vec![0u8; out_cap];
        b.iter(move || {
            let mut total = 0usize;
            for file in corpus.iter() {
                let mut dec = new_decoder(strategy);
                for (i, strip) in file.strips.iter().enumerate() {
                    if i > 0 {
                        dec.reset();
                    }
                    let n = decode_strip(&mut dec, strip, &mut out);
                    total += n;
                    black_box(&out[..n]);
                }
            }
            total
        })
    });
}

/// Decode everything through ONE global persistent Decoder, reset
/// between strips across files. Maximum reuse.
fn bench_reuse_global(
    g: &mut BenchGroup,
    name: String,
    corpus: Arc<Vec<TiffFullStrips>>,
    strategy: TableStrategy,
) {
    let total_bytes: usize = corpus.iter().map(|f| f.total_decompressed).sum();
    g.throughput(Throughput::Bytes(total_bytes as u64));
    g.bench(name, move |b| {
        let corpus = Arc::clone(&corpus);
        let max_strip_rows = corpus
            .iter()
            .map(|f| f.rows_per_strip)
            .max()
            .unwrap_or(256);
        let max_row_bytes = corpus.iter().map(|f| f.row_bytes).max().unwrap_or(8192);
        let out_cap = max_strip_rows * max_row_bytes + 1024;
        let mut out = vec![0u8; out_cap];
        let mut dec = new_decoder(strategy);
        let mut first = true;
        b.iter(move || {
            let mut total = 0usize;
            for file in corpus.iter() {
                for strip in &file.strips {
                    if !first {
                        dec.reset();
                    }
                    first = false;
                    let n = decode_strip(&mut dec, strip, &mut out);
                    total += n;
                    black_box(&out[..n]);
                }
            }
            total
        })
    });
}

fn bench_corpus_group(suite: &mut Suite, name: &str, corpus: Vec<TiffFullStrips>) {
    if corpus.is_empty() {
        eprintln!("warning: {} corpus empty", name);
        return;
    }
    let corpus_arc = Arc::new(corpus);
    let total_strips: usize = corpus_arc.iter().map(|f| f.strips.len()).sum();
    let files = corpus_arc.len();
    eprintln!(
        "[full_image] {} corpus: {} files, {} strips total",
        name, files, total_strips
    );
    suite.group(format!("fullimg-{}", name), |g| {
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
                Arc::clone(&corpus_arc),
                strategy,
            );
            bench_reuse_per_file(
                g,
                format!("{}-reuse-file", tag),
                Arc::clone(&corpus_arc),
                strategy,
            );
            bench_reuse_global(
                g,
                format!("{}-reuse-global", tag),
                Arc::clone(&corpus_arc),
                strategy,
            );
        }
    });
}

fn bench_all(suite: &mut Suite) {
    bench_corpus_group(suite, "conform", conform_full_strips());
    bench_corpus_group(suite, "clic", clic_full_strips());
    bench_corpus_group(suite, "qoi", qoi_full_strips());
    bench_corpus_group(suite, "sc", sc_full_strips());
}

zenbench::main!(bench_all);
