//! Three-way bench: weezl Classic, weezl Chunked, wuffs_lzw (via FFI).
//!
//! Run: `cargo bench --bench wuffs_parity`
//! Save baseline: `cargo bench --bench wuffs_parity -- --save-baseline=main`
//! Compare vs baseline: `cargo bench --bench wuffs_parity -- --baseline=main`

use std::sync::Arc;

use weezl::decode::TableStrategy;
use wuffs_bench::{
    decode_weezl, decode_wuffs, gb82_sc_corpus, qoi_screenshot_corpus, standard_corpus, Input,
};
use zenbench::prelude::*;

fn bench_input(g: &mut BenchGroup, input: Arc<Input>) {
    g.throughput(Throughput::Bytes(input.raw.len() as u64));

    // Pre-allocate one output buffer per bench closure and reuse it across
    // iterations via Arc<Mutex<Vec<u8>>>. Simpler: allocate once inside the
    // iter closure scope — each closure gets its own captured Vec.
    let out_cap = input.raw.len() + 64;

    let i1 = Arc::clone(&input);
    g.bench(format!("classic/{}", input.name), move |b| {
        let input = Arc::clone(&i1);
        let mut out = vec![0u8; out_cap];
        b.iter(move || {
            let n = decode_weezl(&input.encoded, &mut out, TableStrategy::Classic);
            black_box(&out[..n]);
            n
        })
    });

    let i2 = Arc::clone(&input);
    g.bench(format!("chunked/{}", input.name), move |b| {
        let input = Arc::clone(&i2);
        let mut out = vec![0u8; out_cap];
        b.iter(move || {
            let n = decode_weezl(&input.encoded, &mut out, TableStrategy::Chunked);
            black_box(&out[..n]);
            n
        })
    });

    let i4 = Arc::clone(&input);
    g.bench(format!("streaming/{}", input.name), move |b| {
        let input = Arc::clone(&i4);
        let mut out = vec![0u8; out_cap];
        b.iter(move || {
            let n = decode_weezl(&input.encoded, &mut out, TableStrategy::Streaming);
            black_box(&out[..n]);
            n
        })
    });

    let i3 = Arc::clone(&input);
    g.bench(format!("wuffs/{}", input.name), move |b| {
        let input = Arc::clone(&i3);
        let mut out = vec![0u8; out_cap];
        b.iter(move || {
            let n = decode_wuffs(&input.encoded, &mut out, 8);
            black_box(&out[..n]);
            n
        })
    });
}

fn bench_corpus(suite: &mut Suite) {
    let corpus: Vec<Arc<Input>> = standard_corpus().into_iter().map(Arc::new).collect();
    for input in corpus {
        let group_name = format!("{}-r{:.1}", input.name, input.ratio);
        suite.group(group_name, |g| bench_input(g, Arc::clone(&input)));
    }
}

fn bench_qoi(suite: &mut Suite) {
    let corpus: Vec<Arc<Input>> = qoi_screenshot_corpus().into_iter().map(Arc::new).collect();
    if corpus.is_empty() {
        eprintln!("warning: qoi corpus empty (expected at /home/lilith/work/codec-corpus/qoi-benchmark/screenshot_web)");
        return;
    }
    for input in corpus {
        let group_name = format!("{}-r{:.1}", input.name, input.ratio);
        suite.group(group_name, |g| bench_input(g, Arc::clone(&input)));
    }
}

fn bench_sc(suite: &mut Suite) {
    let corpus: Vec<Arc<Input>> = gb82_sc_corpus().into_iter().map(Arc::new).collect();
    if corpus.is_empty() {
        eprintln!("warning: sc corpus empty (expected at /home/lilith/work/codec-corpus/gb82-sc)");
        return;
    }
    for input in corpus {
        let group_name = format!("{}-r{:.1}", input.name, input.ratio);
        suite.group(group_name, |g| bench_input(g, Arc::clone(&input)));
    }
}

zenbench::main!(bench_corpus, bench_qoi, bench_sc);
