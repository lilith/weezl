//! Tight loop that decodes a given input N times. Designed for `perf stat`.
//! Usage: `profile_one <classic|chunked|wuffs> <input-name> <iters>`
//! where <input-name> is one of: solid-4M, rle-4M, pal16-4M, rand-4M,
//! solid-64k, pal16-64k, rand-64k.

use std::env;
use weezl::decode::TableStrategy;
use wuffs_bench::{decode_weezl, decode_wuffs, standard_corpus};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 {
        eprintln!("usage: profile_one <classic|chunked|wuffs> <input-name> <iters>");
        std::process::exit(1);
    }
    let backend = &args[1];
    let name = &args[2];
    let iters: usize = args[3].parse().expect("bad iter count");

    let corpus = standard_corpus();
    let input = corpus
        .iter()
        .find(|i| i.name == *name)
        .unwrap_or_else(|| panic!("unknown input name: {}", name));

    let mut out = vec![0u8; input.raw.len() + 64];

    // Verify once
    let n = match backend.as_str() {
        "classic" => decode_weezl(&input.encoded, &mut out, TableStrategy::Classic),
        "chunked" => decode_weezl(&input.encoded, &mut out, TableStrategy::Chunked),
        "wuffs" => decode_wuffs(&input.encoded, &mut out, 8),
        other => panic!("unknown backend: {}", other),
    };
    assert_eq!(n, input.raw.len(), "decoded size mismatch");

    // Hot loop
    for _ in 0..iters {
        let n = match backend.as_str() {
            "classic" => decode_weezl(&input.encoded, &mut out, TableStrategy::Classic),
            "chunked" => decode_weezl(&input.encoded, &mut out, TableStrategy::Chunked),
            "wuffs" => decode_wuffs(&input.encoded, &mut out, 8),
            _ => unreachable!(),
        };
        std::hint::black_box(&out[..n]);
    }
}
