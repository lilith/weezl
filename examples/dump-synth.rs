//! Dumps the synthetic bench inputs (LSB-encoded LZW) plus their decoded
//! reference output to /tmp/wuffs_bench/, so the C harness can bench the
//! exact same bytes.

use std::fs;
use std::io::Write;
use weezl::{encode::Encoder, BitOrder};

fn make_solid(n: usize) -> Vec<u8> {
    vec![0u8; n]
}

fn make_run_length(n: usize) -> Vec<u8> {
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

fn dump(name: &str, raw: Vec<u8>) {
    let encoded = Encoder::new(BitOrder::Lsb, 8).encode(&raw).unwrap();
    let ratio = raw.len() as f64 / encoded.len() as f64;
    let dir = "/tmp/wuffs_bench/data";
    fs::create_dir_all(dir).unwrap();
    fs::write(format!("{}/{}.lzw", dir, name), &encoded).unwrap();
    fs::write(format!("{}/{}.raw", dir, name), &raw).unwrap();
    println!(
        "{:14} raw={:>10} enc={:>10} ratio={:>8.2}x",
        name,
        raw.len(),
        encoded.len(),
        ratio
    );
}

fn main() {
    const N_LARGE: usize = 4 * 1024 * 1024;
    const N_SMALL: usize = 64 * 1024;
    dump("solid-4M", make_solid(N_LARGE));
    dump("rle-4M", make_run_length(N_LARGE));
    dump("pal16-4M", make_palette16(N_LARGE));
    dump("rand-4M", make_random(N_LARGE));
    dump("solid-64k", make_solid(N_SMALL));
    dump("pal16-64k", make_palette16(N_SMALL));
    dump("rand-64k", make_random(N_SMALL));
    std::io::stdout().flush().ok();
}
