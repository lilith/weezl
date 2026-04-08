//! Shared helpers for the wuffs-parity benchmarks:
//! - FFI wrapper around the vendored wuffs_lzw decoder
//! - Synthetic-data generators (must stay in sync with weezl's dump-synth example)
//! - Decode drivers for weezl Classic and Chunked strategies

use std::os::raw::c_void;

unsafe extern "C" {
    fn weezl_wuffs_bench__decode(
        in_ptr: *const u8,
        in_len: usize,
        out_ptr: *mut u8,
        out_cap: usize,
        literal_width_plus_one: u32,
    ) -> usize;
}

#[allow(dead_code)]
fn _ffi_sanity() {
    // Force the linker to retain the symbol.
    let _ = weezl_wuffs_bench__decode as *const c_void;
}

/// One-shot decode via wuffs_lzw (LSB-only, GIF-style).
/// `literal_width_plus_one` = 9 for 8-bit literals.
/// Returns bytes written, or panics on error.
pub fn decode_wuffs(encoded: &[u8], out: &mut [u8], literal_width: u8) -> usize {
    let plus_one = literal_width as u32 + 1;
    let written = unsafe {
        weezl_wuffs_bench__decode(
            encoded.as_ptr(),
            encoded.len(),
            out.as_mut_ptr(),
            out.len(),
            plus_one,
        )
    };
    if written == usize::MAX {
        panic!("wuffs_lzw decoder error");
    }
    written
}

// --------------------------------------------------------------------------
// weezl decode drivers (shared between strategies)
// --------------------------------------------------------------------------

use weezl::{
    decode::{Configuration, TableStrategy},
    BitOrder, LzwStatus,
};

pub fn decode_weezl(encoded: &[u8], out: &mut [u8], strategy: TableStrategy) -> usize {
    let mut decoder = Configuration::new(BitOrder::Lsb, 8)
        .with_table_strategy(strategy)
        .build();
    let mut written = 0;
    let mut inp = encoded;
    let mut cursor = out;
    loop {
        let r = decoder.decode_bytes(inp, cursor);
        inp = &inp[r.consumed_in..];
        let n = r.consumed_out;
        written += n;
        cursor = &mut std::mem::take(&mut cursor)[n..];
        match r.status.expect("weezl decode error") {
            LzwStatus::Done => return written,
            LzwStatus::NoProgress => return written,
            LzwStatus::Ok => {
                if inp.is_empty() && cursor.is_empty() {
                    return written;
                }
            }
        }
    }
}

// --------------------------------------------------------------------------
// Synthetic data (must match examples/dump-synth.rs byte-for-byte so that
// weezl-internal benches and the external wuffs C bench hit identical inputs)
// --------------------------------------------------------------------------

pub fn make_solid(n: usize) -> Vec<u8> {
    vec![0u8; n]
}

pub fn make_run_length(n: usize) -> Vec<u8> {
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

pub fn make_palette16(n: usize) -> Vec<u8> {
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

pub fn make_random(n: usize) -> Vec<u8> {
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

/// LSB-8 encode via weezl so wuffs can decode it.
pub fn encode_lsb8(data: &[u8]) -> Vec<u8> {
    weezl::encode::Encoder::new(BitOrder::Lsb, 8)
        .encode(data)
        .unwrap()
}

pub struct Input {
    pub name: &'static str,
    pub raw: Vec<u8>,
    pub encoded: Vec<u8>,
    pub ratio: f64,
}

impl Input {
    pub fn build(name: &'static str, raw: Vec<u8>) -> Self {
        let encoded = encode_lsb8(&raw);
        let ratio = raw.len() as f64 / encoded.len() as f64;
        Input {
            name,
            raw,
            encoded,
            ratio,
        }
    }
}

/// Standard corpus of inputs covering the compression-ratio spectrum.
pub fn standard_corpus() -> Vec<Input> {
    const N_LARGE: usize = 4 * 1024 * 1024;
    const N_SMALL: usize = 64 * 1024;
    vec![
        Input::build("solid-4M", make_solid(N_LARGE)),
        Input::build("rle-4M", make_run_length(N_LARGE)),
        Input::build("pal16-4M", make_palette16(N_LARGE)),
        Input::build("rand-4M", make_random(N_LARGE)),
        Input::build("solid-64k", make_solid(N_SMALL)),
        Input::build("pal16-64k", make_palette16(N_SMALL)),
        Input::build("rand-64k", make_random(N_SMALL)),
    ]
}

// --------------------------------------------------------------------------
// Cross-check: run all three decoders on every input and assert byte equality.
// --------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn check(input: &Input) {
        let mut out_classic = vec![0u8; input.raw.len() + 64];
        let mut out_chunked = vec![0u8; input.raw.len() + 64];
        let mut out_tight = vec![0u8; input.raw.len() + 64];
        let mut out_wuffs = vec![0u8; input.raw.len() + 64];

        let n1 = decode_weezl(&input.encoded, &mut out_classic, TableStrategy::Classic);
        let n2 = decode_weezl(&input.encoded, &mut out_chunked, TableStrategy::Chunked);
        let n4 = decode_weezl(&input.encoded, &mut out_tight, TableStrategy::Tight);
        let n3 = decode_wuffs(&input.encoded, &mut out_wuffs, 8);

        assert_eq!(n1, input.raw.len(), "{} classic size", input.name);
        assert_eq!(n2, input.raw.len(), "{} chunked size", input.name);
        assert_eq!(n4, input.raw.len(), "{} tight size", input.name);
        assert_eq!(n3, input.raw.len(), "{} wuffs size", input.name);

        assert_eq!(&out_classic[..n1], &input.raw[..], "{} classic bytes", input.name);
        assert_eq!(&out_chunked[..n2], &input.raw[..], "{} chunked bytes", input.name);
        assert_eq!(&out_tight[..n4], &input.raw[..], "{} tight bytes", input.name);
        assert_eq!(&out_wuffs[..n3], &input.raw[..], "{} wuffs bytes", input.name);
    }

    #[test]
    fn all_three_agree_on_corpus() {
        for input in standard_corpus() {
            check(&input);
        }
    }
}
