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
    decode_weezl_with_order(encoded, out, strategy, BitOrder::Lsb, false)
}

pub fn decode_weezl_with_order(
    encoded: &[u8],
    out: &mut [u8],
    strategy: TableStrategy,
    order: BitOrder,
    tiff: bool,
) -> usize {
    let mut decoder = if tiff {
        Configuration::with_tiff_size_switch(order, 8)
            .with_table_strategy(strategy)
            .build()
    } else {
        Configuration::new(order, 8)
            .with_table_strategy(strategy)
            .build()
    };
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
// Real-world corpus loaders.
//
// QOI benchmark screenshots: PNGs from phoboslab's qoi-benchmark suite.
// gb82-sc (gb82 screen-captures): PNG desktop/mobile screenshots from
// the codec-corpus repo. Both are loaded as raw RGB byte streams and
// LZW-encoded with literal_width=8.
// --------------------------------------------------------------------------

use std::path::Path;

fn try_load_png_rgb(path: &Path) -> Option<Vec<u8>> {
    let file = std::fs::File::open(path).ok()?;
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).ok()?;
    buf.truncate(info.buffer_size());
    Some(buf)
}

/// Take a corpus directory of PNGs and build LZW-encoded Inputs.
/// Skips files that fail to decode or are too small (< 16 KiB raw).
/// Limits to `max_files` for bench runtime.
pub fn load_png_corpus(dir: &str, tag_prefix: &'static str, max_files: usize) -> Vec<Input> {
    let mut out = Vec::new();
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return out,
    };
    let mut paths: Vec<_> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("png"))
        .collect();
    paths.sort();
    for path in paths.into_iter().take(max_files) {
        let raw = match try_load_png_rgb(&path) {
            Some(r) if r.len() >= 16 * 1024 => r,
            _ => continue,
        };
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();
        // Leak the string so we can use &'static str. Fine for bench code.
        let static_name: &'static str = Box::leak(format!("{}/{}", tag_prefix, name).into_boxed_str());
        out.push(Input::build(static_name, raw));
    }
    out
}

/// QOI screenshot_web corpus — 14 screenshots of popular websites as PNG.
/// Typical size: 500KB - 5MB raw RGB.
pub fn qoi_screenshot_corpus() -> Vec<Input> {
    load_png_corpus(
        "/home/lilith/work/codec-corpus/qoi-benchmark/screenshot_web",
        "qoi",
        6,
    )
}

/// gb82-sc corpus — 10 desktop/mobile screenshots (retina, native dark, etc.)
pub fn gb82_sc_corpus() -> Vec<Input> {
    load_png_corpus(
        "/home/lilith/work/codec-corpus/gb82-sc",
        "sc",
        6,
    )
}

// --------------------------------------------------------------------------
// Real TIFF-LZW strip extraction.
//
// The bench above synthetically LZW-encodes raw RGB bytes via weezl's
// own encoder. That's correct, but doesn't exercise the exact bit layout
// produced by real TIFF writers (libtiff, imagemagick, etc.). For real
// coverage we want to load actual .tif files with COMPRESSION_LZW=5 from
// disk and feed the LZW strip bytes directly to the decoder.
//
// Minimal TIFF parser: reads the 8-byte header, first IFD, pulls out
// StripOffsets (tag 273), StripByteCounts (tag 279), and ImageLength
// (tag 257) / SamplesPerPixel (tag 277) / BitsPerSample (tag 258) /
// ImageWidth (tag 256) — enough to compute the expected uncompressed
// size. Concatenates all strip bytes into one buffer (for single-strip
// and multi-strip files both). Assumes little-endian ("II") for now.
// --------------------------------------------------------------------------

pub struct TiffStrips {
    pub name: &'static str,
    pub lzw_bytes: Vec<u8>,
    /// Expected raw (decompressed) size = width * rows_per_strip * samples * (bits/8).
    /// This is used as the output buffer size for the bench.
    pub decompressed_size: usize,
}

/// Returns the LARGEST single strip from a TIFF file, as a self-contained
/// LZW stream. Each TIFF strip is an independent LZW sequence with its own
/// clear code at the start, so strips can't be concatenated into one stream.
/// We take the largest so that the bench sees a reasonable-sized input.
pub fn parse_tiff_lzw_strips(path: &std::path::Path) -> Option<(Vec<u8>, usize)> {
    let bytes = std::fs::read(path).ok()?;
    if bytes.len() < 8 {
        return None;
    }
    let le = match &bytes[0..2] {
        b"II" => true,
        b"MM" => false,
        _ => return None,
    };
    let u16_at = |i: usize| -> u16 {
        if le { u16::from_le_bytes([bytes[i], bytes[i + 1]]) }
        else  { u16::from_be_bytes([bytes[i], bytes[i + 1]]) }
    };
    let u32_at = |i: usize| -> u32 {
        if le { u32::from_le_bytes([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]) }
        else  { u32::from_be_bytes([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]) }
    };
    if u16_at(2) != 42 { return None; }
    let ifd_off = u32_at(4) as usize;
    if ifd_off + 2 > bytes.len() { return None; }
    let n_entries = u16_at(ifd_off) as usize;
    let entries_off = ifd_off + 2;

    let mut compression = 0u16;
    let mut strip_offsets: Vec<u32> = Vec::new();
    let mut strip_byte_counts: Vec<u32> = Vec::new();
    let mut width = 0u32;
    let mut bits_per_sample = 8u32;
    let mut samples_per_pixel = 1u32;
    let mut rows_per_strip = u32::MAX;

    for i in 0..n_entries {
        let entry_off = entries_off + i * 12;
        if entry_off + 12 > bytes.len() { return None; }
        let tag = u16_at(entry_off);
        let field_type = u16_at(entry_off + 2);
        let count = u32_at(entry_off + 4) as usize;
        let value_off = entry_off + 8;

        let read_values = |type_: u16, n: usize, val_off: usize| -> Vec<u32> {
            let elem_size = match type_ {
                1 => 1, 3 => 2, 4 => 4, _ => return Vec::new(),
            };
            let total = n * elem_size;
            let data_off = if total <= 4 { val_off } else { u32_at(val_off) as usize };
            if data_off + total > bytes.len() { return Vec::new(); }
            (0..n).map(|k| match type_ {
                1 => bytes[data_off + k] as u32,
                3 => u16_at(data_off + k * 2) as u32,
                4 => u32_at(data_off + k * 4),
                _ => 0,
            }).collect()
        };

        match tag {
            256 => width = read_values(field_type, 1, value_off)[0],
            258 => bits_per_sample = read_values(field_type, 1, value_off).get(0).copied().unwrap_or(8),
            259 => compression = read_values(field_type, 1, value_off)[0] as u16,
            273 => strip_offsets = read_values(field_type, count, value_off),
            277 => samples_per_pixel = read_values(field_type, 1, value_off).get(0).copied().unwrap_or(1),
            278 => rows_per_strip = read_values(field_type, 1, value_off).get(0).copied().unwrap_or(u32::MAX),
            279 => strip_byte_counts = read_values(field_type, count, value_off),
            _ => {}
        }
    }

    if compression != 5 { return None; }
    if strip_offsets.is_empty() || strip_offsets.len() != strip_byte_counts.len() { return None; }

    // Find the LARGEST strip (by decoded size).
    let mut best: Option<(usize, u32, u32)> = None; // (idx, offset, byte_count)
    for (i, (&off, &n)) in strip_offsets.iter().zip(strip_byte_counts.iter()).enumerate() {
        if off as usize + n as usize > bytes.len() { return None; }
        match best {
            None => best = Some((i, off, n)),
            Some((_, _, bn)) if n > bn => best = Some((i, off, n)),
            _ => {}
        }
    }
    let (_, off, n) = best?;
    let lzw = bytes[off as usize..(off + n) as usize].to_vec();
    // Decompressed size of one strip = width * rows_per_strip * samples * bits/8.
    // rows_per_strip is capped by image height; for single-strip files this is
    // the whole image. We don't check image height here — just trust the tag.
    let decompressed = (width as usize)
        * (rows_per_strip as usize).min(65536) // sanity cap
        * (samples_per_pixel as usize)
        * (bits_per_sample as usize / 8).max(1);
    Some((lzw, decompressed))
}

pub fn load_lzw_tiff_corpus(dir: &str, tag_prefix: &'static str) -> Vec<TiffStrips> {
    let mut out = Vec::new();
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return out,
    };
    let mut paths: Vec<_> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            let ext = p.extension().and_then(|e| e.to_str());
            ext == Some("tif") || ext == Some("tiff")
        })
        .collect();
    paths.sort();
    for path in paths {
        if let Some((lzw, decompressed_size)) = parse_tiff_lzw_strips(&path) {
            if lzw.len() < 1024 || decompressed_size < 4096 {
                continue;
            }
            let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("?").to_string();
            let leaked: &'static str = Box::leak(format!("{}/{}", tag_prefix, name).into_boxed_str());
            out.push(TiffStrips {
                name: leaked,
                lzw_bytes: lzw,
                decompressed_size,
            });
        }
    }
    out
}

/// Real TIFF files from the tiff-conformance corpus (codec-corpus).
pub fn tiff_conformance_lzw() -> Vec<TiffStrips> {
    load_lzw_tiff_corpus(
        "/home/lilith/work/codec-corpus/tiff-conformance/valid",
        "conform",
    )
}

/// LZW TIFFs converted from the QOI screenshot_web PNG corpus via
/// `convert -compress lzw`. Generated to /tmp/tiff_corpus/qoi at bench
/// setup time (see README or the convert script).
pub fn qoi_as_lzw_tiff() -> Vec<TiffStrips> {
    load_lzw_tiff_corpus("/tmp/tiff_corpus/qoi", "qoi-tif")
}

/// LZW TIFFs converted from the gb82-sc corpus.
pub fn sc_as_lzw_tiff() -> Vec<TiffStrips> {
    load_lzw_tiff_corpus("/tmp/tiff_corpus/sc", "sc-tif")
}

/// LZW TIFFs from CLIC2025 photographic corpus WITH horizontal predictor
/// (imagemagick's default for `-compress lzw`). This is what most real
/// photographic TIFF-LZW files look like in the wild.
pub fn clic_as_lzw_tiff_pred() -> Vec<TiffStrips> {
    load_lzw_tiff_corpus("/tmp/tiff_corpus/clic", "clic-pred")
}

/// LZW TIFFs from CLIC2025 WITHOUT predictor (`tiff:predictor=1`). Models
/// pathological "raw photographic data passed straight to LZW" where
/// compression ratio is near 1.0 and most codes are short.
pub fn clic_as_lzw_tiff_nopred() -> Vec<TiffStrips> {
    load_lzw_tiff_corpus("/tmp/tiff_corpus/clic-nopred", "clic-raw")
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

    /// Roundtrip test for MSB + TIFF early-change: encode with classic
    /// TIFF mode, decode with Tight TIFF/MSB, assert byte equality.
    /// Covers the image-tiff usage pattern.
    #[test]
    fn tight_msb_tiff_roundtrip() {
        use weezl::{encode::Encoder, BitOrder};
        let corpus = standard_corpus();
        for input in corpus.iter() {
            // Re-encode as MSB + TIFF early-change (image-tiff's pattern).
            let encoded = Encoder::with_tiff_size_switch(BitOrder::Msb, 8)
                .encode(&input.raw)
                .unwrap();

            let mut out_classic = vec![0u8; input.raw.len() + 64];
            let mut out_chunked = vec![0u8; input.raw.len() + 64];
            let mut out_tight = vec![0u8; input.raw.len() + 64];

            let n1 = decode_weezl_with_order(
                &encoded,
                &mut out_classic,
                TableStrategy::Classic,
                BitOrder::Msb,
                true,
            );
            let n2 = decode_weezl_with_order(
                &encoded,
                &mut out_chunked,
                TableStrategy::Chunked,
                BitOrder::Msb,
                true,
            );
            let n3 = decode_weezl_with_order(
                &encoded,
                &mut out_tight,
                TableStrategy::Tight,
                BitOrder::Msb,
                true,
            );

            assert_eq!(n1, input.raw.len(), "{} classic", input.name);
            assert_eq!(n2, input.raw.len(), "{} chunked", input.name);
            assert_eq!(n3, input.raw.len(), "{} tight", input.name);

            assert_eq!(&out_classic[..n1], &input.raw[..], "{} classic bytes", input.name);
            assert_eq!(&out_chunked[..n2], &input.raw[..], "{} chunked bytes", input.name);
            assert_eq!(&out_tight[..n3], &input.raw[..], "{} tight/msb/tiff bytes", input.name);
        }
    }

    /// Non-TIFF MSB roundtrip (for completeness — not an image-tiff case
    /// but exercises the Tight MSB bit reader on its own).
    #[test]
    fn tight_msb_non_tiff_roundtrip() {
        use weezl::{encode::Encoder, BitOrder};
        for input in standard_corpus() {
            let encoded = Encoder::new(BitOrder::Msb, 8).encode(&input.raw).unwrap();
            let mut out_tight = vec![0u8; input.raw.len() + 64];
            let n = decode_weezl_with_order(
                &encoded,
                &mut out_tight,
                TableStrategy::Tight,
                BitOrder::Msb,
                false,
            );
            assert_eq!(n, input.raw.len(), "{} tight/msb", input.name);
            assert_eq!(&out_tight[..n], &input.raw[..], "{} tight/msb bytes", input.name);
        }
    }
}
