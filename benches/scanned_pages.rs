//! Benchmark comparing LZW decode strategies on real scanned-page TIFFs.
//!
//! Synthetic archetypes can't reproduce the exact byte distribution of a
//! scanned page: wide white margins, a centered text block with realistic
//! font kerning, horizontal-predictor state across rows. This bench loads
//! real 8-bit grayscale TIFF-LZW pages rendered from public-domain text
//! (Project Gutenberg) and measures the decode throughput of each strategy
//! on the raw LZW strip bytes pulled from the files.
//!
//! The TIFFs live outside the repo under `$SCANNED_PAGES_DIR` (default
//! `/mnt/v/input/scanned-docs`). Each file is a letter-size (2550×3300)
//! 8-bit grayscale page with LZW compression and horizontal differencing.
//!
//! Run with (busy host, want the floor):
//!     cargo bench --bench scanned_pages -- --best-of-passes=3
//! or (quiet host, want expected performance):
//!     cargo bench --bench scanned_pages -- --mean-of-passes=5
//!
//! The IFD walker is a minimal little-endian TIFF parser: it finds
//! StripOffsets (0x0111), StripByteCounts (0x0117), and StripsPerImage,
//! then slices each strip's raw LZW bytes out of the file. No LZW
//! decoding happens in the parser — just byte slicing.

use std::env;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use weezl::{
    decode::{Configuration, TableStrategy},
    BitOrder, LzwStatus,
};
use zenbench::prelude::*;

/// One LZW strip pulled out of a TIFF file.
#[derive(Clone)]
struct Strip {
    /// Raw LZW bytes ready to feed to the decoder.
    encoded: Vec<u8>,
    /// Number of decoded bytes this strip produces (rows_per_strip × bytes_per_row).
    decoded_len: usize,
}

/// All strips from one page file.
struct Page {
    name: &'static str,
    strips: Vec<Strip>,
    total_decoded: usize,
}

fn decode_all(encoded: &[u8], out: &mut [u8], strategy: TableStrategy) -> usize {
    // TIFF-LZW: MSB bit order, min_code_size 8, early-change size switch.
    let mut dec = Configuration::with_tiff_size_switch(BitOrder::Msb, 8)
        .with_table_strategy(strategy)
        .build();
    let mut inp = encoded;
    let mut cursor = out;
    let mut written = 0;
    loop {
        let r = dec.decode_bytes(inp, cursor);
        inp = &inp[r.consumed_in..];
        written += r.consumed_out;
        cursor = &mut std::mem::take(&mut cursor)[r.consumed_out..];
        match r.status {
            Ok(LzwStatus::Done | LzwStatus::NoProgress) => return written,
            Ok(LzwStatus::Ok) => {
                if inp.is_empty() && cursor.is_empty() {
                    return written;
                }
            }
            Err(_) => return written,
        }
    }
}

/// Minimal little-endian IFD walker. Returns all strips from the first IFD,
/// or `None` if the file isn't a single-IFD little-endian TIFF with LZW.
fn extract_strips(bytes: &[u8]) -> Option<Page> {
    if bytes.len() < 8 || &bytes[0..2] != b"II" {
        return None;
    }
    let ifd_offset = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
    if ifd_offset + 2 > bytes.len() {
        return None;
    }
    let num_entries = u16::from_le_bytes([bytes[ifd_offset], bytes[ifd_offset + 1]]) as usize;
    let entries_start = ifd_offset + 2;

    let mut image_width: u32 = 0;
    let mut image_length: u32 = 0;
    let mut rows_per_strip: u32 = 0;
    let mut strip_offsets: Vec<u32> = Vec::new();
    let mut strip_byte_counts: Vec<u32> = Vec::new();
    let mut compression: u16 = 0;
    let mut bits_per_sample: u16 = 0;
    let mut samples_per_pixel: u16 = 1;

    for i in 0..num_entries {
        let e = entries_start + i * 12;
        if e + 12 > bytes.len() {
            return None;
        }
        let tag = u16::from_le_bytes([bytes[e], bytes[e + 1]]);
        let ty = u16::from_le_bytes([bytes[e + 2], bytes[e + 3]]);
        let count =
            u32::from_le_bytes([bytes[e + 4], bytes[e + 5], bytes[e + 6], bytes[e + 7]]) as usize;
        let value_bytes = &bytes[e + 8..e + 12];

        // For scalar tags we care about, read from the inline value.
        match tag {
            // ImageWidth / ImageLength: usually LONG, sometimes SHORT.
            0x0100 | 0x0101 => {
                let v = if ty == 3 {
                    u16::from_le_bytes([value_bytes[0], value_bytes[1]]) as u32
                } else {
                    u32::from_le_bytes([
                        value_bytes[0],
                        value_bytes[1],
                        value_bytes[2],
                        value_bytes[3],
                    ])
                };
                if tag == 0x0100 {
                    image_width = v;
                } else {
                    image_length = v;
                }
            }
            // BitsPerSample (SHORT).
            0x0102 => {
                bits_per_sample = u16::from_le_bytes([value_bytes[0], value_bytes[1]]);
            }
            // Compression (SHORT). LZW = 5.
            0x0103 => {
                compression = u16::from_le_bytes([value_bytes[0], value_bytes[1]]);
            }
            // SamplesPerPixel (SHORT).
            0x0115 => {
                samples_per_pixel = u16::from_le_bytes([value_bytes[0], value_bytes[1]]);
            }
            // RowsPerStrip: usually LONG.
            0x0116 => {
                rows_per_strip = if ty == 3 {
                    u16::from_le_bytes([value_bytes[0], value_bytes[1]]) as u32
                } else {
                    u32::from_le_bytes([
                        value_bytes[0],
                        value_bytes[1],
                        value_bytes[2],
                        value_bytes[3],
                    ])
                };
            }
            // StripOffsets (0x0111) and StripByteCounts (0x0117) — array.
            0x0111 | 0x0117 => {
                let elem_size = match ty {
                    3 => 2usize,
                    4 => 4usize,
                    _ => return None,
                };
                let total = elem_size.checked_mul(count)?;
                let read_slice: &[u8] = if total <= 4 {
                    value_bytes
                } else {
                    let off = u32::from_le_bytes([
                        value_bytes[0],
                        value_bytes[1],
                        value_bytes[2],
                        value_bytes[3],
                    ]) as usize;
                    if off + total > bytes.len() {
                        return None;
                    }
                    &bytes[off..off + total]
                };
                let mut out: Vec<u32> = Vec::with_capacity(count);
                for j in 0..count {
                    let v = if elem_size == 2 {
                        u16::from_le_bytes([read_slice[j * 2], read_slice[j * 2 + 1]]) as u32
                    } else {
                        u32::from_le_bytes([
                            read_slice[j * 4],
                            read_slice[j * 4 + 1],
                            read_slice[j * 4 + 2],
                            read_slice[j * 4 + 3],
                        ])
                    };
                    out.push(v);
                }
                if tag == 0x0111 {
                    strip_offsets = out;
                } else {
                    strip_byte_counts = out;
                }
            }
            _ => {}
        }
    }

    if compression != 5 {
        return None; // not LZW
    }
    if strip_offsets.len() != strip_byte_counts.len() || strip_offsets.is_empty() {
        return None;
    }
    if image_width == 0 || image_length == 0 || rows_per_strip == 0 {
        return None;
    }

    // bytes per row = width * samples * bits/8 (grayscale 8-bit = width)
    let bytes_per_row =
        (image_width as usize) * (samples_per_pixel as usize) * ((bits_per_sample as usize) / 8);

    let mut strips: Vec<Strip> = Vec::with_capacity(strip_offsets.len());
    let mut total_decoded = 0usize;
    for (idx, (&off, &len)) in strip_offsets
        .iter()
        .zip(strip_byte_counts.iter())
        .enumerate()
    {
        let off = off as usize;
        let len = len as usize;
        if off + len > bytes.len() {
            return None;
        }
        let rows_in_strip = if idx == strip_offsets.len() - 1 {
            (image_length as usize) - idx * (rows_per_strip as usize)
        } else {
            rows_per_strip as usize
        };
        let decoded_len = rows_in_strip * bytes_per_row;
        strips.push(Strip {
            encoded: bytes[off..off + len].to_vec(),
            decoded_len,
        });
        total_decoded += decoded_len;
    }

    Some(Page {
        name: "",
        strips,
        total_decoded,
    })
}

fn load_page(name: &'static str, dir: &std::path::Path) -> Option<Page> {
    let path = dir.join(format!("{name}.tif"));
    let bytes = fs::read(&path).ok()?;
    let mut page = extract_strips(&bytes)?;
    page.name = name;
    Some(page)
}

fn bench_page(g: &mut BenchGroup, page: &Page) {
    g.throughput(Throughput::Bytes(page.total_decoded as u64));
    g.config().min_sample_ns(10_000_000);
    g.config().max_rounds(500);
    g.config().min_rounds(100);

    // Allocate once per strategy. The page's strips feed the decoder
    // sequentially each iteration; the output buffer is reused.
    let out_cap = page.total_decoded + 4096;

    for &(label, strategy) in &[
        ("bytelink", TableStrategy::ByteLink),
        ("streaming", TableStrategy::Streaming),
    ] {
        // Shallow clone of all strips so each bench closure owns them.
        let strips: Arc<Vec<Strip>> = Arc::new(page.strips.clone());
        g.bench(label, move |b| {
            let strips = Arc::clone(&strips);
            let mut out = vec![0u8; out_cap];
            b.iter(move || {
                let mut written = 0;
                for strip in strips.iter() {
                    let n = decode_all(
                        &strip.encoded,
                        &mut out[written..written + strip.decoded_len + 4096],
                        strategy,
                    );
                    written += n;
                }
                black_box(&out[..written]);
                written
            });
        });
    }
}

fn bench_scanned_pages(suite: &mut Suite) {
    let dir = env::var("SCANNED_PAGES_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/mnt/v/input/scanned-docs"));

    let page_names = ["page-title", "page-sparse", "page-dense", "page-full"];
    let mut pages = Vec::new();
    for name in page_names {
        match load_page(name, &dir) {
            Some(p) => {
                eprintln!(
                    "[scanned_pages] {}: {} strips, {} decoded bytes, {:.1}:1 ratio",
                    name,
                    p.strips.len(),
                    p.total_decoded,
                    p.total_decoded as f64
                        / p.strips.iter().map(|s| s.encoded.len()).sum::<usize>() as f64
                );
                pages.push(p);
            }
            None => {
                eprintln!(
                    "[scanned_pages] skipping {} (not found or invalid TIFF-LZW): {}",
                    name,
                    dir.join(format!("{name}.tif")).display()
                );
            }
        }
    }

    if pages.is_empty() {
        eprintln!(
            "[scanned_pages] no pages loaded — set SCANNED_PAGES_DIR or populate /mnt/v/input/scanned-docs"
        );
        return;
    }

    for page in &pages {
        let name = page.name;
        suite.group(format!("scan/{name}"), |g| bench_page(g, page));
    }
}

zenbench::main!(bench_scanned_pages);
