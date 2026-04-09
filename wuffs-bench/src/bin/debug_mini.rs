use weezl::{decode::{Configuration, TableStrategy}, encode::Encoder, BitOrder};
use wuffs_bench::standard_corpus;

fn main() {
    for input in standard_corpus() {
        let encoded = Encoder::new(BitOrder::Msb, 8).encode(&input.raw).unwrap();
        let mut out = vec![0u8; input.raw.len() + 64];
        let mut dec = Configuration::new(BitOrder::Msb, 8)
            .with_table_strategy(TableStrategy::Tight)
            .build();
        let r = dec.decode_bytes(&encoded, &mut out);
        let ok = r.status.is_ok() && r.consumed_out == input.raw.len() && &out[..r.consumed_out] == &input.raw[..];
        let first_diff = (0..r.consumed_out.min(input.raw.len()))
            .find(|&i| out[i] != input.raw[i]);
        println!(
            "{:12} raw={} enc={} consumed={} status={:?} ok={} first_diff={:?}",
            input.name,
            input.raw.len(),
            encoded.len(),
            r.consumed_out,
            r.status,
            ok,
            first_diff
        );
    }
}
