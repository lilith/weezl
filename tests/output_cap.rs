//! Tests for `Configuration::with_max_output_bytes` (M1 hardening).

use weezl::{decode, encode, BitOrder, LzwError};

/// Encode a known payload, then decode it with a cap that is larger than the output:
/// must succeed and produce the original bytes.
#[test]
fn cap_above_output_succeeds() {
    let data: Vec<u8> = (0..=255u8).cycle().take(8192).collect();
    let encoded = encode::Encoder::new(BitOrder::Msb, 8)
        .encode(&data)
        .expect("encode");

    let mut decoded = Vec::new();
    let result = decode::Configuration::new(BitOrder::Msb, 8)
        .with_max_output_bytes(data.len() * 4)
        .build()
        .into_vec(&mut decoded)
        .decode_all(&encoded);

    assert!(
        result.status.is_ok(),
        "expected success, got {:?}",
        result.status
    );
    assert_eq!(decoded, data);
}

/// Cap exactly equal to the output size — must succeed.
#[test]
fn cap_equal_to_output_succeeds() {
    let data: Vec<u8> = (0..=255u8).cycle().take(4096).collect();
    let encoded = encode::Encoder::new(BitOrder::Msb, 8)
        .encode(&data)
        .expect("encode");

    let mut decoded = Vec::new();
    let result = decode::Configuration::new(BitOrder::Msb, 8)
        .with_max_output_bytes(data.len())
        .build()
        .into_vec(&mut decoded)
        .decode_all(&encoded);

    assert!(
        result.status.is_ok(),
        "expected success, got {:?}",
        result.status
    );
    assert_eq!(decoded, data);
}

/// Cap below the output size — must return `OutputCapExceeded`.
#[test]
fn cap_below_output_errors() {
    let data: Vec<u8> = (0..=255u8).cycle().take(8192).collect();
    let encoded = encode::Encoder::new(BitOrder::Msb, 8)
        .encode(&data)
        .expect("encode");

    let cap = 100;
    let mut decoded = Vec::new();
    let result = decode::Configuration::new(BitOrder::Msb, 8)
        .with_max_output_bytes(cap)
        .build()
        .into_vec(&mut decoded)
        .decode_all(&encoded);

    match result.status {
        Err(LzwError::OutputCapExceeded) => {}
        other => panic!("expected OutputCapExceeded, got {:?}", other),
    }
    // Partial output is preserved; must not exceed the cap.
    assert!(
        decoded.len() <= cap,
        "produced {} bytes but cap was {}",
        decoded.len(),
        cap
    );
}

/// Cap of zero — must immediately error if the stream produces any output.
#[test]
fn cap_zero_errors_immediately() {
    let data = vec![42u8; 1024];
    let encoded = encode::Encoder::new(BitOrder::Msb, 8)
        .encode(&data)
        .expect("encode");

    let mut decoded = Vec::new();
    let result = decode::Configuration::new(BitOrder::Msb, 8)
        .with_max_output_bytes(0)
        .build()
        .into_vec(&mut decoded)
        .decode_all(&encoded);

    match result.status {
        Err(LzwError::OutputCapExceeded) => {}
        other => panic!("expected OutputCapExceeded, got {:?}", other),
    }
    assert_eq!(decoded.len(), 0);
}

/// Default behaviour (no cap) — large output decodes without error.
#[test]
fn no_cap_default_unbounded() {
    let data: Vec<u8> = (0..=255u8).cycle().take(16384).collect();
    let encoded = encode::Encoder::new(BitOrder::Msb, 8)
        .encode(&data)
        .expect("encode");

    let mut decoded = Vec::new();
    let result = decode::Configuration::new(BitOrder::Msb, 8)
        .build()
        .into_vec(&mut decoded)
        .decode_all(&encoded);

    assert!(
        result.status.is_ok(),
        "expected success, got {:?}",
        result.status
    );
    assert_eq!(decoded, data);
}

/// `into_stream` (std-only) also enforces the cap.
#[cfg(feature = "std")]
#[test]
fn cap_below_output_errors_into_stream() {
    let data: Vec<u8> = (0..=255u8).cycle().take(8192).collect();
    let encoded = encode::Encoder::new(BitOrder::Msb, 8)
        .encode(&data)
        .expect("encode");

    let cap = 200;
    let mut decoded = Vec::new();
    let result = decode::Configuration::new(BitOrder::Msb, 8)
        .with_max_output_bytes(cap)
        .build()
        .into_stream(&mut decoded)
        .decode_all(&encoded[..]);

    assert!(
        result.status.is_err(),
        "expected error, got status={:?} bytes_written={}",
        result.status,
        result.bytes_written
    );
    let err = result.status.unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("OutputCapExceeded"),
        "expected OutputCapExceeded in error, got {:?}",
        msg
    );
    assert!(
        decoded.len() <= cap,
        "produced {} bytes but cap was {}",
        decoded.len(),
        cap
    );
}
