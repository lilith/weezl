//! Compiles the vendored wuffs single-file C library.

fn main() {
    let wuffs = std::path::Path::new("vendor/wuffs-v0.4.c");
    println!("cargo:rerun-if-changed=vendor/wuffs-v0.4.c");
    println!("cargo:rerun-if-changed=build.rs");

    if !wuffs.exists() {
        panic!(
            "wuffs-bench/vendor/wuffs-v0.4.c missing. Fetch it from:\n  \
             https://raw.githubusercontent.com/google/wuffs/main/release/c/wuffs-v0.4.c"
        );
    }

    let shim = std::path::Path::new("vendor/wuffs_shim.c");
    println!("cargo:rerun-if-changed=vendor/wuffs_shim.c");
    if !shim.exists() {
        panic!("wuffs-bench/vendor/wuffs_shim.c missing");
    }

    cc::Build::new()
        .file(wuffs)
        .file(shim)
        .define("WUFFS_IMPLEMENTATION", None)
        .define("WUFFS_CONFIG__MODULES", None)
        .define("WUFFS_CONFIG__MODULE__BASE", None)
        .define("WUFFS_CONFIG__MODULE__LZW", None)
        .flag_if_supported("-Wno-unused-function")
        .flag_if_supported("-Wno-unused-variable")
        .flag_if_supported("-Wno-unused-parameter")
        .flag_if_supported("-Wno-implicit-fallthrough")
        .flag_if_supported("-Wno-unreachable-code")
        .opt_level(3)
        .compile("wuffs_lzw");
}
