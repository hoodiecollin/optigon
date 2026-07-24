fn main() {
    // A PyO3 extension module links against an *undefined* libpython — the host
    // interpreter provides those symbols at import time. On macOS the linker must
    // be told to allow the undefined symbols. maturin does this automatically;
    // emitting the link args here (scoped to this crate, so no workspace-wide
    // recompile) makes a plain `cargo build`/`cargo check` of the extension work
    // as a CI compile-check too.
    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-arg=-undefined");
        println!("cargo:rustc-link-arg=dynamic_lookup");
    }
}
