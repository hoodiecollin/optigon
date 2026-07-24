fn main() {
    // Wires platform linker args (incl. macOS `-undefined dynamic_lookup`) so the
    // .node resolves Node-API symbols at load time.
    napi_build::setup();
}
