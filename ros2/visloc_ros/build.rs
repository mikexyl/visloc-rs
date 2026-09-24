fn main() {
    println!("cargo:rerun-if-env-changed=AMENT_PREFIX_PATH");
    for prefix in std::env::var("AMENT_PREFIX_PATH")
        .unwrap_or_default()
        .split(':')
        .filter(|p| !p.is_empty())
    {
        println!("cargo:rustc-link-search=native={prefix}/lib");
    }
    // Local unpacked Humble workaround dependency required by rclrs 0.7.
    // rclrs' own build script does not track AMENT_PREFIX_PATH changes.
    let local = std::path::Path::new(&std::env::var("CARGO_MANIFEST_DIR").unwrap())
        .join("../../.runtime/ros2_system_deps/opt/ros/humble/lib");
    if local.exists() {
        println!(
            "cargo:rustc-link-search=native={}",
            local.canonicalize().unwrap().display()
        );
    }
}
