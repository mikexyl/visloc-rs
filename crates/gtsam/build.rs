use std::{env, path::PathBuf};
fn main() {
    println!("cargo:rerun-if-changed=native");
    println!("cargo:rerun-if-env-changed=VISLOC_GTSAM_ROOT");
    println!("cargo:rerun-if-env-changed=VISLOC_GTSAM_EIGEN_INCLUDE");
    let root = env::var_os("VISLOC_GTSAM_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap())
                .join("../../.runtime/gtsam-native/install")
        });
    let root = root.canonicalize().expect(
        "GTSAM C++ is missing: run bash scripts/build_gtsam_native.sh or set VISLOC_GTSAM_ROOT",
    );
    let eigen = env::var_os("VISLOC_GTSAM_EIGEN_INCLUDE")
        .map(PathBuf::from)
        .unwrap_or_else(|| "/usr/include/eigen3".into());
    assert!(
        root.join("lib/libgtsam.a").exists(),
        "a static GTSAM 4.3.0 build is required"
    );
    for dependency in [
        "include/gtsam/config.h",
        "lib/libgtsam.a",
        "lib/libcephes-gtsam.a",
    ] {
        println!("cargo:rerun-if-changed={}", root.join(dependency).display());
    }
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .include(root.join("include"))
        .include(eigen)
        .file("native/bridge.cpp")
        .warnings(true)
        .compile("visloc_gtsam_bridge");
    println!(
        "cargo:rustc-link-search=native={}",
        root.join("lib").display()
    );
    println!("cargo:rustc-link-lib=static=gtsam");
    println!("cargo:rustc-link-lib=static=cephes-gtsam");
    // No libpython, runtime process, or shared libgtsam dependency.
    println!("cargo:rustc-link-lib=pthread");
}
