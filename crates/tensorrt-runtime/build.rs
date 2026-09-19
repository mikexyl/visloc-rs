fn main() {
    println!("cargo:rerun-if-changed=native");
    #[cfg(feature = "native")]
    native();
}

#[cfg(feature = "native")]
fn native() {
    use std::{env, path::PathBuf};
    for key in [
        "TENSORRT_ROOT",
        "TENSORRT_INCLUDE_DIR",
        "TENSORRT_LIB_DIR",
        "CUDA_HOME",
        "CUDA_INCLUDE_DIR",
        "CUDA_LIB_DIR",
    ] {
        println!("cargo:rerun-if-env-changed={key}");
    }
    assert_eq!(
        env::var("CARGO_CFG_TARGET_OS").unwrap(),
        "linux",
        "Linux is currently supported"
    );
    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    let multiarch = format!("{arch}-linux-gnu");
    let root = env::var_os("TENSORRT_ROOT").map(PathBuf::from);
    let include = env::var_os("TENSORRT_INCLUDE_DIR")
        .map(PathBuf::from)
        .or_else(|| root.as_ref().map(|p| p.join("include")))
        .unwrap_or_else(|| PathBuf::from(format!("/usr/include/{multiarch}")));
    let cuda = env::var_os("CUDA_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| "/usr/local/cuda".into());
    let cuda_include = env::var_os("CUDA_INCLUDE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| cuda.join("include"));
    assert!(
        include.join("NvInferRuntime.h").exists(),
        "TensorRT headers missing; set TENSORRT_INCLUDE_DIR"
    );
    assert!(
        cuda_include.join("cuda_runtime_api.h").exists(),
        "CUDA headers missing; set CUDA_HOME"
    );
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .include(include)
        .include(cuda_include)
        .file("native/bridge.cpp")
        .warnings(true)
        .compile("visloc_trt_bridge");
    if let Some(lib) = env::var_os("TENSORRT_LIB_DIR")
        .map(PathBuf::from)
        .or_else(|| root.map(|p| p.join("lib")))
    {
        println!("cargo:rustc-link-search=native={}", lib.display());
    }
    let lib = env::var_os("CUDA_LIB_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| cuda.join("lib64"));
    println!("cargo:rustc-link-search=native={}", lib.display());
    for lib in ["nvinfer", "nvinfer_plugin", "cudart"] {
        println!("cargo:rustc-link-lib={lib}");
    }
}
