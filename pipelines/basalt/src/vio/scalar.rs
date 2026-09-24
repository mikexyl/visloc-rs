//! Numeric ownership for low-level solver/reference arithmetic.
//!
//! Basalt's public feed is double precision (`ImuData<double>`, calibration
//! and trajectory output). The Rust VIO estimator always uses f64 and exposes
//! no precision selector. The f32 variant remains for upstream numerical
//! reference fixtures and low-level arithmetic tests, not as a VIO mode.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarMode {
    /// The pinned Basalt default: `SqrtKeypointVioEstimator<float>`.
    UpstreamF32,
    /// The fixed precision of the Rust VIO estimator.
    ExtendedF64,
}

impl Default for ScalarMode {
    fn default() -> Self {
        Self::ExtendedF64
    }
}
