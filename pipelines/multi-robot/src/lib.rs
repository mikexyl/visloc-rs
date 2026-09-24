//! Transport-independent SB-SLAM sequence refinement. No ROS, ground truth, or
//! corrected pose ever enters the local VIO estimator.
pub mod backend;
pub mod geometry;
#[cfg(feature = "native")]
pub mod native;
pub mod sequence;
pub mod types;
pub use types::*;

#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct Error(pub String);
pub type Result<T> = std::result::Result<T, Error>;
