pub mod central;
pub mod robot;
pub mod sensor;
pub mod wire;
pub type AnyResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

pub mod traffic;

pub mod archive;
