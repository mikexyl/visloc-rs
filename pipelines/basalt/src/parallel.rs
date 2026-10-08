//! Bounded, ordered parallel work for independent landmark calculations.
//! Floating-point accumulation stays on the caller in the original order.

use rayon::{prelude::*, ThreadPool, ThreadPoolBuilder};
use std::{ops::Range, sync::OnceLock};

static POOL: OnceLock<Option<ThreadPool>> = OnceLock::new();

fn pool() -> Option<&'static ThreadPool> {
    POOL.get_or_init(|| {
        let threads = match std::env::var("VISLOC_BASALT_THREADS") {
            Ok(value) => match value.parse::<usize>() {
                Ok(count) if count > 0 => count,
                _ => {
                    eprintln!("VISLOC_BASALT_THREADS must be a positive integer; using one thread");
                    1
                }
            },
            Err(std::env::VarError::NotPresent) => std::thread::available_parallelism()
                .map_or(1, usize::from)
                .min(4),
            Err(_) => {
                eprintln!("VISLOC_BASALT_THREADS is invalid; using one thread");
                1
            }
        };
        if threads == 1 {
            return None;
        }
        match ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|index| format!("basalt-landmark-{index}"))
            .build()
        {
            Ok(pool) => Some(pool),
            Err(error) => {
                eprintln!("Basalt landmark pool unavailable ({error}); using one thread");
                None
            }
        }
    })
    .as_ref()
}

/// Indexed collection preserves missing entries and landmark identities. Do
/// not replace the caller's ordered floating-point fold with a Rayon reduce.
/// Small work lists and diagnostic callers stay on their originating thread.
pub(crate) fn map_ordered<T: Send>(
    range: Range<usize>,
    enabled: bool,
    map: impl Fn(usize) -> T + Sync + Send,
) -> Vec<T> {
    if enabled && range.len() >= 16 {
        if let Some(pool) = pool() {
            return pool.install(|| range.into_par_iter().map(map).collect());
        }
    }
    range.map(map).collect()
}
