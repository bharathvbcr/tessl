//! The host-side batch checks and segment upload that the encoder models
//! ([`crate::bert`], [`crate::embedgemma2`]) share. Each model keeps its own
//! public entry point and passes its own name, so a message still names the
//! caller; what the two checks refuse is the same for both.

use std::sync::Arc;

use crate::runtime::GpuRuntime;
use crate::tensor::GpuBuffer;

/// The shape of a batch of sequences, refused before any GPU work: not
/// empty, at most `max_batch` sequences, and `trace` only for one.
pub(crate) fn check_batch_shape(what: &str, n: usize, max_batch: usize, trace: bool) -> Result<(), String> {
    if n == 0 {
        return Err(format!("{what}: empty batch"));
    }
    if n > max_batch {
        return Err(format!("{what}: {n} sequences exceeds the batch limit {max_batch}"));
    }
    if trace && n != 1 {
        return Err(format!("{what}: trace takes one sequence, got {n}"));
    }
    Ok(())
}

/// `[start, end)` row ranges checked against `rows` (`start < end <= rows`)
/// and uploaded as `[n, 2]` u32. `owner` names the public `upload_segments`
/// that called, for its messages.
///
/// The buffer is fresh and written without waiting for the GPU: a host write
/// into an ordinary allocation would first commit and wait on everything
/// queued (the whole encoder forward, on a model's first head block).
pub(crate) fn upload_segments(
    rt: &Arc<GpuRuntime>,
    owner: &str,
    segments: &[(u32, u32)],
    rows: u32,
) -> Result<GpuBuffer, String> {
    for (i, &(start, end)) in segments.iter().enumerate() {
        if start >= end || end > rows {
            return Err(format!(
                "{owner}::upload_segments: segment {i} is [{start}, {end}); need start < end <= {rows}"
            ));
        }
    }
    let flat: Vec<u32> = segments.iter().flat_map(|&(a, b)| [a, b]).collect();
    rt.alloc_buffer_from_u32(&flat)
        .map_err(|e| format!("{owner}::upload_segments: {e}"))
}

#[cfg(test)]
mod tests {
    use super::check_batch_shape;

    #[test]
    fn batch_shape_refusals_in_order() {
        assert_eq!(check_batch_shape("w", 0, 4, true).unwrap_err(), "w: empty batch");
        assert_eq!(
            check_batch_shape("w", 5, 4, true).unwrap_err(),
            "w: 5 sequences exceeds the batch limit 4"
        );
        assert_eq!(
            check_batch_shape("w", 2, 4, true).unwrap_err(),
            "w: trace takes one sequence, got 2"
        );
        assert!(check_batch_shape("w", 2, 4, false).is_ok());
        assert!(check_batch_shape("w", 1, 4, true).is_ok());
        assert!(check_batch_shape("w", 4, 4, false).is_ok());
    }
}
