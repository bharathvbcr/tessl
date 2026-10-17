//! The eight EXIF orientations on an HWC byte buffer, shared by the image
//! decoder (EXIF tag) and the video decoder (`preferredTransform`).
//!
//! Orientation `o` names how the stored pixels must be transformed to display
//! upright, with PIL's `ImageOps.exif_transpose` as the definition:
//!
//! | o | PIL transpose | destination `(x, y)` reads source |
//! |---|---|---|
//! | 1 | none | `(x, y)` |
//! | 2 | `FLIP_LEFT_RIGHT` | `(w-1-x, y)` |
//! | 3 | `ROTATE_180` | `(w-1-x, h-1-y)` |
//! | 4 | `FLIP_TOP_BOTTOM` | `(x, h-1-y)` |
//! | 5 | `TRANSPOSE` | `(y, x)` |
//! | 6 | `ROTATE_270` (90 degrees clockwise) | `(y, h-1-x)` |
//! | 7 | `TRANSVERSE` | `(w-1-y, h-1-x)` |
//! | 8 | `ROTATE_90` (90 degrees counter-clockwise) | `(w-1-y, x)` |
//!
//! with `w x h` the source size; 5 to 8 swap width and height.

/// The size after applying orientation `o` to a `width x height` image.
pub fn oriented_size(o: u32, width: usize, height: usize) -> (usize, usize) {
    if (5..=8).contains(&o) {
        (height, width)
    } else {
        (width, height)
    }
}

/// `data` (`height` rows of `width` pixels of `channels` bytes) transformed by
/// EXIF orientation `o`; returns the new buffer and its `(width, height)`.
/// Orientations outside `2..=8` return the input unchanged.
pub fn apply(o: u32, data: Vec<u8>, width: usize, height: usize, channels: usize) -> (Vec<u8>, usize, usize) {
    if !(2..=8).contains(&o) {
        return (data, width, height);
    }
    debug_assert_eq!(data.len(), width * height * channels);
    let (tw, th) = oriented_size(o, width, height);
    let mut out = vec![0u8; data.len()];
    for y in 0..th {
        for x in 0..tw {
            let (sx, sy) = match o {
                2 => (width - 1 - x, y),
                3 => (width - 1 - x, height - 1 - y),
                4 => (x, height - 1 - y),
                5 => (y, x),
                6 => (y, height - 1 - x),
                7 => (width - 1 - y, height - 1 - x),
                _ => (width - 1 - y, x),
            };
            let src = (sy * width + sx) * channels;
            let dst = (y * tw + x) * channels;
            out[dst..dst + channels].copy_from_slice(&data[src..src + channels]);
        }
    }
    (out, tw, th)
}

/// The EXIF orientation a `CGAffineTransform` `[a b c d]` (a point `(x, y)`
/// maps to `(a*x + c*y + tx, b*x + d*y + ty)` with `y` down, the convention of
/// `AVAssetTrack.preferredTransform`) stands for. `None` when the linear part
/// is not one of the eight right-angle rotations and flips.
pub fn orientation_of_transform(a: f64, b: f64, c: f64, d: f64) -> Option<u32> {
    let near = |v: f64, t: f64| (v - t).abs() < 1e-3;
    let m = |ea: f64, eb: f64, ec: f64, ed: f64| near(a, ea) && near(b, eb) && near(c, ec) && near(d, ed);
    if m(1.0, 0.0, 0.0, 1.0) {
        Some(1)
    } else if m(-1.0, 0.0, 0.0, 1.0) {
        Some(2)
    } else if m(-1.0, 0.0, 0.0, -1.0) {
        Some(3)
    } else if m(1.0, 0.0, 0.0, -1.0) {
        Some(4)
    } else if m(0.0, 1.0, 1.0, 0.0) {
        Some(5)
    } else if m(0.0, 1.0, -1.0, 0.0) {
        Some(6)
    } else if m(0.0, -1.0, -1.0, 0.0) {
        Some(7)
    } else if m(0.0, -1.0, 1.0, 0.0) {
        Some(8)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 3x2 image whose pixel value is its own index, one channel.
    fn base() -> Vec<u8> {
        vec![0, 1, 2, 3, 4, 5]
    }

    #[test]
    fn the_eight_orientations_move_pixels_as_pil_does() {
        // (orientation, expected rows, (w, h)), from PIL's transpose methods on
        // [[0 1 2], [3 4 5]].
        let cases: [(u32, &[u8], (usize, usize)); 8] = [
            (1, &[0, 1, 2, 3, 4, 5], (3, 2)),
            (2, &[2, 1, 0, 5, 4, 3], (3, 2)),
            (3, &[5, 4, 3, 2, 1, 0], (3, 2)),
            (4, &[3, 4, 5, 0, 1, 2], (3, 2)),
            (5, &[0, 3, 1, 4, 2, 5], (2, 3)),
            (6, &[3, 0, 4, 1, 5, 2], (2, 3)),
            (7, &[5, 2, 4, 1, 3, 0], (2, 3)),
            (8, &[2, 5, 1, 4, 0, 3], (2, 3)),
        ];
        for (o, want, dims) in cases {
            let (got, w, h) = apply(o, base(), 3, 2, 1);
            assert_eq!((w, h), dims, "orientation {o} size");
            assert_eq!(got, want, "orientation {o}");
        }
    }

    #[test]
    fn transform_matrices_map_to_orientations() {
        assert_eq!(orientation_of_transform(1.0, 0.0, 0.0, 1.0), Some(1));
        assert_eq!(orientation_of_transform(0.0, 1.0, -1.0, 0.0), Some(6));
        assert_eq!(orientation_of_transform(0.0, -1.0, 1.0, 0.0), Some(8));
        assert_eq!(orientation_of_transform(-1.0, 0.0, 0.0, -1.0), Some(3));
        assert_eq!(orientation_of_transform(0.7, 0.7, -0.7, 0.7), None);
    }
}
