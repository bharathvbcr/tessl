//! One float-check home for tessl, ojas, and sparsl (card `ft-29323ca4`).
//!
//! This crate plants non-finite values, checks a subnormal, derives the
//! reduction tolerance `4 · f32::EPSILON · √N · scale`, compares an f32 result
//! with an fp64 oracle, and checks that an arithmetic-preserving pair is
//! bitwise equal. A NaN or an infinity fails those checks.
//!
//! It does not replace sparsl's SpMV and scan bounds (`8 · ε · N`) or the
//! model fixtures in `ojas-oracle`. Tessl's `reduction_bound` delegates here
//! so the `4ε√N` formula has one body.

#![forbid(unsafe_code)]

use std::fmt;

/// A value planted into an input to see whether a result stays honest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Plant {
    /// `f32::NAN`.
    Nan,
    /// `f32::INFINITY`.
    PosInf,
    /// `f32::NEG_INFINITY`.
    NegInf,
    /// Smallest positive subnormal, bit pattern `0x0000_0001`.
    Subnormal,
}

/// Which side of a comparison failed.
///
/// For [`compare_oracle`], left is the f32 result and right is the fp64 oracle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    Left,
    Right,
}

/// Why a planted or compared value is not a finite f32 result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NonFinite {
    Nan,
    PosInf,
    NegInf,
}

/// A check that refused to pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CheckError {
    /// An empty slice is not a passing comparison.
    Empty,
    /// The two slices have different lengths.
    LenMismatch { left: usize, right: usize },
    /// [`plant_at`] was given an index outside the buffer.
    OutOfRange { index: usize, len: usize },
    /// A NaN or an infinity. Those never pass a tolerance or a bitwise check.
    NonFinite { index: usize, kind: NonFinite, side: Side },
    /// `|got - oracle|` is greater than [`reduction_tolerance`].
    OutsideTolerance { index: usize },
    /// Finite values whose f32 bit patterns differ.
    BitsDiffer { index: usize },
    /// The tolerance was negative or non-finite, so nothing may pass.
    ToleranceNotUsable,
    /// [`check_subnormal`] saw a finite value that is not subnormal.
    NotSubnormal,
}

impl fmt::Display for Side {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Side::Left => write!(f, "left"),
            Side::Right => write!(f, "right"),
        }
    }
}

impl fmt::Display for NonFinite {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NonFinite::Nan => write!(f, "NaN"),
            NonFinite::PosInf => write!(f, "+Inf"),
            NonFinite::NegInf => write!(f, "-Inf"),
        }
    }
}

impl fmt::Display for CheckError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CheckError::Empty => write!(f, "empty input is not a passing check"),
            CheckError::LenMismatch { left, right } => {
                write!(f, "length mismatch: left {left}, right {right}")
            }
            CheckError::OutOfRange { index, len } => {
                write!(f, "plant index {index} is outside length {len}")
            }
            CheckError::NonFinite { index, kind, side } => {
                write!(f, "index {index} on the {side} is {kind}")
            }
            CheckError::OutsideTolerance { index } => {
                write!(f, "index {index} is outside the 4ε√N tolerance")
            }
            CheckError::BitsDiffer { index } => {
                write!(f, "index {index} bits differ")
            }
            CheckError::ToleranceNotUsable => {
                write!(f, "tolerance is negative or non-finite")
            }
            CheckError::NotSubnormal => write!(f, "value is not a subnormal"),
        }
    }
}

impl std::error::Error for CheckError {}

/// The f32 value `plant` names.
#[must_use]
pub fn plant_f32(plant: Plant) -> f32 {
    match plant {
        Plant::Nan => f32::NAN,
        Plant::PosInf => f32::INFINITY,
        Plant::NegInf => f32::NEG_INFINITY,
        Plant::Subnormal => f32::from_bits(1),
    }
}

/// Write [`plant_f32`] into `buf[index]`.
///
/// # Errors
///
/// [`CheckError::OutOfRange`] when `index` is past the end of `buf`.
pub fn plant_at(buf: &mut [f32], index: usize, plant: Plant) -> Result<(), CheckError> {
    let len = buf.len();
    let slot = buf.get_mut(index).ok_or(CheckError::OutOfRange { index, len })?;
    *slot = plant_f32(plant);
    Ok(())
}

/// Passes only when `value` is a finite subnormal.
///
/// # Errors
///
/// [`CheckError::NonFinite`] for a NaN or an infinity, and
/// [`CheckError::NotSubnormal`] for zero and for normal numbers.
pub fn check_subnormal(value: f32) -> Result<(), CheckError> {
    if value.is_subnormal() {
        Ok(())
    } else if let Some(kind) = non_finite_f32(value) {
        Err(CheckError::NonFinite {
            index: 0,
            kind,
            side: Side::Left,
        })
    } else {
        Err(CheckError::NotSubnormal)
    }
}

/// `4 · f32::EPSILON · √n · scale`.
///
/// This is the whole formula. Callers compare with `abs(error) <=` this value
/// and do not widen it. A negative or non-finite `scale` is returned as the
/// arithmetic produces it; [`compare_oracle`] then refuses to pass anything.
#[must_use]
pub fn reduction_tolerance(n: usize, scale: f64) -> f64 {
    4.0 * f64::from(f32::EPSILON) * (n as f64).sqrt() * scale
}

/// Compare an f32 result with an fp64 oracle under [`reduction_tolerance`].
///
/// A NaN or an infinity on either side fails, including a NaN that would
/// otherwise sit inside a wide tolerance. Lengths must match, and an empty
/// pair does not pass.
///
/// # Errors
///
/// See [`CheckError`]. `terms` and `scale` are the tolerance inputs; they are
/// not inferred from the slice length.
pub fn compare_oracle(got: &[f32], oracle: &[f64], terms: usize, scale: f64) -> Result<(), CheckError> {
    if got.is_empty() || oracle.is_empty() {
        return Err(CheckError::Empty);
    }
    if got.len() != oracle.len() {
        return Err(CheckError::LenMismatch {
            left: got.len(),
            right: oracle.len(),
        });
    }
    let tol = reduction_tolerance(terms, scale);
    if !tol.is_finite() || tol < 0.0 {
        return Err(CheckError::ToleranceNotUsable);
    }
    for (index, (got_v, oracle_v)) in got.iter().copied().zip(oracle.iter().copied()).enumerate() {
        if let Some(kind) = non_finite_f32(got_v) {
            return Err(CheckError::NonFinite {
                index,
                kind,
                side: Side::Left,
            });
        }
        if let Some(kind) = non_finite_f64(oracle_v) {
            return Err(CheckError::NonFinite {
                index,
                kind,
                side: Side::Right,
            });
        }
        let diff = (f64::from(got_v) - oracle_v).abs();
        if !diff.is_finite() || diff > tol {
            return Err(CheckError::OutsideTolerance { index });
        }
    }
    Ok(())
}

/// Bitwise equality of an arithmetic-preserving pair.
///
/// Same bit pattern, including the sign of zero. A NaN or an infinity fails
/// even when both sides carry the same bits: a non-finite result is not a pass.
///
/// # Errors
///
/// See [`CheckError`]. An empty pair does not pass.
pub fn preserving_equal(left: &[f32], right: &[f32]) -> Result<(), CheckError> {
    if left.is_empty() || right.is_empty() {
        return Err(CheckError::Empty);
    }
    if left.len() != right.len() {
        return Err(CheckError::LenMismatch {
            left: left.len(),
            right: right.len(),
        });
    }
    for (index, (left_v, right_v)) in left.iter().copied().zip(right.iter().copied()).enumerate() {
        if let Some(kind) = non_finite_f32(left_v) {
            return Err(CheckError::NonFinite {
                index,
                kind,
                side: Side::Left,
            });
        }
        if let Some(kind) = non_finite_f32(right_v) {
            return Err(CheckError::NonFinite {
                index,
                kind,
                side: Side::Right,
            });
        }
        if left_v.to_bits() != right_v.to_bits() {
            return Err(CheckError::BitsDiffer { index });
        }
    }
    Ok(())
}

fn non_finite_f32(value: f32) -> Option<NonFinite> {
    if value.is_nan() {
        Some(NonFinite::Nan)
    } else if value == f32::INFINITY {
        Some(NonFinite::PosInf)
    } else if value == f32::NEG_INFINITY {
        Some(NonFinite::NegInf)
    } else {
        None
    }
}

fn non_finite_f64(value: f64) -> Option<NonFinite> {
    if value.is_nan() {
        Some(NonFinite::Nan)
    } else if value == f64::INFINITY {
        Some(NonFinite::PosInf)
    } else if value == f64::NEG_INFINITY {
        Some(NonFinite::NegInf)
    } else {
        None
    }
}
