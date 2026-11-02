//! Distance metrics.
//!
//! # Conventions (smaller is always better)
//!
//! * [`DistanceMetric::L2`] returns the **Euclidean** distance
//!   `sqrt(sum_i (a_i - b_i)^2)`, not the squared distance.
//! * [`DistanceMetric::Cosine`] returns the **cosine distance**
//!   `1 - cos(a, b)`, clamped to `[0, 2]`. Identical directions give `0`,
//!   orthogonal vectors give `1`, opposite vectors give `2`.
//!   If either vector has zero norm the cosine is undefined; this function then
//!   returns `1.0` (treated as orthogonal). The engine rejects zero-norm
//!   vectors for cosine collections at insert and query time, so this fallback
//!   is only a defensive default and is not reachable through the public API.
//!
//! Accumulation is performed in `f64` and the result is converted to `f32`.
//! The same functions are used for graph construction, approximate search and
//! exact search, so all three paths rank vectors identically.

use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum DistanceMetric {
    #[serde(alias = "l2")]
    L2,
    #[serde(alias = "cosine", alias = "Cosine")]
    Cosine,
}

/// Errors returned by [`DistanceMetric::try_calculate`].
#[derive(Debug, Clone, PartialEq)]
pub enum DistanceError {
    DimensionMismatch { left: usize, right: usize },
    NonFinite,
}

impl fmt::Display for DistanceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DistanceError::DimensionMismatch { left, right } => {
                write!(f, "dimension mismatch: {left} vs {right}")
            }
            DistanceError::NonFinite => write!(f, "distance is not a finite number"),
        }
    }
}

impl std::error::Error for DistanceError {}

impl DistanceMetric {
    /// Stable name used in API responses (`"L2"` or `"COSINE"`).
    pub fn name(&self) -> &'static str {
        match self {
            DistanceMetric::L2 => "L2",
            DistanceMetric::Cosine => "COSINE",
        }
    }

    /// Parses `"L2"`/`"l2"` and `"COSINE"`/`"cosine"`.
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "l2" => Some(DistanceMetric::L2),
            "cosine" => Some(DistanceMetric::Cosine),
            _ => None,
        }
    }

    /// Distance between `a` and `b`.
    ///
    /// Precondition: `a.len() == b.len()`. Callers inside RiskANN guarantee
    /// this by validating every vector against the collection dimension. Use
    /// [`DistanceMetric::try_calculate`] when the precondition is not known to
    /// hold.
    #[inline]
    pub fn calculate(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len(), "distance on vectors of different length");
        match self {
            DistanceMetric::L2 => l2_distance(a, b),
            DistanceMetric::Cosine => cosine_distance(a, b),
        }
    }

    /// Checked distance: verifies equal lengths and a finite result.
    pub fn try_calculate(&self, a: &[f32], b: &[f32]) -> Result<f32, DistanceError> {
        if a.len() != b.len() {
            return Err(DistanceError::DimensionMismatch {
                left: a.len(),
                right: b.len(),
            });
        }
        let d = self.calculate(a, b);
        if d.is_finite() {
            Ok(d)
        } else {
            Err(DistanceError::NonFinite)
        }
    }
}

/// Euclidean (not squared) distance, accumulated in `f64`.
#[inline]
pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    // Four independent accumulators give the CPU instruction-level
    // parallelism. The summation order is fixed, so results are deterministic.
    let mut acc = [0.0f64; 4];
    let ca = a.chunks_exact(4);
    let cb = b.chunks_exact(4);
    let ra = ca.remainder();
    let rb = cb.remainder();
    for (x, y) in ca.zip(cb) {
        for ((slot, &xv), &yv) in acc.iter_mut().zip(x).zip(y) {
            let d = xv as f64 - yv as f64;
            *slot += d * d;
        }
    }
    let mut sum = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    for (x, y) in ra.iter().zip(rb) {
        let d = *x as f64 - *y as f64;
        sum += d * d;
    }
    sum.sqrt() as f32
}

/// Squared L2 norm accumulated in `f64`.
#[inline]
pub fn squared_norm(a: &[f32]) -> f64 {
    a.iter().map(|&x| (x as f64) * (x as f64)).sum()
}

/// Cosine distance `1 - cos(a, b)`, clamped to `[0, 2]`.
/// Returns `1.0` when either vector has zero norm (see module docs).
#[inline]
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (x, y) in a.iter().zip(b) {
        let x = *x as f64;
        let y = *y as f64;
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        return 1.0;
    }
    let sim = (dot / (na.sqrt() * nb.sqrt())).clamp(-1.0, 1.0);
    (1.0 - sim) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-6
    }

    #[test]
    fn l2_known_values() {
        assert!(close(l2_distance(&[0.0, 0.0], &[3.0, 4.0]), 5.0));
        assert!(close(l2_distance(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]), 0.0));
        // dim 5 exercises both the chunked path and the remainder path.
        let a = [1.0, 1.0, 1.0, 1.0, 1.0];
        let b = [0.0, 0.0, 0.0, 0.0, 0.0];
        assert!(close(l2_distance(&a, &b), 5.0f32.sqrt()));
    }

    #[test]
    fn l2_is_euclidean_not_squared() {
        // The squared distance would be 4.0.
        assert!(close(DistanceMetric::L2.calculate(&[0.0], &[2.0]), 2.0));
    }

    #[test]
    fn l2_is_symmetric() {
        let a = [0.3, -1.2, 4.5, 0.0, 9.9, -0.1, 2.2];
        let b = [1.3, 0.2, -4.5, 1.0, 0.9, 0.1, -2.2];
        assert_eq!(l2_distance(&a, &b), l2_distance(&b, &a));
    }

    #[test]
    fn cosine_known_values() {
        let m = DistanceMetric::Cosine;
        assert!(close(m.calculate(&[1.0, 0.0], &[2.0, 0.0]), 0.0)); // same direction
        assert!(close(m.calculate(&[1.0, 0.0], &[0.0, 3.0]), 1.0)); // orthogonal
        assert!(close(m.calculate(&[1.0, 0.0], &[-1.0, 0.0]), 2.0)); // opposite
        let d = m.calculate(&[1.0, 1.0], &[1.0, 0.0]);
        assert!(close(d, 1.0 - std::f32::consts::FRAC_1_SQRT_2));
    }

    #[test]
    fn cosine_identical_vectors_is_zero_not_negative() {
        let v = [0.1f32, 0.7, -0.3, 1e-3, 5.0];
        let d = cosine_distance(&v, &v);
        assert!((0.0..1e-6).contains(&d), "{d}");
    }

    #[test]
    fn cosine_zero_vector_is_defined_as_one() {
        assert_eq!(cosine_distance(&[0.0, 0.0], &[1.0, 2.0]), 1.0);
        assert_eq!(cosine_distance(&[0.0, 0.0], &[0.0, 0.0]), 1.0);
    }

    #[test]
    fn try_calculate_rejects_dimension_mismatch() {
        let e = DistanceMetric::L2
            .try_calculate(&[1.0, 2.0], &[1.0])
            .unwrap_err();
        assert_eq!(e, DistanceError::DimensionMismatch { left: 2, right: 1 });
    }

    #[test]
    fn try_calculate_rejects_non_finite() {
        let m = DistanceMetric::L2;
        assert_eq!(
            m.try_calculate(&[f32::NAN], &[0.0]),
            Err(DistanceError::NonFinite)
        );
        assert_eq!(
            m.try_calculate(&[f32::INFINITY], &[0.0]),
            Err(DistanceError::NonFinite)
        );
    }

    #[test]
    fn overflow_boundary_is_reported_not_hidden() {
        // sqrt(8 * (f32::MAX)^2) exceeds f32::MAX, so the distance is +inf and
        // try_calculate reports it instead of returning a bogus value.
        let a = vec![f32::MAX / 2.0; 8];
        let b = vec![-f32::MAX / 2.0; 8];
        assert!(DistanceMetric::L2.try_calculate(&a, &b).is_err());
        // Large but realistic magnitudes stay finite thanks to f64 accumulation.
        let c = vec![1e18f32; 128];
        let d = vec![-1e18f32; 128];
        assert!(DistanceMetric::L2.try_calculate(&c, &d).is_ok());
    }

    #[test]
    fn metric_serde_names_are_backward_compatible() {
        assert_eq!(
            serde_json::to_string(&DistanceMetric::L2).unwrap(),
            "\"L2\""
        );
        assert_eq!(
            serde_json::to_string(&DistanceMetric::Cosine).unwrap(),
            "\"COSINE\""
        );
        let m: DistanceMetric = serde_json::from_str("\"COSINE\"").unwrap();
        assert_eq!(m, DistanceMetric::Cosine);
        let m: DistanceMetric = serde_json::from_str("\"cosine\"").unwrap();
        assert_eq!(m, DistanceMetric::Cosine);
        let m: DistanceMetric = serde_json::from_str("\"l2\"").unwrap();
        assert_eq!(m, DistanceMetric::L2);
        assert!(serde_json::from_str::<DistanceMetric>("\"IP\"").is_err());
    }

    #[test]
    fn parse_names() {
        assert_eq!(DistanceMetric::parse("L2"), Some(DistanceMetric::L2));
        assert_eq!(
            DistanceMetric::parse("Cosine"),
            Some(DistanceMetric::Cosine)
        );
        assert_eq!(DistanceMetric::parse("dot"), None);
    }
}
