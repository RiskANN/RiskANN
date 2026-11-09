//! Deterministic synthetic datasets for tests and benchmarks.
//!
//! The generator uses SplitMix64 implemented here, so the exact same vectors
//! are produced for a given spec on every platform and every Rust version,
//! independent of any third-party RNG crate's version.

use std::fmt;

/// SplitMix64 pseudo-random generator (Steele, Lea & Flood, 2014).
#[derive(Debug, Clone)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)` with 53 bits of precision.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Standard normal sample (Box–Muller, one value per call).
    pub fn next_gaussian(&mut self) -> f64 {
        let u1 = 1.0 - self.next_f64(); // (0, 1]: ln(u1) is finite
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    pub fn below(&mut self, n: u64) -> u64 {
        if n == 0 { 0 } else { self.next_u64() % n }
    }
}

/// Shape of the synthetic data.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Distribution {
    /// Independent uniform components in `[-1, 1)`.
    Uniform,
    /// Independent standard-normal components.
    Gaussian,
    /// Gaussian mixture: `clusters` centres drawn from N(0, 3^2) per
    /// component, points drawn from N(centre, 1).
    Clustered { clusters: usize },
}

impl Distribution {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "uniform" => Some(Distribution::Uniform),
            "gaussian" => Some(Distribution::Gaussian),
            "clustered" => Some(Distribution::Clustered { clusters: 32 }),
            other => {
                let n = other.strip_prefix("clustered:")?.parse().ok()?;
                if n == 0 {
                    None
                } else {
                    Some(Distribution::Clustered { clusters: n })
                }
            }
        }
    }
}

impl fmt::Display for Distribution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Distribution::Uniform => write!(f, "uniform"),
            Distribution::Gaussian => write!(f, "gaussian"),
            Distribution::Clustered { clusters } => write!(f, "clustered:{clusters}"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DatasetSpec {
    pub n: usize,
    pub dim: usize,
    pub seed: u64,
    pub distribution: Distribution,
}

/// Base vectors plus held-out queries from the same distribution.
#[derive(Debug, Clone)]
pub struct Dataset {
    pub base: Vec<Vec<f32>>,
    pub queries: Vec<Vec<f32>>,
}

const STREAM_CENTRES: u64 = 0x0001;
const STREAM_BASE: u64 = 0x0002;
const STREAM_QUERIES: u64 = 0x0003;

fn stream(seed: u64, s: u64) -> SplitMix64 {
    // Mix the stream id into the seed so streams are independent.
    let mut m = SplitMix64::new(seed ^ s.wrapping_mul(0xD6E8_FEB8_6659_FD93));
    SplitMix64::new(m.next_u64())
}

fn sample(rng: &mut SplitMix64, dist: Distribution, centres: &[Vec<f64>], dim: usize) -> Vec<f32> {
    match dist {
        Distribution::Uniform => (0..dim)
            .map(|_| (rng.next_f64() * 2.0 - 1.0) as f32)
            .collect(),
        Distribution::Gaussian => (0..dim).map(|_| rng.next_gaussian() as f32).collect(),
        Distribution::Clustered { .. } => {
            let c = &centres[rng.below(centres.len() as u64) as usize];
            c.iter()
                .map(|&m| (m + rng.next_gaussian()) as f32)
                .collect()
        }
    }
}

/// Generates `spec.n` base vectors and `n_queries` query vectors.
/// Queries are drawn independently of the base set (they are not copies of
/// base vectors), from the same distribution and the same cluster centres.
pub fn generate(spec: &DatasetSpec, n_queries: usize) -> Dataset {
    let centres: Vec<Vec<f64>> = match spec.distribution {
        Distribution::Clustered { clusters } => {
            let mut rng = stream(spec.seed, STREAM_CENTRES);
            (0..clusters.max(1))
                .map(|_| (0..spec.dim).map(|_| 3.0 * rng.next_gaussian()).collect())
                .collect()
        }
        _ => Vec::new(),
    };
    let mut base_rng = stream(spec.seed, STREAM_BASE);
    let base = (0..spec.n)
        .map(|_| sample(&mut base_rng, spec.distribution, &centres, spec.dim))
        .collect();
    let mut query_rng = stream(spec.seed, STREAM_QUERIES);
    let queries = (0..n_queries)
        .map(|_| sample(&mut query_rng, spec.distribution, &centres, spec.dim))
        .collect();
    Dataset { base, queries }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitmix_reference_values() {
        // Reference outputs of SplitMix64 for seed 0 (widely published).
        let mut r = SplitMix64::new(0);
        assert_eq!(r.next_u64(), 0xE220_A839_7B1D_CDAF);
        assert_eq!(r.next_u64(), 0x6E78_9E6A_A1B9_65F4);
    }

    #[test]
    fn generation_is_deterministic_and_seed_sensitive() {
        let spec = DatasetSpec {
            n: 50,
            dim: 8,
            seed: 42,
            distribution: Distribution::Clustered { clusters: 4 },
        };
        let a = generate(&spec, 5);
        let b = generate(&spec, 5);
        assert_eq!(a.base, b.base);
        assert_eq!(a.queries, b.queries);
        let c = generate(&DatasetSpec { seed: 43, ..spec }, 5);
        assert_ne!(a.base, c.base);
        assert_eq!(a.base.len(), 50);
        assert!(
            a.base
                .iter()
                .all(|v| v.len() == 8 && v.iter().all(|x| x.is_finite()))
        );
    }

    #[test]
    fn gaussian_moments_are_plausible() {
        let mut r = SplitMix64::new(7);
        let n = 20_000;
        let xs: Vec<f64> = (0..n).map(|_| r.next_gaussian()).collect();
        let mean = xs.iter().sum::<f64>() / n as f64;
        let var = xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n as f64;
        assert!(mean.abs() < 0.05, "mean {mean}");
        assert!((var - 1.0).abs() < 0.05, "var {var}");
    }

    #[test]
    fn distribution_parsing() {
        assert_eq!(Distribution::parse("uniform"), Some(Distribution::Uniform));
        assert_eq!(
            Distribution::parse("clustered:8"),
            Some(Distribution::Clustered { clusters: 8 })
        );
        assert_eq!(Distribution::parse("clustered:0"), None);
        assert_eq!(Distribution::parse("zipf"), None);
    }
}
