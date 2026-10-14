use anndists::dist::{DistCosine, DistL2, Distance};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Copy, Debug)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum DistanceMetric{
    L2,
    Cosine

}

impl DistanceMetric {
    pub fn calculate(&self, a: &[f32], b: &[f32]) -> f32{
        match self{
            DistanceMetric::L2 => DistL2.eval(a,b),
            DistanceMetric::Cosine => DistCosine.eval(a,b)
        }
    }
}

