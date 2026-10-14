use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

//Candidate Vector
#[derive(Clone, Copy, Debug)]
pub struct Candidate {
    pub id: u32,
    pub distance: f32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance && self.id == other.id
    }
}
impl Eq for Candidate {}

impl Ord for Candidate { // Ordering for max heap
    fn cmp(&self, other: &Self) -> Ordering {
        if self.distance < other.distance {
            Ordering::Less
        } else if self.distance > other.distance {
            Ordering::Greater
        } else {
            if self.id < other.id {
                Ordering::Less
            } else if self.id > other.id {
                Ordering::Greater
            } else {
                Ordering::Equal
            }
        }

    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other)) // use normal ordering
    }
}
//-----------------------------------------------
//-----------------------------------------------


pub fn greedy_search<FDist, FEdges>(start_node: u32, query_vector: &[f32], beam_size_l: usize, mut compute_dist: FDist, mut get_edges: FEdges) 
                                                                                                                            -> Vec<Candidate>
where
    FDist: FnMut(u32, &[f32]) -> f32,
    FEdges: FnMut(u32) -> Vec<u32>,
{
    let mut visited = HashSet::new();
    let mut candidates = BinaryHeap::new(); // size l

    let start_dist = compute_dist(start_node, query_vector);
    let start_cand = Candidate { id: start_node, distance: start_dist };

    visited.insert(start_node);
    candidates.push(start_cand);

    let mut unexpanded = vec![start_cand];

    while !unexpanded.is_empty() {

        // Find closest candidate
        unexpanded.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        let current = unexpanded.remove(0);

        // Terminate if curr best worse than worst in candidadtes
        if candidates.len() >= beam_size_l && current.distance > candidates.peek().unwrap().distance {break;}

        for neighbor in get_edges(current.id) {
            if visited.insert(neighbor) {
                let dist = compute_dist(neighbor, query_vector);
                let cand = Candidate { id: neighbor, distance: dist };

                if candidates.len() < beam_size_l {
                    candidates.push(cand);
                    unexpanded.push(cand);
                } else if dist < candidates.peek().unwrap().distance {
                    candidates.pop();
                    candidates.push(cand);
                    unexpanded.push(cand);
                }
            }
        }
    }

    candidates.into_sorted_vec() // Return sorted
}

pub fn robust_prune<FDistPair>( candidate_pool: &[Candidate], r: usize, alpha: f32, mut dist_between_nodes: FDistPair) -> Vec<u32>
where
    FDistPair: FnMut(u32, u32) -> f32
{
    let mut pool = candidate_pool.to_vec();
    pool.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());

    let mut pruned: Vec<u32> = Vec::with_capacity(r);
    for p in pool {
        if pruned.len() >= r {break;}

        let mut keep = true;
        for &selected_node in &pruned {

            //Check if we need to actually make this bridge
            let dist_check = dist_between_nodes(p.id, selected_node);
            if alpha * dist_check <= p.distance {
                keep = false;
                break;
            }
        }

        if keep {
            pruned.push(p.id);
        }
    }
    
    //return
    pruned
}
