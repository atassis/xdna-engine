//! Agglomerative clustering of speaker embeddings.
//!
//! METHOD IS `centroid` (UPGMC), not average linkage: merge the two clusters whose CENTROIDS are
//! closest, recompute the centroid, repeat until the closest pair exceeds the threshold. Centroid
//! linkage is known for non-monotonic merge heights (a merge can be "closer" than one before it),
//! which is why this merges to a stopping rule directly rather than cutting a dendrogram -- there
//! is no monotone height to cut at.
//!
//! Then the shipped pipeline's post-hoc pass: any cluster smaller than `min_cluster_size` is
//! REASSIGNED to its nearest large cluster rather than re-clustered or discarded. Discarding would
//! silently delete speech; re-clustering would move the large clusters it was meant to preserve.
//!
//! NOT proven bit-identical to scipy's `linkage(method="centroid")` merge order. The parity probe
//! adjudicates that; see the design spec's risk 1.

use ndarray::Array2;

/// Cosine distance, `1 - cos`. Embeddings arrive L2-normalised, but a CENTROID of normalised
/// vectors is not itself normalised, so the norms are recomputed rather than assumed.
fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len().min(b.len()) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na <= 0.0 || nb <= 0.0 { return 1.0 }
    1.0 - dot / (na.sqrt() * nb.sqrt())
}

/// Cluster `[n, dim]` embeddings; one label per row.
///
/// `threshold` is the cosine distance above which two clusters are different speakers
/// (`0.7045654963945799` upstream). `min_cluster_size` is the reassignment floor (12 upstream).
pub fn cluster(embeddings: &Array2<f32>, threshold: f32, min_cluster_size: usize) -> Vec<u32> {
    let n = embeddings.nrows();
    if n == 0 { return Vec::new() }
    if n == 1 { return vec![0] }
    let dim = embeddings.ncols();

    let mut members: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
    let mut centroids: Vec<Vec<f32>> =
        (0..n).map(|i| (0..dim).map(|d| embeddings[[i, d]]).collect()).collect();
    let mut alive: Vec<bool> = vec![true; n];

    loop {
        let mut best: Option<(usize, usize, f32)> = None;
        for i in 0..members.len() {
            if !alive[i] { continue }
            for j in (i + 1)..members.len() {
                if !alive[j] { continue }
                let d = cosine_distance(&centroids[i], &centroids[j]);
                if best.map_or(true, |(_, _, bd)| d < bd) { best = Some((i, j, d)); }
            }
        }
        match best {
            Some((i, j, d)) if d <= threshold => {
                let taken = std::mem::take(&mut members[j]);
                members[i].extend(taken);
                alive[j] = false;
                centroids[i] = centroid_of(embeddings, &members[i], dim);
            }
            _ => break,
        }
    }

    let live: Vec<usize> = (0..members.len()).filter(|&i| alive[i]).collect();
    let mut labels = vec![0u32; n];
    for (label, &ci) in live.iter().enumerate() {
        for &row in &members[ci] { labels[row] = label as u32; }
    }
    reassign_small(embeddings, &mut labels, &live, &members, &centroids, min_cluster_size, dim);
    compact(&mut labels);
    labels
}

fn centroid_of(e: &Array2<f32>, rows: &[usize], dim: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; dim];
    if rows.is_empty() { return c }
    for &r in rows {
        for d in 0..dim { c[d] += e[[r, d]]; }
    }
    for d in 0..dim { c[d] /= rows.len() as f32; }
    c
}

/// Move every member of an undersized cluster to the nearest cluster that IS big enough. No-op when
/// nothing is large enough -- returning small clusters beats collapsing a clip into one speaker.
fn reassign_small(
    e: &Array2<f32>, labels: &mut [u32], live: &[usize], members: &[Vec<usize>],
    centroids: &[Vec<f32>], min_cluster_size: usize, dim: usize,
) {
    let big: Vec<usize> = live.iter().copied()
        .filter(|&ci| members[ci].len() >= min_cluster_size).collect();
    if big.is_empty() { return }
    let big_label: std::collections::HashMap<usize, u32> = live.iter().enumerate()
        .map(|(label, &ci)| (ci, label as u32)).collect();
    for &ci in live {
        if members[ci].len() >= min_cluster_size { continue }
        for &row in &members[ci] {
            let v: Vec<f32> = (0..dim).map(|d| e[[row, d]]).collect();
            let mut best = big[0];
            let mut best_d = f32::INFINITY;
            for &bi in &big {
                let d = cosine_distance(&v, &centroids[bi]);
                if d < best_d { best_d = d; best = bi; }
            }
            labels[row] = big_label[&best];
        }
    }
}

/// Renumber labels to 0..k with no gaps, so `n_clusters` is the max + 1 downstream.
fn compact(labels: &mut [u32]) {
    let mut map = std::collections::BTreeMap::new();
    for l in labels.iter() {
        let next = map.len() as u32;
        map.entry(*l).or_insert(next);
    }
    for l in labels.iter_mut() { *l = map[l]; }
}

/// pyannote 3.1's clusterer, behind the trait.
pub struct AgglomerativeClusterer {
    pub threshold: f32,
    pub min_cluster_size: usize,
}

impl crate::diarize::types::Clusterer for AgglomerativeClusterer {
    fn cluster(&self, embeddings: &ndarray::Array2<f32>) -> Result<Vec<u32>, crate::api::EngineError> {
        Ok(cluster(embeddings, self.threshold, self.min_cluster_size))
    }
}

/// Number of distinct clusters in a label vector.
pub fn n_clusters(labels: &[u32]) -> usize {
    labels.iter().collect::<std::collections::BTreeSet<_>>().len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn two_tight_groups_split_at_the_threshold() {
        let e = arr2(&[[1.0f32, 0.0], [0.99, 0.14], [0.0, 1.0], [0.14, 0.99]]);
        let l = cluster(&e, 0.5, 1);
        assert_eq!(l[0], l[1], "near-identical vectors must share a cluster");
        assert_eq!(l[2], l[3]);
        assert_ne!(l[0], l[2], "orthogonal vectors must not");
        assert_eq!(n_clusters(&l), 2);
    }

    #[test]
    fn a_generous_threshold_collapses_everything_to_one() {
        let e = arr2(&[[1.0f32, 0.0], [0.0, 1.0], [-1.0, 0.0]]);
        assert_eq!(n_clusters(&cluster(&e, 10.0, 1)), 1);
    }

    #[test]
    fn a_strict_threshold_leaves_every_point_alone() {
        let e = arr2(&[[1.0f32, 0.0], [0.0, 1.0], [-1.0, 0.0]]);
        assert_eq!(n_clusters(&cluster(&e, -1.0, 1)), 3);
    }

    #[test]
    fn clusters_below_min_size_are_reassigned_not_deleted() {
        let e = arr2(&[[1.0f32, 0.0], [0.99, 0.1], [0.98, 0.15], [0.0, 1.0]]);
        let l = cluster(&e, 0.3, 2);
        assert_eq!(n_clusters(&l), 1, "the singleton is absorbed");
        assert_eq!(l.len(), 4, "reassignment must not drop points");
        assert!(l.iter().all(|&c| c == l[0]));
    }

    #[test]
    fn reassignment_is_a_noop_when_every_cluster_is_big_enough() {
        let e = arr2(&[[1.0f32, 0.0], [0.99, 0.14], [0.0, 1.0], [0.14, 0.99]]);
        assert_eq!(cluster(&e, 0.5, 2), cluster(&e, 0.5, 1));
    }

    #[test]
    fn the_trait_impl_agrees_with_the_free_function() {
        use crate::diarize::types::Clusterer;
        let e = arr2(&[[1.0f32, 0.0], [0.99, 0.14], [0.0, 1.0], [0.14, 0.99]]);
        let c = AgglomerativeClusterer { threshold: 0.5, min_cluster_size: 1 };
        assert_eq!(c.cluster(&e).unwrap(), cluster(&e, 0.5, 1),
            "the seam must not change behaviour -- it only moves the call site");
    }

    #[test]
    fn an_empty_input_is_not_a_panic() {
        let e = ndarray::Array2::<f32>::zeros((0, 256));
        assert!(cluster(&e, 0.7, 12).is_empty());
    }

    #[test]
    fn labels_are_compact_so_n_clusters_is_the_max_plus_one() {
        let e = arr2(&[[1.0f32, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]);
        let l = cluster(&e, 0.1, 1);
        let max = *l.iter().max().unwrap() as usize;
        assert_eq!(max + 1, n_clusters(&l), "gaps in labels break downstream sizing: {l:?}");
    }
}
