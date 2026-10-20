//! Checked packed token rows and passage-local bidirectional attention.

use std::ops::Range;
use ndarray::{s, Array2};

/// A contiguous partition of packed rows. Nonempty passages cannot overlap or leave gaps.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PassageSpans {
    ranges: Vec<Range<usize>>,
    rows: usize,
}

impl PassageSpans {
    pub fn from_lengths(lengths: &[usize], max_passage: usize, capacity: usize) -> Result<Self, String> {
        if max_passage == 0 || capacity < max_passage {
            return Err(format!("invalid passage limit {max_passage} / row capacity {capacity}"));
        }
        let mut rows = 0usize;
        let mut ranges = Vec::with_capacity(lengths.len());
        for (i, &len) in lengths.iter().enumerate() {
            if len == 0 || len > max_passage {
                return Err(format!("passage {i} has {len} tokens; expected 1..={max_passage}"));
            }
            let end = rows.checked_add(len).ok_or("packed row count overflow")?;
            if end > capacity {
                return Err(format!("packed rows {end} exceed capacity {capacity}"));
            }
            ranges.push(rows..end);
            rows = end;
        }
        Ok(Self { ranges, rows })
    }

    pub fn ranges(&self) -> &[Range<usize>] { &self.ranges }
    pub fn rows(&self) -> usize { self.rows }
    pub fn positions(&self) -> impl Iterator<Item = usize> + '_ {
        self.ranges.iter().flat_map(|r| 0..r.len())
    }
}

/// Partition documents, never a passage, into bounded packed-row batches in input order.
pub fn passage_batches(lengths: &[usize], max_passage: usize, capacity: usize) -> Result<Vec<Range<usize>>, String> {
    // Validate the full request before executing any prefix of it.
    for &len in lengths {
        PassageSpans::from_lengths(&[len], max_passage, capacity)?;
    }
    PassageSpans::from_lengths(&[], max_passage, capacity)?;
    let mut batches = Vec::new();
    let (mut start, mut rows) = (0, 0usize);
    for (i, &len) in lengths.iter().enumerate() {
        if len > capacity - rows {
            batches.push(start..i);
            start = i;
            rows = 0;
        }
        rows += len;
    }
    if start < lengths.len() { batches.push(start..lengths.len()); }
    Ok(batches)
}

pub fn segmented_mha(q: &Array2<f32>, k: &Array2<f32>, v: &Array2<f32>,
                     n_heads: usize, head_dim: usize, spans: &PassageSpans) -> Result<Array2<f32>, String> {
    if q.dim() != k.dim() || q.dim() != v.dim() || q.nrows() != spans.rows()
        || n_heads == 0 || head_dim == 0 || n_heads.checked_mul(head_dim) != Some(q.ncols()) {
        return Err("segmented attention: incompatible Q/K/V, head dimensions or packed rows".into());
    }
    let mut out = Array2::zeros(q.raw_dim());
    for span in spans.ranges() {
        let qs = q.slice(s![span.clone(), ..]).to_owned();
        let ks = k.slice(s![span.clone(), ..]).to_owned();
        let vs = v.slice(s![span.clone(), ..]).to_owned();
        let ctx = crate::mha(&qs, &ks, &vs, n_heads, head_dim, false, span.len());
        out.slice_mut(s![span.clone(), ..]).assign(&ctx);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn positions_offsets_limits_and_tail_above_512() {
        let spans = PassageSpans::from_lengths(&[300, 301, 2], 512, 1024).unwrap();
        assert_eq!(spans.ranges(), &[0..300, 300..601, 601..603]);
        let positions: Vec<_> = spans.positions().collect();
        assert_eq!(&positions[298..303], &[298, 299, 0, 1, 2]);
        assert_eq!(&positions[601..], &[0, 1]);
        assert_eq!(passage_batches(&[512, 512, 3, 511, 2], 512, 1024).unwrap(), vec![0..2, 2..5]);
        assert_eq!(passage_batches(&[512, 512, 512, 1], 512, 1024).unwrap(), vec![0..2, 2..4]);
        assert!(passage_batches(&[], 512, 1024).unwrap().is_empty());
        for bad in [vec![0], vec![513], vec![512, 513]] {
            assert!(passage_batches(&bad, 512, 1024).is_err());
        }
        assert!(PassageSpans::from_lengths(&[512, 1], 512, 512).is_err());
        assert!(PassageSpans::from_lengths(&[], 512, 0).is_err());
        assert!(PassageSpans::from_lengths(&[], 0, 1024).is_err());
    }

    #[test]
    fn numerical_segmented_attention_above_512_with_tail() {
        let spans = PassageSpans::from_lengths(&[257,259,1], 512, 1024).unwrap();
        let q = Array2::from_shape_fn((517,2), |(i,j)| ((i * 3 + j * 7) % 29) as f32 / 29.);
        let packed = segmented_mha(&q, &q, &q, 1, 2, &spans).unwrap();
        for span in spans.ranges() {
            let x = q.slice(s![span.clone(), ..]).to_owned();
            let single = crate::mha(&x, &x, &x, 1, 2, false, span.len());
            assert_eq!(packed.slice(s![span.clone(), ..]), single);
        }
        assert_eq!(packed.slice(s![516.., ..]), q.slice(s![516.., ..]));
    }

    #[test]
    fn numerical_attention_isolates_neighbors_and_rejects_naive_concatenation() {
        let q = array![[0.7, -0.4], [0.2, 0.9], [-0.3, 0.1], [0.8, 0.5], [0.4, -0.7]];
        let mut changed = q.clone();
        changed.slice_mut(s![2.., ..]).mapv_inplace(|x| x * -7.0 + 4.0);
        let spans = PassageSpans::from_lengths(&[2, 3], 3, 5).unwrap();
        let a = segmented_mha(&q, &q, &q, 1, 2, &spans).unwrap();
        let b = segmented_mha(&changed, &changed, &changed, 1, 2, &spans).unwrap();
        assert_eq!(a.slice(s![..2, ..]), b.slice(s![..2, ..]), "neighbor B must not change A");
        let single = q.slice(s![..2, ..]).to_owned();
        assert_eq!(a.slice(s![..2, ..]), crate::mha(&single, &single, &single, 1, 2, false, 2));
        let naive_a = crate::mha(&q, &q, &q, 1, 2, false, 5);
        let naive_b = crate::mha(&changed, &changed, &changed, 1, 2, false, 5);
        assert!((naive_a[[0, 0]] - naive_b[[0, 0]]).abs() > 0.1, "known-bad control must fail isolation");
        assert!(segmented_mha(&q, &q, &q, 2, 2, &spans).is_err());
        let wrong = PassageSpans::from_lengths(&[2], 3, 5).unwrap();
        assert!(segmented_mha(&q, &q, &q, 1, 2, &wrong).is_err());
    }
}
