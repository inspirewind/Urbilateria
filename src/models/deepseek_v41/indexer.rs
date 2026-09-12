//! Scalar reference selection for the two-level CSA2 indexer.

use std::cmp::Ordering;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexerError(String);

impl fmt::Display for IndexerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "invalid DeepSeek-V4.1 index selection: {}",
            self.0
        )
    }
}

impl std::error::Error for IndexerError {}

/// Selects candidate blocks using their maximum position score. The newest partially visible
/// block is pinned, exactly as in the release implementation.
pub fn select_candidate_blocks(
    scores: &[f32],
    visible: usize,
    topk_blocks: usize,
    block_size: usize,
) -> Result<Vec<bool>, IndexerError> {
    validate_scores(scores, visible)?;
    if block_size == 0 || topk_blocks == 0 {
        return Err(IndexerError(
            "candidate block size and count must be non-zero".to_owned(),
        ));
    }
    if visible == 0 {
        return Ok(vec![false; scores.len()]);
    }
    let blocks = scores.len().div_ceil(block_size);
    let newest = (visible - 1) / block_size;
    let mut block_scores = (0..blocks)
        .map(|block| {
            let start = block * block_size;
            let end = (start + block_size).min(scores.len()).min(visible);
            let score = if block == newest {
                f32::INFINITY
            } else if start >= end {
                f32::NEG_INFINITY
            } else {
                scores[start..end]
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, f32::max)
            };
            (block, score)
        })
        .collect::<Vec<_>>();
    sort_best_first(&mut block_scores);
    let mut keep = vec![false; scores.len()];
    for &(block, score) in block_scores.iter().take(topk_blocks.min(blocks)) {
        if score == f32::NEG_INFINITY {
            continue;
        }
        let start = block * block_size;
        let end = (start + block_size).min(scores.len());
        keep[start..end].fill(true);
    }
    Ok(keep)
}

/// Applies visibility/candidate masks, takes top-k by score, then returns positions in causal
/// order. `None` represents the release's `-1` filler for unreachable prefill choices.
pub fn select_positions(
    scores: &[f32],
    visible: usize,
    topk: usize,
    output_offset: usize,
    candidates: Option<&[bool]>,
) -> Result<Vec<Option<usize>>, IndexerError> {
    validate_scores(scores, visible)?;
    if candidates.is_some_and(|mask| mask.len() != scores.len()) {
        return Err(IndexerError(
            "candidate mask length differs from index score width".to_owned(),
        ));
    }
    let count = topk.min(scores.len());
    let mut ranked = scores
        .iter()
        .copied()
        .enumerate()
        .map(|(position, score)| {
            let reachable = position < visible && candidates.map_or(true, |mask| mask[position]);
            (position, if reachable { score } else { f32::NEG_INFINITY })
        })
        .collect::<Vec<_>>();
    sort_best_first(&mut ranked);
    let mut chosen = ranked.into_iter().take(count).collect::<Vec<_>>();
    chosen.sort_unstable_by_key(|(position, _)| *position);
    chosen
        .into_iter()
        .map(|(position, score)| {
            if score == f32::NEG_INFINITY || position >= visible {
                Ok(None)
            } else {
                Ok(Some(position.checked_add(output_offset).ok_or_else(
                    || IndexerError("selected position plus output offset overflows".to_owned()),
                )?))
            }
        })
        .collect()
}

fn validate_scores(scores: &[f32], visible: usize) -> Result<(), IndexerError> {
    if visible > scores.len() || scores.iter().any(|score| score.is_nan()) {
        return Err(IndexerError(format!(
            "visible={visible} must not exceed width {}, and scores must not contain NaN",
            scores.len()
        )));
    }
    Ok(())
}

fn sort_best_first(values: &mut [(usize, f32)]) {
    values.sort_unstable_by(|(left_index, left), (right_index, right)| {
        right
            .partial_cmp(left)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left_index.cmp(right_index))
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candidate_selection_pins_newest_partial_block() {
        let scores = [10.0, 9.0, 1.0, 1.0, -5.0, -6.0, 100.0, 100.0];
        let keep = select_candidate_blocks(&scores, 6, 2, 2).unwrap();
        assert_eq!(
            keep,
            vec![true, true, false, false, true, true, false, false]
        );
    }

    #[test]
    fn final_topk_is_sorted_by_position_and_marks_unreachable_prefill_slots() {
        let selected = select_positions(&[1.0, 4.0, 3.0, 99.0], 3, 4, 128, None).unwrap();
        assert_eq!(selected, vec![Some(128), Some(129), Some(130), None]);
    }

    #[test]
    fn candidate_mask_is_applied_before_topk() {
        let selected = select_positions(
            &[9.0, 8.0, 7.0, 6.0],
            4,
            2,
            0,
            Some(&[false, true, false, true]),
        )
        .unwrap();
        assert_eq!(selected, vec![Some(1), Some(3)]);
    }
}
