//! Dynamic Time Warping (DTW) word-level alignment for whisper.rs.

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlignmentHeadsPreset {
    None = 0,
    NTopMost = 1,
    Custom = 2,
    TinyEn = 3,
    Tiny = 4,
    BaseEn = 5,
    Base = 6,
    SmallEn = 7,
    Small = 8,
    MediumEn = 9,
    Medium = 10,
    LargeV1 = 11,
    LargeV2 = 12,
    LargeV3 = 13,
    LargeV3Turbo = 14,
}

/// Preset that matches a loaded model, so callers do not have to hard-code
/// head tables. English-only checkpoints use the `*En` calibration.
pub fn preset_for_model(
    model_type: whisper_model::header::ModelType,
    is_multilingual: bool,
) -> AlignmentHeadsPreset {
    use whisper_model::header::ModelType;
    match (model_type, is_multilingual) {
        (ModelType::Tiny, false) => AlignmentHeadsPreset::TinyEn,
        (ModelType::Tiny, true) => AlignmentHeadsPreset::Tiny,
        (ModelType::Base, false) => AlignmentHeadsPreset::BaseEn,
        (ModelType::Base, true) => AlignmentHeadsPreset::Base,
        (ModelType::Small, false) => AlignmentHeadsPreset::SmallEn,
        (ModelType::Small, true) => AlignmentHeadsPreset::Small,
        (ModelType::Medium, false) => AlignmentHeadsPreset::MediumEn,
        (ModelType::Medium, true) => AlignmentHeadsPreset::Medium,
        (ModelType::Large, _) => AlignmentHeadsPreset::LargeV1,
        (ModelType::LargeV3, _) => AlignmentHeadsPreset::LargeV3,
        (ModelType::LargeV3Turbo, _) => AlignmentHeadsPreset::LargeV3Turbo,
        (ModelType::Unknown, _) => AlignmentHeadsPreset::None,
    }
}

/// Resolve the heads used for DTW, including the two presets that depend on
/// the loaded model. whisper.cpp selects every head of the `n_top` deepest
/// text layers for `N_TOP_MOST`, and the caller's list for `CUSTOM`.
pub fn resolve_alignment_heads(
    preset: AlignmentHeadsPreset,
    n_top: i32,
    custom: &[(usize, usize)],
    n_layers: usize,
    n_heads: usize,
) -> Vec<(usize, usize)> {
    let valid = |&(layer, head): &(usize, usize)| layer < n_layers && head < n_heads;
    match preset {
        AlignmentHeadsPreset::None => Vec::new(),
        AlignmentHeadsPreset::NTopMost => {
            let n_top = n_top.max(0) as usize;
            if n_top == 0 || n_top > n_layers {
                return Vec::new();
            }
            let first = n_layers - n_top;
            (first..n_layers)
                .flat_map(|layer| (0..n_heads).map(move |head| (layer, head)))
                .collect()
        }
        AlignmentHeadsPreset::Custom => custom.iter().copied().filter(valid).collect(),
        preset => get_alignment_heads(preset)
            .iter()
            .copied()
            .filter(valid)
            .collect(),
    }
}

pub fn get_alignment_heads(preset: AlignmentHeadsPreset) -> &'static [(usize, usize)] {
    // Values mirror `g_aheads_*` in whisper.cpp. A head is `(text_layer,
    // attention_head)`, and the upstream presets are the calibration data used
    // by its DTW token-timestamp computation.
    match preset {
        AlignmentHeadsPreset::TinyEn => &[
            (1, 0),
            (2, 0),
            (2, 5),
            (3, 0),
            (3, 1),
            (3, 2),
            (3, 3),
            (3, 4),
        ],
        AlignmentHeadsPreset::Tiny => &[(2, 2), (3, 0), (3, 2), (3, 3), (3, 4), (3, 5)],
        AlignmentHeadsPreset::BaseEn => &[(3, 3), (4, 7), (5, 1), (5, 5), (5, 7)],
        AlignmentHeadsPreset::Base => &[
            (3, 1),
            (4, 2),
            (4, 3),
            (4, 7),
            (5, 1),
            (5, 2),
            (5, 4),
            (5, 6),
        ],
        AlignmentHeadsPreset::SmallEn => &[
            (6, 6),
            (7, 0),
            (7, 3),
            (7, 8),
            (8, 2),
            (8, 5),
            (8, 7),
            (9, 0),
            (9, 4),
            (9, 8),
            (9, 10),
            (10, 0),
            (10, 1),
            (10, 2),
            (10, 3),
            (10, 6),
            (10, 11),
            (11, 2),
            (11, 4),
        ],
        AlignmentHeadsPreset::Small => &[
            (5, 3),
            (5, 9),
            (8, 0),
            (8, 4),
            (8, 7),
            (8, 8),
            (9, 0),
            (9, 7),
            (9, 9),
            (10, 5),
        ],
        AlignmentHeadsPreset::MediumEn => &[
            (11, 4),
            (14, 1),
            (14, 12),
            (14, 14),
            (15, 4),
            (16, 0),
            (16, 4),
            (16, 9),
            (17, 12),
            (17, 14),
            (18, 7),
            (18, 10),
            (18, 15),
            (20, 0),
            (20, 3),
            (20, 9),
            (20, 14),
            (21, 12),
        ],
        AlignmentHeadsPreset::Medium => &[(13, 15), (15, 4), (15, 15), (16, 1), (20, 0), (23, 4)],
        AlignmentHeadsPreset::LargeV1 => &[
            (9, 19),
            (11, 2),
            (11, 4),
            (11, 17),
            (22, 7),
            (22, 11),
            (22, 17),
            (23, 2),
            (23, 15),
        ],
        AlignmentHeadsPreset::LargeV2 => &[
            (10, 12),
            (13, 17),
            (16, 11),
            (16, 12),
            (16, 13),
            (17, 15),
            (17, 16),
            (18, 4),
            (18, 11),
            (18, 19),
            (19, 11),
            (21, 2),
            (21, 3),
            (22, 3),
            (22, 9),
            (22, 12),
            (23, 5),
            (23, 7),
            (23, 13),
            (25, 5),
            (26, 1),
            (26, 12),
            (27, 15),
        ],
        AlignmentHeadsPreset::LargeV3 => &[
            (7, 0),
            (10, 17),
            (12, 18),
            (13, 12),
            (16, 1),
            (17, 14),
            (19, 11),
            (21, 4),
            (24, 1),
            (25, 6),
        ],
        AlignmentHeadsPreset::LargeV3Turbo => &[(2, 4), (2, 11), (3, 3), (3, 6), (3, 11), (3, 14)],
        // N_TOP_MOST needs the model's layer count and `dtw_n_top`, and CUSTOM
        // needs a caller-provided head list. Neither can be represented by a
        // static preset, so both behave as "no calibration data".
        _ => &[],
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct WordTiming {
    pub word: String,
    pub t0: f64,
    pub t1: f64,
    pub probability: f32,
}

impl WordTiming {
    pub fn new(word: impl Into<String>, t0: f64, t1: f64, probability: f32) -> Self {
        Self {
            word: word.into(),
            t0,
            t1,
            probability,
        }
    }
}
/// Compute optimal alignment between token sequence and audio attention frames via Dynamic Time Warping.
/// `attention`: flattened row-major matrix of shape [n_tokens, n_frames].
/// Returns a vector of `(token_idx, frame_start, frame_end)` for each token.
pub fn align_dtw(
    attention: &[f32],
    n_tokens: usize,
    n_frames: usize,
) -> Vec<(usize, usize, usize)> {
    align_dtw_impl(attention, n_tokens, n_frames, true)
}

/// Dynamic Time Warping over an explicit cost matrix.
///
/// Unlike [`align_dtw`], `cost` already holds the per-cell cost, so entries
/// are accumulated as-is. Cross-attention alignment uses this form: the
/// normalized, median-filtered attention is negated once while building the
/// cost matrix.
pub fn align_dtw_from_cost(
    cost: &[f32],
    n_tokens: usize,
    n_frames: usize,
) -> Vec<(usize, usize, usize)> {
    align_dtw_impl(cost, n_tokens, n_frames, false)
}

fn align_dtw_impl(
    attention: &[f32],
    n_tokens: usize,
    n_frames: usize,
    neg_log: bool,
) -> Vec<(usize, usize, usize)> {
    if n_tokens == 0 || n_frames == 0 || attention.len() != n_tokens * n_frames {
        return Vec::new();
    }

    // Cumulative cost matrix D of shape [n_tokens + 1, n_frames + 1]
    let cols = n_frames + 1;
    let mut d = vec![f32::INFINITY; (n_tokens + 1) * cols];
    d[0] = 0.0;

    for i in 1..=n_tokens {
        let att_row = &attention[(i - 1) * n_frames..i * n_frames];
        for j in 1..=n_frames {
            let att_val = att_row[j - 1];
            let cost = if neg_log {
                -(att_val + 1e-8).ln()
            } else {
                att_val
            };

            let d_diag = d[(i - 1) * cols + (j - 1)];
            let d_top = d[(i - 1) * cols + j];
            let d_left = d[i * cols + (j - 1)];

            let min_prev = d_diag.min(d_top).min(d_left);
            d[i * cols + j] = cost + min_prev;
        }
    }

    // Backtracking from (n_tokens, n_frames) to (0, 0)
    let mut i = n_tokens;
    let mut j = n_frames;
    let mut path = Vec::with_capacity(n_tokens + n_frames);

    while i > 0 && j > 0 {
        path.push((i - 1, j - 1));
        let d_diag = d[(i - 1) * cols + (j - 1)];
        let d_top = d[(i - 1) * cols + j];
        let d_left = d[i * cols + (j - 1)];

        if d_diag <= d_top && d_diag <= d_left {
            i -= 1;
            j -= 1;
        } else if d_top <= d_left {
            i -= 1;
        } else {
            j -= 1;
        }
    }

    while i > 0 {
        path.push((i - 1, 0));
        i -= 1;
    }

    path.reverse();

    // Map each token to [frame_start, frame_end]
    let mut token_bounds = vec![(usize::MAX, 0usize); n_tokens];
    for &(tok_idx, frame_idx) in &path {
        if tok_idx < n_tokens {
            token_bounds[tok_idx].0 = token_bounds[tok_idx].0.min(frame_idx);
            token_bounds[tok_idx].1 = token_bounds[tok_idx].1.max(frame_idx);
        }
    }

    let mut result = Vec::with_capacity(n_tokens);
    let mut last_end = 0;
    for (tok_idx, (start, end)) in token_bounds.into_iter().enumerate() {
        let s = if start == usize::MAX { last_end } else { start };
        let e = if end < s { s } else { end };
        last_end = e;
        result.push((tok_idx, s, e));
    }

    result
}

/// Merge token alignments into word-level timings using BPE token word boundaries.
/// `tokens`: list of token strings and their generation probabilities.
pub fn merge_tokens_to_words(
    tokens: &[(&str, f32)],
    alignments: &[(usize, usize, usize)],
    time_offset: f64,
    frame_step_sec: f64,
) -> Vec<WordTiming> {
    if tokens.is_empty() {
        return Vec::new();
    }

    let mut words: Vec<WordTiming> = Vec::new();
    let mut cur_word = String::new();
    let mut cur_t0 = 0.0;
    let mut cur_t1 = 0.0;
    let mut cur_probs = Vec::new();

    let is_pure_punct = |s: &str| -> bool {
        let trimmed = s.trim();
        !trimmed.is_empty() && trimmed.chars().all(|c| c.is_ascii_punctuation())
    };

    for (i, &(tok_str, prob)) in tokens.iter().enumerate() {
        // Skip special tokens enclosed in brackets e.g. <|...|> or [_...]
        if tok_str.starts_with("<|") || tok_str.starts_with("[_") {
            continue;
        }

        let (tok_idx, f_start, f_end) = alignments.get(i).copied().unwrap_or((i, 0, 0));
        let _ = tok_idx;

        let t0 = time_offset + (f_start as f64) * frame_step_sec;
        let t1 = time_offset + ((f_end + 1) as f64) * frame_step_sec;

        let clean_piece = tok_str
            .strip_prefix(' ')
            .or_else(|| tok_str.strip_prefix('Ġ'))
            .unwrap_or(tok_str);

        // Snap trailing pure punctuation to previous word if idle
        if is_pure_punct(clean_piece) && cur_word.is_empty() && !words.is_empty() {
            if let Some(prev) = words.last_mut() {
                prev.word.push_str(clean_piece);
                prev.t1 = prev.t1.max(t1);
                continue;
            }
        }

        let is_new_word = tok_str.starts_with(' ')
            || tok_str.starts_with('Ġ')
            || words.is_empty() && cur_word.is_empty();

        if is_new_word && !cur_word.is_empty() {
            let avg_prob = if !cur_probs.is_empty() {
                cur_probs.iter().sum::<f32>() / cur_probs.len() as f32
            } else {
                1.0
            };
            words.push(WordTiming::new(cur_word.trim(), cur_t0, cur_t1, avg_prob));
            cur_word.clear();
            cur_probs.clear();
        }

        if cur_word.is_empty() {
            cur_t0 = t0;
        }
        cur_t1 = t1;
        cur_word.push_str(clean_piece);
        cur_probs.push(prob);
    }

    if !cur_word.is_empty() {
        let avg_prob = if !cur_probs.is_empty() {
            cur_probs.iter().sum::<f32>() / cur_probs.len() as f32
        } else {
            1.0
        };
        words.push(WordTiming::new(cur_word.trim(), cur_t0, cur_t1, avg_prob));
    }

    // Enforce monotonic timestamp boundaries: t0[k] >= t0[k-1] and t1[k] >= t0[k]
    let mut last_t0 = time_offset;
    for w in &mut words {
        if w.t0 < last_t0 {
            w.t0 = last_t0;
        }
        if w.t1 < w.t0 {
            w.t1 = w.t0;
        }
        last_t0 = w.t0;
    }

    words
}

/// Build the DTW cost matrix from captured cross-attention weights.
///
/// `attention` is laid out `[token][head][audio_frame]` with `n_heads`
/// selected alignment heads. The transformation mirrors whisper.cpp's DTW
/// timestamp path: normalize each audio/head column over the token axis,
/// median-filter along the audio axis, mean over the alignment heads, then
/// negate so DTW minimizes the accumulated cost.
///
/// Returns an empty matrix when the shapes do not agree.
pub fn dtw_cost_from_attention(
    attention: &[f32],
    n_tokens: usize,
    n_audio: usize,
    n_heads: usize,
    median_width: usize,
) -> Vec<f32> {
    if n_tokens == 0 || n_audio == 0 || n_heads == 0 {
        return Vec::new();
    }
    let expected = n_tokens * n_audio * n_heads;
    if attention.len() < expected {
        return Vec::new();
    }

    // ggml_norm over the token axis: one mean and standard deviation per
    // (head, audio frame) column.
    let mut normalized = vec![0.0f32; expected];
    for h in 0..n_heads {
        for j in 0..n_audio {
            let mut sum = 0.0f64;
            for i in 0..n_tokens {
                sum += attention[(i * n_heads + h) * n_audio + j] as f64;
            }
            let mean = (sum / n_tokens as f64) as f32;

            let mut var = 0.0f64;
            for i in 0..n_tokens {
                let d = attention[(i * n_heads + h) * n_audio + j] - mean;
                var += (d * d) as f64;
            }
            let inv_std = 1.0f32 / (((var / n_tokens as f64) as f32) + 1e-9).sqrt();

            for i in 0..n_tokens {
                let idx = (i * n_heads + h) * n_audio + j;
                normalized[idx] = (attention[idx] - mean) * inv_std;
            }
        }
    }

    // Median filter along the audio axis with reflect padding, like
    // whisper.cpp's `median_filter`.
    let mut width = median_width.max(1);
    if width >= n_audio {
        width = if n_audio.is_multiple_of(2) {
            n_audio.saturating_sub(1)
        } else {
            n_audio
        };
    }
    if width.is_multiple_of(2) {
        width -= 1;
    }
    width = width.max(1);
    let half = (width / 2) as isize;

    // Accumulate the head mean while filtering so the filtered tensor never
    // has to exist: normalized weights and the final cost are the only
    // token-by-audio buffers.
    let mut cost = vec![0.0f32; n_tokens * n_audio];
    let inv_heads = 1.0f32 / n_heads as f32;
    let mut window = Vec::with_capacity(width);
    for i in 0..n_tokens {
        for h in 0..n_heads {
            let base = (i * n_heads + h) * n_audio;
            for j in 0..n_audio {
                window.clear();
                for off in -half..=half {
                    let mut idx = j as isize + off;
                    if idx < 0 {
                        idx = -idx;
                    } else if idx >= n_audio as isize {
                        idx = 2 * (n_audio as isize - 1) - idx;
                    }
                    window.push(normalized[base + idx as usize]);
                }
                window.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                cost[i * n_audio + j] += window[window.len() / 2] * inv_heads;
            }
        }
    }
    for value in &mut cost {
        *value = -*value;
    }
    cost
}

/// Merge token alignments into word timings and clamp them to one segment.
///
/// Word starts and ends are constrained to `[seg_t0, seg_t1]`, and neither
/// boundary moves backwards, so a segment's words never overlap or leave the
/// interval reported for the segment.
pub fn merge_tokens_to_words_clamped(
    tokens: &[(&str, f32)],
    alignments: &[(usize, usize, usize)],
    seg_t0: f64,
    seg_t1: f64,
    time_offset: f64,
    frame_step_sec: f64,
) -> Vec<WordTiming> {
    let mut words = merge_tokens_to_words(tokens, alignments, time_offset, frame_step_sec);
    let mut prev_end = seg_t0;
    for word in &mut words {
        word.t0 = word.t0.clamp(seg_t0, seg_t1);
        if word.t0 < prev_end {
            word.t0 = prev_end;
        }
        word.t1 = word.t1.clamp(word.t0, seg_t1);
        prev_end = word.t1;
    }
    words
}

/// Evenly distribute tokens across a segment when no alignment heads are
/// configured.
///
/// This is a layout approximation, not acoustic alignment: it invents a
/// Gaussian attention row per token so the existing DTW/word merge can be
/// reused. Callers must keep it clearly separate from
/// cross-attention-derived timings.
pub fn approximate_word_timings(
    tokens: &[(&str, f32)],
    seg_t0: f64,
    seg_t1: f64,
    frame_step_sec: f64,
) -> Vec<WordTiming> {
    if tokens.is_empty() {
        return Vec::new();
    }
    let n_frames = (((seg_t1 - seg_t0) / frame_step_sec).round() as usize).max(1);
    let mut att = vec![0.0f32; tokens.len() * n_frames];
    for (ti, row) in att.chunks_exact_mut(n_frames).enumerate() {
        let center = (ti * n_frames) / tokens.len();
        for (frame, value) in row.iter_mut().enumerate() {
            let dist = frame as isize - center as isize;
            *value = (-(dist * dist) as f32 / 20.0).exp();
        }
    }

    let alignments = align_dtw(&att, tokens.len(), n_frames);
    merge_tokens_to_words_clamped(tokens, &alignments, seg_t0, seg_t1, seg_t0, frame_step_sec)
}

#[cfg(test)]
mod head_tests {
    use super::{resolve_alignment_heads, AlignmentHeadsPreset};

    #[test]
    fn n_top_most_takes_every_head_of_the_deepest_layers() {
        let heads = resolve_alignment_heads(AlignmentHeadsPreset::NTopMost, 2, &[], 4, 3);
        assert_eq!(heads, vec![(2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2)]);
    }

    #[test]
    fn invalid_n_top_disables_alignment_instead_of_panicking() {
        assert!(resolve_alignment_heads(AlignmentHeadsPreset::NTopMost, 0, &[], 4, 3).is_empty());
        assert!(resolve_alignment_heads(AlignmentHeadsPreset::NTopMost, 5, &[], 4, 3).is_empty());
        assert!(resolve_alignment_heads(AlignmentHeadsPreset::NTopMost, -1, &[], 4, 3).is_empty());
    }

    #[test]
    fn custom_heads_are_filtered_to_the_model() {
        let heads = resolve_alignment_heads(
            AlignmentHeadsPreset::Custom,
            0,
            &[(1, 2), (4, 0), (2, 9)],
            4,
            3,
        );
        assert_eq!(heads, vec![(1, 2)]);
    }
}
