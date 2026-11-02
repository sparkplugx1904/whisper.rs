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

pub fn get_alignment_heads(preset: AlignmentHeadsPreset) -> &'static [(usize, usize)] {
    match preset {
        AlignmentHeadsPreset::TinyEn => &[(0, 2), (1, 3), (2, 0), (3, 5)],
        AlignmentHeadsPreset::Tiny => &[
            (0, 2),
            (0, 3),
            (1, 3),
            (2, 1),
            (2, 5),
            (3, 0),
            (3, 3),
            (3, 4),
        ],
        AlignmentHeadsPreset::BaseEn => &[
            (0, 1),
            (1, 2),
            (1, 3),
            (2, 1),
            (2, 2),
            (3, 2),
            (4, 4),
            (4, 5),
        ],
        AlignmentHeadsPreset::Base => &[
            (0, 1),
            (1, 2),
            (1, 3),
            (1, 4),
            (1, 5),
            (2, 1),
            (2, 2),
            (2, 3),
            (2, 5),
            (3, 1),
            (4, 1),
            (4, 2),
            (4, 3),
            (4, 4),
        ],
        AlignmentHeadsPreset::SmallEn => &[
            (2, 2),
            (3, 0),
            (3, 2),
            (3, 4),
            (3, 7),
            (4, 2),
            (4, 5),
            (5, 3),
            (5, 6),
            (6, 1),
            (6, 2),
            (7, 0),
            (7, 2),
            (7, 6),
            (7, 7),
            (8, 4),
            (9, 3),
            (9, 6),
            (10, 1),
            (10, 3),
            (10, 4),
            (11, 2),
            (11, 3),
            (11, 4),
            (11, 7),
        ],
        AlignmentHeadsPreset::Small => &[
            (2, 2),
            (3, 0),
            (3, 2),
            (3, 4),
            (3, 7),
            (4, 2),
            (4, 3),
            (4, 7),
            (5, 1),
            (5, 2),
            (5, 4),
            (5, 6),
            (6, 1),
            (6, 2),
            (6, 3),
            (7, 0),
            (7, 1),
            (7, 2),
            (7, 4),
            (7, 6),
            (8, 2),
            (8, 4),
            (8, 7),
            (9, 1),
            (9, 2),
            (9, 4),
            (9, 7),
            (10, 0),
            (10, 1),
            (10, 2),
            (10, 3),
            (10, 4),
            (10, 6),
            (11, 2),
            (11, 3),
            (11, 4),
        ],
        AlignmentHeadsPreset::MediumEn => &[
            (5, 11),
            (6, 0),
            (6, 1),
            (6, 4),
            (6, 7),
            (7, 0),
            (7, 8),
            (7, 10),
            (7, 11),
            (8, 0),
            (8, 2),
            (8, 3),
            (8, 8),
            (9, 0),
            (9, 5),
            (10, 1),
            (10, 5),
            (10, 8),
            (11, 1),
            (11, 4),
            (11, 5),
            (12, 1),
            (12, 4),
            (12, 7),
            (13, 0),
            (13, 2),
            (13, 4),
            (13, 8),
            (14, 1),
            (14, 3),
            (14, 7),
            (14, 11),
            (15, 6),
            (16, 0),
            (16, 5),
            (16, 6),
            (16, 7),
            (16, 8),
            (17, 3),
            (17, 8),
            (18, 7),
            (18, 11),
            (19, 0),
            (19, 1),
            (19, 2),
            (19, 8),
            (20, 0),
            (20, 2),
            (20, 3),
            (20, 6),
            (21, 3),
            (21, 5),
            (21, 11),
            (22, 1),
            (22, 7),
            (22, 8),
            (23, 0),
            (23, 2),
            (23, 6),
        ],
        AlignmentHeadsPreset::Medium => &[
            (6, 6),
            (7, 2),
            (7, 3),
            (7, 5),
            (8, 3),
            (9, 2),
            (9, 8),
            (10, 0),
            (10, 2),
            (10, 7),
            (11, 2),
            (11, 8),
            (12, 4),
            (13, 1),
            (13, 4),
            (13, 8),
            (14, 1),
            (14, 2),
            (14, 4),
            (15, 2),
            (15, 8),
            (16, 1),
            (16, 3),
            (16, 5),
            (17, 1),
            (17, 2),
            (17, 4),
            (18, 0),
            (18, 5),
            (18, 8),
            (19, 4),
            (19, 5),
            (19, 8),
            (20, 2),
            (20, 4),
            (21, 3),
            (21, 5),
            (21, 7),
            (22, 2),
            (22, 8),
            (23, 0),
            (23, 3),
        ],
        AlignmentHeadsPreset::LargeV1 => &[
            (9, 19),
            (11, 2),
            (11, 4),
            (11, 8),
            (12, 12),
            (13, 17),
            (21, 16),
            (22, 1),
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
            (23, 7),
            (23, 8),
            (23, 9),
            (24, 6),
            (24, 12),
            (25, 6),
        ],
        AlignmentHeadsPreset::LargeV3 => &[
            (7, 0),
            (10, 17),
            (11, 16),
            (12, 1),
            (13, 12),
            (14, 5),
            (14, 14),
            (15, 4),
            (16, 15),
            (17, 14),
            (17, 19),
            (18, 5),
            (18, 11),
            (18, 17),
            (19, 11),
            (20, 17),
            (21, 2),
            (21, 3),
            (22, 14),
            (22, 19),
            (23, 7),
            (24, 12),
            (25, 6),
        ],
        AlignmentHeadsPreset::LargeV3Turbo => &[(2, 4), (2, 12), (3, 7), (3, 14), (3, 17)],
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
            let cost = -(att_val + 1e-8).ln();

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

    let mut words = Vec::new();
    let mut cur_word = String::new();
    let mut cur_t0 = 0.0;
    let mut cur_t1 = 0.0;
    let mut cur_probs = Vec::new();

    for (i, &(tok_str, prob)) in tokens.iter().enumerate() {
        // Skip special tokens enclosed in brackets e.g. <|...|> or [_...]
        if tok_str.starts_with("<|") || tok_str.starts_with("[_") {
            continue;
        }

        let is_new_word = tok_str.starts_with(' ')
            || tok_str.starts_with('Ġ')
            || words.is_empty() && cur_word.is_empty();

        let (tok_idx, f_start, f_end) = alignments.get(i).copied().unwrap_or((i, 0, 0));
        let _ = tok_idx;

        let t0 = time_offset + (f_start as f64) * frame_step_sec;
        let t1 = time_offset + ((f_end + 1) as f64) * frame_step_sec;

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

        let clean_piece = tok_str
            .strip_prefix(' ')
            .or_else(|| tok_str.strip_prefix('Ġ'))
            .unwrap_or(tok_str);

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

    words
}
