//! Dynamic Time Warping (DTW) word-level alignment for whisper.rs.

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
