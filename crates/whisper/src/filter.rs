/// Calculate text compression ratio using a pure-Rust LZ77 dictionary estimator.
/// Repetitive/hallucinated text produces high compression ratio (> 2.4).
pub fn compression_ratio(text: &str) -> f64 {
    let bytes = text.as_bytes();
    if bytes.is_empty() {
        return 1.0;
    }

    let n = bytes.len();
    if n <= 4 {
        return 1.0;
    }

    let mut compressed_size = 0usize;
    let mut i = 0;
    let window_size = 1024usize;

    while i < n {
        let win_start = i.saturating_sub(window_size);
        let win = &bytes[win_start..i];

        // Find longest match of bytes[i..] in window
        let mut best_match_len = 0;
        let max_len = (n - i).min(32);

        if max_len >= 3 {
            for start in 0..win.len() {
                let mut match_len = 0;
                while match_len < max_len
                    && start + match_len < win.len()
                    && win[start + match_len] == bytes[i + match_len]
                {
                    match_len += 1;
                }
                if match_len > best_match_len {
                    best_match_len = match_len;
                }
            }
        }

        if best_match_len >= 3 {
            // Represent match as (distance, length) pointer: 2 bytes
            compressed_size += 2;
            i += best_match_len;
        } else {
            // Literal byte: 1 byte
            compressed_size += 1;
            i += 1;
        }
    }

    (n as f64) / (compressed_size.max(1) as f64)
}

/// Compute log-softmax probability log(P(token)) given raw logits.
pub fn compute_token_logprob(logits: &[f32], token: i32) -> f32 {
    let mut max_l = f32::NEG_INFINITY;
    for &l in logits {
        if l > max_l {
            max_l = l;
        }
    }

    let mut sum_exp = 0.0f32;
    for &l in logits {
        sum_exp += (l - max_l).exp();
    }

    let tok_l = logits[token as usize];
    (tok_l - max_l) - sum_exp.ln()
}
