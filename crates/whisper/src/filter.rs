/// Calculate text compression ratio using an LZ77 dictionary estimator.
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

/// Suppress blank/space token at step 0 so the model does not generate leading whitespace or early EOT.
pub fn apply_suppress_blank(logits: &mut [f32], blank_token_id: i32, eot_token_id: i32) {
    if blank_token_id >= 0 && (blank_token_id as usize) < logits.len() {
        logits[blank_token_id as usize] = f32::NEG_INFINITY;
    }
    if eot_token_id >= 0 && (eot_token_id as usize) < logits.len() {
        logits[eot_token_id as usize] = f32::NEG_INFINITY;
    }
}

/// Suppress non-speech symbols (e.g. music, laughter, background noise tokens).
pub fn apply_suppress_non_speech(logits: &mut [f32], vocab: &whisper_model::Vocab) {
    for (i, val) in logits.iter_mut().enumerate() {
        if let Some(tok_str) = vocab.id_to_token(i as i32) {
            if tok_str.starts_with("[*")
                || tok_str.starts_with("[#")
                || tok_str.contains("music")
                || tok_str.contains("laughter")
                || tok_str.contains("applause")
            {
                *val = f32::NEG_INFINITY;
            }
        }
    }
}

/// Apply repetition penalty discount to tokens already generated in the current segment.
/// For positive logits: logit /= penalty
/// For negative logits: logit *= penalty
pub fn apply_repetition_penalty(logits: &mut [f32], history_tokens: &[i32], penalty: f32) {
    if penalty <= 1.0 || penalty.is_nan() {
        return;
    }
    for &tok in history_tokens {
        if tok >= 0 && (tok as usize) < logits.len() {
            let val = &mut logits[tok as usize];
            if *val > 0.0 {
                *val /= penalty;
            } else if *val < 0.0 && *val > f32::NEG_INFINITY {
                *val *= penalty;
            }
        }
    }
}
