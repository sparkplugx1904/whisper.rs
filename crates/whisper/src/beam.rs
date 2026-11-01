use crate::decoder::{DecoderKvCache, WhisperDecoder};

#[derive(Clone)]
pub struct BeamCandidate {
    pub tokens: Vec<i32>,
    pub cumulative_logprob: f64,
    pub cache: DecoderKvCache,
    pub completed: bool,
    pub last_timestamp: Option<i32>,
}

pub struct BeamSearch {
    pub beam_size: usize,
    pub candidates: Vec<BeamCandidate>,
}

impl BeamSearch {
    pub fn new(beam_size: usize, initial_cache: DecoderKvCache) -> Self {
        let root = BeamCandidate {
            tokens: Vec::new(),
            cumulative_logprob: 0.0,
            cache: initial_cache,
            completed: false,
            last_timestamp: None,
        };

        Self {
            beam_size,
            candidates: vec![root],
        }
    }

    /// Step the beam search by decoding the current tokens of each candidate,
    /// filtering logits, computing log-probs, and retaining the top `beam_size` branches.
    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &mut self,
        decoder: &WhisperDecoder,
        encoder_output: &[f32],
        step_idx: usize,
        eot: i32,
        beg: i32,
        no_timestamps: bool,
        suppress_blank: bool,
    ) {
        let mut branches = Vec::new();

        for cand in &self.candidates {
            if cand.completed {
                branches.push(cand.clone());
                continue;
            }

            let last_token = cand.tokens.last().copied();
            let mut cand_cache = cand.cache.clone();

            let decode_input = match last_token {
                Some(t) => vec![t],
                None => Vec::new(), // Root already has prompt processed in cache
            };

            let mut logits = if decode_input.is_empty() {
                // If root with empty tokens, we read logits from initial prompt decoding
                vec![0.0f32; decoder.n_vocab]
            } else {
                decoder.decode(&decode_input, encoder_output, &mut cand_cache)
            };

            let n_vocab = decoder.n_vocab;
            let start_idx = logits.len() - n_vocab;
            let step_logits = &mut logits[start_idx..];

            // 1. Suppress blank at start
            if suppress_blank && step_idx == 0 {
                step_logits[220] = f32::NEG_INFINITY;
                step_logits[eot as usize] = f32::NEG_INFINITY;
            }

            // 2. Timestamps suppression
            if no_timestamps {
                step_logits[(beg as usize)..n_vocab].fill(f32::NEG_INFINITY);
            } else if let Some(last_ts) = cand.last_timestamp {
                if (last_ts as usize) > (beg as usize) {
                    step_logits[(beg as usize)..(last_ts as usize)].fill(f32::NEG_INFINITY);
                }
            }

            // 3. Compute log-softmax for top tokens
            let mut max_l = f32::NEG_INFINITY;
            for &l in step_logits.iter() {
                if l > max_l {
                    max_l = l;
                }
            }
            let mut sum_exp = 0.0f32;
            for &l in step_logits.iter() {
                sum_exp += (l - max_l).exp();
            }
            let log_sum_exp = sum_exp.ln();

            // Find top-K tokens for this candidate
            let mut top_tokens: Vec<(i32, f32)> = Vec::new();
            for (tok_id, &l) in step_logits.iter().enumerate() {
                if l > f32::NEG_INFINITY {
                    let logprob = (l - max_l) - log_sum_exp;
                    top_tokens.push((tok_id as i32, logprob));
                }
            }
            top_tokens.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            top_tokens.truncate(self.beam_size);

            for (tok, logp) in top_tokens {
                let mut new_tokens = cand.tokens.clone();
                new_tokens.push(tok);

                let is_completed = tok == eot;
                let new_ts = if tok >= beg {
                    Some(tok)
                } else {
                    cand.last_timestamp
                };

                branches.push(BeamCandidate {
                    tokens: new_tokens,
                    cumulative_logprob: cand.cumulative_logprob + (logp as f64),
                    cache: cand_cache.clone(),
                    completed: is_completed,
                    last_timestamp: new_ts,
                });
            }
        }

        // Rank branches by normalized cumulative logprob: score / (len^0.7)
        branches.sort_by(|a, b| {
            let len_a = (a.tokens.len().max(1) as f64).powf(0.7);
            let len_b = (b.tokens.len().max(1) as f64).powf(0.7);
            let score_a = a.cumulative_logprob / len_a;
            let score_b = b.cumulative_logprob / len_b;
            score_b
                .partial_cmp(&score_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        branches.truncate(self.beam_size);
        self.candidates = branches;
    }

    pub fn is_all_completed(&self) -> bool {
        self.candidates.iter().all(|c| c.completed)
    }

    pub fn best_candidate(&self) -> Option<&BeamCandidate> {
        self.candidates.first()
    }
}
