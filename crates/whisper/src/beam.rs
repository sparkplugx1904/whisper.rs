//! Beam search over decoder hypotheses.
//!
//! Every candidate refers to a slot of the shared [`DecoderKvCache`] arena
//! instead of owning a private copy of the attention state. Branching therefore
//! costs one slot copy per fork rather than one deep clone of every layer's
//! key and value tensors, which previously dominated the beam step.

use crate::decoder::{DecodeSlot, DecoderKvCache, DecoderWorkspace, WhisperDecoder};

#[derive(Clone)]
pub struct BeamCandidate {
    pub tokens: Vec<i32>,
    pub cumulative_logprob: f64,
    /// Arena slot holding this hypothesis' self-attention history.
    pub slot: usize,
    pub completed: bool,
    pub last_timestamp: Option<i32>,
}

pub struct BeamSearch {
    pub beam_size: usize,
    pub candidates: Vec<BeamCandidate>,
    pub suppress_indices: Vec<usize>,
    /// Scratch row used for filtering one candidate's logits in place.
    logits_buf: Vec<f32>,
    /// Scratch flags: slots that a retained branch claims as its parent slot.
    slot_reserved: Vec<bool>,
    /// Scratch flags: slots already handed to a retained branch.
    slot_claimed: Vec<bool>,
    /// Optional user callback invoked with each candidate's token history and
    /// logits before any built-in suppression runs.
    pub logits_callback: Option<crate::params::LogitsCallback>,
}

impl BeamSearch {
    /// Start a search that borrows the caller's KV arena.
    ///
    /// The arena is only grown to `beam_size` slots: slot 0 carries whatever
    /// context the caller has already decoded (the window prompt in the
    /// transcription loop), and the caller seeds the first candidates by
    /// forking it into the remaining slots. Resetting the arena here would
    /// throw that prefill away, so nothing is cleared.
    pub fn new(beam_size: usize, cache: &mut DecoderKvCache) -> Self {
        cache.ensure_slots(beam_size.max(1));

        let root = BeamCandidate {
            tokens: Vec::new(),
            cumulative_logprob: 0.0,
            slot: 0,
            completed: false,
            last_timestamp: None,
        };

        Self {
            beam_size,
            candidates: vec![root],
            suppress_indices: Vec::new(),
            logits_buf: Vec::new(),
            slot_reserved: Vec::new(),
            slot_claimed: Vec::new(),
            logits_callback: None,
        }
    }

    /// Step the beam search by decoding the current tokens of each candidate,
    /// filtering logits (including GBNF grammar constraints), computing log-probs,
    /// and retaining the top `beam_size` branches according to length penalty.
    ///
    /// All live candidates are decoded in a single batched pass: their rows
    /// share every projection matrix, so the step turns `beam_size` independent
    /// memory-bound GEMVs into one GEMM.
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
        grammar: Option<&whisper_grammar::GbnfGrammar>,
        vocab: &whisper_model::Vocab,
        tokenizer: &whisper_tokenizer::Tokenizer,
        length_penalty: f32,
        cache: &mut DecoderKvCache,
        ws: &mut DecoderWorkspace,
    ) {
        let n_vocab = decoder.n_vocab;
        // Cloned once per step so the per-candidate logits buffer can be
        // borrowed mutably while the callback stays reachable.
        let logits_callback = self.logits_callback.clone();
        if self.logits_buf.len() < n_vocab {
            self.logits_buf.resize(n_vocab, 0.0);
        }

        // Only the last token of each live candidate is new to its slot; the
        // rest of the history is already in the arena. Candidates are ordered
        // by slot so the batched logits arrive slot-major.
        let mut order: Vec<usize> = (0..self.candidates.len())
            .filter(|&i| !self.candidates[i].completed && !self.candidates[i].tokens.is_empty())
            .collect();
        order.sort_by_key(|&i| self.candidates[i].slot);
        debug_assert!(
            order
                .windows(2)
                .all(|w| self.candidates[w[0]].slot != self.candidates[w[1]].slot),
            "active candidates must own distinct arena slots"
        );

        let seqs: Vec<DecodeSlot<'_>> = order
            .iter()
            .map(|&i| {
                let cand = &self.candidates[i];
                DecodeSlot {
                    slot: cand.slot,
                    tokens: &cand.tokens[cand.tokens.len() - 1..],
                }
            })
            .collect();

        // Slot -> logits row, sorted by slot for lookup below.
        let rows: Vec<(usize, usize)> = order
            .iter()
            .enumerate()
            .map(|(row, &i)| (self.candidates[i].slot, row))
            .collect();

        let logits = decoder.decode_batch(&seqs, encoder_output, cache, ws);

        let mut branches: Vec<BeamCandidate> =
            Vec::with_capacity(self.candidates.len() * self.beam_size);

        for cand in &self.candidates {
            if cand.completed {
                branches.push(cand.clone());
                continue;
            }

            let row = rows
                .binary_search_by_key(&cand.slot, |&(slot, _)| slot)
                .ok()
                .map(|pos| rows[pos].1);

            let step_logits = match row {
                Some(r) => {
                    self.logits_buf
                        .copy_from_slice(&logits[r * n_vocab..(r + 1) * n_vocab]);
                    &mut self.logits_buf[..]
                }
                // A candidate without tokens was never seeded with a prompt
                // token, so there is nothing to decode; a flat distribution
                // matches decoding no input at all.
                None => {
                    self.logits_buf.fill(0.0);
                    &mut self.logits_buf[..]
                }
            };

            // 0. User logits callback observes the raw logits of this candidate
            // before any built-in suppression runs.
            if let Some(cb) = &logits_callback {
                (cb)(&cand.tokens, step_logits);
            }

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

            // Suppress regex / custom token indices
            for &idx in &self.suppress_indices {
                if idx < step_logits.len() {
                    step_logits[idx] = f32::NEG_INFINITY;
                }
            }

            // 3. Grammar filtering if active
            if let Some(gbnf) = grammar {
                let cur_text = tokenizer.decode(&cand.tokens, true);
                gbnf.filter_logits(&cur_text, step_logits, vocab);
            }

            // 4. Compute log-softmax for top tokens
            let mut max_l = f32::NEG_INFINITY;
            for &l in step_logits.iter() {
                if l > max_l {
                    max_l = l;
                }
            }
            let mut sum_exp = 0.0f32;
            for &l in step_logits.iter() {
                if l > f32::NEG_INFINITY {
                    sum_exp += (l - max_l).exp();
                }
            }
            let log_sum_exp = sum_exp.max(1e-10).ln();

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

                // The branch inherits the parent slot until the survivors are
                // known; slots are reassigned below.
                branches.push(BeamCandidate {
                    tokens: new_tokens,
                    cumulative_logprob: cand.cumulative_logprob + (logp as f64),
                    slot: cand.slot,
                    completed: is_completed,
                    last_timestamp: new_ts,
                });
            }
        }

        // Rank branches by normalized cumulative logprob: score / (len^alpha)
        let alpha = (length_penalty as f64).clamp(0.0, 2.0);
        branches.sort_by(|a, b| {
            let len_a = (a.tokens.len().max(1) as f64).powf(alpha);
            let len_b = (b.tokens.len().max(1) as f64).powf(alpha);
            let score_a = a.cumulative_logprob / len_a;
            let score_b = b.cumulative_logprob / len_b;
            score_b
                .partial_cmp(&score_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        branches.truncate(self.beam_size);

        // Hand out arena slots to the retained branches.
        //
        // Each parent slot may have produced several surviving children. The
        // first one keeps the parent's slot because the arena already holds the
        // exact token history that branch needs; every further child needs its
        // own copy, which `copy_slot` makes from the parent. Slots that are a
        // parent of some retained branch are reserved up front so a sibling's
        // extra child cannot steal the slot its owner is about to claim.
        // Completed hypotheses are never decoded again, so the slots they keep
        // are taken out of circulation as well: they count as their own parent
        // in the branch list.
        let n_slots = cache.t_past.len();
        self.slot_reserved.clear();
        self.slot_reserved.resize(n_slots, false);
        for branch in &branches {
            debug_assert!(branch.slot < n_slots, "candidate slot out of range");
            self.slot_reserved[branch.slot] = true;
        }
        self.slot_claimed.clear();
        self.slot_claimed.resize(n_slots, false);

        let mut free_cursor = 0usize;
        for branch in branches.iter_mut() {
            let parent = branch.slot;
            if !self.slot_claimed[parent] {
                self.slot_claimed[parent] = true;
                continue;
            }

            let mut new_slot = free_cursor;
            while new_slot < n_slots
                && (self.slot_reserved[new_slot] || self.slot_claimed[new_slot])
            {
                new_slot += 1;
            }
            assert!(new_slot < n_slots, "beam slot arena exhausted");
            self.slot_claimed[new_slot] = true;
            cache.copy_slot(new_slot, parent);
            branch.slot = new_slot;
            free_cursor = new_slot;
        }

        self.candidates = branches;
    }

    pub fn is_all_completed(&self) -> bool {
        self.candidates.iter().all(|c| c.completed)
    }

    pub fn is_completed(&self, patience: f32) -> bool {
        let target = ((self.beam_size as f32) * patience).round() as usize;
        let target = target.clamp(1, self.beam_size);
        let completed_count = self.candidates.iter().filter(|c| c.completed).count();
        completed_count >= target
    }

    pub fn best_candidate(&self) -> Option<&BeamCandidate> {
        self.candidates.first()
    }
}
