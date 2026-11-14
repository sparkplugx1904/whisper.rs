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
    /// Per-token log-probability, used for the fallback average-logprob check.
    pub token_logprobs: Vec<f32>,
    pub cumulative_logprob: f64,
    /// Arena slot holding this hypothesis' self-attention history.
    pub slot: usize,
    pub completed: bool,
    pub last_timestamp: Option<i32>,
    /// Number of tokens up to and including the last timestamp after the
    /// leading `<|0.00|>` marker. whisper.cpp scores only this prefix.
    pub result_len: usize,
}

pub struct BeamSearch {
    pub beam_size: usize,
    pub candidates: Vec<BeamCandidate>,
    /// Finished hypotheses. They no longer need a KV slot and must not compete
    /// with active branches for beam width; whisper.cpp keeps them in separate
    /// decoder states for exactly this reason.
    pub completed_candidates: Vec<BeamCandidate>,
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

fn sequence_score(candidate: &BeamCandidate, length_penalty: f32) -> f64 {
    if candidate.tokens.is_empty() {
        return f64::NEG_INFINITY;
    }

    let len = candidate.result_len.max(1) as f64;
    let penalty = if length_penalty > 0.0 {
        ((5.0 + len) / 6.0).powf(length_penalty as f64)
    } else {
        len
    };
    let score: f64 = candidate
        .token_logprobs
        .iter()
        .take(candidate.result_len)
        .map(|&value| value as f64)
        .sum();
    score / penalty
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
            token_logprobs: Vec::new(),
            cumulative_logprob: 0.0,
            slot: 0,
            completed: false,
            last_timestamp: None,
            result_len: 0,
        };

        Self {
            beam_size,
            candidates: vec![root],
            completed_candidates: Vec::new(),
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
        max_initial_ts: f32,
        suppress_blank: bool,
        grammar: Option<&whisper_grammar::GbnfGrammar>,
        vocab: &whisper_model::Vocab,
        tokenizer: &whisper_tokenizer::Tokenizer,
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

            // 2. Timestamp grammar (pairing, monotonicity, initial limit,
            // and the timestamp-mass rule). This must match the greedy filter;
            // otherwise beam branches can repeat a timestamp indefinitely.
            crate::filter::apply_timestamp_rules(
                step_logits,
                &cand.tokens,
                beg as usize,
                eot as usize,
                n_vocab,
                no_timestamps,
                max_initial_ts,
                false,
            );

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

                let mut token_logprobs = cand.token_logprobs.clone();
                token_logprobs.push(logp);
                let result_len = if tok > beg {
                    new_tokens.len()
                } else {
                    cand.result_len
                };
                let branch = BeamCandidate {
                    tokens: new_tokens,
                    token_logprobs,
                    cumulative_logprob: cand.cumulative_logprob + (logp as f64),
                    slot: cand.slot,
                    completed: is_completed,
                    last_timestamp: new_ts,
                    result_len,
                };
                // A completion with no timestamp after the initial marker is
                // the jingle/station-ID failure mode: upstream rejects it.
                if is_completed && result_len == 0 && !no_timestamps {
                    continue;
                }
                if is_completed {
                    self.completed_candidates.push(branch);
                } else {
                    // Active branches inherit the parent slot until the
                    // survivors are known; slots are reassigned below.
                    branches.push(branch);
                }
            }
        }

        // Prune by raw cumulative log-probability. whisper.cpp only applies the
        // length penalty when ranking the finished sequences; normalizing here
        // lets a short completed branch (for example after a station jingle)
        // permanently outrank longer partial transcripts and stops decoding
        // early.
        branches.sort_by(|a, b| {
            b.cumulative_logprob
                .partial_cmp(&a.cumulative_logprob)
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
        self.candidates.is_empty() && !self.completed_candidates.is_empty()
    }

    pub fn is_completed(&self, _patience: f32) -> bool {
        self.candidates.is_empty()
    }

    pub fn best_candidate(&self, length_penalty: f32) -> Option<&BeamCandidate> {
        self.candidates
            .iter()
            .chain(&self.completed_candidates)
            .max_by(|a, b| {
                sequence_score(a, length_penalty)
                    .partial_cmp(&sequence_score(b, length_penalty))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Replace the seeded candidates, routing already-finished seeds into the
    /// completed pool so they cannot occupy active beam slots.
    pub fn set_seeds(&mut self, mut seeds: Vec<BeamCandidate>) {
        self.candidates.clear();
        self.completed_candidates.clear();
        for seed in seeds.drain(..) {
            if seed.completed {
                self.completed_candidates.push(seed);
            } else {
                self.candidates.push(seed);
            }
        }
    }
}
