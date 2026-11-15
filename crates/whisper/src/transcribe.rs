use crate::beam::BeamSearch;
use crate::decoder::{DecoderKvCache, DecoderWorkspace, WhisperDecoder};
use crate::dtw::{
    align_dtw_from_cost, approximate_word_timings, dtw_cost_from_attention, get_alignment_heads,
    merge_tokens_to_words_clamped,
};
use crate::encoder::WhisperEncoder;
use crate::filter::{
    compression_ratio, CallbackLogitFilter, GrammarLogitFilter, LogitFilterContext,
    LogitFilterPipeline, RegexFilter, RepetitionPenaltyFilter, SuppressBlankFilter,
    SuppressNonSpeechFilter, TimestampLogitFilter,
};
use crate::params::WhisperParams;
use crate::segment::Segment;
use crate::speech_map::SpeechMap;
use crate::state::WhisperState;
use std::path::Path;
use std::time::Instant;
use whisper_audio::mel::MelSpectrogram;
use whisper_audio::wav::WHISPER_SAMPLE_RATE;
use whisper_audio::AudioError;
use whisper_model::vocab::LANGUAGES;
use whisper_model::{ModelError, WhisperModel};
use whisper_tokenizer::Tokenizer;

pub struct WhisperContext {
    pub model: WhisperModel,
    pub encoder: WhisperEncoder,
    pub decoder: WhisperDecoder,
    pub tokenizer: Tokenizer,
}

impl WhisperContext {
    pub fn new(model: WhisperModel) -> Result<Self, ModelError> {
        let encoder = WhisperEncoder::new(&model)?;
        let decoder = WhisperDecoder::new(&model)?;
        let tokenizer = Tokenizer::new(model.vocab.clone());

        Ok(Self {
            model,
            encoder,
            decoder,
            tokenizer,
        })
    }

    /// Load context directly from model file path.
    pub fn load_model(path: impl AsRef<Path>) -> Result<Self, ModelError> {
        let model = WhisperModel::load_from_file(path)?;
        Self::new(model)
    }

    /// Create a new reusable working state for inference.
    pub fn create_state(&self) -> WhisperState {
        WhisperState::new(self.decoder.n_layers, self.decoder.d_model)
    }

    /// Full end-to-end transcription pipeline matching whisper.cpp: whisper_full()
    pub fn full(
        &self,
        state: &mut WhisperState,
        params: &WhisperParams,
        samples: &[f32],
    ) -> Result<(), AudioError> {
        // Every tensor operator uses the current Rayon pool. Building one here
        // makes `n_threads` real for the whole call instead of falling back to
        // the process-wide pool, and keeps two concurrent contexts from
        // oversubscribing each other's workers.
        let threads = params.n_threads.max(1);
        match rayon::ThreadPoolBuilder::new().num_threads(threads).build() {
            Ok(pool) => pool.install(|| self.full_inner(state, params, samples)),
            Err(_) => self.full_inner(state, params, samples),
        }
    }

    fn full_inner(
        &self,
        state: &mut WhisperState,
        params: &WhisperParams,
        samples: &[f32],
    ) -> Result<(), AudioError> {
        state.clear();
        if samples.len() < 1600 {
            return Ok(());
        }

        // Voice activity detection runs on the raw samples before the mel
        // spectrogram is built. Silent windows then cost nothing to process,
        // and audio that contains no speech at all returns immediately instead
        // of letting the decoder hallucinate text over silence.
        let speech_map = SpeechMap::from_samples(samples, params);
        if let Some(map) = &speech_map {
            if map.is_empty() {
                return Ok(());
            }
        }

        let mel_start = Instant::now();
        let mel = whisper_audio::compute_log_mel_spectrogram(
            samples,
            &self.model.filters.data,
            self.model.filters.n_mel,
        );
        state.timings.mel_us += mel_start.elapsed().as_micros() as u64;

        state.segments = self.transcribe_internal(&mel, params, state, speech_map.as_ref());
        Ok(())
    }

    /// Transcribe raw PCM float samples (16,000 Hz mono in range [-1.0, 1.0]).
    pub fn transcribe_pcm(&self, samples: &[f32], params: &WhisperParams) -> Vec<Segment> {
        let mut state = self.create_state();
        let _ = self.full(&mut state, params, samples);
        state.segments
    }

    /// Transcribe a WAV audio file directly from disk.
    pub fn transcribe_file(
        &self,
        wav_path: impl AsRef<Path>,
        params: &WhisperParams,
    ) -> Result<Vec<Segment>, Box<dyn std::error::Error>> {
        let audio = whisper_audio::load_audio_from_file(wav_path)?;
        let samples = if audio.sample_rate != WHISPER_SAMPLE_RATE {
            whisper_audio::resample_to_16k(&audio.samples, audio.sample_rate)
        } else {
            audio.samples
        };
        Ok(self.transcribe_pcm(&samples, params))
    }

    /// Auto-detect the spoken language from the first 30 seconds of audio.
    /// Returns (language_code, probability).
    /// Auto-detect the spoken language from the first 30 seconds of audio.
    /// Optionally populates `probs_out` with normalized language probabilities.
    /// Returns (language_code, language_id, probability).
    pub fn detect_language_probs(
        &self,
        mel: &MelSpectrogram,
        mut probs_out: Option<&mut [f32]>,
    ) -> (String, usize, f32) {
        if !self.model.hparams.is_multilingual() {
            if let Some(probs) = &mut probs_out {
                if !probs.is_empty() {
                    probs[0] = 1.0;
                }
            }
            return ("en".to_string(), 0, 1.0);
        }

        let mel_chunk = mel.get_chunk(0, self.model.hparams.n_audio_ctx as usize);
        let enc_out = self.encoder.forward(&mel_chunk);
        self.detect_language_probs_with_enc(&enc_out, probs_out)
    }

    /// Compute language probabilities from an already encoded audio window.
    ///
    /// The C API allows callers to pick the window with `offset_ms`; keeping
    /// the decoder step separate lets that path reuse an encoder output.
    pub fn detect_language_probs_with_enc(
        &self,
        enc_out: &[f32],
        mut probs_out: Option<&mut [f32]>,
    ) -> (String, usize, f32) {
        if !self.model.hparams.is_multilingual() {
            if let Some(probs) = &mut probs_out {
                if !probs.is_empty() {
                    probs[0] = 1.0;
                }
            }
            return ("en".to_string(), 0, 1.0);
        }

        let mut cache = DecoderKvCache::new(self.decoder.n_layers, 1, self.decoder.d_model);
        let mut workspace = DecoderWorkspace::new();
        let prompt = [self.tokenizer.sot()];
        let logits = self
            .decoder
            .decode(&prompt, enc_out, &mut cache, &mut workspace);
        let logits = &*logits;

        let num_langs = self.model.hparams.num_languages();
        let sot = self.tokenizer.sot();

        let mut lang_scores = Vec::with_capacity(num_langs);
        let mut max_score = f32::NEG_INFINITY;

        for (i, &(code, _)) in LANGUAGES.iter().take(num_langs).enumerate() {
            let token_id = sot + 1 + i as i32;
            let score = logits[token_id as usize];
            if score > max_score {
                max_score = score;
            }
            lang_scores.push((code, score));
        }

        // Softmax
        let mut sum_exp = 0.0f32;
        for (_, score) in &mut lang_scores {
            *score = (*score - max_score).exp();
            sum_exp += *score;
        }

        let inv_sum = 1.0 / sum_exp.max(1e-10);
        let mut best_lang = "en";
        let mut best_idx = 0;
        let mut best_prob = 0.0f32;

        for (i, (code, score)) in lang_scores.into_iter().enumerate() {
            let prob = score * inv_sum;
            if let Some(probs) = &mut probs_out {
                if i < probs.len() {
                    probs[i] = prob;
                }
            }
            if prob > best_prob {
                best_prob = prob;
                best_lang = code;
                best_idx = i;
            }
        }

        (best_lang.to_string(), best_idx, best_prob)
    }

    pub fn detect_language_with_enc(&self, enc_out: &[f32]) -> (String, f32) {
        let (lang, _, prob) = self.detect_language_probs_with_enc(enc_out, None);
        (lang, prob)
    }

    pub fn detect_language(&self, mel: &MelSpectrogram) -> (String, f32) {
        let mel_chunk = mel.get_chunk(0, self.model.hparams.n_audio_ctx as usize);
        let enc_out = self.encoder.forward(&mel_chunk);
        self.detect_language_with_enc(&enc_out)
    }

    /// Transcribe whole audio mel spectrogram into text segments.
    pub fn transcribe(&self, mel: &MelSpectrogram, params: &WhisperParams) -> Vec<Segment> {
        let mut state = self.create_state();
        self.transcribe_internal(mel, params, &mut state, None)
    }

    fn transcribe_internal(
        &self,
        mel: &MelSpectrogram,
        params: &WhisperParams,
        state: &mut WhisperState,
        speech_map: Option<&SpeechMap>,
    ) -> Vec<Segment> {
        let is_multilingual = self.model.hparams.is_multilingual();
        let n_audio_ctx =
            if params.audio_ctx > 0 && params.audio_ctx < self.model.hparams.n_audio_ctx as usize {
                params.audio_ctx
            } else {
                self.model.hparams.n_audio_ctx as usize
            };

        if params.abort_callback.as_ref().is_some_and(|cb| cb()) {
            return Vec::new();
        }

        // The first window is encoded at most once and shared between language
        // detection and the transcription loop. It is left lazy because VAD may
        // skip that window entirely when the audio opens with silence, and an
        // English-only model never needs it for language detection.
        let needs_first_encode_for_language = is_multilingual && params.language == "auto";
        let mut enc_out_0: Option<Vec<f32>> = if needs_first_encode_for_language {
            // The language-detection prefill is a real encoder pass, so the
            // begin callback gates it exactly like the per-window passes.
            if !encoder_begin_allowed(params) {
                return Vec::new();
            }
            let mel_chunk_0 = mel.get_chunk(0, n_audio_ctx);
            let encode_start = Instant::now();
            let enc = self.encoder.forward_with_ctx(&mel_chunk_0, n_audio_ctx);
            state
                .timings
                .add_encode(encode_start.elapsed().as_micros() as u64);
            Some(enc)
        } else {
            None
        };

        let lang = if params.language == "auto" {
            let (detected, prob) = if let Some(enc) = enc_out_0.as_deref() {
                self.detect_language_with_enc(enc)
            } else {
                // English-only models have nothing to detect.
                ("en".to_string(), 1.0)
            };
            state.lang_prob = prob;
            detected
        } else {
            state.lang_prob = 1.0;
            params.language.clone()
        };
        let lang_idx = whisper_model::vocab::lang_id(&lang).unwrap_or(0);
        state.lang_id = lang_idx;

        let sot = self.tokenizer.sot();
        let eot = self.tokenizer.eot();
        let beg = self.tokenizer.beg();
        let not_token = self.tokenizer.not();

        let mut prompt_base = Vec::new();

        // 1. Initial prompt injection (raw tokens or text)
        if !params.prompt_tokens.is_empty() {
            prompt_base.push(self.model.vocab.token_prev);
            prompt_base.extend_from_slice(&params.prompt_tokens);
        } else if let Some(prompt_text) = &params.initial_prompt {
            if !prompt_text.trim().is_empty() {
                prompt_base.push(self.model.vocab.token_prev);
                let encoded_prompt = self.tokenizer.encode(prompt_text);
                prompt_base.extend(encoded_prompt);
            }
        }

        // 2. SOT + Language + Task tokens
        prompt_base.push(sot);
        if is_multilingual {
            let lang_token = sot + 1 + lang_idx as i32;
            prompt_base.push(lang_token);
            if params.translate {
                prompt_base.push(self.tokenizer.translate());
            } else {
                prompt_base.push(self.tokenizer.transcribe());
            }
        }

        if params.no_timestamps {
            prompt_base.push(not_token);
        }

        let mut segments = Vec::new();
        let seek_start = (params.offset_ms / 10).max(0) as usize;
        let mut seek_frame = seek_start;
        let seek_end = if params.duration_ms > 0 {
            (seek_start + (params.duration_ms / 10).max(0) as usize).min(mel.n_len_org)
        } else {
            mel.n_len_org
        };
        let chunk_frames = 2 * n_audio_ctx; // 3000 frames = 30 seconds
        let total_audio_sec = mel.n_len_org as f64 * 0.01;
        let mut history_tokens: Vec<i32> = Vec::new();
        let regex_filter = params
            .suppress_regex
            .as_deref()
            .and_then(|pattern| RegexFilter::new(pattern, &self.model.vocab).ok());

        while seek_frame < seek_end {
            if params.abort_callback.as_ref().is_some_and(|cb| cb()) {
                break;
            }
            if let Some(ref cb) = params.on_progress {
                let total_frames = seek_end.saturating_sub(seek_start).max(1);
                let pct = (((seek_frame.saturating_sub(seek_start)) as f64 / total_frames as f64)
                    * 100.0)
                    .clamp(0.0, 100.0) as i32;
                cb(pct);
            }
            if seek_frame + 10 >= seek_end {
                break;
            }
            // Jump over windows that contain no speech. Skipping the window is
            // free: neither the encoder nor the decoder runs, and because mel
            // frame indices stay absolute the emitted timestamps remain aligned
            // with the original audio.
            if let Some(map) = speech_map {
                if !map.overlaps(seek_frame, seek_frame + chunk_frames) {
                    match map.next_speech_start(seek_frame + chunk_frames) {
                        Some(next) => {
                            seek_frame = next;
                            continue;
                        }
                        None => break,
                    }
                }
            }
            let seek_time = seek_frame as f64 * 0.01;
            let chunk_sec = chunk_frames as f64 * 0.01;
            let enc_out = if seek_frame == 0 {
                // Encode window 0 on first use, then reuse it if the loop
                // returns here; a VAD-skipped opening window never pays for it.
                if enc_out_0.is_none() {
                    if !encoder_begin_allowed(params) {
                        break;
                    }
                    let mel_chunk = mel.get_chunk(0, n_audio_ctx);
                    let encode_start = Instant::now();
                    let enc = self.encoder.forward_with_ctx(&mel_chunk, n_audio_ctx);
                    state
                        .timings
                        .add_encode(encode_start.elapsed().as_micros() as u64);
                    enc_out_0 = Some(enc);
                }
                enc_out_0.clone().expect("window 0 encoded above")
            } else {
                if !encoder_begin_allowed(params) {
                    break;
                }
                let mel_chunk = mel.get_chunk(seek_frame, n_audio_ctx);
                let encode_start = Instant::now();
                let enc = self.encoder.forward_with_ctx(&mel_chunk, n_audio_ctx);
                state
                    .timings
                    .add_encode(encode_start.elapsed().as_micros() as u64);
                enc
            };
            state.encoder_output = enc_out.clone();

            let prompt_builder = WindowPromptBuilder {
                prompt_base: &prompt_base,
                is_multilingual,
                lang_idx,
                sot,
                not_token,
                token_prev: self.model.vocab.token_prev,
            };
            let window_prompt =
                prompt_builder.build(seek_frame, &history_tokens, params, &self.tokenizer);

            let mut temperature = params.temperature;
            let mut tokens_cur = Vec::new();
            let mut token_logprobs = Vec::new();
            let mut token_entropies = Vec::new();
            let mut no_speech_prob = 0.01f32;
            // Upstream retries a failed window at increasing temperature but
            // returns the best attempt when every temperature fails. Keeping
            // the last sample is what turned a repetition loop into duplicated
            // text here: the sampled retry was often shorter and more
            // repetitive than the beam result it replaced.
            let mut best_attempt: Option<DecodeAttempt> = None;

            let gbnf_grammar = params
                .grammar
                .as_deref()
                .and_then(|g| whisper_grammar::GbnfGrammar::parse(g).ok());

            // Temperature fallback loop (e.g. 0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
            let mut force_greedy = false;
            // A looped window also poisons the rolling history. The greedy
            // retry therefore uses a prompt without prior text instead of
            // feeding the loop back into the next decode.
            let clean_prompt = prompt_builder.build(seek_frame, &[], params, &self.tokenizer);
            while temperature <= 1.0 {
                let active_prompt = if force_greedy {
                    &clean_prompt
                } else {
                    &window_prompt
                };
                tokens_cur.clear();
                token_logprobs.clear();
                token_entropies.clear();
                // Beam search keeps one KV arena slot per branch, so branching
                // never deep-copies an entire cache; greedy decoding needs one.
                let n_slots = if params.beam_size > 1 {
                    params.beam_size
                } else {
                    1
                };
                let mut cache =
                    DecoderKvCache::new(self.decoder.n_layers, n_slots, self.decoder.d_model);

                if params.beam_size > 1 && temperature == 0.0 && !force_greedy {
                    // BEAM SEARCH
                    let n_vocab = self.decoder.n_vocab;
                    let mut beam = BeamSearch::new(params.beam_size, &mut cache);
                    // Beam steps filter per-candidate logits inside the search
                    // loop, so the callback travels with the search object.
                    beam.logits_callback = params.on_logits.clone();
                    let decode_start = Instant::now();
                    let logits = self.decoder.decode(
                        active_prompt,
                        &enc_out,
                        &mut cache,
                        &mut state.workspace,
                    );
                    let decode_elapsed = decode_start.elapsed().as_micros() as u64;
                    state.timings.add_decode(decode_elapsed);
                    state.timings.add_prompt(decode_elapsed);
                    let start_idx = logits.len() - n_vocab;
                    let step_logits = &mut logits[start_idx..];

                    let sample_start = Instant::now();
                    let mut beam_pipeline = LogitFilterPipeline::new();
                    if let Some(cb) = &params.on_logits {
                        beam_pipeline.add(CallbackLogitFilter::new(cb.clone()));
                    }
                    if params.suppress_blank {
                        beam_pipeline.add(SuppressBlankFilter::new(220, eot));
                    }
                    beam_pipeline.add(TimestampLogitFilter::with_max_initial_ts(
                        beg as usize,
                        eot as usize,
                        n_vocab,
                        params.no_timestamps,
                        params.max_initial_ts,
                    ));
                    if let Some(rf) = &regex_filter {
                        beam_pipeline.add(rf.clone());
                        beam.suppress_indices = rf.indices.clone();
                    }
                    if let Some(gbnf) = gbnf_grammar.clone() {
                        beam_pipeline.add(GrammarLogitFilter::new(
                            gbnf,
                            self.model.vocab.clone(),
                            self.tokenizer.clone(),
                        ));
                    }
                    // The no-speech probability must be read from the raw row.
                    // The timestamp-mass rule masks every text token, including
                    // `_NO_SPEECH_`, so reading it after filtering reports 0 for
                    // any window that opens with a timestamp.
                    let no_speech_idx = self.model.vocab.token_nosp as usize;
                    if no_speech_idx < step_logits.len() {
                        no_speech_prob = softmax_prob_at(step_logits, no_speech_idx);
                    }

                    let seed_ctx = LogitFilterContext {
                        step: 0,
                        tokens_cur: &[],
                        last_timestamp_token: None,
                    };
                    beam_pipeline.apply(step_logits, &seed_ctx);
                    // `<|notimestamps|>` is never a generation token; upstream
                    // masks it unconditionally. Leaving it in the seed set puts
                    // a special token into the history prompt and can trigger
                    // repetition loops on later windows.
                    let not_idx = self.model.vocab.token_not as usize;
                    if not_idx < step_logits.len() {
                        step_logits[not_idx] = f32::NEG_INFINITY;
                    }

                    // Seed branches compete on log-probability, like every later
                    // step; the raw projection output is a logit, not a log-prob.
                    let seed_lse = log_sum_exp(step_logits);

                    let mut seed_tokens = Vec::new();
                    for (tok, &l) in step_logits.iter().enumerate() {
                        if l > f32::NEG_INFINITY {
                            seed_tokens.push((tok as i32, l - seed_lse));
                        }
                    }
                    seed_tokens
                        .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    seed_tokens.truncate(params.beam_size);
                    state
                        .timings
                        .add_sample(sample_start.elapsed().as_micros() as u64);

                    beam.set_seeds(
                        seed_tokens
                            .into_iter()
                            .enumerate()
                            .map(|(slot, (tok, l))| {
                                // Slot 0 already holds the prompt KV; the remaining
                                // seed branches fork from it instead of replaying the
                                // prompt through the decoder.
                                if slot > 0 {
                                    cache.copy_slot(slot, 0);
                                }
                                crate::beam::BeamCandidate {
                                    tokens: vec![tok],
                                    token_logprobs: vec![l],
                                    cumulative_logprob: l as f64,
                                    slot,
                                    completed: tok == eot,
                                    last_timestamp: if tok >= beg { Some(tok) } else { None },
                                    result_len: usize::from(tok > beg),
                                }
                            })
                            .collect(),
                    );

                    let max_steps = 224;
                    for step in 1..max_steps {
                        if beam.is_completed(1.0) {
                            break;
                        }
                        let batch_start = Instant::now();
                        beam.step(
                            &self.decoder,
                            &enc_out,
                            step,
                            eot,
                            beg,
                            params.no_timestamps,
                            params.max_initial_ts,
                            params.suppress_blank,
                            gbnf_grammar.as_ref(),
                            &self.model.vocab,
                            &self.tokenizer,
                            &mut cache,
                            &mut state.workspace,
                        );
                        state
                            .timings
                            .add_batch_decode(batch_start.elapsed().as_micros() as u64);
                    }

                    let length_penalty = if params.length_penalty > 0.0 {
                        params.length_penalty
                    } else {
                        0.7
                    };
                    if let Some(best) = beam.best_candidate(length_penalty, beg) {
                        tokens_cur = best.tokens.clone();
                        token_logprobs = best.token_logprobs.clone();
                    }
                } else {
                    // GREEDY DECODING
                    let mut last_timestamp_token: Option<i32> = None;
                    let max_steps = 224;
                    let n_vocab = self.decoder.n_vocab;
                    let mut greedy_pipeline = LogitFilterPipeline::new();
                    // The user logits callback runs before every built-in
                    // suppression, matching whisper.cpp's filter order.
                    if let Some(cb) = &params.on_logits {
                        greedy_pipeline.add(CallbackLogitFilter::new(cb.clone()));
                    }
                    if params.suppress_blank {
                        greedy_pipeline.add(SuppressBlankFilter::new(220, eot));
                    }
                    if params.suppress_nst {
                        let nst_indices = crate::filter::get_non_speech_tokens(&self.model.vocab);
                        greedy_pipeline.add(SuppressNonSpeechFilter::new(nst_indices));
                    }
                    if let Some(rf) = &regex_filter {
                        greedy_pipeline.add(rf.clone());
                    }
                    if params.repetition_penalty > 1.0 {
                        greedy_pipeline
                            .add(RepetitionPenaltyFilter::new(params.repetition_penalty));
                    }
                    greedy_pipeline.add(TimestampLogitFilter::with_max_initial_ts(
                        beg as usize,
                        eot as usize,
                        n_vocab,
                        params.no_timestamps,
                        params.max_initial_ts,
                    ));
                    if let Some(gbnf) = gbnf_grammar.clone() {
                        greedy_pipeline.add(GrammarLogitFilter::new(
                            gbnf,
                            self.model.vocab.clone(),
                            self.tokenizer.clone(),
                        ));
                    }

                    // `_NO_SPEECH_` id for this model's vocabulary.
                    let no_speech_token = self.model.vocab.token_nosp;
                    let mut rng_state = (seek_frame as u64)
                        .wrapping_mul(0x9e3779b97f4a7c15)
                        .wrapping_add(0x2545f4914f6cdd1d)
                        | 1;
                    for step in 0..max_steps {
                        if params.abort_callback.as_ref().is_some_and(|cb| cb()) {
                            break;
                        }
                        let start_idx;
                        {
                            // The prompt is decoded on the first step; later
                            // steps append only the token chosen previously, so
                            // the KV cache keeps the whole history.
                            let step_input: &[i32] = if step == 0 {
                                active_prompt
                            } else {
                                &tokens_cur[tokens_cur.len() - 1..]
                            };
                            let decode_start = Instant::now();
                            let logits = self.decoder.decode(
                                step_input,
                                &enc_out,
                                &mut cache,
                                &mut state.workspace,
                            );
                            let decode_elapsed = decode_start.elapsed().as_micros() as u64;
                            state.timings.add_decode(decode_elapsed);
                            if step == 0 {
                                state.timings.add_prompt(decode_elapsed);
                            }
                            start_idx = logits.len() - n_vocab;
                        }
                        let sample_start = Instant::now();
                        // The filter pipeline mutates the logits in place, which
                        // is why the decoder hands back a mutable slice into the
                        // reused workspace instead of a fresh vector per token.
                        let step_logits = &mut state.workspace.logits_mut()[start_idx..];

                        // Same ordering constraint as the beam path: the
                        // `_NO_SPEECH_` logit is a text token and the timestamp
                        // filters can mask it, so sample it from the raw row.
                        if step == 0 {
                            let ns_idx = no_speech_token as usize;
                            if ns_idx < step_logits.len() {
                                no_speech_prob = softmax_prob_at(step_logits, ns_idx);
                            }
                        }

                        let filter_ctx = LogitFilterContext {
                            step,
                            tokens_cur: &tokens_cur,
                            last_timestamp_token,
                        };
                        greedy_pipeline.apply(step_logits, &filter_ctx);
                        let not_idx = self.model.vocab.token_not as usize;
                        if not_idx < step_logits.len() {
                            step_logits[not_idx] = f32::NEG_INFINITY;
                        }

                        // At temperature zero this is argmax; the fallback
                        // ladder raises the temperature, and upstream then
                        // samples from the distribution instead of repeating
                        // the same argmax forever.
                        let (best_id, best_logit) = if temperature > 0.0 {
                            sample_token(step_logits, &mut rng_state)
                        } else {
                            // Greedy can lock into the same phrase as beam. When
                            // the argmax would close a third 4-gram, mask that
                            // token and take the next-best one.
                            let mut best_id;
                            let mut best_logit;
                            loop {
                                best_id = 0i32;
                                best_logit = f32::NEG_INFINITY;
                                for (i, &l) in step_logits.iter().enumerate() {
                                    if l > best_logit {
                                        best_logit = l;
                                        best_id = i as i32;
                                    }
                                }
                                if best_id >= 0
                                    && best_id < eot
                                    && crate::beam::repeats_ngram(&tokens_cur, best_id, eot)
                                {
                                    step_logits[best_id as usize] = f32::NEG_INFINITY;
                                    continue;
                                }
                                break;
                            }
                            (best_id, best_logit)
                        };

                        let max_l = step_logits
                            .iter()
                            .copied()
                            .fold(f32::NEG_INFINITY, f32::max);
                        let sum_exp: f32 = step_logits
                            .iter()
                            .map(|&l| {
                                if l > f32::NEG_INFINITY {
                                    (l - max_l).exp()
                                } else {
                                    0.0
                                }
                            })
                            .sum();
                        let lse = max_l + sum_exp.max(1e-10).ln();
                        let logprob = best_logit - lse;
                        token_logprobs.push(logprob);

                        let step_entropy = crate::filter::compute_distribution_entropy(step_logits);
                        token_entropies.push(step_entropy);

                        state
                            .timings
                            .add_sample(sample_start.elapsed().as_micros() as u64);

                        if best_id == eot {
                            break;
                        }

                        if best_id >= beg {
                            last_timestamp_token = Some(best_id);
                        }

                        tokens_cur.push(best_id);
                    }
                }

                let text = self.tokenizer.decode(&tokens_cur, true);
                let eval = FallbackEvaluation::evaluate(
                    &text,
                    &token_logprobs,
                    &token_entropies,
                    no_speech_prob,
                    params,
                );

                let keep = match &best_attempt {
                    None => true,
                    Some(previous) => {
                        if !eval.fallback_needed && previous.eval.fallback_needed {
                            true
                        } else if eval.fallback_needed && !previous.eval.fallback_needed {
                            false
                        } else if (eval.compression_ratio - previous.eval.compression_ratio).abs()
                            > 0.05
                        {
                            eval.compression_ratio < previous.eval.compression_ratio
                        } else {
                            // Same repetition level: prefer the attempt that
                            // covers more of the window, because a short sample
                            // silently drops speech that the window contained.
                            tokens_cur.len() > previous.tokens.len()
                        }
                    }
                };
                if keep {
                    best_attempt = Some(DecodeAttempt {
                        tokens: tokens_cur.clone(),
                        token_logprobs: token_logprobs.clone(),
                        token_entropies: token_entropies.clone(),
                        no_speech_prob,
                        eval,
                    });
                }

                if !eval.fallback_needed || params.temperature_inc <= 0.0 {
                    break;
                }
                // A beam path that produced a repetition loop gets one clean
                // greedy retry before the temperature ladder. Greedy does not
                // share the beam's degenerate branch, and a sampled retry was
                // still able to reproduce the same loop.
                if params.beam_size > 1 && !force_greedy && temperature == 0.0 {
                    force_greedy = true;
                    continue;
                }
                // Trigger temperature fallback
                temperature += params.temperature_inc;
                if temperature > 1.0 {
                    break;
                }
                // A retry actually runs, so record which criterion rejected the
                // temperature that was just decoded. whisper.cpp keeps the same
                // two counters behind its timing report.
                if token_entropies.len() >= 32 && eval.avg_entropy > params.entropy_thold {
                    state.timings.n_fail_h += 1;
                }
                if (eval.avg_logprob as f64) < params.logprob_threshold
                    && no_speech_prob < params.no_speech_threshold as f32
                {
                    state.timings.n_fail_p += 1;
                }
            }

            // Restore the best-scoring attempt rather than whatever the last
            // temperature produced; the ladder can end on a short, degenerate
            // sample even when the beam result was usable.
            let eval = if let Some(best) = best_attempt {
                tokens_cur = best.tokens;
                token_logprobs = best.token_logprobs;
                token_entropies = best.token_entropies;
                no_speech_prob = best.no_speech_prob;
                best.eval
            } else {
                FallbackEvaluation::evaluate(
                    "",
                    &token_logprobs,
                    &token_entropies,
                    no_speech_prob,
                    params,
                )
            };
            let avg_logprob = eval.avg_logprob;
            let avg_entropy = eval.avg_entropy;

            // The decoder emits a dedicated no-speech probability at step 0.
            // When it is confident that the window holds no speech and the
            // token log-probs are poor, the window is silence rather than
            // quiet speech, and emitting its tokens is exactly how trailing
            // silence becomes invented text (ggml-org/whisper.cpp#1724,
            // #1026). The rule matches OpenAI Whisper's decoding loop: skip
            // only when both signals agree, so whispered speech with a low
            // no-speech score is still transcribed.
            if params.no_speech_threshold > 0.0
                && no_speech_prob > params.no_speech_threshold as f32
                && (avg_logprob as f64) < params.logprob_threshold
            {
                seek_frame += chunk_frames;
                continue;
            }

            // Token-level timestamps need real cross-attention, so the whole
            // window is aligned once and every segment slices its own tokens
            // out of that alignment.
            let dtw_alignment = if params.dtw_token_timestamps {
                self.compute_window_alignment(
                    state,
                    params.dtw_aheads_preset,
                    &tokens_cur,
                    &enc_out,
                    seek_time,
                    lang_idx,
                    total_audio_sec,
                    beg,
                )
            } else {
                None
            };
            let mut dtw_cursor = 0usize;

            // Extract segments from tokens_cur
            if params.no_timestamps || tokens_cur.is_empty() {
                let max_tok = if params.max_tokens > 0 {
                    params.max_tokens
                } else {
                    tokens_cur.len().max(1)
                };
                for chunk in tokens_cur.chunks(max_tok) {
                    let t1 = (seek_time + chunk_sec).min(total_audio_sec);
                    let seg = self.build_segment(
                        params,
                        dtw_alignment.as_ref(),
                        &mut dtw_cursor,
                        seek_time,
                        t1,
                        chunk,
                        SegmentStats {
                            avg_logprob,
                            no_speech_prob,
                            avg_entropy,
                        },
                    );
                    if !seg.text.trim().is_empty() {
                        if let Some(cb) = &params.on_new_segment {
                            cb(&seg);
                        }
                        segments.push(seg);
                        if params.single_segment {
                            break;
                        }
                    }
                }
                seek_frame += chunk_frames;
                if params.single_segment && !segments.is_empty() {
                    break;
                }
            } else {
                // Commit only the tokens up to the last timestamp. Anything
                // after it is an uncommitted tail: the next window re-decodes
                // that audio, so it must not be emitted here and must not enter
                // the prompt, where the model would continue its own partial
                // sentence instead of the new audio.
                let mut committed_len = 0;
                let mut last_ts_abs = None;
                let mut last_ts_index = None;
                for (index, &tok) in tokens_cur.iter().enumerate() {
                    if tok >= beg {
                        last_ts_abs = Some(seek_time + (tok - beg) as f64 * 0.02);
                        committed_len = index + 1;
                        last_ts_index = Some(index);
                    }
                }
                // A closed run `[text, timestamp]` means the window is
                // complete and the next window starts a full chunk later.
                // Re-decoding that tail adds an extra encoder pass per window;
                // whisper.cpp resizes to `result_len` before this test, so it
                // also starts the next window a full chunk later.
                let ends_text_then_timestamp =
                    last_ts_index.is_some_and(|index| index > 0 && tokens_cur[index - 1] < eot);

                // Default to a full chunk when the decoder produced no
                // timestamp (a failed window) or when the last timestamp is at
                // the window start. A sub-second advance would re-encode the
                // same audio, so it also falls back to the full chunk. This is
                // the same 1 s floor the pre-regression loop used.
                let mut seek_delta = chunk_sec;
                if let Some(last_abs) = last_ts_abs {
                    let last_abs = last_abs.clamp(seek_time, seek_time + chunk_sec);
                    let delta = last_abs - seek_time;
                    if !ends_text_then_timestamp && delta >= 1.0 && chunk_sec - delta >= 1.0 {
                        seek_delta = delta;
                    }
                    // If the only timestamp is the leading `<|0.00|>`, the
                    // decode has text but no closing timestamp. Truncating to
                    // that marker would silently drop the whole window; keep
                    // the text and advance a full chunk as the pre-regression
                    // loop did. A real decoder returns a later timestamp.
                    if committed_len > 1 {
                        tokens_cur.truncate(committed_len);
                    }
                }

                let mut seg_tokens = Vec::new();
                let mut seg_t0 = seek_time;
                let mut last_ts_time = seek_time;

                for &tok in &tokens_cur {
                    if tok >= beg {
                        let rel_sec = (tok - beg) as f64 * 0.02;
                        let abs_sec = seek_time + rel_sec;
                        last_ts_time = abs_sec;

                        if seg_tokens.is_empty() {
                            seg_t0 = abs_sec;
                        } else {
                            let seg = self.build_segment(
                                params,
                                dtw_alignment.as_ref(),
                                &mut dtw_cursor,
                                seg_t0,
                                abs_sec,
                                &seg_tokens,
                                SegmentStats {
                                    avg_logprob,
                                    no_speech_prob,
                                    avg_entropy,
                                },
                            );
                            if !seg.text.trim().is_empty() {
                                if let Some(cb) = &params.on_new_segment {
                                    cb(&seg);
                                }
                                segments.push(seg);
                                if params.single_segment {
                                    seg_tokens.clear();
                                    break;
                                }
                            }
                            seg_tokens.clear();
                            seg_t0 = abs_sec;
                        }
                    } else {
                        seg_tokens.push(tok);
                        if params.max_tokens > 0 && seg_tokens.len() >= params.max_tokens {
                            let t_mid =
                                (seg_t0 + (seg_tokens.len() as f64 * 0.2)).min(total_audio_sec);
                            let seg = self.build_segment(
                                params,
                                dtw_alignment.as_ref(),
                                &mut dtw_cursor,
                                seg_t0,
                                t_mid,
                                &seg_tokens,
                                SegmentStats {
                                    avg_logprob,
                                    no_speech_prob,
                                    avg_entropy,
                                },
                            );
                            if !seg.text.trim().is_empty() {
                                if let Some(cb) = &params.on_new_segment {
                                    cb(&seg);
                                }
                                segments.push(seg);
                                if params.single_segment {
                                    seg_tokens.clear();
                                    break;
                                }
                            }
                            seg_tokens.clear();
                            seg_t0 = t_mid;
                        }
                    }
                }
                if !seg_tokens.is_empty() && (!params.single_segment || segments.is_empty()) {
                    // Do not claim the whole window for an unclosed tail: the
                    // next seek must start where the last timestamp ended so
                    // the remaining audio is decoded instead of skipped.
                    let window_end = (seek_time + chunk_sec).min(total_audio_sec);
                    let t1 = if last_ts_time > seg_t0 {
                        last_ts_time.min(window_end)
                    } else {
                        window_end
                    };
                    let seg = self.build_segment(
                        params,
                        dtw_alignment.as_ref(),
                        &mut dtw_cursor,
                        seg_t0,
                        t1,
                        &seg_tokens,
                        SegmentStats {
                            avg_logprob,
                            no_speech_prob,
                            avg_entropy,
                        },
                    );
                    if !seg.text.trim().is_empty() {
                        if let Some(cb) = &params.on_new_segment {
                            cb(&seg);
                        }
                        segments.push(seg);
                    }
                }

                if params.single_segment && !segments.is_empty() {
                    seek_frame += chunk_frames;
                    let _ = seek_frame;
                    break;
                }

                let advanced_frames = (seek_delta / 0.01).round() as usize;
                // A sub-second advance re-encodes the same window; the 1 s
                // floor keeps a degenerate timestamp from turning the loop
                // into thousands of overlapping decodes.
                let step = if advanced_frames >= 100 {
                    advanced_frames
                } else {
                    chunk_frames
                };
                seek_frame += step;
                // Keep only ordinary text tokens; timestamps and all special tokens
                // (including <|notimestamps|>) must not enter the next prompt.
                history_tokens.extend(tokens_cur.iter().copied().filter(|&t| t >= 0 && t < eot));
            }
        }

        if let Some(cb) = &params.on_progress {
            cb(100);
        }

        segments = deduplicate_segments(segments);

        if params.max_len > 0 {
            segments = split_segments_by_max_len(segments, params.max_len, params.split_on_word);
        }

        segments
    }

    #[allow(clippy::too_many_arguments)]
    fn build_segment(
        &self,
        params: &WhisperParams,
        alignment: Option<&WindowAlignment>,
        dtw_cursor: &mut usize,
        seg_t0: f64,
        seg_t1: f64,
        tokens: &[i32],
        stats: SegmentStats,
    ) -> Segment {
        let mut text = self.tokenizer.decode(tokens, true);
        if params.tdrz_enable
            && tokens.iter().any(|&t| t == 50363 || t == 50364)
            && !text.starts_with("[SPEAKER_TURN]")
        {
            text = format!("[SPEAKER_TURN] {}", text.trim());
        }

        // Word timings are derived from ordinary text tokens only. Timestamp
        // and other special tokens are segment separators, not speech.
        let eot = self.tokenizer.eot();
        let text_tokens: Vec<i32> = tokens
            .iter()
            .copied()
            .filter(|&tok| tok >= 0 && tok < eot)
            .collect();
        let token_tuples: Vec<(&str, f32)> = text_tokens
            .iter()
            .map(|&tok| (self.tokenizer.id_to_token(tok).unwrap_or(""), 0.95))
            .collect();

        let words = alignment
            .and_then(|alignment| {
                let start = locate_segment_alignment(alignment, &text_tokens, dtw_cursor)?;
                let alignments = &alignment.alignments[start..start + text_tokens.len()];
                Some(merge_tokens_to_words_clamped(
                    &token_tuples,
                    alignments,
                    seg_t0,
                    seg_t1,
                    alignment.seek_time,
                    0.02,
                ))
            })
            .unwrap_or_else(|| {
                // No calibrated heads, no text, or an unexpected token order:
                // fall back to the uniform layout approximation, which is not
                // acoustic alignment.
                approximate_word_timings(&token_tuples, seg_t0, seg_t1, 0.02)
            });

        Segment::with_stats(
            seg_t0,
            seg_t1,
            text,
            tokens.to_vec(),
            words,
            stats.avg_logprob,
            stats.no_speech_prob,
            stats.avg_entropy,
        )
    }

    /// Align every text token of one audio window with the decoder's real
    /// cross-attention.
    ///
    /// A single decoder pass runs over `SOT + [language] + NOT + text + EOT`
    /// while capturing the selected alignment heads, mirroring whisper.cpp's
    /// `whisper_exp_compute_token_level_timestamps_dtw`. The captured weights
    /// are normalized over the token axis, median-filtered along audio, and
    /// reduced to a DTW cost matrix. Returns `None` when no calibration head
    /// applies to this model or the window carries no text tokens.
    #[allow(clippy::too_many_arguments)]
    fn compute_window_alignment(
        &self,
        state: &mut WhisperState,
        preset: crate::dtw::AlignmentHeadsPreset,
        tokens: &[i32],
        enc_out: &[f32],
        seek_time: f64,
        lang_idx: usize,
        total_audio_sec: f64,
        beg: i32,
    ) -> Option<WindowAlignment> {
        let heads: Vec<(usize, usize)> = get_alignment_heads(preset)
            .iter()
            .copied()
            .filter(|&(layer, head)| layer < self.decoder.n_layers && head < self.decoder.n_heads)
            .collect();
        if heads.is_empty() || enc_out.is_empty() {
            return None;
        }

        let eot = self.tokenizer.eot();
        let text_tokens: Vec<i32> = tokens
            .iter()
            .copied()
            .filter(|&tok| tok >= 0 && tok < eot)
            .collect();
        if text_tokens.is_empty() {
            return None;
        }

        let sot = self.tokenizer.sot();
        let mut align_tokens = Vec::with_capacity(text_tokens.len() + 4);
        align_tokens.push(sot);
        if self.model.hparams.is_multilingual() {
            align_tokens.push(sot + 1 + lang_idx as i32);
        }
        align_tokens.push(self.tokenizer.not());
        let prefix = align_tokens.len();
        align_tokens.extend_from_slice(&text_tokens);
        align_tokens.push(eot);

        let n_audio_ctx = enc_out.len() / self.decoder.d_model;
        if n_audio_ctx == 0 {
            return None;
        }

        // Drop the encoder padding past the audio the window actually
        // consumed: whisper.cpp clamps the DTW audio extent to the last
        // emitted timestamp, and padded frames only add noise to the cost.
        let rel_end = tokens
            .iter()
            .filter(|&&tok| tok >= beg)
            .map(|&tok| (tok - beg) as usize)
            .max()
            .filter(|&frames| frames > 0)
            .unwrap_or_else(|| (((total_audio_sec - seek_time).max(0.0)) / 0.02).ceil() as usize);
        let n_audio = rel_end.max(1).min(n_audio_ctx);

        // The alignment pass is independent of the sampling pass. Reuse the
        // state's arena and activation workspace, but force the cross-attention
        // projections to be recomputed for this encoder output.
        state.cache.clear();
        state.cache.invalidate_cross();
        let _ = self.decoder.decode_with_alignment_heads(
            &align_tokens,
            enc_out,
            &mut state.cache,
            &mut state.workspace,
            &heads,
        );
        let attention = state.workspace.alignment_attention();
        let cost_full =
            dtw_cost_from_attention(attention, align_tokens.len(), n_audio_ctx, heads.len(), 7);
        if cost_full.len() != align_tokens.len() * n_audio_ctx {
            return None;
        }

        // DTW runs over the text rows only; the SOT/language/NOT prefix and the
        // trailing EOT still contribute to the normalization statistics.
        let n_text = text_tokens.len();
        let mut cost = vec![0.0f32; n_text * n_audio];
        for (i, row) in cost.chunks_exact_mut(n_audio).enumerate() {
            let src = (prefix + i) * n_audio_ctx;
            row.copy_from_slice(&cost_full[src..src + n_audio]);
        }

        let alignments = align_dtw_from_cost(&cost, n_text, n_audio);
        if alignments.len() != n_text {
            return None;
        }

        Some(WindowAlignment {
            seek_time,
            text_tokens,
            alignments,
        })
    }
}

/// Find the slice of a window alignment that belongs to one segment.
///
/// Segments are emitted in token order, so the shared cursor normally matches
/// directly. A forward search keeps the mapping correct if a caller ever
/// reorders segment construction; it never rewinds past tokens already used.
fn locate_segment_alignment(
    alignment: &WindowAlignment,
    text_tokens: &[i32],
    cursor: &mut usize,
) -> Option<usize> {
    if text_tokens.is_empty() {
        return None;
    }
    let n = text_tokens.len();
    let available = alignment.text_tokens.len();
    if *cursor + n <= available && alignment.text_tokens[*cursor..*cursor + n] == *text_tokens {
        let start = *cursor;
        *cursor += n;
        return Some(start);
    }
    if available < n {
        return None;
    }
    for start in *cursor..=available - n {
        if alignment.text_tokens[start..start + n] == *text_tokens {
            *cursor = start + n;
            return Some(start);
        }
    }
    None
}

/// Evaluates the optional encoder-begin callback; `true` means the pass may run.
fn encoder_begin_allowed(params: &WhisperParams) -> bool {
    params.on_encoder_begin.as_ref().is_none_or(|cb| cb())
}

fn split_segments_by_max_len(
    segments: Vec<Segment>,
    max_len: usize,
    _split_on_word: bool,
) -> Vec<Segment> {
    if max_len == 0 {
        return segments;
    }

    let mut out = Vec::new();
    for seg in segments {
        if seg.text.trim().chars().count() <= max_len || seg.words.is_empty() {
            out.push(seg);
            continue;
        }

        let mut cur_words: Vec<crate::dtw::WordTiming> = Vec::new();
        let mut cur_len = 0;

        for w in seg.words {
            let is_contraction = w.word.starts_with('\'')
                || cur_words
                    .last()
                    .map(|cw: &crate::dtw::WordTiming| cw.word.ends_with('-'))
                    .unwrap_or(false);
            let w_len = w.word.chars().count() + if cur_len > 0 { 1 } else { 0 };
            if cur_len + w_len > max_len && !cur_words.is_empty() && !is_contraction {
                let t0 = cur_words.first().unwrap().t0;
                let t1 = cur_words.last().unwrap().t1;
                let text = cur_words
                    .iter()
                    .map(|cw| cw.word.as_str())
                    .collect::<Vec<_>>()
                    .join(" ");
                out.push(Segment::with_stats(
                    t0,
                    t1,
                    text,
                    Vec::new(),
                    cur_words,
                    seg.avg_logprob,
                    seg.no_speech_prob,
                    seg.avg_entropy,
                ));
                cur_words = Vec::new();
                cur_len = 0;
            }
            cur_len += w.word.chars().count() + if cur_len > 0 { 1 } else { 0 };
            cur_words.push(w);
        }

        if !cur_words.is_empty() {
            let t0 = cur_words.first().unwrap().t0;
            let t1 = cur_words.last().unwrap().t1;
            let text = cur_words
                .iter()
                .map(|cw| cw.word.as_str())
                .collect::<Vec<_>>()
                .join(" ");
            out.push(Segment::with_stats(
                t0,
                t1,
                text,
                Vec::new(),
                cur_words,
                seg.avg_logprob,
                seg.no_speech_prob,
                seg.avg_entropy,
            ));
        }
    }

    out
}

/// Helper responsible for assembling window prompts according to transcription parameters.
#[derive(Debug)]
struct SegmentStats {
    avg_logprob: f32,
    no_speech_prob: f32,
    avg_entropy: f32,
}

/// Token-level DTW alignment for one audio window.
///
/// Captured once per window so every segment in that window shares the same
/// cross-attention cost matrix, exactly like whisper.cpp's
/// `whisper_exp_compute_token_level_timestamps_dtw`. Frame indices are
/// relative to the window start.
struct WindowAlignment {
    seek_time: f64,
    text_tokens: Vec<i32>,
    alignments: Vec<(usize, usize, usize)>,
}

pub struct WindowPromptBuilder<'a> {
    pub prompt_base: &'a [i32],
    pub is_multilingual: bool,
    pub lang_idx: usize,
    pub sot: i32,
    pub not_token: i32,
    pub token_prev: i32,
}

impl<'a> WindowPromptBuilder<'a> {
    pub fn build(
        &self,
        seek_frame: usize,
        history_tokens: &[i32],
        params: &WhisperParams,
        tokenizer: &Tokenizer,
    ) -> Vec<i32> {
        let mut window_prompt = Vec::new();
        if seek_frame == 0 || params.no_context {
            window_prompt.extend_from_slice(self.prompt_base);
        } else {
            let prev_tokens_slice = if history_tokens.len() > 224 {
                &history_tokens[history_tokens.len() - 224..]
            } else {
                history_tokens
            };

            if !prev_tokens_slice.is_empty() {
                window_prompt.push(self.token_prev);
                if params.carry_initial_prompt {
                    if !params.prompt_tokens.is_empty() {
                        window_prompt.extend_from_slice(&params.prompt_tokens);
                    } else if let Some(prompt_text) = &params.initial_prompt {
                        window_prompt.extend(tokenizer.encode(prompt_text));
                    }
                }
                window_prompt.extend_from_slice(prev_tokens_slice);
            }

            window_prompt.push(self.sot);
            if self.is_multilingual {
                let lang_token = self.sot + 1 + self.lang_idx as i32;
                window_prompt.push(lang_token);
                if params.translate {
                    window_prompt.push(tokenizer.translate());
                } else {
                    window_prompt.push(tokenizer.transcribe());
                }
            }
            if params.no_timestamps {
                window_prompt.push(self.not_token);
            }
        }
        // The decoder context is 448 positions and generation can consume up
        // to 224 more. Keep the prompt at half the context, dropping the oldest
        // history first; the trailing SOT/language/task tokens must survive.
        const MAX_PROMPT_TOKENS: usize = 224;
        if window_prompt.len() > MAX_PROMPT_TOKENS {
            let drop = window_prompt.len() - MAX_PROMPT_TOKENS;
            // `<|startofprev|>` is the only marker that distinguishes prompt
            // conditioning from a bare prefix; never drain it with the history.
            let start = usize::from(window_prompt.first() == Some(&self.token_prev));
            window_prompt.drain(start..start + drop);
        }

        window_prompt
    }
}

/// Log-sum-exp over a finite logits row.
fn log_sum_exp(logits: &[f32]) -> f32 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if max == f32::NEG_INFINITY {
        return f32::NEG_INFINITY;
    }
    let sum: f32 = logits
        .iter()
        .map(|&value| {
            if value > f32::NEG_INFINITY {
                (value - max).exp()
            } else {
                0.0
            }
        })
        .sum();
    max + sum.max(1e-10).ln()
}

/// Softmax probability of one token without allocating.
fn softmax_prob_at(logits: &[f32], index: usize) -> f32 {
    let lse = log_sum_exp(logits);
    if lse == f32::NEG_INFINITY || index >= logits.len() {
        return 0.0;
    }
    (logits[index] - lse).exp().clamp(0.0, 1.0)
}

/// Draw one token from a logits row with an xorshift64 generator.
fn sample_token(logits: &[f32], state: &mut u64) -> (i32, f32) {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if max == f32::NEG_INFINITY {
        return (0, f32::NEG_INFINITY);
    }
    let weight = |value: f32| -> f64 {
        if value > f32::NEG_INFINITY {
            ((value - max) as f64).exp()
        } else {
            0.0
        }
    };

    let sum: f64 = logits.iter().map(|&value| weight(value)).sum();
    if sum <= 0.0 {
        return (0, f32::NEG_INFINITY);
    }

    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    let mut target = (*state as f64 / u64::MAX as f64) * sum;
    for (index, &value) in logits.iter().enumerate() {
        target -= weight(value);
        if target <= 0.0 {
            return (index as i32, value);
        }
    }
    (logits.len() as i32 - 1, logits[logits.len() - 1])
}

/// Normalize segment text for duplicate detection without allocating a regex.
fn normalized_segment_text(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        if ch.is_alphanumeric() {
            out.extend(ch.to_lowercase());
        } else if !out.is_empty() && !out.ends_with(' ') {
            out.push(' ');
        }
    }
    out.trim().to_string()
}

/// Drop segments that repeat an earlier segment over the same audio.
///
/// Only the emitted timeline is compared: identical normalized text plus an
/// identical span (within the 0.05 s timestamp tolerance), or a zero-length run
/// closed by two equal timestamps. A genuinely repeated reference line, such as
/// the station ID 5 s after the first copy, has a different span and is kept.
fn deduplicate_segments(segments: Vec<Segment>) -> Vec<Segment> {
    let mut kept: Vec<Segment> = Vec::with_capacity(segments.len());
    let mut normalized: Vec<String> = Vec::with_capacity(segments.len());
    for segment in segments {
        let text = normalized_segment_text(&segment.text);
        let duplicate = !text.is_empty()
            && kept
                .iter()
                .zip(&normalized)
                .any(|(previous, previous_text)| {
                    if *previous_text != text {
                        return false;
                    }
                    let same_span = (segment.t0 - previous.t0).abs() <= 0.05
                        && (segment.t1 - previous.t1).abs() <= 0.05;
                    let zero_length = segment.t1 - segment.t0 < 0.05;
                    same_span || zero_length
                });
        if !duplicate {
            normalized.push(text);
            kept.push(segment);
        }
    }
    kept
}

/// True when the text contains an obvious repetition loop.
///
/// The compression-ratio check catches many loops, but a repeated phrase inside
/// an otherwise varied window can slip under the 2.4 threshold. A repeated
/// 8-gram is a direct signal that the decoder is stuck, which is exactly when
/// the temperature fallback must retry.
fn has_repetition_loop(text: &str) -> bool {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() < 32 {
        return false;
    }

    let mut counts: std::collections::HashMap<[u64; 8], u32> = std::collections::HashMap::new();
    for window in words.windows(8) {
        let mut key = [0u64; 8];
        for (slot, word) in key.iter_mut().zip(window) {
            // FNV-1a keeps the key cheap; collisions only make the detector
            // slightly conservative.
            let mut hash = 0xcbf29ce484222325u64;
            for byte in word.as_bytes() {
                hash ^= *byte as u64;
                hash = hash.wrapping_mul(0x100000001b3);
            }
            *slot = hash;
        }
        let entry = counts.entry(key).or_insert(0);
        *entry += 1;
        if *entry >= 4 {
            return true;
        }
    }
    false
}

/// One decoded attempt in the temperature fallback ladder.
struct DecodeAttempt {
    tokens: Vec<i32>,
    token_logprobs: Vec<f32>,
    token_entropies: Vec<f32>,
    no_speech_prob: f32,
    eval: FallbackEvaluation,
}

/// Evaluator for decoding quality metrics against temperature fallback criteria.
#[derive(Debug, Clone, Copy)]
pub struct FallbackEvaluation {
    pub avg_logprob: f32,
    pub avg_entropy: f32,
    pub compression_ratio: f64,
    pub fallback_needed: bool,
}

impl FallbackEvaluation {
    pub fn evaluate(
        text: &str,
        token_logprobs: &[f32],
        token_entropies: &[f32],
        no_speech_prob: f32,
        params: &WhisperParams,
    ) -> Self {
        let avg_logprob = if !token_logprobs.is_empty() {
            token_logprobs.iter().sum::<f32>() / token_logprobs.len() as f32
        } else {
            -0.2
        };

        let avg_entropy = if !token_entropies.is_empty() {
            token_entropies.iter().sum::<f32>() / token_entropies.len() as f32
        } else {
            0.0
        };

        let cr = compression_ratio(text);
        let fallback_needed = cr > params.compression_ratio_threshold
            || has_repetition_loop(text)
            || (avg_logprob < params.logprob_threshold as f32
                && no_speech_prob < params.no_speech_threshold as f32)
            || (token_entropies.len() >= 32 && avg_entropy > params.entropy_thold);

        Self {
            avg_logprob,
            avg_entropy,
            compression_ratio: cr,
            fallback_needed,
        }
    }
}

#[cfg(test)]
mod repetition_guard_tests {
    use super::has_repetition_loop;

    #[test]
    fn detects_a_repeated_eight_gram() {
        let text = "dan menerima kasih ".repeat(12);
        assert!(has_repetition_loop(&text));
    }

    #[test]
    fn leaves_varied_speech_alone() {
        let text =
            "Sobat Trisma hari ini kita bakal ngebahas tentang perayaan unik di bulan Desember";
        assert!(!has_repetition_loop(text));
    }
}

#[cfg(test)]
mod dedupe_tests {
    use super::deduplicate_segments;
    use crate::segment::Segment;

    fn segment(t0: f64, t1: f64, text: &str) -> Segment {
        Segment::new(t0, t1, text.to_string(), Vec::new())
    }

    #[test]
    fn drops_a_collapsed_duplicate_at_a_seam() {
        let segments = vec![
            segment(27.94, 29.98, "Jangan lupa buat jaga diri baik-baik"),
            segment(29.98, 30.00, "Jangan lupa buat jaga diri baik-baik"),
        ];
        assert_eq!(deduplicate_segments(segments).len(), 1);
    }

    #[test]
    fn keeps_a_genuinely_repeated_reference_line() {
        let segments = vec![
            segment(0.00, 3.26, "Voice of Trisma, the best online radio in Bali"),
            segment(3.26, 8.94, "Voice of Trisma, the best online radio in Bali"),
        ];
        assert_eq!(deduplicate_segments(segments).len(), 2);
    }
}
