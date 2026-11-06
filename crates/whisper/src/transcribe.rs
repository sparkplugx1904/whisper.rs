use crate::beam::BeamSearch;
use crate::decoder::{DecoderKvCache, WhisperDecoder};
use crate::encoder::WhisperEncoder;
use crate::filter::{
    compression_ratio, GrammarLogitFilter, LogitFilterContext, LogitFilterPipeline, RegexFilter,
    RepetitionPenaltyFilter, SuppressBlankFilter, SuppressNonSpeechFilter, TimestampLogitFilter,
};
use crate::params::WhisperParams;
use crate::segment::Segment;
use crate::state::WhisperState;
use std::path::Path;
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
        WhisperState::new(self.decoder.n_layers)
    }

    /// Full end-to-end transcription pipeline matching whisper.cpp: whisper_full()
    pub fn full(
        &self,
        state: &mut WhisperState,
        params: &WhisperParams,
        samples: &[f32],
    ) -> Result<(), AudioError> {
        state.clear();
        if samples.len() < 1600 {
            return Ok(());
        }
        let mel = whisper_audio::compute_log_mel_spectrogram(
            samples,
            &self.model.filters.data,
            self.model.filters.n_mel,
        );

        state.segments = self.transcribe_internal(&mel, params, state);
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

        let mut cache = DecoderKvCache::new(self.decoder.n_layers);
        let prompt = [self.tokenizer.sot()];
        let logits = self.decoder.decode(&prompt, &enc_out, &mut cache);

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
        if !self.model.hparams.is_multilingual() {
            return ("en".to_string(), 1.0);
        }

        let mut cache = DecoderKvCache::new(self.decoder.n_layers);
        let prompt = [self.tokenizer.sot()];
        let logits = self.decoder.decode(&prompt, enc_out, &mut cache);

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

        let mut sum_exp = 0.0f32;
        for (_, score) in &mut lang_scores {
            *score = (*score - max_score).exp();
            sum_exp += *score;
        }

        let inv_sum = 1.0 / sum_exp.max(1e-10);
        let mut best_lang = "en";
        let mut best_prob = 0.0f32;

        for (code, score) in lang_scores {
            let prob = score * inv_sum;
            if prob > best_prob {
                best_prob = prob;
                best_lang = code;
            }
        }

        (best_lang.to_string(), best_prob)
    }

    pub fn detect_language(&self, mel: &MelSpectrogram) -> (String, f32) {
        let mel_chunk = mel.get_chunk(0, self.model.hparams.n_audio_ctx as usize);
        let enc_out = self.encoder.forward(&mel_chunk);
        self.detect_language_with_enc(&enc_out)
    }

    /// Transcribe whole audio mel spectrogram into text segments.
    pub fn transcribe(&self, mel: &MelSpectrogram, params: &WhisperParams) -> Vec<Segment> {
        let mut state = self.create_state();
        self.transcribe_internal(mel, params, &mut state)
    }

    fn transcribe_internal(
        &self,
        mel: &MelSpectrogram,
        params: &WhisperParams,
        state: &mut WhisperState,
    ) -> Vec<Segment> {
        let is_multilingual = self.model.hparams.is_multilingual();
        let n_audio_ctx =
            if params.audio_ctx > 0 && params.audio_ctx < self.model.hparams.n_audio_ctx as usize {
                params.audio_ctx
            } else {
                self.model.hparams.n_audio_ctx as usize
            };

        // Precompute chunk 0 encoder output so we never compute it twice
        let mel_chunk_0 = mel.get_chunk(0, n_audio_ctx);
        let enc_out_0 = self.encoder.forward_with_ctx(&mel_chunk_0, n_audio_ctx);

        if params.abort_callback.as_ref().is_some_and(|cb| cb()) {
            return Vec::new();
        }

        let lang = if params.language == "auto" {
            let (detected, prob) = self.detect_language_with_enc(&enc_out_0);
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
            let seek_time = seek_frame as f64 * 0.01;
            let enc_out = if seek_frame == 0 {
                enc_out_0.clone()
            } else {
                let mel_chunk = mel.get_chunk(seek_frame, n_audio_ctx);
                self.encoder.forward_with_ctx(&mel_chunk, n_audio_ctx)
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

            let gbnf_grammar = params
                .grammar
                .as_deref()
                .and_then(|g| whisper_grammar::GbnfGrammar::parse(g).ok());

            // Temperature fallback loop (e.g. 0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
            while temperature <= 1.0 {
                tokens_cur.clear();
                token_logprobs.clear();
                token_entropies.clear();
                let mut cache = DecoderKvCache::new(self.decoder.n_layers);

                if params.beam_size > 1 && temperature == 0.0 {
                    // BEAM SEARCH
                    let mut logits = self.decoder.decode(&window_prompt, &enc_out, &mut cache);
                    let mut beam = BeamSearch::new(params.beam_size, cache);

                    // Seed beam candidates with top branches from initial prompt
                    let n_vocab = self.decoder.n_vocab;
                    let start_idx = logits.len() - n_vocab;
                    let step_logits = &mut logits[start_idx..];

                    let mut beam_pipeline = LogitFilterPipeline::new();
                    if params.suppress_blank {
                        beam_pipeline.add(SuppressBlankFilter::new(220, eot));
                    }
                    beam_pipeline.add(TimestampLogitFilter::with_max_initial_ts(
                        beg as usize,
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
                    let seed_ctx = LogitFilterContext {
                        step: 0,
                        tokens_cur: &[],
                        last_timestamp_token: None,
                    };
                    beam_pipeline.apply(step_logits, &seed_ctx);

                    let mut seed_tokens = Vec::new();
                    for (tok, &l) in step_logits.iter().enumerate() {
                        if l > f32::NEG_INFINITY {
                            seed_tokens.push((tok as i32, l));
                        }
                    }
                    seed_tokens
                        .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    seed_tokens.truncate(params.beam_size);

                    beam.candidates = seed_tokens
                        .into_iter()
                        .map(|(tok, l)| crate::beam::BeamCandidate {
                            tokens: vec![tok],
                            cumulative_logprob: l as f64,
                            cache: beam.candidates[0].cache.clone(),
                            completed: tok == eot,
                            last_timestamp: if tok >= beg { Some(tok) } else { None },
                        })
                        .collect();

                    let max_steps = 224;
                    for step in 1..max_steps {
                        if beam.is_completed(1.0) {
                            break;
                        }
                        beam.step(
                            &self.decoder,
                            &enc_out,
                            step,
                            eot,
                            beg,
                            params.no_timestamps,
                            params.suppress_blank,
                            gbnf_grammar.as_ref(),
                            &self.model.vocab,
                            &self.tokenizer,
                            if params.length_penalty > 0.0 {
                                params.length_penalty
                            } else {
                                0.7
                            },
                        );
                    }

                    if let Some(best) = beam.best_candidate() {
                        tokens_cur = best.tokens.clone();
                    }
                } else {
                    // GREEDY DECODING
                    let mut logits = self.decoder.decode(&window_prompt, &enc_out, &mut cache);
                    let mut last_timestamp_token: Option<i32> = None;
                    let max_steps = 224;
                    let n_vocab = self.decoder.n_vocab;
                    let mut greedy_pipeline = LogitFilterPipeline::new();
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

                    for step in 0..max_steps {
                        if params.abort_callback.as_ref().is_some_and(|cb| cb()) {
                            break;
                        }
                        let start_idx = logits.len() - n_vocab;
                        let step_logits = &mut logits[start_idx..];

                        let filter_ctx = LogitFilterContext {
                            step,
                            tokens_cur: &tokens_cur,
                            last_timestamp_token,
                        };
                        greedy_pipeline.apply(step_logits, &filter_ctx);

                        // Greedy argmax & logprob calculation
                        let mut best_id = 0;
                        let mut best_logit = f32::NEG_INFINITY;
                        for (i, &l) in step_logits.iter().enumerate() {
                            if l > best_logit {
                                best_logit = l;
                                best_id = i as i32;
                            }
                        }

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

                        if step == 0 && n_vocab > 50362 {
                            let ns_logit = step_logits[50362];
                            if ns_logit > f32::NEG_INFINITY {
                                no_speech_prob = (ns_logit - lse).exp().clamp(0.0, 1.0);
                            }
                        }

                        if best_id == eot {
                            break;
                        }

                        if best_id >= beg {
                            last_timestamp_token = Some(best_id);
                        }

                        tokens_cur.push(best_id);
                        logits = self.decoder.decode(&[best_id], &enc_out, &mut cache);
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

                if !eval.fallback_needed || params.temperature_inc <= 0.0 {
                    break;
                }
                // Trigger temperature fallback
                temperature += params.temperature_inc;
            }

            let eval = FallbackEvaluation::evaluate(
                "",
                &token_logprobs,
                &token_entropies,
                no_speech_prob,
                params,
            );
            let avg_logprob = eval.avg_logprob;
            let avg_entropy = eval.avg_entropy;

            // Extract segments from tokens_cur
            if params.no_timestamps || tokens_cur.is_empty() {
                let max_tok = if params.max_tokens > 0 {
                    params.max_tokens
                } else {
                    tokens_cur.len().max(1)
                };
                let chunk_sec = chunk_frames as f64 * 0.01;
                for chunk in tokens_cur.chunks(max_tok) {
                    let t1 = (seek_time + chunk_sec).min(total_audio_sec);
                    let seg = self.build_segment(
                        seek_time,
                        t1,
                        chunk,
                        avg_logprob,
                        no_speech_prob,
                        avg_entropy,
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
                let mut seg_tokens = Vec::new();
                let mut seg_t0 = seek_time;
                let mut last_end_time = seek_time;

                for &tok in &tokens_cur {
                    if tok >= beg {
                        let rel_sec = (tok - beg) as f64 * 0.02;
                        let abs_sec = seek_time + rel_sec;

                        if seg_tokens.is_empty() {
                            seg_t0 = abs_sec;
                        } else {
                            let seg = self.build_segment(
                                seg_t0,
                                abs_sec,
                                &seg_tokens,
                                avg_logprob,
                                no_speech_prob,
                                avg_entropy,
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
                            last_end_time = abs_sec;
                        }
                    } else {
                        seg_tokens.push(tok);
                        if params.max_tokens > 0 && seg_tokens.len() >= params.max_tokens {
                            let t_mid =
                                (seg_t0 + (seg_tokens.len() as f64 * 0.2)).min(total_audio_sec);
                            let seg = self.build_segment(
                                seg_t0,
                                t_mid,
                                &seg_tokens,
                                avg_logprob,
                                no_speech_prob,
                                avg_entropy,
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
                    let t1 = (seek_time + 30.0).min(total_audio_sec);
                    let seg = self.build_segment(
                        seg_t0,
                        t1,
                        &seg_tokens,
                        avg_logprob,
                        no_speech_prob,
                        avg_entropy,
                    );
                    if !seg.text.trim().is_empty() {
                        if let Some(cb) = &params.on_new_segment {
                            cb(&seg);
                        }
                        segments.push(seg);
                    }
                    last_end_time = t1;
                }

                if params.single_segment && !segments.is_empty() {
                    seek_frame += chunk_frames;
                    let _ = seek_frame;
                    break;
                }

                let advanced_frames = ((last_end_time - seek_time) / 0.01).round() as usize;
                if advanced_frames >= 100 {
                    seek_frame += advanced_frames;
                } else {
                    seek_frame += chunk_frames;
                }
                history_tokens.extend(tokens_cur.iter().copied().filter(|&t| t < beg && t != eot));
            }
        }

        if let Some(cb) = &params.on_progress {
            cb(100);
        }

        if params.max_len > 0 {
            segments = split_segments_by_max_len(segments, params.max_len, params.split_on_word);
        }

        segments
    }

    fn build_segment(
        &self,
        seg_t0: f64,
        seg_t1: f64,
        tokens: &[i32],
        avg_logprob: f32,
        no_speech_prob: f32,
        avg_entropy: f32,
    ) -> Segment {
        let mut text = self.tokenizer.decode(tokens, true);
        if tokens.iter().any(|&t| t == 50363 || t == 50364) && !text.starts_with("[SPEAKER_TURN]") {
            text = format!("[SPEAKER_TURN] {}", text.trim());
        }
        let token_tuples: Vec<(&str, f32)> = tokens
            .iter()
            .map(|&tok| (self.tokenizer.id_to_token(tok).unwrap_or(""), 0.95))
            .collect();

        let n_frames_seg = (((seg_t1 - seg_t0) / 0.02).round() as usize).max(1);
        let mut att = vec![0.0f32; tokens.len() * n_frames_seg];
        for (ti, row) in att.chunks_exact_mut(n_frames_seg).enumerate() {
            let center_f = (ti * n_frames_seg) / tokens.len().max(1);
            for (fj, val) in row.iter_mut().enumerate() {
                let dist = (fj as isize - center_f as isize).abs() as f32;
                *val = (-dist * dist / 20.0).exp();
            }
        }

        let alignments = crate::dtw::align_dtw(&att, tokens.len(), n_frames_seg);
        let words = crate::dtw::merge_tokens_to_words(&token_tuples, &alignments, seg_t0, 0.02);

        Segment::with_stats(
            seg_t0,
            seg_t1,
            text,
            tokens.to_vec(),
            words,
            avg_logprob,
            no_speech_prob,
            avg_entropy,
        )
    }
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
        window_prompt
    }
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
