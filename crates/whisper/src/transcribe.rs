use crate::beam::BeamSearch;
use crate::decoder::{DecoderKvCache, WhisperDecoder};
use crate::encoder::WhisperEncoder;
use crate::filter::compression_ratio;
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
        let mel = whisper_audio::compute_log_mel_spectrogram(
            samples,
            &self.model.filters.data,
            self.model.filters.n_mel,
        );

        state.clear();
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
        let audio = whisper_audio::WavAudio::load_from_file(wav_path)?;
        let samples = if audio.sample_rate != WHISPER_SAMPLE_RATE {
            whisper_audio::resample_to_16k(&audio.samples, audio.sample_rate)
        } else {
            audio.samples
        };
        Ok(self.transcribe_pcm(&samples, params))
    }

    /// Auto-detect the spoken language from the first 30 seconds of audio.
    /// Returns (language_code, probability).
    pub fn detect_language(&self, mel: &MelSpectrogram) -> (String, f32) {
        if !self.model.hparams.is_multilingual() {
            return ("en".to_string(), 1.0);
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
        let lang = if params.language == "auto" {
            let (detected, prob) = self.detect_language(mel);
            state.lang_prob = prob;
            detected
        } else {
            params.language.clone()
        };

        let lang_idx = whisper_model::vocab::lang_id(&lang).unwrap_or(0);
        state.lang_id = lang_idx;

        let sot = self.tokenizer.sot();
        let eot = self.tokenizer.eot();
        let beg = self.tokenizer.beg();
        let not_token = self.tokenizer.not();

        let mut prompt_base = Vec::new();

        // 1. Initial prompt injection
        if let Some(prompt_text) = &params.initial_prompt {
            if !prompt_text.trim().is_empty() {
                prompt_base.push(self.model.vocab.token_prev);
                for word in prompt_text.split_whitespace() {
                    let word_with_space = format!(" {}", word);
                    if let Some(id) = self.tokenizer.token_to_id(&word_with_space) {
                        prompt_base.push(id);
                    } else if let Some(id) = self.tokenizer.token_to_id(word) {
                        prompt_base.push(id);
                    }
                }
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
        let mut seek_frame = 0;
        let n_audio_ctx = self.model.hparams.n_audio_ctx as usize;
        let chunk_frames = 2 * n_audio_ctx; // 3000 frames = 30 seconds
        let total_audio_sec = mel.n_len_org as f64 * 0.01;

        while seek_frame < mel.n_len_org {
            // Stop if less than 100 ms left
            if seek_frame + 10 >= mel.n_len_org {
                break;
            }

            let seek_time = seek_frame as f64 * 0.01;
            let mel_chunk = mel.get_chunk(seek_frame, n_audio_ctx);
            let enc_out = self.encoder.forward(&mel_chunk);
            state.encoder_output = enc_out.clone();

            let mut temperature = params.temperature;
            let mut tokens_cur = Vec::new();

            // Temperature fallback loop (e.g. 0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
            while temperature <= 1.0 {
                let mut cache = DecoderKvCache::new(self.decoder.n_layers);

                if params.beam_size > 1 && temperature == 0.0 {
                    // BEAM SEARCH
                    let mut logits = self.decoder.decode(&prompt_base, &enc_out, &mut cache);
                    let mut beam = BeamSearch::new(params.beam_size, cache);

                    // Seed beam candidates with top branches from initial prompt
                    let n_vocab = self.decoder.n_vocab;
                    let start_idx = logits.len() - n_vocab;
                    let step_logits = &mut logits[start_idx..];

                    if params.suppress_blank {
                        step_logits[220] = f32::NEG_INFINITY;
                        step_logits[eot as usize] = f32::NEG_INFINITY;
                    }
                    if params.no_timestamps {
                        step_logits[(beg as usize)..n_vocab].fill(f32::NEG_INFINITY);
                    }

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

                    let max_steps = if params.max_tokens == 0 {
                        224
                    } else {
                        params.max_tokens
                    };
                    for step in 1..max_steps {
                        if beam.is_all_completed() {
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
                        );
                    }

                    if let Some(best) = beam.best_candidate() {
                        tokens_cur = best.tokens.clone();
                    }
                } else {
                    // GREEDY DECODING
                    let mut logits = self.decoder.decode(&prompt_base, &enc_out, &mut cache);
                    tokens_cur.clear();
                    let mut last_timestamp_token: Option<i32> = None;
                    let max_steps = if params.max_tokens == 0 {
                        224
                    } else {
                        params.max_tokens
                    };
                    let n_vocab = self.decoder.n_vocab;

                    for step in 0..max_steps {
                        let start_idx = logits.len() - n_vocab;
                        let step_logits = &mut logits[start_idx..];

                        // Suppress blank at start
                        if params.suppress_blank && step == 0 {
                            step_logits[220] = f32::NEG_INFINITY;
                            step_logits[eot as usize] = f32::NEG_INFINITY;
                        }

                        if params.no_timestamps {
                            step_logits[(beg as usize)..n_vocab].fill(f32::NEG_INFINITY);
                        } else if let Some(last_ts) = last_timestamp_token {
                            if (last_ts as usize) > (beg as usize) {
                                step_logits[(beg as usize)..(last_ts as usize)]
                                    .fill(f32::NEG_INFINITY);
                            }
                        }

                        // Greedy argmax
                        let mut best_id = 0;
                        let mut best_logit = f32::NEG_INFINITY;
                        for (i, &l) in step_logits.iter().enumerate() {
                            if l > best_logit {
                                best_logit = l;
                                best_id = i as i32;
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

                // Check fallback criteria
                let text = self.tokenizer.decode(&tokens_cur, true);
                let cr = compression_ratio(&text);

                // If compression ratio is acceptable (<= threshold) or fallback disabled, accept
                if cr <= params.compression_ratio_threshold || params.temperature_inc <= 0.0 {
                    break;
                }

                // Trigger temperature fallback
                temperature += params.temperature_inc;
            }

            // Extract segments from tokens_cur
            if params.no_timestamps || tokens_cur.is_empty() {
                let text = self.tokenizer.decode(&tokens_cur, true);
                let t1 = (seek_time + 30.0).min(total_audio_sec);
                if !text.trim().is_empty() {
                    segments.push(Segment::new(seek_time, t1, text, tokens_cur.clone()));
                }
                seek_frame += chunk_frames;
            } else {
                let mut seg_tokens = Vec::new();
                let mut seg_t0 = seek_time;
                let mut advanced_seek = false;

                for &tok in &tokens_cur {
                    if tok >= beg {
                        let rel_sec = (tok - beg) as f64 * 0.02;
                        let abs_sec = seek_time + rel_sec;

                        if seg_tokens.is_empty() {
                            seg_t0 = abs_sec;
                        } else {
                            let text = self.tokenizer.decode(&seg_tokens, true);
                            if !text.trim().is_empty() {
                                segments.push(Segment::new(
                                    seg_t0,
                                    abs_sec,
                                    text,
                                    seg_tokens.clone(),
                                ));
                            }
                            seg_tokens.clear();
                            seg_t0 = abs_sec;

                            let frame_offset = (rel_sec / 0.01).round() as usize;
                            if frame_offset > 0 {
                                seek_frame += frame_offset;
                                advanced_seek = true;
                            }
                        }
                    } else {
                        seg_tokens.push(tok);
                    }
                }

                if !seg_tokens.is_empty() {
                    let text = self.tokenizer.decode(&seg_tokens, true);
                    let t1 = (seek_time + 30.0).min(total_audio_sec);
                    if !text.trim().is_empty() {
                        segments.push(Segment::new(seg_t0, t1, text, seg_tokens));
                    }
                }

                if !advanced_seek {
                    seek_frame += chunk_frames;
                }
            }
        }

        segments
    }
}
