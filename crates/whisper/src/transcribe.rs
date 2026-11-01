use crate::decoder::{DecoderKvCache, WhisperDecoder};
use crate::encoder::WhisperEncoder;
use crate::params::WhisperParams;
use crate::segment::Segment;
use whisper_audio::mel::MelSpectrogram;
use whisper_model::vocab::LANGUAGES;
use whisper_model::WhisperModel;
use whisper_tokenizer::Tokenizer;

pub struct WhisperContext {
    pub model: WhisperModel,
    pub encoder: WhisperEncoder,
    pub decoder: WhisperDecoder,
    pub tokenizer: Tokenizer,
}

impl WhisperContext {
    pub fn new(model: WhisperModel) -> Result<Self, whisper_model::ModelError> {
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
        let is_multilingual = self.model.hparams.is_multilingual();
        let lang = if params.language == "auto" {
            let (detected, _) = self.detect_language(mel);
            detected
        } else {
            params.language.clone()
        };

        let lang_idx = whisper_model::vocab::lang_id(&lang).unwrap_or(0);

        let sot = self.tokenizer.sot();
        let eot = self.tokenizer.eot();
        let beg = self.tokenizer.beg();
        let not_token = self.tokenizer.not();

        let mut prompt_base = vec![sot];
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

            let mut cache = DecoderKvCache::new(self.decoder.n_layers);

            // Feed prompt
            let mut logits = self.decoder.decode(&prompt_base, &enc_out, &mut cache);

            let mut tokens_cur = Vec::new();
            let mut last_timestamp_token: Option<i32> = None;
            let mut max_steps = params.max_tokens;
            if max_steps == 0 {
                max_steps = 224;
            }

            for step in 0..max_steps {
                assert_eq!(logits.len(), self.decoder.n_vocab);

                // Filter / suppress
                if params.suppress_blank && step == 0 {
                    // Suppress space token (220) and EOT at start
                    logits[220] = f32::NEG_INFINITY;
                    logits[eot as usize] = f32::NEG_INFINITY;
                }

                if params.no_timestamps {
                    // Suppress all timestamp tokens
                    logits[(beg as usize)..self.decoder.n_vocab].fill(f32::NEG_INFINITY);
                } else if let Some(last_ts) = last_timestamp_token {
                    // Timestamps must not go backwards
                    if (last_ts as usize) > (beg as usize) {
                        logits[(beg as usize)..(last_ts as usize)].fill(f32::NEG_INFINITY);
                    }
                }

                // Greedy argmax
                let mut best_id = 0;
                let mut best_logit = f32::NEG_INFINITY;
                for (i, &l) in logits.iter().enumerate() {
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

            // Extract segments from tokens_cur
            if params.no_timestamps || tokens_cur.is_empty() {
                let text = self.tokenizer.decode(&tokens_cur, true);
                let t1 = (seek_time + 30.0).min(total_audio_sec);
                if !text.trim().is_empty() {
                    segments.push(Segment::new(seek_time, t1, text, tokens_cur.clone()));
                }
                seek_frame += chunk_frames;
            } else {
                // Parse timestamps
                let mut seg_tokens = Vec::new();
                let mut seg_t0 = seek_time;
                let mut advanced_seek = false;

                for &tok in &tokens_cur {
                    if tok >= beg {
                        let rel_sec = (tok - beg) as f64 * 0.02;
                        let abs_sec = seek_time + rel_sec;

                        if seg_tokens.is_empty() {
                            // Beginning timestamp
                            seg_t0 = abs_sec;
                        } else {
                            // Ending timestamp
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

                            // Update seek frame to last emitted timestamp
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

                // Any leftover text without closing timestamp
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
