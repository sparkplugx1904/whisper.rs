//! Silero VAD v5/v6 inference for the GGML model format used by whisper.cpp.
//!
//! The model is a small convolutional encoder (STFT basis, four strided 1-D
//! convolutions), a single LSTM layer, and a final 1x1 convolution. Each
//! 512-sample window produces one speech probability and the LSTM state is
//! carried into the next window, so streaming callers can feed successive
//! chunks without resetting between utterances.

use std::collections::HashMap;

use crate::{SpeechSegment, VadParams};

const GGML_FILE_MAGIC: u32 = 0x67676d6c;
const STFT_PAD: usize = 64;

/// Errors produced while loading a Silero GGML model.
#[derive(Debug, thiserror::Error)]
pub enum SileroVadError {
    #[error("invalid VAD model: {0}")]
    InvalidModel(String),
    #[error("unsupported VAD tensor type {0}")]
    UnsupportedTensorType(i32),
}

/// A loaded Silero VAD model plus its recurrent state.
pub struct SileroVad {
    n_window: usize,
    n_context: usize,
    stft: Conv1d,
    encoder: Vec<Conv1d>,
    lstm: Lstm,
    final_conv: Conv1d,
    probs: Vec<f32>,
}

impl std::fmt::Debug for SileroVad {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SileroVad")
            .field("n_window", &self.n_window)
            .field("n_context", &self.n_context)
            .field("encoder_layers", &self.encoder.len())
            .finish()
    }
}

impl SileroVad {
    /// Load a model from the bytes of a `silero-*-ggml.bin` file.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, SileroVadError> {
        let mut reader = Reader {
            data: bytes,
            pos: 0,
        };

        if reader.read_u32()? != GGML_FILE_MAGIC {
            return Err(SileroVadError::InvalidModel("bad magic".into()));
        }

        let type_len = reader.read_len("model type length")?;
        let _model_type = reader.read_bytes(type_len)?;

        let _major = reader.read_i32()?;
        let _minor = reader.read_i32()?;
        let _patch = reader.read_i32()?;

        let n_window = reader.read_len("n_window")?;
        let n_context = reader.read_len("n_context")?;
        if n_window == 0 {
            return Err(SileroVadError::InvalidModel("n_window is zero".into()));
        }

        let n_encoder_layers = reader.read_i32()?;
        if n_encoder_layers != 4 {
            return Err(SileroVadError::InvalidModel(format!(
                "n_encoder_layers is {n_encoder_layers}, expected 4"
            )));
        }

        let mut in_channels = Vec::with_capacity(4);
        let mut out_channels = Vec::with_capacity(4);
        let mut kernels = Vec::with_capacity(4);
        for _ in 0..n_encoder_layers {
            in_channels.push(reader.read_len("encoder input channels")?);
            out_channels.push(reader.read_len("encoder output channels")?);
            kernels.push(reader.read_len("encoder kernel size")?);
        }

        let lstm_input_size = reader.read_len("lstm input size")?;
        let lstm_hidden_size = reader.read_len("lstm hidden size")?;
        let final_conv_in = reader.read_len("final conv input channels")?;
        let final_conv_out = reader.read_len("final conv output channels")?;

        // The graph below assumes the Silero v5/v6 shapes: 512-sample windows,
        // a 128-sample STFT hop, and a 128-wide LSTM. Reject other variants
        // instead of silently computing with the wrong stride.
        if n_window != 512 || lstm_input_size != 128 || lstm_hidden_size != 128 {
            return Err(SileroVadError::InvalidModel(format!(
                "unsupported Silero shape: n_window={n_window}, lstm_input={lstm_input_size}, lstm_hidden={lstm_hidden_size}"
            )));
        }

        let mut tensors = HashMap::new();
        // 1 STFT tensor, 4*2 encoder tensors, 4 LSTM tensors, 2 final tensors.
        while tensors.len() < 15 && reader.remaining() >= 12 {
            let n_dims = reader.read_i32()?;
            // A zero-rank tensor is a scalar (the final convolution bias).
            if !(0..=4).contains(&n_dims) {
                return Err(SileroVadError::InvalidModel(format!(
                    "invalid tensor rank {n_dims}"
                )));
            }
            let name_len = reader.read_len("tensor name length")?;
            let ttype = reader.read_i32()?;

            let mut ne = [1usize; 4];
            for slot in ne.iter_mut().take(n_dims as usize) {
                *slot = reader.read_len("tensor dimension")?;
            }

            let name = String::from_utf8_lossy(reader.read_bytes(name_len)?).into_owned();
            let n_elements: usize = ne.iter().take(n_dims as usize).product();
            let (data, is_f16) = match ttype {
                0 => (reader.read_f32_tensor(n_elements)?, false),
                1 => (reader.read_f16_tensor(n_elements)?, true),
                other => return Err(SileroVadError::UnsupportedTensorType(other)),
            };

            tensors.insert(name, Tensor { ne, data, is_f16 });
        }

        let stft = Conv1d::new(
            tensors_take(
                &mut tensors,
                "_model.stft.forward_basis_buffer",
                &[256, 1, 258],
            )?,
            None,
            128,
            0,
        );
        let encoder_specs: [(&str, &str, &[usize]); 4] = [
            (
                "_model.encoder.0.reparam_conv.weight",
                "_model.encoder.0.reparam_conv.bias",
                &[kernels[0], in_channels[0], out_channels[0]],
            ),
            (
                "_model.encoder.1.reparam_conv.weight",
                "_model.encoder.1.reparam_conv.bias",
                &[kernels[1], in_channels[1], out_channels[1]],
            ),
            (
                "_model.encoder.2.reparam_conv.weight",
                "_model.encoder.2.reparam_conv.bias",
                &[kernels[2], in_channels[2], out_channels[2]],
            ),
            (
                "_model.encoder.3.reparam_conv.weight",
                "_model.encoder.3.reparam_conv.bias",
                &[kernels[3], in_channels[3], out_channels[3]],
            ),
        ];

        let mut encoder = Vec::with_capacity(encoder_specs.len());
        for (index, (weight_name, bias_name, shape)) in encoder_specs.into_iter().enumerate() {
            let weight = tensors_take(&mut tensors, weight_name, shape)?;
            let bias = tensors_take(&mut tensors, bias_name, &[out_channels[index]])?;
            let stride = if index == 1 || index == 2 { 2 } else { 1 };
            encoder.push(Conv1d::new(weight, Some(bias), stride, 1));
        }

        let lstm_ih = tensors_take(
            &mut tensors,
            "_model.decoder.rnn.weight_ih",
            &[lstm_hidden_size, 4 * lstm_hidden_size],
        )?;
        let lstm_hh = tensors_take(
            &mut tensors,
            "_model.decoder.rnn.weight_hh",
            &[lstm_hidden_size, 4 * lstm_hidden_size],
        )?;
        let lstm_bias_ih = tensors_take(
            &mut tensors,
            "_model.decoder.rnn.bias_ih",
            &[4 * lstm_hidden_size],
        )?;
        let lstm_bias_hh = tensors_take(
            &mut tensors,
            "_model.decoder.rnn.bias_hh",
            &[4 * lstm_hidden_size],
        )?;

        let final_weight = tensors_take(
            &mut tensors,
            "_model.decoder.decoder.2.weight",
            &[final_conv_in, final_conv_out],
        )?;
        let final_bias = tensors_take(
            &mut tensors,
            "_model.decoder.decoder.2.bias",
            &[final_conv_out],
        )?;

        if lstm_input_size != lstm_hidden_size {
            return Err(SileroVadError::InvalidModel(format!(
                "lstm input size {lstm_input_size} differs from hidden size {lstm_hidden_size}"
            )));
        }

        Ok(Self {
            n_window,
            n_context,
            stft,
            encoder,
            lstm: Lstm::new(
                lstm_hidden_size,
                lstm_ih.data,
                lstm_hh.data,
                lstm_bias_ih.data,
                lstm_bias_hh.data,
            ),
            final_conv: Conv1d::new(final_weight, Some(final_bias), 1, 0),
            probs: Vec::new(),
        })
    }

    pub fn n_window(&self) -> usize {
        self.n_window
    }

    pub fn n_context(&self) -> usize {
        self.n_context
    }

    /// Clear the recurrent state. Call between independent utterances.
    pub fn reset_state(&mut self) {
        self.lstm.reset();
    }

    /// Run the model over the supplied 16 kHz mono samples.
    pub fn detect_probs(&mut self, samples: &[f32]) -> &[f32] {
        self.probs.clear();

        if samples.is_empty() {
            return &self.probs;
        }

        let n_chunks = samples.len().div_ceil(self.n_window);
        self.probs.reserve(n_chunks);
        let mut window = vec![0.0f32; self.n_window];

        for chunk in 0..n_chunks {
            let start = chunk * self.n_window;
            let end = (start + self.n_window).min(samples.len());
            window.fill(0.0);
            window[..end - start].copy_from_slice(&samples[start..end]);
            let prob = self.forward_frame(&window);
            self.probs.push(prob);
        }

        &self.probs
    }

    pub fn probs(&self) -> &[f32] {
        &self.probs
    }

    fn forward_frame(&mut self, frame: &[f32]) -> f32 {
        let padded = reflect_pad(frame, STFT_PAD);
        let (stft, stft_len) = self.stft.forward(&padded, 1);
        debug_assert_eq!(stft_len, 4);

        // Output channels are 129 real values followed by 129 imaginary ones.
        let bins = self.stft.out_channels / 2;
        let mut magnitude = vec![0.0f32; stft_len * bins];
        for t in 0..stft_len {
            for bin in 0..bins {
                let real = stft[t * self.stft.out_channels + bin];
                let imag = stft[t * self.stft.out_channels + bins + bin];
                magnitude[t * bins + bin] = (real * real + imag * imag).sqrt();
            }
        }

        let mut current = magnitude;
        let mut current_len = stft_len;
        let mut current_channels = bins;

        for conv in &self.encoder {
            let (next, next_len) = conv.forward(&current, current_channels);
            current = next;
            current_len = next_len;
            current_channels = conv.out_channels;

            for value in &mut current {
                *value = value.max(0.0);
            }
        }

        debug_assert_eq!(current_len, 1);
        let hidden = self.lstm.step(&current);
        let mut activated: Vec<f32> = hidden.iter().map(|value| value.max(0.0)).collect();

        // The LSTM output is [length = hidden, channels = 1] for the final
        // 1-D convolution, not a single frame with `hidden` channels.
        let (out, _) = self.final_conv.forward(&activated, 1);
        activated.clear();
        let logit = out[0];
        1.0 / (1.0 + (-logit).exp())
    }
}

/// Convert model probabilities into speech segments, matching
/// `whisper_vad_segments_from_probs()` in whisper.cpp.
pub fn segments_from_probs(
    probs: &[f32],
    params: &VadParams,
    n_window: usize,
) -> Vec<SpeechSegment> {
    if probs.is_empty() {
        return Vec::new();
    }

    let sample_rate = params.sample_rate.max(1);
    let min_speech_samples = sample_rate * params.min_speech_duration_ms / 1000;
    let min_silence_samples = sample_rate * params.min_silence_duration_ms / 1000;
    let speech_pad_samples = sample_rate * params.speech_pad_ms / 1000;
    let audio_length_samples = probs.len() * n_window;
    let min_silence_samples_at_max_speech = sample_rate * 98 / 1000;

    let max_speech_samples = if params.max_speech_duration_s > 100_000.0 {
        i32::MAX as usize / 2
    } else {
        let temp = (sample_rate as f64 * params.max_speech_duration_s as f64)
            - n_window as f64
            - 2.0 * speech_pad_samples as f64;
        if temp <= 0.0 {
            i32::MAX as usize / 2
        } else {
            temp.min(i32::MAX as f64 / 2.0) as usize
        }
    };

    let threshold = params.threshold;
    let mut neg_threshold = threshold - 0.15;
    if neg_threshold < 0.01 {
        neg_threshold = 0.01;
    }

    let mut speeches: Vec<(usize, usize)> = Vec::new();
    let mut is_speech_segment = false;
    let mut temp_end = 0usize;
    let mut prev_end = 0usize;
    let mut next_start = 0usize;
    let mut curr_speech_start = 0usize;
    let mut has_curr_speech = false;

    for (i, &prob) in probs.iter().enumerate() {
        let curr_sample = n_window * i;

        if prob >= threshold && temp_end != 0 {
            temp_end = 0;
            if next_start < prev_end {
                next_start = curr_sample;
            }
        }

        if prob >= threshold && !is_speech_segment {
            is_speech_segment = true;
            curr_speech_start = curr_sample;
            has_curr_speech = true;
            continue;
        }

        if is_speech_segment && curr_sample - curr_speech_start > max_speech_samples {
            if prev_end != 0 {
                speeches.push((curr_speech_start, prev_end));
                has_curr_speech = true;

                if next_start < prev_end {
                    is_speech_segment = false;
                    has_curr_speech = false;
                } else {
                    curr_speech_start = next_start;
                }
                prev_end = 0;
                next_start = 0;
                temp_end = 0;
            } else {
                speeches.push((curr_speech_start, curr_sample));
                prev_end = 0;
                next_start = 0;
                temp_end = 0;
                is_speech_segment = false;
                has_curr_speech = false;
                continue;
            }
        }

        if prob < neg_threshold && is_speech_segment {
            if temp_end == 0 {
                temp_end = curr_sample;
            }

            if curr_sample - temp_end > min_silence_samples_at_max_speech {
                prev_end = temp_end;
            }

            if curr_sample - temp_end < min_silence_samples {
                continue;
            }

            if temp_end - curr_speech_start > min_speech_samples {
                speeches.push((curr_speech_start, temp_end));
            }

            prev_end = 0;
            next_start = 0;
            temp_end = 0;
            is_speech_segment = false;
            has_curr_speech = false;
            continue;
        }
    }

    if has_curr_speech && audio_length_samples - curr_speech_start > min_speech_samples {
        speeches.push((curr_speech_start, audio_length_samples));
    }

    // Merge adjacent segments separated by less than 200 ms.
    let max_merge_gap_samples = sample_rate * 200 / 1000;
    let mut index = 0;
    while index + 1 < speeches.len() {
        if speeches[index + 1].0.saturating_sub(speeches[index].1) < max_merge_gap_samples {
            speeches[index].1 = speeches[index + 1].1;
            speeches.remove(index + 1);
        } else {
            index += 1;
        }
    }

    speeches.retain(|(start, end)| end - start >= min_speech_samples);

    let mut segments = Vec::with_capacity(speeches.len());
    for i in 0..speeches.len() {
        if i == 0 {
            speeches[i].0 = speeches[i].0.saturating_sub(speech_pad_samples);
        }

        if i + 1 < speeches.len() {
            let silence_duration = speeches[i + 1].0 - speeches[i].1;
            if silence_duration < 2 * speech_pad_samples {
                speeches[i].1 += silence_duration / 2;
                speeches[i + 1].0 = speeches[i + 1].0.saturating_sub(silence_duration / 2);
            } else {
                speeches[i].1 = (speeches[i].1 + speech_pad_samples).min(audio_length_samples);
                speeches[i + 1].0 = speeches[i + 1].0.saturating_sub(speech_pad_samples);
            }
        } else {
            speeches[i].1 = (speeches[i].1 + speech_pad_samples).min(audio_length_samples);
        }

        segments.push(SpeechSegment::new(
            speeches[i].0 as f64 / sample_rate as f64,
            speeches[i].1 as f64 / sample_rate as f64,
        ));
    }

    segments
}

#[derive(Debug)]
struct Tensor {
    ne: [usize; 4],
    data: Vec<f32>,
    is_f16: bool,
}

fn tensors_take(
    tensors: &mut HashMap<String, Tensor>,
    name: &str,
    expected: &[usize],
) -> Result<Tensor, SileroVadError> {
    let tensor = tensors
        .remove(name)
        .ok_or_else(|| SileroVadError::InvalidModel(format!("missing tensor {name}")))?;

    let matches = tensor.ne[..expected.len()] == *expected
        && tensor.ne[expected.len()..].iter().all(|&dim| dim == 1);
    if !matches {
        return Err(SileroVadError::InvalidModel(format!(
            "tensor {name} has shape {:?}, expected {expected:?}",
            tensor.ne
        )));
    }

    Ok(tensor)
}

#[derive(Debug, Clone)]
struct Conv1d {
    kernel: usize,
    in_channels: usize,
    out_channels: usize,
    stride: usize,
    padding: usize,
    weight: Vec<f32>,
    bias: Vec<f32>,
    /// `ggml_conv_1d` materializes an F16 im2col buffer when the weight is
    /// F16, so the input values are rounded before the multiply.
    round_input_f16: bool,
}

impl Conv1d {
    fn new(weight: Tensor, bias: Option<Tensor>, stride: usize, padding: usize) -> Self {
        let kernel = weight.ne[0];
        let in_channels = weight.ne[1];
        let out_channels = weight.ne[2];
        Self {
            kernel,
            in_channels,
            out_channels,
            stride,
            padding,
            weight: weight.data,
            bias: bias.map(|t| t.data).unwrap_or_default(),
            round_input_f16: weight.is_f16,
        }
    }

    /// Cross-correlation over `[length, in_channels]`, returning
    /// `[out_length, out_channels]`.
    fn forward(&self, input: &[f32], in_channels: usize) -> (Vec<f32>, usize) {
        debug_assert_eq!(in_channels, self.in_channels);
        let length = input.len() / in_channels;
        let input_storage: Vec<f32> = if self.round_input_f16 {
            input
                .iter()
                .map(|&value| half::f16::from_f32(value).to_f32())
                .collect()
        } else {
            input.to_vec()
        };
        let input = input_storage.as_slice();
        let padded = length + 2 * self.padding;
        let out_length = if padded >= self.kernel {
            (padded - self.kernel) / self.stride + 1
        } else {
            0
        };

        let mut output = vec![0.0f32; out_length * self.out_channels];
        for t in 0..out_length {
            for oc in 0..self.out_channels {
                let mut sum = self.bias.get(oc).copied().unwrap_or(0.0);
                for k in 0..self.kernel {
                    let pos = t * self.stride + k;
                    if pos < self.padding {
                        continue;
                    }
                    let src = pos - self.padding;
                    if src >= length {
                        continue;
                    }
                    let input_base = src * in_channels;
                    let weight_base = k + oc * self.kernel * self.in_channels;
                    for ic in 0..self.in_channels {
                        sum += input[input_base + ic] * self.weight[weight_base + ic * self.kernel];
                    }
                }
                output[t * self.out_channels + oc] = sum;
            }
        }

        (output, out_length)
    }
}

struct Lstm {
    hidden_size: usize,
    weight_ih: Vec<f32>,
    weight_hh: Vec<f32>,
    bias_ih: Vec<f32>,
    bias_hh: Vec<f32>,
    h: Vec<f32>,
    c: Vec<f32>,
}

impl Lstm {
    fn new(
        hidden_size: usize,
        weight_ih: Vec<f32>,
        weight_hh: Vec<f32>,
        bias_ih: Vec<f32>,
        bias_hh: Vec<f32>,
    ) -> Self {
        let mut lstm = Self {
            hidden_size,
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
            h: Vec::new(),
            c: Vec::new(),
        };
        lstm.reset();
        lstm
    }

    fn reset(&mut self) {
        self.h = vec![0.0; self.hidden_size];
        self.c = vec![0.0; self.hidden_size];
    }

    fn step(&mut self, input: &[f32]) -> Vec<f32> {
        let hidden = self.hidden_size;
        let gates = 4 * hidden;
        let mut pre = vec![0.0f32; gates];

        for (gate, slot) in pre.iter_mut().enumerate() {
            let mut sum = self.bias_ih[gate] + self.bias_hh[gate];
            let row = gate * hidden;
            for (k, &value) in input.iter().enumerate().take(hidden) {
                sum += value * self.weight_ih[row + k];
                sum += self.h[k] * self.weight_hh[row + k];
            }
            *slot = sum;
        }

        for j in 0..hidden {
            let input_gate = sigmoid(pre[j]);
            let forget_gate = sigmoid(pre[hidden + j]);
            let cell_gate = pre[2 * hidden + j].tanh();
            let output_gate = sigmoid(pre[3 * hidden + j]);

            self.c[j] = forget_gate * self.c[j] + input_gate * cell_gate;
            self.h[j] = output_gate * self.c[j].tanh();
        }

        self.h.clone()
    }
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

fn reflect_pad(input: &[f32], pad: usize) -> Vec<f32> {
    let n = input.len();
    let mut output = vec![0.0f32; n + 2 * pad];
    output[pad..pad + n].copy_from_slice(input);

    for i in 1..=pad {
        output[pad - i] = input[i];
        output[pad + n - 1 + i] = input[n - 1 - i];
    }

    output
}

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    fn read_bytes(&mut self, len: usize) -> Result<&'a [u8], SileroVadError> {
        if len > self.remaining() {
            return Err(SileroVadError::InvalidModel(
                "unexpected end of model file".into(),
            ));
        }
        let bytes = &self.data[self.pos..self.pos + len];
        self.pos += len;
        Ok(bytes)
    }

    fn read_i32(&mut self) -> Result<i32, SileroVadError> {
        let bytes = self.read_bytes(4)?;
        Ok(i32::from_le_bytes(bytes.try_into().unwrap()))
    }

    fn read_u32(&mut self) -> Result<u32, SileroVadError> {
        let bytes = self.read_bytes(4)?;
        Ok(u32::from_le_bytes(bytes.try_into().unwrap()))
    }

    fn read_len(&mut self, what: &str) -> Result<usize, SileroVadError> {
        let value = self.read_i32()?;
        if value < 0 {
            return Err(SileroVadError::InvalidModel(format!(
                "negative {what}: {value}"
            )));
        }
        Ok(value as usize)
    }

    fn read_f32_tensor(&mut self, n: usize) -> Result<Vec<f32>, SileroVadError> {
        let bytes = self.read_bytes(n * 4)?;
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect())
    }

    fn read_f16_tensor(&mut self, n: usize) -> Result<Vec<f32>, SileroVadError> {
        let bytes = self.read_bytes(n * 2)?;
        Ok(bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|chunk| half::f16::from_bits(u16::from_le_bytes(*chunk)).to_f32())
            .collect())
    }
}
