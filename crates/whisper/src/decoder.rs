use whisper_model::{ModelError, WhisperModel};
use whisper_tensor::ops::{add_inplace, gelu_inplace, layer_norm, linear, multi_head_attention};

#[derive(Clone)]
pub struct DecoderBlockWeights {
    pub attn_ln_w: Vec<f32>,
    pub attn_ln_b: Vec<f32>,
    pub attn_q_w: Vec<f32>,
    pub attn_q_b: Vec<f32>,
    pub attn_k_w: Vec<f32>,
    pub attn_v_w: Vec<f32>,
    pub attn_v_b: Vec<f32>,
    pub attn_out_w: Vec<f32>,
    pub attn_out_b: Vec<f32>,

    pub cross_attn_ln_w: Vec<f32>,
    pub cross_attn_ln_b: Vec<f32>,
    pub cross_attn_q_w: Vec<f32>,
    pub cross_attn_q_b: Vec<f32>,
    pub cross_attn_k_w: Vec<f32>,
    pub cross_attn_v_w: Vec<f32>,
    pub cross_attn_v_b: Vec<f32>,
    pub cross_attn_out_w: Vec<f32>,
    pub cross_attn_out_b: Vec<f32>,

    pub mlp_ln_w: Vec<f32>,
    pub mlp_ln_b: Vec<f32>,
    pub mlp0_w: Vec<f32>,
    pub mlp0_b: Vec<f32>,
    pub mlp2_w: Vec<f32>,
    pub mlp2_b: Vec<f32>,
}

#[derive(Clone)]
pub struct DecoderWeights {
    pub token_embedding: Vec<f32>,
    pub positional_embedding: Vec<f32>,
    pub blocks: Vec<DecoderBlockWeights>,
    pub ln_post_w: Vec<f32>,
    pub ln_post_b: Vec<f32>,
}

impl DecoderWeights {
    pub fn load(model: &WhisperModel) -> Result<Self, ModelError> {
        let token_embedding = model.get_tensor_f32("decoder.token_embedding.weight")?;
        let positional_embedding = model.get_tensor_f32("decoder.positional_embedding")?;

        let n_layers = model.hparams.n_text_layer as usize;
        let mut blocks = Vec::with_capacity(n_layers);

        for l in 0..n_layers {
            let prefix = format!("decoder.blocks.{}", l);
            let block = DecoderBlockWeights {
                attn_ln_w: model.get_tensor_f32(&format!("{}.attn_ln.weight", prefix))?,
                attn_ln_b: model.get_tensor_f32(&format!("{}.attn_ln.bias", prefix))?,
                attn_q_w: model.get_tensor_f32(&format!("{}.attn.query.weight", prefix))?,
                attn_q_b: model.get_tensor_f32(&format!("{}.attn.query.bias", prefix))?,
                attn_k_w: model.get_tensor_f32(&format!("{}.attn.key.weight", prefix))?,
                attn_v_w: model.get_tensor_f32(&format!("{}.attn.value.weight", prefix))?,
                attn_v_b: model.get_tensor_f32(&format!("{}.attn.value.bias", prefix))?,
                attn_out_w: model.get_tensor_f32(&format!("{}.attn.out.weight", prefix))?,
                attn_out_b: model.get_tensor_f32(&format!("{}.attn.out.bias", prefix))?,

                cross_attn_ln_w: model
                    .get_tensor_f32(&format!("{}.cross_attn_ln.weight", prefix))?,
                cross_attn_ln_b: model.get_tensor_f32(&format!("{}.cross_attn_ln.bias", prefix))?,
                cross_attn_q_w: model
                    .get_tensor_f32(&format!("{}.cross_attn.query.weight", prefix))?,
                cross_attn_q_b: model
                    .get_tensor_f32(&format!("{}.cross_attn.query.bias", prefix))?,
                cross_attn_k_w: model
                    .get_tensor_f32(&format!("{}.cross_attn.key.weight", prefix))?,
                cross_attn_v_w: model
                    .get_tensor_f32(&format!("{}.cross_attn.value.weight", prefix))?,
                cross_attn_v_b: model
                    .get_tensor_f32(&format!("{}.cross_attn.value.bias", prefix))?,
                cross_attn_out_w: model
                    .get_tensor_f32(&format!("{}.cross_attn.out.weight", prefix))?,
                cross_attn_out_b: model
                    .get_tensor_f32(&format!("{}.cross_attn.out.bias", prefix))?,

                mlp_ln_w: model.get_tensor_f32(&format!("{}.mlp_ln.weight", prefix))?,
                mlp_ln_b: model.get_tensor_f32(&format!("{}.mlp_ln.bias", prefix))?,
                mlp0_w: model.get_tensor_f32(&format!("{}.mlp.0.weight", prefix))?,
                mlp0_b: model.get_tensor_f32(&format!("{}.mlp.0.bias", prefix))?,
                mlp2_w: model.get_tensor_f32(&format!("{}.mlp.2.weight", prefix))?,
                mlp2_b: model.get_tensor_f32(&format!("{}.mlp.2.bias", prefix))?,
            };
            blocks.push(block);
        }

        let ln_post_w = model.get_tensor_f32("decoder.ln.weight")?;
        let ln_post_b = model.get_tensor_f32("decoder.ln.bias")?;

        Ok(Self {
            token_embedding,
            positional_embedding,
            blocks,
            ln_post_w,
            ln_post_b,
        })
    }
}

#[derive(Clone)]
pub struct DecoderKvCache {
    /// Layer index -> self-attention key cache: [T_past, d_model]
    pub k_self: Vec<Vec<f32>>,
    /// Layer index -> self-attention value cache: [T_past, d_model]
    pub v_self: Vec<Vec<f32>>,
    /// Layer index -> precomputed cross-attention keys: [1500, d_model]
    pub k_cross: Vec<Vec<f32>>,
    /// Layer index -> precomputed cross-attention values: [1500, d_model]
    pub v_cross: Vec<Vec<f32>>,
    pub t_past: usize,
}

impl DecoderKvCache {
    pub fn new(n_layers: usize) -> Self {
        Self {
            k_self: vec![Vec::new(); n_layers],
            v_self: vec![Vec::new(); n_layers],
            k_cross: vec![Vec::new(); n_layers],
            v_cross: vec![Vec::new(); n_layers],
            t_past: 0,
        }
    }

    pub fn clear(&mut self) {
        for k in &mut self.k_self {
            k.clear();
        }
        for v in &mut self.v_self {
            v.clear();
        }
        self.t_past = 0;
    }
}

pub struct WhisperDecoder {
    pub weights: DecoderWeights,
    pub n_vocab: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub max_ctx: usize,
}

impl WhisperDecoder {
    pub fn new(model: &WhisperModel) -> Result<Self, ModelError> {
        let weights = DecoderWeights::load(model)?;
        Ok(Self {
            weights,
            n_vocab: model.hparams.n_vocab as usize,
            d_model: model.hparams.n_text_state as usize,
            n_heads: model.hparams.n_text_head as usize,
            n_layers: model.hparams.n_text_layer as usize,
            max_ctx: model.hparams.n_text_ctx as usize,
        })
    }

    /// Precompute and cache cross-attention keys and values from encoder output.
    pub fn precompute_cross_attn(&self, encoder_output: &[f32], cache: &mut DecoderKvCache) {
        let n_audio_tokens = encoder_output.len() / self.d_model;
        for (l, block) in self.weights.blocks.iter().enumerate() {
            let mut k_cross = vec![0.0f32; n_audio_tokens * self.d_model];
            let mut v_cross = vec![0.0f32; n_audio_tokens * self.d_model];

            linear(
                encoder_output,
                &block.cross_attn_k_w,
                None, // no key bias
                self.d_model,
                self.d_model,
                n_audio_tokens,
                &mut k_cross,
            );

            linear(
                encoder_output,
                &block.cross_attn_v_w,
                Some(&block.cross_attn_v_b),
                self.d_model,
                self.d_model,
                n_audio_tokens,
                &mut v_cross,
            );

            cache.k_cross[l] = k_cross;
            cache.v_cross[l] = v_cross;
        }
    }

    /// Decode a sequence of tokens starting at past position `cache.t_past`.
    /// Returns logits for the tokens: [n_tokens, n_vocab].
    pub fn decode(
        &self,
        tokens: &[i32],
        encoder_output: &[f32],
        cache: &mut DecoderKvCache,
    ) -> Vec<f32> {
        let n_tokens = tokens.len();
        if n_tokens == 0 {
            return Vec::new();
        }

        // If cross keys haven't been precomputed, do so now
        if cache.k_cross[0].is_empty() {
            self.precompute_cross_attn(encoder_output, cache);
        }

        let n_audio_tokens = encoder_output.len() / self.d_model;
        let d_mlp = self.d_model * 4;

        // 1. Token embedding + Positional embedding
        let mut x = vec![0.0f32; n_tokens * self.d_model];
        for (i, &tok) in tokens.iter().enumerate() {
            let pos = cache.t_past + i;
            let tok_idx = tok as usize;

            let tok_slice =
                &self.weights.token_embedding[tok_idx * self.d_model..(tok_idx + 1) * self.d_model];
            let pos_slice =
                &self.weights.positional_embedding[pos * self.d_model..(pos + 1) * self.d_model];

            let x_slice = &mut x[i * self.d_model..(i + 1) * self.d_model];
            for d in 0..self.d_model {
                x_slice[d] = tok_slice[d] + pos_slice[d];
            }
        }

        // Workspace buffers
        let mut norm_buf = vec![0.0f32; n_tokens * self.d_model];
        let mut q_buf = vec![0.0f32; n_tokens * self.d_model];
        let mut k_buf = vec![0.0f32; n_tokens * self.d_model];
        let mut v_buf = vec![0.0f32; n_tokens * self.d_model];
        let mut attn_out = vec![0.0f32; n_tokens * self.d_model];
        let mut proj_out = vec![0.0f32; n_tokens * self.d_model];
        let mut mlp0_out = vec![0.0f32; n_tokens * d_mlp];

        let total_tokens = cache.t_past + n_tokens;

        // 2. Transformer blocks
        for (l, block) in self.weights.blocks.iter().enumerate() {
            // (a) Pre-LN for self-attention
            layer_norm(
                &x,
                &block.attn_ln_w,
                &block.attn_ln_b,
                1e-5,
                n_tokens,
                self.d_model,
                &mut norm_buf,
            );

            // (b) Self-attention Q, K, V
            linear(
                &norm_buf,
                &block.attn_q_w,
                Some(&block.attn_q_b),
                self.d_model,
                self.d_model,
                n_tokens,
                &mut q_buf,
            );
            linear(
                &norm_buf,
                &block.attn_k_w,
                None, // no key bias
                self.d_model,
                self.d_model,
                n_tokens,
                &mut k_buf,
            );
            linear(
                &norm_buf,
                &block.attn_v_w,
                Some(&block.attn_v_b),
                self.d_model,
                self.d_model,
                n_tokens,
                &mut v_buf,
            );

            // Append new K and V to cache
            cache.k_self[l].extend_from_slice(&k_buf);
            cache.v_self[l].extend_from_slice(&v_buf);

            // (c) Causal Multi-Head Self-Attention over KV cache
            // Q has length n_tokens, KV has length total_tokens
            multi_head_attention_with_cache(
                &q_buf,
                &cache.k_self[l],
                &cache.v_self[l],
                n_tokens,
                total_tokens,
                cache.t_past,
                self.n_heads,
                self.d_model,
                &mut attn_out,
            );

            // (d) Self-attention output projection + residual
            linear(
                &attn_out,
                &block.attn_out_w,
                Some(&block.attn_out_b),
                self.d_model,
                self.d_model,
                n_tokens,
                &mut proj_out,
            );
            add_inplace(&mut x, &proj_out);

            // (e) Pre-LN for cross-attention
            layer_norm(
                &x,
                &block.cross_attn_ln_w,
                &block.cross_attn_ln_b,
                1e-5,
                n_tokens,
                self.d_model,
                &mut norm_buf,
            );

            // (f) Cross-attention Query
            linear(
                &norm_buf,
                &block.cross_attn_q_w,
                Some(&block.cross_attn_q_b),
                self.d_model,
                self.d_model,
                n_tokens,
                &mut q_buf,
            );

            // (g) Cross Multi-Head Attention over precomputed K_cross and V_cross
            multi_head_attention(
                &q_buf,
                &cache.k_cross[l],
                &cache.v_cross[l],
                n_tokens,
                n_audio_tokens,
                self.n_heads,
                self.d_model,
                false, // non-causal
                &mut attn_out,
            );

            // (h) Cross-attention output projection + residual
            linear(
                &attn_out,
                &block.cross_attn_out_w,
                Some(&block.cross_attn_out_b),
                self.d_model,
                self.d_model,
                n_tokens,
                &mut proj_out,
            );
            add_inplace(&mut x, &proj_out);

            // (i) Pre-LN for MLP
            layer_norm(
                &x,
                &block.mlp_ln_w,
                &block.mlp_ln_b,
                1e-5,
                n_tokens,
                self.d_model,
                &mut norm_buf,
            );

            // (j) MLP
            linear(
                &norm_buf,
                &block.mlp0_w,
                Some(&block.mlp0_b),
                self.d_model,
                d_mlp,
                n_tokens,
                &mut mlp0_out,
            );
            gelu_inplace(&mut mlp0_out);

            linear(
                &mlp0_out,
                &block.mlp2_w,
                Some(&block.mlp2_b),
                d_mlp,
                self.d_model,
                n_tokens,
                &mut proj_out,
            );
            add_inplace(&mut x, &proj_out);
        }

        // Update past tokens count
        cache.t_past += n_tokens;

        // 3. Final LayerNorm
        let mut final_norm = vec![0.0f32; n_tokens * self.d_model];
        layer_norm(
            &x,
            &self.weights.ln_post_w,
            &self.weights.ln_post_b,
            1e-5,
            n_tokens,
            self.d_model,
            &mut final_norm,
        );

        // 4. Logits: multiply final token states by token embedding matrix
        let mut logits = vec![0.0f32; n_tokens * self.n_vocab];
        linear(
            &final_norm,
            &self.weights.token_embedding,
            None,
            self.d_model,
            self.n_vocab,
            n_tokens,
            &mut logits,
        );

        logits
    }
}
/// Specialized causal multi-head self-attention with KV cache.
/// `t_past`: number of tokens already in cache before this batch.
/// Each query token `i` in `0..n_tokens` has position `t_past + i`,
/// and can attend to keys at `0..= t_past + i`.
#[allow(clippy::too_many_arguments)]
fn multi_head_attention_with_cache(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_tokens: usize,
    total_tokens: usize,
    t_past: usize,
    n_heads: usize,
    d_model: usize,
    output: &mut [f32],
) {
    let d_head = d_model / n_heads;
    assert_eq!(q.len(), n_tokens * d_model);
    assert_eq!(k.len(), total_tokens * d_model);
    assert_eq!(v.len(), total_tokens * d_model);
    assert_eq!(output.len(), n_tokens * d_model);

    if n_tokens == 1 && d_head == 64 {
        whisper_tensor::ops::single_token_mha_d64(q, k, v, t_past + 1, n_heads, d_model, output);
        return;
    }

    let scale = 1.0f32 / (d_head as f32).sqrt();
    let mut scores = vec![0.0f32; total_tokens];
    for h in 0..n_heads {
        let head_offset = h * d_head;

        for i in 0..n_tokens {
            let q_vec = &q[i * d_model + head_offset..i * d_model + head_offset + d_head];
            let max_j = t_past + i + 1; // causal mask: can attend up to current position

            // 1. Scores
            for j in 0..max_j {
                let k_vec = &k[j * d_model + head_offset..j * d_model + head_offset + d_head];
                scores[j] = whisper_tensor::simd::dot_f32(q_vec, k_vec) * scale;
            }

            // 2. Softmax
            whisper_tensor::ops::softmax_inplace(&mut scores[0..max_j]);

            // 3. Weighted values
            let out_vec =
                &mut output[i * d_model + head_offset..i * d_model + head_offset + d_head];
            out_vec.fill(0.0);

            for j in 0..max_j {
                let weight = scores[j];
                if weight < 1e-7 {
                    continue;
                }
                let v_vec = &v[j * d_model + head_offset..j * d_model + head_offset + d_head];
                for d in 0..d_head {
                    out_vec[d] += weight * v_vec[d];
                }
            }
        }
    }
}
