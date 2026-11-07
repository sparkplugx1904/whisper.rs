//! Whisper text decoder: token embedding lookup, causal self-attention backed
//! by a multi-slot KV arena, cross-attention over the encoder output, and the
//! final vocabulary projection.

use whisper_model::{ModelError, WhisperModel};
use whisper_tensor::ops::{add_inplace, gelu_inplace, layer_norm, multi_head_attention};
use whisper_tensor::LinearWeight;

/// Linear weights of one decoder block.
///
/// The projection matrices are kept as [`LinearWeight`] so a quantized model
/// stays compressed in RAM and the single-token GEMV streams the block bytes
/// instead of an expanded float copy. Biases and LayerNorm parameters are
/// already tiny and stay dense.
#[derive(Clone)]
pub struct DecoderBlockWeights {
    pub attn_ln_w: Vec<f32>,
    pub attn_ln_b: Vec<f32>,
    pub attn_q_w: LinearWeight,
    pub attn_q_b: Vec<f32>,
    pub attn_k_w: LinearWeight,
    pub attn_v_w: LinearWeight,
    pub attn_v_b: Vec<f32>,
    pub attn_out_w: LinearWeight,
    pub attn_out_b: Vec<f32>,

    pub cross_attn_ln_w: Vec<f32>,
    pub cross_attn_ln_b: Vec<f32>,
    pub cross_attn_q_w: LinearWeight,
    pub cross_attn_q_b: Vec<f32>,
    pub cross_attn_k_w: LinearWeight,
    pub cross_attn_v_w: LinearWeight,
    pub cross_attn_v_b: Vec<f32>,
    pub cross_attn_out_w: LinearWeight,
    pub cross_attn_out_b: Vec<f32>,

    pub mlp_ln_w: Vec<f32>,
    pub mlp_ln_b: Vec<f32>,
    pub mlp0_w: LinearWeight,
    pub mlp0_b: Vec<f32>,
    pub mlp2_w: LinearWeight,
    pub mlp2_b: Vec<f32>,
}

/// All decoder weights.
///
/// `token_embedding` is a [`LinearWeight`] because the same matrix serves two
/// roles: row lookup while embedding the input tokens and the output
/// projection over the vocabulary. Keeping it in its original GGML format lets
/// the lookup dequantize a single row and the projection stream the matrix.
#[derive(Clone)]
pub struct DecoderWeights {
    pub token_embedding: LinearWeight,
    pub positional_embedding: Vec<f32>,
    pub blocks: Vec<DecoderBlockWeights>,
    pub ln_post_w: Vec<f32>,
    pub ln_post_b: Vec<f32>,
}

impl DecoderWeights {
    pub fn load(model: &WhisperModel) -> Result<Self, ModelError> {
        let token_embedding = model.get_linear_weight("decoder.token_embedding.weight")?;
        let positional_embedding = model.get_tensor_f32("decoder.positional_embedding")?;

        let n_layers = model.hparams.n_text_layer as usize;
        let mut blocks = Vec::with_capacity(n_layers);

        for l in 0..n_layers {
            let prefix = format!("decoder.blocks.{}", l);
            let block = DecoderBlockWeights {
                attn_ln_w: model.get_tensor_f32(&format!("{}.attn_ln.weight", prefix))?,
                attn_ln_b: model.get_tensor_f32(&format!("{}.attn_ln.bias", prefix))?,
                attn_q_w: model.get_linear_weight(&format!("{}.attn.query.weight", prefix))?,
                attn_q_b: model.get_tensor_f32(&format!("{}.attn.query.bias", prefix))?,
                attn_k_w: model.get_linear_weight(&format!("{}.attn.key.weight", prefix))?,
                attn_v_w: model.get_linear_weight(&format!("{}.attn.value.weight", prefix))?,
                attn_v_b: model.get_tensor_f32(&format!("{}.attn.value.bias", prefix))?,
                attn_out_w: model.get_linear_weight(&format!("{}.attn.out.weight", prefix))?,
                attn_out_b: model.get_tensor_f32(&format!("{}.attn.out.bias", prefix))?,

                cross_attn_ln_w: model
                    .get_tensor_f32(&format!("{}.cross_attn_ln.weight", prefix))?,
                cross_attn_ln_b: model.get_tensor_f32(&format!("{}.cross_attn_ln.bias", prefix))?,
                cross_attn_q_w: model
                    .get_linear_weight(&format!("{}.cross_attn.query.weight", prefix))?,
                cross_attn_q_b: model
                    .get_tensor_f32(&format!("{}.cross_attn.query.bias", prefix))?,
                cross_attn_k_w: model
                    .get_linear_weight(&format!("{}.cross_attn.key.weight", prefix))?,
                cross_attn_v_w: model
                    .get_linear_weight(&format!("{}.cross_attn.value.weight", prefix))?,
                cross_attn_v_b: model
                    .get_tensor_f32(&format!("{}.cross_attn.value.bias", prefix))?,
                cross_attn_out_w: model
                    .get_linear_weight(&format!("{}.cross_attn.out.weight", prefix))?,
                cross_attn_out_b: model
                    .get_tensor_f32(&format!("{}.cross_attn.out.bias", prefix))?,

                mlp_ln_w: model.get_tensor_f32(&format!("{}.mlp_ln.weight", prefix))?,
                mlp_ln_b: model.get_tensor_f32(&format!("{}.mlp_ln.bias", prefix))?,
                mlp0_w: model.get_linear_weight(&format!("{}.mlp.0.weight", prefix))?,
                mlp0_b: model.get_tensor_f32(&format!("{}.mlp.0.bias", prefix))?,
                mlp2_w: model.get_linear_weight(&format!("{}.mlp.2.weight", prefix))?,
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

/// Key/value arena shared by every beam branch of one transcription.
///
/// Beam search branches constantly: each retained hypothesis needs the
/// attention state of the tokens it has emitted so far. Storing one cache per
/// candidate meant deep-cloning every layer's key and value tensors at every
/// step. The arena instead keeps one flat buffer per `(layer, slot)` pair and
/// the beam only copies a slot when a hypothesis actually forks, so memory is
/// bounded by `n_layers * n_slots` and branching costs a byte copy rather than
/// a full cache allocation.
pub struct DecoderKvCache {
    /// Layer index -> slot index -> self-attention keys: flat `[t_past * d_model]`.
    pub k_self: Vec<Vec<Vec<f32>>>,
    /// Layer index -> slot index -> self-attention values: flat `[t_past * d_model]`.
    pub v_self: Vec<Vec<Vec<f32>>>,
    /// Layer index -> precomputed cross-attention keys: `[n_audio * d_model]`, shared by all slots.
    pub k_cross: Vec<Vec<f32>>,
    /// Layer index -> precomputed cross-attention values: `[n_audio * d_model]`, shared by all slots.
    pub v_cross: Vec<Vec<f32>>,
    /// Tokens already stored in each slot.
    pub t_past: Vec<usize>,
    /// Width of one cached row.
    pub d_model: usize,
}

impl DecoderKvCache {
    pub fn new(n_layers: usize, n_slots: usize, d_model: usize) -> Self {
        Self {
            k_self: vec![vec![Vec::new(); n_slots]; n_layers],
            v_self: vec![vec![Vec::new(); n_slots]; n_layers],
            k_cross: vec![Vec::new(); n_layers],
            v_cross: vec![Vec::new(); n_layers],
            t_past: vec![0; n_slots],
            d_model,
        }
    }

    /// Reset every slot while keeping the arena, the cross-attention keys and
    /// the allocated capacities, so reusing a cache across transcriptions does
    /// not re-grow the buffers.
    pub fn clear(&mut self) {
        for layer in &mut self.k_self {
            for buf in layer.iter_mut() {
                buf.clear();
            }
        }
        for layer in &mut self.v_self {
            for buf in layer.iter_mut() {
                buf.clear();
            }
        }
        self.t_past.fill(0);
    }

    /// Drop the history of a single slot, leaving the other branches untouched.
    pub fn clear_slot(&mut self, slot: usize) {
        assert!(slot < self.t_past.len(), "slot {slot} out of range");
        for layer in &mut self.k_self {
            layer[slot].clear();
        }
        for layer in &mut self.v_self {
            layer[slot].clear();
        }
        self.t_past[slot] = 0;
    }

    /// Drop the cached cross-attention projections.
    ///
    /// They are derived from one encoder output, so a new encoder pass (the C
    /// API exposes encode and decode separately) must discard them even when
    /// the new window happens to have the same length.
    pub fn invalidate_cross(&mut self) {
        for k in &mut self.k_cross {
            k.clear();
        }
        for v in &mut self.v_cross {
            v.clear();
        }
    }

    /// Grow the arena to at least `n_slots` slots. Existing contents survive,
    /// so a beam can widen without invalidating the hypotheses it already has.
    pub fn ensure_slots(&mut self, n_slots: usize) {
        if self.t_past.len() >= n_slots {
            return;
        }
        for layer in &mut self.k_self {
            layer.resize_with(n_slots, Vec::new);
        }
        for layer in &mut self.v_self {
            layer.resize_with(n_slots, Vec::new);
        }
        self.t_past.resize(n_slots, 0);
    }

    /// Duplicate the attention history of `src` into `dst`.
    ///
    /// Only the valid prefix (`t_past[src]` rows) is copied; the destination
    /// keeps its capacity so a fork does not allocate.
    pub fn copy_slot(&mut self, dst: usize, src: usize) {
        assert!(dst < self.t_past.len(), "slot {dst} out of range");
        assert!(src < self.t_past.len(), "slot {src} out of range");
        if dst == src {
            return;
        }
        let len = self.t_past[src] * self.d_model;
        for layer in &mut self.k_self {
            copy_row_block(layer, dst, src, len);
        }
        for layer in &mut self.v_self {
            copy_row_block(layer, dst, src, len);
        }
        self.t_past[dst] = self.t_past[src];
    }

    /// Number of tokens currently cached in `slot`.
    pub fn slot_len(&self, slot: usize) -> usize {
        self.t_past[slot]
    }
}

/// Copy `len` floats between two slots of the same layer without going through
/// a temporary allocation.
fn copy_row_block(slots: &mut [Vec<f32>], dst: usize, src: usize, len: usize) {
    if dst == src {
        return;
    }
    let (dst_buf, src_buf) = if dst < src {
        let (lo, hi) = slots.split_at_mut(src);
        (&mut lo[dst], &hi[0])
    } else {
        let (lo, hi) = slots.split_at_mut(dst);
        (&mut hi[0], &lo[src])
    };
    dst_buf.clear();
    dst_buf.extend_from_slice(&src_buf[..len]);
}

/// Activation buffers reused across every decode step.
///
/// Decoding allocates one buffer set per generated token: nine activation
/// buffers plus the vocabulary-sized logits row. That is roughly a few hundred
/// kilobytes of allocation traffic per token, with the logits dominating (about
/// 207 KB for a 51865-token model). The allocator churn showed up directly in
/// the decode profile, so the buffers are owned by the caller and only resized:
/// `resize` keeps the capacity of an already-large vector, and nothing calls
/// `shrink_to_fit`, so a transcription reaches steady state after its first
/// tokens.
#[derive(Default)]
pub struct DecoderWorkspace {
    x: Vec<f32>,
    norm_buf: Vec<f32>,
    q_buf: Vec<f32>,
    k_buf: Vec<f32>,
    v_buf: Vec<f32>,
    attn_out: Vec<f32>,
    proj_out: Vec<f32>,
    mlp0_out: Vec<f32>,
    final_norm: Vec<f32>,
    logits: Vec<f32>,
    logits_len: usize,
}

impl DecoderWorkspace {
    pub fn new() -> Self {
        Self::default()
    }

    /// Logits produced by the last decode call, laid out `[rows * n_vocab]`.
    pub fn logits(&self) -> &[f32] {
        &self.logits[..self.logits_len]
    }

    /// Logits produced by the last decode call, mutably.
    ///
    /// The logit filtering pipeline in the transcription loop suppresses and
    /// biases entries in place, so handing out a mutable view avoids copying
    /// the vocabulary-sized row out of the workspace and back.
    pub fn logits_mut(&mut self) -> &mut [f32] {
        let len = self.logits_len;
        &mut self.logits[..len]
    }

    /// Release the logical contents of every buffer while keeping its
    /// allocation.
    ///
    /// A `WhisperState` outlives individual audio windows, so the activation
    /// buffers must be reusable across transcriptions. `Vec::clear` drops the
    /// length but leaves the heap block in place, which is what keeps the next
    /// window from re-growing a few hundred kilobytes per buffer.
    pub fn clear(&mut self) {
        self.x.clear();
        self.norm_buf.clear();
        self.q_buf.clear();
        self.k_buf.clear();
        self.v_buf.clear();
        self.attn_out.clear();
        self.proj_out.clear();
        self.mlp0_out.clear();
        self.final_norm.clear();
        self.logits.clear();
        self.logits_len = 0;
    }

    fn ensure(&mut self, n_tokens: usize, d_model: usize, d_mlp: usize, n_vocab: usize) {
        resize_exact(&mut self.x, n_tokens * d_model);
        resize_exact(&mut self.norm_buf, n_tokens * d_model);
        resize_exact(&mut self.q_buf, n_tokens * d_model);
        resize_exact(&mut self.k_buf, n_tokens * d_model);
        resize_exact(&mut self.v_buf, n_tokens * d_model);
        resize_exact(&mut self.attn_out, n_tokens * d_model);
        resize_exact(&mut self.proj_out, n_tokens * d_model);
        resize_exact(&mut self.mlp0_out, n_tokens * d_mlp);
        resize_exact(&mut self.final_norm, n_tokens * d_model);
        resize_exact(&mut self.logits, n_tokens * n_vocab);
    }
}

/// Shrink or grow a buffer to exactly `len` without releasing its capacity.
fn resize_exact(buf: &mut Vec<f32>, len: usize) {
    if buf.len() != len {
        buf.resize(len, 0.0);
    }
}

/// One sequence to evaluate in a batched decode step.
///
/// `slot` selects the KV arena slot and `tokens` are the tokens whose rows have
/// not been fed to the model yet; the caller is responsible for only passing
/// tokens that are actually new.
pub struct DecodeSlot<'a> {
    pub slot: usize,
    pub tokens: &'a [i32],
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
    ///
    /// These tensors only depend on the audio window, so every beam branch
    /// reads the same copies and no slot-local state is needed.
    pub fn precompute_cross_attn(&self, encoder_output: &[f32], cache: &mut DecoderKvCache) {
        assert_eq!(
            cache.k_cross.len(),
            self.weights.blocks.len(),
            "KV cache layer count does not match the decoder"
        );
        let n_audio_tokens = encoder_output.len() / self.d_model;
        for (l, block) in self.weights.blocks.iter().enumerate() {
            let mut k_cross = vec![0.0f32; n_audio_tokens * self.d_model];
            let mut v_cross = vec![0.0f32; n_audio_tokens * self.d_model];

            block
                .cross_attn_k_w
                .linear(encoder_output, None, n_audio_tokens, &mut k_cross);
            block.cross_attn_v_w.linear(
                encoder_output,
                Some(&block.cross_attn_v_b),
                n_audio_tokens,
                &mut v_cross,
            );

            cache.k_cross[l] = k_cross;
            cache.v_cross[l] = v_cross;
        }
    }

    /// Decode `tokens` in slot 0 and return `[tokens.len() * n_vocab]` logits.
    pub fn decode<'a>(
        &self,
        tokens: &[i32],
        encoder_output: &[f32],
        cache: &mut DecoderKvCache,
        ws: &'a mut DecoderWorkspace,
    ) -> &'a mut [f32] {
        self.decode_batch(&[DecodeSlot { slot: 0, tokens }], encoder_output, cache, ws)
    }

    /// Batched decode: every sequence is embedded, normalized and projected in
    /// the same pass, and only the attention itself runs per slot.
    ///
    /// Rows are grouped slot-major: the slot given by `seqs[0]` occupies the
    /// first rows and so on, and within one slot the tokens keep their order.
    /// The returned logits follow that same row layout.
    pub fn decode_batch<'a>(
        &self,
        seqs: &[DecodeSlot<'_>],
        encoder_output: &[f32],
        cache: &mut DecoderKvCache,
        ws: &'a mut DecoderWorkspace,
    ) -> &'a mut [f32] {
        let total_tokens: usize = seqs.iter().map(|s| s.tokens.len()).sum();
        if total_tokens == 0 {
            ws.logits_len = 0;
            return ws.logits_mut();
        }

        let d_model = self.d_model;
        let d_mlp = d_model * 4;
        let n_vocab = self.n_vocab;

        let n_audio_tokens = encoder_output.len() / d_model;

        // The cross-attention tensors depend on the audio window, not on the
        // text history. Recompute them only when they are absent or sized for a
        // different window; a reused state must never attend to stale audio.
        if cache
            .k_cross
            .first()
            .is_none_or(|k| k.len() != n_audio_tokens * d_model)
        {
            self.precompute_cross_attn(encoder_output, cache);
        }

        // Slots are addressed by index, so a branch that was forked after the
        // cache was created may need a slot that does not exist yet.
        let max_slot = seqs.iter().map(|s| s.slot).max().unwrap_or(0);
        cache.ensure_slots(max_slot + 1);

        ws.ensure(total_tokens, d_model, d_mlp, n_vocab);

        // 1. Token embedding lookup plus learned positional embedding.
        let mut row = 0usize;
        for seq in seqs {
            let t_past = cache.t_past[seq.slot];
            for (i, &tok) in seq.tokens.iter().enumerate() {
                let dst = &mut ws.x[row * d_model..(row + 1) * d_model];
                self.weights.token_embedding.row_into(tok as usize, dst);
                let pos = t_past + i;
                let pe = &self.weights.positional_embedding[pos * d_model..(pos + 1) * d_model];
                for (value, &offset) in dst.iter_mut().zip(pe) {
                    *value += offset;
                }
                row += 1;
            }
        }
        debug_assert_eq!(row, total_tokens);

        // 2. Transformer blocks. Every row of the batch shares the same weights,
        // so the projections run once over all sequences. A single-token decode
        // is an M=1 GEMV: it is limited by how fast the weight matrix can be
        // streamed, not by arithmetic. With one row per beam branch the same
        // matrix feed becomes an M>=2 GEMM, so the weight bytes are read once
        // per step instead of once per candidate.
        for (l, block) in self.weights.blocks.iter().enumerate() {
            // (a) Pre-LN for self-attention.
            layer_norm(
                &ws.x,
                &block.attn_ln_w,
                &block.attn_ln_b,
                1e-5,
                total_tokens,
                d_model,
                &mut ws.norm_buf,
            );

            // (b) Self-attention Q, K, V for all rows.
            block.attn_q_w.linear(
                &ws.norm_buf,
                Some(&block.attn_q_b),
                total_tokens,
                &mut ws.q_buf,
            );
            block
                .attn_k_w
                .linear(&ws.norm_buf, None, total_tokens, &mut ws.k_buf);
            block.attn_v_w.linear(
                &ws.norm_buf,
                Some(&block.attn_v_b),
                total_tokens,
                &mut ws.v_buf,
            );

            // (c) Append each slot's new K/V rows and run causal attention over
            // that slot's contiguous row range. The slot is the unit of beam
            // branching, so the arena layout mirrors the hypothesis tree.
            let mut start = 0usize;
            for seq in seqs {
                let n = seq.tokens.len();
                let lo = start * d_model;
                let hi = (start + n) * d_model;
                start += n;
                if n == 0 {
                    continue;
                }
                let t_past = cache.t_past[seq.slot];
                // The C API lets a caller rewind a sequence by passing an
                // explicit `n_past`. Drop any rows beyond that point so the
                // renewed tokens overwrite the stale history instead of being
                // appended behind it; truncating to the current length is a
                // no-op in the ordinary append path.
                let primed = t_past * d_model;
                cache.k_self[l][seq.slot].truncate(primed);
                cache.v_self[l][seq.slot].truncate(primed);
                cache.k_self[l][seq.slot].extend_from_slice(&ws.k_buf[lo..hi]);
                cache.v_self[l][seq.slot].extend_from_slice(&ws.v_buf[lo..hi]);
                multi_head_attention_with_cache(
                    &ws.q_buf[lo..hi],
                    &cache.k_self[l][seq.slot],
                    &cache.v_self[l][seq.slot],
                    n,
                    t_past + n,
                    t_past,
                    self.n_heads,
                    d_model,
                    &mut ws.attn_out[lo..hi],
                );
            }
            debug_assert_eq!(start, total_tokens);

            // (d) Self-attention output projection and residual for all rows.
            block.attn_out_w.linear(
                &ws.attn_out,
                Some(&block.attn_out_b),
                total_tokens,
                &mut ws.proj_out,
            );
            add_inplace(&mut ws.x, &ws.proj_out);

            // (e) Pre-LN for cross-attention.
            layer_norm(
                &ws.x,
                &block.cross_attn_ln_w,
                &block.cross_attn_ln_b,
                1e-5,
                total_tokens,
                d_model,
                &mut ws.norm_buf,
            );

            // (f) Cross-attention queries for all rows.
            block.cross_attn_q_w.linear(
                &ws.norm_buf,
                Some(&block.cross_attn_q_b),
                total_tokens,
                &mut ws.q_buf,
            );

            // (g) Cross-attention against the shared encoder key/value tensors.
            let mut start = 0usize;
            for seq in seqs {
                let n = seq.tokens.len();
                let lo = start * d_model;
                let hi = (start + n) * d_model;
                start += n;
                if n == 0 {
                    continue;
                }
                multi_head_attention(
                    &ws.q_buf[lo..hi],
                    &cache.k_cross[l],
                    &cache.v_cross[l],
                    n,
                    n_audio_tokens,
                    self.n_heads,
                    d_model,
                    false,
                    &mut ws.attn_out[lo..hi],
                );
            }

            // (h) Cross-attention output projection and residual.
            block.cross_attn_out_w.linear(
                &ws.attn_out,
                Some(&block.cross_attn_out_b),
                total_tokens,
                &mut ws.proj_out,
            );
            add_inplace(&mut ws.x, &ws.proj_out);

            // (i) Pre-LN for the MLP.
            layer_norm(
                &ws.x,
                &block.mlp_ln_w,
                &block.mlp_ln_b,
                1e-5,
                total_tokens,
                d_model,
                &mut ws.norm_buf,
            );

            // (j) MLP over all rows.
            block.mlp0_w.linear(
                &ws.norm_buf,
                Some(&block.mlp0_b),
                total_tokens,
                &mut ws.mlp0_out,
            );
            gelu_inplace(&mut ws.mlp0_out);
            block.mlp2_w.linear(
                &ws.mlp0_out,
                Some(&block.mlp2_b),
                total_tokens,
                &mut ws.proj_out,
            );
            add_inplace(&mut ws.x, &ws.proj_out);
        }

        // Advance each slot only after every layer consumed the previous
        // length: the layer loops above use it as the causal offset.
        for seq in seqs {
            cache.t_past[seq.slot] += seq.tokens.len();
        }

        // 3. Final LayerNorm.
        layer_norm(
            &ws.x,
            &self.weights.ln_post_w,
            &self.weights.ln_post_b,
            1e-5,
            total_tokens,
            d_model,
            &mut ws.final_norm,
        );

        // 4. Vocabulary projection of all rows in one pass.
        self.weights
            .token_embedding
            .linear(&ws.final_norm, None, total_tokens, &mut ws.logits);

        ws.logits_len = total_tokens * n_vocab;
        ws.logits_mut()
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
