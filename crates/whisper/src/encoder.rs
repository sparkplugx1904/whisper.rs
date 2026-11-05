use whisper_model::{ModelError, WhisperModel};
use whisper_tensor::ops::{
    add_inplace, conv1d_c_first, conv1d_t_first, flash_attention_2, gelu_inplace, layer_norm,
    linear,
};

#[derive(Clone)]
pub struct EncoderBlockWeights {
    pub attn_ln_w: Vec<f32>,
    pub attn_ln_b: Vec<f32>,
    pub attn_q_w: Vec<f32>,
    pub attn_q_b: Vec<f32>,
    pub attn_k_w: Vec<f32>,
    pub attn_v_w: Vec<f32>,
    pub attn_v_b: Vec<f32>,
    pub attn_qkv_w: Vec<f32>,
    pub attn_qkv_b: Vec<f32>,
    pub attn_out_w: Vec<f32>,
    pub attn_out_b: Vec<f32>,
    pub mlp_ln_w: Vec<f32>,
    pub mlp_ln_b: Vec<f32>,
    pub mlp0_w: Vec<f32>,
    pub mlp0_b: Vec<f32>,
    pub mlp2_w: Vec<f32>,
    pub mlp2_b: Vec<f32>,
}

#[derive(Clone)]
pub struct EncoderWeights {
    pub conv1_w: Vec<f32>,
    pub conv1_b: Vec<f32>,
    pub conv2_w: Vec<f32>,
    pub conv2_b: Vec<f32>,
    pub positional_embedding: Vec<f32>,
    pub blocks: Vec<EncoderBlockWeights>,
    pub ln_post_w: Vec<f32>,
    pub ln_post_b: Vec<f32>,
}

impl EncoderWeights {
    pub fn load(model: &WhisperModel) -> Result<Self, ModelError> {
        let conv1_w = model.get_tensor_f32("encoder.conv1.weight")?;
        let conv1_b = model.get_tensor_f32("encoder.conv1.bias")?;
        let conv2_w = model.get_tensor_f32("encoder.conv2.weight")?;
        let conv2_b = model.get_tensor_f32("encoder.conv2.bias")?;
        let positional_embedding = model.get_tensor_f32("encoder.positional_embedding")?;

        let n_layers = model.hparams.n_audio_layer as usize;
        let mut blocks = Vec::with_capacity(n_layers);

        for l in 0..n_layers {
            let prefix = format!("encoder.blocks.{}", l);
            let attn_q_w = model.get_tensor_f32(&format!("{}.attn.query.weight", prefix))?;
            let attn_q_b = model.get_tensor_f32(&format!("{}.attn.query.bias", prefix))?;
            let attn_k_w = model.get_tensor_f32(&format!("{}.attn.key.weight", prefix))?;
            let attn_v_w = model.get_tensor_f32(&format!("{}.attn.value.weight", prefix))?;
            let attn_v_b = model.get_tensor_f32(&format!("{}.attn.value.bias", prefix))?;

            let d_model = model.hparams.n_audio_state as usize;
            let mut attn_qkv_w = Vec::with_capacity(3 * d_model * d_model);
            attn_qkv_w.extend_from_slice(&attn_q_w);
            attn_qkv_w.extend_from_slice(&attn_k_w);
            attn_qkv_w.extend_from_slice(&attn_v_w);

            let mut attn_qkv_b = Vec::with_capacity(3 * d_model);
            attn_qkv_b.extend_from_slice(&attn_q_b);
            attn_qkv_b.resize(attn_qkv_b.len() + d_model, 0.0); // Key has no bias
            attn_qkv_b.extend_from_slice(&attn_v_b);

            let block = EncoderBlockWeights {
                attn_ln_w: model.get_tensor_f32(&format!("{}.attn_ln.weight", prefix))?,
                attn_ln_b: model.get_tensor_f32(&format!("{}.attn_ln.bias", prefix))?,
                attn_q_w,
                attn_q_b,
                attn_k_w,
                attn_v_w,
                attn_v_b,
                attn_qkv_w,
                attn_qkv_b,
                attn_out_w: model.get_tensor_f32(&format!("{}.attn.out.weight", prefix))?,
                attn_out_b: model.get_tensor_f32(&format!("{}.attn.out.bias", prefix))?,
                mlp_ln_w: model.get_tensor_f32(&format!("{}.mlp_ln.weight", prefix))?,
                mlp_ln_b: model.get_tensor_f32(&format!("{}.mlp_ln.bias", prefix))?,
                mlp0_w: model.get_tensor_f32(&format!("{}.mlp.0.weight", prefix))?,
                mlp0_b: model.get_tensor_f32(&format!("{}.mlp.0.bias", prefix))?,
                mlp2_w: model.get_tensor_f32(&format!("{}.mlp.2.weight", prefix))?,
                mlp2_b: model.get_tensor_f32(&format!("{}.mlp.2.bias", prefix))?,
            };
            blocks.push(block);
        }

        let ln_post_w = model.get_tensor_f32("encoder.ln_post.weight")?;
        let ln_post_b = model.get_tensor_f32("encoder.ln_post.bias")?;

        Ok(Self {
            conv1_w,
            conv1_b,
            conv2_w,
            conv2_b,
            positional_embedding,
            blocks,
            ln_post_w,
            ln_post_b,
        })
    }
}

pub struct WhisperEncoder {
    pub weights: EncoderWeights,
    pub n_mels: usize,
    pub n_audio_ctx: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
}

impl WhisperEncoder {
    pub fn new(model: &WhisperModel) -> Result<Self, ModelError> {
        let weights = EncoderWeights::load(model)?;
        Ok(Self {
            weights,
            n_mels: model.hparams.n_mels as usize,
            n_audio_ctx: model.hparams.n_audio_ctx as usize,
            d_model: model.hparams.n_audio_state as usize,
            n_heads: model.hparams.n_audio_head as usize,
            n_layers: model.hparams.n_audio_layer as usize,
        })
    }

    /// Forward pass of the audio encoder.
    /// `mel_chunk`: flattened mel spectrogram slice of size [n_mels, 3000].
    /// Returns encoded output of size [1500, d_model].
    pub fn forward(&self, mel_chunk: &[f32]) -> Vec<f32> {
        let do_profile = std::env::var("WHISPER_PROFILE").is_ok();
        let t_start = std::time::Instant::now();

        let l_in = 2 * self.n_audio_ctx; // 3000 frames
        assert_eq!(mel_chunk.len(), self.n_mels * l_in);

        // 1. Conv1: [n_mels, 3000] -> [d_model, 3000] with GELU
        let mut conv1_out = vec![0.0f32; self.d_model * l_in];
        let t0 = std::time::Instant::now();
        conv1d_c_first(
            mel_chunk,
            &self.weights.conv1_w,
            &self.weights.conv1_b,
            self.n_mels,
            self.d_model,
            l_in,
            3,
            1,
            1,
            true,
            &mut conv1_out,
        );
        let d_conv1 = t0.elapsed();

        // 2. Conv2: [d_model, 3000] -> [1500, d_model] with GELU
        let l_out = self.n_audio_ctx; // 1500 tokens
        let mut x = vec![0.0f32; l_out * self.d_model];
        let t0 = std::time::Instant::now();
        conv1d_t_first(
            &conv1_out,
            &self.weights.conv2_w,
            &self.weights.conv2_b,
            self.d_model,
            self.d_model,
            l_in,
            3,
            2,
            1,
            true,
            &mut x,
        );
        let d_conv2 = t0.elapsed();

        // 3. Add positional embedding
        assert_eq!(
            self.weights.positional_embedding.len(),
            l_out * self.d_model
        );
        add_inplace(&mut x, &self.weights.positional_embedding);
        // Workspace buffers to avoid allocations per layer
        let mut norm_buf = vec![0.0f32; l_out * self.d_model];
        let mut q_buf = vec![0.0f32; l_out * self.d_model];
        let mut k_buf = vec![0.0f32; l_out * self.d_model];
        let mut v_buf = vec![0.0f32; l_out * self.d_model];
        let mut attn_out = vec![0.0f32; l_out * self.d_model];
        let mut proj_out = vec![0.0f32; l_out * self.d_model];
        let mut qkv_buf = vec![0.0f32; l_out * 3 * self.d_model];

        let d_mlp = self.d_model * 4;
        let mut mlp0_out = vec![0.0f32; l_out * d_mlp];
        // 4. Transformer blocks
        for block in &self.weights.blocks {
            // (a) Pre-LN for attention
            layer_norm(
                &x,
                &block.attn_ln_w,
                &block.attn_ln_b,
                1e-5,
                l_out,
                self.d_model,
                &mut norm_buf,
            );
            linear(
                &norm_buf,
                &block.attn_qkv_w,
                Some(&block.attn_qkv_b),
                self.d_model,
                3 * self.d_model,
                l_out,
                &mut qkv_buf,
            );

            let d_m = self.d_model;
            let chunk_qkv = 3 * d_m;
            for i in 0..l_out {
                let row = &qkv_buf[i * chunk_qkv..(i + 1) * chunk_qkv];
                q_buf[i * d_m..(i + 1) * d_m].copy_from_slice(&row[..d_m]);
                k_buf[i * d_m..(i + 1) * d_m].copy_from_slice(&row[d_m..2 * d_m]);
                v_buf[i * d_m..(i + 1) * d_m].copy_from_slice(&row[2 * d_m..]);
            }
            // (c) Multi-Head Attention with Flash Attention 2
            flash_attention_2(
                &q_buf,
                &k_buf,
                &v_buf,
                l_out,
                l_out,
                self.n_heads,
                self.d_model,
                false, // non-causal
                &mut attn_out,
            );

            linear(
                &attn_out,
                &block.attn_out_w,
                Some(&block.attn_out_b),
                self.d_model,
                self.d_model,
                l_out,
                &mut proj_out,
            );
            add_inplace(&mut x, &proj_out);

            // (e) Pre-LN for MLP
            layer_norm(
                &x,
                &block.mlp_ln_w,
                &block.mlp_ln_b,
                1e-5,
                l_out,
                self.d_model,
                &mut norm_buf,
            );
            linear(
                &norm_buf,
                &block.mlp0_w,
                Some(&block.mlp0_b),
                self.d_model,
                d_mlp,
                l_out,
                &mut mlp0_out,
            );
            gelu_inplace(&mut mlp0_out);

            linear(
                &mlp0_out,
                &block.mlp2_w,
                Some(&block.mlp2_b),
                d_mlp,
                self.d_model,
                l_out,
                &mut proj_out,
            );
            add_inplace(&mut x, &proj_out);
        }
        // 5. Final LayerNorm
        let mut final_out = vec![0.0f32; l_out * self.d_model];
        layer_norm(
            &x,
            &self.weights.ln_post_w,
            &self.weights.ln_post_b,
            1e-5,
            l_out,
            self.d_model,
            &mut final_out,
        );
        if do_profile {
            let total = t_start.elapsed();
            eprintln!(
                "[PROFILE ENCODER] Total: {:.2}ms | Conv1: {:.2}ms | Conv2: {:.2}ms | Blocks: {:.2}ms",
                total.as_secs_f64() * 1000.0,
                d_conv1.as_secs_f64() * 1000.0,
                d_conv2.as_secs_f64() * 1000.0,
                (total - d_conv1 - d_conv2).as_secs_f64() * 1000.0
            );
        }

        final_out
    }
}
