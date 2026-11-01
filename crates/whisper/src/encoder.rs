use whisper_model::{ModelError, WhisperModel};
use whisper_tensor::ops::{
    add_inplace, conv1d_c_first, conv1d_t_first, gelu_inplace, layer_norm, linear,
    multi_head_attention,
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
            let block = EncoderBlockWeights {
                attn_ln_w: model.get_tensor_f32(&format!("{}.attn_ln.weight", prefix))?,
                attn_ln_b: model.get_tensor_f32(&format!("{}.attn_ln.bias", prefix))?,
                attn_q_w: model.get_tensor_f32(&format!("{}.attn.query.weight", prefix))?,
                attn_q_b: model.get_tensor_f32(&format!("{}.attn.query.bias", prefix))?,
                attn_k_w: model.get_tensor_f32(&format!("{}.attn.key.weight", prefix))?,
                attn_v_w: model.get_tensor_f32(&format!("{}.attn.value.weight", prefix))?,
                attn_v_b: model.get_tensor_f32(&format!("{}.attn.value.bias", prefix))?,
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
        let l_in = 2 * self.n_audio_ctx; // 3000 frames
        assert_eq!(mel_chunk.len(), self.n_mels * l_in);

        // 1. Conv1: [n_mels, 3000] -> [d_model, 3000] with GELU
        let mut conv1_out = vec![0.0f32; self.d_model * l_in];
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

        // 2. Conv2: [d_model, 3000] -> [1500, d_model] with GELU
        let l_out = self.n_audio_ctx; // 1500 tokens
        let mut x = vec![0.0f32; l_out * self.d_model];
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

            // (b) Self-Attention Q, K, V projections
            linear(
                &norm_buf,
                &block.attn_q_w,
                Some(&block.attn_q_b),
                self.d_model,
                self.d_model,
                l_out,
                &mut q_buf,
            );
            linear(
                &norm_buf,
                &block.attn_k_w,
                None, // Note: Whisper encoder key projection has no bias
                self.d_model,
                self.d_model,
                l_out,
                &mut k_buf,
            );
            linear(
                &norm_buf,
                &block.attn_v_w,
                Some(&block.attn_v_b),
                self.d_model,
                self.d_model,
                l_out,
                &mut v_buf,
            );

            // (c) Multi-Head Attention
            multi_head_attention(
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

            // (d) Out projection & residual
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

            // (f) MLP: linear 0 -> GELU -> linear 2 -> residual
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

        final_out
    }
}
