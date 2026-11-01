use wasm_bindgen::prelude::*;
use whisper_rs_pure::audio::WavAudio;
use whisper_rs_pure::model::WhisperModel;
use whisper_rs_pure::{WhisperContext, WhisperParams};

#[wasm_bindgen]
pub struct WhisperWasm {
    ctx: WhisperContext,
}

#[wasm_bindgen]
impl WhisperWasm {
    #[wasm_bindgen(constructor)]
    pub fn new(model_bytes: &[u8]) -> Result<WhisperWasm, JsValue> {
        let model = WhisperModel::load_from_bytes(model_bytes.to_vec())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let ctx = WhisperContext::new(model)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Self { ctx })
    }

    #[wasm_bindgen]
    pub fn transcribe(&self, wav_bytes: &[u8]) -> Result<String, JsValue> {
        let audio = WavAudio::load_from_bytes(wav_bytes)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let params = WhisperParams::default();
        let segments = self.ctx.transcribe_pcm(&audio.samples, &params);
        let text = segments
            .iter()
            .map(|s| s.text.trim())
            .collect::<Vec<_>>()
            .join(" ");
        Ok(text)
    }
}
