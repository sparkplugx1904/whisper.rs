export interface WhisperSegment {
  start: number;
  end: number;
  text: string;
}

export class WhisperContext {
  static fromBuffer(modelBuffer: Uint8Array): WhisperContext;
  transcribe(audioPcmFloat32: Float32Array): string;
  detectLanguage(audioPcmFloat32: Float32Array): string;
}
