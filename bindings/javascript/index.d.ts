export class WhisperWasm {
  /**
   * Create a new WhisperWasm instance from model binary buffer (.bin).
   */
  constructor(modelBytes: Uint8Array);

  /**
   * Transcribe speech from WAV audio file bytes (RIFF/WAV container).
   */
  transcribe(wavBytes: Uint8Array): string;
}

export default function init(): Promise<void>;
