# @whisper-rs/wasm

Pure-Rust WebAssembly (WASM) bindings for `whisper.rs`. Runs directly in modern browsers and Node.js without any C++ toolchains or Emscripten.

## Installation / Build

```bash
cd bindings/javascript
wasm-pack build --target web --out-dir pkg
```

## Browser Usage

```javascript
import init, { WhisperWasm } from './pkg/whisper_wasm.js';

async function run() {
  await init();

  // Load model buffer (.bin)
  const modelResponse = await fetch('/models/ggml-tiny.en.bin');
  const modelBytes = new Uint8Array(await modelResponse.arrayBuffer());

  const whisper = new WhisperWasm(modelBytes);

  // Load WAV audio buffer
  const audioResponse = await fetch('/samples/jfk.wav');
  const wavBytes = new Uint8Array(await audioResponse.arrayBuffer());

  const text = whisper.transcribe(wavBytes);
  console.log('Transcribed Text:', text);
}
```
