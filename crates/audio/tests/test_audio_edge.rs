use byteorder::{LittleEndian, WriteBytesExt};
use std::f32::consts::PI;
use whisper_audio::{
    compute_log_mel_spectrogram, resample_to_16k, WavAudio, N_MELS, WHISPER_SAMPLE_RATE,
};

/// Helper to generate synthetic mel filterbank for testing [n_mels, 201].
fn make_dummy_filters(n_mels: usize) -> Vec<f32> {
    let n_fft_bins = 201;
    let mut filters = vec![0.0f32; n_mels * n_fft_bins];
    // Triangular filter bands
    for m in 0..n_mels {
        let center = (m * n_fft_bins) / n_mels;
        let start = center.saturating_sub(5);
        let end = (center + 5).min(n_fft_bins);
        for k in start..end {
            let weight = 1.0 - ((k as f32 - center as f32).abs() / 5.0);
            filters[m * n_fft_bins + k] = weight.max(0.01);
        }
    }
    filters
}

/// Helper to create a well-formed WAV in-memory byte buffer.
fn create_wav_bytes(
    channels: u16,
    sample_rate: u32,
    bits_per_sample: u16,
    audio_format: u16, // 1 = PCM, 3 = IEEE float
    data: &[u8],
) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"RIFF");
    let file_size = 36u32 + data.len() as u32;
    out.extend_from_slice(&file_size.to_le_bytes());
    out.extend_from_slice(b"WAVE");

    // fmt chunk
    out.extend_from_slice(b"fmt ");
    let fmt_chunk_size = 16u32;
    out.extend_from_slice(&fmt_chunk_size.to_le_bytes());
    out.extend_from_slice(&audio_format.to_le_bytes());
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    let byte_rate = sample_rate * (channels as u32) * (bits_per_sample as u32 / 8);
    out.extend_from_slice(&byte_rate.to_le_bytes());
    let block_align = channels * (bits_per_sample / 8);
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&bits_per_sample.to_le_bytes());

    // data chunk
    out.extend_from_slice(b"data");
    let data_len = data.len() as u32;
    out.extend_from_slice(&data_len.to_le_bytes());
    out.extend_from_slice(data);

    out
}

#[test]
fn test_zero_length_audio_handling() {
    // 1. Resampling empty slices at various sample rates
    for &rate in &[8000, 16000, 22050, 44100, 48000] {
        let resampled = resample_to_16k(&[], rate);
        assert!(
            resampled.is_empty(),
            "Empty input at rate {} must produce empty output, got len {}",
            rate,
            resampled.len()
        );
    }

    // 2. Loading zero-length 16-bit PCM WAV
    let empty_pcm_wav = create_wav_bytes(1, 16000, 16, 1, &[]);
    let audio = WavAudio::load_from_bytes(&empty_pcm_wav)
        .expect("Zero-length PCM WAV must load successfully");
    assert_eq!(audio.sample_rate, 16000);
    assert_eq!(audio.channels, 1);
    assert!(audio.samples.is_empty());

    // 3. Loading zero-length 32-bit float stereo WAV
    let empty_float_wav = create_wav_bytes(2, 48000, 32, 3, &[]);
    let audio_float = WavAudio::load_from_bytes(&empty_float_wav)
        .expect("Zero-length float WAV must load successfully");
    assert_eq!(audio_float.sample_rate, 48000);
    assert_eq!(audio_float.channels, 2);
    assert!(audio_float.samples.is_empty());

    // 4. Spectrogram calculation on zero-length samples
    let filters = make_dummy_filters(N_MELS);
    let mel = compute_log_mel_spectrogram(&[], &filters, N_MELS);
    assert_eq!(mel.n_mel, N_MELS);
    assert!(mel.n_len > 0, "Padded n_len must be non-zero");
    assert_eq!(mel.n_len_org, 0, "n_len_org for 0 samples must be 0");
    for &v in &mel.data {
        assert!(
            v.is_finite(),
            "Zero-length spectrogram must have finite values"
        );
    }

    // 5. Chunk extraction from empty-signal mel
    let chunk = mel.get_chunk(0, 1500);
    assert_eq!(chunk.len(), N_MELS * 3000);
}

#[test]
fn test_8k_upsampling_to_16k() {
    let src_rate = 8000;
    // Step response: ramp [0.0, 1.0, 2.0, 3.0, 4.0]
    let ramp = vec![0.0f32, 1.0, 2.0, 3.0, 4.0];
    let resampled = resample_to_16k(&ramp, src_rate);

    // Target length: 5 * (16000 / 8000) = 10
    assert_eq!(resampled.len(), 10);

    // Exact interpolation verification:
    // Even indices 2*k correspond to exact original points
    assert!((resampled[0] - 0.0).abs() < 1e-5);
    assert!((resampled[2] - 1.0).abs() < 1e-5);
    assert!((resampled[4] - 2.0).abs() < 1e-5);
    assert!((resampled[6] - 3.0).abs() < 1e-5);
    assert!((resampled[8] - 4.0).abs() < 1e-5);

    // Odd indices 2*k + 1 correspond to linear midpoint
    assert!((resampled[1] - 0.5).abs() < 1e-5);
    assert!((resampled[3] - 1.5).abs() < 1e-5);
    assert!((resampled[5] - 2.5).abs() < 1e-5);
    assert!((resampled[7] - 3.5).abs() < 1e-5);

    // Upsampling 1.0s of 400Hz sine wave
    let n_samples = 8000;
    let freq = 400.0f32;
    let mut sine_8k = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let t = i as f32 / src_rate as f32;
        sine_8k.push((2.0 * PI * freq * t).sin());
    }

    let upsampled = resample_to_16k(&sine_8k, src_rate);
    assert_eq!(upsampled.len(), 16000);

    // Peak amplitude must remain bounded within [-1.0, 1.0]
    for &s in &upsampled {
        assert!(
            (-1.0001..=1.0001).contains(&s),
            "Sample out of bounds: {}",
            s
        );
    }

    // Energy / RMS preservation check: RMS of pure sine is ~0.7071
    let rms_8k: f32 = (sine_8k.iter().map(|&x| x * x).sum::<f32>() / n_samples as f32).sqrt();
    let rms_16k: f32 = (upsampled.iter().map(|&x| x * x).sum::<f32>() / 16000.0).sqrt();
    assert!(
        (rms_8k - rms_16k).abs() < 0.05,
        "RMS mismatch: 8k={}, 16k={}",
        rms_8k,
        rms_16k
    );
}

#[test]
fn test_48k_downsampling_to_16k() {
    let src_rate = 48000;

    // 1. Constant amplitude downsampling
    let constant_val = 0.65f32;
    let const_samples = vec![constant_val; 4800]; // 0.1s
    let downsampled = resample_to_16k(&const_samples, src_rate);
    assert_eq!(downsampled.len(), 1600); // 4800 / 3
    for (i, &s) in downsampled.iter().enumerate() {
        assert!(
            (s - constant_val).abs() < 1e-5,
            "Sample at {} deviates: expected {}, got {}",
            i,
            constant_val,
            s
        );
    }

    // 2. Sine wave downsampling from 48k to 16k
    let n_samples = 48000; // 1s
    let freq = 500.0f32;
    let mut sine_48k = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let t = i as f32 / src_rate as f32;
        sine_48k.push((2.0 * PI * freq * t).sin());
    }

    let downsampled_sine = resample_to_16k(&sine_48k, src_rate);
    assert_eq!(downsampled_sine.len(), 16000);

    // Verify peak amplitude
    for &s in &downsampled_sine {
        assert!((-1.0001..=1.0001).contains(&s), "Out of bounds: {}", s);
    }

    // Verify RMS preservation
    let rms_48k: f32 = (sine_48k.iter().map(|&x| x * x).sum::<f32>() / n_samples as f32).sqrt();
    let rms_16k: f32 = (downsampled_sine.iter().map(|&x| x * x).sum::<f32>() / 16000.0).sqrt();
    assert!(
        (rms_48k - rms_16k).abs() < 0.05,
        "RMS mismatch: 48k={}, 16k={}",
        rms_48k,
        rms_16k
    );

    // 3. Fractional 44.1 kHz to 16 kHz
    let samples_44k = vec![0.33f32; 44100];
    let resampled_44k = resample_to_16k(&samples_44k, 44100);
    assert_eq!(resampled_44k.len(), 16000);
}

#[test]
fn test_extreme_audio_amplitudes() {
    let filters = make_dummy_filters(N_MELS);

    // 1. Absolute silence (all 0.0)
    let silence = vec![0.0f32; 16000];
    let mel_silence = compute_log_mel_spectrogram(&silence, &filters, N_MELS);
    for &val in &mel_silence.data {
        assert!(val.is_finite(), "Silence mel data contains NaN or Inf");
        // Silence is clamped to normalized floor (-1.5)
        assert!(
            (-1.51..=-1.49).contains(&val),
            "Expected silence to normalize to -1.5, got {}",
            val
        );
    }

    // 2. Near 0.0 amplitudes (sub-audible signals)
    for &tiny_amp in &[1e-12f32, 1e-8, 1e-5] {
        let tiny_audio: Vec<f32> = (0..16000)
            .map(|i| tiny_amp * ((i as f32) * 0.1).sin())
            .collect();
        let mel_tiny = compute_log_mel_spectrogram(&tiny_audio, &filters, N_MELS);
        for &val in &mel_tiny.data {
            assert!(
                val.is_finite(),
                "Near-zero amplitude {} produced non-finite mel: {}",
                tiny_amp,
                val
            );
        }
    }

    // 3. Maximum scale +/- 1.0 (square wave)
    let square_wave: Vec<f32> = (0..16000)
        .map(|i| if (i / 80) % 2 == 0 { 1.0f32 } else { -1.0f32 })
        .collect();
    let mel_square = compute_log_mel_spectrogram(&square_wave, &filters, N_MELS);
    for &val in &mel_square.data {
        assert!(
            val.is_finite(),
            "Square wave produced non-finite mel: {}",
            val
        );
        // Normalized log-mel values should stay within reasonable bounds
        assert!(
            (-2.0..=3.0).contains(&val),
            "Normalized value {} outside expected range",
            val
        );
    }

    // 4. Full scale sine wave (+/- 1.0 peak)
    let full_sine: Vec<f32> = (0..16000)
        .map(|i| (2.0 * PI * 1000.0 * (i as f32 / 16000.0)).sin())
        .collect();
    let mel_sine = compute_log_mel_spectrogram(&full_sine, &filters, N_MELS);
    for &val in &mel_sine.data {
        assert!(val.is_finite(), "Full sine produced non-finite mel");
    }
}

#[test]
fn test_stereo_to_mono_channel_averaging() {
    // 1. 16-bit PCM Stereo WAV
    // Left: +16384 (0.5), Right: -16384 (-0.5) -> Average = 0.0
    let mut pcm_stereo_data = Vec::new();
    let num_frames = 100;
    for _ in 0..num_frames {
        pcm_stereo_data.write_i16::<LittleEndian>(16384).unwrap(); // L
        pcm_stereo_data.write_i16::<LittleEndian>(-16384).unwrap(); // R
    }

    let pcm_wav = create_wav_bytes(2, WHISPER_SAMPLE_RATE, 16, 1, &pcm_stereo_data);
    let audio = WavAudio::load_from_bytes(&pcm_wav).expect("Load PCM stereo");
    assert_eq!(audio.channels, 2);
    assert_eq!(audio.samples.len(), num_frames);
    for (i, &s) in audio.samples.iter().enumerate() {
        assert!(
            s.abs() < 1e-4,
            "Frame {} expected ~0.0 from stereo cancellation, got {}",
            i,
            s
        );
    }

    // Asymmetric Left/Right averaging:
    // Left: +32768 (~1.0), Right: +16384 (~0.5) -> Average = 0.75
    let mut pcm_stereo_data2 = Vec::new();
    for _ in 0..num_frames {
        pcm_stereo_data2.write_i16::<LittleEndian>(32767).unwrap(); // L: 32767/32768 = ~0.99997
        pcm_stereo_data2.write_i16::<LittleEndian>(16384).unwrap(); // R: 16384/32768 = 0.5
    }
    let pcm_wav2 = create_wav_bytes(2, WHISPER_SAMPLE_RATE, 16, 1, &pcm_stereo_data2);
    let audio2 = WavAudio::load_from_bytes(&pcm_wav2).expect("Load PCM stereo 2");
    let expected = (32767.0 / 32768.0 + 16384.0 / 32768.0) / 2.0;
    for (i, &s) in audio2.samples.iter().enumerate() {
        assert!(
            (s - expected).abs() < 1e-4,
            "Frame {}: expected {}, got {}",
            i,
            expected,
            s
        );
    }

    // 2. 32-bit Float Stereo WAV
    // Left: 0.8f32, Right: -0.2f32 -> Average = 0.3f32
    let mut float_stereo_data = Vec::new();
    for _ in 0..num_frames {
        float_stereo_data.write_f32::<LittleEndian>(0.8).unwrap();
        float_stereo_data.write_f32::<LittleEndian>(-0.2).unwrap();
    }
    let float_wav = create_wav_bytes(2, WHISPER_SAMPLE_RATE, 32, 3, &float_stereo_data);
    let audio_float = WavAudio::load_from_bytes(&float_wav).expect("Load Float stereo");
    assert_eq!(audio_float.channels, 2);
    assert_eq!(audio_float.samples.len(), num_frames);
    for (i, &s) in audio_float.samples.iter().enumerate() {
        assert!(
            (s - 0.3).abs() < 1e-5,
            "Frame {}: expected 0.3, got {}",
            i,
            s
        );
    }

    // 3. Stereo Phase Cancellation (perfect inverse signals)
    let mut phase_cancel_data = Vec::new();
    let n_cancel_frames = 200;
    for i in 0..n_cancel_frames {
        let val = ((i as f32) * 0.1).sin();
        phase_cancel_data.write_f32::<LittleEndian>(val).unwrap(); // Left
        phase_cancel_data.write_f32::<LittleEndian>(-val).unwrap(); // Right = -Left
    }
    let cancel_wav = create_wav_bytes(2, WHISPER_SAMPLE_RATE, 32, 3, &phase_cancel_data);
    let cancel_audio = WavAudio::load_from_bytes(&cancel_wav).expect("Load cancel audio");
    assert_eq!(cancel_audio.samples.len(), n_cancel_frames);
    for (i, &s) in cancel_audio.samples.iter().enumerate() {
        assert!(
            s.abs() < 1e-6,
            "Frame {}: expected 0.0 cancellation, got {}",
            i,
            s
        );
    }
}

#[test]
fn test_short_audio_slices() {
    let filters = make_dummy_filters(N_MELS);

    // Audio shorter than 1 FFT frame (400 samples)
    for &length in &[1, 10, 50, 100, 200, 399] {
        let short_audio = vec![0.5f32; length];
        let resampled = resample_to_16k(&short_audio, 8000);
        assert_eq!(resampled.len(), length * 2);

        let mel = compute_log_mel_spectrogram(&short_audio, &filters, N_MELS);
        assert_eq!(mel.n_mel, N_MELS);
        assert!(mel.n_len > 0);
        for &v in &mel.data {
            assert!(
                v.is_finite(),
                "Length {} produced non-finite mel: {}",
                length,
                v
            );
        }
    }
}
