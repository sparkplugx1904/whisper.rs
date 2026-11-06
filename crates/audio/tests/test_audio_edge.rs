use byteorder::{LittleEndian, WriteBytesExt};
use std::f32::consts::PI;
use whisper_audio::{
    compute_log_mel_spectrogram, resample_to_16k, AudioError, WavAudio,
    KSDATAFORMAT_SUBTYPE_IEEE_FLOAT, KSDATAFORMAT_SUBTYPE_PCM, N_MELS, WAVE_FORMAT_EXTENSIBLE,
    WHISPER_SAMPLE_RATE,
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

/// Helper to create a well-formed WAVE_FORMAT_EXTENSIBLE in-memory byte buffer.
fn create_extensible_wav_bytes(
    channels: u16,
    sample_rate: u32,
    bits_per_sample: u16,
    sub_format: &[u8; 16],
    data: &[u8],
) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"RIFF");
    let fmt_chunk_size = 40u32;
    let file_size = 12u32 + (8u32 + fmt_chunk_size) + (8u32 + data.len() as u32);
    out.extend_from_slice(&(file_size - 8).to_le_bytes());
    out.extend_from_slice(b"WAVE");

    // fmt chunk (WAVE_FORMAT_EXTENSIBLE)
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&fmt_chunk_size.to_le_bytes());
    out.extend_from_slice(&WAVE_FORMAT_EXTENSIBLE.to_le_bytes());
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    let byte_rate = sample_rate * (channels as u32) * (bits_per_sample as u32 / 8);
    out.extend_from_slice(&byte_rate.to_le_bytes());
    let block_align = channels * (bits_per_sample / 8);
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&bits_per_sample.to_le_bytes());
    let cb_size = 22u16;
    out.extend_from_slice(&cb_size.to_le_bytes());
    let valid_bits = bits_per_sample;
    out.extend_from_slice(&valid_bits.to_le_bytes());
    let channel_mask = if channels == 2 { 0x3u32 } else { 0x4u32 };
    out.extend_from_slice(&channel_mask.to_le_bytes());
    out.extend_from_slice(sub_format);

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

#[test]
fn test_sub_200_samples_audio_handling() {
    let filters = make_dummy_filters(N_MELS);

    // Sub-40 samples must have n_len_org == 0
    for &n in &[0, 1, 10, 20, 39] {
        let audio = vec![0.1f32; n];
        let mel = compute_log_mel_spectrogram(&audio, &filters, N_MELS);
        assert_eq!(mel.n_len_org, 0, "n_len_org for {} samples must be 0", n);
    }

    // 40 <= n < 200 samples must have n_len_org >= 1 (specifically 1)
    for &n in &[40, 41, 50, 100, 150, 199] {
        let audio = vec![0.1f32; n];
        let mel = compute_log_mel_spectrogram(&audio, &filters, N_MELS);
        assert!(
            mel.n_len_org >= 1,
            "n_len_org for {} samples must be >= 1, got {}",
            n,
            mel.n_len_org
        );
        assert_eq!(
            mel.n_len_org, 1,
            "n_len_org for {} samples must be 1, got {}",
            n, mel.n_len_org
        );
    }

    // n >= 200 samples: matches 1 + (n + stage_2_pad - frame_size) / frame_step
    for &n in &[200, 359, 360, 400, 520] {
        let audio = vec![0.1f32; n];
        let mel = compute_log_mel_spectrogram(&audio, &filters, N_MELS);
        let expected = 1 + (n + 200 - 400) / 160;
        assert_eq!(
            mel.n_len_org, expected,
            "n_len_org for {} samples: expected {}, got {}",
            n, expected, mel.n_len_org
        );
    }
}

#[test]
fn test_8bit_unsigned_pcm_wav() {
    // 8-bit unsigned PCM: 128 is center/silence, 255 is ~+1.0, 0 is -1.0
    let mono_data = vec![128u8, 255u8, 0u8, 192u8, 64u8];
    let wav = create_wav_bytes(1, 16000, 8, 1, &mono_data);
    let audio = WavAudio::load_from_bytes(&wav).expect("Load 8-bit mono WAV");
    assert_eq!(audio.sample_rate, 16000);
    assert_eq!(audio.channels, 1);
    assert_eq!(audio.samples.len(), 5);

    assert!((audio.samples[0] - 0.0).abs() < 1e-5);
    assert!((audio.samples[1] - (127.0 / 128.0)).abs() < 1e-5);
    assert!((audio.samples[2] - (-1.0)).abs() < 1e-5);
    assert!((audio.samples[3] - 0.5).abs() < 1e-5);
    assert!((audio.samples[4] - (-0.5)).abs() < 1e-5);

    // 8-bit stereo: Left 192 (+0.5), Right 64 (-0.5) -> avg 0.0
    let stereo_data = vec![192u8, 64u8, 255u8, 128u8];
    let stereo_wav = create_wav_bytes(2, 16000, 8, 1, &stereo_data);
    let audio_st = WavAudio::load_from_bytes(&stereo_wav).expect("Load 8-bit stereo WAV");
    assert_eq!(audio_st.channels, 2);
    assert_eq!(audio_st.samples.len(), 2);
    assert!(audio_st.samples[0].abs() < 1e-5);
    let expected_frame1 = ((127.0 / 128.0) + 0.0) / 2.0;
    assert!((audio_st.samples[1] - expected_frame1).abs() < 1e-5);
}

#[test]
fn test_24bit_signed_pcm_wav() {
    // 24-bit little endian:
    // 0: [0x00, 0x00, 0x00] -> 0.0
    // +8388607: [0xFF, 0xFF, 0x7F] -> 8388607 / 8388608.0
    // -8388608: [0x00, 0x00, 0x80] -> -1.0
    // +4194304: [0x00, 0x00, 0x40] -> 0.5
    // -4194304: [0x00, 0x00, 0xC0] -> -0.5
    let mut data = Vec::new();
    data.extend_from_slice(&[0x00, 0x00, 0x00]);
    data.extend_from_slice(&[0xFF, 0xFF, 0x7F]);
    data.extend_from_slice(&[0x00, 0x00, 0x80]);
    data.extend_from_slice(&[0x00, 0x00, 0x40]);
    data.extend_from_slice(&[0x00, 0x00, 0xC0]);

    let wav = create_wav_bytes(1, 16000, 24, 1, &data);
    let audio = WavAudio::load_from_bytes(&wav).expect("Load 24-bit mono WAV");
    assert_eq!(audio.sample_rate, 16000);
    assert_eq!(audio.channels, 1);
    assert_eq!(audio.samples.len(), 5);

    assert!((audio.samples[0] - 0.0).abs() < 1e-6);
    assert!((audio.samples[1] - (8388607.0 / 8388608.0)).abs() < 1e-6);
    assert!((audio.samples[2] - (-1.0)).abs() < 1e-6);
    assert!((audio.samples[3] - 0.5).abs() < 1e-6);
    assert!((audio.samples[4] - (-0.5)).abs() < 1e-6);

    // 24-bit stereo
    let mut stereo_data = Vec::new();
    // L: +4194304 (+0.5), R: -4194304 (-0.5) -> avg 0.0
    stereo_data.extend_from_slice(&[0x00, 0x00, 0x40]);
    stereo_data.extend_from_slice(&[0x00, 0x00, 0xC0]);
    let stereo_wav = create_wav_bytes(2, 16000, 24, 1, &stereo_data);
    let audio_st = WavAudio::load_from_bytes(&stereo_wav).expect("Load 24-bit stereo WAV");
    assert_eq!(audio_st.channels, 2);
    assert_eq!(audio_st.samples.len(), 1);
    assert!(audio_st.samples[0].abs() < 1e-6);
}

#[test]
fn test_32bit_signed_pcm_wav() {
    let mut data = Vec::new();
    data.write_i32::<LittleEndian>(0).unwrap();
    data.write_i32::<LittleEndian>(1073741824).unwrap();
    data.write_i32::<LittleEndian>(-1073741824).unwrap();
    data.write_i32::<LittleEndian>(2147483647).unwrap();
    data.write_i32::<LittleEndian>(i32::MIN).unwrap();

    let wav = create_wav_bytes(1, 16000, 32, 1, &data);
    let audio = WavAudio::load_from_bytes(&wav).expect("Load 32-bit signed PCM WAV");
    assert_eq!(audio.sample_rate, 16000);
    assert_eq!(audio.channels, 1);
    assert_eq!(audio.samples.len(), 5);

    assert!((audio.samples[0] - 0.0).abs() < 1e-6);
    assert!((audio.samples[1] - 0.5).abs() < 1e-6);
    assert!((audio.samples[2] - (-0.5)).abs() < 1e-6);
    assert!((audio.samples[3] - (2147483647.0 / 2147483648.0)).abs() < 1e-6);
    assert!((audio.samples[4] - (-1.0)).abs() < 1e-6);
}

#[test]
fn test_wave_format_extensible_pcm_and_float() {
    // 1. 16-bit PCM Extensible
    let mut pcm16_data = Vec::new();
    pcm16_data.write_i16::<LittleEndian>(16384).unwrap();
    pcm16_data.write_i16::<LittleEndian>(-16384).unwrap();
    let wav16 = create_extensible_wav_bytes(1, 16000, 16, &KSDATAFORMAT_SUBTYPE_PCM, &pcm16_data);
    let a16 = WavAudio::load_from_bytes(&wav16).expect("Load 16-bit extensible PCM");
    assert_eq!(a16.samples.len(), 2);
    assert!((a16.samples[0] - 0.5).abs() < 1e-4);
    assert!((a16.samples[1] - (-0.5)).abs() < 1e-4);

    // 2. 24-bit PCM Extensible
    let mut pcm24_data = Vec::new();
    pcm24_data.extend_from_slice(&[0x00, 0x00, 0x40]); // +0.5
    pcm24_data.extend_from_slice(&[0x00, 0x00, 0xC0]); // -0.5
    let wav24 = create_extensible_wav_bytes(1, 16000, 24, &KSDATAFORMAT_SUBTYPE_PCM, &pcm24_data);
    let a24 = WavAudio::load_from_bytes(&wav24).expect("Load 24-bit extensible PCM");
    assert_eq!(a24.samples.len(), 2);
    assert!((a24.samples[0] - 0.5).abs() < 1e-6);
    assert!((a24.samples[1] - (-0.5)).abs() < 1e-6);

    // 3. 32-bit PCM Extensible
    let mut pcm32_data = Vec::new();
    pcm32_data.write_i32::<LittleEndian>(1073741824).unwrap();
    pcm32_data.write_i32::<LittleEndian>(-1073741824).unwrap();
    let wav32 = create_extensible_wav_bytes(1, 16000, 32, &KSDATAFORMAT_SUBTYPE_PCM, &pcm32_data);
    let a32 = WavAudio::load_from_bytes(&wav32).expect("Load 32-bit extensible PCM");
    assert_eq!(a32.samples.len(), 2);
    assert!((a32.samples[0] - 0.5).abs() < 1e-6);
    assert!((a32.samples[1] - (-0.5)).abs() < 1e-6);

    // 4. 32-bit IEEE Float Extensible
    let mut float32_data = Vec::new();
    float32_data.write_f32::<LittleEndian>(0.75).unwrap();
    float32_data.write_f32::<LittleEndian>(-0.25).unwrap();
    let wav_float = create_extensible_wav_bytes(
        1,
        16000,
        32,
        &KSDATAFORMAT_SUBTYPE_IEEE_FLOAT,
        &float32_data,
    );
    let a_float = WavAudio::load_from_bytes(&wav_float).expect("Load 32-bit extensible Float");
    assert_eq!(a_float.samples.len(), 2);
    assert!((a_float.samples[0] - 0.75).abs() < 1e-6);
    assert!((a_float.samples[1] - (-0.25)).abs() < 1e-6);
}

#[test]
fn test_wave_format_extensible_errors() {
    let unknown_guid = [0xFFu8; 16];
    let bad_guid_wav = create_extensible_wav_bytes(1, 16000, 16, &unknown_guid, &[0u8; 4]);
    let res = WavAudio::load_from_bytes(&bad_guid_wav);
    assert!(matches!(res, Err(AudioError::UnsupportedFormat(_))));
}
