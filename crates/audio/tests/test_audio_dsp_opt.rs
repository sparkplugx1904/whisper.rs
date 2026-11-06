use std::f32::consts::PI;
use whisper_audio::{
    dot_f32, dot_f32_scalar, hann_window, hann_window_400, high_pass_filter, high_pass_filtered,
    project_mel_frame, resample_to_16k, PolyphaseSincFilter, DEFAULT_HIGH_PASS_CUTOFF_HZ,
    HANN_WINDOW_400, N_MELS,
};

#[test]
fn test_hann_window_cache_and_properties() {
    let cached = hann_window_400();
    assert_eq!(cached.len(), 400);

    // Identity with static
    assert_eq!(cached.as_ptr(), HANN_WINDOW_400.as_ptr());

    // Value properties
    assert_eq!(cached[0], 0.0);
    for i in 1..200 {
        let diff = (cached[i] - cached[400 - i]).abs();
        assert!(diff < 1e-6, "Asymmetry at {}: {}", i, diff);
    }
    assert!((cached[200] - 1.0).abs() < 1e-5);

    // Consistency with hann_window(400)
    let fetched = hann_window(400);
    assert_eq!(fetched, cached.to_vec());

    // Other sizes
    let w64 = hann_window(64);
    assert_eq!(w64.len(), 64);
    assert_eq!(w64[0], 0.0);
}

#[test]
fn test_vectorized_dot_product_201_bins() {
    let n_bins = 201;
    let mut power = vec![0.0f32; n_bins];
    let mut filter_row = vec![0.0f32; n_bins];

    for i in 0..n_bins {
        power[i] = (i as f32 * 0.25).sin().abs();
        filter_row[i] = (i as f32 * 0.13).cos().abs() * 0.05;
    }

    let scalar_dot = dot_f32_scalar(&power, &filter_row);
    let simd_dot = dot_f32(&power, &filter_row);

    assert!(
        (scalar_dot - simd_dot).abs() < 1e-5,
        "Mismatch between scalar ({}) and SIMD ({}) dot products",
        scalar_dot,
        simd_dot
    );

    // Test projection across 80 mel bands
    let n_mel = N_MELS;
    let filters = vec![0.02f32; n_mel * n_bins];
    let mut frame_out = vec![0.0f32; n_mel];

    project_mel_frame(&power, &filters, n_mel, &mut frame_out);

    let expected_dot: f32 = power.iter().map(|&p| p * 0.02).sum();
    let expected_log = (expected_dot.max(1e-10)).log10();

    for (idx, &val) in frame_out.iter().enumerate() {
        assert!(
            (val - expected_log).abs() < 1e-5,
            "Mel band {} value {} deviates from expected {}",
            idx,
            val,
            expected_log
        );
    }
}

#[test]
fn test_polyphase_resampler_48k_and_44k() {
    // 1. 48 kHz downsampling
    let samples_48k = vec![0.85f32; 9600]; // 0.2s at 48k
    let resampled_48k = resample_to_16k(&samples_48k, 48000);
    assert_eq!(resampled_48k.len(), 3200);

    for (i, &s) in resampled_48k.iter().enumerate() {
        assert!(
            (s - 0.85).abs() < 1e-5,
            "48k downsample sample {} deviated: {}",
            i,
            s
        );
    }

    // 2. 44.1 kHz downsampling
    let samples_44k = vec![0.42f32; 44100];
    let resampled_44k = resample_to_16k(&samples_44k, 44100);
    assert_eq!(resampled_44k.len(), 16000);

    for (i, &s) in resampled_44k.iter().enumerate() {
        assert!(
            (s - 0.42).abs() < 1e-5,
            "44.1k downsample sample {} deviated: {}",
            i,
            s
        );
    }

    // 3. Custom Polyphase filter instance
    let filter = PolyphaseSincFilter::new(48000, 16000);
    assert_eq!(filter.src_rate(), 48000);
    assert_eq!(filter.dst_rate(), 16000);
    assert_eq!(filter.num_phases(), 512);

    let custom_resampled = filter.resample(&samples_48k);
    assert_eq!(custom_resampled, resampled_48k);
}

#[test]
fn test_polyphase_resampler_sine_fidelity() {
    let src_rate = 48000;
    let n_samples = 48000;
    let freq = 1000.0f32;

    let mut sine_48k = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let t = i as f32 / src_rate as f32;
        sine_48k.push((2.0 * PI * freq * t).sin());
    }

    let downsampled = resample_to_16k(&sine_48k, src_rate);
    assert_eq!(downsampled.len(), 16000);

    // RMS preservation
    let rms_in: f32 = (sine_48k.iter().map(|&x| x * x).sum::<f32>() / n_samples as f32).sqrt();
    let rms_out: f32 = (downsampled.iter().map(|&x| x * x).sum::<f32>() / 16000.0).sqrt();

    assert!(
        (rms_in - rms_out).abs() < 0.05,
        "RMS mismatch: in={}, out={}",
        rms_in,
        rms_out
    );
}

#[test]
fn test_high_pass_filter_dc_and_frequencies() {
    assert_eq!(DEFAULT_HIGH_PASS_CUTOFF_HZ, 100.0);

    // In-place vs out-of-place equality
    let mut data1 = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let data_copy = data1.clone();
    high_pass_filter(&mut data1, 100.0, 16000.0);
    let data2 = high_pass_filtered(&data_copy, 100.0, 16000.0);
    assert_eq!(data1, data2);

    // DC offset attenuation
    let mut dc = vec![2.5f32; 16000];
    high_pass_filter(&mut dc, 100.0, 16000.0);
    assert_eq!(dc[0], 2.5); // First sample
                            // After 1 second, DC is heavily attenuated (< 0.1)
    assert!(dc[15999].abs() < 0.1);

    // Frequency separation: 20 Hz vs 2000 Hz
    let sr = 16000.0;
    let count = 16000;
    let mut sig_20 = Vec::with_capacity(count);
    let mut sig_2000 = Vec::with_capacity(count);

    for i in 0..count {
        let t = i as f32 / sr;
        sig_20.push((2.0 * PI * 20.0 * t).sin());
        sig_2000.push((2.0 * PI * 2000.0 * t).sin());
    }

    let out_20 = high_pass_filtered(&sig_20, 100.0, sr);
    let out_2000 = high_pass_filtered(&sig_2000, 100.0, sr);

    let half = count / 2;
    let rms_20: f32 = (out_20[half..].iter().map(|&x| x * x).sum::<f32>() / half as f32).sqrt();
    let rms_2000: f32 = (out_2000[half..].iter().map(|&x| x * x).sum::<f32>() / half as f32).sqrt();

    assert!(
        rms_20 < rms_2000 * 0.35,
        "20 Hz should be attenuated much more than 2000 Hz: rms_20={}, rms_2000={}",
        rms_20,
        rms_2000
    );
}
