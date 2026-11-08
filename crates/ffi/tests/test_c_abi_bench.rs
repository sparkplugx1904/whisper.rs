use std::ffi::CStr;
use whisper::*;

#[test]
fn test_bench_memcpy_str_returns_report() {
    let ptr = whisper_bench_memcpy_str(0);
    assert!(!ptr.is_null());
    let report = unsafe { CStr::from_ptr(ptr) }
        .to_str()
        .expect("memcpy report is valid UTF-8");
    assert!(report.contains("memcpy:"), "report: {report}");
    assert!(report.contains("sum:"), "report: {report}");
    assert!(report.contains("GB/s"), "report: {report}");

    // The zero-thread path avoids the 1 GB buffers and must still succeed.
    assert_eq!(whisper_bench_memcpy(0), 0);
}

#[test]
fn test_print_system_info_reports_detected_features() {
    let ptr = whisper_print_system_info();
    assert!(!ptr.is_null());
    let info = unsafe { CStr::from_ptr(ptr) }
        .to_str()
        .expect("system info is valid UTF-8");
    assert!(info.starts_with("WHISPER : "), "info: {info}");

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        let avx = std::arch::is_x86_feature_detected!("avx") as u8;
        let avx2 = std::arch::is_x86_feature_detected!("avx2") as u8;
        let fma = std::arch::is_x86_feature_detected!("fma") as u8;
        assert!(info.contains(&format!("AVX = {avx}")), "info: {info}");
        assert!(info.contains(&format!("AVX2 = {avx2}")), "info: {info}");
        assert!(info.contains(&format!("FMA = {fma}")), "info: {info}");
        assert!(info.contains("NEON = 0"), "info: {info}");
    }

    #[cfg(target_arch = "aarch64")]
    {
        assert!(info.contains("AVX = 0"), "info: {info}");
        assert!(info.contains("AVX2 = 0"), "info: {info}");
        assert!(info.contains("FMA = 0"), "info: {info}");
        assert!(info.contains("NEON = 1"), "info: {info}");
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
    {
        assert!(info.contains("AVX = 0"), "info: {info}");
        assert!(info.contains("AVX2 = 0"), "info: {info}");
        assert!(info.contains("FMA = 0"), "info: {info}");
        assert!(info.contains("NEON = 0"), "info: {info}");
    }
}

#[test]
#[ignore = "runs the full whisper.cpp matrix-multiply benchmark; use --release"]
fn bench_ggml_mul_mat_full_report() {
    let ptr = whisper_bench_ggml_mul_mat_str(2);
    assert!(!ptr.is_null());
    let report = unsafe { CStr::from_ptr(ptr) }
        .to_str()
        .expect("matmul report is valid UTF-8");
    eprintln!("{report}");

    let size_lines: Vec<&str> = report.lines().filter(|line| line.contains(" x ")).collect();
    assert_eq!(size_lines.len(), 7 * 3);

    for type_name in ["Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "F16", "F32"] {
        assert!(report.contains(type_name), "missing {type_name}: {report}");
    }

    for line in size_lines {
        assert!(line.contains("GFLOPS"), "line: {line}");
        assert!(line.contains("runs)"), "line: {line}");
        assert!(!line.contains("(  0 runs)"), "line: {line}");
    }
}
