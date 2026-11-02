use whisper_cli::get_model_download_url;

#[test]
fn test_get_model_download_url() {
    let tiny_en = get_model_download_url("tiny.en").expect("url for tiny.en");
    assert_eq!(
        tiny_en,
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin"
    );

    let base = get_model_download_url("ggml-base.bin").expect("url for ggml-base.bin");
    assert_eq!(
        base,
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin"
    );

    let turbo = get_model_download_url("large-v3-turbo").expect("url for large-v3-turbo");
    assert_eq!(
        turbo,
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin"
    );

    let invalid = get_model_download_url("non_existent_model_xyz");
    assert!(invalid.is_err());
}
