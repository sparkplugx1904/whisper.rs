use std::path::{Path, PathBuf};

/// Map model identifier to official Hugging Face repository download URL.
pub fn get_model_download_url(model: &str) -> Result<String, String> {
    let clean = model.trim().to_lowercase();
    let name = clean
        .strip_prefix("ggml-")
        .unwrap_or(&clean)
        .strip_suffix(".bin")
        .unwrap_or(&clean);

    let valid_models = [
        "tiny.en",
        "tiny",
        "tiny-q5_1",
        "tiny.en-q5_1",
        "tiny-q8_0",
        "tiny.en-q8_0",
        "base.en",
        "base",
        "base-q5_1",
        "base.en-q5_1",
        "base-q8_0",
        "base.en-q8_0",
        "small.en",
        "small",
        "small-q5_1",
        "small.en-q5_1",
        "small-q8_0",
        "small.en-q8_0",
        "medium.en",
        "medium",
        "medium-q5_0",
        "medium.en-q5_0",
        "medium-q8_0",
        "medium.en-q8_0",
        "large-v1",
        "large-v2",
        "large-v2-q5_0",
        "large-v2-q8_0",
        "large-v3",
        "large-v3-q5_0",
        "large-v3-turbo",
        "large-v3-turbo-q5_0",
        "large-v3-turbo-q8_0",
    ];

    if valid_models.contains(&name) {
        Ok(format!(
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-{}.bin",
            name
        ))
    } else {
        Err(format!(
            "Unknown model '{}'. Supported models: {}",
            model,
            valid_models.join(", ")
        ))
    }
}

/// Download official GGML model directly to destination directory.
pub fn download_model(model: &str, dest_dir: &Path) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let url = get_model_download_url(model)?;
    let clean = model.trim().to_lowercase();
    let file_name = format!(
        "ggml-{}.bin",
        clean
            .strip_prefix("ggml-")
            .unwrap_or(&clean)
            .strip_suffix(".bin")
            .unwrap_or(&clean)
    );
    let dest_path = dest_dir.join(&file_name);

    if dest_path.exists() {
        println!(
            "Model already exists at '{}', skipping download.",
            dest_path.display()
        );
        return Ok(dest_path);
    }

    std::fs::create_dir_all(dest_dir)?;
    println!("Downloading '{}' from '{}'...", model, url);
    println!("Destination: '{}'", dest_path.display());

    let status = std::process::Command::new("curl")
        .arg("-L")
        .arg("--progress-bar")
        .arg("-o")
        .arg(&dest_path)
        .arg(&url)
        .status()?;

    if !status.success() {
        return Err(format!("Download failed with exit code: {}", status).into());
    }

    println!(
        "\nModel successfully downloaded to '{}'!",
        dest_path.display()
    );
    Ok(dest_path)
}
