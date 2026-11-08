#![forbid(unsafe_code)]

use byteorder::{LittleEndian, WriteBytesExt};
use clap::Parser;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use whisper_model::header::{GGML_FILE_MAGIC, GGML_QNT_VERSION_FACTOR};
use whisper_model::WhisperModel;
use whisper_tensor::DType;

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Quantize GGML models for whisper.rs (q4_0, q4_1, q5_0, q5_1, q8_0)"
)]
struct Args {
    /// Input model file path (.bin)
    #[arg(short = 'i', long = "input")]
    input: String,

    /// Output quantized model file path (.bin)
    #[arg(short = 'o', long = "output")]
    output: String,

    /// Quantization type: q4_0, q4_1, q5_0, q5_1, q8_0
    #[arg(short = 't', long = "type", default_value = "q8_0")]
    quant_type: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let target_dtype = match args.quant_type.to_lowercase().as_str() {
        "q4_0" => DType::Q4_0,
        "q4_1" => DType::Q4_1,
        "q5_0" => DType::Q5_0,
        "q5_1" => DType::Q5_1,
        "q8_0" => DType::Q8_0,
        other => {
            eprintln!("Unsupported quantization type '{}'", other);
            std::process::exit(1);
        }
    };

    let target_ftype = match target_dtype {
        DType::Q4_0 => 2,
        DType::Q4_1 => 3,
        DType::Q8_0 => 7,
        DType::Q5_0 => 8,
        DType::Q5_1 => 9,
        _ => 1,
    };

    println!("Loading input model from '{}'...", args.input);
    let model = WhisperModel::load_from_file(Path::new(&args.input))?;

    println!(
        "Quantizing model to {:?} (ftype={}) -> '{}'...",
        target_dtype, target_ftype, args.output
    );

    let out_file = File::create(&args.output)?;
    let mut writer = BufWriter::new(out_file);

    // 1. Write Magic
    writer.write_u32::<LittleEndian>(GGML_FILE_MAGIC)?;

    // 2. Write Hyperparameters with updated ftype (qntvr = 1)
    let hp = &model.hparams;
    let new_raw_ftype = GGML_QNT_VERSION_FACTOR + target_ftype;

    writer.write_i32::<LittleEndian>(hp.n_vocab)?;
    writer.write_i32::<LittleEndian>(hp.n_audio_ctx)?;
    writer.write_i32::<LittleEndian>(hp.n_audio_state)?;
    writer.write_i32::<LittleEndian>(hp.n_audio_head)?;
    writer.write_i32::<LittleEndian>(hp.n_audio_layer)?;
    writer.write_i32::<LittleEndian>(hp.n_text_ctx)?;
    writer.write_i32::<LittleEndian>(hp.n_text_state)?;
    writer.write_i32::<LittleEndian>(hp.n_text_head)?;
    writer.write_i32::<LittleEndian>(hp.n_text_layer)?;
    writer.write_i32::<LittleEndian>(hp.n_mels)?;
    writer.write_i32::<LittleEndian>(new_raw_ftype)?;

    // 3. Write Mel Filters
    writer.write_i32::<LittleEndian>(model.filters.n_mel as i32)?;
    writer.write_i32::<LittleEndian>(model.filters.n_fft as i32)?;
    for &f in &model.filters.data {
        writer.write_f32::<LittleEndian>(f)?;
    }

    // 4. Write Vocab
    let n_vocab_file = model.vocab.token_bytes.len().min(50257);
    writer.write_i32::<LittleEndian>(n_vocab_file as i32)?;
    for token in model.vocab.token_bytes.iter().take(n_vocab_file) {
        writer.write_u32::<LittleEndian>(token.len() as u32)?;
        writer.write_all(token)?;
    }

    // 5. Write Tensors
    let mut orig_size = 0usize;
    let mut quant_size = 0usize;

    let mut tensor_list: Vec<_> = model.tensors.values().collect();
    tensor_list.sort_by_key(|t| t.offset);

    for t in tensor_list {
        let should_quantize = t.n_dims >= 2
            && (t.name.ends_with(".weight") || t.name == "decoder.token_embedding.weight")
            && !t.name.contains("conv1")
            && !t.name.contains("conv2");

        let (new_dtype, new_bytes) = if should_quantize {
            let f32_data = model.get_tensor_f32(&t.name)?;
            let qbytes = target_dtype.quantize_from_f32(&f32_data)?;
            (target_dtype, qbytes)
        } else {
            let raw = model
                .get_tensor_raw_bytes(&t.name)
                .expect("raw bytes exist");
            (t.dtype, raw.to_vec())
        };

        orig_size += t.byte_size;
        quant_size += new_bytes.len();

        // Write tensor metadata
        writer.write_i32::<LittleEndian>(t.n_dims as i32)?;
        writer.write_i32::<LittleEndian>(t.name.len() as i32)?;
        writer.write_u32::<LittleEndian>(new_dtype as u32)?;

        for &dim in &t.ne {
            writer.write_i32::<LittleEndian>(dim as i32)?;
        }

        writer.write_all(t.name.as_bytes())?;
        writer.write_all(&new_bytes)?;
    }

    writer.flush()?;

    println!(
        "\nQuantization complete:\n  Original weight size:  {:.2} MB ({} bytes)\n  Quantized weight size: {:.2} MB ({} bytes)\n  Size reduction:        {:.1}%\n",
        orig_size as f64 / 1e6,
        orig_size,
        quant_size as f64 / 1e6,
        quant_size,
        (1.0 - (quant_size as f64 / orig_size as f64)) * 100.0
    );

    Ok(())
}
