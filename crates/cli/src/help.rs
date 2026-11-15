//! whisper.cpp-compatible command-line help.
//!
//! The parser accepts the upstream short spellings through
//! `normalize_cli_args`. Clap cannot render multi-character short flags such
//! as `-bs`, so the complete synopsis is kept here instead of relying on the
//! derive-generated help.

pub const HELP: &str = "\
usage: whisper-cli [options] file0 file1 ...

supported audio formats: flac, mp3, ogg, wav

options:
  -h,        --help                 [default] show this help message and exit
  -t N,      --threads N            [4      ] number of threads to use during computation
  -ot N,     --offset-t N           [0      ] time offset in milliseconds
  -d  N,     --duration N           [0      ] duration of audio to process in milliseconds
  -mc N,     --max-context N        [16384  ] maximum number of text context tokens to store
  -ml N,     --max-len N            [0      ] maximum segment length in characters
  -sow,      --split-on-word        [false  ] split on word rather than on token
  -bs N,     --beam-size N          [1      ] beam size for beam search (1 = greedy)
  -bo N,     --best-of N            [1      ] best-of width; >1 is rejected until implemented
  -ac N,     --audio-ctx N          [0      ] audio context size (0 - all)
  -wt,       --word-timestamps      [false  ] enable word-level timestamps
  -et N,     --entropy-thold N      [2.40   ] entropy threshold for decoder fail
  -lpt N,    --logprob-thold N      [-1.00  ] log probability threshold for decoder fail
  -nth N,    --no-speech-thold N    [0.60   ] no speech threshold
  -tp N,     --temperature N        [0.00   ] sampling temperature, between 0 and 1
  -tpi N,    --temperature-inc N    [0.20   ] temperature increment, between 0 and 1
  -nf,       --no-fallback          [false  ] do not use temperature fallback
  -tr,       --translate            [false  ] translate from source language to english
  -di,       --diarize              [false  ] stereo audio diarization
  -tdrz,     --tinydiarize          [false  ] enable tinydiarize (requires a tdrz model)
  -otxt,     --output-txt           [false  ] output result in a text file
  -ovtt,     --output-vtt           [false  ] output result in a vtt file
  -osrt,     --output-srt           [false  ] output result in a srt file
  -olrc,     --output-lrc           [false  ] output result in a lrc file
  -owts,     --output-words         [false  ] output script for generating karaoke video
  -fp,       --font-path            [system] path to a monospace font for karaoke video
  -ocsv,     --output-csv           [false  ] output result in a CSV file
  -oj,       --output-json          [false  ] output result in a JSON file
  -ojf,      --output-json-full     [false  ] include token-level data in JSON
  -of FNAME, --output-file FNAME    [       ] output file path (without file extension)
  -np,       --no-prints            [false  ] do not print anything other than the results
  -pp,       --print-progress       [false  ] print encoder-window progress
             --print-realtime       [true   ] print segments as they are decoded
  -pc,       --print-colors         [false  ] print tokens with ANSI color confidence
             --print-confidence     [false  ] print confidence text styles
  -ps,       --print-special        [false  ] print special tokens as text
  -nt,       --no-timestamps        [false  ] do not print timestamps
  -l LANG,   --language LANG        [auto   ] spoken language ('auto' for auto-detect)
  -dl,       --detect-language      [false  ] exit after automatically detecting language
             --prompt PROMPT        [       ] initial prompt (max n_text_ctx/2 tokens)
             --carry-initial-prompt [false  ] always prepend initial prompt
  -m FNAME,  --model FNAME          [       ] model path
  -f FNAME,  --file FNAME           [       ] input audio file path
  -sns,      --suppress-nst         [false  ] suppress non-speech tokens
             --suppress-regex REGEX [       ] regular expression matching tokens to suppress
             --grammar GRAMMAR      [       ] GBNF grammar source for constrained decoding
             --single-segment       [false  ] force a single output segment
             --max-initial-ts N     [1.0    ] maximum initial timestamp at step 0

Voice Activity Detection (VAD) options:
             --vad                           [false  ] enable Voice Activity Detection (VAD)
  -vm FNAME, --vad-model FNAME               [       ] VAD model path (Silero); omit for energy VAD
  -vt N,     --vad-threshold N               [0.50   ] VAD threshold for speech recognition
  -vspd N,   --vad-min-speech-duration-ms  N [250    ] VAD min speech duration (ms)
  -vsd N,    --vad-min-silence-duration-ms N [100    ] VAD min silence duration (ms)
  -vmsd N,   --vad-max-speech-duration-s   N [FLT_MAX] VAD max speech duration (auto-split longer)
  -vp N,     --vad-speech-pad-ms           N [300    ] VAD speech padding (ms)
  -vo N,     --vad-samples-overlap           [0.10   ] VAD samples overlap (seconds)
";
