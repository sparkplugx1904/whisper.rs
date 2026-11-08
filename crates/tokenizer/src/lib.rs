#![forbid(unsafe_code)]
//! Tokenizer (GPT-2 BPE & multilingual special tokens) for whisper.rs.

use thiserror::Error;
use whisper_model::Vocab;

#[derive(Error, Debug)]
pub enum TokenizerError {
    #[error("Token not found: {0}")]
    TokenNotFound(String),
    #[error("Invalid token id: {0}")]
    InvalidTokenId(i32),
    #[error("Decode error: {0}")]
    DecodeError(String),
}

#[derive(Clone)]
pub struct Tokenizer {
    pub vocab: Vocab,
}

impl Tokenizer {
    pub fn new(vocab: Vocab) -> Self {
        Self { vocab }
    }

    /// Decode a sequence of token IDs to a string.
    /// If `skip_special` is true, tokens >= token_eot are skipped.
    pub fn decode(&self, tokens: &[i32], skip_special: bool) -> String {
        let mut out = Vec::new();
        let eot = self.vocab.token_eot;

        for &tok in tokens {
            if skip_special && tok >= eot {
                continue;
            }
            if let Some(bytes) = self.vocab.id_to_token_bytes(tok) {
                out.extend_from_slice(bytes);
            }
        }
        String::from_utf8_lossy(&out).into_owned()
    }

    /// Encode input text into a sequence of BPE token IDs using greedy longest-match.
    ///
    /// This mirrors `tokenize()` in `whisper.cpp`: a byte-oriented split into
    /// GPT-2 words, followed by the longest vocabulary entry that fits at each
    /// byte offset. Bytes that cannot start a vocabulary entry are skipped, the
    /// same way the reference implementation advances by one byte after a miss.
    pub fn encode(&self, text: &str) -> Vec<i32> {
        self.encode_bytes(text.as_bytes())
    }

    /// Encode arbitrary bytes, for callers that receive non-UTF-8 input through
    /// the C ABI.
    pub fn encode_bytes(&self, text: &[u8]) -> Vec<i32> {
        let mut tokens = Vec::new();
        let mut pos = 0;

        while pos < text.len() {
            let end = match next_word(text, pos) {
                Some(end) => end,
                None => break,
            };
            self.encode_word(&text[pos..end], &mut tokens);
            pos = end;
        }

        tokens
    }

    fn encode_word(&self, word: &[u8], tokens: &mut Vec<i32>) {
        let mut i = 0;

        while i < word.len() {
            let mut j = word.len();
            let mut found = false;

            while j > i {
                if let Some(id) = self.vocab.token_to_id_bytes(&word[i..j]) {
                    tokens.push(id);
                    i = j;
                    found = true;
                    break;
                }
                j -= 1;
            }

            if !found {
                i += 1;
            }
        }
    }

    /// Find token id for an exact token string.
    pub fn token_to_id(&self, token: &str) -> Option<i32> {
        self.vocab.token_to_id(token)
    }

    /// Get token string by id.
    pub fn id_to_token(&self, id: i32) -> Option<&str> {
        self.vocab.id_to_token(id)
    }

    pub fn eot(&self) -> i32 {
        self.vocab.token_eot
    }

    pub fn sot(&self) -> i32 {
        self.vocab.token_sot
    }

    pub fn translate(&self) -> i32 {
        self.vocab.token_translate
    }

    pub fn transcribe(&self) -> i32 {
        self.vocab.token_transcribe
    }

    pub fn not(&self) -> i32 {
        self.vocab.token_not
    }

    pub fn beg(&self) -> i32 {
        self.vocab.token_beg
    }
}

/// Split point of the GPT-2 word regex used by `whisper.cpp`, starting at `pos`.
///
/// The reference pattern is
/// `'s|'t|'re|'ve|'m|'ll|'d| ?[[:alpha:]]+| ?[[:digit:]]+| ?[^\s[:alpha:][:digit:]]+|\s+(?!\S)|\s+`
/// evaluated on bytes in the C locale, so only ASCII letters and digits take
/// part in the alphanumeric alternatives. The regex has no look-around support
/// here, so the `\s+(?!\S)` branch is reproduced by stopping one byte before the
/// end of an interior whitespace run.
fn next_word(text: &[u8], pos: usize) -> Option<usize> {
    let mut start = pos;

    while start < text.len() {
        if let Some(end) = match_word_at(text, start) {
            return Some(end);
        }
        start += 1;
    }

    None
}

fn match_word_at(text: &[u8], start: usize) -> Option<usize> {
    const CONTRACTIONS: [&[u8]; 7] = [b"'s", b"'t", b"'re", b"'ve", b"'m", b"'ll", b"'d"];

    for contraction in CONTRACTIONS {
        if text[start..].starts_with(contraction) {
            return Some(start + contraction.len());
        }
    }

    // ` ?` attaches a single ASCII space to the following run, matching the
    // reference regex. A run of punctuation bytes includes UTF-8 continuation
    // bytes, which is what keeps multi-byte characters together.
    let word_start = if text[start] == b' ' && start + 1 < text.len() {
        start + 1
    } else {
        start
    };

    if word_start < text.len() {
        let byte = text[word_start];

        if byte.is_ascii_alphabetic() {
            return Some(run_end(text, word_start, |b| b.is_ascii_alphabetic()));
        }
        if byte.is_ascii_digit() {
            return Some(run_end(text, word_start, |b| b.is_ascii_digit()));
        }
        if !is_ascii_whitespace(byte) {
            return Some(run_end(text, word_start, |b| {
                !is_ascii_whitespace(b) && !b.is_ascii_alphabetic() && !b.is_ascii_digit()
            }));
        }
    }

    if is_ascii_whitespace(text[start]) {
        let end = run_end(text, start, is_ascii_whitespace);

        // `\s+(?!\S)` consumes an interior run except for its last byte; the
        // remaining whitespace is then attached to the next word by ` ?`.
        if end < text.len() && end - start >= 2 {
            return Some(end - 1);
        }

        return Some(end);
    }

    None
}

fn run_end(text: &[u8], start: usize, predicate: impl Fn(u8) -> bool) -> usize {
    let mut end = start;
    while end < text.len() && predicate(text[end]) {
        end += 1;
    }
    end
}

/// `\s` in the C locale used by the reference implementation.
fn is_ascii_whitespace(byte: u8) -> bool {
    matches!(byte, b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(text: &str) -> Vec<&str> {
        let bytes = text.as_bytes();
        let mut out = Vec::new();
        let mut pos = 0;

        while pos < bytes.len() {
            let end = next_word(bytes, pos).expect("word");
            out.push(std::str::from_utf8(&bytes[pos..end]).unwrap_or("<bytes>"));
            pos = end;
        }

        out
    }

    #[test]
    fn splits_gpt2_words_like_whisper_cpp() {
        assert_eq!(words("Hello, world!"), ["Hello", ",", " world", "!"]);
        assert_eq!(words("don't stop"), ["don", "'t", " stop"]);
        assert_eq!(words("I'll be there"), ["I", "'ll", " be", " there"]);
        assert_eq!(words("abc123def"), ["abc", "123", "def"]);
        assert_eq!(words("TESTING 123 !!!"), ["TESTING", " 123", " !!!"]);
    }

    #[test]
    fn whitespace_runs_keep_their_trailing_byte_for_the_next_word() {
        assert_eq!(words("  leading"), [" ", " leading"]);
        assert_eq!(words("   leading"), ["  ", " leading"]);
        assert_eq!(words("end   "), ["end", "   "]);
        assert_eq!(words("a\tb\nc"), ["a", "\t", "b", "\n", "c"]);
    }

    #[test]
    fn punctuation_and_multibyte_runs_stay_together() {
        assert_eq!(words("word—dash"), ["word", "—", "dash"]);
        assert_eq!(words("你好，世界"), ["你好，世界"]);
    }

    #[test]
    fn longest_match_skips_bytes_without_tokens() {
        let mut vocab = Vocab::new();
        let hello = vocab.add_token("Hello");
        let comma = vocab.add_token(",");
        let world = vocab.add_token(" world");
        let tokenizer = Tokenizer::new(vocab);

        // The exclamation mark has no token in this fixture, so it is skipped
        // exactly like the reference implementation.
        assert_eq!(tokenizer.encode("Hello, world!"), [hello, comma, world]);
    }
}
