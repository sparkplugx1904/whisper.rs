use crate::header::Hyperparameters;
use crate::ModelError;
use byteorder::{LittleEndian, ReadBytesExt};
use std::collections::HashMap;
use std::io::Read;

pub const LANGUAGES: &[(&str, &str)] = &[
    ("en", "english"),
    ("zh", "chinese"),
    ("de", "german"),
    ("es", "spanish"),
    ("ru", "russian"),
    ("ko", "korean"),
    ("fr", "french"),
    ("ja", "japanese"),
    ("pt", "portuguese"),
    ("tr", "turkish"),
    ("pl", "polish"),
    ("ca", "catalan"),
    ("nl", "dutch"),
    ("ar", "arabic"),
    ("sv", "swedish"),
    ("it", "italian"),
    ("id", "indonesian"),
    ("hi", "hindi"),
    ("fi", "finnish"),
    ("vi", "vietnamese"),
    ("he", "hebrew"),
    ("uk", "ukrainian"),
    ("el", "greek"),
    ("ms", "malay"),
    ("cs", "czech"),
    ("ro", "romanian"),
    ("da", "danish"),
    ("hu", "hungarian"),
    ("ta", "tamil"),
    ("no", "norwegian"),
    ("th", "thai"),
    ("ur", "urdu"),
    ("hr", "croatian"),
    ("bg", "bulgarian"),
    ("lt", "lithuanian"),
    ("la", "latin"),
    ("mi", "maori"),
    ("ml", "malayalam"),
    ("cy", "welsh"),
    ("sk", "slovak"),
    ("te", "telugu"),
    ("fa", "persian"),
    ("lv", "latvian"),
    ("bn", "bengali"),
    ("sr", "serbian"),
    ("az", "azerbaijani"),
    ("sl", "slovenian"),
    ("kn", "kannada"),
    ("et", "estonian"),
    ("mk", "macedonian"),
    ("br", "breton"),
    ("eu", "basque"),
    ("is", "icelandic"),
    ("hy", "armenian"),
    ("ne", "nepali"),
    ("mn", "mongolian"),
    ("bs", "bosnian"),
    ("kk", "kazakh"),
    ("sq", "albanian"),
    ("sw", "swahili"),
    ("gl", "galician"),
    ("mr", "marathi"),
    ("pa", "punjabi"),
    ("si", "sinhala"),
    ("km", "khmer"),
    ("sn", "shona"),
    ("yo", "yoruba"),
    ("so", "somali"),
    ("af", "afrikaans"),
    ("oc", "occitan"),
    ("ka", "georgian"),
    ("be", "belarusian"),
    ("tg", "tajik"),
    ("sd", "sindhi"),
    ("gu", "gujarati"),
    ("am", "amharic"),
    ("yi", "yiddish"),
    ("lo", "lao"),
    ("uz", "uzbek"),
    ("fo", "faroese"),
    ("ht", "haitian creole"),
    ("ps", "pashto"),
    ("tk", "turkmen"),
    ("nn", "nynorsk"),
    ("mt", "maltese"),
    ("sa", "sanskrit"),
    ("lb", "luxembourgish"),
    ("my", "myanmar"),
    ("bo", "tibetan"),
    ("tl", "tagalog"),
    ("mg", "malagasy"),
    ("as", "assamese"),
    ("tt", "tatar"),
    ("haw", "hawaiian"),
    ("ln", "lingala"),
    ("ha", "hausa"),
    ("ba", "bashkir"),
    ("jw", "javanese"),
    ("su", "sundanese"),
    ("yue", "cantonese"),
];

pub fn lang_str(id: usize) -> Option<&'static str> {
    LANGUAGES.get(id).map(|&(code, _)| code)
}

pub fn lang_str_full(id: usize) -> Option<&'static str> {
    LANGUAGES.get(id).map(|&(_, full)| full)
}

pub fn lang_id(code_or_name: &str) -> Option<usize> {
    for (i, &(c, name)) in LANGUAGES.iter().enumerate() {
        if c.eq_ignore_ascii_case(code_or_name) || name.eq_ignore_ascii_case(code_or_name) {
            return Some(i);
        }
    }
    None
}

#[derive(Debug, Clone)]
pub struct Vocab {
    pub n_vocab: usize,
    /// Exact token bytes, as stored in the GGML vocabulary. Token strings are
    /// not guaranteed to be valid UTF-8 (byte-level BPE fragments are not), so
    /// lookups must stay byte-oriented.
    pub token_to_id: HashMap<Vec<u8>, i32>,
    /// Lossy UTF-8 rendering for text-facing consumers.
    pub id_to_token: Vec<String>,
    /// Exact bytes for every token id.
    pub token_bytes: Vec<Vec<u8>>,

    pub token_eot: i32,
    pub token_sot: i32,
    pub token_translate: i32,
    pub token_transcribe: i32,
    pub token_solm: i32,
    pub token_prev: i32,
    pub token_nosp: i32,
    pub token_not: i32,
    pub token_beg: i32,
}

impl Default for Vocab {
    fn default() -> Self {
        Self {
            n_vocab: 0,
            token_to_id: HashMap::new(),
            id_to_token: Vec::new(),
            token_bytes: Vec::new(),
            token_eot: 50256,
            token_sot: 50257,
            token_translate: 50358,
            token_transcribe: 50359,
            token_solm: 50360,
            token_prev: 50361,
            token_nosp: 50362,
            token_not: 50363,
            token_beg: 50364,
        }
    }
}

impl Vocab {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_token(&mut self, token: impl Into<String>) -> i32 {
        self.add_token_bytes(token.into().into_bytes())
    }

    pub fn add_token_bytes(&mut self, token: Vec<u8>) -> i32 {
        let id = self.id_to_token.len() as i32;
        self.token_to_id.insert(token.clone(), id);
        self.id_to_token
            .push(String::from_utf8_lossy(&token).into_owned());
        self.token_bytes.push(token);
        self.n_vocab = self.id_to_token.len();
        id
    }
}

impl Vocab {
    pub fn read<R: Read>(reader: &mut R, hparams: &Hyperparameters) -> Result<Self, ModelError> {
        let n_vocab_file = reader.read_i32::<LittleEndian>()? as usize;
        let mut token_to_id = HashMap::with_capacity(hparams.n_vocab as usize);
        let mut id_to_token = vec![String::new(); hparams.n_vocab as usize];
        let mut token_bytes = vec![Vec::new(); hparams.n_vocab as usize];

        for i in 0..n_vocab_file {
            let len = reader.read_u32::<LittleEndian>()? as usize;
            let token = if len > 0 {
                let mut buf = vec![0u8; len];
                reader.read_exact(&mut buf)?;
                buf
            } else {
                Vec::new()
            };

            token_to_id.insert(token.clone(), i as i32);
            if i < id_to_token.len() {
                id_to_token[i] = String::from_utf8_lossy(&token).into_owned();
                token_bytes[i] = token;
            }
        }

        let mut token_eot = 50256;
        let mut token_sot = 50257;
        let mut token_translate = 50357;
        let mut token_transcribe = 50358;
        let mut token_solm = 50359;
        let mut token_prev = 50360;
        let mut token_nosp = 50361;
        let mut token_not = 50362;
        let mut token_beg = 50363;

        let num_languages = hparams.num_languages();
        if hparams.is_multilingual() {
            token_eot += 1;
            token_sot += 1;
            let dt = num_languages as i32 - 98;
            token_translate += dt;
            token_transcribe += dt;
            token_solm += dt;
            token_prev += dt;
            token_nosp += dt;
            token_not += dt;
            token_beg += dt;
        }

        let total_vocab = hparams.n_vocab as usize;

        // Every special token ID (token_beg is the largest) must be a valid index into the logits
        if total_vocab <= token_beg as usize {
            return Err(ModelError::FormatError(format!(
                "invalid model: n_vocab ({}) must be greater than token_beg ({})",
                total_vocab, token_beg
            )));
        }
        if n_vocab_file < total_vocab {
            for (i, slot) in id_to_token
                .iter_mut()
                .enumerate()
                .take(total_vocab)
                .skip(n_vocab_file)
            {
                let i_i32 = i as i32;
                let word = if i_i32 > token_beg {
                    format!("[_TT_{}]", i_i32 - token_beg)
                } else if i_i32 == token_eot {
                    "[_EOT_]".to_string()
                } else if i_i32 == token_sot {
                    "[_SOT_]".to_string()
                } else if i_i32 == token_translate {
                    "[_TRANSLATE_]".to_string()
                } else if i_i32 == token_transcribe {
                    "[_TRANSCRIBE_]".to_string()
                } else if i_i32 == token_solm {
                    "[_SOLM_]".to_string()
                } else if i_i32 == token_prev {
                    "[_PREV_]".to_string()
                } else if i_i32 == token_nosp {
                    "[_NOSP_]".to_string()
                } else if i_i32 == token_not {
                    "[_NOT_]".to_string()
                } else if i_i32 == token_beg {
                    "[_BEG_]".to_string()
                } else if i_i32 > token_sot && i_i32 <= token_sot + num_languages as i32 {
                    let lang_idx = (i_i32 - token_sot - 1) as usize;
                    let lstr = lang_str(lang_idx).unwrap_or("unknown");
                    format!("[_LANG_{}]", lstr)
                } else {
                    format!("[_extra_token_{}]", i)
                };

                token_to_id.insert(word.as_bytes().to_vec(), i_i32);
                if let Some(slot_bytes) = token_bytes.get_mut(i) {
                    *slot_bytes = word.as_bytes().to_vec();
                }
                *slot = word;
            }
        }

        Ok(Self {
            n_vocab: total_vocab,
            token_to_id,
            id_to_token,
            token_bytes,
            token_eot,
            token_sot,
            token_translate,
            token_transcribe,
            token_solm,
            token_prev,
            token_nosp,
            token_not,
            token_beg,
        })
    }

    pub fn token_to_id(&self, token: &str) -> Option<i32> {
        self.token_to_id.get(token.as_bytes()).copied()
    }

    pub fn token_to_id_bytes(&self, token: &[u8]) -> Option<i32> {
        self.token_to_id.get(token).copied()
    }

    pub fn id_to_token(&self, id: i32) -> Option<&str> {
        if id >= 0 && (id as usize) < self.id_to_token.len() {
            Some(&self.id_to_token[id as usize])
        } else {
            None
        }
    }

    /// Exact token bytes for `id`, without UTF-8 replacement.
    pub fn id_to_token_bytes(&self, id: i32) -> Option<&[u8]> {
        if id >= 0 && (id as usize) < self.token_bytes.len() {
            Some(&self.token_bytes[id as usize])
        } else {
            None
        }
    }
}
