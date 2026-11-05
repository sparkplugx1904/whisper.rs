/// Calculate text compression ratio using an LZ77 dictionary estimator.
/// Repetitive/hallucinated text produces high compression ratio (> 2.4).
pub fn compression_ratio(text: &str) -> f64 {
    let bytes = text.as_bytes();
    if bytes.is_empty() {
        return 1.0;
    }

    let n = bytes.len();
    if n <= 4 {
        return 1.0;
    }

    let mut compressed_size = 0usize;
    let mut i = 0;
    let window_size = 1024usize;

    while i < n {
        let win_start = i.saturating_sub(window_size);
        let win = &bytes[win_start..i];

        // Find longest match of bytes[i..] in window
        let mut best_match_len = 0;
        let max_len = (n - i).min(32);

        if max_len >= 3 {
            for start in 0..win.len() {
                let mut match_len = 0;
                while match_len < max_len
                    && start + match_len < win.len()
                    && win[start + match_len] == bytes[i + match_len]
                {
                    match_len += 1;
                }
                if match_len > best_match_len {
                    best_match_len = match_len;
                }
            }
        }

        if best_match_len >= 3 {
            // Represent match as (distance, length) pointer: 2 bytes
            compressed_size += 2;
            i += best_match_len;
        } else {
            // Literal byte: 1 byte
            compressed_size += 1;
            i += 1;
        }
    }

    (n as f64) / (compressed_size.max(1) as f64)
}

/// Compute log-softmax probability log(P(token)) given raw logits.
pub fn compute_token_logprob(logits: &[f32], token: i32) -> f32 {
    let mut max_l = f32::NEG_INFINITY;
    for &l in logits {
        if l > max_l {
            max_l = l;
        }
    }

    let mut sum_exp = 0.0f32;
    for &l in logits {
        sum_exp += (l - max_l).exp();
    }

    let tok_l = logits[token as usize];
    (tok_l - max_l) - sum_exp.ln()
}

/// Compute Shannon entropy H(P) of the softmax probability distribution from logits.
/// H(P) = -sum_i P(i) * ln(P(i)). High entropy (> 2.40) indicates high model uncertainty.
pub fn compute_distribution_entropy(logits: &[f32]) -> f32 {
    let mut max_l = f32::NEG_INFINITY;
    for &l in logits {
        if l > max_l {
            max_l = l;
        }
    }

    if max_l == f32::NEG_INFINITY {
        return 0.0;
    }

    let mut sum_exp = 0.0f32;
    for &l in logits {
        if l > f32::NEG_INFINITY {
            sum_exp += (l - max_l).exp();
        }
    }

    let lse = max_l + sum_exp.max(1e-10).ln();

    let mut entropy = 0.0f32;
    for &l in logits {
        if l > f32::NEG_INFINITY {
            let logp = l - lse;
            let p = logp.exp();
            if p > 1e-10 {
                entropy -= p * logp;
            }
        }
    }

    entropy
}

/// Suppress blank/space token at step 0 so the model does not generate leading whitespace or early EOT.
pub fn apply_suppress_blank(logits: &mut [f32], blank_token_id: i32, eot_token_id: i32) {
    if blank_token_id >= 0 && (blank_token_id as usize) < logits.len() {
        logits[blank_token_id as usize] = f32::NEG_INFINITY;
    }
    if eot_token_id >= 0 && (eot_token_id as usize) < logits.len() {
        logits[eot_token_id as usize] = f32::NEG_INFINITY;
    }
}

/// Extract list of non-speech token indices from vocabulary once at startup.
pub fn get_non_speech_tokens(vocab: &whisper_model::Vocab) -> Vec<usize> {
    let mut tokens = Vec::new();
    for i in 0..vocab.n_vocab {
        if let Some(tok_str) = vocab.id_to_token(i as i32) {
            if tok_str.starts_with("[*")
                || tok_str.starts_with("[#")
                || tok_str.contains("music")
                || tok_str.contains("laughter")
                || tok_str.contains("applause")
            {
                tokens.push(i);
            }
        }
    }
    tokens
}

/// Fast non-speech suppression using pre-indexed token list (zero string allocations).
pub fn apply_suppress_non_speech_indices(logits: &mut [f32], indices: &[usize]) {
    for &idx in indices {
        if idx < logits.len() {
            logits[idx] = f32::NEG_INFINITY;
        }
    }
}

/// Suppress non-speech symbols (e.g. music, laughter, background noise tokens).
pub fn apply_suppress_non_speech(logits: &mut [f32], vocab: &whisper_model::Vocab) {
    let indices = get_non_speech_tokens(vocab);
    apply_suppress_non_speech_indices(logits, &indices);
}

/// Apply repetition penalty discount to tokens already generated in the current segment.
/// For positive logits: logit /= penalty
/// For negative logits: logit *= penalty
pub fn apply_repetition_penalty(logits: &mut [f32], history_tokens: &[i32], penalty: f32) {
    if penalty <= 1.0 || penalty.is_nan() {
        return;
    }
    for &tok in history_tokens {
        if tok >= 0 && (tok as usize) < logits.len() {
            let val = &mut logits[tok as usize];
            if *val > 0.0 {
                *val /= penalty;
            } else if *val < 0.0 && *val > f32::NEG_INFINITY {
                *val *= penalty;
            }
        }
    }
}

/// Generation context provided to `LogitFilter` instances at each decoding step.
#[derive(Debug, Clone)]
pub struct LogitFilterContext<'a> {
    pub step: usize,
    pub tokens_cur: &'a [i32],
    pub last_timestamp_token: Option<i32>,
}

/// Strategy trait for modifying logits before greedy or beam search sampling.
pub trait LogitFilter: Send + Sync {
    fn apply(&self, logits: &mut [f32], ctx: &LogitFilterContext);
}

/// Filter that suppresses whitespace and end-of-text at step 0.
#[derive(Debug, Clone, Copy)]
pub struct SuppressBlankFilter {
    pub blank_token_id: i32,
    pub eot_token_id: i32,
}

impl SuppressBlankFilter {
    pub fn new(blank_token_id: i32, eot_token_id: i32) -> Self {
        Self {
            blank_token_id,
            eot_token_id,
        }
    }
}

impl LogitFilter for SuppressBlankFilter {
    fn apply(&self, logits: &mut [f32], ctx: &LogitFilterContext) {
        if ctx.step == 0 {
            apply_suppress_blank(logits, self.blank_token_id, self.eot_token_id);
        }
    }
}

/// Filter that suppresses non-speech tokens (e.g. music, applause, background tokens).
#[derive(Debug, Clone)]
pub struct SuppressNonSpeechFilter {
    pub indices: Vec<usize>,
}

impl SuppressNonSpeechFilter {
    pub fn new(indices: Vec<usize>) -> Self {
        Self { indices }
    }
}

impl LogitFilter for SuppressNonSpeechFilter {
    fn apply(&self, logits: &mut [f32], _ctx: &LogitFilterContext) {
        apply_suppress_non_speech_indices(logits, &self.indices);
    }
}

/// Filter that discounts repeated tokens using a repetition penalty factor.
#[derive(Debug, Clone, Copy)]
pub struct RepetitionPenaltyFilter {
    pub penalty: f32,
}

impl RepetitionPenaltyFilter {
    pub fn new(penalty: f32) -> Self {
        Self { penalty }
    }
}

impl LogitFilter for RepetitionPenaltyFilter {
    fn apply(&self, logits: &mut [f32], ctx: &LogitFilterContext) {
        apply_repetition_penalty(logits, ctx.tokens_cur, self.penalty);
    }
}

/// Filter that constrains timestamp token generation order or suppresses timestamps altogether.
#[derive(Debug, Clone, Copy)]
pub struct TimestampLogitFilter {
    pub beg: usize,
    pub n_vocab: usize,
    pub no_timestamps: bool,
}

impl TimestampLogitFilter {
    pub fn new(beg: usize, n_vocab: usize, no_timestamps: bool) -> Self {
        Self {
            beg,
            n_vocab,
            no_timestamps,
        }
    }
}

impl LogitFilter for TimestampLogitFilter {
    fn apply(&self, logits: &mut [f32], ctx: &LogitFilterContext) {
        let limit = self.n_vocab.min(logits.len());
        if self.no_timestamps {
            if self.beg < limit {
                logits[self.beg..limit].fill(f32::NEG_INFINITY);
            }
        } else if let Some(last_ts) = ctx.last_timestamp_token {
            let ts_idx = (last_ts as usize).min(limit);
            if ts_idx > self.beg {
                logits[self.beg..ts_idx].fill(f32::NEG_INFINITY);
            }
        }
    }
}

/// Filter that constrains token predictions to match a GBNF grammar specification.
pub struct GrammarLogitFilter {
    pub grammar: whisper_grammar::GbnfGrammar,
    pub vocab: whisper_model::Vocab,
    pub tokenizer: whisper_tokenizer::Tokenizer,
}

impl GrammarLogitFilter {
    pub fn new(
        grammar: whisper_grammar::GbnfGrammar,
        vocab: whisper_model::Vocab,
        tokenizer: whisper_tokenizer::Tokenizer,
    ) -> Self {
        Self {
            grammar,
            vocab,
            tokenizer,
        }
    }
}

impl LogitFilter for GrammarLogitFilter {
    fn apply(&self, logits: &mut [f32], ctx: &LogitFilterContext) {
        let cur_text = if ctx.tokens_cur.is_empty() {
            String::new()
        } else {
            self.tokenizer.decode(ctx.tokens_cur, true)
        };
        self.grammar.filter_logits(&cur_text, logits, &self.vocab);
    }
}

/// Composable pipeline executing a sequence of `LogitFilter` strategies.
#[derive(Default)]
pub struct LogitFilterPipeline {
    filters: Vec<Box<dyn LogitFilter>>,
}

impl LogitFilterPipeline {
    pub fn new() -> Self {
        Self {
            filters: Vec::new(),
        }
    }

    pub fn add<F: LogitFilter + 'static>(&mut self, filter: F) -> &mut Self {
        self.filters.push(Box::new(filter));
        self
    }

    pub fn add_boxed(&mut self, filter: Box<dyn LogitFilter>) -> &mut Self {
        self.filters.push(filter);
        self
    }

    pub fn apply(&self, logits: &mut [f32], ctx: &LogitFilterContext) {
        for filter in &self.filters {
            filter.apply(logits, ctx);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logit_filter_pipeline_execution() {
        let mut logits = vec![2.0f32; 10];
        let mut pipeline = LogitFilterPipeline::new();
        pipeline.add(SuppressBlankFilter::new(0, 1));
        pipeline.add(RepetitionPenaltyFilter::new(2.0));

        let cur_tokens = vec![5];
        let ctx = LogitFilterContext {
            step: 0,
            tokens_cur: &cur_tokens,
            last_timestamp_token: None,
        };

        pipeline.apply(&mut logits, &ctx);
        assert_eq!(logits[0], f32::NEG_INFINITY);
        assert_eq!(logits[1], f32::NEG_INFINITY);
        assert_eq!(logits[5], 1.0);
        assert_eq!(logits[2], 2.0);
    }
}
