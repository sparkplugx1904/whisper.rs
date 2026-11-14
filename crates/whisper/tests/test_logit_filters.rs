use whisper_model::Vocab;
use whisper_rs::filter::{
    apply_repetition_penalty, apply_suppress_blank, apply_suppress_non_speech,
};

#[test]
fn test_apply_suppress_blank() {
    let mut logits = vec![1.0f32; 1000];
    apply_suppress_blank(&mut logits, 220, 50256);
    assert_eq!(logits[220], f32::NEG_INFINITY);
    assert_eq!(logits[50256 % 1000], 1.0); // Out-of-bounds doesn't panic
}

#[test]
fn test_apply_repetition_penalty() {
    let mut logits = vec![2.0f32, -4.0f32, 0.5f32, 10.0f32];
    let history = vec![0, 1];
    apply_repetition_penalty(&mut logits, &history, 2.0);

    // Positive logit should be divided by penalty
    assert_eq!(logits[0], 1.0);
    // Negative logit should be multiplied by penalty
    assert_eq!(logits[1], -8.0);
    // Unrepeated tokens unchanged
    assert_eq!(logits[2], 0.5);
    assert_eq!(logits[3], 10.0);
}

#[test]
fn test_apply_suppress_non_speech() {
    let mut vocab = Vocab::new();
    let id_applause = vocab.add_token("[*applause*]");
    let id_word = vocab.add_token("hello");

    let mut logits = vec![5.0f32; 10];
    apply_suppress_non_speech(&mut logits, &vocab);

    assert_eq!(logits[id_applause as usize], f32::NEG_INFINITY);
    assert_eq!(logits[id_word as usize], 5.0);
}

#[test]
fn test_logit_filter_pipeline_combination() {
    use whisper_rs::filter::{
        LogitFilterContext, LogitFilterPipeline, RepetitionPenaltyFilter, SuppressBlankFilter,
        TimestampLogitFilter,
    };

    let mut logits = vec![1.0f32; 100];
    logits[10] = 4.0;

    let mut pipeline = LogitFilterPipeline::new();
    pipeline.add(SuppressBlankFilter::new(5, 6));
    pipeline.add(RepetitionPenaltyFilter::new(2.0));
    pipeline.add(TimestampLogitFilter::new(50, 40, 100, true));

    let history = vec![10];
    let ctx = LogitFilterContext {
        step: 0,
        tokens_cur: &history,
        last_timestamp_token: None,
    };

    pipeline.apply(&mut logits, &ctx);

    // Blank and EOT suppressed at step 0
    assert_eq!(logits[5], f32::NEG_INFINITY);
    assert_eq!(logits[6], f32::NEG_INFINITY);
    // History token 10 penalized (4.0 / 2.0 = 2.0)
    assert_eq!(logits[10], 2.0);
    // Timestamps 50..100 suppressed because no_timestamps is true
    assert_eq!(logits[50], f32::NEG_INFINITY);
    assert_eq!(logits[99], f32::NEG_INFINITY);
    // Token 2 untouched
    assert_eq!(logits[2], 1.0);
}

#[test]
fn test_regex_filter() {
    use whisper_rs::filter::{LogitFilter, LogitFilterContext, RegexFilter};

    let mut vocab = Vocab::new();
    let id_hello = vocab.add_token("hello");
    let id_world = vocab.add_token("world");
    let id_music = vocab.add_token("[MUSIC]");
    let id_applause = vocab.add_token("[APPLAUSE]");
    let id_num = vocab.add_token("123");

    let filter = RegexFilter::new(r"^\[.*\]$", &vocab).expect("regex compile failed");
    assert!(filter.indices.contains(&(id_music as usize)));
    assert!(filter.indices.contains(&(id_applause as usize)));
    assert!(!filter.indices.contains(&(id_hello as usize)));
    assert!(!filter.indices.contains(&(id_world as usize)));
    assert!(!filter.indices.contains(&(id_num as usize)));

    let mut logits = vec![1.0f32; 10];
    let ctx = LogitFilterContext {
        step: 0,
        tokens_cur: &[],
        last_timestamp_token: None,
    };
    filter.apply(&mut logits, &ctx);

    assert_eq!(logits[id_music as usize], f32::NEG_INFINITY);
    assert_eq!(logits[id_applause as usize], f32::NEG_INFINITY);
    assert_eq!(logits[id_hello as usize], 1.0);
    assert_eq!(logits[id_world as usize], 1.0);
    assert_eq!(logits[id_num as usize], 1.0);
}

#[test]
#[allow(clippy::needless_range_loop)]
fn test_timestamp_logit_filter_max_initial_ts() {
    use whisper_rs::filter::{LogitFilter, LogitFilterContext, TimestampLogitFilter};

    let beg = 50;
    let n_vocab = 100;
    let max_initial_ts = 0.5f32; // 0.5s / 0.02 = 25 tokens. Max initial ts token = 50 + 25 = 75.
    let filter = TimestampLogitFilter::with_max_initial_ts(beg, 40, n_vocab, false, max_initial_ts);

    let mut logits = vec![1.0f32; 100];
    let ctx_step0 = LogitFilterContext {
        step: 0,
        tokens_cur: &[],
        last_timestamp_token: None,
    };
    filter.apply(&mut logits, &ctx_step0);

    // Tokens <= 75 should not be suppressed
    for i in beg..=75 {
        assert_eq!(logits[i], 1.0, "Token {} should not be suppressed", i);
    }
    // Tokens > 75 should be suppressed to -f32::INFINITY
    for i in 76..100 {
        assert_eq!(
            logits[i],
            f32::NEG_INFINITY,
            "Token {} should be suppressed",
            i
        );
    }

    // A text token followed by a timestamp opens a pair: text is masked and
    // timestamps may only move forward.
    let mut logits_step1 = vec![1.0f32; 100];
    let ctx_step1 = LogitFilterContext {
        step: 1,
        tokens_cur: &[10, 60],
        last_timestamp_token: Some(60),
    };
    filter.apply(&mut logits_step1, &ctx_step1);
    for i in 0..40 {
        assert_eq!(
            logits_step1[i],
            f32::NEG_INFINITY,
            "Text token {} must be masked inside a timestamp pair",
            i
        );
    }
    for i in beg..60 {
        assert_eq!(
            logits_step1[i],
            f32::NEG_INFINITY,
            "Timestamp {} must not regress",
            i
        );
    }
    for i in 60..100 {
        assert_eq!(logits_step1[i], 1.0);
    }

    // Two consecutive timestamps close the pair, so no further timestamp may
    // follow until text is emitted.
    let mut logits_pair = vec![1.0f32; 100];
    let ctx_pair = LogitFilterContext {
        step: 2,
        tokens_cur: &[60, 70],
        last_timestamp_token: Some(70),
    };
    filter.apply(&mut logits_pair, &ctx_pair);
    for i in beg..100 {
        assert_eq!(
            logits_pair[i],
            f32::NEG_INFINITY,
            "Timestamp {} must be masked after a closed pair",
            i
        );
    }
}

#[test]
fn test_params_parity_fields() {
    use whisper_rs::{FullParamsBuilder, SamplingStrategy, WhisperParams};

    let default_params = WhisperParams::default();
    assert_eq!(default_params.offset_ms, 0);
    assert_eq!(default_params.duration_ms, 0);
    assert_eq!(default_params.max_initial_ts, 1.0);
    assert!(!default_params.single_segment);
    assert_eq!(default_params.suppress_regex, None);
    assert_eq!(default_params.audio_ctx, 0);

    let params = FullParamsBuilder::new(SamplingStrategy::Greedy { best_of: 1 })
        .offset_ms(1500)
        .duration_ms(3000)
        .max_initial_ts(0.5)
        .single_segment(true)
        .suppress_regex(r"^\[.*\]$")
        .audio_ctx(750)
        .build();

    assert_eq!(params.offset_ms, 1500);
    assert_eq!(params.duration_ms, 3000);
    assert_eq!(params.max_initial_ts, 0.5);
    assert!(params.single_segment);
    assert_eq!(params.suppress_regex.as_deref(), Some(r"^\[.*\]$"));
    assert_eq!(params.audio_ctx, 750);
}
