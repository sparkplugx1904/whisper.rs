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
    pipeline.add(TimestampLogitFilter::new(50, 100, true));

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
