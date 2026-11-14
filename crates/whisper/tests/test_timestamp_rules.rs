//! Unit coverage for the timestamp grammar in [`whisper_rs::filter`].
//!
//! These tests exercise the rule ordering directly on synthetic logits rows so
//! that a regression in the temporal constraints fails without needing a model.

use whisper_rs::filter::{
    apply_timestamp_rules, LogitFilter, LogitFilterContext, TimestampLogitFilter,
};

const N_VOCAB: usize = 260;
const EOT: usize = 120;
const BEG: usize = 150;

fn fresh_logits() -> Vec<f32> {
    vec![0.0; N_VOCAB]
}

fn is_masked(logits: &[f32], index: usize) -> bool {
    logits[index] == f32::NEG_INFINITY
}

#[test]
fn text_then_timestamp_masks_text_and_past_timestamps() {
    let mut logits = fresh_logits();
    // A single text token followed by an opening timestamp: the segment is
    // being closed, so only timestamps at or after it may follow.
    let tokens = [11, (BEG + 10) as i32];
    apply_timestamp_rules(&mut logits, &tokens, BEG, EOT, N_VOCAB, false, 1.0, false);

    assert!(
        (0..EOT).all(|i| is_masked(&logits, i)),
        "text must be masked"
    );
    assert!(
        (BEG..BEG + 10).all(|i| is_masked(&logits, i)),
        "timestamps before the last one must be masked"
    );
    assert_eq!(logits[BEG + 10], 0.0);
    assert_eq!(logits[BEG + 20], 0.0);
}

#[test]
fn two_timestamps_close_the_pair_and_mask_all_timestamps() {
    let mut logits = fresh_logits();
    let tokens = [(BEG + 5) as i32, (BEG + 10) as i32];
    apply_timestamp_rules(&mut logits, &tokens, BEG, EOT, N_VOCAB, false, 1.0, false);

    assert!(
        (BEG..N_VOCAB).all(|i| is_masked(&logits, i)),
        "a closed timestamp pair must force text"
    );
    assert_eq!(logits[7], 0.0, "text stays available after a closed pair");
}

#[test]
fn timestamp_mass_forces_a_timestamp_when_it_outweighs_text() {
    // Many timestamp logits at 1.2 with 50 candidates produce a timestamp
    // log-probability of roughly 1.2 + ln(50) = 5.1, which beats text at 0.0.
    let mut logits = fresh_logits();
    for value in &mut logits[BEG..] {
        *value = 1.2;
    }
    apply_timestamp_rules(&mut logits, &[], BEG, EOT, N_VOCAB, false, 1.0, false);

    assert!(
        (0..BEG).all(|i| is_masked(&logits, i)),
        "timestamp mass above the best text token must mask text"
    );
    assert_eq!(logits[BEG], 1.2);
}

#[test]
fn dominant_text_token_survives_the_mass_rule() {
    let mut logits = fresh_logits();
    for value in &mut logits[BEG..] {
        *value = 1.2;
    }
    logits[42] = 10.0;
    apply_timestamp_rules(&mut logits, &[], BEG, EOT, N_VOCAB, false, 1.0, false);

    assert_eq!(logits[42], 10.0, "a dominant text logit must not be masked");
    assert!(
        (0..BEG)
            .filter(|&i| i != 42)
            .all(|i| !is_masked(&logits, i)),
        "other text tokens are not touched by the mass rule"
    );
}

#[test]
fn no_timestamps_masks_every_token_from_beg() {
    let mut logits = fresh_logits();
    logits[5] = 3.0;
    apply_timestamp_rules(&mut logits, &[], BEG, EOT, N_VOCAB, true, 1.0, false);

    assert_eq!(logits[5], 3.0);
    assert!((BEG..N_VOCAB).all(|i| is_masked(&logits, i)));
}

#[test]
fn max_initial_timestamp_limits_the_opening_timestamp() {
    let mut logits = fresh_logits();
    // max_initial_ts = 1.0 s is 50 timestamp steps of 20 ms.
    apply_timestamp_rules(&mut logits, &[], BEG, EOT, N_VOCAB, false, 1.0, true);

    assert_eq!(logits[BEG], 0.0);
    assert_eq!(logits[BEG + 50], 0.0);
    assert!(
        (BEG + 51..N_VOCAB).all(|i| is_masked(&logits, i)),
        "timestamps later than max_initial_ts must be masked"
    );

    // The rule is disabled outside the first decoding step.
    let mut later = fresh_logits();
    apply_timestamp_rules(&mut later, &[], BEG, EOT, N_VOCAB, false, 1.0, false);
    assert_eq!(later[BEG + 51], 0.0);

    // A tighter limit masks earlier.
    let mut tight = fresh_logits();
    apply_timestamp_rules(&mut tight, &[], BEG, EOT, N_VOCAB, false, 0.5, true);
    assert_eq!(tight[BEG + 25], 0.0);
    assert!((BEG + 26..N_VOCAB).all(|i| is_masked(&tight, i)));
}

#[test]
fn timestamp_filter_applies_initial_limit_on_first_step() {
    let mut logits = fresh_logits();
    let filter = TimestampLogitFilter::with_max_initial_ts(BEG, EOT, N_VOCAB, false, 0.4);
    filter.apply(
        &mut logits,
        &LogitFilterContext {
            step: 0,
            tokens_cur: &[],
            last_timestamp_token: None,
        },
    );
    assert_eq!(logits[BEG + 20], 0.0);
    assert!((BEG + 21..N_VOCAB).all(|i| is_masked(&logits, i)));
}
