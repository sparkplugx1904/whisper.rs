use whisper_rs::dtw::merge_tokens_to_words;

#[test]
fn test_punctuation_snapping_and_monotonicity() {
    let tokens = [(" Hello", 0.95), (",", 0.99), (" world", 0.90), ("!", 0.98)];
    let alignments = [(0, 0, 5), (1, 6, 7), (2, 8, 15), (3, 16, 17)];

    let words = merge_tokens_to_words(&tokens, &alignments, 0.0, 0.02);
    assert_eq!(
        words.len(),
        2,
        "Punctuation must snap to words, leaving 2 words"
    );
    assert_eq!(words[0].word, "Hello,");
    assert_eq!(words[1].word, "world!");

    // Monotonicity check
    assert!(words[0].t0 <= words[0].t1);
    assert!(words[0].t0 <= words[1].t0);
    assert!(words[1].t0 <= words[1].t1);
}
