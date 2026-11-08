use std::collections::HashMap;
use whisper_grammar::GbnfGrammar;
use whisper_model::Vocab;

fn make_test_vocab(tokens: &[&str]) -> Vocab {
    let mut token_to_id = HashMap::new();
    let mut id_to_token = Vec::new();
    let mut token_bytes = Vec::new();
    for (i, &t) in tokens.iter().enumerate() {
        token_to_id.insert(t.as_bytes().to_vec(), i as i32);
        id_to_token.push(t.to_string());
        token_bytes.push(t.as_bytes().to_vec());
    }
    Vocab {
        n_vocab: tokens.len(),
        token_to_id,
        id_to_token,
        token_bytes,
        token_eot: 50256,
        token_sot: 50257,
        token_translate: 50258,
        token_transcribe: 50259,
        token_solm: 50260,
        token_prev: 50261,
        token_nosp: 50262,
        token_not: 50263,
        token_beg: 50264,
    }
}

#[test]
fn test_inverted_character_classes() {
    // 1. Single inverted class: any character EXCEPT a digit 0-9
    let grammar = GbnfGrammar::parse("root ::= [^0-9]").expect("parse inverted class");
    assert!(grammar.accepts_prefix("a"));
    assert!(grammar.accepts_prefix("Z"));
    assert!(grammar.accepts_prefix("-"));
    assert!(grammar.accepts_prefix("_"));
    assert!(grammar.accepts_prefix(" "));
    assert!(grammar.accepts_prefix("!"));
    assert!(grammar.accepts_prefix("@"));

    // Digits must be rejected
    for d in "0123456789".chars() {
        assert!(
            !grammar.accepts_prefix(&d.to_string()),
            "Inverted [^0-9] must not accept digit '{}'",
            d
        );
    }

    // 2. Inverted class with repetition
    let rep_grammar =
        GbnfGrammar::parse("root ::= [^0-9]+").expect("parse inverted class repetition");
    assert!(rep_grammar.accepts_prefix("hello"));
    assert!(rep_grammar.accepts_prefix("hello world!"));
    assert!(rep_grammar.accepts_prefix("alpha_beta-gamma"));
    assert!(!rep_grammar.accepts_prefix("hello 1 world"));
    assert!(!rep_grammar.accepts_prefix("42"));
    assert!(!rep_grammar.accepts_prefix("abc9def"));

    // 3. Inverted multi-range class: non-alphabetic
    let non_alpha = GbnfGrammar::parse("root ::= [^a-zA-Z]+").expect("parse non-alpha inverted");
    assert!(non_alpha.accepts_prefix("12345"));
    assert!(non_alpha.accepts_prefix("!@#$%^&*()"));
    assert!(non_alpha.accepts_prefix("   \t  "));
    assert!(!non_alpha.accepts_prefix("123a"));
    assert!(!non_alpha.accepts_prefix("Test"));
}

#[test]
fn test_multi_range_classes() {
    // Multi-range class [a-zA-Z0-9_-]
    let grammar = GbnfGrammar::parse("root ::= [a-zA-Z0-9_-]+").expect("parse multi-range");

    // All allowed components
    assert!(grammar.accepts_prefix("a"));
    assert!(grammar.accepts_prefix("z"));
    assert!(grammar.accepts_prefix("A"));
    assert!(grammar.accepts_prefix("Z"));
    assert!(grammar.accepts_prefix("0"));
    assert!(grammar.accepts_prefix("9"));
    assert!(grammar.accepts_prefix("_"));
    assert!(grammar.accepts_prefix("-"));

    // Combined words
    assert!(grammar.accepts_prefix("valid-identifier_123"));
    assert!(grammar.accepts_prefix("X_42-test"));
    assert!(grammar.accepts_prefix("---___---"));
    assert!(grammar.accepts_prefix("0123456789"));

    // Disallowed characters
    assert!(!grammar.accepts_prefix("has space"));
    assert!(!grammar.accepts_prefix("has.dot"));
    assert!(!grammar.accepts_prefix("has/slash"));
    assert!(!grammar.accepts_prefix("has@at"));
    assert!(!grammar.accepts_prefix("has+plus"));
    assert!(!grammar.accepts_prefix("has:colon"));

    // Multi-range in an identifier rule
    let id_grammar = GbnfGrammar::parse(
        r#"
root ::= [a-zA-Z_] [a-zA-Z0-9_-]*
"#,
    )
    .expect("parse identifier grammar");
    assert!(id_grammar.accepts_prefix("my_var"));
    assert!(id_grammar.accepts_prefix("_private"));
    assert!(id_grammar.accepts_prefix("k-case"));
    assert!(id_grammar.accepts_prefix("v1"));
    assert!(!id_grammar.accepts_prefix("-k-case")); // Cannot start with '-'
    assert!(!id_grammar.accepts_prefix("0var")); // Cannot start with '0'
}

#[test]
fn test_complex_nested_repetitions() {
    // Specification requirement: (([0-9]+ ", ")* [0-9]+)
    let grammar_text = r#"
root ::= (([0-9]+ ", ")* [0-9]+)
"#;
    let grammar = GbnfGrammar::parse(grammar_text).expect("parse nested repetitions");

    // Single number
    assert!(grammar.accepts_prefix("1"));
    assert!(grammar.accepts_prefix("42"));
    assert!(grammar.accepts_prefix("123456789"));

    // Two numbers
    assert!(grammar.accepts_prefix("1, 2"));
    assert!(grammar.accepts_prefix("42, 99"));

    // Multiple numbers
    assert!(grammar.accepts_prefix("1, 2, 3"));
    assert!(grammar.accepts_prefix("10, 20, 30, 40, 50"));
    assert!(grammar.accepts_prefix("9, 88, 777, 6666, 55555"));

    // Valid intermediate prefixes
    assert!(grammar.accepts_prefix("1,"));
    assert!(grammar.accepts_prefix("1, "));
    assert!(grammar.accepts_prefix("1, 2,"));
    assert!(grammar.accepts_prefix("1, 2, "));
    assert!(grammar.accepts_prefix("100, 200, 3"));

    // Invalid sequences
    assert!(!grammar.accepts_prefix("abc"));
    assert!(!grammar.accepts_prefix(", 1"));
    assert!(!grammar.accepts_prefix("1 2"));
    assert!(!grammar.accepts_prefix("1, 2, ,"));
    assert!(!grammar.accepts_prefix("1, 2, a"));
    assert!(!grammar.accepts_prefix("1, 2, 3x"));
    assert!(!grammar.accepts_prefix("1, -2"));

    // Nested repetition inside a bracketed list
    let list_grammar = GbnfGrammar::parse(
        r#"
root ::= "[" (([0-9]+ ", ")* [0-9]+)? "]"
"#,
    )
    .expect("parse bracketed list grammar");
    assert!(list_grammar.accepts_prefix("[]"));
    assert!(list_grammar.accepts_prefix("[1]"));
    assert!(list_grammar.accepts_prefix("[1, 2, 3]"));
    assert!(list_grammar.accepts_prefix("["));
    assert!(list_grammar.accepts_prefix("[1, "));
    assert!(list_grammar.accepts_prefix("[1, 2"));
    assert!(!list_grammar.accepts_prefix("[1, 2,]"));
    assert!(!list_grammar.accepts_prefix("[a]"));
}

#[test]
fn test_empty_input_and_prefix_boundary() {
    let grammar = GbnfGrammar::parse(
        r#"
root ::= "SELECT " [a-z]+ " FROM " [a-z]+ ";"
"#,
    )
    .expect("parse SQL query grammar");

    // Empty input must always be accepted as a valid prefix
    assert!(grammar.accepts_prefix(""));
    assert!(grammar.accepts_prefix_extended("", ""));

    // Step-by-step prefix boundary verification
    let valid_steps = [
        "",
        "S",
        "SEL",
        "SELECT",
        "SELECT ",
        "SELECT u",
        "SELECT users",
        "SELECT users ",
        "SELECT users F",
        "SELECT users FROM",
        "SELECT users FROM ",
        "SELECT users FROM t",
        "SELECT users FROM table",
        "SELECT users FROM table;",
    ];

    for &step in &valid_steps {
        assert!(
            grammar.accepts_prefix(step),
            "Step '{}' must be accepted as valid prefix",
            step
        );
    }

    // Invalid steps at each boundary
    assert!(!grammar.accepts_prefix("DELETE"));
    assert!(!grammar.accepts_prefix("SELECT 123"));
    assert!(!grammar.accepts_prefix("SELECT users WHERE"));
    assert!(!grammar.accepts_prefix("SELECT users FROM table; EXTRA"));

    // accepts_prefix_extended incremental streaming tests
    assert!(grammar.accepts_prefix_extended("SELECT ", "users"));
    assert!(grammar.accepts_prefix_extended("SELECT users", " FROM"));
    assert!(grammar.accepts_prefix_extended("SELECT users FROM ", "table;"));
    assert!(!grammar.accepts_prefix_extended("SELECT ", "123"));
    assert!(!grammar.accepts_prefix_extended("SELECT users", " WHERE"));
}

#[test]
fn test_stress_repetition_deep() {
    // Stress test with 100 comma-separated numbers
    let grammar = GbnfGrammar::parse(r#"root ::= (([0-9]+ ", ")* [0-9]+)"#).expect("parse");

    let mut numbers_str = String::new();
    for i in 0..100 {
        if i > 0 {
            numbers_str.push_str(", ");
        }
        numbers_str.push_str(&i.to_string());
        assert!(
            grammar.accepts_prefix(&numbers_str),
            "Failed at number index {}",
            i
        );
    }

    // Trailing comma + space should also be a valid prefix
    numbers_str.push_str(", ");
    assert!(grammar.accepts_prefix(&numbers_str));

    // Non-digit suffix must be rejected
    numbers_str.push_str("bad");
    assert!(!grammar.accepts_prefix(&numbers_str));
}

#[test]
fn test_filter_logits_with_nested_grammar() {
    let grammar = GbnfGrammar::parse(r#"root ::= (([0-9]+ ", ")* [0-9]+)"#).expect("parse");

    let vocab = make_test_vocab(&["1", ", ", "abc", " ", "9", "foo"]);
    // ID map: 0: "1", 1: ", ", 2: "abc", 3: " ", 4: "9", 5: "foo"

    // Context: "42" (can be extended with digit "1" or "9", or comma-space ", ")
    let mut logits = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    grammar.filter_logits("42", &mut logits, &vocab);

    // "42" + "1" -> "421" valid prefix
    assert_eq!(logits[0], 1.0);
    // "42" + ", " -> "42, " valid prefix
    assert_eq!(logits[1], 2.0);
    // "42" + "abc" -> "42abc" INVALID
    assert_eq!(logits[2], f32::NEG_INFINITY);
    // "42" + " " -> "42 " INVALID (space without comma)
    assert_eq!(logits[3], f32::NEG_INFINITY);
    // "42" + "9" -> "429" valid prefix
    assert_eq!(logits[4], 5.0);
    // "42" + "foo" -> INVALID
    assert_eq!(logits[5], f32::NEG_INFINITY);

    // Context: "42, " (can only be extended with digits "1" or "9")
    let mut logits2 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    grammar.filter_logits("42, ", &mut logits2, &vocab);

    assert_eq!(logits2[0], 1.0); // "1" valid
    assert_eq!(logits2[1], f32::NEG_INFINITY); // ", " invalid
    assert_eq!(logits2[2], f32::NEG_INFINITY); // "abc" invalid
    assert_eq!(logits2[3], f32::NEG_INFINITY); // " " invalid
    assert_eq!(logits2[4], 5.0); // "9" valid
    assert_eq!(logits2[5], f32::NEG_INFINITY); // "foo" invalid
}
