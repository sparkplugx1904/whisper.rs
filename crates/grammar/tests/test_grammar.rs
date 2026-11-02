use std::fs;
use std::path::Path;
use whisper_grammar::GbnfGrammar;

#[test]
fn test_char_classes_and_ranges() {
    let grammar_text = r#"
root ::= [0-9]+ " " [a-z]+
"#;
    let grammar = GbnfGrammar::parse(grammar_text).expect("parse char class");
    assert!(grammar.accepts_prefix("42 hello"));
    assert!(grammar.accepts_prefix("1 "));
    assert!(grammar.accepts_prefix("999 foo"));
    assert!(!grammar.accepts_prefix("abc"));
    assert!(!grammar.accepts_prefix("42 123"));
}

#[test]
fn test_nested_groups_and_optional() {
    let grammar_text = r#"
root ::= ("red" | "green") " " ("apple" | "pepper") ("!" | "?")?
"#;
    let grammar = GbnfGrammar::parse(grammar_text).expect("parse nested groups");
    assert!(grammar.accepts_prefix("red apple"));
    assert!(grammar.accepts_prefix("green pepper!"));
    assert!(grammar.accepts_prefix("green pepper?"));
    assert!(grammar.accepts_prefix("red "));
    assert!(!grammar.accepts_prefix("blue apple"));
    assert!(!grammar.accepts_prefix("red banana"));
}

#[test]
fn test_unknown_rule_strictly_fails() {
    let grammar_text = r#"
root ::= nonexistent_rule
"#;
    let grammar = GbnfGrammar::parse(grammar_text).expect("parse grammar with missing rule");
    assert!(
        !grammar.accepts_prefix("anything"),
        "Unresolved rule must not accept prefix"
    );
}

#[test]
fn test_parse_real_gbnf_files() {
    for file in &["colors.gbnf", "chess.gbnf", "assistant.gbnf"] {
        let path = Path::new("grammars").join(file);
        let content = if path.exists() {
            fs::read_to_string(&path).unwrap()
        } else {
            let alt_path = Path::new("../../grammars").join(file);
            fs::read_to_string(&alt_path).unwrap()
        };

        let grammar = GbnfGrammar::parse(&content);
        assert!(
            grammar.is_ok(),
            "Failed to parse grammar file {}: {:?}",
            file,
            grammar.err()
        );
    }
}
