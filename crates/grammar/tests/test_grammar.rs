use std::fs;
use std::path::Path;
use whisper_grammar::GbnfGrammar;

#[test]
fn test_parse_colors_grammar() {
    let grammar_text = r#"
root   ::= "red" | "green" | "blue"
"#;
    let grammar = GbnfGrammar::parse(grammar_text).expect("parse colors");
    assert!(grammar.accepts_prefix("r"));
    assert!(grammar.accepts_prefix("red"));
    assert!(grammar.accepts_prefix("green"));
    assert!(!grammar.accepts_prefix("yellow"));
}

#[test]
fn test_parse_real_gbnf_files() {
    for file in &["colors.gbnf", "chess.gbnf", "assistant.gbnf"] {
        let path = Path::new("grammars").join(file);
        if !path.exists() {
            let alt_path = Path::new("../../grammars").join(file);
            if !alt_path.exists() {
                continue;
            }
        }
        let content = fs::read_to_string(&path)
            .or_else(|_| fs::read_to_string(Path::new("../../grammars").join(file)))
            .unwrap();

        let grammar = GbnfGrammar::parse(&content);
        assert!(grammar.is_ok(), "Failed to parse grammar file {}", file);
    }
}
