#![forbid(unsafe_code)]
//! GBNF Grammar-guided constrained sampling for whisper.rs.

use std::collections::HashMap;
use thiserror::Error;
use whisper_model::Vocab;

#[derive(Error, Debug)]
pub enum GrammarError {
    #[error("Parse error: {0}")]
    ParseError(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum GbnfNode {
    Literal(String),
    RuleRef(String),
    CharClass {
        ranges: Vec<(char, char)>,
        negated: bool,
    },
    Alternation(Vec<GbnfNode>),
    Sequence(Vec<GbnfNode>),
    Optional(Box<GbnfNode>),
    Repeat(Box<GbnfNode>, bool), // (inner, min_1) -> true for +, false for *
}

#[derive(Debug, Clone)]
pub struct GbnfGrammar {
    pub rules: HashMap<String, GbnfNode>,
    pub root: String,
}

impl GbnfGrammar {
    /// Parse a GBNF grammar specification string.
    pub fn parse(input: &str) -> Result<Self, GrammarError> {
        let mut rules = HashMap::new();
        let root = "root".to_string();

        for line in input.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            if let Some(pos) = line.find("::=") {
                let name = line[..pos].trim().to_string();
                let expr_str = line[pos + 3..].trim();
                let mut parser = GbnfParser::new(expr_str);
                let node = parser.parse_alternation()?;
                rules.insert(name, node);
            }
        }

        if rules.is_empty() {
            return Err(GrammarError::ParseError(
                "No valid rules found in grammar".to_string(),
            ));
        }

        if !rules.contains_key(&root) {
            return Err(GrammarError::ParseError(
                "Missing 'root' rule in grammar".to_string(),
            ));
        }

        Ok(Self { rules, root })
    }

    /// Check if the text matches or forms a valid prefix under the grammar.
    pub fn accepts_prefix(&self, text: &str) -> bool {
        self.accepts_prefix_extended(text, "")
    }

    /// Check if `current_text + next_chunk` forms a valid prefix under the grammar.
    pub fn accepts_prefix_extended(&self, current_text: &str, next_chunk: &str) -> bool {
        let combined = if next_chunk.is_empty() {
            current_text.to_string()
        } else {
            let mut s = String::with_capacity(current_text.len() + next_chunk.len());
            s.push_str(current_text);
            s.push_str(next_chunk);
            s
        };

        if combined.is_empty() {
            return true;
        }

        let (matched, end_pos) = self.match_rule(&self.root, &combined, 0);
        // Valid prefix if it matched cleanly up to the end of the combined string
        matched && end_pos >= combined.len()
    }

    fn match_rule(&self, rule_name: &str, text: &str, pos: usize) -> (bool, usize) {
        if let Some(node) = self.rules.get(rule_name) {
            self.match_node(node, text, pos)
        } else {
            // Strictly reject unknown rule references
            (false, pos)
        }
    }

    fn match_node(&self, node: &GbnfNode, text: &str, pos: usize) -> (bool, usize) {
        if pos >= text.len() {
            // Prefix boundary reached: valid prefix extension
            return (true, pos);
        }

        match node {
            GbnfNode::Literal(lit) => {
                let remaining = &text[pos..];
                if lit.starts_with(remaining) {
                    (true, text.len())
                } else if remaining.starts_with(lit) {
                    (true, pos + lit.len())
                } else {
                    (false, pos)
                }
            }
            GbnfNode::CharClass { ranges, negated } => {
                if let Some(c) = text[pos..].chars().next() {
                    let in_range = ranges.iter().any(|&(lo, hi)| c >= lo && c <= hi);
                    let matched = if *negated { !in_range } else { in_range };
                    if matched {
                        (true, pos + c.len_utf8())
                    } else {
                        (false, pos)
                    }
                } else {
                    (true, pos)
                }
            }
            GbnfNode::RuleRef(name) => self.match_rule(name, text, pos),
            GbnfNode::Alternation(alts) => {
                for alt in alts {
                    let (ok, next_pos) = self.match_node(alt, text, pos);
                    if ok {
                        return (true, next_pos);
                    }
                }
                (false, pos)
            }
            GbnfNode::Sequence(seq) => {
                let mut cur = pos;
                for item in seq {
                    let (ok, next_pos) = self.match_node(item, text, cur);
                    if !ok {
                        return (false, pos);
                    }
                    cur = next_pos;
                    if cur >= text.len() {
                        return (true, cur);
                    }
                }
                (true, cur)
            }
            GbnfNode::Optional(inner) => {
                let (ok, next_pos) = self.match_node(inner, text, pos);
                if ok {
                    (true, next_pos)
                } else {
                    (true, pos)
                }
            }
            GbnfNode::Repeat(inner, min_1) => {
                let mut cur = pos;
                let mut count = 0;
                while cur < text.len() {
                    let (ok, next_pos) = self.match_node(inner, text, cur);
                    if !ok || next_pos == cur {
                        break;
                    }
                    cur = next_pos;
                    count += 1;
                }
                if *min_1 && count == 0 {
                    (false, pos)
                } else {
                    (true, cur)
                }
            }
        }
    }

    /// Mask invalid token logits with -INFINITY without excessive string allocation.
    pub fn filter_logits(&self, current_text: &str, logits: &mut [f32], vocab: &Vocab) {
        for (tok_id, val) in logits.iter_mut().enumerate() {
            if *val == f32::NEG_INFINITY {
                continue;
            }

            if let Some(tok_str) = vocab.id_to_token(tok_id as i32) {
                if !self.accepts_prefix_extended(current_text, tok_str) {
                    *val = f32::NEG_INFINITY;
                }
            }
        }
    }
}

struct GbnfParser<'a> {
    chars: Vec<char>,
    pos: usize,
    _marker: std::marker::PhantomData<&'a ()>,
}

impl<'a> GbnfParser<'a> {
    fn new(input: &'a str) -> Self {
        Self {
            chars: input.chars().collect(),
            pos: 0,
            _marker: std::marker::PhantomData,
        }
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn advance(&mut self) -> Option<char> {
        let c = self.peek();
        if c.is_some() {
            self.pos += 1;
        }
        c
    }

    fn skip_whitespace(&mut self) {
        while let Some(c) = self.peek() {
            if c == ' ' || c == '\t' || c == '\r' || c == '\n' {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn parse_alternation(&mut self) -> Result<GbnfNode, GrammarError> {
        let mut alts = Vec::new();
        alts.push(self.parse_sequence()?);

        loop {
            self.skip_whitespace();
            if self.peek() == Some('|') {
                self.advance();
                alts.push(self.parse_sequence()?);
            } else {
                break;
            }
        }

        if alts.len() == 1 {
            Ok(alts.remove(0))
        } else {
            Ok(GbnfNode::Alternation(alts))
        }
    }

    fn parse_sequence(&mut self) -> Result<GbnfNode, GrammarError> {
        let mut seq = Vec::new();
        loop {
            self.skip_whitespace();
            if let Some(c) = self.peek() {
                if c == '|' || c == ')' || c == ']' {
                    break;
                }
                let item = self.parse_element()?;
                seq.push(item);
            } else {
                break;
            }
        }

        if seq.is_empty() {
            Ok(GbnfNode::Sequence(Vec::new()))
        } else if seq.len() == 1 {
            Ok(seq.remove(0))
        } else {
            Ok(GbnfNode::Sequence(seq))
        }
    }

    fn parse_element(&mut self) -> Result<GbnfNode, GrammarError> {
        let base = self.parse_primary()?;
        self.skip_whitespace();

        // Check postfix: ?, *, +
        match self.peek() {
            Some('?') => {
                self.advance();
                Ok(GbnfNode::Optional(Box::new(base)))
            }
            Some('*') => {
                self.advance();
                Ok(GbnfNode::Repeat(Box::new(base), false))
            }
            Some('+') => {
                self.advance();
                Ok(GbnfNode::Repeat(Box::new(base), true))
            }
            _ => Ok(base),
        }
    }

    fn parse_primary(&mut self) -> Result<GbnfNode, GrammarError> {
        self.skip_whitespace();
        match self.peek() {
            Some('"') => self.parse_string_literal(),
            Some('(') => {
                self.advance(); // consume '('
                let inner = self.parse_alternation()?;
                self.skip_whitespace();
                if self.peek() == Some(')') {
                    self.advance();
                    Ok(inner)
                } else {
                    Err(GrammarError::ParseError(
                        "Unclosed parenthesis in grammar".to_string(),
                    ))
                }
            }
            Some('[') => self.parse_char_class(),
            Some(c) if c.is_alphanumeric() || c == '_' || c == '-' => self.parse_rule_ref(),
            Some(c) => Err(GrammarError::ParseError(format!(
                "Unexpected character '{}'",
                c
            ))),
            None => Err(GrammarError::ParseError(
                "Unexpected end of grammar expression".to_string(),
            )),
        }
    }

    fn parse_string_literal(&mut self) -> Result<GbnfNode, GrammarError> {
        self.advance(); // consume initial '"'
        let mut s = String::new();
        while let Some(c) = self.advance() {
            if c == '"' {
                return Ok(GbnfNode::Literal(s));
            } else if c == '\\' {
                if let Some(esc) = self.advance() {
                    match esc {
                        'n' => s.push('\n'),
                        'r' => s.push('\r'),
                        't' => s.push('\t'),
                        '\\' => s.push('\\'),
                        '"' => s.push('"'),
                        other => s.push(other),
                    }
                }
            } else {
                s.push(c);
            }
        }
        Err(GrammarError::ParseError(
            "Unterminated string literal".to_string(),
        ))
    }

    fn parse_char_class(&mut self) -> Result<GbnfNode, GrammarError> {
        self.advance(); // consume '['
        let mut negated = false;
        if self.peek() == Some('^') {
            negated = true;
            self.advance();
        }

        let mut ranges = Vec::new();
        while let Some(mut c) = self.advance() {
            if c == ']' {
                return Ok(GbnfNode::CharClass { ranges, negated });
            }

            if c == '\\' {
                if let Some(esc) = self.advance() {
                    c = match esc {
                        'n' => '\n',
                        'r' => '\r',
                        't' => '\t',
                        '\\' => '\\',
                        '-' => '-',
                        ']' => ']',
                        other => other,
                    };
                } else {
                    return Err(GrammarError::ParseError(
                        "Unterminated escape in character class".to_string(),
                    ));
                }
            }

            if self.peek() == Some('-') && self.chars.get(self.pos + 1) != Some(&']') {
                self.advance(); // consume '-'
                if let Some(mut end_c) = self.advance() {
                    if end_c == '\\' {
                        if let Some(esc) = self.advance() {
                            end_c = match esc {
                                'n' => '\n',
                                'r' => '\r',
                                't' => '\t',
                                '\\' => '\\',
                                '-' => '-',
                                ']' => ']',
                                other => other,
                            };
                        } else {
                            return Err(GrammarError::ParseError(
                                "Unterminated escape in character class range".to_string(),
                            ));
                        }
                    }
                    if c > end_c {
                        return Err(GrammarError::ParseError(format!(
                            "Invalid character range: '{}-{}'",
                            c, end_c
                        )));
                    }
                    ranges.push((c, end_c));
                } else {
                    return Err(GrammarError::ParseError(
                        "Unterminated character range".to_string(),
                    ));
                }
            } else {
                ranges.push((c, c));
            }
        }
        Err(GrammarError::ParseError(
            "Unterminated character class".to_string(),
        ))
    }

    fn parse_rule_ref(&mut self) -> Result<GbnfNode, GrammarError> {
        let mut name = String::new();
        while let Some(c) = self.peek() {
            if c.is_alphanumeric() || c == '_' || c == '-' {
                name.push(c);
                self.advance();
            } else {
                break;
            }
        }
        Ok(GbnfNode::RuleRef(name))
    }
}
