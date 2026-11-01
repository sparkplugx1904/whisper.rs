//! GBNF Grammar-guided constrained sampling for whisper.rs.

use std::collections::HashMap;
use thiserror::Error;
use whisper_model::Vocab;

#[derive(Error, Debug)]
pub enum GrammarError {
    #[error("Parse error: {0}")]
    ParseError(String),
}

#[derive(Debug, Clone)]
pub enum GbnfNode {
    Literal(String),
    RuleRef(String),
    Alternation(Vec<GbnfNode>),
    Sequence(Vec<GbnfNode>),
    Optional(Box<GbnfNode>),
    Repeat(Box<GbnfNode>),
}

#[derive(Debug, Clone)]
pub struct GbnfGrammar {
    pub rules: HashMap<String, GbnfNode>,
    pub root: String,
}

impl GbnfGrammar {
    /// Parse a GBNF grammar from string specification.
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
                let node = Self::parse_expr(expr_str);
                rules.insert(name, node);
            }
        }

        if rules.is_empty() {
            return Err(GrammarError::ParseError("No valid rules found".to_string()));
        }

        Ok(Self { rules, root })
    }

    fn parse_expr(expr: &str) -> GbnfNode {
        let expr = expr.trim();
        // Check alternations: A | B
        if expr.contains('|') {
            let alts: Vec<GbnfNode> = expr
                .split('|')
                .map(|s| Self::parse_term(s.trim()))
                .collect();
            return GbnfNode::Alternation(alts);
        }
        Self::parse_term(expr)
    }

    fn parse_term(term: &str) -> GbnfNode {
        let term = term.trim();
        if term.starts_with('"') && term.ends_with('"') && term.len() >= 2 {
            let lit = &term[1..term.len() - 1];
            return GbnfNode::Literal(lit.to_string());
        }
        if let Some(stripped) = term.strip_suffix('?') {
            let inner = Self::parse_term(stripped);
            return GbnfNode::Optional(Box::new(inner));
        }
        if let Some(stripped) = term.strip_suffix('*') {
            let inner = Self::parse_term(stripped);
            return GbnfNode::Repeat(Box::new(inner));
        }
        GbnfNode::RuleRef(term.to_string())
    }

    /// Check if the grammar can accept a prefix string.
    pub fn accepts_prefix(&self, text: &str) -> bool {
        self.match_rule(&self.root, text, 0).0
    }

    fn match_rule(&self, rule_name: &str, text: &str, pos: usize) -> (bool, usize) {
        if let Some(node) = self.rules.get(rule_name) {
            self.match_node(node, text, pos)
        } else {
            (true, pos)
        }
    }

    fn match_node(&self, node: &GbnfNode, text: &str, pos: usize) -> (bool, usize) {
        if pos >= text.len() {
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
                let mut cur_pos = pos;
                for item in seq {
                    let (ok, next_pos) = self.match_node(item, text, cur_pos);
                    if !ok {
                        return (false, pos);
                    }
                    cur_pos = next_pos;
                }
                (true, cur_pos)
            }
            GbnfNode::Optional(inner) => {
                let (ok, next_pos) = self.match_node(inner, text, pos);
                if ok {
                    (true, next_pos)
                } else {
                    (true, pos)
                }
            }
            GbnfNode::Repeat(inner) => {
                let mut cur_pos = pos;
                while cur_pos < text.len() {
                    let (ok, next_pos) = self.match_node(inner, text, cur_pos);
                    if !ok || next_pos == cur_pos {
                        break;
                    }
                    cur_pos = next_pos;
                }
                (true, cur_pos)
            }
        }
    }

    /// Mask invalid token logits with -INFINITY based on GBNF grammar constraints.
    pub fn filter_logits(&self, current_text: &str, logits: &mut [f32], vocab: &Vocab) {
        for (tok_id, val) in logits.iter_mut().enumerate() {
            if *val == f32::NEG_INFINITY {
                continue;
            }

            if let Some(tok_str) = vocab.id_to_token(tok_id as i32) {
                let mut candidate_text = current_text.to_string();
                candidate_text.push_str(tok_str);

                if !self.accepts_prefix(&candidate_text) {
                    *val = f32::NEG_INFINITY;
                }
            }
        }
    }
}
