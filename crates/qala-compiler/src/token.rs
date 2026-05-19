//! tokens: the units the lexer produces and the parser consumes. [`TokenKind`]
//! is the classification (with a payload for literals and identifiers), [`Token`]
//! pairs a kind with a [`Span`]. the stream always ends in an explicit
//! [`TokenKind::Eof`].
//!
//! the operator set is deliberately one variant per operator, even where two
//! operators share a prefix, so the Pratt parser can match a kind and look up
//! its precedence without re-inspecting source text.

use crate::span::Span;

/// what a token is. literals and identifiers carry their decoded value; every
/// keyword, every operator, every piece of punctuation, and end of file are
/// nullary variants.
///
/// `PartialEq` but not `Eq`, because [`TokenKind::Float`] holds an `f64` and
/// `f64` is not `Eq`. that is fine; nothing needs `TokenKind: Eq`.
#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    // ---- literals (payload carried) ----
    /// an integer literal, already parsed (decimal, `0x`, or `0b`; underscores
    /// stripped). stored as the non-negative magnitude; a leading `-` is a
    /// separate token the parser folds in.
    Int(i64),
    /// a float literal, already parsed (underscores stripped, exponent applied).
    Float(f64),
    /// a byte literal `b'X'`, the single byte it denotes.
    Byte(u8),
    /// a string literal with no interpolation, escapes already decoded.
    Str(String),
    /// the decoded text before the first interpolation in a string with
    /// interpolations (may be empty). followed by an [`TokenKind::InterpStart`].
    StrStart(String),
    /// the decoded text between one interpolation's `}` and the next one's `{`
    /// (may be empty). sits between an [`TokenKind::InterpEnd`] and an
    /// [`TokenKind::InterpStart`].
    StrMid(String),
    /// the decoded text after the last interpolation in a string (may be empty).
    /// follows an [`TokenKind::InterpEnd`] and closes the string.
    StrEnd(String),
    /// opens an embedded interpolation expression. between this and the matching
    /// [`TokenKind::InterpEnd`] the lexer emits ordinary tokens, so any
    /// expression (including a nested string) works inside `{ ... }`.
    InterpStart,
    /// closes the interpolation expression opened by [`TokenKind::InterpStart`].
    InterpEnd,
    /// an identifier: ASCII `[A-Za-z_][A-Za-z0-9_]*` that is not a keyword.
    Ident(String),

    // ---- keywords (reserved words and primitive type names) ----
    Fn,
    Let,
    Mut,
    If,
    Else,
    While,
    For,
    In,
    Return,
    Break,
    Continue,
    Defer,
    Match,
    Struct,
    Enum,
    Interface,
    Comptime,
    Is,
    Pure,
    Io,
    Alloc,
    Panic,
    Or,
    /// the `self` keyword. named `SelfKw` because `Self` and `self` are reserved
    /// in Rust and cannot be used as identifiers here.
    SelfKw,
    /// the boolean literal `true`. lexed as a keyword, not an identifier.
    True,
    /// the boolean literal `false`. lexed as a keyword, not an identifier.
    False,
    /// the primitive type name `i64`.
    I64Ty,
    /// the primitive type name `f64`.
    F64Ty,
    /// the primitive type name `bool`.
    BoolTy,
    /// the primitive type name `str`.
    StrTy,
    /// the primitive type name `byte`.
    ByteTy,
    /// the primitive type name `void`.
    VoidTy,

    // ---- operators and punctuation (one variant per operator) ----
    /// `+`
    Plus,
    /// `-`
    Minus,
    /// `*`
    Star,
    /// `/`
    Slash,
    /// `%`
    Percent,
    /// `==`
    EqEq,
    /// `!=`
    BangEq,
    /// `<`
    Lt,
    /// `<=`
    LtEq,
    /// `>`
    Gt,
    /// `>=`
    GtEq,
    /// `&&`
    AmpAmp,
    /// `||`
    PipePipe,
    /// `!`
    Bang,
    /// `=`
    Eq,
    /// `.`
    Dot,
    /// `,`
    Comma,
    /// `:`
    Colon,
    /// `;`
    Semi,
    /// `(`
    LParen,
    /// `)`
    RParen,
    /// `[`
    LBracket,
    /// `]`
    RBracket,
    /// `{`
    LBrace,
    /// `}`
    RBrace,
    /// `->` (return-type arrow)
    Arrow,
    /// `=>` (match-arm arrow)
    FatArrow,
    /// `|>` (pipeline)
    PipeGt,
    /// `?` (postfix error propagation)
    Question,
    /// `..` (exclusive range)
    DotDot,
    /// `..=` (inclusive range)
    DotDotEq,

    /// end of the token stream.
    Eof,
}

/// a token: its kind and the source span it covers.
#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    /// the classification and payload.
    pub kind: TokenKind,
    /// the source region this token spans, opening to closing byte inclusive.
    pub span: Span,
}

impl Token {
    /// build a token from a kind and a span.
    pub fn new(kind: TokenKind, span: Span) -> Self {
        Token { kind, span }
    }
}

/// the keyword for an identifier string, or `None` if it is an ordinary
/// identifier.
///
/// one `match` in one place: easy to audit against the language's reserved-word
/// list, and the compiler turns a small string match into efficient code. the
/// lexer carries no standard-library knowledge: names like `Result`, `Option`,
/// `Ok`, `Err`, `Some`, `None`, and the built-in functions `print`, `println`,
/// `len`, `push`, `pop`, `sqrt`, `abs`, `assert`, `type_of`, `open`, `close`,
/// `map`, `filter`, `reduce` are NOT keywords; the resolver and type-checker
/// know them, the scanner does not.
pub fn keyword(ident: &str) -> Option<TokenKind> {
    use TokenKind::*;
    Some(match ident {
        "fn" => Fn,
        "let" => Let,
        "mut" => Mut,
        "if" => If,
        "else" => Else,
        "while" => While,
        "for" => For,
        "in" => In,
        "return" => Return,
        "break" => Break,
        "continue" => Continue,
        "defer" => Defer,
        "match" => Match,
        "struct" => Struct,
        "enum" => Enum,
        "interface" => Interface,
        "comptime" => Comptime,
        "is" => Is,
        "pure" => Pure,
        "io" => Io,
        "alloc" => Alloc,
        "panic" => Panic,
        "or" => Or,
        "self" => SelfKw,
        "true" => True,
        "false" => False,
        // primitive type names are reserved too.
        "i64" => I64Ty,
        "f64" => F64Ty,
        "bool" => BoolTy,
        "str" => StrTy,
        "byte" => ByteTy,
        "void" => VoidTy,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reserved_words_lex_to_their_keyword_kind() {
        let cases: &[(&str, TokenKind)] = &[
            ("fn", TokenKind::Fn),
            ("let", TokenKind::Let),
            ("mut", TokenKind::Mut),
            ("if", TokenKind::If),
            ("else", TokenKind::Else),
            ("while", TokenKind::While),
            ("for", TokenKind::For),
            ("in", TokenKind::In),
            ("return", TokenKind::Return),
            ("break", TokenKind::Break),
            ("continue", TokenKind::Continue),
            ("defer", TokenKind::Defer),
            ("match", TokenKind::Match),
            ("struct", TokenKind::Struct),
            ("enum", TokenKind::Enum),
            ("interface", TokenKind::Interface),
            ("comptime", TokenKind::Comptime),
            ("is", TokenKind::Is),
            ("pure", TokenKind::Pure),
            ("io", TokenKind::Io),
            ("alloc", TokenKind::Alloc),
            ("panic", TokenKind::Panic),
            ("or", TokenKind::Or),
            ("self", TokenKind::SelfKw),
        ];
        for (src, expected) in cases {
            assert_eq!(keyword(src), Some(expected.clone()), "keyword({src:?})");
        }
    }

    #[test]
    fn true_and_false_are_boolean_keyword_kinds_not_idents() {
        assert_eq!(keyword("true"), Some(TokenKind::True));
        assert_eq!(keyword("false"), Some(TokenKind::False));
    }

    #[test]
    fn primitive_type_names_are_keyword_kinds() {
        assert_eq!(keyword("i64"), Some(TokenKind::I64Ty));
        assert_eq!(keyword("f64"), Some(TokenKind::F64Ty));
        assert_eq!(keyword("bool"), Some(TokenKind::BoolTy));
        assert_eq!(keyword("str"), Some(TokenKind::StrTy));
        assert_eq!(keyword("byte"), Some(TokenKind::ByteTy));
        assert_eq!(keyword("void"), Some(TokenKind::VoidTy));
    }

    #[test]
    fn stdlib_and_result_family_names_are_not_keywords() {
        // the lexer carries no stdlib knowledge: these are ordinary identifiers.
        let not_keywords = [
            "Result", "Option", "Ok", "Err", "Some", "None", "print", "println", "len", "push",
            "pop", "sqrt", "abs", "assert", "type_of", "open", "close", "map", "filter", "reduce",
        ];
        for name in not_keywords {
            assert_eq!(keyword(name), None, "{name:?} must not be a keyword");
        }
    }

    #[test]
    fn ordinary_identifiers_are_not_keywords() {
        for name in ["foo", "_x", "_", "x1", "__", "fooBar", "Fn", "LET"] {
            assert_eq!(keyword(name), None, "{name:?} must not be a keyword");
        }
    }

    #[test]
    fn a_token_pairs_a_kind_with_a_span() {
        let tok = Token::new(TokenKind::Plus, Span::new(3, 1));
        assert_eq!(tok.kind, TokenKind::Plus);
        assert_eq!(tok.span, Span::new(3, 1));
    }

    #[test]
    fn there_is_an_eof_kind() {
        // the lexer emits this at the end of every token stream.
        let eof = Token::new(TokenKind::Eof, Span::new(0, 0));
        assert_eq!(eof.kind, TokenKind::Eof);
    }

    #[test]
    fn literal_kinds_carry_their_payload() {
        assert_eq!(TokenKind::Int(42), TokenKind::Int(42));
        assert_ne!(TokenKind::Int(42), TokenKind::Int(43));
        assert_eq!(TokenKind::Byte(b'A'), TokenKind::Byte(65));
        assert_eq!(
            TokenKind::Str("abc".to_string()),
            TokenKind::Str("abc".to_string())
        );
        assert_eq!(
            TokenKind::StrStart("hi ".to_string()),
            TokenKind::StrStart("hi ".to_string())
        );
        assert_eq!(
            TokenKind::Ident("x".to_string()),
            TokenKind::Ident("x".to_string())
        );
        // float carries an f64; PartialEq works, Eq is intentionally absent.
        assert_eq!(TokenKind::Float(1.5), TokenKind::Float(1.5));
    }
}
