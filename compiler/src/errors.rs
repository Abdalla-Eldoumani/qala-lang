//! the compiler's error type. one enum, [`QalaError`], returned everywhere as
//! `Result<T, QalaError>`. every variant carries a [`Span`] so a diagnostic can
//! point at the exact source text; rich rendering (the source line plus an
//! underline) is built in a later phase on top of that span.
//!
//! the lexer's and the parser's variants exist now. type-checker and runtime
//! variants follow the same `{ span, message }` shape and are sketched in
//! comments below so later phases extend this enum rather than restructure it.

use crate::span::Span;
use crate::token::TokenKind;

/// every way the compiler can reject a program, with the source span of the
/// fault. `PartialEq` so tests can compare errors directly; no `Eq` and no
/// `serde` derives, because the parse variants carry [`TokenKind`], which holds
/// an `f64` and is therefore neither `Eq` nor (currently) `serde`-derivable.
/// the diagnostics layer (Phase 3) builds its own structured editor form from
/// `span()` and `message()` rather than serializing this enum directly.
#[derive(Debug, Clone, PartialEq)]
pub enum QalaError {
    // ---- lex ----
    /// a string literal reached a raw newline or end of file before its closing
    /// `"`. span = the opening quote, since that is where the reader's attention
    /// belongs, not the place the scanner ran out of input.
    UnterminatedString { span: Span },

    /// an interpolation `{` inside a string was never closed by a `}` before the
    /// string ended or the file ended. span = the unmatched `{`.
    UnterminatedInterpolation { span: Span },

    /// a backslash escape sequence that the language does not define (anything
    /// other than `\n \t \r \0 \\ \" \{ \}` or a well-formed `\u{...}`). span =
    /// the backslash, not the character after it.
    InvalidEscape { span: Span, message: String },

    /// a byte that cannot begin any token here: a non-ASCII byte outside a
    /// string or comment, or a lone `&` / `|`. span = the offending byte (the
    /// whole UTF-8 sequence for a non-ASCII character). `ch` is the decoded
    /// character, for the message.
    UnexpectedChar { span: Span, ch: char },

    /// an integer literal whose magnitude does not fit in `i64`. span = the
    /// digits.
    IntOverflow { span: Span },

    /// a numeric literal with a malformed shape: a misplaced digit separator
    /// (`1_`, `1__0`, `1_.0`), an empty or invalid radix body (`0x`, `0xG`,
    /// `0b2`), an exponent with no digits (`1e`). span = the literal.
    MalformedNumber { span: Span, message: String },

    /// a byte literal that is not exactly one ASCII character or one one-byte
    /// escape between `b'` and `'` (`b''`, `b'ab'`, `b'\x'`, a non-ASCII byte
    /// inside). span = the literal.
    BadByteLiteral { span: Span, message: String },

    // ---- parse ----
    /// the parser found a token that is not legal at this position. span = the
    /// offending token, or, when `found` is [`TokenKind::Eof`], a zero-width
    /// point just after the last real token. `expected` is the set of token
    /// kinds the parser could have accepted here.
    UnexpectedToken { span: Span, expected: Vec<TokenKind>, found: TokenKind },

    /// an opening delimiter (`(` / `[` / `{`) was closed by the wrong delimiter
    /// or never closed at all. span = the *opening* delimiter, not the surprising
    /// closer, because the opener is where the reader needs to look. `found` is
    /// the wrong closer, or [`TokenKind::Eof`] if the input ran out first.
    UnclosedDelimiter { span: Span, opener: TokenKind, found: TokenKind },

    /// the input ended while the parser was still expecting more. span = a
    /// zero-width point immediately after the last valid token (offset = end of
    /// that token, length 0), not `src.len()` unless that is where the last
    /// token ends. `expected` is what would have been legal next.
    UnexpectedEof { span: Span, expected: Vec<TokenKind> },

    /// a parse failure that does not fit the structured variants above: the
    /// recursion-depth limit ("expression nests too deeply"), a malformed
    /// pipeline right-hand side, and similar. span = wherever the failure was
    /// detected; `message` is the human description.
    Parse { span: Span, message: String },

    // ---- type / runtime: added in later phases, same shape ----
    // keeping these as comments, not code, means later phases extend the enum
    // and teach `span()` the new arms without changing any existing variant.
    // Type    { span: Span, message: String },
    // Runtime { span: Span, message: String },
}

impl QalaError {
    /// the source span this error points at.
    ///
    /// this match is exhaustive over every variant; that is the real guarantee
    /// that "every error carries a span", and the compiler enforces it. a unit
    /// test below documents the intent for a reader, but the type is the proof.
    pub fn span(&self) -> Span {
        match self {
            QalaError::UnterminatedString { span }
            | QalaError::UnterminatedInterpolation { span }
            | QalaError::InvalidEscape { span, .. }
            | QalaError::UnexpectedChar { span, .. }
            | QalaError::IntOverflow { span }
            | QalaError::MalformedNumber { span, .. }
            | QalaError::BadByteLiteral { span, .. }
            | QalaError::UnexpectedToken { span, .. }
            | QalaError::UnclosedDelimiter { span, .. }
            | QalaError::UnexpectedEof { span, .. }
            | QalaError::Parse { span, .. } => *span,
        }
    }

    /// a plain one-line description of the fault.
    ///
    /// no source snippet, no underline, no color: that formatting is a later
    /// phase's job and reads the span this carries. these strings contain no
    /// host paths and no secrets; there are none in scope at this layer.
    pub fn message(&self) -> String {
        match self {
            QalaError::UnterminatedString { .. } => {
                "unterminated string literal".to_string()
            }
            QalaError::UnterminatedInterpolation { .. } => {
                "unterminated interpolation: missing `}`".to_string()
            }
            QalaError::InvalidEscape { message, .. } => {
                format!("invalid escape sequence: {message}")
            }
            QalaError::UnexpectedChar { ch, .. } => {
                format!("unexpected character {ch:?}")
            }
            QalaError::IntOverflow { .. } => {
                "integer literal is too large for i64".to_string()
            }
            QalaError::MalformedNumber { message, .. } => {
                format!("malformed number literal: {message}")
            }
            QalaError::BadByteLiteral { message, .. } => {
                format!("malformed byte literal: {message}")
            }
            QalaError::UnexpectedToken { expected, found, .. } => {
                format!("expected {}, found {}", expected_list(expected), display_kind(found))
            }
            QalaError::UnclosedDelimiter { opener, found, .. } => {
                format!("unclosed {} -- found {}", display_kind(opener), display_kind(found))
            }
            QalaError::UnexpectedEof { expected, .. } => {
                format!("unexpected end of input, expected {}", expected_list(expected))
            }
            QalaError::Parse { message, .. } => message.clone(),
        }
    }
}

/// the human spelling of a token kind, for error messages: `)` not `RParen`,
/// `end of input` for [`TokenKind::Eof`], a category name for the
/// payload-carrying kinds (`an identifier`, `an integer literal`).
///
/// this is the one place the mapping lives, so a new token kind is one line
/// here. the strings are quoted where the token is a literal symbol so a
/// message reads "expected `,` or `)`", not "expected , or )".
pub fn display_kind(kind: &TokenKind) -> &'static str {
    use TokenKind::*;
    match kind {
        // literals and identifiers: a category, not the value.
        Int(_) => "an integer literal",
        Float(_) => "a float literal",
        Byte(_) => "a byte literal",
        Str(_) => "a string literal",
        StrStart(_) => "the start of a string",
        StrMid(_) => "more string text",
        StrEnd(_) => "the end of a string",
        InterpStart => "`{` (interpolation start)",
        InterpEnd => "`}` (interpolation end)",
        Ident(_) => "an identifier",
        // keywords.
        Fn => "`fn`",
        Let => "`let`",
        Mut => "`mut`",
        If => "`if`",
        Else => "`else`",
        While => "`while`",
        For => "`for`",
        In => "`in`",
        Return => "`return`",
        Break => "`break`",
        Continue => "`continue`",
        Defer => "`defer`",
        Match => "`match`",
        Struct => "`struct`",
        Enum => "`enum`",
        Interface => "`interface`",
        Comptime => "`comptime`",
        Is => "`is`",
        Pure => "`pure`",
        Io => "`io`",
        Alloc => "`alloc`",
        Panic => "`panic`",
        Or => "`or`",
        SelfKw => "`self`",
        True => "`true`",
        False => "`false`",
        // primitive type names.
        I64Ty => "`i64`",
        F64Ty => "`f64`",
        BoolTy => "`bool`",
        StrTy => "`str`",
        ByteTy => "`byte`",
        VoidTy => "`void`",
        // operators and punctuation.
        Plus => "`+`",
        Minus => "`-`",
        Star => "`*`",
        Slash => "`/`",
        Percent => "`%`",
        EqEq => "`==`",
        BangEq => "`!=`",
        Lt => "`<`",
        LtEq => "`<=`",
        Gt => "`>`",
        GtEq => "`>=`",
        AmpAmp => "`&&`",
        PipePipe => "`||`",
        Bang => "`!`",
        Eq => "`=`",
        Dot => "`.`",
        Comma => "`,`",
        Colon => "`:`",
        Semi => "`;`",
        LParen => "`(`",
        RParen => "`)`",
        LBracket => "`[`",
        RBracket => "`]`",
        LBrace => "`{`",
        RBrace => "`}`",
        Arrow => "`->`",
        FatArrow => "`=>`",
        PipeGt => "`|>`",
        Question => "`?`",
        DotDot => "`..`",
        DotDotEq => "`..=`",
        Eof => "end of input",
    }
}

/// render an expected-token set for a message: a single item bare, two items
/// joined with `or`, three or more as a comma list ending in `or`. an empty set
/// (which should not happen in practice) reads "something else".
fn expected_list(expected: &[TokenKind]) -> String {
    let names: Vec<&'static str> = expected.iter().map(display_kind).collect();
    match names.as_slice() {
        [] => "something else".to_string(),
        [only] => only.to_string(),
        [a, b] => format!("{a} or {b}"),
        [head @ .., last] => format!("{}, or {}", head.join(", "), last),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // a distinct sample span per variant, so a wrong span() arm would be caught.
    fn sample(n: usize) -> Span {
        Span::new(n, n + 1)
    }

    #[test]
    fn span_returns_the_carried_span_for_every_lex_variant() {
        let cases: Vec<(QalaError, Span)> = vec![
            (QalaError::UnterminatedString { span: sample(1) }, sample(1)),
            (QalaError::UnterminatedInterpolation { span: sample(2) }, sample(2)),
            (
                QalaError::InvalidEscape { span: sample(3), message: "\\q".to_string() },
                sample(3),
            ),
            (QalaError::UnexpectedChar { span: sample(4), ch: '\u{00e9}' }, sample(4)),
            (QalaError::IntOverflow { span: sample(5) }, sample(5)),
            (
                QalaError::MalformedNumber { span: sample(6), message: "1_".to_string() },
                sample(6),
            ),
            (
                QalaError::BadByteLiteral { span: sample(7), message: "b''".to_string() },
                sample(7),
            ),
        ];
        // the exhaustive match in span() is the real guarantee; this loop just
        // documents that each variant round-trips its span.
        for (err, expected) in cases {
            assert_eq!(err.span(), expected, "span() mismatch for {err:?}");
        }
    }

    #[test]
    fn span_returns_the_carried_span_for_every_parse_variant() {
        let cases: Vec<(QalaError, Span)> = vec![
            (
                QalaError::UnexpectedToken {
                    span: sample(1),
                    expected: vec![TokenKind::RParen],
                    found: TokenKind::RBracket,
                },
                sample(1),
            ),
            (
                QalaError::UnclosedDelimiter {
                    span: sample(2),
                    opener: TokenKind::LParen,
                    found: TokenKind::RBrace,
                },
                sample(2),
            ),
            (
                QalaError::UnexpectedEof { span: sample(3), expected: vec![TokenKind::Semi] },
                sample(3),
            ),
            (
                QalaError::Parse { span: sample(4), message: "expression nests too deeply".to_string() },
                sample(4),
            ),
        ];
        for (err, expected) in cases {
            assert_eq!(err.span(), expected, "span() mismatch for {err:?}");
        }
    }

    #[test]
    fn errors_are_comparable_and_clonable() {
        let a = QalaError::IntOverflow { span: sample(0) };
        let b = a.clone();
        assert_eq!(a, b);
        let c = QalaError::IntOverflow { span: sample(1) };
        assert_ne!(a, c);
    }

    #[test]
    fn message_is_a_plain_nonempty_line_per_variant() {
        let errs = [
            QalaError::UnterminatedString { span: sample(0) },
            QalaError::UnterminatedInterpolation { span: sample(0) },
            QalaError::InvalidEscape { span: sample(0), message: "\\q".to_string() },
            QalaError::UnexpectedChar { span: sample(0), ch: '@' },
            QalaError::IntOverflow { span: sample(0) },
            QalaError::MalformedNumber { span: sample(0), message: "1e".to_string() },
            QalaError::BadByteLiteral { span: sample(0), message: "b''".to_string() },
            QalaError::UnexpectedToken {
                span: sample(0),
                expected: vec![TokenKind::Comma, TokenKind::RParen],
                found: TokenKind::RBracket,
            },
            QalaError::UnclosedDelimiter {
                span: sample(0),
                opener: TokenKind::LParen,
                found: TokenKind::Eof,
            },
            QalaError::UnexpectedEof { span: sample(0), expected: vec![TokenKind::Ident(String::new())] },
            QalaError::Parse { span: sample(0), message: "expression nests too deeply".to_string() },
        ];
        for e in &errs {
            let m = e.message();
            assert!(!m.is_empty());
            assert!(!m.contains('\n'), "message should be one line: {m:?}");
        }
    }

    #[test]
    fn unexpected_token_message_lists_every_expected_kind_and_the_found_one() {
        let e = QalaError::UnexpectedToken {
            span: sample(0),
            expected: vec![TokenKind::Comma, TokenKind::RParen],
            found: TokenKind::RBracket,
        };
        let m = e.message();
        // both human spellings appear, and the message says what was actually found.
        assert!(m.contains("`,`"), "missing first expected kind: {m:?}");
        assert!(m.contains("`)`"), "missing second expected kind: {m:?}");
        assert!(m.contains("found"), "missing the word `found`: {m:?}");
        assert!(m.contains("`]`"), "missing the found kind: {m:?}");
    }

    #[test]
    fn unclosed_delimiter_message_names_the_opener_and_the_surprise() {
        // a wrong closer.
        let e = QalaError::UnclosedDelimiter {
            span: sample(0),
            opener: TokenKind::LParen,
            found: TokenKind::RBrace,
        };
        let m = e.message();
        assert!(m.contains("`(`"), "missing the opener: {m:?}");
        assert!(m.contains("`}`"), "missing the surprising token: {m:?}");
        // running into end of input names it as such, not as `Eof`.
        let e = QalaError::UnclosedDelimiter {
            span: sample(0),
            opener: TokenKind::LBracket,
            found: TokenKind::Eof,
        };
        let m = e.message();
        assert!(m.contains("`[`"));
        assert!(m.contains("end of input"));
        assert!(!m.contains("Eof"));
    }

    #[test]
    fn eof_is_named_end_of_input_in_messages() {
        let e = QalaError::UnexpectedEof { span: sample(0), expected: vec![TokenKind::Semi] };
        let m = e.message();
        assert!(m.contains("end of input"), "{m:?}");
        assert!(m.contains("`;`"), "{m:?}");
    }

    #[test]
    fn display_kind_spells_symbols_and_categories() {
        assert_eq!(display_kind(&TokenKind::RParen), "`)`");
        assert_eq!(display_kind(&TokenKind::LBrace), "`{`");
        assert_eq!(display_kind(&TokenKind::Plus), "`+`");
        assert_eq!(display_kind(&TokenKind::Eof), "end of input");
        assert_eq!(display_kind(&TokenKind::Ident("x".to_string())), "an identifier");
        assert_eq!(display_kind(&TokenKind::Int(7)), "an integer literal");
        assert_eq!(display_kind(&TokenKind::Fn), "`fn`");
    }

    #[test]
    fn expected_list_joins_one_two_or_many() {
        assert_eq!(expected_list(&[TokenKind::RParen]), "`)`");
        assert_eq!(expected_list(&[TokenKind::Comma, TokenKind::RParen]), "`,` or `)`");
        assert_eq!(
            expected_list(&[TokenKind::Comma, TokenKind::RParen, TokenKind::RBracket]),
            "`,`, `)`, or `]`"
        );
        // an empty set should not occur, but it must not panic.
        assert_eq!(expected_list(&[]), "something else");
    }
}
