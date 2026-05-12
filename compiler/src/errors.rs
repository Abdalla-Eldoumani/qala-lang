//! the compiler's error type. one enum, [`QalaError`], returned everywhere as
//! `Result<T, QalaError>`. every variant carries a [`Span`] so a diagnostic can
//! point at the exact source text; rich rendering (the source line plus an
//! underline) is built in a later phase on top of that span.
//!
//! only the lexer's variants exist now. parser, type-checker, and runtime
//! variants follow the same `{ span, message }` shape and are sketched in
//! comments below so later phases extend this enum rather than restructure it.

use crate::span::Span;

/// every way the compiler can reject a program, with the source span of the
/// fault. `serde` derives so the diagnostics layer can later serialize a
/// structured form to the editor without touching this enum; `PartialEq` /
/// `Eq` so tests can compare errors directly.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum QalaError {
    // ---- lex (this phase) ----
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

    // ---- parse / type / runtime: added in later phases, same shape ----
    // keeping these as comments, not code, means later phases extend the enum
    // and teach `span()` the new arms without changing any existing variant.
    // Parse   { span: Span, message: String },
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
            | QalaError::BadByteLiteral { span, .. } => *span,
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
        }
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
        ];
        for e in &errs {
            let m = e.message();
            assert!(!m.is_empty());
            assert!(!m.contains('\n'), "message should be one line: {m:?}");
        }
    }
}
