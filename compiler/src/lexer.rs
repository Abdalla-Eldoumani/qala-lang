//! the hand-written scanner: [`Lexer::tokenize`] turns Qala source text into a
//! `Vec<Token>` ending in [`TokenKind::Eof`], or the first lex error.
//!
//! it is a single pass over `src.as_bytes()`: identifiers, keywords, operators,
//! punctuation, and digit prefixes are all ASCII, so byte-level peeking is enough
//! everywhere except inside string and comment bodies (where a multi-byte char is
//! decoded only to know how far to advance and what to put in the error). every
//! source index is a checked accessor or a position this scanner produced, and no
//! malformed input reaches an `unwrap`/`expect`/`panic!`/`unreachable!` -- that
//! discipline is load-bearing for Phase 6, where this code runs in WASM and a
//! panic aborts the module.
//!
//! the lexer is fail-fast: it returns `Result<Vec<Token>, QalaError>` and stops
//! at the first error. multi-error reporting is deferred to Phase 2's parser,
//! which has resync points; the lexer does not.

use crate::errors::QalaError;
use crate::span::Span;
use crate::token::{Token, TokenKind};

/// which body the scanner is inside. pushed on a small stack so interpolations
/// can nest (`"{ "{a}" }"` is a string inside an interpolation inside a string).
///
/// the offsets carried by `StrText` and `Interp` are not bookkeeping for parsing
/// -- they are the *cause* offsets for error spans: an unterminated string points
/// at its opening quote, an unterminated interpolation at its `{`, even though the
/// scanner only notices the problem later (at EOF or the string's end).
enum Mode {
    /// ordinary token scanning. the bottom of the stack is always this.
    Normal,
    /// inside a string literal's text. carries the byte offset of the opening
    /// quote, for the unterminated-string error span. `saw_interp` is true once
    /// this string has emitted at least one interpolation, so the next text run
    /// closes with `StrEnd` (not `Str`) and an inner run opens with `StrMid` (not
    /// `StrStart`).
    StrText { open_quote: usize, saw_interp: bool },
    /// inside an interpolation expression opened from a string. `brace_depth`
    /// counts `{`/`}` *within* this interpolation (nested blocks, struct
    /// literals); only the depth-0 `}` closes it. `open_brace` is the offset of
    /// the `{` that opened it, for the unterminated-interpolation error span.
    Interp { brace_depth: u32, open_brace: usize },
}

/// the scanner state: the source, its bytes (for cheap ASCII peeking), the
/// current byte offset, and the mode stack.
pub struct Lexer<'src> {
    src: &'src str,
    bytes: &'src [u8],
    pos: usize,
    /// starts as `[Mode::Normal]`; never empties (the bottom `Normal` stays).
    mode_stack: Vec<Mode>,
}

impl<'src> Lexer<'src> {
    /// scan `src` into a token stream ending in [`TokenKind::Eof`], or the first
    /// lex error.
    ///
    /// fail-fast: the first malformed token aborts with its `QalaError`. an empty,
    /// whitespace-only, or comment-only source yields exactly `[Eof]` and no error.
    /// a leading UTF-8 BOM is skipped silently.
    pub fn tokenize(src: &'src str) -> Result<Vec<Token>, QalaError> {
        let mut lx = Lexer {
            src,
            bytes: src.as_bytes(),
            pos: 0,
            mode_stack: vec![Mode::Normal],
        };
        lx.skip_bom();
        let mut out = Vec::new();
        loop {
            // a string body scans differently: no trivia skipping, no maximal
            // munch -- copy/decode characters until the close quote or an
            // interpolation `{`. the string scan pushes one or more tokens itself
            // (Str, or StrStart/StrMid/StrEnd around the interpolation markers).
            if matches!(lx.mode_stack.last(), Some(Mode::StrText { .. })) {
                lx.scan_string_body(&mut out)?;
                continue;
            }
            lx.skip_trivia();
            let start = lx.pos;
            match lx.peek() {
                None => {
                    // a string or interpolation still open at EOF is the cause
                    // of an unterminated error; point at where it opened.
                    if let Some(err) = lx.unterminated_at_eof() {
                        return Err(err);
                    }
                    out.push(Token::new(TokenKind::Eof, Span::new(start, 0)));
                    break;
                }
                // a `"` opens a string literal: push StrText and let the next
                // iteration's string-body branch produce the token(s). handled
                // here rather than in scan_token because a string is not a single
                // token -- an interpolated one is several.
                Some(b'"') => {
                    lx.pos += 1; // consume the opening quote
                    lx.mode_stack
                        .push(Mode::StrText { open_quote: start, saw_interp: false });
                }
                Some(b) => {
                    let kind = lx.scan_token(b, start)?;
                    out.push(Token::new(kind, Span::new(start, lx.pos - start)));
                }
            }
        }
        Ok(out)
    }

    // ---- peeking and advancing -------------------------------------------------

    /// the byte at the cursor, or `None` at end of input.
    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    /// the byte one past the cursor, or `None`.
    fn peek2(&self) -> Option<u8> {
        self.bytes.get(self.pos + 1).copied()
    }

    /// the byte at the cursor; advances the cursor past it if there is one.
    fn bump(&mut self) -> Option<u8> {
        let b = self.peek();
        if b.is_some() {
            self.pos += 1;
        }
        b
    }

    // ---- trivia ---------------------------------------------------------------

    /// strip exactly the 3-byte UTF-8 BOM (`EF BB BF`) if the source starts with
    /// it. only at offset 0, only once.
    fn skip_bom(&mut self) {
        if self.src.starts_with('\u{FEFF}') {
            self.pos += '\u{FEFF}'.len_utf8();
        }
    }

    /// skip ASCII whitespace and `//` line comments until the next real byte.
    /// `//` runs to (not including) the next `\n`; a line that is only `//` is
    /// fine; there are no block comments, so `/x` is `Slash` then `x`.
    fn skip_trivia(&mut self) {
        loop {
            match self.peek() {
                Some(b) if b.is_ascii_whitespace() => {
                    self.pos += 1;
                }
                Some(b'/') if self.peek2() == Some(b'/') => {
                    // consume "//" then everything to end of line.
                    self.pos += 2;
                    while let Some(b) = self.peek() {
                        if b == b'\n' {
                            break;
                        }
                        self.pos += 1;
                    }
                }
                _ => break,
            }
        }
    }

    // ---- the dispatcher -------------------------------------------------------

    /// classify the token starting at `b` (the byte at `start == self.pos`) and
    /// advance the cursor past it. returns the token kind, or a lex error.
    ///
    /// `"` is never seen here -- the main loop handles a string opening before
    /// calling this, because an interpolated string is more than one token.
    fn scan_token(&mut self, b: u8, start: usize) -> Result<TokenKind, QalaError> {
        // an identifier or keyword: ASCII letter or `_`, then letters/digits/`_`.
        if b == b'_' || b.is_ascii_alphabetic() {
            // a `b` immediately followed by `'` is a byte literal, not an
            // identifier; everything else starting with `b` (`by`, `b 1`, `b`)
            // is the identifier `b`.
            if b == b'b' && self.peek2() == Some(b'\'') {
                return self.scan_byte_literal(start);
            }
            return Ok(self.scan_identifier());
        }
        if b.is_ascii_digit() {
            return self.scan_number(start);
        }
        // a brace while an interpolation is on top does brace-depth bookkeeping.
        if (b == b'{' || b == b'}')
            && matches!(self.mode_stack.last(), Some(Mode::Interp { .. }))
        {
            return Ok(self.scan_brace_in_interp(b));
        }
        match b {
            b'(' => { self.pos += 1; Ok(TokenKind::LParen) }
            b')' => { self.pos += 1; Ok(TokenKind::RParen) }
            b'[' => { self.pos += 1; Ok(TokenKind::LBracket) }
            b']' => { self.pos += 1; Ok(TokenKind::RBracket) }
            b'{' => { self.pos += 1; Ok(TokenKind::LBrace) }
            b'}' => { self.pos += 1; Ok(TokenKind::RBrace) }
            b',' => { self.pos += 1; Ok(TokenKind::Comma) }
            b':' => { self.pos += 1; Ok(TokenKind::Colon) }
            b';' => { self.pos += 1; Ok(TokenKind::Semi) }
            b'+' => { self.pos += 1; Ok(TokenKind::Plus) }
            b'*' => { self.pos += 1; Ok(TokenKind::Star) }
            b'%' => { self.pos += 1; Ok(TokenKind::Percent) }
            b'?' => { self.pos += 1; Ok(TokenKind::Question) }
            // `/` here is always division: `//` was consumed by skip_trivia.
            b'/' => { self.pos += 1; Ok(TokenKind::Slash) }
            // `.` `..` `..=` -- never the start of a float (a number only starts
            // on a digit), so this is unambiguous.
            b'.' => {
                self.pos += 1;
                if self.peek() == Some(b'.') {
                    self.pos += 1;
                    if self.peek() == Some(b'=') {
                        self.pos += 1;
                        Ok(TokenKind::DotDotEq)
                    } else {
                        Ok(TokenKind::DotDot)
                    }
                } else {
                    Ok(TokenKind::Dot)
                }
            }
            b'-' => {
                self.pos += 1;
                if self.peek() == Some(b'>') {
                    self.pos += 1;
                    Ok(TokenKind::Arrow)
                } else {
                    Ok(TokenKind::Minus)
                }
            }
            b'=' => {
                self.pos += 1;
                match self.peek() {
                    Some(b'=') => { self.pos += 1; Ok(TokenKind::EqEq) }
                    Some(b'>') => { self.pos += 1; Ok(TokenKind::FatArrow) }
                    _ => Ok(TokenKind::Eq),
                }
            }
            b'!' => {
                self.pos += 1;
                if self.peek() == Some(b'=') {
                    self.pos += 1;
                    Ok(TokenKind::BangEq)
                } else {
                    Ok(TokenKind::Bang)
                }
            }
            b'<' => {
                self.pos += 1;
                if self.peek() == Some(b'=') {
                    self.pos += 1;
                    Ok(TokenKind::LtEq)
                } else {
                    Ok(TokenKind::Lt)
                }
            }
            b'>' => {
                self.pos += 1;
                if self.peek() == Some(b'=') {
                    self.pos += 1;
                    Ok(TokenKind::GtEq)
                } else {
                    Ok(TokenKind::Gt)
                }
            }
            // `&&` is the only `&` form; a lone `&` is rejected (no bitwise `&`,
            // no reference syntax in v1).
            b'&' => {
                if self.peek2() == Some(b'&') {
                    self.pos += 2;
                    Ok(TokenKind::AmpAmp)
                } else {
                    let span = Span::new(start, 1);
                    self.pos += 1;
                    Err(QalaError::UnexpectedChar { span, ch: '&' })
                }
            }
            // `||` and `|>` are the only `|` forms; a lone `|` is rejected (no
            // bitwise `|`, no pattern alternatives in v1).
            b'|' => match self.peek2() {
                Some(b'|') => { self.pos += 2; Ok(TokenKind::PipePipe) }
                Some(b'>') => { self.pos += 2; Ok(TokenKind::PipeGt) }
                _ => {
                    let span = Span::new(start, 1);
                    self.pos += 1;
                    Err(QalaError::UnexpectedChar { span, ch: '|' })
                }
            },
            // any non-ASCII byte outside a string or comment cannot begin a
            // token. decode the whole character (so we know how far to advance
            // and what to report), advance past it, and error.
            _ if b >= 0x80 => {
                let ch = self.src[self.pos..].chars().next().unwrap_or('\u{FFFD}');
                let len = ch.len_utf8();
                let span = Span::new(start, len);
                self.pos += len;
                Err(QalaError::UnexpectedChar { span, ch })
            }
            // a stray ASCII control or symbol byte we do not lex (`@`, `#`, `$`,
            // `` ` ``, `~`, `^`, `\`, `'` outside a byte literal, ...).
            _ => {
                let span = Span::new(start, 1);
                self.pos += 1;
                Err(QalaError::UnexpectedChar { span, ch: b as char })
            }
        }
    }

    /// scan `[A-Za-z_][A-Za-z0-9_]*` from the cursor and classify it: a reserved
    /// word becomes its keyword kind, anything else becomes [`TokenKind::Ident`].
    fn scan_identifier(&mut self) -> TokenKind {
        let start = self.pos;
        while let Some(b) = self.peek() {
            if b == b'_' || b.is_ascii_alphanumeric() {
                self.pos += 1;
            } else {
                break;
            }
        }
        let text = &self.src[start..self.pos];
        match crate::token::keyword(text) {
            Some(kw) => kw,
            None => TokenKind::Ident(text.to_string()),
        }
    }

    /// brace-depth bookkeeping while an interpolation is the top mode. `{` deepens
    /// it; `}` either un-deepens a nested block or, at depth 0, closes the
    /// interpolation (emit [`TokenKind::InterpEnd`], pop back to the enclosing
    /// `StrText`).
    fn scan_brace_in_interp(&mut self, b: u8) -> TokenKind {
        self.pos += 1; // consume the brace
        match b {
            b'{' => {
                if let Some(Mode::Interp { brace_depth, .. }) = self.mode_stack.last_mut() {
                    *brace_depth += 1;
                }
                TokenKind::LBrace
            }
            // b'}'
            _ => {
                let at_depth_zero = matches!(
                    self.mode_stack.last(),
                    Some(Mode::Interp { brace_depth: 0, .. })
                );
                if at_depth_zero {
                    self.mode_stack.pop(); // leave the Interp; the StrText below resumes
                    TokenKind::InterpEnd
                } else {
                    if let Some(Mode::Interp { brace_depth, .. }) = self.mode_stack.last_mut() {
                        // depth > 0 here, so this does not underflow.
                        *brace_depth -= 1;
                    }
                    TokenKind::RBrace
                }
            }
        }
    }

    // ---- stubs filled by later tasks ------------------------------------------

    /// scan a numeric literal. Task 2 owns the real implementation; for now this
    /// scans a bare run of ASCII digits so the skeleton's Eof/operator tests pass.
    fn scan_number(&mut self, _start: usize) -> Result<TokenKind, QalaError> {
        let s = self.pos;
        while let Some(b) = self.peek() {
            if b.is_ascii_digit() {
                self.pos += 1;
            } else {
                break;
            }
        }
        // best-effort placeholder; Task 2 replaces this whole function.
        let value: i64 = self.src[s..self.pos].parse().unwrap_or(0);
        Ok(TokenKind::Int(value))
    }

    /// scan a `b'X'` byte literal. Task 2 owns the real implementation; for now
    /// this is unreachable because Task 1's dispatch never calls it (the dispatch
    /// check is in place, but no test exercises a byte literal yet).
    fn scan_byte_literal(&mut self, _start: usize) -> Result<TokenKind, QalaError> {
        // Task 2 replaces this. Until then, treat `b'` as the identifier `b`
        // followed by whatever comes next, by not consuming the quote here.
        Ok(self.scan_identifier())
    }

    /// scan a string literal's body once `StrText` is the top mode. Task 3 owns
    /// the real implementation (escapes, `\u{...}`, interpolation grouping); this
    /// Task 1 stub copies raw bytes to the next `"` and emits a single `Str`, so
    /// the skeleton does not hang and a simple `"abc"` round-trips. it does not
    /// decode escapes and treats `{` literally -- Task 3 replaces it wholesale.
    fn scan_string_body(&mut self, out: &mut Vec<Token>) -> Result<(), QalaError> {
        let open_quote = match self.mode_stack.last() {
            Some(Mode::StrText { open_quote, .. }) => *open_quote,
            _ => return Ok(()), // not in a string; nothing to do
        };
        let text_start = self.pos;
        while let Some(b) = self.peek() {
            if b == b'"' {
                let text = self.src[text_start..self.pos].to_string();
                self.pos += 1; // consume the closing quote
                self.mode_stack.pop(); // leave StrText
                // span: opening quote through closing quote inclusive.
                let span = Span::new(open_quote, self.pos - open_quote);
                out.push(Token::new(TokenKind::Str(text), span));
                return Ok(());
            }
            if b == b'\n' {
                break; // no multi-line strings; fall through to the error
            }
            self.pos += 1;
        }
        // hit EOF or a newline before the closing quote.
        Err(QalaError::UnterminatedString { span: Span::new(open_quote, 1) })
    }

    /// if EOF was reached with a string or interpolation still open, return the
    /// matching unterminated error pointing at where it opened.
    ///
    /// in Task 1 a string is fully consumed (or errored) inside `scan_string_body`
    /// before this is reached, and interpolations are not yet opened, so any mode
    /// above the bottom `Normal` here means an unterminated form. Task 3 keeps
    /// this contract.
    fn unterminated_at_eof(&self) -> Option<QalaError> {
        match self.mode_stack.last() {
            Some(Mode::StrText { open_quote, .. }) => {
                Some(QalaError::UnterminatedString { span: Span::new(*open_quote, 1) })
            }
            Some(Mode::Interp { open_brace, .. }) => Some(QalaError::UnterminatedInterpolation {
                span: Span::new(*open_brace, 1),
            }),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// tokenize, asserting success, and return the kinds (dropping spans) for
    /// terse stream comparisons.
    fn kinds(src: &str) -> Vec<TokenKind> {
        Lexer::tokenize(src)
            .expect("expected successful tokenize")
            .into_iter()
            .map(|t| t.kind)
            .collect()
    }

    /// tokenize, asserting it failed, and return the error.
    fn err(src: &str) -> QalaError {
        Lexer::tokenize(src).expect_err("expected a lex error")
    }

    #[test]
    fn empty_source_is_just_eof_at_offset_zero() {
        let toks = Lexer::tokenize("").unwrap();
        assert_eq!(toks.len(), 1);
        assert_eq!(toks[0].kind, TokenKind::Eof);
        assert_eq!(toks[0].span, Span::new(0, 0));
    }

    #[test]
    fn whitespace_only_source_is_just_eof() {
        assert_eq!(kinds("   \n\t \r\n"), vec![TokenKind::Eof]);
    }

    #[test]
    fn comment_only_source_is_just_eof() {
        assert_eq!(kinds("// hi\n  \n"), vec![TokenKind::Eof]);
        assert_eq!(kinds("// just a comment, no newline"), vec![TokenKind::Eof]);
        // a line that is only "//" is fine.
        assert_eq!(kinds("//\n"), vec![TokenKind::Eof]);
    }

    #[test]
    fn whitespace_and_comments_mixed_is_just_eof() {
        assert_eq!(kinds("   \n\t // hi\n   // more\n  "), vec![TokenKind::Eof]);
    }

    #[test]
    fn leading_bom_is_skipped_silently() {
        let toks = kinds("\u{FEFF}fn main(){}");
        assert_eq!(
            toks,
            vec![
                TokenKind::Fn,
                TokenKind::Ident("main".to_string()),
                TokenKind::LParen,
                TokenKind::RParen,
                TokenKind::LBrace,
                TokenKind::RBrace,
                TokenKind::Eof,
            ]
        );
        // a bare BOM file is still just Eof, no BOM token.
        assert_eq!(kinds("\u{FEFF}"), vec![TokenKind::Eof]);
    }

    #[test]
    fn identifiers_lex_with_their_text() {
        for name in ["_x", "x1", "__", "fooBar", "_", "main", "abc123", "X"] {
            assert_eq!(
                kinds(name),
                vec![TokenKind::Ident(name.to_string()), TokenKind::Eof],
                "identifier {name:?}"
            );
        }
    }

    #[test]
    fn reserved_words_lex_to_keyword_kinds_not_idents() {
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
        for (src, kind) in cases {
            assert_eq!(kinds(src), vec![kind.clone(), TokenKind::Eof], "keyword {src:?}");
        }
    }

    #[test]
    fn true_and_false_are_boolean_keyword_kinds() {
        assert_eq!(kinds("true"), vec![TokenKind::True, TokenKind::Eof]);
        assert_eq!(kinds("false"), vec![TokenKind::False, TokenKind::Eof]);
    }

    #[test]
    fn primitive_type_names_are_keyword_kinds() {
        assert_eq!(kinds("i64"), vec![TokenKind::I64Ty, TokenKind::Eof]);
        assert_eq!(kinds("f64"), vec![TokenKind::F64Ty, TokenKind::Eof]);
        assert_eq!(kinds("bool"), vec![TokenKind::BoolTy, TokenKind::Eof]);
        assert_eq!(kinds("str"), vec![TokenKind::StrTy, TokenKind::Eof]);
        assert_eq!(kinds("byte"), vec![TokenKind::ByteTy, TokenKind::Eof]);
        assert_eq!(kinds("void"), vec![TokenKind::VoidTy, TokenKind::Eof]);
    }

    #[test]
    fn stdlib_and_result_family_names_lex_as_identifiers() {
        for name in [
            "Result", "Option", "Ok", "Err", "Some", "None", "println", "open", "map",
            "filter", "reduce", "print", "len", "push", "pop", "sqrt", "abs", "assert",
        ] {
            assert_eq!(
                kinds(name),
                vec![TokenKind::Ident(name.to_string()), TokenKind::Eof],
                "{name:?} should be an identifier"
            );
        }
    }

    #[test]
    fn operators_and_punctuation_lex_one_per_kind() {
        let cases: &[(&str, TokenKind)] = &[
            ("+", TokenKind::Plus),
            ("-", TokenKind::Minus),
            ("*", TokenKind::Star),
            ("/", TokenKind::Slash),
            ("%", TokenKind::Percent),
            ("<", TokenKind::Lt),
            (">", TokenKind::Gt),
            ("!", TokenKind::Bang),
            ("=", TokenKind::Eq),
            (".", TokenKind::Dot),
            (",", TokenKind::Comma),
            (":", TokenKind::Colon),
            (";", TokenKind::Semi),
            ("(", TokenKind::LParen),
            (")", TokenKind::RParen),
            ("[", TokenKind::LBracket),
            ("]", TokenKind::RBracket),
            ("{", TokenKind::LBrace),
            ("}", TokenKind::RBrace),
            ("?", TokenKind::Question),
        ];
        for (src, kind) in cases {
            assert_eq!(kinds(src), vec![kind.clone(), TokenKind::Eof], "operator {src:?}");
        }
    }

    #[test]
    fn maximal_munch_pairs_and_triples() {
        let cases: &[(&str, TokenKind)] = &[
            ("->", TokenKind::Arrow),
            ("=>", TokenKind::FatArrow),
            ("|>", TokenKind::PipeGt),
            ("..", TokenKind::DotDot),
            ("..=", TokenKind::DotDotEq),
            ("==", TokenKind::EqEq),
            ("!=", TokenKind::BangEq),
            ("<=", TokenKind::LtEq),
            (">=", TokenKind::GtEq),
            ("&&", TokenKind::AmpAmp),
            ("||", TokenKind::PipePipe),
        ];
        for (src, kind) in cases {
            assert_eq!(kinds(src), vec![kind.clone(), TokenKind::Eof], "operator {src:?}");
        }
        // `a -> b` is Ident, Arrow, Ident -- not Minus, Gt.
        assert_eq!(
            kinds("a -> b"),
            vec![
                TokenKind::Ident("a".to_string()),
                TokenKind::Arrow,
                TokenKind::Ident("b".to_string()),
                TokenKind::Eof,
            ]
        );
        // `a => b` is Ident, FatArrow, Ident.
        assert_eq!(
            kinds("a => b"),
            vec![
                TokenKind::Ident("a".to_string()),
                TokenKind::FatArrow,
                TokenKind::Ident("b".to_string()),
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn line_comments_are_skipped_but_slash_is_division() {
        // `//` to end of line skipped.
        assert_eq!(
            kinds("a // comment\nb"),
            vec![
                TokenKind::Ident("a".to_string()),
                TokenKind::Ident("b".to_string()),
                TokenKind::Eof,
            ]
        );
        // a lone `/` is Slash; there are no block comments.
        assert_eq!(
            kinds("/x"),
            vec![TokenKind::Slash, TokenKind::Ident("x".to_string()), TokenKind::Eof]
        );
        assert_eq!(
            kinds("a / b"),
            vec![
                TokenKind::Ident("a".to_string()),
                TokenKind::Slash,
                TokenKind::Ident("b".to_string()),
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn a_lone_ampersand_is_unexpected_char() {
        match err("a & b") {
            QalaError::UnexpectedChar { span, ch } => {
                assert_eq!(ch, '&');
                assert_eq!(span, Span::new(2, 1));
            }
            other => panic!("expected UnexpectedChar, got {other:?}"),
        }
        // bare `&` too.
        assert!(matches!(err("&"), QalaError::UnexpectedChar { ch: '&', .. }));
    }

    #[test]
    fn a_lone_pipe_is_unexpected_char() {
        match err("a | b") {
            QalaError::UnexpectedChar { span, ch } => {
                assert_eq!(ch, '|');
                assert_eq!(span, Span::new(2, 1));
            }
            other => panic!("expected UnexpectedChar, got {other:?}"),
        }
        // bare `|` and `|x` (not `||` or `|>`).
        assert!(matches!(err("|"), QalaError::UnexpectedChar { ch: '|', .. }));
        assert!(matches!(err("|x"), QalaError::UnexpectedChar { ch: '|', .. }));
    }

    #[test]
    fn a_non_ascii_byte_in_identifier_position_is_unexpected_char() {
        match err("let café = 1") {
            QalaError::UnexpectedChar { span, ch } => {
                assert_eq!(ch, 'é');
                // "let caf" is 7 bytes; the 'é' starts at byte 7 and is 2 bytes.
                let e_start = "let caf".len();
                assert_eq!(span, Span::new(e_start, 'é'.len_utf8()));
                assert_eq!(span.slice("let café = 1"), "é");
            }
            other => panic!("expected UnexpectedChar, got {other:?}"),
        }
        // a non-ASCII char inside a // comment is fine -> just Eof.
        assert_eq!(kinds("// café au lait\n"), vec![TokenKind::Eof]);
    }

    #[test]
    fn representative_token_spans_are_exact() {
        // "fn main()": "main" is bytes 3..7, length 4, slices back to "main".
        let src = "fn main()";
        let toks = Lexer::tokenize(src).unwrap();
        let main_tok = &toks[1];
        assert_eq!(main_tok.kind, TokenKind::Ident("main".to_string()));
        assert_eq!(main_tok.span.start, 3);
        assert_eq!(main_tok.span.len, 4);
        assert_eq!(main_tok.span.slice(src), "main");
        // the `fn` keyword: bytes 0..2.
        assert_eq!(toks[0].span, Span::new(0, 2));
        assert_eq!(toks[0].span.slice(src), "fn");
        // the trailing Eof is a zero-length span at the end.
        let eof = toks.last().unwrap();
        assert_eq!(eof.kind, TokenKind::Eof);
        assert_eq!(eof.span, Span::new(src.len(), 0));
    }

    #[test]
    fn span_after_trivia_starts_at_the_token_not_the_whitespace() {
        // "  +  ": the `+` is at byte 2, length 1.
        let src = "  +  ";
        let toks = Lexer::tokenize(src).unwrap();
        assert_eq!(toks[0].kind, TokenKind::Plus);
        assert_eq!(toks[0].span, Span::new(2, 1));
    }
}
