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

use std::num::IntErrorKind;

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

    // ---- numeric and byte literals --------------------------------------------

    /// scan a numeric literal starting at `start` (the digit at `self.pos`).
    ///
    /// the literal scans as a *non-negative magnitude*: a leading `-` is always a
    /// separate `Minus` token the parser folds in later, so the bare literal
    /// `9223372036854775808` (with no `-`) is reported as overflow -- acceptable
    /// v1 behavior; folding `- 9223372036854775808` into `i64::MIN` is left for
    /// the parser.
    ///
    /// the disambiguation that matters: a `.` is part of the number only when the
    /// byte *after* it is an ASCII digit. that single rule keeps `.5` as `Dot`
    /// then `Int(5)`, `0..5` as `Int(0)` `DotDot` `Int(5)`, and `x.0` as
    /// `Ident(x)` `Dot` `Int(0)` -- field access and ranges stay unambiguous.
    fn scan_number(&mut self, start: usize) -> Result<TokenKind, QalaError> {
        // prefix forms: 0x / 0b. checked before the decimal path so `0x` is not
        // mis-scanned as the decimal `0` followed by the identifier `x`.
        if self.peek() == Some(b'0') {
            match self.peek2() {
                Some(b'x') | Some(b'X') => return self.scan_radix_int(start, 16),
                Some(b'b') | Some(b'B') => return self.scan_radix_int(start, 2),
                _ => {}
            }
        }
        self.scan_decimal(start)
    }

    /// scan a `0x...` (radix 16) or `0b...` (radix 2) integer literal: the two
    /// prefix bytes, then a run of valid digits and `_` separators with at least
    /// one digit. an empty body (`0x`, `0b`), an out-of-range digit (`0xG`,
    /// `0b2`), or a misplaced `_` is [`QalaError::MalformedNumber`]; a magnitude
    /// past `i64::MAX` is [`QalaError::IntOverflow`].
    fn scan_radix_int(&mut self, start: usize, radix: u32) -> Result<TokenKind, QalaError> {
        self.pos += 2; // consume "0x" / "0b"
        let body_start = self.pos;
        let is_digit = |b: u8| match radix {
            16 => b.is_ascii_hexdigit(),
            2 => b == b'0' || b == b'1',
            _ => false,
        };
        // scan digits + underscores, tracking the last char class for the `_` rules.
        let mut last_was_underscore = false;
        let mut saw_digit = false;
        let mut malformed = false;
        while let Some(b) = self.peek() {
            if is_digit(b) {
                last_was_underscore = false;
                saw_digit = true;
                self.pos += 1;
            } else if b == b'_' {
                // a `_` is valid only between digits: it cannot start the body and
                // cannot follow another `_`.
                if !saw_digit || last_was_underscore {
                    malformed = true;
                }
                last_was_underscore = true;
                self.pos += 1;
            } else {
                break;
            }
        }
        // a body that ends in `_` (`0xFF_`) is malformed; an empty body (`0x`) too.
        // also reject a trailing alphanumeric like `0xG` or `0b2`: such a byte
        // right after the body is the user's mistake, not the start of a new
        // identifier. consume the whole offending tail so the error span covers it.
        if self.peek().is_some_and(|b| b == b'_' || b.is_ascii_alphanumeric()) {
            self.consume_alnum_tail();
            malformed = true;
        }
        if malformed || !saw_digit || last_was_underscore {
            return Err(self.malformed_number(start));
        }
        let digits: String = self.src[body_start..self.pos].chars().filter(|&c| c != '_').collect();
        match i64::from_str_radix(&digits, radix) {
            Ok(value) => Ok(TokenKind::Int(value)),
            Err(e) => Err(self.int_parse_error(start, e.kind())),
        }
    }

    /// scan a decimal integer or float starting at `start`. integer part: digits +
    /// `_`. then, only if a `.` is followed by an ASCII digit, a fractional part.
    /// then, optionally, an `e`/`E` exponent (`[+-]?` then a digit run). a
    /// misplaced `_`, an `e` with no exponent digits, or a stray alphanumeric tail
    /// is [`QalaError::MalformedNumber`]; an integer past `i64::MAX` is
    /// [`QalaError::IntOverflow`].
    fn scan_decimal(&mut self, start: usize) -> Result<TokenKind, QalaError> {
        let mut is_float = false;

        // integer part: at least one digit (we were called on a digit), then more
        // digits and `_`. the helper rejects a `_` that does not sit between two
        // digits within this run.
        self.scan_digit_run(start)?;

        // fractional part. a `.` is the decimal point only when something digit-
        // like follows it: a digit (`1.0` -- a float) or a `_` (`1._5` -- a
        // malformed fractional part, consumed so the error span covers it). a `.`
        // followed by anything else is NOT part of the number, so `0..5` stays
        // `Int(0) DotDot Int(5)` and `x.0` stays `Ident(x) Dot Int(0)`.
        if self.peek() == Some(b'.') {
            match self.peek2() {
                Some(b) if b.is_ascii_digit() => {
                    is_float = true;
                    self.pos += 1; // consume the `.`
                    self.scan_digit_run(start)?;
                }
                Some(b'_') => {
                    self.pos += 1; // consume the `.`
                    self.consume_alnum_tail();
                    return Err(self.malformed_number(start));
                }
                _ => {}
            }
        }

        // exponent: `e`/`E`, an optional sign, then a digit run that must be
        // non-empty (`1e` is malformed, `1e10` and `1.5e-3` are floats).
        if matches!(self.peek(), Some(b'e') | Some(b'E')) {
            is_float = true;
            self.pos += 1; // consume `e`/`E`
            if matches!(self.peek(), Some(b'+') | Some(b'-')) {
                self.pos += 1;
            }
            if !self.peek().is_some_and(|b| b.is_ascii_digit()) {
                // no exponent digits: consume any alphanumeric tail for the span,
                // then error.
                self.consume_alnum_tail();
                return Err(self.malformed_number(start));
            }
            self.scan_digit_run(start)?;
        }

        // a stray alphanumeric byte right after the literal (`1abc`, `0x` after a
        // decimal, ...) is the user's mistake, not the start of a new token.
        if self.peek().is_some_and(|b| b == b'_' || b.is_ascii_alphanumeric()) {
            self.consume_alnum_tail();
            return Err(self.malformed_number(start));
        }

        let text: String = self.src[start..self.pos].chars().filter(|&c| c != '_').collect();
        if is_float {
            match text.parse::<f64>() {
                Ok(value) => Ok(TokenKind::Float(value)),
                // a non-round-tripping float is NOT an error (a Phase 4 precision
                // warning handles it); only a shape that slipped past the scanner
                // lands here, which is genuinely malformed.
                Err(_) => Err(self.malformed_number(start)),
            }
        } else {
            match text.parse::<i64>() {
                Ok(value) => Ok(TokenKind::Int(value)),
                Err(e) => Err(self.int_parse_error(start, e.kind())),
            }
        }
    }

    /// advance over a run of ASCII digits and `_` separators, where each `_` must
    /// sit between two digits. the run must contain at least one digit (the caller
    /// only invokes this when the next byte is a digit, or just after a `.`/`e`
    /// that requires digits). a misplaced `_` -> [`QalaError::MalformedNumber`].
    fn scan_digit_run(&mut self, literal_start: usize) -> Result<(), QalaError> {
        let mut last_was_underscore = false;
        let mut saw_digit = false;
        // a leading `_` cannot start a digit run (e.g. `1._5`, `1e_5`).
        if self.peek() == Some(b'_') {
            self.consume_alnum_tail();
            return Err(self.malformed_number(literal_start));
        }
        while let Some(b) = self.peek() {
            if b.is_ascii_digit() {
                last_was_underscore = false;
                saw_digit = true;
                self.pos += 1;
            } else if b == b'_' {
                if last_was_underscore {
                    // a doubled `_` (`1__0`).
                    self.consume_alnum_tail();
                    return Err(self.malformed_number(literal_start));
                }
                last_was_underscore = true;
                self.pos += 1;
            } else {
                break;
            }
        }
        // a run that ends in `_` (`1_`, `1_.0`) is malformed -- the `_` is not
        // between two digits.
        if last_was_underscore || !saw_digit {
            self.consume_alnum_tail();
            return Err(self.malformed_number(literal_start));
        }
        Ok(())
    }

    /// advance past any run of `_`, ASCII alphanumeric, or `.` bytes at the
    /// cursor, so a malformed-literal span covers the whole offending text
    /// (`1_.0`, `0xFG`, `1abc`) rather than ending mid-word. only ever called on
    /// an error path, so a `.` consumed here never matters to a valid token.
    fn consume_alnum_tail(&mut self) {
        while let Some(b) = self.peek() {
            if b == b'_' || b.is_ascii_alphanumeric() || b == b'.' {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    /// build a [`QalaError::MalformedNumber`] spanning the literal from
    /// `literal_start` to the cursor, with a message echoing the offending text.
    fn malformed_number(&self, literal_start: usize) -> QalaError {
        let text = &self.src[literal_start..self.pos];
        QalaError::MalformedNumber {
            span: Span::new(literal_start, self.pos - literal_start),
            message: format!("`{text}`"),
        }
    }

    /// map a [`std::num::IntErrorKind`] from parsing an integer literal to the
    /// right error: overflow (either sign, since the scanner only ever hands a
    /// magnitude) -> [`QalaError::IntOverflow`]; anything else -> malformed.
    fn int_parse_error(&self, literal_start: usize, kind: &IntErrorKind) -> QalaError {
        match kind {
            IntErrorKind::PosOverflow | IntErrorKind::NegOverflow => {
                QalaError::IntOverflow {
                    span: Span::new(literal_start, self.pos - literal_start),
                }
            }
            _ => self.malformed_number(literal_start),
        }
    }

    /// scan a `b'X'` byte literal starting at `start` (the cursor is on `b`, and
    /// the byte after it is `'` -- the dispatcher checked).
    ///
    /// inside `b'...'` is exactly one ASCII char, or one of the one-byte escapes
    /// `\n \t \r \0 \\ \'`. anything else -- empty (`b''`), two or more chars
    /// (`b'ab'`), a bad escape (`b'\x'`), a non-ASCII byte (`b'é'`), or a missing
    /// closing quote -- is [`QalaError::BadByteLiteral`] spanning from `b` to
    /// where scanning stopped.
    fn scan_byte_literal(&mut self, start: usize) -> Result<TokenKind, QalaError> {
        self.pos += 2; // consume `b` and the opening `'`
        let value: u8 = match self.peek() {
            // empty: `b''`.
            Some(b'\'') => {
                self.pos += 1;
                return Err(self.bad_byte_literal(start, "expected one character between `b'` and `'`"));
            }
            // an escape: `b'\X'`.
            Some(b'\\') => {
                self.pos += 1;
                match self.bump() {
                    Some(b'n') => b'\n',
                    Some(b't') => b'\t',
                    Some(b'r') => b'\r',
                    Some(b'0') => 0,
                    Some(b'\\') => b'\\',
                    Some(b'\'') => b'\'',
                    _ => {
                        // consume a closing quote if present so the span is tidy.
                        if self.peek() == Some(b'\'') {
                            self.pos += 1;
                        }
                        return Err(self.bad_byte_literal(start, "unknown escape in byte literal"));
                    }
                }
            }
            // a non-ASCII byte: not a valid byte-literal char.
            Some(b) if b >= 0x80 => {
                let ch = self.src[self.pos..].chars().next().unwrap_or('\u{FFFD}');
                self.pos += ch.len_utf8();
                // consume a closing quote too if it is there, so the span is tidy.
                if self.peek() == Some(b'\'') {
                    self.pos += 1;
                }
                return Err(self.bad_byte_literal(start, "byte literal must be a single ASCII character"));
            }
            // a raw newline or EOF before any content.
            Some(b'\n') | None => {
                return Err(self.bad_byte_literal(start, "unterminated byte literal"));
            }
            // an ordinary ASCII char.
            Some(b) => {
                self.pos += 1;
                b
            }
        };
        // require the closing `'`. if the next byte is another char (not `'`), the
        // literal had two or more chars; consume to the next `'` or boundary so the
        // span covers the mistake.
        match self.peek() {
            Some(b'\'') => {
                self.pos += 1;
                Ok(TokenKind::Byte(value))
            }
            _ => {
                while let Some(b) = self.peek() {
                    if b == b'\'' {
                        self.pos += 1;
                        break;
                    }
                    if b == b'\n' {
                        break;
                    }
                    self.pos += 1;
                }
                Err(self.bad_byte_literal(start, "byte literal must be exactly one character"))
            }
        }
    }

    /// build a [`QalaError::BadByteLiteral`] spanning from `literal_start` to the
    /// cursor.
    fn bad_byte_literal(&self, literal_start: usize, message: &str) -> QalaError {
        QalaError::BadByteLiteral {
            span: Span::new(literal_start, self.pos - literal_start),
            message: message.to_string(),
        }
    }

    // ---- string literals and interpolation ------------------------------------

    /// scan a run of string-literal text once `StrText` is the top mode. the
    /// opening `"` (or, on a resumed call, the closing `}` of the preceding
    /// interpolation) has already been consumed, so `self.pos` is at the first
    /// content byte.
    ///
    /// it decodes escapes into a buffer and stops at one of:
    /// - the closing `"` -- pushes [`TokenKind::Str`] (no interpolation seen) or
    ///   [`TokenKind::StrEnd`] (resuming after the last interpolation), pops the
    ///   `StrText`. the `Str` span runs from the opening quote through the closing
    ///   quote inclusive; the `StrEnd` span runs from just after the preceding `}`
    ///   through the closing quote.
    /// - an unescaped `{` -- pushes [`TokenKind::StrStart`] (first interpolation)
    ///   or [`TokenKind::StrMid`] (a later one) for the text so far, then
    ///   [`TokenKind::InterpStart`] for the `{`, then a [`Mode::Interp`] so the
    ///   main loop scans the embedded expression. a literal `{` in a string is
    ///   written `\{`, handled in the escape path, so a bare `{` always opens an
    ///   interpolation.
    /// - a backslash -- decodes `\n \t \r \0 \\ \" \{ \}` or `\u{1..6 hex}` into
    ///   the buffer; anything else is [`QalaError::InvalidEscape`] whose span is
    ///   the backslash (not the bad character after it).
    /// - a raw newline or EOF -- [`QalaError::UnterminatedString`] whose span is
    ///   the opening quote (no multi-line strings in v1).
    fn scan_string_body(&mut self, out: &mut Vec<Token>) -> Result<(), QalaError> {
        // pull the opening-quote offset and whether this string has already had an
        // interpolation. `open_quote` is needed for the `Str` span and the
        // unterminated-string span; `saw_interp` decides Str-vs-StrEnd and
        // StrStart-vs-StrMid.
        let (open_quote, saw_interp) = match self.mode_stack.last() {
            Some(Mode::StrText { open_quote, saw_interp }) => (*open_quote, *saw_interp),
            _ => return Ok(()), // not in a string; nothing to do
        };
        let text_start = self.pos;
        let mut buf = String::new();
        loop {
            match self.peek() {
                // closing quote: the string ends here.
                Some(b'"') => {
                    self.pos += 1; // consume the closing quote
                    self.mode_stack.pop(); // leave StrText
                    if saw_interp {
                        // span: just after the preceding `}` through the closing
                        // quote.
                        let span = Span::new(text_start, self.pos - text_start);
                        out.push(Token::new(TokenKind::StrEnd(buf), span));
                    } else {
                        // span: opening quote through closing quote inclusive.
                        let span = Span::new(open_quote, self.pos - open_quote);
                        out.push(Token::new(TokenKind::Str(buf), span));
                    }
                    return Ok(());
                }
                // unescaped `{`: opens an interpolation.
                Some(b'{') => {
                    let brace_pos = self.pos;
                    // emit the literal text so far.
                    if saw_interp {
                        // a StrMid spans just the text between the previous `}`
                        // and this `{`.
                        let span = Span::new(text_start, brace_pos - text_start);
                        out.push(Token::new(TokenKind::StrMid(buf), span));
                    } else {
                        // a StrStart spans the opening quote through the text
                        // before this `{`.
                        let span = Span::new(open_quote, brace_pos - open_quote);
                        out.push(Token::new(TokenKind::StrStart(buf), span));
                    }
                    // mark the enclosing StrText so the next text run closes with
                    // StrEnd / opens with StrMid.
                    if let Some(Mode::StrText { saw_interp, .. }) = self.mode_stack.last_mut() {
                        *saw_interp = true;
                    }
                    // emit the InterpStart for the `{` and enter Interp mode.
                    self.pos += 1; // consume the `{`
                    out.push(Token::new(TokenKind::InterpStart, Span::new(brace_pos, 1)));
                    self.mode_stack.push(Mode::Interp { brace_depth: 0, open_brace: brace_pos });
                    return Ok(());
                }
                // backslash: an escape sequence.
                Some(b'\\') => {
                    let backslash_pos = self.pos;
                    self.pos += 1; // consume the `\`
                    match self.bump() {
                        Some(b'n') => buf.push('\n'),
                        Some(b't') => buf.push('\t'),
                        Some(b'r') => buf.push('\r'),
                        Some(b'0') => buf.push('\0'),
                        Some(b'\\') => buf.push('\\'),
                        Some(b'"') => buf.push('"'),
                        Some(b'{') => buf.push('{'),
                        Some(b'}') => buf.push('}'),
                        Some(b'u') => {
                            let ch = self.scan_unicode_escape(backslash_pos)?;
                            buf.push(ch);
                        }
                        _ => {
                            return Err(QalaError::InvalidEscape {
                                span: Span::new(backslash_pos, 1),
                                message: "unknown escape sequence".to_string(),
                            });
                        }
                    }
                }
                // a raw newline before the close: no multi-line strings.
                Some(b'\n') | None => {
                    return Err(QalaError::UnterminatedString {
                        span: Span::new(open_quote, 1),
                    });
                }
                // any other byte (ASCII or the lead of a multi-byte char) is
                // literal content. decode the whole char so a multi-byte char
                // (a non-ASCII char inside a string is fine) lands in the buffer
                // intact and the cursor advances by the right number of bytes.
                Some(_) => {
                    let ch = self.src[self.pos..].chars().next().unwrap_or('\u{FFFD}');
                    self.pos += ch.len_utf8();
                    buf.push(ch);
                }
            }
        }
    }

    /// decode a `\u{1..6 hex}` escape. the cursor is just past the `u`; this
    /// expects `{`, 1..=6 hex digits, then `}`, and validates the codepoint with
    /// [`char::from_u32`] (which rejects surrogates `D800..DFFF` and values above
    /// `10FFFF`). any malformation -- no `{`, no digits, too many digits, no `}`,
    /// or a bad codepoint -- is [`QalaError::InvalidEscape`] whose span is the
    /// backslash at `backslash_pos`.
    fn scan_unicode_escape(&mut self, backslash_pos: usize) -> Result<char, QalaError> {
        let bad = |message: &str| QalaError::InvalidEscape {
            span: Span::new(backslash_pos, 1),
            message: message.to_string(),
        };
        if self.peek() != Some(b'{') {
            return Err(bad("expected `{` after `\\u`"));
        }
        self.pos += 1; // consume `{`
        let hex_start = self.pos;
        let mut count = 0usize;
        while let Some(b) = self.peek() {
            if b.is_ascii_hexdigit() && count < 6 {
                count += 1;
                self.pos += 1;
            } else {
                break;
            }
        }
        if count == 0 {
            return Err(bad("expected 1 to 6 hex digits in `\\u{...}`"));
        }
        if self.peek() != Some(b'}') {
            return Err(bad("expected `}` to close `\\u{...}`"));
        }
        let hex = &self.src[hex_start..self.pos];
        self.pos += 1; // consume `}`
        // count <= 6 hex digits fit in u32; from_str_radix only errors on an
        // empty string here, which count == 0 already ruled out.
        let code = u32::from_str_radix(hex, 16).map_err(|_| bad("invalid `\\u{...}` codepoint"))?;
        char::from_u32(code).ok_or_else(|| bad("invalid Unicode codepoint in `\\u{...}`"))
    }

    /// if EOF was reached with a string or interpolation still open on the mode
    /// stack, return the matching unterminated error pointing at where it opened.
    ///
    /// a string body is always fully consumed (or errored) inside
    /// `scan_string_body` before control returns here, so in practice this fires
    /// for an [`Mode::Interp`] that never saw its closing `}` (`"x{y"`), reporting
    /// at the `{`. the `StrText` arm is a defensive backstop.
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

    // ---- Task 2: numeric and byte literals ------------------------------------

    /// the single kind of a one-token source (plus the trailing Eof).
    fn one(src: &str) -> TokenKind {
        let mut k = kinds(src);
        assert_eq!(k.last(), Some(&TokenKind::Eof), "stream should end in Eof: {src:?}");
        k.pop();
        assert_eq!(k.len(), 1, "expected exactly one token in {src:?}, got {k:?}");
        k.pop().unwrap()
    }

    #[test]
    fn decimal_integers() {
        assert_eq!(one("0"), TokenKind::Int(0));
        assert_eq!(one("42"), TokenKind::Int(42));
        assert_eq!(one("1_000_000"), TokenKind::Int(1_000_000));
        assert_eq!(one("9223372036854775807"), TokenKind::Int(i64::MAX));
    }

    #[test]
    fn hex_integers() {
        assert_eq!(one("0xFF"), TokenKind::Int(255));
        assert_eq!(one("0xFF_FF"), TokenKind::Int(65_535));
        assert_eq!(one("0X1a"), TokenKind::Int(26));
        assert_eq!(one("0x0"), TokenKind::Int(0));
    }

    #[test]
    fn binary_integers() {
        assert_eq!(one("0b1010"), TokenKind::Int(10));
        assert_eq!(one("0b1010_0101"), TokenKind::Int(165));
        assert_eq!(one("0B1"), TokenKind::Int(1));
    }

    #[test]
    fn integer_overflow_errors_at_the_digits() {
        // 2^63 = i64::MAX + 1.
        match err("9223372036854775808") {
            QalaError::IntOverflow { span } => {
                assert_eq!(span, Span::new(0, 19), "span should cover the 19 digits");
            }
            other => panic!("expected IntOverflow, got {other:?}"),
        }
        match err("0x8000000000000000") {
            QalaError::IntOverflow { span } => assert_eq!(span, Span::new(0, 18)),
            other => panic!("expected IntOverflow, got {other:?}"),
        }
        // a 50-digit number.
        let big = "1".repeat(50);
        assert!(matches!(err(&big), QalaError::IntOverflow { .. }));
        // overflow inside a larger source still points at the digits.
        match err("let x = 99999999999999999999\n") {
            QalaError::IntOverflow { span } => {
                assert_eq!(span.slice("let x = 99999999999999999999\n"), "99999999999999999999");
            }
            other => panic!("expected IntOverflow, got {other:?}"),
        }
    }

    #[test]
    fn malformed_numbers_span_the_literal() {
        for src in ["1_", "1__0", "0x", "0xG", "0b2", "1_.0", "1._5", "1e_5", "1e", "0b"] {
            match Lexer::tokenize(src) {
                Err(QalaError::MalformedNumber { span, .. }) => {
                    assert_eq!(
                        span.slice(src),
                        src,
                        "MalformedNumber span should cover the whole literal {src:?}"
                    );
                }
                other => panic!("expected MalformedNumber for {src:?}, got {other:?}"),
            }
        }
        // the `1__0` span is exactly those 4 bytes even mid-source.
        match err("a = 1__0;") {
            QalaError::MalformedNumber { span, .. } => assert_eq!(span.slice("a = 1__0;"), "1__0"),
            other => panic!("expected MalformedNumber, got {other:?}"),
        }
    }

    #[test]
    fn floats_including_exponents() {
        assert_eq!(one("1.0"), TokenKind::Float(1.0));
        assert_eq!(one("1.5e10"), TokenKind::Float(1.5e10));
        assert_eq!(one("1e10"), TokenKind::Float(1e10));
        assert_eq!(one("2.0e-3"), TokenKind::Float(2.0e-3));
        assert_eq!(one("1.5E+2"), TokenKind::Float(150.0));
        assert_eq!(one("7.25"), TokenKind::Float(7.25));
        // a non-round-tripping float is NOT an error.
        assert_eq!(one("0.1"), TokenKind::Float(0.1));
        // underscores inside a float are stripped.
        assert_eq!(one("1_000.000_5"), TokenKind::Float(1000.0005));
    }

    #[test]
    fn the_leading_dot_rule_keeps_dot_and_ranges_unambiguous() {
        // a number only starts on a digit; `.` is never the start of one.
        assert_eq!(
            kinds(".5"),
            vec![TokenKind::Dot, TokenKind::Int(5), TokenKind::Eof]
        );
        assert_eq!(
            kinds("0..5"),
            vec![TokenKind::Int(0), TokenKind::DotDot, TokenKind::Int(5), TokenKind::Eof]
        );
        assert_eq!(
            kinds("0..=5"),
            vec![TokenKind::Int(0), TokenKind::DotDotEq, TokenKind::Int(5), TokenKind::Eof]
        );
        assert_eq!(
            kinds("x.0"),
            vec![
                TokenKind::Ident("x".to_string()),
                TokenKind::Dot,
                TokenKind::Int(0),
                TokenKind::Eof,
            ]
        );
        // the fibonacci example's range.
        assert_eq!(
            kinds("0..15"),
            vec![TokenKind::Int(0), TokenKind::DotDot, TokenKind::Int(15), TokenKind::Eof]
        );
        // `1.0` is still a float -- the `.` here IS followed by a digit.
        assert_eq!(one("1.0"), TokenKind::Float(1.0));
        // `1..2` is a range over integers, not `1.` then `.2`.
        assert_eq!(
            kinds("1..2"),
            vec![TokenKind::Int(1), TokenKind::DotDot, TokenKind::Int(2), TokenKind::Eof]
        );
    }

    #[test]
    fn byte_literals() {
        assert_eq!(one("b'A'"), TokenKind::Byte(65));
        assert_eq!(one("b'\\n'"), TokenKind::Byte(10));
        assert_eq!(one("b'\\t'"), TokenKind::Byte(9));
        assert_eq!(one("b'\\r'"), TokenKind::Byte(13));
        assert_eq!(one("b'\\\\'"), TokenKind::Byte(92));
        assert_eq!(one("b'\\''"), TokenKind::Byte(39));
        assert_eq!(one("b'\\0'"), TokenKind::Byte(0));
        assert_eq!(one("b' '"), TokenKind::Byte(32));
        assert_eq!(one("b'z'"), TokenKind::Byte(122));
    }

    #[test]
    fn bad_byte_literals() {
        for src in ["b''", "b'ab'", "b'\\x'", "b'é'"] {
            match Lexer::tokenize(src) {
                Err(QalaError::BadByteLiteral { span, .. }) => {
                    // the span starts at the `b` and covers the offending literal.
                    assert_eq!(span.start, 0, "span should start at `b` for {src:?}");
                    assert!(span.len >= 3, "span should cover the literal for {src:?}");
                }
                other => panic!("expected BadByteLiteral for {src:?}, got {other:?}"),
            }
        }
    }

    #[test]
    fn b_not_followed_by_quote_is_the_identifier_b() {
        assert_eq!(
            kinds("by"),
            vec![TokenKind::Ident("by".to_string()), TokenKind::Eof]
        );
        assert_eq!(
            kinds("b 1"),
            vec![TokenKind::Ident("b".to_string()), TokenKind::Int(1), TokenKind::Eof]
        );
        assert_eq!(kinds("b"), vec![TokenKind::Ident("b".to_string()), TokenKind::Eof]);
        // `byte` is still the keyword, not `b` + `yte`.
        assert_eq!(kinds("byte"), vec![TokenKind::ByteTy, TokenKind::Eof]);
    }

    #[test]
    fn numbers_in_a_realistic_snippet() {
        // `for i in 0..15 { x = 1_000 + 0xFF }` -- a mix of range and literals.
        let toks = kinds("for i in 0..15 { x = 1_000 + 0xFF }");
        assert_eq!(
            toks,
            vec![
                TokenKind::For,
                TokenKind::Ident("i".to_string()),
                TokenKind::In,
                TokenKind::Int(0),
                TokenKind::DotDot,
                TokenKind::Int(15),
                TokenKind::LBrace,
                TokenKind::Ident("x".to_string()),
                TokenKind::Eq,
                TokenKind::Int(1_000),
                TokenKind::Plus,
                TokenKind::Int(255),
                TokenKind::RBrace,
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn numeric_literal_spans_are_exact() {
        // "x = 42": the `42` is bytes 4..6, length 2.
        let src = "x = 42";
        let toks = Lexer::tokenize(src).unwrap();
        let lit = &toks[2];
        assert_eq!(lit.kind, TokenKind::Int(42));
        assert_eq!(lit.span, Span::new(4, 2));
        assert_eq!(lit.span.slice(src), "42");
        // a float's span covers the whole literal including the exponent.
        let src2 = "y = 1.5e10;";
        let toks2 = Lexer::tokenize(src2).unwrap();
        let f = &toks2[2];
        assert_eq!(f.kind, TokenKind::Float(1.5e10));
        assert_eq!(f.span.slice(src2), "1.5e10");
    }

    // ---- Task 3: strings, escapes, interpolation ------------------------------
    //
    // these tests are written against Qala source, so a `"` in the source is `\"`
    // in the Rust string literal and a `\` in the source is `\\`.

    /// shorthand for `TokenKind::Str(s.to_string())` and friends, to keep the
    /// expected streams readable.
    fn s(text: &str) -> TokenKind { TokenKind::Str(text.to_string()) }
    fn id(text: &str) -> TokenKind { TokenKind::Ident(text.to_string()) }

    #[test]
    fn interpolation_free_strings() {
        assert_eq!(one("\"abc\""), s("abc"));
        assert_eq!(one("\"\""), s(""));
        assert_eq!(one("\"hello, world\""), s("hello, world"));
        // a non-ASCII char inside a string is fine and lands in the payload.
        assert_eq!(one("\"café\""), s("café"));
        // the Str span covers the opening quote through the closing quote.
        let src = "x = \"hi\"";
        let toks = Lexer::tokenize(src).unwrap();
        assert_eq!(toks[2].kind, s("hi"));
        assert_eq!(toks[2].span.slice(src), "\"hi\"");
    }

    #[test]
    fn escapes_are_decoded_into_the_payload() {
        assert_eq!(one("\"a\\nb\""), s("a\nb"));
        assert_eq!(one("\"\\t\""), s("\t"));
        assert_eq!(one("\"\\r\""), s("\r"));
        assert_eq!(one("\"\\0\""), s("\0"));
        assert_eq!(one("\"\\\\\""), s("\\"));
        assert_eq!(one("\"\\\"\""), s("\""));
        assert_eq!(one("\"\\{\""), s("{"));
        assert_eq!(one("\"\\}\""), s("}"));
        assert_eq!(one("\"\\u{41}\""), s("A"));
        assert_eq!(one("\"\\u{1F600}\""), s("\u{1F600}"));
        // a mix: "line1\nline2 \u{2764}" -> "line1", newline, "line2 ", heart.
        assert_eq!(one("\"line1\\nline2 \\u{2764}\""), s("line1\nline2 \u{2764}"));
        // escaped braces do NOT open an interpolation.
        assert_eq!(one("\"a \\{not interp\\} b\""), s("a {not interp} b"));
    }

    #[test]
    fn bad_escapes_error_at_the_backslash() {
        // "a\qb": the `\` is byte 2 (after the `"` and `a`).
        match err("\"a\\qb\"") {
            QalaError::InvalidEscape { span, .. } => {
                assert_eq!(span, Span::new(2, 1), "span should be the backslash byte");
            }
            other => panic!("expected InvalidEscape, got {other:?}"),
        }
        // bad \u{...} forms: empty, out of range, surrogate, no closing brace.
        for src in ["\"\\u{}\"", "\"\\u{110000}\"", "\"\\u{D800}\"", "\"\\u{41\"", "\"\\uABCD\""] {
            assert!(
                matches!(Lexer::tokenize(src), Err(QalaError::InvalidEscape { .. })),
                "expected InvalidEscape for {src:?}"
            );
        }
        // the backslash position is reported even mid-string.
        match err("\"hello \\x world\"") {
            QalaError::InvalidEscape { span, .. } => {
                assert_eq!(span.slice("\"hello \\x world\""), "\\");
            }
            other => panic!("expected InvalidEscape, got {other:?}"),
        }
    }

    #[test]
    fn unterminated_string_errors_at_the_opening_quote() {
        // "abc with EOF before the close.
        match err("\"abc") {
            QalaError::UnterminatedString { span } => {
                assert_eq!(span, Span::new(0, 1), "span should be the opening quote");
            }
            other => panic!("expected UnterminatedString, got {other:?}"),
        }
        // a real newline before the close: no multi-line strings.
        match err("\"abc\ndef\"") {
            QalaError::UnterminatedString { span } => assert_eq!(span, Span::new(0, 1)),
            other => panic!("expected UnterminatedString, got {other:?}"),
        }
        // the opening quote is found even when the string starts mid-source.
        match err("let x = \"oops") {
            QalaError::UnterminatedString { span } => {
                assert_eq!(span.slice("let x = \"oops"), "\"");
            }
            other => panic!("expected UnterminatedString, got {other:?}"),
        }
        // an empty unterminated string `"` at EOF.
        assert!(matches!(err("\""), QalaError::UnterminatedString { .. }));
    }

    #[test]
    fn simple_interpolation() {
        // "hi {name}!"
        assert_eq!(
            kinds("\"hi {name}!\""),
            vec![
                TokenKind::StrStart("hi ".to_string()),
                TokenKind::InterpStart,
                id("name"),
                TokenKind::InterpEnd,
                TokenKind::StrEnd("!".to_string()),
                TokenKind::Eof,
            ]
        );
        // the hello.qala interpolation: println("hello, {name}!").
        assert_eq!(
            kinds("println(\"hello, {name}!\")"),
            vec![
                id("println"),
                TokenKind::LParen,
                TokenKind::StrStart("hello, ".to_string()),
                TokenKind::InterpStart,
                id("name"),
                TokenKind::InterpEnd,
                TokenKind::StrEnd("!".to_string()),
                TokenKind::RParen,
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn multiple_and_empty_fragment_interpolations() {
        // "{a}{b}"
        assert_eq!(
            kinds("\"{a}{b}\""),
            vec![
                TokenKind::StrStart(String::new()),
                TokenKind::InterpStart,
                id("a"),
                TokenKind::InterpEnd,
                TokenKind::StrMid(String::new()),
                TokenKind::InterpStart,
                id("b"),
                TokenKind::InterpEnd,
                TokenKind::StrEnd(String::new()),
                TokenKind::Eof,
            ]
        );
        // "a{x}b{y}c"
        assert_eq!(
            kinds("\"a{x}b{y}c\""),
            vec![
                TokenKind::StrStart("a".to_string()),
                TokenKind::InterpStart,
                id("x"),
                TokenKind::InterpEnd,
                TokenKind::StrMid("b".to_string()),
                TokenKind::InterpStart,
                id("y"),
                TokenKind::InterpEnd,
                TokenKind::StrEnd("c".to_string()),
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn interpolation_with_nested_braces() {
        // "{ {a: 1}.a }" -- the interpolation ends at the matching depth-0 `}`,
        // not the struct literal's `}`. the inner braces appear as LBrace/RBrace.
        assert_eq!(
            kinds("\"{ {a: 1}.a }\""),
            vec![
                TokenKind::StrStart(String::new()),
                TokenKind::InterpStart,
                TokenKind::LBrace,
                id("a"),
                TokenKind::Colon,
                TokenKind::Int(1),
                TokenKind::RBrace,
                TokenKind::Dot,
                id("a"),
                TokenKind::InterpEnd,
                TokenKind::StrEnd(String::new()),
                TokenKind::Eof,
            ]
        );
        // "{ if x { 1 } else { 2 } }" -- the if/else block braces are handled by
        // depth; the interpolation closes at the final `}`.
        assert_eq!(
            kinds("\"{ if x { 1 } else { 2 } }\""),
            vec![
                TokenKind::StrStart(String::new()),
                TokenKind::InterpStart,
                TokenKind::If,
                id("x"),
                TokenKind::LBrace,
                TokenKind::Int(1),
                TokenKind::RBrace,
                TokenKind::Else,
                TokenKind::LBrace,
                TokenKind::Int(2),
                TokenKind::RBrace,
                TokenKind::InterpEnd,
                TokenKind::StrEnd(String::new()),
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn interpolation_with_a_nested_string() {
        // "{ "{inner}" }" -- a string inside an interpolation inside a string.
        // entering the nested `"..."` makes its braces string-rules, not
        // brace-depth-rules, so the outer interpolation still closes at the right
        // `}`.
        assert_eq!(
            kinds("\"{ \"{inner}\" }\""),
            vec![
                TokenKind::StrStart(String::new()),
                TokenKind::InterpStart,
                // the nested string:
                TokenKind::StrStart(String::new()),
                TokenKind::InterpStart,
                id("inner"),
                TokenKind::InterpEnd,
                TokenKind::StrEnd(String::new()),
                // back in the outer interpolation:
                TokenKind::InterpEnd,
                TokenKind::StrEnd(String::new()),
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn the_fibonacci_interpolation() {
        // "fib({i}) = {fibonacci(i)}" -- `fib(` and `) = ` are literal text; the
        // `(i)` inside `{fibonacci(i)}` is a normal call sequence (LParen/RParen
        // do not affect brace depth, only `{`/`}` do).
        assert_eq!(
            kinds("\"fib({i}) = {fibonacci(i)}\""),
            vec![
                TokenKind::StrStart("fib(".to_string()),
                TokenKind::InterpStart,
                id("i"),
                TokenKind::InterpEnd,
                TokenKind::StrMid(") = ".to_string()),
                TokenKind::InterpStart,
                id("fibonacci"),
                TokenKind::LParen,
                id("i"),
                TokenKind::RParen,
                TokenKind::InterpEnd,
                TokenKind::StrEnd(String::new()),
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn unterminated_interpolation_errors_at_the_brace() {
        // "x{y" -- EOF before the `}`. the `{` is byte 2.
        match err("\"x{y") {
            QalaError::UnterminatedInterpolation { span } => {
                assert_eq!(span, Span::new(2, 1), "span should be the `{{` byte");
            }
            other => panic!("expected UnterminatedInterpolation, got {other:?}"),
        }
        // the `{` is found even when the string starts mid-source.
        match err("let s = \"a{b") {
            QalaError::UnterminatedInterpolation { span } => {
                assert_eq!(span.slice("let s = \"a{b"), "{");
            }
            other => panic!("expected UnterminatedInterpolation, got {other:?}"),
        }
        // a nested-brace interpolation that never closes the outermost `}`.
        assert!(matches!(
            err("\"{ {a: 1}.a "),
            QalaError::UnterminatedInterpolation { .. }
        ));
    }

    #[test]
    fn interpolation_fragment_spans_partition_the_source() {
        // "hi {x}!" -- StrStart covers `"hi `, InterpStart the `{`, InterpEnd the
        // `}`, StrEnd `!"`. every byte is covered exactly once.
        let src = "\"hi {x}!\"";
        let toks = Lexer::tokenize(src).unwrap();
        // toks: [StrStart, InterpStart, Ident, InterpEnd, StrEnd, Eof]
        assert_eq!(toks[0].kind, TokenKind::StrStart("hi ".to_string()));
        assert_eq!(toks[0].span.slice(src), "\"hi ");
        assert_eq!(toks[1].kind, TokenKind::InterpStart);
        assert_eq!(toks[1].span.slice(src), "{");
        assert_eq!(toks[2].kind, id("x"));
        assert_eq!(toks[2].span.slice(src), "x");
        assert_eq!(toks[3].kind, TokenKind::InterpEnd);
        assert_eq!(toks[3].span.slice(src), "}");
        assert_eq!(toks[4].kind, TokenKind::StrEnd("!".to_string()));
        assert_eq!(toks[4].span.slice(src), "!\"");
        assert_eq!(toks[5].kind, TokenKind::Eof);
    }

    #[test]
    fn the_six_examples_tokenize_to_eof_with_no_error() {
        // a smoke test that the token set is complete: every bundled example
        // lexes cleanly. the path is relative to this crate's manifest, so it
        // works regardless of the test runner's cwd; a missing file fails loudly.
        for name in ["hello", "fibonacci", "effects", "pattern-matching", "pipeline", "defer-demo"] {
            let path = format!(
                "{}/../playground/public/examples/{}.qala",
                env!("CARGO_MANIFEST_DIR"),
                name
            );
            let src = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("could not read example {path}: {e}"));
            let toks = Lexer::tokenize(&src)
                .unwrap_or_else(|e| panic!("example {name}.qala failed to tokenize: {e:?}"));
            assert_eq!(
                toks.last().map(|t| &t.kind),
                Some(&TokenKind::Eof),
                "example {name}.qala should end in Eof"
            );
            assert!(toks.len() > 1, "example {name}.qala should produce real tokens");
        }
    }

    #[test]
    fn tokenizing_is_deterministic() {
        // the lexer holds no global mutable state, so the same source always
        // produces the same token vector.
        let src = "fn main() is io {\n  let name = \"world\"\n  println(\"hello, {name}!\")\n}\n";
        let a = Lexer::tokenize(src).unwrap();
        let b = Lexer::tokenize(src).unwrap();
        assert_eq!(a, b);
        // also stable across an error case.
        let bad = "let x = \"unterminated";
        assert_eq!(Lexer::tokenize(bad), Lexer::tokenize(bad));
    }
}
