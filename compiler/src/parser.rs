//! the hand-written parser: [`Parser::parse`] turns the lexer's `Vec<Token>`
//! into an untyped [`Ast`], or the first syntax error. recursive descent for
//! items and statements, a Pratt (binding-power) loop for expressions.
//!
//! fail-fast: returns `Result<Ast, QalaError>` and stops at the first error.
//! multi-error collection is the type checker's job (Phase 3); the parser does
//! not synchronize. (the lexer's module doc mentions "Phase 2's parser has
//! resync points" -- a wording drift; recovery is deferred per CONTEXT.md.)
//!
//! a [`depth`] counter on [`Parser`] guards every recursive production against
//! pathologically nested input. Rust has no tail-call optimization; on WASM a
//! stack overflow aborts the module. past [`MAX_DEPTH`] the parser returns a
//! clean [`QalaError::Parse`] instead.
//!
//! no `unwrap` / `expect` / `panic!` / `unreachable!` is reachable from a
//! malformed token stream -- the same discipline the lexer documents; load-
//! bearing for Phase 6 where the compiler runs in a browser tab.

use crate::ast::*;
use crate::errors::QalaError;
use crate::span::Span;
use crate::token::{Token, TokenKind};

/// the recursion-depth cap, in productions. chosen generously enough that real
/// programs never approach it (a hand-written program rarely nests more than a
/// dozen levels deep) and conservatively enough that the native and WASM call
/// stacks have headroom. 256 frames at the rough budget of a Pratt-recursion
/// frame fits comfortably inside the default 1 MiB WASM stack and the default
/// 8 MiB native one.
pub const MAX_DEPTH: u32 = 256;

/// the parser state.
///
/// `tokens` is the lexer's output (assumed to end in [`TokenKind::Eof`], which
/// `lexer::Lexer::tokenize` guarantees). `pos` indexes into it. `depth` counts
/// active recursive productions for the stack-overflow guard. `open_delims`
/// stacks the still-open opener tokens and their spans: on a delimiter
/// mismatch the innermost (top) is blamed -- its span is the error's span,
/// per the "blame the opener" rule.
pub struct Parser<'t> {
    /// the token stream; always ends in [`TokenKind::Eof`].
    tokens: &'t [Token],
    /// the cursor: index into `tokens`. invariant: `pos < tokens.len()`,
    /// preserved because `advance` stops at `Eof` instead of stepping past it.
    pos: usize,
    /// recursion-depth counter. incremented by [`Parser::enter`], decremented
    /// by [`Parser::exit`]. an error past `MAX_DEPTH` keeps WASM alive.
    depth: u32,
    /// the still-open `(` / `[` / `{` delimiters and their opening spans, in
    /// open order. on a missing / wrong closer the top is blamed, so its span
    /// is the error's `span` -- the "look at the opener" pattern.
    open_delims: Vec<(TokenKind, Span)>,
}

impl<'t> Parser<'t> {
    /// parse a token stream into an untyped [`Ast`], or return the first
    /// syntax error.
    ///
    /// the input is expected to be a well-formed token stream ending in
    /// [`TokenKind::Eof`] (what [`crate::lexer::Lexer::tokenize`] produces).
    /// an empty program (just `[Eof]`) parses to an empty item list with no
    /// error. fail-fast: the first parse error stops the parse and is
    /// returned; later errors are not collected.
    pub fn parse(tokens: &'t [Token]) -> Result<Ast, QalaError> {
        let mut p = Parser {
            tokens,
            pos: 0,
            depth: 0,
            open_delims: Vec::new(),
        };
        let mut items: Ast = Vec::new();
        while !matches!(p.peek(), TokenKind::Eof) {
            items.push(p.parse_item()?);
        }
        Ok(items)
    }

    // ---- cursor --------------------------------------------------------------

    /// the kind at the cursor. the stream always ends in `Eof`, so `pos` is in
    /// range; this indexes `self.tokens[self.pos]` directly.
    fn peek(&self) -> &TokenKind {
        &self.tokens[self.pos].kind
    }

    /// the span at the cursor.
    fn peek_span(&self) -> Span {
        self.tokens[self.pos].span
    }

    /// the kind one past the cursor, or `Eof` if `pos+1` is out of range.
    /// useful for the few two-token decisions the grammar needs (e.g. is the
    /// token after `Ident` a `(` opening a call, or a `{` opening a struct
    /// literal).
    #[allow(dead_code)] // wired in by Task 3's struct-literal lookahead
    fn peek2(&self) -> &TokenKind {
        self.tokens
            .get(self.pos + 1)
            .map(|t| &t.kind)
            .unwrap_or(&TokenKind::Eof)
    }

    /// advance past the current token unless it is `Eof`, and return a
    /// reference to the token just consumed (or, if at `Eof`, the `Eof` token
    /// itself with the cursor still parked on it). this keeps `pos` in range
    /// for every subsequent `peek` -- the cursor never steps past `Eof`.
    fn advance(&mut self) -> &Token {
        let tok = &self.tokens[self.pos];
        if !matches!(tok.kind, TokenKind::Eof) {
            self.pos += 1;
        }
        // re-index so the borrow is independent of the local; the borrow
        // checker is happier this way and the runtime cost is one bounds-check.
        &self.tokens[self.pos.saturating_sub(1).min(self.tokens.len() - 1)]
    }

    /// true if the cursor is parked on a token of kind `k`. used for tentative
    /// look-ahead before deciding which production to enter.
    fn check(&self, k: &TokenKind) -> bool {
        self.peek() == k
    }

    /// consume the current token if it matches `k`, returning whether it did.
    /// the cheap "if-then-advance" idiom: `if self.eat(&Comma) { ... }`.
    fn eat(&mut self, k: &TokenKind) -> bool {
        if self.check(k) {
            self.advance();
            true
        } else {
            false
        }
    }

    /// consume a token of kind `k`, returning its span, or produce an error
    /// listing `k` as the sole expected kind.
    fn expect(&mut self, k: &TokenKind) -> Result<Span, QalaError> {
        if self.check(k) {
            Ok(self.advance().span)
        } else {
            Err(self.unexpected(&[k.clone()]))
        }
    }

    /// consume an [`TokenKind::Ident`], returning its decoded name and span,
    /// or produce an "expected an identifier" error.
    fn expect_ident(&mut self) -> Result<(String, Span), QalaError> {
        match self.peek().clone() {
            TokenKind::Ident(name) => {
                let span = self.advance().span;
                Ok((name, span))
            }
            _ => Err(self.unexpected(&[TokenKind::Ident(String::new())])),
        }
    }

    // ---- errors --------------------------------------------------------------

    /// build a parse error for "the current token is not legal here".
    ///
    /// if the cursor is at `Eof`, this is [`QalaError::UnexpectedEof`] with a
    /// zero-width span just past the last real token (computed from the
    /// previous token's `end`, per CONTEXT.md / RESEARCH Pitfall 6 -- not
    /// `src.len()`). otherwise it is [`QalaError::UnexpectedToken`] with the
    /// expected set the caller listed and the current token as `found`.
    fn unexpected(&self, expected: &[TokenKind]) -> QalaError {
        if matches!(self.peek(), TokenKind::Eof) {
            // the EOF span is the zero-width point immediately after the last
            // real token's end (or 0..0 for an empty source). this is more
            // accurate than the `Eof` token's own span when there is trivia
            // (whitespace, comments) before EOF.
            let span = if self.pos > 0 {
                let prev = &self.tokens[self.pos - 1];
                Span::new(prev.span.end(), 0)
            } else {
                Span::new(0, 0)
            };
            QalaError::UnexpectedEof {
                span,
                expected: expected.to_vec(),
            }
        } else {
            QalaError::UnexpectedToken {
                span: self.peek_span(),
                expected: expected.to_vec(),
                found: self.peek().clone(),
            }
        }
    }

    // ---- depth guard ---------------------------------------------------------

    /// enter a recursive production: increment `depth`, error past `MAX_DEPTH`.
    ///
    /// every recursive production (the Pratt loop, block parsing, pattern
    /// parsing, match arms, eventually the statement and item productions) is
    /// wrapped in `enter()? ... exit()` so a `((((...))))`-style input fails
    /// gracefully instead of overflowing the call stack.
    fn enter(&mut self) -> Result<(), QalaError> {
        self.depth = self.depth.saturating_add(1);
        if self.depth > MAX_DEPTH {
            return Err(QalaError::Parse {
                span: self.peek_span(),
                message: "expression nests too deeply".to_string(),
            });
        }
        Ok(())
    }

    /// leave a recursive production: decrement `depth`. saturating, so an
    /// imbalance can never panic.
    fn exit(&mut self) {
        self.depth = self.depth.saturating_sub(1);
    }

    // ---- delimited lists -----------------------------------------------------

    /// parse a `open elem (sep elem)* sep? close` list, returning the elements
    /// and the closing delimiter's span.
    ///
    /// blames the opener on a mismatch: if the closer is wrong or the input
    /// runs out before the close, the returned error is
    /// [`QalaError::UnclosedDelimiter`] with the *opener's* span. trailing
    /// separators are accepted for free because the loop re-checks for the
    /// closer after every `sep`.
    ///
    /// the caller is expected to have peeked the opener; this consumes it.
    fn delimited_list<T>(
        &mut self,
        open: TokenKind,
        close: TokenKind,
        sep: TokenKind,
        mut parse_one: impl FnMut(&mut Self) -> Result<T, QalaError>,
    ) -> Result<(Vec<T>, Span), QalaError> {
        let open_span = self.expect(&open)?;
        self.open_delims.push((open.clone(), open_span));
        let mut items: Vec<T> = Vec::new();
        loop {
            if self.check(&close) {
                break;
            }
            // depth guard the recursion through `parse_one`: a list of nested
            // lists is depth-equivalent to nested parens.
            items.push(parse_one(self)?);
            if !self.eat(&sep) {
                break;
            }
        }
        if self.check(&close) {
            let close_span = self.advance().span;
            self.open_delims.pop();
            Ok((items, close_span))
        } else {
            // the close is wrong or we hit Eof: blame the opener.
            self.open_delims.pop();
            Err(QalaError::UnclosedDelimiter {
                span: open_span,
                opener: open,
                found: self.peek().clone(),
            })
        }
    }

    // ---- items ---------------------------------------------------------------

    /// parse one top-level item.
    ///
    /// the item bodies (`fn`, `struct`, `enum`, `interface`) land in Plan 03.
    /// for now the entry point exists and rejects every item with a clean
    /// `expected an item` error, so `Parser::parse` is complete: an empty
    /// program parses, and any program with items is rejected by a real
    /// `QalaError` (not a `todo!()`).
    fn parse_item(&mut self) -> Result<Item, QalaError> {
        Err(self.unexpected(&[
            TokenKind::Fn,
            TokenKind::Struct,
            TokenKind::Enum,
            TokenKind::Interface,
        ]))
    }

    // ---- expressions (Pratt) -------------------------------------------------
    // filled in by the next task in this plan.
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;

    /// tokenize, asserting success; the parser's tests want a `Vec<Token>`,
    /// not the lexer's `Result`.
    fn lex(src: &str) -> Vec<Token> {
        Lexer::tokenize(src).expect("expected successful tokenize")
    }

    /// parse a whole program, asserting success.
    #[allow(dead_code)] // exercised by Plan 03's item tests
    fn parse_ok(src: &str) -> Ast {
        Parser::parse(&lex(src)).expect("expected a successful parse")
    }

    /// parse a whole program, asserting it failed, and return the error.
    fn parse_err(src: &str) -> QalaError {
        Parser::parse(&lex(src)).expect_err("expected a parse error")
    }

    // ---- skeleton + parse() entry point -------------------------------------

    #[test]
    fn empty_program_parses_to_no_items() {
        // `Ast = Vec<Item>`; the empty program is an empty vec, no error.
        let ast = parse_ok("");
        assert!(ast.is_empty());
    }

    #[test]
    fn whitespace_and_comments_only_parses_to_no_items() {
        // the lexer skips trivia; parser sees only `Eof` and is happy.
        let ast = parse_ok("   \n// a comment\n   ");
        assert!(ast.is_empty());
    }

    #[test]
    fn a_program_with_an_item_is_currently_rejected_with_a_clean_error() {
        // Plan 03 implements the item bodies; until then `parse_item` returns
        // a real `QalaError::UnexpectedToken` listing the four item kinds.
        let err = parse_err("fn f() {}");
        match err {
            QalaError::UnexpectedToken { ref expected, ref found, .. } => {
                assert!(expected.contains(&TokenKind::Fn));
                assert!(expected.contains(&TokenKind::Struct));
                assert!(expected.contains(&TokenKind::Enum));
                assert!(expected.contains(&TokenKind::Interface));
                assert_eq!(found, &TokenKind::Fn);
            }
            other => panic!("expected UnexpectedToken, got {other:?}"),
        }
        // the message says what was expected, naming the actual token found.
        assert!(err.message().contains("expected"));
    }

    // ---- cursor helpers -----------------------------------------------------

    #[test]
    fn advance_does_not_step_past_eof() {
        // the cursor stays on Eof regardless of how many times advance is
        // called -- the invariant the rest of the parser relies on.
        let toks = lex("");
        let mut p = Parser {
            tokens: &toks,
            pos: 0,
            depth: 0,
            open_delims: Vec::new(),
        };
        assert!(matches!(p.peek(), TokenKind::Eof));
        for _ in 0..10 {
            p.advance();
        }
        assert!(matches!(p.peek(), TokenKind::Eof));
    }

    #[test]
    fn check_eat_and_expect_compose() {
        // a stream of `+ - ;`: check sees the +, eat consumes it, expect
        // consumes the -, then expect on `;` succeeds, leaving Eof.
        let toks = lex("+ - ;");
        let mut p = Parser {
            tokens: &toks,
            pos: 0,
            depth: 0,
            open_delims: Vec::new(),
        };
        assert!(p.check(&TokenKind::Plus));
        assert!(!p.check(&TokenKind::Minus));
        assert!(p.eat(&TokenKind::Plus));
        assert!(!p.eat(&TokenKind::Plus)); // already consumed
        p.expect(&TokenKind::Minus).expect("the `-` should be there");
        p.expect(&TokenKind::Semi).expect("the `;` should be there");
        assert!(matches!(p.peek(), TokenKind::Eof));
    }

    #[test]
    fn expect_on_wrong_kind_returns_unexpected_token() {
        // expecting `)` when the cursor is on `,` produces UnexpectedToken
        // with the comma's span and the right expected/found data.
        let toks = lex(",");
        let mut p = Parser {
            tokens: &toks,
            pos: 0,
            depth: 0,
            open_delims: Vec::new(),
        };
        match p.expect(&TokenKind::RParen) {
            Err(QalaError::UnexpectedToken { span, expected, found }) => {
                assert_eq!(span, Span::new(0, 1));
                assert_eq!(expected, vec![TokenKind::RParen]);
                assert_eq!(found, TokenKind::Comma);
            }
            other => panic!("expected UnexpectedToken, got {other:?}"),
        }
    }

    #[test]
    fn expect_at_eof_returns_unexpected_eof_with_a_zero_width_span() {
        // "+" lexes to [Plus, Eof]; advance past the +, then expect a `;`.
        // The error should be UnexpectedEof, span zero-width right after the +.
        let toks = lex("+");
        let mut p = Parser {
            tokens: &toks,
            pos: 0,
            depth: 0,
            open_delims: Vec::new(),
        };
        p.advance(); // consume the `+`
        match p.expect(&TokenKind::Semi) {
            Err(QalaError::UnexpectedEof { span, expected }) => {
                // the `+` ends at byte 1; the EOF span is 1..1 (zero-width).
                assert_eq!(span, Span::new(1, 0));
                assert_eq!(expected, vec![TokenKind::Semi]);
            }
            other => panic!("expected UnexpectedEof, got {other:?}"),
        }
    }

    #[test]
    fn unexpected_at_eof_of_empty_source_is_a_zero_zero_span() {
        // an empty source: the EOF span falls back to 0..0 because there is
        // no previous token to compute from.
        let toks = lex("");
        let p = Parser {
            tokens: &toks,
            pos: 0,
            depth: 0,
            open_delims: Vec::new(),
        };
        match p.unexpected(&[TokenKind::Fn]) {
            QalaError::UnexpectedEof { span, .. } => {
                assert_eq!(span, Span::new(0, 0));
            }
            other => panic!("expected UnexpectedEof, got {other:?}"),
        }
    }

    #[test]
    fn expect_ident_returns_name_and_span() {
        let toks = lex("foo");
        let mut p = Parser {
            tokens: &toks,
            pos: 0,
            depth: 0,
            open_delims: Vec::new(),
        };
        let (name, span) = p.expect_ident().expect("expected ident");
        assert_eq!(name, "foo");
        assert_eq!(span, Span::new(0, 3));
        // expecting again at EOF gives an UnexpectedEof.
        match p.expect_ident() {
            Err(QalaError::UnexpectedEof { .. }) => {}
            other => panic!("expected UnexpectedEof, got {other:?}"),
        }
    }

    #[test]
    fn expect_ident_on_a_keyword_returns_unexpected_token() {
        // `fn` is a keyword; expecting an ident there should error, not bind
        // the keyword as a name.
        let toks = lex("fn");
        let mut p = Parser {
            tokens: &toks,
            pos: 0,
            depth: 0,
            open_delims: Vec::new(),
        };
        match p.expect_ident() {
            Err(QalaError::UnexpectedToken { found, .. }) => {
                assert_eq!(found, TokenKind::Fn);
            }
            other => panic!("expected UnexpectedToken, got {other:?}"),
        }
    }

    // ---- depth guard --------------------------------------------------------

    #[test]
    fn the_depth_guard_returns_a_clean_error_past_the_limit() {
        // enter() can be called MAX_DEPTH times without error; one more call
        // returns QalaError::Parse with a message mentioning deep nesting.
        let toks = lex("");
        let mut p = Parser {
            tokens: &toks,
            pos: 0,
            depth: 0,
            open_delims: Vec::new(),
        };
        for _ in 0..MAX_DEPTH {
            p.enter().expect("under the limit");
        }
        match p.enter() {
            Err(QalaError::Parse { message, .. }) => {
                assert!(message.contains("deeply"), "message was {message:?}");
            }
            other => panic!("expected QalaError::Parse, got {other:?}"),
        }
    }

    #[test]
    fn exit_undoes_enter_so_balanced_pairs_reset_the_counter() {
        let toks = lex("");
        let mut p = Parser {
            tokens: &toks,
            pos: 0,
            depth: 0,
            open_delims: Vec::new(),
        };
        for _ in 0..(MAX_DEPTH / 2) {
            p.enter().expect("under the limit");
            p.exit();
        }
        assert_eq!(p.depth, 0);
        // a long balanced run does not trip the guard.
        for _ in 0..(MAX_DEPTH * 10) {
            p.enter().expect("under the limit -- balanced");
            p.exit();
        }
        assert_eq!(p.depth, 0);
    }

    // ---- delimited list -----------------------------------------------------
    //
    // delimited_list is exercised end-to-end through the expression engine
    // (Task 3) on `(args)`, `[elems]`, struct literal `{ fields }`, match arms,
    // etc. it is also unit-tested here against a tiny "parse a list of idents"
    // closure to pin the open / sep / close behaviour and the "blame the
    // opener" rule independently of any expression machinery.

    #[test]
    fn delimited_list_handles_empty_and_trailing_separators() {
        // empty `()`.
        let toks = lex("()");
        let mut p = Parser {
            tokens: &toks,
            pos: 0,
            depth: 0,
            open_delims: Vec::new(),
        };
        let (items, close): (Vec<(String, Span)>, Span) = p
            .delimited_list(TokenKind::LParen, TokenKind::RParen, TokenKind::Comma, |q| {
                q.expect_ident()
            })
            .expect("empty list parses");
        assert!(items.is_empty());
        assert_eq!(close, Span::new(1, 1));

        // a list with a trailing comma: `(a, b,)`.
        let toks = lex("(a, b,)");
        let mut p = Parser {
            tokens: &toks,
            pos: 0,
            depth: 0,
            open_delims: Vec::new(),
        };
        let (items, _): (Vec<(String, Span)>, Span) = p
            .delimited_list(TokenKind::LParen, TokenKind::RParen, TokenKind::Comma, |q| {
                q.expect_ident()
            })
            .expect("trailing comma OK");
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].0, "a");
        assert_eq!(items[1].0, "b");
    }

    #[test]
    fn delimited_list_blames_the_opener_on_a_missing_closer() {
        // `(a, b` without the `)`: the error's span is the opening `(`.
        let toks = lex("(a, b");
        let mut p = Parser {
            tokens: &toks,
            pos: 0,
            depth: 0,
            open_delims: Vec::new(),
        };
        match p.delimited_list(
            TokenKind::LParen,
            TokenKind::RParen,
            TokenKind::Comma,
            |q| q.expect_ident(),
        ) {
            Err(QalaError::UnclosedDelimiter { span, opener, found }) => {
                assert_eq!(span, Span::new(0, 1));
                assert_eq!(opener, TokenKind::LParen);
                assert_eq!(found, TokenKind::Eof);
            }
            other => panic!("expected UnclosedDelimiter, got {other:?}"),
        }
    }

    #[test]
    fn delimited_list_blames_the_opener_on_a_wrong_closer() {
        // `(a]`: the wrong closer. blame the `(`.
        let toks = lex("(a]");
        let mut p = Parser {
            tokens: &toks,
            pos: 0,
            depth: 0,
            open_delims: Vec::new(),
        };
        match p.delimited_list(
            TokenKind::LParen,
            TokenKind::RParen,
            TokenKind::Comma,
            |q| q.expect_ident(),
        ) {
            Err(QalaError::UnclosedDelimiter { span, opener, found }) => {
                assert_eq!(span, Span::new(0, 1));
                assert_eq!(opener, TokenKind::LParen);
                assert_eq!(found, TokenKind::RBracket);
            }
            other => panic!("expected UnclosedDelimiter, got {other:?}"),
        }
    }

    // ---- determinism --------------------------------------------------------

    #[test]
    fn parsing_is_deterministic() {
        // the parser holds no global state and uses no maps with hashing-order
        // dependence (the precedence tables are `match` arms, not HashMaps;
        // TokenKind is not Eq, so it could not key one anyway).
        let src = "";
        let a = Parser::parse(&lex(src));
        let b = Parser::parse(&lex(src));
        assert_eq!(a, b);
        // and stable across an error case (a program with an item).
        let bad = "fn f() {}";
        assert_eq!(Parser::parse(&lex(bad)), Parser::parse(&lex(bad)));
    }
}
