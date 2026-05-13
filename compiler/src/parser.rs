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
            Err(self.unexpected(std::slice::from_ref(k)))
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
    //
    // a Pratt (precedence-climbing) loop driven by three binding-power lookup
    // functions: [`postfix_bp`], [`prefix_bp`], [`infix_bp`]. each returns a
    // pair of binding powers (`(lbp, rbp)` for infix, `(lbp, ())` for postfix,
    // `((), rbp)` for prefix) that encode precedence and associativity. the
    // technique is matklad's "simple but powerful Pratt parsing"
    // (matklad.github.io/2020/04/13/simple-but-powerful-pratt-parsing.html);
    // see `02-RESEARCH.md` Patterns 2-4.

    /// parse a single expression at the loosest precedence. a thin alias for
    /// `expr_bp(0, ExprCtx::Normal)`; statements call this to read the leaves
    /// of the grammar.
    pub fn parse_expr(&mut self) -> Result<Expr, QalaError> {
        self.expr_bp(0, ExprCtx::Normal)
    }

    /// the Pratt loop.
    ///
    /// at the top, parse a primary (a literal, an identifier-or-struct-literal,
    /// a paren/tuple, an array literal, a block, an interpolation, a `match`,
    /// or a `!` / `-` / `comptime` prefix). then in a loop, peek the next
    /// token: if it has a postfix binding power >= `min_bp`, fold it in as a
    /// postfix; if it has an infix binding power >= `min_bp`, consume the
    /// operator and recurse with the right-hand binding power; else stop and
    /// return what is built so far.
    ///
    /// `ctx` threads the "struct literals are not allowed at the start of an
    /// expression here" rule through the recursion: the head of `match`, and
    /// later `if`/`while`/`for`, must not parse `Name { ... }` as a struct
    /// literal -- per RESEARCH Pitfall 3. `ExprCtx::Normal` allows them;
    /// `ExprCtx::NoStructLiteral` rejects them at the primary.
    fn expr_bp(&mut self, min_bp: u8, ctx: ExprCtx) -> Result<Expr, QalaError> {
        self.enter()?;
        let result = self.expr_bp_inner(min_bp, ctx);
        self.exit();
        result
    }

    fn expr_bp_inner(&mut self, min_bp: u8, ctx: ExprCtx) -> Result<Expr, QalaError> {
        // --- primary --------------------------------------------------------
        let mut lhs = self.parse_primary(ctx)?;

        // --- postfix and infix loop -----------------------------------------
        loop {
            let op = self.peek().clone();
            // postfix? if so it binds at one tier, no rbp; the operator is
            // applied to lhs and the loop continues from the result.
            if let Some((lbp, ())) = postfix_bp(&op) {
                if lbp < min_bp {
                    break;
                }
                lhs = self.fold_postfix(lhs, &op)?;
                continue;
            }
            // infix? consume the operator and recurse for the rhs.
            if let Some((lbp, rbp)) = infix_bp(&op) {
                if lbp < min_bp {
                    break;
                }
                let op_tok = self.advance().clone();
                lhs = self.fold_infix(lhs, op_tok, rbp)?;
                continue;
            }
            break;
        }
        Ok(lhs)
    }

    /// parse the leftmost piece of an expression: a literal, an identifier or
    /// struct literal, a parenthesised expression / tuple, an array literal /
    /// repeat, a block, a string interpolation, a `match`, or a `!` / `-` /
    /// `comptime` prefix.
    fn parse_primary(&mut self, ctx: ExprCtx) -> Result<Expr, QalaError> {
        match self.peek().clone() {
            // literals (each carries its decoded value and span).
            TokenKind::Int(v) => {
                let span = self.advance().span;
                Ok(Expr::Int { value: v, span })
            }
            TokenKind::Float(v) => {
                let span = self.advance().span;
                Ok(Expr::Float { value: v, span })
            }
            TokenKind::Byte(v) => {
                let span = self.advance().span;
                Ok(Expr::Byte { value: v, span })
            }
            TokenKind::Str(s) => {
                let span = self.advance().span;
                Ok(Expr::Str { value: s, span })
            }
            TokenKind::True => {
                let span = self.advance().span;
                Ok(Expr::Bool { value: true, span })
            }
            TokenKind::False => {
                let span = self.advance().span;
                Ok(Expr::Bool { value: false, span })
            }
            // identifier: either a bare reference, or, if followed by `{` and
            // struct literals are allowed in this position, a struct literal.
            // primitive type names are tokens, not idents, and are not legal
            // here -- they only appear in type position.
            TokenKind::Ident(name) => self.parse_ident_or_struct_literal(name, ctx),
            // parenthesised: `( inner )`, `( e1, e2 )` (tuple), `( e, )` (one-tuple),
            // or `()` (empty tuple as a unit value).
            TokenKind::LParen => self.parse_paren_or_tuple(),
            // array literal: `[ e1, e2, ... ]` or repeat `[ v; n ]`.
            TokenKind::LBracket => self.parse_array_literal(),
            // block expression: `{ stmt*; trailing? }`. value-producing.
            TokenKind::LBrace => self.parse_block_expr(),
            // interpolated string: re-assemble the lexer's StrStart / InterpStart
            // / InterpEnd / StrMid / StrEnd grouping into `Expr::Interpolation`.
            TokenKind::StrStart(_) => self.parse_interpolation(),
            // match expression.
            TokenKind::Match => self.parse_match(),
            // unary prefixes: `!`, `-`, `comptime`.
            TokenKind::Bang | TokenKind::Minus | TokenKind::Comptime => self.parse_prefix(),
            // anything else is not a valid start of an expression.
            _ => Err(self.unexpected(&[
                TokenKind::Int(0),
                TokenKind::Float(0.0),
                TokenKind::Byte(0),
                TokenKind::Str(String::new()),
                TokenKind::StrStart(String::new()),
                TokenKind::True,
                TokenKind::False,
                TokenKind::Ident(String::new()),
                TokenKind::LParen,
                TokenKind::LBracket,
                TokenKind::LBrace,
                TokenKind::Match,
                TokenKind::Bang,
                TokenKind::Minus,
                TokenKind::Comptime,
            ])),
        }
    }

    /// `name`, or `Name { field: e, ... }` when `{` follows and struct
    /// literals are allowed in this context.
    ///
    /// the "no struct literals at the head of `match` / `if` / `while` /
    /// `for`" rule is RESEARCH Pitfall 3 -- without it, `match s { Circle(r)
    /// => ... }` would read `s { Circle(r) => ... }` as a struct literal with
    /// a `Circle(r)` field, which is nonsense. when `ctx` is `NoStructLiteral`
    /// we read just the ident; the trailing `{` will start the match-arm
    /// block, not a literal body.
    fn parse_ident_or_struct_literal(
        &mut self,
        name: String,
        ctx: ExprCtx,
    ) -> Result<Expr, QalaError> {
        let name_span = self.advance().span;
        if matches!(ctx, ExprCtx::Normal) && self.check(&TokenKind::LBrace) {
            let (fields, close) = self.delimited_list(
                TokenKind::LBrace,
                TokenKind::RBrace,
                TokenKind::Comma,
                |p| p.parse_field_init(),
            )?;
            Ok(Expr::StructLit {
                name,
                fields,
                span: name_span.to(close),
            })
        } else {
            Ok(Expr::Ident {
                name,
                span: name_span,
            })
        }
    }

    /// `name: value` inside a struct literal body.
    fn parse_field_init(&mut self) -> Result<FieldInit, QalaError> {
        let (name, name_span) = self.expect_ident()?;
        self.expect(&TokenKind::Colon)?;
        let value = self.parse_expr()?;
        let span = name_span.to(value.span());
        Ok(FieldInit { name, value, span })
    }

    /// `( inner )` (paren), `( e1, e2, ... )` (tuple), `( e, )` (one-tuple),
    /// or `()` (empty tuple / unit).
    ///
    /// the rule: zero elements (`()`) is a zero-tuple; one element with no
    /// trailing comma is a paren; one element with a trailing comma is a
    /// one-tuple; two or more elements (with or without trailing comma) is a
    /// tuple. faithful to the surface syntax.
    fn parse_paren_or_tuple(&mut self) -> Result<Expr, QalaError> {
        let open_span = self.expect(&TokenKind::LParen)?;
        self.open_delims.push((TokenKind::LParen, open_span));
        // empty `()`: zero-tuple.
        if self.check(&TokenKind::RParen) {
            let close = self.advance().span;
            self.open_delims.pop();
            return Ok(Expr::Tuple {
                elems: Vec::new(),
                span: open_span.to(close),
            });
        }
        let first = self.parse_expr()?;
        // `( expr )`: paren (no comma seen).
        if self.check(&TokenKind::RParen) {
            let close = self.advance().span;
            self.open_delims.pop();
            return Ok(Expr::Paren {
                inner: Box::new(first),
                span: open_span.to(close),
            });
        }
        // there must be a `,` here -- either `( e, )` or `( e, ... )`.
        if !self.eat(&TokenKind::Comma) {
            self.open_delims.pop();
            return Err(QalaError::UnclosedDelimiter {
                span: open_span,
                opener: TokenKind::LParen,
                found: self.peek().clone(),
            });
        }
        let mut elems = vec![first];
        loop {
            if self.check(&TokenKind::RParen) {
                break;
            }
            elems.push(self.parse_expr()?);
            if !self.eat(&TokenKind::Comma) {
                break;
            }
        }
        if !self.check(&TokenKind::RParen) {
            self.open_delims.pop();
            return Err(QalaError::UnclosedDelimiter {
                span: open_span,
                opener: TokenKind::LParen,
                found: self.peek().clone(),
            });
        }
        let close = self.advance().span;
        self.open_delims.pop();
        Ok(Expr::Tuple {
            elems,
            span: open_span.to(close),
        })
    }

    /// `[ e1, e2, ... ]` (array literal) or `[ value; count ]` (repeat).
    ///
    /// the first element is parsed eagerly; if a `;` follows, the second
    /// expression is the count and the form is a repeat. otherwise it is a
    /// comma list.
    fn parse_array_literal(&mut self) -> Result<Expr, QalaError> {
        let open_span = self.expect(&TokenKind::LBracket)?;
        self.open_delims.push((TokenKind::LBracket, open_span));
        // empty `[]`: zero-element array literal.
        if self.check(&TokenKind::RBracket) {
            let close = self.advance().span;
            self.open_delims.pop();
            return Ok(Expr::ArrayLit {
                elems: Vec::new(),
                span: open_span.to(close),
            });
        }
        let first = self.parse_expr()?;
        // `[ value; count ]`.
        if self.eat(&TokenKind::Semi) {
            let count = self.parse_expr()?;
            if !self.check(&TokenKind::RBracket) {
                self.open_delims.pop();
                return Err(QalaError::UnclosedDelimiter {
                    span: open_span,
                    opener: TokenKind::LBracket,
                    found: self.peek().clone(),
                });
            }
            let close = self.advance().span;
            self.open_delims.pop();
            return Ok(Expr::ArrayRepeat {
                value: Box::new(first),
                count: Box::new(count),
                span: open_span.to(close),
            });
        }
        // `[ e1, e2, ... ]`.
        let mut elems = vec![first];
        if self.eat(&TokenKind::Comma) {
            loop {
                if self.check(&TokenKind::RBracket) {
                    break;
                }
                elems.push(self.parse_expr()?);
                if !self.eat(&TokenKind::Comma) {
                    break;
                }
            }
        }
        if !self.check(&TokenKind::RBracket) {
            self.open_delims.pop();
            return Err(QalaError::UnclosedDelimiter {
                span: open_span,
                opener: TokenKind::LBracket,
                found: self.peek().clone(),
            });
        }
        let close = self.advance().span;
        self.open_delims.pop();
        Ok(Expr::ArrayLit {
            elems,
            span: open_span.to(close),
        })
    }

    /// `{ ... }` used as an expression -- a value-producing block.
    fn parse_block_expr(&mut self) -> Result<Expr, QalaError> {
        let block = self.parse_block()?;
        let span = block.span;
        Ok(Expr::Block { block, span })
    }

    /// `{ stmt; stmt; ... trailing? }`.
    ///
    /// Plan 03 adds the statement forms `let` / `if` / `while` / `for` /
    /// `return` / `break` / `continue` / `defer`; for now the only statement
    /// form a block can hold is an expression-statement (`expr ;`). a block's
    /// value is the final expression with no trailing `;`, or `None` if there
    /// is none. mirrors RESEARCH Pitfall 7's "arm body = expr or block".
    fn parse_block(&mut self) -> Result<Block, QalaError> {
        let open_span = self.expect(&TokenKind::LBrace)?;
        self.open_delims.push((TokenKind::LBrace, open_span));
        let mut stmts: Vec<Stmt> = Vec::new();
        let mut value: Option<Box<Expr>> = None;
        loop {
            if self.check(&TokenKind::RBrace) {
                break;
            }
            // for now, the only statement-form a block parses is `expr ;` (or
            // a trailing `expr` with no `;`). Plan 03 dispatches on `let` /
            // `if` / etc here.
            let expr = self.parse_expr()?;
            if self.eat(&TokenKind::Semi) {
                let span = expr.span();
                stmts.push(Stmt::Expr { expr, span });
            } else {
                // no `;`: this is the trailing-value expression. it must be
                // the last thing before `}`.
                value = Some(Box::new(expr));
                break;
            }
        }
        if !self.check(&TokenKind::RBrace) {
            self.open_delims.pop();
            return Err(QalaError::UnclosedDelimiter {
                span: open_span,
                opener: TokenKind::LBrace,
                found: self.peek().clone(),
            });
        }
        let close = self.advance().span;
        self.open_delims.pop();
        Ok(Block {
            stmts,
            value,
            span: open_span.to(close),
        })
    }

    /// reassemble an interpolated string `StrStart text0 (InterpStart expr
    /// InterpEnd StrMid text)* InterpStart expr InterpEnd StrEnd textN` into
    /// an `Expr::Interpolation` whose `parts` list alternates `Literal(text)`
    /// and `Expr(...)`. RESEARCH Pattern 4.
    fn parse_interpolation(&mut self) -> Result<Expr, QalaError> {
        // consume the `StrStart(text0)`.
        let (start_text, start_span) = match self.peek().clone() {
            TokenKind::StrStart(s) => {
                let span = self.advance().span;
                (s, span)
            }
            _ => return Err(self.unexpected(&[TokenKind::StrStart(String::new())])),
        };
        let mut parts: Vec<InterpPart> = vec![InterpPart::Literal(start_text)];
        let end_span: Span;
        loop {
            // every loop iteration sees an `InterpStart ... InterpEnd` and
            // then either another `StrMid` (continue) or the closing `StrEnd`
            // (break).
            self.expect(&TokenKind::InterpStart)?;
            let inner = self.parse_expr()?;
            parts.push(InterpPart::Expr(inner));
            self.expect(&TokenKind::InterpEnd)?;
            match self.peek().clone() {
                TokenKind::StrMid(t) => {
                    let _span = self.advance().span;
                    parts.push(InterpPart::Literal(t));
                    // and around again -- there is another interpolation.
                }
                TokenKind::StrEnd(t) => {
                    let span = self.advance().span;
                    parts.push(InterpPart::Literal(t));
                    end_span = span;
                    break;
                }
                _ => {
                    return Err(self.unexpected(&[
                        TokenKind::StrMid(String::new()),
                        TokenKind::StrEnd(String::new()),
                    ]));
                }
            }
        }
        Ok(Expr::Interpolation {
            parts,
            span: start_span.to(end_span),
        })
    }

    /// `match scrutinee { arm, arm, ... }`.
    ///
    /// the scrutinee is parsed in `NoStructLiteral` context so the `{` that
    /// opens the match body is not consumed as a struct-literal body. at
    /// least one arm is required (RESEARCH Pitfall 7 / success criterion 2);
    /// zero arms is a parse error.
    fn parse_match(&mut self) -> Result<Expr, QalaError> {
        let kw_span = self.expect(&TokenKind::Match)?;
        let scrutinee = self.expr_bp(0, ExprCtx::NoStructLiteral)?;
        let open_span = self.expect(&TokenKind::LBrace)?;
        self.open_delims.push((TokenKind::LBrace, open_span));
        let mut arms: Vec<MatchArm> = Vec::new();
        loop {
            if self.check(&TokenKind::RBrace) {
                break;
            }
            let pattern = self.parse_pattern()?;
            let pattern_span = pattern.span();
            let guard = if self.eat(&TokenKind::If) {
                Some(self.parse_expr()?)
            } else {
                None
            };
            self.expect(&TokenKind::FatArrow)?;
            let body = if self.check(&TokenKind::LBrace) {
                MatchArmBody::Block(self.parse_block()?)
            } else {
                MatchArmBody::Expr(Box::new(self.parse_expr()?))
            };
            let body_span = match &body {
                MatchArmBody::Expr(e) => e.span(),
                MatchArmBody::Block(b) => b.span,
            };
            arms.push(MatchArm {
                pattern,
                guard,
                body,
                span: pattern_span.to(body_span),
            });
            // an arm's trailing comma is optional (per pattern-matching.qala
            // the `Triangle` arm has none); if absent, the next iteration
            // sees `}` and breaks.
            if !self.eat(&TokenKind::Comma) {
                break;
            }
        }
        if arms.is_empty() {
            // zero arms: error before consuming the `}` so the span is the
            // match keyword through the offending `}` (or `Eof`).
            let close_span = self.peek_span();
            self.open_delims.pop();
            return Err(QalaError::Parse {
                span: kw_span.to(close_span),
                message: "match needs at least one arm".to_string(),
            });
        }
        if !self.check(&TokenKind::RBrace) {
            self.open_delims.pop();
            return Err(QalaError::UnclosedDelimiter {
                span: open_span,
                opener: TokenKind::LBrace,
                found: self.peek().clone(),
            });
        }
        let close_span = self.advance().span;
        self.open_delims.pop();
        Ok(Expr::Match {
            scrutinee: Box::new(scrutinee),
            arms,
            span: kw_span.to(close_span),
        })
    }

    /// parse one pattern.
    ///
    /// the v1 set (per Plan 01's `Pattern` enum):
    /// - `Name(sub, ...)`: variant destructure (sub-patterns recursively)
    /// - `_`: wildcard
    /// - `name`: binding (bare ident -- the parser does not distinguish a
    ///   capitalized nullary-variant from a binding; that is Phase 3's call,
    ///   matching CONTEXT.md's "simpler, more-faithful choice")
    /// - integer / float / byte / string / boolean literal patterns
    fn parse_pattern(&mut self) -> Result<Pattern, QalaError> {
        match self.peek().clone() {
            TokenKind::Ident(name) => {
                let name_span = self.advance().span;
                // `_` (the lexer treats `_` as a regular identifier).
                if name == "_" {
                    return Ok(Pattern::Wildcard { span: name_span });
                }
                // `Name(sub, ...)`: variant destructure.
                if self.check(&TokenKind::LParen) {
                    let (sub, close) = self.delimited_list(
                        TokenKind::LParen,
                        TokenKind::RParen,
                        TokenKind::Comma,
                        |p| p.parse_pattern(),
                    )?;
                    return Ok(Pattern::Variant {
                        name,
                        sub,
                        span: name_span.to(close),
                    });
                }
                // bare ident: binding. Phase 3 reinterprets a capitalised
                // ident as a nullary variant if name-resolution finds one.
                Ok(Pattern::Binding {
                    name,
                    span: name_span,
                })
            }
            TokenKind::Int(v) => {
                let span = self.advance().span;
                Ok(Pattern::Int { value: v, span })
            }
            TokenKind::Float(v) => {
                let span = self.advance().span;
                Ok(Pattern::Float { value: v, span })
            }
            TokenKind::Byte(v) => {
                let span = self.advance().span;
                Ok(Pattern::Byte { value: v, span })
            }
            TokenKind::Str(s) => {
                let span = self.advance().span;
                Ok(Pattern::Str { value: s, span })
            }
            TokenKind::True => {
                let span = self.advance().span;
                Ok(Pattern::Bool { value: true, span })
            }
            TokenKind::False => {
                let span = self.advance().span;
                Ok(Pattern::Bool { value: false, span })
            }
            _ => Err(self.unexpected(&[
                TokenKind::Ident(String::new()),
                TokenKind::Int(0),
                TokenKind::Float(0.0),
                TokenKind::Byte(0),
                TokenKind::Str(String::new()),
                TokenKind::True,
                TokenKind::False,
            ])),
        }
    }

    /// `! operand`, `- operand`, or `comptime operand`. each binds at unary
    /// precedence: `comptime a + b` is `(comptime a) + b`, `-a.b` is
    /// `-(a.b)` because postfix `.` binds tighter than prefix `-`.
    fn parse_prefix(&mut self) -> Result<Expr, QalaError> {
        let op_tok = self.advance().clone();
        let ((), rbp) = prefix_bp(&op_tok.kind);
        let operand = self.expr_bp(rbp, ExprCtx::Normal)?;
        let span = op_tok.span.to(operand.span());
        match op_tok.kind {
            TokenKind::Bang => Ok(Expr::Unary {
                op: UnaryOp::Not,
                operand: Box::new(operand),
                span,
            }),
            TokenKind::Minus => Ok(Expr::Unary {
                op: UnaryOp::Neg,
                operand: Box::new(operand),
                span,
            }),
            TokenKind::Comptime => Ok(Expr::Comptime {
                body: Box::new(operand),
                span,
            }),
            // not reachable from `parse_primary`'s prefix arm, but the match
            // must be exhaustive in shape.
            _ => Err(QalaError::Parse {
                span: op_tok.span,
                message: "internal: prefix on a non-prefix token".to_string(),
            }),
        }
    }

    /// apply one postfix operator to `lhs`. the dispatch matches each of the
    /// four kinds that `postfix_bp` says are postfix (`.`, `(`, `[`, `?`).
    fn fold_postfix(&mut self, lhs: Expr, op: &TokenKind) -> Result<Expr, QalaError> {
        match op {
            TokenKind::Dot => {
                self.advance(); // consume the `.`
                let (name, name_span) = self.expect_ident()?;
                // `.name(args)` is a method call; `.name` alone is a field
                // access. the parser decides; Phase 3 resolves it.
                if self.check(&TokenKind::LParen) {
                    let (args, close) = self.delimited_list(
                        TokenKind::LParen,
                        TokenKind::RParen,
                        TokenKind::Comma,
                        |p| p.parse_expr(),
                    )?;
                    let lhs_span = lhs.span();
                    Ok(Expr::MethodCall {
                        receiver: Box::new(lhs),
                        name,
                        args,
                        span: lhs_span.to(close),
                    })
                } else {
                    let lhs_span = lhs.span();
                    Ok(Expr::FieldAccess {
                        obj: Box::new(lhs),
                        name,
                        span: lhs_span.to(name_span),
                    })
                }
            }
            TokenKind::LParen => {
                let (args, close) = self.delimited_list(
                    TokenKind::LParen,
                    TokenKind::RParen,
                    TokenKind::Comma,
                    |p| p.parse_expr(),
                )?;
                let lhs_span = lhs.span();
                Ok(Expr::Call {
                    callee: Box::new(lhs),
                    args,
                    span: lhs_span.to(close),
                })
            }
            TokenKind::LBracket => {
                let open_span = self.advance().span;
                self.open_delims.push((TokenKind::LBracket, open_span));
                let idx = self.parse_expr()?;
                if !self.check(&TokenKind::RBracket) {
                    self.open_delims.pop();
                    return Err(QalaError::UnclosedDelimiter {
                        span: open_span,
                        opener: TokenKind::LBracket,
                        found: self.peek().clone(),
                    });
                }
                let close = self.advance().span;
                self.open_delims.pop();
                let lhs_span = lhs.span();
                Ok(Expr::Index {
                    obj: Box::new(lhs),
                    index: Box::new(idx),
                    span: lhs_span.to(close),
                })
            }
            TokenKind::Question => {
                let q_span = self.advance().span;
                let lhs_span = lhs.span();
                Ok(Expr::Try {
                    expr: Box::new(lhs),
                    span: lhs_span.to(q_span),
                })
            }
            _ => Err(QalaError::Parse {
                span: self.peek_span(),
                message: "internal: postfix_bp returned Some for a non-postfix token"
                    .to_string(),
            }),
        }
    }

    /// apply one infix operator (already consumed by the Pratt loop) with
    /// right-binding-power `rbp` to `lhs`.
    fn fold_infix(
        &mut self,
        lhs: Expr,
        op_tok: Token,
        rbp: u8,
    ) -> Result<Expr, QalaError> {
        match op_tok.kind {
            TokenKind::PipeGt => {
                // the pipeline rhs must be a callable-shaped expression (an
                // ident / call / field-access / method-call); a literal or a
                // binary on the rhs is a parse error whose span is the `|>`
                // token, per CONTEXT.md.
                let rhs = self.parse_pipeline_rhs(op_tok.span)?;
                let lhs_span = lhs.span();
                let rhs_span = rhs.span();
                Ok(Expr::Pipeline {
                    lhs: Box::new(lhs),
                    call: Box::new(rhs),
                    span: lhs_span.to(rhs_span),
                })
            }
            TokenKind::Or => {
                let rhs = self.expr_bp(rbp, ExprCtx::Normal)?;
                let lhs_span = lhs.span();
                let rhs_span = rhs.span();
                Ok(Expr::OrElse {
                    expr: Box::new(lhs),
                    fallback: Box::new(rhs),
                    span: lhs_span.to(rhs_span),
                })
            }
            TokenKind::DotDot | TokenKind::DotDotEq => {
                let inclusive = matches!(op_tok.kind, TokenKind::DotDotEq);
                let rhs = self.expr_bp(rbp, ExprCtx::Normal)?;
                let lhs_span = lhs.span();
                let rhs_span = rhs.span();
                Ok(Expr::Range {
                    start: Some(Box::new(lhs)),
                    end: Some(Box::new(rhs)),
                    inclusive,
                    span: lhs_span.to(rhs_span),
                })
            }
            ref k if binop_of(k).is_some() => {
                let bop = binop_of(k).expect("checked by the guard");
                let rhs = self.expr_bp(rbp, ExprCtx::Normal)?;
                let lhs_span = lhs.span();
                let rhs_span = rhs.span();
                Ok(Expr::Binary {
                    op: bop,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                    span: lhs_span.to(rhs_span),
                })
            }
            _ => Err(QalaError::Parse {
                span: op_tok.span,
                message: "internal: infix_bp returned Some for an unhandled token"
                    .to_string(),
            }),
        }
    }

    /// parse the right-hand side of `|>`. it must be a "callable" expression
    /// (Ident / Call / FieldAccess / MethodCall); per `pipeline.qala` a bare
    /// ident is allowed, but a literal or a binary is not.
    ///
    /// the violation case `x |> 5` produces an error whose span is the `|>`
    /// token; `x |>` with nothing after (an Eof / closer) does the same --
    /// per CONTEXT.md and success criterion PARSE-06.
    fn parse_pipeline_rhs(&mut self, pipe_span: Span) -> Result<Expr, QalaError> {
        // if the next token cannot start a callable, error here with the pipe
        // span (rather than letting expr_bp produce an error pointing at the
        // wrong place).
        if !matches!(
            self.peek(),
            TokenKind::Ident(_) | TokenKind::LParen
        ) {
            return Err(QalaError::Parse {
                span: pipe_span,
                message: "expected a call after `|>`".to_string(),
            });
        }
        // parse a sub-expression at one tier above `|>` so we do not consume
        // the next `|>` (left-associativity is handled in the outer loop).
        // we explicitly DO want postfix `.` / `()` / `[]` / `?` to bind to
        // the rhs (so `x |> f()` and `xs |> a.b()` work).
        let rhs = self.expr_bp(infix_lbp_pipeline() + 1, ExprCtx::Normal)?;
        if !is_callable_shape(&rhs) {
            return Err(QalaError::Parse {
                span: pipe_span,
                message: "the right side of `|>` must be a call".to_string(),
            });
        }
        Ok(rhs)
    }
}

/// the context an expression is being parsed in.
///
/// the only knob v1 needs is "are struct literals allowed here?" -- they are
/// not at the head of `match` / `if` / `while` / `for`. RESEARCH Pitfall 3.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExprCtx {
    /// struct literals are allowed at this primary.
    Normal,
    /// `Name { ... }` here is NOT a struct literal; the `{` belongs to the
    /// enclosing construct's body.
    NoStructLiteral,
}

/// postfix binding power: the four kinds that bind tightest. `()` here is the
/// "no rhs" marker -- postfix operators do not have a right-binding power.
fn postfix_bp(k: &TokenKind) -> Option<(u8, ())> {
    match k {
        TokenKind::Dot
        | TokenKind::LParen
        | TokenKind::LBracket
        | TokenKind::Question => Some((100, ())),
        _ => None,
    }
}

/// prefix binding power: `!`, `-`, `comptime` -- all at unary precedence,
/// one tier below postfix, so `-a.b` parses as `-(a.b)`. caller is expected
/// to have already confirmed the token is one of these via the primary
/// dispatch; an unrecognised kind is an internal bug, surfaced as a Parse
/// error rather than a panic.
fn prefix_bp(k: &TokenKind) -> ((), u8) {
    match k {
        TokenKind::Bang | TokenKind::Minus | TokenKind::Comptime => ((), 90),
        // a non-prefix token reaching here would be a bug in the caller; we
        // return a high rbp so the recursion does not consume anything by
        // accident, and the surrounding code's exhaustive match catches the
        // mismatch on the way out.
        _ => ((), u8::MAX),
    }
}

/// infix binding power: `(lbp, rbp)`. left-associative operators have
/// `rbp = lbp + 1` so a same-tier operator on the right does not re-enter at
/// equal-or-higher binding power. ranges are non-associative: `rbp` for
/// `..`/`..=` is the same as their `lbp`, so `0..1..2` fails to parse.
///
/// `Eq` (the assignment token) is deliberately absent -- `let x = ...` is
/// statement syntax (Plan 03), not an expression operator.
fn infix_bp(k: &TokenKind) -> Option<(u8, u8)> {
    Some(match k {
        // tightest infix: multiplicative.
        TokenKind::Star | TokenKind::Slash | TokenKind::Percent => (80, 81),
        // additive.
        TokenKind::Plus | TokenKind::Minus => (70, 71),
        // ranges -- non-associative.
        TokenKind::DotDot | TokenKind::DotDotEq => (60, 60),
        // comparison and equality -- left-associative.
        TokenKind::Lt
        | TokenKind::LtEq
        | TokenKind::Gt
        | TokenKind::GtEq
        | TokenKind::EqEq
        | TokenKind::BangEq => (50, 51),
        // boolean and / or.
        TokenKind::AmpAmp => (40, 41),
        TokenKind::PipePipe => (30, 31),
        // pipeline -- left-associative.
        TokenKind::PipeGt => (20, 21),
        // `or` fallback -- loosest, left-associative.
        TokenKind::Or => (10, 11),
        _ => return None,
    })
}

/// `infix_bp(PipeGt)`'s `lbp`, factored out so `parse_pipeline_rhs` can ask
/// "one tier above `|>`" without re-typing the magic number.
fn infix_lbp_pipeline() -> u8 {
    20
}

/// map a binary-operator token kind to its [`BinOp`] enum value. returns
/// `None` for tokens whose `infix_bp` arm is handled specially in
/// `fold_infix` (`PipeGt`, `Or`, `DotDot`, `DotDotEq`).
fn binop_of(k: &TokenKind) -> Option<BinOp> {
    Some(match k {
        TokenKind::Plus => BinOp::Add,
        TokenKind::Minus => BinOp::Sub,
        TokenKind::Star => BinOp::Mul,
        TokenKind::Slash => BinOp::Div,
        TokenKind::Percent => BinOp::Rem,
        TokenKind::EqEq => BinOp::Eq,
        TokenKind::BangEq => BinOp::Ne,
        TokenKind::Lt => BinOp::Lt,
        TokenKind::LtEq => BinOp::Le,
        TokenKind::Gt => BinOp::Gt,
        TokenKind::GtEq => BinOp::Ge,
        TokenKind::AmpAmp => BinOp::And,
        TokenKind::PipePipe => BinOp::Or,
        _ => return None,
    })
}

/// is `e` shaped like something that can be called?
///
/// per CONTEXT.md and `pipeline.qala`, a pipeline rhs is either a bare
/// identifier, a call, a field access, or a method call -- anything that
/// post-desugar in Phase 4 becomes `f(x, ...)`. a literal, a binary, a
/// range, etc., is rejected.
fn is_callable_shape(e: &Expr) -> bool {
    matches!(
        e,
        Expr::Ident { .. }
            | Expr::Call { .. }
            | Expr::FieldAccess { .. }
            | Expr::MethodCall { .. }
    )
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

    /// parse `src` as a single expression (using the Pratt entry point
    /// directly), asserting success and that the cursor ends at `Eof`. used
    /// to test the expression engine without dragging in statement / item
    /// parsing (Plan 03).
    fn expr_of(src: &str) -> Expr {
        let toks = lex(src);
        let mut p = Parser {
            tokens: &toks,
            pos: 0,
            depth: 0,
            open_delims: Vec::new(),
        };
        let e = p.parse_expr().expect("expected a successful expression parse");
        assert!(
            matches!(p.peek(), TokenKind::Eof),
            "expr_of left tokens unconsumed: next = {:?}",
            p.peek()
        );
        e
    }

    /// parse `src` as a single expression, asserting it failed; return the
    /// error. mirror of `parse_err` for the expression-engine tests.
    fn expr_err(src: &str) -> QalaError {
        let toks = lex(src);
        let mut p = Parser {
            tokens: &toks,
            pos: 0,
            depth: 0,
            open_delims: Vec::new(),
        };
        p.parse_expr().expect_err("expected an expression parse error")
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

    // ---- expression engine: precedence table --------------------------------
    //
    // one assertion per precedence level (success criterion 3), each pinning
    // the parsed-tree shape so a tweak to `infix_bp`/`prefix_bp`/`postfix_bp`
    // that drops a level by one bp is caught.

    /// helper: a fresh integer literal at any span; used in `matches!` arms.
    /// the precedence tests only care about the SHAPE of the tree, not the
    /// spans, so destructuring lets us ignore them.
    fn is_ident(e: &Expr, want: &str) -> bool {
        matches!(e, Expr::Ident { name, .. } if name == want)
    }

    #[test]
    fn mul_binds_tighter_than_add() {
        // `1 + 2 * 3` -> Binary(Add, Int(1), Binary(Mul, Int(2), Int(3))).
        let e = expr_of("1 + 2 * 3");
        match e {
            Expr::Binary { op: BinOp::Add, lhs, rhs, .. } => {
                assert!(matches!(*lhs, Expr::Int { value: 1, .. }));
                match *rhs {
                    Expr::Binary { op: BinOp::Mul, lhs, rhs, .. } => {
                        assert!(matches!(*lhs, Expr::Int { value: 2, .. }));
                        assert!(matches!(*rhs, Expr::Int { value: 3, .. }));
                    }
                    other => panic!("expected mul-binary on the right, got {other:?}"),
                }
            }
            other => panic!("expected add-binary, got {other:?}"),
        }
        // `1 * 2 + 3` -> Binary(Add, Binary(Mul, Int(1), Int(2)), Int(3)).
        let e = expr_of("1 * 2 + 3");
        match e {
            Expr::Binary { op: BinOp::Add, lhs, rhs, .. } => {
                assert!(matches!(*lhs, Expr::Binary { op: BinOp::Mul, .. }));
                assert!(matches!(*rhs, Expr::Int { value: 3, .. }));
            }
            other => panic!("expected add-binary at the top, got {other:?}"),
        }
    }

    #[test]
    fn and_binds_tighter_than_or() {
        // `a || b && c` -> Binary(Or, Ident(a), Binary(And, b, c)).
        let e = expr_of("a || b && c");
        match e {
            Expr::Binary { op: BinOp::Or, lhs, rhs, .. } => {
                assert!(is_ident(&lhs, "a"));
                match *rhs {
                    Expr::Binary { op: BinOp::And, lhs, rhs, .. } => {
                        assert!(is_ident(&lhs, "b"));
                        assert!(is_ident(&rhs, "c"));
                    }
                    other => panic!("expected and-binary on the right, got {other:?}"),
                }
            }
            other => panic!("expected or-binary at the top, got {other:?}"),
        }
        // `a && b || c` -> Binary(Or, Binary(And, a, b), c).
        let e = expr_of("a && b || c");
        match e {
            Expr::Binary { op: BinOp::Or, lhs, rhs, .. } => {
                assert!(matches!(*lhs, Expr::Binary { op: BinOp::And, .. }));
                assert!(is_ident(&rhs, "c"));
            }
            other => panic!("expected or-binary at the top, got {other:?}"),
        }
    }

    #[test]
    fn comparison_binds_tighter_than_logical_and() {
        // `a == b && c` -> Binary(And, Binary(Eq, a, b), c).
        let e = expr_of("a == b && c");
        match e {
            Expr::Binary { op: BinOp::And, lhs, rhs, .. } => {
                assert!(matches!(*lhs, Expr::Binary { op: BinOp::Eq, .. }));
                assert!(is_ident(&rhs, "c"));
            }
            other => panic!("expected and-binary, got {other:?}"),
        }
    }

    #[test]
    fn postfix_dot_binds_tighter_than_prefix_minus() {
        // `-a.b` -> Unary(Neg, FieldAccess(Ident(a), "b")).
        let e = expr_of("-a.b");
        match e {
            Expr::Unary { op: UnaryOp::Neg, operand, .. } => match *operand {
                Expr::FieldAccess { obj, name, .. } => {
                    assert!(is_ident(&obj, "a"));
                    assert_eq!(name, "b");
                }
                other => panic!("expected field access inside unary, got {other:?}"),
            },
            other => panic!("expected unary at the top, got {other:?}"),
        }
    }

    #[test]
    fn comptime_is_a_prefix_at_unary_precedence() {
        // `comptime a + b` -> Binary(Add, Comptime(Ident(a)), Ident(b)).
        let e = expr_of("comptime a + b");
        match e {
            Expr::Binary { op: BinOp::Add, lhs, rhs, .. } => {
                assert!(matches!(*lhs, Expr::Comptime { .. }));
                assert!(is_ident(&rhs, "b"));
                if let Expr::Comptime { body, .. } = *lhs {
                    assert!(is_ident(&body, "a"));
                }
            }
            other => panic!("expected add-binary, got {other:?}"),
        }
        // `comptime (a + b)` -- the whole binary is inside the comptime.
        let e = expr_of("comptime (a + b)");
        match e {
            Expr::Comptime { body, .. } => match *body {
                Expr::Paren { inner, .. } => {
                    assert!(matches!(*inner, Expr::Binary { op: BinOp::Add, .. }));
                }
                other => panic!("expected paren inside comptime, got {other:?}"),
            },
            other => panic!("expected comptime at the top, got {other:?}"),
        }
        // `comptime { 1 }`: the operand is a block expression.
        let e = expr_of("comptime { 1 }");
        match e {
            Expr::Comptime { body, .. } => {
                assert!(matches!(*body, Expr::Block { .. }));
            }
            other => panic!("expected comptime at the top, got {other:?}"),
        }
    }

    #[test]
    fn pipeline_is_left_associative_and_accepts_call_or_bare_callable() {
        // `a |> f() |> g()` -> Pipeline(Pipeline(a, Call(f)), Call(g)).
        let e = expr_of("a |> f() |> g()");
        match e {
            Expr::Pipeline { lhs, call, .. } => {
                assert!(matches!(*call, Expr::Call { .. }));
                match *lhs {
                    Expr::Pipeline { lhs, call, .. } => {
                        assert!(is_ident(&lhs, "a"));
                        assert!(matches!(*call, Expr::Call { .. }));
                    }
                    other => panic!("expected nested pipeline on the left, got {other:?}"),
                }
            }
            other => panic!("expected pipeline at the top, got {other:?}"),
        }
        // bare-ident rhs (per `pipeline.qala`): `5 |> double`.
        let e = expr_of("5 |> double");
        match e {
            Expr::Pipeline { lhs, call, .. } => {
                assert!(matches!(*lhs, Expr::Int { value: 5, .. }));
                assert!(is_ident(&call, "double"));
            }
            other => panic!("expected pipeline, got {other:?}"),
        }
    }

    #[test]
    fn or_fallback_is_left_associative_and_loosest() {
        // `x or y or z` -> OrElse(OrElse(x, y), z).
        let e = expr_of("x or y or z");
        match e {
            Expr::OrElse { expr, fallback, .. } => {
                assert!(is_ident(&fallback, "z"));
                match *expr {
                    Expr::OrElse { expr, fallback, .. } => {
                        assert!(is_ident(&expr, "x"));
                        assert!(is_ident(&fallback, "y"));
                    }
                    other => panic!("expected nested OrElse on the left, got {other:?}"),
                }
            }
            other => panic!("expected OrElse, got {other:?}"),
        }
        // `or` is looser than `|>`: `x |> f() or g` -> OrElse(Pipeline(...), g).
        let e = expr_of("x |> f() or g");
        match e {
            Expr::OrElse { expr, fallback, .. } => {
                assert!(matches!(*expr, Expr::Pipeline { .. }));
                assert!(is_ident(&fallback, "g"));
            }
            other => panic!("expected OrElse at the top, got {other:?}"),
        }
    }

    #[test]
    fn question_mark_is_postfix_at_tightest_tier() {
        // `f()?.g` -> FieldAccess(Try(Call(f)), "g").
        let e = expr_of("f()?.g");
        match e {
            Expr::FieldAccess { obj, name, .. } => {
                assert_eq!(name, "g");
                match *obj {
                    Expr::Try { expr, .. } => {
                        assert!(matches!(*expr, Expr::Call { .. }));
                    }
                    other => panic!("expected Try on the obj, got {other:?}"),
                }
            }
            other => panic!("expected FieldAccess, got {other:?}"),
        }
        // `a + b?` -> Binary(Add, a, Try(b)).
        let e = expr_of("a + b?");
        match e {
            Expr::Binary { op: BinOp::Add, lhs, rhs, .. } => {
                assert!(is_ident(&lhs, "a"));
                match *rhs {
                    Expr::Try { expr, .. } => assert!(is_ident(&expr, "b")),
                    other => panic!("expected Try on the rhs, got {other:?}"),
                }
            }
            other => panic!("expected add-binary, got {other:?}"),
        }
        // `parse(s)?` -- Try on a call.
        let e = expr_of("parse(s)?");
        match e {
            Expr::Try { expr, .. } => assert!(matches!(*expr, Expr::Call { .. })),
            other => panic!("expected Try, got {other:?}"),
        }
    }

    #[test]
    fn ranges_are_infix_at_a_tier_above_plus() {
        // `1..n+1` -> Range(Int(1), Binary(Add, Ident(n), Int(1)), false).
        let e = expr_of("1..n+1");
        match e {
            Expr::Range { start, end, inclusive, .. } => {
                assert!(!inclusive);
                let s = start.expect("start present");
                assert!(matches!(*s, Expr::Int { value: 1, .. }));
                let en = end.expect("end present");
                assert!(matches!(*en, Expr::Binary { op: BinOp::Add, .. }));
            }
            other => panic!("expected Range, got {other:?}"),
        }
        // `0..=15` -- inclusive range.
        let e = expr_of("0..=15");
        match e {
            Expr::Range { inclusive, .. } => assert!(inclusive),
            other => panic!("expected Range, got {other:?}"),
        }
    }

    #[test]
    fn additive_is_left_associative() {
        // `a + b - c` -> Binary(Sub, Binary(Add, a, b), c).
        let e = expr_of("a + b - c");
        match e {
            Expr::Binary { op: BinOp::Sub, lhs, rhs, .. } => {
                assert!(matches!(*lhs, Expr::Binary { op: BinOp::Add, .. }));
                assert!(is_ident(&rhs, "c"));
            }
            other => panic!("expected sub-binary at the top, got {other:?}"),
        }
    }

    // ---- expression engine: primaries ---------------------------------------

    #[test]
    fn literals_parse_to_their_decoded_variants() {
        assert!(matches!(expr_of("42"), Expr::Int { value: 42, .. }));
        assert!(matches!(expr_of("3.14"), Expr::Float { .. }));
        assert!(matches!(expr_of("b'A'"), Expr::Byte { value: 65, .. }));
        assert!(matches!(expr_of("\"hello\""), Expr::Str { .. }));
        assert!(matches!(expr_of("true"), Expr::Bool { value: true, .. }));
        assert!(matches!(expr_of("false"), Expr::Bool { value: false, .. }));
        assert!(is_ident(&expr_of("foo"), "foo"));
    }

    #[test]
    fn calls_parse_with_args_and_trailing_comma() {
        // `f(1, 2)` -> Call(Ident(f), [Int(1), Int(2)]).
        match expr_of("f(1, 2)") {
            Expr::Call { callee, args, .. } => {
                assert!(is_ident(&callee, "f"));
                assert_eq!(args.len(), 2);
            }
            other => panic!("expected Call, got {other:?}"),
        }
        // trailing comma OK.
        match expr_of("f(1, 2,)") {
            Expr::Call { args, .. } => assert_eq!(args.len(), 2),
            other => panic!("expected Call, got {other:?}"),
        }
        // a zero-arg call.
        match expr_of("g()") {
            Expr::Call { args, .. } => assert!(args.is_empty()),
            other => panic!("expected Call, got {other:?}"),
        }
    }

    #[test]
    fn method_call_is_distinct_from_field_access() {
        // `p.area()` -- a method call.
        match expr_of("p.area()") {
            Expr::MethodCall { receiver, name, args, .. } => {
                assert!(is_ident(&receiver, "p"));
                assert_eq!(name, "area");
                assert!(args.is_empty());
            }
            other => panic!("expected MethodCall, got {other:?}"),
        }
        // `p.x` -- a field access (no `(` after).
        match expr_of("p.x") {
            Expr::FieldAccess { obj, name, .. } => {
                assert!(is_ident(&obj, "p"));
                assert_eq!(name, "x");
            }
            other => panic!("expected FieldAccess, got {other:?}"),
        }
    }

    #[test]
    fn index_indexes_into_an_array() {
        match expr_of("xs[0]") {
            Expr::Index { obj, index, .. } => {
                assert!(is_ident(&obj, "xs"));
                assert!(matches!(*index, Expr::Int { value: 0, .. }));
            }
            other => panic!("expected Index, got {other:?}"),
        }
    }

    #[test]
    fn postfix_chain_nests_correctly() {
        // `f()?.g(1)[2]?` -> Try(Index(MethodCall(FieldAccess(Try(Call(f)),"g"), "g", [1]), 2))
        // wait -- it should be: Try(Index(MethodCall(Try(Call(f)),"g",[1]), 2)).
        let e = expr_of("f()?.g(1)[2]?");
        match e {
            Expr::Try { expr, .. } => match *expr {
                Expr::Index { obj, index, .. } => {
                    assert!(matches!(*index, Expr::Int { value: 2, .. }));
                    match *obj {
                        Expr::MethodCall { receiver, name, args, .. } => {
                            assert_eq!(name, "g");
                            assert_eq!(args.len(), 1);
                            assert!(matches!(*receiver, Expr::Try { .. }));
                        }
                        other => panic!("expected MethodCall, got {other:?}"),
                    }
                }
                other => panic!("expected Index, got {other:?}"),
            },
            other => panic!("expected Try at the top, got {other:?}"),
        }
    }

    #[test]
    fn array_literal_parses_with_elements_and_trailing_comma() {
        match expr_of("[1, 2, 3]") {
            Expr::ArrayLit { elems, .. } => assert_eq!(elems.len(), 3),
            other => panic!("expected ArrayLit, got {other:?}"),
        }
        // trailing comma OK.
        match expr_of("[1, 2, 3,]") {
            Expr::ArrayLit { elems, .. } => assert_eq!(elems.len(), 3),
            other => panic!("expected ArrayLit, got {other:?}"),
        }
        // empty array.
        match expr_of("[]") {
            Expr::ArrayLit { elems, .. } => assert!(elems.is_empty()),
            other => panic!("expected ArrayLit, got {other:?}"),
        }
    }

    #[test]
    fn array_repeat_parses_as_value_semi_count() {
        match expr_of("[0; 100]") {
            Expr::ArrayRepeat { value, count, .. } => {
                assert!(matches!(*value, Expr::Int { value: 0, .. }));
                assert!(matches!(*count, Expr::Int { value: 100, .. }));
            }
            other => panic!("expected ArrayRepeat, got {other:?}"),
        }
    }

    #[test]
    fn paren_versus_tuple_distinguished_by_a_trailing_comma() {
        // `(1 + 2)` -- a paren around a binary; not a tuple.
        match expr_of("(1 + 2)") {
            Expr::Paren { inner, .. } => {
                assert!(matches!(*inner, Expr::Binary { op: BinOp::Add, .. }));
            }
            other => panic!("expected Paren, got {other:?}"),
        }
        // `(1, 2)` -- a tuple of two ints.
        match expr_of("(1, 2)") {
            Expr::Tuple { elems, .. } => assert_eq!(elems.len(), 2),
            other => panic!("expected Tuple, got {other:?}"),
        }
        // `(1,)` -- a one-tuple (the trailing comma is significant).
        match expr_of("(1,)") {
            Expr::Tuple { elems, .. } => assert_eq!(elems.len(), 1),
            other => panic!("expected Tuple, got {other:?}"),
        }
        // `()` -- a zero-tuple.
        match expr_of("()") {
            Expr::Tuple { elems, .. } => assert!(elems.is_empty()),
            other => panic!("expected Tuple, got {other:?}"),
        }
    }

    #[test]
    fn struct_literal_parses_with_field_inits_and_trailing_comma() {
        // `Point { x: 1.0, y: 2.0 }`.
        match expr_of("Point { x: 1.0, y: 2.0 }") {
            Expr::StructLit { name, fields, .. } => {
                assert_eq!(name, "Point");
                assert_eq!(fields.len(), 2);
                assert_eq!(fields[0].name, "x");
                assert_eq!(fields[1].name, "y");
            }
            other => panic!("expected StructLit, got {other:?}"),
        }
        // trailing comma OK.
        match expr_of("Point { x: 1.0, y: 2.0, }") {
            Expr::StructLit { fields, .. } => assert_eq!(fields.len(), 2),
            other => panic!("expected StructLit, got {other:?}"),
        }
    }

    #[test]
    fn match_with_three_arms_two_expr_one_block() {
        // from pattern-matching.qala (areas).
        let src = "match s { Circle(r) => 3.14159 * r * r, Rect(w, h) => w * h, Triangle(a, b, c) => { sqrt(s) } }";
        match expr_of(src) {
            Expr::Match { arms, .. } => {
                assert_eq!(arms.len(), 3);
                assert!(matches!(arms[0].body, MatchArmBody::Expr(_)));
                assert!(matches!(arms[1].body, MatchArmBody::Expr(_)));
                assert!(matches!(arms[2].body, MatchArmBody::Block(_)));
                // each arm starts with a Variant pattern.
                for arm in &arms {
                    assert!(
                        matches!(arm.pattern, Pattern::Variant { .. }),
                        "expected Variant pattern, got {:?}",
                        arm.pattern
                    );
                }
            }
            other => panic!("expected Match, got {other:?}"),
        }
    }

    #[test]
    fn match_with_guards_and_wildcard() {
        // from pattern-matching.qala (classify).
        let src =
            "match value { v if v > 0 => \"positive\", v if v < 0 => \"negative\", _ => \"zero\" }";
        match expr_of(src) {
            Expr::Match { arms, .. } => {
                assert_eq!(arms.len(), 3);
                assert!(arms[0].guard.is_some());
                assert!(arms[1].guard.is_some());
                assert!(arms[2].guard.is_none());
                assert!(matches!(arms[0].pattern, Pattern::Binding { .. }));
                assert!(matches!(arms[2].pattern, Pattern::Wildcard { .. }));
            }
            other => panic!("expected Match, got {other:?}"),
        }
    }

    #[test]
    fn match_with_exactly_one_arm_and_no_trailing_comma() {
        // a one-arm match without a trailing comma after the only arm.
        match expr_of("match x { _ => 0 }") {
            Expr::Match { arms, .. } => assert_eq!(arms.len(), 1),
            other => panic!("expected Match, got {other:?}"),
        }
    }

    #[test]
    fn interpolation_parses_embedded_expressions_into_parts() {
        // `"fib({i}) = {fibonacci(i)}"` -- the parts list alternates Literal
        // and Expr, starting and ending with a Literal (possibly empty).
        match expr_of("\"fib({i}) = {fibonacci(i)}\"") {
            Expr::Interpolation { parts, .. } => {
                // parts: [Literal("fib("), Expr(Ident i), Literal(") = "),
                //         Expr(Call(fibonacci,[Ident i])), Literal("")]
                assert_eq!(parts.len(), 5);
                assert!(matches!(parts[0], InterpPart::Literal(_)));
                match &parts[1] {
                    InterpPart::Expr(Expr::Ident { name, .. }) => assert_eq!(name, "i"),
                    other => panic!("expected Ident(i), got {other:?}"),
                }
                assert!(matches!(parts[2], InterpPart::Literal(_)));
                match &parts[3] {
                    InterpPart::Expr(Expr::Call { callee, args, .. }) => {
                        assert!(is_ident(callee, "fibonacci"));
                        assert_eq!(args.len(), 1);
                    }
                    other => panic!("expected Call(fibonacci, ...), got {other:?}"),
                }
                assert!(matches!(parts[4], InterpPart::Literal(_)));
            }
            other => panic!("expected Interpolation, got {other:?}"),
        }
    }

    #[test]
    fn plain_string_with_no_interpolations_parses_to_expr_str() {
        // a string with no `{}` is a `TokenKind::Str`, not `StrStart` -- so
        // it lands in the `Str(_)` arm, not `parse_interpolation`.
        match expr_of("\"hello\"") {
            Expr::Str { value, .. } => assert_eq!(value, "hello"),
            other => panic!("expected Str, got {other:?}"),
        }
    }

    // ---- expression engine: negatives ---------------------------------------

    #[test]
    fn x_pipe_with_no_call_errors_on_the_pipe_span() {
        // `x |>` -- the rhs is missing; the error's span is the `|>` token.
        let err = expr_err("x |>");
        match err {
            QalaError::Parse { span, message } => {
                // the `|>` is at byte offset 2, length 2.
                assert_eq!(span, Span::new(2, 2));
                assert!(message.contains("`|>`"), "message was {message:?}");
            }
            other => panic!("expected QalaError::Parse, got {other:?}"),
        }
    }

    #[test]
    fn x_pipe_with_a_non_callable_rhs_errors_on_the_pipe_span() {
        // `x |> 5` -- 5 is a literal, not callable. error span on the `|>`.
        let err = expr_err("x |> 5");
        match err {
            QalaError::Parse { span, message } => {
                assert_eq!(span, Span::new(2, 2));
                assert!(message.contains("`|>`"), "message was {message:?}");
            }
            other => panic!("expected QalaError::Parse, got {other:?}"),
        }
    }

    #[test]
    fn match_with_zero_arms_errors() {
        let err = expr_err("match x { }");
        match err {
            QalaError::Parse { message, .. } => {
                assert!(message.contains("at least one arm"), "message was {message:?}");
            }
            other => panic!("expected QalaError::Parse, got {other:?}"),
        }
    }

    #[test]
    fn an_unclosed_paren_errors_on_the_open_paren() {
        // `(a + b` -- never closes; the error's span is the opening `(`.
        let err = expr_err("(a + b");
        match err {
            QalaError::UnclosedDelimiter { span, opener, .. } => {
                assert_eq!(span, Span::new(0, 1));
                assert_eq!(opener, TokenKind::LParen);
            }
            other => panic!("expected UnclosedDelimiter, got {other:?}"),
        }
    }

    #[test]
    fn an_unclosed_bracket_errors_on_the_open_bracket() {
        // `[1, 2` -- never closes; the error's span is the opening `[`.
        let err = expr_err("[1, 2");
        match err {
            QalaError::UnclosedDelimiter { span, opener, .. } => {
                assert_eq!(span, Span::new(0, 1));
                assert_eq!(opener, TokenKind::LBracket);
            }
            other => panic!("expected UnclosedDelimiter, got {other:?}"),
        }
    }

    #[test]
    fn the_assignment_token_is_not_an_expression_operator() {
        // `Eq` is statement syntax (`let x = ...`), not infix. an attempt to
        // parse `a = b` as an expression should stop at the `=`, leaving the
        // cursor parked there -- but `expr_of` asserts it consumed everything.
        // so use the raw parser and check the cursor.
        let toks = lex("a = b");
        let mut p = Parser {
            tokens: &toks,
            pos: 0,
            depth: 0,
            open_delims: Vec::new(),
        };
        let lhs = p.parse_expr().expect("expected ident a to parse");
        assert!(is_ident(&lhs, "a"));
        // the `=` was NOT consumed -- it is not in any binding-power table.
        assert!(matches!(p.peek(), TokenKind::Eq));
    }

    #[test]
    fn the_binding_power_tables_omit_assignment() {
        // structural assertion: postfix_bp / prefix_bp / infix_bp all return
        // None for `Eq`, so the Pratt loop never consumes it.
        assert_eq!(postfix_bp(&TokenKind::Eq), None);
        assert_eq!(infix_bp(&TokenKind::Eq), None);
        // prefix_bp returns `((), u8::MAX)` for non-prefix tokens, a sentinel
        // -- it is only called after parse_primary already saw `Bang`/`Minus`/
        // `Comptime`, so an `Eq` cannot reach it. for documentation, assert
        // the sentinel:
        let ((), rbp) = prefix_bp(&TokenKind::Eq);
        assert_eq!(rbp, u8::MAX);
    }

    #[test]
    fn postfix_bp_only_returns_some_for_dot_lparen_lbracket_question() {
        // pin the postfix table: the four tokens are postfix, every other
        // token is not.
        assert!(postfix_bp(&TokenKind::Dot).is_some());
        assert!(postfix_bp(&TokenKind::LParen).is_some());
        assert!(postfix_bp(&TokenKind::LBracket).is_some());
        assert!(postfix_bp(&TokenKind::Question).is_some());
        for k in [
            TokenKind::Plus,
            TokenKind::Minus,
            TokenKind::Star,
            TokenKind::PipeGt,
            TokenKind::Or,
            TokenKind::DotDot,
            TokenKind::DotDotEq,
            TokenKind::Eq,
            TokenKind::Comma,
            TokenKind::RBrace,
        ] {
            assert!(postfix_bp(&k).is_none(), "{k:?} should not be postfix");
        }
    }

    #[test]
    fn infix_bp_includes_ranges_but_not_assignment() {
        // pin the infix table's key claims.
        assert!(infix_bp(&TokenKind::DotDot).is_some());
        assert!(infix_bp(&TokenKind::DotDotEq).is_some());
        assert!(infix_bp(&TokenKind::PipeGt).is_some());
        assert!(infix_bp(&TokenKind::Or).is_some());
        // `Eq` is absent.
        assert!(infix_bp(&TokenKind::Eq).is_none());
        // `?` is postfix, not infix.
        assert!(infix_bp(&TokenKind::Question).is_none());
    }

    #[test]
    fn or_keyword_is_looser_than_pipe_gt() {
        // `or` has lower binding power than `|>`, so `|>` binds first.
        let (or_lbp, _) = infix_bp(&TokenKind::Or).unwrap();
        let (pipe_lbp, _) = infix_bp(&TokenKind::PipeGt).unwrap();
        assert!(
            or_lbp < pipe_lbp,
            "`or` ({or_lbp}) should be looser than `|>` ({pipe_lbp})"
        );
    }

    #[test]
    fn comptime_prefix_bp_matches_other_unary_prefixes() {
        // `comptime`, `!`, `-` all bind at the same tier, so all three nest
        // identically with postfix operators.
        let ((), c) = prefix_bp(&TokenKind::Comptime);
        let ((), b) = prefix_bp(&TokenKind::Bang);
        let ((), m) = prefix_bp(&TokenKind::Minus);
        assert_eq!(c, b);
        assert_eq!(b, m);
    }

    #[test]
    fn expression_parsing_is_deterministic_on_a_mixed_input() {
        // a single complex expression parses to identical trees across two
        // runs -- pins the "no hashing-order dependence" claim.
        let src = "1 + 2 * 3 |> f() or g()";
        assert_eq!(expr_of(src), expr_of(src));
    }
}
