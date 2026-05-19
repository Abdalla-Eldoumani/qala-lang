//! statement and control-flow codegen for the ARM64 backend.
//!
//! [`Arm64Backend::compile_stmt`] and [`Arm64Backend::compile_block`] lower a
//! [`TypedStmt`] / [`TypedBlock`] to AArch64 instructions. like `expr.rs`, this
//! file is a second `impl Arm64Backend` block -- Rust allows an `impl` to span
//! several files of the same module -- while the struct and the program /
//! function walk live in `mod.rs`.
//!
//! ## what this file handles
//!
//! `let` bindings into stack slots, `if` / `else` / `else if` chains, `while`
//! loops, `for` loops over a range, `return`, `break`, `continue`, and a
//! statement-position expression. `defer` is a permanent rejection -- it is
//! deferred beyond v2's integer slice. Qala has no free-standing reassignment
//! statement (`ast::Stmt` and `TypedStmt` have nine variants, none an
//! `Assign`; `BinOp` has no `=`; the Pratt infix table omits the assignment
//! token), so there is no assignment arm: a `let` is a single, immutable-style
//! binding and a loop advances state through the loop construct, not through
//! mutation.
//!
//! ## the binding scope stack
//!
//! a `let` binding's stack slot is assigned by
//! [`FrameLayout::plan_frame`](super::frame::FrameLayout::plan_frame) in a
//! fixed walk order; this file re-walks the body in the identical order,
//! keeping a `binding_cursor` that counts binding occurrences. when
//! `compile_stmt` reaches the `k`-th `let` (or `for`) it reads slot `k` via
//! [`binding_slot`](super::frame::FrameLayout::binding_slot) and records
//! `(name, slot)` in the current scope. the scope stack -- a
//! `Vec<Vec<(String, i64)>>` -- is pushed on block entry and popped on exit;
//! [`resolve_name`](Arm64Backend::resolve_name) searches it top-down, so a
//! shadowing `let` resolves to the newest slot and a binding that has gone out
//! of scope is no longer visible.
//!
//! ## the loop-label stack
//!
//! `break` and `continue` need the innermost loop's labels. each loop pushes a
//! [`LoopLabels`] record -- its `continue` target (the test label for a
//! `while`, the increment label for a `for`) and its end label -- onto a stack
//! the [`TypedStmt::Break`] / [`TypedStmt::Continue`] arms read.

use crate::errors::QalaError;
use crate::span::Span;
use crate::typed_ast::{TypedBlock, TypedElseBranch, TypedExpr, TypedStmt};

use super::Arm64Backend;

/// the branch targets of one active loop, for `break` and `continue`.
///
/// pushed onto the backend's loop-label stack when a `while` or `for` opens,
/// popped when it closes. `break` branches to `end`; `continue` branches to
/// `continue_target` -- the test label for a `while`, the increment label for
/// a `for` (so a `continue` runs the `for`'s increment before the test).
pub(crate) struct LoopLabels {
    /// where `continue` branches: a `while`'s test label, a `for`'s increment
    /// label.
    pub(crate) continue_target: String,
    /// where `break` branches: the loop's end label.
    pub(crate) end: String,
}

impl Arm64Backend {
    /// walk a block: push a fresh binding scope, emit every statement, emit the
    /// trailing value, pop the scope.
    ///
    /// the scope push/pop is what makes `let` shadowing and out-of-scope
    /// resolution correct: a `let` inside this block records its `(name, slot)`
    /// in the scope pushed here, and the pop drops those bindings so a sibling
    /// block does not see them. the trailing value's result, if any, lands in
    /// `x0` -- it is the block's value.
    ///
    /// Phase 12 emits straight-line code with no dead-code elimination: a
    /// statement after a `return` still emits, which is correct (merely
    /// unreachable), so this returns `Result<(), QalaError>` rather than a
    /// terminator flag.
    pub(super) fn compile_block(&mut self, block: &TypedBlock) -> Result<(), QalaError> {
        self.scopes.push(Vec::new());
        for stmt in &block.stmts {
            if let Err(e) = self.compile_stmt(stmt) {
                // pop the scope even on an error path so the scope stack stays
                // balanced if the caller continues accumulating errors.
                self.scopes.pop();
                return Err(e);
            }
        }
        if let Some(value) = &block.value {
            if let Err(e) = self.compile_expr(value) {
                self.scopes.pop();
                return Err(e);
            }
        }
        self.scopes.pop();
        Ok(())
    }

    /// compile one statement.
    ///
    /// every construct the integer core does not emit -- `defer`, a `for` over
    /// a non-range iterable, a `for` range with a missing bound, a
    /// `break`/`continue` outside a loop -- returns a [`QalaError::Type`]
    /// carrying the offending span, never a panic.
    pub(super) fn compile_stmt(&mut self, stmt: &TypedStmt) -> Result<(), QalaError> {
        match stmt {
            TypedStmt::Let { name, init, span, .. } => self.compile_let(name, init, *span),
            TypedStmt::If { cond, then_block, else_branch, .. } => {
                self.compile_if(cond, then_block, else_branch.as_ref())
            }
            TypedStmt::While { cond, body, .. } => self.compile_while(cond, body),
            TypedStmt::For { var, iter, body, span, .. } => {
                self.compile_for(var, iter, body, *span)
            }
            // `return` or `return expr`: evaluate the value into x0, then jump
            // to the function's single epilogue block.
            TypedStmt::Return { value, span } => {
                if let Some(e) = value {
                    self.compile_expr(e)?;
                }
                let fn_name = self.current_fn.clone().ok_or_else(|| QalaError::Type {
                    span: *span,
                    message: "arm64 backend: return outside a function".to_string(),
                })?;
                let label = self.epilogue_label(&fn_name);
                self.asm.emit_insn(&format!("b       {label}"));
                Ok(())
            }
            // `break` / `continue` -> a branch to the innermost loop's labels.
            // the loop-label stack is empty only for a break/continue outside
            // any loop -- the typechecker rejects that, so this is defensive.
            TypedStmt::Break { span } => {
                let target = self
                    .loops
                    .last()
                    .map(|l| l.end.clone())
                    .ok_or_else(|| self.unsupported_stmt(*span, "break outside a loop"))?;
                self.asm.emit_insn(&format!("b       {target}"));
                Ok(())
            }
            TypedStmt::Continue { span } => {
                let target = self
                    .loops
                    .last()
                    .map(|l| l.continue_target.clone())
                    .ok_or_else(|| {
                        self.unsupported_stmt(*span, "continue outside a loop")
                    })?;
                self.asm.emit_insn(&format!("b       {target}"));
                Ok(())
            }
            // a statement-position expression: its value is discarded, but the
            // instructions still run.
            TypedStmt::Expr { expr, .. } => self.compile_expr(expr),
            // `defer` is a permanent rejection -- deferred beyond v2.
            TypedStmt::Defer { span, .. } => {
                Err(self.unsupported_stmt(*span, "the defer statement"))
            }
        }
    }

    /// compile a `let` binding: evaluate the initializer into `x0`, store it
    /// into the binding's stack slot, and record the name in the current scope.
    ///
    /// the slot comes from `plan_frame`'s positional binding list -- this is
    /// the `binding_cursor`-th binding the body walk reaches, so it reads slot
    /// `binding_cursor` and advances the cursor. a shadowing `let` of the same
    /// name records a second `(name, slot)` pair in the scope; `resolve_name`'s
    /// top-down search then finds the newest.
    fn compile_let(
        &mut self,
        name: &str,
        init: &TypedExpr,
        span: Span,
    ) -> Result<(), QalaError> {
        // the initializer result lands in x0.
        self.compile_expr(init)?;
        let slot = self.next_binding_slot(span)?;
        self.asm
            .emit_insn_commented(&format!("str     x0, [fp, {slot}]"), name);
        self.bind_name(name, slot);
        Ok(())
    }

    /// compile an `if` / `else` / `else if`.
    ///
    /// the condition's `x0` is a 0/1 bool; `cbz` skips the then-block when it
    /// is false. with an `else`: `cbz` to the else label, the then-block, a
    /// `b` over the else, the else label, the else branch, the end label.
    /// without: `cbz` straight to the end label. an `else if` is a boxed
    /// [`TypedStmt::If`] -- this recurses through `compile_stmt`.
    fn compile_if(
        &mut self,
        cond: &TypedExpr,
        then_block: &TypedBlock,
        else_branch: Option<&TypedElseBranch>,
    ) -> Result<(), QalaError> {
        let end = self.labels.fresh("if_end");
        match else_branch {
            Some(branch) => {
                let else_label = self.labels.fresh("if_else");
                // cond -> x0; if false, branch to the else arm.
                self.compile_expr(cond)?;
                self.asm.emit_insn(&format!("cbz     x0, {else_label}"));
                // the then-block, then a jump over the else arm.
                self.compile_block(then_block)?;
                self.asm.emit_insn(&format!("b       {end}"));
                // the else arm: a final block, or a recursive `else if`.
                self.asm.emit_label(&else_label);
                match branch {
                    TypedElseBranch::Block(b) => self.compile_block(b)?,
                    TypedElseBranch::If(boxed) => self.compile_stmt(boxed)?,
                }
                self.asm.emit_label(&end);
            }
            None => {
                // no else: if the condition is false, skip straight to the end.
                self.compile_expr(cond)?;
                self.asm.emit_insn(&format!("cbz     x0, {end}"));
                self.compile_block(then_block)?;
                self.asm.emit_label(&end);
            }
        }
        Ok(())
    }

    /// compile a `while` loop.
    ///
    /// Task 3 replaces this placeholder with the real test-at-bottom codegen.
    fn compile_while(
        &mut self,
        _cond: &TypedExpr,
        body: &TypedBlock,
    ) -> Result<(), QalaError> {
        Err(self.unsupported_stmt(body.span, "while loops"))
    }

    /// compile a `for` loop over a range.
    ///
    /// Task 3 replaces this placeholder with the real counter-based codegen.
    fn compile_for(
        &mut self,
        _var: &str,
        _iter: &TypedExpr,
        _body: &TypedBlock,
        span: Span,
    ) -> Result<(), QalaError> {
        Err(self.unsupported_stmt(span, "for loops"))
    }

    /// the stack slot of the next body binding occurrence, advancing the cursor.
    ///
    /// `plan_frame` assigned binding slots in a fixed walk order; the
    /// `binding_cursor` counts how many this walk has consumed. an out-of-range
    /// cursor means the `stmt.rs` walk diverged from the `plan_frame` walk -- a
    /// backend bug, surfaced as a [`QalaError`] rather than a panic.
    fn next_binding_slot(&mut self, span: Span) -> Result<i64, QalaError> {
        let index = self.binding_cursor;
        let slot = self.frame().binding_slot(index).ok_or_else(|| QalaError::Type {
            span,
            message: format!("arm64 backend: binding occurrence {index} has no slot"),
        })?;
        self.binding_cursor += 1;
        Ok(slot)
    }

    /// record a name and its stack slot in the current (innermost) scope.
    ///
    /// a shadowing binding appends a second pair; [`resolve_name`] searches
    /// top-down so the newest wins. the scope stack always has at least one
    /// frame here -- `compile_block` pushes one before walking statements.
    fn bind_name(&mut self, name: &str, slot: i64) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.push((name.to_string(), slot));
        }
    }

    /// resolve a name to its stack slot: a `let`/`for` binding (searched
    /// innermost-scope-first), or a function parameter.
    ///
    /// the scope stack is searched from the top down so a shadowing `let`
    /// resolves to its own slot and a binding that has left scope is invisible.
    /// a name not in any scope falls back to the parameter slot map. `None`
    /// means the name resolves to neither -- a backend bug the caller turns
    /// into a [`QalaError`].
    pub(super) fn resolve_name(&self, name: &str) -> Option<i64> {
        for scope in self.scopes.iter().rev() {
            for (n, slot) in scope.iter().rev() {
                if n == name {
                    return Some(*slot);
                }
            }
        }
        self.frame().slot_of(name)
    }

    /// build the rejection error for a statement construct the integer core
    /// does not emit.
    pub(super) fn unsupported_stmt(&self, span: Span, what: &str) -> QalaError {
        QalaError::Type {
            span,
            message: format!("the arm64 backend does not yet support {what}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;
    use crate::parser::Parser;
    use crate::typechecker::check_program;
    use crate::typed_ast::TypedAst;

    /// lex, parse, and typecheck `src` into a typed AST, asserting no errors.
    fn typecheck(src: &str) -> TypedAst {
        let tokens = Lexer::tokenize(src).expect("lex failed");
        let ast = Parser::parse(&tokens).expect("parse failed");
        let (typed, terrors, _) = check_program(&ast, src);
        assert!(terrors.is_empty(), "typecheck errors: {terrors:?}");
        typed
    }

    /// compile `src` to assembly, panicking on any backend error.
    fn compile_ok(src: &str) -> String {
        let typed = typecheck(src);
        super::super::compile_arm64(&typed, src)
            .unwrap_or_else(|e| panic!("arm64 errors: {e:?}"))
    }

    #[test]
    fn a_let_stores_x0_into_its_stack_slot() {
        // `let x = 7` evaluates 7 into x0 then stores it to the binding slot.
        let out = compile_ok("fn f() -> i64 { let x = 7\nx }");
        assert!(out.contains("mov     x0, 7"), "{out}");
        assert!(out.contains("str     x0, [fp,"), "missing the let store: {out}");
    }

    #[test]
    fn an_ident_use_of_a_let_loads_its_slot() {
        // after `let x = 7` the binding is at a slot; `x` then loads from it.
        // the str slot and the ldr slot must be the same offset.
        let out = compile_ok("fn f() -> i64 { let x = 7\nx }");
        let store = out
            .lines()
            .find(|l| l.contains("str     x0, [fp,"))
            .expect("no store");
        let load = out
            .lines()
            .find(|l| l.contains("ldr     x0, [fp,"))
            .expect("no load");
        // both lines reference the same [fp, N] -- compare the bracketed slot.
        let store_slot = &store[store.find("[fp,").unwrap()..store.find(']').unwrap()];
        let load_slot = &load[load.find("[fp,").unwrap()..load.find(']').unwrap()];
        assert_eq!(store_slot, load_slot, "let store and ident load slots differ");
    }

    #[test]
    fn a_shadowing_let_resolves_the_ident_to_the_newest_slot() {
        // `let x = 1  let x = 2  x`: the trailing `x` must load the SECOND
        // binding's slot, not the first. the two `let`s store into distinct
        // slots; the final ldr matches the second store's slot.
        let out = compile_ok("fn f() -> i64 { let x = 1\nlet x = 2\nx }");
        let stores: Vec<&str> = out
            .lines()
            .filter(|l| l.contains("str     x0, [fp,"))
            .collect();
        assert_eq!(stores.len(), 2, "two lets -> two stores: {out}");
        let second_store = stores[1];
        let second_slot =
            &second_store[second_store.find("[fp,").unwrap()..second_store.find(']').unwrap()];
        let load = out
            .lines()
            .rev()
            .find(|l| l.contains("ldr     x0, [fp,"))
            .expect("no load");
        let load_slot = &load[load.find("[fp,").unwrap()..load.find(']').unwrap()];
        assert_eq!(load_slot, second_slot, "ident must resolve to the shadowing let");
    }

    #[test]
    fn an_if_without_else_emits_one_cbz_and_an_end_label() {
        let out = compile_ok("fn f(c: bool) -> i64 { if c { return 1 }\n0 }");
        assert!(out.contains("cbz     x0, .Lif_end_"), "missing cbz: {out}");
        assert!(out.contains(".Lif_end_"), "missing end label: {out}");
        // no else label for an else-less if.
        assert!(!out.contains(".Lif_else_"), "unexpected else label: {out}");
    }

    #[test]
    fn an_if_else_emits_both_the_else_and_end_labels() {
        // an `if`/`else` is a statement, not an expression -- each arm uses an
        // explicit `return` and the function has a trailing fallback value.
        let out = compile_ok("fn f(c: bool) -> i64 { if c { return 1 } else { return 2 }\n0 }");
        assert!(out.contains("cbz     x0, .Lif_else_"), "missing cbz to else: {out}");
        assert!(out.contains(".Lif_else_"), "missing else label: {out}");
        assert!(out.contains(".Lif_end_"), "missing end label: {out}");
        // the then-block ends with a jump over the else arm.
        assert!(out.contains("b       .Lif_end_"), "missing jump over else: {out}");
    }

    #[test]
    fn an_else_if_chain_emits_nested_labels() {
        // `if a {} else if b {} else {}` -- the else-if recurses, so a second
        // pair of if labels appears with distinct counter suffixes.
        let out = compile_ok(
            "fn f(a: bool, b: bool) -> i64 { \
             if a { return 1 } else if b { return 2 } else { return 3 }\n0 }",
        );
        let else_labels: Vec<&str> = out
            .lines()
            .filter(|l| l.trim_start().starts_with(".Lif_else_"))
            .collect();
        assert_eq!(else_labels.len(), 2, "an else-if chain has two else labels: {out}");
    }

    #[test]
    fn a_return_in_a_block_branches_to_the_epilogue() {
        let out = compile_ok("fn f(c: bool) -> i64 { if c { return 9 }\n0 }");
        assert!(out.contains("b       .Lf_epilogue"), "{out}");
    }

    #[test]
    fn a_let_inside_an_if_gets_its_own_slot() {
        // the body has an outer let and a let inside the if-block; both must
        // store, and into different slots.
        let out = compile_ok(
            "fn f(c: bool) -> i64 { let a = 1\nif c { let b = 2\nreturn b }\na }",
        );
        let stores: Vec<&str> = out
            .lines()
            .filter(|l| l.contains("str     x0, [fp,"))
            .collect();
        assert!(stores.len() >= 2, "outer and inner let both store: {out}");
    }

    #[test]
    fn defer_is_rejected_cleanly() {
        // `defer` is a permanent rejection -- a clean error, never a panic.
        // build the typed AST directly: a `defer` expression that is a plain
        // value, so no unsupported call shadows the defer rejection.
        let stmt = TypedStmt::Defer {
            expr: TypedExpr::Int {
                value: 1,
                ty: crate::types::QalaType::I64,
                span: Span::new(0, 1),
            },
            span: Span::new(0, 5),
        };
        let mut backend = Arm64Backend::new("");
        let err = backend
            .compile_stmt(&stmt)
            .expect_err("a defer must be rejected");
        match err {
            QalaError::Type { message, .. } => {
                assert!(message.contains("defer"), "message: {message}");
            }
            other => panic!("expected QalaError::Type, got {other:?}"),
        }
    }

    #[test]
    fn a_break_outside_a_loop_is_a_clean_error_not_a_panic() {
        // the typechecker normally rejects this; the backend is defensive --
        // an empty loop-label stack yields a QalaError, never a panic.
        // build the typed AST directly so the typecheck guard does not fire.
        let stmt = TypedStmt::Break { span: Span::new(0, 1) };
        let mut backend = Arm64Backend::new("");
        let err = backend
            .compile_stmt(&stmt)
            .expect_err("a break outside a loop must be rejected");
        match err {
            QalaError::Type { message, .. } => {
                assert!(message.contains("break"), "message: {message}");
            }
            other => panic!("expected QalaError::Type, got {other:?}"),
        }
    }
}
