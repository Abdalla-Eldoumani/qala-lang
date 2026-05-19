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
        if let Some(value) = &block.value
            && let Err(e) = self.compile_expr(value)
        {
            self.scopes.pop();
            return Err(e);
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
            TypedStmt::Let {
                name, init, span, ..
            } => self.compile_let(name, init, *span),
            TypedStmt::If {
                cond,
                then_block,
                else_branch,
                ..
            } => self.compile_if(cond, then_block, else_branch.as_ref()),
            TypedStmt::While { cond, body, .. } => self.compile_while(cond, body),
            TypedStmt::For {
                var,
                iter,
                body,
                span,
                ..
            } => self.compile_for(var, iter, body, *span),
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
                    .ok_or_else(|| self.unsupported_stmt(*span, "continue outside a loop"))?;
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
    fn compile_let(&mut self, name: &str, init: &TypedExpr, span: Span) -> Result<(), QalaError> {
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

    /// compile a `while` loop, test-at-bottom.
    ///
    /// the shape is `b test`, the body label, the body, the test label, the
    /// condition, `cbnz` back to the body, the end label. testing at the
    /// bottom -- one unconditional branch in, one conditional branch per
    /// iteration -- is one fewer branch per iteration than a test-at-top loop.
    /// `break` branches to the end label, `continue` to the test label, so the
    /// loop-label record's `continue` target is the test label.
    fn compile_while(&mut self, cond: &TypedExpr, body: &TypedBlock) -> Result<(), QalaError> {
        let body_label = self.labels.fresh("while_body");
        let test_label = self.labels.fresh("while_test");
        let end_label = self.labels.fresh("while_end");
        // jump straight to the test: the body runs only if the test passes.
        self.asm.emit_insn(&format!("b       {test_label}"));
        self.asm.emit_label(&body_label);
        // the loop is active for the body walk -- break/continue resolve here.
        self.loops.push(LoopLabels {
            continue_target: test_label.clone(),
            end: end_label.clone(),
        });
        let body_result = self.compile_block(body);
        // pop the record whether or not the body compiled, so the stack stays
        // balanced when the caller accumulates the error.
        self.loops.pop();
        body_result?;
        // the test: re-evaluate the condition, branch back if still true.
        self.asm.emit_label(&test_label);
        self.compile_expr(cond)?;
        self.asm.emit_insn(&format!("cbnz    x0, {body_label}"));
        self.asm.emit_label(&end_label);
        Ok(())
    }

    /// compile a `for` loop over a range: `for var in start..end` or
    /// `start..=end`.
    ///
    /// the integer core iterates a range only -- a `for` over any other
    /// iterable, or a range with a missing bound, is an unsupported construct
    /// and returns a [`QalaError`]. the lowered shape: store `start` into the
    /// loop variable's slot and `end` into a dedicated slot (evaluated once,
    /// not per iteration), `b test`, the body label, the body, the increment
    /// label, the increment, the test label, the `var < end` (or `<=`) check,
    /// `b.lt`/`b.le` back to the body, the end label.
    ///
    /// `continue` branches to the increment label -- so a `continue` runs the
    /// increment before the test -- which is why the increment has its own
    /// label and the loop-label record's `continue` target is that label.
    fn compile_for(
        &mut self,
        var: &str,
        iter: &TypedExpr,
        body: &TypedBlock,
        span: Span,
    ) -> Result<(), QalaError> {
        // the integer core supports a range iterable only.
        let (start, end, inclusive) = match iter {
            TypedExpr::Range {
                start,
                end,
                inclusive,
                ..
            } => (start, end, *inclusive),
            _ => {
                return Err(self.unsupported_stmt(span, "for over a non-range iterable"));
            }
        };
        // a `for` range must have both bounds -- `for i in ..n` is unsupported.
        let start = start
            .as_ref()
            .ok_or_else(|| self.unsupported_stmt(span, "a for range with no start bound"))?;
        let end = end
            .as_ref()
            .ok_or_else(|| self.unsupported_stmt(span, "a for range with no end bound"))?;

        // plan_frame reserved two slots for this `for`: the loop variable
        // first, then the `end` bound. claim them in that order.
        let var_slot = self.next_binding_slot(span)?;
        let end_slot = self.next_binding_slot(span)?;

        // counter init: start -> the var slot.
        self.compile_expr(start)?;
        self.asm
            .emit_insn_commented(&format!("str     x0, [fp, {var_slot}]"), var);
        // the end bound, evaluated ONCE into its dedicated slot.
        self.compile_expr(end)?;
        self.asm
            .emit_insn_commented(&format!("str     x0, [fp, {end_slot}]"), "for end");

        let body_label = self.labels.fresh("for_body");
        let incr_label = self.labels.fresh("for_incr");
        let test_label = self.labels.fresh("for_test");
        let end_label = self.labels.fresh("for_end");

        // the loop variable is in scope for the body -- record it so an Ident
        // reference to it resolves to var_slot.
        self.bind_name(var, var_slot);

        self.asm.emit_insn(&format!("b       {test_label}"));
        self.asm.emit_label(&body_label);
        // `continue` branches to the increment label, not the test, so the
        // increment always runs before the next test.
        self.loops.push(LoopLabels {
            continue_target: incr_label.clone(),
            end: end_label.clone(),
        });
        let body_result = self.compile_block(body);
        self.loops.pop();
        body_result?;

        // the increment: var = var + 1. the body falls through to here.
        self.asm.emit_label(&incr_label);
        self.asm.emit_insn(&format!("ldr     x0, [fp, {var_slot}]"));
        self.asm.emit_insn("add     x0, x0, 1");
        self.asm.emit_insn(&format!("str     x0, [fp, {var_slot}]"));

        // the test: load var and end, compare, branch back into the body. an
        // inclusive `..=` range uses `b.le`, an exclusive `..` uses `b.lt`.
        self.asm.emit_label(&test_label);
        self.asm.emit_insn(&format!("ldr     x0, [fp, {var_slot}]"));
        self.asm.emit_insn(&format!("ldr     x9, [fp, {end_slot}]"));
        self.asm.emit_insn("cmp     x0, x9");
        let branch = if inclusive { "b.le" } else { "b.lt" };
        self.asm.emit_insn(&format!("{branch}    {body_label}"));
        self.asm.emit_label(&end_label);
        Ok(())
    }

    /// the stack slot of the next body binding occurrence, advancing the cursor.
    ///
    /// `plan_frame` assigned binding slots in a fixed walk order; the
    /// `binding_cursor` counts how many this walk has consumed. an out-of-range
    /// cursor means the `stmt.rs` walk diverged from the `plan_frame` walk -- a
    /// backend bug, surfaced as a [`QalaError`] rather than a panic.
    fn next_binding_slot(&mut self, span: Span) -> Result<i64, QalaError> {
        let index = self.binding_cursor;
        let slot = self
            .frame()
            .binding_slot(index)
            .ok_or_else(|| QalaError::Type {
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

    /// the four control-flow snapshot programs and their fixture file names.
    /// each snapshot test compiles its program and asserts the output matches
    /// the committed fixture; `generate_snapshot_fixtures` (ignored) rewrites
    /// the fixtures from the emitter. one source of truth for both.
    const SNAPSHOT_PROGRAMS: [(&str, &str); 4] = [
        (
            "arm64_let.s",
            // several lets, a shadowing `let x`, and ident uses.
            "fn let_demo(a: i64) -> i64 {\n\
            \x20   let x = a + 1\n\
            \x20   let y = x * 2\n\
            \x20   let x = y - a\n\
            \x20   x\n\
            }\n",
        ),
        (
            "arm64_if_else.s",
            // a bare `if`, an `if`/`else`, and a separate `else if` chain.
            // each function ends in a trailing fallback value so the body is
            // always-returning regardless of which branches the typechecker
            // proves terminating.
            "fn sign(n: i64) -> i64 {\n\
            \x20   if n == 0 { return 0 }\n\
            \x20   if n > 0 { return 1 } else { return -1 }\n\
            \x20   0\n\
            }\n\
            fn grade(s: i64) -> i64 {\n\
            \x20   if s >= 90 { return 4 }\n\
            \x20   else if s >= 80 { return 3 }\n\
            \x20   else { return 2 }\n\
            \x20   0\n\
            }\n",
        ),
        (
            "arm64_while.s",
            // a `while` loop with both a `break` and a `continue`.
            "fn loop_demo(c: bool) {\n\
            \x20   while c {\n\
            \x20       if c { break }\n\
            \x20       continue\n\
            \x20   }\n\
            }\n",
        ),
        (
            "arm64_for.s",
            // an exclusive `for i in 0..n` and an inclusive `for j in 0..=n`.
            "fn for_demo(n: i64) {\n\
            \x20   for i in 0..n { }\n\
            \x20   for j in 0..=n { }\n\
            }\n",
        ),
    ];

    /// read a snapshot fixture, normalising CRLF to LF for a Windows-stable
    /// compare. the emitter always produces LF; a checked-out file may carry
    /// CRLF.
    fn read_snapshot(name: &str) -> String {
        let path = format!("{}/tests/snapshots/{name}", env!("CARGO_MANIFEST_DIR"));
        std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {path}: {e}"))
            .replace("\r\n", "\n")
    }

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
        super::super::compile_arm64(&typed, src).unwrap_or_else(|e| panic!("arm64 errors: {e:?}"))
    }

    /// the source of snapshot program `name`, from [`SNAPSHOT_PROGRAMS`].
    fn snapshot_src(name: &str) -> &'static str {
        SNAPSHOT_PROGRAMS
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, src)| *src)
            .unwrap_or_else(|| panic!("no snapshot program named {name}"))
    }

    #[test]
    fn let_bindings_match_the_snapshot() {
        let emitted = compile_ok(snapshot_src("arm64_let.s"));
        assert_eq!(
            emitted,
            read_snapshot("arm64_let.s"),
            "arm64 let emission drifted from snapshot"
        );
    }

    #[test]
    fn if_else_matches_the_snapshot() {
        let emitted = compile_ok(snapshot_src("arm64_if_else.s"));
        assert_eq!(
            emitted,
            read_snapshot("arm64_if_else.s"),
            "arm64 if/else emission drifted from snapshot"
        );
    }

    #[test]
    fn while_loops_match_the_snapshot() {
        let emitted = compile_ok(snapshot_src("arm64_while.s"));
        assert_eq!(
            emitted,
            read_snapshot("arm64_while.s"),
            "arm64 while emission drifted from snapshot"
        );
    }

    #[test]
    fn for_loops_match_the_snapshot() {
        let emitted = compile_ok(snapshot_src("arm64_for.s"));
        assert_eq!(
            emitted,
            read_snapshot("arm64_for.s"),
            "arm64 for emission drifted from snapshot"
        );
    }

    #[test]
    fn a_let_stores_x0_into_its_stack_slot() {
        // `let x = 7` evaluates 7 into x0 then stores it to the binding slot.
        let out = compile_ok("fn f() -> i64 { let x = 7\nx }");
        assert!(out.contains("mov     x0, 7"), "{out}");
        assert!(
            out.contains("str     x0, [fp,"),
            "missing the let store: {out}"
        );
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
        assert_eq!(
            store_slot, load_slot,
            "let store and ident load slots differ"
        );
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
        assert_eq!(
            load_slot, second_slot,
            "ident must resolve to the shadowing let"
        );
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
        assert!(
            out.contains("cbz     x0, .Lif_else_"),
            "missing cbz to else: {out}"
        );
        assert!(out.contains(".Lif_else_"), "missing else label: {out}");
        assert!(out.contains(".Lif_end_"), "missing end label: {out}");
        // the then-block ends with a jump over the else arm.
        assert!(
            out.contains("b       .Lif_end_"),
            "missing jump over else: {out}"
        );
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
        assert_eq!(
            else_labels.len(),
            2,
            "an else-if chain has two else labels: {out}"
        );
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
        let out = compile_ok("fn f(c: bool) -> i64 { let a = 1\nif c { let b = 2\nreturn b }\na }");
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
        let stmt = TypedStmt::Break {
            span: Span::new(0, 1),
        };
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

    #[test]
    fn a_while_emits_the_test_at_bottom_shape() {
        // the test-at-bottom shape: `b while_test`, the body label, the test
        // label, a `cbnz` back to the body.
        let out = compile_ok("fn f(c: bool) { while c { } }");
        assert!(
            out.contains("b       .Lwhile_test_"),
            "missing jump to test: {out}"
        );
        assert!(
            out.lines()
                .any(|l| l.trim_start().starts_with(".Lwhile_body_")),
            "missing body label: {out}"
        );
        assert!(
            out.lines()
                .any(|l| l.trim_start().starts_with(".Lwhile_test_")),
            "missing test label: {out}"
        );
        assert!(
            out.contains("cbnz    x0, .Lwhile_body_"),
            "missing cbnz: {out}"
        );
        assert!(
            out.lines()
                .any(|l| l.trim_start().starts_with(".Lwhile_end_")),
            "missing end label: {out}"
        );
    }

    #[test]
    fn a_while_break_branches_to_the_end_label() {
        // a `break` inside a while branches to that loop's end label.
        let out = compile_ok("fn f(c: bool) { while c { break } }");
        // there is exactly one while; its end label is .Lwhile_end_N and the
        // break is `b .Lwhile_end_N`.
        let end_target = out
            .lines()
            .find_map(|l| {
                let t = l.trim();
                t.strip_prefix(".Lwhile_end_")
                    .map(|n| format!(".Lwhile_end_{}", n.trim_end_matches(':')))
            })
            .expect("no while end label");
        assert!(
            out.contains(&format!("b       {end_target}")),
            "break must branch to {end_target}: {out}"
        );
    }

    #[test]
    fn a_while_continue_branches_to_the_test_label() {
        // a `continue` inside a while branches to that loop's TEST label, so
        // the condition is re-checked.
        let out = compile_ok("fn f(c: bool) { while c { continue } }");
        let test_target = out
            .lines()
            .find_map(|l| {
                let t = l.trim();
                t.strip_prefix(".Lwhile_test_")
                    .map(|n| format!(".Lwhile_test_{}", n.trim_end_matches(':')))
            })
            .expect("no while test label");
        // the `continue` is one `b` to the test label; the loop entry is the
        // other. two branches to the test label total.
        let branches = out
            .lines()
            .filter(|l| l.contains(&format!("b       {test_target}")))
            .count();
        assert_eq!(
            branches, 2,
            "loop entry + continue both branch to the test: {out}"
        );
    }

    #[test]
    fn a_for_over_an_exclusive_range_emits_b_lt() {
        // `for i in 0..n` -- an exclusive range -- tests with `b.lt`.
        let out = compile_ok("fn f(n: i64) { for i in 0..n { } }");
        assert!(
            out.contains("b       .Lfor_test_"),
            "missing jump to test: {out}"
        );
        assert!(
            out.lines()
                .any(|l| l.trim_start().starts_with(".Lfor_incr_")),
            "missing increment label: {out}"
        );
        assert!(
            out.contains("add     x0, x0, 1"),
            "missing the increment: {out}"
        );
        assert!(out.contains("cmp     x0, x9"), "missing the compare: {out}");
        assert!(
            out.contains("b.lt    .Lfor_body_"),
            "exclusive range -> b.lt: {out}"
        );
        assert!(
            !out.contains("b.le"),
            "an exclusive range must not use b.le: {out}"
        );
    }

    #[test]
    fn a_for_over_an_inclusive_range_emits_b_le() {
        // `for i in 0..=n` -- an inclusive range -- tests with `b.le`.
        let out = compile_ok("fn f(n: i64) { for i in 0..=n { } }");
        assert!(
            out.contains("b.le    .Lfor_body_"),
            "inclusive range -> b.le: {out}"
        );
        assert!(
            !out.contains("b.lt"),
            "an inclusive range must not use b.lt: {out}"
        );
    }

    #[test]
    fn a_for_continue_branches_to_the_increment_label() {
        // a `continue` inside a for branches to the INCREMENT label, so the
        // loop variable still advances before the next test.
        let out = compile_ok("fn f(n: i64) { for i in 0..n { continue } }");
        let incr_target = out
            .lines()
            .find_map(|l| {
                let t = l.trim();
                t.strip_prefix(".Lfor_incr_")
                    .map(|n| format!(".Lfor_incr_{}", n.trim_end_matches(':')))
            })
            .expect("no for increment label");
        assert!(
            out.contains(&format!("b       {incr_target}")),
            "continue must branch to {incr_target}: {out}"
        );
    }

    #[test]
    fn a_for_evaluates_its_end_bound_once_into_a_slot() {
        // the `end` bound is stored once, before the loop, with a `// for end`
        // comment -- not re-evaluated each iteration.
        let out = compile_ok("fn f(n: i64) { for i in 0..n { } }");
        let end_stores = out
            .lines()
            .filter(|l| l.contains("str     x0, [fp,") && l.contains("// for end"))
            .count();
        assert_eq!(
            end_stores, 1,
            "the for end bound is stored exactly once: {out}"
        );
    }

    #[test]
    fn a_for_over_a_non_range_iterable_is_rejected_cleanly() {
        // `for x in arr` -- an array iterable -- is outside the integer core.
        // a clean QalaError, never a panic.
        let typed = typecheck("fn f() { let arr = [1, 2, 3]\nfor x in arr { } }");
        let err =
            super::super::compile_arm64(&typed, "").expect_err("a non-range for must be rejected");
        match &err[0] {
            QalaError::Type { message, .. } => {
                assert!(
                    message.contains("non-range") || message.contains("array"),
                    "message: {message}"
                );
            }
            other => panic!("expected QalaError::Type, got {other:?}"),
        }
    }

    #[test]
    fn a_for_range_with_a_missing_bound_is_rejected_cleanly() {
        // a range with no end bound reaches the For arm as a None -- the
        // backend rejects it with a clean error, never a panic. build the
        // typed AST directly: an unbounded range is not valid `for` surface
        // syntax, so it cannot come from a parsed source.
        use crate::types::QalaType;
        let stmt = TypedStmt::For {
            var: "i".to_string(),
            var_ty: QalaType::I64,
            iter: TypedExpr::Range {
                start: Some(Box::new(TypedExpr::Int {
                    value: 0,
                    ty: QalaType::I64,
                    span: Span::new(0, 1),
                })),
                end: None,
                inclusive: false,
                ty: QalaType::Array(Box::new(QalaType::I64), None),
                span: Span::new(0, 4),
            },
            body: TypedBlock {
                stmts: vec![],
                value: None,
                ty: QalaType::Void,
                span: Span::new(5, 6),
            },
            span: Span::new(0, 7),
        };
        let mut backend = Arm64Backend::new("");
        // a frame so binding_slot resolves -- a one-`for` frame has two slots.
        let err = backend
            .compile_stmt(&stmt)
            .expect_err("a for with a missing bound must be rejected");
        match err {
            QalaError::Type { message, .. } => {
                assert!(message.contains("bound"), "message: {message}");
            }
            other => panic!("expected QalaError::Type, got {other:?}"),
        }
    }
}
