//! integer-expression codegen for the ARM64 backend.
//!
//! [`Arm64Backend::compile_expr`] lowers one [`TypedExpr`] to AArch64
//! instructions. it is split into this file as a second `impl Arm64Backend`
//! block -- Rust allows an `impl` to span several files of the same module --
//! while the struct itself and the program/function walk live in `mod.rs`.
//!
//! ## the result convention
//!
//! every `compile_expr` call leaves its result in `x0`. a binary operator
//! evaluates the LHS into `x0`, spills `x0` to a scratch stack slot, evaluates
//! the RHS into `x0`, reloads the LHS into `x9`, and applies the operator
//! writing `x0`. scratch registers `x9`-`x15` hold only momentary values
//! within one operation -- never across a statement boundary, never across a
//! `bl`. nested binary operators each claim a distinct scratch slot from the
//! [`FrameLayout`](super::frame::FrameLayout) and release it after, so the
//! spill bookkeeping is a balanced stack.
//!
//! the integer core handles `i64` and `bool`: integer literals, boolean
//! literals, identifier loads, parenthesised expressions, the unary `!` and
//! `-`, the binary `+ - * / %`, the six comparisons, and the short-circuit
//! `&&` / `||`. every other expression construct -- a float, a string, a
//! struct literal, a method call, a call, a range -- is unsupported and
//! returns a [`QalaError`], never a panic.

use crate::ast::{BinOp, UnaryOp};
use crate::errors::QalaError;
use crate::typed_ast::TypedExpr;

use super::Arm64Backend;

/// the inclusive upper bound for emitting an `i64` literal as a bare `mov`.
///
/// AArch64 `mov` takes a 16-bit immediate; a value in `0..=65535` always
/// encodes. anything outside that range -- a larger constant or a negative
/// value -- is emitted with the `ldr xN, =<value>` literal-pool form, which
/// the assembler resolves for any 64-bit value.
const MOV_IMM_MAX: i64 = 65535;

impl Arm64Backend {
    /// lower one typed expression to AArch64 instructions, leaving the result
    /// in `x0`.
    ///
    /// returns `Ok(())` once the instructions are emitted, or a
    /// [`QalaError::Type`] for an unsupported construct (a float, a string, a
    /// call, a range, ...). the error carries the offending node's span so the
    /// diagnostic points at the exact construct.
    pub(super) fn compile_expr(&mut self, expr: &TypedExpr) -> Result<(), QalaError> {
        match expr {
            TypedExpr::Int { value, .. } => {
                self.emit_int_literal(*value);
                Ok(())
            }
            TypedExpr::Bool { value, .. } => {
                // a bool is 1 (true) or 0 (false) in x0.
                self.asm.emit_insn(if *value { "mov     x0, 1" } else { "mov     x0, 0" });
                Ok(())
            }
            TypedExpr::Ident { name, span, .. } => {
                // a `let`/`for` binding (resolved through the scope stack,
                // newest-first so a shadowing binding wins) or a function
                // parameter -- load its slot. an unresolved name is a backend
                // bug, surfaced as an error rather than a panic (the
                // typechecker already proved the name resolves, so this path
                // is defensive).
                let slot = self.resolve_name(name).ok_or_else(|| QalaError::Type {
                    span: *span,
                    message: format!("arm64 backend: name `{name}` has no stack slot"),
                })?;
                self.asm
                    .emit_insn_commented(&format!("ldr     x0, [fp, {slot}]"), name);
                Ok(())
            }
            TypedExpr::Paren { inner, .. } => {
                // parentheses are semantically transparent.
                self.compile_expr(inner)
            }
            TypedExpr::Unary { op, operand, .. } => self.compile_unary(op, operand),
            TypedExpr::Binary { op, lhs, rhs, .. } => self.compile_binary(op, lhs, rhs),
            // Block recurses into the block via the minimal compile_block held
            // in mod.rs for this plan (plan 12-02 replaces it with the full
            // scope-aware version in stmt.rs).
            TypedExpr::Block { block, .. } => self.compile_block(block),
            // Call and Range are out of this plan's scope: Call arrives in plan
            // 12-03, Range in plan 12-02 (inside a `for`). Until then both are
            // a clean unsupported-construct rejection.
            TypedExpr::Call { span, .. } => Err(QalaError::Type {
                span: *span,
                message: "the arm64 backend does not yet support calls".to_string(),
            }),
            TypedExpr::Range { span, .. } => Err(QalaError::Type {
                span: *span,
                message: "the arm64 backend does not yet support ranges".to_string(),
            }),
            // every remaining construct is beyond the integer core.
            _ => Err(QalaError::Type {
                span: expr.span(),
                message: format!(
                    "the arm64 backend does not yet support {}",
                    unsupported_expr_name(expr)
                ),
            }),
        }
    }

    /// emit an `i64` literal into `x0`.
    ///
    /// a value in `0..=65535` encodes directly as `mov x0, <value>`; anything
    /// else (a large constant, a negative value) uses the `ldr x0, =<value>`
    /// literal-pool form, which the assembler resolves for any 64-bit value.
    fn emit_int_literal(&mut self, value: i64) {
        if (0..=MOV_IMM_MAX).contains(&value) {
            self.asm.emit_insn(&format!("mov     x0, {value}"));
        } else {
            self.asm.emit_insn(&format!("ldr     x0, ={value}"));
        }
    }

    /// emit a unary operator: `!` flips a boolean, `-` negates an integer.
    fn compile_unary(&mut self, op: &UnaryOp, operand: &TypedExpr) -> Result<(), QalaError> {
        // the operand result lands in x0.
        self.compile_expr(operand)?;
        match op {
            // operand is a 0/1 bool; eor with 1 flips bit 0.
            UnaryOp::Not => self.asm.emit_insn("eor     x0, x0, 1"),
            // operand is an i64; two's-complement negate.
            UnaryOp::Neg => self.asm.emit_insn("neg     x0, x0"),
        }
        Ok(())
    }

    /// emit a binary operator. arithmetic and comparison go through the
    /// scratch-slot spill discipline; `&&` / `||` go through their
    /// short-circuit label sequences.
    fn compile_binary(
        &mut self,
        op: &BinOp,
        lhs: &TypedExpr,
        rhs: &TypedExpr,
    ) -> Result<(), QalaError> {
        match op {
            BinOp::And => self.compile_short_circuit(lhs, rhs, ShortCircuit::And),
            BinOp::Or => self.compile_short_circuit(lhs, rhs, ShortCircuit::Or),
            // arithmetic and comparison: evaluate both operands through the
            // spill discipline, then apply the operator.
            _ => self.compile_spilled_binary(op, lhs, rhs),
        }
    }

    /// evaluate `lhs` and `rhs` through the scratch-slot spill discipline --
    /// leaving the LHS in `x9` and the RHS in `x0` -- then emit the operator's
    /// instruction(s).
    ///
    /// the spill discipline: claim a scratch slot, evaluate the LHS into `x0`,
    /// `str` it to the slot, evaluate the RHS into `x0`, `ldr` the LHS back
    /// into `x9`, release the slot. claims stack -- a nested binary RHS claims
    /// its own distinct slot -- so arbitrarily deep expressions never run out.
    fn compile_spilled_binary(
        &mut self,
        op: &BinOp,
        lhs: &TypedExpr,
        rhs: &TypedExpr,
    ) -> Result<(), QalaError> {
        let scratch = self.frame_mut().claim_scratch();
        // LHS -> x0 -> spill to the scratch slot.
        self.compile_expr(lhs)?;
        self.asm
            .emit_insn_commented(&format!("str     x0, [fp, {scratch}]"), "spill lhs");
        // RHS -> x0.
        self.compile_expr(rhs)?;
        // reload the LHS into x9; the slot is now free.
        self.asm
            .emit_insn_commented(&format!("ldr     x9, [fp, {scratch}]"), "reload lhs");
        self.frame_mut().release_scratch();
        // apply the operator: x9 = lhs, x0 = rhs, result -> x0.
        self.emit_binop(op);
        Ok(())
    }

    /// emit the instruction(s) for an arithmetic or comparison operator, with
    /// the LHS already in `x9` and the RHS in `x0`. the result lands in `x0`.
    ///
    /// `&&` and `||` never reach here -- they are short-circuit and handled
    /// separately. the fallthrough arm is unreachable in practice but returns
    /// without emitting rather than panicking, keeping the WASM build crash-free.
    fn emit_binop(&mut self, op: &BinOp) {
        match op {
            BinOp::Add => self.asm.emit_insn("add     x0, x9, x0"),
            BinOp::Sub => self.asm.emit_insn("sub     x0, x9, x0"),
            BinOp::Mul => self.asm.emit_insn("mul     x0, x9, x0"),
            // i64 is signed -> sdiv, not udiv.
            BinOp::Div => self.asm.emit_insn("sdiv    x0, x9, x0"),
            // no modulo instruction: a % b = a - (a/b)*b via msub. the quotient
            // goes in x10 so it does not clobber x9 (the lhs) before msub reads
            // it. `msub xd, xn, xm, xa` computes `xa - xn*xm`.
            BinOp::Rem => {
                self.asm.emit_insn("sdiv    x10, x9, x0");
                self.asm.emit_insn("msub    x0, x10, x0, x9");
            }
            // comparisons: cmp then cset with the SIGNED condition code
            // (lt/le/gt/ge -- not the unsigned lo/ls/hi/hs), since i64 is signed.
            BinOp::Eq => {
                self.asm.emit_insn("cmp     x9, x0");
                self.asm.emit_insn("cset    x0, eq");
            }
            BinOp::Ne => {
                self.asm.emit_insn("cmp     x9, x0");
                self.asm.emit_insn("cset    x0, ne");
            }
            BinOp::Lt => {
                self.asm.emit_insn("cmp     x9, x0");
                self.asm.emit_insn("cset    x0, lt");
            }
            BinOp::Le => {
                self.asm.emit_insn("cmp     x9, x0");
                self.asm.emit_insn("cset    x0, le");
            }
            BinOp::Gt => {
                self.asm.emit_insn("cmp     x9, x0");
                self.asm.emit_insn("cset    x0, gt");
            }
            BinOp::Ge => {
                self.asm.emit_insn("cmp     x9, x0");
                self.asm.emit_insn("cset    x0, ge");
            }
            // && / || are short-circuit and never routed here.
            BinOp::And | BinOp::Or => {}
        }
    }

    /// emit a short-circuit `&&` or `||`.
    ///
    /// `&&`: if the LHS is false the result is false and the RHS is skipped;
    /// `||`: if the LHS is true the result is true and the RHS is skipped. the
    /// RHS instructions sit *after* the first conditional branch, so they run
    /// only when the LHS does not settle the result. both produce a 0/1 in `x0`.
    fn compile_short_circuit(
        &mut self,
        lhs: &TypedExpr,
        rhs: &TypedExpr,
        kind: ShortCircuit,
    ) -> Result<(), QalaError> {
        // a fresh label pair, unique across the whole emitted file.
        let settle = self.labels.fresh(kind.settle_prefix());
        let done = self.labels.fresh(kind.done_prefix());
        // the branch that short-circuits: cbz for &&, cbnz for ||.
        let branch = kind.branch_insn();
        // LHS -> x0; short-circuit if it settles the result.
        self.compile_expr(lhs)?;
        self.asm.emit_insn(&format!("{branch}    x0, {settle}"));
        // RHS -> x0; short-circuit on the same condition.
        self.compile_expr(rhs)?;
        self.asm.emit_insn(&format!("{branch}    x0, {settle}"));
        // neither operand short-circuited: the result is the non-settling value.
        self.asm
            .emit_insn(&format!("mov     x0, {}", kind.fallthrough_value()));
        self.asm.emit_insn(&format!("b       {done}"));
        // the settle label: the result is the short-circuit value.
        self.asm.emit_label(&settle);
        self.asm
            .emit_insn(&format!("mov     x0, {}", kind.settle_value()));
        self.asm.emit_label(&done);
        Ok(())
    }
}

/// which short-circuit operator is being emitted -- `&&` or `||`.
///
/// the two share one code path; this enum selects the branch instruction, the
/// label prefixes, and the two result values that differ between them.
#[derive(Clone, Copy)]
enum ShortCircuit {
    /// the `&&` operator: short-circuits to false on a false operand.
    And,
    /// the `||` operator: short-circuits to true on a true operand.
    Or,
}

impl ShortCircuit {
    /// the branch instruction that detects a short-circuit: `cbz` for `&&`
    /// (branch on a false/zero operand), `cbnz` for `||` (branch on a true one).
    fn branch_insn(self) -> &'static str {
        match self {
            ShortCircuit::And => "cbz ",
            ShortCircuit::Or => "cbnz",
        }
    }

    /// the label prefix for the short-circuit-settle target.
    fn settle_prefix(self) -> &'static str {
        match self {
            ShortCircuit::And => "and_false",
            ShortCircuit::Or => "or_true",
        }
    }

    /// the label prefix for the done/join target.
    fn done_prefix(self) -> &'static str {
        match self {
            ShortCircuit::And => "and_done",
            ShortCircuit::Or => "or_done",
        }
    }

    /// the result value when an operand short-circuits: `0` for `&&` (false),
    /// `1` for `||` (true).
    fn settle_value(self) -> u8 {
        match self {
            ShortCircuit::And => 0,
            ShortCircuit::Or => 1,
        }
    }

    /// the result value when neither operand short-circuits: `1` for `&&` (both
    /// true), `0` for `||` (both false).
    fn fallthrough_value(self) -> u8 {
        match self {
            ShortCircuit::And => 1,
            ShortCircuit::Or => 0,
        }
    }
}

/// the human name of an unsupported expression construct, for the rejection
/// diagnostic. plan 13 (ARM-05) polishes this wording; plan 12 keeps it plain.
fn unsupported_expr_name(expr: &TypedExpr) -> &'static str {
    match expr {
        TypedExpr::Float { .. } => "floats",
        TypedExpr::Byte { .. } => "byte values",
        TypedExpr::Str { .. } => "strings",
        TypedExpr::Tuple { .. } => "tuples",
        TypedExpr::ArrayLit { .. } | TypedExpr::ArrayRepeat { .. } => "arrays",
        TypedExpr::StructLit { .. } => "struct literals",
        TypedExpr::FieldAccess { .. } => "field access",
        TypedExpr::MethodCall { .. } => "method calls",
        TypedExpr::Index { .. } => "indexing",
        TypedExpr::Try { .. } => "the `?` operator",
        TypedExpr::Pipeline { .. } => "the pipeline operator",
        TypedExpr::Comptime { .. } => "comptime blocks",
        TypedExpr::Match { .. } => "match expressions",
        TypedExpr::OrElse { .. } => "the `or` fallback",
        TypedExpr::Interpolation { .. } => "string interpolation",
        // the integer-core variants and Call/Range are handled before this
        // helper is reached.
        _ => "this construct",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;
    use crate::parser::Parser;
    use crate::typechecker::check_program;
    use crate::typed_ast::TypedItem;

    /// lex, parse, and typecheck `src`, then return the trailing-value
    /// expression of the named function. test sources are written as a single
    /// function whose body is one trailing expression.
    fn trailing_expr(src: &str, fn_name: &str) -> TypedExpr {
        let tokens = Lexer::tokenize(src).expect("lex failed");
        let ast = Parser::parse(&tokens).expect("parse failed");
        let (typed, terrors, _) = check_program(&ast, src);
        assert!(terrors.is_empty(), "typecheck errors: {terrors:?}");
        let decl = typed
            .iter()
            .find_map(|item| match item {
                TypedItem::Fn(d) if d.name == fn_name => Some(d.clone()),
                _ => None,
            })
            .unwrap_or_else(|| panic!("function `{fn_name}` not found"));
        *decl
            .body
            .value
            .expect("the test function has no trailing value")
    }

    /// compile `src`'s named function's trailing expression in isolation and
    /// return the emitted instruction text. the backend is given a frame
    /// planned from that function so scratch slots and parameter slots resolve.
    fn emit_expr(src: &str, fn_name: &str) -> String {
        let tokens = Lexer::tokenize(src).expect("lex failed");
        let ast = Parser::parse(&tokens).expect("parse failed");
        let (typed, terrors, _) = check_program(&ast, src);
        assert!(terrors.is_empty(), "typecheck errors: {terrors:?}");
        let decl = typed
            .iter()
            .find_map(|item| match item {
                TypedItem::Fn(d) if d.name == fn_name => Some(d.clone()),
                _ => None,
            })
            .unwrap_or_else(|| panic!("function `{fn_name}` not found"));
        let mut backend = Arm64Backend::new(src);
        backend.begin_function(&decl);
        let expr = decl.body.value.clone().expect("no trailing value");
        backend.compile_expr(&expr).expect("compile_expr failed");
        backend.take_text()
    }

    #[test]
    fn an_integer_literal_emits_a_mov() {
        let asm = emit_expr("fn f() -> i64 { 42 }", "f");
        assert!(asm.contains("mov     x0, 42"), "{asm}");
    }

    #[test]
    fn a_large_integer_literal_uses_the_literal_pool() {
        // a value past the 16-bit mov range falls back to ldr x0, =<value>.
        let asm = emit_expr("fn f() -> i64 { 100000 }", "f");
        assert!(asm.contains("ldr     x0, =100000"), "{asm}");
        assert!(!asm.contains("mov     x0, 100000"), "{asm}");
    }

    #[test]
    fn a_bool_literal_emits_one_or_zero() {
        assert!(emit_expr("fn f() -> bool { true }", "f").contains("mov     x0, 1"));
        assert!(emit_expr("fn f() -> bool { false }", "f").contains("mov     x0, 0"));
    }

    #[test]
    fn an_ident_loads_its_parameter_slot() {
        // a one-parameter function: the parameter lives at [fp, 16], just
        // above the saved fp/lr pair.
        let asm = emit_expr("fn f(a: i64) -> i64 { a }", "f");
        assert!(asm.contains("ldr     x0, [fp, 16]"), "{asm}");
    }

    #[test]
    fn paren_is_transparent() {
        // `(7)` emits exactly what `7` does -- no extra instruction.
        let asm = emit_expr("fn f() -> i64 { (7) }", "f");
        assert!(asm.contains("mov     x0, 7"), "{asm}");
    }

    #[test]
    fn addition_emits_the_spill_discipline_and_add() {
        let asm = emit_expr("fn f() -> i64 { 1 + 2 }", "f");
        assert!(asm.contains("str     x0, [fp, "), "missing spill: {asm}");
        assert!(asm.contains("ldr     x9, [fp, "), "missing reload: {asm}");
        assert!(asm.contains("add     x0, x9, x0"), "missing add: {asm}");
    }

    #[test]
    fn subtraction_and_multiplication_emit_sub_and_mul() {
        assert!(emit_expr("fn f() -> i64 { 5 - 3 }", "f").contains("sub     x0, x9, x0"));
        assert!(emit_expr("fn f() -> i64 { 5 * 3 }", "f").contains("mul     x0, x9, x0"));
    }

    #[test]
    fn division_emits_signed_sdiv() {
        let asm = emit_expr("fn f() -> i64 { 9 / 3 }", "f");
        assert!(asm.contains("sdiv    x0, x9, x0"), "{asm}");
        assert!(!asm.contains("udiv"), "i64 division must be signed: {asm}");
    }

    #[test]
    fn modulo_emits_the_sdiv_msub_idiom() {
        let asm = emit_expr("fn f() -> i64 { 9 % 4 }", "f");
        assert!(asm.contains("sdiv    x10, x9, x0"), "missing quotient: {asm}");
        assert!(asm.contains("msub    x0, x10, x0, x9"), "missing msub: {asm}");
    }

    #[test]
    fn each_comparison_emits_cmp_and_the_signed_condition() {
        for (src, cond) in [
            ("fn f() -> bool { 1 == 2 }", "cset    x0, eq"),
            ("fn f() -> bool { 1 != 2 }", "cset    x0, ne"),
            ("fn f() -> bool { 1 < 2 }", "cset    x0, lt"),
            ("fn f() -> bool { 1 <= 2 }", "cset    x0, le"),
            ("fn f() -> bool { 1 > 2 }", "cset    x0, gt"),
            ("fn f() -> bool { 1 >= 2 }", "cset    x0, ge"),
        ] {
            let asm = emit_expr(src, "f");
            assert!(asm.contains("cmp     x9, x0"), "missing cmp for {src}: {asm}");
            assert!(asm.contains(cond), "missing `{cond}` for {src}: {asm}");
        }
    }

    #[test]
    fn comparisons_use_signed_not_unsigned_conditions() {
        // i64 is signed; the unsigned condition codes must never appear.
        let asm = emit_expr("fn f() -> bool { 1 < 2 }", "f");
        for unsigned in ["lo", "ls", "hi", "hs"] {
            assert!(!asm.contains(&format!("cset    x0, {unsigned}")), "{asm}");
        }
    }

    #[test]
    fn logical_and_emits_the_short_circuit_labels() {
        let asm = emit_expr("fn f() -> bool { true && false }", "f");
        assert!(asm.contains(".Land_false_"), "missing settle label: {asm}");
        assert!(asm.contains(".Land_done_"), "missing done label: {asm}");
        assert!(asm.contains("cbz "), "&& must short-circuit with cbz: {asm}");
    }

    #[test]
    fn logical_or_emits_the_short_circuit_labels() {
        let asm = emit_expr("fn f() -> bool { true || false }", "f");
        assert!(asm.contains(".Lor_true_"), "missing settle label: {asm}");
        assert!(asm.contains(".Lor_done_"), "missing done label: {asm}");
        assert!(asm.contains("cbnz"), "|| must short-circuit with cbnz: {asm}");
    }

    #[test]
    fn not_emits_an_eor() {
        let asm = emit_expr("fn f() -> bool { !true }", "f");
        assert!(asm.contains("eor     x0, x0, 1"), "{asm}");
    }

    #[test]
    fn neg_emits_a_neg() {
        let asm = emit_expr("fn f() -> i64 { -5 }", "f");
        assert!(asm.contains("neg     x0, x0"), "{asm}");
    }

    #[test]
    fn a_nested_expression_claims_distinct_scratch_slots() {
        // `(1 + 2) * (3 + 4)`: the outer `*` spills its LHS while the RHS
        // subtree (itself a `+`) runs and spills again -- the two live spills
        // must land in distinct slots, so two different [fp, N] offsets appear.
        let asm = emit_expr("fn f() -> i64 { (1 + 2) * (3 + 4) }", "f");
        let mut spill_slots: Vec<&str> = asm
            .lines()
            .filter(|l| l.contains("str     x0, [fp, ") && l.contains("spill lhs"))
            .collect();
        spill_slots.sort();
        spill_slots.dedup();
        assert!(
            spill_slots.len() >= 2,
            "nested ops must use >= 2 distinct scratch slots: {asm}"
        );
        assert!(asm.contains("mul     x0, x9, x0"), "{asm}");
    }

    #[test]
    fn an_unsupported_construct_returns_an_error_not_a_panic() {
        // a float literal is outside the integer core -- compile_expr must
        // return a QalaError carrying the float's span, never panic.
        let src = "fn f() -> f64 { 3.5 }";
        let expr = trailing_expr(src, "f");
        let mut backend = Arm64Backend::new(src);
        let err = backend
            .compile_expr(&expr)
            .expect_err("a float must be rejected");
        match err {
            QalaError::Type { message, .. } => {
                assert!(message.contains("float"), "message: {message}");
            }
            other => panic!("expected QalaError::Type, got {other:?}"),
        }
    }
}
