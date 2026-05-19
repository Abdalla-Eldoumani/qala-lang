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
//! `-`, the binary `+ - * / %`, the six comparisons, the short-circuit
//! `&&` / `||`, and a call to a user function. every other expression
//! construct -- a float, a string, a struct literal, a method call, a range,
//! a call to a stdlib function -- is unsupported and returns a [`QalaError`],
//! never a panic.
//!
//! ## a call expression
//!
//! a [`TypedExpr::Call`] to a user function lowers to AAPCS64 argument passing
//! and a `bl`. because every `compile_expr` lands its result in `x0`,
//! evaluating argument 1 into `x0` would clobber argument 0 -- so each
//! argument is evaluated, then a fresh scratch slot is CLAIMED (the same
//! [`claim_scratch`](super::frame::FrameLayout::claim_scratch) stack the
//! nested-binary spill discipline uses) and the result `str`-ed into it.
//! after all arguments are evaluated a run of `ldr x0`, `ldr x1`, ... loads
//! the slots into the AAPCS64 argument registers; then `bl <name>`, the
//! result in `x0`; then every claimed slot is released.
//!
//! claiming -- rather than using one fixed shared run -- is what makes a
//! NESTED call correct. an argument that is itself a call runs to completion
//! while the outer call's already-evaluated arguments sit in claimed slots;
//! the nested call claims its own argument slots STRICTLY BEYOND those, so it
//! cannot overwrite the outer call's saved arguments. the argument is
//! evaluated BEFORE its slot is claimed, so the argument's own transient
//! scratch claims (a binary operator inside it) release before the
//! persistent argument slot is taken.

use crate::ast::{BinOp, UnaryOp};
use crate::errors::QalaError;
use crate::span::Span;
use crate::typed_ast::TypedExpr;

use super::Arm64Backend;

/// the AAPCS64 integer-argument register count: arguments 0-7 go in `x0`-`x7`,
/// a ninth argument would spill onto the stack. the integer core supports at
/// most this many arguments per call; a stack-passed argument is a Phase 13+
/// extension.
const MAX_CALL_ARGS: usize = 8;

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
            // a call to a user function: AAPCS64 argument passing and a bl.
            TypedExpr::Call { callee, args, span, .. } => {
                self.compile_call(callee, args, *span)
            }
            // a Range only appears as a `for` iterable -- `stmt.rs`'s For arm
            // matches it directly. a Range reaching `compile_expr` is a Range
            // used as a value, which the integer core does not support.
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
    /// literal-pool form, which the assembler resolves for ANY 64-bit value --
    /// `i64::MIN` and `i64::MAX` included.
    ///
    /// `value` is always a valid `i64`: an integer literal whose magnitude
    /// exceeds `i64::MAX` is rejected by the lexer as an overflow before this
    /// backend ever runs, and a leading `-` is a separate unary `Neg` over a
    /// non-negative magnitude (handled by `compile_unary`, not folded into the
    /// literal here). so there is no unrepresentable literal to guard against.
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

    /// emit a call to a user function: AAPCS64 argument passing and a `bl`,
    /// the result left in `x0`.
    ///
    /// rejects, with a clean [`QalaError`], a call the integer core does not
    /// support: a computed callee (one that is not a plain identifier), a
    /// callee that resolves to a stdlib function rather than a user-declared
    /// one (the stdlib and `printf` output arrive in Phase 13), and a call
    /// with more than eight arguments (a stack-passed argument is a Phase 13+
    /// extension -- a well-typed program cannot reach this since the callee
    /// has at most eight parameters, but the guard is defensive so a `bl` is
    /// never emitted with a wrong argument count).
    fn compile_call(
        &mut self,
        callee: &TypedExpr,
        args: &[TypedExpr],
        span: Span,
    ) -> Result<(), QalaError> {
        // the callee must be a plain identifier -- a computed callee (a call
        // returning a function, a field holding one) is out of the integer
        // core.
        let name = match callee {
            TypedExpr::Ident { name, .. } => name,
            _ => {
                return Err(QalaError::Type {
                    span,
                    message: "the arm64 backend does not yet support computed callees"
                        .to_string(),
                });
            }
        };
        // the name must be a user-declared function. a name not in `fn_names`
        // resolves to a stdlib function (`print`, `println`, `sqrt`, `len`,
        // ...) -- the stdlib is Phase 13, so reject it cleanly here.
        if !self.fn_names.contains(name) {
            return Err(QalaError::Type {
                span,
                message: format!(
                    "the arm64 backend does not yet support the `{name}` function"
                ),
            });
        }
        // at most eight arguments -- AAPCS64 passes the rest on the stack,
        // which the integer core does not do. defensive: a well-typed call
        // cannot exceed the callee's parameter count, which `compile_fn`
        // already capped at eight.
        if args.len() > MAX_CALL_ARGS {
            return Err(QalaError::Type {
                span,
                message: "the arm64 backend supports at most 8 arguments".to_string(),
            });
        }

        // evaluate every argument into x0 and spill it to a freshly CLAIMED
        // scratch slot, BEFORE loading any argument register. evaluating
        // argument i+1 into x0 would clobber argument i, so the load into
        // x0..x{n-1} happens only after every argument is safely in a slot.
        //
        // each argument is evaluated FIRST, then its slot is claimed: the
        // argument's own transient scratch claims (a binary operator, or a
        // nested call's argument slots) are all released by the time
        // `compile_expr` returns, so the slot claimed here sits beyond them
        // and beyond every earlier argument's slot. a nested call therefore
        // cannot alias the outer call's saved arguments -- the bug a single
        // shared argument run would have. the claimed slots are released only
        // after the `bl`, so they stay reserved across the whole call.
        let mut arg_offsets = Vec::with_capacity(args.len());
        for (i, arg) in args.iter().enumerate() {
            // arg -> x0; a nested call completes entirely here, claiming and
            // releasing its own argument slots within this evaluation.
            self.compile_expr(arg)?;
            // now claim this argument's persistent slot and spill x0 into it.
            let slot = self.frame_mut().claim_scratch();
            self.asm.emit_insn_commented(
                &format!("str     x0, [fp, {slot}]"),
                &format!("arg {i}"),
            );
            arg_offsets.push(slot);
        }
        // every argument is now in a slot -- load them into x0..x{n-1}.
        for (i, slot) in arg_offsets.iter().enumerate() {
            self.asm.emit_insn(&format!("ldr     x{i}, [fp, {slot}]"));
        }
        // the call. the result, by AAPCS64, is in x0 -- which is the result
        // convention compile_expr upholds.
        self.asm.emit_insn(&format!("bl      {name}"));
        // release the argument slots -- one per argument, balancing the claims
        // above so the scratch stack returns to where the call found it.
        for _ in &arg_offsets {
            self.frame_mut().release_scratch();
        }
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
    fn the_i64_min_magnitude_is_rejected_at_the_lexer_before_the_backend() {
        // `emit_int_literal` documents that `value` is always a valid `i64`.
        // the proof of that invariant: the bare literal `9223372036854775808`
        // -- the magnitude of `i64::MIN`, one past `i64::MAX` -- never reaches
        // the backend at all. it is an overflow the LEXER rejects, so a typed
        // AST carrying it cannot be built and the backend's `ldr =` form never
        // has to encode an out-of-range value.
        let over = i64::MAX as u64 + 1; // 9223372036854775808
        let src = format!("fn f() -> i64 {{ -{over} }}");
        let lexed = Lexer::tokenize(&src);
        assert!(
            lexed.is_err(),
            "the i64::MIN magnitude must be rejected by the lexer, not lowered"
        );
    }

    #[test]
    fn an_extreme_negative_literal_round_trips_through_the_backend() {
        // the largest-magnitude literal that IS a valid i64: i64::MAX. negated,
        // it is `i64::MIN + 1`. the source is unary `Neg` over that literal --
        // the literal itself goes through `emit_int_literal`'s `ldr =` path
        // (well past the 16-bit `mov` range) and `neg x0, x0` applies the sign.
        // the backend emits clean assembly, no panic, no overflow.
        let src = format!("fn f() -> i64 {{ -{} }}", i64::MAX);
        let asm = emit_expr(&src, "f");
        assert!(
            asm.contains(&format!("ldr     x0, ={}", i64::MAX)),
            "the i64::MAX magnitude must use the literal pool: {asm}"
        );
        assert!(asm.contains("neg     x0, x0"), "the unary minus must negate: {asm}");
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

    /// compile a whole multi-function program to assembly, panicking on a
    /// backend error. unlike `emit_expr`, this runs the full `compile_arm64`
    /// path, so the `fn_names` set is populated and a call resolves.
    fn compile_program_ok(src: &str) -> String {
        let tokens = Lexer::tokenize(src).expect("lex failed");
        let ast = Parser::parse(&tokens).expect("parse failed");
        let (typed, terrors, _) = check_program(&ast, src);
        assert!(terrors.is_empty(), "typecheck errors: {terrors:?}");
        super::super::compile_arm64(&typed, src)
            .unwrap_or_else(|e| panic!("arm64 errors: {e:?}"))
    }

    /// the instruction lines of `caller`'s emitted function body -- the lines
    /// between the `caller:` label and the `.Lcaller_epilogue:` label.
    fn caller_body(asm: &str) -> Vec<String> {
        asm.lines()
            .skip_while(|l| !l.starts_with("caller:"))
            .take_while(|l| !l.trim_start().starts_with(".Lcaller_epilogue"))
            .map(|l| l.to_string())
            .collect()
    }

    #[test]
    fn a_call_to_a_user_function_emits_argument_spills_loads_and_a_bl() {
        // `add3(1, 2, 3)`: three args spilled to scratch slots, then loaded
        // into x0/x1/x2, then `bl add3`.
        let asm = compile_program_ok(
            "fn add3(a: i64, b: i64, c: i64) -> i64 { a + b + c }\n\
             fn caller() -> i64 { add3(1, 2, 3) }",
        );
        let body = caller_body(&asm).join("\n");
        // three argument spills, one per argument.
        let spills = body.lines().filter(|l| l.contains("// arg ")).count();
        assert_eq!(spills, 3, "three arguments -> three spills: {body}");
        // the loads into the AAPCS64 argument registers.
        assert!(body.contains("ldr     x0, [fp, "), "missing x0 load: {body}");
        assert!(body.contains("ldr     x1, [fp, "), "missing x1 load: {body}");
        assert!(body.contains("ldr     x2, [fp, "), "missing x2 load: {body}");
        assert!(body.contains("bl      add3"), "missing the bl: {body}");
    }

    #[test]
    fn a_call_loads_arguments_after_every_spill_not_interleaved() {
        // the ordering trap: all three `str` spills must come before the first
        // `ldr` load -- otherwise evaluating arg 1 would clobber arg 0 in x0.
        let asm = compile_program_ok(
            "fn add3(a: i64, b: i64, c: i64) -> i64 { a }\n\
             fn caller() -> i64 { add3(1, 2, 3) }",
        );
        let body = caller_body(&asm);
        let last_spill = body
            .iter()
            .rposition(|l| l.contains("str     x0, [fp, ") && l.contains("// arg "))
            .expect("no argument spill");
        let first_load = body
            .iter()
            .position(|l| l.contains("ldr     x0, [fp, "))
            .expect("no x0 load");
        assert!(
            last_spill < first_load,
            "every spill must precede the first load: {body:?}"
        );
    }

    #[test]
    fn a_call_with_no_arguments_emits_just_a_bl() {
        // a zero-argument call: no spill, no load, just the `bl`.
        let asm = compile_program_ok(
            "fn answer() -> i64 { 42 }\n\
             fn caller() -> i64 { answer() }",
        );
        let body = caller_body(&asm).join("\n");
        assert!(body.contains("bl      answer"), "missing the bl: {body}");
        assert!(!body.contains("// arg "), "a no-arg call spills nothing: {body}");
    }

    #[test]
    fn a_nested_call_emits_two_bls_with_the_inner_call_first() {
        // `outer(inner(5))`: the inner call must complete -- its own bl -- and
        // leave its result in x0 BEFORE the outer call spills that x0 as its
        // argument. so `bl inner` precedes the outer call's argument spill,
        // which precedes `bl outer`.
        let asm = compile_program_ok(
            "fn inner(n: i64) -> i64 { n + 1 }\n\
             fn outer(n: i64) -> i64 { n * 2 }\n\
             fn caller() -> i64 { outer(inner(5)) }",
        );
        let body = caller_body(&asm);
        let inner_bl = body
            .iter()
            .position(|l| l.contains("bl      inner"))
            .expect("missing bl inner");
        let outer_bl = body
            .iter()
            .position(|l| l.contains("bl      outer"))
            .expect("missing bl outer");
        assert!(inner_bl < outer_bl, "the inner call must run first: {body:?}");
        // every argument spill: the inner call's own argument (5) spills
        // BEFORE `bl inner`; the outer call's argument -- the inner call's x0
        // result -- spills AFTER `bl inner`. the FIRST spill after `bl inner`
        // captures that result and must precede `bl outer`.
        let outer_spill = body
            .iter()
            .enumerate()
            .find(|(idx, l)| {
                *idx > inner_bl
                    && l.contains("str     x0, [fp, ")
                    && l.contains("// arg ")
            })
            .map(|(idx, _)| idx)
            .expect("missing the outer-call argument spill after bl inner");
        assert!(
            outer_spill < outer_bl,
            "the inner result is spilled between the two bls: {body:?}"
        );
    }

    #[test]
    fn a_call_to_a_stdlib_function_is_rejected_cleanly() {
        // `abs` is a stdlib function, not a user function -- it is not in
        // `fn_names`, so the backend rejects the call with a clean QalaError
        // (the stdlib is Phase 13), never a panic, never a bl. `abs(i64)`
        // type-checks to `i64`, so the rejection is the backend's, not the
        // typechecker's.
        let src = "fn caller() -> i64 { abs(-1) }";
        let tokens = Lexer::tokenize(src).expect("lex failed");
        let ast = Parser::parse(&tokens).expect("parse failed");
        let (typed, terrors, _) = check_program(&ast, src);
        assert!(terrors.is_empty(), "typecheck errors: {terrors:?}");
        let err = super::super::compile_arm64(&typed, src)
            .expect_err("a stdlib call must be rejected");
        assert!(
            err[0].message().contains("abs"),
            "message: {:?}",
            err[0].message()
        );
    }

    #[test]
    fn a_call_with_a_non_ident_callee_is_rejected_cleanly() {
        // a computed callee -- here a parenthesised expression in callee
        // position -- is outside the integer core. build the typed AST
        // directly: the callee is a Paren, not an Ident.
        use crate::types::QalaType;
        let call = TypedExpr::Call {
            callee: Box::new(TypedExpr::Paren {
                inner: Box::new(TypedExpr::Int {
                    value: 0,
                    ty: QalaType::I64,
                    span: Span::new(0, 1),
                }),
                ty: QalaType::I64,
                span: Span::new(0, 3),
            }),
            args: vec![],
            ty: QalaType::I64,
            span: Span::new(0, 5),
        };
        let mut backend = Arm64Backend::new("");
        let err = backend
            .compile_expr(&call)
            .expect_err("a computed callee must be rejected");
        match err {
            QalaError::Type { message, .. } => {
                assert!(message.contains("computed callee"), "message: {message}");
            }
            other => panic!("expected QalaError::Type, got {other:?}"),
        }
    }

    /// the instruction lines of `name`'s emitted function body -- the lines
    /// from the `name:` label to its `.L<name>_epilogue:` label. a generalised
    /// [`caller_body`] for tests that inspect a function other than `caller`.
    fn fn_body(asm: &str, name: &str) -> Vec<String> {
        let label = format!("{name}:");
        let epilogue = format!(".L{name}_epilogue");
        asm.lines()
            .skip_while(|l| l.trim() != label)
            .take_while(|l| !l.trim_start().starts_with(&epilogue))
            .map(|l| l.to_string())
            .collect()
    }

    /// the `[fp, N]` byte offset referenced by an instruction line, or `None`
    /// if the line has no `[fp, ...]` operand.
    fn fp_offset(line: &str) -> Option<i64> {
        let start = line.find("[fp, ")? + "[fp, ".len();
        let rest = &line[start..];
        let end = rest.find(']')?;
        rest[..end].trim().parse().ok()
    }

    #[test]
    fn a_nested_call_in_a_later_argument_does_not_clobber_an_earlier_argument() {
        // CR-01 regression. `h(100, id(7))` where `h` returns its FIRST
        // parameter: `f` must compute h(100, _) and so return 100. the bug was
        // a single shared argument run -- the nested `id(7)` reused argument
        // slot 0 and overwrote the outer call's stored 100 with 7, so `f`
        // returned 7. the fix gives each argument a distinct CLAIMED scratch
        // slot, so the nested call cannot alias the outer call's slot 0.
        //
        // the distinct-value proof: find the slot the outer `h` call stores
        // its first argument (100) into, then assert that slot is NEVER
        // written again before `bl h` consumes it. if it is provably still
        // 100 at the call, the nested call did not clobber it.
        let asm = compile_program_ok(
            "fn id(x: i64) -> i64 { x }\n\
             fn h(a: i64, b: i64) -> i64 { a }\n\
             fn f() -> i64 { h(100, id(7)) }",
        );
        let body = fn_body(&asm, "f");
        // the line that loads the constant 100 into x0.
        let mov_100 = body
            .iter()
            .position(|l| l.trim() == "mov     x0, 100")
            .expect("missing `mov x0, 100` for the outer call's first argument");
        // the very next line spills x0 -- the outer call's argument 0.
        let spill_line = &body[mov_100 + 1];
        assert!(
            spill_line.contains("str     x0, [fp, ") && spill_line.contains("// arg 0"),
            "the 100 must be spilled as argument 0 right after the mov: {body:?}"
        );
        let arg0_slot = fp_offset(spill_line).expect("argument 0 spill has no [fp, N]");
        // the `bl h` that consumes the loaded arguments.
        let bl_h = body
            .iter()
            .position(|l| l.trim() == "bl      h")
            .expect("missing `bl h`");
        // between spilling 100 and `bl h`, the argument-0 slot must not be
        // written again -- not by the nested `id(7)` call, not by anything.
        let clobber = body[mov_100 + 2..bl_h]
            .iter()
            .find(|l| l.contains("str ") && fp_offset(l) == Some(arg0_slot));
        assert!(
            clobber.is_none(),
            "argument 0 (100) at [fp, {arg0_slot}] was overwritten before `bl h`: \
             {clobber:?} in {body:?}"
        );
        // and the inner call genuinely runs in between -- the test would be
        // vacuous if `id(7)` were folded away.
        assert!(
            body[mov_100 + 2..bl_h].iter().any(|l| l.trim() == "bl      id"),
            "the nested `id(7)` call must run between the two outer arguments: {body:?}"
        );
    }

    #[test]
    fn a_deep_arithmetic_call_argument_keeps_every_slot_inside_the_frame() {
        // CR-02 regression. `id(((a+1)*(a+2)) - ((a+3)*(a+4)))` -- a call whose
        // single argument is a deeply-nested arithmetic expression. the bug
        // was that the frame planner's spill-depth pre-walk had no `Call` arm,
        // so the call contributed zero scratch depth even though its argument
        // expression claims scratch slots; the emitter then stored past the
        // end of the frame -- stack corruption. the fix descends into call
        // arguments, so the frame reserves enough scratch.
        //
        // the proof: read the function's own frame size from its epilogue
        // (`ldp fp, lr, [sp], dealloc`) and assert every `[fp, N]` the body
        // touches satisfies N < dealloc -- inside the frame the function owns.
        let asm = compile_program_ok(
            "fn id(x: i64) -> i64 { x }\n\
             fn f(a: i64) -> i64 { id(((a+1)*(a+2)) - ((a+3)*(a+4))) }",
        );
        // the whole `f` function, label through epilogue, so the `ldp` line
        // (which sits after the epilogue label) is included.
        let f_fn: Vec<&str> = asm
            .lines()
            .skip_while(|l| l.trim() != "f:")
            .take_while(|l| l.trim() != "ret" && !l.trim().is_empty())
            .collect();
        // the epilogue's `ldp fp, lr, [sp], N` carries the frame size N.
        let dealloc: i64 = f_fn
            .iter()
            .find_map(|l| {
                let t = l.trim();
                t.strip_prefix("ldp     fp, lr, [sp], ")
                    .and_then(|n| n.trim().parse().ok())
            })
            .expect("missing the epilogue `ldp` line with the frame size");
        // every fp-relative store/load must land strictly inside [fp, 0]..
        // [fp, dealloc-1]: fp == sp after `mov fp, sp`, so [fp, dealloc] and
        // beyond is the caller's frame.
        for line in &f_fn {
            if let Some(offset) = fp_offset(line) {
                assert!(
                    offset < dealloc,
                    "`{}` writes [fp, {offset}] -- outside the {dealloc}-byte frame",
                    line.trim()
                );
                assert!(offset >= 0, "`{}` has a negative fp offset", line.trim());
            }
        }
        // the test is only meaningful if the body actually spilled to scratch
        // -- a deep expression must produce several distinct spill slots.
        let spill_slots: std::collections::BTreeSet<i64> = f_fn
            .iter()
            .filter(|l| l.contains("str     x0, [fp, "))
            .filter_map(|l| fp_offset(l))
            .collect();
        assert!(
            spill_slots.len() >= 3,
            "a deep arithmetic argument must use several scratch slots: {f_fn:?}"
        );
    }
}
