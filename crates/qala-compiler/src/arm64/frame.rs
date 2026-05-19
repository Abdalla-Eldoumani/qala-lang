//! the per-function stack-frame layout for the ARM64 backend.
//!
//! [`FrameLayout`] holds, for one function, the map from a name (a parameter
//! or -- in plan 12-02 -- a `let` binding) to its `[fp, N]` stack slot, the
//! count of scratch spill slots the expression codegen needs, and the derived
//! `alloc` / `dealloc` frame sizes. [`FrameLayout::plan_frame`] builds it from
//! a [`TypedFnDecl`] in one pass before the prologue is emitted.
//!
//! the slot model is deliberately simple -- correctness over density. every
//! parameter, every `let`, and every scratch spill slot occupies a full 8-byte
//! slot (an `i64` and a `bool` both take 8 bytes; no packing).
//!
//! ## frame geometry: positive offsets from `fp`
//!
//! the prologue is `stp fp, lr, [sp, alloc]!` then `mov fp, sp`, so `fp` ends
//! up at the *bottom* of the allocated frame. the saved `fp`/`lr` pair sits at
//! `[fp, 0]` and `[fp, 8]`; every local slot is therefore at a *positive*
//! offset of 16 or more -- above the saved pair, inside the frame. parameter
//! `i` is at `[fp, 16 + 8*i]`. this matches the CPSC 355 reference
//! `Week 8/example1_scores.asm`, which accesses its locals as
//! `[fp, score1_s]` with `score1_s = 16`. (a negative `[fp, -N]` offset would
//! address *below* `sp`, outside the frame -- a stack-corruption bug.)
//!
//! ## the load-bearing detail: 16-byte stack alignment
//!
//! AArch64 faults on a misaligned `sp` at any memory access or `bl`. the frame
//! size must keep `sp` 16-byte aligned. the formula is
//! `alloc = -(16 + locals) & -16`: `16` is the saved `fp`/`lr` pair, `locals`
//! is the total stack-slot bytes, and `& -16` rounds the magnitude up to a
//! 16-byte multiple (`-16` is `...11110000` in two's complement, so the mask
//! clears the low four bits). the backend never touches `sp` between the
//! prologue and the epilogue -- locals are `fp`-relative -- so alignment
//! correctness reduces entirely to this one formula. [`compute_alloc`] is the
//! single implementation; its unit test asserts the four hand-computed values.

use crate::typed_ast::{TypedExpr, TypedFnDecl, TypedStmt};
use std::collections::HashMap;

/// the byte size of one stack slot. an `i64` and a `bool` both occupy a full
/// 8-byte slot -- the integer core does not pack.
const SLOT_SIZE: i64 = 8;

/// the saved `fp`/`lr` pair size: two 8-byte registers, always 16 bytes and
/// always 16-byte aligned on its own. it occupies `[fp, 0]` and `[fp, 8]`, so
/// it is also the byte offset of the first local slot -- locals start at
/// `[fp, 16]`, just above the saved pair.
const FP_LR_PAIR: i64 = 16;

/// compute the AAPCS64 frame-allocation offset for a function with `locals`
/// bytes of stack-slot space.
///
/// returns the (negative) `alloc` used in `stp fp, lr, [sp, alloc]!`. the
/// formula is `-(16 + locals) & -16` -- the `fp`/`lr` pair plus the locals,
/// the magnitude rounded up to the next 16-byte multiple. `dealloc` (the
/// epilogue's positive add-back) is simply `-alloc`.
///
/// signed `i64` arithmetic is required: the `& -16` mask relies on `-16`'s
/// two's-complement bit pattern.
fn compute_alloc(locals: i64) -> i64 {
    -(FP_LR_PAIR + locals) & -16
}

/// the stack-frame layout of one function: the name-to-slot map, the scratch
/// spill-slot reservation, and the derived `alloc` / `dealloc` sizes.
///
/// crate-private: the layout is an internal detail of the `arm64` module.
pub(crate) struct FrameLayout {
    /// map from a parameter (or, from plan 12-02, a `let`) name to its stack
    /// slot -- the positive byte offset `N` in `[fp, N]`.
    slots: HashMap<String, i64>,
    /// the number of scratch spill slots reserved for the expression codegen's
    /// nested-binary-operator spill discipline. their offsets sit immediately
    /// past the named slots. read by [`FrameLayout::scratch_count`], which
    /// plan 12-02 consults when extending the frame with `let`-binding slots.
    #[allow(dead_code)]
    scratch_count: i64,
    /// the byte offset of the first scratch spill slot: `[fp, 16]` past the
    /// saved pair, plus the named slots, is where the scratch stack begins.
    scratch_base: i64,
    /// how many scratch slots are currently claimed. the expression codegen
    /// claims on entering a binary op and releases on leaving; this never
    /// exceeds `scratch_count`.
    scratch_in_use: i64,
    /// the function's `alloc` offset (negative) for the prologue `stp`.
    alloc: i64,
}

impl FrameLayout {
    /// build the frame layout for one function: assign every parameter a slot,
    /// reserve scratch spill slots for the body's deepest binary-operator
    /// nesting, and compute `alloc` / `dealloc`.
    ///
    /// THIS PLAN walks `decl.params` only -- there are no `let` bindings yet
    /// (plan 12-02 adds the body-`let` walk into this same function). the
    /// scratch reservation is derived from a pre-walk of `decl.body`:
    /// [`max_spill_depth`] returns the maximum number of binary-operator LHS
    /// values that are simultaneously spilled. plan 12-02 extends this with the
    /// `let`-slot assignment alongside the existing parameter assignment.
    pub(crate) fn plan_frame(decl: &TypedFnDecl) -> FrameLayout {
        let mut slots = HashMap::new();
        // parameter i lives at [fp, 16 + 8*i]: the first param at [fp, 16],
        // the second at [fp, 24], and so on -- just above the saved fp/lr pair.
        for (i, param) in decl.params.iter().enumerate() {
            let slot = FP_LR_PAIR + SLOT_SIZE * i as i64;
            slots.insert(param.name.clone(), slot);
        }
        let named_slot_count = decl.params.len() as i64;
        // the scratch spill slots sit immediately past the named slots.
        let scratch_base = FP_LR_PAIR + SLOT_SIZE * named_slot_count;
        let scratch_count = max_spill_depth(&decl.body);
        let locals = SLOT_SIZE * (named_slot_count + scratch_count);
        let alloc = compute_alloc(locals);
        FrameLayout {
            slots,
            scratch_count,
            scratch_base,
            scratch_in_use: 0,
            alloc,
        }
    }

    /// the stack slot of a named binding, or `None` if the name is unknown.
    ///
    /// the returned value is the positive offset `N` for `[fp, N]`. an unknown
    /// name is a backend bug -- the caller turns a `None` into a `QalaError`
    /// rather than panicking.
    pub(crate) fn slot_of(&self, name: &str) -> Option<i64> {
        self.slots.get(name).copied()
    }

    /// claim the next scratch spill slot, returning its `[fp, N]` offset.
    ///
    /// the expression codegen calls this on entering a binary operator (to hold
    /// the evaluated LHS while the RHS subtree runs) and [`release_scratch`] on
    /// leaving. claims stack: nested binary ops each get a distinct slot. the
    /// reservation from [`plan_frame`] guarantees a slot is always available.
    pub(crate) fn claim_scratch(&mut self) -> i64 {
        // the slot offset: past the named slots, indexed by how many scratch
        // slots are already in use.
        let offset = self.scratch_base + SLOT_SIZE * self.scratch_in_use;
        self.scratch_in_use += 1;
        offset
    }

    /// release the most recently claimed scratch spill slot.
    ///
    /// the inverse of [`claim_scratch`]; the two must be balanced so the
    /// scratch stack returns to empty at the end of every expression.
    pub(crate) fn release_scratch(&mut self) {
        debug_assert!(self.scratch_in_use > 0, "release_scratch with no slot claimed");
        self.scratch_in_use -= 1;
    }

    /// the prologue instruction text: save `fp`/`lr` and set the frame pointer.
    ///
    /// emits the pre-indexed `stp fp, lr, [sp, <alloc>]!` (allocate the frame
    /// and store the pair) followed by `mov fp, sp`. `alloc` is the computed
    /// numeric literal, not an m4 symbol, so the emitted text is self-contained.
    pub(crate) fn prologue(&self) -> [String; 2] {
        [
            format!("stp     fp, lr, [sp, {}]!", self.alloc),
            "mov     fp, sp".to_string(),
        ]
    }

    /// the epilogue instruction text: restore `fp`/`lr` and return.
    ///
    /// emits the post-indexed `ldp fp, lr, [sp], <dealloc>` (load the pair and
    /// free the frame) followed by `ret`. `dealloc` is `-alloc`, the positive
    /// frame magnitude.
    pub(crate) fn epilogue(&self) -> [String; 2] {
        [
            format!("ldp     fp, lr, [sp], {}", -self.alloc),
            "ret".to_string(),
        ]
    }

    /// the number of scratch spill slots this frame reserves for the
    /// expression codegen's nested-binary-operator spill discipline.
    ///
    /// plan 12-02 consults this when it extends `plan_frame` to also count
    /// `let`-binding slots -- the scratch reservation must survive that
    /// extension. tests assert it against known expression shapes.
    #[allow(dead_code)]
    pub(crate) fn scratch_count(&self) -> i64 {
        self.scratch_count
    }
}

/// the maximum number of binary-operator LHS values that are spilled at once
/// while evaluating `block`.
///
/// the expression codegen, on an arithmetic or comparison `Binary` node, spills
/// the evaluated LHS to a scratch slot while it evaluates the RHS subtree. a
/// nested RHS spills again, so the live-spill count is the deepest
/// left-leaning-through-right chain of those operators. this pre-walk returns
/// that maximum so [`plan_frame`] can reserve exactly that many scratch slots.
/// short-circuit `&&` / `||` do NOT spill -- they branch on `x0` directly and
/// reload no LHS -- so they add no scratch slot of their own.
fn max_spill_depth(block: &crate::typed_ast::TypedBlock) -> i64 {
    let mut max = 0;
    for stmt in &block.stmts {
        max = max.max(stmt_spill_depth(stmt));
    }
    if let Some(value) = &block.value {
        max = max.max(expr_spill_depth(value));
    }
    max
}

/// the maximum simultaneously-live spill count within one statement.
///
/// THIS PLAN's `compile_stmt` handles only `Expr` and `Return`; the other
/// variants still contribute their sub-expressions' depth so the count is
/// correct once plan 12-02 wires the remaining statement codegen.
fn stmt_spill_depth(stmt: &TypedStmt) -> i64 {
    match stmt {
        TypedStmt::Expr { expr, .. } => expr_spill_depth(expr),
        TypedStmt::Return { value, .. } => {
            value.as_ref().map(expr_spill_depth).unwrap_or(0)
        }
        TypedStmt::Let { init, .. } => expr_spill_depth(init),
        TypedStmt::If { cond, then_block, else_branch, .. } => {
            let mut d = expr_spill_depth(cond).max(max_spill_depth(then_block));
            if let Some(branch) = else_branch {
                d = d.max(else_branch_spill_depth(branch));
            }
            d
        }
        TypedStmt::While { cond, body, .. } => {
            expr_spill_depth(cond).max(max_spill_depth(body))
        }
        TypedStmt::For { iter, body, .. } => {
            expr_spill_depth(iter).max(max_spill_depth(body))
        }
        TypedStmt::Defer { expr, .. } => expr_spill_depth(expr),
        TypedStmt::Break { .. } | TypedStmt::Continue { .. } => 0,
    }
}

/// the maximum spill count within an `else` branch (a block or an `else if`).
fn else_branch_spill_depth(branch: &crate::typed_ast::TypedElseBranch) -> i64 {
    use crate::typed_ast::TypedElseBranch;
    match branch {
        TypedElseBranch::Block(b) => max_spill_depth(b),
        TypedElseBranch::If(stmt) => stmt_spill_depth(stmt),
    }
}

/// the maximum number of scratch spill slots that are simultaneously live
/// while evaluating `expr`.
///
/// an arithmetic or comparison `Binary` node claims one scratch slot and holds
/// it for the *entire* evaluation of both operands -- the codegen claims before
/// the LHS runs and releases only after the RHS reload -- so its peak live-slot
/// count is `1 + max(spill_depth(lhs), spill_depth(rhs))`. a short-circuit
/// `&&` / `||` claims no slot of its own (it branches on `x0` and reloads no
/// LHS), so its peak is `max(spill_depth(lhs), spill_depth(rhs))`. non-binary
/// nodes recurse into their children with no added slot.
fn expr_spill_depth(expr: &TypedExpr) -> i64 {
    use crate::ast::BinOp;
    match expr {
        TypedExpr::Binary { op, lhs, rhs, .. } => {
            let operands = expr_spill_depth(lhs).max(expr_spill_depth(rhs));
            match op {
                // && and || branch on x0 -- no LHS reload, no scratch slot.
                BinOp::And | BinOp::Or => operands,
                // arithmetic and comparison hold one slot across BOTH operands.
                _ => 1 + operands,
            }
        }
        TypedExpr::Unary { operand, .. } => expr_spill_depth(operand),
        TypedExpr::Paren { inner, .. } => expr_spill_depth(inner),
        TypedExpr::Block { block, .. } => max_spill_depth(block),
        // a leaf or an unsupported construct contributes no spill.
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::BinOp;
    use crate::effects::EffectSet;
    use crate::span::Span;
    use crate::types::QalaType;

    fn span(n: usize) -> Span {
        Span::new(n, n + 1)
    }

    fn int(value: i64) -> TypedExpr {
        TypedExpr::Int { value, ty: QalaType::I64, span: span(0) }
    }

    /// build a TypedFnDecl with the given parameter count and body.
    fn fn_decl(param_count: usize, body: crate::typed_ast::TypedBlock) -> TypedFnDecl {
        let params = (0..param_count)
            .map(|i| crate::typed_ast::TypedParam {
                is_self: false,
                name: format!("p{i}"),
                ty: QalaType::I64,
                default: None,
                span: span(i),
            })
            .collect();
        TypedFnDecl {
            type_name: None,
            name: "f".to_string(),
            params,
            ret_ty: QalaType::I64,
            effect: EffectSet::pure(),
            body,
            span: span(0),
        }
    }

    fn empty_body() -> crate::typed_ast::TypedBlock {
        crate::typed_ast::TypedBlock {
            stmts: vec![],
            value: None,
            ty: QalaType::Void,
            span: span(0),
        }
    }

    #[test]
    fn alloc_formula_matches_the_hand_computed_values() {
        // THE load-bearing test: alloc = -(16 + locals) & -16 for the four
        // hand-computed cases. these concrete values are independent of any
        // snapshot fixture -- a wrong formula fails here regardless.
        assert_eq!(compute_alloc(0), -16, "locals=0 must round to -16");
        assert_eq!(compute_alloc(8), -32, "locals=8 (24 bytes) must round up to -32");
        assert_eq!(compute_alloc(16), -32, "locals=16 (32 bytes) must round to -32");
        assert_eq!(compute_alloc(24), -48, "locals=24 (40 bytes) must round up to -48");
    }

    #[test]
    fn alloc_is_always_a_negative_multiple_of_sixteen() {
        // every frame size, for a range of locals, keeps sp 16-byte aligned.
        for locals in (0..=512).step_by(8) {
            let a = compute_alloc(locals);
            assert!(a <= -16, "alloc must reserve at least the fp/lr pair");
            assert_eq!(a % 16, 0, "alloc must be a multiple of 16 for locals={locals}");
        }
    }

    #[test]
    fn a_leaf_function_with_no_params_has_a_minus_sixteen_frame() {
        // a 0-parameter function with an empty body: no named slots, no
        // scratch, locals = 0 -> alloc = -16, dealloc = 16. the concrete
        // prologue/epilogue text is asserted verbatim.
        let layout = FrameLayout::plan_frame(&fn_decl(0, empty_body()));
        assert_eq!(layout.alloc, -16);
        assert_eq!(layout.prologue()[0], "stp     fp, lr, [sp, -16]!");
        assert_eq!(layout.prologue()[1], "mov     fp, sp");
        assert_eq!(layout.epilogue()[0], "ldp     fp, lr, [sp], 16");
        assert_eq!(layout.epilogue()[1], "ret");
    }

    #[test]
    fn three_params_no_scratch_gives_a_minus_thirty_two_frame() {
        // a 3-parameter function with a body that needs no scratch (the
        // trailing value is a bare ident): 3 named slots = 24 bytes locals,
        // alloc = -(16+24) & -16 = -48. concrete value, independent of snapshots.
        let body = crate::typed_ast::TypedBlock {
            stmts: vec![],
            value: Some(Box::new(TypedExpr::Ident {
                name: "p0".to_string(),
                ty: QalaType::I64,
                span: span(0),
            })),
            ty: QalaType::I64,
            span: span(0),
        };
        let layout = FrameLayout::plan_frame(&fn_decl(3, body));
        assert_eq!(layout.scratch_count(), 0, "a bare ident needs no scratch");
        assert_eq!(layout.alloc, -48, "3 params (24 bytes) -> alloc -48");
    }

    #[test]
    fn parameter_slots_start_at_sixteen_above_the_saved_pair() {
        // parameter 0 -> [fp, 16], parameter 1 -> [fp, 24], parameter 2 ->
        // [fp, 32]: positive offsets just past the saved fp/lr pair.
        let layout = FrameLayout::plan_frame(&fn_decl(3, empty_body()));
        assert_eq!(layout.slot_of("p0"), Some(16));
        assert_eq!(layout.slot_of("p1"), Some(24));
        assert_eq!(layout.slot_of("p2"), Some(32));
        assert_eq!(layout.slot_of("missing"), None);
    }

    #[test]
    fn max_spill_depth_is_zero_for_a_leaf_expression() {
        // a bare literal trailing value: no binary op, no spill.
        let body = crate::typed_ast::TypedBlock {
            stmts: vec![],
            value: Some(Box::new(int(7))),
            ty: QalaType::I64,
            span: span(0),
        };
        assert_eq!(max_spill_depth(&body), 0);
    }

    #[test]
    fn max_spill_depth_counts_a_single_binary_op_as_one() {
        // `1 + 2`: the LHS is spilled once while the RHS (a leaf) runs.
        let add = TypedExpr::Binary {
            op: BinOp::Add,
            lhs: Box::new(int(1)),
            rhs: Box::new(int(2)),
            ty: QalaType::I64,
            span: span(0),
        };
        let body = crate::typed_ast::TypedBlock {
            stmts: vec![],
            value: Some(Box::new(add)),
            ty: QalaType::I64,
            span: span(0),
        };
        assert_eq!(max_spill_depth(&body), 1);
    }

    #[test]
    fn max_spill_depth_grows_with_right_leaning_nesting() {
        // `1 + (2 + (3 + 4))`: the outer op spills, its RHS is itself a binary
        // op that spills, and so on -- three simultaneously-live spills.
        let inner = TypedExpr::Binary {
            op: BinOp::Add,
            lhs: Box::new(int(3)),
            rhs: Box::new(int(4)),
            ty: QalaType::I64,
            span: span(0),
        };
        let mid = TypedExpr::Binary {
            op: BinOp::Add,
            lhs: Box::new(int(2)),
            rhs: Box::new(inner),
            ty: QalaType::I64,
            span: span(0),
        };
        let outer = TypedExpr::Binary {
            op: BinOp::Add,
            lhs: Box::new(int(1)),
            rhs: Box::new(mid),
            ty: QalaType::I64,
            span: span(0),
        };
        let body = crate::typed_ast::TypedBlock {
            stmts: vec![],
            value: Some(Box::new(outer)),
            ty: QalaType::I64,
            span: span(0),
        };
        assert_eq!(max_spill_depth(&body), 3);
    }

    #[test]
    fn left_leaning_nesting_accumulates_spills() {
        // `((1 + 2) + 3) + 4`: each operator claims its scratch slot BEFORE
        // evaluating its LHS and holds it across the whole LHS subtree, so a
        // left-leaning chain accumulates -- three operators deep means three
        // simultaneously-live slots when the innermost `1 + 2` runs.
        let a = TypedExpr::Binary {
            op: BinOp::Add,
            lhs: Box::new(int(1)),
            rhs: Box::new(int(2)),
            ty: QalaType::I64,
            span: span(0),
        };
        let b = TypedExpr::Binary {
            op: BinOp::Add,
            lhs: Box::new(a),
            rhs: Box::new(int(3)),
            ty: QalaType::I64,
            span: span(0),
        };
        let c = TypedExpr::Binary {
            op: BinOp::Add,
            lhs: Box::new(b),
            rhs: Box::new(int(4)),
            ty: QalaType::I64,
            span: span(0),
        };
        let body = crate::typed_ast::TypedBlock {
            stmts: vec![],
            value: Some(Box::new(c)),
            ty: QalaType::I64,
            span: span(0),
        };
        assert_eq!(max_spill_depth(&body), 3);
    }

    #[test]
    fn claim_and_release_scratch_balance() {
        // claim hands out distinct slots past the named slots; release unwinds.
        let mut layout = FrameLayout::plan_frame(&fn_decl(2, empty_body()));
        // 2 params -> scratch_base = 16 + 8*2 = 32; the scratch slots run
        // [fp, 32], [fp, 40], ...
        let first = layout.claim_scratch();
        let second = layout.claim_scratch();
        assert_eq!(first, 32);
        assert_eq!(second, 40);
        assert_ne!(first, second, "nested claims must be distinct slots");
        layout.release_scratch();
        layout.release_scratch();
        // after a balanced unwind a fresh claim reuses the first slot.
        assert_eq!(layout.claim_scratch(), 32);
    }

    #[test]
    fn scratch_slots_are_counted_into_the_frame() {
        // a body whose trailing value is `1 + 2` reserves one scratch slot;
        // with 1 param that is 2 slots = 16 bytes -> alloc = -32.
        let add = TypedExpr::Binary {
            op: BinOp::Add,
            lhs: Box::new(int(1)),
            rhs: Box::new(int(2)),
            ty: QalaType::I64,
            span: span(0),
        };
        let body = crate::typed_ast::TypedBlock {
            stmts: vec![],
            value: Some(Box::new(add)),
            ty: QalaType::I64,
            span: span(0),
        };
        let layout = FrameLayout::plan_frame(&fn_decl(1, body));
        assert_eq!(layout.scratch_count(), 1);
        // 1 param + 1 scratch = 2 slots = 16 bytes locals -> -(16+16)&-16 = -32.
        assert_eq!(layout.alloc, -32);
    }

    /// lex, parse, and typecheck `src`, returning the first function's decl.
    fn fn_from_source(src: &str) -> TypedFnDecl {
        let tokens = crate::lexer::Lexer::tokenize(src).expect("lex failed");
        let ast = crate::parser::Parser::parse(&tokens).expect("parse failed");
        let (typed, terrors, _) = crate::typechecker::check_program(&ast, src);
        assert!(terrors.is_empty(), "typecheck errors: {terrors:?}");
        typed
            .into_iter()
            .find_map(|item| match item {
                crate::typed_ast::TypedItem::Fn(d) => Some(d),
                _ => None,
            })
            .expect("no function in source")
    }

    #[test]
    fn the_arithmetic_snapshot_function_has_a_minus_sixty_four_frame() {
        // the exact program in the arithmetic snapshot test. its trailing value
        // `(a + b) * (a - b) / 2 % 7` is four operators deep on the left spine
        // -- %, /, *, + -- each holding a scratch slot across its LHS subtree,
        // so the peak live-slot count is 4. with 2 parameters that is 6 slots =
        // 48 bytes of locals -> alloc = -(16 + 48) & -16 = -64. this concrete
        // value is asserted independently of the snapshot fixture: a wrong
        // spill-depth count fails here regardless of what the fixture holds.
        let decl = fn_from_source("fn arith(a: i64, b: i64) -> i64 { (a + b) * (a - b) / 2 % 7 }");
        let layout = FrameLayout::plan_frame(&decl);
        assert_eq!(layout.scratch_count(), 4, "the left spine is 4 operators deep");
        assert_eq!(layout.alloc, -64, "2 params + 4 scratch = 48 bytes -> alloc -64");
        assert_eq!(layout.prologue()[0], "stp     fp, lr, [sp, -64]!");
        assert_eq!(layout.epilogue()[0], "ldp     fp, lr, [sp], 64");
    }

    #[test]
    fn the_comparison_snapshot_function_has_a_minus_forty_eight_frame() {
        // the comparison snapshot's `cmp` function: `a == b` is one operator,
        // one scratch slot. 2 params + 1 scratch = 3 slots = 24 bytes -> alloc
        // = -(16 + 24) & -16 = -48. a concrete value, not an inequality.
        let decl = fn_from_source("fn cmp(a: i64, b: i64) -> bool { a == b }");
        let layout = FrameLayout::plan_frame(&decl);
        assert_eq!(layout.scratch_count(), 1, "one comparison -> one scratch slot");
        assert_eq!(layout.alloc, -48, "2 params + 1 scratch = 24 bytes -> alloc -48");
    }

    #[test]
    fn the_boolean_snapshot_function_has_a_minus_thirty_two_frame() {
        // the boolean snapshot's `andor` function: `a && b || !a`. the `&&` and
        // `||` short-circuit -- they claim no scratch slot -- and `!a` is unary.
        // so the function reserves zero scratch slots. 2 params + 0 scratch = 2
        // slots = 16 bytes -> alloc = -(16 + 16) & -16 = -32.
        let decl = fn_from_source("fn andor(a: bool, b: bool) -> bool { a && b || !a }");
        let layout = FrameLayout::plan_frame(&decl);
        assert_eq!(layout.scratch_count(), 0, "short-circuit ops claim no scratch");
        assert_eq!(layout.alloc, -32, "2 params + 0 scratch = 16 bytes -> alloc -32");
    }
}
