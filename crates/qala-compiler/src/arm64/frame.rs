//! the per-function stack-frame layout for the ARM64 backend.
//!
//! [`FrameLayout`] holds, for one function, the map from a parameter name to
//! its `[fp, N]` stack slot, an indexed list of slots for the body's `let`
//! bindings and `for`-loop slots, the count of scratch spill slots the
//! expression codegen needs, and the derived `alloc` / `dealloc` frame sizes.
//! [`FrameLayout::plan_frame`] builds it from a [`TypedFnDecl`] in one pass
//! before the prologue is emitted.
//!
//! the slot model is deliberately simple -- correctness over density. every
//! parameter, every `let`, and every scratch spill slot occupies a full 8-byte
//! slot (an `i64` and a `bool` both take 8 bytes; no packing).
//!
//! ## binding slots: indexed by walk order, not by name
//!
//! a `let` may shadow an outer `let` of the same name, so a name-keyed map
//! cannot hold every binding's slot -- two `let x` would collide. instead
//! [`plan_frame`] walks the body in a fixed source order and assigns binding
//! occurrence `0, 1, 2, ...` the next slot, recording them in a `Vec` indexed
//! by that occurrence number. `stmt.rs` walks the body in the identical order,
//! keeping its own occurrence counter, and reads
//! [`FrameLayout::binding_slot`] by index. the deterministic walk order is the
//! shared contract between the two files: a parameter slot is named, a `let`
//! or `for` slot is positional. NO SLOT REUSE -- every binding occurrence,
//! including a shadowing `let` and a `for`-loop variable, gets its own
//! distinct slot.
//!
//! ## call arguments: spilled to claimed scratch slots, never a shared run
//!
//! a `TypedExpr::Call` evaluates each argument into `x0` one at a time, then
//! loads `x0`-`x{n-1}` from somewhere stable for the `bl` -- because every
//! `compile_expr` lands its result in `x0`, evaluating arg 1 would clobber
//! arg 0. each evaluated argument is `str`-ed to a freshly CLAIMED scratch
//! slot -- the same [`claim_scratch`](FrameLayout::claim_scratch) /
//! [`release_scratch`](FrameLayout::release_scratch) stack the nested-binary
//! spill discipline uses -- and all of them are released only after the `bl`.
//! a nested call inside one argument therefore claims slots STRICTLY BEYOND
//! the outer call's already-claimed argument slots, so no two live argument
//! values ever alias the same slot. there is no dedicated, shared argument
//! region: an argument slot is just another scratch slot, and the scratch
//! reservation in [`plan_frame`] is sized to cover it (see [`max_spill_depth`]
//! and its [`TypedExpr::Call`] arm).
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
//! address *below* `sp`, outside the frame -- a stack-corruption bug.) the
//! slot regions in `[fp, N]` order: saved pair, parameters, binding slots,
//! expression scratch slots (which also hold a call's spilled arguments).
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

use crate::typed_ast::{
    TypedBlock, TypedElseBranch, TypedExpr, TypedFnDecl, TypedInterpPart, TypedStmt,
};
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

/// the stack-frame layout of one function: the parameter name-to-slot map,
/// the body's binding slots, the scratch spill-slot reservation, and the
/// derived `alloc` / `dealloc` sizes.
///
/// crate-private: the layout is an internal detail of the `arm64` module.
pub(crate) struct FrameLayout {
    /// map from a parameter name to its stack slot -- the positive byte offset
    /// `N` in `[fp, N]`. parameters are name-keyed because a parameter name is
    /// unique within a signature; body `let`/`for` slots are positional (see
    /// [`binding_slots`](FrameLayout::binding_slots)).
    slots: HashMap<String, i64>,
    /// the stack slots for the body's `let` bindings and `for`-loop slots, in
    /// the deterministic walk order [`plan_frame`] visits them. binding
    /// occurrence `k` -- the `k`-th `let`, `for`-variable, or `for`-end slot
    /// the walk reaches -- is at `binding_slots[k]`. `stmt.rs` re-walks in
    /// lockstep and indexes this by occurrence number, which survives a
    /// shadowing `let` that a name-keyed map could not.
    binding_slots: Vec<i64>,
    /// the number of scratch spill slots reserved for the expression codegen.
    /// this covers both the nested-binary-operator spill discipline and a
    /// call's spilled arguments -- a call argument is `str`-ed to a claimed
    /// scratch slot, so the reservation must account for it (see
    /// [`max_spill_depth`]). their offsets sit immediately past the named and
    /// binding slots.
    #[allow(dead_code)]
    scratch_count: i64,
    /// the byte offset of the first scratch spill slot: `[fp, 16]` past the
    /// saved pair, plus the named and binding slots, is where the scratch
    /// stack begins.
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
    /// assign every body `let` binding and `for`-loop slot a distinct slot,
    /// reserve scratch spill slots for the body's deepest expression nesting,
    /// and compute `alloc` / `dealloc`.
    ///
    /// the slot regions, in `[fp, N]` order: the `FP_LR_PAIR` saved-register
    /// pair occupies `[fp, 0]` and `[fp, 8]`; then the parameters at
    /// `[fp, 16 + 8*i]`; then the body's binding slots (every `let` and, per
    /// `for`, the loop variable and a dedicated `end`-bound slot) in walk
    /// order; then the expression scratch slots.
    ///
    /// the binding slots are assigned by [`count_binding_slots`], a recursive
    /// walk over `decl.body` that descends into nested `if`/`while`/`for`
    /// blocks. no slot is reused -- a shadowing `let` and every `for` variable
    /// get their own distinct slot. the scratch reservation is derived from a
    /// pre-walk: [`max_spill_depth`] returns the peak number of scratch slots
    /// simultaneously live -- a binary operator's spilled LHS values and a
    /// call's spilled arguments together.
    pub(crate) fn plan_frame(decl: &TypedFnDecl) -> FrameLayout {
        let mut slots = HashMap::new();
        // parameter i lives at [fp, 16 + 8*i]: the first param at [fp, 16],
        // the second at [fp, 24], and so on -- just above the saved fp/lr pair.
        for (i, param) in decl.params.iter().enumerate() {
            let slot = FP_LR_PAIR + SLOT_SIZE * i as i64;
            slots.insert(param.name.clone(), slot);
        }
        let param_count = decl.params.len() as i64;
        // the body's binding slots begin immediately past the parameter slots:
        // a `let` and, per `for`, two slots (the loop variable plus a dedicated
        // `end`-bound slot the For arm evaluates `end` into once).
        let binding_base = FP_LR_PAIR + SLOT_SIZE * param_count;
        let binding_count = count_binding_slots(&decl.body);
        let binding_slots: Vec<i64> = (0..binding_count)
            .map(|k| binding_base + SLOT_SIZE * k)
            .collect();
        let named_slot_count = param_count + binding_count;
        // the expression scratch slots sit immediately past the binding slots.
        // they hold both a binary operator's spilled LHS and a call's spilled
        // arguments -- `max_spill_depth` accounts for both.
        let scratch_base = FP_LR_PAIR + SLOT_SIZE * named_slot_count;
        let scratch_count = max_spill_depth(&decl.body);
        let locals = SLOT_SIZE * (named_slot_count + scratch_count);
        let alloc = compute_alloc(locals);
        FrameLayout {
            slots,
            binding_slots,
            scratch_count,
            scratch_base,
            scratch_in_use: 0,
            alloc,
        }
    }

    /// the stack slot of a named parameter, or `None` if the name is unknown.
    ///
    /// the returned value is the positive offset `N` for `[fp, N]`. an unknown
    /// name is a backend bug -- the caller turns a `None` into a `QalaError`
    /// rather than panicking. `let` and `for` bindings are NOT resolved here --
    /// they are positional; see [`binding_slot`](FrameLayout::binding_slot).
    pub(crate) fn slot_of(&self, name: &str) -> Option<i64> {
        self.slots.get(name).copied()
    }

    /// the stack slot of body binding occurrence `index` -- the `index`-th
    /// `let`, `for`-variable, or `for`-end slot in [`plan_frame`]'s walk order.
    ///
    /// `stmt.rs` walks the body in the same order [`plan_frame`] used, keeping
    /// its own occurrence counter, and calls this with the next index when it
    /// reaches a `let` or `for`. an out-of-range index is a backend bug -- a
    /// walk-order mismatch between `plan_frame` and `stmt.rs` -- and the caller
    /// turns the `None` into a `QalaError` rather than panicking.
    pub(crate) fn binding_slot(&self, index: usize) -> Option<i64> {
        self.binding_slots.get(index).copied()
    }

    /// claim the next scratch spill slot, returning its `[fp, N]` offset.
    ///
    /// the expression codegen calls this on entering a binary operator (to hold
    /// the evaluated LHS while the RHS subtree runs), and once per argument of
    /// a call (to hold each evaluated argument until the `bl`); the matching
    /// [`release_scratch`] returns the slot. claims stack: nested binary ops
    /// and a nested call each get distinct slots strictly beyond the slots
    /// already claimed, so no two simultaneously-live values ever alias. the
    /// reservation from [`plan_frame`] -- via [`max_spill_depth`], which counts
    /// a call's argument spills -- guarantees a slot is always available.
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

/// the count of body binding slots a function needs: one slot per `let`
/// binding, two slots per `for` loop (the loop variable and a dedicated
/// `end`-bound slot the For arm evaluates `end` into once).
///
/// THE WALK ORDER is the contract between this function and `stmt.rs`. both
/// descend `TypedBlock.stmts` in source order, into `If.then_block` then the
/// `else` branch, into `While.body`, and into `For.body`. a `for` reserves its
/// variable slot first, then its `end` slot, then descends into the body --
/// `stmt.rs` claims the slots in that same order. as long as the two walks
/// match, occurrence `k` here is occurrence `k` there. the `for`-variable slot
/// is registered into the body's scope by `stmt.rs`, so an `Ident` reference
/// to the loop variable resolves to it.
fn count_binding_slots(block: &TypedBlock) -> i64 {
    let mut count = 0;
    for stmt in &block.stmts {
        count += stmt_binding_slots(stmt);
    }
    count
}

/// the binding-slot count contributed by one statement (see
/// [`count_binding_slots`] for the walk-order contract).
fn stmt_binding_slots(stmt: &TypedStmt) -> i64 {
    match stmt {
        // a `let` is one slot; its initializer holds no further bindings of
        // its own (an initializer is an expression, and expressions carry no
        // `let`/`for` -- a block expression's statements would, but the integer
        // core's `Block` expression is reached only as a trailing value, which
        // this walk already covers via the body it belongs to).
        TypedStmt::Let { .. } => 1,
        TypedStmt::If { then_block, else_branch, .. } => {
            let mut c = count_binding_slots(then_block);
            if let Some(branch) = else_branch {
                c += else_branch_binding_slots(branch);
            }
            c
        }
        TypedStmt::While { body, .. } => count_binding_slots(body),
        // a `for` is two slots -- the loop variable and the `end` bound --
        // counted BEFORE descending into the body, matching `stmt.rs`.
        TypedStmt::For { body, .. } => 2 + count_binding_slots(body),
        // these statements introduce no binding slot.
        TypedStmt::Return { .. }
        | TypedStmt::Break { .. }
        | TypedStmt::Continue { .. }
        | TypedStmt::Defer { .. }
        | TypedStmt::Expr { .. } => 0,
    }
}

/// the binding-slot count within an `else` branch (a block or an `else if`).
fn else_branch_binding_slots(branch: &TypedElseBranch) -> i64 {
    match branch {
        TypedElseBranch::Block(b) => count_binding_slots(b),
        TypedElseBranch::If(stmt) => stmt_binding_slots(stmt),
    }
}

/// the peak number of scratch slots simultaneously live while evaluating
/// `block`.
///
/// a scratch slot is claimed in two situations, and this pre-walk counts both
/// so [`plan_frame`] reserves enough for the deepest case:
///
/// - an arithmetic or comparison `Binary` node spills its evaluated LHS to a
///   scratch slot while it evaluates the RHS subtree. a nested operator spills
///   again, so a chain accumulates. short-circuit `&&` / `||` do NOT spill --
///   they branch on `x0` directly and reload no LHS.
/// - a `Call` spills each evaluated argument to a claimed scratch slot and
///   holds every one of them until the `bl`. its `n` arguments therefore
///   occupy `n` simultaneously-live slots, and evaluating any one argument can
///   itself need scratch slots on top.
///
/// the contract with the expression codegen: every slot the codegen claims is
/// counted here. a single shared argument run would NOT be safe (a nested call
/// would alias the outer call's saved arguments), which is why a call argument
/// is a claimed scratch slot like any other.
fn max_spill_depth(block: &TypedBlock) -> i64 {
    let mut max = 0;
    for stmt in &block.stmts {
        max = max.max(stmt_spill_depth(stmt));
    }
    if let Some(value) = &block.value {
        max = max.max(expr_spill_depth(value));
    }
    max
}

/// the peak simultaneously-live scratch-slot count within one statement.
///
/// every statement variant contributes the depth of its sub-expressions, so
/// the count covers a call or a binary operator wherever one appears -- a
/// `let` initializer, an `if` condition, a `for` range bound, a `return`
/// value, a bare expression statement.
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
fn else_branch_spill_depth(branch: &TypedElseBranch) -> i64 {
    match branch {
        TypedElseBranch::Block(b) => max_spill_depth(b),
        TypedElseBranch::If(stmt) => stmt_spill_depth(stmt),
    }
}

/// the peak number of scratch spill slots simultaneously live while
/// evaluating `expr`.
///
/// an arithmetic or comparison `Binary` node claims one scratch slot and holds
/// it for the *entire* evaluation of both operands -- the codegen claims before
/// the LHS runs and releases only after the RHS reload -- so its peak live-slot
/// count is `1 + max(spill_depth(lhs), spill_depth(rhs))`. a short-circuit
/// `&&` / `||` claims no slot of its own (it branches on `x0` and reloads no
/// LHS), so its peak is `max(spill_depth(lhs), spill_depth(rhs))`.
///
/// a `Call` with `n` arguments spills each evaluated argument to a claimed
/// scratch slot and holds all `n` until the `bl` -- so `n` slots are live at
/// once -- and evaluating any single argument can claim further slots on top.
/// the codegen evaluates an argument fully before claiming its slot, so while
/// argument `k` runs at most `k` sibling slots are already held; the peak is
/// therefore bounded by `n + max(depth of each argument subtree)`. that bound
/// (the callee subtree is included for completeness, though a supported callee
/// is always a bare identifier of depth 0) is what this arm returns, matching
/// the review's `args.len() + max(arg subtree depth)` model. non-call,
/// non-binary nodes recurse into their children with no added slot.
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
        // a `print` / `println` call lowers to a `printf` call, not a normal
        // `bl`: its scratch need is one slot per i64 interpolation hole, not
        // one per argument -- the single `str` argument is itself walked into
        // the holes. `printf_call_spill_depth` computes that; a non-printf
        // call falls through to the general arity-plus-depth rule below.
        TypedExpr::Call { callee, args, .. } if is_print_callee(callee) => {
            printf_call_spill_depth(args)
        }
        // a call holds one scratch slot per argument simultaneously (the
        // spilled argument values, live until the `bl`), plus the deepest
        // single argument's own scratch need on top. the callee subtree is
        // folded into the same max -- harmless, since a supported callee is a
        // bare identifier. a single shared argument run is NOT used: it would
        // let a nested call clobber the outer call's saved arguments.
        TypedExpr::Call { callee, args, .. } => {
            let mut deepest = expr_spill_depth(callee);
            for arg in args {
                deepest = deepest.max(expr_spill_depth(arg));
            }
            args.len() as i64 + deepest
        }
        TypedExpr::Unary { operand, .. } => expr_spill_depth(operand),
        TypedExpr::Paren { inner, .. } => expr_spill_depth(inner),
        TypedExpr::Block { block, .. } => max_spill_depth(block),
        TypedExpr::Range { start, end, .. } => {
            let s = start.as_deref().map(expr_spill_depth).unwrap_or(0);
            let e = end.as_deref().map(expr_spill_depth).unwrap_or(0);
            s.max(e)
        }
        // a leaf or an unsupported construct contributes no spill.
        _ => 0,
    }
}

/// whether a call's callee is the `print` or `println` output built-in.
///
/// the `print` / `println` lowering ([`super::print`]) spills one scratch
/// slot per `i64` interpolation hole rather than one per call argument, so the
/// frame planner counts a printf call differently from a user call. a callee
/// is a printf callee when it is a bare identifier named `print` or `println`
/// -- the only callee shape the emitter routes to the printf lowering.
fn is_print_callee(callee: &TypedExpr) -> bool {
    matches!(
        callee,
        TypedExpr::Ident { name, .. } if name == "print" || name == "println"
    )
}

/// the peak scratch-slot count a `print` / `println` call needs.
///
/// the call lowers to a `printf`: each `i64` interpolation hole is evaluated
/// into `x0` and spilled to a claimed scratch slot, and every spilled hole is
/// held until `bl printf`. so `n` holes occupy `n` simultaneously-live slots,
/// and evaluating any single hole can claim further slots on top -- the same
/// `args.len() + max(arg depth)` bound the general `Call` arm uses, but
/// counted over the interpolation's holes instead of the call's arguments. a
/// string-literal argument has no holes and needs no scratch; a non-literal,
/// non-interpolation argument is rejected by the emitter, so a `0` here is
/// safe -- the frame for a program the backend rejects is never emitted.
fn printf_call_spill_depth(args: &[TypedExpr]) -> i64 {
    // a well-typed print / println has exactly one str argument; walk it for
    // the interpolation holes. anything else contributes no scratch.
    let Some(TypedExpr::Interpolation { parts, .. }) = args.first() else {
        return 0;
    };
    let mut hole_count = 0;
    let mut deepest = 0;
    for part in parts {
        if let TypedInterpPart::Expr(expr) = part {
            hole_count += 1;
            deepest = deepest.max(expr_spill_depth(expr));
        }
    }
    hole_count + deepest
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

    #[test]
    fn a_function_with_two_lets_assigns_them_concrete_distinct_slots() {
        // `fn f() { let x = 1 let y = 2 }`: no params, two `let` bindings.
        // binding occurrence 0 (`x`) is at [fp, 16], occurrence 1 (`y`) at
        // [fp, 24] -- distinct, consecutive, just above the saved pair.
        let decl = fn_from_source("fn f() { let x = 1\nlet y = 2 }");
        let layout = FrameLayout::plan_frame(&decl);
        assert_eq!(layout.binding_slot(0), Some(16), "first let -> [fp, 16]");
        assert_eq!(layout.binding_slot(1), Some(24), "second let -> [fp, 24]");
        assert_eq!(layout.binding_slot(2), None, "only two bindings exist");
        // 2 binding slots = 16 bytes locals, no params, no scratch ->
        // alloc = -(16 + 16) & -16 = -32. a concrete value, not an inequality.
        assert_eq!(layout.alloc, -32, "2 lets (16 bytes) -> alloc -32");
    }

    #[test]
    fn let_bindings_are_counted_after_the_parameters() {
        // `fn f(a: i64) { let x = 1 }`: one parameter then one `let`. the
        // parameter takes [fp, 16]; the `let` binding slot starts past it at
        // [fp, 24]. 1 param + 1 let = 2 slots = 16 bytes -> alloc = -32.
        let decl = fn_from_source("fn f(a: i64) { let x = 1 }");
        let layout = FrameLayout::plan_frame(&decl);
        assert_eq!(layout.slot_of("a"), Some(16), "param a -> [fp, 16]");
        assert_eq!(layout.binding_slot(0), Some(24), "the let -> [fp, 24]");
        assert_eq!(layout.alloc, -32, "1 param + 1 let = 16 bytes -> alloc -32");
    }

    #[test]
    fn a_shadowing_let_gets_its_own_distinct_slot() {
        // two `let x` in the same body -- the inner shadows the outer. a
        // name-keyed map would collide; the positional binding list gives each
        // its own slot. occurrence 0 -> [fp, 16], occurrence 1 -> [fp, 24].
        let decl = fn_from_source("fn f() -> i64 { let x = 1\nlet x = 2\nx }");
        let layout = FrameLayout::plan_frame(&decl);
        assert_eq!(layout.binding_slot(0), Some(16), "outer x -> [fp, 16]");
        assert_eq!(layout.binding_slot(1), Some(24), "shadowing x -> [fp, 24]");
        assert_ne!(
            layout.binding_slot(0),
            layout.binding_slot(1),
            "a shadowing let must not reuse the outer slot"
        );
    }

    #[test]
    fn lets_nested_inside_an_if_are_counted_into_the_frame() {
        // a `let` inside an `if` block still needs a distinct slot. the body
        // has one outer `let` then an `if` with a `let` in each arm: three
        // `let`s total. binding occurrences 0/1/2 at [fp, 16]/[fp, 24]/[fp, 32].
        let decl = fn_from_source(
            "fn f(c: bool) { let a = 1\nif c { let b = 2 } else { let d = 3 } }",
        );
        let layout = FrameLayout::plan_frame(&decl);
        assert_eq!(layout.binding_slot(0), Some(24), "outer let a, past param c");
        assert_eq!(layout.binding_slot(1), Some(32), "let b in the then-block");
        assert_eq!(layout.binding_slot(2), Some(40), "let d in the else-block");
        // 1 param + 3 lets = 4 slots = 32 bytes -> alloc = -(16 + 32) & -16 = -48.
        assert_eq!(layout.alloc, -48, "1 param + 3 nested lets -> alloc -48");
    }

    #[test]
    fn a_for_loop_reserves_two_slots_a_variable_and_an_end_bound() {
        // `fn f() { for i in 0..10 { } }`: a `for` reserves two binding slots
        // -- the loop variable `i` and a dedicated `end`-bound slot. they are
        // occurrences 0 and 1 at [fp, 16] and [fp, 24].
        let decl = fn_from_source("fn f() { for i in 0..10 { } }");
        let layout = FrameLayout::plan_frame(&decl);
        assert_eq!(layout.binding_slot(0), Some(16), "for-variable slot");
        assert_eq!(layout.binding_slot(1), Some(24), "for-end-bound slot");
        assert_eq!(layout.binding_slot(2), None, "a bare for is exactly two slots");
        // 2 for slots = 16 bytes, no params, no scratch -> alloc = -32.
        assert_eq!(layout.alloc, -32, "one for loop (16 bytes) -> alloc -32");
    }

    #[test]
    fn a_let_inside_a_for_body_is_counted_after_the_for_slots() {
        // `for` slots come first (var, end), then the body's `let`. the body
        // is `for i in 0..3 { let s = i }`: occurrences 0 (`i`), 1 (`end`),
        // 2 (`s`) at [fp, 16], [fp, 24], [fp, 32].
        let decl = fn_from_source("fn f() { for i in 0..3 { let s = i } }");
        let layout = FrameLayout::plan_frame(&decl);
        assert_eq!(layout.binding_slot(0), Some(16), "for-variable i");
        assert_eq!(layout.binding_slot(1), Some(24), "for-end bound");
        assert_eq!(layout.binding_slot(2), Some(32), "the let inside the body");
        // 3 binding slots = 24 bytes -> alloc = -(16 + 24) & -16 = -48.
        assert_eq!(layout.alloc, -48, "for (2 slots) + nested let -> alloc -48");
    }

    #[test]
    fn a_function_with_no_calls_reserves_no_scratch() {
        // `fn f(a: i64) -> i64 { a }` makes no call and has no binary op -- its
        // trailing value is a bare ident, so the scratch reservation is zero
        // and the frame holds only the one parameter slot.
        let decl = fn_from_source("fn f(a: i64) -> i64 { a }");
        let layout = FrameLayout::plan_frame(&decl);
        assert_eq!(layout.scratch_count(), 0, "a bare ident needs no scratch");
        // 1 param + 0 scratch = 8 bytes -> alloc = -(16 + 8) & -16 = -32.
        assert_eq!(layout.alloc, -32, "1 param, no call -> alloc -32");
    }

    #[test]
    fn a_two_param_one_let_function_with_a_three_arg_call_has_concrete_slots() {
        // a function with 2 params, 1 `let`, and a 3-argument call. named
        // slots: 2 params at [fp, 16]/[fp, 24], the `let` at [fp, 32]. the
        // call spills its three arguments to three claimed scratch slots,
        // immediately past the named slots -- [fp, 40], [fp, 48], [fp, 56].
        // each call argument is a bare ident or literal (depth 0), so the
        // call's scratch need is exactly args.len() = 3. locals =
        // 8 * (3 named + 3 scratch) = 48 -> alloc = -(16 + 48) & -16 = -64.
        let src = "fn add3(a: i64, b: i64, c: i64) -> i64 { a }\n\
                   fn caller(x: i64, y: i64) -> i64 { let r = add3(x, y, 1)\nr }";
        let tokens = crate::lexer::Lexer::tokenize(src).expect("lex failed");
        let ast = crate::parser::Parser::parse(&tokens).expect("parse failed");
        let (typed, terrors, _) = crate::typechecker::check_program(&ast, src);
        assert!(terrors.is_empty(), "typecheck errors: {terrors:?}");
        let caller = typed
            .iter()
            .find_map(|item| match item {
                crate::typed_ast::TypedItem::Fn(d) if d.name == "caller" => Some(d.clone()),
                _ => None,
            })
            .expect("no caller function");
        let layout = FrameLayout::plan_frame(&caller);
        // 2 params + 1 let = 3 named slots at [fp, 16], [fp, 24], [fp, 32].
        assert_eq!(layout.slot_of("x"), Some(16), "param x -> [fp, 16]");
        assert_eq!(layout.slot_of("y"), Some(24), "param y -> [fp, 24]");
        assert_eq!(layout.binding_slot(0), Some(32), "let r -> [fp, 32]");
        // the 3-arg call needs three scratch slots, past the named slots.
        assert_eq!(layout.scratch_count(), 3, "a 3-arg call -> three scratch slots");
        // 3 named + 3 scratch = 48 bytes -> alloc = -64.
        assert_eq!(layout.alloc, -64, "2 params + 1 let + 3-arg call -> alloc -64");
        assert_eq!(layout.prologue()[0], "stp     fp, lr, [sp, -64]!");
        assert_eq!(layout.epilogue()[0], "ldp     fp, lr, [sp], 64");
    }

    #[test]
    fn sibling_calls_in_separate_statements_size_to_the_widest_not_the_sum() {
        // a function with a 1-arg call and a 2-arg call in SEPARATE statements
        // reserves 2 scratch slots -- the widest single call, not 1 + 2. each
        // call claims its argument slots and releases them at its `bl`, so the
        // two calls' slots never overlap and the peak is the larger call.
        let src = "fn one(a: i64) -> i64 { a }\n\
                   fn two(a: i64, b: i64) -> i64 { a }\n\
                   fn f() { let p = one(1)\nlet q = two(2, 3) }";
        let tokens = crate::lexer::Lexer::tokenize(src).expect("lex failed");
        let ast = crate::parser::Parser::parse(&tokens).expect("parse failed");
        let (typed, terrors, _) = crate::typechecker::check_program(&ast, src);
        assert!(terrors.is_empty(), "typecheck errors: {terrors:?}");
        let f = typed
            .iter()
            .find_map(|item| match item {
                crate::typed_ast::TypedItem::Fn(d) if d.name == "f" => Some(d.clone()),
                _ => None,
            })
            .expect("no f function");
        let layout = FrameLayout::plan_frame(&f);
        // the 2-arg call is the widest -> two scratch slots, past the two lets.
        assert_eq!(layout.scratch_count(), 2, "the widest call passes two args");
        // 2 lets + 2 scratch = 32 bytes -> alloc = -(16 + 32) & -16 = -48.
        assert_eq!(layout.alloc, -48, "two lets + a 2-arg call -> alloc -48");
    }

    #[test]
    fn a_nested_call_argument_adds_the_inner_calls_depth_to_the_outer_arity() {
        // `outer(inner(1, 2, 3))`: the outer call passes 1 argument, and that
        // argument is itself a 3-arg call. the outer call's single argument
        // slot is live WHILE `inner(1, 2, 3)` evaluates -- and inner needs
        // three slots of its own -- so the peak is 1 + 3 = 4, NOT 3. the old
        // max-args planner under-counted this at 3; the spill-depth model must
        // see the inner depth stacked on the outer arity.
        let src = "fn inner(a: i64, b: i64, c: i64) -> i64 { a }\n\
                   fn outer(x: i64) -> i64 { x }\n\
                   fn f() { let r = outer(inner(1, 2, 3)) }";
        let tokens = crate::lexer::Lexer::tokenize(src).expect("lex failed");
        let ast = crate::parser::Parser::parse(&tokens).expect("parse failed");
        let (typed, terrors, _) = crate::typechecker::check_program(&ast, src);
        assert!(terrors.is_empty(), "typecheck errors: {terrors:?}");
        let f = typed
            .iter()
            .find_map(|item| match item {
                crate::typed_ast::TypedItem::Fn(d) if d.name == "f" => Some(d.clone()),
                _ => None,
            })
            .expect("no f function");
        let layout = FrameLayout::plan_frame(&f);
        assert_eq!(
            layout.scratch_count(),
            4,
            "outer arity (1) + inner call depth (3) = 4 scratch slots"
        );
        // 1 let + 4 scratch = 40 bytes -> alloc = -(16 + 40) & -16 = -64.
        assert_eq!(layout.alloc, -64, "1 let + a nested 3-arg call -> alloc -64");
    }
}
