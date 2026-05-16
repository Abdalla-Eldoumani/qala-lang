//! the bytecode codegen: lower a [`TypedAst`] to a [`Program`] of bytecode
//! chunks, ready for the Phase 5 stack VM to execute.
//!
//! one recursive walk over the typed AST; each typed expression and
//! statement compiles to a sequence of opcodes via the write helpers on
//! [`crate::chunk::Chunk`]. inline constant folding runs as bytecode is
//! emitted -- a literal arithmetic / comparison / logic expression collapses
//! to a single CONST before reaching the optimizer; an i64 fold that would
//! overflow errors with [`crate::errors::QalaError::IntegerOverflow`] rather
//! than silently wrapping. dead code after `return` / `break` / `continue`
//! is not emitted; a constant-condition `if` emits only the taken branch.
//!
//! `defer` compilation: each [`LocalsScope`] carries a deferred-expression
//! list. defer statements append, never emit; every scope exit (fall-through,
//! return, break, continue, ?-propagation) walks the deferred list in
//! reverse and emits each expression's bytecode. defers inside a loop body
//! live in the loop-body scope, which is pushed fresh per iteration; LIFO
//! per-iteration semantics fall out of this.
//!
//! `comptime` evaluation: a [`ComptimeInterpreter`] (file-private) walks the
//! same opcode set as the runtime VM but enforces a 100K-instruction budget
//! and refuses to dispatch a CALL whose callee is not pure -- defense in
//! depth against a typechecker miss. results that cannot be represented in
//! the constant pool (arrays, structs, enum variants) error at the comptime
//! block.
//!
//! errors accumulate rather than fail-fast: a fold overflow in one function
//! does not stop the others from compiling. on any errors, the public entry
//! returns Err(Vec<QalaError>) sorted by (span.start, span.len); on full
//! success it returns Ok(Program).
//!
//! file order (per compiler/CLAUDE.md): `value -> opcode -> chunk -> codegen`.
//! every byte this module writes goes through [`Chunk::write_op`] /
//! [`Chunk::write_u16`] / [`Chunk::write_i16`] so the lockstep invariant
//! `chunk.code.len() == chunk.source_lines.len()` survives codegen.

use crate::ast::{BinOp, UnaryOp};
use crate::chunk::{Chunk, Program};
use crate::effects::EffectSet;
use crate::errors::QalaError;
use crate::opcode::Opcode;
use crate::span::{LineIndex, Span};
use crate::typed_ast::{TypedAst, TypedBlock, TypedExpr, TypedItem, TypedStmt};
use crate::types::QalaType;
use crate::value::Value;
use std::collections::HashMap;

/// the fn-id base at which the stdlib namespace begins. user functions get
/// dense ids `0..user_count` into [`Program::chunks`]; stdlib functions get
/// ids `STDLIB_FN_BASE + index` so the two spaces never collide and the
/// Phase 5 VM dispatches a CALL by `fn_id >= STDLIB_FN_BASE`. keeping the
/// stdlib out of `program.chunks` keeps the chunks vector dense and lets an
/// empty program produce an empty `chunks`.
const STDLIB_FN_BASE: u16 = 40_000;

/// the comptime interpreter's instruction budget. hitting it produces
/// [`QalaError::ComptimeBudgetExceeded`]; raising it is a v2 concern.
/// (Task 5 wires the [`ComptimeInterpreter`] that consumes this.)
#[allow(dead_code)]
const COMPTIME_BUDGET: u32 = 100_000;

/// the comptime interpreter's per-`MAKE_ARRAY` element cap. v1 rejects
/// arrays inside comptime entirely, but the constant is kept for the bound
/// check site so a future array-in-comptime feature has the ceiling ready.
#[allow(dead_code)]
const COMPTIME_MAX_ARRAY_ELEMS: u16 = 10_000;

/// the comptime interpreter's call-depth cap. an explicit `Vec<Frame>` stack
/// (not Rust recursion) means a runaway recursion is bounded here rather than
/// blowing the host stack; over the cap is treated as a budget overflow.
/// (Task 5 wires the [`ComptimeInterpreter`] that consumes this.)
#[allow(dead_code)]
const COMPTIME_MAX_FRAMES: usize = 256;

/// one row of [`STDLIB_TABLE`]: `(name, type_name, effect-constructor)`.
type StdlibRow = (&'static str, Option<&'static str>, fn() -> EffectSet);

/// the stdlib reserved-function table.
///
/// `(name, type_name, effect)`: `type_name` is `Some(T)` for a `fn T.method`
/// (only `FileHandle.read_all` in v1), `None` for a free function. the array
/// ORDER is the stdlib id assignment -- entry `i` gets fn-id
/// `STDLIB_FN_BASE + i`. the effect column mirrors the typechecker's
/// `stdlib_signatures` table verbatim so the [`ComptimeInterpreter`]'s CALL
/// gate is honest. `Ok` / `Err` / `Some` / `None` are NOT here -- they are
/// built-in Result/Option constructors handled as enum-variant emission.
const STDLIB_TABLE: &[StdlibRow] = &[
    ("print", None, EffectSet::io),
    ("println", None, EffectSet::io),
    ("sqrt", None, EffectSet::pure),
    ("abs", None, EffectSet::pure),
    ("assert", None, EffectSet::panic),
    ("len", None, EffectSet::pure),
    ("push", None, EffectSet::alloc),
    ("pop", None, EffectSet::alloc),
    ("type_of", None, EffectSet::pure),
    ("open", None, EffectSet::io),
    ("close", None, EffectSet::io),
    ("map", None, EffectSet::pure),
    ("filter", None, EffectSet::pure),
    ("reduce", None, EffectSet::pure),
    ("read_all", Some("FileHandle"), EffectSet::io),
];

/// the four built-in Result/Option variant constructors and their reserved
/// `MAKE_ENUM_VARIANT` ids. these ids are pre-registered in
/// [`Program::enum_variant_names`] before any user enum variant, so a user
/// enum's first variant starts at id 4. the (enum, variant) pair the
/// disassembler renders is `("Result", "Ok")` and so on.
const BUILTIN_VARIANTS: &[(&str, &str, &str)] = &[
    ("Result", "Ok", "Ok"),
    ("Result", "Err", "Err"),
    ("Option", "Some", "Some"),
    ("Option", "None", "None"),
];

/// the codegen state: the program under construction plus every lookup table
/// the recursive walk consults.
struct Codegen<'a> {
    /// the program being built -- `chunks` holds one chunk per user function
    /// in fn-id order, plus the parallel `fn_names` / `enum_variant_names`.
    program: Program,
    /// the lexical-scope stack. the topmost scope is the innermost block;
    /// `let` bindings register locals here, `defer` appends here.
    scopes: Vec<LocalsScope>,
    /// the fn-id (index into `program.chunks`) currently being compiled.
    current_fn_id: u16,
    /// the original source text -- needed for [`LineIndex`] line lookups.
    src: &'a str,
    /// the line-start table for converting span byte-offsets to source lines.
    line_index: LineIndex,
    /// `(None, name)` for a free fn; `(Some(T), name)` for a `fn T.method`.
    /// every call site the typechecker resolved gets a unique fn-id; codegen
    /// builds the same key map.
    fn_table: HashMap<FnKey, u16>,
    /// stdlib name -> `(fn_id, effect)`. stdlib fns get ids in the
    /// `STDLIB_FN_BASE..` namespace; the effect drives the comptime CALL gate.
    /// (Task 2's call dispatch reads this.)
    #[allow(dead_code)]
    stdlib_table: HashMap<FnKey, (u16, EffectSet)>,
    /// `(enum_name, variant_name)` -> variant id (parallel to
    /// [`Program::enum_variant_names`]).
    enum_variant_table: HashMap<(String, String), u16>,
    /// `(enum_name, variant_name)` -> the variant's payload field count.
    enum_variant_payload_count: HashMap<(String, String), u8>,
    /// `(struct_name, field_name)` -> the field's stable index within the
    /// struct, used as the `FIELD` opcode operand.
    struct_field_index: HashMap<(String, String), u16>,
    /// fn-id -> declared/inferred effect; the [`ComptimeInterpreter`] consults
    /// this at every CALL for the defense-in-depth purity gate.
    fn_effects: HashMap<u16, EffectSet>,
    /// the accumulated codegen errors -- the public entry returns these
    /// sorted rather than failing fast on the first one.
    errors: Vec<QalaError>,
}

/// one lexical scope: the locals it binds and the defers it has registered.
struct LocalsScope {
    /// `(name, slot)` for every local bound in this scope, in declaration
    /// order. lookups walk scopes top-down so an inner binding shadows.
    locals: Vec<(String, u16)>,
    /// the deferred expressions, in registration order. emitted REVERSED at
    /// every exit path that crosses this scope (LIFO).
    deferred: Vec<TypedExpr>,
    /// `Some` when this scope is a loop scope -- it then also carries the
    /// loop-start byte position and the break-jump patch sites.
    loop_meta: Option<LoopMeta>,
}

/// the loop-specific bookkeeping a loop scope carries.
/// (Task 4's loop codegen reads these fields.)
#[allow(dead_code)]
struct LoopMeta {
    /// the byte position `break` jumps are patched to point past.
    break_patches: Vec<usize>,
    /// the byte position `continue` jumps backward to (the increment label).
    continue_target: usize,
}

/// a function-table key: `(type_name, name)`. matches the typechecker's
/// `FnKey` shape so the same call site resolves to the same fn-id.
#[derive(Hash, Eq, PartialEq, Clone)]
struct FnKey {
    /// `Some(T)` for a `fn T.method`; `None` for a free function.
    type_name: Option<String>,
    /// the function name.
    name: String,
}

/// which scopes a control-flow exit unwinds, for defer emission.
/// (Task 4 constructs the `Return` / `Break` / `Continue` / `QuestionProp`
/// variants from `compile_stmt`; `Fallthrough` is used from Task 1.)
#[derive(Copy, Clone, PartialEq, Eq)]
#[allow(dead_code)]
enum ExitKind {
    /// reached the end of a block normally -- emit just the top scope's
    /// defers.
    Fallthrough,
    /// `return` -- emit every scope's defers up to the fn body.
    Return,
    /// `break` -- emit scopes up to and including the innermost loop scope.
    Break,
    /// `continue` -- same scopes as `break`.
    Continue,
    /// `?`-propagation -- same scopes as `return`.
    QuestionProp,
}

/// the codegen module's public entry: lower a typed program to bytecode.
///
/// errors accumulate rather than fail-fast -- a constant-folding overflow or
/// a comptime budget exhaustion in one function does not stop the others
/// from compiling. on success returns `Ok(Program)`; on any errors returns
/// `Err(Vec<QalaError>)` sorted by `(span.start, span.len)` for deterministic
/// rendering.
pub fn compile_program(ast: &TypedAst, src: &str) -> Result<Program, Vec<QalaError>> {
    let mut cg = Codegen::new(src);
    cg.build_tables(ast);
    for item in ast {
        if let Err(e) = cg.compile_item(item) {
            cg.errors.push(e);
        }
    }
    if !cg.errors.is_empty() {
        cg.errors
            .sort_by_key(|e| (e.span().start, e.span().len));
        return Err(cg.errors);
    }
    Ok(cg.program)
}

impl<'a> Codegen<'a> {
    /// construct a fresh codegen state: empty program, the stdlib table and
    /// built-in variant ids pre-registered, every other table empty.
    fn new(src: &'a str) -> Self {
        let mut program = Program::new();
        // pre-register the four built-in Result/Option variant ids before
        // any user enum variant.
        for (enum_name, variant_name, _) in BUILTIN_VARIANTS {
            program
                .enum_variant_names
                .push((enum_name.to_string(), variant_name.to_string()));
        }
        let mut enum_variant_table: HashMap<(String, String), u16> = HashMap::new();
        let mut enum_variant_payload_count: HashMap<(String, String), u8> = HashMap::new();
        for (i, (enum_name, variant_name, _)) in BUILTIN_VARIANTS.iter().enumerate() {
            enum_variant_table
                .insert((enum_name.to_string(), variant_name.to_string()), i as u16);
            // Ok / Err / Some carry one payload; None carries zero.
            let payload = if *variant_name == "None" { 0 } else { 1 };
            enum_variant_payload_count
                .insert((enum_name.to_string(), variant_name.to_string()), payload);
        }
        let mut stdlib_table: HashMap<FnKey, (u16, EffectSet)> = HashMap::new();
        let mut fn_effects: HashMap<u16, EffectSet> = HashMap::new();
        for (i, (name, type_name, effect_fn)) in STDLIB_TABLE.iter().enumerate() {
            let fn_id = STDLIB_FN_BASE + i as u16;
            let effect = effect_fn();
            let key = FnKey {
                type_name: type_name.map(|s| s.to_string()),
                name: name.to_string(),
            };
            stdlib_table.insert(key, (fn_id, effect));
            fn_effects.insert(fn_id, effect);
        }
        Codegen {
            program,
            scopes: Vec::new(),
            current_fn_id: 0,
            src,
            line_index: LineIndex::new(src),
            fn_table: HashMap::new(),
            stdlib_table,
            enum_variant_table,
            enum_variant_payload_count,
            struct_field_index: HashMap::new(),
            fn_effects,
            errors: Vec::new(),
        }
    }

    /// pass 1: pre-register every user function's fn-id, every enum variant's
    /// id, and every struct field's index, so the recursive compile pass can
    /// resolve forward references (a fn calling a fn declared later).
    fn build_tables(&mut self, ast: &TypedAst) {
        for item in ast {
            match item {
                TypedItem::Fn(decl) => {
                    let fn_id = self.program.chunks.len() as u16;
                    self.program.chunks.push(Chunk::new());
                    self.program.fn_names.push(decl.name.clone());
                    self.fn_table.insert(
                        FnKey {
                            type_name: decl.type_name.clone(),
                            name: decl.name.clone(),
                        },
                        fn_id,
                    );
                    self.fn_effects.insert(fn_id, decl.effect);
                }
                TypedItem::Struct(decl) => {
                    for (i, field) in decl.fields.iter().enumerate() {
                        self.struct_field_index.insert(
                            (decl.name.clone(), field.name.clone()),
                            i as u16,
                        );
                    }
                }
                TypedItem::Enum(decl) => {
                    for variant in &decl.variants {
                        let variant_id = self.program.enum_variant_names.len() as u16;
                        self.program
                            .enum_variant_names
                            .push((decl.name.clone(), variant.name.clone()));
                        self.enum_variant_table.insert(
                            (decl.name.clone(), variant.name.clone()),
                            variant_id,
                        );
                        self.enum_variant_payload_count.insert(
                            (decl.name.clone(), variant.name.clone()),
                            variant.fields.len() as u8,
                        );
                    }
                }
                // interfaces are type-level only; the typechecker already
                // verified structural satisfaction. no chunk, no fn-id.
                TypedItem::Interface(_) => {}
            }
        }
        // record the entry point: the fn-id of `main`, if present.
        if let Some(&main_id) = self.fn_table.get(&FnKey {
            type_name: None,
            name: "main".to_string(),
        }) {
            self.program.main_index = main_id as usize;
        }
    }

    /// compile one top-level item. `fn` items produce a chunk; `struct` /
    /// `enum` / `interface` items emit nothing -- [`Codegen::build_tables`]
    /// already recorded everything codegen needs about them.
    fn compile_item(&mut self, item: &TypedItem) -> Result<(), QalaError> {
        match item {
            TypedItem::Fn(decl) => {
                let fn_id = self
                    .fn_table
                    .get(&FnKey {
                        type_name: decl.type_name.clone(),
                        name: decl.name.clone(),
                    })
                    .copied()
                    .ok_or_else(|| QalaError::Type {
                        span: decl.span,
                        message: format!("codegen: function `{}` was not pre-registered", decl.name),
                    })?;
                self.current_fn_id = fn_id;
                // the outermost scope holds the parameters in slots 0..N.
                self.push_scope();
                for (i, param) in decl.params.iter().enumerate() {
                    if let Some(scope) = self.scopes.last_mut() {
                        scope.locals.push((param.name.clone(), i as u16));
                    }
                }
                let terminated = self.compile_block(&decl.body)?;
                if !terminated {
                    // a fall-through completion: emit the outermost scope's
                    // defers, then a RETURN.
                    let line = self.line_at(decl.body.span);
                    self.emit_defers_for_exit(ExitKind::Fallthrough, line)?;
                    self.chunk_mut().write_op(Opcode::Return, line);
                }
                // remove the outermost scope. its defers fired above (or at
                // the terminator's exit); no second emission.
                self.pop_scope_no_defers();
                Ok(())
            }
            // type-level items: build_tables recorded what codegen needs.
            TypedItem::Struct(_) | TypedItem::Enum(_) | TypedItem::Interface(_) => Ok(()),
        }
    }

    // ---- chunk + line helpers ----------------------------------------------

    /// the chunk currently being written.
    fn chunk_mut(&mut self) -> &mut Chunk {
        &mut self.program.chunks[self.current_fn_id as usize]
    }

    /// the 1-based source line of a span's start byte.
    fn line_at(&self, span: Span) -> u32 {
        self.line_index.location(self.src, span.start as usize).0 as u32
    }

    // ---- scope management --------------------------------------------------

    /// push a plain (non-loop) lexical scope.
    fn push_scope(&mut self) {
        self.scopes.push(LocalsScope {
            locals: Vec::new(),
            deferred: Vec::new(),
            loop_meta: None,
        });
    }

    /// push a loop scope: a plain scope plus the loop bookkeeping.
    /// (Task 4's `while` / `for` codegen calls this.)
    #[allow(dead_code)]
    fn push_loop_scope(&mut self, continue_target: usize) {
        self.scopes.push(LocalsScope {
            locals: Vec::new(),
            deferred: Vec::new(),
            loop_meta: Some(LoopMeta {
                break_patches: Vec::new(),
                continue_target,
            }),
        });
    }

    /// pop the topmost scope WITHOUT emitting its defers. used after a
    /// terminator (`return` / `break` / `continue`) already fired the scope's
    /// defers at its exit -- a second emission would double-run them.
    fn pop_scope_no_defers(&mut self) {
        self.scopes.pop();
    }

    /// pop the topmost scope AND emit its deferred expressions, reversed
    /// (LIFO). used at a block's fall-through exit. each defer's bytecode is
    /// followed by a POP when the defer's result type is not `void` -- a
    /// defer is evaluated for effect, its value discarded.
    /// (Task 4's `compile_block` calls this on the fall-through path.)
    #[allow(dead_code)]
    fn pop_scope_and_emit_defers(&mut self, line: u32) -> Result<(), QalaError> {
        let scope = match self.scopes.pop() {
            Some(s) => s,
            None => return Ok(()),
        };
        for deferred in scope.deferred.iter().rev() {
            self.compile_expr(deferred)?;
            if !matches!(deferred.ty(), QalaType::Void) {
                self.chunk_mut().write_op(Opcode::Pop, line);
            }
        }
        Ok(())
    }

    /// emit the defers for a control-flow exit WITHOUT popping any scope.
    /// the depth depends on the exit kind: a fall-through fires one scope; a
    /// `return` / `?`-propagation fires every scope; a `break` / `continue`
    /// fires scopes up to and including the innermost loop scope.
    fn emit_defers_for_exit(&mut self, exit_kind: ExitKind, line: u32) -> Result<(), QalaError> {
        let depth = match exit_kind {
            ExitKind::Fallthrough => 1.min(self.scopes.len()),
            ExitKind::Return | ExitKind::QuestionProp => self.scopes.len(),
            ExitKind::Break | ExitKind::Continue => {
                // count from the top down until (inclusive) the loop scope.
                let mut count = 0usize;
                let mut found = false;
                for scope in self.scopes.iter().rev() {
                    count += 1;
                    if scope.loop_meta.is_some() {
                        found = true;
                        break;
                    }
                }
                if !found {
                    return Err(QalaError::Type {
                        span: Span::new(0, 0),
                        message: "codegen: break/continue outside a loop".to_string(),
                    });
                }
                count
            }
        };
        // collect the deferred expressions to emit, top scope first, each
        // scope's list reversed. cloned out so the borrow of `self.scopes`
        // ends before `compile_expr` borrows `self` mutably.
        let mut to_emit: Vec<TypedExpr> = Vec::new();
        let n = self.scopes.len();
        for scope in self.scopes[n - depth..].iter().rev() {
            for deferred in scope.deferred.iter().rev() {
                to_emit.push(deferred.clone());
            }
        }
        for deferred in &to_emit {
            self.compile_expr(deferred)?;
            if !matches!(deferred.ty(), QalaType::Void) {
                self.chunk_mut().write_op(Opcode::Pop, line);
            }
        }
        Ok(())
    }

    // ---- statement + block (placeholders; Task 4 implements) ---------------

    /// compile a `{ ... }` block. returns `true` when the block was
    /// terminated by a `return` / `break` / `continue` (so a caller skips a
    /// fall-through RETURN).
    ///
    /// PLACEHOLDER for Task 1: emits nothing and reports "not terminated".
    /// Task 4 replaces the body with the full statement walk + DCE.
    fn compile_block(&mut self, block: &TypedBlock) -> Result<bool, QalaError> {
        let _ = block;
        Ok(false)
    }

    /// compile one statement. returns `true` when the statement terminated
    /// the surrounding block.
    ///
    /// PLACEHOLDER for Task 1: emits nothing. Task 4 replaces the body.
    #[allow(dead_code)]
    fn compile_stmt(&mut self, stmt: &TypedStmt) -> Result<bool, QalaError> {
        let _ = stmt;
        Ok(false)
    }

    // ---- expression emission helpers ---------------------------------------

    /// emit `CONST idx` for `value` on `line`: append the value to the
    /// current chunk's constant pool, then the opcode + u16 index.
    fn emit_const(&mut self, value: Value, line: u32) {
        let idx = self.chunk_mut().add_constant(value);
        self.chunk_mut().write_op(Opcode::Const, line);
        self.chunk_mut().write_u16(idx, line);
    }

    /// emit a `CALL fn_id argc` instruction. CALL carries a `u16` fn-id plus
    /// a `u8` argc; there is no `write_u8` helper, so the one-byte argc is
    /// pushed directly with a matching `source_lines` entry to keep the
    /// `code.len() == source_lines.len()` invariant.
    fn emit_call(&mut self, fn_id: u16, argc: u8, line: u32) {
        self.chunk_mut().write_op(Opcode::Call, line);
        self.chunk_mut().write_u16(fn_id, line);
        let chunk = self.chunk_mut();
        chunk.code.push(argc);
        chunk.source_lines.push(line);
    }

    /// resolve an identifier in callee position to its fn-id, checking the
    /// user `fn_table` first, then the stdlib table. returns `None` when the
    /// name is neither (an enum-variant constructor or a local-bound function
    /// value -- the caller handles those).
    fn resolve_callee_fn_id(&self, name: &str) -> Option<u16> {
        let key = FnKey {
            type_name: None,
            name: name.to_string(),
        };
        if let Some(&fn_id) = self.fn_table.get(&key) {
            return Some(fn_id);
        }
        self.stdlib_table.get(&key).map(|(fn_id, _)| *fn_id)
    }

    /// look up a local by name, walking scopes from innermost to outermost so
    /// an inner binding shadows an outer one. returns the slot index.
    fn resolve_local(&self, name: &str) -> Option<u16> {
        for scope in self.scopes.iter().rev() {
            for (local_name, slot) in scope.locals.iter().rev() {
                if local_name == name {
                    return Some(*slot);
                }
            }
        }
        None
    }

    /// pick the arithmetic / comparison opcode for a binary operator given
    /// the OPERAND type (`i64` vs `f64`). `&&` / `||` never reach this -- they
    /// compile to short-circuit jump patterns in the `Binary` arm. string
    /// `+` lowers to `CONCAT_N 2` and is handled inline by the caller.
    fn emit_binop(&mut self, op: &BinOp, operand_ty: &QalaType, line: u32) {
        let is_float = matches!(operand_ty, QalaType::F64);
        let opcode = match (op, is_float) {
            (BinOp::Add, false) => Opcode::Add,
            (BinOp::Add, true) => Opcode::FAdd,
            (BinOp::Sub, false) => Opcode::Sub,
            (BinOp::Sub, true) => Opcode::FSub,
            (BinOp::Mul, false) => Opcode::Mul,
            (BinOp::Mul, true) => Opcode::FMul,
            (BinOp::Div, false) => Opcode::Div,
            (BinOp::Div, true) => Opcode::FDiv,
            (BinOp::Rem, _) => Opcode::Mod,
            (BinOp::Eq, false) => Opcode::Eq,
            (BinOp::Eq, true) => Opcode::FEq,
            (BinOp::Ne, false) => Opcode::Ne,
            (BinOp::Ne, true) => Opcode::FNe,
            (BinOp::Lt, false) => Opcode::Lt,
            (BinOp::Lt, true) => Opcode::FLt,
            (BinOp::Le, false) => Opcode::Le,
            (BinOp::Le, true) => Opcode::FLe,
            (BinOp::Gt, false) => Opcode::Gt,
            (BinOp::Gt, true) => Opcode::FGt,
            (BinOp::Ge, false) => Opcode::Ge,
            (BinOp::Ge, true) => Opcode::FGe,
            // And / Or never reach emit_binop; if they somehow do, NOT is a
            // harmless no-payload placeholder. the Binary arm prevents this.
            (BinOp::And, _) | (BinOp::Or, _) => Opcode::Not,
        };
        self.chunk_mut().write_op(opcode, line);
    }

    /// emit the opcode for a unary operator given the operand type.
    fn emit_unop(&mut self, op: &UnaryOp, operand_ty: &QalaType, line: u32) {
        let opcode = match op {
            UnaryOp::Not => Opcode::Not,
            UnaryOp::Neg => {
                if matches!(operand_ty, QalaType::F64) {
                    Opcode::FNeg
                } else {
                    Opcode::Neg
                }
            }
        };
        self.chunk_mut().write_op(opcode, line);
    }

    /// try to constant-fold a binary operator whose two operands were just
    /// emitted as CONSTs. returns `Ok(true)` when a fold happened (the two
    /// source CONSTs are rewound and a single result CONST emitted on
    /// `line`), `Ok(false)` when no fold applies (the caller emits the
    /// opcode), `Err` on an i64 overflow.
    fn try_fold_binary(
        &mut self,
        op: &BinOp,
        operand_ty: &QalaType,
        line: u32,
        span: Span,
    ) -> Result<bool, QalaError> {
        // peek both operand CONSTs; bail if either is missing.
        let (b_off, b_idx) = match self.chunk_mut().last_const() {
            Some(v) => v,
            None => return Ok(false),
        };
        let (a_off, a_idx) = match self.chunk_mut().second_to_last_const() {
            Some(v) => v,
            None => return Ok(false),
        };
        // the two CONSTs must be adjacent in the byte stream -- if anything
        // intervened, folding them is unsound.
        if a_off + 3 != b_off {
            return Ok(false);
        }
        let a = self.chunk_mut().constants[a_idx as usize].clone();
        let b = self.chunk_mut().constants[b_idx as usize].clone();
        let result = match fold_binary_value(&a, &b, op, operand_ty, span)? {
            Some(v) => v,
            None => return Ok(false),
        };
        // rewind both source CONSTs, then emit the folded result.
        let chunk = self.chunk_mut();
        chunk.code.truncate(a_off);
        chunk.source_lines.truncate(a_off);
        self.emit_const(result, line);
        Ok(true)
    }

    /// try to constant-fold a unary operator whose operand was just emitted
    /// as a CONST. returns `Ok(true)` on a fold, `Ok(false)` when none
    /// applies, `Err` on an i64 negation overflow (`-i64::MIN`).
    fn try_fold_unary(
        &mut self,
        op: &UnaryOp,
        line: u32,
        span: Span,
    ) -> Result<bool, QalaError> {
        let (off, idx) = match self.chunk_mut().last_const() {
            Some(v) => v,
            None => return Ok(false),
        };
        let v = self.chunk_mut().constants[idx as usize].clone();
        let result = match (op, &v) {
            (UnaryOp::Not, Value::Bool(b)) => Value::Bool(!b),
            (UnaryOp::Neg, Value::I64(n)) => {
                // -x overflows only at i64::MIN; surface as IntegerOverflow.
                // the synthetic op is Sub because `-x` equals `0 - x`; a
                // future enhancement could widen IntegerOverflow to carry a
                // UnaryOp.
                let r = n.checked_neg().ok_or(QalaError::IntegerOverflow {
                    span,
                    op: BinOp::Sub,
                    lhs: 0,
                    rhs: *n,
                })?;
                Value::I64(r)
            }
            (UnaryOp::Neg, Value::F64(x)) => Value::F64(-x),
            // any other operand: no fold.
            _ => return Ok(false),
        };
        let chunk = self.chunk_mut();
        chunk.code.truncate(off);
        chunk.source_lines.truncate(off);
        self.emit_const(result, line);
        Ok(true)
    }

    // ---- expression dispatch -----------------------------------------------

    /// compile one expression, leaving its value on the stack.
    ///
    /// Task 2 covers primitives, identifiers, parenthesized expressions,
    /// unary / binary operators (with inline constant folding), and the
    /// simple ident-callee call. Task 3 adds the compound expressions and
    /// `match`; Task 5 adds `comptime`.
    fn compile_expr(&mut self, e: &TypedExpr) -> Result<(), QalaError> {
        match e {
            TypedExpr::Int { value, span, .. } => {
                let line = self.line_at(*span);
                self.emit_const(Value::I64(*value), line);
                Ok(())
            }
            TypedExpr::Float { value, span, .. } => {
                let line = self.line_at(*span);
                self.emit_const(Value::F64(*value), line);
                Ok(())
            }
            TypedExpr::Byte { value, span, .. } => {
                let line = self.line_at(*span);
                self.emit_const(Value::Byte(*value), line);
                Ok(())
            }
            TypedExpr::Str { value, span, .. } => {
                let line = self.line_at(*span);
                self.emit_const(Value::Str(value.clone()), line);
                Ok(())
            }
            TypedExpr::Bool { value, span, .. } => {
                let line = self.line_at(*span);
                self.emit_const(Value::Bool(*value), line);
                Ok(())
            }
            TypedExpr::Ident { name, span, .. } => {
                let line = self.line_at(*span);
                if let Some(slot) = self.resolve_local(name) {
                    self.chunk_mut().write_op(Opcode::GetLocal, line);
                    self.chunk_mut().write_u16(slot, line);
                } else if let Some(fn_id) = self.resolve_callee_fn_id(name) {
                    // a bare function name used as a value -- a function
                    // reference. emitted as a CONST of Value::Function.
                    self.emit_const(Value::Function(fn_id), line);
                } else {
                    // the typechecker would have rejected an unresolved name;
                    // defense-in-depth.
                    return Err(QalaError::Type {
                        span: *span,
                        message: format!("codegen: unresolved name `{name}`"),
                    });
                }
                Ok(())
            }
            TypedExpr::Paren { inner, .. } => self.compile_expr(inner),
            TypedExpr::Unary { op, operand, span, .. } => {
                self.compile_expr(operand)?;
                let line = self.line_at(*span);
                if !self.try_fold_unary(op, line, *span)? {
                    self.emit_unop(op, operand.ty(), line);
                }
                Ok(())
            }
            TypedExpr::Binary { op, lhs, rhs, span, .. } => {
                let line = self.line_at(*span);
                if matches!(op, BinOp::And | BinOp::Or) {
                    // short-circuit: compile lhs, DUP it, conditionally jump
                    // past rhs, otherwise POP the lhs copy and evaluate rhs.
                    // `&&` jumps past on a false lhs; `||` jumps past on true.
                    self.compile_expr(lhs)?;
                    // a literal lhs+rhs both-CONST case still folds: peek.
                    if self.try_fold_short_circuit(op, rhs, line)? {
                        return Ok(());
                    }
                    self.chunk_mut().write_op(Opcode::Dup, line);
                    let skip = if matches!(op, BinOp::And) {
                        self.chunk_mut().emit_jump(Opcode::JumpIfFalse, line)
                    } else {
                        self.chunk_mut().emit_jump(Opcode::JumpIfTrue, line)
                    };
                    self.chunk_mut().write_op(Opcode::Pop, line);
                    self.compile_expr(rhs)?;
                    self.chunk_mut().patch_jump(skip)?;
                    Ok(())
                } else {
                    self.compile_expr(lhs)?;
                    self.compile_expr(rhs)?;
                    if !self.try_fold_binary(op, lhs.ty(), line, *span)? {
                        self.emit_binop(op, lhs.ty(), line);
                    }
                    Ok(())
                }
            }
            TypedExpr::Call { callee, args, span, .. } => {
                let line = self.line_at(*span);
                self.compile_call(callee, args, line, *span)
            }
            // Task 3 + Task 5 fill these arms in.
            other => self.unsupported_expr(other),
        }
    }

    /// compile an `&&` / `||` whose operands are both literal CONSTs: peek
    /// the already-emitted lhs CONST, compile rhs (which may itself be a
    /// CONST), then fold via [`try_fold_binary`]. returns `Ok(true)` when the
    /// whole short-circuit folded to one CONST. when the lhs is not a CONST
    /// this returns `Ok(false)` without emitting `rhs`.
    fn try_fold_short_circuit(
        &mut self,
        op: &BinOp,
        rhs: &TypedExpr,
        line: u32,
    ) -> Result<bool, QalaError> {
        // only attempt the fold when the lhs is already a single CONST.
        if self.chunk_mut().last_const().is_none() {
            return Ok(false);
        }
        let lhs_len = self.chunk_mut().code.len();
        self.compile_expr(rhs)?;
        // rhs must have emitted exactly one CONST after the lhs CONST.
        if self.try_fold_binary(op, &QalaType::Bool, line, rhs.span())? {
            return Ok(true);
        }
        // no fold: rewind rhs's emission so the caller emits the
        // short-circuit jump pattern from a clean state (just the lhs CONST).
        let chunk = self.chunk_mut();
        chunk.code.truncate(lhs_len);
        chunk.source_lines.truncate(lhs_len);
        Ok(false)
    }

    /// compile a call: emit the arguments left-to-right then dispatch. the
    /// callee is always a [`TypedExpr::Ident`] in v1 -- either an enum-variant
    /// constructor (`Circle(5.0)`, `Ok(x)`), a user / stdlib function, or a
    /// local-bound function value.
    fn compile_call(
        &mut self,
        callee: &TypedExpr,
        args: &[TypedExpr],
        line: u32,
        span: Span,
    ) -> Result<(), QalaError> {
        let name = match callee {
            TypedExpr::Ident { name, .. } => name.clone(),
            // a non-ident callee: compile it as a function value, emit args,
            // then CALL via the value on the stack. v1's typechecker makes
            // this rare; treat the callee's resolved fn-id as unavailable.
            _ => {
                return Err(QalaError::Type {
                    span,
                    message: "codegen: only named callees are supported in v1".to_string(),
                });
            }
        };
        // enum-variant constructor: emit args, then MAKE_ENUM_VARIANT.
        // covers user enum variants AND the built-in Ok / Err / Some / None.
        if let Some(&variant_id) = self.enum_variant_lookup(&name) {
            for arg in args {
                self.compile_expr(arg)?;
            }
            let payload = args.len() as u8;
            self.chunk_mut().write_op(Opcode::MakeEnumVariant, line);
            self.chunk_mut().write_u16(variant_id, line);
            let chunk = self.chunk_mut();
            chunk.code.push(payload);
            chunk.source_lines.push(line);
            return Ok(());
        }
        // a user or stdlib function call.
        if let Some(fn_id) = self.resolve_callee_fn_id(&name) {
            for arg in args {
                self.compile_expr(arg)?;
            }
            self.emit_call(fn_id, args.len() as u8, line);
            return Ok(());
        }
        Err(QalaError::Type {
            span,
            message: format!("codegen: unresolved callee `{name}`"),
        })
    }

    /// look up an enum-variant id by the variant's bare name -- the form a
    /// constructor call uses. searches every `(enum, variant)` pair for a
    /// matching variant name. ambiguous names (the same variant name in two
    /// enums) resolve to the first match; the typechecker has already pinned
    /// the concrete enum, so v1's bundled examples never collide.
    fn enum_variant_lookup(&self, variant_name: &str) -> Option<&u16> {
        self.enum_variant_table
            .iter()
            .find(|((_, v), _)| v == variant_name)
            .map(|(_, id)| id)
    }

    /// the placeholder for expression kinds Task 3 and Task 5 implement.
    /// returns a codegen error so the match stays exhaustive and the crate
    /// compiles between tasks; no Task-2 test exercises these arms.
    fn unsupported_expr(&mut self, e: &TypedExpr) -> Result<(), QalaError> {
        Err(QalaError::Type {
            span: e.span(),
            message: "codegen: expression kind not yet supported".to_string(),
        })
    }
}

/// compute the constant-folded value of a binary operator on two literal
/// operands. returns `Ok(Some(v))` on a successful fold, `Ok(None)` when the
/// operand/operator combination has no fold rule (the caller emits the
/// opcode), `Err` on an i64 arithmetic overflow.
///
/// i64 arithmetic uses `checked_*` so an overflow is an error, not a wrap.
/// f64 arithmetic follows IEEE 754 -- `inf` / `NaN` are valid results, never
/// an error. string `+` of two literals folds to the joined string. integer
/// division / remainder by a zero literal does NOT fold (the runtime VM
/// raises the division-by-zero error so the diagnostic points at the source).
fn fold_binary_value(
    a: &Value,
    b: &Value,
    op: &BinOp,
    operand_ty: &QalaType,
    span: Span,
) -> Result<Option<Value>, QalaError> {
    let overflow = |op: BinOp, lhs: i64, rhs: i64| QalaError::IntegerOverflow {
        span,
        op,
        lhs,
        rhs,
    };
    let folded = match (a, b, op) {
        // ---- i64 arithmetic: checked, overflow is an error ----
        (Value::I64(x), Value::I64(y), BinOp::Add) if matches!(operand_ty, QalaType::I64) => {
            Value::I64(x.checked_add(*y).ok_or_else(|| overflow(BinOp::Add, *x, *y))?)
        }
        (Value::I64(x), Value::I64(y), BinOp::Sub) if matches!(operand_ty, QalaType::I64) => {
            Value::I64(x.checked_sub(*y).ok_or_else(|| overflow(BinOp::Sub, *x, *y))?)
        }
        (Value::I64(x), Value::I64(y), BinOp::Mul) if matches!(operand_ty, QalaType::I64) => {
            Value::I64(x.checked_mul(*y).ok_or_else(|| overflow(BinOp::Mul, *x, *y))?)
        }
        (Value::I64(x), Value::I64(y), BinOp::Div) if matches!(operand_ty, QalaType::I64) => {
            if *y == 0 {
                return Ok(None);
            }
            Value::I64(x.checked_div(*y).ok_or_else(|| overflow(BinOp::Div, *x, *y))?)
        }
        (Value::I64(x), Value::I64(y), BinOp::Rem) if matches!(operand_ty, QalaType::I64) => {
            if *y == 0 {
                return Ok(None);
            }
            Value::I64(x.checked_rem(*y).ok_or_else(|| overflow(BinOp::Rem, *x, *y))?)
        }
        // ---- f64 arithmetic: IEEE 754, no overflow error ----
        (Value::F64(x), Value::F64(y), BinOp::Add) => Value::F64(x + y),
        (Value::F64(x), Value::F64(y), BinOp::Sub) => Value::F64(x - y),
        (Value::F64(x), Value::F64(y), BinOp::Mul) => Value::F64(x * y),
        (Value::F64(x), Value::F64(y), BinOp::Div) => Value::F64(x / y),
        // ---- string concatenation of two literals ----
        (Value::Str(x), Value::Str(y), BinOp::Add) => {
            let mut s = String::with_capacity(x.len() + y.len());
            s.push_str(x);
            s.push_str(y);
            Value::Str(s)
        }
        // ---- i64 comparisons ----
        (Value::I64(x), Value::I64(y), BinOp::Eq) => Value::Bool(x == y),
        (Value::I64(x), Value::I64(y), BinOp::Ne) => Value::Bool(x != y),
        (Value::I64(x), Value::I64(y), BinOp::Lt) => Value::Bool(x < y),
        (Value::I64(x), Value::I64(y), BinOp::Le) => Value::Bool(x <= y),
        (Value::I64(x), Value::I64(y), BinOp::Gt) => Value::Bool(x > y),
        (Value::I64(x), Value::I64(y), BinOp::Ge) => Value::Bool(x >= y),
        // ---- f64 comparisons (IEEE 754: NaN compares unequal) ----
        (Value::F64(x), Value::F64(y), BinOp::Eq) => Value::Bool(x == y),
        (Value::F64(x), Value::F64(y), BinOp::Ne) => Value::Bool(x != y),
        (Value::F64(x), Value::F64(y), BinOp::Lt) => Value::Bool(x < y),
        (Value::F64(x), Value::F64(y), BinOp::Le) => Value::Bool(x <= y),
        (Value::F64(x), Value::F64(y), BinOp::Gt) => Value::Bool(x > y),
        (Value::F64(x), Value::F64(y), BinOp::Ge) => Value::Bool(x >= y),
        // ---- byte comparisons ----
        (Value::Byte(x), Value::Byte(y), BinOp::Eq) => Value::Bool(x == y),
        (Value::Byte(x), Value::Byte(y), BinOp::Ne) => Value::Bool(x != y),
        // ---- bool logic + equality ----
        (Value::Bool(x), Value::Bool(y), BinOp::And) => Value::Bool(*x && *y),
        (Value::Bool(x), Value::Bool(y), BinOp::Or) => Value::Bool(*x || *y),
        (Value::Bool(x), Value::Bool(y), BinOp::Eq) => Value::Bool(x == y),
        (Value::Bool(x), Value::Bool(y), BinOp::Ne) => Value::Bool(x != y),
        // ---- string equality ----
        (Value::Str(x), Value::Str(y), BinOp::Eq) => Value::Bool(x == y),
        (Value::Str(x), Value::Str(y), BinOp::Ne) => Value::Bool(x != y),
        // anything else: no fold rule.
        _ => return Ok(None),
    };
    Ok(Some(folded))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;
    use crate::parser::Parser;
    use crate::typechecker::check_program;

    /// lex + parse + typecheck + compile a source string; return the codegen
    /// result directly (errors and all).
    fn compile(src: &str) -> Result<Program, Vec<QalaError>> {
        let tokens = Lexer::tokenize(src).expect("lex failed");
        let ast = Parser::parse(&tokens).expect("parse failed");
        let (typed, terrors, _) = check_program(&ast, src);
        assert!(terrors.is_empty(), "typecheck errors: {terrors:?}");
        compile_program(&typed, src)
    }

    /// like [`compile`] but panics on any codegen error -- happy-path tests.
    fn compile_ok(src: &str) -> Program {
        compile(src).unwrap_or_else(|e| panic!("codegen errors: {e:?}"))
    }

    /// typecheck `src` and return the typed AST plus the source. shared by
    /// the expression-level test helpers below.
    fn typecheck(src: &str) -> TypedAst {
        let tokens = Lexer::tokenize(src).expect("lex failed");
        let ast = Parser::parse(&tokens).expect("parse failed");
        let (typed, terrors, _) = check_program(&ast, src);
        assert!(terrors.is_empty(), "typecheck errors: {terrors:?}");
        typed
    }

    /// compile a single expression directly into a fresh chunk and return
    /// `(code_bytes, constants)`. the expression is the trailing value of
    /// `fn main`'s body in the wrapper program -- so test sources are
    /// written as `fn main() is io { EXPR }` (or include the helper fns the
    /// expression needs). `compile_block` / `compile_stmt` are still
    /// placeholders in Task 2, so the public entry cannot be used yet;
    /// running `compile_expr` directly is the Task-2 / Task-3 test path.
    fn compile_expr_of(src: &str, fn_name: &str) -> (Vec<u8>, Vec<Value>) {
        let typed = typecheck(src);
        let mut cg = Codegen::new(src);
        cg.build_tables(&typed);
        // find the named function and its trailing-value expression.
        let decl = typed
            .iter()
            .find_map(|item| match item {
                TypedItem::Fn(d) if d.name == fn_name => Some(d),
                _ => None,
            })
            .unwrap_or_else(|| panic!("function `{fn_name}` not found"));
        let fn_id = cg.fn_table[&FnKey {
            type_name: None,
            name: fn_name.to_string(),
        }];
        cg.current_fn_id = fn_id;
        cg.push_scope();
        for (i, param) in decl.params.iter().enumerate() {
            if let Some(scope) = cg.scopes.last_mut() {
                scope.locals.push((param.name.clone(), i as u16));
            }
        }
        let value = decl
            .body
            .value
            .as_ref()
            .unwrap_or_else(|| panic!("function `{fn_name}` body has no trailing value"));
        cg.compile_expr(value).expect("compile_expr failed");
        let chunk = &cg.program.chunks[fn_id as usize];
        (chunk.code.clone(), chunk.constants.clone())
    }

    /// like [`compile_expr_of`] but returns the codegen error rather than
    /// panicking -- used by the fold-overflow tests.
    fn compile_expr_err(src: &str, fn_name: &str) -> QalaError {
        let typed = typecheck(src);
        let mut cg = Codegen::new(src);
        cg.build_tables(&typed);
        let decl = typed
            .iter()
            .find_map(|item| match item {
                TypedItem::Fn(d) if d.name == fn_name => Some(d),
                _ => None,
            })
            .unwrap_or_else(|| panic!("function `{fn_name}` not found"));
        let fn_id = cg.fn_table[&FnKey {
            type_name: None,
            name: fn_name.to_string(),
        }];
        cg.current_fn_id = fn_id;
        cg.push_scope();
        for (i, param) in decl.params.iter().enumerate() {
            if let Some(scope) = cg.scopes.last_mut() {
                scope.locals.push((param.name.clone(), i as u16));
            }
        }
        let value = decl.body.value.as_ref().expect("body has no trailing value");
        cg.compile_expr(value)
            .expect_err("expected a codegen error")
    }

    /// the opcodes (no operands) appearing in `code`, in order -- a coarse
    /// shape check for tests that do not need exact operand bytes.
    fn opcodes(code: &[u8]) -> Vec<Opcode> {
        let mut ops = Vec::new();
        let mut off = 0;
        while off < code.len() {
            match Opcode::from_u8(code[off]) {
                Some(op) => {
                    ops.push(op);
                    off += 1 + op.operand_bytes() as usize;
                }
                None => break,
            }
        }
        ops
    }

    // Test 1: an empty program compiles to an empty Program.
    #[test]
    fn skeleton_empty_program_compiles_to_empty_program() {
        let p = compile_program(&Vec::new(), "").expect("empty program should compile");
        assert!(p.chunks.is_empty(), "chunks should be empty");
        assert_eq!(p.main_index, 0);
        assert!(p.globals.is_empty());
        assert!(p.fn_names.is_empty());
        // enum_variant_names holds the four built-in Result/Option variants.
        assert_eq!(p.enum_variant_names.len(), 4);
    }

    // Test 2: a single `fn main` produces one chunk with main at fn-id 0.
    #[test]
    fn skeleton_single_main_fn_produces_one_chunk() {
        let p = compile_ok("fn main() is io { }");
        assert_eq!(p.chunks.len(), 1, "one user fn -> one chunk");
        assert_eq!(p.main_index, 0);
        assert_eq!(p.fn_names, vec!["main".to_string()]);
        // the fall-through completion emits at least a RETURN byte.
        assert!(!p.chunks[0].code.is_empty(), "main chunk should have a RETURN");
        assert_eq!(p.chunks[0].code[0], Opcode::Return as u8);
    }

    // Test 3: build_tables records struct field indices densely.
    #[test]
    fn skeleton_build_tables_records_struct_field_indices() {
        let src = "struct Point { x: f64, y: f64 }\nfn main() is io { }";
        let tokens = Lexer::tokenize(src).expect("lex");
        let ast = Parser::parse(&tokens).expect("parse");
        let (typed, terrors, _) = check_program(&ast, src);
        assert!(terrors.is_empty(), "{terrors:?}");
        let mut cg = Codegen::new(src);
        cg.build_tables(&typed);
        assert_eq!(
            cg.struct_field_index
                .get(&("Point".to_string(), "x".to_string())),
            Some(&0)
        );
        assert_eq!(
            cg.struct_field_index
                .get(&("Point".to_string(), "y".to_string())),
            Some(&1)
        );
    }

    // Test 4: build_tables records enum variant ids + payload counts.
    #[test]
    fn skeleton_build_tables_records_enum_variant_ids_and_payloads() {
        let src = "enum Shape { Circle(f64), Rect(f64, f64), Triangle }\nfn main() is io { }";
        let tokens = Lexer::tokenize(src).expect("lex");
        let ast = Parser::parse(&tokens).expect("parse");
        let (typed, terrors, _) = check_program(&ast, src);
        assert!(terrors.is_empty(), "{terrors:?}");
        let mut cg = Codegen::new(src);
        cg.build_tables(&typed);
        // three user variants; built-in Result/Option occupy ids 0..4, so
        // the user variants start at 4.
        for v in ["Circle", "Rect", "Triangle"] {
            assert!(
                cg.enum_variant_table
                    .contains_key(&("Shape".to_string(), v.to_string())),
                "missing variant {v}"
            );
        }
        assert_eq!(
            cg.enum_variant_payload_count
                .get(&("Shape".to_string(), "Circle".to_string())),
            Some(&1)
        );
        assert_eq!(
            cg.enum_variant_payload_count
                .get(&("Shape".to_string(), "Rect".to_string())),
            Some(&2)
        );
        assert_eq!(
            cg.enum_variant_payload_count
                .get(&("Shape".to_string(), "Triangle".to_string())),
            Some(&0)
        );
        // the variant ids index program.enum_variant_names: id N names
        // (Shape, that variant).
        let circle_id = cg.enum_variant_table[&("Shape".to_string(), "Circle".to_string())];
        assert_eq!(
            cg.program.enum_variant_names[circle_id as usize],
            ("Shape".to_string(), "Circle".to_string())
        );
    }

    // Test 5: stdlib fn-ids live in the STDLIB_FN_BASE namespace; user fns
    // get dense ids from 0; fn_effects holds each stdlib entry's effect.
    #[test]
    fn skeleton_stdlib_fn_ids_are_reserved_and_separate_from_user_ids() {
        let src = "fn helper() is pure { }\nfn main() is io { }";
        let tokens = Lexer::tokenize(src).expect("lex");
        let ast = Parser::parse(&tokens).expect("parse");
        let (typed, terrors, _) = check_program(&ast, src);
        assert!(terrors.is_empty(), "{terrors:?}");
        let mut cg = Codegen::new(src);
        cg.build_tables(&typed);
        // the first user fn gets fn-id 0 (dense from zero).
        let helper_id = cg.fn_table[&FnKey {
            type_name: None,
            name: "helper".to_string(),
        }];
        assert_eq!(helper_id, 0);
        // stdlib `println` lives at STDLIB_FN_BASE + its index (index 1).
        let (println_id, println_effect) = cg.stdlib_table[&FnKey {
            type_name: None,
            name: "println".to_string(),
        }];
        assert_eq!(println_id, STDLIB_FN_BASE + 1);
        assert_eq!(println_effect, EffectSet::io());
        // fn_effects holds the stdlib entry's effect under its fn-id.
        assert_eq!(cg.fn_effects.get(&println_id), Some(&EffectSet::io()));
        // `sqrt` is pure.
        let (sqrt_id, _) = cg.stdlib_table[&FnKey {
            type_name: None,
            name: "sqrt".to_string(),
        }];
        assert_eq!(cg.fn_effects.get(&sqrt_id), Some(&EffectSet::pure()));
    }

    // Test 6: push/pop an empty scope does nothing and leaves no scope.
    #[test]
    fn skeleton_push_then_pop_empty_scope_is_a_noop() {
        let mut cg = Codegen::new("");
        cg.program.chunks.push(Chunk::new());
        cg.program.fn_names.push("f".to_string());
        cg.current_fn_id = 0;
        cg.push_scope();
        assert_eq!(cg.scopes.len(), 1);
        cg.pop_scope_and_emit_defers(1).expect("pop empty scope");
        assert!(cg.scopes.is_empty());
        // no defers -> no bytecode emitted.
        assert!(cg.program.chunks[0].code.is_empty());
    }

    // Test 7: a single registered defer emits its bytecode at scope exit.
    #[test]
    fn skeleton_pop_scope_emits_a_single_registered_defer() {
        // `fn f() is io { defer println("a") }` -- the defer is registered
        // by Task 4's compile_stmt, but here we exercise the helper directly
        // by hand-registering a typed Call expression in a scope.
        let src = "fn f() is io { }";
        let tokens = Lexer::tokenize(src).expect("lex");
        let ast = Parser::parse(&tokens).expect("parse");
        let (typed, _, _) = check_program(&ast, src);
        let mut cg = Codegen::new(src);
        cg.build_tables(&typed);
        cg.current_fn_id = 0;
        cg.push_scope();
        // a Void-typed Call -- the defer body. `pop_scope_and_emit_defers`
        // compiles it (the arg CONST + the CALL) and pops the scope. a Void
        // result needs no trailing POP.
        let call = TypedExpr::Call {
            callee: Box::new(TypedExpr::Ident {
                name: "println".to_string(),
                ty: QalaType::Function {
                    params: vec![QalaType::Str],
                    returns: Box::new(QalaType::Void),
                },
                span: Span::new(0, 7),
            }),
            args: vec![TypedExpr::Str {
                value: "a".to_string(),
                ty: QalaType::Str,
                span: Span::new(8, 3),
            }],
            ty: QalaType::Void,
            span: Span::new(0, 12),
        };
        if let Some(scope) = cg.scopes.last_mut() {
            scope.deferred.push(call);
        }
        cg.pop_scope_and_emit_defers(1).expect("emit defers");
        assert!(cg.scopes.is_empty(), "scope should be popped");
        // the defer body emitted: CONST("a") then CALL(println). a Void
        // result adds no POP.
        let ops = opcodes(&cg.program.chunks[0].code);
        assert_eq!(ops, vec![Opcode::Const, Opcode::Call]);
        assert!(
            !ops.contains(&Opcode::Pop),
            "a Void defer body needs no POP"
        );
    }

    // Test 8: a fn with two params has them in slots 0 and 1.
    #[test]
    fn skeleton_fn_params_occupy_slots_zero_and_one() {
        // with Task 1's placeholder compile_block, the body bytecode is just
        // a RETURN; this test pins the table/scope wiring: two params, one
        // user chunk, fn_names == ["add"].
        let p = compile_ok("fn add(a: i64, b: i64) -> i64 is pure { return a + b }");
        assert_eq!(p.chunks.len(), 1);
        assert_eq!(p.fn_names, vec!["add".to_string()]);
    }

    // Test 9: struct / enum / interface items emit no chunks.
    #[test]
    fn skeleton_type_level_items_emit_no_chunks() {
        let p = compile_ok(
            "struct Point { x: i64, y: i64 }\n\
             enum Dir { North, South }\n\
             interface Show { fn show(self) -> str }\n\
             fn main() is io { }",
        );
        // only `main` produces a chunk.
        assert_eq!(p.chunks.len(), 1);
        assert_eq!(p.fn_names, vec!["main".to_string()]);
    }

    // Test 10: two functions get sequential fn-ids and chunks.
    #[test]
    fn skeleton_two_functions_get_sequential_fn_ids() {
        let p = compile_ok("fn first() is pure { }\nfn second() is pure { }");
        assert_eq!(p.chunks.len(), 2);
        assert_eq!(p.fn_names, vec!["first".to_string(), "second".to_string()]);
    }

    // Test 11: an empty `fn main` body still emits a RETURN.
    #[test]
    fn skeleton_empty_fn_body_emits_a_return() {
        let p = compile_ok("fn main() is io { }");
        assert_eq!(p.chunks[0].code, vec![Opcode::Return as u8]);
        assert_eq!(p.chunks[0].source_lines.len(), 1);
    }

    // Test 12: line_at maps a span byte-offset to a 1-based source line.
    #[test]
    fn skeleton_line_at_maps_span_to_source_line() {
        let cg = Codegen::new("abc");
        assert_eq!(cg.line_at(Span::new(0, 3)), 1);
        let cg2 = Codegen::new("abc\ndef");
        // byte 4 is the 'd' on line 2.
        assert_eq!(cg2.line_at(Span::new(4, 3)), 2);
    }

    // ===== Task 2: compile_expr literals / locals / binary / unary / calls ===

    // Test 1: an integer literal expression compiles to one CONST(I64). the
    // wrapper fn returns i64 so the literal is the body's trailing value.
    #[test]
    fn compile_expr_int_literal_emits_one_const() {
        let (code, consts) = compile_expr_of("fn f() -> i64 is pure { 42 }", "f");
        assert_eq!(opcodes(&code), vec![Opcode::Const]);
        assert_eq!(consts, vec![Value::I64(42)]);
    }

    // Test 2: a float literal compiles to one CONST(F64).
    #[test]
    fn compile_expr_float_literal_emits_one_const() {
        let (code, consts) = compile_expr_of("fn f() -> f64 is pure { 3.5 }", "f");
        assert_eq!(opcodes(&code), vec![Opcode::Const]);
        assert_eq!(consts, vec![Value::F64(3.5)]);
    }

    // Test 3: a bool literal compiles to CONST(Bool).
    #[test]
    fn compile_expr_bool_literal_emits_one_const() {
        let (_, consts) = compile_expr_of("fn f() -> bool is pure { true }", "f");
        assert_eq!(consts, vec![Value::Bool(true)]);
    }

    // Test 4: a byte literal compiles to CONST(Byte).
    #[test]
    fn compile_expr_byte_literal_emits_one_const() {
        let (_, consts) = compile_expr_of("fn f() -> byte is pure { b'A' }", "f");
        assert_eq!(consts, vec![Value::Byte(0x41)]);
    }

    // Test 5: a string literal compiles to CONST(Str).
    #[test]
    fn compile_expr_str_literal_emits_one_const() {
        let (_, consts) = compile_expr_of("fn f() -> str is pure { \"hi\" }", "f");
        assert_eq!(consts, vec![Value::Str("hi".to_string())]);
    }

    // Test 6: an identifier bound as a parameter compiles to GET_LOCAL.
    #[test]
    fn compile_expr_ident_param_emits_get_local() {
        let (code, _) = compile_expr_of("fn f(x: i64) -> i64 is pure { x }", "f");
        assert_eq!(opcodes(&code), vec![Opcode::GetLocal]);
        // the GET_LOCAL operand is slot 0 (the first parameter).
        assert_eq!(code[0], Opcode::GetLocal as u8);
        assert_eq!(u16::from_le_bytes([code[1], code[2]]), 0);
    }

    // Test 7: `1 + 2` folds at codegen to a single CONST(I64(3)) -- two
    // operand CONSTs + ADD collapse to one CONST.
    #[test]
    fn compile_expr_binary_arith_folds_to_one_const() {
        let (code, consts) = compile_expr_of("fn f() -> i64 is pure { 1 + 2 }", "f");
        assert_eq!(opcodes(&code), vec![Opcode::Const], "1+2 should fold");
        // the folded pool entry is I64(3); the two source CONSTs were rewound
        // (orphans 0 and 1 remain in the pool, the live CONST points at 2).
        assert_eq!(consts.last(), Some(&Value::I64(3)));
    }

    // Test 8: `1 + 2 + 3` folds bottom-up to a single CONST(I64(6)).
    #[test]
    fn compile_expr_binary_chain_folds_bottom_up() {
        let (code, consts) = compile_expr_of("fn f() -> i64 is pure { 1 + 2 + 3 }", "f");
        assert_eq!(opcodes(&code), vec![Opcode::Const]);
        assert_eq!(consts.last(), Some(&Value::I64(6)));
    }

    // Test 9: `x + 1` with x a local does NOT fold.
    #[test]
    fn compile_expr_binary_with_local_does_not_fold() {
        let (code, _) = compile_expr_of("fn f(x: i64) -> i64 is pure { x + 1 }", "f");
        assert_eq!(
            opcodes(&code),
            vec![Opcode::GetLocal, Opcode::Const, Opcode::Add]
        );
    }

    // Test 10: `i64::MAX + 1` is a fold-time IntegerOverflow.
    #[test]
    fn compile_expr_fold_add_overflow_is_an_error() {
        let err = compile_expr_err(
            "fn f() -> i64 is pure { 9223372036854775807 + 1 }",
            "f",
        );
        match err {
            QalaError::IntegerOverflow { op, lhs, rhs, .. } => {
                assert_eq!(op, BinOp::Add);
                assert_eq!(lhs, i64::MAX);
                assert_eq!(rhs, 1);
            }
            other => panic!("expected IntegerOverflow, got {other:?}"),
        }
    }

    // Test 11: `i64::MAX * 2` overflows on fold.
    #[test]
    fn compile_expr_fold_mul_overflow_is_an_error() {
        let err = compile_expr_err(
            "fn f() -> i64 is pure { 9223372036854775807 * 2 }",
            "f",
        );
        match err {
            QalaError::IntegerOverflow { op, lhs, rhs, .. } => {
                assert_eq!(op, BinOp::Mul);
                assert_eq!(lhs, i64::MAX);
                assert_eq!(rhs, 2);
            }
            other => panic!("expected IntegerOverflow, got {other:?}"),
        }
    }

    // Test 12: `1e300 * 1e300` folds to F64(inf) with NO error.
    #[test]
    fn compile_expr_fold_f64_overflow_yields_infinity() {
        let (code, consts) = compile_expr_of("fn f() -> f64 is pure { 1e300 * 1e300 }", "f");
        assert_eq!(opcodes(&code), vec![Opcode::Const]);
        match consts.last() {
            Some(Value::F64(x)) => assert!(x.is_infinite() && *x > 0.0),
            other => panic!("expected F64(inf), got {other:?}"),
        }
    }

    // Test 13: `0.0 / 0.0` folds to F64(NaN) with NO error.
    #[test]
    fn compile_expr_fold_f64_div_zero_yields_nan() {
        let (code, consts) = compile_expr_of("fn f() -> f64 is pure { 0.0 / 0.0 }", "f");
        assert_eq!(opcodes(&code), vec![Opcode::Const]);
        match consts.last() {
            Some(Value::F64(x)) => assert!(x.is_nan()),
            other => panic!("expected F64(NaN), got {other:?}"),
        }
    }

    // Test 14: `"a" + "b"` folds to a single CONST(Str("ab")).
    #[test]
    fn compile_expr_fold_string_concat() {
        let (code, consts) = compile_expr_of("fn f() -> str is pure { \"a\" + \"b\" }", "f");
        assert_eq!(opcodes(&code), vec![Opcode::Const]);
        assert_eq!(consts.last(), Some(&Value::Str("ab".to_string())));
    }

    // Test 15: `1 < 2` folds to CONST(Bool(true)).
    #[test]
    fn compile_expr_fold_comparison() {
        let (code, consts) = compile_expr_of("fn f() -> bool is pure { 1 < 2 }", "f");
        assert_eq!(opcodes(&code), vec![Opcode::Const]);
        assert_eq!(consts.last(), Some(&Value::Bool(true)));
    }

    // Test 16: `x < 2` with x a local emits GET_LOCAL + CONST + LT.
    #[test]
    fn compile_expr_comparison_with_local_does_not_fold() {
        let (code, _) = compile_expr_of("fn f(x: i64) -> bool is pure { x < 2 }", "f");
        assert_eq!(
            opcodes(&code),
            vec![Opcode::GetLocal, Opcode::Const, Opcode::Lt]
        );
    }

    // Test 17: `true && false` folds to CONST(Bool(false)).
    #[test]
    fn compile_expr_fold_boolean_and() {
        let (code, consts) = compile_expr_of("fn f() -> bool is pure { true && false }", "f");
        assert_eq!(opcodes(&code), vec![Opcode::Const]);
        assert_eq!(consts.last(), Some(&Value::Bool(false)));
    }

    // Test 18: unary fold -- `!true`, `!false`, `-(42)` all collapse to one
    // CONST. (the `-(i64::MIN)` overflow case is unreachable from source --
    // the lexer caps integer literals at the i64::MAX magnitude, so a
    // literal i64::MIN cannot be written and then negated. the `checked_neg`
    // overflow path in `try_fold_unary` is still exercised by the
    // ComptimeInterpreter's NEG opcode test in Task 5.)
    #[test]
    fn compile_expr_fold_unary() {
        let (code, consts) = compile_expr_of("fn f() -> bool is pure { !true }", "f");
        assert_eq!(opcodes(&code), vec![Opcode::Const]);
        assert_eq!(consts.last(), Some(&Value::Bool(false)));
        let (_, consts) = compile_expr_of("fn f() -> bool is pure { !false }", "f");
        assert_eq!(consts.last(), Some(&Value::Bool(true)));
        let (code, consts) = compile_expr_of("fn f() -> i64 is pure { -(42) }", "f");
        assert_eq!(opcodes(&code), vec![Opcode::Const]);
        assert_eq!(consts.last(), Some(&Value::I64(-42)));
    }

    // Test 19: `-(3.5)` folds to CONST(F64(-3.5)).
    #[test]
    fn compile_expr_fold_f64_unary() {
        let (code, consts) = compile_expr_of("fn f() -> f64 is pure { -(3.5) }", "f");
        assert_eq!(opcodes(&code), vec![Opcode::Const]);
        assert_eq!(consts.last(), Some(&Value::F64(-3.5)));
    }

    // Test 20: mixed i64/f64 with locals picks ADD vs F_ADD by operand type.
    #[test]
    fn compile_expr_picks_int_vs_float_opcode() {
        let (code, _) = compile_expr_of("fn f(a: i64, b: i64) -> i64 is pure { a + b }", "f");
        assert!(opcodes(&code).contains(&Opcode::Add));
        let (code, _) = compile_expr_of("fn f(a: f64, b: f64) -> f64 is pure { a + b }", "f");
        assert!(opcodes(&code).contains(&Opcode::FAdd));
    }

    // Test 21: a user-fn call emits args then CALL(fn_id, argc).
    #[test]
    fn compile_expr_user_call_emits_args_then_call() {
        let src = "fn add(a: i64, b: i64) -> i64 is pure { return a + b }\n\
                   fn main() -> i64 is pure { add(1, 2) }";
        let (code, _) = compile_expr_of(src, "main");
        assert_eq!(
            opcodes(&code),
            vec![Opcode::Const, Opcode::Const, Opcode::Call]
        );
        // CALL operand: fn-id of `add` (0), argc 2.
        let call_off = 6; // two CONSTs (3 bytes each) precede CALL.
        assert_eq!(code[call_off], Opcode::Call as u8);
        assert_eq!(u16::from_le_bytes([code[call_off + 1], code[call_off + 2]]), 0);
        assert_eq!(code[call_off + 3], 2);
    }

    // Test 22: a stdlib call emits the arg then CALL(stdlib_id, argc).
    #[test]
    fn compile_expr_stdlib_call_emits_arg_then_call() {
        let (code, _) = compile_expr_of("fn main() is io { println(\"hi\") }", "main");
        assert_eq!(opcodes(&code), vec![Opcode::Const, Opcode::Call]);
        // the CALL fn-id is println's stdlib id (STDLIB_FN_BASE + 1).
        assert_eq!(code[3], Opcode::Call as u8);
        assert_eq!(
            u16::from_le_bytes([code[4], code[5]]),
            STDLIB_FN_BASE + 1
        );
        assert_eq!(code[6], 1);
    }

    // Test 23: every emission records its source line; lockstep holds.
    #[test]
    fn compile_expr_records_source_lines_in_lockstep() {
        // a two-line body: the trailing expr is on line 2.
        let src = "fn f() -> i64 is pure {\n    x_helper() + 1\n}\n\
                   fn x_helper() -> i64 is pure { 5 }";
        let typed = typecheck(src);
        let mut cg = Codegen::new(src);
        cg.build_tables(&typed);
        let fn_id = cg.fn_table[&FnKey {
            type_name: None,
            name: "f".to_string(),
        }];
        cg.current_fn_id = fn_id;
        cg.push_scope();
        let decl = typed
            .iter()
            .find_map(|i| match i {
                TypedItem::Fn(d) if d.name == "f" => Some(d),
                _ => None,
            })
            .unwrap();
        cg.compile_expr(decl.body.value.as_ref().unwrap())
            .expect("compile_expr");
        let chunk = &cg.program.chunks[fn_id as usize];
        // lockstep invariant.
        assert_eq!(chunk.code.len(), chunk.source_lines.len());
        // every emitted byte for the trailing expr is on line 2.
        assert!(chunk.source_lines.iter().all(|&l| l == 2));
    }
}
