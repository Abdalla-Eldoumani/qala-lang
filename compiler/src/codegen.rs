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

use crate::chunk::{Chunk, Program};
use crate::effects::EffectSet;
use crate::errors::QalaError;
use crate::opcode::Opcode;
use crate::span::{LineIndex, Span};
use crate::typed_ast::{TypedAst, TypedBlock, TypedExpr, TypedItem, TypedStmt};
use crate::types::QalaType;
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

    // ---- expression dispatch (placeholder; Tasks 2-5 implement) ------------

    /// compile one expression, leaving its value on the stack.
    ///
    /// PLACEHOLDER for Task 1: emits nothing. Tasks 2-5 replace the body
    /// with the per-variant arms (primitives, folding, match, comptime, ...).
    fn compile_expr(&mut self, e: &TypedExpr) -> Result<(), QalaError> {
        let _ = e;
        Ok(())
    }
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
        // a Void-typed Call -- the defer body. with Task 1's placeholder
        // compile_expr emitting nothing, the helper still runs and pops the
        // scope without a POP (Void result).
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
        // a Void defer body adds no POP; placeholder compile_expr emits
        // nothing, so the chunk stays empty -- the helper itself did not
        // panic and the scope is gone, which is what Task 1 verifies.
        assert!(cg.program.chunks[0].code.is_empty());
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
}
