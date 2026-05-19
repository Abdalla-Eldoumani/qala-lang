//! the ARM64 backend: a typed AST lowered to AArch64 assembly text in the
//! CPSC 355 hosted-Linux dialect.
//!
//! the structural twin of [`crate::codegen`]: it consumes the same
//! [`TypedAst`] the bytecode codegen consumes and produces a `String` of
//! assembly instead of a [`Program`](crate::chunk::Program). the entry point
//! [`compile_arm64`] parallels [`crate::codegen::compile_program`].
//!
//! ## the integer core
//!
//! this module emits, for `i64` and `bool`: arithmetic (`+ - * / %`), the six
//! comparisons, the short-circuit `&&` / `||`, the unary `!` and `-`,
//! parameter loads, and a function's prologue / body / epilogue following
//! AAPCS64. control flow (`if` / `while` / `for`) and `let` bindings arrive in
//! plan 12-02; calls arrive in plan 12-03. every construct outside the integer
//! core -- a float, a string, a struct literal, a method, a `>8`-parameter
//! function -- returns a [`QalaError`], never a panic, never a `todo!()`.
//!
//! ## the dialect
//!
//! the emitted text matches the course tutorial files under `docs/references/`:
//! lowercase mnemonics and directives, the m4 `define(fp, x29)` /
//! `define(lr, x30)` preamble, a `.text` section, per-function `.balign 4` /
//! `.global` / `.type ..., @function` directives, and the
//! `stp fp, lr, [sp, alloc]!` / `ldp fp, lr, [sp], dealloc` prologue and
//! epilogue. an `i64` is a 64-bit `x` register and an 8-byte stack slot
//! throughout -- never a 32-bit `w` register.
//!
//! ## module layout
//!
//! `emit.rs` builds the assembly text; `frame.rs` plans each function's stack
//! frame; `labels.rs` generates unique branch labels; `expr.rs` is a second
//! `impl Arm64Backend` block holding the expression codegen. plan 12-02 adds
//! `stmt.rs` for the full statement and control-flow codegen.

mod emit;
mod expr;
mod frame;
mod labels;

use crate::errors::QalaError;
use crate::span::LineIndex;
use crate::typed_ast::{TypedAst, TypedBlock, TypedFnDecl, TypedItem, TypedStmt};
use crate::types::QalaType;
use std::collections::HashSet;

use emit::Asm;
use frame::FrameLayout;
use labels::LabelGen;

/// lower a typed program to AArch64 assembly text in the CPSC 355 dialect.
///
/// errors accumulate rather than fail-fast: an unsupported construct in one
/// function does not stop the others from compiling. on success returns the
/// full assembly text; on any errors returns the errors sorted by span -- the
/// same `(span.start, span.len)` ordering contract as
/// [`crate::codegen::compile_program`].
///
/// the entry point the `qala build --target arm64` CLI seam (plan 12-03)
/// calls. `src` is the original source, kept for span-to-line lookups when
/// rendering an error.
pub fn compile_arm64(ast: &TypedAst, src: &str) -> Result<String, Vec<QalaError>> {
    let mut backend = Arm64Backend::new(src);
    backend.scan_fn_names(ast);
    for item in ast {
        if let Err(e) = backend.compile_item(item) {
            backend.errors.push(e);
        }
    }
    if !backend.errors.is_empty() {
        backend.errors.sort_by_key(|e| (e.span().start, e.span().len));
        return Err(backend.errors);
    }
    Ok(backend.finish())
}

/// the ARM64 backend's walk state -- the analog of
/// [`crate::codegen`]'s `Codegen` struct.
///
/// carries the assembly-text builder, the per-function frame layout, the label
/// generator, the set of user function names (pre-scanned so a forward call
/// resolves), the accumulated errors, and the source plus its line index.
struct Arm64Backend {
    /// the assembly-text builder.
    asm: Asm,
    /// the unique-label generator -- monotonic across the whole compile, so
    /// every branch label in the file is distinct and emission is deterministic.
    labels: LabelGen,
    /// the stack-frame layout of the function currently being compiled, or
    /// `None` between functions.
    frame: Option<FrameLayout>,
    /// the name of the function currently being compiled, or `None` between
    /// functions. a `return` branches to `.L<current_fn>_epilogue`.
    current_fn: Option<String>,
    /// every user-declared function name, pre-scanned before the compile walk.
    /// plan 12-03 consults this to resolve a call target; this plan records it
    /// for that forward reference.
    fn_names: HashSet<String>,
    /// the errors accumulated across every function.
    errors: Vec<QalaError>,
    /// the original source text. plan 12-02 / 12-03 read it when an error
    /// diagnostic needs the offending source line; this plan only stores it.
    #[allow(dead_code)]
    src: String,
    /// the byte-offset-to-line index. plan 12-02 / 12-03 consult it for
    /// span-to-line lookups in richer diagnostics; this plan only stores it.
    #[allow(dead_code)]
    line_index: LineIndex,
}

impl Arm64Backend {
    /// construct a fresh backend over `src`: an empty assembly builder, no
    /// current function, no errors.
    fn new(src: &str) -> Self {
        Arm64Backend {
            asm: Asm::new(),
            labels: LabelGen::new(),
            frame: None,
            current_fn: None,
            fn_names: HashSet::new(),
            errors: Vec::new(),
            src: src.to_string(),
            line_index: LineIndex::new(src),
        }
    }

    /// pre-scan every user function's name into `fn_names`, so a call to a
    /// function declared later in the file still resolves.
    ///
    /// the analog of `Codegen::build_tables`'s function pre-registration.
    fn scan_fn_names(&mut self, ast: &TypedAst) {
        for item in ast {
            if let TypedItem::Fn(decl) = item {
                self.fn_names.insert(decl.name.clone());
            }
        }
    }

    /// the frame layout of the function currently being compiled.
    ///
    /// panics only via a debug assertion if called between functions -- the
    /// per-function flow always sets the frame before walking the body, so a
    /// live call path never hits a `None`.
    fn frame(&self) -> &FrameLayout {
        self.frame
            .as_ref()
            .expect("frame() called with no function in progress")
    }

    /// the frame layout of the current function, mutably -- for the expression
    /// codegen's scratch-slot claim / release.
    fn frame_mut(&mut self) -> &mut FrameLayout {
        self.frame
            .as_mut()
            .expect("frame_mut() called with no function in progress")
    }

    /// compile one top-level item.
    ///
    /// a `Fn` item emits a complete function; `Struct` / `Enum` / `Interface`
    /// items are type-level and emit nothing -- exactly as
    /// [`crate::codegen`]'s `compile_item` treats them.
    fn compile_item(&mut self, item: &TypedItem) -> Result<(), QalaError> {
        match item {
            TypedItem::Fn(decl) => self.compile_fn(decl),
            // type-level items: nothing to emit.
            TypedItem::Struct(_) | TypedItem::Enum(_) | TypedItem::Interface(_) => Ok(()),
        }
    }

    /// compile one function: validate it, plan its frame, emit the directives,
    /// the prologue, the parameter spills, the body, and the epilogue.
    ///
    /// rejects, with a clean [`QalaError`], anything the integer core does not
    /// cover: a method (`type_name` set), a non-`i64`/`bool` parameter, a
    /// non-`i64`/`bool`/`void` return type, a parameter with a default, and a
    /// function with more than eight parameters.
    fn compile_fn(&mut self, decl: &TypedFnDecl) -> Result<(), QalaError> {
        self.validate_fn(decl)?;

        // plan the frame; it stays the current frame for the whole body walk.
        self.frame = Some(FrameLayout::plan_frame(decl));
        self.current_fn = Some(decl.name.clone());

        // a blank separator line before the function, for readability.
        self.asm.emit_line("");
        // the per-function directives: 4-byte align, export, mark as a
        // function, then the global label.
        self.asm.emit_insn(".balign 4");
        self.asm.emit_insn(&format!(".global {}", decl.name));
        self.asm
            .emit_insn(&format!(".type   {}, @function", decl.name));
        self.asm.emit_label(&decl.name);

        // the AAPCS64 prologue.
        for insn in self.frame().prologue() {
            self.asm.emit_insn(&insn);
        }

        // spill the incoming parameters from x0..x7 into their stack slots, so
        // a later Ident load reads a stable slot rather than a clobbered reg.
        for (i, param) in decl.params.iter().enumerate() {
            let slot = self.frame().slot_of(&param.name).ok_or_else(|| QalaError::Type {
                span: param.span,
                message: format!("arm64 backend: parameter `{}` has no slot", param.name),
            })?;
            self.asm.emit_insn_commented(
                &format!("str     x{i}, [fp, {slot}]"),
                &param.name,
            );
        }

        // walk the body. its trailing value, if any, lands in x0.
        self.compile_block(&decl.body)?;

        // the single epilogue block; a `return` branches here.
        let epilogue_label = self.epilogue_label(&decl.name);
        self.asm.emit_label(&epilogue_label);
        // a void function (and main) returns exit code 0; an i64/bool function
        // leaves its result in x0 (the body's trailing value already did).
        if decl.ret_ty == QalaType::Void {
            self.asm
                .emit_insn_commented("mov     x0, 0", "void return -> exit code 0");
        }
        for insn in self.frame().epilogue() {
            self.asm.emit_insn(&insn);
        }

        // the function is done; clear the per-function state.
        self.frame = None;
        self.current_fn = None;
        Ok(())
    }

    /// reject a function the integer core does not support, with a clean
    /// [`QalaError::Type`] carrying the offending span.
    fn validate_fn(&self, decl: &TypedFnDecl) -> Result<(), QalaError> {
        // a method (`fn Type.method`) -- methods are tied to structs, deferred.
        if decl.type_name.is_some() {
            return Err(QalaError::Type {
                span: decl.span,
                message: "the arm64 backend does not yet support methods".to_string(),
            });
        }
        // more than eight parameters spill onto the stack -- out of scope.
        if decl.params.len() > 8 {
            return Err(QalaError::Type {
                span: decl.span,
                message: "the arm64 backend supports at most 8 parameters".to_string(),
            });
        }
        for param in &decl.params {
            // a default value is an unsupported construct.
            if param.default.is_some() {
                return Err(QalaError::Type {
                    span: param.span,
                    message: "the arm64 backend does not yet support default parameters"
                        .to_string(),
                });
            }
            // only i64 and bool parameters are in the integer core.
            if !is_integer_core_type(&param.ty) {
                return Err(QalaError::Type {
                    span: param.span,
                    message: format!(
                        "the arm64 backend does not yet support {} parameters",
                        type_name(&param.ty)
                    ),
                });
            }
        }
        // the return type must be i64, bool, or void.
        if decl.ret_ty != QalaType::Void && !is_integer_core_type(&decl.ret_ty) {
            return Err(QalaError::Type {
                span: decl.span,
                message: format!(
                    "the arm64 backend does not yet support a {} return type",
                    type_name(&decl.ret_ty)
                ),
            });
        }
        Ok(())
    }

    /// walk a block: every statement, then the trailing value expression.
    ///
    /// THIS PLAN's minimal version -- it emits the statements and the trailing
    /// value with no scope handling. plan 12-02 replaces it with the full,
    /// scope-aware `compile_block` in `stmt.rs` (a name-to-slot scope stack for
    /// `let` bindings and shadowing).
    fn compile_block(&mut self, block: &TypedBlock) -> Result<(), QalaError> {
        for stmt in &block.stmts {
            self.compile_stmt(stmt)?;
        }
        if let Some(value) = &block.value {
            // the block's trailing value -- its result lands in x0.
            self.compile_expr(value)?;
        }
        Ok(())
    }

    /// compile one statement.
    ///
    /// THIS PLAN's minimal version handles only `Expr` and `Return`. `Let`,
    /// `If`, `While`, `For`, `Break`, and `Continue` return a clean
    /// [`QalaError`] -- plan 12-02 replaces those arms with real codegen in
    /// `stmt.rs`. `Defer` stays a permanent rejection (deferred beyond v2).
    fn compile_stmt(&mut self, stmt: &TypedStmt) -> Result<(), QalaError> {
        match stmt {
            // a statement-position expression: its value is discarded, but the
            // instructions still run (an effect, a call in a later plan).
            TypedStmt::Expr { expr, .. } => self.compile_expr(expr),
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
            // statements the later plans own.
            TypedStmt::Let { span, .. } => Err(self.unsupported_stmt(*span, "let bindings")),
            TypedStmt::If { span, .. } => {
                Err(self.unsupported_stmt(*span, "if statements"))
            }
            TypedStmt::While { span, .. } => {
                Err(self.unsupported_stmt(*span, "while loops"))
            }
            TypedStmt::For { span, .. } => Err(self.unsupported_stmt(*span, "for loops")),
            TypedStmt::Break { span } => {
                Err(self.unsupported_stmt(*span, "the break statement"))
            }
            TypedStmt::Continue { span } => {
                Err(self.unsupported_stmt(*span, "the continue statement"))
            }
            TypedStmt::Defer { span, .. } => {
                Err(self.unsupported_stmt(*span, "the defer statement"))
            }
        }
    }

    /// build the rejection error for a statement construct the integer core
    /// does not yet emit.
    fn unsupported_stmt(&self, span: crate::span::Span, what: &str) -> QalaError {
        QalaError::Type {
            span,
            message: format!("the arm64 backend does not yet support {what}"),
        }
    }

    /// the local epilogue label for a named function: `.L<name>_epilogue`.
    fn epilogue_label(&self, fn_name: &str) -> String {
        format!(".L{fn_name}_epilogue")
    }

    /// finish the compile: produce the complete assembly text.
    fn finish(self) -> String {
        self.asm.finish()
    }
}

/// whether a type is in the ARM64 integer core: `i64` or `bool`.
fn is_integer_core_type(ty: &QalaType) -> bool {
    matches!(ty, QalaType::I64 | QalaType::Bool)
}

/// a plain human name for a type, for an unsupported-construct message.
fn type_name(ty: &QalaType) -> &'static str {
    match ty {
        QalaType::I64 => "i64",
        QalaType::F64 => "f64",
        QalaType::Bool => "bool",
        QalaType::Str => "str",
        QalaType::Byte => "byte",
        QalaType::Void => "void",
        QalaType::Array(..) => "array",
        QalaType::Tuple(_) => "tuple",
        QalaType::Function { .. } => "function-typed",
        QalaType::Named(_) => "struct or enum",
        QalaType::Result(..) => "Result",
        QalaType::Option(_) => "Option",
        QalaType::FileHandle => "file-handle",
        QalaType::Unknown => "unresolved",
    }
}

#[cfg(test)]
impl Arm64Backend {
    /// plan a function's frame and set it as current, without emitting any
    /// directives or prologue -- a test hook so the expression codegen can be
    /// exercised in isolation against a real frame.
    fn begin_function(&mut self, decl: &TypedFnDecl) {
        self.frame = Some(FrameLayout::plan_frame(decl));
        self.current_fn = Some(decl.name.clone());
    }

    /// the assembly text emitted so far -- a test hook for inspecting partial
    /// output without going through `finish`.
    fn take_text(&mut self) -> String {
        std::mem::replace(&mut self.asm, Asm::new()).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;
    use crate::parser::Parser;
    use crate::typechecker::check_program;

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
        compile_arm64(&typed, src).unwrap_or_else(|e| panic!("arm64 errors: {e:?}"))
    }

    /// read a snapshot fixture, normalising CRLF to LF. a checked-out file on
    /// Windows has CRLF line endings; the emitter always produces LF, so the
    /// fixture's carriage returns are stripped for a platform-stable compare.
    fn read_snapshot(name: &str) -> String {
        let path = format!("{}/tests/snapshots/{name}", env!("CARGO_MANIFEST_DIR"));
        std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {path}: {e}"))
            .replace("\r\n", "\n")
    }

    #[test]
    fn an_empty_program_emits_just_the_preamble() {
        // no functions: the output is the m4 preamble and the .text header.
        let out = compile_arm64(&Vec::new(), "").expect("empty program compiles");
        assert!(out.starts_with("define(fp, x29)\ndefine(lr, x30)\n"));
        assert!(out.contains(".text"));
    }

    #[test]
    fn a_function_emits_its_directives_and_prologue() {
        let out = compile_ok("fn main() { }");
        assert!(out.contains(".balign 4"), "{out}");
        assert!(out.contains(".global main"), "{out}");
        assert!(out.contains(".type   main, @function"), "{out}");
        assert!(out.contains("\nmain:\n"), "{out}");
        assert!(out.contains("stp     fp, lr, [sp, "), "missing prologue: {out}");
        assert!(out.contains("mov     fp, sp"), "{out}");
    }

    #[test]
    fn a_function_emits_one_epilogue_block() {
        let out = compile_ok("fn main() { }");
        assert!(out.contains(".Lmain_epilogue:"), "{out}");
        assert!(out.contains("ldp     fp, lr, [sp], "), "missing epilogue: {out}");
        assert!(out.contains("\n        ret\n"), "{out}");
    }

    #[test]
    fn a_void_function_zeroes_x0_before_the_epilogue() {
        // a void main exits 0; the backend movs 0 into x0 before returning.
        let out = compile_ok("fn main() { }");
        assert!(out.contains("mov     x0, 0"), "{out}");
    }

    #[test]
    fn parameters_are_spilled_into_their_stack_slots() {
        // the two parameters spill to [fp, 16] and [fp, 24] -- positive
        // offsets above the saved fp/lr pair.
        let out = compile_ok("fn add(a: i64, b: i64) -> i64 { a + b }");
        assert!(out.contains("str     x0, [fp, 16]  // a"), "{out}");
        assert!(out.contains("str     x1, [fp, 24]  // b"), "{out}");
    }

    #[test]
    fn a_returning_function_leaves_its_value_in_x0() {
        // an i64 function: the trailing value already put the result in x0, so
        // there is no `mov x0, 0` before the epilogue.
        let out = compile_ok("fn seven() -> i64 { 7 }");
        assert!(out.contains("mov     x0, 7"), "{out}");
        assert!(
            !out.contains("mov     x0, 0"),
            "an i64 function must not zero x0: {out}"
        );
    }

    #[test]
    fn an_explicit_return_branches_to_the_epilogue() {
        let out = compile_ok("fn f() -> i64 { return 3 }");
        assert!(out.contains("b       .Lf_epilogue"), "{out}");
    }

    #[test]
    fn structs_and_enums_emit_nothing() {
        // a struct / enum declaration is type-level; only `main` produces text.
        let out = compile_ok("struct P { x: i64 }\nenum E { A, B }\nfn main() { }");
        assert!(out.contains("\nmain:\n"));
        assert!(!out.contains("\nP:\n"));
        assert!(!out.contains("\nE:\n"));
    }

    #[test]
    fn a_method_is_rejected_cleanly() {
        // `fn Type.method` is out of the integer core -- a clean error, not a
        // panic and not bad assembly.
        let typed = typecheck("struct P { x: i64 }\nfn P.get(self) -> i64 { self.x }");
        let err = compile_arm64(&typed, "struct P { x: i64 }\nfn P.get(self) -> i64 { self.x }")
            .expect_err("a method must be rejected");
        assert!(err[0].message().contains("method"), "{:?}", err);
    }

    #[test]
    fn a_float_parameter_is_rejected_cleanly() {
        let src = "fn f(x: f64) -> f64 { x }";
        let typed = typecheck(src);
        let err = compile_arm64(&typed, src).expect_err("a float param must be rejected");
        assert!(err[0].message().contains("f64"), "{:?}", err);
    }

    #[test]
    fn more_than_eight_parameters_is_rejected_cleanly() {
        let src = "fn f(a: i64, b: i64, c: i64, d: i64, e: i64, g: i64, h: i64, i: i64, j: i64) -> i64 { a }";
        let typed = typecheck(src);
        let err = compile_arm64(&typed, src).expect_err(">8 params must be rejected");
        assert!(err[0].message().contains("at most 8 parameters"), "{:?}", err);
    }

    #[test]
    fn an_unsupported_statement_is_rejected_cleanly() {
        // a `let` is plan 12-02's job; this plan rejects it with a clean error.
        let src = "fn main() { let x = 1 }";
        let typed = typecheck(src);
        let err = compile_arm64(&typed, src).expect_err("a let must be rejected here");
        assert!(err[0].message().contains("let"), "{:?}", err);
    }

    #[test]
    fn emission_is_deterministic() {
        // compiling the same source twice yields byte-identical text -- the
        // guard on the label counter and any map iteration order.
        let src = "fn f(a: i64, b: i64) -> bool { (a + b) > 0 && a != b }";
        let first = compile_ok(src);
        let second = compile_ok(src);
        assert_eq!(first, second, "arm64 emission must be deterministic");
    }

    #[test]
    fn arithmetic_matches_the_snapshot() {
        let src = "fn arith(a: i64, b: i64) -> i64 { (a + b) * (a - b) / 2 % 7 }";
        let emitted = compile_ok(src);
        assert_eq!(
            emitted,
            read_snapshot("arm64_arithmetic.s"),
            "arm64 arithmetic emission drifted from snapshot"
        );
    }

    #[test]
    fn comparisons_match_the_snapshot() {
        let src = "fn cmp(a: i64, b: i64) -> bool { a == b }\n\
                   fn lt(a: i64, b: i64) -> bool { a < b }\n\
                   fn ge(a: i64, b: i64) -> bool { a >= b }";
        let emitted = compile_ok(src);
        assert_eq!(
            emitted,
            read_snapshot("arm64_comparisons.s"),
            "arm64 comparison emission drifted from snapshot"
        );
    }

    #[test]
    fn booleans_match_the_snapshot() {
        let src = "fn andor(a: bool, b: bool) -> bool { a && b || !a }";
        let emitted = compile_ok(src);
        assert_eq!(
            emitted,
            read_snapshot("arm64_booleans.s"),
            "arm64 boolean emission drifted from snapshot"
        );
    }
}
