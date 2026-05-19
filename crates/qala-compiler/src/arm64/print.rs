//! the `print` / `println` to `printf` lowering for the ARM64 backend.
//!
//! [`Arm64Backend::compile_print_call`] is the third `impl Arm64Backend` block
//! (after `expr.rs` and `stmt.rs`); it lowers a call to the `print` or
//! `println` built-in -- both typed `(str) -> void` -- into an AArch64
//! `printf` call. `expr.rs`'s `compile_call` routes a `print` / `println`
//! callee here instead of treating it as a user function.
//!
//! ## the lowering
//!
//! a `print` / `println` argument is always `str`-typed, and the integer core
//! supports two `str` forms:
//!
//! - a plain string literal ([`TypedExpr::Str`]) -- the zero-hole case. the
//!   `printf` format string is the literal text, there are no argument holes.
//! - a string interpolation ([`TypedExpr::Interpolation`]) -- the parts list
//!   is walked: a [`TypedInterpPart::Literal`] appends its text to the format
//!   string, a [`TypedInterpPart::Expr`] whose type is `i64` appends a `%lld`
//!   conversion and queues the expression as the next `printf` argument.
//!
//! `println` appends a trailing `\n` to the format string; `print` does not.
//! a literal `%` in the interpolation text is doubled to `%%` so `printf`
//! prints one literal percent.
//!
//! every other `str` form -- a string variable, a string concatenation, a
//! `str`-returning call, an interpolation hole that is itself a `str` or any
//! non-`i64` type -- is outside the integer core and is rejected with a clean
//! [`QalaError`]. the integer core has no heap and no string representation;
//! only string literals and `i64`-holed interpolations reach `printf`.
//!
//! ## the emitted call
//!
//! the format string is interned into the `.data` section
//! ([`Asm::intern_format`](super::emit::Asm::intern_format)) and reached with
//! `ldr x0, =<label>`. each queued `i64` hole is evaluated through the Phase
//! 12 expression codegen and spilled to a claimed scratch slot -- the same
//! claim/release discipline `compile_call` uses for a user call's arguments --
//! then the slots are loaded into `x1`, `x2`, ... and `bl printf` is emitted.
//! the format-string pointer takes `x0`, so the AArch64 argument registers
//! `x1`-`x7` hold at most seven holes; an interpolation with more is rejected.
//!
//! Phase 12's `main` prologue keeps `sp` 16-byte aligned for the whole body
//! (`sp` never moves mid-body -- locals are `fp`-relative), so the `bl printf`
//! site is correctly aligned.

use crate::errors::QalaError;
use crate::span::Span;
use crate::typed_ast::{TypedExpr, TypedInterpPart};

use super::Arm64Backend;

/// the maximum number of `i64` interpolation holes a single `print` /
/// `println` supports: the AArch64 integer-argument registers are `x0`-`x7`,
/// `printf`'s format-string pointer takes `x0`, so seven holes (`x1`-`x7`)
/// remain. an interpolation with more holes would need a stack-passed
/// argument, which the integer core does not emit.
const MAX_PRINTF_HOLES: usize = 7;

impl Arm64Backend {
    /// lower a `print` / `println` call to a `printf` call, leaving `printf`'s
    /// (discarded) return value in `x0`.
    ///
    /// `name` is `"print"` or `"println"`; `args` is the call's argument list
    /// (a well-typed `print` / `println` has exactly one `str` argument);
    /// `span` is the call site, carried on any rejection.
    ///
    /// returns a [`QalaError`] -- never a panic -- for a construct the integer
    /// core does not support: a `str` argument that is neither a literal nor
    /// an interpolation, an interpolation hole whose type is not `i64`, or an
    /// interpolation with more than seven holes.
    pub(super) fn compile_print_call(
        &mut self,
        name: &str,
        args: &[TypedExpr],
        span: Span,
    ) -> Result<(), QalaError> {
        // `print` / `println` are typed `(str) -> void`: exactly one argument.
        // a well-typed program always satisfies this; the guard is defensive
        // so a malformed call is a clean error rather than an index panic.
        let [arg] = args else {
            return Err(QalaError::Type {
                span,
                message: format!(
                    "the arm64 backend expects `{name}` to take exactly one argument"
                ),
            });
        };
        // `println` terminates its output with a newline; `print` does not.
        let newline = name == "println";

        // build the printf format string and the ordered list of i64 holes.
        let (format, holes) = self.build_format(arg, newline)?;

        // cap the hole count at the seven argument registers x1-x7.
        if holes.len() > MAX_PRINTF_HOLES {
            return Err(QalaError::Type {
                span,
                message: format!(
                    "the arm64 backend supports at most {MAX_PRINTF_HOLES} \
                     interpolation holes in a `{name}`"
                ),
            });
        }

        self.emit_printf(&format, &holes);
        Ok(())
    }

    /// walk a `print` / `println` argument into a `printf` format string and a
    /// borrowed, ordered list of the `i64` hole expressions.
    ///
    /// a [`TypedExpr::Str`] argument is the zero-hole case -- the format
    /// string is the literal text (percent-escaped), the hole list is empty. a
    /// [`TypedExpr::Interpolation`] argument is walked part by part. any other
    /// argument shape is rejected. `newline` appends a trailing `\n` to the
    /// format string for a `println`.
    fn build_format<'a>(
        &self,
        arg: &'a TypedExpr,
        newline: bool,
    ) -> Result<(String, Vec<&'a TypedExpr>), QalaError> {
        let mut format = String::new();
        let mut holes: Vec<&TypedExpr> = Vec::new();
        match arg {
            // a plain string literal: the format string is the literal text,
            // its percents doubled so printf prints them literally.
            TypedExpr::Str { value, .. } => {
                format.push_str(&escape_percent(value));
            }
            // an interpolation: a literal part appends its (percent-escaped)
            // text, an i64 hole appends a `%lld` conversion and is queued.
            TypedExpr::Interpolation { parts, .. } => {
                for part in parts {
                    match part {
                        TypedInterpPart::Literal(text) => {
                            format.push_str(&escape_percent(text));
                        }
                        TypedInterpPart::Expr(expr) => {
                            // only an i64 hole can become a `%lld`. a str, an
                            // f64, a bool -- any non-i64 hole -- is outside the
                            // integer core and is rejected.
                            if !matches!(expr.ty(), crate::types::QalaType::I64) {
                                return Err(QalaError::Type {
                                    span: expr.span(),
                                    message: format!(
                                        "the arm64 backend does not yet support \
                                         {} values in interpolation",
                                        super::type_name(expr.ty())
                                    ),
                                });
                            }
                            format.push_str("%lld");
                            holes.push(expr);
                        }
                    }
                }
            }
            // any other str-valued expression -- a string variable, a string
            // concatenation, a str-returning call -- has no integer-core
            // representation and is rejected.
            other => {
                return Err(QalaError::Type {
                    span: other.span(),
                    message: format!(
                        "the arm64 backend does not yet support {} as a \
                         `print` / `println` argument",
                        print_arg_description(other)
                    ),
                });
            }
        }
        // a println terminates with a newline.
        if newline {
            format.push('\n');
        }
        Ok((format, holes))
    }

    /// emit the `printf` call: intern the format string, evaluate every hole
    /// into an argument register, and `bl printf`.
    ///
    /// each hole is evaluated into `x0` through the Phase 12 expression codegen
    /// and spilled to a freshly claimed scratch slot -- evaluating hole `k+1`
    /// into `x0` would clobber hole `k`, exactly the clobber `compile_call`
    /// avoids for a user call's arguments. once every hole is in a slot the
    /// slots are loaded into `x1`, `x2`, ..., the format-string pointer is
    /// loaded into `x0` with `ldr x0, =<label>`, and `bl printf` is emitted.
    /// the claimed slots are released after the call.
    fn emit_printf(&mut self, format: &str, holes: &[&TypedExpr]) {
        let label = self.asm.intern_format(format);

        // evaluate each hole into x0, then spill it to a claimed scratch slot.
        // the slot is claimed AFTER the hole is evaluated, so a hole that is
        // itself a call releases its own argument slots before this persistent
        // slot is taken -- the same discipline `compile_call` documents.
        let mut hole_slots = Vec::with_capacity(holes.len());
        for (i, hole) in holes.iter().enumerate() {
            // a hole reaching here is i64-typed; compile_expr cannot fail on an
            // integer-core expression, but a defensive `?` keeps a deeper
            // unsupported sub-construct a clean error rather than a panic.
            if let Err(e) = self.compile_expr(hole) {
                self.errors.push(e);
                return;
            }
            let slot = self.frame_mut().claim_scratch();
            self.asm.emit_insn_commented(
                &format!("str     x0, [fp, {slot}]"),
                &format!("printf arg {}", i + 1),
            );
            hole_slots.push(slot);
        }
        // load the holes into x1..x{n}: argument register i+1 holds hole i,
        // since x0 is reserved for the format-string pointer.
        for (i, slot) in hole_slots.iter().enumerate() {
            self.asm
                .emit_insn(&format!("ldr     x{}, [fp, {slot}]", i + 1));
        }
        // the format-string pointer in x0, then the call.
        self.asm
            .emit_insn_commented(&format!("ldr     x0, ={label}"), "printf format");
        self.asm.emit_insn("bl      printf");
        // release the hole slots -- one per hole, balancing the claims above.
        for _ in &hole_slots {
            self.frame_mut().release_scratch();
        }
    }
}

/// double every `%` in interpolation text so `printf` prints a literal percent.
///
/// `printf` reads `%` as the start of a conversion; a literal percent in the
/// program's string text must therefore be written `%%` in the format string.
/// the `%lld` conversions the lowering itself appends are added separately and
/// never pass through here, so this only ever sees the program's own text.
fn escape_percent(text: &str) -> String {
    text.replace('%', "%%")
}

/// a human description of a rejected `print` / `println` argument, for the
/// unsupported-construct diagnostic.
///
/// the integer core prints a string literal and an `i64`-holed interpolation;
/// every other `str`-valued expression is rejected, and this names the shape
/// so the message is specific: a string variable, a string concatenation, a
/// string-returning call.
fn print_arg_description(expr: &TypedExpr) -> &'static str {
    match expr {
        TypedExpr::Ident { .. } => "a string variable",
        TypedExpr::Binary { .. } => "string concatenation",
        TypedExpr::Call { .. } | TypedExpr::MethodCall { .. } => {
            "a string-returning call"
        }
        TypedExpr::Paren { .. } => "a parenthesized string expression",
        TypedExpr::Block { .. } => "a block expression",
        TypedExpr::OrElse { .. } => "an `or` fallback expression",
        TypedExpr::Match { .. } => "a match expression",
        TypedExpr::FieldAccess { .. } => "a string field access",
        TypedExpr::Index { .. } => "a string index expression",
        // a catch-all: any other str-typed expression the integer core does
        // not lower. the specific arms above name the shapes a program is
        // likely to write; this keeps the message honest for the rest.
        _ => "this string expression",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;
    use crate::parser::Parser;
    use crate::typechecker::check_program;
    use crate::typed_ast::TypedItem;

    /// compile a whole program to assembly, panicking on a backend error.
    fn compile_ok(src: &str) -> String {
        let tokens = Lexer::tokenize(src).expect("lex failed");
        let ast = Parser::parse(&tokens).expect("parse failed");
        let (typed, terrors, _) = check_program(&ast, src);
        assert!(terrors.is_empty(), "typecheck errors: {terrors:?}");
        super::super::compile_arm64(&typed, src)
            .unwrap_or_else(|e| panic!("arm64 errors: {e:?}"))
    }

    /// compile a program expected to be rejected, returning the first error's
    /// message.
    fn compile_err(src: &str) -> String {
        let tokens = Lexer::tokenize(src).expect("lex failed");
        let ast = Parser::parse(&tokens).expect("parse failed");
        let (typed, terrors, _) = check_program(&ast, src);
        assert!(terrors.is_empty(), "typecheck errors: {terrors:?}");
        let errors = super::super::compile_arm64(&typed, src)
            .expect_err("the program must be rejected by the backend");
        errors[0].message()
    }

    #[test]
    fn escape_percent_doubles_a_literal_percent() {
        assert_eq!(escape_percent("100% sure"), "100%% sure");
        assert_eq!(escape_percent("no percent here"), "no percent here");
        assert_eq!(escape_percent("%%"), "%%%%");
    }

    #[test]
    fn a_string_literal_println_emits_a_printf_with_no_holes() {
        // `println("done")` -- the zero-hole case. the format string is the
        // literal text plus a newline, in .data; printf is called with just
        // the format-string pointer in x0.
        let out = compile_ok("fn main() is io { println(\"done\") }");
        assert!(out.contains("        .data\n"), "missing .data section: {out}");
        assert!(
            out.contains(".string \"done\\n\"\n"),
            "missing the println format string: {out}"
        );
        assert!(out.contains("ldr     x0, =.Lfmt_0"), "missing the format load: {out}");
        assert!(out.contains("bl      printf"), "missing the printf call: {out}");
    }

    #[test]
    fn a_string_literal_print_omits_the_trailing_newline() {
        // `print` -- unlike `println` -- adds no newline to the format string.
        let out = compile_ok("fn main() is io { print(\"hi\") }");
        assert!(out.contains(".string \"hi\"\n"), "print must not add a newline: {out}");
        assert!(!out.contains(".string \"hi\\n\""), "print added a newline: {out}");
    }

    #[test]
    fn an_interpolation_hole_becomes_a_percent_lld_and_an_argument() {
        // `println("{x}")` -- a single i64 hole. the format string carries one
        // `%lld`; the hole is evaluated and loaded into x1.
        let out = compile_ok("fn main() is io { let x = 7\nprintln(\"{x}\") }");
        assert!(out.contains(".string \"%lld\\n\"\n"), "missing the %lld format: {out}");
        assert!(
            out.contains("str     x0, [fp,") && out.contains("// printf arg 1"),
            "missing the hole spill: {out}"
        );
        assert!(out.contains("ldr     x1, [fp,"), "missing the x1 load: {out}");
        assert!(out.contains("ldr     x0, =.Lfmt_0"), "missing the format load: {out}");
        assert!(out.contains("bl      printf"), "missing the printf call: {out}");
    }

    #[test]
    fn the_fibonacci_interpolation_builds_a_two_hole_format() {
        // the fibonacci.qala line: `println("fib({i}) = {fibonacci(i)}")`.
        // two i64 holes -> "fib(%lld) = %lld\n" and two argument registers.
        let out = compile_ok(
            "fn fib(n: i64) -> i64 { n }\n\
             fn main() is io {\n\
             \x20   for i in 0..3 { println(\"fib({i}) = {fib(i)}\") }\n\
             }",
        );
        assert!(
            out.contains(".string \"fib(%lld) = %lld\\n\"\n"),
            "the two-hole format string is wrong: {out}"
        );
        // two holes -> x1 and x2 both loaded.
        assert!(out.contains("ldr     x1, [fp,"), "missing the x1 load: {out}");
        assert!(out.contains("ldr     x2, [fp,"), "missing the x2 load: {out}");
        assert!(out.contains("bl      printf"), "missing the printf call: {out}");
    }

    #[test]
    fn the_hole_arguments_are_loaded_after_every_spill() {
        // the clobber trap: hole 2's evaluation lands in x0 and would overwrite
        // hole 1 if hole 1 were already in a register. every `str` spill must
        // precede the first `ldr` of an argument register.
        let out = compile_ok(
            "fn fib(n: i64) -> i64 { n }\n\
             fn main() is io {\n\
             \x20   for i in 0..3 { println(\"fib({i}) = {fib(i)}\") }\n\
             }",
        );
        let lines: Vec<&str> = out.lines().collect();
        let last_spill = lines
            .iter()
            .rposition(|l| l.contains("str     x0, [fp,") && l.contains("// printf arg "))
            .expect("no hole spill");
        let first_arg_load = lines
            .iter()
            .position(|l| l.contains("ldr     x1, [fp,"))
            .expect("no x1 load");
        assert!(
            last_spill < first_arg_load,
            "every hole spill must precede the first argument load: {out}"
        );
    }

    #[test]
    fn a_literal_percent_in_the_text_is_doubled() {
        // a literal `%` in the program's string must reach printf as `%%`.
        let out = compile_ok("fn main() is io { println(\"100% done\") }");
        assert!(
            out.contains(".string \"100%% done\\n\"\n"),
            "a literal percent must be doubled to %%: {out}"
        );
    }

    #[test]
    fn two_identical_format_strings_share_one_data_entry() {
        // two `println("done")` calls intern the same format string -- one
        // .data entry, one label, reused.
        let out = compile_ok(
            "fn main() is io { println(\"done\")\nprintln(\"done\") }",
        );
        assert_eq!(
            out.matches(".Lfmt_0:").count(),
            1,
            "an identical format string must be interned once: {out}"
        );
        assert!(!out.contains(".Lfmt_1:"), "no second entry for the same string: {out}");
    }

    #[test]
    fn main_returns_zero_after_a_printf() {
        // `main` is void: it zeroes x0 before the epilogue, so the assembled
        // program exits success even after a printf left a byte count in x0.
        let out = compile_ok("fn main() is io { println(\"done\") }");
        assert!(
            out.contains("mov     x0, 0  // void return -> exit code 0"),
            "main must return 0 after the printf: {out}"
        );
    }

    #[test]
    fn a_non_i64_interpolation_hole_is_rejected_cleanly() {
        // a bool hole -- `{b}` where b is a bool -- is not an i64, so it cannot
        // become a `%lld`. a clean rejection naming the type, never a panic.
        let msg = compile_err("fn main() is io { let b = true\nprintln(\"{b}\") }");
        assert!(
            msg.contains("bool") && msg.contains("interpolation"),
            "the rejection must name the bool hole: {msg}"
        );
    }

    #[test]
    fn a_float_interpolation_hole_is_rejected_cleanly() {
        // an f64 hole is outside the integer core -- a clean rejection. the
        // typed AST is built directly: a float reaching the backend through a
        // `let` would be rejected at the `let` first, so the interpolation
        // node is constructed here to exercise the hole-type check itself.
        use crate::types::QalaType;
        let interp = TypedExpr::Interpolation {
            parts: vec![TypedInterpPart::Expr(TypedExpr::Float {
                value: 1.5,
                ty: QalaType::F64,
                span: Span::new(0, 3),
            })],
            ty: QalaType::Str,
            span: Span::new(0, 5),
        };
        let mut backend = Arm64Backend::new("");
        let err = backend
            .compile_print_call("println", std::slice::from_ref(&interp), Span::new(0, 5))
            .expect_err("an f64 hole must be rejected");
        match err {
            QalaError::Type { message, .. } => assert!(
                message.contains("f64") && message.contains("interpolation"),
                "the rejection must name the f64 hole: {message}"
            ),
            other => panic!("expected QalaError::Type, got {other:?}"),
        }
    }

    #[test]
    fn a_string_variable_argument_is_rejected_cleanly() {
        // `println(s)` where `s` is a str variable -- not a literal, not an
        // interpolation. the integer core has no string values; a clean
        // rejection naming the construct. the typed AST is built directly: a
        // str `let` would be rejected at the `let` first, so the str-typed
        // identifier argument is constructed here to exercise build_format.
        use crate::types::QalaType;
        let ident = TypedExpr::Ident {
            name: "s".to_string(),
            ty: QalaType::Str,
            span: Span::new(0, 1),
        };
        let mut backend = Arm64Backend::new("");
        let err = backend
            .compile_print_call("println", std::slice::from_ref(&ident), Span::new(0, 5))
            .expect_err("a string variable argument must be rejected");
        match err {
            QalaError::Type { message, .. } => assert!(
                message.contains("string variable") && message.contains("arm64"),
                "the rejection must name the string variable: {message}"
            ),
            other => panic!("expected QalaError::Type, got {other:?}"),
        }
    }

    #[test]
    fn a_string_concatenation_argument_is_rejected_cleanly() {
        // `println("a" + "b")` -- a string concatenation reaches the print
        // lowering as a Binary, outside the integer core.
        let msg = compile_err("fn main() is io { println(\"a\" + \"b\") }");
        assert!(
            msg.contains("string concatenation"),
            "the rejection must name the concatenation: {msg}"
        );
    }

    #[test]
    fn more_than_seven_holes_is_rejected_cleanly() {
        // eight i64 holes need eight argument registers -- x0 is the format
        // pointer, leaving only x1-x7. a clean rejection, never a panic.
        let src = "fn main() is io {\n\
                   \x20   let a = 1\n\
                   \x20   println(\"{a}{a}{a}{a}{a}{a}{a}{a}\")\n\
                   }";
        let msg = compile_err(src);
        assert!(
            msg.contains("at most 7") && msg.contains("holes"),
            "more than seven holes must be rejected: {msg}"
        );
    }

    #[test]
    fn seven_holes_is_accepted() {
        // exactly seven holes fits x1-x7 -- the boundary case is supported.
        let src = "fn main() is io {\n\
                   \x20   let a = 1\n\
                   \x20   println(\"{a}{a}{a}{a}{a}{a}{a}\")\n\
                   }";
        let out = compile_ok(src);
        assert!(out.contains("ldr     x7, [fp,"), "the seventh hole must load x7: {out}");
        assert!(!out.contains("ldr     x8, [fp,"), "there is no eighth argument register: {out}");
    }

    #[test]
    fn the_emitted_program_has_a_text_section_and_a_main() {
        // a sanity check that the printf path produces a complete program: the
        // preamble, the .data format string, the .text section, and `main`.
        let out = compile_ok("fn main() is io { println(\"done\") }");
        assert!(out.starts_with("define(fp, x29)\n"), "missing the m4 preamble: {out}");
        assert!(out.contains("        .text\n"), "missing the .text section: {out}");
        assert!(out.contains("\nmain:\n"), "missing the main label: {out}");
    }

    #[test]
    fn a_print_call_is_not_treated_as_a_user_function() {
        // `print` / `println` must route to the printf lowering, never to the
        // user-function `bl` path -- there is no `bl print` / `bl println`.
        // the only `bl` the printf path emits is `bl printf` itself.
        let out = compile_ok("fn main() is io { println(\"done\") }");
        let bl_targets: Vec<&str> = out
            .lines()
            .filter_map(|l| l.trim().strip_prefix("bl      "))
            .collect();
        assert_eq!(
            bl_targets,
            vec!["printf"],
            "the only call the println path emits is `bl printf`: {out}"
        );
    }

    #[test]
    fn a_hole_that_is_a_call_evaluates_before_the_printf() {
        // `println("{fib(3)}")` -- the hole is a call. the call's own `bl`
        // must run before `bl printf`, leaving the result spilled as a hole.
        let out = compile_ok(
            "fn fib(n: i64) -> i64 { n }\n\
             fn main() is io { println(\"{fib(3)}\") }",
        );
        let lines: Vec<&str> = out.lines().collect();
        let bl_fib = lines
            .iter()
            .position(|l| l.contains("bl      fib"))
            .expect("missing bl fib");
        let bl_printf = lines
            .iter()
            .position(|l| l.contains("bl      printf"))
            .expect("missing bl printf");
        assert!(bl_fib < bl_printf, "the hole's call must run before printf: {out}");
    }

    #[test]
    fn typed_items_have_no_extra_emission_for_the_printf_path() {
        // a struct alongside a printing main: the struct emits nothing, the
        // printf path is unaffected.
        let out = compile_ok(
            "struct P { x: i64 }\nfn main() is io { println(\"done\") }",
        );
        assert!(out.contains("bl      printf"), "the printf call is missing: {out}");
        assert!(!out.contains("\nP:\n"), "the struct must emit nothing: {out}");
    }
}
