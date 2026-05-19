//! the CLI's compile pipeline -- the twin of `wasm.rs::compile_core`.

use qala_compiler::chunk::Program;
use qala_compiler::codegen;
use qala_compiler::errors::QalaError;
use qala_compiler::lexer::Lexer;
use qala_compiler::parser::Parser;
use qala_compiler::typechecker::{self, QalaWarning};

/// the result of a successful compile: the optimized program plus any warnings.
pub struct Compiled {
    /// the optimized bytecode program, ready for `Vm::new` or `disassemble`.
    pub program: Program,
    /// non-blocking warnings the typechecker produced.
    pub warnings: Vec<QalaWarning>,
}

/// run the full compile pipeline on `src`: lex -> parse -> typecheck -> codegen
/// -> optimize. errors block and are returned as `Err`; warnings never block.
///
/// mirrors `qala_compiler::wasm`'s `compile_core`. on the first lex, parse,
/// type, or codegen error it stops and returns every error from that stage.
/// the returned `Program` is already optimized.
pub fn compile_source(src: &str) -> Result<Compiled, Vec<QalaError>> {
    // stage 1: lex. a single QalaError on failure.
    let tokens = Lexer::tokenize(src).map_err(|e| vec![e])?;
    // stage 2: parse. a single QalaError on failure.
    let ast = Parser::parse(&tokens).map_err(|e| vec![e])?;
    // stage 3: typecheck. errors block; warnings never block.
    let (typed, errors, warnings) = typechecker::check_program(&ast, src);
    if !errors.is_empty() {
        return Err(errors);
    }
    // stage 4: codegen. a Vec<QalaError> on failure.
    let mut program = codegen::compile_program(&typed, src)?;
    // stage 5: optimize in place.
    program.optimize();
    Ok(Compiled { program, warnings })
}

/// lex + parse + typecheck only -- for `qala check`. returns errors and
/// warnings; does NOT codegen or run.
pub fn check_source(src: &str) -> (Vec<QalaError>, Vec<QalaWarning>) {
    let tokens = match Lexer::tokenize(src) {
        Ok(t) => t,
        Err(e) => return (vec![e], Vec::new()),
    };
    let ast = match Parser::parse(&tokens) {
        Ok(a) => a,
        Err(e) => return (vec![e], Vec::new()),
    };
    let (_typed, errors, warnings) = typechecker::check_program(&ast, src);
    (errors, warnings)
}
