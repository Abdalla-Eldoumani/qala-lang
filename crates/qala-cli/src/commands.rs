//! the `run`, `check`, and `build` subcommand handlers.
//!
//! each handler returns a [`std::process::ExitCode`]: `SUCCESS` (0) when the
//! command did its job, `from(1)` on any qala-level failure. every error path
//! renders a diagnostic to stderr -- never a bare panic, never an unfinished
//! placeholder. program output goes to stdout; diagnostics and the arm64 stub
//! go to stderr.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use qala_compiler::diagnostics::Diagnostic;
use qala_compiler::vm::Vm;

use crate::Target;
use crate::diagnostics::{render_errors, render_warnings};
use crate::pipeline;

/// compile and run a qala program: print its console output to stdout, exit 0.
///
/// on a file-read failure, a compile error, or a runtime fault, render the
/// diagnostic to stderr and exit non-zero. warnings render to stderr but do
/// not block -- the VM still runs.
pub fn run(file: &Path) -> ExitCode {
    let src = match std::fs::read_to_string(file) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: cannot read '{}': {e}", file.display());
            return ExitCode::from(1);
        }
    };

    let compiled = match pipeline::compile_source(&src) {
        Ok(c) => c,
        Err(errors) => {
            eprint!("{}", render_errors(&errors, &src));
            return ExitCode::from(1);
        }
    };

    // warnings never block; render them to stderr before running.
    if !compiled.warnings.is_empty() {
        eprint!("{}", render_warnings(&compiled.warnings, &src));
    }

    let mut vm = Vm::new(compiled.program, src.clone());
    match vm.run() {
        Ok(()) => {
            // program output: VmState.console, one entry per line, to stdout.
            for line in vm.get_state().console {
                println!("{line}");
            }
            ExitCode::SUCCESS
        }
        Err(err) => {
            // a runtime fault: render to stderr, exit non-zero.
            eprint!("{}", Diagnostic::from(err).render(&src));
            ExitCode::from(1)
        }
    }
}

/// type-check a qala program without running it: lex + parse + typecheck only.
///
/// renders errors and warnings to stderr. exits 0 if it type-checks, non-zero
/// on a type error. never codegens and never runs the VM, so no program output
/// reaches stdout.
pub fn check(file: &Path) -> ExitCode {
    let src = match std::fs::read_to_string(file) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: cannot read '{}': {e}", file.display());
            return ExitCode::from(1);
        }
    };

    // stages 1-3 only -- never compile_source, never a Vm.
    let (errors, warnings) = pipeline::check_source(&src);
    if !errors.is_empty() {
        eprint!("{}", render_errors(&errors, &src));
    }
    if !warnings.is_empty() {
        eprint!("{}", render_warnings(&warnings, &src));
    }

    if errors.is_empty() {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}

/// compile a qala program and write its output artifact.
///
/// `--target bytecode` (the default) writes the optimized bytecode disassembly
/// to a `.qbc` file. `--target arm64` is recognized but its code generator is
/// not yet available -- it renders a clear diagnostic to stderr and exits
/// non-zero before any compilation.
pub fn build(file: &Path, target: Target, output: Option<PathBuf>) -> ExitCode {
    // check the target first: arm64 routes to the not-yet-available stub
    // before reading or compiling anything.
    if target == Target::Arm64 {
        eprintln!("error: the arm64 backend is not yet available");
        eprintln!(
            "  = note: `--target arm64` is a recognized target but its code generator ships in a later release"
        );
        eprintln!(
            "  = hint: use the default `--target bytecode`, or omit --target, to compile this program now"
        );
        return ExitCode::from(1);
    }

    let src = match std::fs::read_to_string(file) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: cannot read '{}': {e}", file.display());
            return ExitCode::from(1);
        }
    };

    let compiled = match pipeline::compile_source(&src) {
        Ok(c) => c,
        Err(errors) => {
            eprint!("{}", render_errors(&errors, &src));
            return ExitCode::from(1);
        }
    };

    // the artifact path: the -o override, else the source path with a .qbc
    // extension. the artifact is the optimized bytecode disassembly listing.
    let out_path = match output {
        Some(p) => p,
        None => file.with_extension("qbc"),
    };
    let listing = compiled.program.disassemble();
    if let Err(e) = std::fs::write(&out_path, listing) {
        eprintln!("error: cannot write '{}': {e}", out_path.display());
        return ExitCode::from(1);
    }
    // a confirmation line -- not an error, so it goes to stdout.
    println!("wrote {}", out_path.display());
    ExitCode::SUCCESS
}
