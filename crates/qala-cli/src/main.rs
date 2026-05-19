//! the `qala` command-line interface.
//!
//! a `clap`-based front end for the qala compiler: `run` compiles and executes
//! a program, `check` type-checks it without running, `build` writes a compiled
//! artifact. `clap` generates `--version` and `--help`. the compiler does the
//! work; this crate owns argument parsing, file IO, the stdout/stderr split,
//! and exit codes.

mod commands;
mod diagnostics;
mod pipeline;
mod repl;

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand, ValueEnum};

/// the qala command-line compiler.
#[derive(Parser)]
#[command(name = "qala", version, about = "the qala programming language compiler")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

/// the qala subcommands.
#[derive(Subcommand)]
enum Commands {
    /// compile and run a qala program
    Run {
        /// the .qala source file
        file: PathBuf,
    },
    /// type-check a qala program without running it
    Check {
        /// the .qala source file
        file: PathBuf,
    },
    /// compile a qala program and write its output artifact
    Build {
        /// the .qala source file
        file: PathBuf,
        /// the compilation target
        #[arg(long, value_enum, default_value_t = Target::Bytecode)]
        target: Target,
        /// the output file path (defaults to the source path with a new extension)
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    /// open an interactive qala REPL
    Repl,
}

/// the qala build targets.
#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum Target {
    /// the bytecode VM -- writes an optimized bytecode disassembly listing
    Bytecode,
    /// the ARM64 assembly backend (not available until a later release)
    Arm64,
}

/// parse the command line and dispatch to the chosen subcommand handler.
fn main() -> ExitCode {
    let cli = Cli::parse(); // clap auto-exits with code 2 on a usage error
    match cli.command {
        Commands::Run { file } => commands::run(&file),
        Commands::Check { file } => commands::check(&file),
        Commands::Build {
            file,
            target,
            output,
        } => commands::build(&file, target, output),
        Commands::Repl => repl::run_repl(),
    }
}
