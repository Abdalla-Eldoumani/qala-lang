//! the interactive qala REPL -- a reedline loop driving a persistent VM.

use std::path::PathBuf;

use qala_compiler::vm::Vm;
use reedline::{DefaultPrompt, FileBackedHistory, Reedline, Signal};

/// run the interactive qala REPL. returns the process exit code.
pub fn run_repl() -> std::process::ExitCode {
    // a persistent REPL VM: bindings and console output carry across lines.
    let mut vm = Vm::new_repl();

    // file-backed history under the user's home dir; an in-memory history
    // is the fallback if the path cannot be opened.
    let history = history_path()
        .and_then(|p| FileBackedHistory::with_file(500, p).ok())
        .map(Box::new);

    let mut editor = match history {
        Some(h) => Reedline::create().with_history(h),
        None => Reedline::create(),
    };
    let prompt = DefaultPrompt::default();

    println!(
        "qala {} repl -- type a line, ctrl-d to exit",
        env!("CARGO_PKG_VERSION")
    );

    loop {
        match editor.read_line(&prompt) {
            Ok(Signal::Success(line)) => {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                // optional explicit-quit command in addition to ctrl-d.
                if trimmed == ":quit" || trimmed == ":q" {
                    break;
                }
                // evaluate the line against the persistent VM.
                match vm.repl_eval(trimmed) {
                    // render the result through the pub render_value method.
                    Ok(value) => {
                        let (rendered, type_name) = vm.render_value(value);
                        // void lines render as "()": still printed; harmless.
                        println!("{rendered} : {type_name}");
                    }
                    // a failed line: render the diagnostic, do NOT exit.
                    // the VM already does not add a failed line to history.
                    Err(err) => {
                        eprint!(
                            "{}",
                            qala_compiler::diagnostics::Diagnostic::from(err).render("")
                        );
                    }
                }
            }
            // ctrl-d ends the session cleanly.
            Ok(Signal::CtrlD) => break,
            // ctrl-c abandons the current line, keeps the session.
            Ok(Signal::CtrlC) => continue,
            // any other signal: end the session.
            Ok(_) => break,
            // a reedline IO error: report and exit non-zero.
            Err(e) => {
                eprintln!("repl error: {e}");
                return std::process::ExitCode::from(1);
            }
        }
    }
    std::process::ExitCode::SUCCESS
}

/// locate the REPL history file: `.qala_history` under the user's home
/// directory. tries `HOME`, then `USERPROFILE` for Windows. returns `None`
/// when no home directory is found -- the caller falls back to in-memory
/// history rather than failing the REPL.
fn history_path() -> Option<PathBuf> {
    std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .ok()
        .map(|home| PathBuf::from(home).join(".qala_history"))
}
