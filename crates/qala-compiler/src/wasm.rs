//! the wasm-bindgen bridge: a `Qala` session struct that exposes the finished
//! compile-and-VM pipeline to JavaScript.
//!
//! the module is two layers. the lower layer is plain Rust -- the `Qala`
//! struct's state plus seven core methods that return typed serde structs
//! directly. the native test suite drives this layer, since the core methods
//! need no JavaScript runtime. the upper layer is the thin `#[wasm_bindgen]`
//! impl block whose seven methods each call the matching core method and hand
//! the result to `serde_wasm_bindgen::to_value`, returning a `JsValue`.
//!
//! the bridge never throws and never panics. a failed compile, a runtime
//! error, and misuse such as `run` before `compile` are each structured data
//! in the returned value carrying an `ok: false` or `status: "error"` flag --
//! the playground branches on that flag and never needs a `try`/`catch`. a
//! panic in WASM aborts the browser tab, so the no-panic discipline is the
//! boundary contract; the panic hook installed in `new` is a bug-report
//! safety net only.

use wasm_bindgen::prelude::*;

/// the result of `compile`. read by the playground's editor (the diagnostic
/// underlines) and its bytecode panel (the disassembly).
///
/// the flat ok-flag shape rather than a serde-tagged enum: the playground
/// branches on `ok` and reads a plain JavaScript object. derives
/// `serde::Serialize` so the WASM bridge hands it straight to JS via
/// `serde-wasm-bindgen`.
#[derive(Debug, Clone, serde::Serialize)]
pub struct CompileResult {
    /// true on a clean compile (warnings are allowed), false on any error.
    pub ok: bool,
    /// the optimized bytecode disassembly; `Some` on success, `None` on error.
    pub disassembly: Option<String>,
    /// warnings on success, errors on failure -- `MonacoDiagnostic::severity`
    /// distinguishes the two so the playground renders both from one array.
    pub diagnostics: Vec<crate::diagnostics::MonacoDiagnostic>,
}

/// the result of `compile_arm64`. read by the playground's assembly view.
///
/// the same flat ok-flag shape as `CompileResult`: the playground branches
/// on `ok`. derives `serde::Serialize` so the WASM bridge hands it straight
/// to JS via `serde-wasm-bindgen`.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Arm64Result {
    /// true when the ARM64 backend produced assembly, false on any error
    /// (a front-end error, or an unsupported-construct rejection).
    pub ok: bool,
    /// the AArch64 assembly text; `Some` on success, `None` on error.
    pub assembly: Option<String>,
    /// the diagnostics on failure -- an unsupported-construct rejection or a
    /// front-end error -- empty on success.
    pub diagnostics: Vec<crate::diagnostics::MonacoDiagnostic>,
}

/// the result of `run`. read by the playground's stack, variables, and
/// console panels (the `state`) and its editor (the runtime-error underline).
///
/// derives `serde::Serialize` so the WASM bridge hands it straight to JS.
#[derive(Debug, Clone, serde::Serialize)]
pub struct RunResult {
    /// true when the program ran to `Halt`, false on a runtime error or when
    /// no program is compiled.
    pub ok: bool,
    /// the VM state at the stopping point -- on a runtime error this is the
    /// fault-point snapshot, so the playground can show the stack and console
    /// where execution stopped.
    pub state: crate::vm::VmState,
    /// the runtime-error diagnostic; `Some` on a fault, `None` otherwise.
    pub error: Option<crate::diagnostics::MonacoDiagnostic>,
}

/// the result of `step`. read by the playground's step-through after each
/// instruction.
///
/// derives `serde::Serialize` so the WASM bridge hands it straight to JS.
#[derive(Debug, Clone, serde::Serialize)]
pub struct StepResult {
    /// `"ran"` when an ordinary instruction executed, `"halted"` when the VM
    /// reached `Halt`, `"error"` on a runtime fault or when no program is
    /// compiled. a plain string so the JavaScript side reads the discriminant
    /// directly, without serde enum tagging.
    pub status: String,
    /// the VM state after the step (or the fault-point snapshot on an error).
    pub state: crate::vm::VmState,
    /// the runtime-error diagnostic; `Some` when `status` is `"error"`,
    /// `None` otherwise.
    pub error: Option<crate::diagnostics::MonacoDiagnostic>,
}

/// the result of one REPL evaluation. read by the playground's REPL console.
///
/// derives `serde::Serialize` so the WASM bridge hands it straight to JS.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ReplResult {
    /// true when the line compiled and evaluated, false on any error.
    pub ok: bool,
    /// the evaluated line's value, rendered and type-tagged; `Some` on
    /// success, `None` on an error.
    pub value: Option<crate::vm::StateValue>,
    /// the persistent REPL console output, accumulated across every prior
    /// REPL call.
    pub console: Vec<String>,
    /// the error diagnostic; `Some` on a failed line, `None` otherwise.
    pub error: Option<crate::diagnostics::MonacoDiagnostic>,
}

/// the no-program-compiled `VmState`: a snapshot with every field zero or
/// empty.
///
/// `VmState` has no public constructor and `Vm::get_state` needs a VM, so the
/// misuse paths (run/step/get_state before a successful compile) build this
/// literal directly -- every `VmState` field is `pub`. the playground can
/// always read `.stack`, `.console`, and the rest off the returned value.
fn empty_state() -> crate::vm::VmState {
    crate::vm::VmState {
        chunk_index: 0,
        ip: 0,
        current_line: 0,
        stack: Vec::new(),
        variables: Vec::new(),
        console: Vec::new(),
        leak_log: Vec::new(),
    }
}

/// the diagnostic the misuse paths return when no program is compiled yet.
///
/// built as a struct literal -- every `MonacoDiagnostic` field is `pub`. the
/// message is plain so the playground can show it verbatim; line and column
/// are 1 (a harmless position, since this diagnostic underlines nothing) and
/// `severity` is 1 (error).
fn no_program_diagnostic() -> crate::diagnostics::MonacoDiagnostic {
    crate::diagnostics::MonacoDiagnostic {
        line: 1,
        column: 1,
        end_line: 1,
        end_column: 1,
        severity: 1,
        message: "no program compiled -- call compile() first".to_string(),
        category: None,
    }
}

/// build the failed-compile `CompileResult` from a list of errors.
///
/// each `QalaError` converts to a `MonacoDiagnostic` through the existing
/// diagnostics path: `Diagnostic::from(err)` then `.to_monaco(src)`. the
/// source string is needed to translate the error's byte span into a 1-based
/// line and column.
fn compile_failed(errors: Vec<crate::errors::QalaError>, src: &str) -> CompileResult {
    CompileResult {
        ok: false,
        disassembly: None,
        diagnostics: errors
            .into_iter()
            .map(|e| crate::diagnostics::Diagnostic::from(e).to_monaco(src))
            .collect(),
    }
}

/// build the failed `Arm64Result` from a list of errors -- a front-end
/// error or the backend's unsupported-construct rejection.
///
/// each `QalaError` converts to a `MonacoDiagnostic` through the same
/// diagnostics path `compile_failed` uses: `Diagnostic::from(err)` then
/// `.to_monaco(src)`. the source string translates the error's byte span
/// into a 1-based line and column.
fn arm64_failed(errors: Vec<crate::errors::QalaError>, src: &str) -> Arm64Result {
    Arm64Result {
        ok: false,
        assembly: None,
        diagnostics: errors
            .into_iter()
            .map(|e| crate::diagnostics::Diagnostic::from(e).to_monaco(src))
            .collect(),
    }
}

/// serialize a result struct to a `JsValue`.
///
/// on the practically unreachable serialization failure -- the result structs
/// carry only `bool`, `String`, `Option`, `Vec`, and nested plain structs, so
/// `serde_wasm_bindgen::to_value` does not fail for them -- this falls back to
/// `JsValue::NULL` so the calling method still never panics and never throws.
/// `JsValue::NULL` is the zero-dependency fallback; a richer error object
/// would mean declaring `js-sys` directly.
fn to_js<T: serde::Serialize>(value: &T) -> JsValue {
    serde_wasm_bindgen::to_value(value).unwrap_or(JsValue::NULL)
}

/// the Qala browser session: one JavaScript-side object holding every piece
/// of bridge state across calls.
///
/// the playground constructs one of these per browser tab, then calls methods
/// on the returned object. the struct carries `#[wasm_bindgen]` so it crosses
/// the boundary as a JavaScript class.
#[wasm_bindgen]
pub struct Qala {
    /// the last successfully compiled and optimized program; `None` until the
    /// first clean `compile`. `disassemble` reads it.
    program: Option<crate::chunk::Program>,
    /// the run-and-step VM over `program`, built fresh on each clean compile;
    /// `None` until then. `run` and `step` advance it; `get_state` reads it.
    vm: Option<crate::vm::Vm>,
    /// the persistent REPL VM. constructed once by `Vm::new_repl` and kept for
    /// the session so a binding or console line from one REPL call is visible
    /// on the next.
    repl_vm: crate::vm::Vm,
    /// the source text of the last successful compile. kept so a runtime-error
    /// diagnostic renders its source line against the exact text that produced
    /// the bytecode.
    last_src: String,
}

impl Default for Qala {
    /// a default session is a fresh one -- delegates to [`Qala::new`].
    ///
    /// `new` carries `#[wasm_bindgen(constructor)]` and is the real entry
    /// point; this impl exists so a default-constructible bound is satisfiable
    /// in plain Rust.
    fn default() -> Self {
        Self::new()
    }
}

impl Qala {
    /// run the full compile pipeline on `source` and return a `CompileResult`.
    ///
    /// the pipeline is lex, parse, typecheck, codegen, then `Program::optimize`
    /// followed by `Program::disassemble`; the disassembly is the optimized
    /// bytecode, the same instructions `run` and `step` execute. on the first
    /// lex, parse, type, or codegen error the method stops and returns
    /// `ok: false` with the error diagnostics. on success it stores the
    /// program, builds a fresh VM over the same source, records the source,
    /// and returns `ok: true` with the disassembly and any warnings. never
    /// panics, never throws.
    fn compile_core(&mut self, source: &str) -> CompileResult {
        // stage 1: lex. a single QalaError on failure.
        let tokens = match crate::lexer::Lexer::tokenize(source) {
            Ok(t) => t,
            Err(e) => return compile_failed(vec![e], source),
        };
        // stage 2: parse. a single QalaError on failure.
        let ast = match crate::parser::Parser::parse(&tokens) {
            Ok(a) => a,
            Err(e) => return compile_failed(vec![e], source),
        };
        // stage 3: typecheck. errors block; warnings never block.
        let (typed, errors, warnings) = crate::typechecker::check_program(&ast, source);
        if !errors.is_empty() {
            return compile_failed(errors, source);
        }
        // stage 4: codegen. a Vec<QalaError> on failure.
        let mut program = match crate::codegen::compile_program(&typed, source) {
            Ok(p) => p,
            Err(errs) => return compile_failed(errs, source),
        };
        // stage 5: optimize in place, then disassemble the OPTIMIZED program
        // so the playground shows what run and step actually execute.
        program.optimize();
        let disassembly = program.disassemble();
        // store the optimized program, a VM over it built with the SAME
        // source, and the source itself -- all three together so a runtime
        // error renders against the exact text that was compiled.
        self.vm = Some(crate::vm::Vm::new(program.clone(), source.to_string()));
        self.program = Some(program);
        self.last_src = source.to_string();
        CompileResult {
            ok: true,
            disassembly: Some(disassembly),
            diagnostics: warnings
                .iter()
                .map(|w| crate::diagnostics::Diagnostic::from(w).to_monaco(source))
                .collect(),
        }
    }

    /// compile the given source through the ARM64 backend and return an
    /// `Arm64Result`.
    ///
    /// runs lex, parse, and typecheck -- the same front end `compile_core`
    /// runs -- then `crate::arm64::compile_arm64` on the typed AST. on a
    /// front-end error returns `ok: false` with the front-end diagnostics; on
    /// an unsupported-construct rejection returns `ok: false` with the
    /// backend's diagnostics; on success returns `ok: true` with the assembly.
    /// the source is an argument rather than `self.last_src` -- `last_src` is
    /// set only on a successful `compile_core`, so reusing it would silently
    /// compile stale source after a failed compile. never panics, never throws.
    fn compile_arm64_core(&mut self, source: &str) -> Arm64Result {
        // stage 1: lex. a single QalaError on failure.
        let tokens = match crate::lexer::Lexer::tokenize(source) {
            Ok(t) => t,
            Err(e) => return arm64_failed(vec![e], source),
        };
        // stage 2: parse. a single QalaError on failure.
        let ast = match crate::parser::Parser::parse(&tokens) {
            Ok(a) => a,
            Err(e) => return arm64_failed(vec![e], source),
        };
        // stage 3: typecheck. errors block; warnings never block.
        let (typed, errors, _warnings) = crate::typechecker::check_program(&ast, source);
        if !errors.is_empty() {
            return arm64_failed(errors, source);
        }
        // stage 4: the ARM64 backend. an unsupported construct is Err(Vec<QalaError>).
        match crate::arm64::compile_arm64(&typed, source) {
            Ok(assembly) => Arm64Result {
                ok: true,
                assembly: Some(assembly),
                diagnostics: Vec::new(),
            },
            Err(errs) => arm64_failed(errs, source),
        }
    }

    /// run the compiled program to `Halt` or the first runtime error.
    ///
    /// with no program compiled this returns `ok: false`, an empty state, and
    /// the no-program diagnostic. otherwise it runs the VM and returns the
    /// final state either way -- on a runtime error the state is the
    /// fault-point snapshot and `error` carries the diagnostic. never panics,
    /// never throws.
    fn run_core(&mut self) -> RunResult {
        let vm = match self.vm.as_mut() {
            Some(vm) => vm,
            None => {
                return RunResult {
                    ok: false,
                    state: empty_state(),
                    error: Some(no_program_diagnostic()),
                };
            }
        };
        match vm.run() {
            Ok(()) => RunResult {
                ok: true,
                state: vm.get_state(),
                error: None,
            },
            Err(err) => RunResult {
                ok: false,
                // get_state after the error: the playground wants the stack
                // and console at the fault point.
                state: vm.get_state(),
                error: Some(crate::diagnostics::Diagnostic::from(err).to_monaco(&self.last_src)),
            },
        }
    }

    /// advance the compiled program exactly one instruction.
    ///
    /// with no program compiled this returns `status: "error"`, an empty
    /// state, and the no-program diagnostic. otherwise it steps the VM:
    /// an ordinary instruction yields `"ran"`, reaching `Halt` yields
    /// `"halted"`, and a runtime fault yields `"error"` with the diagnostic.
    /// the state after the step is included in every compiled-program arm.
    /// never panics, never throws.
    fn step_core(&mut self) -> StepResult {
        let vm = match self.vm.as_mut() {
            Some(vm) => vm,
            None => {
                return StepResult {
                    status: "error".to_string(),
                    state: empty_state(),
                    error: Some(no_program_diagnostic()),
                };
            }
        };
        match vm.step() {
            Ok(crate::vm::StepOutcome::Ran) => StepResult {
                status: "ran".to_string(),
                state: vm.get_state(),
                error: None,
            },
            Ok(crate::vm::StepOutcome::Halted) => StepResult {
                status: "halted".to_string(),
                state: vm.get_state(),
                error: None,
            },
            Err(err) => StepResult {
                status: "error".to_string(),
                state: vm.get_state(),
                error: Some(crate::diagnostics::Diagnostic::from(err).to_monaco(&self.last_src)),
            },
        }
    }

    /// snapshot the run-and-step VM's execution state.
    ///
    /// with no program compiled this returns the empty state rather than an
    /// error -- the state of nothing is the zero snapshot. otherwise it
    /// returns `Vm::get_state`. never panics, never throws.
    fn get_state_core(&self) -> crate::vm::VmState {
        self.vm
            .as_ref()
            .map(|vm| vm.get_state())
            .unwrap_or_else(empty_state)
    }

    /// disassemble the compiled program's optimized bytecode.
    ///
    /// with no program compiled this returns a short placeholder line rather
    /// than an error -- the disassembly of nothing is an empty listing. never
    /// panics, never throws.
    fn disassemble_core(&self) -> String {
        self.program
            .as_ref()
            .map(|p| p.disassemble())
            .unwrap_or_else(|| "; no program compiled -- call compile() first".to_string())
    }

    /// evaluate one line of REPL source against the persistent REPL VM.
    ///
    /// the REPL VM accumulates accepted lines, so a `let` binding from an
    /// earlier call is in scope for this one and the console accumulates
    /// across calls. on success the line's value is rendered to a
    /// `StateValue`; on any error -- lex, parse, type, codegen, or runtime --
    /// the diagnostic is returned and the line is not added to history, so a
    /// typo cannot poison later calls. never panics, never throws.
    fn repl_eval_core(&mut self, source: &str) -> ReplResult {
        match self.repl_vm.repl_eval(source) {
            Ok(value) => {
                // value_to_string and runtime_type_name are pub(crate); wasm.rs
                // is in the same crate, so it builds the StateValue directly --
                // the identical pair Vm::get_state builds for each stack slot.
                let rendered = self.repl_vm.value_to_string(value);
                let type_name = self.repl_vm.runtime_type_name(value);
                ReplResult {
                    ok: true,
                    value: Some(crate::vm::StateValue {
                        rendered,
                        type_name,
                    }),
                    console: self.repl_vm.get_state().console,
                    error: None,
                }
            }
            Err(err) => ReplResult {
                ok: false,
                value: None,
                console: self.repl_vm.get_state().console,
                // the REPL VM compiled a synthetic wrapped source the bridge
                // does not hold, so a span into it cannot be rendered against
                // last_src or the raw line. render against an empty source:
                // the MonacoDiagnostic keeps the message -- the part the REPL
                // console shows -- and the position collapses harmlessly.
                error: Some(crate::diagnostics::Diagnostic::from(err).to_monaco("")),
            },
        }
    }

    /// clear all compiled and run state and rebuild a fresh REPL VM.
    ///
    /// after this the session is back to its just-constructed shape: no
    /// program, no run-and-step VM, an empty REPL VM, and an empty last
    /// source. never panics, never throws.
    fn reset_core(&mut self) {
        self.program = None;
        self.vm = None;
        self.repl_vm = crate::vm::Vm::new_repl();
        self.last_src = String::new();
    }
}

#[wasm_bindgen]
impl Qala {
    /// construct a fresh Qala session.
    ///
    /// installs the process-wide panic hook once, then returns a session with
    /// no compiled program, no run-and-step VM, an empty REPL VM, and an empty
    /// last source. the playground calls this once per browser tab. returns a
    /// `Qala` that crosses the boundary as a JavaScript class.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Qala {
        install_panic_hook();
        Qala {
            program: None,
            vm: None,
            repl_vm: crate::vm::Vm::new_repl(),
            last_src: String::new(),
        }
    }

    /// compile Qala source and return a `CompileResult` as a `JsValue`.
    ///
    /// never throws -- a failed compile is `ok: false` data in the value.
    pub fn compile(&mut self, source: &str) -> JsValue {
        to_js(&self.compile_core(source))
    }

    /// compile the source through the ARM64 backend and return an
    /// `Arm64Result` as a `JsValue`.
    ///
    /// never throws -- a front-end error or an unsupported construct is
    /// `ok: false` data in the value.
    pub fn compile_arm64(&mut self, source: &str) -> JsValue {
        to_js(&self.compile_arm64_core(source))
    }

    /// run the compiled program and return a `RunResult` as a `JsValue`.
    ///
    /// never throws -- a runtime error or misuse is `ok: false` data in the
    /// value.
    pub fn run(&mut self) -> JsValue {
        to_js(&self.run_core())
    }

    /// step the compiled program one instruction and return a `StepResult` as
    /// a `JsValue`.
    ///
    /// never throws -- a runtime error or misuse is an `"error"` status in the
    /// value.
    pub fn step(&mut self) -> JsValue {
        to_js(&self.step_core())
    }

    /// snapshot the VM state and return a `VmState` as a `JsValue`.
    ///
    /// never throws -- with no program compiled the value is the empty state.
    pub fn get_state(&self) -> JsValue {
        to_js(&self.get_state_core())
    }

    /// disassemble the compiled program and return the listing string as a
    /// `JsValue`.
    ///
    /// never throws -- with no program compiled the value is a placeholder
    /// listing.
    pub fn disassemble(&self) -> JsValue {
        to_js(&self.disassemble_core())
    }

    /// evaluate one REPL line and return a `ReplResult` as a `JsValue`.
    ///
    /// never throws -- a failed line is `ok: false` data in the value.
    pub fn repl_eval(&mut self, source: &str) -> JsValue {
        to_js(&self.repl_eval_core(source))
    }

    /// clear all session state and return `JsValue::NULL`.
    ///
    /// `reset` produces nothing meaningful -- the playground ignores the
    /// return. never throws.
    pub fn reset(&mut self) -> JsValue {
        self.reset_core();
        JsValue::NULL
    }
}

/// install a panic hook once for the whole process.
///
/// the compiler and VM never panic by contract, so this hook is a bug-report
/// safety net, not a feature: if the discipline is ever violated it captures
/// the panic message into a process-local string for diagnosability rather
/// than leaving an opaque WASM trap. the `Once` guard means repeated session
/// constructions do not stack hooks.
fn install_panic_hook() {
    use std::sync::Once;
    static HOOK: Once = Once::new();
    HOOK.call_once(|| {
        std::panic::set_hook(Box::new(|info| {
            if let Ok(mut last) = LAST_PANIC.lock() {
                *last = Some(info.to_string());
            }
        }));
    });
}

/// the most recent panic message captured by the hook, or `None` if the
/// no-panic contract has held. process-local; read only for bug reports.
static LAST_PANIC: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);

#[cfg(test)]
mod tests {
    use super::*;

    /// build a session and compile `source` against it, returning the session
    /// ready for a run or step call. the compile is asserted to succeed so a
    /// test that wants a compiled session does not silently get an
    /// un-compiled one.
    fn compiled(source: &str) -> Qala {
        let mut qala = Qala::new();
        let result = qala.compile_core(source);
        assert!(
            result.ok,
            "fixture compile failed: {:?}",
            result.diagnostics
        );
        qala
    }

    /// a small program that prints and halts -- the shared fixture for the
    /// run, step, and reset tests.
    const PRINTING_PROGRAM: &str = "fn main() is io {\n  println(\"step output\")\n}";

    // ---- compile_core -----------------------------------------------------

    #[test]
    fn compile_core_on_a_clean_program_returns_ok_with_disassembly() {
        let mut qala = Qala::new();
        let result = qala.compile_core("fn main() is io {\n  println(\"hi\")\n}");
        assert!(
            result.ok,
            "expected a clean compile: {:?}",
            result.diagnostics
        );
        let disassembly = result.disassembly.expect("clean compile has disassembly");
        assert!(!disassembly.is_empty(), "disassembly should not be empty");
    }

    #[test]
    fn compile_core_on_a_broken_program_returns_the_error_diagnostics() {
        let mut qala = Qala::new();
        // line 2 assigns a str where the i64 annotation demands an i64.
        let src = "fn main() is io {\n  let x: i64 = \"not a number\"\n}";
        let result = qala.compile_core(src);
        assert!(!result.ok, "a type error must fail the compile");
        assert!(
            result.disassembly.is_none(),
            "a failed compile has no disassembly"
        );
        assert!(
            !result.diagnostics.is_empty(),
            "the error must surface as a diagnostic"
        );
        assert_eq!(
            result.diagnostics[0].line, 2,
            "the diagnostic should point at the offending line: {:?}",
            result.diagnostics[0],
        );
    }

    #[test]
    fn compile_core_on_an_unused_variable_warns_but_compiles() {
        let mut qala = Qala::new();
        // `unused` is declared and never read -- an unused-variable warning.
        let src = "fn main() is io {\n  let unused = 1\n  println(\"hi\")\n}";
        let result = qala.compile_core(src);
        assert!(result.ok, "a warning must not block the compile");
        assert!(
            result.disassembly.is_some(),
            "a warning compile still disassembles"
        );
        assert!(
            result.diagnostics.iter().any(|d| d.severity == 0),
            "the unused-variable warning should be in the diagnostics: {:?}",
            result.diagnostics,
        );
    }

    // ---- compile_arm64_core -----------------------------------------------

    #[test]
    fn compile_arm64_core_on_an_integer_program_returns_ok_with_assembly() {
        let mut qala = Qala::new();
        // an integer-core program: a `let` and an i64-holed interpolation, both
        // inside the ARM64 backend's shipped integer core.
        let src = "fn main() is io {\n  let x = 1 + 2\n  println(\"{x}\")\n}";
        let result = qala.compile_arm64_core(src);
        assert!(
            result.ok,
            "expected ARM64 success: {:?}",
            result.diagnostics
        );
        let assembly = result.assembly.expect("a success carries assembly");
        assert!(
            !assembly.is_empty(),
            "the assembly text should not be empty"
        );
        assert!(
            result.diagnostics.is_empty(),
            "a success carries no diagnostics"
        );
    }

    #[test]
    fn compile_arm64_core_on_a_float_program_returns_the_backend_rejection() {
        let mut qala = Qala::new();
        // a float is outside the integer core -- the typechecker accepts it,
        // the ARM64 backend rejects it with a clean diagnostic.
        let result = qala.compile_arm64_core("fn main() is io {\n  let x = 1.5\n}");
        assert!(!result.ok, "a float program must fail the ARM64 backend");
        assert!(
            result.assembly.is_none(),
            "a rejected program has no assembly"
        );
        assert!(
            !result.diagnostics.is_empty(),
            "the rejection must surface a diagnostic"
        );
        let message = &result.diagnostics[0].message;
        assert!(
            message.contains("f64") || message.contains("floats"),
            "the diagnostic should name the unsupported float construct: {message:?}",
        );
    }

    #[test]
    fn compile_arm64_core_on_a_broken_program_returns_not_ok() {
        let mut qala = Qala::new();
        // a syntactically broken program fails the front end before the backend.
        let result = qala.compile_arm64_core("fn main( {");
        assert!(!result.ok, "a broken program must fail the compile");
        assert!(
            result.assembly.is_none(),
            "a failed compile has no assembly"
        );
        assert!(
            !result.diagnostics.is_empty(),
            "the syntax error must surface a diagnostic"
        );
    }

    // ---- the WASM-05 end-to-end test --------------------------------------

    #[test]
    fn wasm_end_to_end_compiles_optimizes_and_runs_to_expected_output() {
        // the hello.qala example content, inline so the test has no file
        // dependency. compile_core runs the full pipeline including the
        // optimizer; run_core executes the optimized program.
        let mut qala = Qala::new();
        let src = "fn main() is io {\n  let name = \"world\"\n  println(\"hello, {name}!\")\n}";
        let compiled = qala.compile_core(src);
        assert!(
            compiled.ok,
            "end-to-end compile failed: {:?}",
            compiled.diagnostics
        );
        let run = qala.run_core();
        assert!(run.ok, "end-to-end run failed: {:?}", run.error);
        assert!(
            run.state
                .console
                .iter()
                .any(|l| l.contains("hello, world!")),
            "console did not contain the expected output: {:?}",
            run.state.console,
        );
    }

    // ---- misuse before compile --------------------------------------------

    #[test]
    fn run_core_before_compile_returns_an_error_shaped_result() {
        let mut qala = Qala::new();
        let result = qala.run_core();
        assert!(!result.ok, "run before compile must report failure");
        assert!(
            result.error.is_some(),
            "run before compile must carry a diagnostic"
        );
    }

    #[test]
    fn step_core_before_compile_returns_status_error() {
        let mut qala = Qala::new();
        let result = qala.step_core();
        assert_eq!(
            result.status, "error",
            "step before compile must be an error"
        );
        assert!(
            result.error.is_some(),
            "step before compile must carry a diagnostic"
        );
    }

    #[test]
    fn get_state_core_before_compile_returns_an_empty_state() {
        let qala = Qala::new();
        let state = qala.get_state_core();
        assert!(
            state.stack.is_empty(),
            "an un-compiled session has no stack"
        );
        assert!(
            state.console.is_empty(),
            "an un-compiled session has no console"
        );
    }

    #[test]
    fn disassemble_core_before_compile_returns_a_placeholder() {
        let qala = Qala::new();
        let listing = qala.disassemble_core();
        assert!(!listing.is_empty(), "the placeholder listing is non-empty");
        assert!(
            listing.contains("no program compiled"),
            "the placeholder should mention no program: {listing:?}",
        );
    }

    // ---- run and step -----------------------------------------------------

    #[test]
    fn step_core_advances_then_halts() {
        let mut qala = compiled(PRINTING_PROGRAM);
        let mut saw_ran = false;
        // step through the program; a tiny program halts well within this cap.
        for _ in 0..1000 {
            let result = qala.step_core();
            match result.status.as_str() {
                "ran" => saw_ran = true,
                "halted" => {
                    assert!(saw_ran, "a program should run at least one instruction");
                    return;
                }
                other => panic!("unexpected step status: {other}"),
            }
        }
        panic!("the program did not halt within the step cap");
    }

    // ---- the REPL ---------------------------------------------------------

    #[test]
    fn repl_eval_core_persists_a_binding_across_calls() {
        let mut qala = Qala::new();
        let first = qala.repl_eval_core("let x = 5");
        assert!(
            first.ok,
            "the binding line should evaluate: {:?}",
            first.error
        );
        let second = qala.repl_eval_core("x + 1");
        assert!(
            second.ok,
            "the binding should be visible on the next call: {:?}",
            second.error
        );
        let value = second.value.expect("an expression line has a value");
        assert_eq!(value.rendered, "6", "x + 1 should render as 6");
    }

    #[test]
    fn repl_eval_core_console_persists_across_calls() {
        let mut qala = Qala::new();
        let first = qala.repl_eval_core("println(\"one\")");
        assert!(
            first.ok,
            "the first println line should evaluate: {:?}",
            first.error
        );
        let first_len = first.console.len();
        assert!(
            first_len > 0,
            "the first println should produce console output"
        );
        let second = qala.repl_eval_core("println(\"two\")");
        assert!(
            second.ok,
            "the second println line should evaluate: {:?}",
            second.error
        );
        assert!(
            second.console.len() > first_len,
            "the console should accumulate across calls: {:?}",
            second.console,
        );
        assert!(
            second.console.iter().any(|l| l.contains("one")),
            "the earlier output should still be present: {:?}",
            second.console,
        );
    }

    // ---- reset ------------------------------------------------------------

    #[test]
    fn reset_core_clears_compiled_state() {
        let mut qala = compiled(PRINTING_PROGRAM);
        let run = qala.run_core();
        assert!(run.ok, "the fixture program should run: {:?}", run.error);
        qala.reset_core();
        let state = qala.get_state_core();
        assert!(state.stack.is_empty(), "reset should clear the stack");
        assert!(state.console.is_empty(), "reset should clear the console");
    }

    // ---- the serialize witness --------------------------------------------

    #[test]
    fn result_structs_implement_serialize() {
        // a compile-time check: if a result struct stops deriving
        // serde::Serialize this generic call fails to typecheck and the test
        // build breaks. Serialize has a generic method so it is not
        // dyn-compatible; a generic asserting function is the standard way to
        // spell "T: Serialize" as a witness.
        fn assert_serialize<T: serde::Serialize>(_: &T) {}
        assert_serialize(&CompileResult {
            ok: true,
            disassembly: None,
            diagnostics: Vec::new(),
        });
        assert_serialize(&RunResult {
            ok: true,
            state: empty_state(),
            error: None,
        });
        assert_serialize(&StepResult {
            status: "ran".to_string(),
            state: empty_state(),
            error: None,
        });
        assert_serialize(&ReplResult {
            ok: true,
            value: None,
            console: Vec::new(),
            error: None,
        });
        assert_serialize(&Arm64Result {
            ok: true,
            assembly: None,
            diagnostics: Vec::new(),
        });
    }
}
