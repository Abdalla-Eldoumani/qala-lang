# Qala

[![CI](https://github.com/Abdalla-Eldoumani/qala-lang/actions/workflows/ci.yml/badge.svg)](https://github.com/Abdalla-Eldoumani/qala-lang/actions/workflows/ci.yml)
[![crates.io: qala-cli](https://img.shields.io/crates/v/qala-cli?label=qala-cli)](https://crates.io/crates/qala-cli)
[![crates.io: qala-compiler](https://img.shields.io/crates/v/qala-compiler?label=qala-compiler)](https://crates.io/crates/qala-compiler)

A statically typed programming language where saying is doing.

Qala (قال) is the Arabic word for "he said" or "to say." In Arabic, when you say
something with qala, you are making a declaration, stating what is. Qala the
language works the same way. When you write a function marked `is pure`, it is
pure. The compiler guarantees it. When you write `defer close(f)`, the file
closes. The compiler guarantees that too. There is no gap between what you write
and what happens.

This is the core idea: a language where your code is a truthful declaration of
behavior. Side effects are visible in function signatures. Resources are managed
explicitly. Types are known before the program runs. Nothing is hidden, nothing
is implicit, nothing surprises you.

If Haskell showed that purity makes programs easier to reason about, and Rust
showed that compile-time guarantees prevent real bugs, Qala asks: what if we
took those insights and made them accessible? No monads, no borrow checker, no
lifetime annotations. Just clear annotations (`is pure`, `is io`),
straightforward resource management (`defer`), and a type system that catches
mistakes without getting in your way.

## install

```bash
cargo install qala-cli
qala --help
```

Or download a prebuilt `qala` binary for Linux, macOS, or Windows from the
[releases page](https://github.com/Abdalla-Eldoumani/qala-lang/releases/latest).
Extract the archive and add the directory to your `PATH`, or invoke `qala`
by its full path.

`qala` is a command-line tool, not a desktop application. Run it from a
terminal:

```bash
qala --help        # list every subcommand
qala repl          # open the interactive shell
qala run f.qala    # compile and run a program
qala check f.qala  # type-check without running
```

On Windows, double-clicking `qala.exe` opens a console that closes
immediately, because the binary expects a subcommand and prints its help
to a terminal that disappears. Open Command Prompt or PowerShell first
and run `qala` from there.

## build from source

Run from the workspace root.

```bash
cargo test --workspace
wasm-pack build crates/qala-compiler --target web --out-dir ../../playground/lib/wasm
cd playground && npm install && npm run dev
```

Open http://localhost:3000 for the playground.

## a taste of Qala

```
fn fibonacci(n: i64) -> i64 is pure {
    if n <= 1 { return n }
    return fibonacci(n - 1) + fibonacci(n - 2)
}

fn main() is io {
    for i in 0..15 {
        println("fib({i}) = {fibonacci(i)}")
    }
}
```

`fibonacci` is marked `is pure`. It takes a value and returns a value. It cannot
print, allocate, or crash. The compiler enforces this. `main` is marked `is io`
because it prints to the console. The effects are right there in the signature.
Reading the code tells you exactly what it does.

This is the `fibonacci` example, one of six bundled programs you can open and
run in the playground.

## what Qala offers

**Effect annotations.** Functions declare their side effects: `is pure`,
`is io`, `is alloc`, `is panic`. The compiler tracks these and catches
violations. This is how Qala keeps its promise: what you say is what happens.

**Scope-bound defer.** `defer close(f)` runs when the current scope exits. The
compiler warns if a resource is acquired without a matching cleanup. Resource
management is explicit, visible, and verified.

**Compile-time evaluation.** `comptime` blocks execute during compilation,
embedding the result as a constant. One concept replaces macros, const generics,
and metaprogramming.

**Pipeline operator.** `|>` passes the left value as the first argument to the
right function. Data flows left to right, the way you read it.

**Error handling.** `Result<T, E>` with `?` for propagation and `or` for inline
fallbacks. Errors are values, not surprises.

**Exhaustive matching.** `match` requires all enum variants covered. Guard
clauses supported. The compiler tells you what you forgot.

**Structural interfaces.** If a type has the right methods, it satisfies the
interface. No ceremony.

**Immutable bindings.** `let` binds a value once. Bindings do not change, and
there is no assignment statement. To get a new value, bind a new name. Fewer
surprises.

**String interpolation.** `{expr}` inside strings evaluates and converts.

**A `qala` CLI.** A small command-line front end with `run`, `check`, `build`,
and an interactive `repl`. Diagnostics render through the compiler's rich
diagnostic path; program output goes to stdout, errors to stderr.

**ARM64 backend.** A second codegen target emits AArch64 assembly text in the
CPSC 355 hosted-Linux dialect: `qala build --target arm64 program.qala`. The
integer core is verified end to end under QEMU. The output is meant to be
pasted into the [AArch64 playground](https://aarch64-playground.vercel.app) to
step through execution one instruction at a time.

## error messages

Qala's compiler produces rich diagnostics with source context, underlines, and
suggestions:

```
error: pure function `compute` calls io function `println`
  --> main.qala:7:5
   |
7  |     println("debug")
   |     ^^^^^^^
   |
   = note: pure functions cannot call functions with io effects
   = hint: remove the `is pure` annotation, or refactor to remove the io call
```

Warnings for unused variables, unreachable code, and unmatched defer help catch
mistakes without blocking compilation.

## documentation

The Qala language:

- [docs/language-guide.md](docs/language-guide.md) teaches the language from scratch.
- [docs/stdlib-reference.md](docs/stdlib-reference.md) lists every built-in function.
- [docs/language-spec.md](docs/language-spec.md) is the full feature reference with design rationale.
- [docs/design-philosophy.md](docs/design-philosophy.md) is a short piece on the idea behind it.

The project:

- [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) covers how to build, test, and contribute.
- [docs/CHANGELOG.md](docs/CHANGELOG.md) is the release history.
- [docs/SECURITY.md](docs/SECURITY.md) is the security policy and disclosure channel.

## how it works

The repository is a Cargo workspace with two crates. `crates/qala-compiler` is
the compiler and bytecode VM: a hand-written lexer, a recursive-descent parser
with Pratt expression parsing, a single-pass type checker with effect
inference, and a bytecode compiler with constant folding, dead-code
elimination, and a peephole pass. The VM uses NaN-boxing for compact stack
values and carries source maps for precise runtime error reporting. The same
typed AST also feeds a second codegen target that emits AArch64 assembly text.
`crates/qala-cli` is the `qala` binary. The web playground loads the compiler
as a WebAssembly module.

## license

MIT
