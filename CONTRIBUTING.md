# Contributing to Qala

Thanks for your interest in Qala. This guide covers how to build the project,
run the tests, and submit a change.

## What Qala is

Qala is a statically typed programming language. The compiler and bytecode VM
are written in Rust. A web playground lets you write Qala, see type and effect
errors pinned to the source, compile, and step through execution in the
browser. A second compiler backend emits AArch64 assembly.

## Repository layout

The repository is a Cargo workspace with two crates plus the playground:

- `crates/qala-compiler/` is the compiler and bytecode VM. It also builds to
  WebAssembly, which the playground loads.
- `crates/qala-cli/` is the `qala` command-line tool: `run`, `check`, `build`,
  and `repl`.
- `playground/` is the Next.js web playground. It is a Node project and is not
  part of the Cargo workspace.

## Prerequisites

- Rust 1.95.0. The toolchain is pinned by `rust-toolchain.toml`, so rustup
  selects it for you.
- Node.js 20 or newer, to build the playground.
- `wasm-pack`, to build the compiler to WebAssembly.

## Building and testing

Run the Rust tests for the whole workspace from the repository root:

```bash
cargo test --workspace
```

Before opening a pull request, also run the lint and format checks, since
continuous integration runs them:

```bash
cargo clippy --workspace -- -D warnings
cargo fmt --all -- --check
```

`cargo fmt --all` fixes formatting in place if the check reports drift.

## Running the CLI

```bash
cargo run -p qala-cli -- run examples/hello.qala     # compile and run
cargo run -p qala-cli -- check examples/hello.qala   # type-check only
cargo run -p qala-cli -- build examples/hello.qala   # write a compiled artifact
cargo run -p qala-cli -- repl                        # an interactive session
```

`qala build` takes a `--target` flag. The default is `bytecode`; `--target
arm64` emits AArch64 assembly for the integer core of the language.

## Running the playground

Build the compiler to WebAssembly, then start the playground:

```bash
wasm-pack build crates/qala-compiler --target web --out-dir ../../playground/lib/wasm
cd playground
npm install
npm run dev
```

The playground also has `npm run typecheck`, `npm run lint`, and `npm run
build`. Run all three before submitting a change that touches `playground/`.

## Submitting a change

1. Open an issue first for anything beyond a small fix, so the approach can be
   discussed before you write code.
2. Keep commits small and focused. One coherent change per commit, with a
   short, clear message in the imperative mood.
3. Make sure `cargo test --workspace`, `cargo clippy --workspace -- -D
   warnings`, and `cargo fmt --all -- --check` all pass. If you touched the
   playground, run its typecheck, lint, and build too.
4. Open a pull request. The CI workflow runs the same checks; a green CI run is
   expected before review.

## Code style

- Rust: doc comments on public items, no `unwrap()` outside tests, errors
  carried as `Result<T, QalaError>`. `cargo fmt` is the formatter.
- TypeScript: `strict` mode, no `any`, functional components and hooks.
- Prose, comments, and commit messages: plain and direct. No buzzwords, no
  emoji.

## Reporting bugs and asking questions

Use the GitHub issue tracker. The issue templates will prompt you for the
details that make a report actionable: a minimal program that reproduces the
problem, what you expected, and what happened instead.
