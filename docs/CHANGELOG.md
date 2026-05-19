# Changelog

All notable changes to Qala are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-05-19

The first public release. The `qala-compiler` and `qala-cli` crates are
published on crates.io and the `qala` binary is attached to this release on
GitHub for Linux, macOS, and Windows. The crate-version line starts at 0.1.0;
the language itself is at the v2.0 milestone documented below, atop the
original v1.0 language.

### Added

- Cross-platform `qala` binaries (Linux x86_64, macOS arm64, Windows x86_64)
  attached to the GitHub release, built by the release workflow.
- Both crates published on crates.io: `cargo install qala-cli` installs the
  `qala` binary, and `qala-compiler` is available as a library and a
  WebAssembly module.

## [2.0.0] - 2026-05-19

The production and ARM64 release. The v1.0 language is unchanged; this release
makes the project installable, deployable, continuously verified, and adds a
second compiler backend.

### Added

- A `qala` command-line tool with four subcommands: `run` compiles and executes
  a program, `check` type-checks without running, `build` writes a compiled
  artifact, and `repl` opens an interactive session with line editing and
  history. Diagnostics render through the compiler's rich diagnostic path;
  program output goes to stdout and diagnostics to stderr.
- An AArch64 assembly backend. `qala build --target arm64` emits assembly text
  for the integer core of the language: `i64` arithmetic, comparisons, boolean
  logic, `if`/`while`/`for` control flow, `let` bindings in stack slots, and
  functions with calls and recursion following the AAPCS64 calling convention.
  Integer `print` and `println` output is emitted through `printf`. The emitted
  assembly is in the CPSC 355 hosted-Linux dialect and is verified end to end:
  it assembles with `m4` and an AArch64 toolchain and runs correctly under
  QEMU. A construct the backend does not yet support is rejected with a clear
  diagnostic rather than miscompiled.
- An ARM64 assembly view in the web playground, alongside the bytecode panel.
- A 404 page for the playground, styled to match its design system.
- A `vercel.json` and a tuned build configuration so the playground is ready to
  deploy to Vercel, with the WebAssembly module served with the correct content
  type and a long-lived cache header.
- GitHub Actions workflows: a CI workflow that runs the tests, clippy, the
  format check, the WebAssembly build, and the playground checks on every push
  and pull request; and a release workflow that builds cross-platform `qala`
  binaries on a version tag.
- Repository documentation: this changelog, a contributing guide, a security
  policy, GitHub issue and pull request templates, and `.gitattributes` and
  `.editorconfig` configuration.

### Changed

- The single compiler crate is now a Cargo workspace with two members,
  `qala-compiler` and `qala-cli`, sharing one lockfile and one build profile.
- The web playground received a design pass, raising its craft to a
  professional standard while keeping its existing dark design system.

## [1.0.0] - 2026-05-18

The first release: the Qala language, its compiler and bytecode VM, and the web
playground.

### Added

- The Qala language: a statically typed language whose distinctive features are
  effect annotations, scope-bound `defer`, `comptime` blocks, the pipeline
  operator, `Result<T, E>` with the `?` and `or` operators, exhaustive pattern
  matching with guards, structural interfaces, string interpolation, and
  immutable bindings by default.
- The compiler: a hand-written lexer, a recursive-descent and Pratt parser, a
  single-pass type and effect checker, and bytecode generation with constant
  folding, dead-code elimination, and a peephole optimizer. Errors are reported
  as rich diagnostics with the source line, an underline, and a suggestion.
- The bytecode VM: a stack-based interpreter with NaN-boxed values, call frames,
  a managed heap, single-step execution, and fifteen standard-library built-ins.
- A WebAssembly bridge exposing the compile-and-run pipeline to JavaScript.
- The web playground: a Monaco-based editor with the Qala language definition,
  inline type and effect diagnostics, a bytecode disassembly panel, an animated
  value-stack view, step-through execution, an interactive REPL, and a set of
  bundled example programs.

[0.1.0]: https://github.com/Abdalla-Eldoumani/qala-lang/releases/tag/v0.1.0
