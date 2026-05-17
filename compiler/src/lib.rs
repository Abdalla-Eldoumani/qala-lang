//! the Qala compiler and bytecode VM, as a single crate.
//!
//! the pipeline is lexer -> parser -> type and effect checker -> codegen -> VM.
//! this file declares the modules in dependency order. there are no
//! `wasm-bindgen` exports yet; the browser bridge is added in a later phase.

pub mod span;
pub mod errors;
pub mod token;
pub mod lexer;
pub mod ast;
pub mod parser;
pub mod types;
pub mod effects;
pub mod typed_ast;
pub mod typechecker;
pub mod diagnostics;
// constant-pool value enum -- shared by chunk, codegen, vm
pub mod value;
// bytecode opcode enum -- consumed by chunk, codegen, optimizer, vm
pub mod opcode;
// instruction stream + constant pool + source map + disassembler.
pub mod chunk;
// typed AST -> bytecode; constant folding, DCE, defer compilation, comptime.
pub mod codegen;
// peephole optimizer -- single-pass bytecode rewrites.
pub mod optimizer;
// stack-based bytecode interpreter -- call frames, heap, single-step.
pub mod vm;
