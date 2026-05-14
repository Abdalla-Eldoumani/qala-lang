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
