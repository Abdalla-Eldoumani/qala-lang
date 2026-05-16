//! the bytecode chunk + the whole program: instruction bytes, a constant
//! pool, and a parallel source-line map; plus the disassembler that turns
//! them back into the human-readable listing the playground will render in
//! its bytecode panel.
//!
//! one [`Chunk`] is one compiled function (or one throwaway `comptime`
//! body). a [`Program`] holds every chunk plus the entry-point index, the
//! globals table, and the parallel name tables the disassembler reads
//! by index. constant-pool entries are append-only (no dedupe in v1 per
//! Phase 4 research A6); the resulting deterministic insertion order is
//! the load-bearing invariant for "compile twice, same bytes" testing.
//!
//! the write helpers ([`Chunk::write_op`], [`Chunk::write_u16`],
//! [`Chunk::write_i16`]) are the ONLY way to append to the byte stream;
//! they update [`Chunk::source_lines`] in lockstep so the invariant
//! `code.len() == source_lines.len()` holds after every emission. the
//! peephole optimizer (Plan 04-05) uses `Vec::splice` directly on both
//! vectors, with the same lockstep discipline.
//!
//! forward jumps use [`Chunk::emit_jump`] / [`Chunk::patch_jump`] -- the
//! Crafting Interpreters back-patching pair (Phase 4 research Pattern 4);
//! backward jumps use [`Chunk::emit_loop`] when the target is already
//! emitted. both error with [`crate::errors::QalaError::Parse`] +
//! "branch too far" on i16 range overflow (Phase 4 research A1).
//!
//! the disassembler ([`Chunk::disassemble`] + [`Program::disassemble`])
//! is the documented contract for Phase 6's WASM bridge and Phase 8's
//! bytecode panel. byte-determinism is part of the contract; the inline
//! `compile_twice_disassemble_twice` test guards it.

use crate::errors::QalaError;
use crate::opcode::Opcode;
use crate::span::Span;
use crate::value::Value;
use std::fmt::Write as _;

/// a compiled function (or top-level chunk): instruction bytes + a parallel
/// per-byte source-line map + a constant pool indexed by u16 from the bytes.
///
/// the three vectors stay in lockstep; the [`Chunk::write_op`] /
/// [`Chunk::write_u16`] / [`Chunk::write_i16`] helpers are the only way to
/// append to `code`, so the invariant is structurally enforced.
#[derive(Debug, Clone, Default)]
pub struct Chunk {
    /// the instruction byte stream; one byte per opcode, plus per-operand
    /// bytes per [`Opcode::operand_bytes`].
    pub code: Vec<u8>,
    /// the per-function constant pool, indexed by u16 from CONST / FIELD /
    /// MAKE_ENUM_VARIANT operands. append-only -- a rewound CONST leaves a
    /// dangling entry, harmless because disassemble walks instructions, not
    /// pool entries.
    pub constants: Vec<Value>,
    /// the 1-based source line of the byte at the same index in `code`.
    /// multi-byte instructions duplicate the line across operand bytes so
    /// `source_lines[i]` is valid for any `i in 0..code.len()`.
    pub source_lines: Vec<u32>,
}

/// a whole compiled program: one chunk per function, with the entry-point
/// index and a globals table.
///
/// function indexes in CALL opcodes are indexes into [`Program::chunks`];
/// matching name lookups for the disassembler are in [`Program::fn_names`].
/// enum variant ids in MAKE_ENUM_VARIANT and MATCH_VARIANT are indexes into
/// [`Program::enum_variant_names`].
#[derive(Debug, Clone, Default)]
pub struct Program {
    /// one chunk per function, indexed by function id (u16 in CALL operands).
    pub chunks: Vec<Chunk>,
    /// the chunks index where execution starts. Phase 4 codegen sets this
    /// to the index of the `main` function.
    pub main_index: usize,
    /// global names indexed by GET_GLOBAL / SET_GLOBAL u16 operands. v1 has
    /// no top-level `let` so this is empty in practice -- kept for forward
    /// compatibility.
    pub globals: Vec<String>,
    /// function names parallel to `chunks` -- the disassembler renders
    /// CALL #fn_id(name).
    pub fn_names: Vec<String>,
    /// (enum_name, variant_name) pairs indexed by MAKE_ENUM_VARIANT /
    /// MATCH_VARIANT u16 operands. the disassembler renders
    /// #variant_id(Enum::Variant).
    pub enum_variant_names: Vec<(String, String)>,
}

impl Chunk {
    /// construct an empty chunk. all three vectors start empty; the
    /// lockstep invariant `code.len() == source_lines.len()` holds
    /// trivially.
    pub fn new() -> Self {
        Self::default()
    }

    /// emit a zero-operand opcode. `line` is the 1-based source line of the
    /// originating expression (looked up via `LineIndex::location` at the
    /// call site). pushes one byte to `code` and one entry to `source_lines`
    /// so the invariant `code.len() == source_lines.len()` is preserved.
    pub fn write_op(&mut self, op: Opcode, line: u32) {
        self.code.push(op as u8);
        self.source_lines.push(line);
    }

    /// emit a u16 operand, little-endian. `line` is the same line as the
    /// preceding opcode -- the map duplicates the line per byte so
    /// `source_lines[i]` is valid for any `i in 0..code.len()`.
    pub fn write_u16(&mut self, v: u16, line: u32) {
        let bytes = v.to_le_bytes();
        self.code.push(bytes[0]);
        self.code.push(bytes[1]);
        self.source_lines.push(line);
        self.source_lines.push(line);
    }

    /// emit a signed i16 operand for relative jump offsets. uses
    /// two's-complement little-endian, the same byte sequence on every
    /// platform Rust supports. delegates to [`Chunk::write_u16`] via a
    /// `v as u16` cast (the bit pattern is identical).
    pub fn write_i16(&mut self, v: i16, line: u32) {
        self.write_u16(v as u16, line);
    }

    /// read a u16 operand at byte offset `off`. the inverse of
    /// [`Chunk::write_u16`]. callers must ensure `off + 2 <= code.len()`.
    pub fn read_u16(&self, off: usize) -> u16 {
        u16::from_le_bytes([self.code[off], self.code[off + 1]])
    }

    /// read an i16 operand at byte offset `off`. the inverse of
    /// [`Chunk::write_i16`]. callers must ensure `off + 2 <= code.len()`.
    pub fn read_i16(&self, off: usize) -> i16 {
        i16::from_le_bytes([self.code[off], self.code[off + 1]])
    }

    /// append a value to the constant pool and return its u16 index. the
    /// pool is append-only (no dedupe in v1 per Phase 4 research A6);
    /// determinism follows from this. the assumption is that 65536
    /// constants per chunk is more than realistic; debug builds assert
    /// the cap, production builds keep the cast for v1. widening to u32
    /// is a v2 concern.
    pub fn add_constant(&mut self, v: Value) -> u16 {
        let idx = self.constants.len();
        debug_assert!(
            idx <= u16::MAX as usize,
            "constant pool overflow -- chunk has >65536 constants"
        );
        self.constants.push(v);
        idx as u16
    }

    /// peek the most recent CONST emission. returns `(byte_offset, pool_idx)`
    /// when the last full instruction in `code` is a CONST + u16 operand;
    /// `None` otherwise.
    ///
    /// the constant folder in codegen uses this before emitting a binary
    /// opcode: when both operands are CONSTs the folder can compute the
    /// result at compile time.
    pub fn last_const(&self) -> Option<(usize, u16)> {
        let n = self.code.len();
        if n < 3 {
            return None;
        }
        if self.code[n - 3] != (Opcode::Const as u8) {
            return None;
        }
        let idx = u16::from_le_bytes([self.code[n - 2], self.code[n - 1]]);
        Some((n - 3, idx))
    }

    /// peek the CONST emission BEFORE the most recent CONST. returns
    /// `(byte_offset, pool_idx)` when the chunk ends in two consecutive
    /// `CONST u16` instructions; `None` otherwise. used by the binary-
    /// expression constant folder to peek both operands before deciding
    /// to fold.
    pub fn second_to_last_const(&self) -> Option<(usize, u16)> {
        let (off1, _) = self.last_const()?;
        if off1 < 3 {
            return None;
        }
        if self.code[off1 - 3] != (Opcode::Const as u8) {
            return None;
        }
        let idx = u16::from_le_bytes([self.code[off1 - 2], self.code[off1 - 1]]);
        Some((off1 - 3, idx))
    }

    /// rewind the most recent CONST: drop its 3 bytes from `code` and 3
    /// entries from `source_lines`, returning the pool index. the pool
    /// itself is NOT shrunk -- the rewound entry remains in
    /// [`Chunk::constants`] as a harmless orphan. the determinism contract
    /// (append-only pool) depends on never mutating prior pool entries.
    ///
    /// returns `None` when the chunk does not end with a CONST instruction
    /// (i.e. [`Chunk::last_const`] returns `None`). callers that have
    /// already peeked via `last_const` / `second_to_last_const` will
    /// always receive `Some`; the `None` path exists so the WASM build
    /// cannot panic on misuse.
    pub fn rewind_last_const(&mut self) -> Option<u16> {
        let (off, idx) = self.last_const()?;
        self.code.truncate(off);
        self.source_lines.truncate(off);
        Some(idx)
    }

    /// emit a forward jump with a placeholder i16 operand. returns the
    /// byte position of the FIRST operand byte so the caller can later
    /// [`Chunk::patch_jump`] it to the real target.
    ///
    /// uses [`i16::MAX`] as the placeholder so an unpatched offset is
    /// visually obvious in a disassembly. callers MUST patch the jump
    /// before completing the chunk; an unpatched jump is a codegen bug.
    pub fn emit_jump(&mut self, op: Opcode, line: u32) -> usize {
        self.write_op(op, line);
        let pos = self.code.len();
        self.write_i16(i16::MAX, line);
        pos
    }

    /// patch the i16 operand bytes at `pos` to reach the current end of
    /// `code`. the offset is computed relative to the byte AFTER the
    /// operand (i.e. `code.len() - pos - 2`), matching the VM's branch
    /// arithmetic per Crafting Interpreters chapter 23.
    ///
    /// returns `Err(QalaError::Parse { ..., "branch too far" })` when the
    /// computed offset would not fit in i16. this is a v1 limitation
    /// (Phase 4 research A1); widening to i32 is a v2 concern.
    pub fn patch_jump(&mut self, pos: usize) -> Result<(), QalaError> {
        let target = self.code.len();
        let jump = target as isize - pos as isize - 2;
        if jump < i16::MIN as isize || jump > i16::MAX as isize {
            return Err(QalaError::Parse {
                span: Span::new(0, 0),
                message: format!(
                    "branch too far -- chunk exceeds i16 jump range ({jump} bytes)"
                ),
            });
        }
        let bytes = (jump as i16).to_le_bytes();
        self.code[pos] = bytes[0];
        self.code[pos + 1] = bytes[1];
        Ok(())
    }

    /// emit a backward jump whose target is `loop_start`. unlike
    /// [`Chunk::emit_jump`], the offset is known up-front so no patching
    /// is needed. the offset is computed from the byte AFTER the 2-byte
    /// operand back to `loop_start` (i.e. `loop_start - pos - 3` where
    /// `pos` is the byte position of the JUMP opcode itself).
    ///
    /// returns the same `Err(QalaError::Parse { ..., "branch too far" })`
    /// when the offset exceeds i16 range.
    pub fn emit_loop(&mut self, loop_start: usize, line: u32) -> Result<(), QalaError> {
        let pos = self.code.len();
        let offset = loop_start as isize - pos as isize - 3;
        if offset < i16::MIN as isize || offset > i16::MAX as isize {
            return Err(QalaError::Parse {
                span: Span::new(0, 0),
                message: format!(
                    "branch too far -- chunk exceeds i16 jump range ({offset} bytes)"
                ),
            });
        }
        self.write_op(Opcode::Jump, line);
        self.write_i16(offset as i16, line);
        Ok(())
    }

    /// produce the human-readable listing for this chunk. one line per
    /// instruction: byte offset, opcode name (uppercase), operand
    /// interpretation, source line. the format is the documented contract
    /// for Phase 6's WASM bridge and Phase 8's playground bytecode panel.
    ///
    /// byte-deterministic: same input bytes produce byte-identical output
    /// on every run. no HashMap iteration, no debug formatters; constants
    /// and program-side name tables are accessed by direct index.
    ///
    /// the `program` argument supplies the function-name and enum-variant-
    /// name lookup tables; the chunk's own constant pool supplies CONST
    /// operand renderings.
    pub fn disassemble(&self, program: &Program) -> String {
        let mut out = String::new();
        let mut off = 0usize;
        while off < self.code.len() {
            let byte = self.code[off];
            let line = self.source_lines[off];
            match Opcode::from_u8(byte) {
                None => {
                    // defensive against a corrupted byte stream (e.g. via the
                    // WASM bridge); never produced by codegen.
                    let _ = writeln!(out, "{off:>4}  <unknown:{byte:#x}>  ; line {line}");
                    off += 1;
                    continue;
                }
                Some(op) => {
                    let operand = self.operand_string(op, off, program);
                    let _ = writeln!(
                        out,
                        "{off:>4}  {name:<9} {operand:<24} ; line {line}",
                        name = op.name(),
                    );
                    off += 1 + op.operand_bytes() as usize;
                }
            }
        }
        out
    }

    /// render the operand string for the opcode starting at `off` in
    /// `code`. caller has already verified `Opcode::from_u8(code[off])`
    /// is `Some(op)`. the formatting table is the locked playground
    /// contract per the Phase 4 plan.
    fn operand_string(&self, op: Opcode, off: usize, program: &Program) -> String {
        match op {
            // CONST(u16 idx) -> `#idx value`. the constant lives in the chunk.
            Opcode::Const => {
                let idx = self.read_u16(off + 1);
                let val = self
                    .constants
                    .get(idx as usize)
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "?".to_string());
                format!("#{idx} {val}")
            }
            // GET_LOCAL / SET_LOCAL (u16 slot) -> `slot`.
            Opcode::GetLocal | Opcode::SetLocal => {
                let slot = self.read_u16(off + 1);
                format!("{slot}")
            }
            // GET_GLOBAL / SET_GLOBAL (u16 idx) -> `#idx(name)`.
            Opcode::GetGlobal | Opcode::SetGlobal => {
                let idx = self.read_u16(off + 1);
                let name = program
                    .globals
                    .get(idx as usize)
                    .map(String::as_str)
                    .unwrap_or("?");
                format!("#{idx}({name})")
            }
            // JUMP / JUMP_IF_FALSE / JUMP_IF_TRUE (i16 offset) ->
            // `+offset (-> target)` where target = (off + 1 + 2) + offset.
            Opcode::Jump | Opcode::JumpIfFalse | Opcode::JumpIfTrue => {
                let offset = self.read_i16(off + 1);
                let base = (off + 1 + 2) as isize;
                let target = base + offset as isize;
                format!("{offset:+} (-> {target})")
            }
            // CALL (u16 fn_id + u8 argc) -> `#fn_id(name)/argc`.
            Opcode::Call => {
                let fn_id = self.read_u16(off + 1);
                let argc = self.code[off + 3];
                let name = program
                    .fn_names
                    .get(fn_id as usize)
                    .map(String::as_str)
                    .unwrap_or("?");
                format!("#{fn_id}({name})/{argc}")
            }
            // MAKE_ARRAY / MAKE_TUPLE / MAKE_STRUCT (u16 count) -> `count`.
            Opcode::MakeArray | Opcode::MakeTuple | Opcode::MakeStruct => {
                let count = self.read_u16(off + 1);
                format!("{count}")
            }
            // MAKE_ENUM_VARIANT (u16 variant_id + u8 payload_count) ->
            // `#variant_id(Enum::Variant)/payload_count`.
            Opcode::MakeEnumVariant => {
                let variant_id = self.read_u16(off + 1);
                let payload = self.code[off + 3];
                let (enum_name, variant_name) = program
                    .enum_variant_names
                    .get(variant_id as usize)
                    .map(|(e, v)| (e.as_str(), v.as_str()))
                    .unwrap_or(("?", "?"));
                format!("#{variant_id}({enum_name}::{variant_name})/{payload}")
            }
            // FIELD (u16 name_index) -> `#idx`. v1 does not yet track field
            // names beyond the constant pool; the raw index is shown. the
            // playground can decorate this in Phase 8 if needed.
            Opcode::Field => {
                let idx = self.read_u16(off + 1);
                format!("#{idx}")
            }
            // CONCAT_N (u16 count) -> `count`.
            Opcode::ConcatN => {
                let count = self.read_u16(off + 1);
                format!("{count}")
            }
            // MATCH_VARIANT (u16 variant_id + i16 offset) ->
            // `#variant_id(Enum::Variant) miss-> target`. target is computed
            // from the byte AFTER the 4-byte operand layout.
            Opcode::MatchVariant => {
                let variant_id = self.read_u16(off + 1);
                let offset = self.read_i16(off + 3);
                let (enum_name, variant_name) = program
                    .enum_variant_names
                    .get(variant_id as usize)
                    .map(|(e, v)| (e.as_str(), v.as_str()))
                    .unwrap_or(("?", "?"));
                let base = (off + 1 + 4) as isize;
                let target = base + offset as isize;
                format!("#{variant_id}({enum_name}::{variant_name}) miss-> {target}")
            }
            // zero-operand opcodes: empty operand string.
            Opcode::Pop
            | Opcode::Dup
            | Opcode::Add
            | Opcode::Sub
            | Opcode::Mul
            | Opcode::Div
            | Opcode::Mod
            | Opcode::Neg
            | Opcode::FAdd
            | Opcode::FSub
            | Opcode::FMul
            | Opcode::FDiv
            | Opcode::FNeg
            | Opcode::Eq
            | Opcode::Ne
            | Opcode::Lt
            | Opcode::Le
            | Opcode::Gt
            | Opcode::Ge
            | Opcode::FEq
            | Opcode::FNe
            | Opcode::FLt
            | Opcode::FLe
            | Opcode::FGt
            | Opcode::FGe
            | Opcode::Not
            | Opcode::Return
            | Opcode::Index
            | Opcode::Len
            | Opcode::ToStr
            | Opcode::Halt => String::new(),
        }
    }
}

impl Program {
    /// construct an empty program. all vectors start empty; `main_index`
    /// defaults to 0.
    pub fn new() -> Self {
        Self::default()
    }

    /// produce the human-readable listing for the whole program: one
    /// header line per chunk followed by the chunk's body. byte-
    /// deterministic across runs by construction (iterates `chunks` in
    /// index order; no HashMap iteration). the format matches the locked
    /// playground bytecode-panel contract.
    pub fn disassemble(&self) -> String {
        let mut out = String::new();
        for (i, chunk) in self.chunks.iter().enumerate() {
            let name = self.fn_names.get(i).map(String::as_str).unwrap_or("?");
            let _ = writeln!(out, "== chunk: {name} (fn_id {i}) ==");
            out.push_str(&chunk.disassemble(self));
            if i + 1 < self.chunks.len() {
                out.push('\n');
            }
        }
        out
    }

    /// run the peephole optimizer over every chunk in this program. invokes
    /// [`crate::optimizer::peephole`]. idempotent: a second call does no
    /// work, because the first pass already collapsed every matchable
    /// instruction window.
    ///
    /// optimization is opt-in -- the disassembler can show un-optimized
    /// bytecode by skipping this call, which supports the playground's
    /// "show me what the compiler did" use case.
    pub fn optimize(&mut self) {
        for chunk in &mut self.chunks {
            crate::optimizer::peephole(chunk);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// build a chunk that emits a single CONST instruction for `v` on
    /// line 1. the constant lives at pool index 0.
    fn chunk_with_const(v: Value) -> Chunk {
        let mut c = Chunk::new();
        let idx = c.add_constant(v);
        c.write_op(Opcode::Const, 1);
        c.write_u16(idx, 1);
        c
    }

    /// build a one-chunk Program named "main" wrapping `chunk` so the
    /// disassembler has the parallel name table it needs.
    fn program_with_chunk(chunk: Chunk) -> Program {
        let mut p = Program::new();
        p.chunks.push(chunk);
        p.fn_names.push("main".to_string());
        p
    }

    // Test 1: the lockstep invariant -- after any sequence of
    // write_op/write_u16/write_i16 calls, code.len() == source_lines.len().
    #[test]
    fn code_and_source_lines_stay_in_lockstep_after_mixed_writes() {
        let mut c = Chunk::new();
        c.write_op(Opcode::Const, 1);
        c.write_u16(0, 1);
        c.write_op(Opcode::Pop, 2);
        c.write_op(Opcode::Const, 3);
        c.write_u16(1, 3);
        c.write_op(Opcode::Halt, 4);
        // CONST (1 byte + 2-byte u16) + POP (1) + CONST (1 + 2) + HALT (1)
        // = 3 + 1 + 3 + 1 = 8 bytes.
        assert_eq!(c.code.len(), 8);
        assert_eq!(c.code.len(), c.source_lines.len());
        // every byte has the line of its originating instruction; multi-
        // byte instructions duplicate the line across operand bytes.
        assert_eq!(c.source_lines, vec![1, 1, 1, 2, 3, 3, 3, 4]);
    }

    // Test 2: write_u16 + read_u16 round-trip across the full u16 range.
    #[test]
    fn write_u16_then_read_u16_is_a_round_trip_with_little_endian_bytes() {
        for v in [0u16, 1, 0xFF, 0x100, 0x1234, 0xFFFF] {
            let mut c = Chunk::new();
            c.write_u16(v, 1);
            assert_eq!(c.read_u16(0), v, "round-trip failed for {v:#x}");
        }
        // verify little-endian byte order explicitly: 0x1234 -> [0x34, 0x12].
        let mut c = Chunk::new();
        c.write_u16(0x1234, 1);
        assert_eq!(c.code, vec![0x34, 0x12]);
    }

    // Test 3: write_i16 + read_i16 round-trip across the i16 range.
    #[test]
    fn write_i16_then_read_i16_is_a_round_trip_with_twos_complement() {
        for v in [0i16, 1, -1, i16::MAX, i16::MIN, -32, 32] {
            let mut c = Chunk::new();
            c.write_i16(v, 1);
            assert_eq!(c.read_i16(0), v, "round-trip failed for {v}");
        }
    }

    // Test 4: add_constant returns sequential indices, no dedupe.
    #[test]
    fn add_constant_returns_sequential_indices_without_dedupe() {
        let mut c = Chunk::new();
        assert_eq!(c.add_constant(Value::I64(1)), 0);
        assert_eq!(c.add_constant(Value::I64(2)), 1);
        assert_eq!(c.add_constant(Value::Str("x".to_string())), 2);
        // NO dedupe: the same value pushed twice gets a new index.
        assert_eq!(c.add_constant(Value::I64(1)), 3);
        assert_eq!(c.constants.len(), 4);
    }

    // Test 5: last_const returns None when the trailing op is not CONST,
    // Some when it is.
    #[test]
    fn last_const_returns_none_when_trailing_op_is_not_const() {
        let mut c = Chunk::new();
        c.write_op(Opcode::Const, 1);
        c.write_u16(7, 1);
        c.write_op(Opcode::Pop, 2);
        // last op is POP, not CONST.
        assert_eq!(c.last_const(), None);
    }

    #[test]
    fn last_const_returns_offset_and_index_for_trailing_const_and_rewind_clears_it() {
        let mut c = Chunk::new();
        c.write_op(Opcode::Const, 1);
        c.write_u16(7, 1);
        assert_eq!(c.last_const(), Some((0, 7)));
        let idx = c.rewind_last_const().expect("rewind on a trailing CONST returns Some");
        assert_eq!(idx, 7);
        assert_eq!(c.code.len(), 0);
        assert_eq!(c.source_lines.len(), 0);
    }

    // Test 6: second_to_last_const peeks the CONST before the last CONST.
    #[test]
    fn second_to_last_const_peeks_the_const_before_the_trailing_const() {
        let mut c = Chunk::new();
        c.write_op(Opcode::Const, 1);
        c.write_u16(0, 1);
        c.write_op(Opcode::Const, 1);
        c.write_u16(1, 1);
        c.write_op(Opcode::Const, 1);
        c.write_u16(2, 1);
        // last_const() peeks the third one; second_to_last_const peeks the
        // second (byte offset 3, pool index 1).
        assert_eq!(c.last_const(), Some((6, 2)));
        assert_eq!(c.second_to_last_const(), Some((3, 1)));
    }

    #[test]
    fn second_to_last_const_is_none_when_only_one_const_is_in_the_chunk() {
        let mut c = Chunk::new();
        c.write_op(Opcode::Const, 1);
        c.write_u16(0, 1);
        assert_eq!(c.last_const(), Some((0, 0)));
        // only one CONST in the chunk; nothing before it.
        assert_eq!(c.second_to_last_const(), None);
    }

    // Test 7: the lockstep invariant survives a rewind.
    #[test]
    fn lockstep_invariant_survives_rewind_last_const() {
        let mut c = Chunk::new();
        c.write_op(Opcode::Const, 1);
        c.write_u16(0, 1);
        c.write_op(Opcode::Const, 2);
        c.write_u16(1, 2);
        assert_eq!(c.code.len(), c.source_lines.len());
        // rewind the trailing CONST; one CONST remains.
        let idx = c.rewind_last_const().expect("rewind on a trailing CONST returns Some");
        assert_eq!(idx, 1);
        assert_eq!(c.code.len(), 3);
        assert_eq!(c.source_lines.len(), 3);
        assert_eq!(c.code.len(), c.source_lines.len());
    }

    // Test 7b: rewind_last_const returns None when there is no trailing CONST.
    #[test]
    fn rewind_last_const_returns_none_when_no_trailing_const() {
        let mut c = Chunk::new();
        // empty chunk
        assert_eq!(c.rewind_last_const(), None);
        // chunk ending in a non-CONST instruction
        c.write_op(Opcode::Pop, 1);
        assert_eq!(c.rewind_last_const(), None);
        assert_eq!(c.code.len(), 1, "code must be unchanged after None return");
    }

    // Test 8: emit_jump returns the operand offset, placeholder is i16::MAX.
    #[test]
    fn emit_jump_returns_the_operand_offset_and_writes_i16_max_placeholder() {
        let mut c = Chunk::new();
        let pos = c.emit_jump(Opcode::JumpIfFalse, 5);
        // the JUMP_IF_FALSE opcode is at byte 0; the operand starts at byte 1.
        assert_eq!(pos, 1);
        assert_eq!(c.read_i16(pos), i16::MAX);
        // total emission is 3 bytes (opcode + 2-byte operand).
        assert_eq!(c.code.len(), 3);
        assert_eq!(c.source_lines, vec![5, 5, 5]);
    }

    // Test 9: patch_jump fixes the offset relative to the byte AFTER the
    // operand.
    #[test]
    fn patch_jump_writes_the_offset_to_the_current_end_of_code() {
        let mut c = Chunk::new();
        let pos = c.emit_jump(Opcode::JumpIfFalse, 5);
        // grow the chunk by 4 more bytes so the jump target is at offset 7.
        c.write_op(Opcode::Const, 6);
        c.write_u16(0, 6);
        c.write_op(Opcode::Pop, 6);
        // patch the jump to the current end (offset 7).
        // expected offset = 7 - pos - 2 = 7 - 1 - 2 = 4.
        c.patch_jump(pos).unwrap();
        assert_eq!(c.read_i16(pos), 4);
    }

    // Test 10: patch_jump errors on out-of-range.
    #[test]
    fn patch_jump_returns_branch_too_far_when_offset_exceeds_i16_max() {
        let mut c = Chunk::new();
        // pre-fill the chunk with bytes in lockstep so the jump target is
        // unreachable in i16. resize to slightly beyond i16::MAX so the
        // distance from operand_offset+2 exceeds i16::MAX.
        let len = (i16::MAX as usize) + 100;
        c.code.resize(len, 0);
        c.source_lines.resize(len, 0);
        // pretend pos=0 carries an unpatched operand; the distance from 2
        // to len (= len - 2) is > i16::MAX.
        let pos = 0usize;
        match c.patch_jump(pos) {
            Err(QalaError::Parse { message, .. }) => {
                assert!(
                    message.contains("branch too far"),
                    "expected 'branch too far' in message, got: {message:?}"
                );
            }
            other => panic!("expected QalaError::Parse, got: {other:?}"),
        }
    }

    // Test 11: emit_loop emits a negative offset that lands at loop_start.
    #[test]
    fn emit_loop_writes_a_negative_offset_that_lands_at_loop_start() {
        let mut c = Chunk::new();
        // pretend the loop entry is at byte 0; grow the chunk to byte 10.
        let loop_start = 0usize;
        // grow with a CONST instruction + some padding.
        c.write_op(Opcode::Const, 1);
        c.write_u16(0, 1); // bytes 0..=2
        c.write_op(Opcode::Const, 1);
        c.write_u16(1, 1); // bytes 3..=5
        c.write_op(Opcode::Const, 1);
        c.write_u16(2, 1); // bytes 6..=8
        c.write_op(Opcode::Pop, 1); // byte 9
        assert_eq!(c.code.len(), 10);
        // emit_loop writes JUMP + i16 offset at byte 10; offset = 0 - 10 - 3 = -13.
        c.emit_loop(loop_start, 2).unwrap();
        let operand_offset = 11; // byte after the JUMP opcode at 10.
        assert_eq!(c.read_i16(operand_offset), -13);
        // VM arithmetic: pc = (operand_offset + 2) + offset = 13 + (-13) = 0 = loop_start.
    }

    // Test 12: emit_loop errors on out-of-range backward jumps.
    #[test]
    fn emit_loop_returns_branch_too_far_when_backward_distance_exceeds_i16_min() {
        let mut c = Chunk::new();
        // grow the chunk to be far past i16::MAX bytes from the start.
        let len = (i16::MAX as usize) + 100;
        c.code.resize(len, 0);
        c.source_lines.resize(len, 0);
        // attempt a backward jump to byte 0; the offset is more negative
        // than i16::MIN.
        match c.emit_loop(0, 1) {
            Err(QalaError::Parse { message, .. }) => {
                assert!(
                    message.contains("branch too far"),
                    "expected 'branch too far' in message, got: {message:?}"
                );
            }
            other => panic!("expected QalaError::Parse, got: {other:?}"),
        }
    }

    // Test 13: disassemble of an empty chunk produces an empty string.
    #[test]
    fn disassemble_of_an_empty_chunk_produces_an_empty_string() {
        let c = Chunk::new();
        let p = Program::new();
        assert_eq!(c.disassemble(&p), "");
    }

    // Test 14: disassemble of a single CONST instruction matches the locked
    // format.
    #[test]
    fn disassemble_of_one_const_instruction_matches_the_locked_format() {
        let c = chunk_with_const(Value::I64(42));
        let p = program_with_chunk(c.clone());
        let out = c.disassemble(&p);
        // locked format: byte-offset right-aligned 4, opcode name padded 9,
        // operand padded 24, then "; line N\n".
        // operand for CONST is "#0 42".
        assert_eq!(out, "   0  CONST     #0 42                    ; line 1\n");
    }

    // Test 15: every operand kind renders per the locked table.
    #[test]
    fn disassemble_renders_every_operand_kind_per_the_locked_table() {
        let mut p = Program::new();
        p.globals.push("g".to_string());
        p.fn_names.push("main".to_string());
        p.fn_names.push("callee".to_string());
        p.enum_variant_names.push(("Shape".to_string(), "Circle".to_string()));

        let mut c = Chunk::new();
        // CONST (u16 idx)
        let const_idx = c.add_constant(Value::I64(7));
        c.write_op(Opcode::Const, 1);
        c.write_u16(const_idx, 1);
        // GET_LOCAL 0
        c.write_op(Opcode::GetLocal, 2);
        c.write_u16(0, 2);
        // GET_GLOBAL 0
        c.write_op(Opcode::GetGlobal, 3);
        c.write_u16(0, 3);
        // JUMP +5
        c.write_op(Opcode::Jump, 4);
        c.write_i16(5, 4);
        // CALL #1(callee)/2
        c.write_op(Opcode::Call, 5);
        c.write_u16(1, 5);
        c.code.push(2);
        c.source_lines.push(5);
        // MAKE_ARRAY 3
        c.write_op(Opcode::MakeArray, 6);
        c.write_u16(3, 6);
        // MAKE_ENUM_VARIANT #0(Shape::Circle)/1
        c.write_op(Opcode::MakeEnumVariant, 7);
        c.write_u16(0, 7);
        c.code.push(1);
        c.source_lines.push(7);
        // FIELD #2
        c.write_op(Opcode::Field, 8);
        c.write_u16(2, 8);
        // CONCAT_N 4
        c.write_op(Opcode::ConcatN, 9);
        c.write_u16(4, 9);
        // MATCH_VARIANT #0 miss-> target
        c.write_op(Opcode::MatchVariant, 10);
        c.write_u16(0, 10);
        c.write_i16(8, 10);

        p.chunks.push(c.clone());
        let out = c.disassemble(&p);
        // each rendered fragment must appear somewhere in the output.
        assert!(out.contains("CONST     #0 7"), "missing CONST line in:\n{out}");
        assert!(out.contains("GET_LOCAL 0"), "missing GET_LOCAL line in:\n{out}");
        assert!(out.contains("GET_GLOBAL #0(g)"), "missing GET_GLOBAL line in:\n{out}");
        assert!(out.contains("JUMP      +5"), "missing JUMP signed-offset rendering in:\n{out}");
        assert!(out.contains("CALL      #1(callee)/2"), "missing CALL line in:\n{out}");
        assert!(out.contains("MAKE_ARRAY 3"), "missing MAKE_ARRAY line in:\n{out}");
        assert!(
            out.contains("MAKE_ENUM_VARIANT #0(Shape::Circle)/1"),
            "missing MAKE_ENUM_VARIANT line in:\n{out}"
        );
        assert!(out.contains("FIELD     #2"), "missing FIELD line in:\n{out}");
        assert!(out.contains("CONCAT_N  4"), "missing CONCAT_N line in:\n{out}");
        assert!(
            out.contains("MATCH_VARIANT #0(Shape::Circle) miss-> "),
            "missing MATCH_VARIANT line in:\n{out}"
        );
    }

    // Test 16: disassemble is byte-deterministic across runs.
    #[test]
    fn disassemble_is_byte_identical_when_called_twice_on_the_same_chunk() {
        // build a non-trivial chunk covering 5+ opcode kinds and a few
        // constants.
        let mut p = Program::new();
        p.fn_names.push("main".to_string());
        p.fn_names.push("helper".to_string());

        let mut c = Chunk::new();
        let i0 = c.add_constant(Value::I64(1));
        let i1 = c.add_constant(Value::Str("hi".to_string()));
        let i2 = c.add_constant(Value::F64(2.5));

        c.write_op(Opcode::Const, 1);
        c.write_u16(i0, 1);
        c.write_op(Opcode::Const, 1);
        c.write_u16(i1, 1);
        c.write_op(Opcode::Const, 1);
        c.write_u16(i2, 1);
        c.write_op(Opcode::Add, 2);
        c.write_op(Opcode::Pop, 2);
        c.write_op(Opcode::GetLocal, 3);
        c.write_u16(0, 3);
        c.write_op(Opcode::Call, 4);
        c.write_u16(1, 4);
        c.code.push(1);
        c.source_lines.push(4);
        c.write_op(Opcode::Return, 5);

        p.chunks.push(c.clone());

        let first = c.disassemble(&p);
        let second = c.disassemble(&p);
        assert_eq!(first, second, "disassembler is non-deterministic");
        // sanity: the output spans multiple instructions.
        assert!(first.lines().count() >= 8);
    }

    // Test 17: disassemble produces exactly one line per opcode in the
    // chunk -- no skipping, no extra lines.
    #[test]
    fn disassemble_produces_one_line_per_opcode_for_every_operand_width() {
        let mut p = Program::new();
        p.fn_names.push("main".to_string());
        p.fn_names.push("f".to_string());
        p.enum_variant_names.push(("E".to_string(), "A".to_string()));

        let mut c = Chunk::new();
        // zero-operand
        c.write_op(Opcode::Pop, 1);
        // two-operand (u16)
        c.write_op(Opcode::Const, 2);
        c.write_u16(0, 2);
        // three-operand (u16 + u8)
        c.write_op(Opcode::Call, 3);
        c.write_u16(1, 3);
        c.code.push(0);
        c.source_lines.push(3);
        // four-operand (u16 + i16)
        c.write_op(Opcode::MatchVariant, 4);
        c.write_u16(0, 4);
        c.write_i16(0, 4);

        let _ = c.add_constant(Value::I64(0));
        p.chunks.push(c.clone());

        let out = c.disassemble(&p);
        // 4 instructions -> 4 lines (newline-terminated).
        assert_eq!(out.matches('\n').count(), 4, "wrong line count in:\n{out}");
    }

    // Test 18: Program::optimize runs the peephole pass over every chunk --
    // both chunks have their CONST+POP pairs removed in place.
    #[test]
    fn program_optimize_runs_peephole_over_every_chunk() {
        let mut p = Program::new();
        p.fn_names.push("main".to_string());
        p.fn_names.push("helper".to_string());

        let mut a = Chunk::new();
        let idx_a = a.add_constant(Value::I64(0));
        a.write_op(Opcode::Const, 1);
        a.write_u16(idx_a, 1);
        a.write_op(Opcode::Pop, 1);
        a.write_op(Opcode::Return, 1);

        let mut b = Chunk::new();
        let idx_b = b.add_constant(Value::I64(1));
        b.write_op(Opcode::Const, 1);
        b.write_u16(idx_b, 1);
        b.write_op(Opcode::Pop, 1);
        b.write_op(Opcode::Return, 1);

        p.chunks.push(a);
        p.chunks.push(b);

        p.optimize();

        // the CONST+POP pair is gone from each chunk; only RETURN remains.
        assert_eq!(p.chunks[0].code, vec![Opcode::Return as u8]);
        assert_eq!(p.chunks[1].code, vec![Opcode::Return as u8]);
        // the lockstep invariant survives the whole-program pass.
        assert_eq!(p.chunks[0].code.len(), p.chunks[0].source_lines.len());
        assert_eq!(p.chunks[1].code.len(), p.chunks[1].source_lines.len());
    }

    // Test 18b: Program::optimize on an empty program is a no-op (no panic).
    #[test]
    fn program_optimize_on_an_empty_program_does_nothing() {
        let mut p = Program::new();
        p.optimize();
        assert!(p.chunks.is_empty());
    }

    // Test 18c: Program::optimize is idempotent -- a second call changes
    // nothing because the first pass already collapsed every window.
    #[test]
    fn program_optimize_is_idempotent() {
        let mut p = Program::new();
        p.fn_names.push("main".to_string());

        let mut a = Chunk::new();
        let idx = a.add_constant(Value::I64(0));
        a.write_op(Opcode::Const, 1);
        a.write_u16(idx, 1);
        a.write_op(Opcode::Pop, 1);
        a.write_op(Opcode::Dup, 2);
        a.write_op(Opcode::Pop, 2);
        a.write_op(Opcode::Return, 3);
        p.chunks.push(a);

        p.optimize();
        let after_first = p.chunks[0].code.clone();
        p.optimize();
        assert_eq!(p.chunks[0].code, after_first, "second optimize changed bytes");
    }

    // bonus: Program::disassemble header + body composition is determ.
    #[test]
    fn program_disassemble_emits_a_header_per_chunk_and_concatenates_bodies() {
        let mut p = Program::new();
        p.fn_names.push("main".to_string());
        p.fn_names.push("helper".to_string());

        let mut a = chunk_with_const(Value::I64(0));
        a.write_op(Opcode::Return, 1);
        let mut b = chunk_with_const(Value::I64(1));
        b.write_op(Opcode::Return, 1);

        p.chunks.push(a);
        p.chunks.push(b);

        let out = p.disassemble();
        assert!(out.contains("== chunk: main (fn_id 0) =="));
        assert!(out.contains("== chunk: helper (fn_id 1) =="));
        // the helper's "RETURN" line follows the main's body.
        let main_idx = out.find("== chunk: main").unwrap();
        let helper_idx = out.find("== chunk: helper").unwrap();
        assert!(main_idx < helper_idx);
        // determinism: calling twice produces identical output.
        assert_eq!(p.disassemble(), p.disassemble());
    }
}
