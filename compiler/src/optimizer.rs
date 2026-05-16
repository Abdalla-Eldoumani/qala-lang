//! the peephole optimizer: a single-pass walk over a [`crate::chunk::Chunk`]
//! removing wasteful instruction patterns and folding constant-condition
//! jumps. eight locked patterns -- the v1 set in 04-CONTEXT.md "Optimizations"
//! -- each matched with a 2-3 instruction sliding window.
//!
//! every rewrite preserves the source line of the surviving instruction
//! (the locked OPT-02 contract); when a rewrite removes `delta` bytes at
//! offset `cut_at`, JUMP* and MATCH_VARIANT offsets whose source-to-target
//! span crossed the cut are adjusted by the locked delta. a rewrite whose
//! JUMP fix-up would overflow i16 range is SKIPPED (single-pass discipline
//! locked per OPT-03; v2 may widen to i32).
//!
//! single-pass: the optimizer does NOT iterate to a fixed point. a pattern
//! that becomes matchable only after another rewrite is missed in v1. a
//! `CONST true; JUMP_IF_FALSE off` rewrite removes both, and if the bytes
//! immediately after happen to form a second matchable window, a second
//! pass would catch it -- v1 accepts the missed cascade. the omitted
//! cascades are a v2 idea.
//!
//! the public entry [`peephole`] is invoked by
//! [`crate::chunk::Program::optimize`]. the disassembler (chunk.rs) can
//! render either form (pre-optimize or post-optimize) -- the un-optimized
//! form is a supported disassembler mode for the playground's "show me what
//! the compiler did" use case.

use crate::chunk::Chunk;
use crate::opcode::Opcode;
use crate::value::Value;

/// run the single-pass peephole optimizer over `chunk`.
///
/// walks `chunk.code` left-to-right with a sliding window (2-3
/// instructions). on each match the window is rewritten in place and the
/// JUMP / JUMP_IF_FALSE / JUMP_IF_TRUE / MATCH_VARIANT offsets that crossed
/// the rewritten region are fixed up by the byte delta. the surviving
/// instruction's source line is preserved so a runtime error after
/// optimization still points at the correct line.
///
/// idempotent: a second pass does no further work, because the first pass
/// already collapsed every window matchable in a single left-to-right scan.
/// a rewrite whose JUMP fix-up would overflow i16 range is skipped, leaving
/// that window un-rewritten (no error -- single-pass discipline locked per
/// OPT-03).
pub fn peephole(chunk: &mut Chunk) {
    let mut i: usize = 0;
    while i < chunk.code.len() {
        if let Some(rewrite) = try_match(chunk, i) {
            let replacement_len = rewrite.replacement_bytes.len();
            if apply_rewrite(chunk, i, rewrite) {
                // the cursor advances past the replacement bytes. windows to
                // the right have not been visited yet; windows whose start
                // fell inside the removed region are gone with their bytes.
                i += replacement_len;
                continue;
            }
            // apply_rewrite returned false: a JUMP fix-up would overflow i16,
            // so the rewrite was skipped. fall through and step past the
            // current instruction as if no pattern had matched.
        }
        match Opcode::from_u8(chunk.code[i]) {
            // step past the whole instruction (opcode byte + operand bytes).
            Some(op) => i += 1 + op.operand_bytes() as usize,
            // defensive: an undefined byte (never produced by codegen).
            None => i += 1,
        }
    }
}

/// a matched rewrite: how many bytes the window consumes, what bytes replace
/// it, the source line the replacement carries, and an optional new constant
/// (Pattern 8 -- the folded `CONST a; CONST b; ADD` result).
struct Rewrite {
    /// the number of bytes the matched window spans in `code`.
    window_bytes_in: usize,
    /// the bytes that replace the window. shorter than or equal to
    /// `window_bytes_in` for every v1 pattern, so the chunk only shrinks.
    /// for Pattern 8 the two operand bytes are placeholders that
    /// [`apply_rewrite`] patches with the new pool index.
    replacement_bytes: Vec<u8>,
    /// the source line of the surviving instruction. every replacement byte
    /// is given this line in the `source_lines` splice so the lockstep
    /// invariant `code.len() == source_lines.len()` holds and a runtime
    /// error still points at the right line.
    surviving_line: u32,
    /// Pattern 8 only: the folded constant to append to `chunk.constants`.
    /// `apply_rewrite` pushes it, gets the new u16 index, and patches
    /// `replacement_bytes[1..=2]` with the little-endian index.
    new_constant: Option<Value>,
}

/// try to match a locked pattern starting at byte `i` in `chunk.code`.
/// patterns are tried in declaration order; the first match wins. returns
/// `None` when no pattern matches at `i`.
fn try_match(chunk: &Chunk, i: usize) -> Option<Rewrite> {
    let code = &chunk.code;
    let const_byte = Opcode::Const as u8;
    let pop_byte = Opcode::Pop as u8;
    let dup_byte = Opcode::Dup as u8;
    let not_byte = Opcode::Not as u8;
    let jif_byte = Opcode::JumpIfFalse as u8;
    let jit_byte = Opcode::JumpIfTrue as u8;
    let jump_byte = Opcode::Jump as u8;
    let add_byte = Opcode::Add as u8;
    let fadd_byte = Opcode::FAdd as u8;

    // Pattern 1: CONST(idx) POP -> remove both. the const was never used.
    if i + 4 <= code.len() && code[i] == const_byte && code[i + 3] == pop_byte {
        return Some(Rewrite {
            window_bytes_in: 4,
            replacement_bytes: Vec::new(),
            surviving_line: 0,
            new_constant: None,
        });
    }

    // Pattern 2: DUP POP -> remove both. restores the stack to pre-DUP state.
    if i + 2 <= code.len() && code[i] == dup_byte && code[i + 1] == pop_byte {
        return Some(Rewrite {
            window_bytes_in: 2,
            replacement_bytes: Vec::new(),
            surviving_line: 0,
            new_constant: None,
        });
    }

    // Pattern 3: NOT JUMP_IF_FALSE(off) -> JUMP_IF_TRUE(off). same operand
    // bytes carry over; only the opcode byte changes.
    if i + 4 <= code.len() && code[i] == not_byte && code[i + 1] == jif_byte {
        return Some(Rewrite {
            window_bytes_in: 4,
            replacement_bytes: vec![jit_byte, code[i + 2], code[i + 3]],
            surviving_line: chunk.source_lines[i + 1],
            new_constant: None,
        });
    }

    // Patterns 4-7: CONST <bool> JUMP_IF_FALSE / JUMP_IF_TRUE.
    if i + 6 <= code.len() && code[i] == const_byte {
        let cidx = chunk.read_u16(i + 1) as usize;
        let cond = chunk.constants.get(cidx);
        let jump_op = code[i + 3];

        // Pattern 4: CONST true JUMP_IF_FALSE -> remove both. the branch is
        // never taken; control falls through.
        if jump_op == jif_byte && cond == Some(&Value::Bool(true)) {
            return Some(Rewrite {
                window_bytes_in: 6,
                replacement_bytes: Vec::new(),
                surviving_line: 0,
                new_constant: None,
            });
        }
        // Pattern 5: CONST false JUMP_IF_FALSE -> JUMP. the branch is always
        // taken; the JUMP_IF_FALSE offset bytes carry over unchanged.
        if jump_op == jif_byte && cond == Some(&Value::Bool(false)) {
            return Some(Rewrite {
                window_bytes_in: 6,
                replacement_bytes: vec![jump_byte, code[i + 4], code[i + 5]],
                surviving_line: chunk.source_lines[i + 3],
                new_constant: None,
            });
        }
        // Pattern 6: CONST true JUMP_IF_TRUE -> JUMP. the branch is always
        // taken.
        if jump_op == jit_byte && cond == Some(&Value::Bool(true)) {
            return Some(Rewrite {
                window_bytes_in: 6,
                replacement_bytes: vec![jump_byte, code[i + 4], code[i + 5]],
                surviving_line: chunk.source_lines[i + 3],
                new_constant: None,
            });
        }
        // Pattern 7: CONST false JUMP_IF_TRUE -> remove both. the branch is
        // never taken.
        if jump_op == jit_byte && cond == Some(&Value::Bool(false)) {
            return Some(Rewrite {
                window_bytes_in: 6,
                replacement_bytes: Vec::new(),
                surviving_line: 0,
                new_constant: None,
            });
        }
    }

    // Pattern 8: CONST(a) CONST(b) ADD -> CONST(a+b). a fallback fold for
    // i64+i64 (Opcode::Add) or f64+f64 (Opcode::FAdd) that the codegen
    // inline folder missed. ADD only -- other binary ops are caught by
    // inline folding, whose type matrix is wider (bool / str / comparisons).
    if i + 7 <= code.len()
        && code[i] == const_byte
        && code[i + 3] == const_byte
        && (code[i + 6] == add_byte || code[i + 6] == fadd_byte)
    {
        let a_idx = chunk.read_u16(i + 1) as usize;
        let b_idx = chunk.read_u16(i + 4) as usize;
        let a = chunk.constants.get(a_idx);
        let b = chunk.constants.get(b_idx);
        let folded = match (a, b, code[i + 6]) {
            // i64 + i64: checked_add. on overflow return None so the window
            // is left un-rewritten -- the codegen inline folder is the
            // user-facing IntegerOverflow path; if a fold reaches peephole
            // and overflows, v1 leaves the bytes rather than erroring (an
            // error here would break the build silently).
            (Some(Value::I64(x)), Some(Value::I64(y)), b) if b == add_byte => {
                x.checked_add(*y).map(Value::I64)
            }
            // f64 + f64: IEEE 754 add. inf / NaN are valid fold results.
            (Some(Value::F64(x)), Some(Value::F64(y)), b) if b == fadd_byte => {
                Some(Value::F64(x + y))
            }
            // any other type pair (or opcode/type mismatch): no fold.
            _ => None,
        };
        if let Some(value) = folded {
            return Some(Rewrite {
                window_bytes_in: 7,
                // [Const, lo, hi] -- the operand bytes are placeholders
                // patched by apply_rewrite once the new index is known.
                replacement_bytes: vec![const_byte, 0, 0],
                surviving_line: chunk.source_lines[i + 6],
                new_constant: Some(value),
            });
        }
    }

    None
}

/// apply `rewrite` at byte `at` in `chunk`. returns `true` when the rewrite
/// was applied, `false` when a JUMP fix-up overflow forced a skip (in which
/// case `chunk` is left untouched).
///
/// the order is: resolve the Pattern-8 constant index, pre-validate that no
/// JUMP / MATCH_VARIANT offset would overflow i16 after the splice, then
/// splice `code` and `source_lines` in lockstep, then fix the offsets.
fn apply_rewrite(chunk: &mut Chunk, at: usize, mut rewrite: Rewrite) -> bool {
    let delta = rewrite.window_bytes_in - rewrite.replacement_bytes.len();

    // pre-validate first, BEFORE mutating anything: if any JUMP fix-up would
    // overflow i16, skip the whole rewrite. this keeps peephole correct and
    // idempotent (the alternative -- erroring mid-rewrite -- would not).
    if delta > 0 && !prevalidate_jump_offsets(chunk, at, delta) {
        return false;
    }

    // Pattern 8: append the folded constant and patch the placeholder
    // operand bytes with its little-endian pool index.
    if let Some(value) = rewrite.new_constant.take() {
        let idx = chunk.add_constant(value);
        let bytes = idx.to_le_bytes();
        // replacement_bytes is [Const, 0, 0]; positions 1 and 2 are the u16.
        rewrite.replacement_bytes[1] = bytes[0];
        rewrite.replacement_bytes[2] = bytes[1];
    }

    // fix up JUMP* and MATCH_VARIANT offsets that crossed the cut. this runs
    // BEFORE the splice so every instruction position the case analysis sees
    // is in the same coordinate system as `at` and `delta`; an offset
    // operand belonging to a jump outside the cut survives the splice
    // unchanged (every locked pattern matches whole instructions starting at
    // an instruction boundary, so no jump straddles the cut).
    if delta > 0 {
        fix_jump_offsets(chunk, at, delta);
    }

    // splice the byte stream and the source map in lockstep. every
    // replacement byte gets the surviving instruction's line.
    let end = at + rewrite.window_bytes_in;
    chunk
        .code
        .splice(at..end, rewrite.replacement_bytes.iter().copied());
    chunk.source_lines.splice(
        at..end,
        std::iter::repeat_n(rewrite.surviving_line, rewrite.replacement_bytes.len()),
    );
    debug_assert_eq!(
        chunk.code.len(),
        chunk.source_lines.len(),
        "peephole broke the code/source_lines lockstep invariant"
    );
    true
}

/// the byte position of the i16 offset operand for a relative-jump opcode at
/// `op_pos`, or `None` if the opcode is not a relative jump. the three
/// `JUMP*` opcodes carry the offset immediately after the opcode byte;
/// `MATCH_VARIANT` carries a u16 variant id first, so its offset is two
/// bytes further on.
fn jump_offset_pos(op: Opcode, op_pos: usize) -> Option<usize> {
    match op {
        Opcode::Jump | Opcode::JumpIfFalse | Opcode::JumpIfTrue => Some(op_pos + 1),
        Opcode::MatchVariant => Some(op_pos + 3),
        _ => None,
    }
}

/// compute the adjusted offset for a relative jump whose opcode byte is at
/// `op_pos` and whose i16 offset operand is at `offset_pos`, given that
/// `delta` bytes were removed at `cut_at`. returns `Some(new_offset)` when
/// the jump's source-to-target span crossed the cut and the offset must
/// change; `None` when no change is needed.
///
/// the absolute target is `offset_pos + 2 + offset` -- the byte AFTER the
/// 2-byte operand, per the VM's branch arithmetic.
fn adjusted_offset(
    op_pos: usize,
    offset_pos: usize,
    offset: i16,
    cut_at: usize,
    delta: usize,
) -> Option<isize> {
    let off = offset as isize;
    let target = offset_pos as isize + 2 + off;
    let src = op_pos as isize;
    let cut = cut_at as isize;
    let cut_end = (cut_at + delta) as isize;
    let d = delta as isize;

    if src < cut && target >= cut_end {
        // forward jump from before the cut to after it: the target shifts
        // back by `delta`, the source stays put, so the offset shrinks.
        Some(off - d)
    } else if src >= cut_end && target < cut {
        // backward jump from after the cut to before it: the source shifts
        // back by `delta`, the target stays, so the (negative) offset's
        // absolute value shrinks -- the offset value increases by `delta`.
        Some(off + d)
    } else {
        // entirely on one side of the cut, or entirely within it: no change.
        None
    }
}

/// walk `chunk.code` and fix every JUMP* / MATCH_VARIANT offset that crossed
/// the cut at `cut_at` where `delta` bytes will be removed. called by
/// [`apply_rewrite`] BEFORE the splice, so every position here is in the
/// same (pre-splice) coordinate system as `cut_at`. a jump whose opcode byte
/// lies inside the removed window is skipped -- its bytes vanish in the
/// splice, so its offset is moot.
fn fix_jump_offsets(chunk: &mut Chunk, cut_at: usize, delta: usize) {
    let mut j = 0usize;
    while j < chunk.code.len() {
        let Some(op) = Opcode::from_u8(chunk.code[j]) else {
            // defensive: an undefined byte (never produced by codegen).
            j += 1;
            continue;
        };
        let size = 1 + op.operand_bytes() as usize;
        if let Some(offset_pos) = jump_offset_pos(op, j) {
            let inside_cut = j >= cut_at && j < cut_at + delta;
            if !inside_cut && offset_pos + 2 <= chunk.code.len() {
                let offset = chunk.read_i16(offset_pos);
                if let Some(new_off) = adjusted_offset(j, offset_pos, offset, cut_at, delta) {
                    // prevalidation guaranteed this fits i16; the cast is
                    // exact. clamp defensively so a logic slip cannot panic.
                    let clamped = new_off.clamp(i16::MIN as isize, i16::MAX as isize) as i16;
                    let bytes = clamped.to_le_bytes();
                    chunk.code[offset_pos] = bytes[0];
                    chunk.code[offset_pos + 1] = bytes[1];
                }
            }
        }
        j += size;
    }
}

/// the read-only twin of [`fix_jump_offsets`]: walk `chunk.code` BEFORE the
/// splice and return `false` if any JUMP* / MATCH_VARIANT offset would
/// overflow i16 range after `delta` bytes are removed at `cut_at`.
///
/// because the matched window's bytes are all removed (every v1 pattern
/// either deletes the window or replaces it with a same-width opcode), a
/// jump whose opcode byte lies inside `[cut_at, cut_at + delta)` is itself
/// being removed -- such a jump is skipped here (its bytes are gone after
/// the splice, so its offset is moot).
fn prevalidate_jump_offsets(chunk: &Chunk, cut_at: usize, delta: usize) -> bool {
    let mut j = 0usize;
    while j < chunk.code.len() {
        let Some(op) = Opcode::from_u8(chunk.code[j]) else {
            j += 1;
            continue;
        };
        let size = 1 + op.operand_bytes() as usize;
        if let Some(offset_pos) = jump_offset_pos(op, j) {
            // skip a jump whose opcode byte is inside the removed window --
            // its bytes vanish in the splice.
            let inside_cut = j >= cut_at && j < cut_at + delta;
            if !inside_cut && offset_pos + 2 <= chunk.code.len() {
                let offset = chunk.read_i16(offset_pos);
                if let Some(new_off) = adjusted_offset(j, offset_pos, offset, cut_at, delta)
                    && (new_off < i16::MIN as isize || new_off > i16::MAX as isize)
                {
                    return false;
                }
            }
        }
        j += size;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::Program;
    use crate::lexer::Lexer;
    use crate::parser::Parser;
    use crate::typechecker::check_program;

    /// emit a CONST instruction for pool index `idx` on `line`.
    fn write_const(c: &mut Chunk, idx: u16, line: u32) {
        c.write_op(Opcode::Const, line);
        c.write_u16(idx, line);
    }

    /// emit a JUMP*-family opcode with an explicit i16 offset on `line`.
    fn write_jump(c: &mut Chunk, op: Opcode, offset: i16, line: u32) {
        c.write_op(op, line);
        c.write_i16(offset, line);
    }

    /// compile a source string to a Program through the full front end. used
    /// by the property + smoke tests; panics (test-only) on any front-end
    /// error.
    fn compile(src: &str) -> Program {
        let tokens = Lexer::tokenize(src).expect("lex");
        let ast = Parser::parse(&tokens).expect("parse");
        let (typed, errors, _) = check_program(&ast, src);
        assert!(errors.is_empty(), "typecheck errors: {errors:?}");
        crate::codegen::compile_program(&typed, src).expect("compile")
    }

    // Test 1 (Pattern 1 -- CONST/POP): the pair is removed, leaving HALT.
    #[test]
    fn pattern_1_const_pop_pair_is_removed() {
        let mut c = Chunk::new();
        let idx = c.add_constant(Value::I64(7));
        write_const(&mut c, idx, 1);
        c.write_op(Opcode::Pop, 1);
        c.write_op(Opcode::Halt, 2);
        peephole(&mut c);
        assert_eq!(c.code, vec![Opcode::Halt as u8]);
        assert_eq!(c.code.len(), 1);
        assert_eq!(c.source_lines.len(), 1);
        assert_eq!(c.source_lines, vec![2]);
    }

    // Test 2 (Pattern 2 -- DUP/POP): the pair is removed, the CONST and
    // RETURN around it survive.
    #[test]
    fn pattern_2_dup_pop_pair_is_removed() {
        let mut c = Chunk::new();
        let idx = c.add_constant(Value::I64(0));
        write_const(&mut c, idx, 1);
        c.write_op(Opcode::Dup, 1);
        c.write_op(Opcode::Pop, 1);
        c.write_op(Opcode::Return, 2);
        peephole(&mut c);
        assert_eq!(
            c.code,
            vec![Opcode::Const as u8, 0, 0, Opcode::Return as u8]
        );
        assert_eq!(c.code.len(), c.source_lines.len());
    }

    // Test 3 (Pattern 3 -- NOT/JUMP_IF_FALSE): the opcode byte becomes
    // JUMP_IF_TRUE; the offset operand is unchanged.
    #[test]
    fn pattern_3_not_jump_if_false_becomes_jump_if_true() {
        let mut c = Chunk::new();
        c.write_op(Opcode::Not, 1);
        write_jump(&mut c, Opcode::JumpIfFalse, 5, 1);
        c.write_op(Opcode::Halt, 2);
        peephole(&mut c);
        assert_eq!(c.code[0], Opcode::JumpIfTrue as u8);
        // the offset operand at byte 1 still reads 5.
        assert_eq!(c.read_i16(1), 5);
        assert_eq!(c.code[3], Opcode::Halt as u8);
        assert_eq!(c.code.len(), 4);
        assert_eq!(c.code.len(), c.source_lines.len());
    }

    // Test 4 (Pattern 4 -- CONST true + JUMP_IF_FALSE): both removed -- the
    // jump would never have been taken.
    #[test]
    fn pattern_4_const_true_jump_if_false_is_removed() {
        let mut c = Chunk::new();
        let idx = c.add_constant(Value::Bool(true));
        write_const(&mut c, idx, 1);
        write_jump(&mut c, Opcode::JumpIfFalse, 5, 1);
        c.write_op(Opcode::Halt, 2);
        peephole(&mut c);
        assert_eq!(c.code, vec![Opcode::Halt as u8]);
        assert_eq!(c.code.len(), c.source_lines.len());
    }

    // Test 5 (Pattern 5 -- CONST false + JUMP_IF_FALSE -> JUMP): the CONST is
    // dropped, the JUMP_IF_FALSE becomes an unconditional JUMP with the same
    // offset.
    #[test]
    fn pattern_5_const_false_jump_if_false_becomes_jump() {
        let mut c = Chunk::new();
        let idx = c.add_constant(Value::Bool(false));
        write_const(&mut c, idx, 1);
        write_jump(&mut c, Opcode::JumpIfFalse, 5, 1);
        c.write_op(Opcode::Halt, 2);
        peephole(&mut c);
        assert_eq!(c.code[0], Opcode::Jump as u8);
        assert_eq!(c.read_i16(1), 5);
        assert_eq!(c.code[3], Opcode::Halt as u8);
        assert_eq!(c.code.len(), 4);
        assert_eq!(c.code.len(), c.source_lines.len());
    }

    // Test 6 (Pattern 6 -- CONST true + JUMP_IF_TRUE -> JUMP): mirror of
    // Pattern 5 for the true-condition case.
    #[test]
    fn pattern_6_const_true_jump_if_true_becomes_jump() {
        let mut c = Chunk::new();
        let idx = c.add_constant(Value::Bool(true));
        write_const(&mut c, idx, 1);
        write_jump(&mut c, Opcode::JumpIfTrue, 9, 1);
        c.write_op(Opcode::Halt, 2);
        peephole(&mut c);
        assert_eq!(c.code[0], Opcode::Jump as u8);
        assert_eq!(c.read_i16(1), 9);
        assert_eq!(c.code[3], Opcode::Halt as u8);
        assert_eq!(c.code.len(), c.source_lines.len());
    }

    // Test 7 (Pattern 7 -- CONST false + JUMP_IF_TRUE): both removed.
    #[test]
    fn pattern_7_const_false_jump_if_true_is_removed() {
        let mut c = Chunk::new();
        let idx = c.add_constant(Value::Bool(false));
        write_const(&mut c, idx, 1);
        write_jump(&mut c, Opcode::JumpIfTrue, 5, 1);
        c.write_op(Opcode::Halt, 2);
        peephole(&mut c);
        assert_eq!(c.code, vec![Opcode::Halt as u8]);
        assert_eq!(c.code.len(), c.source_lines.len());
    }

    // Test 8 (Pattern 8 -- i64 ADD fold): CONST 2 + CONST 3 + ADD folds to a
    // single CONST 5. the original 2 and 3 stay in the pool (append-only);
    // the result is a NEW pool entry.
    #[test]
    fn pattern_8_folds_two_i64_const_add() {
        let mut c = Chunk::new();
        let a = c.add_constant(Value::I64(2));
        let b = c.add_constant(Value::I64(3));
        write_const(&mut c, a, 1);
        write_const(&mut c, b, 1);
        c.write_op(Opcode::Add, 1);
        c.write_op(Opcode::Halt, 2);
        peephole(&mut c);
        // the chunk is now CONST(new_idx) + HALT.
        assert_eq!(c.code[0], Opcode::Const as u8);
        assert_eq!(c.code[3], Opcode::Halt as u8);
        assert_eq!(c.code.len(), 4);
        // the folded constant is the latest pool entry, value 5.
        let new_idx = c.read_u16(1);
        assert_eq!(c.constants[new_idx as usize], Value::I64(5));
        // the two original constants are still present at indices 0 and 1.
        assert_eq!(c.constants[0], Value::I64(2));
        assert_eq!(c.constants[1], Value::I64(3));
        assert_eq!(c.constants.len(), 3);
    }

    // Test 9 (Pattern 8 -- f64 ADD fold): CONST 1.5 + CONST 2.5 + F_ADD folds
    // to CONST 4.0.
    #[test]
    fn pattern_8_folds_two_f64_const_fadd() {
        let mut c = Chunk::new();
        let a = c.add_constant(Value::F64(1.5));
        let b = c.add_constant(Value::F64(2.5));
        write_const(&mut c, a, 1);
        write_const(&mut c, b, 1);
        c.write_op(Opcode::FAdd, 1);
        c.write_op(Opcode::Halt, 2);
        peephole(&mut c);
        assert_eq!(c.code[0], Opcode::Const as u8);
        assert_eq!(c.code[3], Opcode::Halt as u8);
        let new_idx = c.read_u16(1);
        assert_eq!(c.constants[new_idx as usize], Value::F64(4.0));
    }

    // Test 10 (Pattern 8 -- i64 overflow): the fold WOULD overflow; per the
    // locked policy the rewrite is skipped and the chunk is unchanged.
    #[test]
    fn pattern_8_skips_fold_on_i64_overflow() {
        let mut c = Chunk::new();
        let a = c.add_constant(Value::I64(i64::MAX));
        let b = c.add_constant(Value::I64(1));
        write_const(&mut c, a, 1);
        write_const(&mut c, b, 1);
        c.write_op(Opcode::Add, 1);
        c.write_op(Opcode::Halt, 2);
        let before = c.code.clone();
        peephole(&mut c);
        // i64::MAX + 1 overflows: no fold, the bytes are left as they were.
        assert_eq!(c.code, before);
        assert_eq!(c.constants.len(), 2);
    }

    // Test 11 (source line preserved): a NOT and a JUMP_IF_FALSE both on
    // line 5; after the rewrite the JUMP_IF_TRUE is on line 5 too.
    #[test]
    fn pattern_3_preserves_the_surviving_source_line() {
        let mut c = Chunk::new();
        c.write_op(Opcode::Not, 5);
        write_jump(&mut c, Opcode::JumpIfFalse, 3, 5);
        peephole(&mut c);
        // JUMP_IF_TRUE opcode byte + 2 operand bytes, all line 5.
        assert_eq!(c.source_lines, vec![5, 5, 5]);
    }

    // Test 12 (JUMP fix-up forward): a forward JUMP whose target lies past a
    // CONST+POP pair has its offset shortened by the 4 removed bytes.
    #[test]
    fn jump_offset_is_fixed_up_after_a_forward_removal() {
        // layout: JUMP(+5) | CONST(0)+POP (4 bytes) | HALT.
        // the JUMP is at byte 0, operand at 1..=2; target = (1+2)+5 = 8.
        // byte 8 is the HALT (3 JUMP + 4 const-pop + 1 halt = 8). after the
        // CONST+POP pair is removed the HALT moves to byte 4; the JUMP's
        // offset must drop by 4 to +1: (1+2)+1 = 4.
        let mut c = Chunk::new();
        let idx = c.add_constant(Value::I64(0));
        write_jump(&mut c, Opcode::Jump, 5, 1);
        write_const(&mut c, idx, 1);
        c.write_op(Opcode::Pop, 1);
        c.write_op(Opcode::Halt, 2);
        assert_eq!(c.code.len(), 8);
        peephole(&mut c);
        // CONST+POP removed -> 4 bytes; JUMP + HALT remain (4 bytes).
        assert_eq!(c.code.len(), 4);
        assert_eq!(c.code[0], Opcode::Jump as u8);
        // the offset is now +1, still landing on the HALT at byte 4.
        assert_eq!(c.read_i16(1), 1);
        assert_eq!(c.code[3], Opcode::Halt as u8);
    }

    // Test 13 (JUMP fix-up backward): a backward JUMP whose target is before
    // a removed CONST+POP pair has its negative offset moved toward zero.
    #[test]
    fn backward_jump_offset_is_fixed_up_after_a_removal() {
        // layout: HALT(byte 0) | CONST(0)+POP (bytes 1..=4) | JUMP back to 0.
        // the JUMP opcode is at byte 5, operand at 6..=7; target byte AFTER
        // the operand is 8, so offset = 0 - 8 = -8. after the CONST+POP pair
        // (bytes 1..=4) is removed the JUMP slides to byte 1, operand 2..=3,
        // byte-after = 4; to still reach byte 0 the offset must be -4.
        let mut c = Chunk::new();
        let idx = c.add_constant(Value::I64(0));
        c.write_op(Opcode::Halt, 1);
        write_const(&mut c, idx, 2);
        c.write_op(Opcode::Pop, 2);
        write_jump(&mut c, Opcode::Jump, -8, 3);
        assert_eq!(c.code.len(), 8);
        peephole(&mut c);
        assert_eq!(c.code.len(), 4);
        // the JUMP is now at byte 1; its offset moved from -8 to -4.
        assert_eq!(c.code[1], Opcode::Jump as u8);
        assert_eq!(c.read_i16(2), -4);
        // VM arithmetic: (operand_pos + 2) + offset = (2 + 2) + (-4) = 0.
    }

    // Test 14 (multiple rewrites in one pass): CONST+POP followed by DUP+POP;
    // a single left-to-right pass removes both pairs.
    #[test]
    fn multiple_patterns_collapse_in_a_single_pass() {
        let mut c = Chunk::new();
        let idx = c.add_constant(Value::I64(0));
        write_const(&mut c, idx, 1);
        c.write_op(Opcode::Pop, 1);
        c.write_op(Opcode::Dup, 2);
        c.write_op(Opcode::Pop, 2);
        peephole(&mut c);
        assert!(c.code.is_empty(), "both pairs removed -> empty chunk");
        assert_eq!(c.source_lines.len(), 0);
    }

    // Test 15 (idempotence): a second pass produces a byte-identical chunk.
    #[test]
    fn peephole_is_idempotent_within_a_single_pass() {
        let mut c = Chunk::new();
        let t = c.add_constant(Value::Bool(false));
        write_const(&mut c, t, 1);
        write_jump(&mut c, Opcode::JumpIfFalse, 7, 1);
        c.write_op(Opcode::Not, 2);
        write_jump(&mut c, Opcode::JumpIfFalse, 4, 2);
        c.write_op(Opcode::Halt, 3);
        peephole(&mut c);
        let after_first = c.code.clone();
        let lines_first = c.source_lines.clone();
        peephole(&mut c);
        assert_eq!(c.code, after_first, "second pass changed the bytes");
        assert_eq!(c.source_lines, lines_first, "second pass changed the lines");
    }

    // Test 16 (MATCH_VARIANT fix-up): a MATCH_VARIANT whose miss-jump target
    // lies past a removed CONST+POP pair has its offset adjusted by the
    // delta -- the offset sits at op_byte_pos + 3, after the u16 variant id.
    #[test]
    fn match_variant_offset_is_fixed_up_after_a_removal() {
        // layout: MATCH_VARIANT(variant 0, offset) (5 bytes) |
        //         CONST(0)+POP (4 bytes) | HALT.
        // MATCH_VARIANT opcode at byte 0; offset operand at bytes 3..=4;
        // byte AFTER the operand is 5. to land on the HALT at byte 9 the
        // offset is 9 - 5 = +4. after the CONST+POP pair is removed the HALT
        // moves to byte 5; the offset must drop by 4 to 0.
        let mut c = Chunk::new();
        let idx = c.add_constant(Value::I64(0));
        c.write_op(Opcode::MatchVariant, 1);
        c.write_u16(0, 1); // variant id
        c.write_i16(4, 1); // miss offset
        write_const(&mut c, idx, 1);
        c.write_op(Opcode::Pop, 1);
        c.write_op(Opcode::Halt, 2);
        assert_eq!(c.code.len(), 10);
        peephole(&mut c);
        assert_eq!(c.code.len(), 6);
        assert_eq!(c.code[0], Opcode::MatchVariant as u8);
        // the miss offset (at byte 3) dropped from +4 to 0.
        assert_eq!(c.read_i16(3), 0);
        assert_eq!(c.code[5], Opcode::Halt as u8);
    }

    // Test 17 (lockstep invariant): code.len() == source_lines.len() after a
    // rewrite. asserted as an explicit property on a mixed chunk.
    #[test]
    fn lockstep_invariant_holds_after_rewrites() {
        let mut c = Chunk::new();
        let n = c.add_constant(Value::I64(1));
        let t = c.add_constant(Value::Bool(true));
        write_const(&mut c, n, 1);
        c.write_op(Opcode::Pop, 1);
        write_const(&mut c, t, 2);
        write_jump(&mut c, Opcode::JumpIfFalse, 3, 2);
        c.write_op(Opcode::Halt, 3);
        peephole(&mut c);
        assert_eq!(
            c.code.len(),
            c.source_lines.len(),
            "code and source_lines drifted"
        );
    }

    // Test 18 (source-line preservation property): every post-peephole
    // instruction's source line stays within the program's line range.
    #[test]
    fn source_lines_stay_within_the_program_line_range_after_optimize() {
        // a 4-line program; every source line is in 1..=4.
        let src = "fn main() is io {\n\
                   let x = 1 + 2\n\
                   println(\"{x}\")\n\
                   }";
        let line_count = src.lines().count() as u32;
        let mut program = compile(src);
        program.optimize();
        for (ci, chunk) in program.chunks.iter().enumerate() {
            for (bi, &line) in chunk.source_lines.iter().enumerate() {
                assert!(
                    (1..=line_count).contains(&line),
                    "chunk {ci} byte {bi}: source line {line} outside 1..={line_count}"
                );
            }
            // the lockstep invariant survives the whole-program optimize.
            assert_eq!(chunk.code.len(), chunk.source_lines.len());
        }
    }

    // Test 19 (six-bundled-examples smoke test): every example still
    // disassembles non-empty after optimize, and no fn-chunk loses its
    // terminating RETURN / HALT.
    #[test]
    fn six_bundled_examples_optimize_without_regression() {
        for name in [
            "hello",
            "fibonacci",
            "effects",
            "pattern-matching",
            "pipeline",
            "defer-demo",
        ] {
            let path = format!(
                "{}/../playground/public/examples/{name}.qala",
                env!("CARGO_MANIFEST_DIR"),
            );
            let src = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("read {path}: {e}"));
            let tokens = Lexer::tokenize(&src)
                .unwrap_or_else(|e| panic!("{name}.qala: lex: {e:?}"));
            let ast = Parser::parse(&tokens)
                .unwrap_or_else(|e| panic!("{name}.qala: parse: {e:?}"));
            let (typed, terrors, _) = check_program(&ast, &src);
            assert!(terrors.is_empty(), "{name}.qala: typecheck: {terrors:?}");
            let mut program = crate::codegen::compile_program(&typed, &src)
                .unwrap_or_else(|e| panic!("{name}.qala: compile: {e:?}"));
            program.optimize();
            assert!(
                !program.chunks.is_empty(),
                "{name}.qala: no chunks after optimize"
            );
            for (i, chunk) in program.chunks.iter().enumerate() {
                assert!(
                    !chunk.code.is_empty(),
                    "{name}.qala: chunk {i} empty after optimize"
                );
                assert_eq!(
                    chunk.code.len(),
                    chunk.source_lines.len(),
                    "{name}.qala: chunk {i} lockstep broken after optimize"
                );
                // every fn-chunk still ends in RETURN or HALT.
                let last = *chunk.code.last().expect("non-empty chunk");
                assert!(
                    last == Opcode::Return as u8 || last == Opcode::Halt as u8,
                    "{name}.qala: chunk {i} lost its terminator"
                );
            }
            assert!(
                !program.disassemble().is_empty(),
                "{name}.qala: disassembly empty after optimize"
            );
        }
    }

    // Test 20 (optimize-twice determinism): compiling and optimizing the
    // same program twice produces byte-identical chunks.
    #[test]
    fn optimize_is_byte_deterministic_across_runs() {
        let src = "fn add(a: i64, b: i64) -> i64 { a + b }\n\
                   fn main() is io {\n\
                   let r = add(1, 2)\n\
                   println(\"{r}\")\n\
                   }";
        let mut first = compile(src);
        let mut second = compile(src);
        first.optimize();
        second.optimize();
        assert_eq!(first.chunks.len(), second.chunks.len());
        for (a, b) in first.chunks.iter().zip(second.chunks.iter()) {
            assert_eq!(a.code, b.code, "optimized code differs across runs");
            assert_eq!(
                a.source_lines, b.source_lines,
                "optimized source map differs across runs"
            );
        }
        assert_eq!(
            first.disassemble(),
            second.disassemble(),
            "optimized disassembly is non-deterministic"
        );
    }

    // bonus: a no-op on an already-tight chunk -- no pattern, no change.
    #[test]
    fn peephole_leaves_an_already_tight_chunk_unchanged() {
        let mut c = Chunk::new();
        let idx = c.add_constant(Value::I64(1));
        write_const(&mut c, idx, 1);
        c.write_op(Opcode::Return, 1);
        let before = c.code.clone();
        let lines = c.source_lines.clone();
        peephole(&mut c);
        assert_eq!(c.code, before);
        assert_eq!(c.source_lines, lines);
    }

    // bonus: an empty chunk is handled without panic.
    #[test]
    fn peephole_on_an_empty_chunk_is_a_no_op() {
        let mut c = Chunk::new();
        peephole(&mut c);
        assert!(c.code.is_empty());
        assert!(c.source_lines.is_empty());
    }
}
