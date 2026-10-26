//! the bytecode opcode enum: every byte the codegen emits is one of these
//! variants encoded as `Opcode::Foo as u8`. dense discriminants 0..=45 for
//! the real opcodes plus [`Opcode::Halt`] at `0xFF` as a sentinel for the
//! decoder so an out-of-bounds byte is detectable rather than silently
//! reinterpretable.
//!
//! the layout is part of the bytecode format. adding a new opcode means
//! three places: the enum variant (with an explicit discriminant), the
//! [`Opcode::from_u8`] arm, the [`Opcode::name`] arm. [`Opcode::operand_bytes`]
//! adds a fourth if the new opcode carries operands. the unit tests in this
//! module loop over every variant and over every byte 0..=255, so a missing
//! arm or a wrong discriminant fails loudly.
//!
//! the disassembler ([`crate::chunk::Chunk::disassemble`]) reads
//! the byte stream back via [`Opcode::from_u8`] and renders via
//! [`Opcode::name`]; the peephole optimizer ([`crate::optimizer::peephole`])
//! uses [`Opcode::operand_bytes`] as its step function for
//! walking past whole instructions. these three methods are the public
//! contract this file owns.
//!
//! no `transmute`-based decode (an anti-pattern for this): the
//! match-based reverse lookup compiles to a branch table and is safe on
//! every byte. zero-cost safety; idiomatic Rust pattern.

/// the first fn-id reserved for the native standard library.
///
/// codegen assigns user functions dense ids `0..N` (into `Program::chunks`)
/// and stdlib functions ids `STDLIB_FN_BASE + i`. the VM's `CALL` handler
/// branches on this threshold: a fn-id at or above it routes to the native
/// stdlib dispatcher; below it looks up a user chunk. all three modules that
/// use this value (`codegen.rs`, `vm.rs`, `stdlib.rs`) import it from here so
/// there is exactly one definition to change.
pub const STDLIB_FN_BASE: u16 = 40_000;

/// the bytecode opcode set.
///
/// one byte per opcode, dense discriminants 0..=45 for the active set plus
/// [`Opcode::Halt`] at `0xFF` as the decoder's "unknown / end-of-stream"
/// sentinel. derives `Copy` because opcodes flow through codegen and the
/// peephole optimizer by value; one byte is cheaper to copy than to
/// reference.
///
/// the discriminants are part of the bytecode format -- changing them
/// reshapes every compiled chunk. add new opcodes by appending discriminants
/// above the current high-water mark; never reuse a freed discriminant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Opcode {
    // ---- stack ----
    /// push the constant pool entry at index `u16` onto the stack. emitted
    /// for every literal and for every constant-folded result.
    Const = 0,
    /// discard the value on top of the stack. emitted at statement-expression
    /// boundaries and after expressions whose result is unused.
    Pop = 1,
    /// duplicate the value on top of the stack. used by `MATCH_VARIANT` chains
    /// so multiple arms can test the same scrutinee.
    Dup = 2,
    // ---- locals + globals ----
    /// read local slot `u16` and push its value. slots are numbered 0..N per
    /// call frame; parameters occupy slots 0..argc.
    GetLocal = 3,
    /// pop the top value and store it into local slot `u16`. used by `let mut`
    /// rebinding and by loop-variable updates.
    SetLocal = 4,
    /// read the global variable at index `u16` and push its value. globals are
    /// keyed by name in the `Program.globals` table.
    GetGlobal = 5,
    /// pop the top value and store it into the global at index `u16`. Qala v1
    /// has no top-level mutable bindings, so this is reserved for forward
    /// compatibility.
    SetGlobal = 6,
    // ---- i64 arithmetic + negation ----
    /// pop two i64 values, push their sum. codegen emits this only when both
    /// operands are statically `i64`-typed; constant folding intercepts the
    /// all-literal case at codegen time.
    Add = 7,
    /// pop two i64 values, push their difference (lhs - rhs).
    Sub = 8,
    /// pop two i64 values, push their product.
    Mul = 9,
    /// pop two i64 values, push their quotient (truncated toward zero).
    Div = 10,
    /// pop two i64 values, push their remainder (Rust's `%` semantics).
    Mod = 11,
    /// pop one i64, push its negation. emitted by the unary `-` operator on
    /// i64-typed expressions.
    Neg = 12,
    // ---- f64 arithmetic + negation ----
    /// pop two f64 values, push their sum (IEEE 754).
    FAdd = 13,
    /// pop two f64 values, push their difference (IEEE 754).
    FSub = 14,
    /// pop two f64 values, push their product (IEEE 754).
    FMul = 15,
    /// pop two f64 values, push their quotient (IEEE 754).
    FDiv = 16,
    /// pop one f64, push its negation (sign-bit flip).
    FNeg = 17,
    // ---- comparisons (i64 / bool / str -- the VM picks by operand type) ----
    /// pop two values of equal type (i64, bool, or str), push the bool result
    /// of `lhs == rhs`. the VM dispatches by operand type at runtime.
    Eq = 18,
    /// pop two values of equal type, push `lhs != rhs`.
    Ne = 19,
    /// pop two values of equal type, push `lhs < rhs`.
    Lt = 20,
    /// pop two values of equal type, push `lhs <= rhs`.
    Le = 21,
    /// pop two values of equal type, push `lhs > rhs`.
    Gt = 22,
    /// pop two values of equal type, push `lhs >= rhs`.
    Ge = 23,
    // ---- f64 comparisons ----
    /// pop two f64 values, push `lhs == rhs`. follows IEEE 754: `NaN == NaN`
    /// is `false`.
    FEq = 24,
    /// pop two f64 values, push `lhs != rhs`. follows IEEE 754: `NaN != NaN`
    /// is `true`.
    FNe = 25,
    /// pop two f64 values, push `lhs < rhs` (IEEE 754).
    FLt = 26,
    /// pop two f64 values, push `lhs <= rhs` (IEEE 754).
    FLe = 27,
    /// pop two f64 values, push `lhs > rhs` (IEEE 754).
    FGt = 28,
    /// pop two f64 values, push `lhs >= rhs` (IEEE 754).
    FGe = 29,
    // ---- logic ----
    /// pop one bool, push its negation. short-circuiting `&&` / `||` compile
    /// to jump patterns, not dedicated opcodes; this is the only logic op.
    Not = 30,
    // ---- control flow ----
    /// branch by the signed `i16` offset relative to the byte AFTER the
    /// operand. negative offsets allowed for backward jumps (loops).
    Jump = 31,
    /// pop one bool; if `false`, branch by the signed `i16` offset relative
    /// to the byte AFTER the operand. otherwise fall through.
    JumpIfFalse = 32,
    /// pop one bool; if `true`, branch by the signed `i16` offset relative to
    /// the byte AFTER the operand. otherwise fall through. emitted by the
    /// peephole rewrite `NOT; JUMP_IF_FALSE` -> `JUMP_IF_TRUE`.
    JumpIfTrue = 33,
    /// call the function at `Program.chunks[u16]` with `u8` arguments already
    /// on the stack (top is the rightmost arg). pushes the returned value
    /// (or [`crate::value::Value::Void`] for void-returning functions).
    Call = 34,
    /// return from the current call frame; the value on top of the stack
    /// becomes the call's result. void-returning functions push
    /// [`crate::value::Value::Void`] before this opcode.
    Return = 35,
    // ---- construction + access ----
    /// pop `u16` values off the stack (top is the last element), push a heap
    /// array containing them in stack order.
    MakeArray = 36,
    /// pop `u16` values off the stack (top is the last element), push a heap
    /// tuple containing them in stack order.
    MakeTuple = 37,
    /// build a heap struct. the `u16` operand is a struct id -- an index into
    /// `Program.structs`, NOT a bare field count. the VM reads the field
    /// count from `Program.structs[id].field_count`, pops that many values
    /// off the stack (top is the last field's value), and labels the struct
    /// with `Program.structs[id].name`. the field order is locked by the
    /// struct declaration; codegen emits the values in declaration order so
    /// the VM pairs them without a per-field name index.
    MakeStruct = 38,
    /// construct an enum variant value: variant id `u16`, then `u8` payload
    /// values already on the stack (top is the rightmost payload field). the
    /// VM keeps the variant id and payload as a heap object.
    MakeEnumVariant = 39,
    /// pop an index (i64) then an array value, push the array's element at
    /// that index. out-of-bounds is a runtime error.
    Index = 40,
    /// pop a struct value, push the field at name-pool index `u16`. the VM
    /// resolves the name to the struct's declared field offset.
    Field = 41,
    /// pop an array (or string), push its length as i64.
    Len = 42,
    // ---- strings + interpolation ----
    /// pop one value, push its string form. used to materialise interpolated
    /// segments whose static type is not already `str`.
    ToStr = 43,
    /// pop `u16` values off the stack (in stack order, top last), concatenate
    /// them as strings, push the result. used to materialise string
    /// interpolation.
    ConcatN = 44,
    // ---- match dispatch ----
    /// test the value on top of the stack against variant id `u16`; on match,
    /// leave the destructured payload on the stack; on miss, branch by the
    /// signed `i16` offset. consumes the scrutinee on match, leaves it on
    /// miss (so a chain of `MATCH_VARIANT` over multiple arms tests against
    /// the same scrutinee via `DUP`).
    MatchVariant = 45,
    // ---- sentinel ----
    /// sentinel discriminant `0xFF`; never emitted by codegen but useful as
    /// the disassembler's "unknown byte" marker. the VM treats it as an
    /// error.
    Halt = 0xFF,
}

impl Opcode {
    /// safe reverse lookup: the disassembler's only entry from raw bytes back
    /// to a typed [`Opcode`]. returns `None` for any undefined discriminant
    /// rather than the UB that a `transmute` would risk. the match compiles
    /// to the same branch table a `transmute`-based decode would, so the
    /// safety is free.
    pub fn from_u8(b: u8) -> Option<Opcode> {
        match b {
            0 => Some(Opcode::Const),
            1 => Some(Opcode::Pop),
            2 => Some(Opcode::Dup),
            3 => Some(Opcode::GetLocal),
            4 => Some(Opcode::SetLocal),
            5 => Some(Opcode::GetGlobal),
            6 => Some(Opcode::SetGlobal),
            7 => Some(Opcode::Add),
            8 => Some(Opcode::Sub),
            9 => Some(Opcode::Mul),
            10 => Some(Opcode::Div),
            11 => Some(Opcode::Mod),
            12 => Some(Opcode::Neg),
            13 => Some(Opcode::FAdd),
            14 => Some(Opcode::FSub),
            15 => Some(Opcode::FMul),
            16 => Some(Opcode::FDiv),
            17 => Some(Opcode::FNeg),
            18 => Some(Opcode::Eq),
            19 => Some(Opcode::Ne),
            20 => Some(Opcode::Lt),
            21 => Some(Opcode::Le),
            22 => Some(Opcode::Gt),
            23 => Some(Opcode::Ge),
            24 => Some(Opcode::FEq),
            25 => Some(Opcode::FNe),
            26 => Some(Opcode::FLt),
            27 => Some(Opcode::FLe),
            28 => Some(Opcode::FGt),
            29 => Some(Opcode::FGe),
            30 => Some(Opcode::Not),
            31 => Some(Opcode::Jump),
            32 => Some(Opcode::JumpIfFalse),
            33 => Some(Opcode::JumpIfTrue),
            34 => Some(Opcode::Call),
            35 => Some(Opcode::Return),
            36 => Some(Opcode::MakeArray),
            37 => Some(Opcode::MakeTuple),
            38 => Some(Opcode::MakeStruct),
            39 => Some(Opcode::MakeEnumVariant),
            40 => Some(Opcode::Index),
            41 => Some(Opcode::Field),
            42 => Some(Opcode::Len),
            43 => Some(Opcode::ToStr),
            44 => Some(Opcode::ConcatN),
            45 => Some(Opcode::MatchVariant),
            0xFF => Some(Opcode::Halt),
            _ => None,
        }
    }

    /// the locked uppercase identifier per [`Opcode`] used by the
    /// disassembler and the playground bytecode panel. one line per variant;
    /// a missing arm fails to compile. the strings are part of the public
    /// disassembler contract.
    pub fn name(self) -> &'static str {
        match self {
            Opcode::Const => "CONST",
            Opcode::Pop => "POP",
            Opcode::Dup => "DUP",
            Opcode::GetLocal => "GET_LOCAL",
            Opcode::SetLocal => "SET_LOCAL",
            Opcode::GetGlobal => "GET_GLOBAL",
            Opcode::SetGlobal => "SET_GLOBAL",
            Opcode::Add => "ADD",
            Opcode::Sub => "SUB",
            Opcode::Mul => "MUL",
            Opcode::Div => "DIV",
            Opcode::Mod => "MOD",
            Opcode::Neg => "NEG",
            Opcode::FAdd => "F_ADD",
            Opcode::FSub => "F_SUB",
            Opcode::FMul => "F_MUL",
            Opcode::FDiv => "F_DIV",
            Opcode::FNeg => "F_NEG",
            Opcode::Eq => "EQ",
            Opcode::Ne => "NE",
            Opcode::Lt => "LT",
            Opcode::Le => "LE",
            Opcode::Gt => "GT",
            Opcode::Ge => "GE",
            Opcode::FEq => "F_EQ",
            Opcode::FNe => "F_NE",
            Opcode::FLt => "F_LT",
            Opcode::FLe => "F_LE",
            Opcode::FGt => "F_GT",
            Opcode::FGe => "F_GE",
            Opcode::Not => "NOT",
            Opcode::Jump => "JUMP",
            Opcode::JumpIfFalse => "JUMP_IF_FALSE",
            Opcode::JumpIfTrue => "JUMP_IF_TRUE",
            Opcode::Call => "CALL",
            Opcode::Return => "RETURN",
            Opcode::MakeArray => "MAKE_ARRAY",
            Opcode::MakeTuple => "MAKE_TUPLE",
            Opcode::MakeStruct => "MAKE_STRUCT",
            Opcode::MakeEnumVariant => "MAKE_ENUM_VARIANT",
            Opcode::Index => "INDEX",
            Opcode::Field => "FIELD",
            Opcode::Len => "LEN",
            Opcode::ToStr => "TO_STR",
            Opcode::ConcatN => "CONCAT_N",
            Opcode::MatchVariant => "MATCH_VARIANT",
            Opcode::Halt => "HALT",
        }
    }

    /// the number of operand bytes following this opcode in the instruction
    /// stream. used by the peephole optimizer's instruction-step function
    /// and by [`crate::chunk::Chunk::disassemble`] to skip past operand
    /// bytes when walking the byte stream. variants reading no operand
    /// return 0; variants reading a `u16` return 2; variants reading a
    /// `u16 + u8` return 3; variants reading a `u16 + i16` return 4. v1 has
    /// no wider operand layouts.
    pub fn operand_bytes(self) -> u8 {
        match self {
            // two-byte operands (u16 index or i16 offset)
            Opcode::Const
            | Opcode::GetLocal
            | Opcode::SetLocal
            | Opcode::GetGlobal
            | Opcode::SetGlobal
            | Opcode::Jump
            | Opcode::JumpIfFalse
            | Opcode::JumpIfTrue
            | Opcode::MakeArray
            | Opcode::MakeTuple
            | Opcode::MakeStruct
            | Opcode::Field
            | Opcode::ConcatN => 2,
            // u16 + u8 (3 bytes total)
            Opcode::Call | Opcode::MakeEnumVariant => 3,
            // u16 + i16 (4 bytes total)
            Opcode::MatchVariant => 4,
            // zero-operand: stack ops, arithmetic, comparison, logic, return, halt
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
            | Opcode::Halt => 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// the source of truth for the variant list -- whenever a new variant is
    /// added to the enum, this constant must grow in lockstep, and the
    /// round-trip test will fail until it does.
    const ALL: &[Opcode] = &[
        Opcode::Const,
        Opcode::Pop,
        Opcode::Dup,
        Opcode::GetLocal,
        Opcode::SetLocal,
        Opcode::GetGlobal,
        Opcode::SetGlobal,
        Opcode::Add,
        Opcode::Sub,
        Opcode::Mul,
        Opcode::Div,
        Opcode::Mod,
        Opcode::Neg,
        Opcode::FAdd,
        Opcode::FSub,
        Opcode::FMul,
        Opcode::FDiv,
        Opcode::FNeg,
        Opcode::Eq,
        Opcode::Ne,
        Opcode::Lt,
        Opcode::Le,
        Opcode::Gt,
        Opcode::Ge,
        Opcode::FEq,
        Opcode::FNe,
        Opcode::FLt,
        Opcode::FLe,
        Opcode::FGt,
        Opcode::FGe,
        Opcode::Not,
        Opcode::Jump,
        Opcode::JumpIfFalse,
        Opcode::JumpIfTrue,
        Opcode::Call,
        Opcode::Return,
        Opcode::MakeArray,
        Opcode::MakeTuple,
        Opcode::MakeStruct,
        Opcode::MakeEnumVariant,
        Opcode::Index,
        Opcode::Field,
        Opcode::Len,
        Opcode::ToStr,
        Opcode::ConcatN,
        Opcode::MatchVariant,
        Opcode::Halt,
    ];

    /// the locked zero-operand cluster: every variant in this list must
    /// return `operand_bytes() == 0`. any new opcode that should be
    /// zero-operand belongs here and any group migration is intentional.
    const ZERO_OPERAND: &[Opcode] = &[
        Opcode::Pop,
        Opcode::Dup,
        Opcode::Add,
        Opcode::Sub,
        Opcode::Mul,
        Opcode::Div,
        Opcode::Mod,
        Opcode::Neg,
        Opcode::FAdd,
        Opcode::FSub,
        Opcode::FMul,
        Opcode::FDiv,
        Opcode::FNeg,
        Opcode::Eq,
        Opcode::Ne,
        Opcode::Lt,
        Opcode::Le,
        Opcode::Gt,
        Opcode::Ge,
        Opcode::FEq,
        Opcode::FNe,
        Opcode::FLt,
        Opcode::FLe,
        Opcode::FGt,
        Opcode::FGe,
        Opcode::Not,
        Opcode::Return,
        Opcode::Index,
        Opcode::Len,
        Opcode::ToStr,
        Opcode::Halt,
    ];

    /// the locked two-operand cluster: every variant in this list must
    /// return `operand_bytes() == 2`. covers u16 indices and i16 offsets.
    const TWO_OPERAND: &[Opcode] = &[
        Opcode::Const,
        Opcode::GetLocal,
        Opcode::SetLocal,
        Opcode::GetGlobal,
        Opcode::SetGlobal,
        Opcode::Jump,
        Opcode::JumpIfFalse,
        Opcode::JumpIfTrue,
        Opcode::MakeArray,
        Opcode::MakeTuple,
        Opcode::MakeStruct,
        Opcode::Field,
        Opcode::ConcatN,
    ];

    /// the locked three-operand cluster: u16 + u8 layouts.
    const THREE_OPERAND: &[Opcode] = &[Opcode::Call, Opcode::MakeEnumVariant];

    /// the locked four-operand cluster: u16 + i16 layouts. only
    /// [`Opcode::MatchVariant`] in v1.
    const FOUR_OPERAND: &[Opcode] = &[Opcode::MatchVariant];

    #[test]
    fn opening_discriminants_are_dense_from_zero() {
        // the byte layout is part of the locked bytecode format; pin the
        // first three discriminants so a future renumbering of the enum is
        // caught here rather than silently in the disassembler.
        assert_eq!(Opcode::Const as u8, 0);
        assert_eq!(Opcode::Pop as u8, 1);
        assert_eq!(Opcode::Dup as u8, 2);
    }

    #[test]
    fn halt_uses_the_sentinel_discriminant() {
        // the sentinel marker -- never emitted by codegen but the locked
        // "unknown byte" output for the disassembler.
        assert_eq!(Opcode::Halt as u8, 0xFF);
    }

    #[test]
    fn from_u8_round_trips_every_variant() {
        // for every defined opcode, encoding to a byte and decoding back
        // yields the same variant.
        assert_eq!(ALL.len(), 47, "ALL must list all 47 opcodes");
        for op in ALL {
            assert_eq!(
                Opcode::from_u8(*op as u8),
                Some(*op),
                "round-trip failed for {op:?}",
            );
        }
    }

    #[test]
    fn from_u8_returns_none_for_every_undefined_discriminant() {
        // sweep every byte 0..=255. for defined discriminants the decoded
        // variant's `as u8` must match the input; for undefined ones the
        // result must be `None`. this catches both missing arms and arms
        // that decode to the wrong variant.
        for b in 0u8..=255 {
            match Opcode::from_u8(b) {
                Some(op) => assert_eq!(
                    op as u8, b,
                    "from_u8({b}) returned the wrong variant {op:?}",
                ),
                None => {
                    // the undefined set is exactly the bytes not present
                    // in the active discriminant set.
                    let defined = ALL.iter().any(|op| *op as u8 == b);
                    assert!(!defined, "byte {b} is defined but from_u8 returned None");
                }
            }
        }
        // spot-check the gap between the dense set and the sentinel.
        assert_eq!(Opcode::from_u8(46), None);
        assert_eq!(Opcode::from_u8(100), None);
        assert_eq!(Opcode::from_u8(200), None);
        assert_eq!(Opcode::from_u8(254), None);
    }

    #[test]
    fn name_returns_the_locked_uppercase_string_per_variant() {
        let cases: &[(Opcode, &str)] = &[
            (Opcode::Const, "CONST"),
            (Opcode::Pop, "POP"),
            (Opcode::Dup, "DUP"),
            (Opcode::GetLocal, "GET_LOCAL"),
            (Opcode::SetLocal, "SET_LOCAL"),
            (Opcode::GetGlobal, "GET_GLOBAL"),
            (Opcode::SetGlobal, "SET_GLOBAL"),
            (Opcode::Add, "ADD"),
            (Opcode::Sub, "SUB"),
            (Opcode::Mul, "MUL"),
            (Opcode::Div, "DIV"),
            (Opcode::Mod, "MOD"),
            (Opcode::Neg, "NEG"),
            (Opcode::FAdd, "F_ADD"),
            (Opcode::FSub, "F_SUB"),
            (Opcode::FMul, "F_MUL"),
            (Opcode::FDiv, "F_DIV"),
            (Opcode::FNeg, "F_NEG"),
            (Opcode::Eq, "EQ"),
            (Opcode::Ne, "NE"),
            (Opcode::Lt, "LT"),
            (Opcode::Le, "LE"),
            (Opcode::Gt, "GT"),
            (Opcode::Ge, "GE"),
            (Opcode::FEq, "F_EQ"),
            (Opcode::FNe, "F_NE"),
            (Opcode::FLt, "F_LT"),
            (Opcode::FLe, "F_LE"),
            (Opcode::FGt, "F_GT"),
            (Opcode::FGe, "F_GE"),
            (Opcode::Not, "NOT"),
            (Opcode::Jump, "JUMP"),
            (Opcode::JumpIfFalse, "JUMP_IF_FALSE"),
            (Opcode::JumpIfTrue, "JUMP_IF_TRUE"),
            (Opcode::Call, "CALL"),
            (Opcode::Return, "RETURN"),
            (Opcode::MakeArray, "MAKE_ARRAY"),
            (Opcode::MakeTuple, "MAKE_TUPLE"),
            (Opcode::MakeStruct, "MAKE_STRUCT"),
            (Opcode::MakeEnumVariant, "MAKE_ENUM_VARIANT"),
            (Opcode::Index, "INDEX"),
            (Opcode::Field, "FIELD"),
            (Opcode::Len, "LEN"),
            (Opcode::ToStr, "TO_STR"),
            (Opcode::ConcatN, "CONCAT_N"),
            (Opcode::MatchVariant, "MATCH_VARIANT"),
            (Opcode::Halt, "HALT"),
        ];
        assert_eq!(cases.len(), ALL.len(), "name table missing a variant");
        for (op, expected) in cases {
            assert_eq!(op.name(), *expected, "wrong name for {op:?}");
            assert!(!op.name().is_empty(), "name() returned empty for {op:?}");
        }
    }

    #[test]
    fn name_is_unique_per_variant() {
        // collect every variant's name into a sorted set; the set's len
        // must equal ALL.len(), or two variants accidentally share a name.
        // a copy-paste bug here would corrupt disassembler output silently.
        let names: BTreeSet<&'static str> = ALL.iter().map(|op| op.name()).collect();
        assert_eq!(
            names.len(),
            ALL.len(),
            "duplicate names found: {names:?} (expected {} unique)",
            ALL.len(),
        );
    }

    #[test]
    fn operand_bytes_matches_the_locked_table_per_variant() {
        let cases: &[(Opcode, u8)] = &[
            (Opcode::Const, 2),
            (Opcode::Pop, 0),
            (Opcode::Dup, 0),
            (Opcode::GetLocal, 2),
            (Opcode::SetLocal, 2),
            (Opcode::GetGlobal, 2),
            (Opcode::SetGlobal, 2),
            (Opcode::Add, 0),
            (Opcode::Sub, 0),
            (Opcode::Mul, 0),
            (Opcode::Div, 0),
            (Opcode::Mod, 0),
            (Opcode::Neg, 0),
            (Opcode::FAdd, 0),
            (Opcode::FSub, 0),
            (Opcode::FMul, 0),
            (Opcode::FDiv, 0),
            (Opcode::FNeg, 0),
            (Opcode::Eq, 0),
            (Opcode::Ne, 0),
            (Opcode::Lt, 0),
            (Opcode::Le, 0),
            (Opcode::Gt, 0),
            (Opcode::Ge, 0),
            (Opcode::FEq, 0),
            (Opcode::FNe, 0),
            (Opcode::FLt, 0),
            (Opcode::FLe, 0),
            (Opcode::FGt, 0),
            (Opcode::FGe, 0),
            (Opcode::Not, 0),
            (Opcode::Jump, 2),
            (Opcode::JumpIfFalse, 2),
            (Opcode::JumpIfTrue, 2),
            (Opcode::Call, 3),
            (Opcode::Return, 0),
            (Opcode::MakeArray, 2),
            (Opcode::MakeTuple, 2),
            (Opcode::MakeStruct, 2),
            (Opcode::MakeEnumVariant, 3),
            (Opcode::Index, 0),
            (Opcode::Field, 2),
            (Opcode::Len, 0),
            (Opcode::ToStr, 0),
            (Opcode::ConcatN, 2),
            (Opcode::MatchVariant, 4),
            (Opcode::Halt, 0),
        ];
        assert_eq!(
            cases.len(),
            ALL.len(),
            "operand_bytes table missing a variant"
        );
        for (op, expected) in cases {
            assert_eq!(
                op.operand_bytes(),
                *expected,
                "wrong operand_bytes for {op:?}",
            );
        }
    }

    #[test]
    fn operand_bytes_is_bounded_by_four_across_every_variant() {
        // v1 has no opcode with more than 4 trailing bytes. a future opcode
        // wider than that must lift this assertion and the disassembler's
        // operand-reading paths in lockstep.
        for op in ALL {
            assert!(
                op.operand_bytes() <= 4,
                "{op:?} reports operand_bytes={} > 4",
                op.operand_bytes(),
            );
        }
    }

    #[test]
    fn operand_width_groups_partition_the_variant_set() {
        // every variant in the zero-operand cluster reports 0.
        for op in ZERO_OPERAND {
            assert_eq!(
                op.operand_bytes(),
                0,
                "{op:?} listed as zero-operand but reports {}",
                op.operand_bytes(),
            );
        }
        // every variant in the two-operand cluster reports 2.
        for op in TWO_OPERAND {
            assert_eq!(
                op.operand_bytes(),
                2,
                "{op:?} listed as two-operand but reports {}",
                op.operand_bytes(),
            );
        }
        // every variant in the three-operand cluster reports 3.
        for op in THREE_OPERAND {
            assert_eq!(
                op.operand_bytes(),
                3,
                "{op:?} listed as three-operand but reports {}",
                op.operand_bytes(),
            );
        }
        // every variant in the four-operand cluster reports 4.
        for op in FOUR_OPERAND {
            assert_eq!(
                op.operand_bytes(),
                4,
                "{op:?} listed as four-operand but reports {}",
                op.operand_bytes(),
            );
        }
        // the groups partition the variant set: every opcode belongs to
        // exactly one group, the union covers ALL.
        let total =
            ZERO_OPERAND.len() + TWO_OPERAND.len() + THREE_OPERAND.len() + FOUR_OPERAND.len();
        assert_eq!(
            total,
            ALL.len(),
            "operand-width groups do not cover every variant",
        );
        let mut seen: BTreeSet<u8> = BTreeSet::new();
        for op in ZERO_OPERAND
            .iter()
            .chain(TWO_OPERAND)
            .chain(THREE_OPERAND)
            .chain(FOUR_OPERAND)
        {
            assert!(
                seen.insert(*op as u8),
                "{op:?} appears in more than one operand-width group",
            );
        }
        assert_eq!(seen.len(), ALL.len());
    }
}
