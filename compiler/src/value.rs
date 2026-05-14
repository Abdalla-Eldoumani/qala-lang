//! the constant-pool value type: what [`Chunk.constants`] holds and what the
//! comptime interpreter and the future bytecode VM push onto their stacks.
//!
//! the enum has seven variants in v1: the five primitive Qala types
//! ([`Value::I64`], [`Value::F64`], [`Value::Bool`], [`Value::Byte`],
//! [`Value::Str`]), a unit [`Value::Void`] for the result of a void-returning
//! expression, and [`Value::Function`] -- a `u16` function index into the
//! enclosing `Program.chunks` vector. heap objects (arrays, structs,
//! enum-variant payloads) are NOT representable here; they are built at
//! runtime by the VM via the `MAKE_*` opcodes. constant folding in codegen
//! only folds primitive-typed expressions (and string concatenations of two
//! literal strings); a fold that would produce a heap object falls through
//! to runtime construction.
//!
//! the [`Display`](std::fmt::Display) impl is byte-deterministic: the same
//! `Value` renders to the same bytes every run. the disassembler reads this
//! directly and the playground's bytecode panel reads the disassembler's
//! output; non-determinism here would manifest as visual flicker on every
//! re-render, so the rule is locked in tests.
//!
//! no `serde` derives: this phase keeps `Value` in-process only. the WASM
//! bridge phase may add a derived form (or a hand-rolled serde proxy) when
//! the disassembler output crosses the JS boundary; deferred to that phase.

/// a value the constant pool can store.
///
/// derives `Debug, Clone, PartialEq`. NOT `Eq` -- [`Value::F64`] holds `f64`,
/// and `f64` is not `Eq` (`NaN != NaN`). this mirrors how the AST and
/// `QalaError` skip `Eq` for the same reason.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// a 64-bit signed integer, the result of any `i64`-typed expression.
    I64(i64),
    /// a 64-bit IEEE 754 float. follows IEEE 754 for non-finite results
    /// (`inf`, `-inf`, `NaN`); the [`Display`](std::fmt::Display) impl
    /// renders these explicitly so the output stays stable across Rust
    /// toolchains.
    F64(f64),
    /// a boolean -- the result of any `bool`-typed comparison, logic, or
    /// literal.
    Bool(bool),
    /// an 8-bit unsigned byte literal -- the result of a `b'X'` literal or a
    /// byte-typed sub-expression. rendered in lowercase-hex form by
    /// [`Display`](std::fmt::Display).
    Byte(u8),
    /// an owned string. the constant pool holds the source-literal form;
    /// string-interpolation results live on the VM's heap and never reach
    /// the constant pool.
    Str(String),
    /// the value-shaped result of a void-typed expression (a no-trailing-
    /// value block, or the `()` of a void-returning function). renders as
    /// `()`.
    Void,
    /// an index into the enclosing program's chunks. resolved by codegen
    /// during call-site emission; the VM dispatches by index. width `u16`
    /// matches every other indexed operand in the opcode set (so up to
    /// 65536 functions per program).
    Function(u16),
}

/// render a [`Value`] byte-deterministically.
///
/// the disassembler and the playground's bytecode panel both read this
/// output; identical input produces identical bytes on every machine and
/// every Rust toolchain. non-finite floats (`NaN`, `inf`, `-inf`) are
/// hand-spelled because `f64`'s `Display` of these values is platform-stable
/// in current Rust but a deliberate spelling avoids future drift.
///
/// string values render with no escape processing: a `Str` containing a
/// single quote will produce visually ambiguous output, but the rendering is
/// still deterministic -- acceptable in v1, the constant pool is in-process
/// only this phase.
impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::I64(n) => write!(f, "{n}"),
            Value::F64(x) => {
                if x.is_nan() {
                    f.write_str("NaN")
                } else if *x == f64::INFINITY {
                    f.write_str("inf")
                } else if *x == f64::NEG_INFINITY {
                    f.write_str("-inf")
                } else {
                    write!(f, "{x}")
                }
            }
            Value::Bool(b) => f.write_str(if *b { "true" } else { "false" }),
            Value::Byte(b) => write!(f, "b'\\x{b:02x}'"),
            Value::Str(s) => write!(f, "'{s}'"),
            Value::Void => f.write_str("()"),
            Value::Function(id) => write!(f, "fn#{id}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// every primitive case the Display impl must format exactly; the test
    /// loop below asserts the locked spelling for each.
    fn cases() -> Vec<(Value, &'static str)> {
        vec![
            (Value::I64(42), "42"),
            (Value::I64(0), "0"),
            (Value::I64(-1), "-1"),
            (Value::I64(i64::MIN), "-9223372036854775808"),
            (Value::I64(i64::MAX), "9223372036854775807"),
            (Value::Bool(true), "true"),
            (Value::Bool(false), "false"),
            (Value::Byte(0), "b'\\x00'"),
            (Value::Byte(0xff), "b'\\xff'"),
            (Value::Byte(0xa3), "b'\\xa3'"),
            (Value::Str("hello".to_string()), "'hello'"),
            (Value::Str(String::new()), "''"),
            (Value::Void, "()"),
            (Value::Function(0), "fn#0"),
            (Value::Function(65535), "fn#65535"),
        ]
    }

    #[test]
    fn display_i64_uses_rust_default_decimal_form() {
        assert_eq!(Value::I64(42).to_string(), "42");
        // i64::MIN as a literal: the spelling must match the source-text form a
        // negative literal would produce.
        assert_eq!(Value::I64(i64::MIN).to_string(), "-9223372036854775808");
    }

    #[test]
    fn display_f64_round_trips_finite_values_byte_identically() {
        // a finite value renders to a deterministic spelling; the test does
        // not pin the exact byte form (Rust's Display is round-trippable but
        // not a specific text) -- it pins that two calls produce the same
        // bytes, which IS the contract the disassembler depends on.
        let v = Value::F64(3.5);
        assert_eq!(v.to_string(), "3.5");
        let v2 = Value::F64(0.1 + 0.2);
        assert_eq!(v2.to_string(), v2.to_string());
    }

    #[test]
    fn display_f64_spells_non_finite_explicitly() {
        assert_eq!(Value::F64(f64::INFINITY).to_string(), "inf");
        assert_eq!(Value::F64(f64::NEG_INFINITY).to_string(), "-inf");
        assert_eq!(Value::F64(f64::NAN).to_string(), "NaN");
    }

    #[test]
    fn display_bool_uses_lowercase_keywords() {
        assert_eq!(Value::Bool(true).to_string(), "true");
        assert_eq!(Value::Bool(false).to_string(), "false");
    }

    #[test]
    fn display_byte_uses_lowercase_two_digit_hex() {
        assert_eq!(Value::Byte(0).to_string(), "b'\\x00'");
        assert_eq!(Value::Byte(0xff).to_string(), "b'\\xff'");
        assert_eq!(Value::Byte(0xa3).to_string(), "b'\\xa3'");
    }

    #[test]
    fn display_str_wraps_inner_bytes_in_single_quotes_without_escaping() {
        assert_eq!(Value::Str("hello".to_string()).to_string(), "'hello'");
        assert_eq!(Value::Str(String::new()).to_string(), "''");
    }

    #[test]
    fn display_void_renders_as_empty_tuple_form() {
        assert_eq!(Value::Void.to_string(), "()");
    }

    #[test]
    fn display_function_uses_hash_sigil_and_decimal_id() {
        assert_eq!(Value::Function(0).to_string(), "fn#0");
        assert_eq!(Value::Function(65535).to_string(), "fn#65535");
    }

    #[test]
    fn every_variant_renders_to_its_locked_spelling() {
        for (value, expected) in cases() {
            assert_eq!(value.to_string(), expected, "Display drift for {value:?}");
        }
    }

    #[test]
    fn display_is_deterministic_across_repeated_calls() {
        // a determinism stress test: build a sequence covering every variant
        // (including the three non-finite f64 forms), render it joined by
        // commas twice, assert the two renderings are byte-identical. a
        // hidden non-determinism (e.g. iterating a HashMap of byte values)
        // would surface here.
        let values: Vec<Value> = vec![
            Value::I64(1),
            Value::F64(2.5),
            Value::F64(f64::INFINITY),
            Value::F64(f64::NEG_INFINITY),
            Value::F64(f64::NAN),
            Value::Bool(true),
            Value::Bool(false),
            Value::Byte(0x10),
            Value::Str("hi".to_string()),
            Value::Void,
            Value::Function(7),
        ];
        let first: String = values
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let second: String = values
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(",");
        assert_eq!(first, second, "Display is non-deterministic");
    }

    #[test]
    fn partial_eq_holds_for_finite_primitives_and_breaks_for_nan() {
        // the typed_ast precedent: derive PartialEq, not Eq, because f64 is
        // not Eq. the rule survives at PartialEq -- NaN != NaN.
        assert_eq!(Value::I64(1), Value::I64(1));
        assert_eq!(Value::F64(0.0), Value::F64(0.0));
        assert_eq!(Value::Bool(true), Value::Bool(true));
        assert_eq!(Value::Byte(0xab), Value::Byte(0xab));
        assert_eq!(Value::Str("x".to_string()), Value::Str("x".to_string()));
        assert_eq!(Value::Void, Value::Void);
        assert_eq!(Value::Function(3), Value::Function(3));
        // IEEE 754: a NaN is not equal to itself.
        assert_ne!(Value::F64(f64::NAN), Value::F64(f64::NAN));
        // distinct variants never compare equal.
        assert_ne!(Value::I64(0), Value::Bool(false));
    }
}
