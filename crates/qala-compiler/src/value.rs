//! the runtime value representation, in two types.
//!
//! [`ConstValue`] is the constant-pool entry -- what [`crate::chunk::Chunk`]'s
//! `constants` vector holds and what codegen builds when it folds a literal
//! expression. it is a plain tagged enum with seven variants in v1: the five
//! primitive Qala types ([`ConstValue::I64`], [`ConstValue::F64`],
//! [`ConstValue::Bool`], [`ConstValue::Byte`], [`ConstValue::Str`]), a unit
//! [`ConstValue::Void`], and [`ConstValue::Function`] -- a `u16` function
//! index. heap objects (arrays, structs, enum-variant payloads) are NOT
//! representable as a `ConstValue`; the VM builds those at runtime via the
//! `MAKE_*` opcodes. constant folding in codegen only folds primitive-typed
//! expressions (and string concatenations of two literal strings); a fold
//! that would produce a heap object falls through to runtime construction.
//!
//! [`Value`] is the NaN-boxed 8-byte runtime value: every slot on the VM's
//! value stack is a `Value`. an IEEE 754 `f64` is stored verbatim as its own
//! bits; every non-float kind ([`bool`], [`byte`](Value::byte), `void`, a
//! heap pointer) lives in the payload of a quiet NaN whose reserved high bits
//! select the kind. the codec is total and safe -- `f64::to_bits` /
//! `f64::from_bits` only, no `unsafe`, no `transmute`. there is no `i64`
//! encoding here: an `i64` value is a heap object and a `Value` holding one
//! is a pointer (the VM owns that heap; `value.rs` is only the bit codec).
//!
//! the [`Display`](std::fmt::Display) impl on `ConstValue` is byte-
//! deterministic: the same value renders to the same bytes every run. the
//! disassembler reads this directly and the playground's bytecode panel
//! reads the disassembler's output; non-determinism here would manifest as
//! visual flicker on every re-render, so the rule is locked in tests.
//!
//! no `serde` derives: this phase keeps both types in-process only. the WASM
//! bridge phase may add a derived form (or a hand-rolled serde proxy) when
//! disassembler output crosses the JS boundary; deferred to that phase.

/// a value the constant pool can store.
///
/// derives `Debug, Clone, PartialEq`. NOT `Eq` -- [`ConstValue::F64`] holds
/// `f64`, and `f64` is not `Eq` (`NaN != NaN`). this mirrors how the AST and
/// `QalaError` skip `Eq` for the same reason.
#[derive(Debug, Clone, PartialEq)]
pub enum ConstValue {
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

/// render a [`ConstValue`] byte-deterministically.
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
impl std::fmt::Display for ConstValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConstValue::I64(n) => write!(f, "{n}"),
            ConstValue::F64(x) => {
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
            ConstValue::Bool(b) => f.write_str(if *b { "true" } else { "false" }),
            ConstValue::Byte(b) => write!(f, "b'\\x{b:02x}'"),
            ConstValue::Str(s) => write!(f, "'{s}'"),
            ConstValue::Void => f.write_str("()"),
            ConstValue::Function(id) => write!(f, "fn#{id}"),
        }
    }
}

// ---- NaN-boxed runtime value -----------------------------------------------

/// the quiet-NaN base: all 11 exponent bits set, plus the top mantissa bit
/// (bit 51, the quiet bit) set. the FPU emits exactly this canonical NaN for
/// every NaN-producing operation, so a real computed NaN has these top bits
/// and no more -- it never collides with a tagged value below.
///
/// `allow(dead_code)`: this is the documented anchor the four `TAG_*`
/// prefixes are defined above; the codec never reads it directly (a real
/// NaN takes the untagged `f64` path via [`Value::is_tagged`]), but the
/// round-trip test asserts every tag prefix sits strictly above it. keeping
/// the named constant makes the bit layout self-documenting.
#[allow(dead_code)]
const QNAN: u64 = 0x7FF8_0000_0000_0000;

/// kind tag for a function value. the low data field carries the `u16`
/// function id directly -- a function is NOT a heap object, it is a tagged
/// scalar like a `bool` or a `byte`. this tag sits one step below `TAG_BOOL`
/// and one step above `QNAN`, so it is still a quiet-NaN pattern the FPU
/// never emits, and `is_tagged` covers it.
const TAG_FN: u64 = 0x7FFB_0000_0000_0000;

/// kind tag for a [`bool`] value. the low data field carries `0` or `1`.
const TAG_BOOL: u64 = 0x7FFC_0000_0000_0000;

/// kind tag for a `byte` value. the low data field carries the `u8`.
const TAG_BYTE: u64 = 0x7FFD_0000_0000_0000;

/// kind tag for the singleton `void` value. the data field is unused (0).
const TAG_VOID: u64 = 0x7FFE_0000_0000_0000;

/// kind tag for a heap pointer. the low 48-bit data field carries the heap
/// slot index (a `u32` slot fits with room to spare).
const TAG_PTR: u64 = 0x7FFF_0000_0000_0000;

/// masks off the top 16 bits -- the kind-select field. a tagged value's top
/// 16 bits land in `0x7FFB..=0x7FFF`; a real `f64` (including a computed NaN,
/// whose top bits are `0x7FF8`) does not.
const KIND_MASK: u64 = 0xFFFF_0000_0000_0000;

/// masks off the low 48 bits -- the data field a tagged value carries.
const DATA_MASK: u64 = 0x0000_FFFF_FFFF_FFFF;

/// the NaN-boxed 8-byte runtime value.
///
/// `Value` is a `u64` newtype. a real IEEE 754 `f64` is stored verbatim via
/// [`f64::to_bits`]; every non-float kind is a quiet NaN whose reserved top
/// 16 bits select the kind and whose low 48 bits carry the data. derives
/// `Clone, Copy, PartialEq` -- one machine word, cheaper to copy than to
/// reference. NOT `Eq`: two `Value`s wrapping `f64::NAN` are not equal, the
/// same IEEE 754 rule the `ConstValue` enum follows.
///
/// there is no `i64` kind. every `i64` value is a heap object and a `Value`
/// holding one is a [`Value::pointer`]; `value.rs` carries no integer
/// encoding. this keeps the codec a single clean match and every arithmetic
/// opcode a single path (the uniform-heap-box decision -- see the module
/// research notes).
#[derive(Clone, Copy, PartialEq)]
pub struct Value(u64);

impl Value {
    /// box an `f64`, storing its bits verbatim. a finite value, `inf`,
    /// `-inf`, `-0.0`, and a genuine `NaN` all round-trip: [`Value::as_f64`]
    /// returns `Some` for every one of them, because a computed NaN's tag
    /// bits (`0x7FF8`) are not in the reserved tagged range.
    pub fn from_f64(x: f64) -> Value {
        Value(x.to_bits())
    }

    /// box a `bool`. the data field is `0` for `false`, `1` for `true`.
    pub fn bool(b: bool) -> Value {
        Value(TAG_BOOL | b as u64)
    }

    /// box a `byte`. the data field carries the `u8` value.
    pub fn byte(b: u8) -> Value {
        Value(TAG_BYTE | b as u64)
    }

    /// the singleton `void` value -- the runtime shape of a void-typed
    /// expression's result. the data field is unused.
    pub fn void() -> Value {
        Value(TAG_VOID)
    }

    /// box a heap pointer. `slot` is the index of a [`crate::vm`] heap
    /// object; the 48-bit data field holds a `u32` slot with room to spare.
    pub fn pointer(slot: u32) -> Value {
        Value(TAG_PTR | slot as u64)
    }

    /// box a function value. `id` is the function's index into
    /// `Program.chunks` (a CALL operand). a function value is a tagged scalar,
    /// NOT a heap object: the `u16` id rides directly in the data field. the
    /// VM's CONST handler builds one of these for a `ConstValue::Function`,
    /// and the higher-order stdlib functions (`map` / `filter` / `reduce`)
    /// recover the id via [`Value::as_function`].
    pub fn function(id: u16) -> Value {
        Value(TAG_FN | id as u64)
    }

    /// the raw 64-bit pattern. exposed for the VM's `get_state` rendering and
    /// for tests that need to inspect the encoding directly.
    pub fn bits(self) -> u64 {
        self.0
    }

    /// `true` when this value is a tagged box (a function, `bool`, `byte`,
    /// `void`, or pointer), `false` when it is a real `f64`.
    ///
    /// the test masks the top 16 bits and range-checks `0x7FFB..=0x7FFF`. a
    /// genuine computed `NaN` has top bits `0x7FF8` and is therefore NOT
    /// tagged -- it correctly reads back as an `f64`. never use
    /// [`f64::is_nan`] for this test: a real NaN and a boxed value share the
    /// exponent and quiet-bit pattern, only the reserved tag bits separate
    /// them.
    pub fn is_tagged(self) -> bool {
        let top = self.0 & KIND_MASK;
        (TAG_FN..=TAG_PTR).contains(&top)
    }

    /// decode this value as an `f64`. returns `Some` for any value that is
    /// not a tagged box -- a finite float, `inf`, `-inf`, `-0.0`, and a
    /// genuine `NaN` all decode here. returns `None` for a `bool` / `byte` /
    /// `void` / pointer.
    pub fn as_f64(self) -> Option<f64> {
        if self.is_tagged() {
            None
        } else {
            Some(f64::from_bits(self.0))
        }
    }

    /// decode this value as a `bool`. returns `None` when the value is not a
    /// boxed `bool`.
    pub fn as_bool(self) -> Option<bool> {
        if self.0 & KIND_MASK == TAG_BOOL {
            Some(self.0 & DATA_MASK != 0)
        } else {
            None
        }
    }

    /// decode this value as a `byte`. returns `None` when the value is not a
    /// boxed `byte`. the data field is masked to 8 bits so a stray high bit
    /// cannot widen the result.
    pub fn as_byte(self) -> Option<u8> {
        if self.0 & KIND_MASK == TAG_BYTE {
            Some((self.0 & 0xFF) as u8)
        } else {
            None
        }
    }

    /// `true` when this value is the singleton `void`. returns a `bool`
    /// rather than `Option<()>` -- `void` carries no data, so the only
    /// question is whether the value is `void` at all.
    pub fn as_void(self) -> bool {
        self.0 == TAG_VOID
    }

    /// decode this value as a heap pointer slot index. returns `None` when
    /// the value is not a boxed pointer. the data field is masked to 32 bits
    /// so the result is exactly the `u32` slot [`Value::pointer`] stored.
    pub fn as_pointer(self) -> Option<u32> {
        if self.0 & KIND_MASK == TAG_PTR {
            Some((self.0 & 0xFFFF_FFFF) as u32)
        } else {
            None
        }
    }

    /// decode this value as a function id. returns `None` when the value is
    /// not a boxed function. the data field is masked to 16 bits so the
    /// result is exactly the `u16` id [`Value::function`] stored.
    pub fn as_function(self) -> Option<u16> {
        if self.0 & KIND_MASK == TAG_FN {
            Some((self.0 & 0xFFFF) as u16)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// every primitive case the Display impl must format exactly; the test
    /// loop below asserts the locked spelling for each.
    fn cases() -> Vec<(ConstValue, &'static str)> {
        vec![
            (ConstValue::I64(42), "42"),
            (ConstValue::I64(0), "0"),
            (ConstValue::I64(-1), "-1"),
            (ConstValue::I64(i64::MIN), "-9223372036854775808"),
            (ConstValue::I64(i64::MAX), "9223372036854775807"),
            (ConstValue::Bool(true), "true"),
            (ConstValue::Bool(false), "false"),
            (ConstValue::Byte(0), "b'\\x00'"),
            (ConstValue::Byte(0xff), "b'\\xff'"),
            (ConstValue::Byte(0xa3), "b'\\xa3'"),
            (ConstValue::Str("hello".to_string()), "'hello'"),
            (ConstValue::Str(String::new()), "''"),
            (ConstValue::Void, "()"),
            (ConstValue::Function(0), "fn#0"),
            (ConstValue::Function(65535), "fn#65535"),
        ]
    }

    #[test]
    fn display_i64_uses_rust_default_decimal_form() {
        assert_eq!(ConstValue::I64(42).to_string(), "42");
        // i64::MIN as a literal: the spelling must match the source-text form a
        // negative literal would produce.
        assert_eq!(
            ConstValue::I64(i64::MIN).to_string(),
            "-9223372036854775808"
        );
    }

    #[test]
    fn display_f64_round_trips_finite_values_byte_identically() {
        // a finite value renders to a deterministic spelling; the test does
        // not pin the exact byte form (Rust's Display is round-trippable but
        // not a specific text) -- it pins that two calls produce the same
        // bytes, which IS the contract the disassembler depends on.
        let v = ConstValue::F64(3.5);
        assert_eq!(v.to_string(), "3.5");
        let v2 = ConstValue::F64(0.1 + 0.2);
        assert_eq!(v2.to_string(), v2.to_string());
    }

    #[test]
    fn display_f64_spells_non_finite_explicitly() {
        assert_eq!(ConstValue::F64(f64::INFINITY).to_string(), "inf");
        assert_eq!(ConstValue::F64(f64::NEG_INFINITY).to_string(), "-inf");
        assert_eq!(ConstValue::F64(f64::NAN).to_string(), "NaN");
    }

    #[test]
    fn display_bool_uses_lowercase_keywords() {
        assert_eq!(ConstValue::Bool(true).to_string(), "true");
        assert_eq!(ConstValue::Bool(false).to_string(), "false");
    }

    #[test]
    fn display_byte_uses_lowercase_two_digit_hex() {
        assert_eq!(ConstValue::Byte(0).to_string(), "b'\\x00'");
        assert_eq!(ConstValue::Byte(0xff).to_string(), "b'\\xff'");
        assert_eq!(ConstValue::Byte(0xa3).to_string(), "b'\\xa3'");
    }

    #[test]
    fn display_str_wraps_inner_bytes_in_single_quotes_without_escaping() {
        assert_eq!(ConstValue::Str("hello".to_string()).to_string(), "'hello'");
        assert_eq!(ConstValue::Str(String::new()).to_string(), "''");
    }

    #[test]
    fn display_void_renders_as_empty_tuple_form() {
        assert_eq!(ConstValue::Void.to_string(), "()");
    }

    #[test]
    fn display_function_uses_hash_sigil_and_decimal_id() {
        assert_eq!(ConstValue::Function(0).to_string(), "fn#0");
        assert_eq!(ConstValue::Function(65535).to_string(), "fn#65535");
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
        let values: Vec<ConstValue> = vec![
            ConstValue::I64(1),
            ConstValue::F64(2.5),
            ConstValue::F64(f64::INFINITY),
            ConstValue::F64(f64::NEG_INFINITY),
            ConstValue::F64(f64::NAN),
            ConstValue::Bool(true),
            ConstValue::Bool(false),
            ConstValue::Byte(0x10),
            ConstValue::Str("hi".to_string()),
            ConstValue::Void,
            ConstValue::Function(7),
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
        assert_eq!(ConstValue::I64(1), ConstValue::I64(1));
        assert_eq!(ConstValue::F64(0.0), ConstValue::F64(0.0));
        assert_eq!(ConstValue::Bool(true), ConstValue::Bool(true));
        assert_eq!(ConstValue::Byte(0xab), ConstValue::Byte(0xab));
        assert_eq!(
            ConstValue::Str("x".to_string()),
            ConstValue::Str("x".to_string())
        );
        assert_eq!(ConstValue::Void, ConstValue::Void);
        assert_eq!(ConstValue::Function(3), ConstValue::Function(3));
        // IEEE 754: a NaN is not equal to itself.
        assert_ne!(ConstValue::F64(f64::NAN), ConstValue::F64(f64::NAN));
        // distinct variants never compare equal.
        assert_ne!(ConstValue::I64(0), ConstValue::Bool(false));
    }

    // ---- NaN-boxed Value round-trip battery --------------------------------

    #[test]
    fn value_is_exactly_eight_bytes() {
        // the headline NaN-boxing contract: every stack slot is one machine
        // word. a regression here (e.g. a stray enum tag) is caught now.
        assert_eq!(std::mem::size_of::<Value>(), 8);
    }

    #[test]
    fn from_f64_round_trips_a_finite_value() {
        let v = Value::from_f64(3.5);
        assert!(!v.is_tagged(), "a finite f64 must not be tagged");
        assert_eq!(v.as_f64(), Some(3.5));
    }

    #[test]
    fn from_f64_round_trips_positive_and_negative_infinity() {
        let pos = Value::from_f64(f64::INFINITY);
        assert!(!pos.is_tagged());
        assert_eq!(pos.as_f64(), Some(f64::INFINITY));

        let neg = Value::from_f64(f64::NEG_INFINITY);
        assert!(!neg.is_tagged());
        assert_eq!(neg.as_f64(), Some(f64::NEG_INFINITY));
    }

    #[test]
    fn from_f64_round_trips_a_genuine_nan_as_a_real_float() {
        // the canonical-NaN-collision guard: a computed NaN must decode back
        // as an f64, NOT as a tagged value. is_tagged uses the tag-mask range
        // check, so f64::NAN's 0x7FF8 prefix falls through to the float path.
        let v = Value::from_f64(f64::NAN);
        assert!(!v.is_tagged(), "a real NaN must not read as tagged");
        let decoded = v.as_f64().expect("a NaN must decode as an f64");
        assert!(decoded.is_nan(), "the decoded value must still be a NaN");
        // and it does not decode as any tagged kind.
        assert_eq!(v.as_bool(), None);
        assert_eq!(v.as_byte(), None);
        assert!(!v.as_void());
        assert_eq!(v.as_pointer(), None);
        assert_eq!(v.as_function(), None);
    }

    #[test]
    fn from_f64_preserves_the_sign_bit_of_negative_zero() {
        // -0.0 and 0.0 are bit-distinct; the codec stores raw bits, so the
        // sign survives and -0.0 stays an f64 (it is not tagged).
        let v = Value::from_f64(-0.0);
        assert!(!v.is_tagged());
        let decoded = v.as_f64().expect("-0.0 decodes as an f64");
        assert!(
            decoded == 0.0 && decoded.is_sign_negative(),
            "sign bit lost"
        );
    }

    #[test]
    fn bool_round_trips_true_and_false() {
        let t = Value::bool(true);
        assert!(t.is_tagged());
        assert_eq!(t.as_bool(), Some(true));
        // a bool is not any other kind.
        assert_eq!(t.as_f64(), None);
        assert_eq!(t.as_byte(), None);

        let f = Value::bool(false);
        assert!(f.is_tagged());
        assert_eq!(f.as_bool(), Some(false));
    }

    #[test]
    fn byte_round_trips_low_high_and_mid_values() {
        for b in [0x00u8, 0xFF, 0xA3] {
            let v = Value::byte(b);
            assert!(v.is_tagged(), "a byte must be tagged");
            assert_eq!(v.as_byte(), Some(b), "byte round-trip failed for {b:#x}");
            // a byte is not a bool or a float.
            assert_eq!(v.as_bool(), None);
            assert_eq!(v.as_f64(), None);
        }
    }

    #[test]
    fn void_is_a_tagged_singleton() {
        let v = Value::void();
        assert!(v.is_tagged());
        assert!(v.as_void(), "void must read as void");
        // void carries no data and is no other kind.
        assert_eq!(v.as_bool(), None);
        assert_eq!(v.as_byte(), None);
        assert_eq!(v.as_f64(), None);
        assert_eq!(v.as_pointer(), None);
        // two voids are equal -- the singleton property. asserted with `==`
        // rather than assert_eq! because `Value` carries no `Debug` derive.
        assert!(Value::void() == Value::void(), "void is not a singleton");
    }

    #[test]
    fn pointer_round_trips_slot_zero_and_slot_u32_max() {
        for slot in [0u32, 1, 0xABCD, u32::MAX] {
            let v = Value::pointer(slot);
            assert!(v.is_tagged(), "a pointer must be tagged");
            assert_eq!(
                v.as_pointer(),
                Some(slot),
                "pointer round-trip failed for slot {slot}"
            );
            // a pointer is not a primitive kind.
            assert!(!v.as_void());
            assert_eq!(v.as_bool(), None);
        }
    }

    #[test]
    fn is_tagged_is_false_for_every_float_and_true_for_every_box() {
        // the discriminator the VM relies on. every float case -- including
        // the non-finite ones -- must read as not-tagged; every boxed case
        // must read as tagged.
        let floats = [
            Value::from_f64(0.0),
            Value::from_f64(-0.0),
            Value::from_f64(1.0),
            Value::from_f64(-7.25),
            Value::from_f64(f64::INFINITY),
            Value::from_f64(f64::NEG_INFINITY),
            Value::from_f64(f64::NAN),
            Value::from_f64(f64::MIN),
            Value::from_f64(f64::MAX),
        ];
        for f in floats {
            assert!(!f.is_tagged(), "float {:#018x} read as tagged", f.bits());
        }
        let boxes = [
            Value::function(0),
            Value::function(u16::MAX),
            Value::bool(true),
            Value::bool(false),
            Value::byte(0),
            Value::byte(0xFF),
            Value::void(),
            Value::pointer(0),
            Value::pointer(u32::MAX),
        ];
        for b in boxes {
            assert!(b.is_tagged(), "box {:#018x} read as not-tagged", b.bits());
        }
    }

    #[test]
    fn the_five_tag_prefixes_are_distinct_and_none_equals_the_bare_qnan() {
        // a computed NaN never decodes as a tagged value: the five reserved
        // tag prefixes are pairwise distinct and all sit strictly above the
        // canonical-NaN prefix QNAN (0x7FF8).
        // the canonical NaN's top 16 bits are below the tagged range. this
        // claim does not depend on the loop variable, so it is a single
        // compile-time check outside the loop.
        const _: () = assert!(
            (QNAN & KIND_MASK) < TAG_FN,
            "QNAN prefix is inside the tagged range"
        );

        let tags = [TAG_FN, TAG_BOOL, TAG_BYTE, TAG_VOID, TAG_PTR];
        for (i, a) in tags.iter().enumerate() {
            for b in &tags[i + 1..] {
                assert_ne!(a, b, "two tag prefixes collide");
            }
            assert!(*a > QNAN, "tag prefix {a:#018x} not above QNAN");
        }
    }

    #[test]
    fn function_round_trips_low_high_and_mid_ids() {
        // a function value is a tagged scalar carrying the u16 fn-id directly;
        // no heap object. every id round-trips through function/as_function.
        for id in [0u16, 1, 0x1234, u16::MAX] {
            let v = Value::function(id);
            assert!(v.is_tagged(), "a function must be tagged");
            assert_eq!(
                v.as_function(),
                Some(id),
                "function round-trip failed for id {id}"
            );
            // a function is no other kind.
            assert_eq!(v.as_bool(), None);
            assert_eq!(v.as_byte(), None);
            assert_eq!(v.as_f64(), None);
            assert_eq!(v.as_pointer(), None);
            assert!(!v.as_void());
        }
    }

    #[test]
    fn a_function_is_not_a_pointer_and_a_pointer_is_not_a_function() {
        // the two tags must not alias: a heap pointer at slot 5 and a function
        // with id 5 are distinct values that decode to distinct kinds.
        let ptr = Value::pointer(5);
        let func = Value::function(5);
        assert_ne!(ptr.bits(), func.bits(), "pointer and function tags alias");
        assert_eq!(ptr.as_pointer(), Some(5));
        assert_eq!(ptr.as_function(), None);
        assert_eq!(func.as_function(), Some(5));
        assert_eq!(func.as_pointer(), None);
    }
}
