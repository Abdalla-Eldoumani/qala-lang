//! the type lattice the type checker resolves AST type expressions and
//! inferred local types into.
//!
//! one enum, [`QalaType`], covers every type the language can talk about: the
//! six primitives, fixed and dynamic arrays, tuples, function types, user
//! `Named` types (`struct` / `enum` / `interface`), the built-in `Result` and
//! `Option`, the opaque `FileHandle` returned by `open`, and the sentinel
//! `Unknown` poison type. equality is structural on primitives and compounds,
//! nominal on `Named` (via [`Symbol`]); `Unknown` is symmetrically equal to
//! anything under [`QalaType::types_match`], which is what suppresses
//! cascading errors after the first type failure.
//!
//! this file depends only on [`crate::ast::PrimType`] (to bridge from the
//! parser's primitive-type tokens to a [`QalaType`] variant). there are no
//! diagnostics, no symbol-table lookups, and no inference logic here -- those
//! belong in later modules. keeping the dependency surface small keeps the
//! type lattice reusable and easy to test.

use crate::ast::PrimType;

/// an interned name: the string that a [`QalaType::Named`] carries when it
/// stands for a user-declared struct, enum, or interface.
///
/// a thin newtype around [`String`] rather than a bare `String` -- it
/// documents intent at every call site (this is a *name*, not arbitrary text)
/// and prevents accidental mixing with display strings.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Symbol(pub String);

/// every type the type checker can talk about.
///
/// derive `Debug, Clone, PartialEq` only -- no `Eq`, because future variants
/// may carry float-valued metadata, and no `serde`, because the typed AST and
/// codegen run in-process and never serialize a type. structural equality on
/// primitives and compounds, nominal equality on [`Named`](Self::Named). use
/// [`QalaType::types_match`] (not bare `==`) for the type-equality predicate
/// the type checker actually consults -- it adds the [`Unknown`](Self::Unknown)
/// poison-as-wildcard rule that `==` does not.
#[derive(Debug, Clone, PartialEq)]
pub enum QalaType {
    /// the 64-bit signed integer primitive.
    I64,
    /// the 64-bit IEEE 754 float primitive.
    F64,
    /// the boolean primitive.
    Bool,
    /// the string primitive.
    Str,
    /// the byte primitive (one 8-bit unsigned value, distinct from `i64`).
    Byte,
    /// the empty type, used as the return type of a function that has no
    /// declared `-> T` and as the value type of statement-shaped expressions.
    Void,
    /// an array. `Some(n)` is the fixed-length `[T; n]` form; `None` is the
    /// dynamic `[T]` form. the element type is boxed because an array of
    /// arrays is legal and the box keeps [`QalaType`] a fixed size.
    Array(Box<QalaType>, Option<usize>),
    /// a tuple: an ordered list of element types. an empty tuple is the
    /// `void`-shaped tuple that the language does not write directly (a
    /// trailing-comma single-element tuple is a [`QalaType::Tuple`] of one).
    Tuple(Vec<QalaType>),
    /// a function type. `params` is the (un-named) parameter types in order;
    /// `returns` is boxed for the same fixed-size reason as `Array`.
    Function {
        /// the parameter types in declaration order.
        params: Vec<QalaType>,
        /// the return type.
        returns: Box<QalaType>,
    },
    /// a user-declared type referenced by name -- `struct` / `enum` /
    /// `interface`. nominal equality: two `Named` values are equal iff their
    /// [`Symbol`]s are equal.
    Named(Symbol),
    /// the built-in `Result<T, E>`. the type checker resolves a generic
    /// `Result<T, E>` type expression to this variant directly, without a
    /// user-defined generic mechanism.
    Result(Box<QalaType>, Box<QalaType>),
    /// the built-in `Option<T>`. the type checker resolves a generic
    /// `Option<T>` type expression to this variant directly.
    Option(Box<QalaType>),
    /// the opaque handle returned by the stdlib `open(path)` call. a built-in
    /// named type rather than a `Named(Symbol("FileHandle"))` so users do not
    /// need to declare it; the stdlib signature table refers to this variant
    /// by name. closing a `FileHandle` (`close(h)`) is checked structurally
    /// the same way as any other call.
    FileHandle,
    /// the poison type emitted on a type error so the rest of the program
    /// keeps typing. equal to anything under [`QalaType::types_match`] -- the
    /// research's Pattern 3 -- so a chain of expressions after the first error
    /// does not cascade into a wall of follow-on errors.
    Unknown,
}

impl QalaType {
    /// the type-equality predicate the type checker uses everywhere.
    ///
    /// structural on primitives, [`Array`](Self::Array), [`Tuple`](Self::Tuple),
    /// [`Function`](Self::Function), [`Result`](Self::Result), and
    /// [`Option`](Self::Option); nominal on [`Named`](Self::Named) (matches by
    /// [`Symbol`] equality); reflexive on [`FileHandle`](Self::FileHandle).
    /// [`Unknown`](Self::Unknown) is symmetrically equal to anything --
    /// `Unknown.types_match(x)` and `x.types_match(Unknown)` are both `true`
    /// for every `x`. that rule is what stops one type error from cascading
    /// into many; it is Pattern 3 in the phase research.
    ///
    /// note this is NOT the same as `==`. `==` is derived structural equality
    /// across the whole enum; it does not treat `Unknown` as a wildcard. always
    /// use `types_match` when asking "do these two types agree, for the
    /// purpose of type-checking?".
    pub fn types_match(&self, other: &QalaType) -> bool {
        use QalaType::*;
        match (self, other) {
            // poison is symmetrically equal to anything.
            (Unknown, _) | (_, Unknown) => true,
            // primitives compare by tag only.
            (I64, I64) | (F64, F64) | (Bool, Bool) | (Str, Str) | (Byte, Byte)
            | (Void, Void) | (FileHandle, FileHandle) => true,
            // arrays compare by element AND length kind (Some(n) only matches
            // the same Some(n); None only matches None).
            (Array(a, asize), Array(b, bsize)) => asize == bsize && a.types_match(b),
            // tuples compare element-wise, including length.
            (Tuple(a), Tuple(b)) => {
                a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| x.types_match(y))
            }
            // function types compare param count, param types pairwise, return.
            (
                Function { params: ap, returns: ar },
                Function { params: bp, returns: br },
            ) => {
                ap.len() == bp.len()
                    && ap.iter().zip(bp.iter()).all(|(x, y)| x.types_match(y))
                    && ar.types_match(br)
            }
            // Named is nominal -- match iff the carried symbols are equal.
            (Named(a), Named(b)) => a == b,
            // Result and Option compare their type arguments structurally.
            (Result(a_ok, a_err), Result(b_ok, b_err)) => {
                a_ok.types_match(b_ok) && a_err.types_match(b_err)
            }
            (Option(a), Option(b)) => a.types_match(b),
            // anything else is a mismatch.
            _ => false,
        }
    }

    /// the canonical lowercase form used in `expected X, found Y` wording.
    ///
    /// `i64`, `f64`, `bool`, `str`, `byte`, `void`, `[i64; 5]` (fixed array),
    /// `[i64]` (dynamic array), `(i64, bool)` (tuple), `fn(i64) -> i64`
    /// (function), `Shape` (named -- the symbol's text verbatim),
    /// `Result<i64, str>`, `Option<i64>`, `FileHandle`, and `?` for
    /// [`Unknown`](Self::Unknown). the question-mark form for `Unknown` keeps
    /// error messages short when poisoning propagates: "expected i64, found ?"
    /// reads better than "expected i64, found unknown".
    ///
    /// recursive: an `Array(Box<I64>, Some(5))` renders `"[i64; 5]"`; a
    /// `Function { params: vec![I64, Bool], returns: Box::new(Str) }`
    /// renders `"fn(i64, bool) -> str"`.
    pub fn display(&self) -> String {
        use QalaType::*;
        match self {
            I64 => "i64".to_string(),
            F64 => "f64".to_string(),
            Bool => "bool".to_string(),
            Str => "str".to_string(),
            Byte => "byte".to_string(),
            Void => "void".to_string(),
            Array(elem, Some(n)) => format!("[{}; {}]", elem.display(), n),
            Array(elem, None) => format!("[{}]", elem.display()),
            Tuple(elems) => {
                let inner: Vec<String> = elems.iter().map(|t| t.display()).collect();
                format!("({})", inner.join(", "))
            }
            Function { params, returns } => {
                let inner: Vec<String> = params.iter().map(|t| t.display()).collect();
                format!("fn({}) -> {}", inner.join(", "), returns.display())
            }
            Named(Symbol(name)) => name.clone(),
            Result(ok, err) => format!("Result<{}, {}>", ok.display(), err.display()),
            Option(inner) => format!("Option<{}>", inner.display()),
            FileHandle => "FileHandle".to_string(),
            Unknown => "?".to_string(),
        }
    }

    /// bridge from the parser's [`PrimType`] (which records *which keyword
    /// appeared*) to the corresponding [`QalaType`] primitive variant.
    ///
    /// the type checker's `TypeExpr` resolver calls this for the primitive
    /// case; the compound cases (`Array` / `DynArray` / `Tuple` / `Fn` /
    /// `Generic` / `Named`) recurse into the resolver instead.
    pub fn from_prim_type(p: &PrimType) -> Self {
        match p {
            PrimType::I64 => QalaType::I64,
            PrimType::F64 => QalaType::F64,
            PrimType::Bool => QalaType::Bool,
            PrimType::Str => QalaType::Str,
            PrimType::Byte => QalaType::Byte,
            PrimType::Void => QalaType::Void,
        }
    }
}

impl core::fmt::Display for QalaType {
    /// delegate to [`QalaType::display`] so a `format!("{ty}")` produces the
    /// canonical form. lets error messages interpolate types without an
    /// explicit `.display()` call at every site.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&self.display())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// a short helper so test rows stay readable.
    fn ty_match(a: &QalaType, b: &QalaType) -> bool {
        a.types_match(b)
    }

    /// boxed convenience so test rows do not repeat `Box::new` everywhere.
    fn b(t: QalaType) -> Box<QalaType> {
        Box::new(t)
    }

    #[test]
    fn every_primitive_equals_itself() {
        let prims = [
            QalaType::I64,
            QalaType::F64,
            QalaType::Bool,
            QalaType::Str,
            QalaType::Byte,
            QalaType::Void,
            QalaType::FileHandle,
        ];
        for p in &prims {
            assert!(ty_match(p, p), "{p:?} should match itself");
            assert_eq!(p.clone(), p.clone(), "{p:?} should be == itself");
        }
    }

    #[test]
    fn distinct_primitives_are_not_equal() {
        // a representative non-equal pair from every position.
        assert!(!ty_match(&QalaType::I64, &QalaType::F64));
        assert!(!ty_match(&QalaType::Bool, &QalaType::Str));
        assert!(!ty_match(&QalaType::Byte, &QalaType::I64));
        assert!(!ty_match(&QalaType::Void, &QalaType::Bool));
        assert!(!ty_match(&QalaType::FileHandle, &QalaType::Str));
    }

    #[test]
    fn arrays_compare_element_and_length_kind() {
        let fixed5 = QalaType::Array(b(QalaType::I64), Some(5));
        let fixed5_again = QalaType::Array(b(QalaType::I64), Some(5));
        let fixed6 = QalaType::Array(b(QalaType::I64), Some(6));
        let dynamic = QalaType::Array(b(QalaType::I64), None);
        assert!(ty_match(&fixed5, &fixed5_again));
        assert!(!ty_match(&fixed5, &fixed6));
        assert!(!ty_match(&fixed5, &dynamic));
        assert!(!ty_match(&dynamic, &fixed6));
        // a different element type does not match either length kind.
        let dynamic_bool = QalaType::Array(b(QalaType::Bool), None);
        assert!(!ty_match(&dynamic, &dynamic_bool));
    }

    #[test]
    fn tuples_compare_elementwise_with_order_significant() {
        let i64_bool = QalaType::Tuple(vec![QalaType::I64, QalaType::Bool]);
        let i64_bool_again = QalaType::Tuple(vec![QalaType::I64, QalaType::Bool]);
        let bool_i64 = QalaType::Tuple(vec![QalaType::Bool, QalaType::I64]);
        let single = QalaType::Tuple(vec![QalaType::I64]);
        assert!(ty_match(&i64_bool, &i64_bool_again));
        // order matters.
        assert!(!ty_match(&i64_bool, &bool_i64));
        // length matters.
        assert!(!ty_match(&i64_bool, &single));
    }

    #[test]
    fn function_types_compare_params_in_order_and_return() {
        let f_i64_to_i64 = QalaType::Function { params: vec![QalaType::I64], returns: b(QalaType::I64) };
        let f_i64_to_i64_again = QalaType::Function { params: vec![QalaType::I64], returns: b(QalaType::I64) };
        let f_str_to_i64 = QalaType::Function { params: vec![QalaType::Str], returns: b(QalaType::I64) };
        let f_i64_to_str = QalaType::Function { params: vec![QalaType::I64], returns: b(QalaType::Str) };
        let f_i64_i64_to_i64 = QalaType::Function {
            params: vec![QalaType::I64, QalaType::I64],
            returns: b(QalaType::I64),
        };
        assert!(ty_match(&f_i64_to_i64, &f_i64_to_i64_again));
        // param-type mismatch.
        assert!(!ty_match(&f_i64_to_i64, &f_str_to_i64));
        // return-type mismatch.
        assert!(!ty_match(&f_i64_to_i64, &f_i64_to_str));
        // param-count mismatch.
        assert!(!ty_match(&f_i64_to_i64, &f_i64_i64_to_i64));
    }

    #[test]
    fn named_types_compare_nominally_by_symbol() {
        let shape = QalaType::Named(Symbol("Shape".to_string()));
        let shape_again = QalaType::Named(Symbol("Shape".to_string()));
        let point = QalaType::Named(Symbol("Point".to_string()));
        assert!(ty_match(&shape, &shape_again));
        assert!(!ty_match(&shape, &point));
        // nominal equality means structurally-identical-but-named-differently
        // types are still distinct.
        assert!(!ty_match(&shape, &QalaType::Named(Symbol("shape".to_string()))));
    }

    #[test]
    fn result_and_option_compare_their_arguments_structurally() {
        let r_i64_str = QalaType::Result(b(QalaType::I64), b(QalaType::Str));
        let r_i64_str_again = QalaType::Result(b(QalaType::I64), b(QalaType::Str));
        let r_i64_bool = QalaType::Result(b(QalaType::I64), b(QalaType::Bool));
        let r_str_str = QalaType::Result(b(QalaType::Str), b(QalaType::Str));
        let o_i64 = QalaType::Option(b(QalaType::I64));
        assert!(ty_match(&r_i64_str, &r_i64_str_again));
        // error type differs.
        assert!(!ty_match(&r_i64_str, &r_i64_bool));
        // ok type differs.
        assert!(!ty_match(&r_i64_str, &r_str_str));
        // a Result is distinct from an Option even with matching ok type.
        assert!(!ty_match(&r_i64_str, &o_i64));
    }

    #[test]
    fn unknown_is_symmetrically_equal_to_anything() {
        // the poison-propagation rule from research Pattern 3.
        let samples = [
            QalaType::I64,
            QalaType::Str,
            QalaType::Array(b(QalaType::I64), None),
            QalaType::Named(Symbol("X".to_string())),
            QalaType::Result(b(QalaType::I64), b(QalaType::Str)),
            QalaType::Unknown,
        ];
        for s in &samples {
            // Unknown on the left matches anything.
            assert!(ty_match(&QalaType::Unknown, s), "Unknown should match {s:?}");
            // Unknown on the right matches anything.
            assert!(ty_match(s, &QalaType::Unknown), "{s:?} should match Unknown");
        }
        // Unknown matches Unknown -- explicit so a regression here is loud.
        assert!(ty_match(&QalaType::Unknown, &QalaType::Unknown));
    }

    #[test]
    fn types_match_is_symmetric_and_reflexive_on_concrete_pairs() {
        // a small sample table covering primitives, an array, a tuple, a
        // function, a named, a Result, and an Option. for every (a, b) the
        // predicate must give the same answer regardless of argument order;
        // for every (a, a) it must say "true".
        let samples: Vec<QalaType> = vec![
            QalaType::I64,
            QalaType::F64,
            QalaType::Bool,
            QalaType::Str,
            QalaType::Byte,
            QalaType::Void,
            QalaType::FileHandle,
            QalaType::Array(b(QalaType::I64), Some(3)),
            QalaType::Array(b(QalaType::I64), None),
            QalaType::Tuple(vec![QalaType::I64, QalaType::Bool]),
            QalaType::Function {
                params: vec![QalaType::I64],
                returns: b(QalaType::Bool),
            },
            QalaType::Named(Symbol("Shape".to_string())),
            QalaType::Result(b(QalaType::I64), b(QalaType::Str)),
            QalaType::Option(b(QalaType::I64)),
        ];
        for a in &samples {
            // reflexive.
            assert!(ty_match(a, a), "reflexive failure on {a:?}");
            for c in &samples {
                // symmetric.
                assert_eq!(
                    ty_match(a, c),
                    ty_match(c, a),
                    "symmetry failure on ({a:?}, {c:?})",
                );
            }
        }
    }

    #[test]
    fn display_produces_the_canonical_lowercase_form() {
        // the wording the TypeMismatch message and the renderer expect.
        assert_eq!(QalaType::I64.display(), "i64");
        assert_eq!(QalaType::F64.display(), "f64");
        assert_eq!(QalaType::Bool.display(), "bool");
        assert_eq!(QalaType::Str.display(), "str");
        assert_eq!(QalaType::Byte.display(), "byte");
        assert_eq!(QalaType::Void.display(), "void");
        assert_eq!(QalaType::FileHandle.display(), "FileHandle");
        assert_eq!(
            QalaType::Array(b(QalaType::I64), Some(5)).display(),
            "[i64; 5]",
        );
        assert_eq!(QalaType::Array(b(QalaType::I64), None).display(), "[i64]");
        assert_eq!(
            QalaType::Tuple(vec![QalaType::I64, QalaType::Bool]).display(),
            "(i64, bool)",
        );
        assert_eq!(
            QalaType::Function { params: vec![QalaType::I64], returns: b(QalaType::I64) }.display(),
            "fn(i64) -> i64",
        );
        assert_eq!(
            QalaType::Function {
                params: vec![QalaType::I64, QalaType::Bool],
                returns: b(QalaType::Str),
            }
            .display(),
            "fn(i64, bool) -> str",
        );
        assert_eq!(QalaType::Named(Symbol("Shape".to_string())).display(), "Shape");
        assert_eq!(
            QalaType::Result(b(QalaType::I64), b(QalaType::Str)).display(),
            "Result<i64, str>",
        );
        assert_eq!(QalaType::Option(b(QalaType::I64)).display(), "Option<i64>");
        assert_eq!(QalaType::Unknown.display(), "?");
        // the `Display` impl matches `.display()`.
        assert_eq!(format!("{}", QalaType::I64), "i64");
        assert_eq!(format!("{}", QalaType::Unknown), "?");
    }

    #[test]
    fn from_prim_type_bridges_every_prim_type_variant() {
        assert!(ty_match(&QalaType::from_prim_type(&PrimType::I64), &QalaType::I64));
        assert!(ty_match(&QalaType::from_prim_type(&PrimType::F64), &QalaType::F64));
        assert!(ty_match(&QalaType::from_prim_type(&PrimType::Bool), &QalaType::Bool));
        assert!(ty_match(&QalaType::from_prim_type(&PrimType::Str), &QalaType::Str));
        assert!(ty_match(&QalaType::from_prim_type(&PrimType::Byte), &QalaType::Byte));
        assert!(ty_match(&QalaType::from_prim_type(&PrimType::Void), &QalaType::Void));
    }
}
