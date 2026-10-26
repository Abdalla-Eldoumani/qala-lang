//! the typed AST: what the type checker produces and codegen consumes.
//!
//! mirrors `ast.rs` node-by-node and adds a resolved [`QalaType`] annotation on
//! every expression node and an [`EffectSet`] annotation on every function
//! node. nothing is desugared -- [`TypedExpr::Pipeline`], [`TypedExpr::Match`],
//! [`TypedExpr::OrElse`], and [`TypedExpr::Interpolation`] stay faithful, same
//! rule as the untyped AST; lowering happens in codegen.
//!
//! the type checker takes `&ast::Ast` and returns a [`TypedAst`] by value plus
//! error and warning vectors; no interior mutability, no side tables, no
//! shared references to the untyped tree. the typed tree is built once and
//! never mutated; codegen reads it through the uniform [`TypedExpr::ty`] /
//! [`TypedExpr::span`] / [`TypedFnDecl::effect`] accessors.
//!
//! `Pattern`, `UnaryOp`, `BinOp` are re-used from `ast.rs` unchanged: in v1
//! patterns carry no resolved type (the scrutinee's [`TypedExpr::ty`] is the
//! relevant fact during exhaustiveness checking), and operators carry no type
//! information either -- the typed nodes wrapping them carry the type.
//!
//! the node enums derive `Debug, Clone, PartialEq` and no more: no `Eq`,
//! because float literals reach the typed AST and `f64` is not `Eq`; no
//! `serde`, because the typed AST is in-process data the renderer never sees.

use crate::ast::{BinOp, Pattern, UnaryOp};
use crate::effects::EffectSet;
use crate::span::Span;
use crate::types::QalaType;

/// a whole typed program: the top-level typed items in source order.
///
/// the mirror of `ast::Ast`. a type alias rather than a wrapper struct, for the
/// same reason: a program's span is just the source's span, which the caller
/// already holds.
pub type TypedAst = Vec<TypedItem>;

// ---- items -----------------------------------------------------------------

/// a typed top-level declaration: a function, a struct, an enum, or an
/// interface. the mirror of `ast::Item`. `fn Type.method` definitions are a
/// [`TypedItem::Fn`] whose [`TypedFnDecl`] carries an optional `type_name`.
#[derive(Debug, Clone, PartialEq)]
pub enum TypedItem {
    /// a typed `fn` declaration -- a free function, or a `fn Type.method`
    /// method definition when `type_name` is set.
    Fn(TypedFnDecl),
    /// a typed `struct` declaration with resolved field types.
    Struct(TypedStructDecl),
    /// a typed `enum` declaration with resolved variant field types.
    Enum(TypedEnumDecl),
    /// a typed `interface` declaration -- method signatures with resolved
    /// parameter and return types and inferred effect sets.
    Interface(TypedInterfaceDecl),
}

impl TypedItem {
    /// the source span of this item, opening keyword to closing brace.
    ///
    /// each item kind wraps a typed decl that carries the span; this delegates
    /// to it. exhaustive over every kind, so a new item kind forces an arm here.
    pub fn span(&self) -> Span {
        match self {
            TypedItem::Fn(d) => d.span,
            TypedItem::Struct(d) => d.span,
            TypedItem::Enum(d) => d.span,
            TypedItem::Interface(d) => d.span,
        }
    }
}

/// a typed `fn` declaration. mirror of `ast::FnDecl` with the parser's
/// [`Option<TypeExpr>`](crate::ast::TypeExpr) return type resolved to a
/// [`QalaType`] (`void` resolves to [`QalaType::Void`]) and the parser's
/// optional `is X` annotation replaced by an inferred [`EffectSet`] that is
/// always present.
#[derive(Debug, Clone, PartialEq)]
pub struct TypedFnDecl {
    /// the receiver type's name for a `fn Type.method` definition, else `None`.
    pub type_name: Option<String>,
    /// the function (or method) name.
    pub name: String,
    /// the typed parameter list, in order; for a method, the first may be
    /// `self`.
    pub params: Vec<TypedParam>,
    /// the resolved return type. `void` is the resolved-form of an omitted
    /// `-> T`.
    pub ret_ty: QalaType,
    /// the inferred effect set (or the annotated set if `is X` was present and
    /// was a superset of the inferred). always present in the typed AST, even
    /// when the source had no `is X` annotation.
    pub effect: EffectSet,
    /// the function body.
    pub body: TypedBlock,
    /// `fn` keyword to closing `}`.
    pub span: Span,
}

impl TypedFnDecl {
    /// the inferred effect set of this function.
    ///
    /// [`EffectSet`] is `Copy`, so returning by value is the same cost as a
    /// reference and reads cleaner at call sites. this is the public accessor
    /// the diagnostics renderer consults when building the
    /// `EffectViolation` message's caller-effect slot.
    pub fn effect(&self) -> EffectSet {
        self.effect
    }
}

/// one typed parameter of a function or method. mirror of `ast::Param`. for the
/// `self` first parameter of a method the type is the receiver's type
/// (resolved by the type checker via the [`TypedFnDecl::type_name`]).
#[derive(Debug, Clone, PartialEq)]
pub struct TypedParam {
    /// true if this is the `self` first parameter of a method.
    pub is_self: bool,
    /// the parameter name (`"self"` when `is_self`).
    pub name: String,
    /// the resolved parameter type. for `self`, the receiver type
    /// ([`QalaType::Named`] of the enclosing method's `type_name`).
    pub ty: QalaType,
    /// the `= expr` default value, type-checked against `ty`, or `None`.
    pub default: Option<TypedExpr>,
    /// the parameter's source span (name to default, or name to type).
    pub span: Span,
}

/// a typed `struct` declaration. mirror of `ast::StructDecl` with field types
/// resolved.
#[derive(Debug, Clone, PartialEq)]
pub struct TypedStructDecl {
    /// the struct's name.
    pub name: String,
    /// the typed fields, in declaration order.
    pub fields: Vec<TypedField>,
    /// `struct` keyword to closing `}`.
    pub span: Span,
}

/// one typed field of a struct: a name and a resolved type.
#[derive(Debug, Clone, PartialEq)]
pub struct TypedField {
    /// the field name.
    pub name: String,
    /// the field's resolved type.
    pub ty: QalaType,
    /// the field's source span (name to type).
    pub span: Span,
}

/// a typed `enum` declaration. mirror of `ast::EnumDecl` with each variant's
/// carried field types resolved.
#[derive(Debug, Clone, PartialEq)]
pub struct TypedEnumDecl {
    /// the enum's name.
    pub name: String,
    /// the typed variants, in declaration order.
    pub variants: Vec<TypedVariant>,
    /// `enum` keyword to closing `}`.
    pub span: Span,
}

/// one typed variant of an enum: a name and zero or more resolved field types.
#[derive(Debug, Clone, PartialEq)]
pub struct TypedVariant {
    /// the variant name.
    pub name: String,
    /// the resolved types of the variant's data fields (possibly empty).
    pub fields: Vec<QalaType>,
    /// the variant's source span.
    pub span: Span,
}

/// a typed `interface` declaration. mirror of `ast::InterfaceDecl` with method
/// signatures' parameter and return types resolved and effect sets present.
#[derive(Debug, Clone, PartialEq)]
pub struct TypedInterfaceDecl {
    /// the interface's name.
    pub name: String,
    /// the typed method signatures the interface requires.
    pub methods: Vec<TypedMethodSig>,
    /// `interface` keyword to closing `}`.
    pub span: Span,
}

/// one typed method signature in an interface: like a [`TypedFnDecl`] without a
/// body and without a `type_name`. the first parameter is typically `self`.
#[derive(Debug, Clone, PartialEq)]
pub struct TypedMethodSig {
    /// the method name.
    pub name: String,
    /// the typed parameter list; the first is typically `self`.
    pub params: Vec<TypedParam>,
    /// the resolved return type. `void` is the resolved-form of an omitted
    /// `-> T`.
    pub ret_ty: QalaType,
    /// the effect set required of any conforming implementation. (an interface
    /// method's `is X` annotation, or the inferred set when conformance is
    /// being checked.)
    pub effect: EffectSet,
    /// the signature's source span.
    pub span: Span,
}

// ---- statements and blocks -------------------------------------------------

/// a typed `{ ... }` block. mirror of `ast::Block`, plus the resolved type of
/// the block's value: `ty` is the type of the trailing value expression, or
/// [`QalaType::Void`] if there is no trailing value (the block ended in a `;`
/// or was empty).
#[derive(Debug, Clone, PartialEq)]
pub struct TypedBlock {
    /// the typed statements, in order.
    pub stmts: Vec<TypedStmt>,
    /// the trailing expression (no terminating `;`) that is the block's value,
    /// or `None` for a `void` block.
    pub value: Option<Box<TypedExpr>>,
    /// the type of the trailing value expression, or [`QalaType::Void`] for an
    /// empty / semicolon-ended block. the typechecker fills this; codegen reads
    /// it to decide whether the block leaves a value on the stack.
    pub ty: QalaType,
    /// `{` to `}`.
    pub span: Span,
}

/// a typed statement. mirror of `ast::Stmt`: `if`/`else`, `while`, `for`,
/// `return`, `break`, `continue`, `defer`, and expression-statements all stay
/// statements -- there is no `TypedExpr::If` in v1.
#[derive(Debug, Clone, PartialEq)]
pub enum TypedStmt {
    /// `let name = init` or `let mut name = init`. `ty` is the resolved type
    /// of the binding (inferred from `init` or checked against an explicit
    /// annotation). `is_mut` is carried forward from the untyped AST so
    /// codegen does not have to re-read the original tree.
    Let {
        is_mut: bool,
        name: String,
        ty: QalaType,
        init: TypedExpr,
        span: Span,
    },
    /// `if cond { ... }` with an optional `else` (a block, or another `if` for
    /// an `else if` chain). a statement, not an expression.
    If {
        cond: TypedExpr,
        then_block: TypedBlock,
        else_branch: Option<TypedElseBranch>,
        span: Span,
    },
    /// `while cond { ... }`.
    While {
        cond: TypedExpr,
        body: TypedBlock,
        span: Span,
    },
    /// `for var in iter { ... }`. `var_ty` is the resolved type of the loop
    /// variable (the element type of the iterable). kept as a `For` node, not
    /// lowered.
    For {
        var: String,
        var_ty: QalaType,
        iter: TypedExpr,
        body: TypedBlock,
        span: Span,
    },
    /// `return` or `return expr`. the value is optional -- a bare `return` in
    /// a `void` function is legal.
    Return {
        value: Option<TypedExpr>,
        span: Span,
    },
    /// `break`. no labels, no value in v1.
    Break { span: Span },
    /// `continue`. no labels, no value in v1.
    Continue { span: Span },
    /// `defer expr` -- `expr` runs at scope exit, LIFO with other defers in the
    /// same scope.
    Defer { expr: TypedExpr, span: Span },
    /// an expression used as a statement (`expr ;`). its value is discarded.
    Expr { expr: TypedExpr, span: Span },
}

impl TypedStmt {
    /// the source span of this statement.
    ///
    /// every variant carries its `span` field; this match is exhaustive, which
    /// is the proof every typed statement has one.
    pub fn span(&self) -> Span {
        match self {
            TypedStmt::Let { span, .. }
            | TypedStmt::If { span, .. }
            | TypedStmt::While { span, .. }
            | TypedStmt::For { span, .. }
            | TypedStmt::Return { span, .. }
            | TypedStmt::Break { span }
            | TypedStmt::Continue { span }
            | TypedStmt::Defer { span, .. }
            | TypedStmt::Expr { span, .. } => *span,
        }
    }
}

/// the `else` part of a [`TypedStmt::If`]: either a final `{ ... }` block, or
/// another `if` (the boxed [`TypedStmt`] is always a [`TypedStmt::If`]) for an
/// `else if` chain. mirror of `ast::ElseBranch`.
#[derive(Debug, Clone, PartialEq)]
pub enum TypedElseBranch {
    /// `else { ... }`.
    Block(TypedBlock),
    /// `else if ...` -- the boxed statement is a [`TypedStmt::If`].
    If(Box<TypedStmt>),
}

// ---- expressions -----------------------------------------------------------

/// a typed expression. mirror of `ast::Expr` with a resolved [`QalaType`]
/// annotation on every variant.
///
/// nothing is desugared: [`TypedExpr::Pipeline`] stays a pipeline (not
/// `f(x, a)`), [`TypedExpr::Interpolation`] stays a parts list (not a `+`
/// chain), [`TypedExpr::Match`] and [`TypedExpr::Block`] produce values
/// directly. the rule is the same as for the untyped AST -- the typechecker
/// only types; codegen lowers.
///
/// every recursive position is a [`Box<TypedExpr>`]; the variant-level `ty`
/// field is the design choice over a wrapper struct (`TypedExpr { kind, span,
/// ty }`) -- per-variant means [`TypedExpr::ty`] is one exhaustive match that
/// the compiler enforces.
#[derive(Debug, Clone, PartialEq)]
pub enum TypedExpr {
    /// an integer literal with its resolved type ([`QalaType::I64`]).
    Int {
        value: i64,
        ty: QalaType,
        span: Span,
    },
    /// a float literal with its resolved type ([`QalaType::F64`]).
    Float {
        value: f64,
        ty: QalaType,
        span: Span,
    },
    /// a byte literal with its resolved type ([`QalaType::Byte`]).
    Byte { value: u8, ty: QalaType, span: Span },
    /// a string literal with its resolved type ([`QalaType::Str`]).
    Str {
        value: String,
        ty: QalaType,
        span: Span,
    },
    /// a boolean literal with its resolved type ([`QalaType::Bool`]).
    Bool {
        value: bool,
        ty: QalaType,
        span: Span,
    },
    /// an identifier reference resolved to its bound type.
    Ident {
        name: String,
        ty: QalaType,
        span: Span,
    },
    /// `( inner )` -- a parenthesized expression. semantically transparent; the
    /// resolved type is the inner's type.
    Paren {
        inner: Box<TypedExpr>,
        ty: QalaType,
        span: Span,
    },
    /// `( e1, e2, ... )` -- a tuple. resolved type is
    /// [`QalaType::Tuple`] of the element types.
    Tuple {
        elems: Vec<TypedExpr>,
        ty: QalaType,
        span: Span,
    },
    /// `[ e1, e2, ... ]` -- an array literal. resolved type is
    /// [`QalaType::Array`] with `Some(n)` length matching `elems.len()`.
    ArrayLit {
        elems: Vec<TypedExpr>,
        ty: QalaType,
        span: Span,
    },
    /// `[ value; count ]` -- the repeat form of an array literal.
    ArrayRepeat {
        value: Box<TypedExpr>,
        count: Box<TypedExpr>,
        ty: QalaType,
        span: Span,
    },
    /// `Name { field: e, ... }` -- a struct literal. resolved type is
    /// [`QalaType::Named`] of the struct.
    StructLit {
        name: String,
        fields: Vec<TypedFieldInit>,
        ty: QalaType,
        span: Span,
    },
    /// `obj.name` -- field access. distinct from [`TypedExpr::MethodCall`]:
    /// this is the form with no `( ... )` after the `.name`. resolved type is
    /// the field's type.
    FieldAccess {
        obj: Box<TypedExpr>,
        name: String,
        ty: QalaType,
        span: Span,
    },
    /// `receiver.name(args)` -- a method call. distinct from a field access
    /// followed by a call; the typechecker resolves it to the
    /// `fn Type.name(self, ...)` declaration. resolved type is the method's
    /// return type.
    MethodCall {
        receiver: Box<TypedExpr>,
        name: String,
        args: Vec<TypedExpr>,
        ty: QalaType,
        span: Span,
    },
    /// `callee(args)` -- a call expression. resolved type is the callee's
    /// return type.
    Call {
        callee: Box<TypedExpr>,
        args: Vec<TypedExpr>,
        ty: QalaType,
        span: Span,
    },
    /// `obj[index]` -- an index expression. resolved type is the element type
    /// of the array.
    Index {
        obj: Box<TypedExpr>,
        index: Box<TypedExpr>,
        ty: QalaType,
        span: Span,
    },
    /// `expr?` -- error propagation. resolved type is the `Ok` payload (for
    /// `Result<T, E>`) or the `Some` payload (for `Option<T>`).
    Try {
        expr: Box<TypedExpr>,
        ty: QalaType,
        span: Span,
    },
    /// a unary-operator application: `!operand` or `-operand`. resolved type
    /// is determined by the operand and the operator.
    Unary {
        op: UnaryOp,
        operand: Box<TypedExpr>,
        ty: QalaType,
        span: Span,
    },
    /// a binary-operator application. resolved type is determined by the
    /// operator (arithmetic: matches the operand type; comparison and equality:
    /// `bool`; `&&`/`||`: `bool`). the `or` fallback is *not* here -- it is
    /// [`TypedExpr::OrElse`].
    Binary {
        op: BinOp,
        lhs: Box<TypedExpr>,
        rhs: Box<TypedExpr>,
        ty: QalaType,
        span: Span,
    },
    /// `start .. end` or `start ..= end`. resolved type is the iterable type
    /// the range stands for.
    Range {
        start: Option<Box<TypedExpr>>,
        end: Option<Box<TypedExpr>>,
        inclusive: bool,
        ty: QalaType,
        span: Span,
    },
    /// `lhs |> call` -- the pipeline operator. NOT desugared in the typed AST
    /// either; desugaring is codegen's job. resolved type is the call's
    /// return type.
    Pipeline {
        lhs: Box<TypedExpr>,
        call: Box<TypedExpr>,
        ty: QalaType,
        span: Span,
    },
    /// `comptime body` -- compile-time evaluated. resolved type is the body's
    /// type (the typechecker only types it; evaluation is codegen).
    Comptime {
        body: Box<TypedExpr>,
        ty: QalaType,
        span: Span,
    },
    /// `{ ...; trailing }` used as an expression. resolved type is the block's
    /// trailing-value type (or `void`); duplicated at this level so the
    /// uniform [`TypedExpr::ty`] accessor reads from a variant field rather
    /// than peeking into the [`TypedBlock`].
    Block {
        block: TypedBlock,
        ty: QalaType,
        span: Span,
    },
    /// `match scrutinee { arm, ... }`. NOT desugared. resolved type is the
    /// common type of all arm bodies; exhaustiveness is checked by the
    /// typechecker against the scrutinee's enum type.
    Match {
        scrutinee: Box<TypedExpr>,
        arms: Vec<TypedMatchArm>,
        ty: QalaType,
        span: Span,
    },
    /// `expr or fallback` -- the inline fallback for a `Result`/`Option`. NOT
    /// desugared. resolved type is the `T` of `Result<T, E>` / `Option<T>`.
    OrElse {
        expr: Box<TypedExpr>,
        fallback: Box<TypedExpr>,
        ty: QalaType,
        span: Span,
    },
    /// a string with `{expr}` interpolations. NOT desugared to a `+` chain;
    /// the conversion-and-concatenation happens in codegen. resolved
    /// type is [`QalaType::Str`].
    Interpolation {
        parts: Vec<TypedInterpPart>,
        ty: QalaType,
        span: Span,
    },
}

impl TypedExpr {
    /// the resolved type of this expression.
    ///
    /// exhaustive match: the proof every typed expression has a type. the
    /// typechecker fills this; codegen reads it to drive instruction selection
    /// (int add vs float add, struct-field offset, exhaustiveness, ...).
    pub fn ty(&self) -> &QalaType {
        match self {
            TypedExpr::Int { ty, .. }
            | TypedExpr::Float { ty, .. }
            | TypedExpr::Byte { ty, .. }
            | TypedExpr::Str { ty, .. }
            | TypedExpr::Bool { ty, .. }
            | TypedExpr::Ident { ty, .. }
            | TypedExpr::Paren { ty, .. }
            | TypedExpr::Tuple { ty, .. }
            | TypedExpr::ArrayLit { ty, .. }
            | TypedExpr::ArrayRepeat { ty, .. }
            | TypedExpr::StructLit { ty, .. }
            | TypedExpr::FieldAccess { ty, .. }
            | TypedExpr::MethodCall { ty, .. }
            | TypedExpr::Call { ty, .. }
            | TypedExpr::Index { ty, .. }
            | TypedExpr::Try { ty, .. }
            | TypedExpr::Unary { ty, .. }
            | TypedExpr::Binary { ty, .. }
            | TypedExpr::Range { ty, .. }
            | TypedExpr::Pipeline { ty, .. }
            | TypedExpr::Comptime { ty, .. }
            | TypedExpr::Block { ty, .. }
            | TypedExpr::Match { ty, .. }
            | TypedExpr::OrElse { ty, .. }
            | TypedExpr::Interpolation { ty, .. } => ty,
        }
    }

    /// the source span of this expression.
    ///
    /// every variant carries its `span` field; this match is exhaustive over
    /// all of them, the same shape as `ast::Expr::span()`. the typechecker
    /// copies the span from the untyped node when it constructs the typed one.
    pub fn span(&self) -> Span {
        match self {
            TypedExpr::Int { span, .. }
            | TypedExpr::Float { span, .. }
            | TypedExpr::Byte { span, .. }
            | TypedExpr::Str { span, .. }
            | TypedExpr::Bool { span, .. }
            | TypedExpr::Ident { span, .. }
            | TypedExpr::Paren { span, .. }
            | TypedExpr::Tuple { span, .. }
            | TypedExpr::ArrayLit { span, .. }
            | TypedExpr::ArrayRepeat { span, .. }
            | TypedExpr::StructLit { span, .. }
            | TypedExpr::FieldAccess { span, .. }
            | TypedExpr::MethodCall { span, .. }
            | TypedExpr::Call { span, .. }
            | TypedExpr::Index { span, .. }
            | TypedExpr::Try { span, .. }
            | TypedExpr::Unary { span, .. }
            | TypedExpr::Binary { span, .. }
            | TypedExpr::Range { span, .. }
            | TypedExpr::Pipeline { span, .. }
            | TypedExpr::Comptime { span, .. }
            | TypedExpr::Block { span, .. }
            | TypedExpr::Match { span, .. }
            | TypedExpr::OrElse { span, .. }
            | TypedExpr::Interpolation { span, .. } => *span,
        }
    }
}

/// one typed field initializer in a struct literal: `name: value`. mirror of
/// `ast::FieldInit`.
#[derive(Debug, Clone, PartialEq)]
pub struct TypedFieldInit {
    /// the field name.
    pub name: String,
    /// the typed value expression.
    pub value: TypedExpr,
    /// the initializer's source span (name to value).
    pub span: Span,
}

/// one piece of a typed string interpolation: either literal text, or an
/// embedded typed expression. matches `ast::InterpPart` but holds
/// [`TypedExpr`] instead of `Expr`.
#[derive(Debug, Clone, PartialEq)]
pub enum TypedInterpPart {
    /// literal text between interpolations (may be empty).
    Literal(String),
    /// an embedded `{ expr }`, type-checked to a value the conversion
    /// machinery can stringify.
    Expr(TypedExpr),
}

/// a typed `match` arm. mirror of `ast::MatchArm`. the pattern is re-used from
/// the untyped AST -- patterns carry no resolved type in v1 (the scrutinee's
/// [`TypedExpr::ty`] is the relevant fact during exhaustiveness checking).
#[derive(Debug, Clone, PartialEq)]
pub struct TypedMatchArm {
    /// the pattern this arm matches; reused unchanged from `ast::Pattern`.
    pub pattern: Pattern,
    /// the `if expr` guard, type-checked to `bool`, or `None`.
    pub guard: Option<TypedExpr>,
    /// the arm body -- a typed expression or a typed block.
    pub body: TypedMatchArmBody,
    /// the arm's source span.
    pub span: Span,
}

/// a typed `match` arm body: a bare typed expression (`Pattern => expr`) or a
/// typed block (`Pattern => { ... }`). mirror of `ast::MatchArmBody`.
#[derive(Debug, Clone, PartialEq)]
pub enum TypedMatchArmBody {
    /// `=> expr`.
    Expr(Box<TypedExpr>),
    /// `=> { ... }`.
    Block(TypedBlock),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{BinOp, Pattern, UnaryOp};
    use crate::effects::EffectSet;
    use crate::span::Span;
    use crate::types::{QalaType, Symbol};

    // a distinct span per call, so a wrong span() arm or a clobbered field is
    // caught: span(n) and span(m) never compare equal for n != m. same shape
    // as the helper in ast.rs's tests.
    fn span(n: usize) -> Span {
        Span::new(n, n + 1)
    }

    #[test]
    fn nodes_are_debug_clone_and_partial_eq() {
        // build a small typed expression tree, clone it, assert equal; build a
        // different one, assert not equal. mirror of ast.rs's analogous test.
        let a = TypedExpr::Binary {
            op: BinOp::Add,
            lhs: Box::new(TypedExpr::Int {
                value: 1,
                ty: QalaType::I64,
                span: span(0),
            }),
            rhs: Box::new(TypedExpr::Int {
                value: 2,
                ty: QalaType::I64,
                span: span(2),
            }),
            ty: QalaType::I64,
            span: span(0),
        };
        let b = a.clone();
        assert_eq!(a, b);
        // Debug must be derived too (assertion messages and the typechecker
        // tests use it).
        let _ = format!("{a:?}");
        let c = TypedExpr::Binary {
            op: BinOp::Mul, // different operator
            lhs: Box::new(TypedExpr::Int {
                value: 1,
                ty: QalaType::I64,
                span: span(0),
            }),
            rhs: Box::new(TypedExpr::Int {
                value: 2,
                ty: QalaType::I64,
                span: span(2),
            }),
            ty: QalaType::I64,
            span: span(0),
        };
        assert_ne!(a, c);
    }

    #[test]
    fn typed_expr_ty_returns_the_stored_type() {
        // a sample of variants, each with a chosen ty; .ty() must return a
        // borrow that compares equal to the chosen ty. this is the proof every
        // variant in the .ty() exhaustive match returns the right field.
        let cases: Vec<(TypedExpr, QalaType)> = vec![
            (
                TypedExpr::Int {
                    value: 0,
                    ty: QalaType::I64,
                    span: span(1),
                },
                QalaType::I64,
            ),
            (
                TypedExpr::Float {
                    value: 0.0,
                    ty: QalaType::F64,
                    span: span(2),
                },
                QalaType::F64,
            ),
            (
                TypedExpr::Byte {
                    value: 0,
                    ty: QalaType::Byte,
                    span: span(3),
                },
                QalaType::Byte,
            ),
            (
                TypedExpr::Str {
                    value: String::new(),
                    ty: QalaType::Str,
                    span: span(4),
                },
                QalaType::Str,
            ),
            (
                TypedExpr::Bool {
                    value: true,
                    ty: QalaType::Bool,
                    span: span(5),
                },
                QalaType::Bool,
            ),
            (
                TypedExpr::Ident {
                    name: "x".to_string(),
                    ty: QalaType::Named(Symbol("Shape".to_string())),
                    span: span(6),
                },
                QalaType::Named(Symbol("Shape".to_string())),
            ),
            (
                TypedExpr::Unary {
                    op: UnaryOp::Neg,
                    operand: Box::new(TypedExpr::Int {
                        value: 1,
                        ty: QalaType::I64,
                        span: span(8),
                    }),
                    ty: QalaType::I64,
                    span: span(7),
                },
                QalaType::I64,
            ),
            (
                TypedExpr::Tuple {
                    elems: vec![
                        TypedExpr::Int {
                            value: 1,
                            ty: QalaType::I64,
                            span: span(10),
                        },
                        TypedExpr::Bool {
                            value: true,
                            ty: QalaType::Bool,
                            span: span(11),
                        },
                    ],
                    ty: QalaType::Tuple(vec![QalaType::I64, QalaType::Bool]),
                    span: span(9),
                },
                QalaType::Tuple(vec![QalaType::I64, QalaType::Bool]),
            ),
            // Unknown is a legal poison type after a type error.
            (
                TypedExpr::Call {
                    callee: Box::new(TypedExpr::Ident {
                        name: "f".to_string(),
                        ty: QalaType::Unknown,
                        span: span(13),
                    }),
                    args: vec![],
                    ty: QalaType::Unknown,
                    span: span(12),
                },
                QalaType::Unknown,
            ),
        ];
        for (expr, expected) in cases {
            assert_eq!(expr.ty(), &expected, "ty() mismatch for {expr:?}");
        }
    }

    #[test]
    fn typed_expr_span_returns_the_stored_span() {
        // mirror of ast.rs's expr_span_returns_the_stored_span_for_a_sample_of_variants:
        // every faithful node form (incl. Pipeline / OrElse / Interpolation /
        // MethodCall) is represented so a missing arm in .span() trips the test.
        let cases: Vec<(TypedExpr, Span)> = vec![
            (
                TypedExpr::Int {
                    value: 0,
                    ty: QalaType::I64,
                    span: span(1),
                },
                span(1),
            ),
            (
                TypedExpr::Float {
                    value: 0.0,
                    ty: QalaType::F64,
                    span: span(2),
                },
                span(2),
            ),
            (
                TypedExpr::Byte {
                    value: 0,
                    ty: QalaType::Byte,
                    span: span(3),
                },
                span(3),
            ),
            (
                TypedExpr::Str {
                    value: String::new(),
                    ty: QalaType::Str,
                    span: span(4),
                },
                span(4),
            ),
            (
                TypedExpr::Bool {
                    value: true,
                    ty: QalaType::Bool,
                    span: span(5),
                },
                span(5),
            ),
            (
                TypedExpr::Ident {
                    name: "x".to_string(),
                    ty: QalaType::I64,
                    span: span(6),
                },
                span(6),
            ),
            (
                TypedExpr::Unary {
                    op: UnaryOp::Neg,
                    operand: Box::new(TypedExpr::Int {
                        value: 1,
                        ty: QalaType::I64,
                        span: span(8),
                    }),
                    ty: QalaType::I64,
                    span: span(7),
                },
                span(7),
            ),
            (
                TypedExpr::MethodCall {
                    receiver: Box::new(TypedExpr::Ident {
                        name: "f".to_string(),
                        ty: QalaType::FileHandle,
                        span: span(10),
                    }),
                    name: "read_all".to_string(),
                    args: vec![],
                    ty: QalaType::Str,
                    span: span(9),
                },
                span(9),
            ),
            (
                TypedExpr::Pipeline {
                    lhs: Box::new(TypedExpr::Int {
                        value: 5,
                        ty: QalaType::I64,
                        span: span(12),
                    }),
                    call: Box::new(TypedExpr::Ident {
                        name: "double".to_string(),
                        ty: QalaType::Function {
                            params: vec![QalaType::I64],
                            returns: Box::new(QalaType::I64),
                        },
                        span: span(14),
                    }),
                    ty: QalaType::I64,
                    span: span(11),
                },
                span(11),
            ),
            (
                TypedExpr::OrElse {
                    expr: Box::new(TypedExpr::Ident {
                        name: "a".to_string(),
                        ty: QalaType::Option(Box::new(QalaType::Str)),
                        span: span(16),
                    }),
                    fallback: Box::new(TypedExpr::Str {
                        value: "no data".to_string(),
                        ty: QalaType::Str,
                        span: span(18),
                    }),
                    ty: QalaType::Str,
                    span: span(15),
                },
                span(15),
            ),
            (
                TypedExpr::Interpolation {
                    parts: vec![
                        TypedInterpPart::Literal("fib(".to_string()),
                        TypedInterpPart::Expr(TypedExpr::Ident {
                            name: "i".to_string(),
                            ty: QalaType::I64,
                            span: span(20),
                        }),
                        TypedInterpPart::Literal(")".to_string()),
                    ],
                    ty: QalaType::Str,
                    span: span(19),
                },
                span(19),
            ),
            (
                TypedExpr::Match {
                    scrutinee: Box::new(TypedExpr::Ident {
                        name: "v".to_string(),
                        ty: QalaType::I64,
                        span: span(22),
                    }),
                    arms: vec![TypedMatchArm {
                        pattern: Pattern::Wildcard { span: span(23) },
                        guard: None,
                        body: TypedMatchArmBody::Expr(Box::new(TypedExpr::Int {
                            value: 0,
                            ty: QalaType::I64,
                            span: span(24),
                        })),
                        span: span(23),
                    }],
                    ty: QalaType::I64,
                    span: span(21),
                },
                span(21),
            ),
        ];
        for (expr, expected) in cases {
            assert_eq!(expr.span(), expected, "span() mismatch for {expr:?}");
        }
    }

    #[test]
    fn typed_stmt_span_returns_the_stored_span() {
        // mirror of ast.rs's stmt-span test.
        let cases: Vec<(TypedStmt, Span)> = vec![
            (
                TypedStmt::Let {
                    is_mut: false,
                    name: "x".to_string(),
                    ty: QalaType::I64,
                    init: TypedExpr::Int {
                        value: 1,
                        ty: QalaType::I64,
                        span: span(1),
                    },
                    span: span(0),
                },
                span(0),
            ),
            (TypedStmt::Break { span: span(2) }, span(2)),
            (TypedStmt::Continue { span: span(3) }, span(3)),
            (
                TypedStmt::Return {
                    value: None,
                    span: span(4),
                },
                span(4),
            ),
            (
                TypedStmt::For {
                    var: "i".to_string(),
                    var_ty: QalaType::I64,
                    iter: TypedExpr::Range {
                        start: Some(Box::new(TypedExpr::Int {
                            value: 0,
                            ty: QalaType::I64,
                            span: span(6),
                        })),
                        end: Some(Box::new(TypedExpr::Int {
                            value: 15,
                            ty: QalaType::I64,
                            span: span(8),
                        })),
                        inclusive: false,
                        ty: QalaType::Array(Box::new(QalaType::I64), None),
                        span: span(6),
                    },
                    body: TypedBlock {
                        stmts: vec![],
                        value: None,
                        ty: QalaType::Void,
                        span: span(9),
                    },
                    span: span(5),
                },
                span(5),
            ),
            (
                TypedStmt::Defer {
                    expr: TypedExpr::Call {
                        callee: Box::new(TypedExpr::Ident {
                            name: "close".to_string(),
                            ty: QalaType::Function {
                                params: vec![QalaType::FileHandle],
                                returns: Box::new(QalaType::Void),
                            },
                            span: span(11),
                        }),
                        args: vec![TypedExpr::Ident {
                            name: "f".to_string(),
                            ty: QalaType::FileHandle,
                            span: span(12),
                        }],
                        ty: QalaType::Void,
                        span: span(11),
                    },
                    span: span(10),
                },
                span(10),
            ),
        ];
        for (stmt, expected) in cases {
            assert_eq!(stmt.span(), expected, "span() mismatch for {stmt:?}");
        }
    }

    #[test]
    fn typed_item_span_delegates_to_the_wrapped_decl() {
        // mirror of ast.rs's item_span_delegates_to_the_wrapped_decl: build
        // one of each TypedItem kind and check the span() delegates correctly.
        let f = TypedItem::Fn(TypedFnDecl {
            type_name: None,
            name: "main".to_string(),
            params: vec![],
            ret_ty: QalaType::Void,
            effect: EffectSet::io(),
            body: TypedBlock {
                stmts: vec![],
                value: None,
                ty: QalaType::Void,
                span: span(1),
            },
            span: span(0),
        });
        assert_eq!(f.span(), span(0));
        let s = TypedItem::Struct(TypedStructDecl {
            name: "S".to_string(),
            fields: vec![],
            span: span(2),
        });
        assert_eq!(s.span(), span(2));
        let e = TypedItem::Enum(TypedEnumDecl {
            name: "E".to_string(),
            variants: vec![],
            span: span(3),
        });
        assert_eq!(e.span(), span(3));
        let i = TypedItem::Interface(TypedInterfaceDecl {
            name: "I".to_string(),
            methods: vec![],
            span: span(4),
        });
        assert_eq!(i.span(), span(4));
    }

    #[test]
    fn typed_fn_decl_effect_returns_the_stored_effect_set() {
        // every effect annotation flavor: pure (empty), single-flag (io), and
        // a unioned set must round-trip through .effect().
        let cases = [
            EffectSet::pure(),
            EffectSet::io(),
            EffectSet::alloc(),
            EffectSet::panic(),
            EffectSet::io().union(EffectSet::alloc()),
            EffectSet::full(),
        ];
        for eff in cases {
            let f = TypedFnDecl {
                type_name: None,
                name: "f".to_string(),
                params: vec![],
                ret_ty: QalaType::Void,
                effect: eff,
                body: TypedBlock {
                    stmts: vec![],
                    value: None,
                    ty: QalaType::Void,
                    span: span(1),
                },
                span: span(0),
            };
            assert_eq!(f.effect(), eff, "effect() mismatch for {eff:?}");
        }
    }

    #[test]
    fn typed_block_ty_is_void_for_empty_block() {
        // an empty block has no trailing value; its resolved ty is Void.
        let b = TypedBlock {
            stmts: vec![],
            value: None,
            ty: QalaType::Void,
            span: span(0),
        };
        assert_eq!(b.ty, QalaType::Void);
        // a block with a trailing value carries that value's type.
        let b2 = TypedBlock {
            stmts: vec![TypedStmt::Let {
                is_mut: false,
                name: "x".to_string(),
                ty: QalaType::I64,
                init: TypedExpr::Int {
                    value: 1,
                    ty: QalaType::I64,
                    span: span(1),
                },
                span: span(0),
            }],
            value: Some(Box::new(TypedExpr::Ident {
                name: "x".to_string(),
                ty: QalaType::I64,
                span: span(3),
            })),
            ty: QalaType::I64,
            span: span(0),
        };
        assert_eq!(b2.ty, QalaType::I64);
        assert!(b2.value.is_some());
        assert_eq!(b2.stmts.len(), 1);
    }

    #[test]
    fn typed_ast_is_a_vec_of_typed_items() {
        // TypedAst is a type alias over Vec<TypedItem> -- an empty source is
        // an empty Vec; pushing items appends as usual. mirror of ast.rs's
        // type alias usage pattern.
        let mut prog: TypedAst = vec![];
        assert!(prog.is_empty());
        prog.push(TypedItem::Struct(TypedStructDecl {
            name: "Point".to_string(),
            fields: vec![
                TypedField {
                    name: "x".to_string(),
                    ty: QalaType::I64,
                    span: span(1),
                },
                TypedField {
                    name: "y".to_string(),
                    ty: QalaType::I64,
                    span: span(2),
                },
            ],
            span: span(0),
        }));
        assert_eq!(prog.len(), 1);
    }

    #[test]
    fn reexported_unary_and_bin_ops_are_usable_in_typed_ast() {
        // UnaryOp and BinOp are reused from ast.rs (no type info to add to a
        // bare operator). this test pins the fact that they are reachable from
        // typed_ast call sites without a duplicate definition.
        let _u: UnaryOp = UnaryOp::Not;
        let _b: BinOp = BinOp::Add;
        // a typed Unary node uses the ast::UnaryOp value directly.
        let neg = TypedExpr::Unary {
            op: UnaryOp::Neg,
            operand: Box::new(TypedExpr::Int {
                value: 5,
                ty: QalaType::I64,
                span: span(1),
            }),
            ty: QalaType::I64,
            span: span(0),
        };
        assert_eq!(neg.ty(), &QalaType::I64);
    }

    #[test]
    fn pattern_is_reused_from_ast_in_typed_match_arms() {
        // patterns carry no resolved type in v1 -- the scrutinee's TypedExpr
        // .ty() is the relevant fact during exhaustiveness checking. so a
        // TypedMatchArm holds the original ast::Pattern unchanged.
        let arm = TypedMatchArm {
            pattern: Pattern::Variant {
                name: "Circle".to_string(),
                sub: vec![Pattern::Binding {
                    name: "r".to_string(),
                    span: span(1),
                }],
                span: span(0),
            },
            guard: Some(TypedExpr::Binary {
                op: BinOp::Gt,
                lhs: Box::new(TypedExpr::Ident {
                    name: "r".to_string(),
                    ty: QalaType::F64,
                    span: span(2),
                }),
                rhs: Box::new(TypedExpr::Float {
                    value: 0.0,
                    ty: QalaType::F64,
                    span: span(3),
                }),
                ty: QalaType::Bool,
                span: span(2),
            }),
            body: TypedMatchArmBody::Expr(Box::new(TypedExpr::Float {
                value: 1.0,
                ty: QalaType::F64,
                span: span(4),
            })),
            span: span(0),
        };
        // the pattern compares equal to itself (it is the ast::Pattern's
        // derived PartialEq, unchanged here).
        assert_eq!(
            arm.pattern,
            Pattern::Variant {
                name: "Circle".to_string(),
                sub: vec![Pattern::Binding {
                    name: "r".to_string(),
                    span: span(1)
                }],
                span: span(0),
            }
        );
        assert!(arm.guard.is_some());
    }

    #[test]
    fn faithful_nodes_pipeline_match_orelse_interpolation_exist() {
        // the typed AST is faithful, same as the untyped AST: Pipeline,
        // Match, OrElse, and Interpolation are real variants, not desugared.
        // build one of each and confirm .ty() / .span() work.
        let pipe = TypedExpr::Pipeline {
            lhs: Box::new(TypedExpr::Int {
                value: 1,
                ty: QalaType::I64,
                span: span(1),
            }),
            call: Box::new(TypedExpr::Ident {
                name: "double".to_string(),
                ty: QalaType::Function {
                    params: vec![QalaType::I64],
                    returns: Box::new(QalaType::I64),
                },
                span: span(3),
            }),
            ty: QalaType::I64,
            span: span(0),
        };
        assert_eq!(pipe.ty(), &QalaType::I64);
        assert_eq!(pipe.span(), span(0));

        let m = TypedExpr::Match {
            scrutinee: Box::new(TypedExpr::Bool {
                value: true,
                ty: QalaType::Bool,
                span: span(5),
            }),
            arms: vec![TypedMatchArm {
                pattern: Pattern::Bool {
                    value: true,
                    span: span(6),
                },
                guard: None,
                body: TypedMatchArmBody::Expr(Box::new(TypedExpr::Int {
                    value: 1,
                    ty: QalaType::I64,
                    span: span(7),
                })),
                span: span(6),
            }],
            ty: QalaType::I64,
            span: span(4),
        };
        assert_eq!(m.ty(), &QalaType::I64);

        let or = TypedExpr::OrElse {
            expr: Box::new(TypedExpr::Ident {
                name: "x".to_string(),
                ty: QalaType::Option(Box::new(QalaType::I64)),
                span: span(9),
            }),
            fallback: Box::new(TypedExpr::Int {
                value: 0,
                ty: QalaType::I64,
                span: span(10),
            }),
            ty: QalaType::I64,
            span: span(8),
        };
        assert_eq!(or.ty(), &QalaType::I64);

        let interp = TypedExpr::Interpolation {
            parts: vec![
                TypedInterpPart::Literal("hello, ".to_string()),
                TypedInterpPart::Expr(TypedExpr::Ident {
                    name: "name".to_string(),
                    ty: QalaType::Str,
                    span: span(12),
                }),
            ],
            ty: QalaType::Str,
            span: span(11),
        };
        assert_eq!(interp.ty(), &QalaType::Str);
    }
}
