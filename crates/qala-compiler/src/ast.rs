//! the untyped AST: what the parser produces and the type checker consumes.
//! every node carries a [`Span`] (so Phase 3's diagnostics underline the source
//! as written); the tree is boxed at recursive positions; nothing is desugared
//! -- [`Expr::Pipeline`], [`Expr::Interpolation`], [`Stmt::For`], and
//! [`Expr::Match`] are real nodes, not lowered to calls / `+` chains / `while`.
//!
//! `if`/`else` is a statement, not an expression: there is no `Expr::If`. only
//! `match` and `{ ... }` blocks produce values in expression position. method
//! calls get their own node ([`Expr::MethodCall`]) distinct from field access
//! ([`Expr::FieldAccess`]) -- the parser decides based on whether `(` follows
//! the `.name`; Phase 3 resolves a method call to the free function
//! `fn Type.name(self, ...)`.
//!
//! the node enums derive `Debug, Clone, PartialEq` and no more: no `Eq`,
//! because float literals reach the AST ([`Expr::Float`]) and `f64` is not
//! `Eq`; no `serde`, because the type checker and codegen run in-process and
//! never serialize the tree. spans are not merged here -- the parser builds
//! each node *with* its computed span (first child to last child, via
//! [`Span::to`]); `ast.rs` only stores and reads the field.

use crate::span::Span;

/// a whole program: the top-level items in source order.
///
/// a type alias rather than a wrapper struct -- a program's span is just the
/// span of its source, which the caller already holds, so a node here would add
/// nothing. an empty source parses to an empty `Vec`.
pub type Ast = Vec<Item>;

// ---- items -----------------------------------------------------------------

/// a top-level declaration: a function, a struct, an enum, or an interface.
/// `fn Type.method` definitions are an [`Item::Fn`] whose [`FnDecl`] carries an
/// optional `type_name`.
#[derive(Debug, Clone, PartialEq)]
pub enum Item {
    /// a `fn` declaration -- a free function, or, if `type_name` is set, a
    /// `fn Type.method` method definition.
    Fn(FnDecl),
    /// a `struct` declaration with typed fields.
    Struct(StructDecl),
    /// an `enum` declaration with data-carrying variants.
    Enum(EnumDecl),
    /// an `interface` declaration -- a set of method signatures, no bodies.
    Interface(InterfaceDecl),
}

impl Item {
    /// the source span of this item, opening keyword to closing brace.
    ///
    /// each item kind wraps a struct that carries the span; this delegates to
    /// it. exhaustive over every kind, so a new item kind forces an arm here.
    pub fn span(&self) -> Span {
        match self {
            Item::Fn(d) => d.span,
            Item::Struct(d) => d.span,
            Item::Enum(d) => d.span,
            Item::Interface(d) => d.span,
        }
    }
}

/// a `fn` declaration. `type_name` is `Some` for a `fn Type.method` definition
/// (the receiver type's name) and `None` for a free function. `ret_ty` is the
/// `-> T` return type, omitted for `void`. `effect` is the `is pure`/`io`/
/// `alloc`/`panic` annotation, omitted when Phase 3 should infer it.
#[derive(Debug, Clone, PartialEq)]
pub struct FnDecl {
    /// the receiver type's name for a `fn Type.method` definition, else `None`.
    pub type_name: Option<String>,
    /// the function (or method) name.
    pub name: String,
    /// the parameter list, in order; for a method, the first may be `self`.
    pub params: Vec<Param>,
    /// the `-> T` return type, or `None` for `void`.
    pub ret_ty: Option<TypeExpr>,
    /// the `is pure`/`io`/`alloc`/`panic` annotation, or `None` to infer.
    pub effect: Option<Effect>,
    /// the function body; an empty `{ }` body is a [`Block`] with no statements.
    pub body: Block,
    /// `fn` keyword to closing `}`.
    pub span: Span,
}

/// one parameter of a function or method. `is_self` is true only for the `self`
/// first parameter of a method, which has no declared `ty`; every other
/// parameter requires a `ty`. `default` is the optional `= expr` default value.
#[derive(Debug, Clone, PartialEq)]
pub struct Param {
    /// true if this is the untyped `self` first parameter of a method.
    pub is_self: bool,
    /// the parameter name (`"self"` when `is_self`).
    pub name: String,
    /// the declared type; `None` only when `is_self`.
    pub ty: Option<TypeExpr>,
    /// the `= expr` default value, or `None`.
    pub default: Option<Expr>,
    /// the parameter's source span (name to default, or name to type).
    pub span: Span,
}

/// the four effect annotations. this is the parser's record of *which keyword
/// appeared* (`is pure` etc.); the type checker's `effects` module (Phase 3)
/// owns the real effect lattice and inference.
#[derive(Debug, Clone, PartialEq)]
pub enum Effect {
    /// `is pure` -- only computes; no IO, no allocation, no panic.
    Pure,
    /// `is io` -- may perform input/output.
    Io,
    /// `is alloc` -- may allocate.
    Alloc,
    /// `is panic` -- may panic.
    Panic,
}

/// a `struct` declaration: a name and its typed fields, in declaration order.
#[derive(Debug, Clone, PartialEq)]
pub struct StructDecl {
    /// the struct's name.
    pub name: String,
    /// the fields, in declaration order.
    pub fields: Vec<Field>,
    /// `struct` keyword to closing `}`.
    pub span: Span,
}

/// one field of a struct: a name and a type (struct fields are always typed --
/// no inference at declaration sites).
#[derive(Debug, Clone, PartialEq)]
pub struct Field {
    /// the field name.
    pub name: String,
    /// the field's declared type.
    pub ty: TypeExpr,
    /// the field's source span (name to type).
    pub span: Span,
}

/// an `enum` declaration: a name and its variants, in declaration order. each
/// variant may carry data (a tuple of types).
#[derive(Debug, Clone, PartialEq)]
pub struct EnumDecl {
    /// the enum's name.
    pub name: String,
    /// the variants, in declaration order.
    pub variants: Vec<Variant>,
    /// `enum` keyword to closing `}`.
    pub span: Span,
}

/// one variant of an enum: a name and zero or more field types. `North` has no
/// fields, `Circle(f64)` one, `Rect(f64, f64)` two.
#[derive(Debug, Clone, PartialEq)]
pub struct Variant {
    /// the variant name.
    pub name: String,
    /// the data the variant carries, as a list of field types (possibly empty).
    pub fields: Vec<TypeExpr>,
    /// the variant's source span (name, plus its `( ... )` if any).
    pub span: Span,
}

/// an `interface` declaration: a name and a set of method signatures (no
/// bodies). a type satisfies the interface structurally -- by having matching
/// methods -- with no `implements` declaration; Phase 3 checks that.
#[derive(Debug, Clone, PartialEq)]
pub struct InterfaceDecl {
    /// the interface's name.
    pub name: String,
    /// the required method signatures.
    pub methods: Vec<MethodSig>,
    /// `interface` keyword to closing `}`.
    pub span: Span,
}

/// a method signature in an interface: like a [`FnDecl`] without a body and
/// without a `type_name` (the receiver is whatever type satisfies the
/// interface). the first parameter is typically `self`.
#[derive(Debug, Clone, PartialEq)]
pub struct MethodSig {
    /// the method name.
    pub name: String,
    /// the parameter list; the first is typically `self`.
    pub params: Vec<Param>,
    /// the `-> T` return type, or `None` for `void`.
    pub ret_ty: Option<TypeExpr>,
    /// the `is ...` effect annotation, or `None`.
    pub effect: Option<Effect>,
    /// the signature's source span.
    pub span: Span,
}

// ---- statements and blocks -------------------------------------------------

/// a `{ ... }` block: a sequence of statements, optionally ending in a trailing
/// expression with no `;`. `value` is that trailing expression; `None` means
/// the block's value is `void` (it ended in a `;`, or it was empty). a block is
/// also an expression in this language -- see [`Expr::Block`].
#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    /// the statements, in order.
    pub stmts: Vec<Stmt>,
    /// the trailing expression (no terminating `;`) that is the block's value,
    /// or `None` for a `void` block.
    pub value: Option<Box<Expr>>,
    /// `{` to `}`.
    pub span: Span,
}

/// a statement. `if`/`else`, `while`, `for ... in ...`, `return`, `break`,
/// `continue`, and `defer expr` are all statements -- not expressions; there is
/// no `Expr::If` in v1. an `if` chains through [`ElseBranch::If`] for
/// `else if`.
#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    /// `let name = init` or `let mut name = init`, with an optional `: ty`. the
    /// initializer is required (no uninitialized bindings); the type is inferred
    /// from `init` when omitted.
    Let {
        is_mut: bool,
        name: String,
        ty: Option<TypeExpr>,
        init: Expr,
        span: Span,
    },
    /// `if cond { ... }` with an optional `else` (a block, or another `if` for
    /// an `else if` chain). a statement, not an expression -- there is no
    /// `let x = if c { a } else { b }` in v1.
    If {
        cond: Expr,
        then_block: Block,
        else_branch: Option<ElseBranch>,
        span: Span,
    },
    /// `while cond { ... }`.
    While { cond: Expr, body: Block, span: Span },
    /// `for var in iter { ... }`. `iter` is any expression that yields an
    /// iterable; a range is just one kind of iterable. kept as a `For` node, not
    /// lowered to a `while`.
    For {
        var: String,
        iter: Expr,
        body: Block,
        span: Span,
    },
    /// `return` or `return expr`. the value is optional -- a bare `return` in a
    /// `void` function is legal.
    Return { value: Option<Expr>, span: Span },
    /// `break`. no labels, no value in v1.
    Break { span: Span },
    /// `continue`. no labels, no value in v1.
    Continue { span: Span },
    /// `defer expr` -- `expr` runs at scope exit, LIFO with other defers in the
    /// same scope.
    Defer { expr: Expr, span: Span },
    /// an expression used as a statement (`expr ;`). its value is discarded.
    Expr { expr: Expr, span: Span },
}

impl Stmt {
    /// the source span of this statement.
    ///
    /// every variant carries its `span` field; this match is exhaustive, which
    /// is the proof every statement has one. the parser builds the span from the
    /// leading keyword (or the first sub-expression) to the last token.
    pub fn span(&self) -> Span {
        match self {
            Stmt::Let { span, .. }
            | Stmt::If { span, .. }
            | Stmt::While { span, .. }
            | Stmt::For { span, .. }
            | Stmt::Return { span, .. }
            | Stmt::Break { span }
            | Stmt::Continue { span }
            | Stmt::Defer { span, .. }
            | Stmt::Expr { span, .. } => *span,
        }
    }
}

/// the `else` part of an [`Stmt::If`]: either a final `{ ... }` block, or
/// another `if` (the boxed [`Stmt`] is always an [`Stmt::If`]) for an
/// `else if` chain.
#[derive(Debug, Clone, PartialEq)]
pub enum ElseBranch {
    /// `else { ... }`.
    Block(Block),
    /// `else if ...` -- the boxed statement is an [`Stmt::If`].
    If(Box<Stmt>),
}

// ---- expressions -----------------------------------------------------------

/// an expression. literals, identifiers, parenthesized and tuple expressions,
/// array literals (including the `[v; n]` repeat form), struct literals, field
/// access, method calls, calls, indexing, `?` propagation, unary and binary
/// operators, ranges, the `|>` pipeline, `comptime`, block expressions, `match`,
/// the `or` fallback, and string interpolation. there is deliberately no `If`
/// variant -- `if`/`else` is a [`Stmt`] in v1.
///
/// nothing is desugared: [`Expr::Pipeline`] stays a pipeline (not `f(x, a)`),
/// [`Expr::Interpolation`] stays a parts list (not a `+` chain), [`Expr::Match`]
/// and [`Expr::Block`] produce values directly. that faithfulness is what lets
/// Phase 3 diagnostics point at the source as written; lowering happens in
/// Phase 4 codegen.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// an integer literal (already decoded; a leading `-` is a separate
    /// [`Expr::Unary`]).
    Int { value: i64, span: Span },
    /// a float literal (already decoded).
    Float { value: f64, span: Span },
    /// a byte literal `b'X'`, the byte it denotes.
    Byte { value: u8, span: Span },
    /// a string literal with no interpolation, escapes already decoded.
    Str { value: String, span: Span },
    /// a boolean literal, `true` or `false`.
    Bool { value: bool, span: Span },
    /// an identifier reference (a variable, a function name, a type name in
    /// value position -- the parser does not distinguish; Phase 3 resolves it).
    Ident { name: String, span: Span },
    /// `( inner )` -- a parenthesized expression. kept as a node so the span and
    /// the parentheses are visible to diagnostics; semantically transparent.
    Paren { inner: Box<Expr>, span: Span },
    /// `( e1, e2, ... )` -- a tuple. a single element with a trailing comma is a
    /// one-tuple; without the comma it is just a [`Expr::Paren`].
    Tuple { elems: Vec<Expr>, span: Span },
    /// `[ e1, e2, ... ]` -- an array literal listing its elements.
    ArrayLit { elems: Vec<Expr>, span: Span },
    /// `[ value; count ]` -- an array literal repeating `value` `count` times.
    ArrayRepeat {
        value: Box<Expr>,
        count: Box<Expr>,
        span: Span,
    },
    /// `Name { field: e, ... }` -- a struct literal. `name` is the struct's
    /// name; whether it names a real struct is Phase 3's call.
    StructLit {
        name: String,
        fields: Vec<FieldInit>,
        span: Span,
    },
    /// `obj.name` -- field access. distinct from [`Expr::MethodCall`]: this is
    /// the form with no `( ... )` after the `.name`.
    FieldAccess {
        obj: Box<Expr>,
        name: String,
        span: Span,
    },
    /// `receiver.name(args)` -- a method call. distinct from a field access
    /// followed by a call; Phase 3 resolves it to `fn Type.name(self, ...)`.
    MethodCall {
        receiver: Box<Expr>,
        name: String,
        args: Vec<Expr>,
        span: Span,
    },
    /// `callee(args)` -- a call expression (`callee` is usually an
    /// [`Expr::Ident`], but can be any expression that yields something
    /// callable).
    Call {
        callee: Box<Expr>,
        args: Vec<Expr>,
        span: Span,
    },
    /// `obj[index]` -- an index expression.
    Index {
        obj: Box<Expr>,
        index: Box<Expr>,
        span: Span,
    },
    /// `expr?` -- error propagation: if `expr` is an error, return it from the
    /// enclosing function; otherwise unwrap it. binds tightest (with `.`, call,
    /// index), so `f()?.x` is `(f()?).x` and `a + b?` is `a + (b?)`.
    Try { expr: Box<Expr>, span: Span },
    /// a unary-operator application: `!operand` or `-operand`.
    Unary {
        op: UnaryOp,
        operand: Box<Expr>,
        span: Span,
    },
    /// a binary-operator application: `lhs op rhs`. covers arithmetic,
    /// comparison, equality, and the boolean `&&` / `||`. the `or` fallback is
    /// *not* here -- it is [`Expr::OrElse`].
    Binary {
        op: BinOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
        span: Span,
    },
    /// `start .. end` or `start ..= end` -- a range. `start` and `end` are each
    /// optional to leave room for `..end` / `start..` forms; `inclusive` is true
    /// for `..=`. ranges are ordinary expressions (an iterable for `for`).
    Range {
        start: Option<Box<Expr>>,
        end: Option<Box<Expr>>,
        inclusive: bool,
        span: Span,
    },
    /// `lhs |> call` -- the pipeline operator. kept as a faithful node, NOT
    /// lowered to `call(lhs, ...)`; that desugaring is Phase 4 codegen. `call`
    /// is whatever expression followed `|>` (typically a call or a bare function
    /// name); the parser does not rewrite it.
    Pipeline {
        lhs: Box<Expr>,
        call: Box<Expr>,
        span: Span,
    },
    /// `comptime body` -- evaluate `body` during compilation, embed the result
    /// as a constant. a prefix operator at unary precedence, so `comptime a + b`
    /// is `(comptime a) + b`; `body` may also be a block expression
    /// (`comptime { ... }`).
    Comptime { body: Box<Expr>, span: Span },
    /// `{ ...; trailing }` used as an expression -- its value is the block's
    /// trailing expression (or `void`). one of only two value-producing block
    /// forms; the other is [`Expr::Match`].
    Block { block: Block, span: Span },
    /// `match scrutinee { arm, ... }` -- pattern matching with at least one arm.
    /// kept as a faithful node. each arm has a pattern, an optional `if` guard,
    /// and a body (an expression or a block); Phase 3 checks exhaustiveness.
    Match {
        scrutinee: Box<Expr>,
        arms: Vec<MatchArm>,
        span: Span,
    },
    /// `expr or fallback` -- inline fallback for a `Result`/`Option`: the value
    /// of `expr` if it is `Ok`/`Some`, else `fallback`. modelled as its own
    /// node (not a [`BinOp`]) because it is faithful surface syntax with the
    /// loosest precedence, left-associative; `a or b or c` is `(a or b) or c`.
    OrElse {
        expr: Box<Expr>,
        fallback: Box<Expr>,
        span: Span,
    },
    /// a string with `{expr}` interpolations -- `parts` alternates literal text
    /// and embedded expressions, in source order. NOT desugared to a `+` chain;
    /// the conversion-and-concatenation happens in Phase 4 codegen.
    Interpolation { parts: Vec<InterpPart>, span: Span },
}

impl Expr {
    /// the source span of this expression.
    ///
    /// every variant carries its `span` field; this match is exhaustive over
    /// all of them, which is the real guarantee that "every expression node has
    /// a span". the parser computes each span from the first token to the last
    /// (via [`Span::to`]) when it builds the node.
    pub fn span(&self) -> Span {
        match self {
            Expr::Int { span, .. }
            | Expr::Float { span, .. }
            | Expr::Byte { span, .. }
            | Expr::Str { span, .. }
            | Expr::Bool { span, .. }
            | Expr::Ident { span, .. }
            | Expr::Paren { span, .. }
            | Expr::Tuple { span, .. }
            | Expr::ArrayLit { span, .. }
            | Expr::ArrayRepeat { span, .. }
            | Expr::StructLit { span, .. }
            | Expr::FieldAccess { span, .. }
            | Expr::MethodCall { span, .. }
            | Expr::Call { span, .. }
            | Expr::Index { span, .. }
            | Expr::Try { span, .. }
            | Expr::Unary { span, .. }
            | Expr::Binary { span, .. }
            | Expr::Range { span, .. }
            | Expr::Pipeline { span, .. }
            | Expr::Comptime { span, .. }
            | Expr::Block { span, .. }
            | Expr::Match { span, .. }
            | Expr::OrElse { span, .. }
            | Expr::Interpolation { span, .. } => *span,
        }
    }
}

/// one field initializer in a struct literal: `name: value`.
#[derive(Debug, Clone, PartialEq)]
pub struct FieldInit {
    /// the field name.
    pub name: String,
    /// the value expression.
    pub value: Expr,
    /// the initializer's source span (name to value).
    pub span: Span,
}

/// a unary prefix operator.
#[derive(Debug, Clone, PartialEq)]
pub enum UnaryOp {
    /// `!` -- boolean negation.
    Not,
    /// `-` -- arithmetic negation.
    Neg,
}

/// a binary operator. note `||` (boolean or) is [`BinOp::Or`]; the `or` fallback
/// keyword is *not* a binary operator -- it is [`Expr::OrElse`].
#[derive(Debug, Clone, PartialEq)]
pub enum BinOp {
    /// `+`
    Add,
    /// `-`
    Sub,
    /// `*`
    Mul,
    /// `/`
    Div,
    /// `%`
    Rem,
    /// `==`
    Eq,
    /// `!=`
    Ne,
    /// `<`
    Lt,
    /// `<=`
    Le,
    /// `>`
    Gt,
    /// `>=`
    Ge,
    /// `&&` -- boolean and.
    And,
    /// `||` -- boolean or (distinct from the `or` fallback keyword).
    Or,
}

/// one piece of a string interpolation: either literal text, or an embedded
/// expression. an [`Expr::Interpolation`]'s `parts` list alternates these,
/// starting and ending with a (possibly empty) `Literal`.
#[derive(Debug, Clone, PartialEq)]
pub enum InterpPart {
    /// literal text between interpolations (escapes already decoded; may be
    /// empty).
    Literal(String),
    /// an embedded `{ expr }`.
    Expr(Expr),
}

// ---- patterns --------------------------------------------------------------

/// a pattern in a `match` arm. v1 has variant destructure (with sub-patterns),
/// the `_` wildcard, a name binding, and literal patterns; no `or`-patterns, no
/// struct patterns, no nested guards (guards live on the [`MatchArm`]).
#[derive(Debug, Clone, PartialEq)]
pub enum Pattern {
    /// `Name(sub, ...)` -- destructure an enum variant; `sub` is the
    /// sub-patterns for its fields (empty for a no-data variant matched as
    /// `Name`). whether `Name` is a real variant is Phase 3's call.
    Variant {
        name: String,
        sub: Vec<Pattern>,
        span: Span,
    },
    /// `_` -- matches anything, binds nothing.
    Wildcard { span: Span },
    /// `name` -- matches anything, binds it to `name`.
    Binding { name: String, span: Span },
    /// an integer literal pattern.
    Int { value: i64, span: Span },
    /// a float literal pattern.
    Float { value: f64, span: Span },
    /// a byte literal pattern.
    Byte { value: u8, span: Span },
    /// a string literal pattern.
    Str { value: String, span: Span },
    /// a boolean literal pattern, `true` or `false`.
    Bool { value: bool, span: Span },
}

impl Pattern {
    /// the source span of this pattern.
    ///
    /// every variant carries its `span` field; the match is exhaustive, the
    /// proof every pattern has one.
    pub fn span(&self) -> Span {
        match self {
            Pattern::Variant { span, .. }
            | Pattern::Wildcard { span }
            | Pattern::Binding { span, .. }
            | Pattern::Int { span, .. }
            | Pattern::Float { span, .. }
            | Pattern::Byte { span, .. }
            | Pattern::Str { span, .. }
            | Pattern::Bool { span, .. } => *span,
        }
    }
}

/// one arm of a `match`: a pattern, an optional `if` guard, and a body. the
/// body is an expression or a block; arms are comma-separated (the last arm's
/// comma is optional).
#[derive(Debug, Clone, PartialEq)]
pub struct MatchArm {
    /// the pattern this arm matches.
    pub pattern: Pattern,
    /// the `if expr` guard, or `None`.
    pub guard: Option<Expr>,
    /// the arm body -- an expression or a block.
    pub body: MatchArmBody,
    /// the arm's source span (pattern to body).
    pub span: Span,
}

/// a `match` arm body: a bare expression (`Pattern => expr`) or a block
/// (`Pattern => { ... }`).
#[derive(Debug, Clone, PartialEq)]
pub enum MatchArmBody {
    /// `=> expr`.
    Expr(Box<Expr>),
    /// `=> { ... }`.
    Block(Block),
}

// ---- type expressions ------------------------------------------------------

/// a type as written in source: a primitive name, a named type, a fixed or
/// dynamic array, a tuple, a function type, or a generic application. type
/// expressions appear in parameter and return positions, struct fields, enum
/// variant data, and `let x: T = ...`.
#[derive(Debug, Clone, PartialEq)]
pub enum TypeExpr {
    /// a primitive type name: `i64`, `f64`, `bool`, `str`, `byte`, `void`.
    Primitive { kind: PrimType, span: Span },
    /// a named type -- a struct, enum, or interface name (the parser does not
    /// distinguish; Phase 3 resolves it).
    Named { name: String, span: Span },
    /// `[ elem; size ]` -- a fixed-size array. `size` is the literal length.
    Array {
        elem: Box<TypeExpr>,
        size: u64,
        span: Span,
    },
    /// `[ elem ]` -- a dynamic array.
    DynArray { elem: Box<TypeExpr>, span: Span },
    /// `( T1, T2, ... )` -- a tuple type.
    Tuple { elems: Vec<TypeExpr>, span: Span },
    /// `fn( T1, T2 ) -> T3` -- a function type.
    Fn {
        params: Vec<TypeExpr>,
        ret: Box<TypeExpr>,
        span: Span,
    },
    /// `Name< T1, T2 >` -- a generic application (used by `Result<T, E>` and
    /// `Option<T>`). the parser allows it on any name; restricting which names
    /// may be generic is Phase 3's job.
    Generic {
        name: String,
        args: Vec<TypeExpr>,
        span: Span,
    },
}

impl TypeExpr {
    /// the source span of this type expression.
    ///
    /// every variant carries its `span` field; the match is exhaustive, the
    /// proof every type node has one.
    pub fn span(&self) -> Span {
        match self {
            TypeExpr::Primitive { span, .. }
            | TypeExpr::Named { span, .. }
            | TypeExpr::Array { span, .. }
            | TypeExpr::DynArray { span, .. }
            | TypeExpr::Tuple { span, .. }
            | TypeExpr::Fn { span, .. }
            | TypeExpr::Generic { span, .. } => *span,
        }
    }
}

/// the six primitive type names.
#[derive(Debug, Clone, PartialEq)]
pub enum PrimType {
    /// `i64`
    I64,
    /// `f64`
    F64,
    /// `bool`
    Bool,
    /// `str`
    Str,
    /// `byte`
    Byte,
    /// `void`
    Void,
}

#[cfg(test)]
mod tests {
    use super::*;

    // a distinct span per call, so a wrong span() arm or a clobbered field is
    // caught: span(n) and span(m) never compare equal for n != m.
    fn span(n: usize) -> Span {
        Span::new(n, n + 1)
    }

    #[test]
    fn nodes_are_debug_clone_and_partial_eq() {
        // build a small expression tree, clone it, assert equal; build a
        // different one, assert not equal. the precedence tests in the parser
        // depend on PartialEq, so it must be derived on every node.
        let a = Expr::Binary {
            op: BinOp::Add,
            lhs: Box::new(Expr::Int {
                value: 1,
                span: span(0),
            }),
            rhs: Box::new(Expr::Int {
                value: 2,
                span: span(2),
            }),
            span: span(0),
        };
        let b = a.clone();
        assert_eq!(a, b);
        // Debug must be derived too (assertion messages and parser tests use it).
        let _ = format!("{a:?}");
        let c = Expr::Binary {
            op: BinOp::Mul, // different operator
            lhs: Box::new(Expr::Int {
                value: 1,
                span: span(0),
            }),
            rhs: Box::new(Expr::Int {
                value: 2,
                span: span(2),
            }),
            span: span(0),
        };
        assert_ne!(a, c);
        // a float literal node round-trips through clone -- this is why the
        // node enums are PartialEq but not Eq.
        let f = Expr::Float {
            value: 1.5,
            span: span(5),
        };
        assert_eq!(f.clone(), f);
    }

    #[test]
    fn expr_span_returns_the_stored_span_for_a_sample_of_variants() {
        let cases: Vec<(Expr, Span)> = vec![
            (
                Expr::Int {
                    value: 0,
                    span: span(1),
                },
                span(1),
            ),
            (
                Expr::Float {
                    value: 0.0,
                    span: span(2),
                },
                span(2),
            ),
            (
                Expr::Byte {
                    value: 0,
                    span: span(3),
                },
                span(3),
            ),
            (
                Expr::Str {
                    value: String::new(),
                    span: span(4),
                },
                span(4),
            ),
            (
                Expr::Bool {
                    value: true,
                    span: span(5),
                },
                span(5),
            ),
            (
                Expr::Ident {
                    name: "x".to_string(),
                    span: span(6),
                },
                span(6),
            ),
            (
                Expr::Unary {
                    op: UnaryOp::Neg,
                    operand: Box::new(Expr::Int {
                        value: 1,
                        span: span(8),
                    }),
                    span: span(7),
                },
                span(7),
            ),
            (
                Expr::MethodCall {
                    receiver: Box::new(Expr::Ident {
                        name: "f".to_string(),
                        span: span(10),
                    }),
                    name: "read_all".to_string(),
                    args: vec![],
                    span: span(9),
                },
                span(9),
            ),
            (
                Expr::Pipeline {
                    lhs: Box::new(Expr::Int {
                        value: 5,
                        span: span(12),
                    }),
                    call: Box::new(Expr::Ident {
                        name: "double".to_string(),
                        span: span(14),
                    }),
                    span: span(11),
                },
                span(11),
            ),
            (
                Expr::OrElse {
                    expr: Box::new(Expr::Ident {
                        name: "a".to_string(),
                        span: span(16),
                    }),
                    fallback: Box::new(Expr::Str {
                        value: "no data".to_string(),
                        span: span(18),
                    }),
                    span: span(15),
                },
                span(15),
            ),
            (
                Expr::Interpolation {
                    parts: vec![
                        InterpPart::Literal("fib(".to_string()),
                        InterpPart::Expr(Expr::Ident {
                            name: "i".to_string(),
                            span: span(20),
                        }),
                        InterpPart::Literal(")".to_string()),
                    ],
                    span: span(19),
                },
                span(19),
            ),
        ];
        for (expr, expected) in cases {
            assert_eq!(expr.span(), expected, "span() mismatch for {expr:?}");
        }
    }

    #[test]
    fn stmt_span_returns_the_stored_span_for_a_sample_of_variants() {
        let cases: Vec<(Stmt, Span)> = vec![
            (
                Stmt::Let {
                    is_mut: false,
                    name: "x".to_string(),
                    ty: None,
                    init: Expr::Int {
                        value: 1,
                        span: span(1),
                    },
                    span: span(0),
                },
                span(0),
            ),
            (Stmt::Break { span: span(2) }, span(2)),
            (Stmt::Continue { span: span(3) }, span(3)),
            (
                Stmt::Return {
                    value: None,
                    span: span(4),
                },
                span(4),
            ),
            (
                Stmt::For {
                    var: "i".to_string(),
                    iter: Expr::Range {
                        start: Some(Box::new(Expr::Int {
                            value: 0,
                            span: span(6),
                        })),
                        end: Some(Box::new(Expr::Int {
                            value: 15,
                            span: span(8),
                        })),
                        inclusive: false,
                        span: span(6),
                    },
                    body: Block {
                        stmts: vec![],
                        value: None,
                        span: span(9),
                    },
                    span: span(5),
                },
                span(5),
            ),
        ];
        for (stmt, expected) in cases {
            assert_eq!(stmt.span(), expected, "span() mismatch for {stmt:?}");
        }
    }

    #[test]
    fn item_span_delegates_to_the_wrapped_decl() {
        let f = Item::Fn(FnDecl {
            type_name: None,
            name: "main".to_string(),
            params: vec![],
            ret_ty: None,
            effect: Some(Effect::Io),
            body: Block {
                stmts: vec![],
                value: None,
                span: span(1),
            },
            span: span(0),
        });
        assert_eq!(f.span(), span(0));
        let s = Item::Struct(StructDecl {
            name: "S".to_string(),
            fields: vec![],
            span: span(2),
        });
        assert_eq!(s.span(), span(2));
        let e = Item::Enum(EnumDecl {
            name: "E".to_string(),
            variants: vec![],
            span: span(3),
        });
        assert_eq!(e.span(), span(3));
        let i = Item::Interface(InterfaceDecl {
            name: "I".to_string(),
            methods: vec![],
            span: span(4),
        });
        assert_eq!(i.span(), span(4));
    }

    #[test]
    fn pattern_span_returns_the_stored_span_for_every_variant() {
        let cases: Vec<(Pattern, Span)> = vec![
            (
                Pattern::Variant {
                    name: "Circle".to_string(),
                    sub: vec![Pattern::Binding {
                        name: "r".to_string(),
                        span: span(1),
                    }],
                    span: span(0),
                },
                span(0),
            ),
            (Pattern::Wildcard { span: span(2) }, span(2)),
            (
                Pattern::Binding {
                    name: "v".to_string(),
                    span: span(3),
                },
                span(3),
            ),
            (
                Pattern::Int {
                    value: 0,
                    span: span(4),
                },
                span(4),
            ),
            (
                Pattern::Float {
                    value: 0.0,
                    span: span(5),
                },
                span(5),
            ),
            (
                Pattern::Byte {
                    value: 0,
                    span: span(6),
                },
                span(6),
            ),
            (
                Pattern::Str {
                    value: String::new(),
                    span: span(7),
                },
                span(7),
            ),
            (
                Pattern::Bool {
                    value: false,
                    span: span(8),
                },
                span(8),
            ),
        ];
        for (pat, expected) in cases {
            assert_eq!(pat.span(), expected, "span() mismatch for {pat:?}");
        }
    }

    #[test]
    fn type_expr_span_returns_the_stored_span_for_every_variant() {
        let cases: Vec<(TypeExpr, Span)> = vec![
            (
                TypeExpr::Primitive {
                    kind: PrimType::I64,
                    span: span(1),
                },
                span(1),
            ),
            (
                TypeExpr::Named {
                    name: "Shape".to_string(),
                    span: span(2),
                },
                span(2),
            ),
            (
                TypeExpr::Array {
                    elem: Box::new(TypeExpr::Primitive {
                        kind: PrimType::I64,
                        span: span(4),
                    }),
                    size: 5,
                    span: span(3),
                },
                span(3),
            ),
            (
                TypeExpr::DynArray {
                    elem: Box::new(TypeExpr::Primitive {
                        kind: PrimType::I64,
                        span: span(6),
                    }),
                    span: span(5),
                },
                span(5),
            ),
            (
                TypeExpr::Tuple {
                    elems: vec![
                        TypeExpr::Primitive {
                            kind: PrimType::I64,
                            span: span(8),
                        },
                        TypeExpr::Primitive {
                            kind: PrimType::Bool,
                            span: span(9),
                        },
                    ],
                    span: span(7),
                },
                span(7),
            ),
            (
                TypeExpr::Fn {
                    params: vec![TypeExpr::Primitive {
                        kind: PrimType::I64,
                        span: span(11),
                    }],
                    ret: Box::new(TypeExpr::Primitive {
                        kind: PrimType::I64,
                        span: span(12),
                    }),
                    span: span(10),
                },
                span(10),
            ),
            (
                TypeExpr::Generic {
                    name: "Result".to_string(),
                    args: vec![
                        TypeExpr::Primitive {
                            kind: PrimType::Str,
                            span: span(14),
                        },
                        TypeExpr::Primitive {
                            kind: PrimType::Str,
                            span: span(15),
                        },
                    ],
                    span: span(13),
                },
                span(13),
            ),
        ];
        for (ty, expected) in cases {
            assert_eq!(ty.span(), expected, "span() mismatch for {ty:?}");
        }
    }

    #[test]
    fn a_match_arm_holds_a_pattern_an_optional_guard_and_a_body() {
        // an expression-body arm with a guard: `v if v > 0 => "positive"`.
        let arm = MatchArm {
            pattern: Pattern::Binding {
                name: "v".to_string(),
                span: span(0),
            },
            guard: Some(Expr::Binary {
                op: BinOp::Gt,
                lhs: Box::new(Expr::Ident {
                    name: "v".to_string(),
                    span: span(2),
                }),
                rhs: Box::new(Expr::Int {
                    value: 0,
                    span: span(4),
                }),
                span: span(2),
            }),
            body: MatchArmBody::Expr(Box::new(Expr::Str {
                value: "positive".to_string(),
                span: span(6),
            })),
            span: span(0),
        };
        assert!(arm.guard.is_some());
        // a block-body arm with no guard.
        let arm2 = MatchArm {
            pattern: Pattern::Wildcard { span: span(10) },
            guard: None,
            body: MatchArmBody::Block(Block {
                stmts: vec![],
                value: None,
                span: span(12),
            }),
            span: span(10),
        };
        assert!(arm2.guard.is_none());
        assert_ne!(arm, arm2);
    }

    #[test]
    fn a_block_separates_statements_from_an_optional_trailing_value() {
        // `{ let x = 1; x }` -- one statement, a trailing-value expression.
        let block = Block {
            stmts: vec![Stmt::Let {
                is_mut: false,
                name: "x".to_string(),
                ty: None,
                init: Expr::Int {
                    value: 1,
                    span: span(1),
                },
                span: span(0),
            }],
            value: Some(Box::new(Expr::Ident {
                name: "x".to_string(),
                span: span(3),
            })),
            span: span(0),
        };
        assert_eq!(block.stmts.len(), 1);
        assert!(block.value.is_some());
        // an empty `{ }` block: no statements, value is void (None).
        let empty = Block {
            stmts: vec![],
            value: None,
            span: span(5),
        };
        assert!(empty.stmts.is_empty());
        assert!(empty.value.is_none());
    }
}
