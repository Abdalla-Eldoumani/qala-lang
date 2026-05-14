//! the type and effect checker: [`check_program`] turns the parser's untyped
//! AST + the source text into a typed AST plus a vector of errors and a vector
//! of warnings. it does NOT abort on the first error -- it accumulates and
//! recovers locally via [`QalaType::Unknown`] poison-propagation, so a user's
//! typo doesn't cascade to twenty errors.
//!
//! two top-level passes plus a third effect-fixed-point pass:
//! - pass 1 walks every top-level item building a [`SymbolTable`] (structs,
//!   enums, interfaces, free fns, methods).
//! - pass 2 walks every function body checking expressions and producing typed
//!   nodes.
//! - pass 3 iterates the call graph until every function's inferred
//!   [`EffectSet`] stabilises (proven to converge in <= 3 monotonic-ascent
//!   rounds on a 4-level lattice), then validates annotated functions against
//!   their annotation.
//!
//! determinism: errors and warnings are sorted by `(span.start, span.len)`
//! before returning; user-visible lists (missing variants, missing methods)
//! are sorted alphabetically via [`BTreeSet`]; the stdlib signatures table
//! is a [`Vec`], not a [`HashMap`], so iteration is source-order. no panics on
//! a malformed but well-formed-by-the-parser AST -- this discipline is
//! load-bearing for Phase 6 where the typechecker runs in WASM.

use crate::ast;
use crate::effects::EffectSet;
use crate::errors::QalaError;
#[allow(unused_imports)] // task 2 uses this in the depth-cap check.
use crate::parser::MAX_DEPTH;
use crate::span::{LineIndex, Span};
use crate::types::{QalaType, Symbol};
use crate::typed_ast;
use std::collections::{BTreeSet, HashMap};

// ---- public surface --------------------------------------------------------

/// one warning produced by the type checker.
///
/// derives `Debug, Clone, PartialEq`. categories are the locked snake_case
/// identifiers also accepted by the `// qala: allow(...)` directive scanner:
/// `unused_var`, `unreachable_code`, `unmatched_defer`, `shadowed_var`,
/// `redundant_annotation`, `overlapping_guards`. errors are never silenced;
/// warnings are silenceable by the directive table built once in
/// [`scan_allow_directives`].
#[derive(Debug, Clone, PartialEq)]
pub struct QalaWarning {
    /// one of the locked snake_case category strings.
    pub category: String,
    /// the human-readable one-line message.
    pub message: String,
    /// the source span the warning points at.
    pub span: Span,
    /// optional extra information, like a `shadowed_var`'s prior-binding
    /// location ("the prior binding is at line {l}:{c}").
    pub note: Option<String>,
}

// ---- entry point -----------------------------------------------------------

/// check a parsed program against the type and effect rules, producing a
/// fully-typed AST plus accumulated errors and warnings.
///
/// the typechecker does NOT fail-fast: it recovers locally via
/// [`QalaType::Unknown`] poison-propagation and keeps going so a typo at the
/// top of a function does not cascade to a wall of follow-on errors. errors
/// and warnings are sorted by `(span.start, span.len)` before returning so
/// downstream renderers can rely on byte-deterministic order.
pub fn check_program(
    ast: &ast::Ast,
    src: &str,
) -> (typed_ast::TypedAst, Vec<QalaError>, Vec<QalaWarning>) {
    let mut c = Checker::new(src);
    // pass 1a: pre-register every top-level type name so forward and mutual
    // references (`struct A { x: B } struct B { y: A }`) resolve correctly
    // when pass 1b walks field-type expressions. without this stage,
    // resolve_type_expr would emit a spurious `UnknownType` for the
    // forward-referenced name.
    c.preregister_type_names(ast);
    // pass 1b: collect every top-level declaration into the symbol table.
    for item in ast {
        c.collect_item(item);
    }
    // detect any recursive-struct cycles before pass 2 runs.
    c.detect_recursive_structs();
    // pass 2 placeholder: build a TypedAst by translating each item. tasks
    // 2-5 fill in the real check_item that walks bodies and types
    // expressions; for task 1 this returns a minimal-but-valid typed shape.
    let typed: typed_ast::TypedAst = ast.iter().map(|item| c.check_item(item)).collect();
    // pass 3 placeholder: task 3 implements the effect fixed-point.
    c.resolve_function_effects();
    // deterministic sort by (span.start, span.len).
    c.errors.sort_by_key(|e| (e.span().start, e.span().len));
    c.warnings.sort_by_key(|w| (w.span.start, w.span.len));
    (typed, c.errors, c.warnings)
}

// ---- symbol table ----------------------------------------------------------

/// the top-level declarations the typechecker collected during pass 1.
///
/// declaration order is preserved separately in [`Checker::struct_decl_order`]
/// for deterministic cycle reporting; iteration of these `HashMap`s is never
/// user-visible.
#[derive(Default)]
struct SymbolTable {
    /// every `struct Decl { ... }` keyed by name.
    structs: HashMap<String, StructInfo>,
    /// every `enum Decl { ... }` keyed by name.
    enums: HashMap<String, EnumInfo>,
    /// every `interface Decl { ... }` keyed by name.
    interfaces: HashMap<String, InterfaceInfo>,
    /// every `fn` or `fn Type.method` declaration keyed by [`FnKey`].
    fns: HashMap<FnKey, FnInfo>,
}

/// a resolved struct declaration. fields are populated in pass 1 and
/// consumed by the bidirectional checker in task 2 plus the
/// recursive-struct detector below.
#[allow(dead_code)]
struct StructInfo {
    /// the fields in declaration order: name + resolved type.
    fields: Vec<(String, QalaType)>,
    /// the declaration's source span.
    span: Span,
}

/// a resolved enum declaration. fields are populated in pass 1 and consumed
/// by the exhaustiveness checker in task 3.
#[allow(dead_code)]
struct EnumInfo {
    /// the variants in declaration order: name + resolved field types.
    variants: Vec<(String, Vec<QalaType>)>,
    /// the declaration's source span.
    span: Span,
}

/// a resolved interface declaration. fields are populated in pass 1 and
/// consumed by the structural-satisfaction checker in task 3.
#[allow(dead_code)]
struct InterfaceInfo {
    /// the required method signatures in declaration order.
    methods: Vec<MethodSigInfo>,
    /// the declaration's source span.
    span: Span,
}

/// a resolved interface-method signature. fields are populated in pass 1
/// and consumed by the structural-satisfaction checker in task 3.
#[allow(dead_code)]
struct MethodSigInfo {
    /// the method name.
    name: String,
    /// the parameter types in declaration order (including `self` typed as
    /// the receiver if present).
    params: Vec<QalaType>,
    /// the return type.
    ret_ty: QalaType,
    /// the annotated effect, or `None` for unannotated.
    effect: Option<EffectSet>,
    /// the signature's source span.
    span: Span,
}

/// the key under which a `fn` or `fn Type.method` is stored.
///
/// `(None, name)` is a free function; `(Some(T), name)` is the `fn T.method`
/// form. derives `Eq + Hash` so it can be a `HashMap` key.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct FnKey {
    /// `Some(T)` for `fn T.method`; `None` for a free function.
    type_name: Option<String>,
    /// the function or method name.
    name: String,
}

/// a resolved function declaration. fields are populated in pass 1 and
/// consumed by the bidirectional checker (task 2), the effect fixed-point
/// (task 3), and the structural-satisfaction checker (task 3).
#[allow(dead_code)]
struct FnInfo {
    /// the parameter list: name + resolved type + an unevaluated default if
    /// the parser captured one (the typechecker does not yet evaluate
    /// defaults; codegen does).
    params: Vec<(String, QalaType, Option<ast::Expr>)>,
    /// the return type. `void` is the resolved-form of an omitted `-> T`.
    ret_ty: QalaType,
    /// the annotated effect, or `None` for unannotated.
    annotated_effect: Option<EffectSet>,
    /// the inferred effect set, filled in by pass 3
    /// ([`Checker::resolve_function_effects`]).
    inferred_effect: Option<EffectSet>,
    /// the declaration's source span.
    span: Span,
    /// true for `fn T.method` declarations.
    is_method: bool,
}

// ---- local-scope info ------------------------------------------------------

/// metadata for one local binding (a `let` or function parameter) currently
/// in scope.
#[allow(dead_code)]
struct LocalInfo {
    /// the resolved type of this binding.
    ty: QalaType,
    /// the binding's source span (the `name` token).
    span: Span,
    /// whether the binding was declared `let mut`.
    is_mut: bool,
    /// whether the binding has been read at least once. tasks 4-5 use this
    /// to fire the `unused_var` warning.
    used: bool,
    /// whether this is a function parameter (exempt from `unused_var` per
    /// open question 4).
    is_param: bool,
}

// ---- function context ------------------------------------------------------

/// per-function state the typechecker maintains during pass 2.
#[allow(dead_code)]
struct FnCtx {
    /// the function name.
    name: String,
    /// `Some(T)` for `fn T.method`; `None` for a free function.
    type_name: Option<String>,
    /// the resolved return type.
    ret_ty: QalaType,
    /// the annotated effect, or `None` for unannotated.
    annotated_effect: Option<EffectSet>,
    /// the intrinsic effect set of the body (the union of every non-call
    /// statement's intrinsic effect). pass 3 unions this with the inferred
    /// effects of every called function.
    body_intrinsic: EffectSet,
    /// the keys of every function called from this body. pass 3 walks these
    /// during the fixed-point iteration.
    called_fns: Vec<FnKey>,
    /// every call site of this function that may need an effect-violation
    /// check after pass 3 settles. populated when the caller has an
    /// annotated effect.
    callsites_to_check: Vec<EffectViolationCandidate>,
}

/// a recorded call site that may produce an [`QalaError::EffectViolation`]
/// after pass 3 resolves every function's inferred effect.
#[allow(dead_code)]
struct EffectViolationCandidate {
    /// the caller's [`FnKey`].
    caller_key: FnKey,
    /// the callee's [`FnKey`].
    callee_key: FnKey,
    /// the source span of the call expression.
    call_span: Span,
}

// ---- checker state ---------------------------------------------------------

/// the type checker's working state. one is constructed per
/// [`check_program`] call.
#[allow(dead_code)]
struct Checker<'src> {
    /// the source text, kept by reference for the directive scanner and the
    /// line-index lookup. the typechecker never copies it.
    src: &'src str,
    /// the line-start table for converting byte offsets to line / column.
    line_index: LineIndex,
    /// the top-level declarations collected in pass 1.
    symbols: SymbolTable,
    /// struct names in declaration order, for deterministic SCC traversal.
    /// also doubles as the duplicate-detection set for structs (the symbols
    /// map is pre-populated with placeholders by `preregister_type_names`).
    struct_decl_order: Vec<String>,
    /// enum names that `collect_enum` has already processed; the symbols map
    /// is pre-populated with placeholders, so a contains-check there cannot
    /// distinguish first-vs-duplicate.
    collected_enum_names: Vec<String>,
    /// interface names that `collect_interface` has already processed.
    collected_interface_names: Vec<String>,
    /// the accumulated errors. sorted by `(span.start, span.len)` before
    /// returning.
    errors: Vec<QalaError>,
    /// the accumulated warnings. sorted by `(span.start, span.len)` before
    /// returning.
    warnings: Vec<QalaWarning>,
    /// the `// qala: allow(...)` directive table built once at the start of
    /// [`check_program`]. tasks 4-5 use it to silence warnings per-line.
    allow: HashMap<usize, BTreeSet<String>>,
    /// the active lexical scopes. each scope is a name -> local-info map;
    /// the topmost is the innermost.
    scopes: Vec<HashMap<String, LocalInfo>>,
    /// the active function context, or `None` outside a function body.
    fn_ctx: Option<FnCtx>,
    /// recursive-depth counter for [`Checker::infer_expr`]. mirrors the
    /// parser's [`MAX_DEPTH`] guard.
    depth: u32,
    /// per-function record (intrinsic body effect, called fns, candidate
    /// call sites). populated during pass 2's `check_fn_decl`; consumed by
    /// pass 3's `resolve_function_effects` in task 3.
    body_records: HashMap<FnKey, BodyEffectRecord>,
}

impl<'src> Checker<'src> {
    /// construct a fresh checker for the given source text.
    fn new(src: &'src str) -> Self {
        Checker {
            src,
            line_index: LineIndex::new(src),
            symbols: SymbolTable::default(),
            struct_decl_order: Vec::new(),
            collected_enum_names: Vec::new(),
            collected_interface_names: Vec::new(),
            errors: Vec::new(),
            warnings: Vec::new(),
            // task 5 replaces this empty map with the real scanner output.
            allow: HashMap::new(),
            scopes: Vec::new(),
            fn_ctx: None,
            depth: 0,
            body_records: HashMap::new(),
        }
    }
}

// ---- pass 1: declaration collection ----------------------------------------

impl<'src> Checker<'src> {
    /// register every top-level type name (struct / enum / interface) with
    /// an empty placeholder entry so forward and mutual references in field
    /// type positions resolve without emitting a spurious `UnknownType`.
    ///
    /// the placeholder is replaced by the real info in `collect_*`. duplicate
    /// names are NOT diagnosed here; `collect_*` does that against the
    /// already-populated map.
    fn preregister_type_names(&mut self, ast: &ast::Ast) {
        for item in ast {
            match item {
                ast::Item::Struct(d) => {
                    self.symbols
                        .structs
                        .entry(d.name.clone())
                        .or_insert(StructInfo {
                            fields: Vec::new(),
                            span: d.span,
                        });
                }
                ast::Item::Enum(d) => {
                    self.symbols
                        .enums
                        .entry(d.name.clone())
                        .or_insert(EnumInfo {
                            variants: Vec::new(),
                            span: d.span,
                        });
                }
                ast::Item::Interface(d) => {
                    self.symbols
                        .interfaces
                        .entry(d.name.clone())
                        .or_insert(InterfaceInfo {
                            methods: Vec::new(),
                            span: d.span,
                        });
                }
                ast::Item::Fn(_) => {}
            }
        }
    }

    /// dispatch on an item kind and call the matching collector.
    fn collect_item(&mut self, item: &ast::Item) {
        match item {
            ast::Item::Fn(d) => self.collect_fn(d),
            ast::Item::Struct(d) => self.collect_struct(d),
            ast::Item::Enum(d) => self.collect_enum(d),
            ast::Item::Interface(d) => self.collect_interface(d),
        }
    }

    /// collect a struct declaration into the symbol table.
    ///
    /// resolves each field's `TypeExpr` to a [`QalaType`]; emits an
    /// `UnknownType` per unresolvable field type so pass 2 can keep going
    /// with `Unknown` in the unresolved slot. emits a generic `Type` error
    /// on a duplicate struct name (the first declaration wins, so the
    /// placeholder kept the first decl's span and the second triggers the
    /// duplicate-check below). detection uses `struct_decl_order` because
    /// `preregister_type_names` already populated the symbols map with
    /// placeholders.
    fn collect_struct(&mut self, decl: &ast::StructDecl) {
        if self.struct_decl_order.contains(&decl.name) {
            self.errors.push(QalaError::Type {
                span: decl.span,
                message: format!("duplicate struct definition `{}`", decl.name),
            });
            return;
        }
        let mut fields: Vec<(String, QalaType)> = Vec::with_capacity(decl.fields.len());
        for f in &decl.fields {
            let ty = self.resolve_type_expr(&f.ty);
            fields.push((f.name.clone(), ty));
        }
        self.symbols
            .structs
            .insert(decl.name.clone(), StructInfo { fields, span: decl.span });
        self.struct_decl_order.push(decl.name.clone());
    }

    /// collect an enum declaration into the symbol table.
    fn collect_enum(&mut self, decl: &ast::EnumDecl) {
        if self.collected_enum_names.contains(&decl.name) {
            self.errors.push(QalaError::Type {
                span: decl.span,
                message: format!("duplicate enum definition `{}`", decl.name),
            });
            return;
        }
        self.collected_enum_names.push(decl.name.clone());
        let mut variants: Vec<(String, Vec<QalaType>)> = Vec::with_capacity(decl.variants.len());
        for v in &decl.variants {
            let fields: Vec<QalaType> = v.fields.iter().map(|t| self.resolve_type_expr(t)).collect();
            variants.push((v.name.clone(), fields));
        }
        self.symbols
            .enums
            .insert(decl.name.clone(), EnumInfo { variants, span: decl.span });
    }

    /// collect an interface declaration into the symbol table.
    fn collect_interface(&mut self, decl: &ast::InterfaceDecl) {
        if self.collected_interface_names.contains(&decl.name) {
            self.errors.push(QalaError::Type {
                span: decl.span,
                message: format!("duplicate interface definition `{}`", decl.name),
            });
            return;
        }
        self.collected_interface_names.push(decl.name.clone());
        let methods: Vec<MethodSigInfo> = decl
            .methods
            .iter()
            .map(|m| {
                // a method signature's params include `self` if present; we
                // resolve self to the receiver -- but at the interface
                // declaration site we don't yet know the receiver type, so
                // we treat `self` as `QalaType::Unknown` here. structural
                // satisfaction (task 3) ignores the self slot when matching
                // against an impl's `fn Type.method`.
                let params: Vec<QalaType> = m
                    .params
                    .iter()
                    .map(|p| {
                        if p.is_self {
                            QalaType::Unknown
                        } else if let Some(ty) = &p.ty {
                            self.resolve_type_expr(ty)
                        } else {
                            QalaType::Unknown
                        }
                    })
                    .collect();
                let ret_ty = m
                    .ret_ty
                    .as_ref()
                    .map(|t| self.resolve_type_expr(t))
                    .unwrap_or(QalaType::Void);
                let effect = m.effect.as_ref().map(effect_set_from_ast);
                MethodSigInfo {
                    name: m.name.clone(),
                    params,
                    ret_ty,
                    effect,
                    span: m.span,
                }
            })
            .collect();
        self.symbols
            .interfaces
            .insert(decl.name.clone(), InterfaceInfo { methods, span: decl.span });
    }

    /// collect a function (free function or `fn T.method`) into the symbol
    /// table.
    fn collect_fn(&mut self, decl: &ast::FnDecl) {
        let key = FnKey {
            type_name: decl.type_name.clone(),
            name: decl.name.clone(),
        };
        if self.symbols.fns.contains_key(&key) {
            // first declaration wins; second is the duplicate.
            let label = match &decl.type_name {
                Some(t) => format!("{}.{}", t, decl.name),
                None => decl.name.clone(),
            };
            self.errors.push(QalaError::Type {
                span: decl.span,
                message: format!("duplicate function definition `{label}`"),
            });
            return;
        }
        let mut params: Vec<(String, QalaType, Option<ast::Expr>)> = Vec::with_capacity(decl.params.len());
        for p in &decl.params {
            let ty = if p.is_self {
                // self's type is the receiver type, when present.
                match &decl.type_name {
                    Some(t) => QalaType::Named(Symbol(t.clone())),
                    None => QalaType::Unknown,
                }
            } else if let Some(t) = &p.ty {
                self.resolve_type_expr(t)
            } else {
                // a non-self param without a type is malformed; the parser
                // should reject this, but recovery is `Unknown`.
                QalaType::Unknown
            };
            params.push((p.name.clone(), ty, p.default.clone()));
        }
        let ret_ty = decl
            .ret_ty
            .as_ref()
            .map(|t| self.resolve_type_expr(t))
            .unwrap_or(QalaType::Void);
        let annotated_effect = decl.effect.as_ref().map(effect_set_from_ast);
        let is_method = decl.type_name.is_some();
        self.symbols.fns.insert(
            key,
            FnInfo {
                params,
                ret_ty,
                annotated_effect,
                inferred_effect: None,
                span: decl.span,
                is_method,
            },
        );
    }

    /// resolve a `TypeExpr` to a [`QalaType`], emitting `UnknownType` on a
    /// name that does not match a primitive, a known struct / enum /
    /// interface, the built-in `FileHandle`, or a known built-in generic.
    fn resolve_type_expr(&mut self, ty: &ast::TypeExpr) -> QalaType {
        match ty {
            ast::TypeExpr::Primitive { kind, .. } => QalaType::from_prim_type(kind),
            ast::TypeExpr::Named { name, span } => {
                // Result / Option without generic args is an explicit error
                // -- the parser allows the bare name here, but the type
                // system requires the arguments.
                if name == "Result" {
                    self.errors.push(QalaError::Type {
                        span: *span,
                        message: "Result requires two type arguments".to_string(),
                    });
                    return QalaType::Unknown;
                }
                if name == "Option" {
                    self.errors.push(QalaError::Type {
                        span: *span,
                        message: "Option requires one type argument".to_string(),
                    });
                    return QalaType::Unknown;
                }
                if name == "FileHandle" {
                    return QalaType::FileHandle;
                }
                if self.symbols.structs.contains_key(name)
                    || self.symbols.enums.contains_key(name)
                    || self.symbols.interfaces.contains_key(name)
                {
                    return QalaType::Named(Symbol(name.clone()));
                }
                self.errors.push(QalaError::UnknownType {
                    span: *span,
                    name: name.clone(),
                });
                QalaType::Unknown
            }
            ast::TypeExpr::Array { elem, size, .. } => {
                QalaType::Array(Box::new(self.resolve_type_expr(elem)), Some(*size as usize))
            }
            ast::TypeExpr::DynArray { elem, .. } => {
                QalaType::Array(Box::new(self.resolve_type_expr(elem)), None)
            }
            ast::TypeExpr::Tuple { elems, .. } => {
                QalaType::Tuple(elems.iter().map(|e| self.resolve_type_expr(e)).collect())
            }
            ast::TypeExpr::Fn { params, ret, .. } => QalaType::Function {
                params: params.iter().map(|p| self.resolve_type_expr(p)).collect(),
                returns: Box::new(self.resolve_type_expr(ret)),
            },
            ast::TypeExpr::Generic { name, args, span } => {
                if name == "Result" {
                    if args.len() != 2 {
                        self.errors.push(QalaError::Type {
                            span: *span,
                            message: "Result requires two type arguments".to_string(),
                        });
                        return QalaType::Unknown;
                    }
                    let ok = self.resolve_type_expr(&args[0]);
                    let err = self.resolve_type_expr(&args[1]);
                    return QalaType::Result(Box::new(ok), Box::new(err));
                }
                if name == "Option" {
                    if args.len() != 1 {
                        self.errors.push(QalaError::Type {
                            span: *span,
                            message: "Option requires one type argument".to_string(),
                        });
                        return QalaType::Unknown;
                    }
                    let inner = self.resolve_type_expr(&args[0]);
                    return QalaType::Option(Box::new(inner));
                }
                self.errors.push(QalaError::Type {
                    span: *span,
                    message: format!("unknown generic type `{name}`"),
                });
                QalaType::Unknown
            }
        }
    }
}

/// bridge from the parser's `Effect` (records *which keyword appeared*) to an
/// [`EffectSet`] (the bitfield the checker actually reasons about).
fn effect_set_from_ast(e: &ast::Effect) -> EffectSet {
    match e {
        ast::Effect::Pure => EffectSet::pure(),
        ast::Effect::Io => EffectSet::io(),
        ast::Effect::Alloc => EffectSet::alloc(),
        ast::Effect::Panic => EffectSet::panic(),
    }
}

// ---- recursive struct detection -------------------------------------------

impl<'src> Checker<'src> {
    /// detect every cycle in the by-value struct graph and emit one
    /// [`QalaError::RecursiveStructByValue`] per cycle.
    ///
    /// implementation: Tarjan's strongly-connected-components algorithm on the
    /// adjacency built from each struct's by-value field-type targets. an SCC
    /// of size > 1 or a singleton with a self-edge is a cycle. the cycle's
    /// lexicographically-smallest struct name is the head; the path walks the
    /// cycle starting and ending at the head so the message reads as a closed
    /// loop when joined with arrows (`A -> B -> A`).
    ///
    /// "by value" excludes `Option<_>`, `Result<_, _>`, dynamic arrays (`[T]`,
    /// stored Some-with-no-length), and zero-size fixed arrays. fixed arrays
    /// of size > 0 and tuples DO carry their element types by value, so they
    /// contribute edges.
    fn detect_recursive_structs(&mut self) {
        // alphabetical traversal order makes the chosen head deterministic
        // when multiple structs participate in a single SCC.
        let mut order = self.struct_decl_order.clone();
        order.sort();

        // adjacency: struct name -> sorted list of by-value struct neighbours.
        let mut adj: HashMap<String, Vec<String>> = HashMap::new();
        for name in &order {
            let mut targets: BTreeSet<String> = BTreeSet::new();
            if let Some(info) = self.symbols.structs.get(name) {
                for (_, ty) in &info.fields {
                    collect_by_value_targets(ty, &mut targets);
                }
            }
            let neighbours: Vec<String> = targets
                .into_iter()
                .filter(|n| self.symbols.structs.contains_key(n))
                .collect();
            adj.insert(name.clone(), neighbours);
        }

        // Tarjan SCC.
        let sccs = tarjan_scc(&order, &adj);

        for scc in sccs {
            // a singleton SCC is a cycle iff there is a self-edge.
            let is_cycle = scc.len() > 1
                || (scc.len() == 1
                    && adj
                        .get(&scc[0])
                        .map(|v| v.contains(&scc[0]))
                        .unwrap_or(false));
            if !is_cycle {
                continue;
            }
            // the head is the lexicographically smallest name in the SCC.
            let head = scc
                .iter()
                .min()
                .cloned()
                .unwrap_or_else(|| scc[0].clone());
            // build the path by DFS from head along the SCC's adjacency
            // until we return to head. for a self-loop the path is `[head,
            // head]`.
            let scc_set: BTreeSet<String> = scc.iter().cloned().collect();
            let mut path: Vec<String> = vec![head.clone()];
            let mut current = head.clone();
            // bounded by scc.len() + 1 -- a cycle of N nodes returns to head
            // in N steps.
            for _ in 0..=scc.len() {
                let next: Option<String> = adj
                    .get(&current)
                    .and_then(|nbrs| {
                        nbrs.iter()
                            .filter(|n| scc_set.contains(*n))
                            .min()
                            .cloned()
                    });
                let Some(next_name) = next else {
                    break;
                };
                path.push(next_name.clone());
                if next_name == head {
                    break;
                }
                current = next_name;
            }
            // span: the head struct's declaration site.
            let span = self
                .symbols
                .structs
                .get(&head)
                .map(|s| s.span)
                .unwrap_or(Span::new(0, 0));
            self.errors
                .push(QalaError::RecursiveStructByValue { span, path });
        }
    }
}

/// collect the names of every "by value" struct reachable from a type, for the
/// recursive-struct-by-value detector. tuples and fixed arrays of nonzero size
/// pass their element type through unchanged; dynamic arrays, `Option`,
/// `Result`, function types, zero-sized fixed arrays, and primitives are all
/// "by reference" (or carry no struct names) and contribute nothing.
fn collect_by_value_targets(ty: &QalaType, out: &mut BTreeSet<String>) {
    match ty {
        QalaType::Named(Symbol(name)) => {
            out.insert(name.clone());
        }
        QalaType::Tuple(elems) => {
            for e in elems {
                collect_by_value_targets(e, out);
            }
        }
        QalaType::Array(inner, Some(n)) if *n > 0 => {
            collect_by_value_targets(inner, out);
        }
        // dynamic arrays, zero-size fixed arrays, function types, Option,
        // Result, FileHandle, primitives, Unknown -- none of these carry the
        // contained type by value at the struct-layout level.
        _ => {}
    }
}

/// Tarjan's strongly-connected-components algorithm, iterative form.
///
/// returns the list of SCCs in the order Tarjan emits them (reverse
/// topological -- leaves first). each SCC is a `Vec<String>` of node names.
/// the caller iterates the SCCs in the order returned; for cycle reporting
/// the per-SCC head is chosen as the lexicographically smallest name, so the
/// emit order is by Tarjan's discovery, not by name.
///
/// the adjacency lists are expected to be sorted (callers iterate them in
/// determined order). nodes is the universe in alphabetical order, so the
/// traversal is reproducible across runs.
fn tarjan_scc(nodes: &[String], adj: &HashMap<String, Vec<String>>) -> Vec<Vec<String>> {
    // node -> Tarjan index, lowlink, on-stack flag.
    let mut index_of: HashMap<String, u32> = HashMap::new();
    let mut lowlink: HashMap<String, u32> = HashMap::new();
    let mut on_stack: HashMap<String, bool> = HashMap::new();
    let mut stack: Vec<String> = Vec::new();
    let mut next_index: u32 = 0;
    let mut sccs: Vec<Vec<String>> = Vec::new();

    // iterative DFS: each Frame tracks where we are in the children list.
    struct Frame {
        node: String,
        next_child: usize,
    }

    for root in nodes {
        if index_of.contains_key(root) {
            continue;
        }
        // begin a DFS rooted at `root`.
        index_of.insert(root.clone(), next_index);
        lowlink.insert(root.clone(), next_index);
        next_index += 1;
        stack.push(root.clone());
        on_stack.insert(root.clone(), true);

        let mut call_stack: Vec<Frame> = vec![Frame {
            node: root.clone(),
            next_child: 0,
        }];

        while let Some(frame) = call_stack.last_mut() {
            let v = frame.node.clone();
            let empty: Vec<String> = Vec::new();
            let children = adj.get(&v).unwrap_or(&empty);
            if frame.next_child < children.len() {
                let w = children[frame.next_child].clone();
                frame.next_child += 1;
                if !index_of.contains_key(&w) {
                    // recurse into w.
                    index_of.insert(w.clone(), next_index);
                    lowlink.insert(w.clone(), next_index);
                    next_index += 1;
                    stack.push(w.clone());
                    on_stack.insert(w.clone(), true);
                    call_stack.push(Frame { node: w, next_child: 0 });
                } else if *on_stack.get(&w).unwrap_or(&false) {
                    // back-edge: update v's lowlink.
                    let w_idx = *index_of.get(&w).unwrap_or(&0);
                    let v_low = *lowlink.get(&v).unwrap_or(&0);
                    lowlink.insert(v.clone(), v_low.min(w_idx));
                }
            } else {
                // finished v. if v is a root of an SCC, pop one.
                let v_idx = *index_of.get(&v).unwrap_or(&0);
                let v_low = *lowlink.get(&v).unwrap_or(&0);
                if v_low == v_idx {
                    let mut component: Vec<String> = Vec::new();
                    loop {
                        let w = stack.pop().unwrap_or_default();
                        on_stack.insert(w.clone(), false);
                        let is_v = w == v;
                        component.push(w);
                        if is_v {
                            break;
                        }
                    }
                    sccs.push(component);
                }
                call_stack.pop();
                // propagate lowlink to the parent (the new top of call_stack).
                if let Some(parent_frame) = call_stack.last() {
                    let p = parent_frame.node.clone();
                    let p_low = *lowlink.get(&p).unwrap_or(&0);
                    let v_low = *lowlink.get(&v).unwrap_or(&0);
                    lowlink.insert(p, p_low.min(v_low));
                }
            }
        }
    }
    sccs
}

// ---- pass 2: bidirectional type checking -----------------------------------

impl<'src> Checker<'src> {
    /// translate an untyped item into a typed one, fully checking bodies.
    ///
    /// for an `Item::Fn`, this opens a fresh scope, binds each parameter as
    /// a `LocalInfo`, checks the body against the declared return type,
    /// closes the scope (firing any per-scope warnings), and constructs the
    /// `TypedFnDecl` with a placeholder effect that pass 3 finalises.
    /// for the data-only item kinds (struct / enum / interface), the typed
    /// shape is constructed directly from the resolved symbol-table entries.
    fn check_item(&mut self, item: &ast::Item) -> typed_ast::TypedItem {
        match item {
            ast::Item::Fn(d) => self.check_fn_decl(d),
            ast::Item::Struct(d) => typed_ast::TypedItem::Struct(typed_ast::TypedStructDecl {
                name: d.name.clone(),
                fields: d
                    .fields
                    .iter()
                    .map(|f| typed_ast::TypedField {
                        name: f.name.clone(),
                        ty: resolve_type_silent(&f.ty, &self.symbols),
                        span: f.span,
                    })
                    .collect(),
                span: d.span,
            }),
            ast::Item::Enum(d) => typed_ast::TypedItem::Enum(typed_ast::TypedEnumDecl {
                name: d.name.clone(),
                variants: d
                    .variants
                    .iter()
                    .map(|v| typed_ast::TypedVariant {
                        name: v.name.clone(),
                        fields: v
                            .fields
                            .iter()
                            .map(|t| resolve_type_silent(t, &self.symbols))
                            .collect(),
                        span: v.span,
                    })
                    .collect(),
                span: d.span,
            }),
            ast::Item::Interface(d) => {
                typed_ast::TypedItem::Interface(typed_ast::TypedInterfaceDecl {
                    name: d.name.clone(),
                    methods: d
                        .methods
                        .iter()
                        .map(|m| typed_ast::TypedMethodSig {
                            name: m.name.clone(),
                            params: m
                                .params
                                .iter()
                                .map(|p| typed_ast::TypedParam {
                                    is_self: p.is_self,
                                    name: p.name.clone(),
                                    ty: if p.is_self {
                                        QalaType::Unknown
                                    } else if let Some(t) = &p.ty {
                                        resolve_type_silent(t, &self.symbols)
                                    } else {
                                        QalaType::Unknown
                                    },
                                    default: None,
                                    span: p.span,
                                })
                                .collect(),
                            ret_ty: m
                                .ret_ty
                                .as_ref()
                                .map(|t| resolve_type_silent(t, &self.symbols))
                                .unwrap_or(QalaType::Void),
                            effect: m.effect.as_ref().map(effect_set_from_ast).unwrap_or(EffectSet::pure()),
                            span: m.span,
                        })
                        .collect(),
                    span: d.span,
                })
            }
        }
    }

    /// check a single fn declaration: open a scope, bind params, walk the
    /// body, close the scope, build the `TypedFnDecl`.
    fn check_fn_decl(&mut self, d: &ast::FnDecl) -> typed_ast::TypedItem {
        let ret_ty = d
            .ret_ty
            .as_ref()
            .map(|t| resolve_type_silent(t, &self.symbols))
            .unwrap_or(QalaType::Void);
        let annotated_effect = d.effect.as_ref().map(effect_set_from_ast);
        // typed-params: resolve each type and bind it as a LocalInfo with
        // is_param=true so the `unused_var` rule never fires on a param.
        let mut typed_params: Vec<typed_ast::TypedParam> = Vec::with_capacity(d.params.len());
        let mut param_locals: Vec<(String, LocalInfo)> = Vec::with_capacity(d.params.len());
        for p in &d.params {
            let ty = if p.is_self {
                match &d.type_name {
                    Some(t) => QalaType::Named(Symbol(t.clone())),
                    None => QalaType::Unknown,
                }
            } else if let Some(t) = &p.ty {
                resolve_type_silent(t, &self.symbols)
            } else {
                QalaType::Unknown
            };
            typed_params.push(typed_ast::TypedParam {
                is_self: p.is_self,
                name: p.name.clone(),
                ty: ty.clone(),
                default: None,
                span: p.span,
            });
            param_locals.push((
                p.name.clone(),
                LocalInfo {
                    ty,
                    span: p.span,
                    is_mut: false,
                    used: false,
                    is_param: true,
                },
            ));
        }
        // open the function-body scope, bind parameters.
        self.fn_ctx = Some(FnCtx {
            name: d.name.clone(),
            type_name: d.type_name.clone(),
            ret_ty: ret_ty.clone(),
            annotated_effect,
            body_intrinsic: EffectSet::pure(),
            called_fns: Vec::new(),
            callsites_to_check: Vec::new(),
        });
        let mut scope: HashMap<String, LocalInfo> = HashMap::new();
        for (name, info) in param_locals {
            scope.insert(name, info);
        }
        self.scopes.push(scope);

        // check the body against the declared return type.
        let body = self.check_block(&d.body, Some(&ret_ty));

        // missing-return check: a function declared to return a non-Void type
        // must produce a value (a trailing block expression or a return
        // statement on every path). v1's check is simple: if ret_ty != Void
        // and the body's trailing value is None AND the body does not end
        // with a Return statement, emit MissingReturn.
        if !ret_ty.types_match(&QalaType::Void) && body.value.is_none() {
            let ends_in_return = d.body.stmts.last().map(|s| matches!(s, ast::Stmt::Return { .. })).unwrap_or(false);
            // also accept an if-with-returns or a while-with-returns -- v1
            // does not do dataflow, so any non-Return tail with a Void
            // trailing value triggers the warning. picking up the trailing
            // brace as the span: the closing token is the body's span.end.
            if !ends_in_return {
                let span = Span::new(
                    d.body.span.end().saturating_sub(1),
                    1,
                );
                self.errors.push(QalaError::MissingReturn {
                    span,
                    fn_name: d.name.clone(),
                    expected: ret_ty.display(),
                });
            }
        }

        // pop the function scope; fire any unused_var warnings.
        if let Some(scope) = self.scopes.pop() {
            for (name, info) in scope {
                if !info.is_param
                    && !info.used
                    && !name.starts_with('_')
                {
                    let w = QalaWarning {
                        category: "unused_var".to_string(),
                        message: format!("unused variable `{name}`"),
                        span: info.span,
                        note: None,
                    };
                    self.emit_warning(w);
                }
            }
        }

        // pull the FnCtx; pass 3 will read body_intrinsic / called_fns /
        // callsites_to_check to settle effect inference.
        let ctx = self.fn_ctx.take();
        if let Some(ctx) = ctx {
            // record the per-fn body-effect record for pass 3.
            let key = FnKey {
                type_name: d.type_name.clone(),
                name: d.name.clone(),
            };
            self.body_records.insert(
                key,
                BodyEffectRecord {
                    intrinsic: ctx.body_intrinsic,
                    called: ctx.called_fns,
                    callsites_to_check: ctx.callsites_to_check,
                },
            );
        }

        typed_ast::TypedItem::Fn(typed_ast::TypedFnDecl {
            type_name: d.type_name.clone(),
            name: d.name.clone(),
            params: typed_params,
            ret_ty,
            effect: annotated_effect.unwrap_or(EffectSet::pure()),
            body,
            span: d.span,
        })
    }

    /// check a block: open a scope, walk each statement, then either check
    /// or infer the trailing value. fire unreachable-code warnings inside
    /// the loop; fire unused-var warnings when the scope is popped.
    fn check_block(&mut self, block: &ast::Block, expected: Option<&QalaType>) -> typed_ast::TypedBlock {
        // a top-level fn body's scope is already on the stack (with params);
        // for any nested block we push a fresh scope so name resolution and
        // unused_var fire correctly.
        let pushed_scope = !self.scopes.is_empty();
        if pushed_scope {
            self.scopes.push(HashMap::new());
        }
        let mut typed_stmts: Vec<typed_ast::TypedStmt> = Vec::with_capacity(block.stmts.len());
        let mut terminator: Option<Span> = None;
        let mut warned_unreachable = false;
        for stmt in &block.stmts {
            if terminator.is_some() && !warned_unreachable {
                let w = QalaWarning {
                    category: "unreachable_code".to_string(),
                    message: "unreachable statement after `return`, `break`, or `continue`"
                        .to_string(),
                    span: stmt.span(),
                    note: None,
                };
                self.emit_warning(w);
                warned_unreachable = true;
                // still walk the stmt so types resolve; the warning fires
                // exactly once per block.
            }
            let typed_stmt = self.check_stmt(stmt);
            let is_terminator = matches!(
                stmt,
                ast::Stmt::Return { .. } | ast::Stmt::Break { .. } | ast::Stmt::Continue { .. }
            );
            if is_terminator && terminator.is_none() {
                terminator = Some(stmt.span());
            }
            typed_stmts.push(typed_stmt);
        }
        let (typed_value, value_ty) = match &block.value {
            Some(v) => {
                let typed = match expected {
                    Some(ty) => self.check_expr(v, ty),
                    None => self.infer_expr(v),
                };
                let ty = typed.ty().clone();
                (Some(Box::new(typed)), ty)
            }
            None => (None, QalaType::Void),
        };
        // pop the nested scope, fire unused_var.
        if pushed_scope && let Some(scope) = self.scopes.pop() {
            for (name, info) in scope {
                if !info.is_param
                    && !info.used
                    && !name.starts_with('_')
                {
                    let w = QalaWarning {
                        category: "unused_var".to_string(),
                        message: format!("unused variable `{name}`"),
                        span: info.span,
                        note: None,
                    };
                    self.emit_warning(w);
                }
            }
        }
        typed_ast::TypedBlock {
            stmts: typed_stmts,
            value: typed_value,
            ty: value_ty,
            span: block.span,
        }
    }

    /// check a single statement.
    fn check_stmt(&mut self, stmt: &ast::Stmt) -> typed_ast::TypedStmt {
        match stmt {
            ast::Stmt::Let { is_mut, name, ty, init, span } => {
                let (typed_init, decl_ty) = match ty {
                    Some(t) => {
                        let expected = resolve_type_silent(t, &self.symbols);
                        let typed_init = self.check_expr(init, &expected);
                        // redundant_annotation: fire iff the inferred type
                        // strictly equals the declared type (not via the
                        // Unknown wildcard, which would let typos slip).
                        if types_strictly_equal(typed_init.ty(), &expected) {
                            let w = QalaWarning {
                                category: "redundant_annotation".to_string(),
                                message: format!(
                                    "redundant type annotation: `{}` matches the inferred type",
                                    expected.display()
                                ),
                                span: t.span(),
                                note: None,
                            };
                            self.emit_warning(w);
                        }
                        (typed_init, expected)
                    }
                    None => {
                        let typed_init = self.infer_expr(init);
                        let ty = typed_init.ty().clone();
                        (typed_init, ty)
                    }
                };
                // shadowed_var: if any outer scope (not the topmost) has a
                // binding with the same name, warn.
                let topmost = self.scopes.len().saturating_sub(1);
                let mut prior_span: Option<Span> = None;
                for (idx, scope) in self.scopes.iter().enumerate() {
                    if idx == topmost {
                        break;
                    }
                    if let Some(prior) = scope.get(name) {
                        prior_span = Some(prior.span);
                        break;
                    }
                }
                if let Some(prior) = prior_span {
                    let (l, c) = self.line_index.location(self.src, prior.start as usize);
                    let w = QalaWarning {
                        category: "shadowed_var".to_string(),
                        message: format!("`{name}` shadows a binding from an outer scope"),
                        span: *span,
                        note: Some(format!("the prior binding is at line {l}:{c}")),
                    };
                    self.emit_warning(w);
                }
                // bind in the topmost scope.
                if let Some(scope) = self.scopes.last_mut() {
                    scope.insert(
                        name.clone(),
                        LocalInfo {
                            ty: decl_ty.clone(),
                            span: *span,
                            is_mut: *is_mut,
                            used: false,
                            is_param: false,
                        },
                    );
                }
                typed_ast::TypedStmt::Let {
                    is_mut: *is_mut,
                    name: name.clone(),
                    ty: decl_ty,
                    init: typed_init,
                    span: *span,
                }
            }
            ast::Stmt::If { cond, then_block, else_branch, span } => {
                let typed_cond = self.check_expr(cond, &QalaType::Bool);
                let typed_then = self.check_block(then_block, None);
                let typed_else = match else_branch {
                    Some(ast::ElseBranch::Block(b)) => {
                        Some(typed_ast::TypedElseBranch::Block(self.check_block(b, None)))
                    }
                    Some(ast::ElseBranch::If(b)) => {
                        let inner = self.check_stmt(b);
                        Some(typed_ast::TypedElseBranch::If(Box::new(inner)))
                    }
                    None => None,
                };
                typed_ast::TypedStmt::If {
                    cond: typed_cond,
                    then_block: typed_then,
                    else_branch: typed_else,
                    span: *span,
                }
            }
            ast::Stmt::While { cond, body, span } => {
                let typed_cond = self.check_expr(cond, &QalaType::Bool);
                let typed_body = self.check_block(body, None);
                typed_ast::TypedStmt::While {
                    cond: typed_cond,
                    body: typed_body,
                    span: *span,
                }
            }
            ast::Stmt::For { var, iter, body, span } => {
                let typed_iter = self.infer_expr(iter);
                // element type: an Array's elem, or i64 for a Range
                // (Range expressions resolve their ty to Array(elem=I64, None)
                // per the convention chosen below in infer_expr).
                let var_ty = match typed_iter.ty() {
                    QalaType::Array(elem, _) => (**elem).clone(),
                    QalaType::Unknown => QalaType::Unknown,
                    _ => {
                        self.errors.push(QalaError::Type {
                            span: iter.span(),
                            message: "for loop expects an array or range".to_string(),
                        });
                        QalaType::Unknown
                    }
                };
                // open a one-binding scope holding the loop variable.
                let mut scope: HashMap<String, LocalInfo> = HashMap::new();
                scope.insert(
                    var.clone(),
                    LocalInfo {
                        ty: var_ty.clone(),
                        span: *span,
                        is_mut: false,
                        // the loop variable is treated like a parameter
                        // (open question 4 style) -- never fires unused_var.
                        used: false,
                        is_param: true,
                    },
                );
                self.scopes.push(scope);
                let typed_body = self.check_block(body, None);
                self.scopes.pop();
                typed_ast::TypedStmt::For {
                    var: var.clone(),
                    var_ty,
                    iter: typed_iter,
                    body: typed_body,
                    span: *span,
                }
            }
            ast::Stmt::Return { value, span } => {
                let typed_value = match value {
                    Some(v) => {
                        let expected = self.fn_ctx.as_ref().map(|c| c.ret_ty.clone());
                        let typed = match expected {
                            Some(ty) => self.check_expr(v, &ty),
                            None => self.infer_expr(v),
                        };
                        Some(typed)
                    }
                    None => {
                        // bare `return` is legal iff ret_ty is Void.
                        if let Some(ctx) = &self.fn_ctx
                            && !ctx.ret_ty.types_match(&QalaType::Void)
                        {
                            self.errors.push(QalaError::TypeMismatch {
                                span: *span,
                                expected: ctx.ret_ty.display(),
                                found: "void".to_string(),
                            });
                        }
                        None
                    }
                };
                typed_ast::TypedStmt::Return { value: typed_value, span: *span }
            }
            ast::Stmt::Break { span } => typed_ast::TypedStmt::Break { span: *span },
            ast::Stmt::Continue { span } => typed_ast::TypedStmt::Continue { span: *span },
            ast::Stmt::Defer { expr, span } => {
                let typed_expr = self.infer_expr(expr);
                typed_ast::TypedStmt::Defer { expr: typed_expr, span: *span }
            }
            ast::Stmt::Expr { expr, span } => {
                let typed_expr = self.infer_expr(expr);
                typed_ast::TypedStmt::Expr { expr: typed_expr, span: *span }
            }
        }
    }

    /// look a name up in the active scope chain (top -> outer); on a hit,
    /// mark the LocalInfo as used and return its type. on a miss in scopes,
    /// fall back to the user fn table; on a miss there, fall back to the
    /// stdlib signatures table. returns `None` if nothing matched.
    fn lookup(&mut self, name: &str) -> Option<QalaType> {
        for scope in self.scopes.iter_mut().rev() {
            if let Some(info) = scope.get_mut(name) {
                info.used = true;
                return Some(info.ty.clone());
            }
        }
        // user-defined free function?
        let key = FnKey { type_name: None, name: name.to_string() };
        if let Some(info) = self.symbols.fns.get(&key) {
            let params: Vec<QalaType> = info.params.iter().map(|(_, ty, _)| ty.clone()).collect();
            return Some(QalaType::Function {
                params,
                returns: Box::new(info.ret_ty.clone()),
            });
        }
        // stdlib?
        let stdlib = stdlib_signatures();
        if let Some(entry) = stdlib.iter().find(|e| e.type_name.is_none() && e.name == name) {
            return Some(QalaType::Function {
                params: entry.params.clone(),
                returns: Box::new(entry.ret_ty.clone()),
            });
        }
        None
    }

    /// infer the type of an expression, producing a typed node.
    ///
    /// every variant of `ast::Expr` is dispatched; type errors emit a
    /// `QalaError` and the result carries `QalaType::Unknown` so subsequent
    /// `types_match` calls don't cascade. recursion is capped at
    /// [`MAX_DEPTH`] -- past the cap, a `QalaError::Type` is emitted and a
    /// poison-typed `Int` placeholder returned so the walk unwinds cleanly.
    fn infer_expr(&mut self, expr: &ast::Expr) -> typed_ast::TypedExpr {
        self.depth = self.depth.saturating_add(1);
        if self.depth > MAX_DEPTH {
            self.errors.push(QalaError::Type {
                span: expr.span(),
                message: "expression nests too deeply".to_string(),
            });
            self.depth = self.depth.saturating_sub(1);
            return typed_ast::TypedExpr::Int {
                value: 0,
                ty: QalaType::Unknown,
                span: expr.span(),
            };
        }
        let out = self.infer_expr_inner(expr);
        self.depth = self.depth.saturating_sub(1);
        out
    }

    /// the body of [`Self::infer_expr`]; split so the depth guard is exactly
    /// once at the top.
    fn infer_expr_inner(&mut self, expr: &ast::Expr) -> typed_ast::TypedExpr {
        match expr {
            ast::Expr::Int { value, span } => typed_ast::TypedExpr::Int {
                value: *value,
                ty: QalaType::I64,
                span: *span,
            },
            ast::Expr::Float { value, span } => typed_ast::TypedExpr::Float {
                value: *value,
                ty: QalaType::F64,
                span: *span,
            },
            ast::Expr::Byte { value, span } => typed_ast::TypedExpr::Byte {
                value: *value,
                ty: QalaType::Byte,
                span: *span,
            },
            ast::Expr::Str { value, span } => typed_ast::TypedExpr::Str {
                value: value.clone(),
                ty: QalaType::Str,
                span: *span,
            },
            ast::Expr::Bool { value, span } => typed_ast::TypedExpr::Bool {
                value: *value,
                ty: QalaType::Bool,
                span: *span,
            },
            ast::Expr::Ident { name, span } => {
                // Some/None/Ok/Err names that appear bare (no call) are
                // also legal -- they look like idents to the parser. handle
                // None/Triangle-like constructors specially: lookup misses
                // are an UndefinedName unless the name is a known
                // zero-arg constructor.
                if let Some(ty) = self.lookup(name) {
                    return typed_ast::TypedExpr::Ident {
                        name: name.clone(),
                        ty,
                        span: *span,
                    };
                }
                // is this a zero-data enum variant from any known enum?
                let mut variant_match: Option<String> = None;
                for (enum_name, info) in &self.symbols.enums {
                    if info.variants.iter().any(|(vn, fields)| vn == name && fields.is_empty()) {
                        variant_match = Some(enum_name.clone());
                        break;
                    }
                }
                if let Some(enum_name) = variant_match {
                    return typed_ast::TypedExpr::Ident {
                        name: name.clone(),
                        ty: QalaType::Named(Symbol(enum_name)),
                        span: *span,
                    };
                }
                self.errors.push(QalaError::UndefinedName {
                    span: *span,
                    name: name.clone(),
                });
                typed_ast::TypedExpr::Ident {
                    name: name.clone(),
                    ty: QalaType::Unknown,
                    span: *span,
                }
            }
            ast::Expr::Paren { inner, span } => {
                let typed_inner = self.infer_expr(inner);
                let ty = typed_inner.ty().clone();
                typed_ast::TypedExpr::Paren {
                    inner: Box::new(typed_inner),
                    ty,
                    span: *span,
                }
            }
            ast::Expr::Tuple { elems, span } => {
                let typed_elems: Vec<typed_ast::TypedExpr> = elems
                    .iter()
                    .map(|e| self.infer_expr(e))
                    .collect();
                let ty = QalaType::Tuple(typed_elems.iter().map(|e| e.ty().clone()).collect());
                typed_ast::TypedExpr::Tuple { elems: typed_elems, ty, span: *span }
            }
            ast::Expr::ArrayLit { elems, span } => {
                if elems.is_empty() {
                    self.errors.push(QalaError::Type {
                        span: *span,
                        message: "cannot infer element type of empty array".to_string(),
                    });
                    return typed_ast::TypedExpr::ArrayLit {
                        elems: Vec::new(),
                        ty: QalaType::Array(Box::new(QalaType::Unknown), Some(0)),
                        span: *span,
                    };
                }
                let first = self.infer_expr(&elems[0]);
                let elem_ty = first.ty().clone();
                let mut typed_elems = vec![first];
                for e in &elems[1..] {
                    let typed = self.check_expr(e, &elem_ty);
                    typed_elems.push(typed);
                }
                let len = typed_elems.len();
                typed_ast::TypedExpr::ArrayLit {
                    elems: typed_elems,
                    ty: QalaType::Array(Box::new(elem_ty), Some(len)),
                    span: *span,
                }
            }
            ast::Expr::ArrayRepeat { value, count, span } => {
                let typed_value = self.infer_expr(value);
                let typed_count = self.check_expr(count, &QalaType::I64);
                let elem_ty = typed_value.ty().clone();
                typed_ast::TypedExpr::ArrayRepeat {
                    value: Box::new(typed_value),
                    count: Box::new(typed_count),
                    ty: QalaType::Array(Box::new(elem_ty), None),
                    span: *span,
                }
            }
            ast::Expr::StructLit { name, fields, span } => {
                // check_struct_lit handles the field-set comparison.
                self.check_struct_lit(name, fields, *span)
            }
            ast::Expr::FieldAccess { obj, name, span } => {
                let typed_obj = self.infer_expr(obj);
                let obj_ty = typed_obj.ty().clone();
                let field_ty = match &obj_ty {
                    QalaType::Named(Symbol(s)) => {
                        if let Some(info) = self.symbols.structs.get(s) {
                            match info.fields.iter().find(|(fn_, _)| fn_ == name) {
                                Some((_, t)) => t.clone(),
                                None => {
                                    self.errors.push(QalaError::Type {
                                        span: *span,
                                        message: format!(
                                            "no field `{name}` on type `{}`",
                                            obj_ty.display()
                                        ),
                                    });
                                    QalaType::Unknown
                                }
                            }
                        } else {
                            self.errors.push(QalaError::Type {
                                span: *span,
                                message: format!("no field `{name}` on type `{}`", obj_ty.display()),
                            });
                            QalaType::Unknown
                        }
                    }
                    QalaType::Unknown => QalaType::Unknown,
                    _ => {
                        self.errors.push(QalaError::Type {
                            span: *span,
                            message: format!("no field `{name}` on type `{}`", obj_ty.display()),
                        });
                        QalaType::Unknown
                    }
                };
                typed_ast::TypedExpr::FieldAccess {
                    obj: Box::new(typed_obj),
                    name: name.clone(),
                    ty: field_ty,
                    span: *span,
                }
            }
            ast::Expr::MethodCall { receiver, name, args, span } => {
                self.check_method_call(receiver, name, args, *span)
            }
            ast::Expr::Call { callee, args, span } => self.check_call(callee, args, *span),
            ast::Expr::Index { obj, index, span } => {
                let typed_obj = self.infer_expr(obj);
                let typed_index = self.check_expr(index, &QalaType::I64);
                let elem_ty = match typed_obj.ty() {
                    QalaType::Array(elem, _) => (**elem).clone(),
                    QalaType::Unknown => QalaType::Unknown,
                    other => {
                        self.errors.push(QalaError::Type {
                            span: *span,
                            message: format!("cannot index `{}`", other.display()),
                        });
                        QalaType::Unknown
                    }
                };
                typed_ast::TypedExpr::Index {
                    obj: Box::new(typed_obj),
                    index: Box::new(typed_index),
                    ty: elem_ty,
                    span: *span,
                }
            }
            ast::Expr::Try { expr: inner, span } => {
                let typed_inner = self.infer_expr(inner);
                let (success_ty, ok_for_fn) = match typed_inner.ty() {
                    QalaType::Result(ok, _) => ((**ok).clone(), true),
                    QalaType::Option(t) => ((**t).clone(), true),
                    QalaType::Unknown => (QalaType::Unknown, true),
                    _ => {
                        self.errors.push(QalaError::Type {
                            span: *span,
                            message: format!(
                                "`?` operand must be Result or Option, found `{}`",
                                typed_inner.ty().display()
                            ),
                        });
                        (QalaType::Unknown, false)
                    }
                };
                // also require the enclosing function's ret_ty to be a
                // Result/Option. emit one error per the locked policy.
                if ok_for_fn && let Some(ctx) = &self.fn_ctx {
                    let ok = matches!(
                        &ctx.ret_ty,
                        QalaType::Result(_, _) | QalaType::Option(_) | QalaType::Unknown
                    );
                    if !ok {
                        self.errors.push(QalaError::RedundantQuestionOperator {
                            span: *span,
                            message: format!(
                                "`?` outside a Result-returning or Option-returning function (return type is `{}`); change the return type to `Result<_, _>` or `Option<_>`",
                                ctx.ret_ty.display()
                            ),
                        });
                    }
                }
                typed_ast::TypedExpr::Try {
                    expr: Box::new(typed_inner),
                    ty: success_ty,
                    span: *span,
                }
            }
            ast::Expr::Unary { op, operand, span } => {
                use ast::UnaryOp;
                let typed_operand = self.infer_expr(operand);
                let op_ty = typed_operand.ty().clone();
                let ty = match op {
                    UnaryOp::Not => {
                        if !op_ty.types_match(&QalaType::Bool) {
                            self.errors.push(QalaError::TypeMismatch {
                                span: operand.span(),
                                expected: "bool".to_string(),
                                found: op_ty.display(),
                            });
                        }
                        QalaType::Bool
                    }
                    UnaryOp::Neg => {
                        if op_ty.types_match(&QalaType::I64) {
                            QalaType::I64
                        } else if op_ty.types_match(&QalaType::F64) {
                            QalaType::F64
                        } else if matches!(op_ty, QalaType::Unknown) {
                            QalaType::Unknown
                        } else {
                            self.errors.push(QalaError::TypeMismatch {
                                span: operand.span(),
                                expected: "i64 or f64".to_string(),
                                found: op_ty.display(),
                            });
                            QalaType::Unknown
                        }
                    }
                };
                typed_ast::TypedExpr::Unary {
                    op: op.clone(),
                    operand: Box::new(typed_operand),
                    ty,
                    span: *span,
                }
            }
            ast::Expr::Binary { op, lhs, rhs, span } => self.check_binary(op, lhs, rhs, *span),
            ast::Expr::Range { start, end, inclusive, span } => {
                let typed_start = start.as_ref().map(|e| self.check_expr(e, &QalaType::I64));
                let typed_end = end.as_ref().map(|e| self.check_expr(e, &QalaType::I64));
                // ranges iterate as `[i64]` so for-loops bind the element to i64.
                typed_ast::TypedExpr::Range {
                    start: typed_start.map(Box::new),
                    end: typed_end.map(Box::new),
                    inclusive: *inclusive,
                    ty: QalaType::Array(Box::new(QalaType::I64), None),
                    span: *span,
                }
            }
            ast::Expr::Pipeline { lhs, call, span } => {
                // type as if the lhs were prepended to the call's arg list.
                // implementation: infer lhs, then build a virtual call AST
                // node and recurse. but we want to keep the typed Pipeline
                // node faithful, so we manually re-implement the lookup.
                let typed_lhs = self.infer_expr(lhs);
                let lhs_ty = typed_lhs.ty().clone();
                // resolve the call target: either Call{callee: Ident, args}
                // or Ident (zero-extra-args).
                let (typed_call, result_ty) = self.check_pipeline_call(call, &lhs_ty);
                typed_ast::TypedExpr::Pipeline {
                    lhs: Box::new(typed_lhs),
                    call: Box::new(typed_call),
                    ty: result_ty,
                    span: *span,
                }
            }
            ast::Expr::Comptime { body, span } => {
                let typed_body = self.infer_expr(body);
                let ty = typed_body.ty().clone();
                typed_ast::TypedExpr::Comptime {
                    body: Box::new(typed_body),
                    ty,
                    span: *span,
                }
            }
            ast::Expr::Block { block, span } => {
                let typed_block = self.check_block(block, None);
                let ty = typed_block.ty.clone();
                typed_ast::TypedExpr::Block {
                    block: typed_block,
                    ty,
                    span: *span,
                }
            }
            ast::Expr::Match { scrutinee, arms, span } => {
                self.check_match(scrutinee, arms, *span)
            }
            ast::Expr::OrElse { expr: e, fallback, span } => {
                let typed_e = self.infer_expr(e);
                let success_ty = match typed_e.ty() {
                    QalaType::Result(ok, _) => (**ok).clone(),
                    QalaType::Option(t) => (**t).clone(),
                    QalaType::Unknown => QalaType::Unknown,
                    other => {
                        self.errors.push(QalaError::Type {
                            span: e.span(),
                            message: format!(
                                "`or` left operand must be Result or Option, found `{}`",
                                other.display()
                            ),
                        });
                        QalaType::Unknown
                    }
                };
                let typed_fallback = self.check_expr(fallback, &success_ty);
                typed_ast::TypedExpr::OrElse {
                    expr: Box::new(typed_e),
                    fallback: Box::new(typed_fallback),
                    ty: success_ty,
                    span: *span,
                }
            }
            ast::Expr::Interpolation { parts, span } => {
                let typed_parts: Vec<typed_ast::TypedInterpPart> = parts
                    .iter()
                    .map(|p| match p {
                        ast::InterpPart::Literal(s) => typed_ast::TypedInterpPart::Literal(s.clone()),
                        ast::InterpPart::Expr(e) => {
                            typed_ast::TypedInterpPart::Expr(self.infer_expr(e))
                        }
                    })
                    .collect();
                typed_ast::TypedExpr::Interpolation {
                    parts: typed_parts,
                    ty: QalaType::Str,
                    span: *span,
                }
            }
        }
    }

    /// check an expression against an expected type, emitting `TypeMismatch`
    /// when they disagree. unknown-poison is a wildcard, so a chain of
    /// expressions after the first error does not cascade.
    fn check_expr(&mut self, expr: &ast::Expr, expected: &QalaType) -> typed_ast::TypedExpr {
        let typed = self.infer_expr(expr);
        if !typed.ty().types_match(expected) {
            self.errors.push(QalaError::TypeMismatch {
                span: typed.span(),
                expected: expected.display(),
                found: typed.ty().display(),
            });
        }
        typed
    }

    /// type-check a struct literal: look up the struct, check declared and
    /// provided field sets match (missing fields named in one error, excess
    /// fields named individually), check each provided value against the
    /// declared field type.
    fn check_struct_lit(
        &mut self,
        name: &str,
        fields: &[ast::FieldInit],
        span: Span,
    ) -> typed_ast::TypedExpr {
        // capture declared fields as (name, type) so we can iterate without
        // re-borrowing self.
        let declared: Option<Vec<(String, QalaType)>> = self
            .symbols
            .structs
            .get(name)
            .map(|s| s.fields.clone());
        let Some(declared) = declared else {
            self.errors.push(QalaError::UndefinedName {
                span,
                name: name.to_string(),
            });
            return typed_ast::TypedExpr::StructLit {
                name: name.to_string(),
                fields: Vec::new(),
                ty: QalaType::Unknown,
                span,
            };
        };
        let mut typed_fields: Vec<typed_ast::TypedFieldInit> = Vec::new();
        // missing-field detection.
        let provided_names: BTreeSet<String> =
            fields.iter().map(|f| f.name.clone()).collect();
        let missing: Vec<String> = declared
            .iter()
            .filter(|(dn, _)| !provided_names.contains(dn))
            .map(|(dn, _)| dn.clone())
            .collect();
        if !missing.is_empty() {
            self.errors.push(QalaError::Type {
                span,
                message: format!(
                    "missing field(s) in struct literal `{name}`: {}",
                    missing.join(", ")
                ),
            });
        }
        for fi in fields {
            let decl_ty: Option<QalaType> = declared
                .iter()
                .find(|(dn, _)| dn == &fi.name)
                .map(|(_, t)| t.clone());
            match decl_ty {
                Some(t) => {
                    let typed_value = self.check_expr(&fi.value, &t);
                    typed_fields.push(typed_ast::TypedFieldInit {
                        name: fi.name.clone(),
                        value: typed_value,
                        span: fi.span,
                    });
                }
                None => {
                    self.errors.push(QalaError::Type {
                        span: fi.span,
                        message: format!(
                            "no field `{}` on struct `{name}`",
                            fi.name
                        ),
                    });
                    // still type the value so cascading errors don't fire.
                    let typed_value = self.infer_expr(&fi.value);
                    typed_fields.push(typed_ast::TypedFieldInit {
                        name: fi.name.clone(),
                        value: typed_value,
                        span: fi.span,
                    });
                }
            }
        }
        typed_ast::TypedExpr::StructLit {
            name: name.to_string(),
            fields: typed_fields,
            ty: QalaType::Named(Symbol(name.to_string())),
            span,
        }
    }

    /// type-check a method call. resolves the method via the symbol table's
    /// `FnKey { type_name: Some(receiver_type_name), name }` first, then
    /// falls back to the stdlib table (e.g. `FileHandle.read_all`).
    fn check_method_call(
        &mut self,
        receiver: &ast::Expr,
        name: &str,
        args: &[ast::Expr],
        span: Span,
    ) -> typed_ast::TypedExpr {
        let typed_receiver = self.infer_expr(receiver);
        let receiver_ty = typed_receiver.ty().clone();
        // resolve method signature: user method via FnKey, else stdlib.
        let resolved: Option<(Vec<QalaType>, QalaType, FnKey)> = (|| {
            // user fn T.method
            let type_name = match &receiver_ty {
                QalaType::Named(Symbol(s)) => Some(s.clone()),
                QalaType::FileHandle => Some("FileHandle".to_string()),
                _ => None,
            };
            if let Some(tn) = type_name {
                let key = FnKey { type_name: Some(tn.clone()), name: name.to_string() };
                if let Some(info) = self.symbols.fns.get(&key) {
                    // params after self.
                    let params: Vec<QalaType> = info
                        .params
                        .iter()
                        .skip(if info.params.first().map(|(n, _, _)| n == "self").unwrap_or(false) {
                            1
                        } else {
                            0
                        })
                        .map(|(_, t, _)| t.clone())
                        .collect();
                    return Some((params, info.ret_ty.clone(), key));
                }
                // stdlib?
                for entry in stdlib_signatures().iter() {
                    if entry.type_name.as_deref() == Some(tn.as_str()) && entry.name == name {
                        let params: Vec<QalaType> = entry.params.clone();
                        return Some((
                            params,
                            entry.ret_ty.clone(),
                            FnKey {
                                type_name: Some(tn.clone()),
                                name: name.to_string(),
                            },
                        ));
                    }
                }
            }
            None
        })();
        let (params, ret_ty) = match resolved {
            Some((p, r, key)) => {
                // record for pass 3's effect resolution.
                if let Some(ctx) = &mut self.fn_ctx {
                    ctx.called_fns.push(key.clone());
                    if ctx.annotated_effect.is_some() {
                        ctx.callsites_to_check.push(EffectViolationCandidate {
                            caller_key: FnKey {
                                type_name: ctx.type_name.clone(),
                                name: ctx.name.clone(),
                            },
                            callee_key: key,
                            call_span: span,
                        });
                    }
                }
                (p, r)
            }
            None => {
                if !matches!(receiver_ty, QalaType::Unknown) {
                    self.errors.push(QalaError::Type {
                        span,
                        message: format!(
                            "no method `{name}` on type `{}`",
                            receiver_ty.display()
                        ),
                    });
                }
                (Vec::new(), QalaType::Unknown)
            }
        };
        let typed_args: Vec<typed_ast::TypedExpr> = if params.is_empty() && args.is_empty() {
            Vec::new()
        } else {
            args.iter()
                .enumerate()
                .map(|(i, a)| {
                    if i < params.len() {
                        self.check_expr(a, &params[i])
                    } else {
                        self.infer_expr(a)
                    }
                })
                .collect()
        };
        if args.len() != params.len() && !matches!(receiver_ty, QalaType::Unknown) && !ret_ty.types_match(&QalaType::Unknown) {
            // arity mismatch -- emit a generic Type error.
            self.errors.push(QalaError::Type {
                span,
                message: format!(
                    "method `{name}` expects {} argument(s), found {}",
                    params.len(),
                    args.len()
                ),
            });
        }
        typed_ast::TypedExpr::MethodCall {
            receiver: Box::new(typed_receiver),
            name: name.to_string(),
            args: typed_args,
            ty: ret_ty,
            span,
        }
    }

    /// type-check a plain call expression `callee(args)`. handles built-in
    /// constructors `Ok`/`Err`/`Some`/`None`, user-defined free functions,
    /// stdlib free functions (including the generic ones), and identifier
    /// callees in general.
    fn check_call(
        &mut self,
        callee: &ast::Expr,
        args: &[ast::Expr],
        span: Span,
    ) -> typed_ast::TypedExpr {
        // special-case the built-in constructor names appearing in callee
        // position as a bare ident.
        if let ast::Expr::Ident { name, .. } = callee {
            if let Some(ctor_ty) = self.try_builtin_constructor(name, args, span) {
                let typed_callee = typed_ast::TypedExpr::Ident {
                    name: name.clone(),
                    ty: QalaType::Function {
                        params: Vec::new(),
                        returns: Box::new(ctor_ty.0.clone()),
                    },
                    span: callee.span(),
                };
                return typed_ast::TypedExpr::Call {
                    callee: Box::new(typed_callee),
                    args: ctor_ty.1,
                    ty: ctor_ty.0,
                    span,
                };
            }
            // user-defined free fn, or stdlib free fn (incl. virtual generics).
            if let Some((params, ret_ty, is_generic, name_clone, callee_key)) =
                self.resolve_free_callable(name)
            {
                // record for pass 3.
                if let Some(ctx) = &mut self.fn_ctx {
                    ctx.called_fns.push(callee_key.clone());
                    if ctx.annotated_effect.is_some() {
                        ctx.callsites_to_check.push(EffectViolationCandidate {
                            caller_key: FnKey {
                                type_name: ctx.type_name.clone(),
                                name: ctx.name.clone(),
                            },
                            callee_key,
                            call_span: span,
                        });
                    }
                }
                let typed_callee = typed_ast::TypedExpr::Ident {
                    name: name_clone.clone(),
                    ty: QalaType::Function {
                        params: params.clone(),
                        returns: Box::new(ret_ty.clone()),
                    },
                    span: callee.span(),
                };
                let typed_args = self.check_call_args(args, &params, is_generic, &name_clone, span);
                let result_ty = self.resolve_generic_return_ty(&name_clone, &ret_ty, &typed_args);
                return typed_ast::TypedExpr::Call {
                    callee: Box::new(typed_callee),
                    args: typed_args,
                    ty: result_ty,
                    span,
                };
            }
            // unresolved free name in callee position.
            self.errors.push(QalaError::UndefinedName {
                span: callee.span(),
                name: name.clone(),
            });
            let typed_callee = typed_ast::TypedExpr::Ident {
                name: name.clone(),
                ty: QalaType::Unknown,
                span: callee.span(),
            };
            let typed_args: Vec<typed_ast::TypedExpr> =
                args.iter().map(|a| self.infer_expr(a)).collect();
            return typed_ast::TypedExpr::Call {
                callee: Box::new(typed_callee),
                args: typed_args,
                ty: QalaType::Unknown,
                span,
            };
        }
        // non-identifier callee (rare in v1): infer it and treat its type as
        // a function shape.
        let typed_callee = self.infer_expr(callee);
        let (params, ret_ty) = match typed_callee.ty() {
            QalaType::Function { params, returns } => (params.clone(), (**returns).clone()),
            _ => {
                self.errors.push(QalaError::Type {
                    span,
                    message: format!(
                        "cannot call value of type `{}`",
                        typed_callee.ty().display()
                    ),
                });
                (Vec::new(), QalaType::Unknown)
            }
        };
        let typed_args: Vec<typed_ast::TypedExpr> = args
            .iter()
            .enumerate()
            .map(|(i, a)| {
                if i < params.len() {
                    self.check_expr(a, &params[i])
                } else {
                    self.infer_expr(a)
                }
            })
            .collect();
        typed_ast::TypedExpr::Call {
            callee: Box::new(typed_callee),
            args: typed_args,
            ty: ret_ty,
            span,
        }
    }

    /// helper: check a call's args against a parameter list. for generic
    /// virtual stdlib calls (`len<T>(coll: [T]) -> i64`) the typechecker
    /// accepts any matching concrete instantiation -- it infers each arg
    /// and applies a loose check.
    fn check_call_args(
        &mut self,
        args: &[ast::Expr],
        params: &[QalaType],
        is_generic: bool,
        name: &str,
        call_span: Span,
    ) -> Vec<typed_ast::TypedExpr> {
        if args.len() != params.len() {
            // emit one arity error, then infer args.
            // skip arity check when the generic flag is set AND the
            // signature was a placeholder (params.len() may be sentinel).
            if !is_generic {
                self.errors.push(QalaError::Type {
                    span: call_span,
                    message: format!(
                        "function `{name}` expects {} argument(s), found {}",
                        params.len(),
                        args.len()
                    ),
                });
            }
        }
        args.iter()
            .enumerate()
            .map(|(i, a)| {
                if !is_generic && i < params.len() {
                    self.check_expr(a, &params[i])
                } else {
                    self.infer_expr(a)
                }
            })
            .collect()
    }

    /// helper: compute the result type of a stdlib generic call once its
    /// args are typed. for non-generic functions, returns the declared
    /// `ret_ty` unchanged. handles `len`, `push`, `pop`, `type_of`, `map`,
    /// `filter`, `reduce`.
    fn resolve_generic_return_ty(
        &self,
        name: &str,
        ret_ty: &QalaType,
        typed_args: &[typed_ast::TypedExpr],
    ) -> QalaType {
        match name {
            "len" => QalaType::I64,
            "push" => QalaType::Void,
            "pop" => {
                // pop returns Option<T> where T is the element type of arg 0.
                if let Some(arg0) = typed_args.first()
                    && let QalaType::Array(elem, _) = arg0.ty()
                {
                    return QalaType::Option(elem.clone());
                }
                QalaType::Option(Box::new(QalaType::Unknown))
            }
            "type_of" => QalaType::Str,
            "map" => {
                // map(coll: [T], f: fn(T) -> U) -> [U]
                let u = typed_args.get(1).and_then(|f| match f.ty() {
                    QalaType::Function { returns, .. } => Some((**returns).clone()),
                    _ => None,
                });
                QalaType::Array(Box::new(u.unwrap_or(QalaType::Unknown)), None)
            }
            "filter" => {
                // filter(coll: [T], pred: fn(T) -> bool) -> [T]
                if let Some(arg0) = typed_args.first() {
                    return arg0.ty().clone();
                }
                QalaType::Array(Box::new(QalaType::Unknown), None)
            }
            "reduce" => {
                // reduce(coll: [T], init: U, f: fn(U, T) -> U) -> U
                if let Some(arg1) = typed_args.get(1) {
                    return arg1.ty().clone();
                }
                QalaType::Unknown
            }
            _ => ret_ty.clone(),
        }
    }

    /// helper: resolve a callable name (user fn, then stdlib). returns
    /// `(param-types, ret-ty, is-generic, name, key)`.
    #[allow(clippy::type_complexity)]
    fn resolve_free_callable(
        &self,
        name: &str,
    ) -> Option<(Vec<QalaType>, QalaType, bool, String, FnKey)> {
        let key = FnKey { type_name: None, name: name.to_string() };
        if let Some(info) = self.symbols.fns.get(&key) {
            let params: Vec<QalaType> = info.params.iter().map(|(_, t, _)| t.clone()).collect();
            return Some((params, info.ret_ty.clone(), false, name.to_string(), key));
        }
        for entry in stdlib_signatures().iter() {
            if entry.type_name.is_none() && entry.name == name {
                return Some((
                    entry.params.clone(),
                    entry.ret_ty.clone(),
                    entry.is_generic,
                    name.to_string(),
                    FnKey { type_name: None, name: name.to_string() },
                ));
            }
        }
        None
    }

    /// type-check the built-in constructors Ok/Err/Some/None when seen in
    /// a call-position bare-ident shape. returns the (result-type, typed-args)
    /// tuple on a hit, None otherwise.
    fn try_builtin_constructor(
        &mut self,
        name: &str,
        args: &[ast::Expr],
        span: Span,
    ) -> Option<(QalaType, Vec<typed_ast::TypedExpr>)> {
        match name {
            "Ok" => {
                if args.len() != 1 {
                    self.errors.push(QalaError::Type {
                        span,
                        message: format!("`Ok` expects 1 argument, found {}", args.len()),
                    });
                    let typed_args: Vec<_> = args.iter().map(|a| self.infer_expr(a)).collect();
                    return Some((
                        QalaType::Result(Box::new(QalaType::Unknown), Box::new(QalaType::Unknown)),
                        typed_args,
                    ));
                }
                let typed_arg = self.infer_expr(&args[0]);
                let ok_ty = typed_arg.ty().clone();
                Some((
                    QalaType::Result(Box::new(ok_ty), Box::new(QalaType::Unknown)),
                    vec![typed_arg],
                ))
            }
            "Err" => {
                if args.len() != 1 {
                    self.errors.push(QalaError::Type {
                        span,
                        message: format!("`Err` expects 1 argument, found {}", args.len()),
                    });
                    let typed_args: Vec<_> = args.iter().map(|a| self.infer_expr(a)).collect();
                    return Some((
                        QalaType::Result(Box::new(QalaType::Unknown), Box::new(QalaType::Unknown)),
                        typed_args,
                    ));
                }
                let typed_arg = self.infer_expr(&args[0]);
                let err_ty = typed_arg.ty().clone();
                Some((
                    QalaType::Result(Box::new(QalaType::Unknown), Box::new(err_ty)),
                    vec![typed_arg],
                ))
            }
            "Some" => {
                if args.len() != 1 {
                    self.errors.push(QalaError::Type {
                        span,
                        message: format!("`Some` expects 1 argument, found {}", args.len()),
                    });
                    let typed_args: Vec<_> = args.iter().map(|a| self.infer_expr(a)).collect();
                    return Some((QalaType::Option(Box::new(QalaType::Unknown)), typed_args));
                }
                let typed_arg = self.infer_expr(&args[0]);
                let inner = typed_arg.ty().clone();
                Some((QalaType::Option(Box::new(inner)), vec![typed_arg]))
            }
            "None" => {
                if !args.is_empty() {
                    self.errors.push(QalaError::Type {
                        span,
                        message: format!("`None` expects 0 arguments, found {}", args.len()),
                    });
                }
                let typed_args: Vec<_> = args.iter().map(|a| self.infer_expr(a)).collect();
                Some((QalaType::Option(Box::new(QalaType::Unknown)), typed_args))
            }
            _ => None,
        }
    }

    /// type a binary expression. arithmetic ops produce the operand type
    /// (with `str + str = str` as the special-case); comparison and
    /// equality produce bool; logical ops require bool operands.
    fn check_binary(
        &mut self,
        op: &ast::BinOp,
        lhs: &ast::Expr,
        rhs: &ast::Expr,
        span: Span,
    ) -> typed_ast::TypedExpr {
        use ast::BinOp;
        let typed_lhs = self.infer_expr(lhs);
        let typed_rhs = self.infer_expr(rhs);
        let lty = typed_lhs.ty().clone();
        let rty = typed_rhs.ty().clone();
        let result_ty = match op {
            BinOp::Add => {
                // str + str = str (concatenation).
                if lty.types_match(&QalaType::Str) && rty.types_match(&QalaType::Str) {
                    QalaType::Str
                } else if lty.types_match(&rty)
                    && (lty.types_match(&QalaType::I64) || lty.types_match(&QalaType::F64))
                {
                    lty.clone()
                } else if matches!(lty, QalaType::Unknown) || matches!(rty, QalaType::Unknown) {
                    QalaType::Unknown
                } else {
                    self.errors.push(QalaError::TypeMismatch {
                        span,
                        expected: lty.display(),
                        found: rty.display(),
                    });
                    QalaType::Unknown
                }
            }
            BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Rem => {
                if lty.types_match(&rty)
                    && (lty.types_match(&QalaType::I64) || lty.types_match(&QalaType::F64))
                {
                    lty.clone()
                } else if matches!(lty, QalaType::Unknown) || matches!(rty, QalaType::Unknown) {
                    QalaType::Unknown
                } else {
                    self.errors.push(QalaError::TypeMismatch {
                        span,
                        expected: lty.display(),
                        found: rty.display(),
                    });
                    QalaType::Unknown
                }
            }
            BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => {
                if !lty.types_match(&rty) {
                    self.errors.push(QalaError::TypeMismatch {
                        span,
                        expected: lty.display(),
                        found: rty.display(),
                    });
                }
                QalaType::Bool
            }
            BinOp::Eq | BinOp::Ne => {
                if !lty.types_match(&rty) {
                    self.errors.push(QalaError::TypeMismatch {
                        span,
                        expected: lty.display(),
                        found: rty.display(),
                    });
                }
                QalaType::Bool
            }
            BinOp::And | BinOp::Or => {
                if !lty.types_match(&QalaType::Bool) {
                    self.errors.push(QalaError::TypeMismatch {
                        span: lhs.span(),
                        expected: "bool".to_string(),
                        found: lty.display(),
                    });
                }
                if !rty.types_match(&QalaType::Bool) {
                    self.errors.push(QalaError::TypeMismatch {
                        span: rhs.span(),
                        expected: "bool".to_string(),
                        found: rty.display(),
                    });
                }
                QalaType::Bool
            }
        };
        typed_ast::TypedExpr::Binary {
            op: op.clone(),
            lhs: Box::new(typed_lhs),
            rhs: Box::new(typed_rhs),
            ty: result_ty,
            span,
        }
    }

    /// resolve a pipeline `lhs |> call` -- the call is either an Ident
    /// (zero extra args) or a Call (1+ extra args). type as if `lhs` were
    /// the first arg of the resolved function. returns (typed call, result-ty).
    fn check_pipeline_call(
        &mut self,
        call: &ast::Expr,
        lhs_ty: &QalaType,
    ) -> (typed_ast::TypedExpr, QalaType) {
        // build the virtual argument list = [lhs_ty] + extra arg types,
        // and look up the call target's signature.
        match call {
            ast::Expr::Ident { name, span } => {
                // zero extra args: signature expects 1 param matching lhs_ty.
                if let Some((params, ret_ty, is_generic, name_clone, key)) =
                    self.resolve_free_callable(name)
                {
                    if let Some(ctx) = &mut self.fn_ctx {
                        ctx.called_fns.push(key.clone());
                        if ctx.annotated_effect.is_some() {
                            ctx.callsites_to_check.push(EffectViolationCandidate {
                                caller_key: FnKey {
                                    type_name: ctx.type_name.clone(),
                                    name: ctx.name.clone(),
                                },
                                callee_key: key,
                                call_span: *span,
                            });
                        }
                    }
                    // check param 0 against lhs_ty (when not generic).
                    if !is_generic
                        && !params.is_empty()
                        && !lhs_ty.types_match(&params[0])
                    {
                        self.errors.push(QalaError::TypeMismatch {
                            span: *span,
                            expected: params[0].display(),
                            found: lhs_ty.display(),
                        });
                    }
                    // result type: generic-aware.
                    let typed_call = typed_ast::TypedExpr::Ident {
                        name: name_clone.clone(),
                        ty: QalaType::Function {
                            params: params.clone(),
                            returns: Box::new(ret_ty.clone()),
                        },
                        span: *span,
                    };
                    return (typed_call, ret_ty);
                }
                // unresolved.
                self.errors.push(QalaError::UndefinedName {
                    span: *span,
                    name: name.clone(),
                });
                (
                    typed_ast::TypedExpr::Ident {
                        name: name.clone(),
                        ty: QalaType::Unknown,
                        span: *span,
                    },
                    QalaType::Unknown,
                )
            }
            ast::Expr::Call { callee, args, span } => {
                // delegate to the normal call handler, but prepend lhs
                // virtually. simplest implementation: infer the call as
                // written (the parser has already collected the actual
                // args), then re-check the first param against lhs_ty. for
                // a generic stdlib call (`map`, `filter`, ...), accept
                // whatever the resolver computed.
                let typed_call = self.check_call(callee, args, *span);
                // the type of the typed call is the result type of the
                // pipeline (lhs flows into the existing first param).
                let ret_ty = typed_call.ty().clone();
                (typed_call, ret_ty)
            }
            other => {
                // any other call shape is unsupported.
                let typed = self.infer_expr(other);
                let ty = typed.ty().clone();
                (typed, ty)
            }
        }
    }

    /// type-check a match expression.
    ///
    /// task 2 implements the basic typing rule: infer the scrutinee, type
    /// each arm body, require a common arm body type (Unknown counts as
    /// poison). exhaustiveness checking is task 3.
    fn check_match(
        &mut self,
        scrutinee: &ast::Expr,
        arms: &[ast::MatchArm],
        span: Span,
    ) -> typed_ast::TypedExpr {
        let typed_scrutinee = self.infer_expr(scrutinee);
        let scrut_ty = typed_scrutinee.ty().clone();
        let mut typed_arms: Vec<typed_ast::TypedMatchArm> = Vec::with_capacity(arms.len());
        let mut result_ty: Option<QalaType> = None;
        for arm in arms {
            // check the pattern in a fresh scope so any bindings it
            // introduces are local to the arm.
            self.scopes.push(HashMap::new());
            self.bind_pattern(&arm.pattern, &scrut_ty);
            let typed_guard = arm.guard.as_ref().map(|g| self.check_expr(g, &QalaType::Bool));
            let typed_body = match &arm.body {
                ast::MatchArmBody::Expr(e) => {
                    let typed = self.infer_expr(e);
                    typed_ast::TypedMatchArmBody::Expr(Box::new(typed))
                }
                ast::MatchArmBody::Block(b) => {
                    typed_ast::TypedMatchArmBody::Block(self.check_block(b, None))
                }
            };
            self.scopes.pop();
            // accumulate the common arm result type.
            let body_ty = match &typed_body {
                typed_ast::TypedMatchArmBody::Expr(e) => e.ty().clone(),
                typed_ast::TypedMatchArmBody::Block(b) => b.ty.clone(),
            };
            result_ty = match (result_ty, body_ty.clone()) {
                (None, b) => Some(b),
                (Some(a), b) if a.types_match(&b) => Some(a),
                (Some(_), _) => Some(QalaType::Unknown),
            };
            typed_arms.push(typed_ast::TypedMatchArm {
                pattern: arm.pattern.clone(),
                guard: typed_guard,
                body: typed_body,
                span: arm.span,
            });
        }
        // task 3 will add exhaustiveness checking here.
        let ty = result_ty.unwrap_or(QalaType::Unknown);
        typed_ast::TypedExpr::Match {
            scrutinee: Box::new(typed_scrutinee),
            arms: typed_arms,
            ty,
            span,
        }
    }

    /// bind any names introduced by a pattern into the topmost scope, with
    /// types derived from the scrutinee type.
    ///
    /// task 2's minimal implementation: for `Variant { sub }`, look up the
    /// variant in any enum and use its field types; for `Binding`, bind
    /// the scrutinee type; for literal / wildcard patterns, nothing.
    fn bind_pattern(&mut self, pattern: &ast::Pattern, scrut_ty: &QalaType) {
        match pattern {
            ast::Pattern::Variant { name, sub, span } => {
                // find the matching variant. for an enum scrutinee, look in
                // that enum specifically. otherwise look in any enum.
                let mut variant_fields: Option<Vec<QalaType>> = None;
                if let QalaType::Named(Symbol(enum_name)) = scrut_ty
                    && let Some(info) = self.symbols.enums.get(enum_name)
                {
                    for (vname, fields) in &info.variants {
                        if vname == name {
                            variant_fields = Some(fields.clone());
                            break;
                        }
                    }
                    if variant_fields.is_none() {
                        self.errors.push(QalaError::Type {
                            span: *span,
                            message: format!(
                                "variant `{name}` is not part of enum `{enum_name}`"
                            ),
                        });
                    }
                }
                if let Some(fields) = variant_fields {
                    for (i, sub_pat) in sub.iter().enumerate() {
                        let sub_ty = fields
                            .get(i)
                            .cloned()
                            .unwrap_or(QalaType::Unknown);
                        self.bind_pattern(sub_pat, &sub_ty);
                    }
                } else {
                    // recurse with Unknown for each sub-pattern.
                    for sub_pat in sub {
                        self.bind_pattern(sub_pat, &QalaType::Unknown);
                    }
                }
            }
            ast::Pattern::Binding { name, span } => {
                if let Some(scope) = self.scopes.last_mut() {
                    scope.insert(
                        name.clone(),
                        LocalInfo {
                            ty: scrut_ty.clone(),
                            span: *span,
                            is_mut: false,
                            // pattern bindings are scoped to the arm; treat
                            // them like loop variables (no unused_var).
                            used: false,
                            is_param: true,
                        },
                    );
                }
            }
            // literal / wildcard patterns introduce no bindings.
            _ => {}
        }
    }

    /// pass 3: resolve every function's inferred effect.
    ///
    /// task 1/2 leaves this as a no-op so the crate compiles end-to-end
    /// while task 3 fills in the fixed-point algorithm.
    fn resolve_function_effects(&mut self) {
        // no-op; task 3 fills this in.
    }

    /// emit a warning, honouring the directive table set by task 5. task 4
    /// installs the placeholder version of this method.
    fn emit_warning(&mut self, w: QalaWarning) {
        // task 5 wires `is_silenced` to the real `allow` table.
        let line = self.line_index.location(self.src, w.span.start as usize).0;
        if let Some(cats) = self.allow.get(&line)
            && cats.contains(&w.category)
        {
            return;
        }
        self.warnings.push(w);
    }
}

/// the per-function record pass 3 reads to settle effect inference.
#[allow(dead_code)]
struct BodyEffectRecord {
    /// the intrinsic effect of the body (the union of every non-call
    /// statement's intrinsic effect set).
    intrinsic: EffectSet,
    /// every function this body calls.
    called: Vec<FnKey>,
    /// call sites that need an effect-violation check after pass 3 settles.
    callsites_to_check: Vec<EffectViolationCandidate>,
}

/// strict type equality: a wildcard `Unknown` does NOT match anything. used
/// by the `redundant_annotation` warning so a typo's `Unknown` does not
/// fire the warning by accident.
fn types_strictly_equal(a: &QalaType, b: &QalaType) -> bool {
    if matches!(a, QalaType::Unknown) || matches!(b, QalaType::Unknown) {
        return false;
    }
    a == b
}

/// one entry in the stdlib signature table. task 3 fills in the actual
/// list; task 2 only needs the shape so dependent code compiles.
#[allow(dead_code)]
struct StdlibFn {
    /// `None` for a free function; `Some(T)` for `fn T.method` (e.g.
    /// `FileHandle.read_all`).
    type_name: Option<String>,
    /// the function name.
    name: String,
    /// parameter types (or sentinels for the virtual generics).
    params: Vec<QalaType>,
    /// return type.
    ret_ty: QalaType,
    /// the function's effect.
    effect: EffectSet,
    /// true for virtual-generic functions (`len`, `push`, `pop`, `type_of`,
    /// `map`, `filter`, `reduce`, `Ok`, `Err`, `Some`, `None`). the
    /// call-site checker accepts any concrete instantiation.
    is_generic: bool,
}

/// the hardcoded stdlib signature table. task 3 finalises this; task 2's
/// stub gives enough entries that pass 2 tests resolve the names they need
/// (println, print, parse_int-shaped placeholders are NOT in the stdlib;
/// the test fixtures define those as user functions when needed).
fn stdlib_signatures() -> Vec<StdlibFn> {
    vec![
        StdlibFn {
            type_name: None,
            name: "print".to_string(),
            params: vec![QalaType::Str],
            ret_ty: QalaType::Void,
            effect: EffectSet::io(),
            is_generic: false,
        },
        StdlibFn {
            type_name: None,
            name: "println".to_string(),
            params: vec![QalaType::Str],
            ret_ty: QalaType::Void,
            effect: EffectSet::io(),
            is_generic: false,
        },
        StdlibFn {
            type_name: None,
            name: "sqrt".to_string(),
            params: vec![QalaType::F64],
            ret_ty: QalaType::F64,
            effect: EffectSet::pure(),
            is_generic: false,
        },
        StdlibFn {
            type_name: None,
            name: "abs".to_string(),
            params: vec![QalaType::I64],
            ret_ty: QalaType::I64,
            effect: EffectSet::pure(),
            is_generic: false,
        },
        StdlibFn {
            type_name: None,
            name: "abs".to_string(),
            params: vec![QalaType::F64],
            ret_ty: QalaType::F64,
            effect: EffectSet::pure(),
            is_generic: false,
        },
        StdlibFn {
            type_name: None,
            name: "assert".to_string(),
            params: vec![QalaType::Bool],
            ret_ty: QalaType::Void,
            effect: EffectSet::panic(),
            is_generic: false,
        },
        StdlibFn {
            type_name: None,
            name: "len".to_string(),
            params: vec![QalaType::Array(Box::new(QalaType::Unknown), None)],
            ret_ty: QalaType::I64,
            effect: EffectSet::pure(),
            is_generic: true,
        },
        StdlibFn {
            type_name: None,
            name: "push".to_string(),
            params: vec![
                QalaType::Array(Box::new(QalaType::Unknown), None),
                QalaType::Unknown,
            ],
            ret_ty: QalaType::Void,
            effect: EffectSet::alloc(),
            is_generic: true,
        },
        StdlibFn {
            type_name: None,
            name: "pop".to_string(),
            params: vec![QalaType::Array(Box::new(QalaType::Unknown), None)],
            ret_ty: QalaType::Option(Box::new(QalaType::Unknown)),
            effect: EffectSet::alloc(),
            is_generic: true,
        },
        StdlibFn {
            type_name: None,
            name: "type_of".to_string(),
            params: vec![QalaType::Unknown],
            ret_ty: QalaType::Str,
            effect: EffectSet::pure(),
            is_generic: true,
        },
        StdlibFn {
            type_name: None,
            name: "open".to_string(),
            params: vec![QalaType::Str],
            ret_ty: QalaType::FileHandle,
            effect: EffectSet::io(),
            is_generic: false,
        },
        StdlibFn {
            type_name: None,
            name: "close".to_string(),
            params: vec![QalaType::FileHandle],
            ret_ty: QalaType::Void,
            effect: EffectSet::io(),
            is_generic: false,
        },
        StdlibFn {
            type_name: None,
            name: "map".to_string(),
            params: vec![
                QalaType::Array(Box::new(QalaType::Unknown), None),
                QalaType::Function {
                    params: vec![QalaType::Unknown],
                    returns: Box::new(QalaType::Unknown),
                },
            ],
            ret_ty: QalaType::Array(Box::new(QalaType::Unknown), None),
            effect: EffectSet::pure(),
            is_generic: true,
        },
        StdlibFn {
            type_name: None,
            name: "filter".to_string(),
            params: vec![
                QalaType::Array(Box::new(QalaType::Unknown), None),
                QalaType::Function {
                    params: vec![QalaType::Unknown],
                    returns: Box::new(QalaType::Bool),
                },
            ],
            ret_ty: QalaType::Array(Box::new(QalaType::Unknown), None),
            effect: EffectSet::pure(),
            is_generic: true,
        },
        StdlibFn {
            type_name: None,
            name: "reduce".to_string(),
            params: vec![
                QalaType::Array(Box::new(QalaType::Unknown), None),
                QalaType::Unknown,
                QalaType::Function {
                    params: vec![QalaType::Unknown, QalaType::Unknown],
                    returns: Box::new(QalaType::Unknown),
                },
            ],
            ret_ty: QalaType::Unknown,
            effect: EffectSet::pure(),
            is_generic: true,
        },
        // FileHandle.read_all -- the only stdlib method in v1.
        StdlibFn {
            type_name: Some("FileHandle".to_string()),
            name: "read_all".to_string(),
            params: vec![],
            ret_ty: QalaType::Result(Box::new(QalaType::Str), Box::new(QalaType::Str)),
            effect: EffectSet::io(),
            is_generic: false,
        },
    ]
}

/// resolve a `TypeExpr` to a [`QalaType`] without emitting errors. used by
/// [`Checker::check_item`] when re-materialising types onto the typed AST
/// from the already-populated symbol table; the collection pass already
/// surfaced any `UnknownType` faults, so a duplicate emit here would
/// double-count.
fn resolve_type_silent(ty: &ast::TypeExpr, symbols: &SymbolTable) -> QalaType {
    match ty {
        ast::TypeExpr::Primitive { kind, .. } => QalaType::from_prim_type(kind),
        ast::TypeExpr::Named { name, .. } => {
            if name == "FileHandle" {
                return QalaType::FileHandle;
            }
            if symbols.structs.contains_key(name)
                || symbols.enums.contains_key(name)
                || symbols.interfaces.contains_key(name)
            {
                return QalaType::Named(Symbol(name.clone()));
            }
            QalaType::Unknown
        }
        ast::TypeExpr::Array { elem, size, .. } => {
            QalaType::Array(Box::new(resolve_type_silent(elem, symbols)), Some(*size as usize))
        }
        ast::TypeExpr::DynArray { elem, .. } => {
            QalaType::Array(Box::new(resolve_type_silent(elem, symbols)), None)
        }
        ast::TypeExpr::Tuple { elems, .. } => {
            QalaType::Tuple(elems.iter().map(|e| resolve_type_silent(e, symbols)).collect())
        }
        ast::TypeExpr::Fn { params, ret, .. } => QalaType::Function {
            params: params.iter().map(|p| resolve_type_silent(p, symbols)).collect(),
            returns: Box::new(resolve_type_silent(ret, symbols)),
        },
        ast::TypeExpr::Generic { name, args, .. } => {
            if name == "Result" && args.len() == 2 {
                return QalaType::Result(
                    Box::new(resolve_type_silent(&args[0], symbols)),
                    Box::new(resolve_type_silent(&args[1], symbols)),
                );
            }
            if name == "Option" && args.len() == 1 {
                return QalaType::Option(Box::new(resolve_type_silent(&args[0], symbols)));
            }
            QalaType::Unknown
        }
    }
}

// ---- tests -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;
    use crate::parser::Parser;

    /// lex, parse, then typecheck. the common entry point for inline tests.
    fn check(src: &str) -> (typed_ast::TypedAst, Vec<QalaError>, Vec<QalaWarning>) {
        let tokens = Lexer::tokenize(src).expect("lex");
        let ast = Parser::parse(&tokens).expect("parse");
        check_program(&ast, src)
    }

    /// like [`check`] but panics on any error. handy for the "happy-path"
    /// tests where the expected outcome is "no errors".
    #[allow(dead_code)]
    fn check_ok(src: &str) -> typed_ast::TypedAst {
        let (typed, errors, _) = check(src);
        assert!(errors.is_empty(), "unexpected errors: {errors:?}");
        typed
    }

    #[test]
    fn collect_empty_program() {
        let (typed, errors, warnings) = check("");
        assert!(typed.is_empty());
        assert!(errors.is_empty());
        assert!(warnings.is_empty());
    }

    #[test]
    fn collect_single_fn() {
        let src = "fn main() is io { }";
        let (typed, errors, warnings) = check(src);
        assert!(errors.is_empty(), "{errors:?}");
        assert!(warnings.is_empty(), "{warnings:?}");
        assert_eq!(typed.len(), 1);
        match &typed[0] {
            typed_ast::TypedItem::Fn(f) => {
                assert_eq!(f.name, "main");
                assert_eq!(f.ret_ty, QalaType::Void);
                assert_eq!(f.effect, EffectSet::io());
            }
            _ => panic!("expected Fn, got {:?}", typed[0]),
        }
    }

    #[test]
    fn collect_struct_with_two_fields() {
        let src = "struct A { x: i64, y: bool }";
        let (typed, errors, _) = check(src);
        assert!(errors.is_empty(), "{errors:?}");
        assert_eq!(typed.len(), 1);
        match &typed[0] {
            typed_ast::TypedItem::Struct(s) => {
                assert_eq!(s.name, "A");
                assert_eq!(s.fields.len(), 2);
                assert_eq!(s.fields[0].name, "x");
                assert_eq!(s.fields[0].ty, QalaType::I64);
                assert_eq!(s.fields[1].name, "y");
                assert_eq!(s.fields[1].ty, QalaType::Bool);
            }
            _ => panic!("expected Struct"),
        }
    }

    #[test]
    fn collect_enum_with_three_variants() {
        let src = "enum Shape { Circle(f64), Rect(f64, f64), Triangle }";
        let (typed, errors, _) = check(src);
        assert!(errors.is_empty(), "{errors:?}");
        match &typed[0] {
            typed_ast::TypedItem::Enum(e) => {
                assert_eq!(e.name, "Shape");
                assert_eq!(e.variants.len(), 3);
                assert_eq!(e.variants[0].name, "Circle");
                assert_eq!(e.variants[0].fields, vec![QalaType::F64]);
                assert_eq!(e.variants[1].name, "Rect");
                assert_eq!(e.variants[1].fields, vec![QalaType::F64, QalaType::F64]);
                assert_eq!(e.variants[2].name, "Triangle");
                assert!(e.variants[2].fields.is_empty());
            }
            _ => panic!("expected Enum"),
        }
    }

    #[test]
    fn collect_interface_with_one_method() {
        let src = "interface Printable { fn to_string(self) -> str }";
        let (typed, errors, _) = check(src);
        assert!(errors.is_empty(), "{errors:?}");
        match &typed[0] {
            typed_ast::TypedItem::Interface(i) => {
                assert_eq!(i.name, "Printable");
                assert_eq!(i.methods.len(), 1);
                assert_eq!(i.methods[0].name, "to_string");
                assert_eq!(i.methods[0].ret_ty, QalaType::Str);
            }
            _ => panic!("expected Interface"),
        }
    }

    #[test]
    fn collect_struct_with_unknown_field_type() {
        let src = "struct A { x: Nope }";
        let (_, errors, _) = check(src);
        assert_eq!(errors.len(), 1);
        match &errors[0] {
            QalaError::UnknownType { name, .. } => {
                assert_eq!(name, "Nope");
            }
            other => panic!("expected UnknownType, got {other:?}"),
        }
    }

    #[test]
    fn collect_recursive_struct_self_loop() {
        let src = "struct A { x: A }";
        let (_, errors, _) = check(src);
        let cycle_errors: Vec<&QalaError> = errors
            .iter()
            .filter(|e| matches!(e, QalaError::RecursiveStructByValue { .. }))
            .collect();
        assert_eq!(cycle_errors.len(), 1, "expected exactly one cycle error: {errors:?}");
        match cycle_errors[0] {
            QalaError::RecursiveStructByValue { path, .. } => {
                assert_eq!(path, &vec!["A".to_string(), "A".to_string()]);
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn collect_recursive_struct_mutual() {
        let src = "struct A { x: B } struct B { y: A }";
        let (_, errors, _) = check(src);
        let cycle_errors: Vec<&QalaError> = errors
            .iter()
            .filter(|e| matches!(e, QalaError::RecursiveStructByValue { .. }))
            .collect();
        assert_eq!(cycle_errors.len(), 1, "expected one cycle: {errors:?}");
        match cycle_errors[0] {
            QalaError::RecursiveStructByValue { path, .. } => {
                // head is alphabetically smallest -> "A".
                assert_eq!(path, &vec!["A".to_string(), "B".to_string(), "A".to_string()]);
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn collect_dynamic_array_self_reference_is_not_a_cycle() {
        // `[A]` is by reference logically; no cycle.
        let src = "struct A { xs: [A] }";
        let (_, errors, _) = check(src);
        let cycle_errors: Vec<&QalaError> = errors
            .iter()
            .filter(|e| matches!(e, QalaError::RecursiveStructByValue { .. }))
            .collect();
        assert!(cycle_errors.is_empty(), "no cycle expected: {errors:?}");
    }

    #[test]
    fn collect_tuple_self_reference_is_a_cycle() {
        // `(A, i64)` carries A by value.
        let src = "struct A { x: (A, i64) }";
        let (_, errors, _) = check(src);
        let cycle_errors: Vec<&QalaError> = errors
            .iter()
            .filter(|e| matches!(e, QalaError::RecursiveStructByValue { .. }))
            .collect();
        assert_eq!(cycle_errors.len(), 1, "expected cycle: {errors:?}");
    }

    #[test]
    fn collect_fn_with_unknown_param_type() {
        let src = "fn f(x: Nope) -> i64 is pure { return 0 }";
        let (_, errors, _) = check(src);
        let unknown_errors: Vec<&QalaError> = errors
            .iter()
            .filter(|e| matches!(e, QalaError::UnknownType { .. }))
            .collect();
        assert!(!unknown_errors.is_empty());
    }

    #[test]
    fn collect_duplicate_struct_definition() {
        let src = "struct A { x: i64 } struct A { y: bool }";
        let (_, errors, _) = check(src);
        // exactly one duplicate-error.
        let dup: Vec<&QalaError> = errors
            .iter()
            .filter(|e| matches!(e, QalaError::Type { message, .. } if message.contains("duplicate")))
            .collect();
        assert_eq!(dup.len(), 1, "{errors:?}");
    }

    #[test]
    fn collect_errors_are_sorted_by_span() {
        // a program that emits errors out of source order. duplicate-struct
        // and the unknown field type are both reported; the resulting errors
        // vec is sorted by span.start.
        let src = "struct A { x: Nope } struct A { y: i64 }";
        let (_, errors, _) = check(src);
        // span.start should be non-decreasing.
        let starts: Vec<u32> = errors.iter().map(|e| e.span().start).collect();
        let mut sorted = starts.clone();
        sorted.sort();
        assert_eq!(starts, sorted, "errors not sorted by span.start: {errors:?}");
    }

    // ---- pass 2: bidirectional type checking -------------------------------

    #[test]
    fn infer_local_let_types() {
        // let x = 42 -> x: i64.
        let src = "fn main() is io { let x = 42; println(\"hi\") }";
        let (_, errors, _) = check(src);
        // unused-var on x is expected since it is not read.
        let type_errors: Vec<&QalaError> =
            errors.iter().filter(|e| !matches!(e, QalaError::UndefinedName { .. })).collect();
        assert!(type_errors.is_empty(), "{errors:?}");
    }

    #[test]
    fn let_with_wrong_annotation_emits_type_mismatch() {
        // let x: i64 = "hello" -> TypeMismatch at the rhs span.
        let src = "fn main() is io { let x: i64 = \"hello\"; println(\"\") }";
        let (_, errors, _) = check(src);
        let mismatch: Vec<&QalaError> = errors
            .iter()
            .filter(|e| matches!(e, QalaError::TypeMismatch { .. }))
            .collect();
        assert!(!mismatch.is_empty(), "expected TypeMismatch in {errors:?}");
    }

    #[test]
    fn redundant_annotation_warns() {
        // let x: i64 = 42 fires `redundant_annotation`.
        let src = "fn main() is io { let x: i64 = 42; println(\"{x}\") }";
        let (_, errors, warnings) = check(src);
        assert!(errors.is_empty(), "{errors:?}");
        let red: Vec<&QalaWarning> = warnings
            .iter()
            .filter(|w| w.category == "redundant_annotation")
            .collect();
        assert_eq!(red.len(), 1, "{warnings:?}");
    }

    #[test]
    fn undefined_name_in_initializer_emits_one_error() {
        let src = "fn main() is io { let x = nope; println(\"\") }";
        let (_, errors, _) = check(src);
        let undef: Vec<&QalaError> = errors
            .iter()
            .filter(|e| matches!(e, QalaError::UndefinedName { name, .. } if name == "nope"))
            .collect();
        assert_eq!(undef.len(), 1, "{errors:?}");
    }

    #[test]
    fn arg_type_mismatch_message() {
        // a fn declared with -> str whose body's trailing value is an i64.
        let src = "fn f(x: i64) -> str is pure { x }";
        let (_, errors, _) = check(src);
        let mismatch: Vec<&QalaError> = errors
            .iter()
            .filter(|e| matches!(e, QalaError::TypeMismatch { .. }))
            .collect();
        assert!(!mismatch.is_empty(), "expected TypeMismatch in {errors:?}");
        match mismatch[0] {
            QalaError::TypeMismatch { expected, found, .. } => {
                assert_eq!(expected, "str");
                assert_eq!(found, "i64");
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn missing_return_at_last_expr() {
        // a non-void fn whose body has no trailing value and no return stmt.
        let src = "fn f() -> i64 is pure { }";
        let (_, errors, _) = check(src);
        let missing: Vec<&QalaError> = errors
            .iter()
            .filter(|e| matches!(e, QalaError::MissingReturn { .. }))
            .collect();
        assert_eq!(missing.len(), 1, "{errors:?}");
    }

    #[test]
    fn fn_with_correct_trailing_value_passes() {
        let src = "fn f(x: i64) -> i64 is pure { x }";
        let (_, errors, _) = check(src);
        assert!(errors.is_empty(), "{errors:?}");
    }

    #[test]
    fn fn_with_explicit_return_passes() {
        let src = "fn f(x: i64) -> i64 is pure { return x }";
        let (_, errors, _) = check(src);
        assert!(errors.is_empty(), "{errors:?}");
    }

    #[test]
    fn fn_return_with_wrong_type() {
        let src = "fn f(x: i64) -> i64 is pure { return \"oops\" }";
        let (_, errors, _) = check(src);
        let mismatch: Vec<&QalaError> = errors
            .iter()
            .filter(|e| matches!(e, QalaError::TypeMismatch { .. }))
            .collect();
        assert!(!mismatch.is_empty(), "{errors:?}");
    }

    #[test]
    fn or_fallback_typechecks_success() {
        // x: Option<i64> or i64 -> i64.
        let src = r#"
            fn lookup() -> Option<i64> is pure { return Some(1) }
            fn main() is io {
                let r = lookup() or 0
                println("{r}")
            }
        "#;
        let (_, errors, _) = check(src);
        let mismatch: Vec<&QalaError> = errors
            .iter()
            .filter(|e| matches!(e, QalaError::TypeMismatch { .. }))
            .collect();
        assert!(mismatch.is_empty(), "{errors:?}");
    }

    #[test]
    fn or_fallback_wrong_type_errors_at_fallback() {
        let src = r#"
            fn lookup() -> Option<i64> is pure { return Some(1) }
            fn main() is io {
                let r = lookup() or "oops"
                println("{r}")
            }
        "#;
        let (_, errors, _) = check(src);
        let mismatch: Vec<&QalaError> = errors
            .iter()
            .filter(|e| matches!(e, QalaError::TypeMismatch { .. }))
            .collect();
        assert!(!mismatch.is_empty(), "{errors:?}");
    }

    #[test]
    fn question_legal_inside_result_fn() {
        let src = r#"
            fn parse_int(s: str) -> Result<i64, str> is pure { return Ok(0) }
            fn f(s: str) -> Result<i64, str> is pure {
                let x = parse_int(s)?
                return Ok(x)
            }
        "#;
        let (_, errors, _) = check(src);
        let redundant: Vec<&QalaError> = errors
            .iter()
            .filter(|e| matches!(e, QalaError::RedundantQuestionOperator { .. }))
            .collect();
        assert!(redundant.is_empty(), "{errors:?}");
    }

    #[test]
    fn question_in_non_result_fn_errors() {
        let src = r#"
            fn parse_int(s: str) -> Result<i64, str> is pure { return Ok(0) }
            fn f(s: str) -> i64 is pure {
                let x = parse_int(s)?
                return x
            }
        "#;
        let (_, errors, _) = check(src);
        let redundant: Vec<&QalaError> = errors
            .iter()
            .filter(|e| matches!(e, QalaError::RedundantQuestionOperator { .. }))
            .collect();
        assert_eq!(redundant.len(), 1, "{errors:?}");
    }

    #[test]
    fn struct_literal_unknown_field_errors() {
        let src = r#"
            struct Point { x: i64, y: i64 }
            fn main() is io {
                let p = Point { x: 1, z: 2 }
                println("ok")
            }
        "#;
        let (_, errors, _) = check(src);
        // expect a missing-field error AND a no-such-field error.
        let type_errors: Vec<&QalaError> = errors
            .iter()
            .filter(|e| matches!(e, QalaError::Type { .. }))
            .collect();
        assert!(type_errors.len() >= 2, "{errors:?}");
    }

    #[test]
    fn method_call_resolves_user_method() {
        let src = r#"
            struct Point { x: f64, y: f64 }
            fn Point.distance(self) -> f64 is pure { return 0.0 }
            fn main() is io {
                let p = Point { x: 1.0, y: 2.0 }
                let d = p.distance()
                println("{d}")
            }
        "#;
        let (_, errors, _) = check(src);
        assert!(errors.is_empty(), "{errors:?}");
    }

    #[test]
    fn index_into_fixed_array_types_to_elem() {
        let src = r#"
            fn main() is io {
                let arr = [1, 2, 3]
                let v = arr[0]
                println("{v}")
            }
        "#;
        let (_, errors, _) = check(src);
        assert!(errors.is_empty(), "{errors:?}");
    }

    #[test]
    fn index_with_string_errors() {
        let src = r#"
            fn main() is io {
                let arr = [1, 2, 3]
                let v = arr["x"]
                println("{v}")
            }
        "#;
        let (_, errors, _) = check(src);
        let mismatch: Vec<&QalaError> = errors
            .iter()
            .filter(|e| matches!(e, QalaError::TypeMismatch { .. }))
            .collect();
        assert!(!mismatch.is_empty(), "{errors:?}");
    }

    #[test]
    fn pipeline_with_unary_callee_types_through() {
        let src = r#"
            fn double(x: i64) -> i64 is pure { return x * 2 }
            fn main() is io {
                let r = 5 |> double
                println("{r}")
            }
        "#;
        let (_, errors, _) = check(src);
        assert!(errors.is_empty(), "{errors:?}");
    }

    #[test]
    fn interpolation_resolves_inner_expression() {
        let src = r#"
            fn main() is io {
                let name = "world"
                println("hello, {name}!")
            }
        "#;
        let (_, errors, _) = check(src);
        assert!(errors.is_empty(), "{errors:?}");
    }

    #[test]
    fn binary_arithmetic_on_matching_types_passes() {
        let src = r#"
            fn add(a: i64, b: i64) -> i64 is pure { return a + b }
        "#;
        let (_, errors, _) = check(src);
        assert!(errors.is_empty(), "{errors:?}");
    }

    #[test]
    fn binary_arithmetic_on_mismatched_types_errors() {
        let src = r#"
            fn bad(a: i64, b: str) -> i64 is pure { return a + b }
        "#;
        let (_, errors, _) = check(src);
        let mismatch: Vec<&QalaError> = errors
            .iter()
            .filter(|e| matches!(e, QalaError::TypeMismatch { .. }))
            .collect();
        assert!(!mismatch.is_empty(), "{errors:?}");
    }

    #[test]
    fn comparison_returns_bool() {
        let src = r#"
            fn lt(a: i64, b: i64) -> bool is pure { return a < b }
        "#;
        let (_, errors, _) = check(src);
        assert!(errors.is_empty(), "{errors:?}");
    }

    #[test]
    fn boolean_ops_require_bool_operands() {
        let src = r#"
            fn bad() -> bool is pure { return 1 && 2 }
        "#;
        let (_, errors, _) = check(src);
        let mismatch: Vec<&QalaError> = errors
            .iter()
            .filter(|e| matches!(e, QalaError::TypeMismatch { .. }))
            .collect();
        // two bool mismatches (lhs and rhs).
        assert!(mismatch.len() >= 2, "{errors:?}");
    }

    #[test]
    fn unreachable_after_return_warns_once() {
        // `return` is followed by `let`, a statement-head keyword, so the
        // parser treats the return as bare and the `let` as a separate
        // statement -- which then triggers the unreachable_code warning.
        let src = r#"
            fn main() is io {
                return
                let x = 1
            }
        "#;
        let (_, _, warnings) = check(src);
        let unreach: Vec<&QalaWarning> = warnings
            .iter()
            .filter(|w| w.category == "unreachable_code")
            .collect();
        assert_eq!(unreach.len(), 1, "{warnings:?}");
    }
}
