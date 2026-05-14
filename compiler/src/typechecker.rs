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

// ---- pass 2 placeholder ----------------------------------------------------

impl<'src> Checker<'src> {
    /// translate an untyped item into a typed one.
    ///
    /// task 1 produces minimal typed shells: every fn returns Void with
    /// pure effect, every struct / enum / interface forwards its resolved
    /// fields. tasks 2-3 replace this with the real bidirectional pass that
    /// walks bodies and types every expression.
    fn check_item(&mut self, item: &ast::Item) -> typed_ast::TypedItem {
        match item {
            ast::Item::Fn(d) => typed_ast::TypedItem::Fn(typed_ast::TypedFnDecl {
                type_name: d.type_name.clone(),
                name: d.name.clone(),
                params: d
                    .params
                    .iter()
                    .map(|p| typed_ast::TypedParam {
                        is_self: p.is_self,
                        name: p.name.clone(),
                        ty: if p.is_self {
                            match &d.type_name {
                                Some(t) => QalaType::Named(Symbol(t.clone())),
                                None => QalaType::Unknown,
                            }
                        } else if let Some(t) = &p.ty {
                            // re-resolve (cheap; doesn't error twice because
                            // collect_fn already populated the symbol table
                            // and any UnknownType errors fired during collect).
                            resolve_type_silent(t, &self.symbols)
                        } else {
                            QalaType::Unknown
                        },
                        default: None,
                        span: p.span,
                    })
                    .collect(),
                ret_ty: d
                    .ret_ty
                    .as_ref()
                    .map(|t| resolve_type_silent(t, &self.symbols))
                    .unwrap_or(QalaType::Void),
                effect: d.effect.as_ref().map(effect_set_from_ast).unwrap_or(EffectSet::pure()),
                body: typed_ast::TypedBlock {
                    stmts: vec![],
                    value: None,
                    ty: QalaType::Void,
                    span: d.body.span,
                },
                span: d.span,
            }),
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

    /// pass 3: resolve every function's inferred effect.
    ///
    /// task 1 leaves this as a no-op so the crate compiles end-to-end while
    /// tasks 2-3 are pending.
    fn resolve_function_effects(&mut self) {
        // no-op; task 3 fills this in.
    }
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
}
