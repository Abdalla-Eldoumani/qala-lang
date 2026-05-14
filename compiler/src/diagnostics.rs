//! the diagnostics layer: one [`Diagnostic`] data model with two rendering
//! paths -- [`Diagnostic::render`] (Task 2) produces a Rust-style
//! ASCII-underlined source block for CLI output and snapshot tests;
//! [`Diagnostic::to_monaco`] produces a serde-Serialize [`MonacoDiagnostic`]
//! for the playground's inline editor underlines. errors and warnings share
//! the model; [`Severity`] distinguishes them; warnings carry their
//! snake_case category as `Diagnostic::category`.
//!
//! determinism: same source + same Diagnostic = byte-identical render
//! output, verified by a render-twice unit test plus a snapshot file checked
//! into `compiler/tests/snapshots/`. user-visible lists (missing variants,
//! missing methods, cycle paths) are pre-sorted by the typechecker so this
//! module never iterates a HashMap into a user-visible string.

use crate::errors::QalaError;
use crate::span::{LineIndex, Span};
use crate::typechecker::QalaWarning;

/// is this diagnostic an error (compilation-blocking) or a warning
/// (informational; never blocks codegen)?
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Severity {
    /// a hard fault: the typechecker produced a [`QalaError`] and codegen
    /// will not run.
    Error,
    /// a soft fault: the typechecker produced a [`QalaWarning`]; codegen
    /// still runs.
    Warning,
}

/// one rendered-ready diagnostic, decoupled from the typechecker's
/// internal error and warning enums.
///
/// not derived `PartialEq`: the contract is the output of
/// [`Diagnostic::render`] (and [`Diagnostic::to_monaco`]), not the
/// internal field layout. notes and hints are ordered lists and the order
/// is part of the rendered output, but two Diagnostics with the same
/// rendering may differ in non-rendered fields without breaking anything.
#[derive(Debug, Clone)]
pub struct Diagnostic {
    /// error or warning -- selects the rendered header word.
    pub severity: Severity,
    /// the warning's snake_case category (`unused_var`, `shadowed_var`,
    /// ...); `None` for errors. this is what `// qala: allow(<category>)`
    /// matches against and what the playground colors yellow.
    pub category: Option<String>,
    /// the one-line description, taken verbatim from
    /// [`QalaError::message`] or [`QalaWarning::message`].
    pub message: String,
    /// the source region the diagnostic points at -- used by the renderer
    /// to find the line text, line:column, and the underline width.
    pub primary_span: Span,
    /// `   = note: {note}` lines appended to the rendered block, in this
    /// order, after the source-line underline.
    pub notes: Vec<String>,
    /// `   = hint: {hint}` lines appended after every note, in this
    /// order. rustc-style: notes describe facts about the program; hints
    /// describe edits the user might make.
    pub hints: Vec<String>,
}

/// the structured form of a diagnostic for the Monaco editor's marker
/// API. Monaco wants 1-based line and column, an end position, a
/// severity-as-integer (0=Hint, 1=Info, 2=Warning, 4=Error per
/// `MarkerSeverity` -- we use the locked 0=Warning, 1=Error pair), a
/// message, and an optional category (which the playground renders as
/// the marker source string).
///
/// derives `serde::Serialize` so the Phase 6 WASM bridge can hand a
/// `Vec<MonacoDiagnostic>` straight to JS via `serde-wasm-bindgen`.
#[derive(Debug, Clone, serde::Serialize)]
pub struct MonacoDiagnostic {
    /// 1-based line of the diagnostic's start, from
    /// [`LineIndex::location`].
    pub line: u32,
    /// 1-based column of the diagnostic's start, character-counted (a
    /// tab is one column, matching Span::location).
    pub column: u32,
    /// 1-based line of one past the diagnostic's end.
    pub end_line: u32,
    /// 1-based column of one past the diagnostic's end (the end of the
    /// underlined region; matches Monaco's exclusive-end convention).
    pub end_column: u32,
    /// 0 = warning, 1 = error. matches the locked Monaco
    /// MarkerSeverity convention adopted by the playground.
    pub severity: u8,
    /// the diagnostic message, verbatim.
    pub message: String,
    /// the warning category (snake_case); `None` for errors.
    pub category: Option<String>,
}

impl Diagnostic {
    /// translate this Diagnostic into the structured Monaco form.
    ///
    /// the source string is required to translate byte offsets into
    /// 1-based line and column via [`LineIndex::location`]; the line
    /// index is built once per call and discarded (the typical playground
    /// path renders a small batch of diagnostics from one source string,
    /// so the build cost is paid per-call rather than cached).
    pub fn to_monaco(&self, src: &str) -> MonacoDiagnostic {
        let line_index = LineIndex::new(src);
        let (line, column) = line_index.location(src, self.primary_span.start as usize);
        let (end_line, end_column) = line_index.location(src, self.primary_span.end());
        MonacoDiagnostic {
            line: line as u32,
            column: column as u32,
            end_line: end_line as u32,
            end_column: end_column as u32,
            severity: match self.severity {
                Severity::Error => 1,
                Severity::Warning => 0,
            },
            message: self.message.clone(),
            category: self.category.clone(),
        }
    }
}

impl From<QalaError> for Diagnostic {
    /// build a Diagnostic from a [`QalaError`]. severity is always
    /// `Error`; category is always `None` (errors have no category --
    /// only warnings do). the message comes from `err.message()` and the
    /// span from `err.span()`. five variants carry structured extra
    /// data which becomes per-method `note:` lines or a single `hint:`
    /// line; the rest leave notes and hints empty.
    fn from(err: QalaError) -> Self {
        let message = err.message();
        let primary_span = err.span();
        let mut notes: Vec<String> = Vec::new();
        let mut hints: Vec<String> = Vec::new();
        // match on a reference so any future variant addition forces a
        // deliberate decision here -- not because the compiler enforces
        // exhaustiveness on Diagnostic construction (a wildcard arm
        // would also compile) but because reading this match is the
        // only place a reader sees the five "structured" variants
        // listed together.
        match &err {
            QalaError::EffectViolation { caller_effect, callee_effect, .. } => {
                hints.push(format!(
                    "remove the `is {caller_effect}` annotation, or refactor to remove the {callee_effect} call"
                ));
            }
            QalaError::NonExhaustiveMatch { missing, .. } if !missing.is_empty() => {
                notes.push(format!("add an arm for: {}", missing.join(", ")));
            }
            QalaError::InterfaceNotSatisfied { missing, mismatched, .. } => {
                // typechecker pre-sorts both lists; render in that order.
                for name in missing {
                    notes.push(format!("missing method `{name}`"));
                }
                for (method, expected, found) in mismatched {
                    notes.push(format!(
                        "method `{method}` has signature {found}, expected {expected}"
                    ));
                }
            }
            QalaError::RecursiveStructByValue { path, .. } => {
                notes.push(format!("cycle path: {}", path.join(" -> ")));
            }
            QalaError::RedundantQuestionOperator { .. } => {
                hints.push(
                    "change the function's return type to a compatible Result/Option, or handle the error explicitly"
                        .to_string(),
                );
            }
            // every other variant -- the lex/parse error family and the
            // simpler type-error variants -- carries enough context in
            // its one-line message. no notes, no hints.
            _ => {}
        }
        Diagnostic {
            severity: Severity::Error,
            category: None,
            message,
            primary_span,
            notes,
            hints,
        }
    }
}

impl From<&QalaWarning> for Diagnostic {
    /// build a Diagnostic from a [`QalaWarning`] reference. severity is
    /// `Warning`; category is `Some(w.category)`. warnings carry their
    /// snake_case category as `Diagnostic::category`; this is what
    /// `// qala: allow(<category>)` matches against and what the
    /// playground colors yellow. the optional `note` field becomes the
    /// first note line.
    fn from(w: &QalaWarning) -> Self {
        let notes = w
            .note
            .as_ref()
            .map_or_else(Vec::new, |n| vec![n.clone()]);
        Diagnostic {
            severity: Severity::Warning,
            category: Some(w.category.clone()),
            message: w.message.clone(),
            primary_span: w.span,
            notes,
            hints: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::QalaError;
    use crate::span::Span;
    use crate::typechecker::QalaWarning;

    /// a small default span; tests that want a specific position
    /// construct `Span::new(...)` inline.
    fn sp() -> Span {
        Span::new(0, 1)
    }

    #[test]
    fn diagnostic_builds_and_round_trips_through_debug_clone() {
        let d = Diagnostic {
            severity: Severity::Error,
            category: None,
            message: "x".to_string(),
            primary_span: Span::new(0, 1),
            notes: Vec::new(),
            hints: Vec::new(),
        };
        let copy = d.clone();
        // Debug works on every field; the format produces something non-empty.
        let dbg = format!("{copy:?}");
        assert!(!dbg.is_empty());
        // shape stayed the same across the clone.
        assert_eq!(copy.message, "x");
        assert!(matches!(copy.severity, Severity::Error));
        assert_eq!(copy.primary_span, Span::new(0, 1));
        assert!(copy.notes.is_empty());
        assert!(copy.hints.is_empty());
        assert!(copy.category.is_none());
    }

    #[test]
    fn from_qala_error_int_overflow_carries_message_and_span() {
        let err = QalaError::IntOverflow { span: Span::new(5, 4) };
        let d: Diagnostic = err.into();
        assert!(matches!(d.severity, Severity::Error));
        assert!(d.category.is_none());
        assert_eq!(d.message, "integer literal is too large for i64");
        assert_eq!(d.primary_span, Span::new(5, 4));
        assert!(d.notes.is_empty());
        assert!(d.hints.is_empty());
    }

    #[test]
    fn from_qala_error_effect_violation_adds_locked_hint() {
        let err = QalaError::EffectViolation {
            span: sp(),
            caller: "compute".to_string(),
            caller_effect: "pure".to_string(),
            callee: "println".to_string(),
            callee_effect: "io".to_string(),
        };
        let d: Diagnostic = err.into();
        assert_eq!(
            d.hints,
            vec![
                "remove the `is pure` annotation, or refactor to remove the io call".to_string()
            ]
        );
        assert!(d.notes.is_empty());
    }

    #[test]
    fn from_qala_error_non_exhaustive_match_adds_note() {
        let err = QalaError::NonExhaustiveMatch {
            span: sp(),
            enum_name: "Shape".to_string(),
            missing: vec!["Bar".to_string(), "Foo".to_string()],
        };
        let d: Diagnostic = err.into();
        assert_eq!(d.notes, vec!["add an arm for: Bar, Foo".to_string()]);
        assert!(d.hints.is_empty());
    }

    #[test]
    fn from_qala_error_interface_not_satisfied_emits_per_method_notes() {
        let err = QalaError::InterfaceNotSatisfied {
            span: sp(),
            ty: "Point".to_string(),
            interface: "Printable".to_string(),
            missing: vec!["a".to_string(), "b".to_string()],
            mismatched: vec![(
                "c".to_string(),
                "fn(Self) -> str".to_string(),
                "fn(Self) -> i64".to_string(),
            )],
        };
        let d: Diagnostic = err.into();
        assert_eq!(
            d.notes,
            vec![
                "missing method `a`".to_string(),
                "missing method `b`".to_string(),
                "method `c` has signature fn(Self) -> i64, expected fn(Self) -> str".to_string(),
            ]
        );
        assert!(d.hints.is_empty());
    }

    #[test]
    fn from_qala_error_recursive_struct_adds_cycle_path_note() {
        let err = QalaError::RecursiveStructByValue {
            span: sp(),
            path: vec!["A".to_string(), "B".to_string(), "A".to_string()],
        };
        let d: Diagnostic = err.into();
        assert_eq!(d.notes, vec!["cycle path: A -> B -> A".to_string()]);
        assert!(d.hints.is_empty());
    }

    #[test]
    fn from_qala_error_redundant_question_operator_adds_hint() {
        let err = QalaError::RedundantQuestionOperator {
            span: sp(),
            message: "?".to_string(),
        };
        let d: Diagnostic = err.into();
        assert_eq!(
            d.hints,
            vec![
                "change the function's return type to a compatible Result/Option, or handle the error explicitly"
                    .to_string()
            ]
        );
        assert!(d.notes.is_empty());
    }

    #[test]
    fn from_qala_warning_without_note_yields_empty_notes() {
        let w = QalaWarning {
            category: "unused_var".to_string(),
            message: "unused variable `x`".to_string(),
            span: Span::new(10, 1),
            note: None,
        };
        let d: Diagnostic = (&w).into();
        assert!(matches!(d.severity, Severity::Warning));
        assert_eq!(d.category, Some("unused_var".to_string()));
        assert_eq!(d.message, "unused variable `x`");
        assert_eq!(d.primary_span, Span::new(10, 1));
        assert!(d.notes.is_empty());
        assert!(d.hints.is_empty());
    }

    #[test]
    fn from_qala_warning_with_note_pushes_one_note_entry() {
        let w = QalaWarning {
            category: "shadowed_var".to_string(),
            message: "shadowed `x`".to_string(),
            span: Span::new(20, 1),
            note: Some("the prior binding is at line 3:5".to_string()),
        };
        let d: Diagnostic = (&w).into();
        assert_eq!(
            d.notes,
            vec!["the prior binding is at line 3:5".to_string()]
        );
        assert!(matches!(d.severity, Severity::Warning));
    }

    #[test]
    fn to_monaco_translates_a_single_line_span_to_1_based_columns() {
        // span Span::new(0, 5) over "hello" -> covers all 5 bytes
        // start byte 0 -> line 1, column 1
        // end   byte 5 -> line 1, column 6 (one past the last char)
        let d = Diagnostic {
            severity: Severity::Error,
            category: None,
            message: "m".to_string(),
            primary_span: Span::new(0, 5),
            notes: Vec::new(),
            hints: Vec::new(),
        };
        let m = d.to_monaco("hello");
        assert_eq!(m.line, 1);
        assert_eq!(m.column, 1);
        assert_eq!(m.end_line, 1);
        assert_eq!(m.end_column, 6);
        assert_eq!(m.severity, 1);
        assert_eq!(m.message, "m");
        assert!(m.category.is_none());
    }

    #[test]
    fn to_monaco_translates_a_multi_line_span_to_two_line_numbers() {
        // src "a\nb\nc": bytes a=0, \n=1, b=2, \n=3, c=4
        // span Span::new(0, 5) starts at byte 0 (line 1, col 1) and
        // ends at byte 5 (one past c, line 3, col 2).
        let d = Diagnostic {
            severity: Severity::Error,
            category: None,
            message: "m".to_string(),
            primary_span: Span::new(0, 5),
            notes: Vec::new(),
            hints: Vec::new(),
        };
        let m = d.to_monaco("a\nb\nc");
        assert_eq!(m.line, 1);
        assert_eq!(m.column, 1);
        assert_eq!(m.end_line, 3);
        assert_eq!(m.end_column, 2);
    }

    #[test]
    fn to_monaco_severity_maps_error_to_1_and_warning_to_0() {
        let err_diag = Diagnostic {
            severity: Severity::Error,
            category: None,
            message: String::new(),
            primary_span: Span::new(0, 0),
            notes: Vec::new(),
            hints: Vec::new(),
        };
        assert_eq!(err_diag.to_monaco("").severity, 1);
        let warn_diag = Diagnostic {
            severity: Severity::Warning,
            category: Some("unused_var".to_string()),
            message: String::new(),
            primary_span: Span::new(0, 0),
            notes: Vec::new(),
            hints: Vec::new(),
        };
        assert_eq!(warn_diag.to_monaco("").severity, 0);
    }

    #[test]
    fn monaco_diagnostic_implements_serialize() {
        // a compile-time check: if MonacoDiagnostic doesn't implement
        // serde::Serialize, this generic call fails to typecheck and
        // the test build breaks. Serialize has a generic method so it
        // is not dyn-compatible; a generic asserting function is the
        // standard way to spell "T: Serialize" as a witness.
        fn assert_serialize<T: serde::Serialize>(_: &T) {}
        let m = MonacoDiagnostic {
            line: 1,
            column: 1,
            end_line: 1,
            end_column: 1,
            severity: 0,
            message: String::new(),
            category: None,
        };
        assert_serialize(&m);
    }

    #[test]
    fn from_qala_error_is_deterministic_across_two_calls() {
        // determinism property: From<QalaError> is a pure transformation
        // -- the match arms are fixed-order and the structured arms
        // preserve typechecker-sorted lists, so two calls produce
        // notes/hints in the same order. the match-arm-fixed-order is
        // the guarantee; this test makes it visible.
        let err = QalaError::InterfaceNotSatisfied {
            span: sp(),
            ty: "Point".to_string(),
            interface: "Printable".to_string(),
            missing: vec!["a".to_string(), "b".to_string()],
            mismatched: vec![(
                "c".to_string(),
                "fn(Self) -> str".to_string(),
                "fn(Self) -> i64".to_string(),
            )],
        };
        let a: Diagnostic = err.clone().into();
        let b: Diagnostic = err.into();
        assert_eq!(a.notes, b.notes);
        assert_eq!(a.hints, b.hints);
        assert_eq!(a.message, b.message);
        assert_eq!(a.primary_span, b.primary_span);
    }
}
