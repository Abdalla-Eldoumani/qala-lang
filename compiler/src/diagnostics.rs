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
    /// render this diagnostic as a Rust-style ASCII-underlined source
    /// block.
    ///
    /// byte-identical for the same `(Diagnostic, src)` pair: same input
    /// produces the same output, regardless of process, build, or
    /// platform. that is the DIAG-04 contract; a snapshot test in
    /// `compiler/tests/snapshots/diagnostics_basic.txt` proves it
    /// cross-process.
    ///
    /// the block has five lines plus optional notes and hints:
    ///
    /// ```text
    /// error: <message>           (or `warning: <message>`)
    ///   --> <line>:<column>
    ///    |
    /// N  |   <source line>
    ///    |   <pad><carets>
    ///    |
    ///    = note: spans multiple lines      (only when the span clips)
    ///    = note: <each note>
    ///    = hint: <each hint>
    /// ```
    ///
    /// the underline is one `^` per byte of the span, clipped to the
    /// source line so a multi-line span does not produce screen-fulls
    /// of carets (Pitfall 4 of the research); when clipping occurs, a
    /// `note: spans multiple lines` is appended before any user notes.
    ///
    /// no color codes, no emoji, no host paths -- the renderer is a
    /// pure function over `(self, src)` and produces only ASCII text.
    pub fn render(&self, src: &str) -> String {
        let line_index = LineIndex::new(src);
        let (line, column) =
            line_index.location(src, self.primary_span.start as usize);
        let line_text = src.lines().nth(line - 1).unwrap_or("");

        let header = match self.severity {
            Severity::Error => "error",
            Severity::Warning => "warning",
        };

        let mut out = String::new();

        // header line: "error: <message>" / "warning: <message>".
        out.push_str(&format!("{header}: {}\n", self.message));

        // arrow line: file path is omitted; Phase 6's WASM bridge
        // supplies the file name separately if needed.
        out.push_str(&format!("  --> {line}:{column}\n"));

        // top separator.
        out.push_str("   |\n");

        // source line: line number left-justified to 3 cols, then
        // `|   `, then the source text. the line numbers in tests can
        // be at most 3 digits in practice (a 999-line program is the
        // outer end of what the playground asks the renderer to draw),
        // and 3 cols of padding is the rustc convention.
        out.push_str(&format!("{line:<3}|   {line_text}\n"));

        // underline width: one `^` per byte of the span, clipped to
        // the source line so a multi-line span does not run off the
        // end (Pitfall 4). `avail` is the remaining bytes from the
        // start column to the end of the source line.
        let avail = line_text.len().saturating_sub(column.saturating_sub(1));
        let span_bytes = self.primary_span.len as usize;
        let underline_width = std::cmp::min(span_bytes, avail).max(1);
        let pad = " ".repeat(column.saturating_sub(1));
        let underline = "^".repeat(underline_width);
        out.push_str(&format!("   |   {pad}{underline}\n"));

        // bottom separator.
        out.push_str("   |\n");

        // multi-line clip note (prepended before any user notes).
        if span_bytes > avail {
            out.push_str("   = note: spans multiple lines\n");
        }

        // user notes, then hints.
        for note in &self.notes {
            out.push_str(&format!("   = note: {note}\n"));
        }
        for hint in &self.hints {
            out.push_str(&format!("   = hint: {hint}\n"));
        }

        out
    }

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
    use crate::lexer;
    use crate::parser;
    use crate::span::Span;
    use crate::typechecker;
    use crate::typechecker::QalaWarning;

    /// a small default span; tests that want a specific position
    /// construct `Span::new(...)` inline.
    fn sp() -> Span {
        Span::new(0, 1)
    }

    /// build a Diagnostic for a small bad program by running it
    /// through lex/parse/check and converting the first error to a
    /// Diagnostic. used by the renderer tests so they exercise the
    /// real From<QalaError> conversion against real typechecker
    /// output rather than hand-constructed errors.
    fn diag_for(src: &str) -> Diagnostic {
        let tokens = lexer::Lexer::tokenize(src).expect("lex");
        let ast = parser::Parser::parse(&tokens).expect("parse");
        let (_, errors, _) = typechecker::check_program(&ast, src);
        assert!(!errors.is_empty(), "expected an error for: {src}");
        Diagnostic::from(errors[0].clone())
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

    // ---- Task 2: render tests ---------------------------------------------

    #[test]
    fn render_header_says_error_for_an_error_diagnostic() {
        let d = Diagnostic {
            severity: Severity::Error,
            category: None,
            message: "oops".to_string(),
            primary_span: Span::new(0, 1),
            notes: Vec::new(),
            hints: Vec::new(),
        };
        let out = d.render("x");
        assert!(
            out.starts_with("error: oops\n"),
            "wrong header: {out:?}"
        );
    }

    #[test]
    fn render_arrow_line_uses_1_based_line_and_column() {
        // src "abcde\nfghij": bytes a=0,b=1,c=2,d=3,e=4,\n=5,f=6,g=7,...
        // byte 6 (the `f`) is on line 2, column 1 -- the Plan's stated
        // expected output "  --> 2:1" corresponds to byte 6, not byte 5
        // (byte 5 is the `\n` on line 1 col 6). picking byte 6 here
        // matches the Plan's intent: "show that a span on line 2 col 1
        // renders the arrow line correctly".
        let d = Diagnostic {
            severity: Severity::Error,
            category: None,
            message: "m".to_string(),
            primary_span: Span::new(6, 1),
            notes: Vec::new(),
            hints: Vec::new(),
        };
        let out = d.render("abcde\nfghij");
        assert!(out.contains("  --> 2:1\n"), "{out:?}");
    }

    #[test]
    fn render_source_line_includes_line_number_and_text() {
        // src "abc\ndef\nghi", span on line 2 (byte 4 is `d`, length 3
        // covers `def`). source line should render as "2  |   def\n".
        let d = Diagnostic {
            severity: Severity::Error,
            category: None,
            message: "m".to_string(),
            primary_span: Span::new(4, 3),
            notes: Vec::new(),
            hints: Vec::new(),
        };
        let out = d.render("abc\ndef\nghi");
        assert!(out.contains("2  |   def\n"), "{out:?}");
    }

    #[test]
    fn render_underline_has_one_caret_per_byte_at_column_1() {
        // src "hello", span 0..5 -> column 1, pad="" (0 chars), 5 carets.
        let d = Diagnostic {
            severity: Severity::Error,
            category: None,
            message: "m".to_string(),
            primary_span: Span::new(0, 5),
            notes: Vec::new(),
            hints: Vec::new(),
        };
        let out = d.render("hello");
        assert!(out.contains("   |   ^^^^^\n"), "{out:?}");
    }

    #[test]
    fn render_underline_pads_for_non_first_column_spans() {
        // src "hello", span 2..5 -> column 3, pad="  " (2 chars), 3 carets.
        let d = Diagnostic {
            severity: Severity::Error,
            category: None,
            message: "m".to_string(),
            primary_span: Span::new(2, 3),
            notes: Vec::new(),
            hints: Vec::new(),
        };
        let out = d.render("hello");
        assert!(out.contains("   |     ^^^\n"), "{out:?}");
    }

    #[test]
    fn render_multi_line_span_clips_underline_and_appends_note() {
        // src "line one\nline two" (line 1 = "line one" = 8 chars).
        // span 0..20 starts at byte 0 (col 1) and runs 20 bytes past
        // the end of line 1. underline clips to 8 carets and a
        // "spans multiple lines" note is appended before any user notes.
        let d = Diagnostic {
            severity: Severity::Error,
            category: None,
            message: "m".to_string(),
            primary_span: Span::new(0, 20),
            notes: Vec::new(),
            hints: Vec::new(),
        };
        let out = d.render("line one\nline two");
        assert!(out.contains("   |   ^^^^^^^^\n"), "{out:?}");
        assert!(out.contains("   = note: spans multiple lines\n"), "{out:?}");
    }

    #[test]
    fn render_emits_bottom_separator_before_notes_and_hints() {
        // a Diagnostic with one note and one hint should have the
        // bottom "   |\n" separator BEFORE either of them, and the
        // notes BEFORE the hints (rustc order).
        let d = Diagnostic {
            severity: Severity::Error,
            category: None,
            message: "m".to_string(),
            primary_span: Span::new(0, 1),
            notes: vec!["a fact".to_string()],
            hints: vec!["a suggestion".to_string()],
        };
        let out = d.render("x");
        let bottom = out.find("   |\n   = note: a fact\n");
        assert!(bottom.is_some(), "bottom separator and note must be adjacent: {out:?}");
        let note_pos = out.find("   = note: a fact\n").unwrap();
        let hint_pos = out.find("   = hint: a suggestion\n").unwrap();
        assert!(note_pos < hint_pos, "note must come before hint: {out:?}");
    }

    #[test]
    fn render_emits_one_line_per_note_in_order() {
        let d = Diagnostic {
            severity: Severity::Error,
            category: None,
            message: "m".to_string(),
            primary_span: Span::new(0, 1),
            notes: vec!["first".to_string(), "second".to_string()],
            hints: Vec::new(),
        };
        let out = d.render("x");
        let first = out.find("   = note: first\n").unwrap();
        let second = out.find("   = note: second\n").unwrap();
        assert!(first < second, "notes must render in vec order: {out:?}");
    }

    #[test]
    fn render_emits_one_line_per_hint_in_order() {
        let d = Diagnostic {
            severity: Severity::Error,
            category: None,
            message: "m".to_string(),
            primary_span: Span::new(0, 1),
            notes: Vec::new(),
            hints: vec!["try this".to_string(), "or that".to_string()],
        };
        let out = d.render("x");
        let first = out.find("   = hint: try this\n").unwrap();
        let second = out.find("   = hint: or that\n").unwrap();
        assert!(first < second, "hints must render in vec order: {out:?}");
    }

    #[test]
    fn render_warning_uses_warning_header() {
        let d = Diagnostic {
            severity: Severity::Warning,
            category: Some("unused_var".to_string()),
            message: "unused variable `x`".to_string(),
            primary_span: Span::new(0, 1),
            notes: Vec::new(),
            hints: Vec::new(),
        };
        let out = d.render("let x = 1");
        assert!(
            out.starts_with("warning: unused variable `x`\n"),
            "wrong warning header: {out:?}"
        );
    }

    #[test]
    fn render_is_byte_identical_across_two_calls() {
        // determinism property in-process: same Diagnostic + same src
        // = byte-identical output across two calls. exercises a real
        // typechecker-produced error via diag_for.
        let src = "fn f() -> i64 { return \"x\" }";
        let d = diag_for(src);
        let a = d.render(src);
        let b = d.render(src);
        assert_eq!(a, b, "render output drifted between calls");
    }

    #[test]
    fn six_bundled_examples_render_no_error_output() {
        // the Plan 04 typechecker smoke test asserted errors is empty;
        // this paired Plan 05 test asserts rendering those (empty)
        // errors collects to the empty string -- proves the renderer
        // is wired in correctly for the no-error case.
        for name in ["hello", "fibonacci", "effects", "pattern-matching", "pipeline", "defer-demo"] {
            let path = format!(
                "{}/../playground/public/examples/{}.qala",
                env!("CARGO_MANIFEST_DIR"),
                name
            );
            let src = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("read {path}: {e}"));
            let tokens = lexer::Lexer::tokenize(&src).expect("lex");
            let ast = parser::Parser::parse(&tokens).expect("parse");
            let (_, errors, _warnings) = typechecker::check_program(&ast, &src);
            assert!(errors.is_empty(), "{name}.qala: unexpected errors: {errors:?}");
            let rendered: String = errors
                .iter()
                .map(|e| Diagnostic::from(e.clone()).render(&src))
                .collect::<Vec<_>>()
                .join("");
            assert!(
                rendered.is_empty(),
                "{name}.qala: errors-rendered-to-empty failed: {rendered:?}"
            );
        }
    }

    #[test]
    fn warnings_never_block_typed_ast_construction() {
        // a program that produces a warning still yields a non-empty
        // TypedAst. this is a regression check that warnings are not
        // errors: the typechecker emits the warning AND keeps going.
        let src = "fn main() is io {\nlet x = 1\nprintln(\"hi\")\n}";
        let tokens = lexer::Lexer::tokenize(src).expect("lex");
        let ast = parser::Parser::parse(&tokens).expect("parse");
        let (typed, errors, warnings) = typechecker::check_program(&ast, src);
        assert!(errors.is_empty(), "unexpected errors: {errors:?}");
        assert!(!warnings.is_empty(), "expected at least one warning");
        assert!(!typed.is_empty(), "warnings must not block typed-AST construction");
    }

    #[test]
    fn render_with_no_notes_or_hints_ends_at_the_bottom_separator() {
        // a clean Diagnostic (no notes, no hints) should produce
        // exactly: header + arrow + top separator + source line +
        // underline + bottom separator. no trailing `= note:` or
        // `= hint:` lines.
        let d = Diagnostic {
            severity: Severity::Error,
            category: None,
            message: "boom".to_string(),
            primary_span: Span::new(0, 1),
            notes: Vec::new(),
            hints: Vec::new(),
        };
        let out = d.render("x");
        // the last non-empty line is the bottom separator.
        assert!(out.ends_with("   |\n"), "unexpected tail: {out:?}");
        assert!(!out.contains("= note:"), "no notes expected: {out:?}");
        assert!(!out.contains("= hint:"), "no hints expected: {out:?}");
    }
}
