//! turn compiler errors and warnings into rendered diagnostic text for stderr.

use qala_compiler::diagnostics::Diagnostic;
use qala_compiler::errors::QalaError;
use qala_compiler::typechecker::QalaWarning;

/// render every error to the compiler's rich text format, joined.
pub fn render_errors(errors: &[QalaError], src: &str) -> String {
    errors
        .iter()
        .map(|e| Diagnostic::from(e.clone()).render(src))
        .collect::<Vec<_>>()
        .join("\n")
}

/// render every warning to the compiler's rich text format, joined.
pub fn render_warnings(warnings: &[QalaWarning], src: &str) -> String {
    warnings
        .iter()
        .map(|w| Diagnostic::from(w).render(src))
        .collect::<Vec<_>>()
        .join("\n")
}
