//! the assembly-text builder for the ARM64 backend.
//!
//! [`Asm`] accumulates emitted instructions and labels into a `.text` section
//! buffer (a `.data` buffer is held but stays empty for the integer core --
//! the integer slice has no strings, no `printf`, nothing to place). [`Asm::finish`]
//! prepends the fixed m4 `define` preamble and the `.text` section header,
//! producing the complete CPSC 355-dialect `.asm` text.
//!
//! the dialect, fixed: lowercase mnemonics and directives throughout, the m4
//! `define(fp, x29)` / `define(lr, x30)` register-alias preamble, GAS sections,
//! column-0 labels and directives, 8-space-indented instructions. matches the
//! course tutorial files under `docs/references/`.

/// the standard CPSC 355 instruction indentation: eight spaces. labels and
/// directives sit at column 0; instructions are indented under them.
const INDENT: &str = "        ";

/// the assembly-text builder. holds the `.text` section buffer (where every
/// function's directives, prologue, body, and epilogue land) and an unused
/// `.data` buffer reserved for Phase 13's format strings.
///
/// crate-private: only the `arm64` module builds assembly text.
pub(crate) struct Asm {
    /// the `.data` section buffer. the `printf` output path is the first
    /// writer: a `print` / `println` of an `i64` registers its `"%lld"` /
    /// `"%lld\n"` format string here. empty for a program with no output.
    data: String,
    /// the `.text` section buffer: per-function directives, labels, and
    /// instructions, in emission order.
    text: String,
    /// set once the println format string (`"%lld\n"`) has been registered
    /// in the `.data` buffer -- guards against a duplicate registration. the
    /// reader is the `printf` call lowering in `expr.rs`, the next commit.
    #[allow(dead_code)]
    println_fmt_used: bool,
    /// set once the print format string (`"%lld"`) has been registered.
    #[allow(dead_code)]
    print_fmt_used: bool,
}

impl Asm {
    /// construct an empty builder. the m4 preamble and the section headers are
    /// written by [`Asm::finish`], not seeded here, so an in-progress `Asm`
    /// holds only the caller's emitted lines.
    pub(crate) fn new() -> Self {
        Asm {
            data: String::new(),
            text: String::new(),
            println_fmt_used: false,
            print_fmt_used: false,
        }
    }

    /// ensure the `"%lld\n"` println format string is in the `.data`
    /// section; return its label. registers the string on the first call
    /// and is a no-op on every later call -- one shared label per program.
    ///
    /// the caller is `compile_print_call` in `expr.rs`, added in the next
    /// commit; the inline tests below exercise it in the meantime.
    #[allow(dead_code)]
    pub(crate) fn ensure_println_fmt(&mut self) -> &'static str {
        if !self.println_fmt_used {
            self.data.push_str(".Lfmt_println:  .string \"%lld\\n\"\n");
            self.println_fmt_used = true;
        }
        ".Lfmt_println"
    }

    /// ensure the `"%lld"` print format string is in the `.data` section;
    /// return its label. one shared label per program -- see [`Asm::ensure_println_fmt`].
    #[allow(dead_code)]
    pub(crate) fn ensure_print_fmt(&mut self) -> &'static str {
        if !self.print_fmt_used {
            self.data.push_str(".Lfmt_print:    .string \"%lld\"\n");
            self.print_fmt_used = true;
        }
        ".Lfmt_print"
    }

    /// append a raw line, plus a trailing newline, to the `.text` buffer.
    ///
    /// the line is written verbatim with no indentation -- the caller controls
    /// its own column. used for blank separators and any line that is neither a
    /// plain label nor a plain instruction.
    pub(crate) fn emit_line(&mut self, line: &str) {
        self.text.push_str(line);
        self.text.push('\n');
    }

    /// append a label definition (`<label>:`) at column 0 to the `.text` buffer.
    ///
    /// both global function labels (`main`) and local assembler labels
    /// (`.Lmain_epilogue`) go through here -- the `.L` prefix, when present, is
    /// already part of `label`.
    pub(crate) fn emit_label(&mut self, label: &str) {
        self.text.push_str(label);
        self.text.push_str(":\n");
    }

    /// append an instruction to the `.text` buffer, indented with the standard
    /// CPSC 355 eight-space indent.
    pub(crate) fn emit_insn(&mut self, insn: &str) {
        self.text.push_str(INDENT);
        self.text.push_str(insn);
        self.text.push('\n');
    }

    /// append an instruction with a trailing `// comment`, indented and
    /// comment-aligned in the worked-example style (`str x0, [fp, -8]  // a`).
    ///
    /// the comment documents what a stack slot or value is, mirroring the
    /// `// a`, `// spill lhs` annotations in the reference assembly.
    pub(crate) fn emit_insn_commented(&mut self, insn: &str, comment: &str) {
        self.text.push_str(INDENT);
        self.text.push_str(insn);
        self.text.push_str("  // ");
        self.text.push_str(comment);
        self.text.push('\n');
    }

    /// concatenate the m4 preamble, the (omitted-if-empty) `.data` section, and
    /// the `.text` section into the complete `.asm` text.
    ///
    /// the preamble is exactly the two `define` lines and a blank line. the
    /// `.data` section is emitted only when its buffer is non-empty -- the
    /// integer core leaves it empty, so `finish` produces preamble + `.text`.
    /// every emitted line ends in `\n`; the result is LF-only regardless of host.
    pub(crate) fn finish(self) -> String {
        let mut out = String::new();
        // the fixed m4 register-alias preamble.
        out.push_str("define(fp, x29)\n");
        out.push_str("define(lr, x30)\n");
        out.push('\n');
        // .data: emitted only when there is data to place (Phase 13 onward).
        if !self.data.is_empty() {
            out.push_str(INDENT);
            out.push_str(".data\n");
            out.push_str(&self.data);
            out.push('\n');
        }
        // .text: the section header, then every function's emitted lines.
        out.push_str(INDENT);
        out.push_str(".text\n");
        out.push_str(&self.text);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finish_on_an_empty_builder_is_the_preamble_and_text_header() {
        // the deterministic baseline: a builder with nothing emitted produces
        // exactly the two define lines, a blank line, and the .text header.
        let asm = Asm::new();
        let expected = "define(fp, x29)\n\
                        define(lr, x30)\n\
                        \n\
                        \x20\x20\x20\x20\x20\x20\x20\x20.text\n";
        assert_eq!(asm.finish(), expected);
    }

    #[test]
    fn finish_omits_the_data_section_when_the_buffer_is_empty() {
        // the integer core never writes .data; finish must not emit a bare
        // ".data" header for an empty buffer.
        let asm = Asm::new();
        assert!(!asm.finish().contains(".data"));
    }

    #[test]
    fn emit_label_writes_the_label_at_column_zero() {
        let mut asm = Asm::new();
        asm.emit_label("main");
        asm.emit_label(".Lmain_epilogue");
        let out = asm.finish();
        assert!(out.contains("\nmain:\n"));
        assert!(out.contains("\n.Lmain_epilogue:\n"));
    }

    #[test]
    fn emit_insn_indents_with_eight_spaces() {
        let mut asm = Asm::new();
        asm.emit_insn("mov x0, 0");
        let out = asm.finish();
        assert!(out.contains("\n        mov x0, 0\n"), "missing 8-space indent: {out:?}");
    }

    #[test]
    fn emit_insn_commented_appends_a_trailing_comment() {
        let mut asm = Asm::new();
        asm.emit_insn_commented("str x0, [fp, -8]", "a");
        let out = asm.finish();
        assert!(out.contains("        str x0, [fp, -8]  // a\n"), "{out:?}");
    }

    #[test]
    fn emit_line_writes_the_line_verbatim() {
        let mut asm = Asm::new();
        asm.emit_line("");
        let out = asm.finish();
        // a blank emit_line produces an empty line inside .text.
        assert!(out.ends_with(".text\n\n"));
    }

    #[test]
    fn finish_output_is_lf_only() {
        // every writer appends '\n'; the result must carry no carriage return.
        let mut asm = Asm::new();
        asm.emit_label("f");
        asm.emit_insn("ret");
        assert!(!asm.finish().contains('\r'));
    }

    #[test]
    fn ensure_println_fmt_registers_the_format_string_and_returns_its_label() {
        let mut asm = Asm::new();
        assert_eq!(asm.ensure_println_fmt(), ".Lfmt_println");
        let out = asm.finish();
        // the label+directive lands inside a .data section.
        assert!(out.contains("        .data\n"), "missing .data section: {out:?}");
        assert!(
            out.contains(".Lfmt_println:  .string \"%lld\\n\"\n"),
            "missing the println format string: {out:?}"
        );
    }

    #[test]
    fn ensure_print_fmt_registers_the_format_string_and_returns_its_label() {
        let mut asm = Asm::new();
        assert_eq!(asm.ensure_print_fmt(), ".Lfmt_print");
        let out = asm.finish();
        assert!(out.contains("        .data\n"), "missing .data section: {out:?}");
        assert!(
            out.contains(".Lfmt_print:    .string \"%lld\"\n"),
            "missing the print format string: {out:?}"
        );
        // the print format string has no trailing newline escape.
        assert!(!out.contains("%lld\\n"), "print must not carry a newline: {out:?}");
    }

    #[test]
    fn ensure_println_fmt_registers_the_string_only_once() {
        // a second call is a no-op -- one shared label per program.
        let mut asm = Asm::new();
        asm.ensure_println_fmt();
        asm.ensure_println_fmt();
        let out = asm.finish();
        assert_eq!(
            out.matches(".Lfmt_println:").count(),
            1,
            "the println format string must be registered exactly once: {out:?}"
        );
    }

    #[test]
    fn finish_still_omits_the_data_section_when_no_format_string_is_registered() {
        // the empty-.data behaviour is unchanged when neither ensure_* is called.
        let asm = Asm::new();
        assert!(!asm.finish().contains(".data"));
    }
}
