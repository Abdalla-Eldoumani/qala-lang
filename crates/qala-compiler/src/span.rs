//! source locations: a [`Span`] is a byte offset plus a length into the source
//! string, and [`LineIndex`] turns a byte offset into a 1-based line and column.
//!
//! the scanner only ever advances byte offsets, so spans stay two `u32`s with no
//! line or column fields. line and column are computed on demand from a line-start
//! table; the cost is paid only when a diagnostic is actually rendered.

/// a region of the source: `start` is a byte offset, `len` is a byte length.
///
/// `u32` fields cap a source file at 4 GiB, which is fine for a single-file
/// teaching language and halves the struct size against `usize`. `Copy` matters
/// because spans are attached to every token and every AST node and get cloned
/// constantly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Span {
    /// byte offset of the first byte of this region, from the start of the source.
    pub start: u32,
    /// length of this region in bytes.
    pub len: u32,
}

impl Span {
    /// construct a span from a byte offset and a byte length.
    pub fn new(start: usize, len: usize) -> Self {
        Span {
            start: start as u32,
            len: len as u32,
        }
    }

    /// the byte offset one past the last byte of this region (`start + len`).
    pub fn end(self) -> usize {
        (self.start + self.len) as usize
    }

    /// the smallest span covering both inputs.
    ///
    /// uses `min` on the start and `max` on the end, so the result is correct
    /// regardless of argument order and whether the two spans overlap, touch, or
    /// have a gap between them. the parser uses this to span a whole expression
    /// from its first token to its last.
    pub fn merge(self, other: Span) -> Span {
        let start = self.start.min(other.start);
        let end = (self.end().max(other.end())) as u32;
        Span {
            start,
            len: end - start,
        }
    }

    /// alias for [`Span::merge`]; reads better at parser call sites
    /// (`first.to(last)`).
    pub fn to(self, other: Span) -> Span {
        self.merge(other)
    }

    /// the source text this span covers.
    ///
    /// the caller must pass the same source the span was produced from; the span
    /// indexes it with its own `start` and `end`, both of which the scanner
    /// produced on UTF-8 boundaries.
    pub fn slice(self, src: &str) -> &str {
        &src[self.start as usize..self.end()]
    }
}

/// a table of line-start byte offsets, built once from the source, used to map a
/// byte offset to a 1-based line and column.
///
/// `line_starts[0]` is always `0` (the first line starts at the first byte); an
/// entry `offset + 1` is pushed after each `\n` byte. `\r\n` needs no special
/// case: the `\n` ends the line and a lone `\r` is just whitespace inside the
/// preceding line.
pub struct LineIndex {
    /// byte offset where each line begins; one entry per line, strictly increasing.
    line_starts: Vec<u32>,
}

impl LineIndex {
    /// scan the source once and record where each line starts.
    pub fn new(src: &str) -> Self {
        let mut line_starts = vec![0u32];
        for (i, b) in src.bytes().enumerate() {
            if b == b'\n' {
                line_starts.push(i as u32 + 1);
            }
        }
        LineIndex { line_starts }
    }

    /// the 1-based line and 1-based column of `byte_offset` in `src`.
    ///
    /// the column is a count of characters from the start of the line, so a tab
    /// counts as one column, not eight. this is deliberate: the logical column is
    /// character-based, and any tab-expanded display column is the renderer's job
    /// later. an offset at end of file returns a valid pair and never panics.
    /// an offset beyond `src.len()` is clamped to `src.len()` rather than
    /// panicking; this keeps WASM alive when a downstream caller (e.g. the
    /// diagnostics layer) passes a span-derived offset that is slightly wrong.
    pub fn location(&self, src: &str, byte_offset: usize) -> (usize, usize) {
        let byte_offset = byte_offset.min(src.len());
        let off = byte_offset as u32;
        // index of the last line start that is <= off, i.e. the line containing
        // off. line_starts[0] == 0 and off >= 0, so partition_point returns at
        // least 1 here and the subtraction never underflows.
        let line = self.line_starts.partition_point(|&s| s <= off) - 1;
        let line_start = self.line_starts[line] as usize;
        let col = src[line_start..byte_offset].chars().count() + 1;
        (line + 1, col)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_overlapping_spans() {
        // 0..5 and 3..8 -> 0..8
        let a = Span::new(0, 5);
        let b = Span::new(3, 5);
        let m = a.merge(b);
        assert_eq!(m.start, 0);
        assert_eq!(m.len, 8);
    }

    #[test]
    fn merge_adjacent_spans() {
        // 0..3 and 3..7 -> 0..7, one contiguous span, no gap
        let a = Span::new(0, 3);
        let b = Span::new(3, 4);
        let m = a.merge(b);
        assert_eq!(m.start, 0);
        assert_eq!(m.len, 7);
    }

    #[test]
    fn merge_gapped_spans() {
        // 0..2 and 5..7 -> 0..7, the gap (2..5) is included
        let a = Span::new(0, 2);
        let b = Span::new(5, 2);
        let m = a.merge(b);
        assert_eq!(m.start, 0);
        assert_eq!(m.len, 7);
    }

    #[test]
    fn merge_is_order_independent() {
        let a = Span::new(2, 3); // 2..5
        let b = Span::new(7, 4); // 7..11
        assert_eq!(a.merge(b), b.merge(a));
        let m = a.merge(b);
        assert_eq!(m.start, 2);
        assert_eq!(m.len, 9);
    }

    #[test]
    fn to_is_an_alias_for_merge() {
        let a = Span::new(1, 2);
        let b = Span::new(10, 3);
        assert_eq!(a.to(b), a.merge(b));
        assert_eq!(b.to(a), a.merge(b));
    }

    #[test]
    fn slice_returns_the_covered_source() {
        let src = "let x = 42;";
        let span = Span::new(4, 1); // the `x`
        assert_eq!(span.slice(src), "x");
        let span = Span::new(8, 2); // the `42`
        assert_eq!(span.slice(src), "42");
        // a token's span round-trips: start/len slice back to the right text.
        let whole = Span::new(0, src.len());
        assert_eq!(whole.slice(src), src);
    }

    #[test]
    fn end_is_start_plus_len() {
        let s = Span::new(5, 3);
        assert_eq!(s.end(), 8);
        assert_eq!(Span::new(0, 0).end(), 0);
    }

    #[test]
    fn location_at_offset_zero_is_line_1_col_1() {
        let src = "abc\ndef";
        let idx = LineIndex::new(src);
        assert_eq!(idx.location(src, 0), (1, 1));
    }

    #[test]
    fn location_at_a_newline_is_on_the_line_it_ends() {
        // "abc\ndef": the '\n' is byte 3, on line 1, column 4.
        let src = "abc\ndef";
        let idx = LineIndex::new(src);
        assert_eq!(idx.location(src, 3), (1, 4));
    }

    #[test]
    fn location_just_after_a_newline_is_next_line_col_1() {
        // byte 4 is 'd', the first char of line 2.
        let src = "abc\ndef";
        let idx = LineIndex::new(src);
        assert_eq!(idx.location(src, 4), (2, 1));
    }

    #[test]
    fn location_at_end_of_file_is_valid_and_does_not_panic() {
        let src = "abc\ndef";
        let idx = LineIndex::new(src);
        // src.len() == 7: end of line 2, column 4 (one past 'f').
        assert_eq!(idx.location(src, src.len()), (2, 4));
        // an empty file: offset 0 == len 0, line 1 column 1.
        let empty = "";
        let eidx = LineIndex::new(empty);
        assert_eq!(eidx.location(empty, 0), (1, 1));
    }

    #[test]
    fn location_clamps_out_of_range_offset_instead_of_panicking() {
        // an offset exactly at src.len() is valid (end-of-file position).
        let src = "abc\ndef";
        let idx = LineIndex::new(src);
        let at_eof = idx.location(src, src.len());
        // an offset past the end must return the same position, not panic.
        assert_eq!(idx.location(src, src.len() + 1), at_eof);
        assert_eq!(idx.location(src, src.len() + 100), at_eof);
        // empty source: offset 0 and offset 5 both map to (1, 1).
        let empty = "";
        let eidx = LineIndex::new(empty);
        assert_eq!(eidx.location(empty, 5), (1, 1));
    }

    #[test]
    fn location_counts_columns_as_characters_so_tabs_are_one_column() {
        // line 2 is "\t\tx": two tabs then 'x'. the 'x' is byte 6, column 3
        // (char count from line start: tab, tab, then x is the 3rd char), not
        // column 17 or anything tab-expanded. this is LEX-06.
        let src = "fn f() {\n\t\tx\n}";
        let idx = LineIndex::new(src);
        let x_offset = src.find('x').unwrap();
        assert_eq!(idx.location(src, x_offset), (2, 3));
        // a space-indented line gives the same column for the same character
        // position: "  y" -> 'y' at column 3.
        let src2 = "fn f() {\n  y\n}";
        let idx2 = LineIndex::new(src2);
        let y_offset = src2.find('y').unwrap();
        assert_eq!(idx2.location(src2, y_offset), (2, 3));
    }

    #[test]
    fn crlf_line_endings_break_on_the_newline_not_the_carriage_return() {
        // "a\r\nb": '\r' is byte 1 (still line 1), '\n' is byte 2 (line 1),
        // 'b' is byte 3 (line 2, column 1). a lone '\r' is not a line break.
        let src = "a\r\nb";
        let idx = LineIndex::new(src);
        assert_eq!(idx.location(src, 1), (1, 2)); // the '\r'
        assert_eq!(idx.location(src, 3), (2, 1)); // the 'b'
    }
}
