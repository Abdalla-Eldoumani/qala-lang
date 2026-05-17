// the bytecode panel: it renders the compiled program's disassembly as
// monospace instruction rows and highlights the row the vm is about to
// execute. phase 7 shipped this as a styled placeholder; phase 8 fills
// the body with the real view. the panel frame and the Bytecode header
// chrome are kept from phase 7 unchanged.
//
// the highlight here keys on VmState.ip -- a byte offset into a chunk.
// it is a different highlight from the editor's current-line highlight,
// which keys on VmState.current_line, a source line. the two are never
// conflated.

import { useEffect, useRef } from "react";
import type { CompileResult, VmState } from "@/lib/types";

// a parsed disassembly line: either a chunk header or one instruction.
// the offset is the leading integer of an instruction line; it is the
// value VmState.ip carries, so an offset/ip equality match selects the
// current instruction within a chunk.
type BytecodeRow =
  | { kind: "header"; fnId: number; label: string }
  | {
      kind: "instruction";
      offset: number;
      opcode: string;
      operand: string;
      line: number | null;
    };

// the locked Program::disassemble format is, per chunk, a header line
// `== chunk: NAME (fn_id N) ==`, then one line per instruction:
//   `{off:>4}  {name:<9} {operand:<24} ; line {line}`
// a 4-char right-aligned offset, two spaces, a 9-char padded opcode, one
// space, a 24-char padded operand, then ` ; line ` and the source line.
const CHUNK_HEADER = /^== chunk: (.+) \(fn_id (\d+)\) ==$/;
const LINE_NOTE = " ; line ";

// the opcode and operand are fixed-width columns of the instruction body
// (the part before the ` ; line ` note). the offset is 4 chars then two
// spaces, so the opcode starts at index 6 and runs to 15; the operand
// starts at index 16. slicing the fixed columns -- rather than splitting
// on whitespace -- is required because an operand can carry an internal
// space (e.g. `#0 42`), which a whitespace split would tear apart.
const OPCODE_START = 6;
const OPCODE_END = 15;
const OPERAND_START = 16;

// the parsed rows plus, per row, the fn_id of the chunk that owns it.
// the owning fn_id is the most recent header's fn_id; -1 before any
// header. precomputing it here keeps the render a pure read -- no
// mutable cursor reassigned inside the JSX map.
interface ParsedDisassembly {
  rows: BytecodeRow[];
  rowFnIds: number[];
}

function parseDisassembly(disassembly: string): ParsedDisassembly {
  const rows: BytecodeRow[] = [];
  const rowFnIds: number[] = [];
  let currentFnId = -1;
  for (const rawLine of disassembly.split("\n")) {
    // blank lines separate chunks; skip them.
    if (rawLine.trim() === "") continue;

    const headerMatch = CHUNK_HEADER.exec(rawLine);
    if (headerMatch) {
      currentFnId = Number.parseInt(headerMatch[2], 10);
      rows.push({
        kind: "header",
        label: headerMatch[1],
        fnId: currentFnId,
      });
      rowFnIds.push(currentFnId);
      continue;
    }

    // an instruction line. split on the literal ` ; line ` note first:
    // the right side is the source-line number, the left side is the
    // fixed-width instruction body.
    const noteIndex = rawLine.indexOf(LINE_NOTE);
    const body = noteIndex >= 0 ? rawLine.slice(0, noteIndex) : rawLine;
    const noteText =
      noteIndex >= 0 ? rawLine.slice(noteIndex + LINE_NOTE.length).trim() : "";

    // parseInt reads the leading integer and ignores the right-aligned
    // column's leading spaces. a line whose leading token is not a
    // number is not an instruction row -- skip it defensively.
    const offset = Number.parseInt(body, 10);
    if (Number.isNaN(offset)) continue;

    const parsedLine = Number.parseInt(noteText, 10);
    rows.push({
      kind: "instruction",
      offset,
      opcode: body.slice(OPCODE_START, OPCODE_END).trim(),
      operand: body.slice(OPERAND_START).trim(),
      line: Number.isNaN(parsedLine) ? null : parsedLine,
    });
    rowFnIds.push(currentFnId);
  }
  return { rows, rowFnIds };
}

function InstructionRow({
  row,
  current,
}: {
  row: Extract<BytecodeRow, { kind: "instruction" }>;
  current: boolean;
}) {
  // the current row scrolls itself into view when it is off-screen. a
  // ref on every instruction row keeps this simple; only the current
  // one acts on it. block: "nearest" is an instant adjust -- a fast run
  // loop must not lag behind a smooth scroll.
  const rowRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    if (current) rowRef.current?.scrollIntoView({ block: "nearest" });
  }, [current]);

  return (
    <div
      ref={rowRef}
      // the highlight is the --accent-soft band plus a 2px --accent left
      // bar, so the current instruction is identifiable without the soft
      // background alone. weight stays 400 -- a monospace column must not
      // shift glyph width on highlight. the 100ms fade eases the move.
      className={`flex items-baseline gap-[var(--space-3)] py-[var(--space-1)] pr-[var(--space-3)] font-mono text-[13px] font-normal leading-[1.6] transition-colors duration-100 ease-out ${
        current
          ? "border-l-2 border-accent bg-accent-soft pl-[calc(var(--space-3)-2px)]"
          : "border-l-2 border-transparent pl-[calc(var(--space-3)-2px)]"
      }`}
    >
      <span className="w-[3ch] shrink-0 text-right text-text-secondary">
        {row.offset}
      </span>
      <span className="text-text-primary">{row.opcode}</span>
      <span className="flex-1 text-text-secondary">{row.operand}</span>
      {row.line !== null && (
        // the source-line note sits at the trailing edge; it is dropped
        // under 768px where row width is tight.
        <span className="hidden shrink-0 text-text-secondary md:block">
          line {row.line}
        </span>
      )}
    </div>
  );
}

export default function BytecodePanel({
  className,
  // the compile result carries the disassembly string; null before the
  // first compile. state carries the ip the highlight keys on; null
  // before the first step. dirty true means the disassembly is stale.
  // these are optional: the phase 7 page renders this panel chrome-only,
  // and plan 08-05 wires the snapshots. the defaults are the empty,
  // not-stepping, not-dirty state.
  compile = null,
  state = null,
  dirty = false,
}: {
  className?: string;
  compile?: CompileResult | null;
  state?: VmState | null;
  dirty?: boolean;
}) {
  // a frame for the scrolling body, kept distinct from the header so the
  // panel chrome (frame, header, bottom rule) renders identically in
  // every state.
  const frame = (children: React.ReactNode) => (
    <section
      className={`flex flex-col overflow-hidden rounded-[var(--radius-panel)] border border-border bg-surface ${className ?? ""}`}
    >
      <header className="border-b border-border p-[var(--space-3)] text-[12px] font-medium uppercase tracking-wide leading-[1.4] text-text-secondary">
        Bytecode
      </header>
      {children}
    </section>
  );

  const message = (text: string) => (
    <div className="p-[var(--space-4)] text-[14px] font-normal leading-[1.5] text-text-secondary">
      {text}
    </div>
  );

  // a dirty disassembly is stale: show the needs-compile sentence instead
  // of stale rows with a frozen highlight.
  if (dirty) {
    return frame(message("bytecode is out of date -- compile to refresh"));
  }

  // compile.disassembly is absent (undefined, never null -- a rust None)
  // when no program compiled or the compile failed. truthiness covers
  // both the null compile result and the absent disassembly string.
  if (!compile?.disassembly) {
    return frame(message("compile a program to see its bytecode"));
  }

  // rowFnIds[i] is the fn_id of the chunk that owns rows[i] -- computed
  // in the pure parse pass. the current instruction is the row whose
  // owning fn_id equals VmState.chunk_index and whose offset equals
  // VmState.ip. scoping to the chunk matters: every chunk has an
  // instruction at offset 0, so an offset-only match would highlight the
  // wrong function.
  const { rows, rowFnIds } = parseDisassembly(compile.disassembly);

  return frame(
    <div className="overflow-y-auto py-[var(--space-1)]">
      {rows.map((row, i) => {
        if (row.kind === "header") {
          return (
            <div
              key={`h${i}`}
              // a quiet full-width band; a hairline rule above every
              // header except the first.
              className={`px-[var(--space-3)] py-[var(--space-1)] text-[12px] font-medium leading-[1.4] text-text-secondary ${
                i > 0 ? "border-t border-border" : ""
              }`}
            >
              {row.label}
            </div>
          );
        }
        const current =
          state !== null &&
          rowFnIds[i] === state.chunk_index &&
          row.offset === state.ip;
        return <InstructionRow key={`i${i}`} row={row} current={current} />;
      })}
    </div>,
  );
}
