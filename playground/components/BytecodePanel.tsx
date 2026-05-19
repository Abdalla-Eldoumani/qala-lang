// the compiler-output panel: a two-tab panel showing what the compiler
// produced from the current source. tab 1 (Bytecode) renders the
// bytecode disassembly as monospace instruction rows and highlights the
// row the vm is about to execute. tab 2 (ARM64) renders the AArch64
// assembly the ARM64 backend emits, read-only, or -- when the program is
// outside the backend's integer core -- the honest unsupported-construct
// diagnostic. the panel frame and the header tab strip are constant
// across every state so the chrome never shifts.
//
// the bytecode highlight keys on VmState.ip -- a byte offset into a
// chunk. the ARM64 tab has no highlight at all: the backend has no
// playground runtime, so there is no current line to track. the two
// tabs do not share a highlight abstraction -- they are deliberately
// distinct.

import { useEffect, useRef, useState } from "react";
import type { Arm64Result, CompileResult, VmState } from "@/lib/types";

// the active compiler-output tab. component-local UI state -- it is not
// execution state and does not belong in useQala.
type OutputTab = "bytecode" | "arm64";

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
      // the highlight is the --accent-soft band alone -- the current
      // instruction is identifiable by the soft background, no stripe.
      // weight stays 400 -- a monospace column must not shift glyph width
      // on highlight. the 100ms fade eases the move.
      className={`flex items-baseline gap-[var(--space-3)] py-[var(--space-1)] pl-[var(--space-3)] pr-[var(--space-3)] font-mono text-[13px] font-normal leading-[1.6] transition-colors duration-100 ease-out ${
        current ? "bg-accent-soft" : ""
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

// tab 1: the bytecode disassembly view. three states -- stale (a dirty
// edit since the last compile), empty (no program compiled), and the
// instruction listing. the stale check comes before the empty check so
// an edit before the first compile shows stale, not empty.
function BytecodeTab({
  compile,
  state,
  dirty,
}: {
  compile: CompileResult | null;
  state: VmState | null;
  dirty: boolean;
}) {
  // a dirty disassembly is stale: show the needs-compile sentence instead
  // of stale rows with a frozen highlight.
  if (dirty) {
    return <Message text="bytecode is out of date, compile to refresh" />;
  }

  // compile.disassembly is absent (undefined, never null -- a rust None)
  // when no program compiled or the compile failed. truthiness covers
  // both the null compile result and the absent disassembly string.
  if (!compile?.disassembly) {
    return <Message text="compile a program to see its bytecode" />;
  }

  // rowFnIds[i] is the fn_id of the chunk that owns rows[i] -- computed
  // in the pure parse pass. the current instruction is the row whose
  // owning fn_id equals VmState.chunk_index and whose offset equals
  // VmState.ip. scoping to the chunk matters: every chunk has an
  // instruction at offset 0, so an offset-only match would highlight the
  // wrong function.
  const { rows, rowFnIds } = parseDisassembly(compile.disassembly);

  return (
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
    </div>
  );
}

// tab 2: the ARM64 assembly view. four states -- stale, empty, assembly,
// and unsupported. the stale check comes before the empty check so an
// edit before the first compile shows stale, matching the bytecode tab.
//
// the unsupported state is calm by design: an ARM64-unsupported
// construct is not an error -- the program is correct, it simply targets
// a backend slice that does not cover it yet -- so it renders in
// --text-secondary / --text-primary, never --error red.
function Arm64Tab({
  arm64,
  dirty,
}: {
  arm64: Arm64Result | null;
  dirty: boolean;
}) {
  // the copy button's label swaps to "copied" for ~1.5s after a click.
  const [copied, setCopied] = useState(false);
  // the swap timer lives in a ref so a fast second click clears the
  // first timer rather than leaving a stale revert pending.
  const copyTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => {
    return () => {
      if (copyTimerRef.current !== null) clearTimeout(copyTimerRef.current);
    };
  }, []);

  // a dirty edit since the last compile means the assembly is stale.
  // checked before the empty state so an edit before the first compile
  // reads as stale, not empty -- the same rule the bytecode tab follows.
  if (dirty) {
    return <Message text="assembly is out of date, compile to refresh" />;
  }

  // arm64 is null before the first compile; assembly is absent (a rust
  // None -> undefined) when the backend produced no assembly. the empty
  // state covers both the null snapshot and a never-ok result.
  const assembly = arm64?.assembly;

  if (assembly) {
    const handleCopy = () => {
      void navigator.clipboard.writeText(assembly).then(() => {
        setCopied(true);
        if (copyTimerRef.current !== null) clearTimeout(copyTimerRef.current);
        copyTimerRef.current = setTimeout(() => setCopied(false), 1500);
      });
    };
    return (
      <div className="flex min-h-0 flex-1 flex-col">
        {/* the meta line: labels the artifact honestly and hosts the
            copy button at its trailing edge. */}
        <div className="flex items-center justify-between border-b border-border px-[var(--space-3)] py-[var(--space-2)]">
          <span className="text-[12px] font-medium leading-[1.4] text-text-secondary">
            AArch64 assembly, CPSC 355 dialect
          </span>
          <button
            type="button"
            onClick={handleCopy}
            className="rounded-[var(--radius-button)] border border-border bg-surface px-[var(--space-3)] py-[var(--space-1)] text-[12px] font-medium leading-[1.4] text-text-primary transition-colors duration-150 ease-out hover:border-text-secondary focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
          >
            {copied ? "copied" : "copy"}
          </button>
        </div>
        {/* the assembly body: the backend's own indentation and column
            alignment are meaningful, so whitespace-pre preserves them
            verbatim -- no re-tokenizing, no re-indenting, no highlight,
            no line numbers. it is a plain scroll container, not
            virtualized -- integer-core programs are short. */}
        <pre className="min-h-0 flex-1 overflow-auto whitespace-pre px-[var(--space-3)] py-[var(--space-1)] font-mono text-[13px] font-normal leading-[1.6] text-text-primary">
          {assembly}
        </pre>
      </div>
    );
  }

  // the unsupported state: the program compiled for the VM but uses a
  // construct outside the ARM64 integer core. show the first diagnostic
  // -- the earliest offending construct -- framed as information, not a
  // failure. the disabled copy button keeps the chrome stable.
  const detail = arm64?.diagnostics[0]?.message;
  if (detail) {
    return (
      <div className="flex min-h-0 flex-1 flex-col">
        <div className="flex items-center justify-between border-b border-border px-[var(--space-3)] py-[var(--space-2)]">
          <span className="text-[12px] font-medium leading-[1.4] text-text-secondary">
            the ARM64 backend is integer-core, this program is outside it
          </span>
          <button
            type="button"
            disabled
            className="rounded-[var(--radius-button)] border border-border bg-surface px-[var(--space-3)] py-[var(--space-1)] text-[12px] font-medium leading-[1.4] text-text-secondary"
          >
            copy
          </button>
        </div>
        <div className="px-[var(--space-3)] py-[var(--space-2)] font-mono text-[13px] font-normal leading-[1.6] text-text-primary">
          {detail}
        </div>
      </div>
    );
  }

  // empty: no program compiled yet, or a compile that produced neither
  // assembly nor a diagnostic.
  return <Message text="compile a program to see its ARM64 assembly" />;
}

// the body-role copy shown by the empty and stale states of either tab.
function Message({ text }: { text: string }) {
  return (
    <div className="p-[var(--space-4)] text-[14px] font-normal leading-[1.5] text-text-secondary">
      {text}
    </div>
  );
}

export default function BytecodePanel({
  className,
  // the compile result carries the disassembly string; null before the
  // first compile. state carries the ip the bytecode highlight keys on;
  // null before the first step. arm64 carries the ARM64 backend's
  // output; null before the first compile. dirty true means both the
  // disassembly and the assembly are stale.
  compile = null,
  state = null,
  arm64 = null,
  dirty = false,
}: {
  className?: string;
  compile?: CompileResult | null;
  state?: VmState | null;
  arm64?: Arm64Result | null;
  dirty?: boolean;
}) {
  // the active tab is component-local UI state, default "bytecode" -- it
  // is not execution state, so it does not live in useQala.
  const [activeTab, setActiveTab] = useState<OutputTab>("bytecode");

  // the two tabs in display order. arrow-key navigation walks this list.
  const tabs: { id: OutputTab; label: string }[] = [
    { id: "bytecode", label: "Bytecode" },
    { id: "arm64", label: "ARM64" },
  ];

  // left/right arrow keys move between tabs, the native tablist pattern.
  const handleTabKeyDown = (event: React.KeyboardEvent) => {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
    event.preventDefault();
    const index = tabs.findIndex((tab) => tab.id === activeTab);
    const delta = event.key === "ArrowRight" ? 1 : -1;
    const next = (index + delta + tabs.length) % tabs.length;
    setActiveTab(tabs[next].id);
  };

  return (
    <section
      aria-label="Output"
      className={`flex flex-col overflow-hidden rounded-[var(--radius-panel)] border border-border bg-surface ${className ?? ""}`}
    >
      {/* the header hosts the tab strip; the page's collapse control is
          positioned over it. the tabs reuse the mobile tab vocabulary. */}
      <header className="border-b border-border p-[var(--space-3)]">
        <div role="tablist" className="flex gap-[var(--space-2)]">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              type="button"
              role="tab"
              aria-selected={activeTab === tab.id}
              tabIndex={activeTab === tab.id ? 0 : -1}
              onClick={() => setActiveTab(tab.id)}
              onKeyDown={handleTabKeyDown}
              className={`rounded-[var(--radius-button)] border px-[var(--space-3)] py-[var(--space-1)] text-[12px] font-medium uppercase tracking-wide leading-[1.4] transition-colors duration-150 ease-out focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent ${
                activeTab === tab.id
                  ? "border-accent text-accent"
                  : "border-border text-text-secondary hover:border-text-secondary"
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </header>
      {/* the body is the active tab's content. it is a single tabpanel
          -- the inactive tab is unmounted, so the bytecode tab's
          scroll-into-view never fires for an off-screen panel. */}
      <div role="tabpanel" className="flex min-h-0 flex-1 flex-col">
        {activeTab === "bytecode" ? (
          <BytecodeTab compile={compile} state={state} dirty={dirty} />
        ) : (
          <Arm64Tab arm64={arm64} dirty={dirty} />
        )}
      </div>
    </section>
  );
}
