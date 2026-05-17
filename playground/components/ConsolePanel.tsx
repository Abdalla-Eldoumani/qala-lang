"use client";

// the console panel: two modes driven entirely by whether a program is
// compiled. program mode shows VmState.console -- the program's stdout --
// as monospace lines with sticky-bottom auto-scroll. repl mode (no
// program compiled) is a single-expression repl: a pinned input row plus
// an accumulated transcript, each line evaluated through the hook's
// replEval. phase 7 shipped this as a styled placeholder; phase 8 fills
// the body. the panel frame and the Console header chrome are kept from
// phase 7 unchanged.
//
// the mode is decided by the programCompiled prop the page derives from
// useQala's lastCompile (compiled and ok). the user never switches modes
// by hand -- compiling a program is the switch.

import { useEffect, useRef, useState } from "react";
import type { MonacoDiagnostic, ReplResult, VmState } from "@/lib/types";

// one entry in the repl transcript: the line the user typed and the
// result the bridge returned. the transcript is component state, kept
// here and not on the hook; it is unbounded by design (no cap, no clear
// button) -- a single-user, single-session, browser-local list.
interface ReplEntry {
  // a stable per-entry key. the input echo and the result render under it.
  id: number;
  input: string;
  result: ReplResult;
}

// a value was produced when the bridge returned an ok result carrying a
// value. ReplResult.value is a StateValue, so the rendered text is
// value.rendered. a faulted line has ok false and an error instead.
function replValueText(result: ReplResult): string | null {
  if (result.value !== undefined) return result.value.rendered;
  return null;
}

export default function ConsolePanel({
  className,
  // the current VM-state snapshot; console is read from it. null before
  // the first step or after a Reset.
  state = null,
  // the last run/step snapshot. its optional error field, when present,
  // is the runtime fault appended as the final program-mode line.
  lastRun = null,
  // true when a program is compiled. drives the mode: true -> program
  // output, false -> the repl. the page derives this from lastCompile.
  programCompiled = false,
  // the hook's repl method. evaluates one line through the bridge and
  // returns a ReplResult, or null when the WASM session is not ready.
  replEval,
}: {
  className?: string;
  state?: VmState | null;
  lastRun?: { error?: MonacoDiagnostic } | null;
  programCompiled?: boolean;
  replEval?: (line: string) => ReplResult | null;
}) {
  // the scrolling body. auto-scroll reads and writes its scrollTop.
  const bodyRef = useRef<HTMLDivElement | null>(null);

  // whether the body was at (or near) the bottom before the latest
  // content change. sticky-bottom: hold the bottom only while the user
  // has not scrolled up. a ref, not state -- it is read in a layout
  // effect and must not trigger a render.
  const atBottomRef = useRef(true);

  // the repl transcript -- one entry per evaluated line. component state,
  // unbounded by design (UI-SPEC R3). a monotonic counter gives each
  // entry a stable key without an index.
  const [transcript, setTranscript] = useState<ReplEntry[]>([]);
  const replSeqRef = useRef(0);

  // the current repl input text -- a controlled input.
  const [replInput, setReplInput] = useState("");

  // the program-mode lines: the console output, plus a final runtime-error
  // line when the last run/step faulted. error is read with truthiness --
  // a rust None arrives as undefined, never null.
  const consoleLines = state?.console ?? [];
  const runtimeError = lastRun?.error ? lastRun.error.message : null;

  // record whether the body is scrolled to the bottom BEFORE react commits
  // the new content, so the post-commit effect knows whether to stick. a
  // 2px slack absorbs sub-pixel rounding. this layout effect runs before
  // paint on every render; it only reads the dom, it sets no state.
  useEffect(() => {
    const body = bodyRef.current;
    if (body === null) return;
    atBottomRef.current =
      body.scrollHeight - body.scrollTop - body.clientHeight < 2;
  });

  // after the content changes, stick to the bottom only if the user was
  // already there. keyed on the program-mode line count and the runtime
  // error so a new output line scrolls, and on the transcript length so a
  // new repl entry scrolls. the user scrolling up pauses the stick until
  // they return to the bottom.
  useEffect(() => {
    const body = bodyRef.current;
    if (body === null) return;
    if (atBottomRef.current) body.scrollTop = body.scrollHeight;
  }, [consoleLines.length, runtimeError, transcript.length]);

  // evaluate the current repl line. called on Enter. a blank line is
  // ignored. the result is appended to the transcript and the input is
  // cleared. this is an event handler -- the setState here is allowed
  // (the react compiler only forbids a synchronous setState in an effect).
  function evaluateReplLine() {
    const line = replInput.trim();
    if (line === "" || replEval === undefined) return;
    const result = replEval(line);
    // replEval returns null only when the session is not ready; there is
    // nothing to record in that case.
    if (result === null) return;
    replSeqRef.current += 1;
    setTranscript((current) => [
      ...current,
      { id: replSeqRef.current, input: line, result },
    ]);
    setReplInput("");
  }

  // the panel frame, kept distinct from the body so the chrome renders
  // identically in both modes.
  const frame = (children: React.ReactNode) => (
    <section
      className={`flex flex-col overflow-hidden rounded-[var(--radius-panel)] border border-border bg-surface ${className ?? ""}`}
    >
      <header className="border-b border-border p-[var(--space-3)] text-[12px] font-medium uppercase tracking-wide leading-[1.4] text-text-secondary">
        Console
      </header>
      {children}
    </section>
  );

  // program mode: render VmState.console as monospace lines, with a final
  // --error line on a runtime fault. needs-compile does NOT apply here --
  // console output is a true historical record, so an edit keeps the last
  // run's output rather than blanking to a stale sentence. a Reset clears
  // state, which empties the lines back to the empty-state sentence.
  if (programCompiled) {
    const empty = consoleLines.length === 0 && runtimeError === null;
    return frame(
      <div
        ref={bodyRef}
        className="min-h-0 flex-1 overflow-y-auto py-[var(--space-1)]"
      >
        {empty ? (
          <p className="px-[var(--space-4)] py-[var(--space-3)] text-[14px] font-normal leading-[1.5] text-text-secondary">
            program output appears here
          </p>
        ) : (
          <>
            {consoleLines.map((line, i) => (
              <p
                // the console is an append-only list; the index is a
                // stable key because earlier lines never reorder.
                key={`out-${i}`}
                className="whitespace-pre-wrap break-words px-[var(--space-3)] py-[var(--space-1)] font-mono text-[13px] font-normal leading-[1.6] text-text-primary"
              >
                {line}
              </p>
            ))}
            {runtimeError !== null && (
              // the runtime fault: a single final line, the plain
              // diagnostic message with a fixed prefix. no stack trace,
              // no error code, no raw exception -- only .message.
              <p className="whitespace-pre-wrap break-words px-[var(--space-3)] py-[var(--space-1)] font-mono text-[13px] font-normal leading-[1.6] text-error">
                {`runtime error: ${runtimeError}`}
              </p>
            )}
          </>
        )}
      </div>,
    );
  }

  // repl mode: the transcript scrolls in the body, the input row is pinned
  // at the bottom. the input row sits outside the scrolling body so it
  // stays visible while the transcript scrolls -- on mobile it stays put
  // when the keyboard opens (the panel body scrolls, the row does not).
  return frame(
    <div className="flex min-h-0 flex-1 flex-col">
      <div
        ref={bodyRef}
        className="min-h-0 flex-1 overflow-y-auto py-[var(--space-1)]"
      >
        {transcript.length === 0 ? (
          <p className="px-[var(--space-4)] py-[var(--space-3)] text-[14px] font-normal leading-[1.5] text-text-secondary">
            repl ready -- type a Qala expression and press enter
          </p>
        ) : (
          transcript.map((entry) => (
            <div key={entry.id} className="py-[var(--space-1)]">
              {/* the echoed input: quieter, it is history -- the thing the
                  user asked, in --text-secondary after a > prompt. */}
              <p className="flex gap-[var(--space-2)] px-[var(--space-3)] font-mono text-[13px] font-normal leading-[1.6] text-text-secondary">
                <span className="shrink-0">{">"}</span>
                <span className="min-w-0 whitespace-pre-wrap break-words">
                  {entry.input}
                </span>
              </p>
              {/* the result: the thing the system produced, prominent in
                  --text-primary -- a produced value, then any console
                  lines, then a runtime-error line if the line faulted. */}
              {replValueText(entry.result) !== null && (
                <p className="whitespace-pre-wrap break-words px-[var(--space-3)] font-mono text-[13px] font-normal leading-[1.6] text-text-primary">
                  {replValueText(entry.result)}
                </p>
              )}
              {entry.result.console.map((line, i) => (
                <p
                  key={`c-${i}`}
                  className="whitespace-pre-wrap break-words px-[var(--space-3)] font-mono text-[13px] font-normal leading-[1.6] text-text-primary"
                >
                  {line}
                </p>
              ))}
              {entry.result.error !== undefined && (
                <p className="whitespace-pre-wrap break-words px-[var(--space-3)] font-mono text-[13px] font-normal leading-[1.6] text-error">
                  {`runtime error: ${entry.result.error.message}`}
                </p>
              )}
            </div>
          ))
        )}
      </div>
      {/* the pinned input row: a > prompt glyph then a single-line input.
          Enter evaluates the line. it sits below the scrolling body so it
          holds the bottom of the panel at every width. */}
      <div className="flex shrink-0 items-center gap-[var(--space-2)] border-t border-border px-[var(--space-3)] py-[var(--space-2)]">
        <span className="shrink-0 font-mono text-[13px] font-normal leading-[1.6] text-text-secondary">
          {">"}
        </span>
        <input
          type="text"
          value={replInput}
          onChange={(event) => setReplInput(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") {
              event.preventDefault();
              evaluateReplLine();
            }
          }}
          aria-label="repl input -- type a Qala expression"
          className="min-w-0 flex-1 rounded-[var(--radius-button)] border border-border bg-surface px-[var(--space-3)] py-[var(--space-2)] font-mono text-[13px] font-normal leading-[1.6] text-text-primary focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
        />
      </div>
    </div>,
  );
}
