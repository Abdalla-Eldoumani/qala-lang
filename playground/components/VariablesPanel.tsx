// the variables panel: the current frame's in-scope variables, listed
// as name : value rows, with an amber flash on a row whose value
// changed. phase 7 shipped this as a styled placeholder; phase 8 fills
// the body with the real view. the panel frame and the Variables header
// chrome are kept from phase 7 unchanged.
//
// the flash is pure css -- the @keyframes var-flash lives in globals.css
// and this panel only applies the .var-row--flashing class. a css
// animation replays only when its element mounts or remounts, so the
// flash is driven by the row's key: a row keyed by name plus a
// per-change counter remounts when the counter advances, which restarts
// the animation cleanly even mid-flash (a fast run loop). a row whose
// value did not change keeps its key, does not remount, and stays quiet.

import { useEffect, useRef, useState } from "react";
import type { NamedValue, VmState } from "@/lib/types";

// one variable row: the name, a colon separator, the rendered value, and
// a quiet trailing type tag. it always carries the var-flash class; the
// animation plays once per mount, so whether it flashes is decided
// entirely by the parent's keying (a new key remounts and replays).
function VariableRow({ variable }: { variable: NamedValue }) {
  return (
    <div className="var-row--flashing flex items-baseline gap-[var(--space-1)] px-[var(--space-3)] py-[var(--space-1)] font-mono text-[13px] font-normal leading-[1.6]">
      <span className="shrink-0 text-text-secondary">{variable.name}</span>
      <span className="shrink-0 text-text-secondary">:</span>
      <span className="min-w-0 flex-1 truncate text-text-primary">
        {variable.value.rendered}
      </span>
      {/* the type tag: a quiet reference chip, the one phase 8 use of the
          2px tag radius. Label type (the body font), --text-secondary,
          a --surface-raised fill so it reads as a tag, not as text. */}
      <span className="shrink-0 rounded-[var(--radius-tag)] bg-surface px-[var(--space-1)] text-[12px] font-medium leading-[1.4] text-text-secondary">
        {variable.value.type_name}
      </span>
    </div>
  );
}

export default function VariablesPanel({
  className,
  // state carries the current frame's variables; null before the first
  // step (or after a Reset). dirty true means the variables are stale.
  // these are optional: the phase 7 page renders this panel chrome-only,
  // and plan 08-05 wires the live snapshots. the defaults are the empty,
  // not-stepping, not-dirty state.
  state = null,
  dirty = false,
}: {
  className?: string;
  state?: VmState | null;
  dirty?: boolean;
}) {
  // the per-variable change counter, name -> count. a row's key is
  // `${name}:${count}`; incrementing a count remounts that row and
  // restarts its flash. this is state because the render reads it to
  // build the keys -- the effect bumps it and the re-render flashes the
  // changed rows.
  const [flashCounts, setFlashCounts] = useState<Map<string, number>>(
    new Map(),
  );

  // the previous frame's variables, kept in a ref so storing it does not
  // itself trigger a render. the diff effect reads it, finds the changed
  // and newly-in-scope names, then writes the new variables here as the
  // next "previous".
  const prevVarsRef = useRef<NamedValue[]>([]);

  // the diff. on each new state it compares the previous variables to
  // the new ones: a name whose rendered value differs flashes; a name
  // not in the previous set is newly in scope and also flashes; a name
  // that left scope just unmounts. a frame change (the whole set swaps
  // on a call or return) is therefore not a sea of flashes -- only a
  // genuine value change or a newly-in-scope name bumps a counter.
  //
  // the counter bump is a setState; the react compiler forbids a
  // synchronous setState inside an effect body, so it is deferred to a
  // microtask -- the same visible result, one commit later, without the
  // cascading-render the rule guards against.
  useEffect(() => {
    const prevVars = prevVarsRef.current;
    const nextVars = state?.variables ?? [];

    // the previous rendered value for each name, for the change test.
    const prevRendered = new Map<string, string>();
    for (const variable of prevVars) {
      prevRendered.set(variable.name, variable.value.rendered);
    }

    // the names that should flash this step: a changed value, or a name
    // that was not in scope before.
    const flashing: string[] = [];
    for (const variable of nextVars) {
      const before = prevRendered.get(variable.name);
      if (before === undefined || before !== variable.value.rendered) {
        flashing.push(variable.name);
      }
    }

    prevVarsRef.current = nextVars;

    // nothing changed -- skip the setState entirely so a step that
    // touched no variable does not churn a render.
    if (flashing.length === 0) return;

    queueMicrotask(() => {
      setFlashCounts((current) => {
        const next = new Map(current);
        for (const name of flashing) {
          next.set(name, (next.get(name) ?? 0) + 1);
        }
        return next;
      });
    });
    // the effect keys on the state identity: the hook hands a fresh
    // VmState object per step, so a new step is a new reference.
  }, [state]);

  // the panel frame, kept distinct from the body so the chrome (frame,
  // header, bottom rule) renders identically in every state.
  const frame = (children: React.ReactNode) => (
    <section
      className={`flex flex-col overflow-hidden rounded-[var(--radius-panel)] border border-border bg-surface ${className ?? ""}`}
    >
      <header className="border-b border-border p-[var(--space-3)] text-[12px] font-medium uppercase tracking-wide leading-[1.4] text-text-secondary">
        Variables
      </header>
      {children}
    </section>
  );

  const message = (text: string) => (
    <div className="p-[var(--space-4)] text-[14px] font-normal leading-[1.5] text-text-secondary">
      {text}
    </div>
  );

  // dirty variables are stale: show the needs-compile sentence rather
  // than a frozen stale list.
  if (dirty) {
    return frame(message("variables are out of date -- compile to refresh"));
  }

  const variables = state?.variables ?? [];

  // empty: no in-scope variables in the current frame (not stepped yet,
  // or a frame with no locals).
  if (variables.length === 0) {
    return frame(message("in-scope variables appear here during stepping"));
  }

  return frame(
    <div className="flex-1 overflow-y-auto py-[var(--space-1)]">
      {variables.map((variable) => (
        // the key is name plus the change counter: stable while the
        // value holds (no remount, the row stays quiet), changing on a
        // value change (a remount restarts the var-flash). a name with
        // no counter yet -- a row on its very first render -- uses 0,
        // and the fresh mount itself plays the flash once.
        <VariableRow
          key={`${variable.name}:${flashCounts.get(variable.name) ?? 0}`}
          variable={variable}
        />
      ))}
    </div>,
  );
}
