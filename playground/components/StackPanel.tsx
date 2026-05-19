// the stack panel: the playground's signature element, the animated
// type-tinted value stack. the body is the real value-stack view; the
// header carries the panel-header treatment shared by the state panels.
//
// the stack grows upward: index 0 (the bottom of the stack) sits at the
// bottom of the panel body, the top of stack sits highest. a pushed
// value slides in from the top edge; a popped value slides down and
// fades. the slide animations are pure css: the @keyframes live in
// globals.css and this panel only applies the classes. no animation
// library, no new dependency.
//
// the hard part is the pop. react unmounts an element the instant it
// leaves the array, which kills a leave animation before it can play.
// the fix: a "leaving" set holds each popped value with the stack-pop
// class until onAnimationEnd unmounts it. the leaving set is derived
// DURING RENDER: when the state reference changes, the diff against
// the previous snapshot runs in the render body and updates the leaving
// state in the same pass (the documented derive-state-during-render
// pattern, guarded by a state-identity check so it cannot loop). this
// is what makes the leaving row mount in the same commit the live row
// unmounts; a post-commit effect would mount it one render late and the
// slide-down would play after the value already vanished. a timeout
// fallback covers a non-firing animationend (a backgrounded tab): the
// fallback timers are armed in an effect (a side effect, not a render
// step), tracked in a ref, and cleared on supersede and on unmount so
// they cannot leak.

import { useCallback, useEffect, useRef, useState } from "react";
import type { StateValue, VmState } from "@/lib/types";

// the panel renders at most the top 20 entries; a deeper stack puts only
// 20 rows in the dom regardless of true depth, so the render cost is
// bounded no matter how deep the program's stack grows.
const WINDOW_SIZE = 20;

// the leave animation is 150ms (the stack-pop keyframe). the fallback
// timeout is a little longer so animationend wins the normal race; the
// fallback only fires when animationend never does.
const LEAVE_FALLBACK_MS = 200;

// a value that was popped and is mid-leave-animation. it is no longer in
// VmState.stack, so it is rendered from this record instead. the id is a
// stable per-leave key; slot is the stack position it was popped from,
// used only to order the leaving entries among themselves.
interface LeavingEntry {
  id: string;
  value: StateValue;
  slot: number;
}

// classify a runtime type name into one of the four tint classes. the
// match is on the three exact primitive names; every other name --
// arrays like [i64], tuples, structs, enum variants like Shape::Circle,
// void, route to the neutral tint without enumerating them. the tint
// classes themselves are defined in globals.css.
function tintClass(typeName: string): string {
  if (typeName === "i64" || typeName === "f64") return "tint-number";
  if (typeName === "str") return "tint-string";
  if (typeName === "bool") return "tint-bool";
  return "tint-other";
}

// the static graded opacity for a visible row when the true stack depth
// exceeds the 20-entry window. the three lowest visible rows (closest to
// the truncation edge) fade so the user reads "more is hidden below".
// indexFromBottom is the row's 0-based position in the visible window.
// this is a static opacity, not an animation; it returns 1 (no fade)
// whenever the stack is not truncated.
function depthOpacity(indexFromBottom: number, truncated: boolean): number {
  if (!truncated) return 1;
  if (indexFromBottom === 0) return 0.25;
  if (indexFromBottom === 1) return 0.4;
  if (indexFromBottom === 2) return 0.6;
  return 1;
}

// one live stack entry. it is keyed by its stack slot index in the
// parent, so a freshly-pushed slot mounts this fresh and the
// stack-entry--entering class animates it in exactly once. an in-place
// value change at an existing slot keeps the same key, so the component
// updates with no slide, which is correct: only a push or a pop slides.
function LiveEntry({
  value,
  opacity,
}: {
  value: StateValue;
  opacity: number;
}) {
  return (
    <div
      // the entering class carries the stack-push keyframe; it plays once
      // on mount and is harmless thereafter. the type tint is a flat
      // full-width band with no radius, per the surface contract. the
      // depth opacity is inline because it is a per-row computed value,
      // not one of a fixed set of classes.
      className={`stack-entry--entering flex items-baseline justify-between gap-[var(--space-3)] px-[var(--space-3)] py-[var(--space-1)] font-mono text-[13px] font-normal leading-[1.6] ${tintClass(
        value.type_name,
      )}`}
      style={{ opacity }}
    >
      <span className="min-w-0 truncate text-text-primary">
        {value.rendered}
      </span>
      <span className="shrink-0 text-text-secondary">{value.type_name}</span>
    </div>
  );
}

// one leaving entry: a value that was popped and is playing its
// slide-down. it carries the stack-pop keyframe via stack-entry--leaving
// and reports back through onDone when the animation ends.
function LeavingRow({
  entry,
  onDone,
}: {
  entry: LeavingEntry;
  onDone: (id: string) => void;
}) {
  return (
    <div
      className={`stack-entry--leaving flex items-baseline justify-between gap-[var(--space-3)] px-[var(--space-3)] py-[var(--space-1)] font-mono text-[13px] font-normal leading-[1.6] ${tintClass(
        entry.value.type_name,
      )}`}
      onAnimationEnd={() => onDone(entry.id)}
    >
      <span className="min-w-0 truncate text-text-primary">
        {entry.value.rendered}
      </span>
      <span className="shrink-0 text-text-secondary">
        {entry.value.type_name}
      </span>
    </div>
  );
}

export default function StackPanel({
  className,
  // state carries the value stack the panel renders; null before the
  // first step (or after a Reset). dirty true means the stack is stale.
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
  // the values that were popped and are mid-leave-animation. they are
  // rendered alongside the live stack until their animation ends.
  const [leaving, setLeaving] = useState<LeavingEntry[]>([]);

  // the state snapshot the leaving set was last derived against. this is
  // the anchor for the derive-during-render diff: a render where `state`
  // differs from this is a new step, and the diff runs in the render
  // body. it is state (not a ref) so the guarded update-during-render
  // settles consistently across react's render passes, and prevState's
  // stack IS the previous stack, so no separate previous-stack store.
  const [prevState, setPrevState] = useState<VmState | null>(null);

  // the fallback timeout ids for the leaving entries, keyed by leave id.
  // tracked so they can be cleared when superseded and on unmount: a
  // leaked animation-fallback timer is the exact failure this panel is
  // built to avoid.
  const fallbackTimersRef = useRef<Map<string, ReturnType<typeof setTimeout>>>(
    new Map(),
  );

  // a monotonic counter for unique leave ids. state, not a ref: the
  // render-phase diff reads it to build the ids, and the react-hooks
  // rules forbid reading a ref during render, so it is state, bumped
  // in the same guarded update-during-render block as the diff. it gives
  // each step's leaving entries a fresh key so the stack-pop keyframe
  // replays even when the same slot is popped on two consecutive steps.
  const [leaveSeq, setLeaveSeq] = useState(0);

  // remove a leaving entry once its animation has ended (or its fallback
  // fired) and drop its fallback timer. stable so the timer-arming effect
  // can depend on it without re-running every render.
  const finishLeave = useCallback((id: string) => {
    setLeaving((current) => current.filter((entry) => entry.id !== id));
    const timer = fallbackTimersRef.current.get(id);
    if (timer !== undefined) {
      clearTimeout(timer);
      fallbackTimersRef.current.delete(id);
    }
  }, []);

  // the diff, computed DURING RENDER. when `state` is a new reference --
  // the hook hands a fresh VmState per step, this compares the previous
  // snapshot's stack to the new one and updates the leaving set in the
  // same render pass. a shorter stack means the top (prev - next) slots
  // were popped: each popped value becomes a leaving entry with a fresh
  // id, walked low-to-high so the array is bottom-first like the live
  // entries (the column-reverse body then renders the highest popped
  // slot highest). a push, an in-place update, a Reset, or a null state
  // produces no leaving entries, so the set is emptied: a Reset is an
  // instant clear, not a sequence of pops.
  //
  // the setState calls run during render but are GUARDED by the
  // state-identity check, so they run once per new step and cannot loop
  // (the documented derive-state-during-render pattern). react re-renders
  // synchronously before paint, so the leaving row is in the very commit
  // that drops the popped slot's live row.
  //
  // known limitation: the diff keys on stack depth and slot index, so an
  // instruction that both pops and pushes (a binary op pops two operands
  // and pushes one result, net depth -1) is seen only as a pop: one
  // leave animation plays and the pushed result, landing on a slot that
  // already held an operand, updates in place with no push slide. the
  // net height is right; the mixed instruction is just quieter than a
  // bare push or bare pop. animating it would need value identity or a
  // VM-provided per-slot id; index keys cannot express it.
  if (state !== prevState) {
    const prevStack = prevState?.stack ?? [];
    const nextStack = state?.stack ?? [];
    const popped: LeavingEntry[] = [];
    if (nextStack.length > 0 && nextStack.length < prevStack.length) {
      for (let slot = nextStack.length; slot < prevStack.length; slot++) {
        // the id pairs this step's sequence number with the slot: the
        // slot is unique among one step's pops, the sequence number
        // separates this step from the next, so a slot popped on two
        // consecutive steps gets two distinct keys and two animations.
        popped.push({
          id: `leave-${leaveSeq}-${slot}`,
          value: prevStack[slot],
          slot,
        });
      }
    }
    setPrevState(state);
    setLeaving(popped);
    // advance the sequence only when this step actually popped, so the
    // counter tracks pops, not every step.
    if (popped.length > 0) setLeaveSeq((seq) => seq + 1);
  }

  // arm the fallback timers for the current leaving set. this is a side
  // effect (it arms a setTimeout, not a setState) so it belongs in an
  // effect, not the render body. it diffs the live leaving ids against
  // the timer map: it clears a timer whose entry has left (superseded or
  // already finished) and arms one for any leaving entry that lacks one,
  // so an entry mid-animation keeps its original fallback clock.
  useEffect(() => {
    const timers = fallbackTimersRef.current;
    const liveIds = new Set(leaving.map((entry) => entry.id));
    for (const [id, timer] of timers) {
      if (!liveIds.has(id)) {
        clearTimeout(timer);
        timers.delete(id);
      }
    }
    for (const entry of leaving) {
      if (!timers.has(entry.id)) {
        timers.set(
          entry.id,
          setTimeout(() => finishLeave(entry.id), LEAVE_FALLBACK_MS),
        );
      }
    }
  }, [leaving, finishLeave]);

  // clear the fallback timers on unmount so a navigated-away panel
  // leaves nothing running.
  useEffect(() => {
    const timers = fallbackTimersRef.current;
    return () => {
      for (const timer of timers.values()) {
        clearTimeout(timer);
      }
      timers.clear();
    };
  }, []);

  // the panel frame, kept distinct from the body so the chrome (frame,
  // header, bottom rule) renders identically in every state.
  const frame = (children: React.ReactNode) => (
    <section
      aria-label="Stack"
      className={`flex flex-col overflow-hidden rounded-[var(--radius-panel)] border border-border bg-surface ${className ?? ""}`}
    >
      <header className="border-b border-border px-[var(--space-3)] py-[var(--space-2)] text-[12px] font-semibold uppercase tracking-wide leading-[1.4] text-text-secondary">
        Stack
      </header>
      {children}
    </section>
  );

  const message = (text: string) => (
    <div className="px-[var(--space-4)] pt-[var(--space-6)] pb-[var(--space-4)] text-[14px] font-normal leading-[1.5] text-text-secondary">
      {text}
    </div>
  );

  // a dirty stack is stale: show the needs-compile sentence rather than a
  // frozen stale stack.
  if (dirty) {
    return frame(message("the stack is out of date, compile to refresh"));
  }

  const stack = state?.stack ?? [];

  // empty: compiled but not stepped, or every value popped. an in-flight
  // leave animation can still be playing after the last pop, so the
  // empty sentence shows only when there is also nothing leaving.
  if (stack.length === 0 && leaving.length === 0) {
    return frame(message("the value stack appears here as the program runs"));
  }

  // the visible window: the top 20 entries. when the stack is deeper, the
  // lower entries are not in the dom at all and the bottom three of the
  // window get the static graded fade.
  const truncated = stack.length > WINDOW_SIZE;
  const windowStart = truncated ? stack.length - WINDOW_SIZE : 0;
  // bottom-to-top slice of slots that are in the visible window.
  const visibleSlots: number[] = [];
  for (let slot = windowStart; slot < stack.length; slot++) {
    visibleSlots.push(slot);
  }

  return frame(
    // column-reverse puts slot 0 at the visual bottom and the top of
    // stack highest; justify-end keeps a short stack pinned to the
    // bottom. overflow-hidden (inherited from the frame) clips a pushed
    // entry so it appears to emerge from the top edge.
    <div className="flex flex-1 flex-col-reverse justify-end overflow-hidden">
      {/* the live entries, rendered bottom-of-stack first so
          column-reverse places them with slot 0 at the bottom. each is
          keyed by its stack slot index: a new slot mounts and animates
          in, a removed slot unmounts into the leaving path. */}
      {visibleSlots.map((slot, indexFromBottom) => (
        <LiveEntry
          key={`slot-${slot}`}
          value={stack[slot]}
          opacity={depthOpacity(indexFromBottom, truncated)}
        />
      ))}
      {/* the leaving entries sit above the live stack (last in a
          column-reverse list renders highest). they are keyed by their
          stable leave id so react keeps them mounted through the
          slide-down. */}
      {leaving.map((entry) => (
        <LeavingRow key={entry.id} entry={entry} onDone={finishLeave} />
      ))}
    </div>,
  );
}
