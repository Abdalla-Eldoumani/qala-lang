// the stack panel: the playground's signature element, the animated
// type-tinted value stack. phase 7 shipped this as a styled placeholder;
// phase 8 fills the body with the real view. the panel frame and the
// Stack header chrome are kept from phase 7 unchanged.
//
// the stack grows upward: index 0 (the bottom of the stack) sits at the
// bottom of the panel body, the top of stack sits highest. a pushed
// value slides in from the top edge; a popped value slides down and
// fades. the slide animations are pure css -- the @keyframes live in
// globals.css and this panel only applies the classes. no animation
// library, no new dependency.
//
// the hard part is the pop. react unmounts an element the instant it
// leaves the array, which kills a leave animation before it can play.
// the fix: a "leaving" set holds each popped value one extra render with
// the stack-pop class; onAnimationEnd unmounts it. a timeout fallback
// covers a non-firing animationend (a backgrounded tab) -- the fallback
// timers are tracked in a ref and cleared on every diff and on unmount
// so they cannot leak.

import { useEffect, useRef, useState } from "react";
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
// void -- routes to the neutral tint without enumerating them. the tint
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
// updates with no slide -- which is correct: only a push or a pop slides.
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

  // the previous stack, kept in a ref so storing it does not itself
  // trigger a render. the diff effect reads it, computes the difference,
  // then writes the new stack here as the next "previous".
  const prevStackRef = useRef<StateValue[]>([]);

  // the fallback timeout ids for the leaving entries, keyed by leave id.
  // tracked so they can be cleared on every diff and on unmount -- a
  // leaked animation-fallback timer is the exact failure this panel is
  // built to avoid.
  const fallbackTimersRef = useRef<Map<string, ReturnType<typeof setTimeout>>>(
    new Map(),
  );

  // a monotonic counter for unique leave ids. a ref, not state -- it must
  // not trigger a render when it advances.
  const leaveSeqRef = useRef(0);

  // clear and forget every pending fallback timer. called before each
  // diff (the timers belong to the prior diff's leaving entries) and on
  // unmount.
  function clearFallbackTimers() {
    for (const timer of fallbackTimersRef.current.values()) {
      clearTimeout(timer);
    }
    fallbackTimersRef.current.clear();
  }

  // remove a leaving entry once its animation has ended (or its fallback
  // fired) and drop its fallback timer.
  function finishLeave(id: string) {
    setLeaving((current) => current.filter((entry) => entry.id !== id));
    const timer = fallbackTimersRef.current.get(id);
    if (timer !== undefined) {
      clearTimeout(timer);
      fallbackTimersRef.current.delete(id);
    }
  }

  // the diff. it runs after each new state: it compares the previous
  // stack to the new one and drives the leave animations. a longer stack
  // is a push -- the push slide is free, the LiveEntry mounts with the
  // entering class. a shorter stack is a pop -- the popped values move
  // into the leaving set. an instant clear (a Reset, a null state) drops
  // the leaving set with no animation.
  //
  // the diff is computed synchronously here -- including the synchronous
  // clearFallbackTimers and prevStackRef write, so a fast re-entrant
  // step always sees a correct previous -- but the resulting setLeaving
  // is deferred to a microtask. the react compiler forbids a synchronous
  // setState inside an effect body (it can cascade renders); a
  // queueMicrotask runs the update right after this commit, which is the
  // same visible result without the cascade.
  useEffect(() => {
    const prevStack = prevStackRef.current;
    const nextStack = state?.stack ?? [];

    // the prior diff's fallback timers belong to entries that are about
    // to be superseded; clear them so they cannot fire late.
    clearFallbackTimers();

    // a Reset or an error surfaces as a null or empty state. clear the
    // leaving set with no slide-down -- a Reset is an instant clear, not
    // a sequence of pops. a shorter stack means the top (prev - next)
    // slots were popped: each popped value becomes a leaving entry with
    // a fresh id, walked low-to-high so the array is bottom-first like
    // the live entries (the column-reverse body then renders the highest
    // popped slot highest). a push or an in-place update has no leave
    // animation, so the leaving set is simply emptied.
    const popped: LeavingEntry[] = [];
    if (nextStack.length > 0 && nextStack.length < prevStack.length) {
      for (let slot = nextStack.length; slot < prevStack.length; slot++) {
        leaveSeqRef.current += 1;
        popped.push({
          id: `leave-${leaveSeqRef.current}`,
          value: prevStack[slot],
          slot,
        });
      }
    }

    // apply the state change after this commit, then arm the fallback
    // timers for exactly the entries that were committed as leaving.
    queueMicrotask(() => {
      setLeaving(popped);
      for (const entry of popped) {
        const timer = setTimeout(
          () => finishLeave(entry.id),
          LEAVE_FALLBACK_MS,
        );
        fallbackTimersRef.current.set(entry.id, timer);
      }
    });

    prevStackRef.current = nextStack;
    // the effect keys on the state identity: the hook hands a fresh
    // VmState object per step, so a new step is a new reference.
  }, [state]);

  // clear the fallback timers on unmount so a navigated-away panel
  // leaves nothing running.
  useEffect(() => {
    return () => clearFallbackTimers();
  }, []);

  // the panel frame, kept distinct from the body so the chrome (frame,
  // header, bottom rule) renders identically in every state.
  const frame = (children: React.ReactNode) => (
    <section
      className={`flex flex-col overflow-hidden rounded-[var(--radius-panel)] border border-border bg-surface ${className ?? ""}`}
    >
      <header className="border-b border-border p-[var(--space-3)] text-[12px] font-medium uppercase tracking-wide leading-[1.4] text-text-secondary">
        Stack
      </header>
      {children}
    </section>
  );

  const message = (text: string) => (
    <div className="p-[var(--space-4)] text-[14px] font-normal leading-[1.5] text-text-secondary">
      {text}
    </div>
  );

  // a dirty stack is stale: show the needs-compile sentence rather than a
  // frozen stale stack.
  if (dirty) {
    return frame(message("stack is out of date -- compile to refresh"));
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
