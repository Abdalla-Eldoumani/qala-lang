"use client";

// the single playground hook. it owns the wasm lifecycle: loadQala runs
// once on mount, the session lives in a useRef (a rust-backed object, not
// react state -- it must not trigger a re-render or be reconstructed), and
// status drives the loading/ready/error machine.
//
// all execution state lives in rust. the hook keeps only the last bridge
// snapshots (lastCompile, lastRun, lastState) -- it never reconstructs or
// mirrors the stack, variables, or ip. compile/run/step/reset are no-ops
// while the session is null; with the status-gated buttons in the layout
// that gives the disabled-until-ready behavior.
//
// the hook also owns the animated run loop. "run" in the playground is a
// recursive-setTimeout step loop, not the bridge's instant run() -- the
// point of the playground is to watch execution, so the loop calls step()
// on a cadence and the speed slider sets the per-tick delay. the loop is
// imperative: play()/pause() functions, a useRef timer id, a single
// clearLoop teardown path. it is never a useEffect-driven loop.

import { useCallback, useEffect, useRef, useState } from "react";
import { loadQala, type QalaSession } from "./compiler";
import type {
  CompileResult,
  ReplResult,
  RunResult,
  StepResult,
  VmState,
} from "./types";

export type QalaStatus = "loading" | "ready" | "error";

// the run-loop speed: a three-stop discrete control. the slider writes one
// of these; the loop reads the matching ms delay from RUN_SPEED_MS.
export type RunSpeed = "slow" | "normal" | "fast";

// ms per tick for each speed stop. kept in one place so the slider and the
// loop never disagree on the mapping.
const RUN_SPEED_MS: Record<RunSpeed, number> = {
  slow: 600,
  normal: 250,
  fast: 80,
};

export interface UseQala {
  status: QalaStatus;
  source: string;
  setSource: (next: string) => void;
  compile: () => void;
  // the bridge's instant run(). kept for completeness; the Controls' Run
  // button drives the animated loop via play(), not this.
  run: () => void;
  step: () => void;
  reset: () => void;
  // the animated run loop. play() starts the recursive-setTimeout step
  // loop; pause() stops it. running is true while the loop is live.
  running: boolean;
  play: () => void;
  pause: () => void;
  // the run-loop speed stop and its setter. changing speed mid-run takes
  // effect on the next tick -- the loop is not restarted.
  speed: RunSpeed;
  setSpeed: (next: RunSpeed) => void;
  // dirty is true when the source has changed since the last successful
  // compile. setSource sets it; a successful compile clears it. a dirty
  // edit also stops the run loop synchronously.
  dirty: boolean;
  // evaluate one repl line through the bridge. returns the ReplResult, or
  // null when the session is not ready. the hook keeps no transcript --
  // that is the ConsolePanel's own state.
  replEval: (line: string) => ReplResult | null;
  // last bridge snapshots -- read-only views, never mirrored vm state
  lastCompile: CompileResult | null;
  lastRun: RunResult | StepResult | null;
  lastState: VmState | null;
  version: number;
}

export function useQala(): UseQala {
  const sessionRef = useRef<QalaSession | null>(null);
  const [status, setStatus] = useState<QalaStatus>("loading");
  // source keeps its "" initial state -- the SSR-safe default, identical on
  // server and client.
  const [source, setSourceState] = useState<string>("");
  const [lastCompile, setLastCompile] = useState<CompileResult | null>(null);
  const [lastRun, setLastRun] = useState<RunResult | StepResult | null>(null);
  const [lastState, setLastState] = useState<VmState | null>(null);
  const [version, setVersion] = useState(0);
  const [dirty, setDirty] = useState(false);

  // the run loop. the timer id lives in a ref so it survives re-renders.
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [running, setRunning] = useState(false);
  // runningRef mirrors `running` so the tick closure reads a current value.
  // a closure captures the `running` state from the render it was created
  // in, which goes stale; the ref is always current.
  const runningRef = useRef(false);
  // the speed setting drives two things: the `speed` state (so the slider
  // re-renders to the new stop) and speedRef (so the next tick reads the
  // new delay without a loop restart).
  const [speed, setSpeedState] = useState<RunSpeed>("normal");
  const speedRef = useRef<number>(RUN_SPEED_MS.normal);

  // load the wasm module once on mount. a load/instantiate failure flips
  // status to "error"; the layout renders the friendly message from that.
  useEffect(() => {
    let cancelled = false;
    loadQala()
      .then((session) => {
        // the component unmounted while loadQala was in flight (notably
        // under dev strict mode, which mounts/unmounts/mounts). the
        // resolved session is unreachable, so free it rather than leave
        // it for the FinalizationRegistry.
        if (cancelled) {
          session.free();
          return;
        }
        sessionRef.current = session;
        setStatus("ready");
      })
      .catch(() => {
        // the error object is discarded on purpose -- the failure surface
        // shows fixed plain-language copy, never an exception message.
        if (!cancelled) setStatus("error");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // clear the run-loop timer. safe to call when no timer is pending. this
  // is the single teardown path -- pause, reset, unmount, and the
  // dirty-edit path all call it, so there is no way to forget to null the
  // ref or reset the flag.
  const clearLoop = useCallback(() => {
    if (timerRef.current !== null) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
    runningRef.current = false;
    setRunning(false);
  }, []);

  // one run-loop tick: step the vm, apply the snapshot, decide whether to
  // schedule the next tick. the runningRef check makes a late tick (one
  // that fired after pause/unmount, a clearTimeout race) a no-op.
  //
  // the recursive schedule reads tickRef.current rather than `tick`
  // directly: a useCallback may not reference itself (its closure would
  // capture a stale copy), so a ref holds the current tick and the
  // setTimeout callback reads through it.
  const tickRef = useRef<() => void>(() => {});
  const tick = useCallback(() => {
    const s = sessionRef.current;
    if (!s || !runningRef.current) return;
    const result = s.step();
    setLastRun(result);
    setLastState(result.state);
    setVersion((v) => v + 1);
    // stop on halt or a runtime fault; otherwise schedule the next tick at
    // the current speed (read from the ref so a mid-run slider change
    // applies here without a restart).
    if (result.status === "ran") {
      timerRef.current = setTimeout(() => tickRef.current(), speedRef.current);
    } else {
      clearLoop();
    }
  }, [clearLoop]);
  // keep tickRef pointed at the current tick so the recursive schedule and
  // play() both call the live closure. synced in an effect -- a ref must
  // not be written during render.
  useEffect(() => {
    tickRef.current = tick;
  }, [tick]);

  // start the animated run loop. a no-op when there is no session or the
  // loop is already running.
  const play = useCallback(() => {
    const s = sessionRef.current;
    if (!s || runningRef.current) return;
    runningRef.current = true;
    setRunning(true);
    timerRef.current = setTimeout(() => tickRef.current(), speedRef.current);
  }, []);

  // stop the run loop.
  const pause = useCallback(() => {
    clearLoop();
  }, [clearLoop]);

  // clear the run-loop timer if the component using the hook unmounts.
  // this effect's only job is the cleanup -- the loop is started by a
  // play() call, never on mount, so strict mode does not double-start it.
  useEffect(() => {
    return () => clearLoop();
  }, [clearLoop]);

  // set the run-loop speed. updates the `speed` state (the slider position)
  // and speedRef (the delay the next tick reads).
  const setSpeed = useCallback((next: RunSpeed) => {
    setSpeedState(next);
    speedRef.current = RUN_SPEED_MS[next];
  }, []);

  // set the editor source. a user edit: marks the program dirty and stops
  // the run loop SYNCHRONOUSLY -- in this same call, not a later effect, so
  // no further tick steps the now-invalid program.
  const setSource = useCallback(
    (next: string) => {
      setSourceState(next);
      setDirty(true);
      clearLoop();
    },
    [clearLoop],
  );

  const compile = useCallback(() => {
    const s = sessionRef.current;
    if (!s) return;
    const result = s.compile(source);
    setLastCompile(result);
    setLastState(s.getState());
    setVersion((v) => v + 1);
    // a successful compile clears dirty -- the program now matches the
    // source. a failed compile leaves it dirty.
    if (result.ok) setDirty(false);
  }, [source]);

  const run = useCallback(() => {
    const s = sessionRef.current;
    if (!s) return;
    const result = s.run();
    setLastRun(result);
    setLastState(result.state);
    setVersion((v) => v + 1);
  }, []);

  const step = useCallback(() => {
    const s = sessionRef.current;
    // a belt-and-braces guard: the Controls disable Step while the loop
    // runs, but step() is still a no-op if running, so the manual step and
    // the loop never interleave.
    if (!s || runningRef.current) return;
    const result = s.step();
    setLastRun(result);
    setLastState(result.state);
    setVersion((v) => v + 1);
  }, []);

  const reset = useCallback(() => {
    const s = sessionRef.current;
    if (!s) return;
    // stop the run loop first, then reset the bridge and clear the
    // snapshots. reset does NOT set dirty -- editing invalidates the
    // program, a reset does not.
    clearLoop();
    s.reset();
    setLastCompile(null);
    setLastRun(null);
    setLastState(null);
    setVersion((v) => v + 1);
  }, [clearLoop]);

  const replEval = useCallback((line: string): ReplResult | null => {
    const s = sessionRef.current;
    if (!s) return null;
    return s.replEval(line);
  }, []);

  return {
    status,
    source,
    setSource,
    compile,
    run,
    step,
    reset,
    running,
    play,
    pause,
    speed,
    setSpeed,
    dirty,
    replEval,
    lastCompile,
    lastRun,
    lastState,
    version,
  };
}
