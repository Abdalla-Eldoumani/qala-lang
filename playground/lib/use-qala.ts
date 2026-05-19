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
  Arm64Result,
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

// the localStorage key the editor source is persisted under. text only --
// never vm state (PGUI-09).
const STORAGE_KEY = "qala.editor.source";

export interface UseQala {
  status: QalaStatus;
  source: string;
  setSource: (next: string) => void;
  // compile the current source through the bridge. returns true on a
  // successful compile, false on a compile error or a not-ready session,
  // so a compile-then-run caller can gate the run on the result.
  compile: () => boolean;
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
  // dirty is true when the source does not match the last successful
  // compile -- including before the first compile, so it starts true.
  // setSource sets it; a successful compile clears it. a dirty edit also
  // stops the run loop synchronously.
  dirty: boolean;
  // finished is true when the last step or run reached Halt -- the
  // program is done. it gates Step and Run off in the Controls so a
  // manual Step past the end does not fault the VM. a Reset clears it.
  finished: boolean;
  // evaluate one repl line through the bridge. returns the ReplResult, or
  // null when the session is not ready. the hook keeps no transcript --
  // that is the ConsolePanel's own state.
  replEval: (line: string) => ReplResult | null;
  // the currently loaded example name, or null when the editor holds the
  // user's own text. session state -- not derived by content-matching.
  loadedExample: string | null;
  loadExample: (file: string) => void;
  // last bridge snapshots -- read-only views, never mirrored vm state
  lastCompile: CompileResult | null;
  lastRun: RunResult | StepResult | null;
  lastState: VmState | null;
  // the ARM64 backend's last output, refreshed in the same compile() call
  // as lastCompile -- so the ARM64 view and the bytecode view always
  // reflect the same source. null before the first compile.
  lastArm64: Arm64Result | null;
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
  const [lastArm64, setLastArm64] = useState<Arm64Result | null>(null);
  const [version, setVersion] = useState(0);
  // dirty starts true: at mount no compile has happened, so the source --
  // the "" initial, the restored text, or the first-visit preload, all
  // applied without a compile -- does not match any bytecode. that is the
  // exact meaning of dirty (source differs from the last successful
  // compile), and it is what keeps Run enabled on a fresh load so the
  // first compile-then-run is reachable. the first successful compile
  // clears it.
  const [dirty, setDirty] = useState(true);
  const [loadedExample, setLoadedExample] = useState<string | null>(null);

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

  // set the editor source. a user edit: marks the program dirty, clears the
  // loaded-example name (the text is no longer that example), and stops the
  // run loop SYNCHRONOUSLY -- in this same call, not a later effect, so no
  // further tick steps the now-invalid program.
  const setSource = useCallback(
    (next: string) => {
      setSourceState(next);
      setDirty(true);
      setLoadedExample(null);
      clearLoop();
    },
    [clearLoop],
  );

  // load a bundled example into the editor. fetches the .qala file, sets
  // the source (which marks dirty, stops the loop, and clears
  // loadedExample), then sets loadedExample AFTER -- so the example name is
  // set last and survives setSource's clear.
  const loadExample = useCallback(
    (file: string) => {
      fetch(`/examples/${file}.qala`)
        .then((res) => res.text())
        .then((text) => {
          setSource(text);
          setLoadedExample(file);
        })
        .catch((error: unknown) => {
          // a failed fetch leaves the editor as it was -- there is no
          // user-facing surface for it (the examples are bundled assets,
          // a failure is a transient network error). log it so a
          // transient failure is still diagnosable from the console.
          console.warn(`could not load the ${file} example`, error);
        });
    },
    [setSource],
  );

  // read the editor source from localStorage on mount. a client-only
  // effect -- localStorage is browser-only, so reading it during render
  // would cause a hydration mismatch. setSourceState here bypasses the
  // dirty/loaded-example side effects of setSource (a restore is not an
  // edit).
  //
  // both branches apply the source asynchronously -- the restore through a
  // microtask, the first-visit fetch through its promise -- so no setState
  // runs synchronously inside the effect body.
  useEffect(() => {
    const saved = window.localStorage.getItem(STORAGE_KEY);
    if (saved !== null) {
      // a return visit -- the key is present. restore the text verbatim,
      // even if it is "" (a user who cleared the editor and reloaded gets
      // their empty editor back, not fibonacci).
      queueMicrotask(() => {
        setSourceState(saved);
      });
    } else {
      // first visit -- the key is ABSENT (not ""). preload the fibonacci
      // example so the editor is never empty on a fresh visit. the write
      // effect below then persists it, so the next visit is a return visit.
      fetch("/examples/fibonacci.qala")
        .then((res) => res.text())
        .then((text) => {
          setSourceState(text);
          setLoadedExample("fibonacci");
        })
        .catch((error: unknown) => {
          // the fetch failed -- leave the editor empty, no user-facing
          // surface. log it so a failed first-visit preload is
          // diagnosable rather than a silent blank editor.
          console.warn("could not preload the fibonacci example", error);
        });
    }
  }, []);

  // persist the editor source to localStorage whenever it changes. an
  // immediate write per change -- the writes are small text writes.
  useEffect(() => {
    window.localStorage.setItem(STORAGE_KEY, source);
  }, [source]);

  // compile returns whether the compile succeeded so a caller in the same
  // tick can gate on it -- the `lastCompile` state set here is not visible
  // until the next render, so a return value is the only in-tick signal.
  const compile = useCallback((): boolean => {
    const s = sessionRef.current;
    if (!s) return false;
    const result = s.compile(source);
    setLastCompile(result);
    // refresh the ARM64 snapshot in the same tick, against the same
    // source, so the ARM64 tab and the bytecode tab never disagree.
    setLastArm64(s.compileArm64(source));
    setLastState(s.getState());
    setVersion((v) => v + 1);
    // a successful compile clears dirty -- the program now matches the
    // source. a failed compile leaves it dirty.
    if (result.ok) setDirty(false);
    return result.ok;
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
    // stop the run loop first, then reset the bridge. reset does NOT set
    // dirty -- editing invalidates the program, a reset does not.
    clearLoop();
    // the bridge reset nulls both the program and the VM, so a re-run
    // after a reset needs a fresh compile. when a program was compiled,
    // recompile the current source so the bridge holds a fresh VM at the
    // entry point and keep lastCompile -- the program stays re-runnable,
    // the controls stay enabled, and the console stays in program mode.
    // when no program was ever compiled there is nothing to rebuild.
    s.reset();
    if (lastCompile?.ok) {
      setLastCompile(s.compile(source));
      // keep the two tabs in lockstep across a reset -- the assembly is
      // static compiler output, unaffected by a VM reset, but a recompile
      // here keeps the ARM64 snapshot consistent with lastCompile.
      setLastArm64(s.compileArm64(source));
      setLastState(s.getState());
    } else {
      setLastCompile(null);
      setLastArm64(null);
      setLastState(null);
    }
    setLastRun(null);
    setVersion((v) => v + 1);
  }, [clearLoop, lastCompile, source]);

  const replEval = useCallback((line: string): ReplResult | null => {
    const s = sessionRef.current;
    if (!s) return null;
    return s.replEval(line);
  }, []);

  // the program has finished when the last step or run reached Halt. a
  // StepResult carries a "status" field and halts at status "halted"; a
  // RunResult has no "status" and ran to Halt when ok is true. the run
  // loop stops itself on the halting tick, so a finished program is not
  // running -- but Step would still be enabled and a manual Step on the
  // empty frame stack faults the VM, so the controls gate Step (and Run)
  // off this flag. a Reset rebuilds the VM at the entry point and clears
  // lastRun, which clears finished.
  const finished =
    lastRun !== null &&
    ("status" in lastRun ? lastRun.status === "halted" : lastRun.ok);

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
    finished,
    replEval,
    loadedExample,
    loadExample,
    lastCompile,
    lastRun,
    lastState,
    lastArm64,
    version,
  };
}
