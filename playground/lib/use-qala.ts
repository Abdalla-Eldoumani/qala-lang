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

import { useCallback, useEffect, useRef, useState } from "react";
import { loadQala, type QalaSession } from "./compiler";
import type { CompileResult, RunResult, StepResult, VmState } from "./types";

export type QalaStatus = "loading" | "ready" | "error";

export interface UseQala {
  status: QalaStatus;
  source: string;
  setSource: (next: string) => void;
  compile: () => void;
  run: () => void;
  step: () => void;
  reset: () => void;
  // last bridge snapshots -- read-only views, never mirrored vm state
  lastCompile: CompileResult | null;
  lastRun: RunResult | StepResult | null;
  lastState: VmState | null;
  version: number;
}

export function useQala(): UseQala {
  const sessionRef = useRef<QalaSession | null>(null);
  const [status, setStatus] = useState<QalaStatus>("loading");
  const [source, setSource] = useState<string>("");
  const [lastCompile, setLastCompile] = useState<CompileResult | null>(null);
  const [lastRun, setLastRun] = useState<RunResult | StepResult | null>(null);
  const [lastState, setLastState] = useState<VmState | null>(null);
  const [version, setVersion] = useState(0);

  // load the wasm module once on mount. a load/instantiate failure flips
  // status to "error"; the layout renders the friendly message from that.
  useEffect(() => {
    let cancelled = false;
    loadQala()
      .then((session) => {
        if (cancelled) return;
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

  const compile = useCallback(() => {
    const s = sessionRef.current;
    if (!s) return;
    setLastCompile(s.compile(source));
    setLastState(s.getState());
    setVersion((v) => v + 1);
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
    if (!s) return;
    const result = s.step();
    setLastRun(result);
    setLastState(result.state);
    setVersion((v) => v + 1);
  }, []);

  const reset = useCallback(() => {
    const s = sessionRef.current;
    if (!s) return;
    s.reset();
    setLastCompile(null);
    setLastRun(null);
    setLastState(null);
    setVersion((v) => v + 1);
  }, []);

  return {
    status,
    source,
    setSource,
    compile,
    run,
    step,
    reset,
    lastCompile,
    lastRun,
    lastState,
    version,
  };
}
