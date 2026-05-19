// the controls bar, wired to useQala through props. the full bar:
// Run/Pause (one button, the label and treatment swap with the run
// loop), Step, Reset, and a three-stop speed slider.
//
// Run is the one solid-amber primary action. while the run loop is live
// the same button becomes Pause, the active-control treatment: a
// --surface fill, a 2px --accent outline, an --accent label, and the
// running-indicator pulse from globals.css. Step and Reset are secondary
// controls (surface fill, border outline, never amber). the disabled
// matrix is the contract implemented exactly below.

import type { QalaStatus, RunSpeed } from "@/lib/use-qala";

// the three speed stops in slider order. the slider is a native range
// input min=0 max=2 step=1; the index into this array is the range value,
// so position 0 is the slowest tick and position 2 the fastest.
const SPEED_STOPS: RunSpeed[] = ["slow", "normal", "fast"];

export default function Controls({
  status,
  // true while the run loop is live. drives the Run/Pause label swap and
  // the running-indicator pulse.
  running,
  // true when the source changed since the last successful compile. Run
  // stays enabled while dirty (it compiles first, then runs); Step is
  // disabled while dirty (stepping stale bytecode is wrong).
  dirty,
  // true when a program is compiled. gates Step, Reset, and the slider.
  // it stays true after an edit (the dirty flag is the separate stale
  // signal) so the needs-compile column of the matrix is covered.
  programCompiled,
  // true when the program has run to its end. gates Step and Run off so
  // a manual Step past the last instruction does not fault the VM;
  // re-running a finished program requires a Reset first.
  finished,
  // the run-loop speed stop and its setter. the slider reads and writes
  // them; a mid-run change applies on the next tick (the hook).
  speed,
  setSpeed,
  // the four run-loop callbacks. Run maps to play (the animated loop),
  // NOT the bridge's instant run. Pause stops the loop; Step advances one
  // instruction; Reset stops the loop and resets the VM.
  onRun,
  onStep,
  onPause,
  onReset,
}: {
  status: QalaStatus;
  running: boolean;
  dirty: boolean;
  programCompiled: boolean;
  finished: boolean;
  speed: RunSpeed;
  setSpeed: (next: RunSpeed) => void;
  onRun: () => void;
  onStep: () => void;
  onPause: () => void;
  onReset: () => void;
}) {
  const ready = status === "ready";

  // the disabled matrix. Run/Pause: disabled when WASM is not ready,
  // when there is no program with the loop stopped and the source
  // clean, or when the program has finished. a dirty source keeps Run
  // enabled (it compiles then runs, the needs-compile column), but a
  // finished program needs a Reset before it can re-run. Step: enabled
  // only with a compiled, non-dirty, unfinished program and the loop
  // stopped, since stepping a finished program faults the empty frame
  // stack. Reset: enabled once compiled, including while the loop runs
  // and after the program finishes, since it is the way back to a
  // re-runnable state. the slider: enabled whenever a program is
  // compiled.
  const runDisabled =
    !ready || finished || (!programCompiled && !running && !dirty);
  const stepDisabled =
    !ready || !programCompiled || running || dirty || finished;
  const resetDisabled = !ready || (!programCompiled && !running);
  const speedDisabled = !ready || !programCompiled;

  // the dirty editor rule (the Editor's leading-edge marker) is the
  // dirty signal; the Run button keeps its label and its solid-amber fill
  // while dirty, and pressing it compiles then runs. the compile-then-run
  // wiring lives in page.tsx; this button just calls the handler it is
  // given.
  return (
    <div className="flex flex-wrap items-center gap-[var(--space-2)] border-t border-border bg-surface p-[var(--space-4)]">
      {/* Run / Pause, one button. running -> Pause: the active-control
          treatment, a --surface fill with a 2px --accent outline and the
          running-indicator opacity pulse on that outline. idle -> Run:
          the one solid-amber primary action. */}
      {running ? (
        <button
          type="button"
          onClick={onPause}
          disabled={!ready}
          className="running-indicator inline-flex min-h-[44px] items-center justify-center rounded-[var(--radius-button)] bg-surface px-[var(--space-3)] py-[var(--space-2)] text-[12px] font-medium leading-[1.4] text-accent outline outline-2 outline-accent transition-colors duration-150 ease-out hover:bg-accent-soft focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent disabled:cursor-not-allowed disabled:text-text-secondary min-[768px]:min-h-0"
        >
          Pause
        </button>
      ) : (
        <button
          type="button"
          onClick={onRun}
          disabled={runDisabled}
          className="inline-flex min-h-[44px] items-center justify-center rounded-[var(--radius-button)] bg-accent px-[var(--space-3)] py-[var(--space-2)] text-[12px] font-medium leading-[1.4] text-bg transition-colors duration-150 ease-out hover:bg-accent/85 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent disabled:cursor-not-allowed disabled:bg-accent/40 disabled:text-bg/50 min-[768px]:min-h-0"
        >
          Run
        </button>
      )}

      {/* Step, a secondary control. enabled only with a compiled,
          non-dirty program and the loop stopped. */}
      <button
        type="button"
        onClick={onStep}
        disabled={stepDisabled}
        className="inline-flex min-h-[44px] items-center justify-center rounded-[var(--radius-button)] border border-border bg-surface px-[var(--space-3)] py-[var(--space-2)] text-[12px] font-medium leading-[1.4] text-text-primary transition-colors duration-150 ease-out hover:border-text-secondary focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent disabled:cursor-not-allowed disabled:text-text-secondary min-[768px]:min-h-0"
      >
        Step
      </button>

      {/* Reset, a secondary control, same treatment as Step. enabled
          once a program is compiled, including while the loop runs:
          Reset while running stops the loop and resets in one action. */}
      <button
        type="button"
        onClick={onReset}
        disabled={resetDisabled}
        className="inline-flex min-h-[44px] items-center justify-center rounded-[var(--radius-button)] border border-border bg-surface px-[var(--space-3)] py-[var(--space-2)] text-[12px] font-medium leading-[1.4] text-text-primary transition-colors duration-150 ease-out hover:border-text-secondary focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent disabled:cursor-not-allowed disabled:text-text-secondary min-[768px]:min-h-0"
      >
        Reset
      </button>

      {/* the speed slider, a three-stop discrete control at the trailing
          edge. a native range input keeps it keyboard-operable (arrows
          move stops) and assistive-tech reachable. the speed label drops
          under 768px where the row is tight. accent-color tints both the
          thumb and the filled track through the --accent token. */}
      <div className="ml-auto flex items-center gap-[var(--space-2)]">
        <label
          htmlFor="speed-slider"
          className="hidden text-[12px] font-medium leading-[1.4] text-text-secondary min-[768px]:inline"
        >
          speed
        </label>
        {/* the track and its three stop labels stacked, so each label
            sits under its slider position; the labels name the discrete
            stops, so the speed is not signalled by the thumb alone. */}
        <div className="flex flex-col gap-[var(--space-1)]">
          <input
            id="speed-slider"
            type="range"
            min={0}
            max={2}
            step={1}
            value={SPEED_STOPS.indexOf(speed)}
            onChange={(event) =>
              setSpeed(SPEED_STOPS[Number(event.target.value)])
            }
            disabled={speedDisabled}
            aria-label={`run speed, ${speed}`}
            // accent-color paints the native thumb and filled track with
            // the codex --accent; the track base shows through as
            // --border. a fixed width keeps the three stops a reach.
            style={{ accentColor: "var(--accent)" }}
            className="w-[108px] cursor-pointer bg-border focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent disabled:cursor-not-allowed disabled:opacity-40"
          />
          <div className="flex w-[108px] justify-between text-[12px] font-medium leading-[1.4] text-text-secondary">
            {SPEED_STOPS.map((stop) => (
              <span
                key={stop}
                className={speed === stop ? "text-accent" : undefined}
              >
                {stop}
              </span>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
