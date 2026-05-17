// the run/step controls bar, wired to useQala through props. phase 7
// scope is exactly two buttons: Run and Step. Pause, Reset, and the
// speed slider are phase 8 and are not rendered here, not even disabled.
//
// Run is the one primary action: amber fill, --bg label. Step is a
// secondary control: surface fill, border outline, --text-primary
// label. both carry the disabled attribute until the wasm session is
// ready, so the disabled-until-ready behavior holds.

import type { QalaStatus } from "@/lib/use-qala";

export default function Controls({
  status,
  onRun,
  onStep,
}: {
  status: QalaStatus;
  onRun: () => void;
  onStep: () => void;
}) {
  const ready = status === "ready";

  return (
    <div className="flex items-center gap-[var(--space-2)] border-t border-border bg-surface p-[var(--space-4)]">
      <button
        type="button"
        onClick={onRun}
        disabled={!ready}
        className="rounded-[var(--radius-button)] bg-accent px-[var(--space-3)] py-[var(--space-2)] text-[12px] font-medium leading-[1.4] text-bg transition-colors duration-150 ease-out hover:bg-accent/85 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent disabled:cursor-not-allowed disabled:bg-accent/40 disabled:text-text-secondary"
      >
        Run
      </button>
      <button
        type="button"
        onClick={onStep}
        disabled={!ready}
        className="rounded-[var(--radius-button)] border border-border bg-surface px-[var(--space-3)] py-[var(--space-2)] text-[12px] font-medium leading-[1.4] text-text-primary transition-colors duration-150 ease-out hover:border-text-secondary focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent disabled:cursor-not-allowed disabled:text-text-secondary"
      >
        Step
      </button>
    </div>
  );
}
