"use client";

// the playground shell: a thin orchestrator. it calls useQala once and
// renders the responsive grid -- the editor, the four placeholder panels,
// the editor header, and the run/step controls. it holds two pieces of
// local UI state (the desktop bytecode-collapsed flag and the mobile
// active tab); execution state lives in rust, owned by the hook.
//
// the editor is imported through next/dynamic with ssr false: monaco
// touches document/window and cannot server-render. that import plus the
// "use client" directive in Editor.tsx are the two layers of protection.

import dynamic from "next/dynamic";
import { useState } from "react";
import { useQala } from "@/lib/use-qala";
import Controls from "@/components/Controls";
import BytecodePanel from "@/components/BytecodePanel";
import StackPanel from "@/components/StackPanel";
import VariablesPanel from "@/components/VariablesPanel";
import ConsolePanel from "@/components/ConsolePanel";

const Editor = dynamic(() => import("@/components/Editor"), { ssr: false });

type MobileTab = "stack" | "variables" | "console";

// the editor header: the wordmark, the quiet subtitle, and the tablet
// bytecode toggle. the ExampleLoader slot is left empty -- it is phase 8.
function EditorHeader({
  bytecodeShown,
  onToggleBytecode,
}: {
  bytecodeShown: boolean;
  onToggleBytecode: () => void;
}) {
  return (
    <header className="flex items-baseline justify-between border-b border-border bg-surface p-[var(--space-4)]">
      <div className="flex items-baseline gap-[var(--space-3)]">
        <span className="font-[family-name:var(--font-display)] text-[28px] font-normal leading-[1.2] text-text-primary">
          Qala
        </span>
        <span className="hidden text-[12px] font-medium leading-[1.4] text-text-secondary min-[768px]:inline">
          a language that means what it says
        </span>
      </div>
      {/* the tablet bytecode toggle: visible only at the tablet width */}
      <button
        type="button"
        onClick={onToggleBytecode}
        aria-pressed={bytecodeShown}
        className={`hidden rounded-[var(--radius-button)] border px-[var(--space-3)] py-[var(--space-1)] text-[12px] font-medium leading-[1.4] transition-colors duration-150 ease-out focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent min-[768px]:block min-[1200px]:hidden ${
          bytecodeShown
            ? "border-accent text-accent"
            : "border-border text-text-secondary hover:border-text-secondary"
        }`}
      >
        Bytecode
      </button>
    </header>
  );
}

// the calm centered surface shown while the wasm module loads.
function LoadingSurface() {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-[var(--space-1)] bg-bg p-[var(--space-6)] text-center">
      <p className="animate-pulse text-[14px] font-normal leading-[1.5] text-text-primary">
        loading the Qala compiler
      </p>
      <p className="text-[14px] font-normal leading-[1.5] text-text-secondary">
        this runs entirely in your browser
      </p>
    </div>
  );
}

// the load-failure surface. it renders fixed plain-language copy -- never
// the caught error. no stack trace, no error code, no exception text.
function ErrorSurface() {
  return (
    <div className="flex h-full flex-col items-center justify-center bg-bg p-[var(--space-6)]">
      <div className="max-w-[420px] border-l-2 border-error pl-[var(--space-4)]">
        <p className="text-[14px] font-normal leading-[1.5] text-text-primary">
          the Qala compiler could not load
        </p>
        <p className="mt-[var(--space-2)] text-[14px] font-normal leading-[1.5] text-text-secondary">
          your browser could not start the WebAssembly module that runs Qala.
          try reloading the page, or open the playground in a recent version
          of Chrome, Firefox, Safari, or Edge.
        </p>
      </div>
    </div>
  );
}

// the mobile tab switcher for the Stack / Variables / Console panels.
function MobileTabs({
  active,
  onSelect,
}: {
  active: MobileTab;
  onSelect: (tab: MobileTab) => void;
}) {
  const tabs: { id: MobileTab; label: string }[] = [
    { id: "stack", label: "Stack" },
    { id: "variables", label: "Variables" },
    { id: "console", label: "Console" },
  ];
  return (
    <div
      role="tablist"
      className="flex gap-[var(--space-2)] border-b border-border bg-surface p-[var(--space-2)]"
    >
      {tabs.map((tab) => (
        <button
          key={tab.id}
          type="button"
          role="tab"
          aria-selected={active === tab.id}
          onClick={() => onSelect(tab.id)}
          className={`rounded-[var(--radius-button)] border px-[var(--space-3)] py-[var(--space-1)] text-[12px] font-medium uppercase tracking-wide leading-[1.4] transition-colors duration-150 ease-out focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent ${
            active === tab.id
              ? "border-accent text-accent"
              : "border-border text-text-secondary hover:border-text-secondary"
          }`}
        >
          {tab.label}
        </button>
      ))}
    </div>
  );
}

export default function Page() {
  const { status, source, setSource, run, step } = useQala();

  // local UI state -- not execution state. the bytecode panel collapse
  // (desktop) and the visible mobile tab live here, in the page.
  const [bytecodeCollapsed, setBytecodeCollapsed] = useState(false);
  const [bytecodeShown, setBytecodeShown] = useState(false);
  const [mobileTab, setMobileTab] = useState<MobileTab>("stack");

  // the editor region: the loading / error surface, or the editor itself.
  const editorRegion =
    status === "loading" ? (
      <LoadingSurface />
    ) : status === "error" ? (
      <ErrorSurface />
    ) : (
      <Editor value={source} onChange={setSource} />
    );

  // the desktop grid columns. when the bytecode panel is collapsed its
  // track shrinks to auto (the slim re-expand rail) and the editor and
  // state columns take the freed width. every breakpoint in this file
  // uses the arbitrary min-[..px] variant family so tailwind sorts them
  // purely by min-width -- mixing a named md variant with an arbitrary
  // min-[1200px] variant sorts them into separate groups and the wrong
  // rule wins at desktop widths.
  const desktopColumns = bytecodeCollapsed
    ? "min-[1200px]:grid-cols-[50fr_auto_50fr]"
    : "min-[1200px]:grid-cols-[50fr_25fr_25fr]";

  return (
    <div className="flex h-dvh flex-col bg-bg min-[1920px]:mx-auto min-[1920px]:max-w-[1920px] min-[1920px]:px-[var(--space-8)]">
      {/* main area: scrolls internally on desktop/tablet, never the page.
          on mobile the controls bar below is the last flex child, so the
          h-dvh shrink when the keyboard opens keeps it visible. */}
      <main
        className={`grid min-h-0 flex-1 grid-cols-1 gap-[var(--space-2)] p-[var(--space-2)] min-[768px]:grid-cols-[60fr_40fr] min-[768px]:gap-[var(--space-6)] min-[1200px]:gap-[var(--space-2)] ${desktopColumns}`}
      >
        {/* the editor column: header, editor surface, controls bar
            (the bar is in-column from tablet up; on mobile it is pinned
            below, outside main) */}
        <section className="flex min-h-0 flex-col overflow-hidden rounded-[var(--radius-panel)] border border-border bg-surface">
          <EditorHeader
            bytecodeShown={bytecodeShown}
            onToggleBytecode={() => setBytecodeShown((v) => !v)}
          />
          <div className="min-h-0 flex-1 overflow-hidden">{editorRegion}</div>
          <div className="hidden min-[768px]:block">
            <Controls status={status} onRun={run} onStep={step} />
          </div>
        </section>

        {/* the bytecode panel: desktop column (collapsible), hidden on
            tablet by default, absent on mobile. when collapsed the column
            shrinks to a slim re-expand rail so the panel can be reopened. */}
        {bytecodeCollapsed ? (
          <button
            type="button"
            onClick={() => setBytecodeCollapsed(false)}
            aria-label="show the bytecode panel"
            className="hidden w-[var(--space-6)] items-center justify-center rounded-[var(--radius-panel)] border border-border bg-surface text-[12px] font-medium leading-[1.4] text-text-secondary transition-colors duration-150 ease-out hover:border-text-secondary focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent min-[1200px]:flex"
          >
            <span className="[writing-mode:vertical-rl] uppercase tracking-wide">
              Bytecode
            </span>
          </button>
        ) : (
          <div className="relative hidden min-h-0 min-[1200px]:flex min-[1200px]:flex-col">
            {/* the collapse control is positioned over the panel header so
                the bytecode panel still starts at the top of the column,
                aligned with the editor and the state panels. */}
            <button
              type="button"
              onClick={() => setBytecodeCollapsed(true)}
              aria-label="collapse the bytecode panel"
              className="absolute right-[var(--space-3)] top-[var(--space-2)] z-10 rounded-[var(--radius-button)] border border-border bg-surface px-[var(--space-2)] py-[var(--space-1)] text-[12px] font-medium leading-[1.4] text-text-secondary transition-colors duration-150 ease-out hover:border-text-secondary focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
            >
              collapse
            </button>
            <BytecodePanel className="min-h-0 flex-1" />
          </div>
        )}

        {/* the state panels: Variables, Stack, Console stacked on
            desktop/tablet; tabbed on mobile */}
        <aside className="hidden min-h-0 min-[768px]:flex min-[768px]:flex-col min-[768px]:gap-[var(--space-2)]">
          <VariablesPanel className="min-h-0 flex-1" />
          <StackPanel className="min-h-0 flex-1" />
          <ConsolePanel className="min-h-0 flex-1" />
        </aside>

        {/* mobile: the same three panels behind a tab switcher */}
        <section className="flex min-h-0 flex-col overflow-hidden rounded-[var(--radius-panel)] border border-border bg-surface min-[768px]:hidden">
          <MobileTabs active={mobileTab} onSelect={setMobileTab} />
          <div className="min-h-0 flex-1 overflow-auto">
            {mobileTab === "stack" && <StackPanel className="border-0" />}
            {mobileTab === "variables" && (
              <VariablesPanel className="border-0" />
            )}
            {mobileTab === "console" && <ConsolePanel className="border-0" />}
          </div>
        </section>
      </main>

      {/* mobile only: the controls bar pinned to the bottom of the
          viewport. it is the last flex child of the h-dvh column, so the
          dynamic viewport unit reflows it above the keyboard rather than
          hiding it. from tablet up the bar lives inside the editor column. */}
      <div className="min-[768px]:hidden">
        <Controls status={status} onRun={run} onStep={step} />
      </div>
    </div>
  );
}
