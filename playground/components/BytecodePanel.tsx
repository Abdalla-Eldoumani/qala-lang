// the bytecode panel, a styled placeholder for phase 7. it is a finished
// container that is simply empty of data -- no spinner, no skeleton, no
// "coming soon". phase 8 fills it with the disassembly view. it holds no
// useQala state this phase.

export default function BytecodePanel({
  className,
}: {
  className?: string;
}) {
  return (
    <section
      className={`flex flex-col overflow-hidden rounded-[var(--radius-panel)] border border-border bg-surface ${className ?? ""}`}
    >
      <header className="border-b border-border p-[var(--space-3)] text-[12px] font-medium uppercase tracking-wide leading-[1.4] text-text-secondary">
        Bytecode
      </header>
      <div className="p-[var(--space-4)] text-[14px] font-normal leading-[1.5] text-text-secondary">
        compile a program to see its bytecode
      </div>
    </section>
  );
}
