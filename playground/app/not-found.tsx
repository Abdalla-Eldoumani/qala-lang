// the 404 page. the next.js app router renders the root not-found.tsx for
// every unmatched url across the whole app. it is a static server
// component -- no client directive, no hook, no wasm import -- so it is
// fully pre-rendered. the fonts and globals.css come from the root layout;
// this page does not re-import them. next.js auto-injects a noindex robots
// tag for a 404, so none is set here.
//
// the layout reuses the centered structure of the LoadingSurface and the
// ErrorSurface in page.tsx, so the 404 reads as a sibling of those two
// surfaces. it shows the message and the one way back -- a 404 is a dead
// end, not a place to add chrome.

import type { Metadata } from "next";
import Link from "next/link";

// the em dash here is the documented metadata.title convention -- it is
// browser-tab chrome, not ui copy, and is the one place an em dash is
// allowed in this file.
export const metadata: Metadata = {
  title: "Qala — page not found",
};

export default function NotFound() {
  return (
    <div className="flex min-h-dvh flex-col items-center justify-center bg-bg p-[var(--space-6)]">
      <div className="flex max-w-[420px] flex-col items-center gap-[var(--space-4)] text-center">
        {/* the wordmark, Display role -- the same treatment as the editor
            header, so the page is unmistakably the playground. */}
        <span className="font-[family-name:var(--font-display)] text-[28px] font-normal leading-[1.2] text-text-primary">
          Qala
        </span>
        {/* the message, Heading role -- a plain serif sentence, not a
            giant 404 numeral. */}
        <h1 className="font-[family-name:var(--font-display)] text-[20px] font-normal leading-[1.3] text-text-primary">
          this page does not exist
        </h1>
        {/* the supporting line, Body role -- why there is nothing here,
            not a restatement of the message. */}
        <p className="text-[14px] font-normal leading-[1.5] text-text-secondary">
          the page you asked for is not part of the Qala playground.
        </p>
        {/* the way back -- the codex primary action, the same treatment
            as the Run button. min-h-[44px] gives it a touch target at the
            mobile breakpoint; from tablet up the height is compact. */}
        <Link
          href="/"
          className="inline-flex min-h-[44px] items-center justify-center rounded-[var(--radius-button)] bg-accent px-[var(--space-3)] py-[var(--space-2)] text-[12px] font-medium leading-[1.4] text-bg transition-colors duration-150 ease-out hover:bg-accent/85 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent min-[768px]:min-h-0"
        >
          back to the playground
        </Link>
      </div>
    </div>
  );
}
