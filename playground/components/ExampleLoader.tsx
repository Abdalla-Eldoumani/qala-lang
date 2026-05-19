// the example loader: a dropdown in the editor header that loads one of
// the six bundled .qala examples into the editor. it is a native
// <select> styled to the codex tokens.
//
// why a native select and not a hand-built listbox: the native control
// is keyboard-reachable for free (it gets the --accent focus ring), and
// its option popup renders in the browser's own layer, so it is never
// clipped by the editor region's overflow: hidden, at any width, with no
// z-index handling. the accepted trade-off is that the open option list
// cannot carry the --accent-soft row background; the closed trigger is
// fully on-token, which is what the user sees most.

import type { ChangeEvent } from "react";

// the six bundled examples. display is the short name shown in the
// trigger and the option list (the Copywriting Contract); file is the
// .qala basename under public/examples/ that loadExample fetches --
// defer -> defer-demo, matching -> pattern-matching, the rest identical.
const EXAMPLES: { display: string; file: string }[] = [
  { display: "hello", file: "hello" },
  { display: "fibonacci", file: "fibonacci" },
  { display: "effects", file: "effects" },
  { display: "defer", file: "defer-demo" },
  { display: "matching", file: "pattern-matching" },
  { display: "pipeline", file: "pipeline" },
];

// the loaded-example name the hook tracks is the file basename. the
// trigger shows a display name, so map the basename back; an unknown
// value (the user's own restored text) falls through to null and the
// trigger shows the `examples` placeholder.
function displayFor(loadedExample: string | null): string | null {
  if (loadedExample === null) return null;
  const match = EXAMPLES.find((example) => example.file === loadedExample);
  return match ? match.display : null;
}

// the chevron is drawn in pure css: a small triangle as an inline svg
// data uri background, no icon font and no icon set. it is encoded once
// here so the select can drop its native arrow (appearance: none) and
// still show a downward cue at the trailing edge.
const CHEVRON =
  "url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='8' height='6' viewBox='0 0 8 6'><path d='M0 0l4 6 4-6z' fill='%238A8690'/></svg>\")";

export default function ExampleLoader({
  // the loaded example, a file basename, or null when the editor holds
  // the user's own text. drives the trigger label.
  loadedExample,
  // the hook method: fetches /examples/<file>.qala and loads it into the
  // editor. loading an example is an edit, so the hook also sets dirty.
  loadExample,
}: {
  loadedExample: string | null;
  loadExample: (file: string) => void;
}) {
  // the trigger label: the loaded example's display name, or null when
  // the editor holds the user's own text (the placeholder shows then).
  const display = displayFor(loadedExample);

  // a selection loads that example. the <select> value is pinned to the
  // loaded file so the trigger can show the loaded example's name (the
  // UI-SPEC surface 7 contract). a native select does not fire onChange
  // when the already-selected option is re-picked, so directly after a
  // load the same example cannot be re-picked. that dead-zone is narrow
  // and harmless: it lasts only until the first edit, and until the
  // first edit the editor still holds the pristine example, so a reload
  // would change nothing. the first edit clears loadedExample (the hook),
  // the value snaps to the placeholder, and every option fires again.
  // re-picking the same example to discard edits is therefore always
  // reachable once there are edits to discard.
  function handleChange(event: ChangeEvent<HTMLSelectElement>) {
    const file = event.target.value;
    if (file === "") return;
    loadExample(file);
  }

  return (
    <select
      // the value tracks the loaded file. when the editor holds the
      // user's own text (loadedExample null, or a name with no known
      // file) the value is "", the disabled placeholder option.
      value={loadedExample !== null && display !== null ? loadedExample : ""}
      onChange={handleChange}
      aria-label="load a bundled example"
      // appearance-none drops the native arrow; the css chevron is a
      // right-anchored background image. the closed trigger is the codex
      // button treatment (a --surface fill, a 1px --border, the button
      // radius, Label type, --text-primary) and takes the --accent
      // focus ring like every other control.
      style={{
        backgroundImage: CHEVRON,
        backgroundRepeat: "no-repeat",
        backgroundPosition: "right var(--space-3) center",
      }}
      className="cursor-pointer appearance-none rounded-[var(--radius-button)] border border-border bg-surface py-[var(--space-1)] pl-[var(--space-3)] pr-[var(--space-6)] text-[12px] font-medium leading-[1.4] text-text-primary transition-colors duration-150 ease-out hover:border-text-secondary focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
    >
      {/* the placeholder: shown when no known example is loaded. it is
          disabled so it cannot be re-selected, and hidden from the open
          list so the menu shows only the six real examples. */}
      <option value="" disabled hidden>
        {display ?? "examples"}
      </option>
      {EXAMPLES.map((example) => (
        <option key={example.file} value={example.file}>
          {example.display}
        </option>
      ))}
    </select>
  );
}
