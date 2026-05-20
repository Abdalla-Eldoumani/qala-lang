"use client";

// monaco mounted as a client component with the qala language and the
// codex theme. monaco touches document/window so it cannot server-render;
// the use-client directive here plus the ssr-false dynamic import in
// page.tsx are the two layers that keep it client-only.
//
// phase 8 adds three things on top of the phase 7 syntax-and-theme base:
// diagnostic underlines from the structured compile result, a whole-line
// highlight tracking the vm's current source line during step-through,
// and a quiet needs-compile rule shown while the source is dirty. the
// markers and the decoration are imperative monaco side effects, so they
// are driven by useEffects keyed on the snapshot props, not by render.

import { useEffect, useRef } from "react";
import { Editor, type Monaco } from "@monaco-editor/react";
import type { editor as MonacoEditor, languages } from "monaco-editor";
import { qalaLanguage, qalaLanguageConfig } from "@/lib/syntax";
import { qalaTheme, QALA_THEME_NAME } from "@/lib/theme";
import type { CompileResult, VmState } from "@/lib/types";

const QALA_LANGUAGE_ID = "qala";

// one stable owner string for every setModelMarkers call so a later set
// (or an empty-array clear) replaces this editor's prior qala markers and
// leaves any other owner's markers untouched.
const MARKER_OWNER = "qala";

function handleBeforeMount(monaco: Monaco) {
  // register only once: beforeMount can fire again on remount.
  const already = monaco.languages
    .getLanguages()
    .some((l: languages.ILanguageExtensionPoint) => l.id === QALA_LANGUAGE_ID);
  if (!already) {
    monaco.languages.register({ id: QALA_LANGUAGE_ID });
    monaco.languages.setMonarchTokensProvider(QALA_LANGUAGE_ID, qalaLanguage);
    monaco.languages.setLanguageConfiguration(
      QALA_LANGUAGE_ID,
      qalaLanguageConfig,
    );
  }
  // defineTheme is safe to call every beforeMount; it overwrites.
  monaco.editor.defineTheme(QALA_THEME_NAME, qalaTheme);
}

export default function QalaEditor({
  value,
  onChange,
  // the last compile result drives the diagnostic markers; null before
  // the first compile. lastState drives the current-line decoration;
  // null before the first step. dirty true means the source changed
  // since the last successful compile: the current-line decoration is
  // stale and gets cleared, but the diagnostic markers stay (a failed
  // compile leaves the source dirty, so its errors must remain visible).
  // plan 08-05 wires these from useQala.
  //
  // the three step-through props are optional: the phase 7 page renders
  // this editor with value/onChange only, and plan 08-05 adds the wiring.
  // the defaults are the inert "no compile, not stepping, not dirty"
  // state: the editor shows no markers, no line highlight, no rule.
  lastCompile = null,
  lastState = null,
  dirty = false,
}: {
  value: string;
  onChange: (next: string) => void;
  lastCompile?: CompileResult | null;
  lastState?: VmState | null;
  dirty?: boolean;
}) {
  // the editor instance and the monaco namespace are both needed: markers
  // go through monaco.editor.setModelMarkers, the decoration is created on
  // the editor instance. onMount is the one place that hands us both.
  const editorRef = useRef<MonacoEditor.IStandaloneCodeEditor | null>(null);
  const monacoRef = useRef<typeof import("monaco-editor") | null>(null);
  // a single decorations collection holds the one moving current-line
  // highlight. .set() replaces it, .clear() removes it. created lazily on
  // first use because it needs the mounted editor instance.
  const lineDecoRef =
    useRef<MonacoEditor.IEditorDecorationsCollection | null>(null);

  function handleMount(
    editor: MonacoEditor.IStandaloneCodeEditor,
    monaco: typeof import("monaco-editor"),
  ) {
    editorRef.current = editor;
    monacoRef.current = monaco;
  }

  // diagnostic markers. keyed on lastCompile only: the diagnostics of the
  // last compile (a failed compile's errors or a successful compile's
  // warnings) map straight to monaco markers and stay visible while the
  // source is edited (monaco shifts the marker ranges to follow the edits)
  // until the next compile refreshes them. clearing on dirty would hide a
  // failed compile's errors, since a failed compile leaves the source
  // dirty. both the MonacoDiagnostic positions and monaco markers are
  // 1-based, so the line/column values pass through with no conversion.
  useEffect(() => {
    const editor = editorRef.current;
    const monaco = monacoRef.current;
    // the editor mounts asynchronously; this effect can fire before
    // onMount has stored the refs.
    if (!editor || !monaco) return;
    const model = editor.getModel();
    if (!model) return;

    if (!lastCompile) {
      monaco.editor.setModelMarkers(model, MARKER_OWNER, []);
      return;
    }

    const markers: MonacoEditor.IMarkerData[] = lastCompile.diagnostics.map(
      (d) => ({
        startLineNumber: d.line,
        startColumn: d.column,
        endLineNumber: d.end_line,
        endColumn: d.end_column,
        // the plain compiler text verbatim, shown as the marker hover.
        message: d.message,
        // severity 1 is an error, 0 is a warning (lib/types.ts).
        severity:
          d.severity === 1
            ? monaco.MarkerSeverity.Error
            : monaco.MarkerSeverity.Warning,
      }),
    );
    monaco.editor.setModelMarkers(model, MARKER_OWNER, markers);
  }, [lastCompile]);

  // the current-source-line highlight. keyed on lastState and dirty: a
  // dirty source invalidates the vm snapshot, so the highlight is cleared.
  // current_line is 1-based; 0 means "no line" (a synthesized instruction
  // or an out-of-range ip) and also clears the highlight. this is a
  // distinct highlight from the bytecode panel's ip highlight: this one
  // keys on a source line, that one on a byte offset.
  useEffect(() => {
    const editor = editorRef.current;
    const monaco = monacoRef.current;
    if (!editor || !monaco) return;
    if (!lineDecoRef.current) {
      lineDecoRef.current = editor.createDecorationsCollection();
    }
    const collection = lineDecoRef.current;

    const line = lastState?.current_line ?? 0;
    if (dirty || line <= 0) {
      collection.clear();
      return;
    }
    collection.set([
      {
        // a zero-width anchor range; isWholeLine spans the whole line
        // regardless of the columns.
        range: new monaco.Range(line, 1, line, 1),
        options: { isWholeLine: true, className: "qala-current-line" },
      },
    ]);
  }, [lastState, dirty]);

  // clear the decoration collection on unmount so a stale highlight does
  // not survive into a remounted editor.
  useEffect(() => {
    return () => {
      lineDecoRef.current?.clear();
    };
  }, []);

  return (
    // a positioned wrapper so the needs-compile rule can sit on the
    // region's leading inner edge. the editor fills the wrapper.
    <div className="relative h-full w-full">
      {dirty && (
        // the quiet needs-compile signal: a 2px --warning rule on the
        // leading inner edge, shown only while dirty. not a banner, not
        // an overlay: the editor stays fully editable underneath it.
        <div
          aria-hidden
          title="source changed, compile to run"
          className="pointer-events-none absolute inset-y-0 left-0 z-10 w-[2px] bg-warning"
        />
      )}
      <Editor
        defaultLanguage={QALA_LANGUAGE_ID}
        language={QALA_LANGUAGE_ID}
        theme={QALA_THEME_NAME}
        value={value}
        onChange={(v) => onChange(v ?? "")}
        beforeMount={handleBeforeMount}
        onMount={handleMount}
        options={{
          fontFamily: "var(--font-mono)",
          fontSize: 13,
          // monaco reads lineHeight >= 8 as a pixel value and floors a
          // fraction. 1.6 * 13 is 20.8, so use the nearest integer, 21.
          lineHeight: 21,
          minimap: { enabled: false },
          scrollBeyondLastLine: false,
          // wrap long lines to the editor's width. on a 375px phone the
          // monaco viewport is too narrow for `return fibonacci(n - 1) +
          // fibonacci(n - 2)` to fit on one line; without this it clips
          // at the right edge and the user has to pan horizontally.
          wordWrap: "on",
        }}
      />
    </div>
  );
}
