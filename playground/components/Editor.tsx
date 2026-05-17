"use client";

// monaco mounted as a client component with the qala language and the
// codex theme. monaco touches document/window so it cannot server-render;
// the use-client directive here plus the ssr-false dynamic import in
// page.tsx are the two layers that keep it client-only.
//
// phase 7 is syntax highlighting and theme only. error underlines and
// the current-line highlight are phase 8 -- not added here. editing
// source does nothing observable downstream until a Run; that is the
// expected foundation/visualization seam.

import { Editor, type Monaco } from "@monaco-editor/react";
import type { languages } from "monaco-editor";
import { qalaLanguage, qalaLanguageConfig } from "@/lib/syntax";
import { qalaTheme, QALA_THEME_NAME } from "@/lib/theme";

const QALA_LANGUAGE_ID = "qala";

function handleBeforeMount(monaco: Monaco) {
  // register only once -- beforeMount can fire again on remount.
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

export default function QalaEditor(props: {
  value: string;
  onChange: (next: string) => void;
}) {
  return (
    <Editor
      defaultLanguage={QALA_LANGUAGE_ID}
      language={QALA_LANGUAGE_ID}
      theme={QALA_THEME_NAME}
      value={props.value}
      onChange={(v) => props.onChange(v ?? "")}
      beforeMount={handleBeforeMount}
      options={{
        fontFamily: "var(--font-mono)",
        fontSize: 13,
        // monaco reads lineHeight >= 8 as a pixel value and floors a
        // fraction. 1.6 * 13 is 20.8, so use the nearest integer, 21.
        lineHeight: 21,
        minimap: { enabled: false },
        scrollBeyondLastLine: false,
      }}
    />
  );
}
