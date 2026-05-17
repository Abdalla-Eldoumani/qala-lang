// the codex monaco theme. token colors are deliberately muted: four
// treatments only -- amber keywords, green strings, muted comments,
// muted literals (numbers and booleans). identifier, operator, and
// delimiter render in text-primary. no rainbow highlighting.
//
// two hex formats are in play. the rules array takes a 6-digit hex
// WITHOUT a leading hash. the colors map takes a leading hash, and
// alpha is 8-digit hash-RRGGBBAA -- accent-soft at 0.15 alpha is
// #D4943A26, the 0.30 selection is #D4943A4D.

import type { editor } from "monaco-editor";

export const QALA_THEME_NAME = "qala-codex";

export const qalaTheme: editor.IStandaloneThemeData = {
  base: "vs-dark",
  inherit: true,
  rules: [
    { token: "keyword", foreground: "D4943A" }, // amber
    { token: "string", foreground: "4B8C6F" }, // muted green
    { token: "string.quote", foreground: "4B8C6F" },
    { token: "string.escape", foreground: "4B8C6F" },
    { token: "comment", foreground: "8A8690" }, // muted
    { token: "number", foreground: "8A8690" }, // muted (numbers + booleans)
    { token: "identifier", foreground: "E8E4DC" }, // text-primary
    { token: "operator", foreground: "E8E4DC" },
    { token: "delimiter", foreground: "E8E4DC" },
  ],
  colors: {
    "editor.background": "#0C0E14", // --bg
    "editorGutter.background": "#13151E", // --surface
    "editorLineNumber.foreground": "#8A8690", // --text-secondary
    "editor.lineHighlightBackground": "#D4943A26", // --accent-soft (0.15 alpha)
    "editor.selectionBackground": "#D4943A4D", // rgba(212,148,58,0.30)
    "editorCursor.foreground": "#D4943A", // --accent
    "editor.foreground": "#E8E4DC", // --text-primary
  },
};
