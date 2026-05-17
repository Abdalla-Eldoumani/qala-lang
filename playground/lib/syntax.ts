// the qala monaco language: a monarch tokenizer plus a language
// configuration. the keyword set is the authoritative list from
// compiler/src/token.rs's keyword() function -- 32 reserved words:
// 24 control/declaration keywords, 2 boolean literals, 6 primitive
// type names.
//
// Result, Option, Ok, Err, Some, None, and the stdlib functions
// (print, println, len, push, pop, sqrt, abs, assert, type_of, open,
// close, map, filter, reduce) are NOT keywords. the lexer treats them
// as ordinary identifiers, so the highlighter must too -- coloring
// any of them as a keyword is a visible defect against the language.

import type { languages } from "monaco-editor";

// keyword set is the authoritative list from compiler/src/token.rs.
// Result/Option/Ok/Err/Some/None and the stdlib fns are NOT keywords --
// the lexer treats them as identifiers, so the highlighter must too.
const KEYWORDS = [
  "fn", "let", "mut", "if", "else", "while", "for", "in", "return",
  "break", "continue", "defer", "match", "struct", "enum", "interface",
  "comptime", "is", "pure", "io", "alloc", "panic", "or", "self",
];
const TYPE_KEYWORDS = ["i64", "f64", "bool", "str", "byte", "void"];
const BOOL_KEYWORDS = ["true", "false"];

export const qalaLanguage: languages.IMonarchLanguage = {
  defaultToken: "",
  keywords: KEYWORDS,
  typeKeywords: TYPE_KEYWORDS,
  booleans: BOOL_KEYWORDS,
  operators: [
    "+", "-", "*", "/", "%", "==", "!=", "<", "<=", ">", ">=",
    "&&", "||", "!", "=", "->", "=>", "|>", "?", "..", "..=",
  ],
  tokenizer: {
    root: [
      [/\/\/.*$/, "comment"],
      [
        /[a-zA-Z_]\w*/,
        {
          cases: {
            "@keywords": "keyword",
            "@typeKeywords": "keyword",
            "@booleans": "number",
            "@default": "identifier",
          },
        },
      ],
      [/0[xX][0-9a-fA-F_]+/, "number"],
      [/0[bB][01_]+/, "number"],
      [/\d[\d_]*\.\d[\d_]*([eE][-+]?\d+)?/, "number"],
      [/\d[\d_]*/, "number"],
      [/b'(\\.|[^'\\])'/, "number"],
      [/"/, { token: "string.quote", next: "@string" }],
      [/[{}()[\]]/, "@brackets"],
      // longest operators first so ..= is not split into .. then =
      [/\.\.=|\.\.|->|=>|\|>|==|!=|<=|>=|&&|\|\||[-+*/%<>!=?]/, "operator"],
      [/[.,:;]/, "delimiter"],
    ],
    string: [
      [/[^"{\\]+/, "string"],
      [/\\./, "string.escape"],
      // interpolation: leave the string state, tokenize the expr as code
      [/\{/, { token: "delimiter.bracket", next: "@interp" }],
      [/"/, { token: "string.quote", next: "@pop" }],
    ],
    interp: [
      [/\}/, { token: "delimiter.bracket", next: "@pop" }],
      { include: "root" },
    ],
  },
};

export const qalaLanguageConfig: languages.LanguageConfiguration = {
  comments: { lineComment: "//" },
  brackets: [
    ["{", "}"],
    ["[", "]"],
    ["(", ")"],
  ],
  autoClosingPairs: [
    { open: "{", close: "}" },
    { open: "[", close: "]" },
    { open: "(", close: ")" },
    { open: '"', close: '"' },
  ],
};
