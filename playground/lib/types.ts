// these mirror the rust serde structs in the WASM bridge. field names are
// snake_case to match serde's default serialization -- the rust
// structs carry no #[serde(rename)]. verified against:
//   compiler/src/wasm.rs        CompileResult/RunResult/StepResult/ReplResult
//   compiler/src/vm.rs          VmState/StateValue/NamedValue
//   compiler/src/diagnostics.rs MonacoDiagnostic
//
// a rust Option<T> serializes through serde-wasm-bindgen's default
// Serializer as the value or `undefined`. serialize_missing_as_null is
// false in 0.6.5, so a rust None becomes JsValue::UNDEFINED, not NULL.
// disassembly, error, value, and category are therefore optional `?`
// properties: the key is absent (value undefined) when the rust side was
// None. note `undefined === null` is false -- test these with `=== undefined`
// or a truthiness check, never `=== null`.

// diagnostics.rs MonacoDiagnostic: severity 0 = warning, 1 = error.
export interface MonacoDiagnostic {
  line: number; // 1-based start line
  column: number; // 1-based start column
  end_line: number; // 1-based end line
  end_column: number; // 1-based end column (exclusive)
  severity: number; // 0 = warning, 1 = error
  message: string;
  category?: string; // snake_case warning category; absent for errors
}

// vm.rs StateValue: a rendered value plus its runtime type name.
export interface StateValue {
  rendered: string; // e.g. "42", "true", "[1, 2, 3]"
  type_name: string; // e.g. "i64", "str", "[i64]", "Shape"
}

// vm.rs NamedValue: an in-scope variable.
export interface NamedValue {
  name: string; // source name, or "slot{i}" for an unnamed temporary
  value: StateValue;
}

// vm.rs VmState: the step-through snapshot.
export interface VmState {
  chunk_index: number;
  ip: number; // instruction pointer (byte offset into the chunk)
  current_line: number; // 1-based source line of the instruction at ip; 0 = none
  stack: StateValue[]; // value stack, bottom to top
  variables: NamedValue[]; // current frame's in-scope variables
  console: string[]; // accumulated print/println output
  leak_log: string[]; // resource-leak messages
}

// wasm.rs CompileResult.
export interface CompileResult {
  ok: boolean; // true on a clean compile (warnings ok)
  disassembly?: string; // a string on success, absent on error
  diagnostics: MonacoDiagnostic[]; // warnings on success, errors on failure
}

// wasm.rs Arm64Result -- the ARM64 backend's ok-flagged output.
export interface Arm64Result {
  ok: boolean; // true when the ARM64 backend produced assembly
  assembly?: string; // the AArch64 assembly text; absent on error
  diagnostics: MonacoDiagnostic[]; // unsupported-construct or front-end errors
}

// wasm.rs RunResult.
export interface RunResult {
  ok: boolean; // true when the program ran to Halt
  state: VmState; // final or fault-point snapshot
  error?: MonacoDiagnostic; // a diagnostic on a runtime fault, else absent
}

// wasm.rs StepResult. status is a plain string, not a tagged enum.
export interface StepResult {
  status: "ran" | "halted" | "error";
  state: VmState;
  error?: MonacoDiagnostic; // a diagnostic on a runtime fault, else absent
}

// wasm.rs ReplResult -- the console's repl result.
export interface ReplResult {
  ok: boolean;
  value?: StateValue; // the evaluated value; absent when there is none
  console: string[];
  error?: MonacoDiagnostic; // a diagnostic on a runtime fault, else absent
}
