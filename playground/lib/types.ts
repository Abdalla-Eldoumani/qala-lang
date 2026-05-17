// these mirror the phase 6 rust serde structs. field names are
// snake_case to match serde's default serialization -- the rust
// structs carry no #[serde(rename)]. verified against:
//   compiler/src/wasm.rs        CompileResult/RunResult/StepResult/ReplResult
//   compiler/src/vm.rs          VmState/StateValue/NamedValue
//   compiler/src/diagnostics.rs MonacoDiagnostic
//
// a rust Option<T> serializes as the value or json null, so disassembly,
// error, value, and category are union-with-null types, not optional `?`
// properties -- they are always present on the object, their value is null
// when the rust side was None.

// diagnostics.rs MonacoDiagnostic: severity 0 = warning, 1 = error.
export interface MonacoDiagnostic {
  line: number; // 1-based start line
  column: number; // 1-based start column
  end_line: number; // 1-based end line
  end_column: number; // 1-based end column (exclusive)
  severity: number; // 0 = warning, 1 = error
  message: string;
  category: string | null; // snake_case warning category; null for errors
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
  stack: StateValue[]; // value stack, bottom to top
  variables: NamedValue[]; // current frame's in-scope variables
  console: string[]; // accumulated print/println output
  leak_log: string[]; // resource-leak messages
}

// wasm.rs CompileResult.
export interface CompileResult {
  ok: boolean; // true on a clean compile (warnings ok)
  disassembly: string | null; // a string on success, null on error
  diagnostics: MonacoDiagnostic[]; // warnings on success, errors on failure
}

// wasm.rs RunResult.
export interface RunResult {
  ok: boolean; // true when the program ran to Halt
  state: VmState; // final or fault-point snapshot
  error: MonacoDiagnostic | null; // a diagnostic on a runtime fault
}

// wasm.rs StepResult. status is a plain string, not a tagged enum.
export interface StepResult {
  status: "ran" | "halted" | "error";
  state: VmState;
  error: MonacoDiagnostic | null;
}

// wasm.rs ReplResult -- declared now for phase 8's console; not used
// by phase 7's useQala surface.
export interface ReplResult {
  ok: boolean;
  value: StateValue | null;
  console: string[];
  error: MonacoDiagnostic | null;
}
