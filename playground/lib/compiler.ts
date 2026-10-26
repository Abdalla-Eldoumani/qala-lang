// the only module that imports from lib/wasm/. it dynamic-imports the
// wasm-pack glue, instantiates the .wasm from public/, constructs one
// Qala session, and returns a typed wrapper. it never throws for a
// compile/run error -- those are data in the ok-false result; it DOES
// throw if the browser cannot load or instantiate the module at all,
// which useQala catches.
//
// the casts in loadQala's returned methods are the single controlled any
// boundary in the playground: the generated qala_compiler.d.ts
// types every Qala method return as any, so the wrapper narrows them once.
import type {
  Arm64Result,
  CompileResult,
  ReplResult,
  RunResult,
  StepResult,
  VmState,
} from "./types";

// the wasm-pack glue's default export: the async init function.
// the generated qala_compiler.d.ts types it loosely; we narrow here.
type QalaInit = (
  moduleOrPath?: Response | Promise<Response> | URL | string,
) => Promise<unknown>;

type QalaClass = new () => QalaInstance;

interface QalaInstance {
  compile(source: string): unknown;
  compile_arm64(source: string): unknown;
  run(): unknown;
  step(): unknown;
  get_state(): unknown;
  disassemble(): unknown;
  repl_eval(source: string): unknown;
  reset(): unknown;
  free(): void;
}

export interface QalaSession {
  compile(source: string): CompileResult;
  // compile the current source through the ARM64 backend. a front-end
  // error or an unsupported-construct rejection comes back as an
  // ok-false Arm64Result, never a throw -- the same contract as compile.
  compileArm64(source: string): Arm64Result;
  run(): RunResult;
  step(): StepResult;
  getState(): VmState;
  disassemble(): string;
  // evaluate one Qala expression line through the rust repl vm. state
  // persists across calls on the same session. a malformed line comes
  // back as an ok-false ReplResult, never a throw.
  replEval(source: string): ReplResult;
  reset(): void;
  // drop the rust-side allocation. the wasm-pack glue also registers each
  // Qala in a FinalizationRegistry, so this is not required for correctness,
  // but a caller that abandons a session (an unmount mid-load) should free
  // it eagerly rather than wait for GC.
  free(): void;
}

export async function loadQala(): Promise<QalaSession> {
  // dynamic import keeps the wasm glue out of the server bundle and
  // off the initial client chunk; it loads only in the browser.
  const wasm = (await import("./wasm/qala_compiler.js")) as unknown as {
    default: QalaInit;
    Qala: QalaClass;
  };

  // instantiate from the copied public/ asset. await the fetch first and
  // pass the resolved Response: the glue's __wbg_init prototype-checks its
  // argument and logs a deprecation warning for anything that is not a
  // plain options object or undefined -- a bare Promise trips that check.
  // a Response still gets WebAssembly.instantiateStreaming, so streaming
  // instantiation is preserved.
  await wasm.default(await fetch("/wasm/qala_compiler_bg.wasm"));

  const inner = new wasm.Qala();

  // the one controlled `any` boundary: every Qala method returns a
  // serde JsValue typed `any` by the generated .d.ts. cast each call
  // into the interface that mirrors the Rust serde struct.
  return {
    compile: (source) => inner.compile(source) as CompileResult,
    compileArm64: (source) => inner.compile_arm64(source) as Arm64Result,
    run: () => inner.run() as RunResult,
    step: () => inner.step() as StepResult,
    getState: () => inner.get_state() as VmState,
    disassemble: () => inner.disassemble() as string,
    replEval: (source) => inner.repl_eval(source) as ReplResult,
    reset: () => {
      inner.reset();
    },
    free: () => {
      inner.free();
    },
  };
}
