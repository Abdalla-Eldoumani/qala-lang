# Qala

A statically typed programming language where saying is doing.

Qala (قال) is the Arabic word for "he said" or "to say." In Arabic, when you say
something with qala, you are making a declaration, stating what is. Qala the
language works the same way. When you write a function marked `is pure`, it is
pure. The compiler guarantees it. When you write `defer close(f)`, the file
closes. The compiler guarantees that too. There is no gap between what you write
and what happens.

This is the core idea: a language where your code is a truthful declaration of
behavior. Side effects are visible in function signatures. Resources are managed
explicitly. Types are known before the program runs. Nothing is hidden, nothing
is implicit, nothing surprises you.

If Haskell showed that purity makes programs easier to reason about, and Rust
showed that compile-time guarantees prevent real bugs, Qala asks: what if we
took those insights and made them accessible? No monads, no borrow checker, no
lifetime annotations. Just clear annotations (`is pure`, `is io`),
straightforward resource management (`defer`), and a type system that catches
mistakes without getting in your way.

## try it

```bash
cd compiler && cargo test
wasm-pack build --target web --out-dir ../playground/lib/wasm
cd ../playground && npm install && npm run dev
```

Open http://localhost:3000.

## a taste of Qala

```
fn fibonacci(n: i64) -> i64 is pure {
    if n <= 1 { return n }
    return fibonacci(n - 1) + fibonacci(n - 2)
}

fn main() is io {
    for i in 0..15 {
        println("fib({i}) = {fibonacci(i)}")
    }
}
```

`fibonacci` is marked `is pure`. It takes a value and returns a value. It cannot
print, allocate, or crash. The compiler enforces this. `main` is marked `is io`
because it prints to the console. The effects are right there in the signature.
Reading the code tells you exactly what it does.

This is the `fibonacci` example, one of six bundled programs you can open and
run in the playground.

## what Qala offers

**Effect annotations.** Functions declare their side effects: `is pure`,
`is io`, `is alloc`, `is panic`. The compiler tracks these and catches
violations. This is how Qala keeps its promise: what you say is what happens.

**Scope-bound defer.** `defer close(f)` runs when the current scope exits. The
compiler warns if a resource is acquired without a matching cleanup. Resource
management is explicit, visible, and verified.

**Compile-time evaluation.** `comptime` blocks execute during compilation,
embedding the result as a constant. One concept replaces macros, const generics,
and metaprogramming.

**Pipeline operator.** `|>` passes the left value as the first argument to the
right function. Data flows left to right, the way you read it.

**Error handling.** `Result<T, E>` with `?` for propagation and `or` for inline
fallbacks. Errors are values, not surprises.

**Exhaustive matching.** `match` requires all enum variants covered. Guard
clauses supported. The compiler tells you what you forgot.

**Structural interfaces.** If a type has the right methods, it satisfies the
interface. No ceremony.

**Immutable bindings.** `let` binds a value once. Bindings do not change, and
there is no assignment statement. To get a new value, bind a new name. Fewer
surprises.

**String interpolation.** `{expr}` inside strings evaluates and converts.

## error messages

Qala's compiler produces rich diagnostics with source context, underlines, and
suggestions:

```
error: pure function `compute` calls io function `println`
  --> main.qala:7:5
   |
7  |     println("debug")
   |     ^^^^^^^
   |
   = note: pure functions cannot call functions with io effects
   = hint: remove the `is pure` annotation, or refactor to remove the io call
```

Warnings for unused variables, unreachable code, and unmatched defer help catch
mistakes without blocking compilation.

## documentation

- `docs/language-guide.md` teaches Qala from scratch.
- `docs/stdlib-reference.md` lists every built-in function.
- `docs/language-spec.md` is the full feature reference with design rationale.
- `docs/design-philosophy.md` is a short piece on the idea behind the language.

## how it works

The compiler is a single Rust crate with a hand-written lexer, recursive descent
parser with Pratt expression parsing, a single-pass type checker with effect
inference, and a bytecode compiler with constant folding and dead code
elimination. Emitted bytecode goes through a peephole optimizer before
execution. The VM uses NaN-boxing for compact stack values and carries source
maps for precise runtime error reporting. The web playground loads the entire
pipeline as WebAssembly.

## license

MIT
