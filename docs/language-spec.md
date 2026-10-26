# Qala language specification

This is the full justified feature set for Qala v1. Every feature exists because of a specific problem in existing languages.

## 1. Effect annotations: `is pure`, `is io`, `is alloc`, `is panic`

Every mainstream language hides side effects. You call a function and have no idea if it prints, allocates, or might crash. Haskell solved this with monads, but monads are hard to teach.

Functions declare their effects. The compiler infers them if omitted. Pure functions cannot call impure ones. This catches bugs (accidentally logging inside math) and teaches students to think about what their code does to the outside world.

```
fn add(a: i64, b: i64) -> i64 is pure {
    return a + b
}

fn greet(name: str) is io {
    println("hello, {name}")
}
```

## 2. Scope-bound `defer` with static verification

C makes you remember to free memory. Rust makes you fight the borrow checker. Go has defer but it's function-scoped. Qala's defer runs at scope exit (not function exit), and the compiler warns if a resource is acquired without a matching defer.

```
fn process(path: str) is io {
    let f = open(path)
    defer close(f)
    // f is guaranteed closed when this block exits
}
```

## 3. `comptime` blocks

C uses preprocessor macros (fragile, untyped). C++ uses templates (complex, slow). Rust uses proc macros (separate crate, hard to debug). Zig proved that compile-time execution with the same language is simpler than all of these.

Any expression or block prefixed with `comptime` executes during compilation. The result is embedded as a constant. Same syntax, same types, same semantics as runtime code.

```
let minutes_per_week = comptime 7 * 24 * 60

let table_size = comptime {
    let rows = 8
    let cols = 8
    rows * cols
}
```

## 4. Pipeline operator `|>`

Nested function calls read inside-out: `reduce(map(filter(data, pred), transform), 0, sum)`. Humans read left to right. F#, Elixir, and OCaml have this. JavaScript has been trying to add it for years.

```
let result = data
    |> filter(is_positive)
    |> map(square)
    |> reduce(0, sum)
```

Desugars to `f(x, a, b)` from `x |> f(a, b)` during codegen.

## 5. `Result<T, E>` with `?` and `or`

Exceptions hide control flow. Go's `if err != nil` is verbose. Rust's `Result + ?` is good but has no inline fallback. Qala adds `or` for the common case where you want a default value on failure.

```
let config = parse_config(text) or Config.default()
let port = config.get("port")? // propagates error if missing
```

## 6. Exhaustive pattern matching with guards

Switch statements in C/Java silently fall through or miss cases. Qala's match requires every variant of an enum to be handled. The compiler tells you exactly which cases you forgot.

```
enum Direction { North, South, East, West }

fn opposite(d: Direction) -> Direction is pure {
    match d {
        North => South,
        South => North,
        East => West,
        // compiler error: missing case `West`
    }
}
```

Guard clauses extend matching with conditions:

```
match score {
    s if s >= 90 => "A",
    s if s >= 80 => "B",
    _ => "below B",
}
```

## 7. Structural interfaces

Java and C# require explicit `implements` declarations. If you don't control the type, you can't add an interface. Go proved that structural typing is simpler and more flexible.

```
interface Printable {
    fn to_string(self) -> str
}

struct Point { x: f64, y: f64 }

fn Point.to_string(self) -> str is pure {
    return "({self.x}, {self.y})"
}
// Point is Printable. No declaration needed.
```

## 8. String interpolation

Printf format strings are error-prone. String concatenation is verbose. Python f-strings and JavaScript template literals proved this is the right approach.

```
println("the answer is {40 + 2}")
```

Side effects in interpolations are evaluated left-to-right, in source order.

## 9. Immutable by default

Most bugs come from unexpected mutation. Rust, Swift, and Kotlin all default to immutable.

```
let x = 5         // bound once
let y = x + 1     // a new binding, not a change to x
```

A `let` binds a value once. Parameters and `let` bindings are immutable. Qala v1
has no assignment statement: a binding is never reassigned, and there is no
index or field assignment. `let mut` is reserved syntax that parses but has no
operative effect in v1. To compute a new value, bind a new name.

## 10. ARM64 compilation target

Qala has two codegen targets that share the same typed AST: a bytecode VM the playground runs, and a second backend that emits AArch64 assembly text in the CPSC 355 hosted-Linux dialect. The CLI picks between them with `qala build --target {bytecode,arm64}`.

The ARM64 backend's integer core is shipped: `i64` and `bool` arithmetic, comparisons and boolean logic, `if`/`while`/`for` control flow, `let` bindings in stack slots, AAPCS64 functions with calls and recursion, and integer `print`/`println` output lowered to `printf`. The emitted assembly is verified end to end: it assembles with `m4` and an AArch64 toolchain and runs correctly under QEMU. Floats, string values, and aggregates are still ahead; the compiler rejects them with a clear "the ARM64 backend does not yet support X" diagnostic rather than miscompile.

The emitted assembly is meant to be pasted directly into the [AArch64 playground](https://aarch64-playground.com) for visual instruction-by-instruction execution.

## Compiler quality features

### Constant folding

During code generation, if both operands of an arithmetic, comparison, or logical expression are compile-time constants, compute the result and emit a single `CONST` instruction instead of `CONST, CONST, OP`. Recursive: `1 + 2 + 3` folds to `6`.

### Dead code elimination

After a `return`, `break`, or `continue`, do not emit bytecode for subsequent statements in the same block. If an `if` condition is a compile-time constant, only emit the taken branch. The type checker emits a warning for unreachable code.

### Peephole optimization

After bytecode is emitted, scan for wasteful patterns:
- `PUSH X, POP` -> remove both
- `PUSH X, PUSH Y, ADD` with X, Y constants -> `PUSH (X+Y)`
- `NOT, JUMP_IF_FALSE` -> `JUMP_IF_TRUE`
- `PUSH true, JUMP_IF_FALSE` -> remove both
- `PUSH false, JUMP_IF_FALSE` -> unconditional `JUMP`
- `DUP, POP` -> remove both

Single pass over the instruction buffer. Match on 2-3 instruction windows.

### Rich diagnostics

Errors include the source line, an underline pointing to the problem, and a suggestion when applicable. Modeled on Rust's compiler output. The playground also receives structured data for Monaco inline error underlines.

```
error: pure function `compute` calls io function `println`
  --> main.qala:7:5
   |
7  |     println("debug")
   |     ^^^^^^^ this function has io effects
   |
   = note: pure functions cannot call functions with io effects
   = hint: remove the `is pure` annotation, or move the println outside this function
```

Five warning categories (yellow, non-blocking):
- Unused variables
- Unreachable code after return/break
- Unmatched defer (resource without cleanup)
- Shadowed variables
- Redundant type annotations

### Source maps

Every bytecode instruction records the source line it came from. Runtime errors report the exact source line. The playground uses this for current-line highlighting during step-through. Source maps survive optimization passes; peephole rewrites preserve the surviving instruction's line info.

### `type_of` intrinsic

`type_of(x)` returns the type name as a string at runtime. Useful for debugging and learning. Returns "i64", "f64", "bool", "str", "[i64]", "Shape::Circle", etc.

### REPL mode

The compiler supports REPL mode for the CLI. Each line compiles and executes immediately. Variables persist across lines. The playground's console panel also works as a REPL when no program is loaded.
