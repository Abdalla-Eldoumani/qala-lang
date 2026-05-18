# the Qala language guide

This guide teaches Qala from scratch. It assumes you have written code in some
language before, but it assumes nothing about Qala. Read it top to bottom and
you will be able to write Qala programs.

Qala is a small, statically typed language. Every example here compiles and
runs. Most are the bundled examples you can open in the playground, so you can
read the code here and run the same code there.

This guide teaches how to write Qala. For why each feature exists, and how it
compares to other languages, see `language-spec.md`. The reference for the
built-in functions is `stdlib-reference.md`.

## hello, world

Here is a complete Qala program. It is the `hello` example.

```
fn main() is io {
    let name = "world"
    println("hello, {name}!")
}
```

Three things are worth naming.

`fn main()` declares a function called `main`. Like in C, `main` is where a
program starts.

`is io` is an effect annotation. It says `main` does input or output. Printing
counts as output, so `main` needs `is io`. The next section on effects covers
this in full.

`println("hello, {name}!")` prints a line. The `{name}` inside the string is
string interpolation: the part in braces is an expression, and its value is
spliced into the string. So this prints `hello, world!`.

A `let` binds a name to a value. `let name = "world"` makes `name` mean the
string `"world"` for the rest of the block. More on `let` in the immutability
section, but the short version is that a `let` binds a value once.

## types

Qala is statically typed. Every value has a type, and the type checker knows
every type before the program runs.

### primitive types

There are six primitive types:

- `i64`: a 64-bit signed integer.
- `f64`: a 64-bit floating point number.
- `bool`: `true` or `false`.
- `str`: a string of text.
- `byte`: a single byte.
- `void`: the type of a function that returns nothing.

### compound types

You build larger types out of the primitives:

- A fixed-size array, written `[i64; 5]`: five `i64` values, length known at
  compile time.
- A dynamic array, written `[i64]`: any number of `i64` values, length known at
  runtime.
- A tuple: a fixed group of values, each with its own type.
- A struct: named fields, each with its own type.
- An enum: a type that is one of several named variants. A variant can carry
  data.
- `Option<T>`: a value that is either `Some(T)` or `None`.
- `Result<T, E>`: a value that is either `Ok(T)` or `Err(E)`.
- A function type: the type of a function value you can pass around.

### type inference

You do not write the type of every value. Function signatures and struct fields
need explicit types. Local `let` bindings do not: the type checker reads the
type off the value you bind.

```
fn add(a: i64, b: i64) -> i64 is pure {
    let sum = a + b      // sum is i64, inferred from a + b
    return sum
}
```

`a` and `b` need their types written because they are parameters. `sum` does
not, because `a + b` is already known to be `i64`.

If you write a type that inference would have found anyway, the type checker
warns you. It is a redundant annotation. The fix is to delete the annotation.

## effects

The effect system is the center of Qala. It is how the language keeps its
promise that what you write is what happens.

Every function has an effect. The effect says what the function does to the
world beyond returning a value. There are four:

- `pure`: the function only computes. It cannot print, cannot grow an array,
  cannot fail at runtime. Given the same arguments, it always does the same
  thing.
- `io`: the function does input or output. It can use the console or a file
  handle.
- `alloc`: the function grows a dynamic array on the heap, with `push` or `pop`.
- `panic`: the function can fail at runtime, because it calls `assert`.

You write the effect after the signature with the `is` keyword.

```
fn square(x: i64) -> i64 is pure {
    return x * x
}

fn greet(name: str) is io {
    println("hello, {name}")
}
```

`square` is `is pure`: it takes a number and returns a number, nothing else.
`greet` is `is io`: it prints.

### the rule

A `pure` function cannot call an `io`, `alloc`, or `panic` function. This is the
rule the whole system rests on. If a function says it is pure, the compiler
checks that it is.

Here is the `effects` example. The commented-out function shows what the rule
catches.

```
// Pure functions only compute. They cannot print, allocate, or crash.
fn square(x: i64) -> i64 is pure {
    return x * x
}

// IO functions can do anything, including call pure functions.
fn display(label: str, value: i64) is io {
    println("{label}: {value}")
}

// Try uncommenting this -- the compiler catches the violation:
// fn broken(x: i64) -> i64 is pure {
//     println("oops")  // ERROR: pure function calls io function
//     return x
// }

fn main() is io {
    display("7 squared", square(7))
}
```

`display` is `is io`, and it calls `square`, which is `is pure`. That is fine: an
`io` function may call a `pure` one. The other direction is the error. Uncomment
`broken` and the compiler rejects it, because `broken` claims to be pure but
calls `println`, which is `io`.

### inferred effects

You can leave the effect off a function. The compiler infers it from the body.
If the body prints, the function is `io`. If the body only computes, it is
`pure`. Writing the effect is still worth doing: it states your intent, and the
compiler checks the body against it. An effect you wrote is a claim the compiler
verifies. An effect you left off is a fact the compiler reports.

## defer

`defer` schedules an expression to run when the current scope exits. It is how
Qala handles cleanup.

```
fn read(path: str) is io {
    let f = open(path)
    defer close(f)
    // ... use f here ...
}
```

`defer close(f)` does not run `close(f)` now. It runs it when the block ends.
Whichever way the block ends, `close(f)` runs first.

That last point is the reason `defer` is useful. A function can exit several
ways: it falls off the end, it hits a `return`, or an error propagates out of
it with `?`. A `defer` runs on all of them. You write the cleanup once, next to
the thing it cleans up, and you do not have to remember it at each exit.

If a scope has more than one `defer`, they run last-in, first-out. The last
`defer` you wrote runs first.

Here is the `defer-demo` example.

```
fn process_file(path: str) -> Result<str, str> is io {
    let f = open(path)
    defer close(f)

    // No matter how this function exits -- normal return,
    // early return, or error propagation -- close(f) runs.

    let content = f.read_all()?
    if len(content) == 0 {
        return Err("file is empty")
    }
    return Ok(content)
}

fn main() is io {
    let result = process_file("data.txt") or "no data"
    println("got: {result}")
}
```

`process_file` opens a handle and immediately defers closing it. After that the
function has three exits: the `?` on `read_all()` can return early with an
error, the empty-content check can `return Err(...)`, and the last line can
`return Ok(content)`. The `defer close(f)` covers all three.

The type checker helps here. If you `open` a handle in a scope and there is no
matching `defer close` in that scope, it warns you. A resource without cleanup
is a mistake the compiler points at.

## pattern matching

`match` checks a value against a set of patterns and runs the arm that matches.
Qala's `match` is exhaustive: you must cover every case. If you miss one, the
compiler tells you which.

Matching is most useful with enums. Here is the `pattern-matching` example.

```
enum Shape {
    Circle(f64),
    Rect(f64, f64),
    Triangle(f64, f64, f64),
}

fn area(s: Shape) -> f64 is pure {
    match s {
        Circle(r) => 3.14159 * r * r,
        Rect(w, h) => w * h,
        Triangle(a, b, c) => {
            let half = (a + b + c) / 2.0
            sqrt(half * (half - a) * (half - b) * (half - c))
        }
    }
}
```

`Shape` is an enum with three variants, and each variant carries data: a
`Circle` holds a radius, a `Rect` holds a width and a height, a `Triangle` holds
three sides.

`match s` checks the shape. Each arm names a variant and binds its data. In the
`Circle(r)` arm, `r` is the radius. In the `Rect(w, h)` arm, `w` and `h` are the
sides. The arm body uses those bindings.

An arm body is an expression or a block. The `Circle` and `Rect` arms are single
expressions. The `Triangle` arm is a block: it binds `half`, then computes the
area with it. A block arm's value is its last expression.

This `match` is exhaustive because it covers all three variants of `Shape`.
Delete the `Triangle` arm and the compiler stops you, naming `Triangle` as the
missing case.

### guards

A match arm can have an `if` guard. The arm matches only when its pattern
matches and its guard is true.

```
match score {
    s if s >= 90 => "A",
    s if s >= 80 => "B",
    _ => "below B",
}
```

The `_` is the wildcard pattern: it matches anything. A wildcard arm is how you
cover the cases the earlier arms did not.

A note on guards: when several arms are bare bindings with guards, the type
checker cannot prove the guards are disjoint, so it warns about overlapping
guards. The `classify` function in the `pattern-matching` example sidesteps this
by using plain `if` statements instead.

```
fn classify(value: i64) -> str is pure {
    if value > 0 { return "positive" }
    if value < 0 { return "negative" }
    return "zero"
}
```

For a value sorted by mutually exclusive numeric conditions, a chain of `if`
statements is often clearer than a guarded match anyway.

## the pipeline operator

The pipeline operator `|>` passes a value as the first argument to a function.
`x |> f(a)` means the same as `f(x, a)`.

It exists for readability. Nested calls read inside-out. A pipeline reads in the
order the data flows, top to bottom.

Here is the `pipeline` example.

```
fn double(x: i64) -> i64 is pure { return x * 2 }
fn add_one(x: i64) -> i64 is pure { return x + 1 }
fn is_even(x: i64) -> bool is pure { return x % 2 == 0 }

fn main() is io {
    // Pipeline: left-to-right data flow.
    // Each |> passes the result as the first argument.
    let result = 5
        |> double
        |> add_one
        |> double

    println("5 -> double -> add_one -> double = {result}")

    // With arrays
    let numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    let evens = numbers
        |> filter(is_even)
        |> map(double)

    for n in evens {
        println("{n}")
    }
}
```

The first pipeline starts with `5`, doubles it to `10`, adds one to get `11`,
doubles again to get `22`. Read top to bottom, each step takes the result of
the step above.

The second pipeline shows `|>` with the array functions. `numbers |> filter(is_even)`
keeps the even numbers, and `|> map(double)` doubles each one. `filter` and `map`
both take the array as their first argument, which is exactly what `|>` supplies.

## results and options

Qala has no exceptions. A function that can fail returns a `Result<T, E>`: it is
either `Ok(value)` on success or `Err(error)` on failure. A value that may be
absent is an `Option<T>`: either `Some(value)` or `None`.

Because errors are ordinary return values, you handle them with ordinary code.
Two pieces of syntax make that short.

The `?` operator propagates an error. `f.read_all()?` evaluates `read_all()`. If
the result is `Ok(content)`, the expression is `content` and the program goes
on. If it is `Err(e)`, the `?` returns `Err(e)` from the whole function right
there. You can only use `?` in a function whose return type is a compatible
`Result` or `Option`.

The `or` operator supplies a fallback. `expr or fallback` evaluates `expr`. If
it succeeds, you get the success value. If it fails, you get `fallback`. The
fallback is type-checked against the success type.

The `defer-demo` example uses both. Inside `process_file`, `f.read_all()?`
propagates a read error. In `main`, `process_file("data.txt") or "no data"`
supplies the string `"no data"` if `process_file` returns an `Err`.

## immutability

A `let` binds a name to a value. In Qala v1, a binding does not change. Once
`let x = 5` runs, `x` is `5` for the rest of the block. Values are bound, not
assigned.

```
let x = 5
let y = x + 1
```

This is not a missing feature you work around. It is how the language is. There
is no assignment statement in Qala v1. You do not reassign a binding, you do not
assign to an array element, you do not assign to a struct field. The `=` sign
appears in two places only: in a `let` that binds a value, and in a struct
literal that sets a field's starting value.

You will see `let mut` in some code. In v1, `let mut` parses, but it does
nothing a plain `let` does not. There is no operation that mutates a binding, so
the `mut` has no effect today. Treat `let mut` as reserved syntax. Write plain
`let`.

To compute a new value, bind a new name. The `add` function earlier binds `sum`
rather than changing `a` or `b`. Function parameters are immutable too. A
function works by returning values, not by changing the values it was given.

## structs and interfaces

A struct groups named fields.

```
struct Point { x: f64, y: f64 }
```

Each field needs an explicit type. You build a struct with a struct literal,
which is the one other place `=` appears:

```
let origin = Point { x: 0.0, y: 0.0 }
```

A method is a function attached to a type. You write it as
`fn TypeName.method(self, ...)`.

```
fn Point.distance_from_origin(self) -> f64 is pure {
    return sqrt(self.x * self.x + self.y * self.y)
}
```

An interface lists methods a type must have.

```
interface Printable {
    fn to_string(self) -> str
}
```

Qala interfaces are structural. A type satisfies an interface when it has the
methods the interface lists, with matching signatures. There is no `implements`
declaration. If `Point` has a `to_string(self) -> str` method, `Point` is
`Printable`, and you did not have to say so.

## comptime

An expression or block prefixed with `comptime` runs during compilation, not at
runtime. The result is baked into the program as a constant.

```
let minutes_per_week = comptime 7 * 24 * 60
```

The `7 * 24 * 60` is computed once, while the program compiles. The compiled
program just contains `10080`.

A `comptime` block can hold several statements, and its value is its last
expression:

```
let table_size = comptime {
    let rows = 8
    let cols = 8
    rows * cols
}
```

There are two limits. A `comptime` block can only call `pure` functions: it runs
before the program does, so there is no console to print to and no heap to grow.
And it runs under an instruction budget, so an accidental infinite loop fails
the compile instead of hanging it. The result must be a primitive or a string.

## string interpolation

You have seen interpolation throughout this guide. Inside a string literal,
`{expr}` evaluates `expr` and splices its value into the string.

```
let name = "Qala"
println("hello from {name}")
```

The expression in the braces can be any expression, not just a name:

```
println("the answer is {40 + 2}")
```

Interpolations can nest, and when a string has several of them, they evaluate
left to right in source order.

## where to go next

You now have the whole language. The bundled examples in
`playground/public/examples/` are the code from this guide; open them in the
playground and step through them.

For the design rationale, why effects instead of monads, why scope-bound
`defer`, why `comptime`, read `language-spec.md`. For the built-in functions
with their signatures and effects, read `stdlib-reference.md`.
