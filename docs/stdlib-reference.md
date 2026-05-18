# Qala standard library reference

Qala v1 has fifteen built-in functions. They are written in Rust, not in Qala,
and the VM dispatches them directly. You do not import them. They are in scope
everywhere.

Each entry below gives the function's signature, its effect, what it does, and
a short example you can run. The signatures and effects here are copied from
`stdlib_signatures()` in `compiler/src/typechecker.rs`, which is the source of
truth. If a doc and the compiler ever disagree, the compiler is right.

## a note on effects

Every function carries one of the four effects: `pure`, `io`, `alloc`, or
`panic`. The effect is part of the function's contract. A `pure` function that
calls an `io` function is a compile error. See the effects section of the
language guide for what each effect means.

Three functions need a footnote. `map`, `filter`, and `reduce` show `pure` in
the table, but that is a floor, not a promise. They run a callback you pass
them, and they inherit the callback's effect. Pass `map` a callback that prints,
and the `map` call is `io`. The `pure` in the table is what the call costs when
the callback is pure.

## a note on files

File I/O in Qala v1 is a mock. The VM compiles to WebAssembly and runs in a
browser, where there is no filesystem to reach. `open` does not open a real
file. It builds an in-memory file handle with the path you gave it and empty
content. `close` marks that handle closed. `read_all` returns the empty content
as `Ok("")`.

This is a deliberate v1 limit. It is enough to show the effect system and
`defer close` working without a real file API. `print` and `println` are the
same kind of mock: they write to the VM's console buffer, not to real stdout.

## the fifteen functions

### print

```
print(str) -> void    is io
```

Render the argument and append it to the console. No trailing newline.

```
fn main() is io {
    print("one ")
    print("line")
}
// console: one line
```

### println

```
println(str) -> void    is io
```

Render the argument and append it to the console as its own line.

```
fn main() is io {
    println("first")
    println("second")
}
// console:
// first
// second
```

### sqrt

```
sqrt(f64) -> f64    is pure
```

The square root of an `f64`.

```
fn hypotenuse(a: f64, b: f64) -> f64 is pure {
    return sqrt(a * a + b * b)
}
```

### abs

```
abs(i64) -> i64    is pure
abs(f64) -> f64    is pure
```

Absolute value. `abs` is generic over the number types. It returns the same
type you passed in: `abs(-3)` gives an `i64`, `abs(-1.5)` gives an `f64`.

```
fn main() is io {
    println("{abs(-3)}")     // 3
    println("{abs(-1.5)}")   // 1.5
}
```

### assert

```
assert(bool) -> void    is panic
```

Check a condition. A `false` argument is a runtime error that stops the
program. This is the only built-in with the `panic` effect, because it is the
only one that can fail at runtime by design.

```
fn main() is panic {
    assert(2 + 2 == 4)
}
```

### len

```
len([T]) -> i64    is pure
```

The element count of an array, or the character count of a string. `len` is
generic: it works on any array element type and on strings.

```
fn main() is io {
    let xs = [10, 20, 30]
    println("{len(xs)}")        // 3
    println("{len("qala")}")    // 4
}
```

### push

```
push([T], T) -> void    is alloc
```

Append a value to a dynamic array. `push` grows the array on the heap, so it
carries the `alloc` effect. It is generic over the element type.

```
fn build() -> [i64] is alloc {
    let xs = [1, 2]
    push(xs, 3)
    return xs
}
```

### pop

```
pop([T]) -> Option<T>    is alloc
```

Remove the last element of a dynamic array and return it as `Option<T>`. An
empty array gives `None`. Like `push`, it touches the heap and carries `alloc`.

```
fn last(xs: [i64]) -> Option<i64> is alloc {
    return pop(xs)
}
```

### type_of

```
type_of(T) -> str    is pure
```

The runtime type name of a value, as a string. Useful while learning or
debugging. It is generic: it accepts a value of any type.

```
fn main() is io {
    println(type_of(42))      // i64
    println(type_of("hi"))    // str
    println(type_of(true))    // bool
}
```

### open

```
open(str) -> FileHandle    is io
```

Open a file handle for the given path. In v1 this is a mock: it builds an
in-memory handle with empty content (see the note on files above). It carries
`io` because opening a file is an effect on the world, even when the file is a
mock.

```
fn main() is io {
    let f = open("data.txt")
    defer close(f)
}
```

### close

```
close(FileHandle) -> void    is io
```

Mark a file handle closed. Pair it with `open` using `defer` so the handle is
closed on every exit path. The type checker warns if you `open` a handle in a
scope with no matching `defer close`.

```
fn read(path: str) is io {
    let f = open(path)
    defer close(f)
}
```

### map

```
map([T], fn(T) -> U) -> [U]    is pure
```

Apply a callback to each element of an array and collect the results into a new
array. Generic over both the input and output element types. `map` inherits the
callback's effect: a pure callback keeps the call pure, an `io` callback makes
the call `io`.

```
fn double(x: i64) -> i64 is pure { return x * 2 }

fn main() is io {
    let doubled = map([1, 2, 3], double)
    for n in doubled {
        println("{n}")        // 2, 4, 6
    }
}
```

### filter

```
filter([T], fn(T) -> bool) -> [T]    is pure
```

Keep the elements of an array for which the callback returns `true`. The result
is a new array of the same element type. Like `map`, `filter` inherits the
callback's effect.

```
fn is_even(x: i64) -> bool is pure { return x % 2 == 0 }

fn main() is io {
    let evens = filter([1, 2, 3, 4], is_even)
    for n in evens {
        println("{n}")        // 2, 4
    }
}
```

### reduce

```
reduce([T], U, fn(U, T) -> U) -> U    is pure
```

Fold an array into a single value. You give a starting value and a callback;
the callback combines the running total with each element. Like `map` and
`filter`, `reduce` inherits the callback's effect.

```
fn add(total: i64, x: i64) -> i64 is pure { return total + x }

fn main() is io {
    let sum = reduce([1, 2, 3, 4], 0, add)
    println("{sum}")          // 10
}
```

### read_all

```
FileHandle.read_all() -> Result<str, str>    is io
```

Read a file handle's full content. This is the only stdlib method in v1: you
call it on a `FileHandle` value with method syntax, `f.read_all()`, not as a
free function. It returns a `Result`, so you can use `?` to propagate a read
error. In v1 the mock content is empty, so the call returns `Ok("")`.

```
fn process(path: str) -> Result<str, str> is io {
    let f = open(path)
    defer close(f)
    let content = f.read_all()?
    return Ok(content)
}
```
