//! the native-Rust standard library.
//!
//! Qala's standard library is fifteen functions implemented directly in Rust,
//! not in Qala. codegen assigns each one a reserved fn-id at or above
//! [`STDLIB_FN_BASE`] (40000); the VM's `CALL` handler routes any such fn-id
//! here through [`dispatch`]. user functions get dense ids `0..N` into
//! `Program.chunks`; the two id spaces never overlap.
//!
//! the functions, in the order their fn-ids are assigned (the order MUST match
//! `codegen.rs`'s `STDLIB_TABLE` -- entry `i` is fn-id `STDLIB_FN_BASE + i`):
//!
//! | fn-id | name       | effect | what it does |
//! |-------|------------|--------|--------------|
//! | 40000 | `print`    | io     | render the argument, append to the console |
//! | 40001 | `println`  | io     | render the argument, append a console line |
//! | 40002 | `sqrt`     | pure   | `f64` square root |
//! | 40003 | `abs`      | pure   | absolute value of an `i64` or an `f64` |
//! | 40004 | `assert`   | panic  | a `false` argument is a runtime error |
//! | 40005 | `len`      | pure   | element count of an array, or char count of a string |
//! | 40006 | `push`     | alloc  | append a value to a heap array |
//! | 40007 | `pop`      | alloc  | remove the last element, return `Option<T>` |
//! | 40008 | `type_of`  | pure   | the runtime type name as a string |
//! | 40009 | `open`     | io     | open a mock file handle |
//! | 40010 | `close`    | io     | mark a file handle closed |
//! | 40011 | `map`      | pure*  | apply a callback to each array element |
//! | 40012 | `filter`   | pure*  | keep the elements a callback accepts |
//! | 40013 | `reduce`   | pure*  | fold an array with a callback |
//! | 40014 | `read_all` | io     | read a file handle's content (`FileHandle.read_all`) |
//!
//! (`map` / `filter` / `reduce` inherit the callback's effect; Phase 3's effect
//! inference already propagated that into the caller, so the native code here
//! does not re-check effects.)
//!
//! the effect column is consistent with the typechecker's `stdlib_signatures`
//! table -- same fifteen functions, same effects. the typechecker owns the
//! signature; this module owns the runtime behavior.
//!
//! file I/O is a mock. `open` does NOT touch a real filesystem: the VM compiles
//! to WASM and runs in a browser sandbox where there is no filesystem to reach.
//! `open` builds a [`HeapObject::FileHandle`] with the requested path and an
//! empty mock content string; `close` flips its `closed` flag; `read_all`
//! returns the (empty) content as `Ok`. this is a deliberate v1 limitation --
//! it makes the effect system and `defer close(f)` demonstrable in the
//! playground without a real file API. `print` / `println` likewise write to
//! the VM's console buffer, not real stdout.
//!
//! every function is total against untrusted bytecode: the fn-id and the
//! argument values come from a `CALL` instruction the VM does not trust, so a
//! wrong-arity or wrong-type call is a [`QalaError::Runtime`], never a panic.
//! [`dispatch`] rejects an unknown fn-id the same way.

use crate::errors::QalaError;
use crate::opcode::STDLIB_FN_BASE;
use crate::value::Value;
use crate::vm::{HeapObject, Vm};

/// the `MAKE_ENUM_VARIANT` id of `Result::Ok`.
///
/// codegen pre-registers the four built-in `Result` / `Option` variants in
/// `Program.enum_variant_names` before any user variant: `Ok` = 0, `Err` = 1,
/// `Some` = 2, `None` = 3. `read_all` builds an `Ok` value and `pop` builds a
/// `Some` / `None` value directly as [`HeapObject::EnumVariant`]s, so they
/// carry the variant names rather than these ids -- the ids are recorded here
/// only to document the codegen contract the names mirror.
#[allow(dead_code)]
const VARIANT_ID_OK: u16 = 0;

/// dispatch a stdlib `CALL` to its native implementation.
///
/// `fn_id` is the `u16` from the `CALL` opcode; the VM routes any `fn_id` at or
/// above [`STDLIB_FN_BASE`] here. `args` are the call's argument values in
/// source order -- `args[0]` is the leftmost argument (for a method call such
/// as `FileHandle.read_all`, `args[0]` is the receiver). the returned [`Value`]
/// is what the VM pushes onto the value stack for the instruction after the
/// `CALL`; a `void`-returning function returns [`Value::void`].
///
/// the `match` lists the functions in the exact `STDLIB_TABLE` order: entry `i`
/// in that table is `STDLIB_FN_BASE + i`, so `print` is 40000, `read_all` is
/// 40014. a `fn_id` below [`STDLIB_FN_BASE`] reaching here is a VM-internal
/// invariant violation (the `CALL` handler should have routed it to a user
/// frame); an unknown `fn_id` at or above the base is malformed bytecode.
/// both are a clean [`QalaError::Runtime`], never a panic.
pub fn dispatch(vm: &mut Vm, fn_id: u16, args: &[Value]) -> Result<Value, QalaError> {
    if fn_id < STDLIB_FN_BASE {
        return Err(vm.runtime_err(&format!(
            "internal error: fn-id {fn_id} is not a stdlib id"
        )));
    }
    match fn_id {
        40000 => print(vm, args),
        40001 => println(vm, args),
        40002 => sqrt(vm, args),
        40003 => abs(vm, args),
        40004 => assert(vm, args),
        40005 => len(vm, args),
        40006 => push(vm, args),
        40007 => pop(vm, args),
        40008 => type_of(vm, args),
        40009 => open(vm, args),
        40010 => close(vm, args),
        40011 => map(vm, args),
        40012 => filter(vm, args),
        40013 => reduce(vm, args),
        40014 => read_all(vm, args),
        other => Err(vm.runtime_err(&format!("unknown stdlib function {other}"))),
    }
}

// ---- argument helpers ------------------------------------------------------

/// the single argument of a one-argument stdlib function.
///
/// a wrong argument count is malformed bytecode (the typechecker checked the
/// arity, so this only fires on a hand-crafted or corrupt `CALL`) -- a clean
/// `Runtime` error, never an index panic.
fn one_arg(vm: &Vm, args: &[Value], name: &str) -> Result<Value, QalaError> {
    match args {
        [a] => Ok(*a),
        _ => Err(vm.runtime_err(&format!(
            "{name} expects 1 argument, got {}",
            args.len()
        ))),
    }
}

/// allocate a [`HeapObject::Int`] for `n` and return a pointer to it. a heap
/// exhaustion is a `Runtime` error.
fn alloc_int(vm: &mut Vm, n: i64) -> Result<Value, QalaError> {
    let slot = vm
        .heap
        .alloc(HeapObject::Int(n))
        .ok_or_else(|| vm.runtime_err("heap exhausted"))?;
    Ok(Value::pointer(slot))
}

/// allocate a [`HeapObject::Str`] for `s` and return a pointer to it. a heap
/// exhaustion is a `Runtime` error.
fn alloc_str(vm: &mut Vm, s: String) -> Result<Value, QalaError> {
    let slot = vm
        .heap
        .alloc(HeapObject::Str(s))
        .ok_or_else(|| vm.runtime_err("heap exhausted"))?;
    Ok(Value::pointer(slot))
}

/// allocate a [`HeapObject::Array`] for `items` and return a pointer to it. a
/// heap exhaustion is a `Runtime` error.
fn alloc_array(vm: &mut Vm, items: Vec<Value>) -> Result<Value, QalaError> {
    let slot = vm
        .heap
        .alloc(HeapObject::Array(items))
        .ok_or_else(|| vm.runtime_err("heap exhausted"))?;
    Ok(Value::pointer(slot))
}

/// build an `Option` enum-variant value: `Some(payload)` when `payload` is
/// `Some`, `None` when it is `None`.
///
/// codegen registers `Some` / `None` as built-in `Option` variants; the heap
/// object carries the `(enum, variant)` names, which is what `MATCH_VARIANT`
/// and the value renderer key on. a heap exhaustion is a `Runtime` error.
fn make_option(vm: &mut Vm, payload: Option<Value>) -> Result<Value, QalaError> {
    let object = match payload {
        Some(v) => HeapObject::EnumVariant {
            type_name: "Option".to_string(),
            variant: "Some".to_string(),
            payload: vec![v],
        },
        None => HeapObject::EnumVariant {
            type_name: "Option".to_string(),
            variant: "None".to_string(),
            payload: Vec::new(),
        },
    };
    let slot = vm
        .heap
        .alloc(object)
        .ok_or_else(|| vm.runtime_err("heap exhausted"))?;
    Ok(Value::pointer(slot))
}

/// build a `Result::Ok(payload)` enum-variant value. used by `read_all`. a heap
/// exhaustion is a `Runtime` error.
fn make_ok(vm: &mut Vm, payload: Value) -> Result<Value, QalaError> {
    let slot = vm
        .heap
        .alloc(HeapObject::EnumVariant {
            type_name: "Result".to_string(),
            variant: "Ok".to_string(),
            payload: vec![payload],
        })
        .ok_or_else(|| vm.runtime_err("heap exhausted"))?;
    Ok(Value::pointer(slot))
}

// ---- the fifteen native functions ------------------------------------------

/// `print(x)` -- render `x` and append it to the VM console.
///
/// the console is a `Vec<String>` -- one entry per call. `print` pushes its
/// rendered text as its own entry. `print` is specified as the no-newline
/// counterpart of `println`, but the v1 console buffer is a flat list of
/// strings with no sub-line cursor, so a `print` entry is shown as its own
/// console line; true mid-line concatenation (`print("a"); print("b")`
/// rendering as one line `ab`) is a documented v1 limitation -- a console with
/// a line cursor is a future enhancement. the bundled examples use only
/// `println`. returns `void`.
fn print(vm: &mut Vm, args: &[Value]) -> Result<Value, QalaError> {
    let arg = one_arg(vm, args, "print")?;
    let text = vm.value_to_string(arg);
    vm.console.push(text);
    Ok(Value::void())
}

/// `println(x)` -- render `x` and append it to the VM console as a line.
///
/// each `println` pushes one console entry with a trailing `'\n'`. the
/// playground renders one entry per line, so the newline is typically invisible
/// when the console displays entries individually, but consumers that
/// concatenate entries into a single text block get the correct newline
/// separation between lines (and no trailing newline after `print` entries).
/// returns `void`.
fn println(vm: &mut Vm, args: &[Value]) -> Result<Value, QalaError> {
    let arg = one_arg(vm, args, "println")?;
    let mut text = vm.value_to_string(arg);
    text.push('\n');
    vm.console.push(text);
    Ok(Value::void())
}

/// `sqrt(x: f64) -> f64` -- the IEEE 754 square root.
///
/// a negative argument yields `NaN` (IEEE 754), not an error. a non-`f64`
/// argument is a `Runtime` error.
fn sqrt(vm: &mut Vm, args: &[Value]) -> Result<Value, QalaError> {
    let arg = one_arg(vm, args, "sqrt")?;
    let x = arg
        .as_f64()
        .ok_or_else(|| vm.runtime_err("sqrt: expected a float"))?;
    Ok(Value::from_f64(x.sqrt()))
}

/// `abs(x)` -- the absolute value of an `i64` or an `f64`.
///
/// `abs` is generic: the typechecker resolves the return type to the argument
/// type, so the native code inspects the runtime kind. an `i64` argument (a
/// pointer to a heap `Int`) returns a heap `Int`; `i64::MIN` overflows `i64`'s
/// range and is a `Runtime` error rather than a wrap. an `f64` argument returns
/// an `f64`. any other kind is a `Runtime` error.
fn abs(vm: &mut Vm, args: &[Value]) -> Result<Value, QalaError> {
    let arg = one_arg(vm, args, "abs")?;
    // an f64 is stored inline; try that first.
    if let Some(x) = arg.as_f64() {
        return Ok(Value::from_f64(x.abs()));
    }
    // otherwise the argument must be a pointer to a heap Int.
    let int = arg
        .as_pointer()
        .and_then(|slot| match vm.heap.get(slot) {
            Some(HeapObject::Int(n)) => Some(*n),
            _ => None,
        })
        .ok_or_else(|| vm.runtime_err("abs: expected an integer or a float"))?;
    let result = int
        .checked_abs()
        .ok_or_else(|| vm.runtime_err("abs: integer overflow"))?;
    alloc_int(vm, result)
}

/// `assert(condition: bool)` -- a `false` condition aborts the program.
///
/// a `true` condition is a no-op returning `void`; a `false` condition is a
/// `Runtime` "assertion failed". a non-`bool` argument is a `Runtime` error.
fn assert(vm: &mut Vm, args: &[Value]) -> Result<Value, QalaError> {
    let arg = one_arg(vm, args, "assert")?;
    let condition = arg
        .as_bool()
        .ok_or_else(|| vm.runtime_err("assert: expected a boolean"))?;
    if condition {
        Ok(Value::void())
    } else {
        Err(vm.runtime_err("assertion failed"))
    }
}

/// `len(collection) -> i64` -- the length of an array, a tuple, or a string.
///
/// an array's / tuple's length is its element count; a string's length is its
/// count of Unicode scalar values (`chars().count()`), matching the
/// VM's `LEN` opcode. the result is a heap `Int`. any other value is a
/// `Runtime` error.
fn len(vm: &mut Vm, args: &[Value]) -> Result<Value, QalaError> {
    let arg = one_arg(vm, args, "len")?;
    let slot = arg
        .as_pointer()
        .ok_or_else(|| vm.runtime_err("len: expected an array or string"))?;
    let length = match vm.heap.get(slot) {
        Some(HeapObject::Array(items)) | Some(HeapObject::Tuple(items)) => items.len(),
        Some(HeapObject::Str(s)) => s.chars().count(),
        _ => return Err(vm.runtime_err("len: expected an array or string")),
    };
    alloc_int(vm, length as i64)
}

/// `push(array, value)` -- append `value` to a heap array in place.
///
/// the first argument is an array pointer, the second the value to append. the
/// array is mutated through the heap. the typechecker types `push` as
/// returning `void`, so this returns `void`. a non-array first argument is a
/// `Runtime` error.
fn push(vm: &mut Vm, args: &[Value]) -> Result<Value, QalaError> {
    let (array, value) = match args {
        [a, v] => (*a, *v),
        _ => {
            return Err(vm.runtime_err(&format!(
                "push expects 2 arguments, got {}",
                args.len()
            )));
        }
    };
    let slot = array
        .as_pointer()
        .ok_or_else(|| vm.runtime_err("push: expected an array"))?;
    match vm.heap.get_mut(slot) {
        Some(HeapObject::Array(items)) => {
            items.push(value);
            Ok(Value::void())
        }
        _ => Err(vm.runtime_err("push: expected an array")),
    }
}

/// `pop(array) -> Option<T>` -- remove and return the last element.
///
/// the argument is an array pointer. the array is mutated through the heap; the
/// removed element is wrapped `Some(element)`, and an empty array yields
/// `None`. a non-array argument is a `Runtime` error.
fn pop(vm: &mut Vm, args: &[Value]) -> Result<Value, QalaError> {
    let arg = one_arg(vm, args, "pop")?;
    let slot = arg
        .as_pointer()
        .ok_or_else(|| vm.runtime_err("pop: expected an array"))?;
    let removed = match vm.heap.get_mut(slot) {
        Some(HeapObject::Array(items)) => items.pop(),
        _ => return Err(vm.runtime_err("pop: expected an array")),
    };
    make_option(vm, removed)
}

/// `type_of(x) -> str` -- the runtime type name of `x` as a string.
///
/// the type name comes from [`Vm::runtime_type_name`] -- the same helper
/// `get_state` uses to type-tint a value -- so `type_of` and the playground's
/// type display always agree. the result is a heap `Str`: `"i64"`, `"f64"`,
/// `"bool"`, `"str"`, `"byte"`, `"void"`, `"[i64]"`, `"(i64, str)"`, a struct's
/// declared name, `"Enum::Variant"`, and so on.
fn type_of(vm: &mut Vm, args: &[Value]) -> Result<Value, QalaError> {
    let arg = one_arg(vm, args, "type_of")?;
    let name = vm.runtime_type_name(arg);
    alloc_str(vm, name)
}

/// `open(path: str) -> FileHandle` -- open a mock file handle.
///
/// the argument is the path string. the VM does no real file I/O (it runs in a
/// WASM sandbox); `open` builds a [`HeapObject::FileHandle`] carrying the path,
/// an empty mock content string, and `closed = false`. the mock content is
/// always empty in v1 -- a real virtual filesystem is deferred. a non-string
/// argument is a `Runtime` error.
fn open(vm: &mut Vm, args: &[Value]) -> Result<Value, QalaError> {
    let arg = one_arg(vm, args, "open")?;
    let path_slot = arg
        .as_pointer()
        .ok_or_else(|| vm.runtime_err("open: expected a string path"))?;
    let path = match vm.heap.get(path_slot) {
        Some(HeapObject::Str(s)) => s.clone(),
        _ => return Err(vm.runtime_err("open: expected a string path")),
    };
    let slot = vm
        .heap
        .alloc(HeapObject::FileHandle {
            path,
            content: String::new(),
            closed: false,
        })
        .ok_or_else(|| vm.runtime_err("heap exhausted"))?;
    Ok(Value::pointer(slot))
}

/// `close(handle: FileHandle)` -- mark a file handle closed.
///
/// the argument is a file-handle pointer; `close` flips its `closed` flag.
/// closing an already-closed handle is a harmless no-op. returns `void`. a
/// non-handle argument is a `Runtime` error.
fn close(vm: &mut Vm, args: &[Value]) -> Result<Value, QalaError> {
    let arg = one_arg(vm, args, "close")?;
    let slot = arg
        .as_pointer()
        .ok_or_else(|| vm.runtime_err("close: expected a file handle"))?;
    match vm.heap.get_mut(slot) {
        Some(HeapObject::FileHandle { closed, .. }) => {
            *closed = true;
            Ok(Value::void())
        }
        _ => Err(vm.runtime_err("close: expected a file handle")),
    }
}

/// `map(array, callback) -> array` -- apply `callback` to each element.
///
/// the first argument is an array pointer, the second a function value (a user
/// callback). the elements are copied into a Rust-local `Vec` first, then each
/// is passed to [`Vm::call_function_value`], which re-enters the VM's call
/// machinery; the results collect into a fresh heap array. the per-element loop
/// state -- the element list, the result vec -- lives in Rust locals, so a
/// callback that itself calls `map` is re-entrant and does not corrupt VM
/// state. a non-array first argument is a `Runtime` error.
fn map(vm: &mut Vm, args: &[Value]) -> Result<Value, QalaError> {
    let (array, callback) = match args {
        [a, c] => (*a, *c),
        _ => {
            return Err(vm.runtime_err(&format!(
                "map expects 2 arguments, got {}",
                args.len()
            )));
        }
    };
    let elements = heap_array_elements(vm, array, "map")?;
    let mut results: Vec<Value> = Vec::with_capacity(elements.len());
    for element in elements {
        let mapped = vm.call_function_value(callback, &[element])?;
        results.push(mapped);
    }
    alloc_array(vm, results)
}

/// `filter(array, callback) -> array` -- keep the elements `callback` accepts.
///
/// the callback returns a `bool`; an element is kept when the callback returns
/// `true`. like [`map`], the loop state stays in Rust locals, so a nested
/// `filter` (or `map`) inside the callback is re-entrant. a non-array first
/// argument, or a callback that returns a non-`bool`, is a `Runtime` error.
fn filter(vm: &mut Vm, args: &[Value]) -> Result<Value, QalaError> {
    let (array, callback) = match args {
        [a, c] => (*a, *c),
        _ => {
            return Err(vm.runtime_err(&format!(
                "filter expects 2 arguments, got {}",
                args.len()
            )));
        }
    };
    let elements = heap_array_elements(vm, array, "filter")?;
    let mut kept: Vec<Value> = Vec::new();
    for element in elements {
        let verdict = vm.call_function_value(callback, &[element])?;
        let keep = verdict
            .as_bool()
            .ok_or_else(|| vm.runtime_err("filter: the callback must return a boolean"))?;
        if keep {
            kept.push(element);
        }
    }
    alloc_array(vm, kept)
}

/// `reduce(array, initial, callback) -> value` -- fold an array.
///
/// the arguments are an array pointer, an initial accumulator value, and a
/// callback. for each element the callback is called with `(accumulator,
/// element)` and its result becomes the new accumulator; the final accumulator
/// is returned. the accumulator lives in a Rust local, so a callback that
/// itself folds is re-entrant. a non-array first argument is a `Runtime` error.
fn reduce(vm: &mut Vm, args: &[Value]) -> Result<Value, QalaError> {
    let (array, initial, callback) = match args {
        [a, i, c] => (*a, *i, *c),
        _ => {
            return Err(vm.runtime_err(&format!(
                "reduce expects 3 arguments, got {}",
                args.len()
            )));
        }
    };
    let elements = heap_array_elements(vm, array, "reduce")?;
    let mut accumulator = initial;
    for element in elements {
        accumulator = vm.call_function_value(callback, &[accumulator, element])?;
    }
    Ok(accumulator)
}

/// `FileHandle.read_all(self) -> Result<str, str>` -- read a handle's content.
///
/// `read_all` is the only stdlib method in v1; codegen emits the receiver as
/// argument 0, so `args[0]` is the file-handle pointer. the mock handle's
/// content (empty in v1 -- see the module doc) is returned as `Ok(content)`.
/// the `Result` lets a Qala program use `?` / `or` on the read, which is what
/// `defer-demo.qala` exercises. a non-handle receiver is a `Runtime` error.
fn read_all(vm: &mut Vm, args: &[Value]) -> Result<Value, QalaError> {
    let arg = one_arg(vm, args, "read_all")?;
    let slot = arg
        .as_pointer()
        .ok_or_else(|| vm.runtime_err("read_all: expected a file handle"))?;
    let content = match vm.heap.get(slot) {
        Some(HeapObject::FileHandle { content, .. }) => content.clone(),
        _ => return Err(vm.runtime_err("read_all: expected a file handle")),
    };
    let payload = alloc_str(vm, content)?;
    make_ok(vm, payload)
}

/// copy a heap array's elements into an owned `Vec`.
///
/// `map` / `filter` / `reduce` all need the element list as a Rust local before
/// they re-enter the VM (so the iteration does not hold a borrow of the heap
/// across a callback that may itself allocate). `value` must point at a
/// [`HeapObject::Array`]; anything else is a `Runtime` error naming the calling
/// function.
fn heap_array_elements(vm: &Vm, value: Value, name: &str) -> Result<Vec<Value>, QalaError> {
    let slot = value
        .as_pointer()
        .ok_or_else(|| vm.runtime_err(&format!("{name}: expected an array")))?;
    match vm.heap.get(slot) {
        Some(HeapObject::Array(items)) => Ok(items.clone()),
        _ => Err(vm.runtime_err(&format!("{name}: expected an array"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::{Chunk, Program};
    use crate::codegen::compile_program;
    use crate::lexer::Lexer;
    use crate::parser::Parser;
    use crate::typechecker::check_program;

    /// a VM over a single empty `main` chunk -- enough to exercise a stdlib
    /// function that does not re-enter the VM (everything except
    /// `map` / `filter` / `reduce`). the native functions take `&mut Vm` for
    /// the heap and the console, not for a running frame, so an empty `main`
    /// is a sufficient fixture.
    fn bare_vm() -> Vm {
        let mut chunk = Chunk::new();
        chunk.write_op(crate::opcode::Opcode::Return, 1);
        let mut program = Program::new();
        program.chunks.push(chunk);
        program.fn_names.push("main".to_string());
        program.main_index = 0;
        Vm::new(program, String::new())
    }

    /// compile a Qala program through the full pipeline. used to build a VM
    /// whose `Program` carries real callback chunks for the `map` / `filter` /
    /// `reduce` tests. panics on any pipeline error -- the fixtures are all
    /// known-good source.
    fn compile(src: &str) -> Program {
        let tokens = Lexer::tokenize(src).expect("lex");
        let ast = Parser::parse(&tokens).expect("parse");
        let (typed, errors, _) = check_program(&ast, src);
        assert!(errors.is_empty(), "typecheck errors: {errors:?}");
        compile_program(&typed, src).expect("codegen")
    }

    /// the fn-id of the user function named `name` in `program`.
    fn fn_id(program: &Program, name: &str) -> u16 {
        program
            .fn_names
            .iter()
            .position(|n| n == name)
            .unwrap_or_else(|| panic!("no function named {name}"))
            as u16
    }

    /// allocate a heap `Int` in `vm` and return the pointer value.
    fn int(vm: &mut Vm, n: i64) -> Value {
        alloc_int(vm, n).expect("alloc int")
    }

    /// allocate a heap `Str` in `vm` and return the pointer value.
    fn string(vm: &mut Vm, s: &str) -> Value {
        alloc_str(vm, s.to_string()).expect("alloc str")
    }

    /// the `i64` a result value decodes to -- a pointer to a heap `Int`.
    fn result_i64(vm: &Vm, value: Value) -> i64 {
        let slot = value.as_pointer().expect("a heap pointer");
        match vm.heap.get(slot) {
            Some(HeapObject::Int(n)) => *n,
            _ => panic!("the value does not point at a heap Int"),
        }
    }

    /// the `str` a result value decodes to -- a pointer to a heap `Str`.
    fn result_str(vm: &Vm, value: Value) -> String {
        let slot = value.as_pointer().expect("a heap pointer");
        match vm.heap.get(slot) {
            Some(HeapObject::Str(s)) => s.clone(),
            _ => panic!("the value does not point at a heap Str"),
        }
    }

    /// the `Vec<Value>` a result value decodes to -- a pointer to a heap array.
    fn result_array(vm: &Vm, value: Value) -> Vec<Value> {
        let slot = value.as_pointer().expect("a heap pointer");
        match vm.heap.get(slot) {
            Some(HeapObject::Array(items)) => items.clone(),
            _ => panic!("the value does not point at a heap array"),
        }
    }

    /// the rendered form of a finished program's result.
    ///
    /// after `run()`, a program's `RETURN` from `main` leaves the result on the
    /// value stack; `get_state` renders the stack, so the last entry's
    /// `rendered` string is the result -- `"55"` for an `i64` 55, and so on.
    /// reading it through the public `get_state` avoids reaching into the VM's
    /// private value stack from this module.
    fn program_result(vm: &Vm) -> String {
        vm.get_state()
            .stack
            .last()
            .expect("a finished program left a result value")
            .rendered
            .clone()
    }

    #[test]
    fn len_of_an_array_and_of_a_string() {
        let mut vm = bare_vm();
        // an array of three i64 elements.
        let a = int(&mut vm, 10);
        let b = int(&mut vm, 20);
        let c = int(&mut vm, 30);
        let array = alloc_array(&mut vm, vec![a, b, c]).expect("alloc array");
        let array_len = len(&mut vm, &[array]).expect("len of an array");
        assert_eq!(result_i64(&vm, array_len), 3, "an array of 3 has len 3");
        // a five-character string.
        let s = string(&mut vm, "hello");
        let str_len = len(&mut vm, &[s]).expect("len of a string");
        assert_eq!(result_i64(&vm, str_len), 5, "\"hello\" has len 5");
    }

    #[test]
    fn push_appends_and_pop_returns_some_then_none() {
        let mut vm = bare_vm();
        let array = alloc_array(&mut vm, Vec::new()).expect("alloc array");
        // push two values.
        let one = int(&mut vm, 1);
        let two = int(&mut vm, 2);
        push(&mut vm, &[array, one]).expect("push 1");
        push(&mut vm, &[array, two]).expect("push 2");
        assert_eq!(result_array(&vm, array).len(), 2, "two pushes -> length 2");
        // pop returns Some(last).
        let popped = pop(&mut vm, &[array]).expect("pop");
        let popped_slot = popped.as_pointer().expect("Some is a heap pointer");
        match vm.heap.get(popped_slot) {
            Some(HeapObject::EnumVariant {
                variant, payload, ..
            }) => {
                assert_eq!(variant, "Some", "a non-empty pop returns Some");
                assert_eq!(result_i64(&vm, payload[0]), 2, "the last element popped");
            }
            _ => panic!("pop must return an Option enum variant"),
        }
        // pop the remaining element, then pop an empty array -> None.
        pop(&mut vm, &[array]).expect("pop the last element");
        let none = pop(&mut vm, &[array]).expect("pop an empty array");
        let none_slot = none.as_pointer().expect("None is a heap pointer");
        match vm.heap.get(none_slot) {
            Some(HeapObject::EnumVariant { variant, .. }) => {
                assert_eq!(variant, "None", "an empty pop returns None");
            }
            _ => panic!("pop of an empty array must return None"),
        }
    }

    #[test]
    fn sqrt_of_four_is_two() {
        let mut vm = bare_vm();
        let result = sqrt(&mut vm, &[Value::from_f64(4.0)]).expect("sqrt");
        assert_eq!(result.as_f64(), Some(2.0), "sqrt(4.0) == 2.0");
    }

    #[test]
    fn abs_of_a_negative_int_and_a_negative_float() {
        let mut vm = bare_vm();
        // abs of an i64 returns a heap i64.
        let neg_three = int(&mut vm, -3);
        let abs_int = abs(&mut vm, &[neg_three]).expect("abs of an int");
        assert_eq!(result_i64(&vm, abs_int), 3, "abs(-3) == 3");
        // abs of an f64 returns an f64.
        let abs_float = abs(&mut vm, &[Value::from_f64(-1.5)]).expect("abs of a float");
        assert_eq!(abs_float.as_f64(), Some(1.5), "abs(-1.5) == 1.5");
    }

    #[test]
    fn assert_true_is_a_no_op_and_assert_false_is_a_runtime_error() {
        let mut vm = bare_vm();
        // assert(true) returns void without error.
        let ok = assert(&mut vm, &[Value::bool(true)]).expect("assert(true)");
        assert!(ok.as_void(), "assert(true) returns void");
        // assert(false) is a Runtime "assertion failed".
        match assert(&mut vm, &[Value::bool(false)]) {
            Err(QalaError::Runtime { message, .. }) => {
                assert!(message.contains("assertion failed"), "got: {message}");
            }
            Err(other) => panic!("expected an assertion-failed Runtime error, got {other:?}"),
            Ok(_) => panic!("assert(false) must error"),
        }
    }

    #[test]
    fn type_of_returns_the_runtime_type_name_for_each_kind() {
        let mut vm = bare_vm();
        // a representative value of each primitive kind.
        let i64_value = int(&mut vm, 7);
        let str_value = string(&mut vm, "hi");
        // an array of i64 -> "[i64]".
        let e0 = int(&mut vm, 1);
        let e1 = int(&mut vm, 2);
        let i64_array = alloc_array(&mut vm, vec![e0, e1]).expect("alloc array");
        // a struct and an enum variant, built directly on the heap.
        let shape_struct = vm
            .heap
            .alloc(HeapObject::Struct {
                type_name: "Shape".to_string(),
                fields: Vec::new(),
            })
            .expect("alloc struct");
        let shape_variant = vm
            .heap
            .alloc(HeapObject::EnumVariant {
                type_name: "Shape".to_string(),
                variant: "Circle".to_string(),
                payload: Vec::new(),
            })
            .expect("alloc variant");
        let cases: Vec<(Value, &str)> = vec![
            (i64_value, "i64"),
            (Value::from_f64(1.5), "f64"),
            (Value::bool(true), "bool"),
            (str_value, "str"),
            (Value::byte(65), "byte"),
            (Value::void(), "void"),
            (i64_array, "[i64]"),
            (Value::pointer(shape_struct), "Shape"),
            (Value::pointer(shape_variant), "Shape::Circle"),
        ];
        for (value, expected) in cases {
            let result = type_of(&mut vm, &[value]).expect("type_of");
            assert_eq!(result_str(&vm, result), expected, "type_of mismatch");
        }
    }

    #[test]
    fn print_and_println_append_to_the_console() {
        let mut vm = bare_vm();
        // println pushes one console line with a trailing newline.
        let line = string(&mut vm, "first line");
        println(&mut vm, &[line]).expect("println");
        assert_eq!(vm.console, vec!["first line\n".to_string()]);
        // print pushes its own console entry without a trailing newline.
        let a = string(&mut vm, "a");
        let b = string(&mut vm, "b");
        print(&mut vm, &[a]).expect("print a");
        print(&mut vm, &[b]).expect("print b");
        assert_eq!(
            vm.console,
            vec![
                "first line\n".to_string(),
                "a".to_string(),
                "b".to_string(),
            ],
            "println entries end with newline; print entries do not"
        );
    }

    #[test]
    fn println_adds_newline_print_does_not() {
        // joining all console entries produces correct newline separation:
        // two println calls each add a '\n'; a print call does not.
        let mut vm = bare_vm();
        let a = string(&mut vm, "a");
        let b = string(&mut vm, "b");
        let c = string(&mut vm, "c");
        println(&mut vm, &[a]).expect("println a");
        println(&mut vm, &[b]).expect("println b");
        print(&mut vm, &[c]).expect("print c");
        let joined: String = vm.console.concat();
        assert_eq!(joined, "a\nb\nc", "two println then print joined is a\\nb\\nc");
    }

    #[test]
    fn open_returns_a_file_handle_and_close_marks_it_closed() {
        let mut vm = bare_vm();
        let path = string(&mut vm, "data.txt");
        let handle = open(&mut vm, &[path]).expect("open");
        let slot = handle.as_pointer().expect("open returns a heap pointer");
        // the freshly opened handle is not closed and carries the path.
        match vm.heap.get(slot) {
            Some(HeapObject::FileHandle { path, closed, .. }) => {
                assert_eq!(path, "data.txt", "the handle carries the path");
                assert!(!closed, "a freshly opened handle is open");
            }
            _ => panic!("open must return a FileHandle"),
        }
        // close flips the flag.
        let void = close(&mut vm, &[handle]).expect("close");
        assert!(void.as_void(), "close returns void");
        match vm.heap.get(slot) {
            Some(HeapObject::FileHandle { closed, .. }) => {
                assert!(closed, "close marks the handle closed");
            }
            _ => panic!("the handle is still a FileHandle after close"),
        }
    }

    #[test]
    fn read_all_returns_ok_with_the_handle_content() {
        let mut vm = bare_vm();
        let path = string(&mut vm, "data.txt");
        let handle = open(&mut vm, &[path]).expect("open");
        // read_all of the mock handle returns Ok(content) -- the v1 mock
        // content is the empty string.
        let result = read_all(&mut vm, &[handle]).expect("read_all");
        let slot = result.as_pointer().expect("Ok is a heap pointer");
        match vm.heap.get(slot) {
            Some(HeapObject::EnumVariant {
                type_name,
                variant,
                payload,
            }) => {
                assert_eq!(type_name, "Result", "read_all returns a Result");
                assert_eq!(variant, "Ok", "the mock read succeeds");
                assert_eq!(result_str(&vm, payload[0]), "", "the v1 mock content is empty");
            }
            _ => panic!("read_all must return a Result enum variant"),
        }
    }

    #[test]
    fn map_applies_a_user_callback_to_each_element() {
        // a program whose `double` user function is the map callback.
        let src = "fn double(x: i64) -> i64 is pure { return x * 2 }\n\
                   fn main() is io {}\n";
        let program = compile(src);
        let double = fn_id(&program, "double");
        let mut vm = Vm::new(program, src.to_string());
        let e0 = int(&mut vm, 1);
        let e1 = int(&mut vm, 2);
        let e2 = int(&mut vm, 3);
        let array = alloc_array(&mut vm, vec![e0, e1, e2]).expect("alloc array");
        let mapped = map(&mut vm, &[array, Value::function(double)]).expect("map");
        let items = result_array(&vm, mapped);
        let values: Vec<i64> = items.iter().map(|v| result_i64(&vm, *v)).collect();
        assert_eq!(values, vec![2, 4, 6], "map(double) doubles every element");
    }

    #[test]
    fn filter_keeps_the_elements_the_callback_accepts() {
        // `is_even` is the filter predicate.
        let src = "fn is_even(x: i64) -> bool is pure { return x % 2 == 0 }\n\
                   fn main() is io {}\n";
        let program = compile(src);
        let is_even = fn_id(&program, "is_even");
        let mut vm = Vm::new(program, src.to_string());
        let mut elements = Vec::new();
        for n in 1..=6 {
            elements.push(int(&mut vm, n));
        }
        let array = alloc_array(&mut vm, elements).expect("alloc array");
        let filtered = filter(&mut vm, &[array, Value::function(is_even)]).expect("filter");
        let items = result_array(&vm, filtered);
        let values: Vec<i64> = items.iter().map(|v| result_i64(&vm, *v)).collect();
        assert_eq!(values, vec![2, 4, 6], "filter(is_even) keeps the evens");
    }

    #[test]
    fn reduce_folds_an_array_with_the_callback() {
        // `add` is the fold callback; reduce sums 1..=4 from an initial 0.
        let src = "fn add(a: i64, b: i64) -> i64 is pure { return a + b }\n\
                   fn main() is io {}\n";
        let program = compile(src);
        let add = fn_id(&program, "add");
        let mut vm = Vm::new(program, src.to_string());
        let mut elements = Vec::new();
        for n in 1..=4 {
            elements.push(int(&mut vm, n));
        }
        let array = alloc_array(&mut vm, elements).expect("alloc array");
        let initial = int(&mut vm, 0);
        let total = reduce(&mut vm, &[array, initial, Value::function(add)]).expect("reduce");
        assert_eq!(result_i64(&vm, total), 10, "reduce(+) of 1..=4 is 10");
    }

    #[test]
    fn dispatch_routes_each_fn_id_and_rejects_an_unknown_one() {
        let mut vm = bare_vm();
        // fn-id 40002 is `sqrt`: dispatch must route there.
        let via_dispatch = dispatch(&mut vm, 40002, &[Value::from_f64(9.0)]).expect("sqrt");
        assert_eq!(via_dispatch.as_f64(), Some(3.0), "dispatch(40002) is sqrt");
        // an fn-id above the table is malformed bytecode -> a clean error.
        match dispatch(&mut vm, 49999, &[]) {
            Err(QalaError::Runtime { message, .. }) => {
                assert!(message.contains("unknown stdlib function"), "got: {message}");
            }
            Err(other) => panic!("expected an unknown-stdlib Runtime error, got {other:?}"),
            Ok(_) => panic!("an unknown fn-id must error"),
        }
        // an fn-id below the stdlib base must not reach dispatch cleanly.
        match dispatch(&mut vm, 5, &[]) {
            Err(QalaError::Runtime { message, .. }) => {
                assert!(message.contains("not a stdlib id"), "got: {message}");
            }
            Err(other) => panic!("expected an invariant Runtime error, got {other:?}"),
            Ok(_) => panic!("a user fn-id must not dispatch as stdlib"),
        }
    }

    #[test]
    fn a_wrong_argument_count_is_a_runtime_error_not_a_panic() {
        let mut vm = bare_vm();
        // sqrt expects one argument; zero is malformed bytecode.
        match sqrt(&mut vm, &[]) {
            Err(QalaError::Runtime { message, .. }) => {
                assert!(message.contains("expects 1 argument"), "got: {message}");
            }
            Err(other) => panic!("expected an arity Runtime error, got {other:?}"),
            Ok(_) => panic!("sqrt with no argument must error"),
        }
        // a wrong-type argument is likewise a clean error, never a panic.
        match len(&mut vm, &[Value::bool(true)]) {
            Err(QalaError::Runtime { message, .. }) => {
                assert!(message.contains("expected an array or string"), "got: {message}");
            }
            Err(other) => panic!("expected a wrong-type Runtime error, got {other:?}"),
            Ok(_) => panic!("len of a bool must error"),
        }
    }

    // ---- re-entrancy + the cross-cutting smoke tests ----

    #[test]
    fn map_reentrant_a_nested_map_inside_the_callback_works() {
        // RESEARCH Pitfall 7: a `map` whose callback itself calls `map` must
        // produce the correct nested result. `double_row` is the outer
        // callback; it calls `map(row, double)` -- a nested re-entry. the loop
        // state of each `map` stays in Rust locals, so the inner map does not
        // corrupt the outer. the grid [[1,2],[3,4]] doubled is [[2,4],[6,8]];
        // the program returns doubled[1][1], which must be 8.
        let src = "fn double(x: i64) -> i64 is pure { return x * 2 }\n\
                   fn double_row(row: [i64]) -> [i64] is pure {\n    \
                   return map(row, double)\n}\n\
                   fn main() -> i64 is pure {\n    \
                   let grid = [[1, 2], [3, 4]]\n    \
                   let doubled = map(grid, double_row)\n    \
                   return doubled[1][1]\n}\n";
        let program = compile(src);
        let mut vm = Vm::new(program, src.to_string());
        vm.run().expect("a nested-map program runs clean");
        assert_eq!(
            program_result(&vm),
            "8",
            "the nested map doubled [3,4] to [6,8]; [1][1] is 8"
        );
    }

    #[test]
    fn fibonacci_smoke_computes_the_correct_numeric_result() {
        // the success-criterion smoke test: a recursive fib compiled and run
        // end to end. fib(10) is 55.
        let src = "fn fib(n: i64) -> i64 is pure {\n    \
                   if n <= 1 { return n }\n    \
                   return fib(n - 1) + fib(n - 2)\n}\n\
                   fn main() -> i64 is pure {\n    return fib(10)\n}\n";
        let program = compile(src);
        let mut vm = Vm::new(program, src.to_string());
        vm.run().expect("the fibonacci program runs clean");
        assert_eq!(program_result(&vm), "55", "fib(10) must be 55");
    }

    /// read a bundled example from `playground/public/examples/`.
    fn example_source(name: &str) -> String {
        let path = format!(
            "{}/../playground/public/examples/{name}.qala",
            env!("CARGO_MANIFEST_DIR"),
        );
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"))
    }

    /// compile and run a bundled example, returning the finished VM so the
    /// caller can inspect its console.
    fn run_example(name: &str) -> Vm {
        let src = example_source(name);
        let program = compile(&src);
        let mut vm = Vm::new(program, src);
        vm.run()
            .unwrap_or_else(|e| panic!("{name}.qala did not run to completion: {e:?}"));
        vm
    }

    #[test]
    fn six_examples_run_to_completion() {
        // every bundled example compiles, runs to completion without a runtime
        // error, and -- for the examples that print -- produces the expected
        // console output. the examples are the acceptance corpus for Phase 5.

        // hello: string interpolation -> "hello, world!".
        let hello = run_example("hello");
        assert!(
            hello.console.iter().any(|l| l.trim_end_matches('\n') == "hello, world!"),
            "hello.qala console: {:?}",
            hello.console,
        );

        // fibonacci: a for loop printing fib(0..14). the console is the 15
        // lines fib(0)=0 .. fib(14)=377.
        let fibonacci = run_example("fibonacci");
        assert!(
            fibonacci.console.iter().any(|l| l.trim_end_matches('\n') == "fib(0) = 0"),
            "fibonacci.qala console: {:?}",
            fibonacci.console,
        );
        assert!(
            fibonacci.console.iter().any(|l| l.trim_end_matches('\n') == "fib(10) = 55"),
            "fibonacci.qala must print fib(10) = 55, console: {:?}",
            fibonacci.console,
        );
        assert!(
            fibonacci.console.iter().any(|l| l.trim_end_matches('\n') == "fib(14) = 377"),
            "fibonacci.qala must print fib(14) = 377, console: {:?}",
            fibonacci.console,
        );

        // effects: a pure function called from an io function -> "7 squared: 49".
        let effects = run_example("effects");
        assert!(
            effects.console.iter().any(|l| l.trim_end_matches('\n') == "7 squared: 49"),
            "effects.qala console: {:?}",
            effects.console,
        );

        // pattern-matching: enums with data + guards. it prints the three
        // shape areas then the three classify results.
        let pattern_matching = run_example("pattern-matching");
        assert!(
            !pattern_matching.console.is_empty(),
            "pattern-matching.qala must print something",
        );
        assert!(
            pattern_matching.console.iter().any(|l| l.trim_end_matches('\n') == "positive"),
            "pattern-matching.qala classify(42) must print positive, console: {:?}",
            pattern_matching.console,
        );
        assert!(
            pattern_matching.console.iter().any(|l| l.trim_end_matches('\n') == "negative"),
            "pattern-matching.qala classify(-7) must print negative, console: {:?}",
            pattern_matching.console,
        );
        assert!(
            pattern_matching.console.iter().any(|l| l.trim_end_matches('\n') == "zero"),
            "pattern-matching.qala classify(0) must print zero, console: {:?}",
            pattern_matching.console,
        );

        // pipeline: |> with map / filter. 5 |> double |> add_one |> double is
        // ((5*2)+1)*2 = 22; the even numbers 1..10 filtered then doubled are
        // 4, 8, 12, 16, 20.
        let pipeline = run_example("pipeline");
        assert!(
            pipeline
                .console
                .iter()
                .any(|l| l.contains("= 22")),
            "pipeline.qala must print the pipeline result 22, console: {:?}",
            pipeline.console,
        );
        assert!(
            pipeline.console.iter().any(|l| l.trim_end_matches('\n') == "20"),
            "pipeline.qala filter+map must print the doubled even 20, console: {:?}",
            pipeline.console,
        );

        // defer-demo: a resource with defer close. process_file's mock read
        // returns empty content, so len(content) == 0 takes the Err path; the
        // `or "no data"` fallback supplies the value and main prints it.
        let defer_demo = run_example("defer-demo");
        assert!(
            defer_demo.console.iter().any(|l| l.starts_with("got: ")),
            "defer-demo.qala must print a got: line, console: {:?}",
            defer_demo.console,
        );
        // the defer closed the handle on every exit path -- no leak.
        assert!(
            defer_demo.leak_log.is_empty(),
            "defer-demo.qala closes its handle via defer -- no leak, got: {:?}",
            defer_demo.leak_log,
        );
    }
}
