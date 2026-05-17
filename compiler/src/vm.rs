//! the stack-based bytecode interpreter.
//!
//! a [`Vm`] takes a [`Program`] (the chunks, the constant pools, the source
//! map) and executes it: a value stack of NaN-boxed [`Value`]s, a call-frame
//! stack, a heap of [`HeapObject`]s, a console buffer, a leak log, and
//! persistent globals for the REPL. `run()` drives it to completion or to the
//! first runtime fault; `step()` advances one instruction for the playground's
//! step-through; both go through the single `dispatch_one()` decoder.
//!
//! the absolute constraint: the VM NEVER panics on any bytecode. every
//! `code[ip]` byte, every operand read, every value-stack pop, every heap
//! access, every constant-pool index is bounds-checked and surfaces a
//! [`QalaError::Runtime`] on failure. the crate compiles to WASM, where a
//! panic aborts the browser tab. there is no `unwrap` outside `#[cfg(test)]`.
//!
//! memory model: an `i64` is uniformly heap-boxed -- a [`HeapObject::Int`]
//! reached through a `TAG_PTR` [`Value`] -- so the codec in `value.rs` carries
//! no integer encoding (the uniform-heap-box decision; see 05-RESEARCH.md). a
//! function value, by contrast, is a tagged scalar: [`Value::function`] rides
//! the `u16` fn-id in the NaN payload with NO heap object. arrays, strings,
//! structs, enum variants, and file handles are heap objects.
//!
//! the heap reclaims slots with reference counting plus a free list.
//! [`Heap::dec`] returns the freed [`HeapObject`] when a refcount hits zero so
//! the caller can inspect it (the stdlib's file-handle leak check needs this).
//! reference cycles are NOT collected in v1 -- a v2 concern; Qala v1's value
//! semantics make cycles hard to create.
//!
//! this module is built up over several commits (the authorized
//! multi-commit-to-one-file exception): the data model and heap here, then the
//! dispatch decoder, then the arithmetic / comparison / logic / jump opcode
//! handlers. CALL / RETURN / the `MAKE_*` family / INDEX / FIELD / LEN / TO_STR
//! / CONCAT_N / MATCH_VARIANT are stubbed as a clean `Runtime` error until a
//! later commit fills them in.

use crate::chunk::Chunk;
use crate::chunk::Program;
use crate::errors::QalaError;
use crate::span::LineIndex;
use crate::span::Span;
use crate::value::Value;

/// the call-frame depth cap. a Qala recursion grows the [`Vm::frames`] vec, not
/// the host Rust stack (the VM is a `while`-loop interpreter), so an unbounded
/// recursion hits this cap and becomes a clean `Runtime` "stack overflow"
/// rather than a host stack overflow. 1024 frames is generous for a teaching
/// language.
///
/// `allow(dead_code)`: the cap is defined here with the value-stack and heap
/// caps as one locked set, but it is enforced in the `CALL` opcode handler,
/// which a later commit adds. keeping the constant now makes the cap set
/// self-documenting and gives the call handler a single source of truth.
#[allow(dead_code)]
const MAX_FRAMES: usize = 1024;

/// the value-stack depth cap. [`Vm::push`] errors past it with "value stack
/// overflow" so a runaway program cannot exhaust WASM memory through the value
/// stack. 65536 slots is far more than any realistic Qala program needs.
const MAX_STACK: usize = 65536;

/// the heap slot cap. [`Heap::alloc`] errors past it so a runaway allocation
/// loop surfaces a clean `Runtime` "heap exhausted" error instead of an
/// out-of-memory abort of the browser tab. one million slots is the
/// WASM memory-exhaustion guard; raising it is a v2 concern.
const MAX_HEAP: usize = 1_000_000;

/// a value living on the VM heap, reached through a `TAG_PTR` [`Value`].
///
/// every variant a `MAKE_*` opcode builds, plus [`HeapObject::Int`] -- because
/// `i64` is uniformly heap-boxed (the codec in `value.rs` has no integer tag).
/// a [`HeapObject::FileHandle`] that is freed while still open is a resource
/// leak the VM logs.
///
/// derives `Clone, PartialEq` but NOT `Debug` -- a variant carries [`Value`],
/// and `Value`'s locked derive list (the NaN-box newtype) omits `Debug`. tests
/// compare `HeapObject`s with `==` rather than `assert_eq!` for that reason.
#[derive(Clone, PartialEq)]
pub enum HeapObject {
    /// a 64-bit signed integer. `i64` has no NaN-box tag, so every integer
    /// runtime value is one of these, reached through a pointer.
    Int(i64),
    /// a dynamic array of values, built by `MAKE_ARRAY` (and by `MAKE_TUPLE` --
    /// a tuple is a fixed-shape array at runtime). `push` / `pop` mutate it.
    Array(Vec<Value>),
    /// an owned string, built by a `CONST` of a `ConstValue::Str` or by
    /// `CONCAT_N` / `TO_STR`.
    Str(String),
    /// a struct instance: the declared type name (from `Program.structs`) plus
    /// the field values in declaration order.
    Struct {
        /// the declared struct name, e.g. `"Point"` -- what `type_of` returns.
        type_name: String,
        /// the field values, in the struct's declaration order.
        fields: Vec<Value>,
    },
    /// an enum-variant instance: the enum name, the variant name, and the
    /// variant's payload values (empty for a payload-less variant).
    EnumVariant {
        /// the enum's declared name, e.g. `"Shape"`.
        type_name: String,
        /// the variant's name, e.g. `"Circle"`.
        variant: String,
        /// the variant's payload values, in declaration order.
        payload: Vec<Value>,
    },
    /// a mock file handle -- `open` builds one, `close` marks it closed. the
    /// VM does no real file I/O (it runs in a WASM sandbox); the handle backs
    /// the effect system and `defer close(f)` demonstrations. a handle freed
    /// while still open is a leak.
    FileHandle {
        /// the path passed to `open`. purely informational in the mock.
        path: String,
        /// the mock file content; `read_all` returns this.
        content: String,
        /// `true` once `close` has run on this handle.
        closed: bool,
    },
}

/// one heap slot: a [`HeapObject`] plus its reference count.
///
/// the count starts at 1 on `alloc`, rises on `inc`, falls on `dec`; a slot
/// whose count reaches zero is freed back onto the heap's free list.
#[derive(Clone)]
struct HeapSlot {
    /// the object this slot holds.
    object: HeapObject,
    /// the live-reference count. zero means the slot is free.
    refcount: u32,
}

/// the VM heap: a slab of [`HeapSlot`]s plus a free list of reusable indices.
///
/// `alloc` reuses a freed slot when one is available, else appends. each slot
/// is reference-counted; a slot whose count hits zero is pushed onto `free`.
/// every accessor takes a slot index and returns an `Option` -- a bad index is
/// never an out-of-bounds panic.
///
/// reference cycles are NOT collected: two objects that point at each other
/// keep each other's refcount above zero forever. this is a documented v1
/// limitation -- a cycle collector is a v2 concern, and Qala v1's value
/// semantics (no mutable cross-object references) make cycles hard to create.
#[derive(Clone, Default)]
pub struct Heap {
    /// the slot slab, indexed by the `u32` a pointer [`Value`] carries.
    objects: Vec<HeapSlot>,
    /// indices of freed slots, reused before the slab grows.
    free: Vec<u32>,
}

impl Heap {
    /// construct an empty heap.
    pub fn new() -> Self {
        Self::default()
    }

    /// allocate `obj` into a slot and return the slot index. a freed slot is
    /// reused if one is available, else the slab grows by one. the new slot's
    /// refcount starts at 1.
    ///
    /// returns `None` when the slab is already at [`MAX_HEAP`] slots and no
    /// free slot can be reused -- the caller maps `None` to a `Runtime` "heap
    /// exhausted" error so a runaway allocation never aborts the host. `None`
    /// rather than `Result<u32, ()>` keeps the signature lint-clean (a unit
    /// error type carries no information `None` does not).
    pub fn alloc(&mut self, obj: HeapObject) -> Option<u32> {
        if let Some(slot) = self.free.pop() {
            let idx = slot as usize;
            // a freed slot is always within bounds: it was a valid index when
            // dec pushed it onto the free list.
            if let Some(s) = self.objects.get_mut(idx) {
                s.object = obj;
                s.refcount = 1;
                return Some(slot);
            }
            // unreachable in practice -- a free-list index is always valid.
            // fall through to the append path rather than panic.
        }
        if self.objects.len() >= MAX_HEAP {
            return None;
        }
        let idx = self.objects.len() as u32;
        self.objects.push(HeapSlot {
            object: obj,
            refcount: 1,
        });
        Some(idx)
    }

    /// borrow the object at `slot`. returns `None` for an out-of-range or
    /// freed slot (a freed slot has refcount 0) -- never an out-of-bounds
    /// index.
    pub fn get(&self, slot: u32) -> Option<&HeapObject> {
        self.objects
            .get(slot as usize)
            .filter(|s| s.refcount > 0)
            .map(|s| &s.object)
    }

    /// mutably borrow the object at `slot`. returns `None` for an out-of-range
    /// or freed slot -- never an out-of-bounds index.
    pub fn get_mut(&mut self, slot: u32) -> Option<&mut HeapObject> {
        self.objects
            .get_mut(slot as usize)
            .filter(|s| s.refcount > 0)
            .map(|s| &mut s.object)
    }

    /// increment the refcount of the object at `slot`. a bad or freed slot is
    /// a silent no-op -- the VM never crashes on a stray inc.
    ///
    /// `allow(dead_code)`: part of the locked `Heap` refcount API. the opcode
    /// handlers that copy a heap value (`DUP` of a pointer, `GET_LOCAL`,
    /// struct/array construction) call `inc`; a later commit wires them in.
    #[allow(dead_code)]
    pub fn inc(&mut self, slot: u32) {
        if let Some(s) = self.objects.get_mut(slot as usize).filter(|s| s.refcount > 0) {
            s.refcount = s.refcount.saturating_add(1);
        }
    }

    /// decrement the refcount of the object at `slot`.
    ///
    /// when the count reaches zero the slot is freed: its index is pushed onto
    /// the free list and the freed [`HeapObject`] is RETURNED so the caller can
    /// inspect it. when the count is still positive after the decrement -- or
    /// the slot is out of range or already free -- the result is `None`.
    ///
    /// the return value is load-bearing: the stdlib's file-handle leak check
    /// needs `dec` to hand back the freed object so the [`Vm`] can detect a
    /// still-open [`HeapObject::FileHandle`] and push to [`Vm::leak_log`]. the
    /// leak-log push is the caller's responsibility -- `Heap` has no access to
    /// `Vm::leak_log`; `dec` only surfaces the freed object. this signature is
    /// the locked contract; later plans must not change it.
    ///
    /// `allow(dead_code)`: part of the locked `Heap` refcount API. the opcode
    /// handlers that drop a heap reference (`POP` of a pointer, scope exit,
    /// `RETURN`) call `dec`; a later commit wires them in.
    #[allow(dead_code)]
    pub fn dec(&mut self, slot: u32) -> Option<HeapObject> {
        let s = self.objects.get_mut(slot as usize)?;
        if s.refcount == 0 {
            return None;
        }
        s.refcount -= 1;
        if s.refcount == 0 {
            // free the slot: hand back the object and recycle the index. the
            // slot keeps a placeholder (a void Int) so the slab stays dense;
            // the next alloc that reuses this index overwrites it.
            let freed = std::mem::replace(&mut s.object, HeapObject::Int(0));
            self.free.push(slot);
            Some(freed)
        } else {
            None
        }
    }
}

/// one call frame: the function being run, the instruction pointer into its
/// chunk, the base slot where this frame's locals begin, and the frame's local
/// slots.
///
/// the frame owns its `locals` vec, which drops when the frame is popped on
/// `RETURN`. 05-RESEARCH.md describes a per-frame bump arena freed wholesale on
/// return; in v1 the heap's free list already reclaims slots, so the frame
/// owning its `locals` (dropped on return) is the whole of the "arena" -- a
/// distinct bump region is not required for correctness, and adding one would
/// be dead weight against the free list. a richer arena stays a v2 idea.
///
/// derives `Clone` but not `Debug` -- `locals` is a `Vec<Value>` and `Value`'s
/// locked derive list omits `Debug`.
#[derive(Clone)]
pub struct CallFrame {
    /// index into [`Program::chunks`] -- which function this frame runs.
    pub chunk_idx: usize,
    /// the instruction pointer: a byte offset into the chunk's `code`.
    pub ip: usize,
    /// the value-stack index where this frame's `locals[0]` conceptually sits.
    /// the call machinery (a later commit) uses it to unwind the stack on
    /// `RETURN`.
    pub base: usize,
    /// this frame's local slots, indexed by `GET_LOCAL` / `SET_LOCAL`
    /// operands. dropped when the frame is popped.
    pub locals: Vec<Value>,
}

/// the bytecode virtual machine.
///
/// holds the program being run, the value stack, the call-frame stack, the
/// heap, the console buffer, the leak log, the persistent globals, and the
/// original source text (needed to build a line-covering [`Span`] for a
/// runtime error and for the REPL pipeline).
pub struct Vm {
    /// the program being executed.
    program: Program,
    /// the value stack -- every slot a NaN-boxed [`Value`]. capped at
    /// [`MAX_STACK`].
    stack: Vec<Value>,
    /// the call-frame stack. capped at [`MAX_FRAMES`]; the topmost frame is
    /// the one currently executing.
    frames: Vec<CallFrame>,
    /// the heap.
    heap: Heap,
    /// captured `print` / `println` output. the VM runs in WASM where stdout
    /// is invisible; the playground renders this buffer instead.
    ///
    /// `allow(dead_code)`: written by the stdlib `print` / `println` functions,
    /// which a later commit adds. the buffer is part of the `Vm`'s locked
    /// shape, so it is declared now.
    #[allow(dead_code)]
    console: Vec<String>,
    /// resource-leak log: a [`HeapObject::FileHandle`] freed while still open
    /// is recorded here, surfaced for the playground.
    ///
    /// `allow(dead_code)`: written by the heap-decrement leak check, which a
    /// later commit wires into the `RETURN` and scope-exit paths. declared now
    /// as part of the `Vm`'s locked shape.
    #[allow(dead_code)]
    leak_log: Vec<String>,
    /// persistent global slots, indexed by `GET_GLOBAL` / `SET_GLOBAL`. kept
    /// across REPL calls.
    globals: Vec<Value>,
    /// the original source text. used to map a `source_lines` line back to a
    /// byte-range [`Span`] for runtime errors, and by the REPL pipeline.
    src: String,
}

impl Vm {
    /// construct a VM ready to run `program` from its entry point.
    ///
    /// pushes the initial [`CallFrame`] for `program.main_index`. `src` is the
    /// program's source text -- the VM keeps it to build a line-covering span
    /// for any runtime error and to drive the REPL pipeline.
    pub fn new(program: Program, src: String) -> Vm {
        let main_index = program.main_index;
        let frames = vec![CallFrame {
            chunk_idx: main_index,
            ip: 0,
            base: 0,
            locals: Vec::new(),
        }];
        Vm {
            program,
            stack: Vec::new(),
            frames,
            heap: Heap::new(),
            console: Vec::new(),
            leak_log: Vec::new(),
            globals: Vec::new(),
            src,
        }
    }

    /// the topmost call frame.
    ///
    /// returns `Err` when the frame stack is empty -- a VM-internal invariant
    /// violation (a well-formed run always has at least the `main` frame until
    /// it returns), surfaced as a `Runtime` error rather than a panic.
    fn frame(&self) -> Result<&CallFrame, QalaError> {
        self.frames.last().ok_or_else(|| QalaError::Runtime {
            span: Span::new(0, 0),
            message: "no active call frame".to_string(),
        })
    }

    /// the topmost call frame, mutably. see [`Vm::frame`] for the empty-stack
    /// case.
    fn frame_mut(&mut self) -> Result<&mut CallFrame, QalaError> {
        self.frames.last_mut().ok_or_else(|| QalaError::Runtime {
            span: Span::new(0, 0),
            message: "no active call frame".to_string(),
        })
    }

    /// the chunk the topmost frame is running.
    ///
    /// returns `Err` when the frame stack is empty or the frame's `chunk_idx`
    /// is out of range -- both VM-internal invariant violations surfaced as a
    /// `Runtime` error, never an out-of-bounds index.
    fn chunk(&self) -> Result<&Chunk, QalaError> {
        let idx = self.frame()?.chunk_idx;
        self.program
            .chunks
            .get(idx)
            .ok_or_else(|| QalaError::Runtime {
                span: Span::new(0, 0),
                message: format!("call frame references missing chunk {idx}"),
            })
    }

    /// push a value onto the value stack.
    ///
    /// returns `Err` with "value stack overflow" when the stack is already at
    /// [`MAX_STACK`] -- a runaway program cannot exhaust host memory through
    /// the value stack.
    fn push(&mut self, v: Value) -> Result<(), QalaError> {
        if self.stack.len() >= MAX_STACK {
            return Err(self.runtime_err("value stack overflow"));
        }
        self.stack.push(v);
        Ok(())
    }

    /// pop the top value off the value stack.
    ///
    /// returns `Err` with "stack underflow" when the stack is empty -- a
    /// malformed instruction stream that pops more than it pushes is a
    /// `Runtime` error, never a panic.
    fn pop(&mut self) -> Result<Value, QalaError> {
        self.stack
            .pop()
            .ok_or_else(|| self.runtime_err("stack underflow"))
    }

    /// build a [`QalaError::Runtime`] whose span covers the source line of the
    /// instruction currently at the topmost frame's `ip`.
    ///
    /// the line comes from `chunk.source_lines[ip]`; the byte range of that
    /// line is found by scanning [`LineIndex`] over `self.src`. when the line
    /// cannot be resolved (an empty frame stack, a missing chunk, an `ip` past
    /// the source map, a line number past the source) the span is a harmless
    /// zero-width span -- this never panics.
    fn runtime_err(&self, message: &str) -> QalaError {
        let span = self.error_span();
        QalaError::Runtime {
            span,
            message: message.to_string(),
        }
    }

    /// the source span covering the line of the current instruction.
    ///
    /// a free function off the error path so [`Vm::runtime_err`] stays a thin
    /// constructor. on any lookup miss it returns a zero-width span at offset
    /// 0 -- a runtime error must never fail to be built.
    fn error_span(&self) -> Span {
        // the 1-based source line of the current instruction, or 0 on a miss.
        let line = self
            .frame()
            .ok()
            .and_then(|f| {
                let ip = f.ip;
                self.chunk().ok().and_then(|c| c.source_lines.get(ip).copied())
            })
            .unwrap_or(0);
        if line == 0 {
            return Span::new(0, 0);
        }
        // find the byte range of that 1-based line in the source.
        let index = LineIndex::new(&self.src);
        line_span(&index, &self.src, line)
    }
}

/// the byte-range [`Span`] of a 1-based `line` in `src`.
///
/// scans for the line's start and end byte offsets. a line number past the end
/// of the source yields a zero-width span at the source end -- the caller
/// (a runtime-error builder) must never panic.
fn line_span(index: &LineIndex, src: &str, line: u32) -> Span {
    // the start byte of the line: the offset whose location is (line, 1).
    // bytes() scan keeps this allocation-free and UTF-8-correct (line breaks
    // are ASCII).
    let mut starts: Vec<usize> = vec![0];
    for (i, b) in src.bytes().enumerate() {
        if b == b'\n' {
            starts.push(i + 1);
        }
    }
    let line_idx = (line as usize).saturating_sub(1);
    let Some(&start) = starts.get(line_idx) else {
        // line past the end of the source: a zero-width span at the end.
        return Span::new(src.len(), 0);
    };
    // the end byte is the start of the next line, or the source end for the
    // last line. trim a trailing '\n' / '\r' so the span covers the line text
    // itself, not its terminator.
    let mut end = starts.get(line_idx + 1).copied().unwrap_or(src.len());
    let bytes = src.as_bytes();
    while end > start && (bytes[end - 1] == b'\n' || bytes[end - 1] == b'\r') {
        end -= 1;
    }
    // `index` is accepted for API symmetry with the caller's LineIndex; the
    // scan above is the authoritative line-start computation.
    let _ = index;
    Span::new(start, end - start)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// a one-chunk program whose `main` chunk is `chunk`, named `"main"`.
    fn program_with(chunk: Chunk) -> Program {
        let mut p = Program::new();
        p.chunks.push(chunk);
        p.fn_names.push("main".to_string());
        p.main_index = 0;
        p
    }

    // ---- heap lifecycle ----
    //
    // HeapObject has no Debug derive (it carries Value, whose locked derive
    // list omits Debug), so these tests compare with `==` and `matches!`
    // rather than assert_eq! / assert_ne! on HeapObject-valued expressions.

    #[test]
    fn heap_alloc_then_get_round_trips_the_object() {
        let mut h = Heap::new();
        let slot = h.alloc(HeapObject::Int(42)).expect("alloc");
        assert!(h.get(slot) == Some(&HeapObject::Int(42)));
    }

    #[test]
    fn heap_alloc_hands_out_distinct_slots() {
        let mut h = Heap::new();
        let a = h.alloc(HeapObject::Int(1)).expect("alloc a");
        let b = h.alloc(HeapObject::Int(2)).expect("alloc b");
        assert_ne!(a, b, "two live allocations must get distinct slots");
        assert!(h.get(a) == Some(&HeapObject::Int(1)));
        assert!(h.get(b) == Some(&HeapObject::Int(2)));
    }

    #[test]
    fn heap_get_of_a_bad_slot_is_none_not_a_panic() {
        let h = Heap::new();
        assert!(h.get(0).is_none(), "empty heap, slot 0 is out of range");
        assert!(h.get(9999).is_none());
    }

    #[test]
    fn heap_get_mut_mutates_the_object_in_place() {
        let mut h = Heap::new();
        let slot = h.alloc(HeapObject::Str("a".to_string())).expect("alloc");
        if let Some(HeapObject::Str(s)) = h.get_mut(slot) {
            s.push('b');
        }
        assert!(h.get(slot) == Some(&HeapObject::Str("ab".to_string())));
    }

    #[test]
    fn heap_inc_then_dec_keeps_the_slot_alive_until_count_reaches_zero() {
        let mut h = Heap::new();
        let slot = h.alloc(HeapObject::Int(7)).expect("alloc"); // refcount 1
        h.inc(slot); // refcount 2
        // a dec from 2 -> 1 leaves the slot alive and returns None.
        assert!(h.dec(slot).is_none(), "dec to a positive count returns None");
        assert!(h.get(slot) == Some(&HeapObject::Int(7)), "slot still alive");
    }

    #[test]
    fn heap_dec_to_zero_frees_the_slot_and_returns_the_freed_object() {
        let mut h = Heap::new();
        let slot = h.alloc(HeapObject::Int(99)).expect("alloc"); // refcount 1
        // the dec that drives the count to zero hands back the freed object.
        assert!(
            h.dec(slot) == Some(HeapObject::Int(99)),
            "dec to zero returns the freed object"
        );
        // the freed slot no longer reads as a live object.
        assert!(h.get(slot).is_none(), "a freed slot reads as None");
    }

    #[test]
    fn heap_dec_returns_the_freed_file_handle_so_the_caller_can_leak_check() {
        // the locked contract: dec hands back the freed object so a caller can
        // see a still-open FileHandle and log a leak.
        let mut h = Heap::new();
        let handle = HeapObject::FileHandle {
            path: "data.txt".to_string(),
            content: String::new(),
            closed: false,
        };
        let slot = h.alloc(handle.clone()).expect("alloc");
        let freed = h.dec(slot).expect("dec to zero returns the object");
        // HeapObject has no Debug; the non-FileHandle arm reports in words.
        match freed {
            HeapObject::FileHandle { closed, path, .. } => {
                assert!(!closed, "the freed handle is still open -- a leak");
                assert_eq!(path, "data.txt");
            }
            _ => panic!("expected a FileHandle from dec, got another variant"),
        }
    }

    #[test]
    fn heap_dec_of_a_bad_or_freed_slot_is_none_not_a_panic() {
        let mut h = Heap::new();
        // a slot that was never allocated.
        assert!(h.dec(0).is_none());
        // a slot freed once cannot be freed again.
        let slot = h.alloc(HeapObject::Int(1)).expect("alloc");
        assert!(h.dec(slot).is_some(), "first dec frees it");
        assert!(h.dec(slot).is_none(), "a second dec of a freed slot is None");
    }

    #[test]
    fn heap_alloc_reuses_a_freed_slot_before_growing_the_slab() {
        let mut h = Heap::new();
        let first = h.alloc(HeapObject::Int(1)).expect("alloc first");
        // free it, then allocate again -- the new object must reuse the slot.
        h.dec(first);
        let reused = h.alloc(HeapObject::Int(2)).expect("alloc reused");
        assert_eq!(reused, first, "a freed slot index is reused by the next alloc");
        assert!(h.get(reused) == Some(&HeapObject::Int(2)));
    }

    #[test]
    fn heap_inc_of_a_bad_slot_is_a_silent_no_op() {
        let mut h = Heap::new();
        // inc on a never-allocated slot must not panic and must not create one.
        h.inc(0);
        h.inc(12345);
        assert!(h.get(0).is_none());
    }

    #[test]
    fn heap_caps_are_the_documented_values() {
        // the three caps the threat model depends on -- lock them so a future
        // edit that loosens a cap is visible here.
        assert_eq!(MAX_FRAMES, 1024);
        assert_eq!(MAX_STACK, 65536);
        assert_eq!(MAX_HEAP, 1_000_000);
    }

    // ---- Vm construction + helpers ----

    #[test]
    fn vm_new_pushes_the_initial_main_frame() {
        let mut p = Program::new();
        p.chunks.push(Chunk::new());
        p.chunks.push(Chunk::new());
        p.fn_names.push("first".to_string());
        p.fn_names.push("main".to_string());
        p.main_index = 1;
        let vm = Vm::new(p, String::new());
        // exactly one frame, pointed at main_index, ip 0.
        assert_eq!(vm.frames.len(), 1);
        let f = vm.frame().expect("the main frame exists");
        assert_eq!(f.chunk_idx, 1);
        assert_eq!(f.ip, 0);
        assert_eq!(f.base, 0);
    }

    #[test]
    fn vm_push_then_pop_round_trips_a_value() {
        let vm_program = program_with(Chunk::new());
        let mut vm = Vm::new(vm_program, String::new());
        vm.push(Value::bool(true)).expect("push");
        let v = vm.pop().expect("pop");
        assert_eq!(v.as_bool(), Some(true));
    }

    #[test]
    fn vm_pop_on_an_empty_stack_is_a_runtime_underflow_not_a_panic() {
        let mut vm = Vm::new(program_with(Chunk::new()), String::new());
        // Value has no Debug, so the failure arm cannot print the Ok payload;
        // it reports the error variant in words instead.
        match vm.pop() {
            Err(QalaError::Runtime { message, .. }) => {
                assert!(message.contains("underflow"), "got: {message}");
            }
            Err(other) => panic!("expected a Runtime underflow, got {other:?}"),
            Ok(_) => panic!("expected a Runtime underflow, got Ok(value)"),
        }
    }

    #[test]
    fn vm_push_past_max_stack_is_a_runtime_overflow_not_a_panic() {
        let mut vm = Vm::new(program_with(Chunk::new()), String::new());
        // fill the stack directly to the cap, then one more push must error.
        vm.stack.resize(MAX_STACK, Value::void());
        match vm.push(Value::void()) {
            Err(QalaError::Runtime { message, .. }) => {
                assert!(message.contains("overflow"), "got: {message}");
            }
            other => panic!("expected a Runtime overflow, got {other:?}"),
        }
    }

    #[test]
    fn vm_runtime_err_carries_the_source_line_of_the_current_instruction() {
        // a chunk whose byte 0 maps to source line 3; the error span must
        // cover line 3's byte range.
        let mut chunk = Chunk::new();
        chunk.code.push(0);
        chunk.source_lines.push(3);
        let src = "line one\nline two\nline three\nline four".to_string();
        let vm = Vm::new(program_with(chunk), src.clone());
        let err = vm.runtime_err("boom");
        let span = err.span();
        // line 3 is "line three" -- the span slices back to exactly that text.
        assert_eq!(span.slice(&src), "line three");
    }

    #[test]
    fn vm_runtime_err_on_a_missing_source_line_is_a_zero_width_span_not_a_panic() {
        // an empty chunk: ip 0 has no source_lines entry. the error must still
        // build, with a harmless zero-width span.
        let vm = Vm::new(program_with(Chunk::new()), "anything".to_string());
        let err = vm.runtime_err("boom");
        let span = err.span();
        assert_eq!(span.len, 0, "an unresolved line yields a zero-width span");
    }

    #[test]
    fn line_span_of_a_line_past_the_source_is_a_zero_width_span_not_a_panic() {
        let src = "only one line";
        let index = LineIndex::new(src);
        // line 99 does not exist; the span must be zero-width, no panic.
        let span = line_span(&index, src, 99);
        assert_eq!(span.len, 0);
    }
}
