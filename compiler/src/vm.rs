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
use crate::opcode::Opcode;
use crate::span::LineIndex;
use crate::span::Span;
use crate::value::ConstValue;
use crate::value::Value;

/// the call-frame depth cap. a Qala recursion grows the [`Vm::frames`] vec, not
/// the host Rust stack (the VM is a `while`-loop interpreter), so an unbounded
/// recursion hits this cap and becomes a clean `Runtime` "stack overflow"
/// rather than a host stack overflow. 1024 frames is generous for a teaching
/// language.
///
/// enforced in [`Vm::op_call`] and [`Vm::call_function_value`] -- both check
/// `frames.len() >= MAX_FRAMES` before pushing a frame.
const MAX_FRAMES: usize = 1024;

/// the first function id reserved for the stdlib. codegen assigns user
/// functions dense ids `0..N` (indexes into [`Program::chunks`]) and stdlib
/// functions ids `STDLIB_FN_BASE + i` -- the two id spaces never collide.
///
/// the [`Vm::op_call`] handler branches on this threshold: a `fn_id` at or
/// above it is a native stdlib call routed to [`Vm::call_stdlib`]; below it is
/// a user function with a chunk in `Program.chunks`. the value must match
/// codegen's `STDLIB_FN_BASE` (see `codegen.rs`); changing one without the
/// other corrupts every stdlib call site.
const STDLIB_FN_BASE: u16 = 40_000;

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
    /// a dynamic array of values, built by `MAKE_ARRAY`. `push` / `pop` mutate
    /// it; `INDEX` reads an element; `LEN` reports the element count.
    Array(Vec<Value>),
    /// a fixed-shape tuple of values, built by `MAKE_TUPLE`. distinct from
    /// [`HeapObject::Array`] so the stdlib's `type_of` (plan 05-05) can render
    /// a tuple's structural type `(i64, str)` rather than an array type. `INDEX`
    /// reads an element exactly as it does for an array.
    Tuple(Vec<Value>),
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

    // ---- dispatch ----------------------------------------------------------

    /// decode and execute exactly one instruction at the current frame's `ip`.
    ///
    /// the shared core of [`Vm::run`] and [`Vm::step`]; there is no dispatch
    /// logic anywhere else. the sequence is: bounds-check `ip < code.len()`,
    /// decode the opcode byte, bounds-check that the whole operand sequence is
    /// in range, advance `ip` PAST the whole instruction, then run the handler.
    ///
    /// ip-advance discipline: `ip` is advanced to the fall-through position
    /// (`opcode_pos + 1 + operand_bytes`) BEFORE the handler body runs. a JUMP
    /// handler then overwrites `ip` with its computed target. every other
    /// handler leaves the advanced `ip` in place. this is why a JUMP target is
    /// `fall_through + offset` -- `fall_through` is already `opcode_pos + 1 + 2`,
    /// exactly the "byte after the operand" the offset is relative to.
    ///
    /// every byte read here is bounds-checked; a bad opcode byte or a truncated
    /// operand is a [`QalaError::Runtime`], never a panic.
    fn dispatch_one(&mut self) -> Result<StepOutcome, QalaError> {
        let ip = self.frame()?.ip;
        let code_len = self.chunk()?.code.len();
        if ip >= code_len {
            return Err(self.runtime_err("instruction pointer past end of chunk"));
        }
        // decode the opcode byte -- guaranteed in range by the check above.
        let byte = self.chunk()?.code[ip];
        let op = Opcode::from_u8(byte)
            .ok_or_else(|| self.runtime_err(&format!("bad opcode byte {byte:#x}")))?;
        // bounds-check the operand sequence before any read of it.
        let operand_len = op.operand_bytes() as usize;
        if ip + 1 + operand_len > code_len {
            return Err(self.runtime_err("truncated operand"));
        }
        // the fall-through ip: one past the whole instruction.
        let next = ip + 1 + operand_len;
        // advance FIRST; a JUMP handler below overwrites this.
        self.frame_mut()?.ip = next;
        self.run_opcode(op, ip, next)
    }

    /// run the handler for `op`, whose opcode byte is at `opcode_pos` and whose
    /// fall-through ip is `next`. split out of [`Vm::dispatch_one`] so the
    /// decode-and-bounds-check is one place and the per-opcode logic another.
    ///
    /// every opcode this plan does not implement returns a clean `Runtime`
    /// "opcode not yet implemented" error so the match stays exhaustive and the
    /// crate compiles; a later commit replaces those arms.
    fn run_opcode(
        &mut self,
        op: Opcode,
        opcode_pos: usize,
        next: usize,
    ) -> Result<StepOutcome, QalaError> {
        match op {
            // ---- stack ----
            Opcode::Const => {
                let idx = self.read_operand_u16(opcode_pos)?;
                self.op_const(idx)?;
            }
            Opcode::Pop => {
                self.pop()?;
            }
            Opcode::Dup => {
                // peek the top value and push a copy. a Value is Copy, so the
                // duplicate is a plain bit copy; refcount bookkeeping for a
                // duplicated heap pointer is a later commit's concern.
                let top = *self
                    .stack
                    .last()
                    .ok_or_else(|| self.runtime_err("stack underflow"))?;
                self.push(top)?;
            }
            // ---- locals + globals ----
            Opcode::GetLocal => {
                let slot = self.read_operand_u16(opcode_pos)? as usize;
                let v = *self
                    .frame()?
                    .locals
                    .get(slot)
                    .ok_or_else(|| self.runtime_err(&format!("bad local slot {slot}")))?;
                self.push(v)?;
            }
            Opcode::SetLocal => {
                let slot = self.read_operand_u16(opcode_pos)? as usize;
                let v = self.pop()?;
                let locals = &mut self.frame_mut()?.locals;
                // a SET_LOCAL past the current end grows the vec with void
                // padding -- codegen numbers slots densely from 0, so the only
                // way a slot lands past the end is the first write to it.
                if slot >= locals.len() {
                    locals.resize(slot + 1, Value::void());
                }
                locals[slot] = v;
            }
            Opcode::GetGlobal => {
                let idx = self.read_operand_u16(opcode_pos)? as usize;
                let v = *self
                    .globals
                    .get(idx)
                    .ok_or_else(|| self.runtime_err(&format!("bad global slot {idx}")))?;
                self.push(v)?;
            }
            Opcode::SetGlobal => {
                let idx = self.read_operand_u16(opcode_pos)? as usize;
                let v = self.pop()?;
                if idx >= self.globals.len() {
                    self.globals.resize(idx + 1, Value::void());
                }
                self.globals[idx] = v;
            }
            // ---- i64 arithmetic ----
            // every result is checked: an overflow (or i64::MIN / -1, or a
            // zero divisor) is a Runtime error, never a Rust panic. the result
            // is a fresh heap Int -- i64 is uniformly heap-boxed.
            Opcode::Add => self.op_arith_i64(IntOp::Add)?,
            Opcode::Sub => self.op_arith_i64(IntOp::Sub)?,
            Opcode::Mul => self.op_arith_i64(IntOp::Mul)?,
            Opcode::Div => self.op_arith_i64(IntOp::Div)?,
            Opcode::Mod => self.op_arith_i64(IntOp::Mod)?,
            Opcode::Neg => {
                let n = self.pop_i64()?;
                let r = n
                    .checked_neg()
                    .ok_or_else(|| self.runtime_err("integer overflow"))?;
                self.push_i64(r)?;
            }
            // ---- f64 arithmetic (IEEE 754, no error -- inf / NaN are valid) ----
            Opcode::FAdd => self.op_arith_f64(FloatOp::Add)?,
            Opcode::FSub => self.op_arith_f64(FloatOp::Sub)?,
            Opcode::FMul => self.op_arith_f64(FloatOp::Mul)?,
            Opcode::FDiv => self.op_arith_f64(FloatOp::Div)?,
            Opcode::FNeg => {
                let x = self.pop_f64()?;
                self.push(Value::from_f64(-x))?;
            }
            // ---- comparisons (the VM dispatches by operand type) ----
            Opcode::Eq => self.op_compare(CmpOp::Eq)?,
            Opcode::Ne => self.op_compare(CmpOp::Ne)?,
            Opcode::Lt => self.op_compare(CmpOp::Lt)?,
            Opcode::Le => self.op_compare(CmpOp::Le)?,
            Opcode::Gt => self.op_compare(CmpOp::Gt)?,
            Opcode::Ge => self.op_compare(CmpOp::Ge)?,
            // ---- f64 comparisons (IEEE 754: NaN compares unequal) ----
            Opcode::FEq => self.op_compare_f64(CmpOp::Eq)?,
            Opcode::FNe => self.op_compare_f64(CmpOp::Ne)?,
            Opcode::FLt => self.op_compare_f64(CmpOp::Lt)?,
            Opcode::FLe => self.op_compare_f64(CmpOp::Le)?,
            Opcode::FGt => self.op_compare_f64(CmpOp::Gt)?,
            Opcode::FGe => self.op_compare_f64(CmpOp::Ge)?,
            // ---- logic ----
            Opcode::Not => {
                let b = self.pop_bool()?;
                self.push(Value::bool(!b))?;
            }
            // ---- control flow ----
            Opcode::Jump => {
                let offset = self.read_operand_i16(opcode_pos)?;
                self.do_jump(next, offset)?;
            }
            Opcode::JumpIfFalse => {
                let offset = self.read_operand_i16(opcode_pos)?;
                if !self.pop_bool()? {
                    self.do_jump(next, offset)?;
                }
            }
            Opcode::JumpIfTrue => {
                let offset = self.read_operand_i16(opcode_pos)?;
                if self.pop_bool()? {
                    self.do_jump(next, offset)?;
                }
            }
            // ---- calls ----
            Opcode::Call => {
                let fn_id = self.read_operand_u16(opcode_pos)?;
                let argc = self.read_operand_u8(opcode_pos)?;
                self.op_call(fn_id, argc)?;
            }
            Opcode::Return => {
                // a RETURN from the last frame ends the program.
                if self.op_return()? {
                    return Ok(StepOutcome::Halted);
                }
            }
            // ---- heap construction ----
            Opcode::MakeArray => {
                let count = self.read_operand_u16(opcode_pos)?;
                self.op_make_collection(count, false)?;
            }
            Opcode::MakeTuple => {
                let count = self.read_operand_u16(opcode_pos)?;
                self.op_make_collection(count, true)?;
            }
            Opcode::MakeStruct => {
                let struct_id = self.read_operand_u16(opcode_pos)?;
                self.op_make_struct(struct_id)?;
            }
            Opcode::MakeEnumVariant => {
                let variant_id = self.read_operand_u16(opcode_pos)?;
                let payload_count = self.read_operand_u8(opcode_pos)?;
                self.op_make_enum_variant(variant_id, payload_count)?;
            }
            // ---- access ----
            Opcode::Index => self.op_index()?,
            Opcode::Field => {
                let field_index = self.read_operand_u16(opcode_pos)?;
                self.op_field(field_index)?;
            }
            Opcode::Len => self.op_len()?,
            // ---- strings ----
            Opcode::ToStr => self.op_to_str()?,
            Opcode::ConcatN => {
                let count = self.read_operand_u16(opcode_pos)?;
                self.op_concat_n(count)?;
            }
            // ---- match dispatch ----
            Opcode::MatchVariant => {
                let variant_id = self.read_operand_u16(opcode_pos)?;
                let offset = self.read_match_variant_offset(opcode_pos)?;
                self.op_match_variant(variant_id, offset, next)?;
            }
            // ---- sentinel ----
            Opcode::Halt => return Ok(StepOutcome::Halted),
        }
        let _ = next;
        Ok(StepOutcome::Ran)
    }

    /// read the `u16` operand of the instruction whose opcode byte is at
    /// `opcode_pos`. the operand sits at `opcode_pos + 1`.
    ///
    /// [`Vm::dispatch_one`] has already bounds-checked the whole operand
    /// sequence, so the two-byte read cannot go out of range -- but this
    /// re-checks defensively rather than trust the caller, keeping the
    /// never-panic guarantee local to this function.
    fn read_operand_u16(&self, opcode_pos: usize) -> Result<u16, QalaError> {
        let chunk = self.chunk()?;
        let off = opcode_pos + 1;
        if off + 2 > chunk.code.len() {
            return Err(self.runtime_err("truncated operand"));
        }
        Ok(chunk.read_u16(off))
    }

    /// read the trailing `u8` operand of a three-byte instruction (the `argc`
    /// of [`Opcode::Call`], the `payload_count` of [`Opcode::MakeEnumVariant`])
    /// whose opcode byte is at `opcode_pos`.
    ///
    /// the byte sits at `opcode_pos + 3` -- after the `u16` that precedes it.
    /// [`Vm::dispatch_one`] has already bounds-checked the whole three-byte
    /// operand sequence, but this re-checks defensively so the never-panic
    /// guarantee stays local to this function.
    fn read_operand_u8(&self, opcode_pos: usize) -> Result<u8, QalaError> {
        let chunk = self.chunk()?;
        let off = opcode_pos + 3;
        chunk
            .code
            .get(off)
            .copied()
            .ok_or_else(|| self.runtime_err("truncated operand"))
    }

    /// read the trailing `i16` miss-offset of a [`Opcode::MatchVariant`]
    /// instruction whose opcode byte is at `opcode_pos`.
    ///
    /// MATCH_VARIANT's operand layout is `u16 variant_id` then `i16 offset`, so
    /// the signed offset sits at `opcode_pos + 3`. [`Vm::dispatch_one`] has
    /// already bounds-checked the four-byte operand sequence; this re-checks
    /// the two-byte read defensively so the never-panic guarantee stays local.
    fn read_match_variant_offset(&self, opcode_pos: usize) -> Result<i16, QalaError> {
        let chunk = self.chunk()?;
        let off = opcode_pos + 3;
        if off + 2 > chunk.code.len() {
            return Err(self.runtime_err("truncated operand"));
        }
        Ok(chunk.read_i16(off))
    }

    /// the constant-pool entry at `idx` converted to a runtime [`Value`] and
    /// pushed onto the stack.
    ///
    /// an `i64` constant allocates a [`HeapObject::Int`] and pushes a pointer
    /// (the uniform-heap-box rule); a `str` constant allocates a
    /// [`HeapObject::Str`]; an `f64` / `bool` / `byte` / `void` becomes a
    /// tagged scalar directly; a function constant becomes a [`Value::function`]
    /// tagged scalar carrying the fn-id (no heap object). a bad pool index or a
    /// heap-exhaustion is a [`QalaError::Runtime`].
    fn op_const(&mut self, idx: u16) -> Result<(), QalaError> {
        let constant = self
            .chunk()?
            .constants
            .get(idx as usize)
            .cloned()
            .ok_or_else(|| self.runtime_err(&format!("bad constant index {idx}")))?;
        let value = match constant {
            ConstValue::I64(n) => {
                let slot = self
                    .heap
                    .alloc(HeapObject::Int(n))
                    .ok_or_else(|| self.runtime_err("heap exhausted"))?;
                Value::pointer(slot)
            }
            ConstValue::F64(x) => Value::from_f64(x),
            ConstValue::Bool(b) => Value::bool(b),
            ConstValue::Byte(b) => Value::byte(b),
            ConstValue::Void => Value::void(),
            ConstValue::Str(s) => {
                let slot = self
                    .heap
                    .alloc(HeapObject::Str(s))
                    .ok_or_else(|| self.runtime_err("heap exhausted"))?;
                Value::pointer(slot)
            }
            // a function value is a tagged scalar, not a heap object: the u16
            // fn-id rides in the NaN payload. the higher-order stdlib functions
            // recover it via Value::as_function.
            ConstValue::Function(id) => Value::function(id),
        };
        self.push(value)
    }

    // ---- typed pops + the i64 push ----------------------------------------

    /// pop a value and decode it as an `i64`.
    ///
    /// an `i64` runtime value is a `TAG_PTR` pointer to a [`HeapObject::Int`]
    /// (the uniform-heap-box rule). a value that is not a pointer, or a pointer
    /// to a non-`Int` heap object, is a [`QalaError::Runtime`] "expected an
    /// integer" -- never a panic.
    fn pop_i64(&mut self) -> Result<i64, QalaError> {
        let v = self.pop()?;
        let slot = v
            .as_pointer()
            .ok_or_else(|| self.runtime_err("expected an integer"))?;
        match self.heap.get(slot) {
            Some(HeapObject::Int(n)) => Ok(*n),
            _ => Err(self.runtime_err("expected an integer")),
        }
    }

    /// pop a value and decode it as an `f64`.
    ///
    /// an `f64` is stored verbatim in the value (not heap-boxed). a non-`f64`
    /// value is a [`QalaError::Runtime`] "expected a float".
    fn pop_f64(&mut self) -> Result<f64, QalaError> {
        let v = self.pop()?;
        v.as_f64()
            .ok_or_else(|| self.runtime_err("expected a float"))
    }

    /// pop a value and decode it as a `bool`.
    ///
    /// a non-`bool` value is a [`QalaError::Runtime`] "expected a boolean".
    fn pop_bool(&mut self) -> Result<bool, QalaError> {
        let v = self.pop()?;
        v.as_bool()
            .ok_or_else(|| self.runtime_err("expected a boolean"))
    }

    /// pop a value and decode it as an owned `String`.
    ///
    /// a `str` runtime value is a `TAG_PTR` pointer to a [`HeapObject::Str`].
    /// the string is cloned out so the caller owns it independently of the
    /// heap slot. a non-string value is a [`QalaError::Runtime`] "expected a
    /// string".
    ///
    /// `allow(dead_code)`: the string-comparison path in `op_compare` uses the
    /// heap slot directly; this typed pop is part of the pop-helper family the
    /// later string opcodes (`CONCAT_N`, `TO_STR`) consume.
    #[allow(dead_code)]
    fn pop_str(&mut self) -> Result<String, QalaError> {
        let v = self.pop()?;
        let slot = v
            .as_pointer()
            .ok_or_else(|| self.runtime_err("expected a string"))?;
        match self.heap.get(slot) {
            Some(HeapObject::Str(s)) => Ok(s.clone()),
            _ => Err(self.runtime_err("expected a string")),
        }
    }

    /// allocate a [`HeapObject::Int`] for `n` and push a pointer to it.
    ///
    /// the shared tail of every i64-producing opcode: `i64` is uniformly
    /// heap-boxed, so an integer result is always a fresh heap object reached
    /// through a pointer. a heap-exhaustion is a [`QalaError::Runtime`].
    fn push_i64(&mut self, n: i64) -> Result<(), QalaError> {
        let slot = self
            .heap
            .alloc(HeapObject::Int(n))
            .ok_or_else(|| self.runtime_err("heap exhausted"))?;
        self.push(Value::pointer(slot))
    }

    /// read the `i16` operand (a signed jump offset) of the instruction whose
    /// opcode byte is at `opcode_pos`.
    ///
    /// like [`Vm::read_operand_u16`] this re-checks the two-byte read is in
    /// range rather than trust the caller, keeping the never-panic guarantee
    /// local.
    fn read_operand_i16(&self, opcode_pos: usize) -> Result<i16, QalaError> {
        let chunk = self.chunk()?;
        let off = opcode_pos + 1;
        if off + 2 > chunk.code.len() {
            return Err(self.runtime_err("truncated operand"));
        }
        Ok(chunk.read_i16(off))
    }

    // ---- arithmetic / comparison / jump handlers --------------------------

    /// pop two `i64` operands and push the checked result of `op`.
    ///
    /// the deeper stack value is the left operand. every operation goes through
    /// `i64::checked_*`: an overflow is "integer overflow", a zero divisor (or
    /// the `i64::MIN / -1` overflow that `checked_div` / `checked_rem` also
    /// reject) is "division by zero" / "modulo by zero". one `None` check on
    /// the checked op covers both faults.
    fn op_arith_i64(&mut self, op: IntOp) -> Result<(), QalaError> {
        // pop order: rhs is on top, lhs is below it.
        let rhs = self.pop_i64()?;
        let lhs = self.pop_i64()?;
        let result = match op {
            IntOp::Add => lhs
                .checked_add(rhs)
                .ok_or_else(|| self.runtime_err("integer overflow"))?,
            IntOp::Sub => lhs
                .checked_sub(rhs)
                .ok_or_else(|| self.runtime_err("integer overflow"))?,
            IntOp::Mul => lhs
                .checked_mul(rhs)
                .ok_or_else(|| self.runtime_err("integer overflow"))?,
            IntOp::Div => lhs
                .checked_div(rhs)
                .ok_or_else(|| self.runtime_err("division by zero"))?,
            IntOp::Mod => lhs
                .checked_rem(rhs)
                .ok_or_else(|| self.runtime_err("modulo by zero"))?,
        };
        self.push_i64(result)
    }

    /// pop two `f64` operands and push the IEEE 754 result of `op`.
    ///
    /// the deeper stack value is the left operand. float arithmetic never
    /// errors -- `FDiv` of a zero divisor yields `inf` / `-inf` / `NaN` per
    /// IEEE 754, exactly as plain Rust `f64` `/` does.
    fn op_arith_f64(&mut self, op: FloatOp) -> Result<(), QalaError> {
        let rhs = self.pop_f64()?;
        let lhs = self.pop_f64()?;
        let result = match op {
            FloatOp::Add => lhs + rhs,
            FloatOp::Sub => lhs - rhs,
            FloatOp::Mul => lhs * rhs,
            FloatOp::Div => lhs / rhs,
        };
        self.push(Value::from_f64(result))
    }

    /// pop two values, compare them by their runtime kind, and push the `bool`
    /// result of `op`.
    ///
    /// the two operands must be the same kind. supported kinds: two heap
    /// `Int`s (compared as `i64`), two heap `Str`s (compared lexicographically
    /// as Rust strings), or two `bool`s. a `bool` ordering follows Rust's
    /// `false < true`; the typechecker only emits an ordering comparison on a
    /// type it has already accepted, so the runtime ordering is well-defined
    /// for every case the typechecker lets through. a kind mismatch between the
    /// two operands is a [`QalaError::Runtime`].
    fn op_compare(&mut self, op: CmpOp) -> Result<(), QalaError> {
        let rhs = self.pop()?;
        let lhs = self.pop()?;
        let ordering = self.compare_values(lhs, rhs)?;
        self.push(Value::bool(op.holds(ordering)))
    }

    /// the [`Ordering`](std::cmp::Ordering) of two same-kind values.
    ///
    /// pulled out of [`Vm::op_compare`] so the kind dispatch is one place. a
    /// pointer is resolved to its heap object before comparison; a kind
    /// mismatch is a [`QalaError::Runtime`].
    fn compare_values(
        &self,
        lhs: Value,
        rhs: Value,
    ) -> Result<std::cmp::Ordering, QalaError> {
        // both bools: order false < true.
        if let (Some(a), Some(b)) = (lhs.as_bool(), rhs.as_bool()) {
            return Ok(a.cmp(&b));
        }
        // otherwise both must be pointers to comparable heap objects.
        match (lhs.as_pointer(), rhs.as_pointer()) {
            (Some(a), Some(b)) => match (self.heap.get(a), self.heap.get(b)) {
                (Some(HeapObject::Int(x)), Some(HeapObject::Int(y))) => Ok(x.cmp(y)),
                (Some(HeapObject::Str(x)), Some(HeapObject::Str(y))) => Ok(x.cmp(y)),
                _ => Err(self.runtime_err("cannot compare values of different types")),
            },
            _ => Err(self.runtime_err("cannot compare values of different types")),
        }
    }

    /// pop two `f64` operands, compare them per IEEE 754, push the `bool`.
    ///
    /// IEEE 754 ordering: a `NaN` compares unequal to everything including
    /// itself, and is unordered, so `FEq` of two `NaN`s is `false`, `FNe` is
    /// `true`, and `FLt` / `FLe` / `FGt` / `FGe` involving a `NaN` are all
    /// `false`. this falls out of Rust's `f64` `PartialOrd` / `==` directly.
    fn op_compare_f64(&mut self, op: CmpOp) -> Result<(), QalaError> {
        let rhs = self.pop_f64()?;
        let lhs = self.pop_f64()?;
        let result = match op {
            CmpOp::Eq => lhs == rhs,
            CmpOp::Ne => lhs != rhs,
            CmpOp::Lt => lhs < rhs,
            CmpOp::Le => lhs <= rhs,
            CmpOp::Gt => lhs > rhs,
            CmpOp::Ge => lhs >= rhs,
        };
        self.push(Value::bool(result))
    }

    /// take a jump: set the current frame's `ip` to `fall_through + offset`.
    ///
    /// `fall_through` is the byte AFTER the whole jump instruction -- which is
    /// exactly the position the codegen's signed `i16` offset is relative to.
    /// the computed target must land in `0..=code.len()`; a target outside the
    /// chunk is a [`QalaError::Runtime`] "jump target out of range", never a
    /// silent out-of-bounds `ip`.
    fn do_jump(&mut self, fall_through: usize, offset: i16) -> Result<(), QalaError> {
        let code_len = self.chunk()?.code.len();
        let target = fall_through as isize + offset as isize;
        if target < 0 || target as usize > code_len {
            return Err(self.runtime_err("jump target out of range"));
        }
        self.frame_mut()?.ip = target as usize;
        Ok(())
    }

    // ---- the call machinery -----------------------------------------------

    /// execute a [`Opcode::Call`]: dispatch a user function or seam to the
    /// stdlib.
    ///
    /// `fn_id` is the `u16` from the opcode; a `fn_id >= STDLIB_FN_BASE`
    /// (40000) is a native stdlib call, handed to [`Vm::call_stdlib`] -- a seam
    /// plan 05-05 fills in. otherwise `fn_id` indexes [`Program::chunks`]: the
    /// frame-depth cap is checked, then a fresh [`CallFrame`] is pushed for the
    /// callee. the `argc` argument values are already on the value stack (the
    /// topmost is the rightmost argument); they are moved off the stack into
    /// the new frame's `locals` so `locals[0]` is the leftmost argument --
    /// matching how `GET_LOCAL` reads `frame.locals` and how codegen numbers
    /// parameter slots `0..argc`.
    ///
    /// the new frame's `base` is the value-stack length AFTER the arguments are
    /// removed -- the watermark `RETURN` truncates the stack back to. the
    /// caller frame's `ip` already sits at the instruction after the CALL
    /// (the ip-advance-first discipline), so `RETURN` resumes it correctly.
    fn op_call(&mut self, fn_id: u16, argc: u8) -> Result<(), QalaError> {
        // a stdlib fn-id: hand off to the native-dispatch seam.
        if fn_id >= STDLIB_FN_BASE {
            return self.call_stdlib(fn_id, argc);
        }
        // a user function: enforce the frame-depth cap before pushing. the VM
        // is a `while` loop, so a Qala recursion grows `frames`, not the host
        // Rust stack -- this cap turns unbounded recursion into a clean
        // `Runtime` error rather than a host stack overflow.
        if self.frames.len() >= MAX_FRAMES {
            return Err(self.runtime_err("stack overflow"));
        }
        // resolve the callee chunk, fallibly.
        let chunk_idx = fn_id as usize;
        if self.program.chunks.get(chunk_idx).is_none() {
            return Err(self.runtime_err(&format!("call to missing function {fn_id}")));
        }
        // move the argc arguments off the value stack into the new frame's
        // locals. they sit at the top of the stack, the rightmost on top; the
        // split keeps them in stack order so locals[0] is the leftmost arg.
        let argc = argc as usize;
        if self.stack.len() < argc {
            return Err(self.runtime_err("stack underflow building a call frame"));
        }
        let base = self.stack.len() - argc;
        let locals = self.stack.split_off(base);
        self.frames.push(CallFrame {
            chunk_idx,
            ip: 0,
            base,
            locals,
        });
        Ok(())
    }

    /// the stdlib-dispatch seam.
    ///
    /// a `CALL` whose `fn_id >= STDLIB_FN_BASE` reaches here. plan 05-05
    /// replaces this body with the real native dispatch (`print` / `println` /
    /// `len` / `push` / `pop` / `sqrt` / `abs` / `assert` / `type_of` / `map` /
    /// `filter` / `reduce` / `open` / `close` / `read_all`). until then it is a
    /// clean `Runtime` error so the crate compiles and a program that reaches a
    /// stdlib call fails loudly rather than silently misbehaving.
    ///
    /// `allow(unused_variables)` on `argc`: 05-05's body pops `argc` values;
    /// this stub does not, but the parameter is part of the locked seam
    /// signature `call_stdlib(fn_id: u16, argc: u8)` 05-05 is written against.
    #[allow(unused_variables)]
    fn call_stdlib(&mut self, fn_id: u16, argc: u8) -> Result<(), QalaError> {
        Err(self.runtime_err(&format!(
            "stdlib function {fn_id} not yet implemented"
        )))
    }

    /// execute a [`Opcode::Return`]: pop the current frame and pass the result
    /// back.
    ///
    /// the value on top of the stack is the call's result. the current
    /// [`CallFrame`] is popped and the value stack truncated back to that
    /// frame's `base`, discarding the callee's transient stack. when no frame
    /// remains the program is finished: returns `Ok(true)` so the caller halts
    /// the VM. otherwise the result value is pushed onto the (now caller's)
    /// stack and `Ok(false)` is returned.
    ///
    /// codegen emits an explicit `RETURN` at every function exit, including
    /// `main`'s -- a `RETURN` from the last frame is therefore the normal end
    /// of a program.
    fn op_return(&mut self) -> Result<bool, QalaError> {
        let result = self.pop()?;
        let frame = self
            .frames
            .pop()
            .ok_or_else(|| self.runtime_err("return with no active call frame"))?;
        // drop the callee's transient value stack back to its base.
        if frame.base <= self.stack.len() {
            self.stack.truncate(frame.base);
        }
        if self.frames.is_empty() {
            // the last frame returned -- the program is done. leave the result
            // on the stack so a test (or the REPL) can inspect a program's
            // final value.
            self.stack.push(result);
            return Ok(true);
        }
        // hand the result back to the caller.
        self.push(result)?;
        Ok(false)
    }

    /// call a function `Value` from native (stdlib) code and run it to its
    /// `RETURN`, returning the result value.
    ///
    /// this is the re-entry point the higher-order stdlib functions
    /// (`map` / `filter` / `reduce` in 05-05) use to invoke a user callback.
    /// `callable` must be a [`Value::function`] -- the callbacks those stdlib
    /// functions receive are user functions; a non-function (or a stdlib fn-id
    /// callable, unsupported in v1) is a `Runtime` error.
    ///
    /// the mechanism: record the current frame depth, push the `args` onto the
    /// value stack, push a [`CallFrame`] for the callee (frame-depth-capped,
    /// exactly as [`Vm::op_call`] does), then loop [`Vm::dispatch_one`] until
    /// the frame stack drops back to the recorded depth -- the callee's
    /// `RETURN` popped its frame. the callee's result is then on top of the
    /// stack. every loop local stays in a Rust local, so a callback that
    /// itself calls `call_function_value` is fully re-entrant.
    ///
    /// `allow(dead_code)`: the higher-order stdlib functions a later plan adds
    /// are the only callers; the helper is part of the locked VM surface and
    /// is declared now so 05-05 has the re-entry point ready.
    #[allow(dead_code)]
    fn call_function_value(
        &mut self,
        callable: Value,
        args: &[Value],
    ) -> Result<Value, QalaError> {
        let fn_id = callable
            .as_function()
            .ok_or_else(|| self.runtime_err("value is not callable"))?;
        // a stdlib fn-id as a callback is not supported in v1: the callbacks
        // map/filter/reduce receive are user functions.
        if fn_id >= STDLIB_FN_BASE {
            return Err(self.runtime_err(
                "a stdlib function cannot be used as a callback in v1",
            ));
        }
        let chunk_idx = fn_id as usize;
        if self.program.chunks.get(chunk_idx).is_none() {
            return Err(self.runtime_err(&format!("call to missing function {fn_id}")));
        }
        if self.frames.len() >= MAX_FRAMES {
            return Err(self.runtime_err("stack overflow"));
        }
        // the depth the callee's RETURN must drop the frame stack back to.
        let depth = self.frames.len();
        // push the arguments, then build the callee frame -- the args become
        // the frame's locals, locals[0] the first argument.
        for arg in args {
            self.push(*arg)?;
        }
        let base = self.stack.len() - args.len();
        let locals = self.stack.split_off(base);
        self.frames.push(CallFrame {
            chunk_idx,
            ip: 0,
            base,
            locals,
        });
        // run a nested dispatch loop until the callee's frame is popped. a
        // Halted outcome before the depth drops back means the callee fell off
        // the end without a RETURN -- a malformed-bytecode Runtime error.
        loop {
            if self.frames.len() == depth {
                break;
            }
            match self.dispatch_one()? {
                StepOutcome::Ran => {}
                StepOutcome::Halted => {
                    if self.frames.len() == depth {
                        break;
                    }
                    return Err(self.runtime_err("callback halted without returning"));
                }
            }
        }
        // the callee's RETURN pushed its result onto the stack.
        self.pop()
    }

    // ---- heap construction + access ---------------------------------------

    /// pop `n` values off the value stack, returned with the deepest popped
    /// value first.
    ///
    /// the `MAKE_*` opcodes push their elements in order, the last on top; a
    /// plain pop loop would yield them reversed, so this restores source order:
    /// `result[0]` is the first element / field / payload. a stack with fewer
    /// than `n` values is a [`QalaError::Runtime`] "stack underflow", never a
    /// panic.
    fn pop_n(&mut self, n: usize) -> Result<Vec<Value>, QalaError> {
        if self.stack.len() < n {
            return Err(self.runtime_err("stack underflow building a heap object"));
        }
        let at = self.stack.len() - n;
        Ok(self.stack.split_off(at))
    }

    /// build a [`HeapObject::Array`] (or [`HeapObject::Tuple`] when `tuple`)
    /// from the top `count` stack values and push a pointer to it.
    ///
    /// the values come off the stack in source order via [`Vm::pop_n`], so
    /// element 0 is the first. a heap exhaustion is a `Runtime` error. `MAKE_*`
    /// shares this one body because an array and a tuple differ only in which
    /// heap variant labels them -- the distinction lets 05-05's `type_of` tell
    /// a tuple from an array.
    fn op_make_collection(&mut self, count: u16, tuple: bool) -> Result<(), QalaError> {
        let elements = self.pop_n(count as usize)?;
        let object = if tuple {
            HeapObject::Tuple(elements)
        } else {
            HeapObject::Array(elements)
        };
        let slot = self
            .heap
            .alloc(object)
            .ok_or_else(|| self.runtime_err("heap exhausted"))?;
        self.push(Value::pointer(slot))
    }

    /// build a [`HeapObject::Struct`] for the struct whose id is `struct_id`
    /// and push a pointer to it.
    ///
    /// `struct_id` indexes [`Program::structs`]; the [`crate::chunk::StructInfo`]
    /// there gives the declared `name` (stored as the heap struct's
    /// `type_name`, so `type_of` returns the declared name) and the
    /// `field_count` (how many values to pop). the field values come off the
    /// stack in declaration order. a bad struct id or a heap exhaustion is a
    /// `Runtime` error.
    fn op_make_struct(&mut self, struct_id: u16) -> Result<(), QalaError> {
        let info = self
            .program
            .structs
            .get(struct_id as usize)
            .ok_or_else(|| self.runtime_err(&format!("bad struct id {struct_id}")))?;
        let type_name = info.name.clone();
        let field_count = info.field_count as usize;
        let fields = self.pop_n(field_count)?;
        let slot = self
            .heap
            .alloc(HeapObject::Struct { type_name, fields })
            .ok_or_else(|| self.runtime_err("heap exhausted"))?;
        self.push(Value::pointer(slot))
    }

    /// build a [`HeapObject::EnumVariant`] for the variant whose id is
    /// `variant_id` and push a pointer to it.
    ///
    /// `variant_id` indexes [`Program::enum_variant_names`], a table of
    /// `(enum_name, variant_name)` pairs; the heap object carries both names so
    /// `type_of` and the value-to-string routine can render it, and so
    /// `MATCH_VARIANT` can compare a scrutinee against an operand variant id by
    /// resolving that id to the same name pair. `payload_count` values are
    /// popped in declaration order. a bad variant id or a heap exhaustion is a
    /// `Runtime` error.
    fn op_make_enum_variant(
        &mut self,
        variant_id: u16,
        payload_count: u8,
    ) -> Result<(), QalaError> {
        let (enum_name, variant_name) = self
            .program
            .enum_variant_names
            .get(variant_id as usize)
            .ok_or_else(|| self.runtime_err(&format!("bad variant id {variant_id}")))?;
        let type_name = enum_name.clone();
        let variant = variant_name.clone();
        let payload = self.pop_n(payload_count as usize)?;
        let slot = self
            .heap
            .alloc(HeapObject::EnumVariant {
                type_name,
                variant,
                payload,
            })
            .ok_or_else(|| self.runtime_err("heap exhausted"))?;
        self.push(Value::pointer(slot))
    }

    /// execute [`Opcode::Index`]: pop an `i64` index then an array/tuple
    /// pointer, push the element at that index.
    ///
    /// the index is the topmost value, the collection below it. the pointer
    /// must reach a [`HeapObject::Array`] or [`HeapObject::Tuple`]; anything
    /// else is a `Runtime` error. a negative index, or one at or past the
    /// length, is a `Runtime` "array index N out of bounds for length L" -- the
    /// message names both the index and the length, and the error span covers
    /// the `INDEX` opcode's source line, never an out-of-bounds heap read.
    fn op_index(&mut self) -> Result<(), QalaError> {
        let index = self.pop_i64()?;
        let collection = self.pop()?;
        let slot = collection
            .as_pointer()
            .ok_or_else(|| self.runtime_err("expected an array"))?;
        let element = match self.heap.get(slot) {
            Some(HeapObject::Array(items)) | Some(HeapObject::Tuple(items)) => {
                let len = items.len();
                if index < 0 || index as usize >= len {
                    return Err(self.runtime_err(&format!(
                        "array index {index} out of bounds for length {len}"
                    )));
                }
                items[index as usize]
            }
            _ => return Err(self.runtime_err("expected an array")),
        };
        self.push(element)
    }

    /// execute [`Opcode::Field`]: pop a struct pointer, push the field at
    /// `field_index`.
    ///
    /// `field_index` is the field's stable position in the struct's
    /// declaration order -- codegen's `struct_field_index`. the pointer must
    /// reach a [`HeapObject::Struct`]; a non-struct, or an index past the
    /// field count, is a `Runtime` error.
    fn op_field(&mut self, field_index: u16) -> Result<(), QalaError> {
        let target = self.pop()?;
        let slot = target
            .as_pointer()
            .ok_or_else(|| self.runtime_err("expected a struct"))?;
        let field = match self.heap.get(slot) {
            Some(HeapObject::Struct { fields, .. }) => fields
                .get(field_index as usize)
                .copied()
                .ok_or_else(|| self.runtime_err(&format!("bad field index {field_index}")))?,
            _ => return Err(self.runtime_err("expected a struct")),
        };
        self.push(field)
    }

    /// execute [`Opcode::Len`]: pop an array, tuple, or string, push its length
    /// as a heap `i64`.
    ///
    /// an array's / tuple's length is its element count. a string's length is
    /// its count of Unicode scalar values (`chars().count()`) -- the
    /// user-facing "number of characters", not the UTF-8 byte count, matching
    /// what a teaching language's `len` of a string is expected to mean. any
    /// other value is a `Runtime` error.
    fn op_len(&mut self) -> Result<(), QalaError> {
        let value = self.pop()?;
        let slot = value
            .as_pointer()
            .ok_or_else(|| self.runtime_err("expected an array or string"))?;
        let len = match self.heap.get(slot) {
            Some(HeapObject::Array(items)) | Some(HeapObject::Tuple(items)) => items.len(),
            Some(HeapObject::Str(s)) => s.chars().count(),
            _ => return Err(self.runtime_err("expected an array or string")),
        };
        self.push_i64(len as i64)
    }

    // ---- the value-to-string routine + string opcodes ---------------------

    /// render a runtime [`Value`] to its display string.
    ///
    /// the spelling is locked and matches `ConstValue`'s `Display` for the
    /// primitive kinds, so a runtime value renders the same way its
    /// compile-time constant would:
    /// - an `i64` (a heap `Int`): the decimal form, e.g. `-7`.
    /// - an `f64`: non-finite values hand-spelled `NaN` / `inf` / `-inf`
    ///   (so a `println` of a NaN float shows `NaN`); a finite value uses
    ///   Rust's default `f64` `Display`.
    /// - a `bool`: `true` / `false`.
    /// - a `byte`: the plain decimal value (e.g. `65`) -- the readable form for
    ///   a `println`, NOT `ConstValue`'s `b'\xNN'` disassembly spelling.
    /// - `void`: `()`.
    /// - a `str` (a heap `Str`): the raw inner text, UNquoted -- `println` of a
    ///   string shows the string itself, not a quoted literal.
    /// - an array `[a, b, c]`, a tuple `(a, b, c)`, a struct
    ///   `Name { f0, f1 }`, an enum variant `Enum::Variant(p0, p1)` (or just
    ///   `Enum::Variant` with no payload) -- rendered recursively.
    /// - a file handle: a `<file "path">` placeholder; the VM does no real I/O.
    ///
    /// reused by [`Vm::op_to_str`], [`Vm::op_concat_n`], and 05-05's
    /// `print` / `println`. a dangling pointer (a freed slot) is rendered as
    /// `<dangling>` rather than erroring -- a display routine must never fail.
    fn value_to_string(&self, v: Value) -> String {
        // a tagged scalar: bool / byte / void / function decode directly.
        if let Some(b) = v.as_bool() {
            return if b { "true".to_string() } else { "false".to_string() };
        }
        if let Some(b) = v.as_byte() {
            return b.to_string();
        }
        if v.as_void() {
            return "()".to_string();
        }
        if let Some(id) = v.as_function() {
            return format!("fn#{id}");
        }
        // a pointer: dispatch on the heap object.
        if let Some(slot) = v.as_pointer() {
            return match self.heap.get(slot) {
                Some(HeapObject::Int(n)) => n.to_string(),
                Some(HeapObject::Str(s)) => s.clone(),
                Some(HeapObject::Array(items)) => {
                    let parts: Vec<String> =
                        items.iter().map(|e| self.value_to_string(*e)).collect();
                    format!("[{}]", parts.join(", "))
                }
                Some(HeapObject::Tuple(items)) => {
                    let parts: Vec<String> =
                        items.iter().map(|e| self.value_to_string(*e)).collect();
                    format!("({})", parts.join(", "))
                }
                Some(HeapObject::Struct { type_name, fields }) => {
                    let parts: Vec<String> =
                        fields.iter().map(|e| self.value_to_string(*e)).collect();
                    format!("{type_name} {{ {} }}", parts.join(", "))
                }
                Some(HeapObject::EnumVariant {
                    type_name,
                    variant,
                    payload,
                }) => {
                    if payload.is_empty() {
                        format!("{type_name}::{variant}")
                    } else {
                        let parts: Vec<String> =
                            payload.iter().map(|e| self.value_to_string(*e)).collect();
                        format!("{type_name}::{variant}({})", parts.join(", "))
                    }
                }
                Some(HeapObject::FileHandle { path, .. }) => format!("<file \"{path}\">"),
                None => "<dangling>".to_string(),
            };
        }
        // not tagged and not a pointer: a real f64. hand-spell non-finite
        // values to match ConstValue's Display.
        match v.as_f64() {
            Some(x) if x.is_nan() => "NaN".to_string(),
            Some(x) if x == f64::INFINITY => "inf".to_string(),
            Some(x) if x == f64::NEG_INFINITY => "-inf".to_string(),
            Some(x) => format!("{x}"),
            // unreachable: a value is a tagged scalar, a pointer, or an f64.
            None => "<unknown>".to_string(),
        }
    }

    /// execute [`Opcode::ToStr`]: pop one value, push its string form as a
    /// heap [`HeapObject::Str`].
    ///
    /// used to materialise an interpolated segment whose static type is not
    /// already `str`. a heap exhaustion is a `Runtime` error.
    fn op_to_str(&mut self) -> Result<(), QalaError> {
        let v = self.pop()?;
        let s = self.value_to_string(v);
        let slot = self
            .heap
            .alloc(HeapObject::Str(s))
            .ok_or_else(|| self.runtime_err("heap exhausted"))?;
        self.push(Value::pointer(slot))
    }

    /// execute [`Opcode::ConcatN`]: pop `count` values, concatenate their
    /// string forms in source order, push the result as a heap `Str`.
    ///
    /// the values come off the stack in source order via [`Vm::pop_n`] (the
    /// last-pushed is the last segment). each is rendered with
    /// [`Vm::value_to_string`], so a string interpolation of a NaN float
    /// renders `NaN`. used to materialise string interpolation. a heap
    /// exhaustion is a `Runtime` error.
    fn op_concat_n(&mut self, count: u16) -> Result<(), QalaError> {
        let parts = self.pop_n(count as usize)?;
        let mut joined = String::new();
        for part in parts {
            joined.push_str(&self.value_to_string(part));
        }
        let slot = self
            .heap
            .alloc(HeapObject::Str(joined))
            .ok_or_else(|| self.runtime_err("heap exhausted"))?;
        self.push(Value::pointer(slot))
    }

    /// execute [`Opcode::MatchVariant`]: test the scrutinee on top of the stack
    /// against `variant_id` and either destructure it or branch.
    ///
    /// `fall_through` is the byte after the whole four-byte-operand instruction
    /// -- the position the signed miss `offset` is relative to.
    ///
    /// the scrutinee must be a [`HeapObject::EnumVariant`]; a non-enum
    /// scrutinee is a `Runtime` error. the comparison key is the
    /// `(enum_name, variant_name)` pair: the heap `EnumVariant` stores names,
    /// not the id, so the operand `variant_id` is resolved through
    /// [`Program::enum_variant_names`] to a name pair and the two pairs are
    /// compared. on a match the scrutinee is consumed (`pop`) and each payload
    /// value is pushed -- the first payload field deepest, the last on top --
    /// so the arm's destructuring `SET_LOCAL`s bind them. on a miss the
    /// scrutinee is left on the stack and `ip` is set to `fall_through + offset`
    /// (bounds-checked into the chunk via [`Vm::do_jump`]) so a `MATCH_VARIANT`
    /// chain can re-test the same scrutinee against the next arm.
    fn op_match_variant(
        &mut self,
        variant_id: u16,
        offset: i16,
        fall_through: usize,
    ) -> Result<(), QalaError> {
        // peek the scrutinee without consuming it -- a miss must leave it.
        let scrutinee = *self
            .stack
            .last()
            .ok_or_else(|| self.runtime_err("stack underflow at a match"))?;
        let slot = scrutinee
            .as_pointer()
            .ok_or_else(|| self.runtime_err("match scrutinee is not an enum value"))?;
        // resolve the operand variant id to its (enum, variant) name pair.
        let (want_enum, want_variant) = self
            .program
            .enum_variant_names
            .get(variant_id as usize)
            .ok_or_else(|| self.runtime_err(&format!("bad variant id {variant_id}")))?
            .clone();
        // read the scrutinee's own enum/variant names and payload.
        let (matches, payload) = match self.heap.get(slot) {
            Some(HeapObject::EnumVariant {
                type_name,
                variant,
                payload,
            }) => {
                let hit = *type_name == want_enum && *variant == want_variant;
                (hit, if hit { payload.clone() } else { Vec::new() })
            }
            _ => {
                return Err(self.runtime_err("match scrutinee is not an enum value"));
            }
        };
        if matches {
            // consume the scrutinee, push the payload (first field deepest).
            self.pop()?;
            for value in payload {
                self.push(value)?;
            }
            Ok(())
        } else {
            // a miss: leave the scrutinee, branch to the next arm.
            self.do_jump(fall_through, offset)
        }
    }

    /// run the program from the current state to `Halt` or the first runtime
    /// error.
    ///
    /// loops over [`Vm::dispatch_one`]; a `Ran` outcome continues, a `Halted`
    /// outcome returns `Ok(())`. shares every byte of execution logic with
    /// [`Vm::step`] -- both go through `dispatch_one`.
    pub fn run(&mut self) -> Result<(), QalaError> {
        loop {
            match self.dispatch_one()? {
                StepOutcome::Ran => continue,
                StepOutcome::Halted => return Ok(()),
            }
        }
    }

    /// advance exactly one instruction.
    ///
    /// one `step()` call executes one full instruction including its operands:
    /// `ip` moves by `1 + operand_bytes()` of the executed opcode. the
    /// playground's step-through calls this in a loop. a thin wrapper over
    /// [`Vm::dispatch_one`] -- no duplicated dispatch logic.
    pub fn step(&mut self) -> Result<StepOutcome, QalaError> {
        self.dispatch_one()
    }
}

/// what one [`Vm::dispatch_one`] call did.
///
/// `Ran` -- an ordinary instruction executed and the VM should keep going.
/// `Halted` -- the VM hit [`Opcode::Halt`] (or fell off the end of `main` once
/// the call machinery lands) and execution is complete. a runtime fault is the
/// `Err` arm of the surrounding `Result`, not a variant here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepOutcome {
    /// an ordinary instruction ran; `run` should dispatch the next one.
    Ran,
    /// the VM halted; `run` returns `Ok(())`.
    Halted,
}

/// which checked `i64` arithmetic the `op_arith_i64` handler performs.
///
/// a small private selector so the five integer-arithmetic opcodes share one
/// handler instead of five near-identical bodies.
#[derive(Clone, Copy)]
enum IntOp {
    /// checked addition.
    Add,
    /// checked subtraction.
    Sub,
    /// checked multiplication.
    Mul,
    /// checked division -- a zero divisor (or `i64::MIN / -1`) is an error.
    Div,
    /// checked remainder -- a zero divisor (or `i64::MIN % -1`) is an error.
    Mod,
}

/// which IEEE 754 `f64` arithmetic the `op_arith_f64` handler performs.
///
/// the float counterpart of [`IntOp`]; there is no `FloatOp::Neg` because
/// `FNeg` is a single sign flip handled inline.
#[derive(Clone, Copy)]
enum FloatOp {
    /// addition.
    Add,
    /// subtraction.
    Sub,
    /// multiplication.
    Mul,
    /// division -- never an error; a zero divisor yields `inf` / `NaN`.
    Div,
}

/// which comparison the `op_compare` / `op_compare_f64` handlers test.
///
/// shared by the integer/string/bool comparisons and the `f64` comparisons so
/// the twelve comparison opcodes route through two handlers.
#[derive(Clone, Copy)]
enum CmpOp {
    /// `==`
    Eq,
    /// `!=`
    Ne,
    /// `<`
    Lt,
    /// `<=`
    Le,
    /// `>`
    Gt,
    /// `>=`
    Ge,
}

impl CmpOp {
    /// whether this comparison holds for an [`Ordering`](std::cmp::Ordering).
    ///
    /// used by the integer / string / bool comparison path, which reduces two
    /// operands to one `Ordering` and then asks each `CmpOp` whether it is
    /// satisfied. the `f64` path does not use this -- IEEE 754 has an
    /// unordered case (`NaN`) that an `Ordering` cannot express, so the float
    /// handler compares with `f64`'s own operators directly.
    fn holds(self, ordering: std::cmp::Ordering) -> bool {
        use std::cmp::Ordering::{Equal, Greater, Less};
        match self {
            CmpOp::Eq => ordering == Equal,
            CmpOp::Ne => ordering != Equal,
            CmpOp::Lt => ordering == Less,
            CmpOp::Le => ordering != Greater,
            CmpOp::Gt => ordering == Greater,
            CmpOp::Ge => ordering != Less,
        }
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

    // ---- dispatch + malformed bytecode + stack/local opcodes ----

    /// emit a `CONST idx` instruction for the pool entry `v` on `line`, return
    /// the pool index. a building block for the dispatch tests.
    fn emit_const(chunk: &mut Chunk, v: ConstValue, line: u32) {
        let idx = chunk.add_constant(v);
        chunk.write_op(Opcode::Const, line);
        chunk.write_u16(idx, line);
    }

    #[test]
    fn dispatch_runs_a_const_then_pop_program_clean() {
        // CONST 7 ; POP ; HALT -- runs to Halted with an empty stack.
        let mut chunk = Chunk::new();
        emit_const(&mut chunk, ConstValue::I64(7), 1);
        chunk.write_op(Opcode::Pop, 1);
        chunk.write_op(Opcode::Halt, 1);
        let mut vm = Vm::new(program_with(chunk), "x".to_string());
        vm.run().expect("a CONST/POP/HALT program runs clean");
        assert!(vm.stack.is_empty(), "POP must leave the stack empty");
    }

    #[test]
    fn dispatch_const_of_an_i64_pushes_a_pointer_to_a_heap_int() {
        // CONST 42 ; HALT -- the stack top is a pointer to a HeapObject::Int.
        let mut chunk = Chunk::new();
        emit_const(&mut chunk, ConstValue::I64(42), 1);
        chunk.write_op(Opcode::Halt, 1);
        let mut vm = Vm::new(program_with(chunk), "x".to_string());
        vm.run().expect("run");
        let top = *vm.stack.last().expect("the CONST left a value");
        assert_eq!(top.as_function(), None, "an i64 constant is not a function");
        let slot = top
            .as_pointer()
            .expect("an i64 constant must push a heap pointer");
        assert!(
            vm.heap.get(slot) == Some(&HeapObject::Int(42)),
            "the pointer must reach a heap Int(42)"
        );
    }

    #[test]
    fn dispatch_const_of_a_bool_pushes_a_tagged_scalar_not_a_pointer() {
        let mut chunk = Chunk::new();
        emit_const(&mut chunk, ConstValue::Bool(true), 1);
        chunk.write_op(Opcode::Halt, 1);
        let mut vm = Vm::new(program_with(chunk), "x".to_string());
        vm.run().expect("run");
        let top = *vm.stack.last().expect("value");
        assert_eq!(top.as_bool(), Some(true));
        assert_eq!(top.as_pointer(), None, "a bool is not a heap pointer");
    }

    #[test]
    fn dispatch_const_of_a_function_pushes_a_function_value_carrying_the_id() {
        // a ConstValue::Function becomes a tagged Value::function, no heap
        // object -- the locked representation 05-05's map/filter/reduce read.
        let mut chunk = Chunk::new();
        emit_const(&mut chunk, ConstValue::Function(13), 1);
        chunk.write_op(Opcode::Halt, 1);
        let mut vm = Vm::new(program_with(chunk), "x".to_string());
        vm.run().expect("run");
        let top = *vm.stack.last().expect("value");
        assert_eq!(top.as_function(), Some(13), "fn-id must round-trip");
        assert_eq!(top.as_pointer(), None, "a function is not a heap pointer");
    }

    #[test]
    fn dispatch_dup_duplicates_the_top_value() {
        // CONST true ; DUP ; HALT -- two equal values on the stack.
        let mut chunk = Chunk::new();
        emit_const(&mut chunk, ConstValue::Bool(true), 1);
        chunk.write_op(Opcode::Dup, 1);
        chunk.write_op(Opcode::Halt, 1);
        let mut vm = Vm::new(program_with(chunk), "x".to_string());
        vm.run().expect("run");
        assert_eq!(vm.stack.len(), 2, "DUP pushes a copy");
        assert!(vm.stack[0] == vm.stack[1], "the copy equals the original");
    }

    #[test]
    fn dispatch_set_local_then_get_local_round_trips_a_value() {
        // CONST b'\x05 ; SET_LOCAL 0 ; GET_LOCAL 0 ; HALT.
        let mut chunk = Chunk::new();
        emit_const(&mut chunk, ConstValue::Byte(5), 1);
        chunk.write_op(Opcode::SetLocal, 1);
        chunk.write_u16(0, 1);
        chunk.write_op(Opcode::GetLocal, 1);
        chunk.write_u16(0, 1);
        chunk.write_op(Opcode::Halt, 1);
        let mut vm = Vm::new(program_with(chunk), "x".to_string());
        vm.run().expect("run");
        let top = *vm.stack.last().expect("GET_LOCAL pushed a value");
        assert_eq!(top.as_byte(), Some(5), "the local round-trips");
    }

    #[test]
    fn dispatch_get_local_of_an_unset_slot_is_a_runtime_error_not_a_panic() {
        // GET_LOCAL 4 with no locals set -- a clean Runtime error.
        let mut chunk = Chunk::new();
        chunk.write_op(Opcode::GetLocal, 1);
        chunk.write_u16(4, 1);
        chunk.write_op(Opcode::Halt, 1);
        let mut vm = Vm::new(program_with(chunk), "x".to_string());
        match vm.run() {
            Err(QalaError::Runtime { message, .. }) => {
                assert!(message.contains("bad local slot"), "got: {message}");
            }
            other => panic!("expected a Runtime bad-local error, got {other:?}"),
        }
    }

    #[test]
    fn dispatch_set_global_then_get_global_round_trips_a_value() {
        let mut chunk = Chunk::new();
        emit_const(&mut chunk, ConstValue::Bool(false), 1);
        chunk.write_op(Opcode::SetGlobal, 1);
        chunk.write_u16(0, 1);
        chunk.write_op(Opcode::GetGlobal, 1);
        chunk.write_u16(0, 1);
        chunk.write_op(Opcode::Halt, 1);
        let mut vm = Vm::new(program_with(chunk), "x".to_string());
        vm.run().expect("run");
        let top = *vm.stack.last().expect("GET_GLOBAL pushed a value");
        assert_eq!(top.as_bool(), Some(false));
    }

    #[test]
    fn malformed_a_bad_opcode_byte_is_a_runtime_error_not_a_panic() {
        // byte 46 decodes to no opcode (the dense set ends at 45).
        let mut chunk = Chunk::new();
        chunk.code.push(46);
        chunk.source_lines.push(1);
        let mut vm = Vm::new(program_with(chunk), "x".to_string());
        match vm.run() {
            Err(QalaError::Runtime { message, .. }) => {
                assert!(message.contains("bad opcode byte"), "got: {message}");
            }
            other => panic!("expected a Runtime bad-opcode error, got {other:?}"),
        }
    }

    #[test]
    fn malformed_a_truncated_operand_is_a_runtime_error_not_a_panic() {
        // a CONST opcode with only one of its two operand bytes present.
        let mut chunk = Chunk::new();
        chunk.code.push(Opcode::Const as u8);
        chunk.code.push(0); // only one operand byte; CONST needs two
        chunk.source_lines.push(1);
        chunk.source_lines.push(1);
        let mut vm = Vm::new(program_with(chunk), "x".to_string());
        match vm.run() {
            Err(QalaError::Runtime { message, .. }) => {
                assert!(message.contains("truncated operand"), "got: {message}");
            }
            other => panic!("expected a Runtime truncated-operand error, got {other:?}"),
        }
    }

    #[test]
    fn malformed_a_bad_constant_index_is_a_runtime_error_not_a_panic() {
        // CONST 9 with an empty constant pool.
        let mut chunk = Chunk::new();
        chunk.write_op(Opcode::Const, 1);
        chunk.write_u16(9, 1);
        chunk.write_op(Opcode::Halt, 1);
        let mut vm = Vm::new(program_with(chunk), "x".to_string());
        match vm.run() {
            Err(QalaError::Runtime { message, .. }) => {
                assert!(message.contains("bad constant index"), "got: {message}");
            }
            other => panic!("expected a Runtime bad-constant error, got {other:?}"),
        }
    }

    #[test]
    fn the_remaining_unimplemented_seam_is_a_clean_runtime_stub_not_a_panic() {
        // every opcode now has a real handler -- the dispatch match has no
        // "not yet implemented" arm left. the one remaining deferred-feature
        // seam is `call_stdlib`: a CALL to a stdlib fn-id (>= STDLIB_FN_BASE)
        // routes there, and until 05-05 fills it in it must be a clean Runtime
        // "not yet implemented" error, never a panic. this guards the
        // stub-then-fill discipline against the last unimplemented path.
        let mut chunk = Chunk::new();
        chunk.write_op(Opcode::Call, 1);
        chunk.write_u16(STDLIB_FN_BASE, 1);
        chunk.code.push(0); // argc 0
        chunk.source_lines.push(1);
        chunk.write_op(Opcode::Return, 1);
        let mut vm = Vm::new(program_with(chunk), "x".to_string());
        match vm.run() {
            Err(QalaError::Runtime { message, .. }) => {
                assert!(message.contains("not yet implemented"), "got: {message}");
            }
            other => panic!("expected a Runtime not-implemented stub, got {other:?}"),
        }
    }

    #[test]
    fn dispatch_an_ip_past_the_end_of_the_chunk_is_a_runtime_error() {
        // an empty chunk -- ip 0 is already past the (zero-length) code.
        let mut vm = Vm::new(program_with(Chunk::new()), "x".to_string());
        match vm.run() {
            Err(QalaError::Runtime { message, .. }) => {
                assert!(
                    message.contains("instruction pointer past end"),
                    "got: {message}"
                );
            }
            other => panic!("expected a Runtime ip-past-end error, got {other:?}"),
        }
    }

    #[test]
    fn step_advances_ip_by_one_full_instruction_each_call() {
        // CONST 1 (3 bytes) ; POP (1 byte) ; HALT (1 byte).
        let mut chunk = Chunk::new();
        emit_const(&mut chunk, ConstValue::I64(1), 1);
        chunk.write_op(Opcode::Pop, 1);
        chunk.write_op(Opcode::Halt, 1);
        let mut vm = Vm::new(program_with(chunk), "x".to_string());
        // before the first step, ip is 0.
        assert_eq!(vm.frame().expect("frame").ip, 0);
        // step 1 executes CONST -- a 3-byte instruction, ip moves 0 -> 3.
        assert_eq!(vm.step().expect("step 1"), StepOutcome::Ran);
        assert_eq!(vm.frame().expect("frame").ip, 3, "CONST advances ip by 3");
        assert_eq!(vm.stack.len(), 1, "CONST pushed one value");
        // step 2 executes POP -- a 1-byte instruction, ip moves 3 -> 4.
        assert_eq!(vm.step().expect("step 2"), StepOutcome::Ran);
        assert_eq!(vm.frame().expect("frame").ip, 4, "POP advances ip by 1");
        assert_eq!(vm.stack.len(), 0, "POP cleared the stack");
        // step 3 executes HALT.
        assert_eq!(vm.step().expect("step 3"), StepOutcome::Halted);
    }

    #[test]
    fn run_and_step_share_dispatch_one_reaching_the_same_end_state() {
        // a small program run two ways: run() to completion, and step() in a
        // loop. both must leave the same value-stack state.
        let build = || {
            let mut chunk = Chunk::new();
            emit_const(&mut chunk, ConstValue::Bool(true), 1);
            emit_const(&mut chunk, ConstValue::Bool(false), 1);
            chunk.write_op(Opcode::Halt, 1);
            program_with(chunk)
        };
        let mut via_run = Vm::new(build(), "x".to_string());
        via_run.run().expect("run");

        let mut via_step = Vm::new(build(), "x".to_string());
        // step until Halted: the loop body is empty, the work is the step call.
        while via_step.step().expect("step") == StepOutcome::Ran {}
        assert_eq!(via_run.stack.len(), via_step.stack.len());
        assert!(
            via_run.stack.len() == 2
                && via_run.stack[0] == via_step.stack[0]
                && via_run.stack[1] == via_step.stack[1],
            "run and step must reach the same stack state"
        );
    }

    // ---- arithmetic / comparison / logic / jump opcodes ----

    /// run a chunk and return the VM so the caller can inspect the end state.
    fn run_chunk(chunk: Chunk, src: &str) -> Result<Vm, QalaError> {
        let mut vm = Vm::new(program_with(chunk), src.to_string());
        vm.run()?;
        Ok(vm)
    }

    /// the `i64` the stack top decodes to: a pointer to a heap `Int`.
    fn top_i64(vm: &Vm) -> i64 {
        let top = *vm.stack.last().expect("a value on the stack");
        let slot = top.as_pointer().expect("the result is a heap pointer");
        match vm.heap.get(slot) {
            Some(HeapObject::Int(n)) => *n,
            _ => panic!("the result pointer does not reach a heap Int"),
        }
    }

    /// build `CONST a ; CONST b ; <op> ; HALT` for two i64 literals.
    fn binary_i64_chunk(a: i64, b: i64, op: Opcode) -> Chunk {
        let mut chunk = Chunk::new();
        emit_const(&mut chunk, ConstValue::I64(a), 1);
        emit_const(&mut chunk, ConstValue::I64(b), 1);
        chunk.write_op(op, 1);
        chunk.write_op(Opcode::Halt, 1);
        chunk
    }

    #[test]
    fn arith_add_sub_mul_compute_correct_i64_results() {
        let add = run_chunk(binary_i64_chunk(20, 22, Opcode::Add), "x").expect("add");
        assert_eq!(top_i64(&add), 42);
        let sub = run_chunk(binary_i64_chunk(50, 8, Opcode::Sub), "x").expect("sub");
        assert_eq!(top_i64(&sub), 42);
        let mul = run_chunk(binary_i64_chunk(6, 7, Opcode::Mul), "x").expect("mul");
        assert_eq!(top_i64(&mul), 42);
    }

    #[test]
    fn arith_div_and_mod_compute_correct_i64_results() {
        // 85 / 2 == 42 (truncated toward zero); 85 % 2 == 1.
        let div = run_chunk(binary_i64_chunk(85, 2, Opcode::Div), "x").expect("div");
        assert_eq!(top_i64(&div), 42);
        let modr = run_chunk(binary_i64_chunk(85, 2, Opcode::Mod), "x").expect("mod");
        assert_eq!(top_i64(&modr), 1);
    }

    #[test]
    fn arith_neg_negates_an_i64() {
        // CONST 42 ; NEG ; HALT.
        let mut chunk = Chunk::new();
        emit_const(&mut chunk, ConstValue::I64(42), 1);
        chunk.write_op(Opcode::Neg, 1);
        chunk.write_op(Opcode::Halt, 1);
        let vm = run_chunk(chunk, "x").expect("neg");
        assert_eq!(top_i64(&vm), -42);
    }

    #[test]
    fn arith_add_overflow_is_a_runtime_error_not_a_wraparound() {
        // i64::MAX + 1 must error, not wrap to i64::MIN.
        let chunk = binary_i64_chunk(i64::MAX, 1, Opcode::Add);
        match run_chunk(chunk, "x") {
            Err(QalaError::Runtime { message, .. }) => {
                assert!(message.contains("integer overflow"), "got: {message}");
            }
            Err(other) => panic!("expected an overflow Runtime error, got {other:?}"),
            Ok(_) => panic!("expected an overflow Runtime error, the program ran clean"),
        }
    }

    #[test]
    fn arith_neg_of_i64_min_is_a_runtime_overflow() {
        // -i64::MIN does not fit in i64.
        let mut chunk = Chunk::new();
        emit_const(&mut chunk, ConstValue::I64(i64::MIN), 1);
        chunk.write_op(Opcode::Neg, 1);
        chunk.write_op(Opcode::Halt, 1);
        match run_chunk(chunk, "x") {
            Err(QalaError::Runtime { message, .. }) => {
                assert!(message.contains("integer overflow"), "got: {message}");
            }
            Err(other) => panic!("expected an overflow Runtime error, got {other:?}"),
            Ok(_) => panic!("expected an overflow Runtime error, the program ran clean"),
        }
    }

    #[test]
    fn div_by_zero_is_a_runtime_error_carrying_the_div_opcode_source_line() {
        // the DIV opcode sits on source line 4; the error span must cover it.
        let mut chunk = Chunk::new();
        emit_const(&mut chunk, ConstValue::I64(1), 1);
        emit_const(&mut chunk, ConstValue::I64(0), 1);
        chunk.write_op(Opcode::Div, 4);
        chunk.write_op(Opcode::Halt, 4);
        let src = "one\ntwo\nthree\nfour line here\nfive";
        match run_chunk(chunk, src) {
            Err(QalaError::Runtime { message, span }) => {
                assert!(message.contains("division by zero"), "got: {message}");
                assert_eq!(
                    span.slice(src),
                    "four line here",
                    "the error must point at the DIV's source line"
                );
            }
            Err(other) => panic!("expected a division-by-zero Runtime error, got {other:?}"),
            Ok(_) => panic!("expected a division-by-zero error, the program ran clean"),
        }
    }

    #[test]
    fn div_by_zero_mod_form_is_a_runtime_modulo_by_zero_error() {
        // MOD by a zero divisor reports "modulo by zero", on the MOD's line.
        let mut chunk = Chunk::new();
        emit_const(&mut chunk, ConstValue::I64(7), 1);
        emit_const(&mut chunk, ConstValue::I64(0), 1);
        chunk.write_op(Opcode::Mod, 2);
        chunk.write_op(Opcode::Halt, 2);
        let src = "first line\nsecond line is the mod";
        match run_chunk(chunk, src) {
            Err(QalaError::Runtime { message, span }) => {
                assert!(message.contains("modulo by zero"), "got: {message}");
                assert_eq!(span.slice(src), "second line is the mod");
            }
            Err(other) => panic!("expected a modulo-by-zero Runtime error, got {other:?}"),
            Ok(_) => panic!("expected a modulo-by-zero error, the program ran clean"),
        }
    }

    #[test]
    fn div_of_i64_min_by_negative_one_is_caught_as_a_runtime_error() {
        // i64::MIN / -1 overflows; checked_div returns None and the VM reports
        // it on the division-by-zero path (one None check covers both faults).
        let chunk = binary_i64_chunk(i64::MIN, -1, Opcode::Div);
        match run_chunk(chunk, "x") {
            Err(QalaError::Runtime { message, .. }) => {
                assert!(message.contains("division by zero"), "got: {message}");
            }
            Err(other) => panic!("expected a Runtime error for i64::MIN / -1, got {other:?}"),
            Ok(_) => panic!("expected a Runtime error for i64::MIN / -1, ran clean"),
        }
    }

    /// build `CONST a ; CONST b ; <op> ; HALT` for two f64 literals.
    fn binary_f64_chunk(a: f64, b: f64, op: Opcode) -> Chunk {
        let mut chunk = Chunk::new();
        emit_const(&mut chunk, ConstValue::F64(a), 1);
        emit_const(&mut chunk, ConstValue::F64(b), 1);
        chunk.write_op(op, 1);
        chunk.write_op(Opcode::Halt, 1);
        chunk
    }

    #[test]
    fn float_arith_add_sub_mul_neg_compute_correct_f64_results() {
        let add = run_chunk(binary_f64_chunk(1.5, 2.0, Opcode::FAdd), "x").expect("fadd");
        assert_eq!(add.stack.last().unwrap().as_f64(), Some(3.5));
        let sub = run_chunk(binary_f64_chunk(5.0, 1.5, Opcode::FSub), "x").expect("fsub");
        assert_eq!(sub.stack.last().unwrap().as_f64(), Some(3.5));
        let mul = run_chunk(binary_f64_chunk(2.0, 1.75, Opcode::FMul), "x").expect("fmul");
        assert_eq!(mul.stack.last().unwrap().as_f64(), Some(3.5));
        // FNEG flips the sign.
        let mut neg = Chunk::new();
        emit_const(&mut neg, ConstValue::F64(3.5), 1);
        neg.write_op(Opcode::FNeg, 1);
        neg.write_op(Opcode::Halt, 1);
        let negv = run_chunk(neg, "x").expect("fneg");
        assert_eq!(negv.stack.last().unwrap().as_f64(), Some(-3.5));
    }

    #[test]
    fn float_arith_div_by_zero_is_ieee754_inf_not_an_error() {
        // 1.0 / 0.0 == inf, with NO runtime error -- IEEE 754.
        let vm = run_chunk(binary_f64_chunk(1.0, 0.0, Opcode::FDiv), "x")
            .expect("float division by zero must not error");
        let top = vm.stack.last().unwrap().as_f64().expect("an f64 result");
        assert_eq!(top, f64::INFINITY, "1.0 / 0.0 is positive infinity");
    }

    #[test]
    fn float_arith_zero_over_zero_is_ieee754_nan_not_an_error() {
        // 0.0 / 0.0 == NaN, with NO runtime error -- IEEE 754.
        let vm = run_chunk(binary_f64_chunk(0.0, 0.0, Opcode::FDiv), "x")
            .expect("0.0 / 0.0 must not error");
        let top = vm.stack.last().unwrap().as_f64().expect("an f64 result");
        assert!(top.is_nan(), "0.0 / 0.0 is NaN");
    }

    #[test]
    fn compare_i64_eq_ne_lt_le_gt_ge_produce_correct_bools() {
        // helper: run CONST a ; CONST b ; <cmp> and read the bool result.
        let cmp = |a: i64, b: i64, op: Opcode| -> bool {
            let vm = run_chunk(binary_i64_chunk(a, b, op), "x").expect("compare");
            vm.stack.last().unwrap().as_bool().expect("a bool result")
        };
        assert!(cmp(3, 3, Opcode::Eq));
        assert!(!cmp(3, 4, Opcode::Eq));
        assert!(cmp(3, 4, Opcode::Ne));
        assert!(cmp(3, 4, Opcode::Lt));
        assert!(!cmp(4, 3, Opcode::Lt));
        assert!(cmp(3, 3, Opcode::Le));
        assert!(cmp(5, 4, Opcode::Gt));
        assert!(cmp(4, 4, Opcode::Ge));
        assert!(!cmp(3, 4, Opcode::Ge));
    }

    #[test]
    fn compare_str_eq_and_lt_compare_lexicographically() {
        // CONST "apple" ; CONST "banana" ; <cmp>.
        let cmp = |a: &str, b: &str, op: Opcode| -> bool {
            let mut chunk = Chunk::new();
            emit_const(&mut chunk, ConstValue::Str(a.to_string()), 1);
            emit_const(&mut chunk, ConstValue::Str(b.to_string()), 1);
            chunk.write_op(op, 1);
            chunk.write_op(Opcode::Halt, 1);
            let vm = run_chunk(chunk, "x").expect("str compare");
            vm.stack.last().unwrap().as_bool().expect("a bool result")
        };
        assert!(cmp("apple", "apple", Opcode::Eq));
        assert!(!cmp("apple", "banana", Opcode::Eq));
        assert!(cmp("apple", "banana", Opcode::Lt), "apple < banana");
        assert!(!cmp("banana", "apple", Opcode::Lt));
        assert!(cmp("apple", "banana", Opcode::Ne));
    }

    #[test]
    fn compare_bool_eq_and_ordering_follow_false_lt_true() {
        let cmp = |a: bool, b: bool, op: Opcode| -> bool {
            let mut chunk = Chunk::new();
            emit_const(&mut chunk, ConstValue::Bool(a), 1);
            emit_const(&mut chunk, ConstValue::Bool(b), 1);
            chunk.write_op(op, 1);
            chunk.write_op(Opcode::Halt, 1);
            let vm = run_chunk(chunk, "x").expect("bool compare");
            vm.stack.last().unwrap().as_bool().expect("a bool result")
        };
        assert!(cmp(true, true, Opcode::Eq));
        assert!(cmp(false, true, Opcode::Ne));
        // the documented bool ordering: false < true.
        assert!(cmp(false, true, Opcode::Lt));
        assert!(!cmp(true, false, Opcode::Lt));
    }

    #[test]
    fn compare_mismatched_operand_types_is_a_runtime_error_not_a_panic() {
        // CONST 1 (heap Int) ; CONST "x" (heap Str) ; EQ -- a kind mismatch.
        let mut chunk = Chunk::new();
        emit_const(&mut chunk, ConstValue::I64(1), 1);
        emit_const(&mut chunk, ConstValue::Str("x".to_string()), 1);
        chunk.write_op(Opcode::Eq, 1);
        chunk.write_op(Opcode::Halt, 1);
        match run_chunk(chunk, "x") {
            Err(QalaError::Runtime { message, .. }) => {
                assert!(message.contains("cannot compare"), "got: {message}");
            }
            Err(other) => panic!("expected a Runtime compare-mismatch error, got {other:?}"),
            Ok(_) => panic!("expected a compare-mismatch error, the program ran clean"),
        }
    }

    #[test]
    fn compare_f64_eq_lt_ge_follow_ieee754_including_nan() {
        let cmp = |a: f64, b: f64, op: Opcode| -> bool {
            let vm = run_chunk(binary_f64_chunk(a, b, op), "x").expect("f64 compare");
            vm.stack.last().unwrap().as_bool().expect("a bool result")
        };
        assert!(cmp(1.5, 1.5, Opcode::FEq));
        assert!(cmp(1.0, 2.0, Opcode::FLt));
        assert!(cmp(2.0, 2.0, Opcode::FGe));
        // IEEE 754: NaN compares unequal to itself and is unordered.
        assert!(!cmp(f64::NAN, f64::NAN, Opcode::FEq), "NaN == NaN is false");
        assert!(cmp(f64::NAN, f64::NAN, Opcode::FNe), "NaN != NaN is true");
        assert!(!cmp(f64::NAN, 1.0, Opcode::FLt), "NaN < x is false");
        assert!(!cmp(f64::NAN, 1.0, Opcode::FGt), "NaN > x is false");
    }

    #[test]
    fn logic_not_negates_a_bool() {
        let mut chunk = Chunk::new();
        emit_const(&mut chunk, ConstValue::Bool(true), 1);
        chunk.write_op(Opcode::Not, 1);
        chunk.write_op(Opcode::Halt, 1);
        let vm = run_chunk(chunk, "x").expect("not");
        assert_eq!(vm.stack.last().unwrap().as_bool(), Some(false));
    }

    #[test]
    fn jumps_jump_moves_ip_over_a_skipped_instruction() {
        // CONST true ; JUMP +4 (skip the next CONST) ; CONST false ; HALT.
        // a JUMP operand is relative to the byte after the operand. the JUMP
        // opcode is at byte 3, its operand at 4..5, fall-through at 6; the
        // CONST-false instruction is bytes 6..8; HALT is byte 9. to land on
        // HALT (byte 9) from fall-through 6 the offset is +3.
        let mut chunk = Chunk::new();
        emit_const(&mut chunk, ConstValue::Bool(true), 1); // bytes 0..2
        chunk.write_op(Opcode::Jump, 1); // byte 3
        chunk.write_i16(3, 1); // bytes 4..5, fall-through is 6
        emit_const(&mut chunk, ConstValue::Bool(false), 1); // bytes 6..8 (skipped)
        chunk.write_op(Opcode::Halt, 1); // byte 9
        let vm = run_chunk(chunk, "x").expect("jump");
        // only the `true` CONST ran; the `false` CONST was jumped over.
        assert_eq!(vm.stack.len(), 1, "the skipped CONST left nothing");
        assert_eq!(vm.stack[0].as_bool(), Some(true));
    }

    #[test]
    fn jumps_jump_if_false_branches_on_false_and_falls_through_on_true() {
        // build CONST <cond> ; JUMP_IF_FALSE +3 ; CONST 111 ; HALT.
        // on a false cond the CONST 111 is skipped; on true it runs.
        let outcome = |cond: bool| -> usize {
            let mut chunk = Chunk::new();
            emit_const(&mut chunk, ConstValue::Bool(cond), 1); // bytes 0..2
            chunk.write_op(Opcode::JumpIfFalse, 1); // byte 3
            chunk.write_i16(3, 1); // bytes 4..5, fall-through 6
            emit_const(&mut chunk, ConstValue::I64(111), 1); // bytes 6..8
            chunk.write_op(Opcode::Halt, 1); // byte 9
            let vm = run_chunk(chunk, "x").expect("jump_if_false");
            vm.stack.len()
        };
        // false: the cond is consumed, the CONST 111 is jumped over -> empty.
        assert_eq!(outcome(false), 0, "false branches past the CONST");
        // true: the cond is consumed, the CONST 111 runs -> one value.
        assert_eq!(outcome(true), 1, "true falls through to the CONST");
    }

    #[test]
    fn jumps_jump_if_true_branches_on_true_and_falls_through_on_false() {
        let outcome = |cond: bool| -> usize {
            let mut chunk = Chunk::new();
            emit_const(&mut chunk, ConstValue::Bool(cond), 1);
            chunk.write_op(Opcode::JumpIfTrue, 1);
            chunk.write_i16(3, 1);
            emit_const(&mut chunk, ConstValue::I64(222), 1);
            chunk.write_op(Opcode::Halt, 1);
            let vm = run_chunk(chunk, "x").expect("jump_if_true");
            vm.stack.len()
        };
        // true branches past the CONST; false falls through to it.
        assert_eq!(outcome(true), 0, "true branches past the CONST");
        assert_eq!(outcome(false), 1, "false falls through to the CONST");
    }

    #[test]
    fn jumps_a_backward_jump_lands_at_an_earlier_instruction() {
        // POP-free loop body would spin forever; instead verify a backward
        // offset lands correctly by stepping once. CONST true ; JUMP back to 0.
        // the JUMP at byte 3, fall-through 6; offset -6 targets byte 0.
        let mut chunk = Chunk::new();
        emit_const(&mut chunk, ConstValue::Bool(true), 1); // bytes 0..2
        chunk.write_op(Opcode::Jump, 1); // byte 3
        chunk.write_i16(-6, 1); // bytes 4..5; fall-through 6, target 0
        let mut vm = Vm::new(program_with(chunk), "x".to_string());
        vm.step().expect("step the CONST");
        assert_eq!(vm.frame().expect("frame").ip, 3, "ip after CONST is 3");
        vm.step().expect("step the JUMP");
        assert_eq!(vm.frame().expect("frame").ip, 0, "the backward JUMP lands at 0");
    }

    #[test]
    fn jumps_a_jump_target_outside_the_chunk_is_a_runtime_error_not_a_panic() {
        // JUMP with an offset that lands far past the end of the code.
        let mut chunk = Chunk::new();
        chunk.write_op(Opcode::Jump, 1); // byte 0
        chunk.write_i16(1000, 1); // fall-through 3, target 1003 -- out of range
        match run_chunk(chunk, "x") {
            Err(QalaError::Runtime { message, .. }) => {
                assert!(message.contains("jump target out of range"), "got: {message}");
            }
            Err(other) => panic!("expected a Runtime jump-out-of-range error, got {other:?}"),
            Ok(_) => panic!("expected a jump-out-of-range error, the program ran clean"),
        }
    }

    #[test]
    fn arith_add_of_non_integer_operands_is_a_runtime_type_error() {
        // CONST true ; CONST false ; ADD -- ADD expects two heap Ints.
        let mut chunk = Chunk::new();
        emit_const(&mut chunk, ConstValue::Bool(true), 1);
        emit_const(&mut chunk, ConstValue::Bool(false), 1);
        chunk.write_op(Opcode::Add, 1);
        chunk.write_op(Opcode::Halt, 1);
        match run_chunk(chunk, "x") {
            Err(QalaError::Runtime { message, .. }) => {
                assert!(message.contains("expected an integer"), "got: {message}");
            }
            Err(other) => panic!("expected a Runtime type error, got {other:?}"),
            Ok(_) => panic!("expected a Runtime type error, the program ran clean"),
        }
    }

    // ---- CALL / RETURN / the frame-depth cap ----

    use crate::lexer::Lexer;
    use crate::parser::Parser;
    use crate::typechecker::check_program;

    /// lex + parse + typecheck + compile a Qala source string into a runnable
    /// `Program`. panics on any pipeline error -- the compiled-program tests
    /// below all use known-good source.
    fn compile_qala(src: &str) -> Program {
        let tokens = Lexer::tokenize(src).expect("lex failed");
        let ast = Parser::parse(&tokens).expect("parse failed");
        let (typed, terrors, _) = check_program(&ast, src);
        assert!(terrors.is_empty(), "typecheck errors: {terrors:?}");
        crate::codegen::compile_program(&typed, src)
            .unwrap_or_else(|e| panic!("codegen errors: {e:?}"))
    }

    /// the `i64` a finished program left on top of its value stack -- a
    /// program's `RETURN` from `main` leaves the result there.
    fn program_result_i64(vm: &Vm) -> i64 {
        let top = *vm.stack.last().expect("the program left a result value");
        let slot = top.as_pointer().expect("the result is a heap pointer");
        match vm.heap.get(slot) {
            Some(HeapObject::Int(n)) => *n,
            _ => panic!("the result pointer does not reach a heap Int"),
        }
    }

    #[test]
    fn call_return_a_callee_returns_a_value_onto_the_callers_stack() {
        // a hand-built two-chunk program: main CALLs chunk 1, which returns the
        // i64 7; after the CALL main's stack top is that 7.
        //   chunk 0 (main): CALL fn 1 argc 0 ; RETURN
        //   chunk 1 (callee): CONST 7 ; RETURN
        let mut main = Chunk::new();
        main.write_op(Opcode::Call, 1);
        main.write_u16(1, 1); // fn-id 1
        main.code.push(0); // argc 0
        main.source_lines.push(1);
        main.write_op(Opcode::Return, 1);

        let mut callee = Chunk::new();
        emit_const(&mut callee, ConstValue::I64(7), 1);
        callee.write_op(Opcode::Return, 1);

        let mut p = Program::new();
        p.chunks.push(main);
        p.chunks.push(callee);
        p.fn_names.push("main".to_string());
        p.fn_names.push("callee".to_string());
        p.main_index = 0;

        let mut vm = Vm::new(p, "x".to_string());
        vm.run().expect("the call/return program runs clean");
        // main's RETURN re-pushed the callee's result; it is the program value.
        assert_eq!(program_result_i64(&vm), 7);
    }

    #[test]
    fn call_passes_arguments_into_the_callee_frames_local_slots() {
        // chunk 1 takes one argument and returns it via GET_LOCAL 0; main
        // passes 42 and the program result is 42 -- the arg reached slot 0.
        //   chunk 0 (main): CONST 42 ; CALL fn 1 argc 1 ; RETURN
        //   chunk 1 (id):   GET_LOCAL 0 ; RETURN
        let mut main = Chunk::new();
        emit_const(&mut main, ConstValue::I64(42), 1);
        main.write_op(Opcode::Call, 1);
        main.write_u16(1, 1);
        main.code.push(1); // argc 1
        main.source_lines.push(1);
        main.write_op(Opcode::Return, 1);

        let mut id = Chunk::new();
        id.write_op(Opcode::GetLocal, 1);
        id.write_u16(0, 1);
        id.write_op(Opcode::Return, 1);

        let mut p = Program::new();
        p.chunks.push(main);
        p.chunks.push(id);
        p.fn_names.push("main".to_string());
        p.fn_names.push("id".to_string());
        p.main_index = 0;

        let mut vm = Vm::new(p, "x".to_string());
        vm.run().expect("run");
        assert_eq!(program_result_i64(&vm), 42, "the argument reached local 0");
    }

    #[test]
    fn call_to_a_missing_function_is_a_runtime_error_not_a_panic() {
        // CALL fn 9 with only one chunk -- fn-id 9 has no chunk.
        let mut main = Chunk::new();
        main.write_op(Opcode::Call, 1);
        main.write_u16(9, 1);
        main.code.push(0);
        main.source_lines.push(1);
        main.write_op(Opcode::Return, 1);
        // run_chunk's Ok arm is a Vm, which has no Debug -- report in words.
        match run_chunk(main, "x") {
            Err(QalaError::Runtime { message, .. }) => {
                assert!(message.contains("missing function"), "got: {message}");
            }
            Err(other) => panic!("expected a Runtime missing-function error, got {other:?}"),
            Ok(_) => panic!("expected a Runtime missing-function error, the program ran clean"),
        }
    }

    #[test]
    fn call_to_a_stdlib_fn_id_seams_to_the_unimplemented_stdlib_stub() {
        // a CALL with fn-id >= STDLIB_FN_BASE routes to call_stdlib, which is
        // a clean Runtime stub until 05-05 fills it in.
        let mut main = Chunk::new();
        main.write_op(Opcode::Call, 1);
        main.write_u16(STDLIB_FN_BASE, 1);
        main.code.push(0);
        main.source_lines.push(1);
        main.write_op(Opcode::Return, 1);
        match run_chunk(main, "x") {
            Err(QalaError::Runtime { message, .. }) => {
                assert!(
                    message.contains("stdlib function") && message.contains("not yet implemented"),
                    "got: {message}"
                );
            }
            Err(other) => panic!("expected a Runtime stdlib-stub error, got {other:?}"),
            Ok(_) => panic!("expected a Runtime stdlib-stub error, the program ran clean"),
        }
    }

    #[test]
    fn fibonacci_recursion_computes_the_correct_numeric_result() {
        // the success-criterion smoke test: a recursive fib compiled from Qala
        // source. fib(10) == 55.
        let src = "\
fn fib(n: i64) -> i64 is pure {
    if n <= 1 { return n }
    return fib(n - 1) + fib(n - 2)
}

fn main() -> i64 is pure {
    return fib(10)
}
";
        let program = compile_qala(src);
        let mut vm = Vm::new(program, src.to_string());
        vm.run().expect("the fibonacci program runs clean");
        assert_eq!(program_result_i64(&vm), 55, "fib(10) must be 55");
    }

    #[test]
    fn deep_recursion_is_a_clean_stack_overflow_runtime_error_with_no_host_panic() {
        // the WASM-safety guarantee: an unbounded recursion hits MAX_FRAMES and
        // becomes a clean Runtime "stack overflow" -- NOT a host stack overflow
        // / abort. the recursion grows `frames`, not the host Rust stack,
        // because the VM is a `while` loop.
        let src = "\
fn r(n: i64) -> i64 is pure {
    return r(n + 1)
}

fn main() -> i64 is pure {
    return r(0)
}
";
        let program = compile_qala(src);
        let mut vm = Vm::new(program, src.to_string());
        match vm.run() {
            Err(QalaError::Runtime { message, .. }) => {
                assert!(
                    message.contains("stack overflow"),
                    "unbounded recursion must report a stack overflow, got: {message}"
                );
            }
            Err(other) => panic!("expected a Runtime stack-overflow error, got {other:?}"),
            Ok(_) => panic!("unbounded recursion must error, the program ran clean"),
        }
        // reaching this line proves the test process did not panic / abort:
        // the recursion was caught by the frame cap, the host stack stayed flat.
    }

    #[test]
    fn call_function_value_runs_a_user_callback_and_returns_its_result() {
        // call_function_value is the stdlib's re-entry point. chunk 1 doubles
        // its argument; calling it via a function Value with arg 21 yields 42.
        //   chunk 0 (main): RETURN  (a placeholder entry chunk)
        //   chunk 1 (double): GET_LOCAL 0 ; GET_LOCAL 0 ; ADD ; RETURN
        let mut main = Chunk::new();
        emit_const(&mut main, ConstValue::I64(0), 1);
        main.write_op(Opcode::Return, 1);

        let mut double = Chunk::new();
        double.write_op(Opcode::GetLocal, 1);
        double.write_u16(0, 1);
        double.write_op(Opcode::GetLocal, 1);
        double.write_u16(0, 1);
        double.write_op(Opcode::Add, 1);
        double.write_op(Opcode::Return, 1);

        let mut p = Program::new();
        p.chunks.push(main);
        p.chunks.push(double);
        p.fn_names.push("main".to_string());
        p.fn_names.push("double".to_string());
        p.main_index = 0;

        let mut vm = Vm::new(p, "x".to_string());
        // build the argument 21 as a heap Int, then re-enter chunk 1.
        let arg_slot = vm.heap.alloc(HeapObject::Int(21)).expect("alloc arg");
        let result = vm
            .call_function_value(Value::function(1), &[Value::pointer(arg_slot)])
            .expect("the callback runs and returns");
        let result_slot = result.as_pointer().expect("a heap-Int result");
        assert!(
            vm.heap.get(result_slot) == Some(&HeapObject::Int(42)),
            "double(21) must be 42"
        );
    }

    #[test]
    fn call_function_value_of_a_non_function_is_a_runtime_error() {
        // a bool is not callable.
        let mut vm = Vm::new(program_with(Chunk::new()), "x".to_string());
        // the Ok arm is a Value, which has no Debug -- report in words.
        match vm.call_function_value(Value::bool(true), &[]) {
            Err(QalaError::Runtime { message, .. }) => {
                assert!(message.contains("not callable"), "got: {message}");
            }
            Err(other) => panic!("expected a Runtime not-callable error, got {other:?}"),
            Ok(_) => panic!("expected a Runtime not-callable error, got a value"),
        }
    }

    // ---- MAKE_ARRAY / MAKE_TUPLE / MAKE_STRUCT / MAKE_ENUM_VARIANT /
    // INDEX / FIELD / LEN ----

    #[test]
    fn make_array_builds_a_heap_array_then_index_and_len_read_it() {
        // CONST 10 ; CONST 20 ; CONST 30 ; MAKE_ARRAY 3 ; ...
        // then INDEX 1 -> 20, and (on a second build) LEN -> 3.
        // element order: MAKE_ARRAY pops in reverse, so the array is [10,20,30].
        let mut chunk = Chunk::new();
        emit_const(&mut chunk, ConstValue::I64(10), 1);
        emit_const(&mut chunk, ConstValue::I64(20), 1);
        emit_const(&mut chunk, ConstValue::I64(30), 1);
        chunk.write_op(Opcode::MakeArray, 1);
        chunk.write_u16(3, 1);
        // INDEX needs the array then the index on top: DUP the array, push 1.
        chunk.write_op(Opcode::Dup, 1);
        emit_const(&mut chunk, ConstValue::I64(1), 1);
        chunk.write_op(Opcode::Index, 1);
        // stack now: [array_ptr, 20]. LEN the array under the 20 is awkward;
        // instead just assert the INDEX result and the array's heap shape.
        chunk.write_op(Opcode::Halt, 1);
        let vm = run_chunk(chunk, "x").expect("make_array");
        // top is the INDEX result -- element 1 is 20.
        assert_eq!(top_i64(&vm), 20, "INDEX 1 of [10,20,30] is 20");
        // under it, the array pointer reaches a 3-element heap Array.
        let array_ptr = vm.stack[vm.stack.len() - 2];
        let slot = array_ptr.as_pointer().expect("an array pointer");
        match vm.heap.get(slot) {
            Some(HeapObject::Array(items)) => assert_eq!(items.len(), 3),
            _ => panic!("MAKE_ARRAY must build a heap Array"),
        }
    }

    #[test]
    fn len_of_a_built_array_pushes_the_element_count() {
        // CONST 1 ; CONST 2 ; MAKE_ARRAY 2 ; LEN ; HALT -> 2.
        let mut chunk = Chunk::new();
        emit_const(&mut chunk, ConstValue::I64(1), 1);
        emit_const(&mut chunk, ConstValue::I64(2), 1);
        chunk.write_op(Opcode::MakeArray, 1);
        chunk.write_u16(2, 1);
        chunk.write_op(Opcode::Len, 1);
        chunk.write_op(Opcode::Halt, 1);
        let vm = run_chunk(chunk, "x").expect("len");
        assert_eq!(top_i64(&vm), 2, "LEN of a 2-element array is 2");
    }

    #[test]
    fn len_of_a_string_counts_unicode_scalar_values() {
        // LEN of a 5-char string is 5.
        let mut chunk = Chunk::new();
        emit_const(&mut chunk, ConstValue::Str("hello".to_string()), 1);
        chunk.write_op(Opcode::Len, 1);
        chunk.write_op(Opcode::Halt, 1);
        let vm = run_chunk(chunk, "x").expect("len str");
        assert_eq!(top_i64(&vm), 5, "LEN of \"hello\" is 5");
    }

    #[test]
    fn make_tuple_builds_a_distinct_heap_tuple_object() {
        // MAKE_TUPLE produces a HeapObject::Tuple, not an Array -- the
        // distinction lets 05-05's type_of tell a tuple from an array.
        let mut chunk = Chunk::new();
        emit_const(&mut chunk, ConstValue::I64(7), 1);
        emit_const(&mut chunk, ConstValue::Bool(true), 1);
        chunk.write_op(Opcode::MakeTuple, 1);
        chunk.write_u16(2, 1);
        chunk.write_op(Opcode::Halt, 1);
        let vm = run_chunk(chunk, "x").expect("make_tuple");
        let top = *vm.stack.last().expect("a tuple pointer");
        let slot = top.as_pointer().expect("a heap pointer");
        match vm.heap.get(slot) {
            Some(HeapObject::Tuple(items)) => {
                assert_eq!(items.len(), 2, "the tuple has two elements");
                // element 0 is the first pushed (7), element 1 the bool.
                assert_eq!(items[1].as_bool(), Some(true));
            }
            _ => panic!("MAKE_TUPLE must build a heap Tuple, not an Array"),
        }
    }

    #[test]
    fn index_out_of_bounds_is_a_runtime_error_carrying_the_index_length_and_line() {
        // a 2-element array indexed at 5: the error names the index AND the
        // length, and its span covers the INDEX opcode's source line.
        let mut chunk = Chunk::new();
        emit_const(&mut chunk, ConstValue::I64(100), 1);
        emit_const(&mut chunk, ConstValue::I64(200), 1);
        chunk.write_op(Opcode::MakeArray, 1);
        chunk.write_u16(2, 1);
        emit_const(&mut chunk, ConstValue::I64(5), 1);
        chunk.write_op(Opcode::Index, 4); // the INDEX is on source line 4
        chunk.write_op(Opcode::Halt, 4);
        let src = "one\ntwo\nthree\nthe index line\nfive";
        match run_chunk(chunk, src) {
            Err(QalaError::Runtime { message, span }) => {
                assert!(message.contains('5'), "the index is named: {message}");
                assert!(message.contains('2'), "the length is named: {message}");
                assert!(message.contains("out of bounds"), "got: {message}");
                assert_eq!(
                    span.slice(src),
                    "the index line",
                    "the error must point at the INDEX's source line"
                );
            }
            Err(other) => panic!("expected an out-of-bounds Runtime error, got {other:?}"),
            Ok(_) => panic!("expected an out-of-bounds error, the program ran clean"),
        }
    }

    #[test]
    fn index_of_a_negative_index_is_a_runtime_error_not_a_panic() {
        // a negative index must not wrap or panic -- it is a clean error.
        let mut chunk = Chunk::new();
        emit_const(&mut chunk, ConstValue::I64(1), 1);
        chunk.write_op(Opcode::MakeArray, 1);
        chunk.write_u16(1, 1);
        emit_const(&mut chunk, ConstValue::I64(-1), 1);
        chunk.write_op(Opcode::Index, 1);
        chunk.write_op(Opcode::Halt, 1);
        match run_chunk(chunk, "x") {
            Err(QalaError::Runtime { message, .. }) => {
                assert!(message.contains("out of bounds"), "got: {message}");
            }
            Err(other) => panic!("expected an out-of-bounds Runtime error, got {other:?}"),
            Ok(_) => panic!("expected an out-of-bounds error, the program ran clean"),
        }
    }

    #[test]
    fn make_struct_labels_the_struct_with_its_declared_name_and_field_can_read_it() {
        // a Program with one struct "Point" of 2 fields. build a Point{1,2},
        // confirm the heap struct's type_name is "Point", then FIELD 1 -> 2.
        let mut chunk = Chunk::new();
        emit_const(&mut chunk, ConstValue::I64(1), 1);
        emit_const(&mut chunk, ConstValue::I64(2), 1);
        chunk.write_op(Opcode::MakeStruct, 1);
        chunk.write_u16(0, 1); // struct id 0
        // DUP so one copy is FIELD-accessed and one stays for the heap check.
        chunk.write_op(Opcode::Dup, 1);
        chunk.write_op(Opcode::Field, 1);
        chunk.write_u16(1, 1); // field index 1
        chunk.write_op(Opcode::Halt, 1);

        let mut p = program_with(chunk);
        p.structs.push(crate::chunk::StructInfo {
            name: "Point".to_string(),
            field_count: 2,
        });
        let mut vm = Vm::new(p, "x".to_string());
        vm.run().expect("make_struct");
        // top is FIELD 1's result -- the second field, 2.
        assert_eq!(top_i64(&vm), 2, "FIELD 1 of Point{{1,2}} is 2");
        // under it, the struct pointer reaches a Struct labelled "Point".
        let struct_ptr = vm.stack[vm.stack.len() - 2];
        let slot = struct_ptr.as_pointer().expect("a struct pointer");
        match vm.heap.get(slot) {
            Some(HeapObject::Struct { type_name, fields }) => {
                assert_eq!(type_name, "Point", "the struct carries its declared name");
                assert_eq!(fields.len(), 2);
            }
            _ => panic!("MAKE_STRUCT must build a heap Struct"),
        }
    }

    #[test]
    fn make_struct_with_a_bad_struct_id_is_a_runtime_error_not_a_panic() {
        // MAKE_STRUCT id 9 with an empty structs table.
        let mut chunk = Chunk::new();
        chunk.write_op(Opcode::MakeStruct, 1);
        chunk.write_u16(9, 1);
        chunk.write_op(Opcode::Halt, 1);
        match run_chunk(chunk, "x") {
            Err(QalaError::Runtime { message, .. }) => {
                assert!(message.contains("bad struct id"), "got: {message}");
            }
            Err(other) => panic!("expected a Runtime bad-struct-id error, got {other:?}"),
            Ok(_) => panic!("expected a Runtime bad-struct-id error, the program ran clean"),
        }
    }

    #[test]
    fn make_enum_variant_builds_a_variant_with_its_enum_and_variant_names() {
        // a Program whose variant id 0 is (Shape, Circle). build Circle(5),
        // confirm the heap object's type_name / variant / payload.
        let mut chunk = Chunk::new();
        emit_const(&mut chunk, ConstValue::I64(5), 1);
        chunk.write_op(Opcode::MakeEnumVariant, 1);
        chunk.write_u16(0, 1); // variant id 0
        chunk.code.push(1); // payload count 1
        chunk.source_lines.push(1);
        chunk.write_op(Opcode::Halt, 1);

        let mut p = program_with(chunk);
        p.enum_variant_names
            .push(("Shape".to_string(), "Circle".to_string()));
        let mut vm = Vm::new(p, "x".to_string());
        vm.run().expect("make_enum_variant");
        let top = *vm.stack.last().expect("a variant pointer");
        let slot = top.as_pointer().expect("a heap pointer");
        match vm.heap.get(slot) {
            Some(HeapObject::EnumVariant {
                type_name,
                variant,
                payload,
            }) => {
                assert_eq!(type_name, "Shape", "the enum name");
                assert_eq!(variant, "Circle", "the variant name");
                assert_eq!(payload.len(), 1, "Circle carries one payload value");
            }
            _ => panic!("MAKE_ENUM_VARIANT must build a heap EnumVariant"),
        }
    }

    #[test]
    fn make_enum_variant_with_a_bad_variant_id_is_a_runtime_error_not_a_panic() {
        let mut chunk = Chunk::new();
        chunk.write_op(Opcode::MakeEnumVariant, 1);
        chunk.write_u16(9, 1);
        chunk.code.push(0);
        chunk.source_lines.push(1);
        chunk.write_op(Opcode::Halt, 1);
        match run_chunk(chunk, "x") {
            Err(QalaError::Runtime { message, .. }) => {
                assert!(message.contains("bad variant id"), "got: {message}");
            }
            Err(other) => panic!("expected a Runtime bad-variant-id error, got {other:?}"),
            Ok(_) => panic!("expected a Runtime bad-variant-id error, the program ran clean"),
        }
    }

    #[test]
    fn field_of_a_non_struct_is_a_runtime_error_not_a_panic() {
        // FIELD on an array pointer -- the pointer does not reach a struct.
        let mut chunk = Chunk::new();
        emit_const(&mut chunk, ConstValue::I64(1), 1);
        chunk.write_op(Opcode::MakeArray, 1);
        chunk.write_u16(1, 1);
        chunk.write_op(Opcode::Field, 1);
        chunk.write_u16(0, 1);
        chunk.write_op(Opcode::Halt, 1);
        match run_chunk(chunk, "x") {
            Err(QalaError::Runtime { message, .. }) => {
                assert!(message.contains("expected a struct"), "got: {message}");
            }
            Err(other) => panic!("expected a Runtime not-a-struct error, got {other:?}"),
            Ok(_) => panic!("expected a Runtime not-a-struct error, the program ran clean"),
        }
    }

    // ---- TO_STR / CONCAT_N / MATCH_VARIANT + value_to_string + defer ----

    /// the `String` the stack top decodes to: a pointer to a heap `Str`.
    fn top_str(vm: &Vm) -> String {
        let top = *vm.stack.last().expect("a value on the stack");
        let slot = top.as_pointer().expect("a heap pointer");
        match vm.heap.get(slot) {
            Some(HeapObject::Str(s)) => s.clone(),
            _ => panic!("the result pointer does not reach a heap Str"),
        }
    }

    #[test]
    fn to_str_of_an_i64_renders_the_decimal_form() {
        let mut chunk = Chunk::new();
        emit_const(&mut chunk, ConstValue::I64(-42), 1);
        chunk.write_op(Opcode::ToStr, 1);
        chunk.write_op(Opcode::Halt, 1);
        let vm = run_chunk(chunk, "x").expect("to_str i64");
        assert_eq!(top_str(&vm), "-42");
    }

    #[test]
    fn to_str_of_a_bool_renders_the_lowercase_keyword() {
        let mut chunk = Chunk::new();
        emit_const(&mut chunk, ConstValue::Bool(true), 1);
        chunk.write_op(Opcode::ToStr, 1);
        chunk.write_op(Opcode::Halt, 1);
        let vm = run_chunk(chunk, "x").expect("to_str bool");
        assert_eq!(top_str(&vm), "true");
    }

    #[test]
    fn to_str_of_a_float_hand_spells_nan_and_infinities() {
        // value_to_string of each non-finite f64 matches ConstValue's Display.
        let to_str = |x: f64| -> String {
            let mut chunk = Chunk::new();
            emit_const(&mut chunk, ConstValue::F64(x), 1);
            chunk.write_op(Opcode::ToStr, 1);
            chunk.write_op(Opcode::Halt, 1);
            top_str(&run_chunk(chunk, "x").expect("to_str f64"))
        };
        assert_eq!(to_str(f64::NAN), "NaN", "a NaN renders as NaN");
        assert_eq!(to_str(f64::INFINITY), "inf");
        assert_eq!(to_str(f64::NEG_INFINITY), "-inf");
        assert_eq!(to_str(3.5), "3.5", "a finite float uses the default form");
    }

    #[test]
    fn value_to_string_renders_each_runtime_value_kind_with_its_locked_spelling() {
        // exercise value_to_string directly across every kind.
        let mut vm = Vm::new(program_with(Chunk::new()), "x".to_string());
        // tagged scalars.
        assert_eq!(vm.value_to_string(Value::bool(false)), "false");
        assert_eq!(vm.value_to_string(Value::byte(65)), "65", "a byte is decimal");
        assert_eq!(vm.value_to_string(Value::void()), "()");
        assert_eq!(vm.value_to_string(Value::function(7)), "fn#7");
        assert_eq!(vm.value_to_string(Value::from_f64(2.0)), "2");
        // heap objects.
        let int_slot = vm.heap.alloc(HeapObject::Int(-3)).expect("alloc");
        assert_eq!(vm.value_to_string(Value::pointer(int_slot)), "-3");
        let str_slot = vm
            .heap
            .alloc(HeapObject::Str("raw text".to_string()))
            .expect("alloc");
        assert_eq!(
            vm.value_to_string(Value::pointer(str_slot)),
            "raw text",
            "a string renders unquoted"
        );
        let arr_slot = vm
            .heap
            .alloc(HeapObject::Array(vec![
                Value::pointer(int_slot),
                Value::bool(true),
            ]))
            .expect("alloc");
        assert_eq!(vm.value_to_string(Value::pointer(arr_slot)), "[-3, true]");
        let variant_slot = vm
            .heap
            .alloc(HeapObject::EnumVariant {
                type_name: "Shape".to_string(),
                variant: "Circle".to_string(),
                payload: vec![Value::pointer(int_slot)],
            })
            .expect("alloc");
        assert_eq!(
            vm.value_to_string(Value::pointer(variant_slot)),
            "Shape::Circle(-3)"
        );
        let bare_variant = vm
            .heap
            .alloc(HeapObject::EnumVariant {
                type_name: "Color".to_string(),
                variant: "Red".to_string(),
                payload: Vec::new(),
            })
            .expect("alloc");
        assert_eq!(
            vm.value_to_string(Value::pointer(bare_variant)),
            "Color::Red",
            "a payload-less variant renders without parentheses"
        );
    }

    #[test]
    fn concat_n_joins_several_values_into_one_string() {
        // CONST "a=" ; CONST 1 ; TO_STR ; CONST "!" ; CONCAT_N 3 -> "a=1!".
        let mut chunk = Chunk::new();
        emit_const(&mut chunk, ConstValue::Str("a=".to_string()), 1);
        emit_const(&mut chunk, ConstValue::I64(1), 1);
        chunk.write_op(Opcode::ToStr, 1);
        emit_const(&mut chunk, ConstValue::Str("!".to_string()), 1);
        chunk.write_op(Opcode::ConcatN, 1);
        chunk.write_u16(3, 1);
        chunk.write_op(Opcode::Halt, 1);
        let vm = run_chunk(chunk, "x").expect("concat_n");
        assert_eq!(top_str(&vm), "a=1!", "CONCAT_N joins in source order");
    }

    #[test]
    fn concat_n_of_an_interpolated_nan_float_renders_the_word_nan() {
        // the VM-04 guarantee at the CONCAT_N layer: a string interpolation of
        // a NaN float renders "NaN", not a numeric token. CONST "v=" ;
        // CONST NaN ; CONCAT_N 2 -> "v=NaN" (CONCAT_N stringifies each part).
        let mut chunk = Chunk::new();
        emit_const(&mut chunk, ConstValue::Str("v=".to_string()), 1);
        emit_const(&mut chunk, ConstValue::F64(f64::NAN), 1);
        chunk.write_op(Opcode::ConcatN, 1);
        chunk.write_u16(2, 1);
        chunk.write_op(Opcode::Halt, 1);
        let vm = run_chunk(chunk, "x").expect("concat_n nan");
        assert_eq!(top_str(&vm), "v=NaN", "an interpolated NaN renders as NaN");
    }

    #[test]
    fn match_variant_on_a_match_destructures_the_payload_onto_the_stack() {
        // a Program whose variant id 0 is (Shape, Circle). build Circle(99),
        // then MATCH_VARIANT 0: a hit consumes the scrutinee and leaves the
        // payload 99 on the stack.
        let mut chunk = Chunk::new();
        emit_const(&mut chunk, ConstValue::I64(99), 1);
        chunk.write_op(Opcode::MakeEnumVariant, 1);
        chunk.write_u16(0, 1);
        chunk.code.push(1); // payload count 1
        chunk.source_lines.push(1);
        // MATCH_VARIANT 0, miss-offset +0 (irrelevant on a hit).
        chunk.write_op(Opcode::MatchVariant, 1);
        chunk.write_u16(0, 1);
        chunk.write_i16(0, 1);
        chunk.write_op(Opcode::Halt, 1);

        let mut p = program_with(chunk);
        p.enum_variant_names
            .push(("Shape".to_string(), "Circle".to_string()));
        let mut vm = Vm::new(p, "x".to_string());
        vm.run().expect("match_variant hit");
        // a hit consumed the scrutinee and left exactly the payload.
        assert_eq!(vm.stack.len(), 1, "only the payload remains");
        assert_eq!(top_i64(&vm), 99, "the destructured payload is 99");
    }

    #[test]
    fn match_variant_on_a_miss_leaves_the_scrutinee_and_branches_by_the_offset() {
        // build Circle(1), then MATCH_VARIANT against the (Shape, Square)
        // variant id -- a miss. the scrutinee stays on the stack, ip branches
        // past a skipped CONST by the i16 offset.
        let mut chunk = Chunk::new();
        emit_const(&mut chunk, ConstValue::I64(1), 1); // bytes 0..2
        chunk.write_op(Opcode::MakeEnumVariant, 1); // byte 3
        chunk.write_u16(0, 1); // bytes 4..5: variant id 0 (Circle)
        chunk.code.push(1); // byte 6: payload count
        chunk.source_lines.push(1);
        // MATCH_VARIANT at byte 7, operand bytes 8..11, fall-through 12.
        chunk.write_op(Opcode::MatchVariant, 1); // byte 7
        chunk.write_u16(1, 1); // bytes 8..9: variant id 1 (Square) -- a miss
        chunk.write_i16(3, 1); // bytes 10..11: miss offset +3 -> target 15
        emit_const(&mut chunk, ConstValue::I64(777), 1); // bytes 12..14 (skipped)
        chunk.write_op(Opcode::Halt, 1); // byte 15
        let mut p = program_with(chunk);
        p.enum_variant_names
            .push(("Shape".to_string(), "Circle".to_string()));
        p.enum_variant_names
            .push(("Shape".to_string(), "Square".to_string()));
        let mut vm = Vm::new(p, "x".to_string());
        vm.run().expect("match_variant miss");
        // a miss left the scrutinee and branched past the CONST 777, so the
        // stack holds exactly the one scrutinee pointer.
        assert_eq!(vm.stack.len(), 1, "the scrutinee stays, the CONST was skipped");
        let top = *vm.stack.last().expect("the scrutinee");
        let slot = top.as_pointer().expect("the scrutinee is an enum pointer");
        assert!(
            matches!(vm.heap.get(slot), Some(HeapObject::EnumVariant { .. })),
            "the value left on the stack is the original scrutinee"
        );
    }

    #[test]
    fn match_variant_of_a_non_enum_scrutinee_is_a_runtime_error_not_a_panic() {
        // MATCH_VARIANT on an i64 -- not an enum value.
        let mut chunk = Chunk::new();
        emit_const(&mut chunk, ConstValue::I64(5), 1);
        chunk.write_op(Opcode::MatchVariant, 1);
        chunk.write_u16(0, 1);
        chunk.write_i16(0, 1);
        chunk.write_op(Opcode::Halt, 1);
        let mut p = program_with(chunk);
        p.enum_variant_names
            .push(("E".to_string(), "V".to_string()));
        let mut vm = Vm::new(p, "x".to_string());
        match vm.run() {
            Err(QalaError::Runtime { message, .. }) => {
                assert!(message.contains("not an enum"), "got: {message}");
            }
            other => panic!("expected a Runtime non-enum-scrutinee error, got {other:?}"),
        }
    }

    #[test]
    fn defer_bytecode_runs_in_lifo_order_at_scope_exit() {
        // VM-03: Phase 4 codegen splices defer bytecode inline at scope exit,
        // emitted in REVERSE (LIFO) order. this test compiles a Qala program
        // with two defers, runs it through the VM, and confirms both that the
        // VM executes the spliced defer bytecode correctly (the program runs
        // clean to completion -- no stack imbalance from the extra calls) AND
        // that the two defer call sites appear LIFO in the emitted bytecode.
        let src = "\
fn first() -> i64 is pure { return 1 }
fn second() -> i64 is pure { return 2 }
fn run() -> i64 is pure {
    defer first()
    defer second()
    return 0
}
fn main() -> i64 is pure { return run() }
";
        let program = compile_qala(src);
        // `first` and `second` are the first two functions compiled -- dense
        // ids 0 and 1. the `run` chunk's defers must CALL id 1 (second) BEFORE
        // id 0 (first): the source order is first-then-second, LIFO reverses it.
        let run_idx = program
            .fn_names
            .iter()
            .position(|n| n == "run")
            .expect("the run function exists");
        let code = &program.chunks[run_idx].code;
        // collect the fn-ids of every CALL in the run chunk, in byte order.
        let mut call_ids: Vec<u16> = Vec::new();
        let mut ip = 0;
        while ip < code.len() {
            let Some(op) = Opcode::from_u8(code[ip]) else { break };
            if op == Opcode::Call {
                call_ids.push(u16::from_le_bytes([code[ip + 1], code[ip + 2]]));
            }
            ip += 1 + op.operand_bytes() as usize;
        }
        let first_id = program.fn_names.iter().position(|n| n == "first").unwrap() as u16;
        let second_id = program.fn_names.iter().position(|n| n == "second").unwrap() as u16;
        assert_eq!(
            call_ids,
            vec![second_id, first_id],
            "the run chunk must CALL second's defer before first's defer -- LIFO"
        );
        // and the VM runs the whole program (defer bytecode included) clean.
        let mut vm = Vm::new(program, src.to_string());
        vm.run().expect("the program with two defers runs clean");
        assert_eq!(program_result_i64(&vm), 0, "run returns 0; the defers ran for effect");
    }
}
