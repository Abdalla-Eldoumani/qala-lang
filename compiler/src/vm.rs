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
            // ---- arithmetic / comparison / logic / jump ----
            // a later commit fills these in; until then they are clean stubs.
            Opcode::Add
            | Opcode::Sub
            | Opcode::Mul
            | Opcode::Div
            | Opcode::Mod
            | Opcode::Neg
            | Opcode::FAdd
            | Opcode::FSub
            | Opcode::FMul
            | Opcode::FDiv
            | Opcode::FNeg
            | Opcode::Eq
            | Opcode::Ne
            | Opcode::Lt
            | Opcode::Le
            | Opcode::Gt
            | Opcode::Ge
            | Opcode::FEq
            | Opcode::FNe
            | Opcode::FLt
            | Opcode::FLe
            | Opcode::FGt
            | Opcode::FGe
            | Opcode::Not
            | Opcode::Jump
            | Opcode::JumpIfFalse
            | Opcode::JumpIfTrue => {
                return Err(self.runtime_err(&format!("opcode {op:?} not yet implemented")));
            }
            // ---- not implemented by this plan (calls, heap construction,
            // access, strings, match) -- a later plan replaces these stubs ----
            Opcode::Call
            | Opcode::Return
            | Opcode::MakeArray
            | Opcode::MakeTuple
            | Opcode::MakeStruct
            | Opcode::MakeEnumVariant
            | Opcode::Index
            | Opcode::Field
            | Opcode::Len
            | Opcode::ToStr
            | Opcode::ConcatN
            | Opcode::MatchVariant => {
                return Err(self.runtime_err(&format!("opcode {op:?} not yet implemented")));
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
    fn malformed_an_unimplemented_opcode_is_a_clean_runtime_stub_not_a_panic() {
        // ADD is stubbed by this plan -- it must be a clean Runtime error.
        let mut chunk = Chunk::new();
        chunk.write_op(Opcode::Add, 1);
        chunk.write_op(Opcode::Halt, 1);
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
        loop {
            match via_step.step().expect("step") {
                StepOutcome::Ran => continue,
                StepOutcome::Halted => break,
            }
        }
        assert_eq!(via_run.stack.len(), via_step.stack.len());
        assert!(
            via_run.stack.len() == 2
                && via_run.stack[0] == via_step.stack[0]
                && via_run.stack[1] == via_step.stack[1],
            "run and step must reach the same stack state"
        );
    }
}
