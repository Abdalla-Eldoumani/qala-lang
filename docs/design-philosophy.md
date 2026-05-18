# the idea behind Qala

In Arabic, qala (قال) means to say, to declare, to state what is. When you say
something with qala, you are not guessing or hoping. You are stating a fact.

Qala the language works the same way. The code you write is a declaration of
what the program does, and the compiler holds you to it. When a function says
`is pure`, it is pure. When you write `defer close(f)`, the file closes. There
is no gap between what the code says and what the program does. Closing that
gap is the whole point.

## the gap in other languages

In most languages, a function signature tells you a little and hides a lot. It
tells you the argument types and the return type. It does not tell you whether
the function prints, opens a file, allocates, or crashes. You read the body, or
the docs, or you find out at runtime.

That gap is where bugs live. A math function quietly logs to a global. A getter
opens a network connection. Cleanup code is three returns away from the resource
it frees, and one of the returns forgets it. None of this shows up in the
signature, so none of it shows up when you read the call.

Qala's answer is to put the truth in the signature and let the compiler check
it.

## purity as the default, effects tracked

Haskell showed that a language can track side effects in the type system, and
that pure code is easier to reason about because it cannot surprise you. The
cost was monads, which are a real barrier to learning the language.

Qala keeps the tracking and drops the monads. A function declares its effect in
plain words: `is pure`, `is io`, `is alloc`, `is panic`. A pure function cannot
call an impure one, and the compiler enforces that. You get the guarantee
Haskell gives you, stated in words a newcomer can read.

## compile-time guarantees

Rust showed that a compiler can catch whole categories of bugs before the
program runs, if the language is built for it. The cost was the borrow checker,
which is powerful and also hard.

Qala takes the goal and picks lighter tools for it. Resource cleanup is handled
by `defer`, which runs at scope exit on every path, and the compiler warns when
a resource has no matching cleanup. Matching is exhaustive, so a missing enum
case is a compile error, not a runtime fall-through. The checks happen at
compile time, where a mistake is cheap to fix.

## accessibility

A guarantee you cannot understand is not much of a guarantee. Qala's third aim
is that its safety is readable.

Effects are English words after the signature, not a type-class hierarchy.
Cleanup is `defer close(f)`, written next to the `open`, not a lifetime the
borrow checker reasons about. Errors are `Result` values you handle with `?` and
`or`, not exceptions that travel up an invisible path. Each of these is a
deliberate trade: a simpler tool that a programmer can hold in their head,
chosen over a more powerful one that they cannot.

## reading a Qala program

The result is that you can read a Qala function signature and know what the
function does to the world. The effect is right there. The resources it manages
are visible. The cases it handles are all accounted for, because the compiler
counted them.

That is what qala means. The code says what is. The compiler makes sure it is
telling the truth.
