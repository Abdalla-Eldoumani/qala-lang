---
name: Bug report
about: Report something in the compiler, the CLI, or the playground that does not work
title: ""
labels: bug
---

## What happened

A clear description of the problem.

## A program that reproduces it

The smallest Qala program that shows the bug. If the issue is in the playground
or the CLI rather than in a specific program, describe the steps instead.

```
fn main() is io {
  println("...")
}
```

## What you expected

What the program, the CLI, or the playground should have done.

## What happened instead

The actual output, the diagnostic, or the panic you saw. Paste the exact text.

## Environment

- Qala version or commit:
- Where it happened: the `qala` CLI, the playground, or the ARM64 backend
- Operating system:
