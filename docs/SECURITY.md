# Security Policy

## Supported versions

Qala is an actively developed project. Security fixes are applied to the latest
released version. Older versions are not maintained.

| Version | Supported |
|---------|-----------|
| 2.x     | yes       |
| 1.x     | no        |

## Reporting a vulnerability

Please do not open a public issue for a security vulnerability.

Report it privately through GitHub. On the repository's **Security** tab, choose
**Report a vulnerability** to open a private advisory visible only to the
maintainers. Include:

- a description of the issue and why it is a security concern,
- a minimal Qala program or set of steps that reproduces it,
- the version or commit you observed it on.

You can expect an acknowledgment within a few days. Once the issue is confirmed,
a fix is prepared and released, and the advisory is published with credit to the
reporter unless you ask to remain anonymous.

## Scope

Qala compiles and runs untrusted Qala source. The bytecode VM is a sandboxed
interpreter with bounded stack, heap, and call depth, and its file built-ins are
mocked rather than touching the real filesystem. The web playground runs the
compiler as a WebAssembly module in the browser.

Reports that are in scope include a way for a Qala program to escape the VM
sandbox, a crash or panic in the compiler or VM reachable from Qala source, and
any issue in the playground's handling of the WebAssembly module. A Qala program
that simply runs slowly or uses a lot of memory within the VM's existing bounds
is expected behavior, not a vulnerability.
