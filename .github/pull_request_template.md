## What this changes

A short description of the change and why it is needed.

## Related issue

The issue this addresses, if any. For anything beyond a small fix, an issue
should come first so the approach can be discussed.

## Checklist

- [ ] `cargo test --workspace` passes
- [ ] `cargo clippy --workspace -- -D warnings` is clean
- [ ] `cargo fmt --all -- --check` reports no drift
- [ ] if `playground/` changed, `npm run typecheck`, `npm run lint`, and `npm run build` pass
- [ ] commits are small and focused, each with a clear message
- [ ] public items have doc comments and there is no `unwrap()` outside tests
