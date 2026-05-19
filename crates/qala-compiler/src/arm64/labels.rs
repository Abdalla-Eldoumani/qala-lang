//! the monotonic label generator for the ARM64 backend.
//!
//! every branch target the backend emits -- the `else` arm of an `if`, a loop
//! test, a short-circuit `&&` fall-through -- needs a name that is unique
//! across the whole emitted file. [`LabelGen`] is a single `u32` counter:
//! [`LabelGen::fresh`] formats `.L<prefix>_<n>` and increments. the counter is
//! per-`compile_arm64`-call, so two compiles of the same source produce the
//! same labels in the same order -- the determinism the snapshot tests require.
//!
//! the `.L` prefix marks a *local* assembler label: GAS keeps it out of the
//! symbol table. function names (`main`, `add3`) are global labels and carry
//! no `.L` -- they are not produced here.

/// a monotonic label-name generator. holds one `u32` counter starting at 0;
/// each [`LabelGen::fresh`] call consumes the next value.
///
/// crate-private: only the `arm64` module needs unique branch labels.
pub(crate) struct LabelGen {
    /// the next label number to hand out. starts at 0, increments on every
    /// `fresh` call.
    counter: u32,
}

impl LabelGen {
    /// construct a fresh generator with the counter at 0.
    pub(crate) fn new() -> Self {
        LabelGen { counter: 0 }
    }

    /// produce the next unique label: `.L<prefix>_<n>`, then advance the counter.
    ///
    /// `prefix` names the construct (`if_else`, `while_test`, `and_done`, ...);
    /// `n` is the monotonic counter value. successive calls with the same
    /// prefix differ only in `n`, so every label in the file is distinct.
    pub(crate) fn fresh(&mut self, prefix: &str) -> String {
        let label = format!(".L{prefix}_{}", self.counter);
        self.counter += 1;
        label
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_numbers_labels_monotonically_from_zero() {
        // two successive fresh() calls with the same prefix differ only in the
        // counter suffix, starting at 0.
        let mut lg = LabelGen::new();
        assert_eq!(lg.fresh("if_end"), ".Lif_end_0");
        assert_eq!(lg.fresh("if_end"), ".Lif_end_1");
    }

    #[test]
    fn the_counter_is_shared_across_prefixes() {
        // the counter is global to the generator, not per-prefix: a fresh()
        // with a new prefix still consumes the next number.
        let mut lg = LabelGen::new();
        assert_eq!(lg.fresh("if_else"), ".Lif_else_0");
        assert_eq!(lg.fresh("while_test"), ".Lwhile_test_1");
        assert_eq!(lg.fresh("if_else"), ".Lif_else_2");
    }

    #[test]
    fn a_fresh_generator_restarts_at_zero() {
        // the determinism guarantee: a new LabelGen always begins at 0, so two
        // compiles of the same program emit byte-identical labels.
        let mut a = LabelGen::new();
        let _ = a.fresh("and_done");
        let _ = a.fresh("and_done");
        let mut b = LabelGen::new();
        assert_eq!(b.fresh("and_done"), ".Land_done_0");
    }
}
