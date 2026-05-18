//! the effect lattice the type checker infers each function against, stored
//! as a `u8` bitfield.
//!
//! Qala tracks four effects -- `Pure`, `Io`, `Alloc`, `Panic` -- but only the
//! last three are stored explicitly: an empty bitfield IS pure. the lattice
//! is the powerset of `{Io, Alloc, Panic}` ordered by subset; lattice height
//! is 4 (the chain `pure -> +io -> +alloc -> +panic`). that height is what
//! makes Plan 4's effect fixed-point loop terminate -- a monotonic ascent on
//! a finite lattice of height `h` converges in at most `h - 1` rounds
//! (Davey & Priestley), so the iteration is guaranteed to settle in <= 3
//! rounds regardless of call-graph shape.
//!
//! the u8 repr is chosen because every operation the type checker needs
//! collapses to a single machine op:
//! - `union` is a bitwise `|`
//! - `is_subset_of` is one mask: `(self & !other) == 0`
//! - equality is a one-byte compare
//!
//! diagnostics need a deterministic textual form -- the renderer must produce
//! byte-identical output for the same set every time. [`EffectSet::display`]
//! enumerates the flags in fixed (Io, Alloc, Panic) order and renders an
//! empty set as `"pure"`. that wording is the contract Plan 5's
//! `EffectViolation` diagnostic wording depends on.
//!
//! this file depends only on `core::fmt`. no inference logic lives here --
//! the type checker builds an [`EffectSet`] from each function's body in a
//! later plan; this module just provides the data and its operations.

/// the `Io` flag's bit in [`EffectSet`]'s u8 repr.
///
/// the specific bit value is the one implementation detail to lock here -- the
/// type checker's fixed-point loop only cares that ascent is monotonic, but the
/// renderer's tests pin the bit layout so a regression in `union` (e.g.
/// accidentally swapping the io and alloc constants) shows up as a clear
/// `display()` test failure.
const IO: u8 = 0b001;

/// the `Alloc` flag's bit in [`EffectSet`]'s u8 repr.
const ALLOC: u8 = 0b010;

/// the `Panic` flag's bit in [`EffectSet`]'s u8 repr.
const PANIC: u8 = 0b100;

/// the set of effects a function performs.
///
/// an empty set IS pure -- `Pure` is implicit, not a stored flag. the u8 repr
/// makes `union` a single `|` op, `is_subset_of` a single mask, and equality a
/// one-byte compare. instances are immutable; every operation returns a fresh
/// [`EffectSet`] rather than mutating in place. `Copy` matters because the
/// type checker stores an [`EffectSet`] on every typed function and reads it
/// at every call site; one byte is cheaper to copy than to reference.
///
/// `Default` derives `EffectSet(0)`, which equals [`EffectSet::pure`] -- the
/// type checker initialises every unannotated function at the bottom of the
/// lattice and unions upward during fixed-point ascent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct EffectSet(u8);

impl EffectSet {
    /// the empty set: no flags set, the lattice bottom.
    ///
    /// equivalent to `is pure` annotated functions. constructing a function's
    /// initial [`EffectSet`] for fixed-point ascent starts here.
    pub fn pure() -> Self {
        Self(0)
    }

    /// every flag set: the lattice top.
    ///
    /// used as the upper bound for the unannotated-caller fallback: when the
    /// type checker calls a function it cannot resolve a signature for, it
    /// treats the call as producing `full()` so an annotated caller's check
    /// fails closed rather than silently accepting.
    pub fn full() -> Self {
        Self(IO | ALLOC | PANIC)
    }

    /// the singleton set `{Io}`.
    pub fn io() -> Self {
        Self(IO)
    }

    /// the singleton set `{Alloc}`.
    pub fn alloc() -> Self {
        Self(ALLOC)
    }

    /// the singleton set `{Panic}`.
    pub fn panic() -> Self {
        Self(PANIC)
    }

    /// is the `Io` flag set?
    pub fn has_io(self) -> bool {
        self.0 & IO != 0
    }

    /// is the `Alloc` flag set?
    pub fn has_alloc(self) -> bool {
        self.0 & ALLOC != 0
    }

    /// is the `Panic` flag set?
    pub fn has_panic(self) -> bool {
        self.0 & PANIC != 0
    }

    /// is this the empty set?
    ///
    /// true iff no flags are set. an `is pure`-annotated function's effect
    /// set is exactly this case.
    pub fn is_pure(self) -> bool {
        self.0 == 0
    }

    /// the lattice join: the union of `self`'s flags and `other`'s flags.
    ///
    /// idempotent (`a.union(a) == a`), commutative (`a.union(b) == b.union(a)`),
    /// and associative. these properties are what guarantee Plan 4's
    /// fixed-point loop converges monotonically.
    pub fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// is every flag set in `self` also set in `other`?
    ///
    /// reflexive (`a.is_subset_of(a) == true`), antisymmetric (`a ⊆ b` and
    /// `b ⊆ a` implies `a == b`), and transitive. the type checker's
    /// annotation check is `inferred.is_subset_of(annotated)` -- the inferred
    /// effect of the body must not exceed what the annotation allows.
    pub fn is_subset_of(self, other: Self) -> bool {
        (self.0 & !other.0) == 0
    }

    /// the canonical comma-joined rendering for diagnostics.
    ///
    /// flags appear in fixed `(Io, Alloc, Panic)` order regardless of how the
    /// set was constructed, so `io().union(alloc())` and
    /// `alloc().union(io())` both render `"io, alloc"`. an empty set renders
    /// `"pure"`. this is the determinism contract the renderer's
    /// byte-identical-output tests depend on.
    pub fn display(&self) -> String {
        if self.is_pure() {
            return "pure".to_string();
        }
        let mut parts: Vec<&'static str> = Vec::with_capacity(3);
        if self.has_io() {
            parts.push("io");
        }
        if self.has_alloc() {
            parts.push("alloc");
        }
        if self.has_panic() {
            parts.push("panic");
        }
        parts.join(", ")
    }
}

impl core::fmt::Display for EffectSet {
    /// delegate to [`EffectSet::display`] so `format!("{e}")` produces the
    /// canonical comma-joined form. lets `EffectViolation` messages
    /// interpolate effect sets without an explicit `.display()` call at every
    /// site.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&self.display())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pure_has_no_flags_set() {
        let p = EffectSet::pure();
        assert!(p.is_pure());
        assert!(!p.has_io());
        assert!(!p.has_alloc());
        assert!(!p.has_panic());
    }

    #[test]
    fn io_singleton_carries_only_io() {
        let e = EffectSet::io();
        assert!(e.has_io());
        assert!(!e.has_alloc());
        assert!(!e.has_panic());
        assert!(!e.is_pure());
    }

    #[test]
    fn alloc_singleton_carries_only_alloc() {
        let e = EffectSet::alloc();
        assert!(!e.has_io());
        assert!(e.has_alloc());
        assert!(!e.has_panic());
        assert!(!e.is_pure());
    }

    #[test]
    fn panic_singleton_carries_only_panic() {
        let e = EffectSet::panic();
        assert!(!e.has_io());
        assert!(!e.has_alloc());
        assert!(e.has_panic());
        assert!(!e.is_pure());
    }

    #[test]
    fn full_has_every_flag_set() {
        let e = EffectSet::full();
        assert!(e.has_io());
        assert!(e.has_alloc());
        assert!(e.has_panic());
        assert!(!e.is_pure());
    }

    #[test]
    fn union_with_pure_returns_the_other_set() {
        // pure() is the lattice bottom; unioning with it is the identity.
        assert_eq!(
            EffectSet::pure().union(EffectSet::io()),
            EffectSet::io(),
        );
        assert_eq!(
            EffectSet::io().union(EffectSet::pure()),
            EffectSet::io(),
        );
        assert_eq!(
            EffectSet::pure().union(EffectSet::full()),
            EffectSet::full(),
        );
    }

    #[test]
    fn union_combines_distinct_flags() {
        let io_alloc = EffectSet::io().union(EffectSet::alloc());
        assert!(io_alloc.has_io());
        assert!(io_alloc.has_alloc());
        assert!(!io_alloc.has_panic());
    }

    #[test]
    fn union_is_commutative() {
        // a sample of pairs: union should give the same set regardless of
        // argument order.
        let pairs = [
            (EffectSet::pure(), EffectSet::io()),
            (EffectSet::io(), EffectSet::alloc()),
            (EffectSet::alloc(), EffectSet::panic()),
            (EffectSet::io(), EffectSet::full()),
            (EffectSet::pure(), EffectSet::full()),
        ];
        for (a, b) in pairs {
            assert_eq!(a.union(b), b.union(a), "union not commutative on {a:?} and {b:?}");
        }
    }

    #[test]
    fn union_is_idempotent() {
        // a.union(a) == a -- the lattice property the fixed-point loop relies on.
        let samples = [
            EffectSet::pure(),
            EffectSet::io(),
            EffectSet::alloc(),
            EffectSet::panic(),
            EffectSet::io().union(EffectSet::alloc()),
            EffectSet::full(),
        ];
        for s in samples {
            assert_eq!(s.union(s), s, "union not idempotent on {s:?}");
        }
    }

    #[test]
    fn is_subset_of_matches_lattice_subset_semantics() {
        // pure is a subset of everything; full is a subset only of itself.
        assert!(EffectSet::pure().is_subset_of(EffectSet::full()));
        assert!(!EffectSet::full().is_subset_of(EffectSet::pure()));
        // io ⊆ {io, alloc}
        let io_alloc = EffectSet::io().union(EffectSet::alloc());
        assert!(EffectSet::io().is_subset_of(io_alloc));
        // {io, alloc} ⊄ {io}
        assert!(!io_alloc.is_subset_of(EffectSet::io()));
        // disjoint singletons are not subsets of each other.
        assert!(!EffectSet::io().is_subset_of(EffectSet::alloc()));
        assert!(!EffectSet::alloc().is_subset_of(EffectSet::io()));
    }

    #[test]
    fn is_subset_of_is_reflexive() {
        // a ⊆ a for every a in the sample.
        let samples = [
            EffectSet::pure(),
            EffectSet::io(),
            EffectSet::alloc(),
            EffectSet::panic(),
            EffectSet::io().union(EffectSet::alloc()),
            EffectSet::io().union(EffectSet::panic()),
            EffectSet::alloc().union(EffectSet::panic()),
            EffectSet::full(),
        ];
        for s in samples {
            assert!(s.is_subset_of(s), "subset reflexivity failure on {s:?}");
        }
    }

    #[test]
    fn display_renders_each_set_in_fixed_io_alloc_panic_order() {
        // the determinism contract: enumerate flags in (Io, Alloc, Panic)
        // order regardless of which order they were unioned in.
        assert_eq!(EffectSet::pure().display(), "pure");
        assert_eq!(EffectSet::io().display(), "io");
        assert_eq!(EffectSet::alloc().display(), "alloc");
        assert_eq!(EffectSet::panic().display(), "panic");
        // io appears before alloc in the rendered string, even when alloc was
        // the left-hand argument to union.
        assert_eq!(
            EffectSet::io().union(EffectSet::alloc()).display(),
            "io, alloc",
        );
        assert_eq!(
            EffectSet::alloc().union(EffectSet::io()).display(),
            "io, alloc",
        );
        assert_eq!(
            EffectSet::io().union(EffectSet::panic()).display(),
            "io, panic",
        );
        assert_eq!(
            EffectSet::alloc().union(EffectSet::panic()).display(),
            "alloc, panic",
        );
        assert_eq!(EffectSet::full().display(), "io, alloc, panic");
    }

    #[test]
    fn display_trait_matches_display_method() {
        // the `Display` impl must produce the same bytes as `.display()` so
        // `format!("{e}")` is interchangeable with `e.display()` at call
        // sites like the `EffectViolation` message construction.
        let cases = [
            EffectSet::pure(),
            EffectSet::io(),
            EffectSet::alloc(),
            EffectSet::panic(),
            EffectSet::io().union(EffectSet::alloc()),
            EffectSet::full(),
        ];
        for e in cases {
            assert_eq!(format!("{e}"), e.display(), "Display mismatch on {e:?}");
        }
    }

    #[test]
    fn equality_distinguishes_distinct_sets() {
        assert_eq!(EffectSet::pure(), EffectSet::pure());
        assert_ne!(EffectSet::io(), EffectSet::alloc());
        assert_ne!(EffectSet::pure(), EffectSet::io());
        // commutativity-of-union implies equality regardless of construction
        // order.
        assert_eq!(
            EffectSet::io().union(EffectSet::alloc()),
            EffectSet::alloc().union(EffectSet::io()),
        );
        // EffectSet::default() equals pure().
        assert_eq!(EffectSet::default(), EffectSet::pure());
    }

    #[test]
    fn monotonic_ascent_converges_in_at_most_three_rounds() {
        // the typechecker's effect fixed-point relies on this property: starting
        // from pure(), unioning with full() reaches full() in one round. starting
        // from a single flag, reaching full() takes at most 2 more rounds. the
        // lattice has height 4 (chains pure -> +io -> +alloc -> +panic at most),
        // so the loop terminates regardless of input order.
        let mut e = EffectSet::pure();
        let target = EffectSet::full();
        let mut rounds = 0;
        while e != target {
            e = e.union(target);
            rounds += 1;
            assert!(rounds <= 3, "should converge within 3 rounds; took {rounds}");
        }
        assert!(rounds <= 3);

        // also verify the worst-case chain: ascend one flag per round.
        let chain = [
            EffectSet::io(),
            EffectSet::alloc(),
            EffectSet::panic(),
        ];
        let mut e = EffectSet::pure();
        let mut rounds = 0;
        for step in chain {
            e = e.union(step);
            rounds += 1;
        }
        assert_eq!(e, EffectSet::full());
        // three rounds to ascend the full chain: pure -> +io -> +alloc -> +panic.
        assert_eq!(rounds, 3);
    }
}
