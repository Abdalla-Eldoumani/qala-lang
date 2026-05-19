#!/usr/bin/env bash
# arm64 backend end-to-end verification: emit, assemble, link, run.
#
# this is a STANDALONE, explicitly-invoked check -- it is NOT part of
# `cargo test`. the `cargo test` suite asserts the emitted assembly TEXT and
# stays toolchain-independent; this script proves the emitted assembly really
# assembles and runs by driving the CPSC 355 hosted-Linux pipeline:
#
#   qala build --target arm64 prog.qala  ->  prog.s   (m4 source)
#   m4 prog.s > prog.gas.s                            (expand the m4 aliases)
#   aarch64-linux-gnu-gcc -static prog.gas.s -o prog   (assemble + link libc)
#   qemu-aarch64 prog                                  (run under emulation)
#
# the AArch64 toolchain (m4, aarch64-linux-gnu-gcc, qemu-aarch64) lives in WSL
# Ubuntu; this script shells into WSL for the assemble/link/run steps. it
# checks two programs: a printing integer demo, and the bundled fibonacci.qala
# (the ARM-06 round-trip target).
#
# usage:  bash crates/qala-compiler/tests/arm64-roundtrip/verify.sh
# exit:   0 if every program's output matches, non-zero on the first mismatch.

set -u

# the repo root, resolved from this script's own location, so the script runs
# correctly regardless of the caller's working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

PASS=0
FAIL=0

# run a WSL command inside one `wsl --exec bash -c`, with the temp directory's
# /mnt/c path supplied as the inner script's `$1`.
#
# the path is passed as a positional argument, not interpolated into the
# script text -- routing it through `wsl`'s own argv keeps it clear of the
# quote-mangling a path embedded in the `bash -c` string would suffer.
# `--exec` runs the command directly (no WSL login-shell wrapper), which is
# what makes the trailing positional arguments reach `bash -c`. MSYS_NO_PATHCONV
# stops Git Bash rewriting the /mnt/c argument on the way in.
#   $1 = the inner shell script (reads the path as `$1`)
#   $2 = the /mnt/c temp directory path
wsl_run() {
    MSYS_NO_PATHCONV=1 wsl -d Ubuntu --exec bash -c "$1" _ "$2"
}

# translate an MSYS path (the form `mktemp -d` returns under Git Bash) into the
# /mnt/c path WSL uses for the same file.
#
# the translation is done entirely on the Git Bash side -- cygpath turns the
# MSYS path into a Windows path `C:\dir\file`, then the drive letter is
# lowercased and the backslashes flipped to give `/mnt/c/dir/file`. passing a
# finished forward-slash /mnt path into the `wsl bash -c` command keeps every
# backslash off the WSL command line, where a double-quoted shell string would
# otherwise eat them.
wsl_path() {
    local win
    if command -v cygpath > /dev/null 2>&1; then
        win="$(cygpath -w "$1")"
    else
        # `/c/Users/...` -> `C:\Users\...` as a fallback when cygpath is absent.
        win="$(printf '%s' "$1" | sed -E 's#^/([a-zA-Z])/#\1:/#; s#/#\\#g')"
    fi
    # `C:\dir\file` -> `/mnt/c/dir/file`: the drive letter (char 1) lowercased,
    # then `/mnt/<drive>` followed by the rest with its backslashes flipped.
    local drive rest
    drive="$(printf '%s' "$win" | cut -c1 | tr 'A-Z' 'a-z')"
    rest="$(printf '%s' "$win" | cut -c3- | sed 's/\\/\//g')"
    printf '/mnt/%s%s' "$drive" "$rest"
}

# build, assemble, link, run one qala program and compare its output.
#   $1 = a label for the report
#   $2 = the qala source path (Windows path)
#   $3 = the expected program output (exact, newline-separated)
check_program() {
    local label="$1"
    local src="$2"
    local expected="$3"

    echo "=== $label ==="

    # a per-run temp directory. `tmp` is the MSYS path the script uses for its
    # own file IO; `tmp_mnt` is the same directory as a /mnt/c path, which is
    # how WSL sees it.
    local tmp
    tmp="$(mktemp -d)"
    local tmp_mnt
    tmp_mnt="$(wsl_path "$tmp")"
    local asm="$tmp/prog.s"

    # 1. emit the AArch64 assembly (m4 source) via the qala CLI.
    if ! "$QALA_BIN" build --target arm64 "$src" -o "$asm" > /dev/null 2>&1; then
        echo "FAIL: qala build --target arm64 failed for $label"
        rm -rf "$tmp"
        FAIL=$((FAIL + 1))
        return 1
    fi

    # 2-4. m4 -> gcc -> qemu, all inside WSL, against the /mnt/c temp path.
    # the temp path arrives as the inner script's `$1`; its stdout is captured.
    local actual
    actual="$(wsl_run '
        set -e
        d="$1"
        m4 "$d/prog.s" > "$d/prog.gas.s"
        aarch64-linux-gnu-gcc -static "$d/prog.gas.s" -o "$d/prog"
        qemu-aarch64 "$d/prog"
    ' "$tmp_mnt" 2>"$tmp/err.txt")"
    local status=$?

    if [ $status -ne 0 ]; then
        echo "FAIL: the assemble/link/run pipeline failed for $label"
        echo "----- pipeline stderr -----"
        cat "$tmp/err.txt"
        echo "---------------------------"
        rm -rf "$tmp"
        FAIL=$((FAIL + 1))
        return 1
    fi

    # 5. compare. normalise CRLF in case WSL emitted carriage returns.
    actual="${actual//$'\r'/}"
    if [ "$actual" = "$expected" ]; then
        echo "PASS: $label"
        echo "  output:"
        echo "$actual" | sed 's/^/    /'
        PASS=$((PASS + 1))
        rm -rf "$tmp"
        return 0
    else
        echo "FAIL: $label output mismatch"
        echo "  expected:"
        echo "$expected" | sed 's/^/    /'
        echo "  actual:"
        echo "$actual" | sed 's/^/    /'
        FAIL=$((FAIL + 1))
        rm -rf "$tmp"
        return 1
    fi
}

echo "building the qala CLI..."
if ! ( cd "$REPO_ROOT" && cargo build -p qala-cli > /dev/null 2>&1 ); then
    echo "FATAL: cargo build -p qala-cli failed"
    exit 1
fi
QALA_BIN="$REPO_ROOT/target/debug/qala"
if [ ! -e "$QALA_BIN" ] && [ -e "$QALA_BIN.exe" ]; then
    QALA_BIN="$QALA_BIN.exe"
fi
echo "qala CLI: $QALA_BIN"
echo ""

# program 1: a printing integer demo -- a string literal, an interpolation
# with i64 holes (one a call), and a loop printing each iteration.
check_program \
    "print_demo.qala" \
    "$SCRIPT_DIR/print_demo.qala" \
    "arm64 round-trip
add(20, 22) = 42
count 1
count 2
count 3"

echo ""

# program 2: the bundled fibonacci.qala -- the ARM-06 round-trip target. it
# prints fib(0)..fib(14) through an i64-holed interpolation.
check_program \
    "fibonacci.qala (ARM-06)" \
    "$REPO_ROOT/playground/public/examples/fibonacci.qala" \
    "fib(0) = 0
fib(1) = 1
fib(2) = 1
fib(3) = 2
fib(4) = 3
fib(5) = 5
fib(6) = 8
fib(7) = 13
fib(8) = 21
fib(9) = 34
fib(10) = 55
fib(11) = 89
fib(12) = 144
fib(13) = 233
fib(14) = 377"

echo ""
echo "=== results: $PASS passed, $FAIL failed ==="
exit $FAIL
