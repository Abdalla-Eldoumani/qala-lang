use assert_cmd::Command;
use predicates::prelude::*;
use std::io::Write;

#[test]
fn build_writes_a_bytecode_artifact() {
    let dir = tempfile::tempdir().unwrap();
    let src_path = dir.path().join("prog.qala");
    std::fs::write(&src_path, "fn main() is io {\n  println(\"hi\")\n}").unwrap();
    let out_path = dir.path().join("prog.qbc");

    Command::cargo_bin("qala")
        .unwrap()
        .args(["build"])
        .arg(&src_path)
        .arg("-o")
        .arg(&out_path)
        .assert()
        .success();

    // the artifact exists and is the non-empty disassembly listing.
    let listing = std::fs::read_to_string(&out_path).unwrap();
    assert!(!listing.is_empty());
    assert!(listing.contains("chunk")); // disassemble() emits "== chunk: ... =="
}

#[test]
fn build_target_arm64_writes_an_assembly_file() {
    // `build --target arm64` on a pure-integer program compiles through the
    // ARM64 backend and writes a .s assembly artifact, exit 0. the program is
    // pure integer code -- a `println` would route through an unsupported
    // stdlib call and the backend would (correctly) reject it.
    let dir = tempfile::tempdir().unwrap();
    let src_path = dir.path().join("prog.qala");
    std::fs::write(&src_path, "fn main() {\n    let x = 2 + 3\n}\n").unwrap();
    let out_path = dir.path().join("prog.s");

    Command::cargo_bin("qala")
        .unwrap()
        .args(["build"])
        .arg(&src_path)
        .args(["--target", "arm64"])
        .arg("-o")
        .arg(&out_path)
        .assert()
        .success();

    // the artifact exists and is the non-empty emitted assembly.
    let assembly = std::fs::read_to_string(&out_path).unwrap();
    assert!(!assembly.is_empty());
    assert!(assembly.contains(".text")); // the GAS section header
    assert!(assembly.contains("main:")); // the function label
}

#[test]
fn build_target_arm64_emits_printf_for_a_printing_program() {
    // `build --target arm64` on a program that prints -- a `println` of a
    // string interpolation with an i64 hole -- emits assembly that calls
    // `printf`: the format string in `.data`, the hole in an argument
    // register, `bl printf`. the integer `print` / `println` output path.
    let dir = tempfile::tempdir().unwrap();
    let src_path = dir.path().join("prog.qala");
    std::fs::write(
        &src_path,
        "fn main() is io {\n    let n = 7\n    println(\"n = {n}\")\n}\n",
    )
    .unwrap();
    let out_path = dir.path().join("prog.s");

    Command::cargo_bin("qala")
        .unwrap()
        .args(["build"])
        .arg(&src_path)
        .args(["--target", "arm64"])
        .arg("-o")
        .arg(&out_path)
        .assert()
        .success();

    let assembly = std::fs::read_to_string(&out_path).unwrap();
    assert!(assembly.contains(".data")); // the format string lands in .data
    assert!(assembly.contains("%lld")); // the i64 hole's conversion
    assert!(assembly.contains("bl      printf")); // the libc call
}

#[test]
fn build_target_arm64_rejects_an_unsupported_construct() {
    // `build --target arm64` on a program using an unsupported construct -- a
    // float -- renders a diagnostic to stderr and exits non-zero. the clean
    // rejection path through the CLI: never a panic, never bad assembly.
    let mut src = tempfile::Builder::new().suffix(".qala").tempfile().unwrap();
    write!(src, "fn main() {{\n    let x = 1.5\n}}").unwrap();

    Command::cargo_bin("qala")
        .unwrap()
        .arg("build")
        .arg(src.path())
        .args(["--target", "arm64"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("does not yet support"));
}
