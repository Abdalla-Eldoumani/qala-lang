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
fn build_target_arm64_reports_not_yet_available_and_exits_nonzero() {
    let mut src = tempfile::Builder::new().suffix(".qala").tempfile().unwrap();
    write!(src, "fn main() is io {{\n  println(\"hi\")\n}}").unwrap();

    Command::cargo_bin("qala")
        .unwrap()
        .arg("build")
        .arg(src.path())
        .args(["--target", "arm64"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("arm64 backend is not yet available"));
}
