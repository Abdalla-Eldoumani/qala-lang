use assert_cmd::Command;
use predicates::prelude::*;
use std::io::Write;

#[test]
fn check_on_a_broken_program_reports_a_diagnostic_and_exits_nonzero() {
    // a deliberate type error: an i64 annotation, a str initializer.
    let mut src = tempfile::Builder::new().suffix(".qala").tempfile().unwrap();
    write!(src, "fn main() is io {{\n  let x: i64 = \"nope\"\n}}").unwrap();

    Command::cargo_bin("qala")
        .unwrap()
        .arg("check")
        .arg(src.path())
        .assert()
        .failure() // non-zero exit
        .stderr(predicate::str::contains("error:")); // a rendered diagnostic on stderr
}

#[test]
fn check_on_a_good_program_succeeds_without_running_it() {
    let mut src = tempfile::Builder::new().suffix(".qala").tempfile().unwrap();
    write!(src, "fn main() is io {{\n  println(\"hi\")\n}}").unwrap();

    Command::cargo_bin("qala")
        .unwrap()
        .arg("check")
        .arg(src.path())
        .assert()
        .success()
        // check does NOT run the VM, so the program's output never appears.
        .stdout(predicate::str::is_empty());
}
