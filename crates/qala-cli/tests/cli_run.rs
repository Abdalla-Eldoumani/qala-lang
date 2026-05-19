use assert_cmd::Command;
use predicates::prelude::*;

const HELLO: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../playground/public/examples/hello.qala"
);

#[test]
fn run_executes_a_known_good_program_and_prints_its_output() {
    Command::cargo_bin("qala")
        .unwrap()
        .args(["run", HELLO])
        .assert()
        .success() // exit 0
        .stdout(predicate::str::contains("hello, world!"));
}

#[test]
fn version_flag_prints_the_crate_version() {
    Command::cargo_bin("qala")
        .unwrap()
        .arg("--version")
        .assert()
        .success()
        // CARGO_PKG_VERSION of qala-cli, currently 0.1.0
        .stdout(predicate::str::contains(env!("CARGO_PKG_VERSION")));
}

#[test]
fn help_flag_lists_the_four_subcommands() {
    Command::cargo_bin("qala")
        .unwrap()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("run"))
        .stdout(predicate::str::contains("check"))
        .stdout(predicate::str::contains("build"))
        .stdout(predicate::str::contains("repl"));
}
