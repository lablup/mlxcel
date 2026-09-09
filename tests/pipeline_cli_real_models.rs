mod common;

use std::process::Command;

use common::{extract_generated_body, repo_model_dir};

fn run_generate(args: &[&str]) -> String {
    let output = Command::new(env!("CARGO_BIN_EXE_mlxcel"))
        .args(args)
        .output()
        .expect("failed to execute mlxcel generate");
    assert!(
        output.status.success(),
        "mlxcel generate failed: stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).expect("stdout must be valid UTF-8")
}

#[test]
#[ignore = "requires local model weights and the mlxcel binary"]
fn pipeline_cli_llama_real_model_parity() {
    let model_dir = repo_model_dir("llama-3.2-1b-instruct-4bit");
    if !model_dir.exists() {
        eprintln!(
            "Skipping test: model directory not found at {}",
            model_dir.display()
        );
        return;
    }

    let model_arg = model_dir.to_string_lossy().to_string();
    let dense_stdout = run_generate(&[
        "generate",
        "-m",
        &model_arg,
        "-p",
        "Hello",
        "-n",
        "8",
        "--temp",
        "0",
        "--no-chat-template",
    ]);
    let pipeline_stdout = run_generate(&[
        "generate",
        "-m",
        &model_arg,
        "-p",
        "Hello",
        "-n",
        "8",
        "--temp",
        "0",
        "--no-chat-template",
        "--pp-size",
        "2",
    ]);

    let dense_body = extract_generated_body(&dense_stdout).expect("missing dense generation body");
    let pipeline_body =
        extract_generated_body(&pipeline_stdout).expect("missing pipeline generation body");
    assert_eq!(pipeline_body, dense_body);
}
