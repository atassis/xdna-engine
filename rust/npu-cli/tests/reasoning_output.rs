use std::io::{BufRead, BufReader, Read, Write};
use std::os::unix::net::UnixListener;
use std::process::{Command, Output, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use npu_models::GenerationReport;
use serde_json::{json, Value};

const REASONING: &str = "Compare the primary colors.";
const ANSWER: &str = "Red, green, blue.";

fn response(stream: bool, reasoning: &str) -> String {
    let report = GenerationReport::default();
    if !stream {
        return json!({
            "id": "chatcmpl-test", "object": "chat.completion", "created": 7, "model": "test",
            "choices": [{"index": 0, "message": {
                "role": "assistant", "content": ANSWER, "reasoning_content": reasoning
            }, "finish_reason": "stop"}], "x_npu_report": report
        }).to_string();
    }
    let mut frames = Vec::new();
    for text in ["", reasoning] {
        frames.push(json!({
            "id": "chatcmpl-test", "object": "chat.completion.chunk", "created": 7, "model": "test",
            "choices": [{"index": 0, "delta": {"reasoning_content": text}, "finish_reason": null}]
        }));
    }
    frames.push(json!({
        "id": "chatcmpl-test", "object": "chat.completion.chunk", "created": 7, "model": "test",
        "choices": [{"index": 0, "delta": {"content": ANSWER}, "finish_reason": null}],
        "x_npu": {"det": {"seq": 0, "tok_id": 1, "text": ANSWER}}
    }));
    frames.push(json!({"x_npu_report": report}));
    let mut body = frames.iter().map(|v| format!("data: {v}\n\n")).collect::<String>();
    body.push_str("data: [DONE]\n\n");
    body
}

fn run(args: &[&str], reasoning: &str) -> (Output, Option<Value>) {
    let dir = tempfile::tempdir().unwrap();
    let socket = dir.path().join("control.sock");
    let listener = UnixListener::bind(&socket).unwrap();
    listener.set_nonblocking(true).unwrap();
    let reasoning = reasoning.to_string();
    let server = thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut stream = loop {
            match listener.accept() {
                Ok((stream, _)) => break stream,
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock && Instant::now() < deadline => {
                    thread::sleep(Duration::from_millis(10));
                }
                Err(_) => return None,
            }
        };
        stream.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
        let mut reader = BufReader::new(stream.try_clone().unwrap());
        let mut length = 0;
        loop {
            let mut line = String::new();
            reader.read_line(&mut line).unwrap();
            if line == "\r\n" { break; }
            if let Some(value) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                length = value.trim().parse::<usize>().unwrap();
            }
        }
        let mut body = vec![0; length];
        reader.read_exact(&mut body).unwrap();
        let request: Value = serde_json::from_slice(&body).unwrap();
        let streaming = request["stream"].as_bool().unwrap();
        let body = response(streaming, &reasoning);
        let content_type = if streaming { "text/event-stream" } else { "application/json" };
        write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\n\r\n{body}", body.len()).unwrap();
        Some(request)
    });
    let output = Command::new(env!("CARGO_BIN_EXE_npu"))
        .args(args).env("NPU_SOCKET_ENDPOINT", &socket).stdin(Stdio::null()).output().unwrap();
    (output, server.join().unwrap())
}

#[test]
fn reasoning_is_on_stderr_for_streamed_and_buffered_text() {
    for command in ["generate", "chat"] {
        for buffered in [false, true] {
            let mut args = vec![command, "hi", "--think"];
            if buffered { args.push("--no-stream"); }
            let (output, _) = run(&args, REASONING);
            assert!(output.status.success(), "{:?}", String::from_utf8_lossy(&output.stderr));
            let stdout = String::from_utf8(output.stdout).unwrap();
            let stderr = String::from_utf8(output.stderr).unwrap();
            assert!(stdout.contains(ANSWER));
            assert!(!stdout.contains(REASONING));
            assert!(stderr.contains(REASONING), "missing reasoning: {stderr:?}");
            assert!(stderr.contains("[reasoning]"));
            assert_eq!(stderr.matches("[reasoning]").count(), 1);
            if command == "generate" { assert_eq!(stdout, format!("{ANSWER}\n")); }
        }
    }
}

#[test]
fn hiding_reasoning_does_not_change_the_generation_request() {
    for command in ["generate", "chat"] {
        for buffered in [false, true] {
            let mut args = vec![command, "hi", "--think", "--hide-reasoning"];
            if buffered { args.push("--no-stream"); }
            let (output, hidden_request) = run(&args, REASONING);
            assert!(output.status.success(), "{:?}", String::from_utf8_lossy(&output.stderr));
            assert!(!String::from_utf8_lossy(&output.stderr).contains(REASONING));
            assert!(!String::from_utf8_lossy(&output.stdout).contains(REASONING));
            args.retain(|arg| *arg != "--hide-reasoning");
            let (_, shown_request) = run(&args, REASONING);
            assert_eq!(hidden_request.unwrap(), shown_request.unwrap());
        }
    }
}

#[test]
fn json_retains_reasoning_even_when_display_is_hidden() {
    for buffered in [false, true] {
        for hidden in [false, true] {
            let mut args = vec!["generate", "hi", "--output", "json"];
            if buffered { args.push("--no-stream"); }
            if hidden { args.push("--hide-reasoning"); }
            let (output, _) = run(&args, REASONING);
            assert!(output.status.success(), "{:?}", String::from_utf8_lossy(&output.stderr));
            assert!(!String::from_utf8_lossy(&output.stderr).contains(REASONING));
            let stdout = String::from_utf8(output.stdout).unwrap();
            let values: Vec<Value> = stdout.lines().map(|line| serde_json::from_str(line).unwrap()).collect();
            if buffered {
                assert_eq!(values.len(), 1);
                assert_eq!(values[0]["choices"][0]["message"]["content"], ANSWER);
                assert_eq!(values[0]["choices"][0]["message"]["reasoning_content"], REASONING);
            } else {
                let reasoning = values.iter().filter_map(|v| v.pointer("/choices/0/delta/reasoning_content")?.as_str()).collect::<String>();
                assert_eq!(reasoning, REASONING);
                let run = npu_models::telemetry::wire::parse_run(&stdout).unwrap();
                assert_eq!(run.steps.len(), 1);
                assert_eq!(run.steps[0].emit, ANSWER);
            }
        }
    }
}

#[test]
fn absent_reasoning_has_no_empty_display_block() {
    for buffered in [false, true] {
        let mut args = vec!["generate", "hi", "--no-think"];
        if buffered { args.push("--no-stream"); }
        let (output, _) = run(&args, "");
        assert!(output.status.success());
        assert_eq!(String::from_utf8(output.stdout).unwrap(), format!("{ANSWER}\n"));
        assert!(!String::from_utf8_lossy(&output.stderr).contains("[reasoning]"));
    }
}

#[test]
fn streamed_reasoning_is_visible_before_the_answer_arrives() {
    let dir = tempfile::tempdir().unwrap();
    let socket = dir.path().join("control.sock");
    let listener = UnixListener::bind(&socket).unwrap();
    let (seen_tx, seen_rx) = std::sync::mpsc::channel();
    let server = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        stream.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
        let mut reader = BufReader::new(stream.try_clone().unwrap());
        let mut length = 0;
        loop {
            let mut line = String::new();
            reader.read_line(&mut line).unwrap();
            if line == "\r\n" { break; }
            if let Some(value) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                length = value.trim().parse::<usize>().unwrap();
            }
        }
        let mut request = vec![0; length];
        reader.read_exact(&mut request).unwrap();
        let body = response(true, &format!("{REASONING}\n"));
        let first = body.split_inclusive("\n\n").take(2).collect::<String>();
        write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n{first}").unwrap();
        stream.flush().unwrap();
        let seen = seen_rx.recv_timeout(Duration::from_secs(3)).is_ok();
        stream.write_all(&body.as_bytes()[first.len()..]).unwrap();
        seen
    });
    let mut child = Command::new(env!("CARGO_BIN_EXE_npu"))
        .args(["generate", "hi", "--think"]).env("NPU_SOCKET_ENDPOINT", &socket)
        .stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap();
    let stderr = child.stderr.take().unwrap();
    let observer = thread::spawn(move || {
        let mut reader = BufReader::new(stderr);
        let mut label = String::new();
        let mut text = String::new();
        reader.read_line(&mut label).unwrap();
        reader.read_line(&mut text).unwrap();
        if label == "[reasoning]\n" && text == format!("{REASONING}\n") {
            seen_tx.send(()).unwrap();
        }
        let mut rest = String::new();
        reader.read_to_string(&mut rest).unwrap();
    });
    let output = child.wait_with_output().unwrap();
    observer.join().unwrap();
    assert!(server.join().unwrap(), "reasoning was buffered until the answer");
    assert!(output.status.success());
    assert_eq!(String::from_utf8(output.stdout).unwrap(), format!("{ANSWER}\n"));
}
