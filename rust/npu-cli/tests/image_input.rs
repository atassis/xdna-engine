use std::io::{BufRead, BufReader, Read, Write};
use std::os::unix::net::UnixListener;
use std::process::{Command, Output, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use base64::Engine;
use npu_models::GenerationReport;
use serde_json::{json, Value};

fn response() -> String {
    json!({
        "id": "chatcmpl-image-test", "object": "chat.completion", "created": 7, "model": "test",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        "x_npu_report": GenerationReport::default(),
    }).to_string()
}

fn encoded_image(format: image::ImageFormat) -> Vec<u8> {
    let mut output = std::io::Cursor::new(Vec::new());
    image::DynamicImage::new_rgb8(2, 2).write_to(&mut output, format).unwrap();
    output.into_inner()
}

fn read_request(stream: &std::os::unix::net::UnixStream) -> Value {
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut length = 0;
    loop {
        let mut line = String::new();
        assert!(reader.read_line(&mut line).unwrap() > 0, "request header ended early");
        if line == "\r\n" { break; }
        if let Some(value) = line.to_ascii_lowercase().strip_prefix("content-length:") {
            length = value.trim().parse::<usize>().unwrap();
        }
    }
    let mut body = vec![0; length];
    reader.read_exact(&mut body).unwrap();
    serde_json::from_slice(&body).unwrap()
}

fn run(args: &[&str], stdin: &[u8], request_count: usize) -> (Output, Vec<Value>) {
    let dir = tempfile::tempdir().unwrap();
    let socket = dir.path().join("control.sock");
    let listener = UnixListener::bind(&socket).unwrap();
    listener.set_nonblocking(true).unwrap();
    let server = thread::spawn(move || {
        let mut requests = Vec::new();
        for _ in 0..request_count {
            let deadline = Instant::now() + Duration::from_secs(5);
            let mut stream = loop {
                match listener.accept() {
                    Ok((stream, _)) => break stream,
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock && Instant::now() < deadline => {
                        thread::sleep(Duration::from_millis(10));
                    }
                    Err(_) => return requests,
                }
            };
            stream.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
            let request = read_request(&stream);
            let body = response();
            write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}", body.len()).unwrap();
            requests.push(request);
        }
        requests
    });
    let output = Command::new(env!("CARGO_BIN_EXE_npu"))
        .args(args).env("NPU_SOCKET_ENDPOINT", &socket)
        .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped())
        .spawn().and_then(|mut child| {
            child.stdin.take().unwrap().write_all(stdin)?;
            child.wait_with_output()
        }).unwrap();
    (output, server.join().unwrap())
}

fn assert_no_request(args: &[&str]) -> Output {
    let (output, requests) = run(args, b"", 1);
    assert!(requests.is_empty(), "CLI contacted the service before rejecting input");
    output
}

#[test]
fn generate_sends_each_image_as_a_data_url_with_exact_bytes() {
    let dir = tempfile::tempdir().unwrap();
    let png = dir.path().join("one.png");
    let jpeg = dir.path().join("two.jpeg");
    let png_bytes = encoded_image(image::ImageFormat::Png);
    let jpeg_bytes = encoded_image(image::ImageFormat::Jpeg);
    std::fs::write(&png, &png_bytes).unwrap();
    std::fs::write(&jpeg, &jpeg_bytes).unwrap();
    let (output, requests) = run(&[
        "generate", "describe these", "--image", png.to_str().unwrap(),
        "--image", jpeg.to_str().unwrap(), "--no-stream",
    ], b"", 1);
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0]["messages"][0]["content"], json!([
        {"type": "text", "text": "describe these"},
        {"type": "image_url", "image_url": {"url": format!("data:image/png;base64,{}", base64::engine::general_purpose::STANDARD.encode(&png_bytes))}},
        {"type": "image_url", "image_url": {"url": format!("data:image/jpeg;base64,{}", base64::engine::general_purpose::STANDARD.encode(&jpeg_bytes))}},
    ]));
    assert_eq!(String::from_utf8(output.stdout).unwrap(), "ok\n");
}

#[test]
fn chat_attaches_images_to_the_first_user_turn_and_retains_them_in_history() {
    let dir = tempfile::tempdir().unwrap();
    let image = dir.path().join("image.png");
    let bytes = encoded_image(image::ImageFormat::Png);
    std::fs::write(&image, &bytes).unwrap();
    let (output, requests) = run(&[
        "chat", "first", "--image", image.to_str().unwrap(), "--no-stream",
    ], b"second\n", 2);
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0]["messages"][0]["content"], json!([
        {"type": "text", "text": "first"},
        {"type": "image_url", "image_url": {"url": format!("data:image/png;base64,{}", base64::engine::general_purpose::STANDARD.encode(&bytes))}},
    ]));
    assert_eq!(requests[1]["messages"][0]["content"], requests[0]["messages"][0]["content"]);
    assert_eq!(requests[1]["messages"][2], json!({"role": "user", "content": "second"}));
}

#[test]
fn missing_image_fails_before_contacting_the_service() {
    let dir = tempfile::tempdir().unwrap();
    let missing = dir.path().join("missing.png");
    let output = assert_no_request(&["generate", "describe", "--image", missing.to_str().unwrap(), "--no-stream"]);
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("read"));
}

#[test]
fn raw_generate_rejects_images_before_contacting_the_service() {
    let image = tempfile::NamedTempFile::with_suffix(".png").unwrap();
    let output = assert_no_request(&["generate", "continue", "--raw", "--image", image.path().to_str().unwrap()]);
    assert!(!output.status.success());
    assert_eq!(output.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&output.stdout).contains("--image"));
}

#[test]
fn plain_text_generate_keeps_the_string_content_shape() {
    let (output, requests) = run(&["generate", "plain", "--no-stream"], b"", 1);
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    assert_eq!(requests[0]["messages"][0], json!({"role": "user", "content": "plain"}));
    assert_eq!(String::from_utf8(output.stdout).unwrap(), "ok\n");
}

#[test]
fn unsupported_image_formats_fail_before_contacting_the_service() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("unsupported.png");
    for bytes in [b"not an image".as_slice(), b"GIF89a".as_slice()] {
        std::fs::write(&path, bytes).unwrap();
        let output = assert_no_request(&["generate", "describe", "--image", path.to_str().unwrap(), "--no-stream"]);
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains("unsupported.png"));
    }
}

#[test]
fn image_generate_json_is_one_completion_without_stderr_output() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("image.png");
    std::fs::write(&path, encoded_image(image::ImageFormat::Png)).unwrap();
    let (output, requests) = run(&["generate", "describe", "--image", path.to_str().unwrap(),
                                 "--no-stream", "--output", "json"], b"", 1);
    assert!(output.status.success());
    let completion: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(completion["choices"][0]["message"]["content"], "ok");
    assert!(output.stderr.is_empty());
    assert!(requests[0]["messages"][0]["content"].is_array());
}
