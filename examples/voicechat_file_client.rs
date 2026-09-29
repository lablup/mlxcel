// Copyright 2025-2026 Lablup Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Drive a WAV file through the `/v1/realtime` VoiceChat WebSocket and write
//! the assistant's speech (issue #1376).
//!
//! ```text
//! cargo run --release --example voicechat_file_client -- \
//!     ws://127.0.0.1:8080/v1/realtime question.wav answer.wav \
//!     --system-prompt "Be concise and answer in one sentence."
//! ```
//!
//! The input is resampled to 16 kHz mono, followed by `--extra-seconds` of
//! silence (default 3, like `mlxcel generate --extra-decoding-seconds`), and
//! sent as `input_audio_buffer.append` chunks of 1280 samples (80 ms), then
//! `input_audio_buffer.commit`. `--realtime` paces the chunks at 80 ms.
//! Every `response.audio.delta` is appended to the output WAV exactly as it
//! arrived on the wire (PCM16, 22.05 kHz). `--events <path>` writes every
//! event as one JSON line, with the base64 audio payload replaced by its
//! sample count. `--api-key` sends `Authorization: Bearer <key>`.

use std::path::PathBuf;
use std::time::Duration;

use base64::Engine as _;
use futures::{SinkExt, StreamExt};
use serde_json::{Value, json};
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;

const FRAME_SAMPLES: usize = 1280;

struct Args {
    url: String,
    input: PathBuf,
    output: PathBuf,
    system_prompt: Option<String>,
    seed: u64,
    extra_seconds: f32,
    realtime: bool,
    events: Option<PathBuf>,
    api_key: Option<String>,
}

fn parse_args() -> Result<Args, String> {
    let mut positional = Vec::new();
    let mut args = Args {
        url: String::new(),
        input: PathBuf::new(),
        output: PathBuf::new(),
        system_prompt: None,
        seed: 0,
        extra_seconds: 3.0,
        realtime: false,
        events: None,
        api_key: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut value = |name: &str| it.next().ok_or(format!("{name} needs a value"));
        match arg.as_str() {
            "--system-prompt" => args.system_prompt = Some(value("--system-prompt")?),
            "--seed" => {
                args.seed = value("--seed")?
                    .parse()
                    .map_err(|e| format!("--seed: {e}"))?
            }
            "--extra-seconds" => {
                args.extra_seconds = value("--extra-seconds")?
                    .parse()
                    .map_err(|e| format!("--extra-seconds: {e}"))?
            }
            "--events" => args.events = Some(value("--events")?.into()),
            "--api-key" => args.api_key = Some(value("--api-key")?),
            "--realtime" => args.realtime = true,
            flag if flag.starts_with("--") => return Err(format!("unknown flag {flag}")),
            _ => positional.push(arg),
        }
    }
    let [url, input, output] = <[String; 3]>::try_from(positional).map_err(|_| {
        "usage: voicechat_file_client <ws-url> <input.wav> <output.wav> [--system-prompt S] \
         [--seed N] [--extra-seconds S] [--realtime] [--events out.jsonl] [--api-key K]"
            .to_string()
    })?;
    args.url = url;
    args.input = input.into();
    args.output = output.into();
    Ok(args)
}

/// 16 kHz input as PCM16 (the exact inverse of the server's `/ 32768`
/// decode for 16-bit sources).
fn load_input(args: &Args) -> Result<Vec<i16>, String> {
    let (samples, rate) = mlxcel::audio::feature_extractor::load_wav_file(&args.input)?;
    let mut samples = mlxcel::audio::whisper_mel::resample_to_16k(&samples, rate);
    let silence = (args.extra_seconds.max(0.0) * 16_000.0).round() as usize;
    samples.resize(samples.len() + silence, 0.0);
    Ok(samples
        .iter()
        .map(|&x| (x * 32768.0).round().clamp(-32768.0, 32767.0) as i16)
        .collect())
}

fn wav_pcm16(pcm: &[u8], sample_rate: u32) -> Vec<u8> {
    let data_len = pcm.len() as u32;
    let mut out = Vec::with_capacity(44 + pcm.len());
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&(sample_rate * 2).to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    out.extend_from_slice(pcm);
    out
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args()?;
    let input = load_input(&args)?;
    let mut request = args.url.as_str().into_client_request()?;
    if let Some(key) = &args.api_key {
        request
            .headers_mut()
            .insert("Authorization", format!("Bearer {key}").parse()?);
    }
    let (socket, _) = tokio_tungstenite::connect_async(request).await?;
    let (mut sink, mut stream) = socket.split();

    // One task owns the write half; everything else queues messages to it.
    let (outgoing, mut queued) = tokio::sync::mpsc::unbounded_channel::<Message>();
    let forwarder = tokio::spawn(async move {
        while let Some(message) = queued.recv().await {
            sink.send(message).await?;
        }
        sink.close().await
    });

    let mut session = json!({"seed": args.seed});
    if let Some(prompt) = &args.system_prompt {
        session["system_prompt"] = json!(prompt);
    }
    outgoing.send(Message::Text(
        json!({"type": "session.update", "session": session}).to_string(),
    ))?;

    let mut writer = None;
    let mut events_out = Vec::new();
    let mut pcm = Vec::new();
    let mut output_rate = 22_050u32;
    let (mut transcript, mut text, mut function) = (String::new(), String::new(), String::new());
    let mut last_audio_frame: Option<u64> = None;
    let (mut audio_deltas, mut bad_audio_deltas) = (0usize, 0usize);
    let mut done = false;
    let mut input_chunks = Some(input);

    while let Some(message) = stream.next().await {
        let text_frame = match message? {
            Message::Text(t) => t,
            Message::Close(frame) => {
                eprintln!("server closed the socket: {frame:?}");
                break;
            }
            _ => continue,
        };
        let mut event: Value = serde_json::from_str(&text_frame)?;
        match event["type"].as_str().unwrap_or_default() {
            "session.updated" => {
                // Start streaming the input once the session is configured.
                if let Some(input) = input_chunks.take() {
                    let realtime = args.realtime;
                    let outgoing = outgoing.clone();
                    writer = Some(tokio::spawn(async move {
                        for chunk in input.chunks(FRAME_SAMPLES) {
                            let bytes: Vec<u8> =
                                chunk.iter().flat_map(|s| s.to_le_bytes()).collect();
                            let append = json!({
                                "type": "input_audio_buffer.append",
                                "audio": base64::engine::general_purpose::STANDARD.encode(bytes),
                                "sample_rate": 16_000,
                            });
                            if outgoing.send(Message::Text(append.to_string())).is_err() {
                                return;
                            }
                            if realtime {
                                tokio::time::sleep(Duration::from_millis(80)).await;
                            }
                        }
                        let commit = json!({"type": "input_audio_buffer.commit"});
                        let _ = outgoing.send(Message::Text(commit.to_string()));
                    }));
                }
            }
            "conversation.item.input_audio_transcription.delta" => {
                transcript = event["transcript"].as_str().unwrap_or_default().to_string();
            }
            "response.text.delta" => {
                print!("{}", event["delta"].as_str().unwrap_or_default());
                text = event["text"].as_str().unwrap_or_default().to_string();
            }
            "response.function.delta" => {
                function = event["text"].as_str().unwrap_or_default().to_string();
            }
            "response.audio.delta" => {
                let bytes = base64::engine::general_purpose::STANDARD
                    .decode(event["delta"].as_str().unwrap_or_default())?;
                let frame = event["frame_index"].as_u64().unwrap_or_default();
                audio_deltas += 1;
                if bytes.len() != 1764 * 2 || last_audio_frame.is_some_and(|last| frame <= last) {
                    bad_audio_deltas += 1;
                }
                last_audio_frame = Some(frame);
                output_rate = event["sample_rate"].as_u64().unwrap_or(22_050) as u32;
                pcm.extend_from_slice(&bytes);
                event["delta"] = json!(bytes.len() / 2);
            }
            "response.done" | "response.cancelled" => done = true,
            "error" => eprintln!("error event: {}", event["error"]),
            _ => {}
        }
        events_out.push(event.to_string());
    }
    println!();

    if let Some(writer) = writer {
        writer.await?;
    }
    drop(outgoing);
    // The server closes first after `response.done`; a failed close of the
    // already-closed socket is expected.
    let _ = forwarder.await;
    std::fs::write(&args.output, wav_pcm16(&pcm, output_rate))?;
    if let Some(path) = &args.events {
        std::fs::write(path, events_out.join("\n") + "\n")?;
    }
    println!("[user] {transcript}");
    println!("[assistant] {text}");
    if !function.is_empty() {
        println!("[function] {function}");
    }
    eprintln!(
        "{audio_deltas} audio deltas ({bad_audio_deltas} not 1764 samples or out of order), \
         {} samples at {output_rate} Hz written to {}, response finished: {done}",
        pcm.len() / 2,
        args.output.display()
    );
    if !done || bad_audio_deltas > 0 {
        return Err("the session did not finish cleanly".into());
    }
    Ok(())
}
