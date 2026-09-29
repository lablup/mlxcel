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

//! Live microphone / speaker loop against the Nemotron VoiceChat
//! `/v1/realtime` WebSocket (issue #1376).
//!
//! ```text
//! mlxcel-server -m models/NemotronLabs-VoiceChat-11B-4bit --port 8080 &
//! cargo run --release --features voicechat-mic --example voicechat_microphone -- \
//!     ws://127.0.0.1:8080/v1/realtime --system-prompt "Be concise and answer in one sentence."
//! ```
//!
//! Captures the input device, downmixes to mono and resamples to 16 kHz,
//! sends `input_audio_buffer.append` every 80 ms, plays every
//! `response.audio.delta` (22.05 kHz, resampled to the output device rate)
//! through a shared ring buffer, and prints the user transcript and the
//! assistant text as they stream. Ctrl-C sends `input_audio_buffer.commit`
//! and exits after `response.done`.
//!
//! There is no acoustic echo cancellation: the model keeps listening while
//! it speaks, so use headphones or it will hear itself.
//!
//! Flags: `--list-devices`, `--input-device <name>`, `--output-device
//! <name>` (substring match on the device name), `--seed <n>`,
//! `--api-key <key>`.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use base64::Engine as _;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, Sample, SampleFormat, SizedSample};
use futures::{SinkExt, StreamExt};
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;

const INPUT_RATE: u32 = 16_000;
const OUTPUT_RATE: u32 = 22_050;
const FRAME_SAMPLES: usize = 1280;
/// Cap on queued playback (10 s) so a stalled output device cannot grow
/// memory without bound.
const MAX_PLAYBACK_SAMPLES: usize = 10 * 48_000;

#[derive(Default)]
struct Args {
    url: String,
    system_prompt: Option<String>,
    seed: u64,
    list_devices: bool,
    input_device: Option<String>,
    output_device: Option<String>,
    api_key: Option<String>,
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args::default();
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut value = |name: &str| it.next().ok_or(format!("{name} needs a value"));
        match arg.as_str() {
            "--list-devices" => args.list_devices = true,
            "--system-prompt" => args.system_prompt = Some(value("--system-prompt")?),
            "--seed" => {
                args.seed = value("--seed")?
                    .parse()
                    .map_err(|e| format!("--seed: {e}"))?
            }
            "--input-device" => args.input_device = Some(value("--input-device")?),
            "--output-device" => args.output_device = Some(value("--output-device")?),
            "--api-key" => args.api_key = Some(value("--api-key")?),
            other if other.starts_with("--") => return Err(format!("unknown flag {other}")),
            other if args.url.is_empty() => args.url = other.to_string(),
            other => return Err(format!("unexpected argument {other}")),
        }
    }
    if args.url.is_empty() && !args.list_devices {
        return Err(
            "usage: voicechat_microphone <ws://host:port/v1/realtime> [--system-prompt TEXT] \
             [--seed N] [--input-device NAME] [--output-device NAME] [--api-key KEY] \
             | --list-devices"
                .to_string(),
        );
    }
    Ok(args)
}

fn list_devices(host: &cpal::Host) -> Result<(), String> {
    let default_in = host.default_input_device().map(|d| d.to_string());
    let default_out = host.default_output_device().map(|d| d.to_string());
    println!("Input devices:");
    for device in host.input_devices().map_err(|e| e.to_string())? {
        let name = device.to_string();
        let mark = if Some(&name) == default_in.as_ref() {
            " (default)"
        } else {
            ""
        };
        println!("  {name}{mark}");
    }
    println!("Output devices:");
    for device in host.output_devices().map_err(|e| e.to_string())? {
        let name = device.to_string();
        let mark = if Some(&name) == default_out.as_ref() {
            " (default)"
        } else {
            ""
        };
        println!("  {name}{mark}");
    }
    Ok(())
}

fn pick_device(
    host: &cpal::Host,
    wanted: Option<&str>,
    input: bool,
) -> Result<cpal::Device, String> {
    let Some(wanted) = wanted else {
        let device = if input {
            host.default_input_device()
        } else {
            host.default_output_device()
        };
        return device.ok_or_else(|| "no default audio device".to_string());
    };
    let devices: Vec<cpal::Device> = if input {
        host.input_devices().map_err(|e| e.to_string())?.collect()
    } else {
        host.output_devices().map_err(|e| e.to_string())?.collect()
    };
    let needle = wanted.to_lowercase();
    devices
        .into_iter()
        .find(|d| d.to_string().to_lowercase().contains(&needle))
        .ok_or_else(|| format!("no audio device matches {wanted:?} (see --list-devices)"))
}

/// Streaming linear resampler (mono) between fixed rates.
struct Resampler {
    step: f64,
    pos: f64,
    last: f32,
}

impl Resampler {
    fn new(from: u32, to: u32) -> Self {
        Self {
            step: f64::from(from) / f64::from(to),
            pos: 0.0,
            last: 0.0,
        }
    }

    fn process(&mut self, input: &[f32], out: &mut Vec<f32>) {
        // `pos` indexes the virtual sequence [last, input...].
        let len = input.len() as f64;
        while self.pos < len {
            let i = self.pos.floor();
            let frac = (self.pos - i) as f32;
            let i = i as usize;
            let a = if i == 0 { self.last } else { input[i - 1] };
            let b = input[i.min(input.len() - 1)];
            out.push(a + (b - a) * frac);
            self.pos += self.step;
        }
        self.pos -= len;
        if let Some(&x) = input.last() {
            self.last = x;
        }
    }
}

/// Wrap an input-device failure with the device name and, on macOS, the
/// microphone-permission hint (a process without Microphone access gets bare
/// CoreAudio errors such as "Unknown property" or a hung stream).
fn input_error(device: &cpal::Device, err: impl std::fmt::Display) -> String {
    let mut msg = format!("input device {device}: {err}.");
    if cfg!(target_os = "macos") {
        msg.push_str(
            " On macOS, grant Microphone access to this terminal in System Settings > Privacy & \
             Security > Microphone and restart the terminal.",
        );
    }
    msg
}

/// The device's default input config, or its first supported config (at the
/// maximum sample rate) when the default cannot be queried.
fn input_config(device: &cpal::Device) -> Result<cpal::SupportedStreamConfig, String> {
    let default_err = match device.default_input_config() {
        Ok(cfg) => return Ok(cfg),
        Err(e) => e,
    };
    let fallback = device
        .supported_input_configs()
        .ok()
        .and_then(|mut configs| configs.next())
        .map(|range| range.with_max_sample_rate());
    match fallback {
        Some(cfg) => {
            eprintln!("default input config unavailable ({default_err}); using a supported config");
            Ok(cfg)
        }
        None => Err(input_error(device, default_err)),
    }
}

fn build_input<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    tx: mpsc::UnboundedSender<Vec<f32>>,
) -> Result<cpal::Stream, String>
where
    T: SizedSample,
    f32: FromSample<T>,
{
    let channels = usize::from(config.channels.max(1));
    let mut resampler = Resampler::new(config.sample_rate, INPUT_RATE);
    device
        .build_input_stream::<T, _, _>(
            *config,
            move |data: &[T], _| {
                let mono: Vec<f32> = data
                    .chunks(channels)
                    .map(|frame| {
                        frame.iter().map(|&s| f32::from_sample(s)).sum::<f32>() / channels as f32
                    })
                    .collect();
                let mut out = Vec::with_capacity(mono.len());
                resampler.process(&mono, &mut out);
                let _ = tx.send(out);
            },
            |err| eprintln!("input stream error: {err}"),
            None,
        )
        .map_err(|e| input_error(device, e))
}

fn build_output<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    queue: Arc<Mutex<VecDeque<f32>>>,
) -> Result<cpal::Stream, String>
where
    T: SizedSample + FromSample<f32>,
{
    let channels = usize::from(config.channels.max(1));
    device
        .build_output_stream::<T, _, _>(
            *config,
            move |data: &mut [T], _| {
                let mut queue = queue.lock().unwrap_or_else(|p| p.into_inner());
                for frame in data.chunks_mut(channels) {
                    let x = queue.pop_front().unwrap_or(0.0);
                    for slot in frame.iter_mut() {
                        *slot = T::from_sample(x);
                    }
                }
            },
            |err| eprintln!("output stream error: {err}"),
            None,
        )
        .map_err(|e| e.to_string())
}

fn pcm16_base64(samples: &[f32]) -> String {
    let mut bytes = Vec::with_capacity(samples.len() * 2);
    for &x in samples {
        let v = (x.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

fn pcm16_decode(b64: &str) -> Vec<f32> {
    let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(b64) else {
        return Vec::new();
    };
    bytes
        .chunks_exact(2)
        .map(|b| f32::from(i16::from_le_bytes([b[0], b[1]])) / 32768.0)
        .collect()
}

#[tokio::main]
async fn main() -> Result<(), String> {
    let args = parse_args()?;
    let host = cpal::default_host();
    if args.list_devices {
        return list_devices(&host);
    }
    let input = pick_device(&host, args.input_device.as_deref(), true)?;
    let output = pick_device(&host, args.output_device.as_deref(), false)?;
    let in_cfg = input_config(&input)?;
    let out_cfg = output.default_output_config().map_err(|e| e.to_string())?;
    eprintln!(
        "input: {input} ({} Hz), output: {output} ({} Hz). Use headphones: there is no echo \
         cancellation.",
        in_cfg.sample_rate(),
        out_cfg.sample_rate()
    );

    let (mic_tx, mut mic_rx) = mpsc::unbounded_channel::<Vec<f32>>();
    let in_stream_cfg = in_cfg.config();
    let in_stream = match in_cfg.sample_format() {
        SampleFormat::F32 => build_input::<f32>(&input, &in_stream_cfg, mic_tx)?,
        SampleFormat::I16 => build_input::<i16>(&input, &in_stream_cfg, mic_tx)?,
        SampleFormat::U16 => build_input::<u16>(&input, &in_stream_cfg, mic_tx)?,
        SampleFormat::I32 => build_input::<i32>(&input, &in_stream_cfg, mic_tx)?,
        other => return Err(format!("unsupported input sample format {other}")),
    };
    let playback = Arc::new(Mutex::new(VecDeque::<f32>::new()));
    let out_stream_cfg = out_cfg.config();
    let out_stream = match out_cfg.sample_format() {
        SampleFormat::F32 => build_output::<f32>(&output, &out_stream_cfg, playback.clone())?,
        SampleFormat::I16 => build_output::<i16>(&output, &out_stream_cfg, playback.clone())?,
        SampleFormat::U16 => build_output::<u16>(&output, &out_stream_cfg, playback.clone())?,
        SampleFormat::I32 => build_output::<i32>(&output, &out_stream_cfg, playback.clone())?,
        other => return Err(format!("unsupported output sample format {other}")),
    };

    let mut request = args
        .url
        .as_str()
        .into_client_request()
        .map_err(|e| e.to_string())?;
    if let Some(key) = &args.api_key {
        let value = format!("Bearer {key}")
            .parse()
            .map_err(|e| format!("{e}"))?;
        request.headers_mut().insert("Authorization", value);
    }
    let (socket, _) = tokio_tungstenite::connect_async(request)
        .await
        .map_err(|e| format!("connect {}: {e}", args.url))?;
    let (mut sink, mut stream) = socket.split();
    let mut session = json!({ "seed": args.seed });
    if let Some(prompt) = &args.system_prompt {
        session["system_prompt"] = json!(prompt);
    }
    let update = json!({ "type": "session.update", "session": session });
    sink.send(Message::Text(update.to_string()))
        .await
        .map_err(|e| e.to_string())?;

    in_stream.play().map_err(|e| input_error(&input, e))?;
    out_stream.play().map_err(|e| e.to_string())?;
    let mut to_device = Resampler::new(OUTPUT_RATE, out_cfg.sample_rate());
    let mut pending: Vec<f32> = Vec::new();
    let mut ready = false;
    let mut committed = false;
    let mut ticker = tokio::time::interval(Duration::from_millis(80));

    loop {
        tokio::select! {
            Some(chunk) = mic_rx.recv() => pending.extend(chunk),
            _ = ticker.tick(), if ready && !committed => {
                if pending.len() >= FRAME_SAMPLES {
                    let whole = pending.len() / FRAME_SAMPLES * FRAME_SAMPLES;
                    let chunk: Vec<f32> = pending.drain(..whole).collect();
                    let msg = json!({
                        "type": "input_audio_buffer.append",
                        "audio": pcm16_base64(&chunk),
                        "sample_rate": INPUT_RATE,
                    });
                    sink.send(Message::Text(msg.to_string())).await.map_err(|e| e.to_string())?;
                }
            }
            // A fresh Ctrl-C future per iteration: the first press commits
            // (or quits before the session is ready), a second one quits.
            _ = tokio::signal::ctrl_c() => {
                if !ready || committed {
                    eprintln!("\nquitting");
                    break;
                }
                committed = true;
                eprintln!("\ncommitting (Ctrl-C again to quit now)...");
                if !pending.is_empty() {
                    let msg = json!({
                        "type": "input_audio_buffer.append",
                        "audio": pcm16_base64(&std::mem::take(&mut pending)),
                        "sample_rate": INPUT_RATE,
                    });
                    sink.send(Message::Text(msg.to_string())).await.map_err(|e| e.to_string())?;
                }
                let msg = json!({ "type": "input_audio_buffer.commit", "pad_partial": true });
                sink.send(Message::Text(msg.to_string())).await.map_err(|e| e.to_string())?;
            }
            message = stream.next() => {
                let Some(message) = message else { break };
                let Message::Text(text) = message.map_err(|e| e.to_string())? else { continue };
                let event: Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
                match event["type"].as_str().unwrap_or_default() {
                    "session.updated" => {
                        ready = true;
                        eprintln!("session ready; speak (Ctrl-C to finish)");
                    }
                    "conversation.item.input_audio_transcription.delta" => {
                        eprintln!("\n[user] {}", event["transcript"].as_str().unwrap_or_default());
                    }
                    "response.text.delta" => {
                        print!("{}", event["delta"].as_str().unwrap_or_default());
                        use std::io::Write as _;
                        let _ = std::io::stdout().flush();
                    }
                    "response.function.delta" => {
                        eprintln!("\n[function] {}", event["delta"].as_str().unwrap_or_default());
                    }
                    "response.audio.delta" => {
                        let samples = pcm16_decode(event["delta"].as_str().unwrap_or_default());
                        let mut out = Vec::with_capacity(samples.len() * 3);
                        to_device.process(&samples, &mut out);
                        let mut queue = playback.lock().unwrap_or_else(|p| p.into_inner());
                        queue.extend(out);
                        let excess = queue.len().saturating_sub(MAX_PLAYBACK_SAMPLES);
                        queue.drain(..excess);
                    }
                    "response.done" | "response.cancelled" => break,
                    "error" => {
                        eprintln!("\nerror: {}", event["error"]);
                        if committed {
                            // The flush failed; response.done will not come.
                            break;
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    // Let the queued answer finish playing before the streams drop.
    while !playback
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .is_empty()
    {
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    println!();
    Ok(())
}
