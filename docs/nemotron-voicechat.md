# Nemotron VoiceChat

NemotronLabs VoiceChat (`model_type: "nemotron_voicechat"`, `architectures: ["NemotronVoiceChatForConditionalGeneration"]`) is a full-duplex speech-to-speech model. It consumes 16 kHz audio in 80 ms frames and, on every frame, emits an assistant text token, a function-channel token, a user-transcript update, and 80 ms of 22.05 kHz assistant speech, all on one shared timeline. Silence and overlapping speech are part of the timeline; there is no voice-activity gate.

Checkpoints: `mlx-community/NemotronLabs-VoiceChat-11B-bf16`, `-8bit`, `-4bit`. The quantized conversions quantize only the LLM (embedding table, both heads, Nemotron-H mixer projections); the speech encoder, RNNT branch, EAR-TTS and codec stay bf16.

## Architecture

| Network | Weights | What it does |
|---|---|---|
| Log-mel frontend | none | Preemphasis 0.97, 25 ms / 10 ms symmetric Hann STFT (n_fft 512, reflect-padded), 128 Slaney mel bins, `ln(x + 2^-24)`, no normalization. |
| FastConformer encoder | `stt_model.perception.encoder` | Causal depthwise-striding subsampling (8x, 10 ms mel frames become 80 ms frames), 24 conformer blocks with relative-position attention limited to 70 frames of left context, causal depthwise convolution, LayerNorm conv norm. |
| Perception projection | `stt_model.perception.proj` | 1024 to 4480, the LLM's audio channel. |
| RNNT branch | `stt_model.rnnt_decoder`, `stt_model.rnnt_joint` | Two-layer LSTM prediction net and joint network; greedy decoding over the SentencePiece `rnnt_vocabulary` gives the user transcript. |
| Duplex LLM | `stt_model.embed_tokens`, `stt_model.llm`, `stt_model.lm_head`, `stt_model.function_head` | Nemotron-H (56 layers, Mamba2 / attention / MLP pattern) run on one fused embedding per position: `E(prev_text) + audio + 2 * E(prev_function)`, with a text head and a function head, both greedy. |
| EAR-TTS | `tts_model.tts_model`, `tts_model.audio_prompt_latents.Aria` | 28-layer Gemma-3-style backbone (sliding and global attention, no embedding scaling) conditioned on the current text token through a character-aware subword encoder and gated fusion, with classifier-free guidance (batch 2). A mixture-of-Gaussians head refines 31 residual codebooks in 8 masked iterations. |
| Codec | `tts_model.audio_codec` | ConvNeXt encoder/decoder with probabilistic residual VQ (31 codebooks of 1024), iSTFT output at n_fft 16 / hop 4; one code frame is 1764 samples (80 ms at 22.05 kHz). |

The mlxcel modules are `src/audio/nemotron_mel.rs`, `src/audio/fastconformer/`, `src/audio/rnnt/`, `src/audio/nemotron_codec/`, `src/models/gemma3_backbone.rs`, and `src/models/nemotron_voicechat/` (config, LLM glue, TTS, model loader, offline session). The port follows the mlx-vlm / mlx-audio reference implementation.

## Offline timeline

1. The input WAV is resampled to 16 kHz mono and `--extra-decoding-seconds` of silence are appended.
2. Log-mel and the FastConformer produce one 4480-wide audio embedding per 80 ms frame.
3. A non-empty system prompt becomes `[BOS] + tokens + [EOS]`; those token embeddings occupy the first timeline positions on the audio channel.
4. EAR-TTS is warmed once with the `Aria` prompt latents and the codec encoding of silence.
5. At every position the LLM emits a text and a function token (pad over the prompt prefix); every position after the first advances EAR-TTS by one frame of 31 codes. An EOS text token resets the fed-back codes to silence.
6. The prompt prefix is dropped, control codes are replaced by the per-codebook silence code, and the codec decodes the whole answer. The RNNT branch transcribes the user audio.

The language model keeps persistent Nemotron-H caches across positions. The reference defaults its offline path to recomputing the full history at every position; both modes give identical tokens and codes on the reference, and the cached mode bounds the per-position work.

## CLI

```
mlxcel generate -m models/NemotronLabs-VoiceChat-11B-4bit \
  --audio question.wav --output-audio response.wav \
  -p "Be concise and answer in one sentence."
```

prints `[user] <transcript>`, the assistant text, and `[function] <text>` when the function channel produced any, and writes `response.wav` (22050 Hz mono PCM16). `-p` is optional (no system prompt when omitted), `-n` is ignored because the output length equals the input timeline, `--extra-decoding-seconds` (default 3) sets the appended silence, and `--seed` (default 0) seeds the EAR-TTS sampling noise so two runs with the same seed are byte-identical. `--image` and `--video` are rejected.

## Online session

`NemotronVoiceChatModel::create_streaming_session(StreamingOptions)` returns a `VoiceChatStreamingSession` that turns PCM chunks of any size into 1280-sample (80 ms) frames and advances every network exactly one frame per frame from persistent state:

| Stage | Persistent state |
|---|---|
| Log-mel | the sample tail the centered STFT still needs (frames are emitted once they trail the input edge by 1280 samples) |
| FastConformer | per layer, the last 70 attention inputs and the last 8 conv inputs, plus a 16-frame mel cache for the causal subsampling stack |
| RNNT | last emitted token and LSTM state |
| Nemotron-H | Mamba2 state and attention KV caches |
| EAR-TTS | backbone KV caches (rotating for the sliding layers) |
| Codec | per-ConvNeXt-block causal overlap and the iSTFT overlap |

```rust
let mut session = model.create_streaming_session(StreamingOptions {
    system_prompt: Some("Be concise and answer in one sentence.".into()),
    seed: 0,
    ..StreamingOptions::default()
})?;
for chunk in pcm_chunks {
    for event in session.push_audio(chunk, 16_000)? { handle(event) }
}
for event in session.flush(true)? { handle(event) }
```

Events carry the audio frame index (the system-prompt prefix advances the timeline but emits nothing):

| Event | Fields |
|---|---|
| `AssistantTextDelta` | `frame_index`, `token_id`, `delta`, cumulative `text` |
| `FunctionDelta` | same shape, function channel |
| `UserTranscriptDelta` | `frame_index`, `delta`, cumulative `text` |
| `Audio` | `frame_index`, 1764 `samples` at 22.05 kHz, the frame's 31 `audio_codes` |
| `Done` / `Cancelled` | `frame_index` |

Text deltas skip the pad, silence, BOS and EOS ids; `delta` is the new suffix when the cumulative decode extends the previous one and otherwise the single-token decode, with the cumulative `text` authoritative. `push_audio` rejects a sample rate other than 16 kHz, non-finite samples, and a closed session; `flush(pad_partial)` optionally zero-pads and runs the partial frame, then closes with `Done`; `cancel()` drops pending audio and closes with `Cancelled`. `max_streaming_seconds` bounds a session (`ContextLimit`). A session is driven on one thread (MLX evaluation is thread-affine) and owns all of its state, so sessions created back to back on one model do not interact.

`use_language_cache: false` recomputes the language model over the full fused-input history every frame and `use_perception_cache: false` recomputes log-mel and the encoder over a sliding sample window; both exist for diagnostics, as in the reference.

With `profile: true` the session records per-frame wall-clock stage timings, each taken after forcing evaluation of that stage's outputs: `perception`, `rnnt`, `language`, `tts`, `codec`, `total`. `profile().summary(drop_first)` gives mean / p50 / p95 / max per stage, `processing_frames_per_second = 1000 / mean total`, and `realtime_factor = mean total / 80 ms` (above 1.0 is slower than real time).

The CLI drives the session with `--stream`:

```
mlxcel generate -m models/NemotronLabs-VoiceChat-11B-4bit --audio question.wav \
  --output-audio response_stream.wav -p "Be concise and answer in one sentence." --stream --profile
```

It pushes the input plus `--extra-decoding-seconds` of silence in 80 ms frames, prints assistant text as frames produce it, writes the concatenated audio events, and with `--profile` prints the summary JSON after dropping 5 cold frames. `--max-streaming-seconds` maps to `max_streaming_seconds`.

The streamed encoder equals the offline encoder for the first frames and then drifts slightly, because the reference's cache-aware subsampling window is not the offline full-utterance convolution; the port keeps the reference rule, and the streamed frames match the reference streaming path to float precision.

## Realtime WebSocket (`/v1/realtime`)

`mlxcel-server -m models/NemotronLabs-VoiceChat-11B-4bit --port 8080` (or `mlxcel serve`) loads the checkpoint on a dedicated realtime engine thread and serves the online session over a WebSocket at `/v1/realtime`. The chat worker, the batch scheduler and the text warmup are not started for this checkpoint; `/v1/chat/completions`, `/v1/completions`, `/v1/responses` and `/v1/messages` answer `501 Not Implemented` with a message naming `/v1/realtime`. `--api-key` applies to the upgrade request (send `Authorization: Bearer <key>`), and `--api-prefix` prefixes the path like every other route. `--cors-origins` and `--allowed-origins` also gate the upgrade by its `Origin` header: a browser page whose origin the policy does not allow gets `403 Forbidden`, while a client that sends no `Origin` (a non-browser client) is accepted. The protocol follows the mlx-vlm reference server (`mlx_vlm/server/realtime.py`).

All MLX work runs on the engine thread: the socket task sends `open`, `push`, `flush`, `cancel` and `close` commands over a channel and awaits a one-shot reply, and the engine serializes each event (base64 audio included) before replying. The MLX buffer cache is cleared when a session closes, not per frame.

### Client messages (JSON text frames)

| `type` | Fields | Behavior |
|---|---|---|
| `session.update` | `session.system_prompt` (string; absent uses the checkpoint default, which is empty), `session.seed` (non-negative integer, default 0), `session.max_streaming_seconds` (number, optional), `session.model` (optional, see below) | Opens the streaming session (warms EAR-TTS and prefills the system prompt), answered with `session.updated`. A second `session.update` on a configured connection is an `invalid_request` error. |
| `input_audio_buffer.append` | `audio` (base64 little-endian PCM16 mono), `sample_rate` (default 16000; any other rate is rejected by the session) | Decoded to f32 by `/ 32768` and pushed in 1280-sample slices, one engine call per slice; each slice's events are sent before the next slice runs. |
| `input_audio_buffer.commit` | `pad_partial` (default true) | Answered with `input_audio_buffer.committed`; the session is flushed (the partial frame zero-padded when `pad_partial`), its events end with `response.done`, and the server closes the socket (code 1000). |
| `session.cancel`, `response.cancel` | | Emits `response.cancelled` and closes the socket (code 1000). |
| `session.ping` | | `session.pong`. |

Before `session.update`, every message other than `session.update` and `session.ping` is answered with an `invalid_request` error "send session.update before audio". After it, an unknown `type` is an `invalid_request` error. A frame that is not a JSON object is an `invalid_request` error and the loop continues. The server runs one model, so `session.model` may be omitted; any value opens the served checkpoint and `session.updated` reports the served id (`--alias`, else the model directory name).

### Server events

Every event carries `event_id` (`event_` plus 16 hex digits).

| `type` | Fields |
|---|---|
| `session.created` | `session.id` (`sess_` plus 16 hex digits), `session.state: "configuring"`, `session.input_audio_format {type: "pcm16", sample_rate: 16000}`, `session.output_audio_format {type: "pcm16", sample_rate: 22050}` |
| `session.updated` | `session.id`, `session.state: "ready"`, `session.model`, `session.frame_samples: 1280`, both formats |
| `conversation.item.input_audio_transcription.delta` | `frame_index`, `delta`, cumulative `transcript` |
| `response.text.delta` | `frame_index`, `token_id`, `delta`, cumulative `text` |
| `response.function.delta` | `frame_index`, `token_id`, `delta`, cumulative `text` |
| `response.audio.delta` | `frame_index`, `delta` (base64 PCM16 of `round(clip(x, -1, 1) * 32767)`, 1764 samples), `format: "pcm16"`, `sample_rate: 22050`, `channels: 1`, `audio_codes` (the frame's 31 codec codes) |
| `input_audio_buffer.committed` | |
| `response.done` / `response.cancelled` | `frame_index` |
| `session.pong` | |
| `error` | `error.code`, `error.message` |

Error codes:

| `error.code` | When |
|---|---|
| `invalid_request` | Unknown `type`, a non-object frame, audio before `session.update`, a second `session.update`. |
| `server_busy` | Another connection holds the session; the socket is then closed with code 1013 (try again later). |
| `session_initialization_failed` | `session.update` failed (an invalid field such as a negative seed or a non-positive `max_streaming_seconds`); the connection stays open and may retry. |
| `inference_error` | An append whose `audio` is not valid base64 or has an odd byte count, a non-integer `sample_rate`, a rate other than 16 kHz, a stream past `max_streaming_seconds`, or a model failure. The loop continues; a failed append stops at the failing slice. |

### One session at a time

Server limits protect the single session slot: a WebSocket message or frame may be at most 1 MiB (about 24 s of base64 PCM16; clients send 80 ms appends), a session lasts at most 600 s of audio (`session.max_streaming_seconds` may lower it, larger values are clamped, and the limit ends the stream with `inference_error`), a connection must send `session.update` within 30 s, and a configured session with no client message for 120 s (WebSocket pings do not count) gets an `invalid_request` error and close code 1008.

The engine admits one connection. A second connection while a session is active receives `server_busy` and close code 1013. The reservation is released when the session ends: after `commit` or `cancel` (before the server's close frame), and when the client disconnects, in which case the server cancels the unflushed session first. The next connection is then admitted normally.

### Example

`examples/voicechat_file_client.rs` drives a WAV file through the socket (16 kHz resampling, `--extra-seconds` of silence, default 3, then `commit`), prints the transcript and answer, and writes the received audio as a 22.05 kHz PCM16 WAV:

```
cargo run --release --example voicechat_file_client -- \
  ws://127.0.0.1:8080/v1/realtime question.wav response_ws.wav \
  --system-prompt "Be concise and answer in one sentence." --seed 0
```

An excerpt of a real session with the 4-bit checkpoint (the file client driving "What is the capital of France?" plus 3 s of silence with the system prompt above; client messages prefixed with `>`, `event_id`s and audio payloads elided). A `websocat ws://127.0.0.1:8080/v1/realtime` session exchanges the same frames:

```
{"type":"session.created","session":{"id":"sess_…","state":"configuring","input_audio_format":{"type":"pcm16","sample_rate":16000},"output_audio_format":{"type":"pcm16","sample_rate":22050}}}
> {"type":"session.update","session":{"system_prompt":"Be concise and answer in one sentence.","seed":0}}
{"type":"session.updated","session":{"id":"sess_…","state":"ready","model":"nemotronlabs-voicechat-11b-4bit","frame_samples":1280,"input_audio_format":{"type":"pcm16","sample_rate":16000},"output_audio_format":{"type":"pcm16","sample_rate":22050}}}
> {"type":"input_audio_buffer.append","audio":"…","sample_rate":16000}
{"type":"response.audio.delta","frame_index":0,"delta":"…","format":"pcm16","sample_rate":22050,"channels":1,"audio_codes":[…31 ints…]}
{"type":"conversation.item.input_audio_transcription.delta","frame_index":5,"delta":"What","transcript":"What"}
{"type":"conversation.item.input_audio_transcription.delta","frame_index":19,"delta":" France","transcript":"What is the capital of France"}
{"type":"response.text.delta","frame_index":24,"token_id":…,"delta":"The","text":"The"}
{"type":"response.text.delta","frame_index":30,"token_id":…,"delta":".","text":"The capital of France is Paris."}
> {"type":"input_audio_buffer.commit"}
{"type":"input_audio_buffer.committed"}
{"type":"response.done","frame_index":61}
```

The whole run produced 61 `response.audio.delta` events (frames 0 to 60, 1764 samples each), 6 transcript deltas, 7 text deltas and `response.done`, then the server closed the socket with code 1000. A second connection opened during the session received `server_busy` and close code 1013; a connection opened after the first closed was configured normally.

The session is the same computation as `mlxcel generate --stream` for the same input, system prompt and seed: the transcript, the answer text, the frame count and the audio agree. The two outputs are not byte-identical files because they quantize differently: the CLI's WAV writer scales by 32768 while the wire format scales by 32767, as the reference does, so each sample may differ by one quantization step.

`examples/voicechat_microphone.rs` holds a live exchange: it captures the default (or `--input-device`) microphone, downmixes and resamples it to 16 kHz, sends an append every 80 ms, plays the `response.audio.delta` stream through the default (or `--output-device`) speaker via a ring buffer, prints the transcript and the answer as they arrive, and commits on Ctrl-C. `--list-devices` lists the audio devices. It needs the `voicechat-mic` feature, which compiles the cpal audio-device crate for this example only:

```
cargo run --release --features voicechat-mic --example voicechat_microphone -- \
  ws://127.0.0.1:8080/v1/realtime --system-prompt "Be concise and answer in one sentence."
```

There is no acoustic echo cancellation and the model keeps listening while it speaks, so use headphones; through open speakers it hears its own answer.

On macOS the terminal needs Microphone access (System Settings > Privacy & Security > Microphone); without it opening the input stream fails, and the example reports the device name and this hint.

## Validation

Validated on `mlx-community/NemotronLabs-VoiceChat-11B-4bit` against the mlx-vlm reference on a synthesized "What is the capital of France?" question with the system prompt "Be concise and answer in one sentence." (env-gated tests under `tests/nemotron_voicechat_*_real.rs`, run with `MLXCEL_VOICECHAT_MODEL` and `MLXCEL_VOICECHAT_REF` set):

- The transcript is "What is the capital of France?" and the answer is "The capital of France is Paris.", as in the reference.
- Text and function ids match the reference exactly at every timeline position, and with `--seed 0` the 31 EAR-TTS codes of every frame match the reference too (the port draws the same MLX global-RNG sequence), so the decoded answer audio matches the reference waveform.
- Transcribing the generated answer audio back through the same checkpoint yields "The capital of France is Paris".
- The FastConformer output matches the reference within 1e-6 (max abs); the codec reproduces the reference codes exactly and its decode within 1e-7.

The codec is a lossy neural codec: a pure tone round-trips with a waveform SNR of about 0.7 dB in the reference too, so round-trip parity is checked against the reference reconstruction rather than an SNR threshold.

## Performance

Measure with `mlxcel generate ... --stream --profile` (the `realtime_factor` of the summary, defined as mean per-frame processing time divided by the 80 ms frame; above 1.0 means slower than real time). The table comes from an otherwise idle validation machine: the "What is the capital of France?" prompt with 3 s of extra decoding (56 frames, the first 5 dropped as cold), one warm-up run, then three measured runs per checkpoint. The three runs agreed within 0.01.

| Checkpoint | Real-time factor | Frame time p50 / p95 | Machine |
|---|---|---|---|
| 4-bit | 1.04 | 82.7 ms / 83.8 ms | Apple M1 Ultra (128 GB), Metal |
| 8-bit | 1.12 | 89.3 ms / 90.7 ms | Apple M1 Ultra (128 GB), Metal |

Neither checkpoint keeps up with real time on this machine: each 80 ms frame takes slightly longer than 80 ms, so a live session falls behind by about 4% (4-bit) or 12% (8-bit) of the elapsed audio. Per-stage p50 for the 4-bit checkpoint: perception 22.7 ms, RNNT 0.4 ms, language 16.1 ms, TTS 36.0 ms, codec 7.4 ms; the 8-bit checkpoint differs only in the language stage (22.4 ms). TTS is the largest stage and the first place to look for speedups.

## Limits

- Only the checkpoint's built-in `Aria` voice; a `speaker` other than `Aria` is rejected at load.
- Batch size 1.
- Offline turns are capped at 20 minutes of input (plus at most 600 s of `--extra-decoding-seconds`): the encoder builds a dense attention mask over the whole utterance, as the reference does.
- No chat surface: interactive `mlxcel generate` without `--audio` and `mlxcel chat` refuse the checkpoint with a pointer to the offline command, and `mlxcel serve` serves only the `/v1/realtime` WebSocket, one session at a time.
- The converted MLX safetensors layout only; the original NeMo `.nemo` checkpoint is not loaded.
