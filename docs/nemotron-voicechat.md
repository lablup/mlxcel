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

## Validation

Validated on `mlx-community/NemotronLabs-VoiceChat-11B-4bit` against the mlx-vlm reference on a synthesized "What is the capital of France?" question with the system prompt "Be concise and answer in one sentence." (env-gated tests under `tests/nemotron_voicechat_*_real.rs`, run with `MLXCEL_VOICECHAT_MODEL` and `MLXCEL_VOICECHAT_REF` set):

- The transcript is "What is the capital of France?" and the answer is "The capital of France is Paris.", as in the reference.
- Text and function ids match the reference exactly at every timeline position, and with `--seed 0` the 31 EAR-TTS codes of every frame match the reference too (the port draws the same MLX global-RNG sequence), so the decoded answer audio matches the reference waveform.
- Transcribing the generated answer audio back through the same checkpoint yields "The capital of France is Paris".
- The FastConformer output matches the reference within 1e-6 (max abs); the codec reproduces the reference codes exactly and its decode within 1e-7.

The codec is a lossy neural codec: a pure tone round-trips with a waveform SNR of about 0.7 dB in the reference too, so round-trip parity is checked against the reference reconstruction rather than an SNR threshold.

## Performance

Measure with `mlxcel generate ... --stream --profile` (the `realtime_factor` of the summary). The table is filled in from a run on an otherwise idle validation machine.

| Checkpoint | Real-time factor | Machine |
|---|---|---|
| 4-bit | TBD (to be measured) | TBD |
| 8-bit | TBD (to be measured) | TBD |

## Limits

- Only the checkpoint's built-in `Aria` voice; a `speaker` other than `Aria` is rejected at load.
- Batch size 1.
- Offline turns are capped at 20 minutes of input (plus at most 600 s of `--extra-decoding-seconds`): the encoder builds a dense attention mask over the whole utterance, as the reference does.
- No chat surface: interactive `mlxcel generate` without `--audio`, `mlxcel chat`, and `mlxcel serve` refuse the checkpoint with a pointer to the offline command. The `/v1/realtime` WebSocket session is tracked in #1376.
- The converted MLX safetensors layout only; the original NeMo `.nemo` checkpoint is not loaded.
