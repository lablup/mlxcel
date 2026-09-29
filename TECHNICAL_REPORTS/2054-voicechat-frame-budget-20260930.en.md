# Technical Report: PR #2054 - VoiceChat streaming under the 80 ms frame budget

**Date**: 2026-09-30

**Status**: Implemented and validated on the Apple M1 Ultra with both the 4-bit and 8-bit checkpoints; pending merge.

**Languages**: Rust

**Risk Level**: Low. Output is byte-identical to the previous binary; the only runtime cost is about 2.7 GB of extra resident memory.

## Executive Summary

Issue #2045 asked for a Nemotron VoiceChat `realtime_factor` below 1.0 on an idle M1 Ultra. Before this PR each 80 ms frame took about 82 ms (4-bit) or 88.6 ms (8-bit). After it, a frame takes 69.0 ms (4-bit, factor 0.866) or 75.3 ms (8-bit, factor 0.942). The whole gain comes from one exact change: the encoder, TTS backbone and MoG head weights that only meet f32 activations are cast to f32 once at load, instead of MLX casting them inside every op on every frame. The PR also adds host-sync counting and opt-in sub-stage profiling, which is how the cause was found.

## 1. Problem Statement

`--stream --profile` reported only five stage totals (perception 22.5 ms, RNNT 0.4, language 16.0, TTS 35.6, codec 7.3 on 4-bit), and the issue's candidate list was ranked by estimate. The issue asked for profiling first, and for exact (output-preserving) changes before any numerics change.

## 2. Change Summary

- `src/audio/stage_probe.rs` (new): a thread-local per-frame host-sync counter and named sub-stage timers. `FrameTiming` and `ProfileSummary` gain `host_syncs` and an optional `sub_stages` map. `StreamingOptions::profile_stages` (CLI: `MLXCEL_VOICECHAT_PROFILE_STAGES=1`) turns the timers on; `MLXCEL_VOICECHAT_PROFILE_FRAMES=<path>` dumps per-frame timings.
- `src/audio/f32_weights.rs` (new): `promoted_subset` returns the weights under a prefix with the selected bf16/f16 entries cast to f32 and evaluated.
- `VoiceChatPerception::from_weights` promotes every encoder and `proj` weight. `RvqEarTtsModel::from_weights` promotes, through `promotes_to_f32`, the backbone attention and MLP projections, the MoG-head MLP projections and output projections, `embed_code` and the fusion `audio_proj`.
- `docs/nemotron-voicechat.md` Performance section: new numbers, the steady-state run, the memory cost and the profiling switches.

## 3. Technical Decisions

### Why the cast was the bottleneck

The sub-stage run (posted on the issue) put the 24 conformer layers at 25.7 ms, the TTS backbone at 24.7 ms and the five MoG-head passes at 12.0 ms. All three run f32 activations against bf16 weights, as the reference does. In MLX, `matmul`, `addmm`, `conv_general`, `fast::layer_norm` and `fast::rms_norm` promote by inserting `astype(weight, out_type)` into the graph. Each frame therefore read 1.22 GB of encoder weights, 1.19 GB of backbone weights and five times 0.32 GB of MoG weights, wrote twice that as f32, and read the f32 copy again. Once the cast is gone, perception drops from 22.5 to 17.0 ms and TTS from 35.6 to 28.1 ms.

### Why the change is exact

bf16 to f32 conversion is lossless. A pre-cast weight hands the same f32 values, in the same layout (a transposed view of a row-contiguous array stays column-contiguous either way), to the same kernel. Unit tests compare the promoted and pre-cast graphs byte for byte for matmul, addmm, conv1d (dense and depthwise) and layer_norm, and an end-to-end test does the same for a bf16 perception module. On the real checkpoints the `--stream --seed 0` WAV md5 is unchanged: 4-bit `aad28bd67e52113444f52a2b4b49e2d0`, 8-bit `3223cf3627a874122fb2bb94b5b81ee8`.

### What stays bf16

A weight can be promoted only when every op that reads it promotes against an f32 activation. Gemma's RMSNorm builds `1 + weight` in the weight dtype, so every backbone norm, `q_norm` and `k_norm` stays bf16. The subword condition runs in bf16, so `embed_subword` and the fusion `text_proj` stay as stored. `proj_mus` and `low_mat` are gathered row by row and cost little per call, so they are left alone as well. `promotes_to_f32` encodes this list, and a unit test pins it against the real key names.

### Where the cast lives

The cast happens inside the component loaders, not in `NemotronVoiceChatModel::load`. That way the real-weight parity tests that build components directly (`tts_real`, `front_real`, `stream_real`) exercise the promoted path. It also leaves `gemma3_backbone.rs` and `gemma3.rs` untouched, so the Gemma 3 text model cannot change.

### Why the other candidates were not applied

The issue says to stop adding risk once the target is met. After this change the 4-bit factor is 0.866 and the 8-bit factor 0.942, which also meets the 8-bit stretch goal. The remaining exact candidates (T1 condition memoization, T2/T3 RVQ gathers, P2/P3 positional caches, C1/L1 sync merges) are worth about 1 to 2 ms each by the sub-stage numbers. They are left for a follow-up if the margin is needed on smaller machines.

## 4. Validation

M1 Ultra (128 GB), alone on the machine, `mlxcel generate -m <ckpt> --audio question.wav --stream --profile --seed 0`, 56 frames with 5 dropped as cold, 1 warm-up plus 3 measured runs, 20 s between runs.

| Checkpoint | Build | Runs (realtime_factor) | Frame p50 | perception | language | tts | codec |
|---|---|---|---|---|---|---|---|
| 4-bit | baseline (ac02b2dc) | 1.0313 / 1.0287 / 1.0300 | 82.2 ms | 22.5 | 16.0 | 35.6 | 7.3 |
| 4-bit | this PR | 0.8650 / 0.8673 / 0.8662 | 69.0 ms | 17.0 | 16.0 | 28.1 | 7.3 |
| 8-bit | baseline | 1.1105 / 1.1114 / 1.1083 | 88.7 ms | 22.5 | 22.2 | 35.8 | 7.4 |
| 8-bit | this PR | 0.9411 / 0.9435 / 0.9422 | 75.3 ms | 16.9 | 22.2 | 28.1 | 7.3 |

RNNT is 0.44 ms in every run. The instrumentation-only commit measured 1.0312 / 1.0301 / 1.0306, so the counters cost nothing measurable. Host syncs stay at 8.2 per frame.

The long run used 12 s of extra decoding (168 frames, past the 71-frame attention window). The baseline measured 1.036 over all frames. This PR measures 0.871 over all frames, and frames 71 and later (102 frames) average 69.8 ms, a factor of 0.872.

Tests:

- `cargo test --release --lib -- models::nemotron_voicechat audio::`: 219 passed.
- Real-weight tests with `MLXCEL_VOICECHAT_MODEL` and `MLXCEL_VOICECHAT_REF`: `streaming_real`, `tts_real`, `codec_real`, `offline_real` and `front_real` pass. `llm_real` and `stream_real` skip because the `MLXCEL_VOICECHAT_PADDED` and `MLXCEL_VOICECHAT_STREAM_REF` dumps are not available on this machine.
- Qwen3-4B 4-bit greedy output (60 tokens, `--show-reasoning`) is identical to the baseline binary.
- fmt, clippy (`-p mlxcel --lib --tests --examples -D warnings`), `cli_help_consistency`, `dead_doc_pointers`, `llama_compat_manifest` and `realtime_ws` pass.

## 5. Known Limitations

- Resident memory grows by about 2.7 GB. That is small next to the 9 GB checkpoint on a 128 GB machine, but it matters on 16 to 24 GB Macs.
- The numbers come from one machine. The margin (0.87 / 0.94) is comfortable here, but a slower Apple Silicon part may still miss real time with the 8-bit checkpoint.
- The baseline steady-state figure is the whole-run summary (1.036), because the baseline binary predates the per-frame dump.
- Sub-stage timings force evaluation at every boundary, so they explain the split but their totals are not the real-time factor.
