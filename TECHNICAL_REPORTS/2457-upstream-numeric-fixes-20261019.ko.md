# PR #2457: Gemma 3n, PhiMoE, Gemma 4 audio, Phi-3-small의 upstream 수치 수정 이식

**작성일**: 2026-10-19
**상태**: 네 계열 모두 실제 체크포인트로 CUDA (GB10)에서 구현 및 검증 완료, PR 열림

## 요약

#2379를 닫는다. mlxcel이 지원하는 계열에 upstream 수치 수정 네 가지를 이슈 요구대로 한 PR에서 이식한다. Gemma 3n KV-shared RoPE offset, PhiMoE sparsemixer 라우팅과 LongRoPE, Gemma 4 audio attention mask, Phi-3-small padding 토큰 억제다.

## 문제 정의

- **Gemma 3n** (ml-explore/mlx-lm#1886): KV-shared 레이어가 query를 owner cache의 offset으로 회전시켰다. 그런데 owner는 같은 forward 안에서 offset을 이미 앞당긴 뒤였다. 그래서 다중 토큰 forward마다 shared 레이어의 query가 chunk 하나만큼 앞서 있었다.
- **PhiMoE** (ml-explore/mlx-lm#1939, #1889, Blaizzy/mlx-vlm#2419): 라우팅이 기준 구현의 sparsemixer 대신 top-k logit에 softmax를 썼다. 또 체크포인트의 `longrope` scaling 블록을 무시해서 긴 프롬프트가 일반 RoPE로 실행되었다.
- **Gemma 4 audio** (Blaizzy/mlx-vlm#2385): causal-valid mask가 `dist == max_past`까지 허용해, 의도한 왼쪽 context보다 한 위치를 더 보았다.
- **Phi-3-small** (ml-explore/mlx-lm#1938): `vocab_size` 아래의 padding id 91개가 샘플링될 수 있었다.

## 기술적 결정

- **Gemma 3n offset은 한 번만 읽음.** RoPE offset을 forward마다 첫 owner cache에서 한 번 읽고, `DecoderLayer`를 거쳐 `Gemma3nAttention::forward`로 넘긴다. 그래서 owner 레이어와 shared 레이어가 같은 위치로 회전한다.
- **PhiMoE는 Phi-3 LongRoPE 코드를 사용.** `sparsemixer`가 top-k softmax를 대체한다. `longrope` 블록은 새 `pub(crate) SuRope::from_scaling`을 통해 `phi3::SuRope`를 거친다. rotary-prefix scale은 `SuRopeTable`로 옮겼고, `rope_table_regime`이 prompt cache를 short/long 테이블 기준으로 구분한다. `rope_scaling`이 null이면 일반 RoPE를 유지한다. `num_experts_per_tok > num_local_experts`와 `num_experts_per_tok == 0`은 로드 시점에 실패한다.
- **fused decode는 score를 그대로 받음.** fused decode kernel(`run_fused_moe_two_kernel`)은 score를 f32로 바꿔 곱할 뿐 다시 정규화하지 않는다. `mlx_cxx_kernels.cpp:1491` 근처의 정규화 코드는 다른 진입점인 DeepSeek `fused_moe_forward`의 것이다. score 합이 1.5025일 때 fused 출력이 graph 경로와 같음을 테스트로 고정했다.
- **Gemma 4 audio mask는 거리로 계산.** `causal_valid_mask`가 `dist = q + max_past - c`로 mask를 만들어 `dist == max_past`를 제외한다.
- **Phi-3-small 억제는 bias로 적용.** `output_suppressed_token_ids`가 padding id 91개를 돌려주고 sampler가 이를 적용한다. forward logit은 그대로 두므로 trace와 scoring에는 영향이 없다.
- **logit_trace의 audio arm.** audio 수정을 trace할 수 있도록 `logit_trace`에 `MLXCEL_TRACE_AUDIO`와 `MLXCEL_TRACE_AUDIO_PROMPT`를 추가했다.

## 변경 사항

- `src/models/gemma3n.rs`, 새 `src/models/gemma3n_tests.rs`: offset 전달과 chunking 테스트. xla-diagnostics 호출 지점도 새 시그니처에 맞게 고쳤다.
- `src/models/phimoe.rs`, `src/models/phimoe_tests.rs`, `src/models/phi3.rs`: sparsemixer, LongRoPE, config 검증.
- `src/audio/encoder.rs`, `src/audio/encoder_tests.rs`: mask.
- `src/models/phi3small.rs`, `src/models/phi3small_tests.rs`: 억제 id.
- `examples/logit_trace/`: audio arm. 문서와 CHANGELOG.

## 검증

GB10, `--profile test-fast --features cuda`, `MLX_ENABLE_TF32=0`. 테스트 바이너리는 `gpu-lock` 아래에서 `--test-threads=1`로 실행했다. `models::phimoe`, `models::gemma3n::tests`, `audio::encoder`, `models::phi3`(phi3small 포함)에서 42/42 통과했다. Gemma 3n chunking 테스트와 mask predicate 테스트는 main에서 실패한다. clippy(`--lib --tests`와 `--example logit_trace`)와 fmt도 깨끗하다.

mlxcel의 Gemma 3n MLP와 AltUp 입력은 bf16으로 반올림된다. 그래서 12행을 한 번에 처리한 결과가 더 좁게 나눈 결과와 bf16 ulp 하나만큼 다르다(fixed 2.7e-3, reverted 2.7e-2~6.7e-2). 여러 번으로 나눈 경우는 1e-3(fixed 1e-6, reverted 최소 5.6e-3), 한 번 처리한 경우는 1e-2로 기준을 잡았다. 이슈의 허용 오차와 다른 부분이며, CUDA에서만 측정했다.

실제 체크포인트 trace, fixed 대 reverted, `--decided 2`:

| 체크포인트 | 폭 | Context | Decided 불일치 | PPL fixed / reverted |
|---|---|---|---|---|
| gemma3n-e2b-4bit | 1 | 512 | 0/151 | 26.44 / 26.20 |
| gemma3n-e2b-4bit | 5 | 512 | 3/177 (1.7%) | 28.48 / 27.36 |
| gemma3n-e2b-4bit | 32 | 512 | 14/334 (4.2%) | 31.75 / 33.09 |
| phi-3.5-moe-4bit | 5 | 512 | 2/141 (1.4%) | 7.91 / 9.62 |
| phi-3.5-moe-4bit | 1 | 512 | 1/48 (2.1%) | 15.65 / 19.50 |
| phi-3.5-moe-4bit | 5 | 4200 (long table) | 118/136 (86.8%) | 7.70 / 3008.7 |
| gemma-4-e2b-it-4bit + audio | prefill | audio 토큰 100개 | 0/78 | 1.657 / 1.657 |

Gemma 3n 폭 1에서 decided 불일치가 없는 이유는 shared 레이어가 KV를 쓰지 않기 때문이다. 이 경우 오류는 30개 레이어 중 10개에서 query가 한 위치 밀리는 것뿐이고, 폭이 커질수록 영향이 커진다. audio 클립은 결정적으로 생성한 4초짜리 합성 chirp, tone, noise WAV다. 두 arm은 다르지만(최대 logit 차이 0.875) decided 위치에서는 같다. Phi-3.5-MoE에서 `mlxcel generate`는 128 토큰까지 자연스럽다. `phi-3-small-8k-instruct-aq4_64`(mlx-community Phi-3-small-8k-instruct-4bit 대체)에서는 억제 bias를 거쳐 256 토큰을 샘플링했고, 목록에 있는 id는 하나도 나오지 않았다. Phi-3.5-MoE(42B)의 transformers float32 기준값은 이 호스트에서 만들 수 없다.

## 후속 작업

- `mlxcel-xla`의 `gemma3n_emit_ops.rs`는 이전의 shared 레이어 query offset을 그대로 따르므로, MLX 대 XLA Gemma 3n parity가 이제 어긋난다.
- `phimoe::ModelArgs`는 `layer_norm_eps`(기본 1e-6)를 읽지만 체크포인트는 `rms_norm_eps: 1e-5`로 표기한다.
- `Gemma3nModel`의 #674 chunked-prefill opt-out은 이 버그 때문이었을 수 있다. 되돌리려면 parity 검증 전체가 필요하다.

## 통계

| 항목 | 값 |
|------|-------|
| 변경 파일 | 14 |
| 추가 줄 | 1200 |
| 삭제 줄 | 135 |

| 커밋 | 메시지 |
|--------|---------|
| `3e942576` | fix(models): port upstream numeric fixes for four model families |
| `63a22410` | test(models): make the Gemma 3n chunking test robust to bf16 rounding |
| `ad4c60ae` | test(models): widen the Gemma 3n multi-pass tolerance and log the fix |
| `9374c11c` | docs: note EmbeddingGemma 2 in the Gemma 4 audio mask entry |
| `8778f29c` | fix(models): reject zero experts per token in PhiMoE configs |

## 관련 항목

- ml-explore/mlx-lm#1886, #1939, #1889, #1938; Blaizzy/mlx-vlm#2419, #2385.
