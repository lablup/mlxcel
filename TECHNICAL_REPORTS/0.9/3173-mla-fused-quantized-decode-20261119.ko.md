# 기술 보고서: PR #3173 - fused Metal decode 커널이 int8 MLA latent를 직접 읽도록 변경

**날짜**: 2026-11-19
**상태**: 완료
**위험도**: 낮음 (Metal, int8 latent, 2048 key 이하 window에만 적용되고 나머지는 #3155 composed 경로를 그대로 씀)

## 요약

fused absorbed-decode Metal 커널에 #3155의 affine int8 MLA latent를 window dequantize 없이 바로 읽는 두 번째 변형을 추가했다. M3 Ultra에서 DeepSeek-V2-Lite의 int8 decode는 이 커널로 512 토큰 뒤에서 composed int8 경로보다 5.9%, 1792 토큰 뒤에서 1.5% 빠르다. 약 2048 key를 넘으면 composed 경로가 더 빨라서 `FUSED_DECODE_QUANTIZED_MAX_KEYS`로 커널을 2048 key까지만 쓴다. teacher-forced trace에서 composed int8 경로와 fp16 latent 모두에 대해 decided disagreement는 0건이다.

## 1. 문제

#3155는 V2 latent를 int8로 저장하고 `quantized_matmul`로 decode했다. 이 연산은 latent head 하나를 쿼리 head 16개에 broadcast하므로 head마다 packed window 전체를 읽는다. fp16 latent는 모든 head에 대해 window를 한 번 읽는 fused 커널(#3121)로 이 비용을 피하는데, `fused_absorbed_decode`는 양자화 window를 받지 않았다. 그래서 512 토큰 뒤에서 int8 latent decode가 fp16보다 15% 느렸다.

## 2. 변경 내용

- `src/lib/mlxcelverse/kernels/mla_absorbed_decode.{cpp,h}`: 기존 커널 템플릿의 두 번째 인스턴스다. latent를 읽는 부분만 다른데, packed code를 바이트로 읽고 그룹별 scale과 bias를 score와 value 합 밖으로 빼서 곱한다. rope 절반, softmax, value 합, merge는 공유한다. dense 커널 소스는 한 글자도 바뀌지 않았다.
- Bridge: `mlx_cxx_bridge.{cpp,h}`와 `mlxcel-core/src/lib.rs`의 `#[cxx::bridge]` 블록에 `mla_absorbed_decode_fused_quantized`와 `_available` 확인 함수를 추가했다. 커널이 거부하면 오류를 돌려주고 스텝을 중단하지 않는다.
- `mla/decode.rs`: `fused_absorbed_decode_quantized`는 마스크 없는 단일 쿼리, int8, group size 64, 64에서 1024 사이 2의 거듭제곱 rank, 1에서 2048 key를 받고 `AbsorbedFused`를 기록한다. #3155의 양자화 decode가 이 함수를 먼저 부르고 `None`이면 composed 경로로 넘어간다. `MLXCEL_MLA_FUSED_DECODE=0`은 두 fused 변형을 모두 끈다.
- 테스트: `mla/fused_quant_decode_tests.rs`가 1, 255, 2048, 8192 key에서 f32, fp16, bf16의 fused 대 composed 일치와 거부 사례(4비트, group size 32, rank 192, 한도 초과, 쿼리 두 개)를 다룬다.
- 문서: `docs/mla-absorbed-decode.md`, `docs/environment-variables.md`, CHANGELOG, CSV 네 개가 딸린 `docs/benchmark_results/mla-fused-quantized-decode-m3ultra-2026-11-19.md`.

## 3. 기술적 결정

**원소마다 dequantize하지 않고 scale과 bias를 합 밖으로 뺐다.** affine 그룹에서는 `sum(q_i * (s * c_i + b)) = s * sum(q_i * c_i) + b * sum(q_i)`가 성립한다. 커널은 그룹마다 정수 code 내적과 쿼리의 그룹 합을 한 번씩 누적한 뒤 scale과 bias를 적용한다. 원소당 작업이 바이트 로드 하나와 곱셈-덧셈 하나로 줄고, window의 fp16 사본은 만들지 않는다.

**window 한도는 단독 커널이 아니라 모델 안 측정으로 정했다.** 커널만 따로 재면 약 6000 key까지 composed 경로보다 빠르다. 모델 안 decode는 512 뒤에서 +6.5%, 2048 뒤에서 +0.4%, 3840 뒤에서 -4.6%였으므로 한도를 2048로 두었다. 단독 벤치마크에는 스텝마다의 quantize-and-store와 레이어의 나머지 연산이 빠져 있고, 두 교차점이 왜 다른지는 더 분리해 보지 않았다.

**비트 폭 분기는 남기되 4는 거부한다.** #3155가 int8만 넣었기 때문에 커널은 `bits` 인자를 받고 8 이외 값은 거부한다. 나중에 4비트 latent를 넣으려면 unpack 경로와 별도 측정만 더하면 된다.

## 4. 측정

M3 Ultra, Metal, deepseek-v2-lite-chat-4bit-mlx, 128 토큰, tok/s 중앙값. 세 조건을 한 GPU lock 안에서 번갈아 돌렸다.

| 프롬프트 | int8 fused (신규) | fp16 fused | int8 composed (#3155) | 신규 / composed |
|---|---|---|---|---|
| 512 | 148.43 | 164.64 | 140.16 | +5.9% |
| 1792 | 132.82 | 149.43 | 130.90 | +1.5% |
| 2048 | 128.82 (window가 2048을 넘어 composed) | 146.72 | 129.14 | -0.2% |
| 16384 | 71.08 (composed) | 109.65 | 71.12 | -0.1% |

fused int8을 후보로 둔 `--decided 2` trace 결과다. composed int8 대비 1024 뒤에서 decided 205개 중 0개(top-1 뒤집힘 512개 중 6개, 최대 차이 0.078), 1920 뒤에서 113개 중 0개(256개 중 3개, 0.109)였다. fp16 latent 대비로는 1024 뒤 209개 중 0개(512개 중 10개, 0.078), 1920 뒤 115개 중 0개(256개 중 5개, 0.078)다. 4096 뒤에서는 두 조건 모두 composed 경로를 타서 trace가 바이트 단위로 같다. 1024와 4096의 fp16 trace는 base 커밋 빌드와 바이트 단위로 같다.

fused 대 composed 일치의 최대 상대 오차는 f32 5e-7, fp16 5.6e-4, bf16 6.25e-3으로, 각각의 허용치 1e-5, 2e-3, 1.6e-2 안에 있다.

## 5. 배운 점

**단독 커널의 교차점은 모델 안 교차점보다 크게 나온다.** 단독 측정은 교차점을 약 6000 key로, 모델은 약 2048로 보여 줬다. 원인은 분리하지 않았기 때문에, 상수는 사용자가 실제로 돌리는 decode 벤치마크에서 정했다.

**fp16과의 남은 차이는 attention에 있지 않다.** 512 토큰에서 int8 fused는 148 tok/s, fp16은 165 tok/s이고 attention 커널끼리는 비슷하다. 차이는 #3155가 쓰기마다 더한 quantize-and-store 단계에서 나온다.

## 6. 후속 작업

- Metal에서 int8만 주고 `MLXCEL_MLA_FUSED_DECODE=0`을 걸면 #3131 기본값 규칙 때문에 `absorbed_composed`가 아니라 decompressed 경로가 선택된다. composed로 가려면 `MLXCEL_MLA_ABSORBED=1`도 필요하다. 이 규칙은 이번에 바꾸지 않았다.
- M3 Ultra에서 fp16 커널의 단독 교차점은 약 4096 key로 보이며, M5 Max에서 잰 8192와 다르다. 모델 안에서는 재지 않았고 `FUSED_DECODE_MAX_KEYS`는 그대로다.
- #3160-#3163이 latent-pair 계열에 int8 latent를 넣으면 같은 경로로 이 커널을 쓸 수 있다.
- CUDA와 ROCm에는 두 latent 모두 fused 커널이 없고 측정하지 않았다.
