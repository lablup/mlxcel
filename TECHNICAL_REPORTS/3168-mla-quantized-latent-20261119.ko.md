# 기술 보고서: PR #3168 - DeepSeek V2의 absorbed MLA latent를 affine int8로 저장

**날짜**: 2026-11-19
**상태**: 완료
**위험도**: 중간 (opt-in 플래그 뒤의 새 캐시 레이아웃이고, fp16 기본 경로는 바뀌지 않음)

## 요약

absorbed decode를 쓰는 DeepSeek V2에서 `--kv-cache-mode int8`을 주면 이제 MLA latent를 affine group-64 8비트 값과 fp16 rope key로 저장한다. V2-Lite 기준 토큰당 레이어당 672바이트로, fp16 latent의 1152바이트, 이 경로가 대체한 decompressed int8 캐시의 5120바이트보다 작다. M3 Ultra에서 decode는 decompressed int8보다 1.15배에서 2.78배 빠르고 fp16 latent보다 13%에서 36% 느리며, teacher-forced trace에서 fp16 대비 decided disagreement는 0건이었다. int4는 같은 게이트를 통과하지 못해 넣지 않았다.

## 1. 문제

#3155 이전에는 `MlaLatentCache`가 dense `(ckv, kpe)` 텐서만 담을 수 있어서, absorbed V2 모델에 양자화 캐시를 요청하면 int8은 decompressed per-head 레이아웃으로 돌아가고 Turbo와 int4는 fp16 latent로 치환됐다. 메모리를 아끼려고 int8을 고르면 오히려 fp16 latent 기본값보다 4.4배 큰 캐시가 만들어진 셈이다. epic #3158의 이후 단계(#3156 fused Metal 커널, #3157 Turbo WHT 평가, #3160-#3163 latent-pair 계열)가 이어받으려면 양자화 latent 저장 타입과 window 계약이 먼저 있어야 했다.

## 2. 변경 내용

- `mlxcel-core/src/mla/quant_latent.rs`에 양자화 latent window, 한 쿼리용 composed decode, 크기 계산 함수 `quantized_latent_bytes_per_token`을 추가했다. 이 함수는 바이트 수가 오버플로하는 rank에 대해 `usize::MAX`를 돌려준다.
- `MlaLatentCache::update_and_fetch`는 `LatentWindow::{Dense, Quantized { ckv_q, scales, biases, kpe, bits, group_size }}`를 돌려준다. Quantized는 per-head INT8 캐시의 버퍼 네 개를 그대로 쓰기 때문에 `trim`, `trim_front`, gather, detach/adopt, prompt-cache 보관이 별도 관리 코드 없이 처리한다.
- Decode: 쿼리가 하나면 latent score와 value 합을 packed row 위에서 `quantized_matmul`로 계산하고, rope score는 쿼리 head를 matmul의 row 축에 놓는다. 여러 토큰 스텝과 cold prefill은 스텝마다 window를 한 번 dequantize한다. fused 커널과 split-KV에는 양자화 window가 들어가지 않는다.
- 모드 결정: absorbed V2의 Turbo 모드는 계속 fp16 latent로 치환되고, 경고 문구가 이제 `int8`을 안내한다. `MLXCEL_MLA_ABSORBED=0`이면 decompressed int8 캐시를 그대로 쓴다. DeepSeek-VL2는 언어 모델 쪽 `deepseek_v2` 설정으로 KV 규칙을 정한다. 이런 모델에서 `--kv-group-size`를 64 이외 값으로 주면 서버 시작이 실패한다.
- 크기 계산: `kv_arch`, `inspect`, 메모리 사전 점검, prompt-cache 예산, KV advisor가 672바이트 비율을 쓴다.
- 문서: `docs/turbo-kv-cache.md`의 "MLA latent caches are FP16 only" 절을 int8 결정, int4 수치, latent-pair 후속 작업으로 다시 썼다. `docs/mla-absorbed-decode.md`, `docs/environment-variables.md`, `CHANGELOG.md`를 갱신했고, 측정값은 실행별 CSV와 함께 `docs/benchmark_results/mla-quantized-latent-m3ultra-2026-11-19.md`에 남겼다.

## 3. 기술적 결정

**스텝마다 dequantize하지 않고 `quantized_matmul`로 조합했다.** decode 쿼리 하나는 window를 한 번 읽는다. 먼저 dequantize하면 매 스텝 window 전체의 fp16 사본을 쓰고 다시 읽어야 해서, 이 포맷으로 아끼려던 대역폭을 그대로 다시 쓰게 된다. `quantized_matmul`과 `dequantize`는 백엔드에 묶이지 않은 MLX 연산이라 Metal 전용 심볼 없이 CUDA와 ROCm에서도 컴파일된다. 여러 토큰 스텝은 원래 쿼리 row가 많은 matmul을 치르므로 한 번 dequantize하고 기존 dense attention 코드를 쓴다.

**window 계약은 일반형으로 두었다.** `LatentWindow::Quantized`는 V2의 8비트와 64를 가정하지 않고 `bits`와 `group_size`를 들고 다닌다. #3156과 #3160-#3163이 같은 타입을 쓰기 때문이다. decode는 비트 폭을 window에서 읽고, 이를 단위 테스트로 확인한다.

**int8만 넣고 int4는 넣지 않았다.** 게이트 기준은 decided disagreement 0건, 그리고 모든 top-1 뒤집힘이 근접 동률(top-two 차이 0.11 logit 이하)에서만 일어나는 것이다. int8은 1024 뒤에서 최대 차이 0.078, 4096 뒤에서 0.109로 기준을 지켰다. int4는 top-1 뒤집힘이 512개 중 45개, 256개 중 14개였고 최대 차이가 0.859였다. 그래서 `KVCacheMode::Int4`와 `(Uniform, 4)` 매핑은 추가하지 않았고 `--kv-bits 4`는 시작 시 오류를 그대로 낸다.

**수치가 똑같은 decode 레이아웃만 반영했다.** 보안·성능 리뷰에서 한 쿼리 레이아웃 세 가지를 측정했다. rope score에서 head를 row 축에 놓는 방식은 trace가 바이트 단위로 같고 1%에서 1.5% 빨라져서 반영했다. head를 latent score에 접어 넣는 방식은 512와 2048에서 3%, 16384에서 13% 빨랐지만 합산 순서가 바뀌었고, top-1 뒤집힘 하나가 0.11 기준 밖인 0.219 logit 차이에서 나왔다. 이 결과는 기록만 하고 기준을 어떻게 할지 결정을 기다린다. value 합까지 접어 넣으면 모든 길이에서 더 느렸다(16384에서 71 대비 38.5 tok/s).

**크기 계산 값에 상한을 두었다.** `kv_arch`는 `config.json`의 `kv_lora_rank`를 그대로 넘기는데 `rank * bits`에 오버플로 검사가 없었다. 64의 배수인 조작된 rank를 주면 debug 빌드에서는 panic하고 release 빌드에서는 작은 값으로 감싸져 `inspect`, 메모리 추정, prompt-cache 예산을 실제보다 작게 계산했다. 이제 결과는 최댓값에서 멈추고, 런타임 shape 검사(`MlaGeometry::check`, `check_quantizable`)가 할당 전에 그런 rank를 거부하는 것도 그대로다.

## 4. 측정

M3 Ultra, Metal, deepseek-v2-lite-chat-4bit-mlx, `mlxcel-bench-decode`, warmup 16 뒤 128 토큰, tok/s 중앙값. 모든 조건을 한 GPU lock 안에서 번갈아 돌렸고 인덱서는 멈춰 두었다.

| 프롬프트 | fp16 latent | int8 latent | int8 decompressed | int8 latent / decompressed |
|---|---|---|---|---|
| 512 | 163.59 | 138.85 | 120.97 | 1.15배 |
| 2048 | 146.69 | 127.58 (rope score 변경 뒤 129.5~130.1) | 88.50 | 1.44배 |
| 16384 | 109.89 | 70.67 (변경 뒤 71.3) | 25.41 | 2.78배 |

fp16 latent 대비 teacher-forced trace(`compare_logit_traces.py --decided 2`)에서 int8의 decided disagreement는 1024 뒤 209개 중 0개, 4096 뒤 105개 중 0개였다. top-1 뒤집힘은 512개 중 12개, 256개 중 3개, perplexity 변화는 +0.29%와 -0.63%다. rope score 변경 뒤 trace는 1024 뒤에서 바이트 단위로 같고 4096 뒤에서도 기록된 수치와 일치한다.

16384 토큰 prefill의 일시 최대 메모리는 int8 latent 11.23 GB, fp16 11.45 GB, decompressed int8 16.95 GB다. 16384 토큰 캐시는 0.30 GB로, decompressed의 2.29 GB보다 작다.

게이트: 8a0137ab에서 `make verify-clippy` 통과. Metal에서 직렬로 돌린 `mlxcel-core`의 `mla::`, `cache::` 테스트와 `mlxcel`의 크기 계산, advisor, CLI, prompt-cache, `models::deepseek_v2` 테스트가 통과했다. 실제 체크포인트로 돌린 `deepseek_v2_greedy_parity`로 fp16 기본값이 바뀌지 않았음을 확인했다.

## 5. 배운 점

**비용은 broadcast되는 head에서 나온다.** V2의 latent는 head가 하나이고 `quantized_matmul`은 이를 쿼리 head 16개에 broadcast하므로, head마다 packed window 전체를 읽는다. fp16 latent의 fused 커널은 모든 head에 대해 window를 한 번 읽는데, int8 decode가 그보다 느린 이유가 여기 있다. 접어 넣기 측정(16384에서 +13%)으로 이 비용이 실제로 확인됐고, 게이트가 측정한 합산 순서를 바꾸지 않고 이를 없앨 자리는 #3156의 fused 커널이다.

**근접 동률 기준은 측정 전에 정해 둬야 한다.** latent score 접어 넣기는 perplexity가 좋아졌고(-0.20%) decided disagreement도 0건이어서, 지표 세 개 중 두 개만 보면 더 나은 결과였다. 결정을 내린 것은 실행 전에 이슈에 적어 둔 기준이었다.

**crate 단위 검사로는 API 변경이 놓친 곳을 못 찾는다.** `LatentWindow` 반환 타입 때문에 `examples/mla_absorbed_decode_bench.rs`가 깨졌는데, 이 파일은 workspace `--all-targets` clippy에서만 컴파일된다. 좁은 crate 검사는 모두 통과했다.

## 6. 후속 작업

- #3156: 양자화 latent용 fused Metal 커널. 모든 head가 window를 한 번만 읽는 레이아웃은 여기서 구현한다.
- #3157: latent에 대한 Turbo WHT 평가.
- #3160-#3163: latent-pair 계열의 양자화 latent. 이 PR에서는 fp16을 유지한다.
- 0.11 근접 동률 기준이 latent score 접어 넣기(16384에서 +13%)를 허용해야 하는지, 그리고 스텝마다의 f32 변환을 없애려고 f32 scale 사이드카(토큰당 레이어당 672 대신 704바이트)를 쓸 가치가 있는지 결정이 필요하다.
- CUDA GB10과 ROCm gfx1151은 이 호스트에서 접근할 수 없어 측정하지 못했다.
