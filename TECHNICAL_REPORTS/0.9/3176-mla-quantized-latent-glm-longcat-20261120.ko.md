# 기술 보고서: PR #3176 - LongCat-Flash의 양자화 MLA latent, GLM-4.7-Flash는 이식 후 비활성 유지

**날짜**: 2026-11-20
**상태**: 완료
**위험도**: 중간 (LongCat에 opt-in int8 latent가 추가되고 로딩 자체가 가능해짐, GLM은 목록에 없는 opt-in 뒤에 이식)

## 요약

`longcat_flash`와 `longcat_flash_ngram`은 int8을 요청하면 MLA latent를 #3155의 affine int8 latent로 저장하고, `QUANTIZED_LATENT_FAMILIES`에 들어간다. LongCat-Flash-Lite의 teacher-forced trace는 1024와 4096 뒤에서 decided 불일치가 0이고, `inspect`는 16384 토큰에서 int8 294.0 MiB, fp16 504.0 MiB를 보고한다. Decode는 512 토큰에서 +8.0%, 2048에서 -15.5%, 16384에서 -49.2%다. 16384에서는 fused 커널 대신 composed 경로가 돈다. `glm4_moe_lite`(GLM-4.7-Flash)는 MTP verify와 rollback까지 이식했지만, 1024 뒤에서 decided 불일치가 1개 있어 필수 게이트에 따라 fp16으로 둔다. main에서 LongCat 체크포인트를 어떤 KV 모드로도 돌리지 못하게 하던 로딩 버그 두 개를 고쳤다.

## 1. 문제

#3160과 #3162 이후에도 fp16 치환에 남은 latent pair 계열은 GLM-4.7-Flash와 LongCat-Flash였다. 둘 다 자체 attention 코드에서 `KVCache::update_and_fetch`를 호출한다. LongCat은 decoder 레이어마다 MLA sub-layer가 두 개 있고, GLM에는 target 캐시를 verify하고 rollback하는 MTP drafter가 있다.

## 2. 변경 내용

- `src/models/longcat_flash_ngram.rs`: 두 MLA sub-layer 모두 latent view로 쓰고, 단일 query step은 `quantized_absorbed_attention`으로 읽으며, 다중 토큰 step은 한 번 dequantize한다. 양자화된 embedding table을 양자화된 채로 로딩하고, MoE가 `SwitchGLU` 전에 `[B, L, hidden]`을 펼친다. 이전에는 64개 이상 위치에서 예외가 났다. 로딩 시 embedding table과 n-gram 설정을 검증한다.
- `src/models/glm4_moe_lite.rs`, `glm4_moe_lite_mtp_hooks.rs`, `glm4_moe_lite_mtp_drafter.rs`: 같은 이식을 모델별 `quantized_latent` opt-in 뒤에 두고, 양자화 window 위의 MTP verify와 rollback을 지원한다.
- `src/models/mla_batched.rs`: `dense_latent_in`가 dequantize한 window를 query dtype으로 변환해 bf16 체크포인트가 float32로 승격되지 않게 한다.
- `mla/latent_pair.rs`, `cache/turbo/allowlist.rs`, `registry.rs`: LongCat을 목록에 추가하고 `kv_modes`를 fp16과 int8로 둔다. GLM은 목록에 없다.
- `kv_arch`: LongCat을 레이어당 latent 캐시 두 개로 계산한다.
- `examples/logit_trace/kv_mode.rs`: trace 도구가 요청한 모드 표를 모델 소유 캐시에 넘긴다.
- 문서: CHANGELOG, `docs/turbo-kv-cache.md`, `docs/supported-models.md`, `docs/benchmark_results/mla-quantized-latent-glm-longcat-m3ultra-2026-11-19.md`.

## 3. 기술적 결정

**decided 불일치 1개로 GLM을 끈 채로 둔다.** 에픽 게이트는 decided 불일치 0을 필수로 한다. 1024 뒤 GLM의 decided 불일치 1개(gap 2.75)는 int8 fused 대 composed 잡음에도 똑같이 1개 있으므로, latent 결함보다는 커널 잡음일 가능성이 높다. 그래도 규칙은 그대로 적용했다. 이식 코드는 트리에 남아 있어, 결정이 나면 목록에 추가하는 것만으로 GLM을 켤 수 있다.

**LongCat 로딩 수정을 이 PR에 넣는다.** main에서는 체크포인트가 어떤 KV 모드로도 돌지 않아서, 두 로더 수정 없이는 게이트에 쓸 fp16 기준이 없었다. 두 수정 모두 tiny 모델 테스트가 있다.

## 4. 측정

M3 Ultra, Metal, 기준은 fp16 latent, `--decided 2`.

| 모델 | 뒤 길이 | decided | top-1 flip | 0.11 초과 flip | ppl fp16 / int8 |
|---|---|---|---|---|---|
| LongCat-Flash-Lite | 1024 | 0 / 239 | 15 / 512 | 9 (최대 0.375) | 14.1036 / 14.0010 |
| LongCat-Flash-Lite | 4096 | 0 / 121 | 9 / 256 | 5 (최대 0.25) | 9.9146 / 9.7928 |
| GLM-4.7-Flash | 1024 | 1 / 193 | 40 / 512 | 18 (최대 2.75) | 21.4881 / 20.8288 |
| GLM-4.7-Flash | 4096 | 0 / 95 | 25 / 256 | 15 (최대 1.0) | 53.6498 / 53.4191 |

LongCat decode, tok/s 중앙값 fp16 / int8: 512에서 125.52 / 135.53(+8.0%, fused), 2048에서 127.40 / 107.59(-15.5%), 16384에서 99.20 / 50.39(-49.2%, composed). decompressed decode step은 없다.

## 5. 배운 점

**bf16 승격은 세 유닛에서 따로 드러났다.** #3162는 decode 저하로 발견했고, 이 PR은 GLM의 perplexity가 8~13% 움직이는 것으로 발견했다. 수정 전 GLM 수치는 decided 불일치가 2개였고 폐기했다. 고정 dtype을 반환하는 dequantize helper는 f32나 fp16으로 돌리는 테스트에서 놓치기 쉽다.

**fp16 잡음 바닥이 없으면 미달의 해석이 달라진다.** GLM의 fp16 latent는 항상 composed로 decode하므로 fp16 fused 대 composed 쌍이 없다. 쓸 수 있는 바닥은 int8 쌍뿐이고, 거기에도 decided 불일치가 똑같이 1개 있다.

## 6. 후속 작업

- `deepseek_v2.rs`와 `deepseek_v3.rs`는 cast 없이 `into_dense`로 dequantize하므로, bf16 체크포인트(Moonlight, Kimi-VL, DeepSeek-V2-Lite)의 int8 prefill이 float32로 돈다. 수정 후 #3155와 #3160의 게이트 수치가 달라질 수 있다.
- composed int8 경로는 16384 토큰에서 최대 49% 느리다. fused 커널을 2048 key 너머로 확장하면 해결된다.
- int8 커널 잡음과 같은 수준인 GLM의 decided 불일치 1개를 근거로 GLM을 켤지 결정해야 한다.
- 560B `longcat_flash`, CUDA, ROCm은 측정하지 않았다. GLM decode 수치는 bf16 수정 전 값이다.
