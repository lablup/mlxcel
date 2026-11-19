# 기술 보고서: PR #3172 - DeepSeekV3Attention 계열에 양자화 MLA latent 저장

**날짜**: 2026-11-19
**상태**: 완료
**위험도**: 중간 (모델 타입 여덟 개에 opt-in int8 latent가 추가되고, 근접 동률 기준 미달을 검토 후 수용함)

## 요약

`DeepSeekV3Attention`을 쓰는 모델 타입 여덟 개(`deepseek_v3`, `kimi_k2`, `joyai_llm_flash`, `kimi_vl`, `kimi_k25`, `youtu`, `youtu_llm`, `youtu_vl`)가 이제 MLA latent를 `MlaLatentCache`로 저장한다. 그래서 `--kv-cache-mode int8`과 `--kv-bits 8`을 주면 fp16으로 치환되지 않고 #3155의 affine int8 latent(토큰당 레이어당 1152 대신 672바이트)를 그대로 쓴다. Moonlight-16B-A3B와 Kimi-VL-A3B의 teacher-forced trace는 1024와 4096 뒤에서 decided disagreement가 0건이지만, Moonlight는 1024 뒤에서 top-1 뒤집힘 두 개가 0.11 근접 동률 기준 밖에 있다(0.117, 0.328). 이 미달은 검토 후 수용했다. decode는 fp16 latent보다 512 토큰에서 10%, 2048에서 13% 느리다.

## 1. 문제

#3155는 `deepseek_v2`에만 int8 latent를 넣었다. latent-pair 계열은 `(kv_latent, k_pe)`를 `KVCache::update_and_fetch`로 바로 저장했기 때문에, `resolve_kv_cache_mode_for_model_with`의 rule 1이 이들에 대한 양자화 요청을 모두 fp16으로 바꿨다. `DeepSeekV3Attention`은 모델 타입 여덟 개가 함께 쓰고, batched decode 헬퍼 `mla_latent_decode_rows`는 다른 계열 네 개도 쓴다.

## 2. 변경 내용

- `mla::latent_pair`(`mla/cache.rs` 옆 새 파일): `QUANTIZED_LATENT_FAMILIES`와 `latent_pair_supports_mode(model_type, mode)`를 두고 rule 1이 이를 부른다. Turbo 모드는 계속 fp16으로 치환되고, 목록 밖 계열은 fp16을 유지한다.
- `src/models/deepseek_v3.rs`와 `src/models/mla_batched.rs`: 단일 시퀀스 forward와 batched 헬퍼가 `store_latent_step`으로 저장한다. 양자화 window의 쿼리 하나짜리 스텝은 `mlxcel_core::mla::quantized_absorbed_attention`을 부르는데, 이 함수는 #3156 이후 fused 양자화 커널(Metal, 2048 key 이하)을 먼저 시도한다. 여러 토큰 스텝은 한 번 dequantize하고, fp16 캐시는 기존 경로를 그대로 탄다.
- batched 헬퍼는 명시적인 `quantized_latent` opt-in을 받는다. `true`를 넘기는 것은 `DeepSeekV3Attention`뿐이어서, `deepseek_v32`, `kimi_linear`, `glm4_moe_lite`, `kimi_k3`는 각자의 이슈(#3161-#3163)가 끝날 때까지 토큰별 경로를 쓴다. 양자화 row에 selector가 오면 계열 이름을 담아 panic한다.
- 크기 계산(`kv_arch`, `inspect`, 사전 점검, prompt-cache 예산), KV advisor(중간·긴 컨텍스트에서 int8 권장), serve의 `--kv-bits` 검사, generate 배너, registry `kv_modes`(양자화 latent registry 항목 다섯 개에 fp16과 int8)가 함께 바뀌었다.
- 문서: `docs/turbo-kv-cache.md`, `docs/supported-models.md`, `docs/environment-variables.md`, CHANGELOG, 그리고 #3156 전후 CSV가 딸린 `docs/benchmark_results/mla-quantized-latent-v3-m3ultra-2026-11-19.md`.

## 3. 기술적 결정

**계열 전용 경로를 만들지 않고 #3155의 경계를 재사용했다.** 저장은 레이어 차원으로 만든 `MlaGeometry`와 함께 `MlaLatentCache::wrap`을 거친다. fp16 캐시는 `Dense`를 돌려주므로 기존 분기가 그대로 돈다. 두 체크포인트 모두 fp16 trace가 base 커밋과 바이트 단위로 같다.

**batched opt-in을 명시적으로 만들었다.** batched 헬퍼는 아직 dense window만 읽는 계열과 공유된다. rule 1을 거치지 않은 mode table로 int8 캐시가 그쪽에 들어갈 수 있어서, 이제는 호출하는 쪽이 `LatentWindow::Quantized`를 읽는다고 밝혀야 한다. 밝히지 않은 계열은 계열마다 한 번 경고를 내고 토큰별 경로를 쓴다.

**decided disagreement를 근거로 근접 동률 미달을 수용했다.** #3155의 기준은 decided disagreement 0건, 그리고 모든 top-1 뒤집힘이 기준 모델 top-two 차이 0.11 logit 이내인 것이다. 두 체크포인트 모두 앞의 조건은 모든 구간에서 지켰다. #3156 병합 전에는 1024 뒤에서 체크포인트마다 뒤집힘 하나가 기준 밖이었고(Moonlight 0.117, Kimi-VL 0.180), 같은 체크포인트에서 fp16 fused 대 composed 잡음 상한은 0.109였다. 사용자가 이 수치를 보고 머지를 결정했다. #3156 병합 뒤에는 fused 커널 때문에 어느 근접 동률 위치가 뒤집히는지가 바뀌었다. Kimi-VL은 기준 밖 뒤집힘이 없어졌고 Moonlight는 두 개(0.117, 0.328)가 되었으며, decided disagreement는 여전히 0건이다. 결과 문서, CHANGELOG, `docs/turbo-kv-cache.md`에 이 미달을 적었다.

## 4. 측정

M3 Ultra, Metal. 따로 적지 않은 값은 #3156을 병합한 트리에서 쟀고, 기준은 fp16 latent다.

Trace (`--decided 2`):

| 모델 | 구간 | decided | top-1 뒤집힘 | 0.11 초과 | 최대 차이 | ppl 변화 | int8 경로 |
|---|---|---|---|---|---|---|---|
| Moonlight | 1024 뒤 | 0 / 192 | 7 / 512 | 2 | 0.328 | +0.76% | fused |
| Kimi-VL | 1024 뒤 | 0 / 188 | 12 / 512 | 0 | 0.094 | +0.99% | fused |
| Moonlight | 4096 뒤 (#3156 전, 양쪽 모두 composed) | 0 / 58 | 5 / 256 | 0 | 0.039 | -0.04% | composed |
| Kimi-VL | 4096 뒤 (#3156 전, 양쪽 모두 composed) | 0 / 51 | 3 / 256 | 0 | 0.094 | -0.44% | composed |

Decode, tok/s 중앙값, fp16 / int8:

| 모델 | 512 (fused) | 2048 (composed) | 긴 프롬프트 (composed, #3156 전) |
|---|---|---|---|
| Moonlight | 159.07 / 142.45 (-10.4%) | 141.80 / 122.97 (-13.3%) | 8064에서 102.90 / 94.82 (-7.9%) |
| Kimi-VL | 158.53 / 142.95 (-9.8%) | 141.85 / 122.61 (-13.6%) | 16384에서 107.59 / 69.05 (-35.8%) |

`inspect`는 토큰당 레이어당 672바이트(16384 토큰에서 283.5 MiB)를 보고하며, 서버 prompt cache가 실제로 잡은 바이트와 같다. 병합한 트리에서 workspace clippy와 mlxcel-core, mlxcel 대상 테스트가 통과했다.

## 5. 배운 점

**커널을 바꾸면 decided disagreement가 늘지 않아도 근접 동률 뒤집힘은 옮겨 간다.** #3156 전 composed int8 trace와 fused trace를 비교하면 decided disagreement는 0건이고, 뒤집힘은 512개 중 7개, 최대 차이 0.469다. 기준 밖 뒤집힘 집합은 (0.117, 0.180)에서 (0.117, 0.328)로 바뀌었지만 decided 수는 그대로였다. 한 합산 순서에서 잰 근접 동률 기준은 다른 합산 순서로 옮겨지지 않으며, 그래서 기준에서 안정적인 부분은 decided 수다.

**다른 PR을 병합하면 이 PR의 수치가 가리키는 코드가 바뀔 수 있다.** #3172는 composed 경로에서 측정했는데, #3156을 병합하자 PR 자체 파일에는 충돌이 없는데도 공유 함수를 통해 짧은 window decode가 fused 커널로 넘어갔다. 병합한 트리에서 다시 측정했기 때문에 PR 본문과 결과 문서가 실제로 머지되는 코드를 설명하게 됐다.

## 6. 후속 작업

- #3161(GLM-4.7-Flash, LongCat), #3162(Kimi Linear, Kimi K3), #3163(DSA 계열)이 같은 변경을 옮긴다. 기준은 decided disagreement 0건이고, 근접 동률 미달은 기록한다.
- `kimi_k2`, `joyai_llm_flash`, `kimi_k25`, Youtu 타입은 단위 테스트로만 확인했고 체크포인트로는 돌리지 않았다.
- 여러 토큰 스텝마다 window 전체를 dequantize하므로, chunked prefill과 verify 스텝 비용이 window 길이에 비례해 커진다.
- rank가 64의 배수가 아니면 row당 scale 하나로 돌아가고 경고를 낸다. 배포된 체크포인트 중에는 그런 rank가 없다.
- CUDA GB10과 ROCm gfx1151은 측정하지 않았다.
