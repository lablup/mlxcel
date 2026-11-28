# 기술 보고서: PR #3174 - Kimi Linear와 Kimi K3에 양자화 MLA latent 저장

**날짜**: 2026-11-20
**상태**: 완료
**위험도**: 중간 (하이브리드 두 계열에 opt-in int8 latent 추가, 측정 중 bf16 dtype 승격 버그를 찾아 수정)

## 요약

`kimi_linear`, `kimi_k3`, Kimi K3 VLM 래퍼가 레이어별 `KvCacheLayerModes` 표를 받아, MLA 레이어의 latent를 #3155의 affine int8 latent(토큰당 레이어당 672바이트)로 저장한다. KDA 레이어는 표의 항목을 무시한다. Kimi-Linear-48B-A3B의 teacher-forced trace는 1024와 4096 뒤에서 decided 불일치가 0이고, prefix마다 0.11 near-tie 기준을 넘는 flip이 다섯 개(최대 0.500) 있다. 에픽 게이트에 따라 기록만 하고 막지 않는다. Decode는 512 토큰에서 +1.5%(fused 커널), 2048에서 -4.7%, 16384에서 -23.3%(composed 경로)다. 크기 계산도 KDA 레이어에 latent를 더하지 않도록 고쳤다. 이전에는 모든 모드에서 Kimi-Linear-48B의 KV를 27/7배로 과대 계산했다.

## 1. 문제

#3160 이후에도 `kimi_linear`와 `kimi_k3`는 `QUANTIZED_LATENT_FAMILIES` 밖에 있어서, 양자화 요청이 fp16으로 바뀌고 batched helper는 이들을 토큰별 경로에 두었다. 두 계열은 하이브리드다. Kimi-Linear-48B는 27개 레이어 중 7개가 MLA이고 나머지는 KDA(recurrent state를 가진 linear attention)이므로, 레이어 단위 모드 표에는 latent와 무관한 항목이 섞인다.

## 2. 변경 내용

- `src/models/kimi_linear.rs`, `src/models/kimi_k3.rs`, `src/vision/kimi_k3_vl.rs`: `set_kv_cache_layer_modes`를 구현하고 VLM 래퍼에서 전달한다. MLA 레이어는 `store_latent_step`으로 쓴다. 양자화 window 위의 단일 query step은 `quantized_absorbed_attention`을 호출한다(마스크가 없으면 fused, 있으면 composed, 마스크는 score dtype으로 변환). 다중 토큰 step은 window를 한 번 dequantize한다. Turbo 항목은 새 `latent_pair_cache_mode` helper로 fp16 캐시를 만든다.
- `src/models/mla_batched.rs`: 두 계열의 batched decode가 양자화 row를 opt-in하고, `dense_latent_in`이 dequantize된 window를 query dtype으로 되돌린다.
- `mla/latent_pair.rs`와 `cache/turbo/allowlist.rs`: `kimi_linear`와 `kimi_k3`를 `QUANTIZED_LATENT_FAMILIES`에 추가했다. registry는 KimiLinear, KimiK3, KimiK3VLM의 `kv_modes`로 fp16과 int8을 나열한다.
- `src/execution/kv_arch.rs`: 1부터 세는 `kda_layers` 목록을 써서 MLA 레이어에만 latent를 계산한다. Kimi-Linear-48B에서 `mlxcel inspect --max-tokens 16384`는 이제 fp16 126.0 MiB, int8 73.5 MiB를 보고한다.
- 문서: CHANGELOG, `docs/turbo-kv-cache.md`, `docs/supported-models.md`, `docs/environment-variables.md`, trace와 decode CSV를 포함한 `docs/benchmark_results/mla-quantized-latent-kimi-m3ultra-2026-11-19.md`.

## 3. 기술적 결정

**KDA 레이어는 모드 항목을 거부하지 않고 무시한다.** 모드 표는 어떤 레이어가 MLA인지 모르는 코드가 레이어별로 만든다. KDA 인덱스의 양자화 항목을 거부하면 하이브리드의 모든 int8 요청이 실패한다. 무시하면 표 모양이 균일하게 유지되고, KDA 인덱스에만 모드를 지정하는 테스트가 이 동작을 고정한다.

**Dequantize한 window를 query dtype으로 되돌린다.** `LatentWindow::into_dense`는 fp16을 반환한다. bf16 query와 만나면 MLX가 다중 토큰 context pass를 float32로 승격했고, Kimi에서는 그 pass가 만든 KDA recurrent state가 float32로 남아 이후 모든 decode step이 float32로 돌았다. 첫 측정에서 int8이 fp16보다 61~63% 느렸던 이유다. `dense_latent_in`의 cast로 해결했고, 두 모델의 회귀 테스트는 cast를 되돌리면 실패한다.

**에픽 게이트에 따라 계열을 켠 채로 둔다.** decided 불일치 0이 게이트의 필수 조건이고 두 prefix 모두 만족한다. near-tie 초과는 int8 fused 대 composed 잡음(16 / 512 flip, 0.11 초과 7개, 최대 0.438) 안에 있다. Kimi의 fp16 단일 시퀀스 decode는 fused 커널을 타지 않으므로 fp16 잡음 바닥은 0이다.

## 4. 측정

M3 Ultra, Metal, Kimi-Linear-48B-A3B-Instruct 4-bit, 기준은 fp16 latent.

Trace (`--decided 2`):

| 뒤 길이 | decided | top-1 flip | 0.11 초과 flip | 최대 gap | ppl fp16 / int8 |
|---|---|---|---|---|---|
| 1024 | 0 / 201 | 12 / 512 | 5 | 0.500 | 12.4213 / 12.4579 (+0.29%) |
| 4096 | 0 / 52 | 13 / 256 | 5 | 0.312 | 24.5971 / 24.4992 (-0.40%) |

Decode, tok/s 중앙값, 한 번의 lock 안에서 ABAB, 편차 최대 1.4%:

| prompt | fp16 | int8 | 차이 | int8 경로 |
|---|---|---|---|---|
| 512 | 127.49 | 129.34 | +1.5% | fused |
| 2048 | 127.43 | 121.43 | -4.7% | composed |
| 16384 | 115.94 | 88.97 | -23.3% | composed |

Prefill 차이는 0.6% 이내다. 27개 레이어 중 7개만 latent를 가지므로 peak 메모리 차이는 0.2 GB 미만이다. 경로 카운터상 decompressed decode step은 없다.

## 5. 배운 점

**상태를 가진 레이어에서는 dtype 불일치가 step마다 누적된다.** attention만 있는 모델이라면 context pass의 float32 승격은 그 pass 한 번의 비용이다. 하이브리드에서는 승격된 출력이 계속 유지되는 recurrent state로 들어가서 decode 전체가 float32로 돌았다. latent 작업으로 설명할 수 없을 만큼 큰 decode 저하가 dtype을 의심하게 만든 단서였다.

**하이브리드의 크기 계산은 레이어 맵을 따라야 한다.** 이 PR 전에도 Kimi-Linear-48B의 fp16 추정은 27개 레이어 전부에 latent를 더해서 틀려 있었다. int8 작업에서 새 수치를 실제 config의 `inspect` 출력과 비교하면서 드러났다.

## 6. 후속 작업

- DeepSeek V2와 `DeepSeekV3Attention` 계열도 같은 `into_dense`를 쓰므로 bf16 체크포인트에서 다중 토큰 step이 float32로 승격될 수 있다. recurrent state가 없어 비용은 그 step에 국한될 것으로 보이지만 측정하지 않았다.
- int8 composed 경로는 16384 토큰에서 23% 느리다. fused 커널의 2048 key 제한이 원인이다.
- Kimi K3는 측정하지 않았다. 공개된 export가 expert를 잘라낸 2-bit 모델뿐이다.
- 하이브리드가 KDA 레이어로 끝나면 `--kv-bits 8` 크기 계산이 한 레이어만큼 과대 추정된다.
- `kimi_linear.rs`의 `kda_layers[0] - 1`은 첫 항목이 0이면 underflow한다(기존 코드, 결과는 쓰이지 않음).
- CUDA와 ROCm은 측정하지 않았다.
