# 기술 보고서: PR #3189 - MLA latent 정밀도를 고르는 실험적 `--mla-latent-quant` 옵션

**날짜**: 2026-11-21
**상태**: 완료
**위험도**: 중간 (opt-in 전용. 기본값 `auto`는 이전과 똑같이 결정하지만, 명시적 값은 품질 게이트가 출시하지 않은 latent를 저장한다. 새 4-bit latent와 분할 DSA key slot을 포함한다)

## 요약

`--mla-latent-quant <auto|fp16|int8|int4>`(환경 변수 `MLXCEL_MLA_LATENT_QUANT`, CLI 플래그가 우선)는 로드한 모델의 모든 MLA latent 캐시를 어떤 정밀도로 저장할지 고른다(#3179, 에픽 #3158의 후속). `auto`는 기본 게이트를 그대로 재현한다. `fp16`은 어떤 KV 모드에서도 latent를 fp16으로 두고, `int8`과 `int4`는 게이트가 막은 GLM-4.7-Flash와 DSA 계열을 포함한 모든 MLA 계열에 affine group-64 latent를 저장한다. M3 Ultra에서 4-bit latent는 Moonlight, LongCat-Flash-Lite, Kimi-Linear-48B, DeepSeek-V2-Lite에서 decided 불일치가 0이지만, 네 모델 모두 0.11 logit near-tie 기준을 넘는다(flip 최대 0.84, 1.13, 0.63, 0.86 logit). GLM-4.7-Flash에서는 1024와 4096 토큰 뒤 모두 decided 불일치가 3개이고 perplexity가 -19.5%, -16.1% 움직인다. 따라서 `int4`는 모든 계열에서 실험적으로 남는다. 옵션을 거친 int8 arm은 에픽의 수치를 그대로 재현한다.

## 1. 문제

에픽 #3158은 품질 게이트를 통과한 계열에만 affine int8 MLA latent를 출시했다. GLM-4.7-Flash(decided 불일치 1개)와 DSA 계열(indexer top-k overlap 0.99 미만)은 fp16으로 남았고, #3155에서 측정한 4-bit latent는 아예 제공되지 않았다. 측정된 위험을 감수하려는 운영자도 별도 빌드 없이는 그 정밀도를 고를 수 없었다.

## 2. 변경 내용

- `mla/latent_option.rs`(신규): 옵션 타입, 파싱, 프로세스 상태(`init_mla_latent_quant`, `mla_latent_quant`, 테스트와 측정 도구용 스레드 로컬 override), 그리고 계열, 요청된 `KVCacheMode`, 옵션을 latent 비트 폭으로 바꾸는 유일한 지점인 `effective_latent_bits`. 시작 시 알림과 측정 결과 표(`EXPERIMENTAL_LATENT_NOTES`, 항목마다 폭 하나)도 여기에 있다.
- 모드 결정(`cache/turbo/allowlist.rs`): MLA latent를 가진 모델에 명시적 값이 주어지면 그 값이 저장 방식을 혼자 결정한다. 폭이 나오면 `Int8` 저장, 아니면 fp16이다. 새 `KVCacheMode` 변형은 없고, `Int8` 저장 안의 폭은 `latent_cache_bits`(8, `int4`에서는 4)다. 옵션이 모델을 결정하는 조건은 `option_decides_latent` 하나로 표현하며, resolver, 알림, DSA key slot 거부, `--kv-bits` 표가 모두 이 함수를 쓰고 대소문자 규칙도 하나다.
- 4-bit latent(`mla/cache.rs`, `mla/quant_latent.rs`, `mla/latent_keys.rs`): 같은 affine codec을 4 bit로 쓰며, 항상 composed `quantized_matmul` 경로로 decode한다(fused Metal 커널은 8-bit 코드만 읽는다). DSA 계열에서는 4-bit latent 옆의 indexer key가 8 bit로 남는다. key slot은 같은 버퍼 안의 두 group 묶음이고, 경계는 버퍼 shape에서 계산한다.
- 크기 계산과 보고: `kv_arch`는 저장 폭으로 계산하고(rank 512에서 레이어, 토큰당 416 B, DSA indexer key 포함 552 B), 메모리 추정의 `KV dtype` 줄은 더 좁은 latent 폭과 그것을 고른 플래그를 적는다.
- 진입점: `generate`, `run`, `serve`, `inspect`, `mlxcel-server`, `mlxcel-bench-decode`는 플래그를 받고, `tune`, `predict`, `mlxcel-bench-engine`, `mlxcel-engine-parity`, `speculative_bench`는 시작 시 환경 변수 값을 검증한다. 잘못된 값이나 UTF-8이 아닌 값은 시작 오류다.
- 서버: prompt cache adoption 검사가 분리된 각 레이어의 저장 latent 폭을 이 프로세스가 쓰는 폭과 비교하므로, 다른 정밀도로 저장된 항목은 거절된다(`kv_mode_mismatch`).
- 문서: `docs/turbo-kv-cache.md`, `docs/environment-variables.md`, `docs/benchmark_results/mla-latent-quant-option-m3ultra-2026-11-20.md`.

## 3. 기술적 결정

**latent 폭을 버퍼 shape에서 읽는다.** 이미 row를 가진 캐시는 packed와 scale 버퍼에서 자기 폭을 보고하고, 새 캐시만 옵션을 읽는다. 그래서 폭이 trim, detach, adoption, prompt cache 항목을 거쳐도 row를 따라가며, 어긋날 수 있는 별도 필드가 없다.

**옵션이 결정하는 모델에서는 `--kv-bits`가 물러난다.** MLA 계열의 KV 캐시는 모두 latent 캐시다(Kimi Linear와 Kimi K3의 KDA 레이어는 KV가 아닌 recurrent state를 가진다). 그래서 레이어별 표가 마지막 레이어를 포함해 더 결정할 것이 없다.

**요청 모드가 fp16이면 불일치 경고를 내지 않는다.** fp16은 기본 요청이므로 거기서 경고하면 명시적 사용마다 경고가 나온다. 불일치 줄은 양자화 모드가 다른 latent 정밀도를 뜻할 때만 출력한다. 실험적 알림은 목록 밖 계열과 모든 `int4`에서 그대로 출력한다.

**rank가 64의 배수가 아니면 강제 latent는 시작 시 fp16을 유지한다.** DSA slot만 지키던 whole-group 검사를 명시적 값에서는 모든 latent pair 계열과 absorbed DeepSeek V2로 넓혔다. 그런 설정은 첫 캐시 쓰기에서 토큰 단위(또는 decompressed) fallback에 닿는 대신, 플래그를 적은 경고와 함께 fp16으로 돌아간다.

## 4. 측정

M3 Ultra, Metal, fp16 latent 대비 teacher-forced, `--decided 2`, 1024 뒤 8 chunk, 4096 뒤 4 chunk.

| 모델 | int4 decided (1024 / 4096) | 최대 flip gap | ppl 변화 (1024 / 4096) |
|---|---|---|---|
| Moonlight-16B-A3B | 0 / 0 | 0.844 | +2.32% / +1.21% |
| LongCat-Flash-Lite | 0 / 0 | 1.125 | -2.15% / -3.90% |
| Kimi-Linear-48B-A3B | 0 / 0 | 0.625 | -0.46% / +0.23% |
| DeepSeek-V2-Lite | 0 / 0 | 0.859 | -0.004% / +2.00% |
| GLM-4.7-Flash | 3 / 3 | 4.000 | -19.52% / -16.09% |

옵션을 거친 int8 arm은 GLM-4.7-Flash의 실패한 1 / 193을 포함해 에픽 문서의 모든 개수, gap, perplexity를 그대로 재현한다.

LongCat-Flash-Lite decode, tok/s 중앙값 fp16 / int8 / int4: 512 토큰에서 124.7 / 135.5 / 124.3, 2048에서 125.1 / 106.9 / 110.0, 16384에서 99.3 / 50.2 / 54.5. Moonlight: 512에서 156.0 / 140.5 / 132.0, 2048에서 139.8 / 121.3 / 123.6. 두 모델 모두 int4는 둘 다 composed로 도는 구간에서 int8보다 2%~9% 빠르고, int8이 fused 커널을 쓰는 512 토큰 뒤에서는 6%~8% 느리다.

`mlxcel inspect --max-tokens 16384`: Moonlight fp16 486.0 MiB, int8 283.5 MiB, int4 175.5 MiB. GLM-4.7-Flash 846.0, 493.5, 305.5 MiB. DeepSeek-V3.2 1.31 GiB, 770.1 MiB, 526.1 MiB(레이어, 토큰당 552 B: latent 416, int8 indexer key 136).

## 5. 측정하지 않은 것

- 옵션 아래의 DeepSeek-V3.2: 강제 int8 slot과 int4 분할 slot 모두 teacher-forced trace와 indexer top-k overlap을 돌리지 않았다. 분할 slot은 tiny 모델 테스트, 200 토큰 generate의 일관된 출력, 4096 토큰 sparse decode 실행에 기대고 있으며, 실제 체크포인트에서의 품질은 측정하지 않았다.
- GLM-5(`glm_moe_dsa`), LongCat-Flash 560B(`longcat_flash`), Kimi K3: 이 호스트에 없거나 돌리지 않았다. 측정한 코드 경로와 tiny 모델 테스트를 공유한다.
- CUDA와 ROCm: trace와 처리량 수치가 없다. int4 경로는 두 백엔드가 이미 int8에서 쓰는 composed `quantized_matmul` decode를 4 bit로 쓰는 것이다.
- GLM-4.7-Flash, Kimi Linear, DeepSeek-V2-Lite의 int4 decode 처리량, 실제 서빙 체크포인트에서의 prompt cache adoption(단위 테스트로만 확인).

## 6. 배운 점

**측정 결과 메모는 측정한 폭에만 연결해야 한다.** GLM int8 메모가 두 폭 모두에 연결되어 있어서, int4 실행이 int8 결과를 int4 결과처럼 출력했다. 이제 표의 각 항목은 폭을 정확히 하나만 가지며, 테스트가 이를 강제한다.

**세 번 쓴 규칙은 어긋난다.** "명시적 값이 이 모델을 결정한다"는 규칙이 세 군데 있었고, 그중 하나는 옆의 helper가 대소문자를 무시하는데도 `deepseek_v2`를 대소문자 구분해 비교했다. 이제 대소문자 규칙 하나를 가진 함수 하나다.

## 7. 후속 작업

- 누군가 DeepSeek-V3.2 int4 분할 slot에 기대기 전에 실제 체크포인트에서 측정(trace와 top-k overlap)해야 한다.
- fused Metal 커널을 4-bit 코드로 확장하면 짧은 window에서 int4가 잃는 속도를 되찾는다.
- `packed_latent_bits`는 4-bit와 8-bit 크기 사이의 packed 폭을 모두 4로 본다. 캐시가 저장할 수 있는 두 폭에서는 맞지만, 다른 폭을 추가하면 다시 봐야 한다.
