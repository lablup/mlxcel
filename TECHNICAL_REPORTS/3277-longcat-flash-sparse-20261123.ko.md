# PR #3277: LongCat-Flash-Lite-Sparse (`longcat_flash_sparse`) 추가

**날짜**: 2026-11-23
**상태**: draft PR로 열림(#2362를 닫음). 단위 테스트는 직렬 실행에서 통과. 4bit 체크포인트를 2048 토큰을 넘는 문맥에서 mlx-vlm과 비교 검증했고, fp16, int8, int4 latent 모두 측정함.
**위험도**: 중간(공유 LongCat MLA 코드를 리팩터링했으나 ngram 계열의 trace는 전후 바이트 단위로 동일)

## 요약

LongCat-Flash-Lite-Sparse는 LongCat ScMoE 디코더에 두 가지 변화가 더해진 모델이다. 기존 `longcat_flash_ngram` 로더로는 둘 다 표현할 수 없다. 하나는 LSA sparse attention으로, lightning indexer가 query마다 2048개의 key를 고른다. 다른 하나는 word embedding을 원래 크기 그대로 두는 n-gram embedding 결합 방식이다. 이 PR 전에는 체크포인트가 `Unsupported model type`으로 거부됐다. ngram 계열로 억지로 읽었다면 embedding 전체가 13으로 나뉘고 attention은 dense로 돌았을 것이다.

이 PR은 `models::longcat_flash_sparse`를 추가한다. `LongcatFlashMLA::forward`를 `project`와 `attend_window`로 나누어 공유 LongCat MLA, MLP, MoE를 그대로 재사용한다. indexer의 선택은 decode에서는 행 gather로, multi-token 단계에서는 additive mask로 attention에 전달된다. 2026-11-23 메인테이너 결정에 따라 이 계열은 양자화된 MLA latent를 저장한다. 기본값은 fp16 그대로이고, `--kv-cache-mode int8`은 8bit latent를 저장하며(fp16 대비 decided 위치 불일치 0건), `--mla-latent-quant int4`는 받아들이되 실험 옵션으로 둔다(decided 불일치 4~6건, perplexity +15%~+23%).

## 1. 문제

`mlx-community/LongCat-Flash-Lite-Sparse-{4,6,8}bit`는 `model_type: longcat_flash_sparse`를 선언한다. Meituan 원본 체크포인트는 `model_type` 없이 `architectures: ["LongcatCausalLM"]`만 선언한다. 둘 다 거부됐다. 이 아키텍처는 `longcat_flash_ngram`과 세 군데가 다르고, 어느 하나를 놓쳐도 겉보기에는 유창한 엉터리 출력이 나온다.

- **결합(fusion).** upstream은 `word + proj / (1 + n)`을 반환한다. ngram 계열은 `(word + proj) / (1 + n)`이다. `n = 12`이므로 잘못된 식은 모든 토큰 embedding을 13배 줄인다. trace에서 이 arm은 실제 reference와 decided 위치 119개 중 59개에서 어긋난다.
- **LSA.** 각 layer의 `self_attn.0`이 indexer를 가진다. `score = sum_h relu(q_h . k) * weights_proj(x)_h * H^-0.5 * D^-0.5`이다. 앞의 16개 열과 query에서 끝나는 1024개 key 창은 `+inf`로 강제되고, causal mask가 적용된 뒤 `argpartition`이 상위 2048개를 남긴다. `self_attn.1`은 같은 선택을 재사용한다. 캐시된 key가 2048개 이하이면 아무것도 고르지 않으므로, 짧은 테스트로는 sparse와 dense를 구별할 수 없다.
- **RoPE.** 8192 위치에 factor 120인 `deepseek_yarn`이다. MLA는 traditional 방식으로 회전하고, indexer는 128차원 중 앞 64차원에 `indexer_rope_interleave`를 따른다.

## 2. 변경 요약

| 영역 | 변경 |
| --- | --- |
| 모델 | 새 `src/models/longcat_flash_sparse.rs`(config, indexer, sparse MLA, decoder layer, n-gram embedding, model, sanitize). |
| 공유 LongCat | `LongcatFlashMLA::forward`를 `project`, `absorbed`, `geometry`, `attend_window`로 분리. n-gram 해싱을 `NgramEmbedding::hashed_ids`와 `advance_context`로 이동. 가시성을 `pub(super)`로 확대. 동작은 그대로. |
| 탐지 | `longcat_flash_sparse` 분기. `model_type`이 없고 `LongcatCausalLM`인 config도 이 계열로 해석. |
| 연결 | `ModelType`, registry(`mlxcel arch`), metadata, `LoadedModel`, 특수 weight 로더, 메모리 추정 라벨, TP replicated 목록, 서버의 hybrid-SSM prompt-cache 목록. |
| KV | `QUANTIZED_LATENT_FAMILIES`와 `MLA_LATENT_CACHE_FAMILIES`에 등록. `kv_arch`가 세 번째 캐시(항상 fp16인 indexer key 캐시)를 계산. `EXPERIMENTAL_LATENT_NOTES`에 `longcat_flash_sparse` int4 안내 추가. |
| 테스트 | 모델 테스트 15개, 그리고 detection, registry, rope-lift, special loader, kv_arch, turbo allowlist, MLA cache, `mla_latent_option_tests`, `turbo_args_tests`, hybrid SSM 테스트에 항목 추가. |
| 도구 | `scripts/longcat_sparse_oracle_trace.py`(mlx-vlm reference arm, 음성 대조군 `--ngram-fusion`, `--no-forced-columns`). |
| 문서 | `docs/supported-models.md` 항목과 KV 표, `docs/turbo-kv-cache.md`, 결과 문서 `docs/benchmark_results/longcat-flash-sparse-m3ultra-2026-11-23.md`, CHANGELOG. |

## 3. 기술적 결정

### 3.1 공유 MLA를 복사하지 않고 나누어 재사용

sparse MLA에는 ngram MLA와 같은 projection, 같은 latent 캐시 쓰기, 같은 attention 본체가 필요하다. 다른 것은 attend하는 창(decode에서는 gather한 행)과 mask(prefill에서는 sparse mask)뿐이다. `LongcatFlashMLA`를 복사했다면 #3161과 #3189에서 다듬은 양자화 latent 분기까지 포함해 300줄 가량이 중복됐을 것이다. `forward`를 `project`와 `attend_window`로 나누면 sparse 모듈은 그 사이에 선택 단계를 끼워 넣고, ngram 경로는 같은 조각을 같은 순서로 호출한다. 대가는 공유 코드 리팩터링이므로, ngram 체크포인트를 세 가지 shape에서 전후로 trace했고 결과 행은 바이트 단위로 같았다.

### 3.2 decode는 gather, prefill은 mask

upstream은 query가 하나일 때 선택된 latent 행을 gather하고, 여러 개일 때 boolean sparse mask를 만든다. 둘 다 유지했다. gather는 DeepSeek V3.2의 DSA decode가 이미 쓰는 `LatentWindow::gather_rows`를 거친다. 이 함수는 packed latent, 그 scale과 bias, rope 스트림을 함께 gather하므로 양자화된 창은 양자화된 채로 남고, absorbed 양자화 커널이 그것을 바로 읽는다. "LSA decode gather가 활성 latent 레이아웃을 읽는다"는 조건은 이렇게 dequantize 단계 없이 충족된다. prefill mask는 additive(`0`은 유지, `-inf`는 제외)이고 causal 조건과 합쳐진다. `attend_window`가 additive `FoldedMask::Array`를 받기 때문이다.

### 3.3 양자화 latent: 계열은 등록하고 기본값은 fp16 유지

메인테이너는 양자화 latent 모드를 거부하지 말고 받아들이라고 했다. `QUANTIZED_LATENT_FAMILIES`에 등록하면 다른 LongCat 계열처럼 `--kv-cache-mode int8`이 8bit latent를 저장한다. 기본 KV 모드는 fp16이므로 플래그를 주지 않는 사용자에게는 변화가 없다. indexer key는 모든 모드에서 fp16으로 남는 별도 캐시를 쓴다. 이 key가 선택을 결정하므로, 양자화하면 정밀도가 아니라 어떤 key를 attend하는지가 바뀐다.

| Latent | fp16 대비 decided 불일치(decode, 문맥 3072) | 결정 |
| --- | --- | --- |
| int8 | 0 / 78 | `--kv-cache-mode int8`로 제공 |
| int4 | 4 / 78, perplexity +23% | 실험 경고와 함께 허용. 경고문이 이 측정을 인용 |

### 3.4 리뷰 반영

리뷰어들은 정확성 결함은 찾지 못했고, 견고성 공백 세 가지를 지적했다. `dd5e0011`에서 고쳤다.

- **indexer 메모리와 정밀도.** 첫 버전은 모든 head의 score를 activation dtype으로 한꺼번에 계산했다. 긴 prompt에서는 indexer layer마다 `[B, 16, L, S]` 텐서가 세 번 만들어진다(128k key에서 2048행 chunk 하나에 약 25 GB의 일시 메모리). 이제 score는 upstream Metal 커널처럼 float32로 계산하고, score 원소가 2^24개를 넘는 단계에서는 head를 하나씩 `[B, 1, L, S]` 버퍼 하나에 더한다. sparse prefill mask는 layer마다 한 번, mask dtype으로 바로 만들어 두 sub-block이 공유한다.
- **로드 시점의 abort 경로.** cxx 연산은 오류를 반환하지 않는다. 그래서 config와 shape이 맞지 않는 weight는 첫 forward에서 프로세스를 abort시켰다(`weights_proj`가 한 행이면 broadcast되어 잘못된 선택을 만들었다). 이제 indexer projection, `k_norm`, 결합된 `post_proj`, embedder별 `oe_embed_proj` 조각을 로드할 때 검사한다.
- **`model_type`이 없는 config.** 탐지는 이미 `LongcatCausalLM`을 이 계열로 매핑했다. 하지만 KV 모드 해석과 prompt-cache의 hybrid 검사는 `model_type`만 봤다. 그래서 원본 체크포인트는 int8 요청이 거부되고 APC가 켜진 채로 남았을 것이다. 두 곳 모두 이제 architecture로 fallback한다.

수정한 빌드로 모든 trace arm을 다시 돌렸다. decided 위치 결과는 하나도 바뀌지 않았다. 근소차(near-tie) flip 수만 몇 위치 움직였는데, 선택 score를 float32로 계산하면 예상되는 변화다.

### 3.5 어느 음성 대조군이 구별하는가

이슈는 실패하는 음성 대조군을 요구했다. 후보는 forced column 끄기와 ngram 결합식이었고 둘 다 돌렸다. forced column을 끄면 이 corpus에서는 근소차만 움직인다(flip 39건, decided 0건). indexer가 이미 sink와 local 창을 높게 평가하기 때문이다. ngram 결합식은 decided 위치의 절반에서 실패한다. 그래서 trace 게이트는 결합식 arm으로 걸고, forced column 규칙은 sink와 local 열의 score를 일부러 꼴찌로 둔 단위 테스트로 고정했다.

## 4. 검증

모든 수치는 `longcat-flash-lite-sparse-4bit`에서 `scripts/compare_logit_traces.py --decided 2`로 얻었다(M3 Ultra, mlx-vlm 0.8.0 `d998a620`). corpus는 `docs/architecture.md` 뒤에 `docs/benchmarks.md`를 이은 것이다.

| 비교 | 위치 수 | decided 불일치 |
| --- | --- | --- |
| mlx-vlm 대 mlxcel, 64폭 forward, 문맥 2048 초과 | 1216 | 0 / 499 |
| mlx-vlm 대 mlxcel, 문맥 3072 뒤 한 토큰 decode(gather) | 256 | 0 / 81 |
| mlx-vlm `--ngram-fusion` 대 mlxcel | 1216 | 59 / 119 |
| fp16 대 int8 latent, decode | 256 | 0 / 78 |
| fp16 대 int8 latent, multi-token | 256 | 0 / 78 |
| fp16 대 int4 latent, decode | 256 | 4 / 78 |
| fp16 대 int4 latent, multi-token | 256 | 6 / 78 |
| int4 multi-token 대 int4 gather decode | 256 | 0 / 70 |

마지막 행은 codec의 손실과 gather를 분리한다. 같은 4bit 행으로 mask 아래 dequantize한 창을 attend하는 것과 gather한 packed 행을 attend하는 것이 일치한다. decode arm의 MLA dispatch 카운터는 `absorbed_composed=7168`(fp16, int4)과 `absorbed_fused=7168`(int8)이다. 256 단계 곱하기 28 sub-block이고, decompress된 decode 단계는 없었다.

- `longcat_flash_ngram` 전후: `64 2 8 2048`, 한 토큰 decode, `MLXCEL_TRACE_START_TOKEN`으로 2048 토큰 뒤에서 모두 결과 행이 바이트 단위로 같다.
- `mlxcel generate`는 3,121 토큰 prompt에 대해 fp16, int8, int4에서 정확한 요약을 낸다(리뷰 수정 후 fp16에서 다시 확인). `mlxcel-server`는 `enable_thinking: false`로 짧은 요청과 3,129 토큰 chat 요청에 일관된 내용을 반환한다.
- 단위 테스트는 CI처럼 직렬로 실행: `longcat_flash`(39), `coverage_tests`, `mla_latent_option`, `turbo_args`, registry, detection, kv_arch, config_fields, special loader, rope lift, hybrid SSM, core `mla::`(117). Clippy `-D warnings`, fmt, 모든 contract check 통과.

## 5. 미검증 및 남은 항목

- MTP head(`model.mtp*`)는 버리고, 결합된 indexer-score Metal 커널은 포팅하지 않았다(score는 일반 MLX 연산). batching과 tensor parallelism은 범위 밖이며, TP에서는 replicated로 등록했다.
- 리뷰에서 나왔지만 하지 않은 것: key 전용 indexer 캐시(fp16 캐시가 한 열짜리 placeholder value를 저장해 1/129만큼 더 쓴다), `SparseRope`의 YaRN `mscale` scalar 캐싱, 같은 전체 head score 텐서 패턴을 가진 `deepseek_v32_indexer`와 chunk 단위 score helper 공유.
- 4bit 변환본만 돌렸다. 6bit, 8bit, 원본 bf16 체크포인트는 돌리지 않았다.
- 병렬 테스트 실행에서는 `longcat_flash_ngram` latent 테스트 두 개가 카운터 차이로 실패한다. `mla::stats`가 프로세스 전역이기 때문이다. CI가 Metal 테스트를 돌리는 방식인 직렬 실행에서는 통과한다. ngram latent 테스트 여덟 개만 골라도 같은 식으로 실패하므로, 이 경쟁은 기존 테스트 사이의 것이고 이 PR이 만든 것이 아니다.
- 발견했지만 고치지 않음: `LongcatFlashMLA::forward`는 ngram 체크포인트의 YaRN `rope_scaling`(32768에 factor 10, theta 5e6)을 무시하고 기본 theta로 회전한다.
- 체크포인트의 chat template은 기본으로 `<longcat_think>` 블록을 연다. 그래서 `enable_thinking: false` 없는 짧은 요청은 텍스트를 `reasoning_content`로 반환한다.

## 6. 배운 점

- **sparse attention 검사에는 선택 예산을 넘는 문맥이 필요하다.** key가 `index_topk`보다 적으면 LSA는 dense attention이다. 수백 토큰짜리 trace나 테스트는 indexer가 망가졌거나 빠져 있어도 통과한다. 단위 fixture는 `index_topk = 4`로 두어 다섯 번째 key부터 선택이 작동하게 했다. 실제 trace는 2048~3072 토큰에서 시작한다.
- **음성 대조군은 실패하는 것을 보여야 한다.** forced-column arm은 당연한 대조군처럼 보였지만 이 corpus에서는 구별하지 못했다. 실제로 돌려 봤기 때문에 게이트를 걸 arm이 결합식 쪽이라는 것을 알았다.
- **스레드가 공유하는 카운터는 직렬 실행에서만 유효하다.** `mla::stats`는 프로세스 전역 atomic 묶음이다. 그 단언은 `--test-threads=1`에서만 성립하고, 병렬에서는 잡음이다.
- **오류를 반환하지 않는 FFI 연산 앞에서는 shape을 로드 시점에 검사한다.** reshape나 matmul의 shape 불일치는 C++ 예외를 거쳐 `std::terminate`가 되므로, 첫 forward까지 미루면 서버 전체가 죽는다.

## 관련 커밋

| 해시 | 메시지 |
| --- | --- |
| `adb9e900` | feat(models): add LongCat-Flash-Lite-Sparse (longcat_flash_sparse) |
| `8c3b9531` | docs(models): record LongCat-Flash-Lite-Sparse traces and int4 note |
| `dd5e0011` | fix(models): harden the LongCat-Flash-Lite-Sparse indexer and config paths |
