# 기술 보고서: PR #3270 - Arcee AFM dense (`arcee`) 지원

**날짜**: 2026-11-23
**상태**: Open (draft PR, 머지 전 작성)
**위험도**: 낮음 (새 모델 타입 추가; 공유 코드 변경은 문서용 `/// Used by:` 및 목록 주석뿐)

## 요약

`mlxcel`은 `arcee-ai/AFM-4.5B`에서 `Unsupported model type: arcee`로 종료했다(#1719). mlx-lm과 mlx-vlm 모두 `arcee`를 구현하지 않아 parity 기준이 없었기 때문에 이슈가 막혀 있었고, 2026-11-23 메인테이너가 Hugging Face transformers `ArceeForCausalLM`을 기준으로 승인했다. 이 PR은 공유 Llama attention(GQA, YaRN, fused QKV, 캐시 소유 dispatch, split-attention batched decode)을 재사용하고 gate 없는 squared-ReLU MLP를 더한 `src/models/arcee.rs`와, `logit_trace` chunk plan을 CPU float32의 transformers 5.24.0으로 채점하는 `scripts/arcee_oracle_trace.py`를 추가한다. 실제 bf16 체크포인트에서 teacher-forced trace는 2048개 위치와 위치 8192까지의 8192개 위치 모두에서 decided 위치 불일치가 0이며, plain ReLU 또는 YaRN을 제거한 기준 arm은 실패한다.

## 1. 문제 정의

AFM-4.5B는 블록 하나를 제외하면 Llama 형태다. `hidden_act: relu2`이고 `gate_proj`가 없어서 MLP는 `down_proj(relu(up_proj(x))^2)`이며 hidden 폭의 7.2배다. RoPE는 원래 4096 컨텍스트 위에 `factor` 20인 YaRN이다. `arcee`를 일반 Llama 경로로 보내면 `gate_proj`가 없어 로드에 실패하고, 디코더를 새로 쓰면 YaRN, KV 캐시, batched decode 구현을 중복하게 된다. 트리와 이슈에는 AFM이라는 이름이 셋(AFMoE, Apple AFM-7, 이 모델) 있으므로 등록 시 서로 구분해야 했다.

두 번째 문제는 기준 구현이다. transformers는 몇몇 경우(선언된 `attention_factor`, `mscale_all_dim` 없는 `mscale`, `truncate: false`, 누락된 `factor`, 최상위 `original_max_position_embeddings`)에서 공유 mlx-lm 방식 경로와 YaRN을 다르게 유도한다. AFM-4.5B에서는 둘이 일치하며, 일치하지 않을 config는 다르게 디코딩하지 않고 거부해야 한다.

## 2. 변경 사항

- `src/models/arcee.rs`(신규): 자체 `Deserialize`를 가진 `ArceeConfig`(공유 `rope_parameters` lift에 더해 `truncate`, `partial_rotary_factor`, 최상위 원래 컨텍스트, `type` / `rope_type` 충돌을 담는 `RopeScreen`), `validate()`, `to_llama3_args()`(전체 struct literal), `ArceeMLP`, `ArceeBlock`(`llama3::Attention` + `ArceeMLP` + `fused_add_rms_norm`을 쓰는 RMSNorm 두 개), 그리고 `last_row_logits`, native batched decode, batched 및 maskless padded prefill, paged decode backend를 지원하는 `ArceeModel`.
- 등록: `ModelType::Arcee`(enum, `ALL_MODEL_TYPES`, "Arcee" 제목 아래 설명, metadata 테스트 목록), `detection.rs`, `registry.rs`, `model_metadata.rs`(config-backed), `LoadedModel`, `execution/memory_estimate.rs`, TP fallback(`"arcee"`, 샤딩 불가), batched-decode coverage 목록.
- `scripts/arcee_oracle_trace.py`(신규): `--arm relu`, `--arm no-yarn`, `--dtype bfloat16`, `--no-prefix-sharing`, `--report-range`를 가진 transformers 기준 arm. safetensors만 로드하고 원격 코드는 실행하지 않는다.
- 목록 주석: `relu_squared`, `fused_add_rms_norm`, `forward_split_attention`, RoPE override 패밀리 목록에 Arcee 추가.
- 문서와 데이터: `docs/supported-models.md`, `CHANGELOG.md`, `benchmarks/metal_m3ultra_2026-11-23_afm-4.5b.csv`.
- 테스트: `src/models/arcee_tests.rs`(16개)와 detection 테스트 하나.

## 3. 기술적 결정

**`llama3::MLP`를 바꾸지 않고 공유 attention 위에 자체 블록을 둔다.** `llama3::MLP`의 public `gate_proj` 필드는 tensor-parallel runtime과 여러 테스트가 읽는다. gate를 optional로 바꾸면 모든 Llama 사용처가 영향을 받는다. 별도 블록은 약 100줄이며 Llama 그래프는 바이트 단위로 그대로 유지된다.

**transformers와 공유 경로가 다른 곳은 거부한다.** transformers `_compute_yarn_parameters`의 attention factor를 공유 경로의 `yarn_get_mscale(f, mscale) / yarn_get_mscale(f, mscale_all_dim)`와 비교해 다르면 거부한다. 공유 resolver가 경고만 내고 스케일 없는 회전으로 떨어뜨릴 경우(알 수 없는 RoPE 타입, factor 없는 `linear` / `llama3` / `yarn` 블록)와 공유 경로가 무시하는 키도 거부한다. 이를 공유 YaRN 코드에 반영하면 Arcee 체크포인트에는 없는 설정 때문에 모든 YaRN 패밀리의 경로가 바뀐다.

**screening은 `Deserialize` 안에서 한다.** 첫 버전은 `truncate`를 `parse_config`에서 읽었는데 adapter 경로는 이를 거치지 않는다. 리뷰에서 이를 `Deserialize` 구현으로 옮겨 `ArceeConfig`를 만드는 모든 경로가 같은 키를 보게 했다.

**여기서는 f16 활성값이 안전하다.** mlxcel은 로드 시 bf16 가중치를 f16으로 변환한다. BitNet과 nanochat은 squared-ReLU나 정규화되지 않은 residual이 f16 범위를 넘어 이 변환을 건너뛴다. AFM-4.5B에서는 기준 구현의 squared-ReLU 최대값이 165, residual 최대값이 363이라 변환을 유지한다.

## 4. 측정

M3 Ultra, `models/afm-4.5b`(디스크 bf16, 로드 시 f16). 기준: transformers 5.24.0, torch 2.15.0, tokenizers 0.23.3, CPU, float32. `compare_logit_traces.py --decided 2`.

| 비교 | 위치 수 | decided 불일치 | perplexity |
|---|---|---|---|
| ref vs mlxcel, `docs/architecture.md 256 8 8` | 2048 | 0 / 686 | 39.808 vs 39.824 |
| ref vs mlxcel, 최대 6144 토큰 뒤 2048 chunk 4개 | 8192 | 0 / 2942 | 17.855 vs 17.859 |
| ref vs mlxcel `MLXCEL_TRACE_BATCH_ROWS=1` / `=4` | 2048 | 각 0 / 686 | +0.05% / +0.07% |
| ref `--arm relu` vs mlxcel | 2048 | 44 / 44 | 16459 vs 39.8 |
| ref `--arm no-yarn` vs mlxcel | 2048 | 11 / 618 | 42.2 vs 39.8 |
| ref `--arm no-yarn` vs mlxcel, long | 8192 | 126 / 1637 | 140.1 vs 17.9 |

positive arm의 모든 불일치는 기준의 top-two 차이가 0.5 미만인 위치이며 rank 2에 해당한다. 생성: 195 토큰 chat prompt에서 decode 60~63 tok/s, prefill 1557 tok/s. `mlxcel-server`는 동시 요청 두 개와 스트리밍 요청 하나를 처리했다.

## 5. 한계

- 양자화된 Arcee export는 테스트하지 않았다.
- Tensor parallelism은 거부된다.
- 리뷰 후 추가한 screening 커밋은 forward 경로를 바꾸지 않으며, 두 번째 실제 체크포인트 trace가 아니라 단위 테스트로 검증했다.
