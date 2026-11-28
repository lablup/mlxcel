# PR #3279: DeepSeek V3.2 어텐션과 인덱서에 YaRN RoPE 적용

**날짜**: 2026-11-23
**상태**: PR 열림(#3178을 닫음). 단위 테스트는 ROCm gfx1151에서만 돌렸다. DeepSeek-V3.2-4bit의 긴 문맥 perplexity 재측정은 M3 Ultra에서 할 예정이며 아직 하지 않았다.

## 요약

`deepseek_v32`(그리고 같은 어텐션을 쓰는 `glm_moe_dsa`)는 MLA `q_pe` / `k_pe`와 DSA 인덱서의 쿼리와 키를 plain RoPE로 회전했다. DeepSeek-V3.2 체크포인트는 YaRN(`factor` 40, `original_max_position_embeddings` 4096)을 선언하는데, `rope_scaling` 블록은 softmax scale 보정에만 쓰였다. 4096 위치를 넘어서면 저주파 쌍이 최대 40배 빠르게 회전했고, 이는 #3178이 보고한 perplexity 급등(M3 Ultra, fp16 캐시 기준 4096 토큰 뒤 7.3, 8192 뒤 1403, 16384 뒤 2080)과 맞아떨어진다.

이 PR은 `DsaRope`(`src/models/deepseek_v32_rope.rs`)를 추가한다. `DsaRope`는 `rope_scaling` 블록 전체를 공용 `RopeScalingKind::resolve` / `yarn_rope_setup` 경로로 모델당 한 번 해석하고, 여섯 군데 회전 지점 모두에서 쓰인다. 블록이 없거나 `default`인 설정은 이전과 똑같은 plain `fast_rope` 호출을 유지한다. 단위 테스트로 텐서 수준의 수정은 확인했다. 다만 작업한 ROCm 호스트에서는 체크포인트를 올릴 수 없어서, YaRN만으로 perplexity 차이가 해소되는지는 아직 모른다.

## 1. 문제 정의

#3178은 `mlx-community/DeepSeek-V3.2-4bit`에서 fp16 캐시로 teacher-forced perplexity를 쟀다. 4096 토큰 뒤 7.3, 8192 뒤 1403, 16384 뒤 2080이었다. prefill을 2048 토큰 단위로 나눠 넣어도 4096 값은 크게 변하지 않아서(7.51 대비 7.32) chunking은 주원인에서 제외됐다. 깨지는 지점이 `original_max_position_embeddings` 바로 뒤였기 때문에 RoPE scaling이 의심을 받았다.

이슈는 `docs/benchmark_results/mla-quantized-latent-dsa-m3ultra-2026-11-19.md`의 16384 토큰 행(int8 대 fp16 top-k overlap)이 이 잘못된 기준 위에서 측정됐다는 점도 지적했다. 재측정 전까지 그 행들은 믿기 어렵다.

## 2. 근본 원인

upstream mlx-lm은 어텐션 RoPE와 인덱서 RoPE를 모두 `initialize_rope(..., scaling_config=args.rope_scaling)`로 만든다. 배포된 설정(`yarn`, `factor` 40, `original_max_position_embeddings` 4096, `beta_fast` 32, `beta_slow` 1, `mscale` 1.0, `mscale_all_dim` 1.0)에서는 이것이 `YarnRoPE`가 된다.

mlxcel의 `RopeScaling` 구조체는 `type`, `factor`, `mscale_all_dim`만 읽었고, 이 값들은 softmax scale 보정에만 들어갔다. 회전은 모두 `fast_rope(.., rope_base, 1.0, offset)`였다.

- `MLAAttention::forward`: `q_pe`와 `k_pe`.
- `MLAAttention::forward_batched_decode`: `fast_rope_batched`로 `q_pe`와 `k_pe`.
- `Indexer`: 키와 쿼리.

`qk_rope_head_dim`이 64이면 주파수 쌍은 32개다. YaRN 보정 램프는 0번부터 10번 쌍은 그대로 두고(extrapolation), 23번 이후 쌍은 40으로 나누며(interpolation), 그 사이는 섞는다. 그래서 plain RoPE는 고주파 쌍에서만 YaRN과 일치한다. 4096 위치 안에서는 저주파 쌍이 아직 충분히 돌지 않아서 오차가 드러나지 않지만, 4096을 넘으면 사전학습 때 보지 못한 각도에 도달한다.

## 3. 변경 요약

| 파일 | 변경 |
|---|---|
| `src/models/deepseek_v32_rope.rs`(신규, 193줄) | `DsaRope`: 블록 해석, `qk_rope_head_dim` 검증, 테이블과 YaRN magnitude 배율 적용 |
| `src/models/deepseek_v32.rs` | `rope_scaling`을 `RopeScalingSpec`으로 바꾸고 private `RopeScaling` 구조체 제거. MLA forward 두 경로 모두 `DsaRope` 호출. `DsaRope`는 모델당(전체 모델, 파이프라인 stage 모델) 한 번 만들고 레이어마다 복제 |
| `src/models/deepseek_v32_indexer.rs` | 인덱서 키와 쿼리가 같은 `DsaRope`를 쓰고, `traditional` 플래그로는 계속 `indexer_rope_interleave`를 씀. `load_with_rope`가 공유 인스턴스를 받음 |
| `src/models/glm_moe_dsa.rs` | `rope_parameters`를 손으로 고른 키 세 개 대신 `RopeScalingSpec::from_lookup`으로 전체 `RopeScalingSpec`에 매핑 |
| `src/models/rope_overrides.rs` | `resolve_base`의 "used by" 주석에 `DsaRope` 추가 |
| `docs/llama-server-compat.md` | DeepSeek V3.2를 "공용 경로 밖에서 YaRN을 구현하는 계열"에서 빼고 `--rope-*` / `--yarn-*` override를 따르는 계열 목록에 추가 |
| `src/models/deepseek_v32_rope_tests.rs`(신규, 505줄) | 회귀 테스트(5절) |
| `CHANGELOG.md` | `[Unreleased]` Fixed 항목. perplexity 재측정이 남아 있다고 명시 |

`DsaRope`의 동작은 세 가지다.

1. `from_args`는 0이거나 홀수이거나 `min(index_head_dim, qk_nope_head_dim + qk_rope_head_dim)`보다 넓은 `qk_rope_head_dim`을 거부한다. `deepseek_yarn`을 `yarn`으로 바꾸고(upstream은 둘 다 받는다), 학습 문맥 fallback 없이 `RopeScalingKind::resolve`를 호출한다.
2. `apply` / `apply_batched`는 주파수 테이블이 있으면 `fast_rope_with_freqs` / `fast_rope_batched_with_freqs`를, 없으면 해석된 base와 scale로 plain `fast_rope` / `fast_rope_batched`를 호출한다.
3. `magnitude`는 upstream `YarnRoPE.__call__`처럼 앞쪽 `dims`개 feature에만 `yarn_get_mscale(factor, mscale) / yarn_get_mscale(factor, mscale_all_dim)`를 곱한다. 배포 설정에서는 정확히 1.0이므로 바로 반환하고 추가 연산이 없다.

`load_mla_attention_with_rope`의 softmax scale 보정은 그대로다. upstream도 보정과 magnitude 배율을 둘 다 적용하므로 어느 쪽도 두 번 들어가지 않는다.

## 4. 기술적 선택과 그 이유

**DeepSeek V2 사본 대신 공용 YaRN resolver 재사용.** `deepseek_v2.rs`에는 자체 YaRN 구현이 있다. 이 PR은 이슈의 제안대로 `rope_utils`(`RopeScalingSpec`, `RopeScalingKind::resolve`, `yarn_rope_setup`)를 거친다. 세 번째 YaRN 구현을 만들지 않게 되고, `--rope-*` / `--yarn-*` 런타임 override도 DeepSeek V3.2와 GLM-5에 닿게 된다. 모델 워커가 확인하는 카운터를 공용 resolver가 올리기 때문에, 이 계열은 이제 거부되지 않고 override를 받아들인다. 문서 변경이 이를 반영한다.

**한 번 만들고 레이어마다 복제.** 테이블은 `DeepSeekV32Model::from_weights`와 파이프라인 stage 로더에서 한 번씩 해석되고, 각 어텐션 블록과 인덱서는 `duplicate()` 핸들(원소 32개 배열의 lazy copy)을 가진다. 로딩 중 레이어마다 `powf` 루프를 다시 돌지 않는다. 블록 단위 로더 `load_mla_attention`과 `Indexer::load`는 `#[cfg(test)]`에서만 남았다.

**plain 경로는 비트 단위로 동일하게.** 블록이 없거나 `default`이면 `DsaRope::apply`는 이전과 같은 `fast_rope(x, dims, traditional, base, 1.0, offset)` 호출을 한다. `absent_rope_scaling_keeps_the_plain_rope_call`은 단일 경로와 batched 경로 모두 최대 차이가 정확히 0.0인지 확인한다. GLM-5에서 이 점이 중요하다. 공개 설정(`zai-org/GLM-5`, `mlx-community/GLM-5-4bit`)은 `rope_parameters` `{rope_theta: 1000000, rope_type: default}`를 가지므로 출력이 바뀌지 않는다.

**학습 문맥 fallback을 넘기지 않음.** 공용 `--yarn-orig-ctx` sentinel 문서는 `max_position_embeddings`로 fallback한다고 설명한다. `DsaRope`는 `None`을 넘기므로 `original_max_position_embeddings`가 없으면 upstream `YarnRoPE`처럼 4096으로 간다. upstream과 맞추기 위한 선택이지만, 일반 문서와 다르다는 점은 문서화되지 않았다(리뷰의 LOW 항목).

**GLM-5 블록 전체 매핑.** upstream은 `self.rope_scaling = self.rope_parameters`로 둔다. 예전 매핑은 키 세 개만 복사하고 나머지를 버렸으므로, GLM-5 YaRN 블록이 들어오면 `original_max_position_embeddings`와 beta 값을 잃었을 것이다. `RopeScalingSpec::from_lookup`은 모든 키를 유지한다.

**할당 전에 rope 폭 검증.** 보안 리뷰는 `from_args`가 가중치 shape 검사보다 먼저 실행되고, `yarn_rope_setup`이 상한 없이 `qk_rope_head_dim / 2`개 원소를 할당한다는 점을 찾았다. 잘못된 설정 하나로 로더가 기가바이트 단위를 할당할 수 있었다. 70ad89cd는 0, 홀수, 너무 넓은 `qk_rope_head_dim`을 테이블을 만들기 전에 이름 있는 에러로 거부한다.

## 5. 테스트와 검증

`src/models/deepseek_v32_rope_tests.rs`:

- `yarn_table_matches_the_upstream_formula_at_checkpoint_geometry`: `qk_rope_head_dim` 64와 배포 블록에서 0, 10, 16, 20, 24, 31번 쌍이 upstream `YarnRoPE` 공식으로 float64에서 따로 계산한 값과 맞는다(상대 오차 1e-5 미만). `yarn`과 `deepseek_yarn` 둘 다 확인하고, magnitude 배율이 정확히 1.0인지도 본다.
- `absent_rope_scaling_keeps_the_plain_rope_call`: plain 경로가 비트 단위로 같다(4절).
- `yarn_magnitude_scales_only_the_rotated_dims`: `mscale` 2.0, `mscale_all_dim` 1.0에서 배율이 upstream 값(1.26948)과 같고 회전하는 feature에만 적용된다.
- `prefill_and_decode_cache_yarn_rotated_keys_past_the_original_context`, `batched_decode_caches_yarn_rotated_keys_per_row`: `original_max_position_embeddings` 앞뒤에서 캐시된 `k_pe`와 인덱서 키가 `yarn_rope_setup` + `fast_rope_with_freqs` 기준값과 같고 plain RoPE와는 다르다. batched 경우에는 경계 양쪽에 행을 하나씩 둔다.
- `indexer_queries_and_keys_rotate_with_yarn`: 인덱서 쿼리와 키에 대한 같은 검사.
- `yarn_model_batched_decode_matches_sequential`: YaRN에서 batched decode가 순차 decode와 같다.
- `mla_queries_rotate_with_the_key_table_past_the_original_context`: 구현 리뷰에서 추가됐다. 캐시된 키만으로는 `q_pe` 회전 오류를 볼 수 없으므로 RoPE의 상대 위치 성질을 쓴다. 위치 0, 1과 먼 위치에서의 두 토큰 prefill, 그리고 멀리 떨어진 두 행의 batched decode가 같은 어텐션 출력을 내야 한다. `q_pe` 지점 하나만 되돌려도 실패한다(허용치 1e-4에 대해 1.45e-3 차이, 수정 후에는 1e-5 미만). 이 테스트 전에는 `q_pe` 두 지점을 모두 되돌려도 `deepseek_v32`, `glm_moe_dsa` 테스트가 전부 통과했다.
- `malformed_rope_width_is_refused_before_the_table_is_built`: 폭 0, 7, 14를 거부한다.

여섯 회전 지점을 main으로 되돌리고 테스트를 남기면 캐시와 인덱서 테스트가 실패한다(캐시된 `k_pe`는 최대 2.04, 인덱서 쿼리는 11.4 차이).

ROCm gfx1151에서 실행한 명령:

- `cargo test --lib --profile test-fast --features rocm -- deepseek_v32 glm_moe_dsa rope_utils rope_overrides`: 99개 통과(두 번 실행).
- `cargo clippy --lib --tests --features rocm -- -D warnings`, `cargo fmt --check`, `scripts/ci/check_used_by.py`: 문제 없음.

워크스페이스 전체 테스트와 clippy는 로컬에서 돌리지 않았고, Metal이나 CUDA에서는 아무것도 실행하지 않았다.

**테스트 락 수정(b92ca275).** forward 테스트 다섯 개가 공용 MLX 테스트 락 없이 decode를 돌렸다. 병렬 테스트 스레드에서 이 테스트들이 프로세스 전역 MLA decode 경로 카운터를 올리는 동안 `a_forced_int4_slot_keeps_the_indexer_key_at_eight_bits`가 그 카운터를 delta로 재고 있었고, 그 테스트가 한 번 실패했다. 이제 forward 테스트도 다른 DeepSeek V3.2 테스트 파일처럼 `embedding_test_support::mlx_test_guard()`를 잡는다.

## 6. 리뷰 결과

**구현 리뷰**(PR 코멘트 7087109122): HIGH 하나, 테스트되지 않은 `q_pe` 지점. 65288130의 상대 위치 테스트로 해결했다. LOW 세 개는 열어 두었다. 문서화되지 않은 4096 학습 문맥 fallback, 공용 YaRN helper가 upstream의 `truncate` 키를 무시하는 점(배포된 DeepSeek-V3.2나 GLM-5 설정에는 없음), `deepseek_v32.rs`의 private `yarn_get_mscale`이 `rope_utils::yarn_get_mscale`과 중복되는 점이다.

**보안과 성능 리뷰**(PR 코멘트 7087141070): CRITICAL이나 HIGH는 없다. `YarnParams::resolve`의 필드별 검사가 없거나 0이거나 음수이거나 유한하지 않은 값을 panic 없이 거부하는 점, magnitude 비율에서 0으로 나누기가 생기지 않는 점, 보정 램프가 underflow하지 않는 점, 배포 설정에서 decode에 추가 연산이 없는 점을 확인했다. MEDIUM 하나(큰 `qk_rope_head_dim`으로 인한 무제한 테이블 할당)는 70ad89cd의 검증으로 고쳤다. LOW 두 개는 고치지 않았고, 둘 다 기존 공용 코드에 있다.

- 필드별 검사를 통과한 값들이 조합되면 여전히 overflow할 수 있다. `f32::MAX`에 가까운 `factor`는 테이블에 NaN을 만들고, 극단적인 `mscale`은 비율을 inf / inf로 만든다. 완성된 테이블과 배율이 유한하고 양수인지 검사하면 막을 수 있다.
- softmax scale 보정은 `factor`와 `mscale_all_dim`을 유한성 검사 없이 읽는다. `inf` factor가 들어오면 YaRN 테이블은 fallback으로 가는데 softmax scale은 inf가 된다. 또 테이블을 바꾸는 `--rope-scaling` override를 보정은 무시한다.

## 7. 남은 작업과 한계

- **perplexity 재측정을 하지 않았다.** ROCm gfx1151 호스트(30 GB)는 DeepSeek-V3.2-4bit를 올릴 수 없다. #3178의 8192, 16384 뒤 fp16 perplexity 기준, #3163의 16384 top-k overlap 재측정, `docs/benchmark_results/mla-quantized-latent-dsa-m3ultra-2026-11-19.md` 갱신은 M3 Ultra에서 할 일로 남아 있다. 명령은 `MLXCEL_TRACE_PREFILL_CHUNK=2048 logit_trace models/deepseek-v3.2-4bit corpus.txt 64 8 8 8192 > fp16-8k.tsv`이다. 이 측정 전까지 YaRN 누락은 텐서 수준에서 확인된 결함일 뿐, perplexity 급등 전체를 설명한다고 확인된 것은 아니다. perplexity가 여전히 높으면 이슈가 꼽은 다음 후보(`index_topk` 2048을 넘었을 때의 인덱서 top-k와 sparse prefill mask, decode gather, chunked prefill 문맥 경로)를 `MLXCEL_DSA_DENSE`로 나눠 확인해야 한다.
- **테스트 위치.** 이슈는 회귀 테스트를 `deepseek_v32_tests.rs`에 두라고 했지만, 테스트 대상 모듈 옆의 새 파일 `deepseek_v32_rope_tests.rs`에 들어갔다.
- **범위 밖: DeepSeek V3.** `src/models/deepseek_v3.rs`는 main에서 같은 결함을 가진다. `DeepSeekV3Attention`은 `q_pe` / `k_pe`에 plain `fast_rope` / `fast_rope_batched`를 쓰고 `rope_scaling`은 softmax scale에만 읽는데, DeepSeek-V3 체크포인트도 같은 YaRN 블록을 가진다. 별도 이슈가 필요하다.

## 8. 학습 포인트

- 설정 블록을 일부만 파싱하면 빠진 기능이 가려진다. 여기서는 `rope_scaling`을 읽고 있어서 모델이 YaRN을 아는 것처럼 보였지만, 실제로는 softmax scale 쪽 절반만 연결돼 있었다. 공용 spec 타입(`RopeScalingSpec`) 전체를 파싱하면 빠진 절반이 드러난다.
- YaRN 자리에 plain RoPE가 들어가도 짧은 문맥 테스트에서는 보이지 않는다. 두 테이블이 고주파 쌍에서는 같고 저주파 오차는 천천히 커지기 때문이다. 테스트는 `original_max_position_embeddings`를 넘는 offset을 써야 한다.
- 캐시된 키 검사로는 쿼리를 검증할 수 없다. 쿼리 회전 결과는 저장되지 않기 때문이다. 같은 토큰을 위치만 옮겨도 같은 어텐션이 나와야 한다는 상대 위치 불변성 검사는 기준 구현 없이 쿼리 쪽을 시험한다.
- 이 crate에서 forward를 돌리는 테스트는 프로세스 전역 카운터를 건드리므로, 그 카운터를 검사하지 않더라도 `mlx_test_guard()`를 잡아야 한다.
