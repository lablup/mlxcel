# PR #2475: Qwen3.8-Flash-Next (qwen4_exp) 지원 구현 리뷰

> **2026-10-23 하드닝 업데이트:** 아래는 2026-10-20 당시의 리뷰와 미완료 항목을 보존한 기록이다. 이후 익명 배치의 행별 상태, OptiQ의 이미 보정된 norm 가중치, bf16 reduction과 recurrent state 정밀도, 네이티브 QSA·MRoPE를 수정했다. 실제 80.8 GB 체크포인트로 CLI·서버의 텍스트·이미지·영상·긴 문맥과 스트리밍을 검증했고, 전체 테스트 13,308개 및 M3 Ultra의 레퍼런스 성능 비교를 통과했다. 최신 결과·원시 데이터·레퍼런스 리비전·제외 기능은 [검증 보고서](../docs/benchmark_results/qwen4-exp-m3-ultra-2026-10-23.md)를 참조한다.

**작성일**: 2026-10-20
**상태**: 리뷰 완료, PR 브랜치에 수정 4건 푸시, 결정이나 128 GB 머신이 필요한 항목 3건 남음
**언어**: Rust, Python(픽스처 생성기만)
**위험도**: Medium (전용 `qwen4_exp` detection arm 뒤의 새 모델 패밀리, 공용 함수 변경 없음, 실제 80.8 GB 체크포인트는 아직 실행하지 않음)

## 결론

이 PR은 issue #2406의 항목 중 실제 체크포인트 없이 확인할 수 있는 것을 모두 구현했고, Qwen3.5 VLM과 같은 방식으로 코드 흐름에 연결되어 있다: detection, `model_metadata`, registry(`mlxcel arch`), `LoadedModel`, `VlmRuntimeRef::Qwen`, `qwen_video_runtime`, `model_type_has_native_video`, CLI 비디오 placeholder 개수. placeholder 코드는 없다. 업스트림이 거부하는 config 조합은 모두 이름 붙은 에러를 갖고, Qwen3.5 대비 추가된 네 가지(zero-centered RMSNorm, gated hyper-connection, 해시 n-gram PLE, QSA)는 mlx-vlm 출력에 고정된 실제 구현이며, 범위 밖 입력(`ple_storage`, FP8 `weight_scale_inv`, 비어 있지 않은 `deepstack_visual_indexes`)은 이름을 밝혀 거부한다.

리뷰에서 CRITICAL이나 HIGH 이슈는 나오지 않았다. MEDIUM 2건과 LOW 3건(재사용 누락 1건, `docs/adding-models.md` 대비 테스트 공백 1건, 죽은 코드, 사소한 테스트 헬퍼, 부정확한 문서)을 고쳤고, 다음 세 항목이 남았다:

- 포트가 새로 호출하는 공용 헬퍼의 `Used by:` 목록에 Qwen4-Exp가 없다. 공용 파일의 주석 수정이라 이번 리뷰에서는 고칠 수 없었다.
- sequence id 없이 `forward_batched`를 부르면 모든 행이 하나의 fallback state를 공유한다. Qwen3.5도 같고 엔진은 항상 id를 넘기므로, Qwen4-Exp만 고칠 일이 아니라 패밀리 전체의 결정이다.
- 업스트림은 prefill에서 폭이 좁은 양자화 projection 네 개를 행마다 계산한다(`_row_invariant_quantized_projection`). 포트는 배치 projection을 쓴다. 영향이 있는지는 실제 체크포인트의 logit trace만 알려 줄 수 있다.

## 1. 이 PR이 해야 하는 일

Issue #2406은 `model_type: qwen4_exp`(`Qwen4ExpForConditionalGeneration`)를 CLI와 서버에서 텍스트, 이미지, 비디오를 처리하는 VLM으로 추가하라고 요구한다. mlx-vlm의 `mlx_vlm/models/qwen4_exp`에서 포팅하고, 컴포넌트별 fixture-parity 테스트, 업스트림 logit과 일치하는 4-layer 모델, config 검증 에러마다 테스트 하나, `mlx-community/Qwen3.8-Flash-Next-OptiQ-2bit` 실행과 teacher-forced logit trace를 갖춰야 한다. 마지막 두 가지는 128 GB 머신이 필요하고, PR 본문은 이를 아직 남은 항목으로 적어 두었다.

리뷰는 PR을 issue, PR 본문, `CONTRIBUTING.md`, `docs/code-guidelines.md`, `docs/adding-models.md`에 비추어 확인했고, 각 컴포넌트를 mlx-vlm `9527f2da`의 업스트림(`language.py`, `config.py`, `qwen4_exp.py`, `qsa_kernel.py`)과 비교했으며, 두 번째 의견으로 Codex 리뷰어(`codex review --base origin/main`)를 돌렸다.

## 2. 발견 사항

| 심각도 | 위치 | 내용 | 상태 |
|--------|------|------|------|
| MEDIUM | `src/models/qwen4_exp_ple.rs:252` (`ShardedEmbedding::lookup`) | shard별 gather 결과를 binary `concatenate`의 left-deep 체인으로 이었다. 건드린 shard마다 노드가 하나씩(최대 128개) 생긴다. 바로 그 체인을 대신하려고 있는 기존 `mlxcel_core::concatenate_many`를 쓰지 않았다 | 수정, `fee17250` |
| MEDIUM | `src/models/qwen4_exp_config_tests.rs:209` | `docs/adding-models.md`("Quantization Parameter Bounds")는 per-prefix quantization 블록을 공용 `HOSTILE_QUANT_PARAMS` 목록으로, 실제 로더를 거쳐, positive control과 함께 테스트하라고 한다. PR은 한 위치에서 한 쌍(bits 0)만 테스트했다 | 수정, `9bf50f43` |
| MEDIUM | `src/models/gated_delta.rs:54`, `:752`; `src/models/switch_layers.rs:1961`; `src/models/qwen_mrope_state.rs:77`-`184` | `docs/code-guidelines.md`("Shared Function Comments", "When adding a new model that uses an existing shared function")에 따르면 `GatedDeltaCache`, `gated_delta_update`, `moe_weighted_sum`, `MRopeState` 메서드의 `Used by:` 목록에 Qwen4-Exp를 추가해야 한다 | 남음: 공용 파일, 4절 참고 |
| LOW | `src/models/qwen4_exp_ple.rs` (`PleLayer::from_weights`) | head vocabulary가 패딩된 테이블에 들어가는지 보는 검사는 결코 실패할 수 없었다. `padded_vocab_size`가 그 합을 올림한 값이다 | 수정, `fee17250` |
| LOW | `src/models/qwen4_exp_parity_tests.rs` | `mrope_for`는 세 필드를 `InterleavedMRoPE::new`에 넘기기만 했다. 첫 모델 테스트는 `tiny_model` / `prompt_ids`가 이미 만드는 것을 다시 만들었다 | 수정, `22dbfacd` |
| LOW | `CHANGELOG.md`, `docs/supported-models.md`, `src/models/qwen4_exp_qsa.rs:15`, `tests/fixtures/qwen4_exp/generate_fixtures.py:9` | "2048 토큰까지 dense"는 세 토큰 어긋난다. 선택은 완전한 블록 513개, 즉 2052번째 토큰에서 시작한다. 상태 목록에는 pooled block key가 빠져 있었다. 생성기는 존재하지 않는 `src/models/qwen4_exp/parity_tests.rs`를 가리켰다 | 수정, `7bb7127b` |
| LOW (Codex P2) | `src/models/qwen4_exp.rs:415` (`forward_rows_sequential`) | `seq_ids = None`이고 행이 둘 이상이면 모든 행이 같은 fallback state를 읽고 쓴다 | 남음: `Qwen35Model::forward_batched_rows_sequential`(`src/models/qwen3_5.rs:1889`)과 같음, 4절 참고 |
| LOW | `src/models/qwen4_exp_hyper.rs:102`, `qwen4_exp_gated_delta.rs:151`-`152`, `qwen4_exp_moe.rs:133` | 업스트림은 `block_inject_weight`, `in_proj_b`, `in_proj_a`, `shared_expert_gate`를 `_row_invariant_quantized_projection`(mlx-vlm `mlx_vlm/models/qwen4_exp/language.py:44`, `:809`-`810`, `:1194`, `:1203`에서 사용)에 통과시켜 양자화된 prefill 행이 decode와 같은 값을 계산하게 한다. 포트는 배치 projection을 쓴다 | 남음: logit trace로 측정, 4절 참고 |

## 3. 수정 내용

### 3.1 n-gram 테이블 gather를 concatenate 한 번으로

`ShardedEmbedding::lookup`은 n-gram id를 각자의 행 shard로 보내 shard별로 gather하고, 원래 위치 순서로 다시 정렬한다. 이어 붙이는 코드는 다음과 같았다:

```rust
gathered = Some(match gathered {
    None => rows,
    Some(previous) => concatenate(&previous, &rows, 0),
});
```

노드마다 그때까지 이은 전부를 복사하므로, 128개 shard 중 여러 개를 건드리는 청크는 커지는 prefix를 shard마다 한 번씩 복사했다. 이제 shard별 행을 모아 `mlxcel_core::concatenate_many(&parts, 0)`를 한 번 부른다(`src/models/qwen4_exp_ple.rs:273`). 마지막 `take`는 그대로이므로 행 순서도 같다. parity 테스트가 이 경로를 덮는다: PLE 픽스처의 테이블은 shard가 세 개이고 테스트 시퀀스의 id가 세 shard에 모두 걸친다(`ple_layer_matches_upstream_and_split_prefill_equals_one_shot`과 4-layer 모델 테스트들).

같은 커밋에서 `PleLayer::from_weights`의 패딩 테이블 검사를 지웠다. 마지막 head의 offset과 크기의 합을 `padded_vocab_size = total.div_ceil(divisor) * divisor`와 비교했기 때문에 실패할 수 없었다.

### 3.2 hostile quantization 쌍을 로더로 통과시키기

`QuantizationTable::from_config`(`src/models/qwen4_exp_config.rs:548`)는 이미 기본값과 각 override 등 모든 항목을 `validate_quantization_params`로 검사한다. 이제 테스트가 공용 목록 전체로 이를 증명한다: `hostile_quantization_pairs_are_refused_as_default_and_as_override`는 `HOSTILE_QUANT_PARAMS`의 다섯 쌍을 로더의 config 진입점인 `text_config_from_full`에 한 번은 text config가 상속하는 top-level 기본값으로, 한 번은 모듈 override로 넣는다. 에러마다 모듈 이름이 나오고 올바른 필드(`quantization.bits (` 또는 `quantization.group_size (`)를 지목하는지 확인하고, 정상적인 상속 블록을 positive control로 쓴다. `rejects!` 테이블의 bits 0 한 줄을 대체한다.

### 3.3 테스트 헬퍼와 문서

`mrope_for`는 두 QSA 테스트 자리에 인라인했고, `tiny_model_matches_upstream_logits_for_prefill_and_eight_decode_steps`는 다른 모델 테스트처럼 `tiny_model` / `prompt_ids`를 쓴다. 문서는 이제 attention이 쿼리의 2051번째 토큰까지 dense이고 2052번째부터 sparse라고 적는다(`QsaIndexer::select`는 `key_len / compress_ratio <= block_topk`인 동안 `None`을 반환한다, `src/models/qwen4_exp_qsa.rs`). index cache가 갖는 pooled block key도 적고, 픽스처 생성기는 `src/models/qwen4_exp_parity_tests.rs`를 가리킨다.

## 4. 남은 항목

### 4.1 `Used by:` 목록 (공용 파일 수정 필요)

제안하는 변경(주석만):

- `src/models/gated_delta.rs:54`(`GatedDeltaCache`)와 `:752`(`gated_delta_update`): `Used by: Qwen3Next, Qwen3.5, Qwen4-Exp, KimiLinear`.
- `src/models/switch_layers.rs:1961`(`moe_weighted_sum`): 목록에 `Qwen4-Exp` 추가.
- `src/models/qwen_mrope_state.rs:77`-`184`(`MRopeState` 메서드): `Used by: Qwen2VL, Qwen3VL, Qwen3VLMoe, Qwen3.5, Qwen4-Exp`.

PR은 `qwen3_next::{SwitchGLU, MLP}`, `qwen3_5::mrope_position_source`, `qwen3_vl::{InterleavedMRoPE, rotate_half}`, `merge::merge_llava_any`도 호출한다. 이들은 지금 `Used by:` 목록이 없거나 패밀리를 일반적으로만 적어 둔다.

### 4.2 sequence id 없는 배치 forward

엔진은 행이 여럿인 호출마다 행 id를 넘긴다(`Engine::forward_rows`, `src/lib/mlxcel-core/src/engine/mod.rs:440`, 그리고 `Engine::prefill_cohort`, `src/lib/mlxcel-core/src/engine/prefill.rs:191`). 따라서 CLI와 서버 경로는 영향을 받지 않는다. `examples/profile_batched_decode.rs`처럼 B > 1로 `LanguageModel::forward_batched`를 부르는 호출자는 행이 섞인다. Qwen3.5의 `forward_batched_rows_sequential`도 같은 동작이므로, 해결 방법(id 없는 다중 행 호출 거부, 또는 행마다 임시 state 부여)은 model-owned 패밀리 전체에 대해 한 번 정해야 한다.

### 4.3 row-invariant 양자화 projection

`_row_invariant_quantized_projection`은 `singleton_quantized_linear`(mlx-vlm `mlx_vlm/models/quantized_verifier.py:392`)를 쓴다. prefill의 모든 행에 단일 행 `qmv` 연산을 유지하는 `gather_qmm` 호출이다. 모듈이 양자화된 경우에만 배치 projection과 달라지므로 float32 픽스처로는 볼 수 없다. `docs/supported-models.md`의 체크포인트 설명에 따르면 OptiQ quant는 hyper-connection(`block_inject_weight`)을 8 bit, `in_proj_b` / `in_proj_a`를 4-bit 기본값으로 두고, router와 shared-expert gate는 bf16이다. PR의 "Still open" 목록에 있는 teacher-forced trace가 prefill 위치에서 불일치가 생기는지 보여 줄 것이다. 포팅하려면 공용 단일 행 양자화 projection 헬퍼가 필요하다.

## 5. 의도된 결정의 구현 확인

- QSA graph form: `QsaIndexer::build_mask`는 선택된 블록을 토큰 축에 흩뿌리고, 쿼리의 미완성 tail 블록을 더하며, 완전한 블록이 `block_topk`개 이하인 쿼리는 causal 행으로 돌아간다. 업스트림 `build_mask`와 같다. KV cache와 index cache가 함께 자라므로 mask의 `key_len`이 가져온 key와 맞는다.
- 행 단위 배칭: `supports_batched_prefill`, `supports_padded_prefill`, `supports_paged_decode_backend`는 `false`이고 `Qwen4ExpVLModel`은 셋 모두를 위임한다(padded prefill의 trait 기본값은 `true`이고 wrapper가 이를 물려받으면 안 된다).
- PLE 호스트 해싱: `Qwen4ExpModel::host_tokens`는 PLE가 있는 레이어가 있을 때만, forward마다 한 번 id를 읽어 온다.
- 픽스처 디바이스: `generate_fixtures.py`는 CPU를 기본 디바이스로 정하고(`:610`), float16 q/k normalization 케이스만 `mx.stream(mx.gpu)` 아래에서 돌린다(`:267`). `qk_normalization_matches_upstream_where_qwen35_normalization_does_not`는 그 케이스를 비트 단위로 고정하고 Qwen3.5의 `normalize_qk`가 1e-3보다 크게 빗나가는지 확인한다.
- final norm 없음: `forward_hidden`은 `hyper_connection_mixer.collapse(...)`를 곧바로 LM head로 넘긴다. 업스트림이 `self.hyper_connection_mixer(hidden_states)`를 반환하는 것과 같다.
- `Qwen4ExpMoeBlock`은 업스트림 `Qwen3_5MoeSparseMoeBlock`처럼 `softmax_precise`와 gate 위의 top-k로 `SparseMoeBlock::forward`의 라우팅을 복제하고, routed expert, shared expert, router, shared-expert gate를 각자 선언된 쌍으로 로드한다. `check_expert_packing`은 다른 폭으로 패킹된 plane을 거부한다.

## 6. 변경 요약

| 항목 | 값 |
|------|----|
| 리뷰가 바꾼 파일 | 7 |
| 추가 / 삭제 라인 | +62 / -45 |
| 테스트 | guard 테스트 하나가 `rejects!` 한 줄을 대체, `qwen4_exp` 필터는 여전히 48개 테스트를 선택 |

| Commit | Message |
|--------|---------|
| `fee17250` | refactor(models): gather Qwen4-Exp PLE shard rows in one concatenate |
| `9bf50f43` | test(models): walk hostile quant pairs through the Qwen4-Exp loader |
| `22dbfacd` | test(models): inline the Qwen4-Exp MRoPE helper, reuse tiny_model |
| `7bb7127b` | docs: state the exact Qwen4-Exp QSA dense threshold and index state |

수정은 좁은 `qwen4_exp` 단위 테스트 필터, `cargo clippy --features metal,accelerate --lib --tests -- -D warnings`, `cargo fmt --all -- --check`로 확인했다. workspace 전체 스위트는 orchestrator에게 맡긴다.

## 7. 후속 조치

- 128 GB 머신에서 #2406의 실제 체크포인트 항목: `mlxcel generate` 텍스트, `--image`, 4096 토큰을 넘는 프롬프트, 같은 요청을 `mlxcel-server`로, 그리고 forced-dense 암을 포함한 teacher-forced logit trace. 이 trace가 4.3절도 결론 낸다.
- 4.1절의 `Used by:` 수정.
- id 없는 `forward_batched`에 대한 model-owned 패밀리 차원의 결정(4.2절).

## 관련

- Issue #2406, 업스트림 Blaizzy/mlx-vlm#2032(mlx-vlm `9527f2da` 기준으로 확인).
