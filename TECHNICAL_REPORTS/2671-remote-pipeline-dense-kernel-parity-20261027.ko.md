# PR #2671: remote pipeline stage를 dense kernel 경로에 맞추기

**날짜**: 2026-10-27
**상태**: 완료 (merge 전에 PR branch에서 작성한 보고서)
**언어**: Rust, YAML, Shell
**위험도**: 중간 (CUDA에서 float16 activation을 쓰는 모든 remote pipeline stage 경계와 모든 Gemma 3, Gemma 4 prefill stage의 수치 경로가 바뀜. dense single-process 경로는 그대로. 인증된 peer가 일으킬 수 있는 heap over-read에 대한 HIGH 수정 포함)

## 요약

GB10에서 2-stage remote pipeline이 llama, gemma3, gemma4에 대해 single-process 모델과 다른 logit을 냈습니다 (이슈 #2654). 코드를 고치기 전에 split 경계의 hidden state를 비교한 결과 서로 무관한 결함 두 개가 드러났습니다. 첫째, wire decoder가 float16 activation을 `from_bytes_f16`으로 복원했는데, 이 함수는 checkpoint loader라서 Apple을 제외한 모든 backend에서 float16을 float32로 넓힙니다. 그래서 dense 모델은 float16으로 도는데 하위 stage와 LM head는 float32로 돌았습니다. 둘째, `Gemma3StageModel`과 `Gemma4StageModel`은 모든 prefill에서 명시적 causal mask와 sliding mask를 만들었지만, dense forward는 window 안에서 mask를 넘기지 않고 MLX의 maskless causal SDPA를 탑니다. 이것은 reduction 순서가 다른 별개의 kernel입니다. 이 PR은 wire tensor를 보낸 dtype 그대로 복원하고, Gemma prefill mask 결정을 dense 모델, stage 모델, `Gemma3Backbone`이 함께 쓰는 helper로 옮기며, wire decoding을 길이 검사가 있는 `kv_cache_serde::reconstruct_mlx_array`로 보내 짧은 payload에서 생기던 기존 heap over-read를 막습니다. 세 family 모두 in-process와 wire 경유 양쪽에서 dense와 byte 단위로 같은 logit을 내며, tolerance는 바꾸지 않았습니다.

## 1. 문제 정의

### 1.1 배경

이슈 #2654는 GB10(spark-102, CUDA)에서 결정적으로 실패하는 `#[ignore]` real-model 테스트 네 개를 기록했습니다.

- `tests/pipeline_remote_real_models.rs`: llama, gemma3, gemma4의 `*_real_model_parity_and_cleanup` 테스트. prompt `[2, 3]`(llama는 `[128000, 9906]`), `num_layers / 2`에서 split(gemma4는 13)한 상태에서 `allclose(&full_prefill, &remote_prefill, 1e-4, 1e-4)`가 실패.
- `tests/pipeline_server_remote_real_models.rs`: `pipeline_server_remote_coordinator_gemma4_matches_dense_baseline`이 `assert_eq!(dense.text, remote.text)`에서 실패.

같은 실행에서 qwen3와 qwen3.5 variant는 통과했습니다. 실패는 #2650(cluster handshake) 전후의 main에서 동일했고 `MLXCEL_SDPA_DETERMINISTIC=1`에서도 그대로였으므로, transport 변경이나 실행 간 noise가 원인이 아니었습니다. CI는 이 테스트들을 돌리지 않았습니다.

### 1.2 이슈의 추정이 틀린 지점과 그 의미

이슈는 코드를 읽고 "wire는 source dtype과 raw byte를 유지하며 암묵적 cast가 없다"고 판단했고, llama는 실패하고 qwen3는 같은 generic stage loop로 통과하므로 llama 고유의 stage 구성을 의심했습니다. 두 추정 모두 배울 점이 있는 방식으로 틀렸습니다. wire 위의 byte는 보존되었지만, 그 byte를 다시 array로 만드는 함수가 dtype을 바꿨습니다. qwen3가 통과한 것은 activation이 bfloat16이고 `from_bytes_f16`이 CUDA에서 bfloat16은 그대로 두기 때문입니다 (`MLX_BF16_NATIVE=0`이 아닌 한). llama stage 구성은 원인이 아니었습니다.

### 1.3 근본 원인의 근거

진단은 dense 모델의 layer `split - 1` 이후 hidden state와 stage 0 출력을 비교하고, 이어서 wire 없이 in-process stage logit을 dense logit과 비교한 다음, wire 왕복을 추가하는 순서로 진행했습니다. 아래 수치는 PR 본문에 있는 GB10 측정값입니다.

| 모델 (CUDA activation dtype) | split 지점 stage 0 출력 vs dense | in-process stage logit vs dense | wire 경유 vs dense | 결함 |
|---|---|---|---|---|
| llama-3.2-1b-4bit (float16) | byte 동일 | byte 동일 | max abs diff 1.5e-2, dtype float32 | wire widening |
| gemma-3-1b (float16) | 보고 없음 | max abs diff 9.4e-2 | 보고 없음 | mask와 wire widening |
| gemma-4-e2b, split 13 (bfloat16) | 보고 없음 | max abs diff 3.4e-1 | 보고 없음 | mask |
| qwen3 (bfloat16) | 보고 없음 | 보고 없음 | 통과 | 없음 |

### 1.4 방치했을 때의 위험

| 위험 | 영향 | 가능성 |
|------|------|--------|
| CUDA에서 float16 checkpoint의 remote pipeline serving이 하위 stage를 조용히 float32로 실행 (메모리 트래픽 증가, 출력이 dense와 다름) | 중간 | float16 checkpoint라면 높음 |
| Gemma 3, Gemma 4 pipeline 출력이 dense와 달라지고 생성 텍스트까지 달라짐 (gemma4 server 테스트) | 높음 | 높음 |
| cluster handshake를 통과한 peer가 큰 shape와 짧은 payload를 보내 수신 stage에서 heap buffer 너머를 읽음 | 높음 | 낮음 (인증된 peer 필요) |

## 2. 기술적 선택과 그 이유

### 2.1 wire tensor를 복원 지점에서 보낸 dtype 그대로 만들기

이전의 `deserialize_wire_tensor`는 half type을 따로 처리했습니다.

```rust
match tensor.dtype {
    TensorDtype::Float16 => Ok(mlxcel_core::from_bytes_f16(&tensor.data, &shape, false)),
    TensorDtype::BFloat16 => Ok(mlxcel_core::from_bytes_f16(&tensor.data, &shape, true)),
    other => {
        let dtype = tensor_dtype_to_mlx(other)?;
        Ok(mlxcel_core::from_bytes(&tensor.data, &shape, dtype))
    }
}
```

이제는 모든 dtype이 같은 경로를 탑니다.

```rust
let dtype = tensor_dtype_to_mlx(tensor.dtype)?;
let raw = RawTensorData { data: tensor.data, shape, dtype };
reconstruct_mlx_array(&raw)
```

MLX dtype id 9(float16)와 12(bfloat16)로 호출한 `from_bytes`는 모든 backend에서 2바이트 half를 그대로 재해석하므로, stage N을 떠난 array가 그대로 stage N+1에 들어갑니다. implementation review가 경계 사례를 확인했습니다. Metal의 float16과 native bfloat16은 원래 손실이 없었고, `MLX_BF16_NATIVE=0`에서는 weight 자체가 float32로 load되므로 activation도 dense 모델과 일관됩니다.

| 대안 | 장점 | 단점 |
|---|---|---|
| parity tolerance 완화 (이슈가 근거를 갖추면 허용) | 코드 변경 없음 | drift가 본질적이지 않았음. dtype 변화와 kernel 선택에서 왔고 둘 다 고칠 수 있음. 느슨한 tolerance는 다음 실제 divergence도 숨김 |
| `from_bytes_f16` 유지 후 보낸 dtype으로 다시 cast | diff가 작음 | 의미 없는 float32 왕복이 생기고, backend 의존 동작을 가진 checkpoint loader가 activation 경로에 계속 남음 |
| **선택: 모든 dtype에 대해 dtype을 보존하는 단일 경로** | 모든 backend에서 byte 동일한 경계, 추론할 코드 경로가 하나, 기존 길이 검증을 그대로 활용 | `from_bytes`가 모든 wire dtype을 지원해야 함. review는 int16을 `uint8_t` default branch로 decode한다고 지적함 (activation은 int16을 쓰지 않음) |

### 2.2 prefill mask 결정을 재구현하지 않고 dense와 공유하기

stage 모델은 별도의 forward로 작성되었고, dense 코드에 maskless causal 경로, native sliding-window route(#2621), long-prefill cutoff(#672)가 추가되는 동안 따라가지 못했습니다. text-only prefill에서 각 경로가 만든 mask는 다음과 같습니다.

| prefill 경우 | dense 모델 (변경 없음) | 변경 전 stage | 변경 후 stage |
|---|---|---|---|
| global (full-attention) layer | `None`, maskless causal SDPA | 명시적 `create_causal_mask(seq_len, global_live_len)` | `None` |
| sliding layer, `live_len + l <= window` | `None` | 명시적 band (두 family 모두) | `None` |
| window 초과 sliding layer, native route 계획됨 | `None` (#2621) | Gemma 3 `None`, Gemma 4 명시적 band | `None` |
| window 초과 sliding layer, native route 계획 안 됨 | dense band | dense band | dense band |
| Gemma 4, `l > 4096` | 둘 다 `None` (#672) | 둘 다 명시적 | 둘 다 `None` |

명시적 bottom-right causal mask와 maskless causal flag는 같은 attention 패턴을 뜻하지만, CUDA에서 서로 다른 SDPA kernel을 고르고 두 kernel의 결과는 bit 단위로 일치하지 않습니다. parity 테스트의 tolerance는 1e-4였고, real checkpoint에서는 이 kernel 차이가 그보다 컸습니다.

수정은 dense의 결정을 유일한 결정으로 만듭니다.

- `gemma3::text_prefill_sliding_mask(sliding_attn, h, caches, seq_len, sliding_live_len, window)`는 sliding mask나 `None`을 돌려줍니다. 이 경로에서 global layer는 mask를 받지 않습니다. 호출자는 `Gemma3Model::forward_with_caches_and_embeddings`, `Gemma3StageModel::execute_hidden`, `Gemma3Backbone::forward_embeds`입니다. `sliding_prefill_plans_native`는 이 helper 안으로 private 처리되고, stage의 `first_global_cache_index`는 삭제됩니다.
- `gemma4::text_prefill_masks(sliding_attn, h, caches, l, window)`는 `(global, sliding)` 쌍을 돌려주며, dense forward에 별도 branch로 있던 `l > DENSE_PREFILL_MASK_MAX_TOKENS` 조기 반환도 포함합니다. `Gemma4TextModel::sliding_prefill_plans_native`는 `Option<&Attention>`을 받는 free function이 되어, `(global_idx, layer)` 쌍을 가진 stage 모델이 자신의 첫 sliding layer를 넘길 수 있습니다.

review는 dense branch가 main과 boolean 동치임을 확인했습니다. Gemma 4 long-prefill branch는 여전히 padding과 vision overlay가 없는 `l > 4096`에서만 도달하고, external-mask, padded, batched-decode, verify branch는 건드리지 않았습니다.

트레이드오프: stage 모델은 이제 dense의 text-only 전제(caller mask 없음, resident padding 없음, vision overlay 없음, per-row valid end 없음)를 물려받습니다. stage는 이전에도 이런 경우를 처리하지 않았고 무조건 단순 causal mask를 만들었으므로 회귀는 아니지만, 앞으로 multimodal이나 padding이 있는 stage 경로가 생기면 이 helper를 그대로 재사용하면 안 됩니다.

`Gemma3StageModel`은 dense loop와 같게 각 layer 뒤에 `pipeline_hint(&hidden, layer_idx, n)`도 호출합니다. `MLXCEL_PIPELINE_GRANULARITY`가 설정되지 않으면 아무 일도 하지 않습니다.

### 2.3 FFI copy 전에 wire payload 길이 검증 (HIGH)

`mlxcel_core::from_bytes`는 raw pointer에서 `product(shape) * element_size` 바이트를 복사하며 payload 길이를 보지 않습니다. wire header의 shape와 payload는 모두 peer가 보내므로, 큰 shape에 짧은 payload를 붙이면 heap buffer 너머를 읽었습니다. 이 문제는 PR 이전부터 있었지만(기존 `from_bytes_f16` branch도 검사하지 않음), 이 PR이 모든 dtype을 `from_bytes`로 보내므로 노출 범위를 넓히지 않도록 security review가 같은 PR 안에서 고쳤습니다.

수정은 `kv_cache_serde::reconstruct_mlx_array`를 재사용합니다. 그 안의 `validate_raw_tensor`(`kv_cache_serde/types.rs`, paged KV handoff용으로 #186에서 추가되었고 이 PR에서는 바뀌지 않음)는 음수 dimension을 거부하고, element 수와 byte 수를 checked multiplication으로 계산하며(빈 shape는 원소 1개짜리 scalar), payload 길이가 정확히 일치해야 통과시킵니다. `deserialize_wire_tensor`는 원래부터 각 `u64` dimension을 `i32::try_from`으로 변환했습니다. validator의 dtype 표는 `tensor_dtype_to_mlx`가 만드는 모든 dtype을 포함하므로, 이전에 받아들이던 wire dtype이 새로 거부되는 일은 없습니다. 노출은 cluster handshake(#2650, #2674)를 통과한 peer로 한정되었습니다.

### 2.4 GB10 CI coverage 결정

이슈는 remote parity를 CI에서 돌릴지에 대한 결정을 기록하라고 요구했습니다. real-model 테스트는 추가하지 않았습니다. PP workflow의 two-host job은 qwen3 fixture만 가진 `ubuntu-latest`(CPU)에서 돌고, PP suite를 돌리거나 llama, gemma checkpoint를 가진 GB10 job은 없습니다. 대신 다음을 했습니다.

- `scripts/ci/run-pp-two-host-logical.sh`의 lib-test filter에 `models::gemma3::gemma3_stage_tests::`와 `models::gemma4::gemma4_stage_tests::`를 추가했습니다. 기존 `distributed::pipeline::` filter가 이미 `wire_tensor::tests`를 포함합니다.
- `.github/workflows/pipeline-parallel-ci.yml`의 `pull_request` path filter에 `src/models/gemma3.rs`, `gemma3_backbone.rs`, `gemma3_stage_tests.rs`, `gemma4.rs`, `gemma4_stage_tests.rs`를 추가했습니다. implementation review가 지적했듯 main CI는 Linux에서 mlxcel lib 테스트를 돌리지 않으므로, 이것이 없으면 PP 경로를 건드리지 않는 dense-mask 변경은 새 parity suite를 한 번도 돌리지 않습니다.

## 3. 변경 요약

| 항목 | 값 |
|------|-----|
| 변경 파일 | 9 |
| 추가 라인 | +1,035 |
| 삭제 라인 | -195 |
| 추가 테스트 | 14 (wire 8, Gemma 3 stage 3, Gemma 4 stage 3) |

- **Wire**: `src/distributed/pipeline/wire_tensor.rs`(dtype 보존, 길이 검사 decode)와 새 `wire_tensor_tests.rs`.
- **Gemma 3**: `src/models/gemma3.rs`(`text_prefill_sliding_mask`, dense와 stage 호출부, stage loop의 `pipeline_hint`), `src/models/gemma3_backbone.rs`(helper 사용), 새 `gemma3_stage_tests.rs`.
- **Gemma 4**: `src/models/gemma4.rs`(`first_sliding_attention`, free function `sliding_prefill_plans_native`, `text_prefill_masks`, dense와 stage 호출부), 새 `gemma4_stage_tests.rs`.
- **CI**: PP workflow path filter와 two-host logical script.

새 테스트:

- `wire_tensor::tests`: `float16_activation_round_trips_in_float16`, `bfloat16_activation_round_trips_in_bfloat16`, `float32_activation_round_trips_in_float32`, `token_ids_round_trip_in_int32`, `hand_built_bfloat16_buffer_round_trips`, `float16_short_payload_is_rejected`, `float32_short_payload_is_rejected`, `scalar_shape_with_empty_payload_is_rejected`. 왕복 fixture는 half precision으로 정확히 표현되지 않는 값을 써서, 어디서든 넓히거나 좁히는 cast가 있으면 byte가 바뀝니다.
- `gemma3_stage_tests`와 `gemma4_stage_tests`에 각각 `two_stage_split_matches_dense_prefill_and_decode_bitwise`(2, 8 token prompt 후 decode), `two_stage_split_matches_dense_over_window_prefill_bitwise`(`WINDOW + 8` token), `two_stage_split_matches_dense_continuation_prefill_bitwise`(비어 있지 않은 cache에 두 번째 chunk, window 안의 4+3과 window를 넘는 12+8, 이후 decode). synthetic 모델은 4 layer, window 16, head dim 256(fused SDPA kernel이 지원하는 크기라 masked 경로와 maskless 경로가 서로 다른 kernel)이고, stage 1이 global layer에서 시작하도록 split합니다. Gemma 3는 float16, Gemma 4는 bfloat16으로, real checkpoint가 CUDA에서 쓰는 dtype과 같습니다. 모든 비교는 `allclose`가 아니라 bit 단위입니다.

| Hash (rebase 후) | 원본 | 메시지 |
|---|---|---|
| `6336cd04` | `2513d03f` | fix(distributed): keep remote pipeline stages on the dense kernel path |
| `4a59dd91` | `25e93ded` | fix(distributed): validate wire tensor length before from_bytes |
| `06cd3d3f` | `1205a313` | test: cover continuation prefill and run Gemma stage suites in PP CI |

세 번째 commit은 implementation review의 MEDIUM 항목 두 개(continuation prefill 사례 없음, PP path filter에 Gemma 파일 누락)와 LOW 항목(Gemma 3 테스트 header가 float16 fixture를 bf16이라고 적음)을 반영합니다.

## 4. 검증

아래 결과는 모두 PR 본문과 review comment에서 가져온 것으로, rebase 전 head에서 GB10(spark-102)으로 기록되었습니다. 이 보고서를 쓰면서 다시 실행하지 않았습니다.

- **먼저 실패 확인.** float16 wire 왕복 테스트와 두 2-token split 테스트는 main의 소스에서 실패합니다 (logit diff 8.8e-3과 2.3e-2, wire 이후 dtype float32). 나머지 새 테스트의 main 대비 결과는 PR에 보고되지 않았습니다.
- **Real model.** 최종 head에서 `MLXCEL_REQUIRE_MODELS=1 MLXCEL_SDPA_DETERMINISTIC=1 --test-threads=1`로 `pipeline_remote_real_models` 6/6(llama, gemma3, gemma4, qwen3, qwen3.5 parity, llama drain), `pipeline_server_remote_real_models` 5/5, `pipeline_multi_stage_two_host_logical_smoke` 통과. `pipeline_stage_executor_real_models`는 첫 commit에서 10/10이었고, 그 gemma variant와 llama worker loop도 main에서 실패하던 것이므로 이 PR은 이슈가 나열한 네 테스트보다 더 많은 것을 고칩니다.
- **Lib suite와 lint.** `distributed::pipeline::`, `models::gemma3`, `models::gemma4`, `models::gemma3_backbone`, `kv_cache_serde`가 `MLXCEL_SDPA_DETERMINISTIC=1` 유무와 관계없이 통과. security 수정 이후 `wire_tensor::`(8)와 `kv_cache_serde::`(41) 통과. `-D warnings` clippy와 fmt 깨끗함.
- **Rebase.** orchestrator가 main `359a335f` 위로 rebase했고, `git range-diff`에서 세 commit 모두 patch 동일입니다. upstream 범위에는 #2674가 있는데, 이 PR은 cluster handshake를 wire version 2로 올리고 real-model remote 테스트가 쓰는 transport인 `tcp_transport.rs`를 바꿨습니다. 위의 GB10 real-model 실행은 이 rebase 이전이며, 이 보고서를 쓰는 시점에 rebase된 head에서 pre-merge check가 진행 중이었습니다.
- **검증하지 않은 것.** 새 lib 테스트는 아직 CPU build에서 돈 적이 없고, 이 PR의 PP CI 실행이 처음입니다. Metal은 확인하지 않았습니다. Apple에서는 float16 wire 경로가 원래 native였으므로 mask 변경만 해당됩니다. throughput과 메모리는 측정하지 않았습니다. 다른 stage family(gpt-oss, glm4, glm5)는 host에 checkpoint가 없어 실행하지 않았습니다.

## 5. 남은 위험과 후속 조치

- **real-model parity의 GB10 CI 없음.** weight 없는 split 테스트가 여기서 찾은 두 메커니즘은 지키지만, 특정 real checkpoint에만 나타나는 새 divergence(quantization layout, 다른 head dim)는 여전히 수동 GB10 실행에서만 드러납니다.
- **기존 wire hardening 공백 (인증된 peer 한정).** security review가 이 diff 바깥에서 찾은 것: (MEDIUM) `tensor_serialize::deserialize_tensor`가 `dequantize_int8/int4` 출력 크기를 `header.num_elements()`로 정하는데, 이 값은 overflow 시 `u64::MAX`로 포화되고 이어서 검사 없이 2를 곱하므로 조작된 header가 panic이나 allocation abort를 일으킴. (MEDIUM) `tensor_compress::decompress`가 shape와 연결된 상한 없이 LZ4 size prefix(최대 4 GiB)를 믿음. (LOW) `from_bytes`가 int16을 `uint8_t` default branch로 decode함. 이 보고서 작성 시점에 이를 추적하는 open issue는 없었습니다.
- **stage 안의 trim된 cache.** 삭제된 stage 코드는 `--max-kv-size` trim 이후 `live_len`으로 global prefill mask 크기를 정했습니다 (#430). 이제 global stage layer는 dense 모델과 똑같이 maskless 경로를 타므로 dense가 맞으면 stage도 맞지만, trim된 cache를 다루는 stage 테스트는 없습니다.
- **sliding layer가 없는 Gemma 4 stage.** `text_prefill_masks`는 `window < l <= 4096`이고 native route가 계획되지 않으면 sliding band를 만듭니다. full-attention layer만 가진 stage(실제로는 1-layer stage)에서는 이 band가 할당만 되고 쓰이지 않습니다. 비용은 prefill당 mask 하나이고 출력에는 영향이 없습니다.
- **helper는 text-only prefill을 전제함.** 2.2 참고. padding, vision overlay, per-row valid end가 있는 stage 경로가 생기면 dense forward처럼 별도의 mask branch가 필요합니다.

## 6. 학습 포인트

- **parity 실패는 가설보다 먼저 경계별로 이분 탐색한다.** 세 번의 byte 비교(stage 0 출력 vs dense hidden state, in-process stage vs dense, wire vs in-process)가 이슈의 코드 읽기로는 엉뚱한 곳에 돌려졌던 무관한 결함 두 개를 분리했습니다. 각 경계의 동등성 검사는 싸고 해석의 여지가 없습니다.
- **"byte가 보존된다"는 "tensor가 보존된다"와 다르다.** serializer가 무손실이어도 deserializer의 생성자가 backend 정책을 적용할 수 있습니다. `from_bytes_f16` 같은 loader helper는 checkpoint에 맞는 결정(backend가 float32를 선호하면 float16을 넓힘)을 담고 있고, 정확히 왕복해야 하는 activation에는 맞지 않습니다.
- **같은 의미의 mask가 같은 kernel은 아니다.** 명시적 causal mask와 causal flag는 수학적으로 같지만, reduction 순서가 다른 SDPA 구현을 고릅니다. 두 코드 경로가 bit 단위로 일치해야 한다면 kernel 선택 결정도 같아야 하고, 가장 쉬운 방법은 같은 결정 함수를 호출하는 것입니다.
- **복제된 forward는 어긋난다.** stage 모델은 dense forward를 다시 구현했고, 이후의 dense 변경 세 개(#2621, #672, maskless causal 경로)를 놓쳤습니다. 결정을 모든 호출자가 쓰는 helper로 뽑아내야 다음 변경이 같은 틈을 다시 열지 않습니다.
- **synthetic 모델에서 정확한 동등성으로 parity를 테스트한다.** 실제 dtype과 kernel 선택에 영향을 주는 head dim을 가진 작은 random 모델은 weight를 내려받지 않고도 kernel 분기를 재현하므로 CPU CI에서 돌릴 수 있고 실패를 분명히 드러냅니다. `allclose`였다면 이번에 고친 종류의 버그를 정확히 숨겼을 것입니다.
- **신뢰할 수 없는 크기는 FFI를 넘기 전에 검증한다.** shape를 믿는 C++ copy는 Rust 코드베이스 안에서도 메모리 안전 경계입니다. 기존 validator(`validate_raw_tensor`)를 재사용해서 수정이 호출 지점 하나로 끝났습니다.
