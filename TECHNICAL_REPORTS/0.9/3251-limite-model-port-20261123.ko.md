# PR #3251: Paradigma Limite 텍스트 모델 포팅

**날짜**: 2026-11-23
**상태**: 드래프트 PR 열림 (#2360 해결). 실제 체크포인트로 M3 Ultra에서 검증 완료

## 요약

Paradigma Limite(`model_type: limite`, `LimiteForCausalLM`, `paradigma-inc/limite-1b-violetto`)는 48층 dense decoder이며, 지금까지 mlxcel은 `Unsupported model type: limite`로 거부했습니다. 이 아키텍처는 하나하나가 "거의 맞게" 구현하기 쉬운 기능들을 합쳐 놓았습니다. dtype에서 epsilon을 가져오는 weight 없는 RMSNorm, bf16으로 반올림한 스칼라 fold, partial interleaved RoPE, 1025 key sliding window, NoPE global layer, value embedding, XSA, head별 attention gate, layer마다 네 개의 residual lambda, 두 layer의 MUDD mixer, sigmoid softcap이 걸린 tied head입니다. 대부분은 조금 틀리게 구현해도 모든 tensor가 로드되고 그럴듯한 텍스트가 나오기 때문에, 생성 텍스트를 읽는 대신 mlx-vlm의 `mlx_vlm/models/limite`와 teacher-forced logit trace로 비교했습니다.

이 PR은 upstream과 같은 수치로 모델을 추가하고, detection, config 기반 loader, `LoadedModel`, registry, 문서에 연결하며, model-owned 시퀀스별 cache와 네이티브 batched decode, `last_row_logits`, layer별 KV mode 보고를 제공합니다. `docs/architecture.md`의 2560개 위치를 trace한 결과 reference가 결정한 814개 위치 중 mlxcel이 다른 위치는 0개였고, MUDD mix를 빼는 negative arm에서는 reference가 자기 자신과 814개 전부에서 달라졌습니다. 즉 trace가 확인하려는 기능에 실제로 반응합니다.

## 1. 변경 요약

| 영역 | 변경 |
|---|---|
| `limite_layers.rs` | 수치 구현: dtype eps RMSNorm, `fold`, 입력 dtype erf GELU, `mudd_combine`, attention block(value embedding, XSA, gate), SwiGLU MLP, residual lambda, sliding/global `Cache` enum. |
| `limite_rope.rs` | forward마다 한 번 만들어 모든 RoPE layer가 공유하는 cos/sin 테이블을 쓰는 partial interleaved RoPE. |
| `limite.rs` | loader, MUDD history와 tap 연결, decoder stack, f32 sigmoid softcap head, KV mode별 cache 생성. |
| `limite_config.rs` | `ModelArgs`, convention 문자열 검사와 거부 규칙, tap index와 softcap 헬퍼, EOS 목록. |
| `limite_shapes.rs` | 체크포인트의 모든 tensor shape를 로드 시점에 검증. |
| `limite_wrapper.rs` | model-owned 시퀀스별 cache, batched decode, `last_row_logits`, KV mode 보고, sequence id로 해제. |
| `sanitize.rs` | `limite`에 대해 bf16에서 f16으로의 로드 캐스트를 건너뛰고, Ampere 미만에서 f16에 취약한 목록에 추가. |
| 통합 | `detection.rs`, `mod.rs`, `loaded_model.rs`, `registry.rs`, `model_metadata.rs`, `memory_estimate.rs`, tensor parallel fallback 이름, 공용 coverage 테스트 표. |
| 도구와 문서 | `scripts/limite_oracle_trace.py`(`--no-prefix-sharing`, `--drop-mudd`를 지원하는 mlx-vlm reference trace), `docs/supported-models.md` 항목과 KV mode 행. |
| 테스트 | `limite_tests.rs`(단위 테스트 59개)와 detection, model-owned KV mode, mixed-cache batched decode, coverage 테스트 항목. |

파일 22개, +2484 / -6.

## 2. 아키텍처

### 2.1 Layer block

각 layer는 `attn_in = rms_norm(x)`(mixing layer에서는 MUDD mix)를 계산하고 attention을 실행한 뒤, 학습된 스칼라 네 개로 residual을 적용합니다.

```
x   = resid_attn * base + post_attn * attn(attn_in)
out = resid_mlp  * x    + post_mlp  * mlp(rms_norm(x))
```

`base`는 일반 layer에서는 `x`이고 mixing layer에서는 MUDD residual mix입니다. lambda는 0차원 tensor이며 로드 시점에 activation dtype으로 한 번 캐스트합니다. MLP는 SwiGLU입니다.

### 2.2 Attention: sliding과 global

Q, K, V는 학습된 `qkv_scale`, `o_scale`을 로드 시점에 weight에 fold한 projection에서 나옵니다. Q와 K에는 weight 없는 RMSNorm을 적용하고, softmax scale은 `1/sqrt(head_dim)`이 아닌 고정값 0.1입니다.

- **Sliding layer**는 `sliding_window + 1` = 1025개 key를 봅니다. query `q`에서 key `k`는 `k >= q - 1024`(경계 포함)일 때 보입니다. `RotatingKVCache(1025)`를 쓰고 partial RoPE를 적용합니다.
- **Global layer**(`global_layers`에 따라 네 번째 layer마다)는 NoPE이고 증가하는 `KVCache` 위에서 full causal입니다. 유한한 `global_window`는 거부합니다.

prefill은 공용 `prefill_sliding_window_mask`와 `prefill_causal_mask` 헬퍼를 쓰고, mask 없는 다중 토큰 forward는 `maskless_prefill_attention`으로, batched decode는 공용 `attend_batched_rows`로 갑니다.

### 2.3 Value embedding

`ve_layers`에서는 두 번째 embedding 테이블 `model.value_embeds`를 같은 token id로 gather하고, `[B, L, ve_stored_heads, ve_dim]`으로 reshape한 뒤 `head_dim`까지 0으로 채우고 KV head 수로 자르며, `gate_scale * sigmoid(attn_in[..., :channels] @ ve_gate.T)`를 곱해 V에 더합니다. 이 덧셈은 Q/K norm과 cache 쓰기보다 먼저 일어나므로 cache와 attention 모두 조정된 값을 봅니다.

### 2.4 XSA

`xsa_layers`에서는 attention 출력에서 (GQA로 반복한) value 방향 성분을 제거합니다. f32로 `v_hat = v / max(||v||, 1e-4)`를 구하고 `y -= tanh(xsa_alpha) * <y, v_hat> * v_hat`을 계산하며, 빼는 항은 activation dtype으로 되돌려 캐스트합니다.

### 2.5 Attention gate

XSA 다음에 각 head 출력에 `gate_scale * sigmoid(attn_in[..., :channels] @ gate.T)`를 곱한 뒤 `o_proj`로 보냅니다.

### 2.6 MUDD mixer

24번과 47번 layer는 `x` 대신 앞선 layer 출력들을 위치별로 학습된 가중치로 섞은 값을 읽습니다. `hist[0]`은 정규화된 embedding이고 `hist[j]`는 `j - 1`번 layer의 출력입니다. mixing layer마다 가중치는 `gelu(rms_norm(x) @ dense1.T) @ dense2[layer].T + bias[layer]`로 tap마다 하나씩 나오고, mix는 `sum_i w_i * hist[tap_i]`입니다. attention 경로는 `rms_norm(mix)`를 attention에 넣고, `mudd_mlp`이면 두 번째 경로가 residual base를 만듭니다. tap이 읽는 history 슬롯만 보관하며 forward가 끝나면 버립니다.

### 2.7 Head

선택적인 tanh `final_softcap` 다음에 마지막 weight 없는 norm과 tied embedding이 raw logit을 만들고, 이를 f32로 캐스트해 `23 * sigmoid((raw + 5) / 7.5)`로 반환합니다. sampler가 받는 값은 reference와 마찬가지로 이 제한된 값입니다. greedy decoding은 영향을 받지 않고, temperature sampling은 더 평평한 분포를 보는데, 이것이 공개된 모델 그대로의 동작입니다. `last_row_logits`는 샘플링할 행에만 head를 적용합니다.

## 3. Parity 함정과 각각의 처리

| 함정 | 그래도 실행되는 잘못된 구현 | 처리 |
|---|---|---|
| dtype epsilon | 고정 `rms_norm_eps`를 쓰거나, f16 캐스트로 `finfo.eps`가 0.0078125에서 0.00098로 바뀜 | 호출마다 `finfo_eps(dtype)`. `limite`는 f16 로드 캐스트를 건너뛰어 체크포인트의 bf16 activation을 유지. dtype별 eps 단위 테스트. |
| bf16 fold | f32 스칼라를 weight에 곱한 뒤 반올림 | `fold`가 스칼라를 먼저 weight dtype으로 반올림(`scale.astype(W.dtype) * W`). 손으로 계산한 bf16 반올림과 비교하는 단위 테스트. |
| Partial interleaved RoPE | 64쌍 전부 회전, half-split lane 회전, f32 cos/sin | 인접 lane 64쌍 중 32쌍만 회전하고 나머지는 주파수 0. 홀수 lane sine 부호 반전. cos/sin은 f32로 계산해 bf16으로 반올림. 64쌍이 모두 회전하면 실패하는 단위 테스트. |
| 1025 key window | 1024 key window(`k > q - 1024`) 또는 1024 슬롯 rotating cache | window와 cache 모두 `sliding_window + 1`. 단위 테스트: 1100 + 1 토큰 후 layer별 보이는 key가 1025/1025/1025/1101. |
| NoPE global | global layer도 회전(위치 0에서는 항등이라 짧은 테스트로는 못 잡음) | global layer는 `uses_rope`가 false이고 full causal mask. |
| Softmax scale | `1/sqrt(head_dim)` = 0.125 | config에서 읽고 기본값은 0.1. |
| Value embedding 순서 | Q/K norm 뒤나 cache 쓰기 뒤에 더함 | projection 직후, cache 쓰기 전에 V에 더함. |
| MUDD 누적 | tap에 대한 stacked contraction(`einsum`)은 bf16에서 반올림이 다름 | activation dtype에서 엄격하게 왼쪽부터 순서대로 더함. 단위 테스트: bf16에서 256 + 1 + 1은 258이 아니라 256. |
| GELU dtype | f32로 계산하고 되돌리는 `mlxcel_core::gelu` | MLX `nn.gelu`처럼 입력 dtype에서 계산하는 `gelu_erf`. |
| Head | raw logit 또는 bf16 softcap | f32 sigmoid softcap. 다른 종류의 softcap은 거부. |

config에는 convention 문자열(`rope_style`, `sliding_window_convention`, `mudd_hist_convention`, `rms_norm_eps_mode` 등)도 있습니다. 각각을 이 포팅이 구현한 단 하나의 convention과 비교하므로, 나중에 convention이 바뀐 체크포인트는 잘못된 산술로 실행되지 않고 로드 시점에 거부됩니다.

## 4. 통합 지점

- **Detection과 로딩**: `detection.rs`의 `ModelType::Limite`, config 기반 loader 경로, `delegate_language_model!`로 위임되는 `LoadedModel::Limite(LimiteWrapper)`.
- **Registry**: `mlxcel arch`가 KV mode와 함께 이 계열을 보여 줍니다. memory estimate와 tensor parallel fallback은 타입을 `limite`로 매핑합니다(tensor parallel과 pipeline parallel은 연결하지 않음).
- **Cache**: Mellum, Laguna처럼 시퀀스별 model-owned입니다. batched decode는 네이티브이고, scheduler의 KV 한도가 적용되며, 해제는 `release_sequence_state_by_id`를 거칩니다.
- **KV mode**: `--kv-cache-mode`는 global layer에 적용되고 layer별로 보고됩니다. 1025 key sliding layer는 1025가 32의 배수가 아니어서 모든 mode에서 FP16으로 저장하며, 문서 표에 이를 기록했습니다.
- **도구**: `scripts/limite_oracle_trace.py`가 `logit_trace`의 chunk 계획을 mlx-vlm으로 채점해, `scripts/compare_logit_traces.py`가 두 TSV를 비교할 수 있게 합니다.

## 5. 리뷰로 인한 변경

### 5.1 로드 시점 shape 검증 (보안 리뷰)

체크포인트 shape를 그대로 믿어서 생긴 문제가 두 가지였습니다.

- **HIGH**: `model.value_embeds`에 행 수 검사가 없었습니다. 행이 `embed_tokens`보다 적은 테이블이면 정상 token id로도 GPU에서 범위 밖 gather가 일어나며, MLX는 gather의 범위를 검사하지 않습니다.
- **MEDIUM**: shape가 틀린 tensor가 있으면 `Result`가 아닌 FFI를 통해 C++ 예외가 올라와 로드 시점이나 첫 forward에서 프로세스가 종료되었습니다.

이제 `limite_shapes.rs`가 어떤 연산보다 먼저 projection, MLP, gate, 스칼라, `xsa_alpha`, MUDD tensor, 두 embedding 테이블(`[vocab_size, ...]`)을 검사하고, tensor 이름을 담은 로드 오류를 반환합니다.

### 5.2 Config 거부 규칙

`validate`는 추가로 `num_hidden_layers: 0`(`layers[0]`에서 panic), 1도 아니고 KV head 수에도 못 미치는 `ve_stored_heads`(다른 값은 broadcast되지 않음), 그리고 구현 리뷰가 지적한 뒤 추가한 `mudd: false`이면서 `mudd_tap_idx`가 설정된 경우를 거부합니다. upstream은 flag와 상관없이 `mudd_tap_idx`로 mix를 적용하므로 mlxcel이 이를 건너뛰면 조용히 어긋납니다. 추측하는 것보다 그 조합을 거부하는 편이 쌉니다. 공개된 체크포인트는 `mudd: true`라서 영향이 없습니다. 이 규칙들은 기존의 `rope_per_layer: true`, `mlp_type: relu2`, sigmoid가 아닌 softcap, 유한한 `global_window`, 양자화 export 거부에 더해집니다.

### 5.3 RoPE 테이블 끌어올리기 (성능 리뷰)

cos/sin 테이블을 layer마다, 그리고 Q와 K 각각에 대해 다시 만들고 있었습니다. 이제 `limite_rope.rs`가 위치 배열(단일 시퀀스는 `[1, L]`, 행별 offset이 있는 batched decode는 `[B, L]`)에서 forward마다 한 번 만들고 모든 RoPE layer가 공유합니다. residual lambda도 호출마다가 아니라 로드 시점에 캐스트합니다. greedy 출력은 바이트 단위로 같고, test-fast 빌드에서 decode가 약 99.4에서 111.3 tok/s로 올랐습니다.

## 6. 검증

### 6.1 단위 테스트

`--lib limite`: 59개 통과. dtype eps, bf16 fold, partial RoPE, MUDD 순서, 1025 key cache, config와 shape 거부, 공용 model-owned 계약, int8 KV, last-row logit을 다룹니다. `coverage_tests`(18개), `models::registry`, `models::detection_tests`, `models::mixed_cache_batched_decode_tests`, `models::model_owned_kv_mode_tests`, `models::last_row_logits` selector가 통과했고, clippy `-D warnings`, fmt, 계약 스크립트도 통과했습니다.

### 6.2 Logit trace

M3 Ultra, bf16 체크포인트, 말뭉치 `docs/architecture.md`. reference의 top-two 차이가 2 이상이면 결정된 위치로 봅니다. 계획 `64 40 8 1536`은 최대 1536 토큰 context 뒤의 64 토큰 chunk입니다. 17번부터 39번 chunk는 context가 1025 토큰을 넘으므로 sliding window가 실제로 작동합니다.

| 비교 | 위치 수 | 결정된 위치 불일치 | ref top-1 최대 차이 |
|---|---|---|---|
| ref(chunk마다 forward 한 번) vs mlxcel, `64 40 8 1536` | 2560 | 0 / 814 | 0.63 |
| ref(prefix 공유 forward) vs mlxcel, 같은 계획 | 2560 | 1 / 815 | 2.74 |
| ref vs ref, 공유 vs chunk별 | 2560 | 1 / 814 | 3.79 |
| ref vs mlxcel 폭 1(decode), `1 1300 8 1536` | 1300 | 1 / 402 | 3.54 |
| mlxcel `MLXCEL_TRACE_BATCH_ROWS=1` vs `=4`, `64 36 8 1536` | 2304 | 0 / 666 | 0.58 |
| ref vs ref `--drop-mudd`(negative arm) | 2560 | 814 / 814 | |

세 행에 나오는 불일치 하나는 항상 말뭉치 위치 1250이며, 그 위치에서는 reference의 두 forward 형태끼리도 이미 다릅니다(3672 대 87, 차이 4.05와 3.49). mlxcel은 prefill과 decode 두 형태 모두에서 chunk별 reference와 일치합니다. perplexity는 reference 797.3, mlxcel 796.2이며, head가 `(0, 23)`으로 제한되어 있어서 절대값이 높습니다.

negative arm이 중요합니다. 24번과 47번 layer의 MUDD mix를 평범한 `rms_norm(x)`로 바꾸면 결정된 위치가 전부 뒤집힙니다. 그 기능을 빼고도 통과하는 trace라면 그 기능에 대해 아무것도 증명하지 못합니다.

### 6.3 Generate와 서버

- `mlxcel generate -m models/limite-1b-violetto -p "Hello" -n 50 --temp 0`은 약 90 tok/s로 실행됩니다. 이 모델은 고정 system prompt를 가진 수학 추론 모델이라 출력은 reasoning 채널입니다("We need to parse the user's request..."). `-p "What is 17 * 23?" -n 600`에서는 340 + 51 = 391을 도출합니다.
- `mlxcel-server`가 1821 토큰 chat 프롬프트와 동시에 들어온 45 토큰 프롬프트를 처리했습니다. 둘 다 `finish_reason: length`(120, 200 토큰)로 끝났고, 짧은 쪽은 42로 추론했습니다.

## 7. 남은 항목

- MUDD history 슬롯을 마지막 tap 이후 해제하지 않고 stack 끝까지 보관합니다(LOW, 미수정).
- `eos_token_id` 값을 범위 검사 없이 i64에서 i32로 좁힙니다(LOW, 미수정).
- 1025 key sliding layer는 양자화 KV 저장을 쓸 수 없어서 `--kv-cache-mode`의 이득은 global layer에서만 납니다.
- 이 계열은 tensor parallel과 pipeline parallel에 연결되어 있지 않습니다.
- 5.3의 성능 수치는 test-fast 빌드에서 나왔습니다. release 빌드 generate 실행은 다른 프롬프트와 길이로 약 90 tok/s였으므로 두 수치를 직접 비교할 수 없습니다.

## 8. 학습 포인트

- 위치 0이나 짧은 입력에서 항등이 되는 기능(NoPE global, 1024 대 1025 window)은 차이가 나는 경우까지 도달하는 테스트가 필요합니다. 두 offset을 쓰거나 window를 넘는 시퀀스를 써야 합니다.
- 모델 산술이 dtype에 의존하면(여기서는 norm epsilon) 범용 로드 시점 f16 캐스트는 정밀도 변경이 아니라 의미 변경입니다.
- bf16에서는 누적 순서도 모델의 일부입니다. stacked contraction과 왼쪽부터 더하는 합은 정확한 산술에서는 둘 다 맞지만 다른 토큰을 냅니다.
- logit trace에는 negative arm이 필요합니다. 기능 하나를 빼서 일치도가 무너지는 것을 보아야 trace가 그 기능을 볼 수 있다는 것이 입증됩니다.
- reference 자체의 두 forward 형태도 거의 동점인 위치에서는 다를 수 있습니다. 불일치 하나를 버그라고 부르기 전에 둘 다와 비교해야 합니다.
