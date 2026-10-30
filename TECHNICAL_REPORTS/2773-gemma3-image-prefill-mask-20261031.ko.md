# PR #2773: Gemma 3 이미지 prefill에 transformers mask 적용

**날짜**: 2026-10-31 (merge 전에 작성, 작성 시점의 PR은 open draft)
**상태**: 리뷰 중
**언어**: Rust, Python
**위험도**: 중간 (모든 Gemma 3 이미지 요청이 다른 attention mask로 prefill되므로 Gemma 3 이미지 출력이 바뀜. PaliGemma와 text-only Gemma 3 경로는 그대로)

## 요약

#2750이 Gemma 3 이미지 prompt를 transformers와 같게 만든 뒤에도, 이미지 logit trace는 float32 reference 대비 decided position 43개 중 5개에서 달랐다. Issue #2760은 그 차이가 양쪽 arm의 attention mask에 있다는 것을 밝혔다. mlxcel은 Gemma 3 이미지 prompt를 모든 layer에서 완전 양방향 mask로 prefill했고 reference는 causal로 실행했지만, transformers는 causal attention에 image block 내부의 양방향 attention을 더하고, sliding layer에서는 이를 sliding window와 교차시킨다. 이 PR은 `vision::merge`에서 transformers mask를 만들고, Gemma 3의 sliding layer에서 그 mask에 window band를 적용하며, PaliGemma에는 별도 merge strategy를 주어 prefix-LM mask를 유지하게 하고, trace reference arm에도 같은 mask를 넘긴다. 양쪽 arm에 transformers mask를 쓰면 decided 불일치는 48개 중 7개에서 48개 중 0개로 줄고, PaliGemma trace는 byte 단위로 같으며, GB10 CLI와 server smoke는 이제 테스트 이미지를 올바르게 설명한다.

## 1. 문제 정의

### 1.1 mlxcel이 하던 일

`prepare_inputs_for_multimodal`(`src/vision/merge.rs`)은 4D mask를 `attention_mask[k] * attention_mask[q]`로 만들었다. 모든 요청이 갖는 all-ones padding mask에서는 이것이 전부 0인 additive mask가 된다. 즉 모든 prompt 위치가 다른 모든 위치에 attend하므로, text token은 뒤쪽 prompt token을 읽고 image token은 뒤따르는 질문을 읽는다. 그리고 `Gemma3Model::hidden_with_caches_and_embeddings`는 이 mask 하나를 모든 layer에 넘겼으므로, sliding-window layer(4B checkpoint에서 window 1024)도 text 경로가 `text_prefill_sliding_mask`로 만드는 band를 건너뛰었다. `MergeStrategy::Gemma3`의 두 사용자, Gemma 3(`load_gemma3_vlm`)와 PaliGemma(`load_paligemma_vlm`)가 모두 이 경로를 탔다. 이 mask는 이미지를 prefill하는 모든 경로, 즉 scheduler의 embedding prefill, `mlxcel generate`, logit trace에서 모델에 도달했다.

### 1.2 reference들이 하는 일

| Runtime | Full-attention layer | Sliding layer |
|---|---|---|
| transformers (`create_masks_for_vision_model`) | `causal OR same_image_block` | `sliding_window AND (causal OR same_image_block)` |
| mlx-vlm 생성 경로 | 전부 1 (양방향) | 같은 mask |
| 이 PR 이전의 trace reference arm (`mask=None`) | causal | causal |
| 이 PR 이전의 mlxcel | 양방향 | 양방향, band 없음 |

image block은 image soft token(`image_token_id`)의 최대 연속 구간이다. `<start_of_image>`와 `<end_of_image>`는 text로 취급된다. PaliGemma는 설계상 다르다. transformers는 PaliGemma를 prefix LM으로 실행하므로 prefill에서 prompt 전체가 양방향이고, mlxcel의 기존 mask는 PaliGemma에는 이미 맞았다.

### 1.3 왜 문제였나

mask는 모델 학습 조건의 일부다. prompt의 나머지를 볼 수 있는 text token은 checkpoint가 학습한 것과 다른 hidden state를 계산하고, 그 영향은 생성 결과에 드러났다. 수정 전 `mlxcel generate --image`는 `test_image_shapes.png`(빨간 사각형, 파란 원, 초록 삼각형)를 "a white square-like shape-dividing it into four equal parts"라고 설명했다. trace 비교로는 이것을 잡을 수 없었다. reference arm이 causal이었고, 이는 다른 방향으로 틀린 mask였기 때문이다. 틀린 mask 두 개가 일치한다고 해서 정확하다는 뜻도 아니었다.

## 2. 기술적 선택과 그 이유

### 2.1 호출자가 mask를 지정하고, PaliGemma는 자체 strategy를 가짐

`prepare_inputs_for_multimodal`은 이제 `MultimodalMask`(`Gemma3ImageBlocks` 또는 `PrefixLm`)를 받고, `MergeStrategy`에는 `load_paligemma_vlm`이 쓰는 `PaliGemma` variant가 추가되었다. `VisionModule::prepare_inputs`는 두 개의 단순한 match arm으로 strategy마다 mask를 지정한다.

token id에서 mask를 추론하는 방식(예: "PaliGemma prompt에는 `<start_of_image>`가 없다")은 정확성을 token layout의 우연에 묶어 두고, 이후 checkpoint가 그 우연을 깰 수 있다. merge 자체(image feature를 `1/sqrt(hidden_size)`로 scale하고 masked-scatter로 넣는 것)는 두 계열에서 같으므로, merge 함수를 따로 만들면 한 단계만 다른데 전체를 중복하게 된다. 대가로 `prepare_inputs_for_multimodal`의 인자가 8개가 되어 `#[allow(clippy::too_many_arguments)]`가 붙었다.

### 2.2 image block 번호를 graph 안에서, batch row별로 매김

`image_block_ids`는 image token이면서 바로 앞 위치는 image token이 아닌 곳을 run 시작으로 표시하고, run 시작의 inclusive 누적합을 sequence 방향으로 구한 뒤 image 표시와 곱한다. 그 결과 각 위치는 자기 row의 n번째 run이면 `n >= 1`, text면 `0`을 갖는다. `same_block`은 `query_block > 0 AND query_block == key_block`이고, mask는 `key_valid AND (causal OR same_block)`이며, 이전과 같은 `[B, 1, S, S]` shape의 additive f32(`0.0` / `f32::MIN`)로 인코딩된다.

모든 계산이 host sync 없이 device에서 이루어지며, mask가 scheduler의 prefill 경로에서 만들어지므로 이 점이 중요하다. row별로 번호를 매기므로 두 batch row의 run이 다른 위치에서 시작해도 block id를 공유하지 않는다. 한 prompt의 이미지 두 개는 block 두 개가 되므로 첫 이미지가 두 번째 이미지를 읽지 않는다. padding된 key만 mask된다. left padding된 query row는 모든 key가 mask되는데, additive `f32::MIN` 인코딩은 이 경우를 유한한 값으로 유지한다(`-inf`였다면 NaN이 된다).

### 2.3 sliding band는 merge가 아니라 모델에서 적용

`vision::merge`는 여전히 mask 하나를 반환한다. `Gemma3Model::hidden_with_caches_and_embeddings`에서 global layer(`i % pattern == pattern - 1`, text 경로 및 `make_caches`와 같은 규칙)는 external mask를 그대로 쓰고, sliding layer는 `external_mask_with_window_band(mask, window)`를 쓴다. 이는 query보다 `window` 이상 뒤에 있는 key를 모두 mask한 external mask다(`q - k < window`, `create_sliding_window_prefill_mask`의 규칙). band가 적용된 mask는 forward당 한 번, key 축이 window보다 넓을 때만 만들어진다. 그렇지 않으면 함수가 `None`을 반환하고 sliding layer는 external mask를 재사용한다. query는 key 축의 끝에 맞춰 정렬되며, boolean 입력 mask는 boolean으로 유지된다.

merge에서 mask 두 개를 반환했다면 `InputEmbeddings`와 mask를 전달하는 모든 호출자(scheduler prefill, `generate`, trace)를 바꿔야 했고, 모델의 layer pattern과 window를 vision module에 넣어야 했다. band를 모델에 두면 merge의 출력 계약은 그대로이고, window는 window가 설정되는 곳에 남는다. external mask를 받는 모든 `Gemma3Wrapper` 진입점은 `hidden_with_caches_and_embeddings`로 모이므로 branch 하나가 전부를 덮는다.

### 2.4 mlx-vlm보다 transformers를 따름

mlx-vlm의 생성 경로는 mlxcel이 쓰던 것과 같은 all-ones mask를 만들므로, mlx-vlm에 맞추는 것이 대안이었다. issue는 이를 기각했다. #2750이 이미 둘이 다를 때 mlx-vlm보다 transformers와 checkpoint를 택했고, Gemma 3는 transformers mask로 학습되었다. smoke 출력(4절)이 이 선택이 맞았는지에 대한 실질적인 확인이다.

### 2.5 reference arm은 표현할 수 없는 입력을 거부

`scripts/mlx_vlm_image_logit_trace.py`는 이제 numpy로 Gemma 3 mask를 만들고(`gemma3_prefill_mask`, 같은 run 번호 방식) `mask=None` 대신 mlx-vlm의 language model에 넘긴다. mlx-vlm은 넘겨받은 mask 하나를 모든 layer에 적용하므로 script는 sliding band를 표현할 수 없다. 그래서 sequence가 window보다 길면 band를 조용히 버리는 대신 오류로 종료한다. trace한 sequence는 1024-token window에 대해 335 token이므로 band는 효과가 없고 reference는 정확하다. PaliGemma reference는 causal로 남는다(script에서 prefix-LM mask를 재현하지 않음).

## 3. 변경 요약

파일 9개, +615/-42. commit 3개(e78337f2 fix, 4215da12 docs, 42c1acae refactor)와 origin/main merge(694f5227).

- `src/vision/merge.rs`: `MultimodalMask` 추가. `prepare_inputs_for_multimodal`이 이를 받아 공개 함수 `multimodal_attention_mask`를 호출하고, 이 함수가 id와 padding mask로 mask를 만든다(`attention_mask_for_image_runs`, `image_block_ids`).
- `src/vision/mod.rs`: `MergeStrategy::PaliGemma` 추가, strategy마다 dispatch arm 하나, strategy 문서에 실제 loader를 명시.
- `src/loading/vlm_siglip.rs`: `load_paligemma_vlm`이 `MergeStrategy::PaliGemma` 사용.
- `src/models/gemma3.rs`: `external_mask_with_window_band` 추가, external-mask branch가 global layer와 sliding layer를 나눔.
- `scripts/mlx_vlm_image_logit_trace.py`: `gemma3_prefill_mask`, window 검사, Gemma 3에 mask 전달.
- `docs/benchmarks.md`: Gemma 3 trace 행과 원인을 밝히는 문단.
- 테스트(모두 checkpoint 불필요):
  - `src/vision/merge_tests.rs`: `gemma3_mask_is_causal_with_bidirectional_image_blocks`(issue의 13-token 행렬), `gemma3_mask_without_image_tokens_is_exactly_causal`, `gemma3_mask_keeps_padding_keys_masked`(right/left padding), `gemma3_mask_finds_image_blocks_per_batch_row`. 기존 all-ones 테스트는 `paligemma_mask_stays_prefix_bidirectional`로 이름을 바꾸고 `PrefixLm`을 사용.
  - `src/models/gemma3_vlm_mask_tests.rs`(신규): `external_gemma3_mask_matches_text_prefill_past_the_window`(tiny f32 Gemma 3, window 4, pattern 2, 12 token. external-mask branch가 text prefill과 1e-4 이내로 같아야 함), `sliding_layers_intersect_the_image_mask_with_the_window`, `sliding_band_is_skipped_when_the_keys_fit_the_window`.

## 4. 결과

`gemma-3-4b-it-4bit`, `tests/fixtures/test_image_shapes.png`, `--mlxcel-prompt`, float32 CPU reference, decided gap 2에서의 trace 쌍:

| mlxcel build | Reference mask | Top-1 불일치 | Decided 불일치 | Perplexity mlxcel 대 reference |
|---|---|---|---|---|
| main | causal (기존) | 20 / 64 | 5 / 43 | 4.8500 대 5.1620 |
| main | transformers | 18 / 64 | 7 / 48 | 4.8500 대 6.0156 |
| 이 PR | transformers | 4 / 64 | 0 / 48 | 6.2651 대 6.0156 |

첫 행은 #2750 실행에서 가져온 값이다(prompt id와 target이 같음. 이번 실행에서 기존 reference 재실행은 시간 초과되었고, main build와 기존 script는 바뀌지 않았다). reference만 고쳤을 때 main은 오히려 나빠 보였는데(48개 중 7개), 이는 기존의 43개 중 5개가 틀린 mask 두 개가 부분적으로 일치한 결과였음을 확인해 준다. 남은 top-1 불일치 4개는 모두 reference 자신의 gap이 1 logit 미만인 위치이고, mlxcel의 선택은 reference의 2순위다.

두 perplexity 모두 올랐다. 둘 다 수정 전에 기록된 고정 reply stream을 채점하므로 절대값은 여기서 품질 지표가 아니고, 두 arm 사이의 일치도가 지표다.

- PaliGemma(`paligemma2-3b-6bit`): trace TSV가 수정 전후 byte 단위로 같음(`cmp`), greedy `mlxcel generate --image` text도 같음.
- GB10 smoke, CLI와 `mlxcel-server`(이미지 chat 요청 하나)가 같은 답을 냄: "three geometric shapes arranged vertically ... Top: A square, colored red. Middle: A circle, colored blue. Bottom: A triangle, colored green."
- 실패 먼저 확인: 수정 로직을 되돌리면 새 테스트 중 6개가 실패하고, tiny model도 포함된다(`row 0 logit 0: external mask 0.5711894, text prefill 0.6478121`). `gemma3.rs`의 layer 분리만 되돌리면 tiny model만 window를 넘는 첫 row에서 실패한다(`row 4 logit 0: external mask 0.76741546, text prefill 0.72449356`). 따라서 layer 분리는 merge 테스트와 별개로 자체 실패 테스트를 가진다.
- `vision::merge` 9개 통과, `models::gemma3` 49개 통과(11개 ignored), `multimodal::` 318개 통과(CUDA, `--test-threads=1`). fmt와 clippy `-D warnings` 통과.

## 5. 트레이드오프와 잔여 항목

- **window를 넘는 경우의 메모리.** sliding window보다 긴 이미지 prompt에서는 sliding layer가 두 번째 `[B, 1, S, S]` f32 mask를 갖는다. 따라서 상주 mask 메모리가 batch row당 4·S² byte에서 8·S² byte로 늘고, 만드는 동안 임시 int32 차이 배열과 bool band가 추가로 생긴다. layer마다가 아니라 forward당 한 번 만들어지며, masked SDPA의 layer별 `[B, H, S, S]` score buffer보다 작다. window 이하 prompt는 추가 할당이 없다.
- **band는 실제 checkpoint에서 reference와 대조되지 않음.** reference arm은 window보다 긴 sequence를 거부하고, trace한 prompt는 335 token이다. band는 tiny model 테스트와 행렬 테스트로만 검증된다.
- **OpenXLA Gemma 3 경로.** `src/lib/mlxcel-xla/src/prepared_gemma3.rs`(`GEMMA3_VLM_MASK_MODE = "gemma3-vlm-bidirectional-padding-f32-min-v1"`)는 여전히 모든 layer에서 기존 양방향 mask를 따른다. PR은 이를 follow-up issue 대상으로 표시했지만 작성 시점에 issue는 등록되지 않았다. 그 전까지 MLX와 XLA backend의 Gemma 3 이미지 출력은 서로 다르다.
- **PaliGemma의 reference arm**은 여전히 causal이고 mlxcel은 PaliGemma를 prefix LM으로 실행하므로, PaliGemma trace 수치는 서로 다른 mask를 비교한다. 이 PR에서 바뀌지 않았다.
- **실행되지 않은 경로:** 이미지 두 개를 넣은 실제 checkpoint 요청(block 두 개는 unit test로만 검증), Gemma3n(별도 경로), 그리고 non-decode `forward_batched_with_ids` 경로. 이 경로는 mask를 받지 않는 `forward_with_sequence_id`를 호출하며, mask를 들고 이 경로에 도달하는 VLM 호출자는 없다.
- **출력 변화.** 같은 Gemma 3 이미지 요청이 이전 release와 다른 text를 생성한다.

## 6. 학습 포인트

- parity 도구는 정확성이 아니라 일치를 검사한다. 여기서는 두 arm이 서로 다른 방향으로 틀렸고(양방향 대 causal), reference만 고치자 측정 대상이 좋아지기 전에 먼저 나빠 보였다. 같은 입력에서도 차이가 남으면 kernel을 의심하기 전에 각 arm이 무엇을 계산하도록 설정되어 있는지 확인한다.
- masking 의미는 호출 위치에서 명시적으로 지정한다. attention 계약이 다른 두 계열(prefix LM 대 image block이 있는 causal)이 merge 루틴을 공유하는 곳은 공유 기본값이 한쪽에서 조용히 틀리게 되는 지점이다.
- 최대 연속 구간은 shift, run 시작 표시, 누적합으로 device에서 번호를 매길 수 있다. host loop나 sync가 필요 없고 번호는 row별로 유지된다.
- layer에 따라 달라지는 mask는 layer pattern이 있는 곳에 둔다. merge는 mask 하나를 반환하고 모델이 sliding 변형을 만들면, window 설정이 한 곳에 남고 모든 호출자의 interface를 넓히지 않아도 된다.
- reference script는 충실하게 모델링할 수 없는 입력을 조용히 근사하지 말고 거부해야 한다. window 검사는 trace가 정확해지는 조건을 명시적으로 만든다.
