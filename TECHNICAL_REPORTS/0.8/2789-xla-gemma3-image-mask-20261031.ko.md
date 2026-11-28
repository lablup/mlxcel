# PR #2789: OpenXLA 경로에 transformers Gemma 3 이미지 mask 적용

**날짜**: 2026-10-31 (merge 전에 작성, 작성 시점의 PR은 open draft)
**상태**: 리뷰 중
**언어**: Rust
**위험도**: 중간 (`MLXCEL_BACKEND=xla`의 모든 Gemma 3 이미지 요청이 다른 attention mask로 prefill되어 출력이 바뀌고, 이전 mask identity로 만든 compiled bundle은 거부됨. MLX 경로, token prefill, 다른 OpenXLA family는 그대로)

## 요약

#2773은 MLX Gemma 3 이미지 prefill을 transformers mask로 바꿨다. causal attention에 image block 내부의 양방향 attention을 더하고, sliding layer에서는 이를 sliding window와 교차시키는 mask다. OpenXLA 경로는 #2773 이전의 mask를 그대로 썼다. padding이 아닌 모든 token이 padding이 아닌 다른 모든 token을 모든 layer에서 보고, window band도 적용되지 않았다. #2779 이후 두 backend가 이미지에 대해 같은 prompt id를 만들기 때문에, `MLXCEL_BACKEND=xla`와 MLX가 Gemma 3 이미지에 다른 답을 내는 원인은 이 mask 하나만 남아 있었다. Issue #2781은 #2773과 #2779 report가 남긴 이 잔여 항목을 등록한 것이다. 이 PR은 `prepare_gemma3_vlm_prefill`이 만드는 host mask를 MLX mask와 cell 단위로 같게 만들고, emitter가 기존 sliding band를 Gemma 3의 caller bias에도 적용하게 하며, mask identity를 올려 오래된 bundle을 거부하고, 작은 모델 IREE 테스트가 OpenXLA와 MLX forward를 row 단위로 비교할 수 있도록 final-position logits API를 추가한다. 그 테스트는 worst row error 2.6e-6(CUDA, f32)으로 통과하고, OpenXLA로 serving한 실제 4B checkpoint는 테스트 이미지의 세 도형을 모두 맞게 말한다.

## 1. 문제 정의

### 1.1 OpenXLA 경로가 하던 일

Gemma 3 이미지 prompt는 prepared embeddings prefill 형태로 IREE에 도달한다. `Gemma3IreeHostPreprocessor`(`src/multimodal/host_preprocessor.rs`)가 host에서 이미지를 처리하고, `prepare_gemma3_vlm_prefill`(`src/lib/mlxcel-xla/src/prepared_gemma3.rs`)이 post-scale embedding과 `[1, 1, S, S]` additive bias를 조립한다. 그 bias는 query와 key가 모두 padding이 아닐 때만 `0.0`, 그 외에는 `f32::MIN`이었고 `causal: false`였다. identity 문자열은 `gemma3-vlm-bidirectional-padding-f32-min-v1`이었다.

graph 쪽에서는 emitter(`src/lib/mlxcel-xla/src/emitter/model.rs`)의 embeddings-prefill arm이 caller bias를 global layer용 `cmask`로 쓰고, `embeddings_prefill_local_window`가 `Some`을 돌려줄 때만 sliding layer용 `cmask_local`(bias와 `i - j < W`의 교집합)을 만든다. `embeddings_prefill_uses_authoritative_mask`가 켜진 config, 즉 Gemma 3 VLM에서는 이 함수가 `None`을 돌려줬으므로 sliding layer도 완전 양방향 bias를 받았다.

### 1.2 #2773 이후 MLX 경로가 하는 일

`vision::merge::attention_mask_for_image_runs`는 `key_valid[k] && (k <= q || (block[q] > 0 && block[q] == block[k]))`일 때만 `(q, k)`를 허용한다. image block은 `image_token_id`의 maximal run이고 1부터 번호가 붙으며, 다른 token은 block 0이다. query validity는 적용하지 않는다. sliding layer에서는 `gemma3::external_mask_with_window_band`가 `q - k >= window`를 추가로 막는다. band는 한쪽 방향이므로 query가 속한 image block의 뒤쪽 member는 계속 보인다.

### 1.3 왜 중요했나

mask는 학습 계약의 일부다. 뒤쪽 prompt token을 읽는 text token은 checkpoint가 만들어 본 적 없는 hidden state를 계산하고, 이것이 #2773 이전 MLX 경로가 `tests/fixtures/test_image_shapes.png`를 "a white square-like shape"라고 설명하게 만든 결함이었다. #2779로 prompt가 같아진 뒤에는 Gemma 3 이미지 요청에서 backend가 갈라지는 원인이 mask 하나뿐이었고, OpenXLA 경로는 #2773이 고친 결함을 그대로 갖고 있었다.

## 2. 기술적 선택과 그 이유

### 2.1 host mask는 MLX predicate를 특이한 부분까지 그대로 따른다

`prepare_gemma3_vlm_prefill`은 이미 token을 검증하던 loop 안에서 위치별 block id를 계산한다. image token이 non-image token 뒤에 오거나 위치 0에 있으면 counter를 올리고, non-image 위치는 0으로 둔다. bias cell은 `attention_mask[k] == 1 && (k <= q || same_block)`일 때만 `0.0`이다. `attention_mask_for_image_runs`와 같은 predicate이고, producer가 원래 host vector로 일하므로 host에서 작성했다.

이 PR은 논란이 있을 수 있는 부분에서도 MLX 동작을 개선하지 않고 그대로 복사한다. query validity를 적용하지 않으므로 padding query row는 predicate가 허용하는 key를 그대로 가진다(그 출력은 읽히지 않는다). 위치 0의 left-padding row는 모든 key가 막히는데, 유한한 `f32::MIN` 덕분에 NaN이 생기지 않는다. 정확히 복사했기 때문에 root crate 테스트(2.4)에서 cell 단위 bit 비교가 가능하다. 여기를 "고치려면" 두 경로를 함께 바꿔야 한다.

### 2.2 band는 graph에 남긴다

`embeddings_prefill_local_window`는 이제 모든 config에 대해 `c.sliding_window`를 돌려주므로, 기존 `PrefillInput::Embeddings`와 `PrefillInput::DeepStack` arm이 sliding layer에서 caller bias에 band를 적용한다. position은 0부터 순차이므로 buffer index가 곧 position이고, graph의 `i - j < W`는 MLX가 쓰는 predicate와 같다.

기각한 대안은 sliding layer용 host mask를 하나 더 만드는 것이었다. 그러면 prepared-prefill payload schema와 그 모든 소비자가 바뀌지만, graph에는 이미 band 코드가 있었고 바꿔야 할 것은 gate 하나였다. authoritative flag를 켜지 않는 config는 원래 band를 받고 있었으므로, 이 변경은 Gemma 3 VLM에만 영향을 준다.

`embeddings_prefill_uses_authoritative_mask`는 남긴다. 여전히 `f32::MIN` masked-value 검사(`model.rs`에서 caller bias의 masked value 선택)와 `iree.rs`의 Gemma 3 VLM gate를 결정하기 때문이다. doc comment는 caller가 authoritative base mask를 주고 emitter가 여전히 band를 적용한다고 고쳤다. band 밖 cell은 MLX가 `f32::MIN`을 쓰는 곳에 emitter의 `neg_big`(-1e30)을 쓴다. 둘 다 softmax weight가 0이 되며, PR은 band cell의 bit 일치를 주장하지 않는다.

### 2.3 identity는 올리고 schema는 그대로

`GEMMA3_VLM_MASK_MODE`는 `gemma3-vlm-causal-image-blocks-sliding-band-f32-min-v2`가 된다. 이 값은 `Gemma3VlmCompatibilityContract::stable_identity`의 일부이므로, 이전 동작으로 만든 bundle이나 compiled artifact는 잘못된 mask 의미로 재사용되지 않고 identity 검사에서 실패한다. 이름에는 변경의 두 부분(host block mask, graph band)이 모두 들어 있다. payload의 shape과 dtype은 바뀌지 않았으므로 `GEMMA3_VLM_PREPARED_PREFILL_CONTRACT`는 그대로다.

### 2.4 parity 테스트를 위한 public logits 진입점

`diagnostics` feature 밖에는 embeddings-prefill logits를 돌려주는 API가 없었고, parity 테스트는 server가 실제로 실행하는 graph가 필요했다. `XlaPreparedPrefillLogits`(`src/lib/mlxcel-xla/src/batch.rs`, `iree` feature로 export)는 `IreeRaggedLlama`를 감싸고, engine의 hidden size, capacity, position mode로 `PreparedIreePrefill::prepare_for_mode`를 실행한 뒤 slot 0의 마지막 위치 logits를 돌려준다. diagnostics 전용 경로가 아니라 ragged engine을 거치므로 테스트가 production band 코드를 실행한다. 대가는 지금은 테스트만 쓰는 작은 public type이다.

### 2.5 세 층의 테스트, 모두 main에서 먼저 실패

- `prepared_gemma3::tests::exports_causal_image_block_mask`는 `[pad, text, img, img, text, img, text]`의 7x7 행렬 전체를 적어 두고, `image_blocks_start_at_position_zero_and_text_only_prompts_stay_causal`은 위치 0에서 시작하는 run과 이미지가 없는 prompt를 다룬다. main에서는 `query 1 key 2: left 0.0, right f32::MIN`.
- `emitter::model::gemma3_vlm_mask_tests::authoritative_embeddings_mask_keeps_the_sliding_window_band`는 flag가 켜져도 `Some(window)`, window가 없으면 `None`인지 확인한다. main에서는 `left: None, right: Some(4096)`.
- `src/loading/vlm_gemma_xla_mask_tests.rs`(root crate, `xla-backend`, IREE 불필요)는 issue의 prompt와 edge case(위치 0의 run, 이미지 없음, text token 하나로 나뉜 두 block, prompt 중간의 padding)에서 `prepare_gemma3_vlm_prefill`과 `vision::merge::multimodal_attention_mask`를 bit 단위로 비교한다. 두 구현을 서로에 고정하므로, 나중에 어느 쪽이 바뀌어도 여기서 실패한다.
- `tests/xla_prepared_prefill.rs::local_task_gemma3_vlm_prefill_matches_mlx_mask`(ignored, 실제 IREE)는 작은 f32 Gemma 3 text checkpoint(2 layer, pattern 2, window 4)와, 위치 3부터 6까지 4-token image block이 있는 12-token prompt를 만든다. engine이 마지막 위치 logits만 돌려주므로 각 row를 그 prefix의 마지막 row로 확인하고, 양쪽 모두 해당 prefix로 mask를 만든다. MLX는 #2773 mask로 `forward_with_embeddings_and_sequence_id`를 실행하며, image row는 `vision::merge`처럼 `sqrt(hidden)`으로 나눠 넣는다. `ctor`가 `MLX_ENABLE_TF32=0`을 고정해 MLX 쪽을 완전한 f32로 만든다. device는 `MLXCEL_XLA_DEVICE`에서 읽는다(기본값 `local-task`).

failing-first 실행은 두 수정을 분리해 보여 준다. 둘 다 되돌리면 row 0은 맞고 row 1부터 11까지 0.06에서 4.8만큼 틀린다. band만 되돌리면 row 0부터 3까지 맞고 row 4부터 11까지 실패하는데(row 4는 1.26 차이), 4-token window를 처음 넘는 row에서 시작한다. 즉 band는 host mask와 별개로 자기만의 실패 증거를 갖는다.

## 3. 변경 요약

12개 파일, +572/-45. commit 두 개(dff56af0 fix, e3b03c28 docs)와 origin/main merge(196fae3f).

- `src/lib/mlxcel-xla/src/prepared_gemma3.rs`: 위치별 image block id, causal image-block mask, v2 mask identity, module doc, 다시 쓴 테스트와 새 테스트.
- `src/lib/mlxcel-xla/src/emitter/model.rs`: `embeddings_prefill_local_window`가 조건 없이 window를 돌려준다. gate 테스트를 다시 썼다.
- `src/lib/mlxcel-xla/src/emitter/config.rs`: `embeddings_prefill_uses_authoritative_mask` doc comment.
- `src/lib/mlxcel-xla/src/batch.rs`, `src/lib/mlxcel-xla/src/lib.rs`: `XlaPreparedPrefillLogits`와 export.
- `src/lib/mlxcel-xla/src/iree.rs`: compatibility identity 테스트가 v2 mask mode를 기대한다.
- `src/loading/vlm_gemma.rs`, `src/loading/vlm_gemma_xla_mask_tests.rs`(신규): OpenXLA 대 MLX mask 테스트 module.
- `src/loading/vlm_gemma_xla_tests.rs`: `xla_gemma3_reference_check` boundary gate의 `expected_mask`가 `image_token_index`로 image-block mask를 만들고, progress 문구도 그에 맞게 바뀌었다.
- `src/multimodal/host_preprocessor.rs`: `Gemma3IreeHostPreprocessor` doc이 mask는 causal과 image block이고 band는 graph에서 적용된다고 설명한다.
- `tests/xla_prepared_prefill.rs`: 작은 Gemma 3 fixture와 ignored IREE parity 테스트.
- `docs/benchmarks.md`: Gemma 3 mask 문단에 OpenXLA 경로가 같은 mask를 만들고 graph에서 band를 적용한다는 내용 추가.

## 4. 결과

- `cargo fmt --all -- --check` clean. mlxcel-xla lib 365개 테스트(`--profile test-fast --features mlxcel-core/cuda`) 통과. mlxcel-xla clippy `--all-targets -D warnings` clean.
- `cuda,xla-backend` root lib의 `loading::` module: 383개 통과.
- release `cuda,xla-iree` clippy(`-p mlxcel-xla -p mlxcel --lib --test xla_prepared_prefill -D warnings`) clean. release `cargo check --features cuda,xla-reference-diagnostics` clean(boundary gate의 expected-mask 변경 포함).
- `MLXCEL_XLA_DEVICE=cuda MLXCEL_XLA_PRECISION=f32`로 `xla_prepared_prefill --ignored`: 새 테스트가 1e-3 허용치 대비 worst row error 2.6e-6으로 통과.
- 실제 checkpoint: `MLXCEL_BACKEND=xla MLXCEL_XLA_DEVICE=cuda MLXCEL_XLA_CONTEXT_CAPACITY=512`로 `gemma-3-4b-it-4bit`를 띄운 `mlxcel-server`(release `cuda,xla-iree`)가 `test_image_shapes.png`(prompt 278 token) 이미지 요청 하나에 "A red square is located in the upper left corner. A blue circle is positioned in the upper right. A green triangle is placed below the circle."라고 답했다.

## 5. 트레이드오프와 잔여 항목

- **band는 실제 checkpoint에서 실행되지 않았다.** 278-token smoke prompt는 1024 window보다 짧아 band가 no-op이다. band는 window 4인 작은 테스트만 다룬다.
- **CUDA 정밀도.** CUDA는 기본으로 f16 contraction을 써서 mask 차이가 없어도 row 0에서 3e-3 차이가 나므로, CUDA에서는 `MLXCEL_XLA_PRECISION=f32`일 때만 parity 테스트가 통과한다. 기본 `local-task` device는 테스트가 원래 겨냥한 f32 경로다.
- **이 host에서 `local-task`를 실행할 수 없었다.** IREE가 해당 device를 만들지 못한다(`thread creation failed with 22`). main과 이 branch 모두 같으므로, `xla_prepared_prefill`의 기존 테스트 두 개(`local-task` 고정)와 새 테스트의 기본 device 실행은 여기서 돌리지 못했다.
- **기존 clippy 경고.** `xla-iree` 없이 `xla-backend`만 켠 root lib은 `host_preprocessor_molmo_tests.rs`와 `worker_failfast.rs` 때문에 clippy를 통과하지 못한다. 이 PR은 두 파일을 건드리지 않는다.
- **bundle 무효화.** identity가 바뀌므로 cache된 Gemma 3 VLM bundle이나 compiled artifact는 다시 만들어야 한다. 의도된 동작이며, 미리 만든 bundle을 쓰는 배포는 조용한 불일치 대신 거부를 보게 된다.
- **출력 변화.** OpenXLA backend의 Gemma 3 이미지 요청은 이전 build와 다른 text를 내며, 이제 MLX backend의 mask와 일치한다.
- **새 public surface.** `XlaPreparedPrefillLogits`는 diagnostics가 아닌 public type이고, 호출자는 테스트 하나이며 batch는 4로 고정되어 있다.
- **범위 밖.** PaliGemma와 Gemma3n에는 OpenXLA Gemma 3 mask 경로가 없다. `spike/openxla/gemma3_vlm_mask_check.py`(#869의 과거 spike, CI 밖)는 여전히 이전 mask를 설명한다.

## 6. 학습 포인트

- 한 backend의 의미가 바뀌면, 형제 backend의 대응 구현은 기본적으로 잔여 항목이 된다. #2773은 OpenXLA 대응을 follow-up이라고 적었지만 issue를 만들지 않았고, #2781은 다음 report가 이를 다시 적었기 때문에 생겼다. follow-up을 만드는 PR에서 issue도 함께 등록해야 한다.
- 원래 "caller의 mask가 완전하니 window를 건너뛰라"는 뜻이던 flag는 두 결정을 묶고 있었다. base mask를 누가 소유하는지와, graph가 band를 적용하는지다. 둘을 분리하는 일(flag는 값 검사와 gate에만 쓰고 band는 항상 적용)은 band 코드가 이미 공유되어 있었기 때문에 한 줄 변경으로 끝났다.
- 같은 predicate의 두 구현은 runtime 없이 bit 단위로 직접 비교하는 테스트로 서로에 고정한다. end-to-end parity 실행보다 싸고, 어느 쪽이 어긋나든 실패한다.
- 마지막 위치 logits만 주는 API로도 모든 prefix를 돌리면 row 단위 parity를 얻을 수 있다. prefix마다 mask를 새로 만들기 때문에, row별 결과가 어떤 mask 요소가 틀렸는지도 짚어 준다(여기서는 window를 처음 넘는 row 4).
- GPU에서 mask만 비교하려면 수치 noise floor가 허용치보다 낮아야 한다. 양쪽을 f32로 고정한 것(`MLX_ENABLE_TF32=0`, `MLXCEL_XLA_PRECISION=f32`)이 3e-3 baseline을 2.6e-6으로 바꿨다.
