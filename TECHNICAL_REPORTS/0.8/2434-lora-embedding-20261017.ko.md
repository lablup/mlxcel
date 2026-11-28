# PR #2434: embed_tokens의 mlx-lm LoRAEmbedding adapter 지원

**작성일**: 2026-10-17
**상태**: 머지 전에 CUDA(GB10)에서 실제 체크포인트로 구현 및 검증 완료. 양자화된 embedding 테이블로의 fuse와 다른 tied 계열은 다루지 않음 (「미해결 사항」 참고)
**위험도**: 중간 (embedding lookup과 tied head에 runtime LoRA term을 추가하고, Llama, Qwen2, Qwen3에 대한 #2416 거부를 해제함. 다른 계열은 거부 유지)

## 요약

#2421을 닫는다. #2416(#2332) 위에 구축되었다. mlxcel은 이제 `model.embed_tokens`의 mlx-lm `LoRAEmbedding` adapter를 fused(`--adapter`, `--lora-fuse`, 파이프라인 stage 로더)와 unfused(`--lora`) 모두에서 mlx-lm과 같은 의미로 제공한다. tied 체크포인트에서는 pair가 lookup과 head를 함께 조정하고, untied 체크포인트에서는 lookup만 조정한다.

## 문제 정의

#2416은 tied 체크포인트의 `embed_tokens` pair를 세 로드 경로 모두에서 거부했다. 두 가지 공백이 남아 있었다.

- mlxcel은 이런 adapter를 unfused로 제공할 수 없었다.
- untied 체크포인트에서 fused 경로는 `embed_tokens` pair를 fuse했지만 runtime 경로는 unclaimed로 거부했다(`src/lora/runtime.rs:387`). linear 생성자만 term을 claim했기 때문이다(`src/lib/mlxcel-core/src/layers.rs:1319`).

기준 의미는 mlx-lm 0.32.2와 mlx 0.32.4의 것이다. `LoRAEmbedding`은 `lora_a [vocab, r]`, `lora_b [r, hidden]`을 저장한다. `__call__`은 `scale * lora_a[x] @ lora_b`를 더하고, `as_linear`는 `scale * (x @ lora_b.T) @ lora_a.T`를 더하며, `fuse`는 `W + (scale * lora_a @ lora_b).astype(W.dtype)`를 쓴다. `linear_to_lora_layers`는 `lora_parameters.keys`가 `model.embed_tokens`를 지정할 때만 이를 감싼다.

## 기술적 선택과 그 이유

- pair 레이아웃. base weight stem이 `embed_tokens`로 끝나는 pair는 테이블의 vocab 행 수에 대해 다음 순서로 해석한다. mlx-lm `LoRAEmbedding`(`a [V, r]`, `b [r, H]`), mlx-lm 방향의 linear `[hidden] -> [vocab]` pair(`a [H, r]`, `b [r, V]`), PEFT 방향(`a [r, H]`, `b [V, r]`). 셋 모두 `delta = A @ B`로 정규화된다. `vocab == hidden`이면 mlx-lm 해석이 우선한다.
- fuse dtype. fused delta는 f32로 계산하고 더하기 전에 테이블 dtype으로 캐스팅한다(`mlx_lm.fuse`와 동일). 따라서 테이블은 dtype을 유지한다. fused linear가 f32로 승격되는 동작은 그대로다.
- fuse는 dense base 한정. 양자화된 테이블(`.scales` plane이 있는 것)은 `--lora`를 안내하는 에러로 거부한다. packed 열 너비와 같은 delta는 shape 불일치를 통과할 수 있으므로, shape 불일치가 아니라 `.scales` plane으로 판단한다.
- runtime staging. embedding pair는 별도 map에 `RuntimeLoraEmbeddingTerm`(f32 `a`, `b`, adapter의 `SharedLoraScale`)으로 담기며 `Arc`로 보관된다. `claim_embedding`이 term을 lookup에 넘긴다. `claim(prefix)`는 embedding의 prefix로 만들어진 모든 linear에 tied-head 형태도 붙인다. `drain_unclaimed`는 lookup이 claim하지 않은 embedding term을 보고하므로, head에만 붙은 경우에도 로드는 실패한다.
- claim 시점의 너비 검사. `UnifiedEmbedding::from_weights`가 모든 term을 테이블의 실제 `[rows, width]`(packed 테이블은 group 수 곱하기 조정된 group size)와 대조한다. hidden 너비가 틀린 pair는 첫 forward에서 abort되는 대신 로드에서 실패한다.
- 거부 해제는 좁게. #2416 거부는 `model.embed_tokens`에 한해, 그리고 해당 계열의 생성자가 head를 embedding에서 읽을 때만(`applies_tied_embedding_pairs`) 해제한다. `tie_word_embeddings: true`인 `ModelType::Llama` / `Qwen2`, 그리고 config가 `false`로 두지 않은 `Qwen3`다. 그 외는 문구가 바뀌지 않은 #2416 에러를 유지한다.

이슈 설계와 다른 점:

- Llama의 tied head는 `Arc`를 직접 들고 있는 대신, 공유 term의 버퍼(transposed view, 같은 scale handle)로 만든 `RuntimeLoraTerm`을 받는다. head는 `UnifiedLinear`이고 그 fused-kernel guard(`has_active_runtime_lora`, `as_quantized_weight`, `quantized_weight`)가 이미 `runtime_loras`를 기준으로 하므로, `UnifiedLinear`에 새 분기를 넣지 않고도 guard가 올바르게 유지된다. 텐서 데이터는 복사되지 않는다.
- `lm_head.weight`가 저장되어 있지 않다는 이유로만 tied인 Llama/Qwen2 체크포인트(`tie_word_embeddings: true` 없음)는 거부를 유지한다. Llama의 `tie_word_embeddings` 기본값은 `false`이므로 `sanitize_tied_embeddings`가 만드는 `lm_head` 사본으로 head를 만든다. fused 경로는 fuse 후에, runtime과 stage 경로는 fuse 전에 이 사본을 만들기 때문에 경로 간 결과가 달라질 수 있다. Qwen3는 기본이 tied이고 `as_linear`를 읽으므로 이 경우에도 해제한다.
- `mistral`(`ModelType::Llama`)은 같은 Llama 생성자로 로드되므로 단일 프로세스 경로에서 지원된다. 파이프라인 stage executor는 이전과 같이 adapter를 전부 거부한다. `qwen2`는 파이프라인 stage 계열이 없으므로 파이프라인 기준은 Llama만 해당한다.

## 변경 사항

- `src/lora/embedding.rs` (신규): 레이아웃 판별과 `delta = A @ B`로의 정규화.
- `src/lora/loader.rs`: fuse 루프가 embedding 대상을 해당 delta로 보낸다. `fuse_lora_weights_into`를 공유하는 `--adapter`, `--lora-fuse`, 파이프라인 stage 로더에 모두 적용된다.
- `src/lora/runtime.rs`: embedding pair를 정규화해 `PendingLoraEmbeddingTerm`으로 staging한다. shape 검사는 `validate_pair_shapes`를 rows = vocab, columns = hidden으로 재사용한다.
- `src/lib/mlxcel-core/src/runtime_lora.rs`: `RuntimeLoraEmbeddingTerm`, `stage_with_embeddings`, `claim_embedding`, `head_term()`을 통한 head 형태, `drain_unclaimed`. `stage`는 embedding map도 비운다.
- `src/lib/mlxcel-core/src/layers.rs` (embedding 영역만): `Embedding`과 `QuantizedEmbedding`이 claim 시점에 한 번 만들어진 `EmbeddingLoraTerms`를 가진다. `forward`는 lookup term을 더하기 전에 출력 dtype으로 캐스팅해 더하고(mlx-lm과 동일), `as_linear`는 `apply_terms`로 head term을 더한다. `clone_shared`는 `Linear::clone_shared`처럼 term을 가져가지 않는다.
- `src/lora/tied_head.rs`: 위에서 설명한 거부 해제.
- 테스트: `src/loading/adapter_embedding_tests.rs`, `src/lora/embedding_tests.rs`, `src/lora/tied_head_tests.rs`, `src/lora/test_support.rs`, `src/loading/adapter_load_tests.rs`, `src/distributed/pipeline/stage_executor/llama_adapter_tests.rs`, `runtime_lora` 테스트.
- 문서: `docs/server-features.md`, `docs/llama-server-compat.md`.

계열별 도달 경로: Llama와 Qwen2는 tied head를 `UnifiedLinear::from_weights(weights, "model.embed_tokens", ..)`로 만들며(`src/models/llama3.rs:1281`, 그리고 stage 로더), 이제 `claim`이 head term을 붙인다. Qwen3는 `embed_tokens.as_linear`로 읽는다(`src/models/qwen3.rs:720`).

## 동작 변경

tied Llama, Qwen2, Qwen3 체크포인트(위 조건)에서 `embed_tokens` LoRA pair는 거부 대신 적용되고, untied 체크포인트에서는 runtime 경로가 lookup용으로 이를 claim한다. 다른 tied 계열, VLM embedding stem, 암묵적으로 tied인 Llama/Qwen2 체크포인트는 #2416 에러를 유지한다. 베이스 실행은 변경 없음.

## 검증

명령(GB10, `--features cuda`, test-fast 프로파일, 모든 GPU 실행은 `gpu-lock` 아래):

- `cargo test ... --lib lora::`: 66 통과.
- `loading::adapter_`: 9 통과, 1 ignored(실제 체크포인트 gate). `llama::adapter_tests`: 4 통과. `partial_loading_adapter`: 16 통과. `-p mlxcel-core runtime_lora`: 8 통과.
- `cargo clippy --profile test-fast --features cuda --lib --tests -- -D warnings`, `cargo fmt --all -- --check`, contract 스크립트 통과.

Revert 확인: 수정을 기능적으로 되돌리면(verdict 해제 끔, fuse 루프와 runtime staging의 embedding 분기 끔) `adapter_embedding_tests` 3개, 새 `tied_head` 수용 테스트 3개, `a_fully_claimed_runtime_adapter_still_loads`가 실패하고, 복원하면 모두 통과한다. `embedding_tests`와 core `runtime_lora` 테스트는 새 함수를 직접 검증하므로 main에 대해서는 실행할 수 없다.

실제 체크포인트: adapter는 `mlx_lm.lora`로 학습했다(12 iter, rank 8, scale 20, keys `model.embed_tokens`와 `self_attn.q_proj`, 마지막 2개 레이어, seed 7). 기준은 `mlx_lm.fuse --dequantize`로 만들었다. logits는 `MLXCEL_SDPA_DETERMINISTIC=1`에서 ignored gate `real_checkpoint_embedding_adapter_parity`로 마지막 위치 기준 측정했다. gate는 argmax 일치, 허용 오차 내 동률을 제외한 top-5 일치, max |diff| <= 0.25, runtime scale 0이 베이스 모델과 byte 단위로 동일함을 검사한다. 프롬프트 "The capital of France is":

| 체크포인트 | 경로 | runtime vs fused | stage vs fused | scale 0 = base |
|---|---|---|---|---|
| Llama-3.2-1B-Instruct-bf16 (tied) | fused, runtime, stage | 0.125 | 0.0000 | 동일 |
| deepseek-coder-1.3b-instruct-mlx (untied) | fused, runtime, stage | 0.024 | 0.0000 | 동일 |
| qwen2.5-0.5b-instruct-bf16 (tied) | fused, runtime | 0.25 | n/a | 동일 |
| qwen3-0.6b-4bit (tied, 양자화) | runtime | n/a | n/a | 동일 |

mlxcel이 로드한 `mlx_lm.fuse --dequantize` 체크포인트와 runtime 경로를 비교하면, Llama-3.2-1B(max |diff| 0.125, 기준 top-5에서 0.0625), deepseek-coder(0.047, 0.031), qwen2.5-0.5b(0.50, 0.125)는 top-5가 같다. qwen3-0.6b-4bit(0.75, 0.125)는 앞 4개가 같고 기준의 4~6위는 12.938에서 3-way 정확 동률이다. 이 비교는 양자화된 base를 dequantize된 fusion과 비교한 것이며, 두 번째 프롬프트에서는 top-5가 순서까지 같다. 참고로 mlx-lm 자체의 unfused LoRA와 fused 모델의 차이는 최대 0.56(Llama), 0.50(Qwen)이다.

라이브 서버(`/v1/completions`, `logprobs: 5`, greedy), Llama-3.2-1B-bf16과 qwen3-0.6b-4bit: scale 1에서 adapter가 출력을 바꾼다. `POST /lora-adapters [{"id": 0, "scale": 0.0}]` 후에는 텍스트와 모든 top-5 logprob가 별도 base 서버와 정확히 같고, 1.0으로 되돌리면 처음의 scale-1 응답과 정확히 같다. `mlxcel generate --adapter`와 `--pp-size 2 --adapter`는 tied Llama와 untied deepseek-coder에서 같은 텍스트를 낸다.

## 리뷰 중 변경

- 보안 리뷰에서 runtime 로더의 packed 테이블 검사(scales group 수)가 최대 4개의 hidden 너비를 허용하므로, 양자화 테이블에서 잘못된 adapter가 로드된 뒤 MLX shape 불일치로 프로세스를 abort시킬 수 있음이 발견되었다. 너비는 이제 claim 시점에 고정된다. 양자화 base 위의 linear term에 대한 같은 약한 검사는 이 PR 이전부터 있었고 변경하지 않았다.
- fused 경로는 양자화 테이블을 `.scales` plane으로 거부한다.
- Qwen3 `as_linear` head term은 forward마다가 아니라 claim 시점에 한 번 만든다. 이 변경 후 실제 체크포인트 runtime logits는 네 체크포인트 모두에서 이전 실행과 byte 단위로 동일하다.
- Pipeline Parallel CI 스모크 테스트(`pipeline_multi_stage_two_host_logical_smoke`, debug 프로파일)가 2 MiB 모델 워커 스택을 넘쳤다. debug 빌드에서 `try_load_config_backed_model_from_dir`는 모든 family 분기의 임시값에 각자 스택 슬롯을 주어(main 기준 약 1.45 MB), term 집합이 처음 `Embedding`과 `QuantizedEmbedding`에 인라인으로 더한 48바이트가 이를 약 24 KB 키워 한도를 넘겼다. 이제 term 집합은 `Option<Box<EmbeddingLoraTerms>>`이고, 디스패처는 각 family를 out-of-line `load_dir_into`로 로드하므로 프레임이 약 15 KB이다. GB10에서 통과하는 최소 `RUST_MIN_STACK`은 약 2.31 MB에서 약 0.92 MB로 줄었고, 실제 체크포인트 runtime logits는 그대로이다.

## 미해결 사항

- 양자화된 embedding 테이블로의 fuse(`--lora`는 unfused로 제공한다).
- Llama decoder와 Qwen3 이외의 tied 계열, VLM embedding stem은 거부를 유지한다.
- 다른 이름의 embedding 테이블(`transformer.wte`, `model.word_embeddings`)은 여전히 linear pair로 읽는다.

## 통계

17개 파일 변경, 1733줄 추가, 83줄 삭제. 커밋: `63306d7d` feat(lora): support mlx-lm LoRAEmbedding adapters on embed_tokens, `bbeef290` fix(lora): justify the Arc around non-Send embedding terms, `848cd90b` fix(lora): pin an embedding term's width at claim and refuse packed fuse.
