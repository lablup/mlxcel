# PR #2326: 미청구 런타임 항과 tied 출력 헤드에서 LoRA 로드 실패 처리

**날짜**: 2026-10-16
**상태**: 머지 전 CUDA(GB10)에서 실제 체크포인트로 구현 및 검증 완료, tied 임베딩 후속 건은 기록만 하고 수정하지 않음 ("미검증 및 후속" 참고)
**위험도**: 중간 (기존에는 로드되면서 base 가중치를 그대로 쓰거나 delta를 조용히 버리던 어댑터가 이제 로드에 실패함)

## 요약

이슈 #1577은 #1328과 PR #1576 이후에 남은 어댑터 검증 구멍 세 개를 닫는다. 세 경우 모두 어댑터 일부가 효과 없이 로드가 성공했다.

1. 어떤 모델 생성자도 청구하지 않은 런타임 어댑터 항은 `warn!`만 남기고 버려졌다. 이제 `finish_runtime_staging()`이 정렬된 미청구 prefix를 반환하고, `load_runtime_adapted_model()`이 이를 나열한 하나의 오류로 로드를 실패시킨다. 생성 오류 경로에서도 drain은 계속 실행되므로 실패한 로드는 staged 항을 남기지 않는다.
2. tied 임베딩 체크포인트의 `lm_head` 어댑터 쌍은 모든 경로(단일 프로세스 fused, 런타임, 파이프라인 stage 로더)에서 `lora::reject_tied_output_head_adapter`가 거부한다. 판정은 가중치를 읽기 전에 safetensors 헤더만으로 한다.
3. 3-D MoE expert LoRA 쌍(`LoRASwitchLinear`)은 "Expected 2D LoRA matrices" 대신 mixture-of-experts 메시지로, 어댑터당 한 번 `validate_adapter_tensors`에서 거부된다.

## 문제 정의

unfused 런타임 경로는 base 가중치 prefix를 키로 레이어별 항을 staging하고, 레이어 생성자가 해당 prefix로 `runtime_lora::claim`을 호출해야만 항이 적용된다. 청구하는 생성자는 세 개(`Linear::from_weights`, `UnifiedLinear::from_weights`, `FusedQKVLinear`)다. `get_weight_copy`로 읽는 DeepSeek MoE router처럼 다른 곳에서 만든 projection은 항이 미청구로 남는다. 기존 `finish_runtime_staging`은 prefix마다 경고를 남기고 `()`를 반환했으므로, 로그는 어댑터가 staged라고 말하는데 서버는 base 모델로 답했다. 이것은 #1328의 실패 양상이 다른 문을 통해 나타난 것이며, `--lora`와 `--lora-scaled`의 기본 채널이다.

tied head는 하나의 어댑터에 두 가지 판정이 있었다. 단일 프로세스는 raw 가중치 맵에 `lm_head.weight`가 없어 `lm_head` 쌍을 거부했다. 파이프라인 경로는 임베딩에서 `lm_head.weight`를 합성하는 `load_text_weights`를 거쳤고, 어댑터는 그 사본에 fuse됐다. 그런데 `LlamaStageExecutor`는 config가 tie일 때 `model.embed_tokens`로 head를 만들었으므로, delta는 아무 메시지 없이 버려졌다.

MoE 제한은 이미 hard error였지만 텐서 rank 불만으로 나타났고 문서화되어 있지 않았다.

## 기술적 선택과 그 이유

- tied head는 적용하지 않고 거부한다. 적용하려면 임베딩 행렬 사본에 delta를 써야 하는데, 이는 체크포인트가 공유하도록 학습된 projection의 tie를 끊는다. `mlx_lm.lora`는 `model.layers` 안의 모듈만 감싸므로 그 도구로 학습한 어댑터는 영향받지 않는다.
- sanitize된 맵이 아니라 체크포인트에서 판정한다. 규칙은 명시적 `tie_word_embeddings: true`이거나 저장된 `lm_head.weight`가 없는 경우다. 플래그가 없고 자체 `lm_head.weight`를 저장한 체크포인트는 untied로 본다. 플래그의 family별 기본값이 다르기 때문이다. safetensors 헤더만 읽으므로 거부가 저렴하고 가중치 로드 전에 일어난다.
- 실패 양상을 통일한다. helper는 `load_model_with_adapter`, `load_model_with_adapter_specs`(fused와 unfused), 파이프라인 Llama stage 로더가 모두 호출하므로 오류 문구가 두 경로에서 같고, 테스트가 글자 단위로 비교한다.
- 검사를 완화하지 않고 생성자를 고친다. 대상 모듈을 청구 생성자 밖에서 만드는 family는 이제 어댑터를 거부하며, 해결책은 그 생성자가 claim하게 만드는 것이다.
- 미청구 목록을 정렬한다. `drain_unclaimed`는 `HashMap`을 drain하므로 정렬하지 않으면 오류 메시지가 실행마다 달라질 수 있다.
- 미청구 prefix의 출처는 `drain_unclaimed` 하나로 유지한다. 두 번째 보고 경로는 만들지 않는다.
- MoE는 공용 validator에서 어댑터당 한 번 보고해 fused, unfused, stage 경로가 일치하게 한다.

## 변경 요약

- `src/lib/mlxcel-core/src/runtime_lora.rs`, `src/lora/runtime.rs`: `finish_runtime_staging`이 `#[must_use]`이며 정렬된 prefix를 반환.
- `src/loading/mod.rs`: 새 `load_runtime_adapted_model`이 drain 결과를 바인딩하고, 생성 `?`를 유지한 뒤, 목록이 비어 있지 않으면 실패. 두 단일 프로세스 진입점에 tied head 검사를 연결.
- `src/lora/tied_head.rs`(신규): `reject_tied_output_head_adapter`. `src/lora/mod.rs`에서 export.
- `src/lora/pair_rank.rs`(신규), `src/lora/loader.rs`: `validate_adapter_tensors`의 3-D 쌍 선별.
- `src/distributed/pipeline/stage_executor/llama.rs`: 어댑터 합성 전에 공용 tied head 검사를 호출.
- 테스트: `src/loading/adapter_load_tests.rs`, `src/lora/tied_head_tests.rs`, `src/lora/pair_rank_tests.rs`, `src/lora/runtime_tests.rs`, `src/lora/test_support.rs`, `src/distributed/pipeline/stage_executor/llama_adapter_tests.rs`.
- `docs/server-features.md`, `docs/llama-server-compat.md`: 세 규칙을 문서화.

## 동작 변경

기존에 로드되던 어댑터가 이제 실패한다. 청구 생성자 밖에서 만든 모듈에 항이 있는 경우, tied 체크포인트의 `lm_head` 쌍(파이프라인 경로), 그리고 3-D expert 쌍은 더 명확한 메시지를 받는다. 런타임 로드 실패는 미청구 레이어를 한 번에 모두 보고한다.

## 검증

실제 체크포인트, `mlxcel-server --lora <adapter>`, `/completion` temperature 0, `MLXCEL_SDPA_DETERMINISTIC=1`. 어댑터는 임의의 0이 아닌 `lora_a`/`lora_b`로 만든 mlx-lm 형식 합성 어댑터이며 rank 8, scale 20이다.

- `qwen3-0.6b-4bit`, 28개 레이어 전체의 q/k/v/o(112쌍, 모두 청구됨): 로드되고 어댑터 없을 때와 출력이 달라진다. `--lora-scaled <adapter>:0.0`에서는 어댑터 없는 출력과 같다.
- `deepseek-v2-lite-chat-4bit-mlx`, `model.layers.{1,2}.mlp.gate`(MoE router) 쌍: 두 prefix를 명시한 "Runtime LoRA adapter terms for 2 layers were not claimed by the model"로 로드가 실패하고 `/health`가 503을 반환한다. `q_proj`/`o_proj` 쌍만 둔 대조군은 로드되어 일관된 답을 낸다.
- `lm_head` 쌍이 있는 `qwen3-0.6b-4bit`: tied head 오류로 로드가 실패한다.
- 되돌림 검사: 미청구 검사, tied head 판정, MoE 선별을 모두 끄면 신규 테스트 11개가 실패하고 양성 대조군은 통과한다.
- `lora::`(49 통과), `loading::`(350 통과), `stage_executor::llama`(3 통과), `cargo clippy --profile test-fast --features cuda --lib --tests -D warnings`, `cargo fmt --check`, 계약 검사가 통과한다.

## 미검증 및 후속

- 미청구 목록의 정렬은 실패하는 테스트로 고정되어 있지 않다. `finish_runtime_staging_returns_every_unclaimed_prefix_sorted...`는 정렬을 제거해도 통과한다. 키 세 개짜리 `HashMap`이 우연히 순서대로 drain되기 때문이다. 정렬은 코드 검토와 오류 형식 테스트로 보장된다.
- 알려진 미해결 건: tied 체크포인트에서 `model.embed_tokens`에 대한 어댑터 쌍은 출력 head가 청구하여 출력 head delta로 적용된다. 별도의 결정과 이슈가 필요하다.
- 끝까지 실행한 것은 dense Llama 계열과 DeepSeek-V2뿐이다. 세 생성자 밖에서 대상 모듈을 만드는 다른 family도 이제 어댑터를 거부하게 되므로, 새 family는 같은 방식으로 확인해야 한다.
