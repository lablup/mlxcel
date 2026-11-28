# PR #2416: tied 체크포인트의 embed_tokens LoRA pair 거부

**작성일**: 2026-10-16
**상태**: 머지 전에 CUDA(GB10)에서 실제 체크포인트로 구현 및 검증 완료. mlx-lm의 embedding adapter 의미는 구현하지 않음 (「미해결 사항과 후속 조치」 참고)
**위험도**: 낮음 (세 로드 경로가 공유하는 verdict 함수 하나의 로드 시점 거부. untied 체크포인트와 베이스 실행은 변경 없음)

## 요약

이슈 #2332는 #2326의 후속이다. tied 체크포인트에서는 출력 head가 embedding 행렬을 읽으므로, `model.embed_tokens`에 대한 LoRA pair는 tie의 양쪽 끝을 모두 가리킨다. fused 경로, runtime 경로, 파이프라인 stage 로더는 이런 pair를 서로 다르게, 그리고 mlx-lm과도 다르게 처리했다. 이 PR은 `lora::reject_tied_output_head_adapter`를 확장해 tied 체크포인트의 `embed_tokens` pair를 모든 경로에서 하나의 에러로, 가중치를 읽기 전에 거부한다.

## 문제 정의

mlx-lm의 `LoRAEmbedding`(`mlx_lm/tuner/lora.py`)은 pair를 `lora_a = [vocab, rank]`, `lora_b = [rank, hidden]`로 저장한다. `__call__`은 lookup에 `scale * lora_a[x] @ lora_b`를 더하고, `as_linear`는 tied head에 `scale * (x @ lora_b.T) @ lora_a.T`를 더하며, `fuse`는 양쪽이 읽는 하나의 행렬에 `W + scale * lora_a @ lora_b`를 쓴다. 즉 mlx-lm은 lookup과 tied head를 함께 조정한다. `mlx_lm.lora`는 `lora_parameters.keys`에 `model.embed_tokens`가 있을 때만 이 pair를 만든다.

mlxcel은 모든 pair를 linear 방향으로 읽기 때문에 결과가 세 가지로 갈렸다.

- fused 경로(`--adapter`, `--lora-fuse`)와 파이프라인 stage: linear 형태(`[hidden] -> [vocab]`)의 pair가 공유 행렬에 합쳐져 양쪽 끝이 모두 조정되었다(dense base 한정). mlx-lm 레이아웃의 pair는 shape 불일치로 실패했다.
- runtime 경로(`--lora`): linear 형태의 pair를 `UnifiedLinear::from_weights(.., "model.embed_tokens", ..)`가 가져갔다. Llama 계열에서는 logits에만 반영되었고, head가 `as_linear`를 쓰는 Qwen3에서는 unclaimed로 실패했다. mlx-lm pair는 다른 shape 검사에서 실패했다.

수정 전 실제 실행에서 `mlxcel-server -m llama-3.2-1b-instruct-4bit --lora <linear pair>`는 로드되었고, 첫 토큰 logprob가 head만 통해 움직였다(` Paris` -0.359에서 -0.3125, `...` -3.477에서 -3.852). 입력 embedding은 변하지 않았다.

## 기술적 선택과 그 이유

- mlx-lm을 따라가는 대신 거부한다. 따라가려면 embedding 방향 pair 타입, `UnifiedEmbedding`의 runtime lookup 항, embedding으로 head를 만드는 모든 계열의 전치된 runtime head 항이 필요하다. 이는 별도 기능이다. 거부는 #2326의 verdict와 호출 지점을 그대로 쓰며, 세 경로 모두 가중치를 읽기 전에 이미 이를 호출한다.
- verdict는 private `TiedTarget` enum으로 양쪽 끝을 표현한다: `Head(stem)`과 `Embedding(stem, head)`. embedding 변형은 tie 여부를 결정하는 head를 함께 들고 있어, stem마다 올바른 head로 tie를 판정한다.
- embedding stem 세 가지를 각자의 head와 짝지어 매칭한다: `model.embed_tokens`와 `lm_head`, `language_model.model.embed_tokens`와 `language_model.lm_head`, `model.language_model.embed_tokens`와 `lm_head`(transformers 4.52+ VLM 레이아웃). `config.json`이 `tie_word_embeddings`를 설정하거나(최상위 또는 `text_config`), 해당 pair의 head에 대한 `<head>.weight`가 저장되어 있지 않으면 tied로 본다.
- `.weight`로 끝나는 stem(`model.embed_tokens.weight.lora_a`, `lm_head.weight.lora_b`)은 base weight resolver와 같은 방식으로 읽는다. head pair에도 있던 우회 경로가 이로써 막힌다.
- 에러 문구는 종류별로 만든다. head pair는 #2326 문구를 유지한다. embedding pair는 거부 이유와 안내를 포함한다: `mlx_lm.lora`로 학습한 adapter라면 `mlx_lm.fuse`로 합친 모델을 로드하고, 아니라면 pair를 제거한다.

트레이드오프: `model.embed_tokens`를 keys에 넣어 `mlx_lm.lora`로 학습한 adapter는 tied 체크포인트에서 adapter로 로드할 수 없고 먼저 fuse해야 한다. 세 가지 조용하거나 혼란스러운 결과 대신 명확한 에러를 낸다.

## 변경 사항

- `src/lora/tied_head.rs`: `INPUT_EMBEDDINGS`, `TiedTarget`, `targeted_tied_ends`, 종류별 에러 문구, `.weight` stem 처리.
- `src/lora/tied_head_tests.rs`: stem별 verdict 단위 테스트, config 기반/head 부재 기반 tied 판정, `.weight` stem, untied 통과.
- `src/loading/adapter_load_tests.rs`: 두 방향에서 fused, runtime, 파이프라인 stage 에러가 동일한 문자열임을 확인하는 parity 테스트와, runtime 경로가 더 이상 head에 도달하지 않음을 보이는 단일 프로세스 테스트.
- `src/distributed/pipeline/stage_executor/llama.rs`, `llama_adapter_tests.rs`: stage 로더 호출 지점과 테스트.
- `docs/llama-server-compat.md`, `docs/server-features.md`: 거부 동작 설명.

## 동작 변경

tied 체크포인트에서 `model.embed_tokens`, `language_model.model.embed_tokens`, `model.language_model.embed_tokens`에 대한 LoRA pair는 `mlxcel generate --adapter`, `--pp-size N --adapter`, `--lora-fuse`, `mlxcel-server --lora`에서 로드 시점에 거부된다. untied 체크포인트는 영향이 없다. head pair는 #2326과 동일하게 동작한다.

## 검증

- 새 테스트는 수정을 기능적으로 되돌리면(embedding 매칭 비활성화) 실패한다. 되돌린 상태에서는 runtime 모델이 항을 가져간 채 로드되고, 한참 뒤 tokenizer 부재에서만 실패한다.
- `cargo clippy --profile test-fast --features cuda --lib --tests -D warnings`, `cargo fmt --check`, contract 스크립트 5개가 통과한다.
- 합성 mlx-lm 형식 adapter(rank 2, `keys: ["model.embed_tokens"]`)와 실제 체크포인트, `MLXCEL_SDPA_DETERMINISTIC=1`, 프롬프트 "The capital of France is". 수정 후 `qwen3-0.6b-4bit`와 `llama-3.2-1b-instruct-4bit`에서 `mlxcel generate --adapter`, `mlxcel generate --pp-size 2 --adapter`, `mlxcel-server --lora`를 두 레이아웃으로 실행하면 모두 같은 거부 메시지를 출력한다. 베이스 실행은 변경 없다.

## 미해결 사항과 후속 조치

- untied 체크포인트에서 `embed_tokens` pair는 fused 경로에서 여전히 합쳐지고, runtime 경로에서는 unclaimed로 거부된다.
- 다른 embedding 이름을 쓰는 tied 계열(`transformer.wte`, `model.word_embeddings`, `backbone.embeddings`)은 목록에 없다. 이들의 head는 `as_linear`를 쓰므로 runtime 경로는 이런 pair를 unclaimed로 거부하며 head에만 단독 적용하는 일은 없다.
- mlx-lm embedding adapter의 네이티브 지원(pair 타입, lookup 항, 전치된 head 항)은 별도 이슈가 필요하다.

## 통계

파일 7개 변경, 추가 508줄, 삭제 55줄. 커밋: `310e26b5` fix(lora): refuse an embed_tokens pair on a tied checkpoint, `56d0cd78` fix(lora): close the .weight stem gap in the tied-head verdict.
