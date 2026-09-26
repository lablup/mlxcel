# 기술 보고서: PR (이슈 #1979), Nemotron-H Nano Omni 시퀀스별 상태

**날짜**: 2026-09-26

**상태**: 구현 및 단위 테스트 완료. Nano Omni 체크포인트가 없어 실제 서버에서 래퍼를 실행하지 못했음. 아직 머지되지 않음

**언어**: Rust

**위험도**: 낮음

## 요약

`NemotronHNanoOmniVlModel`은 `LanguageModel`의 단일 시퀀스 부분만 구현하고 있었다. 시퀀스별 진입점(`forward_with_sequence_id`, `prepare_sequence_state`, `release_sequence_state_by_id`, 스냅샷 메서드)은 트레이트 기본값으로 떨어졌기 때문에, mlxcel-server에서는 모든 요청이 Nemotron-H 텍스트 모델의 단일 fallback 슬롯에서 실행되었다. 동시 요청이 KV와 Mamba 상태를 공유했고, model-owned 프롬프트 재사용은 스냅샷을 한 번도 저장하지 않았다. 이제 래퍼는 시퀀스별 및 스냅샷 메서드 전체를 내부 `NemotronHModel`에 전달하고, 임베딩 prefill은 요청 자신의 슬롯에서 실행된다.

## 문제 정의

스케줄러는 요청마다 model-owned 슬롯을 할당하고 `prepare_sequence_state`를 호출한 뒤, 멀티모달 요청은 `forward_last_logits_with_embeddings_and_sequence_id`로, 텍스트 요청이나 채택된 prefix 이후의 suffix는 `forward_last_logits_with_sequence_id`로 prefill하고, `forward_with_sequence_id`로 디코드한다. 기본값으로는 이 경로 모두가 시퀀스 ID를 무시했다. 내부 모델의 `forward_last_logits_with_embeddings_and_sequence_id` 역시 텍스트 모델이므로 임베딩과 ID를 버리기 때문에, 이 메서드를 단순 위임했다면 이미지와 오디오 행이 사라졌을 것이다.

## 변경 요약

- `NemotronHModel`에 `forward_with_inputs_embeds_and_sequence_id`와 `last_logits_with_inputs_embeds_and_sequence_id`를 추가했다. 두 메서드는 미리 계산된 임베딩에서 시작해 주어진 시퀀스 ID의 슬롯에서 레이어 스택을 실행한다. CLI가 쓰는 `forward_with_inputs_embeds`는 이제 첫 번째 메서드의 `None` 슬롯 경우다.
- 래퍼는 두 임베딩 진입점을 직접 구현하고(임베딩은 새 내부 메서드로, 토큰 경로는 내부 시퀀스 ID 메서드로), `forward_with_sequence_id`, `forward_last_logits`, `forward_last_logits_with_sequence_id`, `sequence_state_layout`, `prepare_sequence_state`, `release_sequence_state_by_id`, `reset_runtime_state`, `supports_snapshot_reuse`, `snapshot_sequence_state`, `restore_sequence_state`, `supports_batching`, `supports_padded_prefill`는 하드코딩 대신 텍스트 모델에 위임한다.
- admission 훅은 추가하지 않았다. Nemotron Omni의 이미지와 오디오 준비는 상태가 없고 시퀀스에 실려 가는 `InputEmbeddings`를 반환하므로, Qwen VL MRoPE나 Gemma 4 per-layer inputs와 달리 바인딩할 fallback 슬롯 부수 상태가 없다.
- `src/vision/snapshot_forwarding_tests.rs`의 `EXEMPT` 항목을 제거해 소스 스캔 테스트가 이 래퍼를 검사한다.

## 검증

- 신규 테스트 `sequence_ids_keep_isolated_state_on_token_and_embedding_paths`는 1개 어텐션 레이어 Nemotron-H와 합성 RADIO 타워로 래퍼를 만들고, 두 시퀀스 ID가 같은 입력에 대해 토큰 경로와 임베딩 경로 모두에서 동일한 prefill 및 디코드 logits를 내는지, 스냅샷과 해제 동작이 맞는지 확인한다. 이전 래퍼에서는 실패하며(두 번째 시퀀스 prefill의 최대 logit 차이 0.0596), 소스 스캔 테스트도 실패한다.
- `cargo test --release --features cuda --lib -- vision:: models::nemotron server::batch --test-threads=1`: 998개 통과.
- `cargo clippy --release --features cuda --lib --tests -- -D warnings`, `cargo fmt --all --check`: 문제 없음.
- 내부 모델 텍스트 전용 확인: `mlxcel-server -m nemotron-h-30b-4bit`로 3턴 대화, 프롬프트 토큰 3294 / 3313 / 3332 중 `cached_tokens` 0 / 3301 / 3320.

## 남은 작업

이슈의 Nano Omni 래퍼 실서버 멀티턴 확인은 수행하지 못했다. `/home/inureyes/models/mlx` 아래에 `nemotron_h_nano_omni` 체크포인트가 없다. 체크포인트를 구할 수 있을 때 실행해야 한다.
