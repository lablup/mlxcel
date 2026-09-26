# 기술 보고서: PR (이슈 #1980), Gemma 3 KV 추정기 기본값

**날짜**: 2026-09-26

**상태**: 실제 gemma-3-4b-it-4bit 체크포인트에서 검증 완료, 아직 머지되지 않음

**언어**: Rust

**위험도**: 낮음

## 요약

`kv_arch.rs::attn_dims`는 설정에 헤드별 필드가 없을 때 `head_dim = hidden_size / num_heads`로 대체하며, 이때 `num_heads`는 기본값 1을 사용한다. Gemma 3의 `text_config`는 `num_attention_heads`, `num_key_value_heads`, `head_dim`, `sliding_window_pattern`을 흔히 생략하는데(gemma-3-4b-it-4bit는 `hidden_size`, `intermediate_size`, `num_hidden_layers`, `sliding_window`만 가지고 있다), 그 결과 추정기를 호출하는 모든 경로가 실제 KV 사용량을 약 2.6배 과대평가했다. 이제 `classify`는 `model_type`이 `gemma3` 또는 `gemma3_text`일 때, 필드가 없는 경우에 한해 HF `Gemma3TextConfig` 기본값(`num_attention_heads` 8, `num_key_value_heads` 4, `head_dim` 256, `sliding_window_pattern` 6)을 채워 넣고, 명시된 값은 그대로 둔다.

## 문제 정의

#1978에서 추가된 prompt-cache 스냅샷 용량 기본값, `estimate_total_memory`, `mlxcel inspect`는 모두 동일한 `kv_arch::classify` 경로를 사용한다. 실제 4B 체크포인트에서는 슬라이딩 윈도우 구조(8헤드, 4 KV헤드, head_dim 256, 6개 레이어마다 1개의 글로벌 레이어) 대신 단순 표준 어텐션(`num_heads=1`, `head_dim=2560`)으로 잘못 분류되어, 토큰당 바이트 비율이 부풀려졌을 뿐 아니라 추정기가 포착해야 할 슬라이딩 윈도우 절감 효과도 가려졌다.

## 변경 요약

- `src/execution/kv_arch.rs`에 `apply_gemma3_defaults(text, model_type)`를 추가했다. `model_type`이 `gemma3` 또는 `gemma3_text`일 때 설정의 `text_config` 객체를 복제하여, 없는 필드(`num_attention_heads`, `num_key_value_heads`, `head_dim`, `sliding_window_pattern`)만 채운다. 이후 `classify` 내부의 모든 조회(`attn_dims`, `sliding_window_pattern` 글로벌/슬라이딩 분할)는 이 병합된 값을 사용한다.
- 단위 테스트 2개를 추가했다. 하나는 실제 4B `text_config` 형태를 재현하여 `marginal_bytes_per_token == 34 * 2 * 4 * 256 * 2`를 검증하고, 다른 하나는 `num_attention_heads`가 명시된 경우 그 값은 유지되지만 여전히 없는 `num_key_value_heads` / `head_dim`은 각각 독립적으로 Gemma 3 기본값을 사용함을 검증한다.
- 다른 모델 계열의 분류 경로는 변경되지 않았다. 기본값은 `model_type`이 `gemma3` 또는 `gemma3_text`일 때만 적용된다.

## 검증

- `cargo test --release --features cuda --lib -- kv_arch:: --test-threads=1`: 28개 통과, 신규 테스트 2개 포함.
- `cargo test --release --features cuda --lib -- execution:: server::prompt_cache --test-threads=1`: 324개 통과.
- `cargo test --release --features cuda --lib -- kv_cache_advisor:: quant_advisor:: memory_estimate:: --test-threads=1`: 83개 통과, 다른 추정기 호출부에 영향이 없음을 확인.
- `cargo clippy --release --features cuda --lib --tests -- -D warnings`: 경고 없음.
- `gemma-3-4b-it-4bit`에서 시작 로그가 `architecture=standard attention (34 layers, full context)`, `kv_bytes_at_representative_tokens=2852126720`(토큰당 348,160바이트)에서 `architecture=sliding-window: 29 layer(s) capped at 1024 tokens, 5 global`, `kv_bytes_at_representative_tokens=1140850688`(토큰당 139,264바이트)로 바뀌었으며, 이는 인수 기준의 `34 x 2 x 4 x 256 x 2` 값과 정확히 일치한다. `recommended_snapshot_capacity_bytes`는 17,112,760,320(15.9 GiB)에서 6,845,104,128(6.37 GiB)로 줄었다.
- 실행 중인 서버에 3턴 대화를 보낸 결과 2, 3번째 턴 모두 prompt cache에 적중했다(프롬프트 토큰 3303/3318 중 캐시된 토큰 3294/3309), 축소된 기본값이 캐시 재사용을 깨뜨리지 않음을 확인했다.

## 남은 작업

이 이슈 범위에서 추가로 확인된 작업 없음. 수정은 Gemma 3 기본 필드 누락 문제로 한정된다.
