# PR #2417: SDPA 마스크를 q의 dtype이 아닌 attention dtype으로 캐스팅

**작성일**: 2026-10-16
**상태**: 머지 전에 CUDA(GB10)에서 실제 체크포인트로 구현 및 검증 완료. 관련 dtype 문제 두 가지는 후속 조치로 남김 ("미검증 및 후속 조치" 참고)
**위험도**: 낮음 (C++ SDPA 브리지의 dtype 캐스트 변경. q, k, v의 dtype이 같으면 기존과 동일한 캐스트. 테스트 2개 추가)

## 요약

이슈 #2336 (참조 #1793, #2330): CUDA에서 `mlxcel-server --kv-cache-mode fp16+turbo4`로 Gemma 4를 실행하면, 1024 토큰 슬라이딩 윈도우를 넘는 프롬프트에서 `[scaled_dot_product_attention] Mask type must promote to output type float32`로 abort되었다. 브리지는 additive 마스크를 쿼리의 dtype으로 캐스팅했지만, SDPA는 `result_type(q, k, v)`로 attention을 수행하고, fp16 TurboQuant K/V 위의 bf16 쿼리에서는 이 값이 float32이다. 이 PR은 floating 마스크(와 attention sinks)를 `result_type(q, k, v)`로 캐스팅하는 공용 헬퍼 `sdpa_mask_for_inputs`를 추가하고 SDPA 브리지 세 곳에서 사용한다.

## 문제 정의

이슈의 가설은 q, k, v 중 하나가 float32라는 것이었다. 크래시가 난 레이어의 SDPA 호출을 로깅하니 달랐다: `q`는 bf16 `[1,16,1980,256]`, `k`와 `v`는 fp16 `[1,8,1980,256]`, 마스크는 f32 `[1980,1980]`.

- TurboQuant 캐시는 항상 fp16 K/V를 반환하므로, bf16 모델은 fp16 K/V 위의 bf16 쿼리를 SDPA에 넘긴다.
- SDPA는 `result_type(q, k, v)`로 attention을 수행하며, bf16 + fp16에서는 float32이다. 이 타입으로 promote되지 않는 dtype의 array 마스크는 거부한다.
- 브리지는 additive 마스크를 q의 dtype(bf16)으로 캐스팅했다. CUDA 오버레이의 패치된 promotion 표는 bf16 + float32를 bf16으로 유지하므로, 마스크가 float32 출력으로 promote되지 않아 SDPA가 예외를 던졌다. 예외는 워커 스레드에서 처리되지 않아 프로세스가 abort되었다.
- 슬라이딩 윈도우 이내에서는 Gemma 4가 마스크 없이 attention을 수행하므로 마스크 검사가 일어나지 않았다. 1024 토큰을 넘는 프롬프트만 크래시했다.
- Metal의 패치되지 않은 표는 bf16 + float32를 float32로 promote하므로 CUDA에서만 발생했다.

`--draft-kind mtp` 사용 여부와 #2330 이전과 이후 모두에서 재현되므로 #2330이 원인이 아니다. 기본 FP16 KV는 abort되지 않았다. 이 문제가 #1793의 마지막 미완료 기준을 막고 있었다.

## 기술적 선택과 그 이유

- `mlx_cxx_internal.h`의 공용 `sdpa_mask_for_inputs(mask, q, k, v)`가 `fast_scaled_dot_product_attention`, `fast_scaled_dot_product_attention_with_sinks`, `fused_metal4_attention`에 있던 "q의 dtype으로 캐스팅" 블록 세 개를 대체한다. Bool과 정수 마스크는 select 의미를 유지해야 하므로(additive 1.0/0.0 오프셋이 되면 안 됨) 그대로 통과시킨다. Floating 마스크는 `result_type(q, k, v)`로 캐스팅한다. SDPA는 받아들인 마스크를 스스로 attention dtype으로 캐스팅하므로, q, k, v의 dtype이 같으면 기존과 같은 캐스트이고, 다르면 q의 dtype을 거치는 불필요한 반올림만 사라진다.
- 브리지에서 고친 이유: 마스크 dtype을 MLX의 계약에 맞추는 일을 이미 담당하는 유일한 지점이고, 그 목적이 "must promote to output type"을 만족시키는 것이었다. 목표 타입을 잘못 골랐을 뿐이다.
- Sinks: SDPA는 attention sinks에도 같은 promotion 규칙을 적용하므로, sinks 브리지도 같은 헬퍼를 거친다(리뷰 지적 사항). `gpt-oss-20b-mxfp4-q4`의 GPT-OSS greedy 출력은 변경 전후 바이트 단위로 동일하다.
- 쿼리 청크 경로의 `mask_in_score_dtype`은 이제 q, k, v를 받고, K와 V가 q와 같은 dtype일 때만 q의 dtype으로 캐스팅한다. dtype이 섞여 있으면 마스크를 건드리지 않고 브리지가 캐스팅한다.
- 기각: Gemma 4에서 K/V를 bf16으로 되돌려 캐스팅. 한 모델만 고치고, 레이어마다 K/V 복사본이 생기며, 모든 곳에서 turbo4 수치가 바뀐다.
- 기각: turbo 캐시가 입력 dtype을 반환하도록 변경. fused turbo 커널이 의존하는 캐시의 fp16 계약이 바뀐다.

트레이드오프: 수정이 C++ 브리지에 있으므로, 브리지 밖에서 직접 SDPA를 호출하는 Rust 코드는 혜택을 받지 못한다. 현재는 그런 코드가 없다.

## 변경 내용

- `src/lib/mlxcel-core/cpp/mlx_cxx_internal.h`: `sdpa_mask_for_inputs`.
- `src/lib/mlxcel-core/cpp/mlx_cxx_bridge.cpp`: 두 SDPA 브리지가 헬퍼를 사용하고, sinks도 헬퍼를 거친다.
- `src/lib/mlxcel-core/cpp/mlx_cxx_ext.cpp`, `mlx_cxx_bridge.h`: Metal 4 브리지가 헬퍼를 사용. 주석 갱신.
- `src/lib/mlxcel-core/src/layers.rs`: `mask_in_score_dtype(mask, q, k, v)`와 호출처 두 곳. 테스트 `additive_mask_with_bf16_query_over_fp16_kv_is_accepted`.
- `src/models/gemma4.rs`: 테스트 `turbo4_past_window_sliding_mask_is_accepted_by_sdpa`.
- `CHANGELOG.md`: Fixed 항목.

합계: 파일 7개, +229 / -39.

## 동작 변경

q, k, v의 dtype이 같으면 달라지는 것이 없다. fp16 TurboQuant KV 캐시 위의 bf16 모델은 CUDA에서 abort되는 대신 마스크가 있는 attention이 실행된다. Metal에서는 마스크가 bf16 대신 float32로 SDPA에 전달되며, 어차피 SDPA가 float32로 promote했을 값이다.

## 검증

- 새 테스트 `layers::tests::additive_mask_with_bf16_query_over_fp16_kv_is_accepted`: fp16 K/V 위의 bf16 q에 슬라이딩 윈도우 causal f32 마스크를 적용해, 직접 dispatch와 쿼리 청크 경로를 float32 attention과 비교한다(허용 오차 1e-3). dtype이 섞인 경우 `mask_in_score_dtype`이 `None`을 반환하는지도 확인한다.
- 새 테스트 `gemma4_unified_mask_tests::turbo4_past_window_sliding_mask_is_accepted_by_sdpa`: bf16 K/V를 넣은 Turbo4Asym `RotatingKVCache`, 윈도우를 넘은 `create_sliding_window_prefill_mask`, `trim_mask_to_keys`, `attention_from_ptr`를 float32 attention과 비교한다(허용 오차 1e-2). fixture가 실제로 fp16 K/V 위의 bf16 q이고 윈도우를 넘는지 assert한다.
- Revert 확인: 브리지 변경을 되돌리면 두 테스트 모두 `Mask type must promote to output type float32`로 abort된다. 변경을 적용하면 통과한다.
- `layers::tests`(100)와 `models::gemma4`(120)가 통과한다. Clippy(`-p mlxcel-core --lib --tests`, 루트 `--lib --tests`), fmt, `scripts/ci` 검사도 깨끗하다.
- 실제 체크포인트: GB10, `~/models/mlx/gemma-4-26b-a4b-it-4bit`, `mlxcel-server`, 1987 토큰 프롬프트, temperature 0으로 300 토큰, `MLXCEL_SDPA_DETERMINISTIC=1`. main에서는 `--kv-cache-mode fp16+turbo4`가 abort된다. 수정 후에는 드래프터 없이, 그리고 `--model-draft ~/models/mlx/gemma-4-26b-a4b-it-assistant-bf16 --draft-kind mtp`(`MLXCEL_MTP_ALLOW_INEXACT=1`)로도 완료된다. turbo4 출력은 일관되고 FP16 KV 출력과 처음 287자까지 일치한 뒤 표현이 갈라지며, 이는 4비트 V 양자화에서 예상되는 결과이다.
- #1793 재확인(#2330을 되돌린 arm, 같은 설정, greedy 400 토큰, verify 약 143 라운드, 어느 PR에도 없는 임시 진단 사용). 수정 arm의 turbo 레이어: verify 블록 호출 3124회, 모두 키 1027개에 대한 1027열 마스크이며 discard 없음. 되돌린 arm의 turbo 레이어: 호출 3080회가 키 1024개를 반환해 1027열 마스크와 맞지 않았고, `trim_mask_to_keys`가 이를 경고 없이 잘라 낸다. 따라서 윈도우를 넘으면 #2330 이전의 결함은 경고가 아니라 조용한 오른쪽 자르기로 나타난다. 두 arm 모두 `mask shorter than key length` 경고를 정확히 한 번 남기며, 모든 discard(수정 393회, 되돌림 387회)는 FP16 저장을 유지하고 MTP rollback 버퍼를 가진 슬라이딩 레이어 세 개에서 나온다. 이는 문서화된 정상적인 buffered-verify 경우이다. 같은 경고는 기본 FP16 KV와 MTP 조합에서도 나온다. 따라서 문구 그대로의 기준("경고 없음")은 어느 arm에서도 충족되지 않지만, 경고의 원인은 turbo4 append가 아니다. 결과는 #1793에 기록했다.

## 미검증 및 후속 조치

- CUDA 청크 게이트(`cuda_sdpa_materializes_scores`, `materializing_sdpa_query_chunk`)는 flash 적합성과 score 바이트를 q의 dtype으로 판단하므로, float32로 실행되는 bf16/fp16 혼합 attention을 과소평가한다.
- softcap composite는 여전히 마스크를 q의 dtype으로 캐스팅하고 v의 dtype을 반환한다.
- CUDA `dtype.cpp` 오버레이의 헤더 주석은 bf16 + fp16이 bf16이라고 하지만, 실제 표는 float32를 준다.
- FP16 KV arm과의 greedy 일치는 처음 287자까지만 확인되었다. 4비트 V에서는 그 이상의 일치를 기대하지 않는다.
