# PR #2646: Metal 프리필에 네이티브 window SDPA 적용

**작성일**: 2026-10-27
**상태**: 완료
**언어**: Rust, C++
**위험도**: 중간

## 요약

키 길이가 sliding window를 넘는 causal multi-query 프리필은 이제 Metal에서 MLX가 해당 shape에 fused 커널이 있다고 답하면 전체 폭 window 마스크를 만드는 대신 upstream MLX의 네이티브 `window_size` 인자로 실행된다(issue #2621). 이 경로는 기본으로 켜져 있다(`MLXCEL_NATIVE_WINDOW_SDPA=auto`). 이 결정은 측정 전에 정한 기준에 근거하는데, 그 기준을 넘은 shape는 정확히 3.0%를 기록한 하나뿐이다. bf16 Gemma 4의 greedy 출력은 근소한 차이로 갈리는 위치에서 바뀌며, `MLXCEL_NATIVE_WINDOW_SDPA=0`이면 이전 경로로 돌아간다.

## 1. 문제 정의

#2614의 MLX 핀 업데이트로 window를 인식하는 Metal vector/matrix 커널(ml-explore/mlx#4734)이 들어왔지만, mlxcel의 모든 호출 지점은 `window_size = -1`을 유지했다. window를 넘는 프리필에서 `causal_attention`은 causal과 window 대역을 함께 담은 `[q_len, k_len]` 마스크를 만들어 배열 마스크로 넘겼다. 여기에는 두 가지 비용이 있었다. 첫째는 마스크 자체로, Gemma 4 31B 형상에서 8192행일 때 호출 피크 메모리가 5376 MiB였다. 둘째로, M5 이전 GPU에서 MLX는 head dim 256(캐시된 sliding-window 체크포인트 전부)의 배열 마스크 호출을 fuse하지 않으므로 프리필 전체가 graph fallback으로 돌았다.

프리필 의미론은 미묘하고 이전에도 회귀가 있었다. window를 넘는 multi-query 프리필은 모든 키를 유지해야 하며, 마지막 window로 자르면 앞쪽 쿼리가 고립된다(#401/#408). rotating cache의 continuation은 유지된 이전 키와 새 키 전부를 돌려주므로, 쿼리 행은 절대 오프셋이 아니라 보이는 키의 끝부분에 맞춰야 한다(#678).

## 2. 기술적 선택과 그 이유

**경로 결정은 한 곳에서, 메타데이터는 명시적으로.** 경로 결정은 `mlxcel_core::window_sdpa`가 맡는다. 입력은 `WindowSdpaShape`(q/k/v의 shape와 dtype)와 `WindowSdpaExtras`(softcap, sink, 추가 마스크)이며, 마스크의 shape로 의미를 추론하지 않는다. 네이티브 진입점은 softcap 0인 `causal_attention`의 window 초과 분기에서만 닿는데, 이 분기에는 마스크 인자도 sink도 없다. 나머지 경우(padding/segment/head별 마스크, `q_len <= 8`, `q_len > k_len`, 빈 배치, 지원하지 않는 head dim이나 dtype, Metal이 아닌 모든 백엔드)는 기존 경로를 쓴다. CUDA와 ROCm에서는 양수 window가 graph fallback을 강제하기 때문에 이 구분이 중요하다.

**MLX 규칙을 베끼지 않고 MLX에 묻는다.** `auto`는 고정된 MLX의 `ScaledDotProductAttention::use_fallback`이 fused 커널을 쓴다고 답할 때만 네이티브 경로를 탄다. `mlx_cxx_window_sdpa.cpp`의 `sdpa_uses_fused_kernel`이 호출 shape의 lazy zero 배열로 그 답을 얻는다. 디스패치 표를 하드코딩하면 다음 핀 업데이트 때 어긋난다. 측정 결과가 이 게이트가 필요한 이유를 보여준다. graph-fallback shape(head dim 256 continuation 청크)에서는 네이티브 쪽이 같거나 최대 4% 느렸다. `1`은 정적으로 적격인 모든 곳에서 네이티브를 강제하고, `0`은 롤백이다.

**모델에서 계획하고 레이어에서 확인한다.** Gemma 3(모델, stage 모델, VLM backbone)와 Gemma 4의 텍스트 전용 프리필은 forward마다 한 번, 네이티브 경로가 실행될지를 결정한다. 실행될 경우 sliding 마스크를 만들지 않는다. 레이어가 네이티브를 거절하면 같은 마스크를 다시 만들므로, 계획과 레이어가 어긋나도 시간만 들고 의미는 바뀌지 않는다. 리뷰에서 실제로 이런 불일치가 발견됐다. `fp16+turbo4`에서 캐시는 f16 K/V를 돌려준다. bf16 쿼리와 섞이면 호출이 f32로 승격되는데, f32의 head dim 256은 fuse되지 않으므로 모든 sliding 레이어가 거절하고 마스크를 다시 만들었다. 이제 계획은 `RotatingKVCache::fetched_kv_dtype`을 쓰고, 모든 sliding 캐시가 같은 dtype을 돌려줄 때만 네이티브를 택한다(`sliding_caches_kv_dtype`). `--turbo-boundary-v`가 레이어마다 캐시 모드를 섞기 때문이다.

**rotating 무마스크 분기는 보존하지 않고 고친다.** `RotatingKVCache::attend`는 window를 넘는 무마스크 multi-token 단계를 causal 없이 일반 무마스크 attention으로 처리했다. 이 진입점에 닿는 모델은 Gemma 3뿐이고, Gemma 3의 모든 경로가 이 경우 마스크를 만들었으므로 이 분기는 죽은 코드였다. 이제는 `causal_attention(..., max_size)`를 호출하고, Gemma 3는 계획이 네이티브일 때 이 분기에 의존한다. continuous batching 행은 항상 `q_len == 1`로 들어오고 speculative 검증 블록은 `q_len <= 8`에서 거절되므로, 어느 쪽도 새 분기에 닿지 않는다.

**probe는 싸게, 그리고 죽지 않게.** probe 결과는 전체 shape, dtype, arm을 키로 스레드별로 memoize한다. 장기 실행 서버에서는 청크마다 `k_len`이 바뀌므로 memo는 64개로 제한하고 가득 차면 비운다. cxx 선언상 브리지 함수가 infallible이라 C++ 예외가 나면 프로세스가 종료된다. 그래서 probe는 양수가 아닌 차원에 false를 돌려주고 `std::exception`을 잡는다. false를 돌려받으면 기존 경로가 실행된다.

## 3. 변경 요약

- `src/lib/mlxcel-core/src/window_sdpa.rs`(신규): 모드 파싱(한 번만 읽고, 알 수 없는 값은 경고 한 번), 적격성 판정, 계획, fused 판정 memo, 디스패치 카운터, `MLXCEL_LOG_WINDOW_SDPA` 근거 출력, 네이티브와 기존 경로.
- `src/lib/mlxcel-core/cpp/mlx_cxx_window_sdpa.cpp`(신규)와 `mlx_cxx_bridge.h`, `lib.rs` cxx 블록, `build.rs`: `fast_scaled_dot_product_attention_causal_window`(upstream `sdpa(..., W, "causal")`)와 `sdpa_uses_fused_kernel`.
- `lib.rs::causal_attention`: softcap 0인 window 초과 분기가 `window_sdpa::causal_window_attention`을 거친다. softcap 조합 경로는 그대로다.
- `cache/attend.rs`, `cache.rs`: 위에서 설명한 rotating 무마스크 분기와 `fetched_kv_dtype`.
- `src/models/gemma3.rs`, `gemma3_backbone.rs`, `gemma4.rs`: 네이티브 계획일 때 마스크 생략.
- 테스트: `window_sdpa_tests.rs`(19개).
- 벤치·문서: `examples/window_sdpa_microbench.rs`, `docs/environment-variables.md`, `docs/benchmark_results/native-window-sdpa-m1ultra-2026-10-26*` 아래의 보고서와 원본 CSV·trace 산출물.

## 검증

M1 Ultra, macOS 27.0.1, MLX `7ca40033`.

**정확성.** 허용 오차는 실행 전에 정했다. 네이티브 경로는 f32, f16, bf16과 MHA, GQA에 걸친 8개 shape 케이스에서 기존 경로와 f64 호스트 기준값 모두와 일치한다. 이 케이스에는 W=1, continuation 청크, window보다 긴 쿼리 블록, Gemma D=256 MQA 형상이 포함된다. window 밖 키와 미래 키는 비트 단위로 영향이 없다. rotating cache는 wrap, trim, 무마스크 decode 20스텝까지 확인했다.

**Logit trace.** 경로가 실제로 도는 폭에서 teacher-forced trace(`--decided 2`)를 떴다. Gemma 3 1B(f16)와 Gemma 4 E4B(bf16) 모두 결정된 위치에서 불일치가 0이고, 되돌린 쪽은 자기 자신과 바이트 단위로 같다.

**전체 게이트.** 최종 head에서 `cargo test --workspace --profile test-fast --no-fail-fast` 결과 13896개 통과, 실패 0개다. workspace clippy와 fmt도 깨끗하다.

**성능.** 커널 수준에서는 `auto`가 고르는 shape가 W=512에서 기존 대비 0.50x~0.11x 시간에 끝나고, 호출 피크 메모리는 3.5~8배 줄었다. 전체 모델 수준에서는 다음과 같다.
- Gemma 4 E4B 프리필은 2105 토큰에서 3.0% 빨라졌고(n=6, 범위 비중첩), 더 긴 프롬프트에서는 노이즈 이내다.
- 31B는 n=2에서 같은 방향이다.
- Gemma 3 1B는 노이즈 대역 안이다.
- decode와 프로세스 RSS는 변화가 없다.

커널 이득에 비해 모델 수준 이득이 작은 이유는 두 가지다. 경로가 첫 2048행 프리필 청크에서만 동작하고, attention이 프리필 시간의 일부일 뿐이다.

## 미검증 사항과 후속 조치

- **M5/NAX.** 미실행. 고정된 디스패치는 M5에서 head dim 256을 모든 폭에서 fuse하므로 `auto`가 더 많은 shape에서 동작하며, 따로 측정해야 한다. M5에서는 네이티브 진입점이 `layers::attention`을 거치지 않으므로 `MLXCEL_METAL4_ATTENTION`과 `MLXCEL_LOG_NA_ATTENTION`으로는 제어되지 않는다. `MLXCEL_NATIVE_WINDOW_SDPA=0`으로는 된다.
- **head dim 128 sliding-window 계열(Ministral 3).** 마이크로벤치가 모델 수준에서 큰 이득을 예측하는 곳이다. 체크포인트가 없었고, Ministral 3는 여전히 자체 프리필 마스크를 만들기 때문에 그 마스크가 없을 때만 이 경로에 닿는다.
- **TurboQuant 캐시 모드.** 계획과 레이어 판단은 이제 일치하지만, 어느 쪽도 시간을 재지 않았다.
- **기본값 결정은 shape 하나에 근거한다.** 이 결정은 3.0% 기준을 넘은 shape 하나에 의존한다. 프리필 청크 크기나 MLX 핀이 바뀌면 다시 측정해야 한다. `DEFAULT_PREFILL_CHUNK`를 키우면 동작 구간이 넓어지지만 청크 분할의 메모리 목적(#672)과 맞바꾸게 된다.
- **불안정한 paged 카운터 테스트.** `attend_tests::batched_paged_one_row_is_one_pooled_launch`가 전체 병렬 실행에서 한 번 실패했다. 이 테스트는 프로세스 전역 paged launch 카운터를 읽으며, 테스트 대상 코드는 이 PR이 건드리지 않았다. 재실행에서는 매번 통과했지만 경합 자체는 실재하므로 따로 격리할 가치가 있다.
