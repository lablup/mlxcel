# 기술 보고서: PR #1623 - perf(qwen3_moe): batch the decode forward instead of per-row forwards

**작성일**: 2026-09-04
**작성자**: mlxcel maintainers
**리뷰어**: implementation review cycle
**상태**: 완료 (M1 Ultra에서 원인 규명과 측정 완료. M5 Max 재측정은 해당 호스트에서 대기)
**언어**: Rust
**위험도**: Medium (한 계열의 디코드 경로를 건드림. B=1은 구조적으로 기존 단일 시퀀스 그래프로 넘김)

---

## 요약

이슈 #1616은 배치 서빙 스케일링 격차의 원인을 `n_tokens > 1`에서 fused MoE 커널이 거절되고 `gather_qmm`으로 떨어지는 것이라고 봤다. 프로파일링 결과 원인은 달랐고, 발행된 가설은 반증됐다.

`supports_batching()`이 true라고 해서 그 계열이 틱마다 forward를 한 번 도는 건 아니다. `forward_batched()`를 오버라이드한 계열만 그렇게 한다. `Qwen3MoeModel`은 오버라이드한 적이 없어서 배치 디코드가 `LanguageModel` 기본 구현을 탔다. 단일 시퀀스 `forward`를 행마다 한 번씩 돌고 함께 평가하는 방식이다. 그 행 하나하나는 라우팅된 토큰이 정확히 하나뿐이라, fused 커널은 모든 행에서 실제로 실행되고 있었다. 토큰 수 게이트는 애초에 장애물이 아니었다.

수정은 이 계열에 진짜 배치 forward를 주는 것이다. M1 Ultra에서 B=4 집계 처리량이 10.3%, 요청당 디코드가 13.8% 올랐고, B=1은 바이트 단위로 동일하며, 밀집 대조군 둘 다 실행 간 변동 범위 안에 남았다.

---

## 1. 문제 정의

### 1.1 발행된 가설

`src/models/qwen3_moe.rs`는 fused MoE 커널을 `array_shape(&x_flat)[0] == 1`로 막고, 모델 계열 전반의 프로덕션 호출 지점 16곳이 같은 게이트를 갖고 있다. 스케줄러는 슬롯이 둘 이상이면 곧바로 배치 차원을 넘긴다. 이슈는 그러므로 B>=2에서 48개 층 전부가 매 틱 `SwitchGLU::forward`와 `gather_qmm`으로 조용히 떨어진다고 결론짓고, 커널에 토큰 차원을 주자고 제안했다.

### 1.2 프로파일이 실제로 찾은 것

`MLXCEL_PROFILE_BLOCKS=1`로 보면 B=4 틱은 직렬화된 단일 토큰 그래프 네 개였다(128틱에 블록 요약 518개). 그리고 각각이 여전히 fused 커널을 실행하고 있었다. 이 GPU에서 그래프들은 30% 정도만 겹친다. 틱당 B=1이 12.3 ms, B=4가 33.6 ms다.

기전은 `supports_batching()`과 `forward_batched()`가 별개라는 데 있다. 앞쪽은 계열을 배치 스케줄러 경로에 넣어 주고, 뒤쪽이 그걸 단일 forward로 만든다. Llama 3, Llama 4, Qwen 3, Qwen 3.5, Gemma 3, Helium, Muse Glimmer가 뒤쪽을 오버라이드한다. Qwen3-MoE를 포함한 나머지 배치 계열은 기본 구현을 탔고, 그 집계 스케일링은 독립적인 단일 토큰 그래프가 겹치는 데서만 나온다.

`MLXCEL_FUSED_MOE=0` 대조가 직접 증거다. B=4 집계가 93.3에서 82.9 tok/s로 움직였다. 이슈 가정대로 B=4에서 fused 커널이 이미 거절되고 있었다면, 그걸 끄는 일이 이 숫자를 조금도 움직일 수 없다.

### 1.3 제안된 커널을 만들지 않은 이유

실제 layer-0 전문가 평면을 대상으로 한 연산 수준 마이크로벤치가 남은 두 제안을 정리했다.

- **토큰별 fused 실행은 선형으로 증가한다.** n=1/2/4/8에서 48개 층당 3.9, 7.5, 14.7, 29.2 ms다. 배치로 묶어서 상쇄할 실행당 오버헤드가 없다.
- **전문가 집합이 같든 완전히 겹치지 않든 모든 n에서 비용이 같다.** 즉 전문가 평면 트래픽이 한계 요인이 아니고, 이슈의 유력 가설이던 코호트 간 전문가 id 중복 제거는 아무것도 벌지 못한다.
- **제안된 배치 커널의 프로토타입이 진다.** grid z를 `k * n_tokens`로 두고 토큰별 실행과 비트 단위로 동일함을 확인해 만들었는데, n=4부터 `gather_qmm`보다 느리다(11.1 ms 대 9.3~9.7 ms).

이슈의 합격 기준은 프로파일이 뒷받침하지 않을 때 속도 향상 대신 문서화된 설명을 명시적으로 허용한다. 이번이 그 경우이고, 조용히 건너뛰는 대신 기록했다.

---

## 2. 기술적 결정

### 2.1 밀집 Qwen3 이식본을 따라간 배치 forward

`Attention::apply_rope_batched`는 `apply_rope`와 같은 회전을 적용하되 텐서 전체에 offset 하나가 아니라 배치 행마다 캐시 offset 하나를 쓴다. 주파수 테이블 방식은 테이블 런처를 그대로 타고, YaRN 크기 곱도 단일 시퀀스 경로와 똑같이 적용된다.

`Attention::forward_split_attention`은 배치 전체의 fused QKV 투영을 `[B, T, proj_dim]` 텐서 셋으로 받아, 배치 상태에서 Q/K RMSNorm과 RoPE를 적용한 다음, 행마다 자기 캐시를 소유하므로 KV 캐시 갱신과 SDPA는 시퀀스별로 돌리고 다시 이어 붙인다. 이 계열은 `supports_paged_decode_backend()`에 참여하지 않으므로 paged 풀 빠른 경로는 재현하지 않았다.

`DecoderLayer::forward_batched`는 norm과 fused QKV 투영, 출력 투영, MoE 블록을 `[B, T, hidden]` 위에서 한 번씩 돌린다. 그래서 B=2부터 전문가가 다중 토큰 `gather_qmm` 체인을 탄다. `Qwen3MoeModel::forward_batched_impl`은 임베딩과 최종 norm과 LM head를 배치로 처리한다.

### 2.2 B=1은 측정이 아니라 구조로 불변이다

`LanguageModel::forward_batched` 오버라이드는 캐시 개수로 분기한다. 0이면 트레이트 기본값의 빈 로짓을 돌려주고, 정확히 1이면 기존 단일 시퀀스 `Qwen3MoeModel::forward`(fused 커널 포함)에 위임하며, 2 이상만 배치 구현에 닿는다. 그래서 B=1은 변경 전과 같은 그래프를 그대로 탄다. 바이트 동일성이 운 좋은 측정 결과가 아니라 구조적 성질인 이유다.

### 2.3 프로파일링 훅 자체가 틀려 있었다

`MLXCEL_PROFILE_QWEN3_MOE_DETAIL=1`은 항상 gather 경로를 돌리면서 어떤 경로를 잰 건지 말하지 않았다. 그래서 단일 토큰 추적이 프로덕션 단계가 실행한 적 없는 커널을 그 결과로 돌리고 있었다. 이제 `SparseMoeBlock::forward_profiled`가 프로덕션 디스패치를 그대로 따라가고 `path`와 `tokens`를 담은 `MoeProfile`을 돌려주므로, 추적에서 배치 단계와 단일 토큰 단계를 구별할 수 있다. 프로덕션 코드가 타지 않는 경로를 보고하는 프로파일링 훅은 훅이 없는 것보다 나쁘다. 자신 있게 틀린 귀속을 만들어 내기 때문이다.

---

## 3. 검증

Apple M1 Ultra 128 GB, Metal, mlxcel 0.7.0-beta.1, MLX 핀 `9a795735`, `--parallel 4 --max-batch-prefill 4`로 띄운 서버에 `scripts/bench_serving_concurrency.py --prompt-tokens 512 --max-tokens 128`.

| 모델 | B=4 변경 전 | B=4 변경 후 | 스케일링 전 | 스케일링 후 |
|---|---|---|---|---|
| qwen3-30b-a3b-4bit | 93.3 | **102.9** | 1.72x | **1.86x** |
| llama-3.1-8b-4bit (대조군) | 120.5 | 118.9 | 2.11x | 2.09x |
| qwen2.5-0.5b-bf16 (대조군) | 367.5 | 369.5 | 1.70x | 1.77x |

B=4에서 MoE 요청당 디코드가 29.8에서 33.9 tok/s로 올랐다. 밀집 대조군 둘 다 실행 간 변동 범위 안에 있고, 이것이 두 측정 사이에 호스트 전체가 흔들린 게 아님을 배제해 준다.

### 3.1 정확성

`qwen3-30b-a3b-4bit`에 대한 greedy `mlxcel generate -p Hello -n 128 --temp 0` 출력이 변경 전후로 바이트 단위 동일하다. B=4에서 동시 greedy 256토큰 요청 넷이 비어 있지 않은 reasoning 내용으로 서로 동일한 행을 돌려주고, 단일 시퀀스 `MLXCEL_FUSED_MOE=0` 출력과 글자 단위로 일치한다.

마지막 일치가 이번 동작 변화의 정확한 진술이다. B=2부터 전문가가 `gather_qmm`을 타므로 B=4 출력이 이제 fused 경로가 아니라 gather 경로와 같아진다. 변경 전에는 B>1이 행마다 단일 시퀀스 경로를 돌았으므로 B=1과 정확히 일치했다. 배치 크기를 가로지르는 비트 단위 동일성은 여기서 보장이었던 적이 없고(#203), 이 변경은 `forward_batched`를 이미 오버라이드한 다른 모든 계열과 이 계열을 나란히 맞춘다.

`cargo test --workspace --profile test-fast --features metal,accelerate`: 10,510 통과, 0 실패. 새 테스트는 행별 RoPE offset을 가진 합성 3전문가 모델에서 배치와 순차가 같음을, 한 행 위임을, 빈 배치를 고정한다.

### 3.2 폐기해야 했던 측정

앞선 after 측정은 B=4에서 103.7 tok/s를 냈다. 최종 커밋에서 `cargo build --release`를 돌리자 재컴파일이 일어났고, 그 측정이 브랜치 마지막 수정 이전 바이너리에서 나온 것임이 드러났다. 그 측정은 버리고 위 내용은 전부 최종 커밋에서 다시 쟀다. 결론은 그대로이고 숫자는 1% 남짓 움직인다. 워크스페이스 게이트도 같은 이유로 다시 돌렸다. 첫 게이트는 트리가 아직 움직이는 동안 시작됐기 때문이다.

---

## 4. 변경 요약

| 항목 | 값 |
|---|---|
| 변경 파일 (코드 PR) | 3 |
| 추가 줄 | 605 |
| 삭제 줄 | 24 |

- `src/models/qwen3_moe.rs`: `apply_rope_batched`, `forward_split_attention`, `DecoderLayer::forward_batched`, `Qwen3MoeModel::forward_batched_impl`, `LanguageModel` 오버라이드, 그리고 교정된 `forward_profiled`.
- `src/models/qwen3_moe_tests.rs`: 배치와 순차의 동등성, 한 행 위임, 빈 배치.
- `docs/CONTINUOUS_BATCHING.md`: 어떤 계열이 `forward_batched`를 오버라이드하는지와 나머지에서 기본 구현이 무엇을 하는지 명시.

### 따로 반영한 부분

`benchmarks/metal_m1ultra_batch_2026-09-04.csv`와 `docs/benchmark_results/moe-batched-decode-m1ultra-2026-09-04.md`, 그리고 M1 Ultra 배치 서빙 절과 M5 Max 쪽 포인터는 `bench/0.7.0-refresh`(PR #1617)에 커밋 `e75fa0e44`로 올렸다. CSV의 `mlxcel_commit` 열이 두 측정을 구분한다. 이 분리 때문에 코드 PR이 머지되어 이슈가 닫혀도 문서 절반은 아직 미머지로 남는다.

### 관련 이슈

Closes #1616. 관련: #268(이번 프로파일이 내내 도달 가능했음을 밝힌 B=1 fused 커널), #725, #628, #632, #203(배치 크기를 가로지르는 비트 동일성은 보장이 아니다).

---

## 5. 후속 작업

- 같은 `== 1` 게이트를 가진 나머지 15개 MoE 계열은 여전히 행별 대체 경로를 탄다. 각자 같은 전제 확인을 거쳐 자기 `forward_batched`가 필요하고, `switch_layers::SwitchGLU`를 공유하는 계열은 어텐션만 배치가 되면 다중 토큰 `gather_qmm` 경로를 거저 얻는다.
- 밀집 백엔드 계열의 프롬프트 캐시 채택이 일회성이라, 동일 프롬프트 N개가 동시에 오면 N-1번의 차가운 prefill을 낸다. paged 백엔드의 clone 채택이 해법 패턴이다. 이슈 표의 TTFT 증가가 실제로 재고 있는 것은 틱 길이가 아니라 이것이다.
- 이 호스트의 밀집 배치 디코드는 M5 Max의 3.25x에 비해 2.09x에 그친다. 시퀀스별 어텐션 루프가 유력한 디스패치 비용이고 자체 프로파일이 필요하다.
- 이 사다리의 M5 Max 재측정은 해당 호스트에서 대기 중이다.

### 옮겨갈 만한 교훈

이슈는 게이트를 지목했고, 그 게이트는 실재했고 도달 가능했으며 무관했다. 한 가지처럼 들리는 두 능력이 사실은 별개였다. 어떤 계열이 배치 스케줄러 경로에 들어가 있으면서도 배치 forward를 한 번도 돌지 않을 수 있다. 이걸 판가름한 확인은 환경 변수 하나로 끝났다. fused 커널을 껐더니 가설상 건드릴 수 없어야 할 숫자가 움직였기 때문이다. 어떤 가설이 특정 레버에 효과가 없다고 예측한다면, 대개 그 레버를 당겨 보는 것이 가장 싼 검증이다.
