# PR #2611: Gemma 4 네이티브 배치 디코드

**작성일**: 2026-10-26
**상태**: Metal에서 완료. ROCm과 CUDA 성능은 측정하지 않음
**언어**: Rust, Markdown
**위험도**: 중간

## 요약

PR #2611(이슈 #2546)은 Gemma 4 서버 디코드의 행별 실행을 식별된 단일 토큰 행을 위한 하나의 네이티브 배치 Transformer 그래프로 바꾼다. 임베딩, projection, normalization, dense 및 MoE feed-forward, per-layer input, LM head는 `[B, 1, hidden]` 전체에서 실행하고, attention과 full/sliding 혼합 캐시 갱신만 행별로 유지한다. Apple M3 Ultra에서 동시성 4의 aggregate output은 Gemma 4 12B가 51.4에서 102.7 tok/s로, Gemma 4 26B A4B가 154.2에서 189.8 tok/s로 증가했다. 동시성 1 성능 차이는 기준 대비 1.5% 이내였다.

공유 sequence-state 경계도 함께 강화했다. 중복 `SequenceId`는 mutable cache slice를 재구성하기 전에 거부하고, 잘못된 model-owned 배치가 살아 있는 상태를 제거하지 않게 했다. 또한 `T == 1`에서 생기는 MLX stride 모호성을 피하기 위해 proportional RoPE는 여러 행의 offset이 같아도 행별로 평가한다.

## 1. 문제 정의

Gemma 4는 batching 지원을 표시했지만 VLM wrapper가 디코드 배치를 행별로 잘라 매 행마다 전체 text model을 호출했다. 따라서 각 행이 weight를 다시 읽고 약 1,400개의 디코드 커널을 반복했으며, 동시성이 늘어도 aggregate throughput은 거의 증가하지 않았다. Gemma 4는 sliding-window/full attention, proportional RoPE, KV-shared layer, sparse expert, token 기반 per-layer input을 함께 쓰므로 단순히 입력 토큰을 쌓는 것만으로는 부족했다. 독립 offset과 서로 다른 cache state를 보존해야 했다.

기존 ROCm 측정에서 26B A4B 체크포인트의 aggregate throughput은 동시성 1부터 8까지 약 74–79 tok/s에 머물렀다. 네이티브 경로가 없으면 클라이언트를 추가할수록 단일 요청 디코드 속도가 나뉘고 지연만 늘었다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| 네이티브 text decode | `Gemma4Wrapper`가 sequence ID가 있는 `[B, 1]` 호출을 골라 새 batched decoder 경로로 보낸다. B=1은 기존 single-sequence graph를 직접 호출하고, 빈 배치는 빈 logits를 반환한다. |
| Transformer 연산 | token embedding, Q/K/V 및 output projection, norm, dense 또는 SwitchGeGLU/MoE block, router, per-layer projection, tied LM head를 배치 상태로 유지한다. 행별 logits는 한 번의 `concatenate_many`로 합친다. |
| Attention과 cache | 각 행은 고유한 full 또는 rotating cache와 논리 위치를 유지한다. KV-shared layer는 원본 layer의 행별 K/V와 offset을 재사용한다. standard/proportional RoPE는 서로 다른 행별 offset을 받는다. |
| VLM과 Unified wrapper | 식별된 text-only decode는 네이티브 text 경로로 위임한다. prefill, mask, ID 누락, 대기 중인 image/audio per-layer input은 기존 wrapper fallback을 그대로 사용한다. |
| Sequence-state 안전성 | `CachePool`은 mutable slice를 만들기 전에 중복 ID를 거부한다. `ModelOwnedSequenceState`는 entry를 제거하기 전에 ID 전체를 검증하므로 중복 또는 누락 ID가 alias를 만들거나 살아 있는 상태 일부를 잃게 할 수 없다. |
| RoPE 정확성 | offset이 모두 같아도 여러 행의 proportional RoPE를 행별로 계산한다. batch와 sequence stride가 같은 transpose된 `[B, H, 1, D]` tensor에서 batch 차원을 sequence 위치로 해석하는 문제를 피한다. |
| 분류와 문서 | Gemma 4 text, VLM, Unified model을 네이티브 batched-decode family로 분류하고 문서화했다. |

## 3. 기술적 선택과 그 이유

### 3.1 Weight 비중이 큰 graph는 배치하고 attention은 행별로 유지

Paged-attention backend를 새로 만들지 않았다. 선택한 경로는 기존 `Cache` variant와 attention 의미를 보존하면서 weight read와 projection launch를 즉시 분담한다. 각 layer는 전체 배치를 한 번 project한 뒤 Q/K/V를 행별로 잘라 cache append와 attention을 수행하고 결과를 합친다. 행별 launch overhead 일부는 남지만 더 큰 storage migration 없이 반복 weight 비용을 직접 제거한다.

### 3.2 B=1과 조건부 wrapper 경로를 정확히 보존

한 행인 경우 새 배치 구현 대신 기존 forward method를 호출한다. VLM과 Unified wrapper는 ID가 있고 mask가 없으며 multimodal per-layer input이 대기하지 않는 단일 토큰 text decode만 위임한다. 따라서 prefill, image/audio state 소비, 익명 model-owned state, 잘못되었거나 대상이 아닌 shape는 기존 경로를 유지한다.

### 3.3 B > 1 proportional RoPE는 항상 행별 적용

초기 최적화는 offset이 같은 배치를 하나의 scalar RoPE 호출로 합쳤다. 최종 리뷰에서 batch와 sequence stride가 같은 transpose decode tensor를 MLX가 연속 위치처럼 처리하여 `b`번 행을 `offset + b`로 회전할 수 있음을 찾았다. 이제 offset이 같아도 모든 multi-row 배치를 자른다. 행별 RoPE 연산은 늘지만 production layout의 위치 정확성을 보장한다.

### 3.4 Unsafe borrow와 destructive extraction 전에 ID 검증

`CachePool::get_batch_caches_mut`는 raw pointer에서 mutable slice를 재구성하므로 ID가 반복되면 겹치는 mutable reference가 생겨 Rust aliasing contract를 위반한다. 이제 pointer를 모으기 전에 중복을 거부한다. Model-owned state container도 entry를 하나라도 제거하기 전에 ID 집합 전체를 검사한다. 따라서 오류가 앞선 행의 상태를 map 밖에 남길 수 없다.

## 4. 검증과 측정 결과

### 4.1 기능 범위

체크포인트가 필요 없는 regression은 prompt 길이가 다른 세 행으로 sliding window wrap, proportional full attention, KV-shared layer, MoE expert, token 기반 per-layer input, B=1 정확한 dispatch, 빈 배치 contract를 다룬다. 별도 테스트는 익명 fallback, 대기 중인 multimodal input, native classification, 중복 pool ID, 상태를 잃지 않는 model-owned 중복/누락 ID 오류, transpose decode layout의 동일 offset proportional RoPE를 검증한다.

전체 workspace 실행 결과는 13,811 passed, 10 failed, 441 ignored였다. `01047d75` clean baseline을 별도로 실행하니 실패 10개가 값과 소스 행까지 똑같이 재현되었다. 따라서 이 실패는 이 PR의 regression이 아니라 기존 cross-backend BF16 oracle 실패다. 중복 ID `CachePool` regression이 통과했고, 새 invalid-ID state-retention case를 포함한 model-owned state 테스트 7개도 모두 통과했다. Sequence-state 수정 뒤 native mixed-cache/MoE/per-layer-input/B=1/empty regression도 통과했다. Formatting과 clippy는 구현 revision `a9fcef02`에서 통과했으며, proportional-RoPE regression과 최종 release build는 리뷰 변경이 반영된 뒤 상위 finalization workflow가 수행한다.

### 4.2 Metal 서버 throughput

측정 환경은 Apple M3 Ultra 512 GiB unified memory, `metal,accelerate` release build, 445-token prompt와 요청 output 128 token이다. 기준 revision은 `01047d75`, 측정한 구현 revision은 이후 sequence-state 및 RoPE hardening 전의 `a9fcef02`다.

| 체크포인트 | Revision | C=1 | C=2 | C=4 | C=8 |
|---|---|---:|---:|---:|---:|
| Gemma 4 12B dense / Unified | Baseline | 50.5 | 51.3 | 51.4 | 49.5 |
| Gemma 4 12B dense / Unified | Native batch | 49.9 | 78.3 | 102.7 | 110.8 |
| Gemma 4 26B A4B MoE | Baseline | 95.8 | 127.8 | 154.2 | 127.1 |
| Gemma 4 26B A4B MoE | Native batch | 94.4 | 145.0 | 189.8 | 220.7 |

C=4에서 네이티브 경로는 자체 C=1 aggregate rate 대비 12B가 2.06배, 26B가 2.01배에 도달했다. 기존 C=4 경로와 비교하면 각각 2.00배와 1.23배다. 두 체크포인트 모두 greedy `Hello` 생성 결과가 byte-identical했고 생성 token 9개 뒤 EOS에서 종료했다.

### 4.3 수치 동작

Decode width 1의 teacher-forced B=4 대 B=1 trace에서 12B는 top-1 disagreement 19/256, decided-position disagreement 2/152였고 26B는 각각 25/256과 3/157이었다. 따라서 batched inference를 bit-exact라고 설명하면 안 된다. `MLXCEL_QMV_WIDE=0`으로 실행한 12B trace는 256개 위치 전부 동일해져 기본 차이가 cache offset이 아니라 Metal의 width-dependent quantized projection kernel에서 비롯됨을 보였다. 26B control에는 top-1 18/256과 decided-position 2/157이 남았으며, single-token fused expert와 batched `gather_qmm`의 reduction 차이와 일치한다.

기존 12B Unified trace는 trait-default batch method가 식별된 prefill 뒤 sequence ID를 버렸으므로 identified-state baseline으로 유효하지 않다. B=1 보존은 단일 요청 생성 직접 비교와 정확한 one-row dispatch regression으로 확인한다.

## 5. 학습 포인트

- 모델이 batching 지원을 표시해도 실제로는 행마다 전체 graph를 실행할 수 있다. Decode classification은 batched input 수용 여부가 아니라 실제 실행 graph를 설명해야 한다.
- 모든 연산을 배치하지 않아도 네이티브 batching의 이점을 얻을 수 있다. Cache mutation과 attention을 행별로 유지하면 복잡한 cache 의미를 보존하면서 decoder의 weight 비중이 큰 대부분을 분담할 수 있다.
- Tensor shape만으로 backend 동작을 판단할 수 없다. Transpose된 singleton dimension이 batch dimension과 stride를 공유하면 그럴듯한 동일-offset shortcut도 잘못될 수 있다.
- Raw pointer에서 여러 mutable reference를 재구성하는 API는 borrow 전에 중복 논리 key를 거부해야 한다. Pointer를 모은 뒤 검증하면 이미 늦다.

## 6. 후속 조치와 제한 사항

- 해당 host를 사용할 수 있을 때 이슈의 MI300X ROCm ladder를 다시 실행하고 CUDA GB10 결과를 수집한다. Metal 결과는 다른 backend의 성능이나 수치 동작을 입증하지 않는다.
- Tensor-parallel Gemma 4와 paged decode backend는 범위 밖이다.
- Attention 결과는 계속 행별로 계산한 뒤 한 번 합친다. 추가 batching은 이득을 가정하지 말고 profiling을 근거로 해야 한다.
- Metal quantized kernel이나 MoE routing kernel이 바뀔 때 decided-position divergence를 계속 관찰한다.

## 참고 자료

- 이슈 #2546: Gemma 4 decode가 행마다 하나의 forward를 실행하는 문제
- PR #2611: 활성 행 전체에서 Gemma 4 decode batching
- `docs/benchmark_results/gemma4-native-batched-decode-2026-10-26.md`: 전체 protocol, trace 해석, raw comparison 맥락
- `docs/CONTINUOUS_BATCHING.md`: 서버 decode 분류와 운영 동작
