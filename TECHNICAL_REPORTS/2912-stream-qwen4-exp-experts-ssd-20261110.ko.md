# 기술 보고서: PR #2912 - feat(moe): stream qwen4_exp routed experts from the SSD

**작성일**: 2026-11-10
**리뷰어**: -
**상태**: 후속 조치 필요
**언어**: Rust
**위험도**: 중간

---

## 요약

`qwen4_exp`의 라우팅 expert를 SSD에 두고, 필요할 때 레이어별 고정 슬롯 아레나로 읽어 실행할 수 있게 했다. 나머지 가중치는 메모리에 상주한다. 라우팅 expert가 Metal working set의 60%를 넘을 때만 켜지므로 모델이 들어가는 머신은 기존 경로를 그대로 쓴다. 512 GB M3 Ultra에서 24 GB와 8 GB 아레나를 강제해 상주 실행과 출력이 일치함을 확인했지만, 실제 목표인 체크포인트보다 메모리가 작은 Mac에서는 실행하지 못했다.

---

## 1. 문제 정의

### 1.1 배경

Qwen3.8-Flash-Next 체크포인트는 대부분의 용량을 MoE 레이어당 512개 라우팅 expert에 쓰고, 토큰 하나는 top-k expert만 사용한다. 모든 expert를 상주시키려면 대부분의 Mac이 가진 것보다 많은 메모리가 필요하다. 이슈 #2882는 상주할 수 없는 환경에서 SSD로 모델을 실행하도록 요청했다.

### 1.2 기존 문제점

- **전부 아니면 전무**: expert가 상주 텐서여서 working set보다 큰 체크포인트는 쓸 만하게 실행할 수 없었다.
- **라우터 결과가 그래프 노드**: 무엇을 읽을지 정하려면 top-k가 호스트에 있어야 하므로, forward 중 레이어마다 `eval`이 필요하다.
- **수치가 배치 구성에 의존**: 상주 sorted kernel은 expert별로 행을 묶고, kernel과 tile 선택은 `E = 512`에서 결정된다. 묶음이 바뀌면 하위 비트가 달라질 수 있다.

### 1.3 위험성

| 위험 | 영향 | 가능성 |
|------|------|--------|
| 오류 없이 상주 실행과 출력이 달라짐 | 높음 | 중간, 첫 버전에서 발생 |
| 대기 중인 명령이 읽는 슬롯을 덮어씀 | 높음 | 낮음, 레이어별 동기화로 방지 |
| 악의적 체크포인트가 검증되지 않은 헤더로 파일 읽기나 할당을 유도 | 높음 | 보안 수정 전 중간 |

---

## 2. 기술적 검토 사항

### 2.1 보안 관점

체크포인트와 매니페스트는 새 리더에 들어오는 신뢰할 수 없는 입력이다. 보안 리뷰에서 HIGH 4건, MEDIUM 8건이 나왔고 `f37da5c4`에서 수정했다.

| 문제 | 심각도 | 상태 |
|------|--------|------|
| `row_bytes`와 shape가 연결되지 않고 크기 계산에서 정수 wrap 가능 | 높음 | 수정: `out * in * element_size` 검사, 정수가 아닌 shape 거부 |
| shard 경로와 매니페스트가 모델 디렉터리를 벗어날 수 있음 | 높음 | 수정: 모두 `resolve_contained` 통과 |
| 텐서 끝 오프셋을 shard 길이와 비교하지 않음 | 높음 | 수정: 열린 핸들의 길이로 checked 연산 |
| 중복 `(layer, suffix)`, 0개 또는 과도한 레이어 | 중간 | 수정: 거부, `MAX_LAYERS`는 4096 |
| `prepare-expert-stream`이 기존 파일을 통해 기록 | 중간 | 수정: unlink, `create_new`, 매니페스트를 마지막에 기록 |
| 스트리밍 중 I/O 오류 시 panic | 낮음 | 미해결 |
| `prepare-expert-stream`의 여유 공간 검사 없음 | 낮음 | 미해결 |
| `shard_names`가 `model.safetensors.index.json`을 `resolve_contained` 없이 읽어, 그 경로의 FIFO가 로드를 멈춤 | 낮음 | 미해결 |

### 2.2 성능 관점

다른 작업이 함께 돌던 호스트(load 9~13)에서 OptiQ 2-bit, 128토큰, arm당 4회 교차 실행으로 측정했다. per-layer sync arm은 상주 가중치에서 라우터 top-k를 평가하므로 I/O 없이 동기화 비용만 분리한다.

| 항목 | 상주 | per-layer sync | 변화 |
|------|------|----------------|------|
| Decode | 42.0 tok/s (토큰당 23.8 ms) | 25.6 tok/s (토큰당 39.1 ms) | 토큰당 약 +15.3 ms, 1.64배 |

레이어당 약 0.32 ms다. 부하가 있는 호스트의 수치이므로 벤치마크는 아니다. `docs/benchmark_results/versions/unreleased.md#expert-streaming-sync-cost-m3ultra-2026-11-10` 참고. 실제 miss가 있는 스트리밍 처리량은 짝지은 측정이 없다.

### 2.3 호환성/의존성 관점

- **Breaking Changes**: 없음. 라우팅 expert가 들어가는 머신은 기본 판단에 따라 상주 실행을 유지한다.
- **새 의존성**: 없음.
- **스위치**: `MLXCEL_EXPERT_STREAM`(`1`, `0`으로 자동 규칙 덮어쓰기), `MLXCEL_EXPERT_ARENA_BYTES`(아레나 크기).
- **새 명령**: `mlxcel prepare-expert-stream`은 contiguous 파일을 기록하며, 이 파일은 해시 검사 후 자동으로 사용된다.

### 2.4 코드 품질 관점

- `qwen4_exp` 라이브러리 테스트 199개 통과. 슬롯 맵, LRU, 배치, 판단 규칙, 악의적 체크포인트, contiguous writer 왕복 테스트를 포함한다.
- `src/models/qwen4_exp_stream_source.rs`(589줄)와 `src/models/qwen4_exp_stream_tests.rs`(592줄)는 500줄 지침을 넘는다.

---

## 3. 기술적 선택과 그 이유

### 3.1 MLX 그래프를 유지하고 레이어마다 한 번 동기화

**컨텍스트:** 호스트는 expert를 읽기 전에 레이어의 top-k를 알아야 하고, 명령이 읽는 동안 슬롯이 바뀌면 안 된다.

| 대안 | 장점 | 단점 |
|------|------|------|
| Metal 핸드셰이크 직접 구현 | 레이어당 비용을 줄일 수 있음 | 네이티브 코드가 크게 늘고, 효과를 입증할 근거가 아직 없음 |
| **선택: MLX 그래프에서 레이어별 top-k `eval`** | 새 네이티브 코드가 없고, 동기화 지점이 슬롯 재사용도 안전하게 함 | hit rate 100%에서 토큰당 약 15.3 ms |

**선택 이유:** 이 모드는 체크포인트가 상주할 수 없을 때만 켜지므로, 토큰당 고정 비용의 대안은 실행 불가다.

**트레이드오프:** miss가 없어도 스트리밍 decode 속도는 동기화 비용에 묶인다.

### 3.2 prefill에서 상주 kernel 선택을 재현

**컨텍스트:** 첫 버전은 큰 prefill 윈도우를 토큰 행 단위로 나눴다. 8 GB 아레나에서 greedy 출력이 5개 프롬프트 모두 상주 실행과 결정적으로 달랐고, 짧은 프롬프트는 일치했다. 그래서 eviction은 원인에서 제외됐다.

**결정:** 각 expert의 행을 한 배치에 유지하고, MLX가 상주 `E = 512` 텐서와 같은 kernel과 tile을 고르도록 배치를 계획한다. 이렇게 계획할 수 없는 윈도우는 하위 비트가 다를 수 있다는 경고를 한 번 남긴다. 어떤 실행에서도 경고는 나오지 않았다.

### 3.3 모델이 들어가는 머신에서는 기본 꺼짐

**컨텍스트:** 스트리밍은 decode 속도를 메모리와 맞바꾼다. 자동 규칙은 Metal working set의 60%를 넘을 때만 켜진다. 스트리밍할 수 없는 레이아웃에서 `On`은 이유와 함께 실패하고 `Auto`는 상주로 돌아간다.

---

## 4. 구현 상세

```
[변경 전]
router top-k -> 상주 expert 512개에 gather_qmm

[변경 후, 스트리밍 시]
router top-k -> 호스트로 eval -> miss를 LRU victim에 읽기 -> 슬롯 인덱스로 gather_qmm
                                    ^ 페이지 정렬 호스트 메모리를 복사 없이 MLX로 래핑
```

주요 모듈:

- `src/models/qwen4_exp_stream.rs`: 모드 판단, `ExpertStream::resolve`, `check_streamable`.
- `src/models/qwen4_exp_stream_arena.rs`, `qwen4_exp_stream_slots.rs`: 아레나, 슬롯 맵, LRU. 크기는 #2883의 `host_memory`와 `AvailabilityGate`를 사용한다.
- `src/models/qwen4_exp_stream_source.rs`: 신뢰 검사를 포함한 체크포인트와 contiguous 파일 읽기.
- `src/commands/prepare_expert_stream.rs`: contiguous 파일 writer.
- `src/server/routes/metrics.rs`: `mlxcel_expert_arena_*`.

96 GB 미만에서 스트리밍 중 드래프터를 설정하면 경고를 남기며, 드래프터를 끄지는 않는다.

---

## 7. 변경 요약

### 통계

| 항목 | 값 |
|------|----|
| 변경 파일 (이 보고서 제외) | 21 |
| 추가 라인 | +2552 |
| 삭제 라인 | -6 |

### 관련 커밋

| Hash | 유형 | 메시지 |
|------|------|--------|
| `eb930a1d` | feat | stream qwen4_exp routed experts from the SSD |
| `b9f14d00` | fix | keep the resident sorted kernel on the expert arena |
| `986cb570` | fix | keep the expert arena stats line after the model is dropped |
| `f37da5c4` | fix | harden expert streaming against hostile checkpoints |
| `98759ec9` | docs | record qwen4_exp expert streaming in the changelog |

---

## 8. 후속 조치

### 완료 필요

- [ ] 체크포인트보다 메모리가 작은 Mac에서 외부 PLE 테이블과 함께 OptiQ 2-bit를 실행하고 swap-out이 없는지 확인. 해당 머신이 없었고, 스트리밍과 외부 PLE 테이블을 함께 실행한 적도 없다.
- [ ] 유휴 호스트에서 두 가지 이상의 아레나 크기, contiguous 파일 유무, 4K 토큰 prefill에 대한 5회 짝지은 중앙값 측정.
- [ ] 실제 체크포인트로 `prepare-expert-stream` 실행. 지금까지는 단위 테스트 왕복만 수행했다.

### 모니터링 필요

- `[Expert arena: ...]`의 hit rate와 I/O, `mlxcel_expert_arena_*`.
- 윈도우 계획 경고. 이 경고가 나오면 출력이 상주 실행과 하위 비트에서 다를 수 있다.

### 향후 개선 사항

- 이슈 항목 중 미구현: read-ahead, warm-start 사용 프로파일, 가용 메모리에 따른 prefill chunk 크기. 아레나 버퍼 wiring은 측정하지 않았다. 스트리밍과 배치 decode 조합은 테스트하지 않았다.
- `preflight_resident_bytes`는 `check_streamable`을 실행하지 않는다. `auto`가 상주 경로로 보내는 레이아웃을 아레나 크기로 추정해 적게 계산한다.

---

## 부록

### A. 테스트 결과

512 GB M3 Ultra, release 빌드, 스트리밍 강제:

- 프롬프트 5개, 256토큰: 24 GB와 8 GB 아레나 모두 상주 실행과 바이트 동일.
- 한 번의 forward에 4608 컨텍스트 토큰을 넣은 뒤 logit trace: width 32와 1, 두 아레나 모두 출력 정밀도 기준 차이 행 0개.
- 3286토큰 프롬프트의 기본 prefill chunking: 두 아레나 모두 동일하지만 생성이 16토큰에서 멈춰 근거로는 짧다.
