# 기술 보고서: PR #2849 - perf(moe): switch gate/up 커널을 단일 토큰 디코드로 제한

**작성일**: 2026-11-04
**상태**: 리뷰 중 (열린 PR, #2515 종료)
**언어**: Rust, Markdown, CSV
**위험도**: 낮음

---

## 요약

PR #2849는 `models/olmoe-1b-7b-0125-4bit`에서 `moe_switch_gate_up` Metal 커널을 실제 배치 디코드 형상으로 다시 측정하고, `SwitchGLU::forward`가 이 커널을 언제 시도할지 정하는 gate를 바꾼다. M3에서 커널을 켜 두게 만든 2026-10-21 기록은 서버를 기본값인 `--parallel 4`로 실행했다. 그래서 동시성 5에서 8까지의 레벨은 4행 디코드 배치 뒤에 대기했고, "batch 8 control"은 64 route짜리 한 스텝이 아니라 4행 두 웨이브였다. `--parallel 8`에서 occupancy가 정확히 B로 확인되고 arm마다 교대 측정을 세 번 한 결과, 커널은 B=2부터 7까지 모든 지점에서 `gather_qmm`보다 스트림당 3.6%에서 7.6% 느리다. 반면 B=8 sorted-path control의 차이는 +0.6%다. Teacher-forced parity trace는 0/256 top-1, 0/32 decided-position 불일치와 로짓 차이 0을 보고하므로, 다중 토큰 형상에서 커널이 측정 가능한 정확도를 사 오지는 않는다. gate는 이제 모든 세대에 대해 `tokens == 1`이고, Inkling의 `forward_activated` 경로는 그대로다.

---

## 1. 문제 정의

### 1.1 배경

PR #2514는 routed gate/up 쌍을 위한 direct-dequant Metal 커널 `moe_switch_gate_up`을 공유 `SwitchGLU`에 추가했다. 이 커널은 `forward_activated`를 통해 Inkling 디코드에 연결되고, 다른 MoE 계열이 다중 토큰 배치 디코드에서 타는 `SwitchGLU::forward`에도 연결된다. B >= 2에서는 #268 fused single-token 커널이 절대 실행되지 않으므로 switch gate/up 커널이 유일한 custom 후보이고, `MLXCEL_FUSED_MOE=0`이 이를 깔끔하게 분리한다.

이슈 #2515는 실제 체크포인트로 배치 디코드를 측정하고, 두 경로의 배치 형상 parity를 기록하고, 커널이 회귀하면 gate를 조정하라고 요구했다. 이슈는 이 PR이 머지될 때까지 열려 있다.

### 1.2 기존 문제점

- **배치 형상을 측정하지 않았다.** 2026-10-21 A/B는 `mlxcel-server`를 기본값 `--parallel 4`로 실행했다. 동시성 4를 넘으면 요청이 4행 디코드 배치 뒤에 대기하므로, B=5부터 B=8로 표시된 행은 B행 배치를 측정하지 못했고, batch-8 control은 64 route 한 스텝이 아니라 4행 두 웨이브(웨이브당 32 route)를 비교한 셈이다.
- **증거가 노이즈 바닥 안에 있었다.** 그 기록의 batch-3과 batch-4 셀은 arm당 측정 두 번으로는 실행 간 변동과 구분할 수 없었다. 명확한 손실은 스트림당 -4.0%인 B=2 하나뿐이었다.
- **gate의 전제가 그 기록이었다.** 공유 gate는 그 숫자를 근거로 M3, M4, M5에 B >= 2 커널을 허용했다. M1과 M2는 측정된 약 10% 손실(#2587) 뒤 이미 B >= 2에서 거절하고 있었다.
- **실제 체크포인트의 배치 parity가 없었다.** 배치 형상 parity는 합성 fixture에서만 확인된 상태였다.
- **제안된 개선안이 너무 좁았다.** 2026-10-21 실행 뒤 `tokens >= 2 && total < 24`가 B=2에서만 커널을 유지하는 carve-out으로 제안됐다. 그 제안은 batch-2 셀 하나에 기대고 있었다.

### 1.3 위험성

| 위험 | 영향 | 발생 가능성 |
|---|---|---|
| M3/M4/M5 예외를 유지하면 M3에서 측정된 스트림당 3.6%에서 7.6%의 디코드 손실이 남는다 | 중간 | 이 변경 없이는 높음 |
| M4와 M5가 측정 없이 거절되면서 다중 토큰 이득 가능성을 잃는다 | 낮음 (성능만 해당) | 양성 셀이 측정될 때까지 확정 |
| 다른 체크포인트가 이 형상에서 다른 결과를 보인다 | 낮음 | 낮음, parity 결과는 체크포인트별이다 |
| 단일 토큰 경로의 동작이 바뀐다 | 낮음 | 매우 낮음, gate 인자 수 말고는 경로가 그대로다 |

---

## 2. 기술적 검토 사항

### 2.1 성능 관점

측정은 이전 설정을 재사용하지 않고 배치 형상을 바로잡았다. `mlxcel-server`는 `--parallel 8 --ctx-size 8192`로 실행해 컨텍스트를 1024 토큰 슬롯 여덟 개로 나눴고, 따라서 클라이언트 동시성 B가 B행 디코드 배치에 도달한다. 프롬프트 캐시와 APC는 껐는데, 공유 prefix 재사용 뒤에 여덟 클라이언트가 서로 다른 시점에 디코드를 시작하지 않게 하기 위해서다. 스케줄러의 occupancy 카운터(`mlxcel_batch_decode_tokens_total / mlxcel_batch_decode_steps_total`, `--metrics`로 노출)는 두 arm 모두 모든 레벨에서 정확히 B를 보고했고, 레벨마다 디코드 스텝 127회와 127 * B개의 진행 토큰이 있었다. ladder가 이름 붙인 배치 크기를 실제로 측정했는지 확인하는 지표는 `--parallel` 값이 아니라 이 카운터다.

harness는 새 서버 프로세스 여섯 개를 A, B, A, B, A, B 순서로 교대 부팅했다. 부팅마다 full-ladder warm-up 한 번을 버리고 측정 패스 한 번을 기록했으므로 arm당 측정 패스가 세 번이다. ladder는 B = 2부터 8까지이고, 명목 512 토큰 프롬프트(실제 445), 128 생성 토큰, temperature 0, greedy 샘플링을 썼다. Arm A는 기본 환경으로, B = 2부터 7(`tokens * top_k` = 16에서 56, sorted-path 임계값 미만)에서 커널을 시도한다. Arm B는 `MLXCEL_FUSED_MOE=0`을 설정해 두 projection이 indexed `gather_qmm` 경로를 탄다. B = 8(64 route)에서는 두 arm 모두 sorted path를 타므로, 그 행은 두 arm이 동일한 코드를 실행하는 control이다.

스트림당 디코드 처리량은 측정 세 번의 중앙값이고, spread는 `(max - min) / median`이다.

| B | kernel 중앙값 tok/s | kernel spread | gather 중앙값 tok/s | gather spread | kernel vs gather |
|---:|---:|---:|---:|---:|---:|
| 2 | 233.0 | 0.5% | 241.8 | 6.4% | -3.6% |
| 3 | 180.4 | 0.7% | 195.2 | 0.4% | -7.6% |
| 4 | 154.6 | 0.5% | 162.2 | 0.5% | -4.7% |
| 5 | 128.6 | 0.8% | 137.9 | 0.1% | -6.7% |
| 6 | 114.5 | 0.3% | 122.3 | 0.2% | -6.4% |
| 7 | 101.8 | 0.2% | 108.9 | 0.0% | -6.5% |
| 8 (control) | 95.0 | 1.9% | 94.4 | 0.1% | +0.6% |

gather arm의 B = 2 spread 6.4%는 첫 실행 한 번이 226.7 tok/s로 낮았고 나머지 두 번이 241.8과 242.2였기 때문이다. kernel arm은 세 번 모두 232.1에서 233.2 사이였다. 나머지 셀은 전부 2% 미만이다. 집계 처리량도 같은 모양이다. B = 2부터 7까지 -3.0%에서 -6.3%이고, B = 8 control은 +0.4%다.

평균 디코드 스텝 시간과 occupancy:

| B | kernel step ms | gather step ms | kernel vs gather | kernel occupancy | gather occupancy |
|---:|---:|---:|---:|---:|---:|
| 2 | 4.30 | 4.14 | +3.9% | 2.00 | 2.00 |
| 3 | 5.56 | 5.14 | +8.2% | 3.00 | 3.00 |
| 4 | 6.50 | 6.20 | +4.8% | 4.00 | 4.00 |
| 5 | 7.82 | 7.30 | +7.1% | 5.00 | 5.00 |
| 6 | 8.80 | 8.25 | +6.7% | 6.00 | 6.00 |
| 7 | 9.90 | 9.27 | +6.8% | 7.00 | 7.00 |
| 8 (control) | 10.63 | 10.69 | -0.6% | 8.00 | 8.00 |

`1000 / step_ms`는 모든 셀에서 스트림당 처리량을 그대로 재현하므로, 두 표는 같은 디코드를 독립적으로 읽은 결과다. 첫 토큰까지의 시간은 모든 레벨에서 0.6% 안에서 일치한다. B=2에서 109.0/109.0 ms, B=3에서 137.0/137.7, B=4에서 172.0/172.1, B=5에서 206.7/206.9, B=6에서 240.8/241.0, B=7에서 273.7/275.1, B=8에서 309.6/308.7이다. 따라서 처리량 차이는 admission이나 prefill이 아니라 디코드 연산에서 나온다.

B = 8 control은 프로토콜 편향의 상한을 스트림당 +0.6%, 집계 +0.4%, 스텝 시간 -0.6%로 제한한다. B = 2부터 7까지의 모든 손실은 그 바닥의 몇 배다. 손실은 B에 대해 단조롭지 않고 B = 3이 가장 나쁘고 B = 4가 가장 완만하므로, 단일 스케줄링 임계값이 아니다. 이 route 수에서는 커널의 threadgroup grid와 route pairing 비용이 indexed gather보다 크다.

### 2.2 정확성과 parity

`examples/logit_trace`를 `MLXCEL_TRACE_BATCH_ROWS=B`로 실행하면, 추적하는 각 위치를 서로 독립적인 prefix를 가진 B행 위에서 width-one `forward_batched_with_ids` 디코드로 재생한다. 그래서 공유 `SwitchGLU`는 `x [B, hidden]`과 `indices [B, top_k]`를 받는데, 이는 `forward`가 두 경로 중 어느 쪽으로도 축약하는 바로 그 형상이다. 코퍼스는 `docs/CONTINUOUS_BATCHING.md`이고, teacher-forced 위치 16개짜리 청크 16개(폭마다 256 위치), top-k 8이며, 처리량 실행과 같은 두 환경 arm을 쓴다. `scripts/compare_logit_traces.py`는 gather arm을 기준으로, kernel arm을 후보로 두고, decided 위치를 기준 상위 두 로짓 격차가 2.0 이상인 위치로 정의한다.

| B | top-1 불일치 | decided-position 불일치 | 로짓 차이 (p50 / p90 / p99 / max) | perplexity |
|---:|---|---|---|---|
| 2 | 0 / 256 | 0 / 32 | 0.0000 / 0.0000 / 0.0000 / 0.0000 | +0.000% |
| 3 | 0 / 256 | 0 / 32 | 0.0000 / 0.0000 / 0.0000 / 0.0000 | +0.000% |
| 4 | 0 / 256 | 0 / 32 | 0.0000 / 0.0000 / 0.0000 / 0.0000 | +0.000% |
| 5 | 0 / 256 | 0 / 32 | 0.0000 / 0.0000 / 0.0000 / 0.0000 | +0.000% |
| 6 | 0 / 256 | 0 / 32 | 0.0000 / 0.0000 / 0.0000 / 0.0000 | +0.000% |
| 7 | 0 / 256 | 0 / 32 | 0.0000 / 0.0000 / 0.0000 / 0.0000 | +0.000% |
| 8 (control) | 0 / 256 | 0 / 32 | 0.0000 / 0.0000 / 0.0000 / 0.0000 | +0.000% |

64 위치짜리 더 짧은 패스도 모든 레벨에서 일치했으므로 결과가 표본 크기에 의존하지 않는다. B = 8 control은 비교 자체가 차이를 만들어 내지 않는다는 것을 확인해 준다. 이는 이 코퍼스와 이 형상에서 OLMoE에 대한 parity 진술이지, 일반적인 bit-identity 주장이 아니다. 커널은 f32로 누적하고 reduction 순서가 `gather_qmm`과 다르므로, 다른 체크포인트에서는 근접 동률 greedy 토큰이 뒤집힐 수 있다.

### 2.3 호환성과 의존성 관점

- **호환성 파괴**: 공개 API에는 없다. `switch_gate_up_preferred_in_forward`는 `pub(crate)`이고, 시그니처에서 `AppleSiliconGen` 인자만 빠진다.
- **새 의존성**: 없다.
- **세대별 동작**: 이제 모든 세대가 B >= 2에서 gather 경로를 탄다. M1과 M2는 이미 거절하고 있었으므로 그 지점에서 동작이 바뀌지 않는다. M4와 M5는 M3의 폐기된 예외를 물려받지 않고 M3와 함께 거절한다. Inkling의 `forward_activated` 경로는 측정된 단일 토큰 이득 지점에서 커널을 직접 호출한다.
- **도구 체인 pin**: 바뀌지 않았다. 측정은 MLX pin `7ca40033132dae444d45f7fc3daae57e24c13f6c`와 `mlxcel 0.8.1`을 썼다.

### 2.4 코드 품질 관점

- **테스트 커버리지**: 세대 조합을 도는 단위 테스트를 `switch_gate_up_in_forward_is_single_token_only`로 교체했다. 토큰 1에서 true, `2..=8`의 모든 `tokens`에서 false를 단언한다.
- **복잡도**: hot path 조건이 입력 두 개와 하드웨어 조회에서 정수 비교 하나로 줄었다.
- **문서화**: 함수 주석에 M1 Ultra와 M3 Ultra 측정, 이전 기록의 형상 오류, 집중 재측정, M4/M5 공백을 모두 기록했다. 커널 자체 주석에는 두 번째 wire-in이 단일 토큰 전용임을 적었다.
- **기술 부채**: 늘지 않았다. 과거 문서 하나는 동시성 5에서 8 행이 대체됐다고 명시했다.

### 2.5 보안 관점

이 변경은 MoE forward pass 안의 커널 선택만 건드린다. 입력 파싱, 인증, 인가, 로깅 표면을 추가하지 않고, 서버가 받거나 돌려주는 데이터도 바꾸지 않는다.

---

## 3. 기술적 선택과 그 이유

### 3.1 교대 부팅과 바로잡은 배치 형상

**컨텍스트:** 이전 기록의 동시성 5에서 8 셀은 이름 붙인 배치 크기가 아니었다. 서버 기본값 `--parallel 4`가 네이티브 배치를 4행으로 제한하기 때문이다. 그래서 batch-8 control은 control 구실을 하지 못했다.

**고려한 대안:**

| 대안 | 장점 | 단점 |
|---|---|---|
| 2026-10-21 프로토콜 재사용 | 새 harness 작업 없이 과거 숫자와 비교 가능 | 형상 오류를 반복하고 B > 4를 측정할 수 없음 |
| `--parallel`을 8로 올리고 occupancy 카운터로 확인 | 실제 B행 배치를 측정하고, 카운터가 잘못된 ladder를 잡아냄 | 컨텍스트 분할이 커지고 새 부팅이 필요 |
| **선택: `--parallel 8 --ctx-size 8192`, occupancy 확인, 교대 부팅** | 형상을 직접 반증하고, arm당 세 패스로 손실과 부팅 순서 효과를 분리 | 실행이 길고 occupancy가 필수 프로토콜 단계가 됨 |

**선택 이유:** 클라이언트 동시성이 B행 디코드 배치가 됐다는 증거는 occupancy 카운터뿐이다. 교대 부팅은 순차 패스 두 번으로는 잡을 수 없는 계통적 순서 효과를 없앤다. 두 arm이 동일한 sorted-path 코드를 실행하는 B = 8 control 행은 남은 프로토콜 편향의 상한을 정한다.

**트레이드오프:** 바로잡은 설정은 컨텍스트 슬롯당 메모리 압력을 높이고 프롬프트 캐시와 APC를 끈다. 측정은 깨끗해지지만 절대 처리량을 운영 기본 설정과 직접 비교할 수는 없다.

### 3.2 Gate 결정: 단일 토큰 전용

**컨텍스트:** gate는 모든 Apple 세대의 모든 다중 토큰 디코드에서 custom 커널과 기준 gather 경로 중 하나를 골라야 한다.

**고려한 대안:**

| 대안 | 장점 | 단점 |
|---|---|---|
| M3+ 예외 유지 | 코드 변경 없음 | 잘못된 형상에 기대고, M3에서 측정된 3.6%에서 7.6% 손실을 남김 |
| `tokens >= 2 && total < 24` | B = 3부터 7까지 셀을 제거 | 보호하려던 B = 2를 그대로 허용하는데, 그 셀이 스트림당 3.6% 손실, route 수 특례도 남음 |
| M4/M5용 세대별 테이블 | 측정되지 않은 이득을 보존할 수 있음 | 양성 다중 토큰 측정이 없고, 증거 없이 코드와 테스트 행렬만 늘어남 |
| **선택: 모든 세대에 `tokens == 1`** | 측정한 모든 셀과 일치하고, 규칙 하나이며, 동작이 균일 | M4와 M5의 측정되지 않은 다중 토큰 이득을 포기 |

**선택 이유:** 커널은 이전 제안이 겨냥한 셀 하나만이 아니라 측정한 모든 다중 토큰 크기에서 진다. B = 2에서 스트림당 233.0 대 241.8 tok/s로 3.6% 손실이고, 집중 `--parallel 4` 재측정은 271.6 대 283.9로 -4.3%다. 손실은 B = 7까지 이어지므로 `total < 24` carve-out이 쓸모 있으려면 다중 토큰 범위 거의 전체를 제외해야 한다. 그 셀들을 제외하고 나면 남는 규칙은 토큰 하나다. 바로잡은 형상은 route 수가 축이 아니라는 것도 보여 준다. B = 3은 24 route이고 -7.6%로 가장 나쁜 셀이며, 64 route인 B = 8 control은 노이즈 바닥에 있다. Inkling은 `forward_activated`가 이 gate를 거치지 않고 커널을 직접 호출하므로 영향받지 않는다.

**트레이드오프:** M4와 M5는 자체 측정 없이 거절된다. 양성 다중 토큰 셀이 나오는 호스트가 생기면 그 세대에 한해 gate를 다시 여는 작은 국소 변경이 필요하다.

### 3.3 `tokens >= 2 && total < 24`를 거절한 이유

이 제안은 2026-10-21 기록에서 나왔다. 그때는 B = 2만 명확한 손실이었고 B = 3과 B = 4는 노이즈 바닥 안이었다. 바로잡은 측정은 그 전제를 두 가지로 무너뜨린다. 첫째, 새 실행에서 B = 2 자체가 실제 손실이므로 carve-out은 제거하려던 바로 그 셀을 허용한다. 둘째, 손실이 B = 3부터 7까지에도 있으므로 토큰 하나 위에서 커널이 이기는 측정 영역이 없다. route 수 임계값 24는 B = 3 이상을 제외하고 B = 2만 남기는데, 커널을 유지하면 안 되는 셀이 바로 그 하나다. 집중 `--parallel 4` 실행은 바로잡은 설정이 원인이 아님을 확인한다. 이전 형상에서 B = 2에 -4.3%, B = 4에 -7.5%를 재현한다.

### 3.4 이전 기록의 batch-4 셀이 부호를 바꾼 이유

2026-10-21 기록은 B = 4에서 스트림당 +0.4%를 읽었고, 집중 재측정은 172.9 대 186.9 tok/s로 -7.5%를 읽는다. kernel arm은 172.9 대 169.2로 +2.2% 움직였으므로 차이는 gather 쪽에 있다. `gather_qmm`이 `0f43e98e` 때보다 현재 head에서 약 11% 빠르다. 이전 batch-3과 batch-4 셀은 실제 3행과 4행 배치를 측정했지만 둘 다 노이즈 바닥 안에 있었으므로 커널을 유지할 근거가 되지 못했다. batch-5부터 8까지의 행은 4행 뒤에 대기해 표시된 크기를 측정하지 못했다.

---

## 4. 구현 상세

### 4.1 아키텍처 변경

```
[Before]
SwitchGLU::forward
  total >= 64  -> sorted path
  else if switch_gate_up_preferred_in_forward(n_tokens, silicon_gen)
               -> switch gate/up kernel (M3/M4/M5 at B >= 2)
  else         -> indexed gather_qmm

[After]
SwitchGLU::forward
  total >= 64  -> sorted path
  else if switch_gate_up_preferred_in_forward(n_tokens)
               -> switch gate/up kernel only when n_tokens == 1
  else         -> indexed gather_qmm
```

Inkling 경로는 별도이고 바뀌지 않았다. `SwitchGLU::forward_activated`는 측정된 단일 토큰 이득 지점에서 `forward_switch_gate_up_kernel`을 직접 호출한다.

### 4.2 주요 코드 변경

**파일: `src/models/switch_layers.rs`**

```rust
// Before
pub(crate) fn switch_gate_up_preferred_in_forward(
    tokens: i32,
    silicon_gen: mlxcel_core::hardware::AppleSiliconGen,
) -> bool {
    tokens == 1 || silicon_gen.wide_quantized_projections()
}

// After
pub(crate) fn switch_gate_up_preferred_in_forward(tokens: i32) -> bool {
    tokens == 1
}
```

호출 지점은 더 이상 `mlxcel_core::hardware::get_hardware().silicon_gen`을 읽지 않는다.

```rust
// Before
} else if let Some(activated) = switch_gate_up_preferred_in_forward(
    n_tokens,
    mlxcel_core::hardware::get_hardware().silicon_gen,
)
.then(|| self.forward_switch_gate_up_kernel(x, indices))
.flatten()

// After
} else if let Some(activated) = switch_gate_up_preferred_in_forward(n_tokens)
    .then(|| self.forward_switch_gate_up_kernel(x, indices))
    .flatten()
```

**변경 이유:** 측정된 다중 토큰 손실은 양성 측정이 있는 모든 세대에 적용되고, 유일한 세대별 예외는 이름 붙인 크기를 측정하지 못한 배치 형상에 기대고 있었다. 하드웨어 조회를 없애면 이 hot path에서 전역 하드웨어 접근도 사라진다.

**테스트 변경:**

```rust
#[test]
fn switch_gate_up_in_forward_is_single_token_only() {
    assert!(switch_gate_up_preferred_in_forward(1));
    for tokens in 2..=8 {
        assert!(
            !switch_gate_up_preferred_in_forward(tokens),
            "multi-token batch decode of {tokens} tokens loses to gather_qmm"
        );
    }
}
```

### 4.3 데이터 모델 변경

없다. 이 PR은 `docs/benchmark_results/data/switch-gate-up-batch-m3ultra-2026-11-04/` 아래에 결과 파일 두 개를 커밋한다. `decode.csv`는 42행이고 arm과 배치 크기마다 세 번씩 측정했으며, `focus-parallel4.csv`는 12행이다.

---

## 5. 학습 포인트

### 5.1 형상 확인 지표는 `--parallel`이 아니라 occupancy다

**개념:** 서버의 동시성 설정은 요청 admission 한도이지, 디코드 스텝이 그만큼의 행을 진행했다는 증거가 아니다. 스케줄러의 occupancy 카운터가 스텝이 실제로 진행한 행 수를 측정한다.

**이 PR에서의 적용:** 이전 프로토콜은 서버 네이티브 배치가 4로 제한된 상태에서 셀을 "batch 5"부터 "batch 8"로 표시했다. `mlxcel_batch_decode_tokens_total / mlxcel_batch_decode_steps_total`이 정확히 B이고 스텝 127회에 진행 토큰이 127 * B임을 확인한 것이 새 ladder의 신뢰 근거다.

**흔한 사용처:** 서버가 동시성과 max-batch를 따로 설정하는 경우, 배치 크기별 결과를 주장하는 모든 서빙 벤치마크.

### 5.2 두 arm이 동일한 코드를 실행하는 control 셀

**개념:** A/B 프로토콜에는 부팅 순서, 열 상태, 백그라운드 부하에서 오는 편향이 쌓인다. 두 arm이 같은 코드를 실행하는 셀은 그 편향을 직접 측정한다.

**이 PR에서의 적용:** B = 8에서 두 arm 모두 sorted path를 타므로 스트림당 +0.6% 차이가 프로토콜의 노이즈 바닥이다. B = 2부터 7까지의 모든 손실은 그보다 몇 배 크므로, 셀별 크기가 달라도 결과의 부호는 견고하다.

### 5.3 배치 폭에서의 teacher-forced parity

**개념:** `MLXCEL_TRACE_BATCH_ROWS=B`는 추적하는 각 위치를 서로 독립적인 prefix를 가진 B행 위에서 width-one 디코드로 재생한다. 그러면 프롬프트 타이밍이 같은 라이브 클라이언트 B개 없이도 배치 경로가 보는 `x [B, hidden]`과 `indices [B, top_k]` 형상이 만들어진다.

**이 PR에서의 적용:** 폭마다 teacher-forced 위치 256개, decided 위치 32개에서 top-1과 decided 불일치가 0이었고, B = 2부터 8까지 모든 지점에서 로짓 차이가 0이었다. B = 8 control은 비교 과정의 인공물을 막는다.

### 5.4 교대 부팅의 중앙값

**개념:** 순차 패스 두 번으로는 arm 효과와 부팅 순서를 분리할 수 없다. arm 순서를 교대한 세 패스는 중앙값과 spread를 주고, spread는 차이가 실행 간 변동을 넘는지 보여 준다.

**이 PR에서의 적용:** A, B, A, B, A, B 순서의 부팅 여섯 번으로 arm당 측정 패스 세 번을 얻었다. 유일하게 큰 spread인 gather arm의 B = 2(6.4%)는 첫 실행 한 번이 낮았기 때문이고, 보고서는 이를 중앙값 뒤에 숨기지 않고 그대로 적었다.

---

## 6. 추가 학습 리소스

### 핵심 키워드

| 키워드 | 설명 | 이 PR에서의 관련성 |
|---|---|---|
| `moe_switch_gate_up` | `SwitchGLU`의 routed gate/up 쌍을 위한 direct-dequant Metal 커널 | 이 PR이 배치 동작을 측정하고 gate로 제한하는 커널 |
| `gather_qmm` | MLX의 indexed quantized matmul 경로 | 기준 arm이자 이제 모든 다중 토큰 디코드가 타는 경로 |
| occupancy | 진행된 디코드 토큰을 디코드 스텝으로 나눈 값 | 클라이언트 동시성 B가 B행 배치를 만들었음을 증명 |
| sorted path | `tokens * top_k >= 64`일 때 `SwitchGLU`가 타는 경로 | 두 arm이 동일 코드를 실행하는 B = 8 control |
| `MLXCEL_FUSED_MOE` | fused MoE 커널용 환경 kill switch | `=0`이 B >= 2에서 switch gate/up 커널을 분리 |
| `MLXCEL_TRACE_BATCH_ROWS` | 한 위치를 B행 위에서 재생하는 logit-trace 옵션 | 라이브 동시성 없이 배치 형상 parity를 생성 |
| `forward_activated` | Inkling 디코드가 커널로 들어가는 진입점 | 바뀌지 않았고 공유 gate를 거치지 않음 |

### 관련 기술

- **MLX**: Apple의 배열 프레임워크. `gather_qmm`과 sorted-path 커널이 indexed MoE 기본 연산이다.
- **OLMoE**: 여기서 쓴 1B active, 7B total sparse MoE 체크포인트로, expert 64개, top-k 8, hidden 2048, expert intermediate 1024다.

### 관련 PR과 이슈

- PR #2514: switch gate/up 커널을 추가하고 `SwitchGLU::forward`에 연결했다.
- 이슈 #2515: 이 배치 디코드 측정, parity 진술, gate 조정을 요청했다.
- 이슈 #2587: M1과 M2를 이미 거절하게 만든 M1 Ultra 다중 토큰 약 10% 손실을 측정했다.
- 이슈 #268: fused single-token MoE 커널로, B >= 2에서는 실행될 수 없다.
- 이슈 #2479: 이 커널 옆에 있는 fused decode-MoE down-projection 작업.
- `docs/benchmark_results/switch-gate-up-batch-m3ultra-2026-10-21.md`: 대체된 기록.
- `docs/benchmark_results/switch-gate-up-baseline-m3ultra-2026-10-21.md`: 이 PR이 다시 측정하지 않은 단일 토큰 기준선.

---

## 7. 변경 요약

### 통계

| 항목 | 값 |
|---|---|
| 변경 파일 | 4 |
| 추가 줄 | +232 |
| 삭제 줄 | -34 |
| 추가 테스트 | 단위 테스트 1개 재작성 |
| 순증 | +198 |

### 카테고리별 변경

| 카테고리 | 수 | 요약 |
|---|---|---|
| 성능 | 1 | gate를 단일 토큰 디코드로 좁히고 결정에서 하드웨어 조회 제거 |
| 문서 | 3 | 바로잡은 형상 벤치마크 보고서와 실행별 CSV 두 개 |
| 코드 품질 | 1 | 단위 테스트 교체, 함수와 커널 주석 갱신 |
| 보안 | 0 | 변경 없음 |

### 관련 커밋

| 해시 | 유형 | 메시지 |
|---|---|---|
| `ff898648` | perf | perf(moe): gate the switch gate/up kernel to single-token decode |

---

## 8. 후속 조치

### 검증되지 않은 부분

- **M4와 M5는 측정하지 않았다.** 이제 M3처럼 B >= 2에서 커널을 거절한다. 다시 열려면 그 호스트 중 하나에서 양성 다중 토큰 셀이 나와야 하고, 새 gate 분기가 필요하다.
- **`SwitchGLU::forward`의 `tokens == 1` 경로는 다시 측정하지 않았다.** 이 A/B는 B = 2부터 8까지를 다뤘다. 첫 분기를 살려 두는 단일 토큰 이득은 `switch-gate-up-baseline-m3ultra-2026-10-21.md`의 Inkling 측정에서 온 것이고, `forward`의 단일 토큰 셀은 여기서 독립적으로 재확인하지 않았다.
- **parity는 체크포인트별이다.** 로짓 차이 0은 이 형상과 이 코퍼스의 OLMoE에 해당한다. 커널의 f32 누적 순서가 `gather_qmm`과 달라서 다른 체크포인트에서는 근접 동률 greedy 토큰이 뒤집힐 수 있다.
- **PR 증거를 정리할 때 CI clippy는 대기 중이었다.** PR은 국소 검증으로 대상 gate 테스트와 `cargo fmt --check`만 인용했다. 원격 `cargo-clippy`는 이후 실행 `41666406529`에서 통과했지만, 기록된 증거는 그 결과에 의존하지 않는다.

### 모니터링 필요

- gate 변경 뒤 M3 Ultra의 다중 토큰 디코드 처리량을 본다. B = 2부터 7까지 스트림당 측정된 3.6%에서 7.6%만큼 좋아지고, B = 8에서는 변하지 않아야 한다.
- 단일 토큰 디코드 처리량의 회귀를 본다. 단일 토큰 분기는 바뀌지 않았으므로 측정된 표면에서는 회귀를 예측할 근거가 없다.

### 향후 개선 사항

- M4나 M5 호스트를 쓸 수 있게 되면 이 A/B를 다시 실행한다. 양성 다중 토큰 셀이 나오면 그 세대에 한해 gate를 다시 연다.
- `SwitchGLU::forward`의 단일 토큰 셀을 다시 측정해, 첫 분기가 물려받은 숫자가 아니라 현재 국소 수치를 갖게 한다.
- 2026-10-21 문서는 역사 기록으로 남긴다. batch-2와 batch-4 셀은 이 보고서가 대체하고, 동시성 5에서 8 행은 4행 뒤에 대기한 것으로 읽어야 한다.
- 원본 sweep 텍스트와 서버 로그는 gitignore된 `bench-results/switch-gate-up-batch/20261104-002134/`와 `20261104-focus-p4/`에 남아 있다.

---

## 부록

### A. 테스트 결과

| 검사 | 결과 |
|---|---|
| `cargo test --offline --profile test-fast --features metal,accelerate -p mlxcel switch_gate_up_in_forward_is_single_token_only` | 통과 |
| `cargo fmt --check` | 통과 |
| 원격 `cargo-clippy` | 통과 (실행 `41666406529`, 증거 정리 시점에는 대기 중) |
| 그 밖의 PR 검사 | 보고 시점에 통과 또는 건너뜀 |

### B. 측정 환경

| 항목 | 값 |
|---|---|
| 호스트 | Mac Studio, Apple M3 Ultra, 통합 메모리 512 GB, macOS 26.6.2 (25G83) |
| GPU 세대 | 15, `AppleSiliconGen::M3` |
| 빌드 | `cargo build --release --features metal,accelerate --bin mlxcel-server --example logit_trace`; `mlxcel 0.8.1`; fat LTO, `codegen-units = 1` |
| 측정 바이너리 | `main`의 `79f0a447`, 소스 변경 없음 |
| 서버 플래그 | `--parallel 8 --ctx-size 8192 --metrics --offline --no-prompt-cache --apc-enabled=false` |
| 대상 | `models/olmoe-1b-7b-0125-4bit`, expert 64개, top-k 8, hidden 2048, expert intermediate 1024, affine q4 group 64 |
| 샘플링 | 명목 512(실제 445) 프롬프트 토큰, 128 생성, temperature 0, greedy |
| MLX pin | `7ca40033132dae444d45f7fc3daae57e24c13f6c` |
| 래퍼 | `caffeinate -i`, `scripts/with_indexers_paused.sh`로 indexer 일시 정지 |

### C. 참고 자료

- `docs/benchmark_results/switch-gate-up-batch-m3ultra-2026-11-04.md`
- `docs/benchmark_results/data/switch-gate-up-batch-m3ultra-2026-11-04/decode.csv`
- `docs/benchmark_results/data/switch-gate-up-batch-m3ultra-2026-11-04/focus-parallel4.csv`
- `docs/benchmark_results/switch-gate-up-batch-m3ultra-2026-10-21.md`
- `docs/benchmark_results/switch-gate-up-baseline-m3ultra-2026-10-21.md`
- `src/models/switch_layers.rs`
