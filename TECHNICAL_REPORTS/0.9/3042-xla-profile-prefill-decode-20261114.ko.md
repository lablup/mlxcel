# 기술 보고서: PR #3042 - fix(cli): XLA prefill과 decode를 분리 보고

**작성일**: 2026-11-14
**상태**: 완료
**언어**: Rust
**위험도**: 낮음

---

## 요약

OpenXLA 백엔드의 `generate --profile`은 prefill을 0으로 보고하고 전체 실행 시간으로 모든 생성 토큰을 나눴다. 그래서 prefill이 decode rate 안에 들어갔다. 42토큰 프롬프트의 prefill 커널 시간은 413.5 ms(벽시계 약 590 ms)였고, 이 고정 비용이 짧은 실행을 크게 희석해 같은 환경이 `n`만 바꿔도 51.5에서 132.6 tok/s 사이로 읽혔다. 이제 세션은 이미 수행하던 두 구간을 `GreedyTiming`으로 돌려주고 CLI가 이를 따로 보고한다. `n=64`부터 `512`까지 decode는 154.8에서 165.9 tok/s로 7퍼센트 이내에서 평탄하다.

---

## 1. 문제 정의

### 1.1 배경

OpenXLA 단일 시퀀스 세션은 이미 작업을 나눈다. `generate_streaming_greedy`가 프롬프트 prefix로 KV를 채우고, 마지막 프롬프트 토큰부터 decode를 시작한다. 그러나 CLI는 이 구조를 버리고 호출 전체를 하나의 구간으로 쟀다.

### 1.2 기존 문제점

- `generate_xla`가 `generation_stats_from_duration(...)`으로 `GenerationStats`를 만들었고, 이 헬퍼는 `prefill_time_ms: 0.0`을 고정하고 `decode_tok_per_sec`을 `generated_tokens / total_time`으로 계산한다. `total_time`은 prefill 이전에 시작한다.
- 그래서 프로파일 출력은 항상 `Prefill: 0.00 ms (0.00 tok/s)`였고, decode 수치는 실제로 prefill과 decode를 합친 시간 기준이었다.
- gfx1151에서 llama-3.2-1b-instruct-4bit, 42토큰 프롬프트의 prefill은 약 590 ms다. 같은 바이너리가 `n=64`에서 51.5, `n=128`에서 75.7, `n=256`에서 114.0, `n=512`에서 132.6 tok/s를 냈고 이 중 어느 것도 decode rate가 아니었다.
- MLX 경로는 `DirectEngine`에서 첫 토큰까지를 prefill로 재고 decode 창에는 `N - 1`개 토큰을 세어 올바르게 처리한다.

### 1.3 위험성

| 위험 | 영향 | 발생 가능성 |
|------|------|-------------|
| 기존 벤치마크의 팔 간 비교는 유효하나 절대 수치를 잘못 읽음 | 잘못된 최적화 판단 | 높음, 이미 관측됨 |
| timed 경로가 토큰 스트림을 바꿈 | 잘못된 출력 | 낮음: untimed 경로가 이를 위임하고 device gate가 스트림을 비교 |
| 이미지, 오디오, Gemma3n prepared 경로도 고쳐졌다고 오해 | 부분 수정을 전체 수정으로 착각 | 중간: 코드와 changelog에 명시 |

---

## 2. 기술적 검토 사항

### 2.1 보안 관점

없음. 입력, 인증, 데이터 처리 표면의 변화는 없고, 내부 결과 타입에 벽시계 측정값 `f64` 두 개가 추가될 뿐이다.

### 2.2 성능 관점

계측만 바꾸고 실행은 바꾸지 않는다. 달라지는 것은 측정값이다.

| n | 이전 (tok/s) | 이후 decode (tok/s) | 이후 prefill (tok/s) |
|---|---|---|---|
| 64 | 51.5 | 159.6 | 68.7 |
| 128 | 75.7 | 165.9 | 65.8 |
| 256 | 114.0 | 154.8 | 71.3 |
| 512 | 132.6 | 162.2 | 69.5 |

"이전" 열은 prefill과 decode를 합친 시간 기준이고, "이후" 열은 두 구간을 따로 잰 값이다. decode가 `n`에 대해 평탄한 것이 수용 기준이다.

### 2.3 호환성/의존성 관점

- **Breaking Changes**: 없음. `generate_streaming_greedy`는 시그니처와 동작을 유지하고 timed 변형만 추가된다.
- **New Dependencies**: 없음.
- **호환성**: 모든 XLA 기기와 텍스트/멀티모달 진입점. 이미지, 오디오, Gemma3n prepared 경로는 기존 집계를 유지한다.

### 2.4 코드 품질 관점

`tests/xla_greedy_timing.rs`에 ignored device gate 두 개를 추가했다. 하나는 구간 분리와 토큰 스트림 동일성, 다른 하나는 budget 0 케이스다. `cargo fmt`와 범위 한정 `cargo clippy`는 새로운 경고를 내지 않는다.

---

## 3. 기술적 선택과 그 이유

### 3.1 생성 구조를 바꾸지 않고 기존 호출을 계측

**컨텍스트:** 분리는 이미 세션 안에 있고 사라진 것은 시간값뿐이었다.

**고려한 대안:**

| 대안 | 장점 | 단점 |
|------|------|------|
| CLI가 `prefill`과 `decode_step`을 직접 구동 | 세션 API 추가 없음 | 분리, EOS 처리, 용량 검증을 복제하며 그 부분이 바로 버그가 나기 쉬운 곳 |
| **선택: `generate_streaming_greedy_timed`** | 실행 경로 하나, 계약 복제 없음 | 크레이트 공개 표면에 결과 타입이 추가됨 |

**선택 이유:** 종료 토큰 계약(이슈 #963)은 `drive_greedy_loop`에 있다. CLI에서 루프를 다시 구현하면 서버와 같은 계약의 사본이 하나 더 생긴다. 기존 호출 주위에서 시간을 재면 구현이 하나로 유지된다.

**트레이드오프:** 공개 `GreedyTiming` 타입과 문서화할 메서드가 하나 늘어난다.

### 3.2 decode 창에 생성 토큰 전부를 센다

**컨텍스트:** 두 백엔드의 구간 분리가 다르다.

**고려한 대안:**

| 대안 | 장점 | 단점 |
|------|------|------|
| MLX의 `N - 1` 규칙을 그대로 재사용 | 모든 곳에 규칙 하나 | XLA의 첫 토큰은 `decode_step`이 만들므로 제외하면 창을 과소평가 |
| **선택: XLA에서는 N 전체** | 창에 실제로 있는 것과 일치 | 백엔드별 주석이 필요 |

**선택 이유:** `decode_tok_per_sec`는 측정 창 안에서 생성된 토큰에 대한 비율이다. MLX의 첫 토큰은 prefill에서 나와 창 밖이고, XLA의 첫 토큰은 `decode_step`에서 나와 창 안이다. 규칙은 같고 산술만 다르다. 메서드 문서와 CLI 주석이 이를 밝힌다.

### 3.3 나머지 XLA 경로는 기존 집계 유지

이미지, 오디오, Gemma3n prepared 입력은 `generate_streaming_greedy`를 지나지 않으므로 구간 분리에 별도 작업이 필요하다. 이들을 `generation_stats_from_duration`에 남겨 두는 사실을 changelog에 기록해 조용히 고쳐졌다고 가정하지 않게 했다.

---

## 4. 구현 상세

### 4.1 아키텍처 변경

```
[이전]
generate_xla
  start_time
  session.generate_greedy(...)        # prefill + decode, 분리 없음
  stats = generation_stats_from_duration(total)

[이후]
generate_xla
  session.generate_streaming_greedy_timed(...)
    prefill_ms  = time(prefill(prefix))
    decode_ms   = time(decode_step(last) + drive_greedy)
  stats = GenerationStats { prefill_time_ms, decode_time_ms, ... }
```

### 4.2 주요 코드 변경

**파일: `src/lib/mlxcel-xla/src/lib.rs`**

```rust
pub struct GreedyTiming {
    pub prefill_ms: f64,
    pub decode_ms: f64,
}

let prefill_started = Instant::now();
self.prefill(&prompt_tokens[..split])?;
let prefill_ms = prefill_started.elapsed().as_secs_f64() * 1000.0;
let decode_started = Instant::now();
let first = self.decode_step(prompt_tokens[split])?;
let tokens = self.drive_greedy(first, max_new_tokens, eos_token_ids, on_token)?;
let decode_ms = decode_started.elapsed().as_secs_f64() * 1000.0;
```

**변경 이유:** 세션이 이미 이 두 단계를 수행하므로 시간값만 밖으로 나오면 된다.

**파일: `src/commands/generate.rs`**

```rust
let (tokens, timing) = s
    .generate_streaming_greedy_timed(prompt_tokens, max_tokens, &eos, |_| true)?;
xla_timing = Some(timing);
```

그리고 match 뒤에서 벽시계 대신 `timing.prefill_ms`와 `timing.decode_ms`로 `GenerationStats`를 만든다.

### 4.3 데이터 모델 변경

없음.

---

## 5. 학습 포인트

### 5.1 비율 안에 들어간 고정 비용

고정 비용을 포함한 창으로 계산한 비율은 창 크기를 바꾸면 비교할 수 없다. 여기서는 590 ms prefill이 모든 decode 수치에 나뉘어 같은 하드웨어가 `n=64`보다 `n=512`에서 2.6배 빠르게 보였다. 첫 증상은 틀려 보이는 숫자가 아니라, 무관한 파라미터에 따라 움직이는 숫자였다.

### 5.2 벤치마크를 trace로 교차 검증

판정은 프로파일러가 했다. `rocprofv3 --kernel-trace`가 CLI가 decode로 귀속한 벽시계 창 안에 prefill 디스패치 블록(693개 커널, 413.5 ms)이 들어 있음을 보였다. 이 trace가 없었다면 불일치를 클럭, 열 상태, 테스트 대상 코드 중 하나로 오인했을 수 있고 셋 다 그럴듯했다.

### 5.3 위임은 계약을 하나로 유지한다

기존 호출을 감싸면 EOS와 budget 의미가 공짜로 보존된다. CLI에서 루프를 다시 구현했다면 이미 테스트가 있는 계약의 사본이 하나 더 생겼을 것이다.

---

## 6. 추가 학습 리소스

### 핵심 키워드

| 키워드 | 설명 | 이 PR에서의 의의 |
|--------|------|------------------|
| prefill | KV를 채우고 MLX 경로에서는 첫 토큰을 만드는 프롬프트 단계 | decode rate에 숨어 있던 고정 비용 |
| decode 영역 | 첫 토큰 이후 토큰을 생성하는 시간 | decode rate가 측정해야 하는 양 |
| `GreedyTiming` | XLA 세션이 돌려주는 두 벽시계 구간 | 세션과 CLI 사이의 새 접점 |

### 관련 PR/이슈

- 이슈 #3041: 측정 근거를 담은 버그 이슈.
- PR #3031: 이 prefill을 포함한 절대 수치를 인용한 decode 스칼라 재사용 PR. 150.1 tok/s 합산 중앙값은 prefill 포함 비율이며 팔 간 비교 자체는 유효하다.
- PR #2976: `xla_batch_bench`에서 로드 시간을 분리했다. 다른 표면의 같은 종류의 집계 오류.
- 이슈 #2929: decode tok/s가 측정 창 안의 토큰만 센다는 규칙을 정했다.

---

## 7. 변경 요약

### 통계

| 항목 | 값 |
|------|-----|
| 변경 파일 | 5 |
| 추가 라인 | +288 |
| 삭제 라인 | -4 |
| 추가 테스트 | ignored device gate 2개 |

### 카테고리별 변경

| 카테고리 | 개수 | 요약 |
|----------|------|------|
| 정확성 | 1 | XLA CLI 경로의 단계별 통계 |
| 테스트 | 1 | `xla_greedy_timing` gate |
| 문서 | 1 | CHANGELOG 항목 |

### 관련 커밋

| 해시 | 유형 | 메시지 |
|------|------|--------|
| `0dcc68f3` | fix | fix(cli): report XLA prefill and decode separately (#3041) |

---

## 8. 후속 조치

### 완료 필요

- [x] device gate와 `n` 스윕.
- [ ] 남은 항목 없음.

### 모니터링 필요

- 이 변경 이후 OpenXLA 백엔드에 인용되는 decode rate는 이전 수치와 직접 비교할 수 없다. 이전 수치를 인용하는 벤치마크 표와 이슈는 어떤 집계를 썼는지 밝혀야 한다.

### 향후 개선 사항

- 이미지, 오디오, Gemma3n prepared 경로에도 같은 계측을 확장한다.
- #3031 비교를 단계별 수치로 다시 기준화한다. decode step에서 남은 커널 외 시간을 귀속하기 위한 전제이기도 하다.

---

## 부록

### A. 테스트 결과

```
cargo test --features "xla-iree test-utils" --test xla_greedy_timing -- --ignored --test-threads=1   # 2 통과
cargo check --features xla-iree                                                                      # 통과
cargo fmt --all -- --check                                                                           # 통과
```

### B. 성능 벤치마크

재빌드 바이너리, native quant, context capacity 1024, 스윕 전 warm-up 1회:

| n | prefill | prefill tok/s | decode | decode tok/s |
|---|---|---|---|---|
| 64 | 597.04 ms | 68.67 | 400.96 ms | 159.62 |
| 128 | 623.11 ms | 65.80 | 771.54 ms | 165.90 |
| 256 | 574.99 ms | 71.31 | 1654.14 ms | 154.76 |
| 512 | 590.05 ms | 69.49 | 3157.45 ms | 162.16 |

### C. 참고 자료

- `src/lib/mlxcel-core/src/generate.rs`의 `decode_tok_per_sec`: 창 토큰 규칙.
- `src/lib/mlxcel-core/src/engine/direct.rs`: MLX 경로의 prefill/decode 분리.
