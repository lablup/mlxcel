# 기술 보고서: PR #1154 - feat(server): warm the next turn's history prefix in the background

**날짜**: 2026-08-14
**작성자**: Jeongkyu Shin
**상태**: 완료
**언어**: Rust
**위험도**: 낮음 (완전히 유휴인 스케줄러에서만 디스패치되는 백그라운드 전용 작업. fire-and-forget 실패 의미론, 킬 스위치 제공)

---

## 요약

#1143의 히스토리 경계 스냅샷은 다음 턴에게 마지막 사용자 메시지까지 덮는 히트를 주지만, 직전 어시스턴트 답변은 여전히 매 턴 포그라운드 경로에서 prefill된다. 이번 변경은 그 작업을 크리티컬 패스 밖으로 옮긴다. 정상 완료 후 서버가 다음 턴의 예상 히스토리 접두사를 렌더링해 두고, 스케줄러가 할 일이 없을 때 대화의 기존 스냅샷을 복원해 그 접두사까지의 델타만 prefill한다.

설계의 하중을 받치는 부분은 타깃 벡터다. warm-up은 자기가 체인을 이어받은 경계 엔트리를 supersede하는 `Boundary` 스냅샷을 저장하므로, 다음 턴이 매치할 수 없는 타깃은 백그라운드 prefill 하나를 낭비하는 데서 그치지 않고 멀쩡히 작동하던 히트를 파괴한다. 출하된 구성 전에 직관적인 구성 둘을 만들어 재고 기각했다. 클립 없는 `render(messages + reply, add_generation_prompt = false)` 타깃은 `cached_tokens`를 0으로 몰아 warm-up이 없느니만 못했고, 프로브 하나에 대고 클립한 버전은 3토큰만 건졌다. 출하된 두 프로브 구성은 qwen3.5-0.8b에서 둘째 턴 캐시 토큰을 227 중 150에서 194로 올리고 둘째 턴 비캐시 prefill을 77에서 33으로 줄이며, 포그라운드 작업이 있는 동안 warm-up이 절대 돌지 않음을 카운터로 보증한다.

---

## 1. 문제 정의

### 1.1 배경

snapshot-only 계열에서 #1143이 히스토리 경계의 캐시 히트를 만들었고, 부분 복원이 불가능한 순환 상태 계열(#1145의 범위 참조)에는 그 경계 히트에 이 warm-up을 더한 것이 재사용 스토리의 전부다. 포그라운드에 남은 것은 직전 답변이다. 히스토리로 재렌더링된 답변을 매 턴 시작마다 prefill해야 하는데, 소형 모델에서는 수십 토큰이고 대형 모델에서는 피할 수 있는 31B급 작업이다. 그 상태를 계산하는 데 필요한 것(경계 스냅샷과 답변 텍스트)은 완료가 끝나는 순간 모두 존재하고, 그 계산 중 어느 것도 사용자가 기다리는 동안 일어날 필요가 없다.

### 1.2 기존 문제

- **피할 수 있는 포그라운드 작업.** 경계 스냅샷과 다음 턴 히스토리 접두사 사이의 델타를 매 턴 크리티컬 패스에서 지불했다.
- **순진한 타깃 벡터는 능동적으로 해롭다.** warm-up이 체인의 출발점인 경계 엔트리를 supersede하므로, 매치 불가능한 벡터를 저장하면 작동하던 히트가 무용한 엔트리로 바뀐다. 가설이 아니라 실측이다. 첫 구성이 `cached_tokens`를 227 중 0, 275 중 0으로 퇴행시켰다.
- **백그라운드 작업이 경합하면 안 된다.** 모델 forward는 중단할 수 없다. 요청이 도착하는 중에 시작된 warm-up은 그 요청을 지연시킨다. 스케줄링 조건은 증명 가능하게 보수적이어야 한다.
- **관측성이 없었다.** 전용 카운터가 없으면 한 번도 돌지 않은 warm-up과 돌아서 무용한 엔트리를 저장한 warm-up을 구분할 수 없다.

### 1.3 위험 평가

| 위험 | 영향 | 가능성 |
|---|---|---|
| warm-up 타깃이 다음 턴과 안 맞아 작동하던 경계 히트를 supersede | 높음 | 낮음 (두 프로브 클립, 실패한 구성들을 테스트로 고정) |
| warm-up forward가 포그라운드 요청을 지연 | 중간 | 낮음 (유휴 전용 디스패치: 빈 배치, 빈 prefill 큐, 주차된 청크 prefill 없음. 부하 아래에서 검증) |
| warm-up 실패가 완료된 요청을 파손 | 낮음 | 매우 낮음 (fire and forget: 응답 채널도 큐 예약도 없고 send 실패는 debug 라인) |
| 스트리밍 경로가 warm되지 않을 답변의 누적 비용을 지불 | 낮음 | 낮음 (warm-up 가능할 때만 content 누적) |

---

## 2. 기술 검토

### 2.1 타깃 구성 셋, 그중 둘은 측정으로 기각

1. **클립 없는 `render(messages + reply, add_generation_prompt = false)`.** 직관적이지만 틀렸다. 템플릿은 마지막 어시스턴트 메시지를 이전 것과 다르게 렌더링하므로, 결과가 다음 턴 프롬프트의 접두사가 아니다. 실측: `cached_tokens`가 227 중 0, 275 중 0으로 떨어졌다. 무용한 엔트리가 작동하던 경계 엔트리를 supersede했기 때문에 warm-up이 없는 것보다 엄격히 나쁘다.
2. **같은 렌더를 프로브 렌더 하나에 대고 클립.** 안전하지만 거의 무용하다. 두 렌더가 어시스턴트 헤더 직후부터 갈라져 클립이 답변을 버렸다. 실측: 캐시 토큰 +3 (153 대 150).
3. **끝의 자리표시자 사용자 턴만 다른 프로브 둘.** 둘 다 답변을 다음 턴이 놓을 자리에 놓으므로, 공통 접두사가 다음 턴의 고유한 말이 시작되는 지점에서 정확히 끝나고 답변이 클립에서 살아남는다. 이것이 출하됐다. `render_next_turn_history`가 프로브 둘을 반환하고 `clip_warmup_target`이 일치하는 머리 부분으로 줄인다.

이 과정은 프로젝트의 측정 규율을 그대로 보여주는 사례다. 직관적 설계를 만들어 프로덕션 경로로 재고, 해롭다는 것을 보이고, 교체했으며, 실패한 구성들을 지우는 대신 기록했다.

### 2.2 유휴는 정말 유휴다

`run_next_prompt_cache_warmup`은 스케줄러의 `Idle` 틱 arm에서만 디스패치되고, `can_run_prompt_cache_warmup`은 추가로 빈 활성 배치, 빈 prefill 큐, 주차된 청크 prefill 없음을 요구한다. 이 논리곱이 포그라운드 작업이 존재하는 동안 중단 불가능한 forward가 시작되지 않음을 보증하는 유일한 방법이다. 큐는 유계이고 최신 우선이라, 완료가 몰려도 낡은 warm-up이 적체되지 않는다.

### 2.3 작업 자체는 최소다

warm-up은 대화가 이미 가진 가장 긴 스냅샷을 복원하고, 델타만 prefill하며, 로짓의 마지막 위치만 평가하고(#1150 리뷰가 경계 forward에서 가르친 것과 같은 교훈), 결과를 `Boundary` origin으로 저장해 체인의 출발점을 supersede한다. 대화는 경계 스냅샷을 정확히 하나 유지한다. 모든 경로가 조용히 반환한다. 클라이언트는 이 작업이 생기기 전에 이미 답을 받았다.

### 2.4 귀속이 카운터에 내장돼 있다

`snapshot_warmups_run`은 warm-up이 실제로 복원하고 prefill하고 저장했을 때만 오를 수 있어, A/B의 팔 귀속 카운터이자 부하 아래 양보의 증거다. `snapshot_warmups_skipped`는 큐 폐기를 계상한다. 동시성 프로브는 프로젝트의 부하 걸린 박스 방법론을 따랐다. 고정 대 증가 카운터는 로드 애버리지 20에서도 조용한 머신에서와 같은 것을 말해 주지만, 지연 백분위는 그렇지 않다.

---

## 3. 기술적 선택과 그 이유

### 3.1 설계로서의 fire and forget

`ModelRequest::PromptCacheWarmup`은 응답 채널을 갖지 않고 큐 예약도 하지 않으며, `submit_prompt_cache_warmup`은 send 실패를 debug 라인으로 처리한다. 클라이언트는 이미 답을 받았으니 실패시킬 호출자가 없다. 이 구조가 작업 내부 모든 실패 모드의 폭발 반경을 "대화는 #1143 경계 스냅샷을 유지한다"로 한정한다.

### 3.2 스트리밍은 조건부로 누적한다

스트리밍 경로는 해당 요청에 warm-up이 가능할 때만 필터링된 `delta.content`를 누적하므로, 평범한 스트림은 아무것도 추가로 할당하지 않는다. 도구 호출 턴은 통째로 건너뛴다. 그 답변은 어시스턴트 히스토리가 아니라 tool result로 모델에 되돌아가므로, warm하면 잘못된 접두사를 만들게 된다.

### 3.3 별도의 킬 스위치

`MLXCEL_DISABLE_CACHE_WARMUP`은 `MLXCEL_DISABLE_BOUNDARY_SNAPSHOT`과 독립이다. 두 기능의 비용 프로필이 다르고(포그라운드 복사 대 유휴 백그라운드 forward), 운영자가 한쪽만 원하는 것이 합리적일 수 있다. 이 스위치는 A/B의 팔 선택기 역할도 해서 두 팔을 한 바이너리로 유지했다.

### 3.4 픽스처 선택은 강제된 것이었고, 기록했다

`enable_thinking = false`의 qwen3.5-0.8b는 발산 클래스 (a)이고, 따라서 경계 스냅샷이 턴이 적중할 수 있는 유일한 대상인 유일한 소형 체크포인트라 warm-up의 이득이 깨끗하게 귀속된다. 기본 thinking의 4b 변형은 이 크기에서 쓸 수 없었다. 0.8b 모델이 어떤 토큰 예산으로도 `<think>` 블록을 닫지 않아 `content`가 비어 오고 warm할 답변이 없다. falcon-h1-tiny는 기록에는 있으나 증거가 아니다. 답변이 정준으로 재토크나이즈되어 warm-up에 남는 델타가 2토큰뿐이다.

---

## 4. 변경 요약

### 통계

| 항목 | 값 |
|---|---|
| 변경 파일 | 17 |
| 라인 | +907 / -8 |
| 새 환경변수 | 1 (`MLXCEL_DISABLE_CACHE_WARMUP`) |
| 새 통계 필드 | 2 (`snapshot_warmups_run`, `snapshot_warmups_skipped`) |
| 포그라운드 동작 변화 | 0 (디스패치가 완전 유휴 스케줄러를 요구) |
| 벤치마크 기록 | `docs/benchmark_results/warmup-snapshot-m1ultra-2026-08-14.md` |

### 영역별 변경

**`src/server/model_provider.rs`**
- fire and forget인 `ModelRequest::PromptCacheWarmup { tokens, ctx }`와 `submit_prompt_cache_warmup`.

**`src/server/batch/scheduler.rs`**
- 유계 최신 우선 warm-up 큐, `can_run_prompt_cache_warmup`, `Idle` arm 전용 디스패치의 `run_next_prompt_cache_warmup`. 복원, 델타 prefill, 마지막 위치 평가, `Boundary` origin 저장.

**`src/server/chat_request.rs`**
- 프로브 렌더 둘을 내는 `render_next_turn_history`와 일치 머리를 취하는 `clip_warmup_target`.

**`src/server/routes/chat.rs`**
- 비스트리밍과 스트리밍 양쪽에 연결된 `submit_next_turn_warmup`, 조건부 content 누적, 도구 호출 턴 건너뛰기.

**`src/server/prompt_cache/policy.rs`, `src/server/batch/observability.rs`, `src/server/routes/cache.rs`**
- 킬 스위치, `/v1/cache/stats`까지 이어지는 카운터 둘.

---

## 5. 검증과 후속

### 통과

- `cargo test --release --lib server::chat_request`(84, 두 프로브 렌더 테스트와 평범한 경우·렌더 중간 발산·전체 발산을 덮는 `clip_warmup_target` 포함), `server::batch::`(357), `server::prompt_cache`(167), `server::routes::`(118), `server::model_provider`(51).
- `cargo clippy --release --lib --tests --features metal,accelerate -- -D warnings`, `cargo fmt --check`.
- Apple M1 Ultra 실물 체크포인트 A/B, 킬 스위치를 팔 선택기로 쓴 단일 바이너리, 4초 생각 시간의 3턴 대화를 `/v1/chat/completions`로 실행. 둘째 턴 캐시 227 중 150에서 194로(두 팔의 프롬프트 길이 동일), 둘째 턴 비캐시 prefill 77에서 33으로(57% 감소), 셋째 턴 비캐시 59에서 24로, `warmups_run` 0 대 2.
- `--parallel 4`에 클라이언트 둘이 2.6초 동안 요청 12개를 몰아넣은 동시성 프로브. 부하 중 `warmups_run`은 0에 고정되고 큐 스킵 10회에 포그라운드 오류 0, 부하가 끝난 뒤에야 2에 도달. 양보는 카운터로 증명했고, 카운터는 지연 수치가 버티지 못할 부하 걸린 박스에서도 유효하다.

### 미검증

- 벽시계 지연과 TTFT(측정 내내 로드 애버리지 4-20, 암시 대신 미측정으로 명시), 턴당 백그라운드 forward 한 번의 유휴 GPU 비용, 31B 규모의 피크 메모리, 유휴 창이 짧지만 0이 아닌 지속적 부분 부하에서의 동작.
- gemma-4-31b-it-4bit. 이 브랜치는 소형 체크포인트만 돌렸으므로 에픽 수준 검증에 넘겼다.
- 셋째 턴의 프롬프트 길이는 prefill 형태가 바뀌면 답변이 갈라져 두 팔 사이에 약간 다르다. 그래서 그 셀의 비교 가능한 수치는 캐시 토큰이 아니라 비캐시 토큰(59 대 24)이다.

### 후속

- 31B 규모 실행은 에픽 수준 검증의 몫이다. warm-up의 절대 이득이 가장 크고, 스냅샷 예산 상호작용(#1146의 용량 산수, #1150 보고서 기준 여전히 미해결)이 가장 조여올 지점이다.
- 지속적 부분 부하는 조사하지 않은 유일한 스케줄링 영역이다. 짧은 유휴 창이 warm-up을 허용했다가 도착과 충돌하는 것으로 드러나면, 히스테리시스를 넣을 자리는 `can_run_prompt_cache_warmup`의 논리곱이다.
