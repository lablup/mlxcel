# PR #2501: 남은 스트리밍 vs 논스트리밍 출력 갭

**작성일**: 2026-10-20
**상태**: 완료
**위험도**: Medium
**언어**: Rust

---

## 요약

#2464와 #2470 종결. #2458이 남긴 여섯 개의 스트리밍 vs 논스트리밍 출력 갭과, 0.8.0-beta 사전 릴리스 스모크 테스트에서 발견된 두 개의 서버 출력 셰이핑 갭(Llama 3.x `<|python_tag|>` 도구 호출이 스트리밍 content로 유출, `/v1/responses`가 harmony reasoning을 버리고 `reasoning_tokens: 0` 리포트)을 닫는다. 모든 수정은 필터 유닛뿐 아니라 chat과 responses 라이브 라우트에 적용돼 있다.

---

## 1. 문제 정의

### 1.1 배경

#2458 (#2394)은 스트리밍 도구 호출과 마커 정리를 논스트리밍 출력에 맞췄지만 여섯 개의 남은 갭을 본문에 나열했다. #2470은 스모크 테스트에서 두 개를 더 추가했다. 각각은 클라이언트가 스트리밍 여부에 따라 다른 `content`나 `reasoning`을 보게 만든다.

### 1.2 기존 문제점

- Harmony (gpt-oss)가 `--reasoning-format none`에서 스트림은 analysis와 answer를 content로 보낸 반면 논스트리밍 응답은 answer만 반환했다.
- `try_harmony`가 role 헤더의 `to=` 수신자를 무시했다. 스트림은 호출로 억제하고 논스트리밍은 commentary로 취급했다.
- 선행 텍스트 없이 닫힌 호출 뒤의 held whitespace는 스트림에서 `\nB`, 논스트리밍은 `B`로 잘렸다.
- 여러 메시지로 나뉜 Harmony content는 논스트리밍에서 `\n`으로 붙고 스트림에서는 연달아 나왔다.
- MiniMax M3는 no-tools 경로에서 양쪽 모두 네임스페이스 잔여물을 남겼다.
- Kimi-VL `◁think▷` / `◁/think▷` 마커가 양쪽 경로에서 content에 도달했다.
- Llama 3.x `<|python_tag|>` 도구 호출: 논스트리밍은 정확했지만(`finish_reason: tool_calls`, 빈 content) 스트리밍에서는 원본 호출 텍스트가 `delta.content`로 도착하고 같은 호출이 `delta.tool_calls`로 다시 도착했다.
- `/v1/responses`는 output_text 아이템만 `reasoning_tokens: 0`으로 반환하고 reasoning 아이템이 없었던 반면 chat 라우트는 analysis 채널을 `reasoning_content`로 노출했다.

### 1.3 위험성

| 위험 | 영향 | 가능성 |
|------|------|--------|
| 클라이언트가 스트리밍 도구 호출 텍스트를 content로 파싱하고 호출을 이중 실행 | High | High (Llama 3.x 도구 프롬프트에서 결정론적) |
| `none`/`deepseek-legacy`에서 Harmony analysis가 content로 유출 | Medium | High |
| Responses API 클라이언트가 reasoning/usage를 읽을 수 없음 | Medium | High |

---

## 2. 기술적 검토 사항

### 2.1 보안 관점

클린. diff에 공유 상태가 없다 (추가된 모든 줄을 static/once_cell/OnceLock/Mutex/RwLock으로 grep: 없음). 새 StreamFilter 플래그는 요청별로 구성된 필터의 인스턴스 필드이고 `FilterOutput.harmony_reasoning`은 emit별이다. `split_reasoning`의 no-tools Harmony 폴백은 요청 자체 raw 출력에 대해 요청당 최대 1회, max_tokens로 경계 지어지며 결과에서 `.reasoning_content`만 취해 도구 호출을 스머글링할 수 없다. 요청 간 reasoning 유출 없음.

### 2.2 성능 관점

토큰당 O(1) 이상의 비용 없음. `count_reasoning_tokens`는 요청당 정확히 한 번의 추가 tokenizer 인코딩(이벤트당 아님)으로, 요청이 이미 수행하는 디토크나이제이션과 같은 차수다. 새 딜리미터 테이블 6행은 기존 max_delim_len(28바이트)보다 짧아 부분 매칭 윈도우와 버퍼 경계가 불변하며, 피드당 테이블 스캔은 ~41행에 대한 상수 인자 증가다.

### 2.3 호환성/의존성 관점

- **호환성 깨짐**: 없음. 스트리밍 출력이 이제 논스트리밍과 일치하고, `reasoning_tokens`는 completed 이벤트에서 0이 아니며 in-progress 이벤트에서는 0으로 유지된다 (완료 전까지 카운터가 없음).
- **새 의존성**: 없음.

### 2.4 코드 품질 관점

- 테스트: `stream_parity_tests.rs`에 12개 새 parity 테스트 + 4개 헬퍼, `routes/responses.rs`에 2개 새 유닛 테스트. 테스트 가능한 9개의 새 parity 테스트를 프리픽스 트리(6a115916에 테스트만 얹음)에서 실행: 0 통과 / 9 실패, 각 panic이 갭을 직접 보여줌 (python-tag 원본 텍스트가 content로, 페이로드가 `<|eom_id|>`에서 닫히지 않음, role 헤더 불일치, held-whitespace 불일치, Harmony 뉴라인 조인 둘 다, `delta.content`로 analysis 유출, M3 잔여물, Kimi-VL 마커). responses 테스트 2개는 main에 존재하지 않는 헬퍼 위의 유닛 테스트다.
- `clean_content_markers`는 이제 스트리밍 억제와 같은 스팬으로 M3 도구 블록과 python_tag 페이로드를 스트립한다(open부터 close까지 포함; 닫히지 않은 실행은 양쪽 경로에서 텍스트 끝까지).

---

## 3. 기술적 선택과 그 이유

### 3.1 `--reasoning-format none`에서 Harmony는 논스트리밍 응답에 맞춤

**컨텍스트:** Harmony는 표준 thinking-마커 쌍이 없어 `none` 렌더링에 결정이 필요했다.

**선택:** 논스트리밍 응답에 맞춘다 (answer만). analysis 채널은 deepseek 포맷에서 보호되지 않는 `emits_reasoning_content` emission을 통해 `reasoning_content`로 계속 도달한다. 코드 코멘트와 PR 본문에 기록.

### 3.2 /v1/responses에서 chat.rs의 harmony 경로 재사용

**컨텍스트:** `split_reasoning` / `extract_reasoning_from_raw`는 think-마커 형태만 알았고 `reasoning_tokens`는 0으로 하드코딩돼 있었다.

**선택:** `split_reasoning`은 `parsed.reasoning_content`를 먼저, 다음 raw 추출, 다음 `parsed.is_none()`과 raw 채널 마커로 게이트된 no-tools Harmony 폴백을 취하며, 두 번째 harmony 파서를 쓰는 대신 `tool_calls::parse_tool_calls(raw, None).reasoning_content`를 호출한다. `reasoning_tokens`는 추출된 reasoning 텍스트를 모델 토크나이저로 인코딩해 양쪽 경로에서 센다.

### 3.3 베어 JSON Llama 3.x 형태는 스트림에서 잡아둘 수 없음

PR 본문에 요구대로 명시: 완전히 파싱되기 전까지는 보통 JSON 산문과 구별할 수 없고, 스트림은 나중에 content로 내보낼 임의 텍스트를 잡아둘 수 없다.

---

## 4. 구현 상세

### 4.1 주요 코드 변경

**파일: `src/server/tool_calls/stream_filter.rs`**
- `FilterOutput`/`Drained`에 `pub harmony_reasoning: bool`; `StreamFilter`에 `has_emitted_content`, `harmony_content_open`, `harmony_join_pending`, `harmony_reasoning_open`, `harmony_reasoning_join_pending`.
- `CHAT_DELIMITERS`에 Kimi-VL think 쌍, 베어 `]<]minimax[>[` Strip 행, `<|eom_id|>` ExitToolCall, `<|python_tag|>` EnterToolCall.
- `canonical_thinking_pair`에 Kimi-VL 쌍; `emit_prefix`, `emit_content_prefix`, `after_transition`, `flush`가 새 플래그, 억제, 미지급 조인 구분자를 전달.

**파일: `src/server/tool_calls/parser.rs`, `formats.rs`**
- `THINKING_MARKER_PAIRS` 5항목으로 Kimi-VL 포함; `clean_content_markers`가 M3/python_tag 페이로드와 stray 마커 바이트를 스트립.
- `strip_minimax_m3_tools_block`, `strip_python_tag_payload`가 스트리밍 억제와 같은 스팬.

**파일: `src/server/routes/chat.rs`, `router_front.rs`, `in_process/chat.rs`**
- item-1 harmony 가드 `&& !emit.harmony_reasoning`을 싱글노드 라우트와 디스애그리게이티드 라우터의 content re-echo 지점(코디네이터 수정)에, 새 필드의 구성 지점에 적용.

**파일: `src/server/routes/responses.rs`**
- `split_reasoning` 우선순위 체인; `count_reasoning_tokens`를 논스트리밍과 스트리밍 최종 지점에 사용; 인코딩 실패를 디버그 레벨로 로그.

---

## 5. 검증

| 검사 | 결과 |
|---|---|
| 새 parity 테스트 (프리픽스 트리) | 0 통과 / 9 실패 (각 갭 고정) |
| stream_parity | 31 통과 / 0 실패 |
| tool_calls | 474 통과 / 0 실패 |
| server:: | 3562 통과 / 0 실패 / 16 무시 (최종 코드에서 두 번 실행) |
| cargo fmt / clippy | 클린 / 클린 |
| 풀 게이트 (fb7d7a1a) | 161 빈 빌드 0, clippy 0, doc 0, contract 0; 160/161은 알려진 #2467 mllama flake, 재실행 8/8 |
| gpt-oss-20b-mxfp4 /v1/responses (temperature 0) | analysis 텍스트를 담은 reasoning 아이템, reasoning_tokens 44, 스트리밍 == 논스트리밍 |
| meta-llama-3.1-8b-instruct-4bit get_weather (temperature 0) | 스트리밍/논스트리밍 모두: finish_reason tool_calls, content delta 없음, 정확히 한 번의 호출, python-tag/원본 JSON 유출 없음 |
| 보안 검토 | PASS; 공유 상태 없음; 폴백은 요청당 1회 bounded; 요청 간 유출 없음 |

체크포인트 치환은 PR에 명시: models/mlx/gpt-oss-20b-mxfp4 (이슈의 gpt-oss-20b-mxfp4-q4 부재), models/mlx/meta-llama-3.1-8b-instruct-4bit.

---

## 6. 변경 요약

### 통계

| 항목 | 값 |
|------|-----|
| 변경 파일 | 8 (메인 커밋), 2 (후속) |
| 추가 | +587 |
| 삭제 | -494 |

### 관련 커밋

| 해시 | 타입 | 메시지 |
|------|------|--------|
| `fb7d7a1a` | fix | fix(server): close remaining streamed vs non-streamed output gaps |
| `58e19969` | fix | fix(router): drop harmony analysis from content under none placement |
| `b46729b8` | refactor | refactor(server): gate split_reasoning fallback and log encode errors |

---

## 7. 후속 조치

### 모니터링 필요

- 머지 후 첫 릴리스에서 다른 python_tag 생산 패밀리의 스트리밍 도구 호출.
- in-progress Responses 이벤트의 `reasoning_tokens`는 0 유지; 클라이언트는 response.completed에서 읽어야 함.

### 향후 개선 사항

- 스트리밍 reasoning 토큰 카운터는 증분 인코딩이 필요해 보류 (현재 카운터 없음).

---

## 부록

### A. 리뷰 판정

pr-reviewer: 6개 포커스 모두 PASS; 하나의 HIGH (router_front re-echo 지점에 item-1 harmony 가드 누락)는 코디네이터가 `58e19969`으로 수정. pr-security-checker: PASS, CRITICAL/HIGH/MEDIUM 0; LOW 2건은 finalizer가 `b46729b8`으로 채택 (중복 파싱 게이트, 인코딩 오류 로그).
