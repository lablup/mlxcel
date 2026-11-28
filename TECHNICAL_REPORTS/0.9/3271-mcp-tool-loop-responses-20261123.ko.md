# PR #3271: /v1/responses 서버 측 MCP tool loop

**작성일**: 2026-11-23
**상태**: draft PR (closes #1457). 스크립트로 만든 stdio MCP 서버를 상대로 unit/route 테스트 통과. M3 Ultra에서 `qwen3.5-4b-4bit`로 두 라운드 tool 작업을 streaming, non-streaming, approval 모드로 실행
**위험도**: High (서버가 자신의 권한을 상속하는 자식 프로세스를 띄움)

## 요약

이 PR 이전에는 `/v1/responses`가 `mcp` tool에 400을 반환했고, `--mcp-servers-config` / `--mcp-servers-json`은 #1457을 가리키며 시작 단계에서 실패했습니다. 새 opt-in 플래그 `--mcp-tools`를 주면 `mlxcel-server`가 Cursor 호환 `mcpServers` 문서에 적힌 stdio MCP 서버를 띄우고, 모델이 답하거나 한도에 걸릴 때까지 tool loop(모델 턴, tool 호출, 결과, 다음 턴)를 직접 돌립니다. 클라이언트는 표준 `mcp_list_tools`, `mcp_call`, `mcp_approval_request` output item과 그 streaming 이벤트를 받습니다. `/v1/chat/completions`는 바뀌지 않았습니다.

기본값은 꺼짐이며 시작 단계에서 막습니다. API key 없는 non-loopback listener, router, image-only, 분산 실행을 거부하고, 상속된 `LLAMA_ARG_MCP_SERVERS_*` 환경 변수만으로는 아무것도 띄우지 않습니다.

## 1. 문제 정의

### 1.1 배경

#1435와 #1494는 llama-server b10621의 MCP 플래그를 not applicable로 분류했습니다. upstream은 agent loop를 브라우저 UI에서 돌리고, mlxcel 번들 WebUI에는 MCP 클라이언트가 없기 때문입니다. #1457은 그 대안으로 loop를 서버에서 돌리고 mlxcel이 이미 제공하는 OpenAI Responses built-in tool 계약으로 노출하는 방안을 남겨 두었습니다. 그러면 OpenAI 호환 클라이언트 누구나 별도 UI 없이 tool 실행을 쓸 수 있습니다.

### 1.2 기존 문제점

- `src/server/responses_translator.rs`의 `UNSUPPORTED_TOOL_TYPES`에 `mcp`가 있어 요청이 400으로 실패했습니다.
- `UiCompatArgs::ensure_supported`가 두 MCP 플래그를 시작 단계에서 거부했습니다.
- `src/`에 MCP 서버와 JSON-RPC로 통신하거나, 자식 프로세스를 관리하거나, `mcp_*` item을 만드는 코드가 없었습니다.

### 1.3 위험성

| 위험 | 영향 | 가능성 |
|------|------|--------|
| 네트워크에 노출된 무인증 서버가 누구의 요청이든 tool을 실행 | High | 시작 단계 gate로 차단 |
| 멈추거나 출력이 많은 자식이 worker thread를 막거나 메모리를 소진 | High | timeout, byte 상한, admission control로 제한 |
| 클라이언트가 모델이 만들지 않은 인자에 대한 승인을 위조 | High | 서버에 저장된 인자만 한 번 실행하므로 차단 |

## 2. 기술적 검토 사항

### 2.1 보안 관점

- [x] Opt-in: `--mcp-tools`에는 환경 변수가 없습니다. 이 플래그 없이 서버 설정 플래그만 주면, `LLAMA_ARG_MCP_SERVERS_*`로만 들어온 경우도 포함해 시작이 실패합니다.
- [x] 노출: `validate_mcp_startup`은 API key가 없는 non-loopback TCP listener를 거부합니다. 자식 실행과 모델 로드보다 먼저 돌고 `is_loopback_bind_host`를 재사용합니다. Unix socket과 loopback은 통과하고, IPv6-mapped loopback 주소는 네트워크로 취급합니다.
- [x] 자식 프로세스: argv만 쓰고 shell은 쓰지 않습니다. `env_clear()` 후 allowlist와 항목 자체의 `env`만 넘기므로 서버의 API key가 자식에 전달되지 않습니다. `kill_on_drop`과 명시적 kill/reap을 함께 씁니다.
- [x] 로그: tool 인자와 출력은 기록하지 않습니다. 호출자에게 돌려주는 spawn 오류에는 OS 사유만 담깁니다. `McpServerSpec`의 `Debug`는 수동 구현으로 env key만 출력하고, JSON 플래그에는 `hide_env_values`를 설정했습니다.
- [x] 설정: 엄격하게 파싱합니다. `url`, `headers`, `transport`, stdio가 아닌 `type`, 알 수 없는 key는 시작을 실패시킵니다. 요청 단위 `server_url`, `headers`, `authorization`, `connector_id`는 400입니다.

| 이슈 (리뷰에서 발견) | 심각도 | 상태 |
|-------|----------|--------|
| 거부된 호출이 `tool_rounds`를 올리지 않아 `store:false` 요청이 모델 턴을 무한 반복할 수 있음 | High | 수정 (81169ec4) |
| resume 시 승인된 호출에 `max_tool_calls`가 적용되지 않음 | High | 수정 (81169ec4) |
| tool-loop 응답마다 queue 집계 밖에서 blocking thread를 점유해, 느린 tool이 동시에 몰리면 Tokio blocking pool이 고갈될 수 있음 | High | 수정 (825bcfa9): 동시 실행을 `--max-queue-depth`로 제한(초과 시 503), 서버 lock 대기를 `--mcp-tool-timeout`으로 제한 |
| 분산 실행이 `--mcp-tools`를 조용히 무시 | Low | 수정 (b17449bf): 시작 단계에서 거부 |
| derive된 `Debug`가 자식 env 값을 출력 | Low | 수정 (b17449bf) |
| `max_tool_calls`가 없을 때 턴당 호출 상한 없음 | Medium | 수정 (b17449bf): 턴당 16회 |
| launcher(`npx`, `uvx`) 뒤에서 멈춘 손자 프로세스가 `kill_on_drop` 후에도 남음 | Medium | 미해결, 후속 조치 참고 |
| 승인이 API key 단위로 분리되지 않음 | Low | 수용: 모든 key가 같은 tool 권한을 갖고 response store도 key 단위가 아님 |
| 동시에 들어온 중복 resume이 stream 시작 후에 거부됨 | Low | 수용: 호출 전에 승인이 소비되므로 두 번 실행되지 않음 |

### 2.2 성능 관점

플래그가 꺼져 있으면 비용이 없습니다. translator 경로는 그대로이고 `mcp`는 여전히 400입니다. 켜져 있으면 각 응답이 `spawn_blocking` task 하나에서 돌고, 이런 응답은 동시에 최대 `--max-queue-depth`개까지만 실행됩니다. tool 대기는 호출마다 `--mcp-tool-timeout`(기본 30초)으로 제한되며, 이 값은 서버 시작과 lock 대기에도 적용됩니다.

### 2.3 호환성/의존성 관점

- **Breaking change**: 없음. 이전에 실패하던 플래그가 새 opt-in 뒤에서만 동작합니다.
- **새 의존성**: 없음.
- **Compat manifest**: 두 플래그가 `ui-tools-mcp-gcp` shard에서 `not_applicable`에서 `by_design`으로 바뀌었습니다. `--tools`, `--tools-runtime`, `--ui-mcp-proxy`, `--agent`는 계속 거부됩니다.

## 3. 기술적 선택과 그 이유

### 3.1 별도 opt-in 플래그

| 대안 | 장점 | 단점 |
|--------|------|------|
| 서버 설정 플래그만으로 기능 활성화 | b10621과 동일 | 상속된 `LLAMA_ARG_MCP_SERVERS_*` 변수가 조용히 프로세스를 띄움 |
| **선택: `--mcp-tools`와 서버 설정을 함께 요구** | 프로세스 실행에 항상 명시적 mlxcel 플래그가 필요 | 플래그 하나 추가 |

### 3.2 승인 기본값과 resume

OpenAI를 따라 `require_approval`을 생략하면 `always`입니다. 대기 중인 승인은 모델의 인자를 담은 in-memory registry(4096개)에 있습니다. resume(`previous_response_id`와 `mcp_approval_response`)은 해당 응답의 대기 요청 전부에 답해야 하고, 서버는 저장된 인자를 실행하며 승인을 원자적으로 소비합니다. 재사용, 잘못된 response id, resume 요청이 더 이상 허용하지 않는 tool은 400입니다. 승인을 받을 수 없으면(`store:false`, store 없음, `conversation`) 호출을 거부하고 모델은 계속 진행합니다. 승인은 재시작 후 유지되지 않으며, in-memory `ResponsesStore`와 같습니다.

### 3.3 탐색 실패는 치명적이지 않음

시작하지 못한 서버는 `warn`으로 기록되고 `mcp_list_tools` item(`error`와 `response.mcp_list_tools.failed`)에 보고됩니다. 그 서버를 지정한 다음 요청이 다시 시도합니다. MCP 서버 하나가 고장 나도 추론은 멈추지 않습니다.

## 4. 구현 상세

```
POST /v1/responses with tools[type=mcp]
  -> mcp::request::plan_request   (labels, allowed_tools, approval policy, collisions)
  -> routes::responses_mcp        (admission permit, prepare_generation)
  -> routes::responses_mcp_loop   (model turn -> calls -> results, bounds)
       -> mcp::runtime -> mcp::client (stdio JSON-RPC, caps, respawn)
       -> mcp::approvals          (store / consume pending requests)
  -> one event sequence -> JSON body or SSE stream
```

- `src/server/mcp/`: `config`(엄격한 `mcpServers` 파싱, 시작 gate), `client`(stdio 위 줄 단위 JSON-RPC, byte 상한 stdout, 4 KiB stderr ring), `runtime`(서버별 상태, lazy respawn, admission), `approvals`, `request`.
- `src/server/routes/responses_mcp*.rs`: loop, item 생성, 라운드별 실행. `prepare_response_run`은 본문 변경 없이 새 `prepare_generation`에 위임합니다.
- `src/server/responses_mcp_lowering.rs`: `mcp_call`, `mcp_approval_request`, `mcp_approval_response` 이력 item을 template용 chat 턴으로 변환합니다.
- 한도: `--mcp-max-tool-rounds`(8), `--mcp-tool-timeout`(30초), `--mcp-max-output-bytes`(65536), `max_tool_calls`가 없을 때 턴당 16회. 한도에 걸리면 응답은 `max_tool_rounds`, `max_tool_calls`, `max_output_tokens` 사유로 `incomplete`가 됩니다. 새 `usage.mcp` 객체가 횟수와 적용된 한도를 보고하고, usage는 모든 모델 턴을 합산합니다.

## 5. 학습 포인트

### 5.1 거부된 작업도 loop 한도에 포함

성공한 반복만 세는 loop 한도는 매번 실패하는 입력으로 우회됩니다. 여기서는 승인 경로가 없어 거부된 호출이 `tool_rounds`를 올리지 않아 모델이 그 호출을 끝없이 반복할 수 있었습니다. 실행이든 거부든 tool 결과를 만든 라운드는 모두 세야 합니다.

### 5.2 Blocking task에는 별도 admission control이 필요

외부 I/O를 기다리는 `spawn_blocking` task는 queue depth 집계에 보이지 않습니다. admission은 generation queue에서 추정하지 말고 task를 만드는 지점에서 제한해야 합니다.

## 7. 변경 요약

| 항목 | 값 |
|------|-------|
| 변경 파일 | 52 |
| 라인 | +7252 / -128 |
| 테스트 | MCP unit/route 테스트 약 60개와 두 바이너리의 시작 gate 테스트 |

| Hash | Type | Message |
|------|------|---------|
| `3562fe22` | feat | add server-side MCP tool loop on /v1/responses |
| `81169ec4` | fix | bound refused MCP rounds and resumed approved calls |
| `825bcfa9` | fix | cap concurrent MCP tool-loop runs and server lock waits |
| `b17449bf` | fix | tighten MCP gate, Debug redaction, per-turn call cap |

## 8. 후속 조치

- timeout 시 process group 단위 kill(`process_group(0)`과 `killpg`). 단일 모델 서버에 SIGINT/SIGTERM handler가 먼저 필요하며, 없으면 Ctrl-C가 자식을 더 이상 종료하지 못합니다.
- 명시적 allowlist 뒤의 HTTP/SSE transport와 요청 단위 `server_url`(#1457 범위 밖).
- Responses `tool_choice: {"type":"mcp",...}` 형식은 파싱하지 않습니다. 인자는 각 모델 턴이 끝난 뒤 delta 하나로 전달되며 점진적으로 오지 않습니다.
- Windows: fixture 테스트는 `cfg(unix)`이고 Windows에서는 실행하지 않았습니다.

## 부록

### A. 테스트 결과

- `cargo test --profile test-fast --features metal,accelerate --lib -- server::mcp responses_mcp ui_compat_args coverage_tests`와 시작 테스트: 104개 통과. fixture `tests/fixtures/mcp/fake_mcp_server.py`가 정상, hang, crash, 과대 출력, 잘못된 JSON-RPC, tool 오류, stderr 폭주, 시작 시 hang/crash/garbage를 다룹니다.
- `--test webui_cli_help`, `--test cli_help_consistency`, `--test llama_compat_manifest`, lib/test clippy, `cargo check --bins`, fmt, 저장소 contract script가 통과했습니다.

### B. 실모델 실행 (M3 Ultra, `qwen3.5-4b-4bit`, GPU lock 보유)

`mlxcel-server -m models/qwen3.5-4b-4bit --host 127.0.0.1 --mcp-tools --mcp-servers-json '{"mcpServers":{"vault":{"command":"python3","args":["tests/fixtures/mcp/fake_mcp_server.py"]}}}'`, prompt "Look up the secret numbers stored under the keys alpha and beta, then use the add tool to add them and tell me the sum."

- Non-streamed: `lookup(alpha)=17`, `lookup(beta)=25`, `add(17,25)=42`, 최종 메시지에 42. `usage.mcp.tool_rounds: 2`, `tool_calls: 3`. 세 턴 합계 입력 1515, 출력 290 token.
- Streamed: sequence number가 빠짐없이 이어지는 235개 이벤트, 같은 8개 output item.
- Approval: 기본 정책으로 `mcp_approval_request add {"a":19,"b":23}` 반환, 승인 후 resume에서 42 반환, 같은 답을 다시 보내면 400.
- Non-streamed와 streamed 실행은 최종 head(`b17449bf`)에서 다시 돌렸고 item, usage, 이벤트 수가 같았습니다. approval 실행은 첫 커밋 기준입니다.
