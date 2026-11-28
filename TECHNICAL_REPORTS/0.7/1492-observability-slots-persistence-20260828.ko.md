# 기술 보고서: PR #1492 - props, slots, metrics, health 및 슬롯 영속화

**날짜**: 2026-08-28

**상태**: 완료

**언어**: Rust

**위험도**: 중간

## 요약

b10621 관측 표면을 정렬한다(#1440). 요청 단위 슬롯 레지스트리가 `GET /slots`와 네이티브 `id_slot` 필드를 실제 값으로 채우고, `POST /slots/:id_slot`이 `--slot-save-path` 뒤에서 save/restore/erase를 제공하며, `GET /props`는 b10621 키 집합을 보고하고 `--props`는 `POST /props`만 게이트한다. `GET /metrics`는 `llamacpp:` 메트릭 패밀리를 내보내고, `/health`는 b10621의 두 가지 응답만 낸다. 매니페스트 7개 항목이 `supported`로 전환되고, 4개는 구현 후에도 기록된 divergence와 함께 `deferred`로 남는다. `--sleep-idle-seconds`는 손대지 않았으므로 이슈 #1440은 열린 채 유지된다.

## 1. 문제 정의

mlxcel의 `/props`, `/slots`, `/metrics`, `/health`는 모니터링 스택이 바로 알아차리는 방식으로 b10621과 어긋나 있었다. `/props`는 업스트림이 게이트 없이 여는 경로를 플래그 뒤에 두고 mlxcel 고유 형태를 보고했고, `/slots`에는 요청 단위 슬롯 식별자가 없어 `id_slot`이 항상 `-1`이었으며, `/metrics`는 llama-server 스크레이프 설정이 하나도 매칭하지 못하는 `mlxcel_` 접두 시리즈만 냈고, `/health`는 부하 시 `503 no slot available`을 반환해 liveness 프로브 뒤의 바쁜 서버를 재시작시켰다. 슬롯 영속화는 존재하지 않았다.

## 2. 기술적 결정

### 2.1 스케줄러가 아니라 HTTP 경계의 슬롯 레지스트리

연속 배칭 스케줄러에는 슬롯 개념이 없고, 그것을 관통시키면 모든 워커 유형을 건드리게 된다. 대신 `--parallel`개 슬롯의 `SlotRegistry`를 `AppState`에 두었다. 모든 생성 라우트가 가장 낮은 빈 슬롯을 획득하고, 이미 갖고 있는 콜백에서 진행 상태를 갱신하며, drop 시 해제하되 마지막 작업의 카운터를 유지한다(b10621의 `task_prev` 동작). 슬롯이 모두 바쁘면 언바운드로 시작해(`id_slot: -1`, 업스트림의 센티널) 슬롯이 비면 늦게 바인딩한다. 프롬프트/생성 텍스트는 `LLAMA_SERVER_SLOTS_DEBUG` 또는 `--slot-save-path`가 있을 때만 유지되므로 기본 `/slots`는 요청하지 않은 내용을 유출할 수 없다.

### 2.2 토큰 스트림 영속화, divergence로 기록

b10621의 슬롯 저장은 `llama_state_seq_save_file`로 KV 상태를 직렬화한다. mlxcel의 KV는 HTTP 계층이 건드리면 안 되는 스케줄러 소유 MLX 배열에 있으므로, 저장 파일은 슬롯의 토큰 스트림과 모델 id, 토크나이저 지문을 담는다. `fs_validate_filename`을 규칙 단위로 이식했고, tmp+rename 원자적 쓰기, 양방향 심링크/경로 탈출을 거부하는 canonical 경로 격리를 갖췄다. restore는 토큰을 복원하고 다음 요청은 다시 prefill(또는 프롬프트 캐시 채택)한다. 해당 항목은 지원을 주장하는 대신 이 divergence를 기록하고 `deferred`로 남는다.

### 2.3 health는 정확히 업스트림의 두 응답

기존의 풍부한 health 페이로드는 b10621이 같은 데이터를 보고하는 곳(`/slots`, `/metrics`)으로 옮겼다. `/health`는 준비되면 부하 중에도 `200 {"status": "ok"}`, 그 전에는 업스트림의 `503 Loading model` 엔벨로프만 낸다. 포화 보고는 `GET /slots?fail_on_no_slot=1`로 이동했다.

## 3. 변경 요약

| 항목 | 값 |
|------|-----|
| 변경 파일 | 38 |
| 라인 | +2905 / -722 |
| 매니페스트 | 7개 supported 전환, 4개 구현 후 deferred 유지, POST /completion(s)의 id_slot divergence 해소 |

검증: 신규 유닛/라우트 테스트 40여 개(레지스트리, 심링크/경로/불일치 거부를 포함한 영속화, 게이트 진단, 인증, 레이블 카디널리티 상한), 컴팻 게이트 green, 실제 체크포인트(`qwen2.5-0.5b-4bit`)로 슬롯 0/1 동시 점유와 실시간 카운터, 포화 시에만 503인 `fail_on_no_slot`, 서버 재시작을 가로지르는 save/restore/erase, `Process-Start-Time-Unix` 헤더와 함께 스크레이프되는 `llamacpp:` 패밀리를 확인했다.

## 4. 후속 조치

- `--sleep-idle-seconds`(유휴 슬립과 웨이크업 수명주기)는 #1440에 남는다. 정직한 구현은 워커 수준의 모델 해제/재적재가 필요하며 #1438의 모델 풀 수명주기 위에 세워야 한다.
- `GET /props` / `GET /slots`의 params 키 부분집합 divergence와 슬롯 액션의 KV 영속화 divergence는 #1440에 기록된 채 유지된다.
