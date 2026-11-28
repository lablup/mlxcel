# PR #3489: jiff 1.x 업그레이드와 strftime_now의 형식·시간대 반영 (issue #3480)

**작성일**: 2026-11-27

**상태**: `9d8949cd7`로 머지. 파일 5개, +137 / -44. M3 Ultra에서 로컬 검증했고, 원격 CI는 메인테이너 결정으로 취소했다. Windows 빌드는 돌리지 않았다.

**위험도**: Low (chat template 헬퍼 하나의 출력만 바뀌고, 나머지 timestamp는 바이트 단위로 같다)

## 요약

#3462는 유지보수가 끊긴 `chrono`를 `jiff` 0.2로 바꾸면서 기존 `strftime_now` 동작을 그대로 뒀다. 이 함수는 인자를 무시하고 항상 UTC 날짜를 `%d %b %Y`로 돌려줬다. 이 PR은 `jiff`를 1.0.3으로 올리고, `strftime_now(format)`이 template이 넘긴 형식으로 호스트 시간대의 현재 시각을 포맷하게 한다. `%Y-%m-%d`나 시각을 요구하는 template이 이제 그 값을 받는다.

## 1. 문제 정의

- **형식 인자 무시.** chat template은 system prompt에 오늘 날짜를 넣으려고 `strftime_now(fmt)`를 호출한다. mlxcel이 등록한 함수는 인자를 버리고 `%d %b %Y`를 돌려줬다. Llama 3.1+(`strftime_now("%d %b %Y")`)와는 우연히 맞아서 드러나지 않았지만, `%Y-%m-%d`나 `%d %B %Y`를 요구하는 template은 `2026-11-27`이나 `27 November 2026` 대신 `27 Nov 2026`을 받았다.
- **로컬 시각 대신 UTC.** Python 기준 구현(transformers Jinja 환경의 `datetime.now().strftime(fmt)`)은 로컬 시각을 쓴다. KST 오전 8시의 호스트는 전날 날짜를 받았다.
- **jiff 0.2 고정.** #3462는 cargo cache에 있던 0.2.46을 썼지만 현재 릴리스 라인은 1.0.3이다. 시간대 작업이 나중에 다시 의존성 bump를 요구하지 않도록 최신으로 올리라는 사용자 요청이 있었다.

## 2. 기술적 선택과 그 이유

### 2.1 jiff feature 구성

| Feature | 이유 |
|---|---|
| `std` | `Timestamp::try_from(SystemTime)`, `strtime::format` |
| `tz-system` | `TimeZone::system()`이 호스트 시간대를 찾는다(`TZ`, `/etc/localtime`, Windows registry) |
| `tzdb-zoneinfo` | Unix에서 `/usr/share/zoneinfo`를 읽으므로 DB를 바이너리에 넣지 않는다 |
| `tzdb-bundle-platform` | zoneinfo 디렉터리가 없는 Windows에서만 DB를 번들한다 |

`default-features = false`로 `serde`, `logging`, 항상 번들되는 DB를 뺐다. lockfile에는 모든 플랫폼에서 `jiff-tzdb`, `jiff-tzdb-platform` 항목이 보이지만 Windows target에서만 컴파일된다.

### 2.2 시간대를 명시하는 단일 포맷 함수

로직은 `strftime_in(t: SystemTime, tz: &TimeZone, format: &str)`에 있고, `strftime_now(format)`은 `SystemTime::now()`와 `TimeZone::system()`을 넘기는 한 줄 wrapper다. 테스트는 고정 시각과 고정 `+09:00` 시간대를 넘기므로 호스트 시계나 `TZ`에 의존하지 않는다. `day_month_year`, `day_month_year_now`는 프로덕션 호출처가 없어 지웠고, 기존 `%d %b %Y` 검증은 `strftime_in`을 거친다.

### 2.3 실패 시 동작

| 경우 | 동작 | 이유 |
|---|---|---|
| 호스트 시간대를 알 수 없음 | jiff가 UTC를 쓴다 | 이전 출력과 같고 prompt는 그대로 렌더링된다 |
| jiff 범위를 벗어난 `SystemTime` | `Timestamp::MIN` / `MAX`로 clamp | `try_from` 실패가 렌더링을 막지 않는다 |
| 지원하지 않는 지시자(`%J`) | 형식 문자열을 그대로 돌려주고 debug 로그를 한 번만 남긴다 | template 오타가 요청을 중단시키면 안 되고, `Once`로 hot path 로그 폭주를 막는다 |

glibc와 macOS의 Python `strftime`도 모르는 지시자를 그대로 남기므로 fallback은 기준 구현과 같은 쪽으로 기운다.

## 3. 변경 요약

| 파일 | 변경 |
|---|---|
| `Cargo.toml`, `Cargo.lock` | jiff 0.2.46에서 1.0.3, jiff-core 0.1.8에서 1.0.2, jiff-static 0.2.46에서 1.0.3, jiff-tzdb 1.0.0과 jiff-tzdb-platform 1.0.0 추가(Windows 전용) |
| `src/server/clock.rs` | `strftime_in`, `strftime_now` 추가, `day_month_year*` 삭제, module doc 재정렬 |
| `src/server/chat_template.rs` | `strftime_now`를 형식 인자와 함께 등록 |
| `src/server/clock_tests.rs` | 고정 시각 포맷, `+09:00`에서 23:30 UTC가 다음 날로 넘어가는 경우, 미지원 지시자 |

| 커밋 | 메시지 |
|---|---|
| `2d6beea79` | update(deps): upgrade jiff to 1.0.3 and honour strftime_now format and zone |
| `49afc8b65` | update(server): reflow the clock module doc after the strftime_now change |

## 4. 검증

| 확인 | 결과 |
|---|---|
| `cargo test ... server::clock` | 9개 통과 |
| `cargo test ... server::chat_template` | 167개 통과, 4개 ignored. 새 케이스가 template 경로로 `strftime_now("%Y-%m-%d")`를 렌더링한다 |
| `cargo clippy -p mlxcel --lib --tests --features metal,accelerate -- -D warnings` | 경고 없음 |
| `cargo deny check` | advisories, bans, licenses, sources 통과, 새 예외 없음 |
| `cargo tree -p mlxcel -e normal -i jiff` | `jiff v1.0.3` |

## 5. 후속 조치

- Windows 시간대 경로(`tzdb-bundle-platform`과 registry 조회)는 빌드도 실행도 하지 않았다.
- 서버의 다른 timestamp(Unix 값과 RFC 3339 `Z` 문자열)는 의도적으로 UTC를 유지한다. 호스트 시간대를 따르는 것은 template 헬퍼뿐이다.
- 호스트와 무관하게 고정 시간대가 필요한 배포는 서버 프로세스에 `TZ`를 설정하면 된다. 이를 위한 mlxcel 옵션은 없다.
