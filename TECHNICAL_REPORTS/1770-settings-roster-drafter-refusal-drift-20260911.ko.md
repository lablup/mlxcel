# 기술 보고서: PR #1770 - test: settings 스키마 로스터와 drafter 거부 메시지 drift 수정

**날짜**: 2026-09-11
**작성**: mlxcel maintainers
**검토**: main `58dc9693`에서 돌린 머지 후 workspace 게이트
**상태**: 머지 전 (수리한 두 스위트가 각자의 필터에서 통과했고 clippy와 fmt도 통과했다. workspace 게이트는 이 PR이 머지된 뒤 main에서 다시 돌리며 #1769 하나만 보고할 것으로 본다)
**언어**: Rust
**위험도**: 낮음 (연산도 요청 경로도 건드리지 않았다. 로스터 항목 두 개, match 분기 세 개, assertion 부분 문자열 하나, 테스트 이름 하나다)

---

## 요약

머지된 main `58dc9693`에서 2026-09-10 머지 묶음 이후 workspace 게이트(`cargo test --workspace --profile test-fast --features metal,accelerate`)를 돌렸더니 10981개 통과, 3개 실패가 나왔다. 셋 중 둘은 파일을 가로지르는 계약 drift이고 이 PR이 둘 다 수리한다. #1759는 `video_max_frames`와 `video_fps`를 `ServerConfig`에 추가하면서 관리용 settings 스키마가 순회하는 분류 로스터에는 넣지 않았다. #1751은 DFlash drafter 거부 메시지를 고쳐 쓰고 그 메시지를 검사하는 유닛 테스트는 함께 고쳤지만 `tests/` 아래 통합 테스트 하나가 옛 표현의 부분 문자열을 그대로 검사하도록 남겨 뒀다.

두 결함 모두 원인이 된 PR의 diff 안에 있지 않다. 그 diff를 읽고 고른 테스트 범위로는 어느 쪽도 닿지 않는다. 두 PR 모두 머지 시점에 자기 head에서는 초록색이었다. 이 보고서가 다루는 비대칭이 바로 그것이다. PR별 테스트 범위는 그 PR이 바꾼 파일에서 유도되는데 계약 위반은 바꾸지 않은 파일 쪽에 있다.

셋째 실패인 `fp8_block_requantize_matches_direct_path`는 drift가 아니다. 호스트 툴체인 업그레이드가 드러낸 허용오차 유도 문제이고 여기서는 일부러 손대지 않았다. #1769에서 따로 추적한다.

---

## 1. 계약 위반 두 건, 어느 쪽도 diff 안에 없다

### 1.1 `ServerConfig`의 필드는 아직 settings API의 knob이 아니다

`src/server/runtime_settings.rs`는 `GET /v1/settings`와 `GET /settings` 뒤의 관리용 settings 표면을 만든다. `schema()`는 `ServerConfig`를 반사해서 읽지 않는다. 손으로 관리하는 `&[&str]`인 `CLASSIFIED_SERVER_CONFIG_FIELDS`를 순회한다. 이 상수의 doc 주석이 의도를 그대로 적어 두었다. `ServerConfig`에 선언된 모든 필드를 소스 순서대로 담아서 '앞으로 추가되는 필드가 settings 스키마에서 조용히 빠지는 일이 없도록' 한다는 것이다.

일을 하는 낱말은 '조용히'다. `schema()`가 구조체가 아니라 로스터를 순회하므로 로스터에 없는 필드는 어느 층위에서도 오류가 아니다. 그냥 게시되지 않을 뿐이다. `--video-max-frames 8`로 띄운 서버는 운영자가 지정한 플래그를 스키마에도 현재값 맵에도 담지 않은 `GET /v1/settings` 응답을 돌려주고 프로세스는 그 사실을 아무 데도 알리지 않는다.

이를 막는 것이 `server_config_schema_classifies_all_105_fields`다. 이 테스트는 `include_str!`로 컴파일 시점에 `config.rs`를 다시 읽어 `ServerConfig` 선언에서 `pub` 필드 이름을 뽑아낸 다음 세 가지를 검사한다. 선언된 개수, 순서까지 포함해 선언 목록과 같은 로스터, `api_name`을 거친 로스터와 같은 스키마 이름 목록이다. #1759가 선언 개수를 107로 밀어 올리는 동안 로스터는 105에 머물렀고 첫 assertion이 애초에 그러라고 적어 둔 메시지 `ServerConfig field count changed`와 함께 터졌다.

수리는 한 곳이 아니라 네 곳이다. `video_max_frames`와 `video_fps`가 `config.rs`의 선언 위치를 따라 로스터의 `vision_cache_size`와 `lang_bias_config` 사이로 들어가고 `read_only_reason`(둘 다 `SCHEDULER_REASON`), `read_only_value`(둘 다 `json!`), `read_only_kind`(각각 `Int`과 `Float`)에도 들어간다.

### 1.2 오류 산문에 건 assertion, 저자의 범위가 빌드하지 않은 유일한 타깃에서

`src/models/detection.rs`의 `dflash_drafter_not_standalone_error`는 체크포인트 디렉터리가 standalone 모델이 아니라 speculative drafter로 판명될 때 모든 진입점이 함께 쓰는 거부 메시지다. 오프라인 `-m` 경로, 서버 기동, 스테이지 로더가 각자 다른 weight 조회 증상을 뱉는 대신 한 메시지를 주도록 #1170에서 만들었다.

메시지는 두 번 고쳐 썼다. #1751(`7fae0b56`)이 `is a DFlash speculative drafter checkpoint`를 `is a DFlash-family speculative drafter checkpoint (Qwen 3.5 DFlash or LFM2 DSpark)`로 바꿨다. #1762(`fb12866a`)는 Muse Glimmer assistant를 넣으려고 괄호 안 목록을 다시 손봤다.

이 메시지를 부분 문자열로 검사하는 파일은 셋이다. 그중 `src/models/detection_tests.rs`와 `src/commands/generate_tests.rs`는 #1751이 스스로 `DFlash-family speculative drafter`로 옮겨 놓았다. 두 번째 고쳐 쓰기가 괄호 안 목록만 건드렸으므로 이 둘은 #1762의 영향을 받지 않았다. 남은 하나인 `tests/speculative_dispatch.rs:485`는 #1170(`865bede2`)에서 테스트를 쓸 때 받은 표현 `DFlash speculative drafter`를 그대로 들고 있었다. 커밋 434개 전의 표현이다.

부분 문자열을 느슨하게 잡은 것은 메시지를 고쳐 써도 테스트가 깨지지 않게 하려는 의도였다. 그래도 깨졌다. 고쳐 쓰기가 `-family`를 부분 문자열 바깥이 아니라 안쪽에 끼워 넣었기 때문이다. 이번 수리는 이 assertion을 나머지 둘이 이미 쓰고 있는 부분 문자열로 옮긴다. 그 구간은 이제 고쳐 쓰기를 한 번 견뎌 낸 유일한 구간이다.

---

## 2. PR별 범위로는 왜 둘 다 보이지 않는가

두 PR 모두 자기 diff에 맞는 테스트를 돌렸다. #1759는 파일 34개를 건드렸고 그중에 `src/server/config.rs`가 있다. `runtime_settings.rs`와 `runtime_settings_tests.rs`는 없다. #1751은 `src/models/`, `src/lib/mlxcel-core/src/drafter/`, `src/server/batch/`에 걸쳐 파일 29개를 건드렸고 `tests/` 아래는 하나도 없다.

두 누락은 모양이 다르다. 좁은 범위를 어디까지 잡을지 정할 때 이 차이가 중요하다.

**필터 누락.** `runtime_settings_tests.rs`는 `runtime_settings.rs` 안의 `#[cfg(test)] mod tests`다. 즉 이 테스트를 깨뜨린 `config.rs` 변경과 같은 `--lib` 타깃에 있다. diff에서 고른 `--lib server::chat_request`나 `--lib server::startup`, 그 밖의 어떤 모듈 필터도 이 테스트를 제외한다. 필터 없는 `cargo test --lib`이었다면 돌았다. 인자 하나 차이였다.

**타깃 누락.** `tests/speculative_dispatch.rs`는 루트 `mlxcel` 패키지의 별도 통합 바이너리다. `--lib` 호출은 필터가 무엇이든 이 바이너리를 빌드하지 않는다. `CLAUDE.md`의 Test scope 절이 적어 둔 대로 `--workspace` 없는 `--all-targets`도 멤버의 테스트 타깃을 컴파일하지 않고 맨 `cargo test`는 `-p mlxcel`로 풀려 `mlxcel-core`를 아예 빌드하지 않는다. 이쪽은 필터 밖이 아니라 타깃 집합 밖이었다.

어느 쪽도 부주의가 아니다. '내가 무엇을 바꿨나'로 고른 범위는 '내가 바꾼 것을 다른 데서 무엇이 검사하나'를 담을 수 없다. 두 번째 질문의 답이 정확히 첫 번째 질문이 배제한 파일 집합이기 때문이다. 이걸 찾으려면 범위를 정하기 전에 바뀐 심벌을 트리 전체에서 검색하거나 그럴 필요가 없을 만큼 넓게 돌려야 한다.

PR 단위 실행이 폭을 최대로 넓혀도 이 둘을 잡지 못하는 이유가 하나 더 있다. 두 결함 모두 각 PR의 head에는 존재하지 않았다. #1759 브랜치는 자기가 쓰인 시점의 로스터에 대해 정확했고 #1751의 assertion도 자기 base의 메시지에 대해 정확했다. drift는 머지된 상태의 성질이고 머지된 상태는 main에서 처음 생긴다. 머지 전에 PR마다 한 번 도는 게이트는 계약이 아직 성립하는 트리를 재고 있다.

그러면 머지된 상태와 전체 타깃 집합이 만나는 층은 하나만 남는다. 머지 묶음 이후 main에서 도는 workspace 게이트다. 이것은 CI 체크가 아니다. GitHub CI는 Metal 테스트 스위트를 아예 돌리지 않으므로 이번에 수리한 두 스위트는 거기서 한 번도 실행된 적이 없다. 이 게이트가 없었다면 두 결함 모두 릴리스에 실렸을 것이고 그중 눈에 보이는 쪽인 settings 로스터는 같은 릴리스가 CLI와 `docs/llama-server-compat.md`에서 광고하는 knob 두 개를 빠뜨린 관리 API를 내보냈을 것이다.

---

## 3. 셋째 실패는 drift가 아니다

`models::fp8_block::fp8_block_tests::fp8_block_requantize_matches_direct_path`도 `58dc9693`에서 실패하는데 이번에는 손대지 않았다.

이 실패는 flake가 아니라 결정적이다. 단일 테스트를 연속 세 번 돌리면 `element 4: mxfp8 error 0.50390625 exceeded 0.3002931 (group max 4.8046875)`가 글자 하나까지 그대로 재현된다. fixture가 시드로 고정돼 있어 입력 바이트와 bf16 블록 스케일이 어느 호스트에서나 같기 때문이다.

1절에서 말한 의미의 drift 증상도 아니다. 이 테스트는 세 가지를 순서대로 검사하는데 바이트 동일성 검사 둘은 통과한다. `requantize_block_fp8_weights`를 거쳐 나온 mxfp8 평면과 스케일 평면이 직접 양자화한 텐서에 대한 `quantize_weights_with_mode` 결과와 여전히 정확히 일치한다. 실패하는 것은 셋째 assertion이다. MLX 자신의 mxfp8 양자화-역양자화 왕복에 건 정확도 상한이고 테스트 이름이 가리키는 관계가 아니다.

상한은 `group_max / 16.0 + f32::EPSILON`이고 소스 주석은 E4M3의 유효 비트 네 개에 대한 최근접 반올림 half-ulp 논증으로 이를 정당화한다. 관측된 오차는 그룹 최댓값의 `0.50390625 / 4.8046875 = 0.10488`로 half ulp(`2^-4`)와 full ulp(`2^-3`) 사이에 있다. 최근접이 아니라 0 방향으로 반올림하는 양자화기의 서명이다. 2026-09-10 머지들이 한 일에 대한 주장이 아니라 이 백엔드에서의 MLX 연산에 대한 주장이다.

바뀐 것은 트리가 아니라 호스트다. 이 장비는 #1742가 머지된 뒤인 2026-09-10에 macOS 27.0과 Xcode 27로 올라갔고 CI는 이 테스트를 Apple GPU 17세대에서 돌린 적이 없다. 이걸 여기 끼워 넣으면 스위트를 초록으로 만들려고 수치 상한을 넓히는 일이 되는데 `CLAUDE.md`가 'Changes that move the numbers' 항목에서 이름 붙인 실패 양식이 그것이다. #1769는 대신 판단 기준을 먼저 둔다. 같은 시드 fixture를 M1 Ultra에서 돌려 두 호스트가 일치하는지 갈리는지로 이것이 백엔드 발산인지 애초에 성립한 적 없는 상한인지 결정한다. 증거의 기준이 다른 일이므로 PR도 따로 간다.

---

## 4. 이번 수리가 담고 있는 것

### 4.1 등록 지점 네 곳, 소리 내는 곳은 하나

읽기 전용 필드를 settings 표면에 올리려면 `runtime_settings.rs`의 네 곳을 건드려야 한다. 스스로 항의하는 곳은 하나뿐이다.

| 지점 | 결정하는 것 | 새 필드가 빠졌을 때 |
|---|---|---|
| `CLASSIFIED_SERVER_CONFIG_FIELDS` | 필드가 게시되는지 여부 | 조용하다. `schema()`가 로스터를 순회하므로 `GET /v1/settings`와 `current()`에서 필드가 사라진다. |
| `read_only_value` | 게시되는 값 | 소리 낸다. `unreachable!("read-only ServerConfig field missing from schema: {field}")`. 다만 로스터 항목이 있어야 도달한다. |
| `read_only_kind` | 선언되는 타입 | 조용하다. `_ => KnobKind::Str`이므로 `video_max_frames`가 문자열 knob으로 게시된다. |
| `read_only_reason` | patch할 수 없는 이유 | 조용하다. `_ => MODEL_REASON`이므로 스케줄러 knob이 모델 provider 때문에 고정됐다고 주장한다. |

개수 테스트는 첫째와 둘째 행을 덮는다. 둘째는 간접적으로만 덮는다. 이 테스트가 `schema(&ServerConfig::default())`를 만들기 때문에 값 분기가 없는 로스터 항목은 돌아가는 서버가 아니라 테스트 안에서 panic한다. 셋째와 넷째 행은 덮지 않는다. 이름, 순서, 가변성, 유일성, 그리고 읽기 전용 spec마다 빈 문자열이 아닌 이유가 붙어 있는지는 검사하지만 어느 kind인지 어느 reason인지는 한 번도 검사하지 않는다. 이 PR이 두 video 필드에 고른 `Int`과 `Float`은 그래서 정확하되 아무도 검사하지 않는 상태다. 7절의 첫 후속 항목이 이것이다.

### 4.2 개수를 테스트 이름에 둔 이유

`server_config_schema_classifies_all_105_fields`를 상수 세 개와 함께 `..._107_fields`로 바꾸는 것은 잡일처럼 보이지만 그렇지 않다. 105를 107로 세 군데 옮기고 이름을 그대로 두는 diff는 기계적 숫자 갱신으로 읽힌다. 테스트 이름을 바꾸면 새 숫자가 리뷰어가 본문보다 먼저 읽는 한 줄에 들어간다. 대가는 필드를 추가할 때마다 테스트 이름이 바뀐다는 것이고 그 마찰이 의도한 것이다.

---

## 5. 검증

| 게이트 | 결과 |
|---|---|
| `cargo test --profile test-fast --features metal,accelerate --lib server::runtime_settings` | 8개 통과 |
| `cargo test --profile test-fast --features metal,accelerate --test speculative_dispatch` | 22개 통과 |
| `cargo clippy --lib --tests --features metal,accelerate -- -D warnings` | 깨끗함 |
| `cargo fmt --all -- --check` | 깨끗함 |
| PR의 CI | 체크 13개 중 10개 통과, 3개는 해당 없는 경로라 건너뜀 |

통과한 10개 중 어느 것도 이번에 수리한 두 스위트를 돌리지 않는다. GitHub CI가 Metal 테스트 스위트를 돌리지 않기 때문이다. 두 테스트가 통과한다는 증거는 로컬 실행이고 계속 통과한다는 증거는 main에서 돌 다음 workspace 게이트다.

그 게이트는 이 PR이 머지되기 전에는 머지된 결과에 대해 돌 수 없다. #1769가 열려 있으므로 그때 실패는 정확히 하나여야 하고 그 밖의 결과는 2026-09-10 실행이 닿지 못한 세 번째 drift가 있다는 뜻이다.

---

## 6. 변경 요약

### 통계

| 항목 | 값 |
|---|---|
| 변경 파일 | 3 |
| 추가 라인 | 15 |
| 삭제 라인 | 7 |
| 추가된 테스트 | 0 (기존 테스트 둘 수리, 하나 이름 변경) |

### 파일별 변경

- `src/server/runtime_settings.rs`: `video_max_frames`와 `video_fps`를 `ServerConfig` 소스 순서대로 `CLASSIFIED_SERVER_CONFIG_FIELDS`에, `SCHEDULER_REASON` 아래 `read_only_reason`에, `json!` 통과로 `read_only_value`에, `Int`과 `Float`으로 `read_only_kind`에 추가했다.
- `src/server/runtime_settings_tests.rs`: 로스터 테스트 이름을 `server_config_schema_classifies_all_107_fields`로 바꾸고 선언 개수, 스키마 길이, 유일성 assertion을 105에서 107로 옮겼다.
- `tests/speculative_dispatch.rs`: `-m <drafter>` 거부 assertion을 `DFlash-family speculative drafter`로 옮겨 `detection_tests.rs`, `generate_tests.rs`와 맞췄다.

### 관련 커밋

| Hash | Type | Subject |
|---|---|---|
| `fc6b10d4` | test | fix settings-schema roster and drafter-refusal drift |

### 관련 이슈와 PR

Refs #1322, #1339, #1343, #1769. #1759(`41a0adc9`)와 #1751(`7fae0b56`)이 만든 drift를 수리한다. `tests/speculative_dispatch.rs`의 assertion은 #1170(`865bede2`)에서 왔다.

---

## 7. 후속 조치

**읽기 전용 필드의 kind와 reason은 아무도 검사하지 않는다.** `read_only_kind`와 `read_only_reason` 둘 다 기본 분기로 떨어지므로 필드가 틀린 타입이나 틀린 근거로 게시돼도 현재 테스트는 전부 통과한다. 로스터 테스트를 확장해 필드마다 kind와 reason 부류를 고정하면 막히지만 로스터와 나란히 관리해야 할 표가 하나 늘어난다. 더 좁은 판본, 곧 선언하지 않은 읽기 전용 필드가 `KnobKind::Str` 기본 분기로 떨어지지 않는지만 검사하는 쪽이 싸고 이번에 일어난 경우를 잡는다.

**`config.rs`는 로스터를 가리키지 않는다.** 의무는 `CLASSIFIED_SERVER_CONFIG_FIELDS` 한 곳에만 적혀 있는데 필드를 추가하는 사람이 열지 않는 파일이 바로 그 파일이다. `video_max_frames`와 `video_fps`의 필드 doc 주석은 fallback 경로를 충실히 설명하면서 settings 표면은 한 마디도 하지 않는다. `ServerConfig` 구조체 doc에 로스터를 지목하는 짧은 문장을 두면 작업이 시작되는 자리에 안내가 놓인다. `docs/code-guidelines.md`가 `// Used by:` 관례에 대해 대는 근거와 같다.

**오류 산문에 건 assertion에는 로스터가 없다.** `dflash_drafter_not_standalone_error`를 검사하는 파일이 셋인데 무엇도 이들을 그 함수에 이어 주지 않는다. 공유 함수가 호출자를 이름으로 적어 두듯 이 메시지의 doc 주석이 셋을 적어 두면 다음 고쳐 쓰기가 기억해 낸 grep이 아니라 목록을 확인하게 된다.

**머지와 다음 게이트 사이의 창.** 이 결함들은 2026-09-10부터 게이트가 돌 때까지 main에 살아 있었다. 그 창의 폭은 두 PR이 할 수 있었던 무엇이 아니라 게이트를 얼마나 자주 도느냐가 정한다. 좁히려면 main 머지마다 workspace 게이트를 돌리거나 Metal 러너를 CI에 넣어야 하고 후자는 #1769 부류까지 함께 덮는다. #1769 자체도 열려 있고 판단 기준의 M1 Ultra 쪽은 아직 돌리지 않았다.

---

### 옮겨 쓸 만한 교훈

diff에 맞게 정확히 잡은 테스트 범위는 원리상 그 diff가 다른 데서 깨뜨리는 계약을 덮을 수 없다. 이번 실패 둘이 그 원리가 드러나는 두 방식이다. 하나는 맞는 타깃에 있으면서 틀린 필터 아래 있었고 다른 하나는 호출이 아예 빌드하지 않는 타깃에 있었다. 두 저자 모두 변호 가능한 범위를 돌렸는데도 결함 둘이 main에 닿았다. 그들이 위반한 것이 어느 브랜치의 성질도 아니었기 때문이다. 그것은 머지된 트리의 성질이고 어느 브랜치도 그 트리를 갖고 있지 않았다.

그래서 넓은 게이트는 좁은 게이트들에 대한 중복이 아니라 다른 대상을 재는 장치다. 좁은 실행은 변경이 동작하는지 답한다. 머지된 main에서 도는 넓은 실행은 트리가 아직 자기 자신과 맞는지 답하는 유일한 실행이고 운영자가 knob을 설정해 보고 알아채기 전에 API에서 knob 두 개가 빠졌다는 사실을 배우기에 가장 싼 자리다.
