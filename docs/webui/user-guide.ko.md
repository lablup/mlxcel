# 번들 WebUI 사용하기

[English](user-guide.md) | [WebUI 문서 목록](README.md)

`mlxcel-server --webui`와 `mlxcel serve --webui`는 서버 바이너리에서 직접 브라우저 인터페이스를 제공합니다. 모델 라이브러리(모델), 로컬 대화 클라이언트(대화), 런타임 관측(활동), 범위별 설정(설정)으로 구성됩니다. 인터페이스는 바이너리에 컴파일되어 있으므로 Node.js, CDN, 그리고 직접 실행한 서버 외의 네트워크 접근이 필요하지 않습니다. 명시적으로 켜야 하는 기능이며 아직 실험 단계입니다. `--webui` 없이 실행하면 서버는 이전과 똑같이 HTTP API만 제공합니다.

WebUI는 관리자용 화면입니다. 키를 가진 사람은 누구나 그 서버에서 모델을 로드, 언로드, 다운로드, 삭제할 수 있으므로 키를 모델 서버의 루트 비밀번호처럼 다루세요.

## 빠른 시작

아래 예시는 따로 적지 않는 한 기본 루프백 주소(`127.0.0.1:8080`)에 바인드합니다. `mlxcel serve`는 `mlxcel-server`와 같은 플래그를 받으므로 모든 `mlxcel-server --webui ...` 명령은 `mlxcel serve --webui ...`로 그대로 바꿔 실행할 수 있습니다.

### 생성된 키로 로컬 세션 시작

대화형 터미널에서 서버를 실행합니다.

```bash
mlxcel-server --webui --models-dir ~/models/mlx
```

API 키가 설정되지 않았으므로 서버는 재시작하면 사라지는 세션 키를 만들어 제어 터미널에 한 번만 출력합니다.

```text
mlxcel WebUI session key (shown once): mlxcel_webui_<generated>
```

`http://127.0.0.1:8080/webui/`를 열고 **세션 키**에 키를 붙여 넣은 뒤 **연결**을 누릅니다. 키는 재시작할 때마다 바뀌며 파일, 로그, URL 어디에도 기록되지 않습니다.

### 키 파일 사용 (서비스, 스크립트, 비대화형 셸)

터미널 없이 시작한 서버(서비스 관리자, 컨테이너, CI 작업)는 생성된 키를 보여 줄 수 없으므로 명시적인 키가 필요합니다.

```bash
mkdir -p ~/.config/mlxcel
(umask 077 && openssl rand -hex 32 > ~/.config/mlxcel/webui.key)
mlxcel-server --webui --api-key-file ~/.config/mlxcel/webui.key --models-dir ~/models/mlx
```

파일에는 한 줄에 키 하나를 적으며 `#`로 시작하는 줄은 주석입니다. 로그인 화면에는 파일의 키를 붙여 넣습니다. 같은 키가 일반 API 클라이언트에도 쓰이므로 파일은 본인만 읽을 수 있게 두세요. `--api-key <KEY>`도 동작하지만 키가 명령줄에 남아 다른 로컬 사용자가 프로세스 목록에서 볼 수 있습니다.

### 라이브러리에 표시할 모델 고르기

라이브러리는 세 곳에서 체크포인트를 모읍니다. 이름이 겹치면 관리 모델 저장소, `--models-dir` 디렉터리, `--models-preset` INI 파일의 섹션 순서로 뒤쪽이 우선합니다.

| 명령 | 라이브러리 내용 |
|---|---|
| `mlxcel-server --webui --models-dir ~/models/mlx` | `~/models/mlx` 바로 아래에서 `config.json`을 가진 모든 하위 디렉터리와 관리 저장소 |
| `mlxcel-server --webui` | 관리 저장소 `${MLXCEL_CACHE_DIR:-$HOME/.cache/mlxcel}/models`만. 저장소가 아직 없으면 오류가 아니라 빈 라이브러리입니다. |
| `mlxcel-server --webui --model-store-root /data/mlxcel-models` | 명시한 경로의 관리 저장소. **모델 추가**로 받은 모델이 이곳에 저장됩니다. 명시한 경로를 읽을 수 없으면 시작이 중단됩니다([문제 해결](#문제-해결) 참고). |
| `mlxcel-server --webui -m ~/models/mlx/qwen3-0.6b-4bit` | 단일 모델 모드. 지정한 체크포인트 하나를 시작할 때 로드합니다. 로드, 언로드, 모델 추가, 삭제는 읽기 전용이며, 모델 상세 패널에 서버가 보낸 사유("single-model mode does not own the router lifecycle pool" 또는 "... the router managed cache")와 `-m`/`--model` 없이 재시작하라는 안내가 함께 표시됩니다. |

`--models-dir`나 관리 저장소를 쓸 때 서버는 아무 모델도 로드하지 않은 상태로 시작하며, 암묵적으로 모델을 로드하지 않습니다. 탐색, 필터링, 모델 선택, 활동 화면 열기, 로드되지 않은 모델로 대화 요청 보내기 모두 모델을 로드하지 않습니다. 모델은 **로드**를 눌렀을 때만 로드됩니다. 예외는 두 가지이며 모두 명시적입니다. 단일 모델 모드는 `-m` 체크포인트를 시작할 때 로드하고, `load-on-startup = true`인 프리셋 섹션은 서버가 서비스를 시작하기 전에 로드를 시작합니다.

### 오프라인 사용

```bash
mlxcel-server --webui --api-key-file ~/.config/mlxcel/webui.key --models-dir ~/models/mlx --offline
```

`--offline`(또는 `LLAMA_ARG_OFFLINE`)은 서버의 네트워크 접근을 막습니다. 로컬 체크포인트의 로드와 대화는 그대로 되고, **모델 추가**는 "offline mode is enabled" 사유와 함께 비활성화됩니다. WebUI 자체는 네트워크가 필요 없습니다. 스크립트, 스타일, 라이선스 문서가 모두 바이너리에 들어 있습니다.

### 실시간 설정, 메트릭, 슬롯

WebUI는 선택형 서버 엔드포인트를 자동으로 켜지 않습니다. 보고 싶은 항목을 서버 시작 시 켜세요.

```bash
mlxcel-server --webui --api-key-file ~/.config/mlxcel/webui.key --models-dir ~/models/mlx --settings --props --metrics
```

`--settings`는 설정의 **모델** 탭에서 샘플링 값을 실시간으로 바꾸는 기능을, `--props`는 **서버** 탭의 관측된 컨텍스트를, `--metrics`는 활동 화면의 요청 카운터를 켭니다. 슬롯 점유율은 기존 `--slots` 기본값(켜짐)을 따릅니다. 플래그가 없으면 해당 패널은 엔드포인트가 꺼져 있거나 값을 얻을 수 없다고 알려 주며, 그 자리에 0을 표시하지 않습니다.

### 원격 접근

다른 컴퓨터에서 접근할 수 있는 WebUI는 명시적인 키와 TLS를 써야 합니다. 지원하는 구성은 두 가지입니다.

**서버에서 직접 TLS 사용.** WebUI는 `Host`가 바인드한 주소와 일치하는 요청만 받으므로 브라우저가 사용할 주소에 바인드합니다.

```bash
mlxcel-server --webui --host 192.0.2.10 --port 8443 \
  --ssl-cert-file /etc/mlxcel/tls/cert.pem --ssl-key-file /etc/mlxcel/tls/key.pem \
  --api-key-file /etc/mlxcel/webui.key --models-dir /srv/models
```

브라우저에서 `https://192.0.2.10:8443/webui/`로 접속합니다. `0.0.0.0`처럼 브라우저가 쓰지 않는 주소에 바인드하면 모든 UI 요청이 호스트 검사에서 거부됩니다.

**TLS 리버스 프록시 뒤의 루프백 백엔드.** 서버는 루프백에 두고 프록시가 TLS를 종료합니다.

```bash
mlxcel-server --webui --host 127.0.0.1 --port 8080 --api-prefix /lab \
  --api-key-file /etc/mlxcel/webui.key --models-dir /srv/models
```

프록시는 `/lab/`을 `http://127.0.0.1:8080/lab/`으로 전달하고, 업스트림에 `Host: 127.0.0.1:8080`을 보내며, 브라우저의 `Origin` 헤더를 `http://127.0.0.1:8080`으로 바꾸고(브라우저가 보내지 않았으면 추가하지 않음), `Authorization`은 그대로 전달하고, `text/event-stream` 응답을 버퍼링하지 않아야 합니다. 서버는 프록시가 보낸 값만 검사할 수 있으므로 공개 호스트와 오리진 검증은 프록시가 직접 해야 합니다. 이 경우 WebUI 주소는 `https://<공개 호스트>/lab/webui/`입니다. 이 헤더 규약을 그대로 검증하는 기준 구현은 `scripts/webui/verify_reverse_proxy.py`(`make verify-webui-reverse-proxy`)입니다.

`--api-prefix`는 API와 WebUI를 그 경로 아래로 옮기고(`/lab/webui/`, `/lab/ui-api/v1/...`), `/`의 상태 확인은 루트에 남아 키 없이 응답합니다.

## 로그인과 키

- 붙여 넣은 키는 페이지 메모리에만 있습니다. 쿠키, 로컬 스토리지, IndexedDB, URL에 저장되지 않으므로 새로고침하거나 새 탭을 열면 다시 입력해야 하고, **키 지우기**를 누르면 즉시 잊힙니다.
- 모든 UI API 요청은 키를 `Authorization: Bearer` 헤더로 보냅니다. 키 없는 요청은 `401 Unauthorized`를 받습니다. 셸 페이지와 정적 파일은 공개되지만 그 뒤의 데이터는 공개되지 않습니다.
- 서버가 키를 거부하면 "세션 키가 거부되었습니다. 로컬 서버 터미널에 표시된 최신 키를 복사하세요."가 표시됩니다. 생성된 키는 서버를 재시작하면 더 이상 동작하지 않으므로 새 키를 복사하세요.
- 키를 링크, 스크린샷, 버그 리포트에 넣지 마세요. 이 문서의 어떤 예시에도 실제 키는 없습니다.

## 모델: 지원, 다운로드, 로드

모델 화면은 혼동하기 쉬운 사실을 분리해서 보여 줍니다.

- **다운로드됨 / 완전함**은 디스크의 파일 상태입니다. 완전한 체크포인트는 기대한 파일 구성을 갖췄다는 뜻이지, 로드에 성공한다는 증거는 아닙니다.
- **지원**은 mlxcel 아키텍처 레지스트리가 체크포인트를 인식한다는 뜻이고, **실행 가능**은 그 아키텍처가 이 서버의 GPU 백엔드(Metal, CUDA, ROCm)에서 지원된다는 뜻입니다. 둘 다 실제 추론 결과를 측정한 것은 아닙니다.
- **로드됨 / 준비됨**은 서버 모델 워커의 현재 상태입니다. 준비된 모델만 대화 요청에 응답합니다.

모델 화면이나 대화 화면에서 모델을 고르는 것은 선택일 뿐입니다. **로드**는 서버 작업을 시작합니다. 버튼이 응답했다는 것은 요청이 접수되었다는 뜻이지 모델이 준비됐다는 뜻이 아니며, 배지는 로드 중을 거쳐 준비됨(또는 사유가 붙은 실패)으로 바뀝니다. `--models-max`에 도달하면 로드는 어떤 로드된 모델을 내릴지 묻고, 확인한 모델만 내립니다.

**언로드**는 그 모델로 들어오는 새 요청을 막고, 진행 중인 요청이 끝나기를 기다린 뒤(드레인) 워커를 해제합니다. 서버가 워커 종료를 확인하기 전까지 그 메모리는 사용 중으로 계산됩니다. 브라우저 탭을 닫아도 모델은 언로드되지 않습니다.

**삭제**는 관리 저장소에서 체크포인트 파일을 지웁니다. 언로드와는 다른 동작입니다. 언로드는 파일을 남기고 삭제는 파일을 지웁니다. 관리 저장소 항목만 삭제할 수 있고, `--models-dir` 디렉터리나 프리셋 경로, 로드 중이거나 로드됨, 다운로드 중, 드레인 중인 모델은 삭제할 수 없으며, 확인 창에 모델 이름이 아니라 창에 표시된 모델 ID(`mdl_`로 시작)를 입력해야 합니다.

**모델 추가**는 공개 Hugging Face 저장소를 `owner/name` ID로 관리 저장소에 내려받습니다. URL과 로컬 경로는 요청을 보내기 전에 거부됩니다. 다운로드는 익명으로 이뤄집니다. `HF_TOKEN`이 설정되어 있어도 서버는 Hugging Face 토큰을 보내지 않으므로 비공개 또는 접근 제한 저장소는 파일을 쓰기 전 메타데이터 확인 단계에서 실패합니다. 비공개 Hub 인증은 이번 릴리스에 포함되지 않습니다. 다운로드는 취소할 수 있으며, 취소는 다운로더가 멈춘 뒤에야 완료되고 일부만 받은 파일이 완료로 표시되지 않습니다. 로드와 언로드는 접수된 뒤에는 취소할 수 없습니다.

기능(capability)이 동작을 제한합니다. 대화 모델 선택기는 대화가 가능한 모델만 보여 주고, 이미지 첨부는 로드된 모델이 비전 입력을 확인한 뒤에만 나타나며, 다른 작업(임베딩, 재순위, 오디오, 이미지 생성)은 대화 버튼이 아니라 기능과 API 예시로 표시됩니다.

## 대화, 설정, 개인정보

설정은 값이 얼마나 유지되고 누구에게 영향을 주는지에 따라 나뉩니다.

| 위치 | 범위 | 유지 기간 |
|---|---|---|
| 설정 **화면 표시** | 이 브라우저 | 브라우저 환경설정(테마, 글래스, 모션, 대비). 서버로 보내지 않습니다. |
| 설정 **요청**, 대화의 **다음 턴 매개변수** | 내가 보낼 요청 | 메모리에만 있습니다. 요청마다 함께 보내며 로드된 모델이나 다른 사용자에게 영향을 주지 않습니다. |
| 설정 **모델**의 실시간 값 | 선택한 로드된 모델, 모든 사용자 | 서버의 `/settings`로 적용합니다(`--settings` 필요). 새로 접수되는 요청부터 적용되며, 일부만 적용되면 거부된 필드를 알려 줍니다. |
| 설정 **모델**의 다음 로드 프로필 | 그 모델의 다음 로드 | 이 브라우저에 저장됩니다. `ctx_size`, `n_parallel`, KV 캐시 모드만 가능하며, 서버 명령줄에서 명시한 플래그가 우선합니다. |
| 설정 **서버** | 실행 중인 서버 | 읽기 전용 시작 값입니다. 바꾸려면 재시작해야 합니다. |

대화 기록은 기본적으로 메모리에만 있고 새로고침하거나 로그아웃하면 사라집니다. 대화 설정 서랍의 **로컬 기록과 개인정보**에서 **이 기기에 대화 저장 (IndexedDB)**을 켜면 이 오리진의 브라우저 IndexedDB에 대화를 저장합니다. 새로고침한 뒤에는 다시 켜야 저장된 데이터를 읽습니다. 저장을 끄면 새로 쓰지 않을 뿐 기존 데이터는 남으며, **모두 지우기**가 저장된 대화와 메모리의 대화를 함께 지웁니다. **JSON 내보내기**는 직접 저장할 파일을 만들고, 가져오기는 파일을 검증한 뒤 현재 기록을 바꾸기 전에 확인을 받습니다. 이미지는 별도로 동의한 경우에만 저장됩니다. 설정은 키와 시스템 프롬프트를 저장하지 않습니다.

WebUI는 원격 측정 데이터를 보내지 않고 제3자 서버에 접속하지 않습니다. 요청은 WebUI를 제공한 서버로만 갑니다. 모델 출력은 신뢰하지 않는 텍스트로 렌더링합니다(원시 HTML, 원격 이미지 없음). 모델이 만든 도구 호출은 실행되지 않았다는 표시와 함께 보여 주며, WebUI와 서버 모두 실행하지 않습니다. **중지**는 스트림을 끝내고 부분 응답을 남깁니다. 서버에서 생성이 이미 멈췄다는 보장은 아닙니다.

## 이번 릴리스에 없는 것

비공개 Hugging Face 인증, 네이티브 데스크톱 패키징, 도구와 MCP 실행, 대화 외 작업 전용 화면은 숨겨진 버튼이 아니라 문서화된 제한입니다. 설치된 산출물에 대한 VoiceOver 스크린 리더 점검은 아직 하지 않았고, Safari의 키보드, 확대, 화면 점검은 마쳤습니다. 남은 항목과 범위는 [통합 매트릭스](../webui-integration-matrix.md)에 기록되어 있습니다.

## 문제 해결

아래 메시지는 서버가 시작을 거부할 때 `Error: ` 접두어와 함께 그대로 출력됩니다. 서버 메시지는 영어로만 출력됩니다.

| 증상 | 메시지 | 해결 |
|---|---|---|
| API 키가 없고, 대화형 터미널이 없거나 루프백이 아닌 주소 사용 | `--webui without --api-key is only allowed on an interactive loopback terminal so the generated session key can be shown once` | `--api-key-file`을 지정하거나(루프백이 아니면 항상 필요), 루프백 서버를 터미널에서 시작합니다. |
| Unix 소켓에서 대기 | `--webui requires a TCP listener; use a loopback TCP listener behind a reverse proxy instead of a Unix socket` | `--host 127.0.0.1`과 리버스 프록시를 사용합니다. |
| 키는 있지만 TLS 없이 루프백이 아닌 주소 사용 | `non-loopback --webui requires an explicit API key and TLS; alternatively bind the WebUI backend to loopback behind a TLS reverse proxy` | `--ssl-cert-file`/`--ssl-key-file`을 추가하거나, 루프백에 바인드하고 프록시를 둡니다. |
| `--models-dir`가 없거나 읽을 수 없음 | `--models-dir /srv/missing: cannot open models directory: No such file or directory (os error 2)` | 존재하는 디렉터리를 `--models-dir`에 지정합니다. |
| 명시한 저장소 경로가 없거나 읽을 수 없음 | `--model-store-root /srv/missing: explicit WebUI/router cache root must be a readable directory: No such file or directory (os error 2)` | 디렉터리를 만들거나 경로를 고칩니다. `MLXCEL_MODELS_DIR`도 같은 검사를 받습니다. |
| WebUI 없이 빌드된 바이너리 | `` --webui requires a binary built with the Cargo `webui` feature; rebuild mlxcel with bundled WebUI support `` | 공식 빌드나 `--no-default-features` 없이 만든 빌드를 사용합니다. |
| llama-server UI 플래그 사용 | `--ui-config is not supported: it configures llama-server's web UI; mlxcel's bundled WebUI does not consume upstream UI JSON.` | 플래그를 지웁니다. `--path`, `--tools`, `--tools-runtime`, `--mcp-servers-config`, `--mcp-servers-json`, `--ui-mcp-proxy`, `--agent`도 같은 방식으로 거부됩니다. |

브라우저에서는 다음과 같이 보입니다.

| 화면 | 의미 |
|---|---|
| 모든 화면에 "인증이 필요합니다" | 페이지 메모리에 키가 없습니다. 키를 다시 붙여 넣으세요. |
| "세션 키가 거부되었습니다. ..." | 키가 틀렸거나 재시작 전에 생성된 키입니다. |
| "로컬 WebUI API에 연결할 수 없습니다. mlxcel-server가 --webui로 계속 실행 중인지 확인하세요." | 서버가 멈췄거나 연결할 수 없습니다. |
| "키 인증은 되었지만 현재 서버 세션에서 이 UI 엔드포인트가 금지되어 있습니다." | 서버가 요청의 `Host`, `Origin`, fetch metadata를 거부했습니다. 서버가 바인드하지 않은 주소로 페이지를 열었거나, 리버스 프록시가 위에서 설명한 대로 `Host`와 `Origin`을 바꾸지 않았습니다. |
| 지표가 "알 수 없음"이나 "아직 측정되지 않음" | 값을 얻을 수 없습니다(`--metrics` 없음, 로드된 모델 없음, 아직 표본 없음). 0으로 표시하지 않습니다. |

키 없이 UI 엔드포인트에 요청하면 HTTP 401과 다음 본문을 받습니다.

```json
{"error":{"code":"unauthorized","message":"authentication is required","retryable":false},"request_id":"req_<id>"}
```

서버가 바인드하지 않은 호스트 이름이나 주소로 들어온 UI 요청(예: `0.0.0.0`에 바인드한 서버를 `https://192.0.2.10:8443/webui/`로 연 경우)은 HTTP 403을 받습니다.

```json
{"error":{"code":"forbidden","message":"Host header is not allowed","retryable":false},"request_id":"req_<id>"}
```

## 릴리스 근거

[통합 요구 사항 매트릭스](../webui-integration-matrix.md)가 번들 WebUI에서 무엇을 어떻게 검증했는지에 대한 기준 기록입니다. 그 숫자를 다른 문서에 옮기지 말고 해당 행을 링크하세요.

- **번들 소스 해시와 크기:** `src/webui/assets/mlxcel-webui-manifest.json`이 소스 다이제스트, 파일별 SHA-256, gzip 크기를 [번들 WebUI 개발](bundling.ko.md)의 예산과 함께 기록합니다. 이번 릴리스의 CI 재현성 검사, 다이제스트, 측정 크기는 매트릭스의 "Bundle budgets and deterministic assets" 행에 기록되어 있습니다.
- **오프라인 이동 바이너리 점검:** 매트릭스의 "Installed artifact, not Vite-only" 행에 위치를 옮긴 바이너리와 외부 네트워크 차단 근거(`integration-1848/offline-52103-pass`)가 있습니다.
- **릴리스 feature 포함:** `webui` Cargo feature는 기본으로 켜져 있으므로 `--features metal,accelerate`와 `--features cuda` 릴리스 빌드는 번들을 포함합니다. 같은 행이 feature를 켠 산출물과 끈 산출물을 모두 다룹니다.
- **런타임 의존성:** WebUI를 포함해도 바이너리가 자기 완결형이 되지는 않습니다. 플랫폼 GPU 런타임은 계속 필요합니다. macOS 릴리스가 함께 패키징하는 `mlx.metallib` 산출물, 또는 Linux의 CUDA와 cuDNN 런타임 라이브러리가 그것이며 [installation.md](../installation.md)에 설명되어 있습니다.
