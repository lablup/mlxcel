# 기술 보고서: PR #1495 - 라우터 모드와 models-dir 마이그레이션

**날짜**: 2026-08-28

**상태**: 완료

**언어**: Rust

**위험도**: 높음 (플래그 의미 변경, breaking)

## 요약

b10621 라우터 모드(#1438)를 인프로세스 모델 풀로 구현하고 `--models-dir` 의미 충돌을 해소한다. 두 llama-server 표면(`mlxcel-server`, `mlxcel serve`)에서 이 플래그는 이제 라우터 모드 모델 발견을 선택하고, 기존 mlxcel 저장소 루트 의미는 `--model-store-root`로 이동했으며 충돌 조합은 시작 시 마이그레이션 진단으로 거부된다. 매니페스트 7개 항목이 `supported`로 전환되고 5개는 정직한 경계에서 `deferred`로 남아 이슈 #1438은 열린 채 유지된다.

## 1. 문제 정의

b10621의 `--models-dir`는 디렉터리에서 모델을 발견하고, 적재된 모델마다 자식 llama-server를 띄우고, 요청의 `model` 필드로 프록시하며, `POST /models/load|unload`, `DELETE /models`, `GET /models/sse`, `--models-max`, `--models-autoload`로 집합을 관리하는 라우터 서버를 시작한다. mlxcel은 같은 철자를 로컬 모델 저장소 루트로 썼기 때문에, 복사해 온 llama-server 명령줄이 라우터를 시작하는 대신 다운로더 설정을 조용히 바꿨다. 이는 에픽 #1431이 제거하려는 바로 그 조용한 어긋남 부류다.

## 2. 기술적 결정

### 2.1 자식 프로세스 대신 인프로세스 풀

업스트림이 자식 소켓으로 프록시하는 자리에서, mlxcel 풀 항목은 각각 완전한 `AppState`와 axum 서브앱을 소유하고 디스패처가 재구성한 요청을 인프로세스로 전달한다. 이렇게 하면 기존 라우트 스택 전체를 모델별로 그대로 재사용하면서 HTTP 계약(누락/미지/미적재 이름에 대한 업스트림의 정확한 거부, `?autoload=` 오버라이드, `?model=` GET 프록시)을 유지하고, 언로드는 참조 카운트로 우아해진다. 진행 중 요청이 서브앱을 붙들고 있다가 끝나면 워커 스레드가 채널 단절로 종료되고 가중치가 해제되며, RSS로 확인했다. 서브앱은 CORS 레이어 없이 만들어(`create_app_without_cors`) 라우터 최상위가 preflight 응답과 CORS 헤더를 정확히 한 번만 붙인다. API 키는 동일한 공개 경로 규칙으로 최상위에서 강제한다.

### 2.2 조용한 의미 변경보다 거부가 낫다

`--models-dir`를 모델 인자와 함께 쓰는 것(기존 저장소 루트 용법)은 `--model-store-root`와 `MLXCEL_MODELS_DIR`를 지목하는 진단과 함께 시작을 실패시킨다. b10621이라면 라우터 밖에서 이 플래그를 무해하게 받아들이지만, repo-id를 다른 루트에서 조용히 해석하는 쪽이 거부보다 나쁘다. 이 차이는 숨기지 않고 항목의 divergence로 기록했다. `--models-preset`도 같은 이유로 파싱 후 시작 시 거부한다. 운영자가 INI 프리셋이 적용된다고 믿는 동안 프리셋 없는 모델을 서빙하는 것은 에픽이 금지하는 수용 후 무시 실패다.

### 2.3 요청 시점이 아니라 발견 시점의 격리

모델 이름은 모델 디렉터리 스캔(`config.json`을 가진 직속 하위 디렉터리 하나가 모델 하나)에서만 나온다. canonical 경로가 canonical 루트를 벗어나는 심링크는 스캔에서 건너뛰고, 요청의 `model` 값은 레지스트리를 통해서만 해석되므로 어떤 요청도 경로를 밀어 넣을 수 없다.

### 2.4 단일 모델 `/v1/models` 패리티

단일 모델 `GET /models` / `GET /v1/models` 응답을 b10621 형태로 옮겼다. `aliases`(전체 `--alias` 목록), `tags`(신규 `--tags` / `LLAMA_ARG_TAGS`), `owned_by: "llamacpp"`, `config.json`과 safetensors 헤더에서 유도한 `meta` 사실 블록(양자화 `U32` 페이로드는 선언된 비트 폭으로 언팩해 `n_params` 계산), 그리고 `format: "safetensors"`를 담은 업스트림의 Ollama 호환 `models` 블록이다.

## 3. 변경 요약

| 항목 | 값 |
|------|-----|
| 변경 파일 | 25 |
| 라인 | +2882 / -152 |
| 매니페스트 | supported: --models-max, --models-autoload, --tags, POST /models/load, POST /models/unload, GET /models, GET /v1/models / deferred: --models-dir, --models-preset, POST /models, DELETE /models, GET /models/sse |

검증: 신규 유닛/라우트 테스트 20여 개(심링크 탈출 스킵을 포함한 발견, 거부 패리티, 적재 실패 SSE 이벤트, 관리 라우트 인증, 경로 모양 이름 거부)와 실제 체크포인트 2개로 발견, 자동 적재, 모델 교차 요청, `--models-max 1` LRU 축출, SSE 이벤트를 동반한 load/unload, `?model=` GET 프록시, 생성 도중 발행한 언로드가 진행 중 150프레임 스트림을 끝까지 완료시키는 것, 재시작 복구, 마이그레이션 진단을 확인했다.

## 4. 후속 조치

- INI 프리셋 번역(`--models-preset`), `POST /models` 다운로드 플로, 캐시 소스 `DELETE /models`, SSE 페이로드 패리티(자식 `info`/`progress` 블록)는 #1438에 남는다.
- #1439(멀티 어댑터 LoRA)는 자체 범위대로 어댑터 상태를 복제하지 않고 이 풀과 통합한다.
