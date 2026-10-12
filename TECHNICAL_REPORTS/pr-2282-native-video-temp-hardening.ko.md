# 기술 보고서: PR #2282 - Native-Video 임시 파일 보안 강화

**작성일**: 2026-10-13
**상태**: 리뷰 준비 완료
**언어**: Rust, Markdown
**위험도**: 중간

## 요약

Native-video data URI와 HTTP payload를 요청별 0700 디렉터리 안의 create-new 0600 파일에 저장합니다. Blocking closure가 어떤 바이트도 쓰기 전에 디렉터리 guard를 생성하므로, 요청 취소와 부분 쓰기 실패가 발생해도 writer가 끝난 뒤 private 디렉터리 전체가 제거됩니다.

Native-video 요청 준비는 이제 frame fallback과 동일한 불변 `MLXCEL_VIDEO_DIR_ALLOWLIST` snapshot을 받습니다. 단일 서버, router가 로드한 모델, in-process startup 경로가 모두 이 snapshot을 설치하며, 환경 변수나 디렉터리를 변경한 뒤에는 재시작해야 한다고 운영 문서에 명시했습니다.

## 1. 문제 정의

기존 native-video resolver는 system temporary directory 바로 아래에 기본 권한으로 파일을 쓰고, 비동기 쓰기가 성공한 뒤에야 cleanup guard를 만들었습니다. 따라서 요청이 중단되면 읽을 수 있는 clip이 남을 수 있었고, 쓰기가 실패하면 일부 내용이 남을 수 있었습니다. 또한 요청마다 directory allowlist를 다시 읽고 canonicalize하여 frame fallback과 다른 동작을 보였습니다.

## 2. 기술적 검토 사항

구현은 디렉터리를 0700으로, 파일을 0600과 `create_new`로 만드는 `PrivateTempDir`를 재사용합니다. Blocking closure는 writer를 호출하기 전에 guard를 소유합니다. 대기 중인 future가 중단되면 closure가 완료된 뒤 수신되지 않은 guarded result가 drop됩니다. Payload buffer는 payload 크기만큼 다시 복사하지 않고 closure로 이동합니다.

Startup allowlist를 request preparation에 명시적으로 전달합니다. 리뷰에서 public in-process server가 media capability만 설치하고 allowlist는 설치하지 않는 통합 누락을 발견했으며, 이제 공통 startup helper가 둘을 함께 설치하고 constructor 회귀 테스트가 이 경로를 검증합니다.

## 3. 기술적 선택과 그 이유

### 동기 파일 생성을 blocking pool에서 수행

`PrivateTempDir`는 배타적 생성과 고정된 Unix mode를 보장하기 위해 동기 파일 시스템 연산을 사용합니다. Guard 생성, 쓰기, 결과 반환 전체를 `spawn_blocking`에서 실행하여 Tokio worker를 막지 않으면서 cleanup을 위한 하나의 ownership 경계를 유지합니다.

### Allowlist를 startup 설정으로 취급

Native 경로와 frame fallback의 모든 요청은 모델 startup 중 canonicalize한 directory 목록을 사용합니다. 이후 추가된 디렉터리는 재시작 전까지 허용되지 않으므로, 요청 동작이 경로 생성 시점에 따라 달라지지 않습니다.

## 7. 변경 요약

| 분류 | 수 | 요약 |
|---|---:|---|
| 보안 | 2 | Private 임시 저장소와 취소 안전 cleanup |
| 통합 | 1 | 모든 production server constructor에 startup allowlist 전달 |
| 테스트 | 18 | Temp lifecycle, startup snapshot, in-process constructor, 기존 resolver 검증 |
| 문서 | 2 | Native 및 fallback video 경로의 재시작 요구사항 |

## 부록: 검증 증거

- 구현 source commit: `790d981647901fdb690e223749a0162ed8f632f6`.
- Candidate `mlxcel-server` SHA-256: `fc7c3ebc3f1948fcd2dd9d535428123535a9cc12fd88b2139f576f87f635f9b4`.
- CUDA feature를 사용한 집중 library test가 통과했습니다. Temp lifecycle 3개, request allowlist snapshot 1개, 기존 resolver 13개, in-process startup 1개입니다.
- Qwen2.5-VL 3B 4-bit native data-video 요청이 `The main color in the video is red.`라고 답했습니다. Prompt token은 83개, completion token은 9개였고 private temp root는 요청 전후 모두 비어 있었습니다. Baseline의 답변과 token 수가 같습니다.
- Contract gate는 Python 123개와 WebUI 49개 검증을 통과했습니다. Formatting 및 diff 검사도 통과했습니다.
- 독립 구현, 보안, finalization 리뷰는 in-process constructor 수정을 반영한 뒤 남은 문제를 보고하지 않았습니다.

End-to-end 모델 실행은 Linux CUDA에서 수행했습니다. 이 호스트에서는 macOS Metal을 실행할 수 없었으며 해당 검증은 CI에서 수행합니다.
