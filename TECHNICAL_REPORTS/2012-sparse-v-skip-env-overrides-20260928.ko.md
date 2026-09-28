# 기술 리포트: PR #2012 - measure_sparse_v_skip_rate.sh에서 MLXCEL_BIN, MODEL 환경변수 오버라이드 지원

**작성일**: 2026-09-28
**상태**: 완료
**언어**: Shell
**위험도**: 낮음

## 요약

PR #2012는 `scripts/measure_sparse_v_skip_rate.sh`가 `MLXCEL_BIN`, `MODEL` 환경변수를 기본값 폴백으로 읽도록 하드코딩된 두 리터럴 대입을 교체한다. 이로써 `scripts/bench_block_width.sh` 같은 다른 벤치마크 스크립트와 동일한 관례를 따르게 되어, 파일을 수정하지 않고도 기본 경로가 아닌 릴리스 바이너리나 모델 저장 위치를 대상으로 실행할 수 있다.

## 1. 문제 정의

### 1.1 배경

`scripts/measure_sparse_v_skip_rate.sh`는 일련의 디코드 컨텍스트에서 sparse-V post-softmax attention skip rate(#377)를 측정한다. 여러 자매 벤치마크 스크립트는 이미 `MLXCEL_BIN`을 기본값 폴백과 함께 읽어 다른 빌드 디렉터리를 가리킬 수 있었지만, 이 스크립트는 그렇지 않았다.

### 1.2 기존 문제점

- **하드코딩된 경로**: `MLXCEL`과 `MODEL`이 고정 리터럴(`./target/release/mlxcel`, `models/qwen3-4b-4bit`)로 대입되어 있었다. 기본 `target/release` 트리 밖에서 빌드했거나 기본 Qwen3 체크포인트가 디스크에 없는 기여자는 스크립트를 직접 수정해야 실행할 수 있었다.

### 1.3 위험성 평가

낮음. 이 스크립트는 요청 경로나 CI의 일부가 아닌 로컬 벤치마킹 도구다. 방치하더라도 기여자가 기본값이 아닌 빌드나 모델 저장 위치를 대상으로 이 특정 벤치마크를 실행하려 할 때만 불편이 드러난다.

## 2. 변경 요약

| 항목 | 값 |
|------|-----|
| 변경된 파일 | 1 |
| 추가된 줄 | 2 |
| 삭제된 줄 | 2 |

- `MLXCEL="./target/release/mlxcel"`가 `MLXCEL=${MLXCEL_BIN:-./target/release/mlxcel}`로 바뀐다.
- `MODEL="models/qwen3-4b-4bit"`가 `MODEL=${MODEL:-models/qwen3-4b-4bit}`로 바뀐다.
- 기존 위치 인자 파싱 루프(`*) MODEL="$1"; shift ;;`)는 그대로 유지되며 기본값 대입 이후에도 계속 실행되므로, 위치 인자로 넘긴 모델은 여전히 `MODEL` 환경변수보다 우선한다.
- 두 변수 모두 설정하지 않았을 때의 기본 동작은 변하지 않는다: 이전과 동일한 기본 경로에 대해 동일한 `mlxcel not found` 오류가 발생한다.

## 3. 기술적 선택과 그 이유

### 3.1 새 플래그가 아닌 `${VAR:-default}` 파라미터 확장

**컨텍스트**: 이 스크립트는 이미 위치 인자로 받는 모델 경로와 `--contexts` / `--decode-tokens` / `--outdir` 플래그를 지원한다. 대안인 스크립트 전용 `--bin` 플래그는 환경변수를 쓰지 않고도 바이너리 경로를 설정 가능하게 만들었을 것이다.

**선택 이유**: `scripts/bench_block_width.sh` 등 자매 스크립트들은 이미 `MLXCEL_BIN`을 비기본 바이너리를 가리키는 저장소 관례로 확립해 두었다. 동일한 셸 파라미터 확장 관용구로 같은 변수를 재사용하면, 이 스크립트에만 존재하는 플래그를 추가하는 대신 스크립트 전반에 걸쳐 관례를 일관되게 유지할 수 있다.

**트레이드오프**: 환경변수는 명시적 플래그보다 설정된 채 잊어버리기 쉽지만, `scripts/` 안의 다른 곳에서 이미 쓰이는 패턴과 일치하므로 다른 벤치 스크립트에 익숙한 기여자는 이 스크립트에서도 같은 오버라이드를 별다른 학습 없이 쓸 수 있다.

## 4. 검증

- `bash -n scripts/measure_sparse_v_skip_rate.sh`와 `shellcheck`: 통과.
- `MLXCEL_BIN` 스텁 바이너리와 가짜 모델 디렉터리로 확인: `MLXCEL_BIN`이 적용되고, `MODEL` 환경변수 오버라이드가 적용되며, 위치 인자로 넘긴 모델이 여전히 `MODEL`보다 우선한다.
- 변수 미설정 시 동작이 변하지 않음을 확인: 이전과 동일한 기본 `./target/release/mlxcel` 경로에 대해 동일한 `mlxcel not found` 오류가 발생한다.
- 실행하지 않은 항목: 실제 모델 체크포인트를 대상으로 한 벤치마크 실행 (공유 GPU 환경이라 이 환경변수 배선 변경의 범위를 벗어남).

## 5. 관련 작업

- 이슈 #1661: 이 변경의 근거가 된 이슈.
- 이슈 #377: `scripts/measure_sparse_v_skip_rate.sh` 자체를 도입한 이슈.
- `scripts/bench_block_width.sh`: 이 PR이 따른 `MLXCEL_BIN` 패턴을 가진 자매 스크립트.
