# PR #1878: FFT 수치 및 plan-cache 검증 도구

**작성일**: 2026-10-15
**상태**: 구현 및 CUDA 검증 완료
**위험도**: 낮음; 독립 진단 예제만 변경

## 요약

#1825와 #1876에서 사용한 임시 FFT 프로그램을 재실행 가능한 mlxcel-core 예제로 만든다. 기본 모드는 실수·복소수 변환을 CPU stream과 비교하며, `--plans N`은 backend cache 한도에서 서로 다른 forward/inverse transform shape를 실행한다. Main `cb73c7d6`와 MLX `243bdf09`를 mlxcelverse 경로에 맞게 통합했다.

## 문제 정의

제출된 stress 모드는 수치 실패를 출력해도 항상 성공으로 종료하여 스크립트가 잘못된 결과를 승인할 수 있었다. 누락되거나 잘못된 plan 수는 조용히 40으로 대체됐고, 0은 아무 작업도 하지 않았으며, 큰 수는 i32 transform 또는 batch 차원을 넘칠 수 있었다. 기존 범위에는 홀수 길이 실수 변환도 없었다.

## 기술적 선택과 변경

두 수치 모드는 공통 비교 집계를 사용한다. 비유한값, 음수 또는 허용 오차 이상의 값은 뒤의 비교가 성공해도 실패로 유지되고 종료 코드 1을 만든다. 엄격한 parser는 잘못된 값, 0, 추가 인자, 산술 범위 초과를 device 작업 전에 종료 코드 2로 거부한다. 상한은 benchmark 예산이 아니라 i32 batch와 length의 곱에서 도출했다. 도움말은 성공으로 종료한다.

길이 3, 7, 401과 홀수 inverse·복소수 변환을 추가하여 35개 비교를 실행한다. CPU stream 기준, 각 device guard 내부의 평가, 결정적 입력, 1e-5 허용 오차를 유지한다. Shape 수와 보유 plan 수를 구분하며 forward/inverse plan은 설정된 cache 한도에 따라 퇴출될 수 있음을 명시했다. Backend feature를 설명하고 빌드 후 runtime timeout을 적용하도록 하여 cold compilation을 과거 ROCm hang으로 오인하지 않게 했다. CPU-only 빌드는 양쪽이 CPU인 smoke test다.

## 검증

NVIDIA GB10에서 CUDA/CPU 비교 35개가 모두 통과했고 보고된 최대 상대 오차는 9.538e-7이었다. 서로 다른 stress shape 40개도 수치 실패 없이 완료됐으며 최대 왕복 오차는 8.946e-7이었다. 두 모드는 종료 코드 0을 반환했다. 실제 subprocess 검증에서 도움말은 0, 누락·0·잘못된 값·범위 초과·추가 인자는 각각 2를 반환했다. 단위 테스트 4개는 parser 경계, 차원 산술, NaN·무한대·허용 오차 경계를 포함한 실패 유지 동작을 검증한다. 경고를 오류로 처리한 전체 CUDA workspace/all-targets Clippy, formatting, license-header, diff 검사도 통과했다.

검증 소스 tree: `590b55dd21d412bd7d071ce6d3ffbd66abd6f673`. Probe SHA-256: `8377fca87228e0d66d87c823735f5a9c1f0d7740745754952062d0ef4f845913`. 이후 추가한 보고서는 해당 프로덕션 소스를 바꾸지 않는다.

원래 PR은 gfx1151에서 5e-7 미만의 오차, cache size 8의 40-shape 완료, size 32의 약 17번째 shape hang을 기록했다. 이는 과거 측정이며 현재 pin의 새 결과가 아니다. 여기에는 AMD 하드웨어가 없어 재현 도구만 제공하며 hipFFT 할당이나 cache 크기 관련 hang의 해결을 주장하지 않는다.

## 변경 요약

단위 테스트를 포함한 실행 예제 하나와 영문·국문 보고서를 추가했다. Runtime backend, 의존성, cache 정책은 바꾸지 않는다. 진단 도구에 남은 blocking 정확성·보안·성능 문제는 발견되지 않았다.

Refs #1825, #1876.
