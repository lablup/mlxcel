# PR #2614: 백엔드 오버레이를 보존하며 MLX pin 업데이트

**작성일**: 2026-10-26
**상태**: 구현 완료; CPU·GB10 검증, 다른 하드웨어 검증 대기
**언어**: C++, CUDA, CMake
**위험도**: 중간

## 요약

MLX pin을 `bff10220`에서 `7ca40033`으로 올리고 로컬 백엔드 동작을 유지했다. upstream 22개 커밋에는 Metal 소규모 배치 커널, CUDA 대형 expert offset 수정, GB10 single-root CUDA graph가 포함된다. mlxcel 처리량 개선 폭은 측정하지 않았다.

## 1. 문제 정의

기존 pin도 정식 v0.32.4 이후이므로 해당 릴리스로 바꾸면 다운그레이드가 된다. SHA만 바꾸면 파일 전체를 덮는 오버레이가 upstream 변경을 누락시킬 수 있고, SDPA의 window-size 인자 추가로 C++ 호출부가 깨진다.

## 3. 기술적 선택과 그 이유

main의 고정 커밋을 선택하고 각 오버레이의 명시된 기준 버전에서 병합했다. CPU·Metal·CUDA 33개 파일을 대조했으며 Windows loader 통합 외에는 변경된 파일의 로컬 차이가 그대로 유지된다. Windows에서는 제한된 의존 DLL 로딩을 보존하면서 upstream의 버전별 경로와 예외 처리를 반영했다. Metal QMV 제어·gather 행 수 보정, CUDA capture barrier·정규화 정밀도·JIT 경로·SDPA 옵션을 유지했다.

### ROCm 오버레이

fork `75915908`을 유지하고 core 파일의 MLX 대상을 변경했다. K/V head를 반복하여 4차원으로 계산하는 GQA fallback을 보존하면서 window/causal mask와 명시적 mask를 조합한다. fused window attention은 fallback으로 보내고 강제 실행 요청은 예외로 거부한다. 잔여 차이 기록을 재생성·검토하고 LOCAL_FIXES 5번에 병합 근거를 기록했다.

## 7. 변경 요약

- CMake의 단일 pin과 영향받는 오버레이를 갱신하고 기존 로컬 helper를 유지했다.
- mask를 전달하는 기존 C++ SDPA 호출부 5곳에 `window_size=-1`을 명시했다.
- [pin 검토 문서](../docs/mlxcelverse/mlx-pin-2026-10-26.md)에 비교와 검증 결과를 기록했다.

## 검증 및 후속 조치

CPU·CUDA sm_121 빌드가 통과했다. GB10의 gpu-lock 아래 layer 104/104, SDPA 22/22, grouped GEMM 3/3이 통과했으며 필터 간 중복이 있다. pin/parser, CPU SDPA·QMM, overlay 기록·drift, ROCm fast.cpp의 호스트 문법 검사도 통과했다.

CPU layer suite는 103/104 통과했다. f16 dense GEMM/QMM 바이트 일치 실패는 기존 pin의 캐시 라이브러리와 새 pin 라이브러리 양쪽에서 1,048,576개 중 598,053개 원소 차이로 재현된다. 같은 CUDA 검사는 통과했다. 이번 변경에서 기존 CPU 문제를 수정하지는 않았다.

Apple Silicon runtime/JIT parity, Windows DLL 로딩, AMD HIP 빌드·실행 검증은 남아 있다. 원격 CI 상태는 로컬 검증과 별도로 PR에서 추적한다.
