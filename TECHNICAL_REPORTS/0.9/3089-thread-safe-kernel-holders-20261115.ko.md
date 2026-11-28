# 기술 보고서: PR #3089 - 지연 초기화되는 custom-kernel holder를 thread 간에 한 번만 초기화

**작성일**: 2026-11-15
**상태**: 완료
**위험도**: 중간

## 요약

`mlx_cxx_kernels.cpp`에 있는 kernel holder 22개는 처음 쓸 때 단순한 `if (!initialized)` flag를 확인한 뒤 `CustomKernelFunction`을 만들었다. 그래서 두 thread가 같은 holder에 동시에 들어오면 race가 생겼다. `granitemoehybrid` 테스트를 병렬로 돌리면 fused MoE launch가 `std::bad_function_call`이나 `__next_prime overflow`를 던졌고, 가끔 프로세스가 SIGSEGV로 죽었다. M5 Max에서 #3069 머지 시점에는 10번 중 7번 실패했고, 바로 직전 커밋에서는 10번 중 0번이었다. race는 #3069 이전부터 있었고, #3069가 thread 타이밍을 바꾸면서 드러났다. 같은 kernel을 처음 쓰는 시점에 두 thread에서 dispatch하는 서버라면 같은 문제를 겪는다. 이제 각 holder는 `std::call_once`로 초기화한다.

## 1. 문제 정의

두 커밋 사이를 bisect하고, `switch_layers::forward_fused_kernel`이 launch의 exception을 삼키던 `.ok()`에 로그를 넣어 race 위치를 찾았다.

## 7. 변경 요약

| 항목 | 값 |
|------|-------|
| Granite 모듈 병렬 실행 | 12번 중 12번 통과 (main에서는 10번 중 3번 실패) |
| mlxcel-core lib suite 병렬 실행 | 3번 중 3번 통과 |
