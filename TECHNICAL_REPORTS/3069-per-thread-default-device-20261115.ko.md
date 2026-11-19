# 기술 보고서: PR #3069 - MLX 기본 device를 thread별로

**작성일**: 2026-11-15
**상태**: 완료
**위험도**: 중간

## 요약

MLX는 기본 device를 process 전체에서 하나만 두지만, 기본 stream은 thread마다 따로 둔다. 그래서 `DefaultDeviceGuard::cpu()`로 CPU를 고정하는 테스트가 있으면 동시에 도는 다른 테스트의 연산까지 CPU로 옮겨 갔다. 병렬로 실행한 `cargo test -p mlxcel-core --lib`는 6번 중 5번 실패했다(#3066). 이제 모든 backend에 적용되는 core overlay `mlx/device.cpp`(ROCm 사본에도 같은 변경을 반영)가 기본 device를 thread별로 둔다. 자기 기본값을 정한 적 없는 thread에는 `set_process_default_device`가 정한 기본값이 적용되며, runtime은 `--device cpu`에 이 함수를 쓴다.

## 3. 기술적 선택과 그 이유

- **thread별 기본값과 process 기본값을 함께 둔다.** thread별 기본값만 두면 `--device cpu`에서도 worker thread는 GPU에 남는다. 그래서 시작 후 생성되는 thread를 위해 별도의 process 기본값을 둔다.
- **CMake가 core overlay를 지키게 한다.** CUDA가 아닌 빌드에서는 수정된 최상위 `mlx/*.cpp` 파일을 원래대로 되돌리는 CMake 단계가 overlay를 조용히 원복하고 있었다.
- **ROCm 기록을 갱신했다.** `CORE_RESIDUAL.diff`를 다시 생성했고, `LOCAL_FIXES.md` 75번 항목에 변경 이유를 적었다.
- **메모리를 읽는 테스트를 옮겼다.** #3063의 state-tail retention 검사는 process 전체 메모리를 읽으므로 별도 테스트 바이너리로 옮겼다.

## 7. 변경 요약

| 항목 | 값 |
|------|-------|
| mlxcel-core lib 병렬 실행 | 5번 중 5번 통과 (이전 6번 중 1번) |
| 새 테스트 | CPU guard가 자기 thread에만 적용되는지, process 기본값이 이후 생성된 thread에 적용되는지 |

## 8. 후속 조치

바뀐 thread 타이밍 때문에 지연 초기화되는 kernel holder 22개에 원래 있던 race가 드러났고, #3089에서 수정했다.
