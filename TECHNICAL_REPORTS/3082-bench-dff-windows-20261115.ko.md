# 기술 보고서: PR #3082 - fused-MoE Dff harness 테스트를 Windows에서 실행

**작성일**: 2026-11-15
**상태**: 완료
**위험도**: 낮음

## 요약

Windows build job의 `tests/` Python suite가 `test_bench_fused_moe_dff`에서 실패했다. 원인은 두 가지였다. `loadavg()`는 Linux에만 있는 `/proc/loadavg`를 읽었고, `main()`은 `harness.log`를 닫지 않고 반환해 Windows가 테스트의 임시 디렉터리를 지우지 못했다. 이제 `loadavg()`는 `os.getloadavg()`로, 그것도 안 되면 `"na"`로 대체하고, `main()`은 `contextlib.ExitStack`으로 파일을 열어 어느 경로로 반환하든 파일을 닫는다. 이 job은 변경 경로에 따라서만 실행되어 거의 돌지 않았기 때문에 문제가 main까지 들어갔고, #3069가 이를 처음 실행시켰다.

## 7. 변경 요약

| 항목 | 값 |
|------|-------|
| 파일 | `scripts/bench_fused_moe_dff.py`, `tests/test_bench_fused_moe_dff.py` |
| 테스트 | `loadavg()` fallback 테스트 2개 추가, `tests/` suite 통과 (240개) |
