# PR #2290: Nemotron-H SSD prefill 메모리 제한

**날짜**: 2026-10-13
**상태**: 부분 완료; 구현 리뷰와 집중 검증 통과
**위험도**: 이슈 범위에서는 높음. 요청별 transient-memory admission이 아직 구현되지 않음

## 요약

PR #2290은 Nemotron-H Mamba prefill을 prompt 전체 길이의 quadratic surrogate-attention graph 하나로 처리하던 방식에서 checkpoint 크기의 SSD chunk로 나누는 방식으로 변경합니다. 각 chunk는 recurrent state를 다음 chunk로 전달하고 다음 graph를 만들기 전에 평가됩니다. 테스트한 M1 Ultra checkpoint에서 6,019-token process peak가 72.2 GiB에서 18.7 GiB로 감소했고, 실질적인 greedy output은 유지되었습니다.

## 문제

기존 `nemotron_h` prefill 경로는 모든 Mamba layer에서 `L × L` surrogate-attention matrix를 생성했습니다. 따라서 긴 prompt의 transient memory peak가 대략 quadratic하게 증가했고 Metal driver에서 process가 종료될 수 있었습니다. Issue #2281은 예측 가능한 oversized request의 거부도 요구하지만, 현재 server에는 request별 activation 또는 transient memory 추정 기능이 없습니다.

## 설계와 리뷰

- `NemotronHConfig`는 이제 `chunk_size`를 읽고 기존 `mamba_chunk_size` alias를 허용하며, 기본값 128을 사용하고 model 생성 전에 0을 거부합니다.
- `nemotron_h_ssm_attn`은 float32 timestep, state, matrix-product 경계를 유지하고 각 quadratic block을 독립적으로 계산합니다. 각 block의 output과 next state를 평가한 뒤 그 state를 다음 block의 prior state로 사용합니다.
- 집중 수치 검증은 nonzero prior state를 전달하는 2+2+1 partial-tail partition을 full step과 비교하고, multi-batch/multi-group f16 및 bf16 결과를 scalar recurrence oracle과 비교합니다.
- 구현은 production Nemotron-H model builder에 연결되어 있고 실제 server benchmark에서 실행됩니다. 선언된 부분 범위 안에서는 남은 code-level correctness 결함을 발견하지 못했습니다.

## Benchmark 근거

1,519, 3,019, 4,519, 6,019 token의 각 cell에서 fresh process로 3회 반복했습니다. Arm별 load 및 forward marker를 사용해 candidate가 새 경로를 실행하고 baseline은 실행하지 않은 경우에만 비교를 보고합니다. Candidate peak memory는 이 범위에서 17.6–18.7 GiB였고, 8,019-token 실행 3회는 19.3 GiB였습니다. 2,810-token prompt를 포함한 실질적인 greedy prompt 3개에서 두 arm의 choice가 동일했습니다. 8,019-token baseline, 제보자의 M4 Pro 및 2-bit checkpoint, M5, CUDA, quiet host에서의 latency는 미측정으로 명시했습니다.

## 검증

- Metal/Accelerate release `mlxcel-server` build: 통과.
- Nemotron-H module suite: 22개 통과, 3개 ignored.
- 전용 serial Metal 수치 테스트: 2개 통과.
- Formatting, diff hygiene, Python harness compilation, benchmark artifact consistency, cross-repository reference 검토: 통과.
- 리뷰 중 확인한 현재 CI formatting, Clippy, CPU-only link, license, Python tooling, repository contract 검사: 통과. 나머지 job은 실행 또는 대기 중이었습니다.

## 남은 작업

Issue #2281을 완료하려면 요청별 activation/transient-memory admission이 필요합니다. 임의의 within-context Metal command-buffer OOM은 여전히 process를 종료할 수 있습니다. 또한 PR 본문은 이 issue를 의도적으로 닫지 않는다고 명시하지만 GitHub는 #2281을 이 PR의 `closingIssuesReferences`로 보고합니다. Issue를 분리하거나 closing relationship을 제거하기 전에는 이 PR을 merge하면 안 됩니다.
