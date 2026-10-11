# PR #2268: 서버 프리필 청크 적응형 선택

**작성일**: 2026-10-11
**상태**: 완료, 성능 승인 및 전체 통합 검증 통과
**언어**: Rust, Markdown
**위험도**: 중간 (실시간 배치 스케줄링, 프리필 분할, CLI 우선순위 변경)

## 요약

PR #2268은 서버 프리필 청크를 하나의 고정값에서 스케줄러 상태에 따른 정책으로 바꾼다. 다른 시퀀스가 디코딩 중이 아니면 2048토큰 청크를 사용하고, 활성 배치에 디코딩 작업이 있으면 512토큰 청크를 사용한다. native 옵션, llama 호환 옵션 또는 환경 변수로 값을 명시하면 두 상태 모두 해당 값으로 고정한다.

프리필 계획은 현재 커서에서 다시 시작할 수 있게 되었다. 경합 중인 요청이 512토큰 경계에서 시작한 뒤 활성 배치가 비워졌을 때, 기존 방식으로 새 2048토큰 계획을 만들면 현재 커서에서 시작하는 조각이 없을 수 있기 때문이다. 집중 CUDA 테스트는 실제 forward 길이와 요청 완료를 검증한다. GB10에서 새 release를 사용한 quiet-host admission 및 TTFT 승인이 통과했고, 선행 PR 2건을 merge한 뒤 전체 workspace 및 CUDA 통합 gate도 통과했다.

## 1. 문제 정의

PR #2205는 단일 스트림 TTFT 측정을 근거로 프리필 청크를 2048로 고정했다. 이후 PR #2226은 동시 요청 환경에서 반대 결과를 측정했다. 네 스트림이 디코딩 중일 때 8192토큰 요청을 추가하면 청크 2048에서 기존 스트림의 ITL p95가 127.9–128.9 ms로 상승했지만, 청크 512에서는 55.8–56.4 ms에 머물렀다. 반면 새 요청의 TTFT는 2048에서 1903–2060 ms였고 512에서는 3768–4067 ms로 늘었다.

어느 고정값도 두 상태를 함께 만족하지 못한다. 2048을 유지하면 새 요청을 받는 동안 기존 스트림이 지연되고, 512를 유지하면 보호할 스트림이 없는 경우에도 긴 TTFT를 지불한다. 긴 프롬프트가 조각 사이에서 대기하는 동안 디코딩 작업이 끝날 수 있으므로 요청 수락 시 한 번만 값을 선택하는 방식도 충분하지 않다.

기존 계획 생성자는 실행 중 값을 바꾸기에 안전하지 않았다. 조각 경계가 최초 adopted offset 또는 history boundary를 기준으로 정해지므로, 512 격자에서 도달한 커서가 새 2048 계획의 조각 시작점이 아닐 수 있었다. 스케줄러는 이 경우를 남은 조각 없음으로 보고 요청을 중단했다.

## 2. 기술적 선택과 그 이유

### 2.1 매 스케줄러 tick에서 청크 선택

PrefillChunkPolicy는 alone과 contended 값을 보관하고 active_batch가 비었는지에 따라 현재 청크를 선택한다. 기본값은 각각 2048과 512다. 대기 중인 프리필은 active_batch와 별도로 보관되므로 자기 자신 때문에 contended 상태로 잘못 판정되지 않는다.

--prefill-chunk-size, --batch-size / LLAMA_ARG_BATCH 호환 별칭 또는 MLXCEL_PREFILL_CHUNK를 명시하면 {n, n}을 만든다. 따라서 운영자가 정책을 고정할 수 있고, 기본값과 같은 2048을 명시한 경우도 옵션이 없는 경우와 구분한다. 우선순위는 native 옵션, 호환 별칭, 환경 변수 순으로 유지한다.

요청 단위로 정책을 고정하는 방안은 채택하지 않았다. 디코딩 스트림 옆에서 수락된 요청은 해당 스트림이 끝난 뒤에도 512를 계속 사용해 불필요한 TTFT 비용을 내기 때문이다. 새 요청에만 상태를 적용하는 방식도 같은 문제가 있다.

### 2.2 현재 커서에서 계획 재개

PrefillPlan::resume_at은 전체 계획과 같은 메타데이터를 유지하면서 현재 prefill_offset부터 남은 조각을 자른다. 따라서 이전 청크 크기와 관계없이 첫 continuation은 정확히 현재 커서에서 시작한다. 패딩, history boundary, prompt cache 메타데이터, 모델 기능 검사는 기존 생성자와 같은 helper를 공유한다.

5000토큰 요청에서 검증한 상태 전환은 다음과 같다.

    활성 디코딩 존재: 0..512
    활성 배치 종료:   512..2560, 2560..4608, 4608..5000
    forward 길이:     [512, 2048, 2048, 392]

남은 suffix가 한 조각에 들어가더라도 continuation은 chunked progress를 보고한다. 앞선 조각에서 시작된 chunked-prefill lifecycle의 reservation 및 진행률 계산을 유지하기 위해서다.

### 2.3 배치 스케줄러 밖에서는 alone 값 유지

Direct engine 생성, engine benchmark, speculative/MTP prefill 경로는 명시적 override가 없으면 기존과 같이 2048 alone 값을 사용한다. 기존 batched-prefill token budget도 alone 값에서 계산한다. 각 row는 이미 512토큰으로 제한되어 있다.

## 3. 구현 상세

Core planner에는 두 값 정책, 프로세스 단위 환경 변수 파싱 캐시, 공통 메타데이터 및 조각 생성 helper, 커서 기반 재개 계획을 추가했다. 다음 padded 길이만 필요한 capacity check는 같은 chunk eligibility, boundary, range, padding helper를 재사용하는 allocation-free next-piece accessor를 사용한다. 스케줄러는 정책을 보관하고 각 시작 또는 continuation 직전에 실시간 디코딩 상태를 확인한다. 실행 시 현재 커서에서 계획을 만들고 실제 실행할 조각에 필요한 block을 예약하며 tracing span에 선택한 청크를 기록한다.

CLI 필드는 기본값이 들어간 usize에서 Option<usize>로 바뀌었다. 이로써 2048을 명시한 경우도 명시값으로 유지한다. 두 서버 바이너리는 같은 우선순위 및 충돌 규칙을 사용한다. Startup, runtime settings, probe, server configuration, model worker, scheduler builder가 두 정책값을 전달하며 startup 진단 로그도 두 값을 모두 표시한다.

문서는 적응형 기본값, override 의미, 커서 재분할, prompt cache partition 영향, 정책의 근거가 된 PR #2226의 trade-off를 설명한다. dependency, 영속 스키마, 인증, 네트워크 protocol 변경은 없다.

## 4. 기술적 검토 사항

### 정확성과 호환성

- 청크 0은 두 정책 상태 모두에서 청크 분할을 비활성화한다.
- 선택한 청크보다 짧은 프롬프트는 하나의 forward로 유지된다.
- Embedding-input/VLM prefill과 분할할 수 없는 history-boundary 구간은 기존 동작을 유지한다.
- Native 옵션, 호환 별칭, 환경 변수 명시값은 두 상태를 같은 값으로 고정한다.
- 기존 direct 및 speculative 호출 지점은 alone 값을 유지한다.
- 새 dependency나 공개 응답 field를 추가하지 않았다.

가장 위험한 경로는 요청이 대기한 뒤 청크 크기가 바뀌는 경우다. Integration regression은 작은 테스트 모델에서 실제 스케줄러 prefill을 실행하고 Engine::prefill 직전에 길이를 기록한다. 모든 continuation을 소비하며 error와 timeout event를 거부하고 최종 Done event를 요구한다.

### 성능

이 정책은 PR #2226에서 확인한 512토큰 admission 성능을 회복하면서 2048토큰의 단일 스트림 TTFT를 유지하는 것이 목적이다. 실행 계획은 실제 실행할 조각 수에 선형이며, 전체 prefix vector를 만든 뒤 버리고 suffix를 다시 만드는 할당을 제거했다. 반복되는 paged-block capacity check는 남은 suffix 전체를 할당하지 않고 다음 조각만 상수 공간에서 계산한다.

새 release 측정은 모든 승인 기준을 통과했다.

| 검증 항목 | 승인 기준 | 결과 |
|---|---|---|
| Live-stream admission, quiet-host 3회 | 모든 회차 ITL p95 62.0 ms 이하 | **통과:** 54.1, 54.8, 55.2 ms |
| Qwen 3 1.7B server TTFT, 5회 | policy delta가 explicit-2048 null 범위 안 | **통과:** policy median +0.01%, null 범위 -4.73%~+3.26% |
| Llama 3.2 1B server TTFT, 5회 | policy delta가 explicit-2048 null 범위 안 | **통과:** policy median -0.20%, null 범위 -1.12%~-0.03% |

### 보안

새로운 외부 trust boundary를 만들지 않는다. CLI와 환경 변수 입력은 기존의 양의 정수 또는 0 parser를 사용하고, 올바르지 않은 환경 변수는 경고 후 적응형 기본값으로 돌아간다. 로그에는 숫자 정책값만 기록한다.

## 5. 검증

구현 worktree에서 완료한 검증:

- cargo check --profile test-fast --features cuda --lib: 통과.
- cargo fmt --all -- --check: 통과.
- git diff --check: 통과.
- CUDA mlxcel-core prefill-plan 테스트: 17개 통과.
- CUDA server CLI 테스트: 148개 통과.
- CUDA adaptive scheduler 실행 regression: 통과.
- CUDA runtime-settings 테스트: 8개 통과.
- CUDA props 호환성 regression: 통과.
- 독립 security/performance review에서 반복되는 paged-block capacity check가 남은 모든 계획 조각을 즉시 생성하는 문제를 발견했다. Commit 88c2e6a0964d4a3a144c2ff1cc3c80dfd2893752는 이를 O(1) accessor로 교체하고 boundary, cursor, chunk, padding mode, model opt-out, embedding input 전반의 동등성 테스트를 추가한다.
- 수정 후 workspace all-targets Clippy가 통과했다. Focused CUDA runtime 검증은 prefill-plan 17개, block-reclaim 20개, adaptive scheduler transition regression이 모두 통과했다.
- 최초 full CUDA gate에서 capability helper 분리 후 source audit가 공통 guard를 인식하지 못한 문제를 발견했다. Audit는 brace-counted interprocedural 인식으로 보완했고 direct-guard 검사를 유지했으며 mutation fixture를 추가했다. 해당 테스트 3개는 모두 통과했다.

Acceptance artifact는 production commit `88c2e6a0964d4a3a144c2ff1cc3c80dfd2893752`를 사용했고 clean worktree HEAD `b2394566a71fc1f6db83ad40b44380b9c129b57c`에서 측정했다. Driver는 두 test file만 다르다는 점을 검증했다. Raw record, server log, 정확한 환경, machine-readable summary, 재현 driver 및 analyzer는 `docs/benchmark_results/data/adaptive-prefill-chunk-gb10-2026-10-11/`에 보관한다.

Driver의 기본 build commit은 squash 전 측정 lineage에서만 유효하다. Squash merge 이후 또는 더 새로운 clean checkout에서 재실행하려면 해당 checkout에서 release binary 2개를 다시 빌드하고 `MLXCEL_BUILD_COMMIT="$(git rev-parse HEAD)"`를 지정해야 한다. 전체 명령은 benchmark report에 기록했다.

PR #2270과 PR #2269를 merge한 뒤 전체 series는 workspace test `--no-run` compile, workspace all-targets Clippy, formatting, kernel dtype-key, kernel port-dispatch, llama-server compatibility contract suite, 그리고 `make verify-test-cuda`의 전체 single-threaded CUDA workspace run을 모두 통과했다. CUDA test summary는 전부 실패 0건이었고 BF16 및 F32 SSM update parity test도 통과했다. Rebase 전 검증 tree `ab8ff77ade8d6b1b4fb494707320cebc95150c8e`는 현재 `origin/main`으로 rebase한 뒤에도 정확히 같았다. 해당 비교 이후 변경은 문서와 PR 상태뿐이다.

## 6. 변경 요약

| 항목 | 값 |
|---|---|
| 변경 파일 | 47 |
| 추가 라인 | 1879 |
| 삭제 라인 | 210 |
| 측정 대상 branch head | b2394566a71fc1f6db83ad40b44380b9c129b57c |

| 분류 | 요약 |
|---|---|
| Core planning | 두 상태 정책, override parsing cache, 커서 기반 재개 계획 |
| Scheduling | 실시간 상태에 따른 청크 선택, 안전한 continuation/reservation, O(1) capacity check |
| Configuration | 명시값 식별과 두 서버 front end 전체의 두 값 전달 |
| Testing | Planner property, CLI 우선순위, schema 호환성, 실제 scheduler forward 순서 |
| Documentation | 적응형 기본값, override 동작, 측정 근거, cache partition 설명 |

최초 구현 commit은 7c43726a0fb792594a69de0fda1e31e78eef4fa1이다. 7b2d17241ea9c16612766ccf37524cbe8be9f913은 CI Clippy 표현식 정책을 만족한다. 88c2e6a0964d4a3a144c2ff1cc3c80dfd2893752는 review에서 발견된 reservation 할당을 제거하며, 808bcccf387957a6da31b961666aadd5bf7cbe65는 all-targets Clippy에 필요한 테스트 전용 type annotation을 추가한다. 6f07f7563e74c5466b5aace6f3e8ee98ec8347e2와 b2394566a71fc1f6db83ad40b44380b9c129b57c는 공통 capability guard를 위한 source audit를 보수적으로 갱신하고 fixture를 강화한다.

## 7. 완료 상태

Issue #2228의 구현, 성능 승인, 통합 검증에 남은 작업은 없다.

## 참고 자료

- Issue #2228: 다른 시퀀스의 디코딩 여부에 따른 프리필 청크 선택.
- PR #2226 및 docs/benchmark_results/unified-engine-final-gb10-2026-10-08.md: 정책 근거가 된 admission 측정.
- ADR 0007: unified batch/native engine 결정과 개정된 프리필 청크 정책.
- docs/CONTINUOUS_BATCHING.md: 서버 정책, 계획 invariant, 운영 제어.
- docs/benchmark_results/adaptive-prefill-chunk-gb10-2026-10-11.md: 통과한 승인 결과, raw data, 재현 protocol.
