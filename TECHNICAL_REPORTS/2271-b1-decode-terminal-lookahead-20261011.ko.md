# 기술 보고서: PR #2271 - B=1 종료 디코드 lookahead

**작성일**: 2026-10-11
**상태**: CUDA(GB10)에서 구현 및 검증 완료, Qwen 프롬프트 캐시 제한은 아래에 기록
**언어**: Rust, Python, Bash, Markdown
**위험도**: 낮음에서 중간

## 요약

B=1 lookahead 스케줄러는 대기 중인 토큰 `n`을 읽기 전에 `n+1`을 제출했다. `n`이 `max_tokens`를 모두 사용하면 speculative append 두 개를 되돌린 뒤 이미 샘플링한 토큰을 동기 방식으로 다시 실행했다. PR #2271은 길이 예산을 먼저 확인하고 `n+1` 제출을 생략한 뒤 기존 fused finish 경로로 대기 토큰을 커밋한다. 따라서 fused lookahead가 가능한 길이 제한 B=1 요청마다 완전한 forward 두 번을 제거하며 B>1 동작은 유지한다.

GB10에서 5회 측정한 네 모델/프롬프트 셀은 모두 median -1.0% 성능 회귀 기준을 통과했다. 모든 candidate 범위가 반복 baseline null 범위와 겹치므로, 이 결과는 기준 통과를 뒷받침하지만 성능 향상을 입증하지 않는다.

## 1. 문제 정의

임시 호스트 타이머로 변경 전 Qwen B=1 서버 tick을 측정한 결과 평균은 5,318.203마이크로초였다. Row lookup과 build 합계는 0.0053%, metrics 발행은 0.0462%, provider channel send는 0.0698%, lookahead collect는 0.0836%였다. 이슈 #2227은 0.1% 미만 후보를 변경하지 않도록 요구하므로 이 경로들은 그대로 두었다. 0.1445%인 fused finish 호출은 공통 샘플링 및 완료 상태 전이를 담당하므로 유지했다.

Nsight 집계에서 서버는 CLI보다 CUDA graph launch가 126회, graph-exec update가 125회 많았다. 이 trace에는 워밍업 요청과 측정 요청이 모두 포함되고 한 model forward가 여러 graph 호출을 만들기 때문에, 이 수치는 보조 증거로만 사용한다. 정확한 요청 단위 원인은 scheduler forward-count 테스트로 검증했다.

## 2. 기술적 선택과 구현

- 활성 sequence가 하나이고 대기 토큰이 길이 예산을 반드시 소진할 때만 단축 경로를 사용한다.
- 대기 토큰은 기존 fused token 적용, finish 상태 평가, 이벤트 전송, 완료 finalization을 그대로 통과한다.
- EOS와 길이 제한이 같은 토큰에서 발생하면 EOS가 우선한다.
- 대기 토큰 읽기 실패나 형상 오류는 기존 unwind 및 동기 폴백을 사용한다.
- Seed가 있는 stochastic sampling은 이미 준비된 device draw를 커밋하며, 미래 draw를 제출하거나 종료 토큰을 다시 뽑지 않는다.
- B>1은 기존 next-submit, teardown, 동기 replay 동작을 유지한다.

실제 dense external-KV cycle-Llama fixture는 scheduler submit 및 step 호출 수를 센다. `max_tokens=3`에서 최적화된 B=1은 2회, B=2는 기존 4회를 유지한다. 예산 guard를 `false`로 바꾸면 실제 4회, 기대 2회로 테스트가 실패하여 이 분기가 측정된 작업을 제거함을 확인한다. 별도의 Muse 테스트는 생성 ID, finish reason, stream text, pool offset, model-owned offset을 비교하며, rewind를 지원하지 않는 Muse를 lookahead 가능한 원인 검증 fixture로 취급하지 않는다.

공개 옵션, 의존성, 저장 형식 변경은 없다. 테스트 전용 counter는 `cfg(test)`에서만 컴파일되어 release 경로 비용이 없다.

## 3. 성능 결과

Acceptance run은 pre-epic `4c44e317` 기준, quiet host에서 5회 interleaved round와 baseline 반복 null arm으로 측정했다.

| 모델 | 프롬프트 | Server dense 대 baseline CLI | Null arm |
|---|---:|---:|---:|
| Qwen3-1.7B 4-bit | 256 | +0.57% (-0.25%..+1.54%) | +0.04% (-0.41%..+0.76%) |
| Qwen3-1.7B 4-bit | 8192 | +0.32% (+0.04%..+0.63%) | +0.22% (-0.13%..+0.66%) |
| Llama-3.2-1B 4-bit | 256 | +0.48% (+0.19%..+1.41%) | +0.07% (-12.38%..+0.54%) |
| Llama-3.2-1B 4-bit | 8192 | -0.16% (-1.15%..+0.05%) | -0.17% (-0.29%..+0.14%) |

네 셀의 median이 모두 -1.0% 기준을 통과했다. 모든 candidate 범위가 null 범위와 겹친다. Llama 256 null arm에는 round 2의 -1.01%와 원인을 알 수 없는 round 3의 -12.38%가 포함되어 있으며, 제외하거나 재실행하지 않고 raw data와 분석에 그대로 보존했다.

## 4. 정확성 및 검증

- 실제 `mlxcel run` 생성 텍스트는 변경 전후 byte 단위로 같고, 두 생성 텍스트의 SHA-256은 `ecbb6c3137fa73714f6fe6df1dd6414b14c6af75796efe19662a3aaab50eaea6`이다.
- Llama 전체 parity는 13개 동일, divergence 0개, 해당 없음 1개다.
- 프롬프트 캐시 case를 제외한 Qwen parity는 9개 동일, divergence 0개, 해당 없음 1개다.
- Qwen 전체 parity는 프롬프트 캐시 miss/hit 비교에서 greedy 토큰 58(`374` 대 `594`)과 seeded 토큰 61(`576` 대 `362`)에 실패한다. 강제 동기 디코드도 같은 위치와 토큰 ID를 재현하며 새 guard는 실행되지 않는다. 이는 종료 lookahead 변경으로 생긴 문제가 아니라 #2227 범위 밖의 기존 프롬프트 캐시 경로 제한이다.
- Formatting, CUDA workspace clippy, repository contract, scheduler test, mutation 이후 binary 복구를 모두 통과했다.
- 복구 후 전체 CUDA suite는 root test 8,930개와 core test 2,072개가 통과했고 integration failure는 0개였다. Root 165개와 core 2개는 ignored였다.
- 보관된 acceptance analyzer는 `PASS (4/4 cells >= -1.00%)`를 재현하며, 추적 중인 모든 evidence 파일은 커밋된 SHA-256 manifest와 일치한다.

## 5. 출처 및 제한

정확성 및 성능 측정에 사용한 release binary는 `7c8830cd5aa01198e6ff93b661f90c048ccaeb1d` 소스에서 빌드했다. 이후 `93e2f9ab`까지의 checkout 차이는 test module 두 개에만 있으며 production file diff는 없다. 변경 후 실제 실행을 캡처할 때 source checkout에는 제어된 guard mutation이 있었지만 release binary를 다시 빌드하지 않았다. Binary hash와 보관한 source patch가 실제 실행 코드를 독립적으로 식별한다.

저장소 정책은 생성된 binary profiling capture를 제외한다. 원본 CLI `.nsys-rep`는 3,077,212바이트이고 SHA-256은 `646508acb7a4193870b5bbf1d93b090e6abc6667cda9da1783239ae63b176719`이다. Server capture는 3,147,242바이트이고 SHA-256은 `19efc62eaf08bcc9b25f3c79c1be9ccb43220f4d84ef1b0091bb5b04e38884ab`이다. 두 파일은 git 외부인 실행 호스트의 `/tmp/mlxcel-auto-plan/2227-profile/` scratch 디렉터리에 남아 있다. 커밋한 Nsight 통계 및 로그, plain log, instrumentation patch, profile analysis, acceptance JSONL, correctness record가 이 보고서에서 사용한 증거를 보존한다.

## 6. 변경 요약

| 분류 | 결과 |
|---|---|
| 성능 | 조건을 만족하는 길이 제한 B=1 요청마다 종료 decode forward 두 번을 제거하며 네 acceptance median이 모두 회귀 기준을 통과 |
| 정확성 | Targeted test에서 token output, finish semantics, cache offset, stochastic draw order, EOS 우선순위, B>1 동작을 유지 |
| 호환성 | 공개 API, 의존성, 모델 형식, B>1 동작 변경 없음 |
| 알려진 제한 | Qwen 전체 프롬프트 캐시 miss/hit parity divergence는 새 분기를 실행할 수 없는 강제 동기 경로에서도 재현됨 |

## 참고 자료

- 이슈 #2227
- PR #2271
- `docs/benchmark_results/b1-decode-overhead-gb10-2026-10-11.md`
- `docs/benchmark_results/data/b1-decode-overhead-gb10-2026-10-11/`
