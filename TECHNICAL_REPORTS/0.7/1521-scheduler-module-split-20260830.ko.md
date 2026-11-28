# 기술 보고서: PR #1521 - 스케줄러 모듈 분리

**날짜**: 2026-08-30
**상태**: 완료
**언어**: Rust
**위험도**: 낮음

---

## 요약

PR #1521은 기존 동작과 `src/server/batch/` 밖의 공개 API 경계를 유지하면서 배치 스케줄러 구현을 디렉터리 모듈로 분리했다. 리베이스된 최신 main 기준 8724줄이던 단일 파일을 관심사별 모듈로 나누고, MTP 디스패치 소스 검사 가드가 실제 컴파일되는 코드를 계속 검사하도록 유지했으며, 파일 크기 회귀를 막는 구조 테스트를 추가했다.

## 1. 문제 정의

### 1.1 배경

`src/server/batch/scheduler.rs`는 스케줄러, 프롬프트 캐시, paged KV, handoff, speculative decoding 변경이 모두 모이는 병합 충돌 핫스팟이 되어 있었다. 이 이슈의 요구사항은 기존 스케줄러 테스트를 수정하지 않고, `src/server/batch/` 밖의 공개 API를 바꾸지 않으며, 동작 변경 없이 자연스러운 하위 모듈로 분리하는 것이었다. 최종화 중 `origin/main`이 PR #1503으로 전진했기 때문에, 해당 context-retention 스케줄러 동작과 테스트를 split 구조 안으로 이식한 뒤 브랜치를 다시 게시했다.

### 1.2 기존 문제

- **과도하게 큰 구현 단위**: 구현 시점의 스케줄러 구현 파일은 8724줄이었고, 문서화된 2000줄 anti-pattern 기준을 넘었지만 예외 근거가 없었다.
- **불명확한 관심사 경계**: admission, prefill, decode, prompt-cache, paged layout, handoff, speculative finalization 코드가 하나의 큰 파일에 섞여 있어 리뷰와 리베이스 비용을 키웠다.
- **소스 검사 결합**: `speculative_burst_tests.rs`는 MTP 디스패치 커버리지를 확인하기 위해 `src/server/batch/scheduler.rs`를 직접 읽으므로, 해당 경로를 단순 삭제하면 가드가 깨지거나 중복된 가짜 마커로 우회될 위험이 있었다.

### 1.3 위험 평가

| 위험 | 영향 | 가능성 |
|------|------|--------|
| 스케줄러 병합 충돌 지속 | 중간 | 높음 |
| 분리 과정의 우발적 동작 변경 | 높음 | 대상 테스트 후 낮음 |
| 소스 검사 가드가 MTP 디스패치 누락을 숨김 | 높음 | 검사 대상 파일을 실제 프로덕션 코드로 컴파일한 뒤 낮음 |

## 2. 기술 검토

### 2.1 보안

요청 파싱, 인증, 권한, 파일시스템 접근, 네트워크 접근, 비밀정보 처리 로직은 변경하지 않았다. 이번 리팩터는 기존 스케줄러 메서드를 이동하고, 형제 모듈 접근에 필요한 범위에서만 가시성을 조정했다.

### 2.2 성능

런타임 동작은 변하지 않는다. 같은 `BatchScheduler` inherent method들이 같은 크레이트 안에서 컴파일되므로 추가 모듈 경계는 동적 디스패치, 할당, 런타임 분기를 만들지 않는다.

### 2.3 호환성과 의존성

- **호환성 깨짐**: `src/server/batch/` 밖의 공개 API 변경 없음.
- **새 의존성**: 없음.
- **호환성**: 기존과 같은 빌드 프로필에서 스케줄러 및 인접 speculative/prompt-cache 대상 테스트가 통과했다.

### 2.4 코드 품질

분리는 admission, configuration, decode ticks, handoff helpers, paged layout, prefill, prompt-cache handling, run-loop helpers, speculative finalization, MTP dispatch source seam으로 구성된다. 새 구조 테스트는 스케줄러 모듈 파일이 문서화된 예외 없이 2000줄을 초과하면 실패한다.

## 3. 기술적 선택과 그 이유

### 3.1 `scheduler.rs`를 실제 컴파일되는 MTP 디스패치 소스로 유지

**맥락:** 기존 `speculative_burst_tests.rs` 소스 검사 가드는 `include_str!("scheduler.rs")`를 사용하며, 이 이슈에서는 테스트 파일을 기준 대비 동일하게 유지해야 했다. 가짜 shim은 테스트를 통과시킬 수 있지만 실제 디스패치 누락을 숨길 수 있다.

**결정:** `src/server/batch/scheduler.rs`를 611줄의 프로덕션 소스 파일로 남겨 실제 MTP 디스패치 메서드를 담고, `scheduler/mod.rs`에서 `#[path = "../scheduler.rs"] mod mtp_dispatch;`로 컴파일한다.

**근거:** 변경되지 않은 소스 검사 테스트가 기존 경로를 계속 읽으면서, 그 경로의 내용은 실제 스케줄러 모듈에 컴파일되는 코드가 된다. 따라서 디스패치 누락 감지 능력을 유지한다.

### 3.2 스케줄러 상태 기계 관심사별 분리

**맥락:** 이슈는 재설계가 아니라 동작 변경 없는 자연스러운 모듈 분리를 요구했다.

**결정:** 메서드 그룹을 디렉터리 모듈의 형제 파일로 이동하고, `BatchScheduler` 타입과 메서드 이름은 유지했으며, 이동 때문에 필요한 private 메서드만 `pub(super)`로 넓혔다.

**근거:** 여러 개의 `impl BatchScheduler` 블록은 호출부를 유지하면서 새 trait, wrapper type, dispatch layer를 만들지 않는다.

## 4. 변경 요약

| 모듈 | 줄 수 | 책임 |
|------|-------|------|
| `src/server/batch/scheduler.rs` | 611 | 기존 소스 검사 경로에 남긴 실제 컴파일 MTP 디스패치 메서드 |
| `src/server/batch/scheduler/mod.rs` | 980 | 공통 import, helper, 상수, 상태 타입, 테스트 모듈 선언 |
| `src/server/batch/scheduler/admission.rs` | 703 | intake, enqueue, scheduler action selection, preemption admission, paged-block admission |
| `src/server/batch/scheduler/config.rs` | 672 | constructor, builder method, resolved scheduler configuration, MTP policy setup |
| `src/server/batch/scheduler/decode_tick.rs` | 1632 | decode execution, lookahead, preemption eviction, completion, cancellation, abort handling |
| `src/server/batch/scheduler/handoff.rs` | 526 | sequence handoff extraction, ingest, role-specific handoff helper |
| `src/server/batch/scheduler/paged_layout.rs` | 312 | sequence-state allocation, KV mode/layout resolution, storage sync |
| `src/server/batch/scheduler/prefill.rs` | 1374 | full, chunked, batched prefill path와 prefill finalization |
| `src/server/batch/scheduler/prompt_cache.rs` | 1229 | prompt-cache adoption, donation, warmup, release, observability bookkeeping |
| `src/server/batch/scheduler/run_loop.rs` | 310 | run loop, structured-mask helper, thinking budget, metrics publishing |
| `src/server/batch/scheduler/speculative_finalize.rs` | 657 | legacy burst finalization, grantee promotion, slice failure/routing helper |
| `src/server/batch/scheduler/structure_tests.rs` | 47 | 파일 크기 회귀 가드 |

## 5. 검증

- `cargo test --lib scheduler_modules_stay_below_documented_anti_pattern_threshold`: 통과, 1 passed, 7417 filtered out.
- `cargo test --lib server::batch::speculative_burst_tests::every_mtp_dispatch_site_covers_every_capable_variant`: 통과, 1 passed, 7416 filtered out.
- `cargo test --lib server::batch::scheduler::`: 통과, 99 passed, 7 ignored hardware/model tests, 7312 filtered out.
- `cargo test --lib server::batch::scheduler_prompt_cache_tests::`: 통과, 24 passed, 7394 filtered out.
- `cargo test --lib server::batch::speculative_burst_tests::`: 통과, 47 passed, 7371 filtered out.
- `cargo test --lib server::batch::speculative_slice_tests::`: 통과, 12 passed, 7406 filtered out.
- `cargo build`: 통과.
- `cargo clippy --lib --tests -- -D warnings`: 통과.
- `cargo fmt --check`: 통과.
- `git diff --check`: 통과.
- `speculative_burst_tests.rs`의 `origin/main` 대비 해시 확인: 통과, base와 working tree SHA-256 모두 `74704868ced3b3627fcd148252e4bbe39ac98ff2812c5ed1354da9ef3d5845b9`.

## 6. 후속 참고

ignored 처리된 스케줄러 테스트는 `qwen3-0.6b-4bit` 체크포인트와 실제 GPU forward를 요구한다. 이번 bounded refactor pass에서는 실행하지 않았고, 일반 대상 테스트로 컴파일 및 비하드웨어 동작 경계를 검증했다.
