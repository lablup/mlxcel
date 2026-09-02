# 기술 보고서: PR #1571 - docs: correct stale "not a server knob" claim in turbo-kv-cache.md

**작성일**: 2026-09-02
**작성자**: AI Code Reviewer
**리뷰어**: (지정 예정)
**상태**: 완료
**언어**: Markdown (문서 전용 변경)
**위험도**: Low

---

## 요약

`docs/turbo-kv-cache.md`는 `MLXCEL_PAGED_ATTENTION_NATIVE`를 라이브러리/벤치마크 전용 제어값이며 "서버 노브가 아니다(not a server knob)"라고 설명하고 있었다. 이 설명은 이슈 #899가 이 변수에 두 번째 서버 측 소비자를 추가하고, force-off 값을 프로덕션 v2 디코드의 킬 스위치로 만들면서 사실과 어긋나게 되었다. 이번 PR은 PR #1119가 `docs/environment-variables.md`에 이미 적용한 수정된 서술을 그대로 재사용해 문단을 다시 작성했고, 그 결과 이 변수를 설명하는 4개 문서 페이지가 서로 일치하게 되었다.

---

## 1. 문제 정의

### 1.1 배경

`MLXCEL_PAGED_ATTENTION_NATIVE`는 원래 소비자가 하나였다. `mlxcel-core` 호출자와 커널 벤치마크에서만 도달하는 라이브러리 전용 진입점 `paged_decode_attention_pooled`다. 이슈 #710은 이 진입점을 `mlxcel serve` 디코드 경로에서 제외했고, `docs/turbo-kv-cache.md`의 "서버 노브가 아니다"라는 서술은 여기서 비롯된 것이다. 이후 이슈 #899가 `src/lib/mlxcel-core/src/layers.rs`에 `resolve_paged_v2_dispatch`를 추가하면서 두 번째 소비자가 생겼다. 서버의 풀 기반 배치 페이지드 디코드도 같은 변수를 읽으며, 그 force-off 값(`0`/`false`/`off`/`no`)이 #899 이전의 gather-then-SDPA 경로로 되돌리는 킬 스위치가 되었다.

### 1.2 기존 문제점

- **문제 1**: `docs/turbo-kv-cache.md`는 여전히 이 변수를 "외부 mlxcel-core 소비자와 커널 벤치마크를 위한 제어값이며, 서버 노브가 아니다"라고 현재형으로 단언하고 있었으나, 이는 #899 이후로 사실이 아니다.
- **문제 2**: PR #1119(이슈 #1104)가 `docs/environment-variables.md`에서 같은 오래된 주장을 이미 수정했지만, 그 인수 조건은 `README.md`와 `docs/CONTINUOUS_BATCHING.md`만 명시했고 `docs/turbo-kv-cache.md`는 해당 PR 본문에서 후속 작업으로 명시적으로 남겨졌다.

### 1.3 위험성

| 위험 | 영향도 | 발생 가능성 |
|-----|-------|-----------|
| 운영자가 `docs/turbo-kv-cache.md`만 읽고 이 변수가 프로덕션에서 무시해도 되는 값이라고 오판하여, 실제로는 v2 킬 스위치이기도 하다는 사실을 놓침 | Medium | Medium |
| 배포된 환경 변수의 동작에 대해 문서 페이지들이 서로 다른 설명을 함 | Low | High (이번 수정 전까지 이미 사실이었음) |

---

## 2. 기술적 검토 사항

### 2.1 보안 관점

해당 없음. 문서 전용 변경이며 코드나 설정 파싱 로직은 건드리지 않았다.

### 2.2 성능 관점

해당 없음. 코드 경로 변경 없음.

### 2.3 호환성/의존성 관점

- **Breaking Changes**: 없음.
- **새로운 의존성**: 없음.
- **호환성**: 해당 없음.

### 2.4 코드 품질 관점

- **테스트 커버리지**: 해당 없음 (문서 전용).
- **코드 복잡도**: 해당 없음.
- **기술 부채**: 감소. 이 변수의 실제 현재 동작과 모순되던 페이지가 하나 줄었다.

---

## 3. 기술적 선택과 그 이유

### 3.1 독자적으로 다시 서술하는 대신 PR #1119의 수정된 서술을 재사용

**컨텍스트:**
`docs/environment-variables.md`는 PR #1119(커밋 `cf4e22cd`) 이후 이미 `MLXCEL_PAGED_ATTENTION_NATIVE`에 대한 수정된 설명을 담고 있다. `docs/turbo-kv-cache.md`도 같은 수정이 필요했지만, 이 페이지는 융합 split-K 커널의 디스패치 방식도 함께 다루는 문단 안에서 서술체로 표현해야 했다.

**고려한 대안:**

| 옵션 | 장점 | 단점 |
|-----|-----|-----|
| Option A: 수정된 동작을 독자적으로 다시 서술 | 이 페이지 맥락에 맞춘 문장 구성 가능 | 미묘하게 다른(그리고 다시 오래된 설명이 될 수 있는) 서술을 재도입할 위험이 있고, 이미 다른 곳에 문서화된 토큰 하한값·로그 라인 세부사항을 중복시킴 |
| Option B: "서버 노브가 아니다" 주장만 삭제하고 현재 동작을 추가하지 않음 | 최소한의 diff | 이 변수의 두 소비자에 대해 이 페이지가 침묵하게 되고, ADR 0001 링크의 맥락이 고아 상태가 됨 |
| **선택: Option C — `cf4e22cd`의 서술을 재사용하고, 중복 대신 상호 링크** | 이 페이지와 `docs/environment-variables.md`가 구조적으로 일치하게 됨; 토큰 하한값과 디스패치 정책은 각각 한 곳에서만 정의됨 | 서버 측 하한값의 전체 세부사항을 확인하려면 독자가 링크 두 개를 따라가야 함 |

**선택 이유:**
이슈는 `git show cf4e22cd -- docs/environment-variables.md`를 먼저 읽고 그 수정된 서술을 재사용한 뒤, 디스패치 정책과 로그 라인 서술을 중복시키지 말고 `docs/environment-variables.md#paged-decode-v2-variables`와 `docs/CONTINUOUS_BATCHING.md#seeing-which-path-ran`을 상호 링크하라고 명시적으로 요구했다. 이렇게 하면 이 변수를 설명하는 네 페이지(`README.md`, `docs/CONTINUOUS_BATCHING.md`, `docs/environment-variables.md`, `docs/turbo-kv-cache.md`)가 다음에 어느 한쪽이 갱신될 때 다시 어긋나는 일을 줄일 수 있다.

**트레이드오프:**
다시 쓴 문단은 두 소비자, 킬 스위치 의미, 링크 두 개를 모두 명시하면서 원문보다 다소 밀도가 높고 길어졌다. 이는 이슈의 인수 조건이 단순 한 줄 수정이 아니라 두 소비자를 명시적으로 이름 붙이도록 요구했기 때문에 받아들여진 트레이드오프다.

### 3.2 의도적으로 과거를 서술하는 두 곳은 그대로 둠

**컨텍스트:**
"not a server knob"이라는 문자열은 정당한 이유로 두 곳에 더 남아 있다. `docs/adr/0001-paged-attention-gather-vs-fused-kernel.md`의 `### Decision` 섹션은 결정 시점에 사실이었던 내용을 기록하고 있고, `docs/environment-variables.md`의 `History:` 절은 예전 해석을 과거형 인용으로 담고 있을 뿐 현재의 주장이 아니다.

**선택 이유:**
ADR은 구조적으로 역사적 기록이며 현재 동작에 맞춰 소급 수정해서는 안 된다. 그렇게 하면 이후 변경으로 이어진 추론 과정 자체가 지워진다. `docs/environment-variables.md`의 `History:` 절은 이미 올바른 패턴(예전 해석을 인용한 뒤 무엇이 왜 바뀌었는지 서술)을 사용하고 있으며, 이번 PR은 그 패턴을 `docs/turbo-kv-cache.md`에도 그대로 반영했다.

---

## 4. 구현 상세

### 4.1 주요 코드 변경

**파일: `docs/turbo-kv-cache.md`**

변경 전 (394-399번째 줄):
```
slab. #710 retired this pooled entry point to a library-only API: neither this
kernel nor its selector is on the `mlxcel serve` decode path (which stays on the
block-table kernel described above), and `MLXCEL_PAGED_ATTENTION_NATIVE` is a
control for external mlxcel-core consumers and the kernel bench, not a server
knob. See ADR 0001's #710 decision record,
[ADR 0001](adr/0001-paged-attention-gather-vs-fused-kernel.md).
```

변경 후:
```
slab. #710 retired the pooled entry point and its selector to a library-only
API, off the `mlxcel serve` decode path, which stays on the block-table kernel
described above. The variable itself keeps two consumers today, per
`resolve_dispatch_decision` and `resolve_paged_v2_dispatch` in
`src/lib/mlxcel-core/src/layers.rs`: that library-only pooled entry point, and
the server's pool-backed batched paged decode. On the server side, issue #899
made the fused v2 kernel the production decode path and named this variable's
force-off values its kill switch; a force-on value pins v2 for every servable
shape, bypassing the measured token floors. See
[Paged decode v2 variables](environment-variables.md#paged-decode-v2-variables)
for the floors and defaults, and
[Continuous batching](CONTINUOUS_BATCHING.md#seeing-which-path-ran) for the
dispatch policy and per-outcome log lines. History: #710's retirement of the
library entry point is where the "not a server knob" reading came from; #899
gave the variable its second, server-side consumer. See ADR 0001's #710 decision
record, [ADR 0001](adr/0001-paged-attention-gather-vs-fused-kernel.md).
```

**변경 이유:** 이 변수의 현재 두 소비자(`resolve_dispatch_decision`, `resolve_paged_v2_dispatch`와 일치)를 명시하고, 서버 측 킬 스위치와 force-on 동작을 구체적으로 서술하며, #710 폐기 사실을 현재형 주장이 아닌 역사적 기록으로 격하시키고, 토큰 하한값과 디스패치 로그 세부사항을 중복시키는 대신 이미 그 내용을 담고 있는 두 페이지로 링크한다.

이 문단은 파일 내 해당 구역의 기존 스타일에 맞춰 약 80칸에서 줄바꿈되어 있다. (같은 파일의 다른 구역은 줄바꿈 없는 단일 라인 문단을 쓰므로, 줄바꿈은 파일 전체 규칙이 아니라 이 구역의 관례다.)

---

## 5. 학습 포인트

### 5.1 부분적 수정 이후 발생하는 문서 간 불일치

**개념:**
환경 변수의 동작이 바뀌면(여기서는 #899를 통해 두 번째 소비자가 생긴 것), 그 변수를 설명하는 모든 페이지가 동일한 수정을 받아야 한다. 앞선 수정(#1104/PR #1119)은 인수 조건 범위를 특정 페이지로 한정했고, 남은 페이지를 조용히 방치하는 대신 후속 작업으로 명시적으로 표시해두었다.

**이 PR에서의 적용:**
이번 PR은 그 명시된 후속 작업을 마무리한다. 검증 단계(`grep -rzoP 'not a server\s+knob' docs README.md`, 혹은 이 변경이 하드 줄바꿈을 걸치기 때문에 단순 한 줄 grep으로는 놓칠 수 있어 이번에 사용한 Python 기반 등가 검색)는 해당 문자열이 정당하게 역사적인 두 맥락에서만 남아 있음을 확인한다.

**일반적인 사용 사례:**
- 시간이 지나며 범위가 커지는 변수, 플래그, API는 범위가 커진 그 페이지뿐 아니라 그것을 참조하는 모든 페이지에 대한 문서 감사가 필요하다.
- 하드 줄바꿈된 다중 라인 소스 파일에서는 줄바꿈을 넘나드는 문구를 안정적으로 찾기 위해 줄바꿈에 관대한 검색(`grep -z`, 다중 라인 정규식, 또는 렌더링된 텍스트 확인)이 필요하다.

---

## 6. 추가 학습 리소스

### 핵심 키워드

| 키워드 | 설명 | 관련성 |
|-------|-----|-------|
| `MLXCEL_PAGED_ATTENTION_NATIVE` | 융합 페이지드 어텐션 커널과 gather-then-SDPA 참조 경로 사이의 디스패치를 강제 고정하거나 선택기에 위임하는 환경 변수 | 이번에 수정된 문서의 대상 |
| `resolve_dispatch_decision` | `src/lib/mlxcel-core/src/layers.rs`에 정의된, 라이브러리 전용 풀 진입점의 디스패치를 구현하는 Rust 함수 | 이 변수의 두 소비자 중 하나 |
| `resolve_paged_v2_dispatch` | `src/lib/mlxcel-core/src/layers.rs`에 정의된, 서버의 풀 기반 배치 페이지드 디코드 디스패치를 구현하는 Rust 함수 | 이슈 #899로 추가된 나머지 소비자 |

### 관련 PR/이슈

- PR #1119(이슈 #1104): `docs/environment-variables.md`에서 같은 오래된 주장을 수정하고, `docs/turbo-kv-cache.md`를 후속 작업으로 표시했다.
- 이슈 #899: 융합 v2 커널을 프로덕션 서버 디코드 경로로 만들고, 이 변수의 킬 스위치 의미를 정의했다.
- 이슈 #710: 라이브러리 전용 풀 진입점을 `mlxcel serve` 디코드 경로에서 제외했다. 지금 수정된 "서버 노브가 아니다"라는 해석의 출처다.
- 이슈 #1139: 이번 PR의 원 이슈.

---

## 7. 변경 요약

### 통계

| 항목 | 값 |
|-----|---|
| 변경된 파일 수 | 1 |
| 추가된 라인 | +16 |
| 삭제된 라인 | -6 |
| 테스트 추가 | 0 (문서 전용) |

### 카테고리별 변경

| 카테고리 | 변경 수 | 주요 내용 |
|---------|--------|----------|
| Documentation | 1 | `docs/turbo-kv-cache.md`의 한 문단을 다시 써서 `MLXCEL_PAGED_ATTENTION_NATIVE`의 현재 두 소비자를 서술하고 #710 폐기 사실을 역사적 기록으로 격하 |

### 관련 커밋

| Hash | Type | Message |
|------|------|---------|
| `d2e263d` | docs | docs: correct stale "not a server knob" claim in turbo-kv-cache.md |

---

## 8. 후속 조치

### 완료 필요

- 없음. 이 변수를 설명하는 네 페이지(`README.md`, `docs/CONTINUOUS_BATCHING.md`, `docs/environment-variables.md`, `docs/turbo-kv-cache.md`)가 이제 서로 일치한다.

### 모니터링 필요

- 없음. 런타임에 영향이 없는 문서 전용 변경.

### 향후 개선 사항

- 식별된 사항 없음.

---

## 부록

### A. 테스트 결과

- `docs/`와 `README.md` 전체에 대해 `not a server\s+knob`을 찾는 (Python 기반, 줄바꿈에 관대한) 검색: 3건 모두 역사적 맥락(`docs/environment-variables.md`의 `History:` 절, `docs/adr/0001-paged-attention-gather-vs-fused-kernel.md`의 `### Decision` 섹션, 그리고 `docs/turbo-kv-cache.md`에 새로 추가된 `History:` 문장). 현재형 주장은 더 이상 남아 있지 않다.
- `python3 scripts/ci/check_cross_repo_refs.py`: 통과. 3자리 이상의 맨 `#NNN` 참조가 새로 추가되지 않았음을 확인.
- `git diff --stat`: `docs/turbo-kv-cache.md`만 변경되었음을 확인.

### B. 성능 벤치마크

해당 없음.

### C. 참고 자료

- 이슈 #1139 (이번 PR의 원 이슈)
- PR #1119 / 커밋 `cf4e22cd` (이번 PR이 재사용한 수정된 서술)
- 이슈 #899, #710
