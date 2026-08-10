# 기술 보고서: PR #1097 - Florence-2 large-ft upstream tracker 문서화

**작성일**: 2026-08-10
**상태**: 열린 PR 기준 작성 완료
**언어**: Markdown
**위험도**: Low

## 요약

PR #1097은 Florence-2 `large-ft` 계열의 실패 원인을 mlxcel loader 결함 후보에서 upstream MLX checkpoint 또는 conversion-family 문제로 더 정확하게 좁혀 문서화한다. 동시에 `mlx-community/Florence-2-base-ft`를 계속 권장 baseline으로 유지하고, 남은 조사는 focused upstream tracker를 통해 바로 추적할 수 있게 했다.

코드 변경은 없지만, published checkpoint가 load는 되는데 usable output을 내지 못하는 상황에서 사용자가 가장 먼저 보는 문서가 `docs/supported-models.md`라는 점에서 운영상 의미가 큰 수정이다.

## 1. 문제 정의

Issue #1085는 Florence-2 `-large-ft` checkpoint가 load는 되지만 degenerate output을 반환한다는 사실을 정리하고 있었다. 기존 support note도 mlx-vlm에서 같은 증상이 재현된다는 점은 적고 있었지만, 실제 upstream tracker 링크와 문제가 된 4-bit conversion의 provenance는 남기지 않았다.

그 상태에서는 독자가 여전히 mlxcel loader가 fault domain인지 추정해야 했고, 남은 원인 분석을 어디서 추적해야 하는지도 문서만으로는 바로 알 수 없었다.

## 2. 기술적 선택과 그 이유

### 2.1 제한 사항을 upstream checkpoint-family 문제로 재정의

수정된 문장은 이 동작이 mlxcel loader보다 published `large-ft` MLX checkpoint 또는 conversion family의 성질로 보인다고 명시한다. 이 표현은 issue #1085에서 수집된 증거와 맞는다. 같은 release에서 mlxcel과 upstream mlx-vlm이 동일한 bad output을 재현했기 때문이다.

### 2.2 구체적인 upstream tracker를 직접 연결

실제 후속 조치 경로는 막연한 "upstream issue"가 아니라 `Blaizzy/mlx-vlm#1840`이다. 이 링크를 `docs/supported-models.md`에 직접 넣어 support page가 막연한 경고가 아니라 actionable handoff가 되도록 했다.

### 2.3 마지막으로 검증된 working baseline 유지

문서는 upstream issue가 해결될 때까지 계속 `-base-ft`를 권장한다. 이렇게 해야 모든 Florence-2 variant가 동일하게 신뢰 가능하다는 잘못된 신호를 주지 않고, 실제로 검증된 baseline과 문서를 일치시킬 수 있다.

## 3. 변경 요약

| 항목 | 값 |
|------|----|
| 변경 파일 | 1 |
| 추가 라인 | 1 |
| 삭제 라인 | 1 |
| 범위 | 문서 전용 |

- `docs/supported-models.md`의 Florence-2 support note를 수정했다.
- `Blaizzy/mlx-vlm#1840` 직접 링크를 추가했다.
- 영향받은 release의 model card에 이미 기록된 provenance를 바탕으로 `mlx-community/Florence-2-large-ft-4bit`가 `prince-canuma/Florence-2-large-ft`를 `mlx-vlm 0.1.0`으로 변환한 결과라는 점을 남겼다.
- 문서상 working fallback으로 `mlx-community/Florence-2-base-ft` 권장을 유지했다.

## 4. 리뷰 발견 사항

| 발견 사항 | 심각도 | 해결 |
|-----------|--------|------|
| Upstream reproduction이 확인된 뒤에도 support page가 failure domain을 모호하게 남김 | Medium | 남은 문제를 mlxcel loader가 아니라 published `large-ft` MLX checkpoint 또는 conversion family 쪽으로 명확히 서술 |
| Support note에서 live upstream investigation으로 바로 갈 경로가 없음 | Low | 본문에 `Blaizzy/mlx-vlm#1840` 링크 추가 |

이번 PR은 prose만 바꾸므로 code path, security, performance 관련 finding은 없다.

## 5. 검증

- `git diff --check origin/main...HEAD`: 통과.
- 변경된 페이지에 대한 minimal MkDocs render: 통과. 임시 config로 문서 자체가 정상 렌더링됨을 확인했다.
- Live repository follow-through: 통과. 정리된 증거와 upstream 링크를 `lablup/mlxcel#1085`에 남겼고, focused upstream issue를 `Blaizzy/mlx-vlm#1840`으로 등록했다.
- Canonical repository docs build 구분: `mkdocs build -f mkdocs.yml -q`는 이 checkout에서 시작 자체가 불가능했다. 체크인된 config가 존재하지 않는 `docs/overrides`, `docs/en` 디렉터리를 참조하기 때문이다.
- `make docs-build` 구분: 이 환경에는 `zensical`이 없어 실행할 수 없었다.

핵심은 변경된 페이지 자체의 렌더링은 성공했지만, 저장소의 canonical docs pipeline은 PR #1097과 무관한 checkout 상태 제약 때문에 여기서 실행 불가능했다는 점이다.

## 6. 관련 작업

- PR #1097: https://github.com/lablup/mlxcel/pull/1097
- Issue #1085: https://github.com/lablup/mlxcel/issues/1085
- Upstream issue `Blaizzy/mlx-vlm#1840`: https://github.com/Blaizzy/mlx-vlm/issues/1840
