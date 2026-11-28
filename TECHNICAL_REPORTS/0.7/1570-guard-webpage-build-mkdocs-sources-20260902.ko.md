# 기술 보고서: PR #1570 - chore(build): guard webpage-build against missing MkDocs sources

**작성일**: 2026-09-02
**작성자**: mlxcel maintainers
**리뷰어**: 자동화된 pr-reviewer, pr-security-checker 검토 완료
**상태**: 완료
**언어**: Makefile (GNU Make, POSIX 셸)
**위험도**: Low

---

## 요약

`make webpage-build`는 열세 개의 `docs-*` 타겟이 이미 방어하고 있던 것과 동일한, 충족되지 않은 MkDocs 매뉴얼 소스 의존성을 그대로 갖고 있었지만 `docs-guard` 전제 조건이 도입되기 전부터 있던 타겟이라 이전 작업(#1111, PR #1122)의 적용 범위에서 빠져 있었다. 매뉴얼 소스가 없는 체크아웃에서는 `webpage/site/public/en/manual`과 `webpage/site/public/ko/manual`을 먼저 삭제한 뒤에야 zensical에서 알기 어려운 오류를 내며 실패했다. 이번 PR은 `webpage-build`에 기존 패턴과 동일한 방식으로 `docs-guard`를 전제 조건으로 추가하고, 이웃한 `webpage-deploy` 타겟에 대해서는 질문을 미해결로 남기지 않고 근거를 문서화한 의도적인 선택을 내렸다.

---

## 1. 문제 정의

### 1.1 배경

`mkdocs.yml`은 `docs_dir: docs/en`을, `mkdocs.ko.yml`은 `docs_dir: docs/ko`를 지정한다. 이 디렉터리들과 `docs/shared`, `docs/requirements.txt`, `docs/scripts`는 별도의 문서 트리에서 관리되며 이 저장소의 체크아웃에는 존재하지 않는다. 열세 개의 `docs-*` 타겟은 바로 이 이유로 이미 `docs-guard`를 전제 조건으로 선언하고 있어서, 각 타겟은 알아보기 어려운 `uv`, `ln`, zensical 오류 대신 즉시 설명과 함께 실패한다. `webpage-build`도 동일하게 존재하지 않는 `docs_dir`를 대상으로 zensical 빌드를 실행하지만, `docs-*` 타겟 자체가 아니었기 때문에 가드를 추가하던 작업의 범위에서 벗어나 있었고, 후속 작업으로 남겨졌다.

### 1.2 기존 문제점

- **파괴적인 동작 후 알아보기 어려운 실패 순서**: `webpage-build`의 레시피는 `rm -rf webpage/site/public/en/manual webpage/site/public/ko/manual`로 시작했다. 매뉴얼 소스가 없는 체크아웃에서는 이전에 빌드해 둔 매뉴얼 출력물(있다면)이 빌드 실패보다 먼저 삭제되었다. 실패 자체도 이 체크아웃에서는 보이지 않는 `docs_dir`를 대상으로 한 zensical 내부 오류로 나타났을 뿐, 실제 원인을 가리키는 메시지가 아니었다.
- **검토되지 않은 `webpage-deploy`**: `webpage-deploy`는 `scripts/deploy_webpage.sh`를 실행하는데, 이 스크립트는 pnpm으로 `webpage/site`를 빌드한 뒤 별도의 `mlxcel-releases` 원격 저장소의 `gh-pages` 브랜치로 정적 산출물을 강제 푸시한다. 이 스크립트는 zensical을 호출하지도, `docs_dir`를 읽지도 않으므로 `webpage-build`와 같은 방식으로는 실패할 수 없지만, 배포하려는 매뉴얼 산출물이 실제로 존재하는지도 확인하지 않는다. 이 부분은 원래의 가드 도입 작업에서 검토되지 않았다.

### 1.3 위험 평가

| 위험 | 영향 | 발생 가능성 |
|------|--------|------------|
| 기여자가 새 체크아웃에서 `make webpage-build`를 실행했다가 zensical `docs_dir` 오류를 진단하는 데 시간을 낭비 | Low (시간 손실뿐, 데이터 손실 없음) | Medium (비공개 문서 트리가 없는 모든 기여자) |
| `make webpage-deploy`가 `/en/manual`, `/ko/manual`이 없거나 오래된 상태인 사이트를 배포 | Low (다시 빌드하고 재배포하면 복구 가능, 오늘 시점 어떤 CI 경로도 두 타겟을 호출하지 않음) | Low (배포는 수동으로, 의도적으로 실행하는 유지보수 작업) |

---

## 3. 기술적 선택과 그 이유

### 3.1 `webpage-build`는 가드하되 `webpage-deploy`는 `docs-guard`로 가드하지 않는 선택

**맥락:**

이슈는 두 가지를 별도로 요청했다. `webpage-build`에 `docs-guard`를 추가하는 것(기존 패턴을 그대로 적용하면 되는 명확한 작업)과, 실패 양상이 구조적으로 다른 `webpage-deploy`에 대해서는 무엇을 할지 별도로, 의도적으로 결정하는 것이다.

**검토한 대안:**

| 옵션 | 장점 | 단점 |
|--------|------|------|
| A: `webpage-deploy`도 `docs-guard`로 가드 | 다른 가드된 타겟들과 겉보기에 일관됨 | 잘못된 전제 조건을 검사하게 된다. `docs-guard`는 빌드 입력인 `docs/en`을 확인하지만, `webpage-deploy`가 실제로 의존하는 것은 이미 존재해야 하는 빌드 출력인 `webpage/site/public/{en,ko}/manual`이다. 이전에 빌드된, 변경되지 않은 사이트를 재배포하려는 관리자나 다른 경로로 매뉴얼 출력물을 가져온 관리자를 이유 없이 막게 된다. |
| B: `webpage-deploy`를 아무 코멘트 없이 그대로 둠 | 동작 변경 위험 없음 | 이슈가 명시적으로 제기한 질문에 답하지 않은 채 남기고, 이후 읽는 사람에게 이 부분이 검토되었는지 그냥 놓친 것인지 알 방법을 남기지 않음 |
| **채택: C. 실제 매뉴얼 출력물 존재 여부를 확인하는 비차단 경고를 추가하고, Makefile에 근거를 코멘트로 기록** | 실제 전제 조건을 검사하고, 변경되지 않은 산출물의 정당한 재배포를 막지 않으며, 결정 자체를 Makefile 안에 문서로 남김 | 매뉴얼 소스가 없는 상태에서 처음 배포할 때 매뉴얼 페이지가 없는 사이트가 배포되는 것을 막지는 못함. 경고만 할 뿐임 |

**근거:**

`docs-guard`의 역할은 `docs-*` 타겟(그리고 구조적으로 동일한 `webpage-build`)이 이 체크아웃에서는 애초에 실행될 수 없는 이유를 설명하는 것이다. `webpage-deploy`는 다르다. 이 타겟을 뒷받침하는 스크립트는 매뉴얼이 빌드되었는지와 무관하게 항상 성공하므로, 강제 실패를 두려면 실제로 존재하지 않는 조건을 만들어 내야 한다. 그리고 그 조건으로 자연스럽게 떠오르는 `docs/en` 부재는 실제로 배포가 올바른지를 결정하는 요소가 아니다. `webpage/site/public/en/manual`과 `webpage/site/public/ko/manual`을 직접 확인하는 쪽은 실제로 중요한 대상을 측정하므로, 매뉴얼 소스가 없는 체크아웃에서 `webpage-build`를 한 번도 실행하지 않은 경우, 성공적으로 실행한 경우, 다른 경로로 매뉴얼이 들어온 경우(예: 복사해 온 빌드 산출물) 모두에서 경고 내용이 정확하다.

**트레이드오프:**

이 경고는 매뉴얼 소스가 없는 새 체크아웃에서의 첫 배포를 막지 않으므로, 경고를 무시하는 관리자는 여전히 매뉴얼 페이지가 404인 사이트를 배포할 수 있다. 강제 실패라는 대안은 검토했지만, CI 연동이 없고 수동으로만 트리거되는 배포 스크립트에서 정당한 재배포에 오작동할 위험이 더 현실적인 경우로 판단되어 기각했다.

---

## 4. 구현 세부사항

### 4.2 주요 코드 변경

**파일: `Makefile`**
```makefile
# 변경 전
.PHONY: webpage-build
webpage-build: ## Build download webpage (static export)
	@echo "$(CYAN)Building documentation for webpage...$(RESET)"
	rm -rf webpage/site/public/en/manual webpage/site/public/ko/manual
	uv run zensical build -f mkdocs.yml -d webpage/site/public/en/manual
	...

.PHONY: webpage-deploy
webpage-deploy: ## Deploy download webpage to GitHub Pages
	@echo "$(CYAN)Deploying webpage...$(RESET)"
	./scripts/deploy_webpage.sh

# 변경 후
.PHONY: webpage-build
webpage-build: docs-guard ## Build download webpage (static export) (manual sources not in this checkout)
	@echo "$(CYAN)Building documentation for webpage...$(RESET)"
	rm -rf webpage/site/public/en/manual webpage/site/public/ko/manual
	uv run zensical build -f mkdocs.yml -d webpage/site/public/en/manual
	...

# webpage-deploy는 의도적으로 docs-guard 뒤에 두지 않았다. deploy_webpage.sh는
# zensical을 호출하지도, docs_dir를 읽지도 않으므로 webpage-build와 같은 방식으로는
# 실패할 수 없고, docs-guard는 빌드 입력인 docs/en을 확인할 뿐 deploy가 실제로
# 읽는 매뉴얼 출력물을 확인하지 않는다. ...
.PHONY: webpage-deploy
webpage-deploy: ## Deploy download webpage to GitHub Pages
	@echo "$(CYAN)Deploying webpage...$(RESET)"
	@if [ ! -d webpage/site/public/en/manual ] || [ ! -d webpage/site/public/ko/manual ]; then \
		echo "$(YELLOW)Warning: webpage/site/public/en/manual or .../ko/manual is missing.$(RESET)"; \
		echo "  Run 'make webpage-build' first, or the deployed site will be missing (or serving a stale copy of) the manual pages."; \
	fi
	./scripts/deploy_webpage.sh
```

**변경 이유:** `docs-guard`를 전제 조건으로 선언하는 것은 기존 열세 개 `docs-*` 타겟이 사용하는 것과 동일한 메커니즘이다. Make는 `docs-guard`가 파일 어디에 정의되어 있든 전제 조건을 정상적으로 해석하므로, 이슈의 기술적 고려사항에 이미 적혀 있던 대로 Makefile 섹션을 재배치할 필요가 없었다. `webpage-deploy`의 경고는 Makefile 상단에 이미 정의된 `$(YELLOW)`/`$(RESET)` 색상 변수를 그대로 쓰고 항상 종료 코드 0을 반환하므로, `./scripts/deploy_webpage.sh`가 실행되는지 여부를 절대 바꾸지 않는다.

---

## 7. 변경 요약

### 통계

| 항목 | 값 |
|------|-------|
| 변경된 파일 | 1개 (`Makefile`) |
| 추가된 줄 | +14 |
| 삭제된 줄 | -1 |
| 추가된 테스트 | 0개 (Makefile 전용 변경이라 직접 `make` 실행으로 검증, 부록 A 참고) |

### 카테고리별 변경

| 카테고리 | 개수 | 요약 |
|----------|-------|---------|
| 빌드 도구 | 2 | `webpage-build`에 `docs-guard` 전제 조건과 갱신된 도움말 문자열 추가, `webpage-deploy`에 비차단 사전 조건 경고와 설명 코멘트 추가 |
| 문서화 | 1 | `make help`가 이제 `webpage-build`의 의존성을 타겟을 직접 실행해서 발견하게 두는 대신 미리 알려줌 |

### 관련 커밋

| 해시 | 유형 | 메시지 |
|------|------|---------|
| `2375ee0` | chore | chore(build): guard webpage-build against missing MkDocs sources |

### 관련 이슈

- 이슈 #1138: chore(build): guard `webpage-build` against the missing MkDocs manual sources, 이 PR로 종료됨.
- 이슈 #1111 / PR #1122: 열세 개 `docs-*` 타겟에 `docs-guard`를 추가한 작업으로, 이번 PR은 그 작업이 다루지 않았던 타겟 하나에 동일한 가드를 확장한 것이다.

---

## 8. 후속 조치

### 필수

없음. 이슈 #1138의 다섯 가지 완료 조건이 모두 충족되고 검증되었다.

### 모니터링 필요

없음. 저장소의 모든 `.github/workflows/*.yml`, `*.sh`, `*.md`에서 `webpage-build`와 `webpage-deploy` 두 타겟 이름을 모두 검색한 결과 어떤 CI 워크플로도 이 타겟들을 호출하지 않으므로, `webpage-build`의 새 강제 실패는 자동화를 깨뜨릴 수 없고 `webpage-deploy`의 경고는 순전히 참고용이다.

### 향후 개선

- `webpage-deploy`의 경고는 신선도가 아닌 존재 여부만 확인하므로, 몇 달 된 `webpage/site/public/en/manual` 디렉터리도 현재 `docs/en` 트리 대비 내용이 오래되었더라도 조용히 통과한다. 오래되었지만 존재는 하는 매뉴얼이 실제로 반복되는 문제가 된다면 향후 빌드 타임스탬프나 체크섬 비교를 추가할 수 있다.
- `make help`는 이제 `webpage-build`를 필터링된 두 카테고리("Help & Documentation", "Test Targets")에 추가로 나열하는데, 이는 새 도움말 문구에 `doc`과 `check`라는 부분 문자열이 포함되어 있기 때문일 뿐이다. 기존의 열세 개 가드된 `docs-*` 타겟 모두가 이미 보이는 동일한 카테고리 필터링 동작의 부산물이다. 미관상의 문제일 뿐 기존 동작과 일관되며, 이번 PR에서는 다루지 않았다.

---

## 부록

### A. 테스트 결과

이번 변경은 애플리케이션 코드가 아니라 Makefile 가드이므로, `docs/en`이 없는 체크아웃에서 직접 `make`를 실행해 모든 검증을 수행했다.

- 현재 체크아웃에서 `make webpage-build`는 `rm -rf`가 실행되기 전 `docs-guard`에서 멈춘다(종료 코드 2). `ls webpage/site/public/`을 실행 전후로 확인한 결과 내용이 동일했으며(`brands/`만 존재), 매뉴얼 디렉터리가 전혀 건드려지지 않았음을 확인했다.
- `make help | grep webpage-build`는 새로 추가된 `(manual sources not in this checkout)` 접미사를 보여준다.
- `make DOCS_MANUAL_DIR=<existing-directory> webpage-build`(실제 MkDocs 트리 없이 가드의 존재-확인 경로만 검증하기 위해 사용한 변수 오버라이드)는 `docs-guard`를 통과해 변경되지 않은 `rm -rf`와 `uv run zensical` 단계로 진행되며, 이 환경에 `zensical`이 설치되어 있지 않다는 이번 변경과 무관한 이유로만 실패한다. 이는 가드가 무조건적인 거부가 아니라 존재 여부 확인임을 확인해 준다.
- `make -n webpage-deploy`는 새 경고 블록이 문법적으로 유효한 Make/셸 코드임을 보여준다. 내장된 셸 조건문은 매뉴얼이 없는 현재 체크아웃에서 독립적으로도 실행해 보았으며, 두 경고 줄을 올바르게 출력했다.
- 독립적으로 실행된 `pr-reviewer`와 `pr-security-checker` 검토 모두 CRITICAL, HIGH, MEDIUM 등급 발견 사항이 없었으며, `-j4`나 `-k` 옵션 아래에서도 가드가 조용히 우회되지 않음을, 경고가 항상 종료 코드 0을 반환하므로 배포 스크립트 자체의 `set -e`에 절대 간섭하지 않음을, 이 PR이 PR 이전 동작 대비 `webpage-build`의 파괴적인 동작 범위를 순수하게 줄였음을 확인했다.

### C. 참고자료

- 이슈 #1138 (이 PR의 출발점).
- 이슈 #1111 및 PR #1122 (열세 개 `docs-*` 타겟에 걸친 원래의 `docs-guard` 도입 작업).
- `scripts/deploy_webpage.sh` (`webpage-deploy`가 호출하는 스크립트).
