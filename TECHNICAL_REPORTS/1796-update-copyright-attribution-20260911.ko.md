# 기술 보고서: PR #1796 - 저작권 표기 정리

**작성일**: 2026-09-11
**상태**: 완료
**언어**: Rust, Python, C/C++, Metal, shell, text
**위험도**: 낮음

## 요약

이 PR은 프로젝트 소유 소스의 헤더를 Lablup Inc. 중심으로 정리하고 NOTICE 표기를 Lablup Inc. and contributors로 변경한다. 개인 저작권 표기를 소스 트리 전체에 반복하는 대신 패키지 메타데이터에 저자 provenance를 유지한다.

## 1. 문제 정의

PR #1728은 저장소 전반의 Apache 헤더 적용 범위를 완성해 소유 소스 영역의 표기를 하나로 맞췄다. 그 표기는 1,812개 파일에서 회사와 개인을 나란히 저작권자로 두어 프로젝트 저작권 소유와 저자 provenance를 혼합했다.

기존 헤더만 바꾸면 헤더 삽입 스크립트를 통해 이전 문구가 다시 들어올 수 있다. 반대로 모든 저자 정보를 제거하면 유용한 provenance까지 사라진다. 따라서 소스 헤더에는 회사 소유권, NOTICE에는 집합적 기여자 표기, 패키지 메타데이터에는 제한된 저자 정보를 두는 경계를 택했다.

## 3. 기술적 선택과 그 이유

세 가지 메타데이터 역할을 다음과 같이 분리했다.

| 위치 | 기록하는 정보 |
|------|---------------|
| 프로젝트 소유 소스 헤더 | `Copyright 2025-2026 Lablup Inc.` |
| `NOTICE` | `Copyright 2025-2026 Lablup Inc. and contributors.` |
| 패키지 메타데이터 | `mlxcel-core` 크레이트를 포함한 저자 provenance |

헤더 생성기도 같은 회사 단독 문구를 사용하므로 새로 적용되는 파일에서 이전 문구가 재발하지 않는다. 기존 서드파티 provenance 감지 동작은 바꾸지 않았다.

## 7. 변경 요약

| 항목 | 값 |
|------|----|
| 변경 파일 | 1,813개 |
| 추가 줄 | 1,814줄 |
| 삭제 줄 | 1,813줄 |
| 런타임 동작 변경 | 없음 |
| 새 의존성 | 없음 |

`mlxcel-core` 패키지 메타데이터의 명시적 항목을 제외하면 diff는 기계적 치환이다. 기존 공동 저작권 문구는 모두 제거했고, 남은 개인 표기는 패키지 저자 메타데이터와 Git 이력으로 한정했다.

## 부록: 검증

- `python3 tests/test_insert_apache_header.py -v`: 11개 테스트 통과
- `python3 scripts/insert_apache_header.py --check`: 모든 대상 파일의 라이선스 헤더 확인
- `cargo metadata --no-deps --format-version 1`: 유지한 저자 정보를 포함해 `mlxcel-core` 메타데이터 파싱 성공
- `git diff --check`: 통과
