# 기술 보고서: PR #1407 - test-fast에서 ThinLTO 비활성화

**날짜**: 2026-08-25
**작성자**: mlxcel contributors
**상태**: 완료
**언어**: Rust, TOML, Make, Markdown, YAML
**위험 수준**: 낮음

---

## 요약

PR #1407은 전체 macOS 워크스페이스 테스트 그래프가 ThinLTO 링크 중 Rust 내부 drop-glue 심볼을 찾지 못하고 반복적으로 실패하던 문제를 해결하기 위해 최적화 테스트 프로필 `test-fast`에서 LTO를 비활성화합니다. MLX 수치 동작에 필요한 `opt-level = 3`은 유지하고 배포용 `release` 프로필은 변경하지 않으면서, CI와 동일한 워크스페이스 게이트가 모든 타깃을 링크하고 실행하도록 복구했습니다.

---

## 1. 문제 정의

### 1.1 배경

이 저장소는 release 프로필의 fat LTO와 단일 코드젠 유닛이 다수의 통합 테스트 바이너리 링크를 지나치게 느리게 만들기 때문에 전체 워크스페이스 테스트 게이트를 `[profile.test-fast]`로 실행합니다. 그러나 빠른 프로필에도 ThinLTO가 명시되어 있어 대형 테스트 바이너리마다 크레이트 간 LTO 링크 경로를 사용했습니다.

### 1.2 관찰된 실패

`cargo test --workspace --profile test-fast --features metal,accelerate --no-run`은 테스트 실행 전에 `molmo_parity`, `qwen3_omni_moe_parity` 등의 타깃을 링크하다 실패했습니다. 링커는 `serde_json::Value`, `RotatingKVCache`, `KVCache` drop glue처럼 LLVM 내부 접미사가 붙은 Rust 심볼을 찾지 못했다고 보고했습니다.

`CARGO_BUILD_JOBS=1`에서도 실패가 재현되어 Cargo 링크 작업 간 동시성은 필수 조건에서 제외됐습니다. 더 작은 유닛 그래프로 구성된 단독 `qwen3_omni_moe_parity` no-run 빌드는 통과했지만 전체 워크스페이스 그래프는 실패했으므로, 테스트 소스 자체가 아니라 명시적 ThinLTO 워크스페이스 링크 형태가 원인임을 격리했습니다.

### 1.3 위험 평가

| 위험 | 영향 | 수정 전 가능성 |
|---|---|---|
| 워크스페이스 게이트가 테스트 판정을 만들지 못함 | 높음 | 높음 |
| 소스가 올바른 PR도 링크 실패 때문에 실패로 보임 | 높음 | 높음 |
| 테스트 수정 중 release 동작이 우발적으로 바뀜 | 높음 | 낮음, 프로필 assertion과 diff 검토로 방어 |

---

## 2. 기술 검토

### 2.1 정확성

`[profile.test-fast]`는 이제 `lto = false`를 설정하면서 `opt-level = 3`, `codegen-units = 16`, `incremental = true`, `strip = false`, 상속된 unwind 동작을 유지합니다. `[profile.release]`는 계속 `lto = true`, `codegen-units = 1`, `strip = true`, `opt-level = 3`, `panic = "unwind"`입니다.

Cargo 주석, Makefile 게이트 계약, 기여자 문서, 설치 문서, nightly workflow가 모두 테스트 프로필에서 크레이트 간 LTO를 비활성화한다고 일관되게 설명합니다. 배포 코드젠에서만 재현되는 결함을 위한 수동 `cargo test --release --features metal,accelerate` 우회 경로도 유지합니다.

### 2.2 보안과 성능

입력, 인증, 데이터 처리, 런타임 추론 경로는 변경하지 않습니다. 독립적인 정확성 검토와 보안·성능 검토에서 어떤 심각도의 문제도 발견되지 않았습니다.

의도한 성능 절충은 테스트 바이너리에만 한정됩니다. 게이트는 더 단순하고 신뢰할 수 있는 링크 그래프를 위해 release 전용 LTO 최적화를 포기합니다. 배포 산출물과 벤치마크는 계속 `[profile.release]`를 사용하며 `.github/workflows/release.yml`은 변경되지 않았습니다.

### 2.3 호환성과 의존성

- **호환성 파괴**: 없음.
- **새 의존성**: 없음.
- **Release 호환성**: 변경 없음.
- **테스트 산출물 호환성**: 테스트 바이너리의 LTO 기반 인라이닝과 배치는 release 바이너리와 다를 수 있으며, 이는 문서화된 의도적 차이입니다.

---

## 3. 기술적 선택과 그 이유

### 3.1 최적화 수치는 유지하고 테스트 시 LTO 제거

| 선택지 | 장점 | 단점 |
|---|---|---|
| 명시적 ThinLTO 유지 | 테스트에서도 크레이트 간 LTO 유지 | 전체 워크스페이스 그래프가 테스트 전에 실패할 수 있음 |
| 증분 컴파일 비활성화 | 다른 주요 빌드 축을 변경 | 재현된 크레이트 간 LTO 경로를 직접 제거하지 못하고 반복 개발 속도를 낮춤 |
| 게이트에 release 프로필 사용 | 배포 코드젠과 일치 | 다시간 링크 비용과 기존 nightly timeout을 다시 유발 |
| **선택: `test-fast.lto = false`** | 실패 링크 경로 제거, 최적화 수치 유지, 빠른 코드젠 유지 | release 전용 LTO 동작은 테스트하지 않음 |

`opt-level = 3`이 최적화된 MLX 수치 동작을 대표하기 위한 요구사항입니다. LTO는 그 테스트 의미론의 전제가 아니라 배포 최적화이므로 전역 최적화 수준을 낮추는 대신 두 관심사를 분리했습니다.

### 3.2 Release 검증을 별도 계약으로 유지

이 변경은 `test-fast`를 release와 동등한 프로필로 재해석하지 않습니다. 릴리스 workflow는 계속 fat LTO와 단일 코드젠 유닛으로 배포 산출물을 빌드하고, 기여자에게는 코드젠 특이 결함을 조사할 수 있는 문서화된 release 프로필 테스트 명령을 제공합니다.

---

## 4. 구현 상세

### 4.1 프로필 변경

```toml
[profile.test-fast]
inherits = "release"
lto = false
codegen-units = 16
strip = false
incremental = true
opt-level = 3
```

### 4.2 문서와 workflow 정렬

- `Cargo.toml`은 재현된 missing-symbol 실패와 release 격리를 기록합니다.
- `Makefile`은 전체 게이트가 ThinLTO를 쓰는 대신 크레이트 간 LTO를 제거한다고 설명합니다.
- `CONTRIBUTING.md`와 `docs/installation.md`는 테스트/release 절충과 수동 우회 경로를 설명합니다.
- `.github/workflows/nightly-verify.yml`은 nightly 게이트가 사용하는 동일한 운영 계약을 반영합니다.

---

## 5. 검증 증거

### 5.1 수정 전

- 전체 워크스페이스 no-run은 여러 통합 테스트 바이너리를 링크하다 LLVM 접미사가 붙은 Rust drop-glue 심볼을 찾지 못해 실패했습니다.
- Cargo 빌드 작업을 직렬화해도 실패를 막지 못했습니다.
- 단독 통합 테스트 빌드는 통과할 수 있었으므로 실패가 더 큰 워크스페이스 유닛 그래프에 의존함을 확인했습니다.

### 5.2 수정 후

- `cargo test --profile test-fast --features metal,accelerate --test molmo_parity --no-run`: 통과.
- `cargo test --profile test-fast --features metal,accelerate --test qwen3_omni_moe_parity --no-run`: 통과.
- `cargo test --workspace --profile test-fast --features metal,accelerate --no-run`: 통과, 수정된 프로필의 콜드 상태에서 9분 19초 만에 모든 워크스페이스 테스트 타깃 링크 완료.
- `cargo test --workspace --profile test-fast --features metal,accelerate --no-fail-fast -- --test-threads=1`: 모든 워크스페이스 테스트 및 doctest 바이너리에서 실패 0건으로 통과.
- `cargo clippy --workspace --all-targets --features metal,accelerate -- -D warnings`: 통과.
- `cargo fmt --all -- --check`: 통과.
- TOML assertion으로 수정된 테스트 프로필과 변경되지 않은 release 프로필을 확인했습니다.
- 포매팅, clippy, 의존성 정책, 크레이트 버전, 커널 dtype 키, 크로스 저장소 참조, OpenXLA 기능 컴파일을 포함한 GitHub CI 검사가 통과했습니다.

---

## 6. 변경 요약

| 항목 | 보고서 커밋 전 값 |
|---|---|
| 구현 변경 파일 | 5 |
| 추가 라인 | 21 |
| 삭제 라인 | 19 |
| 런타임 코드 변경 | 0개 파일 |
| 의존성 변경 | 0 |

| 범주 | 요약 |
|---|---|
| 빌드 프로필 | `test-fast`에서만 LTO 비활성화 |
| Release 동작 | 변경 없음 |
| 문서 | Cargo, Makefile, 기여자, 설치, nightly 안내 정렬 |
| 검토 | 정확성, 보안·성능, 최종화 검토에서 문제 없음 |

### 관련 커밋

| 해시 | 유형 | 메시지 |
|---|---|---|
| `b2204d5dc` | fix | test-fast에서 ThinLTO 비활성화 |

---

## 7. 학습 포인트

- 단독 Rust 테스트 빌드와 전체 워크스페이스 빌드는 서로 다른 코드젠 유닛 및 링크 그래프를 만들 수 있으므로 단독 통과만으로 워크스페이스 전용 링커 결함을 부정할 수 없습니다.
- 빌드 작업 직렬화는 동시성을 검증하지만 명시적 크레이트 간 LTO 자체가 트리거인지 여부는 검증하지 않습니다.
- 최적화 수치 테스트와 배포 바이너리 링크 최적화는 별도 계약입니다. 전자는 `opt-level = 3`으로 유지하고 release LTO는 배포 프로필에 남깁니다.
- 프로필 변경 시 `Cargo.toml`뿐 아니라 개발자와 CI가 운영 계약을 읽는 모든 문서를 함께 갱신해야 합니다.

---

## 8. 후속 조치와 모니터링

필수 후속 조치는 없습니다. 다음 nightly 워크스페이스 게이트의 안정적인 링크 시간을 관찰하고, fat LTO 또는 단일 코드젠 유닛에 의존할 수 있는 결함을 조사할 때는 release 프로필 우회 명령을 계속 사용합니다.

### 관련 작업

- Issue #1406: 워크스페이스 테스트 게이트에서 ThinLTO 비활성화.
- PR #1407: `test-fast`에서 ThinLTO 비활성화.
