# 기술 보고서: PR #1173 - fix(test): skip partial Muse Glimmer checkpoints instead of panicking

**작성일**: 2026-08-16
**작성자**: AI Code Reviewer
**상태**: 완료
**언어**: Rust
**위험도**: Low

---

## 요약

`pinned_post_tower_weight_roots_and_shapes_match_published_contract`는 고정 체크포인트의 인덱스 파일이 없을 때만 온전히 건너뛰었다. 인덱스는 있는데 나머지가 없거나 깨진 경우, 즉 `hf download`나 `mlxcel download`가 중단된 상태에서는 이유를 알려주며 건너뛰는 대신 불투명한 `unwrap`으로 패닉했다. 이 PR은 문제가 된 파일 이름을 담아 건너뛰는 가용성 사전 검사를 추가하되, 진짜 계약 위반은 전부 실패로 남긴다.

핵심 제약은 이것이었다. 잘못된 체크포인트를 조용한 skip으로 바꾸면 고치려던 패닉보다 엄격히 더 나쁘다. 이 테스트가 강제하려는 계약을 영구히 무력화하기 때문이다. 그래서 구현은 두 범주를 관례가 아니라 구조로 분리한다. `load_pinned_checkpoint`는 가용성 문제로만 거부할 수 있고, `assert_post_tower_contract`는 읽을 수 없는 shard 헤더를 제외하면 skip 경로가 없는 어서션 본문이다.

리뷰에서 결함 두 건을 찾아 머지 전에 고쳤다. 둘 다 최초 구현이 무언가를 그 경계의 잘못된 쪽에 두었거나 뒤에 위험을 남긴 경우였다.

---

## 1. 문제 정의

### 1.1 배경

이슈 #1161은 PR #1157(#1155 구현)의 보안 리뷰 LOW 지적에서 나왔다. 그 PR은 인덱스 부재 가드를 추가하면서 부분 체크포인트 경우는 의도적으로 범위 밖에 뒀다. 형제 로딩 테스트 `pinned_weight_index_classifies_each_source_weight_once`는 인덱스 파일 자체만 읽으므로 같은 빈틈이 없다.

### 1.2 기존 문제점

- **문제 1**: 인덱스 존재 가드를 지나면 테스트는 모든 읽기를 unwrap했다. `config.json`, 인덱스 내용, `index["weight_map"].as_object()`, 그리고 `safetensors_shape`를 통한 모든 참조 shard. 그 헬퍼 자체도 `File::open`, `read_exact` 두 번, 헤더 파싱을 unwrap했다.
- **문제 2** (리뷰에서 발견): 가용성 사전 검사가 `config.json`을 곧바로 `MuseGlimmerConfig`로 역직렬화했다. 그래서 JSON으로는 유효하지만 스키마에 맞지 않는 config가 "사용 불가"로 보고되어 skip됐다. 고정 config의 스키마 드리프트나 고정 디렉터리에 다른 모델이 들어온 상황이야말로 이 계약 테스트가 잡아야 할 괴리이므로, 실패하는 쪽에 있어야 한다.
- **문제 3** (보안 검토에서 발견): 새 헤더 길이 가드가 파일 크기 기준뿐이었다. `header_len > file_len.saturating_sub(8)`은 파일 전체 크기만 한 헤더 선언도 허용하는데 고정 shard는 50 GB와 9.6 GB다. 따라서 손상된 8바이트 접두부가 검사를 통과해 `vec![0u8; header_len as usize]`에 도달할 수 있다. 결과는 이유도 출력하지 않고 잡을 수도 없는 `handle_alloc_error` abort이거나, 체크포인트를 보유한 바로 그 장비에서 수 기가바이트의 텐서 페이로드를 호스트 메모리로 끌어오는 것이다. abort는 이 PR이 없애려는 불투명한 unwrap보다 엄격히 더 나쁘다.

### 1.3 위험성

| 위험 | 영향 | 발생 가능성 |
|------|------|------------|
| 부분 체크포인트에서 불투명한 패닉이 나고 코드 결함으로 오독됨 | Low | Medium (중단된 다운로드마다) |
| 잘못된 체크포인트가 조용히 skip되어 계약 검사가 영구 무력화 | Medium | Low (경계 설계와 문제 2 수정으로 차단) |
| 손상된 헤더가 체크포인트 보유 장비에서 잡을 수 없는 할당 abort 유발 | Medium | Low (절대 상한으로 차단) |

---

## 2. 기술적 검토 사항

### 2.1 보안 관점

보안 검토는 MEDIUM 1건(위 문제 3)을 제기하고, 프로덕션 리더가 이미 쓰는 가드를 그대로 옮겨 고쳤다. `read_safetensors_header_bytes`(`src/lib/mlxcel-core/src/weights.rs:196`)는 동일한 8바이트 접두부를 `MAX_HEADER_BYTES = 256 * 1024 * 1024`로 제한하며 "reject absurdly large headers to avoid OOM"이라는 근거를 단다. 이 테스트 리더가 트리에서 그런 상한이 없는 유일한 헤더 리더였다. 옮겨온 상수는 파일 기준 경계보다 먼저 검사하고, `safetensors_shape_refuses_a_header_over_the_allocation_ceiling`은 작은 임시 파일에 `u64::MAX`를 선언하므로 테스트 자체는 큰 할당을 하지 않는다. 실제 고정 shard는 9.6 GB 파일에 11,720바이트 헤더를 기록하므로 약 네 자릿수의 여유가 있어 정상 체크포인트가 거부될 일은 없다.

나머지는 모두 깨끗함을 확인했다. 추적한 모든 경로에서 경계가 유지된다. `weight_map`이 없거나 객체가 아니면 빈 맵이 되어 weight-root 어서션이 실패하고, 정상 파싱된 맵에서 고정 키가 빠지면 루프에서 건너뛴 뒤 같은 어서션이 보고하며, 키가 문자열이 아닌 값에 매핑되면 "must name a shard" 패닉에 걸리고, 헤더가 파싱되지만 해당 키의 shape 항목이 없으면 `Err`가 아니라 패닉한다. 환경변수 읽기는 크레이트 전역 env 락을 블록 스코프로 잡고 `assert!` 전에 해제하므로 실패하는 어서션이 락을 오염시킬 수 없고, 그 함수에서 도달 가능한 어떤 코드도 비재진입 뮤텍스를 다시 잡지 않는다. 파일 헤더만 읽고 텐서 페이로드는 절대 읽지 않으며, 모든 `TempDir`이 살아 있는 지역 변수에 묶여 있어 `should_panic` 테스트도 되감기 시 정리된다.

**발견된 이슈:**
| 이슈 | 심각도 | 상태 |
|------|--------|------|
| 스키마 불일치 `config.json`이 실패가 아니라 skip | Medium | 수정 (`3336732c`) |
| 헤더 할당이 파일 크기로만 제한되어 50 GB shard의 손상된 길이가 프로세스를 abort시킬 수 있음 | Medium | 수정 (`3c61dd21`) |
| `contract_assertion_still_fails_on_a_wrong_recorded_shape`가 세 가지 패닉이 모두 내는 맨 키 이름을 기대 | Low | 수정 (`3336732c`) |
| `pinned_precheck_leaves_an_absent_weight_map_...`에 형제 케이스와 달리 `should_panic` 짝이 없었음 | Low | 수정 (`3336732c`) |

### 2.2 성능 관점

프로덕션 경로를 건드리지 않는다. 고정 검사는 파일 헤더만 읽으므로 60 GB 체크포인트에도 빠르다. 모듈 전체가 약 0.01초에 끝난다.

### 2.3 호환성/의존성 관점

- **호환성 파괴**: 없음
- **신규 의존성**: 없음. `tempfile`은 이미 dev-dependency였다(`Cargo.toml:308`)
- **호환성**: 프로덕션 로딩 경로에 영향 없음

### 2.4 코드 품질 관점

- **테스트 커버리지**: 모듈 기준 5개에서 23개로. 모든 skip 분기에 대한 합성 커버리지와 계약 위반이 여전히 실패한다는 실행 가능한 증명이 추가됐다
- **코드 복잡도**: unwrap으로 채워진 본문 하나 대신 책임이 하나씩인 명명된 함수 셋
- **기술 부채**: 감소

---

## 3. 기술적 선택과 그 이유

### 3.1 이슈가 제시한 두 선택지 대신 하이브리드

**고려한 대안:**

| 선택지 | 장점 | 단점 |
|--------|------|------|
| 선택지 1: 존재 여부 사전 검사만 | 단순 | 존재하지만 다운로드 중 헤더가 잘린 shard를 못 봄 |
| 선택지 2: 모든 unwrap을 skip-with-reason으로 | 손상까지 커버 | 인덱스를 두 번 읽게 되어 이슈가 명시적으로 배제한 방식 |
| **채택: 하이브리드** | 인덱스 1회 파싱을 어서션과 공유하면서 손상까지 커버 | 구조가 조금 늘어남 |

**선택 이유:** `load_pinned_checkpoint`가 인덱스를 한 번 파싱해 `weight_map`을 어서션 본문에 넘기고, `safetensors_shape`는 `Result`를 반환해 헤더 손상도 덮는다.

### 3.2 가용성과 계약을 관례가 아니라 구조로 분리

**선택 이유:** 어서션은 이미 검증된 `PinnedCheckpoint`를 받는 `assert_post_tower_contract`에 있다. 이 함수에 도달했다는 것은 필요한 모든 파일이 읽을 수 있음이 확인됐다는 뜻이므로, 여기서 실패하는 것은 체크포인트가 계약과 어긋난다는 의미다. 그 안의 유일한 skip인 "읽을 수 없는 shard 헤더"는 구성상 가용성에 한정된다. `safetensors_shape`가 `Err`를 내는 경우는 열기 실패, 메타데이터 실패, 잘림, 상한이나 파일 크기를 넘는 선언 길이, JSON이 아닌 헤더뿐이다. 헤더가 파싱되지만 shape가 없거나 차원이 정수가 아니면 패닉한다.

### 3.3 `MuseGlimmerConfig` 역직렬화를 어서션 본문으로 미룸

**선택 이유:** 문제 2의 수정이다. 사전 검사는 이제 `config.json`을 순수 JSON으로만 파싱하므로, 중단된 다운로드가 남기는 잘린 config는 문법 오류이자 실제 가용성 문제로 여전히 skip된다. 반면 파싱은 되지만 스키마에 맞지 않는 config는 어서션 본문의 `serde_json::from_value::<MuseGlimmerConfig>`에 도달해 패닉한다. 분리를 명시하려고 `PinnedCheckpoint`가 원본 `Value`를 들고 다닌다. 이 수정은 헛돌지 않는다. `MuseGlimmerConfig.text_config`와 그 내부 필드에는 serde 기본값이 없어서(`src/models/muse_glimmer_config.rs:26`) 불일치 config는 실제로 역직렬화에 실패한다.

### 3.4 선택 항목이던 환경변수 게이트를 구현

**선택 이유:** `MLXCEL_REQUIRE_PINNED_CHECKPOINTS=1`은 모든 skip을 실패로 바꾸므로, 손상된 체크포인트가 이를 강제할 수 있는 유일한 장비에서 계약 검사를 영구히 무력화하는 일을 막는다. 크레이트 전역 env 락 아래에서 읽어 `muse_glimmer_startup_guard_tests`가 세운 관례를 따른다. 이 크레이트는 edition 2024이고 형제 테스트가 같은 락 아래 `unsafe set_var`로 환경을 바꾸기 때문이다. 결과적으로 정상 경로를 증명하는 가장 깔끔한 수단이 되기도 했다. 부록 참고.

### 3.5 별도 파일로 분리

**선택 이유:** 고정 검사와 그 합성 커버리지는 fusion 수치 테스트와 별개 관심사이고, `#[cfg(test)] #[path = "..."] mod` 형제 패턴은 이미 `src/vision/` 전반에서 쓰인다. 다만 PR과 커밋 본문은 분리 근거로 "500줄 제한"을 들었는데, 저장소 자체 코드 구조 지침의 기준은 800줄이므로 그 근거 서술은 틀렸다. 분리 자체는 타당하다.

---

## 4. 구현 상세

### 4.1 주요 코드 변경

**파일: `src/vision/encoders/muse_glimmer_fusion_pinned_tests.rs`** (신규)
```rust
// 계약 위반을 절대 보아서는 안 되는 가용성 거부 지점.
fn load_pinned_checkpoint(model_dir: &Path) -> Result<PinnedCheckpoint, String> {
    let index_text = read_checkpoint_file(&index_path)?;
    let index: Value = serde_json::from_str(&index_text)
        .map_err(|err| format!("{} does not parse as JSON: {err}", index_path.display()))?;
    // ... config는 순수 JSON으로만 파싱 ...
    // `weight_map`이 없거나 객체가 아니면 파일 누락이 아니라 잘못된 인덱스다.
    // 빈 맵을 넘겨 weight-root 어서션이 계약 위반으로 보고하게 한다.
    let weight_map = index["weight_map"].as_object().cloned().unwrap_or_default();
    // ... 참조된 각 shard의 존재를 확인하고, 없으면 그 이름을 담아 반환 ...
}
```

```rust
// 할당 전에 선언된 헤더 길이를 제한한다.
if header_len > MAX_SAFETENSORS_HEADER_BYTES { /* Err */ }
if header_len > file_len.saturating_sub(8) { /* Err */ }
let mut header = vec![0u8; header_len as usize];
```

**변경 이유:** 기존 `safetensors_shape`는 검증되지 않은 8바이트 리틀엔디언 읽기 결과로 곧장 `vec![0u8; header_len]`을 했다.

---

## 7. 변경 요약

### 통계
| 항목 | 값 |
|------|-----|
| 변경 파일 | 3 |
| 추가 줄 | +609 |
| 삭제 줄 | -79 |
| 추가 테스트 | 18 (모듈 기준 5 → 23) |

### 카테고리별 변경

| 카테고리 | 개수 | 요약 |
|----------|------|------|
| 테스트 견고성 | 1 | 부분 체크포인트가 패닉 대신 이유를 담아 skip |
| 테스트 정확성 | 4 | 계약 위반이 여전히 실패함을 증명, `should_panic` 2건 조임 |
| 보안 | 1 | 할당 전 safetensors 헤더 절대 상한 |
| 도구 | 1 | `MLXCEL_REQUIRE_PINNED_CHECKPOINTS=1`이 skip을 실패로 전환 |

### 관련 커밋
| 해시 | 유형 | 메시지 |
|------|------|--------|
| `5274e7e5` | fix(test) | skip partial Muse Glimmer checkpoints instead of panicking |
| `3336732c` | test | fail rather than skip on a schema-mismatched pinned config |
| `3c61dd21` | fix(test) | cap the safetensors header this reader will allocate |

---

## 8. 후속 조치

### 완료 필요
- [ ] 없음. 선택 항목을 포함한 수용 기준 네 가지 모두 충족

### 향후 개선 사항
- `pinned_weight_index_classifies_each_source_weight_once`(`src/loading/vlm_muse_glimmer_tests.rs:473`)는 `MLXCEL_REQUIRE_PINNED_CHECKPOINTS`를 따르지 않으므로, 체크포인트 보유 장비에서 두 고정 가드 중 하나만 강화된 상태다. 별도 이슈로 발행할 만하며 #1161 범위 밖이다.
- 읽을 수 없는 shard에서 루프 중간에 조기 반환하므로, `fc1` shard를 읽지 못하면 `fc2`나 `vision_projection`의 잘못된 shape가 가려진다. skip 의미상 내재적이고 환경변수 게이트로 한정된다. 읽기 가능 여부를 먼저 확인하고 마지막에 skip하면 더 낫다.
- `should_panic` 테스트 두 건은 여전히 "must name a shard" 패닉도 내는 맨 키 이름을 기대한다. 두 대안 모두 skip이 아니라 실패이므로 불변식은 유지된다. 더 조이려면 `assert_eq!` 출력 형식에 더 의존해야 한다.

---

## 부록

### A. 테스트 결과

| 명령 | 결과 |
|------|------|
| `cargo test --lib vision::encoders::muse_glimmer_fusion` | 23 passed, 0 failed |
| `MLXCEL_REQUIRE_PINNED_CHECKPOINTS=1 cargo test --lib vision::encoders::muse_glimmer_fusion` | 23 passed, 0 failed |
| `cargo clippy --lib --tests -- -D warnings` | clean |
| `cargo fmt --check` | clean |

### B. 수용 기준을 증명한 방법

여기서 쓴 장비에는 고정 체크포인트가 완전히 준비돼 있어(`models`는 `/home/inureyes/models`를 가리키는 gitignore된 심볼릭 링크이고 `config.json`, 인덱스, 50 GB와 9.6 GB 두 shard가 모두 있음) 모든 기준을 직접 시험할 수 있었다.

- **기준 2, 완전한 체크포인트에서 계약이 그대로 검증됨.** 환경변수 게이트가 증거다. `MLXCEL_REQUIRE_PINNED_CHECKPOINTS=1`로 실행하면 모든 skip이 실패로 바뀌므로, 그 상태에서 고정 테스트가 통과했다는 것은 실제 체크포인트를 읽어 weight-root와 shape 계약을 단언했다는 뜻이지 건너뛴 것이 아니다.
- **기준 3, 인덱스 부재 시 기존 동작 유지.** `models/`가 없는 디렉터리에서 테스트 바이너리를 실행해 `models/mlx/muse-glimmer-30b/model.safetensors.index.json is not present`로 skip함을 확인했다.
- **선택 기준 4.** 같은 실행에 게이트를 켜니 skip 대신 실패했다.
- **기준 1, 부분 체크포인트가 이유를 담아 skip.** 합성 테스트가 덮는다. `tempfile::tempdir()`에 일회용 체크포인트를 만든다. 작업 전 구간에서 `models/` 아래 무엇도 옮기거나 이름을 바꾸거나 지우거나 자르거나 쓰지 않았고, 이후 두 shard가 원래 타임스탬프 그대로 온전함을 확인했다.

### C. 참고 자료
- 이슈 #1161(명세), PR #1157(인덱스 부재 가드를 추가했고 그 리뷰가 이 이슈를 발행), #1116
- `src/vision/encoders/muse_glimmer_fusion_pinned_tests.rs`(신규 모듈), `src/lib/mlxcel-core/src/weights.rs:196`(옮겨온 프로덕션 헤더 상한), `src/models/muse_glimmer_config.rs:26`(스키마 검사가 헛돌지 않는 이유)
- PR #1173의 리뷰 및 보안 코멘트
