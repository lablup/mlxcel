# 기술 보고서: PR #1170 - fix(cli): reject a DFlash drafter before the offline standalone load

**작성일**: 2026-08-16
**작성자**: AI Code Reviewer
**상태**: 완료
**언어**: Rust
**위험도**: Low

---

## 요약

`mlxcel generate --draft-model <dflash-drafter>`는 모든 DFlash drafter와 모든 target 조합에서 `Error: Weight not found: model.embed_tokens.weight`로 실패했다. 이 메시지는 실제 원인이 아니라 텐서 이름 하나만 지목한다. DFlash 체크포인트는 바인딩 시점에 target으로부터 `embed_tokens`와 `lm_head`를 빌려 쓰므로 둘 다 자체 보유하지 않는데, `config.json`은 여전히 평범한 `model_type`(`qwen3`)을 선언하고 있어 전체 standalone `LoadedModel` 로더로 그대로 라우팅됐다. 이번 PR은 체크포인트 자체가 갖는 DFlash 마커(중첩된 `dflash_config` 객체와 `architectures: ["DFlashDraftModel"]`)를 기준으로 구조적 프로브를 추가한다. resolve된 `DrafterKind`를 기준으로 삼지 않은 이유는, Gemma 4 assistant가 아닌 모든 drafter가 기본값으로 `Dflash`로 resolve되기 때문이다. classic drafter로 쓰이는 평범한 full model도 여기 포함된다. 이 프로브는 `-m/--model` detection 경로와 offline `--draft-model` 로드 경로 양쪽에서 DFlash drafter를 사전에 거부하며, 실제 문제를 지목하고 `mlxcel-server --draft-kind dflash`를 대안으로 안내하는 메시지를 낸다. 서버 경로는 drafter에 대해 애초에 `get_model_type`을 호출하지 않으므로 영향이 없고, 이는 non-regression 테스트로 고정됐다. 이번 finalization 단계에서는 새로 추가된 테스트 fixture의 예측 가능한 임시 디렉터리 사용 문제(M3)도 함께 수정했다.

---

## 1. 문제 정의

### 1.1 배경

이슈 #1168: DFlash drafter로 offline speculative decoding을 시도하면 `--draft-kind` 유무와 무관하게 항상 실패했다. offline 진입점에는 `DFlashGenerator` round loop이 없고, standalone 모델과 동일한 로더로 drafter를 그대로 밀어넣기 때문이다.

### 1.2 기존 문제점

- **문제 1**: `get_model_type`이 DFlash drafter의 `config.json`을 (체크포인트가 선언한 평범한 `model_type` 그대로) `ModelType::Qwen3` 등으로 분류했다. `model_metadata`는 이를 해당 family 로더로 라우팅했고, 로더는 첫 번째로 찾는 텐서(`UnifiedEmbedding::from_weights`가 찾는 `model.embed_tokens.weight`)가 없다는 이유로 실패했다. 정작 "이 디렉터리는 drafter다"라는 사실은 어디에도 보고되지 않았다.
- **문제 2**: 가장 단순해 보이는 해법인 resolve된 `DrafterKind` 기준 거부는 기존에 정상 동작하던 워크플로를 깨뜨렸을 것이다. `DEFAULT_DRAFTER_KIND`는 `Dflash`이고 `drafter_kind_by_model_type()`은 두 Gemma 4 assistant model_type만 매핑하므로, classic `--draft-model` drafter로 쓰이는 평범한 소형 full model(예: Qwen3 0.6B)도 자동으로 `Dflash`로 resolve된다. 이 조합은 현재 classic `SpeculativeGenerator` 경로에서 정상 동작한다.
- **문제 3** (이번 finalization 단계에서 발견, 보안 항목 M3): 새로 추가된 `src/commands/generate_tests.rs`의 `drafter_fixture_dir`가 `env::temp_dir()` 아래 예측 가능한 경로명을 만들고 `create_dir_all`로 생성했다. 이 방식은 이미 존재하는 디렉터리나 심볼릭 링크에 대해서도 성공해버리는 CWE-377/379 계열 문제이며, 정리(cleanup)도 성공 경로에서만 이루어져 어서션이 실패하면 fixture가 남는다. `TMPDIR`을 여러 job이 공유하는 Linux CI에서 특히 문제가 된다.

### 1.3 위험성

| 위험 | 영향도 | 발생 가능성 |
|-----|-------|-----------|
| offline DFlash `--draft-model`이 오도하는 오류 메시지와 함께 영구적으로 동작 불능 상태로 남음 | Medium (모든 DFlash drafter, 모든 target) | 수정 전 확정적 |
| 잘못된 판별 기준으로 거부하면 classic 소형 모델 drafter 워크플로가 깨짐 | High (판별 기준을 잘못 잡을 경우) | 체크포인트 구조를 기준으로 판별해 회피, 전용 control 테스트로 확인 |
| 공유 `TMPDIR` 환경에서 예측 가능한 임시 디렉터리 fixture가 선점되거나 심볼릭 링크로 악용됨 (M3) | Low (테스트 전용, 프로덕션 표면 없음) | Low이지만 공유 Linux CI 러너에서는 실재 |

---

## 2. 기술적 검토 사항

### 2.1 보안 관점

리뷰와 보안 검토는 오케스트레이터와 리뷰어가 finalization 이전에 완료했다. CRITICAL, HIGH 항목은 없었다. 이번 PR에서 새로 추가한 테스트 전용 fixture 헬퍼에 대해 MEDIUM 항목 하나(M3)가 지적됐다.

**발견된 이슈:**

| 이슈 | 심각도 | 상태 |
|-----|-------|-----|
| `drafter_fixture_dir`가 `env::temp_dir()` 아래 `create_dir_all`로 예측 가능한 경로를 생성(CWE-377/379), 어서션 실패 시 fixture가 남음 | Medium | Fixed (`37136aa6`) |

수정은 이 헬퍼와, 바로 아래에 있던 missing-config fixture의 동일한 인라인 패턴을 모두 `tempfile::tempdir()`로 교체한다. 이 함수는 무작위 이름의 안전한 권한을 가진 디렉터리를 만들고, 테스트가 어떤 경로로 종료되든 drop 시점에 그 디렉터리를 제거한다. 같은 PR에 포함된 `mlxcel-core`의 `drafter::dflash::config` 테스트가 이미 쓰던 패턴과 동일하며, 이로써 PR 내부의 불일치가 해소됐다.

### 2.2 성능 관점

없음. 이 변경은 hot inference 경로가 아니라 `config.json`을 `serde_json`으로 파싱하는 사전 로드 가드에 해당한다. 벤치마크는 필요하지 않았고 실행하지도 않았다.

### 2.3 호환성/의존성 관점

- **Breaking Changes**: offline `--draft-model <dflash-drafter>` 호출은 이전에도 실패했지만, 이제는 더 이른 시점에 설명이 담긴 오류로 실패한다. 이전에 정상 동작하던 호출 방식은 영향을 받지 않는다. offline DFlash round loop을 구성하는 테스트나 문서화된 워크플로는 애초에 존재하지 않았다.
- **새로운 의존성**: 수정 자체에는 없다. 같은 PR에서 이미 dev-dependency로 존재하던 `tempfile`이 수정된 테스트 fixture를 뒷받침한다.
- **호환성**: 서버 경로(`DFlashDrafter::load`, `resolve_drafter_kind`)는 drafter 경로에 대해 `get_model_type`을 전혀 호출하지 않는다. 트리 내 모든 `get_model_type` 호출 지점을 다시 확인해 이를 검증했다. 새로 추가된 통합 테스트는 detection이 이제 거부하는 것과 같은 fixture 디렉터리에 대해 `SpeculativeDispatch::resolve`가 여전히 DFlash variant를 반환함을 고정한다.

### 2.4 코드 품질 관점

- **테스트 커버리지**: 네 곳(`src/commands/generate_tests.rs`, `src/models/detection_tests.rs`, `src/lib/mlxcel-core/src/drafter/dflash/config.rs`, `tests/speculative_dispatch.rs`)에 새 테스트가 추가됐고, 각각 classic 경로가 계속 resolve되어야 함을 확인하는 평범한 full model control을 포함한다. mutation 테스트(새 detection 분기를 `if false && ...`로 막음)로 새 테스트들이 항진명제가 아니라 실제 어서션임을 확인했다. `-m`과 `--draft-model` 거부 테스트 모두 수정 전 분류로 실패했다.
- **코드 복잡도**: 구조적 프로브는 작은 순수 함수 두 개(`is_dflash_drafter_config`, `is_dflash_drafter_dir`)와 공유 오류 생성 함수 하나로 이루어진다. 기존 로드 경로의 제어 흐름은 바뀌지 않았다.
- **기술 부채**: 두 항목 모두 감소했다. CI에 영향을 주던 테스트 위생 불일치(M3)가 해소됐고, offline 진입점이 애초에 지원하도록 만들어지지 않은 종류의 체크포인트를 조용히 잘못 라우팅하는 문제도 사라졌다.

---

## 3. 기술적 선택과 그 이유

### 3.1 resolve된 `DrafterKind`가 아니라 체크포인트 구조를 판별 기준으로 삼음

**고려한 대안:**

| 옵션 | 장점 | 단점 |
|-----|-----|-----|
| `DrafterKind::Dflash`가 resolve되면 거부 | 코드가 적고 기존 resolve 로직을 재사용 | classic 소형 full model drafter 워크플로도 기본값으로 `Dflash`에 resolve되므로 이를 깨뜨림 |
| **선택: `config.json`에서 `dflash_config`와 `architectures: ["DFlashDraftModel"]` 존재 여부를 확인** | `DFlashConfig::from_json`과 HuggingFace `AutoModel` dispatch가 실제로 읽는 값과 정확히 일치. 평범한 full model에는 둘 다 없음 | resolve된 enum 값 하나 대신 독립적인 마커 두 개를 확인해야 함 |

**선택 이유**: resolve된 kind는 체크포인트에 대한 구조적 사실이 아니라 fallback 기본값일 뿐이다. 체크포인트 자체의 마커는 DFlash 로더와 HuggingFace 생태계 양쪽이 이미 권위 있는 값으로 취급하는 정보이므로, 이를 기준으로 삼으면 평범한 drafter를 오분류할 수 없다.

### 3.2 M3 수정: 임시 경로를 강화하는 대신 `tempfile::tempdir()`를 채택

**선택 이유**: 이 PR은 이미 `mlxcel-core`의 새 테스트에서 `tempfile::tempdir()` 패턴을 확립해 두었다. 기존 `env::temp_dir()` 경로를 강화하는 방법(고유 접미사, 권한 검사, panic을 포함한 모든 종료 경로에서의 명시적 정리)은 이미 검증된 crate가 RAII 가드로 제공하는 것을 재구현하는 셈이고, 종료 경로 하나를 놓칠 위험도 더 크다.

---

## 4. 구현 상세

### 4.1 Detection (`src/models/detection.rs`, `src/lib/mlxcel-core/src/drafter/dflash/config.rs`)

`get_model_type`은 이제 `model_type` 기반 dispatch보다 먼저, 공유 함수 `dflash_drafter_not_standalone_error`를 통해 구조적으로 DFlash인 디렉터리를 거부한다. 이 한 곳의 호출 지점이 `-m` 케이스, 서버 startup, distributed stage 로더를 모두 커버한다.

### 4.2 Offline CLI (`src/commands/generate.rs`)

새 함수 `reject_dflash_drafter_offline`이 `run_generation_mode`에서 `load_model(draft_model_path)` 바로 직전에 호출된다. `--draft-kind` 없이 호출한 경우와 `--draft-kind dflash`로 명시한 경우 모두 발화하며, `--draft-kind mtp`는 별도의 요청으로 이미 `run_offline_mtp`가 처리한다.

### 4.3 테스트 fixture 위생 수정 (`src/commands/generate_tests.rs`, 커밋 `37136aa6`)

```rust
// 변경 전
fn drafter_fixture_dir(name: &str, config: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "mlxcel_generate_drafter_{name}_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir_all(&dir).unwrap();
    fs::write(dir.join("config.json"), config).unwrap();
    dir
}
// 각 호출 지점이 성공 경로에서만 fs::remove_dir_all(dir).unwrap()을 수동 호출

// 변경 후
fn drafter_fixture_dir(config: &str) -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("temp dir");
    fs::write(dir.path().join("config.json"), config).unwrap();
    dir
}
// 호출 지점은 dir.path()를 사용. TempDir이 모든 종료 경로에서 drop 시 디렉터리를 제거
```

`name` 매개변수는 제거했다. 이 값은 경로 세그먼트를 어느 정도 고유하게 만드는 용도로만 쓰였는데, `tempfile::tempdir()`가 구조적으로 이미 이를 보장한다. `offline_draft_model_check_defers_missing_or_broken_configs_to_the_loader`의 `empty` fixture에 있던 동일한 인라인 패턴도 같은 방식으로 고쳤다.

---

## 7. 변경 요약

### 통계

| 항목 | 값 |
|-----|---|
| 변경된 파일 수 (기능 커밋 `82096193`) | 9 |
| 변경된 파일 수 (M3 수정 커밋 `37136aa6`) | 1 |
| 추가/삭제 라인 (기능) | +645 / -5 |
| 추가/삭제 라인 (M3 수정) | +19 / -34 |
| 테스트 추가 | `commands::generate` 4건, `models::detection` 3건, `drafter::dflash::config` 3건(기존 커버리지 확장 포함), `speculative_dispatch` 2건 |

### 카테고리별 변경

| 카테고리 | 변경 수 | 주요 내용 |
|---------|--------|----------|
| Detection / 로딩 | 3 | `is_dflash_drafter_config`, `is_dflash_drafter_dir`, `get_model_type` 거부, 공유 오류 메시지 |
| CLI | 1 | `run_generation_mode`의 `reject_dflash_drafter_offline` 사전 로드 가드 |
| 문서 | 2 | `docs/supported-models.md`, `docs/speculative-acceptance.md`에 offline 제약 명시 |
| 테스트 위생 (이번 finalization) | 1 | `drafter_fixture_dir`와 인라인 `empty` fixture를 `tempfile::tempdir()`로 전환 |

### 관련 커밋

| Hash | Type | Message |
|------|------|---------|
| `82096193` | fix | reject a DFlash drafter before the offline standalone load |
| `37136aa6` | fix | back drafter test fixtures with `tempfile::tempdir()` |

---

## 8. 후속 조치

### 완료 필요

- [ ] 없음. 리뷰와 보안 검토에서 CRITICAL이나 HIGH 항목은 보고되지 않았고, MEDIUM 항목(M3) 하나는 이번 finalization에서 수정했다.

### 향후 개선 사항 (알려진 제약으로 기록, 이번 PR에서는 수정하지 않음)

- 이 사전 검사는 target 모델 로드가 끝난 뒤에야 발화한다. target이 큰 경우 오퍼레이터는 전체 로드가 끝날 때까지 기다려야 오류를 보게 된다. 검사를 더 이른 시점으로 옮기면 별도로 라우팅되는 `--draft-kind mtp`까지 가로챌 수 있어 그대로 두었다.
- `--draft-model <dir>/config.json`처럼 디렉터리가 아니라 config 파일 자체를 직접 가리키면 이 사전 검사를 우회하고, 로딩 경로의 나머지 부분이 적용하는 `resolve_model_dir`를 이 검사가 적용하지 않기 때문에 `-m` 문구가 섞인 detection 오류로 떨어진다.
- 이 프로브는 `config.json`을 원시 `serde_json::from_slice`로 파싱하는 반면 `get_model_type`은 먼저 `sanitize_config_json`을 적용한다. 순수 `NaN`이나 `Infinity` 리터럴을 담은 config는 이 사전 검사를 그냥 통과해버릴 수 있다. 현재 어떤 체크포인트도 영향을 받지 않으며, sanitizer는 루트 crate에 있고 프로브는 `mlxcel-core`에 있어 이 간극을 메우려면 crate 경계를 넘는 변경이 필요하므로 남겨두었다.
- 실제 offline `DFlashGenerator` round loop 구현은 PR 요약에서 밝힌 대로 여전히 범위 밖이다. 오늘 이 기능이 필요한 오퍼레이터는 CLI 오류 메시지가 안내하는 대로 `mlxcel-server --draft-kind dflash`를 사용해야 한다.

---

## 부록

### A. 테스트 결과

- `cargo test --release --workspace --features metal,accelerate --no-fail-fast`: `-p mlxcel --lib`를 제외한 모든 타깃 통과. 이 타깃은 5621 passed / 3 failed를 보고했고 실패한 3건은 모두 `multimodal::video::tests::*`다. 이 PR과는 무관함이 확인됐다. `src/multimodal/video.rs`는 이 브랜치에서 단 한 줄도 건드리지 않았고, 같은 바이너리로 PATH를 제어한 실험에서 ffmpeg를 숨기면 34 passed / 0 failed, ffmpeg 9.0.1이 있으면 31 passed / 3 failed가 나온다. 근본 원인은 `src/multimodal/video.rs:1123`가 넘기는 `-vsync` 인자로, ffmpeg 9에서 제거됐다. 별도로 #1172로 등록됐다.
- 새로 추가된 `commands::generate::tests::offline_draft_model_*` 사전 검사 테스트 4건은 `--lib`이 아니라 `--bin mlxcel`로 컴파일되므로 위 `--lib` 실행에는 포함되지 않았다. 별도로 `cargo test --release --bin mlxcel --features metal,accelerate commands::generate`를 실행해 82건 모두 통과했고 그중 4건이 이 테스트다.
- `cargo test --release -p mlxcel-core --lib --features metal,accelerate drafter::dflash::config`: 16건 통과.
- `cargo test --release --test speculative_dispatch --features metal,accelerate`: 22건 통과.
- `cargo test --release --lib --features metal,accelerate models::detection`: 30건 통과.
- `cargo clippy --release --workspace --features metal,accelerate --tests -- -D warnings`: 클린.
- `cargo fmt --check`: 클린.
- classic 경로 non-regression을 실물 체크포인트로 실행: `-m qwen3-0.6b-4bit --draft-model qwen3-0.6b-4bit`가 `SpeculativeGenerator`를 통해 24토큰을 생성했고, acceptance_rate 1.0000, 137.54 tok/s를 기록했다.
- mutation 테스트: 새 detection 분기를 `if false && ...`로 막으면 `-m`과 `--draft-model` 거부 테스트가 두 마커 모두에서 수정 전 분류로 실패한다.

### B. 참고 자료

- 이슈 #1168 (사양)
- 이슈 #1172 (선재하던, 이 PR과 무관한 ffmpeg 9 `-vsync` 회귀. 전체 workspace 테스트 실행 중 드러남)
- `src/lib/mlxcel-core/src/drafter/dflash/config.rs` (구조적 프로브), `src/models/detection.rs` (`get_model_type` 거부), `src/commands/generate.rs` (`reject_dflash_drafter_offline`)
- PR #1170의 리뷰·보안 코멘트
