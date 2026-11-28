# 기술 보고서: PR #1381 - OpenXLA 경고 백로그 정리 후 CI에서 모든 경고 거부

**날짜**: 2026-08-22
**작성자**: Jeongkyu Shin
**상태**: 완료
**언어/기술**: Rust, YAML (GitHub Actions)
**위험도**: 낮음 (린트와 가시성 변경, 런타임 동작 변화 없음)

---

## 요약

`xla-compile` CI job은 모든 경고가 아니라 `unused_imports` 하나만 거부하고 있었다. 의도적인 선택이었다. OpenXLA 기능 조합에 dead-code와 clippy 백로그가 쌓여 있었고, 착륙하는 날부터 빨간 job은 우회당한 뒤 무시되기 때문이다.

이 PR은 백로그를 정리하고 게이트를 `RUSTFLAGS: "-D warnings"`로 확대한다. 두 크레이트에 걸친 14건을 처리했다. 하나는 삭제, 나머지는 cfg 게이트와 무엇이 그 항목을 살려두는지 이름을 적은 범위 한정 `#[allow]`, 그리고 기계적 수정 세 건이다. 14건 중 4건은 이슈 목록에 아예 없었다. 시작 전에 백로그를 다시 측정했기 때문에 찾았고, 이슈 목록은 작성 시점 이후 실제와 어긋나 있었다.

검증은 확대된 게이트가 확대를 수행하는 바로 그 PR에서 초록으로 도는 것으로 이루어졌다. `OpenXLA feature compile`이 새 플래그 아래 7분 44초짜리 실제 재빌드를 수행하고 통과했다.

## 1. 문제 정의

### 1.1 배경

PR #1282가 `RUSTFLAGS: "-D unused_imports"`로 `xla-compile`을 추가했다. 그 린트를 고른 이유는, OpenXLA 커버리지 공백으로 `main`에 들어간 두 결함 중 하나가 정확히 그 린트 부류였기 때문이다(죽은 `load_weights_from_dir_with_filter` 재수출). 나머지 하나는 컴파일 오류가 잡았다(새 `ModelRequest` 변형이 추가된 뒤의 비망라 `match`). 과거의 두 사고는 모두 잡힌다. 잡히지 않는 것은 이 기능들 아래 새로 생기는 dead code다.

### 1.2 기존 문제

백로그 정리는 게이트 확대의 선행 조건이고, 백로그는 `mlxcel-xla`와 `mlxcel` 양쪽에 걸쳐 있었다. 삭제는 대개 틀린 해법이다. 여러 항목이 죽었다고 보고되는 기능 조합이 아닌 다른 조합에서는 살아 있기 때문이다.

### 1.3 위험 평가

단독으로는 낮지만 누적된다. 게이트되지 않은 경고 하나하나가 같은 부류의 다음 진짜 결함이 조용히 도착하는 지점이고, #1282를 촉발한 두 결함이 이미 그 증거다.

## 2. 변경 요약

파일 열 개. Rust 여덟, 워크플로 하나, 그리고 보고서.

| 파일 | 처리 |
|---|---|
| `mlxcel-xla/src/aux.rs` | `Float16`, `Uint32`: `cfg_attr(not(micro-oracle), allow(dead_code))` |
| `mlxcel-xla/src/aux_manifest.rs` | collapsible `if`를 edition 2024 let chain으로 재작성 |
| `mlxcel-xla/src/iree.rs` | `decode_ragged_logits`를 `diagnostics`로 cfg 게이트, `decode_ragged_mrope_logits` 삭제, 떨어져 나간 doc 주석 원위치 |
| `mlxcel-xla/src/phi4_audio.rs` | 범위 한정 `allow(clippy::too_many_arguments)` |
| `src/loading/vlm.rs`, `src/models/sanitize.rs` | `cfg_attr(not(surgery), allow(clippy::needless_match))` |
| `src/server/batch/xla_audio_preprocess.rs` | `cfg_attr(not(test), allow(dead_code))` 일곱 건, 각각 호출자 명시 |
| `src/server/batch/xla_worker_admission.rs` | 사유를 적은 `allow(clippy::while_let_loop)` |
| `src/server/media.rs` | cfg_attr 조건을 `any(not(xla-iree), not(test))`로 확장 |
| `.github/workflows/ci.yml` | `RUSTFLAGS`를 `-D warnings`로, 정책 주석 교체, 커버리지 공백 두 건 기록 |

## 3. 기술적 선택과 그 이유

### 3.1 이슈 목록을 그대로 쓰지 않고 백로그를 다시 측정

이슈 목록은 작성 시점의 측정값이다. 현재 트리에서 두 clippy 명령을 다시 돌리니 이슈의 10건이 아니라 14건이 나왔고, 늘어난 4건은 부수적인 것이 아니었다. 둘은 단순 기계적 수정이었지만, 나머지 둘이 이 변경 전체에서 가장 위험한 `needless_match` 쌍이었다(3.3). 낡은 목록으로 작업했다면 자기 인수 조건을 통과하지 못하는 PR이 나왔을 것이다.

### 3.2 삭제는 한 건, 그것도 어디서도 죽었음을 증명한 뒤에

`decode_ragged_mrope_logits`는 `decode_ragged_mrope_logits_with_modes`를 감싼 얇은 래퍼이고 측정한 두 기능 조합 모두에서 죽어 있다. allow가 아니라 삭제한 이유는, 인수 조건이 모든 `#[allow]`에 그 항목을 살려두는 호출자를 명시하라고 요구하는데 그런 호출자가 없기 때문이다. 근거는 모든 파일 유형에 대한 저장소 전역 검색이며, 리뷰 과정에서 단순 grep이 놓치는 경로까지 확장했다. 매크로(`mlxcel-xla`에는 MLIR 픽스처용 `macro_rules!` 하나뿐이고 `paste!`나 `concat_idents!`는 없다), doc test, `pub use` 재수출, trait impl, 그리고 `micro-oracle`과 `xla-diagnostics-cpu` 아래의 기능 게이트된 호출자. 남은 것은 이름이 같은 C 함수 `xla_llama_decode_ragged_mrope_logits`와 그 extern 선언, 그리고 살아 있는 호출자를 유지하는 `_with_modes` 형제뿐이다.

쌍둥이인 `decode_ragged_logits`는 **삭제하지 않았다.** `cuda,xla-iree`에서는 죽어 있고 `xla-diagnostics`에서는 살아 있으므로, `batch.rs`의 네 호출자가 이미 달고 있는 cfg와 맞춰 `feature = "diagnostics"`로 게이트했다. `diagnostics = ["iree"]`이므로 새 게이트는 이를 감싸는 `#[cfg(feature = "iree")] mod iree`보다 엄격히 좁고 어떤 조합도 깨뜨릴 수 없다. 이것을 지웠다면 경고가 다른 기능 조합의 빌드 실패로 바뀌었을 것이고, 그것이 이 이슈가 명시적으로 경고한 실패다.

### 3.3 거부해야 했던 clippy 제안 두 건

`src/loading/vlm.rs`와 `src/models/sanitize.rs`에는 다음 형태가 있다.

```rust
let resolved_transform = match transform {
    Some(t) => Some(t),
    None => {
        #[cfg(feature = "surgery")]
        { active_pipeline.as_deref().map(...) }
        #[cfg(not(feature = "surgery"))]
        ...
    }
};
```

`--no-default-features`에서는 `surgery` 팔이 컴파일에서 빠지고 `None` 팔이 `None`으로 무너지며, clippy는 정확하게 이 match를 무의미하다고 보고하면서 "`transform`으로 바꿔라"라고 제안한다. 그 제안을 받아들이면 `default = ["surgery"]`인 기본 빌드에서 active pipeline 해석이 사라진다.

해법은 `let` 문에 붙인 `#[cfg_attr(not(feature = "surgery"), allow(clippy::needless_match))]`다. 무너진 형태를 보는 빌드에서만 정확히 침묵시키고, 기본 빌드에서는 계속 린트한다. 동작을 바꾸는 수정은 틀린 수정이라는 이슈의 규칙이 구체화된 사례다.

### 3.4 맨 allow가 아니라 `not(test)`

`xla_audio_preprocess.rs`와 `media.rs`의 모든 억제는 `not(test)` 조건부다. `cargo check --all-targets`는 `mlxcel` lib을 `cfg(test)` 포함해 두 번 컴파일하므로 억제가 스스로 만료된다. 유일한 테스트 호출자를 지우면 린트가 영원히 침묵하는 대신 다시 살아난다. 각 속성에는 호출자를 명시한 주석이 붙어 있고, 리뷰가 하나씩 확인했다.

기록해둘 세부 사항 하나. `#[allow]`는 린트를 억제할 뿐 심볼을 살아 있는 것으로 표시하지 않는다. 그래서 `is_healthy`를 allow한 뒤에도 `healthy` 필드가 따로 경고했다. 항목마다 속성이 필요했다.

### 3.5 `RUSTFLAGS`를 넓히되, 여전히 못 덮는 것을 기록

`cargo check` job에서 `RUSTFLAGS: "-D warnings"`는 rustc 린트만 거부한다. 이 PR이 고친 것의 대부분인 clippy 자체 린트는 이 기능 조합들 아래에서 여전히 아무것도 게이트하지 않는다. `clippy` job은 `mlxcel-xla`가 기본 꺼짐인 기본 기능으로 빌드하기 때문이다. job의 `cargo check`를 `cargo clippy`로 바꾸면 닫히지만, 의도적으로 범위 밖으로 두었다. 이 공백은 암시로 남기지 않고 워크플로의 제외 목록에 기록했다. 정리한 절반이 완전 커버리지를 주장하는 주석 뒤에서 다시 자라나지 않도록.

## 4. 리뷰에서 나온 지적

두 리뷰 모두 CRITICAL과 HIGH는 없었다. 후속 커밋이 정확성 결함 네 건을 고쳤다.

- **빌드에 없는 함수 이름을 담은 오류 메시지.** `iree.rs`의 arity 검사 메시지 둘이 실제로 메시지를 내보내는 `_with_modes`가 아니라 얇은 래퍼 이름을 담고 있었다. 래퍼 하나는 삭제됐고 다른 하나는 diagnostics 전용이 되었으므로, 프로덕션 `cuda,xla-iree` 빌드가 자신이 갖고 있지 않은 함수를 지칭하는 `"decode_ragged_mrope_logits expects ..."`를 반환할 수 있었다.
- **불완전한 호출자 목록.** `AudioPreprocessStage::spawn`의 주석이 테스트 파일 하나만 언급했는데 `xla_worker_tests.rs`도 호출한다. 인수 조건이 모든 allow에 살려두는 대상을 명시하라는 것이므로, 불완전한 목록은 그 조건이 막으려는 바로 그것이다.
- **덮는 범위의 절반만 설명한 주석.** `drain_preprocessed`에는 이미지와 오디오 두 개의 drain 루프가 있고 함수 수준 allow는 둘 다 덮는다.
- **기록되지 않았던 확대 게이트의 성질 두 가지**를 제외 목록에 추가(5.2 참조).

## 5. 검증

### 5.1 통과한 것

모두 IREE 런타임이 준비된 GB10 호스트에서, 최종 커밋 기준으로 재실행했다.

| 명령 | 결과 |
|---|---|
| `cargo clippy --features cuda,xla-iree --all-targets -- -D warnings` | exit 0, 무경고 |
| `cargo clippy --no-default-features --features xla-diagnostics --all-targets -- -D warnings` | exit 0, 무경고 |
| `cargo clippy -p mlxcel --lib --tests -- -D warnings` (기본 기능 회귀) | exit 0, 무경고 |
| `RUSTFLAGS="-D warnings" cargo check --features cuda,xla-iree --all-targets` | exit 0 |
| `RUSTFLAGS="-D warnings" cargo check --no-default-features --features xla-diagnostics --all-targets` | exit 0 |
| `cargo clippy -p mlxcel-xla --lib --features iree,micro-oracle --all-targets -- -D warnings` | exit 0 |
| 이 PR의 `OpenXLA feature compile` | 성공, 새 플래그 아래 7분 44초 실제 재빌드 |

`cargo check` 두 줄은 clippy 줄과 중복이 아니다. `cargo clippy -- -D warnings`는 플래그를 primary 패키지에만 적용하지만 `RUSTFLAGS`는 path로 빌드되는 모든 유닛에 도달한다. 따라서 `mlxcel-core`나 `mlxcel-surgery`의 경고는 어떤 clippy 실행에도 나타나지 않은 채 job을 빨갛게 만들 수 있다. 인수 조건만 검증했다면 job이 초록이 되는지는 알 수 없었다.

### 5.2 게이트가 정확히 무엇을 덮는가

`RUSTFLAGS`는 `mlxcel-core`, `mlxcel-surgery`, `mlxcel-xla`와 빌드 스크립트에 도달한다. 그래서 `mlxcel-core`의 CUDA 전용 경고가 OpenXLA 이름을 단 job을 빨갛게 만든다. 레지스트리와 git 의존성은 그럴 수 없다. cargo가 `--cap-lints allow`로 컴파일하기 때문이다. `rust-toolchain.toml`은 `stable`을 따라가지 않고 `1.97.1`을 정확히 고정하므로, 새 rustc 릴리스가 혼자서 job을 빨갛게 만들 수 없다. 비용은 의도적으로 핀 상향 시점으로 옮겨진다.

`default-members`가 없으므로 여기서 `cargo check`는 `-p mlxcel`로 해석되고 `--all-targets`는 그 패키지 안에서만 전개된다. 따라서 `mlxcel-xla`의 `#[cfg(test)]` 모듈 23개는 어떤 job도 게이트하지 않는다. 이 PR의 테스트 계획은 `-p mlxcel-xla --all-targets`를 수동으로 돌렸지만 CI는 돌리지 않는다.

### 5.3 검증하지 못한 것

`metal,accelerate`는 이 Linux 호스트에서 빌드할 수 없어 실행으로 검증하지 못했다. 영향이 없다는 정적 논거는 이렇다. `make verify-clippy`는 `--no-default-features` 없이 `--features metal,accelerate`로 돌리므로 `surgery`가 켜진 채이고 `needless_match` 속성 둘은 아무것도 확장하지 않는다. 거기서 `mlxcel-xla`는 기능 없이 빌드되므로 이 크레이트에서 바뀐 모든 파일이 `#[cfg(feature = "iree")]` 뒤에 있어 컴파일되지 않는다. `mlxcel` 쪽에서는 `media.rs`와 `xla_audio_preprocess.rs`만 컴파일되는데, 두 변경 모두 빌드를 깨뜨릴 수 없는 `allow` 속성이다.

## 6. 학습 포인트

1. **작성 시점에 측정된 백로그는 명세가 아니라 스냅숏이다.** 그것으로 작업하기 전에 다시 측정할 것. 여기서는 40퍼센트가 어긋나 있었고, 어긋난 항목 중 하나가 동작 회귀를 일으켰을 함정이었다.
2. **인수 조건을 검증하는 것과 산출물을 검증하는 것은 다르다.** 조건은 clippy를 지목했지만 job은 `RUSTFLAGS`와 함께 `cargo check`를 돌리고, 영향 범위가 더 넓다. 산출물이 실제로 실행할 것을 실행할 것.
3. **`#[allow]`는 린트를 억제할 뿐 심볼을 살아 있게 표시하지 않는다.** 접근자를 allow해도 그것이 읽는 필드는 따로 경고한다.
4. **억제에는 만료 조건을 걸 것.** `not(test)`는 테스트 빌드에서 린트를 살려두므로, 근거가 사라진 억제는 근거보다 오래 살아남는 대신 다시 빨개진다.
5. **clippy 제안은 clippy가 실행된 그 구성에 대한 제안이다.** `cfg`로 갈라진 코드에서는, 제안을 적용하면 clippy가 본 적 없는 구성에만 존재하는 동작을 지울 수 있다.

## 7. 후속 작업

- **`mlxcel-xla`의 테스트 모듈은 어떤 CI job도 게이트하지 않는다.** 워크플로 제외 목록에 기록했다. 닫으려면 패키지를 명시적으로 선택하거나 `default-members`를 추가해야 한다.
- **이 기능 조합들 아래 clippy 전용 린트는 여전히 게이트되지 않으므로** 백로그의 그 절반은 다시 자랄 수 있다. 닫으려면 이 job에서 clippy를 돌려야 한다.
- **`media_tests.rs`는 `validate_xla_raw_counts_with_audio`에 `supports_audio = false` 래퍼로만 도달하므로** `true` 분기에 직접적인 단위 테스트가 없고, 그 플래그의 부호가 보안상 의미 있는 비트다.
- **`iree.rs:2296-2305`의 기존 doc 주석 이탈.** 이 PR이 `:1819`에서 고친 것과 같은 결함으로, intra-doc 링크 두 개가 깨져 있다. 이 브랜치 이전부터 있었고 빈 줄이 없어 경고를 내지 않는다.

## 참고

- 이슈 #1304(이 작업), #1282(해당 job과 기록된 제외 목록), #1303 및 PR #1305(같은 제외 목록의 링크 절반)
- `.github/workflows/ci.yml`, `src/lib/mlxcel-xla/src/iree.rs`, `src/server/batch/xla_audio_preprocess.rs`
