# 기술 보고서: PR #1621 - feat(bench): dispatch speculative_bench MTP past Gemma 4 Unified

**작성일**: 2026-09-04
**작성자**: mlxcel maintainers
**리뷰어**: implementation review cycle
**상태**: 완료 (기존에 실패하던 두 페어링을 M1 Ultra에서 측정. M5 Max 재측정은 해당 호스트에서 대기)
**언어**: Rust
**위험도**: Low (벤치마크 하네스 전용. 추론 경로 변경 없음, 진단용 헬퍼 하나만 `pub`으로 확대)

---

## 요약

`run_mtp`은 `LoadedModel::Gemma4Unified`만 매칭하고 나머지 변형은 전부 `std::mem::discriminant`로 bail했다. 그래서 `speculative_bench`가 MTP 수치를 낼 수 있는 체크포인트가 딱 하나였다. 필요한 어댑터는 이미 다 있었고 서버 burst 경로는 이미 그것들을 디스패치하고 있었으니, 지원이 없는 척한 곳은 하네스뿐이었다.

이제 변형 매칭이 `src/server/batch/speculative_burst.rs`를 그대로 따라가고, `REACHABLE_PAIRINGS`에 Qwen 3.8 페어링이 들어갔다. 전에 실패하던 두 페어링이 모두 수치를 낸다.

작업 중에 발견한 별도 결함이 정작 발행된 것보다 더 문제였다. `resolve_model_dir`은 체크포인트 저장소가 `models/mlx/<name>`인 호스트에서 아무 모델도 찾지 못했고, 그런 호스트의 전체 스윕은 아무것도 측정하지 않으면서 오류도 내지 않았다.

---

## 1. 문제 정의

### 1.1 배경

`speculative_bench --sweep`은 `REACHABLE_PAIRINGS`를 훑고 MTP 행이면 `run_mtp`을 부른다. 그 함수는 변형 하나만 받는 매칭으로 시작했다.

```rust
let unified = match &model {
    LoadedModel::Gemma4Unified(u) => u,
    other => anyhow::bail!(
        "MTP bench currently supports a Gemma 4 Unified target; \
         load_model returned a different variant ({:?})",
        std::mem::discriminant(other)
    ),
};
```

### 1.2 그 대가

페어링 둘이다. Gemma 4 31B MTP 행은 K 세 값에서 모두 실패했는데 체크포인트가 없어서가 아니다. `models/gemma-4-31b-it-4bit`는 `Gemma4ForConditionalGeneration`이라 `LoadedModel::Gemma4VLM`로 로드되고, 12B가 `Gemma4UnifiedForConditionalGeneration`이다. Qwen 3.8 MTP는 아예 측정할 수 없었고, 카탈로그 항목만 추가해서는 소용이 없었다. 같은 매칭이 타깃을 거절하기 때문이다.

필요한 조각은 전부 이미 있었다. `speculative_burst.rs`의 변형별 `&dyn LanguageModel` 선택과 변형별 `MtpTarget` 어댑터 선택, 31B 타깃이 필요로 하는 `Gemma4VLMtpTargetAdapter`, Qwen 3.8 타깃이 필요로 하는 `Qwen35MtpTargetAdapter`, `qwen3_5_mtp` 드래프터 해석, 그리고 `release_sequence_state_by_id`를 포함한 enum 자체의 `LanguageModel` 구현까지.

### 1.3 이슈가 말하지 않은 결함

`resolve_model_dir`은 `<CARGO_MANIFEST_DIR>/models/<name>`을 보고 그다음 `../mlxcel-internal/models/<name>`을 봤다. 체크포인트 저장소가 `models/mlx/<name>`이고 형제 체크아웃도 같은 배치인 호스트에서는 두 탐색 모두 빗나간다. 그러면 스윕의 모든 행이 "디스크에 체크포인트 없음" 스킵으로 흘러가고, 스윕은 완주하면서 아무것도 측정하지 않고 오류도 전혀 내지 않는다. 발행된 결함보다 나쁜 실패 양상이다. 조용하기 때문이다.

---

## 2. 기술적 결정

### 2.1 변형마다가 아니라 제네릭 타이밍 함수 하나

`MtpGenerator`가 `MtpTarget`에 대해 제네릭이므로, 워밍업 burst와 시간 측정 generate, 수용률 수집, 시퀀스 해제를 `T: MtpTarget`에 제네릭한 `run_mtp_timed` 하나로 옮기고 각 arm이 자기 어댑터 생성자로 호출하게 했다. 변형마다 다른 건 `MtpTarget` 생성뿐이다. 드래프터는 enum 자신의 `LanguageModel` 구현에 바인드하고 같은 참조로 해제하므로, 생명주기 코드를 arm마다 복사하지 않는다.

### 2.2 공유 라벨은 `pub(crate)`이 아니라 `pub`

원래 계획은 `model_variant_label`을 `pub(crate)`으로 넓히는 것이었다. 그건 통하지 않는다. `speculative_bench`는 `mlxcel` 라이브러리와 별개 크레이트이고 `pub(crate)`은 라이브러리 경계에서 멈춘다. 그래서 함수를 `pub`으로 두고 크레이트 루트에 재수출했으며, 두 자리 모두에 이유를 적었다. 대안은 벤치 쪽에 라벨 표를 하나 더 두는 것이었는데, 그러면 서버 쪽과 어긋나기 시작한다.

노출된 표면은 진단용 이름을 돌려주는 `fn(&LoadedModel) -> &'static str` 하나라서 추가분이 작고 호환을 깨지 않는다. 공유 표는 speculative 가능 집합 밖의 변형을 `other`로 보고하므로, 벤치의 미지원 메시지는 라벨과 함께 타깃 디렉터리도 같이 적는다. 그 표 자체를 넓히는 일은 범위 밖으로 뒀다.

### 2.3 미지원 변형 검사는 드래프터 IO보다 먼저

처음 작성했을 때는 매칭이 `validate_target_compat` 뒤에 있었다. 그러면 hidden size가 안 맞는 미지원 타깃이 *드래프터* 메시지로 실패하고, "로드된 변형과 지원 계열을 알려 준다"는 합격 기준의 메시지는 정작 그 메시지가 설명하려던 입력에서 도달 불가능해진다. 이제 검사가 드래프터 로드보다 먼저 돈다. burst 경로와 같은 순서다.

### 2.4 모든 실패는 행 상태로 남는다

드래프터가 로드는 되지만 `validate_target_compat`에 걸리는 경우, 체크포인트 부재, `block_size < 2`, Qwen 3.5 MTP 정확성의 Metal 전용 가드까지 전부 스윕을 중단시키지 않고 행별 상태 문자열로 기록된다. CUDA 스윕은 런타임이 보증할 수 없는 숫자를 내놓는 대신 그 이유를 적는다.

---

## 3. 검증

Apple M1 Ultra 128 GB, Metal, mlxcel 0.7.0-beta.1, MLX 핀 `9a795735`, `--kind mtp --block-size 4 --max-tokens 128`. 둘 다 변경 전 바이너리에서는 실패한다.

```
gemma-4-31b-it-4bit + gemma-4-31b-it-assistant-bf16
  tok/s=14.8  rounds=34  acceptance_rate=0.529  mean_accepted_len=1.59

qwen3.8-27b-4bit + qwen3.8-27b-mtp-4bit
  tok/s=16.9  rounds=47  acceptance_rate=0.504  mean_accepted_len=1.51
```

전체 `--sweep --batch 1 --max-tokens 128`은 16행을 완주한다. 베이스라인 4개, K=2/4/8에서 측정된 MTP 행 9개, 자체 블로커로 여전히 보류인 DFlash 행 3개이고, `MTP bench currently supports`로 시작하는 상태를 단 한 행도 달고 있지 않다. `gemma-3-4b-it-4bit`를 `--kind mtp`로 넣으면 로드된 변형과 타깃 디렉터리와 지원 계열을 행 상태로 보고하고 중단하지 않는다.

### 3.1 결과는 음성이고, 원인은 호스트다

| 페어링 | 베이스라인 | K=2 | K=4 | K=8 |
|---|---|---|---|---|
| Gemma 4 31B + MTP assistant | 19.9 | 18.4 | 14.9 | 14.9 |
| Gemma 4 Unified 12B + MTP assistant | 38.4 | 36.0 | 28.5 | 29.3 |
| Qwen 3.8 27B + MTP head | 24.9 | 22.5 | 17.2 | 9.3 |

모든 비율이 1.00x 아래다. 같은 Unified 12B 페어링이 M5 Max에서는 K=4에 1.57x를 낸다. 수용률이 설명은 아니다. 같은 페어링이 같은 K에서 M1 Ultra 쪽이 오히려 더 많이 수용하는데도(39.6%에 평균 수용 길이 1.19, M5 Max는 35.0%에 1.05) 0.74x에 머문다.

차이는 verify 라운드다. `docs/benchmark_results/speculative-decoding-m1ultra-2026-08-19.md`는 block-4 verify 라운드 비용을 M1 Ultra 2.70 classic decode step, M3 Ultra 1.50, M5 Max 1.27로 측정했고, M1 Ultra를 Apple GPU 13세대의 첫 호스트로 지목한다. 평균 수용 길이가 1.2 언저리면 2.70 step짜리 verify를 상대로 손익분기를 넘을 수 없다. 이 세대에서 B=1 MTP를 거절하는 정적 게이트도 같은 이유로 존재한다. 즉 이 행들은 회귀 보고가 아니라 호스트의 알려진 성질을 재확인한 것이다.

### 3.2 게이트

`cargo test --workspace --profile test-fast --features metal,accelerate`: 10,507 통과, 0 실패. `cargo clippy --bin speculative_bench --features metal,accelerate -- -D warnings`와 `cargo fmt --all -- --check` 깨끗. CI 13개 항목 전부 통과.

최종 실행 직전의 전체 워크스페이스 실행 하나가 `vision::inkling_vl::tests::mixed_prefill_scatter_order_is_normalized_text_then_image_then_audio`에서 실패했다. 이 PR이 건드리지 않는 파일이다. 단독 실행 3회 모두 통과하고 `--lib` 바이너리 전체도 같은 테스트 수로 두 번 통과하므로 간헐적이고 무관한 실패다. 여기에 흡수하지 않고 따로 발행했다.

---

## 4. 변경 요약

### 통계

| 항목 | 값 |
|---|---|
| 변경 파일 (코드 PR) | 4 |
| 추가 줄 | 268 |
| 삭제 줄 | 65 |

### 분류별 변경

- `src/bin/speculative_bench.rs`: 변형별 어댑터 디스패치, 제네릭 `run_mtp_timed`, Qwen 3.8 페어링, 앞으로 끌어올린 미지원 변형 검사, `resolve_model_dir`의 `models/mlx/<name>` 탐색.
- `src/server/batch/speculative_burst.rs`: `model_variant_label` 가시성만.
- `src/lib.rs`: 크레이트 루트 재수출.
- `docs/benchmarks.md`: 지원 타깃 집합.

### 따로 반영한 부분

`benchmarks/metal_m1ultra_spec_2026-09-04.csv`(16행)와 `docs/benchmark_results/model_tests.md` 갱신은 `bench/0.7.0-refresh`(PR #1617)에 커밋 `f4ff37522`로 올렸다. 벤치마크 산출물은 코드 PR이 아니라 그 브랜치에 속하기 때문이다. 이 분리 때문에 이 PR이 머지되어 이슈가 닫혀도 문서 절반은 아직 미머지 상태로 남는다. PR 본문에 그 사실을 명시했다.

### 관련 이슈

Closes #1613. 관련: #154(Gemma 4 Unified MTP 드래프터), #1165(`qwen3_5_mtp` 드래프터와 Metal 정확성 게이트), #638(`--k-values` 뒤의 K 스윕).

---

## 5. 후속 작업

- 새로 측정 가능해진 두 페어링의 M5 Max 재측정은 해당 호스트에서 대기 중이다. `model_tests.md`의 M5 Max 표는 M1 Ultra 수치로 덮어쓰지 않고 그대로 두되, "unsupported target" 주석만 하네스 제한이 풀렸다는 문장으로 교체했다.
- `--kind dflash` 세 행은 기존 보류(DFlash 로더와 공개 `Qwen3NextCache` API)를 유지한다.
- 배치(`B > 1`) MTP 어댑터와 Inkling 페어링은 범위 밖이다. 후자는 로컬 체크포인트가 없다.

### 옮겨갈 만한 교훈

발행된 이슈는 측정을 거부하는 하네스를 설명했다. 정작 고치는 길에 나온 더 비싼 결함은 아무것도 측정하지 않으면서 그 사실을 어디에도 말하지 않는 하네스였다. 이 호스트의 배치에서 `resolve_model_dir`이 모든 페어링을 조용히 스킵으로 해석했다. 행이 0개인 벤치마크는 눈에 띈다. 모든 행이 스킵된 채 완주한 벤치마크는 깨끗한 실행처럼 보인다. 스윕이 경로 탐색에 의존한다면, 탐색 실패도 측정 실패만큼 시끄러워야 한다.
