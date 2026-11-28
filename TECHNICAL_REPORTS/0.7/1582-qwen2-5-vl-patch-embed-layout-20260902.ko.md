# 기술 보고서: PR #1582 - fix(qwen2_5_vl): normalize the raw HF patch-embed layout at load

**작성일**: 2026-09-02
**작성자**: mlxcel maintainers
**리뷰어**: implementation review cycle
**상태**: 완료 (단위 수준은 로컬에서 검증했고, 실제 바이너리 게이트는 PR에 미체크 항목으로 남겨 둠)
**언어**: Rust
**위험도**: Medium (Qwen-VL 계열 세 소비자의 로드 경로를 건드리지만, 변환 체크포인트 경로는 no-op임이 증명됨)

---

## 요약

HuggingFace 원본 Qwen2.5-VL 체크포인트는 `mlxcel generate`로도 `mlxcel-server`로도 서빙할 수 없었다. 비전 타워가 `vision_tower.*`가 아니라 `visual.*`에 저장돼 있어서 로더가 `Missing vision_tower.patch_embed.proj.weight`로 실패했고, 이름을 맞춰 준다 해도 패치 임베딩 필터가 `Conv3d`의 원래 레이아웃인 `[out, in, kT, kH, kW]`인 반면 인코더는 mlx-vlm 변환본의 채널 마지막 배치 `[out, kT, kH, kW, in]`만 받는다. 두 텐서의 원소 수가 같으므로 이후 reshape는 그냥 성공하고, 타워는 아무 오류 없이 뒤섞인 필터를 읽는다. PR #1414는 레이아웃 쪽만, 그것도 ColQwen2.5 검색 경로에서만 고쳤다. 이번 PR은 그 정규화 함수를 자신이 지키는 계약이 있는 인코더 모듈로 옮기고, 가중치 prefix를 인자로 받게 하고, 두 소비자 모두에서 호출하며, 생성 로더에 빠져 있던 키 리맵을 추가한다.

---

## 1. 문제 정의

### 1.1 배경

`Qwen25VLVisionEncoder::from_weights`는 패치 임베딩을 `PatchEmbed::from_weights`(`src/vision/encoders/qwen2_5_vl.rs`)로 만든다. 이 함수는 5차원 가중치를 받아 `[0, 1, 4, 2, 3]`으로 치환한 뒤 `[out, in * kT * kH * kW]`로 reshape한다. 이 치환은 mlx-vlm 변환본의 채널 마지막 레이아웃에서만 옳다. `transformers`는 PyTorch `Conv3d` 파라미터를 그대로 쓰므로 `in_channels`가 축 1에 온다.

PR #1414는 ColQwen2.5를 포팅하다가 이 문제를 찾았다. 공개 체크포인트 `vidore/colqwen2.5-base`가 원본 export이기 때문이다. 그때 `src/models/colqwen2_5.rs`에 `normalize_patch_embed_layout`을 추가했지만, 해당 모듈의 `VISION_PREFIX`를 함수 안에 박아 넣은 형태였다. 고치지 않았을 때의 실측 대가는 검색 순위 역전이었다. 무관한 페이지가 MaxSim 7.83, 정답 페이지가 8.04였다.

생성 로더 `load_qwen2_5_vl`(`src/loading/vlm_qwen.rs`)에는 그 수정의 어느 쪽도 들어오지 않았다. 선행 `language_model.`만 떼는 `strip_language_model_prefix`를 돌린 뒤, 맵을 그대로 `vision_tower` prefix로 인코더에 넘겼다.

### 1.2 기존 문제점

- **원본 export는 아예 로드되지 않았다.** 이 호스트의 `Qwen/Qwen2.5-VL-3B-Instruct` 스냅샷은 최상위 prefix가 정확히 `model`과 `visual` 둘뿐이다. `vision_tower.*`라는 이름 자체가 없으니, 레이아웃 문제를 따지기도 전에 `PatchEmbed::from_weights`가 `Missing vision_tower.patch_embed.proj.weight`로 죽었다.
- **이름만 고쳤다면 조용히 틀린 답이 나왔을 것이다.** 손으로 이름을 바꾼 export나 리맵만 단독으로 들어간 경우, 키 없음 오류는 통과하고 그다음에 `[1280, 3, 2, 14, 14]`를 `[1280, 2, 14, 14, 3]`인 양 읽는다. 둘 다 원소가 1505280개라 reshape는 성공한다. 예외도 없고, 모델은 여전히 매끄러운 문장을 뱉는다.
- **고칠 코드는 이미 있었지만 호출자 하나에 묶여 있었다.** `normalize_patch_embed_layout`은 prefix를 상수로 박은 채 ColQwen2.5 모듈 안에 `pub(crate)`로 있었고, 생성 로더가 재사용하려면 판별 규칙을 복제하는 수밖에 없었다.
- **명명 규칙의 선례는 같은 파일 안에 이미 있었다.** `load_qwen3_vl`은 자기 계열에 대해 정확히 이 부류의 리맵을 한다. 즉 예외였던 쪽은 Qwen2.5-VL 로더다.

### 1.3 위험성

| 위험 | 영향 | 발생 가능성 |
|------|------|------------|
| HuggingFace 원본 Qwen2.5-VL 체크포인트를 생성/서빙에 쓸 수 없음 | High | 이번 변경 전에는 확정 |
| 이름만 맞춘 원본 체크포인트가 뒤섞인 비전 필터로 오류 없이 생성 | High | High, 출력이 매끄러워 탐지 불가 |
| 두 번째 호출자가 생기는 순간 판별 규칙 사본이 서로 어긋남 | Medium | Medium |

---

## 2. 기술적 검토 사항

### 2.1 근본 원인

같은 로드 경로에 서로 독립적인 구멍이 둘 있었고, 둘 다 모델 출력만 봐서는 보이지 않는다.

첫째는 이름이다. `strip_language_model_prefix`는 mlx-vlm 형태(`language_model.model.*` → `model.*`)만 처리하고 나머지는 손대지 않는다. `transformers` export는 `visual.*` + 맨 `model.*` 디코더(구형)이거나 `model.visual.*` + `model.language_model.*`(신형)인데, 어느 쪽도 그 한 번의 strip으로는 닿지 않는다.

둘째는 레이아웃이다. 두 5차원 shape를 가르는 것은 채널 축이다. 원본 레이아웃은 `in_channels`가 축 1에, 공간 크기(`patch_size`)가 축 4에 있고, 변환본은 `in_channels`가 축 4에 있다. 공개된 Qwen2.5-VL은 모두 `in_channels`가 3, `patch_size`가 14라 실무적으로는 애매함이 없지만, 이것은 어디까지나 shape 휴리스틱이므로 코드가 그렇게 말해야 한다.

### 2.2 호환성/의존성 관점

- **호환성 파괴**: 없음. 새로 들어간 두 단계 모두 mlx-community 변환본에서는 no-op이다. 키가 이미 `vision_tower.*` / `language_model.*`이고 필터도 이미 `[1280, 2, 14, 14, 3]`이다.
- **새 의존성**: 없음.
- **인코더 계약**: 그대로다. `PatchEmbed::from_weights`는 여전히 5차원 레이아웃 하나만 받는다. Qwen2-VL과 ColQwen2.5가 같은 코드를 쓰면서 그 전제에 기대고 있으므로 이 점이 중요하다.
- **양자화 체크포인트**: 4비트 변환본에서도 `patch_embed.proj.weight`는 float로 남으므로, 같은 판별과 치환이 그대로 적용된다.

### 2.3 코드 품질 관점

- 정규화 함수가 자신이 강제하는 불변식이 있는 코드 옆으로 옮겨졌고, 저장소 규약대로 두 호출자를 적은 `Used by:` 줄을 달았다.
- 판별 규칙과 그 테스트가 두 곳이 아니라 한 곳에 있다.
- 테스트는 ColQwen2.5 모듈의 1개에서 인코더 모듈 4개 + 로더 리맵 2개 + 실제 체크포인트 게이트 1개로 늘었다.
- `src/loading/vlm_qwen.rs`에는 인라인 `mod`를 또 추가하는 대신 `#[path]` 테스트 파일(`vlm_qwen_tests.rs`)을 붙였다. 이미 긴 파일을 더 늘리지 않기 위해서다.

---

## 3. 기술적 선택과 그 이유

### 3.1 레이아웃 정규화를 어디에 둘 것인가

**맥락:** `PatchEmbed::from_weights`가 스스로 레이아웃을 판별해 둘 다 받아 주면, 로더를 건드리지 않고 모든 소비자를 한 번에 고칠 수 있다.

**검토한 대안:**

| 선택지 | 장점 | 단점 |
|--------|------|------|
| `PatchEmbed::from_weights` 안에서 판별 | 한 곳에서 Qwen2-VL과 ColQwen2.5까지 자동으로 해결 | 세 소비자가 모두 하나의 레이아웃을 전제하는 모듈이 두 레이아웃을 조용히 받아들이게 됨. 진짜로 잘못된 필터가 들어와도 거부가 아니라 reshape가 됨 |
| ColQwen2.5의 헬퍼를 로더에 복제 | diff가 가장 작음 | 완전히 같아야 하는 shape 휴리스틱 사본이 둘 |
| **채택: 헬퍼를 인코더 모듈로 옮기고 prefix를 인자로 받아 두 로더에서 호출** | 규칙이 자신이 지키는 계약 옆에 있고, 테스트도 한 번이며, 인코더가 받는 레이아웃은 문서화된 하나로 유지됨 | 새 소비자가 생길 때마다 호출을 잊지 말아야 함 |

**근거:** `in_channels`를 손에 쥔 계층이 로더고, 어떤 종류의 체크포인트를 열었는지 아는 것도 로더뿐이다. 인코더를 엄격하게 두면 진짜로 망가진 필터는 거기서 크게 실패한다.

**트레이드오프:** 앞으로 Qwen2.5-VL 계열 로더가 호출을 빠뜨리면 수정 전의 조용한 손상 동작이 그대로 돌아온다. `Used by:` 줄과 공유 위치가 그 완화책이고, `PatchEmbed` 안의 강제 가드는 이 결정이 거부한 모호함과 맞바꾸는 선택이 된다.

### 3.2 두 레이아웃을 구별할 수 없을 때 무엇을 할 것인가

**맥락:** 판별은 채널 축으로 한다. 만약 `in_channels`가 `patch_size`와 같다면 축 1과 축 4가 모두 `in_channels`를 담게 되고, shape만으로는 구별할 수 없다.

기존 조건(`shape.len() != 5 || shape[1] != channels || shape[4] == channels`)도 그 경우 "건드리지 않음"으로 떨어지긴 했지만, 그건 순서에 따른 부수 효과였다. 새로 쓴 형태는 `leading_is_channel`과 `trailing_is_channel`에 이름을 붙이고, 추측하지 않겠다는 결정을 명시한다. 동작은 동일하고, 의도가 읽힌다는 점이 다르다. 다음 사람이 이 fall-through를 "버그"로 보고 고치는 일을 막는 것이 요점이다.

---

## 4. 구현 상세

### 4.1 주요 코드 변경

**파일: `src/vision/encoders/qwen2_5_vl.rs`**

```rust
pub(crate) fn normalize_patch_embed_layout(
    weights: &mut WeightMap,
    prefix: &str,
    in_channels: usize,
) -> bool {
    let key = format!("{prefix}.patch_embed.proj.weight");
    let channels = in_channels as i32;
    let converted = {
        let Some(weight) = weights.get(&key) else {
            return false;
        };
        let shape = mlxcel_core::array_shape(weight);
        if shape.len() != 5 {
            return false;
        }
        let leading_is_channel = shape[1] == channels;
        let trailing_is_channel = shape[4] == channels;
        if !leading_is_channel || trailing_is_channel {
            return false;
        }
        // [out, in, kT, kH, kW] -> [out, kT, kH, kW, in].
        mlxcel_core::transpose_axes(weight, &[0, 2, 3, 4, 1])
    };
    weights.insert(key, converted);
    true
}
```

**파일: `src/loading/vlm_qwen.rs`**

```rust
fn rewrite_qwen2_5_vl_native_key(key: &str) -> String {
    if let Some(rest) = key.strip_prefix("model.visual.") {
        format!("vision_tower.{rest}")
    } else if let Some(rest) = key.strip_prefix("model.language_model.") {
        format!("model.{rest}")
    } else if let Some(rest) = key.strip_prefix("visual.") {
        format!("vision_tower.{rest}")
    } else {
        key.to_string()
    }
}
```

그리고 `load_qwen2_5_vl` 안:

```rust
let mut weights = remap_qwen2_5_vl_native_keys(strip_language_model_prefix(
    load_vlm_weights_common(model_path, None)?,
));
models::sanitize_tied_embeddings(&mut weights, &full_config);

if normalize_patch_embed_layout(&mut weights, "vision_tower", vision_config.in_channels) {
    tracing::debug!(
        "Qwen2.5-VL: converted patch_embed.proj.weight from the PyTorch Conv3d layout"
    );
}
```

**파일: `src/models/colqwen2_5.rs`**

로컬 사본을 지우고, 호출을 `normalize_patch_embed_layout(&mut weights, VISION_PREFIX, vision_config.in_channels)`로 바꿨다.

### 4.2 순서

리맵은 `strip_language_model_prefix` 앞이 아니라 뒤에서 돈다. mlx-vlm 형태에서 이 순서가 중요하다. `language_model.model.layers.*`가 먼저 `model.layers.*`가 돼야 하고, 그렇지 않으면 `model.language_model.` 규칙이 그 키를 볼 일이 없다. `language_model.visual.*`를 담은 체크포인트(알려진 것은 없다)도 같은 방식으로 처리된다. 정규화는 둘 다 끝난 뒤에 돌기 때문에, 체크포인트가 어느 형태였든 최종 `vision_tower.` prefix를 본다.

---

## 5. 검증

치환은 자기 자신이 아니라 실제 변환기 결과와 맞춰 확인했다. 로컬 `Qwen/Qwen2.5-VL-3B-Instruct` 스냅샷의 bf16 `visual.patch_embed.proj.weight`(`[1280, 3, 2, 14, 14]`)를 축 `[0, 2, 3, 4, 1]`로 치환한 뒤, 무작위 20000개 원소를 f16 해상도에서 `mlx-community/Qwen2.5-VL-3B-Instruct-4bit`의 `vision_tower.patch_embed.proj.weight`(`[1280, 2, 14, 14, 3]`, f16)와 비교하면 20000/20000이 정확히 일치하고 최대 절대 오차는 0.0이다. 치환 없이 같은 버퍼를 읽으면, 즉 수정 전 로더가 사실상 하던 일을 하면 최대 1.09e-01까지 어긋난다. 이슈가 주장하던 수치를 가정이 아니라 측정으로 확인한 것이다.

이 브랜치에서 실행한 명령:

```
cargo fmt --all -- --check
cargo clippy --profile test-fast --lib --tests --features metal,accelerate -- -D warnings
cargo test --profile test-fast --features metal,accelerate --lib vision::encoders::qwen2_5_vl   # 4 passed
cargo test --profile test-fast --features metal,accelerate --lib loading::vlm::qwen             # 6 passed, 1 ignored
cargo test --profile test-fast --features metal,accelerate --lib models::colqwen2_5             # 6 passed
```

`#[ignore]`가 붙은 게이트 `qwen2_5_vl_raw_export_matches_mlx_conversion`은 평소에는 `Qwen/Qwen2.5-VL-3B-Instruct`와 `mlx-community/Qwen2.5-VL-3B-Instruct-bf16`을 표준 저장소 탐색으로 찾고, 어느 한쪽이 없으면 실패가 아니라 skip한다. 이 호스트에는 bf16 변환본이 없어서, 대신 `MLXCEL_TEST_QWEN25VL_RAW_DIR` / `MLXCEL_TEST_QWEN25VL_MLX_DIR` 오버라이드로 로컬에 있는 원본 3B export 두 개를 가리켜 돌렸다. 둘 다 생성 경로로 로드됐고 7.6초 만에 동일한 비어 있지 않은 greedy 출력을 냈다. 원본과 변환본 사이의 차분을 검증한 것은 아니지만, 수정 전 로더가 아예 못 하던 일, 즉 `transformers` 원본 export를 열어 거기서 생성하는 것은 증명한다.

### 여기서 검증하지 않은 것

세 가지는 실제 바이너리의 몫이고 PR 본문에 미체크 항목으로 남겼다. `mlxcel generate -m <원본 디렉터리> --image ...`가 이미지에 맞는 설명을 내놓는지, 4비트 변환본 출력이 변경 전후로 바이트 단위 동일한지, 그리고 체크포인트가 있는 상태에서 ColQwen2.5 실제 체크포인트 게이트가 통과하는지다. `mlx-community/Qwen2.5-VL-3B-Instruct-bf16`을 상대로 한 차분 게이트도 그 체크포인트가 없어 실행하지 못했다.

---

## 6. 변경 요약

### 통계

| 항목 | 값 |
|------|-----|
| 변경 파일 | 6 |
| 추가 라인 | 520 |
| 삭제 라인 | 84 |
| 신규 테스트 파일 | 2 |
| 신규 테스트 | 7 (단위 6, ignore된 실제 체크포인트 게이트 1) |

### 카테고리별 변경

| 카테고리 | 파일 |
|----------|------|
| 공용 정규화 함수 이전 및 파라미터화 | `src/vision/encoders/qwen2_5_vl.rs`, `src/models/colqwen2_5.rs` |
| 생성 로더 수정 | `src/loading/vlm_qwen.rs` |
| 테스트 | `src/vision/encoders/qwen2_5_vl_tests.rs`, `src/loading/vlm_qwen_tests.rs`, `src/models/colqwen2_5_tests.rs` |

### 관련 커밋

| 해시 | 유형 | 메시지 |
|------|------|--------|
| `0ea8233f6` | fix(qwen2_5_vl) | normalize the raw HF patch-embed layout at load |

### 관련 PR/이슈

- 이슈 #1423: 이 PR이 닫는 이슈.
- PR #1414: ColQwen2.5 범위의 원래 정규화 함수를 추가했고, 동기가 된 7.83 대 8.04 순위 역전을 측정했다.
- 이슈 #1367: 비전이 제거된 Qwen2.5-VL 체크포인트. 같은 `load_qwen2_5_vl` 본문을 수정하며 이 PR 다음에 들어갈 예정이다.
- 에픽 #1348: 이 후속 작업이 속한 상위 에픽.

---

## 7. 후속 조치

### 더 넓은 교훈

이 결함 부류에서 흥미로운 점은 두 절반이 서로 다르게 실패한다는 것이다. 키 리맵 누락은 로드 시점에 크게 실패하고 진단도 쉽다. 레이아웃 불일치는 추론 시점에 조용히 실패하면서 뒤섞인 필터로도 매끄러운 문장을 만들어 내므로, "그럴듯한 말을 하는가"로 만든 스모크 테스트는 절대 잡지 못한다. 둘을 갈라낸 것은 독립적인 기준과의 수치 비교였다. PR #1414이 애초에 이 문제를 찾은 방법이고, 여기서 치환을 증명한 방법도 같다.

실무 규칙은 "항상 레퍼런스와 비교하라"보다 좁다. 레이아웃 혼동에도 텐서의 원소 수가 불변이면 shape 검사와 reshape 성공은 아무 정보도 주지 않으며, 값만이 정보다. `[1280, 3, 2, 14, 14]`와 `[1280, 2, 14, 14, 3]`은 모두 1505280개를 담고, 수정 전 경로가 아무것도 던지지 않은 이유가 정확히 그것이다.
