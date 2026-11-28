# 기술 보고서: PR #1174 - feat(models): qualify Qwen3.8-27B on the qwen3_5 path

**작성일**: 2026-08-16
**작성자**: AI Code Reviewer
**상태**: 완료
**언어**: Rust
**위험도**: Low

---

## 요약

`mlx-community/Qwen3.8-27B-4bit`는 `model_type: "qwen3_5"`를 선언하고 Qwen3.5-27B와 가중치 맵 키 집합까지 바이트 단위로 동일한 구조라서, 이 PR 이전에도 `main`에서 코드 변경 없이 그대로 로드되고 동작했다. 다만 이건 우연이었지 보장된 동작은 아니었다. `Qwen35Config`에는 `deny_unknown_fields`가 없어서 Qwen3.8 세대가 새로 추가한 config 키는 모두 조용히 버려졌는데, 그중 `output_gate_type`, `rope_parameters.mrope_interleaved`, 최상위 `language_model_only` 세 개는 업스트림에서 실제로 동작을 좌우하는 값이었다. 이번 PR은 Qwen3.5 계열 config 파싱 지점 여섯 곳 중 다섯 곳에서 이 조용한 드롭을 명시적 읽기나 이름 붙은 오류로 바꿨고, `vision_start_token_id`를 낡은 기본값 대신 config에서 필수로 요구하도록 고쳤으며, 이 체크포인트를 문서와 실제 모델 테스트 스위트에 등록했다. 이슈는 업스트림 mlx-vlm의 post-pin 수정 두 건을 포팅해달라고도 요청했는데, 둘 다 코드를 작성하기 전에 이미 `main`에 반영되어 있음이 확인됐다. 그래서 이슈의 이 절반은 재구현이 아니라 확인 후 고정(confirm-and-pin)하는 작업으로 바뀌었다. 이번 finalization 단계에서는 최초 PR 리뷰에서 나온 코드 지적 다섯 건(파싱 지점 하나 누락, 새 가드 두 개의 타입 혼동 우회, 비전 토큰 id의 검증 없는 truncating 캐스트, 대소문자 구분 매칭)을 닫았고 문서 지적 세 건을 바로잡았으며, 필수인 `CHANGELOG.md` 업그레이드 안내 문구도 추가했다.

---

## 1. 문제 정의

### 1.1 배경

이슈 #1163은 두 가지를 요청했다. `Qwen3.8-27B-4bit`를 기존 `qwen3_5` 경로에서 qualify하는 것, 그리고 mlxcel의 MLX 핀 이후에 나온 업스트림 mlx-vlm 수정 두 건(Blaizzy/mlx-vlm#1805, 패딩된 vocabulary structured-output mask 수정, Blaizzy/mlx-vlm#1741, chunked-MRoPE position-slice 재사용 수정)을 포팅하는 것이다. 두 포팅 모두 새 작업이 아니라 전제 교정으로 판명났다. `src/server/structured.rs`의 `apply_structured_mask_to_logits`는 이미 bias 길이를 matcher의 vocabulary가 아니라 모델의 logits 축, 즉 `vocab_size_hint`에 고정하고 있었고, `Qwen35Model::forward_with_mrope_state`도 이미 저장된 `position_ids` 텐서를 chunked prefill 윈도우에 맞춰 슬라이스하고 재사용 전에 배치 차원을 검증하고 있었다. 프로젝트의 성능 이슈·전제 교정 컨벤션에 따라, 이슈 절반의 산출물은 재구현이 아니라 기존 동작을 테스트로 확인하고 고정하는 것으로 바뀌었다.

### 1.2 기존 문제점 (리뷰·보안 검토에서 나온 항목, 이번 finalization에서 해결)

- **S-M1** (Medium): `validate_supported()`는 여섯 개의 `Qwen35Config` 파싱 지점 중 다섯 곳에는 연결됐지만, 여섯 번째인 `src/loading/vlm_special.rs`의 MiniCPM-V 4.6 텍스트 백본 로더에서는 의도적으로 빠졌다. 이 불변조건이 `qwen3_5` model_type 문자열에만 묶여 있다는 근거였는데, 그 근거는 성립하지 않는다. 이 불변조건(하드코딩된 SiLU gated-delta 출력, 무조건 적용되는 interleaved MRoPE)은 `Qwen35Model` 자체에 속하고, MiniCPM-V 4.6 로더도 `from_weights`를 통해 같은 `Qwen35Model`을 만든다. 그래서 `output_gate_type: "sigmoid"`를 선언한 MiniCPM-V 4.6 체크포인트는 이 PR이 막으려던 조용히 틀린 출력 실패를 그대로 재현했다.
- **S-M2** (Medium): `validate_qwen35_wrapper_config`와 `Qwen35Config::mrope_interleaved()` 둘 다 대상 키를 `.as_bool()`로 읽었다. 그래서 값이 있지만 타입이 틀린 경우, 올바른 타입이었다면 걸렸을 이름 붙은 오류 대신 조용히 "값 없음"으로 읽혔다. 릴리스 바이너리에서 확인한 결과 `language_model_only: "true"`, `1`, `{}`와 `mrope_interleaved: "false"`, `0` 모두 검증을 통과하고 그대로 로드됐다.
- **S-M3** (Medium): `qwen35_vl_token_ids`는 `vision_start_token_id`를 `i64`에서 `i32`로 변환할 때 검증 없는 `as i32`, 즉 truncating 캐스트를 썼다. 확인해보니 `vision_start_token_id: 1099511627776`(2^40)는 0으로 잘려 나갔고 `vision_start_token_id: -5`는 아무 진단 없이 그대로 로드됐는데, 이는 이 함수 자신의 오류 메시지가 이 키를 필수로 만든 이유로 명시한 "조용한 mis-segmentation" 실패를 정확히 재현한다. `image_token_id`와 `video_token_id`도 같은 검증 없는 캐스트를 쓰고 있었다.
- **R-M2** (Medium): `output_gate_type` 매칭은 `{"silu", "swish"}` 두 표기와만 정확히 일치하는지 보는 대소문자 구분 비교였다. 체크포인트가 `"SiLU"`로 표기하면, 이미 구현된 경로가 정확히 처리할 수 있는 값인데도 로드가 그대로 하드 실패했을 것이다.
- **S-L2** (Low): `OutputGateType` 오류는 문제가 된 config 값을 길이 제한 없이 그대로 되돌려줬다. 1,002,990바이트짜리 `output_gate_type` 값은 1,000,568바이트의 프로세스 출력으로 재현됐다.
- **R-M1** (문서): `docs/supported-models.md`가 Qwen3.5-VL과 Qwen3.8-VL을 연속된 두 개의 불릿, 즉 family 요약 줄과 새 상세 불릿에 중복해서 실었다. 같은 목록의 다른 family는 모두 한 번씩만 등장한다.
- **R-L1** (문서): KV 사용량 문단에서 측정값 64.1 KiB/토큰이 65,536 B/토큰(64.0 KiB) 아키텍처 최소값과 "정확히" 같다고 서술했는데, 실제로는 두 값이 약 0.16% 차이가 난다.
- **R-L2** (문서): 같은 문단의 가중치 맵 키 목록(`model.language_model.*`, `model.visual.*`, `mtp.*`, `lm_head.weight`)은 업스트림 `Qwen/Qwen3.8-27B` 저장소를 설명하는 것이었는데, 같은 문장에서 검증했다고 밝힌 체크포인트는 `mlx-community/Qwen3.8-27B-4bit` 변환본이었다.

### 1.3 위험성

| 위험 | 영향도 | 발생 가능성 |
|-----|-------|-----------|
| MiniCPM-V 4.6이 잘못된 gated-delta 활성화나 MRoPE 레이아웃으로 로드됨 (S-M1) | Medium (오류 없이 조용히 틀린 생성 결과) | 이 키를 선언하는 미래 체크포인트에는 실재. 로컬에서 확인한 체크포인트에는 전부 해당 없음 |
| 타입이 틀린 `language_model_only`/`mrope_interleaved` 값이 실패 없이 로드됨 (S-M2) | Medium (상위 PR이 추가한 가드를 무력화) | 수정 전 릴리스 바이너리에서 재현 확인 |
| 범위를 벗어나거나 오버플로하는 비전 토큰 id가 MRoPE 비전 구간을 mis-segment함 (S-M3) | Medium (조용히 틀린 VLM 출력) | 수정 전 릴리스 바이너리에서 재현 확인 |
| 대소문자만 다른 `output_gate_type` 별칭이 이미 지원되는 로드를 하드 실패시킴 (R-M2) | Low-Medium (false-positive 시작 실패) | 체크포인트나 변환 도구가 표기를 달리하면 실재 |
| 잘못된 config가 메가바이트급 오류 출력을 만듦 (S-L2) | Low (동작상 불편, 정확성 문제는 아님) | 낮지만 인위적 대형 값으로 직접 재현됨 |
| 문서 오류 (R-M1, R-L1, R-L2) | Low (독자 혼란, 기능 영향 없음) | 해당 없음, 수정 완료 |

---

## 2. 기술적 검토 사항

### 2.1 보안 관점

리뷰와 보안 검토는 오케스트레이터와 리뷰어가 finalization 이전에 완료했다. CRITICAL, HIGH 항목은 없었고, 이번 단계에서 위 MEDIUM/LOW 항목 다섯 건을 닫았다.

**발견된 이슈:**

| 이슈 | 심각도 | 상태 |
|-----|-------|-----|
| 여섯 번째 `Qwen35Config` 파싱 지점(MiniCPM-V 4.6)에 `validate_supported()` 누락 | Medium | Fixed (`4e2fab1e`) |
| `language_model_only`/`mrope_interleaved` 타입이 틀린 값이 조용히 "값 없음"으로 처리됨 | Medium | Fixed (`4e2fab1e`) |
| `vision_start_token_id`/`image_token_id`/`video_token_id` 검증 없는 truncating 캐스트, 범위 검사 없음 | Medium | Fixed (`4e2fab1e`) |
| `output_gate_type` 매칭이 대소문자를 구분함 | Medium | Fixed (`4e2fab1e`) |
| `OutputGateType` 오류에 무제한 길이의 config 값이 그대로 실림 | Low | Fixed (`4e2fab1e`) |

S-M1의 `validate_supported()` 호출을 MiniCPM-V 4.6 로더에 연결하기 전에, 로컬에 있는 MiniCPM-V-4.6 체크포인트 두 개(`minicpm-v-4.6-bf16`, `minicpm-v-4.6-mxfp4`)를 먼저 확인했다. 둘 다 `text_config`에 `output_gate_type`이나 `rope_parameters.mrope_interleaved`를 선언하지 않으므로, 새로 추가한 호출이 둘 중 어느 쪽도 시작 실패로 바꾸지 않는다. 이 실제 배포본 `text_config` 형태(24개 레이어, `hidden_size` 1024, `vocab_size` 248094)는 `src/models/qwen3_5_tests.rs`에 회귀 방지 fixture로 고정해뒀다.

### 2.2 성능 관점

없음. 이번 단계의 모든 수정은 로드 시점의 config 검증(S-M1이 붙는 지점 기준으로는 어떤 가중치도 읽기 전에 한 번 실행됨)이거나 문서 교정이다. 벤치마크는 필요하지 않았고 실행하지도 않았다.

### 2.3 호환성/의존성 관점

- **Breaking Changes**: 상위 PR이 이미 도입한 것 외에 추가된 breaking change는 없다. S-M1이 MiniCPM-V 4.6 지점에 새로 추가한 `validate_supported()` 호출은 로컬에 있는 두 체크포인트 모두에서 안전함을 확인했다(2.1절 참고). S-M2와 S-M3는 지금까지 조용히 틀린 출력을 내던 경우를 로드 실패로 바꾸는데, 이 계열에 알려진 체크포인트는 모두 올바른 타입과 범위 안의 값을 쓰고 있어 현재 정상 동작 중인 배포에는 영향이 없다. R-M2는 순수하게 허용 범위를 넓히는 변경이라, 이전보다 더 많은 표기를 받아들일 뿐 이전에 로드되던 값을 거부하는 경우는 없다.
- **새로운 의존성**: 없다.
- **호환성**: `qwen35_vl_token_ids`의 시그니처에 `vocab_size: usize` 매개변수가 추가됐다. 유일한 프로덕션 호출 지점(`src/loading/vlm_qwen.rs`)은 그 시점에 이미 파싱되어 있는 `text_config.vocab_size`를 넘긴다. 테스트 호출 지점도 모두 맞춰 갱신했다.

### 2.4 코드 품질 관점

- **테스트 커버리지**: 파일 세 곳에 걸쳐 새 테스트 18건이 추가됐다. `src/models/qwen3_5_tests.rs`에는 `output_gate_type_matching_is_case_insensitive`, `output_gate_type_error_truncates_an_oversized_value`, `mrope_interleaved_wrong_type_is_a_named_error_not_absent`, `language_model_only_wrong_type_is_a_named_error_not_absent`와 MiniCPM-V-4.6 형태 테스트 세 건(`minicpmv4_6_text_config_passes_validate_supported`, `..._output_gate_type_sigmoid_is_a_named_error`, `..._mrope_interleaved_false_is_a_named_error`)이 추가됐다. `src/loading/vlm_tests.rs`에는 `qwen35_vl_token_ids`의 범위·오버플로 테스트 네 건이 추가됐다. `src/loading/vlm_special_tests.rs`에는 `load_minicpmv4_6_vlm_rejects_an_unsupported_output_gate_type`이 추가됐는데, 이 테스트는 실제 `load_minicpmv4_6_vlm` 진입점을 처음부터 끝까지 구동한다. `validate_supported()`가 이제 어떤 가중치를 읽기도 전에 실행되므로 가중치 파일 없이 만들어낸 `config.json` 하나로 충분하고, S-M1 수정이 없으면 이 테스트는 실패한다. 로더가 이 fixture가 의도적으로 빼둔 `vision_config`에 도달할 때 다른 지점에서 다른 방식으로 실패하기 때문이다.
- **코드 복잡도**: 각 수정은 국지적이다. S-M2는 기존의 관용적인 접근자의 시그니처를 바꿔서 다른 호출자를 깨뜨리는 대신, 그 옆에 private한 엄격 버전 접근자(`mrope_interleaved_checked`) 하나만 추가했다. S-M3는 범위·오버플로 검사를 작은 헬퍼 함수(`qwen35_vl_token_id_in_range`) 하나로 뽑아내 세 개의 id가 공유하게 했다.
- **기술 부채**: 감소했다. 여섯 번째 파싱 지점이 이제는 문서화된 이유 없이 나머지 다섯 곳과 다르게 동작하지 않고, 두 개의 타입 혼동 우회도 사라져서 S-M2가 만드는 새 가드들이 이미 엄격했던 typed `output_gate_type` 필드와 더 이상 일관성이 어긋나지 않는다.

---

## 3. 기술적 선택과 그 이유

### 3.1 `mrope_interleaved()`는 관용적으로 유지하고, 별도의 엄격한 접근자를 추가

**고려한 대안:**

| 옵션 | 장점 | 단점 |
|-----|-----|-----|
| `mrope_interleaved()`의 반환 타입을 `Result<Option<bool>, Qwen35UnsupportedConfig>`로 변경 | 접근자가 하나뿐이라 중복이 없음 | 다른 호출자와 기존 `Option<bool>` 형태를 전제로 한 테스트 어서션이 모두 깨짐. 엄격한 검증이 필요 없는 호출자에게는 true/absent를 관용적으로 읽는 편이 정당한 사용 방식임 |
| **선택: `validate_supported`에서만 쓰는 private `mrope_interleaved_checked()` 추가** | 시그니처 변경이 없고, 엄격/관용 구분이 타입에 명시적으로 드러남 | 같은 JSON 경로를 읽는 함수가 두 개로 늘어남 |

**선택 이유**: 공개 접근자의 `Option<bool>` 형태는 다른 코드와 기존 테스트 스위트가 이미 true/absent라는 흔한 경우를 위해 의존하고 있는 값이다. 엄격한 버전이 필요한 곳은 정확히 하나, `validate_supported` 뿐이므로, 그 범위 안에 한정하면 지적 사항이 요구하는 것보다 훨씬 넓은 파급 범위를 가진 시그니처 변경을 피할 수 있다.

### 3.2 비전 토큰 id는 고정 상수가 아니라 체크포인트 자신의 `vocab_size`로 범위 검사

**선택 이유**: 이번 지적은 오버플로 사례(기존 캐스트에서 2^40이 0으로 잘림)와, 범위 안에 들어오지만 말이 안 되는 사례(값 자체는 `i32`에 들어가지만 체크포인트의 vocabulary를 넘어서는 경우) 둘 다를 짚었다. 고정 상한값으로는 이 계열 안에서 크기가 다른 체크포인트의 두 번째 사례를 잡아내지 못한다. 이미 파싱되어 있는 유일한 프로덕션 호출 지점의 `text_config.vocab_size`를 그대로 흘려넣으면, family 전체에 걸친 고정 상수를 새로 만들지 않고도 두 사례를 모두 닫을 수 있다. 그런 고정 상수는 이미 제거된 248045 기본값이 그랬던 것처럼 시간이 지나면 낡아버릴 수 있다.

### 3.3 되돌려주는 config 값 절단은 오류 variant마다 따로가 아니라 공용 헬퍼로

**선택 이유**: `S-L2`는 `OutputGateType`을 콕 집어 지적했지만, 이번 단계에서 새로 추가하는 두 개의 wrong-type 오류(S-M2)에도 같은 형태의 무제한 echo 문제가 그대로 존재한다. 공용 `MAX_ERROR_VALUE_CHARS` 상수를 쓰는 `truncate_for_error` 헬퍼 하나로 처리하면, 지적이 우연히 이름 붙인 한 지점뿐 아니라 문제의 근본 원인을 닫을 수 있다.

---

## 4. 구현 상세

### 4.1 S-M1: 여섯 번째 `validate_supported()` 호출 지점 (`src/loading/vlm_special.rs`)

```rust
let text_config: models::qwen3_5::Qwen35Config = serde_json::from_value(text_config_value)
    .map_err(|e| anyhow::anyhow!("Failed to parse MiniCPM-V 4.6 text config: {}", e))?;
// The gated-delta / MRoPE invariants `validate_supported` enforces belong
// to `Qwen35Model`, which this loader builds via `from_weights` below,
// not to the `qwen3_5` model_type string. ...
text_config.validate_supported()?;
```

`text_config`를 파싱한 직후, `vision_config`를 파싱하거나 어떤 가중치를 읽기도 전에 배치했다. 그래서 거부 경로(그리고 그 테스트)는 vision config나 가중치 파일이 전혀 필요 없다.

### 4.2 S-M2: 엄격한 타입 오류 감지 (`src/models/qwen3_5.rs`)

```rust
fn mrope_interleaved_checked(&self) -> Result<Option<bool>, Qwen35UnsupportedConfig> {
    match self.rope_parameters.as_ref().and_then(|rp| rp.get("mrope_interleaved")) {
        None => Ok(None),
        Some(value) => value.as_bool().map(Some).ok_or_else(|| {
            Qwen35UnsupportedConfig::MropeInterleavedWrongType(truncate_for_error(
                &value.to_string(),
                MAX_ERROR_VALUE_CHARS,
            ))
        }),
    }
}
```

`validate_qwen35_wrapper_config`에도 `language_model_only`를 위해 `match value.as_bool() { Some(true) => ..., Some(false) => {}, None => ... }`라는 동일한 형태를 추가했다. `Qwen35UnsupportedConfig`에는 절단된 문제 값을 담는 새 variant `MropeInterleavedWrongType`과 `LanguageModelOnlyWrongType` 두 개가 추가됐다.

### 4.3 S-M3: 범위 검사가 붙은 비전 토큰 id (`src/loading/vlm.rs`)

```rust
fn qwen35_vl_token_id_in_range(field: &str, raw: i64, vocab_size: usize) -> anyhow::Result<i32> {
    let id = i32::try_from(raw).map_err(|_| anyhow::anyhow!(
        "Qwen3.5-family config.json has `{field}={raw}`, which does not fit in a 32-bit token id. ..."
    ))?;
    if id < 0 || (id as usize) >= vocab_size {
        anyhow::bail!(
            "Qwen3.5-family config.json has `{field}={id}`, which is outside the checkpoint's \
             vocabulary (vocab_size={vocab_size}). ..."
        );
    }
    Ok(id)
}
```

`qwen35_vl_token_ids`에 `vocab_size: usize` 매개변수를 추가하고 세 개의 id 모두 이 헬퍼를 거치도록 했다. `src/loading/vlm_qwen.rs`에 있는 유일한 호출 지점은 `text_config.vocab_size`를 넘긴다.

### 4.4 R-M2: 대소문자를 구분하지 않는 gate 매칭 (`src/models/qwen3_5.rs`)

```rust
if !gate.eq_ignore_ascii_case("silu") && !gate.eq_ignore_ascii_case("swish") {
    return Err(Qwen35UnsupportedConfig::OutputGateType(...));
}
```

### 4.5 S-L2: 길이가 제한된 오류 echo (`src/models/qwen3_5.rs`)

```rust
const MAX_ERROR_VALUE_CHARS: usize = 64;

fn truncate_for_error(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        text.to_string()
    } else {
        format!("{}...", text.chars().take(max_chars).collect::<String>())
    }
}
```

### 4.6 문서 (`docs/supported-models.md`, `CHANGELOG.md`)

R-M1은 family 요약 불릿에서 중복된 `Qwen3.5-VL, Qwen3.8-VL` 언급을 제거했다. R-L1은 KV 사용량 문단을 고쳐서, 측정값(64.1 KiB/토큰)과 이론값(64.0 KiB/토큰)을 정확히 같다고 부르는 대신 두 값과 그 사이의 약 0.16% 차이를 따로 서술했다. R-L2는 가중치 맵 키 목록을 업스트림 `Qwen/Qwen3.8-27B` 형태와 검증 대상인 `mlx-community/Qwen3.8-27B-4bit` 형태로 나눴고, 각각을 해당하는 로컬 체크포인트의 `model.safetensors.index.json`으로 직접 확인했다. `CHANGELOG.md`에는 `vision_start_token_id`가 필수가 됐다는 업그레이드 안내 문구와 함께 이슈 #1163을 위한 `## [Unreleased]` 항목을 추가했다.

---

## 5. 전제 교정: 업스트림 포팅 두 건은 이미 구현되어 있었다

이슈의 다른 산출물, 즉 Blaizzy/mlx-vlm#1805와 Blaizzy/mlx-vlm#1741 포팅은 새 구현이 아니라 전제 교정으로 귀결됐다. 코드를 작성하기 전에 먼저 확인했고, 이번 finalization에서는 mutation 테스트로 다시 검증했다.

- **Blaizzy/mlx-vlm#1805** (패딩된 vocabulary structured-output mask): `src/server/structured.rs`의 `apply_structured_mask_to_logits`는 이미 bias 길이를 matcher의 vocabulary가 아니라 모델의 logits 축인 `vocab_size_hint`에 고정하고 있었다. **이번 단계에서 mutation으로 검증**: `vocab_size`를 `vocab_size_hint` 대신 `constraint.vocab_size()`(matcher의 248,077개 vocabulary)로 읽도록 강제로 바꾸자, `apply_mask_covers_the_qwen3_8_padded_lm_head`가 어서션 실패가 아니라 `libc++abi: terminating due to uncaught exception ... [broadcast_shapes] Shapes (1,248320) and (1,248077) cannot be broadcast`로 테스트 프로세스 자체를 abort시켰다. 이건 `mlxcel_core::add` 내부의 FFI 레벨 실패다. 이 abort를 확인한 직후 mutation을 되돌렸고, mutation 이전 상태와 비교한 `git diff`는 비어 있다.
- **Blaizzy/mlx-vlm#1741** (chunked-MRoPE position-slice 재사용): `src/models/qwen3_5.rs`의 `mrope_position_source`는 저장된 `position_ids` 텐서를 재사용하기 전에 이미 `shape[1] == batch`(Blaizzy/mlx-vlm#1040)와 `shape[2] >= cache_offset + seq_len`(Blaizzy/mlx-vlm#1048)을 모두 요구하고 있었고, 조건을 만족하지 않으면 delta 기반 재계산으로 떨어졌다. **이번 단계에서 mutation으로 검증**: `shape[1] == batch` 조건절을 빼자 `mrope_position_source_rejects_a_batch_mismatch_and_a_wrong_rank`가 실패했다(`left: SliceStored { start: 0, end: 8 }, right: Recompute`). 이 실패를 확인한 직후 mutation을 되돌렸고, 상위 커밋과 비교한 diff는 비어 있다.

두 mutation 실행 모두 이 프로젝트의 release-only 빌드 제약(`DEVELOPER_DIR=/Applications/Xcode-26.6.0.app/Contents/Developer`, `--release --features metal,accelerate`)을 그대로 따랐고, 확인이 끝나는 즉시 되돌려서 이후 작업을 이어갔다. 그래서 finalization 커밋은 위에서 다룬 코드 지적 다섯 건과 문서 지적 세 건으로만 한정된다.

---

## 6. 변경 요약

### 통계

| 항목 | 값 |
|-----|---|
| 변경된 파일 수 (코드 수정 커밋 `4e2fab1e`) | 7 |
| 변경된 파일 수 (문서/changelog 커밋 `298b511e`) | 2 |
| 추가/삭제 라인 (코드 수정) | +519 / -45 |
| 추가/삭제 라인 (문서/changelog) | +6 / -2 |
| 테스트 추가 | `models::qwen3_5_tests` 7건 + `loading::vlm::tests` 4건 + `loading::vlm::special::tests` 1건, 총 12개 테스트 함수 |

### 카테고리별 변경

| 카테고리 | 변경 수 | 주요 내용 |
|---------|--------|----------|
| 검증 연결 | 1 | S-M1: MiniCPM-V 4.6 지점에 `validate_supported()` 연결 |
| 타입 안전성 강화 | 2 | S-M2: `mrope_interleaved`/`language_model_only` 엄격 검사 |
| 범위 검사 | 1 | S-M3: 비전 토큰 id 세 개를 위한 `qwen35_vl_token_id_in_range` |
| 매칭 정확도 | 1 | R-M2: 대소문자를 구분하지 않는 `output_gate_type` |
| 오류 출력 안전성 | 1 | S-L2: `truncate_for_error` / `MAX_ERROR_VALUE_CHARS` |
| 문서 | 3 | `docs/supported-models.md`의 R-M1, R-L1, R-L2 |
| 릴리스 노트 | 1 | `CHANGELOG.md`의 `## [Unreleased]` 항목과 업그레이드 안내 |

### 관련 커밋

| Hash | Type | Message |
|------|------|---------|
| `34f14df3` | feat | qualify Qwen3.8-27B on the qwen3_5 path (상위 PR) |
| `4e2fab1e` | fix | harden Qwen3.5-family config validation gaps from PR #1174 review |
| `298b511e` | docs | fix Qwen3.5-VL duplicate listing, KiB precision, and weight-map attribution |

---

## 7. 후속 조치

### 완료 필요

- [ ] 없음. 리뷰와 보안 검토에서 CRITICAL이나 HIGH 항목은 보고되지 않았고, MEDIUM/LOW 코드 지적 다섯 건과 문서 지적 세 건 모두 이번 단계에서 수정했다.

### 향후 개선 사항 (알려진 제약으로 기록, 이번 PR에서는 수정하지 않음)

- `vision_end_token_id`는 여전히 트리 어디에서도 읽지 않는다. 상위 PR이 이미 문서로 남긴 결정 그대로다. 아무도 이 값을 쓰지 않으므로 필드를 추가하면 죽은 코드가 된다.
- 이 family의 MTP speculative decoding 공백(mlx-community 변환본이 `mtp.*`를 빼고 drafter를 별도로 배포하는 문제)은 계속 #1165로, video 입력은 #1166으로 추적되며 둘 다 이번 finalization 범위 밖이다.
- `check_cross_repo_refs.py`는 이 트리에서 bare `#1163`/`#1165`/`#1166` 참조를 advisory로 지적한다. 스크립트의 휴리스틱이 3자리 이상 bare `#NNN`을 전부 업스트림 가능성이 있다고 보기 때문인데, 이들은 실제로는 같은 저장소(`lablup/mlxcel`) 참조이고 스크립트 자체도 advisory 전용(`exit 0`)이라 별도 조치는 하지 않았다.

---

## 부록

### A. 테스트 결과

- `cargo test --release -p mlxcel --lib --features metal,accelerate qwen3_5`: 25건 통과, 11건 무시(기존 serial-MLX 게이트), 0건 실패.
- `cargo test --release -p mlxcel --lib --features metal,accelerate loading::vlm`: 178건 통과, 0건 실패. 새로 추가한 end-to-end 연결 테스트 `load_minicpmv4_6_vlm_rejects_an_unsupported_output_gate_type`를 포함한다.
- `cargo clippy --release --lib --tests --features metal,accelerate -- -D warnings`: 클린.
- `cargo fmt --check`: 클린 (새 코드가 100컬럼을 넘긴 세 줄을 `cargo fmt` 한 번으로 정리한 뒤).
- `python3 scripts/ci/check_cross_repo_refs.py`: advisory 항목만 있음(향후 개선 사항 참고), exit 0.
- S-M1을 위한 로컬 체크포인트 확인: `minicpm-v-4.6-bf16`과 `minicpm-v-4.6-mxfp4`의 `text_config`를 직접 읽어 `output_gate_type`과 `rope_parameters.mrope_interleaved`가 둘 다 없음을 확인했다. 새로 추가한 `validate_supported()` 호출이 어느 쪽도 시작 실패로 바꾸지 않는다.
- R-L2를 위한 가중치 맵 확인: `qwen3.8-27b-hf-bf16/model.safetensors.index.json`(`model.language_model.*`, `model.visual.*`, `mtp.*` 키 15개, `lm_head.weight`)과 `qwen3.8-27b-4bit/model.safetensors.index.json`(`language_model.lm_head.*`를 포함한 `language_model.*`, `vision_tower.*`, `mtp.*` 키 0개)을 직접 확인했다.
- mutation 검증 (이번 단계, 각 경우 실패를 확인한 직후 되돌림): `apply_structured_mask_to_logits`가 `vocab_size_hint` 대신 matcher vocabulary로 크기를 정하도록 바꾸면 `apply_mask_covers_the_qwen3_8_padded_lm_head`가 MLX broadcast-shape 불일치로 abort된다. `mrope_position_source`에서 `shape[1] == batch` 검사를 빼면 `mrope_position_source_rejects_a_batch_mismatch_and_a_wrong_rank`가 실패한다.

### B. 참고 자료

- 이슈 #1163 (사양)
- 이슈 #1165 (이 family의 MTP speculative decoding, 범위 밖)
- 이슈 #1166 (이 family의 video 입력, 범위 밖)
- `src/models/qwen3_5.rs` (`Qwen35Config::validate_supported`, `mrope_interleaved_checked`, `truncate_for_error`), `src/loading/vlm.rs` (`qwen35_vl_token_ids`, `qwen35_vl_token_id_in_range`), `src/loading/vlm_special.rs` (`load_minicpmv4_6_vlm`)
- PR #1174의 리뷰·보안 코멘트
- Blaizzy/mlx-vlm#1805, Blaizzy/mlx-vlm#1741, Blaizzy/mlx-vlm#1040, Blaizzy/mlx-vlm#1048, Blaizzy/mlx-vlm#1812 (상위 PR이 인용했고 이번에 다시 확인한 업스트림 참조)
