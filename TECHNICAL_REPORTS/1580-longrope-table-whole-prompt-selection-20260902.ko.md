# 기술 보고서: PR #1580 - fix(phi3): select the LongRoPE table by whole-prompt position

**작성일**: 2026-09-02
**작성자**: mlxcel maintainers
**리뷰어**: implementation review cycle
**상태**: 구현 완료, 단위 테스트로 검증함. 4096 토큰 위아래 실제 체크포인트 토큰 일치 검증은 아직 남아 있음
**언어**: Rust, Markdown
**위험도**: Medium (모든 `longrope` Phi 체크포인트가 학습 컨텍스트 이하에서 쓰는 회전 주파수를 바꾸고, 어텐션 핫 패스에서 읽는 크로스 크레이트 prefill 힌트를 새로 추가함)

---

## 요약

Phi-3.5, Phi-4-mini, Phi-4-multimodal은 `rope_scaling.type`을 `longrope`(예전 변환본은 `su`)로 선언하고 차원별 factor 리스트를 두 개 담고 있다. 학습된 모델은 시퀀스가 `original_max_position_embeddings` 안에 들어가는 동안 `short_factor`로 회전하고, 그보다 길어지면 `long_factor`로 회전한다. mlxcel은 `long_factor`로 테이블 하나만 만들어 모든 위치에 썼다. `mlx-community/Phi-3.5-mini-instruct-4bit`에서 `long_factor`는 저주파 쌍에서 64.8까지 올라가는데 `short_factor`는 2.9를 넘지 않으므로, 학습된 4096 토큰 컨텍스트 안의 모든 프롬프트가 틀린 테이블로 회전됐다.

이번 수정은 두 테이블을 모두 유지하고, 각 테이블에 자기 어텐션 크기 스케일(`short_mscale` / `long_mscale`이 있으면 그것, 없으면 기본값 `sqrt(1 + ln(M / L) / ln(L))`)을 붙이며, 실제 체크포인트가 담고 있는 최상위 `original_max_position_embeddings`를 읽는다. 이슈에 없던 부분은 이것이다. 선택을 한 forward 패스의 `offset + seq_len`으로 내리면 안 된다. mlxcel은 긴 프롬프트를 청크로 나눠 prefill하고, 그 규칙은 프롬프트 하나를 두 테이블에 걸쳐 쪼개기 때문이다. 새로 만든 `mlxcel_core::prefill_span`이 prefill을 쪼개는 드라이버에서 그 값이 필요한 모델로 전체 프롬프트 길이를 전달한다.

---

## 1. 문제 정의

### 1.1 배경

`src/models/phi3.rs`의 `configure_su_rope`는 `scaling.long_factor`만 읽어 `freqs[i] = long_factor[i] * base^(2i/d)`를 만들고, 그 배열 하나를 모든 `Phi3Attention`에 `su_rope_freqs`로 저장했다. `short_factor`는 `RopeScaling` 구조체에 선언만 돼 있고 트리 어디에서도 읽히지 않았다. 이 단일 테이블이 fused 양자화 경로(`forward_fused_qkv_split_su_scaled_rope`)와 그래프 폴백(`prepare_qkv_with_rope`) 양쪽에 들어갔으므로, 두 경로는 서로 다르게 틀린 게 아니라 똑같이 틀려 있었다.

어텐션 크기 스케일은 항상 기본 공식이었고, `scaling.original_max_position_embeddings`에 4096 하드코딩 폴백을 붙여 계산했다. `models/mlx/phi-3.5-mini-4bit/config.json`은 `original_max_position_embeddings`를 블록이 아니라 최상위에 두므로, 실제로 쓰인 값은 폴백이었고 그 체크포인트가 선언한 값이 마침 4096이었기 때문에만 맞았다. 최상위에 다른 값을 선언한 체크포인트는 조용히 4096으로 읽혔을 것이다. `short_mscale`과 `long_mscale`은 구조체에 필드조차 없었으므로, 그 키를 담은 Phi-4 config는 serde 단계에서 값이 버려졌다.

`src/models/phi4mm.rs`는 `phi3::ModelArgs`와 `phi3::Phi3Model`을 그대로 별칭으로 쓰므로 Phi-4-multimodal의 텍스트 디코더가 이 동작을 전부 물려받았고, 같은 디코더를 쓰는 Phi-3 Vision과 Phi-4 SigLIP VLM도 마찬가지였다.

### 1.2 기존 문제점

- **짧은 프롬프트가 전부 long 테이블로 회전됐다.** Phi-3.5-mini에서 두 리스트는 저주파 쌍에서 한 자릿수 이상 차이 나므로, 반올림 수준의 차이가 아니라 모델이 학습된 영역에 대해 아예 틀린 위치 인코딩이다.
- **평소의 스모크 테스트로는 보이지 않는 결함이다.** 값이 틀렸어도 유한한 회전 테이블은 여전히 유창한 문장을 뽑고, 여섯 토큰짜리 greedy 프롬프트로는 두 테이블을 구분할 수 없다. 위치에 따라 테이블을 고르는 구현과 토큰 단위로 비교해야만 드러난다.
- **수정된 동작의 오라클로 mlx-lm을 쓸 수 없다.** mlx-lm의 `SuScaledRotaryEmbedding`은 `short_factor`와 `short_mscale`을 생성자 인자로 받아놓고 `self._freqs`는 `long_factor`로, `self.scale`은 `long_mscale`로만 만든다. 즉 #1358 이전의 mlxcel은 mlx-lm과 정확히 일치했고, 아마 그래서 리뷰를 통과했다. 여기서 mlx-lm과 일치한다는 것은 체크포인트와 일치하지 않는다는 뜻이다.
- **이슈의 규칙을 그대로 구현했다면 원래 버그보다 나쁜 버그를 냈을 것이다.** 2.1을 보라.

### 1.3 위험성

| 위험 | 영향 | 가능성 |
|------|------|--------|
| 청크 prefill이 프롬프트 하나를 두 테이블에 걸쳐 쪼개 한 KV 캐시에 두 회전이 섞임 | Critical (몇 토큰 만에 출력이 붕괴) | 선택을 forward 패스 단위로 내리면 확실 |
| 이후 기여자가 mlxcel을 mlx-lm의 long 전용 동작 쪽으로 "고침" | High (원래 결함을 조용히 되살림) | 이 트리 대부분에서 mlx-lm이 기준이므로 중간 정도 |
| 패치한 두 곳 밖의 드라이버가 총 길이를 알리지 않고 `longrope` 모델을 청크 prefill함 | High | 현재는 낮음(그런 드라이버가 둘뿐), 리더의 `max` 폴백이 상한을 잡아줌 |
| 디코드 중 `L`을 넘을 때 transformers와 갈라짐 | Low (문서화했고, 학습된 4096 컨텍스트 밖) | 구조상 확실 |

---

## 2. 기술적 검토 사항

### 2.1 근본 원인과 이슈 명세의 교정

이슈는 forward 패스 단위 선택을 명시했다.

```
position_end = o + s
use_long     = position_end > L
```

이는 HuggingFace transformers가 계산하는 것과 정확히 같고(`seq_len = torch.max(position_ids) + 1` 다음 `seq_len > self.original_max_position_embeddings`), transformers에서는 프롬프트 전체를 한 패스로 prefill하므로 맞다. mlxcel은 그렇지 않다. `src/lib/mlxcel-core/src/generate.rs`는 단일 시퀀스 CLI/벤치 경로에 `DEFAULT_PREFILL_CHUNK = 2048`을 두고, 서버 배치 스케줄러는 자체 `--prefill-chunk-size`로 prefill을 틱에 걸쳐 나눈다.

패스 단위 규칙에서 5136 토큰 프롬프트를 기본 청크 크기로 처리하면 이렇게 갈라진다.

| 청크 | offset | len | `offset + len` | 테이블 |
|------|--------|-----|----------------|--------|
| 1 | 0 | 2048 | 2048 | short |
| 2 | 2048 | 2048 | 4096 | short |
| 3 | 4096 | 1040 | 5136 | long |

그러면 KV 캐시가 서로 다른 두 테이블로 회전된 키를 함께 담게 되고, 측정된 greedy 출력은 몇 토큰 만에 반복으로 붕괴한다. 올바른 규칙은 프롬프트 전체가 결정한다는 것이다. `L`보다 긴 프롬프트의 모든 청크는 long 테이블을, 그보다 짧은 프롬프트의 모든 청크는 short 테이블을 쓴다.

### 2.2 전체 프롬프트 길이를 어디서 가져올 것인가

어텐션 레이어가 보는 것은 `(x, cache, mask)`와 캐시의 `offset`뿐이고, 그 시그니처에 미래는 없다. 네 가지를 검토했다.

1. **`KVCache`의 필드.** 시퀀스 단위이고 수명도 정확히 맞지만, `KVCache`는 트리의 모든 모델이 공유하는 캐시 타입이고 생성자가 여럿이며 detach/adopt serde 경로와 paged 백엔드까지 붙어 있다. 한 계열 때문에 건드리기엔 파급 범위가 너무 크다.
2. **`LanguageModel` 트레이트 훅.** 이 트리에서는 자연스러운 방식이지만(`after_prefill`, `trim_internal_caches`, `prepare_sequence_state`가 이미 있다) set/clear 규율이 필요하고, 자연스러운 clear 지점인 `after_prefill`은 prefill 전체가 끝날 때만 호출된다. 서버에서는 청크 사이에 스케줄러가 끼워 넣는 디코드 틱 동안 힌트가 남게 되는데, 그게 바로 피해야 할 누수다.
3. **`Phi3Model`을 청크 prefill에서 제외.** CLI에서는 맞지만 실제 메모리 회귀다. 10만 토큰 Phi 프롬프트를 한 패스로 prefill하게 된다. 서버의 청크 결정은 `supports_chunked_prefill`을 보지도 않으므로 별도 수정도 필요하다.
4. **`mlxcel-core`에 두는 RAII 가드 기반 thread-local 알림.** 이것을 택했다.

### 2.3 호환성/의존성 관점

`Phi3Attention`의 `su_rope_freqs`, `su_rope_scale`, `su_rope_scale_arr`는 pub이었지만 `src/models/phi3.rs` 밖에 읽는 곳이 없어서, 이를 `su_rope: Option<SuRope>` 하나로 바꿔도 깨지는 게 없다. `src/loading/vlm_special.rs`와 `src/model_metadata.rs`는 `phi3::ModelArgs`만 참조하는데, 여기에는 기본값이 있는 `Option` 필드가 하나 늘었을 뿐이라 영향이 없다. `src/lib/mlxcel-xla/src/emitter/config.rs`는 원시 JSON에서 `rope_scaling.long_factor`를 직접 읽으며 이번 변경과 무관하다.

### 2.4 코드 품질 관점

`SuRope`가 두 테이블과 임계값을 함께 소유하므로 "두 테이블이 같이 있거나 둘 다 없거나"가 서로 어긋날 수 있는 `Option` 두 개가 아니라 구조로 보장된다. `Phi3Attention::forward`는 테이블을 한 번만 고르고 같은 `&SuRopeTable`을 fused 경로와 그래프 경로에 넘기므로 두 경로가 어떤 테이블을 썼는지에 대해 어긋날 수 없다. 이전 형태는 주파수와 스케일을 각 경로에 별도 인자로 넘겼다.

---

## 3. 기술적 선택과 그 이유

### 3.1 트레이트 훅이 아니라 thread-local RAII 알림

`mlxcel_core::prefill_span`은 `thread_local!` 안에 `Cell<Option<i32>>`을 두고, drop될 때 이전 값을 복원하는 `PrefillSpan` 가드를 넘겨준다. 이 형태를 택한 이유는 세 가지다.

- **수명이 prefill 전체가 아니라 forward 호출이다.** 서버 스케줄러는 한 프롬프트의 청크 두 개 사이에 다른 시퀀스의 디코드 배치를 돌리므로, 청크 하나의 forward보다 오래 사는 힌트는 다른 시퀀스에 이 프롬프트의 길이를 넘겨준다. `let logits = { ... }` 블록에 가드를 두면 그 제약이 그대로 표현되고, 트레이트 훅은 같은 규율을 컴파일러 도움 없이 지켜야 한다.
- **thread-local이 맞는 폭이다.** `KVCache`는 의도적으로 `Send`도 `Sync`도 아니므로 모델 forward는 항상 그것을 구동하는 스레드에서 돈다.
- **모델별 배선이 필요 없다.** 리더는 루트 `mlxcel` 크레이트에 있고 세터 하나는 `mlxcel-core`에 있으므로 상태는 `mlxcel-core`에 있어야 한다. 거기서라면 CLI 드라이버와 서버 드라이버가 `LanguageModel`의 새 트레이트 메서드, `LoadedModel`의 위임 매크로, VLM 래퍼를 거치지 않고 바로 접근한다.

리더는 알림을 그대로 믿지 않고 `max(알림, offset + seq_len)`을 쓴다. 올바른 알림에서는 두 값이 일치하고, 최댓값을 쓰면 적게 알리는 드라이버가 긴 시퀀스를 짧게 보이게 만들 수는 없고 그 반대만 가능해진다.

### 3.2 디코드에서는 재인코딩이 아니라 제자리 전환

생성이 `L`을 넘으면 transformers는 KV 캐시를 버리고 시퀀스 전체를 long 테이블로 다시 인코딩한다(`prepare_inputs_for_generation`의 `input_ids.shape[1] >= original_max_position_embeddings + 1` 및 `past_length <= original_max_position_embeddings` 분기). 이번 변경은 `offset + 1 > L`에서 long 테이블로 바꾸고 이미 캐시에 들어간 short 테이블 키는 그대로 둔다. 이슈가 요구한 방식이고 훨씬 싸지만, parity가 아니라 근사다. `docs/supported-models.md`와 `SuRope::table_for` 문서 주석에 기록했고, 디코드 중 `L`을 넘는 생성에 대해서는 transformers parity를 주장하지 말라고 명시했다.

### 3.3 `long_factor`만 있는 블록은 기존 동작 유지

트리에 그런 config는 없지만, `short_factor`가 없거나 너무 짧으면 두 테이블 모두 `long_factor`로 만든다. 그런 체크포인트가 있다면 부수 효과로 동작이 바뀌는 대신 변경 전과 비트 단위로 동일하게 남는다.

---

## 4. 구현 상세

### 4.1 두 테이블

```rust
pub struct SuRopeTable {
    pub freqs: UniquePtr<MlxArray>,   // factor[i] * rope_theta^(2i / rope_dims)
    pub scale: f32,                   // short_mscale / long_mscale, 없으면 기본값
    scale_arr: Option<UniquePtr<MlxArray>>,  // scale == 1.0이면 None
}

pub struct SuRope {
    short: SuRopeTable,
    long: SuRopeTable,
    original_max: i32,
}
```

`SuRope::from_args`는 블록이 `longrope`나 `su`를 이름으로 갖고 `long_factor` 길이가 최소 `rope_dims / 2`일 때만 값을 돌려준다. 기존 코드가 걸던 것과 같은 가드이며, 잘린 리스트는 여전히 반쯤 채운 테이블 대신 아무것도 만들지 않는다.

### 4.2 선택

```rust
fn table_for(&self, offset: i32, seq_len: i32) -> &SuRopeTable {
    let pass_end = offset.saturating_add(seq_len);
    let span =
        mlxcel_core::prefill_span::current().map_or(pass_end, |total| total.max(pass_end));
    if span > self.original_max { &self.long } else { &self.short }
}
```

`Phi3Attention::forward`가 이것을 한 번 호출하고 결과를 두 경로에 넘긴다.

### 4.3 알림 지점

`mlxcel-core`의 `chunked_prefill_last_logits`는 청크 루프를 감싸 `prompt_tokens.len()`을 한 번 알린다. 그 함수 안에서는 다른 게 돌지 않는다.

서버에서는 forward마다가 아니라, 한 시퀀스의 prefill 작업을 수행하는 스케줄러 진입점마다 한 번씩 알린다. `BatchScheduler`의 `announce_prefill_span(&seq)` 헬퍼를 쓰고 가드는 함수 스코프인데, 그게 맞는 수명이다. 스케줄러는 한 프롬프트의 청크 두 개 사이에 다른 시퀀스의 디코드 배치를 돌리지만, 이 함수들 안에서 도는 것은 전부 한 시퀀스의 것이기 때문이다. 지점은 `execute_full_prefill`, `start_chunked_prefill`, `continue_chunked_prefill`, `capture_history_boundary_snapshot`, `run_next_prompt_cache_warmup`이다. `run_padded_batched_prefill`은 prefill이지만 의도적으로 알리지 않는다. offset 0에서 시작해 코호트에서 가장 긴 행만큼을 커버하므로 그 행에 대해서는 패스 자체가 이미 답을 담고 있고, 스칼라 하나로 더 짧은 행들에 다른 말을 할 수도 없다.

CLI의 비청크 분기도 자기 `offset + seq_len`이 이미 시퀀스 길이이므로 아무것도 알리지 않는다.

### 4.4 첫 시도는 엉뚱한 곳에 알렸고, 서버만 그걸 잡아냈다

이 변경의 첫 버전은 청크 prefill 두 함수 안, 청크 하나를 돌리는 `let logits = { ... }` 블록에서만 알렸다. CLI 게이트는 전부 통과했다. 4비트 체크포인트가 1078 토큰과 5136 토큰 프롬프트 양쪽에서 위치 선택형 오라클과 30토큰 중 30개 일치했고, bf16 체크포인트도 f16 캐스트 오라클과 일치했으며, phi-3-mini는 무변화였고, `MLXCEL_PREFILL_CHUNK=0`이 2048과 같았다. 그런데 서버는 같은 5136 토큰 프롬프트에 대해 `POST /v1/completions`에서 반복을 뱉었다. 프롬프트 캐시가 기본으로 켜져 있고, 그 경로가 수정이 건드리지 않은 forward 두 개를 더 통과하기 때문이다. `capture_history_boundary_snapshot`은 `prompt_tokens[start..boundary]`를 넘기는데, 이것은 프롬프트는 임계값을 넘어도 자기는 그 아래에서 끝날 수 있는 진부분집합이다. `run_next_prompt_cache_warmup`은 스냅샷 복원 후 delta를 넘기는데, non-batching 계열에서는 복원된 접두가 스케줄러의 `KVCache::offset`에 반영되지 않는다.

대응은 위의 함수 스코프 헬퍼와, `src/server/batch/scheduler/prefill_span_coverage_tests.rs`다. 헬퍼는 저자가 눈으로 본 forward가 아니라 진입점이 도달할 수 있는 모든 forward를 덮고, 후자는 `src/server/batch/scheduler` 아래의 모든 모델 forward를 prefill과 디코드로 분류해 표에 없는 함수에서 forward가 나타나면 실패하는 소스 수준 가드다.

### 4.5 스케일 적용

두 경로 모두 회전 전에 Q와 K의 rotary 앞부분에 테이블의 스케일을 곱한다. 기존 코드가 `su_rope_scale`을 적용하던 자리와 같다. 업스트림은 대신 `cos`와 `sin`에 곱하는데, RoPE는 입력에 대해 선형이므로 둘은 같은 사상이고 입력에 곱하는 쪽이 더 작은 텐서에 곱셈 한 번이다.

---

## 5. 검증

| 명령 | 결과 |
|------|------|
| `cargo test --profile test-fast --features metal,accelerate --lib models::phi3::tests` | 12 passed |
| `cargo test --profile test-fast --features metal,accelerate --lib server::batch::scheduler::prefill_span_coverage` | 3 passed |
| `cargo test --profile test-fast --features metal,accelerate -p mlxcel-core --lib prefill_span` | 5 passed |
| `cargo test --profile test-fast --features metal,accelerate -p mlxcel-core --lib generate::tests` | 36 passed |
| `cargo test --profile test-fast --features metal,accelerate --lib models::rope_utils` | 28 passed |
| `cargo clippy --profile test-fast --lib --tests --features metal,accelerate -- -D warnings` (두 크레이트) | clean |
| `cargo fmt --all -- --check` | clean |
| `python3 scripts/ci/check_cross_repo_refs.py` | OK |

phi3 테스트가 다루는 것: 두 테이블의 주파수 구성을 닫힌 형태와 대조; `(0, L)`, `(0, L + 1)`, `(L - 1, 1)`, `(L, 1)`에서의 `L` 임계 전환; 5136 토큰과 3000 토큰 프롬프트를 2048 토큰 청크로 나눈 청크 prefill 케이스와, 알림이 없으면 5136 토큰 프롬프트가 실제로 두 테이블에 걸쳐 쪼개진다는 단언(알림을 제거하면 테스트가 깨진다); `short_mscale` 재정의와 `long_mscale` 부재 시 기본값 유지; `original_max_position_embeddings`의 최상위/블록 해석과 4096 폴백; `rope_scaling`이 없는 config와 `linear` 블록이 테이블을 만들지 않음; `long_factor`만 있는 블록이 두 테이블에 같은 값을 씀; 합성 4비트 양자화 레이어에서 테이블별 fused 대 그래프 상대 RMS parity, 그리고 두 테이블이 실제로 다른 Q를 낸다는 가드(공허하게 통과하지 못하게).

### 여기서 검증하지 않은 것

실제 체크포인트 게이트는 남아 있고 구현 단위 밖에서 수행한다. 4096 토큰 미만 프롬프트에서 `models/mlx/phi-3.5-mini-4bit`를 위치 선택형 transformers 오라클과 greedy 토큰 대조, 같은 체크포인트로 4096 토큰 초과 프롬프트에서 같은 오라클과 토큰 일치이면서 현재 mlxcel 출력과 무변화 확인, 그리고 회귀 대조군으로 `models/mlx/phi-3-mini-4bit`(`rope_scaling` 없음). 전체 `--workspace` 테스트와 `--all-targets` clippy도 이 단위 밖이다.

---

## 6. 변경 요약

### 통계

| 항목 | 값 |
|------|-----|
| 변경 파일 | 11, 보고서 2개 별도 |
| 추가 | 1629 (보고서 포함) |
| 삭제 | 82 |
| 신규 모듈 | 1 (`mlxcel_core::prefill_span`) |
| 신규 단위 테스트 | 19 (`phi3_tests.rs` 10개, `prefill_span_tests.rs` 5개, `prefill_span_coverage_tests.rs` 3개, `generate.rs` 1개) |

### 카테고리별 변경

| 카테고리 | 파일 |
|----------|------|
| 모델 수정 | `src/models/phi3.rs` |
| 코어 메커니즘 | `src/lib/mlxcel-core/src/prefill_span.rs`, `src/lib/mlxcel-core/src/lib.rs`, `src/lib/mlxcel-core/src/generate.rs` |
| 서버 배선 | `src/server/batch/scheduler/prefill.rs`, `src/server/batch/scheduler/prompt_cache.rs`, `src/server/batch/scheduler/mod.rs` |
| 테스트 | `src/models/phi3_tests.rs`, `src/lib/mlxcel-core/src/prefill_span_tests.rs`, `src/server/batch/scheduler/prefill_span_coverage_tests.rs` |
| 문서 | `docs/supported-models.md` |

### 관련 커밋

- `a770022` fix(phi3): select the LongRoPE table by whole-prompt position
- `docs: add technical report for PR #1580`
- `fix(server): announce the prefill span on the prompt-cache forwards`

### 관련 PR/이슈

- Closes #1358

---

## 7. 후속 조치

1. 병합 전에 5절의 실제 체크포인트 게이트 세 개를 수행할 것.
2. `src/models/minicpm3.rs`에는 JSON 맵에서 스칼라 `long_factor`를 읽는 자체 `SuScaledRoPE`가 있다. 형태가 다르지만(차원별 리스트가 아니라 base를 스케일하는 스칼라 하나) 같은 부류의 질문, 즉 그 체크포인트가 무시되고 있는 short 리스트를 선언하는지 감사해 볼 가치가 있다.
3. Phi-3-small(`phi3small`)과 Phi-3.5-MoE는 자체 어텐션 모듈을 가지며 명시적으로 범위 밖이었다. 둘 중 하나라도 `longrope`를 담고 있다면 같은 처리가 필요하다.
4. `mlxcel_core::prefill_span`은 범용이다. RoPE 테이블, 마스크 형태, 스케일이 전체 시퀀스 길이에 의존하는 향후 계열은 그대로 읽어 쓸 수 있다. 현재 다중 호출 prefill 드라이버는 알림 지점 두 곳이 전부이며, 나중에 세 번째가 생기면 반드시 알려야 한다는 점을 문서 주석에 적어 두었다.

### 더 넓은 교훈

이 부류의 결함은 "출력이 글처럼 읽히는가"를 묻는 모든 테스트를 통과한다. 변경 전 코드는 이 트리에서 보통 가장 강력한 근거인 업스트림 기준 구현과 정확히 일치했고, 그럼에도 틀렸다. 그 기준 구현 자체가 이 기능에 대해서는 불완전하기 때문이다. 두 구현을 가르는 성질은 유창함도 아니고 mlx-lm 대비 RMS도 아니다. 체크포인트가 학습되고 배포될 때 함께 쓰인 구현과의 토큰 일치다. 오라클을 고르는 일은 수정에 앞선 단계가 아니라 수정의 일부다.
