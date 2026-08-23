# 기술 보고서: PR #1389 - dynamic NTK 스케일링에서 RoPE 위치가 2배 되던 문제 수정

**날짜**: 2026-08-23
**작성자**: 신정규
**상태**: 완료
**언어/기술**: Rust, Python (커밋된 오라클)
**위험도**: 중간 (InternLM3 디코드 수치가 모든 문맥 길이에서 바뀌고, InternLM2는 파싱 단계에서 버려지던 블록을 새로 받는다)

---

## 요약

`internlm3`은 `rope_scaling`이 없거나 `rope_type`이 `dynamic`이면 `fast_rope`에 `scale = 2.0`을 넘겼다. `rope_scale()`이 `"linear"`만 처리하고 `.unwrap_or(2.0)`으로 끝나기 때문이다. 검증 체크포인트가 `{"factor": 6.0, "rope_type": "dynamic"}`을 싣고 있으므로, **모든 쿼리와 키가 실제 위치의 2배로 회전**했다. `max_position_embeddings` 이후만이 아니라 **모든 길이에서** 그랬다. 올바른 dynamic NTK는 위치를 절대 스케일하지 않고, 시퀀스가 `max_position_embeddings`를 넘을 때 회전 base만 키운다.

수정 자체는 작다. 이 변경이 보고서 값어치를 갖는 이유는 **교체해야 했던 레퍼런스**와, 그 레퍼런스를 검증하다 나온 것들이다.

**기존 파리티 테스트가 결함을 고정하고 있었다.** `INTERNLM3_REF_OUT`은 같은 잘못된 `scale = 2.0`을 적용한 레퍼런스에서 떠 왔고, `docs/supported-models.md`는 그 레퍼런스 기준 토큰 일치를 광고하고 있었다. 초록 체크는 정확성의 약한 증거가 아니라 **합격 배지를 단 결함의 적극적 증거**였고, 포팅이 들어온 이래 계속 그랬다.

**그리고 그것을 고치는 뻔한 방법이 같은 함정을 다시 놓을 뻔했다.** 오케스트레이터 지시는 "mlx-lm 오라클에서 다시 핀하고, mlxcel 자기 출력에서는 절대 안 된다"였다. **그 지시가 틀렸다.** mlx-lm 0.31.3이 동일한 결함을 갖고 있어, 스톡 mlx-lm이 옛 결함 핀을 24/24로 정확히 재현한다. 레퍼런스는 두 구현 어느 쪽도 오염시킬 수 없는 곳에서 와야 했다.

## 1. 문제 정의

### 1.1 배경

`src/models/internlm3.rs`의 위치 스케일 계산:

```rust
self.rope_scaling.as_ref()
    .and_then(|s| if s.rope_type == "linear" { Some(1.0 / s.factor) } else { None })
    .unwrap_or(2.0)
```

블록이 없어도, `rope_type`이 `"dynamic"`이어도 `2.0`으로 떨어진다. `fast_rope`는 회전각 계산 전에 모든 위치에 `scale`을 곱하므로, 절대 위치 `p`의 토큰이 `2p`에 앉은 것처럼 회전했다. 두 토큰 사이 상대각도 2배가 되므로 어텐션이 학습 때와 **어디서나** 달랐다. 이 방식이 존재하는 장문맥 구간만이 아니다.

`compute_dynamic_ntk_base`는 이미 맞았고, 다시 쓰지 않고 옮겼다.

`internlm2`는 더 조용한 쪽으로 나빴다. `ModelArgs`가 `rope_scaling`을 **선언조차 안 해서**, 체크포인트가 싣고 있는 `{"type": "dynamic", "factor": 2.0}`이 역직렬화에서 버려졌다. internlm3의 '파싱은 되나 안 읽힘'보다 한 단계 앞이다.

### 1.2 기존 문제

결함이 **통과하는 테스트 위에 얹혀** 있었다. `tests/causal_prefill_greedy_parity.rs`가 56-id 프롬프트에 대해 24개 레퍼런스 id를 고정했는데, 스톡 mlx-lm에서 뜬 것이고, mlxcel이 mlx-lm의 식을 그대로 포팅했으므로 둘이 일치했다. **같은 실수의 두 구현이 일치하는 것은 상호 검증이 아니다.**

### 1.3 위험 평가

중간. 파급은 두 계열뿐이다. `internlm3::ModelArgs`와 `internlm2::ModelArgs`는 각각 정확히 한 곳(`src/model_metadata.rs:187`, `:188`)에서만 만들어지고, 둘 다 최상위 `config.json`의 `ConfigBacked` 경로다. 어떤 VLM 로더도 임의의 `text_config`를 이쪽으로 밀어 넣지 않고, 두 계열 다 텐서 병렬 대상이 아니다. 형제 이슈 #1355가 warn-and-continue로 물러서야 했던 자리에서 여기는 **미구현 방식을 로드 에러로 만들어도 안전한** 이유가 그것이다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| `src/models/dynamic_ntk_rope.rs` (신규) | `DynamicNtkRopeMode`, `DynamicNtkRope`, `from_scaling`, `scale()`, `base_for()`, `apply()`. `rope_utils::{RopeScalingSpec, is_usable_scalar, printable_label}` 위에 구축 |
| `src/models/internlm3.rs` | 사설 `RopeScaling`과 `unwrap_or(2.0)` 삭제, 헬퍼 사용 |
| `src/models/internlm2.rs` | `rope_scaling` 선언, 헬퍼 사용, `rope_traditional`을 하드코딩 대신 존중 |
| `tests/causal_prefill_greedy_parity.rs` | `INTERNLM3_REF_OUT` 재핀, 출처와 재생성 방법을 doc에 기록 |
| `scripts/tools/internlm_rope_oracle.py` (신규) | 오라클 커밋, `--mode stock` / `--mode fixed` |
| `docs/supported-models.md` | 결함 레퍼런스 기준 토큰 일치 주장 제거 |

## 3. 기술적 선택과 그 이유

### 3.1 오라클이 같은 결함을 갖고 있어, 레퍼런스를 두 구현 밖에서 만들었다

mlx-lm 0.31.3 `internlm3.py`에 독립적인 문제가 **둘** 있다. 동작으로 추론한 게 아니라 설치된 소스를 열어 확인했다.

1. 104-108행이 `rope_scale = 1 / factor if rope_type == "linear" else 2.0`을 계산해 `mx.fast.rope(..., scale=self.scale, ...)`로 넘긴다. 즉 `dynamic`은 위치 스케일 2.0을 받는다. 그리고 70-72행이 **같은 2.0을 NTK base 공식의 factor로 재사용**한다.
2. `DynamicNTKScalingRoPE.__call__`이 `seq_len = x.shape[1] + offset`으로 읽는다. `transpose(0, 2, 1, 3)` 후 텐서가 `[B, n_heads, L, head_dim]`이므로 `shape[1]`은 **head 개수**이지 시퀀스 길이가 아니다. mlxcel은 원래 맞게 읽고 있었다.

mlxcel이 첫 번째 식을 그대로 포팅했으므로 스톡 mlx-lm과 변경 전 mlxcel이 정확히 일치했고, 그래서 게이트가 초록이었다.

그래서 핀은 mlx-lm에서 **레이어별 rope 모듈만** 체크포인트 자체 remote code가 구현하는 스케줄(`modeling_internlm3.py` → transformers `_compute_dynamic_ntk_parameters`)로 교체해서 떴다. 가중치 로딩, 역양자화, 어텐션, 샘플링은 mlx-lm 그대로다. 이 방향이 옳다는 독립 방증 둘: 체크포인트 자신의 remote code, 그리고 mlxcel의 XLA emitter가 이미 `dynamic`을 `RopeScaling::Plain`으로 매핑하며 "identity within the original context"라 주석해 둔 것. **MLX 경로가 이상치였지 다수가 아니었다.**

### 3.2 오라클을 커밋했다. 설명된 레퍼런스는 검사 가능한 레퍼런스가 아니다

doc 주석이 재구성 가능할 만큼 설명은 잘 돼 있었다. 그걸로는 부족하고, **이 PR 자체가 그 증거다.** 이전 핀도 아마 그것을 뜬 사람에게는 재구성 가능했을 텐데, 몇 달간 결함을 담고 있었다.

`scripts/tools/internlm_rope_oracle.py`가 핀을 **재생성 가능**하게 만든다. `--mode stock`이 이 상수가 담고 있던 결함 id를, `--mode fixed`가 지금 담은 id를 만들어 내므로 차이가 주장이 아니라 **실증**이다. 손으로 재구성할 때 가장 놓치기 쉬운 것, 위의 `x.shape[1]` 대 `x.shape[-2]` 축 혼동도 doc에 박았다. 전에는 산문에만 있었다.

### 3.3 internlm2는 미루지 않고 배선했다

이슈는 헬퍼를 "internlm2가 재사용할" 것이라 적었는데, 그건 일을 축소해 말한 것이다. internlm2는 파싱된 블록을 무시한 게 아니라 **애초에 파싱하지 않았다.**

자기 설정을 계속 버리는 계열을 위해 "모양만 맞춘" 헬퍼를 남기는 것은 미룰 값어치가 없었다. `max_position_embeddings` 안에서는 전부 비트 동일하고, 이건 주장이 아니라 확인이다. 위치 스케일이 이미 올바른 하드코딩 `1.0`이었고 `Dynamic`이 `1.0`을 돌려주며, `factor = 2.0`에서 clamp 아래 base가 정확히 `rope_theta`라, `models/internlm2-7b-4bit`이 변경 전후 **바이트 동일**하게 생성한다.

`rope_traditional`은 이제 하드코딩 대신 존중한다. 출하된 두 체크포인트 다 이 키를 선언하지 않아 `#[serde(default)]`가 이전 하드코딩 값 `false`를 주므로 무해하다.

### 3.4 여기서는 로드 에러, 공유 Llama 경로에서는 경고

이 두 계열에서 미구현 방식은 named 로드 에러다. #1355는 그렇게 못 했다. VLM 로더 8개가 임의의 `text_config`를 `llama3::ModelArgs`로 밀어 넣고 그중 하나(`models/internvl3-1b`)가 `dynamic`을 선언해서, 에러로 만들면 **잘 돌던 모델이 안 열렸을 것**이다.

여기엔 그 제약이 없고, 가정이 아니라 검증했다. 두 InternLM args 타입이 각각 정확히 한 곳에서만 생성되고, InternVL은 InternLM이 아니라 `llama3::ModelArgs`로 파싱하며, `runtime_kind_for`에 InternLM arm이 없어 텐서 병렬 경로는 테이블 완전성 arm일 뿐이다.

## 4. 검증

### 4.1 변경 전 바이너리를 보존한 오라클 대조

| 체크포인트 / 프롬프트 | 변경 전 | 변경 후 | 변경 전 첫 불일치 지점의 오라클 마진 |
|---|---|---|---|
| internlm3, 56-id 게이트 프롬프트 | 2/24 | **24/24** | 0.31 |
| internlm3, 670-id 프롬프트 | 9/32 | **32/32** | 0.48 |
| internlm3, 56-id, `max_pos`를 32로 강제 | 2/24 | **24/24** | 0.39 |
| internlm2, 614-id 프롬프트 | 변화 없음 | **24/24** | 해당 없음 |

변경 전 mlxcel이 스톡 mlx-lm을 24/24, 32/32로 정확히 재현한다. **포팅은 충실했고, 충실하게 옮긴 대상이 틀렸다.**

`max_position_embeddings`를 32로 강제하면(가중치 심링크, 설정 복사) 32768토큰 프롬프트 없이도 base 재조정 분기에 도달한다. 이 하드웨어에서 장문맥 경로를 테스트 가능하게 만든 것이 그 방법이다.

모든 프로브가 두 바이너리를 갈라놓고, 모든 판별 단계의 마진이 건강하다. **양쪽 다 필요하다.** 버그 있는 바이너리도 통과하는 프로브는 아무것도 증명하지 않고, tie 근처의 분기도 아무것도 증명하지 않는다. 이 배치에서 둘 다 한 번씩 나온 뒤에야 둘 다 확인하게 됐다.

CLI 출력 차이는 도구 없이도 보인다. 변경 전은 `"centruty… chnage… indusstry… repplacd"`에 의미까지 뒤집어 공장이 가내수공업으로 대체됐다고 했다. 변경 후는 깨끗하고 정확하다.

### 4.2 공식

worked value 넷이 f32·f64로 독립 재현된다. `seq_len` 100과 32768은 둘 다 변화 없는 `5e7`, 40000은 117777118.66, 65536은 360979300.43. clamp 방향은 `max(seq_len, max_pos)`이고, `scale()`은 `Default`·`Dynamic`에 `1.0`, `Linear`에만 `1.0/factor`를 준다. `factor`는 공유 `is_usable_scalar`로 양수·유한 검사를 거치므로 `factor: 0`은 퇴화 base가 아니라 로드 에러다.

### 4.3 멀티턴 게이트, 그리고 판정 불가였던 프로브

소유자가 런 도중 머지 조건을 추가했다. **실모델 단일턴 + 멀티턴 통과**이고, self-hosted 러너 메인테넌스 동안 GitHub CI는 생략한다.

멀티턴 하네스는 3턴 대화를 프롬프트 캐시 ON과 `--no-prompt-cache`로 각각 돌려 비교한다. 첫 형태(transcript 바이트 동일 요구)에서 `models/internlm3-8b-4bit`이 변경 후 turn 2에서 실패했다.

**캐시 회귀가 아니다.** 캐시가 실제로 adopt된 상태(`hits=1`)에서 재보니 두 경로가 top-5의 모든 토큰에서 bf16 1 ULP 이내로 일치하고, 캐시 ON 쪽이 **정확한 동점**에 앉는다. `Rep`과 `Note`가 둘 다 -1.640625라 tie-break가 결정한다. 캐시 OFF는 0.0078125로 갈리는데, 그 크기에서 정확히 1 ULP다. 변경 전 바이너리가 같은 단언을 통과한 이유는 그 지점 마진이 1.28, 즉 jitter의 약 160배였기 때문이다.

진짜 캐시 결함과의 차이는 미묘하지 않다. 아직 미수정인 #1346의 영향을 받는 `models/gemma3-1b-4bit`에서 같은 측정을 하면 최상위 토큰이 **9.69**만큼 움직이고, 한쪽에서 logprob 0.0000인 `'Not'`이 다른 쪽 top-5에는 아예 없다.

하네스는 이제 **생성된 모든 토큰을 비교**하고, 분기 지점 **자신의** top-2 마진이 jitter 바닥 0.05 아래일 때만 면제한다. bf16 1 ULP의 약 6배, 진짜 결함의 약 200분의 1이다. 이 규칙에서 `llama-3.1-8b-4bit`은 전 턴 동일로 통과, `internlm3-8b-4bit`은 동점을 명시하며 통과, `gemma3-1b-4bit`은 #1346이 들어올 때까지 계속 실패한다.

### 4.4 게이트

`cargo test --workspace --profile test-fast --features metal,accelerate`: 8362 통과, 0 실패. clippy(`--workspace --all-targets -D warnings`)와 fmt 깨끗. 이번 런은 소유자 지시로 GitHub CI 생략.

## 5. 리뷰에서 나온 지적

MEDIUM 초과 없음. 리뷰어가 오라클 산출물을 직접 찾아 커밋된 id와 바이트 일치함을, 그 아티팩트가 커밋보다 앞섬을, 오라클이 mlxcel에서 아무것도 읽지 않음을 확인해 **핀 출처를 독립 검증**했다. MEDIUM 3건 반영: 오라클 커밋, `rope_utils.rs`의 `Used by:` 로스터 3곳에 새 소비자 둘 추가, 위에 적은 과도한 하네스 완화 교정.

LOW 4건은 기록만 하고 안 고쳤다. internlm2가 이전에 무시하던 비-객체 `rope_scaling` 형태에서 이제 하드 실패한다(출하된 설정 중 그런 것 없음), `base_for`가 forward마다 블록당 두 번 평가된다(약 10ms 스텝 대비 수십 나노초), 로드 에러 라벨이 체크포인트 디렉터리가 아니라 아키텍처 이름을 부른다(dedup 경고와 달리 하드 에러라 덜 중요), `dims`와 `rope_theta`가 `factor`처럼 검사되지 않는다(기존 코드와 공유하는 선재 문제).

## 6. 검증하지 못한 것

32768토큰을 실제로 넘는 프롬프트는 돌리지 않았다. `max_position_embeddings` 축소가 같은 코드 경로에 훨씬 적은 시간으로 도달하고 재조정된 base를 f64 레퍼런스로 따로 고정했지만, 전체 길이 사례 자체는 미검증이다.

`max_pos`를 넘는 chunked prefill은 프롬프트당 한 번이 아니라 청크마다 base를 계산한다. mlx-lm과 같은 근사이고 이번에 바꾸지 않았다.

별개로 `models/internlm3-8b-4bit`은 서버 채팅에서 `<|im_end|>`에 멈추지 않는다(`eos_token_ids`가 `[2]`를 반환). 선재 문제이고 두 바이너리에서 동일하다.

## 7. 학습 포인트

- **통과하는 파리티 테스트는 그 레퍼런스의 출처만큼만 좋다.** 레퍼런스를 시험 대상과 코드를 공유하는 구현에서 떴다면, 일치는 증거가 아니라 동어반복이다.
- **오라클이 자기가 검출하려는 결함을 갖고 있을 수 있다.** 레퍼런스 구현을 믿기 전에 그것이 검증 대상과 독립인지부터 본다. 여기서는 한쪽이 다른 쪽에서 포팅됐기 때문에 둘이 일치했다.
- **지시도 이슈 본문처럼 틀릴 수 있다.** "오라클에서 다시 핀하고 우리 출력에서는 절대 안 된다"는 의도는 옳았고 사실은 틀렸으며, 문자 그대로 따랐으면 버그를 다시 새겼다.
- **고정된 레퍼런스를 만드는 도구를 함께 커밋한다.** 설명은 됐는데 재생성이 안 되는 레퍼런스야말로 이 변경이 교체하려던 바로 그 물건이다.
- **차분 게이트에는 분기 지점에 국한된 jitter 바닥이 필요하다.** 한 단계가 동전던지기였다는 이유로 비교 전체를 면제하면, 거짓 경보를 사각지대와 맞바꾸는 것이다.
