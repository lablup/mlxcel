# 기술 보고서: PR #1385 - 공유 Llama 경로에 llama3 rope_scaling 적용

**날짜**: 2026-08-23
**작성자**: 신정규
**상태**: 완료
**언어/기술**: Rust (MLX FFI), Metal (fused 런처 게이팅)
**위험도**: 높음 (하나의 어텐션 구현을 공유하는 Llama 3.x, Qwen2, VLM 8계열의 디코드 수치가 바뀐다)

---

## 요약

Llama 3.1·3.2·3.3 체크포인트는 모두 `rope_type: "llama3"` 블록을 선언한다. 공유 Llama 어텐션은 그 블록을 구조체로 파싱해 놓고 **필드를 하나도 읽지 않았다**. 그래서 이 모델들은 평범한 `base^(-2i/d)` 주파수로 회전했다. 짧은 프롬프트가 이를 가렸는데, 스케일된 표와 안 된 표가 낮은 위치에서는 거의 같기 때문이다. 수천 토큰을 넘어서면 스케일 안 된 저주파 대역이 최대 8배(Llama 3.2 1B/3B는 32배) 빠르게 회전하고, 그 구간이 바로 `llama3` 방식이 존재하는 이유다.

이 변경에서 기억할 만한 것 셋은 전부 이슈에 없던 내용이다.

첫째, 이슈가 밝힌 범위가 자릿수 단위로 틀렸다. `llama3::ModelArgs`는 Llama만의 것이 아니고 Qwen2와 VLM 로더 8개가 이 타입으로 텍스트 디코더를 만든다. 둘째, **이슈 자신의 처방 두 가지가 각각 회귀를 낳았을 것**이다. 제안된 `#[serde(alias)]` 수정은 두 키를 모두 쓰는 로컬 체크포인트 5개에서 하드 에러가 되고, 수용 기준 "unsupported types are named load errors"는 InternVL3를 아예 못 열게 만든다. 셋째, 처음 구현한 수정 자체가 **이슈가 없애려던 바로 그 부류의 조용한 NaN 경로**를 새로 만들었고, 독립적인 리뷰 둘이 따로 찾아냈다.

부수적으로 무관한 미보고 실사용 결함을 하나 고쳤다. `deepseek-coder-1.3b-4bit`가 6토큰 프롬프트에서 `):):):):` 퇴화 출력을 내고 있었는데, 그 모델의 `linear` 스케일링도 함께 무시되고 있었기 때문이다.

## 1. 문제 정의

### 1.1 배경

`src/models/llama3.rs`는 `pub rope_scaling: Option<RopeScaling>`를 선언하고 정확히 그 한 줄에서만 이름을 언급했다. 소비하는 곳이 없다. 더 나쁘게도 `RopeScaling`의 `rope_type` 필드에 `#[serde(rename = "type")]`가 붙어 있어, `rope_type`만 쓰는 설정(모든 Llama 3.x가 그렇다)에서는 **파싱된 구조체에서조차 `None`** 이었다. 블록이 이중으로 죽어 있었다.

`src/models/mllama/config.rs`의 `to_llama3_args`는 고정 키 목록으로 텍스트 args를 재조립하는데 `rope_scaling` 항목이 없어, 공유 경로가 그 블록을 읽었더라도 Llama 3.2 Vision 디코더는 받을 수 없었다.

### 1.2 기존 문제

결함 부류가 위험한 쪽이다. **파싱은 되는데 적용은 안 되는** 종류다. 에러도 경고도 없고 출력이 깨지지도 않는다. 모델은 어느 길이에서든 유창한 텍스트를 내고, 다만 긴 문맥에서 성능이 떨어지는데 그게 버그가 아니라 모델이 원래 약한 것처럼 보인다. `docs/supported-models.md`는 반대편에서 이 구멍을 이미 적어 두고 있었다. TeleChat3 항목에 `llama3::ModelArgs`가 "아무도 읽지 않는 `rope_scaling` 필드를 선언한다"고 쓰여 있었고, 그 문장이 그대로 방치돼 있었다.

### 1.3 위험 평가

높고, 위험은 들어오는 쪽에 있다. 어텐션 구현 하나가 Llama 3.x, Qwen2, Pixtral, LLaVA, SmolVLM, Idefics2, Idefics3, InternVL, FastVLM, LocateAnything를 떠받친다. 새 표의 오류는 그 전부의 오류이고, 새로 강화한 엄격함은 그 전부의 로드 실패다.

## 2. 변경 요약

26개 파일, 약 2060줄 추가. 핵심:

| 영역 | 변경 |
|---|---|
| `src/models/rope_utils.rs` (신규) | 공유 `rope_scaling` 리더와 `llama3_rope_freqs`. mlx-lm `initialize_rope`/`Llama3RoPE` 이식. `serde_json::Map`으로 읽고 모든 스칼라를 검증하며, 미구현 방식은 체크포인트당 한 번 경고 |
| `src/models/llama3.rs` | `Attention`이 `rope_scale`·`rope_freqs` 보유. 그래프·배치 경로가 각각 헬퍼 하나를 거침. fused 런처 3개 게이팅. 진단용 `#[serde(skip)] checkpoint_label` |
| `src/lib/mlxcel-core/src/lib.rs` | `fast_rope_batched_with_freqs`(기존 `fast_rope_batched` 미러) |
| `src/models/mllama/config.rs`, `text.rs` | `rope_scaling` 전달, 레이어 루프 위로 resolve 호이스팅 |
| `src/models/apertus.rs` | 사설 복사본 대신 공유 함수 호출 |
| 로더 11개, 파이프라인 실행기, TP 런타임 | `checkpoint_label` 채움 |

## 3. 기술적 선택과 그 이유

### 3.1 이슈가 제안한 serde 수정은 InternVL3를 깨뜨렸을 것이다

이슈는 두 철자를 모두 읽도록 `#[serde(alias = "rope_type")]`를 붙이라고 했다. serde는 필드마다 `Option` 하나를 만들고 **어느 철자가 값을 만들었든** 두 번째 쓰기를 거절하므로, 두 키를 **모두** 가진 설정은 하드 `duplicate field` 파싱 에러가 된다.

로컬 체크포인트 5개가 둘 다 갖고 있다. `internvl3-1b`, `apertus-8b-instruct-2509-4bit`, `afm-4.5b`, `paddleocr-vl-bfloat16`, `telechat3-36b-thinking-4bit`. 그 '최소 수정'은 조용한 무동작을 다섯 모델의 로드 실패로 바꿨을 것이다.

그래서 리더는 `serde_json::Map`을 훑어 `type` 다음 `rope_type` 순으로 찾는다. 상류 `initialize_rope`와 같은 순서다. 결과적으로 이 리더는 대체한 파생 구조체보다 **엄격하게 더 관대하다**. 리스트형 `factor`(longrope), 실수형 `original_max_position_embeddings`, 비문자열 `type`, 중복 키가 전부 예전에는 하드 로드 에러였는데 이제 파싱된다.

이 코드베이스에서 serde alias가 중복 필드 함정을 만든 것은 두 번째다. 두 철자가 공존할 수 있는 필드의 `#[serde(alias)]`는 편의 기능이 아니라 **결함 패턴**으로 취급할 만하다.

### 3.2 미구현 방식은 로드 실패가 아니라 경고 후 진행

이슈의 수용 기준은 `yarn`·`dynamic`·`longrope`를 이 경로에서 명명된 로드 에러로 만들라고 요구했다. 그대로 구현하면 `models/internvl3-1b`가 안 열린다. 그 모델의 `text_config`가 `rope_type: "dynamic"`에 텍스트 아키텍처 `Qwen2ForCausalLM`이고, `vlm_internvl.rs`가 `text_config` 전체를 이 args로 넘기기 때문이다.

교환 비용이 비대칭이다. 오늘 열리면서 긴 문맥에서 미묘하게 틀린 모델은 맞는 모델보다 나쁘지만, **안 열리는 모델보다는 훨씬 낫다**. 그래서 미구현 방식은 체크포인트와 방식 이름을 담은 경고를 한 번 내고 평범한 표로 디코드한다. 오늘 동작 그대로다.

이것은 실수가 아니라 문서화된 의도적 이탈이고 PR 본문에 명시했다. `dynamic`은 #1324의 공유 DynamicNTK 헬퍼가 들어오면 제대로 구현할 수 있다.

### 3.3 첫 구현이 새로운 조용한 NaN 경로를 만들었다

독립적인 리뷰 둘이 여기로 수렴했다. 발견이 가질 수 있는 가장 강한 신호다.

`linear` arm은 `factor`의 양수·유한성을 검사하고 경고와 함께 폴백했다. 다섯 줄 아래 `llama3` arm은 `spec.factor.unwrap_or(1.0)`을 대역 산술에 그대로 먹였다. Llama 3.1 기하로 시뮬레이션한 결과:

- `"factor": 0`이면 64개 중 35개가 정확히 `0.0`. MLX fast rope는 위치를 표 값으로 나누므로 `reciprocal(0)`은 `inf`이고, **모든 토큰의 모든 로짓이 NaN**이 된다. 아무것도 던지지 않고, `sampling.rs`가 `partial_cmp(..).unwrap_or(Equal)`로 비교하므로 패닉조차 없다. 그냥 쓰레기를 낸다.
- factor가 없거나 JSON 문자열이면 `1.0`으로 기본값이 잡혀 평범한 표와 비트 단위로 같은 표가 나온다. 이 PR이 없애려는 바로 그 조용한 무동작이다. 옛 파생 구조체는 최소한 문자열 factor를 로드 에러로는 만들었다.
- `1e39`는 f64→f32 캐스트에서 `inf`로 포화하고, `-8`은 저대역의 회전 방향을 뒤집는다. 둘 다 유한해 보이고 유창하고 틀렸다.

이제 모든 스칼라를 검사하고, 걸러진 블록은 한 번 경고한 뒤 평범한 표로 디코드한다.

**의도적으로 넣지 않은 가드가 하나 있다.** `low_freq_factor == high_freq_factor`는 smooth 대역 분모 `(hf - lf)`의 0 나눗셈처럼 보이고, 이를 걸러 내는 것이 뻔한 방어적 조치다. 그렇게 했으면 **Llama 4 Scout가 깨졌을 것이다**. 그 모델이 정확히 그 값을 쓴다. 두 값이 같으면 보간 자체가 도달 불가다. 중간 대역이 `wavelen > L/hf && wavelen < L/lf`인데 두 인자가 같으면 경계가 동일해져 논리곱이 충족 불가능하다. 상류도 `mx.where`가 선택되지 않은 분기를 버리는 방식으로 같은 표에 이른다. 이 판단은 추상적으로 산술을 따진 게 아니라 **로컬 `config.json`을 전부 훑어보고** 내렸다.

### 3.4 안전 불변식이 어디에도 없던 `pub` 함수

`Attention::from_weights_with_rope`는 공개 함수인데 args와 표를 따로 받고, 둘이 같은 모델의 것임을 시그니처가 전혀 묶지 않는다. MLX는 표가 `[dims / 2]`이기를 요구하고 아니면 C++에서 throw하는데, `fast_rope_with_freqs`는 `Result`가 아니라 맨 `UniquePtr`로 브리지된다. 따라서 불일치는 로드 에러가 아니라 **첫 생성 토큰에서의 잡을 수 없는 `std::terminate`** 로 온다.

트리 안 호출자 넷은 모두 같은 args에서 뽑은 표를 넘기므로 실제로 발동한 적은 없다. 그래도 로드 에러로 만들었다. "지금 호출자들이 마침 옳게 한다"는 공개 시그니처가 기대어도 되는 속성이 아니고, 이 코드베이스는 [[feedback_mlx_precondition_terminate_at_first_inference]] 부류에 이미 물린 적이 있다.

### 3.5 이슈가 언급하지 않은 세 번째 fused 런처

이슈는 게이팅할 opt-in fused 런처 둘을 지목했다. 셋째가 있다. #905의 `forward_fused_rope_append`는 Metal에서 `rope_base`로 주파수를 유도하므로 표가 이를 우회하지만, **위치 스케일을 진짜 커널 파라미터로 받으면서**(`theta = rope_params[1] * pos * inv_freq`) 하드코딩된 `1.0`을 받고 있었다. 이제 진짜 `rope_scale`을 받으므로 `linear`이 fused 경로에서 조용히 무시되지 않는다.

이 런처의 환경변수도 우회 통지 목록에 들어가야 했고, 감지는 존재 여부가 아니라 `fused_rope_append_enabled()`를 거쳐야 했다. 그 변수는 3상태라 `=0`이 커널 손실로 보고되면 안 되기 때문이다.

## 4. 검증

### 4.1 자기 자신이 아니라 상류와 대조한 표 산술

주파수 표를 실제 `mlx_lm.models.rope_utils.Llama3RoPE._freqs`와 재계산 대조했다. Llama 3.1 기하(128 dims, base 5e5, factor 8)에서 최대 상대 편차 3.14e-7, Llama 3.2(64 dims, factor 32)에서 1.11e-7이고, 잔차는 스칼라 `powf` 대 벡터화 `pow` 차이다.

표의 **방향**은 읽지 않고 실측으로 확인했다. 뒤집혀 있어도 그럴듯해 보이는 것이 가장 유력한 오답 경로이기 때문이다. `mx.fast.rope(x, d, base=b)`가 `mx.fast.rope(x, d, base=None, freqs=b**(arange(0,d,2)/d))`와 비트 비교 가능한 반면, 뒤집은 쪽은 3.3만큼 차이 난다.

### 4.2 토큰 일치, 그리고 기록해 둘 아슬아슬한 지점

첫 롱프롬프트 실행이 64개 중 55개 일치였다. 유용했던 판단은 그것을 받아들이는 것도 디버깅을 시작하는 것도 아니라, **그 불일치 단계가 애초에 결정 가능한 자리인지** 묻는 것이었다. 오라클 자신의 top-2 logprob 마진을 재 보니 55단계가 중앙값 3.375에 대해 **정확히 0.00000 동점**이었다. 그 자리의 argmax는 모델이 아니라 부동소수점 노이즈가 정한다.

프롬프트를 needle-retrieval 과제로 재설계해 판별 단계의 마진을 중앙값 7.32812 대비 0.21875로 만들었다. 동일 가중치에서 mlx-lm 오라클 대조 결과:

| 체크포인트 | 프롬프트 | 결과 |
|---|---|---|
| `llama-3.1-8b-4bit` (factor 8) | 4484토큰, 64 생성 | 토큰 일치. 변경 전 바이너리는 5번째 생성 토큰에서 발산 |
| `llama-3.2-1b-4bit` (factor 32) | 4485토큰, 48 생성 | 토큰 일치. 변경 전은 `Q17`, 변경 후는 정답 `QX-7734` |

factor-32 사례는 오라클에 `rope_scaling: None`을 강제해 변경 전의 틀린 답을 재현하는 것으로 독립 확인했다. 상관이 아니라 **원인**이 표 누락임을 짚는 절차다.

두 체크포인트의 짧은 프롬프트는 변경 전 바이너리와 바이트 동일하다. 주장의 나머지 반이다. 수정은 긴 문맥 동작을 바꾸고 **그 외에는 아무것도 바꾸지 않아야** 한다.

### 4.3 공유 계열 전수 조사

`rope_scaling`을 선언한 로컬 체크포인트 38개를 `llama3::ModelArgs`에 도달하는지로 전부 매핑했다. 8개가 도달한다. 5개는 Llama 3.x 계열이고, 나머지 셋이 흥미롭다.

- **`deepseek-coder-1.3b-4bit`**: `linear` factor 4를 선언하는데 6토큰 프롬프트에서 `):):):):` 퇴화 출력을 내고 있었다. `linear`은 저대역만이 아니라 **모든 대역**을 스케일하므로, Llama 3.x 행들이 짧은 프롬프트에서 바이트 동일한 동안 이 모델만 바뀐 이유가 그것이다. 이제 레퍼런스와 일치한다. 미보고 실사용 결함이었다.
- **`idefics3-8b-llama3-4bit`**: `text_config`에 `llama3` factor 8을 선언하고 이제 적용한다. 이슈가 언급조차 안 한 VLM이다.
- **`internvl3-1b`**: `dynamic`을 선언하고, 열리고, 한 번 경고하고, 바이트 동일하게 디코드한다.

나머지는 전부 자기 모듈이 있어 이 args에 닿지 않는다. 블록이 있는 Qwen2와 없는 Qwen2 모두에서 무회귀를 확인했다.

서버 배치 디코드도 종단 간 실행했다. `--max-batch-size 4`에서 4484토큰 요청과 6토큰 요청을 동시에 넣으면 각각 단독 실행과 동일한 출력이 나온다. `fast_rope_batched_with_freqs`를 단위 테스트가 아니라 **실제 가중치 위에서** 검증하는 절차다.

### 4.4 게이트

`cargo test --workspace --profile test-fast --features metal,accelerate`: 8330 통과, 0 실패. clippy(`--workspace --all-targets -D warnings`)와 fmt 깨끗.

게이트 한 번이 `tests/hunyuan_vl_parity.rs`의 `text_only_forward_produces_finite_logits`에서 실패했다. **#997로 이미 추적 중인 알려진 플레이키**이고 그 이슈가 이 테스트를 이름으로 지목한다. 격리 실행에서는 통과하고, 리뷰 수정 커밋 이전 같은 브랜치에서도 통과했으며, 그 커밋은 hunyuan 경로가 닿는 파일을 하나도 건드리지 않았고, 워크스페이스 전체 부하에서만 실패한다. #997과 닫힌 #1023이 기술하는 signature 그대로다.

## 5. mlx-vlm과의 의도적 이탈

`CLAUDE.md`가 mlx-vlm 파리티를 핵심 원칙으로 명시하므로, 부수 효과로 두지 않고 결정으로 기록한다.

mlx-vlm의 `language.py`는 `idefics2`, `idefics3`(및 SmolVLM), `internvl_chat`에서 평범한 `nn.RoPE(dims, traditional, base)`를 만들어 블록을 통째로 버린다. `llava`, `llava_next`, `pixtral`은 `linear`만 지킬 만큼만 읽는다. `initialize_rope`를 부르는 것은 `mistral3`와 `mllama`뿐이다. 이 변경 후 mlxcel은 이들 전부에 표를 적용한다.

이탈은 의도적이고 **mlxcel 쪽이 옳다**. HuggingFace `Idefics3Model`이 `LlamaModel`을 감싸고 그것은 `rope_scaling`을 적용하므로, mlxcel이 이 체크포인트들이 학습된 동작에 맞고 mlx-vlm이 빠뜨린 구현이다. `models/idefics3-8b-llama3-4bit`에서 관측 가능하다.

## 6. 검증하지 못한 것

`mlx-community/Llama-3.2-11B-Vision-Instruct-4bit`가 로컬에 없어 이슈의 vision 토큰 일치 기준은 **미검증**이다. `mllama` 설정 전달은 단위 테스트로 덮었고 이 args로 가는 VLM 경로는 `idefics3-8b-llama3-4bit`가 실제 가중치로 실행하지만, 그것은 대리 커버리지이지 이슈가 요구한 기준 자체는 아니다.

파이프라인 stage executor와 텐서 병렬 런타임 변경은 단위 테스트만 거쳤다. 다중 노드 하드웨어가 없었다.

## 7. 학습 포인트

- 이슈가 밝힌 범위는 코드에 대한 **주장**이지 사실이 아니다. 공유 타입을 *누가 만드는지* grep한 것이 "Llama 3.x와 Vision" 변경을 코드 한 줄 쓰기 전에 10계열 변경으로 바꿨다.
- 이 이슈의 처방 둘이 각각 회귀를 낳았을 것이다. **수용 기준은 요구사항이고, 구현 계획과 제안된 한 줄 수정은 가설이다.** 그리고 가설은 자신이 만족시키려는 기준과 충돌할 수 있다.
- 방어적 가드가 회귀가 될 수 있다. `low_freq_factor == high_freq_factor` 차단은 뻔한 안전 조치이고 Llama 4 Scout를 깨뜨렸을 것이다. 산술을 따지는 것보다 **실제 체크포인트를 훑는 것**이 이겼다.
- 파리티 실행이 64분의 55로 나오면, 결함으로 보거나 노이즈로 넘기기 전에 **그 불일치 단계가 결정 가능한지부터 측정한다.** 정확한 logprob 동점이라면 다시 만들어야 할 것은 코드가 아니라 테스트다.
- 안전성이 "지금 호출자들이 마침 옳게 한다"에 기대는 `pub` 함수는 불변식을 코드로 박아야 한다. 위반이 FFI 경계를 넘어 에러를 프로세스 abort로 바꾸는 경우에는 특히 그렇다.
