# 기술 보고서: PR #1593 - feat(models): 텍스트 전용 Youtu-LLM을 전용 MLA 디코더로 라우팅

**작성일**: 2026-09-02
**작성자**: mlxcel maintainers
**리뷰어**: 구현 리뷰 사이클
**상태**: 완료 (유닛 커버리지 통과. mlx-lm 오라클과의 실제 체크포인트 greedy 비교는 머지 오케스트레이터가 수행, 부록 C 참조)
**언어**: Rust
**위험도**: Medium (특정 부류의 체크포인트에 대해 기존 라우트 하나가 바뀐다. 공유 MLA 어텐션에 필드가 하나 늘지만 기본값이 기존 호출자 전부를 그대로 보존한다)

---

## 요약

텐센트의 텍스트 전용 Youtu-LLM은 mlxcel이 이미 Youtu-VL의 텍스트 타워로 돌리고 있는 바로 그 디코더인데, 텍스트 전용 라우트가 없었다. 공개된 세 개의 `model_type` 라벨 중 둘은 탐지 단계에서 거부됐고, 나머지 하나는 엉뚱한 디코더로 갔다. PR #1593은 새 아키텍처 코드 없이 기존 `YoutuLanguageModel` 위에 `ModelType::YoutuLLM`을 얹고, `deepseek_v2` 라벨의 행선지를 라벨이 아니라 `architectures[0]`로 결정하게 만든다.

마지막 부분은 추가가 아니라 동작 변경이고, 이슈의 전제가 실측으로 틀렸기 때문에 필요했다. `mlx-community/Youtu-LLM-2B-4bit`는 mlx-lm이 로드할 수 있도록 자신을 `deepseek_v2`로 다시 라벨링한 변환본이다. 이슈는 그래서 이미 mlxcel의 DeepSeek-V2 라우트로 잘 돌고 있고 확인만 하면 된다고 봤다. 로컬 체크포인트에서 그 라우트는 로드도 되고 생성도 되지만 출력이 지리멸렬하고, 같은 가중치를 mlx-lm의 `deepseek_v2` 경로로 돌리면 멀쩡하다. 이 변환본은 여전히 `architectures: ["YoutuForCausalLM"]`과 벤더 자체 모듈을 가리키는 `auto_map`을 달고 있으므로, 재라벨링에도 살아남는 판별자는 아키텍처 문자열이다.

Youtu-VL 형제 체크포인트가 한 번도 건드리지 않는 설정 세 가지가 이 텍스트 라우트에서 처음으로 제대로 반영된다. `rope_interleave`가 가정되지 않고 실제 rope 호출까지 도달하고, `rope_scaling` 블록이 조용히 무시되는 대신 로드 시점에 검증되며, `q_lora_rank`가 선택 필드가 된다.

---

## 1. 문제 정의

### 1.1 배경

`src/models/youtu_vl_lm.rs`에는 Youtu 디코더 전체가 이미 구현돼 있다. DeepSeek-V2 레이아웃의 Multi-head Latent Attention(`q_a_proj` -> `q_a_layernorm` -> `q_b_proj`로 압축된 쿼리를 128 비위치 + 64 회전 차원으로 분할, `kv_a_proj_with_mqa`가 512차원 latent와 모든 헤드가 공유하는 64차원 회전 키를 내놓고, `kv_b_proj`는 로드 시점에 헤드별 `embed_q` / `unembed_out` 쌍으로 분해), dense SwiGLU MLP, tied word embedding. 독립 실행형 `YoutuLanguageModel::load`도 이미 있었고 `(Self, YoutuTextConfig)`를 반환했지만 테스트에서만 호출됐다.

빠져 있던 것은 순수하게 레지스트리 작업이었다. `ModelType`, `LoadedModel`, 메타데이터 레지스트리, 디렉터리 라우트, 그리고 탐지 arm. 이 변경 전까지 `grep YoutuLanguageModel src/`는 `src/loading/vlm_youtu_vl.rs`와 `src/vision/youtu_vl.rs`만 걸렸다.

### 1.2 라벨은 셋인데 하나는 거짓말이다

| 체크포인트 | `model_type` | `architectures[0]` | 변경 전 |
|---|---|---|---|
| `tencent/Youtu-LLM-2B` | `youtu` | `YoutuForCausalLM` | 거부: "Unsupported model type" |
| `mlx-community/Youtu-LLM-2B-mlx-4bit` | `youtu_llm` | `YoutuForCausalLM` | 거부 |
| `mlx-community/Youtu-LLM-2B-4bit` | `deepseek_v2` | `YoutuForCausalLM` | `DeepSeekV2`로 로드, 출력 붕괴 |

앞의 둘은 arm이 없는 문제다. 셋째가 흥미로운 쪽이고, 그 모델 카드 스스로 재라벨링을 인정한다. "Converted using deepseek_v2 architecture mapping (compatible MLA implementation)."

### 1.3 셋째 행이 오라우팅이라는 실측 근거

`models/mlx/youtu-llm-2b-4bit`, `--no-chat-template`, temperature 0 greedy 디코드:

| 프롬프트 | `main`의 mlxcel (DeepSeek-V2 라우트) | 같은 가중치의 mlx-lm `deepseek_v2` |
|---|---|---|
| "The Fibonacci sequence begins with" | "1 and 2, timing with the time of the day as a factor. Given the time of day in minutes since midnight" | "1,1, and each subsequent number is the sum of the two preceding numbers. The sequence is d..." |
| "In distributed systems, consensus protocols such as Raft" | "Okay, so the user" | ", Paxos, and Raft again, are used to achieve agreement among multiple nodes. In a distribu..." |

mlxcel 쪽 출력 둘 다 유한하고 문장으로서는 멀쩡한 영어다. 이건 크래시나 가중치 탐색 실패가 아니라 "로드는 되는데 수치가 틀린" 경로의 전형적 서명이다. 이슈의 수용 기준 "기존 DeepSeek-V2 라우트로 로드되는지 확인"은 로드 체크와 앞 몇 토큰 눈대중으로 통과했을 것이다. 게이트를 로드 단언이 아니라 교차 검증으로 쓰는 이유가 정확히 이것이다.

### 1.4 VLM 형제가 고정해 주지 못하는 것

`models/mlx/youtu-vl-4b-instruct`에는 `rope_scaling` 블록 자체가 없고, `rope_interleave: true`, `q_lora_rank: 1536`이다. 그래서 설정 세 가지가 텍스트 전용 체크포인트와 함께 처음으로 mlxcel에 도달하는데, 셋 다 로드 에러가 아니라 조용한 오답 부류였다.

- `rope_interleave`는 `YoutuTextConfig`로 파싱만 되고 읽히지 않았다. 공유 `DeepSeekV3Attention`은 두 `fast_rope` 호출에 리터럴 `true`를 넘겼다.
- `rope_scaling`은 캐리어 `DeepSeekV3Config`로 전달돼 `get_attention_scale`이 mscale 키를 읽지만, 주파수 테이블 자체는 평범한 `rope_theta`다. 실제로 보간하는 YaRN 블록은 조용히 버려졌을 것이다.
- `q_lora_rank`는 필수 `usize`였으므로 `null`이면 공유 어텐션이 이미 구현해 둔 직접 `q_proj` 분기를 고르는 대신 serde 에러로 로드 전체가 실패했다.

### 1.5 위험성

| 위험 | 영향 | 발생 가능성 |
|---|---|---|
| 진짜 DeepSeek-V2 체크포인트가 Youtu 디코더로 새는 경우 | 발생 시 High | 구조적으로 제거. 분기는 `architectures[0] == "YoutuForCausalLM"` 정확 일치이고, `DeepseekV2ForCausalLM`과 `architectures` 배열 부재 둘 다 테스트로 고정 |
| `rope_traditional` 필드가 DeepSeek-V3 / V3.2 / V4 / Kimi-VL 동작을 바꾸는 경우 | 발생 시 High | 제거. `from_weights`가 `true`로 설정하며 이는 두 `fast_rope` 호출이 쓰던 리터럴 그대로다. Youtu 디코더 레이어만 덮어쓴다 |
| 검증된 config 타입을 공유하게 되면서 Youtu-VL이 퇴행 | Medium | 회피. `validate_rope_scaling`은 텍스트 전용 `load`에서만 호출된다. VLM 로더는 손대지 않았고, 공개된 Youtu-VL 체크포인트 중 이 블록을 가진 것도 없다 |
| DeepSeek-V2 라우트를 틀리게 만든 원인을 새 라우트가 그대로 물려받는 경우 | High | 유닛 테스트로는 제거 불가. 공유 MLA 어텐션은 `deepseek_v2.rs`(prefill용 kv_b_proj 유지, YaRN rope)와 다른 구현(항상 흡수, plain rope)이고 Youtu-VL이 이미 돌리고 있지만, 증명은 오케스트레이터의 실제 체크포인트 실행뿐이다. 부록 C |
| rope_scaling 거부가 정당한 장문맥 변환본을 막는 경우 | Low | 의도적으로 수용. 지원하지 않는 스킴을 이름과 함께 알리는 로드 에러가 유창한 오답보다 낫다. 메시지가 어떤 키의 어떤 값 때문인지 밝힌다 |

---

## 2. 기술적 검토 사항

**보안.** 모든 라우트가 이미 파싱하는 체크포인트 `config.json` 외에 신뢰 불가 입력이 닿는 새 파싱 표면은 없다. 새로 읽는 값은 `architectures[0]` 하나이고, 이미 네 개의 탐지 규칙이 쓰는 기존 `first_architecture` 헬퍼를 경유한다. `validate_rope_scaling`은 선택 키 두 개를 읽어 에러 문자열로 포맷한다. `factor`는 기본값 1.0의 `f64`로 읽으므로 값이 숫자가 아니거나 없어도 가드가 오작동하지 않는다. 새 `parse_eos_token_ids`는 각 id를 `as` 캐스트가 아니라 `i32::try_from`으로 경계 검사하므로, 범위를 벗어난 `eos_token_id`는 래핑된 값이 아니라 빈 결과가 된다.

**성능.** `DeepSeekV3Attention`에 `bool` 하나가 늘었고, 레이어당 forward당 두 번 읽힌다. 상대는 이미 주파수 테이블을 만드는 호출이다. 디코드 경로 어디에도 할당이 추가되지 않는다. `validate_rope_scaling`과 `parse_eos_token_ids`는 로드 시 한 번 실행된다.

**정확성.** 하중을 받는 주장은 `from_weights`의 `rope_traditional: true`가 이전에 두 호출 지점이 하던 것과 정확히 같다는 것이고, 이는 diff를 읽으면 확인된다. 리터럴 `true`가 인자 위치에서 구조체 초기화로 옮겨졌을 뿐이며, `models::deepseek` 유닛 커버리지(인접 레지스트리 스위트 포함 197개)는 그대로다. 두 번째 주장, 즉 플래그가 파싱만 되는 게 아니라는 것은 눈으로 보는 대신 차분 테스트로 고정했다. 같은 합성 가중치로 `rope_interleave`만 뒤집어 만든 두 모델의 로짓이 달라야 한다. 필드 값만 단언하는 테스트는 플래그를 읽고 버리는 구현에도 통과한다.

---

## 3. 기술적 선택과 그 이유

### 3.1 `deepseek_v2` 라벨은 아키텍처 문자열이 결정하고, 그 arm에만 한정한다

대안은 구조적 판별자(MLA 기하, `rope_interleave` 존재 여부, `rope_theta` 크기), 전체 `model_type` 디스패치 앞의 사전 체크, 아니면 사용자에게 라벨을 고쳐 쓰라고 하는 것이었다.

구조적 판별자는 여기서 형태가 맞지 않는다. Youtu의 MLA 기하는 설계상 DeepSeek-V2-Lite와 가깝고, 그 근접성이야말로 mlx-lm에서 재라벨링이 통하는 이유다. 헤드 차원이나 `rope_theta`에 거는 임계값은 미래의 DeepSeek 변종이 넘어갈 수 있는 추측이다.

사전 체크는 규칙을 모든 라벨에 적용하는데, 이는 근거보다 넓다. Youtu 변환본이 빌려 쓴다고 알려진 라벨은 `deepseek_v2` 하나뿐이다. `deepseek_v2_or_youtu`라는 이름 붙은 헬퍼로 그 arm에만 한정하면 영향 범위가 라벨 하나로 묶이고, 근거가 그 arm을 다음에 읽을 사람 눈앞에 놓인다.

라벨을 고쳐 쓰라는 건 탐지 문제에 문서로 답하는 것이다. 공개된 체크포인트를 공개된 그대로는 못 쓰게 만드는데, 그게 바로 이 이슈가 없애려는 상태다.

음성 케이스가 양성만큼 중요하므로 `genuine_deepseek_v2_keeps_its_route`는 `DeepseekV2ForCausalLM`과 `architectures` 배열이 아예 없는 config 둘 다 덮는다. 오래된 MLX 변환본 상당수가 배열을 생략하는데, "없음"을 "DeepSeek-V2 아님"으로 취급하는 규칙이었다면 그들에게 조용한 퇴행이었을 것이다.

### 3.2 새 `DeepSeekV3Config` 키가 아니라, 보존적 기본값을 가진 필드

`rope_interleave`를 반영하려면 플래그가 `fast_rope`까지 가야 했고, 형태는 셋 중 하나였다.

`DeepSeekV3Config`에 `rope_interleave`를 추가하는 게 깔끔해 보이지만, 그 구조체는 Youtu 캐리어와 테스트 픽스처에서 리터럴로 만들어지므로 필드 하나 추가는 호출자 하나만 값을 바꾸는데 모든 리터럴을 기계적으로 고치는 일이 된다. Youtu 고유 개념을 DeepSeek config 타입에 심는 문제도 있다.

`from_weights`의 파라미터로 넘기는 건 네 계열이 호출하는 시그니처를 바꾼다.

선택한 형태는 어텐션 구조체의 `rope_traditional` 필드다. `from_weights`가 `true`로 설정하고, Youtu 디코더 레이어가 `with_rope_traditional`로 덮어쓴다. 기본값이 기존 동작을 보존한다는 것이 증명 가능한 가장 작은 변경이다. 두 rope 호출이 쓰던 리터럴 `true`가 초기화 자리로 옮겨졌을 뿐이다. 공개 필드를 직접 찌르는 대신 빌더를 쓴 것은 호출 지점에서 의도로 읽히고, 덮어쓰기가 `YoutuDecoderLayer::from_weights`의 한 줄로 끝나기 때문이다.

### 3.3 `rope_is_interleaved()`는 같은 스위치의 두 철자를 접는다

`YoutuTextConfig`는 `rope_traditional`(mlx-vlm 포트의 이름)과 `rope_interleave`(벤더의 이름)를 둘 다 갖고 있고 기본값은 모두 true다. 하나만 읽으면 디코더가 "어느 변환기가 만든 체크포인트인가"에 의존하게 된다. 접근자는 `self.rope_interleave && self.rope_traditional`를 반환하므로, 어느 쪽이든 꺼지면 half-split 형태가 선택되고, 둘 다 없는 체크포인트는 늘 하던 대로 동작한다. 공개 체크포인트 중 둘을 반대로 설정한 것은 없으므로 이 논리곱은 실제 충돌을 해소하는 게 아니라, 어느 키가 정본인지 추측하기를 거부하는 것이다.

### 3.4 YaRN factor가 1을 넘으면 무시하지 말고 거부한다

공개된 블록은 `{"type": "yarn", "factor": 1.0, "mscale_all_dim": 0}`이고 두 겹으로 항등이다. factor 1에서 YaRN의 외삽/내삽 주파수가 일치하고, 어텐션 mscale도 1이다. 벤더도 같다. mscale은 `mscale_all_dim`이 참일 때만 적용하고 `yarn_get_mscale`은 factor 1에서 1을 반환한다. mlxcel의 `get_attention_scale`은 이미 `mscale_all_dim > 0 && factor > 1`로 가드하므로 이 블록에 대해 스케일은 원래 맞았다. 이제 그 동치를 매번 다시 유도하는 대신 테스트가 고정한다.

처리되지 않던 쪽은 factor가 1을 넘는 경우로, 공유 MLA 어텐션이 구현하지 않은 주파수 보간을 요구한다. 무시하면 원래 컨텍스트 길이를 넘어가는 지점에서만 위치가 틀린 유창한 출력이 나오는데, 이보다 발견하기 어려운 실패 모드는 없다. 가드는 이를 스킴과 factor를 이름으로 밝히는 로드 에러로 바꾼다.

이 가드는 의도적으로 `YoutuLanguageModel::load`(텍스트 라우트)에서만 호출하고 VLM 로더에서는 부르지 않는다. VLM 로더는 사용자가 있는 출하 경로이고, 공개된 Youtu-VL 체크포인트 중 scaling 블록을 가진 것이 없다고는 해도 근거 없이 새 거부를 넣는 것은 동작 변경이다. VLM 라우트로의 확장은 조용히 처리하지 않고 후속 항목으로 남겼다.

### 3.5 어댑터 weight 라우트 없는 `Nonstandard` 디렉터리 라우트

`YoutuLanguageModel::load`는 `(Self, YoutuTextConfig)`를 반환하는데 이는 `loading::nonstandard::load_pair_from_dir`가 소비하는 형태 그대로다. 그래서 디렉터리 라우트는 `KimiLinear`, `Qwen35` 옆의 arm 하나다. 레지스트리 항목은 `weight: None, adapter: Some(...)`으로 `DiffusionGemma`, `Llada2Moe`와 같다. 이 필드는 `adapter_weight_route`로 LoRA 어댑터 로딩만 구동하고 일반 로드와는 무관하므로, 없다고 선언하는 편이 아무도 쓰지 않는 `SpecialWeightLoaderKind` arm을 배선하는 것보다 정직하다.

### 3.6 호출자가 디스크에서도 읽지만 로더에서 `eos_token_id`를 파싱한다

CLI(`commands/generate.rs`의 `crate::read_eos_token_ids`)와 서버(`model_worker.rs`) 모두 `generation_config.json`과 `tokenizer_config.json`에서 읽은 stop id를 병합하므로 모델 자신의 `eos_token_ids()`가 유일한 출처는 아니다. 그래도 채울 값어치가 있었다. `LanguageModel::eos_token_ids`는 다른 모든 계열이 지키는 계약이고, 그 두 경로 밖에서 생성된 모델은 그렇지 않으면 stop id를 하나도 보고하지 못한다. 네 줄이고, 미래의 호출자가 놀랄 경로 하나를 없앤다.

### 3.7 DeepSeek-V2 디코더는 쫓지 않는다

mlxcel의 DeepSeek-V2 라우트가 왜 Youtu 변환본을 잘못 처리하는지는 진짜 질문이고 여기서 답하지 않는다. 두 구현은 한 군데 이상 다르다. `deepseek_v2.rs`는 `kv_b_proj`를 로드한 채로 두고 prefill에서 up-project하며 흡수 디코드는 선택 경로이고, rope는 사전 계산 주파수를 쓰는 `YarnRoPE`로 돈다. 반면 공유 `DeepSeekV3Attention`은 항상 흡수되어 있고 plain base로 `fast_rope`를 부른다. 어느 쪽이든 원인일 수 있고, 좁히려면 mlx-lm 오라클에 대한 레이어별 트레이스가 필요한데 이는 DeepSeek 네 계열에 영향 범위를 갖는 별도 작업이다. Youtu 변환본을 그것을 위해 쓰인 디코더로 보내는 것은 어느 쪽이든 이 이슈의 옳은 수정이고, 저 조사와 독립적이다.

---

## 4. 구현 상세

### 4.1 탐지

`deepseek_v2_or_youtu(&v)`는 `src/models/detection.rs`의 `first_architecture` 옆에 놓이고 `"deepseek_v2"` arm에서 호출된다. `"youtu" | "youtu_llm"` arm은 `"youtu_vl"` 옆에 있다. 상수 `YOUTU_CAUSAL_LM_ARCHITECTURE`가 판별자를 한 곳에서 이름 짓는다.

### 4.2 rope 플래그가 config에서 커널까지 가는 경로

| 단계 | 코드 |
|---|---|
| 파싱 | `YoutuTextConfig::rope_interleave` / `rope_traditional`, 둘 다 `#[serde(default = "default_true")]` |
| 접기 | `YoutuTextConfig::rope_is_interleaved()` |
| 적용 | `YoutuDecoderLayer::from_weights`가 `.with_rope_traditional(config.rope_is_interleaved())` 호출 |
| 사용 | `DeepSeekV3Attention::forward`가 두 `fast_rope` 호출에 `self.rope_traditional` 전달 |

벤더 쪽 분기는 https://huggingface.co/tencent/Youtu-LLM-2B/blob/main/modeling_youtu.py 의 `YoutuMLAttention.forward`이고, `apply_rotary_pos_emb` 대신 `apply_rotary_pos_emb_interleave`(평소의 `rotate_half` 앞에 `view(..., d // 2, 2).transpose(4, 3)`)를 고른다. 그 reshape가 MLX에서 `traditional=True`로 불리는 것이다.

### 4.3 Config 타입 변경

`q_lora_rank: Option<usize>`에 `#[serde(default)]`. 값은 캐리어 `DeepSeekV3Config`로 그대로 전달되고, `from_weights`는 이미 `is_none()`으로 분기해 LoRA 체인 대신 `q_proj`를 고른다. `validate_rope_scaling()`은 `Result<(), String>`을 반환하며 가중치를 건드리기 전에 `load`에서 호출되므로, 거부되는 체크포인트는 전체 가중치 로드가 아니라 config 읽기 한 번의 비용만 든다.

### 4.4 등록

enum, `ALL_MODEL_TYPES`, `metadata()`(family `Specialized`, `FAMILY_ORDER`에 이미 존재), `all_variants!` 완전성 목록의 `ModelType::YoutuLLM`. `LoadedModel::YoutuLLM`과 그 `delegate_language_model!` arm. `model_metadata.rs`에 `kind: Text, directory: Nonstandard`. `loading/nonstandard.rs`에 `load_pair_from_dir` arm. 그리고 디스패치 테이블을 total로 유지하기 위한 `src/distributed/tensor_parallel/inference.rs`의 arch 문자열 arm(이 계열은 TP 대상이 아니며, 플래너의 지원 아키텍처 검증이 TP 로드 전에 이 문자열을 거부한다).

---

## 5. 학습 포인트

### 5.1 호환성 재라벨링은 가중치에 대한 서술이 아니라 로더에 대한 요청이다

`model_type`은 아키텍처의 서술이 아니라 "여기로 보내 달라"는 요청이다. 변환기가 어떤 런타임에서 로드되게 하려고 체크포인트를 재라벨링하면, 그 라벨은 그 런타임의 디코더에 대한 진술이 되고, 그것을 믿는 다른 런타임은 검증한 적 없는 주장을 물려받는다. 가중치가 무엇인지를 계속 말해 주는 필드는 `architectures[0]`이고, 그래서 재라벨링에도 살아남아 쓸 만한 판별자가 된다.

### 5.2 파싱만 되고 읽히지 않는 config 키는 없는 키보다 나쁘다

`rope_interleave`에는 필드도, 기본값도, 문서 주석도 있었고 소비자만 없었다. 키를 grep한 모든 리뷰가 그것을 찾았다. 네 계열에서 `rope_scaling`이 파싱되고 버려진 것(#1355)과 같은 형태다. 필드가 존재한다는 사실 자체가 "그래서 적용은 되나"를 아무도 묻지 못하게 만든다. 선언이 아니라 소비자를 grep하는 것이 이를 잡는 점검이다.

### 5.3 값을 단언하는 플래그 테스트는 플래그가 동작한다는 테스트가 아니다

`assert!(config.rope_is_interleaved())`는 플래그를 읽고 버리는 디코더에도 통과한다. 이빨이 있는 단언은 차분이다. 플래그를 뒤집어 모델을 두 번 만들고 로짓이 달라야 한다고 요구하는 것. 합성 가중치 맵 하나의 비용으로, 이 변경이 실제로 다루는 유일한 속성을 산다.

### 5.4 구현 불가능한 config 값은 로드에서 시끄럽게 실패해야 한다

이 디코더에는 YaRN 주파수 보간이 없다. `factor: 4.0`을 받아들이고 plain 테이블로 디코드하면 짧은 프롬프트에서는 맞고 원래 컨텍스트 길이를 넘어서만 틀리므로, 장문맥 parity 실행이 아니고서는 아무것도 잡지 못한다. 로드에서 거부하면 보이지 않는 부류가 키 이름이 적힌 메시지로 바뀐다. maskless-prefill과 `rope_scaling` 결함 부류가 가르친 것과 같은 교훈이다. 짧은 프롬프트는 위치 버그를 볼 수 없고, 유창한 출력은 증거가 아니다.

---

## 6. 추가 학습 리소스

### 핵심 키워드

- **MLA (Multi-head Latent Attention)**: 키와 값을 저랭크 latent에서 복원하는 어텐션. 캐시가 full K/V 대신 latent를 저장한다. DeepSeek-V2의 레이아웃이고 Youtu가 그대로 쓴다.
- **흡수(absorbed) MLA**: `kv_b_proj`를 쿼리/출력 프로젝션(`embed_q` / `unembed_out`)에 접어 넣어 디코드가 full key를 만들지 않게 하는 것. `DeepSeekV3Attention`은 항상 그렇게 하고, `deepseek_v2.rs`는 플래그가 있을 때만 한다.
- **Interleaved (traditional) RoPE**: half-split 쌍이 아니라 인접 차원 쌍을 회전. MLX에서는 `traditional=True`, PyTorch에서는 `rotate_half` 앞의 `view(d // 2, 2).transpose`로 만들어진다.
- **YaRN 항등 블록**: `factor`가 1인 `rope_scaling` 항목. 내삽/외삽 주파수 테이블이 일치하고 어텐션 mscale이 1이므로 블록이 없는 것과 같다.

### 관련 PR/이슈

- #1371: 이 이슈.
- #1355: 네 계열에서 `rope_scaling`이 파싱되고 적용되지 않은 건. 5.2와 같은 결함 부류.
- #958, #1026: 이 라우트가 재사용하는 MLA `kv_b_proj` 새니타이저의 공용 하드닝.

---

## 7. 변경 요약

### 통계

| 항목 | 값 |
|---|---|
| 변경 파일 | 13 |
| 추가 라인 | 617 |
| 삭제 라인 | 20 |
| 새 `ModelType` variant | 1 |
| 새 테스트 | 7 |

### 카테고리별 변경

- **탐지**: `src/models/detection.rs`, `src/models/detection_tests.rs`.
- **모델과 config**: `src/models/youtu_vl_lm.rs`, `src/models/youtu_vl_lm_config.rs`, `src/models/youtu_vl_lm_sanitize.rs`(주석만), `src/models/youtu_vl_lm_tests.rs`, `src/models/deepseek_v3.rs`.
- **레지스트리와 로딩**: `src/models/mod.rs`, `src/loaded_model.rs`, `src/model_metadata.rs`, `src/loading/nonstandard.rs`, `src/distributed/tensor_parallel/inference.rs`.
- **문서**: `docs/supported-models.md`.

---

## 8. 후속 조치

### 모니터링 필요

- 새 라우트가 실제 체크포인트에서 수치적으로 맞다는 주장은 전적으로 부록 C의 오케스트레이터 greedy 비교에 기댄다. 유닛 테스트로는 성립시킬 수 없다. 유닛 테스트가 쓰는 합성 가중치에는 비교할 기준이 없기 때문이다.

### 향후 개선

- 출하 중인 VLM 체크포인트에 scaling 블록이 없다는 확인이 되면 `validate_rope_scaling`을 Youtu-VL 로더로 확장. 그 라우트에서도 같은 조용한 오답 부류다.
- 공유 MLA 어텐션에 YaRN 주파수 보간을 구현하면 거부가 지원으로 바뀐다. factor 1 초과의 Youtu 또는 Kimi-VL 변환본이 나타날 때만 할 값어치가 있다.
- mlxcel의 DeepSeek-V2 라우트가 Youtu 형태의 변환본을 왜 잘못 처리하는지 근본 원인 규명(3.7). 이 변경과 독립적이지만, 답이 진짜 DeepSeek-V2 체크포인트에도 영향을 주는 결함일 수 있다.
- `tencent/Youtu-LLM-2B`(bf16, `model_type: youtu`)와 `mlx-community/Youtu-LLM-2B-mlx-4bit`(`model_type: youtu_llm`) 종단 검증. 둘 다 검증 호스트에 없고, 둘 다 유닛 테스트가 덮는 평범한 라벨 arm을 타므로, 새 코드 검증이라기보다 세 라벨 중 마지막을 닫는 일이다.

---

## 부록

### A. 테스트 결과

| 스위트 | 결과 |
|---|---|
| `--lib -- models::youtu_vl_lm models::detection_tests loading::nonstandard` | 60 passed |
| `--lib -- models::deepseek models::metadata_tests model_metadata_tests loading::tests` | 197 passed |
| `--bin mlxcel -- family_order all_model_types supported_models arch` | 10 passed |
| `cargo clippy --profile test-fast --lib --tests --features metal,accelerate -- -D warnings` | clean |
| `cargo fmt --all -- --check` | clean |

### B. 새 테스트가 고정하는 것

- `youtu_llm_model_type_is_detected_for_both_vendor_labels`: `youtu`와 `youtu_llm` 둘 다 `ModelType::YoutuLLM`에 도달.
- `deepseek_v2_labelled_youtu_export_routes_to_the_youtu_decoder`: 로컬 4-bit 체크포인트의 config 형태를(디스크에서 읽지 않고 필드로 구성) Youtu 디코더로 보낸다.
- `genuine_deepseek_v2_keeps_its_route`: `DeepseekV2ForCausalLM`과 `architectures` 배열이 없는 config 둘 다 `DeepSeekV2`에 남는다. 기존 DeepSeek-V2 사용자 전원에 대한 퇴행 가드다.
- `text_only_config_with_identity_yarn_parses`: 실제 필드 셋이 파싱되고, 항등 YaRN 블록에서의 어텐션 스케일이 블록 없는 스케일과도, 리터럴 `(qk_nope + qk_rope) ** -0.5`와도 같다.
- `rope_scaling_that_interpolates_is_refused`: factor 4는 factor와 스킴을 이름으로 밝히는 에러를 낸다.
- `null_q_lora_rank_parses_and_selects_the_direct_q_projection`: null, 부재, 숫자 랭크 모두 파싱되고 캐리어 config가 각각 옳은 분기 선택자를 갖는다.
- `rope_interleave_and_rope_traditional_are_the_same_switch`: 어느 키가 꺼져도 half-split 형태가 선택된다.
- `tied_embeddings_produce_logits_without_an_lm_head`: 합성 tied 빌드가 `lm_head == None`이고 디코더 전체를 통과해 유한한 `[1, 4, vocab]` 로짓을 낸다.
- `rope_interleave_reaches_the_rope_call`: 동일한 합성 가중치로 플래그만 뒤집어 만든 두 모델의 로짓이 다르다. 플래그가 파싱만 되고 버려지지 않는다는 단언이다.

### C. 머지 후 검증 (오케스트레이터)

로컬 `models/mlx/youtu-llm-2b-4bit` 체크아웃이 새 라우트를 직접 태운다. 이 변경이 있으면 `architectures[0]`가 그것을 `YoutuLLM`으로 보내고, `main`에서는 같은 디렉터리가 `DeepSeekV2`로 간다. 그래서 이것은 스모크 테스트가 아니라 차분 게이트다.

```bash
cargo build --release --features metal,accelerate
./target/release/mlxcel arch | grep -i youtu
./target/release/mlxcel generate -m models/mlx/youtu-llm-2b-4bit --no-chat-template -n 32 --temp 0 \
  -p "The Fibonacci sequence begins with"
./target/release/mlxcel generate -m models/mlx/youtu-llm-2b-4bit --no-chat-template -n 32 --temp 0 \
  -p "In distributed systems, consensus protocols such as Raft"
```

기대값: `mlxcel arch`가 `Specialized:` 아래에 `Youtu-LLM (DeepSeek-V2-style MLA decoder, text-only)`를, `Other VLM:` 아래에 기존 Youtu-VL 항목을 함께 보여준다. 두 생성 모두 1.3에 인용한 mlx-lm `deepseek_v2` 오라클 연속과 선두 토큰이 일치해야 하고, 꼬리는 양자화와 리덕션 순서 노이즈를 허용한다. 실패 신호는 역시 1.3에 인용한 현재 `main`의 동작이다.
