# 기술 보고서: PR #1593 - feat(models): 텍스트 전용 Youtu-LLM을 전용 MLA 디코더로 라우팅

**작성일**: 2026-09-02
**작성자**: mlxcel maintainers
**리뷰어**: 구현 리뷰 사이클, 그리고 mlx-lm 오라클 대비 단계별 수치 비교
**상태**: 완료 (유닛 커버리지 통과. 부록 C의 실제 체크포인트 수치는 리뷰 중 실측했고 거기 적힌 명령으로 재현된다)
**언어**: Rust
**위험도**: Low (순수 추가. 그동안 거부되던 `model_type` 라벨 둘에 라우트가 생길 뿐 기존 라우트는 손대지 않는다)

---

## 요약

텐센트의 텍스트 전용 Youtu-LLM은 mlxcel이 이미 Youtu-VL의 텍스트 타워로 돌리고 있는 바로 그 디코더인데, 텍스트 전용 라우트가 없었다. `model_type: "youtu"`와 `"youtu_llm"`은 둘 다 탐지 단계에서 거부됐다. PR #1593은 새 아키텍처 코드 없이 기존 `YoutuLanguageModel` 위에 `ModelType::YoutuLLM`을 얹고, Youtu-VL 형제 체크포인트가 한 번도 건드리지 않는 설정 세 가지를 제대로 반영한다.

이 보고서에서 더 쓸모 있는 절반은 리뷰 사이클이 찾아낸 것이다. 이 PR의 첫 버전은 `deepseek_v2` 라벨을 `architectures[0]`로 쪼개기도 했다. 세 번째 공개 변환본(`mlx-community/Youtu-LLM-2B-4bit`, mlx-lm 호환을 위해 스스로를 `deepseek_v2`로 재라벨링한 것)을 mlxcel의 DeepSeek-V2 디코더가 잘못 처리한다고 믿었기 때문이다. 그 믿음은 결정적으로 보였지만 실제로는 그렇지 않은 greedy 디코드 비교에서 왔다. 직접 재보니 DeepSeek-V2 라우트는 같은 가중치의 mlx-lm 오라클을 잘 따라가고 있었고, 실패처럼 보였던 것은 서로 다른 두 인공물이었다. CLI는 모델의 chat template을 적용하고 있었고 오라클은 아니었으며, 이 체크포인트는 raw completion 프롬프트에서 상당수 위치가 거의 미결정 상태라 greedy 출력 자체가 구현 간에 재현되지 않는다. 분기는 제거했다. 남은 것은 이슈가 요청한 순수 추가 라우트다.

---

## 1. 문제 정의

### 1.1 배경

`src/models/youtu_vl_lm.rs`에는 Youtu 디코더 전체가 이미 구현돼 있다. DeepSeek-V2 레이아웃의 MLA(`q_a_proj` -> `q_a_layernorm` -> `q_b_proj`로 압축된 쿼리를 128 비위치 + 64 회전 차원으로 분할, `kv_a_proj_with_mqa`가 512차원 latent와 모든 헤드가 공유하는 64차원 회전 키를 내놓고, `kv_b_proj`는 로드 시점에 헤드별 `embed_q` / `unembed_out` 쌍으로 분해), dense SwiGLU MLP, tied word embedding. `(Self, YoutuTextConfig)`를 반환하는 독립 실행형 `YoutuLanguageModel::load`도 있었지만 테스트에서만 닿았다.

빠진 것은 레지스트리 작업이었다. `ModelType`, `LoadedModel`, 메타데이터 레지스트리, 디렉터리 라우트, 탐지 arm. 이 변경 전까지 `grep YoutuLanguageModel src/`는 `src/loading/vlm_youtu_vl.rs`와 `src/vision/youtu_vl.rs`만 걸렸다.

### 1.2 라벨은 셋, 그중 둘이 없었다

| 체크포인트 | `model_type` | 변경 전 | 변경 후 |
|---|---|---|---|
| `tencent/Youtu-LLM-2B` | `youtu` | 거부: "Unsupported model type" | `ModelType::YoutuLLM` |
| `mlx-community/Youtu-LLM-2B-mlx-4bit` | `youtu_llm` | 거부 | `ModelType::YoutuLLM` |
| `mlx-community/Youtu-LLM-2B-4bit` | `deepseek_v2` | `ModelType::DeepSeekV2` | 그대로 |

### 1.3 VLM 형제가 고정해 주지 못하는 것

`models/mlx/youtu-vl-4b-instruct`에는 `rope_scaling` 블록이 없고 `rope_interleave: true`, `q_lora_rank: 1536`이다. 그래서 설정 세 가지가 텍스트 전용 체크포인트와 함께 처음으로 mlxcel에 도달하는데, 셋 다 로드 에러가 아니라 조용한 오답 부류였다.

- `rope_interleave`는 `YoutuTextConfig`로 파싱만 되고 읽히지 않았다. 공유 `DeepSeekV3Attention`은 두 `fast_rope` 호출에 리터럴 `true`를 넘겼으므로, half-split 레이아웃을 선언한 체크포인트는 아무 에러 없이 반대로 회전됐을 것이다.
- `rope_scaling`은 캐리어 `DeepSeekV3Config`로 전달돼 `get_attention_scale`이 mscale 키를 읽지만, 주파수 테이블 자체는 평범한 `rope_theta`다. 실제로 보간하는 YaRN 블록은 조용히 버려졌을 것이다.
- `q_lora_rank`는 필수 `usize`였으므로 `null`이면 공유 어텐션이 이미 구현해 둔 직접 `q_proj` 분기를 고르는 대신 serde 에러로 로드 전체가 실패했다.

### 1.4 위험성

| 위험 | 영향 | 발생 가능성 |
|---|---|---|
| 어떤 부류의 체크포인트에 대해 기존 라우트가 바뀜 | 발생 시 High | 제거. 최종 변경은 `model_type` arm 두 개를 추가할 뿐 기존 arm을 건드리지 않는다. `deepseek_v2_label_keeps_the_deepseek_v2_route`가 세 가지 architecture 표기를 모두 고정 |
| `rope_traditional` 필드가 DeepSeek-V3 / V3.2 / V4 / Kimi-VL 동작을 바꿈 | 발생 시 High | 제거. `from_weights`가 `true`로 설정하며 이는 두 `fast_rope` 호출이 쓰던 리터럴 그대로다. Youtu 디코더 레이어만 덮어쓴다 |
| 검증된 config 타입을 공유하면서 Youtu-VL이 퇴행 | Medium | 회피. `validate_rope_scaling`은 텍스트 전용 `load`에서만 호출된다. VLM 로더는 손대지 않았고 공개된 Youtu-VL 체크포인트 중 이 블록을 가진 것도 없다 |
| 새 라우트가 실제 체크포인트에서 수치적으로 틀림 | High | 가정이 아니라 실측. 부록 C. prefill은 mlx-lm 오라클과 전 위치에서 일치하고, chat template 프롬프트에서 greedy 32토큰이 token-exact |
| rope_scaling 거부가 정당한 장문맥 변환본을 막음 | Low | 의도적으로 수용. 지원하지 않는 스킴을 이름과 함께 알리는 로드 에러가 유창한 오답보다 낫다 |

---

## 2. 기술적 검토 사항

**보안.** 모든 라우트가 이미 파싱하는 체크포인트 `config.json` 외에 신뢰 불가 입력이 닿는 새 파싱 표면은 없다. `validate_rope_scaling`은 선택 키 두 개를 읽어 에러 문자열로 포맷하며, `factor`는 기본값 1.0의 `f64`로 읽으므로 값이 숫자가 아니거나 없어도 가드가 오작동하지 않는다. 새 `parse_eos_token_ids`는 각 id를 `as` 캐스트가 아니라 `i32::try_from`으로 경계 검사하므로 범위를 벗어난 `eos_token_id`는 래핑된 값이 아니라 빈 결과가 된다.

**성능.** `DeepSeekV3Attention`에 `bool` 하나가 늘었고 레이어당 forward당 두 번 읽힌다. 상대는 이미 주파수 테이블을 만드는 호출이다. 디코드 경로에 할당은 추가되지 않는다. `validate_rope_scaling`과 `parse_eos_token_ids`는 로드 시 한 번 실행된다.

**정확성.** 하중을 받는 주장은 `from_weights`의 `rope_traditional: true`가 이전에 두 호출 지점이 하던 것과 정확히 같다는 것이고, 리터럴 `true`가 인자 위치에서 구조체 초기화로 옮겨졌을 뿐이며 `models::deepseek` 커버리지는 그대로다. 두 번째 주장(플래그가 파싱만 되는 게 아니다)은 눈이 아니라 차분 테스트로 고정했다. 세 번째 주장(라우트가 수치적으로 건전하다)은 논증이 아니라 부록 C의 실측으로 고정했다.

---

## 3. 기술적 선택과 그 이유

### 3.1 `deepseek_v2` 라벨은 라우트를 유지한다, 그리고 처음에 반대로 판단했던 근거

이 PR의 첫 버전은 `deepseek_v2_or_youtu`를 추가해, `architectures[0]`가 `YoutuForCausalLM`인 `deepseek_v2` 체크포인트를 Youtu 디코더로 보냈다. 근거는 greedy 디코드 비교였다. mlxcel이 "The Fibonacci sequence begins with"를 "1 and 2, timing with the time of the day as a factor"로 이었고, 같은 가중치의 mlx-lm `deepseek_v2` 오라클은 "1,1, and each subsequent number is the sum of the two preceding numbers"를 냈다는 것.

그 비교를 믿는 대신 두 디코더를 오라클에 직접 돌려 보니 근거가 녹아 없어졌다. raw 5토큰 프롬프트에서 mlxcel의 DeepSeek-V2 라우트는 `[220, 16, 11, 16, 11, 323, 1981, 21056, 1692, 371, 290, 3304, 324, 290, 1552, 52512, 6578, 13]`을 내는데, 이는 오라클 자신의 시퀀스와 18토큰까지 같다. 즉 " 1,1, and each subsequent number is the sum of the two preceding numbers." chat template 프롬프트에서는 32토큰 전부 일치한다. 고칠 것이 없었다.

분기는 제거했다. 아무 대가 없이 잘 돌던 체크포인트 부류를 다른 디코더로 옮기는 거래이고, 그냥 두는 쪽보다 명백히 나쁘다. 같은 논리로 다시 추가되지 않도록 가드 테스트가 이유를 기록한다.

### 3.2 새 `DeepSeekV3Config` 키가 아니라, 보존적 기본값을 가진 필드

`rope_interleave`를 반영하려면 플래그가 `fast_rope`까지 가야 했다. `DeepSeekV3Config`에 넣는 게 깔끔해 보이지만 그 구조체는 Youtu 캐리어와 테스트 픽스처에서 리터럴로 만들어지므로, 호출자 하나만 값을 바꾸는데 모든 리터럴을 기계적으로 고치는 일이 되고 Youtu 고유 개념을 DeepSeek config 타입에 심는다. `from_weights` 파라미터로 넘기면 네 계열이 호출하는 시그니처가 바뀐다.

선택한 형태는 어텐션 구조체의 `rope_traditional` 필드다. `from_weights`가 `true`로 설정하고 Youtu 디코더 레이어가 `with_rope_traditional`로 덮어쓴다. 기본값이 기존 동작을 보존한다는 것이 증명 가능한 가장 작은 변경이다. 공개 필드를 직접 찌르는 대신 빌더를 쓴 것은 호출 지점에서 의도로 읽히기 때문이다.

### 3.3 `rope_is_interleaved()`는 같은 스위치의 두 철자를 접는다

`YoutuTextConfig`는 `rope_traditional`(mlx-vlm 포트의 이름)과 `rope_interleave`(벤더의 이름)를 둘 다 갖고 있고 기본값은 모두 true다. 하나만 읽으면 디코더가 "어느 변환기가 만든 체크포인트인가"에 의존한다. 접근자는 논리곱을 반환하므로 어느 쪽이든 꺼지면 half-split 형태가 선택되고, 둘 다 없는 체크포인트는 늘 하던 대로 동작한다. 공개 체크포인트 중 둘을 반대로 설정한 것은 없으므로 이 논리곱은 실제 충돌을 해소하는 게 아니라 어느 키가 정본인지 추측하기를 거부한다.

### 3.4 YaRN factor가 1을 넘으면 무시하지 말고 거부한다

공개된 블록은 `{"type": "yarn", "factor": 1.0, "mscale_all_dim": 0}`이고 두 겹으로 항등이다. factor 1에서 YaRN의 외삽/내삽 주파수가 일치하고 어텐션 mscale도 1이다. 벤더도 mscale은 `mscale_all_dim`이 참일 때만 적용하고 `yarn_get_mscale`은 factor 1에서 1을 반환한다. mlxcel의 `get_attention_scale`은 이미 `mscale_all_dim > 0 && factor > 1`로 가드하므로 스케일은 원래 맞았고, 이제 그 동치를 테스트가 고정한다.

처리되지 않던 쪽은 factor가 1을 넘는 경우로, 공유 MLA 어텐션이 구현하지 않은 주파수 보간을 요구한다. 무시하면 원래 컨텍스트 길이를 넘어가는 지점에서만 위치가 틀린 유창한 출력이 나오는데, 이보다 발견하기 어려운 실패 모드는 없다. 가드는 이를 스킴과 factor를 이름으로 밝히는 로드 에러로 바꾼다. 텍스트 라우트에서만 호출하는데, VLM 로더는 사용자가 있는 출하 경로이고 필요로 하는 체크포인트의 증거 없이 거부를 넣는 것은 이유 없는 동작 변경이기 때문이다.

### 3.5 어댑터 weight 라우트 없는 `Nonstandard` 디렉터리 라우트

`YoutuLanguageModel::load`는 `(Self, YoutuTextConfig)`를 반환하는데 `loading::nonstandard::load_pair_from_dir`가 소비하는 형태 그대로다. 그래서 디렉터리 라우트는 `KimiLinear`, `Qwen35` 옆의 arm 하나다. 레지스트리 항목은 `weight: None, adapter: Some(...)`으로 `DiffusionGemma`, `Llada2Moe`와 같다. 이 필드는 `adapter_weight_route`로 LoRA 어댑터 로딩만 구동하므로, 없다고 선언하는 편이 아무도 쓰지 않는 `SpecialWeightLoaderKind` arm을 배선하는 것보다 정직하다.

### 3.6 호출자가 디스크에서도 읽지만 로더에서 `eos_token_id`를 파싱한다

CLI(`crate::read_eos_token_ids`)와 서버(`model_worker.rs`) 모두 `generation_config.json`과 `tokenizer_config.json`에서 읽은 stop id를 병합하므로 모델 자신의 `eos_token_ids()`가 유일한 출처는 아니다. 그래도 네 줄의 값어치는 있다. 다른 모든 계열이 지키는 계약이고, 그 두 경로 밖에서 생성된 모델은 그렇지 않으면 stop id를 하나도 보고하지 못한다.

---

## 4. 구현 상세

### 4.1 탐지

`"youtu" | "youtu_llm"` arm은 `"youtu_vl"` 옆에 있다. `"deepseek_v2"` arm은 그대로 두는 근거가 된 실측을 기록한 주석 외에 변경이 없다.

### 4.2 rope 플래그가 config에서 커널까지 가는 경로

| 단계 | 코드 |
|---|---|
| 파싱 | `YoutuTextConfig::rope_interleave` / `rope_traditional`, 둘 다 `#[serde(default = "default_true")]` |
| 접기 | `YoutuTextConfig::rope_is_interleaved()` |
| 적용 | `YoutuDecoderLayer::from_weights`가 `.with_rope_traditional(config.rope_is_interleaved())` 호출 |
| 사용 | `DeepSeekV3Attention::forward`가 두 `fast_rope` 호출에 `self.rope_traditional` 전달 |

벤더 쪽 분기는 https://huggingface.co/tencent/Youtu-LLM-2B/blob/main/modeling_youtu.py 의 `YoutuMLAttention.forward`이고, `apply_rotary_pos_emb` 대신 `apply_rotary_pos_emb_interleave`(평소의 `rotate_half` 앞에 `view(..., d // 2, 2).transpose(4, 3)`)를 고른다. 그 reshape가 MLX에서 `traditional=True`로 불린다. 두 형태는 회전 대상 차원의 공통 치환 하나만큼 다르고 query-key 내적은 그 치환에 불변이므로, HF의 interleave 형태와 MLX의 traditional 형태는 같은 어텐션 점수를 낸다.

### 4.3 Config 타입 변경

`q_lora_rank: Option<usize>`에 `#[serde(default)]`. 값은 캐리어 `DeepSeekV3Config`로 그대로 전달되고, `from_weights`는 이미 `is_none()`으로 분기해 LoRA 체인 대신 `q_proj`를 고른다. `validate_rope_scaling()`은 `Result<(), String>`을 반환하며 가중치를 건드리기 전에 `load`에서 호출되므로, 거부되는 체크포인트는 config 읽기 한 번의 비용만 든다.

### 4.4 등록

enum, `ALL_MODEL_TYPES`, `metadata()`(family `Specialized`, `FAMILY_ORDER`에 이미 존재), `all_variants!` 완전성 목록의 `ModelType::YoutuLLM`. `LoadedModel::YoutuLLM`과 그 `delegate_language_model!` arm. `model_metadata.rs`에 `kind: Text, directory: Nonstandard`. `loading/nonstandard.rs`에 `load_pair_from_dir` arm. 디스패치 테이블을 total로 유지하기 위한 `src/distributed/tensor_parallel/inference.rs`의 arch 문자열 arm.

---

## 5. 학습 포인트

### 5.1 레퍼런스가 정밀도에 안정적이지 않으면 greedy 디코드 비교는 증거가 아니다

이 체크포인트는 raw completion 프롬프트에서 거의 미결정 상태다. 126토큰 지문에서 top-1과 top-2 로짓 격차의 중앙값이 0.625이고, 여덟 위치 중 하나는 bf16에서 정확히 동점이다. 그 여유에서 greedy 출력은 구현의 속성이 아니라 마지막 비트의 속성이다. 같은 가중치에 대한 mlx-lm 자신의 bf16 실행과 float32 실행조차 그 위치들의 91.3%에서만 일치하고, 5토큰 Fibonacci 프롬프트에서는 여섯 번째 토큰부터 완전히 다른 글을 낸다.

그러므로 "mlxcel은 X, 레퍼런스는 Y, 따라서 mlxcel이 깨졌다"는 레퍼런스가 더 높은 정밀도에서 자기 자신과 일치할 때만 논증이 된다. 값싼 점검은 레퍼런스를 두 번 돌리는 것이다. 한 번은 원래 정밀도로, 한 번은 올려서, 그리고 비교. 다르면 그 비교는 지워진 하중을 감당할 수 없고, 어떤 결론을 내기 전에 프롬프트를 바꿔야 한다.

### 5.2 출력을 비교하기 전에 양쪽이 실제로 무엇을 받았는지 확인한다

보고된 mlxcel 출력은 "Okay, so the question is about whether the Fibonacci sequence can be generalized"로 시작했고, 프롬프트를 무시하는 모델처럼 읽힌다. 아니다. 같은 프롬프트를 체크포인트 자신의 chat template으로 렌더링해 오라클에 넣으면 `<think>\nOkay, the user is asking about the Fibonacci sequence beginning...`이 나온다. CLI는 chat template을 적용하고 있었고 오라클은 아니었으므로, 양쪽은 서로 다른 질문에 답하고 있었다. 단서는 수치 없이도 있었다. raw completion이어야 할 출력에 reasoning 모델의 `<think>` 오프너가 나타난 것.

### 5.3 단계별 비교는 "어디"에, teacher-forced 비교는 "정말 문제인가"에 답한다

단계별 비교는 첫 수치 발산 지점을 정확히 짚었다(쿼리 경로, latent, 두 rope 호출까지 비트 단위 동일. 첫 차이는 헤드별 key/value 실체화). 하지만 단계별 비교는 그 발산이 문제인지를 말해 주지 못한다. 첫 단계 이후 모든 단계가 앞 단계의 오차를 물려받기 때문이다. teacher-forced 비교가 그 답을 준다. 두 구현에 같은 고정 토큰 열을 먹이고 전 위치의 argmax를 비교하면 피드백 루프가 사라진다. 여기서는 126 중 118 위치 일치가 나왔고, 불일치는 전부 top-2 격차가 0에서 0.75 로짓인 위치였다. 질문을 매듭짓는 숫자는 이것이지 자유 실행 greedy 궤적이 아니다.

### 5.4 파싱만 되고 읽히지 않는 config 키는 없는 키보다 나쁘다

`rope_interleave`에는 필드도, 기본값도, 문서 주석도 있었고 소비자만 없었다. 키를 grep한 모든 리뷰가 그것을 찾았다. 네 계열에서 `rope_scaling`이 파싱되고 버려진 것(#1355)과 같은 형태다. 필드가 존재한다는 사실 자체가 "그래서 적용은 되나"를 아무도 묻지 못하게 만든다. 선언이 아니라 소비자를 grep하는 것이 이를 잡는 점검이고, 이빨이 있는 테스트는 파싱된 값에 대한 단언이 아니라 차분(플래그를 뒤집고 로짓이 달라지기를 요구)이다.

### 5.5 라우팅 규칙에 전제를 새기기 전에 기계에서 검증한다

탐지 분기는 실측이 아니라 보고된 비교에서 쓰였고, 두 디코더를 같은 입력에 돌려 보기 전에 구현과 리뷰와 보고서 사이클을 통째로 살아남았다. 라우팅 규칙은 검증되지 않은 전제가 가장 비싸게 먹히는 자리다. 평상시에는 보이지 않고, 그 자리에 있는 누구도 소유하지 않은 체크포인트의 동작을 바꾼다.

---

## 6. 추가 학습 리소스

### 핵심 키워드

- **MLA (Multi-head Latent Attention)**: 키와 값을 저랭크 latent에서 복원하는 어텐션. 캐시가 full K/V 대신 latent를 저장한다. DeepSeek-V2의 레이아웃이고 Youtu가 그대로 쓴다.
- **흡수(absorbed) MLA**: `kv_b_proj`를 쿼리/출력 프로젝션(`embed_q` / `unembed_out`)에 접어 넣어 디코드가 full key를 만들지 않게 하는 것. `DeepSeekV3Attention`은 항상, `deepseek_v2.rs`는 플래그가 있을 때만.
- **Interleaved (traditional) RoPE**: half-split 쌍이 아니라 인접 차원 쌍을 회전. MLX에서는 `traditional=True`, PyTorch에서는 `rotate_half` 앞의 `view(d // 2, 2).transpose`로 만들어진다.
- **YaRN 항등 블록**: `factor`가 1인 `rope_scaling` 항목. 내삽/외삽 주파수 테이블이 일치하고 어텐션 mscale이 1이므로 블록이 없는 것과 같다.
- **Teacher forcing**: 모델 출력을 되먹이는 대신 고정 토큰 열 위에서 돌려 전 위치의 예측을 읽는 것. 비교에서 궤적 발산을 제거한다.

### 관련 PR/이슈

- #1371: 이 이슈.
- #1355: 네 계열에서 `rope_scaling`이 파싱되고 적용되지 않은 건. 5.4와 같은 결함 부류.
- #958, #1026: 이 라우트가 재사용하는 MLA `kv_b_proj` 새니타이저의 공용 하드닝.

---

## 7. 변경 요약

### 통계

| 항목 | 값 |
|---|---|
| 변경 파일 | 13 |
| 새 `ModelType` variant | 1 |
| 새 테스트 | 8 |
| 수정된 기존 탐지 arm | 0 |

### 카테고리별 변경

- **탐지**: `src/models/detection.rs`(새 `model_type` arm 둘), `src/models/detection_tests.rs`.
- **모델과 config**: `src/models/youtu_vl_lm.rs`, `src/models/youtu_vl_lm_config.rs`, `src/models/youtu_vl_lm_sanitize.rs`(주석만), `src/models/youtu_vl_lm_tests.rs`, `src/models/deepseek_v3.rs`.
- **레지스트리와 로딩**: `src/models/mod.rs`, `src/loaded_model.rs`, `src/model_metadata.rs`, `src/loading/nonstandard.rs`, `src/distributed/tensor_parallel/inference.rs`.
- **문서**: `docs/supported-models.md`.

---

## 8. 후속 조치

### 모니터링 필요

- 이 계열의 parity 비교는 반드시 chat template을 써야 한다. raw 프롬프트 greedy 비교는 어떤 레퍼런스에 대해서도 헛된 실패를 만든다. 문서 항목이 그렇게 적어 두었지만, 읽지 않은 기여자는 5.1을 다시 발견하게 된다.

### 향후 개선

- 출하 중인 VLM 체크포인트에 scaling 블록이 없다는 확인이 되면 `validate_rope_scaling`을 Youtu-VL 로더로 확장. 그 라우트에서도 같은 조용한 오답 부류다.
- 공유 MLA 어텐션에 YaRN 주파수 보간을 구현하면 거부가 지원으로 바뀐다. factor 1 초과의 Youtu 또는 Kimi-VL 변환본이 나타날 때만 값어치가 있다.
- `tencent/Youtu-LLM-2B`(bf16, `model_type: youtu`)와 `mlx-community/Youtu-LLM-2B-mlx-4bit`(`model_type: youtu_llm`) 종단 검증. 둘 다 검증 호스트에 없고 평범한 라벨 arm을 타므로, 새 코드 검증이라기보다 남은 두 라벨을 닫는 일이다.
- 흡수 MLA와 `deepseek_v2.rs`가 쓰는 실체화 형태는 대수적으로 같고 수치적으로 다르며, 이 체크포인트는 raw 프롬프트에서 그 차이가 보일 만큼 민감하다. 어떤 계열이 mlx-lm과 비트 수준 일치를 요구하게 되면, 산술 순서를 공유하는 쪽은 실체화 형태다.

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
- `deepseek_v2_label_keeps_the_deepseek_v2_route`: Youtu 재라벨본, 진짜 `DeepseekV2ForCausalLM`, `architectures` 배열이 없는 config 모두 `DeepSeekV2`에 남는다. 3.1의 분기가 다시 들어오지 않게 하는 가드.
- `text_only_config_with_identity_yarn_parses`: 실제 필드 셋이 파싱되고, 항등 YaRN 블록에서의 어텐션 스케일이 블록 없는 스케일과도 리터럴 `(qk_nope + qk_rope) ** -0.5`와도 같다.
- `rope_scaling_that_interpolates_is_refused`: factor 4는 factor와 스킴을 이름으로 밝히는 에러를 낸다.
- `null_q_lora_rank_parses_and_selects_the_direct_q_projection`: null, 부재, 숫자 랭크 모두 파싱되고 올바른 분기 선택자를 갖는다.
- `rope_interleave_and_rope_traditional_are_the_same_switch`: 어느 키가 꺼져도 half-split 형태가 선택된다.
- `tied_embeddings_produce_logits_without_an_lm_head`: 합성 tied 빌드가 `lm_head == None`이고 디코더 전체를 통과해 유한한 `[1, 4, vocab]` 로짓을 낸다.
- `rope_interleave_reaches_the_rope_call`: 동일한 합성 가중치로 플래그만 뒤집어 만든 두 모델의 로짓이 다르다. 플래그가 파싱만 되고 버려지지 않는다는 단언.
- `prefill_and_decode_paths_agree_position_by_position`: 흡수된 `l == 1` 디코드 arm과 다중 토큰 prefill arm이 전 위치에서 일치한다. 스위트에서 디코드 arm을 태우는 유일한 테스트.

### C. 실제 체크포인트 실측

전부 `mlx-lm` 0.31.3이 같은 `mlx-community/Youtu-LLM-2B-4bit` 가중치를 로드한 것에 대한 비교이고, 양쪽 프롬프트 id는 동일, temperature 0 greedy다.

**Prefill, raw 5토큰 프롬프트.** 단계별로 mlxcel 대 오라클: 임베딩 출력 비트 동일. layer 0 어텐션 출력 평균 상대차 0.33%. 레이어별 은닉 상태는 layer 0의 1.0%에서 layer 31의 5.8%로 드리프트. 최종 norm 7.3%, 로짓 5.4%. 다섯 위치 전부 argmax 동일. layer 0 내부에서 쿼리 경로(`q_a_proj`, `q_a_layernorm`, `q_b_proj`, nope/rope 분할), 압축 KV, latent, 두 rope 호출이 모두 비트 동일하고, 첫 차이는 헤드별 key/value 실체화다. mlxcel은 역양자화한 가중치로, 오라클은 여전히 양자화된 가중치로 축약한다.

**Teacher forcing, 126토큰 지문.** mlxcel은 bf16 오라클과 126개 argmax 위치 중 118개(93.7%)에서 일치한다. 불일치는 모두 top-1과 top-2 격차가 0.0에서 0.75 로짓인 위치이고, 지문 전체의 격차 중앙값은 0.625다. 여덟 중 셋은 정확히 동점이다. 비교 기준으로, bf16 오라클은 같은 위치들에서 자신의 float32 판과 126개 중 115개(91.3%)만 일치한다.

**Greedy 디코드, raw 프롬프트.** "The Fibonacci sequence begins with"에서 bf16 오라클과 float32 오라클은 step 5(top-2 격차 0.250)에서 갈라져 완전히 다른 글을 낸다. " 1,1, and each subsequent number is the sum" 대 " 1,1,1,4,34,". mlxcel은 float32 시퀀스를 12토큰 정확히 재현한다. 이 프롬프트는 parity 게이트로 쓸 수 없다.

**Greedy 디코드, chat template 프롬프트.** 같은 요청을 체크포인트의 chat template으로 렌더링하면 top-2 격차 중앙값이 0.625에서 3.875로 오르고 bf16과 float32 오라클이 32토큰 전부 일치한다. mlxcel은 양쪽 모두에 대해 32토큰 token-exact다.

```
prompt ids  [128000, 128236, 837, 91949, 12082, 13328, 458, 128237]
expected    [128227, 198, 37317, 11, 290, 1483, 371, 11935, 913, 290, 91949, 12082, 7963, 13, 6846, 611, 1311, 603, 125377, 1165, 290, 91949, 12082, 371, 13, 5902, 261, 4326, 324, 6578, 1551, 1981]
text        "<think>\nOkay, the user is asking about the Fibonacci sequence beginning. Let me start by recalling what the Fibonacci sequence is. It's a series of numbers where each"
```

두 번째 chat 프롬프트("In distributed systems, consensus protocols such as Raft")도 32토큰 token-exact다.

**같은 가중치에 대한 DeepSeek-V2 라우트.** chat template에서 32 중 32 오라클과 동일. raw 프롬프트에서 오라클의 첫 18토큰과 동일. 3.1의 탐지 분기를 제거하게 만든 실측이다.

**재현.** 이 라우트는 `loading::nonstandard`가 호출하는 `YoutuLanguageModel::load`로 태운다. 로컬 체크포인트는 `deepseek_v2`를 선언하므로 설계대로 DeepSeek-V2 라우트로 가고, CLI에서 새 라우트를 태우려면 디렉터리를 복사해 `model_type`을 `youtu`로 고쳐 쓴다.

```bash
cargo build --release --features metal,accelerate
./target/release/mlxcel arch | grep -i youtu

DIR=models/mlx/youtu-llm-2b-4bit
COPY=/tmp/youtu-llm-2b-4bit-youtu
mkdir -p "$COPY" && for f in "$DIR"/*; do ln -sf "$(cd "$(dirname "$f")" && pwd)/$(basename "$f")" "$COPY/"; done
rm -f "$COPY/config.json"
python3 -c "import json;c=json.load(open('$DIR/config.json'));c['model_type']='youtu';json.dump(c,open('$COPY/config.json','w'))"

./target/release/mlxcel generate -m "$COPY" -p "The Fibonacci sequence begins with" -n 32 --temp 0
```

`mlxcel arch`는 `Specialized:` 아래에 `Youtu-LLM (DeepSeek-V2-style MLA decoder, text-only)`를, `Other VLM:` 아래에 기존 Youtu-VL 항목을 함께 보여야 한다. 생성 결과는 위 chat template 텍스트를 재현해야 하는데, CLI가 기본적으로 체크포인트의 chat template을 적용하기 때문이다. `--no-chat-template`은 raw 프롬프트 레퍼런스와 짝지을 때만 쓰고, 거기서 결론을 내기 전에 5.1을 먼저 읽는다.
