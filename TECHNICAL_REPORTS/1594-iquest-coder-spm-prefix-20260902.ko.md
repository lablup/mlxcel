# 기술 보고서: PR #1594 - feat(models): add the IQuest-Coder (iquestcoder) route

**작성일**: 2026-09-02
**작성자**: mlxcel maintainers
**리뷰어**: 구현 리뷰 사이클, 그리고 동일 가중치에 대한 mlx-lm 0.31.3 오라클과의 5개 프롬프트 그리디 비교
**상태**: 완료 (단위 테스트 통과. 부록 A와 B의 실제 체크포인트 수치는 `models/mlx/iquest-coder-v1-7b-instruct-8bit`에서 측정했고 부록 C의 명령으로 재현된다)
**언어**: Rust
**위험도**: 모델 경로는 Low (순수 추가. 거부되던 `model_type` 하나가 경로를 얻을 뿐 기존 arm은 건드리지 않는다). 토크나이저 변경은 공유 코드라 Medium이며, `tokenizer_config.json`에 `"add_prefix_space": false`를 명시한 체크포인트로만 범위가 좁혀져 있다는 점이 위험도를 낮추는 근거다.

---

## 요약

IQuest-Coder V1은 `model_type: "iquestcoder"`를 선언하지만 실체는 라벨만 다른 `llama3` 디코더다. 구조상 거부할 이유가 없는데도 mlxcel은 detection 단계에서 이 체크포인트를 막고 있었다. PR #1594는 아키텍처 코드를 새로 쓰지 않고 기존 Llama 로더 위에 `ModelType::IQuestCoder`를 얹으며, 동치 관계를 깨뜨릴 수 있는 두 설정 스위치(`clip_qkv`, 슬라이딩 윈도 어텐션)는 로드 시점에 거부한다.

이 보고서에서 더 값어치 있는 절반은 토크나이저 쪽이다. 이슈를 촉발한 출력 불일치의 원인은 디코더가 아니었다. mlxcel의 SentencePiece 폴백이 `"add_prefix_space": false`를 읽지 않아서, 모든 프롬프트가 앞에 공백이 하나 붙은 것처럼 토크나이즈되고 있었다. 이를 고치려면 로드 전에 SentencePiece `ModelProto`의 필드 하나를 바꿔야 했다. `sentencepiece` 크레이트에 normalizer 설정 수단이 없기 때문이다. 조사 과정에서 이 체크포인트의 오라클이 기대고 있던 참조 토크나이저 자체가 BPE SentencePiece 모델에 대해 손실이 있다는 사실도 드러났고, 이는 이 계열에서 "오라클과 일치한다"는 말이 무엇을 뜻하는지를 바꾼다.

---

## 1. 문제 정의

### 1.1 배경

`mlxcel generate -m models/mlx/iquest-coder-v1-7b-instruct-8bit`는 `Error: Unsupported model type: iquestcoder`로 끝났다. `src/models/detection.rs`는 `"llama" | "mistral"`만 `ModelType::Llama`로 보냈고 나머지는 catch-all 오류 arm으로 떨어졌다.

이 계열(7B / 14B / 40B, Base / Instruct / Thinking)은 평범한 Llama 디코더다. RMSNorm, `head_dim`을 명시한 GQA, SwiGLU, 묶이지 않은 `lm_head`, 스케일링 없는 base 500000 RoPE, 어텐션과 MLP 바이어스 없음. 7B Instruct 기준으로 `hidden_size` 5120, `head_dim` 128, 어텐션 헤드 40개에 KV 헤드 8개, `intermediate_size` 27648, 레이어 14개, `vocab_size` 76800이다. `llama3::ModelArgs`는 선택적 `head_dim`을 포함해 이 설정을 이미 그대로 파싱한다.

### 1.2 Llama에는 없는 키들

이 설정에는 Llama에 없는 키가 셋 있다. `clip_qkv`, `sliding_window`를 동반하는 `use_sliding_window`, 그리고 `max_window_layers`다. 공개된 모든 체크포인트에서 셋 다 비활성 상태이며, 공유 디코더와의 동치는 이들이 비활성일 때만 성립한다. 벤더 디코더는 `clip_qkv`가 null이 아니면 QKV 클램프를 걸고, `use_sliding_window`가 켜져 있고 `sliding_window`가 null이 아니며 레이어 인덱스가 `max_window_layers`에 도달했을 때 해당 레이어를 윈도로 돌린다. 두 동작 모두 mlxcel의 Llama 어텐션에는 없고, 둘 다 로드 오류나 눈에 띄게 깨진 출력을 내지 않는다. 클램프된 모델은 틀린 어텐션 스코어로 유창하게 디코딩하고, 윈도가 걸린 모델은 틀린 수용 영역으로 유창하게 디코딩한다.

### 1.3 실제 결함이 있던 곳, 토크나이저

이 계열은 `tokenizer.json` 없이 SentencePiece `tokenizer.model`만 배포하므로 mlxcel은 `src/tokenizer/mod.rs`의 SentencePiece 폴백 경로를 탄다. 그런데 `tokenizer_config.json`에는 `"add_prefix_space": false`가 적혀 있고, mlxcel은 이 키를 읽지 않았다.

SentencePiece 모델은 기본적으로 `add_dummy_prefix`가 켜진 상태로 정규화한다. 이스케이프 전에 입력 앞에 공백을 붙이므로 `encode("The Fibonacci ...")`는 `The`가 아니라 `▁The`를 만든다. 원문 프롬프트에서는 0번 위치의 토큰 하나가 틀리는 문제로 끝나지만, 채팅 템플릿을 거치면 사정이 나빠진다. mlxcel은 렌더링된 텍스트를 특수 토큰 경계에서 잘라 각 조각을 따로 인코딩하므로, `<|im_start|>`와 `<|im_end|>` 사이의 모든 조각이 유령 공백을 하나씩 얻는다. 학습 때 모델이 본 적 없는 배열이다.

### 1.4 위험 평가

| 위험 | 영향 | 발생 가능성 |
|------|------|-------------|
| 체크포인트를 계속 로드하지 못함 | Medium (계열 전체 접근 불가. 우회책은 `config.json` 손수정) | 이번 변경 전에는 확정 |
| 향후 체크포인트가 `clip_qkv`나 슬라이딩 윈도를 켰는데도 로드됨 | High (유창한 출력, 조용히 틀린 어텐션) | 낮지만 가드 없이는 탐지 불가 |
| 프리픽스를 끈 모든 SentencePiece 체크포인트에 유령 공백 | Medium (측정 가능한 우도 악화, 그리디 디코드 불일치) | 해당 체크포인트에서는 확정 |
| 토크나이저 수정 범위를 너무 넓게 잡아 무관한 체크포인트를 움직임 | High (공유 토크나이저 변경은 모든 SentencePiece 계열에 닿는다) | 명시적 `false`를 요구해 완화 |

---

## 2. 기술 리뷰

### 2.1 보안

토크나이저 변경은 `tokenizer.model` 위를 도는 protobuf 리더를 새로 들인다. 이 파일은 운영자가 내려받은 체크포인트에서 오므로 공격자 영향권 안의 입력이다. `src/tokenizer/spm_proto.rs`는 그 전제로 작성했다. 모든 길이와 전진은 `checked_add`와 버퍼 길이 검사를 함께 거치고, 64비트를 넘는 varint는 레지스터 밖으로 시프트하는 대신 거부하며, 필드 번호 0과 폐기된 group wire type도 순회하지 않고 거부한다. 경계 검사 없이 인덱싱하는 자리는 없다. 잘못된 모델이면 `Err`를 돌려주고, 호출부는 경고 한 줄을 남긴 뒤 원본 그대로 로드한다. 모델 전체를 실패시키지는 않는다. 잘린 메시지와 버퍼 끝을 넘어가는 길이 접두사, 두 가지 손상 사례에 테스트가 붙어 있다.

**발견된 이슈:** 미해결 없음.

### 2.2 성능

재작성은 모델 로드마다 한 번 일어나고 `tokenizer.model`(여기서는 1.28 MB)을 두 번 복사한다. 읽을 때 한 번, 내보낼 때 한 번이다. 디코드 경로는 그대로다. 토크나이제이션 방식 자체도 바뀌지 않았다. `open` 대신 `from_serialized_proto`로 로드할 뿐, 경로가 아니라 버퍼를 받는 같은 C++ 로더다.

다른 이유로 기록해 둘 값이 있다. 이 오버라이드는 시도한 모든 텍스트에서 모델 자신의 teacher-forced 음의 로그우도를 낮춘다. 단지 다른 게 아니라 옳다는 근거다.

| 텍스트 | 변경 전 (`add_dummy_prefix` 켬) | 변경 후 (끔) | 참조 fast 토크나이저 |
|--------|-------------------------------|--------------|---------------------|
| 산문 203자 | 75.44 nats | **73.44** | 79.54 |
| 기술 산문 171자 | 52.69 nats | **51.96** | 51.96 |
| 파이썬 소스 123자 | 31.45 nats | **27.25** | 27.25 |

### 2.3 호환성과 의존성

- **파괴적 변경**: 없다. 기존 `model_type` arm은 하나도 바뀌지 않았고, 토크나이저 오버라이드는 명시적 `"add_prefix_space": false`에서만 발동한다.
- **새 의존성**: 없다. `SentencePieceProcessor::from_serialized_proto`는 이미 고정된 `sentencepiece` 0.13.2에 있다.
- **호환성**: `ModelType`에 variant가 하나 늘어 외부의 exhaustive match 기준으로는 소스 호환이 깨지지만, `ModelType`은 안정성 계약 대상이 아니고 내부 사용처는 컴파일러가 전부 찾아 줬다.

### 2.4 코드 품질

- **테스트 커버리지**: 22개 추가. 9개는 합성 메시지로 protobuf 재작성을 검증하고, 2개는 실제 `tokenizer.model`을 읽으며(체크포인트가 없으면 공용 pinned 게이트로 skip), 10개는 두 거부와 비활성 키 사례, 슬라이딩 윈도 키의 타입 변형, 재라벨된 architecture 경로까지 포함한 detection을, 1개는 `mlxcel arch`에 계열이 나타나는지를 확인한다.
- **복잡도**: 공개 함수 하나짜리 237줄 모듈 하나가 늘었다.
- **기술 부채**: 계속 길어지던 튜플을 반환하던 `parse_special_tokens`를 이름 있는 구조체로 바꿨으므로 소폭 감소.

---

## 3. 기술적 선택과 그 이유

### 3.1 `llama` arm에 철자를 하나 더 붙이는 대신 전용 `ModelType`을 만든 이유

**상황:** 이슈가 제시한 계획은 `"llama" | "mistral" | "iquestcoder" => Ok(ModelType::Llama)`였다. 최소 변경이고 다른 여러 별칭이 쓰는 방식이기도 하다.

**검토한 대안:**

| 선택지 | 장점 | 단점 |
|--------|------|------|
| `ModelType::Llama`로 별칭 처리 | 한 줄. Llama의 모든 기능을 자동 상속 | `mlxcel arch`에 안 보인다. 이 명령은 별칭 문자열이 아니라 `ModelType`을 렌더링한다. 설정 거부 로직도 공유 Llama 로더에 넣어야 하는데 다른 Llama 체크포인트에는 죽은 코드가 된다 |
| **채택: 같은 로더를 쓰는 `ModelType::IQuestCoder`** | `mlxcel arch`에 노출된다. 거부 로직이 이 계열에만 걸린다. 등록 누락은 컴파일러가 잡아 준다 | 등록 지점 네 곳을 추가로 손봐야 한다. 분산 기능은 하나씩 의도적으로 부여해야 한다 |

**근거:** "이 체크포인트 되나요?"에 대한 권위 있는 답은 `mlxcel arch`이고, 별칭은 그 질문에 답하지 못한다. 선례는 하루 전 것이다. PR #1593이 정확히 같은 이유로 기존 디코더 위에 `ModelType::YoutuLLM`을 올렸다.

**트레이드오프:** 새 variant는 기능을 상속하지 않으므로 하나씩 줘야 한다. 여기서 적용한 규칙은 "별칭이었다면 가졌을 기능 집합을 그대로 재현한다"이다. 파이프라인 병렬은 별칭이 해석됐을 `StageFamily::Llama`로 보냈다. 텐서 병렬은 거부 상태로 두었는데, 이것도 별칭이었을 때와 같다. `runtime_kind_for`가 TP Llama 런타임을 architecture 문자열로 게이팅하고 그 문자열은 어느 설계에서든 `iquestcoder`이기 때문이다. TP를 열어 주는 것은 다중 랭크 검증 없이 기능을 주장하는 일이 된다.

### 3.2 인코딩 결과를 후처리하지 않고 SentencePiece proto를 재작성한 이유

**상황:** `sentencepiece` 크레이트는 `encode`, `open`, `from_serialized_proto`를 노출하지만 로드된 모델의 normalizer를 바꾸는 수단은 주지 않는다.

**검토한 대안:**

| 선택지 | 장점 | 단점 |
|--------|------|------|
| 인코딩 후 첫 조각의 `▁`를 떼어낸다 | 새 코드 경로 없음 | 틀렸다. 더미 프리픽스는 첫 조각만이 아니라 분절 전체를 바꾼다. `▁Fibonacci`는 `▁Fi`+`bon`+`acci`로, `Fibonacci`는 아예 다른 조각들로 쪼개진다. 후처리로는 복구할 수 없다 |
| 센티널 문자를 앞에 붙였다가 해당 조각을 버린다 | protobuf를 다루지 않아도 된다 | 어떤 어휘 조각도 센티널 경계를 넘지 않는다는 가정에 기대는데, 이는 알고리즘의 성질이 아니라 어휘의 성질이다. 여러 글자짜리 공백 조각이 많은 코드 토크나이저에서는 안전하지 않다 |
| `build_plamo_tokenizer`처럼 `tokenizers` 크레이트 모델로 재구성한다 | 이미 트리에 있는 패턴을 재사용 | 조각 점수가 필요하니 결국 protobuf 리더가 필요하고, 그 위에 분절까지 다시 구현하게 된다. 충실도는 떨어지는데 표면적만 넓어진다 |
| **채택: `normalizer_spec.add_dummy_prefix`를 재작성하고 다시 로드한다** | 분절은 여전히 C++ 구현이 한다. 조각, 점수, 머지 순서, 바이트 폴백이 바이트 단위로 보존된다 | protobuf 편집기를 직접 써야 하고, 방어적으로 작성하고 손상 입력에 대해 테스트해야 한다 |

**근거:** 편집 대상은 배치가 고정되고 공개된 메시지 안의 불리언 하나다. 토크나이제이션을 결정하는 요소는 전부 그대로 남는다. 중요한 성질이 바로 그것이다. 유령 프리픽스만 바뀌고 나머지는 아무것도 바뀌지 않아야 한다.

**트레이드오프:** 프로젝트에 없던 protobuf 처리 코드 90여 줄이 생겼다. 공개 함수 하나짜리 독립 모듈로 격리하고, 이해하지 못하는 입력은 거부하며, 실패 시 모델을 죽이는 대신 경고를 남기고 원본으로 로드하도록 해서 완화했다.

### 3.3 참조 토크나이저와의 정확한 일치 대신 체크포인트 자신의 BPE를 택한 이유

**상황:** 이 체크포인트의 참조 오라클은 config를 `llama`로 재라벨해 transformers가 `tokenizer.model`에서 fast 토크나이저를 만들게 해서 얻은 것이다. 검증 프롬프트 둘 중 하나에서 그 토크나이저와 체크포인트 자신의 SentencePiece 모델이 "Fibonacci"를 다르게 쪼갰다.

이유는 이 `tokenizer.model`이 Unigram이 아니라 SentencePiece **BPE** 모델(`trainer_spec.model_type == 2`)이기 때문이다. BPE 모델의 머지 순서는 직렬화된 모델이 아니라 학습 이력에 있고, 저장된 것은 조각 점수뿐이다. 그래서 transformers 변환기는 `SentencePieceExtractor`에서 조각 점수로부터 머지를 역산한다. 각 조각을 어휘 안의 두 조각으로 나누는 모든 분할을 기록한 뒤 점수로 정렬하는 방식인데, 이는 손실이 있고 복원된 머지 순서가 학습된 순서와 늘 같지는 않다.

**검토한 대안:**

| 선택지 | 장점 | 단점 |
|--------|------|------|
| transformers의 머지 역산을 다시 구현한다 | 기록된 오라클 트랜스크립트와 두 프롬프트 모두 바이트 단위로 일치 | 변환 아티팩트를 mlxcel에 못 박는다. 측정상 더 나쁘다(산문 샘플에서 73.44 대 79.54 nats) |
| **채택: 체크포인트 자신의 SentencePiece BPE를 쓴다** | 가중치와 함께 배포된 토크나이저이고, 모델 자신의 우도 기준으로도 더 낫다 | 한 프롬프트의 기록된 오라클 트랜스크립트와 더 이상 맞지 않아 게이트를 재실행한 오라클 기준으로 다시 서술해야 한다 |

**근거:** 가중치와 함께 오는 토크나이저가 그 가중치를 학습시킨 토크나이저다. 손실 있는 변환을 거쳐 만든 트랜스크립트는 그 변환에 대한 증거이지 mlxcel에 대한 증거가 아니다.

**트레이드오프:** 이 계열을 mlx-lm 실행과 비교하려는 사람은 두 가지를 알아야 한다. mlx-lm은 `trust_remote_code`나 재라벨 없이는 이 체크포인트를 로드하지 못하고, 재라벨은 토크나이저를 조용히 바꾼다. 대신 무엇을 비교해야 하는지는 부록 B에 적었다.

---

### 3.4 리뷰에서 바뀐 것

첫 리비전의 슬라이딩 윈도 거부는 윈도 크기를 `Value::as_u64`로, 스위치를 `Value::as_bool`로 읽었다. 벤더는 `is not None`과 파이썬 truthiness로 판정하므로 `"sliding_window": 4096.0`, `"sliding_window": "4096"`, `"use_sliding_window": 1`이 모두 그쪽에서는 살아 있는 값인데, 이쪽에서는 셋 다 `None`으로 읽혀 full attention 디코드로 통과했다. 가드가 막으려던 바로 그 결과가 체크 누락이 아니라 JSON 표기를 통해 들어온 셈이다. 두 키 모두 이제 존재 여부로 판정한다. 같은 성격으로, `num_hidden_layers`를 읽지 못하면 `0`으로 떨어져 `max_window_layers < num_layers` 비교가 거짓이 되면서 가드가 열렸다. 지금은 `u64::MAX`로 떨어지므로 레이어 수를 모르면 "어떤 레이어는 윈도에 도달한다"고 가정한다.

더 큰 발견은 가드가 가중치가 아니라 라벨에 붙어 있었다는 점이다. `"model_type": "llama"`로 재라벨한 체크포인트는 평범한 Llama arm으로 가서 두 거부를 모두 건너뛰었다. 이 계열에서 재라벨은 가정이 아니다. `auto_map` 코드를 실행하지 않는 스택이 이 체크포인트를 로드하게 만드는 표준적인 방법이고, 이 PR의 오라클 실행도 그렇게 했다. 이제 `llama`와 `mistral` arm은 `architectures`가 `IQuestCoderForCausalLM`을 담고 있으면 같은 분류기로 위임한다. 두 경로가 같은 디코더에 도달하므로 라우팅은 달라지지 않는다. 달라지는 것은 거부 조건이 가중치를 따라다닌다는 점이다.

리뷰는 `disable_add_dummy_prefix`가 메시지의 모든 필드에 대해 span을 모으고 있다는 점도 짚었다. 실제 76800조각 모델에서는 약 3.7 MB의 일시 할당이고, 최소 폭 필드로 채운 악의적 파일에서는 입력 1바이트당 약 24바이트다. 지금도 메시지 전체를 검증하되 편집 대상 필드만 모은다.

---

## 4. 구현 상세

### 4.1 변경 지점

```
[변경 전]
config.json model_type "iquestcoder"
  -> get_model_type()  ->  Err("Unsupported model type: iquestcoder")

tokenizer.model  ->  SentencePieceProcessor::open()  ->  add_dummy_prefix 켬

[변경 후]
config.json model_type "iquestcoder"
  -> get_model_type() -> iquest_coder_model_type()
       clip_qkv 가 null 이 아님?          -> Err (이유를 밝힌 메시지)
       슬라이딩 윈도가 실제 레이어에 걸림?  -> Err (이유를 밝힌 메시지)
       그 외                              -> ModelType::IQuestCoder
  -> model_metadata 레지스트리 -> Llama3Model::load -> LoadedModel::Llama

tokenizer_config.json add_prefix_space == false
  -> spm_proto::disable_add_dummy_prefix(tokenizer.model 바이트)
  -> SentencePieceProcessor::from_serialized_proto()
```

### 4.2 핵심 코드 변경

**파일: `src/tokenizer/mod.rs`**

```rust
// 변경 전
let processor = SentencePieceProcessor::open(&tokenizer_model_path)
    .map_err(|e| anyhow::anyhow!("Failed to load tokenizer.model: {}", e))?;
let (special_tokens, added_token_contents, add_bos) = parse_special_tokens(model_path);

// 변경 후
let config = parse_special_tokens(model_path);
let processor = open_sentencepiece_processor(&tokenizer_model_path, config.add_prefix_space)?;
```

`add_prefix_space`는 `bool`이 아니라 `Option<bool>`이다. "없음"과 "명시적 false"는 다른 지시이기 때문이다. 없으면 SentencePiece 모델 자신의 설정을 그대로 두고, 명시적 `false`일 때만 덮어쓴다. 불리언이 아닌 표기는 없음으로 읽는다. 다른 체크포인트가 움직이지 않는 이유가 이것이다.

**파일: `src/models/detection.rs`**

거부 조건은 벤더보다 엄격하지 않고 벤더의 활성화 조건을 그대로 옮겼다. 스위치가 꺼진 채 `sliding_window` 크기만 있으면 통과하고, `max_window_layers`가 마지막 레이어를 넘어서도 통과한다. 두 경우 모두 벤더 쪽에서도 실제로 윈도가 걸리는 레이어가 없기 때문이다.

### 4.3 데이터 모델 변경

없다. `llama3::ModelArgs`가 이 계열의 config를 그대로 파싱한다.

---

## 5. 학습 포인트

### 5.1 `add_dummy_prefix`와 `add_prefix_space`는 파일 두 개에 적힌 같은 스위치다

**개념:** SentencePiece 모델은 `normalizer_spec.add_dummy_prefix`(기본 true)를 들고 다닌다. 이스케이프 전에 공백을 붙여 첫 단어도 뒤 단어들과 같은 `U+2581` 표식을 얻게 하는 설정이다. HuggingFace는 같은 결정을 `tokenizer_config.json`의 `add_prefix_space`로 노출하고, 변환기는 이 키가 true일 때만 `Prepend` normalizer를 넣는 방식으로 존중한다.

**이 PR에서의 적용:** 두 파일을 다 배포하는 체크포인트는 자기 자신과 어긋날 수 있고 이 계열이 그렇다. 배포자의 의도를 담은 쪽은 `tokenizer_config.json`이다. HuggingFace 기반 서빙 스택이 읽는 파일이 그쪽이기 때문이다. 이제 mlxcel도 읽는다.

**흔한 적용 사례:**
- `tokenizer.model`은 있고 `tokenizer.json`은 없는 모든 체크포인트. 참조 구현과 그리디 비교를 하기 전에 `grep add_prefix_space tokenizer_config.json`부터 해 볼 것.
- 채팅 템플릿 프롬프트. 효과가 배가된다. 프롬프트당 하나가 아니라 특수 토큰 사이 조각마다 유령 공백이 하나씩 붙는다.

### 5.2 SentencePiece BPE 모델은 HuggingFace 변환기를 왕복하지 못한다

**개념:** `trainer_spec.model_type`은 Unigram이면 1, BPE면 2다. Unigram 모델은 알고리즘에 필요한 것(조각과 점수)을 전부 직렬화한다. BPE 모델은 그러지 못한다. 머지 순서는 학습 상태이고 직렬화된 모델에는 조각 점수만 있다. 그래서 변환기는 머지를 추측한다.

**이 PR에서의 적용:** 검증 프롬프트 하나의 기록된 오라클이 그 추측을 거쳐 만들어졌고, 모델 자신의 토크나이저와 단어를 다르게 쪼갰다. 어느 쪽을 모델이 선호하는지는 teacher-forced 우도가 정리해 줬다.

**확인 코드:**

```python
from sentencepiece import sentencepiece_model_pb2 as pb2
m = pb2.ModelProto(); m.ParseFromString(open("tokenizer.model", "rb").read())
print(m.trainer_spec.model_type)   # 2면 BPE. 변환된 fast 토크나이저가 이 파일과 다를 수 있다
```

### 5.3 모델이 결정을 못 내리는 자리에서는 그리디 트랜스크립트가 패리티 게이트가 되지 못한다

**개념:** 그리디 디코딩이 구현 간에 재현되려면 top-1과 top-2의 로짓 차이가 두 구현 사이의 수치 잡음보다 커야 한다. bf16에서 로짓 64 근처의 인접 표현 가능값 간격은 0.25이므로, 0.25 차이는 딱 한 칸이다.

**이 PR에서의 적용:** Raft 원문 프롬프트의 0번 위치에서 mlx-lm의 `generate_step`과 같은 모델 객체에 대한 eager forward가 서로 다른 토큰을 고른다. 차이가 정확히 0.25인 자리다. 기록된 트랜스크립트 하나와만 비교했다면 여기서 mlxcel을 틀렸다고 판정했을 것이다. 실제로 판별력이 있는 측정은 부록 B에 있다. 참조 구현을 자기 자신과 먼저 비교한 뒤, 그 결과에 상대적으로 mlxcel의 일치도를 읽는다.

---

## 6. 추가 학습

### 핵심 용어

| 키워드 | 설명 | 이 PR에서의 의미 |
|--------|------|------------------|
| `add_dummy_prefix` | SentencePiece normalizer 플래그. 이스케이프 전에 공백을 붙인다 | 이 PR이 재작성하는 필드 |
| `add_prefix_space` | 같은 결정의 HuggingFace `tokenizer_config.json` 표기 | 무시되고 있던 키 |
| `SentencePieceExtractor` | 조각 점수로부터 BPE 머지를 역산하는 transformers 헬퍼 | 참조 토크나이저가 모델 자신의 것과 어긋날 수 있는 이유 |
| `clip_qkv` | 어텐션 전에 Q, K, V에 거는 텐서 단위 클램프 | 거부하는 두 스위치 중 하나 |
| `max_window_layers` | 슬라이딩 윈도가 적용되기 시작하는 레이어 인덱스 | 윈도 설정이 있어도 비활성일 수 있는 이유 |
| `StageFamily` | 파이프라인 병렬 스테이지 로더 식별자 | 새 variant에 PP를 부여한 지점 |

### 관련 기술

- **SentencePiece**: 모델 포맷과 normalizer spec, https://github.com/google/sentencepiece/blob/master/src/sentencepiece_model.proto
- **transformers slow-to-fast 변환**: https://github.com/huggingface/transformers/blob/main/src/transformers/convert_slow_tokenizer.py
- **체크포인트의 벤더 디코더**: https://huggingface.co/mlx-community/IQuest-Coder-V1-7B-Instruct-8bit/blob/main/modeling_iquestcoder.py

### 관련 PR과 이슈

- Issue #1357: 이 PR이 닫는 요청.
- PR #1593: 하루 앞서 기존 디코더 위에 `ModelType::YoutuLLM`을 올렸다. 전용 variant의 선례이자, 여기서 쓴 "오라클을 자기 자신과 먼저 비교한다" 방법의 출처.

---

## 7. 변경 요약

### 통계

| 항목 | 값 |
|------|-----|
| 변경 파일 | 13 |
| 추가 줄 | +1819 |
| 삭제 줄 | -11 |
| 추가 테스트 | 22 |

### 분류별 변경

| 분류 | 개수 | 요약 |
|------|------|------|
| 모델 라우팅 | 5개 파일 | `ModelType::IQuestCoder`, 레지스트리 항목, 거부 둘을 포함한 detection arm, 분산 디스패치 두 곳 |
| 토크나이저 정확성 | 3개 파일 | `add_prefix_space`를 읽어 `ModelProto` 재작성으로 반영 |
| 테스트 | 3개 파일 | 22개. 그중 2개는 pinned 게이트 뒤에서 실제 체크포인트를 읽는다 |
| 문서 | 1개 파일 | `docs/supported-models.md`의 계열 항목. 두 거부와 분산 관련 단서 포함 |

---

## 8. 후속 조치

### 필수

- [ ] 차단 사항 없음.

### 관찰 필요

- `"add_prefix_space": false`를 쓰는 SentencePiece 체크포인트가 추가되면 그 체크포인트의 그리디 비교를 다시 돌려야 한다. 이 PR 이후로는 이전과 다르게 토크나이즈된다. 현재 로컬 모델 집합에는 그런 체크포인트가 없다.
- `ModelProto` 재작성 실패 시 나오는 경고는 모델 경로를 함께 찍는다. 로그에 이 경고가 보인다면 어떤 체크포인트의 `tokenizer.model`이 파싱되지 않았다는 뜻이므로 무시하지 말고 확인할 것.

### 향후 개선

- SentencePiece 인코딩 경로는 `special: true`로 표시된 토큰에서만 분할한다. 이 계열은 특수 표시가 없는 추가 토큰(`<think>`, `<tool_call>`, `<tool_response>`와 닫는 짝)도 배포하는데, HuggingFace는 인코딩 시 이들을 원자 단위로 다루고 mlxcel은 현재 다시 쪼갠다. 첫 턴 생성이 아니라 이미 그 표식을 포함한 assistant 턴을 재인코딩할 때 영향을 주며, 모든 SentencePiece 체크포인트에 대해 이 PR 이전부터 있던 동작이다. 모델 경로 변경에 끼워 넣지 말고 별도 이슈로 다루는 편이 낫다.
- 이 계열의 텐서 병렬은 검증이 아니라 거부 상태다. 열려면 `is_llama_style_architecture`에 `iquestcoder`를 넣고 다중 랭크 호스트에서 측정해야 한다.

---

## 부록

### A. 테스트 결과

```
cargo test --profile test-fast --features metal,accelerate --lib -- \
  tokenizer:: models::detection_tests models::metadata_tests model_metadata_tests \
  loading::tests distributed::pipeline::stage_executor
  -> 261 passed, 0 failed, 5 ignored

cargo test --profile test-fast --features metal,accelerate --bin mlxcel -- \
  family_order all_model_types supported_models arch
  -> 11 passed, 0 failed

cargo clippy --profile test-fast --lib --tests --features metal,accelerate -- -D warnings
cargo fmt --all -- --check
python3 scripts/ci/check_cross_repo_refs.py
  -> 모두 통과
```

실제 체크포인트를 읽는 두 테스트는 `models/mlx/iquest-coder-v1-7b-instruct-8bit`를 열고 `crate::test_support::pinned_checkpoint::skip_or_fail_pinned_checkpoint`를 거치므로, 체크포인트가 없으면 skip하고 `MLXCEL_REQUIRE_PINNED_CHECKPOINTS=1`에서는 실패한다.

### B. 패리티 측정

mlx-lm 0.31.3, MLX 0.32.2. mlx-lm이 로드할 수 있도록 같은 8비트 가중치를 심볼릭 링크 디렉터리에서 `llama`로 재라벨했다. temperature 0 그리디, 32토큰, 양쪽에 동일한 프롬프트 id를 넣었다. `gs`는 `mlx_lm.generate.generate_step`, `eager`는 같은 모델 객체에 대해 매 스텝 `model(prefix)` argmax를 도는 루프다. 칸의 값은 공통 토큰 접두사의 길이다.

| 프롬프트 | `gs` 대 `eager` | mlxcel 대 `gs` | mlxcel 대 `eager` | top-2 차이 중앙값 | 0.5 이내 위치 |
|---|---|---|---|---|---|
| raw: `The Fibonacci sequence begins with` | 32/32 | **32/32** | **32/32** | 3.00 | 19% |
| raw: `In distributed systems, ... such as Raft` | 0/32 | 0/32 | 13/32 | 1.00 | 44% |
| chat: 회문 판별 함수 | 32/32 | **32/32** | **32/32** | 13.00 | 6% |
| chat: 해시 맵 설명 | 10/32 | 10/32 | 22/32 | 3.50 | 19% |
| chat: 이진 탐색 복잡도 | 25/32 | **32/32** | 25/32 | 3.25 | 9% |

첫 열부터 읽어야 한다. mlx-lm이 32토큰 내내 자기 자신과 일치하는 두 프롬프트에서는 mlxcel이 양쪽 경로 모두와 토큰 단위로 일치한다. mlx-lm의 두 평가 경로가 갈리는 곳에서는 mlxcel이 둘 중 하나와, 적어도 그 둘이 서로 일치하는 만큼은 일치한다. Raft 원문 프롬프트의 첫 생성 위치에서 eager forward는 `▁or` 63.75, `▁and` 63.50을 주고 `generate_step`은 `▁and`를 고른다.

### C. 측정 재현

```bash
# 네이티브 경로에서의 탐지와 생성
./target/release/mlxcel arch | head -8
./target/release/mlxcel generate -m models/mlx/iquest-coder-v1-7b-instruct-8bit \
  -p "The Fibonacci sequence begins with" -n 32 --no-chat-template --temp 0

# 가중치를 전혀 로드하지 않고 토크나이저 주장만 확인
python3 - <<'PY'
import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as pb2
raw = open("models/mlx/iquest-coder-v1-7b-instruct-8bit/tokenizer.model", "rb").read()
plain = spm.SentencePieceProcessor(model_proto=raw)
m = pb2.ModelProto(); m.ParseFromString(raw); m.normalizer_spec.add_dummy_prefix = False
patched = spm.SentencePieceProcessor(model_proto=m.SerializeToString())
t = "The Fibonacci sequence begins with"
print(plain.encode(t, out_type=str))    # 첫 단어에 워드 경계 표식이 붙은 형태
print(patched.encode(t, out_type=str))  # ['The', ...] 이제 mlxcel이 내는 결과
PY
```

오라클 쪽은 mlx-lm 환경과 함께, `config.json`의 `model_type`이 `"llama"`이고 `auto_map`을 지운 체크포인트 사본이 필요하다. mlx-lm은 `iquestcoder` 라벨을 로드하지 못한다. 다만 재라벨은 transformers가 만드는 토크나이저까지 바꾼다는 점에 주의할 것. 부록 B의 실행은 각자 프롬프트를 토크나이즈하게 두지 않고 양쪽에 같은 id를 명시적으로 넣었다.
