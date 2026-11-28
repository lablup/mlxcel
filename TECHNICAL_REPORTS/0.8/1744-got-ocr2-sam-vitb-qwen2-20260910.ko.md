# 기술 보고서: PR #1744 - feat(got): port GOT-OCR 2.0 (SAM ViT-B + Qwen2-0.5B)

**날짜**: 2026-09-10
**작성**: mlxcel 메인테이너
**검토**: 구현 및 검증 사이클
**상태**: 완료
**언어**: Rust, Markdown
**위험도**: 중간 (기존 타워와 디코더를 재사용한 새 모델 계열, 다른 계열까지 닿는 공유 토크나이저 변경 1건, 배포된 두 키 레이아웃 모두에서 체크포인트 자신의 레퍼런스와 greedy 토큰 단위 일치 검증)

---

## 요약

PR #1744는 `model_type: "GOT"`를 추가한다. `stepfun-ai/GOT-OCR2_0`으로 배포되고 `mlx-community/GOT-OCR2_0-{bf16,8bit,4bit}`로 변환된 0.58B 문서 OCR VLM이다. 아키텍처 자체에 새로운 것은 거의 없다. SAM 계열 ViT-B 타워는 DeepSeek-OCR의 것을 그대로 재사용하고, 프로젝터는 `Linear(1024, 1024)` 하나이며, 디코더는 Llama 계열 백본 위의 Qwen2-0.5B다. 실제로 포팅한 것은 그 주변의 다섯 가지이고, 그중 넷은 틀려도 조용히 틀린다. tiktoken 로더가 HunYuan 특수 토큰 표를 고정해 두고 있었는데, 그 표에서는 `<|im_end|>`와 `<imgpad>`에 대응하는 id가 아예 없어 정지 토큰과 이미지 자리표시자가 둘 다 사라진다. 배포된 두 키 레이아웃은 모든 접두사와 타워 넥의 이름에서 어긋난다. 정지 집합에는 151645가 들어가야 하는데 체크포인트의 어떤 설정 파일도 그 값을 적어 두지 않았고, 없으면 생성이 종료되지 않고 토큰 상한까지 간다. 체크포인트에 채팅 템플릿이 없어 `modeling_GOT.py::chat`이 만드는 고정 대화를 양쪽 프런트엔드 모두에서 mlxcel이 들고 있어야 한다. 그리고 런타임이 시퀀스 상태 레이아웃을 트레이트 기본값에 맡기지 않고 직접 밝혀야 하는데, 기본값은 서버 스케줄러에 빈 캐시 벡터를 넘긴다.

검증은 체크포인트 자신의 파일들로 조립한 레퍼런스에 대한 greedy 토큰 단위 일치다. 저장소에 커밋한 렌더링 페이지에서 bf16 변환본과 원본 `stepfun-ai` 레이아웃 모두 레퍼런스의 프롬프트 토큰 287개와 생성 id 11개를 그대로 재현하고 151645에서 멈춘다. 종료는 추론이 아니라 차분 빌드로 보였다. 새 테스트 36개, 그리고 건드린 공유 파일의 회귀 범위에서 517개 통과.

---

## 1. 문제 정의

### 1.1 배경

이슈 #1359는 GOT-OCR 2.0(`model_type: "GOT"`, `architectures: ["GOTQwenForCausalLM"]`)을 요청했다. 스택은 짧다. SAM 계열 ViT-B 타워(1024x1024 입력, 윈도 어텐션 14, 전역 블록 2/5/8/11, 분해된 상대 위치 바이어스, 2단 conv 넥, stride-2 압축 conv 2개)가 페이지를 폭 1024의 16x16 격자로 만드니 특징 행이 정확히 256개다. `Linear(1024, 1024)` 하나가 그것을 투영한다. Qwen2-0.5B 디코더(24층, 폭 1024, q/k/v 바이어스, 임베딩 공유)가 읽는다.

이 조각들은 mlxcel에 이미 전부 있었다. `src/vision/encoders/deepseekocr_sam.rs`의 `SamEncoder`가 바로 그 타워다. `got_vision_b.py`와 mlx-vlm의 `sam.py`는 기하 구조, 넥 LayerNorm의 `1e-6` 엡실론, 바이어스 없는 `net_2` / `net_3` 압축기까지 일치하므로 `SamConfig::default()`가 GOT를 정확히 기술한다. `src/models/qwen2.rs`는 어텐션 바이어스와 임베딩 공유를 존중하는 Llama 계열 백본을 재수출한다. 융합은 `merge_llava`다. 그러니 이슈의 방향이 맞다. 지목된 조각은 재사용하고, 진짜 새로운 것만 구현한다.

### 1.2 진짜 새로운 것, 그리고 각각이 위험한 이유

- **QWen tiktoken 특수 토큰 표.** `TiktokenTokenizer::from_file`은 표를 정확히 하나, HunYuan의 것만 만들었다. `<|endoftext|>, <|startoftext|>, <|bos|>, <|eos|>, <|pad|>, <|extra_0..204|>`를 `encoder.len()`부터 번호 매기고, `tokenizer_config.json`의 `added_tokens_decoder`로 덮어쓴다. GOT의 `added_tokens_decoder`는 **비어 있다**. 덮어쓸 것이 없다. HunYuan 표에서는 `<|im_end|>`와 `<imgpad>`가 어휘에 존재하지도 않으므로, 정지 토큰을 샘플링할 수 없고 256토큰 이미지 블록은 그냥 글자로 토크나이즈된다.
- **두 개의 키 레이아웃.** 원본은 `model.vision_tower_high.*`(넥은 `nn.Sequential` 인덱스 `neck.0..3`), `model.mm_projector_vary.*`, 디코더는 `model.*`, 그리고 공유된 `lm_head.weight` 사본을 싣는다. MLX 변환본은 `vision_tower.*`(넥은 `conv1 / norm1 / conv2 / norm2`로 개명), `multi_modal_projector.*`, `language_model.model.*`을 싣고 `lm_head`는 없다.
- **정지 집합.** `config.json`은 `eos_token_id: 151643` 하나만 적는다. OCR 답변은 `<|endoftext|>`를 내지 않는다. 업스트림은 `KeywordsStoppingCriteria`로 `<|im_end|>` 턴 구분자에서 멈추는데, 이는 설정 파일에 표현이 전혀 없는 생성 루프 구조물이다.
- **프롬프트.** mlxcel이 읽는 세 소스 어디에도 채팅 템플릿이 없다. 대화는 `modeling_GOT.py::chat` 안에서 만들어지는 상수다.
- **서버 경로.** VLM은 서버가 디코더 층을 0개 돌리는 동안에도 CLI 테스트를 전부 통과할 수 있다. CLI는 자기 캐시를 직접 만들기 때문이다(Falcon-OCR 선례, PR #1075).

### 1.3 이슈가 경고한 함정을 믿지 않고 확인한 방법

이슈 본문은 `<ref>` 151851부터 `<imgpad>` 151859까지의 특수 토큰 표를 제시했다. 그 숫자를 그대로 믿지 않았다. 체크포인트의 `qwen.tiktoken`(151643 랭크)과 `tokenization_qwen.py`의 순서(`SPECIAL_TOKENS = (ENDOFTEXT, IMSTART, IMEND) + EXTRAS`, 이어서 `IMAGE_ST`)에서 `len(mergeable_ranks)`부터 번호를 다시 매겨 유도했다. 결과는 151643 + 3 + 205 + 9 = 151860이고 이는 `config.json`이 적은 `vocab_size`이며, `<img>` / `</img>` / `<imgpad>`는 151857 / 151858 / 151859로 같은 파일의 `im_start_token` / `im_end_token` / `im_patch_token`과 일치한다. 서로 독립적인 세 진술이 맞아떨어지는 것이 이 표를 신뢰할 근거이지, 이슈가 그렇게 적었다는 사실이 근거가 아니다.

---

## 2. 기술 검토

### 2.1 토크나이저 표, 그리고 HunYuan을 그대로 두기

`src/tokenizer/tiktoken.rs`는 이제 체크포인트에서 표를 고른다. `declares_qwen_tokenizer_class`는 `tokenizer_class == "QWenTokenizer"`와 `auto_map.AutoTokenizer` 항목 양쪽을 받는다. `auto_map`만 남긴 변환본도 QWen 표에 닿아야 하기 때문이고, GOT는 둘 다 적는다. `build_qwen_special_token_list`는 `tokenization_qwen.py`의 순서를 그대로 재현한다. `added_tokens_decoder` 덮어쓰기는 어느 표 뒤에서든 그대로 돈다.

이것이 새 계열 바깥까지 닿는 유일한 공유 파일 변경이므로 HunYuan 경로에 자체 게이트를 뒀다. `hunyuan_table_unchanged_without_qwen_class`는 이름 붙은 특수 토큰 5개와 `<|extra_0|>`가 늘 있던 자리에 있고 QWen 전용 표기는 아무 id로도 풀리지 않음을 확인한다. 오프셋을 눈으로 검산할 수 있는 3랭크 합성 어휘 위에서 한다. QWen 게이트는 그 거울상이다. 같은 어휘에서 `<|im_end|>`가 5, `<imgpad>`가 219이며, 이는 GOT의 어휘에서 두 값을 151645와 151859에 놓는 것과 같은 오프셋이다.

### 2.2 정규화기

`src/loading/vlm_got_ocr.rs`의 `canonicalize_got_keys`가 두 레이아웃을 한 이름 체계로 접는다. 타워 넥은 **원본** 표기를 목표로 삼아 변환본의 `conv1 / norm1 / conv2 / norm2`를 `neck.0..3`으로 되돌린다. `SamEncoder::from_weights`가 DeepSeek-OCR을 위해 이미 그 인덱스를 읽기 때문이다. 다른 계열이 의존하는 공유 인코더에 넥 키 매개변수를 끼워 넣는 것보다 로더에서 키 넷을 고쳐 쓰는 편이 작다.

두 가지 성질이 중요하다. 함수는 멱등이다. 변환본의 타워/프로젝터 접두사가 **곧** 정규 이름이므로 두 번째 통과는 그것들을 건드리지 않아야 하고, 예컨대 `language_model.model.*`을 `language_model.language_model.model.*`로 다시 접두사 붙이면 안 된다. 그리고 `blocks.N.*` 안의 블록별 `norm1` / `norm2`는 넥 개명에서 살아남아야 하므로, 개명은 부분 문자열이 아니라 타워 접두사에 묶어 적용한다. 둘 다 두 체크포인트의 safetensors 헤더에서 읽어낸 실제 키 목록으로 테스트한다.

원본이 싣는 공유 `lm_head.weight` 사본(151860 x 1024 bf16, 약 311 MB)은 `tie_word_embeddings`가 켜져 있을 때 버린다. 백본이 헤드를 임베딩에 직접 묶으므로 사본은 읽히지 않는다. 남겨 두면 상주 메모리만 쓰고, 사본이 어긋났을 때 조용히 그쪽이 이긴다. 버리기는 가정이 아니라 설정 플래그에 걸려 있고, 양쪽 다 테스트한다.

conv 레이아웃에는 새 코드가 필요 없다. `SamEncoder`의 공유 `conv_channels_last` 형상 게이트가 차이를 흡수한다. conv 네 경우 모두 올바르게 통과한다. `[256,768,1,1]`과 `[256,256,3,3]`은 전치되고 `[256,1,1,768]`과 `[256,3,3,256]`은 그대로 있다.

### 2.3 디코더 설정

`text_config`도 `language_config`도 없다. 텍스트 설정이 **곧** 평평한 루트다. `got_text_config`는 루트를 복제해 `model_type`을 `"GOT"`에서 `"qwen2"`로 고쳐 쓰고 `vision_config`와 `quantization_config`를 지운다. 루트의 `quantization` 블록은 일부러 남겨 변환본이 그룹 크기와 비트 폭을 백본까지 가져가게 하고, 테스트는 결과를 `llama3::ModelArgs`로 파싱해 `group_size() == 64`와 `bits() == 4`를 확인한다. serde가 블록을 봤으리라고 믿는 대신이다.

`attention_bias`는 강제하지 않는다. 누락이 아니다. GOT의 `config.json`은 그 키를 적지 않지만, `FusedQKVLinear::from_weights_separate`가 q/k/v 선형 바이어스를 플래그가 아니라 가중치 맵에서 감지한다. 트리 안의 다른 Qwen2 체크포인트가 이미 그렇게 로드된다.

### 2.4 정지 집합

`got_eos_token_ids`는 `generation_config.json`과 `config.json`에서 `eos_token_id`를 스칼라와 리스트 양쪽으로 읽고, 그 뒤에 151645를 중복 없이 무조건 더한다. 상수는 이유를 doc 주석에 달고 있다. 체크포인트만 보는 미래의 독자에게는 그 값을 넣을 근거가 어디에도 없기 때문이다.

### 2.5 프롬프트, 양쪽 프런트엔드에서

대화는 고정되어 있다.

```
<|im_start|>system
        You should follow the instructions carefully and explain your answers in detail.<|im_end|><|im_start|>user
<img><imgpad>x256</img>
OCR: <|im_end|><|im_start|>assistant
```

중요하면서 잃기 쉬운 지점이 셋이다. 시스템 줄은 공백 여덟 칸을 문자 그대로 싣는데, 이는 메서드 본문 안 삼중 따옴표 리터럴의 파이썬 소스 들여쓰기다. `<|im_end|>`는 뒤에 줄바꿈 없이 시스템 턴을 닫는다. 그리고 이미지 블록은 평범한 텍스트로 실려 간다. QWen 표가 선택되고 나면 `<img>` / `<imgpad>` / `</img>`가 실제 어휘 항목이기 때문이다.

그 대화는 두 곳에 있고, 둘은 일치해야 한다. `src/server/chat_template.rs`의 `GOT_OCR_CHAT_TEMPLATE`이 서버가(그리고 같은 이음매를 쓰므로 CLI도) 렌더링하는 내장 템플릿이고, `src/multimodal/got_ocr_prompt.rs`의 `build_got_prompt`가 이미지 블록을 끼우고 아직 감싸이지 않은 텍스트를 감싼다. 내장 템플릿은 일부러 블록을 렌더링하지 않는다. 자리표시자 256개가 타워의 특징 행과 맞아야 하므로, 트리의 다른 계열들처럼 토큰에 가까운 경로에서 끼운다.

배치 규칙은 순서대로 셋이다. 명시적 `<image>` 표시는 그 자리에서 치환된다. 그렇지 않고 이미 감싸인 텍스트라면 블록은 첫 `<|im_start|>user\n` 바로 뒤로 간다. 업스트림이 두는 자리다. 무작정 앞에 붙이면 **시스템** 턴보다 앞에 놓인다. 그것도 아니면 맨 지시문이므로 블록, 줄바꿈, 지시문 순으로 놓고 전체를 감싼다. 이미 감싸인 텍스트는 다시 감싸지 않으며, 그래서 서버 렌더와 맨 CLI 지시문이 시스템 턴 두 개를 겹치지 않고 하나의 프롬프트로 수렴한다.

내장 템플릿은 클라이언트가 보낸 `system` 메시지를 버리고 고정 시스템 턴을 낸다. 예의 문제가 아니다. 비전 특징은 모델이 모든 학습 예제에서 본 접두사를 가진 프롬프트 안으로 흩뿌려지는데, 거기에 호출자의 시스템 텍스트를 대신 넣으면 가리킬 오류 없이 OCR 정확도만 떨어진다.

### 2.6 서버 경로

`GotOcrVlModel`은 `sequence_state_layout`, `supports_batching`, `supports_batched_prefill`, `supports_maskless_padded_prefill`, `supports_paged_decode_backend`를 트레이트 기본값에 맡기지 않고 디코더에 위임한다. 기본값은 층별 시퀀스 상태가 없다고 보고하고, 서버 스케줄러는 빈 캐시 벡터를 받는다. 그러면 모델은 디코더 층을 0개 돌리고 어디에도 오류 없이 그럴듯한 헛소리를 반환하며, CLI는 `make_caches`로 자기 캐시를 만들기 때문에 이를 잡지 못한다.

`output_suppressed_token_ids`는 감싸기 id 셋을 돌려준다. `<imgpad>`를 샘플링하면 다음 턴의 흩뿌리기까지 어긋난다. 자리표시자 개수가 프롬프트와 타워의 행을 짝짓는 값이기 때문이다.

`ensure_image_token_feature_cardinality`가 런타임 분기에서 흩뿌리기를 지킨다. 토크나이저가 태그를 체크포인트의 id가 아닌 무언가로 풀면 자리표시자 개수가 0으로 무너지는데, 페이지를 조용히 무시하는 모델이 아니라 여기서 드러난다.

---

## 3. 레퍼런스

오라클은 `modeling_GOT.py` 대신 체크포인트 자신의 파일들로 조립했다. 그 파일은 transformers 4.37을 겨냥해 5.18에서는 임포트되지 않는다.

- 체크포인트 디렉터리에서 바로 불러온 `got_vision_b.py::build_GOT_vit_b()`. transformers 의존이 없는 순수 torch라 배포된 그대로 돈다.
- `model.mm_projector_vary.{weight,bias}`를 실은 `nn.Linear(1024, 1024)`.
- GOT 자신의 설정 필드로 만든 transformers `Qwen2Model`과 묶인 `lm_head`. 오라클은 둘 중 하나를 쓰기 전에 `lm_head.weight`와 `model.embed_tokens.weight`가 실제로 같은지 단언한다.
- torchvision으로 재현한 `GOTImageEvalProcessor`의 변환 셋: `Resize((1024, 1024), BICUBIC)`, `ToTensor()`, `Normalize(CLIP mean, CLIP std)`.
- `tokenization_qwen.py` 계약: `qwen.tiktoken`의 151643 랭크 위에 `SPECIAL_TOKENS + IMAGE_ST`를 `len(mergeable_ranks)`부터 번호 매긴 `tiktoken.Encoding`.
- `modeling_GOT.py::forward`의 끼워넣기: `cnn_feature.flatten(2).permute(0, 2, 1)` 뒤에 `<img>` 위치에서 `cat(embeds[:pos+1], features, embeds[pos+num_patches+1:])`, 그리고 그 구간 뒤에 정말 `</img>`가 오는지 단언.

torch 2.14.0에서 CPU fp32로 돌았으므로, 검증 대상은 greedy 토큰 단위 일치이지 비트 단위 일치가 아니다.

이 구성이 덤으로 증명하는 것이 하나 있다. mlxcel이 타워의 `(1, 16, 16, 1024)` 격자를 channels-last로 `(1, 256, 1024)`로 reshape한 결과의 행 순서가 레퍼런스의 NCHW `flatten(2).permute(0, 2, 1)`과 같다는 것이다. 둘 다 `h * 16 + w` 순이다. 여기를 틀리면 디코더 아래에서 페이지가 전치되는데, 오류가 아니라 그럴듯하지만 틀린 텍스트로 나타나므로 자체 게이트를 뒀다.

---

## 4. 검증

Linux aarch64, NVIDIA GB10, CUDA sm_121, `--profile test-fast --features cuda`. 체크포인트: `models/mlx/got-ocr2_0-bf16`, `models/mlx/got-ocr2_0-4bit`, `models/mlx/got-ocr2_0-original`.

### 4.1 실제 파일에 대한 특수 토큰 id

`qwen.tiktoken`(151643 랭크, 최대 랭크 151642)과 `tokenization_qwen.py`의 순서에서 다시 유도: `<|endoftext|>` 151643, `<|im_start|>` 151644, `<|im_end|>` 151645, `<|extra_0..204|>` 151646..151850, `<ref>` 151851부터 `<imgpad>` 151859까지, 합계 151860으로 선언된 `vocab_size`와 같다. 세 체크포인트 모두 `tokenizer_config.json`에 `tokenizer_class: "QWenTokenizer"`와 빈 `added_tokens_decoder`를 적는다. `tests/got_ocr_prompt_parity.rs`는 mlxcel이 로드한 토크나이저가 실제 체크포인트에서 감싸기 표기 여섯 개를 그 id로 푸는지 단언한다.

### 4.1b tiktoken 경로 위의 다른 계열

이 호스트에서 GOT 외에 tiktoken 로더에 닿는 체크포인트 디렉터리는 셋뿐이다. `hunyuan-13b`와 `hunyuan-a13b-instruct-4bit`는 `HYTokenizer`를, `phi-3-small-8k-instruct-aq4_64`는 `Phi3SmallTokenizer`를 적는다. 어느 것도 새 분기에 들어갈 수 없다. 변경 후 `hunyuan-13b`를 끝까지 돌려 이전과 같이 `The capital of France is Paris.`를 답하는 것을 확인했다. HunYuan 표에 건 단위 게이트가 할 수 없는 공유 파일 회귀 확인이다.

### 4.2 greedy 일치, 두 키 레이아웃

커밋된 고정 페이지에서 `tests/got_ocr_real_model.rs`:

| | 프롬프트 토큰 | `<imgpad>` | 생성 id | 종결자 |
|---|---|---|---|---|
| 레퍼런스 (CPU fp32) | 287 | 256 | 38 1793 80577 1378 1459 7168 715 54159 419 2150 198 | 151645 |
| `got-ocr2_0-bf16` | 287 | 256 | 동일 | 151645 |
| `got-ocr2_0-original` | 287 | 256 | 동일 | 151645 |

둘 다 `"GOT OCR two point zero \nrenders this page"`로 디코딩된다. 테스트는 64토큰 상한 안에서 엄격히 종료했는지, 중간에 `<|endoftext|>`를 내지 않았는지, `make_caches`가 비어 있지 않은 층별 캐시를 돌려줬는지도 단언한다.

프롬프트는 개수가 아니라 id로 못 박는다. `got_prompt_ids_match_the_reference_tokenizer`가 22토큰 머리, 9토큰 꼬리, 길이, 그리고 `<img>`와 `</img>` 사이가 정확히 256개의 균일한 자리표시자인지를 레퍼런스 토크나이저가 만든 벡터와 비교한다.

### 4.3 CLI, 세 체크포인트

렌더링 페이지에 `OCR: `, `-n 1024`:

| | 프롬프트 토큰 | 생성 | 출력 |
|---|---|---|---|
| `got-ocr2_0-bf16` | 287 | 7 | `GOT OCR two point zero` |
| `got-ocr2_0-4bit` | 287 | 7 | `GOT OCR two point zero` |
| `got-ocr2_0-original` | 287 | 7 | `GOT OCR two point zero` |

셋 다 그대로 옮겨 적고 1024 예산에 대해 7토큰에서 멈춘다. 세 줄짜리 페이지는 15토큰으로 `Hello world\nThe quick brown fox\njumps over the lazy dog`가 되어 레퍼런스와 정확히 일치한다. 레퍼런스의 끝 공백이 없는 `OCR:`도 프롬프트 286토큰으로 동작하고, `OCR with format: `는 289토큰으로 동작한다.

내장 렌더 대신 맨 지시문 감싸기를 태우는 `--no-chat-template`은 고정 페이지에서 동일한 287토큰 프롬프트와 동일한 11토큰 출력을 낸다. 두 분기가 단위 테스트뿐 아니라 실제 체크포인트에서도 수렴한다.

### 4.4 종료, 증명

이슈는 151645가 없으면 생성이 끝나지 않는다고 경고했다. 추론하지 않고 보였다. `push(&mut ids, IM_END_STOP_TOKEN_ID)` 줄을 뺀 차분 빌드로 같은 페이지에 `-n 64`:

```
OCR: GOT OCR two point zero
<|im_end|><|im_end|><|im_end|>... (57개 더)

[Generated 64 tokens]
```

모델은 배포 빌드가 멈추던 바로 그 자리에서 `<|im_end|>`를 내고, 상한까지 그것을 반복한다. `<|endoftext|>`에는 도달하지 않는다. 프로브는 되돌렸고 배포 빌드가 7토큰임을 다시 확인했다.

### 4.5 서버

bf16 변환본으로 `mlxcel-server`, 페이지를 `data:image/png;base64` content part로 넣고 `temperature: 0`으로 `/v1/chat/completions`:

| | prompt_tokens | completion_tokens | finish_reason | content |
|---|---|---|---|---|
| chat completions | 287 | 7 | `stop` | `GOT OCR two point zero` |

모든 필드가 CLI와 같다. 세 줄 페이지에 대한 동시 요청 둘(하나는 `OCR: `, 하나는 `OCR with format: `)이 모두 전체 전사를 반환했고(프롬프트 287/289, 완성 15/14), 이는 배치 디코드 경로를 태운다.

4비트 변환본도 같은 방식으로 서빙해 고정 페이지에서 프롬프트 287토큰, 완성 11토큰, `finish_reason: "stop"`, `GOT OCR two point zero\nrenders this page`를 반환했다. 토큰 수는 bf16과 같고 내용은 한 토큰 다르다(bf16은 줄바꿈 앞 공백을 유지한다). 이 저장소가 게이트 대상이 아니라 기록 대상으로 두는 양자화 드리프트다.

두 프런트엔드가 같은 프롬프트를 렌더링한다는 것은 주장이 아니라 못 박혀 있다. `server_render_and_cli_instruction_tokenize_identically`가 실제 `ChatTemplateProcessor`로 렌더링해 조립된 프롬프트의 바이트 일치와 모든 id의 일치를 단언한다. `prompt_tokens` 개수가 같은 것은 잘못된 렌더에서도 살아남기 때문이다.

### 4.6 게이트

- `cargo clippy --lib --tests --features cuda -- -D warnings`: 통과.
- `cargo fmt --all`: 적용, 잔여 diff 없음.
- `cargo check --lib --tests --features cuda`: 통과.
- 새 테스트, `--profile test-fast --features cuda`: `multimodal::got_ocr_prompt` 12/12, `loading::vlm::got_ocr` 7/7, `vision::got_ocr` 2/2, `vision::processors::got_ocr` 5/5, `server::got_chat_template_tests` 5/5, `tokenizer::tiktoken` 4/4, 탐지 테스트 1/1.
- 실제 체크포인트 게이트, `-- --ignored`: `tests/got_ocr_real_model.rs` 1/1, `tests/got_ocr_prompt_parity.rs` 3/3.
- 건드린 공유 파일의 회귀 범위: `server::chat_template` 154/154, `loading::vlm` 216/216, `multimodal::vlm_runtime` 50/50, `models::detection_tests` 61/61, `execution::memory_estimate` 43/43, `models::registry` 9/9, `model_metadata` 8/8, `vision::merge` 4/4, `cli_help_consistency` 27/27.

---

## 5. 검증하지 못한 것

- **`cargo test --workspace --profile test-fast --features metal,accelerate`는 실행하지 못했다.** 이슈가 인수 조건으로 적었지만 이 호스트는 CUDA를 쓰는 Linux aarch64다. 여기에는 Metal도 Accelerate도 없다. 위의 CUDA 등가물이 실제로 실행한 것이고 이 변경이 건드리는 모듈로 범위를 좁혔다. 전체 워크스페이스 스위트는 머지 게이트에 맡긴다.
- **범위를 지정하지 않은 `cargo test --lib --features cuda`는 이 호스트에서 게이트로 쓸 수 없고, 게이트로 삼지도 않았다.** `terminate called ... cudaStreamEndCapture(stream, &handle_) failed`로 테스트 프로세스 전체가 죽는다. 테스트 실패가 아니라 C++ abort다. 이 브랜치에서 연속 두 번 돌렸을 때 서로 무관한 지점에서 죽었다. 한 번은 `loading::vlm`에서 2184개 통과 후, 한 번은 `audio::phi4mm`에서 1366개 통과 후다. 그래프 캡처와 아무 상관 없는 브랜치들에서도 이 호스트가 보여 온 병렬 CUDA 그래프 캡처 시그니처다. 실제로 게이트로 삼은 것은 4.6의 범위 지정 스위트다. 나중에 전체 스위트를 돌리는 오케스트레이터는 같은 abort를 예상하고, 여기에 귀속하기 전에 베이스 브랜치를 대조 실행해야 한다.
- **4비트 변환본은 greedy 일치 게이트에 넣지 않았다.** CLI와 서버 양쪽에서 페이지를 정확히 옮겨 적고 양자화 블록도 `got_text_config`로 단위 테스트되지만, 레퍼런스 오라클은 원본 fp32 가중치로 돌므로 양자화된 greedy 일치는 의미 있는 비교가 아니고 주장하지 않았다. 고정 페이지에서 bf16과 한 토큰 다른 것은 게이트가 아니라 위에 기록해 뒀다. 8비트 변환본은 내려받지 않았고 전혀 돌리지 않았다.
- **사진 이미지는 오라클과 비교하지 않았다.** mlxcel의 bicubic 리사이즈는 `image`의 Catmull-Rom이고 업스트림은 PIL의 것이라, 사진은 리샘플된 픽셀이 조금 달라 greedy 토큰 일치를 기대할 수 없다. 위에서 쓴 고대비 렌더링 페이지는 필터에 거의 무관하고, 그래서 토큰 단위 비교의 대상이다.
- **세밀 영역 모드는 레퍼런스와 비교하지 않았다.** `[x1,y1,x2,y2] OCR with format: `과 `[red] OCR with format: `는 지시문으로 그대로 전달되며, 정답을 아는 영역이 있는 페이지로 태우지 않았다.
- **텍스트 전용 요청은 지원 모드가 아니고 동작하게 만들지 않았다.** 이미지가 없으면 런타임 분기가 아예 돌지 않고 모델은 쓰레기를 낸다. 레퍼런스 `chat()`이 항상 이미지를 넘기는 OCR 전용 체크포인트의 본질이다. 덮어 가리지 않고 동작을 그대로 뒀다.
- **다중 이미지는 구현이 아니라 거부다.** 디코더는 256토큰 블록 하나로 학습됐고 업스트림 `forward`는 `<img>` 하나당 `</img>` 하나를 단언하므로, 두 번째 이미지는 개수를 밝히며 프롬프트 조립 단계에서 거부된다.
- **다중 크롭 OCR과 모델 카드의 HTML 렌더 헬퍼는 범위 밖**이며, 이슈가 그렇게 적었다.

---

## 6. 교훈

- **이슈 본문의 숫자는 사실이 아니라 가설이다.** 이슈의 특수 토큰 표는 결과적으로 맞았지만, `qwen.tiktoken`의 랭크 수, `tokenization_qwen.py`의 순서, `config.json`의 `vocab_size`와 세 개의 `im_*_token` 필드가 독립적으로 맞아떨어진 뒤에야 믿을 만해졌다. 그 교차 확인은 1분이 들고, 검증된 id 표와 옮겨 적은 id 표를 가른다.
- **공유 토크나이저의 기본값은 그것을 쓰는 모든 계열과 맺은 조용한 계약이다.** HunYuan 표는 HunYuan에게 틀리지 않았다. 보편값으로서 틀렸다. 두 번째 표를 더하려면 첫 번째 표에 게이트가 필요했고, 오프셋을 눈으로 검산할 만큼 작은 합성 어휘 위여야 했다. 아니면 HunYuan 체크포인트 없이는 변경을 반증할 수 없었을 것이다.
- **어떤 동작은 레퍼런스의 설정이 아니라 생성 루프에만 있다.** `<|im_end|>` 정지는 어떤 설정 파일도 언급하지 않는 `KeywordsStoppingCriteria` 안에 산다. `config.json`과 `generation_config.json`만 읽었다면 끝나지 않는 모델이 나왔을 것이고, 그 실패는 정지 id 누락이 아니라 디코드 버그처럼 보였을 것이다.
- **한 줄을 지워 보는 것이 한 줄을 읽는 것보다 나은 증명이다.** 코드를 보고 "151645에서 멈춘다"고 단언하는 것은 순환이다. 그 줄 없이 다시 빌드해 모델이 이전에 멈추던 바로 그 자리에서 `<|im_end|>`를 내는 것을 보는 것은 순환이 아니고, 빌드 두 번이면 된다.
- **체크포인트에 템플릿이 없으면, 한 산출물이 수렴시키지 않는 한 두 프런트엔드는 갈린다.** 내장 템플릿과 이미 감싸인 텍스트를 알아보는 빌더가 CLI `-p`와 서버 채팅 요청을 같은 바이트 위에 두고, 둘 사이의 바이트 일치 테스트가 그 상태를 유지시킨다.
