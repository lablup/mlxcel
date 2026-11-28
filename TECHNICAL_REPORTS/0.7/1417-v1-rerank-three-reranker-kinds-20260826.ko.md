# 기술 보고서: PR #1417 - sequence-classifier, Qwen3 생성형, Qwen3-VL 멀티모달 리랭커를 붙인 `/v1/rerank`

**작성일**: 2026-08-26
**작성자**: mlxcel maintainers
**상태**: 완료. `BAAI/bge-reranker-v2-m3`는 모델 카드와 2.6e-5까지 일치하고, 나머지 네 체크포인트는 공개된 레퍼런스가 없어 순위와 여유폭으로 게이트를 잡았다
**언어**: Rust, Markdown
**위험도**: Medium

---

## 요약

PR #1417은 이슈 #1356을 구현한다. 임베딩 에픽 #1348의 마지막 단위다. `POST /v1/rerank`(Cohere / Jina 호환), 오프라인 `mlxcel rerank` 명령, 그리고 임베딩 워커 옆에 붙는 전용 단일 스레드 리랭크 워커를 추가한다.

체크포인트 형태 세 가지를 서빙하고, 셋 다 결과는 `[0, 1]` 확률이다. BERT, XLM-RoBERTa, ModernBERT 위의 1-label 크로스 인코더는 #1321과 #1332가 이미 머지한 분류 head를 손대지 않고 그대로 쓴다. 새로 쓴 것은 쌍(pair) 토크나이즈, 절단 전략, `sigmoid(logit)`뿐이다. Qwen3 생성형 리랭커는 모델에게 yes/no를 묻고 마지막 프롬프트 위치에서 `sigmoid(logit("yes") - logit("no"))`를 읽는다. Qwen3-VL 멀티모달 리랭커는 질의와 문서가 이미지를 가질 수 있는 프롬프트 위에서 같은 읽기를 한다.

파급 범위가 가장 넓은 변경은 탐지다. 세 인코더 계열 중 하나 위의 `ForSequenceClassification` export가 이제 새 `ModelType::SequenceClassifier`(새 `ModelKind::Reranker`)로 해석된다. 이전에는 생성 디스패치까지 흘러가 에러가 났다. 이것이 `-m <크로스 인코더>`만으로 `/v1/rerank`를 서빙하게 만드는 지점이고, 옛 에러 메시지를 단언하던 머지된 테스트 세 곳을 갱신해야 했다.

공용 기반도 두 군데 움직였다. `EncodedBatch`에 `PaddingSide` 옵션이 생겼고(생성형 리랭커는 왼쪽 패딩이 필요하고, 모든 인코더 계열은 오른쪽 패딩 그대로다), `BertSequenceClassifier`는 `num_labels()`를 `config.json`이 아니라 projection 텐서에서 읽는다.

---

## 1. 문제 정의

### 1.1 배경

임베딩 검색은 독립적으로 계산한 두 벡터의 코사인으로 순위를 매긴다. 리랭커는 질의와 문서를 한 번의 forward에 함께 통과시킨다. 색인은 못 하지만 변별력은 훨씬 낫다. 보통은 `/v1/embeddings`로 후보를 뽑고 상위 후보만 `/v1/rerank`로 재정렬한다.

이슈가 지목한 다섯 체크포인트는 config 형태 두 가지를 쓴 서로 다른 세 종류다.

`BAAI/bge-reranker-v2-m3`는 길이 1의 `id2label`과 함께 `XLMRobertaForSequenceClassification`을 선언한다. `cross-encoder/ms-marco-MiniLM-L6-v2`는 `BertForSequenceClassification`에 역시 1-label, `max_position_embeddings: 512`다. `Alibaba-NLP/gte-reranker-modernbert-base`는 `classifier_pooling: "mean"`과 함께 `ModernBertForSequenceClassification`을 선언한다. 셋 다 작은 head가 붙은 인코더 몸통이고, 세 몸통 모두 임베딩 에픽에서 이미 트리에 들어와 있었다.

`mlx-community/Qwen3-Reranker-0.6B-4bit`는 `Qwen3ForCausalLM`, `model_type: qwen3`, 4-bit `quantization` 블록, tied embedding을 선언한다. config 어디에도 "리랭커"라는 말은 없다. 관련성 신호는 고정된 채팅 프롬프트 끝에서 모델 자신이 내는 `yes` / `no` 다음 토큰 분포다.

`Qwen/Qwen3-VL-Reranker-2B`는 `Qwen3VLForConditionalGeneration`, `model_type: qwen3_vl`을 선언한다. 생성 모델에는 없는 사이드 파일 두 개를 함께 배포한다. 프롬프트를 소유하며 `user`가 아니라 `role: query` / `role: document` 메시지를 읽는 `additional_chat_templates/reranker.jinja`, 그리고 답 토큰 두 개를 지정하는 `1_LogitScore/config.json`(`true_token_id: 9693`, `false_token_id: 2152`)이다. `modules.json`에는 `Pooling`이 아니라 `LogitScore` 모듈이 들어 있다.

### 1.2 기존 장애물

- **탐지는 모든 `ForSequenceClassification` 체크포인트를 그냥 거부했다.** `is_embedding_checkpoint`가 `Ok(None)`을 돌려주며 "리랭커는 임베더가 아니다"를 올바르게 지켰지만, 그 아래에서 아무도 그것을 받아가지 않았다. 그래서 `get_model_type`은 `model_type` 디스패치까지 흘러갔고 거기에는 `bert` / `xlm-roberta` / `modernbert` arm이 없어 `Unsupported model type: bert`를 냈다. 머지된 테스트 세 곳이 정확히 그 메시지를 단언하고 있었다.
- **토크나이저 기반은 오른쪽 패딩만 했다.** `EncodedBatch::from_rows`는 실제 토큰 뒤에 패딩을 쓴다. 모든 인코더 계열이 필요로 하는 형태다. 한 컬럼에서 읽는 생성형 리랭커에는 정반대가 필요하다.
- **BERT 경로의 `num_labels`는 `config.json`에서 왔다.** `ModernBertSequenceClassifier`는 이미 `classifier.weight` 행 수에서 유도하고 불일치 시 경고했지만, `BertSequenceClassifier`는 `args().num_labels`만 노출했다. 그 경로에서 "label은 정확히 하나여야 한다"를 검사하면 head의 실제 폭이 아니라 config의 주장을 검사하는 셈이 된다.
- **`Qwen3VLModel`의 `lm_head`는 비공개 필드다.** logits로 가는 유일한 공개 경로는 `[B, L, vocab]`을 만드는 `forward_for_sequence`뿐이었다. 400 토큰짜리 이미지 프롬프트를 쓰는 2B 체크포인트에서는 행당 약 240 MB의 f32이고, 한 컬럼만 빼고 전부 버려진다.
- **`mlxcel-server`는 `-m`을 요구한다.** 생성형 리랭커는 `-m`에서 도달할 수 없으므로(config가 채팅 모델의 것이다) 그것만 서빙하려면 같은 디렉터리를 `-m`과 `--reranker-model`에 모두 넘겨야 하고, 그러면 채팅 워커가 같은 가중치를 한 벌 더 올린다.
- **Qwen3-VL 이미지 prefill은 단일 행이다.** `compute_rope_index`는 `[B, L]`을 평탄화한 뒤 0번 행만 읽고, DeepStack 상태는 요청 단위다. Qwen3-VL-Embedding 계열이 이미 같은 제약을 문서화해 두었다.

### 1.3 위험 평가

| 위험 | 영향 | 가능성 |
|------|------|--------|
| 탐지 변경이 리랭커가 아닌 체크포인트를 재라우팅한다 | 높음: 무관한 계열이 로드되지 않는다 | 낮음. arm이 `model_type` 세 값으로 제한돼 있다 |
| Qwen3 프롬프트가 레퍼런스 레시피에서 1바이트 어긋난다 | 높음: 그럴듯한 점수가 나오지만 틀렸다 | 중간 |
| 오른쪽 패딩이 생성 경로에 새어 들어간다 | 높음: 가장 긴 행을 뺀 모든 행에서 pad 토큰의 점수를 읽는다 | 중간 |
| 다중 label 분류기가 리랭커로 서빙된다 | 중간: 한 클래스 logit의 `sigmoid`는 의미가 없다 | 낮지만 조용하다 |
| 절단이 assistant 헤더를 먹는다 | 높음: 점수를 읽는 위치가 답 슬롯이 아니게 된다 | 중간 |
| rerank-only 형태가 상주 메모리를 두 배로 쓴다 | 중간: 2B 리랭커가 4 GB 대신 8 GB | 가드가 없으면 높음 |
| 이미지 placeholder 개수와 vision feature 개수가 어긋난다 | 높음: merge가 다른 개수의 벡터를 뿌린다 | 중간 |

---

## 2. 기술적 선택과 그 이유

### 2.1 크로스 인코더만 탐지한다

`is_sequence_classifier_checkpoint`는 `get_model_type` 안에서 `is_embedding_checkpoint`보다 먼저 돌고, `architectures[0]`이 `ForSequenceClassification`으로 끝나면서 **동시에** `model_type`이 `bert`, `xlm-roberta`, `xlm_roberta`, `modernbert` 중 하나일 때만 `ModelType::SequenceClassifier`를 반환한다. 나머지는 전부 `None`이고 기존 라우팅을 그대로 유지한다.

제한이 핵심이다. `DebertaV2ForSequenceClassification` 체크포인트는 여기에 head 포팅이 없고, 그것을 가져가면 명확한 `Unsupported model type: deberta-v2`가 스택 더 깊은 곳의 로드 실패로 바뀐다. `Qwen3ForSequenceClassification` 체크포인트는 또 다른 물건이고 계속 `Qwen3`로 탐지된다. 이미 머지된 테스트가 그것을 단언하고 있다.

생성형 리랭커는 의도적으로 탐지 대상이 아니다. config가 채팅 export와 바이트 단위로 같기 때문에, 그것을 잡는 규칙은 채팅 모델도 잡는다. 이들은 `--reranker-model`로만 리랭크 워커에 도달하고, `detect_reranker_kind`는 운영자가 이미 "이건 리랭커다"라고 말한 뒤에 `model_type`으로 갈래를 나눈다.

### 2.2 `num_labels`를 projection 텐서에서 유도한다

`BertSequenceClassifier`는 이제 `classifier.weight`(BERT) 또는 `classifier.out_proj.weight`(XLM-RoBERTa)의 행 수를 읽고, `config.json`이 다르면 경고를 남긴다. ModernBERT의 `classifier_rows`가 이미 하던 것과 같다. `require_single_label`은 config가 아니라 그 숫자를 검사한다.

양자화는 입력 축을 묶고 출력 축은 절대 묶지 않으므로, 행 수는 dense 경로와 양자화 경로 모두에서 `num_labels`다. 그래서 안전한 진실의 출처다. config는 아니다. `id2label` 갱신을 잊은 재-export는 `num_labels()`가 `logits`의 실제 모양에 대해 거짓말하게 만들고, 리랭커는 여러 클래스 중 하나의 logit에 `sigmoid`를 씌우면서도 알아채지 못한다. 유닛 테스트는 1-label 가중치를 만들고 args의 `num_labels`를 7로 세팅한 뒤, head가 1을 보고하고 `[1, 1]`을 내는지 단언한다.

### 2.3 크로스 인코더 쌍 절단은 토크나이저에 맡긴다

`with_longest_first_truncation`은 공용 `strip_padding_and_truncation`이 `tokenizer.json`에 박힌 설정을 지운 뒤 HuggingFace 토크나이저에 `TruncationParams { max_length, strategy: LongestFirst }`를 설치한다.

대안은 인코딩 후 id를 잘라내는 것이었고, 그것이 임베딩 엔진이 단일 텍스트에 하는 일이다. 여기서는 눈에 잘 안 띄는 방식으로 틀렸을 것이다. `tokenizers::Tokenizer::post_process`는 post-processor가 특수 토큰을 붙이기 **전에** 절단하고, 예산에서 `get_n_added_tokens(is_pair)`를 먼저 뺀다. 그 순서를 손으로 재현하려면 특수 토큰 회계와 교대 삭제를 다시 구현해야 하고, 어긋나면 긴 입력에서 점수가 약간 달라질 뿐 실패로 드러나지 않는다. 위임하면 레퍼런스 `tokenizer(query, document, truncation=True, max_length=...)`의 동작을 그대로 얻는다.

### 2.4 Qwen3 프롬프트는 문자열 하나가 아니라 토큰 id로 조립한다

`PROMPT_PREFIX`, 렌더된 `CONTENT`, `PROMPT_SUFFIX`를 각각(모두 `add_special_tokens: false`로) 인코딩해 이어 붙이고, 가운데 조각만 절단한다.

문자열 하나를 만들어 결과를 자르면, 모델이 가장 필요한 입력에서 하필 assistant 헤더가 잘려 나간다. 그리고 그 헤더가 점수를 읽는 위치다. id에서 조립하면 "suffix는 항상 남는다"가 지켜야 할 한계가 아니라 구조적 성질이 된다. 유닛 테스트는 문서가 3400 단어여도 인코딩된 행이 prefix id로 시작해 suffix id로 끝나는지, 그리고 scaffold가 한계보다 크면 빈 쌍을 조용히 채점하는 대신 로드 시점에 `leaves no room`으로 거부되는지 단언한다.

세 문자열은 `prompt_bytes_match_reference_recipe`에서 모델 카드 레시피와 바이트 단위로 대조한다. `"yes"` / `"no"` 주변의 이스케이프된 따옴표와 suffix의 빈 think 블록까지 포함한다.

### 2.5 공용 배치 빌더에 왼쪽 패딩 옵션을 추가한다

`EncodedBatch::from_rows_with_padding(rows, pad_id, pad_to, side)`가 새 진입점이고, `from_rows`는 `PaddingSide::Right`로 위임하므로 임베딩 계열은 하나도 바뀌지 않았다.

왼쪽 패딩이 모든 행의 마지막 실제 토큰을 `L - 1` 컬럼에 놓는다. 오른쪽 패딩이면 가장 긴 행을 뺀 모든 행에서 pad 토큰의 점수를 읽게 되고, 유한하고 그럴듯하며 틀린 숫자가 나온다. 마스크는 `create_causal_padding_mask`이고, 그 문서 주석은 선행 패딩 행의 softmax를 유한하게 유지하는 rescue를 포함해 왼쪽 패딩 사례를 이미 다루고 있었다.

버그처럼 보이지만 아닌 결과가 하나 있어 기록해 둔다. 절대 위치는 패딩된 행 전체에 대해 여전히 `0..L`로 흐르므로, 한 문서의 점수는 어떤 배치에서 채점됐는지에 따라 미세하게 달라진다. 레퍼런스가 정확히 그렇게 한다. `Qwen3ForCausalLM.forward(input_ids, attention_mask)`는 `position_ids`가 없으면 cache position에서 유도하고, 공식 Qwen3-Reranker 예제는 `padding_side='left'`에 `position_ids` 없이 호출한다. 배치 불변성보다 레퍼런스 일치를 택했다. 손으로 점수를 다시 계산하는 유닛 테스트가 그 문서를 단독으로 채점하는 이유이고, 테스트 안에 그 이유를 적어 두었다.

### 2.6 점수는 한 컬럼에서 읽는다

두 생성 경로 모두 `forward_hidden`(Qwen3에는 이미 `pub(crate)`, Qwen3-VL에는 다섯 줄짜리 `lm_head_forward`로 새로 도달 가능)을 호출하고 `[:, L-1, :]`를 잘라 그 한 컬럼에만 head를 적용한다. 답 logit 두 개는 그 뒤 vocabulary 축에서 `take`로 모은다.

전체 시퀀스에 head를 먼저 적용하면 `[B, L, vocab]`을 할당한다. 227 토큰 이미지 행을 배치 2로 도는 VL 리랭커에서는 호출당 약 276 MB의 임시 메모리이고 즉시 버려진다. head의 두 행만 모아 `[B, 1, H] x [H, 2]`를 하는 대안은 양자화된 head에서 행이 패킹돼 있어 균일하게 동작하지 않으므로 기각했다.

`lm_head_forward`는 의도적인 최소 노출이다. 필드가 아니라 함수로 head를 노출하므로, `layers`와 `norm`을 공개했을 때처럼 두 경로가 갈라질 수 없다.

### 2.7 Qwen3-VL 프롬프트는 체크포인트 자신의 템플릿으로 렌더한다

`Qwen3VlReranker`는 `additional_chat_templates/reranker.jinja`를 `ChatTemplateProcessor`로 읽어 메시지 목록을 그대로 렌더한다. 프롬프트를 Rust로 재현하지 않는다.

그 템플릿은 특이한 일을 셋 한다. `user`가 아니라 `role: query` / `role: document`를 읽고, `system` 메시지가 없으면 자기 기본 instruction을 넣고, 이미지 content 항목마다 `<|vision_start|><|image_pad|><|vision_end|>`를 낸다. 이것을 재현하면 오늘 체크포인트의 프롬프트를 바이너리에 박게 된다. 그래서 `rerank_messages`는 요청에 `instruction`이 없으면 `system` 턴 자체를 생략해 템플릿의 기본값이 이기게 하고, 주면 그것으로 대체된다. 두 동작 모두 실제 템플릿 파일과 대조해 단언하며(체크포인트 디렉터리는 필요하지만 가중치는 필요 없다) 파일이 없으면 soft-skip한다.

답 토큰 id를 `1_LogitScore/config.json`에서 읽는 이유도 같다. 모듈을 읽을 수 없는 체크포인트는 9693 / 2152로 폴백하지 않고 로드 시점에 거부한다.

### 2.8 이미지 행은 한 번에 하나씩, 텍스트 행은 묶어서 채점한다

`Qwen3VlReranker::score`는 인코딩된 행을 나눈다. 이미지를 가진 행은 단독으로 채점하고, 텍스트 전용 행은 왼쪽 패딩된 배치 경로로 보내고, 결과를 인덱스 슬롯에 되돌려 쓴다.

제약은 요청이 아니라 포팅 쪽에 있다. `compute_rope_index`는 평탄화된 `[B, L]`의 0번 행만 읽고, `set_deepstack_state`는 요청 하나의 시각 마스크와 feature를 들고 있다. 머지된 Qwen3-VL-Embedding 계열이 같은 것을 문서화하고 이미지를 하나씩 임베딩한다. 요청 전체를 배치 1로 떨어뜨리는 대신 나누면, 혼합 요청은 이미지 문서에 대해서만 행 단위 비용을 낸다. VL 리랭커를 통과하는 텍스트 전용 요청은 배치 경로를 유지한다.

이것은 "이미지 문서 두 개의 배치가 이미지와 함께 왼쪽 패딩을 시험한다"는 이슈의 제안에서 벗어난 부분이다. M-RoPE 인덱스가 단일 행인 한 이미지가 있는 왼쪽 패딩에는 도달할 수 없다. 대응 게이트는 대신 한 요청 안의 이미지 문서 두 개가 모두 유한한 확률을 내는지 단언하고, 수동 검증이 질의를 바꾸면 순위가 뒤집힌다는 더 강한 확인을 더한다.

### 2.9 Qwen3-VL 쌍은 템플릿 렌더 전에 절단한다

`longest_first_keep(query_tokens, document_tokens, budget)`은 토크나이저의 `longest_first`가 수렴하는 것과 같은 고정점을 계산한다. 짧은 쪽은 자기 절반에 들어가면 통째로 살아남고, 아니면 둘 다 예산의 절반으로 잘린다. 예산은 `max_length`에서 scaffold(로드 시 빈 텍스트로 템플릿을 렌더해 한 번 측정한다)와 이미지가 기여할 시각 토큰을 뺀 값이다.

렌더 뒤 절단은 선택지가 아니었다. 렌더된 프롬프트를 오른쪽에서 자르면 이미지 토큰 런 안으로 들어갈 수 있고, 그러면 merge가 행의 placeholder 수와 다른 개수의 벡터를 뿌린다. 품질 저하가 아니라 명백한 실패다. 텍스트를 먼저 자르면 템플릿 출력의 구조가 온전히 남는다. 대가는 드물게 긴 쌍에서 decode 후 재인코딩 한 번이다.

이 함수는 토크나이저 경로와 같은 기대치로 유닛 테스트되어, 이 PR의 절단 전략 두 개가 서로 고정된다.

### 2.10 rerank-only 형태를 위해 모델을 올리지 않는 채팅 워커

`config.reranker_model_path`가 `startup.model_path`와 같으면 `ModelProvider::new_with_server_config_and_prompt_cache`가 `new_without_chat_model`을 반환한다. 채팅이 왜 불가능한지 로그를 남기고 종료하는 스레드로 provider를 구성하고, 그 과정에서 요청 receiver가 drop된다.

대안 둘을 기각했다. 조합 자체를 거부하면 생성형 리랭커 전용 서버가 불가능해진다. `-m`은 필수인데 그것만으로는 생성형 리랭커를 지목할 수 없기 때문이다. `new_with_full_config_and_speculative_dispatch`(이미 위치 인자 31개)와 래퍼 둘에 `skip_model_load` 플래그를 꿰는 것은 불리언 하나 때문에 서버에서 가장 뜨거운 생성자를 건드리는 일이다.

이렇게 만들어지는 상태는 채팅 로드 실패가 이미 만들던 상태와 같다(`-m <임베딩 체크포인트>`도 여기에 도달한다). `/v1/chat/completions`는 에러, `/health`는 `loading model`이다. 어떤 라우트에도 새 분기가 필요 없었고, 그것이 이 선택의 근거다. 물려받은 거친 부분은 채팅 에러 텍스트가 이유를 말하지 않고 `sending on a closed channel`이라는 점이다. 로그 줄은 이유를 말하고, HTTP 텍스트를 고치면 임베딩 사례의 동작도 바뀌므로 후속 작업으로 남긴다.

### 2.11 임베딩 워커의 큐와 타임아웃 플래그를 공유한다

새 튜닝 손잡이는 `--rerank-batch-size` 하나뿐이고, 큐 깊이와 요청별 타임아웃은 `--embedding-queue-depth` / `--embedding-request-timeout-secs`에서 온다. 두 단일 스레드 워커는 같은 방식으로 부하를 흘려보내고, 운영자가 따로 조율할 이유가 없다. 배치 크기 기본값은 `0`이고 "해당 kind의 기본값"을 뜻한다. 텍스트 리랭커는 8, 각 행이 이미지 한 장 분량의 시각 토큰을 나르는 멀티모달 리랭커는 2다.

---

## 3. 구현 상세

### 3.1 크로스 인코더 경로

```
rows  = [encode_pair_row(tokenizer, query, doc, opts) for doc in chunk]   # [CLS] q [SEP] d [SEP]
batch = EncodedBatch::from_rows(rows, pad_id, None)                       # 오른쪽 패딩
logits = head.logits(ids, mask, type_ids?)                                # [B, 1]
scores = sigmoid(astype(logits, f32))
```

`token_type_ids`는 BERT 방언에서만 요청한다(`BertSequenceClassifier::needs_token_type_ids`). XLM-RoBERTa는 segment 테이블이 한 행뿐이고 ModernBERT는 아예 없다. `max_length`는 공용 `derive_max_length`에서 오고, `is_absolute_position`은 BERT 몸통에서만 참이다. 그 뒤 `BertArgs::max_sequence_length()`가 다시 낮추므로 `bge-reranker-v2-m3`의 8194 position 행은 실제 토큰 8192에서 멈춘다. 관측값은 bge와 gte가 8192, ms-marco가 512다.

두 head 타입은 메서드 네 개(`num_labels`, `needs_token_type_ids`, `weight_max_length`, `logits`)를 가진 비공개 `ClassifierBackbone` enum 뒤에 있어, 배치 루프는 한 번만 쓰였다.

### 3.2 Qwen3 경로

```
ids   = prefix_ids ++ truncate(encode(CONTENT), max_length - |prefix| - |suffix|) ++ suffix_ids
batch = EncodedBatch::from_rows_with_padding(rows, pad_id, None, Left)
mask  = create_causal_padding_mask(batch.attention_mask, 0)               # [B, 1, L, L]
h     = model.forward_hidden(ids, None, fresh caches, Some(mask))         # [B, L, H]
last  = slice_axis(h, 1, L - 1, L)                                        # [B, 1, H]
lg    = lm_head.forward(last) 또는 embed_tokens.as_linear(last)           # [B, 1, vocab]
pick  = take(lg, [yes_id, no_id], axis 2)                                 # [B, 1, 2]
score = sigmoid(pick[..0] - pick[..1])
```

`mlx-community/Qwen3-Reranker-0.6B-4bit`에서 scaffold는 prefix 39 토큰, suffix 9 토큰으로 측정되고, 답 id는 9693과 2152로, `pad_token_id`는 공용 `resolve_pad_token_id`를 통해 `<|endoftext|>`(151643)로 해석된다. 이 체크포인트는 head가 tied이므로 `lm_head`는 `None`이고 `embed_tokens.as_linear`가 돈다.

`max_length`는 `derive_max_length(model_dir, false, override).min(8192)`다. `max_position_embeddings`는 일부러 읽지 않는다. position 테이블이 RoPE이므로 40960은 지킬 가치가 있는 한계가 아니고, 한 쌍의 prefill을 유계로 유지하는 것은 8192 천장이다.

### 3.3 Qwen3-VL 경로

```
images  = query.image? ++ document.image?                                  # 템플릿 순서
counts  = processor.compute_grid_thw(images) -> t * (h/merge) * (w/merge)
(q, d)  = truncate_texts(query, document, sum(counts))
prompt  = reranker.jinja(rerank_messages(instruction, q, has_q_img, d, has_d_img))
ids     = expand_image_placeholders(encode(prompt), image_token_id, counts)
```

이미지 행(배치 1)은 그다음:

```
pixels, grid = processor.preprocess_with_grid(images)
merged       = vlm.get_input_embeddings(ids, pixels, grid)                 # M-RoPE + DeepStack 세팅
h            = text_model.forward_hidden(ids, Some(merged), fresh caches, Some(causal))
```

텍스트 행은 Qwen3 경로와 같은 왼쪽 패딩 배치 형태다. M-RoPE와 DeepStack 슬롯은 매 호출 전후로 비워, 나중의 텍스트 행이 이미지 행의 상태를 물려받지 못하게 한다. `Qwen3VLEmbeddingModel`이 쓰는 것과 같은 규율이다.

`expand_image_placeholders`와 `apply_pixel_bounds`는 `src/models/qwen3_vl_embedding.rs`에서 재사용한다(후자는 `pub(crate)`로 승격). 덕분에 픽셀 천장이 Qwen2-VL 기본값이 아니라 체크포인트의 `preprocessor_config.json`(`max_pixels: 1310720`)에서 오고, 프롬프트를 확장한 개수와 forward가 merge하는 개수가 같은 곳에서 나온다.

### 3.4 탐지와 등록

- `src/models/mod.rs`의 `ModelType::SequenceClassifier`, `ALL_MODEL_TYPES` 등재, 메타데이터 `("Cross-encoder sequence classifier (BERT / XLM-RoBERTa / ModernBERT)", "Reranker")`.
- `src/main.rs`의 `FAMILY_ORDER`에 `"Embedding"` 바로 뒤로 `"Reranker"` 추가. `mlxcel arch`가 결정적으로 묶는다.
- `src/model_metadata.rs`의 `ModelKind::Reranker`와 `is_reranker_model_type`, 그리고 adapter를 설명과 함께 거부하는 등록 행.
- `load_model`은 리랭커 kind에서 임베딩 체크포인트나 Whisper에 쓰는 것과 같은 모양의 메시지로 bail한다. 그래서 `mlxcel generate -m <크로스 인코더>`는 없는 텐서에서 실패하는 대신 `/v1/rerank`를 지목한다.
- tensor-parallel 디스패치 테이블의 `fallback_architecture`에 `"reranker"` 자리표시자를 넣어 match를 총체적으로 유지한다.

### 3.5 HTTP 표면

`RerankInput`은 맨 문자열과 `text` / `image` / `image_url`을 가진 객체 위의 untagged enum이다. `RerankImage`는 맨 URL 문자열과 `{"url": ...}` 위의 untagged enum이라, 이미 OpenAI content part를 만드는 클라이언트가 이 엔드포인트만 특별 취급할 필요가 없다. 둘 다 있으면 `image`가 이긴다. Jina 스키마와 같다.

`create_rerank`의 검증 순서는 본문 파싱, provider 존재(`501`), model id, `top_n >= 1`, `RerankerKind::accepts_instruction` 대비 `instruction`, 그다음 항목별(빈 항목, `supports_images` 대비 이미지)이다. 이미지는 `/v1/embeddings`가 쓰는 `current_image_input_limits` / `try_read_image_url_with_limits` / `decode_request_images_with_limits` 경로를 그대로 통과하므로 페이로드, 해상도, decode 할당 한계를 공유한다.

`sort_and_truncate`는 점수 내림차순, 동점은 인덱스 오름차순으로 정렬한 뒤 `top_n`으로 자른다. 동점 규칙은 장식이 아니다. 배치가 sigmoid를 포화시키면 동점이 흔해지고, 안정 정렬만으로는 그 순서가 워커의 완료 순서에 좌우된다.

### 3.6 서버 연결

`AppState.rerank_model: Option<Arc<dyn RerankModelProvider>>`와 `with_rerank_model`은 임베딩 슬롯을 그대로 따른다. `resolve_rerank_source`는 `Explicit` / `Primary` / `None`을 반환하고, 같은 경로 사례는 `Primary`로 해석된다. 명시적 `--reranker-model`의 로드 실패는 시작 오류이고, `-m` 리랭커의 로드 실패는 로그로 남기고 라우트가 `501`을 답한다.

`/v1/models`는 id가 이미 있지 않으면 리랭커를 나열한다. `-m` 사례(id가 같다)와 rerank-only 사례(항목 하나) 모두를 덮는다.

---

## 4. 테스트 전략

### 4.1 체크포인트 없이

- `src/rerank/mod_tests.rs`(9개): `model_type` 네 표기의 탐지, 두 생성형 kind가 채팅 라우팅을 유지하는지, `deberta-v2`와 맨 `BertModel` 거부, 단일 label 가드, `num_labels` 우선순위, `1_LogitScore` `modules.json`이 Pooling 레이아웃으로 읽히지 않는지.
- `src/rerank/qwen3_generative_tests.rs`(6개): 합성 2층 Qwen3와 단어 단위 Qwen 형태 토크나이저 위에서, 바이트 단위 프롬프트 문자열, 양끝을 지키는 절단, 왼쪽 패딩이 마지막 토큰을 `L - 1`에 놓는지(그리고 오른쪽 패딩은 여전히 반대인지), yes/no 단일 토큰 가드(`yes`를 쪼개는 added token을 선언해 강제), 손으로 다시 계산한 `sigmoid(yes - no)`가 `score()`와 1e-5 이내로 일치하는지.
- `src/rerank/sequence_classifier_tests.rs`(5개): 쌍 segment id, 세 국면의 longest-first 분할, 설치된 truncation 파라미터, 그리고 VL 분할 함수를 같은 고정점에 고정.
- `src/rerank/qwen3_vl_generative_tests.rs`(6개): instruction 유무에 따른 메시지 목록, 이미지 전용 쪽, `1_LogitScore` 리더, 절단 분할.
- `src/server/routes/rerank_tests.rs`(17개): 실제 라우터를 통과하는 stub 리랭커 위에서 동점 포함 정렬, `top_n`, 두 항목 형태를 되돌려주는 `return_documents`, 모든 `400` 사례, `501`, `/rerank` alias, `/v1/models`, provider 오류의 상태 코드 매핑.
- `src/server/rerank_worker_tests.rs`(7개): info 보고, 왕복, 계열 오류 매핑, 로더 실패, 패닉 복구 후 정상 요청, 응답 타임아웃, 유계 큐 shedding.
- `src/models/bert_heads_tests.rs`: 거짓말하는 config 대비 텐서 기반 `num_labels`, 두 방언 모두.
- `src/models/detection_tests.rs`: Pooling `modules.json`이 있어도 네 크로스 인코더 표기가 `SequenceClassifier`로 가는지, `deberta-v2`가 생성 라우팅을 유지하는지.

MLX를 건드리는 테스트는 모두 공용 `mlx_test_guard`를 잡고, 게이트 수치는 `--test-threads=1`로 기록했다.

### 4.2 체크포인트가 있을 때만, 없으면 soft-skip

`src/rerank/real_checkpoint_tests.rs`(6개)는 임베딩 게이트와 같은 `local_checkpoint` 조회를 쓴다. `bge-reranker-v2-m3`는 모델 카드와 비교하고, 나머지 넷은 순위와 여유폭으로 게이트를 잡으며, VL 쪽은 픽스처를 커밋하는 대신 테스트 이미지 두 장을 코드로 그린다. 여섯 번째 테스트는 다섯 체크포인트를 모두 훑어 각각이 이슈의 표가 지정한 kind로 해석되는지 단언한다.

### 4.3 바뀌어야 했던 머지된 테스트

이미 머지된 코드의 단언 세 곳이 옛 동작을 인코딩하고 있어, 이 이슈를 명시한 주석과 함께 갱신했다. `src/embeddings/real_checkpoint_tests.rs`(`Err("Unsupported model type")` 세 행이 `Ok(ModelType::SequenceClassifier)`로), `src/models/modernbert_real_checkpoint_tests.rs`, `src/models/modernbert_tests.rs`다. 세 경우 모두 의도("리랭커는 임베더가 아니다")는 보존되고, 탐지가 그것을 표현하는 방식만 바뀌었다.

---

## 5. 실제 체크포인트 결과

아래 수치는 모두 Linux/CUDA 검증 호스트에서 `test-fast` 프로파일로 냈다. 모든 실행을 3회 반복했고, 따로 적지 않은 한 반복은 바이트 단위로 동일했다.

### `BAAI/bge-reranker-v2-m3` (XLM-RoBERTa 크로스 인코더, `max_length` 8192)

`mlxcel rerank -q "what is panda?" -d "hi" -d "<판다 문단>"`, 3회:

| 쌍 | mlxcel | 모델 카드 | 차이 |
|----|--------|-----------|------|
| 무관 | 0.00027900 | 0.00027803 | 9.7e-7 |
| 관련 | 0.99486631 | 0.99484038 | 2.6e-5 |

다섯 중 공개 레퍼런스 점수가 있는 유일한 체크포인트이고, 이슈의 2e-2 허용치보다 세 자릿수 안쪽이다.

두 번째 플래그 없이 `mlxcel-server -m <이 체크포인트>`로 서빙했을 때 Beijing 요청은 `[0.9999681, 1.5995e-5, 4.4010e-5]`, 순위 `[0, 2, 1]`로 3회 동일했다.

### `cross-encoder/ms-marco-MiniLM-L6-v2` (BERT 크로스 인코더, `max_length` 512)

`[0.9999217, 1.59e-5, 3.35e-5]`, 순위 `[0, 2, 1]`, 3회 동일. 공개 레퍼런스는 없고 게이트는 순위다. `max_length`가 512로 해석된 것은 절대 위치 상한이 읽히고 있다는 확인이다.

### `Alibaba-NLP/gte-reranker-modernbert-base` (ModernBERT 크로스 인코더, `max_length` 8192)

`[0.9800415, 0.1246974, 0.7244105]`, 순위 `[0, 2, 1]`, 3회 동일. 같은 입력에 대해 인프로세스 라이브러리 게이트는 `0.9800463`을 기록했고, 프로세스 간 편차 4.8e-6이다. 동작 차이가 아니라 CUDA 축약 순서 잡음이다. 이 체크포인트는 Berlin 방해 문서에 0.72를 준다. 포팅이 아니라 체크포인트 보정의 성질이고, 순위는 여전히 맞다.

### `mlx-community/Qwen3-Reranker-0.6B-4bit` (생성형, 4-bit)

`[0.9883127, 8.94e-6, 1.47e-5]`, 순위 `[0, 2, 1]`, CLI와 `POST /v1/rerank` 양쪽에서 3회 동일. 이슈 게이트는 `results[0] > 0.9`에 나머지 둘이 0.2 미만이고, 여유폭은 네 자릿수 넘게 남는다.

결합 프로세스: `mlxcel-server -m models/mlx/qwen3-0.6b-4bit --reranker-model <이 체크포인트>`가 한 프로세스에서 `/v1/chat/completions`는 0.6B 채팅 모델로, `/v1/rerank`는 리랭커로 답하고, `/v1/models`는 두 id를 모두 나열한다. 요청별 `instruction`도 예상대로 점수를 바꾼다(기본 instruction을 명시한 짧은 질의에서 0.2451 / 3.76e-5).

### `Qwen/Qwen3-VL-Reranker-2B` (멀티모달 생성형)

같은 프롬프트를 통과하는 텍스트 전용 쌍: `[0.8807971, 0.0953494, 0.2337064]`, 순위 `[0, 2, 1]`, 3회 동일.

이미지 문서: 그려 만든 PNG 두 장("Quarterly revenue" 제목이 붙은 막대 그래프와, 동물 실루엣이 있는 질감 있는 장면)을 data URI로 `POST /v1/rerank`에 보냈고 3회 동일:

| 질의 | chart.png | cat.png | 순위 |
|------|-----------|---------|------|
| "a chart of quarterly revenue" | 0.46879062 | 0.36296920 | 그래프 우선 |
| "a photo of an animal with two ears and eyes" | 0.1919 | 0.4378 | 동물 우선 |

질의를 바꾸면 순위가 뒤집힌다는 것이 핵심 증거다. 산술이 유한한 숫자를 낸다는 것이 아니라 이미지 내용이 모델에 도달하고 있다는 것을 보여 준다. 그래프 여유폭(0.106)은 이슈가 제안한 0.3보다 작은데, 그 제안은 실제 사진을 전제했다. 질의를 바꾼 쪽 여유폭은 0.246이다. 두 이미지 모두 합성이고 조악하므로, 절댓값은 포팅보다 그림에 대해 더 많이 말한다.

혼합 요청(텍스트 문서 둘 + 이미지 둘)은 넷을 합리적으로 정렬한다. 매출 텍스트 0.5622, 그래프 이미지 0.4688, 동물 이미지 0.3630, 고양이 텍스트 0.1192. 배치 텍스트 경로와 행 단위 이미지 경로의 분할을 한 호출에서 시험한다.

rerank-only 형태: `mlxcel-server -m <이 체크포인트> --reranker-model <같은 경로>`는 `Chat generation is disabled: ... the chat worker did not load a second copy of its weights`를 로그로 남기고, `/v1/rerank`를 정상 서빙하고, 모델 id 하나를 나열하고, `/v1/chat/completions`에는 오류를 반환한다.

---

## 6. 검증 요약

| 검사 | 명령 | 결과 |
|------|------|------|
| 리랭크 유닛 + 실제 체크포인트 게이트 | `cargo test --profile test-fast --features cuda --lib rerank:: -- --test-threads=1` | 47 통과 |
| 리랭크 워커 | `cargo test ... --lib server::rerank_worker -- --test-threads=1` | 7 통과 |
| 탐지 | `cargo test ... --lib models::detection_tests -- --test-threads=1` | 42 통과 |
| BERT head | `cargo test ... --lib models::bert -- --test-threads=1` | 28 통과 |
| ModernBERT | `cargo test ... --lib models::modernbert -- --test-threads=1` | 19 통과 |
| 임베딩 서브시스템(회귀) | `cargo test ... --lib embeddings:: -- --test-threads=1` | 69 통과 |
| CLI 레지스트리와 `mlxcel arch` | `cargo test ... --bin mlxcel -- --test-threads=1` | 197 통과 |
| 린트 | `cargo clippy --profile test-fast --features cuda --lib --bins --tests -- -D warnings` | 클린 |
| 포맷 | `cargo fmt --all -- --check` | 클린 |

이슈가 명시한 워크스페이스 전체 `--all-targets` clippy와 `metal,accelerate` 테스트 실행은 macOS 쪽 게이트라 이 호스트에서는 돌릴 수 없었다. 위의 CUDA 대응 명령이 같은 코드를 덮는다.

---

## 7. 변경 요약

파일 52개, +5693 / -40.

**새 서브시스템**(`src/rerank/`, 테스트 포함 2923줄): `mod.rs`(trait, kind, item, 탐지, 단일 label 가드, sigmoid 읽기), `loader.rs`, `sequence_classifier.rs`, `qwen3_generative.rs`, `qwen3_vl_generative.rs`, `stub.rs`, 그리고 테스트 모듈 다섯.

**새 서버 레이어**: `src/server/rerank_model.rs`(78), `src/server/rerank_worker.rs`(380)와 테스트 291줄, `src/server/routes/rerank.rs`(256)와 테스트 464줄, `src/server/types/rerank.rs`(175).

**새 CLI**: `src/commands/rerank.rs`(260)와 `Commands::Rerank` arm.

**손댄 공용 기반**: `src/embeddings/tokenize.rs`(+48, `PaddingSide`와 `from_rows_with_padding`), `src/models/bert_heads.rs`(+53, 텐서 기반 `num_labels`, `max_sequence_length`), `src/models/qwen3_vl.rs`(+10, `lm_head_forward`), `src/models/qwen3_vl_embedding.rs`(+2, `apply_pixel_bounds` 가시성), `src/models/detection.rs`(+48), `src/models/mod.rs`(+17), `src/model_metadata.rs`(+15), `src/loading/mod.rs`(+12), `src/distributed/tensor_parallel/inference.rs`(+5), `src/server/model_provider.rs`(+64).

**플래그와 설정**: `src/bin/mlx_server.rs`(+45), `src/main.rs`(+43), `src/commands/serve.rs`(+18), `src/server/cli_input.rs`(+29), `src/server/config.rs`(+19), `src/server/startup.rs`(+141), `src/server/state.rs`(+16).

**문서**: `docs/embeddings.md`(+141, Reranking 절과 source map, 탐지, 계열 주석 교정), `docs/supported-models.md`(+13, Reranker models 표).

---

## 8. 검증되지 않은 부분

- **다섯 중 넷은 공개된 레퍼런스 점수가 없다.** `ms-marco-MiniLM-L6-v2`, `gte-reranker-modernbert-base`, `Qwen3-Reranker-0.6B-4bit`, `Qwen3-VL-Reranker-2B`는 순위와 여유폭으로만 게이트를 잡았다. 검증 호스트에 PyTorch도 `transformers`도 없으므로 이들에 대한 parity 수치는 계산하지 않았고 주장하지도 않는다.
- **Qwen3-VL 이미지 게이트는 사진이 아니라 그린 이미지를 쓴다.** 절댓값(0.47 / 0.36)은 조악한 합성 PNG 두 장의 성질이다. 의미 있는 신호는 질의 교체 시의 뒤집힘이고, 이슈가 제안한 0.3에 대한 사진 기반 여유폭은 측정하지 않았다.
- **이미지가 있는 왼쪽 패딩은 시험되지 않는다.** Qwen3-VL 포팅이 이미지 행을 한 번에 하나씩 채점하기 때문이다. M-RoPE 인덱스가 배치를 인식하게 되면 이 경로에 새 게이트가 필요하다.
- **생성형 점수의 배치 위치 의존성은 문서화했지만 한계는 재지 않았다.** 한 문서의 점수는 어떤 배치에 들어갔는지에 따라 미세하게 움직이고, 이는 레퍼런스와 같다. 길고 이질적인 배치에서 그 편차가 얼마나 커지는지는 측정하지 않았다.
- **성능 수치는 없다.** 에픽이 성능 측정을 마지막에 조용한 머신에서 한 번 돌리므로 여기서는 아무것도 벤치마크하지 않았다.
- **macOS는 돌리지 않았다.** 모든 검증은 Linux/CUDA에서 했다.
- **이슈 수용 기준의 `metal,accelerate` 워크스페이스 테스트**는 이 호스트에서 돌릴 수 없었다.

---

## 9. 후속 작업

1. rerank-only 형태의 채팅 오류를 `sending on a closed channel`보다 나은 것으로 바꾼다. 고칠 자리는 공용 로드 실패 경로이므로 `-m <임베딩 체크포인트>`도 함께 나아진다.
2. `--rerank-max-length`를 서버 플래그로 둘지 검토한다. 로드 옵션은 있고 `mlxcel rerank --max-length`가 쓰지만, 이슈의 플래그 목록에 서버 쪽 플래그가 없어 조용히 추가하는 대신 뺐다.
3. Qwen3-VL의 `compute_rope_index`가 배치를 인식하게 되면 이미지 배칭을 다시 본다. `Qwen3VlReranker::score`는 이미 분할하므로 이미지 분기만 바뀌면 된다.
4. Qwen3-VL 이미지 여유폭을 실제 사진으로 측정하고, 그린 이미지 수치를 대체하는 대신 그 옆에 기록한다.

---

## 참고

- 이슈 #1356: 이 PR이 구현한 명세
- 에픽 #1348: 임베딩 및 리랭킹 에픽
- PR #1408 / 이슈 #1353: 이 작업이 재사용한 임베딩 기반(토크나이저, 한계, 워커 형태)
- PR #1411 / 이슈 #1321: `BertSequenceClassifier`
- PR #1412 / 이슈 #1332: `ModernBertSequenceClassifier`
- PR #1416 / 이슈 #1345: `forward_hidden` 분리와 이미지 placeholder 확장을 재사용한 Qwen3-VL-Embedding
- `docs/embeddings.md`, "Reranking (`/v1/rerank`) and `mlxcel rerank`" 절
- `docs/supported-models.md`, "Reranker models" 절
