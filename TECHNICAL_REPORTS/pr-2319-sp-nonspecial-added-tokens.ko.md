# PR #2319: SentencePiece 경로에서 non-special added token 인코딩

**날짜**: 2026-10-16
**상태**: 머지 전 CUDA(GB10)에서 실제 IQuest 체크포인트로 구현 및 검증 완료, ERNIE-4.5-VL은 토크나이저만 확인했고 Baichuan-M1 greedy 비교는 쓸 수 없음 ("미검증" 참고)
**위험도**: 중간 (non-special added token을 포함한 SentencePiece 경로 프롬프트의 토큰 id가 바뀜, 포함하지 않은 프롬프트는 그대로)

## 요약

이슈 #1758: SentencePiece 폴백(`tokenizer.model`만 있고 `tokenizer.json` 없음)은 `"special": true`인 added token만 인코딩용으로 등록했다. `"special": false` 항목은 디코드에서는 제 철자로 복원됐지만, 인코드에서는 3~5개의 일반 piece로 쪼개졌다. IQuest-Coder의 `<think>`, `<tool_call>`, `<tools>`, `<tool_response>`와 닫는 태그가 해당되어, 렌더링된 tool 프롬프트가 학습 때 쓰인 단일 예약 id 대신 일반 텍스트로 모델에 전달되었다.

1. 새 `src/tokenizer/sp_added_vocab.rs`가 `added_tokens_decoder`의 모든 항목을 보유하고, leftmost-longest 매칭으로 입력을 나눈 뒤 매칭된 토큰의 `rstrip`, `lstrip`, `single_word` 플래그를 이웃 조각에 적용한다. transformers slow tokenizer의 `tokenize` 루프를 옮긴 것이다.
2. `special`은 이제 디코드의 `skip_special_tokens`와 `parse_special: false`가 매칭을 멈추는 대상만 결정한다. `parse_special: false`에서도 non-special 토큰은 llama.cpp 및 tokenizers 백엔드처럼 계속 매칭된다.
3. decoder 표가 없으면 `added_tokens.json`이 일반 non-special 토큰을 제공한다. tokenizers 경로의 `read_added_tokens_sorted`와 같은 우선순위다.

## 문제 정의

`src/tokenizer/mod.rs`의 `parse_special_tokens`는 decoder를 `special`로 나눴다. special 항목은 인코드 표로, non-special 항목은 디코드만 참조하는 `added_token_contents`로 들어갔다. HuggingFace는 인코드에서 이런 구분을 하지 않는다. 모든 항목이 added vocabulary의 `AddedToken`이며 SentencePiece가 텍스트를 보기 전에 매칭된다.

영향은 tool 호출 프롬프트에서만 보인다. IQuest chat template은 여섯 개의 tool 태그를 문자 그대로 출력하고, 일반 채팅 경로는 special 토큰만 쓴다. 수정 후 7B tool 프롬프트는 193개에서 181개로, Loop는 205개에서 185개로 줄어든다.

### 조사

로컬 `~/models/mlx`(체크포인트 184개)에서 SentencePiece 경로는 IQuest 두 개뿐이다. `docs/model-catalog.tsv`의 153개 repo를 원격 조사한 결과 SentencePiece 경로 스냅샷은 일곱 개였다.

| 체크포인트 | non-special 항목 |
|------------|------------------|
| IQuest-Coder-V1-7B-Instruct-8bit | 8개 (`<think>`, `</think>`, tool 태그 6개) |
| IQuest-Coder-V1-40B-Loop-Instruct-4bit (카탈로그에 없음) | 8개, 같은 구성 |
| Baichuan-M1-14B-Instruct-4bit | 20개 (`<tool_call>`, FIM 및 repo 마커, `<C_A>`, `<B_FUNC>` 등) |
| ERNIE-4.5-VL-28B-A3B-Thinking-4bit | 1015개 (이미지/비디오 마커, `<|LOC_0|>`부터 `<|LOC_1000|>`) |
| ERNIE-4.5-0.3B-PT-4bit, internlm3-8b-instruct-4bit, TeleChat3-36B-Thinking-4bit | 0개 |

조사한 체크포인트 중 `added_tokens.json`에만 있고 decoder에 없는 항목은 없었다.

## 기술적 선택과 그 이유

- fast가 아니라 slow tokenizer를 기준으로 맞춘다. 이 체크포인트들은 slow 전용 tokenizer 클래스를 선언하므로 transformers slow 경로가 기준이다. IQuest가 `</s>`와 `<unk>`를 `single_word`로 선언하므로 플래그(`single_word`, `lstrip`, `rstrip`)는 무시하지 않고 그대로 옮겼다.
- `normalized`는 무시한다. slow tokenizer는 `do_lower_case`일 때만 읽는데, SentencePiece 체크포인트 중 이를 켠 것이 없다.
- `parse_special: false`에서도 non-special 토큰은 매칭 가능하게 둔다. 빼면 `/tokenize`와 logit-bias 텍스트 키가 user-defined 토큰에 대해 llama.cpp와 달라진다.
- 내용이 두 번 선언되면 더 큰 id로 해석하고, `token_to_id`도 인코드와 같은 표를 거치게 해서 둘이 어긋나지 않게 한다.
- added token은 관대하게 읽는다(`read_sp_added_tokens`는 잘못된 항목을 건너뜀). SentencePiece 로더는 잘못된 항목에도 실패한 적이 없고, 지금 로드되는 체크포인트는 계속 로드되어야 하기 때문이다. 항목 파서는 `added_tokens.rs`의 엄격한 tokenizers 경로 리더와 공유한다.
- `added_tokens.json` 항목은 항상 non-special이다. transformers의 레거시 로더는 선언된 special 토큰과 내용이 같으면 special로 표시하지만, 조사한 체크포인트 중 이에 의존하는 것은 없다.

## 변경 사항

- `src/tokenizer/sp_added_vocab.rs` (신규): `SpAddedVocab`과 `split`, `id_of`, `read_sp_added_tokens`.
- `src/tokenizer/mod.rs`: `special_token_to_id` / `special_tokens_sorted` 표와 `split_with_special_tokens`를 `added_vocab`과, 두 인코드 진입점이 함께 쓰는 `encode_split(text, add_special_tokens, parse_special)`으로 교체.
- `src/tokenizer/added_tokens.rs`: 공유를 위해 `parse_added_token_entry` 추출.
- `src/tokenizer/sp_added_vocab_tests.rs` (신규), `CHANGELOG.md`, `docs/supported-models.md` (IQuest 항목에서 기존 한계 설명 제거).

## 동작 변경

IQuest는 `</s>`를 `single_word: true`로 선언하므로 `d</s>e`는 이제 transformers처럼 텍스트로 인코딩된다. 단어에 붙은 `single_word` special 토큰은 더 이상 발동하지 않는다.

## 검증

- 일곱 토크나이저 모두에서 이전 바이너리와 새 바이너리의 토큰화 비교(기본 프롬프트, 각 non-special 토큰을 단어 중간과 공백 구분으로 삽입, 각 chat template을 거친 일반 및 tool 호출 프롬프트)를 transformers 4.57.1 기준과 대조했다. 새 바이너리는 모든 프롬프트에서 일치하고, 이전 바이너리는 non-special 토큰을 포함한 모든 프롬프트에서 불일치했다. 포함하지 않은 프롬프트는 그대로다.
- `iquest-coder-v1-7b-instruct-8bit`와 `iquest-coder-v1-40b-loop-instruct-4bit`에서 greedy id(`/completion`, 온도 0, `MLXCEL_SDPA_DETERMINISTIC=1`, 이전 바이너리 자기 일치 확인): non-special 토큰이 없는 프롬프트는 id가 동일하고, tool 호출 프롬프트는 위처럼 줄며 두 모델 모두 `get_weather` 호출을 그대로 낸다.
- 되돌림 확인: encode 표에서 non-special 항목을 빼면 새로 추가되거나 뒤집힌 테스트 5개가 실패한다.
- `tokenizer::`(137개 통과), fmt, `cargo clippy --profile test-fast --features cuda --lib --tests -D warnings`, 계약 검사가 통과한다.

## 미검증 및 후속

- ERNIE-4.5-VL은 가중치가 이 장비에 없어 토크나이저만 확인했다. 이미지 placeholder와 마커가 단일 id가 되어 이미지 확장이 template의 placeholder를 찾을 것으로 보이지만, end-to-end 실행은 없다.
- Baichuan-M1: 이전 바이너리의 자기 일치가 id가 바뀌지 않은 프롬프트에서도 CUDA에서 이미 깨진다(기존 비결정성). 따라서 greedy 비교는 쓸 수 없다. 이전 경로는 문자 그대로의 `<C_A>`를 반복했고 새 경로는 답한다.
- 쪼개진 piece에 의존하던 체크포인트가 있다면 id가 바뀐다. 조사에서는 없었지만, 흔한 부분 문자열을 added token으로 갖는 새 SentencePiece 체크포인트는 같은 방식으로 비교해야 한다.
