# 기술 보고서: PR #1070 - feat(models): add LocateAnything (locateanything) VLM support

**날짜**: 2026-08-07
**작성**: mlxcel 메인테이너
**검토**: 구현 및 보안 리뷰 사이클
**상태**: 완료 (`pbd` 병렬 박스 디코딩 경로와 좌표 토큰-박스 후처리는 명시적으로 후속 과제로 미룬다)
**언어**: Rust, Markdown
**위험도**: 중간 (신규 모델 계열이며 로더 크래시, 프로세스 abort, 쿼드라틱 DoS를 머지 전에 발견해 수정했다)

---

## 요약

PR #1070은 이슈 #847을 닫으며 NVIDIA의 약 3B 규모 grounding VLM인 LocateAnything(`model_type: locateanything`)을 추가한다. 이 모델은 MoonViT 비전 타워와 MLP 커넥터를 mlxcel이 이미 지원하던 Qwen2 텍스트 디코더에 붙인 구조라서, 새로 짜야 할 부분은 Kimi-VL과 갈라지는 타워의 두 수치 차이, 커넥터, 이미지 토큰 스플라이스, 그리고 제어 플레인 배선뿐이었다. 이 PR은 7개 커밋과 29개 파일(머지 커밋 기준 +4007/-14)을 거쳐 머지됐는데 그중 4개는 실제 체크포인트를 돌려봐야만 드러나는 결함을 잡은 리뷰 수정 커밋이다. 4/8비트 혼합 양자화가 낸 하드 MLX shape 에러, 존재하지 않는 `tokenizer.json`, 기존 bf16-to-f16 변환을 조용히 무력화한 로더 순서 버그, 그리고 악의적인 `added_tokens_decoder`를 만나면 로더를 수십 분씩 묶어둘 수 있는 쿼드라틱 토크나이저 등록 비용이다. 넷 다 고쳤고 회귀 테스트를 붙였다. `mlx-community/LocateAnything-3B-4bit`로 진행한 실제 체크포인트 검증에서는 모델 카드가 문서화한 출력을 바이트 단위로 재현했다.

---

## 1. 문제 정의

### 1.1 배경

LocateAnything의 아키텍처는 mlxcel 입장에서 완전히 새로운 것은 아니다. `docs/adding-models.md`가 말하는 재사용 우선 원칙이 그대로 통했는데 세 구성 요소 중 둘이 이미 트리에 있었기 때문이다.

- **비전 타워**: Kimi-VL(`kimi_vl` / `kimi_k25`)이 쓰는 것과 같은 MoonShot MoonViT-SO-400M이다. LocateAnything의 upstream `vision.py`는 Kimi-VL과 딱 두 곳에서, 그것도 둘 다 수치로만 갈라진다. LayerNorm epsilon이 `1e-6` 대신 `1e-5`이고(이미 config 필드였다), 블록 MLP의 GELU가 정확한 erf 형태 대신 MLX의 tanh 근사인 `nn.GELU(approx="precise")`다.
- **텍스트 백본**: `models::Qwen2Model`을 그대로 재사용한다. Qwen2(`model_type: qwen2`)는 이미 완전히 지원되고 있었다.
- **새로 짠 부분**: 커넥터, 네이티브 해상도 이미지 프로세서, `<image-N>` 마커 스플라이스, 그리고 제어 플레인 배선(`ModelType::LocateAnythingVLM`, `"locateanything"` 탐지 분기, `LoadedModel`, `VlmRuntimeRef`, 로더 라우트, `generate_vlm` 요약 줄, TP arch-string 테이블)이다.

grounding 출력은 그냥 평범한 텍스트다. 체크포인트는 `<ref>`/`<box>` 마커와 1001개 좌표 토큰 `<0>`..`<1000>`(id 151677..152677)을 답변 안에 섞어 넣으므로 평범한 autoregressive 디코드로 충분하고 별도 디토크나이제이션이 필요 없다. 체크포인트가 갖고 있는 병렬 박스 디코딩 헤드(`pbd`, `n_future_tokens: 6`인 multi-token-prediction)와 좌표 토큰-박스 후처리는 애초에 이슈 #847 스코프 밖이었고 이번에도 밖에 남는다.

### 1.2 중복 없는 재사용: MoonViT의 델타

`src/vision/encoders/kimi_vl.rs`를 포크하는 대신 확장하는 쪽을 택했다. 블록 MLP 활성 함수는 config에 실리는 열거형이 됐다.

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MoonViTMlpActivation {
    #[default]
    Gelu,
    GeluTanh,
}
```

`mlxcel_core`는 원소별 tanh 근사 GELU를 노출하지 않는다. `gelu`와 이름과 달리 `gelu_approx`도 둘 다 erf 기반이다. 그래서 `GeluTanh`는 원시 연산으로 직접 합성했다(`0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 x^3)))`). `models::kokoro::ops::gelu_new`가 이미 쓰던 패턴 그대로다. 이 열거형의 `#[default]`는 `Gelu`이고 config 필드는 `#[serde(default)]`를 달고 있어서 그런 키가 없는 Kimi-VL의 `config.json`은 예전 동작 그대로 역직렬화된다. `tests/kimi_vl_parity.rs`도 완화되지 않았다. 거기서 바뀐 건 테스트 config 리터럴에 새로 필수가 된 필드 하나뿐이다.

### 1.3 합성 패리티로는 드러나지 않았을 두 가지 실제 체크포인트 격차

둘 다 `mlx_vlm`을 읽어서가 아니라 실제 체크포인트를 돌려보다가 나왔다.

**4/8비트 혼합 양자화.** `mlx-community/LocateAnything-3B-4bit`는 균일한 4비트가 아니다. 모델 카드가 이유를 밝히고 있는데 tied된 `embed_tokens`를 순수 4비트로 두면 좌표 토큰의 정밀도가 무너지기 때문이다. `mlx_lm --quant-predicate mixed_4_8`로 변환한 이 체크포인트는 36개 레이어 중 18개의 `v_proj`를 8비트로 저장하면서 `q_proj`/`k_proj`는 4비트로 남겨둔다. 텐서 단위 로더는 각자 자기 shape에서 폭을 알아내므로 영향이 없지만 `FusedQKVLinear::from_weights_separate`는 세 개의 패킹된 평면을 axis 0으로 concat하면서 `q_proj` 하나에서 폭을 추론한다. 혼합 레이어에서는 이게 하드 MLX shape 에러가 된다(4비트 `q`는 `[2048, 256]`, 8비트 `v`는 `[256, 512]`). 이건 LocateAnything만의 문제가 아니다. 공유 fused-QKV 로더를 쓰는 계열이라면 어떤 것이든 `mixed_4_8` 변환에서 같은 벽에 부딪힌다.

**어디에도 `tokenizer.json`이 없다.** `nvidia/LocateAnything-3B`도 그 MLX 변환본들도 fast 토크나이저를 export하지 않았고 `vocab.json` + `merges.txt` + `added_tokens.json` + `tokenizer_config.json`만 있다. `load_tokenizer`가 읽을 게 없었으니 모델 자체가 로드되지 않았다.

### 1.4 위험 평가

| 위험 | 영향 | 발생 가능성 |
|------|------|------------|
| `mixed_4_8` 체크포인트가 알아보기 힘든 MLX shape 에러로 로드 실패 | 높음 | 이 체크포인트에서는 확실. 공유 fused-QKV 로더를 쓰는 다른 계열에도 잠재 |
| `tokenizer.json`이 없는 모델 디렉터리가 아예 로드 불가 | 높음 | 이 체크포인트(NVIDIA 원본과 모든 MLX 변환본 전부)에서 확실 |
| 잘못된 `config.json`이나 `preprocessor_config.json`이 깨끗한 `Err` 대신 패닉이나 무제한 할당을 일으킴 | 높음 | 리뷰 중 4가지 별개 방식으로 실제 발생(0 나눗셈, 커넥터 폭 절단, 무제한 리사이즈, `in_token_limit` 상한 부재) |
| 큰 `added_tokens_decoder`가 에러도 타임아웃도 없이 로더를 묶어둠 | 높음 | 리뷰 중 실제 발생. 두 배씩 늘 때마다 비용이 4배씩 뛰는 깔끔한 쿼드라틱 |
| `special` 기본값이 뒤집혀 있어 added token이 `skip_special_tokens` 디코드에서 조용히 사라짐 | 중간 | `added_tokens.json`만 갖고 있는 다른 Qwen2 계열 체크포인트에 잠재. LocateAnything 자체는 영향 없음(자기 디코더 전체에 플래그를 명시하기 때문) |

---

## 2. 기술 검토

### 2.1 혼합 정밀도 로더 (`src/loading/vlm_locateanything_quant.rs`)

`densify_mixed_precision_qkv`는 모든 `self_attn` prefix를 훑으면서 `q_proj`/`k_proj`/`v_proj` 각각의 양자화 레이아웃을 `mlxcel_core::layers::reconcile_quantization_layout`으로 대조하고 비트 폭이나 group size가 서로 어긋나는 레이어를 만나면 세 평면을 전부 역양자화한다.

```rust
// SAFETY: `w` and `s` are borrowed from live map entries for the
// duration of the call, and `b_ptr` is either null or borrowed from a
// live entry in the same map.
let dense = unsafe { mlxcel_core::dequantize(w, s, b_ptr, group_size, bits, mode) };
```

역양자화는 정확하다. 저장 표현 자체의 정의를 되돌리는 연산이라 모델이 계산에 쓰는 값이 하나도 바뀌지 않는다. 합성 혼합 레이어에서 세 평면 각각에 대해 `max_abs_diff(before, after) == 0.0`을 검증하는 테스트가 붙어 있다.

대안으로 좁은(4비트) 평면을 넓은(8비트) 폭까지 재양자화하는 방법도 시도했지만 측정 결과를 보고 버렸다. MLX의 affine 양자화기는 group scale을 평범한 `(max - min) / (2^bits - 1)` 대신 큰 쪽 크기의 극단에 스냅하기 때문에 4비트 그룹이 8비트 격자에 정확히 올라앉지 못한다. 합성 평면에서 재양자화 왕복을 재봤더니 3.7e-3의 오차가 났는데 역양자화는 정확히 0이었다. 역양자화 비용은 공개된 3B 체크포인트에서 대략 190MB다(영향받는 18개 레이어의 QKV 평면이 패킹된 4/8비트에서 밀집 bf16으로 바뀌는 만큼). 그리고 실제로 혼합된 레이어에만 적용된다. 균일 4비트와 균일 8비트 체크포인트 각각에 `densify_mixed_precision_qkv`가 손대지 않는다는 것을 증명하는 포지티브 컨트롤 테스트도 있다. 근본적으로 고치려면 프로젝션별 양자화 폭을 담을 수 있는 `FusedQKVLinear`가 필요한데 이건 공유 레이어 수준의 변경이라 이번 PR 범위 밖으로 남겨뒀다.

### 2.2 토크나이저 폴백 (`src/tokenizer/mod.rs`)

`build_qwen2_bpe_tokenizer`는 `transformers`의 `Qwen2Converter`가 하는 그대로 토크나이저를 구성 요소 하나하나 재구성한다. unk 토큰도 서브워드 접두/접미사도 없는 byte-level BPE, NFC normalizer, `Sequence[Split(PRETOKENIZE_REGEX, isolated), ByteLevel(add_prefix_space, use_regex=false)]` pre-tokenizer(정규식은 `transformers/models/qwen2/tokenization_qwen2.py`에서 그대로 가져왔다), ByteLevel decoder, `trim_offsets = false`인 ByteLevel post-processor다.

이 폴백은 `is_qwen2_slow_tokenizer_dir`로 게이팅된다. `vocab.json`과 `merges.txt`가 둘 다 있어야 하고 `tokenizer_config.json`의 `tokenizer_class`가 `Qwen2Tokenizer`나 `Qwen2TokenizerFast`를 명시해야 한다.

```rust
fn is_qwen2_slow_tokenizer_dir(model_path: &Path) -> bool {
    if !model_path.join("vocab.json").exists() || !model_path.join("merges.txt").exists() {
        return false;
    }
    // ... tokenizer_class must name the Qwen2 tokenizer
}
```

`vocab.json` + `merges.txt`는 GPT-2 계열이 공통으로 쓰는 slow-tokenizer 파일 쌍이라 class 검사가 존재하는 이유는 명확하다. 다른 계열이 같은 파일 쌍을 배포한다고 해서 Qwen2 규칙으로 조용히 토크나이즈되면 안 된다. 이건 가정이 아니다. 실제 `moondream2` 체크포인트 디렉터리가 정확히 이 GPT-2 파일 쌍을 갖고 있으면서 `tokenizer_class`에는 `CodeGenTokenizer`를 선언하고 있어서 두 번 걸러진다. 한 번은 class 불일치로, 또 한 번은 `load_tokenizer`에서 이 검사보다 앞서 있는 `tokenizer.json` 분기가 무조건 먼저 반환하기 때문이다. 이건 코드만 읽고 유추한 게 아니라 실제 로컬 `moondream2` 모델 디렉터리를 놓고 확인한 사실이다. `load_tokenizer` 안에서 새 분기는 `tokenizer.json`, `tokenizer.model`, tiktoken 계열, `tokenizer.jsonl` 뒤 맨 마지막에 자리한다.

added token은 오름차순 id로 등록한 뒤 체크포인트가 선언한 id와 대조해 검증한다. `tokenizers`가 added token id를 base vocab 크기부터 순차 할당하기 때문에 어긋남이 하나라도 있으면 뒤따르는 모든 토큰이 조용히 밀린다. LocateAnything의 added token 1038개 안에는 박스 출력을 담당하는 좌표 토큰 1001개가 들어 있어서 여기서 한 자리만 밀려도 시끄럽게 실패하는 대신 모든 박스가 조용히 망가진다. 이 구성은 실제 체크포인트 디렉터리로 만든 `transformers` `Qwen2Tokenizer` 오라클을 상대로 7개 케이스에 걸쳐 검증했다. 렌더링된 ChatML 프롬프트, `<ref>`/`<box>` grounding 문자열, 반복되는 공백과 줄바꿈, 탭, CJK와 악센트 붙은 라틴 문자, 그리고 정규식이 특별 취급하는 아포스트로피 축약형까지 7개 전부 encode가 토큰 단위로 일치했고 decode 왕복도 완전히 맞았다.

### 2.3 이미지 토큰 스플라이스와 제어 플레인 배선

`<image-N>`은 채팅 템플릿이 렌더링하는 결과물이지 vocabulary 토큰이 아니라 평범한 텍스트다. 그래서 `multimodal::locateanything_prompt`는 렌더링된 프롬프트를 `<img> + <IMG_CONTEXT> * (grid_h * grid_w / merge_length) + </img>`로 다시 쓰고 재인코딩한다. upstream의 `re.sub`와 같은 동작이다. `--no-chat-template`로 도는 경우는 토큰 레벨 스플라이스가 대신 맡는다. 런타임은 그다음 인코딩된 스트림이 타워가 뽑아낼 feature row 수만큼 정확히 `<IMG_CONTEXT>` id를 담고 있는지 `vision::merge::merge_llava`로 흩뿌리기 전에 확인하므로 어긋남이 생겨도 이미지 feature가 조용히 텍스트 위치에 섞여 들어가지 못한다.

`src/models/detection.rs` 최상위 레벨에서는 `"locateanything"`이 범용 `"qwen2"` 분기보다 먼저 이겨야 한다. LocateAnything의 텍스트 서브 config도 `model_type: "qwen2"`를 선언하기 때문에 전용 분기가 없으면 grounding VLM이 텍스트 전용 Qwen2로 감지되고 만다. TP arch-string 테이블(`src/distributed/tensor_parallel/inference.rs`)에서 LocateAnything의 백본은 `llama` 계열로 잡는데 어차피 VLM류 모델은 텐서 병렬이 그보다 앞서 거부되므로 이 항목은 새 기능을 여는 게 아니라 dispatch 테이블을 빠짐없이 채우는 역할만 한다.

### 2.4 호환성

- **파괴적 변경**: 없다. 순전히 추가만 하는 모델 계열이다.
- **손댄 공유 코드**: `src/vision/encoders/kimi_vl.rs`가 config 기반 활성 함수 필드를 얻었다. Kimi-VL과 Kimi-VL 2.5의 동작은 기본값이 이전 하드코딩 값 그대로라 바뀌지 않는다.
- **바꾸지 않고 따른 관례**: `language_weights_subset`(`src/loading/vlm_locateanything.rs`)는 텍스트 스택 가중치 전체를(텐서마다 `mlxcel_core::copy(value)`, 이 체크포인트에서 대략 1.8GB) 옮기거나 참조로 공유하지 않고 진짜로 복사한다. 기존 `vlm_internvl.rs`와 `vlm_lfm2_vl.rs` 로더와 같은 방식이다. 이걸 바꾸는 건 이 PR 하나의 결함이 아니라 세 로더 전체에 걸친 관례 논의 대상이다.

---

## 3. 리뷰 중 발견되고 수정된 결함

첫 기능 커밋 뒤 4개 커밋이 실제 체크포인트나 악의적 config를 훑는 스윕이 드러낸 결함을 고쳤다. 아래는 리뷰 당시 판단한 심각도와 함께 각각을 기록한다.

### 3.1 HIGH: bf16-to-f16 변환이 혼합 정밀도 densification보다 먼저 돌았다

`load_locateanything_vlm`의 Apple Silicon bf16-to-f16 패스(`models::convert_bf16_weights_with_keep`, `.scales`/`.biases`는 bf16으로 유지)가 원래 `densify_mixed_precision_qkv`보다 먼저 실행됐다. MLX의 `dequantize`는 scales의 dtype을 그대로 물려받은 배열을 반환하는데 그 scales가 bf16이니 densification 패스가 새로 끼워 넣은 dense q/k/v 평면은 전부 변환해줄 유일한 패스가 지나간 다음에 태어난 갓 만든 bf16 텐서였다. 공개된 체크포인트 기준 54개 텐서(혼합 레이어 18개 x 프로젝션 3개)가 bf16으로 남았는데 이게 바로 변환 패스가 막으려고 존재하는 M5 JIT 크래시 그 자체다. 순서가 그걸 무력화한 셈이다.

수정은 두 패스의 순서를 바꿔 densification이 먼저 돌게 한다. 그러면 그 출력이 MoonViT 타워, 커넥터와 함께 변환 대상에 들어가고 densification이 만든 평면은 더 이상 자기 `.scales`/`.biases`를 갖고 있지 않으므로 keep-predicate가 더는 그것들을 봐주지 않는다. 이어지는 후속 커밋은 이 두 인라인 패스를 `reconcile_mixed_precision_weights(weights, group_size, bits, mode, convert_bf16: bool)`로 뽑아내면서 Apple Silicon 게이트를 `hardware::get_hardware()`를 내부에서 읽는 대신 파라미터로 받게 했다. 테스트가 어느 호스트에서 돌든 변환 분기를 강제로 탈 수 있게 하려는 목적이었다. 회귀 테스트 두 개는 공개된 체크포인트의 혼합 레이어가 하는 그대로 bf16 소스에서 양자화한 평면을 만들어(scales가 bf16으로 남는다는, 이 버그의 전제부터 확인한다) densify한 다음 `convert_bf16`가 `true`일 때만 densify된 평면이 f16으로 떨어지는지 검증한다. `false`로 둔 컨트롤 테스트는 첫 테스트의 f16 dtype이 densification 자체가 아니라 변환 패스에서 왔다는 걸 증명한다. 이게 중요한 이유는 이 PR을 작업한 CUDA 박스는 테스트 배치와 무관하게 Apple Silicon 분기를 아예 타지 않기 때문이다. 파라미터화된 추출이 없었다면 이 수정은 실제로 돌아갈 수 있는 테스트 없이 그대로 나갔을 것이다.

### 3.2 MEDIUM: added-token `special` 기본값이 HuggingFace 관례와 반대였다

`read_added_tokens_sorted`는 두 분기 모두에서 added token의 `special` 플래그 기본값을 `true`로 잡고 있었다. HuggingFace 자신의 `AddedToken`은 `special` 기본값을 `false`로 둔다. `added_tokens.json` 분기는 더 나빴다. 그 파일은 아무 플래그도 없는 평평한 `content -> id` 맵인데 여기 담긴 항목 전부가 실제로 무엇이든 상관없이 special로 강제됐다.

`AddedVocabulary`의 `special_tokens_set`는 insert-only다. `src/tokenizer/mod.rs`가 이슈 #778 맥락에서 `demote_tool_parser_markers`를 두고 이미 이 성질을 문서화해뒀다. 옛 기본값 아래서 special로 등록된 콘텐츠 토큰은 이후 다시 격을 낮출 방법이 없고 `decode(.., skip_special_tokens = true)`를 부를 때마다 조용히 사라졌다. 이제 두 기본값 모두 `false`다. LocateAnything 자체는 `added_tokens_decoder`의 1038개 항목 전부에 `special`을 명시하고 있어 어느 쪽이든 영향이 없다. 이 버그는 `added_tokens.json`만 배포하는 다른 Qwen2 계열 체크포인트에 잠복해 있었을 뿐이다. 단위 테스트 두 개가 두 분기를 다 덮는다. `added_tokens_decoder_defaults_special_to_false`(명시적 `special: true`는 살아남고 키가 없으면 non-special로 기본 설정됨을 확인)와 `added_tokens_json_fallback_tokens_are_not_special`(이 파일 안에는 어떤 항목을 special로 표시할 근거가 애초에 없음을 확인)이다.

### 3.3 HIGH: `merge_kernel_size` 산술의 0 나눗셈과 절단

`merged_token_count`는 `(merge_kernel_size[0] * merge_kernel_size[1]) as i32`를 계산해 그걸로 나눴다. `merge_kernel_size: [65536, 65536]`는 정확히 2^32인 `usize` 곱인데 좁혀 캐스팅하는 `as i32`가 이걸 0으로 잘라버려서 다음 줄이 "attempt to divide by zero"로 패닉했다. 이 값 쌍은 정사각형이면서 0도 아니라 기존 가드 두 개(`.max(1)`와 정사각형 커널 검사)를 둘 다 통과한다. 수정은 merge 곱을 `saturating_mul`로 만들고 캐스팅 전에 `1..=i32::MAX` 범위로 클램프하며 패치 곱은 `i64`로 계산해서 두 `i32` grid 변이 곱해져도 오버플로가 안 나게 한다.

한 계층 위 커넥터에서도 같은 모양의 버그가 하나 더 나왔다. `(hidden_size * merge_h * merge_w) as i32`도 똑같이 잘렸다. `hidden_size: 1152`에 같은 `[65536, 65536]` merge를 넣으면 `input_dim`이 0이 되는데 이건 `LocateAnythingConnector::forward`의 `reshape(image_features, &[-1, 0])`까지 흘러가 MLX 내부에서 throw를 던진다. cxx 경계를 넘는 C++ throw는 `Result`로 풀리는 대신 프로세스를 abort시킨다. 이 계산은 `checked_mul`과 `i32::try_from`을 쓰는 `connector_input_dim`이 됐고 잘린 0 대신 문제가 된 config 값을 명시한 설명형 `Err`를 반환한다.

`patch_size`와 `merge_kernel_size`는 클램프가 아니라 로더(`to_moonvit_config`)에서 아예 거부한다. 두 값 모두 서로 합의해야 하는 두 소비자가 각자 참조하기 때문이다. 프로세서는 여기서 패치 grid를 뽑아내고 MoonViT conv patch-embed와 patch merger는 같은 자리에서 만들어지는 `KimiVLVisionConfig`를 기준으로 크기가 정해진다. 한쪽만 클램프하면 시끄러운 실패가 조용한 어긋남으로 바뀔 뿐이다. 상한값(patch_size 128, merge 축당 16)은 공개된 지오메트리(`patch_size: 14`, `merge_kernel_size: [2, 2]`)를 한참 벗어나 있고 전용 테스트가 이 값이 상한 안쪽에서 그대로 유지되는지 확인한다.

### 3.4 HIGH: 쿼드라틱 토크나이저 등록(서비스 거부)

`build_qwen2_bpe_tokenizer`는 원래 added token을 하나씩 한 번에 하나씩 등록했다. `Tokenizer::add_tokens`/`add_special_tokens` 호출은 매번 `AddedVocabulary::refresh_added_tokens`로 끝나는데 이 함수가 지금까지 쌓인 토큰 전체를 놓고 Aho-Corasick 오토마톤 두 개를 다시 만들고 `add_tokens`는 거기다 기존 토큰 집합 전체를 먼저 복제까지 한다. N번 따로 호출하면 N번 완전 재구축이 드는 셈이라 깔끔하게 쿼드라틱이다. 고정된 `tokenizers` 0.22.2에서 release 빌드로 재보니 두 배씩 늘 때마다 비용이 4배씩 뛰었다(500개 46.9ms, 1000개 162ms, 2000개 609ms, 4000개 2.42s, 8000개 10.1s, 16000개 46.1s). 이 곡선을 10만 개짜리 `added_tokens_decoder`까지 그대로 늘려보면(악의적인 `tokenizer_config.json`이 그런 값을 선언하는 걸 막을 방법이 딱히 없었다) 에러도 타임아웃도 없이 대략 30분 규모로 걸린다. LocateAnything 자기 것인 1038개 토큰조차 원래 0.5ms면 될 걸 175ms를 냈다. main 대비 리그레션이기도 했는데 이전에는 added token이 `tokenizer.json`을 통해서만 들어왔고 그 역직렬화기는 한 번에 배치로 처리했기 때문이다.

수정은 special 플래그가 같은 연속 구간(run) 단위로 등록을 묶는다. special 그룹 하나 더하기 normal 그룹 하나로 나누는 게 아니다. `tokenizers`가 토큰을 넘겨받은 순서대로 id를 배분하기 때문에 재그룹화는 id를 보존하지 못한다(7, 8, 9로 선언된 `[A(special), B(normal), C(special)]`을 두 그룹으로 나누면 A=7, C=8, B=9로 나와버린다). run 단위로 묶으면 정렬 순서가 그대로 유지되어서 대부분이 균일하게 special인 체크포인트(LocateAnything이 그렇고 일반적인 경우도 대체로 그렇다)는 호출 한 번으로 끝나고 완벽하게 번갈아 나오는 극단적인 시퀀스만 이전 토큰당 비용으로 되돌아가되 id는 하나도 밀리지 않는다. 로더를 통째로 놓고 합성 체크포인트로 재봤을 때 1038개 토큰이 197.6ms에서 1.5ms로, 4000개가 2.54s에서 4.8ms로, 16000개가 44.0s에서 23.3ms로 줄었고 결과로 나온 added-token 테이블과 인코딩은 매번 동일했다. run 배칭은 기존 토큰당 루프를 상대로 33가지 플래그 패턴에 걸쳐 id 단위로 동일한지 검증했다. 전부 special, 전부 plain, 두 가지 교대 패턴, `[special, normal, special]` 함정, 섞인 run, 실제 LocateAnything의 1038개 토큰 모양, 시드를 고정한 무작위 시퀀스 24개까지다.

같은 커밋이 `validate_dense_vocab_ids`도 추가했다. id가 정확히 `0..len`이 아닌 `vocab.json`을 거부한다. `BPE::get_vocab_size()`는 실제로 어떤 id가 쓰이고 있는지와 무관하게 `vocab.len()`을 그대로 보고하고 `AddedVocabulary`는 그 개수부터 added token id를 배분한다. `len` 아래 빈틈이 하나라도 있으면 added token이 base vocab이 이미 가진 id 위로 겹쳐 앉는다. `{"Ġ":0,"h":1,"Ġh":2,"i":3,"COLLIDE":5}`에 id 5를 선언한 added token을 더해 직접 재봤더니 등록 후 `token_to_id("COLLIDE")`와 그 added token이 둘 다 5로 풀렸고 `decode([5])`는 더 이상 `COLLIDE`를 돌려주지 않았다. 라이브러리는 시작 id를 조정할 방법을 전혀 노출하지 않으므로 성긴 파일을 거부하는 게 유일하게 안전한 대응이다. 로컬 모델 세트에 있는 `vocab.json` 65개는 전부 dense라 실제 체크포인트가 이 검사로 걸러지는 일은 없다.

### 3.5 MEDIUM: 그 밖의 방어 강화

- **무제한 리사이즈.** `LocateAnythingProcessor`는 이미지 각 변을 `merge * patch`의 배수로 올림하는데 유일한 상한은 511 패치짜리 grid envelope뿐이었다. `patch_size: 100000`을 선언하면 `resize_exact`가 대략 200000x200000짜리 RGB 버퍼(100GB 넘게)를 요구하며 프로세스를 OOM abort시켰다. `in_token_limit` 다운스케일도 구제해주지 못했는데 `p`가 이미지보다 크면 `(w / p) * (h / p)`가 0이 되어버리기 때문이다. `patch_size`와 `merge_kernel_size`를 거부하는 3.3의 수정이 로더 단에서 이걸 막고 `LocateAnythingProcessor::new`의 public 생성자는 로더 상수와 동일하게 맞춘 백스톱 클램프를 별도 테스트로 고정해뒀다.
- **무제한 `in_token_limit`.** `preprocessor_config.json`에서 읽어오는 이 값만이 다운스케일을 켜는 유일한 조건이다. 무제한으로 두면 이미지 한 장이 grid envelope로만 상한이 걸려서 `patch_size: 14`에서 f32 패치 데이터만 대략 614MB에 리사이즈된 RGB 이미지와 MLX 배열 복사본까지 더해진다. 이건 거부 대신 클램프했다. 타워가 이 값을 직접 보지 않아서 뒤에서 어긋날 여지가 없기 때문이다.
- **조용한 vocabulary 앨리어싱.** 위 3.4의 `validate_dense_vocab_ids`가 이미 다룬다. DoS와 함께 고쳤지만 실패 양상 자체가 다르므로(조용히 틀린 출력) 따로 적어둔다.

---

## 4. `unsafe` 블록 감사

이 PR이 도입하는 유일한 `unsafe` 블록은 `dequantize_plane_in_place`(`src/loading/vlm_locateanything_quant.rs`)에 있다.

```rust
let b_ptr = weights
    .get(&format!("{prefix}.biases"))
    .and_then(|b| b.as_ref())
    .map(|r| r as *const mlxcel_core::MlxArray)
    .unwrap_or(std::ptr::null());
let dense = unsafe { mlxcel_core::dequantize(w, s, b_ptr, group_size, bits, mode) };
```

트리 다른 곳에 17곳 넘게 있는 확립된 cxx nullable-`biases` 브리지 패턴이다. `b.as_ref()`는 null `UniquePtr`를 만나면 `None`을 내놓으니 이게 매달린 포인터가 아니라 진짜 null 포인터로 이어진다. `w`와 `s`는 호출이 지속되는 동안 살아 있는 `WeightMap` 엔트리에서 빌려온다. `dequantize_plane_in_place`에 도달하기 전에 `reconcile_quantization_layout`이 `group_size`, `bits`, `mode` 문자열을 검증하기 때문에 혼합 레이어가 잘못된 체크포인트라면 검증되지 않은 입력이 이 호출까지 오는 대신 그 전에 깨끗한 `Err`가 난다. 새로 생긴 안전성 취약 지점은 없다.

---

## 5. 검증 근거

### 5.1 실제 체크포인트 생성

GB10, CUDA(sm_121), release 빌드, `mlx-community/LocateAnything-3B-4bit`, 모델 카드 자체 사용 예시가 쓰는 COCO 이미지를 대상으로 했다.

```
$ ./target/release/mlxcel generate -m models/mlx/locateanything-3b-4bit \
    --image 000000039769.jpg -p "Detect all objects in the image." -n 128 --temp 0.0
LocateAnything: inserted 1 image block(s) (414 total image tokens)
<ref>object</ref><box><64><152><273><244></box><box><520><160><580><392></box>
```

첫 박스 `<64><152><273><244>`는 이 이미지에 대해 모델 카드가 문서화한 박스와 바이트 단위로 일치한다(카드는 같은 영역을 `remote`로 라벨링하는데 4비트 변환본이 대신 내놓은 `object` 라벨은 카드가 스스로 문서화한 의미상 일반화 그대로다). "the cat on the right"라는 referring query는 `<541><49><1000><778>`을 돌려주는데 픽셀로 환산하면 대략 (346, 24, 640, 372)이고 COCO ground truth 박스 (345, 23, 640, 368)에 가깝다.

### 5.2 토크나이저 오라클 패리티

실제 체크포인트 디렉터리로 만든 `transformers` `Qwen2Tokenizer` 오라클을 상대로 별도 검증했다. ChatML 프롬프트, grounding 문자열, 공백/줄바꿈 반복, 탭, CJK, 축약형까지 7개 케이스 전부 encode가 토큰 단위로 맞았고 decode 왕복도 완전히 일치했다.

### 5.3 전체 테스트 스위트와 머지 게이트

머지 시점 영역별 스위트 카운트는 다음과 같다. `locateanything` 53 passed, `tokenizer` 75 passed(5 ignored), `kimi_vl` 25 passed, `detection` 80 passed, `model_metadata` 8 passed, `loading::` 255 passed, `vision::` 314 passed(17 ignored), `multimodal::` 187 passed(17 ignored), `kimi_vl_parity` 3 passed. `cargo clippy --features cuda --lib --tests -- -D warnings`는 깨끗했고 `cargo fmt --all`도 깨끗했다.

머지 후보를 놓고 돌린 전체 CUDA 머지 게이트는 7365 passed, 5 failed로 나왔는데 같은 5개가 main에서도 똑같이 실패해서 main에서 컨트롤로 다시 돌려 확인했다(4개는 `MLX_ENABLE_TF32=0`이면 통과하는 TF32 아티팩트이고 1개는 `resolve_paged_block_budget`에 있던 기존 정수 언더플로다). 이 PR이 순증한 통과 테스트는 20개다.

---

## 6. 보류한 것

- **`pbd` 병렬 박스 디코딩 경로.** 더 빠른 박스 출력을 위한 multi-token-prediction 헤드(`n_future_tokens: 6`)다. 처음부터 이슈 #847 스코프 밖이었다.
- **좌표 토큰-박스 후처리.** 지금은 박스가 생성된 텍스트 안에 원시 `<N>` 좌표 토큰으로만 나온다. 픽셀이나 정규화 좌표로 옮기는 건 호출자 몫이거나 후속 과제로 남는다.
- **Apple Silicon bf16-to-f16 실전 검증.** 파라미터화된 `reconcile_mixed_precision_weights` 덕에 변환 분기는 이제 단위 테스트로 덮이지만 이 체크포인트 자체는 Linux CUDA 박스에서 개발하고 검증했기 때문에 end-to-end로는 돌려보지 못했다.
- **`FusedQKVLinear`의 프로젝션별 폭.** 2.1의 dequantize-on-mixed-precision 워크어라운드는 당장의 걸림돌은 치웠지만 공유 로더가 프로젝션별 양자화 폭을 네이티브로 표현하지 못하는 근본 문제는 그대로 남겨뒀다. `mixed_4_8`로 변환된 다른 어떤 계열이든 이게 해결되기 전까지는 같은 190MB급 비용을 치른다.

---

## 7. 교훈

- **합성 패리티 하네스를 통과한 체크포인트도 로드 자체가 안 될 수 있다.** 이 PR을 규정한 두 가지 격차(4/8비트 혼합 양자화, 사라진 `tokenizer.json`)는 잘 만들어진 합성 config로 짠 테스트로는 애초에 보이지 않았다. 실제로 배포된 체크포인트를 로드했을 때만 표면으로 드러났다.
- **두 패스가 각자 올바르다고 순서까지 저절로 맞는 건 아니다.** bf16 순서 버그는 `densify_mixed_precision_qkv`나 bf16-to-f16 변환 어느 쪽의 로직 오류도 아니었다. 둘 다 따로 놓고 보면 옳았다. 문제는 순전히 어느 게 먼저 도느냐였고 그게 로더 자신의 문서가 명시적으로 M5 JIT 크래시를 피하려고 존재한다고 밝힌 안전장치를 조용히 무력화했다.
- **upstream 라이브러리 관례를 뒤집는 기본값은 중립적이지 않다.** `AddedToken::special`이 HuggingFace의 `false` 대신 `true`를 기본값으로 삼은 건 이 PR 자체의 체크포인트(디코더 전체에 플래그가 명시돼 있다)에는 아무 영향이 없었지만 같은 코드 경로를 타는 다른 모든 체크포인트에는 살아 있는 버그였다. 게다가 insert-only인 `special_tokens_set` 때문에 한번 걸리면 되돌릴 수도 없었다. 단순히 틀린 정도가 아니었다.
- **정사각형이고 0도 아닌 값도 오버플로할 수 있다.** `merge_kernel_size: [65536, 65536]`는 기형적인 merge 커널을 잡으려고 써둔 가드(비정사각형, 0)를 전부 통과했다. 그 가드들이 좁혀 캐스팅의 오버플로까지 염두에 두고 쓰인 게 아니었기 때문이다.
