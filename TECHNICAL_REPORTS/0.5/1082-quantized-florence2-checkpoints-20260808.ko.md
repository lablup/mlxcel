# 기술 보고서: PR #1082 - feat(models): Florence-2 양자화 체크포인트 지원 (4-bit / 8-bit / 6-bit / 3-bit)

**날짜**: 2026-08-08
**작성**: mlxcel 메인테이너
**검토**: 구현 및 보안 리뷰 사이클
**상태**: 완료 (범위 밖 결함 하나를 발견했고, 고치는 대신 기록했다)
**언어**: Rust, Markdown
**위험도**: 중간 (그동안 아예 거부하던 체크포인트에 새 로딩 경로가 생겼다. dense 경로는 그대로이고 기존 패리티 테스트로 고정되어 있다)

---

## 요약

Florence-2는 양자화 메타데이터가 붙은 체크포인트를 전부 거부했고, 그래서 공개된 `mlx-community` 변환본 여덟 개를 하나도 열 수 없었다. 이번 변경으로 양자화 경로가 들어간다. 프로젝션과 임베딩 테이블, LM 헤드가 트리의 `UnifiedLinear` / `UnifiedEmbedding`을 타고, 이 레이어들이 가중치 prefix마다 dense인지 packed인지 스스로 고른다. #854가 넣은 거부는 없애지 않고, 이 계열이 여전히 dense로만 소비하는 몇 개 텐서로 좁혔다.

검증 기준은 bf16 체크포인트가 아니라 **같은 4-bit 체크포인트를 돌린 upstream mlx-vlm**이다. 이 선택이 보고서의 핵심이다. 4-bit 실행을 bf16 실행과 비교하면 양자화가 얼마나 손실적인지만 알 수 있고 그래프가 맞는지는 끝까지 알 수 없는데, 애초에 두 실행이 서로 다른 가중치를 계산하기 때문이다. 동일한 packed 바이트 위에서 두 번째 구현과 맞춰 보면 포트 자체만 분리된다. 그 위에서 두 가지를 함께 주장한다. mlxcel은 dense 패리티 테스트가 쓰던 허용 오차 안에서 upstream의 4-bit 활성값을 재현하고, bf16에서 벌어진 거리 또한 upstream이 같은 두 체크포인트 사이에서 벌린 거리와 상대 RMS 5.8e-4 이내로 일치한다.

이슈가 요구한 `large-ft` 스모크 테스트에서 두 번째 발견이 나왔는데, 여기서는 일부러 고치지 않았다. `large-ft` 릴리스는 이 트리에서도 upstream mlx-vlm에서도 망가진 출력을 내고, 4-bit뿐 아니라 bf16에서도 그렇고, 이 브랜치 이전부터 그랬다.

---

## 1. 문제 정의

### 1.1 배경

에픽 #850이 #852(BART seq2seq 스택), #853(DaViT 타워), #854(퓨전), #855, #856을 거치며 Florence-2를 올렸지만 대상은 bf16과 f16 export뿐이었다. `Florence2Model::load`는 나머지를 이렇게 거부했다.

```
Florence-2 quantized checkpoints are not supported yet: the BART text stack and
the DaViT tower are built from dense layers. Use a bf16 or f16 export, ...
```

당시엔 맞는 판단이었다. 두 축 모두 dense `Linear`와 `Embedding`으로 짜여 있었고, packed `uint32` 텐서가 MLX까지 닿으면 잡을 수 있는 오류가 나지 않는다. MLX는 C++에서 throw하고 그 예외가 cxx 브릿지를 건너면서 프로세스가 죽는다. 절반만 로드했다가 커널 안에서 죽느니 이름 붙은 오류로 일찍 거부하는 편이 나았다. 문제는 그 거부가 영구화됐다는 점, 그리고 판단 근거가 로더가 실제로 처리 못 하는 무언가가 아니라 config 메타데이터였다는 점이다.

### 1.2 막혀 있던 것

`mlx-community/Florence-2-{base,large}-ft-{3,4,6,8}bit` 여덟 개 전부다. `base-ft` 4-bit 변환본은 163 MB로 bf16의 542 MB와 대비되니, 막혀 있던 경로가 메모리가 빠듯한 맥에서 가장 쓸모 있는 경로이기도 했다.

### 1.3 이슈 본문이 틀렸던 지점

이슈는 스케치였고 체크포인트와 다섯 군데에서 어긋났다. 다음 사람을 오도할 가능성이 가장 큰 부분이라 5절에 전부 적어 두었다. 그중 파급이 큰 것 하나만 미리 적으면, 이슈의 제안 해법에는 선형 프로젝션만 나열되어 있는데 실제로는 임베딩 테이블 쪽이 이번 작업의 더 큰 절반이다.

### 1.4 위험 평가

| 위험 | 영향 | 가능성 |
|------|------|--------|
| packed 텐서가 dense 연산에 닿아 프로세스가 죽는다 | 높음 (잡히지 않고 진단도 남지 않는다) | prefix 하나만 놓쳐도 확실 |
| 잘못된 group size로 역양자화해 그럴듯한 쓰레기가 나온다 | 높음 (조용하다) | 낮지만 교차 구현 기준 없이는 탐지 불가 |
| bf16 scales를 f16으로 올려 복원되는 모든 가중치가 흔들린다 | 중간 (조용하고 작다) | 기존 무조건 변환을 그대로 두면 확실 |
| 배선을 바꾸다 dense 경로를 깨뜨린다 | 중간 | 기존 패리티 테스트 다섯 개가 막아 준다 |

---

## 2. 기술 검토

### 2.1 변환본이 실제로 packed하는 것

`Florence-2-base-ft-4bit` 기준으로 텐서 1062개, 양자화 stem 198개다.

| packed | dense |
|---|---|
| BART q/k/v/out_proj, fc1, fc2 (인코더·디코더) | 그 프로젝션들의 모든 `.bias` |
| `language_model.lm_head` | 모든 LayerNorm weight·bias |
| `language_model.model.shared` | `image_projection` |
| 인코더·디코더 `embed_positions` | `visual_temporal_embed.pos_idx_to_embed` |
| `image_pos_embed.{row,column}_embeddings` | `vision_tower.convs.*.proj` |
| DaViT window/channel attention `qkv`, `proj` | depthwise `*.dw` 합성곱 |
| DaViT `ffn.fn.net.fc1`, `fc2` | |

이 구분은 임의로 정해진 게 아니다. upstream은 `nn.Module`을 훑으며 양자화하니 `nn.Linear`와 `nn.Embedding`은 전부 packed되고, 날 `nn.Parameter`나 `nn.Conv2d`로 등록된 것은 손대지 않는다. 3.3절의 좁힌 거부가 추측이 아니라 근거를 갖게 된 이유가 이 규칙이다.

### 2.2 이슈가 짚지 않은 결함 세 가지

**활성 dtype을 packed 평면에서 읽고 있었다.** `Florence2TextModel::from_weights`는 dtype을 `shared.weight`에서 가져왔는데, 양자화 체크포인트에서 그 텐서는 `uint32`다. 디코더는 그 dtype으로 additive causal mask를 만들고 퓨전 경로는 이미지 특징을 concat 전에 그 dtype으로 캐스팅하니, 정수 dtype이 들어가면 미관 문제로 끝나지 않는다. 지금은 양자화 분기에서 `scales`를 읽는다.

```rust
let dtype = match shared.quantized() {
    Some(quantized) => mlxcel_core::array_dtype(&quantized.scales),
    None => mlxcel_core::array_dtype(shared.weight()),
};
```

**`convert_bf16_weights`가 한 곳이 아니라 세 곳에서 무조건 돌고 있었다.** 이슈는 `model.rs`를 가리키지만 `Florence2TextModel::load`와 `Florence2DaViT::load`도 호출한다. scales와 biases는 활성값이 아니라 역양자화 피연산자여서, 반올림하면 모델이 복원하는 가중치 전부가 흔들린다. 세 곳 모두 체크포인트가 양자화를 선언하지 않은 경우에만 돌도록 막았고, 이는 공용 VLM 로더의 `finish_vlm_weights_common`이 이미 하던 것과 같다.

**`sanitize`가 `model.shared`를 한 평면만 채웠다.** BART는 인코더·디코더 토큰 테이블을 `model.shared`에 묶고 export마다 셋 중 무엇을 실체화하는지가 달라서, `shared`가 없으면 sanitizer가 `embed_tokens`에서 복사한다. `.weight`만 복사하면 packed 테이블이 dense 테이블처럼 보이는데 그게 정확히 abort 케이스다. 이제 `.scales`와 `.biases`도 함께 옮긴다.

### 2.3 위치 테이블은 접근 방식 자체가 바뀌었다

`Florence2Encoder`와 `Florence2Decoder`는 `embed_positions`를 날 텐서로 들고 `slice(table, [offset + 2, 0], [offset + 2 + seq, d_model])`을 했다. packed 테이블은 그렇게 자를 수 없다. 저장된 폭이 모델 폭이 아니라 비트 깊이의 함수이기 때문이다. 둘 다 `UnifiedEmbedding`을 들고 gather하도록 바꿨다.

```rust
let positions = mlxcel_core::arange_i32(POSITION_OFFSET, POSITION_OFFSET + seq, 1);
let pos = self.embed_positions.forward(&positions);
```

dense 분기에서는 gather가 예전 slice와 정확히 같은 행을 돌려주고, 손대지 않은 dense 패리티 테스트가 그것을 확인해 준다. 로드 시점 경계 검사도 손으로 쓴 shape 비교에서 공용 `validate_embedding_table` 가드로 옮겼다. 이 가드의 양자화 분기가 `scales`와 조정된 group size로 논리적 폭을 복원한다. 그 복원이 없으면 예전 `cols == d_model` 비교는 양자화 체크포인트를 전부 거부하는데, 거기서 `cols`는 768이 아니라 96이기 때문이다.

### 2.4 배관

`Florence2Quantization { group_size, bits }`를 최상위 `quantization` 객체에서 파싱해 `Florence2TextConfig`와 `Florence2VisionConfig` 양쪽에 싣는다. 이 블록은 `text_config`·`vision_config` 안이 아니라 **옆에** 있어서 전체 문서를 보는 파서만 채울 수 있다. `from_text_config`와 `from_vision_config`는 dense 기본값을 두고 `from_model_config`가 덮어쓴다. DaViT 쪽은 `BlockParams`로 흘려보내는데, 이 구조체는 `Copy`이고 이미 스테이지별 기하 정보를 나르고 있어서 구조체 세 겹에 새 파라미터를 뚫지 않아도 된다.

---

## 3. 기술적 선택과 그 이유

### 3.1 dense 가중치가 아니라 같은 가중치를 기준으로 삼았다

이슈는 출력이 "bf16 경로와 명시된 허용 오차 안에서 일치"할 것을 요구한다. 글자 그대로 받으면 잘못된 실험이다. 4-bit affine 복원 오차는 가중치마다 그룹 동적 범위의 약 3.3%이고, DaViT 타워는 pre-norm이라 그 오차가 정규화되지 않은 residual 흐름을 따라 열두 블록 동안 쌓인다. 타워 출력에서 측정된 이탈은 상대 RMS 25%다. 이걸 통과시킬 만큼 느슨한 허용 오차는 위치 테이블이 엉뚱한 행을 집어 오는 것을 절대 못 잡고, 그걸 잡을 만큼 빡빡한 허용 오차는 올바른 구현을 떨어뜨린다.

같은 4-bit 체크포인트를 돌린 upstream mlx-vlm에는 이 문제가 없다. 두 런타임이 동일한 바이트를 동일한 파라미터로 역양자화하니, f16 연산 순서 잡음을 넘는 차이는 전부 이 포트의 결함이다. 그래서 이쪽을 1차 기준으로 삼았고 허용 오차는 dense 패리티 테스트가 쓰던 값 그대로다(이미지 특징 1e-2, 인코더 은닉 상태 8e-2, step-0 로짓 5e-2).

### 3.2 dense 비교는 교차 구현 검사로 바꿨다

이탈을 측정해 놓고 버리면 아깝다. `florence2_quantization_cost_matches_mlx_vlm`은 mlxcel의 4-bit 대 bf16 거리를 upstream이 같은 두 체크포인트에서 낸 거리와 맞춘다.

| 단계 | mlxcel 상대 RMS | mlx-vlm 상대 RMS | mlxcel 코사인 | mlx-vlm 코사인 |
|---|---|---|---|---|
| 이미지 특징 | 0.25486 | 0.25540 | 0.967595 | 0.967458 |
| 인코더 은닉 | 0.41641 | 0.41583 | 0.913923 | 0.914181 |
| step-0 로짓 | 0.11858 | 0.11861 | 0.992970 | 0.992966 |

상대 RMS는 5.8e-4, 코사인은 2.6e-4 안에서 맞는다. 테스트 기준은 5e-3과 2e-3으로 대략 한 자릿수 여유를 뒀지만 배선 결함이 빠져나가기엔 여전히 너무 빡빡하다. 이 테스트가 막는 실패 모드는 전부 코사인을 1e-1 이상 움직인다. 손으로 고른 임계값이었다면 "말이 안 되는 값은 아니다" 정도만 말했을 텐데, 이 방식은 포트가 참조 구현이 치르는 비용만큼만 치른다고 말한다.

LayerNorm 뒤에서는 두 지표가 독립이 아니라는 점은 짚어 둔다. `sqrt(2 * (1 - 0.967458))`은 0.2546이고 이는 그 단계의 측정 상대 RMS와 같다. 그럼에도 둘 다 보고하는 이유는, 방향 변화가 아니라 이득(gain) 이동이 지배하는 단계에서는 둘이 더 이상 중복이 아니기 때문이다.

### 3.3 거부는 없앤 게 아니라 좁혔다

`reject_unsupported_quantized_tensors`는 sanitize를 거친 가중치 맵에서, 이 구현이 dense로만 소비하는 stem에 `.scales`가 붙어 있는지 훑는다. 대상은 `image_projection`, `visual_temporal_embed.*`, `convs.*.proj`, 그리고 모든 `*.dw`다. 셋 다 upstream의 모듈 순회가 닿을 수 없는 텐서이므로, 이것들을 packed로 담은 체크포인트는 단지 더 최신인 게 아니라 비표준이고, `conv2d` 안에서 죽는 것보다 텐서 이름을 찍어 거부하는 편이 쓸모 있다. `Florence2Model::load`와 `Florence2DaViT::load` 양쪽에서 `sanitize` 직후에 돈다.

### 3.4 선언된 비트 범위는 넉넉하게 뒀다

`quantization.bits`는 오늘 공개된 네 가지 폭이 아니라 1..=32로 잡았다. `mlxcel_core::layers::validate_quantization_params`와 같은 범위이고 이유도 같다. unified 레이어가 packed shape에서 실효 비트 폭을 다시 유도하니, 허용 목록을 두면 정당한 미래 export를 거부하게 된다. `group_size`는 1..=4096이다. 둘 다 정확성 검사가 아니라, 실제 텐서와 맞을 수 없는 값이 reconciler까지 흘러가지 않게 막는 울타리다.

---

## 4. 검증

### 4.1 수치

`tests/florence2_quantized_parity.rs`, 체크포인트 게이트, 테스트 네 개다. 이 장비에서 관측한 값은 다음과 같다.

```
image_features:  mean -0.004166 (ref -0.004168), std 0.815221 (ref 0.815222)
encoder_hidden:  mean  0.067354 (ref  0.067450), std 3.563928 (ref 3.566599)
step0 logits:    max abs deviation 0.0061 at index 0 (tol 0.05)
greedy ids:      4-bit [0, 879, 27740, 868], bf16 [0, 879, 27740, 868]
```

greedy 시퀀스는 이 입력에서 4-bit와 bf16이 동일하고, 4-bit 가중치 위의 upstream 결과와도 같다.

### 4.2 CLI

캡션과 검출은 COCO `000000039769`, OCR은 `HELLO MLXCEL`을 렌더링한 640x200 PNG를 생성해 썼다. 고양이 사진에는 글자가 없어서 `<OCR>`이 `-`를 돌려주는 게 맞기 때문이다.

| 태스크 | `base-ft-bf16` | `base-ft-4bit` | `base-ft-8bit` |
|---|---|---|---|
| `<CAPTION>` | Two cats are sleeping on a pink blanket. | Two cats laying on a pink blanket next to remotes. | Two cats are sleeping on a pink blanket. |
| `<OD>` | cat, cat, couch, remote, remote | cat, cat, couch, remote, remote | cat, cat, couch, remote, remote |
| `<OCR>` | HELLO MLXCEL | HELLO MLXCEL | HELLO MLXCEL |
| `<OCR_WITH_REGION>` | [43.2, 71.1, 599.4, 71.1, 599.4, 127.7, 43.2, 127.7] | [41.9, 70.9, 599.4, 70.9, 599.4, 127.7, 41.9, 127.7] | 미실행 |

8-bit는 dense 캡션을 단어 하나 안 틀리고 재현하고 박스도 0.7 px 안에서 맞춘다. 4-bit는 검출 레이블을 전부 유지하고 박스를 약 1 px 안에서 맞추며 텍스트를 정확히 읽고, 캡션만 다르게 표현한다. 자유 서술 생성의 표현이 바뀌는 것은 4-bit 가중치가 하는 일이지 배선 결함이 하는 일이 아니다. 정확성 논증을 이 표가 아니라 수치 고정에 두는 이유가 그것이다.

### 4.3 체크포인트 없이 도는 것

`src/models/florence2/florence2_quantized_tests.rs`에 가중치가 필요 없는 테스트 열 개를 뒀고, 그래서 CI에서도 돈다. config 파싱과 범위 가드, "양자화를 선언했다"와 "dense 기본값으로 파싱된다"의 구분(실제 4-bit group-64 export는 dense fallback 값과 정확히 같은 값을 선언하니 블록의 존재 여부만이 둘을 가른다), 좁힌 거부의 양방향, 그리고 세 평면짜리 `model.shared` 채우기를 덮는다.

---

## 5. 이슈 본문 교정

1. **group size는 달라지지 않는다.** 이슈는 달라질 수 있다고 경고한다. 여덟 변환본 모두 `group_size: 64`를 선언하고 `bits`만 바뀐다. 코드는 그래도 읽는다. `UnifiedLinear`가 선언된 group size를 신뢰하고 bits를 shape에서 다시 유도하니, 잘못된 값 하나가 전부를 조용히 어긋난 stride로 읽게 만든다.
2. **임베딩 테이블이 더 큰 절반인데 언급이 없다.** `model.shared`, `lm_head`, `embed_positions` 둘, `image_pos_embed` 둘이 packed다. 선형 프로젝션만 구현했다면 첫 조회에서 죽었다.
3. **`image_projection`은 양자화되지 않는다.** 이슈 문장은 그럴 수도 있는 것처럼 읽힌다. 이건 오른쪽 matmul 피연산자로 쓰이는 날 `nn.Parameter`라 `nn.quantize`가 아예 보지 못한다. temporal 버퍼와 conv 스택도 마찬가지다. 좁힌 거부가 지금도 막는 대상이 정확히 이 셋이다.
4. **활성 dtype과 sanitizer 채우기는 아예 언급이 없다.** 둘 다 2.2절에 있다.
5. **`convert_bf16_weights`는 한 파일이 아니라 세 파일에서 잘못되어 있었다.**
6. **4-bit `config.json`에는 bf16에는 없는 `vision_config.hidden_size: 768`이 있다.** 타워 폭이 `dim_embed`에서 오기 때문에 두 런타임 모두 무시한다. 이 체인의 앞선 이슈가 존재하지 않는 `hidden_size`를 지적한 적이 있어 기록해 둔다.

---

## 6. `large-ft`는 upstream에서도 망가져 있고, 여기서도 이미 망가져 있었다

이슈는 `large-ft` 스모크 테스트를 요구했다. 그 테스트가 무언가를 찾아냈는데, 이번 변경이 만든 것은 아니다.

`Florence-2-large-ft-4bit`는 깨끗하게 로드되고 패킹도 내부적으로 일관된다. 양자화 텐서 294개가 전부 자기 `scales` 폭에 대해 4-bit / group-64로 딱 맞고, dense 집합도 예상 그대로다. 그런데 모든 태스크에서 BOS만 줄줄이 뱉는다. 측정 네 개가 원인을 이 브랜치 밖에 둔다.

- `Florence-2-large-ft-bf16`을 **변경 전 `main` 바이너리**로 돌려도 `<CAPTION>`에서 같은 BOS 반복이 나온다. `<OD>`와 `<OCR>`은 정상이다.
- upstream mlx-vlm으로 `Florence-2-large-ft-4bit`를 돌리면 `greedy_ids = [0, 0, 0, ...]`이 나온다.
- upstream mlx-vlm으로 `Florence-2-large-ft-bf16`을 돌려도 같다.
- `base-ft`는 bf16, 4-bit, 8-bit 모두 두 런타임에서 동일한 코드로 정상이다.

참조 구현도 같은 식으로 동작하니, 여기 양자화 `large-ft` 경로가 dense 경로가 안 하는 무언가를 하고 있는 게 아니다. 진단은 `large-ft` 계열에 대한 별도 이슈의 몫이다. `docs/supported-models.md`는 이제 계열이 동작한다고 주장하는 대신 관측된 동작을 적는데, 이는 이 PR보다 앞선 잘못된 서술을 바로잡는 것이기도 하다. 스모크 목록에는 `large-ft-4bit`를 남겨 뒀다. 로드만으로도 패킹 경로가 `base-ft` 모양에 고착되지 않았음을 증명하고(`d_model` 1024, BART 12층, `dim_embed` 최대 2048), 단언하는 것은 로드와 로짓의 유한성뿐이다.

---

## 7. 학습 포인트

**묻고 있는 질문에 답할 수 있는 기준을 고른다.** 양자화 변경에서 본능적으로 손이 가는 것은 dense 체크포인트와의 차이인데, 그 비교로는 올바른 손실 구현과 잘못된 구현을 구분할 수 없다. 둘 다 dense에서 멀기 때문이다. 동일한 packed 바이트 위의 두 번째 구현은 구분할 수 있다. dense 비교도 자리를 얻긴 했지만, 임계값 검사가 아니라 교차 구현 동등성 검사로서다.

**packed 텐서의 shape는 그 텐서의 shape가 아니다.** 이 계열의 로드 시점 가드 셋이 저장된 폭을 모델 폭과 비교하고 있었고, 양자화 체크포인트에서는 셋 다 틀린다(`d_model`이 768인 자리에서 `cols`는 96이다). 올바른 도구는 `validate_quantized_packing`으로 트리에 이미 있었고, 실제 작업은 값이 아니라 비교 자체가 버그임을 알아채는 것이었다.

**config에서 유도한 술어는 자기가 무엇을 뜻하는지 말해야 한다.** `Florence2Quantization::DENSE`는 `{64, 4}`인데, 이는 진짜 4-bit group-64 export가 선언하는 값과 정확히 같다. 양자화 탐지를 `parsed != DENSE`로 했다면 모든 4-bit 체크포인트를 조용히 dense로 취급하고 bf16 scales를 승격시켰을 것이다. 그 정보를 담고 있는 것은 블록의 존재 여부뿐이고, `config_is_quantized`가 별도 술어로 존재하고 자기 테스트를 갖는 이유가 그것이다.

**"로드된다"만 단언하는 스모크 테스트는 결국 판단해야 할 무언가를 물어 온다.** `large-ft` 결과는 이 이슈의 실패가 아니고 여기서 고치는 것은 범위 확장이었겠지만, 체크포인트를 목록에서 조용히 빼면 진짜 결함이 묻힌다. 기록하고 문서를 바로잡고 로드 커버리지는 남기는 것이 정직한 중간이다.

---

## 8. 다루지 않은 것

- **`mlxcel-server`는 여전히 시작 시 Florence-2를 거부한다.** 설계상 범위 밖이고 이슈 #1073이 바로 다음에 처리한다. 여기서 `src/server/`는 손대지 않았다.
- **`large-ft` 이상 동작은 이 변경 밖에 있다는 것까지만 규명했다.** 원인에 대한 가설은 제시하지 않았고, 6절에서 그런 것을 읽어 내서도 안 된다.
- **3-bit와 6-bit는 종단 간으로 돌리지 않았다.** 같은 `group_size: 64` 패킹을 선언하고 같은 경로로 로드되며, 디렉터리가 있으면 비트 폭 스모크 테스트가 자동으로 잡는다. 다만 이번 실행에서는 내려받지 않았다. 내려받아 돌린 것은 `base-ft`의 4-bit와 8-bit, 그리고 `large-ft`의 4-bit다.
- **성능 측정은 없다.** 이번 변경은 기능 추가이고, 이 하드웨어에서 4-bit Florence-2가 bf16보다 빠르게 디코딩하는지는 측정하지 않았으며 그에 대한 주장도 하지 않는다.

---

## 참고

- 이슈 #1072, 그리고 에픽 체인 #850, #852, #853, #854, #855, #856
- `tests/florence2_quantized_parity.rs`. 모듈 주석에 기준값 재생성 레시피가 적혀 있다
- mlx-vlm florence2: [florence2.py](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/florence2/florence2.py), [language.py](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/florence2/language.py), [vision.py](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/florence2/vision.py)
- PR #1078. 여기서 넣은 `dequantize` biases 가드가 이번 변경이 의존하는 양자화 임베딩 경로를 덮는다
