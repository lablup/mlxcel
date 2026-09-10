# 기술 보고서: PR #1739 - feat(models): add the Laguna family with compressed-tensors NVFP4 experts

**날짜**: 2026-09-10
**작성**: mlxcel 메인테이너
**검토**: 구현 및 보안 리뷰 사이클
**상태**: 완료 (계열 테스트 21개, `models::` 스위트, clippy, fmt, CI 모두 통과. M5 Max에서 공개 체크포인트 두 개 실행. 이 계열에는 외부 토큰 일치 레퍼런스가 없고 Laguna S 2.1은 내려받지 않았다)
**언어**: Rust, Markdown
**위험도**: 중간 (새 계열에 더해, 다른 모든 계열이 함께 쓰는 코드 두 곳을 건드렸다. `SwitchLinear`의 전문가별 출력 배수와, 이제 여덟 개 토크나이즈 지점이 직접 적는 대신 호출하는 BOS 규칙이다)

---

## 요약

Poolside의 Laguna(`model_type: "laguna"`, XS 2.1 / XS.2 / S 2.1)는 슬라이딩과 풀 어텐션을 섞은 MoE 코드 모델인데 공개된 NVFP4 체크포인트는 라우팅 전문가와 공유 전문가를 전부 compressed-tensors `nvfp4-pack-quantized` 레이아웃으로 담고 있다. `src/` 안에는 이 레이아웃을 읽는 코드가 없었다. `sanitize.rs`에 NVFP4 리팩이 있기는 하지만 `.weight_scale_2` 키를 기준으로 삼는 ModelOpt 표기 전용이고 compressed-tensors 삼중조(`weight_packed` / `weight_scale` / `weight_global_scale`)를 읽는 곳은 트리 어디에도 없었다. 이 PR은 `laguna.rs`, `laguna_layers.rs`, `laguna_sanitize.rs` 세 모듈로 계열을 추가하고 로드 시점에 그 레이아웃을 MLX 네이티브 NVFP4 플레인으로 옮긴다. MLX 네이티브 NVFP4에 자리가 없는 스칼라 하나는 라우팅 행렬곱 뒤에 적용하는 전문가별 사이드카로 들고 다닌다.

변경 두 개는 Laguna 밖까지 미친다. `SwitchLinear::Quantized`에 `[num_experts]` 출력 배수를 선택적으로 붙였는데 밀집 선형에 이미 있던 `QuantizedWeight::apply_global_scale`의 라우팅 판본이다. 그리고 `MlxcelTokenizer::prompt_carries_bos`가 렌더링된 프롬프트를 토크나이즈하는 모든 지점에서 `<bos>` / `<s>` 리터럴 접두 검사를 대체했다. 덕분에 Llama 3 계보 전체가 지고 있던 오프바이원도 함께 사라진다. 그쪽 채팅 템플릿은 `<|begin_of_text|>`를 내보내는데 체크포인트의 `TemplateProcessing` 후처리기가 같은 id를 한 번 더 붙이고 있었다.

---

## 1. 읽는 코드가 없던 레이아웃, 그리고 갈 곳 없는 스칼라

### 1.1 텐서 셋이 들어와 텐서 셋이 나간다

compressed-tensors NVFP4는 양자화된 선형 하나를 텐서 셋으로 저장한다. `weight_packed`(E2M1 코드, 바이트당 둘), `weight_scale`(입력 피처 16개당 하나인 E4M3 블록 스케일), `weight_global_scale`(텐서 전체에 하나인 f32)다. 한 행이 복원하는 값은 `code * E4M3(scale) / global`이고 MLX 네이티브 NVFP4는 앞의 둘만 받으며 셋째를 넣을 자리가 없다.

라우팅 쪽은 `laguna_sanitize::transcode_expert_planes`가 맡는다. MoE 레이어마다, 그리고 `gate_proj` / `up_proj` / `down_proj`마다 `num_experts`개의 삼중조를 가져와 셋을 각각 새 선두 축으로 쌓은 다음 최소한의 작업만 한다.

- 패킹된 바이트는 변환하지 않고 재해석한다. `view_packed_as_u32`는 버퍼를 연속으로 만들고 `uint32`로 view하는데 바이트당 E2M1 코드 둘을 낮은 니블부터 놓는 순서가 MLX 네이티브 NVFP4의 워드당 코드 여덟 개 순서와 정확히 같기 때문이다. 그 함수에 복사가 한 번 있는 이유는 로더가 돌려주는 `U8` 배열이 버퍼를 공유할 수 있어서이고 `INT8` 플레인은 제자리에서 view한다.
- E4M3 블록 스케일은 로더가 `U8`로 넘겨준 경우 바이트 단위로 그대로 둔다. 로더가 `F8_E4M3`를 부동소수로 승격해 버린 경우에는 `encode_block_scales`가 f32로 풀었다가 다시 인코딩하는데 원래 E4M3에서 온 값이라면 무손실이다. 이 재인코딩은 최대 16개 청크로 나눠 병렬로 돌린다. 플레인이 234개면 그만한 값어치가 있다.
- `weight_global_scale`은 `1 / global`이 되어 f32 `[num_experts]` 벡터로 `{prefix}.global_scale`에 들어간다.

밀집 쪽(`transcode_dense_planes`)은 새 소비자가 필요 없었다. `UnifiedLinear`는 Inkling ModelOpt 작업 때부터 이미 `{prefix}.global_scale` 키를 읽고 `apply_global_scale`도 갖고 있다. 그래서 양자화되지 않은 키 이름을 쓰는 평범한 SwiGLU MLP인 공유 전문가는 `[1]` 사이드카를 달고 기존 밀집 경로로 로드된다.

### 1.2 `1 / global`을 블록 스케일에 접지 않은 이유

가장 손쉬운 단순화는 트랜스코드 때 `1 / global`을 E4M3 블록 스케일마다 곱해 넣고 사이드카 없이 MLX가 그대로 먹는 플레인을 내보내는 것이다. 이 방법은 버렸다. E4M3는 가수 비트가 3개라 몫이 대개 표현 가능한 값이 아니고 접어 넣으면 체크포인트가 이미 한 번 양자화한 위에 두 번째 양자화를 얹게 된다. 블록 스케일은 이 레이아웃에서 트랜스코드를 그대로 통과하는 유일한 부분이므로, 정확하게 유지하는 값이 라우팅 행렬곱당 곱셈 하나보다 크다.

사이드카의 대가는 숨기지 않고 코드에 적어 두었다. `SwitchLinear::quantized_ref`는 `global_scale`을 든 플레인에 `None`을 돌려주므로 그런 플레인은 융합 affine 디코드 커널에 들어가지 못하고 항상 `gather_qmm` 경로를 탄다. NVFP4 체크포인트에서 융합 실행과 비융합 실행이 바이트 단위로 같은 이유가 이것이다(5.2절). 우연이 아니라 정확성 보장이고 대신 NVFP4 체크포인트는 폭 1에서 융합 커널이 벌어 줬을 몫을 포기한다.

### 1.3 곱셈이 일어나는 위치

`apply_expert_global_scale`은 `gather_qmm` 앞이 아니라 뒤에서 돈다. 행렬곱이 쓴 것과 같은 인덱스 텐서로 선택된 전문가의 항목을 `take`하고 출력 랭크에 맞춰 뒤쪽 축을 늘리고 곱한 다음 출력 dtype으로 되돌린다. 가중치가 아니라 출력에 적용하므로 스케일은 forward당 `num_experts`번이 아니라 토큰당 `k`번 곱해지고 양자화 플레인 자체는 손대지 않으므로 같은 가중치를 두 번 로드해도 결과가 같다.

---

## 2. `config.json`은 신뢰할 수 없는 입력이다

`config.json`은 `mlxcel generate -m <org>/<repo>` 경로로 로더에 들어오고 Laguna의 값 몇 개는 거기서 MLX까지 인자 그대로 간다. 보안 패스(커밋 `d00262ea`)는 끝이 나쁜 경로 네 가지를 막았는데 넷 다 조용하거나 늦게 터지는 종류다.

| 위험 | MLX까지 간 것 | 실패 모양 | 방어 |
|---|---|---|---|
| `num_experts_per_tok` 누락 | 점수 행의 끝을 하나 넘어선 `argpartition(kth = num_experts)` | MLX가 throw하고 그 throw는 cxx 브리지를 건너 잡을 수 없는 abort가 된다. 체크포인트가 전부 메모리에 올라간 뒤 첫 라우팅 토큰에서 터진다 | `ModelArgs::validate`, `validate_router_geometry`, `router_select` 안의 clamp |
| `num_experts` 과소 선언 | 쌓인 플레인이 덮지 못하는 라우터 인덱스 | `gather_qmm`도 사이드카 뒤의 `take`도 양수 인덱스를 범위 검사하지 않으므로, 범위 밖 읽기 결과가 그대로 로짓까지 간다 | `validate_router_geometry`가 라우터 폭과 각 플레인을 대조하고 `check_no_expert_beyond`가 선언보다 많은 전문가를 든 체크포인트를 거부한다 |
| 전문가 플레인의 shape·dtype 불일치 | `mlx::core::stack` | 같은 잡을 수 없는 abort가 로드 도중에 난다. dtype이 섞여 승격되면 더 나쁘다. 패킹 view가 잘못된 폭으로 재해석하고 아무 불평 없이 로드된다 | 플레인을 맵에서 빼기 전에 `PlaneLayout` 동일성을 전체 구간에서 검사 |
| `head_dim < 2` | 부분 회전 폭을 구하는 `Ord::clamp(2, head_dim)` | `min > max`로 Rust 패닉 | `ModelArgs::validate`가 로드에서 거부하고 헬퍼 자체의 하한도 `head_dim`을 따라가므로 `ModelArgs`를 직접 만든 호출자도 패닉하지 않는다 |

넷 중 셋은 결함의 모양이 같다. 하나 어긋난 값을 타입이 잡지 않고 그 값이 도달하는 MLX 커널도 범위를 검사하지 않는다. 결과는 16GB를 이미 올려 둔 채 나는 abort이거나 다른 어떤 숫자와도 구별되지 않는 틀린 숫자다. `num_experts_per_tok`은 serde 기본값만으로 첫 상태에 도달한다. `mlp_layer_types`로 희소 레이어를 선언하면서 이 키를 적지 않은 설정은 0을 주고 그러면 `kth`가 전체 전문가 수가 된다. 반대 방향 값이 오히려 조용했다. `kth`가 음수가 되고 `slice`가 음수 시작을 받아서 블록은 `k`보다 적은 전문가로 라우팅하고 어디에도 아무것도 보고하지 않았다.

라우터를 두 곳에서 막은 것은 의도한 배치다. `ModelArgs::validate`는 가중치를 읽기 전에 도므로 잘못된 설정이 16GB 로드 뒤가 아니라 밀리초 안에 실패하고 체크포인트가 뒤에 없는 설정까지 볼 수 있는 유일한 방어다. `validate_router_geometry`는 로드된 텐서를 상대로 도는데 거기서는 라우터 폭이 설정 키가 아니라 shape이므로 자기 가중치와 어긋나는 `config.json`을 잡을 수 있는 쪽은 이쪽뿐이다. `router_select` 안의 clamp는 세 번째 층이고 블록을 직접 만드는 호출자, 특히 유닛 테스트를 위해 있다.

`check_no_expert_beyond`에는 놓치기 쉬운 성능 이유도 있다. 이것이 없으면 40레이어 256전문가 export가 전문가 8개를 선언했을 때 앞의 8개를 쌓고 남은 플레인을 전부 `transcode_dense_planes`로 보내 아무도 읽지 않는 밀집 선형으로 만든다. 상관없는 이유로 로드가 실패하기 전까지 f32 구체화와 E4M3 재인코딩을 수만 번 하는 셈이다.

---

## 3. BOS 규칙, 그리고 카운트 경로 두 개

### 3.1 리터럴 접두 검사가 못 본 것

트리의 모든 토크나이즈 지점은 `add_special_tokens`를 정할 때 `!prompt.starts_with("<bos>") && !prompt.starts_with("<s>")`를 물었다. Laguna의 채팅 템플릿은 `〈|EOS|〉`, 즉 id 2로 시작하는데 이 체크포인트는 그 id를 BOS와 EOS 양쪽으로 쓰고 `tokenizer.json` 후처리기는 `single: [〈|EOS|〉, A]`인 `TemplateProcessing`이다. 리터럴 어느 쪽도 걸리지 않으니 모든 채팅 요청이 id 2 두 개로 시작했을 것이다.

`MlxcelTokenizer::prompt_carries_bos`는 리터럴 둘을 남기고 그 위에 토크나이저 자신의 BOS 표기를 더한다. 표기는 `bos_token_id()`와 `id_to_token`으로 구한다. 이제 여덟 개 호출 지점이 여기에 위임한다. 서버 스케줄러의 디스패치 스레드 토크나이저, 라우터 프런트엔드, 디퓨전 워커의 요청 경로 둘, XLA admission 워커, 오프라인 `generate`, `chat`, 벤치마크 바이너리 둘이다. `model_worker.rs`의 순서 있는 미디어 경로 둘은 답을 파라미터로 받아 세그먼트마다가 아니라 요청마다 한 번만 계산한다. 디스패치 스레드의 사전 토크나이즈가 스케줄러 스레드의 토크나이즈와 바이트 단위로 같아야 한다는 #633 불변식이 이들을 정의 하나로 묶는 이유다.

이 규칙은 렌더링된 텍스트가 이미 들고 있는 BOS만 뺄 수 있고 유닛 테스트가 양쪽 방향을 다 고정한다. 렌더링된 프롬프트는 BOS id를 정확히 하나 받고 BOS 텍스트를 담지 않은 프롬프트도 정확히 하나를 받는다.

### 3.2 카운트 경로는 별개의 결함이었다

`/tokenize`와 `/v1/messages/count_tokens`는 둘 다 `add_special = true`를 무조건 넘기면서 그래야 숫자가 `tokens_evaluated`와 비교 가능해진다는 주석을 달고 있었다. 실제로는 반대였다. 자기 BOS를 내보내는 템플릿이라면 그 카운트는 요청이 실제로 평가할 값보다 하나 크다. 이제 두 경로 모두 `prompt_carries_bos`를 따르고 Laguna 체크포인트에서는 차이가 51 대 52로 그대로 보인다.

이 PR에서 파급 범위가 가장 넓은 대목이다. Llama 3 계보가 같은 템플릿 모양을 갖고 있어서 그 경로들이 존재한 내내 하나씩 더 세고 있었다. 상류 `apply_chat_template`이 `add_special_tokens=False`를 넘기는 이유도 정확히 이것이다.

---

## 4. mlx-lm을 따르지 않은 두 곳

### 4.1 `attention_factor`를 반영한다

mlx-lm의 `laguna.py`는 YaRN 블록의 `attention_factor` 키를 버리고 언제나 `factor`에서 계수를 유도한다. `transformers.modeling_rope_utils._compute_yarn_parameters`는 설정이 값을 들고 있으면 그 값을 쓰고 공개 체크포인트마다 함께 실려 오는 `modeling_laguna.py`가 그 동작을 물려받는다. 이 포트는 체크포인트 자신의 모델링 파일을 따르므로, `layer_rope`는 엔트리가 `attention_factor`를 선언하면 유도된 `YarnRope.mscale`을 덮어쓴다.

이 선택이 관측되는 공개 체크포인트는 딱 하나다. `mlx-community/Laguna-XS.2-4bit`는 유도값이 1.3466인 자리에 `attention_factor: 1.0`을 선언하므로 두 런타임이 이 체크포인트를 서로 다르게 회전시킨다. XS 2.1은 유도값과 같은 값을 선언하니 양쪽이 일치한다. XS.2로 이 포트와 mlx-lm을 대조하는 사람은 다른 데가 아니라 여기서 차이를 만나게 되고 모듈 헤더가 이 항목을 무엇보다 먼저 적어 둔 이유가 그것이다.

### 4.2 어텐션 레이어가 각자 마스크를 만든다

셸이 프리필 마스크를 하나 만들어 내려보내지 않는다. `Attention::forward`는 키 축을 캐시 오프셋이 아니라 캐시가 실제로 돌려준 텐서에서, 즉 `mlxcel_core::array_shape(&cache_k)[2]`에서 구하고 풀 레이어에는 평범한 인과 밴드를, 슬라이딩 레이어에는 창으로 자른 밴드를 만든다.

돌려받은 텐서에서 구하는 것이 슬라이딩 레이어를 서로 다른 캐시 구현 둘에 대해 정렬시킨다. 평범한 `RotatingKVCache`는 이전 키를 `min(offset, window - 1)`개 노출하고 추측 롤백(#1351)이 장착할 버퍼 판본은 `window + buffer`까지 노출한다. 오프셋에서 크기를 잡은 마스크는 한쪽에는 맞고 다른 쪽에는 틀리는데 여기서 틀린다는 것은 창 경계를 넘어 어텐션한다는 뜻이고 그 결과는 에러가 아니라 그럴듯한 문장이다.

같은 종류의 앞선 대비가 `forward_with_capture`와 `LagunaCache::trim`에도 있다. 둘 다 DFlash 드래프터용으로 준비만 해 둔 것이고 오늘은 도달할 수 없다. 살아 있는 코드처럼 보이게 두는 대신 그렇다고 문서에 적었다.

---

## 5. 검증

호스트는 M5 Max 128GB, Metal이다. 아래는 전부 최종 커밋에서 다시 실행했다.

### 5.1 게이트

| 게이트 | 결과 |
|---|---|
| `cargo test --profile test-fast --features metal,accelerate --lib models::` | 1647 통과, 0 실패, 65 무시. `models::laguna_tests`가 그중 21개 |
| `cargo clippy`(범위 한정), `cargo fmt --all -- --check` | 깨끗함 |
| PR의 CI | 통과 |
| GB10(sm_121) 호스트에서 `--features cuda`로 같은 스위트 | 이전 커밋에서 통과 |
| `cargo test --workspace --profile test-fast --features metal,accelerate` | 이 공용 호스트에서는 통째로 돌리지 않았다. 워크스페이스 게이트는 CI가 돌린다 |

계열 테스트 21개는 체크포인트가 필요 없다. 설정 파싱과 두 스케줄, 두 레이어 타입의 RoPE 해석, 라우터 점수 함수 셋, `scores + bias`로 고르고 가중치는 바이어스 없는 `scores`에서 가져오는 선택 규칙, 헤드별 게이트와 폭 불일치 거부, 새니타이즈 네 경로 전부, 두 레이어 타입의 프리필 인과성, 디코드 로짓과 프리필 로짓의 일치, 그리고 2절의 로드 경로 거부 다섯 가지를 덮는다.

### 5.2 `models/laguna-xs-2.1-nvfp4`, 네이티브 compressed-tensors 경로

`mlxcel generate -m models/laguna-xs-2.1-nvfp4 -p "Write a Python retry wrapper with exponential backoff." -n 64 --no-chat-template --temp 0`은 플레인 234개를 트랜스코드하고 3.2초에 16.52GB로 로드한 다음 53~79 tok/s로 정상적인 Python을 쓴다. 234는 MoE 레이어 39개 곱하기 라우팅 프로젝션 3개(쌓인 플레인 117개)에 같은 레이어의 공유 전문가 삼중조(밀집 플레인 117개)를 더한 값이고 그 export에서 압축된 부분은 이게 전부다.

`MLXCEL_FUSED_MOE=0`은 바이트 단위로 같은 텍스트를 낸다. 놀랄 일이 아니라 1.2절이 예고한 결과다. `global_scale`을 든 플레인은 융합 디코드 커널에서 제외되므로 두 실행 모두 같은 `gather_qmm`을 탄다.

### 5.3 `models/laguna-xs.2-4bit`, 그리고 이 포트 탓이 아닌 갈림

미리 쌓인 affine 체크포인트는 같은 명령을 비융합 121~128 tok/s, 융합 108 tok/s로 돈다. 두 그리디 이어쓰기는 토큰 10 근처부터 갈린다. 다른 것은 기존 affine 디코드 커널의 융합 실행과 비융합 실행이지 이 계열의 무엇이 아니다. `models/qwen3-30b-a3b-4bit`가 같은 호스트에서 같은 A/B로 같은 지점에서 갈리고 그 체크포인트는 이 브랜치보다 한참 앞선다. 대조군을 돌린 덕분에 '새 계열이 `MLXCEL_FUSED_MOE`에서 불안정하다'가 '이 호스트의 융합 affine 커널이 시도한 모든 MoE 체크포인트에서 갈린다'로 바뀌었다. 둘은 주인이 다른 별개의 버그다.

### 5.4 서버

NVFP4 체크포인트를 올린 `mlxcel-server`, 포트 19347:

- `/apply-template`은 `〈|EOS|〉`로 시작하는 프롬프트를 렌더링한다.
- `/tokenize`는 `add_special=false`에서 id 51개(`[2, 97, ...]`), `add_special=true`에서 52개(`[2, 2, ...]`)를 돌려준다. 중복된 BOS가 눈에 보이는 형태다.
- `enable_thinking=false`로 보낸 `/v1/chat/completions`는 `prompt_tokens = 51`을 보고해 토크나이즈 경로와 맞고 허용 토큰 700개 중 672개에서 `finish_reason: "stop"`으로 끝나며 본문에 `</assistant>`가 없다. 길이 상한이 아니라 id 24가 정지를 발화했다.
- 카운트 경로 둘 다 `input_tokens: 51`로 답한다.

---

## 6. 확인하지 못한 것

**외부 레퍼런스와의 토큰 일치.** 고정된 mlx-lm 체크아웃(0.31.3)에는 `laguna.py`가 없으므로 그리디 id를 대조할 오라클이 두 호스트 어디에도 없다. 이 트리에서 토큰 일치를 주장하는 계열은 전부 오라클을 갖고 있는데 Laguna에는 없다. 현재 주장할 수 있는 것은 체크포인트 둘이 로드되고 실행되며 예상 속도로 주제에 맞는 정상적인 코드를 낸다는 사실까지다. 5.2절의 융합/비융합 A/B는 자기 자신과의 비교이지 패리티 결과가 아니다.

**Laguna S 2.1.** 100GB 가량이고 내려받지 않았다. 48레이어 설정 경로, `factor: 128` / `beta_fast: 32` YaRN 블록, 72헤드 슬라이딩 레이어는 유닛 테스트로만 덮인다.

**`sqrtsoftplus` 라우터와 어텐션 싱크.** 둘 다 설정 의미에서 구현했고 공식과 대조해 테스트했지만 어느 공개 체크포인트도 켜지 않는다. `swa_attention_sink_enabled`는 설정 셋 모두에 없고 `moe_router_score_func`도 셋 모두에 없으므로 두 분기는 테스트에서만 돈다.

**어댑터 경로.** `loading/config_backed.rs`는 새니타이즈를 거치지 않고 `LagunaModel::from_weights`에 도달하므로 NVFP4 체크포인트는 거기서 가중치 없음 에러로 실패한다. Laguna는 `adapter: None`으로 등록했고 다른 `ConfigBacked` 계열도 전부 같은 배선이니, 이 계열이 아니라 그 경로의 성질이다.

**텐서 병렬과 배칭.** 이 계열은 TP 배선이 없고 풀/슬라이딩 혼합 캐시가 시퀀스별 KV 격리와 공유할 수 없는 `RefCell`에 살기 때문에 `supports_batching`이 false를 돌려준다.

성능 항목 하나는 검토하고 그대로 두었다. `prompt_carries_bos`는 호출마다 `bos_token_id()`를 다시 계산하는데 토크나이저 인코드 두 번이고 지점이 여덟 개다. 요청 하나에 견주면 마이크로초 단위이고 캐시를 두면 지금 상태가 없는 타입에 상태를 더하게 된다.

---

## 7. 변경 요약

### 통계

| 지표 | 값 |
|---|---|
| 변경 파일 | 31 |
| 추가 줄 | 3271 |
| 삭제 줄 | 44 |
| 새 모듈 | 3개 (`laguna.rs` 672, `laguna_layers.rs` 758, `laguna_sanitize.rs` 489) |
| 추가 테스트 | 계열 21개, 토크나이저 2개, 탐지 1개 |

### 영역별 변경

- `src/models/laguna.rs`: 설정, 레이어별 RoPE 해석(레이어 타입별 엔트리를 느슨한 최상위 스칼라 위에 덮어쓰고 명시된 `attention_factor`를 우선하며 `partial_rotary_factor` 폴백 사슬을 탄다), `validate`, 모델 셸, 혼합 캐시를 소유하는 `LanguageModel` 래퍼.
- `src/models/laguna_layers.rs`: 레이어별 헤드 수와 부분 YaRN 회전을 쓰는 게이트 QK-norm 어텐션, 레이어별 마스크, 라우터와 선택 규칙, `validate_router_geometry`, 공유 전문가를 포함한 MoE 블록, 밀집 MLP, 디코더 레이어.
- `src/models/laguna_sanitize.rs`: compressed-tensors 트랜스코드, 미리 쌓인 `gate_up_proj` 분할, 전문가별 bf16 쌓기, 라우터 키 이름 변경, 묶인 헤드 제거. 설계상 두 번 돌려도 결과가 같다.
- `src/models/switch_layers.rs`: `SwitchLinear::Quantized`의 `global_scale` 필드, `apply_expert_global_scale`, 융합 경로 제외, 그리고 사이드카 shape을 쌓인 플레인 수와 대조하는 검사.
- `src/tokenizer/mod.rs`: `prompt_carries_bos`와 `bos_token_string`, Laguna 모양과 Llama 3 모양을 각각 고정하는 테스트.
- 토크나이즈 지점 여덟 곳과 카운트 경로 둘을 공용 규칙으로 전환.
- 탐지, 레지스트리, 메타데이터, `LoadedModel`, 메모리 추정, TP 폴백 표, `--swa-full` 메시지, `docs/supported-models.md`.

### 커밋

| 해시 | 유형 | 제목 |
|---|---|---|
| `e9bc766b` | feat | add the Laguna family with compressed-tensors NVFP4 experts |
| `e78d14eb` | docs | record what Laguna was validated on, and the YaRN gap |
| `e9863f07` | fix | count prompt tokens with the generation add_special rule |
| `d00262ea` | fix | bound the router and expert-plane geometry at load |

### 관련 이슈

이 PR은 #1347을 닫는다. #633의 토크나이즈 불변식을 건드린다. `LagunaCache::trim`과 캡처 분기를 #1351용으로 준비해 둔다.

---

## 8. 후속 작업

**패리티 오라클이 없다.** 만들려면 고정된 mlx-lm 체크아웃에 `laguna.py`를 포팅하거나 체크포인트의 `modeling_laguna.py`를 `transformers` 아래로 들여온 다음, 같은 프롬프트에서 그리디 id를 대조해야 한다. 그때까지는 `docs/supported-models.md`가 실행으로 확인된 범위만 적는다. 해결이 아니라 완화다.

**XS.2는 mlx-lm과 어긋난다.** 4.1절의 `attention_factor` 결정은 의도한 것이고 세 곳에 적어 두었지만 패리티 오라클을 처음 만드는 사람은 이 지점에 걸리게 된다. 결함으로 다루기 전에 4.1절을 먼저 읽어야 한다.

**융합 affine 디코드의 갈림.** 5.3절은 이 계열의 문제가 아님을 배제하고 원래 자리에 두었다. 같은 호스트의 Qwen3 MoE에서 재현되므로 별도 이슈로 다룰 값이 있다.

### 옮겨 갈 만한 교훈

2절의 위험 네 가지는 모양이 하나다. 어떤 타입도 검사하지 않는 설정 스칼라가, 결국 도달하는 MLX 커널에서도 범위 검사를 받지 않고 실패는 16GB를 이미 올려 둔 abort이거나 다른 어떤 숫자와도 구별되지 않는 숫자로 나온다. 이것들을 찾아낸 것은 새 코드를 더 꼼꼼히 읽어서가 아니었다. 새 모듈이 선언하는 설정 필드마다 그것이 어떤 MLX 인자가 되는지, 그 인자가 범위 밖 값에 무엇을 하는지 물었기 때문이다. 라우터에 대해서는 `afmoe`와 `bailing_moe`가 이미 그 질문에 답해 두었고 그쪽 가드가 본이 됐다. NVFP4 고유의 둘, 즉 전문가 플레인 레이아웃 검사와 `head_dim` clamp는 선례가 없었고 다른 계열이 쓰지 않는 레이아웃에서 같은 경로를 따라가며 나왔다. 새 모델 계열은 대체로 번역 작업이고 번역이 아닌 부분은 정확히 아무도 인자 경로를 걸어 본 적 없는 부분이다.
