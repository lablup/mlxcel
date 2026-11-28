# PR #3367: Inkling 2 Small 및 Mini (`inkling2`, `inkling2_loop`)

**날짜**: 2026-11-24
**상태**: Draft PR 열림 (#3228 종료). 범위를 좁힌 단위 테스트 통과. M3 Ultra(512 GB)에서 Inkling 2 Mini 체크포인트 두 개를 mlx-vlm 466069f5와 대조해 검증함

## 요약

Inkling 2는 이름과 몇몇 프리미티브를 빼면 기존 Inkling 포트와 공유하는 것이 거의 없는 새 아키텍처다. 이 PR은 텍스트 디코더, 루프를 도는 Mini 변형, 타일 ViT, 네이티브 혼합 정밀도 로더와 Transformers 레이아웃 로더, Inkling 도구 호출 파서를 추가하고, 이를 `mlxcel generate`, `mlxcel-server`, `mlxcel arch`에 연결한다. 네이티브 Mini 체크포인트에서 CLI는 mlx-vlm의 greedy 토큰 128개를 그대로 재현한다. teacher-forced trace에서는 네이티브 텍스트(0/490), 네이티브 이미지(0/76), 4비트 익스포트(0/494) 모두 결정된 위치에서 불일치가 하나도 없다. 루프를 1패스로 강제하면 출력이 망가지므로, 루프와 메시지 컨트롤러가 실제로 연결되어 있음을 확인할 수 있다. 디코드 속도는 두 체크포인트 모두 약 32 tok/s다. Metal 커널(#3359), DSpark 드래프터, 오디오는 범위 밖이며, Small은 실제 체크포인트로 실행하지 않았다.

## 1. 문제

#3228은 공개된 두 형태를 모두 요구했다. Small(`inkling2`, 48층, 전문가 384개, 약 393 GB)과 Mini(`inkling2_loop`, 물리 24층 중 가운데 12층을 세 번 실행, 전문가 128개, 약 20 GB)다. 기존 Inkling 포트로는 둘 다 로드할 수 없었다. 이 아키텍처는 블록 attention residual, 쿼리 의존 attention sink, 동적 short convolution, q/k/v 입력별 정규화, sigmoid 라우팅, softcap이 걸린 SwiGLU 전문가, 패스 사이를 잇는 메시지 컨트롤러가 붙은 루프 코어, 그리고 HMLP 타워 대신 RoPE ViT를 추가한다. 서버도 이 계열의 도구 호출을 돌려주지 못했다. `<|content_invoke_tool_json|>`를 처리하지 않아서 호출이 함수 이름이 앞에 붙은 본문 텍스트로 나왔다.

| 위험 | 영향 | 가능성 |
|---|---|---|
| Inkling 2 체크포인트 로드 실패 | 높음: 계열 전체를 쓸 수 없음 | 이 PR이 없으면 확실 |
| 도구 호출이 텍스트로 반환됨 | 서버에서 에이전트로 쓸 때 높음 | 파서가 없으면 확실 |

## 2. 설계

### 2.1 모듈 구성

| 영역 | 파일 |
|---|---|
| 텍스트 디코더 | `src/models/inkling2.rs`(모델, 루프, logits), `inkling2/{config,quant,sanitize,conv,attention,moe,residual,layer,runtime}.rs` |
| 비전 | `src/vision/processors/inkling2.rs`, `src/vision/encoders/inkling2_vit.rs`, `src/vision/inkling2_vl.rs` |
| 로딩과 연결 | `src/loading/vlm_inkling2.rs`, `ModelType::Inkling2VLM`, `LoadedModel::Inkling2VLM`, 감지, 레지스트리, 메타데이터, `vlm_runtime` |
| 도구 호출 | `src/server/tool_calls/inkling.rs`, `parser.rs`, `stream_filter.rs` |

Inkling 1 파일은 바꾸지 않았다. 예외는 sanitize 헬퍼 세 개와 `repeat_kv_heads`로, Inkling 2가 재사용할 수 있도록 crate 가시성으로만 넓혔다. Inkling `--lib inkling` 테스트 129개는 통과한다.

### 2.2 레이어 방문과 캐시

`layer_indices`는 각 방문이 어느 물리 레이어를 실행하는지 나열한다. 형태는 `[0..core_start) + [core_start..core_end) * passes + [core_end..n)`이고, Mini에서는 6 + 36 + 6 = 48회 방문이다. 방문마다 고유한 `InklingLayerCache`(KV 캐시 하나와 f32 convolution 이력 네 개)를 가지므로, `num_layers()`와 모델이 소유한 시퀀스 상태는 둘 다 방문 수를 센다. 메시지 컨트롤러는 이전 패스의 서브레이어 입력을 현재 패스에 섞는다. 작은 루프 모델 테스트는 컨트롤러를 0으로 만들면 출력이 달라지는지로 이를 확인한다.

### 2.3 회전 캐시 없이 구현한 attention sink (upstream f2956e68)

upstream은 루프 모델의 모든 방문에 `AttentionSinkKVCache`를 쓴다. 이 캐시는 `max(0, end - max_size + 1)`부터의 키를 돌려주고, trim은 `end - max_size`까지만 가능하다. 이 포트는 방문마다 시간 순서대로 쌓는 `KVCache`를 둔다. sliding 방문은 키를 최대 `L + W - 1 + 64`개(`SLIDING_KV_MARGIN`이 64) 보관하고, 그중 최신 `L + W - 1`개에만 attention을 건다. 호출할 때마다 0으로 채운 K/V 행을 붙이고 banded mask 뒤에 sink 열을 덧붙인다. 그 결과 upstream 캐시가 돌려주는 것과 정확히 같은 키를 본다.

KV의 64키 여유와 convolution 이력의 16행 여유(`CONV_HISTORY_MARGIN`) 덕분에, 끝에서 16토큰 이하를 trim하면 KV와 convolution 상태가 정확히 되감긴다. Inkling 1은 trim할 때 convolution 상태를 지운다. `trim_caches`는 아무것도 바꾸기 전에 모든 방문을 먼저 검사하므로, 너무 깊은 trim은 상태를 그대로 둔 채 실패한다. decode-after-prefill, chunked prefill, trim 테스트가 이 레이아웃을 검증한다. 10토큰 trim은 성공하고 새로 실행한 forward와 일치하며, 40토큰 trim은 상태 변경 없이 거부된다.

### 2.4 체크포인트 레이아웃

| 레이아웃 | 밀집 가중치 | 전문가 |
|---|---|---|
| 네이티브(`model.llm.*`, `model.visual.*`) | BF16/FP32 cast 그룹 | `hf_quant_config.json`에 따라 NVFP4(0-19층)와 MXFP8(20-23층). u8 코드는 u32로 view하고, global scale은 `2688 * amax / denominator` |
| mlx-community 4비트(`model.language_model.*`, `model.vision_tower.*`, `lm_head.*`) | 비전 테이블을 포함해 affine 4비트, group 64 | 네이티브와 같은 plane. `gate_up_proj`는 네이티브 `w13`(interleaved)과 비트 단위로 같고, `global_scale`은 이미 MLX 네이티브 scale |
| mlx-vlm 변환본 | 그대로 통과 | 그대로 통과 |

전문가 형식은 텐서 shape에서 추론하고, hf 테이블이 있으면 그것과 대조한다. 로드 시점의 bf16→f16 변환은 Inkling과 마찬가지로 이 계열에서 건너뛴다. 네이티브 익스포트가 양자화 정보를 sidecar에만 적어 두기 때문이다. 비전 타워는 position, class, layer-scale 테이블을 `load_dense_param`으로 읽는다. 그래서 affine 양자화된 테이블은 로드할 때 역양자화되고, sidecar가 잘못되면 해당 텐서 이름을 담은 로드 오류가 난다.

### 2.5 도구 호출

Inkling은 모든 구간을 `<|message_model|>[recipient]<|content_KIND|>body<|end_message|>` 형태로 감싸며, 도구 호출은 함수를 수신자로 하는 `invoke_tool_json` 종류다. 생각(thinking)과 호출이 같은 `<|end_message|>`로 닫힌다. 그래서 `parse_tool_calls`는 thinking 제거 단계보다 먼저 원시 출력에서 Inkling 호출을 꺼낸다. 그러지 않으면 thinking 여는 표지가 뒤따르는 호출의 닫는 표지와 짝지어진다. 이 표지들은 Inkling 토크나이저의 특수 토큰이고 Inkling 1 템플릿도 같은 invoke 종류를 내보내므로, Harmony나 Kimi K3처럼 표지가 있으면 이 형식으로 판정한다.

스트림 필터는 `<|message_model|>` 뒤에 수신자 형태의 텍스트(256바이트 이하, `<|` 없음)가 오면 종류 표지가 도착할 때까지 붙잡아 둔다. invoke 표지가 오면 헤더 전체를 하나의 `EnterToolCall`로 처리한다. 첫 서버 실행에서는 수신자 `get_weather`가 `delta.content`로 새어 나왔다. 고정 `<|message_model|>` Strip 행이 헤더 시작 위치와 같은 자리에서 매칭되어, 헤더가 판정되기 전에 소비됐기 때문이다. 대기 중인 Inkling 헤더 위치나 그 뒤에 있는 구분자는 처리하지 않도록 고쳤다. 문자 단위 parity 테스트는 이 수정이 없으면 실패한다.

이슈 본문은 잘못된 payload를 본문에 남기라고 했지만, 이 PR은 저장소 규칙(#1743: 도구 마크업은 절대 본문이 아니다)을 따른다. 잘못된 invoke 메시지는 호출을 만들지 않고 보이는 텍스트에서도 빠진다. 스트림은 이미 이를 억제하고 있으며, parity 테스트는 스트리밍과 비스트리밍 본문이 같기를 요구한다.

## 3. 검증

모든 실행은 M3 Ultra(512 GB)에서 GPU lock으로 직렬화하여 mlx-vlm 466069f5와 대조했다. mlx-vlm 466069f5는 네이티브 레이아웃만 읽을 수 있어서, 4비트 기준값은 이름만 바꾸는 remap으로 익스포트를 로드해 얻었다. 이 remap은 텐서 이름을 바꾸고, mlx-vlm 자체의 `_split_gate_up`으로 `gate_up`을 나누고, `weight_scale_2 = global_scale / 2688`로 두고, 전문가별 모드를 선언한다.

| 항목 | 결과 |
|---|---|
| 네이티브 텍스트, `Explain why the sky is blue.`, greedy 128토큰, 기본 effort | mlx-vlm과 동일(128/128) |
| 네이티브 텍스트 trace, 4 x 256 위치 | top-1 차이 27/1024, 결정된 위치 0/490, 불일치 지점의 최대 gap 0.331, perplexity -0.21% |
| 4비트 텍스트 trace, 같은 코퍼스 | 30/1024, 결정된 위치 0/494, 최대 gap 0.185, perplexity -0.17% |
| 4비트 greedy 텍스트 | 약 45토큰까지 일치한 뒤, 결정되지 않은 위치에서 다른 단어 선택 |
| 네이티브 이미지(1280 x 720, 18 x 32 타일) | 프롬프트 id 600개가 mlx-vlm과 동일. teacher-forced 128토큰 이어쓰기 2/128, 결정된 위치 0/76. greedy는 처음 440자까지 일치 |
| 서버 `image_url` | CLI 응답과 바이트 단위로 동일 |
| 1패스 ablation(`MLXCEL_INKLING2_LOOP_PASSES=1`) | 두 체크포인트 모두 같은 문장을 반복하는 출력으로 무너짐 |
| 서버 `get_weather` 도구 호출 | 스트리밍과 비스트리밍 모두 `tool_calls` 항목 `{"city":"Warsaw"}` 반환, `delta.content`는 비어 있음. 도구 결과를 넣은 다음 턴은 "Warsaw is currently 4°C with light rain."으로 답함 |
| 디코드 | 네이티브 32.1 tok/s, 4비트 32.0 tok/s(mlx-vlm: 32.8, 38.5). 최대 메모리 17.6 GB, 15.5 GB |

단위 테스트 selector(`RUST_MIN_STACK=16777216`, `--profile test-fast --features metal,accelerate`, 직렬 실행):

| Selector | 통과 |
|---|---|
| `inkling2` | 26 |
| `server::tool_calls` | 506 |
| `inkling` | 129 |
| `coverage_tests` | 18 |
| `models::detection` | 97 |
| `models::registry` | 21 |
| `model_metadata`, `model_type`, `loaded_model`, `memory_estimate` | 8, 86, 12, 70 |

## 4. 한계와 후속 작업

- 배치 디코드는 행마다 별도의 단일 시퀀스 forward로 실행한다(`PerRowFallback`). padded prefill은 지원하지 않는다.
- short convolution과 grouped expert용 Metal 커널은 #3359에서 다룬다. 각각 함수 하나(`conv::short_convolution`, `moe::expert_projections`) 뒤에 있다.
- 포팅하지 않은 것: DSpark/MTP 드래프터(`mtp.safetensors`만 있는 디렉터리는 감지 단계에서 거부), 오디오, `is_varloop`.
- Small은 실제 체크포인트로 실행하지 않았다.
- 4비트 이미지 경로는 출력이 말이 되는지만 확인했다. mlx-vlm 이미지 trace 스크립트에 remap을 끼울 수 있는 지점이 없어서 teacher-forced trace는 돌리지 않았다.
- 4비트 디코드는 mlx-vlm보다 약 17% 느리다(32.0 대 38.5 tok/s). ops 경로가 upstream의 grouped 커널 없이 NVFP4/MXFP8 전문가 matvec을 실행하기 때문이며, #3359가 이 차이를 겨냥한다.
