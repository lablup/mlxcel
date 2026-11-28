# PR #2844: MiMo-V2.6 멀티모달 지원

**작성일**: 2026-11-03
**리뷰어**: -
**상태**: 부분 완료 (열린 PR, #2409를 닫음)
**위험도**: 높음

## 요약

PR #2844는 #2829가 #842를 위해 추가한 `mimo_v2_flash` 디코더 위에 `mimo_v2` 계열(Xiaomi MiMo-V2, V2.5, V2.6)을 얹는다. 디코더는 그대로 재사용하고, 새로 들어가는 것은 체크포인트 로더와 `mimovl` 비전 타워다. 로더는 벤더가 텐서 병렬로 합쳐 둔 QKV를 패딩된 샤드별 FP8 스케일 그리드까지 포함해 다시 풀고, MXFP4 라우티드 expert를 재포장하고, expert별 레이아웃을 Flash `switch_mlp` 평면으로 스택한다. 비전 타워는 공유 Qwen2.5-VL 이미지/비디오 런타임에 연결된다. 28개 파일, +12,869/-4. 추가된 줄 중 9,548줄은 커밋된 JSON 픽스처 두 개(V2.6 `config.json`과 parity 덤프)이고, 실제 Rust 코드와 테스트, 문서는 약 3,300줄이다.

검증은 합성 텐서와 업스트림 mlx-vlm에서 덤프한 작은 픽스처 하나에만 근거한다. 173.8 GB 실제 체크포인트가 검증 호스트에 없으므로 로더 산술은 실제 가중치로 확인된 적이 없고, CLI나 서버 생성도 실행되지 않았다. 또한 현재 PR은 `main`과의 시그니처 불일치로 CI가 빨강이며, 이는 5절에 적었다.

## 1. 문제 정의

이슈 #2409는 `mimo_v2` 옴니 계열을 포팅한다. 업스트림 `LanguageModel`은 `mimo_v2_flash` 것을 상속하고 `sanitize`만 재정의한다("The decoder stack is unchanged from MiMo-V2-Flash. V2.6 differs only in how the checkpoint stores its weights"). 따라서 포트는 두 번째 디코더를 만드는 대신 #842 디코더 위에 올라가야 한다. 그 디코더는 이미 hybrid sliding/full attention 스케줄, Q/K와 V/O의 서로 다른 head 폭, 레이어별 베이스를 쓰는 partial RoPE, `noaux_tc` 라우터, 256개 라우티드 expert를 다룬다.

체크포인트 레이아웃 세 가지가 직접 로드를 막는다.

- **평평한 config.** 텍스트 하이퍼파라미터가 `config.json` 최상위에 있고 `text_config` 객체가 없다. `vision_config.in_chans`는 `in_channels`의 배포 표기이고, `audio_config.speech_vocab_size`와 `speech_zeroemb_idx`는 정수가 아니라 JSON 문자열로 온다.
- **합쳐진 텐서 병렬 QKV.** `attention_projection_layout: fused_qkv`는 `q_proj`/`k_proj`/`v_proj`를 `self_attn.qkv_proj.weight` 하나로 대체한다. 이 가중치의 행은 TP 샤드들을 이어 붙인 것이고, 각 샤드는 자기 q, k, v 행을 차례로 담는다. FP8 `weight_scale_inv` 그리드는 샤드별이며 128행 블록 단위로 패딩되므로, 가중치 행 수만으로 그리드 행 수가 결정되지 않는다.
- **네이티브 MXFP4 expert.** 라우티드 expert는 E2M1 코드로 오고, 32개 값마다 E8M0 스케일 하나가 `<plane>.weight_scale`로 붙는다. MLX `gather_qmm`은 이를 그대로 받지 못한다.

비전 쪽은 재사용이 아니라 새 타워다. `mimovl`은 Qwen2.5-VL과 모양이 비슷하지만 GQA 폭, attention sink, 윈도 모양, 토큰 순서가 다르다. 오디오와 MTP 레이어, 번들된 drafter는 이슈 범위에서 명시적으로 제외된다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| `src/models/mimo_v2.rs`, `src/models/mimo_v2_weights.rs` | 계열 진입점과 리맵: `candidate_degrees`, `tensor_parallel_degree`, `unfuse_qkv`, `dequantize_fp8_weights`, `convert_mxfp4_experts`, `stack_experts`, `sanitize_text_weights`, `take_vision_weights` |
| `src/loading/vlm_mimo_v2.rs` | 평평한 config 빌더, 토큰 id 검증, audio config 파싱, 텍스트/비전 가중치 분리, 인코더와 프로세서 생성, `skip_vision` 처리 |
| `src/vision/encoders/mimovl.rs`, `mimovl_layers.rs` | Conv3d patch embed, GQA와 sink, banded window, column-major 스케줄을 쓰는 28블록 ViT, patch merger |
| `src/vision/mimo_v2_vl.rs` | `MimoV2VlModel`: `QwenVlRuntime` 구현, feature scatter, Flash 디코더로의 `LanguageModel` 포워딩, 오디오 거부 |
| `src/models/mimo_v2_flash.rs` | 래퍼가 미리 임베딩된 행으로 디코드할 수 있도록 `forward_prepared`, `input_embeddings`, `forward_prepared_embeddings_with_sequence_id` 추가 |
| 탐지, 레지스트리, 메타데이터, capability, TP와 메모리 테이블 | `ModelType::MiMoV2VLM`, `mimo_v2` 탐지 arm, `mimo_v2_vlm` 레지스트리 id, `TEXT_IMAGE_VIDEO` 모달리티, native video, VLM 종류, adapter 거부 |
| `src/server/model_worker.rs` | MiMo 전용 오디오 거부 메시지 |
| 테스트와 픽스처 | 로더/포워드/parity 테스트 991줄과 JSON 픽스처 두 개 |
| 문서 | README와 `docs/supported-models.md` 계열 항목 |

## 3. 구현 아키텍처

### 3.1 디코더 재사용

`MiMoV2VlModel`은 `MiMoV2FlashModel`을 소유하고 모든 `LanguageModel` 메서드를 위임한다. 캐시, 시퀀스 상태, 배칭 플래그, 스냅샷, 트리밍이 모두 그렇다. 디코더에 더해진 것은 진입점 세 개뿐이다. `input_embeddings`는 래퍼가 디코드 전에 비전 feature를 흩뿌릴 수 있도록 토큰 테이블을 노출한다. `forward_prepared`는 이미 임베딩된 행 위에서 디코더를 실행하며, 고정된 멀티토큰 prefill 경로다. `forward_prepared_embeddings_with_sequence_id`는 시퀀스 id 없이 새 멀티토큰 프롬프트가 오면 모델 소유 캐시를 초기화하고, 그 외에는 시퀀스별 상태로 라우팅한다. MiMo-V2 텍스트 위치는 1-D RoPE이므로 래퍼는 `mrope_only(None)` 스냅샷을 반환하고 미디어 상태를 지우지 않는다.

### 3.2 로더 파이프라인

순서가 중요하며 `sanitize_text_weights`에 그대로 들어가 있다.

1. `audio_encoder.*`와 `speech_embeddings.*`를 버리고, 비전 타워는 별도 맵으로 분리하면서 네이티브 `visual.` 접두사를 `vision_tower.`로 정규화한다. `language_model.`을 벗기고 `.mtp.`와 `model.mtp` 키를 버린다.
2. `weight_scale`로 끝나는 키가 있으면 MXFP4 재포장을 먼저 한다. 이 사이드카들이 `_scale_inv`를 찾기 전에 `.scales`가 되어야 하기 때문이다.
3. 살아남은 `self_attn.qkv_proj.weight`가 있으면 unfuse한다. 합쳐진 텐서의 그리드를 샤드별로 소비하므로 일반 FP8 패스보다 먼저 와야 한다.
4. 남은 `_scale_inv` 투영을 모두 BF16으로 dequantize한다.
5. 레이어마다 `mlp.experts.{e}.{gate,up,down}_proj.{weight,scales,biases}`를 `mlp.switch_mlp.*`로 스택하고, 남은 `mlp.experts.*` 키를 버린다.

일반 FP8 패스와 expert 재포장은 이미 unfuse, dequantize, stack이 끝난 mlx-vlm 변환본에서는 no-op이다.

### 3.3 합쳐진 QKV unfuse

`candidate_degrees`는 `num_key_value_heads`까지 2의 거듭제곱을 열거하며, `degree * (heads/degree * head_dim + kv_heads/degree * (head_dim + v_head_dim))`가 가중치 행 수와 같고 `degree * ceil(shard/128)`이 그리드 행 수와 같은 경우를 남긴다. `tensor_parallel_degree`는 모든 합쳐진 텐서에 대해 후보 집합의 교집합을 구하고 업스트림처럼 `max(common)`을 택한다. 공개된 full attention 레이어의 가중치는 13,568행이다. degree 4 샤드는 3,392행이고 이는 27블록으로 패딩되므로 그리드는 108행이 된다. 이 108은 degree 1이나 2로 설명되지 않으며, 이것이 degree 4를 고정한다. 106행 그리드만으로는 1과 2가 모두 가능하다. 로더는 rank별로 가중치 행과 대응 그리드 행을 슬라이스하고, 공유 LUT로 그 샤드의 block FP8을 복원하고, 행 수에 따라 q/k/v를 자른 뒤 rank 방향으로 이어 붙여 `q_proj`/`k_proj`/`v_proj`를 쓴다.

### 3.4 MXFP4 expert

`convert_mxfp4_experts`는 모든 `<plane>.weight_scale`을 별도 테이블로 옮기고, 패킹된 `<plane>.weight`를 `uint32`로 본 뒤(워드당 E2M1 코드 4개), 사이드카를 `<plane>.scales`로 다시 내보낸다. Flash 로더는 bias 없는 `uint8` 스케일에서 `mode: "mxfp4"`, `group_size: 32`, `bits: 4`를 추론한다. 네이티브 배포본만 이 경로를 타며, 변환본은 이미 `scales`를 갖고 있다.

### 3.5 비전 타워

타워는 hidden 1280, 28블록, 쿼리 head 32개, KV head 8개다. 다만 `qk_channels`가 `1280 / 32 = 40`이 아니라 64다. 그래서 합쳐진 QKV 출력은 3,072폭(`(32 + 2*8) * 64`), attention 출력은 2,048폭(`32 * 64`)이고 이후 1280으로 투영된다. 포트 고유 동작은 네 가지다.

- **Sink.** 윈도 블록에서 `use_sink`는 학습된 head별 로짓을 key 위치 0의 attention 로짓에 더한다. 분모에만 더하는 표준 sink가 아니라 마스크 항으로 더하므로, softmax는 업스트림과 정확히 같은 값을 본다.
- **Banded window.** 각 프레임의 `cu_seqlens` 세그먼트 안에서 `|i - j| > visual_token_window_size`(64)인 위치를 마스킹한다. `fullatt_block_indexes` `[0, 9, 18, 27]`의 full 블록은 마스킹하지 않는다.
- **Column-major 스케줄.** `vit_window_attn_types`는 블록별 목록이다(`1`은 column rotary를 쓰는 column-major, `0`은 row-major, `-1`은 full attention). 인코더는 `1` 구간에 들어갈 때 spatial-merge 블록 단위로 순열을 적용하고, 구간을 나갈 때 역순열로 되돌린다. 공개 config는 `[-1, 0, 0, 0, 0, 1, 1, 1, 1]` 구간을 반복한다.
- **Patch embed와 merger.** Conv3d patch 임베딩은 2-D `[out, C*T*P*P]` 또는 두 가지 5-D 레이아웃으로 들어오며, 공유 Qwen2.5-VL 프로세서가 내보내는 temporal-major `T, C, P, P` 행 순서로 한 번 재배열된다. merger는 weight만 있는 LayerNorm과 선형 두 개, GELU이며 Qwen의 RMSNorm merger가 아니다.

### 3.6 미디어 경로

`MimoV2VlModel`은 `QwenVlRuntime`을 구현하므로 CLI의 `--image`/`--video`, 서버의 native video 경로, 프롬프트 확장, feature scatter가 모두 공유 Qwen2.5-VL 코드를 탄다. `LoadedModel::MiMoV2VLM`은 capability 테이블에 `VlmRuntimeRef::Qwen`으로, `qwen_video_runtime`에, `model_type_has_native_video`에 추가된다. `get_input_embeddings`는 이미지/비디오 placeholder 수를 세고, 타워를 실행하고, feature/placeholder 불일치를 거부하고, hidden 폭이 텍스트 임베딩 폭과 맞는지 확인한 뒤 하나씩 병합한다. `skip_vision: true`이거나 타워가 없는 체크포인트는 프롬프트 정보용 프로세서는 유지하는 텍스트 전용 래퍼가 된다.

## 4. 기술적 선택과 그 이유

### 4.1 #842 디코더를 포크하지 않고 재사용

업스트림 관계가 sanitizer만 재정의한 서브클래스이므로 포트도 그 모양을 유지한다. 디코더는 래퍼가 필요로 하는 미리 임베딩된 진입점 세 개 외에는 건드리지 않는다. 별도 `mimo_v2` 디코더를 만들었다면 hybrid 캐시, 라우터, partial RoPE 로직 1,500줄을 복제하고 두 구현이 어긋날 수 있게 했을 것이다. 대가는 디코더의 가정(모델 소유 이종 캐시, 배칭 정책)을 그대로 물려받는다는 점인데, 가중치가 같은 계열이므로 여기서는 맞다.

### 4.2 비공개 미디어 경로 대신 `QwenVlRuntime` 구현

MiMo-V2는 Qwen2.5-VL 프로세서를 쓴다. `<|vision_start|><|image_pad|><|vision_end|>` 확장과 `grid_thw`가 같다. 이 경로를 MiMo 전용 모듈로 복사했다면 프롬프트 확장, placeholder 계산, 비디오 프레임 경로가 CLI, 서버, 테스트 세 곳에 중복됐을 것이다. 트레이트 구현은 구현을 하나로 유지하고 서버 비디오 연결을 capability 한 줄로 만든다. 대가는 향후 MiMo 전용 미디어 동작이 공유 트레이트를 통해 표현되지 않으면 두 진입점에 도달하지 못한다는 점이다.

### 4.3 패딩된 그리드에서 TP degree를 복원

degree 4를 가정하면 공개 체크포인트와는 맞지만 다른 degree로 변환된 체크포인트를 조용히 잘못 슬라이스하게 된다. 모양에서 degree를 읽고 모든 합쳐진 텐서에 대해 교집합을 구하면, 불일치 체크포인트가 그럴듯하지만 틀린 모델이 아니라 로드 에러가 된다. full attention 레이어에서 degree 4와 2를 구분하는 유일한 단서가 패딩된 그리드이므로, docstring 예제를 주석이 아니라 테스트로 만들었다.

### 4.4 리맵에서 가중치를 복사하지 않고 이동

네이티브 체크포인트는 173.8 GB다. 모든 리맵 함수가 `WeightMap`을 값으로 받고 반환하며 처리한 항목을 제거하므로, 건드리지 않는 평면은 이동되고 대부분 패스가 no-op인 변환본에서는 복사 비용이 아예 없다. API가 제자리 변경 함수 집합이 아니라 소유권 변환 체인으로 읽히는 이유다.

### 4.5 오디오를 명시적으로 거부

체크포인트는 20채널 코덱 토크나이저와 작은 `AudioEncoder`를 함께 배포하지만 mlxcel에는 둘 다 없다. 로더는 해당 키 계열을 버리고, 서버는 빠진 포트를 이름으로 밝히는 메시지를 반환한다. 반쪽 모델을 로드하거나 클립을 조용히 무시하지 않는다. 파서는 문자열 타입 스칼라 두 개를 포함해 `audio_config`를 계속 읽으므로, 나중에 오디오를 포팅할 때 문자열 타입 config를 물려받지 않는다.

### 4.6 토큰 id를 어휘 크기와 대조

이미지, 비디오, 오디오, 비전 경계 토큰 id는 설정 가능하며 기본값은 공개 값이다. 빌더는 `vocab_size`를 벗어난 id를 거부한다. 범위를 벗어난 비전 토큰 id는 유창한 텍스트를 계속 내면서 모든 이미지 feature를 잘못된 행에 놓기 때문이다. 또한 `text_config`가 있으면 그것을 쓰고 없으면 최상위 평평한 루트를 텍스트 config로 취급하며, 이는 업스트림 `ModelConfig.from_dict`와 같다.

## 5. 검증

PR이 보고한 타깃 실행은 다음과 같다.

| 명령 | 보고된 근거 |
|---|---|
| `cargo test --profile test-fast -p mlxcel --lib loading::vlm::mimo_v2::` | 13개 테스트: config 빌더, 탐지, 패딩 그리드를 포함한 두 unfuse degree, MXFP4 리맵, 네이티브 레이아웃 dequant/stack, 작은 end-to-end 이미지 포워드 |
| `cargo test --profile test-fast -p mlxcel --lib vision::encoders::mimovl::` | 작은 타워에서 업스트림 mlx-vlm이 덤프한 텐서와의 fixture parity |
| `models::mimo_v2_flash::` | 18개 테스트 |
| `models::detection_tests::` | 85개 테스트 |
| `models::registry::` | 20개 테스트 |
| `models::metadata_tests::` | 4개 테스트 |
| `model_metadata_tests::` | 7개 테스트 |
| `models::vlm_text_only_detection_tests::` | 9개 테스트 |
| `multimodal::qwen_vl::` | 10개 테스트 |
| `cargo clippy --lib --tests -- -D warnings` | 통과 |
| `cargo fmt --check` | 통과 |

코드에서 읽은, 테스트가 실제로 고정하는 내용은 다음과 같다.

- 로더 테스트 13개는 V2.6 픽스처를 평평한 최상위에서 빌드하고 hidden 4096, 48레이어, head 64/4, head 폭 192/128, sliding KV head 8, expert 256 top-8, RoPE 베이스 두 개, `in_chans`, `qk_channels` 64, `fullatt_block_indexes` `[0, 9, 18, 27]`, 윈도 64, 강제 변환된 `speech_vocab_size` 1280과 `speech_zeroemb_idx` 1024, 토큰 id 일곱 개, `skip_vision` false를 확인한다.
- 탐지는 실제 V2.6 config를 `MiMoV2VLM`으로 라우팅하고, 등록 단언은 `is_vlm_model_type`, native video, 레지스트리 id `mimo_v2_vlm`, family `Xiaomi`, `ALL_MODEL_TYPES` 포함을 다룬다.
- Unfuse 테스트는 상수 FP8 평면(q = 1.0, k = 2.0, v = 4.0)을 만들어 dequantize 후 복원된 모양 `[64*192, 32]`, `[4*192, 32]`, `[4*128, 32]`과 값을 확인한다. 그리드 없는 degree 1과 108행 패딩 그리드를 쓰는 degree 4를 모두 다룬다. 107행 그리드는 "no tensor-parallel degree"로 에러가 난다.
- 네이티브 레이아웃 테스트는 레이어 0 dense FP8 투영, 레이어 1 V2.6 MXFP4 expert, 레이어 2 V2.5 FP8 expert를 다루고, 결과 `switch_mlp`의 dtype(`uint32`와 BF16), 모양, 스케일, 그리고 `.experts.` 키가 하나도 남지 않음을 확인한다.
- MXFP4 테스트는 `[4, 8, 16]` uint8 평면이 `[4, 8, 4]` uint32와 `.scales` 사이드카가 되고 `weight_scale` 키가 사라짐을 확인한다.
- 비전 픽스처는 3블록 작은 타워(full 블록 0, 윈도 블록 1과 2, 1과 2에 sink, `vit_window_attn_types` `[-1, 1, 0]`)에 그리드 `(1, 4, 4)`, 병합 출력 `[4, 6]`이고 `2e-4 * max(|expected|, 1)`로 비교한다. 테스트는 실패 모드를 이름으로 적는다. sink를 빼거나 banded 마스크를 무시하면 기대 텐서가 바뀐다.
- 포워드 테스트는 Flash 디코더 위의 4레이어 랜덤 `mimo_v2`와 작은 타워를 만들어 6토큰 프롬프트에 이미지 feature 4개를 흩뿌리고, 병합된 행이 타워 출력과 같은지, 래퍼 로짓이 직접 `forward_prepared_embeddings_with_sequence_id`를 실행한 결과와 같은지 확인한다. placeholder/feature 불일치는 "placeholders"로 패닉한다.

### 현재 게이트 상태

커밋된 트리는 현재 `main`과 컴파일되지 않는다. `630aaeba`(#2842, 이 브랜치의 부모)가 `reconstruct_block_fp8_tensor`를 6인자로 넓히며 `block_cols`를 추가했지만, `src/models/mimo_v2_weights.rs`는 156행과 293행에서 여전히 5인자로 호출한다. CI 실행 `41651251817`이 정확히 이 지점에서 실패한다. `cargo-clippy`, `OpenXLA feature compile`, `WebUI installed artifact`가 모두 E0061("this function takes 6 arguments but 5 arguments were supplied")로 떨어진다. `cargo-fmt`와 `CPU-only link`는 통과하고, PR은 draft에 `mergeStateStatus: UNSTABLE`이다. 위의 로컬 clippy/fmt 결과는 PR이 보고한 값이며 현재 트리를 설명하지 않는다.

## 6. 미검증 범위

- **실제 체크포인트.** `nativ-community/MiMo-V2.6-Flash-RL-MLX-4bit`는 173.8 GB이고 검증 호스트에 없어서 CLI/서버 일관성 실행이 이뤄지지 않았다. 실제 가중치로 텍스트, 이미지, 비디오 프롬프트를 생성한 적이 없고 mlx-vlm 대비 teacher-forced 로짓 트레이스도 없다. 로더 산술 경로(degree 복원, 샤드별 FP8 복원, MXFP4 스케일, expert stacking)는 모두 합성 검증뿐이다. 네이티브 `XiaomiMiMo/MiMo-V2.6-Flash-RL`(177.7 GB, FP8 + MXFP4)도 내려받지 않았으므로 "두 레이아웃 모두 로드" 주장은 실물 규모에서 검증되지 않았다.
- **오디오.** 명시적으로 범위 밖이다. 20채널 코덱 토크나이저는 mlxcel 포트가 없고, 서버는 MiMo 전용 에러를 반환하며 로더는 가중치를 버린다.
- **MTP 레이어와 번들 drafter.** `num_nextn_predict_layers`는 3이고 체크포인트는 `dflash/`를 포함한다. 둘 다 로드 시 버려지며 실행되지 않는다.
- **CUDA와 ROCm.** 실행 없음. 이 호스트는 Apple Silicon이다.
- **텐서/파이프라인 병렬.** `ModelType::MiMoV2VLM`을 TP fallback 아키텍처 테이블에 넣어 디스패치를 total로 만들었지만 멀티디바이스 실행은 없다.
- **배칭 디코드.** 래퍼는 Flash 디코더의 배칭 플래그를 그대로 전달하며, 이 계열로 동시 서버 요청을 돌린 적이 없다.

## 7. 학습 포인트

패딩된 양자화 그리드는 노이즈가 아니라 신호다. 106블록 가중치에 108행 그리드가 붙는다는 사실이 degree 4를 1, 2와 구분하게 한다. 여분 행을 잘라낼 패딩으로 취급했다면 로더는 degree 2 해석을 받아들여 가중치를 잘못 슬라이스했을 것이다. 이슈 본문을 신뢰하는 대신 체크포인트에서 degree를 복원하고 모든 합쳐진 텐서에 대해 교집합을 구하는 편이 더 싸다.

체크포인트 리맵에서 패스 순서는 계약의 일부다. MXFP4 사이드카는 FP8 패스가 `_scale_inv`를 찾기 전에 이름이 바뀌어야 하고, 합쳐진 QKV는 그리드를 샤드별로 소비하므로 일반 FP8 패스보다 먼저 unfuse되어야 한다. 순서를 로더가 임의 순서로 호출하는 독립 헬퍼가 아니라 주석이 달린 한 함수에 담는 것이, 네이티브본과 변환본 모두에서 필요한 곳이 no-op이 되게 만든다.

멀티모달 래퍼에는 미리 임베딩된 디코드 진입점이 필요하다. 디코더는 원래 토큰 id만 받았으므로 feature를 흩뿌린 래퍼가 재임베딩 없이 돌아올 길이 없었다. `forward_prepared`를 추가하고 토큰 테이블을 노출하는 작은 변경이 미디어 경로와 텍스트 경로가 디코더 하나를 공유하게 한다.

## 8. 위험

높음. 완전히 새로운 비전 타워와 다단계 가중치 리맵을 가진 173.8 GB 계열에 실제 가중치 검증이 없으므로, 로더 오류는 크래시가 아니라 그럴듯하지만 틀린 로짓으로 나타난다. 합성 테스트는 모양, dtype, 값 상수에 강하지만 실제 FP8 그리드 해석 오류, E8M0 스케일 규약 오류, expert 순서 불일치는 잡지 못한다. 또한 브랜치가 현재 `main`과 빌드되지 않으므로 보고된 로컬 게이트 결과는 CI가 보는 트리 이전의 것이다. 디코더 자체는 Flash 체크포인트로 검증된 #2829 포트이지만, V2.6 체크포인트의 텍스트 가중치는 이 경로를 지난 적이 없다.
