# PR #2438: 양자화된 linear base에서 runtime LoRA term 너비 고정

**작성일**: 2026-10-17
**상태**: CUDA(GB10)에서 실제 체크포인트(qwen3-0.6b-4bit)로 구현 및 검증 완료. dequantize된 base로의 fuse는 추가하지 않음 (「미해결 사항」 참고)
**위험도**: 낮음에서 중간 (linear claim 시점에 너비 검사를 추가하고, 양자화된 linear base에 대한 fused 경로의 로드 시점 거부를 새로 추가함)

## 요약

#2437을 닫는다. 양자화된 linear base에 대한 runtime LoRA pair는 이제 레이어가 claim할 때 레이어의 실제 입력 너비와 대조하며, fusion은 양자화된 linear base를 `.scales` plane으로 식별해 거부한다. 둘 다 #2434가 embedding 테이블용으로 추가한 메커니즘을 재사용한다.

## 문제 정의

- Runtime 경로(`mlxcel-server --lora`, `mlxcel serve --lora`): `src/lora/runtime.rs`의 `validate_pair_shapes`는 양자화된 base를 scales group 개수로만 검사하므로 `in / groups`가 {16, 32, 64, 128}인 너비는 모두 통과한다. qwen3-0.6b-4bit에서 너비 512인 `q_proj`의 `lora_a`는 로드된 뒤 warmup에서 `[matmul] Last dimension of first input with shape (1,1,1024) must match ... (512,8)` 오류로 프로세스를 중단시킨다 (SIGABRT, exit 134).
- Fused 경로(`mlxcel generate --adapter`, `--lora-fuse`, 파이프라인 stage 로더): shape guard가 delta를 packed plane과 비교한다. 너비가 `in * bits / 32`인 delta는 일치하여 packed word에 더해지고, plane은 f32가 되며 첫 forward가 `[quantized_matmul] The weight matrix should be uint32 but received float32`로 중단된다.

#2434는 같은 두 메커니즘으로 embedding 경우를 이미 닫았다. `UnifiedEmbedding::from_weights`가 claim 시점에 너비를 고정하고, fusion은 양자화된 embedding 테이블을 `.scales` plane으로 거부한다.

## 기술적 선택과 그 이유

- claim 전에 검사한다. `claim_linear(prefix, in, out)`는 prefix에 staging된 모든 term을 `a_t == [in, r]`, `b_t == [r, out]`에 대해 검사한 뒤에야 claim한다. `EmbeddingLoraTerms::new`를 본뜬 것이다. prefix에 staging된 embedding term의 tied-head 형태도 같은 방식으로 검사한다. 검사가 staging에서 term을 꺼내기 전에 실행되므로, 생성자 오류를 "레이어 없음"으로 바꾸는 호출자(Nemotron-H는 선택적 projection 네 개를 `UnifiedLinear::from_weights(..).ok()`로 만든다)에서도 레이어 없이 모델을 서비스하지 않고 unclaimed로 로드가 실패한다.
- 레이아웃별 실제 너비. packed weight의 `in = scales_cols * group_size`이며, `UnifiedLinear::from_weights_with_mode`에서는 reconcile된 group size를, `FusedQKVLinear`에서는 이어 붙인 plane이 가진 group size를 쓴다. `out`은 weight의 행 수다. `FusedQKVLinear`는 각 projection을 자기 plane에 대해 검사한다.
- dense base도 포함한다. `Linear::from_weights`는 dense 열 수를 검사하며 로더가 이미 정확히 강제하고 있으므로 dense base에는 변화가 없고, embedding prefix로 만든 tied head도 함께 다룬다. staging된 것이 없으면 너비 계산은 건너뛴다. 어떤 struct에도 필드를 추가하지 않는다.
- fused 경로에서 사전에 거부한다. #2434가 embedding 테이블에 추가한 `.scales` 거부를 Conv1D 판정 옆의 단계(`reject_quantized_base_fusion`)로 옮기고 linear weight까지 적용한다. 따라서 거부된 adapter는 base weight map을 건드리지 않는다. 또한 양자화 부속 plane 자체로 해석되는 pair도 거부한다(`q_proj.scales.lora_a`는 `q_proj.scales`로 해석되며, 그 delta가 scales plane과 일치해 scales에 더해졌다). 메시지는 레이어 이름을 밝히고 `--lora`를 안내한다.

## 변경 사항

- `src/lib/mlxcel-core/src/runtime_lora.rs`: `claim_linear`. staging된 term과 embedding term의 tied-head 형태에 대한 너비 검사를 포함한다.
- `src/lib/mlxcel-core/src/layers.rs`: 모든 linear claim이 `claim_linear`를 거친다(`Linear::from_weights`, `UnifiedLinear::from_weights_with_mode`의 양자화 분기, `FusedQKVLinear`의 projection별 claim).
- `src/lora/loader.rs`: 사전에 실행되는 `reject_quantized_base_fusion`.
- `src/lora/runtime.rs`: 4줄 변경(5줄 수정).
- `src/lora/loader_tests.rs`: 사전 거부에 대한 로더 테스트.
- 문서: `docs/server-features.md`에 양자화된 체크포인트에서는 unfused 경로만 adapter를 적용하며 로드 시 각 pair의 너비를 고정한다고 명시했다.

## fused 경로의 동작 변경

양자화된 base에 대한 유효한 adapter는 fused 경로에서 원래 동작하지 않았다. fusion 전에 dequantize 단계가 없고(`load_model_with_adapter`는 원본 safetensors map에 fuse한다), `[out, in]` delta는 `[out, in * bits / 32]` plane에 대한 shape guard에서 실패한다. qwen3-0.6b-4bit에서 `mlx_lm.lora`로 학습한 q_proj adapter를 쓰면 main 8744c87f는 `LoRA shape mismatch fusing adapter layer model.layers.26.self_attn.q_proj ...: base is [2048, 128], delta is [2048, 1024]`를 출력한다. 따라서 이 거부로 깨지는 정상 사례는 없다. 이 메시지를 이유와 대안을 알려주는 메시지로 바꾸고, packed 너비로 인한 손상을 막는다.

## 검증

GB10, `--features cuda`, test-fast 프로파일, 모든 GPU 실행은 `gpu-lock` 아래, `MLXCEL_SDPA_DETERMINISTIC=1`.

실제 체크포인트 `qwen3-0.6b-4bit`(affine 4-bit, group 64). adapter는 `mlx_lm.lora` adapter(rank 8, scale 20, 레이어 26과 27의 q_proj)에서 만들었다. `valid_qproj`(두 pair 모두 학습된 그대로), `bad_w512`(레이어 27의 `lora_a`를 `[512, 8]`로 잘라 runtime 로더의 group 개수 검사를 통과), `packed_w128`(레이어 27만, `lora_a` `[128, 8]`, packed plane과 같은 너비). "이전"은 main 8744c87f 바이너리, "이후"는 이 브랜치다. 서버 요청은 `/v1/completions`, 프롬프트 "The capital of France is", 16 토큰, temperature 0.

| 명령 | Adapter | 이전 (main) | 이후 |
|---|---|---|---|
| `mlxcel-server --lora` | `bad_w512` | warmup에서 abort, exit 134 | `model.layers.27.self_attn.q_proj`를 지목하는 로드 오류("takes 1024 input features and produces 2048"), abort 없음 |
| `mlxcel serve --lora` | `bad_w512` | warmup에서 abort, exit 134 | 같은 로드 오류, abort 없음 |
| `mlxcel generate --adapter` | `packed_w128` | packed plane에 fuse되어 첫 forward에서 abort, exit 134 | 거부: `model.layers.27.self_attn.q_proj.weight is a quantized linear weight`, exit 1 |
| `mlxcel generate --adapter` | `valid_qproj` | `LoRA shape mismatch` (base [2048, 128], delta [2048, 1024]), exit 1 | 양자화 linear 메시지로 거부, exit 1 |
| `mlxcel generate --adapter` | `bad_w512` | `LoRA shape mismatch`, exit 1 | 양자화 linear 메시지로 거부, exit 1 |
| `mlxcel-server --lora` | `packed_w128` | 로더 오류(group 개수) | 같은 로더 오류 |
| `mlxcel-server --lora` | `valid_qproj` | " Paris. 1. If the capital is capital, then the capital is capital" | 동일한 텍스트 |
| `mlxcel-server --lora` | #2434의 embed_tokens + q_proj adapter | " Lyon. The capital of Japan is Osaka." | 동일한 텍스트 |

로드 오류 후에도 서버는 계속 떠 있고 503으로 응답한다. 이는 모델 로드 실패 시의 기존 동작이다.

최종 커밋에서 실제 체크포인트 게이트 `real_checkpoint_embedding_adapter_parity`(ignored 테스트): #2434의 adapter를 쓴 Llama-3.2-1B-Instruct-bf16은 runtime vs fused 0.125, stage vs fused 0.0000, scale 0은 base와 바이트 단위로 동일하며, #2434가 보고한 수치와 같다. qwen3-0.6b-4bit runtime 경로: scale 0은 base와 바이트 단위로 동일하다.

테스트:

- `-p mlxcel-core runtime_lora`: 11개 통과, 신규 3개. `a_term_of_the_wrong_width_on_a_quantized_linear_fails_the_claim`은 4-bit group-64 `[8, 256]` `UnifiedLinear`에 너비 64, 128, 512(모두 로더가 수용)를 staging하고, 거부된 term이 staging에 남은 채 `Err`가 나는지와 잘못된 출력 너비를 확인한 뒤, 올바른 너비를 로드해 forward 값을 검사한다. `every_quantized_linear_layout_pins_the_term_width`는 affine의 bits 2, 3, 4, 5, 6, 8과 group size 32, 64, 128, mxfp4, mxfp8, nvfp4(각 block-float 모드는 백엔드가 네이티브로 실행하는 경우에만), 그리고 reconciler가 선언된 group size를 덮어쓰는 두 레이아웃(선언 64에 32로 저장된 mxfp4, 64로 저장된 4-bit group-16 fallback)을 다룬다. `fused_qkv_pins_each_projection_term_to_its_own_plane`는 fused q/k/v 경로를 plane별로 검사하고 올바른 너비의 forward를 확인한다.
- `lora::`: 69개 통과, 신규 3개: `the_fusion_loop_refuses_a_packed_width_delta_on_a_quantized_linear`(실제 4-bit packed plane과 같은 너비의 delta), `a_quantized_target_is_refused_before_any_dense_pair_is_written`, `a_pair_aimed_at_a_quantization_side_plane_is_refused`. `loading::adapter_`: 9개 통과, 1개 ignored. `llama::adapter_tests`: 4개 통과. `partial_loading_adapter`: 16개 통과.
- 되돌림 확인: `claim_linear`의 검사와 `reject_quantized_base_fusion`을 비활성화하면 신규 core 테스트 3개와 신규 로더 테스트 3개가 실패한다(같은 거부를 공유하는 기존 #2434 embedding 거부 테스트도 실패). 복원하면 통과한다.
- `mlxcel`과 `mlxcel-core` 패키지에 대한 `cargo clippy --profile test-fast --features cuda --lib --tests -- -D warnings`, `cargo fmt --all -- --check`, 계약 스크립트(`check_llama_compat_manifest`, `check_kernel_dtype_keys`, `check_kernel_port_dispatch`, `check_crate_versions`, `insert_apache_header --check`)가 통과한다.

## 리뷰 중 변경

- 구현 리뷰에서 claim 후 검사하면 term이 소비되어, Nemotron-H의 `.ok()` 생성자가 너비가 잘못된 term을 조용히 사라진 projection으로 바꿀 수 있음이 발견되었다. 이제 검사가 claim보다 먼저 실행된다.
- 보안 리뷰에서 `q_proj.scales`를 겨냥한 pair가 scales plane에 fuse됨이 발견되었다. 사전 거부가 양자화 부속 plane까지 포함하도록 확장되었고, 첫 쓰기 이전으로 옮겨졌다.

## 미해결 사항

- dequantize된 base로의 fuse는 추가하지 않았다.
- 정수형 `lora_a` / `lora_b` 텐서는 fused 경로에서 형 변환 없이 `matmul`에 도달하며, 비부동소수 타입에 대해 MLX가 예외를 던진다. 이 PR 이전부터 있던 문제이고 dense base에도 해당하며 별도 수정 대상이다.
- Nemotron-H의 선택적 projection은 모든 생성자 오류를 "레이어 없음"으로 취급한다. 이 PR로 잘못된 adapter는 여전히 로드가 실패하지만(unclaimed로), 메시지는 너비 불일치가 아니라 레이어가 unclaimed라고 알린다.

## 통계

6개 파일 변경, 520줄 추가, 23줄 삭제. 커밋: `9ad38d49` fix(lora): pin runtime LoRA term widths on quantized linear bases, `94071d0d` fix(lora): check staged terms before claiming and refuse packed fusion up front.
