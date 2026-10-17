# PR #2441: mlx-lm NVFP4 per-expert global_scale 네이티브 로딩

**작성일**: 2026-10-17
**상태**: CUDA(GB10)에서 실제 체크포인트로 구현 및 검증 완료. Metal은 Metal nightly에서 검증하며, 「미해결 사항」의 항목은 남아 있음
**위험도**: 중간 (`gather_qmm` FFI 시그니처, `SwitchLinear`의 scale 처리, Laguna sidecar 키를 변경하고, 모든 text 계열에 대해 ModelOpt NVFP4 repack을 활성화함)

## 요약

#2351을 닫는다. mlx-lm의 `quantize_model(..., mode="nvfp4", global_scale=True)`(ml-explore/mlx-lm#1844)는 expert별 `max|w[e]|`를 `{switch layer}.global_scale`로 기록하고 해당 layer에 `"global_scale": true` 플래그를 붙인다. MLX의 `gather_qmm`은 `global_scale / (448 * 6)`으로 dequantize한다. mlxcel은 같은 키를 compressed-tensors 출력 승수로 읽었기 때문에, 이런 export는 routed expert 출력이 모두 2688배 크게 디코딩되었다. 이 PR은 두 scale을 분리하고, 네이티브 scale을 `gather_qmm`에 전달하며, ModelOpt NVFP4 expert 로딩을 Gemma 4 밖으로 확장한다.

## 문제 정의

- `SwitchLinear::from_weights_with_mode`는 모든 `{prefix}.global_scale`을 프로세스 내부 출력 승수(`1 / weight_global_scale`, `gather_qmm` 이후 적용)로 받아들였다. rank-1 shape 검사도 mlx-lm 텐서를 통과시켜서 로드는 성공하고 출력만 조용히 틀렸다. 이 키는 Laguna transcode가 프로세스 내부에서만 생성했다.
- FFI는 네이티브 scale을 전달할 수 없었다. `gather_qmm`에 `global_scale` 파라미터가 없었고 C++ shim은 `std::nullopt`를 넘겼다. MLX는 f32 dtype, `w.shape[:-2]` shape, `nvfp4` 모드, indices 존재, GPU stream을 요구한다.
- ModelOpt `weight_scale_2` 처리는 Gemma 4에서만 도달했다. DeepSeek-V3, Mistral 4, PhiMoE의 expert 스태킹 루프는 `weight`, `scales`, `biases`만 순회하며 scale sidecar를 버렸다.
- mlx-lm nvfp4 MoE export가 로드되지 못하게 하는 문제가 두 가지 더 있었다. `SwitchLinear::from_weights`는 bias가 없는 block-float plane을 affine으로 선언하고 거부했다. 또한 bridge의 양자화 embedding lookup이 block-float 테이블에 uint8 출력 dtype을 요청해 첫 lookup에서 abort되었다(`[dequantize] Only real floating types are supported but output dtype == uint8`).

## 기술적 선택과 그 이유

- 디스크상의 키와 프로세스 내부 키가 더 이상 이름을 공유하지 않는다. Laguna sidecar는 `inv_global_scale`로 이름을 바꾼다(`laguna_sanitize.rs`, `laguna.rs` 감지, 테스트). Laguna의 dense shared-expert `[1]` `.global_scale`은 `UnifiedLinear`로 들어가고 dense global scale은 범위 밖이므로 이름을 유지한다.
- `SwitchLinear::Quantized`는 두 필드를 가진다. `inv_global_scale`(matmul 이후 승수)과 `native_global_scale`(`gather_qmm`에 전달). 둘 중 하나라도 있으면 `quantized_parts`가 `None`을 반환하여 fused 경로는 `forward`로 폴백한다. `SwitchGLU::forward_with_expert_scales`는 `forward_indexed`를 거치므로 네이티브 scale도 받는다.
- 플래그는 키가 아니라 config에서 읽는다. `apply_native_global_scale_flags`가 `quantization` / `quantization_config` 항목(최상위 및 `text_config`)을 읽어, 플래그가 있는 `{path}.global_scale`을 `{path}.native_global_scale`로 옮긴다. `load_text_weights`와, `sanitize_gemma4_nvfp4_weights`를 통해 모든 Gemma 4 로더가 이를 호출한다. 약 55개 계열 파일이 `SwitchLinear` / `SwitchGLU`를 생성하므로 `from_weights_with_mode`에 파라미터를 추가하는 대신 이 방식을 택했다. 판단은 여전히 layer별 config 항목에서, 계열 sanitizer가 키 이름을 바꾸기 전에 한 번만 내려진다.
- 로드 시 실패(모두 prefix를 명시): 텐서 없는 플래그, weight가 `[num_experts, out, in]` stack이 아닌 layer의 플래그, 플래그가 없는데 switch prefix에 남은 `.global_scale`, 체크포인트가 `native_global_scale`을 직접 가진 경우. 네이티브 scale은 MLX의 전제 조건(nvfp4 모드, expert당 f32 항목 하나, Metal 또는 CUDA GPU stream)에 대해서도 검증하므로, 위반은 생성 도중 예외가 아니라 prefix와 backend를 명시한 로드 오류가 된다.
- `SwitchLinear::from_weights`는 shape으로부터 uint8 scale을 가진 bias 없는 block-float plane(nvfp4 / mxfp4 / mxfp8, group당 word 수 기준)을 추론한다.
- ModelOpt 감지는 Gemma 4의 `is_modelopt_nvfp4`를 재사용하며 모듈 수준 `pub(crate)` 함수로 올렸다. `load_text_weights`는 `quantization_config`(최상위 또는 `text_config`) 또는 `hf_quant_config.json`에 ModelOpt NVFP4를 선언한 모든 text 체크포인트에 `normalize_nvfp4_keys`와 `repack_nvfp4_weights_to_quantized`를 실행한다. `hf_quant_config.json`은 검사 전에 `config.json` 형태로 변환된다. `quant_method`와 producer 이름은 대소문자를 구분하지 않고 비교한다.
- repack은 MLX의 safetensors 로더가 raw uint8 바이트로 반환하는 F8_E4M3 block scale을 디코딩한다. 숫자 캐스트를 하면 바이트를 0..255로 읽게 된다.
- `switch_layers.rs`의 공용 helper `stack_expert_leaf`와 `EXPERT_PLANE_LEAVES`를 DeepSeek-V3, Mistral 4, PhiMoE 스태킹 루프와 일반 per-expert stacker가 사용한다. 따라서 expert별 `weight_scale_2` sidecar가 weight/scales/biases와 같은 pass에서 `[num_experts]` 크기의 `inv_global_scale` 벡터가 된다. PhiMoE와 Mistral 4는 16 폭 block 크기를 `quantization_config`(`modelopt_nvfp4_group_size`)에서 읽는다. DeepSeek-V3의 expert 경로는 affine 전용이며, 이제 `.biases` 누락을 보고하는 대신 block-float plane을 이름과 함께 거부한다.

## 변경 사항

- `src/lib/mlxcel-core/src/lib.rs`, `cpp/mlx_cxx_bridge.h`, `cpp/mlx_cxx_bridge.cpp`: `gather_qmm`에 nullable `global_scale: *const MlxArray`가 추가된다. 기존 호출자는 모두 null을 넘긴다. 테스트 전용 `quantize_weights_with_global_scale`이 `mlx::core::quantize(..., global_scale)`를 감싼다. 양자화 embedding lookup은 block-float 테이블에 uint8 출력 dtype을 더 이상 요청하지 않는다.
- `src/models/switch_layers.rs`: 두 scale 필드, config 플래그 처리, 로드 시 검증, 모드 추론, `stack_expert_leaf`.
- `src/models/sanitize.rs`: `apply_native_global_scale_flags`, 모든 text 계열에 대한 ModelOpt repack, F8_E4M3 디코딩.
- `src/models/laguna_sanitize.rs`, `laguna.rs`, `laguna_layers.rs`, `laguna_tests.rs`: `inv_global_scale`로 이름 변경.
- `src/models/deepseek_v3.rs`, `mistral4.rs`, `phimoe.rs`: 공용 helper를 통한 sidecar 스태킹.
- `src/models/gemma4.rs`: `is_modelopt_nvfp4`를 모듈 수준 함수로 올림. Gemma 4 sanitizer의 플래그 해석.
- `src/models/gpt_oss.rs`: 거부 guard (아래 참조).
- `scripts/mlx_lm_logit_trace.py`: `examples/logit_trace` TSV 형식의 mlx-lm 참조 tracer. `scripts/laguna_oracle_trace.py`의 tokenizer와 chunk 계획을 재사용한다.
- `docs/supported-models.md`: 두 형식을 문서화한다.
- 그 외 모델 파일과 ROCm 테스트 파일 두 개는 새 `gather_qmm` 및 `SwitchLinear` 시그니처에 맞춰 인자 한 줄을 갱신한다.

## 검증

모두 GB10 CUDA 호스트(sm_121)에서 `--profile test-fast --features cuda`로 실행했다. 핀된 MLX는 CUDA에서 nvfp4에 대한 `gather_qmm(..., global_scale)`을 지원한다(qmv와 qmm_naive는 적용하고, qmm_sm80과 grouped-GEMM 경로는 거부한다). 먼저 mlx 0.32.4의 Python probe로 확인한 뒤 Rust 테스트로 확인했다. Metal은 Metal nightly에서 검증한다.

새 테스트 (모두 CUDA에서 실행):

- `switch_layers::global_scale_tests`: 정렬/비정렬 레이아웃에서 네이티브 경로와 `x @ dequantize(w[e]).T * gs/2688` 비교, float weight 대비 상대 L2 검사, 플래그 제거 시 로드 실패, 텐서 없는 플래그 실패, 예약 키 거부, nvfp4 / shape / 배타성 검사, ModelOpt sidecar 스태킹 및 적용, 부분 sidecar 거부.
- `sanitize_tests::load_text_weights_repacks_modelopt_nvfp4_for_non_gemma_families`(config.json 및 hf_quant_config.json 감지, E4M3 바이트 보존), `load_text_weights_resolves_flagged_native_global_scales`, `load_text_weights_leaves_undeclared_triplets_alone`.
- `phimoe`, `mistral4`, `deepseek_v3`의 계열별 스태킹 테스트.

되돌림 검사:

- Part 1을 되돌리면(네이티브 scale을 기존 post-multiplier로 적용, bare 키 허용, 플래그 무시) `native_global_scale_matches_dense_dequantized_reference`, `global_scale_without_the_config_flag_fails_to_load`, `config_flag_without_global_scale_tensor_is_a_load_error`, `native_global_scale_needs_nvfp4_and_one_entry_per_expert`, `load_text_weights_resolves_flagged_native_global_scales`가 실패한다.
- gpt-oss guard를 제거하면 `gpt_oss_expert_linear_refuses_a_per_expert_global_scale`이 실패한다. 스태킹되지 않은 sidecar guard를 제거하면 `a_plane_whose_family_left_the_modelopt_sidecars_unstacked_is_refused`가 실패한다.
- Part 2를 되돌리면(sidecar leaf 없음, F8 디코딩 없음) PhiMoE, Mistral 4, DeepSeek-V3 스태킹 테스트, `per_expert_sidecars_stack_into_the_output_multiplier`, `a_partial_per_expert_sidecar_is_refused`, `load_text_weights_repacks_modelopt_nvfp4_for_non_gemma_families`가 실패한다.

`--test-threads=1`로 실행한 모듈 스위트: switch_layers, laguna, sanitize, phimoe, mistral4, deepseek_v3, deepseek, gemma4, olmoe, qwen3_moe, inkling, kimi_k3, deepseek_v4, gpt_oss, glm4_moe, step3p5, nemotron_h, hunyuan, llama4, exaone, ernie, qwen3_next가 모두 통과한다. `cargo clippy --profile test-fast --features cuda --lib --tests -- -D warnings`, `cargo check -p mlxcel-core --lib --tests`, `cargo fmt --check`, 다섯 가지 contract 검사도 통과한다.

실제 체크포인트:

- Laguna 회귀, `laguna-xs-2.1-nvfp4`(`poolside/Laguna-XS-2.1-NVFP4`), 이제 `inv_global_scale` 사용: `MLXCEL_SDPA_DETERMINISTIC=1 mlxcel generate ... -n 64 --temp 0 --show-reasoning` 결과가 `main`(8744c87f, 변경 전에 같은 worktree에서 빌드, main의 자기 일치 먼저 확인)과 토큰 단위로 동일하다.
- 이미 동작하던 ModelOpt 경로, `gemma-4-31b-it-nvfp4`: 같은 명령으로 `main`과 토큰 단위로 동일하다.
- 실제 mlx-lm export, `olmoe-1b-7b-0924-nvfp4-gs`. 제작 방법: `allenai/OLMoE-1B-7B-0924`(HF bf16, config에 `rms_norm_eps`가 없어 transformers 기본값 `1e-5`를 mlx-lm용으로 추가)를 `mlx_lm.utils.load(..., lazy=True)`로 로드하고, float를 bf16으로 캐스트한 뒤 `quantize_model(model, config, None, None, mode="nvfp4", global_scale=True)`, `save(...)`를 실행했다. mlx-lm 0.32.2(mlx 0.32.4, `mlx-cuda-13` wheel)가 이미 `global_scale` 경로를 포함하므로 `ba8a08f` checkout 대신 사용했다. 48개 switch layer가 플래그를 가진다. `main`은 로드 시 거부한다(`quantization.mode is "affine" but the checkpoint ships no .biases plane`). 이 브랜치에서 `mlxcel generate -p "The capital of France is Paris. The capital of Germany is" -n 40 --temp 0`은 "Berlin. The capital of the United Kingdom is London. The capital of the United States is Washington, D.C."를 출력한다(mlx-lm greedy: "Berlin. The capital of Italy is Rome. ..."). config 플래그를 제거하면 `model.layers.0.mlp.switch_mlp.gate_proj`를 명시하며 로드에 실패한다.
- 해당 export에 대한 teacher-forced logit trace (corpus `docs/architecture.md`, `256 4 8 0`, 참조 `scripts/mlx_lm_logit_trace.py`(mlx-lm 0.32.2 CUDA), `scripts/compare_logit_traces.py`로 비교):

| 구성 | 결정된 위치 불일치 (gap >= 2) | 전체 불일치 | 기타 |
|------|-------------------------------|-------------|------|
| 수정 후 | 396개 중 0 | 1024개 중 19, 모두 참조 gap 0.5 미만 | perplexity 20.15 대 20.13 |
| 네이티브 scale을 기존 post-multiply로 교체 | 396개 중 396 | 1024개 중 1024 | greedy 출력이 깨짐 ("ByrPLAYpackavelorschKK Television...") |

- ModelOpt non-Gemma: 이 호스트에 맞는 크기로 공개된 DeepSeek-V3, Mistral 4, PhiMoE의 ModelOpt NVFP4 MoE 체크포인트가 없다(`mistralai/Mistral-Small-4-119B-2603-NVFP4`는 Mistral consolidated 형식). 따라서 합성 triplet 테스트가 스태킹을 검증한다. `nvidia/Qwen3-30B-A3B-NVFP4`도 시도했다. 18624개 triplet이 이제 repack되고 expert가 스태킹되지만, 모델은 여전히 dense fused QKV projection에서 abort된다(`FusedQKVLinear`가 affine을 하드코딩하고 global-scale 경로가 없음). 이는 이 이슈의 범위 밖이다. `main`은 같은 체크포인트에서 matmul shape 오류로 abort된다.

## 리뷰 중 변경

ModelOpt 로딩을 모든 text 계열로 확장하면서 발견된 조용한 오출력 사례를 세 개의 후속 커밋이 막았다.

- `6e13adf1`: 자체 weight/scales/biases 스태킹 루프를 쓰면서 공용 `SwitchLinear`로 로드되는 계열(OLMoE, dots1, Solar Open, Cohere2-MoE 등)은 코드와 block scale만 스태킹하고 expert별 `weight_scale_2` sidecar를 읽지 않은 채 두었다. 새 모드 추론 덕에 일반 nvfp4로 로드되어 모든 routed expert가 `weight_scale_2`만큼 어긋났다. `main`에서는 이런 체크포인트가 로드에 실패했지만, 이 PR이 있으면 조용히 틀리게 디코딩될 상황이었다. `SwitchLinear::load`는 이제 두 global scale이 모두 없는 bias 없는 uint8-scale plane이 MoE block 아래에 expert별 `.global_scale` sidecar를 남겨 두고 있으면, 남은 키를 명시하며 실패한다. 새 `global_scale_tests` 케이스가 OLMoE 방식 스태킹을 재현하며, guard가 없으면 로드되고 guard가 있으면 거부된다.
- `19567aa5`: gpt-oss의 자체 `ExpertLinear`는 config 모드로 tensor scale 없이 `gather_qmm`을 호출하므로, `global_scale: true` mlx-lm export는 로드되지만 모든 expert가 틀리게 디코딩된다. 로더는 이제 `.native_global_scale` 또는 bare `.global_scale`을 가진 plane을 키를 명시하며 거부한다.
- `d61ae3a8`: weight가 `[num_experts, out, in]` stack이 아닌 플래그 경로는 scale이 이름만 바뀌고 읽히지 않으므로, 로드 funnel이 이를 거부한다. ModelOpt 감지도 `is_modelopt_nvfp4`처럼 `quant_method`와 `hf_quant_config.json` producer 이름을 대소문자 구분 없이 비교한다.

`git diff 8744c87f..HEAD`에 대한 보안 및 성능 리뷰에서 CRITICAL 또는 HIGH 항목은 없었다. FFI 전제 조건, `apply_native_global_scale_flags`, `stack_expert_leaf`, ModelOpt repack의 바이트 길이 guard, 디코드 경로를 점검했고 문제가 없었다. 리뷰는 다음 LOW 항목을 나열했으며 리뷰에서 수정되지는 않았다.

- `refuse_unstacked_expert_sidecars`는 bias 없는 block-float plane마다 모든 키를 한 번 스캔하므로 로드 비용이 O(plane 수 x 키 수)이다. 리뷰는 61 layer, 384 expert의 per-expert 맵이 스태킹되지 않은 경우에도 1초 미만으로 추정했다.
- `load_text_weights`와 Gemma 4 sanitizer만 플래그 resolver를 실행하므로, 다른 로더는 체크포인트가 가져온 `.native_global_scale`을 거부하지 않는다. 값은 여전히 MLX 전제 조건으로 검사되므로 abort는 발생하지 않는다.
- 공유 `SwitchLinear` plane이 아닌 플래그 경로는 scale이 `.native_global_scale`로 옮겨지고 아무도 읽지 않는다. 리뷰는 소비되지 않은 `.native_global_scale` 키가 있으면 실패하는 빌드 후 검사를 제안했다.
- `modelopt_quantization_config`가 `quant_method == "modelopt"`를 대소문자 구분하여 비교했다. `d61ae3a8`의 대소문자 무시 비교가 이를 해결한다.
- 기존 문제: DeepSeek v1 전용 로더는 `.scales`는 있고 `.biases`가 없는 plane을 `Regular`로 취급하여 첫 forward에서 `gather_mm`이 abort된다.

리뷰 이후에도 Laguna XS 2.1 greedy 출력은 `main`과 토큰 단위로 동일하며, OLMoE export의 greedy 출력도 변하지 않았다.

## 미해결 사항

- NVFP4 + `weight_scale_2`를 가진 dense fused QKV (Qwen3-MoE ModelOpt 체크포인트를 막음).
- Gemma 4 외의 VLM 로더는 플래그를 해석하지 않는다. expert별 global scale을 가진 mlx-vlm export(Blaizzy/mlx-vlm#2437)는 bare 키가 공유 `SwitchLinear` prefix에 도달하는 곳에서는 로드에 실패한다. 다만 Gemma 4 unified 로더의 `split_fused_moe_key`(`src/loading/vlm_gemma_unified.rs`)는 fused `experts.gate_up_proj`의 `.weight` / `.scales` / `.biases`만 분리하므로 `.global_scale`이 고아로 남는다. 이는 기존 문제이며 이 이슈 범위 밖의 로딩 파일에 있다.
- 위에 나열한 LOW 후속 항목: 로드 시 키 스캔 비용, text 및 Gemma 4 funnel에서만 이뤄지는 플래그 해석, DeepSeek v1 전용 로더.
- #2351에 따라 범위 밖: dense-linear global scale(`QuantizedWeight::global_scale`)과 `global_scale` export를 생성하는 `mlxcel` quantize/convert.

## 통계

| 항목 | 값 |
|------|-----|
| 변경된 파일 | 34 |
| 추가된 줄 | 1995 |
| 삭제된 줄 | 137 |

| 커밋 | 메시지 |
|------|--------|
| `883839e4` | fix(moe): load mlx-lm NVFP4 per-expert global_scale natively |
| `6e13adf1` | fix(moe): refuse NVFP4 expert planes with unstacked ModelOpt sidecars |
| `19567aa5` | fix(gpt-oss): refuse expert planes that carry an NVFP4 global scale |
| `d61ae3a8` | fix(moe): refuse a global_scale flag on a non-expert layer |
