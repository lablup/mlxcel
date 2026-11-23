# PR #3370: OpenXLA Mamba2 config, mixer schedule, 상태를 잇는 그래프

**날짜**: 2026-11-24
**상태**: PR 열림(#3338을 닫음, epic #564 Phase 1 단위 U2). ROCm gfx1151에서 IREE `local-task`와 `hip`으로 검증했다. `generate`에서는 아직 닿지 않는다. shim 상태 버퍼와 런타임은 #3339, 슬롯별 상태를 갖춘 ragged decode는 #3340이다.

## 요약

#3360 이후 OpenXLA/IREE emitter는 Mamba2 mixer 한 층은 표현할 수 있었지만 모델 전체는 표현하지 못했다. `Config::from_json`이 `mamba2`를 거부했고, 모든 그래프가 모든 층을 KV cache가 있는 attention으로 가정했다. 이 PR은 층별 mixer schedule(`MixerKind`), 일반화된 상태 전달 ABI(`StateTensorSpec`), 새 `take_ssm_layer_weights`와 같은 순서로 선언되는 `backbone.*` weight spec, 그리고 `ssm_h [L,H,P,N]`과 `ssm_conv [L,K-1,conv_dim]`을 받고 돌려주는 prefill과 단일 시퀀스 decode 그래프를 추가한다. 이 그래프에는 KV cache, RoPE 테이블, mask가 없다. dense와 MoE 그래프는 바이트 단위로 동일하다.

HF `Mamba2ForCausalLM` fp32와 비교하면 `AntonV/mamba2-130m-hf`(200 토큰, scan 청크 두 개)의 logits 최대 절대 오차는 `local-task`에서 2.0e-4, `hip`에서 3.9e-4이고 argmax가 같다. 이 결과는 세 경로 모두에서 나온다. prefill 한 번, 상태를 넘기는 prefill 두 번, 토큰 단위 decode다. prefill과 decode는 `llvm-cpu`와 `rocm` gfx1151에서 capacity 256과 4096 모두 1.6~2.6초에 컴파일된다. f16에서는 logits가 fp32 대비 0.42~0.45 벗어나지만 argmax와 top 5는 같다. 이 오차는 측정했고 projection 반올림으로 원인을 추적했으며, 게이트로 걸지는 않았다.

이제 `config.json`과 safetensors 헤더는 그래프와 상태 크기를 정하는 신뢰할 수 없는 입력이므로, 이 PR은 SSM 차원을 모두 체크포인트와 대조하고, 각 헤더 항목이 파일 안에서 정확한 바이트 범위로 뒷받침되는지 확인하고, 층별 상태를 `in_proj` 원소 수로 제한하고, 모든 config의 `num_hidden_layers`를 1024로 제한한다.

## 1. 문제 정의

#3360(#3337)은 `SsmConfig`, conv와 scan primitive, `mamba2_mixer_prefill` / `mamba2_mixer_step`을 제공했고, probe 그래프로 한 층씩 검증했다. 이를 모델로 만들려면 emitter에 없던 네 가지가 필요했다.

- **Schedule.** `Config`는 균일한 attention 스택을 기술했다. Mamba2 층은 FFN 없이 `norm -> mixer -> residual`이고, epic #564의 이후 hybrid(Nemotron-H, Jamba, Granite 4)는 attention 층과 순환 층을 섞으므로 선택이 층별이어야 한다.
- **Attention 없는 config.** 공유 코드는 `n_q / n_kv`(`Config::group`)를 읽고, `head_dim`으로 RoPE 테이블을 만들고, causal mask를 만들고, `[L, S, n_kv, d]` cache를 조건 없이 할당했다. head 수가 0이면 0으로 나누거나 폭 0인 텐서를 emit한다.
- **상태 계약.** prefill과 decode는 고정된 KV ABI를 가졌다. 순환 상태는 자기 shape을 가진 자기 텐서가 필요하고, prefill 결과에서 decode로 이어져야 하며, ABI는 이후 계열에도 쓸 수 있게 일반적이어야 한다(#3338에 기록된 오케스트레이터 결정: 전달되는 상태 텐서의 목록, KV는 선택, dummy KV 없음).
- **신뢰할 수 없는 차원.** #3360의 `SsmConfig::validate`는 overflow와 non-finite 값만 막는다. 차원이 모델 디렉터리에서 오면 `config.json`은 함께 배포된 weight보다 훨씬 큰 상태와 그래프를 요구할 수 있다. #3338의 이월 코멘트는 체크포인트 텐서 shape과의 대조와 호스트 쪽 `real_len` 제한을 요구했다.

이월 코멘트는 여기서 정리할 #3360의 한계 두 가지도 적었다. f16 prefill conv가 decode step과 다르게 반올림한다는 것, 그리고 chunked SSD의 청크 간 carry 비용이 `H*(T/Q)^3` 덧셈이라 매우 긴 context bucket에서만 문제가 된다는 것이다.

## 2. 변경 요약

| 파일 | 변경 |
|---|---|
| `emitter/recurrent_config.rs` (신규) | `MixerKind { Attention, Mamba2 }`, `StateElem`, `StateTensorSpec`; `Config::{layer_mixer, layer_has_ffn, recurrent_layer_count, state_layer_index, has_kv_cache, state_tensors, has_recurrent_state, reject_recurrent}`; `mamba2_from_json`; HF의 bare `Infinity`를 위한 `quote_non_finite_json` |
| `emitter/recurrent_checkpoint.rs` (신규) | `Config::validate_recurrent_checkpoint`: 정확한 바이트 범위를 검사하는 safetensors 헤더 리더, 텐서별 shape 대조, 상태 상한 |
| `emitter/model_recurrent.rs` (신규, `model`의 자식) | `RecurrentLayers`(쌓인 상태를 층별 행으로 나누고 다시 합침), `take_ssm_layer_weights`, `emit_mamba2_block`, `take_state_args`, `load_state_globals`, `decode_state_entries`(`@ssm_h` / `@ssm_conv` global과 `@state_import`) |
| `emitter/config.rs` | `ssm`, `layer_mixers` 필드, `WeightScheme::Mamba2`, `MAX_CONFIG_LAYERS = 1024`와 `check_layer_count`(dense parser에도 적용), `from_json`이 체크포인트 대조 실행, KV head가 0이면 `group()`이 0 반환, SSM에서 `supports_packed_quant` false |
| `emitter/model.rs` | `emit_transformer_layer`가 Mamba2 블록으로 분기; RoPE 테이블을 `Option`으로; attention이 없으면 decode에서 `@kcache` / `@vcache` / `@kv_import` 제거; prefill은 `real_len` 뒤에 상태 입력을 받고 logits 뒤에 돌려줌; `emit_decode_buffer_abi`; batched, ragged, 모든 `prefill_embeddings*` 진입점이 `Result`를 반환하고 순환 config를 거부 |
| `emitter/ssm.rs` | prefill conv를 `K`번의 shift된 f32 곱셈-덧셈으로 재작성 |
| `weights.rs`, `weight_names.rs` | `take_ssm_layer_weights` 순서의 `push_ssm_layer_specs`; Mamba2 scheme 이름 |
| `iree.rs`, `shard_path.rs` (신규) | `whole_graph_shape`(MLX `[C,K,1]` conv를 `[C,1,K]`로 업로드), 프로덕션 로더를 거치는 `load_host_weights`; emitter가 `iree` 없이 빌드되도록 `index_shard_path`를 `shard_path.rs`로 이동; 호출부가 새 `Result`를 전파 |
| `ssm_model_check.rs` (신규), `examples/xla_ssm_model_check.rs`, `Cargo.toml`, `auxiliary.rs` | auxiliary IREE ABI 위의 모델 전체 runner: prefill, 나눈 prefill, decode 세션, 컴파일 시간 측정 |
| `validation.rs`, `emitter/mod.rs`, `emitter/context_tests.rs` | `GraphKind::emit`이 `Result` 반환; 실패할 수 있게 된 emitter 호출부에 `.unwrap()` |
| `emitter/recurrent_tests.rs` (신규) | 테스트 16개(파싱, ABI shape, weight 순서, 거부, 대조 강화) |
| `spike/openxla/ssm_model_check.py`, `FINDINGS_ssm.md`, `README.md`, `CHANGELOG.md` | seed 고정 합성 모델을 포함한 HF fp32 오라클, 결과, 컴파일 시간, 이월 항목 처리 |

23개 파일, +3082 / -176(`origin/main` 병합 전 기준).

### Schedule과 ABI

기존 parser는 모두 `ssm = None`, `layer_mixers = [Attention; n_layers]`를 설정하므로 `state_tensors()`는 비어 있고, `has_kv_cache()`는 true, `layer_has_ffn(li)`도 모든 층에서 true다. 새 분기는 모두 이 질의를 기준으로 하므로 dense golden이 바뀌지 않는다. `mamba2_from_json`은 HF 키(`num_heads`, `head_dim`, `state_size` 또는 `ssm_state_size`, `n_groups`, `conv_kernel`, `expand`, `use_conv_bias`, `use_bias`, `time_step_limit`, `layer_norm_epsilon`)로 config를 만들고, attention 필드는 0, `norm_group_size = d_inner`, `chunk_size = 128`(#3360)로 두며, `rms_norm = false`, silu가 아닌 `hidden_act`, `num_heads * head_dim != expand * hidden_size`를 거부한다.

Mamba2의 `state_tensors()`는 `ssm_h [L_ssm, H, P, N]`, 그다음 `ssm_conv [L_ssm, K-1, conv_dim]`이고 둘 다 f32다. `L_ssm`은 순환 층 수이고 `state_layer_index(li)`가 층의 행 번호를 준다.

| 그래프 | 상태 계약 |
|---|---|
| 토큰 prefill(`prefill.main`) | 입력: 기존 입력, 그다음 `real_len` 뒤에 `ssm_h`, `ssm_conv`. 결과: logits, `has_kv_cache()`일 때만 KV cache, 그다음 상태 텐서. 0으로 채우면 새 시퀀스. |
| 단일 시퀀스 decode(`@decode_step`) | `util.global private mutable @ssm_h` / `@ssm_conv`, `@state_import(%s0, %s1)`로 초기화. `@main`은 `token`, `pos`, `cache_len`을 유지하지만 token만 읽는다. `@kcache`, `@vcache`, `@kv_import` 없음. |
| uniform batched, ragged, embeddings prefill(DeepStack 포함) | #3340 전까지 모델 타입을 명시한 `Err` |

그래프 안에서 `RecurrentLayers::split`이 쌓인 텐서에서 각 층의 행을 잘라내고(prefill은 batch 축 없음, decode는 batch 1), `emit_mamba2_block`이 `x + mixer(norm(x))`를 실행해 갱신된 행을 되써넣으며, `join`이 결과나 global 저장을 위해 행을 다시 쌓는다.

### Weight 경로

`push_ssm_layer_specs`는 층마다 `norm.weight`, `mixer.in_proj.weight`(+ bias), `conv1d.weight`(+ bias), `dt_bias`, `A_log`, `D`, `norm.weight`, `out_proj.weight`(+ bias)를 나열한다. `take_ssm_layer_weights`는 같은 순서로 소비해 #3360의 `Mamba2MixerW`를 만들고, 테스트가 두 순서를 비교한다. projection은 `push_proj`를 거치므로 f16 상주 경로가 `take_weight`와 맞는다. 로더 쪽 세부 사항이 두 가지 있다. MLX `[C,K,1]` conv weight는 torch의 `[C,1,K]`로 선언, 업로드된다(row-major 바이트는 같다). 그리고 mixer의 projection이 단일 행 native 경로가 없는 시퀀스 helper를 쓰므로 SSM config에서는 packed / native quant가 꺼진다.

### 체크포인트 대조

`Config::from_json`은 이제 `validate_recurrent_checkpoint(model_dir)`를 호출한다. attention 전용 config에서는 아무것도 하지 않는다. Mamba2에서는 safetensors 헤더(단일 파일 또는 index)를 읽고, 무엇이든 emit하기 전에 다음을 확인한다.

- 층마다 `in_proj [in_proj_dim, hidden]`, `out_proj`, `conv1d`(`[conv_dim, 1, K]` 또는 MLX `[conv_dim, K, 1]`), `dt_bias` / `A_log` / `D [H]`, gated-norm weight `[d_inner]`, embedding과 final norm, untied일 때 `lm_head.weight`. 양자화된 projection은 packed 폭을 checked 산술로 확인한다.
- 각 헤더 항목이 파일 데이터 영역 안에서 정확히 `product(shape) * dtype` 바이트를 차지해야 한다. 그래야 작은 파일이 큰 차원을 보증할 수 없다.
- 층별 상태 `H*P*N`이 그 층의 `in_proj` 원소 수를 넘지 않아야 한다(`H*P`와 `G*N`은 각각 제한됐지만 곱은 제한되지 않았다).
- index shard 이름은 순수 파일 이름이어야 하고(`index_shard_path`), 집합으로 중복을 제거한다.

## 3. 기술적 선택과 그 이유

**일반화된 상태 목록, KV는 선택, dummy KV 없음.** epic 수준에서 결정되어 #3338에 명시됐다. 대안은 Mamba2 전용 ABI(지금은 싸지만 이후 계열마다 ABI가 하나씩 늘어남)나 attention 없는 모델에 크기 0인 KV cache를 두는 것(ABI는 하나지만 shim이 아무 층도 읽지 않는 버퍼를 할당하고 전달함)이었다. 목록 형태에서는 Nemotron-H나 Jamba가 prefill과 decode 배관을 바꾸지 않고 `state_tensors()`에 텐서를 추가하면 된다. 대가는 shim(#3339)이 `has_kv = 0`과 `n_state > 0`을 별개의 경우로 다뤄야 한다는 점이다.

**층마다 텐서가 아니라 쌓은 `[L_ssm, ...]` 상태 텐서.** 깊이와 무관하게 그래프당 인자가 두 개이므로, 그래프 시그니처와 #3338이 shim에 정한 `XLA_MAX_STATE = 8` 상한이 `n_layers`에 의존하지 않는다. 그래프는 층마다 slice와 update 하나씩을 더 치르며, mixer에 비하면 무시할 만하다.

**계열이 아니라 질의로 분기.** 공유 코드는 `weight_scheme == Mamba2`를 검사하지 않고 `has_kv_cache()`, `layer_mixer(li).is_recurrent()`, `state_tensors()`를 묻는다. RoPE 테이블은 `Option<Val>`이 되었고 접근자는 attention 경로에서만 panic하므로, attention 없는 config가 없는 테이블을 조용히 읽을 수 없다. 그러면 hybrid는 prefill이나 decode에 새 분기 없이 두 종류가 섞인 schedule만 있으면 된다.

**상태 계약이 없는 그래프는 거부.** uniform batched decode, ragged decode, 모든 embeddings prefill 진입점은 이제 `Result`를 반환하고 순환 config를 모델 타입을 담은 메시지로 거부한다. 상태를 조용히 버리는 그래프를 emit하지 않는다. 이 때문에 공개 emit 함수의 시그니처와 모든 테스트 호출부가 바뀌었다. 대안인 panic은 해당 진입점만 건너뛰면 되는 config에서 `compile_vmfbs` 전체를 죽였을 것이다. 구현 리뷰가 지적한 대로 #3339는 순환 config에서 `prefill_embeddings`를 건너뛰어야 한다.

**decode는 `token`, `pos`, `cache_len`을 유지.** Mamba2는 token만 읽는다. 세 scalar를 유지하면 shim 입장에서 `@main` 호출 규약(#3060의 scalar ABI 또는 buffer ABI)이 하나로 유지되고, 대가는 읽지 않는 인자 두 개다.

**f32 prefill conv를 `K`번의 shift된 곱셈-덧셈으로.** #3360은 f16 prefill conv가 피연산자를 f16으로 반올림하는 반면 decode conv는 f32로 누적한다는 점, 그리고 IREE에서 mixed-type depthwise convolution이 컴파일되지 않는다는 점을 기록했다. `K = 4`이면 shift 형태는 slice, 곱셈, 덧셈 네 번이고 모든 정밀도 경로에서 같으므로, 이제 prefill과 decode는 같은 산술을 한다. #3360의 SSM probe 그래프는 바뀌지만 dense golden은 바뀌지 않는다.

**HF tie 기본값.** `tie_word_embeddings`가 없으면 HF `Mamba2Config`처럼 false다. 첫 버전은 true를 기본으로 했고, 그러면 자체 `lm_head.weight`를 가진 체크포인트에서 embedding을 head로 조용히 썼을 것이다(구현 리뷰에서 발견, MEDIUM). 이후 대조 단계가 untied config에 `lm_head.weight`를 요구한다.

**프로덕션 로더와 `InOut` 레이아웃을 거치는 모델 검사.** 검사는 런타임의 `upload_weights_with_sink`를 감싼 `load_host_weights`로 weight를 읽으므로, MLX conv 재해석과 f16 상주 경로가 #3339가 실행할 방식 그대로 검증된다. `IreeLlama::load`가 항상 `InOut`을 쓰므로 기본값도 `InOut`이다. 첫 버전은 config의 `OutIn`으로 돌렸다. 두 레이아웃의 수치는 같다.

**bounded chunk carry는 보류.** carry는 `H * nc^3` 덧셈이고 청크 내부 작업은 `H * T * Q * (P + N)` 차수다. 가장 큰 ladder capacity인 32768(`nc = 256`)에서 mamba2-130m 기준 약 2%이고, 역전 지점은 T = 185k~230k 근처다. FINDINGS는 "약 64k를 넘는 context bucket을 켤 때 다시 본다"고 기록한다.

**f16은 측정만 하고 게이트로 걸지 않음.** f16에서 3e-2 게이트는 한 자릿수 차이로 실패하지만(-3.4~16.7 범위의 logits에서 0.42~0.45) argmax와 top 5는 같다. HF 자체의 fp32 forward 안에서 모든 `nn.Linear` 입력과 weight를 f16으로 반올림하면 0.44가 재현된다(`in_proj`만 반올림하면 0.39, `dt` 열을 f32로 두어도 0.40). 따라서 오차는 `x`, `B`, `C`, `z`로 들어오고, 그래프 쪽 변경으로는 없앨 수 없다. 그 시뮬레이션에서 프롬프트 다섯 개의 greedy 40 토큰은 fp32와 같다. #3339가 순환 config에서 f32를 고정할지(`effective_precision`이 MLA에 하듯이)는 그쪽의 토큰 단위 게이트에 맡긴다.

## 4. 리뷰 결과

**구현 리뷰**(e17b0879):

| 발견 사항 | 심각도 | 상태 |
|---|---|---|
| `tie_word_embeddings`가 없을 때 true가 기본값이었다; HF 기본값은 false | MEDIUM | 수정, 회귀 테스트 |
| 모델 검사가 `OutIn`으로 실행됐다; `IreeLlama`는 항상 `InOut` | MEDIUM | 수정: `--layout`, 기본 `inout`; 수치 동일 |
| packed 폭 검사에서 헤더 차원에 32를 unchecked로 곱했다; `2^59 + 2`가 기대 폭으로 wrap | LOW | 수정: checked 곱셈, 테스트 |
| HF torch 경로와 MLX 로더는 gated 출력을 `d_inner` 전체로 정규화; `mamba_ssm`은 `n_groups > 1`이면 그룹별로 정규화 | MEDIUM | 열림: Codestral Mamba 같은 `n_groups > 1` 체크포인트를 켜기 전에 확인 |
| `@state_import`는 여기서 컴파일만 되고 실행되지 않음 | 참고 | 열림: #3339 gate 1이 첫 실행 |
| HF가 무시하는 `rms_norm = false`를 거부 | LOW | 열림, 보수적 선택 |

**보안 및 성능 리뷰**(d9cb7b80). 신뢰 경계: 신뢰할 수 없는 모델 디렉터리.

| 발견 사항 | 심각도 | 상태 |
|---|---|---|
| `num_hidden_layers`가 검사 전에 `layer_mixers` 크기를 정함 | MEDIUM | 수정: 파싱 시점에 1024로 제한, Mamba2와 dense 모두 |
| 헤더 shape을 데이터 영역과 대조하지 않고 신뢰; `compile_ssm_model_graphs`는 weight를 읽지 않으므로 주장된 차원으로 컴파일 | MEDIUM | 수정: 정확한 `product(shape) * dtype` 범위 요구 |
| 층별 상태 `H*P*N`에 상한 없음 | MEDIUM | 수정: 층의 `in_proj` 원소 수로 제한 |
| 로컬 shard 이름 검사가 `..`, `/`, NUL을 통과시킴; 이차 시간 중복 제거 | LOW | 수정: 공유 `index_shard_path`, 집합 |
| Python 검사가 체크포인트와 `.npy`를 느슨하게 읽음 | LOW | 수정: `use_safetensors=True`, `allow_pickle=False` |

변경 없이 확인한 것: conv 재해석은 원소 수를 유지하고, `SafeTensors::deserialize`가 바이트 길이를 강제하며, token id와 프롬프트 길이는 호스트에서 검사되고, `--compile-times`는 호출자가 지정한 디렉터리에만 쓰며, Python runner는 `trust_remote_code`가 없고 subprocess 인자를 리스트로 넘긴다. 각 수정에는 수정 없이 실패하는 단위 테스트가 있고, finalize 커밋(08ed5b36)은 헤더 범위 검사와 층 수 제한을 각각 되돌리면 해당 테스트가 실패함을 확인했다.

**#3360에서 이월된 항목**(FINDINGS 기준): 차원은 대조된다(위). `real_len`은 런타임 prefill과 shim이 이미 `[0, capacity]` 밖을 거부하고, 검사 runner는 빈 프롬프트, capacity 초과, 어휘 밖 토큰을 거부한다. bounded carry는 보류(3절). f16 conv 차이는 해소(3절).

## 5. 테스트와 검증

`spike/openxla/ssm_model_check.py`와 `examples/xla_ssm_model_check.rs`, 마지막 토큰 logits를 HF `Mamba2ForCausalLM` fp32(transformers slow path)와 비교, 허용치 3e-2 최대 절대 오차와 argmax 일치:

| 모델 | 디바이스 | 정밀도 | 전체 | 나눔 | decode | argmax |
|-------|--------|-----------|------:|------:|-------:|--------|
| 합성(2층, H=8, G=2, bias 있음, untied), 40 토큰, capacity 128 | local-task | f32 | 1.7e-6 | 1.3e-6 | 1.7e-6 | 같음 |
| 같음 | hip | f32 | 1.8e-6 | 1.2e-6 | 1.0e-6 | 같음 |
| `AntonV/mamba2-130m-hf` @ `05e8773f`, 200 토큰(청크 두 개), capacity 256 | local-task | f32 | 2.0e-4 | 1.7e-4 | 1.1e-4 | 같음 |
| 같음 | hip | f32 | 3.5e-4 | 3.9e-4 | 3.3e-4 | 같음 |
| 같음 | local-task | f16 | 4.3e-1 | 4.5e-1 | 3.7e-1 | 같음(top 5 같음) |
| 같음 | hip | f16 | 4.2e-1 | 4.4e-1 | 4.3e-1 | 같음(top 5 같음) |

f16 안에서는 세 경로끼리가 f16과 fp32보다 훨씬 가깝다. 전체 대 decode는 6.6e-2(local-task)와 4.2e-2(hip), 전체 대 나눔은 3.8e-2와 5.8e-2다. f16 hip 실행은 예상대로 게이트에 실패하며, 이는 runner가 나쁜 결과에서 실제로 실패한다는 것도 보여준다.

mamba2-130m(24층) f32, 캐시 없는 `iree-compile` 1회 시간:

| 타깃 | capacity | prefill | decode |
|--------|---------:|--------:|-------:|
| llvm-cpu (local-task) | 256 | 2.5 s | 1.7 s |
| llvm-cpu (local-task) | 4096 | 2.6 s | 1.7 s |
| rocm gfx1151 (hip) | 256 | 2.5 s | 1.6 s |
| rocm gfx1151 (hip) | 4096 | 2.6 s | 1.6 s |

단위 테스트: `emitter/recurrent_tests.rs`에 16개(config 파싱과 상태 ABI, attention config 불변, 일관되지 않은 차원, 모든 `Infinity` 표기, JSON quoting helper, `weight_specs_q` 순서와 `take_ssm_layer_weights` 순서 일치, prefill과 decode 시그니처, 상태 계약 없는 모든 그래프의 거부, 모든 정밀도에서의 emit, 층 수 제한, 헤더 범위, index shard 이름, 상태 상한, 모든 SSM 차원, tie 기본값), `recurrent_checkpoint.rs`에 2개, `shard_path.rs`에 1개. `cargo test --profile test-fast -p mlxcel-xla --lib`: `origin/main` 병합 전 PR head에서 506개 통과. 그 밖에 `scripts/xla/validate_arch.sh --structural-only`(dense golden 바이트 동일), `cargo fmt --all -- --check`, 기능 없음, `iree`, example 각각에 대한 clippy `-D warnings`. `origin/main` 병합(충돌 없음) 후 `cargo fmt -p mlxcel-xla -- --check`가 통과한다.

## 6. 남은 작업과 한계

- **통합되지 않음.** 아직 어떤 런타임 경로도 이 그래프를 호출하지 않는다. #3339가 shim 상태 버퍼(`has_kv`, `n_state`, `state_desc`), `RuntimeConfig` 상태 필드, 상태 레이아웃을 반영한 vmfb cache key와 fingerprint, generate 경로를 추가하고, #3340이 ragged decode에 슬롯별 상태를 추가한다. 그때까지 CHANGELOG는 `generate`에서 닿지 않는다고 명시한다.
- **`@state_import`는 실행된 적이 없다.** 모델 검사는 이를 호출할 수 없는 단일 진입점 auxiliary module을 쓰므로, decode 검사는 상태 global을 0 초기값으로 바꿔 쓰고(`zero_initialize_state_globals`) buffer ABI decode(`emit_decode_buffer_abi`)를 실행한다. prefill에서 decode로의 인계와 scalar ABI decode는 컴파일만 되었고 #3339에서 처음 실행된다.
- **`compile_vmfbs`는 이제 Mamba2의 `prefill_embeddings`에서 `Err`를 받으므로** #3339에서 순환 config일 때 그 진입점을 건너뛰어야 한다.
- **`n_groups > 1`의 그룹별 gated norm**(구현 리뷰, MEDIUM, 열림). emitter는 HF torch 경로와 MLX를 따른다(`norm_group_size = d_inner`). 합성 모델은 `G = 2`지만 같은 HF 경로와 비교하므로 이 차이를 가려내지 못한다. Codestral Mamba 같은 체크포인트를 켜기 전에 `mamba_ssm`과 대조해야 한다.
- **f16 오차.** `hip`의 기본값은 f16이고, #3339의 토큰 단위 게이트도 f16으로 돈다. 시뮬레이션상 위험은 낮지만, 순환 config에서 f32를 고정하는 것이 대비책이다.
- **bounded chunk carry 보류**, 약 64k를 넘는 context bucket을 켤 때까지.
- **`rms_norm = false`는 거부된다**, HF는 이 플래그를 무시하는데도.
- **prefill conv 재작성 후 #3360의 probe 게이트 재실행은 보고되지 않았다.** PR의 테스트 계획은 conv를 모델 전체 검사(f32, 합성 모델에서 1.0e-6~1.8e-6)와 단위 테스트로 다룬다.
- **`local-task`와 `hip`(gfx1151)만 실행했다.** CUDA는 측정하지 않았다. 실제 체크포인트는 하나(mamba2-130m, `n_groups = 1`)만 검사했다.

## 7. 학습 포인트

- schedule을 층 구조의 단일 출처로 삼고 공유 코드가 그것에 질문하게 하라(`has_kv_cache`, `is_recurrent`, `state_tensors`). 기존 config가 모두 예전 답을 기본값으로 가지면 golden이 바이트 단위로 유지되고, hybrid는 새 분기가 아니라 데이터만 있으면 된다.
- 실패할 수 있는 emitter는 panic이나 품질이 떨어진 그래프보다 나은 거부 방식이다. 진입점을 건너뛸지 실패할지는 호출자가 정하고, 메시지는 모델 타입을 알려준다.
- safetensors 헤더는 증거가 아니라 주장이다. 각 항목이 파일 안의 정확한 바이트 범위로 뒷받침될 때에만 헤더의 shape이 config를 제한할 수 있다. 그렇지 않으면 몇 KB짜리 파일이 임의의 그래프 크기를 요구할 수 있다.
- 신뢰할 수 없는 차원은 인수만이 아니라 곱도 제한하라. `H*P`와 `G*N`은 각각 텐서 shape에 묶였지만, 상태 `H*P*N`은 `in_proj` 원소 수에 묶기 전까지 제한되지 않았다.
- 정밀도 게이트가 실패하면 그래프를 고치기 전에 참조 구현 안에서 오차를 재현하라. HF 자체의 `nn.Linear` 입력을 f16으로 반올림해 오차를 재현했고, 그 위치가 `in_proj`임을 찾았으며, 그래프 쪽 해결책이 없음을 보였다.
- 런타임이 실제로 쓸 구성을 검사하라. 모델 검사는 처음에 `IreeLlama`가 emit하는 `InOut`이 아니라 config의 projection 레이아웃으로 돌았다. 수치는 우연히 같았지만, 그 상태로는 #3339의 경로를 검증하지 못했을 것이다.
