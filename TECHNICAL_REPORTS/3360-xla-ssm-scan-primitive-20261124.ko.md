# PR #3360: HF fp32 오라클과 측정된 prefill lowering을 갖춘 OpenXLA SSM scan primitive

**날짜**: 2026-11-24
**상태**: PR 열림(#3337을 닫음, epic #564 Phase 1 단위 U1). ROCm gfx1151에서 IREE `local-task`와 `hip`으로 검증했다. `generate`에서는 아직 닿지 않는다. 모델 그래프 연결은 #3338, shim 상태 버퍼와 generate 경로는 #3339와 #3340이다.

## 요약

OpenXLA/IREE emitter에는 순환(recurrent) 혼합 primitive가 없어서 epic #564의 SSM이나 hybrid 계열을 표현할 수 없었다. 이 PR은 `emitter/ssm.rs`에 causal depthwise conv1d(prefill, tail, decode step), selective scan(한 스텝 decode와 프롬프트 전체 prefill), Mamba2 mixer 한 층 전체를 추가하고, `emitter/ssm_scan.rs`에 scan의 prefill lowering 세 가지를 추가한다. mixer 한 층을 독립 probe 그래프로 emit해 IREE auxiliary ABI로 실행하고 HF `Mamba2Mixer.torch_forward` fp32와 비교한다. 게이트 11개 케이스가 `local-task`와 `hip`에서 모두 통과하며, 최악 `y` 오차는 허용치 5e-5에 대해 1.4e-6이다.

prefill 결정은 `chunk_size = 128`의 chunked SSD다. `state-spaces/mamba2-130m` mixer 차원에서 256행과 4096행 모두 StableHLO op 157개, dispatch 31~33개를 내고, `llvm-cpu`와 `rocm` 모두 1초 안에 컴파일되며, 4096행에서 transient는 242 MiB다. associative scan은 정확하지만 4096행에서 transient가 9.2 GiB 필요하고, sequential unroll은 256행에서 dispatch가 783개이며 4096행에서는 두 타깃 모두 1200초 안에 컴파일이 끝나지 않는다.

리뷰에서 원래 게이트가 보지 못하던 케이스 두 가지(네 청크에 걸쳐 이어지는 상태, dt clamp)를 추가했고, 보안 리뷰는 #3338에서 차원 값이 신뢰할 수 없는 `config.json`에서 들어오기 전에 `SsmConfig`의 모든 곱을 checked 산술로 막았다.

## 1. 문제 정의

Mamba2와 epic #564의 모든 hybrid의 SSM 층에는 emitter에 없던 연산자 두 개가 필요하다. `xBC` projection 위의 causal depthwise conv1d, 그리고 selective scan `h_t = exp(dt_t * A) * h_{t-1} + dt_t * (x_t outer B_t)`, `y_t = C_t . h_t + D * x_t`다. 각각 한 스텝 decode 형태와 프롬프트 전체 prefill 형태가 필요하다.

decode 형태는 그대로 옮기면 된다. 실제 문제는 prefill 형태다.

- prefill 그래프는 `context_capacity`(기본 256, `MLXCEL_XLA_CONTEXT_BUCKETS`로 설정, shim의 `xla_llama_prefill`은 `lp == context_capacity`를 요구)까지 패딩되므로, scan은 패딩 행이 상태를 진행시키지 않게 하면서 그 행들 위를 지나가야 한다.
- builder에는 `while` op이 없고, Stage 2a에서 IREE-CUDA가 scatter를 거부했다. scan은 이미 lowering되는 op(`dot_general`, `pad`, slice, `dynamic_slice`, `dynamic_update_slice`, `select`, `reduce_window`)으로만 만들어야 한다.
- `T`행에 대한 순환을 단순하게 쓰면 `T`번 펼친 스텝이 되어 그래프가 capacity에 비례해 커진다. 이슈는 후보 세 가지(chunked SSD, log-depth associative scan, 전체 unroll)를 제시하고, 하나를 고르기 전에 256과 4096에서 측정하도록 요구했다.

상태가 프롬프트 전체에 걸쳐 누적되므로 그래프 정밀도가 f16이어도 순환은 f32로 유지돼야 한다. 기존 `dot_general`은 입력을 그래프 정밀도로 낮추므로, scan에는 낮추지 않는 contraction이 필요했다.

이슈는 이 작업을 스파이크와 primitive로 정의했다. 모델에 연결하기 전에 한 층을 HF와 대조해 증명하고, lowering 결정을 숫자와 함께 기록하는 것이다.

## 2. 변경 요약

| 파일 | 변경 |
|---|---|
| `emitter/ssm.rs` (신규) | `Mamba2MixerW`, `SsmLayerState`, `conv1d_causal_prefill`, `conv1d_tail`, `conv1d_causal_step`, `selective_scan_step`, `selective_scan_prefill`, `mamba2_mixer_prefill`, `mamba2_mixer_prefill_form`, `mamba2_mixer_step`, `ssm_state_types`, 안정적인 `softplus`, gated RMSNorm, 그룹에서 head로의 broadcast |
| `emitter/ssm_scan.rs` (신규) | `SsmPrefillForm { ChunkedSsd, Associative, Sequential }`, `selective_scan_prefill_form`, `exp_segsum`, scan 본체 세 가지 |
| `emitter/config.rs` | `MoeConfig` 옆에 이슈의 필드 그대로인 `SsmConfig`, `d_inner`, `in_proj_dim`, `validate`, `SSM_DEFAULT_CHUNK_SIZE = 128`, 테스트용 생성자 `tiny_mamba2` |
| `emitter/builder.rs` | `abs`, `log_plus_one`, `dot_general_keep_f32` |
| `emitter/model.rs` | `ProbeMode`, `emit_ssm_probe(c, hidden, t, mode)`, `emit_ssm_probe_with(.., form, precision)` |
| `emitter/mod.rs`, `emitter/ssm_tests.rs` (신규) | 모듈 연결, 단위 테스트 9개, ignored 테스트 `dump_ssm_probe_for_execution_check` |
| `ssm_probe.rs` (신규), `lib.rs`, `cache_dir.rs` | `SsmProbeSpec`(가중치 배치, 입출력 shape, emit), auxiliary IREE ABI 위의 `run_ssm_probe`, 경계 검사를 하는 `parse_npy_f32` / `npy_f32_bytes`, 테스트 4개, `XlaCache::SsmProbe`(`ssm-probe`) |
| `examples/xla_ssm_probe.rs`, `Cargo.toml` | `local-task` 또는 `hip`용 probe 실행기, `required-features = ["xla-iree"]` |
| `spike/openxla/ssm_oracle.py`, `ssm_gate.sh`, `ssm_lowering_measure.py` | HF fp32 오라클(입력 생성, 결과 검사), 11개 케이스 게이트, lowering 측정 |
| `spike/openxla/FINDINGS_ssm.md`, `README.md` | 측정과 결정, x86_64 오라클 venv와 probe 명령 |
| `CHANGELOG.md`, `scripts/ci/check_env_var_docs.py` | Added 항목, 테스트 전용 dump 변수 `MLXCEL_SSM_MODE`와 `MLXCEL_SSM_T`를 허용 목록에 추가 |

파일 19개, +2841 / -5.

### Mixer 층

`mamba2_mixer_prefill_form`은 HF `torch_forward`를 단계별로 따른다.

1. `in_projection`이 `zxbcdt`를 만들고 `[z: d_inner, xBC: conv_dim, dt: H]`로 자른다.
2. 이어받은 conv tail `conv0 [K-1, C]`를 `xBC` 앞에 붙이고, 그 왼쪽 문맥 위에서 depthwise conv를 돌린다(`feature_group_count = conv_dim`인 `stablehlo.convolution`, torch `[C,1,K]` 커널을 `[K,1,C]`로 transpose). 새 tail은 `xp`에서 `real_len` 위치로 `K-1`행을 `dynamic_slice`한 것이다. `conv0`를 왼쪽 문맥으로 쓰기 때문에 이어지는 prefill이 정확하고, `real_len = 0`이면 `conv0`가 그대로 돌아온다.
3. `silu` 후 `split_xbc`로 `x [T,H,P]`, `B`, `C [T,G,N]`로 나눈다.
4. `discretize_dt`: `softplus(dt + dt_bias)`를 `max(x,0) + log1p(exp(-|x|))`로 계산해 모든 입력에서 유한하게 만들고, `time_step_limit` clamp를 적용한다(각 쪽은 실제로 걸릴 때만 emit).
5. 패딩 마스크: `[T, H]`에서 `dt = select(iota < real_len, dt, 0)`. `dt = 0`이면 decay는 `exp(0) = 1`, 입력 항은 `0 * x`이므로, 어떤 scan 형태에서도 패딩 행은 상태를 그대로 통과시킨다. 마스크는 clamp 뒤에 적용되므로 0이 아닌 dt 하한이 패딩 행으로 새어 들어가지 않는다.
6. `A = -exp(A_log)`, scan, `D` skip, 그다음 `norm_group_size` 단위 그룹마다 gated RMSNorm `rmsnorm(y * silu(z)) * w`, 마지막으로 `out_proj`.

`mamba2_mixer_step`은 `[B, ..]` 위에서 같은 파이프라인을 `conv1d_causal_step`(이어받은 tail과 새 행으로 이뤄진 창 위의 f32 곱-합)과 `selective_scan_step`으로 수행한다. 모든 step 형태는 명시적 batch 축을 가지므로 ragged decode가 재사용할 수 있다.

### Chunked SSD prefill

`chunked_ssd`는 `T`를 `nc * Q`행으로 패딩하고, unroll 없이 `dot_general_keep_f32` contraction 다섯 번으로 다음을 계산한다.

1. 청크 내부: 청크마다 `y_diag = (C B^T o L)(x dt)`, 여기서 `L = exp(segsum(dt A))`는 `[nc, H, Q, Q]`다.
2. 0에서 시작했을 때 각 청크 자신의 끝 상태: 청크 끝까지 decay된 `x dt`를 `B`와 contraction.
3. 청크 사이 carry: 앞에 `h0`를 붙인 청크 상태 `[nc+1, H, P*N]`를, 청크별 총 decay로 만든 청크 수준 `exp(segsum)` `[H, nc+1, nc+1]`과 contraction한다. `z`행이 청크 `z`로 들어가는 상태이고 마지막 행이 `h_T`다.
4. 대각선 밖 출력: 각 청크로 들어가는 상태와 `C`를 contraction하고, 청크 시작에서 각 행까지 decay를 곱한다.

`exp_segsum`은 HF의 `segment_sum`을 따른다. 스텝별 log decay를 대각선 아래만 남기도록 마스크하고, 마지막 축을 따라 `reduce_window_prefix_add`를 돌린 뒤, `exp` 전에 대각선 위를 `-inf`로 채운다. prefix sum 두 개의 차를 만들지 않으므로 대각선 근처 값에 상쇄 오차가 없고, 양수를 지수화하는 일도 없다.

그룹 공유(`G < H`)는 broadcast와 reshape(`heads_from_groups`)로 처리하며, HF의 `repeat_interleave`와 같다. head `h`는 그룹 `h / (H / G)`를 읽는다.

### f16 경로에서의 f32

`selective_scan_prefill_form`은 들어오는 scan 입력을 모두 f32로 바꾸고, scan 안의 모든 contraction은 `Builder::dot_general_keep_f32`를 거친다. 이 함수는 피연산자가 f32인지 assert하고 builder 정밀도와 상관없이 f32 결과를 낸다. 그래프 정밀도를 따르는 것은 in_proj, out_proj, prefill conv뿐이다.

### Probe와 오라클

`emit_ssm_probe`는 mixer 주위에 norm이나 residual 없이 `main(mixer weights..., x, [real_len,] h0, conv0) -> (y, h_T, conv_tail)`을 emit한다. 가중치 순서는 `Mamba2MixerW` 필드 순서이고, 선택적 bias는 config가 켤 때만 나타난다. `run_ssm_probe`는 이를 `ssm-probe` 캐시로 컴파일해 auxiliary ABI로 실행한다. `ssm_oracle.py`는 seed가 고정된 가중치로 HF `Mamba2Mixer`(transformers 5.12.1, slow path, `mamba_ssm` 미사용) 하나를 만들고, 입력과 기대 `y`, `h`, conv tail을 `.npy`로 쓴 뒤, 결과 디렉터리를 max-abs 오차로 검사한다. HF prefill은 0 상태에서만 시작할 수 있으므로, 이어받은 상태에서 시작하는 prefill 케이스의 기준값은 같은 `h0`와 tail에서 HF 자신의 decode 분기를 토큰 단위로 돌린 결과다.

## 3. 기술적 선택과 그 이유

**associative scan과 unroll 대신 chunked SSD.** 세 형태 모두 같은 probe 인터페이스와 scan 바깥의 같은 mixer를 공유하므로, 비교는 scan만 분리해서 본다. mamba2-130m mixer 차원(`hidden 768`, `d_inner 1536`, `H 24`, `P 64`, `N 128`, `G 1`, `K 4`), `real_len = capacity - capacity/8`, 패딩 행은 0이 아닌 난수로 측정했다.

| 대안 | 4096행에서 | 판단 |
|---|---|---|
| Chunked SSD | op 157개, dispatch 33개, 컴파일 0.6~0.8초, transient 242 MiB(Q=128) | **선택** |
| Associative(`(exp(dt A), dt x outer B)` 위의 Hillis-Steele) | op 213개, dispatch 37개, transient 9385.9 MiB, 실행 3.3초 / 1.1초 | 정확하지만 레벨마다 `[T, H, P, N]`이 생겨 31 GiB에 실제 모델이 들어가지 않음 |
| Sequential unroll | op 102476개, 두 타깃 모두 1200초 안에 컴파일 안 됨 | 기각. 256행에서 이미 op 6476개, dispatch 783개 |

**청크 사이 carry는 이슈가 제안한 펼친 순서가 아니라 `dot_general` 하나다.** 이슈는 `T / Q`개의 청크 상태 갱신을 짧게 펼친 순서로 제안했다. HF의 형태는 대신 청크 수준 decay 행렬을 만들고, 쌓은 청크 상태를 한 번에 contraction한다. 수학은 같고, op 수가 `T`와 무관하게 유지된다(256과 4096 모두 157개). 대가는 `O((T/Q)^2)` 저장 공간과 `exp_segsum` 안의 `reduce_window_prefix_add` 작업이며, 6절에서 그 범위를 다룬다.

**`chunk_size = 128`.** Q=64, 128, 256은 오차와 op 수가 같다. Q=128이 4096행에서 transient가 가장 작고(242 MiB. Q=64는 `[H, nc+1, nc+1]` carry와 청크별 상태가 커져 315, Q=256은 `[nc, H, Q, Q]` decay 행렬이 커져 362), 실행 시간도 가장 좋거나 거의 가장 좋다.

**f16 경로에서도 순환은 f32로 유지.** `MLXCEL_XLA_PRECISION=f16`에서 chunked Q=128의 HF fp32 대비 오차는 `y` 기준 256행 2.0e-3 / 1.9e-3, 4096행 2.2e-3 / 2.1e-3(local-task / hip), `h` 최대 6.0e-5, conv tail 최대 7.5e-4다. `y` 오차가 256행에서 4096행으로 가도 커지지 않으므로 f32 순환은 누적 drift를 더하지 않으며, conv tail 오차는 f16 in_proj 반올림만으로 설명된다. 정밀도를 낮춘 scan 변형은 측정하지 않았다. 설계가 f32 순환을 요구하고, 측정 결과에도 이를 버릴 근거가 없다. `dot_general`을 바꾸지 않고 별도의 `dot_general_keep_f32`를 추가했으므로 dense 그래프와 그 byte-exact golden은 그대로다.

**prefill conv의 왼쪽 문맥으로 `conv0` 사용.** 이슈는 `conv1d_causal_prefill`과 `conv1d_tail`을 0 패딩된 새 시퀀스 계약으로 정했고, 두 함수는 그 계약을 지킨다. mixer prefill은 대신 이어받은 tail을 앞에 붙이므로, 상태에서 이어지는 prefill이 정확하다. 게이트의 "이어받은 상태에서 real_len 77 / 200" 케이스가 이를 검증한다.

**계약 변경은 추가만.** `SsmConfig` 필드와 `ssm.rs` 시그니처는 #3337에 적힌 그대로다. 추가된 것은 `SsmConfig::{d_inner, in_proj_dim, validate}`(tail이 `K-1`행이므로 `conv_kernel >= 2`도 요구), `SSM_DEFAULT_CHUNK_SIZE`, `ssm_state_types`, `mamba2_mixer_prefill_form`, 그리고 `SsmConfig`에 모델 폭이 없어서 `emit_ssm_probe`에 붙은 `hidden` 인자다.

## 4. 리뷰 결과

**구현 리뷰**(2283ee0b). chunked SSD를 `G > 1`을 포함해 HF `torch_forward`와 항 단위로 대조했다. discretization, 0 이하 합만으로 만드는 segsum, 그룹 매핑, `D` skip, gated norm 순서, 청크 수준 carry 모두 수학적 결함은 없었지만, mutation 테스트에서 게이트의 빈틈 두 개가 드러났다.

- 0이 아닌 상태를 두 청크 넘게 이어가는 케이스가 없었다. `h0`가 0이고 청크가 최대 둘이면 각 청크로 들어가는 상태에 decay가 필요 없으므로, 청크 수준 decay 행렬을 0으로 만들어도 모든 prefill 케이스에서 `y`가 1e-6 안에 머물렀고 `h_T`로만 잡혔다. 새로 추가한 4청크 이어받기 케이스(chunk 64, real_len 200)는 이 mutation에서 `y`가 1.05만큼 어긋나 실패한다.
- HF의 기본 `time_step_limit`이 `(0, inf)`여서 clamp가 실행된 적이 없었다. 오라클에 `--time-step-limit`을 추가하고 prefill과 decode clamp 케이스를 넣었다. clamp를 제거하면 `y`가 1.7e-2 / 1.9e-2만큼 어긋나 실패한다.

잘못된 그룹 매핑과 segsum 마스크의 off-by-one은 원래 게이트에서도 실패했다.

**보안과 성능 리뷰**(cf1bfc0b). CRITICAL이나 HIGH는 없다. IREE shim 경계는 이미 모든 버퍼 길이를 checked 산술로 shape와 대조하고 있었다.

| 발견 | 심각도 | 상태 |
|---|---|---|
| `SsmConfig::validate`가 검사 없는 곱을 써서, release에서 wrap된 `H*P`가 `conv_dim` 등식을 만족할 수 있었음 | MEDIUM | 수정: `d_inner`, `conv_dim`, `in_proj_dim`, `H*P*N`, `(K-1)*conv_dim`을 checked로 계산, `+inf` dt 하한, `-inf` 상한, NaN 한계, 음수이거나 유한하지 않은 `norm_eps` 거부 |
| 행이 0이면 chunked scan에서 `t.div_ceil(0)`에 도달 | LOW | 수정: probe spec이 행 0, hidden 0, i32를 넘는 행을 거부하고 emitter가 assert |
| example의 `.npy` 리더가 잘린 헤더에서 panic할 수 있고 선언된 shape를 무시 | LOW | 수정: 경계 검사와 정확한 데이터 길이를 확인하는 `parse_npy_f32`를 라이브러리로 옮김, 각 입력의 shape가 probe와 일치해야 함 |
| README와 측정 스크립트의 고정 `/tmp/ssm*` 경로 | LOW | 수정: `mktemp -d` |
| `exp_segsum`과 carry의 긴 문맥 비용 | MEDIUM(성능) | #3338로 미룸, 6절 참조 |

**FINDINGS에 기록된 남은 LOW.** f16 경로에서 prefill conv는 `Builder::convolution`을 쓰므로 f32 cast 전 결과가 f16이고, decode conv는 f32 곱-합이다. `convolution_f32_accumulate`(f16 피연산자, f32 결과)는 컴파일되지 않는다. iree-compile이 depthwise 혼합 타입 convolution을 `tensor.collapse_shape` 타입 불일치로 거부한다. #3338의 f16 그래프에서 이 차이를 없애야 한다면, convolution op 대신 f32로 K-tap shift 곱-합을 쓰면 된다.

## 5. 테스트와 검증

새 단위 테스트는 `emitter/ssm_tests.rs`에 9개(config 검증, overflow와 유한하지 않은 값 거부, 행 0 거부, prefill과 decode probe 시그니처, 모든 prefill 형태가 같은 인터페이스를 emit, ragged와 짧은 capacity, f16 경로에서 scan contraction이 f32로 유지, conv tail과 step의 shape), `ssm_probe.rs`에 4개다. `cargo test --profile test-fast -p mlxcel-xla --lib`는 최종 head에서 479개 통과했고 dense golden은 바뀌지 않았다.

게이트(`spike/openxla/ssm_gate.sh`)는 capacity 256에서, 그룹 broadcast가 실행되도록 `G = 2 < H = 4`인 작은 mixer로 돌린다. HF fp32 대비 max-abs 오차, 허용치 5e-5:

| 케이스 | y local-task | y hip | h(최악) | conv tail(최악) |
|------|-------------:|------:|----------:|------------------:|
| decode, B=1 | 3.0e-7 | 4.8e-7 | 3.0e-8 | 3.0e-7 |
| decode, B=3, linear bias | 4.8e-7 | 4.8e-7 | 1.2e-7 | 2.4e-7 |
| prefill, real_len 100 < 256 | 7.2e-7 | 7.2e-7 | 3.3e-9 | 1.8e-7 |
| prefill, real_len = 256 | 7.2e-7 | 8.3e-7 | 4.2e-9 | 1.8e-7 |
| prefill, real_len 2 < K-1 | 3.0e-7 | 6.0e-7 | 4.1e-10 | 3.0e-7 |
| prefill, real_len 0 | 0 | 0 | 0 | 0 |
| prefill, 이어받은 상태에서 real_len 0 | 0 | 0 | 0 | 0 |
| prefill, 이어받은 상태에서 real_len 77, bias | 9.5e-7 | 8.3e-7 | 1.2e-6 | 3.6e-7 |
| prefill, 이어받은 상태에서 real_len 200, chunk 64(4청크) | 1.3e-6 | 1.1e-6 | 6.9e-7 | 3.6e-7 |
| prefill, real_len 100, dt를 [0.01, 0.05]로 clamp | 6.3e-7 | 7.7e-7 | 2.9e-8 | 1.8e-7 |
| decode, B=2, dt를 [0.01, 0.05]로 clamp | 4.8e-7 | 6.0e-7 | 6.0e-8 | 2.4e-7 |

표는 `FINDINGS_ssm.md`에서 가져왔고, PR의 테스트 계획은 최종 head에서 최악 `y`를 1.4e-6으로 보고한다. 리뷰에서 추가한 각 케이스는 해당 mutation에서 실패한다.

lowering 측정(`ssm_lowering_measure.py`)은 mamba2-130m mixer 차원, gfx1151, 툴체인 `iree-rocm-3.12.0rc20260721`에서 했다. 컴파일 열은 `초 / peak RSS MiB`, 오차 열은 max-abs `y`이며, `h`와 conv tail 오차는 측정한 모든 행에서 각각 3e-7, 2.2e-6 미만이다. 실행 시간은 첫 호출을 포함한 한 번의 호출이고 warmup이나 반복은 없다.

context_capacity 256:

| 형태 | ops | dispatches | llvm-cpu 컴파일 | rocm 컴파일 | transient MiB | 오차 local-task | 오차 hip | 실행 local-task / hip (s) |
|------|----:|-----------:|-----------------:|-------------:|--------------:|---------------:|--------:|-------------------------:|
| chunked Q=64 | 157 | 33 | 0.6 / 174 | 0.8 / 210 | 21.8 | 5.2e-6 | 6.0e-6 | 0.012 / 0.008 |
| chunked Q=128 | 157 | 33 | 0.6 / 174 | 0.8 / 211 | 15.9 | 5.2e-6 | 6.0e-6 | 0.012 / 0.009 |
| chunked Q=256 | 157 | 31 | 0.6 / 175 | 0.8 / 214 | 26.4 | 5.2e-6 | 6.0e-6 | 0.015 / 0.009 |
| associative | 173 | 29 | 0.6 / 167 | 0.6 / 201 | 587.4 | 5.0e-6 | 6.0e-6 | 0.126 / 0.053 |
| sequential | 6476 | 783 | 11.8 / 248 | 12.4 / 307 | 11.6 | 5.2e-6 | 6.0e-6 | 0.026 / 0.010 |

context_capacity 4096:

| 형태 | ops | dispatches | llvm-cpu 컴파일 | rocm 컴파일 | transient MiB | 오차 local-task | 오차 hip | 실행 local-task / hip (s) |
|------|----:|-----------:|-----------------:|-------------:|--------------:|---------------:|--------:|-------------------------:|
| chunked Q=64 | 157 | 33 | 0.8 / 176 | 0.8 / 208 | 315.1 | 6.7e-6 | 6.0e-6 | 0.129 / 0.085 |
| chunked Q=128 | 157 | 33 | 0.6 / 174 | 0.8 / 209 | 241.9 | 6.7e-6 | 6.0e-6 | 0.125 / 0.088 |
| chunked Q=256 | 157 | 33 | 0.6 / 173 | 0.8 / 212 | 362.3 | 6.7e-6 | 6.0e-6 | 0.151 / 0.097 |
| associative | 213 | 37 | 0.6 / 167 | 0.6 / 201 | 9385.9 | 6.7e-6 | 6.0e-6 | 3.292 / 1.146 |
| sequential | 102476 | n/a | 1200초 초과 timeout / 종료 시 약 0.8 GiB | 1200초 초과 timeout / 종료 시 약 0.8 GiB | n/a | 실행 안 함 | 실행 안 함 | n/a |

이 측정의 HF 기준값은 4096행 기준값이 메모리에 들어가도록 `chunk_size=32`를 쓴다. HF 청크 크기는 수학을 바꾸지 않는다.

게이트: `cargo fmt --all -- --check`, mlxcel-xla에 대한 clippy `-D warnings`(`--all-targets`, 그리고 `--features iree`로 lib와 테스트), `xla_ssm_probe` example에 대한 clippy.

## 6. 남은 작업과 한계

- **아직 연결되지 않음.** mixer를 호출하는 모델 그래프는 아직 없다. `config.json` 파싱, 가중치 spec, 모델 그래프는 #3338, shim 상태 버퍼와 generate 경로는 #3339와 #3340이다. CHANGELOG 항목에도 그렇게 적혀 있다.
- **chunked SSD의 긴 문맥 비용.** `reduce_window_prefix_add`가 전체 폭 창으로 lowering되므로 `exp_segsum`은 행마다 `O(L^2)`이다. 청크 내부에서는 덧셈이 `T*H*Q^2`번으로, Q=128, P=64에서 `y_diag` contraction의 약 2배다. 청크 사이 carry는 `[H, nc+1, nc+1]` 텐서 위에서 덧셈 `H*(T/Q)^3`번이다. 4096(nc = 32)에서는 무시할 만하지만 T = 185k 근처에서 청크 내부 작업을 넘어서고, T = 1M에서는 중간값 하나에 약 6 GiB가 필요하다. #3338이 긴 문맥을 목표로 한다면 carry에 범위가 정해진 형태(청크 총합에 대한 prefix sum, 또는 청크 단위 scan)가 필요하며, hip에서 HF와 대조해 측정해야 한다.
- **f16 경로에서 prefill과 decode conv의 반올림이 다름**(4절). 필요하면 convolution op 대신 f32 K-tap 곱-합으로 고친다.
- **#3338을 위한 입력 경계.** 모든 config 차원을 실제 상한인 체크포인트 텐서 shape와 대조하고, host에서 `real_len`을 `[0, capacity]`로 clamp해야 한다.
- **`local-task`와 `hip`(gfx1151)만 실행했다.** CUDA는 측정하지 않았다.
- **오라클은 HF slow path다.** `mamba_ssm`이 아니다. 실제 CUDA 커널은 이와 5e-5 넘게 다를 수 있으며, 게이트는 slow path를 기준으로 정의된다.
- **실행 시간은 참고용이다.** 첫 호출을 포함한 한 번의 호출이다.

## 7. 학습 포인트

- `while`을 쓸 수 없는 scan도 unroll을 피할 수 있다. Mamba2의 state-space duality는 순환을 청크 내부의 마스크된 행렬곱과 청크 사이의 행렬곱 하나로 바꾼다. 그러면 그래프 크기가 capacity가 아니라 shape에만 의존하고, 그래서 4096행에서도 op이 157개로 유지됐다.
- decay 행렬은 `exp(cumsum_i - cumsum_j)`가 아니라, 대각선 위를 `-inf`로 채운 마스크된 prefix sum(HF의 `segment_sum`)으로 만든다. 차 형태는 대각선 근처에서 심한 상쇄가 생기고 큰 양수를 지수화할 수 있다.
- 순환을 통과하는 패딩은 discretization 단계에서 처리하는 것이 가장 싸다. `dt = 0`이면 decay가 1, 입력이 0이 되므로 모든 scan 형태가 별도 마스크 없이 상태를 그대로 통과시킨다. 0이 아닌 하한을 가진 clamp가 있다면 마스크는 그 뒤에 둔다.
- 출력만 검사하는 게이트는 항 하나를 통째로 놓칠 수 있다. 0 초기 상태와 두 청크 때문에 청크 수준 decay가 `y`에서 보이지 않았고, HF의 기본 `(0, inf)` 한계 때문에 clamp는 죽은 코드였다. 항마다 mutation을 넣어 게이트에 돌려 보니 둘 다 드러났고, 리뷰에서 추가한 각 케이스는 이제 해당 mutation에서 실패한다.
- 기준 구현이 어떤 케이스를 표현할 수 없을 때(HF prefill은 항상 0에서 시작), 그 구현 자신의 한 스텝 경로를 토큰 단위로 돌린 결과가 이어지는 케이스의 유효한 기준값이 된다.
- 신뢰할 수 없는 파일에서 들어올 config 차원은 처음 쓰이기 전에 checked 산술로 검증해야 한다. wrap된 곱이 원래 걸러 내려던 일관성 등식을 만족해 버릴 수 있기 때문이다.
