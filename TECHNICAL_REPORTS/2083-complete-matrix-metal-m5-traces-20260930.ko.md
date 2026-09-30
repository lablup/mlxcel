# 기술 보고서: PR #2083 - Metal M5 trace로 #1809 매트릭스 행 완성

**날짜**: 2026-09-30

**상태**: `update/issue-1809-metal-m5-rows`에 trace와 비교 결과 커밋 완료, 머지 대기 중.

**언어**: Markdown (correctness 문서, trace README), TSV logit trace와 메타데이터

**위험도**: 낮음 (trace 데이터와 문서만 바뀌며 모델, 커널, 스크립트, 런타임 코드는 바뀌지 않습니다)

## 요약

PR #2059는 이슈 #1809를 위해 ROCm gfx1151에서 체크포인트 다섯 개(두 번째 dense 모델, sliding-window 모델, SSM hybrid 두 개, VLM)를 trace했지만, ROCm 호스트에서 접근할 수 있는 Metal 호스트가 없어 `docs/benchmark_results/rocm-correctness-gfx1151-2026-09-30.md`의 모든 행이 기준(reference) 없이 남았습니다. 이 PR은 Apple M5 Max(Apple GPU generation 17, NAX 경로 사용 가능)에서 trace한 Metal 쪽을 가져와, 첫 실행(2026-09-12)의 기준으로 비교를 완성합니다. 기준은 reference의 decided position(top-2 gap이 2.0 이상인 위치)에서 불일치가 하나도 없으면 통과입니다.

- **원래의 16쌍**: 모든 행에서 decided mismatch 0. 15쌍은 통과이고, `nemotron-3-nano-30b-a3b` `w1`은 Metal reference에 decided position이 없어(`0 / 0`) 판정 불가(inconclusive)입니다. 16쌍 전체에서 top-1은 6720개 위치 중 155곳에서 다르고, decided position 2696곳에서는 0곳입니다. 불일치 지점의 reference gap 최댓값은 1.250 logit입니다(Nemotron-H `w256`, Metal 토큰이 ROCm의 2위).
- **정정된 caveat**: hybrid `w1` 행은 Metal의 fused SSM update 커널과 ROCm의 SSD graph를 비교할 것으로 예상했지만 그렇지 않았습니다. `logit_trace`는 chunk마다 새 cache를 만들고, fused 단계는 이미 존재하는 `ssm_state`가 필요하므로, `w1`은 양쪽 모두 graph 대 graph입니다. ROCm 오케스트레이터와 M5 세션이 이 사실을 각자 독립적으로 발견했습니다.
- **새 `w1ctx512` 행 두 개**는 매 단일 토큰 forward 전에 512 토큰을 prefill하므로, Metal의 fused `ssm_update_kernel`과 state가 있는 ROCm SSD graph를 처음으로 실제로 비교합니다. 양쪽 모두 코드 커밋 `3c9edea0`입니다. granite는 decided mismatch 0 / 60, 최대 gap 0.125, Nemotron-H는 0 / 71, 최대 gap 0.250이며, 둘 다 0.5와 1.0에서도 0입니다.

Metal trace는 사용자의 M5에서 실행 중인 Remote Control 세션이 만들었습니다. ROCm 오케스트레이터가 그 세션에 trace를 요청했고, 두 쪽은 `data/issue-1809-metal-traces` 브랜치로 데이터를 주고받았습니다.

## 1. 문제 정의

이슈 #1809는 Metal baseline 대비 ROCm correctness 매트릭스를 모델과 width별 decided-position mismatch로 보고하도록 요구합니다. 첫 실행(#1826, 2026-09-12)은 M1 Ultra를 기준으로 dense와 MoE 체크포인트를 다뤘습니다. sliding-window, SSM-hybrid, VLM 행이 빠져 있어 이슈가 다시 열렸습니다. #2059는 이 행들을 ROCm에서 trace했고(`benchmarks/logit_traces/rocm_gfx1151_c5fe9a16/`), 그 과정에서 ROCm 결함 하나(두 hybrid 모두에서 GPU fault를 일으킨 strided-scan launch)를 찾아 고쳤습니다. 하지만 머지 시점의 이슈 코멘트는 Metal 쪽을 미완으로 남겼습니다: "No Metal host was reachable and no Metal trace for these checkpoints is in the repository."

Metal 쪽이 없으면 ROCm trace는 체크포인트가 로드되고, 생성하고, NaN 없이 trace된다는 것만 보여줍니다. ROCm이 Metal과 같은 모델을 계산하는지는 말해주지 않습니다. 이 PR이 닫는 것이 바로 그 acceptance criterion입니다.

#2059가 Metal 호스트용으로 남긴 README는 like-for-like가 아닐 쌍 두 가지도 예측했습니다. 그중 하나는 틀렸고, 그 이유를 밝히는 과정에서 매트릭스가 SSM 커널에 대해 주장할 수 있는 범위가 바뀌었습니다(4절).

## 2. 변경 요약

커밋 다섯 개, 모두 `benchmarks/`와 `docs/` 아래입니다.

- **`bd6c5863`** (data 브랜치의 `9830bd60`을 작성자 정보를 유지해 cherry-pick): `benchmarks/logit_traces/metal_m5_d1128266/`, `d1128266`(#2059 머지 커밋)에서 만든 Metal trace 16개와 `METADATA.txt`, `RUNS.txt`, `SHA256SUMS`.
- **`556e62b2`**: 2026-09-30 문서의 모든 행을 이 trace로 채우고, threshold sweep, rank 분포, "What is not like-for-like" 절을 추가하고, `rocm_gfx1151_c5fe9a16/README.md`의 포인터와 문구를 고쳤습니다.
- **`8e650514`** (`d2043c94`에서): granite와 Nemotron-H의 Metal `w1ctx512` trace를 `d1128266`에서 만들어 `metal_m5_d1128266/`에 추가.
- **`ce6b2bd7`** (`c5706667`에서): 같은 두 행을 `3c9edea0`에서 다시 trace해 `benchmarks/logit_traces/metal_m5_3c9edea0/`에 추가.
- **`3ca19723`**: `3c9edea0`의 ROCm `w1ctx512` trace를 `benchmarks/logit_traces/rocm_gfx1151_3c9edea0/`에 추가하고, 문서에 "The fused SSM kernel against the graph with state" 절을 추가.

`docs/installation.md`와 2026-09-12 문서의 한 줄짜리 포인터는 새 행이 M5 Max 기준과 비교되었다고 고쳤습니다.

## 3. 결과

### 3.1 16쌍

`python3 scripts/compare_logit_traces.py <metal.tsv> <rocm.tsv> --decided 2.0`, Metal이 reference입니다. ROCm trace는 `c5fe9a16`, Metal은 `d1128266`에서 만들었으며, `git diff c5fe9a16 d1128266`은 `src/models/`, `examples/logit_trace.rs`, `scripts/compare_logit_traces.py`의 어떤 파일도 건드리지 않습니다.

| 모델 | Width | Top-1 불일치 | Decided mismatch | 최대 gap | Perplexity 차이 | 판정 |
|---|---|---|---|---|---|---|
| qwen2.5-7b-instruct | w1 / w8 / w256 | 3/128, 4/640, 2/512 | 0/19, 0/274, 0/204 | 0.008, 0.047, 0.008 | +0.219%, -0.030%, -0.010% | 통과 |
| gemma-3-4b-it | w1 / w8 / w256 | 9/128, 10/640, 19/512 | 0/2, 0/320, 0/204 | 0.250, 0.250, 0.375 | -0.211%, -0.326%, +0.890% | 통과 |
| gemma-3-4b-it | w8ctx1536 | 5/320 | 0/189 | 0.125 | -0.341% | 통과 |
| granite-4.0-h-tiny | w1 / w8 / w256 | 0/128, 18/640, 18/512 | 0/112, 0/260, 0/208 | 없음, 0.500, 0.500 | +4.345%, +0.302%, +0.349% | 통과 |
| nemotron-3-nano-30b-a3b | w1 | 20/128 | 0/0 | 0.688 | -0.181% | 판정 불가 |
| nemotron-3-nano-30b-a3b | w8 / w256 | 25/640, 18/512 | 0/287, 0/216 | 0.750, 1.250 | +0.248%, +0.396% | 통과 |
| qwen2.5-vl-3b-instruct | w1 / w8 / w256 | 1/128, 3/640, 0/512 | 0/49, 0/198, 0/154 | 0.000, 0.016, 없음 | -0.107%, +0.066%, +0.006% | 통과 |

불일치 155곳 중 142곳은 Metal 토큰이 ROCm의 2위, 9곳은 3위, 3곳은 4위, 1곳은 8위입니다(8위는 Nemotron-H `w1`의 undecided 위치). ROCm top-8 밖으로 나간 경우는 없습니다.

판정 불가 행은 첫 실행이 `w1` 행 두 개에서 기록한 것과 같은 `0 / 0` 경우입니다. 각 `w1` chunk는 context도 BOS도 없는 단일 토큰이고(#1785), Metal reference가 2위보다 2.0 이상 앞선 위치가 하나도 없습니다. 같은 체크포인트의 `w8`과 `w256`은 decided position 287개와 216개에서 mismatch가 없으므로, 모델 자체는 측정되었지만 단일 토큰 shape에서는 아닙니다.

### 3.2 hybrid의 threshold sweep

dense 모델, sliding-window 모델, VLM은 0.5, 1.0, 2.0 모두에서 0입니다. hybrid는 그렇지 않으므로, 이 모델들의 2.0에서의 0은 선을 어디에 그었느냐에 달려 있습니다.

| 모델 | Width | gap >= 0.5 | gap >= 1.0 | gap >= 2.0 |
|---|---|---|---|---|
| granite-4.0-h-tiny | w1 | 0 / 128 | 0 / 128 | 0 / 112 |
| granite-4.0-h-tiny | w8 | 1 / 498 | 0 / 392 | 0 / 260 |
| granite-4.0-h-tiny | w256 | 1 / 401 | 0 / 299 | 0 / 208 |
| nemotron-3-nano-30b-a3b | w1 | 3 / 36 | 0 / 2 | 0 / 0 |
| nemotron-3-nano-30b-a3b | w8 | 2 / 515 | 0 / 414 | 0 / 287 |
| nemotron-3-nano-30b-a3b | w256 | 3 / 427 | 1 / 334 | 0 / 216 |

granite는 gap 0.5 이상에서 불일치 2개, Nemotron-H는 0.5 이상에서 8개, 1.0 이상에서 1개입니다. threshold와 무관한 수치는 불일치 지점의 최대 gap인 1.250이며, `--decided`를 1.3 이상으로 두면 모두 0이라는 같은 결과가 나옵니다. 첫 실행의 최댓값은 1.125였고, 그때도 128-expert MoE에서 나왔습니다.

### 3.3 granite `w1` perplexity 변화

granite `w1`은 top-1이 모두 일치하는데 perplexity가 +4.345%입니다. 위치별로 보면 ROCm의 NLL이 평균 0.042 nats 높고 standard error는 0.017이며, 128곳 중 73곳에서 높습니다. 약 2.5 standard error로, 작은 bias가 있다는 약한 증거입니다. 이 width에서 Mamba2 layer는 양쪽 모두 같은 SSD graph를 실행하므로(4절) 커널 대 graph 효과가 아닙니다. `w8`과 `w256`에서 평균 NLL 변화는 0.003, 0.004 nats입니다.

## 4. 정정된 caveat와 `w1ctx512` 행

### 4.1 `w1`이 graph 대 graph인 이유

#2059의 README는 hybrid `w1` 행이 Metal의 fused SSM update 커널과 ROCm의 SSD graph를 비교한다고 적었습니다. `d1128266`의 코드는 다릅니다. fused 단일 토큰 단계에는 `seq_len == 1`, `ssm_kernel_available()`, 그리고 cache에 이미 있는 `ssm_state`가 필요합니다(`src/models/granitemoehybrid.rs`의 `ssm_step_kernel`, `src/models/nemotron_h.rs`의 `forward_fused`는 `conv_state`도 필요). `logit_trace`는 chunk마다 새 cache를 만들고 `w1` chunk의 `PREFILL`은 0이므로, 그 한 토큰은 항상 빈 cache를 만나고 Metal은 ROCm과 같은 `ssm_step` graph로 돌아갑니다. `w8`과 `w256` chunk는 여러 토큰이라 단일 토큰 분기를 타지 않습니다. 따라서 16쌍 중 어느 것도 Metal에서 fused 커널에 도달하지 않습니다.

ROCm 오케스트레이터는 행을 채우면서 `556e62b2`에 이를 기록했고, M5 세션은 이를 보완할 trace를 추가하면서 `d2043c94`(`8e650514`로 cherry-pick)에 기록했습니다. 두 쪽이 독립적으로 같은 결론에 도달했습니다. README와 문서는 이제 예측이 아니라 정정된 쌍을 적습니다.

### 4.2 `w1ctx512` shape

`w1ctx512` = `1 128 8 512`, `MLXCEL_TRACE_START_TOKEN=512`: 128개 chunk 각각이 앞선 corpus 토큰 512개를 새 cache에 prefill하고, 그 행은 버린 뒤, 토큰 하나를 trace합니다. 이 forward는 이미 존재하는 SSM state를 만납니다. Metal에서 granite의 Mamba2 layer는 `ssm_step_kernel`을 거쳐 `ssm_update_kernel`을 호출하고, Nemotron-H의 layer는 input projection, convolution, 같은 `ssm_update_kernel`, output projection을 합친 `fused_mamba2_forward`를 호출합니다. ROCm에서는 `ssm_kernel_available()`이 false이므로 같은 forward가 prefill된 state로 `ssm_step`, 즉 SSD graph를 실행합니다. 문서에서 이 커널과 graph를 비교하는 행은 이 둘뿐입니다.

### 4.3 커밋

양쪽 모두 코드 커밋 `3c9edea0`(`origin/main`, #2082 머지)입니다.

- Metal `metal_m5_3c9edea0/`: logit_trace sha256 `decf1fa9...`, metallib `dca1bb42...`(다른 Metal trace와 같음). 이 행들의 첫 Metal trace(`d1128266`)는 `metal_m5_d1128266/`에 남겨 두었고, 두 모델 모두 그 data row(`#`로 시작하지 않는 모든 줄)가 `3c9edea0` trace와 바이트 단위로 같으므로 결과는 어느 쪽을 기준으로 해도 같습니다. `3c9edea0` 파일은 #2071에서 `logit_trace`에 추가된 `# device` 헤더 때문에 한 줄 더 깁니다.
- ROCm `rocm_gfx1151_3c9edea0/`: logit_trace sha256 `00686315e5fbd9a4fcb9aad26a525b4d89ed60e37acf7a855edbcec2f1a66ae6`, 이 브랜치의 `8e650514`에서 빌드했으며 이 커밋은 `3c9edea0`과 `benchmarks/`, `docs/` 아래만 다릅니다.

`3c9edea0`은 나머지 16행의 `c5fe9a16` / `d1128266`보다 나중 커밋입니다. 문서는 그 사이에 머지된 ROCm 수정 각각이 SSM-with-state 경로와 어떤 관계인지 적었습니다: #2070(`SliceUpdate` donation, 경로 근처지만 경로 위는 아님), #2076(HIP header rebuild, 빌드 전용이지만 `segsum` scan 수정과 관련), #2079와 #2071(CPU stream 전용, `logit_trace`는 GPU stream에서 실행), #2073과 #2078(경로와 무관).

### 4.4 결과

| 모델 | Top-1 불일치 | Decided mismatch | 최대 gap | Logit delta p50 / p90 / p99 / max | Perplexity Metal / ROCm | 차이 |
|---|---|---|---|---|---|---|
| granite-4.0-h-tiny | 4 / 128 (3.125%) | 0 / 60 | 0.125 | 0.1250 / 0.3750 / 0.7500 / 1.0000 | 7.090 / 7.234 | +2.024% |
| nemotron-3-nano-30b-a3b | 7 / 128 (5.469%) | 0 / 71 | 0.250 | 0.1250 / 0.3750 / 0.6250 / 0.6250 | 7.489 / 7.489 | -0.003% |

0.5와 1.0에서: granite 0 / 104, 0 / 84, Nemotron-H 0 / 106, 0 / 89. 두 행의 모든 불일치는 gap 0.5 미만에서 일어나고 Metal 토큰을 ROCm의 2위에 두므로, 결과는 threshold에 의존하지 않습니다.

granite의 +2.024%는 위치별로 확인했습니다. ROCm의 NLL이 평균 0.020 nats 높고 standard error도 0.020(약 1.0 standard error)이며, 128곳 중 66곳에서 높고 58곳에서 낮고, 단일 위치 최대 차이는 2.25 nats입니다. 128개 위치로는 noise 범위입니다. Nemotron-H의 평균 변화는 0.000 nats(standard error 0.009)입니다.

이 행들이 입증하는 것: 512 토큰 state로 128번의 decode step 동안, state가 있는 ROCm SSD graph와 Metal fused SSM update 커널이 두 hybrid 모두에서 모든 decided position에서 일치합니다. 입증하지 않는 것: 이 비교는 두 백엔드에 걸친 커널 대 graph이지, 한 백엔드 안에서의 커널 대 graph가 아닙니다. Apple에는 graph 경로를 강제하는 런타임 스위치가 없어서 fused 경로 선택은 관찰이 아니라 코드에서 읽은 것이고, Metal에서 A/B 테스트할 수 없었습니다. 커널 대 커널 비교에는 #1814의 ROCm 포팅이 필요합니다.

## 5. Like-for-like가 아닌 것

- **첫 실행의 M1 Ultra가 아니라 M5 NAX 커널.** MLX pin `81ba1c6a`에서 `metal::is_nax_available()`은 macOS 26.2 이상의 generation 17 이상에서 true입니다. 이 값은 `qmm`, `gather_qmm`, dense GEMM, `gather_mm_rhs`, full self-attention SDPA의 NAX 변형을 결정하고, 단일 토큰 경로(`qmv`, `gather_qmv`, vector SDPA)에는 NAX gate가 없습니다. metallib hash는 M1 Ultra 실행과 같으므로 차이는 런타임 커널 선택입니다. Metal의 multi-token width는 NAX 커널을 탔을 가능성이 높고, `w1` width는 M1 Ultra와 같은 커널 계열을 탔을 가능성이 높습니다. 각 op가 어떤 커널을 탔는지는 기록되지 않았습니다. 따라서 첫 실행의 1.125와 이번 실행의 1.250은 서로 다른 Metal 산술을 기준으로 한 값이고, 이 체크포인트들의 M1 Ultra trace가 없어 M5의 기여를 분리할 수 없습니다.
- **Nemotron-H MoE.** Metal은 `fused_moe_forward`(`MLXCEL_FUSED_MOE_RELU2`가 설정되지 않았으므로 C++ graph 경로)를, ROCm은 `use_fused`가 `custom_kernels_available()`을 요구하기 때문에 `forward_nonfused`를 탑니다. Metal에서 `forward_nonfused`를 고르는 환경 변수가 없어 Metal 대조군이 없습니다. `w1ctx512`를 포함한 모든 Nemotron-H 쌍은 두 백엔드뿐 아니라 두 MoE 구현도 비교하며, 원래의 Nemotron-H 세 쌍이 이번 실행의 가장 큰 불일치 gap 세 개(1.250, 0.750, 0.688)를 차지합니다. granite의 MoE는 양쪽 모두 `SwitchGLU::forward`(`gather_qmm`)입니다.
- **VLM 행은 language model만 trace합니다.** `logit_trace`는 텍스트만 넣으므로 `qwen2.5-vl-3b-instruct` 세 쌍은 vision tower와 image-token merge를 전혀 거치지 않습니다. 그 경로는 문서의 이미지 생성 확인만 다루며, 그것도 ROCm 전용입니다.

## 6. 머신 간 작업 흐름

ROCm 오케스트레이터는 Metal이 없는 gfx1151 호스트에서 실행됩니다. Metal 호스트가 생길 때까지 행을 비워 두는 대신, 사용자의 M5 Max에서 실행 중인 Remote Control 세션에 Metal trace를 요청했습니다. 두 쪽은 파일시스템을 공유하지 않았고 git으로 데이터를 주고받았습니다.

1. M5 세션이 `rocm_gfx1151_c5fe9a16/README.md`의 loop로 `d1128266`에서 `logit_trace`를 빌드하고, 각 체크포인트의 모든 파일을 고정된 Hugging Face revision과 대조한 뒤(LFS 파일은 sha256, 그 외는 git blob sha1), trace 16개와 메타데이터를 `data/issue-1809-metal-traces`에 push했습니다(`9830bd60`).
2. ROCm 쪽이 그 커밋을 작성자 정보를 유지해 PR 브랜치에 cherry-pick하고 모든 비교를 실행했습니다(`556e62b2`).
3. M5 세션이 `d1128266`의 `w1ctx512` trace를 push했습니다(`d2043c94`). ROCm 쪽은 `8e650514`(코드는 `3c9edea0`과 같음)에서 바이너리를 빌드해 자기 쪽을 trace했습니다.
4. 두 쪽이 같은 커밋을 쓰도록 M5 세션이 `3c9edea0`에서 다시 trace했고(`c5706667`), ROCm 쪽이 자기 trace와 비교 결과를 커밋했습니다(`3ca19723`).

data 브랜치는 trace 데이터와 메타데이터만 담습니다. 비교에 필요한 모든 것(커밋, corpus hash, 인자, 체크포인트 revision, 바이너리와 metallib hash, 종료 상태와 행 수)이 각 디렉터리의 `METADATA.txt`와 `RUNS.txt`에 기록되어 있어, ROCm 쪽은 Mac에 접근하지 않고도 Metal 쪽을 검증할 수 있었습니다.

## 7. 기술적 선택과 그 이유

- **2026-09-12 기준을 그대로 유지.** `--decided 2.0`에서 decided position mismatch 0, `0 / 0`은 통과가 아니라 판정 불가로 보고합니다. 데이터를 본 뒤 threshold를 바꾸면 결과를 반증할 수 없게 됩니다. 대신 sweep과 threshold와 무관한 최대 gap을 함께 보고해, 2.0이 얼마나 여유를 남기는지 독자가 볼 수 있게 했습니다.
- **Nemotron-H `w1`은 통과가 아니라 판정 불가로 보고.** decided position이 없는 쌍은 어느 쪽 증거도 되지 않습니다. 증거는 같은 모델의 `w8`, `w256` 행이 제공합니다.
- **`w1`의 라벨을 바꾸지 않고 새 shape 추가.** `w1`이 graph 대 graph임을 안 뒤로는 fused 커널을 다루는 행이 없었습니다. `w1ctx512`는 기존의 `MLXCEL_TRACE_START_TOKEN` 메커니즘을 재사용하므로(`w8ctx1536`이 sliding window에 쓰는 것처럼) 코드 변경이 필요 없습니다.
- **ROCm 커밋에서 Metal을 다시 trace.** `w1ctx512`용 ROCm 바이너리는 `c5fe9a16` 이후의 ROCm 수정 여러 개가 들어간 현재 main에서 빌드했습니다. Metal을 `3c9edea0`에서 다시 trace해 두 쪽이 같은 커밋을 쓰게 했고, `d1128266` trace를 남겨 data row가 바이트 단위로 같다고 기록해 재trace가 Metal 쪽에서 아무것도 바꾸지 않았음을 보였습니다.
- **Nemotron-H loader 줄은 커밋된 trace가 아니라 복사본에서 걸러냄.** 커밋된 파일은 각 바이너리가 출력한 그대로이고 checksum도 유효하게 유지됩니다. 필터는 문서화된 비교 명령의 일부입니다.

## 8. 검증

PR 본문에 따르면 gfx1151 호스트에서: 새로 추가되거나 바뀐 trace 디렉터리 세 곳에서 `sha256sum -c SHA256SUMS`, `8e650514`에서 `cargo build --release --features rocm --example logit_trace`와 ROCm `w1ctx512` trace 두 개(exit 0, data row 128개, NaN 없음, 각각 `/sys/class/kfd/kfd/proc`에 다른 프로세스가 없을 때 시작), 비교 18개 전체, `make verify-versions verify-kernel-dtype-keys verify-kernel-port-dispatch verify-llama-compat verify-rocm-overlay verify-fmt verify-binary-assets` 통과, `cargo test --features rocm --test dead_doc_pointers` 통과(테스트 2개).

이 보고서를 쓰면서 커밋된 trace로 다시 확인한 것: `metal_m5_d1128266/`, `metal_m5_3c9edea0/`, `rocm_gfx1151_3c9edea0/`의 checksum 통과, `w1ctx512` 비교 두 개가 4.4절 표를 재현, Nemotron-H `w1` 비교가 20 / 128, decided position 없음, 최대 gap 0.688, 판정 불가를 재현, 두 모델 모두 `d1128266`과 `3c9edea0` Metal `w1ctx512`의 data row가 동일, 걸러내지 않은 Nemotron-H 비교가 `ValueError: not enough values to unpack (expected 6, got 1)`을 발생.

## 9. 학습 포인트

- **함수 이름이 아니라 경로 선택 조건을 확인할 것.** "Metal의 단일 토큰 단계"는 fused 커널처럼 들렸지만, 커널에는 state도 필요했고 harness는 state를 주지 않았습니다. 이름만 보고 쓴 caveat는 모든 hybrid `w1` 행에서 틀렸고, 모델 코드의 guard를 한 번 읽는 것으로 정정되었습니다.
- **비교 harness는 다루려던 경로를 조용히 건너뛸 수 있습니다.** chunk마다 새 cache를 만드는 것은 격리를 위해 옳지만, prefill이 0인 trace에서는 어떤 decode step도 state를 갖지 못한다는 뜻이기도 합니다. 커버리지 주장에는 그 경로에 도달하는 shape를 명시해야 합니다.
- **threshold 판정과 함께 threshold와 무관한 수치를 공개할 것.** hybrid에서는 "2.0에서 0"과 "1.0에서 1"이 모두 참입니다. 선택에 의존하지 않는 수치는 불일치 지점의 최대 gap(1.250)입니다.
- **trace 디렉터리가 스스로를 설명하게 할 것.** 각 디렉터리가 커밋, hash, 인자, 체크포인트 revision, 실행 상태를 기록하므로, 다른 머신에서 다른 세션이 만든 trace도 그 세션을 신뢰하지 않고 검증해 쓸 수 있었습니다.
- **stdout은 trace 형식의 일부입니다.** 모델 loader의 `println!` 하나가 stdout을 파싱하는 모든 소비자를 깨뜨립니다.

## 10. 주의사항, 검증하지 않은 것, 후속 작업

- **ROCm 호스트에서 Metal은 실행하지 않았습니다.** Metal trace는 `METADATA.txt`, `RUNS.txt`, checksum을 근거로 받아들였고, fused 커널 경로 선택은 코드에서 읽은 것입니다.
- **다른 M5에서 비트 단위로 재현되지 않습니다.** `examples/logit_trace`는 Apple GPU generation 15 이상에서 바이트 동일성이 성립하지 않는다고 적고 있습니다. decided-position 기준은 이에 의존하지 않습니다.
- **SSM update의 커널 대 커널 비교**에는 #1814의 ROCm 포팅이 필요합니다.
- **noise floor는 측정하지 않았습니다.** 필요 없다는 첫 실행의 근거가 여전히 적용됩니다.
- **후속: Nemotron-H loader 출력이 stdout으로 나감.** `src/models/nemotron_h.rs`가 양쪽 백엔드 모두에서 `#` 헤더 앞에 `[NemotronH] ...` 다섯 줄을 `println!`으로 출력합니다. `compare_logit_traces.py`는 이를 건너뛰지 않고 `ValueError`로 종료합니다. 이 PR의 모든 Nemotron-H 비교는 `grep -v '^\[NemotronH\] '`로 걸러낸 복사본에서 실행했습니다. loader가 stderr로 로그를 남기거나 스크립트가 trace가 아닌 줄을 건너뛰어야 합니다. 새 trace의 `# device` 헤더 줄은 메타데이터로 읽히므로 필터가 필요 없습니다.
- **후속: granite `w1` perplexity.** +4.345%, 위치별 약 2.5 standard error이며, top-1은 모두 일치하고 양쪽 모두 graph 코드입니다. 작지만 이 표본에서 noise로 무시할 수는 없습니다.

참고: #1809, PR #2059, PR #1826, #1814, #1785, #2070, #2071, #2076, #2079.
