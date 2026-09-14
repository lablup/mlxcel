# 기술 리포트: PR #1888 - refactor(moe): migrate qwen3_moe onto the shared SwitchGLU

**작성일**: 2026-09-14
**상태**: 완료, 머지 대기
**언어**: Rust, Markdown
**위험도**: 중간

## 요약

`qwen3_moe`는 `SwitchLinear`, `SwitchGLU`, `forward_fused_kernel`의 사본을 따로 갖고 있었고, `qwen3_vl_moe`(Qwen3-Omni-MoE thinker도 이것으로 만든다)는 그 사본을 직접 조립해 썼다. 공유 `switch_layers::SwitchGLU`가 #311과 #643에서 얻은 fused kernel `dff` 상한이 사본에는 반영되지 않아, 두 계열 모두에서 `MLXCEL_FUSED_MOE_MAX_DFF`가 조용히 무시되었다. PR #1888은 사본을 지우고 두 계열을 공유 타입으로 옮긴다. GB10에서 `qwen3-30b-a3b-4bit`와 `qwen3-vl-30b-a3b-4bit`의 greedy token id는 fused 경로와 `MLXCEL_FUSED_MOE=0` 모두에서 변경 전후가 같고, decode trace는 여전히 `path=fused tokens=1`을 보이며, 상한 512를 주면 두 계열이 이제 `gather_qmm`으로 내려간다(기존 바이너리는 이를 무시했다).

## 1. 문제 정의

### 1.1 배경

단일 토큰 decode용 fused kernel(#268), 공유 `forward_fused_kernel`, 그리고 Qwen3-MoE 사본은 `7fd4dcac`(#275)에서 함께 들어왔다. 이후 공유 함수만 `d846fd4f`(#311)에서 `dff` 상한을 얻었고, `37931d7f`(#643)에서 backend별 기본값(Metal 4096, CUDA 8192, `MLXCEL_FUSED_MOE_MAX_DFF`로 재정의 가능)이 정해졌다. 상한을 넘는 expert에서는 `gather_qmm`이 이미 GPU를 포화시켜 두 번 launch하는 fused 경로가 측정상 손해다(M1 Ultra에서 Dff 6400인 phi-3.5-moe가 느려진다).

### 1.2 기존 문제점

- **상한 누락**: 사본의 `forward_fused_kernel`에는 `dff` 검사가 없어, 상한보다 넓은 expert를 가진 Qwen3-MoE 또는 Qwen3-VL-MoE 변형이 손해 구간에서 fused 경로를 탔고, 환경 변수 재정의도 두 계열에 효과가 없었다.
- **잘못된 문서**: `switch_layers.rs`는 상한이 Qwen3-MoE에도 적용된다고 적고, Qwen3Moe를 공유 타입 사용처로 나열했다.
- **발동할 수 없는 가드**: 사본에는 `mode` 필드와 activation 검사가 없었다. 사본의 `gather_qmm` 호출은 `"affine"`을 하드코딩하고 loader가 `.biases`를 요구하므로 non-affine checkpoint는 애초에 로드되지 않아 영향이 없었다.

### 1.3 위험성

| 위험 | 영향 | 가능성 |
|------|------|--------|
| 넓은 expert를 가진 Qwen3-MoE 변형이 fused 경로에서 느려짐 | 중간 | 낮음(로컬 checkpoint는 모두 상한 아래) |
| 공유 가드가 바뀔 때마다 사본이 다시 어긋남 | 중간 | 높음(이미 한 번 발생) |
| 마이그레이션이 기존 checkpoint에서 fused 경로를 조용히 잃음 | 높음 | 낮음, 부록 B의 trace로 배제 |

## 2. 기술적 검토 사항

### 2.1 기존 checkpoint에서의 동등성

두 구현은 같은 연산을 수행한다. routed slot 64개 기준의 `do_sort`, 같은 `gather_sort`와 `scatter_unsort`, `compiled_swiglu_activation`, 같은 `gather_qmm` 인자(`transpose=true`, mode `"affine"`)를 쓴다. 둘 다 `fused_moe_expert_kernel`을 같은 순서의 인자 18개로 호출한다. 공유 타입은 decline 조건 네 개(activation, mode, biases 누락, `dff`)를 더하지만, Dff 768 또는 2560의 affine 4-bit plane에서는 어느 것도 발동하지 않는다.

loader 동작은 두 가지가 바뀐다.

- **저장되는 bit 폭**: 공유 loader는 tensor shape에서 추론한 bit(`packed_in * 32 / (num_groups * group_size)`)를 저장하고, 사본은 설정값을 저장했다. 로컬 checkpoint 네 개의 모든 expert plane을 safetensors header로 확인했다. `qwen3-30b-a3b-4bit`, `qwen3-moe-4bit`, `qwen3-vl-30b-a3b-4bit`는 각 144개, `qwen3-coder-480b-a35b-instruct-4bit`는 186개이며, 모두 4 bit로 풀리고 `.biases` shape이 `.scales`와 같아 저장되는 값이 동일하다.
- **biases 누락**: `.biases`가 없는 affine plane은 이제 `Weight not found: ...biases` 대신 `validate_quantization_biases` 메시지로 실패한다. biases가 없는 로컬 checkpoint는 없다.

### 2.2 보안 관점

사본이 가졌던 load 시점 보호(선언된 쌍에 대한 `validate_expert_quantization_params`)는 `SwitchLinear::from_stacked_parts`가 그대로 유지하며, 여기에 packing과 tensor shape, biases와 scales shape, mode와 biases의 교차 검사가 더해진다. 추가된 거부는 모두 원래라면 `gather_qmm` 안에서 abort했을 load만 막는다. 그 C++ throw는 cxx bridge를 넘으며 잡을 수 없는 abort가 된다.

### 2.3 성능 관점

`qwen3-30b-a3b-4bit`의 decode tok/s(`mlxcel-bench-decode -n 128 --ignore-eos --warmup-tokens 20`), arm당 15회, base와 branch를 번갈아 실행하고 round마다 순서를 바꿨다.

| Arm | n | 최소 | 중앙값 | 최대 | 평균 |
|-----|---|------|--------|------|------|
| base `b8d10fb1` | 15 | 66.21 | 78.00 | 87.76 | 77.70 |
| branch | 15 | 61.27 | 73.63 | 86.28 | 73.85 |

두 범위는 거의 완전히 겹친다. GB10의 단일 실행 decode는 bimodal이고 mode 분포가 달랐다(base 고속 11회·저속 4회, branch 고속 9회·저속 6회). 고속 mode 안의 평균은 81.06과 79.65다. Mann-Whitney 검정은 p = 0.21이고, round별 짝 차이는 평균 -3.85 tok/s, 표준편차 10.21(t = -1.46)이다. 구분 가능한 변화는 없다. 호출마다 추가되는 작업은 공유 상한의 `std::env::var` 조회와 `metal_is_available()` 질의뿐이며, 토큰당 48회로 13 ms 토큰 대비 약 10 µs 수준이다.

### 2.4 호환성/의존성 관점

- **Breaking change**: `qwen3_moe::SwitchLinear`와 `qwen3_moe::SwitchGLU`를 더 이상 export하지 않는다. tree 안의 사용처는 `qwen3_vl_moe`와 계열 테스트뿐이었고 모두 옮겼다.
- **새로 로드되는 레이아웃**: 공유 loader는 stack되지 않은 `experts.{idx}` tensor를 쌓는 fallback이 있어, 전에는 `Weight not found`로 실패하던 원본 Hugging Face Qwen3-MoE checkpoint가 이제 그 경로에 도달한다. Qwen2-MoE는 이미 그랬다.
- **새 의존성**: 없음.

## 3. 기술적 선택과 그 이유

### 3.1 사본 수정 대신 타입 이전

| 선택지 | 장점 | 단점 |
|--------|------|------|
| 사본에 `dff` 검사 추가 | 한 줄 수정 | 애초에 어긋난 중복을 그대로 둠 |
| **선택: 두 계열을 `switch_layers::SwitchGLU`로 이전** | 공유 kernel을 쓰는 모든 계열이 가드 한 벌을 공유, 모델 코드 약 330줄 제거 | 2.1의 loader 동작 변화 두 가지 |

`from_weights_with_mode`가 아니라 `SwitchGLU::from_weights`(affine)를 쓴다. non-affine Qwen3-MoE 로딩은 범위 밖이고, 들어오게 되면 공유 가드가 처음부터 적용된다.

### 3.2 가드 테스트는 계열 block loader를 통과

#958 가드 테스트는 사본의 loader를 직접 호출했다. 계열 전용 loader가 없어졌으므로 각 계열의 `SparseMoeBlock::from_weights`를 호출하게 했다. 이렇게 하면 공유 loader가 상한을 갖는다는 점과, 계열이 자기 config 타입(`ModelArgs`, 또는 accessor가 0으로 떨어지는 `Qwen3VLMoeConfig`)의 값을 그 loader에 넘긴다는 점을 함께 고정한다.

## 4. 구현 상세

- `src/models/qwen3_moe.rs`: 로컬 타입과 그 loader, `validate_expert_quantization_params` import를 제거했다. `SparseMoeBlock::experts`는 공유 `SwitchGLU`이며 `SwitchGLU::from_weights(weights, "{prefix}.switch_mlp", group_size, bits)`로 만든다. `forward`, `forward_profiled`, batched 호출부는 바뀌지 않았다.
- `src/models/qwen3_vl_moe.rs`: `load_switch_glu`, `load_switch_linear`, 더는 쓰이지 않는 `get_weight_copy`를 제거하고 block loader가 공유 생성자를 호출한다.
- `src/models/switch_layers.rs`: 실제 affine 4-bit SwiGLU plane을 양자화하는 테스트 helper `insert_honest_affine_swiglu_experts`와 mxfp4 decline 테스트를 추가했다. 문서는 공유 fused kernel 호출 계열과 아직 상한을 우회하는 경로를 명시하고, 계열 수를 바로잡고, greedy 일치 주장에 조건을 달았다.
- 문서: `docs/environment-variables.md`와 `docs/benchmark_results/fused-moe-decode-kernel-design.md`에 두 계열이 상한을 읽게 되었음을 기록했다.

## 5. 학습 포인트

### 5.1 id 비교만으로는 dispatch를 증명할 수 없다

변경 전후 greedy id가 같다는 결과는, 두 경로가 그 prompt에서 일치하는 한 마이그레이션이 조용히 `gather_qmm`으로 떨어졌어도 똑같이 나온다. 이 틈은 서로 독립적인 두 신호로 막았다. `MLXCEL_PROFILE_QWEN3_MOE_DETAIL` trace는 호출마다 경로를 기록하며, 두 바이너리 모두 decode MoE 호출 144개가 전부 `path=fused tokens=1`이었다. 또 checkpoint의 Dff보다 낮은 상한(`MLXCEL_FUSED_MOE_MAX_DFF=512`)은 두 구현을 직접 가른다. base trace는 fused 144회를 유지하고 branch trace는 `gather_qmm` 144회를 보인다.

`qwen3_vl_moe`에는 trace가 없어, fused CUDA kernel이 프로세스에서 처음 launch될 때의 비용(첫 MoE 호출에서 약 1.3초)을 신호로 썼다. 64토큰 기본 실행은 두 바이너리 모두 2.26~2.55초, `MLXCEL_FUSED_MOE=0`은 1.27~1.34초였다. 상한 512를 주면 base는 2.31~2.35초에 머물고 branch는 1.28~1.32초로 내려간다.

### 5.2 fused와 gather의 greedy 일치는 prompt에 따라 다르다

공유 문서는 `qwen3-30b-a3b`에서 kernel on/off의 greedy 64토큰이 같다고 적었다. 이번 prompt에서는 생성 39토큰까지 같다가 갈라지며, 두 바이너리에서 결정적이고 동일하게 재현된다. 덕분에 이 checkpoint에서는 fused id 비교 자체도 판별력이 있었다. 조용한 fallback이었다면 39번째 토큰에서 갈라졌을 것이다.

### 5.3 decline 테스트는 mutation으로 확인한다

`None`을 기대하는 테스트는 함수가 어떤 이유로 decline해도 통과한다. 같은 geometry의 Dff 64 positive control로 다른 가드를 배제했고, `dff > max_dff` 분기만 끈 임시 mutant에서 새 decline 테스트 두 개가 모두 실패해 상한 자체를 고정한다는 것을 확인했다.

## 6. 추가 학습 리소스

| 키워드 | 설명 | 관련성 |
|--------|------|--------|
| `gather_qmm` | stacked expert에 대한 MLX gathered quantized matmul | fallback 및 prefill 경로 |
| `fused_moe_expert_kernel` | 단일 토큰 SwiGLU expert용 2-launch custom kernel | 상한이 빠져 있던 경로 |
| `MLXCEL_FUSED_MOE_MAX_DFF` | fused 경로의 expert 폭 상한 | 이제 두 계열이 읽음 |

관련: #268, #275, #311, #643, #958, #1045, #1803, #1859.

## 7. 변경 요약

| 항목 | 값 |
|------|----|
| 변경 파일 | 7개(이 리포트 제외) |
| 라인 | +524 / -432 |
| 추가 테스트 | 3개(qwen3_moe loader 경유 Dff decline, qwen3_vl_moe loader 경유 Dff decline, mxfp4 decline) |
| 대상 변경 테스트 | 2개(두 계열의 #958 가드) |

| 해시 | 유형 | 메시지 |
|------|------|--------|
| `44e58eeb` | refactor | move qwen3_moe experts onto the shared SwitchGLU |
| `a78d39e9` | test | pin the dff decline through the Qwen3-VL-MoE loader too |
| `b7431338` | docs | state which families read MLXCEL_FUSED_MOE_MAX_DFF |
| `92e3faf9` | docs | qualify the qwen3-30b-a3b fused greedy parity claim |
| `c314b253` | docs | add technical report for PR #1888 |
| `0c80472a` | docs | name the Qwen3-Omni thinker and fix issue references |

## 8. 후속 조치

### 향후 개선 사항

- `qwen3_next.rs`(Qwen3Next, Qwen3.5, qwen3_omni_moe talker)에 같은 SwiGLU kernel 사본이 있고 상한, mode, activation 가드가 똑같이 빠져 있다.
- `gemma4.rs`는 GeGLU kernel을 쓰며 mode 검사는 있으나 상한이 없다. 통합하려면 먼저 GeGLU용 `SwitchGluActivation`이 필요하다.
- NemotronH의 opt-in `MLXCEL_FUSED_MOE_RELU2` 경로도 상한 없이 down kernel을 launch한다.
- fused 경로 없이 로컬 `SwitchGLU`를 가진 계열이 10개 있다: deepseek, deepseek_v2, deepseek_v3, deepseek_v32, ernie4_5_moe, exaone_moe, glm4_moe, glm4_moe_lite, hunyuan_moe, llama4.
- 공유 상한은 호출마다 환경 변수를 읽고 backend를 질의한다. `fused_moe_enabled`처럼 `OnceLock`에 캐시하면 이 작업이 없어진다.
- `deepseek_v4_moe`는 `validate_expert_quantization_params`를 호출하지만 Used-by 목록에 빠져 있다.
- 보안 검토에서 나온, 공유 loader에 원래 있던 문제: 세 expert plane끼리의 shape과 router 행 수 대 stack된 expert 수를 비교하지 않아, plane이 서로 맞지 않는 checkpoint는 경계 검사 없는 kernel 인덱스에 도달할 수 있다. stack되지 않은 `experts.{idx}` fallback도 선언된 expert 수 없이 쌓고, shape을 먼저 비교하지 않은 채 `stack`을 호출한다. 두 가지 모두 공유 `SwitchGLU` loader에 넣어야 모든 계열이 혜택을 받는다.
- `docs/benchmark_results/fused-moe-decode-kernel-design.md`의 계열 목록("eleven model paths")은 오래되었다.
- CI의 `OpenXLA feature compile` job은 `main`의 `b8d10fb1`에서 이미 `--no-default-features` 조건의 `src/models/mod.rs`와 server 모듈 unused import 오류로 실패하며, 이 PR에서도 같은 이유로 실패한다. 이 PR과는 무관하다.

## 부록

### A. 테스트 결과

GB10(`test-fast` profile, `--features cuda`)에서 `models::qwen3_moe`, `models::qwen3_vl_moe`, `models::switch_layers` 아래 테스트 38개를 각각 별도 프로세스로 실행해 `92e3faf9`와 최종 `0c80472a`에서 모두 통과했다. Dff decline positive control은 skip 없이 실행되었다. `cargo fmt --check`와 `cargo clippy --profile test-fast --features cuda -p mlxcel --lib --tests --no-deps -- -D warnings`는 경고가 없다. 실행하지 않은 항목: `metal,accelerate` gate(Linux에서 실행 불가), workspace 전체 `verify-test-cuda`.

### B. GB10 검증(sm_121, CUDA, release build)

| 확인 항목 | Base `b8d10fb1` | Branch |
|-----------|-----------------|--------|
| `qwen3-30b-a3b-4bit` id, fused(64토큰, `-t 0`) | 기준 | 동일 |
| `qwen3-30b-a3b-4bit` id, `MLXCEL_FUSED_MOE=0` | 기준 | 동일 |
| `qwen3-vl-30b-a3b-4bit` id, fused | 기준 | 동일 |
| `qwen3-vl-30b-a3b-4bit` id, `MLXCEL_FUSED_MOE=0` | 기준 | 동일 |
| decode trace, `-n 4` | `path=fused tokens=1` 144회, `gather_qmm` 0회 | `path=fused tokens=1` 144회, `gather_qmm` 0회 |
| decode trace, `MLXCEL_FUSED_MOE_MAX_DFF=512` | fused 144회(상한 무시) | `path=gather_qmm tokens=1` 144회 |
| `mlxcel-server` `qwen3-30b-a3b-4bit`, temperature 0, 64토큰 | 기준 | content와 reasoning 동일 |
| `mlxcel-server` `qwen3-vl-30b-a3b-4bit`, temperature 0, 64토큰 | 기준 | 동일, 일관된 문장, CLI id와 같은 시작 토큰 |

Branch 열은 `a78d39e9`에서 빌드한 바이너리로 측정했고, 이후 커밋은 주석과 문서만 바꾼다. 최종 `0c80472a`에서 다시 빌드한 바이너리로 id, trace, 상한, server 확인을 반복해 base와 다시 일치했다. 같은 최종 바이너리로 Qwen3-Omni-MoE thinker(`qwen3-omni-30b-a3b-instruct-4bit`, 텍스트 전용)도 확인했다. fused 경로와 `MLXCEL_FUSED_MOE=0` 모두 base와 id가 같고, 두 경로끼리는 26번째 토큰에서 갈라진다. `MLXCEL_FUSED_MOE_MAX_DFF=512`에서 base id는 fused id와, branch id는 `MLXCEL_FUSED_MOE=0` id와 같다.

`qwen3-coder-480b-a35b-instruct-4bit`는 header로만 확인했다. 전 과정에서 kernel driver의 `NV_ERR_NO_MEMORY`는 0건이었다.
