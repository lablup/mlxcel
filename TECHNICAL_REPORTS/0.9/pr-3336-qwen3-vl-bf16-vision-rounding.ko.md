# PR #3336: Qwen3-VL vision 타워를 eager의 BF16 경계에서 반올림

**날짜**: 2026-11-24
**상태**: PR 열림 (#3291 해결, #3290에 의존, #3212에서 발견, #932 및 에픽 #566의 일부). 단위 테스트 통과. GB10 CUDA에서 실제 Qwen3-VL 체크포인트로 eager MLX와 비교 검증

## 요약

XLA Qwen3-VL vision 그래프는 모든 연산을 f32로 실행했지만, eager MLX는 BF16 체크포인트의 vision 타워를 BF16 텐서로 실행합니다. 24개 블록을 지나며 둘이 어긋났고, fixture 이미지에서는 생성 인덱스 8의 near-tie를 XLA가 반대로 결정하면서 greedy 출력이 갈라졌습니다. 전처리는 #3290이 이미 고쳤으므로 남은 원인은 반올림이었습니다. 이 PR은 f32 carrier를 유지하면서, eager가 BF16 텐서를 저장하는 모든 지점에서 BF16으로 반올림합니다. GEMM, norm reduction, softmax 통계의 f32 누산은 그대로 둡니다.

반올림은 f32 비트 패턴에 대한 정수 round-to-nearest-even이며 NaN 가드를 갖추고 있어, BF16 타입이 없는 Metal에서도 같은 그래프가 lowering됩니다. 연산자 수치 계약이 새 반올림을 선언하므로 계약 digest와 아티팩트 digest가 바뀌고, 캐시된 Qwen3-VL vision 아티팩트는 한 번 다시 컴파일됩니다. fixture에서 첫 greedy 분기 지점은 인덱스 8에서 27로 이동했고, 27은 eager 자체의 정확한 동점입니다. 관리자는 #3291의 수용 기준 두 개를 수정했습니다. 연쇄 block-23 drift는 기록만 하고 게이트로 두지 않으며, f32 host-dequant 대조군은 더 이상 목표가 아닙니다.

## 1. 문제 정의

### 1.1 배경

#3212가 Qwen3-VL의 XLA와 eager를 비교하다가 256x256 단색 이미지에서 분기를 발견했습니다. 생성 인덱스 8에서 XLA는 6437=22.8980이 4285=22.8781보다 높았고(margin 0.020), eager는 4285=22.875가 6437=22.75보다 높았습니다(margin 0.125, 해당 크기에서 BF16 한 단계). XLA vision feature를 eager 언어 모델에 넣으면 정확한 BF16 동점이 나오므로 언어 쪽은 원인이 아니었습니다. #3290이 이미지 전처리를 고쳐 그 부분의 차이를 없앴고, 남은 것은 vision 타워 자체, 즉 `QWEN3_VL_VISION_PRECISION = F32` 대 eager의 BF16이었습니다.

### 1.2 f32가 더 정확해도 해결책이 아닌 이유

기준은 eager의 결과이고, eager의 결과에는 저장되는 텐서마다 BF16 반올림이 들어 있습니다. 그 반올림을 건너뛴 그래프는 실수 연산의 정답에는 가깝지만 eager에서는 멀어집니다. 이 타워의 마지막 블록 활성값은 2500을 넘고, 그 크기에서 BF16 한 단계는 16입니다. 앞쪽 블록에서 반올림을 건너뛰면 뒤쪽 블록이 보는 값이 달라집니다. 비트 단위 일치는 애초에 목표가 아니었습니다(GEMM 내부 합산 순서는 백엔드마다 다릅니다). 목표는 eager의 반올림 경계를 맞춰 남는 차이를 합산 순서로만 한정하는 것이었습니다.

### 1.3 방치했을 때의 위험

| 위험 | 영향 | 가능성 |
|---|---|---|
| 일반 이미지의 near-tie에서 XLA Qwen3-VL 출력이 eager와 갈라짐 | 중간: 백엔드 간 토큰 차이가 사용자에게 보임 | 활성값이 큰 이미지에서 높음 |
| #932가 Qwen3-VL 패밀리를 검증할 때 통과하는 vision 타워 oracle을 근거로 댈 수 없음 | #932에 높음 | 이 PR이 없으면 확실 |

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| `emitter/qwen3_vl_ops.rs` | `round_bf16`, `linear_bf16`, `layer_norm_bf16`, `exact_gelu_bf16`, `tanh_gelu_bf16`, `apply_vision_rope_bf16`, `attention_bf16`, `add_bf16`. |
| `emitter/qwen3_vl.rs`, `emitter/mod.rs`, `emitter/builder.rs` | 새 연산으로 타워를 구성하고 `Precision`을 인자로 받음. `round_bf16`에 필요한 정수 및 bitcast 헬퍼 추가. `QWEN3_VL_VISION_PRECISION`은 `qwen3_vl_vision_precision(device)`로 바뀜. |
| `qwen3_vl_runtime.rs` | 디바이스별로 precision을 정해 emitter와 `ProductionNumerics`에 같은 값을 전달하므로 그래프와 계약이 한 값을 읽음. |
| `operator_numeric_contract/families.rs`, `family_tests.rs` | attention scores를 제외한 모든 Qwen3-VL 연산자에 BF16 출력 반올림을 선언. pin 갱신. |
| `CHANGELOG.md` | 1회 재컴파일을 포함한 Changed 항목. |
| 테스트 | `qwen3_vl_vision_precision`을 타깃별로 pin (Metal은 F32, CUDA는 BF16. `precision_lowers_on(device, Bf16)`이 참이면 BF16). 가장 작은 patch bucket과 가장 큰 bucket에서 Metal precision으로 전체 vision 그래프를 emit하여 `bf16` 타입이 없음을 검증. |

8개 파일, +603/-229. 허용 오차는 바꾸지 않았습니다(atol/rtol 0.08).

## 3. 기술적 선택과 그 이유

### 3.1 연산별 반올림 설계

각 규칙은 이 체크포인트(vision 타워는 양자화되지 않은 BF16)에서 eager MLX CUDA 경로가 저장하는 값을 그대로 따릅니다.

- **입력 patch**는 BF16으로 한 번 캐스팅합니다.
- **Patch embedding과 모든 linear.** MLX는 `matmul` 다음에 bias를 별도의 `add`로 더합니다. 곱은 BF16으로 저장되고(GEMM 내부는 f32 누산), bias 덧셈 결과가 다시 저장됩니다. 그래프도 두 번 반올림합니다. `linear_bf16`은 곱을 반올림하고, broadcast한 bias를 더하고, 다시 반올림합니다.
- **보간된 position embedding.** 네 모서리 항을 f32에서 `(c0+c1)+(c2+c3)` 순서로 더하고 한 번 반올림합니다. 입력, patch embedding, position embedding 단계에서 fixture의 eager 대비 차이는 정확히 0입니다.
- **LayerNorm**은 MLX CUDA 커널(`layer_norm.cu`)을 따릅니다. f32 평균과 분산 뒤에 정규화 값, `w * n`, `+ b`를 각각 반올림하므로 통계 이후 반올림이 세 번입니다. main merger는 정규화 후 reshape하고 DeepStack merger는 reshape 후 정규화하는데, 그래프도 각각의 순서를 유지합니다.
- **RoPE.** q와 k를 f32로 넓혀 회전한 뒤 한 번 반올림합니다.
- **Attention**은 cuDNN flash SDPA를 따릅니다. eager는 head_dim 64의 마스크 없는 BF16 attention에서 이 경로를 선택합니다. score와 행 합은 f32로 두고, 정규화 전 확률은 value GEMM 전에 BF16으로 반올림하며, 정규화된 context를 반올림합니다. attention scores만 출력 반올림이 선언되지 않은 이유입니다.
- **GELU**(블록은 tanh 형, merger는 exact 형)는 f32로 계산하고 한 번 반올림합니다.
- **Residual add**는 한 번 반올림합니다. **main과 DeepStack merger**도 같은 linear, norm, GELU 연산을 씁니다.

### 3.2 NaN 가드를 갖춘 정수 round-to-nearest-even

가장 단순한 구현은 `f32`를 `bf16`으로 변환했다가 되돌리는 것입니다. 그러면 lowering된 모듈에 BF16 타입이 필요한데 Metal 타깃에는 없습니다(#612). `round_bf16`은 비트 패턴으로 처리합니다.

```rust
let bits = bitcast_convert(value, ui32);
let lsb = (bits >> 16) & 1;
let rounded = (bits + 0x7fff + lsb) & 0xffff_0000;
select(value != value, value, bitcast_convert(rounded, f32))
```

`0x7fff`와 유지되는 최하위 비트를 더하면 짝수 쪽으로 가까운 값으로 반올림되고, 마스크가 하위 16비트를 버립니다. 올림이 지수로 넘어가는 것은 다음 2의 거듭제곱으로 올림되는 값에서 올바른 동작입니다. NaN에는 가드가 필요합니다. CUDA의 표준 NaN은 `0x7fffffff`인데 정수 올림이 이를 `0x80000000`, 즉 `-0.0`으로 바꿉니다. `value != value`에 대한 `select`가 NaN을 그대로 통과시킵니다.

contraction의 입력은 `precision_lowers_on(device, Bf16)`이 참인 타깃(CUDA)에서는 BF16, Metal에서는 BF16 값만 담는 f32 carrier입니다. carrier에는 BF16으로 표현 가능한 값만 들어 있으므로 곱은 같습니다. 이 선택은 `qwen3_vl_vision_precision(device)`가 하고, Metal precision 그래프에 `bf16` 타입이 없음을 테스트가 검증하므로 Metal 컴파일 가능성이 구성에만 의존하지 않습니다.

| 옵션 | 장점 | 단점 |
|---|---|---|
| `bf16`으로 변환 후 복원 | 연산 하나, 가장 단순 | BF16 타입이 필요, Metal에서 실패 |
| 타워 전체를 BF16으로 실행 | 연산 수 감소 | eager가 유지하는 f32 누산과 통계를 잃음 |
| **선택: f32 carrier와 정수 반올림** | 모든 타깃에서 lowering, 누산은 f32 유지 | elementwise 연산 증가(XLA가 fuse), 반올림 헬퍼 사본 4번째 |

### 3.3 계약과 digest 변경

Qwen3-VL 연산자 목록은 attention scores를 제외한 각 연산자에 BF16 출력 반올림을 선언하고, `ProductionNumerics`는 emitter가 쓴 것과 같은 `Precision`을 받습니다. 그 결과 두 가지가 바뀝니다. v3 매니페스트의 계약 digest는 `ae5bfb35...`에서 `bdd1632d...`로, 아티팩트 digest는 `18e85256...`(`defac5fb`)에서 `4d3934a5...`(`775dbb9f`)로 바뀝니다. VMFB 캐시 경로는 emit된 MLIR 텍스트에서 만들어지므로 새 그래프는 새 경로도 갖습니다. 저장된 계약이 다른 아티팩트는 로드 전에 다시 컴파일되므로(#3242의 메커니즘) 이전 f32 아티팩트는 재사용되지 않습니다. 비용은 업그레이드 후 첫 사용 시 patch bucket마다 컴파일이 한 번 더 일어나는 것입니다.

### 3.4 관리자가 수정한 수용 기준

#3291의 기준 두 개가 2026-11-23에 수정되었고, 이유가 이슈에 기록되어 있습니다.

- **연쇄 vision drift가 Qwen2.5-VL 수준 이하.** final logit은 충족했습니다(0.198 대 0.85). 연쇄 block 23은 충족하지 못했습니다(184 대 133). block 10이 1단계 합산 순서 차이를 두 테스트 이미지에서 약 40배 증폭하는데, GEMM 내부 f32 합의 순서에서 오는 차이는 반올림 경계를 맞춰도 없앨 수 없습니다. 이 값은 게이트가 아니라 기록으로 둡니다.
- **모든 same-input 대조군 통과.** report-only인 block-23 `host_dequant_dense_f32` 대조군은 f32 기준과 비교합니다. eager의 BF16 반올림을 맞추면 f32와의 거리는 넓어지는 것이 당연하므로 더 이상 목표가 아닙니다. shapes224에서 6개 값이 허용 오차를 넘으며, 모두 크기 2545(한 단계가 16)에서의 BF16 상쇄 오차입니다. eager BF16과 비교하는 fused 대조군은 main에 이미 있던 2건의 실패를 그대로 유지합니다.

허용 오차는 완화하지 않았습니다.

## 4. 결과

GB10 CUDA, `xla_qwen3_vl_deepstack_check`, atol/rtol 0.08 유지, 같은 fixture에서 전후 비교.

| 측정 | 이전 | 이후 |
|---|---|---|
| shapes224 final-logit max_abs | 0.205 | 0.198 |
| shapes224 연쇄 block 23 (기준 2528) | 230 | 184 |
| shapes224 main merger | 1.82 | 1.00 |
| fixture 연쇄 block 23 (기준 2656) | 1020 | 1100 |
| fixture main merger | 1.52 | 5.16 |
| fixture same-input block 23 대조군 (fused) | 실패 3건 | 실패 0건 |
| Qwen3-VL greedy 대 eager (fixture, 64 토큰) | 8에서 분기 | 27에서 분기 |
| Qwen2.5-VL greedy 대 eager | 일치 (8에서 EOS) | 일치 (8에서 EOS) |
| warm `iree_vision_seconds` (1회) | 1.85 s | 1.67 s |

운영 eager 블록의 same-input 대조군은 fixture block 10(2단계, report-only, 이전 0.338)을 제외한 모든 블록에서 BF16 한 단계 이하입니다. fixture의 main merger가 나빠진 것(1.52에서 5.16)은 입력인 연쇄 block 23이 나빠졌기 때문(1020에서 1100)이며, merger 자체의 반올림이 원인이 아닙니다. 이미지와 merger 8쌍 모두에서 각 merger의 drift는 그 입력 블록의 연쇄 drift와 같은 방향으로 움직입니다.

인덱스 27은 eager의 정확한 동점입니다. 토큰 576과 2619가 모두 30.375입니다(margin 0.0). XLA는 2619가 576보다 0.021 높은데, 이는 그 크기에서 BF16 한 단계(0.125) 안입니다. 분기가 margin 한 단계 이하인 곳에서만 일어나야 한다는 기준을 충족합니다.

## 5. 한계

- **합산 순서 증폭.** cuBLAS와 IREE 커널의 GEMM 합산 순서가 다릅니다. block 10이 이 차이를 두 이미지에서 약 40배 증폭합니다. 따라서 block 23의 연쇄 drift(184 대 Qwen2.5-VL의 133)는 기존 목표보다 높고, 누산 순서 자체가 맞춰지기 전에는 그대로입니다.
- **타일별 running max는 재현하지 않음.** flash 커널은 key 타일마다 running maximum을 유지하지만 그래프는 행 최댓값을 씁니다. 일부 타일에서 확률을 반올림하기 전에 빼는 오프셋이 달라 반올림이 한 단계 이동할 수 있습니다. 코드에 문서화되어 있습니다.
- **CUDA 커널의 반올림을 모든 타깃에 적용.** LayerNorm과 attention의 반올림은 MLX CUDA 커널과 cuDNN flash SDPA를 따르며 Metal과 CPU 타깃을 포함한 모든 타깃에 적용됩니다. Metal이나 CPU의 eager는 해당 연산에서 다르게 반올림할 수 있고, Metal 또는 CPU eager와의 비교는 수행하지 않았습니다.
- **BF16 정확 carrier 가정.** Metal의 f32 carrier가 같은 곱을 만드는 것은 vision 가중치가 BF16으로 정확히 표현될 때뿐입니다. 양자화된 vision 타워는 호스트에서 역양자화하면 BF16로 표현되지 않는 값이 나오므로, BF16 체크포인트에서만 성립합니다.
- **same-input merger 대조군 없음.** merger 근거는 간접적입니다(연쇄 drift와의 방향 일치).
- **타이밍은 1회.** vision 시간 변화(1.85 s에서 1.67 s)는 warm 1회 측정이며 벤치마크가 아닙니다.
- **네 번째 반올림 헬퍼.** `round_bf16`은 emitter의 BF16 반올림 헬퍼 네 번째 사본입니다. 정수 형태는 Metal 때문에 필요하고, 나머지 셋은 `bf16`을 거쳐 변환합니다.

## 6. 학습 포인트

- BF16 기준에 맞춘다는 것은 기준이 반올림하는 위치를 맞추는 것이지, 반올림을 많이 하거나 적게 하는 것이 아닙니다. 더 높은 정밀도로 실행하면 기준에서 멀어집니다.
- 비트 패턴으로 BF16 반올림하기(`0x7fff`와 유지되는 하위 비트를 더하고 하위 절반을 마스크)는 BF16 변환을 대체하는 이식 가능한 방법이며, 명시적인 NaN 가드가 필요합니다.
- 차이가 합산 순서로만 줄었다면, 그것을 닫으라고 요구하는 게이트는 엉뚱한 것을 재고 있는 것입니다. 기록으로 남기고 실제로 중요한 결과(final logit, greedy margin)로 게이트를 겁니다.
- emitter와 계약에 같은 `Precision` 값을 넘기면 그래프와 기록된 수치가 서로 어긋날 수 없습니다.

## 관련 커밋

| 해시 | 유형 | 메시지 |
|---|---|---|
| `65ac67e2` | update | round the Qwen3-VL vision tower at eager's BF16 boundaries |
| `7dfef804` | test | pin the Qwen3-VL vision precision per target and add changelog |

관련: #3291(이 변경), #3290(전처리), #3212(분기 발견), #3211과 PR #3242(계약 결합), #932, #566.

## 검증

- `cargo test -p mlxcel-xla --lib qwen3_vl`: 17개 통과. `operator_numeric_contract`: 19개 통과. `cargo clippy -p mlxcel-xla --lib --tests -- -D warnings` 통과.
- `cargo check --features cuda,xla-iree --all-targets`, `cargo check --no-default-features --features xla-diagnostics --all-targets`.
- shapes224와 fixture에서 인덱스 8과 27에 `--token-ids`를 주고 deepstack check를 전후로 실행.
- `mlxcel generate -n 64 --temp 0`으로 eager와 `MLXCEL_BACKEND=xla`를 비교 (qwen3-vl-2b-instruct-4bit, qwen2.5-vl-3b-instruct-4bit).
