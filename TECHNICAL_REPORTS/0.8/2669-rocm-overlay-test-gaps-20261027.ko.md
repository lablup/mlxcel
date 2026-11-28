# PR #2669: ROCm에서 donate된 LayerNorm cotangent, gemv fallback, 빈 norm 입력 테스트 추가

**작성일**: 2026-10-27
**상태**: Linux gfx1151 (ROCm 7.15)에서 완료
**언어**: Rust, Markdown
**위험도**: 낮음 (테스트와 기록만 변경)

## 요약

이슈 #2663은 ROCm overlay에서 테스트가 해당 코드에 도달하지 않아 통과하던 네 곳을 지적했습니다. PR #2669는 그중 세 곳에 테스트를 추가하고, 나머지 한 곳의 오래된 기록을 바로잡습니다. 커널 코드는 바꾸지 않았습니다. 새 테스트는 각 경로를 일부러 망가뜨리면 실패함을 확인했습니다. 작업 내용은 LOCAL_FIXES 항목 68에 기록했습니다.

## 1. 문제

- `LayerNormVJP::eval_gpu`는 bias gradient를 VJP 커널보다 먼저 reduce합니다. `gx`나 weight-gradient 임시 버퍼로 donate된 cotangent를 커널이 덮어쓰기 때문입니다(항목 64). 그런데 테스트 bridge는 호출자가 보유한 cotangent를 넘기므로 donate가 일어나지 않았고, 이 순서는 검증되지 않았습니다.
- `gemv`는 batch 차원이 8개를 넘을 때만 shape과 stride를 device 메모리에 두고 `gemv_batched`를 실행합니다. 9개 차원을 쓰는 테스트가 없었습니다.
- 0행 norm 테스트에는 weight 없는 `rms_norm`이 빠져 있었습니다. 빈 입력이 `RMSNorm::eval_gpu`의 `strided` 분기를 타는 경우도 없었습니다.
- LOCAL_FIXES 항목 46은 y grid가 65535 block을 넘으면 거부되고, row/tile 테스트가 수정 전에 실패했다고 적고 있었습니다. 그러나 항목 61이 이미 ROCm 7.15가 이 한도를 강제하지 않음을 측정했습니다.

## 2. 변경 내용

| 영역 | 변경 |
|---|---|
| LayerNorm VJP | 전치된 cotangent는 `check_input`이 복사하고, 그 복사본은 donate할 수 있습니다. 연속 `x`를 보유하면 복사본이 `gx`로 donate됩니다. `x`도 전치되고 weight가 있으면 weight-gradient 임시 버퍼로 donate됩니다. 두 경우 모두 weight 유무별로 f32, f16, bf16과 24 x 8, 148 x 33 크기에서 실행합니다. `gx`, `gw`, `gb`는 호스트 f64 기준값과 비교합니다. |
| gemv | 크기 2인 batch 축 9개에 대해 `A @ v`를 계산합니다. broadcast stride가 행렬과 벡터에 번갈아 있어 `collapse_contiguous_dims`가 축을 합치지 못합니다. 512개 batch 모두를 호스트에서 확인합니다. |
| Norm | weight 없는 0행 `rms_norm`을 추가했습니다. stride `[16, 1]`인 빈 `as_strided` view `[0, 8]`에 대해 weight 유무별 `rms_norm`도 추가했습니다. |
| 기록 | 항목 46과 `tests/rocm_hip_grid_clamp.rs`, `gather_qmv.rs`의 주석에 측정된 한도를 적었습니다. cap은 유지합니다. |

## 3. 기술적 결정

### 3.1 bridge 변경 없이 donate 경로 진입

이슈가 제시한 방법은 새 bridge 함수가 필요 없습니다. `check_input`이 `donate_g |= g_copied`를 설정하므로, cotangent를 전치하는 것만으로 충분합니다. `x`도 전치하면 두 번째 경우가 생깁니다. `x`의 복사본이 `gx`를 차지하므로, cotangent는 weight-gradient 임시 버퍼로 갑니다.

### 3.2 strided 분기는 도달 가능

MLX는 빈 slice에 새 row-major stride를 주므로, slice는 packed로 판정됩니다. 반면 `AsStrided::eval`은 받은 stride를 유지하고, 빈 view는 모두 contiguous로 표시합니다. 그러면 축 크기 8에 대한 row stride 16 때문에 `packed`는 false, `strided`는 true가 됩니다. 이 분기에 임시로 throw를 넣어 테스트가 이 분기에 도달함을 확인했습니다.

## 4. 검증

- **Red, gx donate**: 커널 뒤에서 `gb`를 reduce하면 `gb element 0 is -0.23338366, expected 1.125`가 나옵니다.
- **Red, gw donate**: 같은 조작에서 `-16.554958`이 나옵니다. 이 조작에서도 기존 항목 64 테스트는 통과했습니다.
- **Red, gemv**: `d_vec_strides`를 0으로 채우면 `batch 1 (matrix 0, vector 1) row 0: got -23, expected -5`가 나옵니다.
- **Red, strided norm**: 테스트가 던져진 메시지로 실패합니다.
- **Probe**: HIP 71526333에서 `hipLaunchKernel`은 65535부터 16777216 block까지의 y, z grid를 모든 block까지 실행했습니다. 이 runtime의 `maxGridSize`는 y와 z에 65535를 보고합니다. 거부된 것은 한 차원에 2^32 thread를 요청한 경우뿐이었습니다.
- **Green**: `rocm_hip_grid_clamp`는 29개 통과, 10개 ignore입니다. overlay records, fmt, 수정한 테스트 target의 clippy도 통과했습니다.

## 5. 남은 한계

- 이 크기의 cap 없는 y grid도 이 runtime에서는 실행되므로, 빠른 grid 테스트를 y cap의 red/green 테스트로 볼 수는 없습니다. 이 테스트들은 cap이 적용된 grid-stride 실행을 검증하는 용도로 남습니다.
- probe는 ROCm 7.15에서만 실행했습니다. 다른 runtime은 `maxGridSize`를 강제할 수 있으므로 cap을 유지합니다.
