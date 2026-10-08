# 기술 보고서: PR #2235 - warp-shared gather_qmv 커널에 mxfp4 경로 추가

**날짜**: 2026-10-08

**상태**: gfx1151 호스트(ROCm/HIP)에서 구현 및 검증 완료. 코드 head `16be6bf9`, main `6cd0139d` 기준, PR 열림, 머지 대기. #2178 종료.

**언어**: C++/HIP(ROCm 오버레이 `qmm.hip`), Rust(`tests/rocm_mxfp4_quant.rs`의 신규 테스트, 수정 `tests/rocm_gather_qmm_expert_batched.rs`), Markdown(`patches-rocm/LOCAL_FIXES.md`, 벤치마크 페이지)

**위험도**: 낮음에서 중간. 새 경로는 그룹 크기 32이고 `K % 32 == 0`인 bf16 mxfp4 `gather_qmm`에 기본으로 켜지므로 gpt-oss의 모든 디코드 스텝이 이 경로를 탄다. 합산 순서가 행별 커널과 달라 일부 bf16 출력이 마지막 비트에서 다르다. `MLX_ROCM_GATHER_QMV_USE_WARP=0`으로 이전 커널로 돌아간다. f16과 f32 mxfp4, mxfp8, affine 디스패치는 바뀌지 않는다. Metal과 CUDA 코드 경로는 건드리지 않는다.

## 요약

#2164가 gpt-oss-20b 프리필을 빠르게 만든 뒤에도 디코드는 약 8.5 tok/s에 머물렀다. 디코드 스텝은 토큰 하나를 `top_k = 4`개 전문가로 보내므로 `B = 4`이고, expert-batched 커널의 `B >= 64` 게이트에 한참 못 미친다. `GatherQMM::eval_gpu`의 다른 빠른 경로는 모두 affine 전용이어서, bf16 mxfp4는 행별 `gather_qmv_kernel`로 떨어졌다. 출력 열마다 스레드 하나가 `K` 전체를 걷는 커널로, 호출당 1.39 ms, 스텝당 72회, 115 ms 스텝 중 약 100 ms였다. #2164 보고서가 이 커널을 다음 목표로 지목했다.

수정은 `gather_qmv_warp_shared_kernel`에 mxfp4 경로를 추가한다. 각 레인이 스텝마다 32비트 패킹 워드 하나(e2m1 니블 8개, 모두 32짜리 그룹 하나 안)를 `shared_x` 청크 전체에 걸쳐 처리하고, 그 그룹의 E8M0 스케일을 읽고, 분기 없이 디코드하며, 워드의 내적을 한 번에 되돌려 곱한다. 기존의 그룹별 루프는 그룹 크기 32에서 16개 레인 중 0~3번 레인만 일했다. 새 디스패치 경로가 bf16 mxfp4를 기본으로 여기에 보낸다.

gfx1151에서 `rocprofv3`는 새 인스턴스를 호출당 108.3 us로 504회 호출하고 `gather_qmv_kernel`은 없음을 보여 준다. 번갈아 실행한 가드 6라운드에서 디코드 중앙값은 8.32에서 63.62 tok/s로(7.65배, 범위 겹치지 않음), 512토큰 프리필은 511.73에서 510.67 tok/s로(호스트 편차 안) 바뀐다. 행별 커널과의 teacher-forced `logit_trace` 비교에서 결정된 불일치는 89개 중 0개다. 이 유닛의 `make verify-rocm`은 통과했다(스위트 161개, 12,188개 통과, 0개 실패, 398개 무시). Metal, CUDA, wave64는 검증하지 못했다.

## 1. 문제 정의

### 1.1 디코드가 expert-batched 커널에 닿지 못한 이유

#2164의 커널은 `M == 1`, `B >= 64`, `E <= 64`, `B / E >= 4`인 정렬·전치 `gather_qmm`을 처리한다. 토큰이 충분한 프리필이 해당한다. 디코드 스텝은 토큰이 하나이고 `top_k = 4`이므로 `B = 4`이며, 32토큰 미만의 프리필도 게이트를 넘지 못한다. affine 빠른 경로(wide, warp-shared)는 affine 전용이다. 그래서 bf16 mxfp4 호출은 일반 non-affine 경로(LOCAL_FIXES 항목 10)의 `gather_qmv_kernel<T, uint8_t, 4, 32, false>`로 갔다.

### 1.2 그 비용

행별 커널은 출력 열마다 스레드 하나를 두고 혼자 `K` 전체를 순회한다. gfx1151의 가드된 `rocprofv3` 트레이스(#2106, #2164 프로파일)에서 호출당 1.39 ms, 디코드 스텝당 72회(24개 레이어, gather 프로젝션 3개), 7스텝에 504회였다. 115 ms 스텝 중 약 100 ms이므로 디코드는 약 8.5 tok/s였다.

### 1.3 기존 warp-shared 커널이 mxfp4에 도움이 되지 못한 이유

`gather_qmv_warp_shared_kernel`은 `x`의 한 청크를 공유 메모리에 올리고 `THREADS_PER_COL`개(16) 레인이 출력 열 하나를 함께 계산한다. 안쪽 루프는 그룹을 걷고, 그룹 안에서 각 레인은 `lane * 8`에서 시작한다. 그룹 크기 32에서는 0~3번 레인만 일하고 4~15번 레인(16개 중 12개)은 논다. 디코드도 분기가 많은 `fp4_e2m1_to_float` switch를 거쳤다. 이 커널은 affine에만 인스턴스화되어 있었다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| `qmm.hip`, `gather_qmv_warp_shared_kernel` | 새 `kMxfp4WordPath`(`!AFFINE && BITS == 4 && GROUP_SIZE == 32`): 청크 전체에 걸친 워드 순회. 이 경우 그룹별 루프는 건너뛰고 다른 경우를 위해 유지한다. |
| `qmm.hip`, `GatherQMM::eval_gpu` | affine warp-shared 경로 뒤에 새 경로: bf16 mxfp4, `group_size_ == 32`, `bits_ == 4`, `K % 32 == 0`, 스레드 수 16 또는 `WARP_SIZE`. 인스턴스 두 개 추가. 일반 non-affine 경로의 주석 갱신. |
| `tests/rocm_mxfp4_quant.rs` | 신규 `mxfp4_warp_shared_gather_qmv_matches_per_row_and_reference`(5.1절). |
| `tests/rocm_gather_qmm_expert_batched.rs` | 행별 디스패치 검사가 `MLX_ROCM_GATHER_QMV_USE_WARP=0`을 설정한다(5.2절). |
| `patches-rocm/LOCAL_FIXES.md` | 항목 41 신설, 항목 10 갱신. 이슈는 새 항목을 32로 불렀지만 이미 쓰였고, 40은 #2220에 갔다. |
| `docs/benchmark_results/rocm-moe-decode-mxfp4-gfx1151-2026-10-08.md` | 새 결과 페이지. |

커밋 1개, 파일 5개, 428줄 추가, 6줄 삭제.

## 3. 설계

### 3.1 워드 경로

mxfp4에서 각 레인은 스텝마다 패킹된 `uint32_t`(4비트 값 8개) 하나를 처리한다.

```cpp
for (int k = chunk_start + lane * 8; k < chunk_end; k += THREADS_PER_COL * 8) {
  const uint32_t packed = *reinterpret_cast<const uint32_t*>(&w_row[k / 2]);
  const float scale = load_scale_value<ScaleT, GROUP_SIZE, false>(scales_row[k / GROUP_SIZE]);
  const float* xs = &shared_x[k - chunk_start];
  float qx = 0.0f;
  for (int j = 0; j < 8; ++j)
    qx = fmaf(xs[j], fp4_e2m1_to_float_scaled((packed >> (4 * j)) & 0xFu), qx);
  acc = fmaf(scale, qx * kFp4HalfBitsScale, acc);
}
```

- **모든 레인이 일한다.** 연속한 레인이 연속한 워드를 맡으므로, 그룹마다 4개 레인이 32개 값을 처리하는 대신 16개 레인이 스텝마다 128개 값을 처리한다.
- **워드당 스케일 하나.** 8은 32의 약수라 워드가 그룹을 넘지 않는다. 스케일은 워드마다 한 번 읽어 워드의 내적에 곱한다.
- **분기 없는 디코드.** `fp4_e2m1_to_float_scaled`는 switch 없이 비트 배치로 값에 2^-14를 곱해 돌려준다. 워드의 합은 한 번에 2^14(`kFp4HalfBitsScale`)를 곱해 되돌린다. expert-batched 커널이 쓰는 디코드와 같다(LOCAL_FIXES 항목 31).
- **청크 경계를 넘지 않는다.** 디스패치가 `K % 32 == 0`만 보내고 `CHUNK_SIZE`는 32의 배수라 워드가 청크 끝을 넘지 않는다.

합산 순서(레인별로 워드 위에서 부분합, 그다음 레인 리덕션)는 expert-batched 커널과 같고 행별 커널의 단일 직렬 순회와 다르다.

### 3.2 디스패치 경로와 추가 가드

이 경로는 affine warp-shared 경로 뒤, 일반 non-affine 경로 앞에 있다. `group_size_ == 32`, `bits_ == 4`, `K % 32 == 0`인 bf16 mxfp4를 정렬·비정렬 호출 모두 받는다(affine 경로처럼 `use_sorted_rhs_schedule`을 암묵적 lhs 플래그로 넘긴다). 또한 `fast_threads_per_col`이 16 또는 `WARP_SIZE`여야 한다.

추가 가드가 있는 이유는 `MLX_ROCM_GATHER_QMV_THREADS_PER_COL`이 임의의 스레드 수를 지정할 수 있는데 인스턴스는 두 개뿐이고, 커널의 레인 리덕션은 블록의 x 차원이 `THREADS_PER_COL`과 같다고 가정하기 때문이다. 어느 쪽과도 맞지 않는 값은 커널을 찾지 못하거나 잘못 리덕션하는 대신 행별 커널에 남는다. 컴파일 시간을 억제하려고 bf16만 인스턴스화했다(gpt-oss는 bf16 활성값). f16과 f32 mxfp4, mxfp8은 행별 커널을 유지한다.

### 3.3 A/B 스위치

`MLX_ROCM_GATHER_QMV_USE_WARP=0`은 호출마다 읽혀 이 경로를 건너뛰고 행별 커널로 보낸다. 벤치마크의 "off" 쪽이자 테스트의 기준 쪽이며, 어떤 모델에서 회귀가 보일 때의 탈출구다.

### 3.4 채택하지 않은 안

- **새 커널.** warp-shared 커널은 이미 `x`를 올리고 레인 간 리덕션을 한다. mxfp4에는 안쪽 루프의 모양만 달라지면 된다.
- **f16과 f32 인스턴스화.** gpt-oss에 필요 없고 컴파일 시간만 늘린다.
- **옵트인 기본값.** 이슈의 규칙은 디코드가 호스트 편차보다 크게 개선되고 프리필이 편차 안에 있으면 기본값으로 한다. 둘 다 충족했다(6.3절).

## 4. 변경 고유의 위험

- 새 경로는 디코드의 합산 순서를 바꾸므로 출력이 행별 커널과 bf16 마지막 비트에서 다를 수 있다. 5.1절과 6.4절이 이를 한정한다.
- expert-batched 게이트를 넘지 못하는, `K % 32 == 0`인 모든 bf16 mxfp4 gather 호출(32토큰 미만의 프리필과 `B < 64`인 정렬 호출 포함)이 이제 이 경로를 탄다.

## 5. 검증

### 5.1 새 테스트

`tests/rocm_mxfp4_quant.rs`의 `mxfp4_warp_shared_gather_qmv_matches_per_row_and_reference`는 gpt-oss-20b의 전문가 레이어(전문가 32개, top 4, `K = 2880`, `shared_x` 청크 2개)를 `N = 2880`과 `N = 516`, bf16으로 돌린다. 케이스는 비정렬 1토큰(`B = 4`)과 8토큰(`B = 32`), 정렬 16토큰(`B = 64`, expert-batched 게이트 미달)이다. CPU에서 역양자화한 가중치의 전문가별 dense f32 행렬곱에 대한 기본 경로와 행별 경로의 상대 L2 오차를 보고한다.

| 케이스 | `N = 2880`, 기본 / 행별 | 달라진 출력 | `N = 516`, 기본 / 행별 | 달라진 출력 |
|---|---|---|---|---|
| 비정렬, 1토큰 | 1.6358e-3 / 1.6358e-3 | 11520개 중 0 | 1.6588e-3 / 1.6588e-3 | 2064개 중 0 |
| 비정렬, 8토큰 | 1.6609e-3 / 1.6609e-3 | 92160개 중 5 | 1.6827e-3 / 1.6827e-3 | 16512개 중 1 |
| 정렬, 16토큰 | 1.6569e-3 / 1.6569e-3 | 184320개 중 7 | 1.6606e-3 / 1.6606e-3 | 33024개 중 0 |

두 경로는 다섯 자리까지 일치하고, 기본 경로 두 번 실행은 비트 단위로 같다.

**이슈와 다른 점.** 이슈는 `N = K = 2880`에서 기본 경로와 행별 경로의 출력이 최소 1바이트 달라야 한다고 요구했다. 이는 `N = 2880` 케이스 세 개를 합쳐서는(출력 288,000개) 성립하지만, 1토큰 케이스 하나만으로는 성립하지 않는다. 합이 같은 bf16 출력으로 반올림되기 때문이다. f32 항 2880개에 걸친 두 합산 순서는 bf16 반올림 경계를 넘나드는 일이 드물다. 그래서 테스트는 `N = 2880` 케이스에서 달라진 출력 수를 누적해 0보다 큰지 단언하며, 이것으로 디스패치가 행별 커널을 벗어났음을 충분히 증명한다. 측정 실행에서 테스트가 출력한 그 수는 288,000개 중 12개다(1토큰 케이스 0, 8토큰 케이스 5, 정렬 16토큰 케이스 7). 벤치마크 페이지의 표, PR 본문, LOCAL_FIXES 항목 41도 같은 값을 적는다.

**변이 검사.** 새 디코드에서 부호 비트를 빼면 첫 케이스가 상대 L2 오차 1.211로 실패한다(행별 경로는 1.636e-3).

### 5.2 expert-batched 테스트의 변경

`tests/rocm_gather_qmm_expert_batched.rs`는 게이트가 expert-batched 커널에 닿는다는 것을, 커널을 끄고 정렬 호출의 바이트가 켠 결과와 달라야 한다고 요구하는 방식으로 증명한다. 꺼진 호출은 합산 순서가 다른 행별 커널을 돌렸다. 이 PR 이후에는 게이트를 넘지 못한 호출이 expert-batched 커널과 같은 순서로 합산하는 warp-shared 워드 경로를 타므로 바이트가 일치하여, 디스패치가 올바른데도 검사가 실패한다. 그래서 테스트는 그 호출 주위에서 새 헬퍼 `force_gather_warp_off`로 `MLX_ROCM_GATHER_QMV_USE_WARP=0`도 설정해 행별 커널을 명시적으로 고르고, 주석을 갱신했다. #2164 보고서가 위험으로 적어 둔 "디스패치 검사가 리덕션 길이에 의존한다"가 이것이다.

### 5.3 커널 트레이스

기본 디스패치로 gpt-oss-20b-MXFP4-Q4에서 `mlxcel-bench-decode`(`--prompt-tokens 512 -n 8 --warmup-tokens 1`)를 `rocprofv3 --kernel-trace --stats`로 측정했다.

| 커널 | 호출 수 | 호출당 평균 |
|---|---|---|
| `gather_qmv_expert_batched_kernel<hip_bfloat16, unsigned char, 4, 32, false, 16>`(프리필) | 144 | 12.29 ms |
| `gather_qmv_warp_shared_kernel<hip_bfloat16, unsigned char, 4, 32, false, 16>`(디코드) | 504 | 108.3 us |
| `gather_qmv_kernel`(모든 인스턴스) | 0 | |

디코드 호출은 #2106 트레이스가 행별 커널의 호출당 1.39 ms로 귀속한 504회와 같은 호출이다. 호출당 12.8배다.

## 6. 결과

### 6.1 환경

Ryzen AI MAX+ 395와 Radeon 8060S(`gfx1151`), HIP 7.15.26333, rocprofv3 1.3.5. 모든 GPU 실행은 `scripts/rocm_gpu_guard.sh`를 거쳤고, 가드 시도 15회가 모두 첫 시도에 승인되었다. 벤치마크 페이지는 기준 바이너리를 `ad844354`의 `main`에 이 변경을 더한 것으로 적었고, PR의 기준은 `6cd0139d`다.

### 6.2 디코드와 프리필

바이너리 하나, 프롬프트 512토큰, 생성 128토큰, 워밍업 20토큰, `--ignore-eos`로 `MLX_ROCM_GATHER_QMV_USE_WARP=0`(off)과 기본값(on)을 실행마다 번갈아 6라운드 돌렸다.

| 지표 | Off(중앙값) | On(중앙값) | 비 |
|---|---|---|---|
| 디코드 tok/s | 8.32 | 63.62 | 7.65배 |
| 프리필 tok/s | 511.73 | 510.67 | 0.998배 |

디코드: on의 모든 실행(63.42~63.79)이 off의 모든 실행(8.23~8.39)보다 빠르며, 128토큰 디코드가 15.3초에서 2.0초로 줄었다. 프리필: 범위가 겹치며(off 509.9~512.5, on 509.2~512.2), #2106이 양쪽에 같은 코드를 두고 측정한 약 1% 호스트 편차 안이다. 512토큰 프리필은 양쪽 모두 expert-batched 커널에 닿는다.

### 6.3 결정

디코드는 7.65배 개선되고 프리필은 0.2% 움직였으므로, 이 경로는 옵트인 없이 기본으로 켠다.

### 6.4 로짓 트레이스

`tests/fixtures/wikitext2_excerpt.txt`에 대한 `examples/logit_trace`(`w8`: 8토큰 청크마다 `B = 32` 비정렬로 실행되어 새 경로를 탄다), 같은 바이너리에서 행별 대 기본, `scripts/compare_logit_traces.py --decided 2.0`: top-1 불일치 640개 중 23개, 결정된 불일치 89개 중 0개, 불일치 지점의 최대 격차 0.250, 퍼플렉시티 행별 159.95, 기본 158.99. 모든 불일치는 기준의 상위 두 후보 격차가 0.5 미만인 곳에 있고, 스크립트의 판정은 동작 변화가 아니라 반올림 부류다. 트레이스는 커밋하지 않는다.

### 6.5 게이트

- `make verify-versions verify-kernel-dtype-keys verify-kernel-port-dispatch verify-llama-compat verify-fmt verify-rocm-overlay`, `cargo test --test dead_doc_pointers`, 두 테스트 타깃의 clippy `-D warnings`: 통과.
- `rocm_mxfp4_quant`(테스트 5개)와 `rocm_gather_qmm_expert_batched`(테스트 2개): 통과.
- 유닛의 `make verify-rocm`(`MLXCEL_ROCM_SMOKE_MODEL=models/mlx/Qwen3-0.6B-4bit`): OK, 스위트 161개, 12,188개 통과, 0개 실패, 398개 무시.
- 오케스트레이터 게이트(`16be6bf9`에서 `make verify-rocm`): 통과, 12,188개 통과, 0개 실패, 398개 무시.

검증하지 않은 것: Metal과 CUDA(이 호스트에 없음. 변경은 ROCm 오버레이와 ROCm 전용 테스트만 건드린다). wave64(CDNA)는 시험하지 않았다. 16레인 인스턴스는 두 너비 모두에서 웨이브 안에서 리덕션하고, `WARP_SIZE` 인스턴스는 라우팅 항목이 하나이고 `K >= 16384`일 때나 `MLX_ROCM_GATHER_QMV_THREADS_PER_COL`로만 선택된다.

## 7. LOCAL_FIXES 항목 41과 10

항목 41 "mxfp4 in the warp-shared gather qmv"는 행별 커널로의 낙하와 그 비용, 워드 경로, 디스패치 조건과 스위치, 인스턴스 두 개, 테스트와 변이 결과, 트레이스와 벤치마크 수치를 기록한다. 항목 10은 이제 그룹 크기 32의 bf16 mxfp4가 항목 41 이후 warp-shared 커널에 닿고, f16과 f32 mxfp4, mxfp8, `MLX_ROCM_GATHER_QMV_USE_WARP=0`은 여전히 행별 커널에 닿는다고 적는다. 항목 41은 포크 정책 문구로 끝난다: "Applies to the fork; kept in mlxcelverse under the 2026-10-06 fork policy, not proposed there."

## 8. 기술적 선택과 그 이유

- **커널이 아니라 루프의 모양을 바꾼다.** 스테이징과 리덕션은 문제가 없었고, 루프 모양이 그룹 크기 32에서 레인의 75%를 놀렸다.
- **워드 단위.** 워드는 한 그룹 안에 머무는 가장 큰 단위라 스케일 로드와 되돌림 곱셈 한 번이 값 8개를 담당한다.
- **항목 31의 디코드 재사용.** 같은 분기 없는 디코드와 상수가 이미 expert-batched 테스트를 통과했다.
- **스레드 수 가드.** 지원하지 않는 `MLX_ROCM_GATHER_QMV_THREADS_PER_COL` 값에도 동작하는 경로를 남긴다.
- **합계에 대한 바이트 검사.** 테스트는 데이터가 뒷받침하는 것(`N = 2880` 케이스 전체에서의 차이)을 단언하고, 정확성은 L2 오차와 변이 검사에 맡긴다.

## 9. 남은 위험

- **적용 범위 한계.** f16과 f32 mxfp4, mxfp8은 이 경로에 닿지 않는다. 이 호스트에는 다른 mxfp4 MoE 체크포인트가 없어 "기본값" 결정은 모델 하나에 근거한다.
- **wave64 미시험.**
- **마지막 비트 차이.** 합산 순서가 달라 디코드 출력이 이전 커널과 바이트 단위로 같지 않다. 한계는 L2 오차와 결정된 불일치 89개 중 0개다.
- **expert-batched 테스트의 디스패치 검사가 다시 환경에 의존한다.** 행별 커널을 고르려면 스위치가 필요하다.
- **기준 바이너리.** 벤치마크 페이지는 측정 바이너리의 기준을 PR 기준 `6cd0139d`가 아니라 `ad844354`로 적는다.

## 10. 배운 점

- **한 양자화 모드만 덮는 빠른 경로는 다른 모드를 가장 느린 커널에 남긴다.** 새 모드의 디코드는 프리필이 고쳐질 때까지 아무도 프로파일하지 않은 경로를 탔다.
- **유휴 레인을 그룹 크기와 대조해 센다.** 그룹 크기 64나 128에 맞춘 루프는 32에서 웨이브 대부분을 낭비한다.
- **바이트 차이 검사에는 달라질 수 있는 데이터가 필요하다.** 합이 어디서나 같은 bf16 값으로 반올림되면 "달라야 한다"는 검사가 올바른 디스패치에서도 실패하므로, 충분한 출력 수에 걸쳐 단언한다.
- **커널의 합산 순서를 바꾸면 다른 테스트의 관측 대상이 무효가 될 수 있다.** expert-batched 테스트에는 명시적 스위치가 필요했다.

## 11. 후속 작업

dense mxfp4 `qmv_warp_shared_kernel`도 같은 그룹별 레인 패턴(그룹 크기 32에서 4~15번 레인이 놂)을 가지며 여기서는 바꾸지 않았다.
