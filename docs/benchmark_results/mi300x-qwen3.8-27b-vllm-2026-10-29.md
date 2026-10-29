# mlxcel vs vLLM on AMD MI300X — Qwen3.8-27B (2026-10-29)

요약판은 `mi300x-qwen3.8-27b-vllm-summary-2026-10-29.md`. 이 문서는 측정 방법, 원자료, 프로파일 상세다.

## 요약

- **엔진 비교는 같은 checkpoint, 같은 정밀도, 같은 작업량으로만 했다.** `Qwen/Qwen3.8-27B` BF16(양자화 없음)을 vLLM과 mlxcel에서 같은 파일로 돌렸고, 모든 요청이 정확히 256 토큰을 생성했는지 확인했다.
- mlxcel(`c5ef46be`)의 처리량은 vLLM의 **B=1~16에서 74~83%, B=64·128에서 71%·64%** 다 (`b7c23d45`에서는 60~74%).
- TPOT은 B≤16에서 1.2~1.3배, B=64·128에서 1.4~1.6배 느리다. TTFT는 1.1~1.5배(B=1은 2.0배).
- 작업량은 요청당 입력 447~449 토큰, 출력 256 토큰.
- 측정 중 mlxcel ROCm 경로의 정확도 버그(MI300X에서 출력이 `!`만 나옴)와 B≥64 OOM을 찾아 고쳤다. 고치기 전 main은 이 하드웨어에서 정상 출력을 내지 못했으므로 비교 대상으로 쓰지 않는다.

> **측정 정정 (같은 날):** 이 문서의 이전 판은 mlxcel이 EOS에서 멈춘 상태로 측정되었다. mlxcel의 `/v1/chat/completions`는 요청 필드 `ignore_eos`를 무시하고(`src/server/routes/chat.rs`), 서버 옵션 `--ignore-eos`만 적용한다. 그 결과 vLLM은 요청당 256 토큰, mlxcel은 40~110 토큰을 생성해 작업량이 달랐다. 이전 판의 vLLM 대비 수치(19~63%)와 커밋별 tok/s 표는 무효이며 이 판에서 삭제했다.

## 환경

| 항목 | 값 |
|---|---|
| GPU | AMD Instinct MI300X VF (gfx942, CDNA 3, wave64, 304 CU, 192 GB HBM3) |
| ROCm | 7.14 |
| vLLM | `vllm/vllm-openai-rocm:v0.27.1` (docker), `--max-num-seqs 128 --max-model-len 4096 --gpu-memory-utilization 0.9` |
| mlxcel | branch `fix/rocm-cdna-warp-size-qwen35-mrope-batch` @ `c5ef46be` (이전 표는 `6cb2bbcc`, `b7c23d45`), release build, 환경 변수 없이 기본값, `MLX_ROCM_ARCHITECTURES=gfx942`, `--parallel 128 --max-batch-size 128 --max-queue-depth 256 -c 524288 --ignore-eos` (시퀀스당 4096 토큰) |
| 모델 | `Qwen/Qwen3.8-27B` (BF16, 52 GB), 양쪽 동일 파일 |

## 측정 방법

- 클라이언트: OpenAI 호환 `/v1/chat/completions` 스트리밍, 표준 라이브러리만 사용한 동일 스크립트를 양쪽에 사용.
- 프롬프트 약 448 토큰(요청마다 앞에 `[idx]`를 붙여 prefix 공유 없음), 출력 256 토큰, greedy(`temperature 0`), thinking 끔.
- EOS 무시: vLLM은 요청 필드 `ignore_eos: true`, mlxcel은 서버 옵션 `--ignore-eos`. 짧은 프롬프트("Say hi.", `max_tokens 64`)로 양쪽 모두 `finish_reason: length`, 64 토큰이 되는 것을 확인했다.
- 클라이언트가 요청별 `usage.completion_tokens`를 집계해 batch별 평균·최소를 기록하고, 256에 못 미치는 요청이 있으면 경고한다. 아래 모든 행은 평균·최소 256이다.
- batch B: 요청 B개를 동시에 보내는 burst를 반복. batch마다 최소 32개 요청, 최소 2라운드 (B=64는 128개, B=128은 256개).
- 지표
  - **tok/s**: 생성 토큰 합 / 라운드 wall time 합 (prefill 시간 포함 처리량)
  - **TTFT**: 요청 시작부터 첫 스트리밍 토큰까지
  - **TPOT**: (마지막 토큰 시각 − 첫 토큰 시각) / (생성 토큰 수 − 1), 요청별
  - p50 / p99는 해당 batch의 모든 요청에 대한 값

## 결과: 같은 checkpoint (BF16), 같은 작업량

입력은 요청당 447~449 토큰(서버가 보고한 `prompt_tokens`, 양쪽 같음; B=16만 vLLM 448, mlxcel 449), 출력은 요청당 256 토큰.

mlxcel `c5ef46be` (현재):

| B | 요청 수 | vLLM tok/s | mlxcel tok/s | mlxcel / vLLM | vLLM TTFT p50 / p99 (ms) | mlxcel TTFT p50 / p99 (ms) | vLLM TPOT p50 / p99 (ms) | mlxcel TPOT p50 / p99 (ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 32 | 62.8 | 46.4 | 74% | 87 / 89 | 173 / 177 | 15.6 / 15.7 | 20.9 / 21.0 |
| 4 | 32 | 207.3 | 161.4 | 78% | 260 / 263 | 344 / 439 | 18.3 / 18.6 | 23.3 / 24.2 |
| 8 | 32 | 353.1 | 293.0 | 83% | 446 / 447 | 497 / 733 | 21.0 / 22.0 | 25.5 / 26.8 |
| 16 | 32 | 637.8 | 497.2 | 78% | 835 / 844 | 1166 / 1658 | 21.9 / 24.4 | 28.1 / 31.8 |
| 64 | 128 | 1531.3 | 1088.9 | 71% | 1889 / 3235 | 2951 / 5326 | 34.3 / 40.0 | 47.6 / 56.9 |
| 128 | 256 | 1887.2 | 1213.7 | 64% | 3684 / 6424 | 5550 / 10215 | 53.2 / 63.5 | 83.3 / 104.5 |

mlxcel: 오류 0, 모든 요청 256 토큰, decode lookahead step 12,695회. B=16 TTFT p50은 `6cb2bbcc` 측정(834 ms)보다 높아 재측정 대상이다.

mlxcel `6cb2bbcc`:

| B | vLLM tok/s | mlxcel tok/s | mlxcel / vLLM | vLLM TTFT p50 / p99 (ms) | mlxcel TTFT p50 / p99 (ms) | vLLM TPOT p50 / p99 (ms) | mlxcel TPOT p50 / p99 (ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 62.8 | 46.1 | 73% | 87 / 89 | 212 / 269 | 15.6 / 15.7 | 20.9 / 20.9 |
| 4 | 207.3 | 160.8 | 78% | 260 / 263 | 368 / 379 | 18.3 / 18.6 | 23.5 / 23.5 |
| 8 | 353.1 | 297.6 | 84% | 446 / 447 | 525 / 685 | 21.0 / 22.0 | 24.9 / 25.5 |
| 16 | 637.8 | 519.3 | 81% | 835 / 844 | 834 / 1294 | 21.9 / 24.4 | 27.6 / 29.4 |
| 64 | 1531.3 | 976.2 | 64% | 1889 / 3235 | 2733 / 5056 | 34.4 / 40.0 | 54.6 / 63.6 |
| 128 | 1887.2 | 1133.6 | 60% | 3684 / 6424 | 5327 / 10082 | 53.2 / 63.5 | 91.4 / 110.0 |

mlxcel: 오류 0, 모든 요청 256 토큰, decode lookahead step 12,695회. 같은 날 vLLM을 한 번 더 측정한 값(63.0 / 208.6 / 354.3 / 642.0 / 1540.9 / 1910.0 tok/s)과의 차이는 1.5% 이내다. mlxcel B=128은 같은 설정에서도 1라운드 기준 951~1,138 tok/s로 흔들렸다.

mlxcel `b7c23d45` (이전):

| B | vLLM tok/s | mlxcel tok/s | mlxcel / vLLM | vLLM TTFT p50 / p99 (ms) | mlxcel TTFT p50 / p99 (ms) | vLLM TPOT p50 / p99 (ms) | mlxcel TPOT p50 / p99 (ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 62.8 | 41.2 | 66% | 87 / 89 | 207 / 262 | 15.6 / 15.7 | 23.6 / 23.6 |
| 4 | 207.3 | 142.4 | 69% | 260 / 263 | 356 / 366 | 18.3 / 18.6 | 26.8 / 26.9 |
| 8 | 353.1 | 261.4 | 74% | 446 / 447 | 506 / 657 | 21.0 / 22.0 | 28.7 / 29.4 |
| 16 | 637.8 | 436.7 | 68% | 835 / 844 | 977 / 1883 | 21.9 / 24.4 | 32.3 / 36.9 |
| 64 | 1531.3 | 945.1 | 62% | 1889 / 3235 | 2604 / 4858 | 34.4 / 40.0 | 57.2 / 66.1 |
| 128 | 1887.2 | 1138.5 | 60% | 3684 / 6424 | 5030 / 9698 | 53.2 / 63.5 | 92.1 / 110.9 |

mlxcel: 오류 0.

mlxcel `b7c23d45`, prompt-cache boundary snapshot 끔 (`MLXCEL_DISABLE_BOUNDARY_SNAPSHOT=1`, 그 외 동일):

| B | tok/s | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) |
|---:|---:|---:|---:|
| 1 | 41.8 | 133 / 135 | 23.5 / 23.6 |
| 4 | 142.5 | 338 / 381 | 26.8 / 27.7 |
| 8 | 261.6 | 493 / 669 | 28.7 / 29.9 |
| 16 | 448.6 | 880 / 1435 | 32.3 / 35.3 |
| 64 | 948.3 | 2541 / 4882 | 57.3 / 66.5 |
| 128 | 1136.3 | 4986 / 9669 | 92.6 / 111.3 |

관찰:

- 행렬곱은 차이의 원인이 아니다. B=1에서 mlxcel의 GEMV는 step당 15.3 ms로 54 GB를 약 3.5 TB/s에 읽어, vLLM의 step 전체(15.6 ms)와 같다.
- boundary snapshot은 요청마다 prefill forward를 한 번 더 돌린다 (history 경계 441 토큰 + 나머지 7 토큰). B=1 TTFT에서는 크지만 (207 → 133 ms), 큰 batch 처리량에는 거의 영향이 없다.

## mlxcel 자체 개선 (같은 엔진, 같은 checkpoint)

커밋별 효과는 decode step 시간으로 확인했다 (출력 길이와 무관한 지표). 커밋별 tok/s는 EOS 문제로 작업량이 일정하지 않아 싣지 않는다.

4-bit (`mlx-community/Qwen3.8-27B-4bit`), B=64, steady state decode 1 step의 median:

| 커밋 | step | 내용 |
|---|---:|---|
| `a5b6a75f` | 225 ms | (기준) |
| `ba920c59` | 195 ms | GDN state를 step 간 재사용 |
| `186ad7bd` | 171 ms | ROCm batch rope 1커널 |
| `24c91d0b` | 140 ms | stacked KV + mask 지원 vector SDPA (step당 커널 22,000 → 약 6,800) |

BF16, B=128, 순수 decode step: `f5fe633b` 89.4 ms → `b7c23d45` 70.6 ms (GDN 커널 26.8 → 9.2 ms).

decode lookahead (`8d15a3fc`, `5e428e84`, `6cb2bbcc`). 같은 test-fast 바이너리, BF16, 447 토큰 프롬프트, 256 토큰, `MLXCEL_FORCE_SYNC=1`(동기) 대 lookahead:

| B | 동기 tok/s | lookahead tok/s | 비고 |
|---:|---:|---:|---|
| 1 | 41.2 | 46.0 | TPOT 23.5 → 21.0 ms |
| 16 | 449.1 | 514.4 | TPOT 32.3 → 27.7 ms |
| 64 | 941.0 | 852.9 → 959.9 | cache 상한 2 GiB → 없음 |
| 128 | 1129.0 | 262.7 → 1129.0 | cache 상한 2 GiB → 없음 |

- `-c`로 띄운 서버는 slot당 `max_kv_size`가 생기고, 스케줄러가 이 값이 있으면 lookahead를 무조건 껐다 (모든 모델). `8d15a3fc` 전에는 lookahead가 한 번도 돌지 않았다.
- cache 상한 2 GiB에서 lookahead B=128은 HBM이 가득 차(rocm-smi 206 GB, MLX active 122 GB) `hipMalloc` 한 번이 최대 2.9초 걸렸다 (rocprofv3 HIP API trace: hipMalloc 합계 42.7 s, 동기는 0.46 s).
- B=128에서는 lookahead를 켜도 GPU 유휴가 31%(동기)·36%(lookahead)로 남는다. ROCm in-flight 예산(1 GiB)을 step 할당량이 넘어 그래프 생성 도중 GPU를 기다리기 때문이다. 예산 sweep (lookahead, cache 상한 없음, B=128): 1 GiB 1,120~1,138, 4 GiB 1,180, 8 GiB 1,222~1,236, 16 GiB 1,036, 32 GiB 753~757 tok/s. 기본값 변경은 메모리 추정기와 함께 바꿔야 해서 이번에는 넣지 않았다.

GDN state ring (`c5ef46be`). 같은 test-fast 바이너리, BF16, 입력 447 토큰, 출력 256 토큰, `MLXCEL_GDN_STATE_RING=0` 대 기본값:

| B | ring 끔 tok/s (TPOT p50) | ring 켬 tok/s (TPOT p50) |
|---:|---:|---:|
| 1 | 46.5 (20.91 ms) | 46.6 (20.87 ms) |
| 16 | 515.6 (27.79 ms) | 511.5 (27.98 ms) |
| 64 | 959.3 (55.77 ms) | 1083.9 (47.53 ms) |
| 128 | 1088.5 (92.51 ms) | 1209.6 (81.64 ms) |

greedy 출력은 ring 켬/끔이 단독·동시·시차 요청 모두 8/8 같다. ring 이후 B=128 decode step(rocprofv3, 입력 447 / 출력 128): wall 63.1 ms, GPU 유휴 16% (ring 전 31~36%), 행렬곱 24.5 ms, GDN 11.6 ms, decode attention 5.9 ms. B=1: wall 21.7 ms, 유휴 1%, GEMV 15.5 ms.

## 브랜치 커밋

| 커밋 | 내용 | 범위 |
|---|---|---|
| `b6c45909` | `fix(rocm)`: hipcc로 컴파일하는 HIP 객체에 `MLX_HOST_WARP_SIZE` 전달. 없으면 CDNA 전용 빌드에서 host는 warp 32, device는 64로 어긋나 `cumsum`·`all()`이 틀리고 Qwen3.8 출력이 `!`만 나옴 | ROCm, CDNA 전용 빌드 |
| `e819f35f` | `perf(qwen3_5)`: MRoPE full-attention 레이어를 batched로 | 모든 백엔드 |
| `a5b6a75f` | `fix(qwen3_5)`: batched prefill 후 cache를 logits와 함께 evaluate (B=64 burst peak 176 → 42 GB, B=128 OOM 해결) | 모든 백엔드 |
| `ba920c59` | `perf(qwen3_5)`: stacked GDN state를 step 간 재사용 | 모든 백엔드 |
| `186ad7bd` | `perf(core)`: ROCm decode batch rope를 vector offset 1커널로 (bit 동일) | ROCm |
| `84838fcf` | `perf(rocm)`: vector SDPA 커널이 additive mask를 받도록 | ROCm |
| `24c91d0b` | `perf(qwen3_5)`: batched decode attention을 stacked KV 버퍼에서 | 모든 백엔드 |
| `986f8c5e` | `perf(rocm)`: GDN recurrence를 HIP 커널 하나로 (decode·prefill) | ROCm |
| `f5fe633b` | `perf(rocm)`: decode GDN 입력 준비 융합, gated RMSNorm 6커널 → 1커널 (gated RMSNorm은 bit 동일) | ROCm |
| `b7c23d45` | `perf(rocm)`: 융합 커널의 준비 연산을 head당 1번으로, GDN state 복사 3곳 제거 | ROCm (복사 제거는 모든 백엔드) |
| `8d15a3fc` | `fix(scheduler)`: KV 상한이 생성 중단 기준일 뿐이면 decode lookahead 유지 | 모든 모델·백엔드 |
| `5e428e84` | `perf(qwen3_5)`: lookahead의 추측 decode step을 되돌리는 rewind 지원 | 모든 백엔드 |
| `6cb2bbcc` | `fix(rocm)`: discrete GPU에서는 MLX buffer cache 기본값 유지 (2 GiB 상한은 APU에만) | ROCm |
| `c23ce6c2` | `perf(scheduler)`: idle prompt-cache warm-up 전에 최대 100 ms 요청 대기 (직후 요청이 warm-up forward를 기다리던 것) | 모든 백엔드 |
| `629bd500` | `perf(prefill)`: discrete GPU에서 history 경계의 buffer cache 비우기 생략 | ROCm (discrete) |
| `68e1f051` | `fix(rocm)`: GDN HIP 커널을 port 표로 (`verify-kernel-port-dispatch` 위반 해결) | ROCm |
| `f76cb6d6` | `style(runtime)`: clippy | |
| `5e8f9dec` | `feat(rocm)`: CDNA 전용 port 칸 `.rocm_cdna`, GDN 커널을 그 칸으로 | ROCm |
| `c5ef46be` | `perf(qwen3_5)`: batched decode의 GDN state를 CDNA state ring에 기록 (step당 약 19 GB 할당 제거) | ROCm (CDNA) |

## 비교하지 않은 것

- **FP8 (`Qwen/Qwen3.8-27B-FP8`)**: mlxcel은 이 파일을 로드할 때 MLX `mxfp8`로 재양자화하고, vLLM은 E4M3 block scale 그대로 FP8 행렬 코어를 쓴다. 입력 파일은 같아도 연산 형식이 달라 같은 조건의 엔진 비교가 아니다. mlxcel ROCm mxfp8 경로는 batch마다 weight를 다시 읽는 per-row GEMV를 탄다 (B=1 decode step 약 106 ms).
- **MLX 4-bit와 vLLM의 비교**: checkpoint가 달라 하지 않았다.
- **gpt-oss-120b**: Qwen 분석과 개선을 먼저 하기로 해서 측정하지 않았다.
- **고치기 전 main(`2f5fb7a5`)**: 이 하드웨어에서 출력이 깨지고(`!`만 생성) B≥64에서 OOM으로 요청이 실패했다.

## 남은 차이의 원인 (BF16 프로파일 기준)

`rocprofv3 --kernel-trace`로 decode만 있는 step을 나눈 결과다 (프로파일러 오버헤드 포함).

| | B=16 (`f5fe633b`) | B=64 (`f5fe633b`) | B=128 (`f5fe633b`) | B=128 (`b7c23d45`) |
|---|---:|---:|---:|---:|
| step wall | 36.3 ms | 57.1 ms | 89.4 ms | 70.6 ms |
| 행렬곱 | 17.3 ms | 19.6 ms | 24.5 ms | 24.5 ms |
| GDN decode 커널 | 4.5 ms | 13.4 ms | 26.8 ms | 9.2 ms |
| attention | 0.8 ms | 2.6 ms | 4.8 ms | 4.4 ms |
| norm·elementwise·copy | 5.2 ms | 6.8 ms | 8.6 ms | 8.3 ms |
| GPU idle (CPU 쪽 시간) | 8.4 ms | 13.2 ms | 20.6 ms | 21.2 ms |
| step당 커널 수 | 1,429 | 1,429 | 1,429 | 1,429 |

1. **CPU 측 오버헤드.** GPU가 step의 23~30%를 쉬고, 그 시간이 batch에 비례해 늘어난다. batched decode가 레이어·행마다 FFI로 identity 확인, view 생성, 위치 계산을 하고, 매 step 그래프를 새로 만들어 커널을 하나씩 launch한다. vLLM은 decode step을 HIP graph로 캡처해 재실행한다.
2. **연산 융합 부족.** RMSNorm, residual add, gate sigmoid/multiply, dtype 변환 copy가 각각 별도 커널이다. residual add + RMSNorm은 Qwen 3.5 모델 코드가 fused helper를 쓰지 않고, ROCm fused add-RMSNorm 커널은 wave32에서만 허용된다.
3. **prefill.** B=128 burst에서 prefill(약 12~16 s)이 먼저 처리되고 decode(약 19~20 s)가 뒤에 돈다. prefill 처리량은 mlxcel 약 3.5~5k tok/s로 vLLM(약 9k tok/s, TTFT로 추정)보다 낮다. 원인: boundary snapshot의 추가 forward(prefill 시간의 18%), 토큰을 하나씩 처리하는 GDN prefill 커널(약 25%), 작은 커널 약 3만 개와 GPU idle 약 20%.
4. 양자화 경로(4-bit, mxfp8)는 이와 별개로 GEMV/GEMM 자체가 RDNA 기준 튜닝이다 (4-bit GEMV는 피크 대역폭의 약 12%). BF16 비교에는 해당하지 않는다.

개선 방향 제안은 이슈 초안(백엔드별 hot path를 공통 interface 뒤에 두는 구조)에 정리했다.
