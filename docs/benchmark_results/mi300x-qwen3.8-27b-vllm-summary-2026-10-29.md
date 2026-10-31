# mlxcel vs vLLM on AMD MI300X — Qwen3.8-27B (2026-10-29)

현재 버전: 브랜치 `fix/rocm-cdna-warp-size-qwen35-mrope-batch` @ `c5ef46be` (local, push 안 함). 원자료와 프로파일 상세는 `mi300x-qwen3.8-27b-vllm-2026-10-29.md`.

## 측정 조건 (양쪽 동일)

| 항목 | 값 |
|---|---|
| GPU | AMD Instinct MI300X (gfx942, CDNA 3, wave64, 192 GB HBM3), ROCm 7.14 |
| 모델 | `Qwen/Qwen3.8-27B` BF16, 양쪽 같은 파일 (양자화 없음) |
| vLLM | `vllm/vllm-openai-rocm:v0.27.1`, `--max-num-seqs 128 --max-model-len 4096 --gpu-memory-utilization 0.9` |
| mlxcel | release build, `--parallel 128 --max-batch-size 128 -c 524288 --ignore-eos`, 환경 변수 없이 기본값 |
| **입력** | 요청당 **447~449 토큰** (chat template 적용 후, 서버가 보고한 `prompt_tokens`). 같은 영문 문장을 반복한 프롬프트 앞에 요청마다 `[번호]`를 붙여 요청 간 prefix를 공유하지 않음 |
| **출력** | 요청당 **정확히 256 토큰** (`max_tokens 256` + EOS 무시, 요청별 `usage.completion_tokens`로 모두 확인) |
| 디코딩 | greedy (`temperature 0`), thinking 끔, 스트리밍 `/v1/chat/completions` |
| 방식 | 요청 B개를 동시에 보내는 burst. B=1·4·8·16은 32개 요청, B=64는 128개, B=128은 256개 (최소 2라운드) |
| 지표 | tok/s = 생성(출력) 토큰 합 / wall time (prefill 포함). TTFT = 요청 시작부터 첫 토큰까지. TPOT = (마지막 − 첫 토큰 시각) / (출력 토큰 수 − 1). p50·p99는 해당 B의 모든 요청 기준 |

EOS 무시: vLLM은 요청 필드 `ignore_eos`, mlxcel은 서버 옵션 `--ignore-eos` (mlxcel chat 경로는 요청의 `ignore_eos`를 무시함).

## 현재 결과 (`c5ef46be`)

| B | 요청 수 | 입력 / 출력 (토큰/요청) | tok/s vLLM | tok/s mlxcel | mlxcel / vLLM | TTFT p50 (ms) vLLM / mlxcel | TTFT p99 (ms) vLLM / mlxcel | TPOT p50 (ms) vLLM / mlxcel | TPOT p99 (ms) vLLM / mlxcel |
|---:|---:|---|---:|---:|---:|---|---|---|---|
| 1 | 32 | 447 / 256 | 62.8 | 46.4 | **74%** | 87 / 173 | 89 / 177 | 15.6 / 20.9 | 15.7 / 21.0 |
| 4 | 32 | 448 / 256 | 207.3 | 161.4 | **78%** | 260 / 344 | 263 / 439 | 18.3 / 23.3 | 18.6 / 24.2 |
| 8 | 32 | 448 / 256 | 353.1 | 293.0 | **83%** | 446 / 497 | 447 / 733 | 21.0 / 25.5 | 22.0 / 26.8 |
| 16 | 32 | 448~449 / 256 | 637.8 | 497.2 | **78%** | 835 / 1,166 | 844 / 1,658 | 21.9 / 28.1 | 24.4 / 31.8 |
| 64 | 128 | 449 / 256 | 1,531.3 | 1,088.9 | **71%** | 1,889 / 2,951 | 3,235 / 5,326 | 34.3 / 47.6 | 40.0 / 56.9 |
| 128 | 256 | 449 / 256 | 1,887.2 | 1,213.7 | **64%** | 3,684 / 5,550 | 6,424 / 10,215 | 53.2 / 83.3 | 63.5 / 104.5 |

- 입력 토큰 수는 양쪽 서버가 보고한 값이 같다 (B=16에서만 vLLM 448, mlxcel 449). 출력은 양쪽 모든 요청이 256.
- mlxcel 오류 0, decode lookahead step 12,695회.
- mlxcel B=128은 같은 설정에서도 1라운드 기준 ±10% 정도 흔들린다. B=16의 TTFT p50(1,166 ms)은 직전 측정(834 ms)보다 높아 재측정 대상.

## 변화 추이 (같은 조건: 입력 447~449 / 출력 256, mlxcel tok/s와 vLLM 대비)

| B | `b7c23d45` (lookahead 전) | `6cb2bbcc` (lookahead) | `c5ef46be` (state ring, 현재) |
|---:|---:|---:|---:|
| 1 | 41.2 (66%) | 46.1 (73%) | 46.4 (74%) |
| 4 | 142.4 (69%) | 160.8 (78%) | 161.4 (78%) |
| 8 | 261.4 (74%) | 297.6 (84%) | 293.0 (83%) |
| 16 | 436.7 (68%) | 519.3 (81%) | 497.2 (78%) |
| 64 | 945.1 (62%) | 976.2 (64%) | 1,088.9 (71%) |
| 128 | 1,138.5 (60%) | 1,133.6 (60%) | 1,213.7 (64%) |

| 지표 | `b7c23d45` | `6cb2bbcc` | `c5ef46be` | vLLM |
|---|---:|---:|---:|---:|
| TTFT p50, B=1 | 207 ms | 212 ms | 173 ms | 87 ms |
| TPOT p50, B=1 | 23.6 ms | 20.9 ms | 20.9 ms | 15.6 ms |
| TPOT p50, B=64 | 57.2 ms | 54.6 ms | 47.6 ms | 34.3 ms |
| TPOT p50, B=128 | 92.1 ms | 91.4 ms | 83.3 ms | 53.2 ms |

시작점인 main (`2f5fb7a5`)은 이 하드웨어에서 출력이 깨지고 (`!`만 생성) B≥64에서 OOM으로 실패해 비교 대상이 아니다.

## 적용한 개선 (커밋 순)

| 커밋 | 분류 | 내용 | 범위 |
|---|---|---|---|
| `b6c45909` | 정확도 | hipcc로 컴파일하는 HIP 객체에 host warp size 전달. CDNA 전용 빌드에서 host 32 / device 64로 어긋나 `cumsum`·`all()`이 틀리고 출력이 `!`만 나왔다 | ROCm (CDNA) |
| `e819f35f` | decode | MRoPE full-attention 레이어를 batched로 (VLM wrapper 때문에 행별로 돌았다) | 모든 백엔드 |
| `a5b6a75f` | 메모리 | batched prefill 후 cache를 logits와 함께 evaluate (B=64 peak 176 → 42 GB, B=128 OOM 해결) | 모든 백엔드 |
| `ba920c59` | decode | GDN stacked state를 step 간 재사용 | 모든 백엔드 |
| `186ad7bd` | decode | batch rope를 vector offset 커널 하나로 (bit 동일) | ROCm |
| `84838fcf` | decode | vector SDPA 커널이 additive mask를 받도록 | ROCm |
| `24c91d0b` | decode | attention을 stacked KV 버퍼에서 (step당 커널 22,000 → 약 6,800) | 모든 백엔드 |
| `986f8c5e` | GDN | GDN recurrence를 HIP 커널 하나로 (decode·prefill) | ROCm |
| `f5fe633b` | GDN | decode의 GDN 입력 준비를 recurrence에 융합, gated RMSNorm 1커널 | ROCm |
| `b7c23d45` | GDN | 융합 커널의 head별 준비를 head당 1번(LDS 공유), state 복사 3곳 제거 | ROCm |
| `8d15a3fc` | 파이프라인 | `-c`가 만드는 slot당 KV 상한 때문에 decode lookahead가 항상 꺼져 있던 것을 수정 (context shift가 없으면 상한 직전만 동기) | 모든 모델·백엔드 |
| `5e428e84` | 파이프라인 | Qwen 3.5가 lookahead의 추측 step을 되돌릴 수 있게(rewind) 해 pipelining 대상이 되게 함. B=1·16 +12~15% | 모든 백엔드 |
| `6cb2bbcc` | 메모리 | ROCm buffer cache 상한 2 GiB(APU 기준값)를 discrete GPU에서는 적용하지 않음. lookahead와 겹치면 HBM이 가득 차 B=128이 263 tok/s로 붕괴하던 것 해결 | ROCm (discrete) |
| `c23ce6c2` | TTFT | idle 직후의 prompt-cache warm-up이 바로 다음 요청을 약 50 ms 막던 것을 해결 (warm-up 전 최대 100 ms 대기) | 모든 백엔드 |
| `629bd500` | TTFT | history 경계에서 buffer cache를 비우던 것(13~18 ms)을 discrete GPU에서 생략. 앞 커밋과 합쳐 B=1 TTFT 212 → 171 ms | ROCm (discrete) |
| `68e1f051`, `f76cb6d6` | 정리 | GDN 커널을 port 표로 (CI `verify-kernel-port-dispatch` 위반 해결), clippy | ROCm |
| `5e8f9dec` | 구조 | port 표에 CDNA 전용 칸 `.rocm_cdna` 추가 (wave64 장비에서만 선택). GDN 커널을 그 칸으로 옮겨 RDNA는 graph 유지 | ROCm |
| `c5ef46be` | decode | GDN state ring: CDNA에서 batched decode가 state를 미리 할당한 3개 버퍼에 돌아가며 기록 (step당 약 19 GB 할당 제거). B=64 +13%, B=128 +11%, 출력 동일 | ROCm (CDNA) |

넣지 않은 것 (측정 후 제외):

| 항목 | 이유 |
|---|---|
| add + RMSNorm 융합 (CDNA, Qwen 3.5) | graph와 byte 동일하게 만들었으나 TPOT 변화가 오차 수준 (B=1 20.90 → 20.82 ms) |
| CDNA in-flight 예산 8 GiB | ring 전에는 B=128 +8~10%, ring 후에는 오히려 느려짐 (B=128 약 1,079 tok/s) |
| boundary snapshot을 forward 한 번으로 | B=1 TTFT 약 31 ms 이득이지만 prompt-cache hit의 bit 단위 재현 보장이 깨짐 |
| multi-row 4-bit GEMV, prefill GDN 준비 융합 | 기존 경로보다 느림 |
| `MLX_ROCM_HIPBLASLT_TUNE=1` | B=128 행렬곱 24.5 → 24.9 ms, 효과 없음 |

## 남은 차이 (현재 버전 rocprofv3, decode step)

프로파일 작업량: 입력 447 토큰, 출력 128 토큰, B개 동시 요청 한 라운드.

| | B=1 | B=128 |
|---|---:|---:|
| step (프로파일 기준) | 21.7 ms | 63.1 ms |
| GPU 유휴 | 1% | 16% (약 10 ms) |
| 행렬곱 | 15.5 ms (GEMV, 약 3.5 TB/s) | 24.5 ms (hipBLASLt, 약 2.1 TB/s) |
| GDN recurrence | 0.7 ms | 11.6 ms |
| decode attention | 0.4 ms | 5.9 ms |
| norm·elementwise·copy | 약 5 ms | 약 7 ms |

| 순위 | 항목 | 근거 | 방향 |
|---|---|---|---|
| 1 | 큰 batch 행렬곱 | B=128에서 weight 약 51 GB를 2.1 TB/s로 읽음 (B=1 GEMV는 3.5 TB/s). 4 TB/s면 약 12 ms 감소 | CDNA용 skinny GEMM 또는 다른 커널 경로. vLLM이 같은 shape에 쓰는 커널과 비교 중 |
| 2 | B=128 GPU 유휴 약 10 ms | CPU가 step 그래프를 만드는 시간 | decode step HIP graph replay (B=1은 유휴 1%라 효과 없음) |
| 3 | decode attention 5.9 ms | KV 읽기량 대비 느림 (GQA에서 KV를 query head마다 다시 읽는 것으로 의심) | GQA를 고려한 decode attention 커널 |
| 4 | B=1 GEMV | 3.5 TB/s, 4.3 TB/s 이상이면 약 3 ms 감소 | GEMV 커널 튜닝 |
| 5 | TTFT | B=1 2배 (173 vs 87 ms). GDN prefill의 순차 recurrence, boundary snapshot의 7 토큰 forward | chunked 병렬 scan |

(이전 판의 "행렬곱은 병목이 아니다"는 B=1 기준 판단이었고, 큰 batch에는 틀렸다.)
