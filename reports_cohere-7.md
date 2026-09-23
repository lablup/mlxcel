# command-r7b prefill/decode 측정과 decode 최적화 보고서

작성일: 2026-09-21. 대상 체크포인트: `c4ai-command-r7b-12-2024-4bit` (Cohere2ForCausalLM, 32 레이어, 4-bit group 64, 4.516 GB). 모든 측정은 이 보고서에 적힌 조건의 M1 Ultra 한 대에서 수행했다.

## 1. 요약

- 출발점: HEAD `0bbfa95d`에서 pp512/tg128 기준 prefill 약 695 tok/s, decode 약 102 tok/s. 09-06 같은 조건 기록(107.13)보다 decode가 약 4.7% 낮았다.
- 하락 원인: 코드 변경이 아니다(09-06 측정 커밋 `30ab5a39`를 오늘 다시 빌드해도 같은 값). 배경 부하도 아니다(인덱서를 멈춘 상태에서도 102.8). 09-10에 설치된 macOS 27 쪽이 가장 유력하지만 재현할 방법이 없어 추정으로 남긴다.
- 120 tok/s 가능성: 이 기기에서 attention·norm·RoPE·KV 캐시를 모두 뺀 matvec 체인만의 바닥이 118.5 tok/s다. 토큰당 4.516 GB를 읽는 한 그래프나 커널 개선만으로는 120에 닿지 않는다.
- 찾아서 고친 것: MLX command buffer 입력 예산(`MLX_MAX_MB_PER_BUFFER`)이 M1~M4 decode에서 토큰을 약 23개 command buffer로 쪼개고 있었다. decode 구간에서만 예산을 1000으로 올리는 런타임 전환을 구현했다. cohere2의 잔차 덧셈 두 개도 커널 하나로 합쳤다.
- 결과: command-r7b decode 102.69 → 109.88 tok/s (+7.0%, ABBA 8쌍 중앙값). prefill 속도, prefill peak 메모리, temp 0 생성 결과는 모두 그대로다. 예열 없는 단발 `mlxcel generate`에서는 104.6 → 114.0 (+9%)이다. 다른 계열 decode는 변화 없음(Gemma 3 4B) ~ +21%.
- 후속 단계(13절): 서버 decode는 lookahead(파이프라인) 경로에만 예산을 올리도록 바로잡았고(동기 경로 회귀 수정), 잔차 덧셈과 다음 LayerNorm을 바이트 동일한 커널 하나로 합쳤다(+1.1%). RoPE+캐시 쓰기 융합(상한 +1.2%)과 matvec 커널 튜닝(비트 동일 설정에서 약 1%)은 이득이 작아 반영하지 않았다.
- 최종(`main` 대비 HEAD): CLI 벤치 102.75 → 111.08 tok/s (+8.1%), 서버 일반 요청 약 107.7 → 약 118.4 tok/s (+10%). prefill과 메모리는 그대로.
- 커밋: `rust/mlxcel-2` 브랜치 `perf/cohere2-decode-m1-ultra`에 `793d69f5` (cohere2 add 융합), `cdaa94b8` (decode 전용 예산 전환), `c567ff61` (잔차 덧셈 + LayerNorm 융합). push와 PR은 하지 않았다.

## 2. 측정 환경

| 항목 | 값 |
|---|---|
| 기기 | Mac Studio, Apple M1 Ultra, 128 GB 통합 메모리 |
| OS | macOS 27.0 (26A428), Xcode 27 RC. 09-10 설치, 09-14 재부팅 |
| 기준 코드 | `main` `0bbfa95d`, MLX 핀 `81ba1c6a` |
| 빌드 | `cargo build --release --features metal,accelerate` |
| 측정 도구 | `scripts/bench_decode.sh` 및 같은 프로세스 벤치 `mlxcel-bench-decode` (`--prompt-tokens N -n 128 --ignore-eos --warmup-tokens 20`) |
| 배경 부하 (오후 초반) | 사진 앱 라이브러리 분석: `mediaanalysisd` CPU 60~160%, `suggestd` 약 95%, `com.apple.photos.ImageConversionService` GPU 시간 3.2~5%. load average 약 4. Time Machine 미실행 |
| 배경 부하 (저녁 최종 측정) | `mediaanalysisd` CPU 0%, load average 3.3 |

측정 규칙: 비교는 같은 조건에서 번갈아(ABBA) 실행하고, 대표값은 중앙값과 범위로 적는다. 빌드 직후 첫 실행은 버렸다. 벤치 중에는 다른 cargo나 벤치를 돌리지 않았다.

## 3. 1차 측정: prefill/decode (HEAD `0bbfa95d`)

| prompt 길이 | 회수 | prefill tok/s | decode tok/s (128 토큰 생성) |
|---|---|---|---|
| 512 | 7 | 689.1 ~ 703.1 (중앙값 699.1) | 99.9 ~ 102.1 (5회는 101.5~102.1, 2회는 약 100) |
| 2048 | 3 | 688.1 ~ 691.7 | 92.6 ~ 93.1 |
| 4096 | 3 | 664.6 ~ 668.4 | 82.9 ~ 83.9 |

과거 기록과의 비교:

| 기록 | 조건 | prefill | decode |
|---|---|---|---|
| 09-06 mlxcel `30ab5a39` (M1 Ultra) | pp512/tg128 | 694.92 | 107.13 |
| 09-06 mlx-lm 0.31.3 (M1 Ultra) | pp512/tg128 | 776.14 | 96.28 |
| 08-19 mlxcel 0.5.2 (M1 Ultra) | 8토큰 prompt, 100토큰 생성 (구 조건) | 102.72 | 113.12 |
| 09-06 mlxcel (M5 Max) | pp512/tg128 | 3256.89 | 112.63 |

같은 날 3257 tok/s와 695 tok/s가 섞여 보였던 것은 M5 Max와 M1 Ultra 기록이 같은 파일군에 있었기 때문이다. 08-19의 113 tok/s는 짧은 prompt 조건이라 pp512 결과와 직접 비교할 수 없다.

## 4. decode 하락 원인 조사

### 4.1 코드 변경 A/B

09-06 측정 커밋 `30ab5a39`(옛 MLX 핀 `9a795735`)를 `rust/mlxcel-2`의 worktree `rust/mlxcel-2-old`에 따로 빌드했다(mlxcel-2의 새 핀 빌드 캐시를 보존하기 위해). 오늘 같은 환경에서 HEAD와 ABBA로 각 6회 측정했다.

| 빌드 | decode tok/s | prefill tok/s |
|---|---|---|
| `30ab5a39` (MLX `9a795735`) | 102.11 ~ 103.97 (중앙값 약 102.9) | 687.8 ~ 699.1 |
| `0bbfa95d` (MLX `81ba1c6a`) | 102.34 ~ 103.12 (중앙값 약 102.7) | 684.3 ~ 697.4 |

차이 0.2%로 잡음 범위다. 09-06 이후 커밋 226개(MLX 핀 교체 #1772 포함)는 원인이 아니다.

### 4.2 환경

- 소프트웨어 업데이트 기록: macOS 27과 Command Line Tools for Xcode 27.0이 09-10에 설치됐다. 09-06 기준선 이후다.
- 배경 부하: 측정 중 사진 분석 프로세스가 GPU 시간 3.2%를 썼다. 7회 중 2회가 약 100으로 떨어진 것은 이 경합으로 보인다.
- 최종 확인: 저녁에 `scripts/with_indexers_paused.sh`로 사진·Spotlight 데몬과 `suggestd`를 멈춘 상태에서도 기준 빌드는 102.77 (102.23~103.18)이었다. 배경 부하는 하락 원인이 아니다.
- 결론: 남는 후보는 macOS 27 / Xcode 27 (Metal 드라이버·셰이더 컴파일러). 이전 OS로 되돌려 재현할 수 없어 추정으로 둔다.

## 5. 120 tok/s 가능성: 대역폭 roofline

실제 체크포인트 가중치(토큰당 4.516 GB)를 Python MLX 0.32.2(두 MLX 핀 사이 시기의 버전)로 올려 조건별 바닥을 쟀다. 8 토큰을 연결해 한 번에 eval하고, 그래프 구성은 타이머 밖에서 했다(`roofline.py`).

| 조건 | 토큰당 시간 | 실효 대역폭 | tok/s 환산 |
|---|---|---|---|
| 순수 읽기 상한 (`mx.sum`, 4.5 GB 버퍼 하나) | 6.3 ms | 717 GB/s | 159 |
| 모든 matvec을 의존성 없이 실행 | 7.28 ms | 620.7 GB/s | 137.5 |
| cohere2 의존 순서의 matvec + swiglu 체인 (attention·norm·RoPE·캐시 제외) | 8.44 ms | 535.1 GB/s | 118.5 |
| 위 체인 + q\|k\|v\|gate\|up 입력 투영과 o\|down 출력 투영을 합침 | 8.16 ms | 553.6 GB/s | 122.6 |
| 직접 짠 벡터 읽기 커널 (참고용, 비효율) | 8.0 ms | 564.7 GB/s | 125.0 |

- 120 tok/s는 토큰당 8.33 ms이고, attention 등을 뺀 체인만으로 8.44 ms가 걸린다.
- 의존성 없는 경우(137.5)와 체인(118.5)의 차이 16%는 레이어 간 의존 직렬화 비용이다. 단일 스트림 decode에서는 피할 수 없다.
- 이 측정으로 08-19 메모리의 "mlxcel이 자기 matvec 복제본보다 7% 빠르다"(9448 us 바닥)는 결론을 바로잡았다. 그때 바닥은 GPU 바닥이 아니었다. 다만 "120은 바이트를 줄이거나 speculative로만 가능하다"는 결론은 이번 측정으로도 유지된다.

## 6. decode 병목 분석

### 6.1 파이프라인 분해 (`MLXCEL_PROFILE_PIPELINE_DETAIL=1`)

측정 구간 127 토큰: 그래프 구성(forward) 0.369 ms, 샘플러 0.010 ms, `async_eval` 9.224 ms, `item` 대기 0.048 ms (토큰당). decode 루프는 mlx-lm과 같이 평가 전 토큰 위에 다음 그래프를 쌓는다. `async_eval`의 9.2 ms는 MLX가 GPU 진행을 기다리는 역압이라, 그래프 구성 0.37 ms는 GPU 실행 뒤에 숨는다. 루프 구조는 개선 대상이 아니다.

### 6.2 decode 한 토큰의 그래프 (`MLXCEL_EXPORT_DECODE_DOT`)

- prompt 500 토큰 기준 노드 872개: QuantizedMatmul 161, Slice 160, Reshape 131, Transpose 128, Add 65, SliceUpdate 64, RoPE 48 (슬라이딩 레이어 24개 × q,k), LayerNorm 33, SDPA 32, compiled swiglu 32, 샘플러 관련 소수.
- prompt 512에서는 노드가 1064개였다. 추가된 Concatenate 64, Full 65, Broadcast 66은 KV 캐시가 256 단위로 커지는 토큰에서만 생긴다. 토큰당 비용이 아니다.
- 로짓 dtype은 f16이고, `AsType` 2개는 샘플러 안의 작은 노드다.

### 6.3 의존 단계(barrier) 비용

MLX는 앞 커널의 출력을 읽는 커널 앞에 메모리 barrier를 넣는다. matvec 체인에 레이어마다 의존 커널을 인위로 추가해 단계당 비용을 쟀다.

| 레이어당 추가 단계 | 토큰당 시간 | 단계당 비용 |
|---|---|---|
| 0 | 8.354 ms | |
| 2 | 8.599 ms | 3.82 us |
| 4 | 8.830 ms | 3.72 us |
| 8 | 9.110 ms | 2.95 us |

레이어당 단계 하나를 줄이면 토큰당 약 0.12 ms(약 1.2%)가 빠진다.

### 6.4 Python 구조 복제본과 비교

cohere2 decode를 그대로 옮긴 Python 복제본(`replica.py`: LayerNorm → fused qkv → RoPE(슬라이딩 레이어) → slice_update KV 캐시 → SDPA → o_proj, gate/up → swiglu → down, 잔차, tied lm_head, logit_scale, argmax)을 mlxcel과 같은 루프 방식으로 쟀다.

| 컨텍스트 | mlxcel decode | 복제본 decode | 차이 |
|---|---|---|---|
| 16 | 104.43 / 105.25 tok/s | 109.2 tok/s (9.155 ms) | 약 0.37 ms |
| 500 | 104.41 tok/s (9.58 ms) | 108.5 tok/s (9.221 ms) | 약 0.36 ms |
| 2000 | 92.62 / 92.46 tok/s | 96.3 tok/s (10.386 ms) | 약 0.42 ms |

그래프 프리미티브는 1:1로 같은데 mlxcel만 컨텍스트와 무관하게 토큰당 약 0.37 ms가 더 들었다. 다음 항목을 확인해 배제했다.

- 가중치 적재 방식: mlxcel도 Python `mx.load`와 같은 MLX `load_safetensors`를 쓴다 (mmap no-copy 경로는 gemma4 전용).
- MLX 빌드: Release, `MLX_METAL_JIT=OFF`로 Python wheel과 같다. Metal overlay 3개(`compiled.cpp`, `kernels/utils.h`, `quantized.cpp`) 중 `quantized.cpp`는 M1 세대에서 커널 선택에 영향이 없다.
- 캐시 비우기 주기: 256 토큰마다라 128 토큰 측정에는 걸리지 않는다.
- dtype: 로짓과 잔차는 f16이다.

이 차이의 원인은 다음 절의 command buffer 분할이었다.

### 6.5 GPU 타임라인 (xctrace)

- `xcrun xctrace record --template "Metal System Trace" --launch`는 헤드리스 환경에서 대상 프로세스가 CPU 0%로 멈춰 쓸 수 없었다(5분 후 직접 종료).
- 실행 중인 프로세스에 `--attach`로 2초 기록하는 방식은 동작했다. `metal-gpu-intervals` 표를 내보내 command buffer 단위 실행 구간을 합쳤다.

| 대상 | GPU 가동률 | command buffer 사이 빈틈 중앙값 | p90 |
|---|---|---|---|
| mlxcel decode | 89.5% | 42.8 us | 52.2 us |
| Python 복제본 decode | 88.1% | 41.8 us | 49.3 us |

토큰당 command buffer가 약 23개였고, 경계마다 GPU가 약 40 us씩 쉬었다. 참고로 `ioreg`의 프로세스별 `accumulatedGPUTime`은 앱 프로세스(사진 서비스 등)에만 기록되고 CLI 프로세스에는 남지 않아 mlxcel 측정에는 쓸 수 없었다.

## 7. command buffer 입력 예산 (`MLX_MAX_MB_PER_BUFFER`)

### 7.1 동작 원리

MLX는 두 조건 중 하나라도 넘으면 Metal command buffer를 commit한다 (`CommandEncoder::needs_commit`, `mlx/backend/metal/device.cpp`).

```cpp
return (buffer_ops_ > max_ops) || ((buffer_sizes_ >> 20) > max_mb);
```

- `buffer_sizes_`는 서로 다른 입력 배열의 `data_size()` 합이다. 바이트가 아니라 원소 수다.
- MLX 기본값: Ultra와 Max는 연산 50 / 예산 50, base·Pro는 40 / 40.
- mlxcel은 M1~M4에서 연산 한도만 1000으로 올린다(#353, Gemma3n 대응). 그래서 예산 50(약 5200만 원소)이 decode에서 먼저 걸린다. 4-bit 7B 모델은 packed u32 원소가 약 11억 개이고 decode는 토큰마다 전부 읽으므로, 레이어 1~2개마다 buffer가 끊긴다. bf16 체크포인트는 원소 수가 8배라 더 잘게 끊긴다.

### 7.2 한도 조합 첫 스윕 (command-r7b, decode tok/s, 3회)

| 설정 | decode |
|---|---|
| mlxcel 기본 (연산 1000, 예산 50) | 100.80 / 102.49 / 102.87 |
| 예산 100 | 108.51 / 108.98 / 109.51 |
| 예산 200 | 107.43 / 107.66 / 107.81 |
| 예산 1000 | 108.46 / 108.77 / 109.57 |
| 연산 25, 예산 1000 | 104.26 / 104.67 / 105.37 |
| 연산 100, 예산 1000 | 106.35 / 107.83 / 108.45 |
| 연산 200, 예산 1000 | 109.51 / 109.56 / 109.95 |
| 연산 1000, 예산 1000 | 108.10 / 109.65 / 110.28 |
| 연산 5000, 예산 5000 | 108.72 / 109.99 / 110.92 |

Python 복제본(MLX wheel 기본 연산 50)은 예산 50에서 107.9~108.4, 예산 1000에서 110.7, 연산·예산 모두 1000에서 108.0이었다.

### 7.3 예산 세분 스윕 (command-r7b, 연산 1000 고정, 3회)

| 예산 | prefill tok/s | decode tok/s |
|---|---|---|
| 50 (기본) | 665.2 ~ 668.0 | 101.38 ~ 103.25 |
| 100 | 661.0 ~ 676.1 | 107.42 ~ 109.19 |
| 200 | 657.0 ~ 678.4 | 106.39 ~ 107.49 |
| 400 | 654.9 ~ 670.6 | 107.31 ~ 109.64 |
| 1000 | 655.8 ~ 675.1 | 109.40 ~ 110.33 |
| 4000 | 638.2 ~ 653.8 | 109.16 ~ 109.92 |
| 100000 | 649.7 ~ 654.0 | 108.69 ~ 109.56 |

decode는 1000에서 포화하고, prefill은 4000 이상에서 2~3% 떨어진다.

### 7.4 계열별 교차 검증 (기본 대 1000, pp500/tg128, 3회)

| 체크포인트 | decode 기본 | decode 1000 | prefill 기본 | prefill 1000 |
|---|---|---|---|---|
| command-r7b 4bit | 101.4 ~ 103.3 | 109.4 ~ 110.3 (+7%) | 665 ~ 668 | 656 ~ 675 |
| Llama 3.1 8B Instruct 4bit | 99.0 ~ 99.2 | 103.4 ~ 105.1 (+5.6%) | 713 ~ 716 | 696 ~ 700 |
| Qwen2.5 7B Instruct 4bit | 98.2 ~ 100.1 | 106.4 ~ 108.1 (+8%) | 754 ~ 762 | 722 ~ 745 |
| Gemma 3n E4B 4bit | 59.7 ~ 60.6 | 62.1 ~ 62.4 (+3%) | 759 ~ 769 | 735 ~ 762 |
| Gemma 3 4B 4bit | 95.6 ~ 97.3 | 96.3 ~ 96.9 (변화 없음) | 917 ~ 935 | 913 ~ 953 |
| Granite 4.0 H Tiny 4bit | 103.3 ~ 104.0 | 113.4 ~ 113.8 (+10%) | 1605 ~ 1612 | 1607 ~ 1620 |
| Qwen3-30B-A3B 4bit | 72.9 ~ 76.0 | 90.0 ~ 90.4 (+20%) | 837 ~ 845 | 780 ~ 851 |
| Mixtral 8x7B Instruct 4bit | 51.0 ~ 52.1 | 61.9 ~ 62.8 (+21%) | 323 ~ 326 | 320 ~ 327 |
| Llama 3.1 8B Instruct bf16 | 34.2 ~ 34.7 | 40.1 ~ 40.4 (+17%) | 769 ~ 777 | 751 ~ 755 |

decode가 느려진 칸은 없다. prefill은 4-bit에서 변화 없음 ~ -3.3%, bf16에서 -2.7%였다.

### 7.5 절충값 재측정 (Llama 3.1 8B, Qwen2.5 7B, 두 prompt 길이, 3회)

| 체크포인트 | prompt | 예산 50 / 200 / 400 / 1000: prefill | decode |
|---|---|---|---|
| Llama 3.1 8B 4bit | 500 | 717~718 / 714~723 / 691~714 / 715~721 | 96.3~99.6 / 100.3~103.5 / 103.6~104.9 / 103.6~105.6 |
| Llama 3.1 8B 4bit | 2048 | 751~759 / 748~751 / 737~742 / 742~755 | 91.3~93.0 / 90.1~90.9 / 92.5~93.0 / 92.5~93.9 |
| Qwen2.5 7B 4bit | 500 | 747~762 / 738~762 / 749~760 / 732~747 | 98.6~100.6 / 104.9~107.1 / 106.6~108.0 / 106.6~108.3 |
| Qwen2.5 7B 4bit | 2048 | 799~808 / 800~803 / 790~797 / 790~794 | 96.7~96.9 / 98.0~100.4 / 99.0~100.5 / 100.3~101.5 |

두 번째 측정에서는 1000의 prefill 손해가 변화 없음 ~ -1.4%로 첫 스윕보다 작았다. decode는 네 칸 모두 1000이 가장 빨랐다.

### 7.6 메모리 비용: 예산을 전역으로 올리면 안 되는 이유

MLX peak 메모리 (`MLX peak memory`, 2048 토큰 prompt, 32 토큰 생성):

| 체크포인트 | 50 | 100 | 200 | 400 | 1000 |
|---|---|---|---|---|---|
| Qwen2.5 7B 4bit | 6.01 GB | 6.90 | 8.93 | 11.09 | 12.77 |
| Qwen3-30B-A3B 4bit | 19.75 GB | 20.74 | 23.36 | 28.78 | 36.19 |
| command-r7b 4bit | 7.11 GB | | | | 13.94 |
| Llama 3.1 8B bf16 | 17.93 GB | | | | 24.24 |

생성 토큰을 1개로 줄여 prefill만 떼어 보면(Qwen2.5): 512 토큰 5.07 → 6.56 GB, 2048 토큰 6.01 → 12.59 GB, 64 토큰 4.46 → 4.89 GB. 증가분은 prefill에서 나오고 prompt 길이에 비례한다. command buffer가 끝나기 전까지 prefill 중간 버퍼가 해제되지 않기 때문이다. 128 GB인 이 기기에서는 문제가 없지만, 16/32 GB M1~M4 기기에 전역 기본값으로 넣으면 OOM 위험이 있다.

MLX `Device`는 예산을 생성 시점에 환경변수에서 한 번만 읽으므로, 고정값 하나로는 두 단계를 모두 만족시킬 수 없다. 그래서 prefill은 기본 예산, decode만 1000으로 바꾸는 런타임 전환을 택했다(사용자 결정).

## 8. cohere2 구조 변경 실험 (Python 복제본, 예산 1000)

| 변형 | 토큰당 시간 | 기준 대비 |
|---|---|---|
| 기준 | 8.923 ms (112.1 tok/s) | |
| 잔차 덧셈 둘을 한 커널로 (`add3`) | 8.773 ms (114.0) | +1.7% |
| 입력 투영 합치기 (q\|k\|v\|gate\|up 한 번의 matvec) | 9.053 ms (110.5) | -1.4% |
| 출력 투영 합치기 (concat + o\|down 한 번의 matvec) | 9.061 ms (110.4) | -1.5% |
| 입력·출력 모두 합치기 | 9.156 ms (109.2) | -2.5% |
| (비용 측정용) RoPE 제거 | 8.746 ms | +2.0%, RoPE 비용 약 0.18 ms |
| (비용 측정용) LayerNorm 제거 | 8.687 ms | +2.7%, norm 비용 약 0.24 ms |
| (비용 측정용) SDPA를 가벼운 연산으로 대체 | 8.893 ms | +0.3%, 컨텍스트 500에서 SDPA는 거의 무비용 |

별도 실행: 기준 9.008 ms, `add3` 8.772 ms (+2.7%), MLP를 attention 앞으로 당기는 스케줄(`mx.depends`로 swiglu를 RoPE 앞에, down_proj를 SDPA 앞에 강제) 9.333 ms (-3.5%), 그 스케줄 + `add3` 9.220 ms (-2.3%).

별도 실행(예산 1000, 3회): 기준 9.019 ms, `add3` 8.924 ms (+1.1%), MLP 분기(gate/up/swiglu/down)를 두 번째 GPU 스트림(별도 command queue)에 올린 변형 29.448 ms (34.0 tok/s, -69.4%). 병렬 블록의 attention과 MLP를 서로 다른 큐에서 겹치려는 시도였지만, MLX는 스트림 간 의존성마다 command buffer를 끊고 fence로 기다리므로 레이어당 두 번의 동기화 비용이 이득을 압도한다.

해석:
- 투영 합치기는 matvec 체인만 잴 때는 +3.4%였지만 실제 그래프에서는 손해다. gate/up이 이미 o_proj와 같은 barrier 단계에서 병렬로 돌고 있어서, 합치면 그 겹침이 사라지고 gate/up이 attention 앞의 임계 경로로 올라온다.
- MLX는 실행 순서(tape)를 출력에서 거꾸로 넓이 우선으로 만들고 노드를 되도록 늦게 배치한다(`MLX_BFS_MAX_WIDTH` 기본 20). MLP를 앞당겨 작은 커널과 겹치게 하는 시도는 역효과였다.

mlxcel에서 BFS 폭을 바꿔 본 결과 (예산 1000, 2회): 폭 20 110.20 / 109.55, 폭 1 106.98 / 107.07, 폭 5 109.18 / 109.62, 폭 50 110.69 / 108.61, 폭 200 108.86 / 110.42. 기본값이 가장 낫거나 같다.

## 9. 구현

### 9.1 decode 전용 입력 예산 전환 (커밋 `cdaa94b8`)

- MLX overlay `src/lib/mlx-cpp/patches/mlx/backend/metal/device.cpp`: upstream `81ba1c6a` 원본에 세 군데만 더했다. `<atomic>` include, `needs_commit`에서 0이 아닌 override 값을 예산으로 쓰는 부분, 브리지용 진입점 `mlxcel_set_mb_per_buffer_override` / `mlxcel_mb_per_buffer_override`. `CMakeLists.txt`의 Metal overlay 목록에 등록했다. MLX 핀을 올릴 때 새 upstream 파일에 이 세 hunk를 다시 적용해야 한다.
- 브리지: `set_metal_mb_per_buffer_override(i32)`, `metal_mb_per_buffer_override()`. Metal 빌드(`MLXCEL_BRIDGE_METAL_BACKEND`)에서만 overlay를 부르고, 그 외 빌드에서는 아무 일도 하지 않고 0을 돌려준다.
- `mlxcel_core::DecodeCommandBufferBudget` (RAII 가드): 생성 시 decode 예산을 설정하고 drop 시 이전 값으로 되돌린다. 중첩 가능.
- `hardware::decode_mb_per_buffer()`: 예산 결정 순서는 다음과 같다. 운영자가 `MLX_MAX_MB_PER_BUFFER`를 설정했으면 전환하지 않는다(두 단계 모두 그 값). `MLXCEL_DECODE_MB_PER_BUFFER`가 있으면 그 값을 쓴다(`0`/`off`/`false`/`no`는 끔). 그 외에는 하드웨어 기본값으로, 연산 한도와 같은 게이트인 M1~M4에서 1000, M5 이상과 Apple 외 환경에서는 끔.
- 적용 지점: `generate.rs`의 decode 루프 4개(`generate_streaming`, `generate_streaming_with_embeddings`, `generate_with_stats_and_embeddings`, `generate_with_stats`). 모두 prefill을 평가한 직후 `prepare_turbo4_delegated_before_decode` 다음에 가드를 건다(`MLXCEL_FORCE_SYNC` 강제 동기 모드에서는 걸지 않음). 서버 배치 스케줄러에서는 `run_decode_tick` 안의 파이프라인 구간(steady lookahead tick, lookahead prime)에만 건다. 동기 decode 단계와 chunked prefill(`continue_chunked_prefill`)은 기본 예산으로 인코딩된다. 처음에는 서버의 `execute_decode_step` 전체에 걸었는데, 13.1절의 동기 경로 회귀를 찾아 범위를 좁혔다.
- 문서: `docs/environment-variables.md`의 `MLX_MAX_MB_PER_BUFFER` 행을 고쳤다(Metal에도 적용됨). `MLXCEL_DECODE_MB_PER_BUFFER` 행과 `mlxcel --help` 항목을 추가했고, 측정 기록 `docs/benchmark_results/metal-mb-per-buffer-m1ultra-2026-09-21.md`를 새로 썼다.
- 테스트: overlay round-trip(음수는 0으로 처리), 가드 중첩과 복원, 예산 없는 가드는 값을 건드리지 않음, 게이트가 연산 한도 기본값과 같은 하드웨어에 적용됨, 운영자 `MLX_MAX_MB_PER_BUFFER`가 전환을 끔, `MLXCEL_DECODE_MB_PER_BUFFER`의 덮어쓰기·끄기·잘못된 값 처리.

처음에는 1000을 M1~M4 고정 기본값으로 넣는 방식으로 구현했다. 7.6절의 메모리 비용을 확인한 뒤 이 전환 방식으로 바꿨다.

### 9.2 cohere2 잔차 덧셈 융합 (커밋 `793d69f5`)

`(attn_h + ff_h) + x`를 `mlxcel_core::compiled_add3`(MLX compile, shapeless) 한 커널로 바꿨다. 결합 순서가 같아 결과가 바이트 단위로 같고, 레이어마다 dispatch와 barrier 단계가 하나씩 줄어든다.

## 10. 검증

| 항목 | 결과 |
|---|---|
| temp 0 생성 일치 | 3개 prompt × 200 토큰, `main`과 변경 빌드의 생성 텍스트 해시 동일 |
| `add3` 단독 효과 | 양쪽 모두 예산 1000으로 고정, ABBA 8쌍: 110.54 (109.63~111.31) 대 109.35 (108.36~110.03), +1.1% |
| 고정 기본값 방식의 적용 여부 (중간 단계) | 환경변수 없이 109.55~110.35, `MLX_MAX_MB_PER_BUFFER=50`으로 덮으면 101.70~103.09 |
| 전환 켬 / 끔 (`MLXCEL_DECODE_MB_PER_BUFFER=0`) | decode 110.07~111.19 대 104.23~104.43, prefill 671~680 대 674~676 |
| 전환 상태의 peak 메모리 (2048 토큰) | Qwen2.5 7B 6.07 GB, Qwen3-30B-A3B 19.75 GB로 기본값과 같음 (고정 1000이면 12.77 / 36.19) |
| 전환 상태의 연산 한도 재확인 | 50: 106.25~106.39, 100: 108.02~108.16, 200: 108.43~109.77, 400: 108.89~109.52, 1000: 109.21~110.06. 1000 유지 |
| 예열 없는 단발 생성 (`mlxcel generate -n 128 --temp 0`, 3회) | `main` 104.35~104.79 대 브랜치 113.85~114.21 (+9%). CLI에는 첫 decode 지연이 없음 |
| 600 토큰 생성 (256 토큰마다 도는 캐시 비우기 2회 포함, 3회) | `main` 104.53~104.71 대 브랜치 110.77~111.20 (+6%). 긴 생성에서도 이득 유지 |
| 단위 테스트 | 새 테스트 5개와 기존 hardware 테스트 통과 |
| 전체 게이트 | `make verify-test`: 바이너리 126개, 11,411 통과, 실패 0, 무시 368. `cargo clippy --workspace --all-targets -D warnings`, `cargo fmt --check`, 교차 저장소 참조 검사, 라이선스 헤더 검사 통과 |

### 10.1 최종 비교 (ABBA 8쌍, pp500/tg128)

| 조건 | `main` `0bbfa95d` decode | 변경 브랜치 decode | prefill `main` / 브랜치 |
|---|---|---|---|
| 일반 환경 (load average 3.3) | 102.69 (100.33~103.02) | 109.88 (108.94~110.37) | 670.5 / 671.5 |
| 인덱서 정지 (`with_indexers_paused.sh` + `suggestd`) | 102.77 (102.23~103.18) | 109.50 (108.73~110.06) | 670.4 / 670.6 |

인덱서를 멈췄던 데몬은 측정 뒤 모두 재개된 것을 확인했다.

### 10.2 서버 경로

서버 경로의 측정과 해석은 13.1절에 정리했다. 이 절의 초기 측정(`ignore_eos` 요청으로 "서버 decode가 CLI보다 약 9% 느리다", "전환을 켰을 때 첫 요청이 약 15% 느리다")은 둘 다 lookahead에서 제외되는 동기 경로를 잰 결과였고, 13.1절에서 바로잡았다.

## 11. 남은 과제

- 120 tok/s: 서버 일반 요청은 약 118.4로 1.3% 모자라고, CLI 벤치(`ignore_eos`)는 111.1이다. 출력을 바꾸지 않는 남은 후보는 모두 1% 안팎이다: RoPE+KV 캐시 쓰기 융합(상한 +1.2%, buffer donation을 하는 C++ primitive 필요), qkv matvec의 threadgroup 폭 조정(약 +1%, 칩별 튜닝).
- 120을 확실히 넘으려면 선택이 필요하다: 읽는 바이트 줄이기(tied lm_head 590 MB가 매 토큰의 13%, 저비트화는 출력이 바뀜) 또는 speculative decoding(MTP head 없음, 같은 토크나이저의 draft 필요, 과거 이 기기에서 산문 0.95배).
- 동기 decode 경로(`ignore_eos`, logit bias, penalty, logprobs, grammar 요청)는 lookahead가 없어 일반 요청보다 약 15% 느리다(약 101 대 118). token bias를 장치 쪽 샘플링에서 처리하면 `ignore_eos`와 logit bias 요청도 lookahead를 탈 수 있다(별도 과제 후보).
- 측정하지 않은 것: M2~M4의 Ultra 외 칩, M5 이상, speculative 루프(DFlash, MTP)에 전환 연결, 2048 토큰을 넘는 컨텍스트의 decode.
- `rust/mlxcel-2-old` worktree(`30ab5a39`, 옛 MLX 핀 빌드 포함)는 이후 A/B용으로 남겨 두었다.

## 12. 부록: 측정 스크립트와 원본

측정 스크립트와 원본 결과는 세션 scratchpad(`/private/tmp/claude-501/-Volumes-Monolith-Development-rust-mlxcel/496d5b59-6120-4569-8f0b-d2c61c9e9574/scratchpad/`)에 있다. 이 경로는 임시 디렉터리라 재부팅 등으로 사라질 수 있다.

| 파일 | 내용 |
|---|---|
| `run_bench.sh` | 1차 prefill/decode 사다리 (pp512 × 4, pp2048 × 3, pp4096 × 3, 실행 전 부하 샘플링) |
| `ab.sh`, `ab2.sh` | ABBA 비교 (코드 A/B, `add3` 단독, 최종 비교) |
| `roofline.py`, `readmax.py`, `levels.py` | 대역폭 roofline, 순수 읽기 상한, barrier 단계 비용 |
| `replica.py` | cohere2 decode 구조 복제본과 변형 실험 |
| `sweep.sh`, `mb_sweep.sh` | command buffer 연산·예산 한도 스윕 (mlxcel 대상) |
| `server_ab.sh` | 서버 경로 전환 켬/끔 비교 |
| `gpu_busy.py`, `gpu_clients.py` | xctrace 구간 분석, 프로세스별 GPU 시간 |
| `cr7b_*.csv`, `ab/`, `mb_*.txt`, `mbc_*.txt`, `final_*.txt`, `ab_add3.txt` | 원본 측정값 |

주의: 측정 도중 `~/.gitconfig`(iCloud Drive를 가리키는 심볼릭 링크)를 읽을 수 없게 되어 CMake FetchContent의 git 단계가 실패했다. 빌드는 `GIT_CONFIG_GLOBAL=/dev/null`로, 커밋은 작성자 정보를 명시해서 진행했다.

## 13. 후속 최적화 (1 → 2 → 3 단계)

### 13.1 서버 경로

- `ignore_eos`는 서버에서 -inf token bias로 구현되고, lookahead 파이프라인의 적용 조건(`lookahead_params`)은 token bias, penalty, per-token logprobs, grammar가 있는 요청을 거부한다. 그래서 10.2절의 초기 서버 측정은 모두 동기 경로를 잰 것이었다. `mlxcel_batch_decode_lookahead_steps_total` 지표로 확인했다(일반 요청은 128토큰 중 125스텝이 lookahead, `ignore_eos` 요청은 0).
- 동기 경로에서는 큰 예산이 해롭다. 한 스텝을 모두 인코딩하고 commit한 뒤 기다리므로, command buffer가 하나가 되면 GPU가 CPU 인코딩이 끝날 때까지 쉰다. 기본 예산에서는 buffer가 잘게 나뉘어 스텝 안에서 CPU 인코딩과 GPU 실행이 겹친다. 그래서 서버 가드를 파이프라인 구간(steady lookahead tick, prime)으로 좁혔다.
- 이 회귀는 예산 전환의 첫 커밋(`f5f45c45`, amend 전)에 그대로 들어 있었다. CLI 벤치는 파이프라인 루프라서 회귀가 보이지 않았고, 서버에서도 기동당 요청 4개로는 흔들림이 우연처럼 보였다. 기동당 요청 10개로 늘려서야 드러났다. push 전이라 같은 커밋에 수정을 합쳤다(`cdaa94b8`).
- 서버 예열을 1토큰에서 8토큰으로 늘리는 수정도 시험했지만, 느린 요청이 두 번째 요청으로 옮겨 갔을 뿐이다(해법 아님, 되돌림). 서버는 prefill이 끝날 때마다 `clear_memory_cache()`를 부르므로 예열 효과가 요청 사이에 남지도 않는다.

command-r7b, 128토큰 `/completion`(temp 0), 서버 기동당 요청 10개, 기동 2회 (tok/s):

| 경로 / 설정 | 전환 끔 | 전환 켬 |
|---|---|---|
| 동기(`ignore_eos`), 가드를 decode 단계 전체에 건 초기 구현 | 97.8 ~ 101.8 | 67.6 ~ 102.1 (불안정) |
| 동기(`ignore_eos`), 가드를 파이프라인 구간에만 건 최종 구현 | 100.6 ~ 101.1 | 100.5 ~ 102.1 |
| lookahead(일반 요청), 최종 구현 | 106.8 ~ 110.1 (중앙값 110.0) | 115.8 ~ 119.1 (중앙값 118.5), +7.7% |

`main`과 최종 HEAD의 서버 일반 요청 비교: `main` 105.5 ~ 108.3(각 기동의 첫 요청 제외 시 107.2 ~ 108.3), HEAD 116.9 ~ 119.0.

### 13.2 잔차 덧셈 + 다음 LayerNorm 융합 (커밋 `c567ff61`)

- cohere2 레이어 경계는 `(attn + mlp) + x`(compiled add)와 다음 레이어의 LayerNorm, 두 개의 의존 커널이다. Metal 커스텀 커널 `fused_add3_layer_norm` 하나로 잔차와 정규화 결과를 함께 쓴다. 마지막 레이어는 최종 norm과 합친다.
- 바이트 동일성: 고정된 MLX(`81ba1c6a`)의 `layer_norm_single_row`를 그대로 옮겼다. threadgroup 크기(32 × ceil(ceil(D/8)/32)), 스레드당 8개 읽기, simdgroup·threadgroup 두 단계 합, `metal::precise::rsqrt`, T 타입의 affine 단계를 모두 같게 했다. bias가 없을 때도 `fast::layer_norm`처럼 0을 메모리에서 읽는다. 잔차는 `compiled_add3`와 같은 결합 순서로 입력 dtype 안에서 계산한다.
- 구현 중 걸린 함정: `metal_kernel`은 0차원 입력을 포인터가 아닌 스칼라로, 작은 입력을 `constant` 주소 공간으로 넘긴다(원소 1개 배열과 `auto` 포인터로 해결). MLX 원본 커널에는 폭이 좁은 행에서 음수 인덱스로 쓰는 루프가 있어, 결과가 같도록 시작 인덱스를 0으로 막았다.
- 검증: 단위 테스트가 f16·bf16, bias 유무, 여러 행, 8의 배수가 아닌 폭에서 정확히 같은지 확인한다. 커널의 rsqrt에 0.1%를 곱하면 테스트가 실패해, fused 경로가 실제로 실행되고 테스트에 판별력이 있음을 확인했다. temp 0 생성(3 prompt × 200토큰)이 `main`과 같다.
- 효과: 같은 바이너리에서 `MLXCEL_FUSED_ADD_NORM=0`과 ABBA 8쌍, decode 112.45 (112.01 ~ 112.99) → 113.72 (113.46 ~ 114.15), +1.1%. prefill 동일.

### 13.3 RoPE + KV 캐시 쓰기 융합 (보류)

MLX 커스텀 커널은 기존 캐시 버퍼에 제자리 쓰기를 할 수 없어 buffer donation을 하는 C++ primitive가 필요하다. 먼저 상한을 쟀다. Python 복제본에서 k에 RoPE를 빼면 캐시 쓰기가 q RoPE와 같은 barrier 단계에 들어가는데, 이것이 융합했을 때의 단계 구조다. 결과는 8.659 → 8.552 ms/tok(+1.2%)이었고 k RoPE 연산까지 뺀 값이라 실제 이득은 이보다 작다. 위험 대비 이득이 작아 보류했다.

### 13.4 decode matvec 커널 튜닝

- 모양별 대역폭(실제 가중치, 서로 다른 32개 행렬): 의존 체인에서 qkv(6144×4096)와 o_proj(4096×4096)는 약 410 GB/s, down(4096×14336)은 약 480 GB/s, gate/up(14336×4096)은 550 ~ 630 GB/s, lm_head는 540 GB/s. 읽기 상한(717 GB/s)이었다면 토큰당 약 2.1 ms가 줄어드는 셈이다.
- 직접 짠 단순 커널(nibble shift 방식)은 모든 모양에서 MLX보다 느렸다(0.75 ~ 0.93배).
- MLX `qmv_fast`의 안쪽 루프(`load_vector`, `qdot`)를 그대로 옮긴 파라미터형 커널을 만들었다. MLX와 같은 설정에서는 결과가 비트 단위로 같았고, 행 수와 simdgroup 수만 바꾼 설정도 비트 단위로 같았다. K 분할이나 읽기 폭을 바꾸면 합산 순서가 달라져 미세하게 달랐다.
- 실제 레이어처럼 qkv→o, gate→down을 잇는 체인으로 설정을 훑었다. 반복 간 편차가 약 20%로 커서, 최솟값 기준으로 qkv(16,4,8,1)가 약 6%, lm_head(16,4,16,1)가 약 7% 빨랐다. o_proj, gate, down에서는 더 나은 설정이 없었다.
- 비트 동일성 재확인: 권한 설정 (16,4,8,1)과 (16,4,16,1), (16,2,16,1), (16,8,8,1) 모두 qkv, o, gate, down 네 모양에서 `mx.quantized_matmul`과 비트 단위로 같았다.
- 전체 모델 복제본에서 확인한 결과(6라운드): 기준 8.869 ms, qkv 설정 변경 8.773 ms(-1.1%), lm_head까지 바꿔도 8.763 ms. 비트 동일하게 얻을 수 있는 이득은 약 1%이고, 칩별로 튜닝해야 하는 커스텀 matvec을 모델 하나에 넣을 만한 값은 아니라고 판단했다. 일반화한다면 MLX 쪽에서 N에 따라 threadgroup 폭을 고르는 변경이 맞다.
- 참고 시도: MLP 분기를 두 번째 GPU 스트림으로 돌리는 변형은 29.4 ms/tok(-69%)였다. MLX는 스트림을 넘는 의존마다 command buffer를 끊고 fence로 기다리게 한다.

### 13.5 최종 비교 (`main` `0bbfa95d` 대비 HEAD `c567ff61`)

| 측정 | `main` | HEAD |
|---|---|---|
| CLI 벤치 decode (pp500/tg128, `ignore_eos`, ABBA 8쌍 중앙값) | 102.75 (100.56 ~ 103.14) | 111.08 (110.40 ~ 111.40), +8.1% |
| CLI 벤치 prefill | 670.8 | 669.8 |
| 서버 일반 요청 decode (요청 10개 × 기동 2회, 첫 요청 제외) | 107.2 ~ 108.3 | 117.3 ~ 119.0, 약 +10% |

세 수치의 관계: CLI 벤치는 500토큰 prompt로 재고, 서버 수치는 약 20토큰 prompt로 쟀다. 14.2절에서 확인했듯 같은 컨텍스트에서는 두 경로의 속도가 같다(약 490토큰 prompt에서 서버 112.7 ~ 114.3, CLI 벤치 112 ~ 113). 차이는 컨텍스트 길이에서 오고, CLI `generate`가 출력하는 tok/s는 첫 토큰과 시작·마무리 구간을 포함해 이보다 낮게 나온다. 같은 조건끼리 비교한 개선 폭은 CLI 벤치 +8.1%, 서버 +10%다.

HEAD `c567ff61`에서 `make verify-test`: 바이너리 126개, 11,412 통과, 실패 0, 무시 368. `cargo clippy --workspace --all-targets -D warnings`, `cargo fmt --check`, 라이선스 헤더 검사 통과.

## 14. PR 제출과 추가 탐색

### 14.1 PR

최신 main(`6f257ab1`) 위로 두 PR로 나눠 올렸다. 각 브랜치에서 fmt, clippy, 새 테스트를 따로 확인했다.

- #1947 `perf/metal-decode-command-buffer-budget`: decode 전용 command buffer 예산 전환. 커밋 메시지와 벤치 기록을 이 변경만의 수치(같은 바이너리 켬/끔)로 고쳤다.
- #1948 `perf/cohere2-fused-residual-norm`: `compiled_add3`와 잔차 덧셈 + LayerNorm 융합. 수치는 #1947을 적용한 상태에서 잰 것이다.

### 14.2 CLI decode 루프를 lookahead로 바꾸기: 변경 불필요

- CLI 루프는 이미 평가하지 않은 토큰 위에 다음 그래프를 쌓는 파이프라인 구조다. 정상 상태 토큰당 시간을 분해하면(`MLXCEL_PROFILE_PIPELINE_DETAIL`, 서버와 같은 20토큰 prompt) 8.2 ~ 8.7 ms로, 서버 lookahead(8.44 ms)와 같다.
- Metal System Trace도 같다: CLI GPU 가동률 97.6%, 서버 97.1%, command buffer 경계 빈틈 중앙값 둘 다 약 61 us(토큰당 buffer 약 2개).
- 서버에 약 490토큰 prompt를 주면 112.7 ~ 114.3 tok/s로, CLI 벤치(500토큰 prompt, 112 ~ 113)와 같다.
- 앞서 본 약 4% 차이의 출처는 두 가지였다. 비교한 컨텍스트 길이가 달랐고(서버 약 20토큰, 벤치 500토큰), CLI `generate`가 기본으로 출력하는 tok/s는 prefill 시간까지 분모에 넣는다(짧은 prompt에서 108.7 ~ 115.2. 값의 수준이 서버보다 낮은 것은 prefill 포함 때문이고, 범위가 넓은 것은 예열 없는 단발 실행의 회차 편차다; 16절 참고). `ignore_eos` 유무는 CLI 속도에 영향이 없었다(112.2 ~ 113.0 양쪽).
- 서버는 이 모델에서 paged 저장을 요청해도 dense로 돌아가므로(로그: "Paged decode storage requested but unavailable for this worker; falling back to dense"), 저장 방식 차이도 아니다.

### 14.3 decode attention(SDPA): 남은 것 중 가장 큰 후보

컨텍스트가 20에서 500으로 늘면 토큰당 약 0.43 ms가 늘어나는데, KV를 읽는 대역폭 비용은 약 0.13 ms뿐이다. SDPA만 떼어 레이어 32개 의존 체인으로 잰 결과(q 32 head, KV 8 head, head_dim 128, f16):

| KV 길이 | 레이어당 SDPA | 사용 커널 |
|---|---|---|
| 128 | 16.4 us | 1-pass |
| 256 | 19.5 us | 1-pass |
| 512 | 26.6 us | 1-pass |
| 768 | 37.9 us | 1-pass |
| 1000 | 46.8 us | 1-pass |
| 1023 | 53.3 us | 1-pass |
| 1024 | 38.3 us | 2-pass (Ultra의 전환 임계값) |
| 2048 | 47.1 us | 2-pass |
| 4096 | 69.3 us | 2-pass |

- 1-pass 커널은 q head마다 threadgroup 하나(32개)만 띄워, 코어 64개인 M1 Ultra에서 지연에 묶인다. 위 표는 SDPA만 이어 붙인 직렬 상한이다. 실제 모델에서는 SDPA 비용 상당 부분이 다른 커널에 가려지므로, 되찾을 수 있는 양은 전체 모델 복제본에서 SDPA를 가벼운 연산으로 바꿔 쟀다: 컨텍스트 500에서 +0.6%, 800에서 +1.9%, 1500에서 +10.3%(토큰당 0.93 ms).
- MLX는 Ultra에서 KV 1024 이상일 때만 2-pass로 바꾸는데, 경계에서 2-pass가 28% 빠르다(1023: 53.3 us, 1024: 38.3 us). 교차점은 512와 768 사이로 보인다.
- 2-pass의 블록 수(`MLX_SDPA_BLOCKS`, 런타임 설정)는 64가 모든 길이에서 가장 빨랐다(레이어당 1024: 35.0 → 29.0 us, 2048: 45.2 → 40.0, 4096: 69.2 → 60.5; 32와 256은 더 느림). 실제 모델에서는 컨텍스트 2048에서 96.3 → 97.5 tok/s(+1.3%), 4000에서 약 +0.5%(잡음 섞임)였다. 블록 수가 바뀌면 2-pass 합산 순서가 바뀌어 수치가 미세하게 달라진다.
- 따라서 decode attention 개선은 120 목표를 재는 조건(pp512)에는 거의 도움이 되지 않고, 컨텍스트 1000 이상의 긴 decode에서 큰 레버다. 방법은 둘이다. MLX의 2-pass 선택 기준과 블록 수를 이 기기에 맞추는 작은 overlay가 있고, GQA를 묶어 KV를 한 번만 읽으면서 K를 여러 threadgroup으로 나누는 decode attention 커널이 있다. mlxcel의 paged 저장용 split-K decode 커널(v2)이 후자의 출발점이 될 수 있다. 어느 쪽이든 수치가 미세하게 바뀌므로 logit trace 검증이 필요하다.

### 14.4 남은 후보 정리

| 후보 | 예상 이득 | 출력 | 비용 |
|---|---|---|---|
| 병렬도 높은 decode attention 커널 | 컨텍스트 500에서 최대 0.6%, 800에서 1.9%, 1500에서 최대 10% | 미세 변화, logit trace 필요 | 큼 |
| Ultra에서 1-pass/2-pass 임계값 낮추기 (MLX overlay) | 컨텍스트 768~1023에서 일부(800에서 SDPA 전체가 1.9%) | 미세 변화 | 중간 (overlay 파일 추가) |
| `MLX_SDPA_BLOCKS=64` 기본값 (Ultra, GQA 4) | 컨텍스트 2048에서 +1.3% | 미세 변화 | 작음, 다른 계열 검증 필요 |
| RoPE + KV 캐시 쓰기 융합 | 상한 +1.2% | 바이트 동일 가능 | 중간 (C++ primitive) |
| qkv matvec threadgroup 폭 | 약 +1% | 바이트 동일 | 중간 (칩별 튜닝) |

## 15. 바이트 동일 1%대 후보 마무리 (둘 다 실제 엔진에서 이득 없음)

### 15.1 qkv matvec의 threadgroup 폭

- MLX `qmv_fast`의 4-bit 안쪽 루프를 그대로 옮기고 threadgroup당 simdgroup을 2개에서 8개로 늘린 Metal 커널(`quantized_matvec_wide`)을 구현했다. 행마다 같은 레인이 같은 순서로 더하므로 `quantized_matmul`과 비트 단위로 같고, 단위 테스트가 f16과 bf16에서 이를 확인했다. 출력에 0.1%를 곱하면 테스트가 실패해 판별력도 확인했다. cohere2의 fused QKV projection에 decode(L=1)에서만 연결했고, greedy 생성 3 prompt × 200토큰이 main과 같았다.
- 같은 바이너리에서 `MLXCEL_QKV_WIDE_MATVEC=0`과 ABBA: 500토큰 prompt 8쌍 111.77 대 111.85(0.999배), 16토큰 prompt 6쌍 116.70 대 116.77(0.999배). Python 복제본에서 본 -1.1%가 실제 엔진에서는 재현되지 않았다.
- 반영하지 않았다. 변경분은 세션 scratchpad `wide_matvec_experiment.patch`에 남겼다.

### 15.2 RoPE + KV 캐시 쓰기 융합

- mlxcel의 기존 `fused_rope_append`(#905)는 캐시에 제자리로 쓰지 않는다. MLX `metal_kernel` 출력은 항상 새 버퍼라 donation이 불가능해서, RoPE 적용 결과만 만들고 저장은 `slice_update`에 맡긴다. 그래서 barrier 단계가 줄지 않고 Llama에서도 이득이 0이었다. 진짜 융합에는 buffer donation을 하는 MLX `Primitive` 서브클래스(내부 Metal API 사용)가 필요한데, mlxcel에는 아직 그런 선례가 없다.
- 구현 전에 실제 엔진에서 상한을 쟀다. k의 RoPE를 건너뛰는 임시 스위치를 넣은 빌드로 ABBA 8쌍을 돌렸고, 결과는 112.28 대 111.41(+0.8%, 범위 겹침)이었다. RoPE 계산 자체까지 뺀 값이라 실제 융합 커널의 이득은 이보다 작다.
- 새 primitive 인프라를 들일 만한 값이 아니라 구현하지 않았다. 스위치는 측정 후 되돌렸다.

### 15.3 PR 상태

- #1948: CI 전 항목 통과.
- #1947: GB10 러너의 WebUI Activity 성능 게이트가 기준(decode 저하 중앙값 2% 이하)을 두 번 연속 조금 넘었다(2.12%, 2.14%). 같은 job이 최근 통과한 네 run의 값은 1.30, -0.80, 1.67, 0.80%였다. 이 PR이 CUDA에서 이 측정에 영향을 줄 경로는 찾지 못했다(가드는 비활성 상태로 생성만 되고, overlay는 Metal 빌드에서만 컴파일된다). 재시도는 한 번만 했고, 분석을 PR 댓글로 남겼다. 같은 러너에서 main을 비교 측정하면 가릴 수 있다.

## 16. 같은 조건에서 CLI와 서버 비교

브랜치 `perf/cohere2-decode-followups`(`2fe9761b`)를 다시 빌드한 같은 바이너리에서, 벤치가 만드는 512토큰 prompt를 텍스트로 복원해(다시 토큰화하면 id 512개가 그대로 나오는 것을 확인) 모든 경로에 같은 입력을 줬다. 생성 128토큰, greedy, 각 3회. 서버는 `cache_prompt: false`, 측정 전 요청 2개로 예열했다. 부하 평균 2.1 ~ 3.0.

| 경로 | 표시 값 (tok/s) | 분모 | decode 방식 |
|---|---|---|---|
| `mlxcel-bench-decode` | 111.3 ~ 112.1 | decode 구간 (첫 토큰 이후) | 파이프라인 |
| `mlxcel-bench-decode --ignore-eos` | 110.6 ~ 112.2 | decode 구간 | 파이프라인 |
| `mlxcel generate --profile` | 113.1 ~ 113.8 | decode 구간 | 파이프라인 |
| 서버 `/completion` (`ignore_eos` 없음) | 111.0 ~ 113.3 | 첫 토큰부터 마지막 토큰까지, 간격 n-1개 | lookahead (단계 125/128) |
| 서버 `/completion` (`ignore_eos: true`) | 97.0 ~ 97.4 | 위와 같음 | 동기 (lookahead 단계 0) |
| `mlxcel generate` (기본 출력) | 69.1 ~ 69.7 | prefill + decode 전체 | 파이프라인 |

- 같은 조건의 정상 상태 decode 속도는 CLI와 서버가 같다(111 ~ 114). 서버를 CLI 방식(n/decode 구간)으로 다시 계산하면 111.9 ~ 114.2다.
- 차이 1: 서버에서 `ignore_eos`는 EOS 토큰에 -inf token bias를 거는 방식이고, lookahead 자격 검사(`lookahead_params`가 재사용하는 batched fused 샘플러 조건)가 token bias를 거부해 동기 경로로 떨어진다. 약 13% 느려진다. CLI는 같은 bias를 파이프라인 루프 안의 샘플링 그래프에서 처리하므로 속도가 그대로다. 앞서 서버가 CLI보다 느리다고 본 측정들은 모두 `ignore_eos: true`였다.
- 차이 2: `mlxcel generate`가 기본으로 출력하는 값은 생성 토큰 수를 prefill + decode 전체 시간으로 나눈다(`generation_stats_from_duration`). 512토큰 prompt에서는 prefill이 약 0.73초라 69 tok/s로 보인다. 서버 값을 같은 정의(n/(prompt_ms + predicted_ms))로 계산하면 68.2 ~ 69.1로 일치한다. 짧은 prompt에서는 prefill이 작아 이 차이가 작게 보였을 뿐이다.
- 차이 3: 컨텍스트 길이. 서버를 짧은 prompt(약 20토큰)로 잰 118.5와 벤치 512토큰의 112를 비교하면, 차이는 attention이 읽는 KV 길이에서 온다(14.3절).
- 차이 4 (작음): 분모 정의. 벤치와 `--profile`은 n / (n-1 step 구간), 서버는 (n-1) / 같은 구간이라 128토큰에서 CLI 쪽이 0.8% 높게 나온다.
- prefill: `generate --profile`은 예열 없이 첫 prefill을 재서 821 ~ 837 ms였고, 예열한 벤치는 743 ms, 서버는 723 ~ 732 ms였다. 차이는 예열 여부다.
- 남은 설명 안 된 차이: 같은 `generate_with_stats` 루프인데 `mlxcel-bench-decode`가 `generate --profile`보다 낮다. 번갈아(ABBA) 6회씩 다시 재면 bench 109.9 ~ 112.0(중앙값 약 110.8), `--profile` 110.2 ~ 113.4(중앙값 약 112.8)로 약 1.8%다. 서버는 높은 쪽(`--profile`)과 맞는다. 원인은 확인하지 않았다. 앞 절들의 "CLI 벤치" 값은 모두 bench 도구 값이다.
- 개선 후보: 거부 사유는 상태가 아니라 입력이다. `config_supports_fused_batch`가 `token_bias.is_empty()`를 요구하는데, fused 샘플러는 모든 행이 공유하는 스칼라 파라미터만 받고 행별 bias 입력이 없기 때문이다. token bias는 이전 토큰과 무관한 고정 logits 덧셈이므로, fused 샘플 앞에 행별 bias 덧셈을 넣고 게이트를 다시 열면 `ignore_eos`와 `logit_bias` 요청도 lookahead로 갈 수 있다(이 조건에서 97 → 약 112, 약 +16%). penalty처럼 토큰 이력이 필요한 기능과는 성격이 다르다.

## 17~20절 머리말 (2026-09-23 작업분)

측정 바이너리: `rust/mlxcel` 메인 체크아웃의 `target/release` 빌드(main 기준). 17절 실험을 위해 `src/bin/bench_decode.rs`에 임시 프로브(`MLXCEL_BENCH_PROBE`)를 넣었고 커밋하지 않았다(`rust/mlxcel`의 stash@{0}). 모든 측정은 `scripts/with_indexers_paused.sh`로 인덱서를 멈춘 상태에서 했고, 같은 조건 반복 편차는 0.1 tok/s 미만이다. 17~20절은 처음에 이 보고서가 없어진 줄 알고 `reports_cohere-7b.md`에 따로 썼던 내용을 옮긴 것이다(작업 디렉터리 이동을 삭제로 오인).

## 17. 벤치가 `mlxcel generate --profile`보다 낮게 나온 이유

원인은 decode 정상 상태 속도가 아니라, **decode 구간 맨 앞에 한 번 붙는 고정 비용**이다.

`mlxcel-bench-decode`는 예열을 generator A에서 돌리고 A를 버린 뒤, 측정은 새로 만든 generator B에서 한다. `mlxcel generate --profile`은 generator 하나만 만들어 바로 측정한다. 서버는 하나를 계속 재사용한다.

프롬프트 16토큰에서 decode 시간을 토큰 수별로 재고 직선을 맞추면(기울기 = 토큰당 시간, 절편 = 고정 비용):

| 방식 | 토큰당 | decode 시작 고정 비용 |
|---|---|---|
| 예열과 측정이 같은 generator (`reuse`) | 8.97 ms | 약 1 ms |
| 예열 없음, 새 generator (`--warmup-tokens 0`) | 8.97 ms | 약 12 ms |
| 예열 후 새 generator (현재 벤치 기본) | 8.97 ms | 약 44 ms |

- 정상 상태 토큰당 시간은 세 방식이 같다. 차이는 전부 절편이다.
- 그래서 표시되는 tok/s 차이는 생성 길이에 반비례한다. n=128에서 108.6 대 111.5 대 113.0 (2.7 ~ 4.1% 차이), n=512에서 109.3 ~ 110.1 대 110.7 ~ 111.0 대 110.8 ~ 111.1 (0.8% 이하).
- 이전 세션에서 "같은 루프인데 벤치가 1.5 ~ 1.8% 낮다"고 남긴 미해결 항목이 이것이다. 서버가 CLI 벤치보다 약간 높게 나온 것도 같은 이유다(서버는 generator를 재사용하므로 절편이 거의 없다).

### 배제한 원인

| 가설 | 검증 | 결과 |
|---|---|---|
| MLX 버퍼 캐시 상태 | `MLXCEL_CACHE_LIMIT=0` (캐시 끔) | 차이 그대로 |
| 주기적 캐시 비우기 | `MLXCEL_CACHE_CLEAR_INTERVAL=4` | 차이 그대로 |
| generator마다 새로 만드는 MLX 스트림 | 순수 MLX에서 스트림 1 ~ 4개를 만들고 decode 모양 체인을 동기 `eval`로 실행 | 차이 없음. **단 이 검증은 틀렸다. 바로 아래 항목 참고** |
| generator 생성 자체 | 아무 생성도 하지 않는 generator를 만들고 버린 뒤 측정 (`gen_only`) | 예열 없음과 동일 (111.8 대 111.5) |
| 측정용 generator를 먼저 할당 (`pre`) | 스트림·KV 캐시를 예열보다 먼저 확보 | 효과 없음 (108.7) |
| 페이지 폴트 / 상주 메모리 | `/usr/bin/time -l` | page reclaims 294k, RSS 4.5 GB로 세 방식 동일 |
| 전력·발열 상태 | 예열 후 5초 대기 (`sleep`) | 효과 없음 (108.0 ~ 108.3) |
| MLX 그래프 compile 캐시 | `MLX_DISABLE_COMPILE=1` | 절편 그대로 (재사용 1 ms, 예열 후 새 generator 46 ms) |

### 스트림이 원인의 일부다

위 표의 스트림 검증은 동기 `eval`로 쟀기 때문에 비용을 놓쳤다. 이 비용은 파이프라인(`async_eval`로 다음 스텝을 먼저 띄우는 경로)에서만 나타난다. 순수 MLX에서 decode 모양 체인을 mlxcel과 같은 파이프라인 방식으로 돌리면:

```
s1 처음 8스텝 (프로세스 최초): 15.4 13.1 14.4 8.0 8.0 8.0 8.0 7.9 ms
s1 다시                     :  4.5  8.0  8.0 8.0 7.9 8.0 8.1 8.1
s2 처음 8스텝 (s1이 먼저 작업): 15.1 14.2 13.6 8.3 8.0 8.0 8.2 8.0
s3 처음 8스텝               : 13.5 13.5 13.6 8.0 8.0 8.0 8.0 8.0
```

새 스트림은 처음 세 스텝에 걸쳐 약 19 ms를 더 쓰고, 같은 스트림에서 다시 돌리면 그 비용이 없다. mlxcel은 `CxxGenerator`마다 `new_thread_local_generation_stream()`으로 스트림을 새로 만들기 때문에, 새 generator는 항상 이 비용을 치른다. 재사용 arm의 절편이 1 ms인 것과 정확히 맞는다.

순수 MLX에서는 앞선 스트림이 작업을 했든 안 했든 새 스트림 비용이 같았다(s2, s3 모두 s1과 동일). mlxcel에서 12 ms가 44 ms로 커지는 부분은 아직 설명되지 않았다. 측정 arm의 prefill 길이가 달라(차가운 833 ms 대 예열된 723 ms) 스트림 준비 비용 일부가 prefill 쪽에 흡수됐을 가능성이 있고, 확인하지 않았다.

### 고칠 지점

현상은 벤치 도구의 문제로 보이지만, 원인은 런타임 쪽이다. generator마다 스트림을 새로 만드는 대신 스레드당 하나를 공유하면 새 generator의 이 고정 비용이 사라진다. 그러면 벤치·CLI·서버가 같은 조건으로 수렴한다.

### 해야 할 일

- 먼저 런타임 쪽(스레드당 생성 스트림 공유)을 고치는 편이 낫다. 벤치만 재사용으로 바꾸면 mlx-lm 대조가 어긋난다. `scripts/bench_mlxlm.py`도 예열 후 새 `generate` 호출을 재는 구조라 지금의 mlxcel 벤치와 조건이 같기 때문에, mlxcel만 재사용으로 옮기면 n=128 대조 열이 근거 없이 약 3% 유리해진다.
- 벤치 자체를 어느 조건으로 고정할지는 별도 결정이다. 재사용(서버와 같음)과 예열 없음(단발 CLI 실행과 같음) 둘 다 실제 조건이고, 지금 기본값만 둘 중 어느 것도 아니다.
- 고치면 n=128 기준 표시값이 약 3% 올라가므로 과거 벤치 기록과의 비교선이 끊긴다. `docs/benchmark_results/`에 이 전환을 기록해야 한다.
- 지금까지의 A/B 비교(예산 전환, add3, LayerNorm 융합 등)는 양쪽 arm이 같은 절편을 내므로 유효하다.

## 18. 같은 날 mlx-lm 대조 (pp512 / tg128)

| 런타임 | prefill | decode |
|---|---|---|
| mlx-lm 0.31.3 | 778.50 tok/s (657.7 ms) | 94.92 tok/s |
| mlxcel main (벤치 기본) | 약 700 tok/s (723 ~ 743 ms) | 102.2 ~ 102.9 |

- decode는 mlxcel이 8 ~ 10% 빠르다.
- prefill은 mlxcel이 약 10% 느리다. 512토큰에서 65 ms 차이이고, 프롬프트가 길수록 절대값이 커진다. 이 격차는 09-06 기록(694.92 대 776.14)과 같고, 이번에도 그대로 재현됐다.

## 19. 다음 최적화 후보

| 후보 | 크기 | 상태 |
|---|---|---|
| prefill 격차 (mlx-lm 대비 10%) | pp512에서 65 ms, 긴 프롬프트에서 비례 증가 | 미조사. decode에 했던 그래프·커널 대조를 prefill에 하면 된다 |
| 서버 lookahead에 token bias 허용 | `ignore_eos` +6.2%, `logit_bias` +6.5% (main 기준). #1947 예산 빌드에서는 13~16% | **완료**. 이슈 #1950, PR #1951, 20절 |
| decode SDPA (컨텍스트 1000 이상) | `MLX_SDPA_BLOCKS=64` 기본값 +1.3% (ctx 2048), 2-pass 임계값 조정, GQA split-K 커널 최대 +10% (ctx 1500) | 이전 세션에서 측정 완료, 미구현 |
| decode 시작 고정 비용 | 차가운 generator 12 ms, 직전 generator가 있으면 44 ms | 원인 미확정. 요청당 지연이므로 서버 첫 요청에도 영향 |
| PR #1947 / #1948 | CLI +8.1%, 서버 +10% | 두 건 모두 OPEN. #1947은 GB10 러너 게이트가 흔들려 대기 |

## 20. 2번 항목 구현 결과 (이슈 #1950, PR #1951)

lookahead 게이트가 `row_supports_fused_batch`를 재사용하는데, 그 술어는 `token_bias`가 비어 있을 것을 요구한다. 거부 이유는 샘플러 상태가 아니라 입력 모양이었다. fused `[B, vocab] -> [B]` 디스패치는 모든 행이 공유하는 스칼라 파라미터만 받고 행별 bias 입력이 없다. token bias는 생성 이력을 읽지 않으므로 디스패치 전에 로짓에 더하면 되고, 이는 per-row 샘플러가 이미 하는 일이다.

- 엄격 술어(`config_supports_fused_batch`, `row_supports_fused_batch`)는 bias를 적용하지 않는 호출자(`uniform_fused_batch_params`, 그리고 그것을 쓰는 `batched_sample`)를 위해 그대로 두고, bias를 직접 적용하는 호출자용으로 `..._except_bias` 변형을 추가했다.
- `apply_token_bias_rows`는 행마다 맵 하나를 받아 biased id 합집합 한 번의 패스로 처리한다. 어휘 크기에 비례하지 않고, bias가 없는 행에는 정확히 `0.0`을 더해 비트 단위로 그대로 둔다. 행 수가 로짓과 다르면 panic한다(조용히 bias를 흘리면 `ignore_eos` 요청이 EOS를 뽑는다).
- 스케줄러의 두 디스패치 지점(동기 fused 분기, lookahead priming)에 per-row 샘플러와 같은 체인 위치로 넣었다.

측정(같은 조건, 바이너리 2개, 512토큰 prompt, 128토큰, greedy):

| 요청 | main | 이 브랜치 |
|---|---|---|
| 일반 | 105.0 tok/s, lookahead 125단계 | 105.3 tok/s, 125단계 |
| `ignore_eos` | 98.6 tok/s, 0단계 | 104.7 tok/s, 125단계 |
| `logit_bias` | 98.3 tok/s, 0단계 | 104.7 tok/s, 125단계 |

bias 있는 요청이 bias 없는 요청과 같아졌고(+6.2%, +6.5%), bias 없는 요청은 변화가 없다. 앞서 적어 둔 +16%는 #1947 예산이 적용된 빌드에서 잰 값이다. 그 예산은 파이프라인 경로만 올리므로 동기 경로에 남아 있던 요청과의 격차가 그만큼 더 컸다.

출력 동등성 확인에서 한 번 헛디뎠다. 처음 고른 bias 토큰(1734)이 생성문에 나오지 않아 출력이 그대로였고, 그건 bias가 적용됐다는 증거가 되지 못한다. 실제로 생성되는 첫 토큰(17939 `Ġprompt`)을 막자 텍스트가 바뀌었고, 동기 경로(main)와 lookahead 경로(브랜치)의 SHA-1이 같았다(`ef3caa8646c3`).

게이트는 workspace 11417 passed / 0 failed, clippy·fmt 통과. 기록은 `docs/benchmark_results/lookahead-token-bias-m1ultra-2026-09-23.md`.

## 21. 통합 브랜치에서 한계 재확인 (2026-09-23)

통합 브랜치 `perf/cohere2-decode-m1-ultra`(`f86ebbc1`: #1947 예산 전환 + #1948 융합 두 개 + #1951 token bias)를 새 클론에서 빌드해, 저장소 하네스의 표준 조건(pp512 / tg128)으로 다시 쟀다. 인덱서 정지 상태였지만 부하 평균은 8 ~ 10으로 이전 측정(2 ~ 3)보다 높았다. 스크립트는 `.work/cohere2/limits.sh`(로컬 전용).

| 경로 | 하네스 | 이 브랜치 | main | 이전 한계 |
|---|---|---|---|---|
| CLI decode | `scripts/bench_decode.sh` (`--ignore-eos`, 예열 20), 3회 | 111.38 / 112.80 / 112.93 | 미측정 | 111.3 ~ 112.1 (16절) |
| CLI prefill | 같은 실행 | 702.9 / 706.1 / 708.6 | 미측정 | mlx-lm 778.5 (18절) |
| 서버 decode | `scripts/bench_serving_concurrency.py --concurrency 1`, 3회 | 113.0 (첫 요청) / 117.1 / 116.7 | 101.9 (첫 요청) / 106.4 / 107.1 | 약 +10% (13.5절) |

- CLI decode는 이전 한계에 도달했다. 부하가 높았는데도 범위가 겹친다.
- 서버는 첫 요청을 빼면 main 대비 +9.6%로, 13.5절의 약 +10%와 같다. lookahead 카운터가 두 바이너리 모두 요청당 125씩 올랐으므로 둘 다 파이프라인 경로다.
- 서버 2, 3회차의 TTFT가 73 ms인 것은 이 하네스가 같은 prompt를 반복해 prompt cache가 적중했기 때문이다. 서버 하네스로 prefill을 잴 때는 이 점을 감안해야 한다.
- 서버 하네스의 decode 값(117)이 CLI(112)보다 높은 이유(분모 정의 또는 chat template으로 달라진 컨텍스트)는 확인하지 않았다. 같은 하네스 안의 브랜치 대 main 비교만 유효하다고 본다.
- prefill 격차는 그대로다: mlxcel 약 706 대 mlx-lm 778.5(09-23 같은 조건), 약 9%.

## 22. 생성 길이별 decode 속도 (pp512, tg 64 / 128 / 256)

같은 브랜치(`291a6903`, 코드는 `f86ebbc1`과 동일)에서 `scripts/bench_decode.sh --max-tokens N`으로 길이를 번갈아 3라운드 쟀다. 인덱서 정지, 부하 평균 7 ~ 9. 스크립트는 `.work/cohere2/tg_sweep.sh`.

| tg | decode tok/s, 3회 | 중앙값 | decode_ms 중앙값 | prefill tok/s |
|---|---|---|---|---|
| 64 | 107.21 / 108.97 / 108.89 | 108.89 | 587.75 | 702.9 ~ 703.8 |
| 128 | 110.83 / 112.90 / 112.22 | 112.22 | 1140.58 | 702.5 ~ 704.3 |
| 256 | 115.17 / 115.69 / 115.39 | 115.39 | 2218.47 | 702.8 ~ 705.9 |

- 생성이 길수록 tok/s가 오른다. 이 하네스의 tok/s는 생성 토큰 수를 decode 구간 시간으로 나눈 값이라, 17절에서 찾은 decode 시작 고정 비용이 짧은 생성일수록 크게 잡힌다.
- decode_ms 중앙값에 직선을 맞추면 토큰당 약 8.4 ~ 8.6 ms(128 → 256 구간 8.42 ms, 약 118.7 tok/s), 절편 약 42 ms다. 17절의 "예열 후 새 generator 약 44 ms"와 맞는다.
- 그래서 decode 시작 고정 비용(17절, 19절의 스트림 공유 후보)을 없애면 tg64는 약 +7%, tg128은 약 +4%, tg256은 약 +2% 오를 여지가 있다. 정상 상태 속도 자체는 길이와 무관하다.
- KV가 길어지는 효과(pp512 + 256에서 컨텍스트 768)는 이 범위에서 작다. 14.3절 기준 컨텍스트 800에서 SDPA 몫은 약 2%다.
