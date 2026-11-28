# 기술 보고서: PR #3152 - ROCm에서 DeepSeek V2 decode를 기본으로 흡수

**작성일**: 2026-11-18
**상태**: 완료
**위험도**: 중간 (ROCm의 모든 DeepSeek V2 사용자에 대해 기본 캐시 레이아웃을 바꾼다)

## 요약

DeepSeek V2가 이제 ROCm에서 MLA latent `(ckv, kpe)`를 캐시하고 흡수 decode를 기본으로 실행한다. Metal에서 정한 #3131의 규칙과 CUDA에서 정한 #3133의 규칙을 ROCm으로 확장한 것이다. ROCm에는 융합 흡수 decode 커널이 없어서 이 기본값은 조합 경로를 실행하는데, 조합 경로는 프롬프트 512 토큰에서 arm당 5회 실행 중앙값 기준 3.4% 느리고 2048에서 17%, 16384에서 65% 빨랐으며, teacher-forced trace는 1024와 4096에서 확정 불일치 0건이었다.

## 1. 문제 정의

#3133은 2026년 11월의 MLA와 prefill 메모리 작업을 CUDA(#3149)에서 측정하고 호스트가 없어 ROCm은 측정하지 않은 채 닫혔다. 그때 ROCm 기본값은 여전히 decompressed였으므로 `gfx1151` 머신은 이미 알려진 레이아웃보다 KV 캐시를 8.9배 크게 잡았고, `mla::absorbed_enabled()`를 읽는 `kv_arch`도 같은 크기로 추정했다. 빠진 것은 코드가 아니라 근거 행 하나였다.

## 2. 변경 내용

- `mla::default_absorbs_on(fused_available, backend)`가 융합 커널이 도는 곳과 `GpuBackendKind::Cuda`, `GpuBackendKind::Rocm`에서 true를 돌려준다. CPU 백엔드는 decompressed로 남는다.
- `docs/mla-absorbed-decode.md`, `docs/environment-variables.md`, `CHANGELOG.md`가 새 기본값을 서술하고 결과 문서를 연결한다.
- `docs/benchmark_results/mla-and-prefill-memory-rocm-gfx1151-2026-11-18.md`가 이 호스트의 3, 4, 5, 7절을 기록하고, 실행별 CSV가 측정 커밋을 담는다.
- `src/execution/kv_arch.rs`는 모듈 주석에 ROCm을 언급하는 것만 바뀐다. 크기 추정은 코드 변경 없이 런타임 레이아웃을 따른다.

## 3. 기술적 선택과 그 이유

**한 셀이 아니라 근거 규칙으로 바꾼다.** #3102가 세운 규칙은 decode 처리량이 decompressed 대비 몇 퍼센트 안에 들고 teacher-forced trace에 유의미한 불일치가 없을 때 latent 레이아웃을 기본으로 삼는 것이다. ROCm은 둘 다 만족한다. 512에서 -3.4%로 Metal을 전환한 기준(-1.7%) 안쪽이고, 확정 불일치는 208건 중 0건과 105건 중 0건이었다.

**512 토큰 셀은 두 번 측정했다.** 처음 두 실행은 +5.6%로 읽혔지만 decompressed arm이 두 실행 사이에 14% 흔들렸고, 그래서 arm당 5회로 반복했다. 반복 결과는 -3.4%였고 조합 arm의 폭은 11.7%(한 실행이 36.94)였다. 가장 느린 실행을 빼도 조합 중앙값은 41.45로 여전히 3% 정도 뒤진다. 보고서는 유리한 첫 쌍이 아니라 반복 결과를 적는다.

**gpt-oss prefill은 측정값 그대로 둔다.** #3119의 sink prefill은 MLX가 융합한다고 `maskless_route_is_fused`가 답할 때만 MLX의 causal 모드를 쓰고, 그 답은 Metal에서만 참이다. CUDA와 ROCm은 band를 유지하므로 gpt-oss는 두 트리 모두 26.21에서 26.43 GB를 피크로 쓰고 19,773 토큰 단일 패스는 양쪽에서 100.35 GB 할당을 요청한다. 이 사실을 백엔드 고유 성질로 적으면 회귀로도, 얻지 않은 이득으로도 읽히지 않는다.

**호스트에 맞춘 gate 호출을 받아들인다.** gate 기본값 6 job으로 돌린 첫 `make verify-rocm-held`는 이 30 GiB 머신에서 테스트 바이너리를 빌드하다 종료됐다. `ROCM_JOBS="-j 3"`으로 다시 돌린 실행은 모든 사전 검사와 워크스페이스 테스트를 통과했으므로, 기록은 gate를 낮추는 대신 job 상한과 그 이유를 적는다.

## 4. 측정 결과

decode는 `mlxcel-bench-decode`, deepseek-v2-lite-chat-4bit-mlx, `--ignore-eos`, tg128, 실행마다 60초 유휴를 둔 ABBA 순서다.

| 프롬프트 | decompressed | 조합 경로 | 변화 |
|---|---|---|---|
| 512, arm당 5회 (중앙값, 범위) | 42.72 (41.62–43.57) | 41.28 (36.94–41.78) | -3.4% |
| 2048 | 35.28 | 41.41 | +17% |
| 16384 | 22.04 | 36.49 | +65% |

16K에서 phase 피크는 16.06에서 12.39 GB로 줄고, 32K에서 KV 크기 추정은 8.44 GiB에서 972.0 MiB가 된다.

prefill은 GLM-4.7-Flash 2048 토큰 청크가 25.26에서 20.53 GB로 줄고 `generate` prefill이 919.2초에서 306.9초(3.0배)로 빨라졌다. #3114 이전 트리에서 62.72 GB 할당으로 실패하던 28K 단일 패스는 이제 21.13 GB에 들어간다.

gate는 `make verify-rocm-held ROCM_JOBS="-j 3"`이고 테스트 바이너리 192개, 15,954 통과, 0 실패, 524 무시다.

## 5. 학습 포인트

**빠진 근거 행은 코드 구멍처럼 보인다.** #3133은 코드 주석에 "ROCm과 CPU 백엔드는 측정하지 않았다"를 남기고 닫혔고 그 주석은 정확했다. 커널도 측정도 없는 백엔드에서는 보수적 기본값이 유일하게 방어 가능한 선택이었다. 바꾼 것은 호스트 한 대와, 형제 CUDA 기록에 이미 있던 측정 절차를 반나절 돌린 일이다.

**한 arm의 잡음이 판정의 부호를 바꾼다.** 같은 셀이 두 실행에서는 +5.6%, 다섯 실행에서는 -3.4%로 읽혔다. 분산이 레이아웃 사이가 아니라 decompressed arm 안의 실행 간 폭에 있었기 때문에, 두 arm을 가로질러 실행을 늘리지 않고 arm별로 반복한 것이 둘을 갈랐다.

## 6. 후속 조치

- #3153: DeepSeek V2 KV 추정은 int8 감소만 모델링하는데 런타임은 FP16이 아닌 모든 캐시 모드를 감소시킨다. 그래서 turbo-quant 캐시에서는 추정이 실제보다 작게 나올 수 있다. 이번 기본값 변경으로 그 경로가 기본으로 도달 가능해진다.
- Llama 4 Scout(#3117, #3126)는 이 호스트에 들어가지 않으므로 CUDA와 Metal 기록이 계속 담당한다.
