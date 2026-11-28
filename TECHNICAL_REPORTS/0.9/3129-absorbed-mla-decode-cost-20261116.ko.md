# 기술 보고서: PR #3129 - absorbed MLA decode에 quantized fold와 fused kernel 적용

**작성일**: 2026-11-16
**상태**: 완료 (항상 absorbed 경로를 쓰는 계열은 unit test로만 확인했고, DeepSeek V2 기본값 전환은 별도 변경으로 남김)
**위험도**: 중간 (absorbed MLA를 쓰는 모든 모델의 decode 핵심 경로에 새 Metal kernel이 들어감)

## 요약

absorbed MLA decode에는 단계마다 고정 비용이 있어서, DeepSeek V2의 latent 구조가 짧은 문맥에서 decompressed 구조보다 약 10% 느렸다(#3102). 512 key에서 profiling해 보니 비용은 세 부분에서 나왔다.

- dense f16 `W_UK`/`W_UV` fold: 0.36 ms
- rope-score mask: 0.17 ms
- fused되지 않아 kernel 지연에 묶인 512 폭 attention 블록: 0.93 ms

두 가지 변경으로 이를 해결했다(#3121).

- quantized `kv_b_proj`는 이제 자기 packed plane 위에서 batched `quantized_matmul`로 fold된다. V2-Lite에서 약 113 MiB가 30.4 MiB로 줄었다.
- mask 없는 single-query absorbed attention은 8192 key 이하 window에서 새 fused Metal kernel로 실행된다.

deepseek-v2-lite의 pp512/tg128 absorbed decode는 182.6에서 201.4 tok/s가 됐다. 기본값인 decompressed(205.5)와의 차이가 11%에서 2.0%로 줄었다. 2048 토큰에서는 6.2%, 16384 토큰에서는 56% 앞선다. teacher-forced trace의 decided position 불일치는 0개다.

## 3. 기술적 선택과 그 이유

- **key를 두 절반으로 나누고 병합한다.** decode에서 query는 한 행뿐이라, head당 threadgroup 하나만 쓰면 GPU 대부분이 논다. key를 절반으로 나누면 병렬성이 두 배가 되고, 병합 단계가 두 online softmax 상태를 정확하게 합친다.
- **8192 key 상한.** prototype 측정에서 fused와 조합형 attention 블록의 속도가 뒤집힌 지점이다. 이보다 긴 window, mask가 있는 단계, Metal이 아닌 backend, latent 폭이 32의 배수가 아닌 경우는 조합형 경로를 그대로 쓴다.
- **fallback.** 양자화되지 않은 층, NVFP4 global scale, runtime LoRA 항, quantization mode 불일치에는 dense fold가 남아 있다. `MLXCEL_MLA_FUSED_DECODE=0`으로 kernel을 끌 수 있다. 새 `absorbed_fused` 통계 경로가 있어서 trace와 benchmark마다 어느 경로를 탔는지 드러난다.
- **기본값은 여기서 바꾸지 않는다.** `docs/mla-absorbed-decode.md`가 DeepSeek V2 기본값을 absorbed로 바꾸는 조건으로 정한 두 가지가 이제 모두 충족된다. 하지만 기본값 전환은 모든 V2 사용자의 동작을 바꾸므로 별도 PR로 한다.

## 7. 변경 요약

| 프롬프트 | main absorbed | 브랜치 absorbed | 브랜치 fold만 | decompressed |
|---|---|---|---|---|
| 512 (4회 중앙값) | 182.6 | 201.4 | 187.8 | 205.5 |
| 2048 | 148~161 | 180.7 | | 170.1 |
| 16384 | 113.5 | 114.4 | | 73.5 |

16384 행은 window가 kernel 상한을 넘기 때문에 두 경우 모두 조합형 경로로 실행된다.

| trace, 폭 1 | 비교 대상 | decided 불일치 |
|---|---|---|
| 1024 뒤 | fused | 0 / 209 |
| 4096 뒤 | fused | 0 / 106 (256개 위치 모두 일치) |
| 8192 뒤 | 조합형 (상한 초과) | 0 / 78 |

테스트:
- 새 `mla::fused_decode_tests`는 kernel 결과를 host f64 reference와 비교하며, Apple Silicon에서 kernel이 동작하지 않으면 실패한다.
- `absorb_tests`에 quantized fold 케이스를 추가했다.
- `mlxcel-core` lib suite(2,328개)와 전체 `mlxcel` lib suite(11,986개)가 통과했고, workspace clippy도 경고가 없다.

## 8. 후속 조치

- 이번 측정을 근거로 DeepSeek V2 기본값을 absorbed 구조로 전환한다.
- 항상 absorbed 경로를 쓰는 계열을 처음부터 끝까지 실제로 측정한다(Kimi-VL은 `kv_b_proj`가 양자화되어 있지 않아 kernel만 적용된다).
