# 기술 보고서: PR #3138 - CUDA cuDNN 디코드 판정에서 버퍼 중간 KV slice 거부

**날짜**: 2026-11-16
**상태**: 완료
**위험도**: 낮음 (CUDA 오버레이의 디스패치 판정에 비교 하나 추가. 해당 호출은 MLX 자체 SDPA 커널로 돌아감)

## 요약

#3133의 CUDA 측정 중, GB10에서 Llama 4 Scout가 약 8448 토큰을 넘는 프롬프트 뒤에 쓰레기를 디코드했습니다(`Item` 뒤에 U+FFFD 연속). prefill 청크 여부와 무관했고, Metal은 정상이었으며, `MLXCEL_SDPA_DETERMINISTIC=1`로는 CUDA도 정상이어서 CUDA SDPA 오버레이의 cuDNN 디스패치가 의심되었습니다. 오버레이의 `is_kv_cache_slice`가 버퍼 중간에서 시작하는 KV view를 받아들였고, `unslice_kv`가 그 view를 버퍼 시작점부터 넓혀서 cuDNN이 디코드 윈도우가 아닌 버퍼 앞쪽의 key를 참조했습니다. 수정은 `kv_cache_slice_extent`에 이미 있던 것과 같은 `kv.offset() != 0` 거부를 추가하고, 수정 없이는 실패하는 회귀 테스트를 더합니다.

## 1. 문제 정의

Llama 4는 청크 로컬 어텐션(`attention_chunk_size` 8192)을 씁니다. 쿼리는 자기 청크 시작점 이후의 key만 봅니다. `ChunkedKVCache`는 첫 청크 이후의 디코드에 `keys[..., chunk_start:end, :]`를 넘기는데, 이 slice의 stride는 버퍼 전체의 것이고 `offset()`은 0이 아닙니다.

CUDA에서 key가 256개(`kv_cache_step`) 이상인 1행 디코드는 `use_cudnn_for_decoding`이 K/V를 "KV cache slice"로 판정하면 cuDNN으로 갑니다. 이때 오버레이는 `unslice_kv`를 호출해 `T_kv = strides(1) / strides(2)` 행, `offset = -kv.offset()`으로, 즉 할당 시작점부터 view를 다시 만들고, 실제 key 개수는 시퀀스 길이 패딩으로 cuDNN에 알려 줍니다. 그래서 cuDNN이 읽는 key는 윈도우가 아니라 버퍼의 첫 `k_len`행입니다.

`is_kv_cache_slice`(오버레이가 그대로 둔 upstream 판정)는 넓힌 view의 원소 수가 할당과 같은지만 확인했습니다. 시퀀스 축으로 자른 slice는 stride가 그대로라서 이 판정을 통과합니다. prefill 버킷팅 경로가 쓰는 오버레이 자체 확장판 `kv_cache_slice_extent`는 이미 0이 아닌 offset을 거부했지만, 주석에 "그런 slice를 넘기는 mlxcel 코드는 없다"고 적혀 있어 미래 대비용 가드로 취급되고 있었습니다.

발동 조건이 기존 확인을 빠져나간 이유를 설명합니다.
- 윈도우가 key 256개에 닿아야 하므로 프롬프트가 청크 경계보다 256 토큰 이상 길어야 합니다(Scout는 8448). 8720 토큰 프롬프트는 529 key 윈도우를 만들었습니다(`MLXCEL_SDPA_PLAN_DEBUG=1`에서 `k_len=529 decode=1`).
- 8001 토큰 prefill 후 8192를 넘어가는 디코드는 decided 0/100이었습니다. 추적이 8448 위치에서, 새 청크의 윈도우가 막 256 key에 닿을 때 끝났기 때문입니다. 마지막 세 스텝이 결함의 시작점으로, 참조 대비 NLL이 13.1 vs 4.3, 23.2 vs 10.2, 12.4 vs 5.5였고, decided 필터에 들지 않는 위치였습니다.
- 9216 토큰 단일 forward와 4608 두 청크 비교도 일치했습니다(0/3642). prefill만 돌리므로 디코드 디스패치를 타지 않습니다.
- Metal은 이 오버레이를 쓰지 않습니다.

## 3. 기술적 선택과 그 이유

- **생산자가 아니라 디스패치 판정에서 거부.** `ChunkedKVCache`가 stride를 가진 윈도우를 넘기는 것은 정상적인 MLX view입니다. 버그는 할당 크기와 맞는 view가 모두 시작점에서 시작한다고 가정한 소비자 쪽에 있습니다. 판정을 고치면 현재와 미래의 모든 생산자가 보호되고, 거부된 호출은 view의 offset을 존중하는 MLX 자체 SDPA 커널로 갑니다.
- **기존 가드를 그대로 따름.** `kv_cache_slice_extent`에 같은 검사와 설명이 이미 있었습니다. 새 주석은 그쪽을 가리키고, 이 경우를 가상의 상황으로 적은 옛 주석은 chunked cache가 실제 생산자라는 내용으로 고쳤습니다.
- **특정 PR에 원인을 돌리지 않음.** 쓰레기 출력은 #3114, #3119 이전 빌드(#3117의 청크 로컬 디코드 재작성보다도 앞섬)에서도 재현되었습니다. 그래서 보고서와 코드 주석은 메커니즘만 적고 도입 변경을 지목하지 않습니다. 버퍼 중간 윈도우가 언제부터 이 경로에 닿았는지는 확인되지 않았습니다.
- **캐시 수준에서, 실제 디스패치로 테스트.** 회귀 테스트는 f16, 청크 512, 800 토큰 prefill로 `ChunkedKVCache::attend`를 돌려 첫 디코드 윈도우가 앞쪽 trim으로 할당 중간에 놓인 289 key가 되게 하고, `chunked_reference`(mlx-lm식 마스크 계산)와 비교합니다. CUDA 빌드에서는 수정 없이 cuDNN 분기를 타고, Metal이나 CPU에서는 일반 정합성 테스트입니다.

## 7. 변경 요약

| 영역 | 변경 |
|---|---|
| `overlays/cuda/.../scaled_dot_product_attention.cpp` | `is_kv_cache_slice`가 `kv.offset() != 0`이면 false 반환. `kv_cache_slice_extent` 주석 정정 |
| `mlxcel-core/src/cache/attend_tests.rs` | `chunked_decode_over_a_long_mid_buffer_window_matches_the_chunk_mask` 추가 |

GB10(sm_121, CUDA 13.0) 검증:

| 확인 | 수정 전 | 수정 후 |
|---|---|---|
| Scout, 8720 토큰 프롬프트, `--temp 0`, `MLXCEL_PREFILL_CHUNK=0`과 기본값 | 쓰레기 | 정답(item 275, bin 6206, weight 136), 두 모드 동일 |
| 8720 prefill 후 300 위치 teacher-forced 디코드, 같은 텍스트의 9216 폭 단일 prefill forward 대비 | 첫 스텝 NLL 21.5 vs 2.85, 이후 non-finite | decided 0/149, 평균 NLL 1.853 vs 1.854 |
| 새 회귀 테스트 | 실패, relative rms 1.12 | 통과 |

수정 후 8720 토큰 프롬프트에서 Scout 처리량은 prefill 단일 패스 633 tok/s, 기본 청크 451 tok/s, 디코드 10.9 / 11.5 tok/s입니다. 수정 전 출력이 틀렸으므로 의미 있는 비교 기준은 없습니다. CUDA에서 `cache::attend` 테스트는 모두 통과했고, PR CI도 통과했습니다(20개 통과, 12개 path filter로 skip).

## 8. 후속 조치

- ROCm은 확인하지 않았습니다. ROCm 백엔드에 비슷한 slice 판정과 시작점 재설정 헬퍼가 있다면 같은 가드가 필요합니다.
- 일반 debug 프로파일에서 `cache::paged_detach` 테스트 2개가 `paged_detach.rs:1095`의 `debug_assert`에 걸립니다. 이 변경 유무와 무관하고 `test-fast`에서는 통과합니다. SDPA와 무관하므로 별도로 추적합니다.
- GB10에서 8720 토큰 Scout의 단일 패스 prefill(약 630 tok/s)이 기본 청크(약 450 tok/s)보다 빨랐습니다. CUDA에서 Llama 4의 기본 청크 크기를 ABBA 측정으로 재검토할 만합니다.
