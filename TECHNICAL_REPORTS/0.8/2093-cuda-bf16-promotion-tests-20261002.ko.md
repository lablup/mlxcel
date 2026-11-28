# 기술 보고서: PR #2093 - CUDA 오버레이가 강등하는 곳에서 bf16 테이블을 명시적으로 캐스트 (issue #2087)

**날짜**: 2026-10-02

**상태**: GB10(`--features cuda`)과 CPU 전용 Linux 빌드에서 구현 및 검증 완료, 머지 대기.

**언어**: Rust (Nemotron VoiceChat TTS 헤드의 프로덕션 코드, 테스트)

**위험도**: 낮음 (TTS 경로에 정확한 `astype` 캐스트 두 개를 추가했고, 비 CUDA 빌드에서는 no-op이다. 나머지는 테스트 코드다)

## 요약

GB10에서 PR #2077을 게이트하던 중 lib 테스트 6건이 실패했고, 베이스에서도 똑같이 실패했다. 이 중 5건은 원인이 하나다. mlxcel이 직접 둔 MLX 승격 테이블의 CUDA 오버레이 `src/lib/mlx-cpp/patches-cuda/dtype.cpp`가 bf16과 f32의 조합을 upstream의 f32가 아니라 bf16으로 결정한다. 고정된 MLX 버전의 버그도 아니고 `mlxcel-core` 래퍼 문제도 아니다. 이 오버레이는 의도된 것(issue #636, 단일 dtype bf16 디코드 그래프)이므로 그대로 둔다. 테스트와 프로덕션 코드 두 곳이 upstream 승격 규칙을 전제하고 있었다. 여섯 번째 실패인 `gelu_approx_matches_mlx_nn_bit_for_bit`는 main에서 이미 #2080으로 수정되었고, 변경 없이 CUDA에서 통과한다.

## 1. 근본 원인

`dtype::BFLOAT16 == 12`, `dtype::FLOAT32 == 10`이므로 `left: 12, right: 10` 단언은 혼합 dtype 연산이 bf16을 돌려줬다는 뜻이다. 오버레이 헤더에 변경 내용이 적혀 있다. bf16 행의 f32 열과 f32 행의 bf16 열이 모두 `bfloat16`이다. 이 테이블은 MLX 라이브러리에 컴파일되므로 CUDA 빌드에서는 `MLXCEL_DEVICE=cpu`를 포함한 모든 디바이스에서 이 규칙이 적용된다. ROCm은 upstream `dtype.cpp`를 컴파일하고 Metal은 `patches-cuda/`를 보지 않으므로, 규칙을 바꾸는 것은 `cuda` 피처뿐이다.

Nemotron VoiceChat TTS 경로의 두 곳은 테스트만의 문제가 아니라 CUDA에서 실제 출력 dtype 결함이었다.

- `RvqCodebooks::depthsum_embedding`은 f32 합을 반환한다고 문서화되어 있지만, gather한 bf16 코드북 행을 f32 `zeros`에 더했으므로 CUDA에서는 합이 bf16이었다.
- `MogHead::infer`는 `proj_mus`와 `low_mat`(설계상 bf16 그대로 보관, `promotes_to_f32` 참조)에서 gather한 슬랩을 f32 활성값과 matmul했으므로, `mu`와 그에 따라 샘플링된 latent가 bf16이었다.

## 2. 변경 사항

- **프로브 테스트** `audio::f32_weights::tests::mixed_bf16_f32_promotion_follows_the_build`: f32 활성값과 bf16 피연산자에 대해 `add`(양쪽 순서), `matmul`(양쪽 순서), `fast_rms_norm`의 출력 dtype을 단언한다. `cuda` 피처가 없으면 f32, 있으면 bf16을 기대하므로 어느 쪽 규칙이 바뀌어도 바로 실패한다. 이슈는 양쪽 모두 f32를 요구했지만 오버레이를 유지하는 한 성립할 수 없으므로, 프로브는 실제로 성립하는 계약을 고정하고 프로덕션 코드는 명시적으로 캐스트한다.
- **`rvq.rs`**: gather한 행을 더하기 전에 f32로 캐스트한다.
- **`mog_head.rs`**: gather한 `mus`와 `low` 슬랩을 matmul 전에 활성값 dtype으로 캐스트한다. bf16에서 f32로의 변환은 정확하고 upstream 승격이 삽입하는 것과 같은 `astype`이므로 비 CUDA 출력은 비트 단위로 같으며, dtype이 이미 같으면 no-op이다.
- **`f32_weights.rs` 테스트**: "승격된" 기준값은 CUDA가 아니면 MLX의 암묵적 승격, CUDA에서는 그래프 안의 명시적 `astype(weight, float32)`(`as_promoted`)이다. 모듈 문서에 CUDA에서는 로드 시점 사전 캐스트가 단지 빠른 것이 아니라 필수라는 점을 적었다.
- **`fastconformer/tests.rs`**: 기준 인코더와 프로젝션을 암묵적 승격에 기대지 않고, bf16 가중치를 f32로 캐스트한 값(upstream 승격이 계산하는 값)으로 구성한다.
- **`tts_tests.rs`**: `mog_infer_shapes_are_finite_with_guidance`가 `RvqEarTtsModel`과 같은 방식(`promoted_subset`과 `promotes_to_f32`)으로 가중치를 로드한다. 프로덕션이 만들지 않는 전부 bf16인 헤드가 아니라 실제 로드 경로를 검증한다.

## 3. gelu 테스트

`gelu_approx_matches_mlx_nn_bit_for_bit`는 Metal에서 측정한 값을 하드코딩하고 있었다. #2080(`5486e404`, 2026-09-30 머지)이 이를 같은 디바이스에서 `mlx.nn.gelu_approx`를 옮겨 쓴 기준과 4097개 점에서 비트 단위로 비교하는 방식으로 바꿨다. 1 ulp 차이는 백엔드별 `tanh`/`power` 커널의 반올림 차이였다. 양쪽 모두 컴파일되지 않고 matmul도 없는 원소별 그래프이므로 fusion, 연산 간 FMA 축약, TF32와는 무관하다. 이 PR의 변경 없이 CUDA와 CPU 빌드에서 통과한다.

## 4. 검증

GB10, `--features cuda --release`, 각 실행은 `gpu-lock` 아래에서 `--test-threads=1`:

| 필터 | 결과 |
|---|---|
| `audio::f32_weights` (프로브 포함) | 4 통과 |
| `audio::fastconformer` | 14 통과 |
| `models::nemotron_voicechat::tts` | 13 통과 |
| `models::gemma3_backbone` (gelu 포함) | 5 통과 |

테스트는 그대로 두고 프로덕션 캐스트 두 개만 되돌리면 CUDA에서 `depthsum_ignores_mask_index`와 `mog_infer_shapes_are_finite_with_guidance`가 실패한다. 즉 두 테스트를 통과시키는 것은 이 캐스트다.

CPU 전용 Linux 빌드(GPU 피처 없음, 별도 target 디렉터리): 같은 네 필터가 같은 개수(4, 14, 13, 5)로 통과하며, 여기서는 프로브가 f32를 단언한다. 오버레이가 라이브러리에 컴파일되어 해당 빌드의 모든 디바이스에 적용되므로, CUDA 빌드의 `MLXCEL_DEVICE=cpu`는 대체 검증으로 쓰지 않았다.

## 5. 남은 항목

- **CUDA에서 TTS 백본은 여전히 bf16으로 강등된다.** `RvqEarTtsModel`은 Gemma 노름 가중치(`1 + w`를 저장 dtype으로 계산), `bos_emb`, `null_emb`, `audio_prompt_projection_w`를 bf16 그대로 두며, 각각이 f32 스트림과 만난다. CUDA가 아니면 reference처럼 스트림을 f32로 승격하지만, CUDA에서는 프로브가 `add`, `matmul`, `fast_rms_norm`에 대해 고정한 규칙에 따라 bf16으로 강등한다. 이번 수정으로 코드 임베딩과 MoG 헤드는 f32가 되었지만, 그 사이의 백본은 CUDA에서 여전히 bf16이다. 영향을 확인하려면 실제 Nemotron VoiceChat 체크포인트가 필요한데 이 호스트에는 없으므로 후속 작업으로 남긴다.
- **main에서 일반 Linux CPU lib 테스트 빌드가 링크되지 않는다.** `src/lib/mlx-cpp/turbo/kv_inplace_write.cpp`(#1961)가 GPU 없는 MLX 빌드에는 정의되지 않은 `mlx::core::copy_gpu_inplace`를 참조하므로, `cargo test --release --lib`는 테스트를 실행하기 전에 링크 단계에서 실패한다. 위 CPU 결과는 이 미해결 심볼 하나를 무시하는 링커 래퍼(`-Wl,--unresolved-symbols=ignore-in-object-files`)로 얻었다. 이 심볼은 GPU 스트림에서만 도달한다.
