# PR #2730: Qwen3 conditioner f32 attention 부정적 결과 기록

**날짜**: 2026-10-30
**상태**: 완료(부정적 결과 기록, 동작 변경 없음). CUDA(GB10)에서만 측정, Metal과 ROCm은 실행 안 함
**언어**: Rust(주석, ignored 테스트 1개), Markdown
**위험도**: 낮음(`src/models/qwen3.rs`는 main과 바이트 단위로 동일하며, 추가된 코드는 ignored 진단 테스트뿐)

## 요약

PR #2730(이슈 #2666 종료, 에픽 #2348)은 하나의 가설을 검증하고 그 가설이 틀렸음을 보고합니다. #2666은 공유 Qwen3 conditioner의 attention을 업스트림 mlx-vlm `73d366ec`(`models/flux2/qwen/attention.py:148-162`)처럼 f32로 실행하면 FLUX.2, Z-Image, Bonsai, Ideogram 4의 패리티 드리프트 일부가 줄어들 것이라고 제안했습니다.

이를 opt-in 플래그로 구현해 GB10에서 CUDA 오라클과 비교했습니다. f32 attention은 conditioning을 업스트림에 가깝게 만들지 않습니다. FLUX.2 prompt embeds는 두 경우 모두 업스트림과 rel-RMS 1.19e-2 떨어져 있습니다. end-to-end에서는 bf16 노이즈 드리프트의 양상만 바뀌며, Bonsai(30 dB 게이트에 29.21 dB)와 Ideogram 4(pooled latent 4.74e-2, 게이트 4.5e-2)가 커밋된 게이트를 넘어섭니다. 그래서 플래그를 제거했습니다. 이 PR은 수정된 주석과 문서, ignored 진단 테스트 1개를 담습니다. 실제 원인으로 의심되는 bf16 GEMM reduction order와 mlx 0.32.4 대비 커널 선택은 #2732에서 추적합니다.

## 1. 문제 정의

네 이미지 패밀리(FLUX.2, Z-Image, Bonsai, Ideogram 4)가 `Qwen3TextEncoder`를 공유합니다. 각 패리티 테스트 주석은 업스트림 대비 conditioner 드리프트의 원인을 "bf16 attention"으로 적고 있었습니다. 구체적인 단서는 실제 차이 하나뿐이었습니다. 업스트림은 SDPA에서 q, k, v를 f32로 캐스팅하고 출력을 query dtype으로 되돌리는 반면, mlxcel의 공유 Qwen3 attention은 activation dtype을 유지합니다.

#2666은 이 캐스팅을 요구했습니다. 목표 conditioner 거리 1.9e-3은 업스트림이 자신의 attention dtype 변화에 보이는 민감도에서 가져온 값이었습니다. 남은 질문은 포트의 1.19e-2 잔차가 같은 효과인지였습니다.

## 2. 변경 요약

브랜치 이력은 세 단계이며, 머지되는 결과물은 마지막 단계뿐입니다.

| 커밋 | 내용 |
|---|---|
| `670c34ce` | `src/models/qwen3.rs`의 공유 `Attention`에 `attend_in_f32`(모든 곳에서 false, 꺼져 있으면 그래프 불변)와 `Qwen3Model::set_attention_f32`를 추가. 캐스팅은 fused-decode와 graph 분기가 합쳐진 뒤 `cache.attend` 이전에 위치하고, 출력은 `o_proj` 이전에 query dtype으로 복귀. 작은 인코더 테스트, 두 경우를 모두 보고하는 ignored FLUX.2 prompt-embeds 테스트, 실제 Qwen3 logits 테스트의 완화된 f32 검사를 추가. |
| `3ff261c7` | `origin/main` 머지. |
| `63eafc08` | 플래그 제거. `src/models/qwen3.rs`는 main과 동일하게 복귀하고, 측정 결과는 플래그를 언급하지 않는 주석과 문서로 남김. |

이 PR이 담는 것(6개 파일, +55 / -6):

| 파일 | 변경 |
|---|---|
| `src/imagegen/qwen3_text_encoder.rs` | 모듈 헤더와 `from_weights` 주석에 attention이 가중치 dtype으로 실행되며 업스트림은 f32를 쓴다는 점과 측정 결과를 명시. |
| `src/imagegen/bonsai/parity_tests.rs`, `z_image/mod_tests.rs`, `ideogram4/parity_tests.rs` | 주석이 더 이상 드리프트를 bf16 attention 탓으로 돌리지 않고 f32 attention 수치를 기록. |
| `docs/image-generation.md` | Qwen3 conditioner, Z-Image, Ideogram 4 섹션을 같은 방식으로 정정. |
| `src/imagegen/flux2/parity_tests.rs` | 새 ignored 테스트 `klein_4b_prompt_embeds_match_upstream`: generate 프롬프트를 파이프라인의 bucketed `encode` 경로로 인코딩하고 업스트림 `embeds_bf16_bits`(`[1, 32, 7680]`) 대비 rel-RMS를 보고. 측정값 1.190e-2이며 assert 0.1은 패리티 게이트가 아니라 sanity 경계. |

경계, 노이즈 검사, PSNR 게이트는 하나도 바꾸지 않았습니다.

## 3. 측정 결과

모든 실행은 GB10에서 `MLX_ENABLE_TF32=0`으로, CUDA 오라클(mlx 0.32.4, mlx-vlm `73d366ec`)을 기준으로 했습니다.

업스트림까지의 conditioner 거리(rel-RMS):

| Conditioner 출력 | bf16 attention(main) | f32 attention |
|---|---|---|
| FLUX.2 prompt embeds `[1, 32, 7680]` | 1.190e-2 | 1.192e-2 |
| Z-Image `cap0` (turbo / base / edit) | 9.308e-3 / 9.308e-3 / 9.303e-3 | 9.349e-3 / 9.349e-3 / 9.314e-3 |
| Ideogram 4 텍스트 피처 | 6.4013e-2 | 6.4005e-2 |
| Bonsai prompt embeds `[1, 512, 7680]` | 2.306e-2 | 2.228e-2 |

end-to-end(최종 latent rel-RMS, PNG PSNR). 초기 노이즈는 모든 경우에 비트 단위로 같았습니다.

| 패밀리 / 케이스 | main(= 이 PR) | f32 attention | 게이트 |
|---|---|---|---|
| FLUX.2 generate | 1.026e-1, 32.27 dB | 1.094e-1, 32.02 dB | 0.2, 30 dB |
| FLUX.2 edit | 4.109e-2, 40.54 dB | 4.244e-2, 39.07 dB | 0.2, 30 dB |
| Bonsai generate | 1.001e-1, 31.18 dB | 1.348e-1, **29.21 dB(실패)** | 0.2, 30 dB |
| Z-Image turbo | 8.607e-2, 32.86 dB | 6.402e-2, 34.53 dB | 1.2e-1, 30 dB |
| Z-Image base | 6.198e-2, 34.71 dB | 9.965e-2, 30.19 dB | 1.0e-1, 30 dB |
| Z-Image edit | 1.725e-2, 46.14 dB | 1.758e-2, 45.50 dB | 5e-2, 30 dB |
| Ideogram 4 turbo_512(pooled latents, pooled image) | 3.16e-2, 43.72 dB(PNG 37.13 dB) | **4.74e-2(실패)**, 38.37 dB | 4.5e-2, 38 dB |

Ideogram 4의 f32 실행은 pooled-latent assert가 먼저 실패해 전체 PNG PSNR까지 도달하지 못했습니다.

오라클 출처: FLUX.2와 Bonsai는 커밋된 fixture(Bonsai step-0 입력은 CUDA 오라클 venv로 실행한 `generate_bonsai_fixtures.py --step0`), Z-Image는 에픽의 오라클 디렉터리, Ideogram 4는 PR #2427의 오라클 디렉터리를 사용했습니다.

## 4. 기술적 선택과 그 이유

### 4.1 플래그 제거

**배경:** f32 attention은 이 모듈에서 포트와 업스트림 사이의 문서화된 유일한 차이였기에 가장 자연스러운 후보였습니다.

**근거:**

- 업스트림 자신의 bf16-to-f32 attention 전환은 FLUX.2 prompt embeds를 1.03e-3만큼 움직입니다. 포트의 f32 attention은 9.4e-4만큼 움직여 같은 크기이므로, 포트가 그 효과를 재현하는 것은 맞습니다.
- 그러나 포트는 두 경우 모두 업스트림에서 1.19e-2 떨어져 있습니다. attention dtype이 설명하는 양보다 한 자릿수 크고, 두 변화량은 상관이 없습니다(코사인 -0.12). 즉 attention dtype은 격차를 만드는 경로에 있지 않습니다.
- Z-Image와 Ideogram 4 피처도 움직이지 않습니다(두 경우 모두 9.3e-3, 6.4e-2). Bonsai는 조금 개선되지만(2.306e-2에서 2.228e-2) end-to-end에서 1.97 dB를 잃습니다.
- end-to-end 결과는 양방향으로 움직이며(Z-Image turbo는 개선, Z-Image base는 3.7 dB 악화) conditioner 거리와 무관한 양상입니다. 증폭된 반올림 노이즈의 특징입니다.

**결정:** 플래그를 제거합니다. 프로덕션 호출처가 없는 휴면 스위치를 라이브 텍스트 모델의 attention에 두면 이득 없이 표면만 늘고, 켜면 커밋된 게이트 두 개가 실패합니다. 존재하던 동안 기본값은 꺼짐이었고, teacher-forced `logit_trace` 실행은 `qwen3-0.6b-4bit`와 bf16 Qwen3-4B 텍스트 인코더에서 main과 바이트 단위로 같았습니다. 제거 후 파일은 main과 바이트 단위로 동일합니다.

**기각한 대안:**

- 향후 용도로 기본값 꺼짐 상태로 유지: 위의 표면 문제로 기각. 결과는 기록되어 있고 플래그는 몇 줄이면 다시 넣을 수 있습니다.
- conditioner에서만 활성화: Bonsai와 Ideogram 4 게이트를 깨므로 기각.
- Bonsai나 Ideogram 4 경계를 완화해 수용: 업스트림과의 거리를 줄이지 못하는 변경을 위해 게이트를 내주는 일이므로 기각.

### 4.2 진단 테스트 유지

`klein_4b_prompt_embeds_match_upstream`은 FLUX.2의 conditioner 거리를 최종 latent로 추론하지 않고 직접 측정하는 유일한 곳입니다. 체크포인트와 GPU가 필요해 `#[ignore]`이므로 CI에 영향이 없고, 0.1 경계는 심각한 고장만 잡습니다. #2732가 비교할 기준 수치를 제공합니다.

### 4.3 이전 귀속을 남기지 않고 주석을 정정

이전 주석은 다음 독자를 attention dtype으로 이끌었는데, 이제 그것은 배제되었습니다. 새 주석은 측정한 내용과 원인이 확정되지 않았다는 점을 적습니다.

## 5. 검증

모두 GB10, `MLX_ENABLE_TF32=0`, `--test-threads=1`.

| 필터 | 통과 | 실패 | ignored |
|---|---|---|---|
| `imagegen::qwen3_text_encoder` | 17 | 0 | 1 |
| `imagegen::flux2` | 29 | 0 | 7 |
| `models::qwen3` | 98 | 0 | 21 |

ignored인 `klein_4b_prompt_embeds_match_upstream`도 통과합니다(1.190e-2). `cargo clippy --profile test-fast --features cuda -p mlxcel --lib --bins --tests --no-deps -- -D warnings`, `cargo fmt --check`, `make verify-*` 타깃, `check_binary_assets.py`, `check_env_var_docs.py`도 통과합니다.

Mage-Flow, Qwen-Image, LLaDA-Image, ERNIE-Image, Ming은 `Qwen3TextEncoder`를 거치지 않아 영향이 없습니다.

## 6. 학습 포인트

- 재현된 업스트림 민감도는 재현된 격차가 아닙니다. 포트의 f32 attention이 업스트림과 같은 1e-3만큼 출력을 움직였다는 것은 캐스팅이 동작함을 보일 뿐이고, 1.19e-2 잔차는 다른 양입니다. 수정을 기대하기 전에 효과의 크기와 격차의 크기를 비교해야 합니다.
- 크기뿐 아니라 변화량의 상관도 봐야 합니다. attention dtype 변화량과 포트-업스트림 잔차 사이의 코사인 -0.12는 둘이 독립임을 말해 줍니다.
- conditioner에 중립인 변경도 end-to-end PSNR을 양방향으로 몇 dB 움직일 수 있습니다. 이런 변경은 conditioner 거리로 먼저 판단하고, conditioner가 움직이지 않았다면 end-to-end 변동은 노이즈로 취급해야 합니다.
- 기준 구현이 자기 설정에 보이는 민감도에서 가져온 목표치(1.9e-3)는 포트가 도달할 수 있는 수치가 아닙니다.

## 7. 후속 작업과 한계

- 1.19e-2 conditioner 잔차의 원인은 확정되지 않았습니다. 현재 의심은 이 빌드의 MLX와 mlx 0.32.4 사이에서 bf16 GEMM reduction order와 커널 선택이 다르다는 것입니다. 여기서는 측정하지 않았으며 #2732에서 추적합니다.
- CUDA(GB10)에서만 측정했습니다. Metal과 ROCm은 실행하지 않았습니다.
- f32 경로는 더 이상 트리에 없습니다. 재현하려면 `670c34ce`를 다시 적용해야 합니다.
- 레이어별 conditioner 거리는 수집하지 않아 잔차가 처음 나타나는 위치는 알 수 없습니다.

## 참고

- 이슈 #2666(종료), #2732(후속), 에픽 #2348, 메인테이너 결정 #932(패리티 계약).
- 관련: PR #2427(Ideogram 4 오라클), PR #2717(LLaDA-Image).
- 업스트림: mlx-vlm `73d366ec`, `models/flux2/qwen/attention.py:148-162`.
- 문서: `docs/image-generation.md`.
