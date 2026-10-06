# 기술 보고서: PR #2115 - CUDA에서 VoiceChat TTS 백본 스트림을 f32로 유지 (이슈 #2109)

**날짜**: 2026-10-05

**상태**: GB10(`--features cuda`)에서 합성 가중치로 구현 및 검증 완료, 머지 대기 중.

**언어**: Rust (Nemotron VoiceChat EAR-TTS 로더, Gemma 3 백본, 테스트)

**위험도**: 낮음 (정확한 확장(widening)만 사용, 검증용 실제 체크포인트 없음)

## 요약

mlxcel의 MLX 승격 테이블 CUDA 오버레이(`src/lib/mlx-cpp/patches-cuda/dtype.cpp`, 이슈 #636)는 bf16과 f32를 bf16으로 결정합니다. EAR-TTS 레퍼런스는 bf16 가중치에 대해 백본을 f32로 실행하므로, 변환 없이 f32 스트림과 만나는 저장된 bf16 텐서는 CUDA에서 스트림을 bf16으로 강등시킵니다. PR #2093이 코드 임베딩과 MoG 헤드를 고쳤고, 이 PR은 남은 네 곳을 고칩니다. 오버레이와 공유 `GemmaRMSNorm`은 변경하지 않습니다.

## 1. 강등 지점

| 지점 | 연산 | 수정 |
|------|------|------|
| `bos_emb` | `multiply(bool, bf16)` 후 `add(f32, bf16)` | 로드 시 승격 (`promotes_to_f32`) |
| `audio_prompt_projection_W` | `matmul(f32, bf16)` | 로드 시 승격 |
| 백본 norm (레이어당 6개 + 최종) | `fast::rms_norm(f32, bf16 1 + w)` | `Gemma3Backbone::widen_norms_to_f32` |
| gated fusion 텍스트 분기 | `multiply(f32, bf16)` | 오디오 dtype으로 `astype` |

fusion 지점은 이슈에 없었습니다. `null_emb`는 이슈에 있었지만 `text_proj` 전에 bf16 subword 조건과 연결되므로 f32 스트림과 직접 만나지 않습니다. 이를 승격하면 비 CUDA 빌드에서 조건이 f32가 되어 `text_proj` 출력이 바뀝니다. 따라서 bf16으로 유지하고, 그 경로는 fusion 캐스트가 처리합니다.

## 2. Norm 확장

`GemmaRMSNorm::new`는 `1 + w`를 가중치 dtype으로 만들고 비공개로 보관합니다. `widen_norm_to_f32`는 `adjusted_weight()`로 만들어진 `a = 1 + w`를 읽어 f32로 캐스트한 뒤 f32의 `a - 1`로 새 norm을 생성합니다. half 정밀도 `w`에서 `a`는 0이거나 `2^-11`의 배수이므로(`w = -1` 근처에서 bf16 ulp는 `2^-8`, f16 ulp는 `2^-11`), `|a| < 2^24`이면 `a - 1`과 `1 + (a - 1)`이 모두 정확하고 재구성된 norm은 확장된 `a`를 비트 단위로 그대로 가집니다. 모든 forward 경로(일반, 배치 디코드, 양자화 fused QKV 커널)는 조정된 가중치만 읽고 fused 커널도 `fast::rms_norm`을 호출하므로 모두 f32 가중치를 사용합니다. `1 + w` 전에 `w`를 승격하는 방법은 Metal과 CPU에서 `1 + w`의 반올림이 달라지므로 채택하지 않았습니다.

## 3. 테스트

- `tts_dtype_tests::backbone_stream_is_f32_with_bf16_weights`: 모든 가중치를 bf16으로 저장한 작은 EAR-TTS로 프롬프트 조립(투영과 `bos_emb`) 후, fusion 후, 최종 norm 후, step에서 스트림 dtype을 확인합니다. 33c45053에서 `--features cuda`로 실행하면 첫 확인에서 실패합니다(`left: 12` bf16, `right: 10` f32).
- `gemma3_backbone_tests::widened_norms_hold_the_stored_one_plus_w_in_f32`: 확장된 모든 norm이 캐스트된 저장 `1 + w`와 바이트 단위로 같은지 확인하며, 경계값 `-1`, `-0.99609375`, `-1.0078125`, `0`, `+/-300`, `65504`를 포함하고 norm과 전체 백본을 지나는 f32 스트림도 확인합니다.
- `f32_promotion_keeps_norms_and_the_subword_path_as_stored`는 이제 `bos_emb`와 `audio_prompt_projection_W`를 승격 대상으로 나열하고 `null_emb`는 저장 dtype으로 유지합니다.

`warmup`을 분리하여 프롬프트 조립(`prompt_embeds`)과 fusion(`backbone_inputs`)을 따로 테스트할 수 있게 했습니다. latent 형태 검사가 코드 임베딩보다 먼저 실행된다는 점 외에 동작은 같습니다.

## 4. 검증

GB10, `--release --features cuda`, `gpu-lock` 아래 `--test-threads=1`: `models::nemotron_voicechat` 28/28, `models::gemma3_backbone` 6/6, `audio::f32_weights` 4/4. `cargo clippy --release --features cuda --lib --tests -- -D warnings`와 `cargo fmt --check` 통과.

비 CUDA 비트 동일성은 측정하지 않고 코드로 논증했습니다. CPU 전용 Linux 빌드는 링크되지 않습니다(이슈 #2108). 각 변경은 업스트림 승격이 `matmul`, `add`, `multiply`, `fast::rms_norm` 내부에 삽입하는 정확한 bf16에서 f32로의 `astype`이므로 커널은 같은 값을 받습니다. 유일하게 자명하지 않은 단계는 norm 테스트가 고정합니다.

실제 Nemotron VoiceChat 체크포인트는 사용하지 않았습니다. GB10에 없습니다. 양자화 fused QKV 경로는 실행하지 않았습니다(공개된 TTS 가중치는 dense).
