# PR #2459: LFM2-VL resampling, Qwen3-VL Conv3d layout guard, 양자화 pos_embed

**작성일**: 2026-10-19
**상태**: LFM2-VL, LFM2.5-VL, Qwen3.8-27B-FP8, Qwen3-VL 체크포인트로 CUDA (GB10)에서 구현 및 검증 완료, PR 열림

## 요약

#2388을 닫는다. upstream의 vision 수정 세 가지를 이식한다. 첫째, LFM2-VL position embedding을 bilinear antialias로 resample하고 processor는 설정된 resample filter를 쓴다. 둘째, 모든 Qwen3-VL tower 로드 지점에 Conv3d patch-embed layout guard를 둔다. 셋째, Qwen3-VL의 양자화된 `pos_embed`를 지원한다. 또한 첫 번째 항목의 검증을 막던 `lfm2-vl-1.6b-4bit` 로드 실패도 고친다.

## 문제 정의

- **LFM2-VL tower.** 학습된 2D position 테이블을 bicubic 보간으로 크기 조정했다. 기준 구현은 antialias가 있는 bilinear를 쓰므로, patch grid가 테이블 크기와 다른 모든 이미지에서 position embedding이 어긋났다.
- **LFM2-VL processor.** 체크포인트의 `resample` 값과 상관없이 항상 고정된 filter 하나로 이미지를 크기 조정했다.
- **Qwen3-VL Conv3d.** Qwen3-VL 계열 네 곳과 Cohere Compass가 `normalize_patch_embed_layout` 없이 vision tower를 만들었다. 그래서 hub layout의 patch weight(`[O, C, T, H, W]`)를 MLX layout으로 읽었다. Qwen3.8-27B-FP8은 도형 세 개가 있는 그림을 "빨강과 초록이 번갈아 나오는 세로 줄무늬"라고 설명했다.
- **Qwen3-VL pos_embed.** `PositionEmbedding`이 dense 테이블을 가정해 양자화된 테이블을 로드할 수 없었다.
- **LFM2-VL-1.6B 로드(이슈 범위 밖).** config는 레이어 27개와 `vision_feature_layer: -2`를 선언하지만 체크포인트에는 26개만 있어, `Weight not found: vision_tower.encoder.layers.26...`로 로드에 실패했다.

## 기술적 결정

- **공유 resample 행렬.** SAM의 `interp_matrix`를 `src/vision/encoders/resample_matrix.rs`로 옮겼다. 이제 cubic 또는 triangle kernel을 받고 separable contraction도 수행한다. `Learnable2DInterpPosEmb`에는 `PosEmbResample::BilinearAntialias`를 추가했다. 이 모드는 float32로 resample한 뒤 테이블 dtype으로 되돌리며, LFM2 tower가 `POSITION_RESAMPLE`로 선택한다. SAM은 이전과 같은 값을 낸다. MoonViT는 `Bicubic`을 유지하고 자체 edge-replicated 행렬을 쓰며, 비트 단위 동일성을 테스트로 고정했다.
- **processor filter는 체크포인트에서 읽음.** `processor_config.json`(중첩 또는 최상위)이 `preprocessor_config.json`보다 우선하고, 둘 다 없으면 bicubic이다. PIL id 0/1/2/3은 Nearest, Lanczos3, Triangle, CatmullRom에 대응한다. 그 밖의 id는 upstream처럼 조용히 대체하지 않고 로드 오류로 처리한다. filter는 크기 조정 지점 세 곳 모두에 전달된다.
- **layout guard용 로드 helper 하나.** `src/loading/vlm_qwen3_vision.rs`의 `load_qwen3_vl_vision_encoder`가 `normalize_patch_embed_layout`을 실행한 뒤 tower를 만든다. `vlm_qwen.rs`의 Qwen3-VL, Qwen3-VL-MoE, Qwen3.5/3.8, Qwen3-Omni와 Cohere Compass가 모두 이 helper를 쓴다. 어떤 파일이든 encoder 생성자를 다시 직접 호출하면 구조 테스트가 실패한다.
- **양자화 pos_embed.** `PositionEmbedding`은 `UnifiedEmbedding`을 갖고, `hidden_size`는 테이블 shape가 아니라 config에서 읽는다.
- **Vision feature layer.** LFM2 tower는 `vision_feature_layer`까지의 레이어만 유지하고 `post_layernorm`은 그대로 적용한다. HF `modeling_lfm2_vl.py`와 mlx-vlm과 같은 동작이다.
- **이미지 trace.** `examples/logit_trace`가 `MLXCEL_TRACE_IMAGE`와 `MLXCEL_TRACE_IMAGE_PROMPT`를 받는다. 대응하는 mlx-vlm 기준값은 `scripts/mlx_vlm_image_logit_trace.py`가 만든다. `--float32`는 CPU 전용 mlx 설치에서 쓰도록 모델을 dequantize한다.

## 변경 사항

- `src/vision/encoders/resample_matrix.rs`(신규), `kimi_vl_pos_emb.rs`, `lfm2_vl.rs`, `sam.rs`, `qwen3_vl.rs`: resample 모드, feature layer 범위, 양자화 pos_embed.
- `src/vision/processors/lfm2_vl.rs`, `lfm2_vl_tests.rs`: 설정된 resample filter.
- `src/loading/vlm_qwen3_vision.rs`(신규), `vlm_qwen.rs`, `vlm_cohere_compass.rs`: layout guard helper와 호출 지점 다섯 곳.
- `tests/fixtures/lfm2_vl/reference.json`과 `generate_reference.py`: torch 2.14.1+cpu와 mlx-vlm 0.7.8(git `73d366ec`) 기준값. 두 구현은 3e-8 이내로 일치한다.
- `examples/logit_trace/image.rs`, `scripts/mlx_vlm_image_logit_trace.py`, `docs/supported-models.md`, `docs/benchmarks.md`.

## 검증

GB10, `--profile test-fast --features cuda`. 테스트 바이너리는 `gpu-lock` 아래에서 `--test-threads=1`로 실행했다. `MLX_ENABLE_TF32=0`으로 `vision:: loading:: multimodal::`를 돌린 결과 1241개 통과, 0개 실패, 49개 무시였다. 새 테스트 25개는 TF32 기본값에서도 통과한다. clippy(`--lib --tests`와 `--example logit_trace`), fmt, Python tooling 테스트(124개)도 통과한다.

수정을 되돌리면 새 테스트 8개가 실패한다. hub/MLX patch-embed 동등성, dequantize한 테이블 대비 양자화 pos_embed, vision feature layer 범위, mlx-vlm 대비 LFM2 grid(12x20 0번 항목에서 0.1085 대신 0.0399), processor resample 테스트 4개다. 호출 지점 검사 테스트는 구조상 main에서 실패한다.

이미지 프롬프트 teacher-forced trace, 74개 위치, 212 토큰 프롬프트(mlx-vlm도 같은 212 토큰으로 렌더링). 기준은 mlx-vlm `73d366ec`를 CPU에서 dequantize한 float32로 실행한 결과다. reverted arm은 tower를 다시 bicubic으로 둔 것이며, LFM2.5에서 그 trace는 main 빌드와 바이트 단위로 같다.

| 체크포인트 | Arm | Top-1 불일치 | 기준 gap >= 2 위치 불일치 | Logit 차이 p50 / max | 기준 대비 ppl |
|---|---|---|---|---|---|
| LFM2.5-VL-1.6B-4bit | reverted | 4/74 | 0/36 | 0.193 / 0.673 | +2.36% |
| LFM2.5-VL-1.6B-4bit | fixed | 0/74 | 0/36 | 0.034 / 0.133 | +0.14% |
| LFM2-VL-1.6B-4bit | reverted | 0/74 | 0/41 | 0.092 / 0.396 | +1.25% |
| LFM2-VL-1.6B-4bit | fixed | 0/74 | 0/41 | 0.046 / 0.185 | +1.12% |

temperature 0에서 CLI와 `mlxcel-server`를 실행했다. LFM2 1.6B 체크포인트 둘 다 빨간 사각형, 파란 원, 초록 삼각형을 설명하고, 서버 출력은 212 프롬프트 토큰에서 CLI와 같다. LFM2-VL-450M은 CLI에서 자연스럽다. Qwen3.8-27B-FP8(hub patch weight `[1152, 3, 2, 16, 16]`)은 main에서 줄무늬를 설명하고, 수정 후에는 CLI와 서버 모두에서 도형을 올바르게 설명한다. `qwen3-vl-2b-4bit`(MLX layout, dense pos_embed)는 main과 수정본의 출력이 바이트 단위로 같다.

## 후속 작업

- 로컬 LFM2 체크포인트 세 개 모두 `resample: 2`를 선언하므로, 이 호스트에서 processor 변경은 수치에 영향이 없다.
- `pos_embed`를 양자화한 Qwen3-VL export가 로컬에 없어, 해당 항목은 합성 테스트로만 검증된다.
- SAM의 antialiased bicubic은 `a = -0.75`를 쓰지만, ATen과 mlx-vlm은 antialiased kernel에 `-0.5`를 쓴다. 이 PR에서는 비트 단위로 그대로 두었다.

## 통계

| 항목 | 값 |
|------|-------|
| 변경 파일 | 24 |
| 추가 줄 | 4799 |
| 삭제 줄 | 301 |

| 커밋 | 메시지 |
|--------|---------|
| `dda17a4b` | feat(examples): trace image-prompted continuations in logit_trace |
| `5686d0a4` | fix(vision): port LFM2-VL resampling, Qwen3-VL Conv3d guard and pos_embed |
| `3bf82f87` | test(vision): pin the LFM2-VL tower's resample mode to the mlx-vlm grid |
| `377664c9` | chore(scripts): mark mlx_vlm_image_logit_trace.py executable |

## 관련 항목

- `normalize_patch_embed_layout`은 다른 Qwen VL 로더용으로 이미 있었다. 이 PR은 모든 Qwen3-VL tower 지점이 이 함수를 거치게 한다.
