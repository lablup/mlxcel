# PR #3490: 남은 vision resize와 frame sampling parity 차이 정리 (issue #3478)

**작성일**: 2026-11-27

**상태**: `e705ac818`로 머지. 파일 8개, +312 / -65. M3 Ultra에서 로컬 검증했고, 원격 CI는 메인테이너 결정으로 취소했다. ERNIE와 Hunyuan-VL 체크포인트로 end to end 실행은 하지 않았다.

**위험도**: Low~Medium (얇은 이미지, 극단적 종횡비, 정확히 `.5`에 걸린 frame 위치에서 patch grid와 선택 frame이 바뀐다. 나머지는 그대로다)

## 요약

#3464, #3466 hardening 뒤에도 mlxcel의 이미지 resize나 비디오 frame 선택이 Python 기준 구현과 다른 곳이 네 군데 남아 있었다. 이 PR이 모두 맞춘다. frame index는 `np.round`처럼 half to even으로 반올림하고, Qwen2-VL 계열은 첫 반올림 단계에서 floor를 걸지 않으며, ERNIE 4.5 VL은 체크포인트가 함께 배포하는 `processing_ernie4_5_vl.py`(200:1 비율 clamp 포함)를 따르고, Hunyuan-VL은 0 길이 변에서 overflow하지 않는다. grid가 달라지면 vision token 수와 위치가 바뀌므로, 이 차이들은 픽셀 수만이 아니라 모델 입력 자체를 다르게 만들었다.

## 1. 문제 정의

| 차이 | 이전 | 기준 구현 |
|---|---|---|
| frame sampling (`uniform_indices`, `evenly_spaced_indices`, Gemma 4 `uniform_sample_frames`, Inkling `pair_adjacent_frame_indices`) | `f64::round`, half away from zero: `uniform_indices(6, 3)` = `[0, 3, 5]` | `np.linspace(...).round()`와 Python `round`, half to even: `[0, 2, 5]` |
| Qwen2-VL, dots.ocr, Hunyuan-VL 얇은 변 | 첫 단계 `max(1, round(e / f)) * f`: factor 28에서 10x1000이 28x1008 유지 | transformers `round(e / f) * f`가 0이 되어 `min_pixels` 아래로 떨어지고 rescale: 28x560 |
| ERNIE 4.5 VL 200:1 초과 | clamp 없음: 10x3000이 28x2996 | 체크포인트 코드가 비율을 clamp: 28x5600 |
| Hunyuan-VL 0 길이 변 | `smart_resize(0, 10)`에서 u32 연산 overflow | 기준 구현은 예외를 던지지만 mlxcel은 유한한 답이 필요하다 |

## 2. 기술적 선택과 그 이유

### 2.1 ERNIE가 따를 기준 구현

ERNIE에는 후보 기준이 셋 있고 서로 다르다.

| 선택지 | 얇은 변 10x1000 | `min_pixels` 아래 4x15 | 200:1 초과 10x3000 |
|---|---|---|---|
| transformers 5.16.1 `ernie4_5_vl_moe` | 28x560 (floor 없음) | 56x112 | 예외 |
| mlx-vlm 포트 | 28x1008 | 28x112 (`ceil` 전 `int()`) | 28x2996 |
| **선택: 체크포인트 `processing_ernie4_5_vl.py`** | 28x1008 | 56x112 | 28x5600 |

모든 ERNIE 4.5 VL 체크포인트(`baidu/ERNIE-4.5-VL-28B-A3B-PT`, `-Thinking`, 검증된 `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit`)가 `auto_map`으로 `Ernie4_5_VLImageProcessor`를 지정하므로, 학습 시 grid를 만든 것은 그 파일이다. transformers 클래스는 어떤 ERNIE 체크포인트도 로드하지 않는다. 그 결과 2번 항목(첫 단계 floor 제거)은 의도적으로 ERNIE에 적용하지 않았다.

### 2.2 한 factor floor가 여전히 남는 곳

첫 단계 floor를 없애면 변이 bounds 검사 전에 0이 될 수 있고, 이는 기준 구현과 같다. 최종 결과는 `min_pixels == 0`인 경우에 한해 여전히 한 factor로 floor한다. 기준 구현은 0 길이 변을 돌려주고 vision encoder는 빈 grid를 받게 되기 때문이다. 각 함수에 이 차이를 주석으로 남겼다.

### 2.3 Qwen2-VL의 200:1 trim을 두 분기 뒤로 이동

transformers는 200:1을 넘으면 예외를 던지고, mlxcel은 `max_image_tokens`가 지켜지도록 긴 변을 자른다. 이전에는 이 trim이 `max_pixels` 분기에서만 돌았다. floor가 사라지면 얇은 변이 작은 cap보다 긴 변을 가진 채 `min_pixels` 분기에 들어갈 수 있으므로, 이제 trim은 어느 분기 뒤에서든 돈다. `smart_resize_extreme_aspect_under_min_pixels_keeps_the_cap`이 이를 검증한다.

### 2.4 연산 폭

ERNIE와 Hunyuan은 `saturating_mul`과 함께 u64로 옮겼고, 0 길이 변은 1픽셀로 취급해 `beta`를 유한하게 유지한다. Hunyuan의 `smart_resize(0, 10)`은 overflow 대신 192x1632를 돌려준다. 결과는 `try_from(..).unwrap_or(u32::MAX)`로 u32에 되돌린다.

### 2.5 MiniMax-M3는 변경하지 않음

mlx-vlm 기준 구현에도 `max(factor, round_by_factor(..))`가 있으므로, 거기서의 floor는 차이가 아니라 parity다.

## 3. 변경 요약

| 파일 | 변경 |
|---|---|
| `src/multimodal/video.rs` | `uniform_indices`, `evenly_spaced_indices`, `pair_adjacent_frame_indices`가 half to even 반올림 |
| `src/vision/processors/gemma4.rs` | `uniform_sample_frames`가 half to even 반올림 |
| `src/vision/processors/qwen2_vl.rs` | 첫 단계 floor 제거, 200:1 trim을 두 분기 뒤로 |
| `src/vision/processors/dots_ocr.rs` | 첫 단계 floor 제거 |
| `src/vision/processors/hunyuan_vl.rs` | 첫 단계 floor 제거, 0 길이 변을 1픽셀로, saturating u64 |
| `src/vision/processors/ernie4_5_vl.rs` | 체크포인트 코드 방식 resize와 비율 clamp, u64 |
| `video_tests.rs`, `gemma4_tests.rs` | 기대 index 헬퍼가 half to even으로 반올림, 새 tie 케이스 |

기존 기대값은 하나도 바뀌지 않았다. 기존 테스트에서 바뀐 것은 `load_video` decode 헬퍼 하나로, index 12가 24가 되지만 color 허용 오차 안에 있다. 새 값은 모두 Python 기준 구현(transformers 5.16.1 `smart_resize`, 체크포인트 ERNIE 파일, numpy)으로 계산했고, 각 테스트의 doc comment에 계산 과정이 있다.

| 커밋 | 메시지 |
|---|---|
| `80d8f7c93` | fix: close remaining vision resize and frame-sampling parity gaps |

## 4. 검증

| 확인 | 결과 |
|---|---|
| `vision::processors::qwen2_vl` | 14개 통과 |
| `vision::processors::dots_ocr` | 6개 통과 |
| `vision::processors::ernie4_5_vl` | 8개 통과 (비율 clamp 테스트는 main에서 실패) |
| `vision::processors::hunyuan_vl` | 4개 통과 |
| `vision::processors::gemma4` | 21개 통과, 1개 ignored |
| `multimodal::video` | 57개 통과, 11개 ignored. ffmpeg fixture 테스트를 포함하면(`MLXCEL_TEST_VIDEO=1 ... --include-ignored`) 68개 통과 |
| clippy `-D warnings`, fmt | 경고 없음 |

## 5. 후속 조치

- 로컬에 ERNIE 4.5 VL과 Hunyuan-VL 체크포인트가 없어 새 grid를 실제 생성 결과와 대조하지 못했다.
- ERNIE 얇은 이미지와 4x15 테스트는 main에서도 통과하도록 설계했다. 나중에 "transformers parity" 변경이 ERNIE를 조용히 움직이지 않도록 체크포인트 동작을 고정하는 용도다.
- 새 Qwen2-VL 계열 processor는 #3490 이전 코드가 아니라 floor 없는 첫 단계를 따라야 한다.
