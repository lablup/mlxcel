# PR #2841: OmniParser v2 아이콘 검출용 YOLO11 디텍터

**Date**: 2026-11-03
**Status**: Completed (머지됨, #2413 닫음)

## 요약

PR #2841은 OmniParser v2 `icon_detect` 체크포인트를 위한 네이티브 YOLO11 추론을 추가해 OmniParser의 검출 절반을 완성한다. `mlxcel detect`는 이제 `model_type: yolo11`을 mlx-vlm 그래프 포팅으로 디스패치한다. C3k2 백본, SPPF, C2PSA 어텐션 넥, 스트라이드 8/16/32의 anchor-free DFL 헤드다. 전처리와 후처리는 mlx-vlm의 `prepare_image`와 `non_max_suppression`을 따른다. 114 회색 캔버스에 중앙 정렬한 스트라이드 32 레터박스, 클래스 인지 그리디 NMS, 원본 픽셀로 되돌리는 역변환 순서다.

공유 디스패치 계약도 바뀌었다. `load_detector`가 `Option<f32>`를 받아 `--threshold`가 없으면 각 패밀리의 기본값을 쓴다. `mlxcel detect`에는 `--size`(가장 긴 변, 미지정 시 원본 해상도)와 `--iou`(NMS, 기본 0.1)가 추가됐다. 두 플래그는 YOLO11 전용이며 다른 디텍터에 주면 무시된다는 경고를 낸다. 10개 파일, +2114/-19이며 그중 약 600줄이 테스트와 문서다.

## 1. 변경 요약

| 영역 | 변경 |
|---|---|
| `src/vision/detection/yolo11/` (신규) | `config.rs`, `layers.rs`, `model.rs`, `weights.rs`, `predictor.rs`, 단위 테스트. 정확한 텐서 스키마와 미사용 키 보고 |
| `src/vision/detection/mod.rs` | `detector_info` / `load_detector`에 `yolo11` 추가, threshold를 `Option<f32>`로, `DetectorOptions`와 `predict_path_with_options` |
| `src/commands/detect.rs` | 런타임 초기화 전 `--size` / `--iou` 검증, 무시 경고, 디텍터로 옵션 전달 |
| `src/main.rs` | `DetectArgs` 플래그와 도움말 |
| `docs/supported-models.md` | YOLO11 행과 OmniParser 검출-후-캡션 레시피 |

## 2. 기술적 선택

### 2.1 OmniParser 토폴로지를 고정 테이블로 포팅

이 체크포인트는 하나의 고정 토폴로지이지 패밀리가 아니다. mlx-vlm의 레이어 테이블을 그대로 데이터로 옮겼고, `Yolo11Config::validate`는 `ch`나 `stride`가 `[256, 512, 512]` / `[8, 16, 32]`와 다르면 거부한다. 다른 깊이나 폭은 이 그래프로 표현할 수 없으므로, 그럴듯하지만 틀린 순전파를 내는 대신 로드에서 실패하는 편이 맞다. `nc`와 `reg_max`는 헤드 출력 폭만 바꾸므로 지원한다.

### 2.2 정확한 스키마로 가중치 로드

`weights::Loader`는 기대하는 모든 텐서를 기대 형상으로 한 번씩 가져가고, 그래프가 소비하지 않은 키를 보고한다. 파라미터가 없는 레이어(concat, upsample)의 가중치나 범위를 벗어난 레이어 인덱스는 해당 키를 지목하며 로드에서 실패한다. 접두사로 로드하고 남은 키를 무시하는 방식이었다면 추가 스테이지가 있는 체크포인트가 조용히 일부를 버린 채 로드됐을 것이다.

### 2.3 호스트에서 디코드

모델은 `(1, 4 + nc, N)`을 MLX로 반환한다. NMS, 점수 필터, 클래스 오프셋, 역 레터박스는 읽어 온 `f32` 버퍼 위에서 호스트가 수행한다. 후보 상한(3만 개, 300개 유지)을 거치면 배열이 작고, 그리디 루프를 Rust에 두면 클래스 인지 억제를 합성 박스로 직접 테스트할 수 있다. 업스트림의 순차 비교(유지한 박스 대 남은 목록)도 그대로 유지했다.

### 2.4 threshold 기본값을 패밀리로 이동

CLI에서 `--threshold`는 이미 선택이었지만 `load_detector`는 해석된 값을 요구했다. `Option<f32>`를 로더까지 옮기면서 패밀리 기본값이 패밀리와 함께 있게 됐다. RT-DETRv2 0.3, PP-DocLayoutV3 0.3, RF-DETR 0.5, YOLO11 0.05다. 명시한 값은 `detector_info`보다 먼저 검증하므로 잘못된 `--threshold`는 여전히 모델 파일을 열기 전에 실패한다.

## 3. 검증

- `yolo11` 단위 테스트 17개: 레터박스 기하와 중앙 정렬, 역매핑, 클래스 인지 NMS, IoU 임계, max-det/max-NMS 상한, 잘못된 예측, 설정과 라우팅.
- `cargo test --profile test-fast --lib vision::detection --features metal,accelerate`: 106 통과.
- `cargo test --profile test-fast --bin mlxcel commands::detect`: 8 통과(RT-DETRv2 순서와 마스크 불변).
- `axiom-of-choice/OmniParser-v2-icon-detect`를 합성 1000x600 스크린샷에 `--size 1280`으로 실행: 그린 사각형 다섯 개에 `icon` 박스 다섯 개, `--threshold 0.95`는 둘, `--iou 0`은 하나, 원본 해상도는 다섯 개.
- 로컬에서 `cargo clippy --lib --tests -- -D warnings`와 `cargo fmt --check` 통과. CI의 `cargo-clippy`와 `CPU-only link` 잡은 mlxcel-core CMake 빌드에서 `Could NOT find BLAS`로 실패했으며 이 변경과 무관하다. `mergeStateStatus`는 `UNSTABLE`이었고 저장소가 머지를 허용했다.

## 4. 미검증

mlx-vlm 원시 출력 패리티 픽스처와 종단 간 박스 패리티 비교는 만들지 못했다. 이 호스트에 Python mlx-vlm 설치가 없다. 실제 체크포인트에 대한 구조 검증과 출력 확인은 했지만 `mlx_vlm.models.yolo11`과 수치 비교는 하지 않았다.
