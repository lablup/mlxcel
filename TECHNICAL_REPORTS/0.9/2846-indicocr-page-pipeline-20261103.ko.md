# PR #2846: IndicOCR 레이아웃과 Qwen3.5 OCR 파이프라인

**Date**: 2026-11-03
**Status**: Completed (머지됨, #2414 닫음)

## 요약

PR #2846은 2단계 IndicOCR 페이지 파이프라인을 추가한다. `mlxcel generate -m <indic-ocr-root> --image page.png`는 IndicDocLayout으로 영역과 읽기 순서를 찾고, IndicBlockOCR(Qwen3.5-0.8B)로 각 영역을 클래스별 지시문에 따라 전사한 뒤 읽기 순서 Markdown을 출력한다. OCR 스테이지는 밀집 Qwen3.5 VLM으로 직접 로드되므로 `-m weights/ocr`과 프롬프트로 단일 크롭도 전사할 수 있다.

명령 자체 외에 세 가지가 필요했다. PP-DocLayoutV3 예측기에 `MlxVlmDetect` 디코드 모드를 추가했다. IndicDocLayout은 Hugging Face 후처리가 아니라 mlx-vlm의 `detect()` 의미로 학습됐다. 순수 라우팅, 정리, 중첩 수식, 크롭, 재구성 규칙은 `src/vision/indic_ocr/`로 분리했다. 실제 체크포인트에서 잠복 설정 버그 두 개가 드러났다. 레이아웃 설정의 중복 serde 별칭과, IndicOCR의 262146과 맞지 않는 하드코딩된 Qwen3.5 EOS 쌍이다. 27개 파일, +2084/-34이며 그중 약 900줄이 테스트와 문서다.

## 1. 변경 요약

| 영역 | 변경 |
|---|---|
| `src/vision/indic_ocr/` (신규) | 두 레이블 체계 라우팅, `clean_layout`, 중첩 수식 해소, 면적 클램프 Lanczos 크롭, 원문 그대로의 프롬프트, Markdown 하이픈 복원과 수식 보정 |
| `pp_doclayout_v3` | `PpDocLayoutDecode::{HuggingFace, MlxVlmDetect}`, `decode_mlx_vlm_detections`, 정사각 입력 완화, `out_indices`/`return_idx`와 `feat_strides`/`feature_strides`의 선택 필드 해소 |
| `src/models/detection.rs` | `indic_ocr` OCR 스테이지 설정을 `Qwen35VLM`으로, 2단계 루트는 페이지 파이프라인으로 안내 |
| `src/models/qwen3_5.rs` | EOS id를 설정의 `eos_token_id`에서 읽고, 없을 때만 기존 쌍으로 폴백 |
| `src/commands/generate_indic_ocr.rs` (신규) | 스테이지 경로 해석, OCR 스테이지 채팅 템플릿을 통한 블록별 OCR, `--output-json`, `--table-format`, `--layout-detections` 재생 |
| `src/commands/generate.rs` | 모델 로드 전에 2단계 루트를 라우팅(프롬프트 없는 실행 포함), `run_generation_mode`에 명시적 토큰 예산 전달 |
| 문서 | `docs/supported-models.md` IndicOCR 절 |

## 2. 기술적 선택

### 2.1 모델 로드 전에 루트를 라우팅

2단계 IndicOCR 루트는 로드 가능한 VLM이 아니다. `stages`는 있지만 최상위 `text_config`가 없다. Qwen3.5로 로드하면 토크나이저 단계에서 "no tokenizer"라는 혼란스러운 오류로 실패한다. 따라서 `run_generate`는 `-m` 해석 직후 `model_type: indic_ocr`과 `stages`를 확인해 페이지 드라이버로 보낸다. 같은 검사를 프롬프트 없는 분기에도 넣었다. 인수 명령에 `-p`가 없어 그대로 두면 대화형 REPL로 빠지기 때문이다. `arch` 같은 다른 호출자를 위해 `get_model_type`은 페이지 파이프라인을 안내하는 명명된 오류를 반환한다.

### 2.2 레이아웃 디코드 모드를 예측기에 둔다

`PpDocLayoutDecode`는 CLI 플래그가 아니라 예측기 수준의 선택이다. `mlxcel detect`는 Hugging Face 후처리를 유지하고, IndicOCR은 `MlxVlmDetect`로 예측기를 만든다. 이 모드는 전처리(프로세서의 800 대신 1024 정사각), 필터링(플랫 top-k 대신 쿼리별 argmax), 순서 디코드(전체 쿼리 대신 유지된 부분행렬)를 바꾼다. 하나의 enum 뒤에 두면 세 차이가 함께 있고 CLI 계약은 그대로다.

### 2.3 부분행렬 순서 디코드는 의도된 선택

Hugging Face 후처리는 전체 쿼리로 GlobalPointer 순서를 디코드한 뒤 필터링한다. mlx-vlm의 `detect()`는 먼저 필터링하고 유지된 부분행렬을 디코드한다. 둘은 달라질 수 있고 IndicOCR 파인튜닝은 두 번째로 학습됐다. 새 단위 테스트는 전체 행렬을 쓰면 살아남은 두 쿼리의 순서가 뒤집히는 필터링 사례를 구성해 이 동작을 고정한다.

### 2.4 모델 I/O와 순수 로직 분리

`src/vision/indic_ocr/`는 무엇을 전사하고 텍스트를 어떻게 조립하는지 결정하는 모든 규칙을 담는다. 레이블 라우팅, 포함 관계 중복 제거, 여백 요소 처리, 중첩 수식, 크롭 면적 클램프, 프롬프트, 하이픈 복원, 수식 보정이다. 그래서 19개 테스트가 가중치 없이 돈다. 명령 파일은 스테이지 경로, 디텍터, VLM, 생성, JSON 출력을 맡는다.

### 2.5 두 번 나타나는 설정 필드

IndicDocLayout의 `config.json`에는 `out_indices`와 `return_idx`, `feat_strides`와 `feature_strides`가 모두 있다. serde `alias`는 두 번째 철자를 같은 필드로 취급해 문서를 중복으로 거부한다. 이제 별도 선택 필드로 두고 명시적으로 해소하며(`out_indices`, `feat_strides` 우선), 그래프는 해소된 값을 읽는다.

### 2.6 설정에서 읽는 EOS id

`Qwen35Model::eos_token_ids`는 `[248046, 248044]`를 하드코딩했다. IndicOCR OCR 스테이지는 262146을 선언하므로 생성이 멈추지 않고 토큰 상한까지 답과 `<|im_end|>`를 반복했다. 이제 `eos_token_id`(숫자 또는 배열)를 읽고, 없거나 쓸 수 없을 때만 기존 쌍으로 폴백한다. 다른 Qwen3.5 체크포인트는 같은 쌍을 선언하므로 동작이 바뀌지 않는다.

## 3. 검증

- `indic_ocr` 단위 테스트 19개: 두 체계 레이블 라우팅, 프롬프트 선택, 전사 제외, `clean_layout`, 헤더/푸터 규칙, 모드별 중첩 수식, densify, 크롭 클램프와 패딩, `area_clamp`, Markdown 재구성.
- `decode_mlx_vlm_detections` 테스트 4개: 쿼리별 argmax, 엄격한 신뢰도 필터, 유지 부분행렬 순서, 잘못된 입력.
- 검출 라우팅 87개, Qwen3.5 33개, PP-DocLayoutV3 35개, 명령 테스트 3개 통과.
- `HashNuke/indic-ocr-mlx weights/layout` 실제 게이트 테스트: 파인튜닝 로드, `MlxVlmDetect` 디코드, 조밀한 읽기 순서와 페이지 내 박스 확인.
- 렌더링한 힌디어 페이지 종단 간 실행: 6개 블록이 읽기 순서대로, 헤더는 빈 텍스트로 유지, 문단과 수식은 정확히 전사, JSON은 0 기반 순서와 픽셀 박스로 기록.
- `mlxcel detect --format json` 결과를 `--layout-detections`로 재생하면 같은 정리 후 같은 6개 블록과 Markdown이 나온다. `--table-format csv`는 모델 로드 전에 실패한다.
- 단일 크롭 `-m weights/ocr`이 페이지를 정확히 전사한다(그리디, 56 토큰).
- 로컬에서 `cargo clippy --lib --tests -- -D warnings`와 `cargo fmt --check` 통과. CI의 `cargo-clippy`와 `CPU-only link` 잡은 mlxcel-core CMake 빌드에서 `Could NOT find BLAS`로 실패했으며 이 변경과 무관하다.

## 4. 미검증

커밋된 mlx-vlm 레이아웃 패리티 픽스처와 mlx-vlm `parse()` 대비 첫 세 블록 텍스트 비교는 만들지 못했다. 이 호스트에 Python mlx-vlm 설치가 없다. 실제 레이아웃/OCR 체크포인트와 렌더링한 페이지로 종단 간 검증은 했지만 mlx-vlm과 수치 비교는 하지 않았다. 스톡 PP-DocLayoutV3 레이블 표는 단위 테스트로만 덮었고 실제 스톡 검출 파일 재생으로는 확인하지 않았다.
