# 기술 보고서: PR #1405 - LFM2-VL 이미지 분할

**작성일**: 2026-08-25
**작성자**: mlxcel contributors
**상태**: 문서화된 parity 후속 과제를 포함해 완료
**언어**: Rust, Markdown
**위험도**: Medium

---

## 요약

PR #1405는 이슈 #1352에서 요구한 LFM2-VL 고해상도 `do_image_splitting` 경로를 구현한다. 큰 이미지는 이제 체크포인트의 종횡비 타일 정책, row-major view, `<|img_row_r_col_c|>` 및 선택적 `<|img_thumbnail|>` prompt framing을 사용하며 작은 이미지는 기존 single-view 바이트 경로를 유지한다.

리뷰와 보안 hardening 과정에서 view별 patch budget의 출처를 바로잡고, 잘못되거나 무제한인 체크포인트 metadata를 거부하며, image embedding이 masked scatter에 도달하기 전에 layout과 placeholder cardinality를 검증하도록 했다. 집중 테스트, workspace clippy, formatting, release build, GitHub CI, 실제 체크포인트 inference는 통과했으며, 로컬 전체 workspace test gate는 `qwen3_omni_moe_parity`의 별도 ThinLTO linker 결함 때문에 실행 전에 차단된다.

---

## 1. 문제 정의

LFM2-VL 체크포인트는 `processor_config.json`에 이미지 분할 정책을 제공하지만 mlxcel은 이전에 이 정책을 무시하고 모든 이미지를 한 view로 smart-resize했다. 따라서 1920x1080 스크린샷이 reference preprocessing에서 사용하는 여러 512픽셀 view 대신 single-view token budget으로 압축됐다.

누락된 기능은 prompt 구성과 embedding projection에도 영향을 줬다. runtime에는 tile marker, thumbnail marker, 이미지별 view layout이 없었고 이미지마다 view가 하나라는 암묵적 가정이 있었다. Crop loop만 추가하면 prompt와 feature cardinality가 일치하지 않게 된다.

| 위험 | 영향 | 수정 전 가능성 |
|------|------|----------------|
| 큰 스크린샷의 고해상도 세부 정보 손실 | High | High |
| Tile prompt marker와 projected view 불일치 | High | end-to-end layout 계약이 없으면 High |
| 잘못된 체크포인트 metadata로 과도한 메모리 할당 | High | Medium |

---

## 2. 기술적 선택

### 2.1 이미지별 layout을 전체 VLM 경로로 전달

`Lfm2VlImageLayout`은 각 view의 patch grid와 논리적 tile row/column을 기록한다. Processor는 원본 이미지 순서대로 layout을 반환하고 vision tower는 같은 순서로 모든 평탄화된 view를 project하며 prompt expansion은 동일한 layout으로 marker와 정확한 수의 `<image>` placeholder를 방출한다.

이 방식은 preprocessing 이후 tensor shape에서 geometry를 다시 추론하지 않으며 multi-image prompt를 명확하게 유지한다. 또한 masked scatter 전에 잘못된 tile/view cardinality를 거부할 단일 runtime 경계를 제공한다.

### 2.2 Processor metadata를 신뢰할 수 없는 입력으로 처리

Loader는 `processor_config.json`을 읽고 중첩된 `image_processor`를 해제하며 호환되는 기존 필드에는 `config.json` fallback을 제공하지만, 잘못된 값을 조용히 허용하지 않는다. Tolerance와 downsample factor는 finite positive 값이어야 하고 token id는 runtime 범위 안이어야 하며 view별 patch 수는 제한되고 tile canvas 계산은 checked arithmetic을 사용하며 잘못된 JSON은 기본값으로 대체되지 않고 오류로 보고된다.

### 2.3 Vision table 길이 대신 processor patch budget 사용

공개 체크포인트는 학습된 single-view position table에 `vision_config.num_patches=256`, padded processor row에 `max_num_patches=1024`를 제공한다. 512픽셀 tile은 32x32 patch grid이므로 256을 기준으로 검증하면 지원되는 체크포인트를 잘못 거부한다. 최종 loader는 processor metadata에서 limit을 얻고 1024-patch view를 해당 budget으로 검증한다.

### 2.4 Thumbnail을 강제하지 않고 체크포인트 정책 보존

로컬 체크포인트는 `use_thumbnail=false`를 선언하며 mlxcel은 이를 존중한다. 체크포인트가 활성화하면 thumbnail-last layout을 지원하지만, 공개된 metadata를 다른 reference-library 기본값으로 덮어쓰지 않는다.

---

## 3. 구현 상세

| 영역 | 변경 |
|------|------|
| `src/loading/vlm_lfm2_vl.rs` | Parsing한 tiling metadata와 marker id를 processor 및 runtime 생성에 연결한다. |
| `src/loading/vlm_lfm2_vl_metadata.rs` | Processor policy와 tokenizer marker metadata를 parsing, validation, bounding한다. |
| `src/loading/vlm_lfm2_vl_tests.rs` | 기본값, 중첩 설정, 잘못된 metadata, marker resolution, patch budget 동작을 테스트한다. |
| `src/vision/processors/lfm2_vl.rs` | 종횡비 grid 선택, 단일 resize, row-major crop, 선택적 thumbnail append, 모든 view packing, layout 반환을 구현한다. |
| `src/multimodal/lfm2_vl_prompt.rs` | 논리적 image placeholder를 framing된 view별 token run으로 확장하고 image/layout cardinality를 검증한다. |
| `src/vision/lfm2_vl.rs` | 모든 view를 project하고 prompt 순서대로 feature row를 연결한다. |
| `src/multimodal/vlm_runtime.rs` | Layout과 resolved marker table을 prompt expansion에 전달한다. |
| `docs/supported-models.md` | 구현된 image splitting 기능을 기록한다. |

확장된 loader는 metadata와 test sibling 파일로 refactor하여 구현을 검토하기 쉽게 유지했고 새로 확장된 모든 source file을 저장소의 500-line 제한 아래로 유지했다.

---

## 4. 리뷰, 보안, 품질

### 4.1 리뷰 수정

초기 구현이 512픽셀 tile을 `vision_config.num_patches` 기준으로 검증하여 공개 체크포인트를 load할 수 없게 만드는 문제가 리뷰에서 발견됐다. 수정 후 `max_num_patches`를 processor metadata에서 사용하며 두 필드의 의미를 구분하는 regression test를 추가했다.

### 4.2 보안 hardening

| 발견 사항 | 심각도 | 해결 |
|-----------|--------|------|
| 잘못된 processor 또는 tokenizer sidecar가 조용히 fallback됨 | Medium | JSON parse 오류가 load를 실패시킨다. |
| Non-finite tolerance 또는 non-positive downsample 값이 bound를 훼손할 수 있음 | Medium | 산술 전에 값을 명시적으로 검증한다. |
| Tile dimension과 patch budget이 overflow되거나 과도한 canvas를 할당할 수 있음 | Medium | Checked arithmetic과 명시적 upper bound로 안전하지 않은 metadata를 거부한다. |
| Prompt placeholder, layout, projected row가 달라질 수 있음 | Medium | Merge 전에 layout과 논리적 image cardinality를 검증한다. |

Hardening 이후 Critical 또는 High 심각도의 보안 및 성능 발견 사항은 남지 않았다.

### 4.3 테스트 커버리지

Loader metadata, image processing, prompt expansion에 집중 테스트 19개를 추가했다. Reference 기본값, 중첩 metadata, 잘못된 token 및 숫자 필드, tile ratio 열거, area tie-break, row-major 색상 순서, 선택적 thumbnail 위치, multi-image expansion, 잘못된 layout, byte-identical small-image preprocessing을 검증한다.

---

## 5. 실제 체크포인트 결과

로컬 `models/lfm2-vl-450m-4bit` 체크포인트에서 `tile_size=512`, `min_tiles=2`, `max_tiles=10`, `max_pixels_tolerance=2.0`, `use_thumbnail=false`, `max_num_patches=1024`를 load했다.

| 입력 | Rust layout | Reference layout | Prompt 결과 |
|------|-------------|------------------|-------------|
| 1920x1080 스크린샷 | 4x2 tile, 32x32 patch view 8개 | 4x2 tile, 32x32 patch view 8개 | Tile당 image token 256개, 총 2048개, marker 397-400 및 407-410, thumbnail 없음 |
| 640x480 이미지 | 26x36 patch view 1개 | 26x36 patch view 1개 | Image token 234개, 변경되지 않은 single-view 경로 |

Release binary는 두 입력 모두에서 finite 64-token Metal 출력을 생성했다. 큰 이미지의 생성 텍스트는 Python reference와 token-exact하지 않았고 변경되지 않은 작은 경로도 뒤쪽 token 하나가 달랐으므로, 현재 정확한 tile geometry와 prompt framing 밖에 기존 runtime 또는 preprocessing 수치 차이가 남아 있음을 보여준다. 이 보고서는 생성 token parity를 이 PR에서 확립한 structural 및 prompt-token parity와 명확히 구분한다.

---

## 6. 검증 요약

| 검증 | 결과 |
|------|------|
| `cargo test --profile test-fast lfm2_vl --lib` | 통과, 28개 테스트 |
| `cargo clippy --workspace --all-targets -- -D warnings` | 통과 |
| `cargo fmt --all -- --check` | 통과 |
| `git diff --check` | 통과 |
| `cargo build --release --features metal,accelerate --bin mlxcel` | 통과 |
| GitHub CI | 모든 필수 check 통과 |
| 전체 workspace test | `qwen3_omni_moe_parity` link 중 별도 ThinLTO missing-symbol 실패로 실행 전에 차단됨; `CARGO_BUILD_JOBS=1`에서도 재현됨 |

---

## 7. 변경 요약

| 항목 | 값 |
|------|----|
| 이 보고서 전 변경 파일 | 8 |
| 추가 라인 | 1,307 |
| 삭제 라인 | 145 |
| 추가된 집중 테스트 | 19 |

| Commit | 목적 |
|--------|------|
| `c79d86255` | LFM2-VL image splitting 추가. |
| `c1db2e3e4` | Split tolerance 검증. |
| `11b40c73a` | 올바른 processor patch budget load. |
| `426bd1b22` | Metadata 및 layout validation hardening. |
| `5f3d2f1de` | Tiling test coverage 확장. |
| `35e250fae` | Loader metadata helper와 test 분리. |

---

## 8. 후속 조치

- 제어된 pixel array 및 embedding 비교로 기존 LFM2-VL numerical parity 차이를 조사하고, 정확한 geometry와 prompt token sequence를 생성 token 동일성의 증거로 취급하지 않는다.
- macOS ThinLTO integration-test linker 결함을 수정한 뒤 전체 workspace test gate를 다시 실행한다.

## 참고

- Issue #1352: LFM2-VL image splitting 및 marker framing.
- PR #1405: LFM2-VL image splitting 구현.
- `docs/supported-models.md`: 지원 모델 동작.
