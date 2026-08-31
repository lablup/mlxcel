# 기술 보고서: PR #1535 - Inkling HMLP 이미지 지원

**작성일**: 2026-08-31
**작성자**: mlxcel maintainers
**상태**: 결정론적 레퍼런스 검증 완료. 실제 체크포인트 검증은 보류
**언어**: Rust, Markdown
**위험 수준**: 높음

---

## 요약

PR #1535는 이슈 #1318에서 추가한 텍스트 백본 위에 Inkling의 네이티브 이미지 입력을 구현한다. 레퍼런스 hierarchical MLP 비전 타워, 정확한 40x40 타일링과 정규화, 동적 placeholder 확장, 순서가 보장된 feature scatter, VLM 감지와 로딩, CLI/서버 공통 런타임 통합을 추가했다. `vision_config`와 `model.visual.*` 텐서를 모두 갖지 않은 체크포인트는 계속 텍스트 전용 Inkling 경로로 로드된다.

Inkling 이미지 스택은 transformer 기반 비전 타워와 다르다. 각 타일을 두 temporal slot으로 복제하고, 시간과 공간 축을 점진적으로 채널에 접은 뒤 텍스트 폭 soft token 하나로 직접 투영한다. 여러 오류가 그럴듯한 shape를 유지하면서 값이나 미디어 순서만 바꿀 수 있으므로 선택 scale row, fold permutation, 의도적인 trailing tile column, 이미지별 placeholder cardinality, 원본 체크포인트 key mapping을 결정론적 테스트로 고정했다.

## 1. 문제 정의

mlxcel에는 Inkling 디코더가 있었지만 HMLP encoder와 VLM runtime은 없었다. 기존 이미지 processor는 정확한 너비의 이미지에도 완전 padding 열을 하나 추가하는 Inkling의 특이한 `columns = width / 40 + 1` 규칙을 재현하지 않았다. 고정 토큰 수를 가정하는 일반 prompt 확장으로는 한 요청 안의 이미지마다 다른 tile 수를 표현할 수도 없었다.

로더는 기존 Inkling 텍스트 sanitizer를 재사용하면서 `model.visual.*` 텐서를 보존하고 이름을 바꾸어야 했으며, dense 및 affine-quantized visual projection을 모두 지원해야 했다. Vision config만 남은 text-only export를 VLM으로 잘못 분류해서도 안 됐다. Generation 경계에서는 이미 정규화된 text embedding과 final-normalized visual feature를 합친 뒤 text embedding norm을 다시 적용하지 않아야 했다.

## 2. 변경 요약

| 영역 | 결과 |
| --- | --- |
| HMLP graph | 소인수 scale 계획, 최소 비용 injective row 선택, time-space-to-depth fold, bias 없는 projection, 중간 RMSNorm 및 exact GELU, final RMSNorm 추가 |
| 전처리 | RGB 변환, 선택적 upstream 호환 Lanczos resize, row-major 40x40 tiling, rescale 전 `-1` padding, CLIP 정규화, temporal 복제, 이미지별 tile 수 추가 |
| 프롬프트 | 이미지당 marker 하나를 실제 tile token 수로 확장하고 plain prompt에 삽입하는 fail-closed 경로 추가 |
| 병합 | 정규화된 Inkling text embedding에 `merge_llava`로 순서대로 scatter하고 두 번째 input norm을 건너뛰는 prepared-embedding decoder 진입점 추가 |
| 로딩 | visual key 정규화, dense 및 affine projection 검증, processor config 파싱, text width 호환성 검사 추가 |
| 감지 | indexed 및 unindexed safetensors visual-weight 감지를 추가하면서 text-only model 경로 유지 |
| 런타임 | `LoadedModel` dispatch와 CLI `--image` 및 OpenAI 호환 `image_url` 요청이 공유하는 이미지 경로에 `InklingVLM` 등록 |
| 문서 | 지원 모델 문서에 이미지 기능과 검증 한계 반영 |

## 3. 기술적 선택과 이유

### 3.1 레퍼런스 graph를 정확히 유지하기

공개 mlx-vlm Inkling `vision.py` graph를 따랐다. 4-layer 체크포인트는 `[0, 1, 2, 4, 5]` grid row를 선택해 입력 폭 75, 512, 5120, 9600의 projection을 만든다. Fold는 flatten 전에 `t, row, column, channel` 순서가 되도록 reshape와 transpose를 수행한다. 중간 layer는 RMSNorm 뒤 exact erf GELU를 적용하고 마지막 projection에는 activation을 적용하지 않은 뒤 tower final RMSNorm을 수행한다.

1-layer 0.6B 설정은 첫 grid와 마지막 grid를 직접 선택해 전체 `[2, 40, 40, 3]` 타일을 9600개 채널로 접는다. 두 계획 모두 hard-code하지 않고 config에서 계산하지만, 로더는 공개 체크포인트 계약인 `T=2, H=W=40, C=3` 이외의 geometry를 거부한다.

### 3.2 토큰 수를 세기 전에 padding과 resize 의미론을 재현하기

Processor는 channel-first 타일을 `-1`로 초기화하고 사용 가능한 uint8 pixel을 복사한 뒤 `1/255` rescale과 채널별 normalize를 수행하며, channel-last 결과를 시간축에 복제한다. 너비에는 ceil division이 아니라 floor division 뒤 1을 더한다. 이 차이는 40의 정확한 배수인 모든 이미지에서 보이며 prompt 길이를 직접 결정한다.

선택적 resize는 upstream의 long-edge fraction, downscale을 막는 cap 식, half-up dimension rounding, Lanczos filter를 따른다. 이미지마다 tile 수를 따로 반환해 첫 이미지의 tile 수를 모든 media block에 잘못 적용하지 않도록 했다.

### 3.3 정규화된 embedding과 원본 embedding 진입점 분리하기

Inkling은 visual scatter 전에 `embed_norm`을 적용한다. 그래서 VLM wrapper는 정규화된 text embedding을 요청하고 image marker 행을 final-normalized HMLP feature로 바꾼 뒤 prepared-embedding decoder 진입점을 호출한다. 일반 텍스트 호출은 기존 raw-embedding 경로를 유지한다. 이 인터페이스는 의존 오디오 구현의 확장 지점이기도 하며 겉으로 알아보기 어려운 이중 정규화 오류를 막는다.

### 3.4 체크포인트 구조와 media cardinality를 신뢰하지 않는 입력으로 취급하기

로더는 추론 전에 projection row와 packed width, scale과 bias shape, normalization width, text hidden-size 일치, 정확한 visual geometry, 양의 유한 epsilon, MLX `i32` shape 한계를 검증한다. 감지는 safetensors header만 읽고 unindexed header를 128 MiB로 제한하며 config와 visual weight를 모두 요구한다.

런타임은 확장 뒤 실제 image marker 수를 세고 HMLP feature 수와 정확히 일치해야 scatter한다. 빈 prompt, 크기 0 media block, 산술 overflow, 부분적으로 확장된 marker layout, 맞지 않는 embedding width는 text와 이미지 위치를 조용히 이동시키는 대신 오류를 반환한다.

## 4. 리뷰와 보강

정확성·보안·성능·finalizer 리뷰를 통해 머지 전에 다음을 보강했다.

- 복구 불가능한 MLX layer를 만들기 전에 dense 및 quantized projection shape를 검사했다.
- Dense projection bias와 잘못된 RMSNorm 텐서를 거부했다.
- Unindexed safetensors에 대해 크기가 제한된 header-only 감지 fallback을 추가했다.
- Tile allocation과 count 산술을 checked 연산으로 바꾸고 allocation 실패를 보고했다.
- 잘못된 resize·정규화·text-width 설정을 거부했다.
- 모델 소유 Inkling cache를 위해 prepared-embedding sequence-ID 및 last-logits 경로를 유지했다.
- Inkling 런타임이 동적 tile 수를 보고할 때 all-target 및 test build의 match가 완전하도록 명시적인 CLI preparation-summary arm을 추가했다.
- 불안전한 cache 계약으로 one-shot visual embedding을 재사용하지 않고 chunked prefill을 비활성화했으며 image feature cache를 범위 밖으로 유지했다.

리뷰된 변경에 해결되지 않은 CRITICAL 또는 HIGH 정확성·보안·성능 문제는 남지 않았다.

## 5. 검증

| 게이트 | 결과 |
| --- | --- |
| `cargo test --lib inkling --profile test-fast --features metal,accelerate -- --test-threads=1` | 통과, 40/40 |
| `cargo check --lib --features metal,accelerate` | 통과 |
| `cargo clippy -p mlxcel --lib --tests -- -D warnings` | 통과 |
| `cargo fmt --all -- --check` | 통과 |
| `git diff --check` | 통과 |

집중 테스트는 두 레퍼런스 scale plan, channel fold 순서, tower 출력 shape, 잘못된 geometry와 weight, 정확한 너비의 trailing padding, 이미지별 tile 수, 이미지당 marker 하나 확장, plain prompt 삽입, 이미 확장된 layout과 모호한 layout, indexed 및 unindexed 감지, 원본 visual key 정규화를 포함한다. 기존 Inkling text·cache·sanitizer·reasoning marker·chat template 테스트도 같은 40-test 선택에서 모두 통과했다.

## 6. 검증 한계와 후속 작업

공개 Inkling-Small affine MLX 체크포인트는 약 153.5 GB, native NVFP4 체크포인트는 약 170.7 GB다. 검증 호스트에서 어느 쪽도 사용할 수 없었다. 따라서 CLI 및 서버의 실제 이미지 답변 품질, 1-layer 0.6B 체크포인트와의 feature parity, peak memory, Apple GPU 처리량은 확인하지 않았고 이 보고서도 이를 주장하지 않는다.

이슈 #1323은 이 공통 visual shell 위에 네이티브 인접 프레임 video pair를 추가한다. 오디오, image feature caching, tensor-parallel vision execution, fused kernel은 별도 작업으로 남는다. 넓은 workspace test와 all-target clippy gate는 이 집중 issue worktree에서 중복 실행하지 않고 epic-level 최종 검증에 맡긴다.

## 참고

- 에픽 #1313, 이슈 #1327, 선행 이슈 #1318
- PR #1535
- 공개 mlx-vlm Inkling `vision.py`, `inkling.py`, processing 구현
- `docs/supported-models.md`
