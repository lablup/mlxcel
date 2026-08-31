# 기술 보고서: PR #1546 - Inkling 인접 프레임 비디오 지원

**작성일**: 2026-08-31
**작성자**: mlxcel maintainers
**상태**: 결정론적 레퍼런스 검증 완료. 실제 체크포인트 검증은 보류
**언어**: Rust, Markdown
**위험 수준**: 높음

---

## 요약

PR #1546은 PR #1535에서 구현한 HMLP 이미지 shell 위에 Inkling 네이티브 비디오 입력을 추가한다. 기존 기본값인 2 fps로 비디오 프레임을 디코딩하고 요청 전체에서 최대 16개의 균등한 인접 pair를 선택하며, 각 첫 번째 프레임을 일반 이미지 entity 하나로 표현한 뒤 해당 비디오 타일 suffix의 temporal slot 1만 두 번째 프레임으로 교체한다. CLI `--video`와 OpenAI 호환 서버 `video_url` 요청은 동일한 prompt, tiling, splice, embedding 경로를 공유한다.

구현은 비디오 프레임을 서로 독립적인 still image로 다루지 않고 공개 mlx-vlm Inkling graph를 따른다. Timestamp prompt part는 시간 순서의 grounding을 유지하고, companion still image는 비디오 entity 앞에 남으며, shape·cardinality·FPS·placeholder를 fail-closed 방식으로 검사해 media row가 잘못된 text 위치에 scatter되는 것을 막는다.

## 1. 문제 정의

Inkling 이미지 patch의 shape는 `[N, 2, 40, 40, 3]`이다. PR #1535는 still image를 두 temporal slot에 복제했으며 이는 이미지에는 맞지만 움직임을 표현하지 못한다. 샘플링한 모든 프레임을 독립 이미지로 전달하는 일반 video fallback은 동일한 두 프레임 근거에 visual token을 두 배 사용하고 체크포인트가 학습한 HMLP temporal axis도 활용하지 못한다.

따라서 네이티브 비디오에는 동기화된 두 표현이 필요하다. 선택한 각 pair의 첫 프레임이 image entity, tile 수, prompt placeholder run, slot 0 pixel을 결정한다. 샘플링 결과에서 바로 다음 프레임은 동일한 tiler를 사용하고 해당 비디오 타일의 slot 1만 교체해야 한다. 인접하지 않은 pair, tile 순서 변화, 잘못된 suffix 경계는 tensor shape가 정상인 채로 double exposure를 만들거나 companion still image를 변경하거나 feature를 잘못된 prompt row에 조용히 scatter할 수 있다.

## 2. 변경 요약

| 영역 | 결과 |
| --- | --- |
| Pair 선택 | 홀수 프레임 padding, 최소 두 pair, 요청 전체 최대 16 pair, 반올림한 균등 anchor, 엄격한 `anchor + 1` companion 추가 |
| Prompt | 시간 순서 안내문, pair마다 `frame at t=<seconds>s:` text part와 image marker 하나, 그 뒤 원래 사용자 질문을 배치 |
| 이미지 처리 | 첫 프레임과 두 번째 프레임에 정확한 Inkling 40x40 tiler를 각각 재사용하고 pair별 tile layout 일치 확인 |
| Temporal splice | 앞선 still-image row의 두 temporal plane은 유지하면서 마지막 video-tile row만 `[first_slot0, second_slot0]`으로 재구성 |
| 공통 runtime | 기존 HMLP tower와 normalized embedding scatter 전에 CLI와 서버가 함께 쓰는 host 준비 함수 추가 |
| CLI | Companion `--image`와 함께 쓰는 경우를 포함해 Inkling `--video`를 네이티브 pair 경로로 연결 |
| 서버 | Inkling `video_url` 활성화, fd 기반 ffmpeg 입력 유지, 설정된 제한으로 선택적 `image_url` companion 디코딩, 요청 내 FPS 일치 강제 |
| Capability 감지 | 합성 indexed-checkpoint 회귀 테스트와 함께 서버 video admission에 `InklingVLM` 추가 |
| 문서 | 비디오 동작, 인접 pair 의미론, 16-pair 제한, 실제 체크포인트 검증 한계 문서화 |

## 3. 기술적 선택과 이유

### 3.1 인접성과 레퍼런스 anchor 공식을 유지하기

홀수인 마지막 프레임을 복제한 뒤 pair 수는 `max(2, min(max_pairs, len / 2))`다. Anchor `i`는 `min(round(i * (len - 2) / max(n_pairs - 1, 1)), len - 2)`이며 두 번째 프레임은 항상 `anchor + 1`이다. 이 규칙은 전체 샘플 clip에 근거를 분산하면서 국소적인 움직임을 보존한다. 최소 두 pair 규칙은 한 프레임 입력에서 같은 anchor를 반복할 수 있게 하며 별도의 임의적인 short-clip 규칙을 만들지 않고 레퍼런스를 그대로 따른다.

Pair 예산은 요청 전체에 적용한다. 여러 CLI 경로나 서버 video part는 선언 순서로 합친 뒤 선택하므로 한 요청은 첫 프레임 image entity 16개를 넘을 수 없다. 서로 다른 FPS의 서버 part는 거부한다. 여러 시간 기준이 있으면 하나의 anchor index를 올바른 timestamp로 변환할 수 없기 때문이다.

### 3.2 각 pair를 prompt image entity 하나로 취급하기

Prompt에는 companion still-image marker를 먼저 두고 정확한 안내문 `Here is a video as a sequence of frames in chronological order.`를 추가한 뒤 pair마다 timestamp text와 image marker 하나를 배치한다. 이미 render된 사용자 질문은 그 뒤에 유지된다. 기존 template marker는 하나도 없거나 companion still image마다 정확히 하나일 때만 허용한다. 그 외의 수는 승인된 media에서 나오지 않은 예약 image token을 소비하지 않고 실패한다.

첫 프레임을 전처리한 뒤 각 marker는 해당 entity의 실제 tile 수만큼 확장된다. `merge_llava`가 normalized text embedding을 교체하기 전에 확장된 marker 총수가 HMLP feature 수와 계속 일치해야 한다. 이로써 prompt와 feature cardinality가 고정 token 수 가정이 아니라 전처리 결과에 결합된다.

### 3.3 Tower 실행 전에 temporal suffix를 splice하기

Companion still과 첫 프레임을 그 순서대로 함께 전처리한다. 두 번째 프레임은 별도로 전처리하고 그 slot-0 plane을 `[M, 40, 40, 3]`으로 만든다. 모델 경계는 전체 `[N, 2, 40, 40, 3]`과 교체 tensor shape를 검증하고 `pixel_values[..N-M]`을 그대로 유지하며 마지막 `M` row의 slot 0을 취해 교체 tensor를 slot 1로 연결한 뒤 이미지 요청과 같은 HMLP tower를 실행한다.

이 경계는 공개 mlx-vlm `get_input_embeddings` 구현과 같으며 비디오 전용 vision graph를 하나 더 만들지 않는다. 또한 PR #1535의 prepared-embedding sequence-ID 및 last-logit lifecycle을 바꾸지 않으므로 이후 audio와 MTP 작업이 visual prefill을 우회하지 않고 공개 `InklingVlModel.text` wrapper와 결합할 수 있다.

### 3.4 서버 media 보안 속성을 유지하기

서버 비디오 디코딩은 계속 `ResolvedVideo.source`를 사용한다. Unix에서는 allowlist와 canonical path 검증 뒤 얻은 fd 기반 handle이므로 admission과 decode 사이에 경로가 바뀌어도 ffmpeg가 바뀐 파일을 다시 열 수 없다. Companion image byte는 기존의 제한 인지 decoder를 사용한다. 빈 frame set, 유한하지 않거나 양수가 아닌 FPS, 여러 비디오 사이의 FPS 불일치, 인접 프레임 tile layout 불일치, 잘못된 tensor rank나 dimension, `1..=N` 밖의 suffix 크기, 산술 overflow, 맞지 않는 image placeholder는 vision 실행 전에 오류를 반환한다.

## 4. 리뷰와 보강

정확성·보안·성능·finalizer 리뷰를 통해 머지 전에 다음을 보강했다.

- 10개 프레임에서 정확한 anchor `[0, 3, 5, 8]`, 홀수 마지막 프레임 복제, 최소 두 pair, 엄격한 인접성을 결정론적 테스트로 고정했다.
- 앞의 still tile 세 개가 값 0인 두 plane을 유지하고 마지막 video row 두 개의 slot 1만 교체 값을 받는지 검사했다.
- MLX splice 전에 두 번째 프레임 tile 수와 해당 첫 프레임 suffix를 비교했다.
- Timestamp prompt를 재구성하기 전에 신뢰할 수 없거나 모호한 image-marker cardinality를 거부했다.
- Path 기반 지름길을 만들지 않고 서버의 fd 기반 decode 경계와 설정된 이미지 제한을 유지했다.
- 일반 2 fps 기본값과 CLI override를 유지하면서 여러 서버 video part에는 하나의 FPS 시간 기준을 요구했다.
- 첫 프레임 visual entity를 요청 전체 16개로 제한했고 두 번째 프레임은 marker run을 추가하지 않고 같은 visual token row를 재사용했다.
- 공개 Inkling VLM 체크포인트를 text-only export와 구분하는 indexed visual-weight 증거를 사용해 명시적인 startup capability 테스트를 추가했다.
- Audio, MTP, fused kernel, padded batching, image-feature caching, 넓은 epic-level 검증을 이 이슈의 범위 밖으로 유지했다.

리뷰된 변경에 해결되지 않은 CRITICAL 또는 HIGH 정확성·보안·성능 문제는 남지 않았다.

## 5. 검증

| 게이트 | 결과 |
| --- | --- |
| `cargo test -p mlxcel inkling --lib` | 통과, 45/45 |
| `cargo test -p mlxcel pair_adjacent_frames --lib` | 통과, 1/1 |
| `cargo test -p mlxcel timestamped_pair_messages --lib` | 통과, 1/1 |
| `cargo test -p mlxcel slot1_overwrite_touches_only_the_tail --lib` | 통과, 1/1 |
| `cargo clippy -p mlxcel --lib --tests -- -D warnings` | 통과 |
| `cargo check -p mlxcel --lib --features metal,accelerate` | 통과 |
| `cargo check -p mlxcel --bin mlxcel --bin mlxcel-server` | 통과 |
| `cargo fmt --all -- --check` | 통과 |
| `git diff --check` | 통과 |

집중 테스트는 pair 간격과 인접성, 단일·홀수·짧은 프레임 동작, 시간 순서 prompt part layout, still/video 혼합 marker 순서, 예약 token 거부, suffix 전용 slot 교체, 잘못된 교체 cardinality, Inkling VLM media admission, 기존 이미지·HMLP·텍스트·cache·sanitizer·reasoning marker·chat template 회귀를 포함한다.

이 호스트에는 `IREE_DIST`가 없어 OpenXLA feature gate를 로컬에서 재현할 수 없었으며 PR CI가 해당 compile 검사를 담당한다. 넓은 workspace test와 all-target clippy는 epic-level 최종 검증 범위로 남긴다.

## 6. 검증 한계와 후속 작업

공개 Inkling-Small affine MLX 체크포인트는 약 153.5 GB, native NVFP4 체크포인트는 약 170.7 GB다. 검증 호스트에는 두 체크포인트와 의도한 실제 bouncing-ball fixture가 없었다. 따라서 CLI와 서버의 실제 비디오 답변 품질, 움직임 방향 정확도, peak memory, Apple GPU 처리량은 확인하지 않았으며 이 보고서도 이를 주장하지 않는다.

Audio와 MTP는 독립적인 epic 작업이다. 비디오 method는 추가형 API이며 temporal splice 뒤 같은 normalized image prefill에 위임한다. 이 작업들이 사용하는 공개 text wrapper와 prepared-embedding 계약은 변경하지 않는다. 이후 실제 체크포인트 검증에서는 2 fps의 합성 인접 움직임 clip을 공개 mlx-vlm 결과와 비교하고 답변 방향과 확장된 visual-token cardinality를 모두 확인해야 한다.

## 참고

- 에픽 #1313, 이슈 #1323, 선행 이슈 #1327
- PR #1546 및 선행 PR #1535
- 공개 mlx-vlm Inkling `inkling.py`, `vision.py`, 이미지 processing, 일반 video helper 구현
- `docs/supported-models.md`
