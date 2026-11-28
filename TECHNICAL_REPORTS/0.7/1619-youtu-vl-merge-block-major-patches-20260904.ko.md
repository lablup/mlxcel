# 기술 보고서: PR #1619 - fix(youtu_vl): emit patches in merge-block-major order

**작성일**: 2026-09-04
**작성자**: mlxcel maintainers
**리뷰어**: implementation review cycle
**상태**: 완료 (실제 체크포인트로 세 가지 픽스처 크기에서 검증함. 같은 경로에 남은 결함 하나는 #1618로 분리)
**언어**: Rust
**위험도**: Low (단일 계열 전처리 경로이고, 불일치를 저장소 안에서만으로 증명할 수 있었음)

---

## 요약

`YoutuVLProcessor::try_preprocess_with_spatial`은 패치 격자를 평범한 래스터 순서로 훑으면서 한 패치당 한 행을 내보냈고, 행 내부 특징은 `(c, dy, dx)` 순서였다. 그런데 이 행을 받는 쪽 두 곳은 모두 merge 블록 우선 행과 채널 마지막 `(dy, dx, c)` 특징을 기대한다. 비전 타워는 자기가 가정하는 위치와 묶음에 대응하지 않는 패치를 받아 온 셈이다. 224x224 단색 주황 정사각형을 "completely black"이라고 설명했다. PR #1619는 방출 루프를 `(block_y, block_x, inner_y, inner_x)`에 `(dy, dx, c)` 특징으로 바꿨고, 같은 픽스처가 이제 "a solid, uniform block of bright orange color"로 나온다.

이 결함은 외부 오라클 없이 증명할 수 있었다. 인코더가 받는 두 입력이 서로 어긋나 있었기 때문이다. 이 보고서에서 다른 작업으로 옮겨갈 만한 부분이 바로 그 지점이다.

---

## 1. 문제 정의

### 1.1 배경

Youtu-VL 프로세서는 `src/vision/processors/qwen2_vl.rs`의 형태를 따라 이식했다. 그 파일은 패치 내부 특징을 `(c, dy, dx)`로 내보내는데, Qwen2-VL에서는 그게 맞다. 업스트림 프로세서가 `(C, H, W)` 이미지 텐서 위에서 unfold를 하기 때문이다. 이식 과정에서 그 내부 순서를 그대로 가져오면서 바깥쪽 merge 블록 루프는 빠뜨렸고, 결국 평범한 래스터 행만 남았다. 루프 위에 달려 있던 낡은 주석이 흔적이었다. "match how upstream unfolds patches via `unfold`"라고 적혀 있었지만 이 체크포인트는 unfold를 하지 않는다.

### 1.2 무엇이 어긋났나

루프가 만들어 내는 결과에 두 소비자가 동시에 반대한다.

체크포인트 자신의 프로세서인 [image_processing_siglip2_fast.py](https://huggingface.co/tencent/Youtu-VL-4B-Instruct/blob/main/image_processing_siglip2_fast.py)의 `convert_image_to_patches`는 `(C, nh/m, m, ps, nw/m, m, ps)`로 reshape한 뒤 `(1, 4, 2, 5, 3, 6, 0)`으로 permute한다. merge 블록 우선 행에 채널 마지막 특징이다. `Siglip2VisionEmbeddings.patch_embedding`은 그 행에 곧바로 적용되는 `nn.Linear(num_channels * patch_size * patch_size, embed_dim)`이고, `src/loading/vlm_youtu_vl.rs`의 `remap_youtu_vl_weights`는 그 가중치의 이름만 바꿀 뿐 순서를 바꾸지 않는다. 다른 순서로 내보내면 학습된 투영에 뒤섞인 벡터를 먹이게 된다.

mlxcel 자신의 인코더도 같은 블록 우선 묶음을 전제한다. `YoutuVLVisionEncoder::rot_pos_emb`는 `reshape(&[h / merge, merge, w / merge, merge])` 다음에 `transpose_axes(&[0, 2, 1, 3])`으로 위치 id를 만들고, `forward_with_spatial`은 은닉 상태를 `[n_groups, spatial_merge_unit, dim]`으로 바꿔 그룹 축을 따라 gather한다. 연속한 `spatial_merge_size ** 2`개 행이 2x2 블록 하나여야 성립하는 계산이다. 래스터 행에서는 토큰마다 엉뚱한 공간 위치의 회전 위치를 달고 있었고, 병합기는 블록이 아닌 네 패치를 묶었다.

### 1.3 위험도 판단

증상이 크래시나 NaN이 아니라 유창하고 자신 있는 오답이다. 빌드도 테스트 스위트도 아무것도 걸러 내지 못했다. 같은 저장소의 Qwen2-VL 프로세서에는 올바른 묶음을 고정하는 테스트(`owned_and_mlx_paths_share_spatial_merge_grouped_patch_order`)가 있지만 Youtu-VL에는 대응물이 없었다. 이식 과정의 이탈이 리뷰를 통과한 이유가 여기에 있다.

---

## 2. 기술 검토

### 2.1 근본 원인

방출 루프는 평평한 패치 인덱스 하나를 돌면서 거기서 `(py, px)`를 되짚어 냈다.

```rust
for patch_idx in 0..total_patches_img {
    let py = patch_idx / w_patches as usize;
    let px = patch_idx % w_patches as usize;
    // ...
    for c in 0..in_channels {
        for dy in 0..self.patch_size {
            for dx in 0..self.patch_size {
```

바깥 행 순서와 안쪽 특징 순서가 둘 다 이 체크포인트에 맞지 않는다. 두 오류는 서로 독립이라, 하나만 고치면 여전히 투영에 뒤섞인 벡터가 들어간다.

### 2.2 오라클 없이 증명한 방법

인코더가 받는 두 입력이 저장소 안에서 서로 모순이었다. `rot_pos_emb`와 merge 단위 gather는 블록 우선인데, 이들에게 데이터를 넘기는 프로세서는 래스터였다. 둘 중 하나가 틀렸다는 사실을 확인하는 데 참조 구현도, mlx-vlm venv도, 로짓 추적도 필요 없었다. 어느 쪽이 틀렸는지는 업스트림의 `convert_image_to_patches`가 정해 줬다.

### 2.3 이웃 결함과 분리해 준 픽스처

이 계열에는 같은 경로에 이미 고쳐 둔 결함이 하나 더 있었다. #1600 / #1603에서 `argsort(window_index)`로 바로잡은 윈도 역치환이다. 주황이 검정으로 읽히는 증상을 patch 순서 탓으로 돌리려면 윈도 치환이 영향을 줄 수 없다고 증명 가능한 픽스처가 필요했다. 224x224에서 병합 격자는 7x7이고 8x8 윈도 하나에 들어간다. 그 크기에서 `get_window_index`는 항등이라 두 번 적용해도 아무 변화가 없다. 따라서 그 크기에서 나온 오답은 윈도 역치환 탓일 수 없고, 남는 후보가 패치 순서였다.

---

## 3. 기술적 결정

### 3.1 Qwen2-VL의 루프 모양만 따르고 내부 순서는 따르지 않는다

`src/vision/processors/qwen2_vl.rs`에는 네 겹 블록 루프가 이미 올바르게 들어 있어서 바깥 구조는 거기서 가져왔다. 반면 그 파일의 `(c, dy, dx)` 내부 순서는 일부러 가져오지 않았다. Qwen2-VL에서는 맞고 여기서는 틀리기 때문이다. 다음 이식에서 어느 방향으로든 같은 치환이 반복되지 않도록 그 차이를 주석에 적어 뒀다. `qwen2_vl.rs`는 건드리지 않았다.

### 3.2 나누어떨어짐을 믿지 말고 검사한다

`smart_resize`가 양쪽 변을 `patch_size * spatial_merge_size`의 배수로 올림하므로 `h_patches`와 `w_patches`는 언제나 merge 인자의 정확한 배수이고 블록 루프에 패딩이 필요 없다. 그 사실에 말없이 기대는 대신, 성립하지 않으면 새 오류 `YoutuVLPreprocessError::UnalignedPatchGrid`를 반환하도록 했다. 그러지 않으면 나중에 리사이즈 정책이 바뀔 때 끝자락 부분 블록이 아무 신호 없이 버려진다.

### 3.3 실패가 예정된 단언도 정답 그대로 적는다

`tests/youtu_vl_parity.rs`는 다중 윈도 픽스처 둘을 `#[ignore]` 내용 단언으로 담되, 지금 빌드가 통과할 만한 수준으로 낮추지 않고 완전한 정답에 맞춰 적고 #1618을 known failing으로 표시했다. 낮춰 적으면 현재의 오답을 기대값으로 기록하는 셈이고, 결함이 명세가 되어 버리는 바로 그 안티패턴이다.

---

## 4. 구현 내용

### 4.1 핵심 변경

```rust
let merge = self.spatial_merge_size;
// ... 나누어떨어짐 검사, 아니면 UnalignedPatchGrid 반환 ...
let mut row = 0usize;
for block_y in 0..hp / merge {
    for block_x in 0..wp / merge {
        for inner_y in 0..merge {
            for inner_x in 0..merge {
                let py = block_y * merge + inner_y;
                let px = block_x * merge + inner_x;
                // ... row로 row_start를 잡은 뒤:
                for dy in 0..self.patch_size {
                    for dx in 0..self.patch_size {
                        for c in 0..in_channels {
```

### 4.2 단위 테스트가 두 순서를 갈라낸다

`patches_are_emitted_merge_block_major_with_channel_last_features`는 4x4 패치 격자를 만든다. merge 블록으로는 2x2이고, 블록 우선과 래스터를 구분할 수 있는 최소 크기다. 2x2 패치에서는 두 순서가 일치해 버린다. 이 테스트는 행 순서가 `0,1,4,5, 2,3,6,7, 8,9,12,13, 10,11,14,15`임을 단언한다. `0..15`가 아니라는 점이 핵심이라, 래스터 방출이면 바로 깨진다.

내부 순서는 같은 검사 안에서 함께 고정한다. 세 채널에 서로 다른 값을 심어서, 빨강에 패치 id, 초록에 `dy`, 파랑에 `dx`를 넣었다. `(dy * patch_size + dx) * 3 + c` 위치를 읽어 셋을 모두 확인하므로 `(c, dy, dx)` 배치에서는 즉시 실패한다.

---

## 5. 검증

`models/mlx/youtu-vl-4b-instruct`, M1 Ultra, `metal,accelerate` 릴리스 빌드, greedy(`-t 0 -n 48`)로 측정했다.

| 픽스처 | 변경 전 | 변경 후 |
|---|---|---|
| `test_image.png`, 224x224 단색 주황 | "The image is completely black and contains no visible content, objects, text, or details." | "The image is a solid, uniform block of bright orange color." |
| `test_image_shapes.png`, 448x448 | "a single, solid black circle on a white background" | "a single, solid black circle on a plain white background" |
| `test_image_shapes_336.png`, 336x336 | "a single, solid black circle on a white background" | "a single white circle on a black background" |

단색 픽스처가 합격 기준이다. 도형 픽스처 둘은 바뀌었지만 여전히 틀렸고, 그게 예상한 결과다. 이 계열 비전 경로에 결함이 최소 하나 더 남아 있으며 여기에 합치지 않고 #1618로 발행했다.

게이트: `cargo test --workspace --profile test-fast --features metal,accelerate`가 10,502개 통과, 실패 0. `cargo clippy --lib --tests --features metal,accelerate -- -D warnings`, `cargo fmt --all -- --check`, `scripts/ci/check_cross_repo_refs.py` 모두 깨끗하다.

---

## 6. 변경 요약

### 통계

| 항목 | 값 |
|---|---|
| 변경 파일 | 3 |
| 추가 줄 | 364 |
| 삭제 줄 | 19 |

### 분류별 변경

- `src/vision/processors/youtu_vl.rs`: 방출 루프 재작성, 낡은 `unfold` 주석을 `convert_image_to_patches`를 지목하고 업스트림 파일을 링크하는 주석으로 교체, `UnalignedPatchGrid` 오류 변형 추가.
- `src/vision/processors/youtu_vl_tests.rs`: 체크포인트 없이 도는 테스트로 행 순서와 내부 특징 순서를 함께 고정.
- `tests/youtu_vl_parity.rs`: 신규. `tests/qwen2_5_vl_parity.rs` 형태이며, 통과하는 224 대조군과 #1618이 추적하는 다중 윈도 단언 둘을 담았다.

### 관련 이슈

Closes #1610. #1600 / #1603(같은 계열의 윈도 역치환으로, 이번 결함이 그것을 가리고 있었다)의 후속이다. Qwen2.5-VL의 결함 쌍 #1596 / #1601과 같은 구조다. 남은 다중 윈도 결함으로 #1618을 열었다. 프로세서의 패치 수 상한은 #1611이다.

---

## 7. 후속 작업

### 옮겨갈 만한 교훈

VLM 이식본이 유창하지만 틀린 설명을 내놓을 때, 외부 오라클을 찾기 전에 이식본 자신의 구성 요소끼리 이미 어긋나 있지는 않은지 본다. 이번에는 프로세서의 행 순서와 인코더의 `rot_pos_emb`를 같은 저장소에서 끌어낼 수 있었고 둘이 모순이었다. 비용 없이 결함 위치가 좁혀진다.

나머지 절반은 대조군 픽스처다. 결과가 바이트 단위로 같게 나오는 변경은 실패한 실험이 아니라 발견이며, 반대도 성립한다. 어떤 픽스처는 고치고 어떤 픽스처는 못 고치는 수정은 결함이 몇 개인지에 대한 증거다. 224 단일 윈도 사례를 다중 윈도 사례 옆에 나란히 둔 덕분에 "고쳤는데 안 됐다"로 끝나지 않고 남은 #1618이 드러났다.

### 미해결

#1618이 남은 결함을 추적한다. 거기 기록해 둔 증거는 이렇다. 정답이 나온 유일한 픽스처가 병합 격자를 어텐션 윈도 하나에 담는 유일한 경우이고, 틀린 둘은 모두 2x2 윈도에 걸치면서 마지막 행과 열이 부분 윈도로 남으며, 두 오답의 양상도 서로 다르다.
