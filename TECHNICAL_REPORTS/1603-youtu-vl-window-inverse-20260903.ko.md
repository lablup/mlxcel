# 기술 보고서: PR #1603 - fix: build the Youtu-VL window inverse as `argsort`, not the permutation itself

**작성일**: 2026-09-03
**작성자**: mlxcel maintainers
**리뷰어**: implementation review cycle
**상태**: 완료 (체크포인트 없는 테스트와 도달 가능한 격자 전수 열거로 산술을 증명했고, 로컬 Youtu-VL 체크포인트로 종단 검증했다. 같은 계열의 두 번째 결함이 결과를 지배해서 출력 변화는 없었고, 그 결함은 #1610으로 발행했다)
**언어**: Rust, Markdown
**위험도**: Low (Youtu-VL의 병합 비전 토큰 순서만 바뀐다. 윈도 하나보다 큰 격자에서만 해당하고, 이 헬퍼를 쓰는 다른 계열은 없다)

---

## 요약

Youtu-VL 비전 인코더의 `reverse_window_indices`는 `argsort(window_index)`를 돌려줘야 했다. 패치 머저를 지난 병합 비전 토큰을 다시 래스터 순서로 되돌리는 것이 그 함수의 일이다. 그런데 실제로는 `window_index` 자신을 돌려줬다. `(값, 인덱스)` 쌍을 정렬한 뒤 `reverse[orig_idx] = rank`를 쓰는데, `0..N`의 순열에서는 값이 `v`인 원소가 항상 `v`번째로 정렬되므로 이 반복문은 `reverse[i] = window_index[i]`로 무너진다. 호출부 주석은 결과가 `argsort(window_index)`와 같다고 적어 뒀으니, 코드와 주석이 어긋난 셈이고 의도를 제대로 담고 있던 쪽은 주석이었다.

순열을 두 번 적용해서 원래 순서로 돌아오려면 그 순열이 involution이어야 한다. #1601이 Qwen2.5-VL에서 고친 것과 같은 결함이지만, 두 계열이 노출되는 방식은 꽤 다르고 앞으로 기억해 둘 만한 부분도 바로 그 차이다. Youtu-VL의 윈도는 병합 토큰 8x8이고, `build_processor`가 프로세서 상한을 `vision_config.num_patches`(4096)로 잡으므로 병합 격자는 1x1에서 32x32까지 나온다. 그 범위를 전부 열거해 보면 순열이 involution이 되는 경우는 이미지 전체가 윈도 하나에 들어갈 때뿐이다. 안전한 크기가 하나 더 있지는 않다. 9x9부터 32x32까지 모든 병합 격자가 영향을 받고, 1024x1024에서는 1024개 중 960개가 엉뚱한 자리에 놓인다.

이제 헬퍼는 `inverse[window_index[rank]] = rank`를 직접 만든다. 잘못된 격자가 들어와도 패닉하지 않도록 범위를 확인하며, #1601이 Qwen2.5-VL에 넣은 `invert_window_index`와 같은 모양이다.

앞으로 이 보고서를 볼 사람이 가장 먼저 읽어야 할 대목은 종단 검증 결과다. `tencent/Youtu-VL-4B-Instruct`에서 448x448과 336x336 둘 다 **이 변경 전후의 생성 텍스트가 바이트 단위로 같고**, 양쪽 다 틀렸다. 고친 것이 무의미해서가 아니다. 윈도 재정렬보다 앞 단계에서 더 큰 결함이 비전 경로를 모든 이미지 크기에서 망가뜨리고 있기 때문이다. 이번 작업 중에 그 원인을 규명했고 #1610으로 발행했다.

---

## 1. 문제 정의

### 1.1 배경

이슈 #1600은 #1596 / #1601 Qwen2.5-VL 조사에서 갈라져 나왔다. 같은 구성이 `src/vision/encoders/qwen2_5_vl.rs`에도 있었기 때문이다. Youtu-VL은 인코더도 체크포인트도 검증 경로도 따로 가진 별개 계열이고, 수정을 정당화하는 오배치 토큰 수가 계열마다 다른 윈도 폭과 프로세서 상한에 달려 있어서 그 PR에서 일부러 빼 뒀다. 이슈는 그 수치를 가정하지 말고 직접 재라고 요구했다.

### 1.2 결함

```rust
let mut indexed: Vec<(i32, usize)> = window_index.iter().enumerate().map(|(i, &v)| (v, i)).collect();
indexed.sort_by_key(|&(v, _)| v);
let mut reverse = vec![0i32; window_index.len()];
for (rank, &(_, orig_idx)) in indexed.iter().enumerate() {
    reverse[orig_idx] = rank as i32;
}
```

`get_window_index`는 `0..N`의 순열을 돌려준다. 정렬이 끝나면 `indexed[rank]`는 `window_index[i] == rank`인 `(rank, i)`이므로, 쓰기는 결국 `reverse[i] = window_index[i]`가 된다. 입력을 비싸게 복사하는 항등 함수인 셈이다.

인코더에 필요한 것은 역순열이다. `forward_with_spatial`이 `take(h_grouped, window_index, 0)`으로 은닉 상태를 모으므로, 모으고 난 뒤 `j`번 자리에는 래스터 인덱스가 `window_index[j]`인 병합 토큰이 들어 있다. 래스터 순서를 되돌리려면 `window_index[j] == i`인 `j`를 찾아 `out[i] = merged[j]`로 놓아야 하고, 그것이 바로 `argsort`다.

### 1.3 이 계열이 다르게 노출되는 이유

윈도 크기는 `window_size / (patch_size * spatial_merge_size)` 병합 토큰이고, 이 체크포인트에서는 `256 / (16 * 2)`이라 8이다. `build_processor`가 프로세서의 패치 상한으로 `vision_config.num_patches`(4096)를 넘기므로 `effective_max_pixels`는 1,048,576이 되고, 패치 격자는 64x64, 병합 격자는 32x32에서 멈춘다.

로더가 만들 수 있는 병합 격자를 전부 열거하면 이렇다.

| 이미지 | 리사이즈 | 패치 격자 | 병합 격자 | 윈도 | 오배치 병합 토큰 |
|---|---|---|---|---|---|
| 224x224 | 224x224 | 14x14 | 7x7 | 1x1 | 49개 중 0개 |
| 256x256 | 256x256 | 16x16 | 8x8 | 1x1 | 64개 중 0개 |
| 336x336 | 352x352 | 22x22 | 11x11 | 2x2 패딩 | 121개 중 99개 |
| 384x384 | 384x384 | 24x24 | 12x12 | 2x2 패딩 | 144개 중 120개 |
| 448x448 | 448x448 | 28x28 | 14x14 | 2x2 패딩 | 196개 중 160개 |
| 512x512 | 512x512 | 32x32 | 16x16 | 2x2 | 256개 중 192개 |
| 768x768 | 768x768 | 48x48 | 24x24 | 3x3 | 576개 중 528개 |
| 1024x1024 이상 | 1024x1024 | 64x64 | 32x32 | 4x4 | 1024개 중 960개 |

여기서 두 가지가 따라 나오는데, 둘 다 개별 수치보다 중요하다.

첫째, 순열이 involution이 되는 경우는 윈도가 하나일 때뿐이다. 변당 윈도 개수가 윈도 폭과 같아도 involution이 되지만, 폭이 8이면 그것은 64x64 병합 격자이고 패치 16384개가 필요해서 4096 상한을 넘는다. 도달할 수 없으니 **안전한 경우는 윈도 하나뿐이다.** Qwen2.5-VL은 윈도 폭이 4라서 흔한 16x16 병합 격자가 involution이 되고, 그래서 #1596이 기본 픽스처 크기인 448x448에서 보이지 않았다. 대비가 뚜렷하다.

둘째, 저장소의 유일한 범용 이미지 픽스처인 `tests/fixtures/test_image.png`는 224x224다. 병합 격자 7x7, 윈도 하나, 항등 순열이다. 그 픽스처를 쓰는 테스트는 무엇을 단언하든 이 결함으로 실패할 수 없었다.

---

## 2. 변경 요약

| 파일 | 변경 |
|---|---|
| `src/vision/encoders/youtu_vl_window.rs` | `reverse_window_indices`가 `inverse[window_index[rank]] = rank`를 직접 만들고 범위를 확인한다. 옛 구성이 왜 무너졌는지 rustdoc에 남겼다. 이름과 시그니처는 그대로다. |
| `src/vision/encoders/youtu_vl.rs` | 호출부 주석에서 옛 코드가 `argsort`와 같다는 문장을 뺐다. |
| `src/vision/encoders/youtu_vl_tests.rs` | 체크포인트 없는 테스트 4개. 수정 전 코드에서는 전부 실패한다. |
| `CHANGELOG.md` | Unreleased / Fixed 항목. 주장 범위를 해당 단계의 토큰 순서로 좁히고 #1610을 가리킨다. |

새 테스트는 기존 테스트가 쓰던 폭 4짜리 합성 설정 대신 체크포인트의 실제 파라미터로 `get_window_index`를 부른다. 그래서 고정한 수치가 실제 이미지가 맞닥뜨리는 값이다. 최소 비involution인 `[2, 0, 1]`과 그 옆의 involution, 256개 중 192개를 고정한 32x32 패치 격자, 결함을 가려 온 단일 윈도 항등 사례, 그리고 둘째 격자가 한 변에 패딩이 필요한 2장짜리 배치가 들어 있다. 마지막 것은 `get_window_index`가 붙이는 이미지별 오프셋 처리를 확인한다.

---

## 3. 기술적 선택과 그 이유

### 3.1 새로 짜지 않고 #1601의 구성을 그대로 가져왔다

`src/vision/encoders/qwen2_5_vl.rs`의 `invert_window_index`는 똑같은 문제로 이미 리뷰와 머지를 거쳤다. 같은 반복문, 같은 범위 확인, 같은 주석 모양을 쓰면 두 계열을 나란히 놓고 비교하기 쉽다. 이 저장소가 업스트림 디렉터리 구조를 따라가는 이유가 바로 그것이다.

### 3.2 바로 인덱싱하지 않고 범위를 확인했다

`get_window_index`는 `0..len`의 순열을 내보내므로 이 가드는 실제로는 걸리지 않는다. 그럼에도 넣은 이유는 잘못된 격자가 들어왔을 때 대안이 비전 순전파 한가운데의 패닉이기 때문이고, 이 헬퍼가 `pub(super)`라 호출부가 하나 더 생길 수 있기 때문이다. 비용은 토큰당 `try_from` 하나와 `get_mut` 하나인데, 어차피 벡터를 할당하는 경로다.

### 3.3 이름과 시그니처를 유지했다

호출부를 건드리지 않으므로 변경이 헬퍼 안에 머문다. 실수로 다른 것이 바뀔 여지가 없다.

### 3.4 눈에 보이는 출력 변화가 없는데도 머지했다

종단 실행에서는 차이가 없다(4장). 그래도 머지하는 근거는 이 결함이 통계가 아니라 산술이라는 데 있다. 테스트가 옛 구성의 오배치 개수를 정확히 재현하고, 새 역순열이 시도한 모든 격자에서 항등으로 되돌아옴을 증명한다. 증명된 수정을 무관한 결함 뒤에 붙잡아 두면 틀린 순열을 트리에 남기게 되고, #1610을 검증할 사람이 결함 두 개를 동시에 붙들어야 해서 오히려 어려워진다.

### 3.5 #1610을 이 PR에 합치지 않았다

이슈의 Scope가 배제하고, 저장소의 one-issue-one-PR 관행도 그렇다. 실질적인 이유도 있다. #1610은 프로세서 출력 레이아웃을 바꾸므로 모든 크기의 모든 Youtu-VL 실행에 영향을 주고, 이번 변경은 윈도 하나를 넘는 경우의 병합 토큰 순서에만 영향을 준다. bisect 지점을 나눠 두는 편이 낫다.

### 3.6 내용을 단언하는 parity 테스트는 보류했다

이번 작업 중에 `tests/qwen2_5_vl_parity.rs` 모양의 `tests/youtu_vl_parity.rs`를 써 뒀다. 두 도형 픽스처 모두 이 계열에서는 비involution이라 이번 수정을 잘 지켜 준다. 그런데도 이 PR에 넣지 않은 이유는 #1610이 해결되기 전에는 통과할 수 없기 때문이다. 실패할 것을 아는 테스트를 커밋하는 쪽이 아예 안 넣는 쪽보다 나쁘다. #1610에 그 이슈를 고친 뒤 추가할 관문으로 적어 뒀다.

---

## 4. 검증

### 4.1 체크포인트 없는 검증

`cargo clippy --workspace --all-targets --features metal,accelerate -- -D warnings` 통과. `cargo test --workspace --profile test-fast --features metal,accelerate` 통과: 스위트 116개, 10499개 성공, 0개 실패, 332개 ignored. `cargo fmt --all -- --check` 통과. 여기서 `--workspace` 범위가 중요한데, 가정하지 않고 확인했다. 로그에 `mlxcel-core` 스위트가 찍혀 있으므로 #1007이 고친 루트 전용 `-p mlxcel` 형태가 아니다.

1.3장의 격자 표는 `smart_resize`와 `get_window_index`를 소스에서 읽어 크레이트 바깥에 따로 구현해 돌린 결과이고, 새 단위 테스트가 고정한 수치를 그대로 재현한다. 테스트를 되풀이한 것이 아니라 테스트를 검산한 것이다.

### 4.2 실제 체크포인트, 그리고 아무것도 안 보이는 이유

`models/mlx/youtu-vl-4b-instruct`(`tencent/Youtu-VL-4B-Instruct`), M1 Ultra, `metal,accelerate` 릴리스 빌드, greedy(`-t 0 -n 48`).

| 픽스처 | 병합 격자 | 수정 전 | 수정 후 |
|---|---|---|---|
| `test_image_shapes.png` (448) | 14x14, 196개 중 160개 오배치 | "The image contains a single, solid black circle on a white background." | 동일 |
| `test_image_shapes_336.png` (352) | 11x11, 121개 중 99개 오배치 | "The image contains a single, solid black circle on a white background." | 동일 |
| `test_image.png` (224) | 7x7, 단일 윈도, 오배치 0개 | "The image is completely black and contains no visible content, objects, text, or details." | 동일 |

셋째 줄이 진단용이다. 순열이 항등이라 이번 변경이 결과를 바꿀 수 없음이 증명되는데도, 단색 **주황** 사각형을 이미 검다고 묘사한다. 윈도 재정렬보다 앞에서 무언가가 모든 크기의 비전 경로를 망가뜨리고 있었다는 뜻이고, 그래서 비involution인 두 줄에서도 차이가 나올 수 없었다.

### 4.3 지배적 결함의 원인 규명

외부 오라클 없이도 두 가지 사실로 위치를 좁혔다. `YoutuVLVisionEncoder::rot_pos_emb`는 `reshape(&[h / merge, merge, w / merge, merge])` 뒤에 `transpose_axes(&[0, 2, 1, 3])`로 위치 id를 만드는데 이것이 merge-block-major이고, `forward_with_spatial`은 `[n_groups, spatial_merge_unit, dim]`으로 reshape하므로 연속한 네 행이 2x2 패치 블록 하나라고 가정한다. 그런데 `YoutuVLProcessor::try_preprocess_with_spatial`은 행을 평범한 래스터 순서로 내보낸다. 인코더의 두 입력이 서로 어긋나 있다. 별개로, 체크포인트의 `convert_image_to_patches`는 `(1, 4, 2, 5, 3, 6, 0)`으로 permute해서 merge-block-major 행과 `(dy, dx, c)` 내부 순서를 만드는데, 프로세서는 `(c, dy, dx)`를 내보내고 로더는 `patch_embedding.weight`를 이름만 바꿀 뿐 permute하지 않는다.

그 방출 반복문만 업스트림 순서로 바꿔 다시 빌드하니 224x224 픽스처가 "completely black"에서 "a solid, uniform block of bright orange color"로 바뀌었다. 기전이 확인된 것이다. 도형 세 개짜리 픽스처는 바뀌긴 했지만 여전히 틀렸으므로 **순서 수정은 필요조건이지 충분조건이 아니고**, 이 계열에 결함이 최소 하나 더 남아 있다. 그 패치는 되돌렸고 이 PR에 들어 있지 않다. #1610으로 발행했으며, 프로세서의 패치 상한 불일치는 #1611로 따로 발행했다.

---

## 5. 학습 포인트

**순열 버그는 하필 당신이 테스트하는 크기에서 안 보일 수 있다.** 이 결함도 #1596도 픽스처 크기가 그 계열에서 마침 involution이라 살아남았다. involution이 되는 집합은 윈도 폭과 프로세서 상한에 달려 있으므로, 버그 코드가 똑같아도 계열마다 다르다. 어떤 픽스처가 대표성이 있다고 판단하기 전에 도달 가능한 격자를 열거하는 편이 낫다.

**"동작이 안 바뀐다"는 실패가 아니라 발견이다.** 여기서 전후가 바이트 단위로 같았던 덕에 #1610이 드러났다. 아무것도 안 보이는 실행이 가치 있는 이유는 차이가 없다는 사실 자체를 설명해야 하기 때문이고, 그 설명이 아무도 발행하지 않은 더 큰 결함이었다.

**움직일 수 없는 대조군을 두자.** 이 보고서가 두 번째 결함의 존재를 추측이 아니라 주장으로 말할 수 있는 근거는 224x224 단일 윈도 사례다. 구성상 변경의 효과가 0으로 고정되므로, 거기서 나온 오류는 반드시 다른 곳에서 온 것이다.

**두 구성 요소가 업스트림과 각각 다르면서 서로도 어긋날 수 있다.** 프로세서와 `rot_pos_emb`가 서로 어긋나 있었고, 덕분에 참조 구현을 보기 전에 저장소 안에서만으로 결함을 증명할 수 있었다.

---

## 6. 후속 작업과 미검증 항목

- **#1610** (priority:high): 프로세서의 패치 방출 순서. 여기서 원인을 규명하고 재현했으며, 이 PR의 사용자 관점 이득을 가린다.
- **#1611**: `build_processor`가 프로세서 상한으로 `preprocessor_config.json`의 `max_num_patches`(256) 대신 `vision_config.num_patches`(4096)를 쓴다. 업스트림 상한을 따르면 병합 격자가 8x8 이하로 묶여서 이 PR이 고친 결함을 고치는 게 아니라 가리게 된다는 점을 알아 둘 필요가 있다. 둘을 대안으로 취급하면 안 된다.
- **Youtu-VL 비전 경로에 결함이 최소 하나 더 남아 있다.** 순서를 바로잡은 상태에서도 도형 세 개짜리 픽스처는 여전히 틀리게 묘사됐다. 아직 위치를 좁히지 못했다.
- **이 계열에 대한 참조 구현 대조는 하지 않았다.** #1601이 쓴 단계별 transformers 비교는 #1611이 해결되기 전에는 막혀 있다. 같은 이미지에 대해 mlxcel과 HuggingFace 프로세서가 서로 다른 패치 격자를 만들기 때문이다.
- **`docs/supported-models.md`는 Youtu-VL을 지원 목록에 올려 두고 있다.** #1610이 살아 있는 동안 그 주장은 낙관적이지만, 무관한 PR에서 흔들지 않고 그대로 뒀다.
