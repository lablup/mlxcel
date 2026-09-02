# 기술 보고서: PR #1601 - fix(qwen2_5_vl): f32 tower norm and a correct window inverse

**작성일**: 2026-09-03
**작성자**: mlxcel maintainers
**리뷰어**: implementation review cycle
**상태**: 완료 (두 가지 이미지 크기에서 transformers float32 오라클로 구간을 특정했고, 로컬 Qwen2.5-VL 체크포인트 두 개로 종단 검증함. 전체 워크스페이스 게이트는 중앙에서 실행)
**언어**: Rust, Python, Markdown
**위험도**: Medium (Qwen2.5-VL과 ColQwen2.5의 모든 이미지 순전파 수치가 바뀌고, involution이 아닌 격자에서는 병합 비전 토큰의 순서가 바뀐다)

---

## 요약

mlxcel이 Qwen2.5-VL 비전 타워로 보낸 이미지는 전부 일부가 지워진 채로 돌아왔다. 타워의 RMSNorm은 1280칸짜리 특징 축에서 `sum(x * x)`를 축약하는데, Qwen2.5-VL은 거대 활성값(massive activation)을 갖고 있다. 잔차 스트림의 한 채널이 블록 15에서 5.4e2, 블록 31에서 2.3e4에 이른다. float16에서 그 합은 65504에서 포화하고, norm은 `rsqrt(inf) = 0`을 돌려주며, 해당 토큰 행 전체가 0으로 블록을 빠져나간다. 448x448 이미지에서 블록 16의 1024개 타워 토큰 중 15개가, 블록 17에서 21개가 지워졌고 손상은 남은 블록을 지나며 누적됐다.

보고된 증상은 가벼워 보였다. 도형 세 개짜리 이미지를 "The image contains a red square and a green triangle."로 묘사하면서 파란 원을 빠뜨린 것이다. 실제 파급 범위는 그렇지 않았다. 저장소가 갖고 있는 유일한 이미지 픽스처인 224x224 단색 주황 사각형은 "a person wearing a white shirt and black pants, standing in front of a white wall with a black and white patterned rug on the floor"로 묘사됐다. 그 픽스처를 쓰는 VLM 테스트는 전부 토큰이 나왔는지만 확인하므로, 이 결함에는 커버리지가 아예 없었다.

패치 머저 뒤에는 별개의 두 번째 결함이 있었다. 병합 토큰을 윈도 순서에서 래스터 순서로 되돌리는 역재정렬이 `reverse_indices[orig_idx] = rank`를 만들었는데, `0..N`의 순열에서는 이 식이 `argsort(window_index)`가 아니라 `window_index` 자신을 재현한다. 순열을 두 번 적용해서 원래 순서로 돌아오려면 그 순열이 involution이어야 하고, 병합 격자 폭이 16이면 마침 그 조건이 성립한다. 그래서 보고서의 448x448은 멀쩡했고 336x336에서는 144개 병합 토큰 중 120개가 엉뚱한 자리에 놓였다.

이제 `VisionRMSNorm::forward`는 float16 입력을 축약 구간에서만 float32로 올리고, `invert_window_index`는 역순열을 직접 만든다. 타워의 투영과 어텐션은 그대로 float16으로 돈다. 승격 비용은 실행 간 잡음 안에 들어간다.

---

## 1. 문제 정의

### 1.1 배경

이슈 #1596은 합성 이미지(밝은 회색 배경 위 빨간 사각형, 파란 원, 초록 삼각형)를 그리디 디코딩에서 "The image contains a red square and a green triangle."로 묘사한다고 보고했다. mlx-community 4비트 변환본과 `Qwen/Qwen2.5-VL-3B-Instruct` 원본 bf16 익스포트 모두 같은 문장을 냈다. 같은 PNG와 프롬프트에서 transformers는 세 물체를 모두 언급했다.

보고서의 두 사실이 코드를 읽기 전에 이미 범위를 좁혀 줬다. bf16 익스포트와 4비트 변환본이 같은 문장을 냈으니 양자화 가중치 수치는 원인이 아니고, PR #1582 전후로 출력이 바이트 단위로 같았으니 원본 HF 로더 수정도 원인이 아니다.

조사 과정에서 나온 세 번째 사실이 첫 번째를 설명한다. mlx-community 4비트 변환본은 **비전 타워를 양자화하지 않는다.** `model.safetensors.index.json`에 `vision_tower.blocks.N.*.weight`는 있지만 짝이 되는 `.scales`가 없다. 두 체크포인트 모두 같은 bf16 타워를 돌리고, Apple Silicon에서는 `finish_vlm_weights_common`이 그것을 float16으로 변환한다. 블록 16 덤프가 두 체크포인트에서 바이트 단위로 같았고, 그래서 둘은 서로 일치하면서 transformers와만 어긋났다.

### 1.2 구간 특정

이슈가 요구한 단계별 diff는 transformers **float32** 오라클로 돌렸다. 참조 쪽의 bf16 반올림이 실제 격차를 가리지 못하게 하기 위해서다. 양쪽에 프로세서 출력, 패치 임베딩, 윈도 인덱스와 로터리 테이블, 모든 비전 블록, 역재정렬 전후의 패치 머저, 병합 입력 임베딩, MRoPE 위치 id, 프리필 로짓에 탭을 걸었다. 이미지 크기는 두 가지를 썼다. 448x448(병합 격자 16x16, involution)과 336x336(병합 격자 12x12, involution 아님)이다.

아래 `rel`은 오라클의 평균 절댓값 대비 평균 절대 오차이며, 수정 전 448x448 기준이다.

| 단계 | 결과 |
|---|---|
| 프로세서 `pixel_values`, `grid_thw`, 이미지 토큰 수 | rel 6.8e-8, `(1, 32, 32)`, 256개로 동일 |
| `PatchEmbed` `[1024, 1280]` | rel 2.2e-4, float16 반올림 바닥 |
| `window_index`, `cu_window_seqlens`, `cu_seqlens` | 정수까지 동일 |
| `rot_pos_emb` `[1024, 40]` | 정확히 0.0 |
| 비전 블록 0에서 14 | rel 4.6e-4에서 3.3e-3까지, 평탄 |
| 비전 블록 15 | rel 3.5e-3, 최대 절대 오차 4.06 (거대 활성값 536이 여기서 처음 등장) |
| **비전 블록 16** | **rel 3.8e-2, 최대 절대 오차 94.2** |
| **비전 블록 17** | **rel 2.8e-1, 최대 절대 오차 6.4e3** |
| 비전 블록 31 | rel 8.0e-1, 최대 절대 오차 1.9e4 |
| `PatchMerger` 출력 | rel 9.9e-1 |
| MRoPE 위치 id, `rope_deltas` | 정수까지 동일, -240 |

처음 갈라지는 단계는 **블록 16**이다. 그 위쪽은, 이슈가 용의선상에 올렸던 인덱스 산술을 포함해 전부 이미 맞았다.

같은 타워를 float32 활성값(가중치는 float16, MLX 타입 승격)으로 돌리면 블록 0부터 31까지 평탄하게 **rel 8e-6**이 되고, 블록 31의 `max|.|`도 23410.50 대 23410.75로 맞는다. 이슈가 물은 정밀도냐 구조냐에 대한 답이 여기서 나온다. 정밀도다.

### 1.3 넘치는 연산

블록 15, 16, 17에 블록 내부 탭을 걸어 보면 첫 발산은 어텐션이나 MLP가 아니라 `norm1` 안에 있다. 블록 16에서 norm 출력의 `max|diff|`가 11.56인데 `max|.|`가 12.10이다. 전체 행에 고르게 퍼진 반올림 오차가 아니라 일부 행이 통째로 틀렸다는 뜻이다.

float16 덤프에서 그 행들을 들여다보면, **블록 16의 `norm1`에서 1024개 행 중 15개가 정확히 0으로 나왔고 블록 17에서는 21개였다.** norm 입력에서 그 행들의 제곱합은 float16 상한을 기준으로 깔끔하게 갈린다.

| | 1280칸 축에서의 `sum(x * x)` |
|---|---|
| 지워진 행 중 최소 | 67,332 |
| 지워진 행 중 최대 | 348,750 |
| 살아남은 행 중 최대 | 54,696 |
| float16 최댓값 | 65,504 |

살아남은 행 중 65504를 넘는 것이 없고, 지워진 행 중 65504 아래인 것도 없다. float16에서 축약이 무한대로 포화하고 norm이 `rsqrt(inf) = 0`을 돌려준 것이다.

블록 안의 다른 연산은 넘치지 않는다. norm 출력 자체는 17 아래에 머물러 QKV와 MLP 투영은 작은 입력을 보고, MLX는 matmul을 float32로 누산한다. 거대 활성값을 *만드는* 것은 SwiGLU의 `down_proj`(블록 17에서 출력이 6359에 이른다)이지만 그건 제대로 만든다. 그 값을 제곱해야 하는 다음 norm만 실패한다.

### 1.4 두 번째 결함

`forward_with_grid`는 `(값, 인덱스)` 쌍을 정렬한 뒤 `reverse_indices[orig_idx] = rank`를 써서 병합 토큰을 역재정렬했다. `0..N`의 순열에서 값이 `v`인 원소는 항상 순위 `v`에 놓이므로, 이 루프는 `reverse_indices[i] = window_index[i]`로 퇴화한다. 역순열이 아니라 순열 자신을 재현한 것이다. 업스트림은 `reverse_indices = mx.argsort(window_index, axis=0)`으로 만든다(https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/qwen2_5_vl/vision.py).

순열을 두 번 적용하는 것이 항등이 되려면 involution이어야 한다. 덤프한 머저 출력으로 측정한 결과다.

| 병합 격자 | 이미지 | involution | 잘못 놓인 토큰 | 오라클 대비 머저 탭 |
|---|---|---|---|---|
| 16x16 | 448x448 | 예 | 256개 중 0개 | 어느 쪽이든 rel 3.3e-3 |
| 12x12 | 336x336 | 아니오 | 144개 중 120개 | 기존 rel 8.5e-1, `argsort` rel 3.2e-3 |

보고서의 크기로는 이 결함이 드러날 수 없었던 이유이고, 이슈가 두 번째 크기에서도 단계 diff를 요구한 것이 옳았던 이유이기도 하다.

### 1.5 결과

- **물체 누락과 오인.** 보고된 증상이다.
- **없는 내용 지어내기.** 지워진 행이 충분히 많으면 언어 모델에는 접지할 신호가 남지 않고 장면을 지어낸다. 단색 주황 픽스처에서 사람, 셔츠, 벽, 러그가 나왔다.
- **조용하다.** 출력은 끝까지 유한하고 자연스러웠다. NaN도, 오류도, 로그 한 줄도 없었다.
- **커버리지 없음.** 기존 VLM 테스트는 전부 단색 픽스처에 대해 출력이 비지 않았는지 또는 유한한지만 본다. 완전히 지어낸 묘사도 통과한다.

---

## 2. 변경 요약

| 파일 | 변경 |
|---|---|
| `src/vision/encoders/qwen2_5_vl.rs` | `VisionRMSNorm::forward`가 float16 입력을 축약 구간에서 float32로 올리고 float16으로 되돌린다. `invert_window_index`가 잘못된 역순열 구성을 대체하며 인덱스를 경계 검사한다. `get_window_index`의 본문을 자유 함수 `window_index_for_grid`로 추출했다. |
| `src/vision/encoders/qwen2_5_vl_tests.rs` | 윈도 역순열에 대한 체크포인트 없는 테스트 4개와 float16 축약에 대한 테스트 1개. 다섯 개 모두 수정 전 코드에서 실패한다. |
| `tests/qwen2_5_vl_parity.rs` (신규) | 묘사 내용을 단언하는 `#[ignore]` 실체크포인트 테스트 3개. 세 개 모두 수정 전 코드에서 실패한다. |
| `tests/fixtures/test_image_shapes.png`, `test_image_shapes_336.png`, `generate_test_image_shapes.py` (신규) | involution 크기와 아닌 크기 각각에 대한 물체 3개짜리 픽스처. 커밋된 생성기(Pillow 12.3.0)로 재현된다. |

---

## 3. 기술적 선택과 그 이유

### 3.1 타워가 아니라 축약만 승격한다

float32 진단 실행은 올바른 수정이면서 나쁜 수정이다. 32블록 1280폭 타워의 활성값 메모리를 두 배로 만들고 모든 투영에서 float16 처리량을 포기하는데, 문제는 연산 하나에서만 일어난다. 그래서 승격 범위를 `VisionRMSNorm::forward`로 한정했다. 범위(range)가 필요한 곳은 거기뿐이다.

비용은 두 갈래를 하나의 바이너리에 넣고 환경 변수로 토글해 측정했다. 빌드 배치 차이가 비교에 새어들지 않게 하기 위해서다. 타워 순전파 30회의 최솟값이다.

| 이미지 | 승격 적용 | 미적용 |
|---|---|---|
| 448x448 | 144.525 ms | 144.201 ms |
| 336x336 | 95.185 ms | 95.196 ms |

`[N, 1280]` 텐서 캐스트 두 번은 1280폭 투영 32블록 옆에서 무시할 만하고, 차이는 잡음 안에 있다(336 쪽은 승격했을 때가 명목상 더 빠르다). 디코드는 손대지 않았다. 타워는 이미지당 프리필에서 한 번만 돈다. CLI 벽시계 시간으로 이걸 재려던 초기 시도는 버렸다. 같은 소스의 두 빌드 사이에서 텍스트 전용 기준선이 77 ms나 움직였으니, 프로세스 단위 측정으로는 밀리초 미만 변화를 분해할 수 없다.

### 3.2 `mlxcel_core::rms_norm`은 건드리지 않는다

포화하는 축약은 워크스페이스의 모든 계열이 쓰는 공유 `rms_norm` 진입점의 성질이다. 거기서 dtype 동작을 바꾸면 계열 하나의 측정을 근거로 모든 float16 텍스트/비전 모델의 수치를 한 커밋에서 바꾸게 된다. 이 PR은 발견 사실을 기록하고, Qwen2.5-VL 자신의 비공개 `VisionRMSNorm`만 고친다. 이 타입은 같은 인코더를 통해 ColQwen2.5에서만 닿을 수 있고 그 밖에서는 닿을 수 없다. 다른 float16 계열도 같은 상한에 부딪히는지는 이 변경이 일부러 답하지 않은 진짜 질문이다.

### 3.3 타워를 bf16으로 유지하지 않는다

타워 가중치를 float16 대신 bf16으로 올려도 고쳐진다. bf16은 float32의 지수 범위를 갖기 때문이다. 채택하지 않았다. 공유 VLM 로더의 `finish_vlm_weights_common`을 건드려야 하고, 텍스트 임베딩 공간으로 넘어가는 머저 출력의 dtype이 바뀌며, 타워 전체에서 가수 비트를 11개에서 8개로 바꾸는 대가를 치른다. 결함이 요구하는 것보다 큰 동작 변경이다. RMSNorm 승격은 안전한 곳에서는 float16의 정밀도를 유지하고 안전하지 않은 곳에서만 범위를 산다.

### 3.4 순열을 가정하지 말고 역순열을 검사한다

`window_index_for_grid`는 구성상 `0..N`의 순열을 돌려주므로 검사 없는 `inverse[window_index[rank]] = rank`도 맞다. 그래도 인덱스를 경계 검사한다. 격자가 잘못되면 비전 타워 안에서 패닉이 나는데, 실제 경로에서는 이 검사의 비용이 0이다.

### 3.5 `window_index_for_grid` 추출

윈도 순열은 격자와 설정 정수 세 개에만 의존하는데, 정작 체크포인트가 있어야 만들 수 있는 인코더의 메서드로 살고 있었다. 본문을 자유 함수로 들어올리면 정말 중요한 성질인 "역재정렬이 역순열인가"를 가중치 없이 시험할 수 있고, 새 단위 테스트가 하는 일이 그것이다.

### 3.6 픽스처를 하나가 아니라 둘 커밋한다

이슈는 448x448 픽스처 하나를 요구했다. 336x336 렌더링을 함께 커밋한 이유는 448x448이 바로 역재정렬 결함이 보이지 않는 크기이기 때문이다. 그 크기만으로 만든 회귀 테스트는 재발을 잡지 못한다. 둘 다 생성기 하나에서 나오고, 448 출력은 이슈 재현 스니펫의 이미지와 바이트 단위로 같다.

### 3.7 내용을 단언하되 동의어를 허용한다

`tests/qwen2_5_vl_parity.rs`는 묘사가 각 물체와 각 색을 언급하는지를 대소문자 무시로 단언하되, 서로 바꿔 쓸 수 있는 단어 묶음으로 비교한다. 빨간 도형은 렌더 크기에 따라 "square"로도 "rectangle"로도 정당하게 불리므로 한쪽을 고정하면 얻는 것 없이 테스트만 깨지기 쉬워진다. 단색 테스트는 수정 전 환각이 실제로 만들어 낸 단어들의 *부재*까지 단언한다. "orange를 언급했는가"만으로는 지어낸 장면 안에 주황이 섞여 있어도 통과하기 때문이다.

---

## 4. 검증

전부 Apple M1 Ultra, `metal,accelerate`, 그리디(`-t 0`), 신규 토큰 48개, 별도 표시가 없으면 프롬프트는 "What shapes and colors are in this image? Answer briefly."다.

| 사례 | 수정 전 | 수정 후 | 오라클 |
|---|---|---|---|
| 448x448, bf16 익스포트 | The image contains a red square and a green triangle. | square, circle, triangle, red, blue, green | square, circle, triangle, red, blue, green |
| 448x448, 4비트 | 위와 동일 | The image contains a red square, a blue circle, and a green triangle. | 위와 동일 |
| 336x336, bf16 익스포트 | The image contains a red circle and a blue triangle. | rectangle, circle, triangle, red, blue, green. | rectangle, circle, triangle; red, blue, green |
| 336x336, 4비트 | A red circle and a blue triangle. | The image contains a red square, a blue circle, and a green triangle. | 위와 동일 |
| 1036x1036, 4비트 (병합 37x37, 패딩 윈도) | Green triangle and blue square. | Red square, blue circle, green triangle. | Red square, blue circle, green triangle. |
| 이미지 2장, 448x448 + 224x448, 4비트 ("Describe each image briefly, in order.") | 없는 이미지 4장, 첫 장이 반복됨 | 두 장에 걸친 실제 물체 4개를 모두 언급 | `grid_thw` `[[1, 32, 32], [1, 32, 16]]` 일치 |
| `tests/fixtures/test_image.png`, 4비트 ("Describe this image briefly.") | The image shows a person wearing a white shirt and black pants ... | The image is a solid orange color. | 해당 없음 |
| `qwen2-vl-2b-4bit`, 같은 이미지 | rectangle, circle, triangle | rectangle, circle, triangle | 인코더 파일이 다른 계열이므로 예상대로 무변화 |

448x448 bf16 결과는 transformers 그리디 출력과 **토큰 단위로 동일하다.** 1036x1036 4비트 결과는 오라클과 바이트 단위로 같다. 4비트 행이 오라클과 다른 것은 내용이 아니라 표현이며, 언어 모델이 4비트 양자화된 반면 오라클은 bf16이므로 예상된 차이다.

수정 후에는 (a)부터 (g)까지 모든 탭이 두 크기에서 일치한다. 프로세서 rel 6.8e-8, 패치 임베딩 rel 2.2e-4, 윈도와 MRoPE 인덱스 정수 동일, 최악 비전 블록 rel 5.4e-3, 머저 448에서 rel 3.3e-3 및 336에서 3.2e-3, 병합 이미지 슬롯 rel 3.3e-3 및 3.2e-3이다. (h)는 bf16 익스포트에서 정확히 일치한다.

스위트: `--lib vision::encoders::qwen2_5_vl` 9 passed, `--lib vision::processors::qwen2_vl` 4 passed, `--lib models::colqwen2_5` 6 passed, `--lib loading::vlm` 197 passed, `--test qwen2_5_vl_parity -- --ignored` 3 passed, `--test vlm_concurrency qwen2_5_vl -- --ignored` 통과, `cargo clippy --profile test-fast --lib --tests --features metal,accelerate -- -D warnings` 무경고, `cargo fmt --all -- --check` 통과. 픽스처를 재생성해 커밋된 PNG와 `cmp`하면 바이트 단위로 같다.

---

## 5. 학습 포인트

**값이 들어가는 dtype이 축약까지 담아 주지는 않는다.** float16은 23410을 잘 표현한다. 23410의 제곱은 표현하지 못하는데, RMSNorm은 그걸 해야 한다. 거대 활성값을 갖는 계열이라면 감사할 연산은 큰 값을 들고 있는 쪽이 아니라 그 값을 제곱하거나 누산해야 하는 다음 연산이다.

**포화하는 축약은 수식에서는 요란하고 출력에서는 조용하다.** `rsqrt(inf) = 0`은 유한하고 잘 정의된 0이다. 가드에 걸릴 NaN도, 예외도, 실패할 유한성 검사도 없다. 1024행 중 15행이 지워져도 문장은 여전히 매끄럽다.

**체크포인트 둘이 일치하는 것은 독립적인 측정 둘이 아니다.** 이슈는 4비트 변환본과 bf16 익스포트를 양자화 원인을 배제하는 독립 증거로 다뤘다. 실제로는 그보다 강하면서 약하다. mlx-community 변환본은 비전 타워를 양자화하지 않으므로 둘은 같은 타워를 비트 단위로 똑같이 돌렸다. 타워 접두어 아래에 `.scales`가 있는지 가중치 인덱스를 한 번 확인했으면 바로 드러났을 사실이다.

**순열 버그는 하필 시험하는 크기에서만 보이지 않을 수 있다.** involution은 자기 자신의 역순열이고, 448x448의 병합 격자 폭 16이 그 경우다. 역순열을 검증하는 테스트는 순열이 자기역이 아닌 크기에서 돌아야 하고, 새 단위 테스트는 두 경우를 모두 남겨 이 구분을 문서로 남긴다.

**틀릴 수 없는 픽스처는 아무것도 잡지 못한다.** 단색 주황 사각형이 테스트에 보여 줄 수 있는 실패는 두 가지뿐이다. 출력 없음과 비유한 출력. 환각은 보여 주지 못한다. 그 공백의 대가가 여기서는, 이 계열이 처리하는 모든 이미지를 망가뜨리던 결함이 초록색 스위트 아래 그대로 앉아 있던 것이다.

**밀리초 미만 변화는 바이너리 하나 안에서 잰다.** 빌드 둘을 CLI 벽시계로 비교하니 두 빌드 어느 쪽도 바꾸지 않았을 텍스트 전용 기준선이 77 ms나 흔들렸다. 단일 바이너리 안의 환경 변수 A/B는 같은 질문을 0.3 ms까지 분해했다.

---

## 6. 후속 작업과 검증하지 않은 부분

- **#1600**: `src/vision/encoders/youtu_vl_window.rs:147-159`의 `reverse_window_indices`가 동일한 잘못된 역순열을 갖고 있고 `youtu_vl.rs:425`가 쓴다. 자체 체크포인트와 검증 경로를 가진 별개 계열이라 여기서 눈감고 고치지 않았다.
- **다른 float16 계열.** 이 변경은 `mlxcel_core::rms_norm`에 에너지가 65504를 넘는 행을 넣는 모델이 또 있는지 조사하지 않았다. 공유 축약은 그대로이므로 그런 계열이 있다면 여전히 노출돼 있다.
- **비디오와 다중 프레임 입력.** 여기의 모든 측정은 `t = 1`인 정지 이미지다. `grid_t > 1`인 윈도 순열 경로는 단위 테스트로만 돌았고 오라클과 대조하지 않았다.
- **이슈가 범위 밖으로 둔 두 가지 차이**는 시험한 크기에서 무해함을 확인하고 그대로 뒀다. 프로세서가 `Qwen2VLImageProcessor`의 bicubic 대신 Lanczos3로 리샘플하는 점, 로더가 `preprocessor_config.json` 대신 코드 기본값으로 프로세서를 만드는 점이다. 448과 336 모두 28의 배수라 `smart_resize`가 리샘플을 하지 않고, 이 체크포인트의 사이드카는 기본값과 같은 `min_pixels` / `max_pixels`를 담고 있다. 28의 배수가 아닌 크기라면 리샘플러가 실제로 동작하는데, 그 측정은 하지 않았다.
