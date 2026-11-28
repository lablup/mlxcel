# 기술 보고서: PR #2080 - #2034와 #2037이 남긴 게이트 실패 수정

**날짜**: 2026-09-30

**상태**: gfx1151 호스트에서 구현 및 검증 완료, 머지 대기 중.

**언어**: Rust (테스트, 상수 하나), Python (fixture 생성기), PNG fixture

**위험도**: 낮음 (테스트 파일 두 개, fixture 하나와 그 생성기, `FAMILY_ORDER` 한 줄만 바뀌며 모델, 커널, 런타임 코드는 바뀌지 않습니다)

## 요약

hosted CI가 멈춰 있는 동안 기능 PR 두 개, #2034(Nemotron-Parse 포팅)와 #2037(Nemotron VoiceChat)이 머지되었고, 둘 다 CI였다면 걸렸을 게이트 실패를 main에 남겼습니다. #2034의 페이지 fixture는 `make verify-binary-assets`가 `tests/fixtures/*.png`에 적용하는 크기 한도를 넘습니다. #2037은 새 모델 family를 `FAMILY_ORDER`에 넣지 않아 `tests::family_order_is_exhaustive`를 깨뜨렸고, Metal에서 측정한 f32 값을 하드코딩한 gelu 테스트를 추가해 ROCm에서 1 ulp 차이로 실패합니다. 에픽 #1801 실행의 오케스트레이터는 두 테스트 실패를 모든 로컬 ROCm 게이트에서 발견했습니다. 이 둘은 에픽의 모든 PR을 판정할 때 기준으로 삼아야 했던 `verify-test-rocm` baseline 실패 세 개 중 두 개였습니다.

이 PR은 런타임 동작을 바꾸지 않고 세 가지를 모두 고칩니다.

1. fixture를 8비트 grayscale로 다시 인코딩했습니다. 41.1 KB에서 20.6 KB가 되었고, RGB로 디코딩하면 이전 파일과 바이트 단위로 같은 픽셀이 나옵니다.
2. `FAMILY_ORDER`의 `Text-to-speech` 다음에 `Speech`를 추가했습니다.
3. gelu 테스트는 고정된 MLX 버전의 upstream 식으로 같은 디바이스에서 실행 시점에 기준값을 만들고, 세 dtype에서 4097개 점을 비트 단위로 비교합니다. 백엔드와 무관한 f64 기준점을 함께 두어, 양쪽이 똑같이 틀린 식이어도 테스트가 실패하게 했습니다.

## 1. 문제 정의

#2034와 #2037이 머지될 때 hosted CI를 쓸 수 없었으므로, 두 PR 모두 이 회귀를 잡았을 게이트를 거치지 않았습니다. 세 실패는 서로 독립적입니다.

**Fixture 크기.** `scripts/ci/check_binary_assets.py`는 `tests/fixtures/*.png` 파일마다 32 KB 한도를 둡니다. #2034는 Nemotron-Parse 전처리와 parity 테스트용으로 텍스트 한 페이지를 렌더링한 `tests/fixtures/nemotron_parse_page.png`를 RGB로 저장해 41.1 KB로 추가했고, `make verify-binary-assets`가 이 파일에서 실패했습니다.

**Family 순서.** `src/main.rs`의 `FAMILY_ORDER`는 `mlxcel arch` 출력의 섹션 순서를 정합니다. 여기에 없는 family도 출력은 되지만(알파벳 순으로 뒤에 붙음), `src/main_tests.rs`의 `family_order_is_exhaustive`가 누락을 테스트 실패로 만들어 표시 위치를 의도적으로 정하게 합니다. #2037은 Nemotron VoiceChat(speech-to-speech)용 `Speech` family를 도입하면서 이 목록에 넣지 않았습니다.

**Gelu 기준값.** #2037은 Gemma 3 backbone의 `gelu_approx`를 위한 `gelu_approx_matches_mlx_nn_bit_for_bit`도 추가했습니다. 이 테스트는 입력 7개에서 f32와 bf16 출력을 정확히 단언했고, 그 값은 Metal의 mlx 0.32로 측정한 것이었습니다. ROCm의 x = 4.1 f32 결과는 4.0999565로, 하드코딩된 4.099957과 1 ulp 다릅니다. 둘 다 각 백엔드에서 옳은 값입니다. 포팅은 upstream과 같은 op를 실행하고, 백엔드마다 `tanh`/`power` 커널이 마지막 비트를 다르게 반올림할 뿐입니다. 한 백엔드의 커널 출력을 고정한 테스트로는 다른 백엔드에서 "upstream과 비트 단위로 같다"를 표현할 수 없습니다.

두 테스트가 모든 ROCm 실행에서 실패하는 동안, 에픽의 각 PR은 `verify-test-rocm` 결과를 이미 실패하는 baseline과 대조해 읽어야 했고, 그러면 같은 타깃의 새 실패가 가려집니다.

## 2. 변경 요약

- **`tests/fixtures/nemotron_parse_page.png`**: Pillow `optimize=True`로 8비트 grayscale(`L`)로 무손실 재인코딩, 42104바이트에서 21077바이트. 페이지는 흰 바탕에 antialiasing으로 렌더링한 검은 글자이므로 모든 픽셀이 이미 R == G == B입니다.
- **`tests/fixtures/generate_nemotron_parse_page.py`**: `img.convert("L").save(path, optimize=True)`로 저장하며, docstring에 인코딩 방식, 32 KB 한도, 디코딩한 픽셀이 바뀌지 않는 이유를 적었습니다.
- **`src/main.rs`**: `FAMILY_ORDER`의 `"Text-to-speech"` 바로 다음에 `"Speech"` 추가.
- **`src/models/gemma3_backbone_tests.rs`**: `gelu_approx_matches_mlx_nn_bit_for_bit`를 다시 작성하고 helper 두 개를 추가했습니다.
  - `mlx_nn_gelu_approx_reference`: 고정된 `python/mlx/nn/layers/activations.py`의 `0.5 * x * (1 + mx.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))` 줄을 옮긴 것.
  - `gelu_approx_f64`: 같은 식을 호스트에서 f64로 계산.

모델, 커널, CLI 동작, 문서는 바뀌지 않았습니다. `docs/supported-models.md`는 이미 ASR, TTS, speech-to-speech 섹션을 새 순서대로 나열합니다.

## 3. 기술적 선택과 그 이유

### 3.1 Fixture: 파일별 크기 예외 대신 grayscale 재인코딩

`check_binary_assets.py`는 파일별 규칙을 허용하므로 이 파일만 한도를 올리는 것은 한 줄 변경이었습니다. 그럴 필요가 없었습니다. 모든 픽셀이 회색이므로 RGB 파일은 각 샘플을 세 번 저장하고 있고, `L` 인코딩은 손실 없이 그 중복을 없앱니다. 다른 방법은 결과가 더 나빴거나 쓸 수 없었습니다. RGB `optimize=True` 재인코딩은 39.6 KB에 그쳤고, oxipng, optipng, pngcrush, zopflipng 모두 호스트에 설치되어 있지 않습니다.

중요한 성질은 소비하는 쪽이 같은 픽셀을 본다는 것입니다. Rust 쪽은 `image` crate(0.25.10)로 디코딩하고 `NemotronParseImageProcessor::preprocess_to_vec`에서 `to_rgb8()`를 호출하며, 기준값 쪽은 Pillow를 씁니다. 두 디코딩 경로를 모두 확인했습니다. 새 파일에 Pillow `convert("RGB")`와 `image`의 `to_rgb8()`를 적용하면 이전 RGB 파일과 같은 바이트가 나옵니다. 따라서 #2034가 체크포인트로 기록한 parity 값은 다시 유도하지 않아도 유효합니다. 생성기도 같은 인코딩으로 저장하므로, fixture를 다시 생성해도 한도를 넘는 파일이 돌아오지 않습니다.

### 3.2 `Text-to-speech` 다음의 `Speech`

위치는 오디오 섹션을 ASR, TTS, speech-to-speech 순으로 둔 `docs/supported-models.md`를 따릅니다. 순서를 맞추면 `mlxcel arch`와 문서가 같은 순서가 되어 문서를 고칠 필요가 없습니다. family가 실제로 쓰이므로 `family_order_has_no_orphans`도 계속 통과합니다.

### 3.3 Gelu: 기준값을 고정하지 말고 계산

테스트의 목적은 Rust 포팅이 `mlx.nn.gelu_approx`와 비트 단위로 같다는 것입니다. 출력을 하드코딩하면 그 주장이 한 백엔드의 커널에 묶입니다. 새 테스트는 같은 디바이스와 스트림에서 MLX op로 upstream 식을 계산하고 `gelu_approx`와 raw bit를 비교하므로, 포팅이 upstream과 같은 op를 실행하면 어느 백엔드에서든 정확히 통과합니다.

옮겨 적을 때는 포팅 코드가 아니라 Python 의미론을 따랐습니다.

- `*`와 `+`는 왼쪽 결합이므로 앞의 인자는 `(0.5 * x)`이고, tanh 항은 그 뒤에 곱합니다.
- `x**3`은 반복 곱셈이 아니라 `mx.power`입니다.
- 모든 Python float는 weak scalar가 되며, 바인딩은 이를 `array(float(v), x.dtype)`로 만듭니다. 즉 먼저 f32로 반올림하고 입력 dtype으로 cast합니다.

upstream은 함수를 `@partial(mx.compile, shapeless=True)`로 감쌉니다. 여기서는 양쪽 모두 compile하지 않은 식입니다. Metal에서 compile된 커널은 x = 4.1에서 compile하지 않은 식과 f32 1 ulp 다르고(#2037의 측정), bf16 출력은 compile하지 않은 식과 같았습니다. 그래서 테스트는 compile하지 않은 그래프와 op 단위로 같은지를 확인한다고 명시합니다.

입력은 점 7개 대신 [-8, 8] 구간의 4097개 점으로 된 촘촘한 격자이며, f32, bf16, f16에서 실행합니다. op 하나가 바뀌면 보통 원래와 다른 입력은 몇 개뿐입니다. ROCm에서 `power` 대신 `x * x * x`를 쓰면 4097개 f32 입력 중 2개에서만 다르므로, 점이 적으면 놓칠 수 있습니다. 실패하면 단언문이 불일치 개수와 처음 8개의 (x, got, want)를 출력합니다.

### 3.4 f64 기준점을 두는 이유

기준값은 같은 upstream 줄을 옮긴 것이므로 설계상 포팅과 같은 op를 실행합니다. 양쪽에 같은 잘못된 상수가 들어가면 비트 비교는 통과합니다. 테스트의 후반부는 원래의 입력 7개에서 `gelu_approx`를 호스트의 `gelu_approx_f64`와 비교하며, 허용 오차는 어느 백엔드에서든 성립합니다.

- f32: `1e-6 + 1e-6 * |want|`.
- bf16: `4e-3 + 1e-2 * |want|`. bf16은 op마다 반올림하므로 -3.0의 값이 모든 백엔드에서 -0.00364에서 -0.00586으로 바뀝니다. 그래서 한도는 가장 큰 중간값 기준으로 bf16 ulp 몇 개입니다.

이 허용 오차는 #2037이 Metal에서 고정한 값을 포함하며, Metal에서도 테스트가 통과하리라 보는 근거입니다.

### 3.5 의도적 파손

PR 작성자는 테스트가 실제 회귀를 여전히 잡는지 확인하려고 `gelu_approx`를 네 가지 방식으로 잠시 망가뜨렸습니다. 네 경우 모두 테스트가 실패했습니다.

| 파손 | 위치 | 잡은 검사 |
|---|---|---|
| `power(x, 3)` 대신 `x * x * x` | 포팅만 | 비트 비교, f32 4097개 중 2개 불일치 |
| `0.044715` 대신 `0.0447` | 포팅만 | 비트 비교, f32 4097개 중 2011개 불일치 |
| f32로 계산하고 입력 dtype으로 한 번만 반올림 | 포팅만 | 비트 비교, bf16 4097개 중 1270개 불일치 |
| `0.044715` 대신 `0.0447` | 포팅과 기준값 모두 | f32 f64 기준점 검사 |

마지막 행이 기준점을 둔 이유입니다. 양쪽이 일치하므로 비트 비교는 통과하지만, 호스트 검사는 여전히 실패합니다.

## 4. 검증

PR 작성자 (gfx1151, `--features rocm`):

- `cargo test -p mlxcel --lib models::gemma3_backbone::gemma3_backbone_tests`: 5개 통과.
- `cargo test -p mlxcel --bin mlxcel tests::family_order`: 2개 통과 (`family_order_is_exhaustive`, `family_order_has_no_orphans`).
- `cargo test --test nemotron_parse_real_model -- --ignored`: 빌드되고 실행됩니다. 호스트에 Nemotron-Parse 체크포인트가 없어 세 테스트 모두 skip합니다.
- `cargo test --test cli_help_consistency`: 27개 통과.
- `make verify-versions verify-kernel-dtype-keys verify-kernel-port-dispatch verify-llama-compat verify-rocm-overlay verify-fmt verify-binary-assets`: 통과.
- `cargo clippy -p mlxcel --lib --bin mlxcel --tests -- -D warnings`: 경고 없음.
- fixture의 디코딩 픽셀을 Pillow와 `image` crate로 이전 파일과 비교: 동일.

오케스트레이터:

- 이 PR에 대해 전체 `make verify-rocm`은 다시 실행하지 않았습니다. 변경은 테스트 파일 두 개, fixture 하나와 그 생성기, `FAMILY_ORDER` 한 줄에 한정되고, 위의 대상 실행이 각각을 다룹니다. 머지 후에는 에픽 실행의 마지막 게이트가 새 빌드에서 전체 suite를 확인합니다.

## 5. 학습 포인트

- **비트 단위 테스트는 기준값을 저장하지 말고 계산해야 합니다.** "upstream과 비트 단위로 같다"는 같은 백엔드에서 두 계산 사이의 관계입니다. 한 백엔드의 출력을 저장하면 "Metal과 같다"는 다른 주장이 되고, 다른 곳의 올바른 포팅에서 실패합니다.
- **옮겨 적은 기준값에는 독립적인 기준점이 필요합니다.** 기준값과 구현이 같은 줄에서 나오면, 비트 일치는 두 쪽이 같은 op를 실행한다는 것만 보여주고 그 op가 옳다는 것은 보여주지 않습니다. 정밀도가 낮고 백엔드와 무관한 검사가 그 틈을 메웁니다. 네 가지 의도적 파손은 어떤 검사가 어떤 종류의 오류를 잡는지 보여줍니다.
- **식이 아니라 언어의 의미론을 옮겨야 합니다.** Python의 왼쪽부터의 `*`, `mx.power`인 `**`, f32를 거쳐 반올림되는 weak scalar는 모두 마지막 비트를 바꿉니다. 수학 식에서 바로 쓴 기준값은 곳곳에서 upstream과 어긋났을 것입니다.
- **무손실 재압축은 소비자의 디코더로 확인해야 합니다.** grayscale fixture가 안전한 이유는 Pillow와 `image` 두 디코딩 경로가 모두 이전 RGB 바이트를 돌려주기 때문입니다. 파일 크기만 확인해서는 이를 알 수 없습니다.
- **CI가 멈춘 동안 건너뛴 게이트는 baseline 비용으로 돌아옵니다.** 이 회귀들은 고치기 쉬웠지만, 남아 있는 동안 에픽 실행의 모든 PR이 이미 알려진 실패 세 개와 대조해 평가되었습니다. baseline을 일찍 고쳐야 같은 타깃의 새 실패가 다시 보입니다.

## 6. 주의 사항과 검증하지 않은 부분

- **Metal과 CUDA**는 실행하지 않았습니다(이 호스트에 없음). gelu 테스트는 그곳에서도 통과할 것으로 봅니다. 포팅과 기준값이 같은 스트림에서 같은 MLX op를 같은 순서로 실행하므로 각 백엔드가 똑같이 반올림하고, 기준점 허용 오차는 #2037이 Metal에서 고정한 값을 포함합니다. fixture와 `FAMILY_ORDER` 변경은 백엔드에 따라 달라지지 않습니다.
- **compile된 upstream 커널**은 테스트의 비교 대상이 아닙니다. `mx.compile`은 compile하지 않은 그래프와 f32 1 ulp 다를 수 있으므로, compile된 커널과 정확히 일치하는 포팅은 이 테스트에서 실패합니다. 포팅과 테스트 모두 의도적으로 compile하지 않은 식을 목표로 합니다.
- **재인코딩한 fixture로 Nemotron-Parse end-to-end parity**는 체크포인트가 없어 skip 지점까지만 실행되었습니다. parity 값이 유지된다는 근거는 디코딩 후 픽셀이 같다는 사실입니다.
- **전체 `make verify-rocm` suite**는 이 브랜치에서 다시 실행하지 않았습니다(검증 절 참고).
- **세 번째 baseline 실패**인 `layers::tests::prefill_dense_gemm_matches_qmm_bytes_where_eligible`는 이 PR에서 다루지 않습니다.

참고: #1801, PR #2034, PR #2037, PR #2079.
