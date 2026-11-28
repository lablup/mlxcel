# PR #3310: OpenXLA packed 양자화 경로에서 bf16 scales와 biases 받기

**날짜**: 2026-11-23
**상태**: PR 열림(#603을 닫음). ROCm gfx1151에서 IREE `hip`으로 검증했다. GB10에서 CUDA로 돌리는 확인(`MLXCEL_XLA_DEVICE=cuda`, qwen3-0.6b-4bit의 packed와 dequant-at-load 비교)은 아직 남아 있다.

## 요약

`MLXCEL_XLA_QUANT=packed`는 MLX affine 양자화 가중치를 디바이스에 packed 상태로 두고 StableHLO 그래프 안에서 역양자화한다. 이 PR 전에는 그래프가 `scales` / `biases` 인자를 f16으로 선언했고 로더도 f16 바이트만 받았다. 그래서 affine 메타데이터를 bf16으로 저장하는 mlx-lm 체크포인트(Qwen3, Gemma3-27B, Qwen3-MoE)는 모두 로드 단계에서 실패했다. qwen3-0.6b-4bit에서는 `quant part model.layers.0.mlp.down_proj.biases dtype BF16, expected F16`이 나왔다.

이제 로더가 packed `scales` / `biases` 텐서 전부의 dtype을 safetensors 헤더에서 읽어 새 필드 `Config::quant_scale_dtype`에 저장하고, 이 필드 하나가 그래프 인자의 선언 타입과 업로드 버퍼의 dtype 코드를 함께 정한다. C shim에는 코드 3(`IREE_HAL_ELEMENT_TYPE_BFLOAT_16`)이 추가됐고, `dequant_affine`은 f16과 같은 방식으로 bf16을 그래프 안에서 f32로 넓힌다. f16과 bf16 메타데이터가 섞인 체크포인트는 경고를 남기고 로드 시점에 역양자화한다. 커널이 `half`로 읽는 `MLXCEL_XLA_QUANT=native`는 bf16 체크포인트에서 실패하는 대신 경고를 남기고 f16 가중치 경로로 돌아간다.

보안 리뷰에서 새 헤더 probe가 기존 경로 탈출 문제에 닿는다는 것이 드러났다. `model.safetensors.index.json`의 shard 이름을 검사 없이 모델 디렉터리에 붙이고 있었다. PR은 IREE 로더와 Phi-4 오디오 로더 양쪽에서 이를 고쳤다.

## 1. 문제 정의

packed 경로는 #568에서 f16 메타데이터만 지원하는 형태로 들어왔다. #569가 dequant-at-load 경로를 bf16 scale까지 넓혔지만, packed 경로는 여전히 `take_weight`와 `take_lm_head`(`emitter/model.rs`)에 `"f16"`을 하드코딩했고, `QuantRaw` 업로드(`iree.rs`)에서 `QuantPart::Scales | QuantPart::Biases`를 `(WDT_F16, Dtype::F16)`으로 고정했다. 이 이슈는 우선순위가 낮은 지원 범위 확장 작업으로 #574에서 분리됐다. #573 스파이크에서 IREE가 CUDA에서 dequant를 matmul에 융합하지 않는다는 것이 확인됐기 때문에 packed는 기본값이 꺼져 있고 그 환경에서는 f16 상주 가중치보다 느리다. #603의 목적은 upstream에서 융합된 quantized matmul이 들어올 때 packed 표현이 더 많은 체크포인트에 바로 쓰일 수 있게 준비하는 것이다.

핵심 어려움은 emitter가 `config.json`으로 그래프를 만드는데, `config.json`에는 메타데이터 dtype이 없다는 점이다. 타입은 가중치에서 가져와야 하고, 그래프 인자와 업로드 버퍼가 반드시 일치해야 한다. 어긋나면 IREE는 f16 선언 아래 bf16 바이트를 받거나 그 반대가 되어, 오류 없이 잘못된 값을 계산한다.

native GEMV 경로(`MLXCEL_XLA_QUANT=native`)는 같은 packed 버퍼를 쓰지만 커널(`csrc/qmv4.hip`, `csrc/qmv4.cu`)은 scale을 `half`로 읽는다. bf16 바이트를 넘기면 거부되지 않고 잘못 해석되므로, bf16이 packed 입력으로 허용되는 순간 이 경로에는 명시적인 차단이 필요했다.

## 2. 변경 요약

| 파일 | 변경 |
|---|---|
| `emitter/quant_scale.rs` (신규) | `QuantScaleDtype { F16, Bf16 }`, `elt()`(StableHLO 타입)와 `weight_dtype_code()`(1 또는 3), 텐서별 dtype을 하나의 값 또는 오류로 줄이는 `resolve_quant_scale_dtype` |
| `emitter/config.rs` | `Config::quant_scale_dtype`, 두 생성자 모두 기본값 `F16` |
| `emitter/model.rs` | `take_weight`와 `take_lm_head`가 `scales` / `biases`를 `c.quant_scale_dtype.elt()`로 선언 |
| `emitter/builder.rs` | `dequant_affine`이 f16 또는 bf16이 짝으로 맞는지 assert, 문서와 주석 갱신, 테스트 2개 |
| `iree.rs` | 헤더 probe `packed_scale_dtype`, `native_scale_dtype_check`, `RuntimeConfig::{set_,}quant_scale_dtype`, config 필드를 읽는 `QuantRaw` 업로드, packed와 native fallback, `index_shard_path` |
| `csrc/xla_iree.c` | `case 3`을 `IREE_HAL_ELEMENT_TYPE_BFLOAT_16`, esize 2로 매핑, 테스트용 바이트 카운터도 코드 3을 2바이트로 셈 |
| `phi4_audio.rs` | `resolve_weight_shards`가 `index_shard_path`를 거침 |
| `docs/environment-variables.md`, `CHANGELOG.md` | `MLXCEL_XLA_QUANT` 행에 bf16 메타데이터, 혼합 fallback, native fallback 설명, Changed와 Fixed 항목 |

### 로드 흐름

단일 시퀀스 `IreeLlama` 로드에서 `quant_in_graph()`가 켜져 있으면 로더는 `packed_args`를 켠 뒤 `packed_scale_dtype(model_dir, &cfg)`를 호출한다. probe는 `cfg.weight_specs()`에서 `part: Scales | Biases`인 `QuantRaw` spec만 골라내므로, 로더가 실제로 업로드할 텐서만 정확히 검사한다. `model.safetensors` 또는 index에 적힌 shard를 열어 mmap하고, `SafeTensors::deserialize`로 헤더만 파싱한 뒤, 각 이름을 그대로와 `language_model.`로 감싼 형태 둘 다로 찾는다. dtype은 `F16`, `Bf16`, `None` 중 하나로 바뀌고, `resolve_quant_scale_dtype`은 단일 타입, 빈 목록이면 f16 기본값, 혼합이거나 지원하지 않는 dtype이면 해당 텐서 이름을 담은 오류를 돌려준다.

- `Ok(dtype)`: `cfg.set_quant_scale_dtype(dtype)`. emitter는 그 타입을 선언하고, 업로드는 `Dtype::F16` 또는 `Dtype::BF16`을 기대하며 코드 1 또는 3을 넘긴다.
- `Err`: 경고 한 줄(`MLXCEL_XLA_QUANT=packed cannot keep this checkpoint packed (...); dequantizing at load instead`)을 남기고 `set_packed_args(false)`. #569에서 bf16까지 넓어진 dequant-at-load 경로가 대신 체크포인트를 처리한다. 5절의 혼합 dtype 실행에서 로드되고 결과가 일치하는 것을 확인했다.

업로드 쪽은 아무것도 다시 계산하지 않는다. `QuantRaw` 분기는 기대하는 safetensors dtype과 C 코드 둘 다를 `cfg.quant_scale_dtype()`에서 읽으므로, 그래프 인자와 버퍼가 같은 필드에서 나온다. Gemma3n은 가중치 목록에 raw 양자화 부분이 없으므로 `F16`을 돌려준다.

### 그래프 안의 역양자화

`dequant_affine`은 원래도 `q * scale + bias` 전에 메타데이터를 f32로 변환했다. f16과 bf16 모두 f32로 정확하게 넓어지므로, bf16 형태는 메타데이터 타입만 바뀐 f16 그래프와 같고, 호스트의 bf16 역양자화와 비트 단위로 일치한다. 새 assert는 짝이 맞지 않는 경우(bf16 scales에 f16 biases)나 다른 element type을 emit 시점에 거부한다.

### Native fallback

native 진입점 두 곳(단일 시퀀스와 ragged) 모두 packed 플래그를 켠 뒤 `native_scale_dtype_check`를 호출한다. 전부 f16이 아니면 `...; falling back to the f16 weight path`를 기록하고 `packed_args`를 끈다(단일 시퀀스 그래프에서는 `native_head`도 끈다). native 분기는 `quant_scale_dtype`을 f16 기본값 그대로 두므로, native가 켜져 있는 동안 업로드는 계속 F16 바이트를 요구하고 bf16 바이트는 커널에 닿을 수 없다.

### Index shard 이름

`checkpoint_candidate_shards`(새 probe, native head probe, 가중치 로더가 사용)와 `phi4_audio.rs`의 `resolve_weight_shards`는 `weight_map` 값을 그대로 모델 디렉터리에 붙였다. 그래서 `../x`, 절대 경로, `nested/x`를 쓰면 로더가 체크포인트 밖의 파일을 열었다. `index_shard_path`는 NUL이 없고 component가 정확히 `Component::Normal` 하나인 이름만 받는다. `mlxcel-core`의 `validate_index_shards`가 이미 쓰는 규칙과 같다. 악의적인 항목은 이제 index와 shard 이름을 담은 오류로 로드를 실패시키며, 오래된 index가 여전히 타는 디렉터리 스캔 fallback으로 넘어가지 않는다.

## 3. 기술적 선택과 그 이유

**dtype은 `config.json`이나 데이터 probe가 아니라 safetensors 헤더에서 읽는다.** 이슈는 precision을 `resolve_precision`으로 전달하는 방식을 따라가자고 제안했지만, precision은 환경 변수와 디바이스에서 오고 메타데이터 dtype은 가중치의 속성이다. 헤더가 유일하게 믿을 수 있는 출처이고, 로더가 원래 파싱하는 것이며, 읽어도 헤더 페이지만 건드린다. 리뷰에서 Qwen3-4B, Gemma3-12B, Qwen3.6-35B-A3B(shard 1개에서 4개)의 헤더가 100에서 270 KB이고, 로드당 몇 ms라고 측정했다.

**선언과 업로드에 config 필드 하나.** #603 전 코드는 emitter와 로더가 둘 다 f16을 하드코딩하는 방식으로 일치를 유지했다. 상수 두 개를 독립된 계산 두 개로 바꾸면 어긋날 위험이 다시 생긴다. 결과를 `Config`에 한 번 저장하고 `take_weight`, `take_lm_head`, `QuantRaw` 분기가 그것을 읽게 하면 구조적으로 불일치가 불가능하다.

**체크포인트당 scale dtype 하나, 혼합이면 fallback.**

| 대안 | 장점 | 단점 |
|---|---|---|
| 텐서별 선언 타입 | 혼합 체크포인트도 packed 유지 | 그래프 모양이 텐서별 맵에 의존, 어떤 변환기도 쓰지 않는 레이아웃 때문에 MLIR 변형과 캐시 항목이 늘어남 |
| 혼합 메타데이터를 로드 오류로 거부 | 단순함 | 로드되던 체크포인트가 성능 플래그 하나 때문에 로드되지 않음 |
| **선택: dtype 하나, 혼합이면 경고와 함께 dequant-at-load** | 그래프가 `Config`만의 함수로 남고 체크포인트는 계속 로드됨 | 혼합 체크포인트는 packed 표현을 잃음 |

qwen3-0.6b-4bit의 layer 3 `up_proj`를 F16으로 바꿔 써서 fallback을 확인했다. 경고가 나왔고 토큰 id가 일치했다.

**bf16 커널을 추가하지 않고 native를 막는다.** `qmv4.hip` / `qmv4.cu`를 bf16으로 확장하는 일은 별도의 커널과 성능 작업이다. PR은 커널을 그대로 두고, 지원하지 않는 경우를 경고와 f16 가중치 경로로 처리한다. 전에는 로드 오류였다.

**Rust 코드와 C `switch`를 소스 텍스트 테스트로 묶는다.** `weight_dtype_codes_match_the_c_shim`은 `include_str!`로 `csrc/xla_iree.c`를 읽어 두 코드에 대해 `case N: elt = ...; esize = 2;` 텍스트가 정확히 있는지 확인한다. 이 코드를 쓰는 곳은 C shim뿐이고, 어긋나면 런타임 오류 없이 bf16 바이트가 다른 element type으로 올라간다. 이 테스트는 해당 `switch`의 포맷 변경에 민감하지만, 실패가 분명하게 드러나므로 받아들일 만하다.

**악의적인 index에는 fail-closed.** 경로 탈출 이름은 오래된 index가 아니라 악의적인 체크포인트의 증거로 보고, 디렉터리를 스캔하는 대신 로드를 실패시킨다. 로컬 체크포인트 index 49개가 모두 단순한 이름을 쓰므로 실제 체크포인트의 동작은 바뀌지 않는다.

## 4. 리뷰 결과

**보안과 성능 리뷰**(PR 코멘트): LOW 하나, bf08e0a2에서 수정. 기존부터 있던 문제지만 새 probe가 지나는 경로에 있다. 새 테스트 `candidate_shards_reject_index_names_outside_the_model_dir`는 `../outside.safetensors`, 절대 경로, `nested/inner.safetensors`, `./../outside.safetensors`를 다루며, 이전 코드에서는 실패한다.

확인했고 변경이 필요 없었던 항목:

- 헤더 파싱은 mmap 위의 `SafeTensors::deserialize`를 쓴다. 헤더는 100 MB로 제한되고, shape와 offset 산술은 검사되며, 각 텐서의 바이트 범위는 BF16을 포함해 shape 곱하기 element 크기와 같아야 한다.
- `QuantRaw` 업로드: dtype이 config의 scale dtype과 같아야 하고 rank는 2여야 하므로, C 복사 길이(코드 3이면 `dims * 2`)는 safetensors 범위와 같다. `xla_aux.c`는 코드 3을 알 수 없는 코드로 거부하고, aux 경로에는 packed 부분이 들어오지 않는다.
- VMFB 캐시 키와 번들 fingerprint는 MLIR 텍스트를 해시하므로, f16 아티팩트가 bf16 그래프에 쓰일 수 없다.

**정확성 리뷰**: 수정이 필요한 지적은 없었다. 다시 빌드한 바이너리로 qwen3-0.6b-4bit를 돌렸고, packed와 native(fallback) 모두 32토큰에서 dequant-at-load와 일치했다.

## 5. 테스트와 검증

새 테스트:

- `quant_scale::tests`(6개): 전부 bf16이면 bf16, 빈 목록과 f16은 f16 유지, 혼합 메타데이터는 두 텐서 이름을 담아 거부, F32 scale 거부, `packed_graphs_declare_the_config_scale_dtype`(두 projection 레이아웃에서 decode와 prefill을 emit하고 bf16 그래프가 f16 그래프에서 `xf16>`을 `xbf16>`으로 바꾼 것과 같은지 확인), `weight_dtype_codes_match_the_c_shim`.
- `builder` 테스트(2개): bf16 메타데이터가 그래프 안에서 f32로 넓어지고 f16 형태와는 메타데이터 타입만 다름, bf16 / f16 짝은 panic.
- `checkpoint_name_tests::candidate_shards_reject_index_names_outside_the_model_dir`.

main에서는 shim 코드 테스트와 emitter bf16 선언 테스트가 실패한다.

gfx1151, IREE `hip`, `--temp 0 -n 32` 실제 모델 실행:

| 체크포인트 | 실행 | 결과 |
|---|---|---|
| qwen3-0.6b-4bit (BF16) | main, packed | 로드 오류, `expected F16` |
| qwen3-0.6b-4bit | packed 대 dequant-at-load | 32/32 id 동일 |
| qwen3-1.7b-4bit (BF16) | packed 대 dequant-at-load | 32/32 id 동일 |
| qwen3-0.6b-4bit | native | 로드 오류였으나 이제 경고 후 fallback, id 동일 |
| qwen3-0.6b-4bit, layer 3 `up_proj`를 F16으로 변경 | packed | 혼합 dtype 경고, 로드 시 역양자화, id 동일 |
| llama-3.2-1b-instruct-4bit (F16) | packed, main 대 브랜치 | id 동일, 그래프 세 개 모두 main의 캐시 항목 재사용 |

#515 게이트는 토큰 단위로 정확히 일치하는 greedy 궤적이고, 이 실행들은 모두 토큰 단위로 일치한다. llama 행은 f16 체크포인트가 바이트 단위로 같은 MLIR을 만든다는 근거다. VMFB 캐시 키가 MLIR 텍스트를 해시하므로, main의 항목에 캐시가 적중했다는 것은 텍스트가 바뀌지 않았다는 뜻이다. `registered_fixtures_are_byte_exact`도 그대로 통과한다.

게이트: `cargo test --profile test-fast -p mlxcel-xla --lib` 455개 통과(`iree` feature를 켜고 ROCm IREE 인자로 직접 링크하면 480개), mlxcel-xla lib과 테스트에 대해 `--features iree` 유무 모두 clippy `-D warnings`, `cargo fmt`.

처리량은 측정하지 않았다. PR은 속도 향상을 주장하지 않는다. gfx1151에서는 호스트에서 역양자화한 f16이 packed보다 decode가 빠르다는 점이 이미 문서에 있고, 거기서 packed의 이점은 업로드 바이트가 약 3.5배 적다는 것이다.

## 6. 남은 작업과 한계

- **CUDA(GB10) 일치 확인이 남아 있다.** #603의 마지막 수용 기준(`MLXCEL_XLA_DEVICE=cuda`에서 qwen3-0.6b-4bit의 packed와 dequant-at-load 비교)은 아직 실행하지 않았다. 검증된 것은 `hip` 타깃뿐이다.
- **native 커널은 f16 전용으로 남는다.** bf16 체크포인트는 native GEMV 가속을 잃고 f16 가중치 경로로 돈다. `qmv4.hip` / `qmv4.cu`의 bf16 변형은 별도 변경이다.
- **혼합 메타데이터 체크포인트는 packed로 가지 않는다.** 이런 파일을 쓰는 변환기는 알려져 있지 않으며, 로드 시 역양자화한다.
- **ragged 배치 그래프에는 일반 packed 경로가 없다.** native 분기에만 차단을 추가했고, 이 PR은 그쪽에 packed 지원을 넣지 않는다.
- **packed는 여전히 기본값이 꺼져 있고**, #574(IREE upstream의 융합 quantized matmul)에 묶여 있다.
- **소스 텍스트 shim 테스트**는 `xla_iree.c`의 `switch` 포맷이 바뀌면 같이 고쳐야 한다.

## 7. 학습 포인트

- 그래프 인자 타입과 호스트 버퍼 타입이 일치해야 할 때는 타입을 한 번만 계산해 양쪽이 읽는 곳에 저장한다. 올바른 계산 두 개도 어긋날 수 있지만, 저장된 값 하나는 어긋나지 않는다.
- bf16에서 f32로의 변환은 정확한 확장이다(bf16은 f32의 상위 절반). 그래서 확장을 호스트에서 그래프 안으로 옮겨도 결과가 비트 단위로 같고, 허용 오차 없이 토큰 일치 게이트를 통과한다.
- 바이트를 재해석하는 커널(bf16 데이터를 `half`로 읽기)은 조용히 틀린다. 차단은 바이트가 커널에 닿기 전에 있어야 하고, 공유 필드의 기본값(f16)은 제약이 있는 소비자가 받아들이는 값이어야 한다.
- 생성된 소스 텍스트를 키로 쓰는 캐시는 공짜 회귀 검사가 된다. 관련 없는 설정이 이전 캐시 항목을 재사용한다면 그 생성 소스는 바뀌지 않은 것이다.
- 신뢰할 수 없는 입력을 읽는 새 코드는 검사 없는 기존 경로를 새 위치에서 닿게 만들 수 있다. probe 자체만이 아니라 probe가 호출하는 헬퍼까지 검토했기 때문에 index 경로 탈출을 찾았다.
