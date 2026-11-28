# PR #3242: 운영 보조 아티팩트에 연산자 수치 계약 결합

**날짜**: 2026-11-23
**상태**: PR 열림 (#3211 해결, #932 및 에픽 #566의 일부). 단위 테스트 통과. GB10에서 CUDA로 vision 런타임 6종, `local-task`로 Gemma3n audio를 실제 체크포인트로 검증

## 요약

OpenXLA/IREE 백엔드는 멀티모달 모델의 vision 인코더와 audio 타워를 캐시된 VMFB "보조 아티팩트(auxiliary artifact)"로 컴파일하고, `IreeAuxiliaryModule::load`가 모듈을 실행하기 전에 각 아티팩트의 매니페스트가 일치하는지 확인합니다. #934와 #937이 연산자 수준의 수치 계약을 담는 v3 매니페스트 스키마를 정의했지만 운영 경로의 어떤 패밀리도 이를 쓰지 않았습니다. 운영 호출 지점 11곳 모두 `new_legacy_unqualified` 계약을 만들어 수치 정보가 없는 v1 매니페스트를 썼고, v3를 쓰는 곳은 micro-oracle 프로브뿐이었습니다. 그래서 일곱 패밀리 포팅이 모두 머지된 뒤에도 #932의 첫 번째 수용 기준(영향받는 모든 아티팩트의 fingerprint에 버전이 붙은 수치 계약이 포함되고, 불일치하면 실행 전에 거부)은 충족되지 않은 상태였습니다.

이 PR은 패밀리가 선언한 연산자 목록, emitter가 사용한 `Precision`, 로더가 관찰한 체크포인트 dtype 분류, IREE 컴파일러, 런타임, 디바이스 정보를 하나의 `OperatorNumericContractSet`으로 바꾸는 빌더를 추가합니다. 이제 모든 운영 패밀리가 이 빌더를 거쳐 v3 매니페스트를 씁니다. 캐시된 아티팩트에 저장된 계약이 어느 필드에서든 다르면 로드 전에 다시 컴파일합니다. 리뷰 중에 구멍 하나를 찾아 막았습니다. Gemma3n의 BF16 호스트 역양자화 반올림이 F32 역양자화와 같은 계약을 만들고 있었습니다. 또 다른 수정은 v3 아티팩트를 스키마 이름의 하위 디렉터리에 두어, 같은 `/tmp` 캐시를 쓰는 이전 바이너리가 계속 동작하게 합니다. IREE가 디스패치마다 고르는 커널은 관찰 불가로 기록합니다. 이 계약은 아티팩트의 정체성을 묶을 뿐 운영 수준 검증(qualification)을 주장하지 않습니다.

## 1. 문제 정의

### 1.1 v1 매니페스트가 묶은 것과 놓친 것

v1 보조 계약은 entry 이름, config identity, generation identity(컴파일러 경로와 SHA-256, `iree-compile --version`, 플래그, MLIR digest)를 기록합니다. 그래프와 컴파일러 변경은 잡아내지만, 모듈이 내는 수치를 결정하는 입력 일부는 이름이 없었습니다.

- **링크된 IREE 런타임.** 런타임 아카이브는 컴파일러 호출이 아니라 이를 사용하는 바이너리의 일부이므로, 런타임을 다시 빌드해도 매니페스트는 그대로였습니다.
- **호스트 쪽 가중치 전처리.** Qwen2-VL, Qwen3-VL, Gemma3n은 affine Q4 가중치를 업로드 전에 호스트에서 역양자화합니다. MLIR에는 dense F32 입력만 보이므로, 역양자화 순서와 반올림은 MLIR digest에 반영되지 않았습니다.
- **체크포인트 dtype 분류와 연산자별 정밀도.** 이 값들은 config identity 안의 자유 형식 문자열로만 들어가 있었고, 검증되는 필드가 아니었습니다.

#932는 이것들을 패밀리별 문자열이 아니라 버전이 붙은 공용 descriptor 하나로 요구했습니다. descriptor 자체(`operator_numeric_contract.rs`, `aux_manifest.rs` v3)는 있었지만, 생성자 `new_with_operator_numeric_contract`에는 `#[allow(dead_code)]`가 붙어 있었습니다.

### 1.2 방치했을 때의 위험

| 위험 | 영향 | 가능성 |
|---|---|---|
| 런타임이나 호스트 역양자화가 바뀌었는데, 기록된 수치가 더 이상 실제 실행을 설명하지 못하는 캐시 아티팩트를 재사용 | 중간: 나중의 oracle이 원인을 특정할 수 없는 조용한 drift | `/tmp` 캐시를 유지하는 개발 장비와 self-hosted runner에서 중간 |
| #3212의 연산자 프로브가 운영 아티팩트에는 없는 계약 정체성을 보고하여 프로브 근거와 운영 근거를 연결할 수 없음 | #932에 높음 | 이 PR이 없으면 확실 |

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| `operator_numeric_contract/production.rs` (신규) | `production_operator_contract(family, operators, precision, weights, backend)`. 네 입력으로부터 각 항목의 체크포인트 dtype, 입력 dtype, accumulator(항상 F32), 결과 dtype, 반올림 경계, 가중치 실행 방식, 역양자화 계약을 유도합니다. 연산자 분류에 맞지 않는 입력 정책은 거부합니다. `CheckpointSourceObserver`는 가중치 로딩 중 텐서별 dtype 분류를 모으고, affine 레이아웃이 섞이면 실패합니다(fail closed). |
| `operator_numeric_contract/families.rs` (신규) | 런타임마다 선언 함수 하나: LLaVA/SigLIP/Gemma3, Qwen2-VL/Qwen2.5-VL, Qwen3-VL(main과 DeepStack), Molmo, Molmo2, Youtu-VL, Gemma3n audio encode와 merge, Phi4MM audio. 공용 헬퍼가 공통 인코더 레이어, attention 블록, merger를 만듭니다. |
| `iree_identity.rs` (신규) | `compiler_version`과 `runtime_archive_identity`를 `numeric_probe.rs`에서 옮겨 프로브, 운영 경로, #3212가 구현 하나를 공유합니다. `compiler_version_output`은 컴파일러 경로, 크기, mtime 기준으로 memoize됩니다. `production_artifact_contract`가 v3 계약을 조립합니다. |
| `build.rs`, `Cargo.toml` | dist, CUDA, ROCm, macOS 레이아웃에서 링크하는 `libiree_runtime_unified.a`의 SHA-256을 `MLXCEL_XLA_IREE_RUNTIME_SHA256`으로 빌드 시점에 넣습니다. build dependency로 `sha2`를 추가했습니다. |
| 런타임 파일 7개와 `phi4_audio.rs` | `new_legacy_unqualified` 호출 11곳을 모두 교체했습니다. 로더가 observer에 dtype을 넘깁니다. 정밀도가 고정된 emitter(LLaVA, Qwen3-VL, Molmo, Molmo2, Youtu-VL)와 Gemma3n audio는 emitter와 계약이 함께 읽는 이름 붙은 상수에서 정밀도를 가져옵니다. Qwen2-VL과 Phi4MM은 emit할 때 쓴 resolve된 값을 그대로 넘깁니다. |
| `aux_manifest.rs` | `schema_scoped_vmfb_path`와 `qualify_auxiliary_artifact`가 v1이 아닌 아티팩트를 매니페스트 스키마 이름의 하위 디렉터리에 둡니다. |
| `operator_numeric_contract.rs` | `AffineQ8Dequantize` 분류와 `SeparateMultiplyThenAddBf16Rounded` 평가 순서를 추가했습니다. 모듈 전체의 `dead_code` 허용을 없앴고, 운영 항목이 만들지 않는 variant만 이유를 단 항목 단위 허용을 유지합니다. |
| 테스트 | `production_tests.rs`, `family_tests.rs`: 필드 분류마다 오래된 캐시 거부와 재컴파일 하나씩, v1에서 v3로의 교체, observer 동작, 기본 CUDA와 CPU 정밀도에서 패밀리별 고정(pin), 항목 제거 감지, emitter와 계약의 정밀도 불일치, 호스트 역양자화 반올림의 정체성 반영. |
| 문서 | `docs/environment-variables.md`에 v3 매니페스트, 스키마 하위 디렉터리, 계약이 주장하지 않는 것을 설명했습니다. |

파일 26개, +2340/-224. 텍스트 전용 token/decode emitter와 그 golden, 모든 tolerance는 바뀌지 않았습니다.

## 3. 기술적 선택과 그 이유

### 3.1 패밀리는 연산자 분류만 선언하고 나머지는 유도

각 패밀리는 그래프가 내는 연산자 분류를 피연산자 출처와 입력 정책이 다른 경우마다 한 항목씩 나열합니다. 예를 들면 `weighted("encoder.mlp-down-projection", DenseMatmul)`나 `attention(scores, softmax, context)`입니다. 선언에는 네 가지만 들어갑니다. 무엇을 읽는지(`MatrixWeight`, `FloatParameter`, `Activation`), 입력을 어떻게 materialize하는지(`FollowsPrecision`, `PinnedF32`, `F32Carrier`), association 정책, 그리고 선택적인 명시적 출력 반올림입니다. 나머지는 빌더가 유도합니다. `FollowsPrecision` contraction은 emitter의 narrow 타입을 입력 dtype으로 받고 `OperatorInput` 반올림 경계가 붙습니다. affine 체크포인트의 `MatrixWeight` 연산자는 `AffineU4`/`AffineU8`, 역양자화 계약, `HostAffineDequantized` 또는 `PackedAffineInGraph`를 받습니다. 누적하는 분류에는 `AccumulatorResult`가 붙습니다.

대안은 패밀리마다 항목별 dtype을 모두 직접 적는 방식이었습니다. 그러면 F32로 emit된 그래프에 BF16 입력을 선언하는 상태를 쓸 수 있습니다. emitter가 받은 바로 그 `Precision` 값에서 유도하면 그런 상태는 표현할 수 없고, `precision_mismatch_between_emitter_and_contract_changes_identity`가 정밀도가 다르면 정체성이 바뀌는지 확인합니다. pooling, RoPE, masking, gather는 한 번만 반올림하고 accumulator가 없으므로 목록에서 뺐습니다. 이것들은 generation identity의 MLIR digest가 여전히 묶습니다.

### 3.2 상수 하나가 emitter와 계약을 함께 결정

정밀도가 고정된 vision 그래프 다섯 개는 기본값이 F32인 `Builder::new()`를 호출했고, 런타임에는 묶을 값이 없었습니다. 이 PR은 `LLAVA_VISION_PRECISION: Precision = Precision::F32` 같은 상수를 추가하고 emitter가 `.with_precision(LLAVA_VISION_PRECISION)`를 호출하게 합니다. 값이 기본값과 같으므로 emit되는 MLIR은 바뀌지 않지만, 이후 상수를 고치면 그래프와 계약이 함께 바뀝니다. Qwen2-VL과 Phi4MM은 이미 디바이스나 환경 변수에 따라 정밀도를 resolve하므로, 런타임이 그 값을 그대로 넘깁니다.

### 3.3 Gemma3n BF16 반올림 구멍 (리뷰에서 발견)

Gemma3n audio는 affine 가중치를 `dequantize_affine_bf16_sequential`로 역양자화합니다. 곱하고, BF16으로 반올림하고, bias를 더하고, 다시 BF16으로 반올림합니다. MLX의 BF16 affine 동작과 같습니다. Qwen2-VL과 Qwen3-VL은 F32에서 끝나는 `dequantize_affine`을 씁니다. 두 경로가 모두 `AffineEvaluationOrder::SeparateMultiplyThenAdd`로 매핑되어 계약이 같았습니다. 역양자화가 호스트에서 일어나므로 MLIR digest도 이를 보지 못했습니다. 따라서 Gemma3n의 역양자화기를 바꿔도 캐시된 모든 아티팩트의 정체성은 그대로였을 것입니다.

커밋 `5c0bf520`은 `SeparateMultiplyThenAddBf16Rounded`와 `AffineExecution::HostBf16Sequential`을 추가하고 상수 `GEMMA3N_AUDIO_AFFINE` 하나를 도입합니다. `decode_weight`는 이 상수로 역양자화기를 고르고, 계약도 같은 값으로 observer를 resolve합니다. 그래서 반올림하는 코드와 반올림을 기록하는 코드가 어긋날 수 없습니다. `host_dequantization_rounding_is_part_of_the_identity`는 두 호스트 모드가 서로 다른 정체성을 만드는지 확인합니다.

### 3.4 스키마별 캐시 디렉터리

이 변경은 운영 경로에서 v3 매니페스트를 쓰는 첫 변경입니다. `reject_manifest_downgrade`는 더 새로운 스키마의 매니페스트를 이전 스키마로 바꾸기를 거부합니다. 의도된 정책이지만, 모든 보조 캐시는 고정된 `std::env::temp_dir()` 경로에 있습니다. 같은 장비의 이전 바이너리(공유 self-hosted runner의 rebase되지 않은 브랜치, 개발 빌드 옆의 릴리스 빌드)는 v3 매니페스트를 만나 `refusing to replace auxiliary artifact manifest ... from schema v3 with v1` 오류로 vision 또는 audio 로드에 실패했을 것입니다.

| 선택지 | 장점 | 단점 |
|---|---|---|
| 공유 경로 유지, "캐시를 지우라"고 문서화 | 코드 변경 없음 | 누군가 `/tmp`를 지울 때까지 공유 runner에서 하드 실패 |
| downgrade 거부 완화 | 이전 바이너리가 재컴파일 | 두 바이너리를 번갈아 실행할 때마다 서로의 항목을 덮어쓰고, 거부 장치가 아무것도 지키지 못함 |
| **선택: 스키마 이름의 하위 디렉터리** | 스키마마다 자기 항목을 가지고, 한 스키마 안에서는 거부 장치가 여전히 동작 | 업그레이드 후 컴파일 한 번 추가, 이전 v1 항목은 디스크에 남음 |

이제 모든 운영 호출 지점이 `qualify_auxiliary_artifact` 하나를 씁니다. v1이 아닌 계약에 대해 `<cache>/mlxcel-xla-aux-artifact-v3-operator-numeric/<file>`을 계산하고, 디렉터리를 만들고, 그 위치에서 `ensure_qualified_auxiliary_artifact`를 실행한 뒤 `IreeAuxiliaryModule::load`에 넘겨야 할 경로를 반환합니다. 호출하는 쪽마다 경로를 다시 계산하지 않고 반환값을 쓰게 해서 qualify 경로와 load 경로가 갈라지지 않게 했습니다. v1 계약은 원래 경로를 유지하므로 `aux_smoke`와 downgrade 테스트는 영향받지 않습니다.

### 3.5 런타임 digest를 빌드 시점에 기록

CPU 프로브는 실행 시점에 IREE dist root 아래의 `lib/libiree_runtime_unified.a`를 해시했습니다. CUDA, ROCm, macOS 빌드는 `build/runtime/src/iree/runtime/`의 소스 빌드 런타임을 링크하므로 읽을 dist 아카이브가 없고, 배포된 바이너리는 빌드 트리를 아예 못 볼 수도 있습니다. 이제 `build.rs`가 링크하는 아카이브를 해시해 컴파일 시점 환경 변수로 넣습니다. `runtime_archive_identity`는 이 값을 우선 쓰고, 빌드가 아카이브를 읽지 못한 경우에만 dist 레이아웃으로 돌아갑니다. 리뷰에서 아카이브가 없을 때도 `rerun-if-changed`를 내보내 아카이브가 생길 때까지 build script가 매번 다시 실행된다는 지적이 있었습니다. digest가 저절로 복구되도록 그대로 두었고, 링크하는 모든 레이아웃은 이미 아카이브를 요구합니다.

### 3.6 정체성이지 검증이 아님

`BackendNumericIdentity`에는 algorithm 필드가 있는데, IREE는 컴파일된 VMFB의 디스패치별 커널 정체성을 노출하지 않습니다. 모든 운영 항목은 `unobserved-iree-dispatch-selection`과 emitter revision `mlxcel-xla-stablehlo-emitter-v1`을 기록합니다. 이렇게 해서 #932가 구분하라고 한 세 가지 reference 주장이 섞이지 않습니다. 이 PR은 아티팩트가 계약이 설명하는 바로 그 아티팩트임을 보장할 뿐이고, 그 계약이 운영 MLX와 일치하는지는 #3212의 몫입니다.

### 3.7 아티팩트당 컴파일러 실행 한 번

계약을 만들 때와 generation identity를 만들 때 모두 `iree-compile --version`이 필요합니다. 공유하지 않으면 아티팩트마다 컴파일러를 두 번 실행했고, 리뷰에서 아티팩트당 약 45 ms의 추가 로드 시간으로 측정됐습니다. `compiler_version_output`은 경로, 크기, 수정 시각을 키로 trim된 stdout을 memoize하므로, 컴파일러가 교체되면 다시 조회합니다. generation identity 문자열은 바이트 단위로 이전과 같아서, 이 리팩터링만으로 v1 항목이 무효화되지는 않았습니다.

## 4. 검증

단위 테스트와 빌드 검사(브랜치 최종 상태):

- `cargo test -p mlxcel-xla --lib --profile test-fast`: 435개 통과(첫 커밋 시점 433개: 기준 422개와 신규 11개).
- runtime, manifest, contract, probe 모듈 대상 `--features iree,micro-oracle`: 105개 통과.
- feature 없음, `iree`, `iree,micro-oracle,diagnostics` 각각에서 `cargo clippy -p mlxcel-xla --lib --tests -D warnings`, `cargo fmt --check`, `cargo check --features cuda,xla-iree --all-targets`, `cargo check --no-default-features --features xla-diagnostics --all-targets`.

오래된 캐시 테스트는 필드 분류(operator dtype, accumulator, association, precision, compiler, runtime, device)를 하나씩 바꾸고 캐시된 VMFB가 거부되어 재컴파일되는지 확인합니다. 패밀리 고정 테스트는 기본 CUDA와 CPU 정밀도에서 각 패밀리의 연산자 목록을 고정하고, 연산자, cast, association 정책이 제거되면 실패합니다.

GB10에서 실제 체크포인트, `MLXCEL_BACKEND=xla`, `gpu-lock` 아래, `-n 24 --temp 0`. main 빌드를 먼저 실행하고, 브랜치 빌드로 두 번 로드했습니다. 첫 번째 로드는 v1 매니페스트를 v3로 교체했고, 두 번째 로드는 이를 재사용했습니다(VMFB와 매니페스트의 mtime 변화 없음).

| 체크포인트 | 런타임 | 디바이스 | 3회 실행의 greedy 출력 |
|---|---|---|---|
| llava-interleave-qwen-0.5b-bf16 | `vision_runtime` | cuda | 동일 |
| qwen2-vl-2b-instruct-4bit | `qwen2_vl_runtime` (호스트 역양자화 Q4) | cuda | 동일 |
| qwen3-vl-2b-instruct-4bit | `qwen3_vl_runtime` | cuda | 동일 |
| molmo-7b-d-0924-4bit | `molmo_vision_runtime` (그래프 내 Q4, `MLXCEL_XLA_CONTEXT_CAPACITY=512`) | cuda | 동일 |
| molmo2-4b | `molmo2_vision_runtime` | cuda | 동일 |
| youtu-vl-4b-instruct | `youtu_vl_runtime` | cuda | 동일 (5절 참조) |
| gemma-3n-e4b-it-4bit | `gemma3n_audio_runtime` | local-task | 동일 |

공존 검증: qwen3-vl-2b-instruct-4bit에서 main과 브랜치 바이너리를 같은 캐시 root에 번갈아 실행했습니다. 각자 자기 항목(이전 경로의 v1, 스키마 하위 디렉터리의 v3)을 재사용했고, 어느 쪽도 downgrade 거부에 걸리지 않았으며, greedy 출력도 동일했습니다.

## 5. 미검증 및 남은 항목

- **Phi4MM.** 이 장비에 체크포인트가 없어 단위 고정 테스트로만 확인했습니다.
- **CUDA에서의 Gemma3n audio.** main에서도 Gemma3n의 텍스트 decode 그래프가 CUDA에서 `iree-compile`에 실패하므로(`failed to distribute`, e2b와 e4b 모두) audio 런타임은 `local-task`에서만 검증했습니다.
- **main의 기존 실패로 막힌 계획 실행.** qwen2.5-vl-3b는 16410 토큰 프롬프트의 텍스트 prefill에서 메모리가 부족합니다. llava-1.5-7b-4bit는 IREE vision에서 config가 거부됩니다. gemma-3-4b는 요청 크기 기반 context capacity에 걸립니다. 이 PR이 바꾼 경로는 아니지만, Qwen2.5-VL과 Gemma3 projector 계약은 실제 로드로 확인되지 않았습니다.
- **Youtu-VL 출력 품질.** 변경 전후 모두 `D`를 반복 출력합니다. 바이트 동일성은 이 PR이 출력을 바꾸지 않았다는 뜻이지, 출력이 올바르다는 뜻은 아닙니다.
- **선언과 그래프의 일치.** 패밀리 목록은 손으로 작성합니다. 고정 테스트는 항목 제거나 정책 변경을 잡지만, 목록을 emitter가 쓰는 MLIR과 대조하는 장치는 없습니다. emitter에 새 연산자 분류를 추가하고 선언을 빠뜨리면 그 연산자는 MLIR digest로만 묶입니다.
- **아티팩트당 부동소수 분류 하나.** observer는 가장 많이 나온 부동소수 dtype을 기록합니다(동률이면 더 넓은 타입). 혼합 체크포인트의 텐서별 dtype은 config identity의 기존 `checkpoint_schema_sha256`으로만 묶입니다.
- **고정 테스트가 유도 필드를 생략.** 패밀리 고정 테스트는 `round=`와 `dequant=`를 적지 않습니다. 이 필드는 고정된 필드에서 유도되고 필드 분류별 거부 테스트가 다룹니다(리뷰 LOW, 유지).
- **이전 v1 항목은 지워지지 않음.** 업그레이드 후 캐시를 지울 때까지 각 보조 아티팩트가 `/tmp`에 두 벌 남습니다.
- **후속 작업.** #3212는 CUDA에서 MLX를 상대로 연산자 micro-oracle을 실행하고, `MLX_ENABLE_TF32=0`을 고정하고, #923 이후 Qwen VL Q4 근거를 새로 만듭니다. `iree_identity.rs`를 재사용하므로 프로브 보고서가 운영 아티팩트와 같은 계약 정체성을 갖습니다. 이 PR의 보안 리뷰에서 나온 #3248은 모든 XLA 캐시를 사용자 전용으로 만듭니다. 이 PR의 스키마 하위 디렉터리를 포함해 모든 캐시가 소유권 확인 없이 누구나 쓸 수 있는 `/tmp`에 있고, 매니페스트의 모든 필드는 공개된 입력으로 계산할 수 있습니다. 매니페스트는 그것이 어떤 아티팩트인지를 증명할 뿐 누가 썼는지는 증명하지 않습니다. #932의 나머지 기준(공용 emitter 헬퍼, CUDA oracle, 패밀리 게이트)은 열려 있습니다.

## 6. 학습 포인트

- 입력이 호스트 쪽 변환을 거쳐 컴파일된 모듈에 들어가면 MLIR digest는 그 변환을 볼 수 없습니다. 그 변환을 정체성에 명시적으로 묶고, 구현과 정체성을 같은 값으로 고르게 하십시오.
- 계약 필드를 emitter 옆에 선언하는 것보다 emitter가 소비한 값에서 유도하는 편이 강합니다. 어긋난 상태를 아예 쓸 수 없기 때문입니다.
- 바이너리 하나에게는 옳은 downgrade 거부도, 두 세대의 바이너리가 캐시 경로를 공유하면 장애가 됩니다. 운영에서 새 스키마를 켜기 전에 위치를 스키마별로 나누십시오.
- 관찰할 수 없는 필드는 생략하지 말고 관찰 불가로 기록하십시오. 나중의 검증 단계가 채울 자리가 남고, 읽는 사람이 정체성을 검증으로 오해하지 않습니다.

## 관련 커밋

| 해시 | 유형 | 메시지 |
|---|---|---|
| `666ef90b` | feat | bind operator numeric contracts into production artifacts |
| `5c0bf520` | fix | bind Gemma3n BF16 host dequant rounding into its contract |
| `b322a139` | fix | scope v3 auxiliary caches by schema and spawn version once |
| `3bce42a8` | docs | tidy the v3 auxiliary cache paragraph in environment-variables |
