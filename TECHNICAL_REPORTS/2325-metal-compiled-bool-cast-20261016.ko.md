# PR #2325: Metal 컴파일 커널에서 비교 입력을 bool로 캐스팅하지 않도록 수정

**날짜**: 2026-10-16
**상태**: 구현 완료, Metal 되돌림 검사와 수정 후 검증은 self-hosted macOS 러너에서, 실제 체크포인트는 머지 전 CUDA(GB10)에서 검증, 오버레이를 제거할 수 있는지는 미해결 ("미검증 및 후속" 참고)
**위험도**: 낮음 (Metal 전용 오버레이의 bool 출력 분기 하나, bool이 아닌 연산이 내는 소스는 그대로, 브리지 항목은 테스트 전용)

## 요약

이슈 #1792는 Metal에서 컴파일(융합)된 비교 연산이 틀린 결과를 내는 문제를 고친다.

1. MLX `backend/metal/compiled.cpp`의 mlxcel 오버레이에서, bool을 출력하는 연산(비교, `isnan`, `isinf`)은 입력 타입이 다르면 `bool`이 아니라 promote된 입력 타입으로 캐스팅한다. 오버레이 헤더는 기준 upstream을 243bdf09로 기록한다.
2. 테스트 전용 브리지 항목 `compile_dtype_probe_for_test`(`src/lib/mlxcel-core/cpp/mlx_cxx_compile_probe.cpp`)가 고정된 그래프 7개 중 하나를 eager, 컴파일, shapeless 컴파일로 실행한다.
3. `src/lib/mlxcel-core/src/compiled_dtype_cast_tests.rs`(테스트 5개)가 두 컴파일 모드가 eager와 같은지 검증한다.

## 문제 정의

오버레이는 타입이 연산의 출력 타입과 다른 모든 입력을 그 타입으로 캐스팅한다. bf16/f32 혼합 산술을 위해 추가된 것이다(1923da32). 비교의 출력 타입은 `bool`이므로 float 피연산자가 비교 전에 `bool`로 캐스팅됐다. 컴파일된 `less(0.2, 0.7)`은 `Less(true, true)`가 되어 false였고, `isnan`은 NaN을 그대로 통과시켰다. 컴파일된 `where(a < b, a, b)`는 `[0.2, -0.5, 0.2, 0, 3]` 대신 `[0.7, 0.3, 0.2, 0, 3]`을 반환했고, min-p 필터(#1379)는 입력을 걸러내지 않고 돌려줬다. CUDA에는 이런 오버레이가 없어 영향이 없다.

## 기술적 선택과 그 이유

- 출력이 `bool`일 때만 `bool_`에서 시작해 연산의 입력들에 `promote_types`를 접어 적용한다. bool이 아닌 연산은 기존 캐스팅을 유지하므로, 캐스팅이 추가된 원래 목적인 bf16/f32 산술의 소스는 바뀌지 않는다.
- bool이 아닌 경로는 건드리지 않는다. 모든 연산을 promote된 입력 타입으로 캐스팅하면 `Real`/`Imag`도 고쳐지지만, 원래 캐스팅이 지키는 경로가 바뀌고 이 호스트에서는 Metal을 테스트할 수 없다.
- Rust에서 그래프를 만들지 않고 C++ 브리지 probe의 고정 그래프로 테스트한다. eager, 컴파일, shapeless 컴파일이 같은 그래프를 실행하므로 차이를 컴파일 탓으로 돌릴 수 있다.
- 같은 dtype f32의 eager 결과는 호스트 산술에 고정한다. eager가 틀려서 비교가 우연히 통과하는 일을 막기 위해서다.
- 오래된 "Synced to 81ba1c6a" 헤더를 바로잡는다. upstream 243bdf09와 비교하면 오버레이는 헤더와 캐스팅 블록에서만 다르다.

## 변경 요약

- `src/lib/mlxcelverse/overlays/metal/mlx/backend/metal/compiled.cpp`: 캐스팅의 bool 출력 분기(약 10줄)와 헤더, `overlays/metal/README.md`에 규칙 설명.
- `src/lib/mlxcel-core/cpp/mlx_cxx_compile_probe.cpp`(신규), 그리고 `compile_dtype_probe_for_test`를 위한 `mlx_cxx_bridge.cpp`, `mlx_cxx_bridge.h`, `build.rs`, `lib.rs` 연결.
- `src/lib/mlxcel-core/src/compiled_dtype_cast_tests.rs`(신규): f32 비교 6종, f16/f32, f32/f16, bf16/f32 혼합 비교, f32/f16/bf16의 `where(isnan(x), 7, x)`와 `where(isinf(x), 9, x)`, min-p 그래프, bf16/f32 혼합 산술 체인.

## 동작 변경

Metal에서 컴파일된 비교, `isnan`/`isinf`, 이를 쓰는 그래프(min-p)가 eager와 같은 결과를 낸다. CUDA는 바뀌지 않고, 테스트 브리지 밖에 추가된 API는 없다.

## 검증

- 수정 없는 테스트 전용 브랜치에서 Nightly verify로 Metal 되돌림 검사: 실행 38780888405가 정확히 4개 테스트(같은 dtype f32 비교, f16/f32 혼합 비교, isnan/isinf, min-p)에서 실패했고 2093개는 통과했다. bf16/f32 혼합 산술 테스트는 예상대로 통과했다.
- 수정 포함: 실행 38780921019에서 5개 테스트와 전체 스위트가 통과했다.
- CUDA는 오버레이를 쓰지 않으므로 5개 테스트가 수정 유무와 관계없이 통과한다. 따라서 되돌림 검사는 Metal에서 해야 한다.
- 실제 체크포인트 `qwen3-0.6b-4bit`, 이 브랜치로 CUDA에서 `generate`: "The capital of France is Paris."로 끝나는 일관된 출력.
- fmt, clippy, 계약 검사가 통과한다.

## 미검증 및 후속

- 미해결 질문: 243bdf09에서 오버레이를 완전히 제거할 수 있는가. 캐스팅은 macOS 26.4에서 융합 커널의 `Divide(bfloat16_t, float)` 템플릿 추론이 실패했기 때문에 존재한다. upstream이 이제 primitive를 만들기 전에 이항 입력을 promote한다면 융합 연산은 혼합 dtype을 보지 않는다. 확인하려면 오버레이 없는 Metal 빌드와 1923da32의 모델(gemma3-4b, nemotron-h-30b)이 필요하다.
- 후속 후보: `Real`과 `Imag`(complex64 입력, f32 출력)는 Metal에서 여전히 입력이 float로 캐스팅되어 컴파일된 `imag()`는 0을 반환한다. 이를 쓰는 컴파일 지점은 없다.
- Metal 수정은 macOS 러너의 테스트 스위트로만 검증했고, Metal에서 실제 체크포인트로 검증하지는 않았다.
