# PR #2943: qwen4_exp arena zero-copy 테스트를 macOS 전용으로 제한

**날짜**: 2026-11-10
**상태**: 완료 (draft PR 열림, #2939 종료)
**언어**: Rust
**위험도**: 낮음

## 요약

PR #2943은 #2912 이후 CUDA에서 항상 실패하던 테스트를 고칩니다. `models::qwen4_exp::stream::arena::tests::a_write_into_a_slot_is_seen_through_the_wrapped_array`는 MLX가 페이지 정렬된 expert plane을 복사하지 않고 감싼다고 단언하는데, 이는 Metal 할당자만 할 수 있습니다. 이제 이 테스트 모듈은 macOS에서만 컴파일되고, macOS가 아닌 플랫폼에는 이 제한이 안전한 근거를 고정하는 새 테스트가 추가되었습니다. 근거는 `ExpertStream::resolve`가 Metal이 아닌 환경에서 스트리밍을 거부한다는 점입니다. 프로덕션 코드는 바뀌지 않았습니다. GB10(CUDA)에서 `qwen4_exp::stream` 테스트 33개가 모두 통과합니다.

## 1. 문제 정의

`from_bytes_nocopy`(`src/lib/mlxcel-core/cpp/mlx_cxx_bridge.cpp`)는 MLX의 `array(void*, Shape, Dtype, deleter)` 생성자를 호출합니다. 이 생성자는 `allocator::make_buffer`에 포인터를 넘겨받게 하고, `nullptr`이 돌아오면 malloc 후 복사로 대신합니다. Metal은 `make_buffer`를 `newBuffer(ptr, ...)`로 구현하지만 CUDA 할당자는 이를 재정의하지 않으므로 기본 구현이 `nullptr`을 돌려주고, CUDA에서는 감싸기가 항상 복사가 됩니다. arena 테스트는 `array_data_address`와 plane 포인터를 비교하므로 CUDA에서는 매번 실패합니다(단독 실행 3회 중 3회, 11,641개 lib 테스트 실행에서 유일한 실패). 메인 CI는 CUDA lib 스위트를 돌리지 않아 #2912가 실패 상태로 병합되었습니다.

프로덕션에는 영향이 없었습니다. 대상이 macOS가 아니면 `ExpertStream::resolve`는 `MLXCEL_EXPERT_STREAM=1`에 오류를, `auto`에 `Ok(None)`을 돌려주고, `LayerStream::new`는 MLX가 복사한 plane을 거부하므로 CUDA는 detached arena를 만들지 않습니다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| `src/models/qwen4_exp_stream_arena.rs` | 테스트 모듈을 `#[cfg(all(test, target_os = "macos"))]`로 제한하고 이유를 주석으로 남김 |
| `src/models/qwen4_exp_stream_tests.rs` | `#[cfg(not(target_os = "macos"))]` 테스트 `resolve_refuses_streaming_off_metal`과 환경 변수를 복원하는 drop guard 추가 |

## 3. 기술적 선택과 그 이유

### 3.1 브리지가 아니라 테스트를 제한

CUDA에서 `from_bytes_nocopy`를 zero-copy로 만들려면 고정된 MLX 버전의 CUDA 할당자에 `make_buffer`를 구현해야 하는데, 이는 upstream 작업이고 스트리밍이 Metal 전용인 동안 CUDA에서 쓸 곳도 없습니다. 이슈가 브리지 변경을 범위 밖으로 정했으므로, 테스트의 플랫폼을 기능의 플랫폼에 맞췄습니다.

### 3.2 모듈 전체를 제한

arena의 `mod tests`에는 이 테스트 하나와 `use super::*;`만 있습니다. 함수만 제한하면 Linux에서 import가 사용되지 않은 채 남고, clippy는 `-D warnings`로 돌기 때문에 실패합니다. 모듈 단위 cfg로 블록 전체를 컴파일에서 뺐습니다.

### 3.3 잃은 Linux 검증을 거부 동작 자체로 대체

제한으로 Linux의 유일한 arena 검증이 사라지므로, 새 테스트는 제한이 기대는 불변식을 단언합니다. `MLXCEL_EXPERT_STREAM=1`이면 `resolve`가 "Metal only"를 포함한 오류를, `auto`면 `Ok(None)`을 돌려줘야 합니다. 나중에 누군가 CUDA에서 스트리밍을 켜면 이 테스트가 실패하면서 그 플랫폼에서도 돌아야 할 arena 테스트를 가리키게 됩니다. 플랫폼 검사가 파일을 열기 전에 실행되므로 일부러 존재하지 않는 모델 디렉터리를 넘깁니다.

### 3.4 환경 변수 변경 규율

테스트는 crate의 모든 환경 변수 변경 테스트처럼 `crate::test_support::env_lock::env_lock()`을 실행 내내 잡습니다. 락 다음에 선언한 drop guard가 `var_os`로 이전 값을 저장했다가 복원하거나 제거합니다. 지역 변수는 선언의 역순으로 drop되므로 복원이 락 해제보다 먼저 일어나고, panic이 나도 실행됩니다. `set_var`는 edition 2024 요구대로 SAFETY 주석과 함께 `unsafe`로 감쌌습니다.

## 4. 검증

- `gpu-lock run --tag issue-2939 -- cargo test --profile test-fast --features cuda -p mlxcel --lib qwen4_exp::stream -- --test-threads=1`: spark-102(GB10)에서 33개 통과, 0개 실패. 새 테스트가 포함되어 있고 arena 테스트는 Linux에서 더 이상 컴파일되지 않습니다.
- `cargo clippy -p mlxcel --lib --tests --profile test-fast --features cuda -- -D warnings`, `cargo fmt --all -- --check`, `scripts/ci/check_env_var_docs.py`, `scripts/insert_apache_header.py --check`: 모두 통과.

## 5. 검증하지 못한 부분

이 호스트에는 macOS 타깃이 설치되어 있지 않아 macOS cfg 경로는 컴파일하지 못했습니다. cfg를 읽어 보면 macOS에서는 arena 모듈이 이전과 똑같이 보이고, 새 테스트와 guard는 보이지 않으므로 사용되지 않는 항목이 남지 않습니다. 이를 실제로 확인하는 첫 빌드는 macOS CI 작업입니다.

## 6. 학습 포인트

- MLX의 no-copy 배열 생성자는 `allocator::make_buffer`에 의존하며, 고정된 MLX 버전에서는 Metal만 이를 구현합니다. zero-copy 감싸기를 단언하는 테스트는 어느 모듈에 있든 Metal 테스트입니다.
- 어떤 플랫폼에서 테스트를 제한할 때는, 그 플랫폼이 해당 코드 경로에 들어가지 못하게 막는 검사를 그 플랫폼에서 테스트로 추가해야 제한이 이후의 동작 변화를 가리지 않습니다.
