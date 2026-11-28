# 기술 보고서: PR #1578 - perf(server): pack structured-output masks as u32 bitmasks

**작성일**: 2026-09-02
**작성자**: mlxcel maintainers
**리뷰어**: 없음
**상태**: 완료
**언어**: Rust
**위험도**: Medium

---

## 요약

제약 디코딩은 문법 엔진이 내놓은 패킹된 토큰 비트마스크를 어휘 폭만큼의 `Vec<bool>`로 펼치고, 로짓 축을 한 번 더 훑어 `Vec<f32>` 바이어스를 만든 뒤 그것을 업로드해 로짓에 더했다. 제약이 걸린 시퀀스마다, 토큰마다, 포워드 패스와 샘플링 사이의 스케줄러 스레드에서 매번 벌어지는 일이었다. 이 PR은 마스크를 패킹된 채로 디바이스까지 보내고 브로드캐스트 시프트로 그곳에서 펼친다. Qwen3.8-27B 형상에서 마스크를 준비하고 업로드하는 스케줄러 스레드 비용은 스텝당 711us에서 7.4us로, 스텝당 호스트-디바이스 복사는 970KiB에서 30KiB로 줄었다. 출력 배열은 그대로이며, 대체된 구현과 원소 단위로 대조해 확인했다.

---

## 1. 문제 정의

### 1.1 배경

OpenAI 형식의 `response_format: {"type": "json_schema", ...}` 요청은 스키마를 `llguidance` 문법으로 컴파일하고 시퀀스에 `StructuredOutputConstraint`를 붙인다. 매 토큰을 샘플링하기 전에 스케줄러는 부분 출력이 스키마를 유지하려면 어떤 토큰 id가 허용되는지 매처에 묻고, 나머지 로짓을 전부 음의 무한대로 마스킹한다. 샘플러가 스키마를 깨는 토큰을 뽑을 수 없게 만드는 장치다.

`llguidance`는 그 답을 자신이 계산한 형태 그대로 돌려준다. `SimpleVob`, 즉 토큰 id 하나당 1비트, `u32` 워드 하나당 32개 id가 들어가는 비트셋이다. 마스크 적용부의 일은 그 답을 MLX가 로짓 행과 결합할 수 있는 형태로 GPU에 올리는 것이었다.

### 1.2 기존 문제

- **비트셋을 호스트에서 두 번 펼쳤다.** `compute_mask`가 `mask_buf`를 `vocab_size`로 resize한 뒤 `iter_set_entries`로 허용된 id마다 `bool` 하나를 썼다. 이어서 `apply_structured_mask_to_logits`가 `bias_buf`를 `vocab_size_hint`로 resize하고 한 칸씩 훑으며 허용 위치에는 `0.0`을, 나머지에는 `f32::NEG_INFINITY`를 채웠다. 248320행 헤드라면 대략 25만 회짜리 루프 두 개를, 토큰마다, 제약 시퀀스마다 돈다는 뜻이다.
- **업로드가 실어 나르는 정보량의 32배였다.** 값이 `0.0`과 `-inf` 둘뿐인 f32 바이어스는 토큰당 1비트짜리 정보를 32비트에 담은 것이다. 248320행에서는 스텝당 970KiB 복사이고, 패킹 형태라면 30KiB면 된다.
- **전부 스케줄러 스레드에서 벌어졌다.** `apply_structured_mask`는 `decode_tick`과 `prefill`에서 포워드 패스와 샘플링 사이에, 틱마다 제약 시퀀스 수만큼 호출된다. 동시 제약 시퀀스가 B개면 틱당 B번 지불되고, 배치 자체의 진행과 직렬화된다.
- **`mask_buf` / `bias_buf` 재사용은 할당 비용만 없앴다.** 두 버퍼를 생성 시점에 미리 잡아두면 토큰마다 새 `Vec`을 만드는 일은 피하지만, 두 번의 순회도 복사도 그대로다. 게다가 그 어휘 크기에서 제약 시퀀스당 약 1MB의 호스트 메모리를 상주시킨다.

### 1.3 위험 평가

| 위험 | 영향 | 가능성 |
|------|------|--------|
| 어휘가 커질수록 제약 디코딩 처리량이 나빠지는데 상한이 보이지 않음 | Medium | High |
| 비용이 동시 제약 시퀀스 수에 비례하므로, 부하가 걸릴 때 구조화 출력이 정확히 그만큼 나빠짐 | Medium | High |
| 마스크 경로를 갈아엎으면서 샘플러가 고르는 토큰이 조용히 바뀜 | High | Low |
| 갈아엎는 과정에서 패딩 헤드 마스킹 규칙이 빠져 이름 없는 행이 샘플링됨 | High | Low |

뒤의 둘은 이 변경 자체가 들여오는 위험이며, 2절과 3절이 그것을 어떻게 묶어뒀는지 설명한다.

---

## 2. 기술 검토

### 2.1 보안

마스크는 제약 요청과 비적합 출력 사이에 서 있는 유일한 장치다. 그러니 너무 관대한 마스크는 성능 문제가 아니라 정확성과 신뢰의 문제다. 재작성에서 반드시 살아남아야 했던 규칙이 둘이고, 지금은 둘 다 전용 테스트가 붙어 있다.

- 매처 어휘를 넘어선 로짓 행은 마스킹된 채로 남아야 한다. Qwen3.5와 Qwen3.8은 `lm_head`를 토크나이저보다 넓게 패딩한다. 토크나이저 항목 248077개에 대해 248320행이다. 그 243행은 어떤 토큰도 이름 붙이지 않으므로 문법이 본 적조차 없고 결코 만족시킬 수 없다. 패킹 형태에서는 0인 워드를 읽으므로 구조적으로 비허용이 된다.
- 매처의 비트셋에는 잉여 비트가 있다. `SimpleVob`은 `ceil(size / 32)` 워드를 저장하므로 어휘 248077개면 마지막 워드에 어떤 토큰도 가리키지 않는 19비트가 남는다. `toktrie`는 일부 경로(`set_all`, `negated`)에서 이를 지우지만 API 경계에서 보장되는 불변식은 아니다. 그래서 `pack_mask_words`는 그것을 믿는 대신 마지막 부분 워드를 유효 비트 범위로 잘라낸다.

입력 파싱, 인증, 로깅 표면은 바뀌지 않았다. 빈 마스크에 대한 공개 에러 메시지는 이전과 바이트 단위로 동일하다.

### 2.2 성능

이 변경의 목적 자체다. 측정치는 8절과 부록 B에 있다. 요지는 호스트 작업이 사라졌고, 복사가 1/32로 줄었으며, 디바이스 쪽 확장은 어휘 길이에 대한 원소별 연산 네 개로 선택 연산과 같은 그래프에 융합된다는 것이다.

### 2.3 호환성 및 의존성

새 의존성은 없다. 사용한 MLX 연산(`from_slice_u32`, `right_shift`, `bitwise_and`, `reshape`, `slice`, `astype`, `where_cond`)은 전부 이미 cxx 브리지에 바인딩돼 있어 `src/lib/mlxcel-core`는 손대지 않았고, CUDA 커널을 추가하지 않으므로 `make verify-kernel-dtype-keys`는 자명하게 통과한다.

`apply_structured_mask_to_logits`는 시그니처를 그대로 유지하므로 `src/server/batch/scheduler/run_loop.rs`의 호출부와 `decode_tick.rs`, `prefill.rs`의 세 호출 지점은 변경이 없다. `compute_mask`는 `Vec<bool>` 접근자로 남겼다. `tests/structured_outputs.rs`의 아홉 개 호출부가 이것에 의존하고, 개별 항목을 들여다보는 읽기 좋은 방법이기도 하다. 다만 이제 핫 패스에는 없다.

### 2.4 코드 품질

비트 연산은 `pack_mask_words`로 분리했다. 매처도 디바이스도 끼지 않는 슬라이스 위의 순수 함수이며, 무작위 등가성 테스트가 애초에 가능해진 이유가 이것이다. 디바이스 확장은 역시 독립 함수인 `expand_packed_mask`다. 그 결과 `apply_structured_mask_to_logits`는 정책 선언처럼 읽을 만큼 짧아졌다. 게이트 확인, 패킹, 공집합 확인, 선택.

`cargo clippy --profile test-fast --lib --tests --features metal,accelerate -- -D warnings`와 `cargo fmt --all -- --check`는 깨끗하다.

---

## 3. 기술적 선택과 그 이유

### 3.1 토큰별 인덱스 테이블을 gather하는 대신 비트 위치를 브로드캐스트

이슈 #1316은 제약마다 디바이스 배열 두 개를 미리 만들어 두자고 했다. `i >> 5`를 담는 `word_idx = i32[V]`와 `i & 31`을 담는 `bit_idx = u32[V]`를 두고 `bitwise_and(right_shift(take(words, word_idx, 0), bit_idx), 1)`로 펼치는 안이다.

구현은 대신 브로드캐스트한다. 패킹된 워드는 `u32[n_words, 1]` 열로, 32개 비트 위치는 `u32[1, 32]` 행으로 올라가므로 `right_shift` 한 번이 `[n_words, 32]`로 브로드캐스트되고 원소 `(w, b)`가 워드 `w`의 비트 `b`가 된다. 행 우선으로 보면 그 평탄 인덱스는 `w * 32 + b`, 정확히 토큰 id다. `[1, n_words * 32]`로 reshape하고 로짓 폭으로 잘라내면 토큰 순서의 마스크가 그대로 복원된다.

계획 대신 이 형태를 고른 이유는 이렇다.

- **제약별 인덱스 테이블이 없다.** gather 설계는 제약 시퀀스마다 `2 * 4 * V` 바이트, 248320이면 약 2MB의 디바이스 메모리를 제약의 수명 내내 붙들고 있어야 한다. 브로드캐스트에 필요한 것은 32개짜리 행 하나다.
- **gather가 없다.** `V`개 인덱스에 대한 `take`는 시프트가 시작되기도 전에 `V`개짜리 인덱스 배열을 읽고 `V`개짜리 결과를 쓴다. 브로드캐스트는 `n_words`개 워드만 읽으니 1/32이다.
- **폭에 묶인 캐시 상태가 없다.** 계획의 배열들은 `vocab_size_hint`나 로짓 dtype이 바뀔 때 다시 만들어야 했고, 그래서 검증 목록에 `packed_apply_rebuilds_index_arrays_on_width_change`가 들어 있었다. 여기서는 폭에 의존하는 것이 아무것도 없으니 그 실패 양상 자체가 존재하지 않는다. 해당 테스트는 버퍼 하나에 여섯 개 폭을 흘려보내며 각각을 확인하는 `packed_apply_handles_a_width_change_between_calls`가 되었다.
- **MLX 자신의 관용구다.** `dequantize`는 2의 거듭제곱이 아닌 비트폭의 양자화 가중치를 문자 그대로 `bitwise_and(right_shift(w, arange(32, uint32)), 1)`로 푼다. 양자화 포워드마다 MLX가 돌리는 형태를 재사용하는 편이 새로운 형태를 만드는 것보다 안전한 내기다.

### 3.2 디바이스 상수를 제약 객체에 캐시하지 않기

계획은 `neg_inf`와 인덱스 배열을 `StructuredOutputConstraint`에 `Option<UniquePtr<MlxArray>>`로 두자고도 했다. 그것은 컴파일되지 않으며, 구조체만 읽어서는 드러나지 않는 이유이므로 기록해 둘 값어치가 있다.

`StructuredOutputConstraint`는 `SequenceInfo` 안에 `Arc<Mutex<StructuredOutputConstraint>>`로 살면서 스케줄러 스레드로 옮겨지므로 `Send`여야 한다. cxx는 불투명 `extern "C++"` 타입에 `Send`를 자동으로 붙이지 않고 `mlxcel-core`에도 수동 impl이 없으므로, `UniquePtr<MlxArray>`는 `Send`가 아니다.

```
error[E0277]: `*const cxx::void` cannot be sent between threads safely
   = help: within `MlxArray`, the trait `Send` is not implemented for `*const cxx::void`
   = note: required for `UniquePtr<MlxArray>` to implement `Send`
```

그런 필드를 넣으려면 `src/server/prompt_cache/entry.rs`가 `DetachedKvSetHolder`에 쓰는 식의 `unsafe impl Send` 래퍼와, MLX 배열 핸들이 스레드를 넘나드는 데 대한 실제 안전성 논증이 필요했을 것이다. 이 설계가 필요로 하는 상수 셋(비트 위치 32개, 스칼라 `1`, 스칼라 `-inf`)은 합쳐서 152바이트라, 그냥 호출마다 만든다. 8절의 측정은 그 세 번의 생성을 포함한 준비-업로드 구간 전체를 7.4us로 잰 것이므로, 이 결정은 관측 가능한 비용이 없다.

### 3.3 바이어스를 더하는 대신 `where_cond`로 선택

이전 코드는 `0.0`과 `-inf`로 채운 f32 배열을 더했다. 새 코드는 로짓과 스칼라 `-inf` 사이에서 고른다. 출력이 단지 동등한 것이 아니라 동일한 이유는 이렇다.

- `mlx::core::where`는 `promote_types(b.dtype(), c.dtype())`를 계산해 두 피연산자를 그 타입으로 캐스팅한다. `add`가 하던 것과 똑같다. 따라서 f16이나 bf16 로짓을 f32 `-inf`와 짝지으면 f32 바이어스가 내던 것과 같은 f32 출력이 나온다. 어떤 로짓 정밀도에서도 dtype 거동은 바뀌지 않았다.
- `where`는 `add`가 쓰던 것과 같은 규칙으로 `broadcast_arrays`를 통해 브로드캐스트하므로, prefill의 `[1, V]`와 decode 틱의 `[1, 1, V]` 둘 다 이전과 같은 shape로 돌아온다.
- 허용 위치에서는 `0.0`을 더하는 대신 로짓이 그대로 통과한다. IEEE 부동소수점에서 `x + 0.0 == x`이므로 같은 값이고, 반올림 기회가 하나 늘어나는 것이 아니라 하나 줄어드는 쪽이다.

### 3.4 매처 없이 패킹 연산을 테스트

단위 테스트 모듈에는 `MlxcelTokenizer::stub()`밖에 없고 그 어휘는 비어 있어서, 매처를 구동하는 단위 테스트로는 의미 있는 마스크를 만들 수 없다. `tests/structured_outputs.rs`의 80줄짜리 인라인 `tokenizer.json`을 복제하는 대신, 패킹을 순수 함수로 분리하고 명시적으로 구성한 비트셋에 대해 테스트했다.

이 선택은 더 싼 쪽이 아니라 더 강한 쪽으로 드러났다. 열세 가지 폭 조합에 대한 무작위 허용 집합, 잉여 비트 누출을 드러내는 전부 1인 소스, 그리고 테스트 전용 참조로 남겨둔 옛 바이어스 루틴과의 직접적인 원소 단위 비교가 가능해진다. 매처를 구동하는 테스트는 그 특정 문법이 마침 허용하는 것만 단언할 수 있다. 매처 구동 거동은 변경 없이 통과하는 `tests/structured_outputs.rs`의 실행 가능한 21개 테스트가 계속 담당한다.

### 3.5 공집합 판정을 패킹된 워드에서 계산

옛 코드는 `Vec<bool>` 위에서 `[0, vocab_size_hint)` 구간의 허용 항목을 세고 그 수가 0이면 `StructuredOutputError::Matcher`를 냈다. 새 코드는 모든 패킹 워드가 0인지를 본다. `compute_packed_mask`가 정확히 로짓 폭으로 패킹하고 그 바깥을 모두 0으로 만들기 때문에, "모든 워드가 0"은 곧 "모델의 로짓 어휘 안에 매처가 허용하는 도달 가능한 토큰이 없다"와 정확히 같다. 따로 경계를 둘 필요가 없고 에러 메시지도 그대로다. 멈춘 매처는 빈 슬라이스를 돌려주는데 `[].iter().all(...)`은 `true`이므로 이전과 같은 에러 경로를 탄다.

---

## 4. 구현 상세

### 4.1 아키텍처 변경

변경 전의 스텝당 경로:

```
matcher.compute_mask_or_eos() -> SimpleVob (패킹된 u32)
  -> iter_set_entries        -> Vec<bool>, vocab_size 항목        [호스트 루프 1]
  -> 바이어스 채우기          -> Vec<f32>, vocab_size_hint 항목    [호스트 루프 2]
  -> from_slice_f32          -> 970KiB 업로드
  -> add(logits, bias)
```

변경 후:

```
matcher.compute_mask_or_eos() -> SimpleVob (패킹된 u32)
  -> pack_mask_words         -> Vec<u32>, ceil(V / 32) 워드       [워드 복사]
  -> from_slice_u32          -> 30KiB 업로드
  -> right_shift / bitwise_and / reshape+slice / astype -> 디바이스 위의 bool[1, V]
  -> where_cond(allowed, logits, -inf)
```

### 4.2 주요 코드 변경

`StructuredOutputConstraint::compute_packed_mask(vocab_size_hint)`는 `get_error` 확인까지 포함해 `compute_mask`와 똑같이 매처를 구동한 뒤, 펼치는 대신 패킹한다. 짝을 이루는 `compute_mask`와 달리 폭을 인자로 받는데, 패딩 폭은 매처가 아니라 모델 로짓 축의 성질이기 때문이다.

무엇이 살아남는지를 정하는 부분:

```rust
let matcher_vocab = self.vocab_size.min(vob.len());
pack_mask_words(vob.as_slice(), matcher_vocab, vocab_size_hint, &mut self.packed_buf);
```

`compute_mask`는 `self.vocab_size` 이상의 항목을 전부 버렸고, 그보다 짧은 비트셋에는 읽을 비트 자체가 없다. 둘 중 작은 쪽을 취하면 그 규칙이 한 표현식으로 재현된다.

`pack_mask_words`가 패킹 규칙의 전부다:

```rust
let n_words = vocab_size_hint.div_ceil(32);
out.clear();
out.resize(n_words, 0);

let valid_bits = matcher_vocab.min(vocab_size_hint).min(src.len() * 32);
let full_words = valid_bits / 32;
let rem = valid_bits % 32;

out[..full_words].copy_from_slice(&src[..full_words]);
if rem != 0 {
    out[full_words] = src[full_words] & ((1u32 << rem) - 1);
}
```

세 겹의 `min`은 의미론이면서 동시에 안전성 논증이다. `valid_bits <= vocab_size_hint`에서 `full_words <= n_words`가 따라오고, `rem != 0` 분기에서는 `valid_bits <= src.len() * 32`에 `valid_bits`가 워드 배수가 아니라는 사실이 더해져 `full_words < src.len()`이 따라온다. 두 인덱싱 지점 모두 런타임 검사 없이 범위 안이다.

`expand_packed_mask`가 디바이스 쪽 절반이다:

```rust
let packed  = from_slice_u32(words, &[n_words, 1]);
let bit_pos = from_slice_u32(&PACKED_MASK_BIT_POSITIONS, &[1, 32]);
let bits    = bitwise_and(&right_shift(&packed, &bit_pos), &one);
let flat    = reshape(&bits, &[1, n_words * 32]);
let trimmed = if flat_len as usize == vocab_size { flat } else { slice(&flat, &[0, 0], &[1, vocab_size as i32]) };
astype(&trimmed, dtype::BOOL)
```

`debug_assert_eq!`가 자르기 연산이 의존하는 불변식인 `words.len() == vocab_size.div_ceil(32)`를 못박는다. 한 폭으로 패킹해 다른 폭으로 펼치려는 미래의 호출자는 평탄 길이 밖을 자르는 대신 디버그 빌드에서 크게 실패한다.

`apply_structured_mask_to_logits`는 정책만 남는다:

```rust
if constraint.is_gated() { return Ok(copy(logits)); }
let words = constraint.compute_packed_mask(vocab_size_hint)?;
if words.iter().all(|word| *word == 0) { return Err(StructuredOutputError::Matcher(...)); }
Ok(apply_packed_mask_to_logits(words, vocab_size_hint, logits))
```

### 4.3 데이터 모델 변경

`StructuredOutputConstraint::bias_buf: Vec<f32>`가 제거되고 그 자리를 `packed_buf: Vec<u32>`가 대신한다. 어휘 248320개 기준으로 제약 시퀀스당 약 1MB의 호스트 상주 메모리가 줄고, 생성자가 잡아두는 용량도 이전의 1/32이다. `mask_buf`는 그대로이며 여전히 `compute_mask`를 뒷받침한다.

---

## 5. 학습 포인트

### 5.1 마스크는 토큰당 1비트다. 그보다 넓은 것은 전송 방식의 선택일 뿐이다

바이어스 배열은 토큰당 1비트짜리 정보를 32비트에 실어 나르고 있었고, 호스트 루프 두 개는 오직 그 넓히기를 수행하려고 존재했다. 질문을 "이 답을 담을 수 있는 가장 좁은 형태는 무엇인가"로 바꿔 놓으면 답은 생산자가 이미 계산해 둔 그 형태이고, 두 루프는 필요한 작업이 아니라 순수한 오버헤드로 드러난다. 매처는 비트셋 생산을 멈춘 적이 없다. 소비자가 부동소수점을 원했기 때문에 코드가 계속 그것을 풀고 있었을 뿐이다.

일반화하면 이렇다. 생산자와 소비자의 표현이 어긋날 때, 토큰마다 도는 변환기를 쓰기 전에 소비자에게 생산자의 형태를 가르칠 수 있는지부터 확인하라.

### 5.2 인덱스 패턴이 아핀이면 gather보다 브로드캐스트다

비트셋을 토큰당 한 항목으로 펼치는 자연스러운 방법은 토큰마다 그것이 사는 워드를 찾아보는 것이다. 그것이 gather이고, 출력만큼 넓은 인덱스 배열이 필요하다. 그런데 토큰에서 (워드, 비트)로 가는 사상은 임의가 아니라 `(i >> 5, i & 31)`이고, 이것이야말로 2차원 reshape가 공짜로 표현해 주는 것이다. 워드를 한 축에, 비트 위치를 다른 축에 놓으면 인덱스 테이블이 shape 안에 암묵적으로 들어간다.

이건 눈에 익혀 둘 만하다. 내용이 자기 위치의 산술 함수인 인덱스 배열은 대개 변장한 reshape다. MLX도 같은 이유로 `dequantize`에서 같은 결론에 도달했다.

### 5.3 `UniquePtr<MlxArray>`는 `Send`가 아니므로 시퀀스별 구조체에 캐시할 수 없다

`SequenceInfo`에서 닿는 것은 전부 스케줄러 스레드를 건너가므로 `Send`여야 한다. cxx는 불투명 `extern "C++"` 타입에 `Send`를 유도하지 않고 `mlxcel-core`에도 수동 impl이 없으므로, MLX 배열 핸들은 명시적인 `unsafe impl Send` 래퍼와 그에 딸린 안전성 논증 없이는 요청별 구조체에 저장할 수 없다(`src/server/prompt_cache/entry.rs`에 정확히 이 이유로 두 개의 래퍼가 있다). 디바이스 쪽 조회 테이블을 스텝 사이에 살려 두려는 설계는 모두 여기에 먼저 부딪힌다. 미묘한 버그가 아니라 컴파일 에러이긴 하지만, 설계가 아니라 구현 도중에 발견되면 계획을 늦게 뒤집는다.

### 5.4 지우려는 코드와의 A/B는 테스트로 남길 값어치가 있다

옛 바이어스 루틴을 테스트 전용 참조로 보존했고, 그 덕에 두 가지가 동시에 가능해졌다. 무작위 허용 집합과 일곱 가지 어휘 형상에 걸친 원소 단위 등가성 단언, 그리고 두 경로를 한 프로세스 안에서 동일한 조건으로 재는 마이크로벤치마크다. 옛 경로가 사라진 뒤에는 둘 다 불가능하고, 나중에 diff에서 복원하는 것은 그 복원 자체가 검증되지 않았다는 점에서 확실히 더 나쁘다. 참조 구현의 비용은 25줄이다.

---

## 6. 추가 학습

### 핵심 용어

- **`SimpleVob`**: 토큰 id에 대한 `toktrie`의 비트셋으로, `Vec<u32>`와 크기로 이뤄진다. 워드 `i / 32`의 비트 `i % 32`가 토큰 `i`이고 `as_slice()`가 워드를 그대로 노출한다. 백킹 스토어가 선언된 크기보다 넓을 수 있어 잉여 비트가 0이라는 보장은 없다.
- **패딩된 `lm_head`**: 출력 프로젝션의 행 수가 토크나이저의 토큰 수보다 많은 모델. Qwen3.8-27B는 토크나이저 항목 248077개에 대해 `vocab_size: 248320`을 선언하며, 어떤 토큰 id도 이름 붙이지 않는 243행이 남는다.
- **`vocab_size_hint`**: 호출부에서 로짓 배열의 마지막 축으로 읽은 모델의 로짓 폭. 마스크가 로짓 위로 브로드캐스트돼야 하므로 의도적으로 매처의 어휘가 아니다.
- **브로드캐스트 시프트**: `[n, 1]`에 대한 `right_shift`를 `[1, 32]`와 짝지어 `[n, 32]`를 만드는 것. MLX의 이항 연산은 NumPy 규칙으로 브로드캐스트하므로 명시적 `broadcast_to`가 필요 없다.

### 관련 기술/프레임워크

- `llguidance` 1.7 / `toktrie` 1.8. `response_format` 뒤의 문법 엔진으로, Blaizzy/mlx-vlm#1047에서 상류가 선택했고 mlxcel도 공유한다.
- MLX의 `Select` 프리미티브. `where_cond`를 뒷받침하며 값 피연산자를 공통 dtype으로 승격한다.
- MLX `dequantize`. 브로드캐스트 비트 언팩 형태의 인트리 선례다: https://github.com/ml-explore/mlx/blob/main/mlx/ops.cpp

### 관련 PR/이슈

- #1316. 이 PR이 닫는 이슈.
- Blaizzy/mlx-vlm#1805. `apply_mask_covers_the_qwen3_8_padded_lm_head`가 못박는 패딩 `lm_head` 크기 규칙의 상류 대응물.

---

## 7. 변경 요약

### 통계

| 항목 | 값 |
|------|-----|
| 변경 파일 | 2 |
| 추가 줄 | 698 |
| 삭제 줄 | 65 |
| 신규 단위 테스트 | 6개와 ignore된 벤치마크 1개 |
| 영향받은 크레이트 | `mlxcel`만 |

### 카테고리별 변경

| 파일 | 변경 |
|------|------|
| `src/server/structured.rs` | +218 / -65. `compute_packed_mask`, `pack_mask_words`, `expand_packed_mask`, `apply_packed_mask_to_logits`, `PACKED_MASK_BIT_POSITIONS` 추가. `apply_structured_mask_to_logits` 재작성. `bias_buf`를 `packed_buf`로 교체. |
| `src/server/structured_tests.rs` | +480. 단위 테스트 6개, 테스트 전용 바이어스 참조, 결정론적 xorshift, 배열 읽기와 argmax 헬퍼, ignore된 마이크로벤치마크. |

### 관련 커밋

- `ff3d8bd` perf(server): pack structured-output masks as u32 bitmasks

---

## 8. 후속 조치

### 필수

- PR 본문의 라이브 서버 확인을 `models/mlx/qwen3-4b-4bit`에 대해 실행할 것. `response_format: json_schema` 요청이 스키마에 맞는 객체를 `finish_reason: stop`과 함께 돌려줘야 하고, `temperature: 0`에서 완성 결과가 변경 이전 바이너리와 토큰 단위로 동일해야 하며, 제약 없는 대조군은 변화가 없어야 한다.
- 워크스페이스 전체 게이트를 실행할 것. 이번 변경은 좁은 명령(`--lib server::structured`, `--test structured_outputs`, `clippy --lib --tests`)으로 검증했고, `cargo test --workspace --profile test-fast --features metal,accelerate`와 `cargo clippy --workspace --all-targets -- -D warnings`는 여기서 실행하지 않았다.

### 모니터링 필요

- 동시성 하에서 넓은 어휘 체크포인트의 제약 디코딩 처리량. 제거된 비용은 동시 제약 시퀀스 수에 비례했으므로 이득이 가장 잘 보이는 곳이 거기이고, 회귀가 가장 먼저 드러날 곳도 거기다.

### 향후 개선

- 이슈 #1316은 네 연산짜리 확장이 지배적이 될 경우의 선택적 후속으로 `logits[i] = (words[i >> 5] >> (i & 31)) & 1 ? logits[i] : -inf` 융합 커널을 적어 두었다. 아래 측정은 그렇지 않다고 말한다. 248k 어휘에서 준비-업로드 구간 전체가 7.4us이므로 남은 비용은 로짓에 대한 원소별 순회이고, 그것은 어떤 구현이든 지불해야 한다.
- 이슈 자신의 범위 밖 목록도 그대로 유효하다. 그리디 제약 디코딩용 융합 masked-argmax 샘플러, 그리고 같은 문법 상태를 가진 시퀀스들이 마스크 업로드 하나를 공유하는 것.

---

## 부록

### A. 테스트 결과

```
cargo test --profile test-fast --features metal,accelerate --lib server::structured
  21 passed, 0 failed, 1 ignored (마이크로벤치마크)

cargo test --profile test-fast --features metal,accelerate --test structured_outputs
  21 passed, 0 failed, 2 ignored (로컬 모델 가중치 필요)

cargo clippy --profile test-fast --lib --tests --features metal,accelerate -- -D warnings
  clean

cargo fmt --all -- --check
  clean
```

신규 단위 테스트:

| 테스트 | 못박는 것 |
|--------|-----------|
| `packed_mask_matches_bool_mask` | 열세 가지 폭 조합, 무작위 허용 집합. 모든 비트가 불리언 마스크와 일치하고 로짓 폭 너머의 모든 레인이 0이다. |
| `packed_mask_trims_the_matchers_own_excess_bits` | 전부 1인 소스에 77토큰 매처. 어떤 토큰도 가리키지 않는 19비트가 살아남으면 안 된다. |
| `packed_mask_zero_pads_past_the_matcher_vocabulary` | 패딩 헤드 방향. 200폭 축에 40토큰. |
| `packed_mask_of_an_empty_allow_set_is_all_zero` | 빈 마스크 에러가 읽는 바로 그 술어, 그리고 마지막 부분 워드에 홀로 있는 허용 토큰. |
| `packed_apply_matches_bias_apply` | 일곱 가지 형상. 패킹 출력이 모든 위치에서 IEEE 동등성으로 f32 바이어스 출력과 같고, 허용 위치는 입력 로짓을 정확히 그대로 담으며, 비허용 위치는 `-inf`이고, `argmax`가 일치한다. |
| `packed_apply_handles_the_all_allowed_and_single_allowed_edges` | 전부 허용인 마스크는 무연산이고, 마지막 부분 워드의 단 하나뿐인 허용 토큰은 로짓 값과 무관하게 그리디 디코딩에서 이긴다. |
| `packed_apply_handles_a_width_change_between_calls` | 버퍼 하나에 여섯 가지 폭을 흘려보내며 각각을 전부 확인. |

### B. 성능 벤치마크

```
cargo test --profile test-fast --features metal,accelerate --lib \
  server::structured::tests::bench_packed_mask_apply -- --ignored --nocapture
```

M1 Ultra, Time Machine 중지, 토크나이저 항목 248077개에 대한 로짓 248320행, 시행당 min-of-60, 네 번 실행의 중앙값:

| 경로 | 준비 + 업로드 | eval 포함 | 업로드 |
|------|---------------|-----------|--------|
| f32 바이어스 (이전) | 711us | 1002us | 970KiB |
| 패킹 u32 (이후) | 7.4us | 322us | 30KiB |

네 번의 실행에서 관측된 범위: 707에서 714us, 6.7에서 8.9us, 986에서 1009us, 300에서 326us.

"준비 + 업로드"는 스케줄러 스레드 비용이다. 마스크 준비와 호스트-디바이스 복사를 거쳐 출력 그래프 노드를 손에 넣기까지다. "eval 포함"은 결과에 대한 강제 `mlxcel_core::eval`을 더한 것이다. 패킹 경로의 eval 수치는 알려진 MLX 단발 eval 바닥값 약 300us에 걸려 있으므로, 디바이스 쪽 이득을 과장하는 것이 아니라 축소해서 보여준다. 이 측정에서 정직하게 주장할 수 있는 것은 스케줄러 스레드에서 제거된 제약 시퀀스당 토큰당 약 0.7ms이고, 이는 이슈가 예상한 0.5에서 1.5ms 안에 있다.

두 경로는 같은 프로세스에서 동일한 조건으로 측정했으며, 참조 경로는 이 PR이 삭제하는 코드 그 자체다.

### C. 참고 자료

- 이슈 #1316. 변경 내용과 인수 기준을 규정한다.
- `src/server/structured.rs`의 `compute_packed_mask` / `pack_mask_words` / `expand_packed_mask`.
- `tests/structured_outputs.rs`의 `apply_mask_covers_the_qwen3_8_padded_lm_head`. 실제 크기의 부분 워드 사례다.
- MLX `dequantize`. 브로드캐스트 비트 언팩의 인트리 선례: https://github.com/ml-explore/mlx/blob/main/mlx/ops.cpp
- 공유 함수 관례는 `docs/code-guidelines.md`. 여기서는 어떤 공유 함수도 넓히지 않았다.
