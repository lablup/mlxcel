# 기술 리포트: PR #1775 - chore(core): fix create_causal_mask roster count and add a drift guard

**날짜**: 2026-09-11
**작성자**: mlxcel maintainers
**리뷰어**: 구현 리뷰 사이클(구현 리뷰, 보안·성능 리뷰)
**상태**: 완료
**언어**: Rust(문서 주석과 단위 테스트), Markdown
**위험도**: 낮음(프로덕션 코드 경로 변경 없음; 새 테스트는 `#[cfg(test)]` 전용)

---

## 요약

`src/lib/mlxcel-core/src/utils.rs`의 `// Used by:` 로스터는 기여자가 '이 헬퍼를 바꾸면 무엇이 깨지는가'를 확인하는 수단이고, `docs/code-guidelines.md`가 예시로 가리키는 것이 `create_causal_mask`다. 그 수치는 `src/models` 아래 호출 파일이 45개라고 적었는데, 주석이 안내하는 grep은 49를 돌려줬다. 주석에 찍힌 명령은 `src` 전체를 뒤져 64를 냈다. 형제 로스터 여섯 개도 어긋나 있었고, 그중 둘은 함수에 실제로 도달하지 않는 호출자를 적고 있었다.

이 PR은 기준 커밋 `8ad62d4b`에서 grep을 새로 돌리고 각 호출 지점을 읽어 로스터 일곱 개를 모두 바로잡는다. 찍힌 명령은 수치가 가리키는 모집단으로 좁혔다. 적힌 수치와 트리가 어긋나면 고칠 방법까지 알려 주며 실패하는 단위 테스트도 더했다.

---

## 문제 정의

수치가 마지막으로 맞았던 것은 `f0bf3a2c`(#1121, 44)였다. 그 뒤로 호출 파일이 다섯 개 들어왔는데, 본문에 이름이 오른 것은 그중 하나뿐이었다. 이슈는 어긋남을 커밋 단위로 추적했다. 로스터를 강제하는 장치는 없었다. `make verify`는 크레이트 버전, CUDA 커널 dtype 키, llama-server 매니페스트는 검사하지만 로스터는 보지 않는다. 그래서 마스크 헬퍼를 부르는 새 계열이 들어올 때마다 이 확인 수단은 소리 없이 틀려졌다.

첫 커밋 리뷰는 낡은 수치보다 나쁜 실패를 찾아냈다. 호출하지 않는 쪽을 호출자로 적은 로스터다.

- **`create_causal_mask_with_window`**는 Gemma2, Gemma3, Qwen3를 적고 있었다. Gemma2와 Qwen3는 `causal_attention`에 창 크기 `0`을 넘기므로 일반 인과 마스크 분기로 간다. Gemma3는 `causal_attention`을 아예 부르지 않고, 창 마스크를 `create_sliding_window_prefill_mask`에서 얻는다. 0이 아닌 창을 넘겨 실제로 이 함수에 도달하는 계열은 Baichuan, Cohere2, Cohere2MoE, CohereCompass, Exaone4, ExaoneMoE, Gemma3n, Gemma4, IQuestLoopCoder, Laguna, Mellum, Ministral3, MuseGlimmer, Step3P5다.
- **`softplus`**는 `lib.rs`에 있는 `ffi::softplus`의 cxx 선언과, Apertus가 따로 정의한 스칼라 `fn softplus(x: f32)`를 호출자로 셌다. 그러면서 `pub use ffi::*` 때문에 `mlxcel_core::softplus`가 FFI 함수 그 자체라는 점을 놓쳤다. Mamba, Mamba2, Jamba, RecurrentGemma, FalconH1, GraniteMoeHybrid, NemotronH, Plamo2와 오디오 어텐션 파일 둘은 FFI 함수를 직접 부르고 래퍼를 거치지 않는다. `utils::softplus`를 부르는 것은 GatedDelta, Laguna, DeepSeekV4MoE뿐이다.

어느 헬퍼든 바꾸기 전에 이 로스터를 읽은 기여자는 엉뚱한 모델을 확인했을 것이다.

---

## 변경 요약

- **로스터.** `create_causal_mask`는 49를 적고, 모든 파일을 이름 붙은 그룹에 넣는다. Laguna는 하이브리드 그룹, CohereCompass는 슬라이딩 윈도 그룹에 들어가고, GLM4-MoE-Lite MTP 드래프터와 LFM2 DSpark 드래프터는 각자 계열에 합쳤다. 'src/models 밖' 문장은 이제 grep 결과를 빠짐없이 다룬다. 여기엔 `cache.rs`의 테스트 전용 사용과 Qwen3.5 MTP 드래프터의 주석 언급도 들어간다. 형제 로스터 여섯 개는 직접 호출자와, 래퍼나 공유 `causal_attention` 디스패치를 거쳐 오는 호출자를 구분한다. grep이 잘못 잡은 결과도 그렇다고 명시한다. `repeat_kv`와 `softcap`은 이미 정확해서 손대지 않았다.
- **재생성 명령.** 이제 `grep -rln '\bcreate_causal_mask(' src/models --include='*.rs' | grep -v 'tests\.rs$'`이고, 적힌 수치를 그대로 돌려준다. `docs/code-guidelines.md`의 사본도 같은 명령이다. 가이드라인의 예시는 저절로 낡을 수치 대신 `N`을 쓴다.
- **가드 테스트** `utils::tests::create_causal_mask_roster_count_matches_repo`:
  - `src/models`를 `std::fs`로 훑는다. `DirEntry::file_type()`으로 재귀하므로 `grep -r`처럼 심볼릭 링크를 따라가지 않는다.
  - 단어 경계에 걸리는 `create_causal_mask(`를 담은 파일을 세고, 이름이 `tests.rs`로 끝나는 파일은 뺀다(`diffusion_gemma/tests.rs`도 여기서 빠진다).
  - `include_str!("utils.rs")`의 고정 문장 접두어에서 적힌 수치를 읽어 둘이 같은지 단언한다.
  - 실패하면 적힌 수와 잰 수, 정렬된 파일 목록, 고칠 파일과 줄, 다시 돌릴 명령을 출력한다. 줄 번호는 마커 위치에서 계산하므로 낡지 않는다.
  - 계산한 루트에서 워크스페이스의 `utils.rs`에 닿지 못할 때(벤더링 빌드 등)만 건너뛰고, 그 밖에는 `src/models`가 있다고 단언한다. I/O 오류는 엉뚱한 어긋남 메시지가 아니라 경로와 함께 패닉한다. 의존성은 늘리지 않는다.

---

## 기술적 선택과 그 이유

**`make verify-*` 스크립트가 아닌 단위 테스트.** 수치는 고정 접두어 뒤의 정수 하나이고, `verify-test`는 이미 모든 PR에서 돈다. 스크립트로 하면 정수 하나 때문에 Makefile 대상과 CI 잡을 새로 만들어야 한다. 비교 대상이던 스크립트형 방식은 `scripts/ci/check_kernel_dtype_keys.py`이고, 이 경우에는 기각했다.

**앞 문자 `:`는 여전히 센다.** 이슈는 앞 문자가 `_`, 영숫자, `:`인 매치를 빼라고 했다. 그런데 `:`까지 빼면 35가 나온다. 14개 파일이 이 헬퍼를 `utils::create_causal_mask(`로 부르기 때문이다. 그러면 테스트가 문서 주석이 고정한 grep 정의와 어긋난다. 구현은 `grep`의 `\b` 의미를 따르고, 누군가 이슈 문구에 맞춰 '고치지' 않도록 그 규칙 옆에 이유를 주석으로 남겼다.

**수치만 보고 계열 이름은 보지 않는다.** 형제 로스터는 수치가 아니라 계열 이름을 적으므로 비교할 정수가 없다. 부재 탐지나 계열-경로 대응표 같은 일반 `// Used by:` 검사기는 #1141의 설계 문제로 남겨 범위 밖에 둔다.

---

## 검증

- 재생성 명령은 기준 커밋에서 이 셸의 `grep`, BSD `/usr/bin/grep`, `rg` 모두로 49를 돌려준다.
- 음성 증명: 수치를 48로, 다음엔 50으로 고치면 가드가 고칠 방법을 담은 메시지 전체와 함께 실패하고, 49로 되돌리면 통과한다.
- 첫 커밋에서 워크스페이스 게이트(바이너리 123개, 11,015 통과, 0 실패, 359 무시), `-D warnings` 워크스페이스 clippy, fmt, `make verify-versions` / `verify-kernel-dtype-keys` / `verify-llama-compat`가 모두 통과했다. 리뷰 수정 커밋은 같은 파일의 문서 주석과 테스트 모듈만 건드리므로, `mlxcel-core` lib 전체와 워크스페이스 clippy로 다시 검증했다.

---

## 학습 포인트

- **호출하지 않는 쪽을 적은 로스터는 불완전한 로스터보다 나쁘다.** 불완전한 로스터는 독자를 더 찾아보게 만든다. 틀린 로스터는 영향이 없는 Gemma2와 Qwen3를 확인하게 하고, 실제로 영향받는 열네 계열은 건너뛰게 만든다. 수치 가드로는 이것을 잡을 수 없고, 호출 지점마다 인자를 읽어야만 잡힌다. 창 로스터의 오류도 리뷰에서 그렇게 찾았다.
- **정의는 세 곳에서 같아야 한다.** 문장, 찍힌 명령, 테스트가 각자 '호출 파일이란 무엇인가'를 담는다. 셋은 한 번 어긋난 적이 있다(명령은 `src`, 문장은 `src/models`). 이 PR 이후로는 나머지 둘이 어긋나면 테스트가 실패한다.

---

## 후속 과제

- `utils.rs`, `layers.rs`, `switch_layers.rs`, `gated_delta.rs`를 대상으로 하는 일반 `// Used by:` 검사기는 #1141로 계속 열려 있다. 그 기준 중 '현재 `main`에서 그대로 통과'는 이제 `utils.rs`에 대해서는 다시 참이 됐다.
