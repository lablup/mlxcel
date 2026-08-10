# 기술 보고서: PR #1099 - fix(models): correct Molmo v1 image decoding seams

**날짜**: 2026-08-11
**작성**: mlxcel 메인테이너
**검토**: 구현, 보안, finalizer 리뷰 사이클
**상태**: 열린 PR 기준 작성 완료
**언어**: Rust, Markdown
**위험도**: 중간

---

## 요약

PR #1099은 이슈 #1087의 Molmo v1 이미지 조건 생성 결함을 고친다. 실제 COCO 고양이 사진에서 `molmo-7b`만 incoherent한 출력을 내고 `molmo2-4b`는 같은 환경에서 정상 응답하던 문제가 대상이었다. 원인은 서버 경로나 BOS 처리, feature scatter가 아니었다. Molmo v1 디코드 스택의 두 seam이 핵심이었다. `rope_impl`를 생략한 flat config가 체크포인트의 실질적인 LLaMA rotate-half 레이아웃이 아니라, 로컬의 오래된 MLX traditional interleave 기본값으로 떨어지고 있었고, 이미지 전처리도 raw-black padding과 fractional patch coverage를 유지해야 하는 reference 동작에서 벗어나 있었다.

이 PR은 `src/models/molmo.rs`, `src/vision/processors/molmo.rs`에서 두 seam을 바로잡고, 해당 경계를 고정하는 회귀 테스트를 추가했다. 그리고 처음 문제를 재현했던 실제 NVIDIA GB10 CUDA 경로에서 다시 검증했다. 수정 전 CLI는 `prompt_tokens=749`에서 보고된 깨진 출력을 바이트 단위로 그대로 재현했고, 수정 후 CLI와 정상 BatchScheduler 경로의 서버는 둘 다 "빨간 소파와 분홍 담요 위에서 자는 고양이 두 마리"를 일관되게 설명했다. 두 개의 독립 리뷰는 추가 발견 사항이 없다고 결론냈고, finalizer 패스도 PR을 merge-ready로 판단했다.

---

## 1. 문제 정의

### 1.1 사용자에게 보인 실패

이슈 #1087은 `molmo-7b`가 이미지 프롬프트에서 incoherent한 출력을 생성하는 반면, 같은 GB10 CUDA 환경의 `molmo2-4b`는 정상이라는 점을 보고했다. 이 실패는 실제 640x480 COCO 고양이 사진에서 재현됐고, feature 브랜치뿐 아니라 `main`에도 이미 존재했기 때문에 인접 작업 회귀가 아니라 Molmo v1의 기존 결함이었다.

### 1.2 수용 기준

이슈는 세 가지를 요구했다.

- Molmo v1이 실제 사진을 CLI에서 일관되게 설명할 것,
- 프롬프트 우회가 아니라 실제 root-cause seam을 고칠 것,
- 그리고 `mlxcel` CLI와 `mlxcel-server` 양쪽에서 검증할 것.

---

## 2. 근본 원인

서로 독립적인 두 seam이 확인됐다.

### 2.1 생략된 `rope_impl`가 잘못된 RoPE 레이아웃을 선택했다

Molmo v1 flat config는 `rope_impl`를 생략할 수 있다. 배포된 체크포인트의 실질 동작은 여전히 LLaMA rotate-half 레이아웃을 기대하는데, 로컬 dataclass 기본값은 이 생략을 MLX traditional interleave RoPE로 해석하고 있었다. 이 불일치는 이미지 조건 추론 이전의 positional rotation부터 어긋나게 만들어, 프롬프트와 서버 경로, 이미지 feature가 맞더라도 디코드가 무의미해진다.

수정은 생략된 Molmo v1 `rope_impl`를 체크포인트의 실질적인 LLaMA 경로로 기본화하고, 명시적인 `interleave`만 MLX traditional 경로로 유지하는 것이다.

### 2.2 이미지 전처리가 reference와 달랐다

`pad_and_partial_pad`에는 두 가지 reference 불일치가 있었다.

- 정규화 뒤에 padding해서, 패딩 픽셀이 normalized black이 아니라 zero-valued normalized-space 픽셀이 됐다.
- `image_masks`를 boolean으로 threshold해서, reference가 유지하는 fractional border coverage 값이 사라졌다.

이 두 문제는 특히 부분 border patch에서 Molmo v1이 소비하는 visual token을 손상시킨다. 수정은 정규화 전에 raw black으로 padding하고, float mask coverage를 끝까지 유지하는 것이다.

---

## 3. 원인이 아니었던 것

가능성 있어 보였지만 명시적으로 배제된 항목들도 있다.

- `749` 프롬프트 토큰 확장 자체는 문제가 아니었다. 수정 전 CLI와 수정 후 CLI 모두 같은 실제 프롬프트 길이를 썼기 때문에, 문제는 프롬프트 길이가 아니라 디코드 의미론에 있었다.
- Molmo v1 BOS 처리는 차이를 만든 요소가 아니었다. 보고된 깨진 출력도 기존 CLI 경로에서 나왔고, 수정된 정상 출력도 같은 BOS 경로 위에서 모델/전처리 수정만으로 얻어졌다.
- 이미지 feature scatter 경로가 root cause는 아니었다. 수정 후 정상 결과는 같은 공용 CLI/server 멀티모달 런타임 경로에서 RoPE와 전처리 수정만으로 나왔다.
- 서버 전송 경로만의 문제도 아니었다. 정상 BatchScheduler 기반 `mlxcel-server --parallel 1` 경로가 수정된 CLI와 같은 coherent 응답과 같은 usage totals를 반환했다.

---

## 4. 구현 요약

| 항목 | 값 |
|------|----|
| 구현 변경 파일 | 2 |
| 구현 추가 라인 | +158 |
| 구현 삭제 라인 | -16 |
| 구현 커밋 수 | 1 |
| 검토한 구현 head | `8d30a09b79138d408860eb04cf23f94d7be06897` |

위 구현 집계에는 영문·국문 보고서 산출물과 보고서 전용 커밋을 포함하지 않았다.

- `src/models/molmo.rs`: 생략된 Molmo v1 `rope_impl`를 체크포인트의 실질적인 LLaMA split-half 경로로 기본화하고, 생략/명시 RoPE 제어에 대한 회귀 테스트를 추가했다.
- `src/vision/processors/molmo.rs`: 정규화 전에 raw black padding을 수행하고 fractional `image_masks`를 유지하도록 고쳤으며, normalized padding과 알려진 640x480 `9/14` partial border patch 케이스를 고정하는 결정적 테스트를 추가했다.

---

## 5. 검증

### 5.1 로컬 결정적 검증

- `cargo fmt --all -- --check`: 통과.
- `cargo test -p mlxcel models::molmo::tests`: 통과.
- `cargo test -p mlxcel vision::processors::molmo::tests`: 통과.
- `cargo test -p mlxcel --test molmo_parity`: 통과했지만, 네 개 테스트 모두 `models/molmo-7b`를 찾다가 조기 종료했다. 이 머신의 실제 체크포인트 경로는 `/home/inureyes/models/mlx/molmo-7b`이므로, 이 스위트는 이번 PR의 real-checkpoint 증거가 아니다.
- `cargo clippy -p mlxcel --lib --tests -- -D warnings`: 통과.
- NVIDIA GB10(`sm_121`)에서 `cargo build --release --features cuda --bin mlxcel --bin mlxcel-server --locked`: 통과.

### 5.2 원래 실패 경로에서의 실제 체크포인트 A/B

CLI, GB10 CUDA, 실제 640x480 COCO 고양이 이미지:

- 수정 전: 보고된 incoherent 출력을 `prompt_tokens=749`, `completion_tokens=40`, `9.35s`, `4.28 tok/s`로 바이트 단위까지 그대로 재현.
- 수정 후: `prompt_tokens=749`, `completion_tokens=40`, `4.06s`, `9.85 tok/s`로 "빨간 소파와 분홍 담요 위에서 자는 고양이 두 마리"를 일관되게 설명.

### 5.3 실제 서버 검증

같은 이미지와 프롬프트로 normal BatchScheduler 서버 경로(`mlxcel-server --parallel 1`)를 검증했다.

- 수정된 CLI와 동일한 coherent 텍스트를 반환했고,
- `prompt_tokens=749`, `completion_tokens=40`, `total_tokens=789`를 보고했다.

기존 `--no-batch` 모드도 시도했지만, generation 전에 기존의 `CachePool` max-capacity-zero admission error로 실패했다. 이 실패는 이번 PR 이전부터 있던 것이고 Molmo v1 decode/preprocessor seam과 독립적이므로, 정상 production scheduler 경로 검증을 무효화하지 않는다.

### 5.4 PR #1099에서 확인한 hosted checks

- `Detect changes`: pass
- `crate versions`: pass
- `kernel dtype keys`: pass
- `cross-repo refs`: pass
- `cargo-deny`: pass
- `cargo-fmt`: pass
- `license/cla`: pass
- `MLX pin extraction`: skipped

---

## 6. 리뷰 결과

첫 수정 커밋 이후 구현은 두 번의 독립 리뷰를 거쳤다.

- 구현 리뷰는 real-path CLI/server 검증과 추가된 회귀 테스트 이후 남은 정확성 결함이 없다고 판단했다.
- 보안 리뷰도 새 attacker-controlled surface나 추가 보안 이슈를 발견하지 못했다.

finalizer 패스는 정상 production scheduler 경로에서 이슈 수용 기준이 충족됐고, 남은 `--no-batch` `CachePool` 실패는 기존 이슈이자 범위 밖의 non-blocking 항목이라고 정리했다.

---

## 7. 핵심 기술적 교훈

- 모델 계열의 실제 체크포인트 동작과 오래된 로컬 기본값이 어긋난 상황에서는 flat config omission이 특히 위험하다. 신뢰해야 할 경계는 stale dataclass 가정이 아니라 체크포인트의 실질 constructor 동작이다.
- 비전 전처리는 padding과 mask 의미론까지 reference에 충실해야 한다. fractional border coverage를 잃는 것만으로도 하드 에러 없이 멀티모달 디코드가 망가질 수 있다.
- parity 하니스는 실제 체크포인트에 도달할 때만 수용 근거가 된다. 이번 `cargo test --test molmo_parity`는 코드 수준 회귀 스위트로는 유용했지만, 이 머신에서는 조기 종료했기 때문에 acceptance proof가 될 수 없었다.

---

## 8. 관련 작업

- PR #1099: https://github.com/lablup/mlxcel/pull/1099
- Issue #1087: https://github.com/lablup/mlxcel/issues/1087
