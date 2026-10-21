# PR #2513: Gemma 3n emitter를 공유 레이어 query offset으로 이동

**작성일**: 2026-10-21
**상태**: 완료
**위험도**: Medium
**언어**: Rust

---

## 요약

#2466 종결. mlxcel-xla Gemma 3n emitter가 여전히 공유-KV 레이어 query를 `positions + real_len`(prefill)과 `position + 1`(decode)로 회전시켜(#2457 이전 MLX 동작 미러링), #2457이 MLX 경로를 고친 뒤 MLX-vs-XLA Gemma 3n 패리티가 갈라졌다. 이 PR은 post-#2457 offset 의미론을 두 그래프에 이식하고(rope 테이블 2*capacity→capacity로 축소), `#[ignore]` 상태의 xla-diagnostics 핀을 출처와 함께 갱신하며, 이후 빌드되지 않았던 `src/models/gemma3n.rs`의 xla-diagnostics 호출 지점을 컴파일한다.

---

## 1. 문제 정의

### 1.1 배경

PR #2457이 MLX 경로의 Gemma 3n 공유-KV 레이어 query offset을 수정했다(`rope_offset`을 포워드당 한 번 읽음, mlx-lm의 `cache[0].offset`과 일치). 당시 XLA emitter는 의도적으로 기존 offset을 미러링해 뒀다. 에픽 #566 관련.

### 1.2 기존 문제점

- Prefill: emitter가 공유 레이어 query 위치를 `positions + real_len`으로 만들었지만 수정된 MLX 경로는 모든 레이어의 query를 미리 읽은 같은 offset으로 회전시킨다.
- Decode: emitter가 공유 레이어 query를 스텝당 `position + 1`로 전진시켰지만 MLX는 미리 읽은 `position`으로 회전시킨다.
- `src/models/gemma3n.rs`의 xla-diagnostics 호출 지점은 #2457 이후 손으로 갱신됐지만 컴파일된 적이 없다(진단 피처 표면은 기본 빌드에 없음).

### 1.3 위험성

| 위험 | 영향 | 가능성 |
|------|------|--------|
| 공유-KV 레이어에서 MLX-vs-XLA Gemma 3n 패리티 갈라짐 | Medium (잘못된 query 회전) | High (모든 Gemma 3n XLA 요청) |
| 기존 pre-#2457 핀이 옛 동작을 옳게 보여줌 | Low | High |

---

## 2. 기술적 검토 사항

### 2.1 보안 관점

클린. emitter 수학과 테스트 핀뿐이다: 새 입력 파싱 표면 없음, 공유·가변 상태 없음(추가된 모든 줄을 static/once_cell/OnceLock/Mutex/RwLock/unsafe로 grep: 0), IREE/런타임 변경 없음. 축소된 rope 테이블 경계는 세 decode 경로 모두에서 검증(단일 토큰 decode는 0 <= cache_len < context_capacity가 아니면 에러, ragged decode는 실행 전 모든 행 검증, admission 불변식이 cache_len 경계).

### 2.2 성능 관점

엄격히 유리. rope 테이블 절반: 4개 const 텐서 [2C,d]→[C,d]가 모듈에 베이크(C=32768/d=256에서 모듈 상수 128MB→64MB), 모듈 로드/컴파일 일회성 이득. 토큰당 decode는 공유-KV 레이어당 스텝당 스칼라 add 하나 감소, prefill은 공유 레이어당 [rows] i32 broadcast+add 하나 감소.

### 2.3 호환성/의존성 관점

- **호환성 깨짐**: 없음. servable 스키마 불변(`real_len`은 graph 입력으로 유지, gemma3n_emit.rs:461에서 출력 행 선택).
- **새 의존성**: 없음.

### 2.4 코드 품질 관점

- 테스트: gemma3n 40 통과 / 0 실패(`--features cuda,xla-diagnostics`, 기존 ignore 22), xla_gemma3n_prefill `--ignored` 1 통과 / 0 실패(MLXCEL_XLA_DEVICE=cuda; token vs dense-PLE greedy 동등성, query RoPE가 정확히 capacity-1까지 가는 근접 용량 prefill, 혼합 cancel/reuse 배치). crate 유닛 테스트 364/0. gemma3n 등록 그래프 픽스처는 없으며 #932 계약상 이 패밀리의 게이트는 pinned 진단이고 GB10에서 양방향 직접 측정으로 검증.
- 피처 명명: root crate에는 `xla` 피처가 없다. 표면은 xla-iree(프로덕션)와 xla-diagnostics(cuda + iree + diagnostics)다. 이슈의 `cargo build --features xla`는 존재할 수 없는 명령.

---

## 3. 기술적 선택과 그 이유

### 3.1 post-#2457 의미론을 정확히 이식

**컨텍스트:** 수정된 MLX 경로는 `rope_offset(caches)`를 포워드당 한 번 읽어 모든 `DecoderLayer::forward`로 전달하고, 모든 레이어(공유·구체)의 query를 그 미리 읽은 offset으로 회전시키며, 공유 레이어는 소유자의 live K/V 윈도우를 슬라이스한다.

**선택:** 공유 레이어 `positions + real_len` 블록(prefill)과 `position + 1` 블록(decode)을 제거해 모든 레이어가 같은 위치에서 query를 회전시키고, rope 테이블을 2*context_capacity→context_capacity 행으로 축소(최대 gather 인덱스 capacity-1이 정확히 맞음; position == capacity decode는 K/V 쓰기 자체가 범위 밖이라 이전부터 무효). #2457 diff와 현재 MLX 포워드 경로로 검증.

### 3.2 출처를 남겨 ignore된 핀 갱신

**컨텍스트:** xla-diagnostics 핀(top1 236798)은 #2457 이전 것이고 테스트가 `#[ignore]`라 갱신된 적이 없었다.

**선택:** prefix-decode top1 핀 236798→854 갱신, main 551039ee에서(MLX 핀 bff10220) mlx-vlm 0.7.8 CPU 오라클 동의하에 재캡처; full-prefill과 greedy 핀은 불변 유지; 크로스 디바이스 근접 랭킹 뒤집힘(CPU가 단일 패스 full prefill에서 0.75 logit 차로 854를 먼저 랭크)은 #932 유지자 계약대로 기록만 하고 gating하지 않음.

### 3.3 잔여 드리프트는 숨기지 않고 기록

이슈가 겨냥한 prefill 갭은 해소됐다(token-prefill logits max_abs 1.9375→0.8203, 1.5 엔벨로프 내, top1 일치). prefix-decode 비교는 수정 전(2.2617, cosine 0.9432)과 후(2.7266, cosine 0.8710) 모두 핀 엔벨로프 밖이고, 수정이 못 건드리는 모든 스테이지(scaled_embeddings 0.03125 vs 엔벨로프 0, layer0_ple_injected 2 vs 0.25, layer 중간 평면 15 vs 3)는 수정 전 emitter에서 byte-identical로 측정돼 그 잔여가 #2457과 이 PR 양쪽 이전부터 존재함을 증명한다(핀 마지막 캡처 2026-07-24, 이후 MLX 핀이 PR #2505로 이동). #932 계약대로 기록된 드리프트는 gating하지 않는다.

---

## 4. 구현 상세

**파일: `src/lib/mlxcel-xla/src/emitter/gemma3n_emit_ops.rs`** — `constants()`가 rope 테이블 축소; `attention()`이 real_len 파라미터와 공유 레이어 offset 블록 제거; `attention_decode_traced()`가 공유 레이어 스텝 블록 제거.

**파일: `src/lib/mlxcel-xla/src/emitter/gemma3n_emit.rs`** — 호출 지점이 `&args.real_len` 제거; `args.real_len`은 마지막 토큰 동적 슬라이스에 계속 사용.

**파일: `src/models/gemma3n.rs`** — `#[cfg(all(test, feature = "xla-diagnostics"))]` 모듈 내 핀 갱신과 출처 코멘트; 프로덕션 코드 불변; 진단 호출 지점 클린 컴파일.

**파일: `tests/xla_gemma3n_prefill.rs`** — full-capacity 코멘트 갱신.

---

## 5. 검증

| 검사 | 결과 |
|---|---|
| cargo test gemma3n (cuda,xla-diagnostics, --test-threads=1) | 40 통과 / 0 실패 |
| xla_gemma3n_prefill --ignored (MLXCEL_XLA_DEVICE=cuda) | 1 통과 / 0 실패 |
| mlxcel-xla crate 유닛 테스트 | 364 통과 / 0 실패 |
| cargo fmt (변경 파일) / 스코프 clippy | 클린 / 클린 |
| 풀 게이트 (161 바이너리, HEAD 4a1b99fc) | build 0, clippy 0, doc 0; 160/161 (mllama_parity 7/1 = 알려진 #2467 1-ulp flake, main 통제로 환경적임을 증명); CT_EXIT=2 = 병렬 PR #2505가 main에 남긴 기존 고장(kernel-port dispatch, env-var 문서, fmt), 이 PR과 무관하고 별도 추적 |
| fail-before (수정 전 emitter, 직접 측정) | token-prefill max_abs 1.9375 FAIL (엔벨로프 1.5); prefix-decode top1 xla=236798 vs mlx=854 (pre-#2457 MLX 재현) |
| post-fix (직접 측정) | token-prefill 0.8203 엔벨로프 내, top1 일치; greedy [236798, 236789] 양쪽 |
| 보안 검토 | CLEAN; admission 불변식으로 3개 decode 경로 경계 검증; rope 테이블 절반(32768/256에서 모듈 상수 128MB→64MB) |
| IREE 툴체인 메모 | pinned 쌍 .cache/mlxcel/iree-cuda-3.12.0rc20260721 + venv가 맞는 쌍; Development/iree-cuda 트리의 런타임이 더 오래됨(hal 버전 불일치 6 vs 7) |

---

## 6. 변경 요약

### 통계

| 항목 | 값 |
|------|-----|
| 변경 파일 | 4 |
| 추가 | +35 |
| 삭제 | -28 |

### 관련 커밋

| 해시 | 타입 | 메시지 |
|------|------|--------|
| `4a1b99fc` | fix | fix(xla): update Gemma 3n emitter to the shared-layer query offset |

---

## 7. 후속 조치

### 모니터링 필요

- pinned 진단 테스트는 기록된 기존 드리프트 스테이지 때문에 명시적으로 실행하면 여전히 실패; 실패하는 모든 스테이지는 수정 전 emitter에서 byte-identical.
- 정적 0..capacity-1 positions 버킷 때문에 0이 아닌 offset에서의 가정적 continuation prefill은 여전히 갈라짐; Gemma 3n은 chunked/multi-call prefill을 opt-out하므로 현재 경로에서 도달 불가.

### 향후 개선 사항

- `mod gemma3n_math`(src/lib/mlxcel-xla/src/emitter/mod.rs:59)는 죽은 코드(base에서 이미 죽음, 마지막 터치 #922)에 pre-#2457 attention_bias 오라클이 낡음; 후속 클린업 이슈 후보.
- 기존 tiny() 컨벤션을 따르는 non-ignored rope-table-capacity 테스트가 pinned 체크포인트 없이 축소된 테이블 형태를 고정할 수 있다.

---

## 부록

### A. 리뷰 판정

pr-reviewer: 6개 포커스 전부 PASS, 0건(#2457 diff와 현재 MLX 포워드 경로로 의미 검증; ragged batched decode와 serve 경로가 고친 함수를 통과; rope 테이블 경계 타당). pr-security-checker: CLEAN, CRITICAL/HIGH/MEDIUM 0; LOW 정보성 2건(죽은 gemma3n_math 모듈; 854 핀은 테스트가 ignore라 정적으로 검증 불가). pr-finalizer: 완료, 갭 없음; PR 본문 주장 전부 GB10 직접 측정으로 재현(fail-before와 post-fix 수치, 잔여 테이블, IREE 툴체인 쌍).
