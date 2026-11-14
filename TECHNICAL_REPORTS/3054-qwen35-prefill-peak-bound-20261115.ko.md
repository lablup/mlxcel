# 기술 보고서: PR #3054 - fix(inference): Qwen 3.5 장문 프롬프트 prefill 피크 제한

**날짜**: 2026-11-15
**리뷰어**: -
**상태**: 완료
**언어**: Rust, Bash, Markdown
**위험도**: 중간

---

## 요약

#3049로 attention score 버퍼를 제한한 뒤에도 Qwen 3.5 계열의 28K 토큰 prefill은 M3 Ultra에서 44 GB(4B), 78 GB(27B)까지 올라갔고, 서버는 기본 청크가 2048 토큰인데도 single pass와 같은 피크를 보였습니다(#3045). 원인은 서로 독립된 두 가지였습니다. Qwen 3.5 모델이 프롬프트의 모든 위치를 LM head에 통과시켰고(28K 토큰에서 logits 12.95 GiB), prefill plan이 prompt cache 스냅샷 구간을 한 번의 forward로 실행했는데 채팅 프롬프트에서는 이 구간이 사실상 프롬프트 전체입니다. 수정 후 서버 피크는 9.1 GB와 25.9 GB로 청크 CLI와 1.5% 이내가 되었고, TTFT는 16~27% 줄었으며, greedy 출력은 main과 바이트 단위로 같습니다.

---

## 1. 문제 정의

### 1.1 배경

`LanguageModel::forward_last_logits*`는 엔진이 CLI와 서버 모두에서 프롬프트 조각마다 호출하는 prefill 진입점입니다. 기본 구현은 `forward`로 모든 위치를 계산한 뒤 마지막 행을 잘라냅니다. 어휘가 큰 모델은 샘플링할 행만 projection하도록 이를 재정의하며, Gemma 4, Llama 3, Qwen 3 등은 이미 그렇게 합니다(#672). Qwen 3.5 계열, 즉 `Qwen35Model`, `Qwen3NextModel`, 그리고 `Qwen3_5ForConditionalGeneration` 체크포인트가 실제로 로드되는 래퍼 `Qwen35VLModel`은 재정의하지 않았습니다.

prompt cache는 분할 지점(history boundary, 선택적으로 시스템 프롬프트 root와 주기적 checkpoint)에 모델 상태 스냅샷을 저장합니다. `PrefillPlan`은 분할 지점마다 조각을 끝냈고, 그 앞 구간 전체를 "청크 크기와 무관하게" 한 번의 forward로 유지했습니다.

### 1.2 기존 문제

- Logits: 28000 x 248320 x 2 B = 12.95 GiB. 한 행만 읽기 위해 prefill에서 가장 큰 단일 버퍼를 만들었습니다.
- 서버 분할: history boundary는 생성 프롬프트를 뺀 채팅 렌더링이라 프롬프트 끝에서 몇 토큰 앞에 있습니다. 따라서 단일 턴 28K 채팅 요청은 약 28700 토큰을 한 조각으로, 4 토큰을 다음 조각으로 보냈고, 서버 피크가 single pass와 정확히 같았습니다(서버 44.6 GB, CLI 44.4 GB).
- `MLXCEL_MEMORY_LIMIT`는 한도를 넘으면 MLX가 "예외를 던진다"고 문서화되어 있었지만 MLX는 그렇게 하지 않습니다. 16 GB 한도에서도 prefill은 45.8 GB에 도달했습니다.

---

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| `qwen3_5.rs` | `forward_internal_at`, `forward_with_sequence_caches_at`, `forward_with_mrope_state`의 `last_pos` 인자 추가. 세 `forward_last_logits*` 재정의는 `forward_with_sequence_id`와 같은 sequence 상태와 MRoPE 상태를 사용합니다. hidden 행은 final norm 전에 잘라내며, RMSNorm이 행 단위라 결과 행은 같습니다. |
| `qwen3_next.rs` | `forward_for_sequence_at` / `forward_at`를 통한 같은 재정의. |
| `qwen3_5_vl.rs` | 세 진입점을 텍스트 모델에 위임합니다. 이것이 없으면 래퍼가 trait 기본 구현으로 돌아갑니다. |
| `prefill_plan.rs` | `append_segment_pieces` / `segment_piece`: `with_prefix`, `resume_at`, `next_piece_at`에서 구간을 최대 `chunk` 토큰의 패딩 없는 조각으로 나누고 마지막 조각이 boundary에서 끝나게 합니다. |
| Scheduler | 청크 plan에서는 boundary에서 끝나는 조각을 포함한 모든 조각이 `mlxcel_batch_prefill_chunks_total`에 집계되고 `prompt_progress`를 보냅니다. tick 사이에 유지하는 block 예약에는 boundary 조각과 같은 tick에 실행되는 다음 조각의 패딩도 포함합니다. |
| 문서 | `MLXCEL_MEMORY_LIMIT`를 soft limit로, 그렇게 유지하는 이유와 함께 문서화; CHANGELOG; CSV를 포함한 벤치마크 문서; 재실행 가능한 A/B 하네스 `scripts/bench/prefill_peak/ab_prefill.sh`. |
| 테스트 | `qwen3_5_prefill_memory_tests.rs`(행 일치, 청크 이어하기, 피크 메모리 가드, ignored 단계별 probe); plan, scheduler, MTP 분할 기대값 갱신. |

---

## 3. 기술적 선택과 그 이유

### 3.1 분할을 없애지 않고 구간을 청크로 나눔

조각을 반드시 끝내야 하는 곳은 boundary뿐입니다. 그곳에 저장되는 스냅샷은 키가 된 토큰을 정확히 기술해야 하고, recurrent 계열은 패딩을 되감을 수 없으므로 boundary 조각은 패딩하지 않습니다. 그 앞 구간을 어떻게 나누는지는 이 두 성질에도, prompt cache hit가 재실행하는 boundary 이후 조각에도 영향을 주지 않습니다. 구간 내부 조각도 패딩하지 않아 "스냅샷 전에는 패딩 없음" 규칙을 계열별 판단 없이 지킵니다. 대가는 cold prefill에서 구간 내부 KV가 한 번이 아닌 여러 번의 forward로 만들어진다는 점인데, 이는 plan이 `--prefill-chunk-size`에 대해 이미 문서화한 마지막 비트 수준의 차이와 같은 종류이며, 측정한 출력은 모두 같았습니다.

### 3.2 `MLXCEL_MEMORY_LIMIT`를 soft로 유지

MLX의 eval 루프는 active memory가 한도를 넘는 동안 진행 중인 command buffer를 기다리고, Metal allocator는 버퍼 캐시를 줄이지만, `malloc`은 할당을 거부하지 않습니다. hard cap을 만들려면 MLX가 제공하지 않는 hook으로 할당을 거부해야 하고, prefill 도중 요청을 실패시킬 뿐 필요한 메모리를 줄이지는 못합니다. 문서화한 해법은 prefill 청크로 피크를 제한하는 것이며, 이제 서버도 이를 따릅니다.

### 3.3 기여도 측정 방법

ignored probe는 단계마다 eval하고, 그 단계 시작 시점의 active bytes 대비 `peak_memory()`를 읽습니다. 하나의 lazy graph로는 단계별 transient를 분리할 수 없기 때문입니다. 같은 probe가 서빙 진입점을 lazy하게 실행해 전체 효과도 보여 줍니다(4B에서 가중치 대비 25.80에서 20.27 GiB).

---

## 4. 검증

| 경로 (28K 토큰, M3 Ultra) | qwen3.5-4b-4bit | qwen3.8-27b-4bit |
|---|---|---|
| CLI single pass | 44.39 → 30.13 GB | 78.47 → 64.20 GB |
| CLI 2048 청크 | 11.00 → 8.97 GB | 27.83 → 25.80 GB |
| 서버 `--parallel 1` | 44.57 → 9.10 GB | 78.64 → 25.93 GB |

시간 측정은 하나의 GPU lock 안에서 A/B, B/A 순서로 교차 실행했습니다. 더 빠른 쪽은 모두 새 빌드였고 반복 간 차이는 0.5초 이내입니다. 단위 테스트는 전체 forward 대비 재정의 결과, 이어지는 조각, 1024 x 131072 피크 가드(전체 logits 512 MiB, 한도는 그 1/4), 청크로 나뉜 구간, 구간 내부 이어하기, scheduler의 청크 집계를 다룹니다.

---

## 5. 남은 작업

- 28K single pass는 4B에서 여전히 30 GB까지 올라갑니다. 3.65 GiB causal array mask와 lazy graph 하나의 진행 중 attention 중간값 때문이며, full-attention 레이어를 MLX causal mode로 바꾸는 작업은 #3051입니다.
- 청크보다 짧은 구간 나머지(28703 boundary 앞 31 토큰)는 이제 별도 조각입니다. #3052의 tail 재조정이 여기에도 적용됩니다.
- M5에서는 측정하지 않았습니다. M5 전용 경로는 건드리지 않았습니다.
