# PR #2610: Baichuan-M1, Mllama, Qwen3-VL DeepStack 상태의 sequence별 분리

**Date**: 2026-10-26
**Status**: Metal(M1 Ultra)에서 완료. Cohere Compass는 synthetic 테스트로만 검증
**Languages**: Rust, Markdown
**Risk Level**: Medium

## 요약

PR #2610(이슈 #2567)은 세 모델 계열의 요청별 상태 격리를 고칩니다. 세 계열 모두 배칭을 지원한다고 보고하면서 요청별 상태는 모델 단위 슬롯 하나에 두고 있었습니다.
- Baichuan-M1: K/V convolution 상태.
- Mllama: cross-attention 이미지 상태와 레이어별 이미지 K/V.
- Qwen3-VL, Qwen3-VL-MoE, Qwen3-Omni, Cohere Compass: DeepStack 시각 특징.

`main`에서 `mlxcel-server --parallel 4`에 요청을 동시에 보내면 서로의 상태를 읽었습니다. 실측 예로, Llama 3.2 Vision 요청이 다른 요청의 이미지를 설명했고, Baichuan-M1은 요청 하나만 보내도 출력이 깨졌습니다. 이제 세 계열 모두 저장소가 Qwen MRoPE와 Gemma 4 `per_layer_inputs`에 이미 쓰는 구조(fallback 슬롯과 `HashMap<SequenceId, T>`)를 따릅니다.

검증 중에 이와 별개인 두 번째 메커니즘도 드러났습니다. decode lookahead 파이프라인이 투기적 스텝을 철회할 때 KV 캐시 밖에 있는 상태를 되감지 않았습니다. PR은 이를 위한 core hook을 추가합니다.

## 1. 문제 정의

스케줄러는 멀티모달 상태를 enqueue 시점에 준비하고 prefill은 나중에 실행합니다. 그래서 한 drain tick에 함께 승인된 요청 묶음에서는 요청 A의 prefill이 공유 슬롯을 읽기 전에 요청 B의 준비가 그 슬롯을 덮어씁니다. 요청을 차례로 보내도 깨지는 계열이 둘 있었습니다.
- Baichuan의 conv 상태는 요청 사이에 초기화되지 않았습니다.
- Mllama의 이미지 상태는 지워지지 않아서, 이미지 요청 뒤에 온 텍스트 전용 요청이 이전 이미지에 attention했습니다.

배칭을 끄는 방안(`supports_batching() == false`)은 이슈에서 두 가지 이유로 기각됐습니다. 순차 누수를 막지 못하고, 묶음 덮어쓰기는 decode 폭과 무관하게 승인 단계에서 이미 일어나기 때문입니다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| Baichuan-M1 | conv 상태를 `ModelOwnedSequenceState`로 옮기고, CLI용 fallback은 `make_caches`에서 초기화합니다. padded prefill을 끕니다. prompt-cache adopt/donate, KV handoff, APC(`HYBRID_SSM_MODEL_TYPES` 경유)는 이 모델을 거절합니다. |
| Mllama | `MllamaImageState`가 cross 상태와 decoder 레이어별 cross K/V 슬롯을 fallback과 map에 담습니다. 승인 시 bind하고, preemption 때는 take 후 새 id로 install합니다. 이미지 없이 bind된 sequence에는 명시적 텍스트 전용 표식을 남기고, bind되지 않은 id(CLI)는 fallback을 읽습니다. 텍스트 전용 decode 행은 네이티브로 배치하고, 이미지 행은 행별로 실행합니다. |
| DeepStack | `DeepStackState`가 MRoPE와 같은 수명주기를 따릅니다. 같은 승인 호출에서 bind되고, preemption snapshot에 함께 실리며(`QwenVlMRopeSnapshot`은 struct로 변경), 함께 해제됩니다. 그동안 bind된 적이 없던 Qwen3-Omni도 포함했습니다. |
| Core lookahead | `LanguageModel::kv_cache_captures_sequence_state()`(기본값 `true`)를 추가했습니다. 값이 `false`인 모델은 되감기를 구현한 경우에만 파이프라인에 들어가고, 엔진은 lookahead를 철회할 때마다 `rewind_decode_appends`를 호출합니다. |
| Baichuan 되감기 | 레이어마다 conv 상태 이력을 정해진 개수(`DECODE_LOOKAHEAD_MAX_SPECULATIVE_APPENDS + 1`)만 보관하고, 그보다 깊이 되감으려 하면 실패 처리합니다. prefill 토큰은 fused QKV 버퍼에서 복사해 두므로, 이력이 2048-token chunk당 약 1.1 GB를 붙잡지 않습니다. |

## 3. 기술적 선택과 그 이유

### 3.1 기존 fallback + map 구조 재사용

저장소는 같은 종류의 문제를 이미 두 번 해결했습니다(`MRopeState`, `Gemma4PerLayerInputsState`). 같은 구조를 쓰면 스케줄러의 승인 bind, preemption take/install, 해제가 계열 사이에서 일관되고, CLI는 익명 슬롯 하나를 그대로 씁니다.

### 3.2 bind되지 않은 id는 fallback을 읽고, bind된 텍스트 전용 행은 읽지 않음

첫 구현은 항목이 없는 id를 텍스트 전용으로 취급했습니다. pr-reviewer가 이 때문에 Mllama에서 `mlxcel generate --image`가 깨진다는 것을 찾았습니다. CLI는 자기 `SequenceId`로 실행되지만 스케줄러의 bind를 호출하지 않아서, 모델이 이미지를 무시했습니다. 최종 규칙은 `MRopeState`와 같습니다.
- bind되지 않은 id는 fallback을 읽습니다.
- 이미지 없이 bind된 sequence는 명시적 텍스트 전용 항목으로 기록합니다.

이 규칙이면 승인이 실패한 뒤 fallback에 남은 이미지가 bind된 텍스트 전용 행에 닿을 수 없습니다.

### 3.3 lookahead 철회는 Baichuan만이 아니라 엔진에서 고침

lookahead 철회는 기본 backend가 model-owned인 계열에서만 모델 상태를 되감았습니다. Baichuan은 dense라서 K/V는 스케줄러 slice에 있지만 conv 상태는 그 밖에 있습니다. 새 trait hook은 정확히 이 성질을 표현합니다. 같은 판정이 prompt-cache와 handoff 게이트도 결정하므로, 앞으로 KV 밖에 상태를 둔 dense 모델은 질문 하나에만 답하면 세 경로 모두에서 올바르게 동작합니다.

기각한 대안은 Baichuan의 lookahead를 끄는 것이었습니다. 정확하긴 하지만 느리고, 같은 성질을 가진 다음 모델은 여전히 노출된 채로 남습니다.

## 4. 검증과 측정 결과

### 4.1 red-then-green 테스트

새 테스트는 모두 수정 전에 실패 상태로 먼저 커밋했습니다.
- Baichuan: 연속 생성 사이 conv 상태 상속, 교차 실행된 sequence의 불일치, lookahead 철회 후 재실행이 동기 스텝과 다름(1.09e-3), prefill projection이 붙잡힘(64 MiB 잔존).
- Mllama: 이미지 sequence 뒤의 텍스트 전용 sequence, prefill 전에 bind된 이미지, bind되지 않은 CLI sequence의 fallback 읽기.
- DeepStack, 세 텍스트 모델 각각: bind된 항목, 텍스트 전용 offset-0 forward.

### 4.2 실제 체크포인트(M1 Ultra, release 빌드)

서버 검증은 `--parallel 4`에 서로 다른 요청 두 개를 동시에 보낸 뒤, 이미지 요청 다음에 텍스트 전용 요청을 보냅니다. 모두 greedy로 생성하고, 같은 요청을 새로 띄운 서버에 단독으로 보낸 결과와 비교합니다.

| 체크포인트 | `main` 01047d75 | 브랜치 66888608 |
|---|---|---|
| Qwen3-VL-4B-Instruct-4bit | B 불일치 | 동일 |
| Llama-3.2-11B-Vision-Instruct-4bit | B가 A의 이미지를 설명, 이미지 뒤 텍스트 전용 요청 불일치 | 동일 |
| Baichuan-M1-14B-Instruct-4bit | 전부 불일치 | `max_tokens` 40과 48 모두 동일 |

CLI 단일 요청 출력(`--temp 0`)은 Qwen3-VL과 Mllama에서 텍스트와 이미지 프롬프트 모두 `main` 1cb18ccc와 브랜치가 동일합니다. Baichuan-M1의 단일 요청 출력은 의도적으로 바뀌었습니다. `# # # ...`에서 정상 문장으로 바뀌었고, 내용은 mlx-lm 출력과 일치합니다. 이 비교는 mlx-lm으로 가중치를 읽고 sentencepiece로 직접 토크나이즈한 것으로, 토큰 단위 일치가 아닌 정성적 확인입니다.

## 5. 학습 포인트

- **마지막 토큰에서만 나는 불일치는 추적할 가치가 있는 증상입니다.** 결정적으로 재현됐고, logprobs로 동기 decode를 강제하면 사라졌고, Qwen3-VL에서는 나타나지 않았습니다. 이 조합이 수치 오차가 아니라 lookahead 철회 경로를 가리켰습니다.
- **MLX slice는 view입니다.** 큰 projection에서 한 토큰만 잘라 저장해도 버퍼 전체가 살아 있으므로, forward보다 오래 사는 상태는 contiguous로 만들어야 합니다.
- **단일 요청 동일성 확인은 변경이 건드리는 입력 모드를 모두 포함해야 합니다.** 첫 비교는 텍스트 프롬프트만 써서 Mllama CLI 이미지 회귀를 놓쳤습니다.

## 6. 후속 조치와 제한 사항

- `--enable-vlm-prefix-cache`에서 미디어 준비 후 승인이 실패하면 fallback 슬롯이 채워진 채 남습니다. 모든 계열에 원래 있던 문제이고 이 PR 범위 밖입니다.
- Cohere Compass는 로컬 체크포인트가 없어 synthetic 테스트가 게이트입니다.
- 에픽 #2566 Metal 수용 검증(#2587)에서 영향받는 VLM 항목을 최종 통합 리비전으로 다시 실행해야 합니다.

## 참고 자료

- 이슈 #2567, 에픽 #2574, 관련 #2557, #2564, #2566, #2587
- `src/vision/mllama_image_state.rs`, `src/models/deepstack_state.rs`, `src/models/baichuan.rs`, `src/lib/mlxcel-core/src/engine/lookahead.rs`
