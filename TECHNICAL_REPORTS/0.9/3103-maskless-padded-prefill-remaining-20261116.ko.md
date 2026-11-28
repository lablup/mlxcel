# 기술 보고서: PR #3103 - 남은 계열을 maskless padded prefill에 opt-in

**작성일**: 2026-11-16
**상태**: 완료
**위험도**: 중간 (padded embedding prefill에서 GLM4V MRoPE 위치가 바뀜)

## 요약

#3097에서 빠졌던 마지막 네 계열이 이제 `supports_maskless_padded_prefill`에 opt-in한다. 그래서 padded 또는 batched prefill piece가 dense `[B, 1, L, L + offset]` padding mask를 만들지 않는다(#3101).
- **Cohere2-MoE와 EXAONE-MoE:** forward가 이미 mask 없는 piece를 처리하고 있었다.
- **GLM4V와 GLM4V-MoE:** text backbone과 VLM wrapper가 함께 opt-in한다.
- **Nemotron-NAS:** 새 `trim_state`가 모델이 직접 관리하는 cache를 trim한다. 이전에는 padded prefill 자체를 지원하지 않았다.

작업 중에 masked 경로에도 있던 버그를 고쳤다. GLM4V의 padded embedding prefill은 모든 행에 `index + rope_deltas`를 썼다. 저장된 이미지 위치가 실제 프롬프트 구간만 덮기 때문이다. 그 결과 이미지 뒤에 오는 실제 토큰이 잘못된 위치를 받았다. 이제 `glm4v::padded_tail_positions`가 실제 토큰에는 저장된 행을 쓰고, pad 행에는 그 다음 위치부터 이어서 붙인다. 이 함수를 끄면 새 GLM4V와 GLM4V-MoE parity 테스트가 실패한다.

## 7. 변경 요약

| 검사 | maskless 대 masked (prefill / decode) |
|---|---|
| 작은 GLM4V와 GLM4V-MoE, 이미지 위치 유무, ids와 embeddings | 0 / 0 |
| 작은 Nemotron-NAS, Cohere2-MoE, EXAONE-MoE | 0 / 0 (EXAONE-MoE의 window를 넘는 경우: 순수 causal padding mask에는 window가 없어서 maskless를 unpadded와 비교, 0 / 0) |
| glm-4.1v-9b, llama-3.3-nemotron-super-49b, glm-4.5v checkpoint | 0 / 0 |

테스트: 전체 `mlxcel` lib suite가 직렬로 통과한다(11,955개). roster 테스트도 통과한다.

## 8. 후속 조치

GLM4V 위치 수정이 padded batch의 출력을 어떻게 바꾸는지 실제 이미지 프롬프트로 측정하지는 않았다. GLM-OCR은 손대지 않았다.
