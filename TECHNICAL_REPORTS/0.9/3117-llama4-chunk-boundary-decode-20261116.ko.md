# 기술 보고서: PR #3117 - Llama 4 chunked decode를 query가 속한 chunk 안으로 제한

**작성일**: 2026-11-16
**상태**: 완료 (mlx-lm과 다른 Llama 4 RoPE 문제를 별도로 발견해 따로 추적)
**위험도**: 높음 (8192 토큰 이후 decode 출력이 reference 쪽으로 바뀜)

## 요약

Llama 4의 chunked(local) 층은 decode에서 mask 없이, cache가 들고 있는 마지막 8192개 key 전체에 attention했다. 그래서 위치 8192를 넘으면 이전 chunk의 key까지 읽었다. mlx-lm은 매 단계마다 `block_pos == 0` chunk mask를 다시 만들기 때문에, 각 query는 자기 chunk 안에서만 attention한다(#3105). 이제 `ChunkedKVCache::attend`는 decode 단계의 window를 query가 속한 chunk의 시작 위치 `floor(p / chunk) * chunk`부터 잘라 쓴다. 다중 토큰 단계는 #3100의 chunk별 causal 분할을 그대로 쓰며, 이 분할 코드도 같은 함수로 옮겼다. single-row, batched rows, batched decode 경로가 모두 이 함수 하나를 거친다.

llama-4-scout에서 경계 이후 구간의 teacher-forced trace를 보면, decided position 불일치가 75개 중 20개에서 0개로 줄었다. 위치 8192~8448의 perplexity는 27.87에서 reference 값인 5.86이 됐고, scalar decode와 B=2 batched decode 모두 같은 결과다. 긴 문맥 decode는 4% 빨라졌는데, 경계를 넘은 뒤 chunked 층이 8192개가 아니라 최대 286개 key에만 attention하기 때문이다.

## 3. 기술적 선택과 그 이유

- **mask를 만들지 않고 잘라낸다.** decode에서 chunk 제한은 window의 앞부분을 잘라내는 것과 같다. 그래서 기존의 mask 없는 SDPA가 더 적은 key로 돌고, mask 배열은 만들지 않는다.
- **기준은 mlx-lm의 규칙이다.** trace의 reference는 mlxcel의 one-pass prefill이며, 이 결과는 RoPE를 mlxcel 방식으로 맞춘 mlx-lm과 decided position 941개 모두에서 일치한다.

## 7. 변경 요약

| 셀 | decode tok/s, main → 브랜치 | 출력 |
|---|---|---|
| 554 토큰 프롬프트, 128 토큰 | 45.55 / 44.94 → 45.56 / 44.77 | 동일 |
| 8077 토큰 프롬프트, 400 토큰 | 27.04 / 27.12 → 28.09 / 28.37 | 문맥이 8192를 넘은 뒤부터 달라짐 (수정된 동작) |

새 테스트는 main에서 실패하고 브랜치에서 통과한다.
- 세 경계를 지나는 chunked decode를 host에서 만든 mlx-lm chunk mask와 비교한다.
- 작은 모델에서 chunk 경계를 지나는 decode를 one-pass prefill과 비교한다.
- chunked attend 결과를 mlx-lm mask reference와 비교한다.

`mlxcel-core` lib suite(2,319개)와 `llama4` 테스트도 통과한다.

## 8. 후속 조치

기본 설정의 mlx-lm은 mlxcel의 Llama 4와 모든 문맥 길이에서 decided position의 46%가 다르다(perplexity 1.18 대 14.05). mlx-lm에 mlxcel의 RoPE를 넣으면 이 차이가 사라진다. mlx-lm은 `traditional=True`와 checkpoint의 `llama3` rope scaling을 쓰는데, mlxcel의 Llama 4 attention은 이를 적용하지 않는다.
