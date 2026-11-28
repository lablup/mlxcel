# 기술 보고서: PR #3094 - EXAONE-MoE sliding mask를 dense cache 크기에 맞춤

**작성일**: 2026-11-15
**상태**: 완료
**위험도**: 낮음

## 요약

`src/models/exaone_moe.rs`의 `forward_standard_caches`는 dense `KVCache`로 동작하는데, sliding 층 mask는 window 크기에 맞추는 rotating-cache helper로 만들고 있었다. 청크 continuation이 sliding window 이상 위치에서 시작하면 dense cache에는 mask가 덮는 것보다 많은 key가 들어 있고, MLX가 `broadcast_shapes` 오류(예: `Shapes (6,9) and (1,2,6,11)`)로 프로세스를 abort시킨다. 이제 이 경로는 `create_sliding_window_prefill_mask_dense`를 쓴다. 회귀 테스트는 offset 4, 5, 6, 8에서 시작하는 청크 continuation을 단일 패스 prefill과 비교한다(#3093).

## 1. 문제 정의

#3090 작업 중 발견한 이 문제는 오류를 반환하는 것이 아니라 abort이므로 프로세스 전체가 종료되고, 서버라면 모든 요청이 실패한다. 수정을 되돌리면 새 테스트가 SIGABRT로 이를 재현한다.

## 3. 기술적 선택과 그 이유

dense helper는 모든 key를 남긴 채 window를 적용하므로, 이 경로가 쓰는 cache 종류와 맞는다. rotating mask helper를 쓰는 다른 모델(mellum, exaone4, gpt_oss, ministral3, step3p5, afmoe, mimo_v2_flash, gemma3, deepseek_v4, tensor-parallel gemma3)도 점검했는데, 모두 모델이 직접 관리하는 rotating cache와 함께 쓰고 있어 다른 모델은 바꿀 필요가 없었다.

## 7. 변경 요약

| 항목 | 값 |
|------|-------|
| 변경 파일 | 3개 (`exaone_moe.rs`, `exaone_moe_tests.rs`, `CHANGELOG.md`) |
| 테스트 | `exaone_moe_chunked_continuation_past_the_window_matches_single_pass` (window 4, 11 토큰, 분할 4가지, 단일 패스와 1e-4 이내): 수정 전에는 abort, 수정 후에는 통과 |
| 테스트 묶음 | `exaone_moe`와 `last_row` 테스트 통과 (168개) |
