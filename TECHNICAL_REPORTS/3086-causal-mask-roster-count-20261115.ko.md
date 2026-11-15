# 기술 보고서: PR #3086 - #3083 이후 create_causal_mask 호출자 목록 갱신

**작성일**: 2026-11-15
**상태**: 완료
**위험도**: 낮음

## 요약

#3083에서 DeepSeek V3의 다중 토큰 MLA attention이 causal-mode SDPA로 바뀌면서 `deepseek_v3.rs`는 더 이상 `create_causal_mask`를 호출하지 않는다. 그 결과 main에서 `utils::tests::create_causal_mask_roster_count_matches_repo`가 실패했다. 문서 주석은 호출 파일이 51개라고 적고 있었는데 grep으로는 50개가 나왔기 때문이다. 이제 주석의 숫자는 50개이고, MLA 그룹에는 DeepSeekV3 대신 DeepSeekV2가 들어간다. PR job 중 `mlxcel-core` lib suite를 돌리는 것이 없다. 이 테스트는 빨리 끝나므로, 이후 orchestration 실행에서는 머지 시점 contract 검사에 포함했다.
