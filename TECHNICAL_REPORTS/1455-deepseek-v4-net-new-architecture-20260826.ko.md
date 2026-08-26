# 기술 보고서: PR #1455 - DeepSeek-V4를 별개 아키텍처로 이식

**작성일**: 2026-08-26
**작성자**: mlxcel maintainers
**상태**: 완료. `mlx-community/DeepSeek-V4-Flash-4bit`에서 실제 체크포인트 게이트 3개가 모두 통과했고, 리뷰 보강 전후의 생성 텍스트가 바이트 단위로 같다
**언어**: Rust, Markdown
**위험도**: Medium

---

## 요약

PR #1455는 이슈 #523을 구현한다. 저장소 안의 레퍼런스 `references/mlx-vlm/mlx_vlm/models/deepseek_v4/`를 기준으로 DeepSeek-V4를 이식했다. `src/models/deepseek_v4*.rs` 열 개 파일과 실제 체크포인트 통합 테스트, 그리고 등록 경로를 추가한다.

여기서 남길 만한 기록은 "모델을 하나 추가했다"가 아니다. 같은 이슈의 첫 시도가 왜 성립할 수 없었는지, 그리고 코드를 쓰기 전에 계열을 어떻게 분류해야 하는지가 핵심이다. PR #592는 V4를 V3의 변종으로 보고 `DeepSeekV4Model`을 `DeepSeekV3Model` 위의 얇은 래퍼에 델타 두 개를 얹은 형태로 만들었다. 머지되지 못하고 닫혔다. V4는 V3와 구조를 거의 공유하지 않는다. 잔차 연결을 바꾸고, MLA를 바꾸고, 라우팅 규칙을 바꾸고, 블록 사이로 흐르는 은닉 상태의 랭크까지 바꾼다. 래퍼로는 실제 V4 체크포인트를 적재조차 할 수 없다.

## 1. 문제 정의

### 1.1 배경

mlxcel에는 `deepseek`, `deepseek_v2`, `deepseek_v3`, `deepseek_v32` 네 구현이 있었다. 공통 코어 레이어만 공유하는 별개 구현이다. 이 순서를 보면 `deepseek_v4`도 같은 계열의 다섯 번째 구성원이고 설정값과 기능 한둘만 다를 것이라고 짐작하기 쉽다.

그 짐작은 틀렸는데, 설정 파일만 봐서는 잘 드러나지 않는다. `deepseek_v4`의 `config.json`에는 익숙한 키(`q_lora_rank`, `qk_rope_head_dim`, `n_routed_experts`, `routed_scaling_factor`, `norm_topk_prob`, `topk_method: "noaux_tc"`)와 낯선 키(`hc_mult`, `hc_sinkhorn_iters`, `compress_ratios`, `num_hash_layers`, `o_groups`, `index_topk`)가 함께 들어 있다. "V3에 몇 가지가 더 붙었다"고 읽는 것도 첫 판단으로는 무리가 아니다. 이 판단을 뒤집는 근거는 레퍼런스 구현을 직접 읽는 데서 나온다.

### 1.2 V3 래퍼 전제가 무너지는 지점

V4를 구분하는 요소는 다섯 가지이고, 전부 선택 사항이 아니라 핵심이다.

**HyperConnection이 평범한 잔차 연결을 대체한다.** 블록 사이로 넘어가는 상태는 `[B, L, D]`가 아니라 `hc_mult = 4`인 랭크-4 텐서 `[B, L, hc_mult, D]`다. 임베딩 출력을 입구에서 네 벌로 복제하고, 마지막 정규화 직전에 `HyperHead`가 다시 하나로 접는다. 서브레이어마다 학습된 게이트가 `pre` 벡터, `post` 벡터, 그리고 `hc_mult x hc_mult` 혼합 행렬을 만들고, 혼합 행렬은 소프트맥스를 거친 뒤 20회 교대 Sinkhorn 정규화를 받는다. V3 블록에는 이것을 끼워 넣을 자리가 없다. 스택을 흐르는 텐서의 모양 자체가 달라지기 때문이다.

**풀링 KV `Compressor`가 MLA를 대체한다.** V3는 토큰마다 저랭크 잠재 벡터로 KV를 압축한다. V4는 토큰 윈도 전체를 압축 행 하나로 풀링하고, 윈도 크기는 레이어별로 `compress_ratios`가 정하며, 그 옆에 작은 지역 KV 윈도를 따로 유지한다. 같은 기법의 다른 설정이 아니라 서로 다른 기법이다.

**HiSA 희소 선택.** 희소 레이어는 자체 압축기를 가진 `Indexer`를 두고, 풀링된 행에 점수를 매겨 상위 `index_topk`개를 고른다. 선택 결과는 지역 KV와 로그 정규화 상수를 공유하는 분할 소프트맥스로 들어간다.

**해시 라우팅 MoE와 `sqrtsoftplus` 게이팅.** 앞쪽 `num_hash_layers`개 레이어는 점수로 라우팅하지 않는다. 전문가 인덱스가 원본 토큰 id로 색인하는 `tid2eid` 조회 테이블에서 나온다. V3의 그룹 제한 소프트맥스 라우팅은 아예 없다.

**`MultiLinear` 그룹 출력 투영.** 그 앞에는 어텐션 출력에 적용하는 역방향 RoPE가 붙는다.

### 1.3 위험 평가

이 이식의 실패 양상은 크래시가 아니다. 다섯 요소 각각을 그럴듯하지만 틀리게 구현해도 유한하고 모양이 맞고 유창한 출력이 나온다.

- Sinkhorn 반복 횟수를 틀리거나 행/열 정규화 순서를 바꿔도 그럴듯한 이중 확률 행렬이 나온다.
- 겹침 압축기(비율 4)와 단순 압축기(비율 128)는 출력 모양이 같다.
- 라우팅 가중치를 편향 없는 점수가 아니라 편향 보정 점수에서 모으면, 모든 라우팅 기여가 잘못 가중되면서도 모델은 멀쩡히 말이 된다. `bailing_moe`, `afmoe`, `klear`가 문서에 남긴 것과 같은 계약이다.
- 인덱서가 엉뚱한 행을 고르면 긴 문맥 회상만 나빠지고 짧은 프롬프트에는 아무 영향이 없다.

그래서 이 이식은 종단 간 일관성만이 아니라 구성 요소 단위 패리티 픽스처로 게이트를 잡았다. 실제 체크포인트 게이트 중 긴 문맥 항목의 비중이 가장 큰 이유도 같다.

## 2. 기술적 선택과 그 이유

### 2.1 Metal 커널이 아니라 순수 연산 Sinkhorn 경로를 이식

레퍼런스는 HyperConnection 게이트를 두 가지로 구현해 둔다. 융합 `mx.fast.metal_kernel`(`_hc_sinkhorn_collapse_kernel`)과 순수 연산 경로(`_hc_ops` / `_hc_split_sinkhorn_ops`)다. 레퍼런스 자신도 학습할 때와 Metal이 없을 때는 연산 경로를 쓴다. 즉 연산 경로가 정확성 기준이고 커널은 그것의 최적화다.

이 이식은 연산 경로만 구현한다. 처음 이식하는 아키텍처에 Metal 커널까지 함께 쓰면, 새 알고리즘과 새 커널을 동시에 디버깅하게 되고 둘 중 어느 쪽에도 독립적인 기준이 없다. 커널은 후속 작업으로 남겼다.

### 2.2 모델이 소유하는 이종 레이어 캐시 상태

`LanguageModel::forward`는 레이어당 하나씩 동종 항목을 담은 `caches: &mut [KVCache]`를 받는다. V4는 레이어마다 회전 윈도 KV 캐시 하나에 풀링 캐시 한두 개가 더 필요하고, 개수는 그 레이어의 `compress_ratio`에 따라 달라진다. `PoolingCache`는 Rust 쪽에 대응물이 없고, 그 의미론(나머지 버퍼, 프롬프트/디코드 분기 구분, 윈도가 다 찼을 때만 방출)은 `KVCache`에 맞지 않는다.

트레이트를 넓히는 대신 `src/models/model_owned.rs`의 `ModelOwnedSequenceState<T>`를 썼다. `mamba2`, `gemma3`, `afmoe`, `qwen3_next`가 비-KV 상태나 이종 상태에 쓰는 바로 그 우회로다. 트레이트 호환을 위해서는 자리표시자 `KVCache::new()`를 돌려준다. 덕분에 모델별 캐시 모양이 공용 트레이트로 새지 않지만, 대신 이 모델은 페이지드 KV 경로와 양자화 KV 경로에 참여하지 못한다.

### 2.3 어텐션 구조체를 셋이 아니라 하나로

레퍼런스는 `LocalAttention`, `CompressedAttention`, `SparseCompressedAttention` 세 클래스와 팩토리를 둔다. 셋은 투영 집합이 전부 같고, 압축기가 있는지, 인덱서가 있는지, 순전파가 어느 분기를 타는지만 다르다. 이 이식은 종류 판별자를 가진 구조체 하나로 합쳤다. 공유 가중치 적재가 한곳에 모이고, 세 분기가 서로 붙어 있어 눈에 들어온다.

### 2.4 상위 설정을 믿지 말고 경로별 양자화 재정의 표를 읽기

실제 체크포인트는 최상위에 `{group_size: 64, bits: 4, mode: "affine"}`를 선언하고, 그 아래 경로별 재정의를 641개 둔다. 그중 129개, 즉 라우팅 전문가 투영 전부가 `{group_size: 32, bits: 4, mode: "mxfp4"}`다. 나머지 512개는 최상위 값을 그대로 반복한다.

최상위만 읽는 적재기는 mxfp4/32 전문가 텐서에 affine/64를 적용한다. 이 이식은 재정의 표를 읽는다. 선언된 값 쌍도 모두 검증한다. MLX가 해석하지 못하는 모드 문자열은 적재 오류가 아니라 첫 순전파에서 잡을 수 없는 `std::terminate`가 되기 때문이다.

### 2.5 `compress_ratios`는 길이를 맞추라고 요구하지 말고 잘라 쓰기

실제 체크포인트는 레이어 **43개**에 `compress_ratios` 항목 **44개**를 싣는다. 레퍼런스의 `__post_init__`은 앞에서부터 `num_hidden_layers`개만 남기는데, 그러면 끝의 `0`이 잘려 나가고 마지막 레이어는 비율 4, 즉 희소 레이어가 된다.

여기서 그럴듯한 두 해석이 모두 실패한다. `len == num_hidden_layers`를 요구하면 실제 체크포인트를 적재 단계에서 거부한다. 뒤에서 43개를 취하면 스케줄 전체가 한 칸 밀려 모든 레이어에서 어텐션 종류가 어긋나는데, 적재는 깨끗하게 끝나고 유창한 헛소리가 나온다. 이 이식은 레퍼런스를 따르고, 인덱스 42에 대한 단언으로 동작을 고정했다.

### 2.6 적재 시점의 엄격한 가중치 커버리지 게이트

`from_weights`는 어느 모듈 경로에도 대응되지 않는 체크포인트 텐서, 그리고 대응 텐서를 찾지 못한 모듈 파라미터를 이름과 함께 보고하며 적재를 실패시킨다. 텐서 이름 평면이 둘인 아키텍처를 처음 이식하는 상황에서, 맞지 않는 키를 조용히 넘기는 것과 실패시키는 것의 차이는 "이 체크포인트는 지원하지 않는다"와 "무작위 초기화된 서브모듈을 얹은 채 돌아간다"의 차이다.

## 3. 구현 상세

| 파일 | 내용 |
| --- | --- |
| `deepseek_v4.rs` | 설정 파싱과 검증, 블록/모델 조립, 모델 소유 시퀀스 상태, `LanguageModel` 구현 |
| `deepseek_v4_hyper.rs` | `HyperConnection`, `HyperHead`, `hc_expand`, float32 Sinkhorn 연산 경로 |
| `deepseek_v4_rope.rs` | inf 패딩 비례 RoPE, Yarn 보정, 역방향 테이블, `freq_scale` 분할 |
| `deepseek_v4_compress.rs` | `PoolingCache`, 단순/겹침 압축기, 풀링 가시성 마스크 |
| `deepseek_v4_indexer.rs` | HiSA 인덱서: 디코드 고속 경로, 배치 타일 경로, 평면 폴백 |
| `deepseek_v4_attention.rs` | 어텐션 세 종류, 싱크, 역방향 출력 RoPE, `MultiLinear` 출력 경로, 분할 소프트맥스 희소 어텐션 |
| `deepseek_v4_moe.rs` | 해시/편향 라우팅, `sqrtsoftplus`, 제한 SwiGLU, 경로별 전문가 양자화 |
| `deepseek_v4_sanitize.rs` | 텐서 이름 평면 둘, `mtp.*` 제거, 엄격한 커버리지 검사 |
| `deepseek_v4_tests.rs` | 단위 테스트 39개 |
| `tests/deepseek_v4_real_model.rs` | 실제 설정 예비 검사와 ignore 게이트 3개 |

### 3.1 Sinkhorn 순서

이 순서는 대칭이 아니어서 틀리기 쉽다. 레퍼런스는 마지막 축 소프트맥스, `hc_eps` 더하기, **초기 열 정규화**, 그다음 `hc_sinkhorn_iters - 1`회의 (행 정규화, 열 정규화)다. 이식본은 이 순서를 그대로 따르고, float64 픽스처가 레퍼런스의 산술을 재현해 확인한다. 여러 순서가 동시에 만족할 만한 성질을 단언하는 대신 값을 직접 비교했다.

### 3.2 겹침 압축기

비율 4에서 압축기는 겹침 모드로 돈다. `out_dim = 2 * head_dim`이고, kv와 게이트를 특징 축에서 반으로 가른 뒤 앞쪽 절반을 윈도 하나만큼 뒤로 밀면서 앞에 0을 채우고(게이트 쪽은 `-inf`), 두 절반을 윈도 축에서 이어 붙인 다음 소프트맥스를 건다. 비율 128에서는 단순 경로가 돈다. 두 경로의 출력 모양이 같아서 모양만 보는 테스트로는 구분되지 않는다. 둘 다 레퍼런스의 float64 재현으로 고정했다.

버그처럼 보이는 결과가 하나 있어 남긴다. 이동은 준비된 배치 안에서만 일어나므로, 디코드 스텝이나 청크 경계에서 완성된 윈도는 앞 윈도의 값 대신 0 접두를 본다. 이식 과정의 오류가 아니라 레퍼런스의 동작이고, 그래서 비율 4의 풀링 행은 설계상 배치에 민감하다.

### 3.3 `input_ids`를 모든 블록으로 넘기기

해시 라우팅은 MoE 레이어마다 원본 토큰 id가 필요하다. 전문가 인덱스가 `tid2eid[input_ids]`에서 나오기 때문이다. 지엽적인 처리가 아니라 블록 루프 전체를 지나는 시그니처 변경이고, 이 계열의 MoE 순전파만 다른 계열에 없는 인자를 하나 더 받는 이유다.

## 4. 테스트 전략

단위 테스트는 1.3절의 관찰을 전제로 짰다. 실패는 크래시가 아니라 그럴듯한 출력으로 나타난다. 그래서 모양이나 유한성을 단언하는 대신, 작은 픽스처 위에서 레퍼런스 산술을 float64로 재현해 수치를 비교한다.

- 레퍼런스 산술 픽스처: Sinkhorn 게이트, `hc_expand`, Yarn 테이블, RoPE 테이블 구성과 정/역 항등, 압축 함수 둘, 제한 SwiGLU, `sqrt(softplus(x))`.
- 계약 테스트: 편향 선택과 무편향 가중, `tid2eid`에서 인덱스를 가져오는 해시 라우팅, 동점이 없는 입력에서 HiSA 계층 선택과 평면 폴백의 일치.
- 대조군을 둔 거부 테스트: 악의적 설정값, 양자화 재정의, 적재 시점 모양 검사, 불완전한 레거시 전문가 평면, 엄격한 커버리지.
- 통합 성격: 세 종류를 모두 쓰는 소형 종단 간 모델, 윈도 정렬 청크 프리필 패리티, 캐시 배리어 중립성.

## 5. 실제 체크포인트 결과

체크포인트는 `mlx-community/DeepSeek-V4-Flash-4bit`, 43레이어, 텐서 2481개, 디스크 기준 약 151 GB다. 호스트는 M3 Ultra 512 GB다.

```
cargo test --test deepseek_v4_real_model --release --features metal,accelerate -- --ignored --nocapture

[deepseek-v4] long-context prompt: 2234 tokens
[deepseek-v4] long-context answer: " The Pacific Ocean is the largest ocean on Earth. The Pacific"
test deepseek_v4_real_model_long_context_hits_sparse_and_compressed_paths ... ok
[deepseek-v4] prompt: "The capital of France is"
[deepseek-v4] greedy continuation (24 tokens): " Paris. The capital of France is Paris. ..."
test deepseek_v4_real_model_loads_and_generates_coherently ... ok
test deepseek_v4_real_model_decode_crosses_pooling_windows ... ok

test result: ok. 3 passed; 0 failed; finished in 80.37s
```

비중이 가장 큰 항목은 긴 문맥 게이트다. 2234 토큰이면 비율 4 레이어의 풀링 개수가 `index_topk`(512)를 넘어서, 희소 레이어가 조밀 연결 폴백이 아니라 HiSA 선택과 분할 소프트맥스 분기를 탄다. 그 상태에서도 문맥 앞쪽의 사실을 찾아와야 한다. 짧은 프롬프트의 일관성은 인덱서에 대해 아무것도 말해 주지 않는다. 애초에 그 경로까지 가지 않기 때문이다.

짧은 탐욕 디코딩에서 문장이 반복되는 것은 베이스 체크포인트를 온도 0으로 디코딩할 때 흔히 나오는 결과다. 이식 결함의 근거도 아니고, 깨끗한 출력이라고 보고할 것도 아니다.

### 5.1 체크포인트 검증에 관한 기록

로컬 체크포인트는 다운로드가 중간에 끊긴 상태로 있었다. 이를 채우는 과정에서 "완료됐다"는 잘못된 신호가 두 번 나왔다. 둘 다 손상된 가중치 위에서 성공적인 실행을 만들어 냈을 신호다.

1. **샤드 네 개가 잘려 있는데 파일 개수 검사는 통과했다.** `curl -C -`와 `--retry`를 함께 쓰면 이어받기가 안정적으로 동작하지 않는다. 내부 재시도가 출력 파일을 다시 열면서 잘라 버려, 수 GB까지 받아 둔 샤드가 조용히 0에 가깝게 되돌아갔다. 파일은 그대로 있었다.
2. **바이트 수 검사도 손상된 데이터를 통과시킬 뻔했다.** 응답 본문을 파일에 이어 붙이면서 `curl -w '%{http_code}'`를 함께 쓴 페처는, `--write-out`이 표준 출력으로 나가는 탓에 재시도 경계마다 `206`이라는 세 바이트를 샤드 한가운데에 적었다. 한 샤드가 선언 크기보다 정확히 3바이트 커진 덕분에 겨우 잡혔다.

믿을 수 있는 검사는 두 가지를 모두 하는 것이다. 모든 파일이 업스트림이 선언한 바이트 수와 일치하는지, 그리고 모든 샤드가 자기 safetensors 헤더와 일관되는지(`filesize == 8 + header_len + max(data_offsets)`). 새로 받은 체크포인트에서 이 게이트를 다시 돌린다면, 실패를 이식 버그로 판단하기 전에 두 검사를 먼저 통과시키는 편이 좋다.

## 6. 검증 요약

| 항목 | 결과 |
| --- | --- |
| `cargo test --lib models::deepseek_v4` | 39/39 |
| `cargo test --lib models::deepseek` (계열 전체) | 68/68, 회귀 없음 |
| `cargo test --lib models::detection` | 43/43, 새 분기 포함 |
| `cargo test --lib loaded_model` / `models::switch_layers` | 5/5, 23/23 |
| `cargo clippy --lib --tests -- -D warnings` | exit 0 |
| `cargo fmt --check` | clean |
| 실제 인덱스 대비 가중치 커버리지 | 텐서 2481개, 누락 0, 미상 0 |
| 양자화 평면 정합성 | 641/641이 `packed_in * 32 == bits * num_groups * group_size` 만족 |
| 실제 체크포인트 게이트 | 3/3 |

실제 체크포인트 게이트는 두 번 돌렸다. 7절의 리뷰 보강 전에 한 번, 후에 한 번이다. 두 실행의 생성 텍스트가 바이트 단위로 같았다. 어느 한 번의 결과보다 이 일치가 더 중요하다. `PoolingCache::update_and_fetch`가 깊은 복사에서 빌림으로 바뀐 것은 디코드 경로의 별칭(aliasing) 변경이고, 별칭 버그는 단위 테스트를 통과한 뒤 긴 디코드를 망가뜨리기 때문이다.

## 7. 리뷰에서 보강한 부분

**체크포인트 데이터로 인한 범위 밖 읽기(HIGH).** `tid2eid`는 모양만 검증하고 내용은 검증하지 않았다. 이 트리에서 전문가 인덱스가 점수 행에 대한 `argpartition`이 아니라 체크포인트 텐서에서 나오는 라우팅 경로는 V4뿐이다. 그래서 `bailing_moe`, `afmoe`, `klear`를 덮는 "인덱스는 구성상 범위 안"이라는 논거가 여기에는 닿지 않는다. 이 인덱스는 `take_along_axis`와 `gather_qmm`으로 들어가고, MLX의 gather는 음수 인덱스만 접어 줄 뿐 잘라 주지 않는다. 적재 시점에 범위를 검사하도록 고쳤다.

**풀링 캐시 깊은 복사(HIGH, 성능).** `PoolingCache::update_and_fetch`가 호출마다 풀링 버퍼 전체를 실체화했다. 디코드를 지배하는 무작업 분기까지 포함해서다(비율 4에서 네 스텝 중 세 번, 비율 128에서 128번 중 127번). 8k 문맥에서 토큰당 대략 54 MB다. 레퍼런스처럼 빌림을 돌려주도록 바꿨다.

**미평가 그래프 누적(MEDIUM, 성능).** 인덱서의 풀링 캐시가 디코드 스텝마다 `slice_update` 노드를 하나씩 쌓았고, 강제 평가가 없어 각 스텝의 은닉 상태를 붙잡고 있었다. 이 캐시의 유일한 소비자가 희소 분기 셋 중 둘에서 버려지기 때문이다. 레이어 루프 뒤의 `eval_state`로 `KVCache::eval_state`와 같은 형태를 맞췄다.

**검사 없이 MLX로 흘러가던 체크포인트 데이터(MEDIUM).** `e_score_correction_bias`의 모양 검사 누락, 검사 없는 2차원 `wo_a` 재구성, 하한만 있고 상한이 없던 아키텍처 스칼라, 그리고 `i32`로 곱해지면서도 검증되지 않던 `index_block` / `index_keep`.

## 8. 검증되지 않은 부분

- **레퍼런스와의 토큰 단위 일치 비교는 하지 않았다.** 레퍼런스가 트리 안의 Python/MLX 구현이긴 하지만, 같은 입력으로 이식본과 나란히 돌릴 하네스가 없고 체크포인트가 커서 임시 비교도 저렴하지 않다. 검증은 구성 요소 패리티와 종단 간 동작이지 출력 동일성이 아니다.
- **배치 크기 2 이상**은 실제 체크포인트에서 시험하지 않았다. 비율 4 겹침 압축기가 레퍼런스 설계상 배치에 민감하므로(3.2절), 배치 의존 불일치가 나온다면 이 부근일 가능성이 가장 높다.
- **2200 토큰을 크게 넘는 문맥.** 희소 경로는 밟았지만 설정이 선언한 1M `max_position_embeddings` 근처는 아니고, 긴 문맥에서 `RotatingKVCache`가 한 바퀴 도는 상황도 아니다.
- **출하된 mxfp4/affine 혼합 배치 외의 양자화 조합.** 실제 체크포인트가 하나뿐이다.
- **추측 디코딩, 접두 캐시 재사용, 서버 배치 경로**는 이 계열에서 확인하지 않았다.

## 9. 후속 작업

- 레퍼런스의 융합 HC Sinkhorn+collapse Metal 커널. 지금 들어간 연산 경로에 대한 최적화로 붙인다.
- MTP 드래프팅. 현재 `mtp.*` 텐서는 적재 시 버리고 `num_nextn_predict_layers`는 무시한다.
- 텐서 병렬과 파이프라인 샤딩(레퍼런스의 `shard()`).
- `forward_last_logits` 재정의. 없으면 프리필 청크가 한 행을 뽑으려고 `[1, 2048, 129280]` 로짓 전체를 실체화한다. bf16으로 약 530 MB다.
- `sparse_pooled_attention`의 f32 승격 재검토. 레퍼런스는 활성 dtype을 유지하는 자리라 이 모델에서 가장 큰 중간 텐서에 약 2배가 든다. 수치가 움직이는 변경이라 미뤘다.
- 이슈 #549(HiSA)와 #550(공유 전문가 `swiglu_limit` 클램프)은 폐기된 V3 래퍼 범위에서 만들어졌고 이 이식이 둘 다 포함한다. 사람이 닫도록 열어 두었다.

## 참고

- 이슈 #523, 그리고 그 전제를 바로잡은 닫힌 PR #592
- 레퍼런스 구현: `references/mlx-vlm/mlx_vlm/models/deepseek_v4/`
- `PoolingCache` 레퍼런스: `references/mlx-vlm/mlx_vlm/models/cache.py`
- `docs/adding-models.md`의 Text Model Checklist와 Quantization Parameter Bounds
- `docs/supported-models.md`의 `deepseek_v4` 항목
