# 기술 보고서: PR #1540 - Inkling 네이티브 MTP

**작성일**: 2026-08-31

**작성자**: mlxcel maintainers

**상태**: 구현 및 결정론적 검증 완료. 실제 체크포인트 검증은 보류

**언어**: Rust, Markdown

**위험 수준**: 높음

---

## 요약

PR #1540은 Inkling의 chained multi-token-prediction head를 사용하는 네이티브 B=1 speculative decoding을 추가한다. 원본 `model.mtp.layers.*` 텐서를 로드하고 타깃과 같은 Inkling decoder-layer 구현을 재사용하며, 타깃의 embedding·final norm·LM head를 바인딩한다. 타깃의 pre-norm hidden state를 캡처하고 snapshot 및 accepted-prefix replay로 KV와 네 개 convolution state를 정확히 복원한다.

또한 standalone Inkling text target과 공개 멀티모달 체크포인트가 로드되는 `InklingVLM` variant 모두에 offline 및 server dispatch를 등록하고 기본 verify 폭을 `num_nextn_predict_layers + 2`로 계산하며, 별도 4.46 GB `mtp.safetensors`만 안전하게 받는 선택적 downloader 경로를 추가했다. Wrapper의 text-only speculative decode는 `vlm.text`를 사용하고 image 요청은 기존 HMLP prepared-embedding prefill을 유지한다. Head-only 디렉터리를 standalone target으로 사용하면 실행 방법을 포함한 명확한 오류를 반환한다. 결정론적 tiny-model 테스트로 chained forward의 유한성, target block과 incremental 실행의 parity, partial-accept rollback 뒤 bitwise continuation equality, wrapper dispatch 완전성, image prefill 보존을 확인했으며, 검증 호스트에 없던 실제 체크포인트 결과는 주장하지 않는다.

## 1. 문제 정의

Inkling-Small은 32개 target shard 옆에 MTP head를 별도 safetensors 파일로 배포하지만 일반 MLX 변환본에는 이 head가 없다. 기존 MTP 구현은 Inkling의 learned relative-position attention과 네 개의 recurrent short-convolution state를 지원하지 않았다. Speculative verify forward 뒤에는 KV tail 길이만 줄이는 방식으로 recurrent state를 되돌릴 수 없다.

구현은 config flag가 아니라 실제 MTP 텐서를 구분해야 했고, 원본 전체 저장소를 drafter 디렉터리로 지정해도 수백 GB target shard를 열지 않아야 했다. B=1의 모든 진입점도 보존해야 했다. 반복되는 server adapter match 중 하나라도 빠지면 특정 scheduler 경로만 조용히 classic decoding으로 되돌아간다.

## 2. 변경 요약

| 영역 | 결과 |
| --- | --- |
| 공통 decoder | Inkling attention, short convolution, cache, decoder shell, dense MLP primitive를 `mlxcel-core`로 이동했다. 타깃은 dense-or-sparse MLP를, 모든 MTP block은 dense 구현을 주입한다. |
| Drafter | 검증된 config fallback, 원본 체크포인트 sanitizer, chained block 실행, target module binding, prompt seed prefill, round snapshot, accepted-token replay를 추가했다. |
| Target | Pre-norm hidden capture, 정확한 pre-verify state capture, block verify, snapshot restore, accepted-plus-bonus replay, B=1 linear-only capability 경계를 추가했다. |
| 로딩 | Indexed target shard를 열지 않으면서 unindexed auxiliary safetensors를 검사하는 index-aware filtered loading 및 header-only tensor detection을 추가했다. |
| Dispatch | Offline generation, 기존 server burst, tick-cooperative slice start/step/park/finalize, drafter binding, source-scanned coverage test에 `LoadedModel::Inkling`과 `LoadedModel::InklingVLM`을 모두 등록했다. |
| CLI | 기본 `K = n + 2`, 명시적 override 우선순위, 기존 안전 path/type allow-list 뒤에 적용되는 반복 가능한 `--include` glob filter를 추가했다. |
| 감지 | Drafter auto-detection에 실제 MTP 텐서를 요구하고 `config.json`과 `mtp.safetensors`만 있는 디렉터리를 standalone model로 거부했다. |
| 문서 | 정확한 선택 다운로드·실행 예제, B=1/tree 제한, rollback 의미론, 검증 한계를 추가했다. |

## 3. 기술적 선택과 이유

### 3.1 하나의 decoder 구현 재사용

`hidden_norm`, `embed_norm`, concatenation, `2H -> H` projection 다음의 MTP transform은 일반 Inkling decoder layer다. Generic `InklingDecoderLayer<M>`이 attention, residual short convolution, norm, cache 동작을 소유하고, `InklingFeedForward`가 타깃의 dense/sparse plane 또는 MTP head의 dense plane을 선택한다. Recurrence와 relative-attention 로직의 두 번째 복사본을 없애 타깃과 drafter의 수정이 독립적으로 어긋나지 않게 했다.

### 3.2 Trim 대신 restore와 replay 사용

각 target layer는 KV cache와 네 개의 causal convolution state를 가진다. Verify block은 다섯 state component를 모두 변경하며 token count만 줄여 convolution tail을 재구성할 수 없다. Adapter는 전체 pre-verify state를 캡처하고 block을 한 번 실행한 뒤 snapshot을 복원하고 정확히 `accepted + 1`개 입력을 replay한다. Drafter도 proposal 전에 모든 block cache를 독립적으로 snapshot하고, 복원 뒤 accepted draft와 새 bonus를 block 0으로 replay한다.

Round마다 짧은 replay 비용이 들지만 greedy invariant를 보존한다. 결정론적 regression은 다음 hidden state와 logits를 한 번도 speculate하지 않은 target과 bitwise 비교한다.

### 3.3 Model index를 완전한 inventory가 아니라 최적화로 취급

공개 Inkling-Small index에는 32개 target shard만 있고 `mtp.safetensors`는 top-level unindexed auxiliary file이다. Filtered loader는 먼저 일치하는 indexed shard를 고른 뒤 index에 없는 top-level safetensors만 검사한다. 공개 구조에서는 MTP head만 열고 indexed target shard는 모두 건너뛴다. Header detection은 크기를 제한하고, index에서 선택한 shard 이름에는 plain-filename path-traversal 검증을 유지하며, downloader include pattern은 기존 safe relative-path 및 file-type 검사 뒤에만 적용한다.

### 3.4 지원하지 않는 shape를 명시적으로 제한

Inkling MTP는 B=1만 구현한다. B>1은 classic decode로 되돌아가며, recurrent convolution state는 attention mask만으로 sibling branch를 분리할 수 없으므로 target은 tree-aware verify capability를 제공하지 않는다. 요청한 block size는 linear chain으로 유지하고 native chain 깊이를 넘으면 마지막 MTP block을 재사용한다.

## 4. 리뷰와 보강

정확성·보안·성능·finalizer 리뷰를 통해 handoff 전에 다음을 보강했다.

- 초기 indexed-only fixture가 공개 전체 저장소를 놓치던 문제를 수정해 target-only index와 unindexed `mtp.safetensors` 실제 구조를 로딩·감지한다.
- Source-scanned dispatch test가 드러낸 반복 adapter 경계를 따라 모든 tick-cooperative scheduler site에 Inkling을 추가했다.
- Glob include를 downloader safe allow-list 아래에 유지하고 잘못된 pattern은 cache 또는 network reuse 전에 거부한다.
- 보이는 sliding-window slab과 함께 absolute KV offset 및 네 개의 optional convolution tensor를 정확히 보존한다.
- 불필요한 raw-pointer concatenate를 안전 wrapper로 바꾸고 남은 attention FFI lifetime invariant를 문서화했다.
- 구현 파일을 500줄 미만으로 분리하고 decoder를 복제하지 않고 target primitive를 재사용했다.
- PR #1535와 branch를 조정해 HMLP image detection/loading 및 MTP-only detection을 모두 유지했다.
- 제출 후 HIGH finding을 수정했다. 공개 체크포인트는 `LoadedModel::InklingVLM`으로 로드되지만 최초 MTP gate는 `LoadedModel::Inkling`만 허용했다. 전용 wrapper adapter가 CLI, 기존 burst, cooperative slice start/step, drafter bind/return, finalization의 text-only speculative 작업을 모두 `vlm.text`로 전달한다.
- 기존 multimodal request gate로 image 요청이 MTP에 들어가지 않게 유지하고, classic wrapper가 HMLP로 병합한 prepared embedding을 대체하거나 다시 정규화하지 않고 `vlm.text`에 직접 전달함을 tiny real-HMLP regression으로 검증했다.

집중 리뷰 결과 해결되지 않은 CRITICAL 또는 HIGH 정확성·보안·성능 문제는 남지 않았다.

## 5. 검증

| 게이트 | 결과 |
| --- | --- |
| `cargo check --lib` | 통과 |
| `cargo test -p mlxcel-core inkling_mtp --lib` | 통과, 7/7 |
| `cargo test --lib inkling -- --test-threads=1` | 통과, 48/48. Target, HMLP, detection, 양쪽 variant MTP dispatch, raw-image classic gate, prepared-image prefill |
| `cargo test --lib every_mtp_dispatch_site_covers_every_capable_variant` | 통과 |
| `cargo test --lib burst_declined_for_vlm_embeddings` | 통과. Multimodal 요청은 classic prefill 유지 |
| `cargo test --lib resolve_draft_block_size_derives_inkling_default_from_the_mtp_layer_count` | 통과 |
| `cargo test --lib isolated_inkling_mtp_download_is_not_a_standalone_model` | 통과 |
| `cargo test --lib include_globs_select_only_safe_allow_list_files` | 통과 |
| `cargo clippy -p mlxcel-core --lib -- -D warnings` | 통과 |
| `cargo clippy --lib -- -D warnings` | 통과 |
| `cargo clippy --lib --tests -- -D warnings` | 통과 |
| `cargo fmt --all -- --check` | 통과 |
| `git diff --check` | 통과 |

Core suite는 config fallback과 local/global block 파생, config-only negative detection, actual-tensor detection, 실제 구조 index skipping, sanitizer mapping, 5-token finite logits, 정확한 flat KV 및 네 convolution 복원을 검증한다. Target suite는 block verify와 incremental argmax 비교 및 partial acceptance 뒤 bitwise continuation을 검증한다.

## 6. 검증 한계와 후속 작업

약 153.5 GB Inkling-Small target, 4.46 GB MTP head, 0.6B 체크포인트를 검증 호스트에서 사용할 수 없었다. 따라서 실제 체크포인트의 128-token greedy parity, 평균 accepted length, throughput, peak memory, Apple GPU 동작은 검증하지 않았고 이 보고서도 이를 주장하지 않는다.

넓은 workspace Metal/Accelerate test와 all-target clippy gate는 epic-level 최종 검증에 맡겼다. B>1 acceptance, tree verification, standalone split tool, 성능 최적화는 issue #1315 범위 밖이다.

## 참고

- 에픽 #1313, 이슈 #1315, 선행 이슈 #1318
- PR #1540
- [공개 mlx-vlm Inkling MTP 레퍼런스](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/speculative/drafters/inkling_mtp/inkling_mtp.py)
- [thinkingmachines/Inkling-Small](https://huggingface.co/thinkingmachines/Inkling-Small)
- `docs/supported-models.md`
