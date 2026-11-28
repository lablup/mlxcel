# PR #2647: PaddleOCR, ERNIE, Hunyuan VL의 MRoPE를 시퀀스별로 바인딩

**날짜**: 2026-10-27
**상태**: Metal(M1 Ultra)에서 PaddleOCR-VL bf16, HunyuanOCR 4-bit 검증 완료. ERNIE-4.5-MoE-VL은 단위 테스트로만 검증
**언어**: Rust
**위험도**: 중간

## 요약

PR #2647(이슈 #2634)은 `mlxcel-server`에서 Qwen VL 계열이 이미 갖고 있던 시퀀스별 MRoPE 수명주기를 PaddleOCR-VL, ERNIE-4.5-MoE-VL, Hunyuan-VL에도 적용합니다. 지금까지 admission은 이 세 계열을 바인딩하지 않았기 때문에, 동시에 처리 중인 모든 요청이 `MRopeState`의 fallback 슬롯 하나에서 디코딩했습니다. 이 슬롯에는 가장 최근에 준비된 이미지 요청의 위치 정보가 들어 있습니다. `--parallel 4`로 실제 체크포인트를 돌려 보면, PaddleOCR-VL에서는 동시 이미지 요청과 그 뒤의 텍스트 전용 요청이 단독 실행 결과 대신 같은 숫자를 반복해서 출력했고, HunyuanOCR에서는 텍스트 전용 요청이 단독 실행 결과와 달라졌습니다. 수정 후에는 두 모델 모두 동시 실행 결과가 단독 실행 결과와 일치합니다.

## 1. 문제 정의

세 텍스트 모델은 원래 MRoPE 상태를 `MRopeState`(fallback 슬롯과 `HashMap<SequenceId, MRopeEntry>`)에 저장하고 있었고, 해제도 `release_mrope_sequence`로 이미 연결되어 있었습니다. 빠진 곳은 `LoadedModel`이었습니다. `bind_qwen_vl_mrope_state_to_sequence`, `take_qwen_vl_mrope_entry`, `install_qwen_vl_mrope_entry`가 `VlmRuntimeRef::Qwen`과 `Qwen3OmniMoe`만 처리했기 때문에, 나머지 세 런타임에서는 바인딩이 아무 경고 없이 no-op이 되었습니다. `MRopeState::with_entry`와 `batched_decode_positions`는 등록되지 않은 id를 fallback으로 해석하므로 오류가 드러나지 않았고, 요청들이 위치 정보를 공유하게 되었을 뿐입니다.

PR #2630은 이 세 계열을 `clear_vlm_fallback_state()`에서 의도적으로 제외했습니다. 이들이 fallback에서 디코딩하는 동안에는 다른 요청의 admission이 중단될 때 fallback을 지우면 실행 중인 요청의 위치 정보가 사라지기 때문입니다. 당시 상태에서는 맞는 제외였고, 이번 PR이 그 원인을 해소합니다.

이 이슈의 전제는 텍스트 프롬프트가 충분히 길어야 재현됩니다. 짧은 텍스트 요청을 40토큰만 생성하면 main에서도 단독 결과와 일치했고, 텍스트 요청의 위치가 남아 있던 이미지 위치와 겹칠 만큼 길어지자 결과가 갈라졌습니다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| `LoadedModel` | bind, take, install, `clear_vlm_fallback_state`에 `PaddleOcr`, `Ernie45MoeVl`, `HunyuanVl` 분기를 추가했습니다. take와 install은 Qwen3-Omni 분기처럼 `QwenVlMRopeSnapshot`의 MRoPE 부분만 사용합니다. clear에서 제외한 이유를 설명하던 주석은 삭제했습니다. |
| 텍스트 모델 | PaddleOCR-VL, ERNIE-4.5-MoE-VL, Hunyuan-VL 텍스트 모델에 `qwen3_vl.rs`와 같은 형태의 `take_mrope_entry` / `install_mrope_entry`를 추가했습니다. |
| 스케줄러 | `admission.rs`와 `decode_tick.rs`는 주석만 바꿨습니다. 호출부는 이미 모든 모델에 대해 실행되고 있었습니다. |
| `qwen_mrope_state.rs` | `// Used by:` 목록에 각 헬퍼를 쓰는 계열을 모두 적었습니다. |
| 테스트 | 새 모듈 `server::batch::scheduler::mrope_family_binding_tests`(21개)는 계열마다 실제 `enqueue_request` admission, 선점, `Server busy` 중단, 해제를 실행합니다. `vlm_fallback_abort_tests`는 이제 이 계열들이 지워질 것을 기대합니다. |

## 3. 기술적 선택과 그 이유

- **새 진입점을 만들지 않고 Qwen 이름이 붙은 기존 진입점을 재사용했습니다.** 스케줄러는 이미 모든 모델에 대해 `bind_qwen_vl_mrope_state_to_sequence`와 take/install 쌍을 호출하므로, match 분기만 추가하면 스케줄러를 고치지 않고 적용 범위를 넓힐 수 있습니다. 함수 이름이 실제 범위보다 좁아졌지만, 이름 변경은 외형상의 문제이므로 diff를 작게 유지하려고 이번 PR에서는 하지 않았습니다.
- **`with_entry`가 없는 id를 fallback으로 처리하는 동작은 유지했습니다.** CLI는 `seq_id == None`으로 실행되며 fallback에 의존합니다. 리뷰어와 보안 점검에서 서버의 prefill, decode, speculative decode, `route_batched_decode_with_ids` 경로가 모두 id를 넘긴다는 점과 세 래퍼가 `BatchedDecodeKind::Native`를 쓴다는 점을 확인했습니다. 따라서 서버는 이제 이 계열에서 fallback을 읽지 않습니다. CLI 가드 테스트 3개가 fallback이 여전히 동작하는지 고정합니다.
- **텍스트 전용 요청은 빈 엔트리를 바인딩받습니다.** 바인딩은 fallback이 비어 있어도 그 내용을 가져가므로, 텍스트 전용 요청은 나중에 fallback에 무엇이 들어오든 그것을 읽지 않고 자기 빈 엔트리를 소유합니다.

## 4. 검증

- Red 커밋 4116a975: 새 테스트 18개 중 15개가 main에서 실패했습니다. 통과한 3개는 양쪽에서 모두 통과해야 하는 CLI 가드입니다. 리뷰어는 수정을 되돌리면 테스트가 실패함을 확인했습니다.
- M1 Ultra, `mlxcel-server --parallel 4`, greedy, 120토큰. 이미지 OCR 요청 두 개(A, B)를 보낸 뒤 긴 텍스트 요청 하나(T)를 보냈습니다.

| 체크포인트 | main 5610f8f5 (A / B / T 단독 결과와 일치) | 브랜치 |
|---|---|---|
| `mlx-community/PaddleOCR-VL-bfloat16` | 아니오 / 예 / 아니오 | 예 / 예 / 예 |
| `hadeseus/HunyuanOCR-mlx-4bit` | 예 / 예 / 아니오 | 예 / 예 / 예 |

- 단독 실행 결과와 CLI `mlxcel generate` 결과(이미지, 텍스트 프롬프트)는 main과 브랜치가 동일합니다.
- 오케스트레이터 게이트는 main 11463e50을 병합한 브랜치에서 돌렸습니다. fmt, 크로스 레포 참조 검사, `-D warnings` workspace clippy, 스케줄러, 모델, vision 테스트 모듈을 실행했습니다.

## 5. 검증하지 못한 것

- ERNIE-4.5-MoE-VL 실제 체크포인트: MLX 체크포인트가 28B 모델뿐입니다. 분기 형태는 다른 두 계열과 같고 단위 테스트로 검증했습니다.
- `mlx-community/PaddleOCR-VL-4bit`는 main에서도 CLI와 서버 모두 vision tower에서 크래시합니다(`[broadcast_shapes] Shapes (N,1152) and (N,144)`). MRoPE 바인딩과 무관하므로 별도 이슈가 필요합니다.

## 6. 학습 포인트

- capability 디스패처의 조용한 `_ => {}` 분기는 "연결되지 않은 계열"을 "전역 상태를 공유하는 계열"로 만들 수 있습니다. 상태 컨테이너가 시퀀스별 엔트리를 지원한다면, 그 상태를 바인딩하는 디스패처는 엔트리를 가진 모든 런타임을 처리해야 합니다. 테스트도 런타임 하나만 고르지 말고 모든 런타임을 순회해야 합니다.
- 요청 간 상태 누수는 특정 조건이 겹쳐야만 드러날 수 있습니다. 짧은 프롬프트에서 재현되지 않았다는 사실만으로 상태가 분리되어 있다고 볼 수는 없습니다.
