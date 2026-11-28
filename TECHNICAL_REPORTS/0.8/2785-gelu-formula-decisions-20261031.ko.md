# PR #2785: #2423이 미결로 남긴 계열의 GELU 공식 확정

**작성일**: 2026-10-31 (머지 전 작성, 당시 PR은 열린 상태)
**상태**: 리뷰 중
**언어**: Rust, C++
**위험도**: 중간 (약 30개 계열의 GELU 공식이 바뀐다. 각 변경은 체크포인트 config를 따르며, 로컬 체크포인트가 있는 계열은 모두 CLI와 서버에서 스모크 테스트를 거쳤다)

## 요약

#2423은 모든 GELU 호출 지점을 목록화하고 첫 번째 그룹을 체크포인트가 선언한 공식으로 바꾼 뒤, 나머지를 이슈 #2548의 미결 항목으로 남겼다. 이 PR은 하나의 명시된 규칙으로 이를 마무리한다. 체크포인트 config가 결정하고, 키가 생략된 경우에는 해당 `model_type`의 Hugging Face config 클래스 기본값이 `ACT2FN` 기준으로 결정한다. mlx-lm 또는 mlx-vlm 포트가 Hugging Face와 다르면 mlxcel은 Hugging Face를 따르고 그 차이를 기록한다. `docs/gelu-call-sites.md` 0b 절에 결정마다 한 행과 이유가 있다.

## 1. 문제 정의

`mlxcel_core::gelu_approx`는 이름과 달리 정확한 erf GELU를 계산하고, MLX `nn.gelu_approx`는 tanh 형태다. 많은 포트가 이름만 옮겨 와서, 체크포인트가 `gelu_new`나 `gelu_pytorch_tanh`를 선언하거나 CLIP 타워의 생략된 `hidden_act`가 `quick_gelu`를 뜻하는 곳에서도 erf를 실행했다. erf와 tanh의 차이는 원소당 1e-3 미만이고 logit-trace 게이트로는 구분되지 않으므로, 측정된 회귀가 아니라 config 판단의 문제였다. 두 지점은 공식 선택보다 심각했다. Nemotron-NAS는 `gelu_new`, `gelu_fast`와 모르는 이름을 모두 SiLU로 보냈고, Gemma 4의 tensor-parallel per-layer gate는 단일 호스트 경로가 tanh인데도 erf를 계산했다.

## 2. 기술적 검토 사항

- **계열별 공식.** 바뀐 행마다 로컬 체크포인트의 `config.json`이나 원격 모델링 코드와 대조했다. StarCoder2와 Gemma v1은 `gelu_pytorch_tanh`, Phi-2는 `gelu_new`를 선언하고, Molmo 원격 코드는 CLIP ViT를 `image_mlp_activations="quick_gelu"`로 만들며, DeepSeek-VL2의 `siglip_vit.py`는 `nn.GELU(approximate="tanh")`를 쓰고, Phi-3.5-V는 `openai/clip-vit-large-patch14-336`를 지정한다.
- **공유 커널의 영향 범위.** `compiled_gelu_mlp_forward_fp16`이 erf GeGLU에서 tanh GeGLU로 바뀌었다. 호출자는 Gemma v1, Gemma 4 dense MLP, Gemma 4 assistant drafter뿐이며 모두 `gelu_pytorch_tanh`다.
- **공유 vision config.** 생략된 `vision_config.hidden_act`는 필요한 로더에서 타워 `model_type`으로 해석된다 (SigLIP은 tanh, CLIP은 `quick_gelu`). 다른 타워를 위해 serde 기본값은 erf로 유지된다. 선언된 `quick_gelu`는 이제 fast로 파싱되며, 이를 명시적으로 선언한 로컬 체크포인트는 없어서 이 경로로 바뀐 기존 타워는 없다.
- **Gemma3n eager와 OpenXLA tape.** 커널, fused MLP의 sparse 분기, XLA `sparse_gelu_stages` tape, host oracle이 함께 tanh로 바뀌었다. 이 PR 이전부터 있던 구조적 차이가 하나 있다. eager 커널은 bf16 `stddev`에 f32 `array(std_multiplier)`를 곱하므로 MLX가 cutoff 이후 전부를 f32로 승격하고 마지막에 한 번만 반올림한다. 반면 tape은 (mlx-lm의 Python weak scalar처럼) 모든 중간값을 bf16으로 반올림한다. erf tape도 같은 구조였다. IREE 호스트에서 같은 입력으로 비교해야 한다.

## 3. 기술적 선택과 그 이유

### 3.1 StarCoder2: XLA 그래프가 아니라 eager를 옮긴다

이슈는 eager와 mlx-lm이 erf를 쓰므로 OpenXLA StarCoder2의 tanh 그래프를 버그로 보았다. 그러나 체크포인트는 `gelu_pytorch_tanh`를 선언하고 이는 `Starcoder2Config` 기본값이기도 하며, XLA golden은 그 activation으로 만든 HF fp32 oracle과 token-exact로 일치한 뒤 고정되었다 (`spike/openxla/dense_arch_check.py`, PR #561). 결정 규칙에 따라 옮겨야 할 쪽은 eager이며, 고정된 golden은 그대로 둔다.

### 3.2 지점별 tanh 변형

erf에서 바뀐 계열은 prefill 시간을 늘리지 않는 #2423의 선택인 `GeluActivation::TanhF32Compiled`를 쓴다. Gemma 지점은 기존 입력 dtype tanh GeGLU 커널을 써서 bf16 경로와 양자화 경로가 같은 계산을 하게 한다. Nemotron codec (erf)과 VoiceChat TTS backbone (`power`를 쓰는 tanh)의 private 복사본은 `GeluActivation`으로 합쳐졌다. 두 경우 모두 일부 half-precision 원소의 마지막 비트가 달라질 수 있으며 tolerance 테스트로 범위를 묶었다.

### 3.3 Nemotron-NAS는 모르는 activation을 거부한다

`hidden_act`는 로드 시 한 번 `NasActivation`으로 파싱된다. GELU 이름은 `GeluActivation::from_hf_name`을 거치고, SiLU, ReLU, GELU 변형이 아닌 이름은 조용히 SiLU를 쓰는 대신 로드 오류가 된다.

## 4. 변경 요약

| 항목 | 값 |
|---|---|
| 변경 파일 | 50 |
| 라인 | +948 / -305 |
| 새 테스트 | tanh `gelu_topk`와 tanh dense GeGLU의 FFI 테스트, 타워 기본값 해석 (eager와 XLA), MiniCPM-o activation 파싱, Nemotron-NAS activation 파싱, `quick_gelu` emitter op, 두 통합에 대한 tolerance 테스트 |

| 카테고리 | 요약 |
|---|---|
| Gemma | v1 bf16, Gemma 4 dense bf16과 drafter, Gemma 4 TP gate, Gemma3n sparse layer (eager와 XLA), RecurrentGemma가 tanh로 이동 |
| 텍스트 LM | Phi, Phixtral, GPT-2, StarCoder2 (eager), Moondream2/3 dense MLP가 tanh로 이동 |
| Vision | SigLIP 계열 타워 9개와 Moondream vision MLP 및 projector가 tanh로, Qwen2-VL, Molmo, LLaVA 1.5 / NeXT, Phi-3.5-V CLIP이 `quick_gelu`로 이동, MiniCPM-o는 config를 읽음 |
| 이유와 함께 유지 | GPT-NeoX, ModernBERT, Llama 4 vision, Kokoro PLBert, Molmo2, Qwen3-VL-MoE, Qwen3-Omni, LLaVA-Interleave, Llama-Nemotron-VL-Embed |

## 5. 후속 조치

- GPT-BigCode (T5)는 `gelu_pytorch_tanh`를 선언하지만 여전히 erf로 실행된다. 이 PR이 문서화한 규칙을 알고도 어기는 유일한 행이다.
- Idefics2 (S6)와 Granite 4 Vision (S7)은 (mlx-vlm과 같이) `quick_gelu`를 강제하지만 Hugging Face는 tanh로 계산한다. 인벤토리가 mlx-vlm 기준 MATCH로 표시해 #2548 범위 밖이었다.
- IREE 호스트에서 `xla-diagnostics` Gemma3n 그래프를 실행해 위의 eager와 tape 간 carrier 차이를 정리한다.
- 이슈 스레드의 fused single-kernel tanh GELU는 #2780에서 별도로 추적한다.

## 부록: 검증

GB10에서 바뀐 모듈마다 좁은 CUDA 테스트 필터 (`--test-threads=1`)가 통과하고, mlxcel-xla emitter 테스트는 StarCoder2 golden이 byte-exact인 상태로 통과하며, clippy와 fmt가 깨끗하다. 이 브랜치의 전체 workspace 테스트는 실패가 하나 (`tests/step3p7_parity.rs` `text_decode_delegates_sequence_local_native_batching`) 있으며 base 커밋에서도 똑같이 실패한다. 로컬 체크포인트가 있는 바뀐 계열은 모두 greedy `mlxcel generate` 한 번과 `mlxcel-server` 요청 한 번에서 일관된 출력을 냈다. 체크포인트가 없는 계열 (Gemma v1 bf16, RecurrentGemma, Phixtral, GPT-2, Phi-4-MM, GELU activation의 Nemotron-NAS, VoiceChat TTS, Gemma 4 TP gate)은 PR에 적힌 합성 테스트에 의존한다.
