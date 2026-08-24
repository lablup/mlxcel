# 기술 보고서: PR #1395 - MLA latent 캐시에서 symmetric Turbo4가 sign 벡터 밖을 읽던 문제 수정

**날짜**: 2026-08-24
**작성자**: 신정규
**상태**: 완료
**언어/기술**: Rust
**위험도**: 수정 전 높음 (출하 바이너리의 범위 밖 읽기, 지원 체크포인트 하나에서 하드 패닉)

---

## 요약

`KVCache::update_turbo4_sym`이 **V** head 차원으로 만든 `TurboQuantParams` 하나를 K에도 썼고, 가드는 `debug_assert_eq!` 하나였다. `[profile.release]`도 `[profile.test-fast]`도 `debug-assertions`를 켜지 않으므로 **출하 바이너리와 테스트 게이트 어디서도 그 가드가 돈 적이 없다.**

MLA-latent 계열은 `(kv_latent, k_pe)` 쌍을 공유 `KVCache`로 캐시하므로 K는 `kv_lora_rank` 폭(512, DSA 인덱서 키를 이어붙이는 계열은 640)이고 V는 `qk_rope_head_dim` 폭(64)이다. 그래서 K 양자화기가 **64개짜리 sign 벡터를 512 또는 640 좌표에** 적용했다. `from_slice_f32(signs1, &[1,1,1,512])`가 64개 슬라이스 끝을 **448 float 넘어** 읽고, 인덱서 보유 계열에서는 576 넘어 읽었다.

수정 전 재현: `glm4-flash-4bit --kv-cache-mode turbo4`가 40토큰 전부 `!`를 냈고 `fp16`은 정상이었다.

이 변경이 드러낸, 이슈가 기술하지 않은 것 셋.

**지원 체크포인트에서의 하드 패닉.** `deepseek_v2`가 `--kv-cache-mode turbo4`에서 `wht: last axis must be a non-zero power of 2; got shape=[1, 16, 18, 192]`로 죽었다. 경로가 다르다. decompressed 경로가 K=192/V=128이라 128개 sign 벡터를 256바이트 넘어 읽고 Walsh-Hadamard 변환에서 죽는다.

**크래시보다 오래 살아남은 조용한 계약 역전.** 비대칭 모드에서 이 캐시들의 "V" 슬롯은 `k_pe`, 즉 RoPE **키** 스트림이다. 그래서 모드가 **키를 3·4·8비트로 양자화하고** latent(흡수 후에는 K이자 V다)는 정확한 채로 뒀다. 출력이 유한해서 아무도 못 봤고, `docs/turbo-kv-cache.md`는 계약을 "FP16 K, 4-bit V"로 문서화하고 있었다.

**아무도 구현하지 않은 폴백을 약속하던 배너.** `turbo4` 배너는 늘 "non-allowlisted models fall back to Turbo4Asym"이라고 말해 왔다. `is_symmetric_turbo_allowed`의 유일한 비테스트 호출자는 advisor였고, `cache.rs`는 호출자에게 그것을 참조하라고 **주석으로** 요구했다. 강제 없는 문서화된 전제조건이다.

## 1. 문제 정의

### 1.1 배경

네 계열이 latent/rope 쌍을 하나의 `KVCache`에 저장한다. `glm4_moe_lite`, `deepseek_v3`, `kimi_linear`, `longcat_flash_ngram`. 리뷰가 **다섯 번째 호출 지점** `deepseek_v32.rs:381`을 찾았고, 여기에 `model_type` 문자열 셋(`deepseek_v32`, `deepseek_v3.2`, `glm_moe_dsa`. 마지막은 `DeepSeekV32Model`을 감싼다)이 도달하는데 **가드가 전혀 없다.**

`deepseek_v2`는 이미 가드하던 유일한 MLA 계열이다. `MlaLatentCache::supports`가 비-Fp16 모드에 대해 이 이슈가 일반화하려던 바로 그 근거로 `Err`을 돌려준다.

### 1.2 기존 문제

`ffi::from_slice_f32`는 raw 포인터에서 MLX 배열을 만들며 **자기 길이 검사 없이 즉시 복사**한다. 그래서 차원 불일치는 틀린 답이 아니라 범위 밖 읽기다. 출하 바이너리와 그 읽기 사이에 있던 유일한 것이 프로파일이 컴파일 아웃한 단언이었다.

### 1.3 위험 평가

높음. 출하 중인 체크포인트 계열 둘이 쓰레기를 냈고, 하나가 패닉했고, 셋이 더 문서화된 양자화 계약을 조용히 역전시켰다. 수정 자체는 **거절**이므로 위험이 반대다. 너무 많이 거절하면 잘 돌던 구성이 삭제된다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| `cache/turbo/quant.rs` | 차원·sign 벡터 길이 검사를 `debug_assert`에서 `assert`로 승격. `quantize_into_packed`와 `dequantize_from_packed` 양쪽 |
| `cache/turbo/quant3.rs`, `sparse_v.rs` | `from_slice_f32(signs, ...)` 3곳에 같은 승격 |
| `mla/cache.rs` | 모드 규칙을 `latent_layout_supports_mode`로 추출. `MLA_LATENT_CACHE_FAMILIES`, `caches_mla_latent_pair` |
| `cache.rs` | `update_turbo4_sym`의 K/V head-dim 무조건 `assert_eq!` |
| CLI·서버 | 실제 allowlist 폴백. 모드를 알리는 모든 곳에서 **실효** 모드 보고 |
| `server/routes/props.rs`, `types/response.rs` | `/props`가 `kv_cache_mode`·`kv_bits` 노출 |
| `commands/serve.rs`, `inspect.rs` | 메모리 preflight가 실효 모드를 해석 |
| `execution/kv_cache_advisor.rs` | resolver가 거부하는 모드를 더는 권고하지 않음 |
| `docs/turbo-kv-cache.md` | latent 규칙, 역전된 계약, 계열 목록 |

## 3. 기술적 선택과 그 이유

### 3.1 프로파일이 지우는 단언은 가드가 아니다

검사가 `debug_assert_eq!`였고 출하 프로파일 둘 다 `debug-assertions`를 끄므로, 전제조건이 문서화만 되고 강제되지 않았다. 이제 `assert!`인데, 이 선택이 정당한 이유가 구체적이다. **대안이 범위 밖 읽기이기 때문이다.** 틀린 답을 크래시로 바꾸는 것은 나쁜 거래지만 메모리 안전성 위반을 크래시로 바꾸는 것은 옳은 거래이고, 이 검사는 Walsh-Hadamard 변환과 host readback이 지배하는 호출에 대해 O(1)이다.

리뷰가 이슈가 언급하지 않은 형제 3곳에서 같은 위험을 찾았다(`quantize_v_turbo3`, `dequantize_v_turbo3`, `sparse_v` 런처 셋). 오늘 mlxcel 자기 경로로는 도달 불가다. 그 모드들은 V head dim으로 params를 만들고 V만 양자화하기 때문이다. 그래도 승격했다. `TurboQuantParams`가 필드 전부 `pub`이고 `#[non_exhaustive]`도 아니라 짧은 `signs1`을 구조체 리터럴로 만들 수 있고, **한 파일에만 적용하고 형제에는 적용하지 않은 규칙은 규칙을 세우지 않느니만 못하다.**

기록해 둘 배치 판단 하나: `sparse_v`에서 assert를 기존 2의 거듭제곱 게이트 **뒤**에 뒀다. 그 게이트는 Gemma 4의 192차원 head에 대한 **정당한 우아한 폴백**이고, 앞에 두면 잘 돌던 폴백이 패닉이 된다.

### 3.2 규칙 하나, 둘이 아니라

latent 계열용 병렬 규칙을 쓰는 대신, 모드 검사를 `MlaLatentCache::supports`에서 `latent_layout_supports_mode`로 들어냈고 `supports`와 `resolve_kv_cache_mode_for_model`이 둘 다 그것을 부른다. 규칙 문장과 그 근거가 한 곳에만 존재한다. `supports`는 원래 호출자에 대해 동작이 동일하다. 같은 에러 문구, 같은 순서.

### 3.3 이슈가 요구한 변경이 잘 돌던 구성을 삭제할 뻔했다

이슈 3단계는 `deepseek_v2`를 latent 계열로 선언하라고 한다. **의도적으로 빼뒀다.**

`deepseek_v2`는 forward 호출마다 `supports`를 묻고, 모드가 양자화면 decompressed per-head layout으로 떨어진다. 그 경로에서 K는 `qk_nope_head_dim + qk_rope_head_dim`, V는 `v_head_dim`이고 `Int8`과 `Turbo4Asym`이 **정상 동작한다.** params가 V에서만 오고 K는 WHT에 닿지 않기 때문이다. 더 강하게는, 그 계열의 흡수 경로가 환경변수 opt-in이고 기본 off라 어차피 decompressed로 돈다. 목록에 넣었으면 **동작하는 모드를 거절**했을 것이다. `turbo4`는 allowlist로 따로 고쳐지고, 그것이 3.4의 패닉을 막는다.

다음 독자가 목록을 "완성"하지 않도록 회귀 테스트로 구분을 고정했다.

### 3.4 이슈가 몰랐던 하드 패닉

`deepseek_v2`가 이 PR 전에 `turbo4`에서 죽었다. DeepSeek-V2-Lite는 decompressed 경로에서 K=192/V=128을 준다. V=128로 만든 params가 192 폭 K에 `quantize_k_turbo4`를 돌려 복사 시점에 256바이트를 넘어 읽고, 관측되는 실패는 한 줄 뒤 Walsh-Hadamard 변환에서 그 자신의 assert가 인용된 메시지를 낸다.

수정은 옮기지 않고 **막는다.** resolver가 `Turbo4`를 `Turbo4Asym`으로 강등하고, 그 모드는 `quantize_k_turbo4`를 부르지도 192 폭 K를 WHT에 보내지도 않는다.

### 3.5 문서가 latent 폭에 대해 틀렸는데, 아무도 눈치채지 못한 이유로

리뷰가 `glm_moe_dsa`의 `kv_lora_rank`가 512가 아니라 128일 수 있다고 지적했다. **그 전제는 틀렸다.** 128은 `#[cfg(test)]` 픽스처 안에 있고 실제 serde 기본값은 형제들과 같은 512다.

그런데 doc 주석은 **다른 이유로 틀려 있었다.** `deepseek_v32.rs:377-380`이 캐시 전에 DSA 인덱서 키를 latent에 이어붙이므로, 이 세 계열의 "K" 슬롯은 `kv_lora_rank + index_head_dim`이고 로컬 `glm_moe_dsa` 설정에서는 512가 아니라 **640**이다. 범위 밖 읽기 수치도 거기서는 448이 아니라 576 float다. doc 주석과 `docs/turbo-kv-cache.md` 양쪽을 정정했다.

기록할 값어치가 있는 이유: **리뷰의 지적이 틀렸는데도 생산적이었다.** 거짓으로 판명된 주장을 확인하는 과정이 그 옆의 참인 오류를 드러냈다.

### 3.6 실효 모드 알리기, 그리고 서버의 거울상 버그

`turbo4` 배너가 어떤 코드도 수행하지 않는 폴백을 약속했다. 그것을 실제로 만들려면 모드를 알리는 **모든** 표면이 실효 모드를 보고해야 했다.

구현하다 한 층 아래에서 같은 부류의 문제가 나왔다. `into_startup_config`가 `initialize_server_logging`보다 **먼저** 돌아서 새 경고가 구독자 없는 곳으로 갔다. **서버가 올바른 폴백을 하면서 그것에 대해 아무 말도 안 한 것**이고, 이는 원래 결함의 거울상이다. 이제 통지가 `ServerStartupConfig::kv_cache_mode_notices`에 실려 로깅이 선 뒤에 방출되고 `effective KV cache mode` 줄이 함께 나간다.

두 표면이 더 실효가 아닌 요청 모드를 해석하고 있었다. 둘 다 메모리 preflight(`serve.rs`, `inspect.rs`)이고 `generate.rs`에서 이미 고친 같은 버그다. 결과가 측정됐다. `glm4-flash-4bit` 32768토큰에서 `--kv-cache-mode int8`이 실제 **29.38 GiB**에 대해 **14.69 GiB**를 보고했다. **2배 과소 계상**이고, 그 추정이 기동을 하드 중단시킬 수 있다. 이제 둘 다 29.38을 보고하고, `llama-3.1-8b-4bit` 대조군은 여전히 정상적으로 절반이 된다.

`mlxcel recommend`도 resolver가 이제 거부하는 MLA 분류에 `Int8`을 권고하고 있어, 운영자에게 바이너리가 거부하는 모드를 쓰라고 말할 참이었다. 이제 latent 계열 술어로 분기한다.

## 4. 검증

### 4.1 재현, 전과 후

`models/glm4-flash-4bit`, `-n 40 -t 0 --seed 1350`, 같은 머지 베이스에서 변경을 경로별로 되돌려 만든 기준 바이너리 대조:

| 모드 | 전 | 후 |
|---|---|---|
| `fp16` | 정상, 46.16 tok/s | 정상, 46.67 tok/s |
| `turbo4` | **40토큰 `!`**, 12.06 tok/s | 정상, 배너 `fp16 (requested turbo4; not supported on this model family)` |
| `turbo4-asym` | 정상이나 **fp16과 다름** | fp16과 동일 |

눈으로가 아니라 계산으로: 변경 후 `fp16 vs turbo4`와 `fp16 vs turbo4-asym`이 둘 다 IDENTICAL, 변경 전에는 둘 다 DIFFERENT. **마지막 비교가 조용한 결함의 증거다.** 유창한 출력을 내서 눈에 보이는 흔적을 남기지 않았다.

### 4.2 바뀌면 안 되는 것

`models/qwen3.5-0.8b-4bit`(`qwen3_5`, symmetric allowlist 소속)는 symmetric Turbo4를 유지하고 변경 전후 **바이트 동일**이며 `fp16`도 그렇다. 배너만 바뀌었는데, 그 arm은 이제 폴백을 탈 수 없기 때문이다. 리뷰 수정 후 7개 모드/모델 조합 전체를 재포착해 1라운드 결과와 diff한 것도 타이밍 줄을 빼면 동일하다.

### 4.3 게이트

`cargo test --workspace --profile test-fast --features metal,accelerate`: 리뷰 수정 전 8399 통과, 0 실패. 로컬 CI(`ci.yml` 잡 10개 중 8개, 이번 세션 GitHub Actions 불가): 7 pass, 0 fail, 2 skip. skip 둘은 CUDA가 필요하고 이 변경은 XLA 경로를 안 건드린다. 실모델 게이트 `glm4-flash-4bit`·`llama-3.1-8b-4bit` 둘 다 PASS.

clippy를 `--lib --tests`뿐 아니라 `-p mlxcel --bins`에도 돌렸다. 여기서 중요한데, `serve.rs`와 `inspect.rs`는 `--lib --tests`가 절대 컴파일하지 않는 바이너리 타깃 모듈이다.

## 5. 리뷰에서 나온 지적

HIGH 1건, MEDIUM 4건, LOW 2건, 전부 반영.

HIGH는 **다섯 번째 latent 호출 지점**이었다. `MLA_LATENT_CACHE_FAMILIES`가 네 계열을 나열했는데 `deepseek_v32.rs:381`이 같은 latent/rope 쌍을 가드 없이 캐시하고 `model_type` 문자열 셋이 거기 도달한다. `caches_mla_latent_pair`가 설계상 정확 일치라 `"deepseek_v3"`가 `"deepseek_v32"`를 덮지 않았다.

메모리 안전성 절반은 우연히 막혔다. 그 계열들이 symmetric allowlist에 없어 이미 강등되기 때문이다. 그러나 **PR 자신의 두 번째 결함 주장이 그들에 대해 참이 아니었다.** 비대칭 모드와 서버 `--kv-bits` 경로가 여전히 그들의 `k_pe` 키 스트림을 양자화했고, `k_pe`가 64 폭이라 유효한 2의 거듭제곱이고 아무것도 fault하지 않아 조용했다.

일반화할 값어치가 있다. **"이 변경이 결함 X를 닫는다"는 주장은 수정 자체와 같은 열거 규율을 요구한다.** 다섯 중 넷을 고치고 닫혔다고 기술하는 것이 더 위험하다. 그 기술이 다음 사람이 들여다보는 것을 막기 때문이다.

## 6. 검증하지 못한 것

**`deepseek_v32`나 `deepseek_v3.2` 체크포인트가 로컬에 없다.** 유일한 `glm_moe_dsa` 체크포인트 `models/glm-5-4bit`는 가중치도 토크나이저도 없는 부분 다운로드다. resolver가 모델 로드 전에 돌기 때문에 거절 경로는 그 위에서 완전히 실행되지만, 세 계열의 종단 간 생성은 구동할 수 없었다. 동작은 동일 코드 경로를 공유하며 직접 측정된 `glm4_moe_lite`에서 추론했다.

`deepseek_v3`, `kimi_linear`, `longcat_flash_ngram`도 로컬 체크포인트가 없다. latent 규칙은 실하드웨어에서 `glm4_moe_lite`로만 검증됐고 나머지는 단위 테스트가 덮는다.

perplexity나 장문맥 품질 측정은 없다. 여기 모든 실모델 검사는 짧은 프롬프트다.

## 7. 학습 포인트

- **출하 프로파일이 지우는 단언은 가드가 아니라 문서다.** 그것이 지키는 전제조건이 메모리 안전성이라면 검사는 출하 바이너리에 있어야 하고, O(1) 비용은 판단 요소가 아니다.
- **호출 지점은 이슈의 목록이 아니라 기계적으로 열거한다.** 다섯 중 넷은 이슈를 따라가 찾았고, 다섯째는 패턴을 grep해서 찾았다.
- **"결함 X를 닫는다"는 주장은 수정과 같은 엄밀함을 요구한다.** 다섯 계열 중 넷에서 닫고 그렇게 말하는 것은 아무 말도 안 하느니만 못하다. 그 주장이 다음 독자의 확인을 막는다.
- **거절 형태의 수정은 위험 프로파일이 반대다.** 위험은 너무 많이 거절하는 쪽이다. 이슈는 `deepseek_v2`를 목록에 넣으라고 했고 그러면 동작하는 구성이 삭제됐을 텐데, 그것이 동작하는 이유는 호출별 폴백을 읽어야만 보인다.
- **틀린 리뷰 지적도 생산적일 수 있다.** `kv_lora_rank: 128` 전제는 거짓이었고, 그것을 확인하는 과정이 그 옆의 참인 오류(인덱서 키 때문에 latent가 512가 아니라 640)를 드러냈다.
