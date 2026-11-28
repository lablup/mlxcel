# 기술 보고서: PR #2578 - SAM 3D Objects 재구성 추가

**날짜**: 2026-10-23
**최종 갱신**: 2026-10-24
**상태**: Completed — 구현 및 로컬 인수 완료, 최종 문서/fixture commit CI 대기
**언어**: Rust, C++, Python, Markdown
**위험 수준**: High

## 요약

PR #2578은 SAM 3D Objects를 `mlxcel predict`에 이식합니다. 구현은 RGB 이미지와 마스크, 알파 채널을 마스크로 사용하는 RGBA 이미지, 또는 선택적 포인트 맵을 입력받아 공개된 structure, pose, sparse-latent flow를 실행합니다. 결과로 객체 로컬 좌표계의 OBJ 메시와 Gaussian PLY, pose, 점유 좌표, sparse latent, conditioning 메타데이터를 기록합니다. 기본값은 seed 42, structure 25단계, latent 25단계, 번들 MoGe-3 자동 깊이 추정, Gaussian과 메시 전체 디코딩입니다.

Revision `68cd161e`는 9분 50초의 ROCm 테스트 빌드와 집중 단위 테스트 61개를 통과했습니다. 인증된 전체 RGBA/dropout 캡처도 고정된 엄격 비교 계약을 통과하여 정렬된 좌표 5,793개가 정확히 일치했고, latent 상대 RMS는 0, 메시 정점 수 상대 오차는 `0.0001589479763`, 정규화 대칭 Chamfer는 `4.450226e-6`이었습니다. Revision `6bd85cf3`는 이어서 기본 깊이 focal/shift 차이를 닫았습니다. Focal solver 회귀 테스트 5개가 통과했고 기본 conditioning 17단계가 모두 bit-exact였으며, 인증된 기본 캡처는 좌표 6,307개 정확히 일치, latent 상대 RMS 0, 메시 정점 189,623개 정확히 일치, 정규화 대칭 Chamfer `5.141699913725434e-6`로 통과했습니다. Literal 무옵션 CLI도 같은 좌표와 latent 결과, 정확한 정점 수, 정규화 Chamfer `5.141710000502124e-6`로 통과했습니다. 구현 및 로컬 런타임 인수는 **Completed**입니다. 최종 fixture staging과 검토 지적 수정도 완료했습니다. 마지막 inventory signoff도 통과했습니다. 보고서 작성 시점에는 최종 문서/fixture commit의 CI가 남아 있으므로 final-head CI 결과를 주장하지 않습니다.

## 1. 문제 정의

SAM 3D Objects는 마스크된 객체를 pose, sparse voxel 점유와 latent 상태, 텍스처 삼각형 지오메트리, 3D Gaussian으로 함께 재구성합니다. 이 PR 전에는 mlxcel에 SAM 3D Objects family, 이 dense prediction 경로의 마스크/포인트 맵 sidecar, OBJ와 Gaussian PLY 출력 경로가 없었습니다.

다음 경계는 근사 구현 시 겉보기에는 그럴듯하지만 잘못된 지오메트리를 만들 수 있습니다.

- 공개 번들은 본 모델 BF16 텐서 3,288개와 번들 MoGe F32 텐서 607개를 포함합니다.
- 네 DINOv2 scope는 키, shape, dtype, 값이 정확히 같은지 확인한 뒤에만 하나의 trunk를 공유할 수 있습니다.
- seed 42 난수 추출, 스케줄 순서, 점유 순서, sparse neighbor topology, pooling, shifted window, FlexiCubes tie 순서가 좌표와 지오메트리에 영향을 줍니다.
- 자동 깊이, 제공된 포인트 맵, 학습된 point-token dropout은 서로 다른 conditioning 경로입니다.
- resize 축 순서, nearest-neighbor 비율 순서, boolean attention mask, fused linear 누적, tanh GELU 평가 순서, 백엔드 LayerNorm 동기화는 수치적으로 관찰됩니다.
- sparse 및 메시 topology는 수 GiB의 live set을 만들 수 있습니다. admission과 단계별 계산은 허용된 결과를 샘플링하거나 자르지 않으면서 위험한 할당을 사전에 거부해야 합니다.
- parity 주장은 metric 계산 전에 소스, 번들, 네이티브 런타임, 입력, 옵션, 바이너리, 출력, 비교 프로그램을 인증해야 유효합니다.

따라서 고정된 수치 계약은 좌표 값과 순서의 정확한 일치, latent 상대 RMS `1e-2` 미만, 메시 정점 수 오차 1% 이하, 기준 메시 AABB 대각선 대비 전체 집합 대칭 평균 Euclidean 최근접 거리 `1e-3` 미만을 요구합니다. 비교기는 정점을 필터링하거나 샘플링하지 않습니다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| Family 통합 | 정지 이미지 mesh prediction에 `sam3d_objects`를 등록하고 architecture 및 dense predictor registry에 연결했습니다. |
| CLI sidecar | 타입이 지정된 `--mask`, `--pointmap` 입력을 추가하고 지원하지 않는 family에서는 모델 생성 전에 거부합니다. |
| 입력 처리 | RGB/RGBA 처리, 객체 crop 및 전체 이미지 view, 제공된 포인트 맵 정규화, MoGe 자동 깊이, 축 변환, point-token dropout을 구현했습니다. |
| 모델 | Structure/pose flow, sparse latent flow, 공유 DINO conditioning, sparse convolution/attention, decoder, Gaussian/mesh head를 이식했습니다. |
| 체크포인트 로딩 | 인덱스 텐서 3,895개를 정확한 키, shape, dtype으로 검증하고 BF16/F32 분리를 유지합니다. |
| 수치 정책 | Objects 전용 resize, DINO fused-linear 정책, boolean exclusion fast SDPA, Power 기반 tanh GELU 평가, 정확한 focal solver bridge를 추가했습니다. 다른 family의 공유 기본 동작은 유지됩니다. |
| Sparse 자원 | 검사된 단계별 peak 수식, 제한된 cache eviction/recompute, 제공된 map guard, 실패 가능한 host 할당, 완전한 결과 또는 오류 정책을 추가했습니다. |
| 지오메트리 | 안정적인 sparse topology와 FlexiCubes를 이식하고 upstream 호환 OBJ 및 binary little-endian Gaussian PLY writer를 추가했습니다. |
| 출력 schema | `result.safetensors`에 pose, 정수 `coords`, `latents`, boolean `pointmap_conditioned`를 기록하고 메타데이터에 seed, 실제 단계 수, 포맷, conditioning 경로를 기록합니다. |
| 검증 | 실제 소형 fixture, 고정된 전체 reference/capture identity, 비교 규칙, 프로세스 receipt, 실제 모델 capture target을 추가했습니다. |
| 문서 | 기본값, 입력, 축, dtype, 출력 파일, 객체 로컬 지오메트리, 지원하지 않는 batch/video/server 범위를 문서화했습니다. |

인수된 RGBA 소스 revision은 `68cd161e1a45018559d00fca61e5516f99a7f803`입니다. 소스 manifest는 `391be3af3fec1a7552f6355dac55cec5b13dcdc1b5259148c1ac159ca3865e4e`, 보존된 118,876,160바이트 소스 archive는 `fdf3bb87f072fcd021b65a15405e366a98cdebf98c05fee50997220191f2ebff`입니다. 인수된 기본 깊이 검증은 revision `6bd85cf38e70ea43f2ba2e5a63363c35193c2db7`, manifest `dd573f15f508ce241e51658260e0faf7baf88f6edbd678c12ebe60bb971d8981`, 보존된 119,029,760바이트 archive `f2224b6175c38f54e2b84070cf28710ec8beb25b6c6a1e990a44135a9b2cfdb5`를 사용합니다. 이후 변경은 기본 깊이 focal solver 경로로 제한되므로 이전 RGBA/dropout 통과 결과는 소스 동등성을 통해 유지됩니다. 그 외에는 어느 identity의 증거도 다른 identity로 이월하지 않습니다. `869af0c296e748e8c62075bd53c5cd18db5cc874` 대비 기록된 `68cd161e` 작업 범위는 130개 파일, 14,660줄 추가, 418줄 삭제입니다.

## 3. 아키텍처 및 기술 결정

### 3.1 공개된 전체 파이프라인 보존

```text
RGB/RGBA 이미지 + 마스크
  -> 객체 crop 및 전체 이미지 view
  -> 제공된 point map 또는 번들 MoGe-3 또는 학습된 dropout 경로
  -> 공유 DINOv2 conditioning
  -> 25단계 structure 및 pose flow
  -> 정렬된 점유 좌표
  -> 25단계 sparse latent flow
  -> Gaussian 및 mesh decoder
  -> pose/coords/latents + OBJ/PLY + metadata
```

구현은 seed 42, `ss_steps=25`, `slat_steps=25`, `estimate_depth=true`, 두 출력 포맷을 기본값으로 유지합니다. 제공된 HWC point tensor는 +X 왼쪽, +Y 위쪽, +Z 앞쪽을 사용하고 NaN으로 유효하지 않은 점을 표시합니다. 포인트 맵 없이 깊이를 비활성화하면 학습된 dropout 경로를 사용합니다. 지오메트리와 parity 배열은 자원 예산을 맞추기 위해 샘플링하거나 자르지 않습니다.

### 3.2 Dtype 및 그래프 정책의 범위 제한

Loader는 추론 전에 본 모델 BF16 텐서 3,288개와 깊이 모델 F32 텐서 607개를 검증합니다. 중복 scope의 동일성을 확인한 뒤에만 DINO를 공유합니다. Objects 전용 fused-linear 및 activation 정책은 독립 DINOv2나 다른 dense family를 변경하지 않습니다.

좁은 compiled helper는 upstream 그래프 경계를 유지합니다. Objects feed-forward 경로는 upstream Power 및 연산 순서를 따르는 native tanh GELU helper를 사용합니다. Resize는 width-first, height-first, tie case를 포함해 upstream adaptive antialias 구성과 ratio-first nearest mapping을 사용합니다. Sparse attention은 boolean mask를 수치적으로 더하지 않고 boolean exclusion 의미를 사용합니다.

### 3.3 완전한 live phase 기준으로 sparse 작업 제한

MoGe refiner는 실행 전에 보수적인 row pyramid를 계산하고, 정수 연산을 검사하며, residual, pool, projection, fusion, upsample live set을 계산합니다. 기본 cache eviction 목표는 512 MiB, neighbor map 전체 저장 한도는 256 MiB, 지원하는 hard envelope는 8 GiB입니다. 실제 capture configuration의 보수적 계산 한도는 5,598,663,168바이트입니다.

Sparse convolution은 제공된 neighbor map을 borrow하고 제한된 chunk를 materialize한 뒤 concat 전에 단계별 입력을 해제합니다. Pooling도 반환 배열을 materialize하면서 topology 및 gather scratch의 scope를 제한합니다. Cache map은 제거한 뒤 정확한 정수 좌표로 다시 계산할 수 있으며, 필요한 map을 제거하기 전에 feature를 materialize합니다. 허용된 작업은 완전한 topology를 만들거나 명시적으로 실패합니다.

### 3.4 Native LayerNorm 동기화를 런타임 선행 조건으로 취급

독립 소스 검토에서 ROCm forward LayerNorm kernel의 race를 확인했습니다. 다른 thread가 공유 mean snapshot을 읽는 동안 warp zero가 variance 계산에 같은 shared storage를 다시 사용할 수 있었습니다. Commit `edd9e7fd5ed8dc592a320e723e3a655627990247`는 mean snapshot 이후 read-completion barrier 하나를 추가합니다. 산술과 backward 경로는 변경하지 않습니다.

Canonical reference는 수정된 native library `c1d8203b257364271e1ea6298705dd3bb0c82a7f06634f5578a209883601a9fc`를 사용합니다. 이전 library `9e38e9292550ea8eec9eb1877d06b72f5155ff382ac0e4ec1054818e15276094`에는 이 barrier가 없었습니다. 이전 library를 사용한 canonical 시도는 5,821개와 5,783개의 불안정한 좌표 수를 냈습니다. 작은 LayerNorm probe는 패치 전후 모두 안정적이었으므로 해당 probe는 인과 증거가 아닙니다. 소스 수준 race와 전체 파이프라인 불안정성은 별도로 기록하며, 보고서는 probe로 인과관계를 추론하지 않습니다.

수정된 런타임의 두 RGBA canonical reference는 같은 5,793개 좌표를 반복 생성했습니다. 14개 텐서 중 12개는 bit-exact였고, 나머지 mesh 텐서의 차이는 약 `1e-7`이었습니다. 이 결과가 엄격한 전체 모델 비교에 사용한 반복성 범위를 정합니다.

### 3.5 실제 upstream 평가 경계 일치

초기 custom-stream reference는 yield된 모든 event를 평가했습니다. 실제 upstream `Pipeline.generate`는 중간 event를 소비하고 완성된 출력만 평가하므로, event별 평가는 그래프 materialization 경계를 바꾸고 긴 BF16 GPU trajectory를 변화시킬 수 있었습니다.

Canonical reference는 이제 실제 pipeline 완료 동작을 실행합니다. 이전의 모든 실패 출력, 로그, exit, receipt는 진단 증거로 보존됩니다. 과거 39단계 sampling-trajectory 비교는 관찰한 경계에서 정확했지만, instrumentation이 평가 경계를 바꾸었으므로 instrumentation 없는 전체 trajectory를 증명하지 못했습니다.

### 3.6 남은 기본 깊이 차이의 위치 확인

`b000e46c` 수정은 upstream adaptive resize, DINO 누적, Python 호환 UV endpoint 생성을 Objects MoGe 경로로 제한합니다. standalone MoGe와 기본 DINO 동작은 바뀌지 않습니다. 실제 `estimate_depth` 호출 한 번을 관찰한 43단계 진단에서 입력, encoder, neck, head, refiner, 최종 affine transform, mask를 포함한 39단계가 bit-exact였습니다. 남은 focal 및 shift scalar 차이는 최대 절대 오차 `1.9073486328125e-6`, raw 상대 RMS `4.0945834798765365e-8`의 point map 차이로 전파되었습니다. 이는 차이 위치를 확인하는 증거이며 전체 모델 인수 결과가 아닙니다.

후속 수정은 upstream의 eager 초기 평가와 30개 compiled focal/shift solver step을 Objects 전용 free-focal dispatch로 이식합니다. fixed-focal 동작과 standalone MoGe는 기존 경로를 유지합니다. Shape에서 계산한 byte product는 모든 MLX 연산 전에 검사합니다. Focal 회귀 테스트 5개가 통과했고, 해당 backend의 실제 solver 상태 31개가 모두 bit-exact였으며, 기본 conditioning 17단계 비교도 전부 bit-exact가 되었습니다. 이어진 instrumentation 없는 기본 캡처는 전체 수치 계약을 통과했습니다.

## 4. 검토 결과

### 4.1 정확성 및 통합

독립 검토는 CLI routing, sidecar, checkpoint schema, 입력 처리, 두 flow, 공유 DINO 정책, compiled helper, sparse topology, FlexiCubes, writer, 결과 schema, 기본값, 결정적 순서를 다뤘습니다. 검토 중 다음 구체적 지적 사항을 수정했습니다.

- `pointmap_conditioned`를 원소 한 개의 boolean tensor로 직렬화
- sparse attention의 boolean exclusion 의미 보존
- upstream crop/resize 축 및 비율 순서 일치
- 정확한 Objects tanh GELU 평가 순서 사용
- Objects 전용 DINO fused-linear 누적 보존
- sparse convolution과 pooling의 완전한 live phase 계산
- 안정적인 sparse map cache 및 정확한 recompute
- MLX 할당 전 제공된 map과 preprocessing 할당 검증
- upstream OBJ `%.8g` 포맷 및 완전한 PLY 의미 일치

최종 정적 구현 검토에서 열린 프로덕션 정확성 지적 사항은 없습니다.

### 4.2 보안 및 성능

독립 보안/성능 검토는 발견된 모든 지적 사항을 닫았습니다. Public boundary는 옵션 타입과 범위, 이미지/마스크/포인트 맵 shape와 내용, checkpoint schema, sparse cardinality, neighbor map dtype과 범위, geometry index와 값, 출력 이름을 검증합니다. 검사된 산술은 preprocessing, sparse phase, host geometry allocation을 포함합니다. 위험한 할당 전에 자원 거부가 발생하며, 허용된 지오메트리는 필터링, 샘플링, truncation하지 않습니다.

이 제어는 실패 동작을 제한하지만 처리량이나 메모리 benchmark를 의미하지 않습니다.

### 4.3 엄격한 RGBA parity 결과

인수된 RGBA/dropout case는 RGBA alpha mask, `estimate_depth=false`, seed 42, 25+25단계, TF32 비활성화, 완전한 Gaussian/mesh decoding, 수정된 native runtime, 실제 upstream 완료 경계를 사용했습니다.

| Metric | 계약 | 결과 | 상태 |
|---|---:|---:|---|
| 정렬된 점유 좌표 | 정확히 일치 | 5,793개 중 5,793개 정확히 일치 | Pass |
| Latent 상대 RMS | `< 1e-2` | `0` | Pass |
| Mesh 정점 수 상대 오차 | `<= 1%` | `0.0001589479763` | Pass |
| 정규화 대칭 전체 집합 Chamfer | `< 1e-3` | `4.4502263869e-6` | Pass |

Reference는 메시 정점 188,741개, Rust 결과는 188,711개입니다. 평균 최근접 거리는 actual에서 reference 방향으로 `5.3297777143e-6`, reference에서 actual 방향으로 `5.7469257839e-6`이며 reference AABB 대각선은 `1.2445101142`입니다. 고정된 comparison JSON의 failure 목록은 비어 있습니다.

이 통과 결과는 인증된 RGBA/dropout 전체 모델 경로를 증명합니다. 기본 자동 깊이 경로나 직렬화된 무옵션 CLI 경로는 아직 증명하지 않습니다.

### 4.4 엄격한 기본 경로 및 CLI parity 결과

수정된 런타임의 canonical default reference는 정렬된 좌표 6,307개를 정확히 반복 생성했습니다. 14개 텐서 중 12개가 bit-exact였고 두 mesh 텐서의 최대 차이는 `2.086162567138672e-7`이었습니다. 이전 `b000e46c` Rust 캡처는 완료되었지만 좌표 6,300개, 누락 52개, 추가 45개로 좌표 gate에 실패했습니다. 해당 immutable 실패 artifact는 그대로 보존됩니다.

`6bd85cf3` 기본 v7 캡처는 child, wrapper, guard exit가 모두 0이었고 고정된 비교기를 통과했습니다.

| Metric | 계약 | 결과 | 상태 |
|---|---:|---:|---|
| 정렬된 점유 좌표 | 정확히 일치 | 6,307개 중 6,307개 정확히 일치 | Pass |
| Latent 상대 RMS | `< 1e-2` | `0` | Pass |
| Mesh 정점 수 상대 오차 | `<= 1%` | `0` (189,623개 중 189,623개) | Pass |
| 정규화 대칭 전체 집합 Chamfer | `< 1e-3` | `5.141699913725434e-6` | Pass |

이전 `68cd161e` identity의 실제 무옵션 CLI 실행은 exit 0이었지만 좌표 6,295개로 수치 인수에 실패했습니다. 직렬화된 텐서는 같은 소스 integration capture와 bit-exact였으므로 실패한 예측의 직렬화 충실도는 확인했습니다. 인수된 소스의 literal-default v4 재실행은 child, wrapper, guard, comparison tool exit가 모두 0으로 통과했습니다. GPU, binary SHA-256 `c25ac839e070c2ab450a6bc543898e198ff438ea0f6957a0ee933b4cfe696be3`, 문자 그대로의 CLI 기본값을 사용했습니다. 예상 파일 5개가 모두 존재하고 인증되었으며, 좌표가 정확히 일치하고 latent 상대 RMS는 0, 메시 정점 수는 189,623개로 정확히 일치하며 정규화 Chamfer는 `5.141710000502124e-6`이었습니다.

Pose, mesh export, Gaussian export field 비교는 엄격한 인수 metric 밖의 진단 전용 관찰로 보존합니다. 특히 row-aligned OBJ face 배열은 정확히 일치하지 않았으므로 이 보고서는 정확한 face 순서를 주장하지 않습니다. 완전한 export 파일과 필수 schema 및 의미는 CLI gate를 통과했습니다.

## 5. 검증 증거

| 검증 | 증거 |
|---|---|
| 저장소 계약 및 포맷 | 범위 내 저장소 계약 8개와 포맷 검사를 통과했습니다. |
| ROCm 테스트 빌드 | Child와 wrapper가 0으로 종료했습니다. Cargo는 9분 50초에 `test-fast` profile을 완료하고 immutable integration, CLI, unit-test artifact를 생성했습니다. |
| 집중 단위 테스트 | 총 61개를 통과했습니다. MoGe refiner 6개, six-voxel sparse oracle 1개, grid-budget 1개, SAM 3D Objects 40개, 기본 DINO 8개, Objects DINO 정책 5개입니다. Receipt executable SHA-256은 `368e71d0ffe7dd8ca5b88c5ba3ec3a44dac7bbb21a69b17d75c745619c907765`입니다. |
| `b000e46c` 수정 gate | 집중 테스트 17개를 통과했습니다. Objects resize 8개, Objects DINO 7개, UV endpoint 1개, geometry recovery 1개입니다. Executable SHA-256은 `bc417a120c1f2dce149f77d0dcf28534bd1dd0a43cdc456749d87ca241479228`이며 Clippy, 저장소 계약 8개, CI도 통과했습니다. |
| Exact focal solver 테스트 | 5개를 통과했으며 해당 backend의 실제 solver 상태 31개가 모두 bit-exact였습니다. Executable SHA-256은 `1f275327a08cbbfd1083a4a40479235436e4a1f9756e297afde1a34075bf29f6`입니다. |
| 현재 빌드, lint 및 CI | 앞선 overall-status-101 시도는 무관한 benchmark linker `SmallVector` abort를 만났고, 변경 없는 재시도는 6.31초에 완료되었습니다. 두 이력을 모두 보존합니다. 필수 Clippy v4는 1분 57초에 통과했고 committed `6bd85cf3` CI는 모두 green입니다. Integration SHA-256은 `df1db4715beced04f2415cb5aae268140a0ce921d88534b981c036a5bf45857e`, CLI SHA-256은 `c25ac839e070c2ab450a6bc543898e198ff438ea0f6957a0ee933b4cfe696be3`입니다. 최종 report-head CI는 아직 남아 있습니다. |
| 인수된 RGBA 소스 identity | Head `68cd161e1a45018559d00fca61e5516f99a7f803`, manifest `391be3af3fec1a7552f6355dac55cec5b13dcdc1b5259148c1ac159ca3865e4e`, archive `fdf3bb87f072fcd021b65a15405e366a98cdebf98c05fee50997220191f2ebff`입니다. |
| 인수된 기본 소스 identity | Head `6bd85cf38e70ea43f2ba2e5a63363c35193c2db7`, manifest `dd573f15f508ce241e51658260e0faf7baf88f6edbd678c12ebe60bb971d8981`, archive `f2224b6175c38f54e2b84070cf28710ec8beb25b6c6a1e990a44135a9b2cfdb5`입니다. |
| Canonical upstream RGBA reference | Child, wrapper, guard가 0으로 종료했습니다. Receipt SHA-256은 `1f08fdffb6d1425ae779bfffdba3c546de39aa74e35a004d6ed3d1cb2bca0dc6`, reference safetensors SHA-256은 `8d8cc32b5a5d0f223d6bce474e7f90ff4eff11cf6c28c58eeaacd230393eecaf`입니다. |
| 인증된 Rust RGBA capture | Receipt SHA-256은 `f9241815539d5779b86a6edfcd44a573cee820b69648bf248b539d507c820013`, capture safetensors SHA-256은 `3f3922d9967d472ce0a81f5a889f1e5012b3ae7bf7c7f1ca0aad83f5a0d95cf0`입니다. |
| 고정 비교기 | Comparator `d919c9b740afe05359f14f592b9416cafd1d833ed8ea67a698d94eb17d809580`, 계약 `afe87b621e49c59742c11d71d8f314592b6acabb0ba5664de49a1285759d3d07`, comparison JSON SHA-256 `a0e699db55189851c8de5874ef39e17d2bfe5e420786eb12a64dc77102332809`입니다. |
| 전체 RGBA 비교 | 정확한 좌표, latent RMS 0, 정점 수 오차 0.0158948%, 정규화 전체 집합 Chamfer `4.450226e-6`로 엄격 비교를 통과했습니다. |
| MoGe 차이 위치 이력 | 이전 43단계 진단은 39단계 exact, point map raw 상대 RMS `4.0945834798765365e-8`이었습니다. Solver 수정 후 기본 conditioning 17단계가 모두 bit-exact였습니다. Actual 출력 SHA-256은 `9b3639a65168c7db5c3df5bf1c4f970bc806dd8ca0f48328ee64d3a0cefb97ec`입니다. |
| 전체 기본 자동 깊이 비교 | **Pass.** 좌표 6,307개 exact, latent RMS 0, 정점 수 exact, 정규화 Chamfer `5.141699913725434e-6`입니다. 출력 SHA-256은 `f1820a3ff55bbbb29f0ff0404cf51c06b8f5be076f82e78afcd15ee2b5378cbe`, receipt SHA-256은 `108b97bd77042e03d75e408e7108f6182fec0720f75b837c13b9abe81614991a`입니다. |
| 실제 무옵션 CLI | **Pass.** 문자 그대로의 기본값, 예상 파일 5개, 좌표 6,307개 exact, latent RMS 0, 정점 수 189,623개 exact, 정규화 Chamfer `5.141710000502124e-6`입니다. Capture receipt SHA-256은 `9e35367675653e55d34ddf6975f1c9f5f0a246d326b77df83178f6e59c56f299`, comparison SHA-256은 `864c58a61faaf27303f0c07da53698133d991fb72344696cb478877d0789a5de`입니다. |
| 외부 export 상호 운용성 | Trimesh 5.1.1은 OBJ 정점 189,623개와 face 376,618개를 읽었고, plyfile 1.1.5는 필수 property 17개를 가진 Gaussian 201,824개를 읽었습니다. 상호 운용성 JSON SHA-256은 `48f34c90b6b168a34ab22a33bc8f2938a637951a055457ee02542afbe02ed527`입니다. 이는 parser 상호 운용성 결과이며 rendering 또는 face 순서 주장이 아닙니다. |
| 최종 fixture helper | 제한된 mock test 33개가 통과했습니다. 최종 CPU generator는 정확한 31-state focal fixture(`cf5bf5c2...`)와 resize payload를 재현했습니다. 복사된 child/wrapper/guard status는 hash로 인증되며 portable generator는 실행 helper를 hard-pin합니다. |

## 6. 변경 통계

| 항목 | 값 |
|---|---:|
| 변경 파일 | 181 |
| 추가 줄 | 18,221 |
| 삭제 줄 | 455 |
| 인수된 RGBA head | `68cd161e1a45018559d00fca61e5516f99a7f803` |
| 인수된 기본 head | `6bd85cf38e70ea43f2ba2e5a63363c35193c2db7` |
| Base | `869af0c296e748e8c62075bd53c5cd18db5cc874` |
| 통과한 집중 테스트 | 61 |
| 인수된 전체 모델 경로 | 필수 2개 중 2개 |

## 7. 후속 작업

### 남은 게시 단계

- [x] 최종 fixture bundle 검토와 staging을 완료하고 검토 지적 사항을 수정합니다.
- [x] 마지막 fixture inventory signoff를 완료했습니다.
- [ ] 최종 문서와 fixture를 commit한 뒤 해당 head에서 CI를 실행합니다.
- [x] 무시된 두 보고서 파일만 정확한 경로로 force-stage했습니다.

### 증거의 한계

- LayerNorm 소스 race는 독립적으로 확인했지만, 안정적이었던 작은 probe는 해당 race가 이전 전체 파이프라인 불안정성을 일으켰음을 증명하지 않습니다.
- 이전 5,821/5,783 실행과 모든 custom-stream 실패는 진단 증거로만 유지합니다.
- 39개 instrumented stage의 정확한 일치는 관찰이 평가 경계를 바꾸었으므로 실제 instrumentation 없는 trajectory를 증명하지 않습니다.
- 인수된 RGBA 비교는 위에 기록된 인증 소스, 바이너리, checkpoint, 입력, 옵션, native runtime, comparator identity에 적용됩니다.
- 43단계 MoGe 진단은 과거 기본 경로 차이의 위치를 확인했지만, 현재 완료된 instrumentation 없는 기본 비교를 대신하지 않았습니다.
- RGBA 결과는 `68cd161e`에 묶여 있으며, 이후 소스에는 깊이 전용 변경의 검토된 소스 동등성을 통해서만 적용됩니다.
- 기본 자동 깊이와 literal 무옵션 CLI는 `6bd85cf3`에서 인수되었습니다. Pose와 export 비교는 진단 전용이며 정확한 OBJ face 순서를 주장하지 않습니다.
- Fixture staging과 마지막 inventory signoff는 완료했지만 최종 문서/fixture commit의 CI는 보고서 완료 시점에 남아 있었습니다. Green인 `6bd85cf3` CI를 final-head CI로 제시하지 않습니다.

## 참고 자료

- PR #2578: SAM 3D Objects reconstruction
- Issue #2381: 메시 및 Gaussian 출력을 포함한 SAM 3D Objects 이식
- Epic #2382: dense-prediction model delivery
- `docs/dense-prediction.md`: CLI 및 출력 계약
- `tests/fixtures/sam3d_objects/README.md`: reference 및 capture 절차
- `tests/fixtures/sam3d_objects/provenance.json`: 소스, 런타임, checkpoint, 입력 identity
- `tests/fixtures/sam3d_objects/comparison-contract.json`: 고정된 전체 배열 인수 metric
- `.artifacts/sam3d_objects/reference-canonical-ln-*`: 수정된 런타임 canonical reference
- `/tmp/objects-full-rgba-v5-comparison.json`: 인수된 RGBA 엄격 비교
- `/tmp/objects-moge-stages-v2-comparison.json`: 43단계 기본 깊이 차이 위치 진단
- `/tmp/objects-default-v6-coordinate-failure.json`: 완료된 기본 실행의 좌표 실패
- `/tmp/objects-default-cli-integration-comparison.json`: 실제 CLI와 같은 소스 integration 출력 비교
- `/tmp/objects-default-stages-v3-comparison.json`: 수정 후 17단계 exact 비교
- `/tmp/objects-full-default-v7-comparison.json`: 인수된 기본 경로 엄격 비교
- `.artifacts/sam3d_objects/cli-default-v4/comparison.json`: 인수된 literal-default CLI 비교
- `/tmp/objects-cli-v4-interop.json`: 독립 OBJ/PLY parser 상호 운용성
