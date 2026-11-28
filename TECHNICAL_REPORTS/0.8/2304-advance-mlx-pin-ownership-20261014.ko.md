# 기술 보고서: PR #2304 - MLX 핀 업데이트 및 백엔드 소유권 조정

- **작성일**: 2026-10-14
- **상태**: 구현 및 사용 가능한 하드웨어 검증 완료; 일반 PR CI는 대기 중이며 expanded 결합 통합 게이트는 연기됨
- **위험도**: 높음 (111개 커밋 규모의 의존성 업데이트가 CPU, CUDA, Metal, ROCm 통합을 변경하며 ROCm 하드웨어 검증은 수행할 수 없음)

## 요약

PR #2304는 고정된 `ml-explore/mlx` 소스를 `59d600b5e64c238427d0f8d897ab7c682ef4d3d2`에서 `243bdf09c0cccbb85ca6a686dab3cc0102cf1f7e`로 업데이트합니다. 이 구간에는 업스트림 커밋 111개와 변경 파일 312개가 포함됩니다. 모든 CPU/CUDA/Metal 전체 파일 오버레이를 조정하고, ROCm 오버레이를 move-only 할당자 소유권과 신규 프리미티브에 맞추며, 필요한 수치 정확도 및 런타임 동작을 보존하고, 소유권 및 정렬 MXFP 테스트를 추가합니다. CUDA 컴파일 및 수치 테스트, 명시적 RHS 경로 실행, 결합 트리 Kimi 컨트롤, 프로덕션 헤드 Metal CI가 통과했습니다. 실제 ROCm GPU 및 M5 전용 검증은 수행할 수 없습니다.

## 1. 문제 정의

### 1.1 배경

mlxcel은 고정된 MLX 체크아웃 위에 로컬 소스 오버레이를 복사합니다. 핀만 업데이트하면 오래된 전체 파일이 업스트림 수정사항을 조용히 덮어쓸 수 있으므로 각 오버레이를 순수 Git 객체와 비교하여 리베이스하거나, 경로 드리프트가 0임을 확인하여 유지하거나, 제거해야 합니다. 이번 핀은 백엔드 전반의 할당자 소유권과 프리미티브 API도 변경합니다.

### 1.2 확인된 위험

- **오버레이 덮어쓰기**: 오래된 전체 파일 오버레이가 111개 커밋 구간의 업스트림 변경을 유실시킬 수 있습니다.
- **소유권과 수명**: 업스트림 할당은 이제 move-only `allocator::Data`를 반환합니다. 잘못 전달하면 이중 해제, 빌린 메모리의 donation, 비동기 작업 종료 전 버퍼 해제가 발생할 수 있습니다.
- **백엔드 동작**: CUDA gathered GEMM, Metal gathered-row 브로드캐스트, ROCm 폴백, SDPA sink, floor division, gated-delta VJP, 홀수 길이 inverse FFT가 모두 갱신된 의존성 경계를 통과합니다.
- **검증 귀속**: Kimi는 독립 PR #2296의 소스 수정이 필요합니다. 결합 검증 트리는 두 소스 헤드를 모두 기록하고 해당 모델 수정을 PR #2304의 변경으로 귀속하지 않아야 합니다.

## 2. 구현 상세

### 2.1 CPU, CUDA, Metal 오버레이 조정

기존 인벤토리는 업스트림 기반 오버레이 30개와 로컬 헬퍼 2개였습니다. 변경된 파일 11개를 리베이스하고, 경로별 드리프트가 0인 파일 17개와 추출된 로컬 헬퍼 2개를 유지했습니다. 오버레이 2개를 제거했습니다. 병합된 업스트림 #4514는 CUDA 핸들 오버레이의 non-throwing 종료 의미를 제공하며, 업스트림 #4554는 `cuda/device/qmm_sm80.cuh`에 있던 #910 async-copy 완료 및 CTA 순서 residual 전체를 제공합니다. 겹치는 업스트림 #4531은 #4514가 해당 내용을 포함하여 병합 없이 닫혔습니다. #910 provenance와 로컬 barrier hunk를 제거하면 target의 순수 파일과 바이트 단위로 동일합니다. 최종 인벤토리는 유지된 파일 30개입니다. `docs/upstream/mlx-pin-243bdf09-reconciliation.md`는 각 파일의 처리 방식과 보존된 동작을 기록합니다.

보존된 동작에는 CPU float 누적, CUDA 배치 청킹 및 dtype/런타임 제어, 업스트림 gathered GEMM 경로, Metal gathered matrix 두 경로의 #1599 `B * M` 브로드캐스트 행 수정이 포함됩니다. Xcode 27 cooperative-tensor 우회는 업스트림 #4594를 사용하므로 과거 전체 파일 우회 구현은 가져오지 않습니다.

### 2.2 할당 소유권 및 ROCm 리타깃

CUDA reduction 중간 버퍼는 이제 `allocator::Data`를 직접 전달합니다. 공개 deleter 생성자는 빌린 저장소를 뜻하므로 ROCm은 owned-data 브리지를 추가합니다. 빌린 slab shell은 커스텀 release를 유지하며 donation 대상이 아닙니다. 리타깃은 포크 커밋 `75915908dfe5028335d318b10340313744fd3a8d`를 유지하고, 백엔드 파일 109개, 코어 파일 16개, 로컬 수정 47개, 검토된 코어 residual 5개를 기록합니다. 또한 compiled callback, reduction 중간 버퍼, SDPA sink, gated-delta VJP 폴백, floor division, 홀수 inverse FFT 출력 길이를 새 API에 맞춥니다.

호스트 소유권 회귀 테스트는 실제 오버레이 헤더를 사용하여 move-only 전달, owned/borrowed 해제, 빈 할당, 공유 소유권 유지, 예외 정리를 검증합니다. pending command 소유권을 모델링하지만 HIP을 실행하거나 GPU 완료 순서를 증명하지는 않습니다.

### 2.3 CUDA Gathered GEMM 후속 수정

커밋 `6424eb916e2bdfe7c576cba9f8991d9b614ae9d9`는 보존된 gathered GEMM 타일 선택기에 `encoder.device()`를 전달하여 최초 후보 빌드 실패를 수정합니다. 커밋 `5a0d98303f9830ecf9cd34501dc48c3a6db8306f`는 기존 N=320 테스트가 정렬 RHS admission을 만족하지 않으므로 N=384 sorted-prefill MXFP4/MXFP8 참조 테스트를 추가합니다. 기본 128행 실행은 유지된 #629 grouped GEMM을 먼저 선택하므로 경로 검증에서는 문서화된 `MLXCEL_GATHER_QMM_GROUPED=0` 제어를 사용합니다. 기본 타일 선택과 명시적 M=128 선택이 모두 통과하고 f16/bf16 × MXFP4/MXFP8에 대한 새로운 RHS PTX를 생성합니다.

## 3. 리뷰 상태

- 소스 통합 리뷰는 `6424eb916e2bdfe7c576cba9f8991d9b614ae9d9`까지 차단 이슈가 없음을 확인했습니다.
- 독립적인 오버레이 및 할당 소유권 리뷰에서도 차단 이슈가 발견되지 않았습니다.
- 오래된 CUDA ops provenance 주석은 `5a0d98303f9830ecf9cd34501dc48c3a6db8306f`에서 수정되었습니다.
- 보안 리뷰에서 critical, high, medium 발견 사항은 없었습니다. 낮은 심각도일 수 있었던 #910 barrier 중복은 업스트림 #4554가 기능 residual 전체를 포함함을 증명하고 오버레이를 제거하여 해결했습니다.

## 4. 검증

### 4.1 통과

1. 핀/파서 검증은 정확히 `243bdf09c0cccbb85ca6a686dab3cc0102cf1f7e`를 해석하며 27개 테스트가 모두 통과했습니다.
2. CPU/CUDA/Metal 순수 객체 provenance 검토에서 설명되지 않은 경로 드리프트나 충돌 마커가 없습니다. ROCm 레코드 및 순수 객체 드리프트 검사도 통과했으며 오버레이 도구 검사 29개와 호스트 소유권 회귀 테스트를 포함합니다.
3. 버전, 커널 dtype 키, 커널 포트 디스패치, llama 호환성, Python 도구 123개 테스트, 포맷 컨트랙트 게이트가 통과했습니다.
4. 소스 `6424eb916e2bdfe7c576cba9f8991d9b614ae9d9`의 CUDA no-run 컴파일은 누락된 device 인자를 수정한 뒤 14분 23초에 완료되었습니다.
5. 소스 `6424eb916e2bdfe7c576cba9f8991d9b614ae9d9`에서 10개 집중 수치 그룹이 실패 없이 완료되었습니다. 로그에는 94회 테스트 실행이 기록되며 타일 override 그룹이 MXFP 테스트 5개를 의도적으로 반복하므로 94는 고유 테스트 수가 아니라 실행 횟수입니다. 동결 바이너리는 `candidate-2303-numerics/provenance.json`에 기록되며 SHA-256은 root `8477d85748c60dd1298ab71b4c0e2f8c615f516583641b2ccff562a1e5b60f03`, core `e1734a8aae76a7809e5e9d7bcb3046de585b3833144ee1cfe334a14b1ff9b77e`, surgery `8b873c576b8696806ff125caf2d61e7c637a09b6562fc9ca478181aa3bb1218e`입니다.
6. 프로덕션 헤드 Apple Silicon 워크플로 실행 `38492190049`는 `3aeccc2b7ac39ea17be3e9959cbb4111cfc02782`에서 통과했습니다. 테스트 12,285개 통과, 0개 실패, 390개 무시와 비디오 테스트 51개 통과, 0개 실패를 기록했습니다. 실제 단계 로그는 `/tmp/mlxcel-auto-20261012T1250/logs/metal-2303-3aeccc2b-pass.zip`에 보관됩니다.
7. 제거된 generated-cache 헤더는 프로덕션 헤드 재빌드 전에 순수 핀 `243bdf09`에서 복원되었습니다. `/tmp/mlxcel-auto-20261012T1250/2303-generated-cache-restoration.json`은 캐시 파일 SHA-256이 `a9046084ccb93381e366e0d9d6730a4fd1cbd0d49e3dff5a43d2df7e66ffbebf`에서 순수 파일 `c3c4ba7727db7e67929fa4966d6fffa4d11bf47bb8773213c7d8899ec03b9374`로 변경되었음을 기록합니다.
8. 결합 소스 `53473340`은 3분 57초에 빌드됩니다. 최종 기본 수치 실행에서 switch-layer 테스트 38개, 명시적 정렬 M=128 테스트 1개, global-scale 테스트 6개가 통과했습니다(총 45개, 실패 0개). 결과는 `/tmp/mlxcel-auto-20261012T1250/2303-final-numerics-validation.json`에 기록됩니다.
9. 명시적 RHS 검증은 `MLXCEL_GATHER_QMM_GROUPED=0`을 사용한 N=384 실행 2개를 모두 통과했습니다. 하나는 기본 타일 선택, 다른 하나는 `MLXCEL_QMM_TILE_M=128`을 사용합니다. 새로운 캐시에 float16 및 bfloat16과 MXFP4 및 MXFP8 조합을 포괄하는 `gather_qmm_rhs_sm80` PTX artifact 4개가 생성되었고 provenance 및 바이너리 해시는 `/tmp/mlxcel-auto-20261012T1250/2303-rhs-validation.json`에 기록됩니다.
10. 결합 트리 Kimi 검증은 집중 테스트 3개와 정확 일치 모델 컨트롤 5개를 통과했습니다. 여기에는 답 `4`, 마커 `AMBER-RAVEN-7429`, 청킹 및 비청킹 prefill, 기본 SDPA가 포함됩니다. 별도로 검토한 58-token 출력은 Rayleigh scattering을 정확하고 일관된 두 문장으로 설명하며 token-zero 반복이 없습니다. 소스 `5347334039e48d9444be77f2aebeaac17c35c6d1`, 핀 `243bdf09`, 바이너리 SHA-256 `b52d70388e0420582c4f60f3341d570c2afd736685b37e2c0c6e50ed9eaf5fae`는 `2303-latestpin-model-validation.json`과 `2303-latestpin-natural-language-validation.json`에 기록되며 결합 트리의 독립 PR #2296 수정은 별도로 귀속됩니다.

### 4.2 대기 또는 수행 불가

- **일반 PR CI**: runner 용량을 기다리는 작업 7개가 queued 상태입니다. queued 상태를 실패나 통과로 간주하지 않습니다.
- **ROCm**: 이 호스트에는 ROCm 도구 체인과 AMD 하드웨어가 없습니다. `check_api_drift.sh`, ROCm 컴파일, 실제 encoder 수명 실행, `make verify-rocm`, ROCm 추론은 수행되지 않았습니다.
- **M5 전용 실행**: M5 결과가 없으며 다른 Apple Silicon 결과로 대체하지 않습니다.
- **최종 결합 게이트**: 사용자가 요청한 전체 통합 실행은 모든 expanded 후속 작업이 준비될 때까지 연기되었습니다. 이 PR의 후보 검증은 최종 전체 배치 게이트가 아닙니다.

## 5. 변경 요약

| 항목 | 값 |
| --- | ---: |
| 변경 파일 | 47 |
| 추가 라인 | 1,646 |
| 삭제 라인 | 767 |
| 업스트림 구간 | 커밋 111개 / 파일 312개 |
| 제거 후 CPU/CUDA/Metal 오버레이 | 30개 |
| 집중 수치 테스트 실행 | 94회 통과 / 0회 실패 (MXFP 5개 반복으로 고유 테스트는 89개) |
| 프로덕션 헤드 Metal 워크플로 | 12,285개 통과 / 0개 실패 / 390개 무시, 비디오 테스트 51개 추가 통과 |
| 결합 트리 Kimi 검증 | 집중 테스트 3개 + 정확 일치 컨트롤 5개 + 58-token 자연어 검사 통과 |

## 6. 후속 조치

- [ ] queued 상태인 일반 PR CI 작업 7개가 완료되면 결과를 검토합니다.
- [ ] AMD 호스트에서 ROCm 드리프트/빌드/런타임 게이트를 실행하거나 수행 불가 기록을 유지합니다.
- [ ] expanded 배치가 준비된 뒤에만 전체 CUDA 테스트/Clippy 게이트와 새로운 main Metal nightly를 실행합니다.

## 부록: 증거 범위

이 보고서는 PR #2304가 열린 초안 상태에서 확보된 증거를 기록합니다. 병합 권한을 부여하지 않으며 PR #2304가 #1599, #2158, #1986을 종결한다고 주장하지 않습니다. 또한 명시된 소스 동등성 범위를 넘어서 이전 핀 Kimi 결과, 호스트 전용 ROCm 소유권 테스트, 다른 소스 리비전의 Metal 결과를 현재 헤드 검증으로 간주하지 않습니다.

## 검증 이후 최신 main 동기화

#2305의 mlxcelverse 디렉터리 이동 및 이후 ROCm·MoE·요청 메모리 변경을 포함한 main `59974283`까지 병합했다. 폐기한 CUDA 오버레이 두 파일은 삭제를 유지하고 `mlx/allocator.h`는 `src/lib/mlxcelverse/overlays/rocm/`으로 이동했다. 호스트 소유권 검사 경로, 로컬 수정 번호 중복(최종 48개), 오버레이 문서 경로·개수를 수정했다. main의 HIP 그리드 제한, SDPA 최적화, dequant 캐시 기본값 및 MoE 변경은 보존했다.

병합 후 CUDA 테스트 빌드, 포맷, pin/parser 27개, 오버레이 provenance, ROCm residual·도구 검사 29개, 호스트 소유권 회귀, 계약 게이트 및 Python 123개가 통과했다. GPU 잠금 아래 요청 메모리 25개, Switch 38개, Inkling MoE 3개, DeepSeek V4 45개, 강제 RHS 1개가 통과했다(정렬 사례 재실행을 포함한 112회). 새 RHS PTX 네 개로 f16/bf16 × MXFP4/MXFP8 경로 실행을 확인했다. 로그와 정확한 소스 트리·바이너리 해시는 `/tmp/mlx-2304-merge-runtime.json`에 있다. 이전 Metal·실제 Kimi 검증은 기록된 소스에 대한 결과이며 이번 동기화에서 재실행하지 않았다.

최종 CI에서 기존 rustls 0.23.45에 대한 새 RustSec 권고 네 건(RUSTSEC-2026-0346~0349)이 확인됐다. rustls만 0.23.46으로 갱신했고 예외 추가 없이 로컬 `cargo deny check`를 통과했다.
