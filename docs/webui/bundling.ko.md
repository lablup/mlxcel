# 번들 WebUI 개발

[English](bundling.md)

## 범위

이슈 #1836은 최소 React 셸, 재현 가능한 정적 자산, 재사용 가능한 Rust 정적 라우터를 제공했습니다. 이슈 #1838은 이제 `mlxcel-server --webui`와 `mlxcel serve --webui`에서 WebUI 보안 래퍼와 typed UI API 뒤에 번들을 마운트합니다. 운영 페이지는 여전히 에픽 #1834의 후속 이슈에서 완성됩니다. 셸은 이후 채팅·시각화·다운로드·상세 metrics 페이지까지 완료됐다고 암시하지 않고, 각 기능의 실제 백엔드 구현 상태를 표시해야 합니다.

프런트엔드는 기존 `webpage/` 사이트와 별개인 `webui/`에 있고, 배포 자산은 `src/webui/assets/`에 있습니다. Cargo는 커밋된 파일을 임베드하며 Node 실행이나 프런트엔드 의존성 다운로드를 하지 않습니다. 선택적 Cargo feature인 `webui`는 기본 활성화됩니다. `rust-embed`의 `debug-embed` 설정으로 개발용·릴리스 산출물 모두 자산 바이트를 포함하며 실행 시 소스 체크아웃을 읽지 않습니다.

## 표준 빌드 파이프라인

Python 3, Node **26.5.1**, pnpm **11.18.0**을 설치한 뒤 저장소 루트에서 실행합니다.

```bash
pnpm --dir webui install --frozen-lockfile
pnpm --dir webui run typecheck
pnpm --dir webui run lint
pnpm --dir webui run unit
python3 scripts/webui/build_bundle.py
make verify-webui-bundle
```

Python 진입점은 정확한 Node/pnpm 버전을 확인하고 Vite 실행, 매니페스트 작성, 결과 검증을 수행합니다. 프런트엔드 소스나 빌드 입력 변경 후 이 진입점을 실행하고 자산과 매니페스트를 함께 커밋하세요. `pnpm --dir webui run build`는 하위 Vite 단계일 뿐, 매니페스트를 갖춘 배포 번들을 완성하지 **않습니다**. 커밋할 자산을 이 명령만으로 갱신하지 마세요.

`make verify-webui-bundle` 또는 `pnpm --dir webui run verify-generated`는 임시 디렉터리에서 두 번 새로 빌드하여 바이트를 비교하고, 커밋된 번들과도 비교합니다. 누락 파일, 오래된 소스 다이제스트, 자산 해시 불일치, 크기 예산 초과를 거부합니다. 의존성을 설치하지는 않습니다. 특정 Python이 필요하면 `WEBUI_BUNDLE_PY=/path/to/python3`을 지정하세요.

매니페스트는 고정 패키지·도구 버전, 소스 다이제스트, 파일별 SHA-256 및 원본/gzip 크기, 번들 예산을 기록합니다. 타임스탬프나 절대 빌드 경로는 포함하지 않습니다. 생성 자산에 소스 맵을 넣지 않고 HTML은 상대 자산 URL을 사용합니다. 런타임 CDN, 서비스 워커, SSR, Node 서비스는 필요하지 않습니다. React·React DOM·scheduler의 라이선스는 `third-party-licenses.txt`에 포함되며 저장소 `NOTICE`에도 출처를 기록합니다.

| 예산 | 상한 |
|---|---:|
| 초기 JavaScript, gzip | 200 KiB |
| 전체 JavaScript, gzip | 700 KiB |
| 매니페스트 포함 전체 임베드 자산 | 5 MiB |

현재 측정값은 `src/webui/assets/mlxcel-webui-manifest.json`에서 확인하세요. 이전 번들의 수치를 새 PR에 옮기지 마세요. 후속 채팅·시각화 구현도 필요하면 지연 로딩을 적용해 이 상한을 유지해야 합니다.

## Rust feature와 라우터 경계

```bash
# Apple Silicon 배포 빌드: 기본값에 WebUI 포함.
cargo build --release --features metal,accelerate
# 보통 기본 활성화되는 surgery를 유지하면서 WebUI 제외.
cargo build --release --no-default-features --features metal,accelerate,surgery
# 다른 기본 feature 없이 WebUI를 명시적으로 선택.
cargo build --release --no-default-features --features metal,accelerate,webui
# Linux/CUDA는 기존 백엔드 전제 조건과 도구 체인을 사용.
cargo build --release --features cuda
```

`mlxcel::server::webui::router::<S>()`의 라우트에는 이미 `/webui`가 포함됩니다. 루트에 병합하거나 **검증된 서버 API 접두사** 아래 한 번 중첩하세요. `/webui` 아래 다시 중첩하지 마세요. 부모 라우터의 루트 상태 확인·추론 경로는 유지됩니다. `/webui`는 `/webui/`로 리디렉션되고, `#models` 형태의 탐색은 클라이언트 안에서 처리됩니다. 없는 정적 경로는 API 라우팅 오류를 감출 수 있는 HTML 대체 응답 대신 404를 반환합니다.

정적 응답은 GET/HEAD와 조건부 ETag를 지원하고, 다른 메서드는 `Allow: GET, HEAD`와 함께 405를 반환합니다. HTML·매니페스트는 재검증하며 콘텐츠 해시 자산에는 immutable 캐시를 허용합니다. 라우터는 인코딩·경로 탐색 표현을 거부하며 일반·오류·리디렉션·304 응답에 동일 출처 CSP, `nosniff`, no-referrer 헤더를 설정합니다. 이 정적 정책이 후속 관리 API의 Host/Origin/인증 검사를 대신하지는 않습니다.

## 정적 자산과 운영 시작 경로의 분리 검증

```bash
pnpm --dir webui exec playwright install chromium
pnpm --dir webui run browser
cargo test --profile test-fast --features metal,accelerate server::webui::assets::assets_tests
cargo run --example webui_static_harness --features metal,accelerate
```

브라우저 명령은 Vite preview 서버에서 셸을 확인합니다. 운영 서버·Safari·인증·실모델 인수 테스트가 **아닙니다**. 정적 하네스는 루프백 주소를 출력하고 상태 확인·모델 스텁 경로와 `/webui/`를 제공하므로 재배치와 임베딩 검증에는 계속 유용하지만, `mlxcel-server --webui` 시작, 생성된 터미널 자격 증명, Host/Origin/Fetch-Metadata 검사, 모델 없는 카탈로그 접근의 증거는 아닙니다. 이러한 항목은 운영 바이너리와 실제 마운트된 라우트로 검증해야 합니다.

이동한 산출물 검증에서는 개발용·릴리스 하네스를 각각 빌드하여 체크아웃 밖으로 옮기고, 소스 자산 디렉터리와 `node_modules` 접근을 차단하며, 루프백 HTTP만 남기고 외부 네트워크를 금지해야 합니다. 각 산출물에서 셸·매니페스트·JavaScript·CSS를 가져와 확인하세요. 기존 MLX 동적 라이브러리·Metal 리소스 요구 사항은 별도로 다뤄야 합니다. WebUI 자산의 독립성을 입증하는 것이 모든 MLX 런타임 리소스의 임베딩을 의미하지는 않습니다. 이 안내를 실행 완료로 간주하지 말고 실제 실행 근거와 수행하지 못한 feature·백엔드 검증을 기록하세요.
