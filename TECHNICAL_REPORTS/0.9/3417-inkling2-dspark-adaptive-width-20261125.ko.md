# PR #3417: Inkling 2 DSpark의 acceptance 기반 verify 폭

이슈 #3410, #3280 / PR #3405의 후속이다.

## 문제

PR #3405에서 Inkling 2 verify 블록이 decode와 바이트 단위로 같아졌고, 그래서 DSpark가 기본으로 켜졌다. 그런데 Small에서는 산문 프롬프트에서 drafter가 여전히 classic보다 느렸다(0.90x). 원인은 비용이다. Small은 토큰마다 384개 NVFP4 expert 중 8개로 라우팅되므로, verify 행이 하나 늘 때마다 expert 약 8개 분량의 가중치를 더 읽는다. gather 경로에서 8행 라운드는 150 ms가 걸렸고, decode 한 스텝은 52~56 ms였다. 산문은 라운드당 약 2.3토큰을 수락하는데, 이걸로는 라운드 비용을 회수하지 못한다.

## 방법

1. **라운드마다 폭을 고른다.** #3280의 row-invariant forward 덕분에 폭 1~16의 어떤 블록도 같은 수의 decode 스텝과 결과가 같다. 그래서 target이 `exact_at_every_verify_width`를 선언할 수 있다. 이런 target에서는 round loop가 `ExpectedThroughputWidth`를 쓰고, 동작은 다음과 같다.
   - draft 위치별 조건부 acceptance를 감쇠 추정한다. 이전 위치의 값을 사전값으로 이어받는다.
   - 실제로 돌려 본 폭은 측정 비용의 EMA를 두고, 돌려 보지 않은 폭은 선형 보간한다.
   - 매 라운드 E(토큰)/C(폭)이 가장 큰 폭을 고른다. 폭 1은 draft 없는 classic decode 한 스텝이다.
   - classic 스텝이 8번 이어지면 draft 폭을 다시 시도한다. 그래서 텍스트가 반복적으로 바뀌면 drafter가 돌아온다.
   - 각 폭의 첫 라운드는 비용 표본에서 뺀다. 그 폭의 커널을 컴파일하느라 230~260 ms가 걸리기 때문이다(정상은 80 ms). 초기 실행에서는 이 표본 하나 때문에 정책이 그 폭을 영영 버렸다.
2. **classic 스텝을 파이프라인으로 돌린다.** 한 번에 하나씩 돌리면 스텝마다 자기 결과를 읽을 때까지 기다리느라 57~58 ms가 걸렸다(classic decode는 52 ms). 이제 `classic_steps.rs`가 classic decode와 같은 방식으로, 스텝 i의 lazy argmax 위에 스텝 i+1을 쌓는다. 스텝마다 capture한 hidden 행을 모아 두어서 다음 draft가 커밋된 문맥 전체를 본다.
3. **Small에서도 grouped expert matvec을 쓴다.** upstream은 이 커널을 `loop_config`로 제한하는데, Small에는 `loop_config`가 없다. 이 커널은 expert마다 가중치를 한 번만 디코드해서 그 expert로 라우팅된 모든 행에 쓰고, 결과는 gather 경로와 행 단위로 같다. Small 측정값은 다음과 같다.

   | 행 수 | gather 경로 | grouped 커널 |
   |---|---|---|
   | 3 | 81 ms | 75 ms |
   | 8 | 150 ms | 129 ms |

## 결과

Small, greedy 256토큰, ABAB 순서:

| 프롬프트 | 이전 | 이후 | 목표 |
|---|---|---|---|
| 산문 | 0.90x | 1.12x (21.6 대 19.3 tok/s) | 1.0x |
| 코드 | 1.54x | 1.58x (30.4 대 19.2 tok/s) | 1.5x |

- 모든 실행에서 출력 텍스트가 classic decode와 같다.
- 정책은 산문에서는 주로 3행 라운드로, 코드에서는 3~8행으로 수렴한다.
- Mini는 회귀가 없고, 산문에서는 고정 폭보다 이득이 더 크다.

## 나머지 항목

- **서버 prefill chunk.** 서버 burst는 원래 Inkling 2를 forward 한 번으로 prefill했다. 이제 새 `DFlashTargetModel::prefill_burst` 훅을 통해 `--prefill-chunk-size`를 따른다. 다른 target은 기본 동작을 유지한다.
- **Mini 4-bit의 행 단위 projection.** 8행 라운드당 최대 4.1 ms가 든다. 비교용 plain forward가 행 단위 sliding attention도 건너뛰므로 이 값은 상한이다. 이 정도 절감을 위해 양자화 row-invariant 커널을 만들 이유는 없다.

## 위험

- 정책이 실측 시간에 의존하므로 호스트 부하가 크면 더 좁은 폭으로 기운다. 바뀌는 것은 처리량뿐이고 출력 텍스트는 바뀌지 않는다.
- grouped 커널은 Small에서 per-visit 진단으로 폭 2, 4, 8과 1100토큰 프롬프트 이후를 확인했다.
