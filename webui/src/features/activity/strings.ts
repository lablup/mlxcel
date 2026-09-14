// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import type { Locale } from '../../i18n/catalog';
const en = {
  title: 'Activity', intro: 'Operations and runtime observations. Browsing never loads a model.',
  session: 'History belongs to this server session: up to 200 terminal operations for one hour. A restart clears it; reconnect reconciles the authoritative snapshot.',
  export: 'Export sanitized diagnostics', refresh: 'Refresh observations', stale: 'Observations are stale. Last values are not current measurements.',
  updated: 'Last successful snapshot', pending: 'Waiting for the first snapshot', operations: 'Operations', empty: 'No operations in this session',
  emptyBody: 'Explicit downloads, loads, drains and removals will appear here.', cancel: 'Request cancellation', cancelling: 'Cancellation requested; waiting for worker acknowledgement.',
  failure: 'The action failed. Check its state and refresh before retrying from Models.', cancelFailed: 'Cancellation could not be confirmed. Refresh to reconcile the operation.',
  cancelUnsupported: 'Cancellation is unavailable for this operation; worker execution is not interrupted.',
  runtime: 'Selected model runtime', select: 'Select a model to observe', selectBody: 'Use the model picker. Selection does not load the model.',
  unavailable: 'N/A — no authoritative sample is available.', memory: 'Memory figures have separate scopes and may overlap. Do not add process, allocator, device and cache values together.',
  timing: 'Server counters are not browser TTFT. Chat shows request-send to first reasoning/content delta separately; no rendered word counts are used as tokens.',
  slots: 'Request slots', slot: 'Slot', processing: 'Processing', idle: 'Idle', context: 'Request context window', pool: 'Shared pool context', parallel: 'Effective parallelism',
  noContext: 'N/A — context denominator is unknown; no occupancy percentage is inferred.',
  history: 'Show recent history', hideHistory: 'Hide recent history', historyNote: 'Five-minute in-memory history, at most 150 two-second samples. Hidden tabs stop observation; gaps and counter resets are not interpolated.',
  measure: 'Measurement', value: 'Value', scope: 'Scope', observed: 'Observed at', reason: 'Availability / source',
  bytes: 'bytes downloaded', details: 'Operation details', status: 'Operation status', unknownTime: 'Unknown',
};
const ko: typeof en = {
  title: '활동', intro: '작업과 런타임 관측입니다. 조회는 모델을 로드하지 않습니다.',
  session: '기록은 서버 세션에 속합니다. 완료 작업 최대 200개를 한 시간 보관하며, 재시작 시 초기화하고 재연결 시 서버 상태와 대조합니다.',
  export: '민감 정보 없는 진단 내보내기', refresh: '관측 새로고침', stale: '관측이 최신 상태가 아닙니다. 마지막 값은 현재 측정값이 아닙니다.',
  updated: '마지막 성공 스냅샷', pending: '첫 스냅샷을 기다리는 중', operations: '작업', empty: '이 세션의 작업 없음', emptyBody: '명시적인 다운로드, 로드, 드레인 및 삭제 작업이 표시됩니다.',
  cancel: '취소 요청', cancelling: '취소 요청됨. 작업자 확인을 기다립니다.', failure: '작업에 실패했습니다. 상태를 확인하고 새로고침한 후 모델 화면에서 재시도하세요.', cancelFailed: '취소를 확인하지 못했습니다. 새로고침하여 작업 상태를 대조하세요.', cancelUnsupported: '이 작업은 취소할 수 없습니다. 실행 중인 작업자를 강제로 중단하지 않습니다.',
  runtime: '선택한 모델 런타임', select: '관측할 모델 선택', selectBody: '모델 선택기를 사용하세요. 선택만으로 모델을 로드하지 않습니다.', unavailable: 'N/A — 확인된 관측값이 없습니다.',
  memory: '메모리 수치는 서로 다른 범위이며 중복될 수 있습니다. 프로세스, 할당자, 장치 및 캐시 값을 합산하지 마세요.',
  timing: '서버 카운터는 브라우저 TTFT가 아닙니다. 채팅은 요청 전송부터 첫 추론/내용 델타까지 별도로 표시하며, 단어 수를 토큰으로 사용하지 않습니다.',
  slots: '요청 슬롯', slot: '슬롯', processing: '처리 중', idle: '유휴', context: '요청 컨텍스트 한도', pool: '공유 풀 컨텍스트', parallel: '실제 병렬도', noContext: 'N/A — 컨텍스트 분모를 알 수 없어 점유율을 추정하지 않습니다.',
  history: '최근 기록 보기', hideHistory: '최근 기록 숨기기', historyNote: '메모리 내 5분 기록이며 2초 간격 최대 150개입니다. 숨겨진 탭은 관측을 중지합니다. 공백과 카운터 초기화를 보간하지 않습니다.',
  measure: '측정 항목', value: '값', scope: '범위', observed: '관측 시각', reason: '가용성 / 출처', bytes: '다운로드 바이트', details: '작업 상세', status: '작업 상태', unknownTime: '알 수 없음',
};
export function strings(locale: Locale): typeof en { return locale === 'ko' ? ko : en; }
