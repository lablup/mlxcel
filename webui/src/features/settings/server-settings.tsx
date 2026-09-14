import './settings.css';
import { ContextCheck } from './context-check';
import React, { useEffect, useState } from 'react';
import type { Locale } from '../../i18n/catalog';
import { ErrorBanner, Field, Select } from '../../design-system/primitives';
import { useWebUi, useWebUiActions } from '../../state';
import type { ModelProps } from '../../api/settings';
import { GenerationSettings } from './generation-settings';
import { LiveSettings } from './live-settings';
import { ProfileSettings } from './profile-settings';

export function ServerSettings({ locale }: { locale: Locale }): React.JSX.Element {
  const snapshot = useWebUi();
  const actions = useWebUiActions();
  const selected = snapshot.catalog.find((entry) => entry.identity.id === snapshot.selectedModelId) ?? null;
  const [props, setProps] = useState<ModelProps | null>(null);
  const words = (en: string, ko: string): string => locale === 'ko' ? ko : en;
  const connected = snapshot.auth.status === 'authenticated' && ['ready', 'streaming', 'polling'].includes(snapshot.connection);
  const readyId = connected && selected?.lifecycle.state === 'ready' ? selected.identity.id : null;
  useEffect(() => {
    setProps(null);
    if (readyId === null) return;
    const controller = new AbortController();
    void actions.getModelProps(readyId, controller.signal).then((value) => { if (!controller.signal.aborted) setProps(value); }).catch(() => undefined);
    return () => controller.abort();
  }, [actions, readyId, selected?.identity.revision]);
  return <div className="screen-stack settings-sections"><section className="screen-stack"><h2>{words('Privacy · browser only', '개인정보 · 브라우저 전용')}</h2><p>{words('Appearance and explicitly saved load profiles use this browser’s preferences. Request defaults stay in memory; API credentials and system prompts are never stored by Settings. Conversation history is controlled separately in Chat.', '모양과 명시적으로 저장한 로드 프로필은 브라우저 환경설정을 사용합니다. 요청 기본값은 메모리에만 유지하며 API 자격 증명과 시스템 프롬프트는 설정에서 저장하지 않습니다. 대화 기록은 채팅에서 별도로 관리합니다.')}</p></section><GenerationSettings locale={locale} />{!connected ? <ErrorBanner tone="warning" title={words('Server controls unavailable', '서버 제어 사용 불가')} body={words('Connect from Models or Chat, or refresh the connection before changing server settings. Browser appearance and request defaults remain available. Schema mismatch disables server controls.', '모델이나 채팅에서 연결하거나 연결을 새로 고친 후 서버 설정을 변경하세요. 브라우저 모양과 요청 기본값은 계속 사용할 수 있습니다. 스키마 불일치 시 서버 제어가 비활성화됩니다.')} /> : <><Select locale={locale} label={words('Selected model for settings', '설정 대상 모델')} value={snapshot.selectedModelId ?? ''} options={[{ value: '', label: words('No model selected', '선택한 모델 없음') }, ...snapshot.catalog.map((entry) => ({ value: entry.identity.id, label: `${entry.identity.display_name} · ${entry.lifecycle.state}` }))]} onChange={(value) => actions.selectModel(value === '' ? null : value)} hint={words('Selection changes browser context only. It never loads, unloads or reroutes an existing request.', '선택은 브라우저 문맥만 변경하며 모델을 로드하거나 언로드하거나 기존 요청의 대상을 변경하지 않습니다.')} testId="settings-model-selector" /><ProfileSettings key={selected?.identity.id ?? 'reusable'} model={selected} locale={locale} single={snapshot.bootstrap?.server.mode === 'single_model'} /><section className="screen-stack"><h2>{words('Effective context · observed, not estimated', '실제 컨텍스트 · 추정이 아닌 관측값')}</h2><Field label={words('Per-slot context tokens (/props n_ctx)', '슬롯별 컨텍스트 토큰 (/props n_ctx)')} value={props?.nCtx?.toString() ?? words('Unknown', '알 수 없음')} disabled hint={words('Zero/missing is unknown; do not multiply this value into a shared-pool capacity. Unified KV remains separate work (#1815).', '0 또는 누락은 알 수 없음입니다. 이 값에 슬롯 수를 곱해 공유 풀 용량으로 간주하지 않습니다. 통합 KV는 별도 작업입니다 (#1815).')} /><Field label={words('Configured slots (/props total_slots)', '설정된 슬롯 (/props total_slots)')} value={props?.totalSlots?.toString() ?? words('Unknown', '알 수 없음')} disabled /><Field label={words('Active resolved KV cache mode (/props)', '현재 실제 KV 캐시 모드 (/props)')} value={props?.kvCacheMode ?? words('Unknown', '알 수 없음')} disabled /><p>{words('Reported context geometry', '보고된 컨텍스트 구성')}: <code>{props?.geometry === null || props === null ? words('Unknown / props disabled or unavailable', '알 수 없음 / props 비활성화 또는 사용 불가') : JSON.stringify(props.geometry)}</code></p>{readyId !== null ? <ContextCheck key={readyId} modelId={readyId} nCtx={props?.nCtx ?? null} locale={locale} /> : null}</section>{readyId === null ? <ErrorBanner tone="info" title={words('No ready model selected', '준비된 모델이 선택되지 않음')} body={words('Selection never loads a model. Load explicitly from Models to read live settings.', '모델 선택은 로드를 수행하지 않습니다. 모델 화면에서 명시적으로 로드하여 실시간 설정을 확인하세요.')} /> : snapshot.bootstrap?.features.includes('settings') === true ? <LiveSettings key={readyId} modelId={readyId} locale={locale} /> : <ErrorBanner tone="info" title={words('Live settings disabled by operator', '운영자가 실시간 설정을 비활성화함')} body={words('Restart with --settings to enable this endpoint. WebUI never enables settings, props, metrics or slots implicitly.', '엔드포인트를 활성화하려면 --settings로 재시작하세요. WebUI는 settings, props, metrics 또는 slots를 자동으로 활성화하지 않습니다.')} />}</>}</div>;
}
