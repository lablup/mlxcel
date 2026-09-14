import React, { useState } from 'react';
import { Button, Dialog, ErrorBanner, Field } from '../../design-system/primitives';
import type { Locale } from '../../i18n/catalog';
import { GENERATION_FIELDS, validateGenerationDefaults } from './generation-defaults';
import { useGenerationDefaults } from './generation-preferences';

export function GenerationSettings({ locale }: { locale: Locale }): React.JSX.Element {
  const { defaults, setDefaults, reset } = useGenerationDefaults();
  const [draft, setDraft] = useState<Record<string, string>>(() => Object.fromEntries(Object.entries(defaults).map(([key, value]) => [key, String(value)])));
  const [error, setError] = useState('');
  const [resetOpen, setResetOpen] = useState(false);
  const words = (en: string, ko: string): string => locale === 'ko' ? ko : en;
  const apply = (): void => {
    try { setDefaults(validateGenerationDefaults(Object.fromEntries(Object.entries(draft).filter(([, value]) => value.trim() !== '').map(([key, value]) => [key, Number(value)])))); setError(''); } catch (cause) { setError(cause instanceof Error ? cause.message : 'Invalid defaults'); }
  };
  return <section className="screen-stack"><h2>{words('Generation · next request', '생성 · 다음 요청')}</h2><p>{words('These defaults stay in browser memory only. Blank fields are omitted and inherit server defaults. Existing turns are frozen; changing defaults never alters an in-flight request. System prompts belong to individual conversations in Chat and are not stored here.', '기본값은 브라우저 메모리에만 남습니다. 빈 필드는 생략되어 서버 기본값을 사용합니다. 진행 중인 요청은 변경되지 않습니다. 시스템 프롬프트는 채팅의 개별 대화에서 설정하며 여기에 저장하지 않습니다.')}</p><div className="settings-grid">{GENERATION_FIELDS.map((name) => <Field key={name} label={name} value={draft[name] ?? ''} onChange={(value) => setDraft((old) => ({ ...old, [name]: value }))} hint={`${words('Current next-request default', '현재 다음 요청 기본값')}: ${defaults[name] ?? words('inherit', '상속')}`} />)}</div>{error ? <ErrorBanner title={words('Invalid request defaults', '잘못된 요청 기본값')} body={error} /> : null}<div><Button onClick={apply}>{words('Save session defaults', '세션 기본값 저장')}</Button> <Button onClick={() => setResetOpen(true)}>{words('Reset request defaults…', '요청 기본값 초기화…')}</Button></div><Field label={words('Reasoning / structured output', '추론 / 구조화 출력')} value={words('Endpoint-specific controls are available only where explicitly supported in Chat.', '채팅에서 명시적으로 지원되는 엔드포인트에만 제공됩니다.')} disabled hint={words('No unsupported parameters are sent.', '지원되지 않는 매개변수는 전송하지 않습니다.')} /><Dialog open={resetOpen} title={words('Reset request defaults?', '요청 기본값을 초기화할까요?')} onClose={() => setResetOpen(false)}><p>{words('Only browser-session request defaults will be cleared. Server and next-load profiles remain unchanged.', '브라우저 세션 요청 기본값만 지워집니다. 서버와 다음 로드 프로필은 변경되지 않습니다.')}</p><Button onClick={() => { reset(); setDraft({}); setError(''); setResetOpen(false); }}>{words('Reset request defaults', '요청 기본값 초기화')}</Button></Dialog></section>;
}
