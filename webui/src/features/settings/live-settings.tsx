import React, { useEffect, useRef, useState } from 'react';
import { Button, Dialog, ErrorBanner, Field } from '../../design-system/primitives';
import { parseSettingInput, settingInput, type SettingsResponse } from '../../api/settings';
import { useWebUiActions } from '../../state';
import type { Locale } from '../../i18n/catalog';

export function LiveSettings({ modelId, locale }: { modelId: string; locale: Locale }): React.JSX.Element {
  const actions = useWebUiActions();
  const [current, setCurrent] = useState<SettingsResponse | null>(null);
  const [draft, setDraft] = useState<Record<string, string>>({});
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [message, setMessage] = useState('');
  const [busy, setBusy] = useState(false);
  const [resetOpen, setResetOpen] = useState(false);
  const request = useRef<AbortController | null>(null);
  const words = (en: string, ko: string): string => locale === 'ko' ? ko : en;
  useEffect(() => { const controller = new AbortController(); request.current = controller; setBusy(true); void actions.getSettings(modelId, controller.signal).then((value) => { if (!controller.signal.aborted) setCurrent(value); }).catch(() => { if (!controller.signal.aborted) setMessage('Settings unavailable. The server may have disabled --settings.'); }).finally(() => { if (!controller.signal.aborted) setBusy(false); }); return () => controller.abort(); }, [actions, modelId]);
  const refresh = async (): Promise<void> => {
    const controller = request.current;
    if (controller === null || controller.signal.aborted) return;
    setBusy(true);
    try { const response = await actions.getSettings(modelId, controller.signal); if (!controller.signal.aborted) { setCurrent(response); setMessage(words('Current values refreshed. Review the retained draft before applying.', '현재 값을 새로 고쳤습니다. 남아 있는 초안을 검토한 후 적용하세요.')); } } catch { if (!controller.signal.aborted) setMessage(words('Refresh failed. No changes submitted.', '새로 고침에 실패했습니다. 변경 사항은 전송되지 않았습니다.')); } finally { if (!controller.signal.aborted) setBusy(false); }
  };
  const apply = async (): Promise<void> => {
    const controller = request.current;
    if (controller === null || controller.signal.aborted) return;
    if (current === null) return;
    const values: Record<string, unknown> = {};
    const invalid: Record<string, string> = {};
    for (const spec of current.schema) if (Object.hasOwn(draft, spec.name)) {
      try { values[spec.name] = parseSettingInput(spec, draft[spec.name]); } catch (error) { invalid[spec.name] = error instanceof Error ? error.message : 'Invalid value'; }
    }
    setErrors(invalid);
    if (Object.keys(invalid).length > 0) return;
    setBusy(true);
    try {
      const fresh = await actions.getSettings(modelId, controller.signal);
      if (controller.signal.aborted) return;
      if (fresh.fingerprint !== current.fingerprint) { setCurrent(fresh); setMessage(words('Another client changed these settings. Current values refreshed; review and Apply again to reconfirm.', '다른 클라이언트가 설정을 변경했습니다. 갱신된 현재 값을 검토하고 다시 적용하여 확인하세요.')); return; }
      const result = await actions.patchSettings(modelId, values, controller.signal);
      if (controller.signal.aborted) return;
      setErrors(Object.fromEntries(result.rejected.map((entry) => [entry.name, entry.reason])));
      setDraft((old) => Object.fromEntries(Object.entries(old).filter(([name]) => !Object.hasOwn(result.applied, name))));
      setMessage(result.rejected.length > 0 ? words(`${Object.keys(result.applied).length} applied; ${result.rejected.length} rejected. Rejected drafts retained.`, `${Object.keys(result.applied).length}개 적용, ${result.rejected.length}개 거부됨. 거부된 초안은 유지됩니다.`) : words('Accepted fields applied; reading effective values.', '승인된 필드를 적용했습니다. 실제 값을 다시 읽습니다.'));
      const effective = await actions.getSettings(modelId, controller.signal);
      if (!controller.signal.aborted) setCurrent(effective);
    } catch { if (!controller.signal.aborted) setMessage(words('The outcome may be unknown. Refresh effective values before retrying; no automatic PATCH retry.', '결과를 확정할 수 없습니다. 재시도 전에 실제 값을 새로 고치세요. PATCH는 자동 재시도하지 않습니다.')); } finally { if (!controller.signal.aborted) setBusy(false); }
  };
  return <section className="screen-stack"><h2>{words('Loaded model · live server values', '로드된 모델 · 실시간 서버 값')}</h2><p>{words('Changes affect newly admitted requests. Fingerprints detect already-observed external changes, not atomic compare-and-swap. Numeric bounds remain enforced by the server; this schema publishes types and allowed enums, not numeric min/max.', '변경은 새로 수락되는 요청에 적용됩니다. 지문은 관측된 외부 변경을 감지하지만 원자적 비교 교환은 아닙니다. 숫자 범위는 서버가 검증합니다. 스키마에는 타입과 허용 열거값만 있습니다.')}</p>{message ? <ErrorBanner tone="warning" title={words('Settings result', '설정 결과')} body={message} /> : null}<div><Button onClick={() => void refresh()} disabled={busy}>{words('Refresh current values', '현재 값 새로 고침')}</Button> <Button onClick={() => void apply()} disabled={busy || Object.keys(draft).length === 0}>{words('Apply live draft', '실시간 초안 적용')}</Button> <Button onClick={() => setResetOpen(true)} disabled={busy || current === null}>{words('Reset live draft…', '실시간 초안 초기화…')}</Button></div>{current?.schema.filter((spec) => spec.mutable).map((spec) => <Field key={spec.name} label={spec.name} value={draft[spec.name] ?? settingInput(spec, current.current[spec.name])} onChange={(value) => setDraft((old) => ({ ...old, [spec.name]: value }))} disabled={busy} error={errors[spec.name]} hint={`${spec.help} Current: ${settingInput(spec, current.current[spec.name])}. Type: ${spec.type}${spec.allowed === null ? '' : `; allowed: ${spec.allowed.join(', ')}`}`} />)}<details><summary>{words('Server startup · restart required (read-only)', '서버 시작 · 재시작 필요 (읽기 전용)')}</summary>{current?.schema.filter((spec) => !spec.mutable).map((spec) => <Field key={spec.name} label={spec.name} value={settingInput(spec, current.current[spec.name])} disabled hint={spec.reason ?? spec.help} />)}</details><Dialog open={resetOpen} title={words('Reset live draft only?', '실시간 초안만 초기화할까요?')} onClose={() => setResetOpen(false)}><p>{words('Stage server startup defaults for mutable fields. No server change occurs until Apply.', '변경 가능한 필드에 서버 시작 기본값을 초안으로 설정합니다. 적용 전에는 서버가 변경되지 않습니다.')}</p><Button onClick={() => { if (current !== null) setDraft(Object.fromEntries(current.schema.filter((spec) => spec.mutable).map((spec) => [spec.name, settingInput(spec, spec.default)]))); setErrors({}); setResetOpen(false); }}>{words('Reset draft', '초안 초기화')}</Button></Dialog></section>;
}
