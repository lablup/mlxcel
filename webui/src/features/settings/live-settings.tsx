import React, { useEffect, useRef, useState } from 'react';
import { Button, Dialog, ErrorBanner, Field } from '../../design-system/primitives';
import { parseSettingInput, settingInput, type SettingSpec, type SettingsResponse } from '../../api/settings';
import { useWebUiActions } from '../../state';
import { t, type Locale } from '../../i18n/catalog';

export function LiveSettings({ modelId, locale }: { modelId: string; locale: Locale }): React.JSX.Element {
  const actions = useWebUiActions();
  const [current, setCurrent] = useState<SettingsResponse | null>(null);
  const [draft, setDraft] = useState<Record<string, string>>({});
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [message, setMessage] = useState('');
  const [busy, setBusy] = useState(false);
  const [resetOpen, setResetOpen] = useState(false);
  const request = useRef<AbortController | null>(null);
  const localeRef = useRef(locale); localeRef.current = locale;
  useEffect(() => { const controller = new AbortController(); request.current = controller; setBusy(true); void actions.getSettings(modelId, controller.signal).then((value) => { if (!controller.signal.aborted) setCurrent(value); }).catch(() => { if (!controller.signal.aborted) setMessage(t(localeRef.current, 'settings.live.unavailable')); }).finally(() => { if (!controller.signal.aborted) setBusy(false); }); return () => controller.abort(); }, [actions, modelId]);
  const refresh = async (): Promise<void> => {
    const controller = request.current;
    if (controller === null || controller.signal.aborted) return;
    setBusy(true);
    try { const response = await actions.getSettings(modelId, controller.signal); if (!controller.signal.aborted) { setCurrent(response); setMessage(t(localeRef.current, 'settings.live.refreshed')); } } catch { if (!controller.signal.aborted) setMessage(t(localeRef.current, 'settings.live.refresh_failed')); } finally { if (!controller.signal.aborted) setBusy(false); }
  };
  const apply = async (): Promise<void> => {
    const controller = request.current;
    if (controller === null || controller.signal.aborted) return;
    if (current === null) return;
    const values: Record<string, unknown> = {};
    const invalid: Record<string, string> = {};
    for (const spec of current.schema) if (Object.hasOwn(draft, spec.name)) {
      try { values[spec.name] = parseSettingInput(spec, draft[spec.name]); } catch (error) { invalid[spec.name] = error instanceof Error ? error.message : t(locale, 'settings.live.invalid_value'); }
    }
    setErrors(invalid);
    if (Object.keys(invalid).length > 0) return;
    setBusy(true);
    try {
      const fresh = await actions.getSettings(modelId, controller.signal);
      if (controller.signal.aborted) return;
      if (fresh.fingerprint !== current.fingerprint) { setCurrent(fresh); setMessage(t(localeRef.current, 'settings.live.conflict')); return; }
      const result = await actions.patchSettings(modelId, values, controller.signal);
      if (controller.signal.aborted) return;
      setErrors(Object.fromEntries(result.rejected.map((entry) => [entry.name, entry.reason])));
      setDraft((old) => Object.fromEntries(Object.entries(old).filter(([name]) => !Object.hasOwn(result.applied, name))));
      setMessage(result.rejected.length > 0 ? t(localeRef.current, 'settings.live.partial', { applied: String(Object.keys(result.applied).length), rejected: String(result.rejected.length) }) : t(localeRef.current, 'settings.live.applied'));
      const effective = await actions.getSettings(modelId, controller.signal);
      if (!controller.signal.aborted) setCurrent(effective);
    } catch { if (!controller.signal.aborted) setMessage(t(localeRef.current, 'settings.live.unknown_outcome')); } finally { if (!controller.signal.aborted) setBusy(false); }
  };
  const hint = (spec: SettingSpec, value: unknown): string => {
    const values = { help: spec.help, current: settingInput(spec, value), type: spec.type };
    return spec.allowed === null ? t(locale, 'settings.live.hint', values) : t(locale, 'settings.live.hint_allowed', { ...values, allowed: spec.allowed.join(', ') });
  };
  return <section className="screen-stack"><h2>{t(locale, 'settings.live.title')}</h2><p>{t(locale, 'settings.live.body')}</p>{message ? <ErrorBanner tone="warning" title={t(locale, 'settings.live.result_title')} body={message} /> : null}<div><Button onClick={() => void refresh()} disabled={busy}>{t(locale, 'settings.live.refresh')}</Button> <Button onClick={() => void apply()} disabled={busy || Object.keys(draft).length === 0}>{t(locale, 'settings.live.apply')}</Button> <Button onClick={() => setResetOpen(true)} disabled={busy || current === null}>{t(locale, 'settings.live.reset_open')}</Button></div>{current?.schema.filter((spec) => spec.mutable).map((spec) => <Field key={spec.name} label={spec.name} value={draft[spec.name] ?? settingInput(spec, current.current[spec.name])} onChange={(value) => setDraft((old) => ({ ...old, [spec.name]: value }))} disabled={busy} error={errors[spec.name]} hint={hint(spec, current.current[spec.name])} />)}<details><summary>{t(locale, 'settings.live.startup')}</summary>{current?.schema.filter((spec) => !spec.mutable).map((spec) => <Field key={spec.name} label={spec.name} value={settingInput(spec, current.current[spec.name])} disabled hint={spec.reason ?? spec.help} />)}</details><Dialog open={resetOpen} title={t(locale, 'settings.live.reset.title')} closeLabel={t(locale, 'common.close')} onClose={() => setResetOpen(false)}><p>{t(locale, 'settings.live.reset.body')}</p><Button onClick={() => { if (current !== null) setDraft(Object.fromEntries(current.schema.filter((spec) => spec.mutable).map((spec) => [spec.name, settingInput(spec, spec.default)]))); setErrors({}); setResetOpen(false); }}>{t(locale, 'settings.live.reset.confirm')}</Button></Dialog></section>;
}
