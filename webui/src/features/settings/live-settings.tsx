import React, { useEffect, useId, useRef, useState } from 'react';
import { Button, Dialog, ErrorBanner } from '../../design-system/primitives';
import { formatSettingValue, parseSettingInput, settingInput, type SettingSpec, type SettingsResponse } from '../../api/settings';
import { useWebUiActions } from '../../state';
import { t, type Locale, type StringKey } from '../../i18n/catalog';
import { recordServerSettings } from './server-defaults';
import { HelpTip, SettingControl } from './setting-control';

export type LiveSettingGroup = 'sampling' | 'dry' | 'diffusion' | 'template' | 'other';
const GROUP_ORDER: readonly LiveSettingGroup[] = ['sampling', 'dry', 'diffusion', 'template', 'other'];
const GROUP_LABEL: Readonly<Record<LiveSettingGroup, StringKey>> = { sampling: 'settings.live.group.sampling', dry: 'settings.live.group.dry', diffusion: 'settings.live.group.diffusion', template: 'settings.live.group.template', other: 'settings.live.group.other' };

/** Group by name prefix, never by a fixed list: the schema may hold up to 256 entries and grow. */
export function liveSettingGroup(name: string): LiveSettingGroup {
  if (name.startsWith('default_dry_')) return 'dry';
  if (name.startsWith('default_')) return 'sampling';
  if (name.startsWith('diffusion_')) return 'diffusion';
  if (name === 'chat_template_kwargs' || name === 'lang_bias_config' || name.includes('template') || name.includes('bias')) return 'template';
  return 'other';
}
export function groupLiveSettings(specs: readonly SettingSpec[]): { group: LiveSettingGroup; specs: SettingSpec[] }[] {
  return GROUP_ORDER.map((group) => ({ group, specs: specs.filter((spec) => liveSettingGroup(spec.name) === group) })).filter((entry) => entry.specs.length > 0);
}

/** Reset stages a startup default as display text when that text still reads back as the same value, else as exact JSON. */
export function stagedDefault(spec: SettingSpec): string {
  if (spec.default === null) return 'null';
  const shown = formatSettingValue(spec, spec.default);
  if (typeof spec.default !== 'number') return shown;
  return Math.fround(Number(shown)) === Math.fround(spec.default) ? shown : settingInput(spec, spec.default);
}

export function LiveSettings({ modelId, revision, locale }: { modelId: string; revision?: number; locale: Locale }): React.JSX.Element {
  const actions = useWebUiActions();
  const [current, setCurrentState] = useState<SettingsResponse | null>(null);
  const [draft, setDraft] = useState<Record<string, string>>({});
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [message, setMessage] = useState('');
  const [busy, setBusy] = useState(false);
  const [resetOpen, setResetOpen] = useState(false);
  const request = useRef<AbortController | null>(null);
  const localeRef = useRef(locale); localeRef.current = locale;
  const titleId = useId();
  // Every read also refreshes the server defaults the Requests tab and the Chat hint resolve against.
  const setCurrent = (value: SettingsResponse): void => { setCurrentState(value); if (revision !== undefined) recordServerSettings(modelId, revision, value); };
  const setCurrentRef = useRef(setCurrent); setCurrentRef.current = setCurrent;
  useEffect(() => { const controller = new AbortController(); request.current = controller; setBusy(true); void actions.getSettings(modelId, controller.signal).then((value) => { if (!controller.signal.aborted) setCurrentRef.current(value); }).catch(() => { if (!controller.signal.aborted) setMessage(t(localeRef.current, 'settings.live.unavailable')); }).finally(() => { if (!controller.signal.aborted) setBusy(false); }); return () => controller.abort(); }, [actions, modelId]);
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
  const edit = (name: string, value: string | undefined): void => setDraft((old) => value !== undefined ? { ...old, [name]: value } : Object.fromEntries(Object.entries(old).filter(([key]) => key !== name)));
  const mutable = current?.schema.filter((spec) => spec.mutable) ?? [];
  return (
    <section className="screen-stack settings-live" aria-labelledby={titleId}>
      <div className="settings-heading"><h2 id={titleId}>{t(locale, 'settings.live.title')}</h2><HelpTip label={t(locale, 'settings.help.live')} content={t(locale, 'settings.live.body')} /><div className="control-row settings-actions"><Button onClick={() => void refresh()} disabled={busy}>{t(locale, 'settings.live.refresh')}</Button><Button tone="primary" onClick={() => void apply()} disabled={busy || Object.keys(draft).length === 0}>{t(locale, 'settings.live.apply')}</Button><Button onClick={() => setResetOpen(true)} disabled={busy || current === null}>{t(locale, 'settings.live.reset_open')}</Button></div></div>
      {message ? <ErrorBanner tone="warning" title={t(locale, 'settings.live.result_title')} body={message} testId="settings-live-result" /> : null}
      <div className="settings-live-groups">
        {groupLiveSettings(mutable).map(({ group, specs }) => (
          // Groups of one or two share a row where there is room for both.
          <fieldset key={group} className="settings-group" data-group={group} data-size={specs.length <= 2 ? 'small' : 'large'}>
            <legend>{t(locale, GROUP_LABEL[group])}</legend>
            <div className="settings-grid settings-grid-dense">
              {specs.map((spec) => <SettingControl key={spec.name} spec={spec} value={current?.current[spec.name] ?? null} draft={Object.hasOwn(draft, spec.name) ? draft[spec.name] : undefined} onChange={(value) => edit(spec.name, value)} error={errors[spec.name]} disabled={busy} locale={locale} />)}
            </div>
          </fieldset>
        ))}
      </div>
      <Dialog open={resetOpen} title={t(locale, 'settings.live.reset.title')} closeLabel={t(locale, 'common.close')} onClose={() => setResetOpen(false)}><p>{t(locale, 'settings.live.reset.body')}</p><Button onClick={() => { if (current !== null) setDraft(Object.fromEntries(current.schema.filter((spec) => spec.mutable).map((spec) => [spec.name, stagedDefault(spec)]))); setErrors({}); setResetOpen(false); }}>{t(locale, 'settings.live.reset.confirm')}</Button></Dialog>
    </section>
  );
}
