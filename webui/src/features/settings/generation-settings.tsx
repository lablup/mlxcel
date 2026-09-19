import React, { useEffect, useId, useState } from 'react';
import { Button, Dialog, ErrorBanner } from '../../design-system/primitives';
import { t, type Locale } from '../../i18n/catalog';
import { describeEffectiveParameter, GENERATION_FIELDS, resolveEffectiveParameter, validateGenerationDefaults, type GenerationField } from './generation-defaults';
import { useGenerationDefaults } from './generation-preferences';
import { recordServerSettings, useServerGenerationDefaults } from './server-defaults';
import { HelpTip, NumberField, settingLabel } from './setting-control';
import { useSettingsTarget, type SettingsTarget } from './settings-target';

type ServerRead = 'idle' | 'loading' | 'ready' | 'failed';
const INTEGER_FIELDS: ReadonlySet<GenerationField> = new Set(['max_tokens', 'top_k', 'seed']);

/** Reads the selected ready model's /settings once per model revision, so each field can name its server default. */
function useServerDefaultsRead(target: SettingsTarget): { state: ServerRead; model: string | null; reason: 'offline' | 'none' | 'disabled' | null } {
  const { actions, readyId, revision, liveEnabled } = target;
  const [state, setState] = useState<ServerRead>('idle');
  useEffect(() => {
    if (readyId === null || revision === null || !liveEnabled) { setState('idle'); return undefined; }
    const controller = new AbortController();
    setState('loading');
    void actions.getSettings(readyId, controller.signal).then((response) => {
      if (controller.signal.aborted) return;
      recordServerSettings(readyId, revision, response);
      setState('ready');
    }).catch(() => { if (!controller.signal.aborted) setState('failed'); });
    return () => controller.abort();
  }, [actions, readyId, revision, liveEnabled]);
  const reason = !target.connected ? 'offline' : readyId === null ? 'none' : !liveEnabled ? 'disabled' : null;
  return { state, model: target.selected?.identity.display_name ?? null, reason };
}

export function GenerationSettings({ locale }: { locale: Locale }): React.JSX.Element {
  const { defaults, setDefaults, reset } = useGenerationDefaults();
  const target = useSettingsTarget();
  const server = useServerGenerationDefaults(target.readyId, target.revision);
  const read = useServerDefaultsRead(target);
  const [draft, setDraft] = useState<Record<string, string>>(() => Object.fromEntries(Object.entries(defaults).map(([key, value]) => [key, String(value)])));
  const [error, setError] = useState('');
  const [resetOpen, setResetOpen] = useState(false);
  const titleId = useId();
  const apply = (): void => {
    try { setDefaults(validateGenerationDefaults(Object.fromEntries(Object.entries(draft).filter(([, value]) => value.trim() !== '').map(([key, value]) => [key, Number(value)])))); setError(''); } catch (cause) { setError(cause instanceof Error ? cause.message : t(locale, 'settings.generation.invalid')); }
  };
  const model = read.model ?? '';
  const status = read.reason === 'offline' ? t(locale, 'settings.requests.server_offline') : read.reason === 'none' ? t(locale, 'settings.requests.server_none') : read.reason === 'disabled' ? t(locale, 'settings.requests.server_disabled') : read.state === 'failed' ? t(locale, 'settings.requests.server_failed', { model }) : read.state === 'ready' ? t(locale, 'settings.requests.server_from', { model }) : t(locale, 'settings.requests.server_loading', { model });
  return (
    <section className="screen-stack" aria-labelledby={titleId}>
      <div className="settings-heading"><h2 id={titleId}>{t(locale, 'settings.generation.title')}</h2><HelpTip label={t(locale, 'settings.help.requests')} content={t(locale, 'settings.generation.body')} /></div>
      <p className="settings-note" role="status" data-testid="settings-requests-server">{status}</p>
      <div className="settings-grid settings-grid-dense">
        {GENERATION_FIELDS.map((name) => {
          const resolved = resolveEffectiveParameter(name, undefined, defaults[name], server?.[name]);
          const { label } = settingLabel(locale, `default_${name}`);
          return <NumberField key={name} label={label} value={draft[name] ?? ''} step={INTEGER_FIELDS.has(name) ? '1' : 'any'} min={0} onChange={(value) => setDraft((old) => ({ ...old, [name]: value }))} hint={`${name} · ${t(locale, 'settings.requests.effective', { value: describeEffectiveParameter(locale, resolved) })}`} testId={`settings-request-${name}`} />;
        })}
      </div>
      {error ? <ErrorBanner title={t(locale, 'settings.generation.error_title')} body={error} /> : null}
      <div className="control-row settings-actions"><Button tone="primary" onClick={apply}>{t(locale, 'settings.generation.save')}</Button><Button onClick={() => setResetOpen(true)}>{t(locale, 'settings.generation.reset_open')}</Button></div>
      <Dialog open={resetOpen} title={t(locale, 'settings.generation.reset.title')} closeLabel={t(locale, 'common.close')} onClose={() => setResetOpen(false)}><p>{t(locale, 'settings.generation.reset.body')}</p><Button onClick={() => { reset(); setDraft({}); setError(''); setResetOpen(false); }}>{t(locale, 'settings.generation.reset.confirm')}</Button></Dialog>
    </section>
  );
}
