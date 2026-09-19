import React, { useEffect, useId, useState } from 'react';
import { Button, Dialog, ErrorBanner, Select } from '../../design-system/primitives';
import type { CatalogEntry, LoadProfile } from '../../api/types';
import { t, type Locale } from '../../i18n/catalog';
import { KV_MODES, profileCli, useLoadProfile, validateLoadProfile } from './load-profiles';
import { HelpTip, NumberField } from './setting-control';

type CopyState = 'idle' | 'copied' | 'failed';

export function ProfileSettings({ model, locale, single }: { model: CatalogEntry | null; locale: Locale; single: boolean }): React.JSX.Element {
  const store = useLoadProfile(model?.identity.id ?? null);
  const [draft, setDraft] = useState<LoadProfile>(store.profile);
  const [scope, setScope] = useState<'model' | 'reusable'>(model === null ? 'reusable' : 'model');
  const [transfer, setTransfer] = useState('');
  const [message, setMessage] = useState('');
  const [resetOpen, setResetOpen] = useState(false);
  const [copy, setCopy] = useState<CopyState>('idle');
  const titleId = useId();
  const scopedProfile = scope === 'reusable' ? store.reusable : store.modelProfile;
  useEffect(() => { setDraft(scopedProfile); }, [scopedProfile]);
  const run = (action: () => void): void => { try { action(); setMessage(t(locale, 'settings.profile.saved')); } catch (error) { setMessage(error instanceof Error ? error.message : t(locale, 'settings.profile.failed')); } };
  const cli = profileCli(scopedProfile);
  const copyCli = async (): Promise<void> => {
    // The Clipboard API exists only in a secure context; a LAN http origin falls back to selecting the text.
    try { if (!navigator.clipboard) throw new Error('Clipboard unavailable'); await navigator.clipboard.writeText(cli); setCopy('copied'); } catch { setCopy('failed'); }
  };
  const integer = (raw: string): number | undefined => raw.trim() === '' ? undefined : Number(raw);
  return (
    <section className="screen-stack" aria-labelledby={titleId}>
      <div className="settings-heading"><h2 id={titleId}>{t(locale, 'settings.profile.title')}</h2><HelpTip label={t(locale, 'settings.help.profile')} content={t(locale, 'settings.profile.body')} /></div>
      {single ? <ErrorBanner tone="info" title={t(locale, 'settings.profile.single.title')} body={t(locale, 'settings.profile.single.body')} /> : null}
      <div className="settings-grid settings-grid-dense">
        <Select locale={locale} label={t(locale, 'settings.profile.scope')} value={scope} onChange={(value) => setScope(value as 'model' | 'reusable')} options={[{ value: 'reusable', label: t(locale, 'settings.profile.scope.reusable') }, ...(model === null ? [] : [{ value: 'model', label: model.identity.display_name }])]} />
        <NumberField label={t(locale, 'settings.profile.ctx_size')} value={draft.ctx_size?.toString() ?? ''} step="1" min={1} max={262144} onChange={(raw) => setDraft({ ...draft, ctx_size: integer(raw) })} hint={t(locale, 'settings.profile.ctx_hint')} />
        <NumberField label={t(locale, 'settings.profile.n_parallel')} value={draft.n_parallel?.toString() ?? ''} step="1" min={1} max={32} onChange={(raw) => setDraft({ ...draft, n_parallel: integer(raw) })} hint={t(locale, 'settings.profile.parallel_hint')} />
        <Select locale={locale} label={t(locale, 'settings.profile.kv_cache_mode')} value={draft.kv_cache_mode ?? ''} onChange={(value) => setDraft({ ...draft, kv_cache_mode: value === '' ? undefined : value as NonNullable<LoadProfile['kv_cache_mode']> })} options={[{ value: '', label: t(locale, 'settings.profile.inherit') }, ...KV_MODES.map((value) => ({ value, label: value }))]} />
      </div>
      {message ? <ErrorBanner tone="info" title={t(locale, 'settings.profile.result_title')} body={message} testId="settings-profile-result" /> : null}
      <div className="control-row settings-actions"><Button tone="primary" onClick={() => run(() => store.save(validateLoadProfile(draft), scope))}>{t(locale, 'settings.profile.save')}</Button><Button onClick={() => { setDraft(scopedProfile); setMessage(''); }}>{t(locale, 'settings.profile.discard')}</Button><Button onClick={() => setResetOpen(true)}>{t(locale, 'settings.profile.reset_open')}</Button></div>
      <div className="settings-disclosures">
      <details className="settings-disclosure" onToggle={() => setCopy('idle')}>
        <summary>{t(locale, 'settings.profile.show_cli')}</summary>
        <p>{t(locale, 'settings.profile.saved_pending')}: <code>{JSON.stringify(scopedProfile)}</code></p>
        <p>{t(locale, 'settings.profile.cli')}</p>
        <code className="settings-cli" data-testid="settings-profile-cli">{cli}</code>
        <div className="control-row"><Button onClick={() => void copyCli()}>{t(locale, 'settings.profile.copy')}</Button><span role="status">{copy === 'copied' ? t(locale, 'settings.profile.copied') : copy === 'failed' ? t(locale, 'settings.profile.copy_failed') : ''}</span></div>
      </details>
      <details className="settings-disclosure">
        <summary>{t(locale, 'settings.profile.transfer')}</summary>
        <label className="ds-field"><span>{t(locale, 'settings.profile.transfer.label')}</span><textarea value={transfer} onChange={(event) => setTransfer(event.currentTarget.value)} maxLength={65536} rows={6} /></label>
        <div className="control-row"><Button onClick={() => setTransfer(store.exportJson())}>{t(locale, 'settings.profile.transfer.export')}</Button><Button onClick={() => run(() => store.importJson(transfer))}>{t(locale, 'settings.profile.transfer.import')}</Button></div>
      </details>
      </div>
      <Dialog open={resetOpen} title={t(locale, 'settings.profile.reset.title')} closeLabel={t(locale, 'common.close')} onClose={() => setResetOpen(false)}><p>{t(locale, 'settings.profile.reset.body')}</p><Button onClick={() => { run(() => store.reset(scope)); setResetOpen(false); }}>{t(locale, 'settings.profile.reset.confirm')}</Button></Dialog>
    </section>
  );
}
