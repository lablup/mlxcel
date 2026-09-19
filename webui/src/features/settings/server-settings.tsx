import './settings.css';
import { ContextCheck } from './context-check';
import React, { useEffect, useState } from 'react';
import { t, type Locale } from '../../i18n/catalog';
import { lifecycleLabel } from '../../provider-surfaces';
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
  const connected = snapshot.auth.status === 'authenticated' && ['ready', 'streaming', 'polling'].includes(snapshot.connection);
  const readyId = connected && selected?.lifecycle.state === 'ready' ? selected.identity.id : null;
  useEffect(() => {
    setProps(null);
    if (readyId === null) return;
    const controller = new AbortController();
    void actions.getModelProps(readyId, controller.signal).then((value) => { if (!controller.signal.aborted) setProps(value); }).catch(() => undefined);
    return () => controller.abort();
  }, [actions, readyId, selected?.identity.revision]);
  const unknown = t(locale, 'settings.server.unknown');
  return <div className="screen-stack settings-sections"><section className="screen-stack"><h2>{t(locale, 'settings.server.privacy.title')}</h2><p>{t(locale, 'settings.server.privacy.body')}</p></section><GenerationSettings locale={locale} />{!connected ? <ErrorBanner tone="warning" title={t(locale, 'settings.server.unavailable.title')} body={t(locale, 'settings.server.unavailable.body')} /> : <><Select locale={locale} label={t(locale, 'settings.server.model')} value={snapshot.selectedModelId ?? ''} options={[{ value: '', label: t(locale, 'settings.server.model.none') }, ...snapshot.catalog.map((entry) => ({ value: entry.identity.id, label: `${entry.identity.display_name} · ${lifecycleLabel(locale, entry.lifecycle.state)}` }))]} onChange={(value) => actions.selectModel(value === '' ? null : value)} hint={t(locale, 'settings.server.model.hint')} testId="settings-model-selector" /><ProfileSettings key={selected?.identity.id ?? 'reusable'} model={selected} locale={locale} single={snapshot.bootstrap?.server.mode === 'single_model'} /><section className="screen-stack"><h2>{t(locale, 'settings.server.context.title')}</h2><Field label={t(locale, 'settings.server.context.n_ctx')} value={props?.nCtx?.toString() ?? unknown} disabled hint={t(locale, 'settings.server.context.n_ctx_hint')} /><Field label={t(locale, 'settings.server.context.slots')} value={props?.totalSlots?.toString() ?? unknown} disabled /><Field label={t(locale, 'settings.server.context.kv_mode')} value={props?.kvCacheMode ?? unknown} disabled /><p>{t(locale, 'settings.server.context.geometry')}: <code>{props?.geometry === null || props === null ? t(locale, 'settings.server.context.geometry_unknown') : JSON.stringify(props.geometry)}</code></p>{readyId !== null ? <ContextCheck key={readyId} modelId={readyId} nCtx={props?.nCtx ?? null} locale={locale} /> : null}</section>{readyId === null ? <ErrorBanner tone="info" title={t(locale, 'settings.server.no_model.title')} body={t(locale, 'settings.server.no_model.body')} /> : snapshot.bootstrap?.features.includes('settings') === true ? <LiveSettings key={readyId} modelId={readyId} locale={locale} /> : <ErrorBanner tone="info" title={t(locale, 'settings.server.live_disabled.title')} body={t(locale, 'settings.server.live_disabled.body')} />}</>}</div>;
}
