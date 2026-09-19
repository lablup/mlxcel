// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import React, { lazy, Suspense, useState } from 'react';
import { Button, EmptyState, ErrorBanner, Select } from '../../design-system/primitives';
import { t, testId, type Locale } from '../../i18n/catalog';
import { useWebUi, useWebUiActions } from '../../state';
import { downloadDiagnostics } from './format';
import { Operations } from './operations';
import { RuntimeView } from './runtime';
import './activity.css';

const History = lazy(() => import('./history'));

export function ActivityPage({ locale }: { locale: Locale }): React.JSX.Element {
  const snapshot = useWebUi();
  const actions = useWebUiActions();
  const [showHistory, setShowHistory] = useState(false);
  const stale = !['ready', 'streaming', 'polling'].includes(snapshot.connection);
  const latestRuntimeAt = snapshot.runtimeHistory.at(-1)?.receivedAt ?? null;
  const runtimeStale = stale || latestRuntimeAt === null || (snapshot.lastUpdatedAt !== null && snapshot.lastUpdatedAt - latestRuntimeAt > 4_000);
  const runtime = snapshot.selectedModelId === null ? undefined : snapshot.runtimes.get(snapshot.selectedModelId);
  return <div className="screen-stack" data-testid="activity-page">
    <section className="screen-heading"><h1 data-testid={testId('activity.title')}>{t(locale, 'activity.title')}</h1><p>{t(locale, 'activity.intro')}</p><p>{t(locale, 'activity.updated')}: {snapshot.lastSuccessfulAt === null ? t(locale, 'activity.pending') : <time>{new Date(snapshot.lastSuccessfulAt).toLocaleString(locale)}</time>}</p></section>
    <Select locale={locale} label={t(locale, 'activity.select')} value={snapshot.selectedModelId ?? ''} onChange={(id) => actions.selectModel(id || null)} options={[{ value: '', label: t(locale, 'activity.select') }, ...snapshot.catalog.map((entry) => ({ value: entry.identity.id, label: entry.identity.display_name }))]} />
    <div className="activity-toolbar"><Button onClick={() => { void actions.refresh(); }}>{t(locale, 'activity.refresh')}</Button><Button onClick={() => downloadDiagnostics(snapshot)}>{t(locale, 'activity.export')}</Button></div>
    {stale ? <ErrorBanner tone="warning" title={t(locale, 'activity.stale')} body={t(locale, 'activity.refresh')} /> : null}
    <Operations operations={snapshot.operations} locale={locale} stale={stale} />
    {snapshot.selectedModelId === null ? <EmptyState title={t(locale, 'activity.select')} body={t(locale, 'activity.select_body')} /> : <RuntimeView runtime={runtime} locale={locale} stale={runtimeStale} />}
    <Button onClick={() => setShowHistory(!showHistory)}>{showHistory ? t(locale, 'activity.history.hide') : t(locale, 'activity.history.show')}</Button>
    {showHistory ? <Suspense fallback={<p>{t(locale, 'activity.pending')}</p>}><History samples={snapshot.runtimeHistory} locale={locale} /></Suspense> : null}
  </div>;
}
