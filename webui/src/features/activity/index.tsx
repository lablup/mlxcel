// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import React, { lazy, Suspense, useState } from 'react';
import { Button, EmptyState, ErrorBanner, Select } from '../../design-system/primitives';
import { testId, type Locale } from '../../i18n/catalog';
import { useWebUi, useWebUiActions } from '../../state';
import { downloadDiagnostics } from './format';
import { Operations } from './operations';
import { RuntimeView } from './runtime';
import { strings } from './strings';
import './activity.css';

const History = lazy(() => import('./history'));

export function ActivityPage({ locale }: { locale: Locale }): React.JSX.Element {
  const snapshot = useWebUi();
  const actions = useWebUiActions();
  const text = strings(locale);
  const [showHistory, setShowHistory] = useState(false);
  const stale = !['ready', 'streaming', 'polling'].includes(snapshot.connection);
  const latestRuntimeAt = snapshot.runtimeHistory.at(-1)?.receivedAt ?? null;
  const runtimeStale = stale || latestRuntimeAt === null || (snapshot.lastUpdatedAt !== null && snapshot.lastUpdatedAt - latestRuntimeAt > 4_000);
  const runtime = snapshot.selectedModelId === null ? undefined : snapshot.runtimes.get(snapshot.selectedModelId);
  return <div className="screen-stack" data-testid="activity-page">
    <section className="screen-heading"><h1 data-testid={testId('activity.title')}>{text.title}</h1><p>{text.intro}</p><p>{text.updated}: {snapshot.lastSuccessfulAt === null ? text.pending : <time>{new Date(snapshot.lastSuccessfulAt).toLocaleString(locale)}</time>}</p></section>
    <Select locale={locale} label={text.select} value={snapshot.selectedModelId ?? ''} onChange={(id) => actions.selectModel(id || null)} options={[{ value: '', label: text.select }, ...snapshot.catalog.map((entry) => ({ value: entry.identity.id, label: entry.identity.display_name }))]} />
    <div className="activity-toolbar"><Button onClick={() => { void actions.refresh(); }}>{text.refresh}</Button><Button onClick={() => downloadDiagnostics(snapshot)}>{text.export}</Button></div>
    {stale ? <ErrorBanner tone="warning" title={text.stale} body={text.refresh} /> : null}
    <Operations operations={snapshot.operations} locale={locale} stale={stale} />
    {snapshot.selectedModelId === null ? <EmptyState title={text.select} body={text.selectBody} /> : <RuntimeView runtime={runtime} locale={locale} stale={runtimeStale} />}
    <Button onClick={() => setShowHistory(!showHistory)}>{showHistory ? text.hideHistory : text.history}</Button>
    {showHistory ? <Suspense fallback={<p>{text.pending}</p>}><History samples={snapshot.runtimeHistory} locale={locale} /></Suspense> : null}
  </div>;
}
