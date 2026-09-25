// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import React, { useMemo } from 'react';
import type { CatalogEntry } from '../../api/types';
import { Button, EmptyState, PageHeader, Select } from '../../design-system/primitives';
import { t, testId, type Locale } from '../../i18n/catalog';
import { useWebUi, useWebUiActions } from '../../state';
import { activePoints } from './history';
import { clockTime, downloadDiagnostics } from './format';
import { Operations } from './operations';
import { RuntimeView } from './runtime';
import './activity.css';

const LIVE = new Set(['ready', 'streaming', 'polling']);

// Loaded models first so the one being served is the first choice, otherwise catalog
// order (Array.prototype.sort is stable). Labels stay the verbatim display name:
// scripts/activity-performance.mjs selects the option by that exact name.
export function modelOptions(catalog: ReadonlyArray<CatalogEntry>, locale: Locale): { value: string; label: string }[] {
  const ready = (entry: CatalogEntry): number => Number(entry.lifecycle.state === 'ready');
  return [{ value: '', label: t(locale, 'activity.select') }, ...[...catalog].sort((a, b) => ready(b) - ready(a)).map((entry) => ({ value: entry.identity.id, label: entry.identity.display_name }))];
}

export function ActivityPage({ locale }: { locale: Locale }): React.JSX.Element {
  const snapshot = useWebUi();
  const actions = useWebUiActions();
  // The catalog changes rarely; runtime polls every 2 s must not re-sort it or rebuild the name index.
  const options = useMemo(() => modelOptions(snapshot.catalog, locale), [snapshot.catalog, locale]);
  const names = useMemo(() => new Map(snapshot.catalog.map((entry) => [entry.identity.id, entry.identity.display_name])), [snapshot.catalog]);
  const points = useMemo(() => activePoints(snapshot.runtimeHistory), [snapshot.runtimeHistory]);
  const stale = !LIVE.has(snapshot.connection);
  const latestRuntimeAt = snapshot.runtimeHistory.at(-1)?.receivedAt ?? null;
  const runtimeStale = stale || latestRuntimeAt === null || (snapshot.lastUpdatedAt !== null && snapshot.lastUpdatedAt - latestRuntimeAt > 4_000);
  const runtime = snapshot.selectedModelId === null ? undefined : snapshot.runtimes.get(snapshot.selectedModelId);
  const lastSnapshot = snapshot.lastSuccessfulAt === null ? t(locale, 'activity.pending') : t(locale, 'activity.updated', { time: clockTime(snapshot.lastSuccessfulAt, locale) });
  return <div className="screen-stack activity-page" data-testid="activity-page">
    <PageHeader
      className="activity-header"
      title={t(locale, 'activity.title')}
      titleTestId={testId('activity.title')}
      description={`${t(locale, 'activity.intro')} ${lastSnapshot}`}
      descriptionTestId={testId('activity.intro')}
      actions={<>
        <div className="activity-model-picker"><Select locale={locale} label={t(locale, 'activity.model')} value={snapshot.selectedModelId ?? ''} onChange={(id) => actions.selectModel(id || null)} options={options} testId={testId('activity.model')} /></div>
        <Button onClick={() => { void actions.refresh(); }} data-testid={testId('activity.refresh')}>{t(locale, 'activity.refresh')}</Button>
        <Button onClick={() => downloadDiagnostics(snapshot)} data-testid={testId('activity.export')}>{t(locale, 'activity.export')}</Button>
      </>}
      error={stale ? t(locale, 'activity.stale') : null}
      errorTestId={testId('activity.stale')}
      onRetry={() => { void actions.refresh(); }}
      retryLabel={t(locale, 'activity.refresh')}
    />
    {snapshot.selectedModelId === null ? <EmptyState title={t(locale, 'activity.select')} body={t(locale, 'activity.select_body')} /> : <RuntimeView runtime={runtime} points={points} historyEnd={latestRuntimeAt} locale={locale} stale={stale} runtimeStale={runtimeStale} />}
    <Operations operations={snapshot.operations} names={names} locale={locale} stale={stale} now={snapshot.lastUpdatedAt ?? Date.now()} />
  </div>;
}
