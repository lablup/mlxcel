// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import React from 'react';
import type { RuntimeSnapshot } from '../../api/types';
import { EmptyState, ErrorBanner, ProgressBar } from '../../design-system/primitives';
import { t, type Locale } from '../../i18n/catalog';
import { metricValue } from './format';
import { activityStrings, type ActivityMetricKey } from './strings';

const metricKeys = new Set<string>(activityStrings.map((entry) => entry.key).filter((key) => key.startsWith('activity.metric.')));
const isMetricKey = (key: string): key is ActivityMetricKey => metricKeys.has(key);

export function RuntimeView({ runtime, locale, stale = false }: { runtime: RuntimeSnapshot | undefined; locale: Locale; stale?: boolean }): React.JSX.Element {
  if (runtime === undefined) return <EmptyState title={t(locale, 'activity.runtime')} body={t(locale, 'activity.unavailable')} />;
  const entries = Object.entries(runtime.measurements);
  const primary = new Set(['active_requests', 'queued_requests', 'completed_requests_total', 'completion_tokens_total']);
  const availablePrimary = entries.filter(([name, metric]) => primary.has(name) && metric.value !== null && Number.isFinite(metric.value));
  const unavailableCount = entries.filter(([, metric]) => metric.value === null || !Number.isFinite(metric.value)).length;
  const metricLabel = (name: string): string => { const key = `activity.metric.${name}`; return isMetricKey(key) ? t(locale, key) : name.replaceAll('_', ' '); };
  const notAvailable = t(locale, 'activity.not_available');
  return <section aria-label={t(locale, 'activity.runtime')}><h2>{t(locale, 'activity.runtime')}</h2>{stale ? <ErrorBanner tone="warning" title={t(locale, 'activity.stale')} body={t(locale, 'activity.observed')} /> : null}
    <div className="activity-metrics activity-metrics--summary" data-testid="runtime-summary">{availablePrimary.map(([name, metric]) => <dl className="activity-metric" key={name}><dt>{metricLabel(name)}</dt><dd>{metricValue(metric)}</dd></dl>)}</div>
    {availablePrimary.length === 0 ? <p>{t(locale, 'activity.no_primary')}</p> : null}
    {unavailableCount > 0 ? <p>{t(locale, 'activity.unavailable_count')}: {unavailableCount}. {t(locale, 'activity.unavailable_reason')}</p> : null}
    <h3>{t(locale, 'activity.slots')}</h3><p>{t(locale, 'activity.parallel')}: {runtime.slots.effective_parallelism ?? notAvailable} / {runtime.slots.configured_parallelism}</p><p>{t(locale, 'activity.context_line', { context: t(locale, 'activity.context'), request: runtime.slots.request_context_tokens?.toString() ?? notAvailable, pool: t(locale, 'activity.pool'), shared: runtime.slots.shared_pool_context_tokens?.toString() ?? notAvailable })}</p>
    {runtime.slots.reason !== null ? <p>{runtime.slots.reason}</p> : null}
    <p>{t(locale, 'activity.observed')}: {runtime.slots.measured_at === null ? t(locale, 'activity.unknown_time') : new Date(runtime.slots.measured_at).toLocaleString(locale)}</p>
    {!runtime.slots.available ? runtime.slots.reason === null ? <p>{t(locale, 'activity.unavailable')}</p> : null : runtime.slots.items.map((slot) => <section key={slot.id} aria-label={`${t(locale, 'activity.slot')} ${slot.id}`}>
      <h4>{t(locale, 'activity.slot')} {slot.id} · {slot.processing ? t(locale, 'activity.processing') : t(locale, 'activity.idle')}</h4>
      {slot.prompt_tokens !== null && runtime.slots.request_context_tokens !== null && runtime.slots.request_context_tokens > 0 ? <ProgressBar label={`${t(locale, 'activity.slot')} ${slot.id} ${t(locale, 'activity.context')}`} value={slot.prompt_tokens / runtime.slots.request_context_tokens * 100} detail={t(locale, 'activity.slot_progress', { used: String(slot.prompt_tokens), total: String(runtime.slots.request_context_tokens) })} /> : <p>{t(locale, 'activity.no_context')}</p>}
      <p>{t(locale, 'activity.slot_tokens', { decoded: slot.decoded_tokens?.toString() ?? notAvailable, cached: slot.cached_prompt_tokens?.toString() ?? notAvailable })}</p>
    </section>)}
    <details className="activity-measurement-details"><summary>{t(locale, 'activity.metric_details')}</summary><p>{t(locale, 'activity.memory')}</p><p>{t(locale, 'activity.timing')}</p>
      <div className="activity-metrics">{entries.map(([name, metric]) => <dl className="activity-metric" key={name}>
        <dt>{metricLabel(name)}</dt><dd>{metricValue(metric)}</dd><dd>{t(locale, 'activity.scope')}: {metric.scope}</dd><dd>{t(locale, 'activity.observed')}: {metric.measured_at === null ? t(locale, 'activity.unknown_time') : new Date(metric.measured_at).toLocaleString(locale)}</dd>{metric.reason === null ? null : <dd>{metric.reason}</dd>}
      </dl>)}</div>
    </details>
  </section>;
}
