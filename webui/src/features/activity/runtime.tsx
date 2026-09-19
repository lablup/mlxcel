// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import React from 'react';
import type { RuntimeSnapshot } from '../../api/types';
import { EmptyState, ErrorBanner, ProgressBar, StatCard } from '../../design-system/primitives';
import type { Locale } from '../../i18n/catalog';
import { metricValue } from './format';
import { strings } from './strings';

export function RuntimeView({ runtime, locale, stale = false }: { runtime: RuntimeSnapshot | undefined; locale: Locale; stale?: boolean }): React.JSX.Element {
  const text = strings(locale);
  if (runtime === undefined) return <EmptyState title={text.runtime} body={text.unavailable} />;
  const entries = Object.entries(runtime.measurements);
  const primary = new Set(['active_requests', 'queued_requests', 'completed_requests_total', 'completion_tokens_total']);
  const availablePrimary = entries.filter(([name, metric]) => primary.has(name) && metric.value !== null && Number.isFinite(metric.value));
  const unavailableCount = entries.filter(([, metric]) => metric.value === null || !Number.isFinite(metric.value)).length;
  const metricLabel = (name: string): string => text.metricLabels[name as keyof typeof text.metricLabels] ?? name.replaceAll('_', ' ');
  return <section aria-label={text.runtime}><h2>{text.runtime}</h2>{stale ? <ErrorBanner tone="warning" title={text.stale} body={text.observed} /> : null}
    <div className="activity-metrics activity-metrics--summary" data-testid="runtime-summary">{availablePrimary.map(([name, metric]) => <StatCard className="activity-metric" key={name} label={metricLabel(name)} value={metricValue(metric)} />)}</div>
    {availablePrimary.length === 0 ? <p>{text.noPrimary}</p> : null}
    {unavailableCount > 0 ? <p>{text.unavailableCount}: {unavailableCount}. {text.unavailableReason}</p> : null}
    <h3>{text.slots}</h3><p>{text.parallel}: {runtime.slots.effective_parallelism ?? 'N/A'} / {runtime.slots.configured_parallelism}</p><p>{text.context}: {runtime.slots.request_context_tokens ?? 'N/A'} tokens · {text.pool}: {runtime.slots.shared_pool_context_tokens ?? 'N/A'} tokens</p>
    {runtime.slots.reason !== null ? <p>{runtime.slots.reason}</p> : null}
    <p>{text.observed}: {runtime.slots.measured_at === null ? text.unknownTime : new Date(runtime.slots.measured_at).toLocaleString(locale)}</p>
    {!runtime.slots.available ? runtime.slots.reason === null ? <p>{text.unavailable}</p> : null : runtime.slots.items.map((slot) => <section key={slot.id} aria-label={`${text.slot} ${slot.id}`}>
      <h4>{text.slot} {slot.id} · {slot.processing ? text.processing : text.idle}</h4>
      {slot.prompt_tokens !== null && runtime.slots.request_context_tokens !== null && runtime.slots.request_context_tokens > 0 ? <ProgressBar label={`${text.slot} ${slot.id} ${text.context}`} value={slot.prompt_tokens / runtime.slots.request_context_tokens * 100} detail={`${slot.prompt_tokens} / ${runtime.slots.request_context_tokens} tokens`} /> : <p>{text.noContext}</p>}
      <p>Accepted decode: {slot.decoded_tokens ?? 'N/A'} tokens · Cached prompt: {slot.cached_prompt_tokens ?? 'N/A'} tokens</p>
    </section>)}
    <details className="activity-measurement-details"><summary>{text.metricDetails}</summary><p>{text.memory}</p><p>{text.timing}</p>
      <div className="activity-metrics">{entries.map(([name, metric]) => <StatCard className="activity-metric" key={name} label={metricLabel(name)} value={metricValue(metric)} hint={<>
        <p>{text.scope}: {metric.scope}</p><p>{text.observed}: {metric.measured_at === null ? text.unknownTime : new Date(metric.measured_at).toLocaleString(locale)}</p>{metric.reason === null ? null : <p>{metric.reason}</p>}
      </>} />)}</div>
    </details>
  </section>;
}
