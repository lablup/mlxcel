// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import React from 'react';
import type { RuntimeSnapshot } from '../../api/types';
import { EmptyState, ErrorBanner, ProgressBar } from '../../design-system/primitives';
import type { Locale } from '../../i18n/catalog';
import { metricValue } from './format';
import { strings } from './strings';

export function RuntimeView({ runtime, locale, stale = false }: { runtime: RuntimeSnapshot | undefined; locale: Locale; stale?: boolean }): React.JSX.Element {
  const text = strings(locale);
  if (runtime === undefined) return <EmptyState title={text.runtime} body={text.unavailable} />;
  return <section aria-label={text.runtime}><h2>{text.runtime}</h2>{stale ? <ErrorBanner tone="warning" title={text.stale} body={text.observed} /> : null}<p>{text.memory}</p><p>{text.timing}</p>
    <div className="activity-metrics">{Object.entries(runtime.measurements).map(([name, metric]) => <dl className="activity-metric" key={name}>
      <dt>{name.replaceAll('_', ' ')}</dt><dd>{metricValue(metric)}</dd><dd>{text.scope}: {metric.scope}</dd><dd>{text.observed}: {metric.measured_at === null ? text.unknownTime : new Date(metric.measured_at).toLocaleString(locale)}</dd>{metric.reason === null ? null : <dd>{metric.reason}</dd>}
    </dl>)}</div>
    <h3>{text.slots}</h3><p>{text.parallel}: {runtime.slots.effective_parallelism ?? 'N/A'} / {runtime.slots.configured_parallelism}</p><p>{text.context}: {runtime.slots.request_context_tokens ?? 'N/A'} tokens · {text.pool}: {runtime.slots.shared_pool_context_tokens ?? 'N/A'} tokens</p>
    {runtime.slots.reason !== null ? <p>{runtime.slots.reason}</p> : null}
    <p>{text.observed}: {runtime.slots.measured_at === null ? text.unknownTime : new Date(runtime.slots.measured_at).toLocaleString(locale)}</p>
    {!runtime.slots.available ? runtime.slots.reason === null ? <p>{text.unavailable}</p> : null : runtime.slots.items.map((slot) => <section key={slot.id} aria-label={`${text.slot} ${slot.id}`}>
      <h4>{text.slot} {slot.id} · {slot.processing ? text.processing : text.idle}</h4>
      {slot.prompt_tokens !== null && runtime.slots.request_context_tokens !== null && runtime.slots.request_context_tokens > 0 ? <ProgressBar label={`${text.slot} ${slot.id} ${text.context}`} value={slot.prompt_tokens / runtime.slots.request_context_tokens * 100} detail={`${slot.prompt_tokens} / ${runtime.slots.request_context_tokens} tokens`} /> : <p>{text.noContext}</p>}
      <p>Accepted decode: {slot.decoded_tokens ?? 'N/A'} tokens · Cached prompt: {slot.cached_prompt_tokens ?? 'N/A'} tokens</p>
    </section>)}
  </section>;
}
