// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import React, { useId } from 'react';
import type { MeasuredValue, RuntimeSnapshot } from '../../api/types';
import { Badge, EmptyState, ErrorBanner, StatCard } from '../../design-system/primitives';
import { t, type Locale } from '../../i18n/catalog';
import { Disclosure } from './disclosure';
import { clockTime, metricValue } from './format';
import { HistoryList, Sparkline, type HistoryPoint } from './history';
import { metricReasonKey } from './reasons';
import { SlotTable } from './slots';
import { activityStrings, type ActivityMetricKey } from './strings';

const metricKeys = new Set<string>(activityStrings.map((entry) => entry.key).filter((key) => key.startsWith('activity.metric.')));
const isMetricKey = (key: string): key is ActivityMetricKey => metricKeys.has(key);
const metricLabel = (name: string, locale: Locale): string => { const key = `activity.metric.${name}`; return isMetricKey(key) ? t(locale, key) : name.replaceAll('_', ' '); };
const measured = (metric: MeasuredValue | undefined): metric is MeasuredValue => metric !== undefined && metric.value !== null && Number.isFinite(metric.value);

// Always four tiles in this order, whether or not the server reported each counter.
const PRIMARY = ['active_requests', 'completed_requests_total', 'completion_tokens_total', 'queued_requests'] as const;
const SCOPE_KEYS = { model: 'activity.scope.model', slot: 'activity.scope.slot', pool: 'activity.scope.pool', server: 'activity.scope.server', unknown: 'activity.scope.unknown' } as const;

type RuntimeViewProps = { runtime: RuntimeSnapshot | undefined; points: readonly HistoryPoint[]; locale: Locale; stale: boolean; runtimeStale: boolean };

export function RuntimeView({ runtime, points, locale, stale, runtimeStale }: RuntimeViewProps): React.JSX.Element {
  const headingId = useId();
  if (runtime === undefined && stale) return <EmptyState title={t(locale, 'activity.runtime')} body={t(locale, 'activity.unavailable')} />;
  const heading = <h2 id={headingId}>{t(locale, 'activity.runtime')}</h2>;
  if (runtime === undefined) {
    // Live connection, no sample yet: one localized status for the waiting tiles, whose skeletons are decorative.
    return <section className="activity-runtime" aria-labelledby={headingId}>{heading}
      <p className="activity-note" role="status">{t(locale, 'activity.waiting')}</p>
      <div className="activity-metrics activity-metrics--summary" data-testid="runtime-summary" aria-busy="true">{PRIMARY.map((name) => <StatCard className="activity-metric" key={name} label={metricLabel(name, locale)} value="" hint=" " loading />)}</div>
    </section>;
  }
  const entries = Object.entries(runtime.measurements);
  const unavailable = entries.filter(([, metric]) => !measured(metric)).length;
  const tiles = PRIMARY.map((name) => {
    const metric = runtime.measurements[name];
    const hint = measured(metric) ? t(locale, 'activity.observed_at', { time: clockTime(metric.measured_at, locale) }) : t(locale, metricReasonKey(metric));
    const sparkline = name === 'active_requests' && points.length >= 2 ? <Sparkline points={points} locale={locale} /> : undefined;
    return <StatCard className="activity-metric" key={name} label={metricLabel(name, locale)} value={metric === undefined ? t(locale, 'format.unknown') : metricValue(metric, locale)} hint={hint} sparkline={sparkline} />;
  });
  return <section className="activity-runtime" aria-labelledby={headingId}>{heading}
    {runtimeStale && !stale ? <ErrorBanner tone="warning" title={t(locale, 'activity.stale')} body={t(locale, 'activity.runtime_stale')} /> : null}
    <div className="activity-metrics activity-metrics--summary" data-testid="runtime-summary">{tiles}</div>
    <Disclosure className="activity-disclosure activity-measurement-details" summary={<>{t(locale, 'activity.metric_details')}{unavailable > 0 ? <>{' '}<Badge>{t(locale, 'activity.unavailable_badge', { count: String(unavailable) })}</Badge></> : null}</>}>
      {() => <>
        <p>{t(locale, 'activity.memory')}</p>
        <p>{t(locale, 'activity.timing')}</p>
        <ul className="activity-measurements">{entries.map(([name, metric]) => <li key={name}>
          <span className="activity-measurement__name">{metricLabel(name, locale)}</span>
          <span className="activity-measurement__value">{metricValue(metric, locale)}</span>
          <span className="activity-muted">{t(locale, 'activity.scope', { scope: t(locale, SCOPE_KEYS[metric.scope]) })}</span>
          <span className="activity-muted">{metric.measured_at === null ? t(locale, 'activity.unknown_time') : t(locale, 'activity.observed_at', { time: clockTime(metric.measured_at, locale) })}</span>
          {/* The server's own provenance or availability text, kept verbatim for diagnosis. */}
          {metric.reason === null ? null : <span className="activity-raw">{metric.reason}</span>}
        </li>)}</ul>
        <HistoryList points={points} locale={locale} />
      </>}
    </Disclosure>
    <SlotTable slots={runtime.slots} locale={locale} />
  </section>;
}
