// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import React from 'react';
import type { WebUiSnapshot } from '../../api/types';
import { t, type Locale } from '../../i18n/catalog';
import { clockTime, formatCount, pluralKey } from './format';

type Samples = WebUiSnapshot['runtimeHistory'];
export type HistoryPoint = { readonly time: number; readonly value: number };

const WINDOW_MS = 300_000;
const WIDTH = 120;
const HEIGHT = 32;
const INSET = 3;

// Discrete observations only: missing measurements and polling gaps stay blank, and
// cumulative counters never become rates by a guessed token/time conversion.
export function activePoints(samples: Samples): HistoryPoint[] {
  return samples.flatMap((sample) => {
    const value = sample.runtime.measurements.active_requests?.value;
    return value === null || value === undefined || !Number.isFinite(value) ? [] : [{ time: sample.receivedAt, value }];
  });
}

/**
 * Active requests over the five-minute window as a dot plot in one <path>: each point
 * is a zero-length subpath that the round line cap in activity.css draws as a dot, so a
 * full 150-sample ring costs one DOM node per poll instead of one <circle> per sample.
 * The right edge is `end`, the latest runtime sample whether or not it carried a
 * measurement, so a trailing run of missing measurements stays blank instead of sliding
 * older dots to the edge where they would read as current.
 */
export function Sparkline({ points, end, locale }: { points: readonly HistoryPoint[]; end: number; locale: Locale }): React.JSX.Element {
  const max = Math.max(1, ...points.map((point) => point.value));
  const span = WIDTH - 2 * INSET;
  const d = points.map((point) => {
    const x = INSET + Math.max(0, (point.time - end + WINDOW_MS) / WINDOW_MS) * span;
    const y = HEIGHT - INSET - point.value / max * (HEIGHT - 2 * INSET);
    return `M${x.toFixed(1)} ${y.toFixed(1)}h0`;
  }).join('');
  return <svg className="activity-sparkline" viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img" aria-label={t(locale, 'activity.chart.label')}><path d={d} /></svg>;
}

/** The text alternative: every sample with its local time, rendered only inside the open disclosure. */
export function HistoryList({ points, locale }: { points: readonly HistoryPoint[]; locale: Locale }): React.JSX.Element {
  return <>
    <h3>{t(locale, 'activity.chart.title')}</h3>
    <p>{t(locale, 'activity.history.note')}</p>
    {points.length === 0 ? <p>{t(locale, 'activity.history.empty')}</p> : <ol className="activity-history">{points.map((point) => <li key={point.time}><time dateTime={new Date(point.time).toISOString()}>{clockTime(point.time, locale)}</time> {t(locale, pluralKey(point.value, ['activity.chart.point_one', 'activity.chart.point'], locale), { value: formatCount(point.value, locale) })}</li>)}</ol>}
  </>;
}
