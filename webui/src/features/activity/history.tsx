// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import React from 'react';
import type { WebUiSnapshot } from '../../api/types';
import { t, type Locale } from '../../i18n/catalog';

export default function History({ samples, locale }: { samples: WebUiSnapshot['runtimeHistory']; locale: Locale }): React.JSX.Element {
  // Discrete dots intentionally leave missing measurements and polling gaps blank.
  // Cumulative counters never become rates by a guessed token/time conversion.
  const end = samples.at(-1)?.receivedAt ?? 0;
  const points = samples.flatMap((sample) => {
    const value = sample.runtime.measurements.active_requests?.value;
    return value === null || value === undefined ? [] : [{ time: sample.receivedAt, value }];
  });
  const max = Math.max(1, ...points.map((point) => point.value));
  return <section><h3>{t(locale, 'activity.history.show')}</h3><p>{t(locale, 'activity.history.note')}</p>{points.length === 0 ? <p>{t(locale, 'activity.unavailable')}</p> : <>
    <svg className="activity-chart" viewBox="0 0 600 180" role="img" aria-label={t(locale, 'activity.chart.label')}><title>{t(locale, 'activity.chart.title')}</title>{points.map((point) => <circle key={point.time} cx={5 + (point.time - end + 300_000) / 300_000 * 590} cy={170 - point.value / max * 160} r="3" />)}</svg>
    <details><summary>{t(locale, 'activity.details')}</summary><ol>{points.map((point) => <li key={point.time}><time>{new Date(point.time).toLocaleTimeString(locale)}</time>: {t(locale, 'activity.chart.point', { value: String(point.value) })}</li>)}</ol></details>
  </>}</section>;
}
