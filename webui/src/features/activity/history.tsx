// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import React from 'react';
import type { WebUiSnapshot } from '../../api/types';
import type { Locale } from '../../i18n/catalog';
import { strings } from './strings';

export default function History({ samples, locale }: { samples: WebUiSnapshot['runtimeHistory']; locale: Locale }): React.JSX.Element {
  const text = strings(locale);
  // Discrete dots intentionally leave missing measurements and polling gaps blank.
  // Cumulative counters never become rates by a guessed token/time conversion.
  const end = samples.at(-1)?.receivedAt ?? 0;
  const points = samples.flatMap((sample) => {
    const value = sample.runtime.measurements.active_requests?.value;
    return value === null || value === undefined ? [] : [{ time: sample.receivedAt, value }];
  });
  const max = Math.max(1, ...points.map((point) => point.value));
  return <section><h3>{text.history}</h3><p>{text.historyNote}</p>{points.length === 0 ? <p>{text.unavailable}</p> : <>
    <svg className="activity-chart" viewBox="0 0 600 180" role="img" aria-label="Active requests: discrete observations over the last five minutes"><title>Active requests (requests, selected model)</title>{points.map((point) => <circle key={point.time} cx={5 + (point.time - end + 300_000) / 300_000 * 590} cy={170 - point.value / max * 160} r="3" />)}</svg>
    <details><summary>{text.details}</summary><ol>{points.map((point) => <li key={point.time}><time>{new Date(point.time).toLocaleTimeString(locale)}</time>: {point.value} requests</li>)}</ol></details>
  </>}</section>;
}
