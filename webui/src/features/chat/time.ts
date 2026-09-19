// Copyright 2026 Lablup Inc. Licensed under Apache-2.0.
// Locale-aware clock and relative times for the conversation list and messages.
import { useEffect, useState } from 'react';
import { localeTag } from '../../design-system/format';
import type { Locale } from '../../i18n/catalog';

/** The current time, refreshed every `intervalMs` while the caller is mounted. */
export function useNow(intervalMs: number): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), intervalMs);
    return () => window.clearInterval(timer);
  }, [intervalMs]);
  return now;
}

const UNITS: ReadonlyArray<readonly [Intl.RelativeTimeFormatUnit, number]> = [
  ['year', 365 * 86_400_000], ['month', 30 * 86_400_000], ['week', 7 * 86_400_000],
  ['day', 86_400_000], ['hour', 3_600_000], ['minute', 60_000],
];

// Formatters are built once per locale: the list and the streaming message re-render on
// every 50 ms stream flush.
const relativeFormats = new Map<Locale, Intl.RelativeTimeFormat>();
const clockFormats = new Map<Locale, Intl.DateTimeFormat>();
function cached<T>(cache: Map<Locale, T>, locale: Locale, create: (tag: string) => T): T {
  const existing = cache.get(locale);
  if (existing !== undefined) return existing;
  const created = create(localeTag(locale));
  cache.set(locale, created);
  return created;
}

/** "5 minutes ago" / "5분 전"; anything under a minute reads as now. */
export function formatRelative(then: number, now: number, locale: Locale): string {
  const format = cached(relativeFormats, locale, (tag) => new Intl.RelativeTimeFormat(tag, { numeric: 'auto' }));
  const delta = then - now;
  for (const [unit, size] of UNITS) if (Math.abs(delta) >= size) return format.format(Math.round(delta / size), unit);
  return format.format(0, 'second');
}

/** Hour and minute in the active locale. */
export function formatClock(epochMs: number, locale: Locale): string {
  return cached(clockFormats, locale, (tag) => new Intl.DateTimeFormat(tag, { hour: 'numeric', minute: '2-digit' })).format(epochMs);
}

/** The machine-readable value for a `<time>` element. */
export function isoTime(epochMs: number): string {
  return new Date(epochMs).toISOString();
}
