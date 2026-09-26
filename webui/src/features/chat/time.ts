// Copyright 2026 Lablup Inc. Licensed under Apache-2.0.
// Locale-aware clock and relative times for the conversation list and messages.
import { useEffect, useState } from 'react';
import { cached, localeTag } from '../../design-system/format';
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

/** "5 minutes ago" / "5분 전"; anything under a minute reads as now. */
export function formatRelative(then: number, now: number, locale: Locale): string {
  const format = cached(relativeFormats, locale, () => new Intl.RelativeTimeFormat(localeTag(locale), { numeric: 'auto' }));
  const delta = then - now;
  for (const [unit, size] of UNITS) if (Math.abs(delta) >= size) return format.format(Math.round(delta / size), unit);
  return format.format(0, 'second');
}

// Imported history bounds updatedAt and elapsedMs only as safe or finite numbers, which
// reach past the largest time a Date holds (8.64e15 ms). Formatting such a time throws a
// RangeError, and a throw during render takes the whole WebUI down, so it reads as unknown.
function representable(epochMs: number): boolean {
  return !Number.isNaN(new Date(epochMs).getTime());
}

/** Hour and minute in the active locale; empty for a time a Date cannot hold. */
export function formatClock(epochMs: number, locale: Locale): string {
  if (!representable(epochMs)) return '';
  return cached(clockFormats, locale, () => new Intl.DateTimeFormat(localeTag(locale), { hour: 'numeric', minute: '2-digit' })).format(epochMs);
}

/** The machine-readable value for a `<time>` element; undefined for a time a Date cannot hold. */
export function isoTime(epochMs: number): string | undefined {
  return representable(epochMs) ? new Date(epochMs).toISOString() : undefined;
}
