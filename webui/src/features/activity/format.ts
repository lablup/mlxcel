// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import type { MeasuredValue, Operation, WebUiSnapshot } from '../../api/types';
import { formatBytes, localeTag } from '../../design-system/format';
import { t, type Locale, type StringKey } from '../../i18n/catalog';

// Server units with a catalog template, as [plural "one", every other count]. `bytes`
// goes through formatBytes; any other unit is shown raw after the localized number,
// the only place a raw unit remains.
type UnitKeys = readonly [one: StringKey, other: StringKey];
const UNIT_KEYS: ReadonlyMap<string, UnitKeys> = new Map<string, UnitKeys>([
  ['requests', ['activity.unit.request_one', 'activity.unit.requests']],
  ['tokens', ['activity.unit.token_one', 'activity.unit.tokens']],
  ['ms', ['activity.unit.ms', 'activity.unit.ms']],
  ['us', ['activity.unit.us', 'activity.unit.us']],
  ['percent', ['activity.unit.percent', 'activity.unit.percent']],
  ['entries', ['activity.unit.entry_one', 'activity.unit.entries']],
  ['tokens/s', ['activity.unit.tokens_per_second', 'activity.unit.tokens_per_second']],
]);

// Formatters are cached per locale: every 2 s runtime poll formats every tile and row.
const numberFormats = new Map<Locale, Intl.NumberFormat>();
const pluralRules = new Map<Locale, Intl.PluralRules>();
const relativeFormats = new Map<Locale, Intl.RelativeTimeFormat>();
const timeFormats = new Map<Locale, Intl.DateTimeFormat>();
const dateTimeFormats = new Map<Locale, Intl.DateTimeFormat>();
function cached<T>(cache: Map<Locale, T>, locale: Locale, create: (tag: string) => T): T {
  let format = cache.get(locale);
  if (format === undefined) { format = create(localeTag(locale)); cache.set(locale, format); }
  return format;
}

export function formatCount(value: number, locale: Locale): string {
  if (!Number.isFinite(value)) return t(locale, 'format.unknown');
  return cached(numberFormats, locale, (tag) => new Intl.NumberFormat(tag, { maximumFractionDigits: 2 })).format(value);
}

/** Picks the catalog key for a count: English "1 request", "2 requests"; Korean has one form. */
export function pluralKey(value: number, [one, other]: UnitKeys, locale: Locale): StringKey {
  return cached(pluralRules, locale, (tag) => new Intl.PluralRules(tag)).select(value) === 'one' ? one : other;
}

export function metricValue(metric: MeasuredValue, locale: Locale): string {
  if (metric.value === null || !Number.isFinite(metric.value)) return t(locale, 'format.unknown');
  if (metric.unit === 'bytes') return formatBytes(metric.value, locale);
  const value = formatCount(metric.value, locale);
  const keys = UNIT_KEYS.get(metric.unit);
  return keys === undefined ? `${value} ${metric.unit}` : t(locale, pluralKey(metric.value, keys, locale), { value });
}

export function tokenCount(value: number | null, locale: Locale): string {
  return value === null ? t(locale, 'format.unknown') : t(locale, pluralKey(value, ['activity.unit.token_one', 'activity.unit.tokens'], locale), { value: formatCount(value, locale) });
}

function toDate(at: string | number | null): Date | null {
  if (at === null) return null;
  const date = new Date(at);
  return Number.isFinite(date.getTime()) ? date : null;
}

/** Wall-clock time of an observation, or the localized "no observation time". */
export function clockTime(at: string | number | null, locale: Locale): string {
  const date = toDate(at);
  return date === null ? t(locale, 'activity.unknown_time') : cached(timeFormats, locale, (tag) => new Intl.DateTimeFormat(tag, { timeStyle: 'medium' })).format(date);
}

export function dateTime(at: string | number | null, locale: Locale): string {
  const date = toDate(at);
  return date === null ? t(locale, 'activity.unknown_time') : cached(dateTimeFormats, locale, (tag) => new Intl.DateTimeFormat(tag, { dateStyle: 'medium', timeStyle: 'medium' })).format(date);
}

const RELATIVE_STEPS: ReadonlyArray<[Intl.RelativeTimeFormatUnit, number]> = [['second', 60], ['minute', 60], ['hour', 24], ['day', Number.POSITIVE_INFINITY]];

/** "3 minutes ago" for `iso` seen from `now`. `now` comes from the render, never a timer. */
export function relativeTime(iso: string, now: number, locale: Locale): string {
  const at = Date.parse(iso);
  if (!Number.isFinite(at) || !Number.isFinite(now)) return t(locale, 'format.unknown');
  const format = cached(relativeFormats, locale, (tag) => new Intl.RelativeTimeFormat(tag, { numeric: 'auto' }));
  let value = (at - now) / 1000;
  for (const [unit, size] of RELATIVE_STEPS) {
    const rounded = Math.round(value);
    if (Math.abs(rounded) < size) return format.format(rounded, unit);
    value /= size;
  }
  return t(locale, 'format.unknown');
}

export function operationProgress(operation: Operation): number | undefined {
  const progress = operation.progress;
  return !progress.indeterminate && progress.total_bytes !== null && progress.total_bytes > 0
    ? Math.min(100, 100 * progress.completed_bytes / progress.total_bytes) : undefined;
}

// Export an allowlist, never a redacted copy of the server snapshot: even debug
// slots, operation error messages, target repo names and paths are excluded.
export function diagnostics(snapshot: WebUiSnapshot): string {
  const runtime = snapshot.selectedModelId === null ? undefined : snapshot.runtimes.get(snapshot.selectedModelId);
  return JSON.stringify({
    format: 'mlxcel-webui-diagnostics-v1',
    connection: snapshot.connection,
    last_successful_at: snapshot.lastSuccessfulAt,
    operation_counts: Object.fromEntries(['queued', 'running', 'cancelling', 'succeeded', 'failed', 'cancelled'].map((state) => [state, [...snapshot.operations.values()].filter((op) => op.state === state).length])),
    catalog_count: snapshot.catalog.length,
    runtime_present: runtime !== undefined,
    history_samples: snapshot.runtimeHistory.length,
    // Export only numeric value and availability, not server-provided strings.
    measurements: runtime === undefined ? [] : Object.values(runtime.measurements).map((metric) => ({ value: metric.value, available: metric.value !== null })),
  }, null, 2);
}

export function downloadDiagnostics(snapshot: WebUiSnapshot): void {
  const url = URL.createObjectURL(new Blob([diagnostics(snapshot)], { type: 'application/json' }));
  const link = document.createElement('a');
  link.href = url;
  link.download = 'mlxcel-diagnostics.json';
  link.click();
  URL.revokeObjectURL(url);
}
