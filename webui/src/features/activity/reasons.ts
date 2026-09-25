// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import type { MeasuredValue } from '../../api/types';
import type { StringKey } from '../../i18n/catalog';

// Raw server reasons are not user copy. These are the closed sets from
// src/server/webui/runtime.rs that can reach the summary tiles and the slot table;
// reasons.test.ts fails if the server wording drifts away from them. The raw text
// stays available inside the "All measurements and sources" disclosure.
export const METRIC_REASON_KEYS: ReadonlyMap<string, StringKey> = new Map<string, StringKey>([
  ['metrics disabled; restart with --metrics', 'activity.reason.metrics_disabled'],
  ['model provider is not loaded; no live counters are available', 'activity.reason.not_loaded'],
  ['single-stream provider does not publish this batch counter', 'activity.reason.single_stream'],
  ['counter unavailable', 'activity.reason.counter_unavailable'],
]);

export const SLOT_REASON_KEYS: ReadonlyMap<string, StringKey> = new Map<string, StringKey>([
  ['slots disabled; restart with --slots', 'activity.slots_reason.disabled'],
  ['model provider is not loaded; slots are unavailable', 'activity.slots_reason.not_loaded'],
  ['slot registry is unavailable', 'activity.slots_reason.registry'],
  ['showing the first 256 observational slots', 'activity.slots_reason.partial'],
]);

/**
 * Localized hint for a tile without a value; an unrecognized reason points at the disclosure.
 * A metric absent from the snapshot is not listed there, so it says the server did not report it.
 */
export function metricReasonKey(metric: MeasuredValue | undefined): StringKey {
  if (metric === undefined) return 'activity.reason.counter_unavailable';
  return (metric?.reason === null || metric?.reason === undefined ? undefined : METRIC_REASON_KEYS.get(metric.reason)) ?? 'activity.reason.see_details';
}
