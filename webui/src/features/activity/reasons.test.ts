// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { METRIC_REASON_KEYS, SLOT_REASON_KEYS, metricReasonKey } from './reasons';

// The mapped reasons are server literals. Reading the server source keeps a wording
// change there from silently sending every tile to the generic fallback.
const RUNTIME_RS = join(import.meta.dirname, '../../../../src/server/webui/runtime.rs');

describe('server reason mapping', () => {
  it('maps only reasons the server still emits', () => {
    const source = readFileSync(RUNTIME_RS, 'utf8');
    for (const reason of [...METRIC_REASON_KEYS.keys(), ...SLOT_REASON_KEYS.keys()]) expect(source, reason).toContain(`"${reason}"`);
  });
  it('falls back to the disclosure pointer for a missing or unrecognized reason', () => {
    const metric = { value: null, unit: 'requests', scope: 'model', measured_at: null } as const;
    expect(metricReasonKey({ ...metric, reason: 'metrics disabled; restart with --metrics' })).toBe('activity.reason.metrics_disabled');
    expect(metricReasonKey({ ...metric, reason: 'something new' })).toBe('activity.reason.see_details');
    expect(metricReasonKey({ ...metric, reason: null })).toBe('activity.reason.see_details');
    // Absent from the snapshot, so the disclosure does not list it either.
    expect(metricReasonKey(undefined)).toBe('activity.reason.counter_unavailable');
  });
});
