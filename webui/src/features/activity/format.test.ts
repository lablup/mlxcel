import { describe, expect, it } from 'vitest';
import { diagnostics, metricValue, operationProgress, relativeTime } from './format';
import { initialSnapshot } from '../../state/reducer';
import runtimeFixture from '../../../../tests/fixtures/webui/examples/runtime.snapshot.json';
import operationsFixture from '../../../../tests/fixtures/webui/examples/operations.list.json';
import { validateOperationsList, validateRuntime } from '../../api/validation';

describe('honest Activity formatting and diagnostics', () => {
  it('preserves real zero but never turns an unknown into zero', () => {
    const metric = { value: null, unit: 'bytes', scope: 'server', measured_at: null, reason: 'unavailable' } as const;
    expect(metricValue(metric, 'en')).toBe('unknown');
    expect(metricValue(metric, 'ko')).toBe('알 수 없음');
    expect(metricValue({ ...metric, value: Number.NaN }, 'en')).toBe('unknown');
    expect(metricValue({ ...metric, value: 0 }, 'en')).toBe('0 B');
    expect(metricValue({ ...metric, value: 0, unit: 'requests' }, 'en')).toBe('0 requests');
    expect(metricValue({ ...metric, value: 0, unit: 'requests' }, 'ko')).toBe('0건');
  });
  it('formats the number for the locale and names every known server unit from the catalog', () => {
    const metric = { value: 1234567.891, unit: 'tokens', scope: 'model', measured_at: null, reason: null } as const;
    expect(metricValue(metric, 'en')).toBe('1,234,567.89 tokens');
    expect(metricValue(metric, 'ko')).toBe('1,234,567.89 토큰');
    expect(metricValue({ ...metric, value: 1, unit: 'requests' }, 'en')).toBe('1 request');
    expect(metricValue({ ...metric, value: 1, unit: 'tokens' }, 'en')).toBe('1 token');
    expect(metricValue({ ...metric, value: 1, unit: 'requests' }, 'ko')).toBe('1건');
    expect(metricValue({ ...metric, value: 3, unit: 'entries' }, 'ko')).toBe('항목 3개');
    expect(metricValue({ ...metric, value: 12.5, unit: 'tokens/s' }, 'en')).toBe('12.5 tokens/s');
    expect(metricValue({ ...metric, value: 124.5, unit: 'ms' }, 'en')).toBe('124.5 ms');
    expect(metricValue({ ...metric, value: 42, unit: 'percent' }, 'ko')).toBe('42%');
    expect(metricValue({ ...metric, value: 1536, unit: 'bytes' }, 'en')).toBe('1.5 KiB');
    // Only an unknown unit is shown raw.
    expect(metricValue({ ...metric, value: 2, unit: 'widgets' }, 'en')).toBe('2 widgets');
  });
  it('writes operation age relative to the render time in both locales', () => {
    const now = Date.parse('2026-09-15T00:10:00Z');
    expect(relativeTime('2026-09-15T00:09:30Z', now, 'en')).toBe('30 seconds ago');
    expect(relativeTime('2026-09-15T00:05:00Z', now, 'en')).toBe('5 minutes ago');
    expect(relativeTime('2026-09-14T21:10:00Z', now, 'en')).toBe('3 hours ago');
    expect(relativeTime('2026-09-13T00:10:00Z', now, 'en')).toBe('2 days ago');
    expect(relativeTime('2026-09-15T00:10:00Z', now, 'en')).toBe('now');
    expect(relativeTime('2026-09-15T00:09:30Z', now, 'ko')).toBe('30초 전');
    expect(relativeTime('2026-09-15T00:05:00Z', now, 'ko')).toBe('5분 전');
    expect(relativeTime('2026-09-14T00:10:00Z', now, 'ko')).toBe('어제');
    expect(relativeTime('not a time', now, 'en')).toBe('unknown');
  });
  it('only computes progress with a positive known denominator', () => {
    const op = validateOperationsList(JSON.parse(JSON.stringify(operationsFixture), (key, value: unknown) => key === '$schemaName' ? undefined : value)).items[0];
    expect(operationProgress({ ...op, progress: { completed_bytes: 10, total_bytes: 0, indeterminate: false } })).toBeUndefined();
    expect(operationProgress({ ...op, progress: { completed_bytes: 10, total_bytes: 100, indeterminate: false } })).toBe(10);
  });
  it('exports an allowlist excluding every free-text diagnostic, identifier and payload', () => {
    const secret = 'Bearer key /Users/private/checkpoint prompt model-output';
    const runtime = validateRuntime(runtimeFixture);
    const op = validateOperationsList(JSON.parse(JSON.stringify(operationsFixture), (key, value: unknown) => key === '$schemaName' ? undefined : value)).items[0];
    const result = diagnostics({ ...initialSnapshot(), selectedModelId: secret, serverInstanceId: secret, error: { code: secret, message: secret, retryable: true }, runtimes: new Map([[secret, { ...runtime, model_id: secret, measurements: { [secret]: { value: 1, unit: secret, scope: 'unknown', measured_at: secret, reason: secret } } }]]), operations: new Map([[secret, { ...op, operation_id: secret, error: { code: 'unavailable', message: secret, retryable: true } }]]) });
    expect(result).not.toContain(secret);
    expect(result).not.toContain('Bearer');
    expect(JSON.parse(result).measurements).toEqual([{ value: 1, available: true }]);
  });
});
