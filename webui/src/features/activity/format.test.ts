import { describe, expect, it } from 'vitest';
import { diagnostics, metricValue, operationProgress } from './format';
import { initialSnapshot } from '../../state/reducer';
import runtimeFixture from '../../../../tests/fixtures/webui/examples/runtime.snapshot.json';
import operationsFixture from '../../../../tests/fixtures/webui/examples/operations.list.json';
import { validateOperationsList, validateRuntime } from '../../api/validation';

describe('honest Activity formatting and diagnostics', () => {
  it('preserves real zero but never turns an unknown into zero', () => {
    const metric = { value: null, unit: 'bytes', scope: 'server', measured_at: null, reason: 'unavailable' } as const;
    expect(metricValue(metric)).toBe('N/A');
    expect(metricValue({ ...metric, value: 0 })).toBe('0 bytes');
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
