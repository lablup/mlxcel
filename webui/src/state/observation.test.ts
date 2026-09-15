import { describe, expect, it } from 'vitest';
import fixture from '../../../tests/fixtures/webui/examples/runtime.snapshot.json';
import operations from '../../../tests/fixtures/webui/examples/operations.list.json';
import { validateRuntime, validateOperationsList } from '../api/validation';
import { appendRuntimeHistory, boundOperations } from './observation';
import { initialSnapshot, reduceWebUiSnapshot } from './reducer';
import type { Operation, WebUiSnapshot } from '../api/types';

const runtime = validateRuntime(fixture);
describe('bounded observation state', () => {
  it('retains only 150 two-second samples and evicts old or other-session samples', () => {
    let history: WebUiSnapshot['runtimeHistory'] = [];
    for (let time = 0; time < 800_000; time += 1_000) history = appendRuntimeHistory(history, runtime, time);
    expect(history).toHaveLength(150);
    expect(history[0]?.receivedAt).toBeGreaterThan(499_000);
    expect(appendRuntimeHistory(history, { ...runtime, server_instance_id: 'restart' }, 800_000)).toHaveLength(1);
    expect(appendRuntimeHistory(history, { ...runtime, model_id: 'other' }, 800_000)).toHaveLength(1);
  });
  it('refreshes counters at the same lifecycle sequence without accepting older sequence', () => {
    const state = reduceWebUiSnapshot({ ...initialSnapshot(), selectedModelId: runtime.model_id }, { type: 'runtime', runtime, sequence: 10, now: 0 });
    const refreshed = reduceWebUiSnapshot(state, { type: 'runtime', runtime, sequence: 10, now: 2_000 });
    expect(refreshed.runtimeHistory).toHaveLength(2);
    expect(reduceWebUiSnapshot(refreshed, { type: 'runtime', runtime, sequence: 9, now: 4_000 })).toBe(refreshed);
    expect(reduceWebUiSnapshot(refreshed, { type: 'logout', now: 5_000 }).runtimeHistory).toHaveLength(0);
  });
  it('bounds terminal history independently from active operations and TTL', () => {
    const source = validateOperationsList(JSON.parse(JSON.stringify(operations), (key, value: unknown) => key === '$schemaName' ? undefined : value)).items[0];
    const entries = Array.from({ length: 300 }, (_, i): Operation => ({ ...source, operation_id: `op_${i}`, state: i < 70 ? 'running' : 'succeeded', updated_at: new Date(1_000_000 + i).toISOString() }));
    const bounded = boundOperations(new Map(entries.map((op) => [op.operation_id, op])), 1_100_000);
    expect(bounded.size).toBe(270);
    expect(boundOperations(bounded, 10_000_000).size).toBe(70);
  });
  it('expires terminal operations on same-sequence snapshots while preserving active operations', () => {
    const source = validateOperationsList(JSON.parse(JSON.stringify(operations), (key, value: unknown) => key === '$schemaName' ? undefined : value));
    const response = { ...source, items: [{ ...source.items[0], state: 'succeeded' as const, operation_id: 'terminal', updated_at: new Date(0).toISOString() }, { ...source.items[0], state: 'running' as const, operation_id: 'active', updated_at: new Date(0).toISOString() }] };
    const state = reduceWebUiSnapshot(initialSnapshot(), { type: 'operations-snapshot', response, now: 1 });
    expect(state.operations.size).toBe(2);
    const later = reduceWebUiSnapshot(state, { type: 'operations-snapshot', response, now: 3_600_001 });
    expect([...later.operations.keys()]).toEqual(['active']);
    expect([...later.resourceFences.operations.keys()]).toEqual(['active']);
  });

  it('does not let another model runtime response or event replace selected history', () => {
    const selected = { ...runtime, model_id: 'mdl_selected' };
    let state = reduceWebUiSnapshot({ ...initialSnapshot(), selectedModelId: selected.model_id }, { type: 'runtime', runtime: selected, sequence: 1, now: 0 });
    const history = state.runtimeHistory;
    state = reduceWebUiSnapshot(state, { type: 'runtime', runtime, sequence: 2, now: 2_000 });
    expect(state.runtimeHistory).toBe(history);
    state = reduceWebUiSnapshot(state, { type: 'event', event: { schema_version: 'webui.ui-api.v1', server_instance_id: runtime.server_instance_id, sequence: 3, type: 'runtime', payload: { runtime }, event_id: 'evt_other', emitted_at: new Date(3_000).toISOString() }, now: 3_000 });
    expect(state.runtimeHistory).toBe(history);
    expect(state.runtimes.has(runtime.model_id)).toBe(true);
  });

});
