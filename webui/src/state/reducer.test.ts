import { describe, expect, it } from 'vitest';
import bootstrapFixture from '../../../tests/fixtures/webui/examples/bootstrap.model-free.json';
import type { CatalogListResponse, Operation, PendingReconciliation, UiEvent } from '../api/types';
import { validateBootstrap } from '../api/validation';
import { initialSnapshot, reduceWebUiSnapshot } from './reducer';

const lifecycle = { state: 'ready', download: 'complete', busy: false, active_requests: 0, draining_requests: 0, worker_exit_observed: true, last_error: null } as const;
const entry = { identity: { id: 'mdl_a', inference_id: 'model/a', display_name: 'Model A', source: 'cache', source_key_hash: 'h', generation: 1, revision: 5, content_fingerprint: null }, capabilities: [], lifecycle, complete: true, supported: true, removable: true, metadata: { architecture: null, input_tasks: ['chat'], output_tasks: ['chat'], quantization: null, format: null, parameter_count: null, disk_bytes: null, memory_estimate_bytes: null, support: { architecturally_supported: true, runnable_on_backend: true, complete: true, reason: null } } } as const;
const bootstrap = validateBootstrap(bootstrapFixture);

function catalog(sequence: number): CatalogListResponse {
  return { schema_version: 'webui.ui-api.v1', items: [entry], pagination: { limit: 50, next_cursor: null, total_known: 1 }, server_instance_id: 'srv', snapshot_sequence: sequence };
}

function event(sequence: number, revision: number): Extract<UiEvent, { type: 'model_revision' }> {
  return { schema_version: 'webui.ui-api.v1', server_instance_id: 'srv', sequence, type: 'model_revision', payload: { model_id: 'mdl_a', revision, lifecycle: { ...lifecycle, state: 'draining' } }, event_id: `evt_${sequence}`, emitted_at: '2026-09-12T00:00:00Z' };
}

function operation(id: string): Operation {
  return { operation_id: id, kind: 'model_load', state: 'running', created_at: '2026-09-12T00:00:00Z', updated_at: '2026-09-12T00:00:01Z', idempotency_scope: 'server_instance', target: { target_kind: 'model', model_id: 'mdl_a', requested_revision: 5 }, progress: { completed_bytes: 0, total_bytes: null, indeterminate: true }, result: null, error: null, cancellable: false, cancel_reason: null };
}

describe('WebUI reducer', () => {
  it('tracks last successful data freshness separately from state transitions', () => {
    let state = initialSnapshot();
    expect(state.lastSuccessfulAt).toBeNull();
    state = reduceWebUiSnapshot(state, { type: 'login-start' });
    state = reduceWebUiSnapshot(state, { type: 'login-success', bootstrap: { ...bootstrap, server: { ...bootstrap.server, server_instance_id: 'srv' } }, now: 1 });
    expect(state.lastSuccessfulAt).toBeNull();
    state = reduceWebUiSnapshot(state, { type: 'catalog', response: catalog(10), now: 10 });
    expect(state.lastSuccessfulAt).toBe(10);
    state = reduceWebUiSnapshot(state, { type: 'event', event: { schema_version: 'webui.ui-api.v1', server_instance_id: 'srv', sequence: 11, type: 'snapshot', payload: { snapshot_sequence: 11, catalog_changed: true, operations_changed: true, runtime_model_ids: ['mdl_a'] }, event_id: 'evt_snapshot', emitted_at: '2026-09-12T00:00:00Z' }, now: 15 });
    expect(state.lastUpdatedAt).toBe(15);
    expect(state.lastSuccessfulAt).toBe(10);
    state = reduceWebUiSnapshot(state, { type: 'connection', connection: 'offline', error: { code: 'offline', message: 'offline', retryable: true }, now: 20 });
    expect(state.lastUpdatedAt).toBe(20);
    expect(state.lastSuccessfulAt).toBe(10);
    state = reduceWebUiSnapshot(state, { type: 'event', event: { schema_version: 'webui.ui-api.v1', server_instance_id: 'srv', sequence: 11, type: 'heartbeat', payload: { server_time: '2026-09-12T00:00:00Z' }, event_id: 'evt_heartbeat', emitted_at: '2026-09-12T00:00:00Z' }, now: 30 });
    expect(state.lastSuccessfulAt).toBe(10);
    state = reduceWebUiSnapshot(state, { type: 'event', event: { schema_version: 'webui.ui-api.v1', server_instance_id: 'srv', sequence: 12, type: 'gap', payload: { reason: 'gap', resnapshot: true }, event_id: 'evt_gap', emitted_at: '2026-09-12T00:00:00Z' }, now: 40 });
    expect(state.connection).toBe('stale');
    expect(state.lastSuccessfulAt).toBe(10);
    state = reduceWebUiSnapshot(state, { type: 'event', event: { schema_version: 'webui.ui-api.v1', server_instance_id: 'srv2', sequence: 1, type: 'server_restart', payload: { reason: 'server_restart', resnapshot: true }, event_id: 'evt_restart', emitted_at: '2026-09-12T00:00:00Z' }, now: 50 });
    expect(state.lastSuccessfulAt).toBeNull();
    state = reduceWebUiSnapshot(state, { type: 'catalog', response: { ...catalog(1), server_instance_id: 'srv2' }, now: 60 });
    expect(state.lastSuccessfulAt).toBe(60);
    state = reduceWebUiSnapshot(state, { type: 'logout', now: 70 });
    expect(state.lastSuccessfulAt).toBeNull();
  });

  it('does not refresh data freshness for duplicate or older per-resource events', () => {
    let state = reduceWebUiSnapshot(initialSnapshot(), { type: 'catalog', response: catalog(10), now: 10 });
    state = reduceWebUiSnapshot(state, { type: 'event', event: event(12, 6), now: 12 });
    expect(state.lastSuccessfulAt).toBe(12);
    state = reduceWebUiSnapshot(state, { type: 'event', event: event(11, 99), now: 20 });
    expect(state.catalog[0]?.identity.revision).toBe(6);
    expect(state.lastSuccessfulAt).toBe(12);
  });

  it('uses per-resource sequence fences rather than one global maximum', () => {
    let state = reduceWebUiSnapshot(initialSnapshot(), { type: 'catalog', response: catalog(100), now: 1 });
    state = reduceWebUiSnapshot(state, { type: 'event', event: { ...event(101, 6), event_id: 'evt_101' }, now: 2 });
    state = reduceWebUiSnapshot(state, { type: 'operation', operation: operation('op_1'), sequence: 90, now: 3 });
    expect(state.catalog[0]?.identity.revision).toBe(6);
    expect(state.operations.get('op_1')?.state).toBe('running');
    expect(state.lastSequence).toBe(90);
  });

  it('drops obsolete model revision responses for the same model only', () => {
    let state = reduceWebUiSnapshot(initialSnapshot(), { type: 'catalog', response: catalog(10), now: 1 });
    state = reduceWebUiSnapshot(state, { type: 'event', event: event(12, 6), now: 2 });
    state = reduceWebUiSnapshot(state, { type: 'event', event: event(11, 99), now: 3 });
    expect(state.catalog[0]?.identity.revision).toBe(6);
  });

  it('marks gap and reset events stale until a full resnapshot runs', () => {
    let state = reduceWebUiSnapshot(initialSnapshot(), { type: 'catalog', response: catalog(10), now: 1 });
    state = reduceWebUiSnapshot(state, { type: 'event', event: { schema_version: 'webui.ui-api.v1', server_instance_id: 'srv', sequence: 11, type: 'gap', payload: { reason: 'gap', resnapshot: true }, event_id: 'evt_gap', emitted_at: '2026-09-12T00:00:00Z' }, now: 2 });
    expect(state.connection).toBe('stale');
    expect(state.error?.retryable).toBe(true);
    expect(state.catalog).toEqual([]);
    expect(state.lastSequence).toBeNull();
  });

  it('clears stale instance data when login succeeds against a new server instance', () => {
    const pending: PendingReconciliation = { kind: 'model-action', idempotencyKey: 'idem-1', operationId: null, modelId: 'mdl_a', createdAt: 1 };
    let state = reduceWebUiSnapshot(initialSnapshot(), { type: 'select-model', modelId: 'mdl_a' });
    state = reduceWebUiSnapshot(state, { type: 'pending', item: pending });
    state = reduceWebUiSnapshot(state, { type: 'catalog', response: catalog(10), now: 1 });
    state = reduceWebUiSnapshot(state, { type: 'login-success', bootstrap: { ...bootstrap, server: { ...bootstrap.server, server_instance_id: 'srv2' } }, now: 2 });
    expect(state.serverInstanceId).toBe('srv2');
    expect(state.catalog).toEqual([]);
    expect(state.selectedModelId).toBeNull();
    expect(state.pendingReconciliations.size).toBe(0);
    expect(state.resourceFences.catalog).toBeNull();
  });

  it('preserves unknown POST reconciliation records across server restart reset', () => {
    const pending: PendingReconciliation = { kind: 'model-action', idempotencyKey: 'idem-1', operationId: null, modelId: 'mdl_a', createdAt: 1 };
    let state = reduceWebUiSnapshot(initialSnapshot(), { type: 'pending', item: pending });
    state = reduceWebUiSnapshot(state, { type: 'event', event: { schema_version: 'webui.ui-api.v1', server_instance_id: 'srv2', sequence: 1, type: 'server_restart', payload: { reason: 'server_restart', resnapshot: true }, event_id: 'evt_restart', emitted_at: '2026-09-12T00:00:00Z' }, now: 2 });
    expect(state.connection).toBe('stale');
    expect(state.pendingReconciliations.has('idem-1')).toBe(true);
  });

  it('logout clears selected model, operations and memory-only auth state', () => {
    let state = reduceWebUiSnapshot(initialSnapshot(), { type: 'select-model', modelId: 'mdl_a' });
    state = reduceWebUiSnapshot(state, { type: 'operation', operation: operation('op_1'), sequence: 1, now: 1 });
    state = reduceWebUiSnapshot(state, { type: 'logout', now: 2 });
    expect(state.auth.status).toBe('signed-out');
    expect(state.selectedModelId).toBeNull();
    expect(state.operations.size).toBe(0);
  });
});
