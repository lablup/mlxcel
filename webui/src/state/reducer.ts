// Copyright 2026 Lablup Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import { WEBUI_SCHEMA_VERSION } from '../api/types';
import type { BootstrapResponse, CatalogEntry, CatalogListResponse, ModelId, Operation, OperationId, OperationsListResponse, PendingReconciliation, RuntimeSnapshot, UiClientError, UiEvent, WebUiSnapshot } from '../api/types';

export type WebUiAction =
  | { readonly type: 'login-start' }
  | { readonly type: 'login-success'; readonly bootstrap: BootstrapResponse; readonly now: number }
  | { readonly type: 'logout'; readonly now: number }
  | { readonly type: 'select-model'; readonly modelId: ModelId | null }
  | { readonly type: 'catalog'; readonly response: CatalogListResponse; readonly now: number }
  | { readonly type: 'operations-snapshot'; readonly response: OperationsListResponse; readonly now: number }
  | { readonly type: 'operation'; readonly operation: Operation; readonly sequence: number | null; readonly now: number }
  | { readonly type: 'runtime'; readonly runtime: RuntimeSnapshot; readonly sequence: number | null; readonly now: number }
  | { readonly type: 'event'; readonly event: UiEvent; readonly now: number }
  | { readonly type: 'pending'; readonly item: PendingReconciliation }
  | { readonly type: 'reconciled'; readonly idempotencyKey: string }
  | { readonly type: 'connection'; readonly connection: WebUiSnapshot['connection']; readonly error?: UiClientError | null; readonly now: number };

export function initialSnapshot(): WebUiSnapshot {
  return { schemaVersion: WEBUI_SCHEMA_VERSION, auth: { status: 'signed-out', tokenPresent: false }, connection: 'idle', bootstrap: null, catalog: [], catalogSequence: null, operations: new Map(), runtimes: new Map(), selectedModelId: null, serverInstanceId: null, lastEventId: null, lastSequence: null, lastUpdatedAt: null, lastSuccessfulAt: null, error: null, pendingReconciliations: new Map(), resourceFences: { catalog: null, operationsSnapshot: null, operations: new Map(), models: new Map(), runtimes: new Map() } };
}

export function reduceWebUiSnapshot(state: WebUiSnapshot, action: WebUiAction): WebUiSnapshot {
  if (action.type === 'login-start') return { ...state, auth: { status: 'authenticating', tokenPresent: true }, connection: 'bootstrapping', error: null };
  if (action.type === 'login-success') return loginSuccess(state, action.bootstrap, action.now);
  if (action.type === 'logout') return { ...initialSnapshot(), lastUpdatedAt: action.now };
  if (action.type === 'select-model') return { ...state, selectedModelId: action.modelId };
  if (action.type === 'catalog') return applyCatalog(state, action.response, action.now);
  if (action.type === 'operations-snapshot') return applyOperationsSnapshot(state, action.response, action.now);
  if (action.type === 'operation') return applyOperation(state, action.operation, action.sequence, action.now);
  if (action.type === 'runtime') return applyRuntimeSnapshot(state, action.runtime, action.sequence, action.now);
  if (action.type === 'event') return applyEvent(state, action.event, action.now);
  if (action.type === 'pending') return { ...state, pendingReconciliations: mapSet(state.pendingReconciliations, action.item.idempotencyKey, action.item) };
  if (action.type === 'reconciled') return { ...state, pendingReconciliations: mapDelete(state.pendingReconciliations, action.idempotencyKey) };
  return { ...state, connection: action.connection, error: action.error ?? null, lastUpdatedAt: action.now };
}

function loginSuccess(state: WebUiSnapshot, bootstrap: BootstrapResponse, now: number): WebUiSnapshot {
  const serverInstanceId = bootstrap.server.server_instance_id;
  if (state.serverInstanceId !== null && state.serverInstanceId !== serverInstanceId) {
    return { ...initialSnapshot(), auth: { status: 'authenticated', tokenPresent: true }, connection: 'ready', bootstrap, serverInstanceId, lastUpdatedAt: now };
  }
  return { ...state, auth: { status: 'authenticated', tokenPresent: true }, connection: 'ready', bootstrap, serverInstanceId, lastUpdatedAt: now, error: null };
}

function applyCatalog(state: WebUiSnapshot, response: CatalogListResponse, now: number): WebUiSnapshot {
  if (state.serverInstanceId !== null && response.server_instance_id !== state.serverInstanceId) return restart(state, response.server_instance_id, now);
  if (state.catalogSequence !== null && response.snapshot_sequence < state.catalogSequence) return state;
  const shouldMerge = state.catalogSequence === response.snapshot_sequence;
  return markSuccessful({ ...state, connection: state.connection === 'bootstrapping' ? 'ready' : state.connection, catalog: mergeCatalogPage(shouldMerge ? state.catalog : [], response), catalogSequence: response.snapshot_sequence, serverInstanceId: response.server_instance_id, lastSequence: minReplaySequence({ ...state.resourceFences, catalog: response.snapshot_sequence }), error: null, resourceFences: { ...state.resourceFences, catalog: response.snapshot_sequence } }, now);
}

function applyOperationsSnapshot(state: WebUiSnapshot, response: OperationsListResponse, now: number): WebUiSnapshot {
  if (state.serverInstanceId !== null && response.server_instance_id !== state.serverInstanceId) return restart(state, response.server_instance_id, now);
  if (state.resourceFences.operationsSnapshot !== null && response.snapshot_sequence < state.resourceFences.operationsSnapshot) return state;
  const shouldMerge = state.resourceFences.operationsSnapshot === response.snapshot_sequence;
  const operations: Map<OperationId, Operation> = shouldMerge ? new Map(state.operations) : new Map();
  const operationFences: Map<OperationId, number> = shouldMerge ? new Map(state.resourceFences.operations) : new Map();
  for (const operation of response.items) {
    operations.set(operation.operation_id, operation);
    operationFences.set(operation.operation_id, response.snapshot_sequence);
  }
  const resourceFences = { ...state.resourceFences, operationsSnapshot: response.snapshot_sequence, operations: operationFences };
  return markSuccessful({ ...state, operations, serverInstanceId: response.server_instance_id, lastSequence: minReplaySequence(resourceFences), error: null, resourceFences }, now);
}

function applyOperation(state: WebUiSnapshot, operation: Operation, sequence: number | null, now: number): WebUiSnapshot {
  if (sequence !== null && state.resourceFences.operationsSnapshot !== null && sequence <= state.resourceFences.operationsSnapshot) return state;
  const previousFence = state.resourceFences.operations.get(operation.operation_id);
  if (sequence !== null && previousFence !== undefined && sequence <= previousFence) return state;
  const operations = mapSet(state.operations, operation.operation_id, operation);
  const fences = sequence === null ? state.resourceFences.operations : mapSet(state.resourceFences.operations, operation.operation_id, sequence);
  const resourceFences = { ...state.resourceFences, operations: fences };
  return markSuccessful({ ...state, operations, lastSequence: minReplaySequence(resourceFences), error: null, resourceFences }, now);
}

function applyRuntimeSnapshot(state: WebUiSnapshot, runtime: RuntimeSnapshot, sequence: number | null, now: number): WebUiSnapshot {
  if (state.serverInstanceId !== null && runtime.server_instance_id !== state.serverInstanceId) return restart(state, runtime.server_instance_id, now);
  const previousFence = state.resourceFences.runtimes.get(runtime.model_id);
  if (sequence !== null && previousFence !== undefined && sequence <= previousFence) return state;
  const fences = sequence === null ? state.resourceFences.runtimes : mapSet(state.resourceFences.runtimes, runtime.model_id, sequence);
  const resourceFences = { ...state.resourceFences, runtimes: fences };
  return markSuccessful({ ...state, runtimes: mapSet(state.runtimes, runtime.model_id, runtime), serverInstanceId: runtime.server_instance_id, lastSequence: minReplaySequence(resourceFences), error: null, resourceFences }, now);
}

function applyEvent(state: WebUiSnapshot, event: UiEvent, now: number): WebUiSnapshot {
  if (event.schema_version !== WEBUI_SCHEMA_VERSION) return { ...state, connection: 'schema-mismatch', error: { code: 'schema_mismatch', message: `Unsupported WebUI schema ${event.schema_version}`, retryable: false }, lastUpdatedAt: now };
  if (state.serverInstanceId !== null && event.server_instance_id !== state.serverInstanceId) return restart(state, event.server_instance_id, now);
  if (event.type === 'heartbeat') return { ...state, serverInstanceId: event.server_instance_id, lastEventId: event.event_id, connection: state.connection === 'polling' ? 'polling' : 'streaming', lastUpdatedAt: now };
  if (event.type === 'server_restart' || event.type === 'gap' || event.type === 'reset') return resetForResnapshot(state, event, now);
  if (event.type === 'operation') return { ...applyOperation(state, event.payload.operation, event.sequence, now), serverInstanceId: event.server_instance_id, lastEventId: event.event_id, connection: 'streaming' };
  if (event.type === 'model_revision') return applyModelRevision(state, event, now);
  if (event.type === 'runtime') return applyRuntime(state, event, now);
  if (event.type === 'snapshot') {
    const resourceFences = { ...state.resourceFences, catalog: event.payload.catalog_changed ? event.sequence : state.resourceFences.catalog };
    return { ...state, serverInstanceId: event.server_instance_id, lastEventId: event.event_id, lastSequence: minReplaySequence(resourceFences), connection: 'streaming', lastUpdatedAt: now, resourceFences };
  }
  return markSuccessful({ ...state, serverInstanceId: event.server_instance_id, lastEventId: event.event_id, connection: 'streaming' }, now);
}

function applyModelRevision(state: WebUiSnapshot, event: Extract<UiEvent, { type: 'model_revision' }>, now: number): WebUiSnapshot {
  const modelId = event.payload.model_id;
  const previousFence = state.resourceFences.models.get(modelId);
  if (previousFence !== undefined && event.sequence <= previousFence) return state;
  const catalog = state.catalog.map((entry) => entry.identity.id === modelId && event.payload.revision >= entry.identity.revision ? { ...entry, identity: { ...entry.identity, revision: event.payload.revision }, lifecycle: event.payload.lifecycle } : entry);
  const resourceFences = { ...state.resourceFences, models: mapSet(state.resourceFences.models, modelId, event.sequence) };
  return markSuccessful({ ...state, catalog, serverInstanceId: event.server_instance_id, lastEventId: event.event_id, lastSequence: minReplaySequence(resourceFences), connection: 'streaming', resourceFences }, now);
}

function applyRuntime(state: WebUiSnapshot, event: Extract<UiEvent, { type: 'runtime' }>, now: number): WebUiSnapshot {
  const runtime = event.payload.runtime;
  const previousFence = state.resourceFences.runtimes.get(runtime.model_id);
  if (previousFence !== undefined && event.sequence <= previousFence) return state;
  const resourceFences = { ...state.resourceFences, runtimes: mapSet(state.resourceFences.runtimes, runtime.model_id, event.sequence) };
  return markSuccessful({ ...state, runtimes: mapSet(state.runtimes, runtime.model_id, runtime), serverInstanceId: event.server_instance_id, lastEventId: event.event_id, lastSequence: minReplaySequence(resourceFences), connection: 'streaming', resourceFences }, now);
}

function resetForResnapshot(state: WebUiSnapshot, event: Extract<UiEvent, { type: 'server_restart' | 'gap' | 'reset' }>, now: number): WebUiSnapshot {
  const lastSuccessfulAt = event.type === 'server_restart' ? null : state.lastSuccessfulAt;
  return { ...initialSnapshot(), auth: state.auth, connection: 'stale', serverInstanceId: event.server_instance_id, selectedModelId: state.selectedModelId, lastEventId: event.event_id, error: { code: event.type, message: event.payload.reason, retryable: event.payload.resnapshot }, pendingReconciliations: state.pendingReconciliations, lastUpdatedAt: now, lastSuccessfulAt };
}

function restart(state: WebUiSnapshot, serverInstanceId: string, now: number): WebUiSnapshot {
  return { ...initialSnapshot(), auth: state.auth, connection: 'stale', serverInstanceId, selectedModelId: state.selectedModelId, error: { code: 'server_restarted', message: 'The mlxcel server restarted; refresh the authoritative snapshot before continuing.', retryable: true }, pendingReconciliations: state.pendingReconciliations, lastUpdatedAt: now };
}

function markSuccessful(state: WebUiSnapshot, now: number): WebUiSnapshot {
  return { ...state, lastUpdatedAt: now, lastSuccessfulAt: now };
}

function mergeCatalogPage(current: ReadonlyArray<CatalogEntry>, response: CatalogListResponse): ReadonlyArray<CatalogEntry> {
  if (current.length > 0) {
    const map = new Map(current.map((entry) => [entry.identity.id, entry]));
    for (const item of response.items) map.set(item.identity.id, item);
    return Array.from(map.values()).sort((left, right) => left.identity.display_name.localeCompare(right.identity.display_name));
  }
  return response.items;
}

export function minReplaySequence(fences: WebUiSnapshot['resourceFences']): number | null {
  const values = [fences.catalog, fences.operationsSnapshot, ...fences.operations.values(), ...fences.models.values(), ...fences.runtimes.values()].filter((value): value is number => value !== null);
  return values.length === 0 ? null : Math.min(...values);
}

function mapSet<K, V>(input: ReadonlyMap<K, V>, key: K, value: V): ReadonlyMap<K, V> {
  const next = new Map(input);
  next.set(key, value);
  return next;
}

function mapDelete<K, V>(input: ReadonlyMap<K, V>, key: K): ReadonlyMap<K, V> {
  const next = new Map(input);
  next.delete(key);
  return next;
}
