// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import catalogFixture from '../../tests/fixtures/webui/examples/catalog.page.json' with { type: 'json' };
import bootstrapFixture from '../../tests/fixtures/webui/examples/bootstrap.model-free.json' with { type: 'json' };
import operationsFixture from '../../tests/fixtures/webui/examples/operations.list.json' with { type: 'json' };
import runtimeFixture from '../../tests/fixtures/webui/examples/runtime.snapshot.json' with { type: 'json' };
import type { BootstrapResponse, CatalogEntry, CatalogListResponse, OperationsListResponse, RuntimeSnapshot } from '../src/api/types';
import { model } from './models-fixtures';

type SchemaValidator = (schema: string, body: unknown) => void;

export function withoutFixtureAnnotations<T>(value: T): T {
  const copy = structuredClone(value) as T;
  if (typeof copy === 'object' && copy !== null && '$schemaName' in copy) delete (copy as Record<string, unknown>).$schemaName;
  return copy;
}

export function bootstrapResponse(): BootstrapResponse {
  return withoutFixtureAnnotations(bootstrapFixture as BootstrapResponse);
}

export function operationPage(sequence: number): OperationsListResponse {
  return { ...withoutFixtureAnnotations(operationsFixture as OperationsListResponse), items: [], server_instance_id: bootstrapFixture.server.server_instance_id, snapshot_sequence: sequence, pagination: { limit: 200, next_cursor: null, total_known: 0 } };
}

function validModelId(index: number): string {
  return `mdl_${index.toString(36).padStart(43, 'A')}`;
}

function validHex(index: number): string {
  return index.toString(16).padStart(64, '0').slice(-64);
}

export function catalogEntry(index: number, state: 'unloaded' | 'ready' = 'unloaded'): CatalogEntry {
  const base = model();
  return {
    ...base,
    identity: {
      ...base.identity,
      id: validModelId(index),
      inference_id: `perf-model-${index}`,
      display_name: `Catalog performance model ${String(index).padStart(4, '0')}`,
      source_key_hash: validHex(index + 1),
      revision: base.identity.revision + index,
    },
    lifecycle: { ...base.lifecycle, state, worker_exit_observed: state !== 'ready', busy: false, active_requests: 0, draining_requests: 0 },
    capabilities: [{ task: 'chat', phase: state === 'ready' ? 'provider_ready' : 'pre_load', available: true, reason: null }],
  };
}

export function catalogPage(items: readonly CatalogEntry[], offset: number, limit: number, sequence: number, total: number): CatalogListResponse {
  return {
    ...withoutFixtureAnnotations(catalogFixture as CatalogListResponse),
    items: items.slice(offset, offset + limit),
    pagination: { limit, next_cursor: offset + limit < items.length ? `cursor_${offset + limit}` : null, total_known: total },
    server_instance_id: bootstrapFixture.server.server_instance_id,
    snapshot_sequence: sequence,
  };
}

export function runtimeFor(entry: CatalogEntry, sequence: number): RuntimeSnapshot {
  return { ...withoutFixtureAnnotations(runtimeFixture as RuntimeSnapshot), server_instance_id: bootstrapFixture.server.server_instance_id, model_id: entry.identity.id, revision: entry.identity.revision, snapshot_sequence: sequence };
}

export function validatePerformanceFixtureSet(validate: SchemaValidator): string[] {
  const checked: string[] = [];
  const check = (schema: string, body: unknown, label: string): void => {
    validate(schema, body);
    checked.push(`${schema}:${label}`);
  };
  check('BootstrapResponse', bootstrapResponse(), 'bootstrap');
  check('OperationsListResponse', operationPage(3000), 'empty-operations');
  const entries = Array.from({ length: 1000 }, (_, index) => catalogEntry(index));
  for (let offset = 0; offset < entries.length; offset += 200) check('CatalogListResponse', catalogPage(entries, offset, 200, 3000, entries.length), `catalog-${offset}`);
  const ready = catalogEntry(0, 'ready');
  check('CatalogListResponse', catalogPage([ready], 0, 200, 3000, 1), 'ready-chat-catalog');
  check('RuntimeSnapshot', runtimeFor(ready, 3000), 'ready-runtime');
  return checked;
}
