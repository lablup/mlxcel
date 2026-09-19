// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
// Server defaults for the request fields, as last read from a model's /settings.
// Settings records every read it makes; Chat only reads this record, so its
// "Inherited" hint and the Requests tab agree without Chat issuing requests of its own.
import { useSyncExternalStore } from 'react';
import type { SettingsResponse } from '../../api/settings';
import type { CatalogEntry, WebUiSnapshot } from '../../api/types';
import { useWebUi } from '../../state';
import { GENERATION_FIELDS, type GenerationField } from './generation-defaults';

/** A field is absent when the model's schema has no `default_<field>` entry: that server default is unknown. */
export type ServerGenerationDefaults = Readonly<Partial<Record<GenerationField, unknown>>>;

/**
 * Which worker a record belongs to. A reload bumps the catalog revision and starts a new worker; a server
 * restart can reuse both the model id and the revision (single-model mode is always revision 1), so the
 * server instance is part of the key too.
 */
export interface ServerDefaultsScope { readonly instance: string; readonly modelId: string; readonly revision: number }

/** The scope of a Ready catalog entry on the connected server, or null when there is nothing to read. */
export function serverDefaultsScope(snapshot: WebUiSnapshot, entry: CatalogEntry | null): ServerDefaultsScope | null {
  const instance = snapshot.bootstrap?.server.server_instance_id;
  if (instance === undefined || entry === null || entry.lifecycle.state !== 'ready') return null;
  return { instance, modelId: entry.identity.id, revision: entry.identity.revision };
}

// Bounded: one entry per worker read in this page session. Memory only.
const MAX_RECORDS = 32;
const records = new Map<string, ServerGenerationDefaults>();
const listeners = new Set<() => void>();
function subscribe(listener: () => void): () => void { listeners.add(listener); return () => listeners.delete(listener); }
const recordKey = (scope: ServerDefaultsScope): string => JSON.stringify([scope.instance, scope.modelId, scope.revision]);

export function serverGenerationDefaults(response: SettingsResponse): ServerGenerationDefaults {
  const names = new Set(response.schema.map((spec) => spec.name));
  const result: Partial<Record<GenerationField, unknown>> = {};
  for (const field of GENERATION_FIELDS) {
    const name = `default_${field}`;
    if (names.has(name) && Object.hasOwn(response.current, name)) result[field] = response.current[name];
  }
  return Object.freeze(result);
}

export function recordServerSettings(scope: ServerDefaultsScope, response: SettingsResponse): void {
  const key = recordKey(scope);
  records.delete(key);
  records.set(key, serverGenerationDefaults(response));
  while (records.size > MAX_RECORDS) {
    const oldest = records.keys().next();
    if (oldest.done) break;
    records.delete(oldest.value);
  }
  for (const listener of listeners) listener();
}

export function useServerGenerationDefaults(scope: ServerDefaultsScope | null): ServerGenerationDefaults | null {
  const key = scope === null ? null : recordKey(scope);
  return useSyncExternalStore(subscribe, () => key === null ? null : records.get(key) ?? null, () => null);
}

/** The record for the model selected across the app, the one Chat sends to and Settings reads. */
export function useSelectedServerGenerationDefaults(): ServerGenerationDefaults | null {
  const snapshot = useWebUi();
  const selected = snapshot.catalog.find((entry) => entry.identity.id === snapshot.selectedModelId) ?? null;
  return useServerGenerationDefaults(serverDefaultsScope(snapshot, selected));
}
