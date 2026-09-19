// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
// Server defaults for the request fields, as last read from a model's /settings.
// Settings records every read it makes; Chat only reads this record, so its
// "Inherited" hint and the Requests tab agree without Chat issuing requests of its own.
import { useSyncExternalStore } from 'react';
import type { SettingsResponse } from '../../api/settings';
import { useWebUi } from '../../state';
import { GENERATION_FIELDS, type GenerationField } from './generation-defaults';

/** A field is absent when the model's schema has no `default_<field>` entry: that server default is unknown. */
export type ServerGenerationDefaults = Readonly<Partial<Record<GenerationField, unknown>>>;

// Bounded: one entry per model revision read in this page session. Memory only.
const MAX_RECORDS = 32;
const records = new Map<string, ServerGenerationDefaults>();
const listeners = new Set<() => void>();
function subscribe(listener: () => void): () => void { listeners.add(listener); return () => listeners.delete(listener); }
// A reload bumps the catalog revision and starts a new worker, so a revision never reuses another's defaults.
const recordKey = (modelId: string, revision: number): string => `${revision}:${modelId}`;

export function serverGenerationDefaults(response: SettingsResponse): ServerGenerationDefaults {
  const names = new Set(response.schema.map((spec) => spec.name));
  const result: Partial<Record<GenerationField, unknown>> = {};
  for (const field of GENERATION_FIELDS) {
    const name = `default_${field}`;
    if (names.has(name) && Object.hasOwn(response.current, name)) result[field] = response.current[name];
  }
  return Object.freeze(result);
}

export function recordServerSettings(modelId: string, revision: number, response: SettingsResponse): void {
  const key = recordKey(modelId, revision);
  records.delete(key);
  records.set(key, serverGenerationDefaults(response));
  while (records.size > MAX_RECORDS) {
    const oldest = records.keys().next();
    if (oldest.done) break;
    records.delete(oldest.value);
  }
  for (const listener of listeners) listener();
}

export function useServerGenerationDefaults(modelId: string | null, revision: number | null): ServerGenerationDefaults | null {
  return useSyncExternalStore(subscribe, () => modelId === null || revision === null ? null : records.get(recordKey(modelId, revision)) ?? null, () => null);
}

/** The record for the model selected across the app, the one Chat sends to and Settings reads. */
export function useSelectedServerGenerationDefaults(): ServerGenerationDefaults | null {
  const snapshot = useWebUi();
  const selected = snapshot.catalog.find((entry) => entry.identity.id === snapshot.selectedModelId) ?? null;
  const ready = selected?.lifecycle.state === 'ready';
  return useServerGenerationDefaults(ready ? selected.identity.id : null, ready ? selected.identity.revision : null);
}
