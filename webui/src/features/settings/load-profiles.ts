import { useSyncExternalStore } from 'react';
import type { KvCacheModeName, LoadProfile } from '../../api/types';
export const KV_MODES: readonly KvCacheModeName[] = ['fp16', 'float16', 'int8', 'i8', 'turbo4-asym', 'fp16+turbo4', 'turbo3-asym', 'fp16+turbo3', 'turbo3', 'turbo4', 'turbo4-sym', 'turbo4-delegated', 'fp16+turbo4-delegated'];
const EMPTY: LoadProfile = Object.freeze({});
export function validateLoadProfile(value: unknown): LoadProfile {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) throw new Error('Profile must be an object.');
  const result: {ctx_size?: number; n_parallel?: number; kv_cache_mode?: KvCacheModeName} = {};
  for (const [key, item] of Object.entries(value)) {
    if (!['ctx_size', 'n_parallel', 'kv_cache_mode'].includes(key)) throw new Error(`Unsafe or unsupported profile field: ${key}`);
    if (item === undefined || item === null) continue;
    if (key === 'kv_cache_mode') {
      if (typeof item !== 'string' || !KV_MODES.includes(item as KvCacheModeName)) throw new Error('Unknown KV cache mode.');
      result.kv_cache_mode = item as KvCacheModeName;
    } else {
      if (typeof item !== 'number' || !Number.isInteger(item) || item < 1 || item > (key === 'ctx_size' ? 262144 : 32)) throw new Error(`${key} is outside its supported range.`);
      if (key === 'ctx_size') result.ctx_size = item; else result.n_parallel = item;
    }
  }
  return Object.freeze(result);
}
export interface ProfileDocument { readonly version: 1; readonly reusable: LoadProfile; readonly models: Readonly<Record<string, LoadProfile>> }
export function importProfiles(text: string): ProfileDocument {
  if (text.length > 65536) throw new Error('Profile import exceeds 64 KiB.');
  const value: unknown = JSON.parse(text);
  if (typeof value !== 'object' || value === null || Array.isArray(value)) throw new Error('Invalid profile document.');
  const data = value as Record<string, unknown>;
  // Version zero was a single reusable profile. No arbitrary keys survive migration.
  if (data.version === 0 && Object.keys(data).every((key) => ['version', 'profile'].includes(key))) return { version: 1, reusable: validateLoadProfile(data.profile), models: {} };
  if (data.version !== 1 || Object.keys(data).some((key) => !['version', 'reusable', 'models'].includes(key))) throw new Error('Unsupported profile version or fields.');
  if (typeof data.models !== 'object' || data.models === null || Array.isArray(data.models)) throw new Error('Invalid model profiles.');
  const models: Record<string, LoadProfile> = Object.create(null) as Record<string, LoadProfile>;
  if (Object.keys(data.models).length > 200) throw new Error('Too many model profiles.');
  for (const [id, profile] of Object.entries(data.models)) {
    if (!/^mdl_[A-Za-z0-9_-]{43}$/.test(id)) throw new Error('Model profiles require opaque catalog IDs.');
    models[id] = validateLoadProfile(profile);
  }
  return { version: 1, reusable: validateLoadProfile(data.reusable), models };
}
const STORAGE_KEY = 'mlxcel.webui.load-profiles.v1';
let document: ProfileDocument = { version: 1, reusable: EMPTY, models: {} };
let loaded = false;
const listeners = new Set<() => void>();
function subscribe(listener: () => void): () => void { listeners.add(listener); return () => listeners.delete(listener); }
function load(): void {
  if (loaded) return;
  loaded = true;
  try { const raw = localStorage.getItem(STORAGE_KEY); if (raw !== null) document = importProfiles(raw); } catch { /* Untrusted/unavailable storage never authorizes a profile. */ }
}
function publish(next: ProfileDocument): void {
  const encoded = JSON.stringify(next);
  const checked = importProfiles(encoded);
  // Explicit save fails visibly rather than claiming persistence in private browsing.
  localStorage.setItem(STORAGE_KEY, encoded);
  document = checked;
  for (const listener of listeners) listener();
}
export function useLoadProfile(modelId: string | null): { profile: LoadProfile; reusable: LoadProfile; modelProfile: LoadProfile; save: (profile: LoadProfile, scope: 'model' | 'reusable') => void; reset: (scope?: 'model' | 'reusable') => void; exportJson: () => string; importJson: (text: string) => void } {
  load();
  const current = useSyncExternalStore(subscribe, () => document);
  return {
    reusable: current.reusable,
    modelProfile: modelId === null ? EMPTY : current.models[modelId] ?? EMPTY,
    profile: modelId !== null && Object.hasOwn(current.models, modelId) ? current.models[modelId] : current.reusable,
    save: (profile, scope) => {
      const checked = validateLoadProfile(profile);
      if (scope === 'model' && modelId === null) throw new Error('Select a model first.');
      publish(scope === 'reusable' ? { ...document, reusable: checked } : { ...document, models: { ...document.models, [modelId as string]: checked } });
    },
    reset: (scope = modelId === null ? 'reusable' : 'model') => { const models = Object.fromEntries(Object.entries(document.models).filter(([id]) => id !== modelId)); publish(scope === 'reusable' ? { ...document, reusable: EMPTY } : { ...document, models }); },
    exportJson: () => JSON.stringify(document, null, 2),
    importJson: (text) => publish(importProfiles(text)),
  };
}
export function profileCli(profile: LoadProfile): string {
  const checked = validateLoadProfile(profile);
  return ['mlxcel-server --webui', checked.ctx_size === undefined ? '' : `--ctx-size ${checked.ctx_size}`, checked.n_parallel === undefined ? '' : `--parallel ${checked.n_parallel}`, checked.kv_cache_mode === undefined ? '' : `--kv-cache-mode ${checked.kv_cache_mode}`].filter(Boolean).join(' ');
}
/** What an explicit load of `modelId` submits right now: its own saved profile, else the reusable defaults. Read at submit time, so a row can load any entry, not only the selected one. */
export function loadProfileFor(modelId: string): LoadProfile {
  load();
  return Object.hasOwn(document.models, modelId) ? document.models[modelId] : document.reusable;
}
