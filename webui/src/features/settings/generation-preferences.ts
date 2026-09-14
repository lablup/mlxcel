import { useSyncExternalStore } from 'react';
import { DEFAULT_GENERATION_DEFAULTS, validateGenerationDefaults, type GenerationDefaults } from './generation-defaults';
let defaults = DEFAULT_GENERATION_DEFAULTS;
const listeners = new Set<() => void>();
function subscribe(listener: () => void): () => void { listeners.add(listener); return () => listeners.delete(listener); }
function setDefaults(next: GenerationDefaults): void { defaults = validateGenerationDefaults(next); for (const listener of listeners) listener(); }
function reset(): void { setDefaults(DEFAULT_GENERATION_DEFAULTS); }
/** Memory only: never saves prompts, credentials or generation defaults to storage. */
export function useGenerationDefaults(): { defaults: GenerationDefaults; setDefaults: typeof setDefaults; reset: typeof reset } {
  return { defaults: useSyncExternalStore(subscribe, () => defaults, () => DEFAULT_GENERATION_DEFAULTS), setDefaults, reset };
}
