// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import { useSyncExternalStore } from 'react';
import { DEFAULT_SORT, type InventoryFilter, type InventorySort } from './policy';

// The library's search, filters and sort. The router remounts the Models screen on every
// route change, so they live here for the session instead of in component state: a Chat round
// trip keeps them. A reload starts over; nothing is persisted. The page number is not kept.
// The view belongs to the server instance it was set against: another instance (a restarted
// or different server, whose catalog may lack a remembered source or task) reads the
// defaults, and logout clears it so the next session in this tab does not inherit it.
export type LibraryView = { readonly filter: InventoryFilter; readonly sort: InventorySort };
type StoredView = LibraryView & { readonly instance: string | null };

const INITIAL: StoredView = { filter: { query: '', source: '', task: '', status: '' }, sort: DEFAULT_SORT, instance: null };
let view: StoredView = INITIAL;
const listeners = new Set<() => void>();
function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}
function publish(next: StoredView): void {
  view = next;
  for (const listener of listeners) listener();
}

/** The view for `instance`; the same object until it changes, so it can key a memo. */
export function useLibraryView(instance: string | null): LibraryView {
  const stored = useSyncExternalStore(subscribe, () => view);
  return stored.instance === null || stored.instance === instance ? stored : INITIAL;
}
function current(instance: string | null): StoredView {
  return view.instance === null || view.instance === instance ? view : INITIAL;
}
export function setLibraryFilter(instance: string | null, filter: InventoryFilter): void {
  publish({ ...current(instance), filter, instance });
}
export function setLibrarySort(instance: string | null, sort: InventorySort): void {
  publish({ ...current(instance), sort, instance });
}
/** Back to the defaults, as a reload would. Logout calls it; tests start from here. */
export function resetLibraryView(): void {
  publish(INITIAL);
}
