// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import { useSyncExternalStore } from 'react';
import { DEFAULT_SORT, type InventoryFilter, type InventorySort } from './policy';

// The library's search, filters and sort. The router remounts the Models screen on every
// route change, so they live here for the session instead of in component state: a Chat round
// trip keeps them. A reload starts over; nothing is persisted. The page number is not kept.
export type LibraryView = { readonly filter: InventoryFilter; readonly sort: InventorySort };

const INITIAL: LibraryView = { filter: { query: '', source: '', task: '', status: '' }, sort: DEFAULT_SORT };
let view: LibraryView = INITIAL;
const listeners = new Set<() => void>();
function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}
function publish(next: LibraryView): void {
  view = next;
  for (const listener of listeners) listener();
}

/** The current view; the same object until it changes, so it can key a memo. */
export function useLibraryView(): LibraryView {
  return useSyncExternalStore(subscribe, () => view);
}
export function setLibraryFilter(filter: InventoryFilter): void {
  publish({ ...view, filter });
}
export function setLibrarySort(sort: InventorySort): void {
  publish({ ...view, sort });
}
/** Back to the defaults, as a reload would. Tests start from here. */
export function resetLibraryView(): void {
  publish(INITIAL);
}
