// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import { useSyncExternalStore } from 'react';

// A request from outside the Models screen (a toolbar chip, the command palette) to show the
// selected model's inspector. At 1100 px and wider the inspector is a pane that follows the
// selection anyway; below that it is a drawer, which opens only on an explicit request so that
// arriving at Models with a remembered selection does not cover the list.
let request = 0;
let counter = 0;
const listeners = new Set<() => void>();
function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function requestInspector(): void {
  counter += 1;
  request = counter;
  for (const listener of listeners) listener();
}
/** The pending request token, 0 when none; changes on every new request. */
export function useInspectorRequest(): number {
  return useSyncExternalStore(subscribe, () => request);
}
/** Clears the pending request and reports whether there was one. */
export function consumeInspectorRequest(): boolean {
  const pending = request !== 0;
  request = 0;
  return pending;
}

const WIDE_QUERY = '(min-width: 1100px)';
// One MediaQueryList per matchMedia implementation, not one per render; tests swap the global.
let wideQuery: { readonly matchMedia: typeof window.matchMedia; readonly list: MediaQueryList } | null = null;
function wideList(): MediaQueryList | null {
  if (typeof window.matchMedia !== 'function') return null;
  if (wideQuery?.matchMedia !== window.matchMedia) wideQuery = { matchMedia: window.matchMedia, list: window.matchMedia(WIDE_QUERY) };
  return wideQuery.list;
}
function subscribeWide(listener: () => void): () => void {
  const list = wideList();
  if (!list) return () => undefined;
  list.addEventListener('change', listener);
  return () => list.removeEventListener('change', listener);
}
/** True at 1100 px and wider, where the inspector is a pane beside the list. A browser without matchMedia counts as wide. */
export function useWideInspector(): boolean {
  return useSyncExternalStore(subscribeWide, () => wideList()?.matches ?? true);
}
