// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import type { CatalogEntry, Operation, WebUiSnapshot } from '../../api/types';
import { WebUiHttpError } from '../../api/client';
import { t, type Locale } from '../../i18n/catalog';

export const terminal = (operation: Operation): boolean =>
  ['succeeded', 'failed', 'cancelled'].includes(operation.state);
export const current = (state: WebUiSnapshot): boolean =>
  state.auth.status === 'authenticated' &&
  state.bootstrap !== null &&
  state.catalogSequence !== null &&
  ['ready', 'streaming', 'polling'].includes(state.connection);
export function allowed(state: WebUiSnapshot, action: string): boolean {
  return current(state) && state.bootstrap?.actions[action]?.state === 'enabled';
}
export function modelPending(state: WebUiSnapshot, id: string): boolean {
  return (
    [...state.operations.values()].some(
      (op) =>
        !terminal(op) &&
        op.target.target_kind === 'model' &&
        (op.target.model_id === id || op.target.eviction_target_id === id),
    ) || [...state.pendingReconciliations.values()].some((item) => item.modelId === id)
  );
}
export function canLoad(state: WebUiSnapshot, entry: CatalogEntry): boolean {
  return (
    allowed(state, 'load') &&
    entry.supported &&
    entry.complete &&
    entry.metadata.support.runnable_on_backend &&
    !entry.lifecycle.busy &&
    ['unloaded', 'failed'].includes(entry.lifecycle.state) &&
    !modelPending(state, entry.identity.id)
  );
}
export function canUnload(state: WebUiSnapshot, entry: CatalogEntry): boolean {
  return allowed(state, 'unload') && entry.lifecycle.state === 'ready' && !modelPending(state, entry.identity.id);
}
export function canDelete(state: WebUiSnapshot, entry: CatalogEntry): boolean {
  return (
    allowed(state, 'cache_delete') &&
    entry.identity.source === 'cache' &&
    entry.removable &&
    entry.removal.eligible &&
    !entry.lifecycle.busy &&
    entry.lifecycle.active_requests === 0 &&
    ['unloaded', 'failed'].includes(entry.lifecycle.state) &&
    !modelPending(state, entry.identity.id)
  );
}
export function canChat(state: WebUiSnapshot, entry: CatalogEntry): boolean {
  return (
    current(state) &&
    entry.lifecycle.state === 'ready' &&
    entry.capabilities.some((cap) => cap.task === 'chat' && cap.phase === 'provider_ready' && cap.available)
  );
}
export function evictionCandidates(state: WebUiSnapshot, id: string): CatalogEntry[] {
  return state.catalog.filter(
    (entry) =>
      entry.identity.id !== id &&
      canUnload(state, entry) &&
      !entry.lifecycle.busy &&
      entry.lifecycle.active_requests === 0 &&
      entry.lifecycle.draining_requests === 0,
  );
}
export function bytes(value: number | null, locale: Locale): string {
  if (value === null) return t(locale, 'models.library.unknown');
  if (value === 0) return '0 B';
  const unit = Math.min(4, Math.floor(Math.log(value) / Math.log(1024)));
  return `${(value / 1024 ** unit).toLocaleString(locale, { maximumFractionDigits: 1 })} ${['B', 'KiB', 'MiB', 'GiB', 'TiB'][unit]}`;
}
export function errorMessage(error: unknown, locale: Locale): string {
  if (error instanceof WebUiHttpError) {
    if (error.envelope?.error.code === 'stale_revision') return t(locale, 'models.library.stale');
    return error.message;
  }
  return t(locale, 'models.library.pending');
}
export type InventoryFilter = { query: string; source: string; task: string; status: string; sort: string };
export function inventory(entries: ReadonlyArray<CatalogEntry>, filter: InventoryFilter): CatalogEntry[] {
  const query = filter.query.trim().toLocaleLowerCase();
  return entries
    .filter(
      (entry) =>
        (!query ||
          [entry.identity.display_name, entry.identity.inference_id, entry.metadata.architecture ?? ''].some((text) =>
            text.toLocaleLowerCase().includes(query),
          )) &&
        (!filter.source || entry.identity.source === filter.source) &&
        (!filter.task || entry.capabilities.some((cap) => cap.task === filter.task)) &&
        (!filter.status || entry.lifecycle.state === filter.status),
    )
    .sort((a, b) => {
      const key = (entry: CatalogEntry): string =>
        filter.sort === 'status'
          ? entry.lifecycle.state
          : filter.sort === 'source'
            ? entry.identity.source
            : filter.sort === 'task'
              ? entry.capabilities.map((cap) => cap.task).join(',')
              : entry.identity.display_name;
      return key(a).localeCompare(key(b)) || a.identity.id.localeCompare(b.identity.id);
    });
}
export function validRepo(repo: string): boolean {
  const parts = repo.split('/');
  return parts.length === 2 && parts.every((part) => part.length <= 96 && /^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(part));
}
export function validRevision(revision: string): boolean {
  return (
    revision.length === 0 ||
    (revision.length <= 128 &&
      revision.split('/').every((part) => part !== '.' && part !== '..' && /^[A-Za-z0-9._~+-]+$/.test(part)))
  );
}
