// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import type { CatalogEntry, Operation, TaskKind, WebUiSnapshot } from '../../api/types';
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
export type InventoryFilter = { query: string; source: string; task: string; status: string };
/** The table columns that sort; every sort keeps the lifecycle pin first. */
export type SortColumn = 'name' | 'size' | 'quantization' | 'state';
export type InventorySort = { column: SortColumn; direction: 'asc' | 'desc' };
export const DEFAULT_SORT: InventorySort = { column: 'name', direction: 'asc' };
const LIFECYCLE_ORDER: Readonly<Record<CatalogEntry['lifecycle']['state'], number>> = {
  ready: 0,
  loading: 1,
  draining: 2,
  unloading: 3,
  failed: 4,
  unloaded: 5,
};
// Numeric collation keeps `qwen3-2b` ahead of `qwen3-14b`, the order a person reads.
const collator = new Intl.Collator(undefined, { numeric: true });
/** The lifecycle pin: ready models first, then models in transition, then the rest. */
export function lifecycleGroup(entry: CatalogEntry): number {
  const state = entry.lifecycle.state;
  return state === 'ready' ? 0 : state === 'loading' || state === 'draining' || state === 'unloading' ? 1 : 2;
}
/** What the Quantization column shows: the declared quantization, else the weight dtype. */
export function quantizationValue(entry: CatalogEntry): string | null {
  return entry.metadata.quantization ?? entry.metadata.dtype ?? null;
}
function columnValue(entry: CatalogEntry, column: SortColumn): string | number | null {
  if (column === 'size') return entry.metadata.disk_bytes;
  if (column === 'quantization') return quantizationValue(entry);
  if (column === 'state') return LIFECYCLE_ORDER[entry.lifecycle.state];
  return entry.identity.display_name;
}
/**
 * The one ordering of the library: the lifecycle pin, then the chosen column in the chosen
 * direction with unknown values last either way, then name and opaque id so equal rows keep
 * a stable place. The table sorts its page with this same comparator, so the page order and
 * the pagination order never disagree.
 */
export function compareEntries(sort: InventorySort): (a: CatalogEntry, b: CatalogEntry) => number {
  return (a, b) => {
    const pinned = lifecycleGroup(a) - lifecycleGroup(b);
    if (pinned !== 0) return pinned;
    const left = columnValue(a, sort.column);
    const right = columnValue(b, sort.column);
    if (left === null || right === null) {
      if (left !== right) return left === null ? 1 : -1;
    } else {
      const order = typeof left === 'number' && typeof right === 'number' ? left - right : collator.compare(String(left), String(right));
      if (order !== 0) return sort.direction === 'asc' ? order : -order;
    }
    return collator.compare(a.identity.display_name, b.identity.display_name) || a.identity.id.localeCompare(b.identity.id);
  };
}
export function inventory(
  entries: ReadonlyArray<CatalogEntry>,
  filter: InventoryFilter,
  sort: InventorySort = DEFAULT_SORT,
): CatalogEntry[] {
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
    .sort(compareEntries(sort));
}
/** Distinct capability tasks, in the order the entry declares its output tasks; any other task follows. */
export function entryTasks(entry: CatalogEntry): TaskKind[] {
  const tasks = [...new Set(entry.capabilities.map((cap) => cap.task))];
  const rank = (task: TaskKind): number => {
    const index = entry.metadata.output_tasks.indexOf(task);
    return index === -1 ? entry.metadata.output_tasks.length : index;
  };
  return tasks.map((task, index) => ({ task, index })).sort((a, b) => rank(a.task) - rank(b.task) || a.index - b.index).map((item) => item.task);
}
/** The newest model_load for this entry failed on capacity, so a load with an explicit eviction is the recovery. */
export function capacityFailed(state: WebUiSnapshot, id: string): boolean {
  const latest = [...state.operations.values()]
    .filter((op) => op.kind === 'model_load' && op.target.target_kind === 'model' && op.target.model_id === id)
    .sort((a, b) => b.created_at.localeCompare(a.created_at))[0];
  return latest?.state === 'failed' && latest.error?.code === 'conflict';
}
/** Downloads the library still has to show: in flight, failed or cancelled, newest first. */
export function visibleDownloads(state: WebUiSnapshot, limit = 5): Operation[] {
  return [...state.operations.values()]
    .filter((op) => op.kind === 'download' && op.target.target_kind === 'download' && op.state !== 'succeeded')
    .sort((a, b) => Number(terminal(a)) - Number(terminal(b)) || b.created_at.localeCompare(a.created_at))
    .slice(0, limit);
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
