// Copyright 2026 Lablup Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import type { EventReplayCursor, WebUiApiClient } from '../api/client';
import type { CatalogListResponse, OperationsListResponse, PendingReconciliation, UiEvent, WebUiSnapshot } from '../api/types';
import type { WebUiAction } from './reducer';

export interface SyncClock {
  readonly setTimeout: (callback: () => void, milliseconds: number) => ReturnType<typeof globalThis.setTimeout>;
  readonly clearTimeout: (handle: ReturnType<typeof globalThis.setTimeout>) => void;
  readonly now: () => number;
}

export interface VisibilitySource {
  readonly hidden: () => boolean;
  readonly subscribe: (callback: () => void) => () => void;
}

export interface SyncOptions {
  readonly client: WebUiApiClient;
  readonly dispatch: (action: WebUiAction) => void;
  readonly getSnapshot: () => WebUiSnapshot;
  readonly clock?: SyncClock;
  readonly visibility?: VisibilitySource;
  readonly random?: () => number;
}

const visiblePollMs = 2_000;
const hiddenPollMs = 30_000;
const maxBackoffMs = 30_000;
const reconciliationTtlMs = 60_000;

export class WebUiSynchronizer {
  private readonly client: WebUiApiClient;
  private readonly dispatch: (action: WebUiAction) => void;
  private readonly getSnapshot: () => WebUiSnapshot;
  private readonly clock: SyncClock;
  private readonly visibility: VisibilitySource;
  private readonly random: () => number;
  private timer: ReturnType<typeof globalThis.setTimeout> | null = null;
  private inflight: AbortController | null = null;
  private eventAbort: AbortController | null = null;
  private stopped = true;
  private failures = 0;
  private readonly unsubscribe: () => void;

  constructor(options: SyncOptions) {
    this.client = options.client;
    this.dispatch = options.dispatch;
    this.getSnapshot = options.getSnapshot;
    this.clock = options.clock ?? browserClock;
    this.visibility = options.visibility ?? browserVisibility;
    this.random = options.random ?? Math.random;
    this.unsubscribe = this.visibility.subscribe(() => this.reschedule(0));
  }

  start(): void {
    if (!this.stopped) return;
    this.stopped = false;
    this.reschedule(0);
  }

  stop(): void {
    this.stopped = true;
    if (this.timer !== null) this.clock.clearTimeout(this.timer);
    this.timer = null;
    this.inflight?.abort();
    this.inflight = null;
    this.eventAbort?.abort();
    this.eventAbort = null;
  }

  dispose(): void {
    this.stop();
    this.unsubscribe();
  }

  async refresh(): Promise<void> {
    if (this.inflight !== null) return;
    const controller = new AbortController();
    this.inflight = controller;
    const now = this.clock.now();
    try {
      if (this.getSnapshot().auth.status === 'signed-out') return;
      const bootstrap = await this.client.bootstrap(controller.signal);
      if (this.getSnapshot().auth.status === 'signed-out') return;
      this.dispatch({ type: 'login-success', bootstrap, now });
      const catalogPages = await this.catalogSnapshot(controller.signal);
      for (const page of catalogPages) this.dispatch({ type: 'catalog', response: page, now });
      const catalog = catalogPages.at(-1);
      const operationPages = await this.operationSnapshot(controller.signal);
      const firstOperationPage = operationPages[0];
      if (catalog !== undefined && firstOperationPage !== undefined && firstOperationPage.server_instance_id !== catalog.server_instance_id) {
        this.dispatch({ type: 'connection', connection: 'stale', error: { code: 'snapshot_mismatch', message: 'Operation and catalog snapshots came from different server instances.', retryable: true }, now });
        return;
      }
      const operations = operationPages.flatMap((page) => {
        for (const operation of page.items) this.dispatch({ type: 'operation', operation, sequence: page.snapshot_sequence, now });
        return page.items;
      });
      await this.refreshSelectedRuntime(controller.signal, now);
      const unresolvedExpired = this.reconcilePending(operations, now);
      this.failures = 0;
      if (!unresolvedExpired) this.dispatch({ type: 'connection', connection: 'ready', now });
      if (this.eventAbort === null) this.startEvents();
    } catch (error) {
      if (!controller.signal.aborted) {
        this.failures += 1;
        this.dispatch({ type: 'connection', connection: classifyConnectionError(error), error: safeClientError(error), now: this.clock.now() });
      }
    } finally {
      if (this.inflight === controller) this.inflight = null;
      if (!this.stopped) this.reschedule(this.pollDelay());
    }
  }

  noteUnknownPost(item: PendingReconciliation): void {
    this.dispatch({ type: 'pending', item });
    this.reschedule(0);
  }

  private startEvents(): void {
    this.eventAbort?.abort();
    const controller = new AbortController();
    this.eventAbort = controller;
    const cursor = replayCursor(this.getSnapshot());
    void this.client.events({ onEvent: (event) => this.handleEvent(event), onRetryAfter: (ms) => this.reschedule(ms) }, controller.signal, cursor).then(() => {
      if (this.eventAbort === controller) this.eventAbort = null;
      if (!controller.signal.aborted && !this.stopped) this.reschedule(this.backoffDelay());
    }).catch((error: unknown) => {
      if (controller.signal.aborted || this.stopped) return;
      this.failures += 1;
      this.eventAbort = null;
      this.dispatch({ type: 'connection', connection: classifyConnectionError(error), error: safeClientError(error), now: this.clock.now() });
      this.reschedule(this.backoffDelay());
    });
  }

  private handleEvent(event: UiEvent): void {
    this.dispatch({ type: 'event', event, now: this.clock.now() });
    if ((event.type === 'server_restart' || event.type === 'gap' || event.type === 'reset') && event.payload.resnapshot) {
      this.eventAbort?.abort();
      this.eventAbort = null;
      void this.refresh();
    }
  }

  private async catalogSnapshot(signal: AbortSignal): Promise<ReadonlyArray<CatalogListResponse>> {
    const pages: CatalogListResponse[] = [];
    let cursor: string | undefined;
    for (;;) {
      const page = await this.client.catalog(cursor === undefined ? {} : { cursor }, signal);
      pages.push(page);
      if (page.pagination.next_cursor === null) return pages;
      cursor = page.pagination.next_cursor;
    }
  }

  private async operationSnapshot(signal: AbortSignal): Promise<ReadonlyArray<OperationsListResponse>> {
    const pages: OperationsListResponse[] = [];
    let cursor: string | undefined;
    for (;;) {
      const page = await this.client.operationsPage(cursor === undefined ? {} : { cursor }, signal);
      pages.push(page);
      if (page.pagination.next_cursor === null) return pages;
      cursor = page.pagination.next_cursor;
    }
  }

  private async refreshSelectedRuntime(signal: AbortSignal, now: number): Promise<void> {
    const modelId = this.getSnapshot().selectedModelId;
    if (modelId === null) return;
    const runtime = await this.client.runtime(modelId, signal);
    this.dispatch({ type: 'runtime', runtime, sequence: null, now });
  }

  private reconcilePending(operations: ReadonlyArray<{ readonly operation_id: string }>, now: number): boolean {
    let unresolvedExpired = false;
    const operationIds = new Set(operations.map((operation) => operation.operation_id));
    for (const pending of this.getSnapshot().pendingReconciliations.values()) {
      if (pending.operationId !== null && operationIds.has(pending.operationId)) {
        this.dispatch({ type: 'reconciled', idempotencyKey: pending.idempotencyKey });
      } else if (now - pending.createdAt >= reconciliationTtlMs) {
        this.dispatch({ type: 'reconciled', idempotencyKey: pending.idempotencyKey });
        this.dispatch({ type: 'connection', connection: 'stale', error: { code: 'unknown_post_unresolved', message: 'A previous control request could not be matched to an operation after reconciliation.', retryable: true }, now });
        unresolvedExpired = true;
      }
    }
    return unresolvedExpired;
  }

  private reschedule(milliseconds: number): void {
    if (this.stopped) return;
    if (this.timer !== null) this.clock.clearTimeout(this.timer);
    this.timer = this.clock.setTimeout(() => {
      this.timer = null;
      void this.refresh();
    }, milliseconds);
  }

  private pollDelay(): number {
    return this.visibility.hidden() ? hiddenPollMs : visiblePollMs;
  }

  private backoffDelay(): number {
    const base = Math.min(maxBackoffMs, 500 * 2 ** Math.min(6, this.failures));
    return Math.round(base / 2 + this.random() * (base / 2));
  }
}

function replayCursor(snapshot: WebUiSnapshot): EventReplayCursor | undefined {
  if (snapshot.serverInstanceId !== null && snapshot.lastSequence !== null) {
    return { lastEventId: null, serverInstanceId: snapshot.serverInstanceId, afterSequence: snapshot.lastSequence };
  }
  if (snapshot.lastEventId !== null) return { lastEventId: snapshot.lastEventId, serverInstanceId: null, afterSequence: null };
  return undefined;
}

function classifyConnectionError(error: unknown): WebUiSnapshot['connection'] {
  const status = typeof error === 'object' && error !== null && 'status' in error && typeof error.status === 'number' ? error.status : null;
  if (status === 401 || (error instanceof Error && /401|unauthorized/i.test(error.message))) return 'unauthorized';
  if (status === 403 || (error instanceof Error && /403|forbidden/i.test(error.message))) return 'forbidden';
  if (error instanceof DOMException && error.name === 'AbortError') return 'stale';
  return 'offline';
}

function safeClientError(error: unknown) {
  const message = error instanceof Error ? error.message.replace(/Bearer\s+\S+/gi, 'Bearer [redacted]') : 'Unknown WebUI client error';
  return { code: 'sync_error', message, retryable: true };
}

const browserClock: SyncClock = {
  setTimeout: (callback, milliseconds) => globalThis.setTimeout(callback, milliseconds),
  clearTimeout: (handle) => globalThis.clearTimeout(handle),
  now: () => Date.now(),
};

const browserVisibility: VisibilitySource = {
  hidden: () => typeof document !== 'undefined' && document.hidden,
  subscribe: (callback) => {
    if (typeof document === 'undefined') return () => undefined;
    document.addEventListener('visibilitychange', callback);
    return () => document.removeEventListener('visibilitychange', callback);
  },
};
