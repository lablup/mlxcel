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
import { minReplaySequence, type WebUiAction } from './reducer';

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
const observationTimeoutMs = 10_000;
const maxBackoffMs = 30_000;
const reconciliationTtlMs = 60_000;
// The contract makes polling a fallback to the event stream, so the 2 s tick may only carry what
// genuinely changes on its own: pending operations and the selected model's runtime. Server
// identity and the model inventory change on events the stream already delivers, and re-reading
// them every tick made a 200-entry library cost seven requests every two seconds, five of them
// catalog pages. They are re-read when the stream cannot be trusted: before it is connected, when
// an event says the inventory moved, on a forced resnapshot, and on this safety net if the stream
// stays silent about a change for longer than it should.
const fullSnapshotSafetyNetMs = 60_000;
// Events whose arrival means the inventory or server identity may have moved under us. The
// resnapshot-forcing kinds (reset, gap, server_restart) are handled separately because they also
// tear down the stream.
const catalogAffectingEventTypes: ReadonlySet<string> = new Set(['snapshot', 'model_revision']);

export class WebUiSynchronizer {
  private readonly client: WebUiApiClient;
  private readonly dispatch: (action: WebUiAction) => void;
  private readonly getSnapshot: () => WebUiSnapshot;
  private readonly clock: SyncClock;
  private readonly visibility: VisibilitySource;
  private readonly random: () => number;
  private timer: ReturnType<typeof globalThis.setTimeout> | null = null;
  private observationTimer: ReturnType<typeof globalThis.setTimeout> | null = null;
  private inflight: AbortController | null = null;
  private eventAbort: AbortController | null = null;
  private stopped = true;
  private failures = 0;
  private generation = 0;
  private needsFullSnapshot = true;
  private lastFullSnapshotAt: number | null = null;
  private readonly unsubscribe: () => void;

  constructor(options: SyncOptions) {
    this.client = options.client;
    this.dispatch = options.dispatch;
    this.getSnapshot = options.getSnapshot;
    this.clock = options.clock ?? browserClock;
    this.visibility = options.visibility ?? browserVisibility;
    this.random = options.random ?? Math.random;
    this.unsubscribe = this.visibility.subscribe(() => this.visibilityChanged());
  }

  start(): void {
    if (!this.stopped) return;
    this.stopped = false;
    this.generation += 1;
    this.needsFullSnapshot = true;
    this.reschedule(0);
  }

  stop(): void {
    this.generation += 1;
    this.stopped = true;
    if (this.timer !== null) this.clock.clearTimeout(this.timer);
    this.timer = null;
    if (this.observationTimer !== null) this.clock.clearTimeout(this.observationTimer);
    this.observationTimer = null;
    this.inflight?.abort();
    this.inflight = null;
    this.eventAbort?.abort();
    this.eventAbort = null;
  }

  // Cancels observation only; inference streams belong to the request owner.
  selectionChanged(): void {
    this.cancelObservation();
    this.reschedule(0);
  }

  /// True when this tick must re-read server identity and the whole inventory rather than only
  /// the pending operations and the selected model's runtime.
  ///
  /// The first condition reads the published state rather than an internal flag on purpose.
  /// Clearing the session and a server-instance change both reset the snapshot, and a
  /// synchronizer that had already taken a full snapshot would otherwise keep taking light ticks
  /// against a view that no longer has a bootstrap or a catalog, leaving the library stuck on
  /// "waiting for an authoritative catalog snapshot" with every control disabled.
  private fullSnapshotDue(now: number): boolean {
    const snapshot = this.getSnapshot();
    if (snapshot.bootstrap === null || snapshot.catalogSequence === null) return true;
    if (this.needsFullSnapshot || this.lastFullSnapshotAt === null) return true;
    if (this.eventAbort === null) return true;
    return now - this.lastFullSnapshotAt >= fullSnapshotSafetyNetMs;
  }

  private cancelObservation(): void {
    this.generation += 1;
    if (this.observationTimer !== null) this.clock.clearTimeout(this.observationTimer);
    this.observationTimer = null;
    this.inflight?.abort();
    this.inflight = null;
    this.eventAbort?.abort();
    this.eventAbort = null;
    if (this.timer !== null) this.clock.clearTimeout(this.timer);
    this.timer = null;
  }

  private visibilityChanged(): void {
    this.cancelObservation();
    // Becoming visible again resumes from an unknown age: the stream was torn down while hidden.
    this.needsFullSnapshot = true;
    if (this.stopped) return;
    this.dispatch({ type: 'connection', connection: 'stale', now: this.clock.now() });
    this.reschedule(0);
  }

  dispose(): void {
    this.stop();
    this.unsubscribe();
  }

  /// The explicit, user-facing refresh. Asking for server state is a request for authoritative
  /// state, so it always re-reads server identity and the whole inventory: a steady-state tick
  /// would not notice a server restart, which is precisely what an operator presses this for.
  async resnapshot(): Promise<void> {
    this.needsFullSnapshot = true;
    await this.refresh();
  }

  async refresh(): Promise<void> {
    if (this.inflight !== null) return;
    const controller = new AbortController();
    this.inflight = controller;
    const generation = this.generation;
    const now = this.clock.now();
    const timeout = this.clock.setTimeout(() => {
      controller.abort();
      if (this.generation === generation) this.dispatch({ type: 'connection', connection: 'stale', error: { code: 'observation_timeout', message: 'Runtime observation timed out; retrying.', retryable: true }, now: this.clock.now() });
    }, observationTimeoutMs);
    this.observationTimer = timeout;
    try {
      if (this.getSnapshot().auth.status === 'signed-out') return;
      const full = this.fullSnapshotDue(now);
      let bootstrap = null;
      let catalogPages: ReadonlyArray<CatalogListResponse> = [];
      if (full) {
        bootstrap = await this.client.bootstrap(controller.signal);
        if (!this.isCurrent(controller, generation) || this.getSnapshot().auth.status === 'signed-out') return;
        this.dispatch({ type: 'login-success', bootstrap, now });
        catalogPages = await this.catalogSnapshot(controller.signal);
        if (!this.isCurrent(controller, generation)) return;
        for (const page of catalogPages) this.dispatch({ type: 'catalog', response: page, now });
      }
      const operationPages = await this.operationSnapshot(controller.signal);
      if (!this.isCurrent(controller, generation)) return;
      const catalog = catalogPages.at(-1);
      const operationsSnapshot = operationPages[0];
      if (catalog !== undefined && operationsSnapshot !== undefined && operationsSnapshot.server_instance_id !== catalog.server_instance_id) {
        this.dispatch({ type: 'connection', connection: 'stale', error: { code: 'snapshot_mismatch', message: 'Operation and catalog snapshots came from different server instances.', retryable: true }, now });
        return;
      }
      const operations = operationPages.flatMap((page) => {
        this.dispatch({ type: 'operations-snapshot', response: page, now });
        return page.items;
      });
      // Only a complete, unfiltered, same-instance catalog can clear selection.
      // Never clear from an individual page or a stale request generation, and never from a tick
      // that did not re-read the catalog at all.
      const selected = this.getSnapshot().selectedModelId;
      const selectedAbsent = full && bootstrap !== null && selected !== null && catalogPages.every((page) => page.server_instance_id === bootstrap.server.server_instance_id) && !catalogPages.some((page) => page.items.some((entry) => entry.identity.id === selected));
      if (selectedAbsent) {
        this.dispatch({ type: 'select-model', modelId: null });
      }
      if (!selectedAbsent) await this.refreshSelectedRuntime(controller.signal, generation);
      if (full) {
        this.needsFullSnapshot = false;
        this.lastFullSnapshotAt = now;
      }
      if (!this.isCurrent(controller, generation)) return;
      const unresolvedExpired = this.reconcilePending(operations, now);
      this.failures = 0;
      if (!unresolvedExpired) this.dispatch({ type: 'connection', connection: 'ready', now });
      if (this.eventAbort === null && !this.visibility.hidden()) this.startEvents();
    } catch (error) {
      if (!controller.signal.aborted && this.generation === generation) {
        this.failures += 1;
        this.dispatch({ type: 'connection', connection: classifyConnectionError(error), error: safeClientError(error), now: this.clock.now() });
      }
    } finally {
      this.clock.clearTimeout(timeout);
      if (this.observationTimer === timeout) this.observationTimer = null;
      if (this.inflight === controller) this.inflight = null;
      if (!this.stopped && this.generation === generation) this.reschedule(this.pollDelay());
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
    const generation = this.generation;
    const cursor = replayCursor(this.getSnapshot());
    void this.client.events({ onEvent: (event) => {
      if (this.generation === generation && !controller.signal.aborted && !this.stopped) this.handleEvent(event);
    }, onRetryAfter: (ms) => {
      if (this.generation === generation && !controller.signal.aborted && !this.stopped) this.reschedule(ms);
    } }, controller.signal, cursor).then(() => {
      if (this.generation !== generation) return;
      if (this.eventAbort === controller) this.eventAbort = null;
      if (!controller.signal.aborted && !this.stopped) this.reschedule(this.backoffDelay());
    }).catch((error: unknown) => {
      if (controller.signal.aborted || this.stopped || this.generation !== generation) return;
      this.failures += 1;
      this.eventAbort = null;
      this.dispatch({ type: 'connection', connection: classifyConnectionError(error), error: safeClientError(error), now: this.clock.now() });
      this.reschedule(this.backoffDelay());
    });
  }

  private handleEvent(event: UiEvent): void {
    this.dispatch({ type: 'event', event, now: this.clock.now() });
    // An event that moves the inventory or server identity is what makes the next tick re-read
    // them. Without this the light tick would never notice a download, removal or rescan.
    if (catalogAffectingEventTypes.has(event.type)) this.needsFullSnapshot = true;
    if ((event.type === 'server_restart' || event.type === 'gap' || event.type === 'reset') && event.payload.resnapshot) {
      this.needsFullSnapshot = true;
      this.eventAbort?.abort();
      this.eventAbort = null;
      void this.refresh();
    }
  }

  private async catalogSnapshot(signal: AbortSignal): Promise<ReadonlyArray<CatalogListResponse>> {
    const pages: CatalogListResponse[] = [];
    let cursor: string | undefined;
    let first: CatalogListResponse | null = null;
    for (;;) {
      const page = await this.client.catalog(cursor === undefined ? {} : { cursor }, signal);
      if (first === null) first = page;
      else assertSameSnapshotPage('catalog', first, page);
      pages.push(page);
      if (page.pagination.next_cursor === null) return pages;
      cursor = page.pagination.next_cursor;
    }
  }

  private async operationSnapshot(signal: AbortSignal): Promise<ReadonlyArray<OperationsListResponse>> {
    const pages: OperationsListResponse[] = [];
    let cursor: string | undefined;
    let first: OperationsListResponse | null = null;
    for (;;) {
      const page = await this.client.operationsPage(cursor === undefined ? {} : { cursor }, signal);
      if (first === null) first = page;
      else assertSameSnapshotPage('operations', first, page);
      pages.push(page);
      if (page.pagination.next_cursor === null) return pages;
      cursor = page.pagination.next_cursor;
    }
  }

  private async refreshSelectedRuntime(signal: AbortSignal, generation: number): Promise<void> {
    const modelId = this.getSnapshot().selectedModelId;
    if (modelId === null) return;
    let runtime;
    try {
      runtime = await this.client.runtime(modelId, signal);
    } catch (error) {
      // Removal can commit between catalog and runtime reads. The next complete
      // catalog reconciles selection; this is stale observation, not offline.
      if (typeof error === 'object' && error !== null && 'status' in error && error.status === 404) throw new SnapshotConsistencyError('Selected model changed during observation; refreshing the catalog.');
      throw error;
    }
    if (signal.aborted || this.generation !== generation) return;
    if (this.getSnapshot().selectedModelId !== modelId) return;
    this.dispatch({ type: 'runtime', runtime, sequence: runtime.snapshot_sequence, now: this.clock.now() });
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
    if (this.stopped || this.visibility.hidden()) return;
    if (this.timer !== null) this.clock.clearTimeout(this.timer);
    this.timer = this.clock.setTimeout(() => {
      this.timer = null;
      void this.refresh();
    }, milliseconds);
  }

  private pollDelay(): number {
    return visiblePollMs;
  }

  private backoffDelay(): number {
    const base = Math.min(maxBackoffMs, 500 * 2 ** Math.min(6, this.failures));
    return Math.round(base / 2 + this.random() * (base / 2));
  }

  private isCurrent(controller: AbortController, generation: number): boolean {
    return this.inflight === controller && this.generation === generation && !controller.signal.aborted;
  }
}

function replayCursor(snapshot: WebUiSnapshot): EventReplayCursor | undefined {
  const afterSequence = minReplaySequence(snapshot.resourceFences);
  if (snapshot.serverInstanceId !== null && afterSequence !== null) {
    return { lastEventId: null, serverInstanceId: snapshot.serverInstanceId, afterSequence };
  }
  if (snapshot.lastEventId !== null) return { lastEventId: snapshot.lastEventId, serverInstanceId: null, afterSequence: null };
  return undefined;
}

function classifyConnectionError(error: unknown): WebUiSnapshot['connection'] {
  if (error instanceof SnapshotConsistencyError) return 'stale';
  const status = typeof error === 'object' && error !== null && 'status' in error && typeof error.status === 'number' ? error.status : null;
  if (status === 401 || (error instanceof Error && /401|unauthorized/i.test(error.message))) return 'unauthorized';
  if (status === 403 || (error instanceof Error && /403|forbidden/i.test(error.message))) return 'forbidden';
  if (error instanceof DOMException && error.name === 'AbortError') return 'stale';
  return 'offline';
}

function safeClientError(error: unknown) {
  if (error instanceof SnapshotConsistencyError) return { code: error.code, message: error.message, retryable: true };
  const message = error instanceof Error ? error.message.replace(/Bearer\s+\S+/gi, 'Bearer [redacted]') : 'Unknown WebUI client error';
  return { code: 'sync_error', message, retryable: true };
}

interface SnapshotPage {
  readonly server_instance_id: string;
  readonly snapshot_sequence: number;
}

class SnapshotConsistencyError extends Error {
  readonly code = 'snapshot_mismatch';
  constructor(message: string) {
    super(message);
    this.name = 'SnapshotConsistencyError';
  }
}

function assertSameSnapshotPage(kind: 'catalog' | 'operations', first: SnapshotPage, page: SnapshotPage): void {
  if (page.server_instance_id !== first.server_instance_id) throw new SnapshotConsistencyError(`${kind} pagination crossed a server restart; refresh the authoritative snapshot.`);
  if (page.snapshot_sequence !== first.snapshot_sequence) throw new SnapshotConsistencyError(`${kind} pagination crossed snapshot sequence boundaries; refresh the authoritative snapshot.`);
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
