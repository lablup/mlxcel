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

import type { WebUiApiClient } from '../api/client';
import type { PendingReconciliation, WebUiSnapshot } from '../api/types';
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
    this.startEvents();
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
      const bootstrap = await this.client.bootstrap(controller.signal);
      this.dispatch({ type: 'login-success', bootstrap, now });
      const catalog = await this.client.catalog({}, controller.signal);
      this.dispatch({ type: 'catalog', response: catalog, now });
      const operations = await this.client.operations(controller.signal);
      for (const operation of operations) this.dispatch({ type: 'operation', operation, sequence: catalog.snapshot_sequence, now });
      for (const pending of this.getSnapshot().pendingReconciliations.values()) this.dispatch({ type: 'reconciled', idempotencyKey: pending.idempotencyKey });
      this.failures = 0;
      this.dispatch({ type: 'connection', connection: 'ready', now });
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
    this.client.events({ onEvent: (event) => this.dispatch({ type: 'event', event, now: this.clock.now() }), onRetryAfter: (ms) => this.reschedule(ms) }, controller.signal).catch((error: unknown) => {
      if (controller.signal.aborted || this.stopped) return;
      this.failures += 1;
      this.dispatch({ type: 'connection', connection: classifyConnectionError(error), error: safeClientError(error), now: this.clock.now() });
      this.reschedule(this.backoffDelay());
    });
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
