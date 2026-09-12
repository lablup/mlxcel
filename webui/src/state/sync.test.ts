import { describe, expect, it } from 'vitest';
import bootstrapFixture from '../../../tests/fixtures/webui/examples/bootstrap.model-free.json';
import catalogFixture from '../../../tests/fixtures/webui/examples/catalog.page.json';
import operationsFixture from '../../../tests/fixtures/webui/examples/operations.list.json';
import { WebUiApiClient } from '../api/client';
import { validateBootstrap } from '../api/validation';
import type { PendingReconciliation } from '../api/types';
import { initialSnapshot, reduceWebUiSnapshot } from './reducer';
import { WebUiSynchronizer, type SyncClock, type VisibilitySource } from './sync';

class FakeClock implements SyncClock {
  nowValue = 0;
  readonly delays: number[] = [];
  private next = 1;
  private readonly timers = new Map<number, () => void>();

  setTimeout(callback: () => void, milliseconds: number): ReturnType<typeof globalThis.setTimeout> {
    this.delays.push(milliseconds);
    const id = this.next;
    this.next += 1;
    this.timers.set(id, callback);
    return id as unknown as ReturnType<typeof globalThis.setTimeout>;
  }

  clearTimeout(handle: ReturnType<typeof globalThis.setTimeout>): void {
    this.timers.delete(Number(handle));
  }

  now(): number {
    return this.nowValue;
  }

  runOne(): void {
    const next = this.timers.entries().next().value as [number, () => void] | undefined;
    if (next === undefined) throw new Error('No fake timer is scheduled.');
    this.timers.delete(next[0]);
    next[1]();
  }

  count(): number {
    return this.timers.size;
  }
}

function visibleSource(hiddenValue: () => boolean): VisibilitySource {
  return { hidden: hiddenValue, subscribe: () => () => undefined };
}

function streamDone(): ReadableStream<Uint8Array> {
  return new ReadableStream({
    start(controller) {
      controller.enqueue(new TextEncoder().encode('data: [DONE]\n\n'));
      controller.close();
    },
  });
}

const bootstrap = validateBootstrap(bootstrapFixture);
const operations = stripSchemaName(operationsFixture) as typeof operationsFixture;

function stripSchemaName(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(stripSchemaName);
  if (typeof value !== 'object' || value === null) return value;
  const result: Record<string, unknown> = {};
  for (const [key, entry] of Object.entries(value)) if (key !== '$schemaName') result[key] = stripSchemaName(entry);
  return result;
}

function makeImmediateFetch(methods?: string[]): typeof fetch {
  return async (input, init) => {
    methods?.push(`${init?.method ?? 'GET'} ${String(input)}`);
    if (String(input).endsWith('/events')) return new Response(streamDone(), { status: 200 });
    if (String(input).endsWith('/bootstrap')) return new Response(JSON.stringify(bootstrap));
    if (String(input).endsWith('/catalog')) return new Response(JSON.stringify(catalogFixture));
    return new Response(JSON.stringify(operations));
  };
}

describe('WebUI synchronizer', () => {
  it('keeps one polling request in flight and returns timer count to baseline on dispose', async () => {
    const clock = new FakeClock();
    let active = 0;
    let maxActive = 0;
    let release: () => void = () => undefined;
    const fetchImpl: typeof fetch = async (input) => {
      if (String(input).endsWith('/events')) return new Response(streamDone(), { status: 200 });
      active += 1;
      maxActive = Math.max(maxActive, active);
      await new Promise<void>((resolve) => { release = resolve; });
      active -= 1;
      if (String(input).endsWith('/bootstrap')) return new Response(JSON.stringify(bootstrap));
      if (String(input).endsWith('/catalog')) return new Response(JSON.stringify(catalogFixture));
      return new Response(JSON.stringify(operations));
    };
    let snapshot = reduceWebUiSnapshot(initialSnapshot(), { type: 'login-success', bootstrap, now: 0 });
    const sync = new WebUiSynchronizer({ client: new WebUiApiClient({ fetchImpl }), clock, visibility: visibleSource(() => false), getSnapshot: () => snapshot, dispatch: (action) => { snapshot = reduceWebUiSnapshot(snapshot, action); } });
    sync.start();
    clock.runOne();
    expect(clock.count()).toBe(0);
    await Promise.resolve();
    expect(maxActive).toBe(1);
    release();
    await Promise.resolve();
    await Promise.resolve();
    await Promise.resolve();
    sync.dispose();
    expect(clock.count()).toBe(0);
  });

  it('reconciles unknown POST outcomes by polling without replaying the POST', async () => {
    const clock = new FakeClock();
    const methods: string[] = [];
    let snapshot = reduceWebUiSnapshot(initialSnapshot(), { type: 'login-success', bootstrap, now: 0 });
    const sync = new WebUiSynchronizer({ client: new WebUiApiClient({ fetchImpl: makeImmediateFetch(methods) }), clock, visibility: visibleSource(() => true), getSnapshot: () => snapshot, dispatch: (action) => { snapshot = reduceWebUiSnapshot(snapshot, action); } });
    const pending: PendingReconciliation = { kind: 'model-action', idempotencyKey: 'idem-unknown', operationId: operations.items[0]?.operation_id ?? null, modelId: 'mdl_a', createdAt: 1 };
    sync.noteUnknownPost(pending);
    await sync.refresh();
    expect(methods.every((entry) => !entry.startsWith('POST /ui-api/v1/model-actions'))).toBe(true);
    expect(snapshot.pendingReconciliations.size).toBe(0);
    sync.dispose();
  });



  it('expires unmatched unknown POST outcomes explicitly instead of silently clearing them', async () => {
    const clock = new FakeClock();
    clock.nowValue = 61_000;
    let snapshot = reduceWebUiSnapshot(initialSnapshot(), { type: 'login-success', bootstrap, now: 0 });
    const sync = new WebUiSynchronizer({ client: new WebUiApiClient({ fetchImpl: makeImmediateFetch() }), clock, visibility: visibleSource(() => false), getSnapshot: () => snapshot, dispatch: (action) => { snapshot = reduceWebUiSnapshot(snapshot, action); } });
    sync.noteUnknownPost({ kind: 'model-action', idempotencyKey: 'idem-lost', operationId: null, modelId: 'mdl_a', createdAt: 0 });
    await sync.refresh();
    expect(snapshot.pendingReconciliations.size).toBe(0);
    expect(snapshot.error?.code).toBe('unknown_post_unresolved');
    sync.dispose();
  });

  it('uses the hidden-tab polling interval after a successful refresh', async () => {
    const clock = new FakeClock();
    let snapshot = reduceWebUiSnapshot(initialSnapshot(), { type: 'login-success', bootstrap, now: 0 });
    const sync = new WebUiSynchronizer({ client: new WebUiApiClient({ fetchImpl: makeImmediateFetch() }), clock, visibility: visibleSource(() => true), getSnapshot: () => snapshot, dispatch: (action) => { snapshot = reduceWebUiSnapshot(snapshot, action); } });
    sync.start();
    clock.runOne();
    await new Promise((resolve) => globalThis.setTimeout(resolve, 0));
    expect(clock.delays).toContain(30_000);
    sync.dispose();
  });
});
