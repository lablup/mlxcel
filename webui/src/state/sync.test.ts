import { describe, expect, it } from 'vitest';
import bootstrapFixture from '../../../tests/fixtures/webui/examples/bootstrap.model-free.json';
import catalogFixture from '../../../tests/fixtures/webui/examples/catalog.page.json';
import operationsFixture from '../../../tests/fixtures/webui/examples/operations.list.json';
import runtimeFixture from '../../../tests/fixtures/webui/examples/runtime.snapshot.json';
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
    const url = String(input);
    if (url.includes('/events')) return new Response(streamDone(), { status: 200 });
    if (url.endsWith('/bootstrap')) return new Response(JSON.stringify(bootstrap));
    if (url.includes('/catalog')) return new Response(JSON.stringify(catalogFixture));
    if (url.includes('/runtime')) return new Response(JSON.stringify(runtimeFixture));
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
      if (String(input).includes('/events')) return new Response(streamDone(), { status: 200 });
      active += 1;
      maxActive = Math.max(maxActive, active);
      await new Promise<void>((resolve) => { release = resolve; });
      active -= 1;
      if (String(input).endsWith('/bootstrap')) return new Response(JSON.stringify(bootstrap));
      if (String(input).includes('/catalog')) return new Response(JSON.stringify(catalogFixture));
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

  it('publishes lastSuccessfulAt from the synchronizer clock only on successful data refreshes', async () => {
    const clock = new FakeClock();
    clock.nowValue = 1_234;
    let failBootstrap = false;
    const fetchImpl: typeof fetch = async (input) => {
      const url = String(input);
      if (url.includes('/events')) return new Response(streamDone(), { status: 200 });
      if (url.endsWith('/bootstrap')) {
        if (failBootstrap) return new Response(JSON.stringify({ error: { code: 'offline', message: 'offline', retryable: true }, request_id: 'req_offline' }), { status: 503 });
        return new Response(JSON.stringify(bootstrap));
      }
      if (url.includes('/catalog')) return new Response(JSON.stringify(catalogFixture));
      if (url.includes('/runtime')) return new Response(JSON.stringify(runtimeFixture));
      return new Response(JSON.stringify(operations));
    };
    let snapshot = reduceWebUiSnapshot(initialSnapshot(), { type: 'login-success', bootstrap, now: 0 });
    const sync = new WebUiSynchronizer({ client: new WebUiApiClient({ fetchImpl }), clock, visibility: visibleSource(() => false), getSnapshot: () => snapshot, dispatch: (action) => { snapshot = reduceWebUiSnapshot(snapshot, action); } });
    await sync.refresh();
    expect(snapshot.lastSuccessfulAt).toBe(1_234);
    clock.nowValue = 2_000;
    failBootstrap = true;
    await sync.refresh();
    expect(snapshot.lastUpdatedAt).toBe(2_000);
    expect(snapshot.lastSuccessfulAt).toBe(1_234);
    sync.dispose();
  });

  it('starts SSE with the minimum authoritative resource fence as paired replay query', async () => {
    const methods: string[] = [];
    let snapshot = reduceWebUiSnapshot(initialSnapshot(), { type: 'login-success', bootstrap, now: 0 });
    const sync = new WebUiSynchronizer({ client: new WebUiApiClient({ fetchImpl: makeImmediateFetch(methods) }), getSnapshot: () => snapshot, dispatch: (action) => { snapshot = reduceWebUiSnapshot(snapshot, action); } });
    await sync.refresh();
    expect(methods).toContain('GET /ui-api/v1/events?server_instance_id=srv_20260912_a&after_sequence=42');
    sync.dispose();
  });

  it('includes empty operation snapshots in the minimum replay fence', async () => {
    const methods: string[] = [];
    const emptyOperations = { ...structuredClone(operations), items: [], snapshot_sequence: 41 };
    const fetchImpl: typeof fetch = async (input, init) => {
      methods.push(`${init?.method ?? 'GET'} ${String(input)}`);
      const url = String(input);
      if (url.includes('/events')) return new Response(streamDone(), { status: 200 });
      if (url.endsWith('/bootstrap')) return new Response(JSON.stringify(bootstrap));
      if (url.includes('/catalog')) return new Response(JSON.stringify(catalogFixture));
      return new Response(JSON.stringify(emptyOperations));
    };
    let snapshot = reduceWebUiSnapshot(initialSnapshot(), { type: 'login-success', bootstrap, now: 0 });
    const sync = new WebUiSynchronizer({ client: new WebUiApiClient({ fetchImpl }), getSnapshot: () => snapshot, dispatch: (action) => { snapshot = reduceWebUiSnapshot(snapshot, action); } });
    await sync.refresh();
    expect(methods).toContain('GET /ui-api/v1/events?server_instance_id=srv_20260912_a&after_sequence=41');
    sync.dispose();
  });

  it('records selected runtime snapshot sequence as a resource fence', async () => {
    let snapshot = reduceWebUiSnapshot(initialSnapshot(), { type: 'login-success', bootstrap, now: 0 });
    snapshot = reduceWebUiSnapshot(snapshot, { type: 'select-model', modelId: runtimeFixture.model_id });
    const sync = new WebUiSynchronizer({ client: new WebUiApiClient({ fetchImpl: makeImmediateFetch() }), getSnapshot: () => snapshot, dispatch: (action) => { snapshot = reduceWebUiSnapshot(snapshot, action); } });
    await sync.refresh();
    expect(snapshot.resourceFences.runtimes.get(runtimeFixture.model_id)).toBe(runtimeFixture.snapshot_sequence);
    sync.dispose();
  });

  it('rejects operation pagination that crosses a server restart', async () => {
    const firstOperations = { ...structuredClone(operations), pagination: { limit: 50, next_cursor: 'ops_1', total_known: 2 } };
    const secondOperations = { ...structuredClone(operations), server_instance_id: 'srv_after_restart' };
    const fetchImpl: typeof fetch = async (input) => {
      const url = String(input);
      if (url.endsWith('/bootstrap')) return new Response(JSON.stringify(bootstrap));
      if (url.includes('/catalog')) return new Response(JSON.stringify(catalogFixture));
      if (url === '/ui-api/v1/operations') return new Response(JSON.stringify(firstOperations));
      if (url === '/ui-api/v1/operations?cursor=ops_1') return new Response(JSON.stringify(secondOperations));
      if (url.includes('/events')) return new Response(streamDone(), { status: 200 });
      return new Response(JSON.stringify(runtimeFixture));
    };
    let snapshot = reduceWebUiSnapshot(initialSnapshot(), { type: 'login-success', bootstrap, now: 0 });
    const sync = new WebUiSynchronizer({ client: new WebUiApiClient({ fetchImpl }), getSnapshot: () => snapshot, dispatch: (action) => { snapshot = reduceWebUiSnapshot(snapshot, action); } });
    await sync.refresh();
    expect(snapshot.connection).toBe('stale');
    expect(snapshot.error?.code).toBe('snapshot_mismatch');
    sync.dispose();
  });

  it('keeps catalog pages and operation pages until their final cursor', async () => {
    const secondCatalog = structuredClone(catalogFixture);
    secondCatalog.items[0].identity.id = 'mdl_lR1nHQwFUguxLqHbEzH2DJLdZYiDFJ0S3FzxIzY5MUV';
    secondCatalog.items[0].identity.display_name = 'Zed Model';
    const firstCatalog = { ...structuredClone(catalogFixture), pagination: { limit: 50, next_cursor: 'cat_2', total_known: 2 } };
    const secondOperations = structuredClone(operations);
    secondOperations.items[0].operation_id = 'op_second';
    const firstOperations = { ...structuredClone(operations), pagination: { limit: 50, next_cursor: 'ops_1', total_known: 2 } };
    const fetchImpl: typeof fetch = async (input) => {
      const url = String(input);
      if (url.includes('/events')) return new Response(streamDone(), { status: 200 });
      if (url.endsWith('/bootstrap')) return new Response(JSON.stringify(bootstrap));
      if (url === '/ui-api/v1/catalog') return new Response(JSON.stringify(firstCatalog));
      if (url === '/ui-api/v1/catalog?cursor=cat_2') return new Response(JSON.stringify(secondCatalog));
      if (url === '/ui-api/v1/operations') return new Response(JSON.stringify(firstOperations));
      if (url === '/ui-api/v1/operations?cursor=ops_1') return new Response(JSON.stringify(secondOperations));
      return new Response(JSON.stringify(runtimeFixture));
    };
    let snapshot = reduceWebUiSnapshot(initialSnapshot(), { type: 'login-success', bootstrap, now: 0 });
    const sync = new WebUiSynchronizer({ client: new WebUiApiClient({ fetchImpl }), getSnapshot: () => snapshot, dispatch: (action) => { snapshot = reduceWebUiSnapshot(snapshot, action); } });
    await sync.refresh();
    expect(snapshot.catalog.map((item) => item.identity.display_name)).toEqual(['Qwen3 4B 4-bit', 'Zed Model']);
    expect([...snapshot.operations.keys()].sort()).toEqual(['op_second', operations.items[0].operation_id].sort());
    sync.dispose();
  });
});
