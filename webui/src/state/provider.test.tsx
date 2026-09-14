import React from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { act } from 'react';
import { describe, expect, it, vi } from 'vitest';
import bootstrapFixture from '../../../tests/fixtures/webui/examples/bootstrap.model-free.json';
import catalogFixture from '../../../tests/fixtures/webui/examples/catalog.page.json';
import operationsFixture from '../../../tests/fixtures/webui/examples/operations.list.json';
import runtimeFixture from '../../../tests/fixtures/webui/examples/runtime.snapshot.json';
import { WebUiApiClient, WebUiHttpError } from '../api/client';
import type { WebUiSnapshot } from '../api/types';
import { WebUiProvider, useWebUi, useWebUiActions, type WebUiActions } from './provider';

function Harness({ onState }: { readonly onState: (snapshot: WebUiSnapshot, actions: WebUiActions) => void }): React.JSX.Element {
  const snapshot = useWebUi();
  const actions = useWebUiActions();
  React.useEffect(() => onState(snapshot, actions), [snapshot, actions, onState]);
  return <span>{snapshot.auth.status}</span>;
}

function mount(fetchImpl: typeof fetch): { root: Root; element: HTMLDivElement; latest: () => { snapshot: WebUiSnapshot; actions: WebUiActions } } {
  const element = document.createElement('div');
  document.body.append(element);
  let current: { snapshot: WebUiSnapshot; actions: WebUiActions } | null = null;
  const root = createRoot(element);
  act(() => {
    root.render(<React.StrictMode><WebUiProvider fetchImpl={fetchImpl}><Harness onState={(snapshot, actions) => { current = { snapshot, actions }; }} /></WebUiProvider></React.StrictMode>);
  });
  return { root, element, latest: () => {
    if (current === null) throw new Error('Provider did not publish state yet.');
    return current;
  } };
}

describe('WebUiProvider auth races', () => {
  it('does not resurrect authentication when login bootstrap resolves after logout', async () => {
    let release: () => void = () => undefined;
    const fetchImpl: typeof fetch = async () => {
      await new Promise<void>((resolve) => { release = resolve; });
      return new Response(JSON.stringify(bootstrapFixture), { status: 200 });
    };
    const mounted = mount(fetchImpl);
    const login = mounted.latest().actions.login('token');
    act(() => mounted.latest().actions.logout());
    release();
    await expect(login).rejects.toMatchObject({ name: 'AbortError' });
    await act(async () => Promise.resolve());
    expect(mounted.latest().snapshot.auth.status).toBe('signed-out');
    expect(mounted.latest().snapshot.lastSuccessfulAt).toBeNull();
    act(() => mounted.root.unmount());
    mounted.element.remove();
  });

  it('central 401 handling signs out and does not record deterministic POST errors as unknown outcomes', async () => {
    const fetchImpl: typeof fetch = async (input) => {
      if (String(input).endsWith('/bootstrap')) return new Response(JSON.stringify(bootstrapFixture), { status: 200 });
      return new Response(JSON.stringify({ error: { code: 'unauthorized', message: 'bad key', retryable: false }, request_id: 'op_unauthorized' }), { status: 401 });
    };
    const mounted = mount(fetchImpl);
    await act(async () => mounted.latest().actions.login('token'));
    await expect(mounted.latest().actions.refreshCatalog('idem-deterministic')).rejects.toBeInstanceOf(WebUiHttpError);
    await act(async () => Promise.resolve());
    expect(mounted.latest().snapshot.auth.status).toBe('signed-out');
    expect(mounted.latest().snapshot.pendingReconciliations.size).toBe(0);
    act(() => mounted.root.unmount());
    mounted.element.remove();
  });

  it('does not dispatch a runtime snapshot that resolves after logout', async () => {
    let release: () => void = () => undefined;
    const fetchImpl: typeof fetch = async (input) => {
      if (String(input).endsWith('/bootstrap')) return new Response(JSON.stringify(bootstrapFixture), { status: 200 });
      await new Promise<void>((resolve) => { release = resolve; });
      return new Response(JSON.stringify(runtimeFixture), { status: 200 });
    };
    const mounted = mount(fetchImpl);
    await act(async () => mounted.latest().actions.login('token'));
    const runtime = mounted.latest().actions.refreshRuntime(runtimeFixture.model_id);
    act(() => mounted.latest().actions.logout());
    release();
    await expect(runtime).rejects.toMatchObject({ name: 'AbortError' });
    await act(async () => Promise.resolve());
    expect(mounted.latest().snapshot.auth.status).toBe('signed-out');
    expect(mounted.latest().snapshot.runtimes.size).toBe(0);
    expect(mounted.latest().snapshot.lastSuccessfulAt).toBeNull();
    act(() => mounted.root.unmount());
    mounted.element.remove();
  });
});



describe('operation session fences', () => {
  it.each(['accepted', 'rejected'] as const)('ignores late %s operation outcomes after logout and relogin', async (outcome) => {
    let resolve: ((value: { operation_id: string; state: 'queued'; idempotent_replay: boolean }) => void) | undefined;
    let reject: ((reason: Error) => void) | undefined;
    const spy = vi.spyOn(WebUiApiClient.prototype, 'refreshCatalog').mockImplementation(() => new Promise((done, fail) => { resolve = done; reject = fail; }));
    const fetchImpl: typeof fetch = async (input) => new Response(JSON.stringify(String(input).endsWith('/bootstrap') ? bootstrapFixture : { error: { code: 'not_found', message: 'No fixture', retryable: false }, request_id: 'test' }), { status: String(input).endsWith('/bootstrap') ? 200 : 404 });
    const mounted = mount(fetchImpl);
    try {
      await act(async () => mounted.latest().actions.login('old-token'));
      let pending: Promise<void> = Promise.resolve();
      act(() => { pending = mounted.latest().actions.refreshCatalog('old-session-operation'); });
      const observed = pending.catch(() => undefined);
      act(() => mounted.latest().actions.logout());
      await act(async () => mounted.latest().actions.login('new-token'));
      await act(async () => { if (outcome === 'accepted') resolve?.({ operation_id: 'op_late', state: 'queued', idempotent_replay: false }); else reject?.(new TypeError('late network error')); await observed; });
      expect(mounted.latest().snapshot.auth.status).toBe('authenticated');
      expect(mounted.latest().snapshot.pendingReconciliations.size).toBe(0);
    } finally { spy.mockRestore(); act(() => mounted.root.unmount()); mounted.element.remove(); }
  });
});

describe('WebUiProvider selection and inference isolation', () => {
  it('keeps a frozen stream alive across model selection but aborts on logout', async () => {
    vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true);
    let signal: AbortSignal | null = null;
    const fetchImpl: typeof fetch = async (input, init) => {
      const url = String(input);
      if (url.includes('/bootstrap')) return new Response(JSON.stringify(bootstrapFixture));
      if (url.includes('/catalog')) return new Response(JSON.stringify({ ...catalogFixture, server_instance_id: bootstrapFixture.server.server_instance_id }));
      if (url.includes('/operations')) return new Response(JSON.stringify({ ...JSON.parse(JSON.stringify(operationsFixture), (key, value: unknown) => key === '$schemaName' ? undefined : value), server_instance_id: bootstrapFixture.server.server_instance_id }));
      if (url.includes('/chat/completions')) {
        signal = init?.signal ?? null;
        expect(JSON.parse(String(init?.body)).model).toBe(catalogFixture.items[0].identity.inference_id);
      }
      return new Response(new ReadableStream<Uint8Array>({ start() {} }), { headers: { 'content-type': 'text/event-stream' } });
    };
    const mounted = mount(fetchImpl);
    await act(async () => mounted.latest().actions.login('token'));
    await act(async () => mounted.latest().actions.refresh());
    const stream = mounted.latest().actions.streamChatCompletions(catalogFixture.items[0].identity.id, {messages:[]}, {onFrame:()=>undefined});
    const result = stream.catch((error: unknown) => error);
    await act(async () => { await Promise.resolve(); mounted.latest().actions.selectModel(null); });
    expect(signal).not.toBeNull();
    expect((signal as AbortSignal | null)?.aborted).toBe(false);
    act(() => mounted.latest().actions.logout());
    expect((signal as AbortSignal | null)?.aborted).toBe(true);
    await result; // Client abort rejection race is separately fixed/tested by Chat.
    act(() => mounted.root.unmount()); mounted.element.remove(); vi.unstubAllGlobals();
  });
});
