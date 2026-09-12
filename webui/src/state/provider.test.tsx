import React from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { act } from 'react';
import { describe, expect, it } from 'vitest';
import bootstrapFixture from '../../../tests/fixtures/webui/examples/bootstrap.model-free.json';
import { WebUiHttpError } from '../api/client';
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
    await expect(login).resolves.toBeUndefined();
    await act(async () => Promise.resolve());
    expect(mounted.latest().snapshot.auth.status).toBe('signed-out');
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
});
