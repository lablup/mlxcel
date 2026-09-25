import React from 'react';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import bootstrapFixture from '../../tests/fixtures/webui/examples/bootstrap.model-free.json';
import catalogFixture from '../../tests/fixtures/webui/examples/catalog.page.json';
import operationsFixture from '../../tests/fixtures/webui/examples/operations.list.json';
import { App } from './app';
import { replaceConversations, useConversations } from './features/chat/session';
import { WebUiProvider, WebUiSynchronizer } from './state';

interface FetchCall {
  readonly url: string;
  readonly method: string;
  readonly auth: string;
  readonly body: string;
}

type Scenario = 'happy' | 'bad-key' | 'offline' | 'malformed-bootstrap' | 'sync-401';

let root: Root | null = null;
let host: HTMLDivElement | null = null;
let conversationCount = 0;
function ConversationsProbe(): null { conversationCount = useConversations().length; return null; }

beforeEach(() => {
  HTMLDialogElement.prototype.showModal = vi.fn(function showModal(this: HTMLDialogElement) { this.setAttribute('open', ''); });
  HTMLDialogElement.prototype.close = vi.fn(function close(this: HTMLDialogElement) { this.removeAttribute('open'); this.dispatchEvent(new Event('close')); });
  localStorage.clear();
  sessionStorage.clear();
  replaceConversations([]);
  window.history.replaceState(null, '', '/webui/#models');
  host = document.createElement('div');
  host.id = 'root';
  document.body.append(host);
  root = createRoot(host);
});

afterEach(() => {
  vi.restoreAllMocks();
  act(() => root?.unmount());
  host?.remove();
  root = null;
  host = null;
});

function renderApp(fetchImpl: typeof fetch): void {
  act(() => root?.render(<React.StrictMode><WebUiProvider fetchImpl={fetchImpl}><App /><ConversationsProbe /></WebUiProvider></React.StrictMode>));
}

// The catalog fixture's single model, loaded and chat-ready on the server.
function readyCatalog(page: typeof catalogFixture): typeof catalogFixture {
  const entry = page.items[0];
  entry.lifecycle.state = 'ready';
  entry.lifecycle.worker_exit_observed = false;
  entry.capabilities = [{ task: 'chat', phase: 'provider_ready', available: true, reason: null }];
  return page;
}

// The fixture's unloaded model next to a second one whose load failed: neither holds a worker.
function unloadedAndFailedCatalog(page: typeof catalogFixture): typeof catalogFixture {
  const failed = structuredClone(page.items[0]);
  failed.identity = { ...failed.identity, id: `mdl_${'f'.repeat(43)}`, inference_id: 'beta', display_name: 'beta', source_key_hash: 'f'.repeat(64) };
  failed.lifecycle = { ...failed.lifecycle, state: 'failed' };
  page.items = [page.items[0], failed];
  page.pagination = { ...page.pagination, total_known: 2 };
  return page;
}

function createMockFetch(scenario: Scenario, catalog: (page: typeof catalogFixture) => typeof catalogFixture = (page) => page): { readonly calls: FetchCall[]; readonly fetchImpl: typeof fetch } {
  const calls: FetchCall[] = [];
  let bootstrapCount = 0;
  const fetchImpl: typeof fetch = async (input, init) => {
    const url = String(input);
    calls.push({ url, method: init?.method ?? 'GET', auth: new Headers(init?.headers).get('authorization') ?? '', body: init?.body === undefined ? '' : String(init.body) });
    if (scenario === 'offline') throw new TypeError('Failed to fetch');
    if (url.startsWith('/ui-api/v1/bootstrap')) {
      bootstrapCount += 1;
      if (scenario === 'bad-key') return jsonResponse({ error: { code: 'unauthorized', message: 'bad key', retryable: false }, request_id: 'req_bad_key' }, 401);
      if (scenario === 'malformed-bootstrap') return jsonResponse({ schema_version: 'wrong', server: { server_instance_id: 'srv_bad' } });
      return jsonResponse(makeBootstrap());
    }
    if (url.startsWith('/ui-api/v1/catalog')) return jsonResponse(catalog(makeCatalog()));
    if (url.startsWith('/ui-api/v1/operations')) {
      if (scenario === 'sync-401' && bootstrapCount >= 2) return jsonResponse({ error: { code: 'unauthorized', message: 'stale session', retryable: false }, request_id: 'req_sync_401' }, 401);
      return jsonResponse(makeOperations());
    }
    if (url.startsWith('/ui-api/v1/events')) return new Response('data: [DONE]\n\n', { status: 200, headers: { 'content-type': 'text/event-stream' } });
    return jsonResponse({ error: { code: 'not_found', message: `Unexpected ${url}`, retryable: false }, request_id: 'req_unexpected' }, 404);
  };
  return { calls, fetchImpl };
}

async function submitSessionKey(token: string): Promise<void> {
  const form = document.querySelector<HTMLFormElement>('[data-testid="auth-login"]');
  const input = form?.querySelector<HTMLInputElement>('input[type="password"]');
  expect(form).not.toBeNull();
  expect(input).not.toBeNull();
  const valueSetter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set;
  await act(async () => {
    valueSetter?.call(input, token);
    input?.dispatchEvent(new Event('input', { bubbles: true }));
    form?.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
  });
}

async function waitFor(assertion: () => unknown, attempts = 80): Promise<void> {
  let lastError: unknown;
  for (let i = 0; i < attempts; i += 1) {
    try {
      const result = assertion();
      if (result !== false) return;
    } catch (error) {
      lastError = error;
    }
    await act(async () => { await new Promise((resolve) => window.setTimeout(resolve, 10)); });
  }
  if (lastError) throw lastError;
  throw new Error('Timed out waiting for assertion.');
}

function withoutSchemaName<T>(value: T): T {
  const copy = structuredClone(value);
  if (typeof copy === 'object' && copy !== null && '$schemaName' in copy) delete (copy as Record<string, unknown>).$schemaName;
  return copy;
}

function makeBootstrap(): typeof bootstrapFixture {
  return withoutSchemaName(bootstrapFixture);
}

function makeCatalog(): typeof catalogFixture {
  const page = withoutSchemaName(catalogFixture);
  page.server_instance_id = bootstrapFixture.server.server_instance_id;
  return page;
}

function makeOperations(): typeof operationsFixture {
  const page = withoutSchemaName(operationsFixture);
  page.server_instance_id = bootstrapFixture.server.server_instance_id;
  return page;
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } });
}

describe('provider-backed WebUI shell', () => {
  it('does not contact the local API before explicit login and keeps the gallery isolated', () => {
    const mock = createMockFetch('happy');
    renderApp(mock.fetchImpl);
    expect(mock.calls).toEqual([]);
    expect(document.querySelector('[data-testid="auth-login"]')).not.toBeNull();
    act(() => { window.history.replaceState(null, '', '#gallery'); window.dispatchEvent(new HashChangeEvent('hashchange')); });
    expect(window.location.hash).toBe('#gallery');
    expect(mock.calls).toEqual([]);
  });

  it('uses Bearer authorization for bootstrap, catalog, operations and event snapshots', async () => {
    const mock = createMockFetch('happy');
    renderApp(mock.fetchImpl);
    await submitSessionKey('good-key');
    await waitFor(() => expect(mock.calls.some((call) => call.url.startsWith('/ui-api/v1/operations'))).toBe(true));
    const snapshotCalls = mock.calls.filter((call) => call.url.startsWith('/ui-api/v1/bootstrap') || call.url.startsWith('/ui-api/v1/catalog') || call.url.startsWith('/ui-api/v1/operations') || call.url.startsWith('/ui-api/v1/events'));
    expect(snapshotCalls.map((call) => call.auth)).toEqual(snapshotCalls.map(() => 'Bearer good-key'));
    expect(snapshotCalls.map((call) => call.url).join('\n')).toContain('/ui-api/v1/catalog');
    expect(document.querySelector('[data-testid="connection-authenticated-detail"]')?.textContent).toContain('catalog 1');
    expect(document.querySelector('[data-testid="connection-ready"]')?.textContent).toContain('model_free');
    expect(document.querySelector('[data-testid="connection-ready"]')?.textContent).not.toContain('seq');
    expect(`${localStorage.getItem('good-key')} ${sessionStorage.getItem('good-key')}`).not.toContain('good-key');
    expect(mock.calls.map((call) => call.url).join(' ')).not.toContain('good-key');
  });

  it('purges the stale session and provider data when the authoritative refresh returns 401', async () => {
    const mock = createMockFetch('sync-401');
    renderApp(mock.fetchImpl);
    await submitSessionKey('good-key');
    await waitFor(() => expect(mock.calls.some((call) => call.url.startsWith('/ui-api/v1/operations'))).toBe(true));
    await waitFor(() => expect(document.querySelector('[data-testid="auth-login"]')).not.toBeNull());
    expect(document.body.textContent).toContain('Authentication required');
    expect(document.querySelector('[data-testid="connection-ready"]')?.textContent).toBe('Shell loaded; local API not connected');
    expect(document.body.textContent).not.toContain('Authenticated local API session');
  });

  it('fails closed on malformed bootstrap schema and offers reload recovery', async () => {
    const mock = createMockFetch('malformed-bootstrap');
    renderApp(mock.fetchImpl);
    await submitSessionKey('good-key');
    await waitFor(() => expect(document.body.textContent).toContain('UI schema mismatch'));
    expect(mock.calls.map((call) => call.url)).toEqual(['/ui-api/v1/bootstrap']);
    expect(document.body.textContent).toContain('Reload');
    expect(document.body.textContent).not.toContain('Authenticated local API session');
  });

  it('shows localized wrong-key and offline login errors without persisting tokens', async () => {
    const badKey = createMockFetch('bad-key');
    renderApp(badKey.fetchImpl);
    await submitSessionKey('bad-key-token');
    await waitFor(() => expect(document.body.textContent).toContain('The session key was rejected'));
    expect(`${localStorage.getItem('bad-key-token')} ${sessionStorage.getItem('bad-key-token')}`).not.toContain('bad-key-token');
    expect(document.body.textContent).not.toContain('bad-key-token');
    act(() => root?.unmount());
    if (host === null) throw new Error('Missing test host.');
    host.innerHTML = '';
    root = createRoot(host);
    const offline = createMockFetch('offline');
    renderApp(offline.fetchImpl);
    await submitSessionKey('offline-token');
    await waitFor(() => expect(document.body.textContent).toContain('Could not reach the local WebUI API'));
    expect(`${localStorage.getItem('offline-token')} ${sessionStorage.getItem('offline-token')}`).not.toContain('offline-token');
    expect(document.body.textContent).not.toContain('offline-token');
  });


  it('shows pending counts until authoritative snapshots arrive and keeps failed snapshots honest', async () => {
    const calls: FetchCall[] = [];
    let catalogRequested = false;
    const fetchImpl: typeof fetch = async (input, init) => {
      const url = String(input);
      calls.push({ url, method: init?.method ?? 'GET', auth: new Headers(init?.headers).get('authorization') ?? '', body: init?.body === undefined ? '' : String(init.body) });
      if (url.startsWith('/ui-api/v1/bootstrap')) return jsonResponse(makeBootstrap());
      if (url.startsWith('/ui-api/v1/catalog')) {
        catalogRequested = true;
        throw new TypeError('Failed to fetch catalog snapshot');
      }
      if (url.startsWith('/ui-api/v1/operations')) return jsonResponse(makeOperations());
      if (url.startsWith('/ui-api/v1/events')) return new Response('data: [DONE]\n\n', { status: 200, headers: { 'content-type': 'text/event-stream' } });
      return jsonResponse({ error: { code: 'not_found', message: `Unexpected ${url}`, retryable: false }, request_id: 'req_unexpected' }, 404);
    };
    renderApp(fetchImpl);
    await submitSessionKey('good-key');
    await waitFor(() => expect(document.querySelector('[data-testid="connection-authenticated-detail"]')?.textContent).toContain('catalog pending'));
    expect(document.querySelector('[data-testid="connection-authenticated-detail"]')?.textContent).toContain('operations pending');
    await waitFor(() => expect(catalogRequested).toBe(true));
    await waitFor(() => expect(document.querySelector('[data-testid="connection-error-title"]')).not.toBeNull());
    expect(document.querySelector('[data-testid="connection-authenticated-detail"]')?.textContent).toContain('catalog pending');
    expect(document.querySelector('[data-testid="connection-authenticated-detail"]')?.textContent).toContain('operations pending');
    expect(calls.map((call) => call.url).join('\n')).toContain('/ui-api/v1/catalog');
  });

  it('ignores stale authentication failures after logout and allows a fresh attempt', async () => {
    let firstBootstrap: ((response: Response) => void) | null = null;
    let bootstrapCount = 0;
    const calls: FetchCall[] = [];
    const fetchImpl: typeof fetch = async (input, init) => {
      const url = String(input);
      calls.push({ url, method: init?.method ?? 'GET', auth: new Headers(init?.headers).get('authorization') ?? '', body: init?.body === undefined ? '' : String(init.body) });
      if (url.startsWith('/ui-api/v1/bootstrap')) {
        bootstrapCount += 1;
        if (bootstrapCount === 1) return new Promise<Response>((resolve) => { firstBootstrap = resolve; });
        return jsonResponse(makeBootstrap());
      }
      if (url.startsWith('/ui-api/v1/catalog')) return jsonResponse(makeCatalog());
      if (url.startsWith('/ui-api/v1/operations')) return jsonResponse(makeOperations());
      if (url.startsWith('/ui-api/v1/events')) return new Response('data: [DONE]\n\n', { status: 200, headers: { 'content-type': 'text/event-stream' } });
      return jsonResponse({ error: { code: 'not_found', message: `Unexpected ${url}`, retryable: false }, request_id: 'req_unexpected' }, 404);
    };
    renderApp(fetchImpl);
    await submitSessionKey('old-bad-key');
    await waitFor(() => expect(document.querySelector('[data-testid="toolbar-logout"]')).not.toBeNull());
    act(() => document.querySelector<HTMLButtonElement>('[data-testid="toolbar-logout"]')?.click());
    expect(firstBootstrap).not.toBeNull();
    await act(async () => {
      firstBootstrap?.(jsonResponse({ error: { code: 'unauthorized', message: 'old bad key', retryable: false }, request_id: 'req_old_bad' }, 401));
      await Promise.resolve();
    });
    expect(document.body.textContent).not.toContain('The session key was rejected');
    expect(document.body.textContent).not.toContain('old-bad-key');
    await submitSessionKey('good-key');
    await waitFor(() => expect(document.querySelector('[data-testid="connection-authenticated-detail"]')?.textContent).toContain('catalog 1'));
    expect(calls.map((call) => call.auth).filter(Boolean)).toContain('Bearer good-key');
  });

  it('clears the Models search on logout so the next session in this tab starts from the default view', async () => {
    const mock = createMockFetch('happy');
    renderApp(mock.fetchImpl);
    await submitSessionKey('good-key');
    const search = (): HTMLInputElement | null => document.querySelector<HTMLInputElement>('[data-testid="models-search"]');
    await waitFor(() => expect(search()).not.toBeNull());
    const field = search() as HTMLInputElement;
    act(() => {
      Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set?.call(field, 'previous-user-query');
      field.dispatchEvent(new Event('input', { bubbles: true }));
    });
    expect(search()?.value).toBe('previous-user-query');
    act(() => document.querySelector<HTMLButtonElement>('[data-testid="toolbar-logout"]')?.click());
    await submitSessionKey('good-key');
    await waitFor(() => expect(search()).not.toBeNull());
    expect(search()?.value).toBe('');
  });

  it('does not autoload models or call inference endpoints while browsing authenticated routes', async () => {
    const mock = createMockFetch('happy');
    renderApp(mock.fetchImpl);
    await submitSessionKey('good-key');
    await waitFor(() => expect(mock.calls.some((call) => call.url.startsWith('/ui-api/v1/operations'))).toBe(true));
    for (const route of ['nav-chat', 'nav-activity', 'nav-settings', 'nav-models']) {
      act(() => document.querySelector<HTMLAnchorElement>(`[data-testid="${route}"]`)?.click());
    }
    await act(async () => { await Promise.resolve(); });
    const urls = mock.calls.map((call) => call.url).join('\n');
    expect(urls).not.toContain('/v1/chat/completions');
    expect(urls).not.toContain('/v1/responses');
    expect(urls).not.toContain('autoload=true');
    expect(urls).not.toContain('/ui-api/v1/runtime');
    expect(urls).not.toContain('/ui-api/v1/model-actions');
    expect(urls).not.toContain('/ui-api/v1/downloads');
  });

  it('names the server-loaded model in the toolbar with no browser selection and opens its inspector from the chip', async () => {
    const mock = createMockFetch('happy', readyCatalog);
    renderApp(mock.fetchImpl);
    const region = document.querySelector<HTMLElement>('[data-testid="toolbar-loaded"]');
    expect(region?.getAttribute('role')).toBe('group');
    expect(region?.getAttribute('aria-label')).toBe('Loaded models');
    // Signed out, nothing has been read from the server, so the toolbar claims nothing about it.
    expect(region?.textContent).toBe('Loaded models unknown');
    expect(region?.querySelector('[data-testid="toolbar-loaded-unknown"]')).not.toBeNull();
    expect(region?.querySelector('[data-testid="toolbar-loaded-none"]')).toBeNull();
    await submitSessionKey('good-key');
    await waitFor(() => expect(document.querySelector('[data-testid="toolbar-loaded-count"]')?.textContent).toBe('1 loaded'));
    const chips = document.querySelectorAll<HTMLButtonElement>('[data-testid="toolbar-loaded-chip"]');
    expect(chips).toHaveLength(1);
    expect(chips[0].textContent).toBe('alphaReady');
    expect(chips[0].title).toBe('alpha');
    expect(chips[0].hasAttribute('aria-label')).toBe(false);
    expect(chips[0].querySelector('.ds-status-wrap[data-state="ready"]')?.textContent).toBe('Ready');
    expect(document.querySelector('[data-testid="toolbar-loaded-more"]')).toBeNull();
    expect(document.querySelector('aside[aria-label="Model details"]')).toBeNull();
    act(() => document.querySelector<HTMLAnchorElement>('[data-testid="nav-settings"]')?.click());
    expect(window.location.hash).toBe('#settings');
    act(() => document.querySelector<HTMLButtonElement>('[data-testid="toolbar-loaded-chip"]')?.click());
    await waitFor(() => expect(document.querySelector('aside[aria-label="Model details"]')?.textContent).toContain('alpha'));
    expect(window.location.hash).toBe('#models');
    expect(mock.calls.map((call) => call.url).join('\n')).not.toContain('/ui-api/v1/model-actions');
  });

  it('says the loaded models are unknown until the first catalog page, then that none is loaded when every entry is unloaded or failed', async () => {
    let releaseCatalog: (() => void) | null = null;
    const catalogGate = new Promise<void>((resolve) => { releaseCatalog = resolve; });
    const fetchImpl: typeof fetch = async (input) => {
      const url = String(input);
      if (url.startsWith('/ui-api/v1/bootstrap')) return jsonResponse(makeBootstrap());
      if (url.startsWith('/ui-api/v1/catalog')) {
        await catalogGate;
        return jsonResponse(unloadedAndFailedCatalog(makeCatalog()));
      }
      if (url.startsWith('/ui-api/v1/operations')) return jsonResponse(makeOperations());
      if (url.startsWith('/ui-api/v1/events')) return new Response('data: [DONE]\n\n', { status: 200, headers: { 'content-type': 'text/event-stream' } });
      return jsonResponse({ error: { code: 'not_found', message: `Unexpected ${url}`, retryable: false }, request_id: 'req_unexpected' }, 404);
    };
    renderApp(fetchImpl);
    const region = (): HTMLElement | null => document.querySelector<HTMLElement>('[data-testid="toolbar-loaded"]');
    await submitSessionKey('good-key');
    // Authenticated, but the catalog snapshot has not arrived: still unknown, not "none".
    await waitFor(() => expect(document.querySelector('[data-testid="connection-authenticated-detail"]')?.textContent).toContain('catalog pending'));
    expect(document.querySelector('[data-testid="toolbar-logout"]')).not.toBeNull();
    expect(region()?.textContent).toBe('Loaded models unknown');
    await act(async () => { releaseCatalog?.(); await Promise.resolve(); });
    await waitFor(() => expect(document.querySelector('[data-testid="connection-authenticated-detail"]')?.textContent).toContain('catalog 2'));
    expect(region()?.textContent).toBe('No model loaded');
    expect(region()?.querySelector('[data-testid="toolbar-loaded-none"]')).not.toBeNull();
    expect(region()?.querySelectorAll('[data-testid="toolbar-loaded-chip"]')).toHaveLength(0);
  });

  it('does not restart observation when a chip or the palette opens the model that is already selected', async () => {
    // Every selection change reaches the synchronizer through selectionChanged, which cancels
    // observation, refetches and (with the reducer) clears the runtime history.
    const selectionChanged = vi.spyOn(WebUiSynchronizer.prototype, 'selectionChanged');
    const mock = createMockFetch('happy', readyCatalog);
    renderApp(mock.fetchImpl);
    await submitSessionKey('good-key');
    await waitFor(() => expect(document.querySelector('[data-testid="toolbar-loaded-count"]')?.textContent).toBe('1 loaded'));
    expect(selectionChanged).not.toHaveBeenCalled();
    act(() => document.querySelector<HTMLButtonElement>('[data-testid="toolbar-loaded-chip"]')?.click());
    await waitFor(() => expect(document.querySelector('aside[aria-label="Model details"]')?.textContent).toContain('alpha'));
    expect(selectionChanged).toHaveBeenCalledTimes(1);
    act(() => document.querySelector<HTMLAnchorElement>('[data-testid="nav-settings"]')?.click());
    act(() => document.querySelector<HTMLButtonElement>('[data-testid="toolbar-loaded-chip"]')?.click());
    expect(window.location.hash).toBe('#models');
    act(() => document.querySelector<HTMLAnchorElement>('[data-testid="nav-activity"]')?.click());
    act(() => document.querySelector<HTMLButtonElement>('[data-testid="toolbar-command"]')?.click());
    const dialog = document.querySelector<HTMLElement>('[data-testid="command-dialog"]');
    act(() => dialog?.querySelector<HTMLButtonElement>('[data-testid="command-model"]')?.click());
    expect(window.location.hash).toBe('#models');
    expect(dialog?.hasAttribute('open')).toBe(false);
    expect(selectionChanged).toHaveBeenCalledTimes(1);
  });

  it('keeps the footer to mode, status and version and puts the instance and sequence only in its details', async () => {
    const mock = createMockFetch('happy');
    renderApp(mock.fetchImpl);
    expect(document.querySelector('[data-testid="connection-footer-details"]')).toBeNull();
    await submitSessionKey('good-key');
    await waitFor(() => expect(document.querySelector('[data-testid="connection-footer-details"]')).not.toBeNull());
    const instance = bootstrapFixture.server.server_instance_id;
    const label = document.querySelector<HTMLElement>('[data-testid="connection-ready"]');
    expect(label?.textContent).toMatch(/^model_free · \w+ · v0\.7\.0$/);
    const footer = label?.closest('footer');
    const details = footer?.querySelector<HTMLDetailsElement>('details.connection-details');
    expect(details?.open).toBe(false);
    expect(details?.querySelector('summary')?.textContent).toBe('Connection details');
    expect(details?.querySelector('[data-testid="connection-footer-instance"]')?.textContent).toBe(`Server instance: ${instance}`);
    expect(details?.querySelector('[data-testid="connection-footer-sequence"]')?.textContent).toMatch(/^Event sequence: \d+$/);
    // Outside the disclosure, the footer shows neither identifier.
    const outside = Array.from(footer?.childNodes ?? []).filter((node) => node !== details).map((node) => node.textContent).join(' ');
    expect(outside).not.toContain(instance);
    expect(outside).not.toMatch(/sequence|seq/i);
  });

  it.each(['models', 'activity', 'settings'])('starts a new empty conversation with Cmd/Ctrl+N from %s', async (route) => {
    const mock = createMockFetch('happy', readyCatalog);
    renderApp(mock.fetchImpl);
    await submitSessionKey('good-key');
    await waitFor(() => expect(document.querySelector('[data-testid="toolbar-loaded-count"]')).not.toBeNull());
    act(() => document.querySelector<HTMLAnchorElement>(`[data-testid="nav-${route}"]`)?.click());
    act(() => { (document.activeElement as HTMLElement | null)?.blur(); });
    expect(conversationCount).toBe(0);
    let event: KeyboardEvent | undefined;
    act(() => { event = new KeyboardEvent('keydown', { key: 'n', ctrlKey: true, bubbles: true, cancelable: true }); document.body.dispatchEvent(event); });
    expect(event?.defaultPrevented).toBe(true);
    await waitFor(() => expect(conversationCount).toBe(1));
    expect(window.location.hash).toBe('#chat');
    expect(document.querySelector('[data-testid="chat-title"]')).not.toBeNull();
    act(() => { document.body.dispatchEvent(new KeyboardEvent('keydown', { key: 'n', metaKey: true, bubbles: true, cancelable: true })); });
    await waitFor(() => expect(conversationCount).toBe(2));
    expect(mock.calls.map((call) => call.url).join('\n')).not.toContain('/v1/chat/completions');
  });

  it('finds a catalog model in the command palette and opens its inspector without any model action', async () => {
    const mock = createMockFetch('happy');
    renderApp(mock.fetchImpl);
    await submitSessionKey('good-key');
    await waitFor(() => expect(document.querySelector('[data-testid="connection-authenticated-detail"]')?.textContent).toContain('catalog 1'));
    act(() => document.querySelector<HTMLAnchorElement>('[data-testid="nav-activity"]')?.click());
    act(() => document.querySelector<HTMLButtonElement>('[data-testid="toolbar-command"]')?.click());
    const dialog = document.querySelector<HTMLElement>('[data-testid="command-dialog"]');
    expect(dialog?.hasAttribute('open')).toBe(true);
    // Nothing is loaded, so the empty query lists commands only.
    expect(dialog?.querySelectorAll('[data-testid="command-model"]')).toHaveLength(0);
    const search = dialog?.querySelector<HTMLInputElement>('[data-testid="command-search"]');
    act(() => { Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set?.call(search, 'ALP'); search?.dispatchEvent(new Event('input', { bubbles: true })); });
    const hits = dialog?.querySelectorAll<HTMLButtonElement>('[data-testid="command-model"]');
    expect(Array.from(hits ?? []).map((hit) => hit.textContent)).toEqual(['alpha · Unloaded']);
    act(() => { Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set?.call(search, catalogFixture.items[0].identity.id.slice(4, 14)); search?.dispatchEvent(new Event('input', { bubbles: true })); });
    expect(dialog?.querySelectorAll('[data-testid="command-model"]')).toHaveLength(1);
    act(() => dialog?.querySelector<HTMLButtonElement>('[data-testid="command-model"]')?.click());
    await waitFor(() => expect(document.querySelector('aside[aria-label="Model details"]')?.textContent).toContain('alpha'));
    expect(window.location.hash).toBe('#models');
    expect(dialog?.hasAttribute('open')).toBe(false);
    expect(mock.calls.filter((call) => call.method !== 'GET')).toEqual([]);
    expect(mock.calls.map((call) => call.url).join('\n')).not.toContain('/ui-api/v1/model-actions');
  });
});
