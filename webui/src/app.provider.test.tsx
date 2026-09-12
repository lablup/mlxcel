import React from 'react';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import bootstrapFixture from '../../tests/fixtures/webui/examples/bootstrap.model-free.json';
import catalogFixture from '../../tests/fixtures/webui/examples/catalog.page.json';
import operationsFixture from '../../tests/fixtures/webui/examples/operations.list.json';
import { App } from './app';
import { WebUiProvider } from './state';

interface FetchCall {
  readonly url: string;
  readonly method: string;
  readonly auth: string;
  readonly body: string;
}

type Scenario = 'happy' | 'bad-key' | 'offline' | 'malformed-bootstrap' | 'sync-401';

let root: Root | null = null;
let host: HTMLDivElement | null = null;

beforeEach(() => {
  HTMLDialogElement.prototype.showModal = vi.fn(function showModal(this: HTMLDialogElement) { this.setAttribute('open', ''); });
  HTMLDialogElement.prototype.close = vi.fn(function close(this: HTMLDialogElement) { this.removeAttribute('open'); this.dispatchEvent(new Event('close')); });
  localStorage.clear();
  sessionStorage.clear();
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
  act(() => root?.render(<React.StrictMode><WebUiProvider fetchImpl={fetchImpl}><App /></WebUiProvider></React.StrictMode>));
}

function createMockFetch(scenario: Scenario): { readonly calls: FetchCall[]; readonly fetchImpl: typeof fetch } {
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
    if (url.startsWith('/ui-api/v1/catalog')) return jsonResponse(makeCatalog());
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

function makeBootstrap(): typeof bootstrapFixture {
  return structuredClone(bootstrapFixture);
}

function makeCatalog(): typeof catalogFixture {
  const page = structuredClone(catalogFixture);
  page.server_instance_id = bootstrapFixture.server.server_instance_id;
  return page;
}

function makeOperations(): typeof operationsFixture {
  const page = structuredClone(operationsFixture);
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
});
