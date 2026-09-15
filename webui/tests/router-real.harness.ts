// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import { mkdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { expect, test, type APIRequestContext, type Page } from '@playwright/test';
import { expectAxeClean, expectSafeLayout } from './browser-assertions';

type Operation = { operation_id: string; kind: string; state: string; result?: unknown; error?: unknown };
type CatalogItem = { identity: { id: string; display_name: string; inference_id: string; revision: number }; lifecycle: { state: string; worker_exit_observed: boolean; active_requests: number }; capabilities: Array<{ task: string; phase: string; available: boolean }> };
type HarnessContext = { target: URL; apiBase: string; artifacts: string; keyPath: string; token: string; auth: { Authorization: string } };
type ErrorEnvelope = { error: { code: string; operation_id?: string | null; field_errors?: Array<{ field: string; code: string; message: string }> | null } };
type HttpObservation = { method: string; path: string; status?: number; failure?: string };

function harnessContext(): HarnessContext {
  const rawTarget = process.env.MLXCEL_WEBUI_ROUTER_URL;
  const artifacts = process.env.MLXCEL_WEBUI_ROUTER_ARTIFACTS;
  const keyPath = process.env.MLXCEL_WEBUI_ROUTER_KEY_FILE;
  if (!rawTarget || !artifacts || !keyPath) throw new Error('Set MLXCEL_WEBUI_ROUTER_URL, MLXCEL_WEBUI_ROUTER_KEY_FILE and MLXCEL_WEBUI_ROUTER_ARTIFACTS; missing real-router setup is not a skipped pass.');
  const target = new URL(rawTarget);
  if (!['http:', 'https:'].includes(target.protocol) || !['127.0.0.1', 'localhost', '[::1]'].includes(target.hostname) || target.username || target.password || !target.pathname.endsWith('/webui/')) throw new Error('The Rust-router harness must target a loopback /webui/ URL without URL credentials.');
  const token = readFileSync(keyPath, 'utf8').trim();
  return { target, apiBase: target.pathname.slice(0, -'/webui/'.length), artifacts, keyPath, token, auth: { Authorization: `Bearer ${token}` } };
}

function saveArtifact(ctx: HarnessContext, name: string, value: unknown): void {
  mkdirSync(ctx.artifacts, { recursive: true });
  writeFileSync(join(ctx.artifacts, name), typeof value === 'string' ? value : JSON.stringify(value, null, 2), { mode: 0o600, flag: 'wx' });
}

async function apiJson<T>(request: APIRequestContext, ctx: HarnessContext, method: 'get' | 'post' | 'delete', path: string, body?: unknown): Promise<T> {
  const response = await request[method](`${ctx.target.origin}${ctx.apiBase}${path}`, { headers: ctx.auth, data: body });
  expect(response.ok(), `${method.toUpperCase()} ${path}: ${response.status()} ${await response.text()}`).toBe(true);
  return (await response.json()) as T;
}

async function waitOperation(request: APIRequestContext, ctx: HarnessContext, operationId: string): Promise<Operation> {
  for (let attempt = 0; attempt < 200; attempt += 1) {
    const operation = await apiJson<Operation>(request, ctx, 'get', `/ui-api/v1/operations/${operationId}`);
    if (['succeeded', 'failed', 'cancelled'].includes(operation.state)) {
      expect(operation.state, JSON.stringify(operation)).toBe('succeeded');
      return operation;
    }
    await new Promise(resolve => setTimeout(resolve, 25));
  }
  throw new Error(`operation ${operationId} did not reach a terminal state`);
}

async function waitOperationKind(request: APIRequestContext, ctx: HarnessContext, kind: string): Promise<string> {
  for (let attempt = 0; attempt < 200; attempt += 1) {
    const operations = (await apiJson<{ items: Operation[] }>(request, ctx, 'get', '/ui-api/v1/operations')).items;
    const operation = operations.find(item => item.kind === kind);
    if (operation) return operation.operation_id;
    await new Promise(resolve => setTimeout(resolve, 25));
  }
  throw new Error(`operation kind ${kind} was not observed`);
}

function requireCatalogItem(item: CatalogItem | undefined, label: string): CatalogItem {
  if (!item) throw new Error(`${label} catalog entry not found`);
  return item;
}

async function catalog(request: APIRequestContext, ctx: HarnessContext): Promise<CatalogItem[]> {
  const body = await apiJson<{ items: CatalogItem[] }>(request, ctx, 'get', '/ui-api/v1/catalog?limit=200');
  return body.items;
}

async function login(page: Page, ctx: HarnessContext, observations: HttpObservation[]): Promise<void> {
  await page.goto(`${ctx.target.href}#models`);
  await page.getByLabel(/Session key|세션 키/i).fill(ctx.token);
  await page.getByRole('button', { name: /Connect|연결/i }).click();
  await page.waitForTimeout(750);
  saveArtifact(ctx, 'router-login-observations.json', { api_base: ctx.apiBase, current_path: new URL(page.url()).pathname, observations });
  await expect(page.getByTestId('models-table')).toBeVisible();
}

async function refresh(page: Page): Promise<void> {
  await page.getByRole('button', { name: 'Refresh server state', exact: true }).first().click();
}

test.describe('Rust router harness', () => {
  test('secured Rust router drives download, load, chat, Stop/drain, unload, remove and negative security cases', async ({ page, request }) => {
    const ctx = harnessContext();
    expect(statSync(ctx.keyPath).mode & 0o777).toBe(0o600);
    const external: string[] = [];
    const httpObservations: HttpObservation[] = [];
    const remember = (observation: HttpObservation): void => { httpObservations.push(observation); if (httpObservations.length > 200) httpObservations.shift(); };
    page.on('request', req => { const url = new URL(req.url()); if (!['data:', 'blob:'].includes(url.protocol) && url.origin !== ctx.target.origin) external.push(req.url()); });
    page.on('response', response => { const url = new URL(response.url()); if (url.origin === ctx.target.origin) remember({ method: response.request().method(), path: `${url.pathname}${url.search}`, status: response.status() }); });
    page.on('requestfailed', request => { const url = new URL(request.url()); if (url.origin === ctx.target.origin) remember({ method: request.method(), path: `${url.pathname}${url.search}`, failure: request.failure()?.errorText ?? 'unknown' }); });
    await page.addInitScript(() => { const violations: string[] = []; Object.defineProperty(window, '__routerCspViolations', { value: violations }); document.addEventListener('securitypolicyviolation', event => violations.push(`${event.effectiveDirective}: ${event.blockedURI}`)); });

    const shell = await page.goto(`${ctx.target.href}#models`);
    expect(shell?.ok()).toBe(true);
    const csp = shell?.headers()['content-security-policy'];
    expect(csp).toMatch(/(?:^|;)\s*style-src 'self'(?:;|$)/);
    expect(csp).not.toContain("'unsafe-inline'");
    expect(csp).not.toContain("'unsafe-eval'");
    await login(page, ctx, httpObservations);
    await expectSafeLayout(page); await expectAxeClean(page);

    await page.getByTestId('models-add').click();
    await page.getByTestId('models-repo').fill('mlx-community/Router-Harness-Fake-4bit');
    await page.getByTestId('models-public-repo').check();
    await page.getByTestId('models-download-submit').click();
    const acceptedDownload = await waitOperationKind(request, ctx, 'download');
    await waitOperation(request, ctx, acceptedDownload);
    await refresh(page);
    await expect.poll(async () => (await catalog(request, ctx)).length).toBeGreaterThan(1);

    const downloaded = requireCatalogItem((await catalog(request, ctx)).find(item => item.identity.display_name.includes('Router-Harness-Fake')) ?? (await catalog(request, ctx)).find(item => item.identity.id.includes('Router-Harness-Fake')), 'downloaded fake');
    await page.getByRole('button', { name: new RegExp(`Inspect .*${downloaded.identity.display_name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}`) }).click();
    await page.getByTestId('models-load').click();
    const loadAccepted = await waitOperationKind(request, ctx, 'model_load');
    await waitOperation(request, ctx, loadAccepted);
    const cancelFinished = await request.post(`${ctx.target.origin}${ctx.apiBase}/ui-api/v1/operations/${loadAccepted}/cancel`, { headers: ctx.auth });
    expect(cancelFinished.status()).toBe(422);
    await refresh(page);
    await expect.poll(async () => (await catalog(request, ctx)).find(item => item.identity.id === downloaded.identity.id)?.capabilities.some(cap => cap.task === 'chat' && cap.phase === 'provider_ready' && cap.available)).toBe(true);

    const ready = requireCatalogItem((await catalog(request, ctx)).find(item => item.identity.id === downloaded.identity.id), 'ready fake');
    const staleAction = await request.post(`${ctx.target.origin}${ctx.apiBase}/ui-api/v1/model-actions`, { headers: ctx.auth, data: { model_id: ready.identity.id, action: 'unload', expected_revision: ready.identity.revision + 1000, idempotency_key: 'router-real-stale-revision' } });
    expect(staleAction.status()).toBe(409);
    expect(((await staleAction.json()) as ErrorEnvelope).error.code).toBe('stale_revision');
    const unknownField = await request.post(`${ctx.target.origin}${ctx.apiBase}/ui-api/v1/model-actions`, { headers: ctx.auth, data: { model_id: ready.identity.id, action: 'load', expected_revision: ready.identity.revision, idempotency_key: 'router-real-unknown-field', extra_field: true } });
    expect(unknownField.status()).toBe(400);

    await page.goto(`${ctx.target.href}#chat`);
    await page.getByRole('combobox', { name: 'Model for next turn' }).click();
    await page.getByRole('option', { name: new RegExp(ready.identity.display_name) }).click();
    await page.getByRole('textbox', { name: 'Message', exact: true }).fill('Say hello from the router harness.');
    await page.getByRole('button', { name: 'Send', exact: true }).click();
    await expect(page.locator('.chat-turn header')).toContainText('complete', { timeout: 30000 });
    await expect(page.locator('.chat-markdown')).toContainText(/router harness/i);

    await page.getByRole('button', { name: 'New conversation', exact: true }).click();
    await page.getByRole('textbox', { name: 'Message', exact: true }).fill('Stream until this browser presses Stop.');
    await page.getByRole('button', { name: 'Send', exact: true }).click();
    await expect.poll(async () => (await catalog(request, ctx)).find(item => item.identity.id === ready.identity.id)?.lifecycle.active_requests, { timeout: 30000 }).toBeGreaterThan(0);
    await page.getByRole('button', { name: 'Stop', exact: true }).click();
    await expect(page.locator('.chat-turn header')).toContainText('cancelled');
    await expect.poll(async () => (await catalog(request, ctx)).find(item => item.identity.id === ready.identity.id)?.lifecycle.active_requests, { timeout: 30000 }).toBe(0);

    const afterStop = requireCatalogItem((await catalog(request, ctx)).find(item => item.identity.id === ready.identity.id), 'post-stop fake');
    const unloadAccepted = await apiJson<{ operation_id: string }>(request, ctx, 'post', '/ui-api/v1/model-actions', { model_id: afterStop.identity.id, action: 'unload', expected_revision: afterStop.identity.revision, idempotency_key: 'router-real-unload' });
    await waitOperation(request, ctx, unloadAccepted.operation_id);
    await expect.poll(async () => (await catalog(request, ctx)).find(item => item.identity.id === ready.identity.id)?.lifecycle.worker_exit_observed).toBe(true);
    const unloaded = requireCatalogItem((await catalog(request, ctx)).find(item => item.identity.id === ready.identity.id), 'unloaded fake');
    const removeAccepted = await apiJson<{ operation_id: string }>(request, ctx, 'post', '/ui-api/v1/model-removals', { model_id: unloaded.identity.id, expected_revision: unloaded.identity.revision, confirm_model_id: unloaded.identity.id, idempotency_key: 'router-real-remove' });
    await waitOperation(request, ctx, removeAccepted.operation_id);
    await expect.poll(async () => (await catalog(request, ctx)).some(item => item.identity.id === ready.identity.id)).toBe(false);

    const unauth = await request.get(`${ctx.target.origin}${ctx.apiBase}/ui-api/v1/catalog`);
    expect(unauth.status()).toBe(401);
    const badBearer = await request.get(`${ctx.target.origin}${ctx.apiBase}/ui-api/v1/catalog`, { headers: { Authorization: 'Bearer wrong-router-key' } });
    expect(badBearer.status()).toBe(401);
    const hostile = await request.post(`${ctx.target.origin}${ctx.apiBase}/ui-api/v1/model-actions`, { headers: { ...ctx.auth, Host: 'foreign.invalid', Origin: 'http://foreign.invalid', 'Sec-Fetch-Site': 'cross-site' }, data: {} });
    expect(hostile.status()).toBe(403);
    const legacyHostile = await request.post(`${ctx.target.origin}${ctx.apiBase}/v1/chat/completions`, { headers: { ...ctx.auth, Host: 'foreign.invalid', Origin: 'http://foreign.invalid', 'Sec-Fetch-Site': 'cross-site' }, data: {} });
    expect(legacyHostile.status()).toBe(403);
    const missingAsset = await request.get(`${ctx.target.origin}${ctx.apiBase}/webui/assets/not-present.js`);
    expect(missingAsset.status()).toBe(404);

    const storageDump = await page.evaluate(() => JSON.stringify({ local: { ...localStorage }, session: { ...sessionStorage } }));
    expect(storageDump).not.toContain(ctx.token);
    const violations = await page.evaluate(() => Reflect.get(window, '__routerCspViolations'));
    expect(violations).toEqual([]); expect(external).toEqual([]);
    await expectSafeLayout(page); await expectAxeClean(page);
    saveArtifact(ctx, 'router-real-evidence.json', { csp, external, violations, stop_drain: { observed_active_request: true, final_active_requests: 0 }, negative_cases: ['terminal operation cancel unsupported', 'stale revision 409', 'unknown field 400', 'unauth 401', 'bad bearer 401', 'hostile UI and legacy routes 403', 'missing asset 404', 'no token persistence'], final_catalog_size: (await catalog(request, ctx)).length, api_base: ctx.apiBase, note: 'Fake model/downloader leaves only; secured Rust router, embedded bundle, auth, CSP, Stop/drain and lifecycle routes were real.' });
  });
});
