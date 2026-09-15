// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import { mkdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { expect, test, type APIRequestContext, type BrowserContext, type Page } from '@playwright/test';
import { expectAxeClean, expectSafeLayout } from './browser-assertions';

type Operation = { operation_id: string; kind: string; state: string; result?: unknown; error?: unknown };
type CatalogItem = { identity: { id: string; display_name: string; inference_id: string; revision: number }; lifecycle: { state: string; worker_exit_observed: boolean; active_requests: number }; capabilities: Array<{ task: string; phase: string; available: boolean }> };
type HarnessContext = { target: URL; apiBase: string; artifacts: string; keyPath: string; token: string; auth: { Authorization: string }; modelsDir: string };
type ErrorEnvelope = { error: { code: string; operation_id?: string | null; field_errors?: Array<{ field: string; code: string; message: string }> | null } };
type DownloadResult = { repo_id?: string; revision?: string | null; model_id?: string | null };
type Accepted = { operation_id: string };
type BrowserJsonResponse<T> = { status: number; body: T };
type HttpObservation = { method: string; path: string; status?: number; failure?: string };

const SUCCESS_DOWNLOAD_REPO = 'mlx-community/Router-Harness-Fake-4bit';
const FAILING_DOWNLOAD_REPO = 'mlx-community/Router-Harness-Fail-4bit';
const CANCELLABLE_DOWNLOAD_REPO = 'mlx-community/Router-Harness-Slow-4bit';

function harnessContext(): HarnessContext {
  const rawTarget = process.env.MLXCEL_WEBUI_ROUTER_URL;
  const artifacts = process.env.MLXCEL_WEBUI_ROUTER_ARTIFACTS;
  const keyPath = process.env.MLXCEL_WEBUI_ROUTER_KEY_FILE;
  const modelsDir = process.env.MLXCEL_WEBUI_ROUTER_MODELS_DIR;
  if (!rawTarget || !artifacts || !keyPath || !modelsDir) throw new Error('Set MLXCEL_WEBUI_ROUTER_URL, MLXCEL_WEBUI_ROUTER_KEY_FILE, MLXCEL_WEBUI_ROUTER_ARTIFACTS and MLXCEL_WEBUI_ROUTER_MODELS_DIR; missing real-router setup is not a skipped pass.');
  const target = new URL(rawTarget);
  if (!['http:', 'https:'].includes(target.protocol) || !['127.0.0.1', 'localhost', '[::1]'].includes(target.hostname) || target.username || target.password || !target.pathname.endsWith('/webui/')) throw new Error('The Rust-router harness must target a loopback /webui/ URL without URL credentials.');
  const token = readFileSync(keyPath, 'utf8').trim();
  return { target, apiBase: target.pathname.slice(0, -'/webui/'.length), artifacts, keyPath, token, auth: { Authorization: `Bearer ${token}` }, modelsDir };
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

async function submitDownload(request: APIRequestContext, ctx: HarnessContext, repoId: string, idempotencyKey: string): Promise<Accepted> {
  return apiJson<Accepted>(request, ctx, 'post', '/ui-api/v1/downloads', { repo_id: repoId, idempotency_key: idempotencyKey });
}

async function waitOperationState(request: APIRequestContext, ctx: HarnessContext, operationId: string, expected: 'succeeded' | 'failed' | 'cancelled' = 'succeeded'): Promise<Operation> {
  for (let attempt = 0; attempt < 200; attempt += 1) {
    const operation = await apiJson<Operation>(request, ctx, 'get', `/ui-api/v1/operations/${operationId}`);
    if (['succeeded', 'failed', 'cancelled'].includes(operation.state)) {
      expect(operation.state, JSON.stringify(operation)).toBe(expected);
      return operation;
    }
    await new Promise(resolve => setTimeout(resolve, 25));
  }
  throw new Error(`operation ${operationId} did not reach a terminal state`);
}

async function waitOperation(request: APIRequestContext, ctx: HarnessContext, operationId: string): Promise<Operation> {
  return waitOperationState(request, ctx, operationId, 'succeeded');
}

async function waitOperationKind(request: APIRequestContext, ctx: HarnessContext, kind: string, exclude: ReadonlySet<string> = new Set()): Promise<string> {
  for (let attempt = 0; attempt < 200; attempt += 1) {
    const operations = (await apiJson<{ items: Operation[] }>(request, ctx, 'get', '/ui-api/v1/operations')).items;
    const operation = operations.find(item => item.kind === kind && !exclude.has(item.operation_id));
    if (operation) return operation.operation_id;
    await new Promise(resolve => setTimeout(resolve, 25));
  }
  throw new Error(`operation kind ${kind} was not observed`);
}

async function operationIds(request: APIRequestContext, ctx: HarnessContext): Promise<Set<string>> {
  return new Set((await apiJson<{ items: Operation[] }>(request, ctx, 'get', '/ui-api/v1/operations')).items.map(item => item.operation_id));
}

function requireCatalogItem(item: CatalogItem | undefined, label: string): CatalogItem {
  if (!item) throw new Error(`${label} catalog entry not found`);
  return item;
}

async function catalog(request: APIRequestContext, ctx: HarnessContext): Promise<CatalogItem[]> {
  const body = await apiJson<{ items: CatalogItem[] }>(request, ctx, 'get', '/ui-api/v1/catalog?limit=200');
  return body.items;
}

async function login(page: Page, ctx: HarnessContext, observations: HttpObservation[], artifactName = 'router-login-observations.json'): Promise<void> {
  await page.goto(`${ctx.target.href}#models`);
  await page.getByLabel(/Session key|세션 키/i).fill(ctx.token);
  await page.getByRole('button', { name: /Connect|연결/i }).click();
  await page.waitForTimeout(750);
  saveArtifact(ctx, artifactName, { api_base: ctx.apiBase, current_path: new URL(page.url()).pathname, observations });
  await expect(page.getByTestId('models-table')).toBeVisible();
}

async function refresh(page: Page): Promise<void> {
  await page.getByRole('button', { name: 'Refresh server state', exact: true }).first().click();
}

async function rescanLocalRoots(page: Page, request: APIRequestContext, ctx: HarnessContext): Promise<string> {
  const before = await operationIds(request, ctx);
  await page.getByTestId('models-rescan').click();
  const operationId = await waitOperationKind(request, ctx, 'catalog_refresh', before);
  await waitOperation(request, ctx, operationId);
  await refresh(page);
  return operationId;
}

function createLocalSeedModel(ctx: HarnessContext): void {
  const seed = join(ctx.modelsDir, 'seed-local');
  mkdirSync(seed, { recursive: true });
  writeFileSync(join(seed, 'config.json'), '{"model_type":"llama","architectures":["LlamaForCausalLM"],"quantization_config":{"bits":4}}', { mode: 0o600, flag: 'wx' });
  writeFileSync(join(seed, 'model.safetensors'), 'seed weights', { mode: 0o600, flag: 'wx' });
}

async function exerciseDownloadFailureAndCancel(request: APIRequestContext, ctx: HarnessContext): Promise<{ failed: string; cancelled: string }> {
  const failing = await submitDownload(request, ctx, FAILING_DOWNLOAD_REPO, 'router-real-download-failure');
  await waitOperationState(request, ctx, failing.operation_id, 'failed');

  const slow = await submitDownload(request, ctx, CANCELLABLE_DOWNLOAD_REPO, 'router-real-download-cancel');
  await expect.poll(async () => (await catalog(request, ctx)).some(item => item.identity.inference_id === CANCELLABLE_DOWNLOAD_REPO)).toBe(true);
  const cancel = await request.post(`${ctx.target.origin}${ctx.apiBase}/ui-api/v1/operations/${slow.operation_id}/cancel`, { headers: ctx.auth });
  expect(cancel.status()).toBe(202);
  await waitOperationState(request, ctx, slow.operation_id, 'cancelled');
  return { failed: failing.operation_id, cancelled: slow.operation_id };
}

async function browserModelAction(page: Page, ctx: HarnessContext, body: unknown): Promise<BrowserJsonResponse<Accepted | ErrorEnvelope>> {
  return page.evaluate(async ({ url, token, body }) => {
    const response = await fetch(url, {
      method: 'POST',
      headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    return { status: response.status, body: await response.json() as Accepted | ErrorEnvelope };
  }, { url: `${ctx.target.origin}${ctx.apiBase}/ui-api/v1/model-actions`, token: ctx.token, body });
}

function operationIdFrom(body: Accepted | ErrorEnvelope): string {
  if ('operation_id' in body && typeof body.operation_id === 'string') return body.operation_id;
  throw new Error(`expected operation id body, got ${JSON.stringify(body)}`);
}

async function triggerLostCatalogRefresh(page: Page, request: APIRequestContext, ctx: HarnessContext): Promise<string> {
  const url = `${ctx.target.origin}${ctx.apiBase}/ui-api/v1/catalog/refresh`;
  let accepted: string | null = null;
  await page.route(url, async route => {
    const response = await route.fetch();
    const body = await response.json() as Accepted;
    accepted = body.operation_id;
    await route.abort('failed');
  }, { times: 1 });
  const browserResult = await page.evaluate(async ({ url, token }) => {
    try {
      await fetch(url, { method: 'POST', headers: { Authorization: `Bearer ${token}` } });
      return 'unexpected-success';
    } catch (error) {
      return error instanceof Error ? error.name : String(error);
    }
  }, { url, token: ctx.token });
  expect(browserResult).not.toBe('unexpected-success');
  if (accepted === null) throw new Error('catalog refresh POST did not reach the server before the response was lost');
  await waitOperation(request, ctx, accepted);
  return accepted;
}

async function injectRealUnauthorizedCatalog(page: Page, ctx: HarnessContext): Promise<void> {
  await page.route(`${ctx.target.origin}${ctx.apiBase}/ui-api/v1/catalog**`, async route => {
    await route.continue({ headers: { ...route.request().headers(), authorization: 'Bearer expired-router-key' } });
  }, { times: 1 });
  await refresh(page);
  await expect(page.getByLabel(/Session key|세션 키/i)).toBeVisible();
}

async function performTwoTabStaleRevision(
  context: BrowserContext,
  request: APIRequestContext,
  ctx: HarnessContext,
  ready: CatalogItem,
): Promise<CatalogItem> {
  const second = await context.newPage();
  const observations: HttpObservation[] = [];
  try {
    await login(second, ctx, observations, 'router-login-observations-tab-b.json');
    const unload = await browserModelAction(second, ctx, { model_id: ready.identity.id, action: 'unload', expected_revision: ready.identity.revision, idempotency_key: 'router-real-tab-b-unload' });
    expect(unload.status).toBe(202);
    await waitOperation(request, ctx, operationIdFrom(unload.body));
    const unloaded = requireCatalogItem((await catalog(request, ctx)).find(item => item.identity.id === ready.identity.id), 'tab-b unloaded fake');
    const staleFromTabA = await request.post(`${ctx.target.origin}${ctx.apiBase}/ui-api/v1/model-actions`, { headers: ctx.auth, data: { model_id: ready.identity.id, action: 'unload', expected_revision: ready.identity.revision, idempotency_key: 'router-real-tab-a-stale-after-tab-b' } });
    expect(staleFromTabA.status()).toBe(409);
    expect(((await staleFromTabA.json()) as ErrorEnvelope).error.code).toBe('stale_revision');
    const reload = await browserModelAction(second, ctx, { model_id: unloaded.identity.id, action: 'load', expected_revision: unloaded.identity.revision, idempotency_key: 'router-real-tab-b-reload' });
    expect(reload.status).toBe(202);
    await waitOperation(request, ctx, operationIdFrom(reload.body));
    await expect.poll(async () => (await catalog(request, ctx)).find(item => item.identity.id === ready.identity.id)?.capabilities.some(cap => cap.task === 'chat' && cap.phase === 'provider_ready' && cap.available)).toBe(true);
    return requireCatalogItem((await catalog(request, ctx)).find(item => item.identity.id === ready.identity.id), 'tab-b reloaded fake');
  } finally {
    await second.close();
  }
}

test.describe('Rust router harness', () => {
  test('secured Rust router drives download, load, chat, Stop/drain, unload, remove and negative security cases', async ({ page, request, context }) => {
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
    await expect.poll(async () => (await catalog(request, ctx)).length).toBe(0);
    createLocalSeedModel(ctx);
    const seedRefreshOperation = await rescanLocalRoots(page, request, ctx);
    await expect.poll(async () => (await catalog(request, ctx)).some(item => item.identity.inference_id === 'seed-local')).toBe(true);
    await expectSafeLayout(page); await expectAxeClean(page);
    const downloadNegatives = await exerciseDownloadFailureAndCancel(request, ctx);

    const operationsBeforeDownload = await operationIds(request, ctx);
    await page.getByTestId('models-add').click();
    await page.getByTestId('models-repo').fill(SUCCESS_DOWNLOAD_REPO);
    await page.getByTestId('models-public-repo').check();
    await page.getByTestId('models-download-submit').click();
    const acceptedDownload = await waitOperationKind(request, ctx, 'download', operationsBeforeDownload);
    const downloadOperation = await waitOperation(request, ctx, acceptedDownload);
    await refresh(page);
    await expect.poll(async () => (await catalog(request, ctx)).length).toBeGreaterThan(1);

    const downloadResult = downloadOperation.result as DownloadResult | undefined;
    const downloaded = requireCatalogItem((await catalog(request, ctx)).find(item => item.identity.id === downloadResult?.model_id) ?? (await catalog(request, ctx)).find(item => item.identity.inference_id === SUCCESS_DOWNLOAD_REPO), 'downloaded fake');
    await page.getByRole('button', { name: new RegExp(`Inspect .*${downloaded.identity.display_name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}`) }).click();
    await page.getByTestId('models-load').click();
    const loadAccepted = await waitOperationKind(request, ctx, 'model_load');
    await waitOperation(request, ctx, loadAccepted);
    const cancelFinished = await request.post(`${ctx.target.origin}${ctx.apiBase}/ui-api/v1/operations/${loadAccepted}/cancel`, { headers: ctx.auth });
    expect(cancelFinished.status()).toBe(422);
    await refresh(page);
    await expect.poll(async () => (await catalog(request, ctx)).find(item => item.identity.id === downloaded.identity.id)?.capabilities.some(cap => cap.task === 'chat' && cap.phase === 'provider_ready' && cap.available)).toBe(true);

    let ready = requireCatalogItem((await catalog(request, ctx)).find(item => item.identity.id === downloaded.identity.id), 'ready fake');
    const lostRefreshOperation = await triggerLostCatalogRefresh(page, request, ctx);
    ready = await performTwoTabStaleRevision(context, request, ctx, ready);
    await injectRealUnauthorizedCatalog(page, ctx);
    await login(page, ctx, httpObservations, 'router-relogin-observations.json');
    await refresh(page);
    ready = requireCatalogItem((await catalog(request, ctx)).find(item => item.identity.id === ready.identity.id), 'ready fake after auth recovery');
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
    let observedActiveRequests = 0;
    await expect.poll(async () => {
      const active = (await catalog(request, ctx)).find(item => item.identity.id === ready.identity.id)?.lifecycle.active_requests ?? 0;
      observedActiveRequests = Math.max(observedActiveRequests, active);
      return active;
    }, { timeout: 30000 }).toBeGreaterThan(0);
    const stopClickedAt = new Date().toISOString();
    await page.getByRole('button', { name: 'Stop', exact: true }).click();
    await expect(page.locator('.chat-turn header')).toContainText('cancelled');
    await expect.poll(async () => (await catalog(request, ctx)).find(item => item.identity.id === ready.identity.id)?.lifecycle.active_requests, { timeout: 30000 }).toBe(0);
    const stopSettledAt = new Date().toISOString();
    const finalActiveRequests = (await catalog(request, ctx)).find(item => item.identity.id === ready.identity.id)?.lifecycle.active_requests ?? -1;

    const afterStop = requireCatalogItem((await catalog(request, ctx)).find(item => item.identity.id === ready.identity.id), 'post-stop fake');
    const unloadAccepted = await apiJson<{ operation_id: string }>(request, ctx, 'post', '/ui-api/v1/model-actions', { model_id: afterStop.identity.id, action: 'unload', expected_revision: afterStop.identity.revision, idempotency_key: 'router-real-unload' });
    await waitOperation(request, ctx, unloadAccepted.operation_id);
    await expect.poll(async () => (await catalog(request, ctx)).find(item => item.identity.id === ready.identity.id)?.lifecycle.worker_exit_observed).toBe(true);
    const unloaded = requireCatalogItem((await catalog(request, ctx)).find(item => item.identity.id === ready.identity.id), 'unloaded fake');
    const removeAccepted = await apiJson<{ operation_id: string }>(request, ctx, 'post', '/ui-api/v1/model-removals', { model_id: unloaded.identity.id, expected_revision: unloaded.identity.revision, idempotency_key: 'router-real-remove' });
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
    saveArtifact(ctx, 'router-real-evidence.json', { csp, external, violations, seed_refresh_operation_id: seedRefreshOperation, download_negative_operations: downloadNegatives, lost_response_operation_id: lostRefreshOperation, stop_drain: { observed_active_requests: observedActiveRequests, final_active_requests: finalActiveRequests, stop_clicked_at: stopClickedAt, settled_at: stopSettledAt }, negative_cases: ['empty library before local seed', 'local models-dir seed discovered by real catalog refresh', 'terminal operation cancel unsupported', 'two-tab stale revision 409', 'single-tab stale revision 409', 'unknown field 400', 'real 401 after credential change and reconnect', 'download failure operation failed', 'download cancellation operation cancelled', 'lost POST response recovered by operation id', 'unauth 401', 'bad bearer 401', 'hostile UI and legacy routes 403', 'missing asset 404', 'no token persistence'], final_catalog_size: (await catalog(request, ctx)).length, api_base: ctx.apiBase, note: 'Fake model/downloader leaves only; secured Rust router, embedded bundle, auth, CSP, Stop/drain and lifecycle routes were real.' });
  });
});
