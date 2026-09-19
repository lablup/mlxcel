import { expect, test, type Locator, type Page } from '@playwright/test';
import catalog from '../../tests/fixtures/webui/examples/catalog.page.json' with { type: 'json' };
import bootstrap from '../../tests/fixtures/webui/examples/bootstrap.model-free.json' with { type: 'json' };
import runtime from '../../tests/fixtures/webui/examples/runtime.snapshot.json' with { type: 'json' };
import { bootProduct, browserStorageDump, installMockApi, loginWithMockApi, productVariants, readyCatalog, type CatalogPage } from './browser-fixtures';
import { expectAxeClean, expectSafeLayout } from './browser-assertions';
import { fixtureString, localizedPair, type FixtureLocale } from './strings-fixture';

const sse = (frames: unknown[]): string => frames.map((frame) => `data: ${JSON.stringify(frame)}\n\n`).join('') + 'data: [DONE]\n\n';
const shortReply = (text: string): string => sse([{ choices: [{ index: 0, delta: { content: text }, finish_reason: 'stop' }] }, { choices: [], usage: { prompt_tokens: 5, completion_tokens: 12, total_tokens: 17 } }]);

async function openChat(page: Page, width: number, height = 900, locale: FixtureLocale = 'en'): Promise<void> {
  await bootProduct(page, { ...productVariants[0], name: 'chat', width, height, appearance: { ...productVariants[0].appearance, locale } });
  await loginWithMockApi(page);
  await page.evaluate(() => { window.location.hash = 'chat'; });
  await expect(page.getByTestId('chat-title')).toBeVisible();
}
// Axe samples colors, so let a drawer finish its fade before checking it (reduced motion skips the fade).
async function settled(locator: Locator): Promise<void> {
  await locator.evaluate(async (element) => { await Promise.all((element.closest('.drawer__backdrop') ?? element).getAnimations({ subtree: true }).map((animation) => animation.finished)); });
}
// Selecting a model observes its runtime; answer for whichever model was asked about.
async function runtimeRoute(page: Page): Promise<void> {
  await page.route('**/ui-api/v1/runtime?**', async (route) => {
    const modelId = new URL(route.request().url()).searchParams.get('model_id') ?? catalog.items[0].identity.id;
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ ...runtime, server_instance_id: bootstrap.server.server_instance_id, model_id: modelId }) });
  });
}

for (const width of [1440, 390]) {
  test(`chat safe transcript, IME and privacy at ${width}`, async ({ page }, testInfo) => {
    await installMockApi(page, 'happy');
    await page.route('**/ui-api/v1/catalog*', async (route) => {
      const fixture = structuredClone(catalog);
      const entry = fixture.items[0];
      entry.lifecycle.state = 'ready'; entry.lifecycle.worker_exit_observed = false;
      entry.capabilities = [{task:'chat',phase:'provider_ready',available:true,reason:null}];
      fixture.server_instance_id = bootstrap.server.server_instance_id;
      await route.fulfill({status:200,contentType:'application/json',body:JSON.stringify(fixture)});
    });
    await runtimeRoute(page);
    const requests: unknown[] = [];
    let remoteRequests = 0;
    page.on('request', (request) => { if (request.url().startsWith('https://untrusted.invalid')) remoteRequests++; });
    await page.route('**/v1/chat/completions?autoload=false', async (route) => {
      const body: unknown = route.request().postDataJSON(); requests.push(body);
      const content = '<script>window.chatXss=true</script>\n\n![blocked](https://untrusted.invalid/track.png)\n\n[bad](javascript:alert(1))\n\n```js\nconst value = 42;\n```\n\n' + 'token '.repeat(10000);
      const frames = [
        {choices:[{index:0,delta:{reasoning_content:'Reasoning is separate.'},finish_reason:null}]},
        {choices:[{index:0,delta:{content},finish_reason:null}]},
        {choices:[{index:0,delta:{},finish_reason:'length'}]},
        {choices:[],usage:{prompt_tokens:5,completion_tokens:10000,total_tokens:10005}},
      ];
      await route.fulfill({status:200,contentType:'text/event-stream',body:sse(frames)});
    });
    const variant={...productVariants[width===390 ? 2 : 0],width};
    // The 390 variant boots in Korean, so every accessible name comes from the string fixture.
    const locale = variant.appearance.locale as FixtureLocale;
    const text = (key: string, values?: Record<string, string>): string => fixtureString(locale, key, values);
    await bootProduct(page, variant); await loginWithMockApi(page);
    await page.evaluate(()=>{window.location.hash='chat';});
    const picker = page.getByRole('combobox', {name:text('chat.model.label')});
    await picker.click(); await page.getByRole('option',{name:`alpha. ${text('chat.model.group.ready')}`,exact:true}).click();
    const composer=page.getByRole('textbox',{name:text('chat.composer.label'),exact:true});
    await composer.fill('안녕하세요');
    await composer.dispatchEvent('compositionstart');
    await composer.dispatchEvent('keydown',{key:'Enter',isComposing:true});
    expect(requests).toHaveLength(0);
    await composer.dispatchEvent('compositionend');
    await composer.press('Shift+Enter'); expect(requests).toHaveLength(0);
    const before=Date.now(); await page.getByRole('button',{name:text('common.send'),exact:true}).click();
    await expect(page.getByRole('status').filter({hasText:text('chat.announce.complete')})).toBeVisible();
    const elapsed=Date.now()-before;
    await testInfo.attach('10000-token-fixture-render.json',{body:JSON.stringify({elapsed_ms:elapsed,kind:'browser fixture end-to-end render, not inference performance'}),contentType:'application/json'});
    expect(requests).toHaveLength(1);expect(requests[0]).toMatchObject({model:'alpha',stream:true});
    expect(remoteRequests).toBe(0);expect(await page.evaluate(()=>Object.hasOwn(window,'chatXss'))).toBe(false);
    await expect(page.locator('.chat-turn img')).toHaveCount(0);
    await expect(page.locator('a[href^="javascript:"]')).toHaveCount(0);
    await expect(page.getByText('Reasoning is separate.',{exact:true})).toBeHidden();
    // Roles read without the text: the prompt hugs the inline end under "You", the answer
    // starts at the inline start under the model badge, and the finished state is a label.
    await expect(page.locator('.chat-message--user .chat-role')).toHaveText(text('chat.message.you'));
    await expect(page.locator('.chat-message--assistant .chat-message-meta .badge').first()).toHaveText('alpha');
    await expect(page.locator('.chat-status')).toHaveText(text('chat.transcript.status.complete'));
    const sides = await page.evaluate(() => {
      const box = (selector: string): DOMRect => document.querySelector(selector)?.getBoundingClientRect() ?? new DOMRect();
      const list = box('.chat-messages'); const user = box('.chat-message--user .chat-bubble'); const assistant = box('.chat-message--assistant .chat-bubble');
      return { userEnd: list.right - user.right, userStart: user.left - list.left, assistantStart: assistant.left - list.left };
    });
    expect(sides.userEnd).toBeLessThan(32); expect(sides.userStart).toBeGreaterThan(64); expect(sides.assistantStart).toBeLessThan(32);
    await expectAxeClean(page);await expectSafeLayout(page);
    expect(await browserStorageDump(page)).not.toContain('안녕하세요');
    // Conversation settings, next-turn parameters and local history sit behind one header control.
    const settings = page.getByRole('button',{name:text('chat.settings.title'),exact:true});
    await settings.click();
    const drawer = page.getByTestId('chat-settings-drawer');
    await expect(drawer).toBeVisible(); await settled(drawer);
    for (const key of ['chat.settings.summary', 'chat.params.summary', 'chat.privacy.summary']) await expect(drawer.getByRole('heading',{name:text(key),exact:true})).toBeVisible();
    await expect(drawer.getByLabel(text('chat.settings.name'),{exact:true})).toBeVisible();
    await expect(drawer.getByLabel(text('chat.settings.system_prompt'),{exact:true})).toBeVisible();
    await expect(drawer.getByLabel(text('chat.params.field',{name:'max_tokens'}),{exact:true})).toBeVisible();
    await expect(drawer.getByRole('checkbox',{name:text('chat.privacy.save'),exact:true})).toBeVisible();
    if (locale === 'ko') for (const key of ['chat.intro', 'chat.params.summary', 'chat.privacy.summary']) {
      const { shown, hidden } = localizedPair(locale, key);
      await expect(page.getByText(shown, {exact:true}).first()).toBeVisible();
      await expect(page.getByText(hidden, {exact:true})).toHaveCount(0);
    }
    await expectAxeClean(page);await expectSafeLayout(page);
    await page.keyboard.press('Escape');
    await expect(drawer).toBeHidden(); await expect(settings).toBeFocused();
    // No disclosure is left in the conversation column; only a message's own reasoning.
    await expect(page.locator('.chat-screen details:not(.chat-reasoning)')).toHaveCount(0);
    // Edit asks through the design-system dialog; Escape keeps the turn and returns focus.
    const edit = page.getByRole('button',{name:text('chat.transcript.edit'),exact:true});
    if (locale === 'ko') await expect(page.getByRole('button',{name:localizedPair(locale,'chat.transcript.edit').hidden,exact:true})).toHaveCount(0);
    await edit.click();
    const editDialog = page.getByTestId('chat-edit-dialog');
    await expect(editDialog).toBeVisible(); await expect(editDialog).toContainText(text('chat.transcript.edit.confirm.body'));
    await page.keyboard.press('Escape');
    await expect(editDialog).toBeHidden(); await expect(edit).toBeFocused(); await expect(page.locator('.chat-turn')).toHaveCount(1);
    await testInfo.attach(`chat-${width}.png`,{body:await page.screenshot({fullPage:true}),contentType:'image/png'});
    await page.getByRole('button',{name:text('chat.new_conversation'),exact:true}).click();
    await expect(composer).toBeEnabled();
    await expect(page.locator('.chat-turn')).toHaveCount(0);
    // Switching back is one click in the list: a pane beside the conversation from 961 px,
    // a Drawer opened from the header below it, which closes on selection.
    if (width === 390) {
      await expect(page.getByRole('navigation',{name:text('chat.list.label'),exact:true})).toHaveCount(0);
      await page.getByRole('button',{name:text('chat.list.open'),exact:true}).click();
      const listDrawer = page.getByTestId('chat-list-drawer');
      await expect(listDrawer).toBeVisible(); await settled(listDrawer);
      await expectAxeClean(page); await expectSafeLayout(page);
      await testInfo.attach('chat-390-list.png',{body:await page.screenshot(),contentType:'image/png'});
      await listDrawer.locator('button.chat-row-select').nth(1).click();
      await expect(listDrawer).toBeHidden();
    } else {
      await page.getByRole('navigation',{name:text('chat.list.label'),exact:true}).locator('button.chat-row-select').nth(1).click();
    }
    await expect(page.locator('.chat-turn')).toHaveCount(1);
    await expect(page.locator('button.chat-row-select[aria-current="true"]')).toContainText('안녕하세요');
  });
}

test('chat picker lists the Ready chat model first and loads only after a confirmation', async ({ page }) => {
  // alpha is Ready. The unloaded model is a second entry (the fixture's running operation
  // names alpha as its eviction target, so alpha itself could not load), and its name sorts
  // first, so its position after alpha comes from the Ready-first grouping alone.
  const unloadedId = `mdl_${'u'.repeat(43)}`;
  const revision = catalog.items[0].identity.revision;
  await installMockApi(page, 'happy', { catalog: (fixture: CatalogPage) => {
    const [ready] = readyCatalog(fixture).items;
    const unloaded = { ...structuredClone(catalog.items[0]), identity: { ...catalog.items[0].identity, id: unloadedId, display_name: 'able-unloaded', inference_id: 'able-unloaded' } };
    fixture.items = [unloaded, ready];
    fixture.pagination = { ...fixture.pagination, total_known: 2 };
    return fixture;
  } });
  const posts: Array<{ url: string; body: unknown }> = [];
  page.on('request', (request) => { if (request.method() !== 'GET') posts.push({ url: new URL(request.url()).pathname, body: request.postDataJSON() as unknown }); });
  await runtimeRoute(page);
  await page.route('**/ui-api/v1/model-actions', async (route) => {
    await route.fulfill({ status: 202, contentType: 'application/json', body: JSON.stringify({ operation_id: 'op_model_load_000001', state: 'queued', idempotent_replay: false }) });
  });
  await bootProduct(page, { ...productVariants[0], name: 'chat-load', width: 1440, height: 900 });
  await page.getByLabel(/Session key/i).fill('good-key');
  await page.getByRole('button', { name: /Connect/i }).click();
  await expect(page.getByTestId('connection-authenticated-detail')).toContainText(/catalog 2/i);
  await page.evaluate(() => { window.location.hash = 'chat'; });
  await expect(page.getByTestId('chat-title')).toBeVisible();
  posts.length = 0;
  await page.getByRole('combobox', { name: 'Model for next turn' }).click();
  const options = page.getByRole('option');
  // The placeholder leads only while nothing is selected; the first model is the Ready one.
  await expect(options.nth(0)).toHaveText('Select a model');
  await expect(options.nth(1)).toHaveAccessibleName('alpha. Ready to chat');
  await expect(options.nth(2)).toHaveAccessibleName('able-unloaded. Not loaded');
  await options.nth(2).click();
  await expect(page.getByTestId('chat-load')).toBeVisible();
  await expect(page.getByTestId('chat-model-status')).toHaveText('Not loaded. Load it to chat; selecting a model never loads it.');
  expect(posts).toEqual([]);
  await expect(page.getByRole('button', { name: 'Send', exact: true })).toBeDisabled();
  await page.getByTestId('chat-load').click();
  const dialog = page.getByTestId('chat-load-dialog');
  await expect(dialog).toBeVisible();
  await expect(dialog).toContainText('Load able-unloaded?');
  expect(posts).toEqual([]);
  await expectAxeClean(page);
  await page.getByTestId('chat-load-dialog-confirm').click();
  await expect.poll(() => posts.length).toBe(1);
  await expect(page.getByTestId('models-confirm')).toHaveCount(0);
  expect(posts[0].url).toBe('/ui-api/v1/model-actions');
  const body = posts[0].body as Record<string, unknown>;
  expect(Object.keys(body).sort()).toEqual(['action', 'expected_revision', 'idempotency_key', 'model_id']);
  expect(body).toMatchObject({ action: 'load', model_id: unloadedId, expected_revision: revision });
  await page.waitForTimeout(300);
  expect(posts).toHaveLength(1);
});

test('chat streaming status counts elapsed seconds until the answer arrives', async ({ page }) => {
  await installMockApi(page, 'happy', { catalog: readyCatalog });
  await runtimeRoute(page);
  let release: () => void = () => undefined;
  const held = new Promise<void>((resolve) => { release = resolve; });
  await page.route('**/v1/chat/completions?autoload=false', async (route) => { await held; await route.fulfill({ status: 200, contentType: 'text/event-stream', body: shortReply('Held reply.') }); });
  await openChat(page, 1440);
  await page.getByRole('combobox', { name: 'Model for next turn' }).click();
  await page.getByRole('option', { name: 'alpha. Ready to chat', exact: true }).click();
  await page.getByRole('textbox', { name: 'Message', exact: true }).fill('Hold this one open.');
  await page.getByRole('button', { name: 'Send', exact: true }).click();
  const status = page.locator('.chat-status[data-status="streaming"]');
  try {
    await expect(status).toHaveText(/^Generating · [0-9]+ s$/);
    await expect(status).toHaveText(/^Generating · [2-9] s$/, { timeout: 6000 });
    await expect(page.getByRole('button', { name: 'Stop', exact: true })).toBeEnabled();
    // The settings drawer's trailing controls are disabled while the answer streams; Tab
    // must still cycle inside the modal drawer rather than reach the page behind it.
    await page.getByRole('button', { name: 'Chat settings', exact: true }).click();
    const drawer = page.getByTestId('chat-settings-drawer');
    const close = drawer.getByRole('button', { name: 'Close', exact: true });
    const lastUsable = drawer.getByRole('button', { name: 'Clear next-turn overrides', exact: true });
    await expect(close).toBeFocused();
    await lastUsable.focus();
    await page.keyboard.press('Tab');
    await expect(close).toBeFocused();
    await page.keyboard.press('Shift+Tab');
    await expect(lastUsable).toBeFocused();
    await page.keyboard.press('Escape');
    await expect(drawer).toBeHidden();
  } finally { release(); }
  await expect(page.locator('.chat-status')).toHaveText('Done');
  await expect(page.locator('.chat-markdown')).toHaveText('Held reply.');
});

test('a 30-turn conversation at 1440x900 keeps the composer pinned and only the messages scroll', async ({ page }, testInfo) => {
  test.setTimeout(120_000);
  await installMockApi(page, 'happy', { catalog: readyCatalog });
  await runtimeRoute(page);
  await page.route('**/v1/chat/completions?autoload=false', async (route) => { await route.fulfill({ status: 200, contentType: 'text/event-stream', body: shortReply('A short fixture reply that spans a couple of sentences, so each assistant message has a realistic height.') }); });
  await openChat(page, 1440, 900);
  await page.getByRole('combobox', { name: 'Model for next turn' }).click();
  await page.getByRole('option', { name: 'alpha. Ready to chat', exact: true }).click();
  const composer = page.getByRole('textbox', { name: 'Message', exact: true });
  for (let index = 1; index <= 30; index++) {
    await composer.fill(`Prompt number ${index}: tell me something short.`);
    await composer.press('Enter');
    await expect(composer).toBeEnabled();
    await expect(composer).toHaveValue('');
  }
  await expect(page.locator('article.chat-turn')).toHaveCount(30);
  const layout = await page.evaluate(() => {
    const field = document.querySelector('textarea[aria-label="Message"]')?.getBoundingClientRect();
    const scrollers = Array.from(document.querySelectorAll<HTMLElement>('*')).filter((node) => node.scrollHeight > node.clientHeight + 1 && ['auto', 'scroll'].includes(getComputedStyle(node).overflowY)).map((node) => node.className || node.tagName);
    return { scrollHeight: document.documentElement.scrollHeight, innerHeight: window.innerHeight, top: field?.top ?? -1, bottom: field?.bottom ?? Number.POSITIVE_INFINITY, scrollers };
  });
  expect(layout.scrollHeight).toBeLessThanOrEqual(layout.innerHeight);
  expect(layout.top).toBeGreaterThanOrEqual(0);
  expect(layout.bottom).toBeLessThanOrEqual(layout.innerHeight);
  expect(layout.scrollers).toEqual(['chat-messages']);
  await expectSafeLayout(page);
  // The composer grows from one to eight rows with its content, then scrolls internally.
  await expect(composer).toHaveAttribute('rows', '1');
  await composer.fill('one\ntwo\nthree');
  await expect(composer).toHaveAttribute('rows', '3');
  await composer.fill(Array.from({ length: 12 }, (_, index) => `line ${index}`).join('\n'));
  await expect(composer).toHaveAttribute('rows', '8');
  await composer.fill('');
  await expect(composer).toHaveAttribute('rows', '1');
  await testInfo.attach('chat-1440-30-turns.png', { body: await page.screenshot(), contentType: 'image/png' });
});
