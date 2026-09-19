import { expect, test } from '@playwright/test';
import catalog from '../../tests/fixtures/webui/examples/catalog.page.json' with { type: 'json' };
import bootstrap from '../../tests/fixtures/webui/examples/bootstrap.model-free.json' with { type: 'json' };
import runtime from '../../tests/fixtures/webui/examples/runtime.snapshot.json' with { type: 'json' };
import { bootProduct, browserStorageDump, installMockApi, loginWithMockApi, productVariants } from './browser-fixtures';
import { expectAxeClean, expectSafeLayout } from './browser-assertions';
import { fixtureString, localizedPair, type FixtureLocale } from './strings-fixture';

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
    await page.route('**/ui-api/v1/runtime?**', async (route) => {
      await route.fulfill({status:200,contentType:'application/json',body:JSON.stringify({...runtime,server_instance_id:bootstrap.server.server_instance_id,model_id:catalog.items[0].identity.id})});
    });
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
      await route.fulfill({status:200,contentType:'text/event-stream',body:frames.map(frame=>`data: ${JSON.stringify(frame)}\n\n`).join('')+'data: [DONE]\n\n'});
    });
    const variant={...productVariants[width===390 ? 2 : 0],width};
    // The 390 variant boots in Korean, so every accessible name comes from the string fixture.
    const locale = variant.appearance.locale as FixtureLocale;
    const text = (key: string): string => fixtureString(locale, key);
    await bootProduct(page, variant); await loginWithMockApi(page);
    await page.evaluate(()=>{window.location.hash='chat';});
    const picker = page.getByRole('combobox', {name:text('chat.model.label')});
    await picker.click(); await page.getByRole('option',{name:`alpha · ${text('models.status.ready')}`}).click();
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
    await expectAxeClean(page);await expectSafeLayout(page);
    expect(await browserStorageDump(page)).not.toContain('안녕하세요');
    if (locale === 'ko') for (const key of ['chat.intro', 'chat.transcript.edit', 'chat.params.summary', 'chat.privacy.summary']) {
      const { shown, hidden } = localizedPair(locale, key);
      await expect(page.getByText(shown, {exact:true}).first()).toBeVisible();
      await expect(page.getByText(hidden, {exact:true})).toHaveCount(0);
    }
    // Edit asks through the design-system dialog; Escape keeps the turn and returns focus.
    const edit = page.getByRole('button',{name:text('chat.transcript.edit'),exact:true});
    await edit.click();
    const editDialog = page.getByTestId('chat-edit-dialog');
    await expect(editDialog).toBeVisible(); await expect(editDialog).toContainText(text('chat.transcript.edit.confirm.body'));
    await page.keyboard.press('Escape');
    await expect(editDialog).toBeHidden(); await expect(edit).toBeFocused(); await expect(page.locator('.chat-turn')).toHaveCount(1);
    await page.getByRole('button',{name:text('chat.new_conversation'),exact:true}).click();
    await expect(composer).toBeEnabled();
    await testInfo.attach(`chat-${width}.png`,{body:await page.screenshot({fullPage:true}),contentType:'image/png'});
  });
}
