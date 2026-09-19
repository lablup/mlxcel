import { readFileSync, writeFileSync } from 'node:fs';
import { expect, test } from '@playwright/test';
import { expectAxeClean, expectSafeLayout } from './browser-assertions';

/** Root-owned serialized acceptance. Never run on an arbitrary shared server. */
test('real bundled chat Stop releases the selected model request lease', async ({ page, request }, testInfo) => {
  // Explicit files survive reporter=list and later assertion failures. Only this
  // isolated synthetic acceptance writes generated output; never credentials.
  const saveArtifact = async (name: string, body: string | Buffer, contentType: string): Promise<void> => {
    const path = testInfo.outputPath(name);
    writeFileSync(path, body, { mode: 0o600, flag: 'wx' });
    await testInfo.attach(name, { path, contentType });
  };
  const base = process.env.MLXCEL_CHAT_REAL_URL;
  const keyPath = process.env.MLXCEL_CHAT_REAL_KEY_FILE;
  const modelId = process.env.MLXCEL_CHAT_REAL_MODEL_ID;
  if (!base || !keyPath || !modelId || process.env.MLXCEL_CHAT_REAL_ISOLATED !== '1') throw new Error('Explicit isolated server URL, private key file and already-loaded opaque model ID are required. Missing setup is not a skipped pass.');
  const target = new URL(base);
  if (!['127.0.0.1', '[::1]', 'localhost'].includes(target.hostname) || !target.pathname.endsWith('/webui/')) throw new Error('Real chat acceptance requires a loopback bundled WebUI URL.');
  const apiBase = target.pathname.slice(0, -'/webui/'.length);
  const token = readFileSync(keyPath, 'utf8').trim();
  // The model's own catalog entry: the list is paginated, so a large catalog can put the
  // model beyond the first page.
  const entryUrl = `${target.origin}${apiBase}/ui-api/v1/catalog/${encodeURIComponent(modelId)}`;
  const headers = {Authorization:`Bearer ${token}`};
  const readEntry = async (): Promise<{identity:{id:string;display_name:string};lifecycle:{state:string;active_requests:number}}> => {
    const response = await request.get(entryUrl,{headers});expect(response.ok()).toBe(true);
    return await response.json() as {identity:{id:string;display_name:string};lifecycle:{state:string;active_requests:number}};
  };
  const initial = await readEntry();
  expect(initial?.lifecycle.state).toBe('ready');expect(initial?.lifecycle.active_requests).toBe(0);
  const external: string[] = [];
  page.on('request', request => {
    const url = new URL(request.url());
    if (!['data:', 'blob:'].includes(url.protocol) && url.origin !== target.origin) external.push(request.url());
  });
  await page.addInitScript(() => {
    const violations: string[] = [];
    Object.defineProperty(window, '__chatCspViolations', { value: violations });
    document.addEventListener('securitypolicyviolation', event => violations.push(`${event.effectiveDirective}: ${event.blockedURI}`));
  });
  const response = await page.goto(`${base}#chat`);
  expect(response?.ok()).toBe(true);
  const csp = response?.headers()['content-security-policy'];
  expect(csp).toMatch(/(?:^|;)\s*style-src 'self'(?:;|$)/);
  expect(csp).not.toContain("'unsafe-inline'"); expect(csp).not.toContain("'unsafe-eval'");
  await page.getByLabel(/Session key|세션 키/i).fill(token);
  await page.getByRole('button', {name:/Connect|연결/i}).click();
  await page.getByRole('combobox',{name:'Model for next turn'}).click();
  await page.getByRole('option',{name:`${initial?.identity.display_name}. Ready to chat`, exact:true}).click();
  // Next-turn parameters live in the settings drawer; close it again before composing.
  await page.getByRole('button', {name:'Chat settings', exact:true}).click();
  await page.getByLabel('Next turn max_tokens', { exact: true }).fill('64');
  await page.keyboard.press('Escape');
  const imagePath = process.env.MLXCEL_CHAT_REAL_IMAGE_FILE;
  if (imagePath) await page.locator('input[type="file"][accept^="image/"]').setInputFiles(imagePath);
  await page.getByRole('textbox',{name:'Message',exact:true}).fill(imagePath ? 'Describe what is visible in this image in one short sentence.' : 'Say Hello in one short sentence.');
  await page.getByRole('button',{name:'Send',exact:true}).click();
  await expect(page.locator('.chat-turn .chat-status')).toHaveText('Done', {timeout:90000});
  const reply = await page.locator('.chat-markdown').innerText();
  await saveArtifact('real-response.json', JSON.stringify({model_id:modelId,image_input:Boolean(imagePath),reply,evaluation:'Output captured; root must review whether the content is sensible.'}), 'application/json');
  expect(reply.trim().length).toBeGreaterThan(0);
  await expectSafeLayout(page); await expectAxeClean(page);
  await page.evaluate(() => {
    window.scrollTo(0, 0);
    for (const element of document.querySelectorAll<HTMLElement>('.app-main, .app-content, .app-content-grid')) element.scrollTop = 0;
  });
  await saveArtifact('real-chat-viewport.png', await page.screenshot(), 'image/png');
  await saveArtifact('real-chat-render.png', await page.screenshot({ fullPage: true }), 'image/png');
  await page.getByRole('button',{name:'New conversation',exact:true}).click();
  await page.getByRole('button', {name:'Chat settings', exact:true}).click();
  await page.getByLabel('Next turn max_tokens', { exact: true }).fill('1024');
  await page.keyboard.press('Escape');
  await page.getByRole('textbox',{name:'Message',exact:true}).fill('Write a very long numbered explanation of integers from 1 to 10000, without stopping early.');
  await page.getByRole('button',{name:'Send',exact:true}).click();
  await expect.poll(async()=> (await readEntry()).lifecycle.active_requests,{timeout:30000}).toBeGreaterThan(0);
  await page.getByRole('button',{name:'Stop',exact:true}).click();
  await expect(page.locator('.chat-turn .chat-status')).toHaveText('Stopped');
  await expect.poll(async()=> (await readEntry()).lifecycle.active_requests,{timeout:30000}).toBe(0);
  await saveArtifact('real-stop-evidence.json', JSON.stringify({model_id:modelId,scope:'isolated server with no other request producers',initial_active:0,observed_during_positive:true,final_active:0,automatic_retry:false}), 'application/json');
  await expect(page.getByRole('button',{name:'Send',exact:true})).toBeDisabled(); // Empty composer, not a rerun.
  await expectSafeLayout(page); await expectAxeClean(page);
  const violations = await page.evaluate(() => Reflect.get(window, '__chatCspViolations'));
  await saveArtifact('real-security-evidence.json', JSON.stringify({ csp, violations, external, axe: 'No violations in completed and cancelled states' }), 'application/json');
  expect(violations).toEqual([]); expect(external).toEqual([]);
});
