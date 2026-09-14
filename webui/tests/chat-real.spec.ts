import { readFileSync } from 'node:fs';
import { expect, test } from '@playwright/test';

/** Root-owned serialized acceptance. Never run on an arbitrary shared server. */
test('real bundled chat Stop releases the selected model request lease', async ({ page, request }, testInfo) => {
  const base = process.env.MLXCEL_CHAT_REAL_URL;
  const keyPath = process.env.MLXCEL_CHAT_REAL_KEY_FILE;
  const modelId = process.env.MLXCEL_CHAT_REAL_MODEL_ID;
  if (!base || !keyPath || !modelId || process.env.MLXCEL_CHAT_REAL_ISOLATED !== '1') throw new Error('Explicit isolated server URL, private key file and already-loaded opaque model ID are required. Missing setup is not a skipped pass.');
  const target = new URL(base);
  if (!['127.0.0.1', '[::1]', 'localhost'].includes(target.hostname) || !target.pathname.endsWith('/webui/')) throw new Error('Real chat acceptance requires a loopback bundled WebUI URL.');
  const apiBase = target.pathname.slice(0, -'/webui/'.length);
  const token = readFileSync(keyPath, 'utf8').trim();
  const catalogUrl = `${target.origin}${apiBase}/ui-api/v1/catalog`;
  const headers = {Authorization:`Bearer ${token}`};
  const readCatalog = async (): Promise<Array<{identity:{id:string;display_name:string};lifecycle:{state:string;active_requests:number}}>> => {
    const response = await request.get(catalogUrl,{headers});expect(response.ok()).toBe(true);
    return (await response.json() as {items:Array<{identity:{id:string;display_name:string};lifecycle:{state:string;active_requests:number}}>}).items;
  };
  const initial = (await readCatalog()).find(entry=>entry.identity.id===modelId);
  expect(initial?.lifecycle.state).toBe('ready');expect(initial?.lifecycle.active_requests).toBe(0);
  await page.goto(`${base}#chat`);
  await page.getByLabel(/Session key|세션 키/i).fill(token);
  await page.getByRole('button', {name:/Connect|연결/i}).click();
  await page.getByRole('combobox',{name:'Model for next turn'}).click();
  await page.getByRole('option',{name:`${initial?.identity.display_name} · ready`}).click();
  const imagePath = process.env.MLXCEL_CHAT_REAL_IMAGE_FILE;
  if (imagePath) await page.getByLabel('Local images', {exact:true}).setInputFiles(imagePath);
  await page.getByRole('textbox',{name:'Message',exact:true}).fill(imagePath ? 'Describe what is visible in this image in one short sentence.' : 'Say Hello in one short sentence.');
  await page.getByRole('button',{name:'Send',exact:true}).click();
  await expect(page.locator('.chat-turn header')).toContainText('complete', {timeout:90000});
  const reply = await page.locator('.chat-markdown').innerText();
  expect(reply.trim().length).toBeGreaterThan(0);
  await testInfo.attach('real-response.json',{body:JSON.stringify({model_id:modelId,image_input:Boolean(imagePath),reply,evaluation:'Nonempty output observed; root must review whether the content is sensible.'}),contentType:'application/json'});
  await page.getByRole('button',{name:'New conversation',exact:true}).click();
  await page.getByRole('textbox',{name:'Message',exact:true}).fill('Write a very long numbered explanation of integers from 1 to 10000, without stopping early.');
  await page.getByRole('button',{name:'Send',exact:true}).click();
  await expect.poll(async()=> (await readCatalog()).find(entry=>entry.identity.id===modelId)?.lifecycle.active_requests,{timeout:30000}).toBeGreaterThan(0);
  await page.getByRole('button',{name:'Stop',exact:true}).click();
  await expect(page.locator('.chat-turn header')).toContainText('cancelled');
  await expect.poll(async()=> (await readCatalog()).find(entry=>entry.identity.id===modelId)?.lifecycle.active_requests,{timeout:30000}).toBe(0);
  await expect(page.getByRole('button',{name:'Send',exact:true})).toBeDisabled(); // Empty composer, not a rerun.
  await testInfo.attach('real-stop-evidence.json',{body:JSON.stringify({model_id:modelId,scope:'isolated server with no other request producers',initial_active:0,observed_during_positive:true,final_active:0,automatic_retry:false}),contentType:'application/json'});
});
