// Copyright 2026 Lablup Inc. Licensed under Apache-2.0.
import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { beforeEach, afterEach, describe, expect, it, vi } from 'vitest';
import bootstrap from '../../../../tests/fixtures/webui/examples/bootstrap.model-free.json';
import catalog from '../../../../tests/fixtures/webui/examples/catalog.page.json';
import { initialSnapshot } from '../../state/reducer';
import type { WebUiSnapshot } from '../../api/types';
import { WebUiHttpError, type ChatStreamHandlers } from '../../api/client';
import { Chat } from './chat';
import { replaceConversations } from './session';
const mocked = vi.hoisted(() => ({ snapshot: null as WebUiSnapshot | null, stream: vi.fn(), select: vi.fn(), runtime: vi.fn() }));
vi.mock('../../state', () => ({useWebUi:()=>mocked.snapshot,useWebUiActions:()=>({streamChatCompletions:mocked.stream,selectModel:mocked.select,refreshRuntime:mocked.runtime})}));
let host: HTMLDivElement, root: Root;
function button(name:string): HTMLButtonElement { const value=Array.from(host.querySelectorAll('button')).find(item=>item.textContent===name);if(!value)throw new Error(`Missing ${name}`);return value; }
function input(text:string): HTMLTextAreaElement {const field=host.querySelector<HTMLTextAreaElement>('textarea[aria-label="Message"]');if(!field)throw new Error('Missing composer');act(()=>{Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype,'value')?.set?.call(field,text);field.dispatchEvent(new Event('input',{bubbles:true}));});return field;}
beforeEach(()=>{
 vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT',true);vi.stubGlobal('confirm',()=>true);replaceConversations([]);
 const entry={...catalog.items[0],lifecycle:{...catalog.items[0].lifecycle,state:'ready'},capabilities:[{task:'chat',phase:'provider_ready',available:true,reason:null}]};
 mocked.snapshot={...initialSnapshot(),auth:{status:'authenticated',tokenPresent:true},bootstrap,connection:'ready',catalog:[entry],selectedModelId:entry.identity.id} as WebUiSnapshot;
 mocked.stream.mockReset();mocked.runtime.mockReset();mocked.runtime.mockResolvedValue({measurements:{}});
 host=document.createElement('div');document.body.append(host);root=createRoot(host);act(()=>root.render(<Chat locale="en"/>));
});
afterEach(()=>{act(()=>root.unmount());host.remove();replaceConversations([]);vi.unstubAllGlobals();});
describe('Chat real component composition',()=>{
 it('sends frozen identity and explicit request parameters, then retains cancelled partial output',async()=>{
  let frame:ChatStreamHandlers|undefined;let signal:AbortSignal|undefined;
  mocked.stream.mockImplementation((_id:string,_body:unknown,handlers:ChatStreamHandlers,upstream:AbortSignal)=>{frame=handlers;signal=upstream;return new Promise<void>((_resolve,reject)=>upstream.addEventListener('abort',()=>reject(new DOMException('aborted','AbortError'))));});
  input('Hello');await act(async()=>button('Send').click());
  expect(mocked.stream.mock.calls[0][0]).toBe(catalog.items[0].identity.id);expect(mocked.stream.mock.calls[0][1]).toMatchObject({model:'alpha',stream:true,messages:[{role:'user',content:'Hello'}],stream_options:{include_usage:true}});
  await act(async()=>frame?.onFrame({data:JSON.stringify({choices:[{index:0,delta:{content:'Partial'},finish_reason:null}]}),event:'message',id:null,retry:null}));
  await act(async()=>button('Stop').click());
  expect(signal?.aborted).toBe(true);expect(host.textContent).toContain('cancelled');expect(host.textContent).toContain('Partial');expect(mocked.stream).toHaveBeenCalledOnce();expect(mocked.runtime).toHaveBeenCalledOnce();
 });
 it('does not send Enter while composing or Shift+Enter',async()=>{
  const field=input('한글');
  await act(async()=>{field.dispatchEvent(new CompositionEvent('compositionstart',{bubbles:true}));field.dispatchEvent(new KeyboardEvent('keydown',{key:'Enter',bubbles:true,isComposing:true}));});expect(mocked.stream).not.toHaveBeenCalled();
  act(()=>{field.dispatchEvent(new CompositionEvent('compositionend',{bubbles:true}));field.dispatchEvent(new KeyboardEvent('keydown',{key:'Enter',shiftKey:true,bubbles:true}));});expect(mocked.stream).not.toHaveBeenCalled();
 });
 it('does not call inference for a response beyond the complete JSON transport budget',async()=>{
  if(mocked.snapshot?.bootstrap)mocked.snapshot={...mocked.snapshot,bootstrap:{...mocked.snapshot.bootstrap,media_limits:{...mocked.snapshot.bootstrap.media_limits,max_body_bytes:10}}};
  act(()=>root.render(<Chat locale="en"/>));input('Hello');await act(async()=>button('Send').click());expect(mocked.stream).not.toHaveBeenCalled();expect(host.textContent).toContain('complete request exceeds');
 });
 it('preserves errored partial output without automatic retry on disconnect',async()=>{
  mocked.stream.mockImplementation(async(_id:string,_body:unknown,handlers:ChatStreamHandlers)=>{handlers.onFrame({data:JSON.stringify({choices:[{index:0,delta:{reasoning_content:'Thinking'},finish_reason:null}]}),event:'message',id:null,retry:null});throw new Error('disconnect');});
  input('Hello');await act(async()=>button('Send').click());expect(host.textContent).toContain('Thinking');expect(host.textContent).toContain('Generation failed or disconnected');expect(mocked.stream).toHaveBeenCalledOnce();
 });
 it('allows text-only generation when server image admission is disabled', async()=>{
  if(mocked.snapshot?.bootstrap)mocked.snapshot={...mocked.snapshot,bootstrap:{...mocked.snapshot.bootstrap,media_limits:{...mocked.snapshot.bootstrap.media_limits,max_images:0}}};
  act(()=>root.render(<Chat locale="en"/>));
  mocked.stream.mockImplementation(async(_id:string,_body:unknown,handlers:ChatStreamHandlers)=>{handlers.onFrame({data:JSON.stringify({choices:[{index:0,delta:{content:'Hello'},finish_reason:'stop'}]}),event:'message',id:null,retry:null});});
  input('Hello');await act(async()=>button('Send').click());
  expect(mocked.stream).toHaveBeenCalledOnce();expect(host.querySelector('input[accept="image/png,image/jpeg,image/webp"]')).toBeNull();expect(host.textContent).toContain('Response complete.');
 });

 it.each([['context overflow',400],['authentication expiration',401],['unavailable server',503]] as const)('does not retry or complete after %s',async(_name,status)=>{
  mocked.stream.mockRejectedValue(new WebUiHttpError(status,null));
  input('Hello');await act(async()=>button('Send').click());
  expect(mocked.stream).toHaveBeenCalledOnce();expect(host.textContent).toContain('Generation failed or disconnected');expect(host.textContent).not.toContain('Response complete.');
 });
 it('marks an empty completed transport as an error rather than a completed answer',async()=>{
  mocked.stream.mockResolvedValue(undefined);input('Hello');await act(async()=>button('Send').click());
  expect(mocked.stream).toHaveBeenCalledOnce();expect(host.textContent).toContain('Generation failed or disconnected');
 });

});
