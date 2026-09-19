// Copyright 2026 Lablup Inc. Licensed under Apache-2.0.
import { describe, expect, it } from 'vitest';
import { SseParser, splitUtf8 } from '../../api/sse';
import type { ChatTurn } from './history';
import { appendFrame, buildMessages, completeTurn, decodeRate, MAX_TURN_CHARACTERS } from './stream';
const turn = (): ChatTurn => ({ id:'turn',modelId:'opaque',inferenceId:'actual/model',modelRevision:3,modelName:'Model',prompt:'Hi',content:'',reasoning:'',tools:[],status:'streaming',finishReason:null,usage:null,ttftMs:null,elapsedMs:null,error:null,parameters:{temperature:0.1},images:[],startedAt:0 });
const frame = (delta: unknown, finish_reason: string|null = null): string => JSON.stringify({choices:[{index:0,delta,finish_reason}]});
describe('frozen chat stream',()=>{
 it('handles split UTF8 reasoning-only, mixed content and reported usage',()=>{
  let value=turn(); let done=false;
  const parser=new SseParser({onMessage:msg=>{value=appendFrame(value,msg.data,20);},onDone:()=>{done=true;}});
  const stream=`data: ${frame({reasoning_content:'생각',reasoning:'생각'})}\n\ndata: ${frame({content:'Hello'})}\n\ndata: ${frame({},'length')}\n\ndata: ${JSON.stringify({choices:[],usage:{prompt_tokens:2,completion_tokens:4,total_tokens:6}})}\n\ndata: [DONE]\n\n`;
  for(const chunk of splitUtf8(stream,[1,2,3]))parser.push(chunk);
  expect(done).toBe(true);expect(value.reasoning).toBe('생각');expect(value.content).toBe('Hello');expect(value.status).toBe('streaming');
  value=completeTurn(value,1020);expect(value.finishReason).toBe('length');expect(value.status).toBe('complete');expect(decodeRate(value)).toBe(3);
  expect(value.modelId).toBe('opaque');expect(value.inferenceId).toBe('actual/model');expect(value.parameters).toEqual({temperature:0.1});
 });
 it('merges structured tools without executing them or repeating reasoning alias',()=>{
  let value=appendFrame(turn(),frame({tool_calls:[{index:0,id:'call',function:{name:'shell',arguments:'{"x":'}}]}),10);
  value=appendFrame(value,frame({tool_calls:[{index:0,function:{arguments:'"not executed"}'}}]},'tool_calls'),20);
  expect(completeTurn(value,30).tools).toEqual([{index:0,id:'call',name:'shell',arguments:'{"x":"not executed"}'}]);
 });
 it('rejects malformed, empty, error and excessive response data',()=>{
  for(const data of ['{','{}',JSON.stringify({error:{message:'secret'}}),frame({content:3}),frame({tool_calls:[{index:99}]})])expect(()=>appendFrame(turn(),data,10)).toThrow();
  expect(()=>completeTurn(turn(),20)).toThrow();expect(()=>appendFrame(turn(),frame({content:'x'.repeat(MAX_TURN_CHARACTERS)}),10)).toThrow();
 });
 it('preserves a 10000-token-sized response within a bounded transcript',()=>{
  const value=appendFrame(turn(),frame({content:'token '.repeat(10000)},'stop'),10);
  expect(completeTurn(value,20).content.length).toBe(60000);expect(decodeRate(value)).toBeNull();
 });
 it('does not replay reasoning or unexecuted tools or partial assistant content',()=>{
  const value={...turn(),status:'cancelled' as const,content:'partial',reasoning:'private',tools:[{index:0,id:'1',name:'shell',arguments:'evil'}]};
  expect(buildMessages('system',[value])).toEqual([{role:'system',content:'system'},{role:'user',content:'Hi'}]);
 });
});
