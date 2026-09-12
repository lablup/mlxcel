import { describe, expect, it } from 'vitest';
import { SseParser, splitUtf8, type SseMessage } from './sse';

function parse(text: string, chunks: readonly number[]): { messages: SseMessage[]; done: boolean } {
  const messages: SseMessage[] = [];
  let done = false;
  const parser = new SseParser({ onMessage: (message) => messages.push(message), onDone: () => { done = true; }, maxFrameBytes: 128 });
  for (const chunk of splitUtf8(text, chunks)) parser.push(chunk);
  parser.close();
  return { messages, done };
}

describe('fetch SSE parser', () => {
  it('decodes byte-split multibyte CJK and CRLF frames', () => {
    const result = parse('id: 1\r\nevent: reasoning\r\ndata: {"delta":"안녕"}\r\n\r\n', [1, 2, 3]);
    expect(result.messages).toEqual([{ event: 'reasoning', data: '{"delta":"안녕"}', id: '1', retry: null }]);
  });

  it('preserves distinct content, reasoning, tool and usage event frames', () => {
    const result = parse('event: reasoning\ndata: r\n\nevent: content\ndata: c\n\nevent: tool\ndata: t\n\nevent: usage\ndata: u\n\n', [5]);
    expect(result.messages.map((message) => `${message.event}:${message.data}`)).toEqual(['reasoning:r', 'content:c', 'tool:t', 'usage:u']);
  });

  it('ignores keepalive comments and honors DONE without emitting a fake message', () => {
    const result = parse(': keepalive\n\ndata: [DONE]\n\n', [4]);
    expect(result.messages).toHaveLength(0);
    expect(result.done).toBe(true);
  });

  it('joins multiline data and reports retry', () => {
    const result = parse('retry: 250\ndata: one\ndata: two\n\n', [2]);
    expect(result.messages[0]).toEqual({ event: 'message', data: 'one\ntwo', id: null, retry: 250 });
  });

  it('rejects oversized frames before dispatch', () => {
    const parser = new SseParser({ maxFrameBytes: 8, onMessage: () => undefined });
    expect(() => parser.push(new TextEncoder().encode('data: too long\n\n'))).toThrow(/byte limit/);
  });

  it('emits buffered data on midstream close', () => {
    const messages: SseMessage[] = [];
    const parser = new SseParser({ onMessage: (message) => messages.push(message) });
    parser.push(new TextEncoder().encode('event: content\ndata: tail'));
    parser.close();
    expect(messages).toEqual([{ event: 'content', data: 'tail', id: null, retry: null }]);
  });
});
