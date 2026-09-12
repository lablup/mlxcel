import { describe, expect, it } from 'vitest';
import { validateApiBase } from './url';
import bootstrapFixture from '../../../tests/fixtures/webui/examples/bootstrap.model-free.json';
import runtimeFixture from '../../../tests/fixtures/webui/examples/runtime.snapshot.json';
import { WebUiApiClient, WebUiHttpError } from './client';

describe('WebUI API client', () => {
  it('accepts only same-origin relative API bases', () => {
    expect(validateApiBase('/private')).toBe('/private');
    expect(() => validateApiBase('https://example.test')).toThrow(/same-origin/);
    expect(() => validateApiBase('//example.test')).toThrow(/same-origin/);
    expect(() => validateApiBase('/private/../other')).toThrow(/dot path/);
  });

  it('keeps bearer credentials in memory and off URLs', async () => {
    const urls: string[] = [];
    const authHeaders: string[] = [];
    const fetchImpl: typeof fetch = async (input, init) => {
      urls.push(String(input));
      authHeaders.push(new Headers(init?.headers).get('authorization') ?? '');
      return new Response(JSON.stringify(bootstrapFixture), { status: 200 });
    };
    const client = new WebUiApiClient({ fetchImpl });
    client.setBearerToken('secret-token');
    await client.bootstrap();
    expect(urls).toEqual(['/ui-api/v1/bootstrap']);
    expect(authHeaders).toEqual(['Bearer secret-token']);
    expect(urls.join(' ')).not.toContain('secret-token');
    expect(localStorage.length).toBe(0);
    expect(sessionStorage.length).toBe(0);
  });

  it('adds autoload=false to runtime observation', async () => {
    const urls: string[] = [];
    const fetchImpl: typeof fetch = async (input) => {
      urls.push(String(input));
      return new Response(JSON.stringify(runtimeFixture), { status: 200 });
    };
    await new WebUiApiClient({ fetchImpl }).runtime(runtimeFixture.model_id);
    expect(urls[0]).toBe(`/ui-api/v1/runtime?model_id=${runtimeFixture.model_id}&autoload=false`);
  });

  it('allows only approved inference stream paths, real inference ids, and autoload=false outside the UI API namespace', async () => {
    const urls: string[] = [];
    const bodies: unknown[] = [];
    const fetchImpl: typeof fetch = async (input, init) => {
      urls.push(String(input));
      bodies.push(JSON.parse(String(init?.body)));
      return new Response('data: [DONE]\n\n', { status: 200 });
    };
    const client = new WebUiApiClient({ fetchImpl });
    await client.chatCompletions('mlx-community/Qwen3-4B-4bit', { messages: [] }, { onFrame: () => undefined });
    await client.responses('mlx-community/Qwen3-4B-4bit', { input: [] }, { onFrame: () => undefined });
    expect(urls).toEqual(['/v1/chat/completions?autoload=false', '/v1/responses?autoload=false']);
    expect(bodies).toEqual([{ messages: [], model: 'mlx-community/Qwen3-4B-4bit', stream: true }, { input: [], model: 'mlx-community/Qwen3-4B-4bit', stream: true }]);
  });

  it('exposes shared SSE frames for chat content, reasoning, tool calls and usage', async () => {
    const chunks = [
      'data: {"choices":[{"delta":{"content":"Hello","reasoning_content":"Because","tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{}"}}]}}],"usage":null}\n\n',
      'data: {"choices":[{"delta":{}}],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}\n\n',
      'data: [DONE]\n\n',
    ];
    const fetchImpl: typeof fetch = async () => new Response(new ReadableStream({
      start(controller) {
        const encoder = new TextEncoder();
        for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
        controller.close();
      },
    }), { status: 200 });
    const frames: unknown[] = [];
    await new WebUiApiClient({ fetchImpl }).chatCompletions('mlx-community/Qwen3-4B-4bit', {}, { onFrame: (message) => { frames.push(JSON.parse(message.data)); } });
    expect(frames).toHaveLength(2);
    expect(frames[0]).toMatchObject({ choices: [{ delta: { content: 'Hello', reasoning_content: 'Because', tool_calls: [{ function: { name: 'lookup' } }] } }] });
    expect(frames[1]).toMatchObject({ usage: { total_tokens: 3 } });
  });
});

describe('WebUI API client security edges', () => {
  it('revokes the token on 401 before parsing an invalid response body', async () => {
    let unauthorized = 0;
    const seenAuth: string[] = [];
    const fetchImpl: typeof fetch = async (_input, init) => {
      seenAuth.push(new Headers(init?.headers).get('authorization') ?? '');
      return new Response('{invalid json', { status: 401 });
    };
    const client = new WebUiApiClient({ fetchImpl, onUnauthorized: () => { unauthorized += 1; } });
    client.setBearerToken('secret-token');
    await expect(client.bootstrap()).rejects.toThrow();
    await expect(client.bootstrap()).rejects.toThrow();
    expect(unauthorized).toBe(2);
    expect(seenAuth).toEqual(['Bearer secret-token', '']);
  });

  it('sends legacy Last-Event-ID and rejects EOF without DONE so sync can reconnect', async () => {
    const lastIds: string[] = [];
    const fetchImpl: typeof fetch = async (_input, init) => {
      lastIds.push(new Headers(init?.headers).get('last-event-id') ?? '');
      return new Response(new ReadableStream({
        start(controller) {
          controller.enqueue(new TextEncoder().encode(': eof\n\n'));
          controller.close();
        },
      }), { status: 200 });
    };
    const client = new WebUiApiClient({ fetchImpl });
    await expect(client.events({ onEvent: () => undefined }, undefined, { lastEventId: 'evt_0000000000000001', serverInstanceId: null, afterSequence: null })).rejects.toThrow(/ended before/);
    expect(lastIds).toEqual(['evt_0000000000000001']);
  });

  it('uses paired sequence replay query instead of fabricating an SSE id from snapshot fences', async () => {
    const seen: { url: string; lastId: string }[] = [];
    const fetchImpl: typeof fetch = async (input, init) => {
      seen.push({ url: String(input), lastId: new Headers(init?.headers).get('last-event-id') ?? '' });
      return new Response('data: [DONE]\n\n', { status: 200 });
    };
    const client = new WebUiApiClient({ fetchImpl });
    await client.events({ onEvent: () => undefined }, undefined, { lastEventId: 'evt_latest_999', serverInstanceId: 'srv_one', afterSequence: 41 });
    expect(seen).toEqual([{ url: '/ui-api/v1/events?server_instance_id=srv_one&after_sequence=41', lastId: '' }]);
  });

  it('keeps malformed 403 and 409 bodies as HTTP errors, redacts tokens and rejects redirects', async () => {
    const seenRedirectModes: RequestRedirect[] = [];
    const fetchImpl: typeof fetch = async (_input, init) => {
      seenRedirectModes.push(init?.redirect ?? 'follow');
      return new Response(JSON.stringify({ error: { code: 'forbidden', message: 'leaked secret-token in detail', retryable: false }, request_id: 'req' }), { status: 403 });
    };
    const client = new WebUiApiClient({ fetchImpl });
    client.setBearerToken('secret-token');
    await expect(client.bootstrap()).rejects.toMatchObject({ status: 403 });
    await expect(client.bootstrap()).rejects.not.toThrow(/secret-token/);
    expect(seenRedirectModes).toEqual(['error', 'error']);
  });

  it('turns malformed 409 bodies into WebUiHttpError instead of schema errors', async () => {
    const fetchImpl: typeof fetch = async () => new Response('{invalid json', { status: 409 });
    const client = new WebUiApiClient({ fetchImpl });
    await expect(client.bootstrap()).rejects.toBeInstanceOf(WebUiHttpError);
  });

  it('bounds JSON error bodies before parsing', async () => {
    const fetchImpl: typeof fetch = async () => new Response('x'.repeat(2 * 1024 * 1024 + 1), { status: 403 });
    const client = new WebUiApiClient({ fetchImpl });
    await expect(client.bootstrap()).rejects.toMatchObject({ status: 403 });
  });

  it('cancels pending JSON readers when a custom fetch ignores AbortSignal', async () => {
    let cancelled = false;
    const fetchImpl: typeof fetch = async () => new Response(new ReadableStream<Uint8Array>({
      cancel() {
        cancelled = true;
      },
    }), { status: 200 });
    const controller = new AbortController();
    const request = new WebUiApiClient({ fetchImpl }).bootstrap(controller.signal);
    await Promise.resolve();
    controller.abort();
    await expect(request).rejects.toMatchObject({ name: 'AbortError' });
    expect(cancelled).toBe(true);
  });

  it('cancels pending SSE readers when a custom fetch ignores AbortSignal', async () => {
    let cancelled = false;
    const fetchImpl: typeof fetch = async () => new Response(new ReadableStream<Uint8Array>({
      cancel() {
        cancelled = true;
      },
    }), { status: 200 });
    const controller = new AbortController();
    const request = new WebUiApiClient({ fetchImpl }).chatCompletions('mlx-community/Qwen3-4B-4bit', {}, { onFrame: () => undefined }, controller.signal);
    await Promise.resolve();
    controller.abort();
    await expect(request).rejects.toMatchObject({ name: 'AbortError' });
    expect(cancelled).toBe(true);
  });
});
