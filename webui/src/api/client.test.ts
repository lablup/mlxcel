import { describe, expect, it } from 'vitest';
import { validateApiBase } from './url';
import bootstrapFixture from '../../../tests/fixtures/webui/examples/bootstrap.model-free.json';
import runtimeFixture from '../../../tests/fixtures/webui/examples/runtime.snapshot.json';
import { WebUiApiClient } from './client';

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

  it('sends Last-Event-ID and rejects EOF without DONE so sync can reconnect', async () => {
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
    await expect(client.events({ onEvent: () => undefined }, undefined, 'evt_0000000000000001')).rejects.toThrow(/ended before/);
    expect(lastIds).toEqual(['evt_0000000000000001']);
  });
});
