// Copyright 2026 Lablup Inc. Licensed under Apache-2.0.
import { afterEach, describe, expect, it } from 'vitest';
import { apiBaseFromDocument, apiPath, validateApiBase } from './url';
import { WebUiApiClient } from './client';
import bootstrap from '../../../tests/fixtures/webui/examples/bootstrap.model-free.json';
import catalog from '../../../tests/fixtures/webui/examples/catalog.page.json';

const originalUrl = window.location.href;
afterEach(() => { document.querySelectorAll('base,meta[name="mlxcel-ui-api-base"]').forEach(element => element.remove()); window.history.replaceState(null, '', originalUrl); });
const at = (path: string): void => window.history.replaceState(null, '', path);
const base = (href: string): void => { const element = document.createElement('base'); element.setAttribute('href', href); document.head.append(element); };
describe('document mount API authority', () => {
  it.each([['/lab/webui/', '/lab'], ['/lab/nested/webui/index.html#chat', '/lab/nested'], ['/webui/', ''], ['/webui/index.html', ''], ['/#gallery', ''], ['/index.html', '']])('derives %s without injected metadata', (path, expected) => { at(path); expect(apiBaseFromDocument()).toBe(expected); expect(validateApiBase(undefined)).toBe(expected); });
  it('preserves an already escaped prefix without double encoding', () => { at('/caf%C3%A9/webui/'); expect(validateApiBase(undefined)).toBe('/caf%C3%A9'); expect(validateApiBase('/caf%C3%A9')).toBe('/caf%C3%A9'); expect(validateApiBase('/café')).toBe('/caf%C3%A9'); });
  it.each(['webui/', './webui/'])('resolves relative base %s against the document URL rather than its origin', href => { at('/lab/index.html'); base(href); expect(validateApiBase(undefined)).toBe('/lab'); });
  it.each(['.', './'])('resolves a same-directory relative base %s', href => { at('/lab/webui/'); base(href); expect(validateApiBase(undefined)).toBe('/lab'); });
  it('accepts an explicit same-origin base and gives validated metadata precedence', () => {
    at('/lab/webui/'); base(`${window.location.origin}/explicit/webui/`); expect(validateApiBase(undefined)).toBe('/explicit');
    const meta = document.createElement('meta'); meta.name = 'mlxcel-ui-api-base'; meta.content = '/configured'; document.head.append(meta); expect(validateApiBase(undefined)).toBe('/configured');
    meta.content = '//external.invalid'; expect(() => validateApiBase(undefined)).toThrow();
  });
  it('maps an explicit root base to empty rather than a network-path URL', () => { expect(validateApiBase('/')).toBe(''); expect(apiPath(validateApiBase('/'), '/ui-api/v1/bootstrap')).toBe('/ui-api/v1/bootstrap'); });
  it.each(['/a//b', '/a/../b', '/a/%2e%2e/b', '/a/%252e%252e/b', '/a/%2fb', '/a/%5cb', '/a/%3fb', '/a/%23b', '/a/%00b', '/a/%', '/a/%ff', '/a b', '/a\nb', '//elsewhere.invalid', 'https://elsewhere.invalid'])('rejects ambiguous API authority %s', candidate => { expect(() => validateApiBase(candidate)).toThrow(); });
  it.each(['../webui/', '%2e/webui/', './%2e%2e/webui/', 'https://elsewhere.invalid/webui/', '/a/%2e%2e/webui/', '/a/%252e/webui/', '/a//webui/', '/a\\webui/', '/webui/?key=value', '/webui/#fragment', 'http://user:password@localhost/webui/'])('rejects unsafe document base %s', href => { base(href); expect(() => validateApiBase(undefined)).toThrow(); });
  it.each(['/a//webui/', '//webui/', '/a/%2f/webui/'])('rejects ambiguous mounted path %s', path => { at(path.startsWith('//') ? `${window.location.origin}${path}` : path); expect(() => validateApiBase(undefined)).toThrow(); });
  it('retains the endpoint allowlist', () => { expect(() => apiPath('/lab', 'https://elsewhere.invalid')).toThrow(); expect(() => apiPath('/lab', '/download')).toThrow(); });
  it('routes authenticated bootstrap, catalog, events, inference and settings beneath the mounted prefix', async () => {
    at('/lab/webui/'); const calls: {url:string;auth:string;redirect:RequestRedirect|undefined}[] = [];
    const client = new WebUiApiClient({ fetchImpl: async (input, init) => {
      const url = String(input); calls.push({url,auth:new Headers(init?.headers).get('authorization') ?? '',redirect:init?.redirect});
      if (url.includes('/events') || url.includes('/v1/chat/') || url.includes('/v1/responses')) return new Response('data: [DONE]\n\n');
      const body = url.includes('/bootstrap') ? bootstrap : url.includes('/catalog') ? catalog : url.includes('/props') ? {default_generation_settings:{n_ctx:8192},total_slots:1} : url.includes('/tokenize') ? {tokens:[1]} : {schema:[],current:{},fingerprint:'a'.repeat(64)};
      return new Response(JSON.stringify(body));
    } });
    client.setBearerToken('prefix-secret'); await client.bootstrap(); await client.catalog(); await client.events({onEvent:()=>undefined}); await client.chatCompletions('alpha', {}, {onFrame:()=>undefined}); await client.responses('alpha', {}, {onFrame:()=>undefined}); await client.settings('alpha'); await client.modelProps('alpha'); await client.tokenCount('alpha','Hello');
    expect(calls.map(call=>call.url)).toEqual(['/lab/ui-api/v1/bootstrap','/lab/ui-api/v1/catalog','/lab/ui-api/v1/events','/lab/v1/chat/completions?autoload=false','/lab/v1/responses?autoload=false','/lab/settings?model=alpha&autoload=false','/lab/props?model=alpha&autoload=false','/lab/tokenize?model=alpha&autoload=false']);
    expect(calls.every(call=>call.auth==='Bearer prefix-secret' && call.redirect==='error' && !call.url.includes('prefix-secret'))).toBe(true);
  });
});
