import { describe, expect, it } from 'vitest';
import { validateGenerationDefaults, snapshotGenerationDefaults } from './generation-defaults';
import { importProfiles, profileCli, validateLoadProfile, KV_MODES } from './load-profiles';
import { contextBudget } from './context-check';
import { validateAgainstSchema } from '../../api/jsonSchema';
import { parseSettingInput, validateModelProps, validateSettings, validateSettingsPatch, type SettingSpec } from '../../api/settings';
import { WebUiApiClient } from '../../api/client';
const spec: SettingSpec = { name: 'default_temperature', type: 'float', default: 1, mutable: true, allowed: null, help: 'Temperature' };
const settings = { schema: [spec], current: { default_temperature: 1 }, fingerprint: 'a'.repeat(64) };
describe('generation defaults', () => {
  it('omits unset values and freezes independent snapshots', () => { const source = { temperature: 0.2 }; const copy = snapshotGenerationDefaults(source); source.temperature = 2; expect(copy).toEqual({ temperature: 0.2 }); expect(Object.isFrozen(copy)).toBe(true); expect(validateGenerationDefaults({ seed: undefined })).toEqual({}); });
  it.each([{ max_tokens: 0 }, { temperature: -1 }, { top_p: 2 }, { min_p: -1 }, { top_k: 0.5 }, { seed: -1 }, { temperature: NaN }, { temperature: null }, { system_prompt: 'secret' }, { token: 'secret' }])('rejects invalid or secret request fields %j', (value) => { expect(() => validateGenerationDefaults(value)).toThrow(); });
});
describe('load profile imports and canonical parity', () => {
  it('uses the canonical schema for every exposed enum and boundary', () => { for (const kv_cache_mode of KV_MODES) { const value = validateLoadProfile({ ctx_size: 262144, n_parallel: 32, kv_cache_mode }); expect(() => validateAgainstSchema('LoadProfile', value, '$')).not.toThrow(); } });
  it('omits unset/nulls, not zero', () => { expect(validateLoadProfile({ ctx_size: null, n_parallel: undefined })).toEqual({}); expect(() => validateLoadProfile({ ctx_size: 0 })).toThrow(); });
  it.each([{ ctx_size: 262145 }, { n_parallel: 33 }, { kv_cache_mode: 'q8_0' }, { model_path: '/tmp' }, { argv: '--token abc' }, { ctx_size: Infinity }])('rejects unsafe or invalid profiles %j', (value) => { expect(() => validateLoadProfile(value)).toThrow(); });
  it('migrates version zero explicitly, rejects unknown versions and fields', () => { expect(importProfiles('{"version":0,"profile":{"ctx_size":2048}}')).toEqual({ version: 1, reusable: { ctx_size: 2048 }, models: {} }); expect(() => importProfiles('{"version":2}')).toThrow(); expect(() => importProfiles('{"version":0,"profile":{},"secret":"x"}')).toThrow(); });
  it('rejects prototype keys and nonopaque IDs and only renders validated numeric/enumerated CLI flags', () => { expect(() => importProfiles('{"version":1,"reusable":{},"models":{"__proto__":{}}}')).toThrow(); expect(profileCli({ ctx_size: 2048, n_parallel: 2, kv_cache_mode: 'fp16' })).toBe('mlxcel-server --webui --ctx-size 2048 --parallel 2 --kv-cache-mode fp16'); });
});
describe('existing opt-in settings adapter', () => {
  it('validates schema and preserves partial success rather than an all-success boolean', () => { expect(validateSettings(settings)).toEqual(settings); const mixed = { applied: { default_temperature: 0.5 }, rejected: [{ name: 'ctx_size', reason: 'restart required' }], current: { default_temperature: 0.5 }, fingerprint: 'b'.repeat(64) }; expect(validateSettingsPatch(mixed)).toEqual(mixed); });
  it('honors type, enum and readonly reasons; server owns numeric ranges', () => { expect(parseSettingInput(spec, '0.4')).toBe(0.4); expect(() => parseSettingInput(spec, 'null')).toThrow(); expect(() => parseSettingInput({ ...spec, mutable: false, reason: 'worker-owned' }, '1')).toThrow('worker-owned'); expect(() => parseSettingInput({ ...spec, type: 'str', allowed: ['a'] }, 'b')).toThrow(/Allowed/); });
  it('treats zero context as unknown and never estimates pool capacity', () => { expect(validateModelProps({ default_generation_settings: { n_ctx: 0 }, total_slots: 4, geometry: { kv_unified: false } })).toEqual({ nCtx: null, kvCacheMode: null, totalSlots: 4, geometry: { kv_unified: false } }); expect(contextBudget(0, 1, 2)).toBe('unknown'); expect(contextBudget(100, 90, 20)).toBe('exceeds'); expect(contextBudget(100, 10, 20)).toBe('raw-fits'); });
  it('uses auth, configured prefix, exact legacy model identity and autoload=false for every operation', async () => {
    const calls: { url: string; auth: string | null; method: string; body: unknown }[] = [];
    const fetchImpl: typeof fetch = async (url, init) => { calls.push({ url: String(url), auth: new Headers(init?.headers).get('authorization'), method: init?.method ?? '', body: init?.body ? JSON.parse(String(init.body)) : null }); const body = String(url).includes('/tokenize') ? { tokens: [1, 2] } : String(url).includes('/props') ? { default_generation_settings: { n_ctx: 2048 }, total_slots: 1 } : init?.method === 'PATCH' ? { applied: { default_temperature: 0.5 }, rejected: [], current: {}, fingerprint: 'b'.repeat(64) } : settings; return new Response(JSON.stringify(body)); };
    const client = new WebUiApiClient({ apiBase: '/prefix', fetchImpl }); client.setBearerToken('secret');
    await client.settings('org/model'); await client.patchSettings('org/model', { default_temperature: 0.5 }); await client.modelProps('org/model'); expect(await client.tokenCount('org/model', 'hello')).toBe(2);
    expect(calls.map((call) => call.url)).toEqual(['/prefix/settings?model=org%2Fmodel&autoload=false', '/prefix/settings?model=org%2Fmodel&autoload=false', '/prefix/props?model=org%2Fmodel&autoload=false', '/prefix/tokenize?model=org%2Fmodel&autoload=false']); expect(calls.every((call) => call.auth === 'Bearer secret')).toBe(true); expect(calls[1].body).toEqual({ op: 'merge', values: { default_temperature: 0.5 } });
  });
});
