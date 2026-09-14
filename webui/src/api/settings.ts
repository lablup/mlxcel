/** Typed adapters for the existing (unversioned) opt-in settings endpoint. */
export type SettingKind = 'bool' | 'int' | 'int_or_null' | 'float' | 'str' | 'str_or_null' | 'array' | 'object' | 'object_or_null';
export interface SettingSpec { readonly name: string; readonly type: SettingKind; readonly default: unknown; readonly mutable: boolean; readonly allowed: readonly string[] | null; readonly help: string; readonly reason?: string }
export interface SettingsResponse { readonly schema: readonly SettingSpec[]; readonly current: Readonly<Record<string, unknown>>; readonly fingerprint: string }
export interface SettingsPatchResponse { readonly applied: Readonly<Record<string, unknown>>; readonly rejected: readonly {name: string; reason: string}[]; readonly current: Readonly<Record<string, unknown>>; readonly fingerprint: string }
export interface ModelProps { readonly nCtx: number | null; readonly kvCacheMode: string | null; readonly totalSlots: number | null; readonly geometry: Readonly<Record<string, unknown>> | null }
function object(value: unknown): Record<string, unknown> { if (typeof value !== 'object' || value === null || Array.isArray(value)) throw new Error('Malformed settings response.'); return value as Record<string, unknown>; }
function fingerprint(value: unknown): string { if (typeof value !== 'string' || !/^[0-9a-f]{64}$/.test(value)) throw new Error('Malformed settings fingerprint.'); return value; }
export function validateSettings(value: unknown): SettingsResponse {
  const data = object(value);
  if (!Array.isArray(data.schema) || data.schema.length > 256) throw new Error('Malformed settings schema.');
  const names = new Set<string>();
  const schema = data.schema.map((raw: unknown): SettingSpec => {
    const spec = object(raw);
    if (typeof spec.name !== 'string' || names.has(spec.name) || typeof spec.type !== 'string' || !['bool', 'int', 'int_or_null', 'float', 'str', 'str_or_null', 'array', 'object', 'object_or_null'].includes(spec.type) || typeof spec.mutable !== 'boolean' || typeof spec.help !== 'string' || !(spec.allowed === null || Array.isArray(spec.allowed) && spec.allowed.every((item: unknown) => typeof item === 'string')) || spec.reason !== undefined && typeof spec.reason !== 'string' || !Object.hasOwn(spec, 'default')) throw new Error('Malformed setting specification.');
    names.add(spec.name);
    return spec as unknown as SettingSpec;
  });
  return { schema, current: object(data.current), fingerprint: fingerprint(data.fingerprint) };
}
export function validateSettingsPatch(value: unknown): SettingsPatchResponse {
  const data = object(value);
  if (!Array.isArray(data.rejected) || !data.rejected.every((item: unknown) => { const entry = object(item); return typeof entry.name === 'string' && typeof entry.reason === 'string'; })) throw new Error('Malformed settings rejection list.');
  return { applied: object(data.applied), rejected: data.rejected as {name: string; reason: string}[], current: object(data.current), fingerprint: fingerprint(data.fingerprint) };
}
export function validateModelProps(value: unknown): ModelProps {
  const data = object(value);
  const defaults = object(data.default_generation_settings);
  const positive = (item: unknown): number | null => typeof item === 'number' && Number.isSafeInteger(item) && item > 0 ? item : null;
  return { nCtx: positive(defaults.n_ctx), kvCacheMode: typeof data.kv_cache_mode === 'string' && data.kv_cache_mode.length <= 64 && /^[a-z0-9+_-]+$/.test(data.kv_cache_mode) ? data.kv_cache_mode : null, totalSlots: positive(data.total_slots), geometry: data.geometry === undefined ? null : object(data.geometry) };
}
export function parseSettingInput(spec: SettingSpec, raw: string): unknown {
  if (!spec.mutable) throw new Error(spec.reason ?? 'Restart required.');
  let value: unknown = raw;
  if (!['str', 'str_or_null'].includes(spec.type) || spec.type === 'str_or_null' && raw === 'null') {
    try { value = JSON.parse(raw); } catch { throw new Error(`Enter a valid ${spec.type} value.`); }
  }
  const nullable = spec.type.endsWith('_or_null');
  const base = spec.type.replace('_or_null', '');
  if (!(nullable && value === null) && (base === 'bool' ? typeof value !== 'boolean' : base === 'int' ? typeof value !== 'number' || !Number.isSafeInteger(value) : base === 'float' ? typeof value !== 'number' || !Number.isFinite(value) : base === 'str' ? typeof value !== 'string' : base === 'array' ? !Array.isArray(value) : typeof value !== 'object' || value === null || Array.isArray(value))) throw new Error(`Expected ${spec.type}.`);
  if (spec.allowed !== null && (typeof value !== 'string' || !spec.allowed.includes(value))) throw new Error(`Allowed values: ${spec.allowed.join(', ')}.`);
  return value;
}
export function settingInput(spec: SettingSpec, value: unknown): string { return typeof value === 'string' && ['str', 'str_or_null'].includes(spec.type) ? value : JSON.stringify(value) ?? ''; }

export function validateTokenCount(value: unknown): number {
  const data = object(value);
  if (!Array.isArray(data.tokens) || !data.tokens.every((token: unknown) => typeof token === 'number' && Number.isInteger(token))) throw new Error('Malformed tokenization response.');
  return data.tokens.length;
}
