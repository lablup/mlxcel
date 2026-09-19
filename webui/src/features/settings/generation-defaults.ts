import { formatF32 } from '../../api/settings';
import { t, type Locale, type StringKey } from '../../i18n/catalog';
/** Browser-session request defaults. Missing fields inherit server defaults. */
export interface GenerationDefaults {
  readonly max_tokens?: number;
  readonly temperature?: number;
  readonly top_p?: number;
  readonly top_k?: number;
  readonly min_p?: number;
  readonly repetition_penalty?: number;
  readonly seed?: number;
}
export const DEFAULT_GENERATION_DEFAULTS: GenerationDefaults = Object.freeze({});
export const GENERATION_FIELDS = ['max_tokens', 'temperature', 'top_p', 'top_k', 'min_p', 'repetition_penalty', 'seed'] as const;
export type GenerationField = typeof GENERATION_FIELDS[number];
/** Request fields that take whole numbers; the rest are floats. */
export const INTEGER_GENERATION_FIELDS: ReadonlySet<GenerationField> = new Set(['max_tokens', 'top_k', 'seed']);
export function validateGenerationDefaults(value: unknown): GenerationDefaults {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) throw new Error('Request defaults must be an object.');
  const result: Record<string, number> = {};
  for (const [key, item] of Object.entries(value)) {
    if (!(GENERATION_FIELDS as readonly string[]).includes(key)) throw new Error(`Unsupported request default: ${key}`);
    if (item === undefined) continue;
    if (typeof item !== 'number' || !Number.isFinite(item)) throw new Error(`${key} must be a finite number; leave blank to inherit.`);
    if (INTEGER_GENERATION_FIELDS.has(key as GenerationField) && !Number.isSafeInteger(item)) throw new Error(`${key} must be a safe integer.`);
    if (key === 'max_tokens' && item < 1 || key === 'temperature' && item < 0 || key === 'top_k' && item < 0 || ['top_p', 'min_p'].includes(key) && (item < 0 || item > 1) || key === 'repetition_penalty' && item <= 0 || key === 'seed' && item < 0) throw new Error(`${key} is outside its supported range.`);
    result[key] = item;
  }
  return Object.freeze(result);
}
export function snapshotGenerationDefaults(value: GenerationDefaults): GenerationDefaults {
  return validateGenerationDefaults(value);
}
/** Where the value of a request field for the next request comes from, in precedence order. */
export type ParameterSource = 'override' | 'session' | 'server' | 'unknown';
export interface EffectiveParameter { readonly value: number | null; readonly source: ParameterSource }
/**
 * The value the next request uses for `name`: a next-turn override, else the browser-session default,
 * else the server default. `serverDefault` is `current["default_" + name]` from the selected ready model's
 * /settings, or `undefined` when that was not read or the schema has no such entry. A server `null`
 * (an unset default, such as a random seed) is still a known server value; anything else unreadable is `unknown`.
 */
export function resolveEffectiveParameter(name: GenerationField, override: number | undefined, sessionDefault: number | undefined, serverDefault: unknown): EffectiveParameter {
  if (typeof override === 'number' && Number.isFinite(override)) return { value: override, source: 'override' };
  if (typeof sessionDefault === 'number' && Number.isFinite(sessionDefault)) return { value: sessionDefault, source: 'session' };
  if (serverDefault === null) return { value: null, source: 'server' };
  if (typeof serverDefault !== 'number' || !Number.isFinite(serverDefault)) return { value: null, source: 'unknown' };
  if (INTEGER_GENERATION_FIELDS.has(name)) return Number.isSafeInteger(serverDefault) ? { value: serverDefault, source: 'server' } : { value: null, source: 'unknown' };
  // Floats are stored as f32 on the server; report the value a person set, not its f64 widening.
  return { value: Number(formatF32(serverDefault)), source: 'server' };
}
const SOURCE_LABEL: Readonly<Record<Exclude<ParameterSource, 'unknown'>, StringKey>> = { override: 'settings.requests.source.override', session: 'settings.requests.source.session', server: 'settings.requests.source.server' };
/** One wording for Settings and the Chat hint, so the same state reads the same in both. */
export function describeEffectiveParameter(locale: Locale, resolved: EffectiveParameter): string {
  if (resolved.source === 'unknown') return t(locale, 'settings.requests.unknown');
  const value = resolved.value === null ? t(locale, 'settings.value.unset') : String(resolved.value);
  return t(locale, 'settings.requests.value_source', { value, source: t(locale, SOURCE_LABEL[resolved.source]) });
}
