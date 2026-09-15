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
export function validateGenerationDefaults(value: unknown): GenerationDefaults {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) throw new Error('Request defaults must be an object.');
  const result: Record<string, number> = {};
  for (const [key, item] of Object.entries(value)) {
    if (!(GENERATION_FIELDS as readonly string[]).includes(key)) throw new Error(`Unsupported request default: ${key}`);
    if (item === undefined) continue;
    if (typeof item !== 'number' || !Number.isFinite(item)) throw new Error(`${key} must be a finite number; leave blank to inherit.`);
    if (['max_tokens', 'top_k', 'seed'].includes(key) && !Number.isSafeInteger(item)) throw new Error(`${key} must be a safe integer.`);
    if (key === 'max_tokens' && item < 1 || key === 'temperature' && item < 0 || key === 'top_k' && item < 0 || ['top_p', 'min_p'].includes(key) && (item < 0 || item > 1) || key === 'repetition_penalty' && item <= 0 || key === 'seed' && item < 0) throw new Error(`${key} is outside its supported range.`);
    result[key] = item;
  }
  return Object.freeze(result);
}
export function snapshotGenerationDefaults(value: GenerationDefaults): GenerationDefaults {
  return validateGenerationDefaults(value);
}
