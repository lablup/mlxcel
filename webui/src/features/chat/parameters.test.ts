import { describe, expect, it } from 'vitest';
import { resolveTurnParameters } from './parameters';
describe('next-turn canonical validation', () => {
  it('omits absent values and inherits blank overrides without mutating defaults', () => {
    const defaults = Object.freeze({ temperature: 0.4 });
    expect(resolveTurnParameters(defaults, { temperature: '  ', max_tokens: '' })).toEqual({ temperature: 0.4 });
    expect(resolveTurnParameters({}, {})).toEqual({});
    expect(resolveTurnParameters(defaults, { temperature: undefined })).toEqual(defaults);
    expect(resolveTurnParameters(defaults, { temperature: '0' })).toEqual({ temperature: 0 });
    expect(defaults.temperature).toBe(0.4);
  });
  it.each([{ max_tokens: '1.5' }, { top_p: '1.1' }, { min_p: '-1' }, { temperature: 'Infinity' }, { repetition_penalty: '0' }, { seed: '-1' }])('rejects malformed and out-of-range overrides %j', (draft) => {
    expect(() => resolveTurnParameters({}, draft)).toThrow();
  });
});
