import tokens from '../../design-system/tokens.css?raw';
import css from './chat.css?raw';
import { describe, expect, it } from 'vitest';
describe('chat shared theme authority', () => {
  it('only references defined shared semantic tokens', () => {
    const declared = new Set(Array.from(tokens.matchAll(/(--[a-z0-9-]+):/g), (match) => match[1]));
    for (const match of css.matchAll(/var\((--[a-z0-9-]+)/g)) expect(declared.has(match[1]), match[1]).toBe(true);
  });
});
