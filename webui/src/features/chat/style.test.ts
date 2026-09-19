import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

// Read from disk: Vitest blanks CSS modules, `?raw` included, so an imported
// stylesheet is an empty string and a check over it passes without looking.
const here = dirname(fileURLToPath(import.meta.url));
const read = (path: string): string => readFileSync(join(here, path), 'utf8');
const tokens = read('../../design-system/tokens.css');
const css = read('./chat.css');

describe('chat shared theme authority', () => {
  it('reads real stylesheets', () => {
    expect(tokens).toContain('--space-1');
    expect(css).toMatch(/var\(--/);
  });

  it.each(['mlxcel', 'glass'])('only references shared semantic tokens the %s theme defines', (family) => {
    const theme = read(`../../design-system/themes/${family}.css`);
    const declared = new Set(Array.from(`${tokens}\n${theme}`.matchAll(/(--[a-z0-9-]+):/g), (match) => match[1]));
    for (const match of css.matchAll(/var\((--[a-z0-9-]+)/g)) expect(declared.has(match[1]), match[1]).toBe(true);
  });
});
