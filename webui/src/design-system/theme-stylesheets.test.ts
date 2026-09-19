import { readdirSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import mainSource from '../main.tsx?raw';
import { THEME_IDS } from './theme';

// Stylesheets are read from disk: Vitest blanks CSS modules, `?raw` included,
// so an imported stylesheet would be an empty string and every check below
// would pass without looking.
// Paths are joined by hand: Vite rewrites `new URL(literal, import.meta.url)` into a served asset URL.
const srcRoot = join(dirname(fileURLToPath(import.meta.url)), '..');
const read = (path: string): string => readFileSync(join(srcRoot, path), 'utf8');
const uiCommonBase = read('../node_modules/@lablup/ui-common/dist/styles/base.css');
const commonTokens = read('design-system/common-tokens.css');
const glassIntensity = read('design-system/glass-intensity.css');
const themeEntry = read('design-system/themes/index.css');
const tokens = read('design-system/tokens.css');
// Every product stylesheet, theme files included, keyed by path relative to src/.
const productCss: Record<string, string> = Object.fromEntries(
  readdirSync(srcRoot, { recursive: true, encoding: 'utf8' })
    .filter((path) => path.endsWith('.css'))
    .map((path) => [path.split('\\').join('/'), read(path)]),
);
const themeCss = Object.fromEntries(Object.entries(productCss).filter(([path]) => path.startsWith('design-system/themes/')));

type Rule = { selectors: string[]; body: string };

function stripComments(css: string): string {
  return css.replace(/\/\*[\s\S]*?\*\//g, '');
}

/** Top-level style rules only; at-rule blocks (@media, @supports, @import) are skipped. */
function topLevelRules(css: string): Rule[] {
  const source = stripComments(css);
  const rules: Rule[] = [];
  let index = 0;
  while (index < source.length) {
    const open = source.indexOf('{', index);
    const semicolon = source.indexOf(';', index);
    if (open === -1) break;
    if (semicolon !== -1 && semicolon < open && source.slice(index, semicolon).trim().startsWith('@')) {
      index = semicolon + 1;
      continue;
    }
    const prelude = source.slice(index, open).trim();
    let depth = 1;
    let cursor = open + 1;
    while (cursor < source.length && depth > 0) {
      if (source[cursor] === '{') depth += 1;
      else if (source[cursor] === '}') depth -= 1;
      cursor += 1;
    }
    if (!prelude.startsWith('@')) rules.push({ selectors: prelude.split(',').map((selector) => selector.trim()), body: source.slice(open + 1, cursor - 1) });
    index = cursor;
  }
  return rules;
}

function declaredProperties(body: string): Set<string> {
  return new Set(Array.from(body.matchAll(/(--[A-Za-z0-9-]+)\s*:/g), (match) => match[1]));
}

function referencedProperties(body: string): Set<string> {
  return new Set(Array.from(body.matchAll(/var\(\s*(--[A-Za-z0-9-]+)/g), (match) => match[1]));
}

function rootDeclarations(css: string): Set<string> {
  const declared = new Set<string>();
  for (const rule of topLevelRules(css)) if (rule.selectors.includes(':root')) for (const name of declaredProperties(rule.body)) declared.add(name);
  return declared;
}

/** Custom properties a theme id declares across every theme file, through exact-id selectors. */
function themeDeclarations(id: string): { declared: Set<string>; referenced: Set<string> } {
  const declared = new Set<string>();
  const referenced = new Set<string>();
  for (const css of Object.values(themeCss)) {
    for (const rule of topLevelRules(css)) {
      if (!rule.selectors.includes(`[data-theme="${id}"]`)) continue;
      for (const name of declaredProperties(rule.body)) declared.add(name);
      for (const name of referencedProperties(rule.body)) referenced.add(name);
    }
  }
  return { declared, referenced };
}

const baseContract = new Set(Array.from(uiCommonBase.matchAll(/(--token-[A-Za-z0-9]+)\s*:/g), (match) => match[1]));
const structural = rootDeclarations(commonTokens);
const rootLevel = new Set([...rootDeclarations(tokens), ...structural]);
const intensityInputs = new Set(Array.from(glassIntensity.matchAll(/(--material-glass-[a-z]+)\s*:/g), (match) => match[1]));

describe('theme stylesheets', () => {
  it('reads real stylesheets', () => {
    expect(baseContract.size).toBeGreaterThanOrEqual(119);
    expect(Object.keys(themeCss).sort()).toEqual(['design-system/themes/contract.css', 'design-system/themes/glass.css', 'design-system/themes/index.css', 'design-system/themes/mlxcel.css']);
    expect(Object.keys(productCss)).toContain('design-system/components.css');
  });

  it('split the ui-common token contract between :root structure and per-theme values without overlap', () => {
    for (const id of THEME_IDS) {
      const themed = new Set([...themeDeclarations(id).declared].filter((name) => name.startsWith('--token-')));
      expect([...themed].filter((name) => structural.has(name)), `${id} re-declares structural tokens`).toEqual([]);
      const covered = new Set([...structural, ...themed]);
      expect([...baseContract].filter((name) => !covered.has(name)), `${id} leaves contract tokens at ui-common defaults`).toEqual([]);
      expect([...themed].filter((name) => !baseContract.has(name)), `${id} sets names outside the contract`).toEqual([]);
    }
  });

  it('keep no color, shadow or button-fill token on :root, where it would outlive a theme switch', () => {
    expect([...structural].filter((name) => /color|shadow|button(?!BorderRadius)|focusRingColor|tab(Active|Hover|Focus)/i.test(name))).toEqual([]);
  });

  it('define every custom property a theme block reads, for every shipped id', () => {
    for (const id of THEME_IDS) {
      const { declared, referenced } = themeDeclarations(id);
      const missing = [...referenced].filter((name) => !declared.has(name) && !rootLevel.has(name) && !intensityInputs.has(name));
      expect(missing, id).toEqual([]);
    }
  });

  it('define every semantic token the product stylesheets read, for every shipped id', () => {
    const locallyDeclared = new Set<string>();
    const referenced = new Set<string>();
    for (const [path, css] of Object.entries(productCss)) {
      if (path.startsWith('design-system/themes/')) continue;
      for (const name of declaredProperties(stripComments(css))) locallyDeclared.add(name);
      for (const name of referencedProperties(stripComments(css))) referenced.add(name);
    }
    for (const id of THEME_IDS) {
      const { declared } = themeDeclarations(id);
      const missing = [...referenced].filter((name) => !declared.has(name) && !locallyDeclared.has(name) && !name.startsWith('--token-'));
      expect(missing, id).toEqual([]);
    }
  });

  it('import one theme entry directly after ui-common base.css, and no ui-common theme file', () => {
    const imports = Array.from(mainSource.matchAll(/^import\s+'([^']+\.css)';/gm), (match) => match[1]);
    expect(imports.slice(0, 2)).toEqual(['@lablup/ui-common/styles/base.css', './design-system/themes/index.css']);
    expect(imports.filter((path) => path.includes('/themes/'))).toEqual(['./design-system/themes/index.css']);
    expect(Array.from(themeEntry.matchAll(/@import\s+"([^"]+)"/g), (match) => match[1])).toEqual(['./contract.css', './mlxcel.css', './glass.css']);
  });

  it('write -webkit-backdrop-filter before backdrop-filter, or the minifier drops the standard property', () => {
    // Lightning CSS treats a prefixed declaration that follows the standard one
    // as the winner and emits only the prefixed form, which Chromium and
    // Firefox ignore. That silently disabled the glass blur in shipped builds.
    for (const [path, css] of Object.entries(productCss)) {
      for (const rule of topLevelRules(css)) {
        const standard = rule.body.search(/(^|[;\s])backdrop-filter\s*:/);
        const prefixed = rule.body.search(/-webkit-backdrop-filter\s*:/);
        if (standard !== -1 && prefixed !== -1) expect(prefixed, `${path} ${rule.selectors.join(', ')}`).toBeLessThan(standard);
      }
    }
  });
});
