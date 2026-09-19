import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import indexHtml from '../../index.html?raw';
import bootstrapSource from '../../public/theme-bootstrap.js?raw';
import { APPEARANCE_STORAGE_KEY, applyAppearance, loadAppearance } from './preferences';
import { COLOR_SCHEME_PREFERENCES, THEME_FAMILIES, isThemeId } from './theme';

const THEME_ATTRIBUTES = ['data-theme', 'data-theme-family', 'data-color-scheme'] as const;

type HostScheme = 'light' | 'dark' | 'missing' | 'throws';

function clearThemeAttributes(): void {
  for (const attribute of THEME_ATTRIBUTES) document.documentElement.removeAttribute(attribute);
}

function readThemeAttributes(): Record<string, string | null> {
  return Object.fromEntries(THEME_ATTRIBUTES.map((attribute) => [attribute, document.documentElement.getAttribute(attribute)]));
}

function setHostScheme(host: HostScheme): void {
  if (host === 'missing') vi.stubGlobal('matchMedia', undefined);
  else if (host === 'throws') vi.stubGlobal('matchMedia', () => { throw new Error('blocked'); });
  else vi.stubGlobal('matchMedia', () => ({ matches: host === 'dark', addEventListener: () => undefined, removeEventListener: () => undefined }));
}

function runBootstrap(): Record<string, string | null> {
  clearThemeAttributes();
  new Function(bootstrapSource)();
  return readThemeAttributes();
}

function runApp(): Record<string, string | null> {
  clearThemeAttributes();
  applyAppearance(document.documentElement, loadAppearance());
  return readThemeAttributes();
}

const STORED: Array<[string, string | null]> = [
  ['nothing stored', null],
  ['corrupt JSON', '{'],
  ['an array', '[]'],
  ['an empty object', '{}'],
  ['legacy flat theme: dark', JSON.stringify({ theme: 'dark', material: 'glass' })],
  ['legacy flat theme: light', JSON.stringify({ theme: 'light' })],
  ['legacy flat theme: system', JSON.stringify({ theme: 'system' })],
  ['glass, dark', JSON.stringify({ themeFamily: 'glass', colorScheme: 'dark' })],
  ['glass, system', JSON.stringify({ themeFamily: 'glass', colorScheme: 'system' })],
  ['glass family only', JSON.stringify({ themeFamily: 'glass' })],
  ['mlxcel, light', JSON.stringify({ themeFamily: 'mlxcel', colorScheme: 'light' })],
  ['colorScheme wins over the legacy field', JSON.stringify({ colorScheme: 'light', theme: 'dark' })],
  ['an invalid colorScheme does not fall back to the legacy field', JSON.stringify({ colorScheme: 'sepia', theme: 'dark' })],
  ['unknown family and scheme', JSON.stringify({ themeFamily: 'orange', colorScheme: 'glass-dark' })],
  ['a full theme id in the legacy field', JSON.stringify({ theme: 'glass-dark' })],
  // Every family theme.ts ships, so a family added there but not to the bootstrap fails here.
  ...THEME_FAMILIES.flatMap((themeFamily) => COLOR_SCHEME_PREFERENCES.map((colorScheme): [string, string] => [`${themeFamily}, ${colorScheme}`, JSON.stringify({ themeFamily, colorScheme })])),
];

beforeEach(() => {
  localStorage.clear();
});

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  localStorage.clear();
  clearThemeAttributes();
});

describe('pre-mount theme bootstrap (public/theme-bootstrap.js)', () => {
  for (const host of ['light', 'dark', 'missing', 'throws'] as const) {
    it.each(STORED)(`matches applyAppearance for %s under a ${host} host`, (_label, stored) => {
      if (stored !== null) localStorage.setItem(APPEARANCE_STORAGE_KEY, stored);
      setHostScheme(host);
      const early = runBootstrap();
      expect(isThemeId(early['data-theme'])).toBe(true);
      expect(early).toEqual(runApp());
    });
  }

  it('keeps the defaults when storage access throws', () => {
    setHostScheme('dark');
    vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => { throw new DOMException('blocked', 'SecurityError'); });
    const early = runBootstrap();
    expect(early).toEqual({ 'data-theme': 'mlxcel-dark', 'data-theme-family': 'mlxcel', 'data-color-scheme': 'system' });
    expect(early).toEqual(runApp());
  });

  it('never writes system, light or dark alone into data-theme', () => {
    for (const theme of ['system', 'light', 'dark']) {
      localStorage.setItem(APPEARANCE_STORAGE_KEY, JSON.stringify({ theme }));
      setHostScheme('light');
      expect(isThemeId(runBootstrap()['data-theme'])).toBe(true);
    }
  });
});

describe('index.html', () => {
  const head = indexHtml.slice(0, indexHtml.indexOf('</head>'));

  it('loads the bootstrap as a classic, render-blocking script in <head>', () => {
    const tag = head.match(/<script\b[^>]*\bsrc="\.\/theme-bootstrap\.js"[^>]*><\/script>/);
    expect(tag).not.toBeNull();
    // Any type attribute (module included), async or defer would stop it blocking the first paint.
    expect(tag?.[0]).not.toMatch(/\btype=|\basync\b|\bdefer\b/);
  });

  it('carries no inline script, which the served CSP (script-src self) would block', () => {
    for (const match of indexHtml.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script>/g)) {
      expect(match[1]).toMatch(/\bsrc="/);
      expect(match[2].trim()).toBe('');
    }
  });

  it('runs the bootstrap before the app module', () => {
    const bootstrap = indexHtml.indexOf('./theme-bootstrap.js');
    const app = indexHtml.search(/<script\b[^>]*\bsrc="\/src\/main\.tsx"/);
    expect(bootstrap).toBeGreaterThan(-1);
    expect(app).toBeGreaterThan(-1);
    expect(bootstrap).toBeLessThan(app);
  });
});
