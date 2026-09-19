import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  COLOR_SCHEME_PREFERENCES,
  THEME_FAMILIES,
  SYSTEM_DARK_QUERY,
  THEME_IDS,
  getSystemColorScheme,
  isColorSchemePreference,
  isThemeFamily,
  isThemeId,
  onSystemColorSchemeChange,
  resolveThemeSelection,
} from './theme';

type Listener = (event: MediaQueryListEvent) => void;

function installMatchMedia(dark: boolean): { flip: (next: boolean) => void; listeners: Set<Listener> } {
  const listeners = new Set<Listener>();
  const query = {
    matches: dark,
    media: SYSTEM_DARK_QUERY,
    addEventListener: (_type: string, listener: Listener) => listeners.add(listener),
    removeEventListener: (_type: string, listener: Listener) => listeners.delete(listener),
  };
  vi.stubGlobal('matchMedia', vi.fn(() => query));
  return {
    listeners,
    flip: (next: boolean) => {
      query.matches = next;
      for (const listener of listeners) listener({ matches: next } as MediaQueryListEvent);
    },
  };
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('theme ids', () => {
  it('are always <family>-<scheme> and never a bare mode', () => {
    expect(THEME_IDS).toEqual(['mlxcel-light', 'mlxcel-dark', 'glass-light', 'glass-dark']);
    for (const id of ['light', 'dark', 'system', '', 'orange-light', 'glass']) expect(isThemeId(id)).toBe(false);
    for (const id of THEME_IDS) expect(isThemeId(id)).toBe(true);
  });

  it('validates the stored preference separately from the applied id', () => {
    expect(THEME_FAMILIES).toEqual(['mlxcel', 'glass']);
    expect(COLOR_SCHEME_PREFERENCES).toEqual(['system', 'light', 'dark']);
    expect(isThemeFamily('glass')).toBe(true);
    expect(isThemeFamily('glass-dark')).toBe(false);
    expect(isThemeFamily(undefined)).toBe(false);
    expect(isColorSchemePreference('system')).toBe(true);
    expect(isColorSchemePreference('mlxcel-dark')).toBe(false);
  });

  it('resolves a system preference against the host scheme and keeps the raw preference', () => {
    expect(resolveThemeSelection('glass', 'system', 'dark')).toEqual({ id: 'glass-dark', family: 'glass', scheme: 'dark', preference: 'system' });
    expect(resolveThemeSelection('mlxcel', 'system', 'light')).toEqual({ id: 'mlxcel-light', family: 'mlxcel', scheme: 'light', preference: 'system' });
    expect(resolveThemeSelection('mlxcel', 'dark', 'light').id).toBe('mlxcel-dark');
    expect(resolveThemeSelection('glass', 'light', 'dark').id).toBe('glass-light');
  });
});

describe('host color scheme', () => {
  it('reads matchMedia and treats a missing or throwing implementation as light', () => {
    installMatchMedia(true);
    expect(getSystemColorScheme()).toBe('dark');
    vi.unstubAllGlobals();
    vi.stubGlobal('matchMedia', undefined);
    expect(getSystemColorScheme()).toBe('light');
    vi.stubGlobal('matchMedia', () => { throw new Error('blocked'); });
    expect(getSystemColorScheme()).toBe('light');
  });

  it('reports host scheme flips until unsubscribed', () => {
    const media = installMatchMedia(false);
    const seen: string[] = [];
    const unsubscribe = onSystemColorSchemeChange((scheme) => seen.push(scheme));
    media.flip(true);
    media.flip(false);
    unsubscribe();
    media.flip(true);
    expect(seen).toEqual(['dark', 'light']);
    expect(media.listeners.size).toBe(0);
  });

  it('is a no-op without matchMedia', () => {
    vi.stubGlobal('matchMedia', undefined);
    const unsubscribe = onSystemColorSchemeChange(() => undefined);
    expect(() => unsubscribe()).not.toThrow();
  });
});
