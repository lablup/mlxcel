import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  ASTRYX_THEME_NAMES,
  COLOR_SCHEMES,
  COLOR_SCHEME_PREFERENCES,
  THEME_FAMILIES,
  SYSTEM_DARK_QUERY,
  getSystemColorScheme,
  isColorScheme,
  isColorSchemePreference,
  isThemeFamily,
  onSystemColorSchemeChange,
  resolveThemeSelection,
} from './theme';
import { mlxcelTheme } from './themes/built/mlxcel.js';
import { mlxcelGlassTheme } from './themes/built/mlxcel-glass.js';

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

describe('theme identity', () => {
  it('applies a bare scheme to data-theme, as Astryx does, and never system', () => {
    expect(COLOR_SCHEMES).toEqual(['light', 'dark']);
    for (const value of ['system', '', 'mlxcel-dark', 'glass']) expect(isColorScheme(value)).toBe(false);
    for (const value of COLOR_SCHEMES) expect(isColorScheme(value)).toBe(true);
  });

  it('names one unique Astryx theme per family, matching the built modules', () => {
    expect(ASTRYX_THEME_NAMES).toEqual({ mlxcel: 'mlxcel', glass: 'mlxcel-glass' });
    expect(new Set(Object.values(ASTRYX_THEME_NAMES)).size).toBe(THEME_FAMILIES.length);
    expect(Object.values(ASTRYX_THEME_NAMES)).not.toContain('neutral');
    expect(mlxcelTheme.name).toBe(ASTRYX_THEME_NAMES.mlxcel);
    expect(mlxcelGlassTheme.name).toBe(ASTRYX_THEME_NAMES.glass);
    // Pre-built: <Theme> injects no <style> element for them (the CSP allows none).
    expect(mlxcelTheme.__built).toBe(true);
    expect(mlxcelGlassTheme.__built).toBe(true);
  });

  it('validates the stored preference separately from the applied values', () => {
    expect(THEME_FAMILIES).toEqual(['mlxcel', 'glass']);
    expect(COLOR_SCHEME_PREFERENCES).toEqual(['system', 'light', 'dark']);
    expect(isThemeFamily('glass')).toBe(true);
    expect(isThemeFamily('glass-dark')).toBe(false);
    expect(isThemeFamily(undefined)).toBe(false);
    expect(isColorSchemePreference('system')).toBe(true);
    expect(isColorSchemePreference('mlxcel-dark')).toBe(false);
  });

  it('resolves a system preference against the host scheme and keeps the raw preference', () => {
    expect(resolveThemeSelection('glass', 'system', 'dark')).toEqual({ family: 'glass', astryxTheme: 'mlxcel-glass', scheme: 'dark', preference: 'system' });
    expect(resolveThemeSelection('mlxcel', 'system', 'light')).toEqual({ family: 'mlxcel', astryxTheme: 'mlxcel', scheme: 'light', preference: 'system' });
    expect(resolveThemeSelection('mlxcel', 'dark', 'light').scheme).toBe('dark');
    expect(resolveThemeSelection('glass', 'light', 'dark').scheme).toBe('light');
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
