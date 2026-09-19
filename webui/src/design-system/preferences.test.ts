import { afterEach, describe, expect, it } from 'vitest';
import { APPEARANCE_STORAGE_KEY, DEFAULT_APPEARANCE, applyAppearance, loadAppearance, saveAppearance } from './preferences';

afterEach(() => {
  localStorage.clear();
  for (const attribute of ['data-theme', 'data-theme-family', 'data-color-scheme']) document.documentElement.removeAttribute(attribute);
});

describe('appearance preferences', () => {
  it('default to the mlxcel family following the host scheme', () => {
    expect(DEFAULT_APPEARANCE.themeFamily).toBe('mlxcel');
    expect(DEFAULT_APPEARANCE.colorScheme).toBe('system');
    expect(loadAppearance()).toEqual(DEFAULT_APPEARANCE);
  });

  it('migrate the pre-#1903 flat theme field into the color scheme', () => {
    localStorage.setItem(APPEARANCE_STORAGE_KEY, JSON.stringify({ theme: 'dark', material: 'tinted', glassIntensity: 60 }));
    expect(loadAppearance()).toMatchObject({ themeFamily: 'mlxcel', colorScheme: 'dark', material: 'tinted', glassIntensity: 60 });
    localStorage.setItem(APPEARANCE_STORAGE_KEY, JSON.stringify({ colorScheme: 'light', theme: 'dark' }));
    expect(loadAppearance().colorScheme).toBe('light');
    localStorage.setItem(APPEARANCE_STORAGE_KEY, JSON.stringify({ theme: 'glass-dark' }));
    expect(loadAppearance()).toMatchObject({ themeFamily: 'mlxcel', colorScheme: 'system' });
  });

  it('round-trip the family and scheme and drop unknown values', () => {
    saveAppearance({ ...DEFAULT_APPEARANCE, themeFamily: 'glass', colorScheme: 'dark' });
    expect(JSON.parse(localStorage.getItem(APPEARANCE_STORAGE_KEY) ?? '{}')).toMatchObject({ themeFamily: 'glass', colorScheme: 'dark' });
    expect(loadAppearance()).toMatchObject({ themeFamily: 'glass', colorScheme: 'dark' });
    localStorage.setItem(APPEARANCE_STORAGE_KEY, JSON.stringify({ themeFamily: 'orange', colorScheme: 'sepia' }));
    expect(loadAppearance()).toMatchObject({ themeFamily: 'mlxcel', colorScheme: 'system' });
  });

  it('write the applied id to data-theme and keep the raw preference apart', () => {
    const root = document.documentElement;
    applyAppearance(root, { ...DEFAULT_APPEARANCE, themeFamily: 'glass', colorScheme: 'system' }, 'dark');
    expect(root.dataset.theme).toBe('glass-dark');
    expect(root.dataset.themeFamily).toBe('glass');
    expect(root.dataset.colorScheme).toBe('system');
    applyAppearance(root, { ...DEFAULT_APPEARANCE, colorScheme: 'system' }, 'light');
    expect(root.dataset.theme).toBe('mlxcel-light');
    applyAppearance(root, { ...DEFAULT_APPEARANCE, colorScheme: 'light' }, 'dark');
    expect(root.dataset.theme).toBe('mlxcel-light');
    expect(root.dataset.colorScheme).toBe('light');
  });

  it('never write a bare mode into data-theme, whatever the host reports', () => {
    const root = document.documentElement;
    for (const colorScheme of ['system', 'light', 'dark'] as const) {
      for (const host of ['light', 'dark'] as const) {
        applyAppearance(root, { ...DEFAULT_APPEARANCE, colorScheme }, host);
        expect(['system', 'light', 'dark']).not.toContain(root.dataset.theme);
      }
    }
    // jsdom has no matchMedia, which must resolve to light rather than throw.
    applyAppearance(root, DEFAULT_APPEARANCE);
    expect(root.dataset.theme).toBe('mlxcel-light');
  });
});
