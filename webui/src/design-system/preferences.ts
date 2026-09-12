import type { Locale } from '../i18n/catalog';

export type ThemePreference = 'system' | 'light' | 'dark';
export type MaterialPreference = 'glass' | 'tinted' | 'opaque';

export type AppearancePreferences = {
  theme: ThemePreference;
  material: MaterialPreference;
  glassIntensity: number;
  reduceMotion: boolean;
  reduceTransparency: boolean;
  highContrast: boolean;
  locale: Locale;
};

export const DEFAULT_APPEARANCE: AppearancePreferences = {
  theme: 'system',
  material: 'glass',
  glassIntensity: 35,
  reduceMotion: false,
  reduceTransparency: false,
  highContrast: false,
  locale: 'en',
};

const STORAGE_KEY = 'mlxcel.webui.appearance';

function clampIntensity(value: unknown): number {
  const numeric = typeof value === 'number' && Number.isFinite(value) ? value : DEFAULT_APPEARANCE.glassIntensity;
  return Math.max(0, Math.min(100, Math.round(numeric)));
}

function isTheme(value: unknown): value is ThemePreference {
  return value === 'system' || value === 'light' || value === 'dark';
}

function isMaterial(value: unknown): value is MaterialPreference {
  return value === 'glass' || value === 'tinted' || value === 'opaque';
}

function isLocale(value: unknown): value is Locale {
  return value === 'en' || value === 'ko';
}

export function loadAppearance(): AppearancePreferences {
  if (typeof window === 'undefined') return DEFAULT_APPEARANCE;
  const raw = window.localStorage.getItem(STORAGE_KEY);
  if (!raw) return DEFAULT_APPEARANCE;
  try {
    const parsed: Record<string, unknown> = JSON.parse(raw) as Record<string, unknown>;
    return {
      theme: isTheme(parsed.theme) ? parsed.theme : DEFAULT_APPEARANCE.theme,
      material: isMaterial(parsed.material) ? parsed.material : DEFAULT_APPEARANCE.material,
      glassIntensity: clampIntensity(parsed.glassIntensity),
      reduceMotion: parsed.reduceMotion === true,
      reduceTransparency: parsed.reduceTransparency === true,
      highContrast: parsed.highContrast === true,
      locale: isLocale(parsed.locale) ? parsed.locale : DEFAULT_APPEARANCE.locale,
    };
  } catch {
    return DEFAULT_APPEARANCE;
  }
}

export function saveAppearance(preferences: AppearancePreferences): void {
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(preferences));
}

export function applyAppearance(root: HTMLElement, preferences: AppearancePreferences): void {
  root.dataset.theme = preferences.theme;
  root.dataset.material = preferences.reduceTransparency ? 'opaque' : preferences.material;
  root.dataset.glassIntensity = String(clampIntensity(preferences.glassIntensity));
  root.dataset.reduceMotion = String(preferences.reduceMotion);
  root.dataset.reduceTransparency = String(preferences.reduceTransparency);
  root.dataset.highContrast = String(preferences.highContrast);
  root.lang = preferences.locale;
}
