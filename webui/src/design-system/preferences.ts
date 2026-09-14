import type { Locale } from '../i18n/catalog';

export type ThemePreference = 'system' | 'light' | 'dark';
export type MaterialPreference = 'glass' | 'tinted' | 'opaque';
export type ContrastPreference = 'system' | 'on' | 'off';

export type AppearancePreferences = {
  theme: ThemePreference;
  material: MaterialPreference;
  glassIntensity: number;
  reduceMotion: boolean;
  reduceTransparency: boolean;
  highContrast: ContrastPreference;
  locale: Locale;
};

export const DEFAULT_APPEARANCE: AppearancePreferences = {
  theme: 'system',
  material: 'glass',
  glassIntensity: 35,
  reduceMotion: false,
  reduceTransparency: false,
  highContrast: 'system',
  locale: 'en',
};

const STORAGE_KEY = 'mlxcel.webui.appearance';

function clampIntensity(value: unknown): number {
  const numeric = typeof value === 'number' && Number.isFinite(value) ? value : DEFAULT_APPEARANCE.glassIntensity;
  return Math.max(0, Math.min(100, Math.round(numeric)));
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function isTheme(value: unknown): value is ThemePreference {
  return value === 'system' || value === 'light' || value === 'dark';
}

function isMaterial(value: unknown): value is MaterialPreference {
  return value === 'glass' || value === 'tinted' || value === 'opaque';
}

function isContrast(value: unknown): value is ContrastPreference {
  return value === 'system' || value === 'on' || value === 'off';
}

function normalizeContrast(value: unknown): ContrastPreference {
  if (isContrast(value)) return value;
  if (value === true) return 'on';
  if (value === false) return 'off';
  return DEFAULT_APPEARANCE.highContrast;
}

function isLocale(value: unknown): value is Locale {
  return value === 'en' || value === 'ko';
}

export function loadAppearance(): AppearancePreferences {
  if (typeof window === 'undefined') return DEFAULT_APPEARANCE;
  let raw: string | null;
  try {
    raw = window.localStorage.getItem(STORAGE_KEY);
  } catch {
    return DEFAULT_APPEARANCE;
  }
  if (!raw) return DEFAULT_APPEARANCE;
  try {
    const parsed: unknown = JSON.parse(raw);
    if (!isRecord(parsed)) return DEFAULT_APPEARANCE;
    return {
      theme: isTheme(parsed.theme) ? parsed.theme : DEFAULT_APPEARANCE.theme,
      material: isMaterial(parsed.material) ? parsed.material : DEFAULT_APPEARANCE.material,
      glassIntensity: clampIntensity(parsed.glassIntensity),
      reduceMotion: parsed.reduceMotion === true,
      reduceTransparency: parsed.reduceTransparency === true,
      highContrast: normalizeContrast(parsed.highContrast),
      locale: isLocale(parsed.locale) ? parsed.locale : DEFAULT_APPEARANCE.locale,
    };
  } catch {
    return DEFAULT_APPEARANCE;
  }
}

export function saveAppearance(preferences: AppearancePreferences): void {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(preferences));
  } catch {
    // Browser appearance preferences remain functional in memory when storage is blocked or full.
  }
}

function supportsBackdropFilter(): boolean {
  if (typeof CSS === 'undefined' || typeof CSS.supports !== 'function') return false;
  return CSS.supports('backdrop-filter: blur(1px)') || CSS.supports('-webkit-backdrop-filter: blur(1px)');
}

export function applyAppearance(root: HTMLElement, preferences: AppearancePreferences): void {
  root.dataset.theme = preferences.theme;
  root.dataset.material = preferences.reduceTransparency ? 'opaque' : preferences.material;
  root.dataset.glassIntensity = String(clampIntensity(preferences.glassIntensity));
  root.dataset.reduceMotion = String(preferences.reduceMotion);
  root.dataset.reduceTransparency = String(preferences.reduceTransparency);
  root.dataset.highContrast = preferences.highContrast;
  root.dataset.backdropFilter = supportsBackdropFilter() ? 'supported' : 'unsupported';
  root.lang = preferences.locale;
}
